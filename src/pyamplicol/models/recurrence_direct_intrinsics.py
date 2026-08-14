# SPDX-License-Identifier: 0BSD
# ruff: noqa: RUF001
"""Exact certification of model-generic Direct-Arena recurrence intrinsics."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from .recurrence_template import ExactComplexRationalV1


class _LazyCompilerSymbolica:
    """Keep planning metadata independent of compiler-module imports."""

    def __getattr__(self, name: str) -> object:
        module = importlib.import_module(
            ".compiler_symbolica",
            package=__package__,
        )
        return getattr(module, name)


_sym = _LazyCompilerSymbolica()

RECURRENCE_INTRINSIC_SCALE_KIND = "intrinsic-scale-v1"
RECURRENCE_MASSIVE_DIRAC_FINALIZER_KIND = "massive-dirac-propagator-v1"
WEYL_PROPAGATOR_POSITIVE_TEMPLATE = "rusticol.recurrence-intrinsic.weyl-propagator-a.v1"
WEYL_PROPAGATOR_NEGATIVE_TEMPLATE = "rusticol.recurrence-intrinsic.weyl-propagator-b.v1"
FEYNMAN_VECTOR_PROPAGATOR_TEMPLATE = (
    "rusticol.recurrence-intrinsic.vector-propagator-feynman.v1"
)
DIRAC_VECTOR_PARTICLE_TEMPLATE = (
    "rusticol.recurrence-intrinsic.dirac-vector-to-dirac-particle.v1"
)
DIRAC_VECTOR_ANTIPARTICLE_TEMPLATE = (
    "rusticol.recurrence-intrinsic.dirac-vector-to-dirac-antiparticle.v1"
)
DIRAC_SCALAR_TO_DIRAC_TEMPLATE = (
    "rusticol.recurrence-intrinsic.dirac-scalar-to-dirac.v1"
)
WEYL_PAIR_TO_VECTOR_A_TEMPLATE = (
    "rusticol.recurrence-intrinsic.weyl-pair-to-vector-a.v1"
)
WEYL_PAIR_TO_VECTOR_B_TEMPLATE = (
    "rusticol.recurrence-intrinsic.weyl-pair-to-vector-b.v1"
)
MASSIVE_DIRAC_PARTICLE_TEMPLATE = (
    "rusticol.recurrence-intrinsic.massive-dirac-propagator-particle.v1"
)
MASSIVE_DIRAC_ANTIPARTICLE_TEMPLATE = (
    "rusticol.recurrence-intrinsic.massive-dirac-propagator-antiparticle.v1"
)

DiracOrientation: TypeAlias = Literal["particle", "antiparticle"]


def _f64_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


MASSIVE_DIRAC_RUNTIME_SCALE_BITS = (_f64_bits(0.0), _f64_bits(1.0))


@dataclass(frozen=True, slots=True)
class CertifiedRecurrenceIntrinsic:
    """One exact algebra witness plus its model-owned scalar scale."""

    runtime_template: str
    contract_digest: str
    constant_scale: complex
    model_parameter_index: int | None
    parent_permutation: tuple[int, int] = (0, 1)

    def scale_projection(self) -> dict[str, object]:
        real = 0.0 if self.constant_scale.real == 0.0 else self.constant_scale.real
        imag = 0.0 if self.constant_scale.imag == 0.0 else self.constant_scale.imag
        return {
            "constant_imag_bits": _f64_bits(imag),
            "constant_real_bits": _f64_bits(real),
            "kind": RECURRENCE_INTRINSIC_SCALE_KIND,
            "parameter_index": self.model_parameter_index,
        }


@dataclass(frozen=True, slots=True)
class CertifiedRecurrenceFinalizationIntrinsic:
    """One exact runtime-owned finalization primitive."""

    runtime_template: str
    contract_digest: str
    constant_scale: complex
    orientation: DiracOrientation | None = None
    mass_parameter_index: int | None = None
    width_parameter_index: int | None = None

    def __post_init__(self) -> None:
        parameterized = (
            self.orientation,
            self.mass_parameter_index,
            self.width_parameter_index,
        )
        if parameterized == (None, None, None):
            return
        if (
            self.orientation not in {"particle", "antiparticle"}
            or type(self.mass_parameter_index) is not int
            or self.mass_parameter_index < 0
            or type(self.width_parameter_index) is not int
            or self.width_parameter_index < 0
            or self.mass_parameter_index == self.width_parameter_index
        ):
            raise ValueError(
                "massive Dirac finalization requires an orientation and distinct "
                "nonnegative mass/width parameter indices"
            )

    def scale_projection(self) -> dict[str, object]:
        real = 0.0 if self.constant_scale.real == 0.0 else self.constant_scale.real
        imag = 0.0 if self.constant_scale.imag == 0.0 else self.constant_scale.imag
        if self.orientation is not None:
            assert self.mass_parameter_index is not None
            assert self.width_parameter_index is not None
            return {
                "constant_imag_bits": _f64_bits(imag),
                "constant_real_bits": _f64_bits(real),
                "kind": RECURRENCE_MASSIVE_DIRAC_FINALIZER_KIND,
                "mass_parameter_index": self.mass_parameter_index,
                "orientation": self.orientation,
                "width_parameter_index": self.width_parameter_index,
            }
        return {
            "constant_imag_bits": _f64_bits(imag),
            "constant_real_bits": _f64_bits(real),
            "kind": RECURRENCE_INTRINSIC_SCALE_KIND,
            "parameter_index": None,
        }


@dataclass(frozen=True, slots=True)
class _IntrinsicWitness:
    runtime_template: str
    expressions: tuple[str, ...]
    anchor_monomial: str
    inverse_anchor_coefficient: str
    parent_component_counts: tuple[int, int]
    destination_component_count: int
    contract_digest: str = ""

    def __post_init__(self) -> None:
        payload = {
            "anchor_monomial": self.anchor_monomial,
            "destination_component_count": self.destination_component_count,
            "expressions": list(self.expressions),
            "inverse_anchor_coefficient": self.inverse_anchor_coefficient,
            "parent_component_counts": list(self.parent_component_counts),
            "runtime_template": self.runtime_template,
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
        object.__setattr__(self, "contract_digest", digest)


_WITNESSES = (
    _IntrinsicWitness(
        runtime_template=("rusticol.recurrence-intrinsic.scalar-product.v1"),
        expressions=("l0*r0",),
        anchor_monomial="l0*r0",
        inverse_anchor_coefficient="1",
        parent_component_counts=(1, 1),
        destination_component_count=1,
    ),
    _IntrinsicWitness(
        runtime_template=(
            "rusticol.recurrence-intrinsic.color-ordered-three-vector.v1"
        ),
        expressions=(
            "(l0*r0-l1*r1-l2*r2-l3*r3)*(p0-q0)"
            "+2*((l0*q0-l1*q1-l2*q2-l3*q3)*r0"
            "-(r0*p0-r1*p1-r2*p2-r3*p3)*l0)",
            "(l0*r0-l1*r1-l2*r2-l3*r3)*(p1-q1)"
            "+2*((l0*q0-l1*q1-l2*q2-l3*q3)*r1"
            "-(r0*p0-r1*p1-r2*p2-r3*p3)*l1)",
            "(l0*r0-l1*r1-l2*r2-l3*r3)*(p2-q2)"
            "+2*((l0*q0-l1*q1-l2*q2-l3*q3)*r2"
            "-(r0*p0-r1*p1-r2*p2-r3*p3)*l2)",
            "(l0*r0-l1*r1-l2*r2-l3*r3)*(p3-q3)"
            "+2*((l0*q0-l1*q1-l2*q2-l3*q3)*r3"
            "-(r0*p0-r1*p1-r2*p2-r3*p3)*l3)",
        ),
        anchor_monomial="l0*p0*r0",
        inverse_anchor_coefficient="-1",
        parent_component_counts=(4, 4),
        destination_component_count=4,
    ),
    _IntrinsicWitness(
        runtime_template=("rusticol.recurrence-intrinsic.weyl-vector-to-weyl-a.v1"),
        expressions=(
            "-1𝑖*l0*r3-1𝑖*l1*r1+l1*r2+1𝑖*l0*r0",
            "-l0*r2-1𝑖*l0*r1+1𝑖*l1*r0+1𝑖*l1*r3",
        ),
        anchor_monomial="l0*r3",
        inverse_anchor_coefficient="1𝑖",
        parent_component_counts=(2, 4),
        destination_component_count=2,
    ),
    _IntrinsicWitness(
        runtime_template=("rusticol.recurrence-intrinsic.weyl-vector-to-weyl-b.v1"),
        expressions=(
            "-l1*r2+1𝑖*l0*r0+1𝑖*l0*r3+1𝑖*l1*r1",
            "-1𝑖*l1*r3+l0*r2+1𝑖*l0*r1+1𝑖*l1*r0",
        ),
        anchor_monomial="l1*r2",
        inverse_anchor_coefficient="-1",
        parent_component_counts=(2, 4),
        destination_component_count=2,
    ),
    _IntrinsicWitness(
        runtime_template=WEYL_PAIR_TO_VECTOR_A_TEMPLATE,
        expressions=(
            "l0*r0+l1*r1",
            "-l1*r0-l0*r1",
            "1\U0001d456*(-l1*r0+l0*r1)",
            "-l0*r0+l1*r1",
        ),
        anchor_monomial="l0*r0",
        inverse_anchor_coefficient="1",
        parent_component_counts=(2, 2),
        destination_component_count=4,
    ),
    _IntrinsicWitness(
        runtime_template=WEYL_PAIR_TO_VECTOR_B_TEMPLATE,
        expressions=(
            "l0*r0+l1*r1",
            "l0*r1+l1*r0",
            "1\U0001d456*(-l0*r1+l1*r0)",
            "l0*r0-l1*r1",
        ),
        anchor_monomial="l0*r0",
        inverse_anchor_coefficient="1",
        parent_component_counts=(2, 2),
        destination_component_count=4,
    ),
    _IntrinsicWitness(
        runtime_template=(
            "rusticol.recurrence-intrinsic.antisymmetric-tensor-vector.v1"
        ),
        expressions=(
            "l0*r1+l1*r2+l2*r3",
            "l0*r0+l3*r2+l4*r3",
            "-l3*r1+l1*r0+l5*r3",
            "-l4*r1-l5*r2+l2*r0",
        ),
        anchor_monomial="l0*r1",
        inverse_anchor_coefficient="1",
        parent_component_counts=(6, 4),
        destination_component_count=4,
    ),
    _IntrinsicWitness(
        runtime_template=("rusticol.recurrence-intrinsic.vector-wedge-vector.v1"),
        expressions=(
            "-l1*r0+l0*r1",
            "-l2*r0+l0*r2",
            "-l3*r0+l0*r3",
            "-l2*r1+l1*r2",
            "-l3*r1+l1*r3",
            "-l3*r2+l2*r3",
        ),
        anchor_monomial="l1*r0",
        inverse_anchor_coefficient="-1",
        parent_component_counts=(4, 4),
        destination_component_count=6,
    ),
    _IntrinsicWitness(
        runtime_template=DIRAC_VECTOR_PARTICLE_TEMPLATE,
        expressions=(
            "(r0+r3)*l2+(r1+1\U0001d456*r2)*l3",
            "(r0-r3)*l3+(r1-1\U0001d456*r2)*l2",
            "(r0-r3)*l0-(r1+1\U0001d456*r2)*l1",
            "(r0+r3)*l1-(r1-1\U0001d456*r2)*l0",
        ),
        anchor_monomial="l2*r0",
        inverse_anchor_coefficient="1",
        parent_component_counts=(4, 4),
        destination_component_count=4,
    ),
    _IntrinsicWitness(
        runtime_template=DIRAC_VECTOR_ANTIPARTICLE_TEMPLATE,
        expressions=(
            "(-r0+r3)*l2+(r1-1\U0001d456*r2)*l3",
            "(-r0-r3)*l3+(r1+1\U0001d456*r2)*l2",
            "(-r0-r3)*l0+(-r1+1\U0001d456*r2)*l1",
            "(-r0+r3)*l1+(-r1-1\U0001d456*r2)*l0",
        ),
        anchor_monomial="l2*r0",
        inverse_anchor_coefficient="-1",
        parent_component_counts=(4, 4),
        destination_component_count=4,
    ),
    _IntrinsicWitness(
        runtime_template=DIRAC_SCALAR_TO_DIRAC_TEMPLATE,
        expressions=("l0*r0", "l1*r0", "l2*r0", "l3*r0"),
        anchor_monomial="l0*r0",
        inverse_anchor_coefficient="1",
        parent_component_counts=(4, 1),
        destination_component_count=4,
    ),
)

RECURRENCE_INTRINSIC_RUNTIME_TEMPLATES = frozenset(
    witness.runtime_template for witness in _WITNESSES
)
RECURRENCE_INTRINSIC_CONTRACT_DIGESTS = {
    witness.runtime_template: witness.contract_digest for witness in _WITNESSES
}


@dataclass(frozen=True, slots=True)
class _FinalizationWitness:
    runtime_template: str
    expressions: tuple[str, ...]
    anchor_monomial: str
    inverse_anchor_coefficient: str
    component_count: int
    runtime_owned_scale: complex
    contract_digest: str = ""

    def __post_init__(self) -> None:
        payload = {
            "anchor_monomial": self.anchor_monomial,
            "component_count": self.component_count,
            "expressions": list(self.expressions),
            "inverse_anchor_coefficient": self.inverse_anchor_coefficient,
            "runtime_owned_scale_bits": [
                _f64_bits(self.runtime_owned_scale.real),
                _f64_bits(self.runtime_owned_scale.imag),
            ],
            "runtime_template": self.runtime_template,
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
        object.__setattr__(self, "contract_digest", digest)


_FINALIZATION_WITNESSES = (
    _FinalizationWitness(
        runtime_template=WEYL_PROPAGATOR_POSITIVE_TEMPLATE,
        expressions=(
            "((-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1))*(-1𝑖*l0*p3-1𝑖*l1*p1+l1*p2+1𝑖*l0*p0)",
            "((-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1))*(-l0*p2-1𝑖*l0*p1+1𝑖*l1*p0+1𝑖*l1*p3)",
        ),
        anchor_monomial="((-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1))*l0*p3",
        inverse_anchor_coefficient="1𝑖",
        component_count=2,
        runtime_owned_scale=1.0 + 0.0j,
    ),
    _FinalizationWitness(
        runtime_template=WEYL_PROPAGATOR_NEGATIVE_TEMPLATE,
        expressions=(
            "((-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1))*(-l1*p2+1𝑖*l0*p0+1𝑖*l0*p3+1𝑖*l1*p1)",
            "((-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1))*(-1𝑖*l1*p3+l0*p2+1𝑖*l0*p1+1𝑖*l1*p0)",
        ),
        anchor_monomial="((-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1))*l1*p2",
        inverse_anchor_coefficient="-1",
        component_count=2,
        runtime_owned_scale=1.0 + 0.0j,
    ),
    _FinalizationWitness(
        runtime_template=FEYNMAN_VECTOR_PROPAGATOR_TEMPLATE,
        expressions=(
            "((-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1))*l0",
            "((-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1))*l1",
            "((-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1))*l2",
            "((-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1))*l3",
        ),
        anchor_monomial="((-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1))*l0",
        inverse_anchor_coefficient="1",
        component_count=4,
        runtime_owned_scale=0.0 - 1.0j,
    ),
)


@dataclass(frozen=True, slots=True)
class _MassiveDiracFinalizationWitness:
    runtime_template: str
    orientation: DiracOrientation
    expressions: tuple[str, str, str, str]
    runtime_owned_scale: complex = 0.0 + 1.0j
    contract_digest: str = ""

    def __post_init__(self) -> None:
        payload = {
            "component_count": 4,
            "expressions": list(self.expressions),
            "orientation": self.orientation,
            "parameter_roles": ["mass", "width"],
            "runtime_owned_scale_bits": [
                _f64_bits(self.runtime_owned_scale.real),
                _f64_bits(self.runtime_owned_scale.imag),
            ],
            "runtime_template": self.runtime_template,
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
        object.__setattr__(self, "contract_digest", digest)


_MASSIVE_DIRAC_FINALIZATION_WITNESSES = (
    _MassiveDiracFinalizationWitness(
        runtime_template=MASSIVE_DIRAC_PARTICLE_TEMPLATE,
        orientation="particle",
        expressions=(
            "(p0^2-p1^2-p2^2-p3^2-m^2+1.00000000000000\U0001d456*m*w)^(-1)"
            "*((p0+p3)*l2+(p1+1\U0001d456*p2)*l3+m*l0)",
            "(p0^2-p1^2-p2^2-p3^2-m^2+1.00000000000000\U0001d456*m*w)^(-1)"
            "*((p0-p3)*l3+(p1-1\U0001d456*p2)*l2+m*l1)",
            "(p0^2-p1^2-p2^2-p3^2-m^2+1.00000000000000\U0001d456*m*w)^(-1)"
            "*((p0-p3)*l0-(p1+1\U0001d456*p2)*l1+m*l2)",
            "(p0^2-p1^2-p2^2-p3^2-m^2+1.00000000000000\U0001d456*m*w)^(-1)"
            "*((p0+p3)*l1-(p1-1\U0001d456*p2)*l0+m*l3)",
        ),
    ),
    _MassiveDiracFinalizationWitness(
        runtime_template=MASSIVE_DIRAC_ANTIPARTICLE_TEMPLATE,
        orientation="antiparticle",
        expressions=(
            "(p0^2-p1^2-p2^2-p3^2-m^2+1.00000000000000\U0001d456*m*w)^(-1)"
            "*((-p0+p3)*l2+(p1-1\U0001d456*p2)*l3+m*l0)",
            "(p0^2-p1^2-p2^2-p3^2-m^2+1.00000000000000\U0001d456*m*w)^(-1)"
            "*((-p0-p3)*l3+(p1+1\U0001d456*p2)*l2+m*l1)",
            "(p0^2-p1^2-p2^2-p3^2-m^2+1.00000000000000\U0001d456*m*w)^(-1)"
            "*((-p0-p3)*l0+(-p1+1\U0001d456*p2)*l1+m*l2)",
            "(p0^2-p1^2-p2^2-p3^2-m^2+1.00000000000000\U0001d456*m*w)^(-1)"
            "*((-p0+p3)*l1+(-p1-1\U0001d456*p2)*l0+m*l3)",
        ),
    ),
)

RECURRENCE_FINALIZATION_INTRINSIC_CONTRACT_DIGESTS = {
    witness.runtime_template: witness.contract_digest
    for witness in (
        *_FINALIZATION_WITNESSES,
        *_MASSIVE_DIRAC_FINALIZATION_WITNESSES,
    )
}


def certify_recurrence_contribution_intrinsic(
    *,
    exact_expressions: Sequence[str],
    input_contracts: Sequence[str],
    parent_component_counts: tuple[int, ...],
    destination_component_count: int,
    binding_coupling: ExactComplexRationalV1 | None,
    factored_output_parameter_index: int | None = None,
    allow_nontrivial_parent_permutation: bool = False,
) -> CertifiedRecurrenceIntrinsic | None:
    """Prove that one prepared kernel is exactly one known arena primitive.

    Model-owned namespaces and parameter numbers do not participate in the
    witness identity. Coupling inputs are first replaced by the exact
    recurrence binding. A remaining scalar may be either a finite numerical
    constant or that constant times one model parameter.

    A coupling factored out of the prepared kernel remains a live parameter.
    Its prepared slot is supplied only after the semantic transition catalog
    has authenticated the output-factor owner; it cannot coexist with another
    scalar parameter retained by the exact kernel algebra.

    A prepared model may expose the same commutative vertex with its two
    parents in the opposite order. Such a certificate records the exact
    canonical-to-prepared permutation. Non-trivial permutations are rejected
    by default because the current Direct-Arena payload has no permutation
    field; a caller must opt in only when it will consume the permutation.
    """

    if (
        len(parent_component_counts) != 2
        or not exact_expressions
        or len(exact_expressions) != destination_component_count
        or (
            factored_output_parameter_index is not None
            and (
                type(factored_output_parameter_index) is not int
                or factored_output_parameter_index < 0
            )
        )
    ):
        return None
    try:
        candidates = _normalization_candidates(
            exact_expressions,
            input_contracts,
            parent_component_counts=parent_component_counts,
            binding_coupling=binding_coupling,
            allow_nontrivial_parent_permutation=(allow_nontrivial_parent_permutation),
        )
    except (TypeError, ValueError):
        return None

    _sym._ensure_symbolica()
    for (
        normalized,
        parameter_symbols,
        parent_permutation,
        normalized_shape,
    ) in candidates:
        if factored_output_parameter_index is not None and parameter_symbols:
            continue
        for witness in _WITNESSES:
            if (
                witness.parent_component_counts != normalized_shape
                or witness.destination_component_count != destination_component_count
            ):
                continue
            references = tuple(_sym.E(value).expand() for value in witness.expressions)
            coefficient = normalized[0].coefficient(_sym.E(witness.anchor_monomial))
            scale = (coefficient * _sym.E(witness.inverse_anchor_coefficient)).expand()
            if any(
                not _symbolically_equal(candidate, scale * reference)
                for candidate, reference in zip(normalized, references, strict=True)
            ):
                continue
            scalar = _extract_scalar_scale(scale, parameter_symbols)
            if scalar is None:
                continue
            constant, parameter_index = scalar
            if (
                parameter_index is not None
                and factored_output_parameter_index is not None
            ):
                continue
            if factored_output_parameter_index is not None:
                parameter_index = factored_output_parameter_index
            return CertifiedRecurrenceIntrinsic(
                runtime_template=witness.runtime_template,
                contract_digest=witness.contract_digest,
                constant_scale=constant,
                model_parameter_index=parameter_index,
                parent_permutation=parent_permutation,
            )
    return None


def certify_recurrence_finalization_intrinsic(
    *,
    exact_expressions: Sequence[str],
    input_contracts: Sequence[str],
    component_count: int,
) -> CertifiedRecurrenceFinalizationIntrinsic | None:
    """Prove that one prepared propagator/finalizer is one known arena primitive."""

    if not exact_expressions or len(exact_expressions) != component_count:
        return None
    try:
        normalized, parameter_symbols = _normalized_expressions(
            exact_expressions,
            input_contracts,
            binding_coupling=None,
        )
    except (TypeError, ValueError):
        return None
    if parameter_symbols:
        try:
            parameter_indices = _massive_dirac_parameter_indices(input_contracts)
        except (TypeError, ValueError):
            return None
        if set(parameter_symbols.values()) != set(parameter_indices):
            return None
        return _certify_massive_dirac_finalization(
            normalized,
            parameter_indices=parameter_indices,
            component_count=component_count,
        )

    _sym._ensure_symbolica()
    for witness in _FINALIZATION_WITNESSES:
        if witness.component_count != component_count:
            continue
        references = tuple(_sym.E(value).expand() for value in witness.expressions)
        coefficient = normalized[0].coefficient(_sym.E(witness.anchor_monomial))
        scale = (coefficient * _sym.E(witness.inverse_anchor_coefficient)).expand()
        if any(
            not _symbolically_equal(candidate, scale * reference)
            for candidate, reference in zip(normalized, references, strict=True)
        ):
            continue
        scalar = _extract_scalar_scale(scale, {})
        if scalar is None:
            continue
        constant, parameter_index = scalar
        if parameter_index is not None or (
            _f64_bits(constant.real),
            _f64_bits(constant.imag),
        ) != (
            _f64_bits(witness.runtime_owned_scale.real),
            _f64_bits(witness.runtime_owned_scale.imag),
        ):
            continue
        return CertifiedRecurrenceFinalizationIntrinsic(
            runtime_template=witness.runtime_template,
            contract_digest=witness.contract_digest,
            constant_scale=constant,
        )
    return None


def _certify_massive_dirac_finalization(
    normalized: Sequence[object],
    *,
    parameter_indices: tuple[int, int],
    component_count: int,
) -> CertifiedRecurrenceFinalizationIntrinsic | None:
    if component_count != 4:
        return None
    _sym._ensure_symbolica()
    generic_mass = _sym.E("m")
    generic_width = _sym.E("w")
    for witness in _MASSIVE_DIRAC_FINALIZATION_WITNESSES:
        for mass_index, width_index in (
            parameter_indices,
            tuple(reversed(parameter_indices)),
        ):
            mass = _sym.Expression.symbol(
                f"recurrence_intrinsic::parameter_{mass_index}"
            )
            width = _sym.Expression.symbol(
                f"recurrence_intrinsic::parameter_{width_index}"
            )
            replacements = (
                _sym.Replacement(generic_mass, mass),
                _sym.Replacement(generic_width, width),
            )
            references = tuple(
                _sym.E(expression).replace_multiple(replacements).expand()
                for expression in witness.expressions
            )
            if any(
                not _symbolically_equal(candidate, 1j * reference)
                for candidate, reference in zip(
                    normalized,
                    references,
                    strict=True,
                )
            ):
                continue
            return CertifiedRecurrenceFinalizationIntrinsic(
                runtime_template=witness.runtime_template,
                contract_digest=witness.contract_digest,
                constant_scale=witness.runtime_owned_scale,
                orientation=witness.orientation,
                mass_parameter_index=mass_index,
                width_parameter_index=width_index,
            )
    return None


def _massive_dirac_parameter_indices(
    input_contracts: Sequence[str],
) -> tuple[int, int]:
    """Validate the unary Dirac input shape and return its unnamed parameters."""

    components: dict[str, set[int]] = {
        "current": set(),
        "momentum": set(),
    }
    parameter_indices: list[int] = []
    declared_symbols: set[str] = set()
    for raw in input_contracts:
        contract = json.loads(raw)
        if not isinstance(contract, Mapping):
            raise ValueError("prepared input contract is not an object")
        role = contract.get("role")
        component = contract.get("component")
        symbol = contract.get("symbol")
        if (
            not isinstance(role, str)
            or type(component) is not int
            or component < 0
            or not isinstance(symbol, str)
            or not symbol
            or symbol in declared_symbols
        ):
            raise ValueError("massive Dirac input contract is malformed")
        declared_symbols.add(symbol)
        if role in components:
            if component in components[role]:
                raise ValueError("massive Dirac input component is duplicated")
            components[role].add(component)
        elif role == "model-parameter":
            parameter_index = contract.get("model_parameter_index")
            if type(parameter_index) is not int or parameter_index < 0:
                raise ValueError("massive Dirac parameter has no stable index")
            parameter_indices.append(parameter_index)
        else:
            raise ValueError("massive Dirac input has an unsupported role")
    if components != {
        "current": {0, 1, 2, 3},
        "momentum": {0, 1, 2, 3},
    }:
        raise ValueError("massive Dirac input has the wrong current/momentum shape")
    if len(parameter_indices) != 2 or parameter_indices[0] == parameter_indices[1]:
        raise ValueError("massive Dirac input requires two distinct parameters")
    return parameter_indices[0], parameter_indices[1]


def _normalized_expressions(
    exact_expressions: Sequence[str],
    input_contracts: Sequence[str],
    *,
    binding_coupling: ExactComplexRationalV1 | None,
    parent_permutation: tuple[int, int] = (0, 1),
) -> tuple[tuple[object, ...], dict[str, int]]:
    _sym._ensure_symbolica()
    if parent_permutation not in {(0, 1), (1, 0)}:
        raise ValueError("intrinsic parent permutation is not binary")
    raw_to_canonical = {
        raw_parent: canonical_parent
        for canonical_parent, raw_parent in enumerate(parent_permutation)
    }
    replacements: list[object] = []
    parameter_symbols: dict[str, int] = {}
    declared: set[str] = set()
    for raw in input_contracts:
        contract = json.loads(raw)
        if not isinstance(contract, Mapping):
            raise ValueError("prepared input contract is not an object")
        role = contract.get("role")
        component = contract.get("component")
        symbol = contract.get("symbol")
        if (
            not isinstance(role, str)
            or type(component) is not int
            or component < 0
            or not isinstance(symbol, str)
            or not symbol
        ):
            raise ValueError("prepared input contract is malformed")
        if symbol in declared:
            continue
        declared.add(symbol)
        source = _sym.E(symbol)
        if role in {"coupling-real", "coupling-imag"}:
            if binding_coupling is None:
                raise ValueError("intrinsic coupling input has no exact binding")
            value = (
                binding_coupling.real
                if role == "coupling-real"
                else binding_coupling.imag
            )
            target = _sym.E(f"({value.numerator})/({value.denominator})")
        elif role == "model-parameter":
            parameter_index = contract.get("model_parameter_index")
            if type(parameter_index) is not int or parameter_index < 0:
                raise ValueError("model-parameter input has no stable index")
            target = _sym.Expression.symbol(
                f"recurrence_intrinsic::parameter_{parameter_index}"
            )
            parameter_symbols[target.to_canonical_string()] = parameter_index
        else:
            if role in {"current", "momentum"}:
                prefix = "l" if role == "current" else "p"
            elif role in {
                "left-current",
                "left-momentum",
                "right-current",
                "right-momentum",
            }:
                raw_parent = 0 if role.startswith("left-") else 1
                canonical_parent = raw_to_canonical[raw_parent]
                prefix = (
                    ("l", "r")[canonical_parent]
                    if role.endswith("-current")
                    else ("p", "q")[canonical_parent]
                )
            else:
                prefix = None
            if prefix is None:
                raise ValueError(f"unsupported intrinsic input role {role!r}")
            target = _sym.Expression.symbol(f"{prefix}{component}")
        replacements.append(_sym.Replacement(source, target))

    normalized = tuple(
        _sym.E(value).replace_multiple(replacements).expand()
        for value in exact_expressions
    )
    allowed = set(parameter_symbols)
    used = {
        symbol.to_canonical_string()
        for expression in normalized
        for symbol in expression.get_all_symbols(False)
    }
    if not used.issubset(
        allowed
        | {
            _sym.Expression.symbol(f"{side}{component}").to_canonical_string()
            for side in ("l", "p", "q", "r")
            for component in range(64)
        }
    ):
        raise ValueError("normalized intrinsic retains an undeclared symbol")
    return normalized, parameter_symbols


def _normalization_candidates(
    exact_expressions: Sequence[str],
    input_contracts: Sequence[str],
    *,
    parent_component_counts: tuple[int, ...],
    binding_coupling: ExactComplexRationalV1 | None,
    allow_nontrivial_parent_permutation: bool,
) -> tuple[
    tuple[
        tuple[object, ...],
        dict[str, int],
        tuple[int, int],
        tuple[int, int],
    ],
    ...,
]:
    if len(parent_component_counts) != 2:
        raise ValueError("contribution intrinsic must have two parents")
    raw_shape = _binary_parent_shape(input_contracts)
    if raw_shape != parent_component_counts:
        raise ValueError(
            "prepared current-input shape disagrees with recurrence parent shape"
        )
    permutations = [(0, 1)]
    if allow_nontrivial_parent_permutation:
        permutations.append((1, 0))
    return tuple(
        (
            *_normalized_expressions(
                exact_expressions,
                input_contracts,
                binding_coupling=binding_coupling,
                parent_permutation=parent_permutation,
            ),
            parent_permutation,
            tuple(raw_shape[index] for index in parent_permutation),
        )
        for parent_permutation in permutations
    )


def _binary_parent_shape(
    input_contracts: Sequence[str],
) -> tuple[int, int]:
    components: dict[str, set[int]] = {
        "left-current": set(),
        "right-current": set(),
    }
    for raw in input_contracts:
        contract = json.loads(raw)
        if not isinstance(contract, Mapping):
            raise ValueError("prepared input contract is not an object")
        role = contract.get("role")
        if role not in components:
            continue
        component = contract.get("component")
        if type(component) is not int or component < 0:
            raise ValueError("prepared current-input component is malformed")
        if component in components[role]:
            raise ValueError("prepared current-input component is duplicated")
        components[role].add(component)
    shape: list[int] = []
    for role in ("left-current", "right-current"):
        values = components[role]
        if not values or values != set(range(max(values) + 1)):
            raise ValueError(f"prepared {role} components are not a contiguous basis")
        shape.append(len(values))
    return shape[0], shape[1]


def _extract_scalar_scale(
    scale: object,
    parameter_symbols: Mapping[str, int],
) -> tuple[complex, int | None] | None:
    _sym._ensure_symbolica()
    used = {symbol.to_canonical_string() for symbol in scale.get_all_symbols(False)}
    if not used:
        coefficient = scale
        parameter_index = None
    elif len(used) == 1 and next(iter(used)) in parameter_symbols:
        canonical = next(iter(used))
        parameter = next(
            symbol
            for symbol in scale.get_all_symbols(False)
            if symbol.to_canonical_string() == canonical
        )
        coefficient = scale.coefficient(parameter)
        if not _symbolically_equal(scale, coefficient * parameter):
            return None
        parameter_index = parameter_symbols[canonical]
    else:
        return None
    if coefficient.get_all_symbols(False):
        return None
    try:
        value = complex(coefficient.evaluate({}))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value.real) or not math.isfinite(value.imag) or value == 0.0:
        return None
    return value, parameter_index


def _symbolically_equal(left: object, right: object) -> bool:
    """Compare exact algebra without conflating spelling with semantics.

    Canonical expanded equality retains the compiler's common-scale
    normalization when decimal binary64 coefficients are re-expanded.  The
    exact-difference fallback also accepts a floating unit where a compact
    witness uses an integer unit.  A perturbed coefficient satisfies neither
    condition and remains rejected.
    """

    expanded_left = left.expand()
    expanded_right = right.expand()
    if expanded_left.to_canonical_string() == expanded_right.to_canonical_string():
        return True
    return (expanded_left - expanded_right).expand().to_canonical_string() == "0"


__all__ = [
    "DIRAC_SCALAR_TO_DIRAC_TEMPLATE",
    "DIRAC_VECTOR_ANTIPARTICLE_TEMPLATE",
    "DIRAC_VECTOR_PARTICLE_TEMPLATE",
    "MASSIVE_DIRAC_ANTIPARTICLE_TEMPLATE",
    "MASSIVE_DIRAC_PARTICLE_TEMPLATE",
    "MASSIVE_DIRAC_RUNTIME_SCALE_BITS",
    "RECURRENCE_FINALIZATION_INTRINSIC_CONTRACT_DIGESTS",
    "RECURRENCE_INTRINSIC_CONTRACT_DIGESTS",
    "RECURRENCE_INTRINSIC_RUNTIME_TEMPLATES",
    "RECURRENCE_INTRINSIC_SCALE_KIND",
    "RECURRENCE_MASSIVE_DIRAC_FINALIZER_KIND",
    "WEYL_PAIR_TO_VECTOR_A_TEMPLATE",
    "WEYL_PAIR_TO_VECTOR_B_TEMPLATE",
    "CertifiedRecurrenceFinalizationIntrinsic",
    "CertifiedRecurrenceIntrinsic",
    "certify_recurrence_contribution_intrinsic",
    "certify_recurrence_finalization_intrinsic",
]
