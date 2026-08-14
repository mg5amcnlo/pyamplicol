# SPDX-License-Identifier: 0BSD
"""Model-generic algebra certificates shared by native evaluator lowerings.

The exact expression matcher currently lives with the Direct-Arena lowering,
but its proof is not backend-specific.  This module exposes a small semantic
vocabulary so spinor and component lowerers consume the same certificate
instead of recognizing particles, UFO names, or builtin-model labels.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from . import compiler_symbolica as _sym
from .base import Model, Vertex
from .prepared_catalog import (
    PreparedKernelCatalog,
    PreparedKernelInput,
    PreparedPropagatorBinding,
    PreparedVertexBinding,
)
from .recurrence_direct_intrinsics import (
    FEYNMAN_VECTOR_PROPAGATOR_TEMPLATE,
    WEYL_PROPAGATOR_NEGATIVE_TEMPLATE,
    WEYL_PROPAGATOR_POSITIVE_TEMPLATE,
    _normalized_expressions,
    certify_recurrence_contribution_intrinsic,
    certify_recurrence_finalization_intrinsic,
)
from .recurrence_template import ExactComplexRationalV1

CertifiedSpinorProcessFamily = Literal[
    "pure-gluon",
    "single-massless-quark-line",
    "single-massive-quark-line",
    "single-massless-quark-line-massive-neutral-vector",
]

_MASSIVE_QUARK_FAMILY = "single-massive-quark-line"
_MASSIVE_NEUTRAL_VECTOR_FAMILY = "single-massless-quark-line-massive-neutral-vector"

# These are the binary64 values in the certified color-ordered component
# kernels.  Keeping the scalar part of the witness is essential: recognizing
# only the tensor shape would admit a model with a different relative weight
# between exchange and contact diagrams.
_CERTIFIED_INVERSE_SQRT_TWO = 0.707106781186547


class KernelPrimitiveKind(StrEnum):
    """Certified interaction/propagator algebra, independent of its backend."""

    COLOR_ORDERED_THREE_VECTOR = "color-ordered-three-vector"
    WEYL_VECTOR_TO_WEYL_A = "weyl-vector-to-weyl-a"
    WEYL_VECTOR_TO_WEYL_B = "weyl-vector-to-weyl-b"
    DIRAC_VECTOR_TO_DIRAC = "dirac-vector-to-dirac"
    DIRAC_PAIR_TO_VECTOR = "dirac-pair-to-vector"
    ANTISYMMETRIC_TENSOR_VECTOR = "antisymmetric-tensor-vector"
    VECTOR_WEDGE_VECTOR = "vector-wedge-vector"
    WEYL_PROPAGATOR_A = "weyl-propagator-a"
    WEYL_PROPAGATOR_B = "weyl-propagator-b"
    FEYNMAN_VECTOR_PROPAGATOR = "feynman-vector-propagator"
    MASSIVE_DIRAC_PROPAGATOR = "massive-dirac-propagator"


_CONTRIBUTION_KINDS = {
    "rusticol.recurrence-intrinsic.color-ordered-three-vector.v1": (
        KernelPrimitiveKind.COLOR_ORDERED_THREE_VECTOR
    ),
    "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-a.v1": (
        KernelPrimitiveKind.WEYL_VECTOR_TO_WEYL_A
    ),
    "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-b.v1": (
        KernelPrimitiveKind.WEYL_VECTOR_TO_WEYL_B
    ),
    "rusticol.recurrence-intrinsic.antisymmetric-tensor-vector.v1": (
        KernelPrimitiveKind.ANTISYMMETRIC_TENSOR_VECTOR
    ),
    "rusticol.recurrence-intrinsic.vector-wedge-vector.v1": (
        KernelPrimitiveKind.VECTOR_WEDGE_VECTOR
    ),
}
_FINALIZATION_KINDS = {
    WEYL_PROPAGATOR_POSITIVE_TEMPLATE: KernelPrimitiveKind.WEYL_PROPAGATOR_A,
    WEYL_PROPAGATOR_NEGATIVE_TEMPLATE: KernelPrimitiveKind.WEYL_PROPAGATOR_B,
    FEYNMAN_VECTOR_PROPAGATOR_TEMPLATE: (KernelPrimitiveKind.FEYNMAN_VECTOR_PROPAGATOR),
}


@dataclass(frozen=True, slots=True)
class CertifiedKernelPrimitive:
    """One exact primitive witness plus its model-owned scalar scale."""

    kind: KernelPrimitiveKind
    runtime_template: str
    contract_digest: str
    constant_scale: complex
    model_parameter_index: int | None
    parent_permutation: tuple[int, int] = (0, 1)


@dataclass(frozen=True, slots=True)
class SpinorAlgebraCertificate:
    """Ephemeral proof that one spinor slice matches its model algebra."""

    process_family: CertifiedSpinorProcessFamily
    gluon_count: int
    quark_pdg: int | None
    primitives: tuple[KernelPrimitiveKind, ...]
    spinor_parameter_names: tuple[str, ...] = ()


class SpinorAlgebraCertificationError(ValueError):
    """The selected model algebra is not the algebra lowered by the DAG."""


def certify_contribution_kernel_primitive(
    *,
    exact_expressions: Sequence[str],
    input_contracts: Sequence[str],
    parent_component_counts: tuple[int, ...],
    destination_component_count: int,
    binding_coupling: ExactComplexRationalV1 | None,
    allow_nontrivial_parent_permutation: bool = False,
) -> CertifiedKernelPrimitive | None:
    """Classify one exact binary contribution into the shared vocabulary."""

    certified = certify_recurrence_contribution_intrinsic(
        exact_expressions=exact_expressions,
        input_contracts=input_contracts,
        parent_component_counts=parent_component_counts,
        destination_component_count=destination_component_count,
        binding_coupling=binding_coupling,
        allow_nontrivial_parent_permutation=allow_nontrivial_parent_permutation,
    )
    if certified is None:
        return None
    kind = _CONTRIBUTION_KINDS.get(certified.runtime_template)
    if kind is None:
        return None
    return CertifiedKernelPrimitive(
        kind=kind,
        runtime_template=certified.runtime_template,
        contract_digest=certified.contract_digest,
        constant_scale=certified.constant_scale,
        model_parameter_index=certified.model_parameter_index,
        parent_permutation=certified.parent_permutation,
    )


def certify_finalization_kernel_primitive(
    *,
    exact_expressions: Sequence[str],
    input_contracts: Sequence[str],
    component_count: int,
) -> KernelPrimitiveKind | None:
    """Classify one exact current finalizer into the shared vocabulary."""

    certified = certify_recurrence_finalization_intrinsic(
        exact_expressions=exact_expressions,
        input_contracts=input_contracts,
        component_count=component_count,
    )
    if certified is None:
        return None
    return _FINALIZATION_KINDS.get(certified.runtime_template)


def certify_spinor_process_algebra(
    model: Model,
    *,
    process_family: CertifiedSpinorProcessFamily,
    gluon_count: int,
    quark_pdg: int | None = None,
    spinor_parameters: Sequence[tuple[str, float]] = (),
) -> SpinorAlgebraCertificate:
    """Prove the exact local model algebra consumed by one spinor graph.

    This certificate is deliberately generation-local.  It reuses the exact
    prepared-kernel boundary, but neither persists another digest nor makes a
    prepared model a prerequisite for the compact spinor artifact.
    """

    if not isinstance(model, Model):
        raise TypeError("spinor algebra certification requires a Model")
    if process_family not in {
        "pure-gluon",
        "single-massless-quark-line",
        _MASSIVE_QUARK_FAMILY,
        _MASSIVE_NEUTRAL_VECTOR_FAMILY,
    }:
        raise SpinorAlgebraCertificationError(
            f"unsupported spinor process family {process_family!r}"
        )
    if type(gluon_count) is not int or gluon_count < 0:
        raise SpinorAlgebraCertificationError(
            "spinor algebra certification requires a nonnegative gluon count"
        )
    if process_family == "pure-gluon":
        if gluon_count < 4 or quark_pdg is not None:
            raise SpinorAlgebraCertificationError(
                "pure-gluon algebra certification requires at least four gluons "
                "and cannot select a quark"
            )
    elif process_family == "single-massless-quark-line" and gluon_count < 2:
        raise SpinorAlgebraCertificationError(
            "massless quark-line algebra certification requires at least two gluons"
        )
    elif process_family == _MASSIVE_QUARK_FAMILY and gluon_count != 2:
        raise SpinorAlgebraCertificationError(
            "massive quark-line algebra certification requires exactly two gluons"
        )
    elif process_family == _MASSIVE_NEUTRAL_VECTOR_FAMILY and gluon_count > 2:
        raise SpinorAlgebraCertificationError(
            "massive-neutral-vector algebra certification supports at most two gluons"
        )
    if process_family not in {"pure-gluon", _MASSIVE_QUARK_FAMILY} and (
        type(quark_pdg) is not int
        or quark_pdg <= 0
        or not model.is_fundamental_colored_fermion(quark_pdg)
        or not model.is_chiral_eligible(quark_pdg)
        or model.mass(quark_pdg) != 0.0
    ):
        raise SpinorAlgebraCertificationError(
            "quark-line algebra certification requires one massless fundamental "
            "colored fermion"
        )
    if process_family == _MASSIVE_QUARK_FAMILY:
        if (
            quark_pdg != 6
            or not model.is_fundamental_colored_fermion(6)
            or not model.is_fermion(6)
            or model.is_chiral_eligible(6)
            or not math.isfinite(float(model.mass(6)))
            or model.mass(6) <= 0.0
            or not math.isfinite(float(model.width(6)))
            or model.width(6) < 0.0
        ):
            raise SpinorAlgebraCertificationError(
                "massive quark-line algebra certification requires massive Dirac top"
            )
        certified_parameters = _validated_spinor_parameters(spinor_parameters)
        if certified_parameters != (
            ("particle.6.mass", float(model.mass(6))),
            ("particle.6.width", float(model.width(6))),
        ):
            raise SpinorAlgebraCertificationError(
                "massive quark-line algebra requires ordered top mass and width "
                "parameters"
            )
    elif process_family == _MASSIVE_NEUTRAL_VECTOR_FAMILY:
        if model.mass(23) <= 0.0:
            raise SpinorAlgebraCertificationError(
                "massive-neutral-vector algebra certification requires massive PDG 23"
            )
        certified_parameters = _validated_spinor_parameters(spinor_parameters)
        if len(certified_parameters) != 2:
            raise SpinorAlgebraCertificationError(
                "massive-neutral-vector algebra requires two ordered chiral "
                "coupling parameters"
            )
    elif spinor_parameters:
        raise SpinorAlgebraCertificationError(
            "only parameterized spinor algebra accepts local parameters"
        )
    else:
        certified_parameters = ()
    if gluon_count and (
        not model.is_massless_adjoint_vector(21) or model.mass(21) != 0.0
    ):
        raise SpinorAlgebraCertificationError(
            "spinor algebra requires PDG 21 to be a massless adjoint vector"
        )

    # Lazy construction keeps importing spinor metadata independent of the
    # compiler stack.  Only an explicitly requested spinor artifact pays for
    # this one exact model-algebra proof.
    from .prepared_catalog import build_prepared_kernel_catalog

    try:
        catalog = build_prepared_kernel_catalog(model)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise SpinorAlgebraCertificationError(
            f"cannot construct exact model kernels for spinor certification: {exc}"
        ) from exc
    return _certify_spinor_catalog(
        model,
        catalog,
        process_family=process_family,
        gluon_count=gluon_count,
        quark_pdg=quark_pdg,
        spinor_parameters=certified_parameters,
    )


def _certify_spinor_catalog(
    model: Model,
    catalog: PreparedKernelCatalog,
    *,
    process_family: CertifiedSpinorProcessFamily,
    gluon_count: int,
    quark_pdg: int | None,
    spinor_parameters: Sequence[tuple[str, float]] = (),
) -> SpinorAlgebraCertificate:
    """Certify one already constructed catalog (split out for focused tests)."""

    certified_parameters = _validated_spinor_parameters(spinor_parameters)
    witnessed: list[KernelPrimitiveKind] = []
    if process_family == "pure-gluon" or gluon_count >= 2:
        three_vector = _unique_vertex_binding(
            catalog,
            "three-gluon current",
            lambda item: (
                item.key.particles == (21, 21, 21)
                and item.key.left_chirality == 0
                and item.key.right_chirality == 0
                and item.key.result_chirality == 0
            ),
        )
        _certify_vertex_binding(
            model,
            catalog,
            three_vector,
            context="three-gluon current",
            expected_kind=KernelPrimitiveKind.COLOR_ORDERED_THREE_VECTOR,
            expected_scale=complex(0.0, _CERTIFIED_INVERSE_SQRT_TWO),
            expected_model_parent_order=(0, 1),
            expected_equivalence_factor=(1.0, 0.0),
            expected_color_structure="adjoint-structure-constant",
        )
        witnessed.append(KernelPrimitiveKind.COLOR_ORDERED_THREE_VECTOR)
        _certify_propagator_binding(
            catalog,
            particle_id=21,
            chirality=0,
            expected_kind=KernelPrimitiveKind.FEYNMAN_VECTOR_PROPAGATOR,
            context="massless-gluon propagator",
        )
        witnessed.append(KernelPrimitiveKind.FEYNMAN_VECTOR_PROPAGATOR)

    if process_family == "pure-gluon" or gluon_count >= 3:
        wedge = _unique_vertex_binding(
            catalog,
            "two-gluon auxiliary contact",
            lambda item: (
                item.key.particles[:2] == (21, 21)
                and item.key.left_chirality == 0
                and item.key.right_chirality == 0
                and item.key.result_chirality == 0
                and item.result_state.dimension == 6
                and model.auxiliary_kind(item.key.particles[2]) is not None
            ),
        )
        auxiliary_pdg = wedge.key.particles[2]
        _certify_vertex_binding(
            model,
            catalog,
            wedge,
            context="two-gluon auxiliary contact",
            expected_kind=KernelPrimitiveKind.VECTOR_WEDGE_VECTOR,
            expected_scale=complex(1.0, 0.0),
            expected_model_parent_order=(0, 1),
            expected_equivalence_factor=(1.0, 0.0),
            expected_color_structure="adjoint-structure-constant",
        )
        witnessed.append(KernelPrimitiveKind.VECTOR_WEDGE_VECTOR)
        for particles, parent_order, factor, context in (
            (
                (auxiliary_pdg, 21, 21),
                (0, 1),
                (1.0, 0.0),
                "auxiliary-gluon contact",
            ),
            (
                (21, auxiliary_pdg, 21),
                (1, 0),
                (-1.0, 0.0),
                "gluon-auxiliary contact",
            ),
        ):
            binding = _unique_vertex_binding(
                catalog,
                context,
                lambda item, particles=particles: (
                    item.key.particles == particles
                    and item.key.left_chirality == 0
                    and item.key.right_chirality == 0
                    and item.key.result_chirality == 0
                ),
            )
            _certify_vertex_binding(
                model,
                catalog,
                binding,
                context=context,
                expected_kind=(KernelPrimitiveKind.ANTISYMMETRIC_TENSOR_VECTOR),
                expected_scale=complex(0.0, 0.5),
                expected_model_parent_order=parent_order,
                expected_equivalence_factor=factor,
                expected_color_structure="adjoint-structure-constant",
            )
            witnessed.append(KernelPrimitiveKind.ANTISYMMETRIC_TENSOR_VECTOR)
        auxiliary_propagators = tuple(
            item
            for item in catalog.propagator_bindings
            if item.key.particle_id == auxiliary_pdg and item.key.chirality == 0
        )
        if (
            len(auxiliary_propagators) != 1
            or auxiliary_propagators[0].applies_propagator
        ):
            raise SpinorAlgebraCertificationError(
                "the certified four-gluon contact auxiliary must be algebraic"
            )

    if process_family not in {"pure-gluon", _MASSIVE_QUARK_FAMILY} and gluon_count:
        assert quark_pdg is not None
        for chirality, kind in (
            (
                -1,
                KernelPrimitiveKind.WEYL_VECTOR_TO_WEYL_B,
            ),
            (
                1,
                KernelPrimitiveKind.WEYL_VECTOR_TO_WEYL_A,
            ),
        ):
            context = f"quark-gluon chirality {chirality:+d} current"
            binding = _unique_vertex_binding(
                catalog,
                context,
                lambda item, chirality=chirality: (
                    item.key.particles == (quark_pdg, 21, quark_pdg)
                    and item.key.left_chirality == chirality
                    and item.key.right_chirality == 0
                    and item.key.result_chirality == chirality
                ),
            )
            _certify_vertex_binding(
                model,
                catalog,
                binding,
                context=context,
                expected_kind=kind,
                expected_scale=complex(_CERTIFIED_INVERSE_SQRT_TWO, 0.0),
                expected_model_parent_order=(0, 1),
                expected_equivalence_factor=(1.0, 0.0),
                expected_color_structure="fundamental-generator",
            )
            witnessed.append(kind)

    if process_family == _MASSIVE_QUARK_FAMILY:
        assert quark_pdg == 6
        binding = _unique_vertex_binding(
            catalog,
            "massive quark-gluon current",
            lambda item: (
                item.key.particles == (6, 21, 6)
                and item.key.left_chirality == 0
                and item.key.right_chirality == 0
                and item.key.result_chirality == 0
            ),
        )
        _certify_massive_dirac_qg_binding(model, catalog, binding)
        witnessed.append(KernelPrimitiveKind.DIRAC_VECTOR_TO_DIRAC)
        closure_binding = _unique_vertex_binding(
            catalog,
            "massive Dirac-vector closure",
            lambda item: (
                item.key.particles == (-6, 6, 21)
                and item.key.left_chirality == 0
                and item.key.right_chirality == 0
                and item.key.result_chirality == 0
            ),
        )
        _certify_massive_dirac_pair_to_vector_binding(
            model,
            catalog,
            closure_binding,
        )
        witnessed.append(KernelPrimitiveKind.DIRAC_PAIR_TO_VECTOR)
        _certify_massive_dirac_propagator(catalog, particle_id=6)
        witnessed.append(KernelPrimitiveKind.MASSIVE_DIRAC_PROPAGATOR)

    if process_family == _MASSIVE_NEUTRAL_VECTOR_FAMILY:
        assert quark_pdg is not None
        if len(certified_parameters) != 2:
            raise SpinorAlgebraCertificationError(
                "massive-neutral-vector algebra requires two ordered chiral "
                "coupling parameters"
            )
        parameter_names = tuple(name for name, _value in certified_parameters)
        coupling_values = (
            certified_parameters[0][1],
            certified_parameters[1][1],
        )
        for chirality, kind, output_factor_source in (
            (-1, KernelPrimitiveKind.WEYL_VECTOR_TO_WEYL_B, "coupling-real"),
            (1, KernelPrimitiveKind.WEYL_VECTOR_TO_WEYL_A, "coupling-imag"),
        ):
            context = f"quark-Z chirality {chirality:+d} current"
            binding = _unique_vertex_binding(
                catalog,
                context,
                lambda item, chirality=chirality: (
                    item.key.particles == (quark_pdg, 23, quark_pdg)
                    and item.key.left_chirality == chirality
                    and item.key.right_chirality == 0
                    and item.key.result_chirality == chirality
                ),
            )
            if binding.key.coupling != coupling_values:
                raise SpinorAlgebraCertificationError(
                    f"{context} coupling components do not match the ordered "
                    "spinor parameters"
                )
            if _runtime_parameter_names_for_binding(model, binding) != parameter_names:
                raise SpinorAlgebraCertificationError(
                    f"{context} coupling names do not match the active model binding"
                )
            _certify_vertex_binding(
                model,
                catalog,
                binding,
                context=context,
                expected_kind=kind,
                expected_scale=complex(_CERTIFIED_INVERSE_SQRT_TWO, 0.0),
                expected_model_parent_order=(0, 1),
                expected_equivalence_factor=(1.0, 0.0),
                expected_color_structure="color-identity",
                expected_binding_coupling=coupling_values,
                expected_output_factor_source=output_factor_source,
                expected_coupling_orders=(("QED", 1),),
            )
            witnessed.append(kind)

    return SpinorAlgebraCertificate(
        process_family=process_family,
        gluon_count=gluon_count,
        quark_pdg=quark_pdg,
        primitives=tuple(witnessed),
        spinor_parameter_names=tuple(name for name, _value in certified_parameters),
    )


def _validated_spinor_parameters(
    parameters: Sequence[tuple[str, float]],
) -> tuple[tuple[str, float], ...]:
    result: list[tuple[str, float]] = []
    for item in parameters:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or type(item[1]) not in {int, float}
            or not math.isfinite(item[1])
        ):
            raise SpinorAlgebraCertificationError(
                "spinor parameters must be ordered nonempty name/real-value pairs"
            )
        result.append((item[0], float(item[1])))
    names = tuple(name for name, _value in result)
    if len(set(names)) != len(names):
        raise SpinorAlgebraCertificationError(
            "spinor coupling parameter names must be distinct"
        )
    return tuple(result)


def _runtime_parameter_names_for_binding(
    model: Model,
    binding: PreparedVertexBinding,
) -> tuple[str, str]:
    # This import is intentionally local: parameter names belong to the
    # generation contract, while the exact tensor proof above remains usable
    # by model and backend code without importing the generation package.
    from ..generation.contracts import runtime_coupling_parameter_names

    names = runtime_coupling_parameter_names(
        binding.key.kind,
        binding.key.particles,
        binding.key.coupling,
        model=model,
    )
    if len(names) != 2 or names[0] is None or names[1] is None:
        raise SpinorAlgebraCertificationError(
            "the certified chiral vertex requires two named coupling components"
        )
    return str(names[0]), str(names[1])


def _unique_vertex_binding(
    catalog: PreparedKernelCatalog,
    context: str,
    predicate: Callable[[PreparedVertexBinding], bool],
) -> PreparedVertexBinding:
    matches = tuple(item for item in catalog.vertex_bindings if predicate(item))
    if len(matches) != 1:
        raise SpinorAlgebraCertificationError(
            f"spinor algebra requires exactly one {context}; found {len(matches)}"
        )
    return matches[0]


def _certify_vertex_binding(
    model: Model,
    catalog: PreparedKernelCatalog,
    binding: PreparedVertexBinding,
    *,
    context: str,
    expected_kind: KernelPrimitiveKind,
    expected_scale: complex,
    expected_model_parent_order: tuple[int, int],
    expected_equivalence_factor: tuple[float, float],
    expected_color_structure: str,
    expected_binding_coupling: tuple[float, float] = (1.0, 0.0),
    expected_output_factor_source: str = "none",
    expected_coupling_orders: tuple[tuple[str, int], ...] = (("QCD", 1),),
) -> None:
    kernel = catalog.by_id[binding.kernel_id]
    canonical_states = tuple(
        (binding.left_state, binding.right_state)[index]
        for index in binding.canonical_input_order
    )
    certified = certify_contribution_kernel_primitive(
        exact_expressions=kernel.exact_expressions,
        input_contracts=_input_contracts(kernel.inputs),
        parent_component_counts=tuple(state.dimension for state in canonical_states),
        destination_component_count=binding.result_state.dimension,
        binding_coupling=ExactComplexRationalV1.from_binary64(*binding.key.coupling),
        allow_nontrivial_parent_permutation=True,
    )
    if certified is None:
        raise SpinorAlgebraCertificationError(
            f"{context} is not a certified shared kernel primitive"
        )
    model_parent_order = tuple(
        binding.canonical_input_order[index] for index in certified.parent_permutation
    )
    if (
        certified.kind is not expected_kind
        or certified.constant_scale != expected_scale
        or certified.model_parameter_index is not None
        or model_parent_order != expected_model_parent_order
    ):
        raise SpinorAlgebraCertificationError(
            f"{context} has unsupported certified algebra: "
            f"kind={certified.kind.value!r}, scale={certified.constant_scale!r}, "
            f"parameter={certified.model_parameter_index!r}, "
            f"parent_order={model_parent_order!r}"
        )
    vertex = Vertex(
        binding.key.kind,
        binding.key.particles,
        binding.key.coupling,
    )
    if (
        binding.equivalence_factor != expected_equivalence_factor
        or binding.output_factor_source != expected_output_factor_source
        or binding.contact_orbit_certificates
        or binding.contact_orbit_steps
        or binding.key.coupling != expected_binding_coupling
        or model.vertex_coupling_orders(vertex) != expected_coupling_orders
        or model.vertex_color_structure(vertex) != expected_color_structure
        or model.vertex_color_weight(vertex, color_accuracy="lc") != (1.0, 0.0)
    ):
        raise SpinorAlgebraCertificationError(
            f"{context} has unsupported coupling, color, or binding factors"
        )


def _certify_massive_dirac_qg_binding(
    model: Model,
    catalog: PreparedKernelCatalog,
    binding: PreparedVertexBinding,
) -> None:
    """Authenticate the exact four-component q-g current used by the DAG."""

    context = "massive quark-gluon current"
    if (
        binding.canonical_input_order != (0, 1)
        or binding.left_state.basis != "dirac"
        or binding.left_state.dimension != 4
        or binding.right_state.dimension != 4
        or binding.result_state.basis != "dirac"
        or binding.result_state.dimension != 4
        or binding.equivalence_factor != (1.0, 0.0)
        or binding.output_factor_source != "none"
        or binding.contact_orbit_certificates
        or binding.contact_orbit_steps
        or binding.key.coupling != (1.0, 0.0)
        or model.vertex_coupling_orders(
            Vertex(binding.key.kind, binding.key.particles, binding.key.coupling)
        )
        != (("QCD", 1),)
        or model.vertex_color_structure(
            Vertex(binding.key.kind, binding.key.particles, binding.key.coupling)
        )
        != "fundamental-generator"
        or model.vertex_color_weight(
            Vertex(binding.key.kind, binding.key.particles, binding.key.coupling),
            color_accuracy="lc",
        )
        != (1.0, 0.0)
    ):
        raise SpinorAlgebraCertificationError(
            f"{context} has unsupported state, coupling, color, or binding metadata"
        )
    kernel = catalog.by_id[binding.kernel_id]
    try:
        normalized, parameter_symbols = _normalized_expressions(
            kernel.exact_expressions,
            _input_contracts(kernel.inputs),
            binding_coupling=None,
        )
    except (TypeError, ValueError) as exc:
        raise SpinorAlgebraCertificationError(
            f"{context} exact algebra could not be normalized"
        ) from exc
    if parameter_symbols:
        raise SpinorAlgebraCertificationError(
            f"{context} unexpectedly depends on model parameters"
        )
    _sym._ensure_symbolica()
    fermion = tuple(_sym.Expression.symbol(f"l{index}") for index in range(4))
    vector = tuple(_sym.Expression.symbol(f"r{index}") for index in range(4))
    f1, f2, f3, f4 = fermion
    v0, v1, v2, v3 = vector
    tmp1, tmp2 = v0 + v3, v0 - v3
    tmp3, tmp4 = v1 + 1j * v2, v1 - 1j * v2
    prefactor = complex(0.0, _CERTIFIED_INVERSE_SQRT_TWO)
    expected = (
        prefactor * (tmp1 * f3 + tmp3 * f4),
        prefactor * (tmp2 * f4 + tmp4 * f3),
        prefactor * (tmp2 * f1 - tmp3 * f2),
        prefactor * (tmp1 * f2 - tmp4 * f1),
    )
    if _canonical_expressions(normalized) != _canonical_expressions(expected):
        raise SpinorAlgebraCertificationError(
            f"{context} does not match the certified Dirac-vector algebra"
        )


def _certify_massive_dirac_pair_to_vector_binding(
    model: Model,
    catalog: PreparedKernelCatalog,
    binding: PreparedVertexBinding,
) -> None:
    """Authenticate the terminal qbar-q-vector bilinear used by the DAG."""

    context = "massive Dirac-vector closure"
    vertex = Vertex(binding.key.kind, binding.key.particles, binding.key.coupling)
    if (
        binding.canonical_input_order != (0, 1)
        or binding.left_state.orientation != "antiparticle"
        or binding.left_state.basis != "dirac"
        or binding.left_state.dimension != 4
        or binding.right_state.orientation != "particle"
        or binding.right_state.basis != "dirac"
        or binding.right_state.dimension != 4
        or binding.result_state.basis != "lorentz-vector"
        or binding.result_state.dimension != 4
        or binding.equivalence_factor != (1.0, 0.0)
        or binding.output_factor_source != "none"
        or binding.contact_orbit_certificates
        or binding.contact_orbit_steps
        or binding.key.coupling != (1.0, 0.0)
        or model.vertex_coupling_orders(vertex) != (("QCD", 1),)
        or model.vertex_color_structure(vertex) != "fundamental-generator"
        or model.vertex_color_weight(vertex, color_accuracy="lc") != (1.0, 0.0)
    ):
        raise SpinorAlgebraCertificationError(
            f"{context} has unsupported state, coupling, color, or binding metadata"
        )
    kernel = catalog.by_id[binding.kernel_id]
    try:
        normalized, parameter_symbols = _normalized_expressions(
            kernel.exact_expressions,
            _input_contracts(kernel.inputs),
            binding_coupling=None,
        )
    except (TypeError, ValueError) as exc:
        raise SpinorAlgebraCertificationError(
            f"{context} exact algebra could not be normalized"
        ) from exc
    if parameter_symbols:
        raise SpinorAlgebraCertificationError(
            f"{context} unexpectedly depends on model parameters"
        )
    _sym._ensure_symbolica()
    antiquark = tuple(_sym.Expression.symbol(f"l{index}") for index in range(4))
    quark = tuple(_sym.Expression.symbol(f"r{index}") for index in range(4))
    a1, a2, a3, a4 = antiquark
    f1, f2, f3, f4 = quark
    left = (
        f3 * a1 + f4 * a2,
        -(f4 * a1 + f3 * a2),
        1j * (-f4 * a1 + f3 * a2),
        -f3 * a1 + f4 * a2,
    )
    right = (
        f1 * a3 + f2 * a4,
        f1 * a4 + f2 * a3,
        1j * (-f1 * a4 + f2 * a3),
        f1 * a3 - f2 * a4,
    )
    prefactor = complex(0.0, _CERTIFIED_INVERSE_SQRT_TWO)
    expected = tuple(
        prefactor * (left[index] + right[index]) for index in range(4)
    )
    if _canonical_expressions(normalized) != _canonical_expressions(expected):
        raise SpinorAlgebraCertificationError(
            f"{context} does not match the certified qbar-gamma-vector-q bilinear"
        )


def _certify_massive_dirac_propagator(
    catalog: PreparedKernelCatalog,
    *,
    particle_id: int,
) -> None:
    """Authenticate i(p-slash+m)/(p²-m²+i m width) in the Dirac basis."""

    context = "massive Dirac propagator"
    matches = tuple(
        item
        for item in catalog.propagator_bindings
        if item.key.particle_id == particle_id and item.key.chirality == 0
    )
    if len(matches) != 1:
        raise SpinorAlgebraCertificationError(
            f"spinor algebra requires exactly one {context}; found {len(matches)}"
        )
    binding = matches[0]
    if (
        not binding.applies_propagator
        or binding.kernel_id is None
        or binding.propagator_kind != "dirac-fermion"
        or binding.mass_class != "massive"
        or binding.gauge is not None
        or binding.state.basis != "dirac"
        or binding.state.dimension != 4
        or binding.model_parameters
        != ("particle.6.mass", "particle.6.width")
    ):
        raise SpinorAlgebraCertificationError(
            f"{context} has unsupported state or parameter metadata"
        )
    kernel = catalog.by_id[binding.kernel_id]
    parameter_indices = {
        item.model_parameter_name: item.model_parameter_index
        for item in kernel.inputs
        if item.model_parameter_name is not None
    }
    if set(parameter_indices) != {"particle.6.mass", "particle.6.width"} or any(
        type(index) is not int for index in parameter_indices.values()
    ):
        raise SpinorAlgebraCertificationError(
            f"{context} does not expose the exact top mass/width inputs"
        )
    try:
        normalized, normalized_parameters = _normalized_expressions(
            kernel.exact_expressions,
            _input_contracts(kernel.inputs),
            binding_coupling=None,
        )
    except (TypeError, ValueError) as exc:
        raise SpinorAlgebraCertificationError(
            f"{context} exact algebra could not be normalized"
        ) from exc
    expected_parameter_indices = set(parameter_indices.values())
    if set(normalized_parameters.values()) != expected_parameter_indices:
        raise SpinorAlgebraCertificationError(
            f"{context} does not consume exactly its mass/width inputs"
        )
    _sym._ensure_symbolica()
    current = tuple(_sym.Expression.symbol(f"l{index}") for index in range(4))
    momentum = tuple(_sym.Expression.symbol(f"p{index}") for index in range(4))
    mass = _sym.Expression.symbol(
        f"recurrence_intrinsic::parameter_{parameter_indices['particle.6.mass']}"
    )
    width = _sym.Expression.symbol(
        f"recurrence_intrinsic::parameter_{parameter_indices['particle.6.width']}"
    )
    energy, px, py, pz = momentum
    denominator = energy * energy - px * px - py * py - pz * pz - mass * mass
    denominator += 1j * mass * width
    prefactor = 1j / denominator
    tmp1, tmp2 = energy + pz, energy - pz
    tmp3, tmp4 = px + 1j * py, px - 1j * py
    f1, f2, f3, f4 = current
    expected = (
        (tmp1 * f3 + tmp3 * f4 + mass * f1) * prefactor,
        (tmp2 * f4 + tmp4 * f3 + mass * f2) * prefactor,
        (tmp2 * f1 - tmp3 * f2 + mass * f3) * prefactor,
        (tmp1 * f2 - tmp4 * f1 + mass * f4) * prefactor,
    )
    if _canonical_expressions(normalized) != _canonical_expressions(expected):
        raise SpinorAlgebraCertificationError(
            f"{context} does not match i(p-slash+m)/(p²-m²+i m width)"
        )


def _canonical_expressions(expressions: Sequence[object]) -> tuple[str, ...]:
    return tuple(
        expression.expand().to_canonical_string() for expression in expressions
    )


def _certify_propagator_binding(
    catalog: PreparedKernelCatalog,
    *,
    particle_id: int,
    chirality: int,
    expected_kind: KernelPrimitiveKind,
    context: str,
) -> None:
    matches = tuple(
        item
        for item in catalog.propagator_bindings
        if item.key.particle_id == particle_id and item.key.chirality == chirality
    )
    if len(matches) != 1:
        raise SpinorAlgebraCertificationError(
            f"spinor algebra requires exactly one {context}; found {len(matches)}"
        )
    binding: PreparedPropagatorBinding = matches[0]
    if (
        not binding.applies_propagator
        or binding.kernel_id is None
        or binding.mass_class != "massless"
        or binding.model_parameters
    ):
        raise SpinorAlgebraCertificationError(
            f"{context} is not a parameter-free massless propagator"
        )
    kernel = catalog.by_id[binding.kernel_id]
    kind = certify_finalization_kernel_primitive(
        exact_expressions=kernel.exact_expressions,
        input_contracts=_input_contracts(kernel.inputs),
        component_count=binding.state.dimension,
    )
    if kind is not expected_kind:
        raise SpinorAlgebraCertificationError(
            f"{context} has unsupported exact algebra {kind!r}"
        )


def _input_contracts(inputs: Sequence[PreparedKernelInput]) -> tuple[str, ...]:
    return tuple(
        json.dumps(
            item.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for item in inputs
    )


__all__ = [
    "CertifiedKernelPrimitive",
    "CertifiedSpinorProcessFamily",
    "KernelPrimitiveKind",
    "SpinorAlgebraCertificate",
    "SpinorAlgebraCertificationError",
    "certify_contribution_kernel_primitive",
    "certify_finalization_kernel_primitive",
    "certify_spinor_process_algebra",
]
