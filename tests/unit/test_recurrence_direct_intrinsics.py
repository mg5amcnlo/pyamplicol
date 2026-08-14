# SPDX-License-Identifier: 0BSD
# ruff: noqa: RUF001

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from pyamplicol.models.recurrence_direct_intrinsics import (
    CHIRAL_DIRAC_VECTOR_ANTIPARTICLE_TEMPLATE,
    CHIRAL_DIRAC_VECTOR_PARTICLE_TEMPLATE,
    DIRAC_SCALAR_TO_DIRAC_TEMPLATE,
    DIRAC_VECTOR_ANTIPARTICLE_TEMPLATE,
    DIRAC_VECTOR_PARTICLE_TEMPLATE,
    MASSIVE_DIRAC_ANTIPARTICLE_TEMPLATE,
    MASSIVE_DIRAC_PARTICLE_TEMPLATE,
    MASSIVE_VECTOR_UNITARY_TEMPLATE,
    RECURRENCE_INTRINSIC_SCALE_KIND,
    RECURRENCE_MASSIVE_DIRAC_FINALIZER_KIND,
    RECURRENCE_MASSIVE_VECTOR_FINALIZER_KIND,
    WEYL_PAIR_TO_VECTOR_A_TEMPLATE,
    WEYL_PAIR_TO_VECTOR_B_TEMPLATE,
    CertifiedChiralDiracVectorIntrinsic,
    certify_recurrence_contribution_intrinsic,
    certify_recurrence_finalization_intrinsic,
)
from pyamplicol.models.recurrence_template import ExactComplexRationalV1


def _contracts(
    left_components: int,
    right_components: int,
    *,
    parameter_index: int | None = None,
    coupling: bool = False,
) -> tuple[str, ...]:
    values: list[dict[str, object]] = []
    for role, count, prefix in (
        ("left-current", left_components, "left"),
        ("right-current", right_components, "right"),
    ):
        values.extend(
            {
                "component": component,
                "role": role,
                "symbol": f"model::prepared::{prefix}_{component}",
            }
            for component in range(count)
        )
    if parameter_index is not None:
        values.append(
            {
                "component": 0,
                "model_parameter_index": parameter_index,
                "role": "model-parameter",
                "symbol": "model::prepared::parameter",
            }
        )
    if coupling:
        values.extend(
            (
                {
                    "component": 0,
                    "role": "coupling-real",
                    "symbol": "model::prepared::coupling_re",
                },
                {
                    "component": 0,
                    "role": "coupling-imag",
                    "symbol": "model::prepared::coupling_im",
                },
            )
        )
    return tuple(
        json.dumps(
            item,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for item in values
    )


def _dirac_vector_contracts_with_parameters(
    *parameter_indexes: int,
) -> tuple[str, ...]:
    parameters = tuple(
        json.dumps(
            {
                "component": 0,
                "model_parameter_index": parameter_index,
                "role": "model-parameter",
                "symbol": f"model::prepared::parameter_{parameter_index}",
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for parameter_index in parameter_indexes
    )
    return (*_contracts(4, 4), *parameters)


def _substitute(expression: str) -> str:
    result = expression
    for side, count in (("l", 6), ("r", 4)):
        prefix = "left" if side == "l" else "right"
        for component in range(count):
            result = result.replace(
                f"{side}{component}",
                f"model::prepared::{prefix}_{component}",
            )
    return result


_WEYL_PAIR_TO_VECTOR_EXPRESSIONS = {
    WEYL_PAIR_TO_VECTOR_A_TEMPLATE: (
        "l0*r0+l1*r1",
        "-l1*r0-l0*r1",
        "1\U0001d456*(-l1*r0+l0*r1)",
        "-l0*r0+l1*r1",
    ),
    WEYL_PAIR_TO_VECTOR_B_TEMPLATE: (
        "l0*r0+l1*r1",
        "l0*r1+l1*r0",
        "1\U0001d456*(-l0*r1+l1*r0)",
        "l0*r0-l1*r1",
    ),
}


def _reversed_contracts(
    canonical_left_components: int,
    canonical_right_components: int,
    *,
    parameter_indexes: tuple[int, ...] = (),
) -> tuple[str, ...]:
    values: list[dict[str, object]] = []
    for role, count, prefix in (
        ("left-current", canonical_right_components, "vector"),
        ("right-current", canonical_left_components, "weyl"),
    ):
        values.extend(
            {
                "component": component,
                "role": role,
                "symbol": f"ufo::prepared::{prefix}_{component}",
            }
            for component in range(count)
        )
    values.extend(
        {
            "component": 0,
            "model_parameter_index": parameter_index,
            "role": "model-parameter",
            "symbol": f"ufo::prepared::parameter_{parameter_index}",
        }
        for parameter_index in parameter_indexes
    )
    return tuple(
        json.dumps(
            item,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for item in values
    )


def _substitute_reversed(expression: str) -> str:
    result = expression
    for component in range(2):
        result = result.replace(
            f"l{component}",
            f"ufo::prepared::weyl_{component}",
        )
    for component in range(4):
        result = result.replace(
            f"r{component}",
            f"ufo::prepared::vector_{component}",
        )
    return result


def _substitute_reversed_shape(
    expression: str,
    canonical_left_components: int,
    canonical_right_components: int,
) -> str:
    result = expression
    for component in range(canonical_left_components):
        result = result.replace(
            f"l{component}",
            f"ufo::prepared::weyl_{component}",
        )
    for component in range(canonical_right_components):
        result = result.replace(
            f"r{component}",
            f"ufo::prepared::vector_{component}",
        )
    return result


def _three_vector_contracts() -> tuple[str, ...]:
    values = [
        {
            "component": component,
            "role": role,
            "symbol": f"model::prepared::{prefix}_{component}",
        }
        for role, prefix in (
            ("left-current", "left"),
            ("right-current", "right"),
            ("left-momentum", "left_momentum"),
            ("right-momentum", "right_momentum"),
        )
        for component in range(4)
    ]
    return tuple(
        json.dumps(
            item,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for item in values
    )


def _substitute_three_vector(expression: str) -> str:
    result = expression
    for symbol, prefix in (
        ("l", "left"),
        ("r", "right"),
        ("p", "left_momentum"),
        ("q", "right_momentum"),
    ):
        for component in range(4):
            result = result.replace(
                f"{symbol}{component}",
                f"model::prepared::{prefix}_{component}",
            )
    return result


def _finalization_contracts(components: int) -> tuple[str, ...]:
    values = [
        {
            "component": component,
            "role": "current",
            "symbol": f"model::prepared::current_{component}",
        }
        for component in range(components)
    ]
    values.extend(
        {
            "component": component,
            "role": "momentum",
            "symbol": f"model::prepared::momentum_{component}",
        }
        for component in range(4)
    )
    return tuple(
        json.dumps(
            item,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for item in values
    )


def _substitute_finalization(expression: str, components: int) -> str:
    result = expression
    for component in range(components):
        result = result.replace(
            f"l{component}",
            f"model::prepared::current_{component}",
        )
    for component in range(4):
        result = result.replace(
            f"p{component}",
            f"model::prepared::momentum_{component}",
        )
    return result


def _massive_finalization_contracts(
    first_parameter_index: int,
    second_parameter_index: int,
) -> tuple[str, ...]:
    values = [json.loads(item) for item in _finalization_contracts(4)]
    values.extend(
        (
            {
                "component": 0,
                "model_parameter_index": first_parameter_index,
                "model_parameter_name": "opaque.alpha",
                "role": "model-parameter",
                "symbol": "model::prepared::alpha",
            },
            {
                "component": 0,
                "model_parameter_index": second_parameter_index,
                "model_parameter_name": "opaque.beta",
                "role": "model-parameter",
                "symbol": "model::prepared::beta",
            },
        )
    )
    return tuple(
        json.dumps(
            item,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for item in values
    )


def _massive_finalization_expressions(
    orientation: str,
    *,
    mass_symbol: str = "model::prepared::alpha",
    width_symbol: str = "model::prepared::beta",
) -> tuple[str, ...]:
    denominator = (
        "(p0^2-p1^2-p2^2-p3^2"
        f"-{mass_symbol}^2+1.00000000000000\U0001d456*"
        f"{mass_symbol}*{width_symbol})^(-1)"
    )
    if orientation == "particle":
        numerators = (
            f"(p0+p3)*l2+(p1+1\U0001d456*p2)*l3+{mass_symbol}*l0",
            f"(p0-p3)*l3+(p1-1\U0001d456*p2)*l2+{mass_symbol}*l1",
            f"(p0-p3)*l0-(p1+1\U0001d456*p2)*l1+{mass_symbol}*l2",
            f"(p0+p3)*l1-(p1-1\U0001d456*p2)*l0+{mass_symbol}*l3",
        )
    else:
        numerators = (
            f"(-p0+p3)*l2+(p1-1\U0001d456*p2)*l3+{mass_symbol}*l0",
            f"(-p0-p3)*l3+(p1+1\U0001d456*p2)*l2+{mass_symbol}*l1",
            f"(-p0-p3)*l0+(-p1+1\U0001d456*p2)*l1+{mass_symbol}*l2",
            f"(-p0+p3)*l1+(-p1-1\U0001d456*p2)*l0+{mass_symbol}*l3",
        )
    return tuple(
        _substitute_finalization(f"1\U0001d456*{denominator}*({numerator})", 4)
        for numerator in numerators
    )


def _massive_vector_finalization_expressions(
    *,
    mass_symbol: str = "model::prepared::alpha",
    width_symbol: str = "model::prepared::beta",
) -> tuple[str, ...]:
    denominator = (
        "(p0^2-p1^2-p2^2-p3^2"
        f"-{mass_symbol}^2+1.00000000000000\U0001d456*"
        f"{mass_symbol}*{width_symbol})^(-1)"
    )
    current_dot_momentum = "(l0*p0-l1*p1-l2*p2-l3*p3)"
    return tuple(
        _substitute_finalization(
            f"-1.00000000000000\U0001d456*{denominator}*"
            f"(l{component}-p{component}*{mass_symbol}^(-2)*"
            f"{current_dot_momentum})",
            4,
        )
        for component in range(4)
    )


@pytest.mark.parametrize(
    ("expressions", "shape", "destination", "expected_template", "expected_scale"),
    (
        (
            (
                "-1𝑖*l0*r3-1𝑖*l1*r1+l1*r2+1𝑖*l0*r0",
                "-l0*r2-1𝑖*l0*r1+1𝑖*l1*r0+1𝑖*l1*r3",
            ),
            (2, 4),
            2,
            "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-a.v1",
            1.0 + 0.0j,
        ),
        (
            (
                "-l1*r2+1𝑖*l0*r0+1𝑖*l0*r3+1𝑖*l1*r1",
                "-1𝑖*l1*r3+l0*r2+1𝑖*l0*r1+1𝑖*l1*r0",
            ),
            (2, 4),
            2,
            "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-b.v1",
            1.0 + 0.0j,
        ),
        (
            (
                "1𝑖/2*(l0*r1+l1*r2+l2*r3)",
                "1𝑖/2*(l0*r0+l3*r2+l4*r3)",
                "1𝑖/2*(-l3*r1+l1*r0+l5*r3)",
                "1𝑖/2*(-l4*r1-l5*r2+l2*r0)",
            ),
            (6, 4),
            4,
            "rusticol.recurrence-intrinsic.antisymmetric-tensor-vector.v1",
            0.0 + 0.5j,
        ),
        (
            (
                "-l1*r0+l0*r1",
                "-l2*r0+l0*r2",
                "-l3*r0+l0*r3",
                "-l2*r1+l1*r2",
                "-l3*r1+l1*r3",
                "-l3*r2+l2*r3",
            ),
            (4, 4),
            6,
            "rusticol.recurrence-intrinsic.vector-wedge-vector.v1",
            1.0 + 0.0j,
        ),
    ),
)
def test_certifies_model_namespace_independent_intrinsics(
    expressions: tuple[str, ...],
    shape: tuple[int, int],
    destination: int,
    expected_template: str,
    expected_scale: complex,
) -> None:
    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(_substitute(item) for item in expressions),
        input_contracts=_contracts(*shape),
        parent_component_counts=shape,
        destination_component_count=destination,
        binding_coupling=None,
    )

    assert result is not None
    assert result.runtime_template == expected_template
    assert result.constant_scale == expected_scale
    assert result.model_parameter_index is None
    assert result.parent_permutation == (0, 1)
    assert result.scale_projection()["kind"] == RECURRENCE_INTRINSIC_SCALE_KIND


@pytest.mark.parametrize(
    ("base", "expected_template"),
    (
        (
            (
                "-1𝑖*l0*r3-1𝑖*l1*r1+l1*r2+1𝑖*l0*r0",
                "-l0*r2-1𝑖*l0*r1+1𝑖*l1*r0+1𝑖*l1*r3",
            ),
            "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-a.v1",
        ),
        (
            (
                "-l1*r2+1𝑖*l0*r0+1𝑖*l0*r3+1𝑖*l1*r1",
                "-1𝑖*l1*r3+l0*r2+1𝑖*l0*r1+1𝑖*l1*r0",
            ),
            "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-b.v1",
        ),
    ),
)
def test_built_in_and_ufo_parent_orders_certify_the_same_weyl_family(
    base: tuple[str, str],
    expected_template: str,
) -> None:
    built_in = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(
            f"-1𝑖*model::prepared::parameter*({_substitute(item)})" for item in base
        ),
        input_contracts=_contracts(2, 4, parameter_index=17),
        parent_component_counts=(2, 4),
        destination_component_count=2,
        binding_coupling=None,
    )
    ufo = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(
            f"-1𝑖*ufo::prepared::parameter_91*({_substitute_reversed(item)})"
            for item in base
        ),
        input_contracts=_reversed_contracts(
            2,
            4,
            parameter_indexes=(91,),
        ),
        parent_component_counts=(4, 2),
        destination_component_count=2,
        binding_coupling=None,
        allow_nontrivial_parent_permutation=True,
    )

    assert built_in is not None
    assert ufo is not None
    assert built_in.runtime_template == expected_template
    assert ufo.runtime_template == expected_template
    assert built_in.contract_digest == ufo.contract_digest
    assert built_in.constant_scale == ufo.constant_scale == -1.0j
    assert built_in.parent_permutation == (0, 1)
    assert ufo.parent_permutation == (1, 0)
    assert built_in.model_parameter_index == 17
    assert ufo.model_parameter_index == 91


def test_reversed_parent_intrinsic_is_fail_closed_without_permutation_support() -> None:
    base = (
        "-l1*r2+1𝑖*l0*r0+1𝑖*l0*r3+1𝑖*l1*r1",
        "-1𝑖*l1*r3+l0*r2+1𝑖*l0*r1+1𝑖*l1*r0",
    )

    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(_substitute_reversed(item) for item in base),
        input_contracts=_reversed_contracts(2, 4),
        parent_component_counts=(4, 2),
        destination_component_count=2,
        binding_coupling=None,
    )

    assert result is None


def test_rejects_reversed_weyl_family_with_two_independent_parameters() -> None:
    base = (
        "-l1*r2+1𝑖*l0*r0+1𝑖*l0*r3+1𝑖*l1*r1",
        "-1𝑖*l1*r3+l0*r2+1𝑖*l0*r1+1𝑖*l1*r0",
    )
    scale = "(ufo::prepared::parameter_91+ufo::prepared::parameter_92)"

    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(
            f"{scale}*({_substitute_reversed(item)})" for item in base
        ),
        input_contracts=_reversed_contracts(
            2,
            4,
            parameter_indexes=(91, 92),
        ),
        parent_component_counts=(4, 2),
        destination_component_count=2,
        binding_coupling=None,
        allow_nontrivial_parent_permutation=True,
    )

    assert result is None


def test_rejects_noncontiguous_reversed_parent_basis() -> None:
    contracts = [json.loads(item) for item in _reversed_contracts(2, 4)]
    contracts = [
        item
        for item in contracts
        if not (item["role"] == "left-current" and item["component"] == 2)
    ]

    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=(
            _substitute_reversed("-l1*r2+1𝑖*l0*r0+1𝑖*l0*r3+1𝑖*l1*r1"),
            _substitute_reversed("-1𝑖*l1*r3+l0*r2+1𝑖*l0*r1+1𝑖*l1*r0"),
        ),
        input_contracts=tuple(
            json.dumps(
                item,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            for item in contracts
        ),
        parent_component_counts=(4, 2),
        destination_component_count=2,
        binding_coupling=None,
        allow_nontrivial_parent_permutation=True,
    )

    assert result is None


@pytest.mark.parametrize(
    "runtime_template",
    (WEYL_PAIR_TO_VECTOR_A_TEMPLATE, WEYL_PAIR_TO_VECTOR_B_TEMPLATE),
)
def test_certifies_weyl_pair_to_vector_from_algebra_and_shape(
    runtime_template: str,
) -> None:
    expressions = _WEYL_PAIR_TO_VECTOR_EXPRESSIONS[runtime_template]
    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(
            _substitute(f"7.07106781186547e-1\U0001d456*({item})")
            for item in expressions
        ),
        input_contracts=_contracts(2, 2),
        parent_component_counts=(2, 2),
        destination_component_count=4,
        binding_coupling=None,
        factored_output_parameter_index=73,
        allow_nontrivial_parent_permutation=True,
    )

    assert result is not None
    assert result.runtime_template == runtime_template
    assert result.constant_scale == 0.0 + 0.707106781186547j
    assert result.model_parameter_index == 73
    assert result.parent_permutation == (0, 1)
    assert result.scale_projection() == {
        "constant_imag_bits": 4604544271217802184,
        "constant_real_bits": 0,
        "kind": RECURRENCE_INTRINSIC_SCALE_KIND,
        "parameter_index": 73,
    }


@pytest.mark.parametrize(
    "runtime_template",
    (WEYL_PAIR_TO_VECTOR_A_TEMPLATE, WEYL_PAIR_TO_VECTOR_B_TEMPLATE),
)
def test_weyl_pair_to_vector_preserves_authenticated_parent_permutation(
    runtime_template: str,
) -> None:
    expressions = _WEYL_PAIR_TO_VECTOR_EXPRESSIONS[runtime_template]
    swapped = tuple(
        _substitute(item.replace("l", "x").replace("r", "l").replace("x", "r"))
        for item in expressions
    )

    assert (
        certify_recurrence_contribution_intrinsic(
            exact_expressions=swapped,
            input_contracts=_contracts(2, 2),
            parent_component_counts=(2, 2),
            destination_component_count=4,
            binding_coupling=None,
        )
        is None
    )
    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=swapped,
        input_contracts=_contracts(2, 2),
        parent_component_counts=(2, 2),
        destination_component_count=4,
        binding_coupling=None,
        allow_nontrivial_parent_permutation=True,
    )
    assert result is not None
    assert result.runtime_template == runtime_template
    assert result.parent_permutation == (1, 0)


@pytest.mark.parametrize(
    "mutation",
    ("coefficient", "chirality-orientation", "shape", "ambiguous-coupling"),
)
def test_weyl_pair_to_vector_rejects_contract_drift(mutation: str) -> None:
    expressions = list(_WEYL_PAIR_TO_VECTOR_EXPRESSIONS[WEYL_PAIR_TO_VECTOR_A_TEMPLATE])
    contracts = _contracts(2, 2)
    parent_shape = (2, 2)
    factored_slot = None
    if mutation == "coefficient":
        expressions[0] = expressions[0].replace("l1*r1", "2*l1*r1")
    elif mutation == "chirality-orientation":
        expressions[1] = _WEYL_PAIR_TO_VECTOR_EXPRESSIONS[
            WEYL_PAIR_TO_VECTOR_B_TEMPLATE
        ][1]
    elif mutation == "shape":
        contracts = _contracts(2, 3)
        parent_shape = (2, 3)
    else:
        contracts = _contracts(2, 2, parameter_index=19)
        expressions = [f"model::prepared::parameter*({item})" for item in expressions]
        factored_slot = 73

    assert (
        certify_recurrence_contribution_intrinsic(
            exact_expressions=tuple(_substitute(item) for item in expressions),
            input_contracts=contracts,
            parent_component_counts=parent_shape,
            destination_component_count=4,
            binding_coupling=None,
            factored_output_parameter_index=factored_slot,
            allow_nontrivial_parent_permutation=True,
        )
        is None
    )


def test_certifies_one_model_parameter_scale() -> None:
    base = (
        "-l1*r2+1𝑖*l0*r0+1𝑖*l0*r3+1𝑖*l1*r1",
        "-1𝑖*l1*r3+l0*r2+1𝑖*l0*r1+1𝑖*l1*r0",
    )
    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(
            f"-2*model::prepared::parameter*({_substitute(item)})" for item in base
        ),
        input_contracts=_contracts(2, 4, parameter_index=17),
        parent_component_counts=(2, 4),
        destination_component_count=2,
        binding_coupling=None,
    )

    assert result is not None
    assert result.constant_scale == -2.0 + 0.0j
    assert result.model_parameter_index == 17


@pytest.mark.parametrize("parameter_index", (None, 51))
def test_certifies_scalar_product_with_exact_prepared_parameter_owner(
    parameter_index: int | None,
) -> None:
    scale = "" if parameter_index is None else "model::prepared::parameter*"
    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=(
            f"{scale}model::prepared::left_0*model::prepared::right_0",
        ),
        input_contracts=_contracts(1, 1, parameter_index=parameter_index),
        parent_component_counts=(1, 1),
        destination_component_count=1,
        binding_coupling=None,
    )

    assert result is not None
    assert (
        result.runtime_template
        == "rusticol.recurrence-intrinsic.scalar-product.v1"
    )
    assert result.constant_scale == 1.0 + 0.0j
    assert result.model_parameter_index == parameter_index
    assert result.parent_permutation == (0, 1)


def test_certifies_color_ordered_three_vector_with_momentum_operands() -> None:
    dot_lr = "(l0*r0-l1*r1-l2*r2-l3*r3)"
    dot_lq = "(l0*q0-l1*q1-l2*q2-l3*q3)"
    dot_rp = "(r0*p0-r1*p1-r2*p2-r3*p3)"
    expressions = tuple(
        _substitute_three_vector(
            f"3𝑖/2*({dot_lr}*(p{component}-q{component})"
            f"+2*({dot_lq}*r{component}-{dot_rp}*l{component}))"
        )
        for component in range(4)
    )

    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=expressions,
        input_contracts=_three_vector_contracts(),
        parent_component_counts=(4, 4),
        destination_component_count=4,
        binding_coupling=None,
    )

    assert result is not None
    assert (
        result.runtime_template
        == "rusticol.recurrence-intrinsic.color-ordered-three-vector.v1"
    )
    assert result.constant_scale == 0.0 + 1.5j
    assert result.model_parameter_index is None


def test_certifies_binary64_scaled_three_vector_without_tolerance() -> None:
    dot_lr = "(l0*r0-l1*r1-l2*r2-l3*r3)"
    dot_lq = "(l0*q0-l1*q1-l2*q2-l3*q3)"
    dot_rp = "(r0*p0-r1*p1-r2*p2-r3*p3)"
    scale = "7.07106781186547e-1𝑖"
    expressions = tuple(
        _substitute_three_vector(
            f"{scale}*({dot_lr}*(p{component}-q{component})"
            f"+2*({dot_lq}*r{component}-{dot_rp}*l{component}))"
        )
        for component in range(4)
    )

    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=expressions,
        input_contracts=_three_vector_contracts(),
        parent_component_counts=(4, 4),
        destination_component_count=4,
        binding_coupling=None,
    )
    assert result is not None
    assert result.constant_scale == 0.0 + 0.707106781186547j

    perturbed = (
        expressions[0].replace(scale, "7.07106781186548e-1𝑖"),
        *expressions[1:],
    )
    assert (
        certify_recurrence_contribution_intrinsic(
            exact_expressions=perturbed,
            input_contracts=_three_vector_contracts(),
            parent_component_counts=(4, 4),
            destination_component_count=4,
            binding_coupling=None,
        )
        is None
    )


def test_substitutes_exact_binding_coupling_before_certification() -> None:
    base = (
        "-l1*r2+1𝑖*l0*r0+1𝑖*l0*r3+1𝑖*l1*r1",
        "-1𝑖*l1*r3+l0*r2+1𝑖*l0*r1+1𝑖*l1*r0",
    )
    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(
            f"model::prepared::coupling_re*({_substitute(item)})" for item in base
        ),
        input_contracts=_contracts(2, 4, coupling=True),
        parent_component_counts=(2, 4),
        destination_component_count=2,
        binding_coupling=ExactComplexRationalV1.from_fractions(
            Fraction(3, 2), Fraction(0)
        ),
    )

    assert result is not None
    assert result.constant_scale == 1.5 + 0.0j


def test_rejects_near_match_with_extra_tensor_term() -> None:
    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=(
            _substitute("-l1*r2+1𝑖*l0*r0+1𝑖*l0*r3+1𝑖*l1*r1+l0*r2"),
            _substitute("-1𝑖*l1*r3+l0*r2+1𝑖*l0*r1+1𝑖*l1*r0"),
        ),
        input_contracts=_contracts(2, 4),
        parent_component_counts=(2, 4),
        destination_component_count=2,
        binding_coupling=None,
    )

    assert result is None


@pytest.mark.parametrize(
    ("orientation", "expressions", "expected_template"),
    (
        (
            "particle",
            (
                "(r0+r3)*l2+(r1+1\U0001d456*r2)*l3",
                "(r0-r3)*l3+(r1-1\U0001d456*r2)*l2",
                "(r0-r3)*l0-(r1+1\U0001d456*r2)*l1",
                "(r0+r3)*l1-(r1-1\U0001d456*r2)*l0",
            ),
            DIRAC_VECTOR_PARTICLE_TEMPLATE,
        ),
        (
            "antiparticle",
            (
                "(-r0+r3)*l2+(r1-1\U0001d456*r2)*l3",
                "(-r0-r3)*l3+(r1+1\U0001d456*r2)*l2",
                "(-r0-r3)*l0+(-r1+1\U0001d456*r2)*l1",
                "(-r0+r3)*l1+(-r1-1\U0001d456*r2)*l0",
            ),
            DIRAC_VECTOR_ANTIPARTICLE_TEMPLATE,
        ),
    ),
)
def test_certifies_both_oriented_dirac_vector_transitions(
    orientation: str,
    expressions: tuple[str, ...],
    expected_template: str,
) -> None:
    del orientation
    scale = "7.07106781186547e-1\U0001d456"
    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(
            _substitute(f"{scale}*({expression})") for expression in expressions
        ),
        input_contracts=_contracts(4, 4),
        parent_component_counts=(4, 4),
        destination_component_count=4,
        binding_coupling=None,
    )

    assert result is not None
    assert result.runtime_template == expected_template
    assert result.constant_scale == 0.0 + 0.707106781186547j
    assert result.model_parameter_index is None
    assert result.parent_permutation == (0, 1)


@pytest.mark.parametrize(
    ("orientation", "expressions", "expected_template"),
    (
        (
            "particle",
            (
                "(r0+r3)*l2+(r1+1\U0001d456*r2)*l3",
                "(r0-r3)*l3+(r1-1\U0001d456*r2)*l2",
                "(r0-r3)*l0-(r1+1\U0001d456*r2)*l1",
                "(r0+r3)*l1-(r1-1\U0001d456*r2)*l0",
            ),
            CHIRAL_DIRAC_VECTOR_PARTICLE_TEMPLATE,
        ),
        (
            "antiparticle",
            (
                "(-r0+r3)*l2+(r1-1\U0001d456*r2)*l3",
                "(-r0-r3)*l3+(r1+1\U0001d456*r2)*l2",
                "(-r0-r3)*l0+(-r1+1\U0001d456*r2)*l1",
                "(-r0+r3)*l1+(-r1-1\U0001d456*r2)*l0",
            ),
            CHIRAL_DIRAC_VECTOR_ANTIPARTICLE_TEMPLATE,
        ),
    ),
)
def test_certifies_independent_chiral_dirac_vector_scale_owners(
    orientation: str,
    expressions: tuple[str, ...],
    expected_template: str,
) -> None:
    left_scale = "2\U0001d456*model::prepared::parameter_17"
    right_scale = "-3\U0001d456*model::prepared::parameter_19"
    # Coupling component ownership is part of the orientation witness:
    # particle left/right own upper/lower, while antiparticle swaps them.
    upper_scale, lower_scale = (
        (left_scale, right_scale)
        if orientation == "particle"
        else (right_scale, left_scale)
    )
    exact = tuple(
        _substitute(f"{upper_scale if component < 2 else lower_scale}*({value})")
        for component, value in enumerate(expressions)
    )

    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=exact,
        input_contracts=_dirac_vector_contracts_with_parameters(17, 19),
        parent_component_counts=(4, 4),
        destination_component_count=4,
        binding_coupling=None,
    )

    assert isinstance(result, CertifiedChiralDiracVectorIntrinsic)
    assert result.runtime_template == expected_template
    assert result.orientation == orientation
    assert result.left_constant_scale == 0.0 + 2.0j
    assert result.left_model_parameter_index == 17
    assert result.right_constant_scale == 0.0 - 3.0j
    assert result.right_model_parameter_index == 19
    assert result.scale_projection() == {
        "kind": "chiral-dirac-vector-scales-v1",
        "left_scale": {
            "constant_imag_bits": 4611686018427387904,
            "constant_real_bits": 0,
            "kind": "intrinsic-scale-v1",
            "parameter_index": 17,
        },
        "orientation": orientation,
        "right_scale": {
            "constant_imag_bits": 13837309855095848960,
            "constant_real_bits": 0,
            "kind": "intrinsic-scale-v1",
            "parameter_index": 19,
        },
    }


def test_chiral_dirac_vector_allows_one_exact_zero_half_and_rejects_drift() -> None:
    particle = (
        "(r0+r3)*l2+(r1+1\U0001d456*r2)*l3",
        "(r0-r3)*l3+(r1-1\U0001d456*r2)*l2",
        "(r0-r3)*l0-(r1+1\U0001d456*r2)*l1",
        "(r0+r3)*l1-(r1-1\U0001d456*r2)*l0",
    )
    pure_left = tuple(
        _substitute(f"2\U0001d456*({value})") if component < 2 else "0"
        for component, value in enumerate(particle)
    )
    certified = certify_recurrence_contribution_intrinsic(
        exact_expressions=pure_left,
        input_contracts=_contracts(4, 4),
        parent_component_counts=(4, 4),
        destination_component_count=4,
        binding_coupling=None,
    )

    assert isinstance(certified, CertifiedChiralDiracVectorIntrinsic)
    assert certified.left_constant_scale == 2.0j
    assert certified.right_constant_scale == 0.0
    assert certified.right_model_parameter_index is None

    drifted = (
        f"{pure_left[0]}+model::prepared::left_0*model::prepared::right_0",
        *pure_left[1:],
    )
    assert (
        certify_recurrence_contribution_intrinsic(
            exact_expressions=drifted,
            input_contracts=_contracts(4, 4),
            parent_component_counts=(4, 4),
            destination_component_count=4,
            binding_coupling=None,
        )
        is None
    )


def test_dirac_vector_transition_accepts_only_authenticated_parent_permutation() -> (
    None
):
    expressions = (
        "(r0+r3)*l2+(r1+1\U0001d456*r2)*l3",
        "(r0-r3)*l3+(r1-1\U0001d456*r2)*l2",
        "(r0-r3)*l0-(r1+1\U0001d456*r2)*l1",
        "(r0+r3)*l1-(r1-1\U0001d456*r2)*l0",
    )
    prepared = tuple(
        _substitute_reversed_shape(f"1\U0001d456*({item})", 4, 4)
        for item in expressions
    )

    assert (
        certify_recurrence_contribution_intrinsic(
            exact_expressions=prepared,
            input_contracts=_reversed_contracts(4, 4),
            parent_component_counts=(4, 4),
            destination_component_count=4,
            binding_coupling=None,
        )
        is None
    )
    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=prepared,
        input_contracts=_reversed_contracts(4, 4),
        parent_component_counts=(4, 4),
        destination_component_count=4,
        binding_coupling=None,
        allow_nontrivial_parent_permutation=True,
    )
    assert result is not None
    assert result.runtime_template == DIRAC_VECTOR_PARTICLE_TEMPLATE
    assert result.parent_permutation == (1, 0)


def test_certifies_dirac_scalar_transition_without_particle_identity() -> None:
    expressions = tuple(
        _substitute(f"-7.07106781186547e-1\U0001d456*l{component}*r0")
        for component in range(4)
    )
    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=expressions,
        input_contracts=_contracts(4, 1),
        parent_component_counts=(4, 1),
        destination_component_count=4,
        binding_coupling=None,
    )

    assert result is not None
    assert result.runtime_template == DIRAC_SCALAR_TO_DIRAC_TEMPLATE
    assert result.constant_scale == 0.0 - 0.707106781186547j
    assert result.model_parameter_index is None

    perturbed = (f"{expressions[0]}+model::prepared::left_1", *expressions[1:])
    assert (
        certify_recurrence_contribution_intrinsic(
            exact_expressions=perturbed,
            input_contracts=_contracts(4, 1),
            parent_component_counts=(4, 1),
            destination_component_count=4,
            binding_coupling=None,
        )
        is None
    )


def test_dirac_scalar_transition_retains_factored_dynamic_coupling_slot() -> None:
    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=tuple(
            _substitute(f"-7.07106781186547e-1\U0001d456*l{component}*r0")
            for component in range(4)
        ),
        input_contracts=_contracts(4, 1),
        parent_component_counts=(4, 1),
        destination_component_count=4,
        binding_coupling=None,
        factored_output_parameter_index=73,
    )

    assert result is not None
    assert result.runtime_template == DIRAC_SCALAR_TO_DIRAC_TEMPLATE
    assert result.constant_scale == 0.0 - 0.707106781186547j
    assert result.model_parameter_index == 73
    assert result.scale_projection() == {
        "constant_imag_bits": 13827916308072577992,
        "constant_real_bits": 0,
        "kind": RECURRENCE_INTRINSIC_SCALE_KIND,
        "parameter_index": 73,
    }
    assert (
        certify_recurrence_contribution_intrinsic(
            exact_expressions=tuple(
                _substitute(f"-1\U0001d456*l{component}*r0") for component in range(4)
            ),
            input_contracts=_contracts(4, 1, parameter_index=12),
            parent_component_counts=(4, 1),
            destination_component_count=4,
            binding_coupling=None,
            factored_output_parameter_index=73,
        )
        is None
    )


def test_reversed_dirac_scalar_transition_records_prepared_parent_order() -> None:
    prepared = tuple(
        _substitute_reversed_shape(f"-1\U0001d456*l{component}*r0", 4, 1)
        for component in range(4)
    )
    result = certify_recurrence_contribution_intrinsic(
        exact_expressions=prepared,
        input_contracts=_reversed_contracts(4, 1),
        parent_component_counts=(1, 4),
        destination_component_count=4,
        binding_coupling=None,
        allow_nontrivial_parent_permutation=True,
    )

    assert result is not None
    assert result.runtime_template == DIRAC_SCALAR_TO_DIRAC_TEMPLATE
    assert result.parent_permutation == (1, 0)


@pytest.mark.parametrize(
    ("expressions", "components", "expected_template"),
    (
        (
            (
                "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)"
                "*(-1𝑖*l0*p3-1𝑖*l1*p1+l1*p2+1𝑖*l0*p0)",
                "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)"
                "*(-l0*p2-1𝑖*l0*p1+1𝑖*l1*p0+1𝑖*l1*p3)",
            ),
            2,
            "rusticol.recurrence-intrinsic.weyl-propagator-a.v1",
        ),
        (
            (
                "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)"
                "*(-l1*p2+1𝑖*l0*p0+1𝑖*l0*p3+1𝑖*l1*p1)",
                "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)"
                "*(-1𝑖*l1*p3+l0*p2+1𝑖*l0*p1+1𝑖*l1*p0)",
            ),
            2,
            "rusticol.recurrence-intrinsic.weyl-propagator-b.v1",
        ),
        (
            (
                "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)*-1𝑖*l0",
                "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)*-1𝑖*l1",
                "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)*-1𝑖*l2",
                "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)*-1𝑖*l3",
            ),
            4,
            "rusticol.recurrence-intrinsic.vector-propagator-feynman.v1",
        ),
    ),
)
def test_certifies_finalization_intrinsics(
    expressions: tuple[str, ...], components: int, expected_template: str
) -> None:
    result = certify_recurrence_finalization_intrinsic(
        exact_expressions=tuple(
            _substitute_finalization(item, components) for item in expressions
        ),
        input_contracts=_finalization_contracts(components),
        component_count=components,
    )

    assert result is not None
    assert result.runtime_template == expected_template
    assert result.constant_scale == (
        -1.0j if expected_template.endswith("vector-propagator-feynman.v1") else 1.0
    )
    assert len(result.contract_digest) == 64


def test_certifies_binary64_unit_weyl_finalizer_without_tolerance() -> None:
    denominator = "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)"
    unit = "1.00000000000000"
    expressions = (
        f"{denominator}*(-{unit}*l1*p2+{unit}𝑖*l0*p0"
        f"+{unit}𝑖*l0*p3+{unit}𝑖*l1*p1)",
        f"{denominator}*(-{unit}𝑖*l1*p3+{unit}*l0*p2"
        f"+{unit}𝑖*l0*p1+{unit}𝑖*l1*p0)",
    )
    prepared = tuple(_substitute_finalization(item, 2) for item in expressions)

    result = certify_recurrence_finalization_intrinsic(
        exact_expressions=prepared,
        input_contracts=_finalization_contracts(2),
        component_count=2,
    )
    assert result is not None
    assert result.runtime_template.endswith("weyl-propagator-b.v1")

    perturbed = (prepared[0].replace(unit, "1.00000000000001", 1), prepared[1])
    assert (
        certify_recurrence_finalization_intrinsic(
            exact_expressions=perturbed,
            input_contracts=_finalization_contracts(2),
            component_count=2,
        )
        is None
    )


def test_rejects_finalization_intrinsic_with_wrong_global_scale() -> None:
    expressions = tuple(
        f"2*({_substitute_finalization(item, 4)})"
        for item in (
            "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)*-1𝑖*l0",
            "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)*-1𝑖*l1",
            "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)*-1𝑖*l2",
            "(-1*p1^2+-1*p2^2+-1*p3^2+p0^2)^(-1)*-1𝑖*l3",
        )
    )

    assert (
        certify_recurrence_finalization_intrinsic(
            exact_expressions=expressions,
            input_contracts=_finalization_contracts(4),
            component_count=4,
        )
        is None
    )


@pytest.mark.parametrize(
    ("orientation", "expected_template"),
    (
        ("particle", MASSIVE_DIRAC_PARTICLE_TEMPLATE),
        ("antiparticle", MASSIVE_DIRAC_ANTIPARTICLE_TEMPLATE),
    ),
)
def test_certifies_massive_dirac_finalizer_and_discovers_opaque_parameter_roles(
    orientation: str,
    expected_template: str,
) -> None:
    contracts = list(_massive_finalization_contracts(41, 9))
    contracts[-2:] = reversed(contracts[-2:])
    result = certify_recurrence_finalization_intrinsic(
        exact_expressions=_massive_finalization_expressions(orientation),
        input_contracts=tuple(contracts),
        component_count=4,
    )

    assert result is not None
    assert result.runtime_template == expected_template
    assert result.constant_scale == 1.0j
    assert result.orientation == orientation
    assert result.mass_parameter_index == 41
    assert result.width_parameter_index == 9
    assert result.scale_projection() == {
        "constant_imag_bits": 4607182418800017408,
        "constant_real_bits": 0,
        "kind": RECURRENCE_MASSIVE_DIRAC_FINALIZER_KIND,
        "mass_parameter_index": 41,
        "orientation": orientation,
        "width_parameter_index": 9,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "numerator-sign",
        "mass-sign",
        "width-sign",
        "denominator-sign",
        "extra-parameter",
        "duplicate-parameter-index",
    ),
)
def test_massive_dirac_finalizer_fails_closed_on_algebra_and_parameter_drift(
    mutation: str,
) -> None:
    expressions = list(_massive_finalization_expressions("particle"))
    contracts = list(_massive_finalization_contracts(41, 9))
    if mutation == "numerator-sign":
        expressions[0] = expressions[0].replace(
            "+model::prepared::momentum_3",
            "-model::prepared::momentum_3",
            1,
        )
    elif mutation == "mass-sign":
        expressions[0] = expressions[0].replace(
            "+model::prepared::alpha*model::prepared::current_0",
            "-model::prepared::alpha*model::prepared::current_0",
            1,
        )
    elif mutation == "width-sign":
        expressions = [
            item.replace(
                "+1.00000000000000\U0001d456*model::prepared::alpha*"
                "model::prepared::beta",
                "-1.00000000000000\U0001d456*model::prepared::alpha*"
                "model::prepared::beta",
            )
            for item in expressions
        ]
    elif mutation == "denominator-sign":
        expressions = [
            item.replace(
                "-model::prepared::momentum_1^2",
                "+model::prepared::momentum_1^2",
            )
            for item in expressions
        ]
    elif mutation == "extra-parameter":
        contracts.append(
            json.dumps(
                {
                    "component": 0,
                    "model_parameter_index": 77,
                    "model_parameter_name": "opaque.gamma",
                    "role": "model-parameter",
                    "symbol": "model::prepared::gamma",
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    else:
        second = json.loads(contracts[-1])
        second["model_parameter_index"] = 41
        contracts[-1] = json.dumps(second, separators=(",", ":"), sort_keys=True)

    assert (
        certify_recurrence_finalization_intrinsic(
            exact_expressions=tuple(expressions),
            input_contracts=tuple(contracts),
            component_count=4,
        )
        is None
    )


def test_massive_dirac_finalizer_rejects_wrong_orientation_formula() -> None:
    particle = certify_recurrence_finalization_intrinsic(
        exact_expressions=_massive_finalization_expressions("particle"),
        input_contracts=_massive_finalization_contracts(3, 8),
        component_count=4,
    )
    antiparticle = certify_recurrence_finalization_intrinsic(
        exact_expressions=_massive_finalization_expressions("antiparticle"),
        input_contracts=_massive_finalization_contracts(3, 8),
        component_count=4,
    )

    assert particle is not None and antiparticle is not None
    assert particle.orientation == "particle"
    assert antiparticle.orientation == "antiparticle"
    mixed = list(_massive_finalization_expressions("particle"))
    mixed[0] = _massive_finalization_expressions("antiparticle")[0]
    assert (
        certify_recurrence_finalization_intrinsic(
            exact_expressions=tuple(mixed),
            input_contracts=_massive_finalization_contracts(3, 8),
            component_count=4,
        )
        is None
    )


def test_certifies_massive_vector_unitary_finalizer_and_discovers_parameter_roles() -> (
    None
):
    contracts = list(_massive_finalization_contracts(41, 9))
    contracts[-2:] = reversed(contracts[-2:])
    result = certify_recurrence_finalization_intrinsic(
        exact_expressions=_massive_vector_finalization_expressions(),
        input_contracts=tuple(contracts),
        component_count=4,
    )

    assert result is not None
    assert result.runtime_template == MASSIVE_VECTOR_UNITARY_TEMPLATE
    assert result.constant_scale == -1.0j
    assert result.orientation is None
    assert result.mass_parameter_index == 41
    assert result.width_parameter_index == 9
    assert result.scale_projection() == {
        "constant_imag_bits": 13830554455654793216,
        "constant_real_bits": 0,
        "kind": RECURRENCE_MASSIVE_VECTOR_FINALIZER_KIND,
        "mass_parameter_index": 41,
        "width_parameter_index": 9,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "longitudinal-sign",
        "dot-metric",
        "mass-power",
        "width-sign",
        "overall-scale",
        "duplicate-parameter-index",
    ),
)
def test_massive_vector_unitary_finalizer_fails_closed_on_contract_drift(
    mutation: str,
) -> None:
    expressions = list(_massive_vector_finalization_expressions())
    contracts = list(_massive_finalization_contracts(41, 9))
    if mutation == "longitudinal-sign":
        expressions[0] = expressions[0].replace(
            "current_0-model::prepared::momentum_0",
            "current_0+model::prepared::momentum_0",
            1,
        )
    elif mutation == "dot-metric":
        expressions = [
            item.replace(
                "current_0*model::prepared::momentum_0-"
                "model::prepared::current_1*model::prepared::momentum_1",
                "current_0*model::prepared::momentum_0+"
                "model::prepared::current_1*model::prepared::momentum_1",
            )
            for item in expressions
        ]
    elif mutation == "mass-power":
        expressions = [
            item.replace("model::prepared::alpha^(-2)", "model::prepared::alpha^(-1)")
            for item in expressions
        ]
    elif mutation == "width-sign":
        expressions = [
            item.replace(
                "+1.00000000000000\U0001d456*model::prepared::alpha*"
                "model::prepared::beta",
                "-1.00000000000000\U0001d456*model::prepared::alpha*"
                "model::prepared::beta",
            )
            for item in expressions
        ]
    elif mutation == "overall-scale":
        expressions = [
            item.replace("-1.00000000000000\U0001d456", "1.00000000000000\U0001d456", 1)
            for item in expressions
        ]
    else:
        second = json.loads(contracts[-1])
        second["model_parameter_index"] = 41
        contracts[-1] = json.dumps(second, separators=(",", ":"), sort_keys=True)

    assert (
        certify_recurrence_finalization_intrinsic(
            exact_expressions=tuple(expressions),
            input_contracts=tuple(contracts),
            component_count=4,
        )
        is None
    )
