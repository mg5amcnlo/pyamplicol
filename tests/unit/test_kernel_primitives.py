# SPDX-License-Identifier: 0BSD
# ruff: noqa: RUF001
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from pyamplicol.models.builtin.model import BuiltinSMModel
from pyamplicol.models.kernel_primitives import (
    KernelPrimitiveKind,
    SpinorAlgebraCertificationError,
    _certify_spinor_catalog,
    certify_contribution_kernel_primitive,
)
from pyamplicol.models.prepared_catalog import (
    PreparedKernelCatalog,
    build_prepared_kernel_catalog,
)


@pytest.fixture(scope="module")
def builtin_qcd_catalog() -> tuple[BuiltinSMModel, PreparedKernelCatalog]:
    model = BuiltinSMModel()
    return model, build_prepared_kernel_catalog(model)


def test_shared_classifier_uses_exact_algebra_not_model_names() -> None:
    contracts = tuple(
        json.dumps(
            {
                "component": component,
                "role": role,
                "symbol": f"arbitrary::model::{side}_{component}",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        for role, side, count in (
            ("left-current", "fermion", 2),
            ("right-current", "gauge_field", 4),
        )
        for component in range(count)
    )
    expressions = (
        "-1𝑖*l0*r3-1𝑖*l1*r1+l1*r2+1𝑖*l0*r0",
        "-l0*r2-1𝑖*l0*r1+1𝑖*l1*r0+1𝑖*l1*r3",
    )
    for component in range(2):
        expressions = tuple(
            value.replace(f"l{component}", f"arbitrary::model::fermion_{component}")
            for value in expressions
        )
    for component in range(4):
        expressions = tuple(
            value.replace(f"r{component}", f"arbitrary::model::gauge_field_{component}")
            for value in expressions
        )

    certified = certify_contribution_kernel_primitive(
        exact_expressions=expressions,
        input_contracts=contracts,
        parent_component_counts=(2, 4),
        destination_component_count=2,
        binding_coupling=None,
    )

    assert certified is not None
    assert certified.kind is KernelPrimitiveKind.WEYL_VECTOR_TO_WEYL_A
    assert certified.constant_scale == 1.0 + 0.0j
    assert certified.model_parameter_index is None
    assert certified.parent_permutation == (0, 1)


def test_spinor_qcd_certificate_is_exact_and_process_scoped(
    builtin_qcd_catalog: tuple[BuiltinSMModel, PreparedKernelCatalog],
) -> None:
    model, catalog = builtin_qcd_catalog

    two_gluon_quark = _certify_spinor_catalog(
        model,
        catalog,
        process_family="single-massless-quark-line",
        gluon_count=2,
        quark_pdg=2,
    )
    assert two_gluon_quark.primitives == (
        KernelPrimitiveKind.COLOR_ORDERED_THREE_VECTOR,
        KernelPrimitiveKind.FEYNMAN_VECTOR_PROPAGATOR,
        KernelPrimitiveKind.WEYL_VECTOR_TO_WEYL_B,
        KernelPrimitiveKind.WEYL_VECTOR_TO_WEYL_A,
    )

    three_gluon_quark = _certify_spinor_catalog(
        model,
        catalog,
        process_family="single-massless-quark-line",
        gluon_count=3,
        quark_pdg=2,
    )
    assert KernelPrimitiveKind.VECTOR_WEDGE_VECTOR in three_gluon_quark.primitives
    assert (
        three_gluon_quark.primitives.count(
            KernelPrimitiveKind.ANTISYMMETRIC_TENSOR_VECTOR
        )
        == 2
    )


def test_massive_quark_certificate_authenticates_dirac_qg_and_propagator(
    builtin_qcd_catalog: tuple[BuiltinSMModel, PreparedKernelCatalog],
) -> None:
    model, catalog = builtin_qcd_catalog
    parameters = (
        ("particle.6.mass", model.mass(6)),
        ("particle.6.width", model.width(6)),
    )

    certificate = _certify_spinor_catalog(
        model,
        catalog,
        process_family="single-massive-quark-line",
        gluon_count=2,
        quark_pdg=6,
        spinor_parameters=parameters,
    )

    assert certificate.primitives == (
        KernelPrimitiveKind.COLOR_ORDERED_THREE_VECTOR,
        KernelPrimitiveKind.FEYNMAN_VECTOR_PROPAGATOR,
        KernelPrimitiveKind.DIRAC_VECTOR_TO_DIRAC,
        KernelPrimitiveKind.DIRAC_PAIR_TO_VECTOR,
        KernelPrimitiveKind.MASSIVE_DIRAC_PROPAGATOR,
    )
    assert certificate.spinor_parameter_names == (
        "particle.6.mass",
        "particle.6.width",
    )

    binding = next(
        item
        for item in catalog.vertex_bindings
        if item.key.particles == (6, 21, 6)
        and item.key.left_chirality == 0
        and item.key.right_chirality == 0
        and item.key.result_chirality == 0
    )
    kernel = catalog.by_id[binding.kernel_id]
    perturbed = replace(
        kernel,
        exact_expressions=(
            f"({kernel.exact_expressions[0]})+1",
            *kernel.exact_expressions[1:],
        ),
    )
    perturbed_catalog = replace(
        catalog,
        kernels=tuple(
            perturbed if item.kernel_id == kernel.kernel_id else item
            for item in catalog.kernels
        ),
    )
    with pytest.raises(
        SpinorAlgebraCertificationError,
        match=r"massive quark-gluon current.*certified Dirac-vector algebra",
    ):
        _certify_spinor_catalog(
            model,
            perturbed_catalog,
            process_family="single-massive-quark-line",
            gluon_count=2,
            quark_pdg=6,
            spinor_parameters=parameters,
        )

    closure_binding = next(
        item
        for item in catalog.vertex_bindings
        if item.key.particles == (-6, 6, 21)
        and item.key.left_chirality == 0
        and item.key.right_chirality == 0
        and item.key.result_chirality == 0
    )
    closure_kernel = catalog.by_id[closure_binding.kernel_id]
    perturbed_closure = replace(
        closure_kernel,
        exact_expressions=(
            f"({closure_kernel.exact_expressions[0]})+1",
            *closure_kernel.exact_expressions[1:],
        ),
    )
    perturbed_closure_catalog = replace(
        catalog,
        kernels=tuple(
            perturbed_closure
            if item.kernel_id == closure_kernel.kernel_id
            else item
            for item in catalog.kernels
        ),
    )
    with pytest.raises(
        SpinorAlgebraCertificationError,
        match=r"massive Dirac-vector closure.*qbar-gamma-vector-q bilinear",
    ):
        _certify_spinor_catalog(
            model,
            perturbed_closure_catalog,
            process_family="single-massive-quark-line",
            gluon_count=2,
            quark_pdg=6,
            spinor_parameters=parameters,
        )


def test_spinor_z_certificate_authenticates_chiral_kernels_and_parameters(
    builtin_qcd_catalog: tuple[BuiltinSMModel, PreparedKernelCatalog],
) -> None:
    model, catalog = builtin_qcd_catalog
    quark_pdg = 2
    coupling = model.z_fermion_coupling(quark_pdg)
    parameters = (
        (f"coupling.10.{quark_pdg}_23_{quark_pdg}.component_0", coupling[0]),
        (f"coupling.10.{quark_pdg}_23_{quark_pdg}.component_1", coupling[1]),
    )

    zero_gluon = _certify_spinor_catalog(
        model,
        catalog,
        process_family="single-massless-quark-line-massive-neutral-vector",
        gluon_count=0,
        quark_pdg=quark_pdg,
        spinor_parameters=parameters,
    )
    assert zero_gluon.primitives == (
        KernelPrimitiveKind.WEYL_VECTOR_TO_WEYL_B,
        KernelPrimitiveKind.WEYL_VECTOR_TO_WEYL_A,
    )
    assert zero_gluon.spinor_parameter_names == tuple(
        name for name, _value in parameters
    )

    two_gluon = _certify_spinor_catalog(
        model,
        catalog,
        process_family="single-massless-quark-line-massive-neutral-vector",
        gluon_count=2,
        quark_pdg=quark_pdg,
        spinor_parameters=parameters,
    )
    assert two_gluon.primitives[:2] == (
        KernelPrimitiveKind.COLOR_ORDERED_THREE_VECTOR,
        KernelPrimitiveKind.FEYNMAN_VECTOR_PROPAGATOR,
    )
    assert two_gluon.primitives.count(KernelPrimitiveKind.WEYL_VECTOR_TO_WEYL_A) == 2
    assert two_gluon.primitives.count(KernelPrimitiveKind.WEYL_VECTOR_TO_WEYL_B) == 2

    with pytest.raises(
        SpinorAlgebraCertificationError,
        match=r"quark-Z chirality -1 current.*coupling names",
    ):
        _certify_spinor_catalog(
            model,
            catalog,
            process_family="single-massless-quark-line-massive-neutral-vector",
            gluon_count=0,
            quark_pdg=quark_pdg,
            spinor_parameters=(
                ("wrong-left-name", coupling[0]),
                (parameters[1][0], coupling[1]),
            ),
        )

    with pytest.raises(
        SpinorAlgebraCertificationError,
        match=r"ordered nonempty name/real-value pairs",
    ):
        _certify_spinor_catalog(
            model,
            catalog,
            process_family="single-massless-quark-line-massive-neutral-vector",
            gluon_count=0,
            quark_pdg=quark_pdg,
            spinor_parameters=(
                (parameters[0][0], float("nan")),
                parameters[1],
            ),
        )


def test_spinor_z_certificate_rejects_perturbed_exact_algebra(
    builtin_qcd_catalog: tuple[BuiltinSMModel, PreparedKernelCatalog],
) -> None:
    model, catalog = builtin_qcd_catalog
    quark_pdg = 1
    coupling = model.z_fermion_coupling(quark_pdg)
    binding = next(
        item
        for item in catalog.vertex_bindings
        if item.key.particles == (quark_pdg, 23, quark_pdg)
        and item.key.left_chirality == -1
        and item.key.right_chirality == 0
        and item.key.result_chirality == -1
    )
    kernel = catalog.by_id[binding.kernel_id]
    perturbed_kernel = replace(
        kernel,
        exact_expressions=(
            f"({kernel.exact_expressions[0]})+1",
            *kernel.exact_expressions[1:],
        ),
    )
    perturbed_catalog = replace(
        catalog,
        kernels=tuple(
            perturbed_kernel if item.kernel_id == kernel.kernel_id else item
            for item in catalog.kernels
        ),
    )

    with pytest.raises(
        SpinorAlgebraCertificationError,
        match=r"quark-Z chirality -1 current.*certified shared kernel primitive",
    ):
        _certify_spinor_catalog(
            model,
            perturbed_catalog,
            process_family="single-massless-quark-line-massive-neutral-vector",
            gluon_count=0,
            quark_pdg=quark_pdg,
            spinor_parameters=(
                (f"coupling.10.{quark_pdg}_23_{quark_pdg}.component_0", coupling[0]),
                (f"coupling.10.{quark_pdg}_23_{quark_pdg}.component_1", coupling[1]),
            ),
        )


def test_spinor_qcd_certificate_rejects_perturbed_exact_algebra(
    builtin_qcd_catalog: tuple[BuiltinSMModel, PreparedKernelCatalog],
) -> None:
    model, catalog = builtin_qcd_catalog
    binding = next(
        item for item in catalog.vertex_bindings if item.key.particles == (21, 21, 21)
    )
    kernel = catalog.by_id[binding.kernel_id]
    perturbed_kernel = replace(
        kernel,
        exact_expressions=(
            f"({kernel.exact_expressions[0]})+1",
            *kernel.exact_expressions[1:],
        ),
    )
    perturbed_catalog = replace(
        catalog,
        kernels=tuple(
            perturbed_kernel if item.kernel_id == kernel.kernel_id else item
            for item in catalog.kernels
        ),
    )

    with pytest.raises(
        SpinorAlgebraCertificationError,
        match=r"three-gluon current.*certified shared kernel primitive",
    ):
        _certify_spinor_catalog(
            model,
            perturbed_catalog,
            process_family="pure-gluon",
            gluon_count=4,
            quark_pdg=None,
        )
