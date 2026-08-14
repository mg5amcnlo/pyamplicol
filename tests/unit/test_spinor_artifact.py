# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any, cast

import pytest

from pyamplicol.api.errors import GenerationError
from pyamplicol.api.requests import ProcessRequest, ProcessSet
from pyamplicol.api.results import BenchmarkTimingBreakdown
from pyamplicol.api.services import Runtime
from pyamplicol.artifacts import inspection as artifact_inspection
from pyamplicol.color.plan import build_color_plan
from pyamplicol.generation import artifact_writer
from pyamplicol.generation import service as service_module
from pyamplicol.generation.artifact_writer import SpinorProcessArtifact
from pyamplicol.generation.recurrence_physics import build_graph_spinor_physics
from pyamplicol.generation.spinor_physics import (
    build_spinor_physics,
    spinor_graph_parameters,
)
from pyamplicol.generation.validation import ValidationPointRecord
from pyamplicol.models import BuiltinSMModel
from pyamplicol.processes.ir import (
    CanonicalProcessIR,
    ColorEndpointSummary,
    ProcessLegIR,
)


def _gluon_process(count: int) -> CanonicalProcessIR:
    return CanonicalProcessIR(
        process="g g > " + " ".join("g" for _ in range(count - 2)),
        key=f"gg_{count}g",
        color_accuracy="lc",
        legs=tuple(
            ProcessLegIR(
                label=index + 1,
                side="initial" if index < 2 else "final",
                particle="g",
                outgoing_particle="g",
                pdg=21,
                outgoing_pdg=21,
                statistics="boson",
                wavefunction_family="vector",
                color_role="adjoint",
                source_orientation="self-conjugate",
            )
            for index in range(count)
        ),
        color_endpoints=ColorEndpointSummary(0, 0, 0),
    )


def _scalar_process() -> CanonicalProcessIR:
    return CanonicalProcessIR(
        process="scalar_0 scalar_0 > scalar_1 scalar_2",
        key="scalar_contact",
        color_accuracy="lc",
        legs=tuple(
            ProcessLegIR(
                label=index + 1,
                side="initial" if index < 2 else "final",
                particle=f"scalar_{0 if index < 2 else index - 1}",
                outgoing_particle=f"scalar_{0 if index < 2 else index - 1}",
                pdg=9000001 + (0 if index < 2 else index - 1),
                outgoing_pdg=9000001 + (0 if index < 2 else index - 1),
                statistics="boson",
                wavefunction_family="scalar",
                color_role="singlet",
                source_orientation="self-conjugate",
            )
            for index in range(4)
        ),
        color_endpoints=ColorEndpointSummary(0, 0, 0),
    )


def _quark_process(gluon_count: int) -> CanonicalProcessIR:
    legs = [
        ProcessLegIR(
            label=1,
            side="initial",
            particle="d",
            outgoing_particle="d~",
            pdg=1,
            outgoing_pdg=-1,
            statistics="fermion",
            wavefunction_family="fermion",
            color_role="antifundamental",
            source_orientation="antiparticle",
        ),
        ProcessLegIR(
            label=2,
            side="initial",
            particle="d~",
            outgoing_particle="d",
            pdg=-1,
            outgoing_pdg=1,
            statistics="fermion",
            wavefunction_family="fermion",
            color_role="fundamental",
            source_orientation="particle",
        ),
    ]
    legs.extend(
        ProcessLegIR(
            label=index + 3,
            side="final",
            particle="g",
            outgoing_particle="g",
            pdg=21,
            outgoing_pdg=21,
            statistics="boson",
            wavefunction_family="vector",
            color_role="adjoint",
            source_orientation="self-conjugate",
        )
        for index in range(gluon_count)
    )
    return CanonicalProcessIR(
        process="d d~ > " + " ".join("g" for _ in range(gluon_count)),
        key=f"ddbar_{gluon_count}g",
        color_accuracy="lc",
        legs=tuple(legs),
        color_endpoints=ColorEndpointSummary(1, 1, 1),
    )


def _quark_order(gluon_count: int) -> tuple[int, ...]:
    return (2, *range(3, gluon_count + 3), 1)


def _gg_ttbar_process() -> CanonicalProcessIR:
    particles = (
        ("initial", "g", 21, "boson", "vector", "adjoint", "self-conjugate"),
        ("initial", "g", 21, "boson", "vector", "adjoint", "self-conjugate"),
        ("final", "t", 6, "fermion", "fermion", "fundamental", "particle"),
        (
            "final",
            "t~",
            -6,
            "fermion",
            "fermion",
            "antifundamental",
            "antiparticle",
        ),
    )
    return CanonicalProcessIR(
        process="g g > t t~",
        key="gg_ttbar",
        color_accuracy="lc",
        legs=tuple(
            ProcessLegIR(
                label=index + 1,
                side=side,
                particle=particle,
                outgoing_particle=particle,
                pdg=pdg,
                outgoing_pdg=pdg,
                statistics=statistics,
                wavefunction_family=wavefunction_family,
                color_role=color_role,
                source_orientation=source_orientation,
            )
            for index, (
                side,
                particle,
                pdg,
                statistics,
                wavefunction_family,
                color_role,
                source_orientation,
            ) in enumerate(particles)
        ),
        color_endpoints=ColorEndpointSummary(1, 1, 1),
    )


def _quark_z_process(gluon_count: int) -> CanonicalProcessIR:
    quark_process = _quark_process(0)
    legs = [
        *quark_process.legs,
        ProcessLegIR(
            label=3,
            side="final",
            particle="z",
            outgoing_particle="z",
            pdg=23,
            outgoing_pdg=23,
            statistics="boson",
            wavefunction_family="vector",
            color_role="singlet",
            source_orientation="self-conjugate",
        ),
    ]
    legs.extend(
        ProcessLegIR(
            label=index + 4,
            side="final",
            particle="g",
            outgoing_particle="g",
            pdg=21,
            outgoing_pdg=21,
            statistics="boson",
            wavefunction_family="vector",
            color_role="adjoint",
            source_orientation="self-conjugate",
        )
        for index in range(gluon_count)
    )
    suffix = " ".join(("z", *("g" for _ in range(gluon_count))))
    return CanonicalProcessIR(
        process=f"d d~ > {suffix}",
        key=f"ddbar_z_{gluon_count}g",
        color_accuracy="lc",
        legs=tuple(legs),
        color_endpoints=ColorEndpointSummary(1, 1, 1),
    )


def _quark_z_color_order(gluon_count: int) -> tuple[int, ...]:
    return (2, *range(4, gluon_count + 4), 1)


def _quark_gluon_to_quark_z_process() -> CanonicalProcessIR:
    return CanonicalProcessIR(
        process="d g > d z",
        key="dg_dz",
        color_accuracy="lc",
        legs=(
            ProcessLegIR(
                label=1,
                side="initial",
                particle="d",
                outgoing_particle="d~",
                pdg=1,
                outgoing_pdg=-1,
                statistics="fermion",
                wavefunction_family="fermion",
                color_role="antifundamental",
                source_orientation="antiparticle",
            ),
            ProcessLegIR(
                label=2,
                side="initial",
                particle="g",
                outgoing_particle="g",
                pdg=21,
                outgoing_pdg=21,
                statistics="boson",
                wavefunction_family="vector",
                color_role="adjoint",
                source_orientation="self-conjugate",
            ),
            ProcessLegIR(
                label=3,
                side="final",
                particle="d",
                outgoing_particle="d",
                pdg=1,
                outgoing_pdg=1,
                statistics="fermion",
                wavefunction_family="fermion",
                color_role="fundamental",
                source_orientation="particle",
            ),
            ProcessLegIR(
                label=4,
                side="final",
                particle="z",
                outgoing_particle="z",
                pdg=23,
                outgoing_pdg=23,
                statistics="boson",
                wavefunction_family="vector",
                color_role="singlet",
                source_orientation="self-conjugate",
            ),
        ),
        color_endpoints=ColorEndpointSummary(1, 1, 1),
    )


@pytest.mark.parametrize(
    ("count", "color_average_symmetry"),
    ((4, 81.0 / 512.0), (5, 81.0 / 512.0), (6, 243.0 / 2048.0)),
)
def test_spinor_physics_authenticates_aggregate_axis_and_normalization(
    count: int,
    color_average_symmetry: float,
) -> None:
    model = BuiltinSMModel()
    order = tuple(range(1, count + 1))
    physics = build_spinor_physics(
        _gluon_process(count),
        model,
        process_id=f"gg_{count}g",
        fixed_color_order=order,
    )

    extensions = cast(dict[str, Any], physics["extensions"])
    assert extensions["spinor_dag"] == {
        "helicity_axis": "always-summed-aggregate",
        "fixed_color_order": list(order),
    }
    assert cast(dict[str, object], physics["selectors"])["helicity"] is False
    assert cast(list[dict[str, object]], physics["helicities"])[0]["id"] == "h:sum"
    reduction = cast(dict[str, Any], physics["reduction"])
    assert cast(list[dict[str, object]], reduction["groups"])[0]["id"] == (
        "reduction:0"
    )
    normalization = cast(dict[str, Any], extensions["normalization"])
    actual = (
        normalization["color_factor"]
        * normalization["global_coupling_factor"]
        / (normalization["average_factor"] * normalization["identical_factor"])
    )
    expected = color_average_symmetry * (4.0 * math.pi * model.alpha_s_me_check) ** (
        count - 2
    )
    assert actual == pytest.approx(expected, rel=2.0e-15)


def test_spinor_writer_emits_only_authenticated_semantics(tmp_path) -> None:
    process = _gluon_process(4)
    order = (1, 2, 3, 4)
    physics = build_spinor_physics(
        process,
        BuiltinSMModel(),
        process_id="gg_4g",
        fixed_color_order=order,
    )
    artifact = SpinorProcessArtifact(
        process_id="gg_4g",
        expression=process.process,
        color_accuracy="lc",
        external_pdgs=(21, 21, 21, 21),
        aliases=(),
        physics=physics,
        fixed_color_order=order,
        validation_point=ValidationPointRecord(
            process_id="gg_4g", process=process.process, seed=1, error="unused"
        ),
        generation_filters={},
    )

    manifest = artifact_writer._spinor_execution_manifest(artifact)

    assert manifest["kind"] == "pyamplicol-runtime-spinor-dag-execution"
    assert manifest["external_count"] == 4
    assert manifest["fixed_color_order"] == [1, 2, 3, 4]
    assert manifest["helicity_reduction"] == "complete-incoherent-sum"
    assert manifest["coupling_stripped"] is True
    assert set(manifest) == {
        "schema_version",
        "kind",
        "required_runtime_capabilities",
        "process",
        "key",
        "color_accuracy",
        "external_pdg_order",
        "spinor_dag_abi",
        "external_count",
        "fixed_color_order",
        "helicity_reduction",
        "coupling_stripped",
    }
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(json.dumps(manifest), encoding="utf-8")
    inspected = artifact_inspection._execution_inspection(
        cast(Any, None),
        execution_path,
        cast(list[object], manifest["required_runtime_capabilities"]),
        process={
            "id": artifact.process_id,
            "expression": artifact.expression,
            "color_accuracy": artifact.color_accuracy,
            "external_pdgs": list(artifact.external_pdgs),
        },
    )
    assert inspected.execution_mode == "spinor"
    assert inspected.physical_helicity_count == 1
    assert inspected.physical_color_flow_count == 1


@pytest.mark.parametrize(
    ("selected_flow_ids", "selected_sector_ids", "coverage", "runtime_contract"),
    (
        (None, None, "complete", "complete-reusable"),
        ((0,), (7,), "selected", "generation-specialized"),
    ),
)
def test_graph_spinor_physics_preserves_singlet_flow_and_coverage(
    selected_flow_ids: tuple[int, ...] | None,
    selected_sector_ids: tuple[int, ...] | None,
    coverage: str,
    runtime_contract: str,
) -> None:
    process = _scalar_process()
    exact_one = SimpleNamespace(
        real_numerator=1,
        real_denominator=1,
        imag_numerator=0,
        imag_denominator=1,
    )
    logical = cast(
        Any,
        SimpleNamespace(
            process_id=process.key,
            external_legs=tuple(
                SimpleNamespace(source_slot=index, public_label=index + 1)
                for index in range(4)
            ),
            public_flows=(
                SimpleNamespace(
                    flow_id=0,
                    public_id="flow:singlet",
                    reduction_weight=exact_one,
                    word_source_slots=(),
                ),
            ),
            selected_public_flow_ids=selected_flow_ids,
        ),
    )
    physics = build_graph_spinor_physics(
        process,
        logical,
        cast(Any, SimpleNamespace(parameters=())),
        process_id="scalar_contact",
        normalization={},
        selected_color_sector_ids=selected_sector_ids,
    )

    assert cast(dict[str, object], physics["coverage"])["color"] == coverage
    assert cast(list[dict[str, object]], physics["helicities"])[0]["values"] == [
        0,
        0,
        0,
        0,
    ]
    component = cast(list[dict[str, object]], physics["color_components"])[0]
    assert component["id"] == "flow:singlet"
    assert component["word"] == []
    selector = cast(dict[str, Any], physics["extensions"])["runtime_selectors"]
    color_axis = cast(dict[str, Any], selector)["axes"]["color_flow"]
    assert color_axis == {
        "generation_coverage": coverage,
        "generation_selection": list(selected_sector_ids or ()),
        "runtime_contract": runtime_contract,
    }


def test_graph_spinor_manifest_is_family_free_and_payload_backed(tmp_path) -> None:
    process = _scalar_process()
    artifact = SpinorProcessArtifact(
        process_id=process.key,
        expression=process.process,
        color_accuracy="lc",
        external_pdgs=tuple(int(leg.pdg) for leg in process.legs),
        aliases=(),
        physics={},
        fixed_color_order=(),
        validation_point=ValidationPointRecord(
            process_id=process.key,
            process=process.process,
            seed=1,
            error="unused",
        ),
        generation_filters={},
        process_family=None,
        graph_payload=b"authenticated-v2-payload",
        runtime_metadata={"prepared_parameter_defaults": []},
        referenced_kernel_ids=frozenset({7}),
    )

    manifest = artifact_writer._spinor_execution_manifest(artifact)

    assert manifest["graph_payload"] == {
        "abi": "pyamplicol-spinor-dag-binary-v3",
        "path": "spinor-dag-v3.bin",
    }
    assert manifest["kernel_pack"] == {
        "manifest_path": "model/eager-kernel-pack.json",
        "payload_root": "model/eager-kernels",
    }
    assert manifest["fixed_color_order"] == []
    assert manifest["coupling_stripped"] is False
    assert "process_family" not in manifest
    assert "ordered_source_labels" not in manifest
    assert "spinor_parameter_names" not in manifest
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(json.dumps(manifest), encoding="utf-8")
    inspected = artifact_inspection._execution_inspection(
        cast(Any, None),
        execution_path,
        cast(list[object], manifest["required_runtime_capabilities"]),
        process={
            "id": artifact.process_id,
            "expression": artifact.expression,
            "color_accuracy": artifact.color_accuracy,
            "external_pdgs": list(artifact.external_pdgs),
        },
    )
    assert inspected.execution_mode == "spinor"
    assert inspected.physical_helicity_count == 1
    assert inspected.physical_color_flow_count == 1


@pytest.mark.parametrize("gluon_count", (2, 3, 4))
def test_spinor_physics_supports_one_crossed_massless_quark_line(
    gluon_count: int,
) -> None:
    process = _quark_process(gluon_count)
    order = _quark_order(gluon_count)
    plan = build_color_plan(
        process,
        color_accuracy="lc",
        reference_color_order=order,
        fold_trace_reflections=False,
    )

    process_family, fixed_color_order, source_order = (
        service_module._spinor_process_semantics(
            process,
            BuiltinSMModel(),
            plan.sectors[0],
            order,
        )
    )
    physics = build_spinor_physics(
        process,
        BuiltinSMModel(),
        process_id=process.key,
        fixed_color_order=fixed_color_order,
        process_family=process_family,
        ordered_source_labels=source_order,
    )

    assert process_family == "single-massless-quark-line"
    assert fixed_color_order == source_order
    assert tuple(
        next(leg for leg in process.legs if leg.label == label).outgoing_pdg
        for label in source_order
    ) == (1, *(21 for _ in range(gluon_count)), -1)
    extensions = cast(dict[str, Any], physics["extensions"])
    assert extensions["spinor_dag"] == {
        "helicity_axis": "always-summed-aggregate",
        "fixed_color_order": list(order),
        "process_family": "single-massless-quark-line",
        "ordered_source_labels": list(order),
    }
    assert cast(list[dict[str, object]], physics["color_components"])[0][
        "word"
    ] == list(order)


@pytest.mark.parametrize("order", ((3, 1, 2, 4), (3, 2, 1, 4)))
def test_spinor_physics_and_manifest_support_gg_ttbar_flows(
    order: tuple[int, ...],
    tmp_path,
) -> None:
    process = _gg_ttbar_process()
    model = BuiltinSMModel()
    plan = build_color_plan(
        process,
        color_accuracy="lc",
        reference_color_order=order,
        fold_trace_reflections=False,
    )
    family, fixed_color_order, source_order = service_module._spinor_process_semantics(
        process,
        model,
        plan.sectors[0],
        order,
    )
    graph_parameters = spinor_graph_parameters(
        process,
        model,
        process_family=family,
        ordered_source_labels=source_order,
    )
    physics = build_spinor_physics(
        process,
        model,
        process_id=process.key,
        fixed_color_order=fixed_color_order,
        process_family=family,
        ordered_source_labels=source_order,
    )

    assert family == "single-massive-quark-line"
    assert fixed_color_order == source_order == order
    assert graph_parameters == (
        ("particle.6.mass", model.mass(6)),
        ("particle.6.width", model.width(6)),
    )
    extension = cast(dict[str, Any], physics["extensions"])
    assert extension["spinor_dag"] == {
        "helicity_axis": "always-summed-aggregate",
        "fixed_color_order": list(order),
        "process_family": family,
        "ordered_source_labels": list(order),
        "spinor_parameter_names": ["particle.6.mass", "particle.6.width"],
    }
    normalization = cast(dict[str, object], extension["normalization"])
    assert normalization["qcd_coupling_power"] == 2
    assert normalization["electroweak_coupling_power"] == 0
    assert normalization["couplings_in_stage_evaluators"] is False
    parameters = {
        str(record["name"]): record
        for record in cast(list[dict[str, object]], physics["model_parameters"])
    }
    assert parameters["particle.6.mass"]["kind"] == "mass"
    assert parameters["particle.6.width"]["kind"] == "width"

    artifact = SpinorProcessArtifact(
        process_id=process.key,
        expression=process.process,
        color_accuracy="lc",
        external_pdgs=(21, 21, 6, -6),
        aliases=(),
        physics=physics,
        fixed_color_order=order,
        validation_point=ValidationPointRecord(
            process_id=process.key,
            process=process.process,
            seed=1,
            error="unused",
        ),
        generation_filters={},
        process_family=family,
        ordered_source_labels=source_order,
        spinor_parameter_names=("particle.6.mass", "particle.6.width"),
    )
    manifest = artifact_writer._spinor_execution_manifest(artifact)
    assert manifest["coupling_stripped"] is True
    execution_path = tmp_path / f"execution-{order[1]}.json"
    execution_path.write_text(json.dumps(manifest), encoding="utf-8")
    inspected = artifact_inspection._execution_inspection(
        cast(Any, None),
        execution_path,
        cast(list[object], manifest["required_runtime_capabilities"]),
        process={
            "id": artifact.process_id,
            "expression": artifact.expression,
            "color_accuracy": artifact.color_accuracy,
            "external_pdgs": list(artifact.external_pdgs),
        },
    )
    assert inspected.execution_mode == "spinor"


@pytest.mark.parametrize("gluon_count", (0, 1, 2))
def test_spinor_physics_supports_one_massive_z_insertion(gluon_count: int) -> None:
    process = _quark_z_process(gluon_count)
    color_order = _quark_z_color_order(gluon_count)
    plan = build_color_plan(
        process,
        color_accuracy="lc",
        reference_color_order=color_order,
        fold_trace_reflections=False,
    )
    model = BuiltinSMModel()

    process_family, fixed_color_order, source_order = (
        service_module._spinor_process_semantics(
            process,
            model,
            plan.sectors[0],
            color_order,
        )
    )
    graph_parameters = spinor_graph_parameters(
        process,
        model,
        process_family=process_family,
        ordered_source_labels=source_order,
    )
    physics = build_spinor_physics(
        process,
        model,
        process_id=process.key,
        fixed_color_order=fixed_color_order,
        process_family=process_family,
        ordered_source_labels=source_order,
    )

    assert process_family == "single-massless-quark-line-massive-neutral-vector"
    assert fixed_color_order == color_order
    assert source_order == (*color_order, 3)
    assert graph_parameters == (
        ("coupling.10.1_23_1.component_0", model.z_fermion_coupling(1)[0]),
        ("coupling.10.1_23_1.component_1", model.z_fermion_coupling(1)[1]),
    )
    extensions = cast(dict[str, Any], physics["extensions"])
    assert extensions["spinor_dag"] == {
        "helicity_axis": "always-summed-aggregate",
        "fixed_color_order": list(color_order),
        "process_family": process_family,
        "ordered_source_labels": list(source_order),
        "spinor_parameter_names": [name for name, _value in graph_parameters],
    }
    normalization = cast(dict[str, object], extensions["normalization"])
    assert normalization["couplings_in_stage_evaluators"] is True
    parameters = {
        str(record["name"]): record
        for record in cast(list[dict[str, object]], physics["model_parameters"])
    }
    assert parameters["particle.23.mass"]["kind"] == "mass"
    assert parameters[graph_parameters[0][0]]["kind"] == "coupling"
    assert parameters[graph_parameters[1][0]]["kind"] == "coupling"


def test_spinor_writer_keeps_z_out_of_the_physical_color_word(tmp_path) -> None:
    process = _quark_z_process(1)
    color_order = _quark_z_color_order(1)
    source_order = (*color_order, 3)
    model = BuiltinSMModel()
    family = "single-massless-quark-line-massive-neutral-vector"
    graph_parameters = spinor_graph_parameters(
        process,
        model,
        process_family=family,
        ordered_source_labels=source_order,
    )
    physics = build_spinor_physics(
        process,
        model,
        process_id=process.key,
        fixed_color_order=color_order,
        process_family=family,
        ordered_source_labels=source_order,
    )
    artifact = SpinorProcessArtifact(
        process_id=process.key,
        expression=process.process,
        color_accuracy="lc",
        external_pdgs=(1, -1, 23, 21),
        aliases=(),
        physics=physics,
        fixed_color_order=color_order,
        validation_point=ValidationPointRecord(
            process_id=process.key,
            process=process.process,
            seed=1,
            error="unused",
        ),
        generation_filters={},
        process_family=family,
        ordered_source_labels=source_order,
        spinor_parameter_names=tuple(name for name, _value in graph_parameters),
    )

    manifest = artifact_writer._spinor_execution_manifest(artifact)

    assert manifest["fixed_color_order"] == [2, 4, 1]
    assert manifest["ordered_source_labels"] == [2, 4, 1, 3]
    assert manifest["spinor_parameter_names"] == [
        "coupling.10.1_23_1.component_0",
        "coupling.10.1_23_1.component_1",
    ]
    assert manifest["coupling_stripped"] is False
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(json.dumps(manifest), encoding="utf-8")
    inspected = artifact_inspection._execution_inspection(
        cast(Any, None),
        execution_path,
        cast(list[object], manifest["required_runtime_capabilities"]),
        process={
            "id": artifact.process_id,
            "expression": artifact.expression,
            "color_accuracy": artifact.color_accuracy,
            "external_pdgs": list(artifact.external_pdgs),
        },
    )
    assert inspected.execution_mode == "spinor"


def test_spinor_writer_does_not_conjugate_a_crossed_initial_gluon() -> None:
    process = _quark_gluon_to_quark_z_process()
    color_order = (3, 2, 1)
    model = BuiltinSMModel()
    plan = build_color_plan(
        process,
        color_accuracy="lc",
        reference_color_order=color_order,
        fold_trace_reflections=False,
    )
    family, fixed_color_order, source_order = service_module._spinor_process_semantics(
        process,
        model,
        plan.sectors[0],
        color_order,
    )
    graph_parameters = spinor_graph_parameters(
        process,
        model,
        process_family=family,
        ordered_source_labels=source_order,
    )
    artifact = SpinorProcessArtifact(
        process_id=process.key,
        expression=process.process,
        color_accuracy="lc",
        external_pdgs=(1, 21, 1, 23),
        aliases=(),
        physics=build_spinor_physics(
            process,
            model,
            process_id=process.key,
            fixed_color_order=fixed_color_order,
            process_family=family,
            ordered_source_labels=source_order,
        ),
        fixed_color_order=fixed_color_order,
        validation_point=ValidationPointRecord(
            process_id=process.key,
            process=process.process,
            seed=1,
            error="unused",
        ),
        generation_filters={},
        process_family=family,
        ordered_source_labels=source_order,
        spinor_parameter_names=tuple(name for name, _value in graph_parameters),
    )

    manifest = artifact_writer._spinor_execution_manifest(artifact)

    assert fixed_color_order == (3, 2, 1)
    assert source_order == (3, 2, 1, 4)
    assert manifest["fixed_color_order"] == [3, 2, 1]
    assert manifest["ordered_source_labels"] == [3, 2, 1, 4]


def test_spinor_writer_emits_minimal_quark_traversal_contract() -> None:
    process = _quark_process(2)
    order = _quark_order(2)
    physics = build_spinor_physics(
        process,
        BuiltinSMModel(),
        process_id=process.key,
        fixed_color_order=order,
        process_family="single-massless-quark-line",
        ordered_source_labels=order,
    )
    artifact = SpinorProcessArtifact(
        process_id=process.key,
        expression=process.process,
        color_accuracy="lc",
        external_pdgs=(1, -1, 21, 21),
        aliases=(),
        physics=physics,
        fixed_color_order=order,
        validation_point=ValidationPointRecord(
            process_id=process.key,
            process=process.process,
            seed=1,
            error="unused",
        ),
        generation_filters={},
        process_family="single-massless-quark-line",
        ordered_source_labels=order,
    )

    manifest = artifact_writer._spinor_execution_manifest(artifact)

    assert manifest["process_family"] == "single-massless-quark-line"
    assert manifest["ordered_source_labels"] == [2, 3, 4, 1]
    assert "ordered_outgoing_pdgs" not in manifest
    assert set(manifest) == {
        "schema_version",
        "kind",
        "required_runtime_capabilities",
        "process",
        "key",
        "color_accuracy",
        "external_pdg_order",
        "spinor_dag_abi",
        "external_count",
        "fixed_color_order",
        "helicity_reduction",
        "coupling_stripped",
        "process_family",
        "ordered_source_labels",
    }


def test_public_runtime_and_benchmark_metadata_accept_spinor_mode() -> None:
    runtime = object.__new__(Runtime)
    runtime._backend = cast(Any, SimpleNamespace(execution_mode="spinor"))

    assert runtime.execution_mode == "spinor"
    assert (
        BenchmarkTimingBreakdown(sample_count=1, execution_mode="spinor").execution_mode
        == "spinor"
    )


def test_spinor_generation_rejects_more_than_one_concrete_process(tmp_path) -> None:
    process = _gluon_process(4)
    requests = (
        ProcessRequest(process.process, "first"),
        ProcessRequest(process.process, "second"),
    )
    expanded = tuple(
        service_module._ExpandedProcess(request=request, process_ir=process)
        for request in requests
    )
    backend = service_module.GenerationBackend(
        None,
        None,
        process_selection=service_module._ProcessSelection(
            experimental_spinor_dag=True
        ),
    )

    with pytest.raises(GenerationError, match="exactly one concrete process"):
        backend._generate_spinor_processes(
            expanded,
            output=tmp_path / "artifact",
            write_mode="error",
            requested_processes=ProcessSet(requests),
            resolved_model=cast(Any, None),
            artifact_model=cast(Any, None),
            generation_model=BuiltinSMModel(),
            reporter=cast(Any, None),
            generation_started=0.0,
        )


def test_spinor_generation_rejects_selected_source_helicities(tmp_path) -> None:
    process = _gluon_process(4)
    request = ProcessRequest(process.process, process.key)
    backend = service_module.GenerationBackend(
        None,
        None,
        process_selection=service_module._ProcessSelection(
            selected_source_helicities={1: -1},
            experimental_spinor_dag=True,
        ),
    )

    with pytest.raises(GenerationError, match="always-summed helicity axis"):
        backend._generate_spinor_processes(
            (
                service_module._ExpandedProcess(
                    request=request,
                    process_ir=process,
                ),
            ),
            output=tmp_path / "artifact",
            write_mode="error",
            requested_processes=ProcessSet((request,)),
            resolved_model=cast(Any, None),
            artifact_model=cast(Any, None),
            generation_model=BuiltinSMModel(),
            reporter=cast(Any, None),
            generation_started=0.0,
        )
