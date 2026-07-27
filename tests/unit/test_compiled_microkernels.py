# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import symbolica as symbolica_module

import pyamplicol.generation.service as service_module
from pyamplicol._internal.versions import (
    SYMJIT_APPLICATION_ABI,
    SYMJIT_F64_RUNTIME_CAPABILITY,
)
from pyamplicol.api import ProcessRequest
from pyamplicol.config import GenerationConfig
from pyamplicol.evaluators.symbolica_settings import SymbolicaEvaluatorSettings
from pyamplicol.generation.compiled_microkernels import (
    CompiledMicrokernelSession,
    _KernelSource,
    _output_chunk_ranges,
    _PlaneCatalog,
    _prune_residual_stage_inputs,
    _residual_stage,
    _selector_partitions,
)
from pyamplicol.generation.eager_tables import (
    EAGER_OUTPUT_FACTOR_NONE,
    MISSING_U32,
)
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.generation.service import _ProcessSelection
from pyamplicol.generation.stage_artifacts import (
    build_and_write_generic_stage_evaluator_artifacts,
)
from pyamplicol.generation.stage_planning import (
    _prepare_stage_for_output_chunking,
    build_generic_stage_compiler_blueprint,
)
from pyamplicol.generation.stage_types import (
    GenericCompiledStageBlueprint,
    GenericStageInputComponent,
    GenericStageOutputSlot,
)
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.prepared_catalog import (
    PreparedKernelInput,
    PreparedKernelSpec,
)


def _evaluator_process(
    expression: str,
    *,
    selection: _ProcessSelection,
) -> tuple[
    service_module.GenerationBackend,
    BuiltinSMModel,
    service_module._EvaluatorProcess,
]:
    model = BuiltinSMModel()
    backend = service_module.GenerationBackend(
        GenerationConfig(),
        None,
        process_selection=selection,
    )
    process_ir = build_process_ir(expression, color_accuracy="lc")
    dag, coverage = backend._compile_concrete_process(process_ir, model)
    prepared = backend._prepare_warmup_process(
        service_module._DagProcess(
            expanded=service_module._ExpandedProcess(
                request=ProcessRequest.parse(expression, name="microkernel_test"),
                process_ir=process_ir,
            ),
            dag=dag,
            coverage=coverage,
        ),
        model,
        index=0,
        phase=PhaseHandle("test", None, 1),
    )
    return (
        backend,
        model,
        backend._construct_evaluator(
            prepared,
            model,
            PhaseHandle("test", None, 1),
        ),
    )


def _fake_kernel_source(
    session: CompiledMicrokernelSession,
    table_kernel_id: int,
    signature: str,
) -> _KernelSource:
    spec = session._composite_kernel_spec(signature)
    return _KernelSource(
        table_kernel_id=table_kernel_id,
        canonical_signature=spec.canonical_signature,
        source_application_path=f"kernel-{table_kernel_id}.symjit",
        source_application_size_bytes=100,
        source_application_sha256="a" * 64,
        descriptor_path=f"kernel-{table_kernel_id}.bin",
        descriptor_size_bytes=20,
        descriptor_sha256="b" * 64,
        input_complex_count=spec.input_arity,
        output_complex_count=spec.output_dimension,
        input_contracts=tuple(item.to_dict() for item in spec.inputs),
        output_layout=spec.output_layout,
    )


def _lower_all_stages(
    session: CompiledMicrokernelSession,
    evaluator: service_module._EvaluatorProcess,
    settings: object,
) -> list[object]:
    lowerings = []

    def consume(stage: object, position: int, stage_count: int) -> None:
        prepared = _prepare_stage_for_output_chunking(
            stage,  # type: ignore[arg-type]
            blueprint=None,
            symbolica_settings=settings,
            current_stage_position=position,
            current_stage_count=stage_count,
        )
        lowerings.append(
            session.lower_stage(
                prepared,
                chunk_size=getattr(settings, "compiled_output_chunk_size", None),
            )
        )

    build_generic_stage_compiler_blueprint(
        evaluator.stage_input,
        model=evaluator.stage_input.model,
        runtime_schema=evaluator.runtime_schema.to_mapping(),
        stage_local_parameter_layout=True,
        stage_consumer=consume,
        release_consumed_expressions=True,
    )
    return lowerings


def test_qq_z6g_microkernel_census_and_call_partition_are_frozen(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    backend, model, evaluator = _evaluator_process(
        "u u~ > z g g g g g g",
        selection=_ProcessSelection(
            reference_color_order=(2, 4, 5, 6, 7, 8, 9, 1),
            selected_color_sector_ids=frozenset({0}),
        ),
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        CompiledMicrokernelSession,
        "_compile_kernel_source",
        _fake_kernel_source,
    )
    settings = backend._symbolica_settings()
    session = CompiledMicrokernelSession(
        dag=evaluator.compiled.dag,
        model=model,
        runtime_schema=evaluator.runtime_schema.to_mapping(),
        artifact_dir=tmp_path,
        symbolica_settings=settings,
    )
    lowerings = _lower_all_stages(session, evaluator, settings)

    assert session.profitability_diagnostics == {
        "contract": ("materialized-active-repeated-prepared-kernel-occurrences-v1"),
        "active_occurrence_count": 203,
        "eligible_occurrence_count": 103,
        "coverage_basis_points": 5073,
        "unique_projected_source_bytes": 26233,
        "replaced_projected_source_bytes": 391469,
        "projected_source_basis_points": 670,
        "kernel_identity_count": 6,
        "admitted_current_count": 31,
        "admitted": True,
    }
    current_stages = lowerings[:-1]
    assert [
        [int(call["invocation_rows"]["count"]) for call in item.table_calls]
        for item in current_stages
    ] == [
        [1, 1, 1, 1, 5],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1],
    ]
    assert sum(len(item.table_calls) for item in current_stages) == 27
    assert sum(len(item.owned_current_ids) for item in current_stages) == 31
    assert (
        sum(
            int(call["invocation_rows"]["count"])
            for item in current_stages
            for call in item.table_calls
        )
        == 31
    )
    assert all(
        int(call["invocation_rows"]["count"])
        == len(call["owned_current_ids"])
        == int(call["attachment_rows"]["count"])
        for item in current_stages
        for call in item.table_calls
    )
    [four_component_call] = [
        call
        for item in current_stages
        for call in item.table_calls
        if int(call["invocation_rows"]["count"]) == 5
    ]
    assert four_component_call["owned_current_ids"] == [15, 17, 19, 21, 23]
    assert four_component_call["interaction_ids"] == [4, 6, 8, 10, 12]
    assert four_component_call["invocation_rows"]["size_bytes"] == 840
    assert four_component_call["attachment_rows"]["size_bytes"] == 200
    four_component_source = session._kernel_sources[
        int(four_component_call["table_kernel_id"])
    ]
    assert four_component_source.input_complex_count == 20
    assert four_component_source.output_complex_count == 4
    assert len(session._kernel_sources) == 27
    assert (
        max(source.input_complex_count for source in session._kernel_sources.values())
        == 43
    )
    assert {
        source.output_complex_count for source in session._kernel_sources.values()
    } == {2, 4}
    assert all(
        item.factor_catalog
        == (
            {
                "factor_id": 0,
                "base": [1.0, 0.0],
                "model_parameter_index": None,
                "parameter_component": "none",
            },
        )
        for item in current_stages
    )
    composite_specs = tuple(session._composite_specs.values())
    assert (
        sum(
            any(
                prepared_input.role in {"coupling-real", "coupling-imag"}
                for prepared_input in spec.inputs
            )
            for spec in composite_specs
        )
        == 14
    )
    assert (
        sorted(
            prepared_input.role
            for spec in composite_specs
            for prepared_input in spec.inputs
            if prepared_input.role in {"coupling-real", "coupling-imag"}
        )
        == ["coupling-imag"] * 7 + ["coupling-real"] * 7
    )
    assert all(
        ("contribution_0_factor" in "\n".join(spec.exact_expressions))
        == any(
            prepared_input.role in {"coupling-real", "coupling-imag"}
            for prepared_input in spec.inputs
        )
        for spec in composite_specs
    )
    assert all(
        {
            symbol.to_canonical_string()
            for expression in spec.exact_expressions
            for symbol in symbolica_module.Expression.parse(expression).get_all_symbols(
                False
            )
        }
        == {
            symbolica_module.Expression.parse(
                prepared_input.symbol
            ).to_canonical_string()
            for prepared_input in spec.inputs
        }
        for spec in composite_specs
    )
    for lowering in current_stages:
        for call in lowering.table_calls:
            expected_order = [
                interaction_id
                for current_id in call["owned_current_ids"]
                for interaction_id in lowering.original_stage.interaction_ids
                if (
                    evaluator.compiled.dag.interactions[interaction_id].result_id
                    == current_id
                )
            ]
            assert call["interaction_ids"] == expected_order
        if lowering.table_calls:
            plan = session.build_stage_plan(
                lowering,
                residual_evaluator={},
                residual_leaves=tuple(
                    {} for _ in lowering.residual_original_chunk_indices
                ),
                residual_output_bindings=tuple(
                    {} for _ in lowering.residual_original_output_indices
                ),
            )
            assert [call["interaction_ids"] for call in plan["table_calls"]] == [
                call["interaction_ids"] for call in lowering.table_calls
            ]
            assert plan["finalizer_calls"] == []
            assert plan["scratch_current_component_count"] == 0
    assert [
        (
            item.original_stage.parameter_count,
            item.residual_stage.parameter_count,
        )
        for item in lowerings
    ] == [
        (94, 24),
        (176, 134),
        (244, 186),
        (292, 194),
        (278, 132),
        (194, 0),
        (118, 0),
        (8, 8),
    ]
    subset_three_residual = current_stages[1].residual_stage
    assert subset_three_residual.parameter_count == 134
    assert subset_three_residual.value_parameter_count == 74
    assert subset_three_residual.momentum_parameter_count == 60
    assert subset_three_residual.model_parameter_count == 0
    assert tuple(
        component.parameter_index
        for component in subset_three_residual.input_components
    ) == tuple(range(134))
    assert {
        component.global_component
        for component in subset_three_residual.input_components
        if component.kind == "value"
    }.isdisjoint(range(36, 44))
    assert set(subset_three_residual.input_value_slot_ids).isdisjoint({11, 12, 13, 14})
    assert not lowerings[-1].has_islands


def test_small_ddbar_z3g_is_residual_only(tmp_path: Path) -> None:
    backend, model, evaluator = _evaluator_process(
        "d d~ > z g g g",
        selection=_ProcessSelection(
            selected_color_sector_ids=frozenset({0}),
        ),
    )
    settings = backend._symbolica_settings()
    session = CompiledMicrokernelSession(
        dag=evaluator.compiled.dag,
        model=model,
        runtime_schema=evaluator.runtime_schema.to_mapping(),
        artifact_dir=tmp_path,
        symbolica_settings=settings,
    )

    assert session.profitability_diagnostics["eligible_occurrence_count"] == 34
    assert session.profitability_diagnostics["admitted"] is False
    assert not session._admitted_current_ids
    assert all(
        not lowering.has_islands
        for lowering in _lower_all_stages(session, evaluator, settings)
    )


def test_complex_parameter_projection() -> None:
    session = object.__new__(CompiledMicrokernelSession)
    complex_records = {
        "real": {
            "name": "complex_coupling.real",
            "runtime_name": "complex_coupling",
            "complex_component": "real",
            "complex_domain": "complex",
            "parameter_index": 5,
        },
        "imag": {
            "name": "complex_coupling.imag",
            "runtime_name": "complex_coupling",
            "complex_component": "imag",
            "complex_domain": "complex",
            "parameter_index": 9,
        },
    }
    session._logical_model_parameters = {
        "complex_coupling": complex_records,
    }
    session._model_parameters_by_name = {
        "real_mass": {
            "name": "real_mass",
            "parameter_index": 3,
        }
    }
    session._model_parameters = {
        5: complex_records["real"],
        9: complex_records["imag"],
    }
    complex_input = PreparedKernelInput(
        role="model-parameter",
        component=0,
        symbol="complex_coupling",
        model_parameter_name="complex_coupling",
        model_parameter_index=0,
    )
    real_input = PreparedKernelInput(
        role="model-parameter",
        component=0,
        symbol="real_mass",
        model_parameter_name="real_mass",
        model_parameter_index=1,
    )
    spec = PreparedKernelSpec(
        kernel_id=0,
        contract_kind="propagator",
        canonical_signature="c" * 64,
        exact_expressions=("complex_coupling + real_mass",),
        inputs=(complex_input, real_input),
        output_layout=("current:0",),
    )

    projection = session._runtime_model_parameter_projection(complex_input)
    assert projection == {
        "real_parameter_index": 5,
        "imag_parameter_index": 9,
    }
    planes = _PlaneCatalog()
    real_plane, imag_plane = planes.model_parameter_pair(**projection)
    assert planes.entries[real_plane] == {
        "plane_id": real_plane,
        "storage": "model-parameter",
        "component": 5,
        "part": "real",
        "current_id": None,
        "proven_real": True,
    }
    assert planes.entries[imag_plane] == {
        "plane_id": imag_plane,
        "storage": "model-parameter",
        "component": 9,
        "part": "imag",
        "current_id": None,
        "proven_real": True,
    }
    assert session._real_kernel_parameter_indices(spec) == (1,)


def test_nonzero_imaginary_composite_factor_has_no_free_unit_symbol() -> None:
    contribution = PreparedKernelSpec(
        kernel_id=3,
        contract_kind="vertex",
        canonical_signature="e" * 64,
        exact_expressions=("1", "2"),
        inputs=(),
        output_layout=("current:0", "current:1"),
    )
    session = object.__new__(CompiledMicrokernelSession)
    session._specs = {3: contribution}
    session._composite_specs = {}
    session._value_slots = {
        7: {
            "value_slot_id": 7,
            "current_id": 9,
            "component_start": 4,
            "component_stop": 6,
        }
    }
    item = SimpleNamespace(
        current_id=9,
        dimension=2,
        vertex_kernel_ids=(3,),
        finalizer_kernel_id=None,
        original_chunk_index=0,
        helicity_selector_domain_ids=(),
        color_selector_domain_ids=(),
    )
    interaction = SimpleNamespace(id=11, left_id=1, right_id=2)
    invocation = SimpleNamespace(
        kernel_id=3,
        output_factor_source=EAGER_OUTPUT_FACTOR_NONE,
        left_value_slot_id=0,
        right_value_slot_id=0,
        left_momentum_slot_id=0,
        right_momentum_slot_id=0,
    )
    attachment = SimpleNamespace(factor_real=2.0, factor_imag=-0.5)
    finalization = SimpleNamespace(
        unpropagated_value_slot_id=7,
        propagated_value_slot_id=MISSING_U32,
    )

    mismatched_item = SimpleNamespace(**{**vars(item), "vertex_kernel_ids": (4,)})
    with pytest.raises(ValueError, match="prepared-kernel witness changed"):
        session._composite_current_record(
            mismatched_item,
            (interaction,),
            ((invocation, attachment),),
            finalization,
            planes=_PlaneCatalog(),
        )
    record = session._composite_current_record(
        item,
        (interaction,),
        ((invocation, attachment),),
        finalization,
        planes=_PlaneCatalog(),
    )

    assert record is not None
    [spec] = session._composite_specs.values()
    factor = symbolica_module.Expression.parse("(2.0)+sqrt(-1)*(-0.5)")
    assert spec.exact_expressions == tuple(
        (symbolica_module.Expression.parse(expression) * factor).to_canonical_string()
        for expression in contribution.exact_expressions
    )
    assert all(
        not symbolica_module.Expression.parse(expression).get_all_symbols(False)
        for expression in spec.exact_expressions
    )


def test_residual_only_v2_reuses_each_outer_evaluator_once(
    tmp_path: Path,
) -> None:
    _backend, _model, evaluator = _evaluator_process(
        "d d~ > z g g g",
        selection=_ProcessSelection(
            selected_color_sector_ids=frozenset({0}),
        ),
    )
    compile_labels: list[str] = []

    def compiler(
        stage: GenericCompiledStageBlueprint,
        _parameters: object,
        _real: object,
    ) -> dict[str, object]:
        label = str(stage.evaluator_label)
        compile_labels.append(label)
        input_len = int(stage.parameter_count)
        output_len = int(stage.output_length)
        partitions = tuple(stage.selector_output_partitions)
        if not partitions:
            partitions = ((0, output_len),)
        chunks = [
            {
                "kind": "symjit-application-evaluator",
                "runtime_capability": SYMJIT_F64_RUNTIME_CAPABILITY,
                "input_len": input_len,
                "output_len": stop - start,
                "application_path": f"evaluators/{label}-{index}.symjit",
                "application_abi": SYMJIT_APPLICATION_ABI,
                "element_layout": "complex-f64",
                "batch_layout": "row-major",
                "optimization_level": 3,
            }
            for index, (start, stop) in enumerate(partitions)
        ]
        return {
            "kind": "chunked-symbolica-evaluator",
            "input_len": input_len,
            "chunk_input_indices": [list(range(input_len)) for _ in chunks],
            "required_runtime_capabilities": [SYMJIT_F64_RUNTIME_CAPABILITY],
            "chunks": chunks,
        }

    blueprint, artifacts = build_and_write_generic_stage_evaluator_artifacts(
        evaluator.stage_input,
        evaluator.runtime_schema.to_mapping(),
        tmp_path,
        compiler=compiler,
        symbolica_settings=SymbolicaEvaluatorSettings(),
        enable_compiled_microkernels=False,
    )
    records = [*artifacts["stages"], artifacts["amplitude_stage"]]
    assert len(compile_labels) == blueprint.stage_count == len(records)
    assert all("_microkernel_residual" not in label for label in compile_labels)
    for record in records:
        outer = record["evaluator"]
        plan = record["compiled_plane_arena"]
        assert plan["residual_evaluator"]["chunks"] == outer["chunks"]
        assert plan["table_kernels"] == []
        assert plan["table_calls"] == []
        assert plan["finalizer_calls"] == []


def test_kernel_source_is_published_once_and_discards_unused_state(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    class FakeExpression:
        @staticmethod
        def parse(value: str) -> str:
            return value

    monkeypatch.setattr(  # type: ignore[attr-defined]
        symbolica_module,
        "Expression",
        FakeExpression,
    )
    from pyamplicol.evaluators import symbolica_compile, symbolica_helpers

    monkeypatch.setattr(  # type: ignore[attr-defined]
        symbolica_compile,
        "_compile_symbolica_outputs",
        lambda *_args, **_kwargs: object(),
    )

    def manifest(_adapter: object, artifact_dir: Path) -> dict[str, object]:
        source = artifact_dir / "evaluators" / "kernel.symjit"
        state = artifact_dir / "evaluators" / "kernel.evaluator.bin"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"one machine-code application")
        state.write_bytes(b"unused serialized evaluator")
        return {
            "kind": "symjit-application-evaluator",
            "application_abi": SYMJIT_APPLICATION_ABI,
            "optimization_level": 3,
            "input_len": 1,
            "output_len": 1,
            "application_path": "evaluators/kernel.symjit",
            "evaluator_state_path": "evaluators/kernel.evaluator.bin",
        }

    monkeypatch.setattr(  # type: ignore[attr-defined]
        symbolica_helpers,
        "_symbolica_evaluator_artifact_manifest",
        manifest,
    )
    spec = PreparedKernelSpec(
        kernel_id=7,
        contract_kind="propagator",
        canonical_signature="d" * 64,
        exact_expressions=("momentum_0",),
        inputs=(
            PreparedKernelInput(
                role="momentum",
                component=0,
                symbol="momentum_0",
            ),
        ),
        output_layout=("current:0",),
    )
    session = object.__new__(CompiledMicrokernelSession)
    session._specs = {}
    session._composite_specs = {spec.canonical_signature: spec}
    session.settings = SymbolicaEvaluatorSettings()
    session.artifact_dir = tmp_path
    session._total_source_bytes = 0
    session._descriptor_builder = lambda *_args, **_kwargs: b"descriptor"
    descriptor_root = tmp_path / "compiled-microkernels" / "kernels"
    assert not descriptor_root.exists()

    with pytest.raises(ValueError, match="must be a composite current"):
        session._compile_kernel_source(  # type: ignore[arg-type]
            0,
            ("prepared", 7),
        )
    source = session._compile_kernel_source(
        0,
        spec.canonical_signature,
    )
    assert source.source_application_path == "evaluators/kernel.symjit"
    assert (tmp_path / source.source_application_path).is_file()
    assert not (tmp_path / "evaluators" / "kernel.evaluator.bin").exists()
    assert list(tmp_path.rglob("*.symjit")) == [
        tmp_path / source.source_application_path
    ]
    assert (tmp_path / source.descriptor_path).read_bytes() == b"descriptor"


def _chunk_test_stage(
    widths: tuple[int, ...],
    *,
    selector_partitions: tuple[tuple[int, int], ...],
) -> GenericCompiledStageBlueprint:
    slots: list[GenericStageOutputSlot] = []
    outputs: list[object] = []
    cursor = 0
    for current_id, width in enumerate(widths, start=1):
        start = cursor
        cursor += width
        slots.append(
            GenericStageOutputSlot(
                value_slot_id=current_id,
                current_id=current_id,
                variant="current",
                component_start=0,
                component_stop=width,
                output_start=start,
                output_stop=cursor,
            )
        )
        outputs.extend(
            SimpleNamespace(to_canonical_string=lambda index=index: f"output_{index}")
            for index in range(start, cursor)
        )
    return GenericCompiledStageBlueprint(
        stage_index=2,
        stage_kind="current-combine",
        subset_size=3,
        evaluator_label="chunk_test",
        parameter_layout="stage-local-value-momentum",
        output_length=cursor,
        output_slots=tuple(slots),
        input_value_slot_ids=(),
        output_value_slot_ids=tuple(range(1, len(slots) + 1)),
        interaction_ids=(),
        input_components=(),
        parameter_count=0,
        value_parameter_count=0,
        momentum_parameter_count=0,
        model_parameter_count=0,
        real_valued_inputs=(),
        expression_ready=True,
        blockers=(),
        first_output_previews=(),
        selector_output_partitions=selector_partitions,
        output_expressions=tuple(outputs),
    )


def test_output_chunk_ranges_preserve_fixed_scalar_boundaries() -> None:
    stage = _chunk_test_stage(
        (4, 6, 4, 3, 7),
        selector_partitions=((0, 14), (14, 24)),
    )

    assert _output_chunk_ranges(stage, chunk_size=8) == (
        (0, 8),
        (8, 14),
        (14, 22),
        (22, 24),
    )


def test_residual_stage_clips_split_slots_and_maps_original_outputs() -> None:
    stage = _chunk_test_stage(
        (2, 4, 4),
        selector_partitions=((0, 10),),
    )
    ranges = _output_chunk_ranges(stage, chunk_size=8)

    residual, residual_chunks, original_outputs = _residual_stage(
        stage,
        dag=SimpleNamespace(interactions=()),
        owned_current_ids={2},
        original_chunk_ranges=ranges,
    )

    assert ranges == ((0, 8), (8, 10))
    assert residual_chunks == (0, 1)
    assert original_outputs == (0, 1, 6, 7, 8, 9)
    assert residual.selector_output_partitions == ((0, 4), (4, 6))
    assert residual.output_value_slot_ids == (1, 3)
    assert [
        (
            slot.current_id,
            slot.component_start,
            slot.component_stop,
            slot.output_start,
            slot.output_stop,
        )
        for slot in residual.output_slots
    ] == [
        (1, 0, 2, 0, 2),
        (3, 0, 2, 2, 4),
        (3, 2, 4, 4, 6),
    ]


def test_residual_input_projection_retains_function_closure_and_renumbers() -> None:
    value, unused, momentum, model, argument = symbolica_module.S(
        "residual_value",
        "residual_unused",
        "residual_momentum",
        "residual_model",
        "residual_argument",
    )
    function = symbolica_module.S("residual_function")
    output = momentum + function(value)
    stage = replace(
        _chunk_test_stage((1,), selector_partitions=((0, 1),)),
        input_value_slot_ids=(10, 11),
        input_components=(
            GenericStageInputComponent("value", 10, 0, 4, 0),
            GenericStageInputComponent("value", 11, 0, 5, 1),
            GenericStageInputComponent(
                "momentum",
                2,
                0,
                100,
                2,
                real_valued=True,
            ),
            GenericStageInputComponent(
                "model_parameter",
                7,
                0,
                200,
                3,
                real_valued=True,
            ),
        ),
        parameter_count=4,
        value_parameter_count=2,
        momentum_parameter_count=1,
        model_parameter_count=1,
        real_valued_inputs=(2, 3),
        parameter_symbols=(value, unused, momentum, model),
        output_expressions=(output,),
        symbolica_functions=((function, (argument,), argument + model),),
    )

    projected = _prune_residual_stage_inputs(stage)

    assert value in set(output.get_all_symbols(False))
    assert projected.parameter_symbols == (value, momentum, model)
    assert projected.parameter_count == 3
    assert projected.value_parameter_count == 1
    assert projected.momentum_parameter_count == 1
    assert projected.model_parameter_count == 1
    assert projected.real_valued_inputs == (1, 2)
    assert projected.input_value_slot_ids == (10,)
    assert [
        (
            component.kind,
            component.source_id,
            component.global_component,
            component.parameter_index,
        )
        for component in projected.input_components
    ] == [
        ("value", 10, 4, 0),
        ("momentum", 2, 100, 1),
        ("model_parameter", 7, 200, 2),
    ]
    assert projected.output_expressions == stage.output_expressions
    assert projected.selector_output_partitions == stage.selector_output_partitions


def test_empty_residual_input_projection_has_an_empty_abi() -> None:
    value = symbolica_module.S("unused_empty_residual_value")
    stage = replace(
        _chunk_test_stage((1,), selector_partitions=((0, 1),)),
        input_value_slot_ids=(10,),
        input_components=(GenericStageInputComponent("value", 10, 0, 4, 0),),
        parameter_count=1,
        value_parameter_count=1,
        momentum_parameter_count=0,
        model_parameter_count=0,
        parameter_symbols=(value,),
        output_length=0,
        output_slots=(),
        output_value_slot_ids=(),
        output_expressions=(),
        selector_output_partitions=(),
    )

    projected = _prune_residual_stage_inputs(stage)

    assert projected.parameter_count == 0
    assert projected.input_components == ()
    assert projected.input_value_slot_ids == ()
    assert projected.parameter_symbols == ()


def test_selector_partitions_cover_every_chunk_overlapped_by_a_slot() -> None:
    stage = _chunk_test_stage(
        (2, 4, 4),
        selector_partitions=((0, 10),),
    )
    ranges = _output_chunk_ranges(stage, chunk_size=8)

    assert _selector_partitions(
        stage,
        ranges,
        (
            {
                "kind": "residual-leaf",
                "index": 0,
                "original_chunk_index": 0,
            },
            {
                "kind": "residual-leaf",
                "index": 1,
                "original_chunk_index": 1,
            },
        ),
    ) == (
        {
            "partition_id": 0,
            "helicity_selector_domain_ids": [],
            "color_selector_domain_ids": [],
            "original_chunk_indices": [0, 1],
        },
    )
