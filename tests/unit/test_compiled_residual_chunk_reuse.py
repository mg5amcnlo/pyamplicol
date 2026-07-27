# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

from pyamplicol._internal.versions import (
    SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
    SYMJIT_APPLICATION_ABI,
    SYMJIT_F64_RUNTIME_CAPABILITY,
)
from pyamplicol.generation.stage_artifacts import (
    _compiled_plane_arena_stage,
    _reuse_exact_residual_evaluator_chunks,
)
from pyamplicol.generation.stage_types import (
    GenericCompiledStageBlueprint,
    GenericStageInputComponent,
    GenericStageOutputSlot,
)


@dataclass(frozen=True)
class _Symbol:
    name: str

    def to_canonical_string(self) -> str:
        return self.name


def _leaf(path: str, input_len: int, *, digest: str) -> dict[str, object]:
    return {
        "kind": "symjit-application-evaluator",
        "runtime_capability": SYMJIT_F64_RUNTIME_CAPABILITY,
        "input_len": input_len,
        "output_len": 2,
        "application_path": path,
        "application_abi": SYMJIT_APPLICATION_ABI,
        "element_layout": "complex-f64",
        "batch_layout": "row-major",
        "compiler_type": "native",
        "translation_mode": "indirect",
        "optimization_level": 3,
        "word_bits": 64,
        "endianness": "little",
        "required_defuns": [],
        "evaluator_state_path": f"{path}.state",
        "evaluator_state_runtime_capability": (SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY),
        "payload_sha256": digest,
    }


def _original_stage() -> GenericCompiledStageBlueprint:
    parameters = (_Symbol("p0"), _Symbol("p1"), _Symbol("p2"))
    return GenericCompiledStageBlueprint(
        stage_index=2,
        stage_kind="current-combine",
        subset_size=3,
        evaluator_label="stage-2",
        parameter_layout="stage-local-value-momentum",
        output_length=4,
        output_slots=(
            GenericStageOutputSlot(
                value_slot_id=10,
                current_id=10,
                variant="propagated",
                component_start=10,
                component_stop=12,
                output_start=0,
                output_stop=2,
                selector_domain_ids=(1,),
            ),
            GenericStageOutputSlot(
                value_slot_id=11,
                current_id=11,
                variant="propagated",
                component_start=20,
                component_stop=22,
                output_start=2,
                output_stop=4,
                selector_domain_ids=(2,),
            ),
        ),
        input_value_slot_ids=(3,),
        output_value_slot_ids=(10, 11),
        interaction_ids=(40, 41),
        input_components=(
            GenericStageInputComponent(
                kind="value",
                source_id=3,
                component=0,
                global_component=30,
                parameter_index=0,
            ),
            GenericStageInputComponent(
                kind="momentum",
                source_id=7,
                component=1,
                global_component=41,
                parameter_index=1,
                real_valued=True,
            ),
            GenericStageInputComponent(
                kind="model_parameter",
                source_id=9,
                component=0,
                global_component=50,
                parameter_index=2,
            ),
        ),
        parameter_count=3,
        value_parameter_count=1,
        momentum_parameter_count=1,
        model_parameter_count=1,
        real_valued_inputs=(1,),
        expression_ready=True,
        blockers=(),
        first_output_previews=("o0", "o1", "o2"),
        selector_output_partitions=((0, 2), (2, 4)),
        parameter_symbols=parameters,
        output_expressions=(
            _Symbol("o0"),
            _Symbol("o1"),
            _Symbol("o2"),
            _Symbol("o3"),
        ),
    )


def _exact_residual_stage(
    original: GenericCompiledStageBlueprint,
) -> GenericCompiledStageBlueprint:
    retained = original.output_slots[1]
    return replace(
        original,
        output_length=2,
        output_slots=(replace(retained, output_start=0, output_stop=2),),
        input_value_slot_ids=(),
        output_value_slot_ids=(11,),
        interaction_ids=(41,),
        input_components=(
            replace(original.input_components[1], parameter_index=0),
            replace(original.input_components[2], parameter_index=1),
        ),
        parameter_count=2,
        value_parameter_count=0,
        momentum_parameter_count=1,
        model_parameter_count=1,
        real_valued_inputs=(0,),
        selector_output_partitions=((0, 2),),
        parameter_symbols=original.parameter_symbols[1:],
        output_expressions=original.output_expressions[2:],
    )


def _outer_evaluator() -> dict[str, object]:
    return {
        "kind": "chunked-symbolica-evaluator",
        "input_len": 3,
        "chunk_input_indices": [[0, 1], [1, 2]],
        "chunks": [
            _leaf("evaluators/chunk-0.symjit", 2, digest="a" * 64),
            _leaf("evaluators/chunk-1.symjit", 2, digest="b" * 64),
        ],
        "required_runtime_capabilities": [SYMJIT_F64_RUNTIME_CAPABILITY],
    }


def _lowering(
    original: GenericCompiledStageBlueprint,
    residual: GenericCompiledStageBlueprint,
    *,
    original_output_indices: tuple[int, ...] = (2, 3),
) -> Any:
    return SimpleNamespace(
        original_stage=original,
        residual_stage=residual,
        original_chunk_ranges=((0, 2), (2, 4)),
        residual_original_chunk_indices=(1,),
        residual_original_output_indices=original_output_indices,
    )


def _outer_direct(
    original: GenericCompiledStageBlueprint,
    evaluator: dict[str, object],
) -> dict[str, object]:
    payload = original.to_json_dict()
    payload["evaluator"] = evaluator
    direct = _compiled_plane_arena_stage(payload)
    assert direct is not None
    return direct


def test_reuses_an_exact_complete_outer_chunk_without_changing_payload_identity() -> (
    None
):
    original = _original_stage()
    residual = _exact_residual_stage(original)
    evaluator = _outer_evaluator()

    reused = _reuse_exact_residual_evaluator_chunks(
        cast(Any, _lowering(original, residual)),
        outer_evaluator=evaluator,
        outer_direct=_outer_direct(original, evaluator),
    )

    assert reused is not None
    assert reused["chunks"] == [evaluator["chunks"][1]]
    assert reused["chunks"][0]["application_path"] == "evaluators/chunk-1.symjit"
    assert reused["chunks"][0]["payload_sha256"] == "b" * 64
    assert reused["build_timing"]["reused_outer_chunk_count"] == 1.0


def test_partial_outer_chunk_falls_back_to_residual_compilation() -> None:
    original = _original_stage()
    exact = _exact_residual_stage(original)
    residual = replace(
        exact,
        output_length=1,
        output_slots=(
            replace(
                exact.output_slots[0],
                component_stop=21,
                output_stop=1,
            ),
        ),
        selector_output_partitions=((0, 1),),
        output_expressions=exact.output_expressions[:1],
    )
    evaluator = _outer_evaluator()

    assert (
        _reuse_exact_residual_evaluator_chunks(
            cast(
                Any,
                _lowering(
                    original,
                    residual,
                    original_output_indices=(2,),
                ),
            ),
            outer_evaluator=evaluator,
            outer_direct=_outer_direct(original, evaluator),
        )
        is None
    )


def test_reused_chunk_remaps_inputs_and_rejects_unproven_bindings() -> None:
    original = _original_stage()
    residual = _exact_residual_stage(original)
    evaluator = _outer_evaluator()
    lowering = cast(Any, _lowering(original, residual))

    reused = _reuse_exact_residual_evaluator_chunks(
        lowering,
        outer_evaluator=evaluator,
        outer_direct=_outer_direct(original, evaluator),
    )

    assert reused is not None
    assert reused["input_len"] == 2
    assert reused["chunk_input_indices"] == [[0, 1]]
    payload = residual.to_json_dict()
    payload["evaluator"] = reused
    residual_direct = _compiled_plane_arena_stage(payload)
    assert residual_direct is not None
    assert residual_direct["leaves"][0]["input_indices"] == [0, 1]
    assert residual_direct["leaves"][0]["output_start"] == 0
    assert residual_direct["leaves"][0]["output_stop"] == 2
    assert [binding["component"] for binding in residual_direct["output_bindings"]] == [
        20,
        21,
    ]

    mismatched = replace(
        residual,
        input_components=(
            replace(residual.input_components[0], global_component=999),
            residual.input_components[1],
        ),
    )
    assert (
        _reuse_exact_residual_evaluator_chunks(
            cast(Any, _lowering(original, mismatched)),
            outer_evaluator=evaluator,
            outer_direct=_outer_direct(original, evaluator),
        )
        is None
    )
