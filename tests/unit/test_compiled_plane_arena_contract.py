# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from copy import deepcopy

import pytest

from pyamplicol._internal.versions import (
    COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
    COMPILED_PLANE_DIRECT_APPLICATION_ABI,
    COMPILED_STAGE_PLAN_ABI,
    EAGER_DIRECT_TABLE_BINDING_ABI,
    EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
    NATIVE_COMPILED_DIRECT_APPLICATION_ABI,
    SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    SYMBOLICA_CPP_RUNTIME_CAPABILITY,
    SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
    SYMJIT_APPLICATION_ABI,
    SYMJIT_F64_RUNTIME_CAPABILITY,
)
from pyamplicol.generation.artifact_writer import (
    _prefix_evaluator_payload_paths,
    _stage_evaluator_set,
)
from pyamplicol.generation.stage_artifacts import (
    _compiled_plane_arena_stage,
)


def _leaf(
    path: str,
    input_len: int,
    output_len: int,
    *,
    optimization_level: int = 3,
) -> dict[str, object]:
    return {
        "kind": "symjit-application-evaluator",
        "runtime_capability": SYMJIT_F64_RUNTIME_CAPABILITY,
        "input_len": input_len,
        "output_len": output_len,
        "application_path": path,
        "application_abi": SYMJIT_APPLICATION_ABI,
        "element_layout": "complex-f64",
        "batch_layout": "row-major",
        "compiler_type": "native",
        "translation_mode": "indirect",
        "optimization_level": optimization_level,
        "word_bits": 64,
        "endianness": "little",
        "required_defuns": [],
        "evaluator_state_path": None,
        "evaluator_state_runtime_capability": (SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY),
    }


def _stage(*, amplitude: bool = False) -> dict[str, object]:
    evaluator = {
        "kind": "chunked-symbolica-evaluator",
        "input_len": 3,
        "chunk_input_indices": [[0, 2], [1]],
        "required_runtime_capabilities": [SYMJIT_F64_RUNTIME_CAPABILITY],
        "chunks": [
            _leaf("evaluators/left.symjit", 2, 1),
            _leaf("evaluators/right.symjit", 1, 1),
        ],
    }
    return {
        "stage_index": 2,
        "stage_kind": "amplitude-roots" if amplitude else "current-combine",
        "subset_size": None if amplitude else 3,
        "evaluator_label": "amplitude" if amplitude else "stage-2",
        "parameter_layout": "stage-local-value-momentum",
        "output_length": 2,
        "output_slots": [
            {
                "value_slot_id": -1 if amplitude else 7,
                "current_id": -1 if amplitude else 7,
                "variant": "amplitude-root" if amplitude else "propagated",
                "component_start": 0 if amplitude else 12,
                "component_stop": 2 if amplitude else 14,
                "output_start": 0,
                "output_stop": 2,
                "color_selector_domain_ids": [0],
            }
        ],
        "input_value_slot_ids": [0],
        "output_value_slot_ids": [] if amplitude else [7],
        "interaction_ids": [4],
        "input_components": [
            {
                "kind": "value",
                "source_id": 0,
                "component": 0,
                "global_component": 0,
                "parameter_index": 0,
                "real_valued": False,
            },
            {
                "kind": "momentum",
                "source_id": 3,
                "component": 2,
                "global_component": 22,
                "parameter_index": 1,
                "real_valued": True,
            },
            {
                "kind": "model_parameter",
                "source_id": 1,
                "component": 0,
                "global_component": 31,
                "parameter_index": 2,
                "real_valued": False,
            },
        ],
        "parameter_count": 3,
        "value_parameter_count": 1,
        "momentum_parameter_count": 1,
        "model_parameter_count": 1,
        "real_valued_inputs": [1],
        "expression_ready": True,
        "blockers": [],
        "evaluator": evaluator,
    }


def _native_leaf(
    path: str,
    input_len: int,
    output_len: int,
    runtime_capability: str,
) -> dict[str, object]:
    return {
        "kind": "compiled-complex-evaluator",
        "runtime_capability": runtime_capability,
        "backend": "compiled-complex",
        "number_type": "complex",
        "function_name": path.replace("/", "_").replace(".", "_"),
        "input_len": input_len,
        "output_len": output_len,
        "settings": {
            "compiled_optimization_level": 3,
            "compiled_inline_asm": (
                "none"
                if runtime_capability == SYMBOLICA_CPP_RUNTIME_CAPABILITY
                else "default"
            ),
        },
        "source_path": f"{path}.direct.cpp",
        "library_path": f"{path}.direct",
        "evaluator_state_path": f"{path}.evaluator.bin",
        "native_direct_application": {
            "application_abi": NATIVE_COMPILED_DIRECT_APPLICATION_ABI,
            "function_name": path.replace("/", "_").replace(".", "_"),
            "source_path": f"{path}.direct.cpp",
            "library_path": f"{path}.direct",
            "target": {
                "triple": "aarch64-apple-darwin",
                "cpu_features": [],
            },
            "evaluator_state_sha256": "a" * 64,
            "instruction_count": 1,
            "temporary_count": 0,
            "input_plane_count": input_len,
            "scalar_input_count": 0,
            "output_plane_count": 2 * output_len,
            "simd_lane_width": 2,
            "logical_stack_bytes": 32 * output_len,
            "output_semantics": "factor-free-overwrite",
        },
    }


def _residual_plan(payload: dict[str, object]) -> dict[str, object]:
    direct = _compiled_plane_arena_stage(payload)
    assert direct is not None
    leaves = list(direct["leaves"])
    chunk_indices = list(range(len(leaves)))
    return {
        "schema_version": 2,
        "kind": "compiled-stage-plan",
        "plan_abi": COMPILED_STAGE_PLAN_ABI,
        "residual_application_abi": direct["application_abi"],
        "table_source_application_abi": SYMJIT_APPLICATION_ABI,
        "direct_table_descriptor_abi": EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
        "direct_table_binding_abi": EAGER_DIRECT_TABLE_BINDING_ABI,
        "element_layout": "split-complex-component-major",
        "input_bindings": direct["input_bindings"],
        "output_bindings": [
            {**binding, "original_output_index": binding["output_index"]}
            for binding in direct["output_bindings"]
        ],
        "residual_evaluator": payload["evaluator"],
        "residual_leaves": [
            {
                **leaf,
                "residual_leaf_index": index,
                "original_chunk_index": index,
            }
            for index, leaf in enumerate(leaves)
        ],
        "scratch_current_component_count": 0,
        "plane_catalog": [],
        "factor_catalog": [],
        "table_kernels": [],
        "table_calls": [],
        "finalizer_calls": [],
        "execution_order": [
            {
                "kind": "residual-leaf",
                "index": index,
                "original_chunk_index": index,
            }
            for index in chunk_indices
        ],
        "selector_partitions": [
            {
                "partition_id": 0,
                "helicity_selector_domain_ids": [],
                "color_selector_domain_ids": [0],
                "original_chunk_indices": chunk_indices,
            }
        ],
        "diagnostics": {
            "island_count": 0,
            "kernel_count": 0,
            "invocation_count": 0,
            "attachment_count": 0,
            "table_source_bytes": 0,
            "descriptor_bytes": 0,
            "semantic_row_bytes": 0,
            "scratch_current_component_count": 0,
        },
    }


def _set() -> dict[str, object]:
    stage = _stage()
    amplitude = _stage(amplitude=True)
    for payload in (stage, amplitude):
        payload["compiled_plane_arena"] = _residual_plan(payload)
    return {
        "kind": "generic-dag-stage-evaluator-artifacts",
        "required_runtime_capabilities": [
            COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
            SYMJIT_F64_RUNTIME_CAPABILITY,
        ],
        "runtime_available": True,
        "runtime_unavailable_message": None,
        "parameter_count": 0,
        "value_parameter_count": 0,
        "momentum_parameter_count": 0,
        "model_parameter_count": 0,
        "real_valued_inputs": [],
        "parameter_layout": "stage-local-value-momentum",
        "stage_count": 2,
        "stages": [stage],
        "amplitude_stage": amplitude,
    }


def test_compiled_stage_plan_v2_freezes_residual_leaf_and_plane_bindings() -> None:
    plan = _residual_plan(_stage())

    assert plan["schema_version"] == 2
    assert plan["kind"] == "compiled-stage-plan"
    assert plan["plan_abi"] == COMPILED_STAGE_PLAN_ABI
    assert plan["residual_application_abi"] == COMPILED_PLANE_DIRECT_APPLICATION_ABI
    assert [leaf["input_indices"] for leaf in plan["residual_leaves"]] == [
        [0, 2],
        [1],
    ]
    assert [leaf["output_start"] for leaf in plan["residual_leaves"]] == [0, 1]
    assert [binding["component"] for binding in plan["output_bindings"]] == [
        12,
        13,
    ]
    original_output_indices = [
        binding["original_output_index"] for binding in plan["output_bindings"]
    ]
    assert original_output_indices == [
        0,
        1,
    ]


@pytest.mark.parametrize("optimization_level", [0, 1, 2, 3])
def test_compiled_plane_contract_covers_every_jit_optimization_level(
    optimization_level: int,
) -> None:
    stage = _stage()
    stage["evaluator"]["chunks"][0]["optimization_level"] = optimization_level

    plan = _residual_plan(stage)

    assert [leaf["optimization_level"] for leaf in plan["residual_leaves"]] == [
        optimization_level,
        3,
    ]
    assert [
        leaf["direct_codegen_optimization_level"] for leaf in plan["residual_leaves"]
    ] == [3, 3]


def test_compiled_plane_contract_rejects_unknown_jit_optimization_level() -> None:
    stage = _stage()
    stage["evaluator"]["chunks"][0]["optimization_level"] = 4

    with pytest.raises(ValueError, match="optimization level must be 0, 1, 2, or 3"):
        _compiled_plane_arena_stage(stage)


@pytest.mark.parametrize(
    "runtime_capability",
    [SYMBOLICA_CPP_RUNTIME_CAPABILITY, SYMBOLICA_ASM_RUNTIME_CAPABILITY],
)
def test_native_compiled_contract_publishes_direct_library_leaves(
    runtime_capability: str,
) -> None:
    stage = _stage()
    stage["evaluator"] = {
        "kind": "chunked-symbolica-evaluator",
        "input_len": 3,
        "chunk_input_indices": [[0, 2], [1]],
        "required_runtime_capabilities": [runtime_capability],
        "chunks": [
            _native_leaf("compiled/left", 2, 1, runtime_capability),
            _native_leaf("compiled/right", 1, 1, runtime_capability),
        ],
    }

    plan = _residual_plan(stage)

    assert plan["residual_application_abi"] == NATIVE_COMPILED_DIRECT_APPLICATION_ABI
    assert [leaf["application_path"] for leaf in plan["residual_leaves"]] == [
        "compiled/left.direct",
        "compiled/right.direct",
    ]


def test_stage_set_requires_complete_capability_bound_metadata() -> None:
    serialized = _stage_evaluator_set(_set())
    assert (
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY
        in serialized["required_runtime_capabilities"]
    )
    assert all(
        "compiled_plane_arena" in stage
        for stage in (*serialized["stages"], serialized["amplitude_stage"])
    )

    missing = deepcopy(_set())
    del missing["stages"][0]["compiled_plane_arena"]
    with pytest.raises(ValueError, match="compiled f64 artifacts require"):
        _stage_evaluator_set(missing)

    pre_arena = deepcopy(_set())
    pre_arena["required_runtime_capabilities"].remove(
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY
    )
    del pre_arena["stages"][0]["compiled_plane_arena"]
    del pre_arena["amplitude_stage"]["compiled_plane_arena"]
    with pytest.raises(ValueError, match="compiled f64 artifacts require"):
        _stage_evaluator_set(pre_arena)


def test_stage_set_accepts_pruned_residual_input_bindings() -> None:
    payload = _set()
    stage = payload["stages"][0]
    residual = deepcopy(stage)
    residual["input_value_slot_ids"] = []
    residual["input_components"] = [
        {
            **stage["input_components"][2],
            "parameter_index": 0,
        }
    ]
    residual["parameter_count"] = 1
    residual["value_parameter_count"] = 0
    residual["momentum_parameter_count"] = 0
    residual["model_parameter_count"] = 1
    residual["real_valued_inputs"] = []
    residual["evaluator"] = _leaf("evaluators/residual.symjit", 1, 2)
    stage["compiled_plane_arena"] = _residual_plan(residual)

    serialized = _stage_evaluator_set(payload)

    direct = serialized["stages"][0]["compiled_plane_arena"]
    assert serialized["stages"][0]["parameter_count"] == 3
    assert direct["residual_evaluator"]["input_len"] == 1
    assert direct["input_bindings"] == residual["input_components"]


def test_direct_source_paths_follow_nested_lane_prefixes() -> None:
    serialized = _stage_evaluator_set(_set())
    prefixed = _prefix_evaluator_payload_paths(serialized, "lane-7")
    stage = prefixed["stages"][0]
    assert stage["evaluator"]["chunks"][0]["application_path"] == (
        "lane-7/evaluators/left.symjit"
    )
    assert stage["compiled_plane_arena"]["residual_leaves"][0]["application_path"] == (
        "lane-7/evaluators/left.symjit"
    )
