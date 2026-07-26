# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.developer import four_quark_compiled_gate as gate


def test_checked_file_identity_rejects_a_symlink(tmp_path: Path) -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("O_NOFOLLOW is unavailable")
    target = tmp_path / "target.bin"
    target.write_bytes(b"authenticated")
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    with pytest.raises(gate.GateError, match="checked fd"):
        gate._regular_file_identity(link)


def _source_leaf(
    *,
    path: str = "evaluators/stage.symjit",
    output_len: int = 1,
) -> dict[str, object]:
    return {
        "kind": "symjit-application-evaluator",
        "runtime_capability": gate.SYMJIT_RUNTIME_CAPABILITY,
        "input_len": 1,
        "output_len": output_len,
        "application_path": path,
        "application_abi": gate.SYMJIT_APPLICATION_ABI,
        "optimization_level": 3,
    }


def _input_binding() -> dict[str, object]:
    return {
        "kind": "value",
        "parameter_index": 0,
        "source_id": 0,
        "component": 0,
        "global_component": 0,
        "real_valued": False,
    }


def _base_plan() -> dict[str, object]:
    return {
        "kind": gate.COMPILED_STAGE_PLAN_KIND,
        "schema_version": gate.COMPILED_STAGE_PLAN_SCHEMA_VERSION,
        "plan_abi": gate.COMPILED_STAGE_PLAN_ABI,
        "residual_application_abi": gate.COMPILED_PLANE_DIRECT_APPLICATION_ABI,
        "table_source_application_abi": gate.SYMJIT_APPLICATION_ABI,
        "direct_table_descriptor_abi": gate.COMPILED_DIRECT_TABLE_DESCRIPTOR_ABI,
        "direct_table_binding_abi": gate.COMPILED_DIRECT_TABLE_BINDING_ABI,
        "element_layout": "split-complex-component-major",
        "input_bindings": [_input_binding()],
    }


def _selector_partition() -> list[dict[str, object]]:
    return [
        {
            "partition_id": 0,
            "helicity_selector_domain_ids": [],
            "color_selector_domain_ids": [],
            "original_chunk_indices": [0],
        }
    ]


def _residual_leaf(path: str) -> dict[str, object]:
    return {
        "residual_leaf_index": 0,
        "original_chunk_index": 0,
        "application_path": path,
        "source_application_abi": gate.SYMJIT_APPLICATION_ABI,
        "optimization_level": 3,
        "direct_codegen_optimization_level": 3,
        "input_indices": [0],
        "input_len": 1,
        "output_len": 1,
        "output_start": 0,
        "output_stop": 1,
    }


def _amplitude_stage() -> dict[str, object]:
    stage: dict[str, object] = {
        "stage_index": 1,
        "stage_kind": "amplitude-roots",
        "parameter_count": 1,
        "interaction_ids": [],
        "output_length": 1,
        "output_slots": [
            {
                "current_id": -1,
                "component_start": 0,
                "component_stop": 1,
                "output_start": 0,
                "output_stop": 1,
            }
        ],
        "evaluator": _source_leaf(),
    }
    stage["compiled_plane_arena"] = {
        **_base_plan(),
        "residual_evaluator": _source_leaf(),
        "output_bindings": [
            {
                "arena": "amplitude",
                "component": 0,
                "output_index": 0,
                "original_output_index": 0,
            }
        ],
        "residual_leaves": [_residual_leaf("evaluators/stage.symjit")],
        "scratch_current_component_count": 0,
        "plane_catalog": [],
        "factor_catalog": [],
        "table_kernels": [],
        "table_calls": [],
        "finalizer_calls": [],
        "execution_order": [
            {"kind": "residual-leaf", "index": 0, "original_chunk_index": 0}
        ],
        "selector_partitions": _selector_partition(),
        "diagnostics": {
            "island_count": 0,
            "kernel_count": 0,
            "invocation_count": 0,
            "attachment_count": 0,
            "table_source_bytes": 0,
            "descriptor_bytes": 0,
            "residual_source_bytes": 1,
            "semantic_row_bytes": 0,
            "scratch_current_component_count": 0,
        },
    }
    return stage


def _kernel(
    *,
    kernel_id: int,
    role: str,
    source: str,
) -> dict[str, object]:
    return {
        "table_kernel_id": kernel_id,
        "prepared_kernel_id": 7 if role == "contribution" else None,
        "role": role,
        "canonical_signature": str(kernel_id + 1) * 64,
        "source_application": {
            "path": f"evaluators/{source}.symjit",
            "size_bytes": 128,
            "sha256": "a" * 64,
        },
        "descriptor": {
            "path": f"evaluators/{source}.table",
            "size_bytes": 64,
            "sha256": "b" * 64,
        },
        "source_application_abi": gate.SYMJIT_APPLICATION_ABI,
        "descriptor_abi": gate.COMPILED_DIRECT_TABLE_DESCRIPTOR_ABI,
        "binding_abi": gate.COMPILED_DIRECT_TABLE_BINDING_ABI,
        "input_complex_count": 1,
        "output_complex_count": 2,
        "scalar_input_count": 0,
        "optimization_level": 3,
        "input_contracts": [{"role": "current"}],
        "output_layout": ["current:0", "current:1"],
    }


def _rows(path: str, row_size: int) -> dict[str, object]:
    return {
        "path": f"compiled-microkernels/{path}.bin",
        "size_bytes": row_size,
        "sha256": "c" * 64,
        "count": 1,
        "row_size": row_size,
    }


def _call(kernel_id: int) -> dict[str, object]:
    return {
        "table_kernel_id": kernel_id,
        "invocation_rows": _rows(f"invocations-{kernel_id}", 16),
        "attachment_rows": _rows(f"attachments-{kernel_id}", 24),
        "owned_current_ids": [11],
        "dependency_current_ids": [1],
        "dependency_current_components": [0],
        "selector_partition_ids": [0],
    }


def _mixed_current_stage(*, table_only: bool = False) -> dict[str, object]:
    output_offset = 0 if table_only else 1
    stage: dict[str, object] = {
        "stage_index": 0,
        "stage_kind": "current-stage",
        "parameter_count": 1,
        "interaction_ids": [100],
        "output_length": 2 if table_only else 3,
        "output_slots": (
            []
            if table_only
            else [
                {
                    "current_id": 10,
                    "component_start": 4,
                    "component_stop": 5,
                    "output_start": 0,
                    "output_stop": 1,
                }
            ]
        )
        + [
            {
                "current_id": 11,
                "component_start": 5,
                "component_stop": 7,
                "output_start": output_offset,
                "output_stop": output_offset + 2,
            }
        ],
        "evaluator": _source_leaf(
            path="evaluators/current-full.symjit",
            output_len=2 if table_only else 3,
        ),
    }
    residual_evaluator = (
        {
            "kind": "compiled-stage-empty-residual",
            "input_len": 0,
            "output_len": 0,
            "required_runtime_capabilities": [],
        }
        if table_only
        else _source_leaf(path="evaluators/residual.symjit")
    )
    residual_leaves = (
        [] if table_only else [_residual_leaf("evaluators/residual.symjit")]
    )
    output_bindings = (
        []
        if table_only
        else [
            {
                "arena": "current",
                "component": 4,
                "output_index": 0,
                "original_output_index": 0,
            }
        ]
    )
    execution_order = (
        []
        if table_only
        else [{"kind": "residual-leaf", "index": 0, "original_chunk_index": 0}]
    ) + [
        {"kind": "table-call", "index": 0, "original_chunk_index": 0},
        {"kind": "finalizer-call", "index": 0, "original_chunk_index": 0},
    ]
    stage["compiled_plane_arena"] = {
        **_base_plan(),
        "residual_evaluator": residual_evaluator,
        "output_bindings": output_bindings,
        "residual_leaves": residual_leaves,
        "scratch_current_component_count": 2,
        "plane_catalog": [],
        "factor_catalog": [],
        "table_kernels": [
            _kernel(kernel_id=0, role="contribution", source="contribution"),
            _kernel(kernel_id=1, role="finalizer", source="finalizer"),
        ],
        "table_calls": [_call(0)],
        "finalizer_calls": [_call(1)],
        "execution_order": execution_order,
        "selector_partitions": _selector_partition(),
        "diagnostics": {
            "island_count": 1,
            "kernel_count": 2,
            "invocation_count": 2,
            "attachment_count": 2,
            "table_source_bytes": 256,
            "descriptor_bytes": 128,
            "residual_source_bytes": 0 if table_only else 1,
            "semantic_row_bytes": 80,
            "scratch_current_component_count": 2,
        },
    }
    return stage


def _execution_payload() -> dict[str, object]:
    return {
        "required_runtime_capabilities": [
            gate.COMPILED_PLANE_ARENA_CAPABILITY,
            gate.SYMJIT_RUNTIME_CAPABILITY,
        ],
        "compiled": {
            "model_parameter_evaluator": {
                "kind": "model-parameter-evaluator",
                "evaluator": _source_leaf(),
            },
            "stage_evaluators": {
                "kind": "generic-dag-stage-evaluator-artifacts",
                "runtime_available": True,
                "required_runtime_capabilities": [
                    gate.COMPILED_PLANE_ARENA_CAPABILITY,
                    gate.SYMJIT_RUNTIME_CAPABILITY,
                ],
                "stage_count": 2,
                "stages": [_mixed_current_stage()],
                "amplitude_stage": _amplitude_stage(),
            },
        },
    }


def _capture(offset: float = 0.0) -> gate.EvaluationCapture:
    return gate.EvaluationCapture(
        physics_axes={
            "process": gate.PROCESS,
            "helicities": ["h:one"],
            "color_flows": ["flow:one"],
        },
        f64_total=(complex(1.0 + offset), complex(2.0 + offset)),
        exact_total=(complex(1.0 + offset), complex(2.0 + offset)),
        f64_components=(complex(0.25 + offset), complex(0.75)),
        exact_components=(complex(0.25 + offset), complex(0.75)),
        color_probe=(complex(0.5 + offset), complex(1.0)),
        helicity_probe=(complex(0.4 + offset), complex(0.8)),
    )


def test_recursive_direct_arena_audit_accepts_complete_o3_stage() -> None:
    result = gate._audit_execution_payload(_execution_payload())

    assert result == {
        "capability": gate.COMPILED_PLANE_ARENA_CAPABILITY,
        "capability_declaration_count": 2,
        "stage_set_count": 1,
        "stage_count": 2,
        "descriptor_count": 2,
        "leaf_count": 2,
        "kernel_count": 2,
        "island_count": 1,
        "invocation_count": 2,
        "attachment_count": 2,
        "residual_destination_count": 2,
        "table_destination_count": 1,
        "model_parameter_slot_count": 1,
        "model_parameter_evaluator_count": 1,
        "model_parameter_direct_application_count": 0,
        "passes": True,
    }


def test_recursive_audit_rejects_v1_and_split_ownership() -> None:
    payload = _execution_payload()
    amplitude = payload["compiled"]["stage_evaluators"]["amplitude_stage"]
    amplitude["compiled_plane_arena"]["schema_version"] = 1
    with pytest.raises(gate.GateError, match="wrong schema"):
        gate._audit_execution_payload(payload)

    payload = _execution_payload()
    current = payload["compiled"]["stage_evaluators"]["stages"][0]
    descriptor = current["compiled_plane_arena"]
    descriptor["output_bindings"][0]["original_output_index"] = 3
    with pytest.raises(gate.GateError, match="ownership"):
        gate._audit_execution_payload(payload)


def test_table_only_stage_uses_an_explicit_empty_residual() -> None:
    stage = _mixed_current_stage(table_only=True)
    descriptor = stage["compiled_plane_arena"]

    result = gate._audit_direct_descriptor(
        descriptor,
        stage=stage,
        label="table-only",
    )

    assert result["leaf_count"] == 0
    assert result["table_destination_count"] == 1


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda payload: payload["compiled"]["stage_evaluators"][
                "amplitude_stage"
            ].pop("compiled_plane_arena"),
            "compiled_plane_arena",
        ),
        (
            lambda payload: payload["compiled"]["stage_evaluators"]["amplitude_stage"][
                "compiled_plane_arena"
            ]["residual_leaves"][0].update({"optimization_level": 2}),
            "residual leaf identity",
        ),
        (
            lambda payload: payload["compiled"]["stage_evaluators"]["amplitude_stage"][
                "compiled_plane_arena"
            ]["residual_leaves"][0].update({"direct_codegen_optimization_level": 2}),
            "residual leaf identity",
        ),
        (
            lambda payload: payload["compiled"]["stage_evaluators"]["amplitude_stage"][
                "compiled_plane_arena"
            ]["residual_leaves"][0].update({"output_len": 2}),
            "residual leaf identity",
        ),
        (
            lambda payload: payload["compiled"]["model_parameter_evaluator"].update(
                {"compiled_plane_arena": {"kind": "compiled-plane-arena-stage"}}
            ),
            "model_parameter_evaluator illegally",
        ),
        (
            lambda payload: payload["compiled"]["model_parameter_evaluator"].update(
                {
                    "nested": {
                        "application_abi": (gate.NATIVE_COMPILED_DIRECT_APPLICATION_ABI)
                    }
                }
            ),
            "direct application ABI",
        ),
    ],
)
def test_recursive_direct_arena_audit_rejects_descriptor_drift(
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    payload = deepcopy(_execution_payload())
    mutation(payload)

    with pytest.raises(gate.GateError, match=message):
        gate._audit_execution_payload(payload)


def test_lc_cross_layout_audit_requires_exact_axes_and_pointwise_parity() -> None:
    result = gate._cross_lc_parity(_capture(), _capture())

    assert result["passes"] is True
    assert result["physical_axes_match"] is True
    assert set(result["comparisons"]) == {
        "f64_totals",
        "precision32_totals",
        "f64_resolved_components",
        "precision32_resolved_components",
        "color_selector",
        "helicity_selector",
    }

    with pytest.raises(gate.GateError, match="physical axes differ"):
        drift = _capture()
        drift.physics_axes["color_flows"] = ["flow:other"]
        gate._cross_lc_parity(_capture(), drift)

    with pytest.raises(gate.GateError, match="f64 totals failed"):
        gate._cross_lc_parity(_capture(), _capture(offset=1.0e-3))


def test_union_rejection_is_checked_through_public_color_config() -> None:
    result = gate._assert_union_rejections()

    assert [entry["color_accuracy"] for entry in result] == ["nlc", "full"]
    assert all(entry["rejected"] is True for entry in result)
    assert all("all-flow-union" in entry["message"] for entry in result)


def test_content_identity_addresses_the_complete_evidence_body() -> None:
    body = {"kind": gate.RESULT_KIND, "passes": True, "lanes": {"lc": [1, 2]}}

    addressed = gate._attach_content_identity(body)

    assert addressed["content_identity"]["sha256"] == gate._canonical_sha256(body)
    assert gate._attach_content_identity(body) == addressed
    assert gate._attach_content_identity({**body, "passes": False}) != addressed


def test_compiled_profile_requires_direct_calls_and_zero_boundary_traffic() -> None:
    profile = {
        "execution_mode": "compiled",
        "evaluator_backend_call_count": 7,
        "compiled_direct_arena_engine_count": 2,
        "compiled_direct_arena_call_count": 7,
    }
    profile.update(dict.fromkeys(gate._ZERO_PROFILE_COUNTERS, 0))
    profile.update(dict.fromkeys(gate._ZERO_PROFILE_TIMES, 0.0))

    evidence = gate._audit_compiled_direct_profile(profile)

    assert evidence["compiled_direct_arena_call_count"] == 7
    assert evidence["compiled_direct_arena_engine_count"] == 2
    assert evidence["legacy_boundary_component_total"] == 0
    with pytest.raises(gate.GateError, match="no backend calls"):
        gate._audit_compiled_direct_profile(
            {**profile, "compiled_direct_arena_call_count": 0}
        )
    with pytest.raises(gate.GateError, match="legacy boundary traffic"):
        gate._audit_compiled_direct_profile(
            {**profile, "stage_input_copy_component_count": 1}
        )
    with pytest.raises(gate.GateError, match="every evaluator backend call"):
        gate._audit_compiled_direct_profile(
            {**profile, "compiled_direct_arena_call_count": 6}
        )


def test_helicity_probe_is_computed_nonzero_and_self_representing() -> None:
    def helicity(
        identifier: str,
        *,
        computed: bool,
        structural_zero: bool,
        coefficient: float,
        representative_id: str,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            id=identifier,
            computed=computed,
            structural_zero=structural_zero,
            coefficient=coefficient,
            representative_id=representative_id,
        )

    physics = SimpleNamespace(
        helicities=(
            helicity(
                "zero",
                computed=False,
                structural_zero=True,
                coefficient=0.0,
                representative_id="zero",
            ),
            helicity(
                "first",
                computed=True,
                structural_zero=False,
                coefficient=1.0,
                representative_id="first",
            ),
            helicity(
                "alias",
                computed=False,
                structural_zero=False,
                coefficient=1.0,
                representative_id="first",
            ),
            helicity(
                "second",
                computed=True,
                structural_zero=False,
                coefficient=1.0,
                representative_id="second",
            ),
        )
    )

    selected, count = gate._select_helicity_probe(physics, lane_name="test")

    assert selected.id == "second"
    assert count == 2

    with pytest.raises(gate.GateError, match="no executable helicity"):
        gate._select_helicity_probe(
            SimpleNamespace(helicities=(physics.helicities[0],)),
            lane_name="test",
        )


def test_orchestrator_uses_exactly_four_isolated_generation_lanes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generated: list[str] = []
    capture = _capture()
    monkeypatch.setattr(gate, "_source_identity", lambda: {"revision": "a" * 40})
    monkeypatch.setattr(gate, "_runtime_identity", lambda source: {"bound": source})
    monkeypatch.setattr(
        gate,
        "_file_identity",
        lambda path: {"path": str(path), "sha256": "c" * 64},
    )
    monkeypatch.setattr(
        gate,
        "_assert_union_rejections",
        lambda: [{"color_accuracy": "nlc"}, {"color_accuracy": "full"}],
    )
    monkeypatch.setattr(
        gate,
        "_prepare_output_root",
        lambda output_root, replace: tmp_path,
    )
    monkeypatch.setattr(gate, "_deterministic_points", lambda count: (((),),) * count)
    monkeypatch.setattr(
        gate,
        "_points_identity",
        lambda points: {"sha256": "b" * 64, "point_count": len(points)},
    )

    def generate(
        artifact: Path,
        lane: gate.Lane,
        *,
        timeout: float,
        expected_revision: str,
        expected_runtime_sha256: str,
        expected_script_sha256: str,
    ) -> dict[str, object]:
        generated.append(lane.name)
        return {
            "lane": lane.name,
            "timeout": timeout,
            "artifact": str(artifact),
            "expected_revision": expected_revision,
            "expected_runtime_sha256": expected_runtime_sha256,
            "expected_script_sha256": expected_script_sha256,
        }

    monkeypatch.setattr(gate, "_invoke_generation_worker", generate)
    monkeypatch.setattr(
        gate,
        "_artifact_identity_and_audit",
        lambda artifact, lane: ({"artifact_id": lane.name}, "process"),
    )
    monkeypatch.setattr(
        gate,
        "_evaluate_artifact",
        lambda artifact, process_id, lane, points: ({"passes": True}, capture),
    )
    monkeypatch.setattr(
        gate,
        "_cross_lc_parity",
        lambda topology, union: {"passes": True},
    )
    monkeypatch.setattr(gate, "_peak_rss_gib", lambda: 1.25)
    arguments = argparse.Namespace(
        output_root=tmp_path,
        replace=False,
        points=2,
        generation_timeout=123.0,
    )

    result = gate.run_gate(arguments)

    assert generated == [lane.name for lane in gate.LANES]
    assert set(result["lanes"]) == {lane.name for lane in gate.LANES}
    assert result["passes"] is True
    assert result["status"] == "ok"
    assert result["source_identity_match"] is True
    assert result["runtime_identity_match"] is True
    assert result["gate_script_identity_match"] is True
    assert result["content_identity"]["sha256"] == gate._canonical_sha256(
        {key: value for key, value in result.items() if key != "content_identity"}
    )


def test_parser_documents_mandatory_watchdog_and_two_point_default() -> None:
    parsed = gate.parser().parse_args(["--output-root", "out"])

    assert parsed.points == 2
    assert parsed.generation_timeout == 10_800.0
    assert "--limit-gib 30" in gate.WATCHDOG_INVOCATION
    assert "--output-root" in gate.WATCHDOG_INVOCATION
