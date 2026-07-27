# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.developer import compiled_terminal_superkernel_probe as probe


def _component(
    parameter_index: int,
    *,
    kind: str = "value",
    source_id: int = 10,
    component: int = 0,
    global_component: int | None = None,
    real_valued: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        kind=kind,
        source_id=source_id,
        component=component,
        global_component=(
            100 + parameter_index if global_component is None else global_component
        ),
        parameter_index=parameter_index,
        real_valued=real_valued,
    )


def _composition(kind: str) -> SimpleNamespace:
    stage = SimpleNamespace(
        parameter_count=2,
        input_components=(
            _component(1, source_id=11, component=1),
            _component(0, source_id=10, component=0),
        ),
        output_length=2,
        output_slots=(
            SimpleNamespace(
                output_start=0,
                output_stop=2,
                component_start=700 if kind == "pair" else 0,
                component_stop=702 if kind == "pair" else 2,
                value_slot_id=91,
                current_id=92,
                variant="default",
            ),
        ),
    )
    return SimpleNamespace(
        kind=kind,
        stage=stage,
        elided_stage_indices=(6, 7),
        dependency_components=(501, 502),
    )


def _compile_record(
    *,
    payload_bytes: int,
    compile_seconds: float,
    process_peak_rss_gib: float = 2.0,
    replaced_source_payload_bytes: int | None = None,
) -> dict[str, object]:
    return {
        "status": "ok",
        "payload_bytes": payload_bytes,
        "replaced_source_payload_bytes": (
            payload_bytes + 50_000
            if replaced_source_payload_bytes is None
            else replaced_source_payload_bytes
        ),
        "compile_seconds": compile_seconds,
        "process_peak_rss_gib": process_peak_rss_gib,
    }


def _direct_record(
    kind: str,
    speedup_128: float,
    speedup_1024: float,
    *,
    lowered_stack_bytes: int | None = None,
    warmed_arena_allocation_bytes: int = 0,
) -> dict[str, object]:
    benchmarks: dict[str, object] = {}
    for batch, speedup in ((128, speedup_128), (1024, speedup_1024)):
        baseline = [float(index + 10) * 1.0e-9 for index in range(9)]
        candidate = [value * (1.0 - speedup) for value in baseline]
        benchmarks[str(batch)] = {
            "baseline_samples_seconds_per_point": baseline,
            "candidate_samples_seconds_per_point": candidate,
            "baseline_median_seconds_per_point": baseline[4],
            "candidate_median_seconds_per_point": candidate[4],
            "speedup_fraction": speedup,
            "alternating_order": [
                "baseline-first" if index % 2 == 0 else "candidate-first"
                for index in range(9)
            ],
            "baseline_iterations": [100 + index for index in range(9)],
            "candidate_iterations": [110 + index for index in range(9)],
        }
    return {
        "lowering": {
            "status": "ok",
            "source_stack_bytes": 4096,
            "lowered_stack_bytes": lowered_stack_bytes,
            "configured_stack_limit_bytes": probe.STACK_CAP_BYTES,
            "stack_limit_enforced": True,
            "warmed_arena_allocation_bytes": warmed_arena_allocation_bytes,
        },
        "numerical": {
            "status": "ok",
            "point_count": 64,
            "rtol": probe.NUMERICAL_RTOL,
            "atol": probe.NUMERICAL_ATOL,
            "max_absolute_difference": 0.0,
            "max_relative_difference": 0.0,
        },
        "benchmarks": benchmarks,
        "projection": {
            "baseline_call_count": 13,
            "candidate_call_count": 10 if kind == "pair" else 9,
            "baseline_input_plane_exposures": 100,
            "candidate_input_plane_exposures": 80,
            "baseline_output_plane_stores": 100,
            "candidate_output_plane_stores": 80,
            "baseline_logical_input_exposures": 100,
            "candidate_logical_input_exposures": 80,
        },
    }


def _candidate_evidence(
    *,
    kind: str,
    payload_bytes: int,
    compile_seconds: float,
    speedup_128: float,
    speedup_1024: float,
    lowered_stack_bytes: int | None = None,
    warmed_arena_allocation_bytes: int = 0,
) -> dict[str, object]:
    return {
        "compile": _compile_record(
            payload_bytes=payload_bytes,
            compile_seconds=compile_seconds,
        ),
        "numerical": {
            "status": "ok",
            "point_count": probe.VALIDATION_POINT_COUNT,
            "rtol": probe.NUMERICAL_RTOL,
            "atol": probe.NUMERICAL_ATOL,
            "max_absolute_difference": 0.0,
            "max_relative_difference": 0.0,
        },
        "direct": _direct_record(
            kind,
            speedup_128,
            speedup_1024,
            lowered_stack_bytes=lowered_stack_bytes,
            warmed_arena_allocation_bytes=warmed_arena_allocation_bytes,
        ),
    }


def _baseline_leaves() -> list[dict[str, object]]:
    return [
        {
            "leaf_index": index,
            "stage_ordinal": stage_ordinal,
            "source_application": {"size_bytes": 100_000 + index},
        }
        for index, stage_ordinal in enumerate(probe.EXPECTED_BASELINE_STAGE_ORDINALS)
    ]


def test_runner_schedules_are_explicit_and_canonical() -> None:
    schedules = probe._runner_schedules(_baseline_leaves())

    assert [item["leaf_index"] for item in schedules["baseline"]] == list(range(13))
    assert schedules["pair"] == [
        *[{"source": "baseline", "leaf_index": index} for index in range(8)],
        {"source": "candidate", "kind": "pair"},
        {"source": "baseline", "leaf_index": 12},
    ]
    assert schedules["full-tail"] == [
        *[{"source": "baseline", "leaf_index": index} for index in range(8)],
        {"source": "candidate", "kind": "full-tail"},
    ]

    malformed = _baseline_leaves()
    malformed[8] = {**malformed[8], "stage_ordinal": 4}
    with pytest.raises(probe.ProbeError, match="not canonical"):
        probe._runner_schedules(malformed)


def test_replaced_source_payload_gate_uses_only_elided_tail_leaves(tmp_path) -> None:
    capture = probe.CapturedSchedule(
        blueprint=object(),
        artifact=tmp_path / "unused",
        execution_path=tmp_path / "unused.json",
        leaf_bundle={"baseline_leaves": _baseline_leaves()},
        capture_evidence={},
    )

    assert probe._replaced_source_payload_bytes(capture, "pair") == sum(
        100_000 + index for index in range(8, 12)
    )
    assert probe._replaced_source_payload_bytes(capture, "full-tail") == sum(
        100_000 + index for index in range(8, 13)
    )


def test_process_member_path_resolves_only_process_relative_evaluators() -> None:
    relative = "helicity-sum/color-selector/sector-0/evaluators/stage.symjit"
    assert probe._process_member_path(relative) == (
        f"processes/{probe.PROCESS_ID}/{relative}"
    )

    for malformed in (
        "",
        "/absolute/stage.symjit",
        "../stage.symjit",
        "helicity-sum/../stage.symjit",
        f"processes/{probe.PROCESS_ID}/stage.symjit",
    ):
        with pytest.raises(probe.ProbeError, match="process-relative"):
            probe._process_member_path(malformed)


def test_selected_lane_proof_is_derived_from_every_reduction_group() -> None:
    lane = {
        "physics_reduction": {
            "groups": [
                {"physical_color_ids": [probe.SELECTED_FLOW]},
                {"physical_color_ids": [probe.SELECTED_FLOW]},
            ]
        }
    }

    assert probe._selected_lane_proof(lane) == {
        "status": "proven",
        "materialized_sector_id": 0,
        "selected_flow": probe.SELECTED_FLOW,
        "reduction_group_count": 2,
        "all_groups_exact_selected_flow": True,
        "runtime_selector_boundary_in_tail": False,
    }
    lane["physics_reduction"]["groups"][1] = {"physical_color_ids": ["other"]}
    with pytest.raises(probe.ProbeError, match="another physical color flow"):
        probe._selected_lane_proof(lane)


def test_arena_shape_is_derived_from_runtime_schema() -> None:
    shape = probe._arena_shape(
        {
            "runtime_schema": {
                "parameter_layout": {
                    "value_component_count": 4000,
                    "momentum_parameter_count": 124,
                    "model_parameter_count": 17,
                },
                "current_storage": {"component_count": 4000},
                "amplitude_stage": {"output_count": 384},
            }
        }
    )

    assert shape == {
        "value_component_count": 4000,
        "current_component_count": 4000,
        "amplitude_component_count": 384,
        "momentum_scalar_component_count": 124,
        "momentum_form_count": 31,
        "model_parameter_count": 17,
    }

    with pytest.raises(probe.ProbeError, match="four-vector aligned"):
        probe._arena_shape(
            {
                "runtime_schema": {
                    "parameter_layout": {
                        "value_component_count": 4,
                        "momentum_parameter_count": 5,
                        "model_parameter_count": 0,
                    },
                    "current_storage": {"component_count": 4},
                    "amplitude_stage": {"output_count": 1},
                }
            }
        )


def test_direct_stage_contract_rejects_aliases_and_incompatible_schema() -> None:
    direct = {
        "kind": "compiled-plane-arena-stage",
        "schema_version": 1,
        "source_application_abi": probe.SYMJIT_APPLICATION_ABI,
        "application_abi": probe.DIRECT_APPLICATION_ABI,
        "element_layout": "split-complex-component-major",
        "input_output_aliasing": "forbidden",
        "output_output_aliasing": "forbidden",
        "output_operation": "overwrite",
        "output_factor": "identity",
        "leaves": [{"application_path": "application.symjit"}],
        "input_bindings": [{"parameter_index": 0}],
        "output_bindings": [
            {"output_index": 0, "arena": "current", "component": 5},
        ],
    }
    evaluator = {"application_path": "application.symjit"}

    leaves, inputs, outputs = probe._validate_direct_stage_contract(
        direct,
        evaluator,
    )
    assert len(leaves) == len(inputs) == len(outputs) == 1

    malformed = dict(direct)
    malformed["output_operation"] = "accumulate"
    with pytest.raises(probe.ProbeError, match="output_operation"):
        probe._validate_direct_stage_contract(malformed, evaluator)

    aliased = dict(direct)
    aliased["output_bindings"] = [
        {"output_index": 0, "arena": "current", "component": 5},
        {"output_index": 1, "arena": "current", "component": 5},
    ]
    with pytest.raises(probe.ProbeError, match="outputs alias"):
        probe._validate_direct_stage_contract(aliased, evaluator)


def test_candidate_harness_record_preserves_semantic_order_and_outputs(
    tmp_path,
) -> None:
    application = tmp_path / "candidate.symjit"
    application.write_bytes(b"candidate")
    composition = _composition("pair")

    record = probe._candidate_harness_record(
        composition,
        application,
        {
            "application_sha256": probe._file_sha256(application),
            "payload_bytes": application.stat().st_size,
        },
    )

    assert [item["parameter_index"] for item in record["logical_inputs"]] == [0, 1]
    assert [item["global_component"] for item in record["logical_inputs"]] == [
        100,
        101,
    ]
    assert record["outputs"] == [
        {
            "output_index": 0,
            "arena": "current",
            "component": 700,
            "value_slot_id": 91,
            "current_id": 92,
            "variant": "default",
        },
        {
            "output_index": 1,
            "arena": "current",
            "component": 701,
            "value_slot_id": 91,
            "current_id": 92,
            "variant": "default",
        },
    ]
    assert record["elided_stage_indices"] == [6, 7]
    assert record["dependency_components"] == [501, 502]


def test_decision_prefers_full_tail_only_with_margin_and_resource_bounds() -> None:
    decision = probe.choose_candidate(
        {
            "pair": _candidate_evidence(
                kind="pair",
                payload_bytes=100_000,
                compile_seconds=2.0,
                speedup_128=0.13,
                speedup_1024=0.14,
            ),
            "full-tail": _candidate_evidence(
                kind="full-tail",
                payload_bytes=120_000,
                compile_seconds=2.5,
                speedup_128=0.17,
                speedup_1024=0.18,
            ),
        }
    )

    assert decision["accepted"] is True
    assert decision["selected"] == "full-tail"
    assert decision["reasons"] == []


def test_decision_can_select_full_tail_when_pair_misses_probe_threshold() -> None:
    decision = probe.choose_candidate(
        {
            "pair": _candidate_evidence(
                kind="pair",
                payload_bytes=100_000,
                compile_seconds=2.0,
                speedup_128=0.11,
                speedup_1024=0.11,
            ),
            "full-tail": _candidate_evidence(
                kind="full-tail",
                payload_bytes=120_000,
                compile_seconds=2.5,
                speedup_128=0.15,
                speedup_1024=0.15,
            ),
        }
    )

    assert decision["accepted"] is True
    assert decision["selected"] == "full-tail"
    assert decision["reasons"] == []


def test_decision_keeps_pair_when_full_tail_margin_is_too_small() -> None:
    decision = probe.choose_candidate(
        {
            "pair": _candidate_evidence(
                kind="pair",
                payload_bytes=100_000,
                compile_seconds=2.0,
                speedup_128=0.13,
                speedup_1024=0.14,
            ),
            "full-tail": _candidate_evidence(
                kind="full-tail",
                payload_bytes=110_000,
                compile_seconds=2.1,
                speedup_128=0.15,
                speedup_1024=0.16,
            ),
        }
    )

    assert decision["accepted"] is True
    assert decision["selected"] == "pair"
    assert decision["reasons"] == [
        "full-tail: batch 128 margin is below 3%",
        "full-tail: batch 1024 margin is below 3%",
    ]


def test_decision_rejects_subthreshold_or_allocating_candidates() -> None:
    decision = probe.choose_candidate(
        {
            "pair": _candidate_evidence(
                kind="pair",
                payload_bytes=100_000,
                compile_seconds=2.0,
                speedup_128=0.11,
                speedup_1024=0.14,
            ),
            "full-tail": _candidate_evidence(
                kind="full-tail",
                payload_bytes=110_000,
                compile_seconds=2.1,
                speedup_128=0.17,
                speedup_1024=0.18,
                warmed_arena_allocation_bytes=16,
            ),
        }
    )

    assert decision["accepted"] is False
    assert decision["selected"] == "neither"
    assert any("below 12%" in reason for reason in decision["reasons"])
    assert any("allocated arena bytes" in reason for reason in decision["reasons"])


def test_decision_records_a_slow_candidate_as_rejected_evidence() -> None:
    decision = probe.choose_candidate(
        {
            "pair": _candidate_evidence(
                kind="pair",
                payload_bytes=100_000,
                compile_seconds=2.0,
                speedup_128=-0.25,
                speedup_1024=-0.10,
            ),
            "full-tail": _candidate_evidence(
                kind="full-tail",
                payload_bytes=110_000,
                compile_seconds=2.1,
                speedup_128=-0.30,
                speedup_1024=-0.12,
            ),
        }
    )

    assert decision["accepted"] is False
    assert decision["selected"] == "neither"
    assert decision["reasons"] == [
        "pair: batch 128 speedup is below 12%",
        "pair: batch 1024 speedup is below 12%",
        "full-tail: batch 128 speedup is below 12%",
        "full-tail: batch 1024 speedup is below 12%",
    ]


def test_stack_contract_accepts_enforced_cap_without_fake_lowered_measurement() -> None:
    candidate = _candidate_evidence(
        kind="pair",
        payload_bytes=100_000,
        compile_seconds=2.0,
        speedup_128=0.13,
        speedup_1024=0.14,
        lowered_stack_bytes=None,
    )
    assert probe._candidate_failures("pair", candidate) == []

    direct = candidate["direct"]
    assert isinstance(direct, dict)
    lowering = direct["lowering"]
    assert isinstance(lowering, dict)
    lowering["stack_limit_enforced"] = False
    assert probe._candidate_failures("pair", candidate) == [
        "pair: DirectApplication stack limit was not enforced"
    ]


def test_decision_rejects_source_expansion() -> None:
    pair = _candidate_evidence(
        kind="pair",
        payload_bytes=100_000,
        compile_seconds=2.0,
        speedup_128=0.13,
        speedup_1024=0.14,
    )
    compile_record = pair["compile"]
    assert isinstance(compile_record, dict)
    compile_record["replaced_source_payload_bytes"] = 99_999

    assert probe._candidate_failures("pair", pair) == [
        "pair: source payload expands over the replaced tail"
    ]


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    (
        ("speedup_fraction", 0.99, "speedup is not recomputable"),
        (
            "alternating_order",
            ["baseline-first"] * 9,
            "timing order is not the strict alternation",
        ),
        (
            "baseline_samples_seconds_per_point",
            [1.0e-9] * 8,
            "exactly 9 samples",
        ),
        ("baseline_iterations", [0] * 9, "must be a positive integer"),
    ),
)
def test_direct_benchmark_evidence_is_recomputed_fail_closed(
    field: str,
    replacement: object,
    match: str,
) -> None:
    candidate = _candidate_evidence(
        kind="pair",
        payload_bytes=100_000,
        compile_seconds=2.0,
        speedup_128=0.13,
        speedup_1024=0.14,
    )
    direct = candidate["direct"]
    assert isinstance(direct, dict)
    benchmarks = direct["benchmarks"]
    assert isinstance(benchmarks, dict)
    batch = benchmarks["128"]
    assert isinstance(batch, dict)
    batch[field] = replacement

    with pytest.raises(probe.ProbeError, match=match):
        probe._candidate_failures("pair", candidate)


def test_run_probe_uses_only_injected_fakes_and_emits_digest(tmp_path) -> None:
    capture = probe.CapturedSchedule(
        blueprint=object(),
        artifact=tmp_path / "unused-artifact",
        execution_path=tmp_path / "unused-execution.json",
        leaf_bundle={
            "arena_shape": {
                "value_component_count": 4000,
                "current_component_count": 4000,
                "amplitude_component_count": 384,
                "momentum_scalar_component_count": 124,
                "momentum_form_count": 31,
                "model_parameter_count": 17,
            },
            "baseline_leaves": _baseline_leaves(),
        },
        capture_evidence={"capture": "fake"},
    )

    def compiler(composition, output_dir):
        candidate_dir = output_dir / composition.kind
        candidate_dir.mkdir(parents=True)
        application = candidate_dir / "application.symjit"
        application.write_bytes(composition.kind.encode())
        compile_record = _compile_record(
            payload_bytes=100_000 if composition.kind == "pair" else 120_000,
            compile_seconds=2.0 if composition.kind == "pair" else 2.5,
        )
        compile_record["application_sha256"] = probe._file_sha256(application)
        return probe.CompiledCandidate(
            kind=composition.kind,
            composition=composition,
            stage=composition.stage,
            evaluator=object(),
            application_path=application,
            compile_evidence=compile_record,
            harness_record=probe._candidate_harness_record(
                composition,
                application,
                compile_record,
            ),
        )

    def numerical_validator(_capture, candidates):
        return {
            kind: {
                "status": "ok",
                "point_count": probe.VALIDATION_POINT_COUNT,
                "rtol": probe.NUMERICAL_RTOL,
                "atol": probe.NUMERICAL_ATOL,
                "max_absolute_difference": 0.0,
                "max_relative_difference": 0.0,
            }
            for kind in candidates
        }

    def direct_runner(_capture, candidates, _output_dir):
        return {
            kind: _direct_record(
                kind,
                0.13 if kind == "pair" else 0.17,
                0.14 if kind == "pair" else 0.18,
            )
            for kind in candidates
        }

    payload = probe.run_probe(
        capture,
        (_composition("pair"), _composition("full-tail")),
        tmp_path / "probe",
        compiler=compiler,
        numerical_validator=numerical_validator,
        direct_runner=direct_runner,
    )

    assert payload["decision"]["selected"] == "full-tail"
    digest = payload.pop("content_sha256")
    assert digest == probe.canonical_sha256(payload)


def test_canonical_json_rejects_nonfinite_and_duplicate_keys() -> None:
    with pytest.raises(probe.ProbeError, match="canonical JSON"):
        probe.canonical_sha256({"value": float("nan")})
    with pytest.raises(probe.ProbeError, match="duplicate JSON key"):
        probe._strict_json_bytes(b'{"value":1,"value":2}', "fixture")
