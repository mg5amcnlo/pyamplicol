# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "developer" / "symjit_2_22_migration_benchmark_gate.py"
SPEC = importlib.util.spec_from_file_location(
    "symjit_2_22_migration_benchmark_gate",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _address(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["content_sha256"] = _sha256(result)
    return result


def _peak(value: int) -> dict[str, object]:
    return {
        "source": "resource.getrusage",
        "self_peak_bytes": value,
        "maximum_child_peak_bytes": 0,
        "observed_lower_bound_bytes": value,
        "semantics": "synthetic process high-water mark",
    }


def _statistics(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    median = ordered[len(ordered) // 2]
    deviations = sorted(abs(value - median) for value in values)
    return median, deviations[len(deviations) // 2]


def _semantic_identity(layout: str, mode: str) -> dict[str, object]:
    normalization = {"average_factor": 36, "color_factor": 2187}
    reduction_ordering = {
        "kind": "lc-diagonal",
        "pair_order_abi": "helicity-major-color-minor-v1",
    }
    selector_semantics = {
        "kind": "pyamplicol-runtime-selectors",
        "layout": layout,
    }
    common_model = {
        "name": "built-in-sm",
        "content_sha256": "9" * 64,
        "compiled_schema_version": 9,
        "restriction": None,
    }
    return {
        "kind": "pyamplicol-benchmark-artifact-semantic-identity",
        "schema_version": 3,
        "mode_internal_marker": mode,
        "generation_specialized_axes": [],
        "coverage": {
            "color": "complete",
            "helicities": "complete",
            "complete_physical_axes": True,
        },
        "reduction_coverage": {
            "complete": True,
            "errors": [],
            "expected_physical_pair_count": 4,
            "observed_physical_pair_count": 4,
        },
        "normalization": normalization,
        "normalization_sha256": _sha256(normalization),
        "reduction_ordering": reduction_ordering,
        "reduction_ordering_sha256": _sha256(reduction_ordering),
        "runtime_selector_semantics": selector_semantics,
        "runtime_selector_semantics_sha256": _sha256(selector_semantics),
        "manifest_model_identity": {
            "common_physics_identity": common_model,
            "common_physics_identity_sha256": _sha256(common_model),
        },
        "physical_color_flows": {
            "count": 2,
            "ordered_entries": [{"index": 0}, {"index": 1}],
        },
        "physical_helicities": {
            "count": 2,
            "ordered_entries": [{"index": 0}, {"index": 1}],
        },
    }


def _sample(
    *,
    identifier: int,
    schedule_index: int,
    round_index: int,
    runtime: float,
    cold_load: float,
    cold_rss: int,
    profiled_rss: int,
) -> tuple[dict[str, object], dict[str, object]]:
    invocation = _address({"kind": "invocation", "identifier": identifier})
    process_record = _address({"kind": "worker-process", "identifier": identifier})
    sample: dict[str, object] = {
        "schedule_index": schedule_index,
        "round": round_index,
        "wall_seconds_per_point": runtime,
        "cold_load_seconds": cold_load,
        "peak_rss_after_cold_load": _peak(cold_rss),
        "peak_rss_after_profile": _peak(profiled_rss),
        "internal_sample_count": 7,
        "interrupted": False,
        "timing_configuration": {
            "warmup_runs": 2,
            "target_runtime_seconds": 5.0,
        },
        "timing_sources": {"wall": "runtime_core_repeated_wall_time"},
        "worker_invocation": invocation,
        "worker_process_record": process_record,
    }
    result_record = {
        "kind": "pyamplicol-retained-profile-worker-result",
        "schema_version": 1,
        "recorded_at_utc": "2026-07-30T00:00:00+00:00",
        "addressed_payload_sha256": _sha256(sample),
        "upstream_worker_result_record_sha256": "a" * 64,
        "worker_process_record_sha256": process_record["content_sha256"],
        "worker_invocation_sha256": invocation["content_sha256"],
    }
    sample["worker_result_record"] = _address(result_record)
    schedule_entry = {
        "schedule_index": schedule_index,
        "round": round_index,
        "worker_result_sha256": result_record["addressed_payload_sha256"],
    }
    return sample, schedule_entry


def _aggregate(batch_size: int, samples: list[dict[str, object]]) -> dict[str, object]:
    runtime = [float(sample["wall_seconds_per_point"]) for sample in samples]
    cold = [float(sample["cold_load_seconds"]) for sample in samples]
    cold_rss = [
        float(sample["peak_rss_after_cold_load"]["observed_lower_bound_bytes"])
        for sample in samples
    ]
    profiled_rss = [
        float(sample["peak_rss_after_profile"]["observed_lower_bound_bytes"])
        for sample in samples
    ]
    runtime_median, runtime_mad = _statistics(runtime)
    cold_median, cold_mad = _statistics(cold)
    cold_rss_median, _ = _statistics(cold_rss)
    profiled_rss_median, _ = _statistics(profiled_rss)
    return {
        "batch_size": batch_size,
        "sample_count": len(samples),
        "subprocess_sample_count": len(samples),
        "wall_seconds_per_point": runtime_median,
        "wall_seconds_per_point_median": runtime_median,
        "wall_seconds_per_point_mad": runtime_mad,
        "statistics_contract": "subprocess-median-and-raw-mad-v1",
        "cold_load_seconds_median": cold_median,
        "cold_load_seconds_mad": cold_mad,
        "cold_load_peak_rss_bytes_median": cold_rss_median,
        "profiled_peak_rss_bytes_median": profiled_rss_median,
        "resource_statistics_contract": "subprocess-median-and-raw-mad-v1",
        "subprocess_samples": samples,
        "interrupted": False,
    }


def _capture(
    *,
    role: str,
    layout: str,
    process: str,
    runtime_ratio: float = 1.0,
    compiled_to_recurrence_ratio: float = 1.05,
    generation_ratio: float = 1.0,
    resource_ratio: float = 1.0,
    eager_runtime_ratio: float = 1.0,
    eager_generation_ratio: float = 1.0,
    eager_resource_ratio: float = 1.0,
) -> dict[str, object]:
    candidate = role == "candidate"
    revision = "2" * 40 if candidate else gate.BASELINE_REVISION
    runtime_role_ratio = runtime_ratio if candidate else 1.0
    generation_role_ratio = generation_ratio if candidate else 1.0
    resource_role_ratio = resource_ratio if candidate else 1.0
    eager_runtime_role_ratio = eager_runtime_ratio if candidate else 1.0
    eager_generation_role_ratio = eager_generation_ratio if candidate else 1.0
    eager_resource_role_ratio = eager_resource_ratio if candidate else 1.0
    workload = (
        "single-runtime-selected-flow/helicity-sum"
        if layout == "topology-replay"
        else "all-flows/runtime-selected-single-helicity"
    )
    selector = {
        "color_flow_count": 2,
        "helicity_count": 2,
        "workload": workload,
        "color_flow_request": "flow:2,1" if layout == "topology-replay" else "1",
        "helicity_request": "1" if layout == "topology-replay" else "h:-1,+1",
    }
    color_request = selector["color_flow_request"]
    helicity_request = selector["helicity_request"]
    generation: dict[str, object] = {}
    profiles: dict[str, object] = {}
    samples_by_cell: dict[tuple[str, int], list[dict[str, object]]] = {
        (mode, batch): [] for mode in gate.CAPTURE_MODES for batch in gate.BATCH_SIZES
    }
    schedule_entries: list[dict[str, object]] = []
    schedule_index = 0
    identifier = 1
    deviations = (-0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003)
    for round_index, deviation in enumerate(deviations):
        offset = round_index % len(gate.CAPTURE_MODES)
        modes = gate.CAPTURE_MODES[offset:] + gate.CAPTURE_MODES[:offset]
        for batch_size in gate.BATCH_SIZES:
            for mode in modes:
                mode_ratio = (
                    compiled_to_recurrence_ratio
                    if mode == "compiled"
                    else 1.02
                    if mode == "eager"
                    else 1.0
                )
                base_runtime = (
                    1.0 + 0.1 * gate.BATCH_SIZES.index(batch_size)
                ) * mode_ratio
                mode_runtime_ratio = (
                    runtime_role_ratio * eager_runtime_role_ratio
                    if mode == "eager"
                    else runtime_role_ratio
                )
                mode_resource_ratio = (
                    resource_role_ratio * eager_resource_role_ratio
                    if mode == "eager"
                    else resource_role_ratio
                )
                runtime = base_runtime * mode_runtime_ratio * (1.0 + deviation)
                sample, entry = _sample(
                    identifier=identifier,
                    schedule_index=schedule_index,
                    round_index=round_index,
                    runtime=runtime,
                    cold_load=0.2 * mode_resource_ratio * (1.0 + deviation),
                    cold_rss=int(100_000_000 * mode_resource_ratio * (1.0 + deviation)),
                    profiled_rss=int(
                        120_000_000 * mode_resource_ratio * (1.0 + deviation)
                    ),
                )
                entry.update({"mode": mode, "batch_size": batch_size})
                schedule_entries.append(entry)
                samples_by_cell[(mode, batch_size)].append(sample)
                identifier += 1
                schedule_index += 1

    for mode in gate.CAPTURE_MODES:
        semantic = _semantic_identity(layout, mode)
        semantic_sha256 = _sha256(semantic)
        optimization = 2 if mode == "recurrence" else 3
        mode_generation_ratio = (
            generation_role_ratio * eager_generation_role_ratio
            if mode == "eager"
            else generation_role_ratio
        )
        mode_resource_ratio = (
            resource_role_ratio * eager_resource_role_ratio
            if mode == "eager"
            else resource_role_ratio
        )
        generation[mode] = {
            "mode": mode,
            "generation_wall_seconds": (
                (12.0 if mode == "compiled" else 11.0 if mode == "eager" else 10.0)
                * mode_generation_ratio
            ),
            "generation_reused": False,
            "peak_rss": _peak(int(400_000_000 * mode_resource_ratio)),
            "artifact_stats": {
                "file_count": 4,
                "size_bytes": int(10_000_000 * mode_resource_ratio),
            },
            "artifact_identity": {
                "semantic_identity": semantic,
                "semantic_identity_sha256": semantic_sha256,
            },
            "effective_contract": {
                "backend": "jit",
                "color_accuracy": "lc",
                "execution_mode": mode,
                "jit_optimization_level": optimization,
                "lc_flow_layout": layout,
            },
        }
        profiles[mode] = {
            "mode": mode,
            "process_expression": process,
            "process_id": "d_dbar_to_z_6g"
            if process.startswith("d ")
            else "u_ubar_to_z_6g",
            "selector_contract": selector,
            "validation": {
                "passes": True,
                "fixture": {
                    "point_count": 1,
                    "points_sha256": "b" * 64,
                },
            },
            "artifact_semantic_identity": semantic,
            "artifact_semantic_identity_sha256": semantic_sha256,
            "profiles": [
                _aggregate(batch, samples_by_cell[(mode, batch)])
                for batch in gate.BATCH_SIZES
            ],
        }

    native_sha = "d" * 64 if candidate else "c" * 64
    build_inputs_sha = "f" * 64 if candidate else "e" * 64
    lane_measurements = {
        mode: {
            "passes": True,
            "observed_batch_sizes": list(gate.BATCH_SIZES),
            "missing_batch_sizes": [],
            "errors": [],
        }
        for mode in gate.CAPTURE_MODES
    }
    artifact_lane_contracts = {
        mode: {"synthetic_lane_identity": mode} for mode in gate.CAPTURE_MODES
    }
    capture_acceptance = {
        "kind": "pyamplicol-three-lane-layout-capture",
        "schema_version": 4,
        "complete": True,
        "evidence_complete": True,
        "passes": True,
        "authoritative_eligible": True,
        "authoritative_ineligibility_reasons": [],
        "generation_specialized_axes_by_mode": {},
        "incomplete_physical_axes": [],
        "required_modes": list(gate.CAPTURE_MODES),
        "observed_modes": list(gate.CAPTURE_MODES),
        "missing_modes": [],
        "required_batch_sizes": list(gate.BATCH_SIZES),
        "observed_batch_sizes": list(gate.BATCH_SIZES),
        "missing_batch_sizes": [],
        "measurement_contract": {
            "passes": True,
            "lanes": lane_measurements,
            "minimum_authoritative_samples": 7,
            "configured_internal_minimum_samples": 7,
            "configured_subprocess_samples": 7,
            "configured_warmup_runs": 2,
            "root_processes_match": True,
            "schedule": {
                "passes": True,
                "errors": [],
                "entry_count": len(schedule_entries),
                "unique_worker_command_count": len(schedule_entries),
                "subprocess_samples_per_cell": 7,
            },
        },
        "artifact_semantic_contract": {
            "passes": True,
            "errors": [],
            "lanes_match": True,
            "lane_contracts": artifact_lane_contracts,
            "common_physics_contract": {"synthetic": True},
        },
        "layout": layout,
        "generation_only": False,
        "lane_self_validation_passes": True,
        "pairwise_validation_passes": True,
    }
    return {
        "kind": gate.CAPTURE_KIND,
        "schema_version": gate.MINIMUM_CAPTURE_SCHEMA,
        "complete": True,
        "passes": True,
        "capture_acceptance": capture_acceptance,
        "process": process,
        "process_name": ("ddbar_Z_6g" if process.startswith("d ") else "uubar_Z_6g"),
        "workload": workload,
        "source": {
            "revision": revision,
            "dirty": False,
            "untracked_files_checked": True,
        },
        "provenance": {
            "host": {
                "system": "TestOS",
                "machine": "test64",
                "cpu_model": "synthetic",
                "logical_cpu_count": 8,
            }
        },
        "runtime_provenance": {
            "interpreter": {
                "implementation": "CPython",
                "python_version": "3.12.6",
                "sha256": "1" * 64,
            },
            "native_extension": {
                "sha256": native_sha,
                "build_inputs_sha256": build_inputs_sha,
                "package_version": "0.1.0+candidate",
            },
            "active_build_info": {
                "payload": {
                    "source_revision": revision,
                    "version": "0.1.0+candidate",
                }
            },
        },
        "configuration": {
            "batch_sizes": list(gate.BATCH_SIZES),
            "target_runtime_seconds": 5.0,
            "minimum_samples": 7,
            "subprocess_samples": 7,
            "warmup_runs": 2,
            "color_flow_request": color_request,
            "helicity_request": helicity_request,
            "lc_flow_layout": layout,
            "validation_seed": 12345,
            "point_tile_size": 1024,
            "jit_optimization_level": 3,
            "generation_only": False,
            "modes": list(gate.CAPTURE_MODES),
        },
        "generation": generation,
        "profile_schedule": {
            "kind": "pyamplicol-interleaved-subprocess-profile-schedule",
            "schema_version": 2,
            "algorithm": "round-major-cyclic-mode-and-batch-interleave-v1",
            "sample_unit": "independent-profile-worker-subprocess",
            "subprocess_samples_per_cell": 7,
            "modes": list(gate.CAPTURE_MODES),
            "batch_sizes": list(gate.BATCH_SIZES),
            "entries": schedule_entries,
        },
        "profiles": profiles,
        "validation_summary": {
            "passes": True,
            "selectors_match": True,
            "fixtures_match": True,
            "lane_validation_passes": True,
            "pairwise_validation_passes": True,
        },
        "selector_contracts_match": True,
        "validation_fixtures_match": True,
    }


def _write_campaign(
    tmp_path: Path,
    *,
    process: str = "d d~ > Z g g g g g g",
    runtime_ratio: float = 1.0,
    compiled_to_recurrence_ratio: float = 1.05,
    generation_ratio: float = 1.0,
    resource_ratio: float = 1.0,
    eager_runtime_ratio: float = 1.0,
    eager_generation_ratio: float = 1.0,
    eager_resource_ratio: float = 1.0,
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for role in ("baseline", "candidate"):
        for layout in gate.LAYOUTS:
            payload = _capture(
                role=role,
                layout=layout,
                process=process,
                runtime_ratio=runtime_ratio,
                compiled_to_recurrence_ratio=compiled_to_recurrence_ratio,
                generation_ratio=generation_ratio,
                resource_ratio=resource_ratio,
                eager_runtime_ratio=eager_runtime_ratio,
                eager_generation_ratio=eager_generation_ratio,
                eager_resource_ratio=eager_resource_ratio,
            )
            path = tmp_path / f"{role}-{layout}.json"
            path.write_bytes(_canonical(payload) + b"\n")
            result[f"{role}-{layout}"] = path
    return result


def _compare(paths: dict[str, Path]) -> dict[str, object]:
    return gate.compare_capture_files(
        baseline_topology=paths["baseline-topology-replay"],
        baseline_all_flow_union=paths["baseline-all-flow-union"],
        candidate_topology=paths["candidate-topology-replay"],
        candidate_all_flow_union=paths["candidate-all-flow-union"],
    )


@pytest.mark.parametrize(
    "process",
    (
        "d d~ > z + 6*g",
        "d d~ > Z g g g g g g",
        "u u~ > Z g g g g g g",
    ),
)
def test_gate_accepts_exact_d_route_and_existing_u_route(
    tmp_path: Path,
    process: str,
) -> None:
    comparison = _compare(_write_campaign(tmp_path, process=process))
    assert comparison["complete"] is True
    assert comparison["passes"] is True
    assert comparison["failures"] == []
    unsigned = dict(comparison)
    digest = unsigned.pop("content_sha256")
    assert digest == _sha256(unsigned)
    assert len(comparison["runtime_cells"]) == 12
    assert len(comparison["paired_compiled_recurrence_ratios"]) == 4
    assert len(comparison["eager_diagnostic"]["runtime_cells"]) == 6
    assert comparison["eager_diagnostic"]["passes"] is True
    assert (
        comparison["eager_diagnostic"]["paired_compiled_recurrence_ratio_applicable"]
        is False
    )


def test_gate_recomputes_runtime_statistics_and_rejects_forged_aggregate(
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-topology-replay"]
    payload = json.loads(path.read_text())
    aggregate = payload["profiles"]["compiled"]["profiles"][0]
    aggregate["wall_seconds_per_point_median"] *= 0.5
    path.write_bytes(_canonical(payload) + b"\n")
    with pytest.raises(gate.EvidenceError, match="not reproducible"):
        _compare(paths)


def test_gate_independently_revalidates_eager_raw_samples(tmp_path: Path) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-topology-replay"]
    payload = json.loads(path.read_text())
    aggregate = payload["profiles"]["eager"]["profiles"][0]
    aggregate["wall_seconds_per_point_median"] *= 0.5
    path.write_bytes(_canonical(payload) + b"\n")
    with pytest.raises(gate.EvidenceError, match="not reproducible"):
        _compare(paths)


def test_gate_reports_runtime_and_same_round_ratio_failures(tmp_path: Path) -> None:
    runtime = _compare(_write_campaign(tmp_path / "runtime", runtime_ratio=1.04))
    assert runtime["passes"] is False
    assert any(
        failure.startswith("runtime regression:") for failure in runtime["failures"]
    )

    ratio = _compare(
        _write_campaign(
            tmp_path / "ratio",
            compiled_to_recurrence_ratio=1.20,
        )
    )
    assert ratio["passes"] is False
    assert any(
        failure.startswith("compiled/recurrence ratio regression:")
        for failure in ratio["failures"]
    )


def test_gate_enforces_generation_cell_and_geometric_mean_limits(
    tmp_path: Path,
) -> None:
    comparison = _compare(_write_campaign(tmp_path, generation_ratio=1.11))
    assert comparison["passes"] is False
    assert comparison["generation"]["geometric_mean_passes"] is False
    assert "generation geometric-mean regression" in comparison["failures"]
    assert (
        sum(
            failure.startswith("generation regression:")
            for failure in comparison["failures"]
        )
        == 4
    )


def test_eager_diagnostic_runtime_generation_and_resource_failures(
    tmp_path: Path,
) -> None:
    runtime = _compare(
        _write_campaign(
            tmp_path / "runtime",
            eager_runtime_ratio=1.04,
        )
    )
    assert runtime["passes"] is False
    assert all(cell["passes"] is True for cell in runtime["runtime_cells"])
    assert runtime["eager_diagnostic"]["runtime_passes"] is False
    assert any(
        failure.startswith("eager diagnostic runtime regression:")
        for failure in runtime["failures"]
    )

    generation = _compare(
        _write_campaign(
            tmp_path / "generation",
            eager_generation_ratio=1.11,
        )
    )
    assert generation["passes"] is False
    assert generation["generation"]["geometric_mean_passes"] is True
    assert generation["eager_diagnostic"]["generation"]["passes"] is False
    assert any(
        failure.startswith("eager diagnostic generation regression:")
        for failure in generation["failures"]
    )

    resources = _compare(
        _write_campaign(
            tmp_path / "resources",
            eager_resource_ratio=1.04,
        )
    )
    assert resources["passes"] is False
    assert resources["eager_diagnostic"]["resource_passes"] is False
    assert any(
        failure.startswith("eager diagnostic resource regression:")
        for failure in resources["failures"]
    )


def test_eager_resource_growth_uses_the_runtime_gain_exception(
    tmp_path: Path,
) -> None:
    comparison = _compare(
        _write_campaign(
            tmp_path,
            eager_runtime_ratio=0.85,
            eager_resource_ratio=1.20,
        )
    )
    eager = comparison["eager_diagnostic"]
    assert comparison["passes"] is True
    assert eager["passes"] is True
    assert any(
        resource["within_3_percent_growth_limit"] is False
        and resource["runtime_gain_exception"]["passes"] is True
        for resource in eager["payload_cold_load_rss"]
    )


def test_resource_growth_needs_a_ten_percent_noise_clear_runtime_gain(
    tmp_path: Path,
) -> None:
    regression = _compare(_write_campaign(tmp_path / "regression", resource_ratio=1.04))
    assert regression["passes"] is False
    assert any(
        failure.startswith("resource regression:") for failure in regression["failures"]
    )

    exception = _compare(
        _write_campaign(
            tmp_path / "exception",
            runtime_ratio=0.85,
            resource_ratio=1.20,
        )
    )
    assert exception["passes"] is True
    assert all(
        resource["passes"] is True for resource in exception["payload_cold_load_rss"]
    )
    assert any(
        resource["within_3_percent_growth_limit"] is False
        and resource["runtime_gain_exception"]["passes"] is True
        for resource in exception["payload_cold_load_rss"]
    )


def test_gate_fails_closed_on_workload_identity_drift(tmp_path: Path) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-all-flow-union"]
    payload = json.loads(path.read_text())
    payload["profiles"]["compiled"]["selector_contract"]["color_flow_count"] = 3
    path.write_bytes(_canonical(payload) + b"\n")
    with pytest.raises(gate.EvidenceError, match="workload identities differ"):
        _compare(paths)


@pytest.mark.parametrize(
    ("field", "value"),
    (("complete", False), ("passes", None)),
)
def test_gate_requires_top_level_complete_passing_captures(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-topology-replay"]
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_bytes(_canonical(payload) + b"\n")
    with pytest.raises(
        gate.EvidenceError,
        match="not top-level complete and passing",
    ):
        _compare(paths)


def test_gate_requires_consistent_authoritative_capture_acceptance(
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["candidate-all-flow-union"]
    payload = json.loads(path.read_text())
    payload["capture_acceptance"]["evidence_complete"] = False
    path.write_bytes(_canonical(payload) + b"\n")
    with pytest.raises(
        gate.EvidenceError,
        match="not authoritative, complete, and passing",
    ):
        _compare(paths)


def test_legacy_capture_without_cold_load_has_actionable_error(
    tmp_path: Path,
) -> None:
    paths = _write_campaign(tmp_path)
    path = paths["baseline-topology-replay"]
    payload = json.loads(path.read_text())
    del payload["profiles"]["compiled"]["profiles"][0]["subprocess_samples"][0][
        "cold_load_seconds"
    ]
    path.write_bytes(_canonical(payload) + b"\n")
    with pytest.raises(
        gate.EvidenceError,
        match=r"predates the resource-evidence contract.*re-run",
    ):
        _compare(paths)


def test_cli_writes_canonical_content_addressed_report(tmp_path: Path) -> None:
    paths = _write_campaign(tmp_path)
    output = tmp_path / "comparison.json"
    exit_code = gate.main(
        [
            "--baseline-topology",
            str(paths["baseline-topology-replay"]),
            "--baseline-all-flow-union",
            str(paths["baseline-all-flow-union"]),
            "--candidate-topology",
            str(paths["candidate-topology-replay"]),
            "--candidate-all-flow-union",
            str(paths["candidate-all-flow-union"]),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    payload = json.loads(output.read_text())
    unsigned = copy.deepcopy(payload)
    digest = unsigned.pop("content_sha256")
    assert digest == _sha256(unsigned)
    assert output.read_bytes() == _canonical(payload) + b"\n"
