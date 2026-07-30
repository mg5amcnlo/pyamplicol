#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Compare authenticated SymJIT 2.22 migration benchmark captures.

The recurrence z+6g harness deliberately records raw, independently warmed
subprocess measurements.  This gate consumes one topology-replay and one
all-flow-union capture for both the immutable pre-migration baseline and the
candidate.  It validates their workload identities, recomputes every statistic
used for acceptance from the retained raw samples, and emits one
content-addressed comparison record.

This is a focused migration gate.  It requires the compiled and recurrence
lanes only; the separately requested eager campaign remains diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CAPTURE_KIND = "pyamplicol-recurrence-z6g-benchmark"
MINIMUM_CAPTURE_SCHEMA = 6
COMPARISON_KIND = "pyamplicol-symjit-2.22-migration-benchmark-comparison"
COMPARISON_SCHEMA = 1
BASELINE_REVISION = "172e58fd33a3c65563866c50cfbb5e1ddcd7b302"

LAYOUTS = ("topology-replay", "all-flow-union")
MODES = ("compiled", "recurrence")
CAPTURE_MODES = ("compiled", "eager", "recurrence")
BATCH_SIZES = (1, 128, 1024)
RATIO_BATCH_SIZES = (128, 1024)
MINIMUM_SUBPROCESS_SAMPLES = 7
MINIMUM_INTERNAL_SAMPLES = 7
MINIMUM_WARMUPS = 2
MINIMUM_TARGET_RUNTIME_SECONDS = 5.0

RUNTIME_RELATIVE_LIMIT = 1.03
RUNTIME_NOISE_MAD_MULTIPLIER = 3.0
COMPILED_RECURRENCE_RATIO_LIMIT = 1.15
GENERATION_CELL_LIMIT = 1.10
GENERATION_GEOMEAN_LIMIT = 1.05
RESOURCE_GROWTH_LIMIT = 1.03
RESOURCE_EXCEPTION_GAIN = 0.10

_SHA256_LENGTH = 64
_REVISION_LENGTH = 40


class EvidenceError(RuntimeError):
    """Raised when a capture cannot support an acceptance decision."""


@dataclass(frozen=True)
class Distribution:
    """Raw values and statistics recomputed by this gate."""

    values: tuple[float, ...]
    median: float
    raw_mad: float

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_count": len(self.values),
            "values": list(self.values),
            "median": self.median,
            "raw_mad": self.raw_mad,
        }


@dataclass(frozen=True)
class CellEvidence:
    """One validated mode/batch cell from a capture."""

    runtime: Distribution
    cold_load: Distribution
    cold_load_rss: Distribution
    profiled_rss: Distribution
    by_round: Mapping[int, float]


@dataclass(frozen=True)
class GenerationEvidence:
    """Fresh generation metrics for one mode."""

    wall_seconds: float
    payload_size_bytes: int
    peak_rss_bytes: int


@dataclass(frozen=True)
class CaptureEvidence:
    """Validated evidence extracted from one recurrence-z6g capture."""

    input_identity: Mapping[str, object]
    layout: str
    logical_process: str
    source_revision: str
    build_identity: Mapping[str, object]
    host_identity: Mapping[str, object]
    campaign_identity: Mapping[str, object]
    workload_identities: Mapping[str, Mapping[str, object]]
    cells: Mapping[tuple[str, int], CellEvidence]
    generation: Mapping[str, GenerationEvidence]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise EvidenceError("evidence is not canonical-JSON serializable") from error


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise EvidenceError(f"cannot read capture: {path}") from error
    return digest.hexdigest()


def _input_identity(path: Path) -> dict[str, object]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        stat = resolved.stat()
    except OSError as error:
        raise EvidenceError(f"cannot inspect capture: {path}") from error
    if not resolved.is_file():
        raise EvidenceError(f"capture is not a regular file: {path}")
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(resolved),
    }


def _read_capture(path: Path) -> tuple[dict[str, Any], dict[str, object]]:
    identity = _input_identity(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"capture is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"capture root is not an object: {path}")
    return value, identity


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{label} is missing or is not an object")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise EvidenceError(f"{label} is missing or is not an array")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{label} is missing or is not a non-empty string")
    return value


def _exact_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"{label} is missing or is not an integer")
    if minimum is not None and value < minimum:
        raise EvidenceError(f"{label} must be at least {minimum}")
    return value


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} is missing or is not numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise EvidenceError(f"{label} must be a positive finite number")
    return result


def _nonnegative_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} is missing or is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise EvidenceError(f"{label} must be a finite non-negative number")
    return result


def _valid_sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if len(result) != _SHA256_LENGTH:
        raise EvidenceError(f"{label} is not a SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise EvidenceError(f"{label} is not a SHA-256 digest") from error
    if result != result.lower():
        raise EvidenceError(f"{label} is not a lowercase SHA-256 digest")
    return result


def _validate_content_address(value: object, label: str) -> Mapping[str, Any]:
    record = _mapping(value, label)
    unsigned = dict(record)
    digest = _valid_sha256(
        unsigned.pop("content_sha256", None), f"{label}.content_sha256"
    )
    if digest != _sha256(unsigned):
        raise EvidenceError(f"{label} content address does not match its payload")
    return record


def _distribution(values: Sequence[float], label: str) -> Distribution:
    if len(values) < MINIMUM_SUBPROCESS_SAMPLES:
        raise EvidenceError(
            f"{label} has {len(values)} subprocess samples; "
            f"at least {MINIMUM_SUBPROCESS_SAMPLES} are required"
        )
    checked = tuple(_positive_number(value, f"{label} raw sample") for value in values)
    median = float(statistics.median(checked))
    raw_mad = float(statistics.median(abs(value - median) for value in checked))
    return Distribution(values=checked, median=median, raw_mad=raw_mad)


def _rss_observation(value: object, label: str) -> float:
    peak = _mapping(value, label)
    if peak.get("source") != "resource.getrusage":
        raise EvidenceError(f"{label} has an unsupported RSS source")
    self_peak = _exact_int(
        peak.get("self_peak_bytes"), f"{label}.self_peak_bytes", minimum=0
    )
    child_peak = _exact_int(
        peak.get("maximum_child_peak_bytes"),
        f"{label}.maximum_child_peak_bytes",
        minimum=0,
    )
    observed = _exact_int(
        peak.get("observed_lower_bound_bytes"),
        f"{label}.observed_lower_bound_bytes",
        minimum=1,
    )
    if observed != max(self_peak, child_peak):
        raise EvidenceError(f"{label} is not a consistent resource peak")
    return float(observed)


def _recorded_statistic(
    observed: object,
    expected: float,
    label: str,
) -> None:
    value = _nonnegative_number(observed, label)
    if not math.isclose(value, expected, rel_tol=1.0e-14, abs_tol=0.0):
        raise EvidenceError(
            f"{label} is not reproducible from retained raw subprocess samples"
        )


def _logical_process(value: object, label: str) -> str:
    normalized = " ".join(_string(value, label).split()).casefold()
    supported = {
        "u u~ > z g g g g g g": "u u~ > Z g g g g g g",
        "d d~ > z + 6*g": "d d~ > Z g g g g g g",
        "d d~ > z g g g g g g": "d d~ > Z g g g g g g",
    }
    try:
        return supported[normalized]
    except KeyError as error:
        raise EvidenceError(
            f"{label} is not the supported u u~ or exact d d~ > Z + 6g route"
        ) from error


def _validate_capture_acceptance(
    payload: Mapping[str, Any],
    configuration: Mapping[str, Any],
    *,
    expected_layout: str,
    subprocess_samples: int,
    internal_samples: int,
    warmups: int,
) -> Mapping[str, Any]:
    label = f"{expected_layout}.capture_acceptance"
    if payload.get("complete") is not True or payload.get("passes") is not True:
        raise EvidenceError(
            f"{expected_layout} capture is not top-level complete and passing"
        )
    acceptance = _mapping(payload.get("capture_acceptance"), label)
    if (
        acceptance.get("kind") != "pyamplicol-three-lane-layout-capture"
        or _exact_int(
            acceptance.get("schema_version"),
            f"{label}.schema_version",
        )
        < 4
    ):
        raise EvidenceError(f"{label} has an unsupported identity")
    required_true = (
        "complete",
        "evidence_complete",
        "passes",
        "authoritative_eligible",
        "lane_self_validation_passes",
        "pairwise_validation_passes",
    )
    if any(acceptance.get(field) is not True for field in required_true):
        raise EvidenceError(f"{label} is not authoritative, complete, and passing")
    if (
        acceptance.get("authoritative_ineligibility_reasons") != []
        or acceptance.get("generation_specialized_axes_by_mode") != {}
        or acceptance.get("incomplete_physical_axes") != []
        or acceptance.get("missing_modes") != []
        or acceptance.get("missing_batch_sizes") != []
        or acceptance.get("generation_only") is not False
        or acceptance.get("layout") != expected_layout
    ):
        raise EvidenceError(f"{label} contains authoritative ineligibility evidence")
    if (
        list(
            _sequence(
                acceptance.get("required_modes"),
                f"{label}.required_modes",
            )
        )
        != list(CAPTURE_MODES)
        or list(
            _sequence(
                acceptance.get("observed_modes"),
                f"{label}.observed_modes",
            )
        )
        != list(CAPTURE_MODES)
        or set(
            _sequence(
                acceptance.get("required_batch_sizes"),
                f"{label}.required_batch_sizes",
            )
        )
        != set(BATCH_SIZES)
        or set(
            _sequence(
                acceptance.get("observed_batch_sizes"),
                f"{label}.observed_batch_sizes",
            )
        )
        != set(BATCH_SIZES)
    ):
        raise EvidenceError(f"{label} required/observed inventory is inconsistent")

    measurement = _mapping(
        acceptance.get("measurement_contract"),
        f"{label}.measurement_contract",
    )
    if (
        measurement.get("passes") is not True
        or measurement.get("root_processes_match") is not True
        or measurement.get("configured_internal_minimum_samples") != internal_samples
        or measurement.get("configured_subprocess_samples") != subprocess_samples
        or measurement.get("configured_warmup_runs") != warmups
        or _exact_int(
            measurement.get("minimum_authoritative_samples"),
            f"{label}.measurement_contract.minimum_authoritative_samples",
            minimum=MINIMUM_SUBPROCESS_SAMPLES,
        )
        > subprocess_samples
    ):
        raise EvidenceError(f"{label} measurement contract is inconsistent")
    schedule_contract = _mapping(
        measurement.get("schedule"),
        f"{label}.measurement_contract.schedule",
    )
    expected_entry_count = len(CAPTURE_MODES) * len(BATCH_SIZES) * subprocess_samples
    if (
        schedule_contract.get("passes") is not True
        or schedule_contract.get("errors") != []
        or schedule_contract.get("entry_count") != expected_entry_count
        or schedule_contract.get("unique_worker_command_count") != expected_entry_count
        or schedule_contract.get("subprocess_samples_per_cell") != subprocess_samples
    ):
        raise EvidenceError(f"{label} schedule contract is incomplete")
    lane_contracts = _mapping(
        measurement.get("lanes"),
        f"{label}.measurement_contract.lanes",
    )
    if set(lane_contracts) != set(CAPTURE_MODES):
        raise EvidenceError(f"{label} measurement lane inventory is inconsistent")
    for mode in CAPTURE_MODES:
        lane = _mapping(
            lane_contracts.get(mode),
            f"{label}.measurement_contract.lanes.{mode}",
        )
        if (
            lane.get("passes") is not True
            or lane.get("errors") != []
            or set(
                _sequence(
                    lane.get("observed_batch_sizes"),
                    f"{label}.measurement_contract.lanes.{mode}.observed_batch_sizes",
                )
            )
            != set(BATCH_SIZES)
            or lane.get("missing_batch_sizes") != []
        ):
            raise EvidenceError(f"{label} {mode} measurement lane is incomplete")

    semantic_contract = _mapping(
        acceptance.get("artifact_semantic_contract"),
        f"{label}.artifact_semantic_contract",
    )
    semantic_lanes = _mapping(
        semantic_contract.get("lane_contracts"),
        f"{label}.artifact_semantic_contract.lane_contracts",
    )
    if (
        semantic_contract.get("passes") is not True
        or semantic_contract.get("errors") != []
        or semantic_contract.get("lanes_match") is not True
        or not isinstance(semantic_contract.get("common_physics_contract"), Mapping)
        or set(semantic_lanes) != set(CAPTURE_MODES)
    ):
        raise EvidenceError(f"{label} artifact-semantic contract is incomplete")

    validation = _mapping(payload.get("validation_summary"), "validation_summary")
    if (
        validation.get("passes") is not True
        or validation.get("selectors_match") is not True
        or validation.get("fixtures_match") is not True
        or validation.get("lane_validation_passes") is not True
        or validation.get("pairwise_validation_passes") is not True
        or payload.get("selector_contracts_match") is not True
        or payload.get("validation_fixtures_match") is not True
    ):
        raise EvidenceError(
            f"{expected_layout} capture validation summary is not passing"
        )
    if set(
        _sequence(
            configuration.get("modes"),
            f"{expected_layout}.configuration.modes",
        )
    ) != set(CAPTURE_MODES):
        raise EvidenceError(f"{label} does not match the configured lane inventory")
    return acceptance


def _semantic_digest(
    semantic: Mapping[str, Any],
    field: str,
    digest_field: str,
    label: str,
) -> tuple[object, str]:
    value = semantic.get(field)
    if value is None:
        raise EvidenceError(f"{label}.{field} is missing")
    digest = _valid_sha256(
        semantic.get(digest_field),
        f"{label}.{digest_field}",
    )
    if digest != _sha256(value):
        raise EvidenceError(f"{label}.{digest_field} does not match {field}")
    return value, digest


def _workload_identity(
    *,
    profile: Mapping[str, Any],
    generation: Mapping[str, Any],
    mode: str,
    layout: str,
    logical_process: str,
    expected_workload: str,
    label: str,
) -> dict[str, object]:
    if profile.get("mode") != mode:
        raise EvidenceError(f"{label}.mode does not match its lane")
    if (
        _logical_process(
            profile.get("process_expression"), f"{label}.process_expression"
        )
        != logical_process
    ):
        raise EvidenceError(f"{label} process expression drifted from its capture")
    process_id = _string(profile.get("process_id"), f"{label}.process_id")
    selector = _mapping(profile.get("selector_contract"), f"{label}.selector_contract")
    if selector.get("workload") != expected_workload:
        raise EvidenceError(f"{label}.selector_contract has the wrong workload")
    validation = _mapping(profile.get("validation"), f"{label}.validation")
    if validation.get("passes") is not True:
        raise EvidenceError(f"{label} numerical validation did not pass")
    fixture = _mapping(validation.get("fixture"), f"{label}.validation.fixture")
    point_count = _exact_int(
        fixture.get("point_count"),
        f"{label}.validation.fixture.point_count",
        minimum=1,
    )
    points_sha256 = _valid_sha256(
        fixture.get("points_sha256"),
        f"{label}.validation.fixture.points_sha256",
    )

    semantic = _mapping(
        profile.get("artifact_semantic_identity"),
        f"{label}.artifact_semantic_identity",
    )
    semantic_digest = _valid_sha256(
        profile.get("artifact_semantic_identity_sha256"),
        f"{label}.artifact_semantic_identity_sha256",
    )
    if semantic_digest != _sha256(semantic):
        raise EvidenceError(
            f"{label} artifact semantic identity is not content-addressed"
        )
    generation_identity = _mapping(
        generation.get("artifact_identity"),
        f"{label}.generation.artifact_identity",
    )
    if (
        generation_identity.get("semantic_identity") != semantic
        or generation_identity.get("semantic_identity_sha256") != semantic_digest
    ):
        raise EvidenceError(
            f"{label} profile and generation artifact identities differ"
        )
    if semantic.get("generation_specialized_axes") != []:
        raise EvidenceError(f"{label} has generation-specialized physical axes")
    coverage = _mapping(semantic.get("coverage"), f"{label}.coverage")
    if coverage.get("complete_physical_axes") is not True:
        raise EvidenceError(f"{label} does not contain complete physical axes")
    reduction_coverage = _mapping(
        semantic.get("reduction_coverage"),
        f"{label}.reduction_coverage",
    )
    if reduction_coverage.get("complete") is not True:
        raise EvidenceError(f"{label} reduction coverage is incomplete")

    normalization, normalization_sha256 = _semantic_digest(
        semantic,
        "normalization",
        "normalization_sha256",
        label,
    )
    reduction_ordering, reduction_ordering_sha256 = _semantic_digest(
        semantic,
        "reduction_ordering",
        "reduction_ordering_sha256",
        label,
    )
    selector_semantics, selector_semantics_sha256 = _semantic_digest(
        semantic,
        "runtime_selector_semantics",
        "runtime_selector_semantics_sha256",
        label,
    )
    model_identity = _mapping(
        semantic.get("manifest_model_identity"),
        f"{label}.manifest_model_identity",
    )
    common_model = _mapping(
        model_identity.get("common_physics_identity"),
        f"{label}.manifest_model_identity.common_physics_identity",
    )
    common_model_sha256 = _valid_sha256(
        model_identity.get("common_physics_identity_sha256"),
        f"{label}.manifest_model_identity.common_physics_identity_sha256",
    )
    if common_model_sha256 != _sha256(common_model):
        raise EvidenceError(f"{label} common model identity is not content-addressed")
    physical_color_flows = semantic.get("physical_color_flows")
    physical_helicities = semantic.get("physical_helicities")
    if physical_color_flows is None or physical_helicities is None:
        raise EvidenceError(f"{label} physical-axis identity is incomplete")

    effective = _mapping(
        generation.get("effective_contract"),
        f"{label}.generation.effective_contract",
    )
    expected_optimization = 2 if mode == "recurrence" else 3
    expected_effective = {
        "backend": "jit",
        "color_accuracy": "lc",
        "execution_mode": mode,
        "jit_optimization_level": expected_optimization,
        "lc_flow_layout": layout,
    }
    if any(effective.get(key) != value for key, value in expected_effective.items()):
        raise EvidenceError(
            f"{label} does not use {mode} JIT O{expected_optimization} "
            f"with the requested {layout} layout"
        )

    # Store digests for the large physical inventories so the comparison
    # remains compact while still binding every ordered entry.
    return {
        "process": logical_process,
        "process_id": process_id,
        "layout": layout,
        "mode": mode,
        "workload": expected_workload,
        "selector_contract_sha256": _sha256(selector),
        "validation_fixture": {
            "point_count": point_count,
            "points_sha256": points_sha256,
        },
        "common_model_identity_sha256": common_model_sha256,
        "normalization_sha256": normalization_sha256,
        "normalization_payload_sha256": _sha256(normalization),
        "reduction_ordering_sha256": reduction_ordering_sha256,
        "reduction_ordering_payload_sha256": _sha256(reduction_ordering),
        "runtime_selector_semantics_sha256": selector_semantics_sha256,
        "runtime_selector_semantics_payload_sha256": _sha256(selector_semantics),
        "physical_color_flows_sha256": _sha256(physical_color_flows),
        "physical_helicities_sha256": _sha256(physical_helicities),
        "coverage_sha256": _sha256(coverage),
        "effective_contract": expected_effective,
    }


def _validate_sample_address(
    sample: Mapping[str, Any],
    schedule_entry: Mapping[str, Any],
    label: str,
    seen_addresses: set[str],
) -> None:
    invocation = _validate_content_address(
        sample.get("worker_invocation"), f"{label}.worker_invocation"
    )
    process_record = _validate_content_address(
        sample.get("worker_process_record"),
        f"{label}.worker_process_record",
    )
    result_record = _validate_content_address(
        sample.get("worker_result_record"),
        f"{label}.worker_result_record",
    )
    invocation_sha256 = _valid_sha256(
        invocation.get("content_sha256"),
        f"{label}.worker_invocation.content_sha256",
    )
    process_sha256 = _valid_sha256(
        process_record.get("content_sha256"),
        f"{label}.worker_process_record.content_sha256",
    )
    result_sha256 = _valid_sha256(
        result_record.get("content_sha256"),
        f"{label}.worker_result_record.content_sha256",
    )
    for address_name, address in (
        ("invocation", invocation_sha256),
        ("process", process_sha256),
        ("result", result_sha256),
    ):
        scoped = f"{address_name}:{address}"
        if scoped in seen_addresses:
            raise EvidenceError(f"{label} reuses a {address_name} subprocess identity")
        seen_addresses.add(scoped)
    if (
        result_record.get("worker_invocation_sha256") != invocation_sha256
        or result_record.get("worker_process_record_sha256") != process_sha256
    ):
        raise EvidenceError(f"{label} worker result is not bound to its subprocess")
    unsigned_sample = {
        str(key): value
        for key, value in sample.items()
        if key != "worker_result_record"
    }
    addressed_payload = _valid_sha256(
        result_record.get("addressed_payload_sha256"),
        f"{label}.worker_result_record.addressed_payload_sha256",
    )
    if addressed_payload != _sha256(unsigned_sample):
        raise EvidenceError(f"{label} retained raw sample content address is invalid")
    if schedule_entry.get("worker_result_sha256") != addressed_payload:
        raise EvidenceError(f"{label} is not bound to its profile schedule entry")


def _extract_cell(
    *,
    aggregate: Mapping[str, Any],
    mode: str,
    batch_size: int,
    subprocess_samples: int,
    warmups: int,
    target_runtime: float,
    schedule_by_index: Mapping[int, Mapping[str, Any]],
    seen_addresses: set[str],
    label: str,
) -> CellEvidence:
    if aggregate.get("batch_size") != batch_size:
        raise EvidenceError(f"{label}.batch_size does not match its inventory slot")
    samples = _sequence(
        aggregate.get("subprocess_samples"), f"{label}.subprocess_samples"
    )
    if len(samples) != subprocess_samples:
        raise EvidenceError(
            f"{label} has {len(samples)} subprocess samples, but its campaign "
            f"declares {subprocess_samples}"
        )
    runtime_values: list[float] = []
    cold_values: list[float] = []
    cold_rss_values: list[float] = []
    profiled_rss_values: list[float] = []
    by_round: dict[int, float] = {}
    for sample_index, raw_sample in enumerate(samples):
        sample_label = f"{label}.subprocess_samples[{sample_index}]"
        sample = _mapping(raw_sample, sample_label)
        round_index = _exact_int(
            sample.get("round"), f"{sample_label}.round", minimum=0
        )
        if round_index in by_round:
            raise EvidenceError(f"{label} duplicates subprocess round {round_index}")
        schedule_index = _exact_int(
            sample.get("schedule_index"),
            f"{sample_label}.schedule_index",
            minimum=0,
        )
        schedule_entry = schedule_by_index.get(schedule_index)
        if schedule_entry is None:
            raise EvidenceError(
                f"{sample_label} has no matching profile schedule entry"
            )
        if (
            schedule_entry.get("mode") != mode
            or schedule_entry.get("batch_size") != batch_size
            or schedule_entry.get("round") != round_index
        ):
            raise EvidenceError(f"{sample_label} does not match its schedule slot")
        if sample.get("interrupted") is not False:
            raise EvidenceError(f"{sample_label} was interrupted")
        timing = _mapping(
            sample.get("timing_configuration"),
            f"{sample_label}.timing_configuration",
        )
        sample_warmups = _exact_int(
            timing.get("warmup_runs"),
            f"{sample_label}.timing_configuration.warmup_runs",
            minimum=MINIMUM_WARMUPS,
        )
        sample_target = _positive_number(
            timing.get("target_runtime_seconds"),
            f"{sample_label}.timing_configuration.target_runtime_seconds",
        )
        if sample_warmups != warmups or sample_target != target_runtime:
            raise EvidenceError(f"{sample_label} timing configuration drifted")
        _exact_int(
            sample.get("internal_sample_count"),
            f"{sample_label}.internal_sample_count",
            minimum=MINIMUM_INTERNAL_SAMPLES,
        )
        sources = _mapping(
            sample.get("timing_sources"), f"{sample_label}.timing_sources"
        )
        if sources.get("wall") != "runtime_core_repeated_wall_time":
            raise EvidenceError(
                f"{sample_label} is not an unprofiled native-wall sample"
            )

        runtime = _positive_number(
            sample.get("wall_seconds_per_point"),
            f"{sample_label}.wall_seconds_per_point",
        )
        if "cold_load_seconds" not in sample:
            raise EvidenceError(
                f"{sample_label} lacks cold_load_seconds. The capture predates "
                "the resource-evidence contract; re-run the baseline/candidate "
                "recurrence_z6g campaign with the current harness before comparing."
            )
        cold = _positive_number(
            sample.get("cold_load_seconds"),
            f"{sample_label}.cold_load_seconds",
        )
        if (
            "peak_rss_after_cold_load" not in sample
            or "peak_rss_after_profile" not in sample
        ):
            raise EvidenceError(
                f"{sample_label} lacks cold-load/profile RSS evidence. Re-run "
                "the baseline/candidate recurrence_z6g campaign with the current "
                "harness before comparing."
            )
        cold_rss = _rss_observation(
            sample.get("peak_rss_after_cold_load"),
            f"{sample_label}.peak_rss_after_cold_load",
        )
        profiled_rss = _rss_observation(
            sample.get("peak_rss_after_profile"),
            f"{sample_label}.peak_rss_after_profile",
        )
        _validate_sample_address(sample, schedule_entry, sample_label, seen_addresses)
        runtime_values.append(runtime)
        cold_values.append(cold)
        cold_rss_values.append(cold_rss)
        profiled_rss_values.append(profiled_rss)
        by_round[round_index] = runtime
    expected_rounds = set(range(subprocess_samples))
    if set(by_round) != expected_rounds:
        raise EvidenceError(f"{label} does not retain exactly one sample per round")

    runtime_distribution = _distribution(runtime_values, f"{label}.runtime")
    cold_distribution = _distribution(cold_values, f"{label}.cold_load")
    cold_rss_distribution = _distribution(cold_rss_values, f"{label}.cold_load_rss")
    profiled_rss_distribution = _distribution(
        profiled_rss_values,
        f"{label}.profiled_rss",
    )
    if aggregate.get("statistics_contract") != "subprocess-median-and-raw-mad-v1":
        raise EvidenceError(f"{label} has an unsupported statistics contract")
    if (
        aggregate.get("resource_statistics_contract")
        != "subprocess-median-and-raw-mad-v1"
    ):
        raise EvidenceError(f"{label} has an unsupported resource statistics contract")
    if aggregate.get("interrupted") is not False:
        raise EvidenceError(f"{label} aggregate is interrupted")
    if (
        aggregate.get("sample_count") != subprocess_samples
        or aggregate.get("subprocess_sample_count") != subprocess_samples
    ):
        raise EvidenceError(f"{label} aggregate sample count is inconsistent")
    _recorded_statistic(
        aggregate.get("wall_seconds_per_point_median"),
        runtime_distribution.median,
        f"{label}.wall_seconds_per_point_median",
    )
    _recorded_statistic(
        aggregate.get("wall_seconds_per_point_mad"),
        runtime_distribution.raw_mad,
        f"{label}.wall_seconds_per_point_mad",
    )
    _recorded_statistic(
        aggregate.get("cold_load_seconds_median"),
        cold_distribution.median,
        f"{label}.cold_load_seconds_median",
    )
    _recorded_statistic(
        aggregate.get("cold_load_seconds_mad"),
        cold_distribution.raw_mad,
        f"{label}.cold_load_seconds_mad",
    )
    _recorded_statistic(
        aggregate.get("cold_load_peak_rss_bytes_median"),
        cold_rss_distribution.median,
        f"{label}.cold_load_peak_rss_bytes_median",
    )
    _recorded_statistic(
        aggregate.get("profiled_peak_rss_bytes_median"),
        profiled_rss_distribution.median,
        f"{label}.profiled_peak_rss_bytes_median",
    )
    return CellEvidence(
        runtime=runtime_distribution,
        cold_load=cold_distribution,
        cold_load_rss=cold_rss_distribution,
        profiled_rss=profiled_rss_distribution,
        by_round=by_round,
    )


def _validate_capture(
    payload: Mapping[str, Any],
    input_identity: Mapping[str, object],
    *,
    role: str,
    expected_layout: str,
) -> CaptureEvidence:
    label = f"{role}/{expected_layout}"
    if payload.get("kind") != CAPTURE_KIND:
        raise EvidenceError(f"{label} has the wrong capture kind")
    schema = _exact_int(payload.get("schema_version"), f"{label}.schema_version")
    if schema < MINIMUM_CAPTURE_SCHEMA:
        raise EvidenceError(
            f"{label} schema {schema} predates the required raw-evidence contract"
        )
    logical_process = _logical_process(payload.get("process"), f"{label}.process")
    expected_workload = (
        "single-runtime-selected-flow/helicity-sum"
        if expected_layout == "topology-replay"
        else "all-flows/runtime-selected-single-helicity"
    )
    if payload.get("workload") != expected_workload:
        raise EvidenceError(f"{label} has the wrong workload")

    configuration = _mapping(payload.get("configuration"), f"{label}.configuration")
    if configuration.get("lc_flow_layout") != expected_layout:
        raise EvidenceError(f"{label} is in the wrong layout slot")
    if configuration.get("generation_only") is not False:
        raise EvidenceError(f"{label} is a generation-only capture")
    configured_modes = list(
        _sequence(configuration.get("modes"), f"{label}.configuration.modes")
    )
    if len(configured_modes) != len(CAPTURE_MODES) or set(configured_modes) != set(
        CAPTURE_MODES
    ):
        raise EvidenceError(
            f"{label} must contain the complete compiled, eager, and recurrence lanes"
        )
    configured_batches = tuple(
        _exact_int(value, f"{label}.configuration.batch_sizes")
        for value in _sequence(
            configuration.get("batch_sizes"),
            f"{label}.configuration.batch_sizes",
        )
    )
    if len(configured_batches) != len(BATCH_SIZES) or set(configured_batches) != set(
        BATCH_SIZES
    ):
        raise EvidenceError(f"{label} must contain exactly batches {list(BATCH_SIZES)}")
    subprocess_samples = _exact_int(
        configuration.get("subprocess_samples"),
        f"{label}.configuration.subprocess_samples",
        minimum=MINIMUM_SUBPROCESS_SAMPLES,
    )
    internal_samples = _exact_int(
        configuration.get("minimum_samples"),
        f"{label}.configuration.minimum_samples",
        minimum=MINIMUM_INTERNAL_SAMPLES,
    )
    warmups = _exact_int(
        configuration.get("warmup_runs"),
        f"{label}.configuration.warmup_runs",
        minimum=MINIMUM_WARMUPS,
    )
    target_runtime = _positive_number(
        configuration.get("target_runtime_seconds"),
        f"{label}.configuration.target_runtime_seconds",
    )
    if target_runtime < MINIMUM_TARGET_RUNTIME_SECONDS:
        raise EvidenceError(
            f"{label} target runtime is below {MINIMUM_TARGET_RUNTIME_SECONDS:g}s"
        )
    if configuration.get("jit_optimization_level") != 3:
        raise EvidenceError(f"{label} compiled JIT is not configured for O3")
    acceptance = _validate_capture_acceptance(
        payload,
        configuration,
        expected_layout=expected_layout,
        subprocess_samples=subprocess_samples,
        internal_samples=internal_samples,
        warmups=warmups,
    )

    source = _mapping(payload.get("source"), f"{label}.source")
    revision = _string(source.get("revision"), f"{label}.source.revision")
    if len(revision) != _REVISION_LENGTH:
        raise EvidenceError(f"{label}.source.revision is not a full Git revision")
    try:
        int(revision, 16)
    except ValueError as error:
        raise EvidenceError(f"{label}.source.revision is not a Git revision") from error
    if (
        source.get("dirty") is not False
        or source.get("untracked_files_checked") is not True
    ):
        raise EvidenceError(f"{label} source checkout is not clean and authenticated")

    provenance = _mapping(payload.get("provenance"), f"{label}.provenance")
    host = dict(_mapping(provenance.get("host"), f"{label}.provenance.host"))
    runtime = _mapping(payload.get("runtime_provenance"), f"{label}.runtime_provenance")
    interpreter = _mapping(
        runtime.get("interpreter"), f"{label}.runtime_provenance.interpreter"
    )
    native_extension = _mapping(
        runtime.get("native_extension"),
        f"{label}.runtime_provenance.native_extension",
    )
    build_info = _mapping(
        _mapping(
            runtime.get("active_build_info"),
            f"{label}.runtime_provenance.active_build_info",
        ).get("payload"),
        f"{label}.runtime_provenance.active_build_info.payload",
    )
    if build_info.get("source_revision") != revision:
        raise EvidenceError(
            f"{label} installed build does not match its source revision"
        )
    build_identity = {
        "source_revision": revision,
        "native_extension_sha256": _valid_sha256(
            native_extension.get("sha256"),
            f"{label}.runtime_provenance.native_extension.sha256",
        ),
        "native_build_inputs_sha256": _valid_sha256(
            native_extension.get("build_inputs_sha256"),
            f"{label}.runtime_provenance.native_extension.build_inputs_sha256",
        ),
        "package_version": _string(
            native_extension.get("package_version"),
            f"{label}.runtime_provenance.native_extension.package_version",
        ),
        "build_info_version": _string(
            build_info.get("version"),
            f"{label}.runtime_provenance.active_build_info.payload.version",
        ),
    }
    interpreter_identity = {
        "implementation": _string(
            interpreter.get("implementation"),
            f"{label}.runtime_provenance.interpreter.implementation",
        ),
        "python_version": _string(
            interpreter.get("python_version"),
            f"{label}.runtime_provenance.interpreter.python_version",
        ),
        "executable_sha256": _valid_sha256(
            interpreter.get("sha256"),
            f"{label}.runtime_provenance.interpreter.sha256",
        ),
    }

    schedule = _mapping(payload.get("profile_schedule"), f"{label}.profile_schedule")
    if (
        schedule.get("kind") != "pyamplicol-interleaved-subprocess-profile-schedule"
        or schedule.get("schema_version") != 2
        or schedule.get("algorithm")
        != "round-major-cyclic-mode-and-batch-interleave-v1"
        or schedule.get("sample_unit") != "independent-profile-worker-subprocess"
        or schedule.get("subprocess_samples_per_cell") != subprocess_samples
        or set(
            _sequence(
                schedule.get("modes"),
                f"{label}.profile_schedule.modes",
            )
        )
        != set(CAPTURE_MODES)
        or set(
            _sequence(
                schedule.get("batch_sizes"),
                f"{label}.profile_schedule.batch_sizes",
            )
        )
        != set(BATCH_SIZES)
    ):
        raise EvidenceError(f"{label} profile schedule identity is inconsistent")
    entries = _sequence(schedule.get("entries"), f"{label}.profile_schedule.entries")
    expected_entry_count = len(CAPTURE_MODES) * len(BATCH_SIZES) * subprocess_samples
    if len(entries) != expected_entry_count:
        raise EvidenceError(
            f"{label} profile schedule has {len(entries)} entries; "
            f"{expected_entry_count} are required"
        )
    schedule_by_index: dict[int, Mapping[str, Any]] = {}
    for position, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"{label}.profile_schedule.entries[{position}]")
        index = _exact_int(
            entry.get("schedule_index"),
            f"{label}.profile_schedule.entries[{position}].schedule_index",
            minimum=0,
        )
        if index != position or index in schedule_by_index:
            raise EvidenceError(f"{label} profile schedule index is not canonical")
        schedule_by_index[index] = entry

    generations = _mapping(payload.get("generation"), f"{label}.generation")
    raw_profiles = _mapping(payload.get("profiles"), f"{label}.profiles")
    if set(generations) != set(CAPTURE_MODES) or set(raw_profiles) != set(
        CAPTURE_MODES
    ):
        raise EvidenceError(
            f"{label} three-lane generation/profile inventory is incomplete"
        )
    generation_evidence: dict[str, GenerationEvidence] = {}
    workload_identities: dict[str, Mapping[str, object]] = {}
    cells: dict[tuple[str, int], CellEvidence] = {}
    seen_addresses: set[str] = set()
    for mode in CAPTURE_MODES:
        generation = _mapping(generations.get(mode), f"{label}.generation.{mode}")
        if generation.get("mode") != mode:
            raise EvidenceError(f"{label}.generation.{mode} has the wrong mode")
        if generation.get("generation_reused") is not False:
            raise EvidenceError(
                f"{label}.generation.{mode} was reused; a fresh generation "
                "time is required"
            )
        generation_wall = _positive_number(
            generation.get("generation_wall_seconds"),
            f"{label}.generation.{mode}.generation_wall_seconds",
        )
        stats = _mapping(
            generation.get("artifact_stats"),
            f"{label}.generation.{mode}.artifact_stats",
        )
        _exact_int(
            stats.get("file_count"),
            f"{label}.generation.{mode}.artifact_stats.file_count",
            minimum=1,
        )
        payload_size = _exact_int(
            stats.get("size_bytes"),
            f"{label}.generation.{mode}.artifact_stats.size_bytes",
            minimum=1,
        )
        generation_peak = int(
            _rss_observation(
                generation.get("peak_rss"),
                f"{label}.generation.{mode}.peak_rss",
            )
        )
        generation_evidence[mode] = GenerationEvidence(
            wall_seconds=generation_wall,
            payload_size_bytes=payload_size,
            peak_rss_bytes=generation_peak,
        )

        profile = _mapping(raw_profiles.get(mode), f"{label}.profiles.{mode}")
        workload_identities[mode] = _workload_identity(
            profile=profile,
            generation=generation,
            mode=mode,
            layout=expected_layout,
            logical_process=logical_process,
            expected_workload=expected_workload,
            label=f"{label}.profiles.{mode}",
        )
        aggregates = _sequence(
            profile.get("profiles"),
            f"{label}.profiles.{mode}.profiles",
        )
        by_batch: dict[int, Mapping[str, Any]] = {}
        for aggregate_index, raw_aggregate in enumerate(aggregates):
            aggregate = _mapping(
                raw_aggregate,
                f"{label}.profiles.{mode}.profiles[{aggregate_index}]",
            )
            batch = _exact_int(
                aggregate.get("batch_size"),
                f"{label}.profiles.{mode}.profiles[{aggregate_index}].batch_size",
                minimum=1,
            )
            if batch in by_batch:
                raise EvidenceError(f"{label}.profiles.{mode} duplicates batch {batch}")
            by_batch[batch] = aggregate
        if set(by_batch) != set(BATCH_SIZES):
            raise EvidenceError(
                f"{label}.profiles.{mode} does not contain exactly {list(BATCH_SIZES)}"
            )
        for batch_size in BATCH_SIZES:
            cells[(mode, batch_size)] = _extract_cell(
                aggregate=by_batch[batch_size],
                mode=mode,
                batch_size=batch_size,
                subprocess_samples=subprocess_samples,
                warmups=warmups,
                target_runtime=target_runtime,
                schedule_by_index=schedule_by_index,
                seen_addresses=seen_addresses,
                label=f"{label}.profiles.{mode}.batch[{batch_size}]",
            )

    # Each round must retain exactly one independently scheduled subprocess for
    # every lane and batch. Compiled/recurrence ratios are paired by this
    # shared round identity, and eager is independently evaluated below.
    for batch_size in BATCH_SIZES:
        for round_index in range(subprocess_samples):
            pair = [
                entry
                for entry in schedule_by_index.values()
                if entry.get("batch_size") == batch_size
                and entry.get("round") == round_index
                and entry.get("mode") in CAPTURE_MODES
            ]
            if len(pair) != len(CAPTURE_MODES) or {
                entry.get("mode") for entry in pair
            } != set(CAPTURE_MODES):
                raise EvidenceError(
                    f"{label} batch {batch_size} round {round_index} does not "
                    "contain one subprocess for every capture lane"
                )

    campaign_identity = {
        "logical_process": logical_process,
        "layout": expected_layout,
        "workload": expected_workload,
        "batch_sizes": list(BATCH_SIZES),
        "subprocess_samples": subprocess_samples,
        "minimum_internal_samples": configuration.get("minimum_samples"),
        "warmup_runs": warmups,
        "target_runtime_seconds": target_runtime,
        "color_flow_request": configuration.get("color_flow_request"),
        "helicity_request": configuration.get("helicity_request"),
        "validation_seed": configuration.get("validation_seed"),
        "point_tile_size": configuration.get("point_tile_size"),
        "interpreter": interpreter_identity,
    }
    addressed_input_identity = dict(input_identity)
    addressed_input_identity["capture_acceptance_sha256"] = _sha256(acceptance)
    return CaptureEvidence(
        input_identity=addressed_input_identity,
        layout=expected_layout,
        logical_process=logical_process,
        source_revision=revision,
        build_identity=build_identity,
        host_identity=host,
        campaign_identity=campaign_identity,
        workload_identities=workload_identities,
        cells=cells,
        generation=generation_evidence,
    )


def _geometric_mean(values: Sequence[float], label: str) -> float:
    checked = [_positive_number(value, label) for value in values]
    if not checked:
        raise EvidenceError(f"{label} is empty")
    return math.exp(math.fsum(math.log(value) for value in checked) / len(checked))


def _runtime_cell_comparison(
    *,
    layout: str,
    mode: str,
    batch_size: int,
    baseline: CellEvidence,
    candidate: CellEvidence,
) -> dict[str, object]:
    relative_limit = RUNTIME_RELATIVE_LIMIT * baseline.runtime.median
    noise_limit = (
        baseline.runtime.median
        + RUNTIME_NOISE_MAD_MULTIPLIER * baseline.runtime.raw_mad
    )
    passes_relative = candidate.runtime.median <= relative_limit
    passes_noise = candidate.runtime.median <= noise_limit
    gain_at_least_ten_percent = (
        candidate.runtime.median
        <= (1.0 - RESOURCE_EXCEPTION_GAIN) * baseline.runtime.median
    )
    gain_beyond_noise = (
        candidate.runtime.median
        <= baseline.runtime.median
        - RUNTIME_NOISE_MAD_MULTIPLIER * baseline.runtime.raw_mad
    )
    return {
        "layout": layout,
        "mode": mode,
        "batch_size": batch_size,
        "baseline": baseline.runtime.as_dict(),
        "candidate": candidate.runtime.as_dict(),
        "candidate_to_baseline_ratio": (
            candidate.runtime.median / baseline.runtime.median
        ),
        "relative_limit_seconds_per_point": relative_limit,
        "noise_limit_seconds_per_point": noise_limit,
        "passes_relative_limit": passes_relative,
        "passes_noise_limit": passes_noise,
        "passes": passes_relative and passes_noise,
        "resource_exception_runtime_gain": {
            "gain_at_least_10_percent": gain_at_least_ten_percent,
            "gain_beyond_3_baseline_raw_mad": gain_beyond_noise,
            "passes": gain_at_least_ten_percent and gain_beyond_noise,
        },
    }


def _paired_ratio(
    *,
    layout: str,
    batch_size: int,
    capture: CaptureEvidence,
) -> dict[str, object]:
    compiled = capture.cells[("compiled", batch_size)].by_round
    recurrence = capture.cells[("recurrence", batch_size)].by_round
    if set(compiled) != set(recurrence):
        raise EvidenceError(
            f"{layout} batch {batch_size} compiled/recurrence rounds do not match"
        )
    rounds = sorted(compiled)
    ratios = [compiled[round_index] / recurrence[round_index] for round_index in rounds]
    distribution = _distribution(ratios, f"{layout}/{batch_size} paired ratio")
    upper_bound = (
        distribution.median + RUNTIME_NOISE_MAD_MULTIPLIER * distribution.raw_mad
    )
    return {
        "rounds": rounds,
        "compiled_to_recurrence_ratio": distribution.as_dict(),
        "median_plus_3_raw_mad": upper_bound,
    }


def _resource_comparison(
    *,
    layout: str,
    mode: str,
    metric: str,
    baseline: Distribution,
    candidate: Distribution,
    runtime_gain: Mapping[str, object],
    batch_size: int | None = None,
) -> dict[str, object]:
    ratio = candidate.median / baseline.median
    within_growth_limit = ratio <= RESOURCE_GROWTH_LIMIT
    exception = runtime_gain.get("passes") is True
    result: dict[str, object] = {
        "layout": layout,
        "mode": mode,
        "metric": metric,
        "baseline": baseline.as_dict(),
        "candidate": candidate.as_dict(),
        "candidate_to_baseline_ratio": ratio,
        "within_3_percent_growth_limit": within_growth_limit,
        "runtime_gain_exception": dict(runtime_gain),
        "passes": within_growth_limit or exception,
    }
    if batch_size is not None:
        result["batch_size"] = batch_size
    return result


def _single_resource_comparison(
    *,
    layout: str,
    mode: str,
    metric: str,
    baseline: float,
    candidate: float,
    runtime_gain: Mapping[str, object],
) -> dict[str, object]:
    baseline_distribution = Distribution((baseline,), baseline, 0.0)
    candidate_distribution = Distribution((candidate,), candidate, 0.0)
    return _resource_comparison(
        layout=layout,
        mode=mode,
        metric=metric,
        baseline=baseline_distribution,
        candidate=candidate_distribution,
        runtime_gain=runtime_gain,
    )


def _assert_role_consistency(
    captures: Mapping[str, CaptureEvidence],
    *,
    role: str,
) -> None:
    topology = captures["topology-replay"]
    union = captures["all-flow-union"]
    if topology.source_revision != union.source_revision:
        raise EvidenceError(f"{role} captures use different source revisions")
    if topology.build_identity != union.build_identity:
        raise EvidenceError(f"{role} captures use different native builds")
    if topology.host_identity != union.host_identity:
        raise EvidenceError(f"{role} captures were not measured on the same host")
    if topology.logical_process != union.logical_process:
        raise EvidenceError(f"{role} captures use different processes")
    topology_campaign = dict(topology.campaign_identity)
    union_campaign = dict(union.campaign_identity)
    for field in ("layout", "workload", "color_flow_request", "helicity_request"):
        topology_campaign.pop(field, None)
        union_campaign.pop(field, None)
    if topology_campaign != union_campaign:
        raise EvidenceError(f"{role} capture campaign configuration drifted by layout")


def _assert_pair_identity(
    baseline: Mapping[str, CaptureEvidence],
    candidate: Mapping[str, CaptureEvidence],
) -> None:
    if baseline["topology-replay"].source_revision != BASELINE_REVISION:
        raise EvidenceError(
            "baseline source revision is not the immutable migration baseline "
            f"{BASELINE_REVISION}"
        )
    if candidate["topology-replay"].source_revision == BASELINE_REVISION:
        raise EvidenceError("candidate capture still identifies the baseline revision")
    baseline_host = baseline["topology-replay"].host_identity
    candidate_host = candidate["topology-replay"].host_identity
    if baseline_host != candidate_host:
        raise EvidenceError(
            "baseline and candidate captures were not run on the same host"
        )
    for layout in LAYOUTS:
        baseline_capture = baseline[layout]
        candidate_capture = candidate[layout]
        if baseline_capture.campaign_identity != candidate_capture.campaign_identity:
            raise EvidenceError(
                f"{layout} baseline/candidate campaign configurations differ"
            )
        for mode in CAPTURE_MODES:
            if (
                baseline_capture.workload_identities[mode]
                != candidate_capture.workload_identities[mode]
            ):
                raise EvidenceError(
                    f"{layout}/{mode} baseline/candidate workload identities differ"
                )


def compare_captures(
    *,
    baseline: Mapping[str, CaptureEvidence],
    candidate: Mapping[str, CaptureEvidence],
) -> dict[str, object]:
    """Build the content-addressed comparison for four validated captures."""

    if set(baseline) != set(LAYOUTS) or set(candidate) != set(LAYOUTS):
        raise EvidenceError("comparison requires both layouts for both revisions")
    _assert_role_consistency(baseline, role="baseline")
    _assert_role_consistency(candidate, role="candidate")
    _assert_pair_identity(baseline, candidate)

    runtime_cells: list[dict[str, object]] = []
    eager_runtime_cells: list[dict[str, object]] = []
    runtime_by_key: dict[tuple[str, str, int], Mapping[str, object]] = {}
    failures: list[str] = []
    for layout in LAYOUTS:
        for mode in CAPTURE_MODES:
            for batch_size in BATCH_SIZES:
                cell_result = _runtime_cell_comparison(
                    layout=layout,
                    mode=mode,
                    batch_size=batch_size,
                    baseline=baseline[layout].cells[(mode, batch_size)],
                    candidate=candidate[layout].cells[(mode, batch_size)],
                )
                if mode == "eager":
                    eager_runtime_cells.append(cell_result)
                else:
                    runtime_cells.append(cell_result)
                runtime_by_key[(layout, mode, batch_size)] = cell_result
                if cell_result["passes"] is not True:
                    prefix = (
                        "eager diagnostic runtime regression"
                        if mode == "eager"
                        else "runtime regression"
                    )
                    failures.append(f"{prefix}: {layout}/{mode}/batch-{batch_size}")

    paired_ratios: list[dict[str, object]] = []
    for layout in LAYOUTS:
        for batch_size in RATIO_BATCH_SIZES:
            baseline_ratio = _paired_ratio(
                layout=layout,
                batch_size=batch_size,
                capture=baseline[layout],
            )
            candidate_ratio = _paired_ratio(
                layout=layout,
                batch_size=batch_size,
                capture=candidate[layout],
            )
            candidate_upper = _positive_number(
                candidate_ratio["median_plus_3_raw_mad"],
                f"{layout}/{batch_size} candidate paired-ratio upper bound",
            )
            passes = candidate_upper <= COMPILED_RECURRENCE_RATIO_LIMIT
            paired_ratios.append(
                {
                    "layout": layout,
                    "batch_size": batch_size,
                    "baseline": baseline_ratio,
                    "candidate": candidate_ratio,
                    "limit": COMPILED_RECURRENCE_RATIO_LIMIT,
                    "passes": passes,
                }
            )
            if not passes:
                failures.append(
                    f"compiled/recurrence ratio regression: {layout}/batch-{batch_size}"
                )

    generation_cells: list[dict[str, object]] = []
    generation_ratios: list[float] = []
    for layout in LAYOUTS:
        for mode in MODES:
            baseline_wall = baseline[layout].generation[mode].wall_seconds
            candidate_wall = candidate[layout].generation[mode].wall_seconds
            ratio = candidate_wall / baseline_wall
            passes = ratio <= GENERATION_CELL_LIMIT
            generation_ratios.append(ratio)
            generation_cells.append(
                {
                    "layout": layout,
                    "mode": mode,
                    "baseline_wall_seconds": baseline_wall,
                    "candidate_wall_seconds": candidate_wall,
                    "candidate_to_baseline_ratio": ratio,
                    "limit": GENERATION_CELL_LIMIT,
                    "passes": passes,
                }
            )
            if not passes:
                failures.append(f"generation regression: {layout}/{mode}")
    generation_geomean = _geometric_mean(
        generation_ratios,
        "generation candidate/baseline ratios",
    )
    generation_geomean_passes = generation_geomean <= GENERATION_GEOMEAN_LIMIT
    if not generation_geomean_passes:
        failures.append("generation geometric-mean regression")

    eager_generation_cells: list[dict[str, object]] = []
    for layout in LAYOUTS:
        baseline_wall = baseline[layout].generation["eager"].wall_seconds
        candidate_wall = candidate[layout].generation["eager"].wall_seconds
        ratio = candidate_wall / baseline_wall
        passes = ratio <= GENERATION_CELL_LIMIT
        eager_generation_cells.append(
            {
                "layout": layout,
                "mode": "eager",
                "baseline_wall_seconds": baseline_wall,
                "candidate_wall_seconds": candidate_wall,
                "candidate_to_baseline_ratio": ratio,
                "limit": GENERATION_CELL_LIMIT,
                "passes": passes,
            }
        )
        if not passes:
            failures.append(f"eager diagnostic generation regression: {layout}/eager")

    # A mode-level payload or generation-RSS exception requires a >=10%
    # geometric-mean runtime gain and a gain beyond three baseline raw MADs in
    # every measured batch.  Cell-level cold-load/RSS exceptions use the
    # corresponding batch's runtime evidence directly.
    mode_runtime_gain: dict[tuple[str, str], dict[str, object]] = {}
    for layout in LAYOUTS:
        for mode in CAPTURE_MODES:
            comparisons = [
                runtime_by_key[(layout, mode, batch_size)] for batch_size in BATCH_SIZES
            ]
            ratios = [
                _positive_number(
                    comparison["candidate_to_baseline_ratio"],
                    f"{layout}/{mode} candidate/baseline runtime ratio",
                )
                for comparison in comparisons
            ]
            ratio_geomean = _geometric_mean(
                ratios,
                f"{layout}/{mode} runtime ratios",
            )
            beyond_noise_all = all(
                _mapping(
                    comparison["resource_exception_runtime_gain"],
                    "runtime gain evidence",
                ).get("gain_beyond_3_baseline_raw_mad")
                is True
                for comparison in comparisons
            )
            mode_runtime_gain[(layout, mode)] = {
                "runtime_ratio_geometric_mean": ratio_geomean,
                "gain_at_least_10_percent": (
                    ratio_geomean <= 1.0 - RESOURCE_EXCEPTION_GAIN
                ),
                "gain_beyond_3_baseline_raw_mad_in_every_batch": beyond_noise_all,
                "passes": (
                    ratio_geomean <= 1.0 - RESOURCE_EXCEPTION_GAIN and beyond_noise_all
                ),
            }

    resources: list[dict[str, object]] = []
    eager_resources: list[dict[str, object]] = []
    for layout in LAYOUTS:
        for mode in CAPTURE_MODES:
            baseline_generation = baseline[layout].generation[mode]
            candidate_generation = candidate[layout].generation[mode]
            aggregate_gain = mode_runtime_gain[(layout, mode)]
            mode_resources = eager_resources if mode == "eager" else resources
            mode_resources.extend(
                (
                    _single_resource_comparison(
                        layout=layout,
                        mode=mode,
                        metric="artifact_payload_size_bytes",
                        baseline=float(baseline_generation.payload_size_bytes),
                        candidate=float(candidate_generation.payload_size_bytes),
                        runtime_gain=aggregate_gain,
                    ),
                    _single_resource_comparison(
                        layout=layout,
                        mode=mode,
                        metric="generation_peak_rss_bytes",
                        baseline=float(baseline_generation.peak_rss_bytes),
                        candidate=float(candidate_generation.peak_rss_bytes),
                        runtime_gain=aggregate_gain,
                    ),
                )
            )
            for batch_size in BATCH_SIZES:
                baseline_cell = baseline[layout].cells[(mode, batch_size)]
                candidate_cell = candidate[layout].cells[(mode, batch_size)]
                gain = _mapping(
                    runtime_by_key[(layout, mode, batch_size)][
                        "resource_exception_runtime_gain"
                    ],
                    "runtime gain evidence",
                )
                mode_resources.extend(
                    (
                        _resource_comparison(
                            layout=layout,
                            mode=mode,
                            batch_size=batch_size,
                            metric="cold_load_seconds",
                            baseline=baseline_cell.cold_load,
                            candidate=candidate_cell.cold_load,
                            runtime_gain=gain,
                        ),
                        _resource_comparison(
                            layout=layout,
                            mode=mode,
                            batch_size=batch_size,
                            metric="cold_load_peak_rss_bytes",
                            baseline=baseline_cell.cold_load_rss,
                            candidate=candidate_cell.cold_load_rss,
                            runtime_gain=gain,
                        ),
                        _resource_comparison(
                            layout=layout,
                            mode=mode,
                            batch_size=batch_size,
                            metric="profiled_peak_rss_bytes",
                            baseline=baseline_cell.profiled_rss,
                            candidate=candidate_cell.profiled_rss,
                            runtime_gain=gain,
                        ),
                    )
                )
    for diagnostic, inventory in (
        (False, resources),
        (True, eager_resources),
    ):
        for resource in inventory:
            if resource["passes"] is not True:
                batch_suffix = (
                    ""
                    if "batch_size" not in resource
                    else f"/batch-{resource['batch_size']}"
                )
                prefix = (
                    "eager diagnostic resource regression"
                    if diagnostic
                    else "resource regression"
                )
                failures.append(
                    f"{prefix}: "
                    f"{resource['layout']}/{resource['mode']}{batch_suffix}/"
                    f"{resource['metric']}"
                )

    identity = {
        "logical_process": baseline["topology-replay"].logical_process,
        "host_identity_sha256": _sha256(baseline["topology-replay"].host_identity),
        "baseline_source_revision": baseline["topology-replay"].source_revision,
        "candidate_source_revision": candidate["topology-replay"].source_revision,
        "baseline_build_identity_sha256": _sha256(
            baseline["topology-replay"].build_identity
        ),
        "candidate_build_identity_sha256": _sha256(
            candidate["topology-replay"].build_identity
        ),
        "workloads": {
            layout: {
                mode: _sha256(baseline[layout].workload_identities[mode])
                for mode in CAPTURE_MODES
            }
            for layout in LAYOUTS
        },
    }
    eager_runtime_passes = all(cell["passes"] is True for cell in eager_runtime_cells)
    eager_generation_passes = all(
        cell["passes"] is True for cell in eager_generation_cells
    )
    eager_resource_passes = all(
        resource["passes"] is True for resource in eager_resources
    )
    comparison_result: dict[str, object] = {
        "kind": COMPARISON_KIND,
        "schema_version": COMPARISON_SCHEMA,
        "complete": True,
        "passes": not failures,
        "identity": identity,
        "identity_sha256": _sha256(identity),
        "inputs": {
            role: {layout: dict(captures[layout].input_identity) for layout in LAYOUTS}
            for role, captures in (
                ("baseline", baseline),
                ("candidate", candidate),
            )
        },
        "thresholds": {
            "runtime_candidate_median_relative_to_baseline": RUNTIME_RELATIVE_LIMIT,
            "runtime_baseline_raw_mad_multiplier": RUNTIME_NOISE_MAD_MULTIPLIER,
            "compiled_to_recurrence_ratio_median_plus_3_raw_mad": (
                COMPILED_RECURRENCE_RATIO_LIMIT
            ),
            "generation_cell_candidate_to_baseline": GENERATION_CELL_LIMIT,
            "generation_geometric_mean_candidate_to_baseline": (
                GENERATION_GEOMEAN_LIMIT
            ),
            "resource_growth_candidate_to_baseline": RESOURCE_GROWTH_LIMIT,
            "resource_exception_minimum_runtime_gain": RESOURCE_EXCEPTION_GAIN,
        },
        "runtime_cells": runtime_cells,
        "paired_compiled_recurrence_ratios": paired_ratios,
        "generation": {
            "cells": generation_cells,
            "candidate_to_baseline_ratio_geometric_mean": generation_geomean,
            "geometric_mean_limit": GENERATION_GEOMEAN_LIMIT,
            "geometric_mean_passes": generation_geomean_passes,
            "passes": (
                generation_geomean_passes
                and all(cell["passes"] is True for cell in generation_cells)
            ),
        },
        "payload_cold_load_rss": resources,
        "eager_diagnostic": {
            "runtime_cells": eager_runtime_cells,
            "runtime_passes": eager_runtime_passes,
            "generation": {
                "cells": eager_generation_cells,
                "passes": eager_generation_passes,
            },
            "payload_cold_load_rss": eager_resources,
            "resource_passes": eager_resource_passes,
            "paired_compiled_recurrence_ratio_applicable": False,
            "passes": (
                eager_runtime_passes
                and eager_generation_passes
                and eager_resource_passes
            ),
        },
        "failures": failures,
    }
    comparison_result["content_sha256"] = _sha256(comparison_result)
    return comparison_result


def compare_capture_files(
    *,
    baseline_topology: Path,
    baseline_all_flow_union: Path,
    candidate_topology: Path,
    candidate_all_flow_union: Path,
) -> dict[str, object]:
    """Load, validate, and compare the four required capture files."""

    paths = {
        "baseline": {
            "topology-replay": baseline_topology,
            "all-flow-union": baseline_all_flow_union,
        },
        "candidate": {
            "topology-replay": candidate_topology,
            "all-flow-union": candidate_all_flow_union,
        },
    }
    evidence: dict[str, dict[str, CaptureEvidence]] = {
        "baseline": {},
        "candidate": {},
    }
    for role in ("baseline", "candidate"):
        for layout in LAYOUTS:
            payload, input_identity = _read_capture(paths[role][layout])
            evidence[role][layout] = _validate_capture(
                payload,
                input_identity,
                role=role,
                expected_layout=layout,
            )
    return compare_captures(
        baseline=evidence["baseline"],
        candidate=evidence["candidate"],
    )


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _canonical_json_bytes(value) + b"\n"
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        raise EvidenceError(f"cannot write comparison JSON: {path}") from error
    finally:
        temporary_path = locals().get("temporary")
        if isinstance(temporary_path, Path) and temporary_path.exists():
            temporary_path.unlink()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline-topology", type=Path, required=True)
    result.add_argument("--baseline-all-flow-union", type=Path, required=True)
    result.add_argument("--candidate-topology", type=Path, required=True)
    result.add_argument("--candidate-all-flow-union", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        comparison = compare_capture_files(
            baseline_topology=arguments.baseline_topology,
            baseline_all_flow_union=arguments.baseline_all_flow_union,
            candidate_topology=arguments.candidate_topology,
            candidate_all_flow_union=arguments.candidate_all_flow_union,
        )
        _write_json_atomic(arguments.output, comparison)
    except EvidenceError as error:
        print(f"benchmark evidence error: {error}", file=sys.stderr)
        return 2
    return 0 if comparison["passes"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
