#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Validate and benchmark mixed-process spinor DAG artifacts.

The candidate is expected to expose the always-summed ``h:sum`` axis and one
fixed LC flow.  Its normalized total is compared with the same flow selected
from an existing component-recurrence artifact.  Timing uses only the native
f64 wall timer; artifact loading, validation, and JSON emission are untimed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BATCH_SIZE = 128
WARMUP_RUNS = 2
DEFAULT_BLOCKS = 9
DEFAULT_TARGET_SECONDS = 0.25


class MixedBenchmarkError(RuntimeError):
    """Raised when a candidate/reference pair cannot be compared faithfully."""


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    candidate: Path
    reference: Path
    flow: str | None = None
    candidate_process: str | None = None
    reference_process: str | None = None


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def minimum_five(value: str) -> int:
    parsed = int(value)
    if parsed < 5:
        raise argparse.ArgumentTypeError("must be at least 5")
    return parsed


def mean_statistics(samples: Sequence[float]) -> dict[str, float]:
    """Return mean, standard error, and RSE for positive timing samples."""

    values = tuple(float(value) for value in samples)
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise MixedBenchmarkError("timing samples must be finite and positive")
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    standard_error = deviation / math.sqrt(len(values))
    return {
        "mean": mean,
        "standard_deviation": deviation,
        "standard_error": standard_error,
        "relative_standard_error": standard_error / mean,
    }


def comparison_errors(
    candidate: Sequence[complex], reference: Sequence[complex]
) -> dict[str, float]:
    """Return maximum absolute and symmetric relative differences."""

    if len(candidate) != len(reference) or not candidate:
        raise MixedBenchmarkError("candidate/reference value vectors do not match")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for actual, expected in zip(candidate, reference, strict=True):
        absolute = abs(complex(actual) - complex(expected))
        relative = absolute / max(
            abs(complex(actual)), abs(complex(expected)), 1.0e-300
        )
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
    return {
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
    }


def cycle_batch(points: Sequence[Any], size: int = BATCH_SIZE) -> tuple[Any, ...]:
    """Cycle one or more validation points into an identical fixed-size batch."""

    source = tuple(points)
    if not source:
        raise MixedBenchmarkError("artifact exposes no validation momenta")
    return tuple(source[index % len(source)] for index in range(size))


def calibrated_repetitions(probe_seconds: float, target_seconds: float) -> int:
    """Choose the smallest repetition count expected to fill one timing block."""

    if not math.isfinite(probe_seconds) or probe_seconds <= 0.0:
        raise MixedBenchmarkError("native timing probe was not finite and positive")
    if not math.isfinite(target_seconds) or target_seconds <= 0.0:
        raise MixedBenchmarkError("target block duration must be finite and positive")
    return max(1, math.ceil(target_seconds / probe_seconds))


def _flow_id(value: object, context: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and value
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        return "flow:" + ",".join(str(item) for item in value)
    raise MixedBenchmarkError(
        f"{context} must be a flow ID or a non-empty integer word"
    )


def _case_from_mapping(raw: Mapping[str, object], *, base: Path, index: int) -> Case:
    candidate = raw.get("candidate")
    reference = raw.get("reference")
    if not isinstance(candidate, str) or not candidate:
        raise MixedBenchmarkError(f"case {index} has no candidate artifact path")
    if not isinstance(reference, str) or not reference:
        raise MixedBenchmarkError(f"case {index} has no reference artifact path")
    name = raw.get("name", Path(candidate).name)
    if not isinstance(name, str) or not name:
        raise MixedBenchmarkError(f"case {index} has an invalid name")

    def optional_string(key: str) -> str | None:
        value = raw.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise MixedBenchmarkError(f"case {index} {key} must be a non-empty string")
        return value if isinstance(value, str) else None

    return Case(
        name=name,
        candidate=(base / candidate).resolve(),
        reference=(base / reference).resolve(),
        flow=_flow_id(raw.get("flow"), f"case {index} flow"),
        candidate_process=optional_string("candidate_process"),
        reference_process=optional_string("reference_process"),
    )


def load_cases(path: Path) -> tuple[Case, ...]:
    """Load a compact case object, list, or ``{"cases": [...]}`` document."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MixedBenchmarkError(f"cannot read case JSON: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MixedBenchmarkError(f"invalid case JSON: {exc}") from exc
    if isinstance(raw, Mapping) and "cases" in raw:
        rows = raw["cases"]
    elif isinstance(raw, Mapping):
        rows = [raw]
    else:
        rows = raw
    if not isinstance(rows, list) or not rows:
        raise MixedBenchmarkError("case JSON must contain at least one case")
    if not all(isinstance(row, Mapping) for row in rows):
        raise MixedBenchmarkError("every case JSON entry must be an object")
    return tuple(
        _case_from_mapping(row, base=path.parent, index=index)
        for index, row in enumerate(rows)
    )


def _validation_momenta(runtime: Any, label: str) -> tuple[Any, ...]:
    operation = getattr(runtime._backend, "validation_momenta", None)
    if not callable(operation):
        raise MixedBenchmarkError(
            f"{label} runtime cannot read artifact validation momenta"
        )
    points = operation()
    if points is None:
        raise MixedBenchmarkError(
            f"{label} artifact has no available validation momenta"
        )
    result = tuple(points)
    if not result:
        raise MixedBenchmarkError(f"{label} artifact has empty validation momenta")
    return result


def _momenta_error(left: Sequence[Any], right: Sequence[Any]) -> float:
    if len(left) != len(right):
        return math.inf
    maximum = 0.0
    try:
        for left_point, right_point in zip(left, right, strict=True):
            if len(left_point) != len(right_point):
                return math.inf
            for left_vector, right_vector in zip(left_point, right_point, strict=True):
                if len(left_vector) != len(right_vector):
                    return math.inf
                for left_value, right_value in zip(
                    left_vector, right_vector, strict=True
                ):
                    scale = max(abs(float(left_value)), abs(float(right_value)), 1.0)
                    maximum = max(
                        maximum,
                        abs(float(left_value) - float(right_value)) / scale,
                    )
    except (TypeError, ValueError):
        return math.inf
    return maximum


def _native_timer(runtime: Any) -> Any:
    timer = getattr(runtime._backend, "_benchmark_f64_wall_time", None)
    if not callable(timer):
        raise MixedBenchmarkError("runtime has no native f64 wall timer")
    return timer


def _timed_block(
    runtime: Any,
    batch: Sequence[Any],
    repetitions: int,
    flow: str,
) -> float:
    elapsed = float(
        _native_timer(runtime)(
            batch,
            repetitions,
            helicities=None,
            color_flows=(flow,),
            precision=16,
        )
    )
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise MixedBenchmarkError("native f64 wall timer returned an invalid duration")
    return elapsed / (len(batch) * repetitions)


def _resolve_flow(candidate: Any, reference: Any, requested: str | None) -> str:
    candidate_flows = tuple(candidate.physics.color_ids)
    reference_flows = set(reference.physics.color_ids)
    if requested is None:
        if len(candidate_flows) != 1:
            raise MixedBenchmarkError(
                "candidate must expose one fixed flow or the case must specify flow"
            )
        selected = candidate_flows[0]
    else:
        selected = requested
    if selected not in candidate_flows:
        raise MixedBenchmarkError(
            f"candidate does not expose selected flow {selected!r}"
        )
    if selected not in reference_flows:
        raise MixedBenchmarkError(
            f"reference does not expose selected flow {selected!r}"
        )
    return selected


def _artifact_inspection(path: Path, process_id: str) -> dict[str, object]:
    try:
        from pyamplicol.artifacts import inspect_artifact

        inspected = inspect_artifact(path)
        process = next(
            (item for item in inspected.processes if item.id == process_id), None
        )
        return {
            "available": True,
            "artifact_id": inspected.artifact_id,
            "target": inspected.target,
            "cpu_features": list(inspected.cpu_features),
            "payload_count": inspected.payload_count,
            "payload_size_bytes": inspected.payload_size_bytes,
            "integrity": inspected.integrity,
            "process": (
                None
                if process is None
                else {
                    "id": process.id,
                    "execution_mode": process.execution_mode,
                    "workspace_limit_bytes": process.workspace_limit_bytes,
                    "workspace_bytes": process.workspace_bytes,
                    "effective_point_tile_size": process.effective_point_tile_size,
                    "arena_component_count": process.arena_component_count,
                    "direct_contribution_row_count": (
                        process.direct_contribution_row_count
                    ),
                    "finalization_count": process.finalization_count,
                    "closure_count": process.closure_count,
                }
            ),
        }
    except Exception as exc:  # Inspection is supplementary to the loaded runtime.
        return {"available": False, "error": str(exc)}


def _runtime_inspection(runtime: Any) -> dict[str, object]:
    try:
        return {"available": True, "value": dict(runtime.inspect())}
    except Exception as exc:  # Older runtimes need not expose compact inspection.
        return {"available": False, "error": str(exc)}


def _complex_records(values: Sequence[complex]) -> list[dict[str, float]]:
    return [
        {"real": complex(value).real, "imaginary": complex(value).imag}
        for value in values
    ]


def run_case(
    case: Case,
    *,
    blocks: int,
    target_seconds: float,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    """Run one selected-flow validation and paired/interleaved timing cell."""

    from pyamplicol import Runtime

    candidate = Runtime.load(case.candidate, process=case.candidate_process)
    reference = Runtime.load(case.reference, process=case.reference_process)
    if candidate.execution_mode != "spinor":
        raise MixedBenchmarkError(
            f"{case.name}: candidate execution mode is {candidate.execution_mode!r}, "
            "not 'spinor'"
        )
    if candidate.physics.helicity_ids != ("h:sum",):
        raise MixedBenchmarkError(
            f"{case.name}: candidate does not expose only aggregate h:sum"
        )
    flow = _resolve_flow(candidate, reference, case.flow)
    candidate_points = _validation_momenta(candidate, "candidate")
    reference_points = _validation_momenta(reference, "reference")
    momenta_relative_error = _momenta_error(candidate_points, reference_points)
    if momenta_relative_error > 1.0e-12:
        raise MixedBenchmarkError(
            f"{case.name}: candidate/reference validation momenta differ "
            f"(maximum relative component error {momenta_relative_error:.3e})"
        )

    selectors = {"color_flows": (flow,), "precision": 16}
    candidate_values = tuple(
        complex(value) for value in candidate.evaluate(candidate_points, **selectors)
    )
    reference_values = tuple(
        complex(value) for value in reference.evaluate(candidate_points, **selectors)
    )
    errors = comparison_errors(candidate_values, reference_values)
    passes = bool(
        errors["maximum_absolute_difference"] <= absolute_tolerance
        or errors["maximum_relative_difference"] <= relative_tolerance
    )
    if not passes:
        raise MixedBenchmarkError(
            f"{case.name}: selected-flow totals disagree: "
            f"abs={errors['maximum_absolute_difference']:.3e}, "
            f"rel={errors['maximum_relative_difference']:.3e}"
        )

    batch = cycle_batch(candidate_points)
    runtimes = {"candidate": candidate, "reference": reference}
    for warmup in range(WARMUP_RUNS):
        order = (
            ("candidate", "reference")
            if warmup % 2 == 0
            else (
                "reference",
                "candidate",
            )
        )
        for role in order:
            runtimes[role].evaluate(batch, **selectors)

    probes = {
        role: _timed_block(runtime, batch, 1, flow)
        for role, runtime in runtimes.items()
    }
    repetitions = {
        role: calibrated_repetitions(seconds_per_point * len(batch), target_seconds)
        for role, seconds_per_point in probes.items()
    }
    samples: dict[str, list[float]] = {"candidate": [], "reference": []}
    schedule: list[list[str]] = []
    for block in range(blocks):
        order = (
            ["candidate", "reference"]
            if block % 2 == 0
            else [
                "reference",
                "candidate",
            ]
        )
        schedule.append(order)
        for role in order:
            samples[role].append(
                _timed_block(runtimes[role], batch, repetitions[role], flow)
            )

    speedups = tuple(
        reference_sample / candidate_sample
        for candidate_sample, reference_sample in zip(
            samples["candidate"], samples["reference"], strict=True
        )
    )
    candidate_stats = mean_statistics(samples["candidate"])
    reference_stats = mean_statistics(samples["reference"])
    speedup_stats = mean_statistics(speedups)
    candidate_process_id = candidate.physics.process_id
    reference_process_id = reference.physics.process_id
    return {
        "name": case.name,
        "process": candidate.physics.process,
        "flow": flow,
        "selectors": {
            "candidate_helicity_axis": "h:sum (implicit)",
            "reference_helicities": "all (implicit sum)",
            "color_flows": [flow],
            "precision": 16,
        },
        "validation": {
            "point_count": len(candidate_points),
            "momenta": candidate_points,
            "candidate_values": _complex_records(candidate_values),
            "reference_values": _complex_records(reference_values),
            "momenta_maximum_relative_component_difference": momenta_relative_error,
            **errors,
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "passes": passes,
        },
        "timing": {
            "timer": "runtime._benchmark_f64_wall_time",
            "batch_size": len(batch),
            "warmup_runs_per_runtime": WARMUP_RUNS,
            "paired_block_count": blocks,
            "target_seconds_per_block": target_seconds,
            "interleaved_schedule": schedule,
            "calibration": {
                role: {
                    "probe_seconds_per_point": probes[role],
                    "repetitions_per_block": repetitions[role],
                }
                for role in ("candidate", "reference")
            },
            "candidate": {
                "execution_mode": candidate.execution_mode,
                "samples_seconds_per_point": samples["candidate"],
                "mean_microseconds_per_point": candidate_stats["mean"] * 1.0e6,
                "relative_standard_error": candidate_stats["relative_standard_error"],
            },
            "reference": {
                "execution_mode": reference.execution_mode,
                "samples_seconds_per_point": samples["reference"],
                "mean_microseconds_per_point": reference_stats["mean"] * 1.0e6,
                "relative_standard_error": reference_stats["relative_standard_error"],
            },
            "paired_speedup_reference_over_candidate": {
                "samples": speedups,
                "mean": speedup_stats["mean"],
                "relative_standard_error": speedup_stats["relative_standard_error"],
            },
        },
        "inspection": {
            "candidate_artifact": _artifact_inspection(
                case.candidate, candidate_process_id
            ),
            "reference_artifact": _artifact_inspection(
                case.reference, reference_process_id
            ),
            "candidate_runtime": _runtime_inspection(candidate),
            "reference_runtime": _runtime_inspection(reference),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--candidate", type=Path, help="spinor candidate artifact")
    inputs.add_argument("--case-json", type=Path, help="compact case JSON")
    parser.add_argument("--reference", type=Path, help="component reference artifact")
    parser.add_argument("--name", help="single-case label")
    parser.add_argument("--flow", help="fixed color-flow ID, inferred from candidate")
    parser.add_argument("--candidate-process")
    parser.add_argument("--reference-process")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=minimum_five, default=DEFAULT_BLOCKS)
    parser.add_argument(
        "--target-seconds", type=positive_float, default=DEFAULT_TARGET_SECONDS
    )
    parser.add_argument("--absolute-tolerance", type=positive_float, default=1.0e-12)
    parser.add_argument("--relative-tolerance", type=positive_float, default=1.0e-11)
    parser.add_argument("--force", action="store_true", help="replace output JSON")
    return parser


def _arguments_cases(arguments: argparse.Namespace) -> tuple[Case, ...]:
    if arguments.case_json is not None:
        if any(
            value is not None
            for value in (
                arguments.reference,
                arguments.name,
                arguments.flow,
                arguments.candidate_process,
                arguments.reference_process,
            )
        ):
            raise MixedBenchmarkError(
                "--case-json cannot be combined with single-case options"
            )
        return load_cases(arguments.case_json.resolve())
    if arguments.reference is None:
        raise MixedBenchmarkError("--reference is required with --candidate")
    candidate = arguments.candidate.resolve()
    return (
        Case(
            name=arguments.name or candidate.name,
            candidate=candidate,
            reference=arguments.reference.resolve(),
            flow=_flow_id(arguments.flow, "--flow"),
            candidate_process=arguments.candidate_process,
            reference_process=arguments.reference_process,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        cases = _arguments_cases(arguments)
        output = arguments.output.resolve()
        if output.exists() and not arguments.force:
            raise MixedBenchmarkError(f"output already exists: {output} (use --force)")
        results = [
            run_case(
                case,
                blocks=arguments.blocks,
                target_seconds=arguments.target_seconds,
                absolute_tolerance=arguments.absolute_tolerance,
                relative_tolerance=arguments.relative_tolerance,
            )
            for case in cases
        ]
        payload = {
            "kind": "pyamplicol-spinor-dag-mixed-benchmark",
            "schema_version": 1,
            "configuration": {
                "batch_size": BATCH_SIZE,
                "warmup_runs_per_runtime": WARMUP_RUNS,
                "paired_block_count": arguments.blocks,
                "target_seconds_per_block": arguments.target_seconds,
            },
            "cases": results,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for result in results:
            timing = result["timing"]
            validation = result["validation"]
            candidate = timing["candidate"]
            reference = timing["reference"]
            speedup = timing["paired_speedup_reference_over_candidate"]
            print(
                f"{result['name']}: spinor "
                f"{candidate['mean_microseconds_per_point']:.4f} us/point "
                f"(RSE {candidate['relative_standard_error']:.2%}), component "
                f"{reference['mean_microseconds_per_point']:.4f} us/point "
                f"(RSE {reference['relative_standard_error']:.2%}), paired "
                f"{speedup['mean']:.3f}x (RSE "
                f"{speedup['relative_standard_error']:.2%}), max error "
                f"abs={validation['maximum_absolute_difference']:.3e} "
                f"rel={validation['maximum_relative_difference']:.3e}"
            )
        print(f"wrote {output}")
        return 0
    except MixedBenchmarkError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
