#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Measure left-, right-, and mixed-endpoint BG recurrence construction.

This is deliberately a developer experiment.  It keeps the unstable endpoint
choice out of the public configuration while producing normal authenticated
recurrence artifacts.  ``both`` means a mixed physical-sector policy: for each
single open colour line in an all-flow union, choose the endpoint selected by
canonicalizing its interior word against its reversal.  It never sums two
copies of an amplitude.  Topology replay has one representative per symmetry
class, so its mixed policy is deliberately reported as not applicable.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

from pyamplicol import BenchmarkRunner, Runtime
from pyamplicol.api.requests import ModelSource
from pyamplicol.assets.prepared_models import (
    BUILTIN_SM_JIT_O2,
    packaged_prepared_model_path,
)
from pyamplicol.config import (
    Action,
    BenchmarkConfig,
    ColorAccuracy,
    ColorConfig,
    EvaluatorBackend,
    EvaluatorConfig,
    EvaluatorExecutionMode,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    LCFlowLayout,
    RecurrenceEvaluatorConfig,
    RelationDiscoveryMode,
    RunConfig,
)
from tools.developer.generation_slice import GenerationSlice, generate_slice

AnchorPolicy = Literal["right", "left", "both"]
FlowLayout = Literal["topology-replay", "all-flow-union"]

POLICIES: tuple[AnchorPolicy, ...] = ("right", "left", "both")
LAYOUTS: tuple[FlowLayout, ...] = ("topology-replay", "all-flow-union")
SCHEMA = "pyamplicol-recurrence-anchor-study-v2"
VALIDATION_SEED = 0x4247414E


class StudyError(RuntimeError):
    """The requested study cannot produce comparable evidence."""


def _csv_choices(
    value: str,
    choices: tuple[str, ...],
    name: str,
) -> tuple[str, ...]:
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    if not selected:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    unknown = tuple(item for item in selected if item not in choices)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown {name} {unknown!r}; choose from {choices!r}"
        )
    if len(set(selected)) != len(selected):
        raise argparse.ArgumentTypeError(f"{name} must not contain duplicates")
    return selected


def _gluon_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("gluon counts must be integers") from error
    if not counts or any(count < 0 or count > 5 for count in counts):
        raise argparse.ArgumentTypeError("gluon counts must be unique values in 0..5")
    if len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError("gluon counts must not contain duplicates")
    return counts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".artifacts/recurrence-anchor-study"),
    )
    parser.add_argument("--gluon-counts", type=_gluon_counts, default=(2, 3, 4, 5))
    parser.add_argument(
        "--layouts",
        type=lambda value: _csv_choices(value, LAYOUTS, "layouts"),
        default=LAYOUTS,
    )
    parser.add_argument(
        "--policies",
        type=lambda value: _csv_choices(value, POLICIES, "policies"),
        default=POLICIES,
    )
    parser.add_argument("--target-runtime", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--minimum-samples", type=int, default=5)
    parser.add_argument("--generation-only", action="store_true")
    return parser


def _generation_config(layout: FlowLayout) -> RunConfig:
    return RunConfig(
        action=Action.GENERATE,
        color=ColorConfig(
            accuracy=ColorAccuracy.LC,
            lc_flow_layout=LCFlowLayout(layout),
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=True,
                samples=1,
                seed=VALIDATION_SEED,
                relative_tolerance=1.0e-11,
                absolute_tolerance=1.0e-300,
                post_build_validation=True,
            ),
            relation_discovery=GenerationRelationDiscoveryConfig(
                mode=RelationDiscoveryMode.OFF,
            ),
        ),
        evaluator=EvaluatorConfig(
            backend=EvaluatorBackend.JIT,
            execution_mode=EvaluatorExecutionMode.RECURRENCE,
            batch_size=128,
            output_chunk_size=512,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
            recurrence=RecurrenceEvaluatorConfig(
                point_tile_size=1024,
                workspace_mib=512,
            ),
        ),
    )


def _process(gluon_count: int) -> str:
    return "d d~ > z" + " g" * gluon_count


def _cell_name(gluon_count: int, layout: FlowLayout, policy: AnchorPolicy) -> str:
    return f"z{gluon_count}g-{layout}-{policy}"


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StudyError(f"cannot read JSON evidence {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise StudyError(f"JSON evidence is not an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _recurrence_profile(artifact: Path) -> dict[str, object]:
    manifest = _read_json(artifact / "artifact.json")
    try:
        profiles = manifest["extensions"]["generation"][  # type: ignore[index]
            "recurrence_schedule_profiles"
        ]
    except (KeyError, TypeError) as error:
        raise StudyError(
            f"artifact has no recurrence generation profile: {artifact}"
        ) from error
    if not isinstance(profiles, Mapping) or len(profiles) != 1:
        raise StudyError(f"artifact must have one recurrence profile: {artifact}")
    profile_id, raw_profile = next(iter(profiles.items()))
    if not isinstance(raw_profile, Mapping):
        raise StudyError(f"artifact recurrence profile is malformed: {artifact}")
    try:
        final = raw_profile["native_passes"]["final"]  # type: ignore[index]
        counters = final["operation_counters"]
        timings = final["timings_seconds"]
        serialized = final["serialized_bytes"]
    except (KeyError, TypeError) as error:
        raise StudyError(
            f"artifact final recurrence profile is incomplete: {artifact}"
        ) from error
    if not all(isinstance(item, Mapping) for item in (counters, timings, serialized)):
        raise StudyError(f"artifact final recurrence profile is malformed: {artifact}")
    generation = manifest["extensions"]["generation"]  # type: ignore[index]
    phase_timings = generation.get("phase_timings_seconds", {})
    producer = cast(Mapping[str, object], manifest["producer"])
    return {
        "profile_id": str(profile_id),
        "operation_counters": dict(counters),
        "timings_seconds": dict(timings),
        "serialized_bytes": dict(serialized),
        "artifact_phase_timings_seconds": dict(phase_timings),
        "artifact_id": str(manifest.get("artifact_id")),
        "source_identity": {
            "git_revision": producer["git_revision"],
            "native_build_inputs_sha256": producer[
                "native_build_inputs_sha256"
            ],
        },
    }


def _hot_selectors(runtime: Runtime, layout: FlowLayout) -> dict[str, tuple[str, ...]]:
    physics = runtime.physics
    if layout == "all-flow-union":
        helicity = next(
            (
                item
                for item in physics.helicities
                if item.computed and not item.structural_zero
            ),
            None,
        )
        if helicity is None:
            raise StudyError("all-flow-union artifact has no computed live helicity")
        return {"helicity_ids": (helicity.id,), "color_flow_ids": ()}
    color_flow = next((item for item in physics.color_flows if item.computed), None)
    if color_flow is None:
        raise StudyError("topology-replay artifact has no computed color flow")
    return {"helicity_ids": (), "color_flow_ids": (color_flow.id,)}


def _benchmark(
    artifact: Path,
    layout: FlowLayout,
    *,
    target_runtime: float,
    batch_size: int,
    minimum_samples: int,
) -> dict[str, object]:
    runtime = Runtime.load(artifact)
    selectors = _hot_selectors(runtime, layout)
    result = BenchmarkRunner(
        BenchmarkConfig(
            target_runtime=target_runtime,
            batch_size=batch_size,
            precision=16,
            warmup_runs=2,
            minimum_samples=minimum_samples,
            **selectors,
        )
    ).run(runtime)
    timing = result.timing_breakdown
    recurrence_seconds = None
    if timing is not None and timing.recurrence_schedule_time is not None:
        recurrence_seconds = timing.recurrence_schedule_time.mean_seconds_per_point
    return {
        "selectors": {name: list(values) for name, values in selectors.items()},
        "wall_seconds_per_point": result.wall_time_per_point,
        "wall_relative_standard_error": result.uncertainty.relative_standard_error,
        "evaluator_seconds_per_point": result.evaluator_time_per_point,
        "evaluator_total_seconds_per_point": result.evaluator_total_time_per_point,
        "recurrence_schedule_seconds_per_point": recurrence_seconds,
        "sample_count": result.sample_count,
        "repetitions_per_sample": result.repetitions_per_sample,
        "evaluated_point_count": result.evaluated_point_count,
        "effective_config": asdict(result.effective_config),
    }


def _flatten(values: Iterable[Iterable[Iterable[complex]]]) -> tuple[complex, ...]:
    return tuple(complex(value) for point in values for row in point for value in row)


def _comparison(
    reference: tuple[complex, ...],
    candidate: tuple[complex, ...],
) -> dict[str, object]:
    if len(reference) != len(candidate):
        raise StudyError("resolved matrix-element component counts differ")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for expected, observed in zip(reference, candidate, strict=True):
        absolute = abs(observed - expected)
        relative = absolute / max(abs(expected), abs(observed), 1.0e-300)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
    return {
        "component_count": len(reference),
        "maximum_absolute_residual": maximum_absolute,
        "maximum_relative_residual": maximum_relative,
        "passed": maximum_relative <= 1.0e-11 or maximum_absolute <= 1.0e-300,
    }


def _resolved_values(
    runtime: Runtime,
    points: object,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[complex, ...]]:
    resolved = runtime.evaluate_resolved(cast(Any, points))
    return (
        tuple(resolved.helicity_ids),
        tuple(resolved.color_ids),
        _flatten(resolved.values),
    )


def _validate_group(
    artifacts: Mapping[AnchorPolicy, Path],
) -> dict[str, object]:
    if "right" not in artifacts:
        return {"status": "not-run-no-right-reference", "comparisons": {}}
    reference_runtime = Runtime.load(artifacts["right"])
    loader = getattr(reference_runtime._backend, "validation_momenta", None)
    if not callable(loader):
        raise StudyError("runtime exposes no deterministic validation point")
    points = loader()
    if points is None:
        raise StudyError("runtime deterministic validation point is unavailable")
    reference_axes = _resolved_values(reference_runtime, points)
    comparisons: dict[str, object] = {}
    for policy, artifact in artifacts.items():
        runtime = Runtime.load(artifact)
        candidate_axes = _resolved_values(runtime, points)
        if candidate_axes[:2] != reference_axes[:2]:
            raise StudyError(f"{policy} resolved selector axes differ from right")
        comparison = _comparison(reference_axes[2], candidate_axes[2])
        if comparison["passed"] is not True:
            raise StudyError(f"{policy} does not reproduce the right-endpoint result")
        comparisons[policy] = comparison
    return {"status": "passed", "comparisons": comparisons}


def _generate_cell(
    *,
    output_root: Path,
    prepared_model: Path,
    gluon_count: int,
    layout: FlowLayout,
    policy: AnchorPolicy,
    target_runtime: float,
    batch_size: int,
    minimum_samples: int,
    generation_only: bool,
) -> tuple[Path, dict[str, object]]:
    name = _cell_name(gluon_count, layout, policy)
    cell = output_root / "cells" / name
    artifact = cell / "artifact"
    result_path = cell / "result.json"
    if result_path.is_file():
        result = dict(_read_json(result_path))
        if result.get("schema") != SCHEMA:
            raise StudyError(f"saved result has the wrong schema: {result_path}")
        return artifact, result
    if artifact.exists() and not (artifact / "artifact.json").is_file():
        raise StudyError(f"partial artifact requires manual inspection: {artifact}")

    generation_seconds: float | None = None
    if not (artifact / "artifact.json").is_file():
        started = time.perf_counter()
        generate_slice(
            _process(gluon_count),
            artifact,
            selection=GenerationSlice(
                recurrence_closure_anchor_policy=policy,
            ),
            model=ModelSource.from_path(prepared_model),
            config=_generation_config(layout),
        )
        generation_seconds = time.perf_counter() - started

    result: dict[str, object] = {
        "schema": SCHEMA,
        "cell": name,
        "process": _process(gluon_count),
        "gluon_count": gluon_count,
        "layout": layout,
        "policy": policy,
        "generation_wall_seconds": generation_seconds,
        "recurrence_profile": _recurrence_profile(artifact),
        "benchmark": None,
    }
    if not generation_only:
        result["benchmark"] = _benchmark(
            artifact,
            layout,
            target_runtime=target_runtime,
            batch_size=batch_size,
            minimum_samples=minimum_samples,
        )
    _write_json(result_path, result)
    return artifact, result


def _host() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not math.isfinite(args.target_runtime) or args.target_runtime <= 0:
        raise StudyError("target runtime must be positive and finite")
    if args.batch_size < 1 or args.minimum_samples < 2:
        raise StudyError("batch size and minimum samples must be positive")
    output_root = args.output_root.expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, object]] = []
    validations: dict[str, object] = {}
    omissions: list[dict[str, object]] = []
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        for gluon_count in args.gluon_counts:
            for raw_layout in args.layouts:
                layout = cast(FlowLayout, raw_layout)
                artifacts: dict[AnchorPolicy, Path] = {}
                for raw_policy in args.policies:
                    policy = cast(AnchorPolicy, raw_policy)
                    if layout == "topology-replay" and policy == "both":
                        print(
                            f"[{gluon_count}g/{layout}/{policy}] not applicable: "
                            "one anchor is required per replay class",
                            flush=True,
                        )
                        omissions.append(
                            {
                                "gluon_count": gluon_count,
                                "layout": layout,
                                "policy": policy,
                                "reason": (
                                    "topology replay requires one closure anchor per "
                                    "replay-equivalence class, so a per-flow mixed "
                                    "endpoint policy has no distinct evaluator"
                                ),
                            }
                        )
                        continue
                    print(
                        f"[{gluon_count}g/{layout}/{policy}] generating or reusing",
                        flush=True,
                    )
                    artifact, result = _generate_cell(
                        output_root=output_root,
                        prepared_model=prepared_model,
                        gluon_count=gluon_count,
                        layout=layout,
                        policy=policy,
                        target_runtime=args.target_runtime,
                        batch_size=args.batch_size,
                        minimum_samples=args.minimum_samples,
                        generation_only=args.generation_only,
                    )
                    artifacts[policy] = artifact
                    cells.append(result)
                    gc.collect()
                validation_key = f"{gluon_count}g/{layout}"
                validations[validation_key] = _validate_group(artifacts)
                _write_json(
                    output_root / "study.json",
                    {
                        "schema": SCHEMA,
                        "host": _host(),
                        "settings": {
                            "gluon_counts": list(args.gluon_counts),
                            "layouts": list(args.layouts),
                            "policies": list(args.policies),
                            "target_runtime": args.target_runtime,
                            "batch_size": args.batch_size,
                            "minimum_samples": args.minimum_samples,
                            "generation_only": args.generation_only,
                            "relation_discovery": "off",
                            "both_semantics": (
                                "per-sector canonical endpoint in all-flow-union; "
                                "never sum duplicate closures"
                            ),
                        },
                        "cells": cells,
                        "omissions": omissions,
                        "validation": validations,
                    },
                )
    print(output_root / "study.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StudyError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
