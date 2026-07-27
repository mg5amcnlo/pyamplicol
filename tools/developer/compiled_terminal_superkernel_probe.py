#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Disposable go/no-go probe for the compiled terminal superkernel.

The production compiled-DAG runtime deliberately does not contain an
experimental execution path.  This tool captures the exact materialized
``qq_Z6g`` selected-flow stage blueprint while the ordinary artifact is being
generated, asks the pure terminal composer for pair and full-tail candidates,
and compiles both candidates as ordinary unchunked O3 SymJIT applications.

DirectApplication lowering and schedule timing are injected through a strict
JSON runner contract.  This keeps the experiment outside the production
schema and Rust runtime while still giving a standalone arena harness all
canonical source, input, and output bindings required to compare:

* the existing thirteen-leaf selected-flow schedule;
* the existing prefix plus a pair-only stage-6-to-stage-7 application; and
* the existing prefix plus a full stage-6-to-amplitude application.

Running the real probe is intentionally opt-in and expensive.  It must be
placed under the repository's memory watchdog.  Unit tests inject fake
capture/compiler/runner hooks and never invoke Symbolica or generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn, Protocol
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for source_root in (ROOT, SRC):
    if str(source_root) in sys.path:
        sys.path.remove(str(source_root))
sys.path[:0] = [str(SRC), str(ROOT)]

PROCESS = "u u~ > Z g g g g g g"
NORMALIZED_PROCESS = "u u~ > z g g g g g g"
PROCESS_ID = "u_ubar_to_z_g_g_g_g_g_g"
SELECTED_FLOW = "flow:2,4,5,6,7,8,9,1"
RESULT_KIND = "pyamplicol-compiled-terminal-superkernel-probe"
RESULT_SCHEMA_VERSION = 1
RUNNER_REQUEST_KIND = "pyamplicol-terminal-direct-runner-request"
RUNNER_RESULT_KIND = "pyamplicol-terminal-direct-runner-result"
RUNNER_SCHEMA_VERSION = 1
SYMJIT_APPLICATION_ABI = "symjit-application-storage-v3"
DIRECT_APPLICATION_ABI = "symjit-direct-application-storage-v1"
PAYLOAD_CAP_BYTES = 768 * 1024
STACK_CAP_BYTES = 1 << 20
RSS_CAP_GIB = 30.0
MIN_PROJECTED_SPEEDUP = 0.12
FULL_TAIL_MARGIN = 0.03
FULL_TAIL_PAYLOAD_RATIO = 1.25
FULL_TAIL_RESOURCE_RATIO = 1.5
NUMERICAL_RTOL = 1.0e-12
NUMERICAL_ATOL = 1.0e-15
BENCHMARK_BATCHES = (128, 1024)
BENCHMARK_SAMPLE_COUNT = 9
BENCHMARK_TILE_SIZE = 32
VALIDATION_POINT_COUNT = 3
EXPECTED_BASELINE_STAGE_ORDINALS = (0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7)


class ProbeError(RuntimeError):
    """The disposable probe could not produce authoritative evidence."""


def _die(message: str) -> NoReturn:
    raise ProbeError(message)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProbeError("probe evidence is not canonical JSON") from error


def canonical_sha256(value: object) -> str:
    """Return the digest used by both the probe and its unit-test fakes."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ProbeError(f"cannot hash probe file {path}") from error
    return digest.hexdigest()


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _die(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda token: _die(
                f"{label} contains non-finite JSON number {token}"
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProbeError(f"{label} is not strict JSON") from error
    if not isinstance(payload, dict):
        _die(f"{label} must be a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(raw)
        os.replace(temporary, path)
    except OSError as error:
        raise ProbeError(f"cannot write probe evidence {path}") from error


def _peak_rss_gib() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = float(1024**3 if platform.system() == "Darwin" else 1024**2)
    return raw / divisor


def _finite_nonnegative(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        _die(f"{label} must be finite and non-negative")
    return float(value)


def _finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _die(f"{label} must be finite")
    return float(value)


def _finite_positive(value: object, label: str) -> float:
    result = _finite_number(value, label)
    if result <= 0.0:
        _die(f"{label} must be finite and positive")
    return result


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _die(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _die(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class CapturedSchedule:
    """Exact in-memory blueprint plus its disposable baseline artifact."""

    blueprint: Any
    artifact: Path
    execution_path: Path
    leaf_bundle: dict[str, Any]
    capture_evidence: dict[str, Any]


@dataclass(slots=True)
class CompiledCandidate:
    """One composed stage and its retained O3 evaluator."""

    kind: str
    composition: Any
    stage: Any
    evaluator: Any
    application_path: Path
    compile_evidence: dict[str, Any]
    harness_record: dict[str, Any]


class CandidateCompiler(Protocol):
    def __call__(
        self,
        composition: Any,
        output_dir: Path,
    ) -> CompiledCandidate: ...


class NumericalValidator(Protocol):
    def __call__(
        self,
        capture: CapturedSchedule,
        candidates: Mapping[str, CompiledCandidate],
    ) -> Mapping[str, Mapping[str, Any]]: ...


class DirectRunner(Protocol):
    def __call__(
        self,
        capture: CapturedSchedule,
        candidates: Mapping[str, CompiledCandidate],
        output_dir: Path,
    ) -> Mapping[str, Mapping[str, Any]]: ...


def _component_record(component: Any) -> dict[str, Any]:
    return {
        "kind": str(component.kind),
        "source_id": int(component.source_id),
        "component": int(component.component),
        "global_component": int(component.global_component),
        "parameter_index": int(component.parameter_index),
        "real_valued": bool(component.real_valued),
    }


def _stage_outputs(stage: Any, *, arena: str) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any] | None] = [None] * int(stage.output_length)
    for slot in stage.output_slots:
        output_start = int(slot.output_start)
        output_stop = int(slot.output_stop)
        component_start = int(slot.component_start)
        if (
            output_start < 0
            or output_stop > len(outputs)
            or output_start >= output_stop
        ):
            _die("candidate output slot exceeds the composed stage")
        for output_index in range(output_start, output_stop):
            if outputs[output_index] is not None:
                _die("candidate output slots overlap")
            outputs[output_index] = {
                "output_index": output_index,
                "arena": arena,
                "component": component_start + output_index - output_start,
                "value_slot_id": int(slot.value_slot_id),
                "current_id": int(slot.current_id),
                "variant": str(slot.variant),
            }
    if any(output is None for output in outputs):
        _die("candidate output slots do not cover every composed output")
    return [output for output in outputs if output is not None]


def _candidate_harness_record(
    composition: Any,
    application_path: Path,
    compile_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    stage = composition.stage
    kind = str(composition.kind)
    if kind not in {"pair", "full-tail"}:
        _die(f"composer returned unsupported candidate kind {kind!r}")
    inputs = sorted(
        (_component_record(component) for component in stage.input_components),
        key=lambda item: int(item["parameter_index"]),
    )
    if [int(item["parameter_index"]) for item in inputs] != list(
        range(int(stage.parameter_count))
    ):
        _die("candidate semantic inputs are not dense and canonical")
    arena = "current" if kind == "pair" else "amplitude"
    return {
        "kind": kind,
        "source_application": {
            "path": str(application_path.resolve(strict=True)),
            "sha256": str(compile_evidence["application_sha256"]),
            "size_bytes": int(compile_evidence["payload_bytes"]),
            "abi": SYMJIT_APPLICATION_ABI,
            "optimization_level": 3,
            "direct_application_abi": DIRECT_APPLICATION_ABI,
        },
        "logical_inputs": inputs,
        "outputs": _stage_outputs(stage, arena=arena),
        "elided_stage_indices": [
            int(index) for index in composition.elided_stage_indices
        ],
        "dependency_components": [
            int(index) for index in composition.dependency_components
        ],
    }


def _expression_bytes(stage: Any) -> int:
    total = 0
    for expression in stage.output_expressions:
        getter = getattr(expression, "get_byte_size", None)
        if not callable(getter):
            _die("composed expression exposes no get_byte_size()")
        size = getter()
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            _die("composed expression returned an invalid byte size")
        total += size
    return total


def _instruction_shape(evaluator: Any, output_count: int) -> dict[str, int]:
    source = getattr(evaluator, "_source_evaluator", None)
    getter = getattr(source, "get_instructions", None)
    if not callable(getter):
        _die("compiled candidate exposes no retained instruction stream")
    try:
        raw = getter()
    except Exception as error:
        raise ProbeError("cannot inspect candidate instruction stream") from error
    if (
        not isinstance(raw, tuple)
        or len(raw) != 3
        or not isinstance(raw[0], list)
        or isinstance(raw[1], bool)
        or not isinstance(raw[1], int)
        or raw[1] < 0
    ):
        _die("candidate instruction stream has an invalid shape")
    temporary_count = int(raw[1])
    return {
        "instruction_count": len(raw[0]),
        "temporary_count": temporary_count,
        # This is deliberately labelled a projection.  Only the injected
        # DirectApplication runner may claim the actual lowered stack size.
        "projected_logical_stack_bytes": ((temporary_count + output_count) * 2 * 8 * 2),
    }


def compile_o3_candidate(
    composition: Any,
    output_dir: Path,
) -> CompiledCandidate:
    """Compile one composition with the existing unchunked O3 SymJIT path."""

    from pyamplicol.evaluators.symbolica_compile import _compile_symbolica_outputs
    from pyamplicol.evaluators.symbolica_settings import SymbolicaEvaluatorSettings

    stage = composition.stage
    kind = str(composition.kind)
    candidate_dir = output_dir / kind
    candidate_dir.mkdir(parents=True, exist_ok=False)
    settings = SymbolicaEvaluatorSettings(
        backend="jit",
        n_cores=1,
        jit_optimization_level=3,
        compiled_output_chunk_size=None,
        compiled_chunk_compile_workers=1,
        compiled_output_dir=str(candidate_dir),
    )
    rss_before = _peak_rss_gib()
    started = time.perf_counter()
    evaluator = _compile_symbolica_outputs(
        tuple(stage.output_expressions),
        list(stage.parameter_symbols),
        merge_evaluators_strategy=False,
        verbose_evaluator_build=False,
        functions={
            (function, arguments): body
            for function, arguments, body in stage.symbolica_functions
        },
        real_params=tuple(stage.real_valued_inputs),
        symbolica_settings=settings,
        jit_compile=True,
        label=f"compiled_terminal_superkernel_{kind}",
        output_partitions=(),
    )
    manifest = evaluator.artifact_manifest(candidate_dir)
    compile_seconds = time.perf_counter() - started
    application_raw = manifest.get("application_path")
    if not isinstance(application_raw, str):
        _die("candidate evaluator emitted no application path")
    application_path = candidate_dir / application_raw
    try:
        application_path = application_path.resolve(strict=True)
    except OSError as error:
        raise ProbeError("candidate application path does not exist") from error
    if manifest.get("application_abi") != SYMJIT_APPLICATION_ABI:
        _die("candidate emitted an incompatible SymJIT application ABI")
    if manifest.get("optimization_level") != 3:
        _die("candidate did not retain O3 source optimization")
    payload_bytes = application_path.stat().st_size
    shape = _instruction_shape(evaluator, int(stage.output_length))
    evidence = {
        "status": "ok",
        "expression_bytes": _expression_bytes(stage),
        "output_count": int(stage.output_length),
        "parameter_count": int(stage.parameter_count),
        "payload_bytes": payload_bytes,
        "application_sha256": _file_sha256(application_path),
        "compile_seconds": compile_seconds,
        "process_peak_rss_gib_before": rss_before,
        "process_peak_rss_gib": _peak_rss_gib(),
        **shape,
        "build_timing": {
            str(key): float(value)
            for key, value in dict(manifest.get("build_timing") or {}).items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        },
    }
    harness_record = _candidate_harness_record(
        composition,
        application_path,
        evidence,
    )
    return CompiledCandidate(
        kind=kind,
        composition=composition,
        stage=stage,
        evaluator=evaluator,
        application_path=application_path,
        compile_evidence=evidence,
        harness_record=harness_record,
    )


def _random_external_values(
    stages: Sequence[Any],
    *,
    point_count: int,
) -> dict[tuple[str, int], Any]:
    import numpy as np

    produced = {
        int(slot.component_start) + output_index - int(slot.output_start)
        for stage in stages
        for slot in stage.output_slots
        for output_index in range(int(slot.output_start), int(slot.output_stop))
    }
    values: dict[tuple[str, int], Any] = {}
    rng = np.random.default_rng(0x5A17_2026)
    for stage in stages:
        for component in stage.input_components:
            key = (str(component.kind), int(component.global_component))
            if (
                component.kind == "value"
                and int(component.global_component) in produced
            ):
                continue
            if key in values:
                continue
            real = rng.uniform(0.75, 1.25, size=point_count)
            if bool(component.real_valued):
                value = real.astype(np.complex128)
            else:
                imag = rng.uniform(-0.25, 0.25, size=point_count)
                value = real + 1j * imag
            if component.kind == "model_parameter":
                value[:] = value[0]
            values[key] = value
    return values


def _compile_dense_stage(stage: Any, label: str) -> Any:
    from pyamplicol.evaluators.symbolica_compile import _compile_symbolica_outputs
    from pyamplicol.evaluators.symbolica_settings import SymbolicaEvaluatorSettings

    return _compile_symbolica_outputs(
        tuple(stage.output_expressions),
        list(stage.parameter_symbols),
        merge_evaluators_strategy=False,
        verbose_evaluator_build=False,
        functions={
            (function, arguments): body
            for function, arguments, body in stage.symbolica_functions
        },
        real_params=tuple(stage.real_valued_inputs),
        symbolica_settings=SymbolicaEvaluatorSettings(
            backend="jit",
            n_cores=1,
            jit_optimization_level=3,
            compiled_output_chunk_size=None,
            compiled_chunk_compile_workers=1,
        ),
        jit_compile=False,
        label=label,
        output_partitions=(),
    )


def _parameter_rows(
    stage: Any,
    values: Mapping[tuple[str, int], Any],
    point_count: int,
) -> Any:
    import numpy as np

    rows = np.empty((point_count, int(stage.parameter_count)), dtype=np.complex128)
    seen: set[int] = set()
    for component in stage.input_components:
        index = int(component.parameter_index)
        key = (str(component.kind), int(component.global_component))
        if index in seen or key not in values:
            _die(f"numerical validation cannot bind stage input {key!r}")
        seen.add(index)
        rows[:, index] = values[key]
    if seen != set(range(int(stage.parameter_count))):
        _die("numerical validation stage inputs are incomplete")
    return rows


def _publish_stage_outputs(
    stage: Any,
    output: Any,
    values: dict[tuple[str, int], Any],
) -> None:
    import numpy as np

    matrix = np.asarray(output, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[1] != int(stage.output_length):
        _die("numerical validation stage output has the wrong shape")
    for slot in stage.output_slots:
        for output_index in range(int(slot.output_start), int(slot.output_stop)):
            component = (
                int(slot.component_start) + output_index - int(slot.output_start)
            )
            values[("value", component)] = matrix[:, output_index].copy()


def validate_dense_candidates(
    capture: CapturedSchedule,
    candidates: Mapping[str, CompiledCandidate],
) -> Mapping[str, Mapping[str, Any]]:
    """Numerically compare composed O3 outputs with the unfused symbolic tail."""

    import numpy as np

    blueprint = capture.blueprint
    current_stages = tuple(blueprint.stages)
    if len(current_stages) < 2:
        _die("captured blueprint has no terminal current-stage pair")
    tail = (current_stages[-2], current_stages[-1], blueprint.amplitude_stage)
    initial = _random_external_values(tail, point_count=VALIDATION_POINT_COUNT)
    baseline_values = {key: value.copy() for key, value in initial.items()}
    baseline_outputs: dict[str, Any] = {}
    for index, stage in enumerate(tail):
        evaluator = _compile_dense_stage(stage, f"terminal_baseline_{index}")
        rows = _parameter_rows(stage, baseline_values, VALIDATION_POINT_COUNT)
        output = np.asarray(evaluator.evaluate_complex(rows), dtype=np.complex128)
        if index == 1:
            baseline_outputs["pair"] = output.copy()
        if index < 2:
            _publish_stage_outputs(stage, output, baseline_values)
        else:
            baseline_outputs["full-tail"] = output.copy()

    evidence: dict[str, Mapping[str, Any]] = {}
    for kind, candidate in candidates.items():
        rows = _parameter_rows(candidate.stage, initial, VALIDATION_POINT_COUNT)
        actual = np.asarray(
            candidate.evaluator.evaluate_complex(rows),
            dtype=np.complex128,
        )
        expected = np.asarray(baseline_outputs[kind], dtype=np.complex128)
        if actual.shape != expected.shape:
            _die(f"{kind} numerical output shape differs from the baseline")
        delta = np.abs(actual - expected)
        tolerance = NUMERICAL_ATOL + NUMERICAL_RTOL * np.abs(expected)
        ok = bool(np.all(delta <= tolerance))
        max_abs = float(np.max(delta, initial=0.0))
        relative = delta / np.maximum(np.abs(expected), NUMERICAL_ATOL)
        max_rel = float(np.max(relative, initial=0.0))
        evidence[kind] = {
            "status": "ok" if ok else "mismatch",
            "point_count": VALIDATION_POINT_COUNT,
            "rtol": NUMERICAL_RTOL,
            "atol": NUMERICAL_ATOL,
            "max_absolute_difference": max_abs,
            "max_relative_difference": max_rel,
        }
    return evidence


def _execution_lane(execution: Mapping[str, Any]) -> Mapping[str, Any]:
    helicity_sum = execution.get("helicity_sum_execution")
    if not isinstance(helicity_sum, Mapping):
        _die("qq_Z6g execution has no helicity-sum lane")
    raw_sectors = helicity_sum.get("color_selector_executions")
    if not isinstance(raw_sectors, list):
        _die("qq_Z6g helicity-sum lane has no materialized color sectors")
    sectors = [
        sector
        for sector in raw_sectors
        if isinstance(sector, Mapping) and sector.get("materialized_sector_id") == 0
    ]
    if len(sectors) != 1 or not isinstance(sectors[0].get("execution"), Mapping):
        _die("qq_Z6g selected materialized sector is not unique")
    lane = sectors[0]["execution"]
    reduction = lane.get("physics_reduction")
    if not isinstance(reduction, Mapping):
        _die("qq_Z6g selected sector has no physics reduction")
    groups = reduction.get("groups")
    if not isinstance(groups, list) or not groups:
        _die("qq_Z6g selected sector has no reduction groups")
    for group in groups:
        if not isinstance(group, Mapping) or group.get("physical_color_ids") != [
            SELECTED_FLOW
        ]:
            _die("qq_Z6g materialized sector does not represent the selected flow")
    return lane


def _selected_lane_proof(lane: Mapping[str, Any]) -> dict[str, Any]:
    """Prove that the captured lane is the unique materialized selected flow."""

    reduction = lane.get("physics_reduction")
    if not isinstance(reduction, Mapping):
        _die("qq_Z6g selected sector has no physics reduction")
    groups = reduction.get("groups")
    if not isinstance(groups, list) or not groups:
        _die("qq_Z6g selected sector has no reduction groups")
    if any(
        not isinstance(group, Mapping)
        or group.get("physical_color_ids") != [SELECTED_FLOW]
        for group in groups
    ):
        _die("qq_Z6g selected sector contains another physical color flow")
    return {
        "status": "proven",
        "materialized_sector_id": 0,
        "selected_flow": SELECTED_FLOW,
        "reduction_group_count": len(groups),
        "all_groups_exact_selected_flow": True,
        "runtime_selector_boundary_in_tail": False,
    }


def _evaluator_leaves(evaluator: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    chunks = evaluator.get("chunks")
    if isinstance(chunks, list):
        if not all(isinstance(chunk, Mapping) for chunk in chunks):
            _die("compiled stage evaluator chunks are malformed")
        return list(chunks)
    return [evaluator]


def _validate_direct_stage_contract(
    direct: Mapping[str, Any],
    evaluator: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    expected = {
        "kind": "compiled-plane-arena-stage",
        "schema_version": 1,
        "source_application_abi": SYMJIT_APPLICATION_ABI,
        "application_abi": DIRECT_APPLICATION_ABI,
        "element_layout": "split-complex-component-major",
        "input_output_aliasing": "forbidden",
        "output_output_aliasing": "forbidden",
        "output_operation": "overwrite",
        "output_factor": "identity",
    }
    for key, value in expected.items():
        if direct.get(key) != value:
            _die(f"selected qq_Z6g Direct-Arena {key} is incompatible")
    leaves = direct.get("leaves")
    inputs = direct.get("input_bindings")
    outputs = direct.get("output_bindings")
    evaluator_leaves = _evaluator_leaves(evaluator)
    if (
        not isinstance(leaves, list)
        or not all(isinstance(leaf, Mapping) for leaf in leaves)
        or not isinstance(inputs, list)
        or not all(isinstance(binding, Mapping) for binding in inputs)
        or not isinstance(outputs, list)
        or not all(isinstance(binding, Mapping) for binding in outputs)
        or len(leaves) != len(evaluator_leaves)
    ):
        _die("selected qq_Z6g leaf metadata is incomplete")
    if [binding.get("parameter_index") for binding in inputs] != list(
        range(len(inputs))
    ):
        _die("selected qq_Z6g Direct-Arena inputs are not dense")
    output_identities: list[tuple[str, int]] = []
    for output_index, binding in enumerate(outputs):
        if binding.get("output_index") != output_index:
            _die("selected qq_Z6g Direct-Arena outputs are not dense")
        arena = binding.get("arena")
        component = binding.get("component")
        if (
            arena not in {"current", "amplitude"}
            or isinstance(component, bool)
            or not isinstance(component, int)
            or component < 0
        ):
            _die("selected qq_Z6g Direct-Arena output binding is invalid")
        output_identities.append((str(arena), component))
    if len(set(output_identities)) != len(output_identities):
        _die("selected qq_Z6g Direct-Arena outputs alias")
    return (
        [dict(leaf) for leaf in leaves],
        [dict(binding) for binding in inputs],
        [dict(binding) for binding in outputs],
    )


def _validate_leaf_identity(
    *,
    direct: Mapping[str, Any],
    raw_leaf: Mapping[str, Any],
    raw_evaluator: Mapping[str, Any],
    application_path: str,
    input_count: int,
    output_count: int,
) -> None:
    expected = {
        "application_path": application_path,
        "application_abi": SYMJIT_APPLICATION_ABI,
        "optimization_level": raw_leaf.get("optimization_level"),
        "input_len": input_count,
        "output_len": output_count,
    }
    for key, value in expected.items():
        if raw_evaluator.get(key) != value:
            _die(f"selected qq_Z6g evaluator leaf {key} differs from Direct-Arena")
    if raw_leaf.get("source_application_abi") != SYMJIT_APPLICATION_ABI:
        _die("selected qq_Z6g leaf source ABI is incompatible")
    if raw_leaf.get("direct_codegen_optimization_level") != 3:
        _die("selected qq_Z6g leaf is not lowered with DirectApplication O3")
    if direct.get("application_abi") != DIRECT_APPLICATION_ABI:
        _die("selected qq_Z6g leaf DirectApplication ABI is incompatible")


def _baseline_leaf_records(
    artifact: Path,
    lane: Mapping[str, Any],
    bundle_dir: Path,
) -> list[dict[str, Any]]:
    from pyamplicol.generation.evaluator_container import PacbinReader

    compiled = lane.get("compiled")
    if not isinstance(compiled, Mapping):
        _die("selected qq_Z6g lane has no compiled plan")
    plan = compiled.get("stage_evaluators")
    if not isinstance(plan, Mapping):
        _die("selected qq_Z6g lane has no stage evaluators")
    raw_stages = plan.get("stages")
    amplitude = plan.get("amplitude_stage")
    if not isinstance(raw_stages, list) or not isinstance(amplitude, Mapping):
        _die("selected qq_Z6g stage plan is incomplete")
    stages = [*raw_stages, amplitude]
    bundle_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    with PacbinReader.open(
        artifact / "evaluators.pacbin",
        verify_payloads=True,
    ) as reader:
        for stage_ordinal, raw_stage in enumerate(stages):
            if not isinstance(raw_stage, Mapping):
                _die("selected qq_Z6g stage record is malformed")
            direct = raw_stage.get("compiled_plane_arena")
            evaluator = raw_stage.get("evaluator")
            if not isinstance(direct, Mapping) or not isinstance(evaluator, Mapping):
                _die("selected qq_Z6g stage has no Direct-Arena metadata")
            leaves, inputs, outputs = _validate_direct_stage_contract(
                direct,
                evaluator,
            )
            evaluator_leaves = _evaluator_leaves(evaluator)
            for leaf_index, (raw_leaf, raw_evaluator) in enumerate(
                zip(leaves, evaluator_leaves, strict=True)
            ):
                application_path = raw_leaf.get("application_path")
                if not isinstance(application_path, str):
                    application_path = raw_evaluator.get("application_path")
                if not isinstance(application_path, str):
                    _die("selected qq_Z6g leaf has no source application")
                member_path = _process_member_path(application_path)
                member = reader.member(member_path)
                payload = reader.read_member(member_path, length=member.length)
                extracted = bundle_dir / f"baseline-leaf-{len(records):02d}.symjit"
                extracted.write_bytes(payload)
                input_indices = raw_leaf.get("input_indices")
                start = raw_leaf.get("output_start")
                stop = raw_leaf.get("output_stop")
                if (
                    not isinstance(input_indices, list)
                    or isinstance(start, bool)
                    or not isinstance(start, int)
                    or isinstance(stop, bool)
                    or not isinstance(stop, int)
                    or start < 0
                    or stop <= start
                    or stop > len(outputs)
                ):
                    _die("selected qq_Z6g leaf ranges are invalid")
                _validate_leaf_identity(
                    direct=direct,
                    raw_leaf=raw_leaf,
                    raw_evaluator=raw_evaluator,
                    application_path=application_path,
                    input_count=len(input_indices),
                    output_count=stop - start,
                )
                logical_inputs: list[dict[str, Any]] = []
                for leaf_parameter_index, stage_parameter_index in enumerate(
                    input_indices
                ):
                    if (
                        isinstance(stage_parameter_index, bool)
                        or not isinstance(stage_parameter_index, int)
                        or stage_parameter_index < 0
                        or stage_parameter_index >= len(inputs)
                        or not isinstance(inputs[stage_parameter_index], Mapping)
                    ):
                        _die("selected qq_Z6g leaf input map is invalid")
                    binding = dict(inputs[stage_parameter_index])
                    logical_inputs.append(
                        {
                            "leaf_parameter_index": leaf_parameter_index,
                            "stage_parameter_index": stage_parameter_index,
                            "kind": str(binding["kind"]),
                            "source_id": int(binding["source_id"]),
                            "component": int(binding["component"]),
                            "global_component": int(binding["global_component"]),
                            "real_valued": bool(binding.get("real_valued", False)),
                        }
                    )
                logical_outputs: list[dict[str, Any]] = []
                for output_index, binding in enumerate(outputs[start:stop]):
                    if not isinstance(binding, Mapping):
                        _die("selected qq_Z6g leaf output map is invalid")
                    logical_outputs.append(
                        {
                            "leaf_output_index": output_index,
                            "stage_output_index": start + output_index,
                            "arena": str(binding["arena"]),
                            "component": int(binding["component"]),
                        }
                    )
                records.append(
                    {
                        "leaf_index": len(records),
                        "stage_ordinal": stage_ordinal,
                        "stage_index": int(raw_stage.get("stage_index", stage_ordinal)),
                        "stage_kind": str(raw_stage.get("stage_kind", "")),
                        "stage_leaf_index": leaf_index,
                        "source_application": {
                            "path": str(extracted.resolve(strict=True)),
                            "logical_path": application_path,
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "size_bytes": len(payload),
                            "abi": str(raw_leaf.get("source_application_abi")),
                            "optimization_level": int(
                                raw_leaf.get("optimization_level", -1)
                            ),
                            "direct_codegen_optimization_level": int(
                                raw_leaf.get(
                                    "direct_codegen_optimization_level",
                                    -1,
                                )
                            ),
                            "direct_application_abi": str(
                                direct.get("application_abi")
                            ),
                        },
                        "logical_inputs": logical_inputs,
                        "outputs": logical_outputs,
                    }
                )
    if len(records) != 13:
        _die(f"selected qq_Z6g schedule has {len(records)} leaves instead of 13")
    ordinals = tuple(int(record["stage_ordinal"]) for record in records)
    if ordinals != EXPECTED_BASELINE_STAGE_ORDINALS:
        _die(
            "selected qq_Z6g baseline leaf stage ordinals differ from "
            f"{list(EXPECTED_BASELINE_STAGE_ORDINALS)}"
        )
    return records


def _process_member_path(application_path: str) -> str:
    """Resolve one process-relative evaluator path into the root PACBIN."""

    path = Path(application_path)
    if (
        not application_path
        or path.is_absolute()
        or application_path.startswith("processes/")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _die("selected qq_Z6g application path is not process-relative")
    return f"processes/{PROCESS_ID}/{application_path}"


def _arena_shape(lane: Mapping[str, Any]) -> dict[str, int]:
    runtime_schema = lane.get("runtime_schema")
    if not isinstance(runtime_schema, Mapping):
        _die("selected qq_Z6g lane has no runtime schema")
    layout = runtime_schema.get("parameter_layout")
    current_storage = runtime_schema.get("current_storage")
    amplitude_stage = runtime_schema.get("amplitude_stage")
    if (
        not isinstance(layout, Mapping)
        or not isinstance(current_storage, Mapping)
        or not isinstance(amplitude_stage, Mapping)
    ):
        _die("selected qq_Z6g runtime arena metadata is incomplete")
    value_count = _positive_int(
        layout.get("value_component_count"),
        "qq_Z6g value-component count",
    )
    current_count = _positive_int(
        current_storage.get("component_count"),
        "qq_Z6g current-component count",
    )
    amplitude_count = _positive_int(
        amplitude_stage.get("output_count"),
        "qq_Z6g amplitude-component count",
    )
    momentum_count = _positive_int(
        layout.get("momentum_parameter_count"),
        "qq_Z6g momentum scalar-component count",
    )
    model_parameter_count = _nonnegative_int(
        layout.get("model_parameter_count"),
        "qq_Z6g model-parameter count",
    )
    if momentum_count % 4 != 0:
        _die("qq_Z6g momentum scalar-component count is not four-vector aligned")
    if current_count < value_count:
        _die("qq_Z6g current arena is smaller than its value-component domain")
    return {
        "value_component_count": value_count,
        "current_component_count": current_count,
        "amplitude_component_count": amplitude_count,
        "momentum_scalar_component_count": momentum_count,
        "momentum_form_count": momentum_count // 4,
        "model_parameter_count": model_parameter_count,
    }


def _runner_schedules(
    baseline_leaves: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if len(baseline_leaves) != len(EXPECTED_BASELINE_STAGE_ORDINALS):
        _die("runner schedule requires the exact thirteen-leaf baseline")
    ordinals = tuple(
        _nonnegative_int(leaf.get("stage_ordinal"), "baseline stage ordinal")
        for leaf in baseline_leaves
    )
    if ordinals != EXPECTED_BASELINE_STAGE_ORDINALS:
        _die("runner schedule baseline stage ordinals are not canonical")

    def baseline(index: int) -> dict[str, Any]:
        return {"source": "baseline", "leaf_index": index}

    prefix = [baseline(index) for index in range(8)]
    return {
        "baseline": [baseline(index) for index in range(13)],
        "pair": [
            *prefix,
            {"source": "candidate", "kind": "pair"},
            baseline(12),
        ],
        "full-tail": [
            *prefix,
            {"source": "candidate", "kind": "full-tail"},
        ],
    }


def _leaf_bundle(
    artifact: Path,
    execution_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        execution = _strict_json_bytes(
            execution_path.read_bytes(),
            "qq_Z6g execution manifest",
        )
    except OSError as error:
        raise ProbeError("cannot read qq_Z6g execution manifest") from error
    lane = _execution_lane(execution)
    selected_lane_proof = _selected_lane_proof(lane)
    records = _baseline_leaf_records(
        artifact,
        lane,
        output_dir / "baseline-applications",
    )
    payload: dict[str, Any] = {
        "kind": "pyamplicol-terminal-superkernel-leaf-bundle",
        "schema_version": 1,
        "process": NORMALIZED_PROCESS,
        "process_id": PROCESS_ID,
        "selected_flow": SELECTED_FLOW,
        "source_application_abi": SYMJIT_APPLICATION_ABI,
        "direct_application_abi": DIRECT_APPLICATION_ABI,
        "selected_lane_proof": selected_lane_proof,
        "arena_shape": _arena_shape(lane),
        "baseline_leaf_count": len(records),
        "baseline_leaves": records,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def capture_exact_selected_blueprint(output_dir: Path) -> CapturedSchedule:
    """Generate the disposable baseline and retain its exact sector-0 blueprint."""

    from pyamplicol import Generator
    from pyamplicol.config import (
        ColorConfig,
        EvaluatorConfig,
        EvaluatorOptimizationConfig,
        GenerationConfig,
        GenerationValidationConfig,
        JITConfig,
        OutputConfig,
        RunConfig,
    )
    from pyamplicol.generation import service, stage_artifacts

    artifact = output_dir / "baseline-artifact"
    if artifact.exists():
        _die(f"disposable baseline artifact already exists: {artifact}")
    captured: list[Any] = []
    original_service_builder = service.build_and_write_generic_stage_evaluator_artifacts
    original_blueprint_builder = stage_artifacts.build_generic_stage_compiler_blueprint

    def service_builder(*args: Any, **kwargs: Any) -> Any:
        raw_root = args[2] if len(args) >= 3 else kwargs.get("artifact_dir")
        if raw_root is None:
            _die("stage compiler call exposes no artifact root")
        lane_root = Path(raw_root)
        target = (
            ".helicity-sum-color-selector" in lane_root.parts
            and "sector-0" in lane_root.parts
        )
        if not target:
            return original_service_builder(*args, **kwargs)

        def blueprint_builder(*builder_args: Any, **builder_kwargs: Any) -> Any:
            consumer = builder_kwargs.get("stage_consumer")
            if not callable(consumer):
                _die("streamed target stage compiler exposes no consumer")
            retained_stages: list[Any] = []

            def tee(stage: Any, position: int, stage_count: int) -> None:
                retained_stages.append(stage)
                consumer(stage, position, stage_count)

            rewritten = dict(builder_kwargs)
            rewritten["stage_consumer"] = tee
            released = original_blueprint_builder(*builder_args, **rewritten)
            current = tuple(
                stage
                for stage in retained_stages
                if not str(stage.stage_kind).startswith("amplitude")
            )
            amplitude = tuple(
                stage
                for stage in retained_stages
                if str(stage.stage_kind).startswith("amplitude")
            )
            if len(amplitude) != 1:
                _die("target stage compiler did not expose one amplitude stage")
            captured.append(
                replace(
                    released,
                    stages=current,
                    amplitude_stage=amplitude[0],
                )
            )
            return released

        with patch.object(
            stage_artifacts,
            "build_generic_stage_compiler_blueprint",
            blueprint_builder,
        ):
            return original_service_builder(*args, **kwargs)

    run = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc", lc_flow_layout="topology-replay"),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False,
                samples=1,
                seed=0x5A17_2026,
                relative_tolerance=NUMERICAL_RTOL,
                absolute_tolerance=NUMERICAL_ATOL,
                post_build_validation=False,
            ),
        ),
        evaluator=EvaluatorConfig(
            backend="jit",
            execution_mode="compiled",
            output_chunk_size=512,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=3),
        ),
        output=OutputConfig(format="json", color="never", progress="off"),
    )
    started = time.perf_counter()
    with patch.object(
        service,
        "build_and_write_generic_stage_evaluator_artifacts",
        service_builder,
    ):
        result = Generator(run).generate(PROCESS, artifact, mode="error")
    if len(captured) != 1:
        _die(f"captured {len(captured)} selected blueprints instead of one")
    generated = Path(result.output).resolve(strict=True)
    execution_path = (generated / "processes" / PROCESS_ID / "execution.json").resolve(
        strict=True
    )
    blueprint = captured[0]
    current_shapes = [
        (int(stage.parameter_count), int(stage.output_length))
        for stage in blueprint.stages
    ]
    expected_tail = [(1490, 640), (1590, 768)]
    if current_shapes[-2:] != expected_tail or (
        int(blueprint.amplitude_stage.parameter_count),
        int(blueprint.amplitude_stage.output_length),
    ) != (772, 384):
        _die("captured qq_Z6g blueprint has the wrong terminal geometry")
    leaf_bundle = _leaf_bundle(generated, execution_path, output_dir)
    capture_evidence = {
        "artifact": str(generated),
        "execution_manifest": str(execution_path),
        "generation_seconds": time.perf_counter() - started,
        "process_peak_rss_gib": _peak_rss_gib(),
        "stage_count": len(blueprint.stages),
        "stage_shapes": [
            {
                "stage_index": int(stage.stage_index),
                "parameter_count": int(stage.parameter_count),
                "output_count": int(stage.output_length),
            }
            for stage in blueprint.stages
        ],
        "amplitude_shape": {
            "parameter_count": int(blueprint.amplitude_stage.parameter_count),
            "output_count": int(blueprint.amplitude_stage.output_length),
        },
        "leaf_bundle_sha256": str(leaf_bundle["content_sha256"]),
        "selected_lane_proof": dict(leaf_bundle["selected_lane_proof"]),
    }
    return CapturedSchedule(
        blueprint=blueprint,
        artifact=generated,
        execution_path=execution_path,
        leaf_bundle=leaf_bundle,
        capture_evidence=capture_evidence,
    )


def _unavailable_direct_runner(
    _capture: CapturedSchedule,
    candidates: Mapping[str, CompiledCandidate],
    _output_dir: Path,
) -> Mapping[str, Mapping[str, Any]]:
    return {
        kind: {
            "lowering": {
                "status": "unavailable",
                "source_stack_bytes": None,
                "lowered_stack_bytes": None,
                "configured_stack_limit_bytes": STACK_CAP_BYTES,
                "stack_limit_enforced": False,
                "warmed_arena_allocation_bytes": None,
            },
            "numerical": {"status": "unavailable"},
            "benchmarks": {},
        }
        for kind in candidates
    }


def external_direct_runner(command: Sequence[str]) -> DirectRunner:
    """Create the strict-JSON adapter used by an out-of-tree arena harness."""

    executable = tuple(str(item) for item in command)
    if not executable:
        _die("direct runner command is empty")

    def run(
        capture: CapturedSchedule,
        candidates: Mapping[str, CompiledCandidate],
        output_dir: Path,
    ) -> Mapping[str, Mapping[str, Any]]:
        request: dict[str, Any] = {
            "kind": RUNNER_REQUEST_KIND,
            "schema_version": RUNNER_SCHEMA_VERSION,
            "process": NORMALIZED_PROCESS,
            "selected_flow": SELECTED_FLOW,
            "batches": list(BENCHMARK_BATCHES),
            "tile_size": BENCHMARK_TILE_SIZE,
            "samples": BENCHMARK_SAMPLE_COUNT,
            "sample_seconds": 1.0,
            "rtol": NUMERICAL_RTOL,
            "atol": NUMERICAL_ATOL,
            "baseline": capture.leaf_bundle,
            "arena_shape": dict(capture.leaf_bundle["arena_shape"]),
            "schedules": _runner_schedules(capture.leaf_bundle["baseline_leaves"]),
            "candidates": {
                kind: candidate.harness_record for kind, candidate in candidates.items()
            },
        }
        request["content_sha256"] = canonical_sha256(request)
        request_path = output_dir / "direct-runner-request.json"
        result_path = output_dir / "direct-runner-result.json"
        _write_json(request_path, request)
        completed = subprocess.run(
            (*executable, "--request", str(request_path), "--output", str(result_path)),
            check=False,
            capture_output=True,
            text=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace")
            raise ProbeError(
                f"DirectApplication runner failed with exit {completed.returncode}: "
                f"{stderr[-2000:]}"
            )
        try:
            payload = _strict_json_bytes(
                result_path.read_bytes(),
                "DirectApplication runner result",
            )
        except OSError as error:
            raise ProbeError("DirectApplication runner emitted no result") from error
        if (
            payload.get("kind") != RUNNER_RESULT_KIND
            or payload.get("schema_version") != RUNNER_SCHEMA_VERSION
        ):
            _die("DirectApplication runner result has the wrong contract")
        if payload.get("request_content_sha256") != request["content_sha256"]:
            _die("DirectApplication runner result references the wrong request")
        expected_digest = payload.pop("content_sha256", None)
        if expected_digest != canonical_sha256(payload):
            _die("DirectApplication runner result digest is invalid")
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, Mapping) or set(raw_candidates) != set(
            candidates
        ):
            _die("DirectApplication runner did not assess both candidates")
        return {
            str(kind): dict(value)
            for kind, value in raw_candidates.items()
            if isinstance(value, Mapping)
        }

    return run


def _validate_numerical_record(
    kind: str,
    record: Mapping[str, Any],
    *,
    label: str,
    expected_point_count: int,
) -> list[str]:
    if record.get("status") != "ok":
        return [f"{kind}: {label} numerical parity failed"]
    point_count = _positive_int(
        record.get("point_count"),
        f"{kind} {label} numerical point count",
    )
    if point_count != expected_point_count:
        _die(
            f"{kind} {label} numerical point count is {point_count}, "
            f"expected {expected_point_count}"
        )
    rtol = _finite_nonnegative(
        record.get("rtol"),
        f"{kind} {label} numerical rtol",
    )
    atol = _finite_nonnegative(
        record.get("atol"),
        f"{kind} {label} numerical atol",
    )
    if rtol != NUMERICAL_RTOL or atol != NUMERICAL_ATOL:
        _die(f"{kind} {label} numerical tolerances differ from the probe contract")
    _finite_nonnegative(
        record.get("max_absolute_difference"),
        f"{kind} {label} maximum absolute difference",
    )
    _finite_nonnegative(
        record.get("max_relative_difference"),
        f"{kind} {label} maximum relative difference",
    )
    return []


def _positive_float_series(
    value: object,
    label: str,
    *,
    expected_length: int,
) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_length:
        _die(f"{label} must contain exactly {expected_length} samples")
    return [
        _finite_positive(item, f"{label}[{index}]") for index, item in enumerate(value)
    ]


def _positive_int_series(
    value: object,
    label: str,
    *,
    expected_length: int,
) -> list[int]:
    if not isinstance(value, list) or len(value) != expected_length:
        _die(f"{label} must contain exactly {expected_length} samples")
    return [
        _positive_int(item, f"{label}[{index}]") for index, item in enumerate(value)
    ]


def _benchmark_speedup(
    kind: str,
    batch: int,
    record: Mapping[str, Any],
) -> float:
    baseline = _positive_float_series(
        record.get("baseline_samples_seconds_per_point"),
        f"{kind} batch {batch} baseline timing",
        expected_length=BENCHMARK_SAMPLE_COUNT,
    )
    candidate = _positive_float_series(
        record.get("candidate_samples_seconds_per_point"),
        f"{kind} batch {batch} candidate timing",
        expected_length=BENCHMARK_SAMPLE_COUNT,
    )
    expected_order = [
        "baseline-first" if index % 2 == 0 else "candidate-first"
        for index in range(BENCHMARK_SAMPLE_COUNT)
    ]
    if record.get("alternating_order") != expected_order:
        _die(f"{kind} batch {batch} timing order is not the strict alternation")
    _positive_int_series(
        record.get("baseline_iterations"),
        f"{kind} batch {batch} baseline iterations",
        expected_length=BENCHMARK_SAMPLE_COUNT,
    )
    _positive_int_series(
        record.get("candidate_iterations"),
        f"{kind} batch {batch} candidate iterations",
        expected_length=BENCHMARK_SAMPLE_COUNT,
    )
    baseline_median = _finite_positive(
        record.get("baseline_median_seconds_per_point"),
        f"{kind} batch {batch} baseline median",
    )
    candidate_median = _finite_positive(
        record.get("candidate_median_seconds_per_point"),
        f"{kind} batch {batch} candidate median",
    )
    expected_baseline_median = float(statistics.median(baseline))
    expected_candidate_median = float(statistics.median(candidate))
    if not math.isclose(
        baseline_median,
        expected_baseline_median,
        rel_tol=1.0e-12,
        abs_tol=1.0e-18,
    ):
        _die(f"{kind} batch {batch} baseline median is not recomputable")
    if not math.isclose(
        candidate_median,
        expected_candidate_median,
        rel_tol=1.0e-12,
        abs_tol=1.0e-18,
    ):
        _die(f"{kind} batch {batch} candidate median is not recomputable")
    speedup = _finite_number(
        record.get("speedup_fraction"),
        f"{kind} batch {batch} speedup",
    )
    expected_speedup = 1.0 - expected_candidate_median / expected_baseline_median
    if not math.isclose(
        speedup,
        expected_speedup,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        _die(f"{kind} batch {batch} speedup is not recomputable")
    return speedup


def _validate_projection(kind: str, direct: Mapping[str, Any]) -> None:
    projection = direct.get("projection")
    if not isinstance(projection, Mapping):
        _die(f"{kind} projection evidence is absent")
    expected_candidate_calls = 10 if kind == "pair" else 9
    baseline_calls = _positive_int(
        projection.get("baseline_call_count"),
        f"{kind} baseline projected call count",
    )
    candidate_calls = _positive_int(
        projection.get("candidate_call_count"),
        f"{kind} candidate projected call count",
    )
    if baseline_calls != 13 or candidate_calls != expected_candidate_calls:
        _die(f"{kind} projected schedule call counts are incompatible")
    for metric in (
        "input_plane_exposures",
        "output_plane_stores",
        "logical_input_exposures",
    ):
        baseline_value = _positive_int(
            projection.get(f"baseline_{metric}"),
            f"{kind} baseline projected {metric}",
        )
        candidate_value = _positive_int(
            projection.get(f"candidate_{metric}"),
            f"{kind} candidate projected {metric}",
        )
        if candidate_value >= baseline_value:
            _die(f"{kind} projected {metric} does not decrease")


def _candidate_assessment(
    kind: str,
    evidence: Mapping[str, Any],
) -> tuple[list[str], list[str], dict[int, float]]:
    hard_failures: list[str] = []
    threshold_failures: list[str] = []
    speedups: dict[int, float] = {}
    compile_record = evidence.get("compile")
    numerical = evidence.get("numerical")
    direct = evidence.get("direct")
    if not isinstance(compile_record, Mapping) or compile_record.get("status") != "ok":
        return [f"{kind}: compile failed"], threshold_failures, speedups
    payload = _positive_int(
        compile_record.get("payload_bytes"),
        f"{kind} payload bytes",
    )
    if payload > PAYLOAD_CAP_BYTES:
        hard_failures.append(f"{kind}: payload cap exceeded")
    replaced_payload = _positive_int(
        compile_record.get("replaced_source_payload_bytes"),
        f"{kind} replaced baseline source payload bytes",
    )
    if payload > replaced_payload:
        hard_failures.append(f"{kind}: source payload expands over the replaced tail")
    _finite_positive(
        compile_record.get("compile_seconds"),
        f"{kind} compile seconds",
    )
    rss = _finite_nonnegative(
        compile_record.get("process_peak_rss_gib"),
        f"{kind} process-lifetime peak RSS",
    )
    if rss > RSS_CAP_GIB:
        hard_failures.append(f"{kind}: RSS cap exceeded")
    if not isinstance(numerical, Mapping):
        hard_failures.append(f"{kind}: dense numerical parity failed")
    else:
        hard_failures.extend(
            _validate_numerical_record(
                kind,
                numerical,
                label="dense",
                expected_point_count=VALIDATION_POINT_COUNT,
            )
        )
    if not isinstance(direct, Mapping):
        return (
            [*hard_failures, f"{kind}: direct evidence absent"],
            threshold_failures,
            speedups,
        )
    lowering = direct.get("lowering")
    if not isinstance(lowering, Mapping) or lowering.get("status") != "ok":
        hard_failures.append(f"{kind}: DirectApplication lowering failed")
    else:
        try:
            _nonnegative_int(
                lowering.get("source_stack_bytes"),
                f"{kind} source stack bytes",
            )
            configured_limit = _positive_int(
                lowering.get("configured_stack_limit_bytes"),
                f"{kind} configured stack limit",
            )
            warmed_allocation = _nonnegative_int(
                lowering.get("warmed_arena_allocation_bytes"),
                f"{kind} warmed arena allocation bytes",
            )
        except ProbeError as error:
            hard_failures.append(f"{kind}: {error}")
            configured_limit = STACK_CAP_BYTES + 1
            warmed_allocation = -1
        if lowering.get("stack_limit_enforced") is not True:
            hard_failures.append(
                f"{kind}: DirectApplication stack limit was not enforced"
            )
        if configured_limit > STACK_CAP_BYTES:
            hard_failures.append(f"{kind}: configured stack cap exceeds probe cap")
        lowered_stack = lowering.get("lowered_stack_bytes")
        if lowered_stack is not None and (
            isinstance(lowered_stack, bool)
            or not isinstance(lowered_stack, int)
            or lowered_stack < 0
        ):
            hard_failures.append(f"{kind}: lowered stack bytes are invalid")
        elif isinstance(lowered_stack, int) and lowered_stack > STACK_CAP_BYTES:
            hard_failures.append(f"{kind}: DirectApplication stack cap exceeded")
        if warmed_allocation != 0:
            hard_failures.append(f"{kind}: warmed execution allocated arena bytes")
    direct_numerical = direct.get("numerical")
    if not isinstance(direct_numerical, Mapping):
        hard_failures.append(f"{kind}: direct numerical parity failed")
    else:
        hard_failures.extend(
            _validate_numerical_record(
                kind,
                direct_numerical,
                label="direct",
                expected_point_count=BENCHMARK_TILE_SIZE * len(BENCHMARK_BATCHES),
            )
        )
    benchmarks = direct.get("benchmarks")
    if not isinstance(benchmarks, Mapping):
        hard_failures.append(f"{kind}: benchmark evidence is absent")
    else:
        for batch in BENCHMARK_BATCHES:
            record = benchmarks.get(str(batch), benchmarks.get(batch))
            if not isinstance(record, Mapping):
                hard_failures.append(f"{kind}: batch {batch} benchmark is absent")
                continue
            speedup = _benchmark_speedup(kind, batch, record)
            speedups[batch] = speedup
            if speedup < MIN_PROJECTED_SPEEDUP:
                threshold_failures.append(
                    f"{kind}: batch {batch} speedup is below "
                    f"{MIN_PROJECTED_SPEEDUP:.0%}"
                )
    _validate_projection(kind, direct)
    return hard_failures, threshold_failures, speedups


def _candidate_failures(
    kind: str,
    evidence: Mapping[str, Any],
) -> list[str]:
    hard_failures, threshold_failures, _speedups = _candidate_assessment(
        kind,
        evidence,
    )
    return [*hard_failures, *threshold_failures]


def choose_candidate(candidates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the fail-closed pair/full-tail decision from the development plan."""

    if set(candidates) != {"pair", "full-tail"}:
        _die("probe decision requires exactly pair and full-tail candidates")
    pair = candidates["pair"]
    full = candidates["full-tail"]
    pair_hard, pair_threshold, pair_speedups = _candidate_assessment("pair", pair)
    full_hard, full_threshold, full_speedups = _candidate_assessment(
        "full-tail",
        full,
    )
    pair_failures = [*pair_hard, *pair_threshold]
    full_failures = [*full_hard, *full_threshold]

    comparison_failures: list[str] = []
    if not pair_hard and not full_hard and not full_threshold:
        pair_compile = pair["compile"]
        full_compile = full["compile"]
        assert isinstance(pair_compile, Mapping)
        assert isinstance(full_compile, Mapping)
        pair_payload = float(pair_compile["payload_bytes"])
        full_payload = float(full_compile["payload_bytes"])
        if full_payload > pair_payload * FULL_TAIL_PAYLOAD_RATIO:
            comparison_failures.append("full-tail: payload ratio exceeds 1.25")
        pair_seconds = _finite_nonnegative(
            pair_compile.get("compile_seconds"),
            "pair compile seconds",
        )
        full_seconds = _finite_nonnegative(
            full_compile.get("compile_seconds"),
            "full-tail compile seconds",
        )
        if full_seconds > pair_seconds * FULL_TAIL_RESOURCE_RATIO:
            comparison_failures.append("full-tail: compile-time ratio exceeds 1.5")
        for batch in BENCHMARK_BATCHES:
            pair_speedup = pair_speedups[batch]
            full_speedup = full_speedups[batch]
            if full_speedup < pair_speedup + FULL_TAIL_MARGIN:
                comparison_failures.append(
                    f"full-tail: batch {batch} margin is below {FULL_TAIL_MARGIN:.0%}"
                )

    if (
        not full_hard
        and not full_threshold
        and not comparison_failures
        and not pair_hard
    ):
        selected = "full-tail"
        accepted = True
        reasons: list[str] = []
    elif not pair_hard and not pair_threshold:
        selected = "pair"
        accepted = True
        reasons = [*full_failures, *comparison_failures]
    else:
        selected = "neither"
        accepted = False
        reasons = [*pair_failures, *full_failures, *comparison_failures]
    return {
        "accepted": accepted,
        "selected": selected,
        "minimum_projected_speedup_fraction": MIN_PROJECTED_SPEEDUP,
        "full_tail_required_margin_fraction": FULL_TAIL_MARGIN,
        "reasons": reasons,
    }


def _replaced_source_payload_bytes(
    capture: CapturedSchedule,
    kind: str,
) -> int:
    raw_leaves = capture.leaf_bundle.get("baseline_leaves")
    if not isinstance(raw_leaves, list) or len(raw_leaves) != 13:
        _die("baseline source-expansion gate requires thirteen leaves")
    indices = range(8, 12) if kind == "pair" else range(8, 13)
    total = 0
    for index in indices:
        leaf = raw_leaves[index]
        if not isinstance(leaf, Mapping):
            _die("baseline source-expansion leaf is malformed")
        source = leaf.get("source_application")
        if not isinstance(source, Mapping):
            _die("baseline source-expansion leaf has no source application")
        total += _positive_int(
            source.get("size_bytes"),
            f"baseline leaf {index} source size",
        )
    return total


def run_probe(
    capture: CapturedSchedule,
    compositions: Sequence[Any],
    output_dir: Path,
    *,
    compiler: CandidateCompiler = compile_o3_candidate,
    numerical_validator: NumericalValidator = validate_dense_candidates,
    direct_runner: DirectRunner = _unavailable_direct_runner,
) -> dict[str, Any]:
    """Compile, validate, assess, and decide without changing production code."""

    output_dir.mkdir(parents=True, exist_ok=True)
    compiled: dict[str, CompiledCandidate] = {}
    for composition in compositions:
        kind = str(composition.kind)
        if kind in compiled:
            _die(f"composer returned duplicate candidate {kind!r}")
        compiled[kind] = compiler(composition, output_dir / "candidates")
    if set(compiled) != {"pair", "full-tail"}:
        _die("composer did not return pair and full-tail candidates")
    numerical = numerical_validator(capture, compiled)
    direct = direct_runner(capture, compiled, output_dir)
    evidence: dict[str, dict[str, Any]] = {}
    for kind, candidate in compiled.items():
        raw_numerical = numerical.get(kind)
        raw_direct = direct.get(kind)
        if not isinstance(raw_numerical, Mapping) or not isinstance(
            raw_direct, Mapping
        ):
            _die(f"probe hooks did not assess {kind}")
        compile_evidence = dict(candidate.compile_evidence)
        compile_evidence["replaced_source_payload_bytes"] = (
            _replaced_source_payload_bytes(capture, kind)
        )
        evidence[kind] = {
            "compile": compile_evidence,
            "numerical": dict(raw_numerical),
            "direct": dict(raw_direct),
            "harness": dict(candidate.harness_record),
        }
    decision = choose_candidate(evidence)
    payload: dict[str, Any] = {
        "kind": RESULT_KIND,
        "schema_version": RESULT_SCHEMA_VERSION,
        "process": NORMALIZED_PROCESS,
        "process_id": PROCESS_ID,
        "selected_flow": SELECTED_FLOW,
        "capture": dict(capture.capture_evidence),
        "leaf_bundle": dict(capture.leaf_bundle),
        "candidates": evidence,
        "decision": decision,
        "caps": {
            "payload_bytes": PAYLOAD_CAP_BYTES,
            "stack_bytes": STACK_CAP_BYTES,
            "peak_rss_gib": RSS_CAP_GIB,
            "rtol": NUMERICAL_RTOL,
            "atol": NUMERICAL_ATOL,
        },
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        _die(f"cannot resolve probe Git identity: {' '.join(arguments)}")
    return completed.stdout.strip()


def _verify_source_checkout() -> dict[str, Any]:
    revision = _git_output("rev-parse", "HEAD")
    if len(revision) != 40:
        _die("probe source revision is invalid")
    top_level = Path(_git_output("rev-parse", "--show-toplevel")).resolve()
    if top_level != ROOT.resolve():
        _die("probe source checkout does not own the tool")
    dirty = _git_output("status", "--porcelain", "--untracked-files=no")
    if dirty:
        _die("probe source checkout has tracked changes")

    import pyamplicol
    from pyamplicol.generation import compiled_terminal_superkernel

    package_path = Path(pyamplicol.__file__).resolve(strict=True)
    composer_path = Path(compiled_terminal_superkernel.__file__).resolve(strict=True)
    try:
        package_relative = package_path.relative_to(SRC.resolve())
        composer_relative = composer_path.relative_to(SRC.resolve())
        composer_repo_relative = composer_path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ProbeError(
            "probe imported pyamplicol outside the source checkout"
        ) from error
    if _git_output("ls-files", "--error-unmatch", str(composer_repo_relative)) != str(
        composer_repo_relative
    ):
        _die("probe composer is not tracked by the source revision")
    return {
        "revision": revision,
        "pyamplicol_module": str(package_relative),
        "composer_module": str(composer_relative),
        "tracked_tree_clean": True,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="new disposable directory for the baseline, candidates, and evidence",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--direct-runner",
        nargs="+",
        help="out-of-tree strict-JSON DirectApplication arena harness command",
    )
    parser.add_argument(
        "--run-heavy-probe",
        action="store_true",
        help="required acknowledgement that generation and O3 compilation are intended",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not args.run_heavy_probe:
        _die("refusing to run without --run-heavy-probe")
    source_identity = _verify_source_checkout()
    output_root = args.output_root.expanduser()
    if output_root.exists():
        _die(f"probe output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    capture = capture_exact_selected_blueprint(output_root)
    try:
        from pyamplicol.generation.compiled_terminal_superkernel import (
            compose_terminal_superkernels,
        )
    except ImportError as error:
        raise ProbeError("terminal superkernel composer is unavailable") from error
    pair, full = compose_terminal_superkernels(
        capture.blueprint,
        execution_mode="compiled",
        backend="jit",
        optimization_level=3,
        selector_structural_zero_proven=bool(
            capture.capture_evidence.get("selected_lane_proof", {}).get("status")
            == "proven"
        ),
    )
    runner = (
        _unavailable_direct_runner
        if args.direct_runner is None
        else external_direct_runner(args.direct_runner)
    )
    payload = run_probe(
        capture,
        (pair, full),
        output_root,
        direct_runner=runner,
    )
    payload["probe_identity"] = {
        "source": source_identity,
        "direct_runner_command": (
            None if args.direct_runner is None else list(args.direct_runner)
        ),
    }
    payload["content_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    json_out = (
        output_root / "compiled-terminal-superkernel-probe.json"
        if args.json_out is None
        else args.json_out.expanduser()
    )
    _write_json(json_out, payload)
    print(
        json.dumps(
            {
                "result": str(json_out.resolve(strict=True)),
                "decision": payload["decision"],
                "content_sha256": payload["content_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if bool(payload["decision"]["accepted"]) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProbeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
