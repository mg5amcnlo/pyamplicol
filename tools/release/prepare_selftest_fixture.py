# SPDX-License-Identifier: 0BSD
"""Prepare the portable 64-bit installed-package physics self-test fixture."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import shutil
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "src" / "pyamplicol" / "assets" / "selftest"
PORTABLE_TEMPLATE = "portable-64le"
PORTABLE_OPTIMIZATION_LEVEL = 2
PORTABLE_DIRECT_CODEGEN_OPTIMIZATION_LEVEL = 3
COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY = "rusticol.compiled.plane-arena.v2"
COMPILED_STAGE_PLAN_ABI = "pyamplicol-compiled-stage-plan-v2"
COMPILED_PLANE_DIRECT_APPLICATION_ABI = "symjit-direct-application-storage-v1"
SYMJIT_APPLICATION_ABI = "symjit-application-storage-v3"
DIRECT_TABLE_DESCRIPTOR_ABI = "symjit-direct-table-descriptor-v1"
DIRECT_TABLE_BINDING_ABI = "symjit-direct-table-binding-v1"
SYMJIT_F64_RUNTIME_CAPABILITY = "symjit.application.complex-f64.v1"
COMPATIBLE_TARGETS = (
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
)


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _artifact_id(manifest: Mapping[str, object]) -> str:
    content = dict(manifest)
    content.pop("artifact_id", None)
    return hashlib.sha256(_canonical_json(content)).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compiled_model_payload(
    manifest: Mapping[str, object],
) -> tuple[dict[str, Any], Path]:
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("artifact payload inventory is invalid")
    matches = [
        payload
        for payload in payloads
        if isinstance(payload, dict) and payload.get("role") == "compiled-model"
    ]
    if len(matches) != 1:
        raise RuntimeError("self-test artifact must contain one compiled model")
    payload = matches[0]
    relative = payload.get("path")
    if not isinstance(relative, str):
        raise RuntimeError("self-test compiled-model payload has no path")
    path = Path(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError("self-test compiled-model payload path is unsafe")
    return payload, path


def _validate_source_compiled_model(
    source: Path,
    manifest: Mapping[str, object],
) -> None:
    """Require the source artifact to match the active model compiler exactly."""

    from pyamplicol.models.loading import load_compiled_model

    _payload, relative = _compiled_model_payload(manifest)
    load_compiled_model(source / relative)


def _normalize_compiled_model_version(
    artifact: Path,
    manifest: Mapping[str, object],
    *,
    version: str,
) -> None:
    """Retarget the embedded compiler fingerprint to the release package version."""

    payload, relative = _compiled_model_payload(manifest)
    path = artifact / relative
    try:
        compiled_model = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("self-test compiled model is invalid") from error
    if not isinstance(compiled_model, dict):
        raise RuntimeError("self-test compiled model must be an object")
    producer = compiled_model.get("producer")
    if not isinstance(producer, dict):
        raise RuntimeError("self-test compiled-model producer is invalid")
    producer["pyamplicol"] = version
    path.write_bytes(_canonical_json(compiled_model))
    payload["sha256"] = _sha256(path)
    payload["size_bytes"] = path.stat().st_size


def _workspace_version() -> str:
    with (ROOT / "Cargo.toml").open("rb") as stream:
        cargo = tomllib.load(stream)
    try:
        version = cargo["workspace"]["package"]["version"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Cargo.toml has no workspace package version") from exc
    if not isinstance(version, str) or not version:
        raise RuntimeError("Cargo workspace package version is invalid")
    return version.replace("-dev.", ".dev")


def _walk_evaluator_manifests(value: object) -> Sequence[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("kind") == "symjit-application-evaluator":
            found.append(value)
        for child in value.values():
            found.extend(_walk_evaluator_manifests(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_evaluator_manifests(child))
    return found


def _walk_compiled_execution_manifests(
    value: object,
) -> Sequence[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        compiled = value.get("compiled")
        if (
            isinstance(compiled, dict)
            and compiled.get("kind") == "generic-dag-stage-blueprint"
        ):
            found.append(value)
        for child in value.values():
            found.extend(_walk_compiled_execution_manifests(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_compiled_execution_manifests(child))
    return found


def _portable_nonnegative_int(
    record: Mapping[str, object],
    name: str,
    *,
    context: str,
) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{context} has an invalid {name.replace('_', ' ')}")
    return value


def _portable_sequence(value: object, *, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise RuntimeError(f"{context} must be a list")
    return value


def _validate_portable_capability_owner(
    owner: Mapping[str, object],
    *,
    context: str,
) -> None:
    capabilities = _portable_sequence(
        owner.get("required_runtime_capabilities"),
        context=f"{context} runtime capabilities",
    )
    if any(
        not isinstance(capability, str) or not capability for capability in capabilities
    ) or len(capabilities) != len(set(capabilities)):
        raise RuntimeError(f"{context} runtime capabilities are invalid")
    for capability in (
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
        SYMJIT_F64_RUNTIME_CAPABILITY,
    ):
        if capability not in capabilities:
            raise RuntimeError(f"{context} must require {capability}")


def validate_portable_artifact_capabilities(
    manifest: Mapping[str, object],
    *,
    context: str,
) -> int:
    """Require Arena and SymJIT capabilities at artifact and process scope."""

    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError(f"{context} has no artifact runtime contract")
    _validate_portable_capability_owner(
        runtime,
        context=f"{context} artifact runtime",
    )

    processes = _portable_sequence(
        manifest.get("processes"),
        context=f"{context} processes",
    )
    if not processes:
        raise RuntimeError(f"{context} has no processes")
    for process_index, process in enumerate(processes):
        if not isinstance(process, Mapping):
            raise RuntimeError(f"{context} process {process_index} is invalid")
        _validate_portable_capability_owner(
            process,
            context=f"{context} process {process_index}",
        )
    return len(processes)


def _validate_portable_symjit_source(
    evaluator: Mapping[str, object],
) -> None:
    if (
        evaluator.get("compiler_type") != "native"
        or evaluator.get("translation_mode") != "indirect"
    ):
        raise RuntimeError(
            "portable self-test requires native indirect SymJIT evaluators"
        )
    if evaluator.get("optimization_level") != PORTABLE_OPTIMIZATION_LEVEL:
        raise RuntimeError(
            "portable self-test source must use SymJIT optimization level "
            f"{PORTABLE_OPTIMIZATION_LEVEL}; other optimization levels "
            "may contain source-architecture register allocation"
        )


def _portable_residual_source_leaves(
    evaluator: Mapping[str, object],
    parent_inputs: tuple[int, ...],
    output_start: int,
    *,
    context: str,
) -> tuple[list[dict[str, object]], int]:
    kind = evaluator.get("kind")
    if kind == "compiled-stage-empty-residual":
        if (
            evaluator.get("input_len") != 0
            or evaluator.get("output_len") != 0
            or evaluator.get("required_runtime_capabilities") != []
            or parent_inputs
            or output_start != 0
        ):
            raise RuntimeError(f"{context} has an invalid empty residual evaluator")
        return [], 0
    if kind == "chunked-symbolica-evaluator":
        input_len = evaluator.get("input_len")
        if input_len is not None and input_len != len(parent_inputs):
            raise RuntimeError(f"{context} chunk input width is inconsistent")
        chunks = _portable_sequence(
            evaluator.get("chunks"),
            context=f"{context} chunks",
        )
        if not chunks:
            raise RuntimeError(f"{context} has no evaluator leaves")
        raw_maps = evaluator.get("chunk_input_indices")
        if raw_maps is None:
            input_maps = [parent_inputs] * len(chunks)
        else:
            maps = _portable_sequence(
                raw_maps,
                context=f"{context} chunk input maps",
            )
            if len(maps) != len(chunks):
                raise RuntimeError(f"{context} chunk input maps are incomplete")
            input_maps: list[tuple[int, ...]] = []
            for raw_map in maps:
                indices = _portable_sequence(
                    raw_map,
                    context=f"{context} chunk input map",
                )
                mapped: list[int] = []
                for raw_index in indices:
                    if (
                        isinstance(raw_index, bool)
                        or not isinstance(raw_index, int)
                        or not 0 <= raw_index < len(parent_inputs)
                    ):
                        raise RuntimeError(
                            f"{context} chunk input map references an absent input"
                        )
                    mapped.append(parent_inputs[raw_index])
                input_maps.append(tuple(mapped))

        result: list[dict[str, object]] = []
        cursor = output_start
        for chunk_index, (raw_chunk, mapped) in enumerate(
            zip(chunks, input_maps, strict=True)
        ):
            if not isinstance(raw_chunk, Mapping):
                raise RuntimeError(f"{context} evaluator child is invalid")
            child, cursor = _portable_residual_source_leaves(
                raw_chunk,
                mapped,
                cursor,
                context=f"{context} chunk {chunk_index}",
            )
            result.extend(child)
        return result, cursor

    if kind != "symjit-application-evaluator":
        raise RuntimeError(f"{context} is not an all-SymJIT compiled Plane-Arena stage")
    _validate_portable_symjit_source(evaluator)
    if (
        evaluator.get("runtime_capability") != SYMJIT_F64_RUNTIME_CAPABILITY
        or evaluator.get("application_abi") != SYMJIT_APPLICATION_ABI
        or evaluator.get("element_layout") != "complex-f64"
        or evaluator.get("batch_layout") != "row-major"
    ):
        raise RuntimeError(f"{context} has an incompatible SymJIT source ABI")
    application_path = evaluator.get("application_path")
    if not isinstance(application_path, str) or not application_path:
        raise RuntimeError(f"{context} has no SymJIT application path")
    input_len = _portable_nonnegative_int(evaluator, "input_len", context=context)
    output_len = _portable_nonnegative_int(evaluator, "output_len", context=context)
    if input_len != len(parent_inputs) or output_len == 0:
        raise RuntimeError(f"{context} has inconsistent evaluator dimensions")
    output_stop = output_start + output_len
    return (
        [
            {
                "application_path": application_path,
                "source_application_abi": SYMJIT_APPLICATION_ABI,
                "optimization_level": PORTABLE_OPTIMIZATION_LEVEL,
                "direct_codegen_optimization_level": (
                    PORTABLE_DIRECT_CODEGEN_OPTIMIZATION_LEVEL
                ),
                "input_len": input_len,
                "output_len": output_len,
                "input_indices": list(parent_inputs),
                "output_start": output_start,
                "output_stop": output_stop,
            }
        ],
        output_stop,
    )


def _portable_stage_input_bindings(
    stage: Mapping[str, object],
    parameter_count: int,
    *,
    context: str,
) -> list[dict[str, object]]:
    inputs = _portable_sequence(
        stage.get("input_components"),
        context=f"{context} input components",
    )
    expected: list[dict[str, object] | None] = [None] * parameter_count
    for raw_input in inputs:
        if not isinstance(raw_input, Mapping):
            raise RuntimeError(f"{context} input component is invalid")
        parameter_index = _portable_nonnegative_int(
            raw_input,
            "parameter_index",
            context=f"{context} input component",
        )
        if (
            parameter_index >= parameter_count
            or expected[parameter_index] is not None
            or raw_input.get("kind") not in {"value", "momentum", "model_parameter"}
            or not isinstance(raw_input.get("real_valued"), bool)
        ):
            raise RuntimeError(f"{context} input components are inconsistent")
        binding = {
            "parameter_index": parameter_index,
            "kind": raw_input["kind"],
            "source_id": _portable_nonnegative_int(
                raw_input,
                "source_id",
                context=f"{context} input component",
            ),
            "component": _portable_nonnegative_int(
                raw_input,
                "component",
                context=f"{context} input component",
            ),
            "global_component": _portable_nonnegative_int(
                raw_input,
                "global_component",
                context=f"{context} input component",
            ),
            "real_valued": raw_input["real_valued"],
        }
        expected[parameter_index] = binding
    if any(binding is None for binding in expected):
        raise RuntimeError(f"{context} input components are incomplete")
    return [binding for binding in expected if binding is not None]


def _portable_stage_output_bindings(
    stage: Mapping[str, object],
    output_length: int,
    *,
    arena: str,
    context: str,
) -> list[dict[str, object]]:
    slots = _portable_sequence(
        stage.get("output_slots"),
        context=f"{context} output slots",
    )
    expected: list[dict[str, object] | None] = [None] * output_length
    seen_components: set[int] = set()
    for raw_slot in slots:
        if not isinstance(raw_slot, Mapping):
            raise RuntimeError(f"{context} output slot is invalid")
        output_start = _portable_nonnegative_int(
            raw_slot,
            "output_start",
            context=f"{context} output slot",
        )
        output_stop = _portable_nonnegative_int(
            raw_slot,
            "output_stop",
            context=f"{context} output slot",
        )
        component_start = _portable_nonnegative_int(
            raw_slot,
            "component_start",
            context=f"{context} output slot",
        )
        component_stop = _portable_nonnegative_int(
            raw_slot,
            "component_stop",
            context=f"{context} output slot",
        )
        if (
            output_stop < output_start
            or output_stop > output_length
            or component_stop - component_start != output_stop - output_start
        ):
            raise RuntimeError(f"{context} output slot range is inconsistent")
        for offset in range(output_stop - output_start):
            output_index = output_start + offset
            component = component_start + offset
            if expected[output_index] is not None or component in seen_components:
                raise RuntimeError(f"{context} output slots alias")
            expected[output_index] = {
                "output_index": output_index,
                "arena": arena,
                "component": component,
            }
            seen_components.add(component)
    if any(binding is None for binding in expected):
        raise RuntimeError(f"{context} output slots are incomplete")
    return [binding for binding in expected if binding is not None]


def _portable_sorted_unique_nonnegative_ints(
    value: object,
    *,
    context: str,
) -> list[int]:
    values = _portable_sequence(value, context=context)
    result: list[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise RuntimeError(f"{context} contains an invalid index")
        result.append(item)
    if result != sorted(set(result)):
        raise RuntimeError(f"{context} is not canonical")
    return result


def _portable_sorted_unique_ints(
    value: object,
    *,
    context: str,
) -> list[int]:
    values = _portable_sequence(value, context=context)
    result: list[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int):
            raise RuntimeError(f"{context} contains an invalid index")
        result.append(item)
    if result != sorted(set(result)):
        raise RuntimeError(f"{context} is not canonical")
    return result


def _validate_portable_table_kernels(
    descriptor: Mapping[str, object],
    *,
    context: str,
) -> list[Mapping[str, object]]:
    raw_kernels = _portable_sequence(
        descriptor.get("table_kernels"),
        context=f"{context} compiled stage-plan table kernels",
    )
    if len(raw_kernels) > 8:
        raise RuntimeError(f"{context} exceeds the eight-kernel slice cap")
    kernels: list[Mapping[str, object]] = []
    signatures: set[tuple[object, object]] = set()
    for kernel_index, raw_kernel in enumerate(raw_kernels):
        if not isinstance(raw_kernel, Mapping):
            raise RuntimeError(f"{context} table kernel {kernel_index} is invalid")
        input_count = _portable_nonnegative_int(
            raw_kernel,
            "input_complex_count",
            context=f"{context} table kernel {kernel_index}",
        )
        output_count = _portable_nonnegative_int(
            raw_kernel,
            "output_complex_count",
            context=f"{context} table kernel {kernel_index}",
        )
        role = raw_kernel.get("role")
        signature = (role, raw_kernel.get("canonical_signature"))
        if (
            raw_kernel.get("table_kernel_id") != kernel_index
            or role not in {"contribution", "finalizer"}
            or not isinstance(signature[1], str)
            or not signature[1]
            or signature in signatures
            or not 1 <= input_count <= 16
            or not 1 <= output_count <= 4
            or raw_kernel.get("scalar_input_count") != 0
            or raw_kernel.get("optimization_level") != 3
            or raw_kernel.get("source_application_abi") != SYMJIT_APPLICATION_ABI
            or raw_kernel.get("descriptor_abi") != DIRECT_TABLE_DESCRIPTOR_ABI
            or raw_kernel.get("binding_abi") != DIRECT_TABLE_BINDING_ABI
        ):
            raise RuntimeError(
                f"{context} table kernel {kernel_index} has an invalid ABI or shape"
            )
        input_contracts = _portable_sequence(
            raw_kernel.get("input_contracts"),
            context=f"{context} table kernel {kernel_index} input contracts",
        )
        output_layout = _portable_sequence(
            raw_kernel.get("output_layout"),
            context=f"{context} table kernel {kernel_index} output layout",
        )
        if len(input_contracts) != input_count or len(output_layout) != output_count:
            raise RuntimeError(
                f"{context} table kernel {kernel_index} dimensions are inconsistent"
            )
        for name in ("source_application", "descriptor"):
            payload = raw_kernel.get(name)
            if (
                not isinstance(payload, Mapping)
                or not isinstance(payload.get("path"), str)
                or not payload["path"]
                or _portable_nonnegative_int(
                    payload,
                    "size_bytes",
                    context=f"{context} table kernel {kernel_index} {name}",
                )
                == 0
            ):
                raise RuntimeError(
                    f"{context} table kernel {kernel_index} has an empty {name}"
                )
        signatures.add(signature)
        kernels.append(raw_kernel)
    return kernels


def _validate_portable_table_rows(
    value: object,
    *,
    expected_row_size: int,
    context: str,
) -> int:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} is invalid")
    count = _portable_nonnegative_int(value, "count", context=context)
    row_size = _portable_nonnegative_int(value, "row_size", context=context)
    size_bytes = _portable_nonnegative_int(value, "size_bytes", context=context)
    if (
        count == 0
        or row_size != expected_row_size
        or size_bytes != count * row_size
        or not isinstance(value.get("path"), str)
        or not value["path"]
    ):
        raise RuntimeError(f"{context} has an invalid shape")
    return count


def _validate_portable_table_calls(
    value: object,
    kernels: Sequence[Mapping[str, object]],
    selector_partitions: Sequence[object],
    *,
    expected_role: str,
    context: str,
) -> set[int]:
    calls = _portable_sequence(
        value,
        context=f"{context} compiled stage-plan {expected_role} calls",
    )
    owned: set[int] = set()
    for call_index, raw_call in enumerate(calls):
        if not isinstance(raw_call, Mapping):
            raise RuntimeError(
                f"{context} {expected_role} call {call_index} is invalid"
            )
        kernel_id = _portable_nonnegative_int(
            raw_call,
            "table_kernel_id",
            context=f"{context} {expected_role} call {call_index}",
        )
        if kernel_id >= len(kernels) or kernels[kernel_id].get("role") != expected_role:
            raise RuntimeError(
                f"{context} {expected_role} call {call_index} references "
                "an absent kernel"
            )
        call_owned = _portable_sorted_unique_nonnegative_ints(
            raw_call.get("owned_current_ids"),
            context=f"{context} {expected_role} call {call_index} ownership",
        )
        if not call_owned:
            raise RuntimeError(f"{context} {expected_role} call has no owner")
        _portable_sorted_unique_nonnegative_ints(
            raw_call.get("dependency_current_ids"),
            context=f"{context} {expected_role} call {call_index} dependencies",
        )
        _portable_sorted_unique_nonnegative_ints(
            raw_call.get("dependency_current_components"),
            context=f"{context} {expected_role} call {call_index} components",
        )
        partitions = _portable_sorted_unique_nonnegative_ints(
            raw_call.get("selector_partition_ids"),
            context=f"{context} {expected_role} call {call_index} partitions",
        )
        if len(partitions) != 1 or partitions[0] >= len(selector_partitions):
            raise RuntimeError(
                f"{context} {expected_role} call has an invalid selector partition"
            )
        kernel = kernels[kernel_id]
        input_count = int(kernel["input_complex_count"])
        output_count = int(kernel["output_complex_count"])
        _validate_portable_table_rows(
            raw_call.get("invocation_rows"),
            expected_row_size=(input_count * 2 + 2) * 4,
            context=f"{context} {expected_role} call {call_index} invocation rows",
        )
        _validate_portable_table_rows(
            raw_call.get("attachment_rows"),
            expected_row_size=(output_count * 2 + 2) * 4,
            context=f"{context} {expected_role} call {call_index} attachment rows",
        )
        owned.update(call_owned)
    return owned


def _validate_portable_execution_order(
    descriptor: Mapping[str, object],
    residual_chunks: set[int],
    selector_partitions: Sequence[object],
    *,
    context: str,
) -> None:
    order = _portable_sequence(
        descriptor.get("execution_order"),
        context=f"{context} compiled stage-plan execution order",
    )
    table_calls = _portable_sequence(
        descriptor.get("table_calls"),
        context=f"{context} compiled stage-plan table calls",
    )
    finalizer_calls = _portable_sequence(
        descriptor.get("finalizer_calls"),
        context=f"{context} compiled stage-plan finalizer calls",
    )
    residual_leaves = _portable_sequence(
        descriptor.get("residual_leaves"),
        context=f"{context} compiled stage-plan residual leaves",
    )
    expected_count = len(residual_leaves) + len(table_calls) + len(finalizer_calls)
    if len(order) != expected_count:
        raise RuntimeError(f"{context} execution order is incomplete")
    seen = {
        "residual-leaf": set(),
        "table-call": set(),
        "finalizer-call": set(),
    }
    limits = {
        "residual-leaf": len(residual_leaves),
        "table-call": len(table_calls),
        "finalizer-call": len(finalizer_calls),
    }
    chunks: list[int] = []
    for raw_unit in order:
        if not isinstance(raw_unit, Mapping) or raw_unit.get("kind") not in limits:
            raise RuntimeError(f"{context} execution order contains an invalid unit")
        kind = str(raw_unit["kind"])
        index = _portable_nonnegative_int(
            raw_unit,
            "index",
            context=f"{context} execution unit",
        )
        chunk = _portable_nonnegative_int(
            raw_unit,
            "original_chunk_index",
            context=f"{context} execution unit",
        )
        if index >= limits[kind] or index in seen[kind]:
            raise RuntimeError(f"{context} execution order contains a duplicate unit")
        if kind == "residual-leaf":
            leaf = residual_leaves[index]
            if (
                not isinstance(leaf, Mapping)
                or leaf.get("original_chunk_index") != chunk
            ):
                raise RuntimeError(
                    f"{context} residual execution unit lost its original chunk"
                )
        seen[kind].add(index)
        chunks.append(chunk)
    if chunks != sorted(chunks) or set(chunks) != set(
        range(max(chunks, default=-1) + 1)
    ):
        raise RuntimeError(
            f"{context} execution order reverses or skips original chunks"
        )
    if not residual_chunks.issubset(chunks):
        raise RuntimeError(f"{context} execution order omits a residual chunk")

    covered = [False] * len(order)
    for expected_id, raw_partition in enumerate(selector_partitions):
        if not isinstance(raw_partition, Mapping):
            raise RuntimeError(f"{context} selector partition {expected_id} is invalid")
        if raw_partition.get("partition_id") != expected_id:
            raise RuntimeError(f"{context} selector partitions are not dense")
        for name in ("helicity_selector_domain_ids", "color_selector_domain_ids"):
            _portable_sorted_unique_ints(
                raw_partition.get(name),
                context=f"{context} selector partition {expected_id} {name}",
            )
        _portable_sorted_unique_nonnegative_ints(
            raw_partition.get("original_chunk_indices"),
            context=(
                f"{context} selector partition {expected_id} original_chunk_indices"
            ),
        )
        partition_chunks = set(raw_partition["original_chunk_indices"])
        for unit_index, raw_unit in enumerate(order):
            assert isinstance(raw_unit, Mapping)
            if raw_unit["original_chunk_index"] in partition_chunks:
                if covered[unit_index]:
                    raise RuntimeError(f"{context} selector partitions overlap")
                covered[unit_index] = True
    if any(not item for item in covered):
        raise RuntimeError(f"{context} selector partitions are incomplete")


def _validate_portable_arena_stage(
    stage: Mapping[str, object],
    *,
    arena: str,
    context: str,
) -> None:
    if stage.get("parameter_layout") != "stage-local-value-momentum":
        raise RuntimeError(f"{context} does not use stage-local parameters")
    parameter_count = _portable_nonnegative_int(
        stage,
        "parameter_count",
        context=context,
    )
    output_length = _portable_nonnegative_int(
        stage,
        "output_length",
        context=context,
    )
    if output_length == 0:
        raise RuntimeError(f"{context} has no outputs")
    evaluator = stage.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise RuntimeError(f"{context} has no compiled evaluator")
    _outer_leaves, outer_output_length = _portable_residual_source_leaves(
        evaluator,
        tuple(range(parameter_count)),
        0,
        context=f"{context} exact/reference evaluator",
    )
    if outer_output_length != output_length:
        raise RuntimeError(
            f"{context} exact/reference evaluator does not cover the original stage"
        )
    descriptor = stage.get("compiled_plane_arena")
    if not isinstance(descriptor, Mapping):
        raise RuntimeError(
            f"{context} requires a complete compiled_plane_arena descriptor"
        )

    contract = {
        "schema_version": 2,
        "kind": "compiled-stage-plan",
        "plan_abi": COMPILED_STAGE_PLAN_ABI,
        "residual_application_abi": COMPILED_PLANE_DIRECT_APPLICATION_ABI,
        "table_source_application_abi": SYMJIT_APPLICATION_ABI,
        "direct_table_descriptor_abi": DIRECT_TABLE_DESCRIPTOR_ABI,
        "direct_table_binding_abi": DIRECT_TABLE_BINDING_ABI,
        "element_layout": "split-complex-component-major",
    }
    if any(descriptor.get(name) != value for name, value in contract.items()):
        raise RuntimeError(
            f"{context} is not a compiled stage-plan v2 with DirectTable binding-v1"
        )

    expected_inputs = _portable_stage_input_bindings(
        stage,
        parameter_count,
        context=context,
    )
    input_bindings = _portable_sequence(
        descriptor.get("input_bindings"),
        context=f"{context} compiled_plane_arena input bindings",
    )
    if list(input_bindings) != expected_inputs:
        raise RuntimeError(
            f"{context} compiled_plane_arena input bindings are incomplete"
        )

    canonical_outputs = _portable_stage_output_bindings(
        stage,
        output_length,
        arena=arena,
        context=context,
    )
    residual_evaluator = descriptor.get("residual_evaluator")
    if not isinstance(residual_evaluator, Mapping):
        raise RuntimeError(f"{context} compiled stage-plan has no residual evaluator")
    residual_inputs = (
        ()
        if residual_evaluator.get("kind") == "compiled-stage-empty-residual"
        else tuple(range(parameter_count))
    )
    expected_leaves, residual_output_length = _portable_residual_source_leaves(
        residual_evaluator,
        residual_inputs,
        0,
        context=f"{context} residual evaluator",
    )
    output_bindings = _portable_sequence(
        descriptor.get("output_bindings"),
        context=f"{context} compiled stage-plan output bindings",
    )
    residual_outputs: list[dict[str, object] | None] = [None] * residual_output_length
    seen_original_outputs: set[int] = set()
    for raw_binding in output_bindings:
        if not isinstance(raw_binding, Mapping):
            raise RuntimeError(f"{context} residual output binding is invalid")
        output_index = _portable_nonnegative_int(
            raw_binding,
            "output_index",
            context=f"{context} residual output binding",
        )
        original_output_index = _portable_nonnegative_int(
            raw_binding,
            "original_output_index",
            context=f"{context} residual output binding",
        )
        if (
            output_index >= residual_output_length
            or original_output_index >= output_length
            or residual_outputs[output_index] is not None
            or original_output_index in seen_original_outputs
        ):
            raise RuntimeError(f"{context} residual output bindings alias or escape")
        canonical = canonical_outputs[original_output_index]
        if (
            raw_binding.get("arena") != canonical["arena"]
            or raw_binding.get("component") != canonical["component"]
        ):
            raise RuntimeError(
                f"{context} residual output binding disagrees with the original DAG"
            )
        residual_outputs[output_index] = dict(raw_binding)
        seen_original_outputs.add(original_output_index)
    if any(binding is None for binding in residual_outputs):
        raise RuntimeError(f"{context} residual output bindings contain holes")

    leaves = _portable_sequence(
        descriptor.get("residual_leaves"),
        context=f"{context} compiled stage-plan residual leaves",
    )
    if len(leaves) != len(expected_leaves):
        raise RuntimeError(f"{context} residual leaf bindings are incomplete")
    residual_chunks: set[int] = set()
    for leaf_index, (raw_leaf, expected_leaf) in enumerate(
        zip(leaves, expected_leaves, strict=True)
    ):
        if not isinstance(raw_leaf, Mapping):
            raise RuntimeError(f"{context} residual leaf {leaf_index} is invalid")
        original_chunk_index = _portable_nonnegative_int(
            raw_leaf,
            "original_chunk_index",
            context=f"{context} residual leaf {leaf_index}",
        )
        if (
            raw_leaf.get("residual_leaf_index") != leaf_index
            or original_chunk_index in residual_chunks
            or any(raw_leaf.get(name) != value for name, value in expected_leaf.items())
            or raw_leaf.get("direct_codegen_optimization_level")
            != PORTABLE_DIRECT_CODEGEN_OPTIMIZATION_LEVEL
        ):
            raise RuntimeError(
                f"{context} residual leaf {leaf_index} does not bind its "
                "portable SymJIT source at direct codegen optimization level "
                f"{PORTABLE_DIRECT_CODEGEN_OPTIMIZATION_LEVEL}"
            )
        residual_chunks.add(original_chunk_index)

    table_kernels = _validate_portable_table_kernels(
        descriptor,
        context=context,
    )
    selector_partitions = _portable_sequence(
        descriptor.get("selector_partitions"),
        context=f"{context} compiled stage-plan selector partitions",
    )
    table_owned = _validate_portable_table_calls(
        descriptor.get("table_calls"),
        table_kernels,
        selector_partitions,
        expected_role="contribution",
        context=context,
    )
    finalizer_owned = _validate_portable_table_calls(
        descriptor.get("finalizer_calls"),
        table_kernels,
        selector_partitions,
        expected_role="finalizer",
        context=context,
    )
    if table_owned != finalizer_owned:
        raise RuntimeError(f"{context} contribution and finalizer ownership differ")
    if arena == "amplitude" and table_owned:
        raise RuntimeError(f"{context} tableizes the amplitude stage")
    _validate_portable_execution_order(
        descriptor,
        residual_chunks,
        selector_partitions,
        context=context,
    )


def _validate_portable_compiled_execution(
    execution: Mapping[str, object],
    *,
    context: str,
) -> None:
    _validate_portable_capability_owner(execution, context=context)
    compiled = execution.get("compiled")
    if not isinstance(compiled, Mapping):
        raise RuntimeError(f"{context} has no compiled DAG")
    stage_set = compiled.get("stage_evaluators")
    if not isinstance(stage_set, Mapping):
        raise RuntimeError(f"{context} has no compiled DAG stage evaluators")
    if stage_set.get("kind") != "generic-dag-stage-evaluator-artifacts":
        raise RuntimeError(f"{context} stage evaluator contract is invalid")
    _validate_portable_capability_owner(
        stage_set,
        context=f"{context} stage evaluators",
    )
    stages = _portable_sequence(
        stage_set.get("stages"),
        context=f"{context} stages",
    )
    stage_count = _portable_nonnegative_int(
        stage_set,
        "stage_count",
        context=f"{context} stage evaluator set",
    )
    if stage_count != len(stages) + 1:
        raise RuntimeError(f"{context} stage evaluator set is incomplete")
    for stage_index, stage in enumerate(stages):
        if not isinstance(stage, Mapping):
            raise RuntimeError(f"{context} stage {stage_index} is invalid")
        _validate_portable_arena_stage(
            stage,
            arena="current",
            context=f"{context} stage {stage_index}",
        )
    amplitude_stage = stage_set.get("amplitude_stage")
    if not isinstance(amplitude_stage, Mapping):
        raise RuntimeError(f"{context} amplitude stage is absent")
    _validate_portable_arena_stage(
        amplitude_stage,
        arena="amplitude",
        context=f"{context} amplitude stage",
    )


def validate_portable_execution_manifest(
    execution: Mapping[str, object],
    *,
    context: str,
) -> int:
    """Validate one root compiled execution and every nested selector execution.

    This is the shared release-facing contract used while preparing, staging,
    and auditing the installed self-test fixture. It deliberately validates the
    portable O2 SymJIT source together with the complete O3 Direct-Arena
    descriptor rather than treating either representation as sufficient alone.
    """

    evaluator_count = 0
    for evaluator in _walk_evaluator_manifests(execution):
        evaluator_count += 1
        _validate_portable_symjit_source(evaluator)
    if evaluator_count == 0:
        raise RuntimeError(f"{context} has no SymJIT evaluator manifests")

    compiled_executions = _walk_compiled_execution_manifests(execution)
    if not compiled_executions or compiled_executions[0] is not execution:
        raise RuntimeError(f"{context} has no root compiled DAG")
    for execution_index, compiled_execution in enumerate(compiled_executions):
        _validate_portable_compiled_execution(
            compiled_execution,
            context=f"{context} compiled execution {execution_index}",
        )
    return evaluator_count


def _validate_portable_evaluator_configuration(
    artifact: Path,
    manifest: Mapping[str, object],
) -> int:
    """Require architecture-neutral MIR before retargeting an evaluator state."""

    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("artifact payload inventory is invalid")
    evaluator_count = 0
    for payload in payloads:
        if (
            not isinstance(payload, dict)
            or payload.get("role") != "evaluator-manifest"
            or not str(payload.get("path", "")).endswith("/execution.json")
        ):
            continue
        relative = payload.get("path")
        if not isinstance(relative, str):
            raise RuntimeError("self-test evaluator manifest has no path")
        execution = json.loads((artifact / relative).read_text(encoding="utf-8"))
        if not isinstance(execution, dict):
            raise RuntimeError("self-test evaluator manifest must be an object")
        evaluator_count += validate_portable_execution_manifest(
            execution,
            context=f"portable self-test evaluator manifest {relative}",
        )
    if evaluator_count == 0:
        raise RuntimeError("self-test artifact has no SymJIT evaluator manifests")
    return PORTABLE_OPTIMIZATION_LEVEL


def _strip_symbolica_fallbacks(artifact: Path, manifest: dict[str, Any]) -> None:
    removed: set[str] = set()
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("artifact payload inventory is invalid")
    execution_payloads = [
        payload
        for payload in payloads
        if isinstance(payload, dict) and payload.get("role") == "evaluator-manifest"
    ]
    for payload in execution_payloads:
        relative = payload.get("path")
        if not isinstance(relative, str) or not relative.endswith("/execution.json"):
            continue
        path = artifact / relative
        execution = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(execution, dict):
            raise RuntimeError(f"invalid evaluator manifest: {relative}")
        for evaluator in _walk_evaluator_manifests(execution):
            state = evaluator.get("evaluator_state_path")
            if isinstance(state, str):
                removed.add((Path(relative).parent / state).as_posix())
            evaluator["evaluator_state_path"] = None
            evaluator["evaluator_state_runtime_capability"] = None
        path.write_bytes(_canonical_json(execution))
        payload["sha256"] = _sha256(path)
        payload["size_bytes"] = path.stat().st_size
    if not removed:
        raise RuntimeError("source artifact has no Symbolica fallback evaluator states")
    removed_loose: set[str] = set()
    for relative in sorted(removed):
        path = artifact / relative
        if path.is_file():
            path.unlink()
            removed_loose.add(relative)
    removed_packed = removed - removed_loose
    if removed_packed:
        _strip_packed_symbolica_fallbacks(artifact, manifest, removed_packed)
    manifest["payloads"] = [
        payload
        for payload in payloads
        if not (
            isinstance(payload, dict)
            and isinstance(payload.get("path"), str)
            and payload["path"] in removed_loose
        )
    ]


def _strip_packed_symbolica_fallbacks(
    artifact: Path,
    manifest: dict[str, Any],
    removed: set[str],
) -> None:
    """Remove exact fallback states from a packed evaluator container."""

    from pyamplicol.generation.evaluator_container import (
        PacbinMemberKind,
        PacbinMemberSource,
        PacbinReader,
        write_pacbin_atomic,
    )

    extensions = manifest.get("extensions")
    if not isinstance(extensions, dict):
        raise RuntimeError("artifact extensions are invalid")
    extension = extensions.get("evaluator_payload_container")
    if not isinstance(extension, dict):
        missing = ", ".join(sorted(removed))
        raise RuntimeError(f"missing Symbolica fallback state: {missing}")
    relative = extension.get("path")
    if not isinstance(relative, str):
        raise RuntimeError("evaluator payload container has no path")
    container_path = artifact / relative

    with PacbinReader.open(container_path, verify_payloads=True) as reader:
        by_path = {member.logical_path: member for member in reader.members}
        missing = sorted(removed - by_path.keys())
        if missing:
            raise RuntimeError(
                "missing Symbolica fallback state: " + ", ".join(missing)
            )
        invalid = sorted(
            path
            for path in removed
            if by_path[path].kind is not PacbinMemberKind.SYMBOLICA_EXACT_STATE
        )
        if invalid:
            raise RuntimeError(
                "Symbolica fallback state has an invalid packed kind: "
                + ", ".join(invalid)
            )
        retained = tuple(
            PacbinMemberSource(
                member.logical_path,
                member.kind,
                reader.open_member_stream(member.logical_path),
            )
            for member in reader.members
            if member.logical_path not in removed
        )
        if not retained:
            raise RuntimeError("portable self-test evaluator container is empty")
        index = write_pacbin_atomic(container_path, retained)

    extension["member_count"] = len(index.members)
    extension["unpacked_size_bytes"] = sum(member.length for member in index.members)
    extension["index_sha256"] = index.index_sha256

    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("artifact payload inventory is invalid")
    matches = [
        payload
        for payload in payloads
        if isinstance(payload, dict) and payload.get("path") == relative
    ]
    if len(matches) != 1:
        raise RuntimeError("artifact must declare one evaluator payload container")
    payload = matches[0]
    payload["sha256"] = _sha256(container_path)
    payload["size_bytes"] = container_path.stat().st_size


def _sanitize_configuration_paths(artifact: Path, manifest: dict[str, Any]) -> None:
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("artifact payload inventory is invalid")
    roles = {"configuration-requested", "configuration-effective"}
    found: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict) or payload.get("role") not in roles:
            continue
        role = str(payload["role"])
        relative = payload.get("path")
        if not isinstance(relative, str):
            raise RuntimeError(f"self-test {role} payload has no path")
        path = artifact / relative
        text = path.read_text(encoding="utf-8")
        updated = _sanitize_generation_output(text, role)
        path.write_text(updated, encoding="utf-8")
        payload["sha256"] = _sha256(path)
        payload["size_bytes"] = path.stat().st_size
        found.add(role)
    if found != roles:
        raise RuntimeError(
            "self-test artifact lacks requested/effective configurations"
        )


def _sanitize_generation_output(text: str, role: str) -> str:
    pattern = r'(?m)^output\s*=\s*"[^"]*"$'
    matches = tuple(re.finditer(pattern, text))
    if len(matches) > 1:
        raise RuntimeError(f"self-test {role} has multiple generation outputs")
    if not matches:
        return text
    return re.sub(pattern, 'output = "."', text, count=1)


def _retarget_portable_manifest(
    manifest: dict[str, Any],
    *,
    source_target: str,
) -> None:
    producer = manifest.get("producer")
    if not isinstance(producer, dict):
        raise RuntimeError("self-test artifact producer metadata is invalid")
    target = producer.get("target")
    if not isinstance(target, dict) or target.get("triple") != source_target:
        raise RuntimeError("self-test artifact target differs from the active runtime")
    target["triple"] = PORTABLE_TEMPLATE
    target["cpu_features"] = []

    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("self-test artifact payload inventory is invalid")
    evaluator_targets = 0
    for payload in payloads:
        if not isinstance(payload, dict):
            raise RuntimeError("self-test artifact payload entry is invalid")
        payload_target = payload.get("target")
        if payload_target is None:
            continue
        if (
            not isinstance(payload_target, dict)
            or payload_target.get("triple") != source_target
        ):
            raise RuntimeError("self-test evaluator target metadata is invalid")
        payload_target["triple"] = PORTABLE_TEMPLATE
        payload_target["cpu_features"] = []
        evaluator_targets += 1
    if evaluator_targets == 0:
        raise RuntimeError("self-test artifact has no target-tagged evaluator state")


def _complex_pairs(values: Sequence[object]) -> list[list[float]]:
    return [
        [float(complex(value).real), float(complex(value).imag)] for value in values
    ]


def _complex_sequences_close(
    left: Sequence[object],
    right: Sequence[object],
) -> bool:
    if len(left) != len(right):
        return False
    for left_value, right_value in zip(left, right, strict=True):
        actual = complex(left_value)
        expected = complex(right_value)
        if not (
            math.isclose(actual.real, expected.real, rel_tol=1.0e-12, abs_tol=1.0e-14)
            and math.isclose(
                actual.imag, expected.imag, rel_tol=1.0e-12, abs_tol=1.0e-14
            )
        ):
            return False
    return True


def prepare(source: Path, destination: Path | None = None) -> Path:
    source = source.expanduser().resolve(strict=True)
    if not source.is_dir() or not (source / "artifact.json").is_file():
        raise RuntimeError("self-test source must be a schema-v3 artifact directory")

    try:
        source_manifest = json.loads(
            (source / "artifact.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("self-test source manifest is invalid") from error
    if (
        not isinstance(source_manifest, dict)
        or source_manifest.get("schema_version") != 3
    ):
        raise RuntimeError("self-test source is not a schema-v3 artifact")
    _validate_source_compiled_model(source, source_manifest)

    from pyamplicol import Runtime

    native = importlib.import_module("pyamplicol._rusticol")
    target = str(native.target_info().triple)
    if target not in COMPATIBLE_TARGETS:
        raise RuntimeError(f"unsupported portable self-test target: {target}")
    runtime = Runtime.load(source, mute_warnings=True)
    momentum_loader = getattr(runtime._backend, "validation_momenta", None)
    if not callable(momentum_loader):
        raise RuntimeError("artifact runtime cannot expose its validation point")
    momenta = momentum_loader()
    if momenta is None:
        raise RuntimeError("artifact has no deterministic validation point")
    total = runtime.evaluate(momenta)
    resolved = runtime.evaluate_resolved(momenta)
    if not _complex_sequences_close(resolved.total(), total):
        raise RuntimeError("resolved self-test values do not reproduce evaluate()")

    output = (destination or (FIXTURE_ROOT / PORTABLE_TEMPLATE)).expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"self-test destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.staging")
    if temporary.exists():
        raise RuntimeError(f"stale self-test staging directory exists: {temporary}")
    try:
        artifact = temporary / "artifact"
        shutil.copytree(source, artifact, symlinks=False)
        manifest_path = artifact / "artifact.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 3:
            raise RuntimeError("self-test source is not a schema-v3 artifact")
        producer = manifest.get("producer")
        runtime_metadata = manifest.get("runtime")
        if not isinstance(producer, dict) or not isinstance(runtime_metadata, dict):
            raise RuntimeError(
                "self-test artifact producer/runtime metadata is invalid"
            )
        if producer.get("target", {}).get("triple") != target:
            raise RuntimeError(
                "self-test artifact target differs from the active runtime"
            )
        validate_portable_artifact_capabilities(
            manifest,
            context="portable self-test artifact",
        )
        portable_optimization_level = _validate_portable_evaluator_configuration(
            artifact,
            manifest,
        )
        _strip_symbolica_fallbacks(artifact, manifest)
        _sanitize_configuration_paths(artifact, manifest)
        _retarget_portable_manifest(manifest, source_target=target)
        version = _workspace_version()
        _normalize_compiled_model_version(
            artifact,
            manifest,
            version=version,
        )
        producer["version"] = version
        runtime_metadata["engine_version"] = version
        manifest["artifact_id"] = _artifact_id(manifest)
        manifest_path.write_bytes(_canonical_json(manifest))
        expected = {
            "schema_version": 1,
            "artifact_path": "artifact",
            "target": PORTABLE_TEMPLATE,
            "compatible_targets": list(COMPATIBLE_TARGETS),
            "source_generation_target": target,
            "serialization": {
                "kind": "symjit-application-mir-v3",
                "endianness": "little",
                "word_size_bits": 64,
                "load_behavior": "recompile-mir-for-loading-host",
                "source_optimization_level": portable_optimization_level,
            },
            "process_id": runtime.physics.process_id,
            "process": runtime.physics.process,
            "momenta": momenta,
            "resolved_shape": list(resolved.shape),
            "helicity_ids": list(resolved.helicity_ids),
            "color_ids": list(resolved.color_ids),
            "total": _complex_pairs(total),
        }
        (temporary / "expected.json").write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args(argv)
    output = prepare(args.source, args.destination)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
