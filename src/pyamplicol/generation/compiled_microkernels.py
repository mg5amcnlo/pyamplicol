# SPDX-License-Identifier: 0BSD
"""Lower complete compiled-DAG currents into O3 DirectTable islands.

The compiled stage remains the owner of scheduling and selector semantics.
This module merely replaces complete, structurally admitted current
destinations with prepared-kernel calls.  A destination is never split
between the residual DirectApplication and a table island.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .._internal.versions import (
    COMPILED_PLANE_DIRECT_APPLICATION_ABI,
    COMPILED_STAGE_PLAN_ABI,
    EAGER_DIRECT_TABLE_BINDING_ABI,
    EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
    SYMJIT_APPLICATION_ABI,
)
from ..models.base import Model
from ..models.prepared_catalog import (
    PreparedKernelCatalog,
    PreparedKernelInput,
    PreparedKernelSpec,
    VertexKernelKey,
    build_prepared_kernel_catalog,
)
from .dag_types import GenericDAG, InteractionNode
from .eager_lowering import (
    EagerStageTables,
    PreparedCatalogEagerKernelResolver,
    lower_eager_execution_tables,
)
from .eager_tables import (
    EAGER_OUTPUT_FACTOR_COUPLING_IMAG,
    EAGER_OUTPUT_FACTOR_COUPLING_REAL,
    EAGER_OUTPUT_FACTOR_NONE,
    MISSING_U32,
)
from .stage_types import GenericCompiledStageBlueprint, GenericStageOutputSlot

COMPILED_MICROKERNEL_MAX_IDENTITIES = 64
COMPILED_MICROKERNEL_MAX_INPUTS = 64
COMPILED_MICROKERNEL_MAX_OUTPUTS = 4
COMPILED_MICROKERNEL_MAX_SOURCE_BYTES = 64 * 1024
COMPILED_MICROKERNEL_MAX_TABLE_BYTES = 4 * 1024 * 1024
COMPILED_MICROKERNEL_EMPTY_EVALUATOR_KIND = "compiled-stage-empty-residual"
COMPILED_MICROKERNEL_MIN_ELIGIBLE_OCCURRENCES = 64
COMPILED_MICROKERNEL_MIN_COVERAGE_BASIS_POINTS = 5_000
COMPILED_MICROKERNEL_MAX_PROJECTED_SOURCE_BASIS_POINTS = 2_500

_U32 = struct.Struct("<I")
_OVERWRITE = 0

DescriptorBuilder = Callable[[bytes, int, int], bytes]
PlaneStorage = Literal[
    "current",
    "momentum",
    "model-parameter",
    "zero",
    "amplitude",
]


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _complex_binary64_expression(value: complex) -> str:
    """Return a deterministic exact-IR spelling of one finite binary64 pair."""

    real = repr(float(value.real))
    imag = repr(float(value.imag))
    return f"({real})+sqrt(-1)*({imag})"


def _checked_u32(value: int, context: str) -> int:
    if isinstance(value, bool) or not 0 <= int(value) < 1 << 32:
        raise ValueError(f"{context} must fit in u32")
    return int(value)


def _pack_u32_rows(
    rows: Sequence[Sequence[int]],
    *,
    width: int,
    context: str,
) -> bytes:
    if width < 1:
        raise ValueError(f"{context} row width must be positive")
    payload = bytearray(len(rows) * width * _U32.size)
    for row_index, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"{context} row {row_index} has width {len(row)}, expected {width}"
            )
        for column, value in enumerate(row):
            _U32.pack_into(
                payload,
                (row_index * width + column) * _U32.size,
                _checked_u32(value, f"{context}[{row_index}][{column}]"),
            )
    return bytes(payload)


@dataclass(frozen=True, slots=True)
class _BinaryTable:
    path: str
    size_bytes: int
    sha256: str
    count: int
    row_size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "count": self.count,
            "row_size": self.row_size,
        }


@dataclass(frozen=True, slots=True)
class _KernelSource:
    table_kernel_id: int
    canonical_signature: str
    source_application_path: str
    source_application_size_bytes: int
    source_application_sha256: str
    descriptor_path: str
    descriptor_size_bytes: int
    descriptor_sha256: str
    input_complex_count: int
    output_complex_count: int
    input_contracts: tuple[dict[str, object], ...]
    output_layout: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "table_kernel_id": self.table_kernel_id,
            "prepared_kernel_id": None,
            "role": "contribution",
            "canonical_signature": self.canonical_signature,
            "source_application": {
                "path": self.source_application_path,
                "size_bytes": self.source_application_size_bytes,
                "sha256": self.source_application_sha256,
            },
            "source_application_abi": SYMJIT_APPLICATION_ABI,
            "descriptor": {
                "path": self.descriptor_path,
                "size_bytes": self.descriptor_size_bytes,
                "sha256": self.descriptor_sha256,
            },
            "descriptor_abi": EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
            "binding_abi": EAGER_DIRECT_TABLE_BINDING_ABI,
            "input_complex_count": self.input_complex_count,
            "output_complex_count": self.output_complex_count,
            "scalar_input_count": 0,
            "optimization_level": 3,
            "input_contracts": list(self.input_contracts),
            "output_layout": list(self.output_layout),
        }


@dataclass(frozen=True, slots=True)
class CompiledMicrokernelStageLowering:
    """One original stage plus its residual-only evaluator blueprint."""

    original_stage: GenericCompiledStageBlueprint
    residual_stage: GenericCompiledStageBlueprint
    original_chunk_ranges: tuple[tuple[int, int], ...]
    residual_original_chunk_indices: tuple[int, ...]
    residual_original_output_indices: tuple[int, ...]
    owned_current_ids: tuple[int, ...]
    table_calls: tuple[dict[str, object], ...]
    execution_order: tuple[dict[str, object], ...]
    selector_partitions: tuple[dict[str, object], ...]
    plane_catalog: tuple[dict[str, object], ...]
    factor_catalog: tuple[dict[str, object], ...]
    semantic_row_bytes: int

    @property
    def has_islands(self) -> bool:
        return bool(self.owned_current_ids)


class _PlaneCatalog:
    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []
        self._ids: dict[tuple[str, int, str, int | None, bool], int] = {}
        self.zero = self.intern(
            "zero",
            0,
            "real",
            current_id=None,
            proven_real=True,
        )

    def intern(
        self,
        storage: PlaneStorage,
        component: int,
        part: Literal["real", "imag"],
        *,
        current_id: int | None,
        proven_real: bool,
    ) -> int:
        component = _checked_u32(component, "plane component")
        if current_id is not None:
            current_id = _checked_u32(current_id, "plane current id")
        owns_current = storage == "current"
        if owns_current != (current_id is not None):
            raise ValueError(
                f"compiled {storage} plane current ownership is inconsistent"
            )
        if proven_real and storage in {"current", "amplitude"}:
            raise ValueError(
                f"compiled complex {storage} plane cannot be marked proven real"
            )
        key = (storage, component, part, current_id, proven_real)
        existing = self._ids.get(key)
        if existing is not None:
            return existing
        plane_id = len(self.entries)
        self.entries.append(
            {
                "plane_id": plane_id,
                "storage": storage,
                "component": component,
                "part": part,
                "current_id": current_id,
                "proven_real": proven_real,
            }
        )
        self._ids[key] = plane_id
        return plane_id

    def complex_pair(
        self,
        storage: PlaneStorage,
        component: int,
        *,
        current_id: int | None = None,
        real_valued: bool = False,
    ) -> tuple[int, int]:
        return (
            self.intern(
                storage,
                component,
                "real",
                current_id=current_id,
                proven_real=real_valued,
            ),
            self.zero
            if real_valued
            else self.intern(
                storage,
                component,
                "imag",
                current_id=current_id,
                proven_real=False,
            ),
        )

    def model_parameter_pair(
        self,
        *,
        real_parameter_index: int,
        imag_parameter_index: int | None,
    ) -> tuple[int, int]:
        return (
            self.intern(
                "model-parameter",
                real_parameter_index,
                "real",
                current_id=None,
                proven_real=True,
            ),
            self.zero
            if imag_parameter_index is None
            else self.intern(
                "model-parameter",
                imag_parameter_index,
                "imag",
                current_id=None,
                proven_real=True,
            ),
        )


class _FactorCatalog:
    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []
        self._ids: dict[tuple[float, float, int | None, str], int] = {}

    def intern(
        self,
        base: complex,
        *,
        model_parameter_index: int | None = None,
        parameter_component: Literal["none", "real", "imag"] = "none",
    ) -> int:
        if model_parameter_index is None:
            parameter_component = "none"
        else:
            model_parameter_index = _checked_u32(
                model_parameter_index,
                "factor model parameter index",
            )
            if parameter_component == "none":
                raise ValueError("mutable factor has no parameter component")
        key = (
            float(base.real),
            float(base.imag),
            model_parameter_index,
            parameter_component,
        )
        existing = self._ids.get(key)
        if existing is not None:
            return existing
        factor_id = len(self.entries)
        self.entries.append(
            {
                "factor_id": factor_id,
                "base": [key[0], key[1]],
                "model_parameter_index": model_parameter_index,
                "parameter_component": parameter_component,
            }
        )
        self._ids[key] = factor_id
        return factor_id


@dataclass(frozen=True, slots=True)
class _EligibleCurrent:
    current_id: int
    dimension: int
    vertex_kernel_ids: tuple[int, ...]
    finalizer_kernel_id: int | None
    original_chunk_index: int
    helicity_selector_domain_ids: tuple[int, ...]
    color_selector_domain_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _CompositeFactor:
    base: complex
    model_parameter_index: int | None
    parameter_component: Literal["none", "real", "imag"]


@dataclass(frozen=True, slots=True)
class _CompositeCurrentRecord:
    current_id: int
    kernel_signature: str
    input_planes: tuple[int, ...]
    output_planes: tuple[int, ...]
    original_chunk_index: int
    helicity_selector_domain_ids: tuple[int, ...]
    color_selector_domain_ids: tuple[int, ...]
    dependency_current_ids: tuple[int, ...]
    interaction_ids: tuple[int, ...]


class CompiledMicrokernelSession:
    """State shared while streaming every stage of one compiled execution lane."""

    def __init__(
        self,
        *,
        dag: GenericDAG,
        model: Model,
        runtime_schema: Mapping[str, object],
        artifact_dir: Path,
        symbolica_settings: Any,
        descriptor_builder: DescriptorBuilder | None = None,
        catalog: PreparedKernelCatalog | None = None,
    ) -> None:
        if getattr(symbolica_settings, "backend", None) != "jit":
            raise ValueError("compiled microkernel islands require the JIT backend")
        if getattr(symbolica_settings, "jit_optimization_level", None) != 3:
            raise ValueError("compiled microkernel islands require SymJIT O3")
        self.dag = dag
        self.model = model
        self.runtime_schema = runtime_schema
        self.artifact_dir = artifact_dir
        self.settings = symbolica_settings
        self.catalog = catalog or build_prepared_kernel_catalog(model)
        self.resolver = PreparedCatalogEagerKernelResolver(
            dag,
            self.catalog.resolver_manifest(),
        )
        self.eager_tables = lower_eager_execution_tables(
            dag,
            model,
            runtime_schema,
            self.resolver,
        )
        self._descriptor_builder = descriptor_builder
        self._value_slots = {
            int(record["value_slot_id"]): record
            for record in _mapping_sequence(
                _mapping(runtime_schema.get("value_storage"), "value_storage").get(
                    "value_slots"
                ),
                "value_storage.value_slots",
            )
        }
        self._momentum_slots = {
            int(record["momentum_slot_id"]): record
            for record in _mapping_sequence(
                runtime_schema.get("momentum_slots"),
                "momentum_slots",
            )
        }
        self._model_parameters = {
            int(record["parameter_index"]): record
            for record in _mapping_sequence(
                runtime_schema.get("model_parameters", ()),
                "model_parameters",
            )
        }
        self._model_parameters_by_name = {
            str(record["name"]): record for record in self._model_parameters.values()
        }
        logical_model_parameters: dict[
            str,
            dict[Literal["real", "imag"], Mapping[str, object]],
        ] = {}
        for record in self._model_parameters.values():
            runtime_name = record.get("runtime_name")
            component = record.get("complex_component")
            if isinstance(runtime_name, str) and component in {"real", "imag"}:
                logical_model_parameters.setdefault(runtime_name, {})[component] = (
                    record
                )
        self._logical_model_parameters = logical_model_parameters
        self._stage_tables = {
            stage.stage_index: stage for stage in self.eager_tables.stages
        }
        self._specs = dict(self.catalog.by_id)
        self._vertex_bindings = {
            binding.key: binding for binding in self.catalog.vertex_bindings
        }
        self._table_kernel_ids: dict[str, int] = {}
        self._composite_specs: dict[str, PreparedKernelSpec] = {}
        self._kernel_sources: dict[int, _KernelSource] = {}
        self._total_source_bytes = 0
        self._total_semantic_row_bytes = 0
        (
            self._admitted_current_ids,
            self.profitability_diagnostics,
        ) = self._profitability_preflight()

    def lower_stage(
        self,
        stage: GenericCompiledStageBlueprint,
        *,
        chunk_size: int | None,
    ) -> CompiledMicrokernelStageLowering:
        """Select complete currents and emit their semantic DirectTable rows."""

        original_ranges = _output_chunk_ranges(stage, chunk_size=chunk_size)
        if str(stage.stage_kind).startswith("amplitude"):
            return self._residual_only(stage, original_ranges)
        eager_stage = self._stage_tables.get(stage.stage_index)
        if eager_stage is None:
            raise ValueError(
                f"compiled stage {stage.stage_index} has no eager lowering witness"
            )
        eligible = self._eligible_currents(stage, eager_stage, original_ranges)
        if not eligible:
            return self._residual_only(stage, original_ranges)
        plane_catalog = _PlaneCatalog()
        factors = _FactorCatalog()
        interaction_groups = _stage_interaction_groups(self.dag, stage)
        if len(interaction_groups) != len(eager_stage.invocations):
            raise ValueError("prepared invocation witness changed group cardinality")
        invocation_witnesses: dict[int, tuple[object, object]] = {}
        for invocation, interactions in zip(
            eager_stage.invocations,
            interaction_groups,
            strict=True,
        ):
            start = int(invocation.attachment_start)
            stop = start + int(invocation.attachment_count)
            attachments = eager_stage.attachments[start:stop]
            if len(attachments) != len(interactions):
                raise ValueError("prepared invocation attachment witness changed")
            for interaction, attachment in zip(
                interactions,
                attachments,
                strict=True,
            ):
                if interaction.id in invocation_witnesses:
                    raise ValueError("prepared invocation witnesses overlap")
                invocation_witnesses[interaction.id] = (invocation, attachment)

        interactions_by_current: dict[int, list[InteractionNode]] = defaultdict(list)
        for interaction_id in stage.interaction_ids:
            interaction = self.dag.interactions[interaction_id]
            interactions_by_current[interaction.result_id].append(interaction)
        finalizations = {int(row.current_id): row for row in eager_stage.finalizations}
        records: list[_CompositeCurrentRecord] = []
        for item in eligible:
            interactions = tuple(interactions_by_current[item.current_id])
            if len(interactions) != len(item.vertex_kernel_ids):
                raise ValueError(
                    "compiled complete-current interaction witness changed"
                )
            witnesses: list[tuple[object, object]] = []
            for interaction in interactions:
                try:
                    witness = invocation_witnesses[interaction.id]
                except KeyError as error:
                    raise ValueError(
                        f"interaction {interaction.id} lacks a prepared "
                        "invocation witness"
                    ) from error
                invocation, attachment = witness
                if (
                    int(attachment.result_current_id)  # type: ignore[attr-defined]
                    != item.current_id
                ):
                    raise ValueError("prepared attachment changed its result current")
                witnesses.append((invocation, attachment))
            record = self._composite_current_record(
                item,
                interactions,
                tuple(witnesses),
                finalizations[item.current_id],
                planes=plane_catalog,
            )
            if record is not None:
                records.append(record)
        if not records:
            return self._residual_only(stage, original_ranges)
        new_signatures = {record.kernel_signature for record in records}
        if (
            len(set(self._table_kernel_ids) | new_signatures)
            > COMPILED_MICROKERNEL_MAX_IDENTITIES
        ):
            return self._residual_only(stage, original_ranges)
        self._reserve_kernel_signatures(new_signatures)
        owned = {record.current_id for record in records}
        residual, residual_chunks, original_outputs = _residual_stage(
            stage,
            dag=self.dag,
            owned_current_ids=owned,
            original_chunk_ranges=original_ranges,
        )
        identity_factor = factors.intern(1 + 0j)
        table_calls, table_row_bytes = self._write_composite_groups(
            stage,
            records,
            identity_factor_id=identity_factor,
        )
        semantic_row_bytes = table_row_bytes
        if (
            self._total_semantic_row_bytes + semantic_row_bytes
            > COMPILED_MICROKERNEL_MAX_TABLE_BYTES
        ):
            raise ValueError(
                "compiled microkernel semantic tables exceed the 4 MiB bound"
            )
        self._total_semantic_row_bytes += semantic_row_bytes

        execution_order: list[dict[str, object]] = []
        residual_index_by_chunk = {
            original_chunk: residual_index
            for residual_index, original_chunk in enumerate(residual_chunks)
        }
        table_by_chunk: dict[int, list[int]] = defaultdict(list)
        for index, call in enumerate(table_calls):
            table_by_chunk[int(call["original_chunk_index"])].append(index)
        for original_chunk_index in range(len(original_ranges)):
            residual_index = residual_index_by_chunk.get(original_chunk_index)
            if residual_index is not None:
                execution_order.append(
                    {
                        "kind": "residual-leaf",
                        "index": residual_index,
                        "original_chunk_index": original_chunk_index,
                    }
                )
            execution_order.extend(
                {
                    "kind": "table-call",
                    "index": index,
                    "original_chunk_index": original_chunk_index,
                }
                for index in table_by_chunk.get(original_chunk_index, ())
            )
        selector_partitions = _selector_partitions(
            stage,
            original_ranges,
            execution_order,
        )
        return CompiledMicrokernelStageLowering(
            original_stage=stage,
            residual_stage=residual,
            original_chunk_ranges=original_ranges,
            residual_original_chunk_indices=residual_chunks,
            residual_original_output_indices=original_outputs,
            owned_current_ids=tuple(sorted(owned)),
            table_calls=table_calls,
            execution_order=tuple(execution_order),
            selector_partitions=selector_partitions,
            plane_catalog=tuple(plane_catalog.entries),
            factor_catalog=tuple(factors.entries),
            semantic_row_bytes=semantic_row_bytes,
        )

    def build_stage_plan(
        self,
        lowering: CompiledMicrokernelStageLowering,
        *,
        residual_evaluator: Mapping[str, object],
        residual_leaves: Sequence[Mapping[str, object]],
        residual_output_bindings: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        """Finish the authenticated plan after residual evaluator compilation."""

        global_kernel_ids = sorted(
            {int(call["table_kernel_id"]) for call in lowering.table_calls}
        )
        local_kernel_ids = {
            global_kernel_id: local_kernel_id
            for local_kernel_id, global_kernel_id in enumerate(global_kernel_ids)
        }
        kernels = [
            replace(
                self._kernel_sources[global_kernel_id],
                table_kernel_id=local_kernel_id,
            )
            for local_kernel_id, global_kernel_id in enumerate(global_kernel_ids)
        ]
        selector_partition_ids = {
            (
                tuple(
                    int(value)
                    for value in _sequence(
                        partition["helicity_selector_domain_ids"],
                        "helicity selector domains",
                    )
                ),
                tuple(
                    int(value)
                    for value in _sequence(
                        partition["color_selector_domain_ids"],
                        "color selector domains",
                    )
                ),
                int(original_chunk_index),
            ): int(partition["partition_id"])
            for partition in lowering.selector_partitions
            for original_chunk_index in _sequence(
                partition["original_chunk_indices"],
                "selector original chunks",
            )
        }

        def published_call(call: Mapping[str, object]) -> dict[str, object]:
            signature = (
                tuple(
                    int(value)
                    for value in _sequence(
                        call["helicity_selector_domain_ids"],
                        "call helicity domains",
                    )
                ),
                tuple(
                    int(value)
                    for value in _sequence(
                        call["color_selector_domain_ids"],
                        "call color domains",
                    )
                ),
                int(call["original_chunk_index"]),
            )
            try:
                partition_id = selector_partition_ids[signature]
            except KeyError as error:
                raise ValueError(
                    "compiled table call has no exact selector partition"
                ) from error
            return {
                "table_kernel_id": local_kernel_ids[int(call["table_kernel_id"])],
                "invocation_rows": call["invocation_rows"],
                "attachment_rows": call["attachment_rows"],
                "owned_current_ids": call["owned_current_ids"],
                "dependency_current_ids": call["dependency_current_ids"],
                "dependency_current_components": (
                    call["dependency_current_components"]
                ),
                "interaction_ids": call["interaction_ids"],
                "selector_partition_ids": [partition_id],
            }

        table_calls = [published_call(call) for call in lowering.table_calls]
        table_source_bytes = sum(
            kernel.source_application_size_bytes for kernel in kernels
        )
        descriptor_bytes = sum(kernel.descriptor_size_bytes for kernel in kernels)
        residual_records: list[dict[str, object]] = []
        if len(residual_leaves) != len(lowering.residual_original_chunk_indices):
            raise ValueError(
                "residual evaluator leaf count changed original chunk mapping"
            )
        for residual_leaf_index, (leaf, original_chunk_index) in enumerate(
            zip(
                residual_leaves,
                lowering.residual_original_chunk_indices,
                strict=True,
            )
        ):
            residual_records.append(
                {
                    **dict(leaf),
                    "residual_leaf_index": residual_leaf_index,
                    "original_chunk_index": original_chunk_index,
                }
            )
        outputs = []
        if len(residual_output_bindings) != len(
            lowering.residual_original_output_indices
        ):
            raise ValueError("residual output binding count changed")
        for binding, original_output_index in zip(
            residual_output_bindings,
            lowering.residual_original_output_indices,
            strict=True,
        ):
            outputs.append(
                {
                    **dict(binding),
                    "original_output_index": original_output_index,
                }
            )
        invocation_count = sum(
            int(_mapping(call["invocation_rows"], "invocation_rows")["count"])
            for call in table_calls
        )
        attachment_count = sum(
            int(_mapping(call["attachment_rows"], "attachment_rows")["count"])
            for call in table_calls
        )
        return {
            "schema_version": 2,
            "kind": "compiled-stage-plan",
            "plan_abi": COMPILED_STAGE_PLAN_ABI,
            "residual_application_abi": COMPILED_PLANE_DIRECT_APPLICATION_ABI,
            "table_source_application_abi": SYMJIT_APPLICATION_ABI,
            "direct_table_descriptor_abi": EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
            "direct_table_binding_abi": EAGER_DIRECT_TABLE_BINDING_ABI,
            "element_layout": "split-complex-component-major",
            "residual_evaluator": dict(residual_evaluator),
            "input_bindings": _residual_input_bindings(lowering.residual_stage),
            "output_bindings": outputs,
            "residual_leaves": residual_records,
            "scratch_current_component_count": 0,
            "plane_catalog": list(lowering.plane_catalog),
            "factor_catalog": list(lowering.factor_catalog),
            "table_kernels": [kernel.to_dict() for kernel in kernels],
            "table_calls": table_calls,
            "finalizer_calls": [],
            "execution_order": list(lowering.execution_order),
            "selector_partitions": list(lowering.selector_partitions),
            "diagnostics": {
                "island_count": len(lowering.owned_current_ids),
                "kernel_count": len(kernels),
                "invocation_count": invocation_count,
                "attachment_count": attachment_count,
                "table_source_bytes": table_source_bytes,
                "descriptor_bytes": descriptor_bytes,
                "semantic_row_bytes": lowering.semantic_row_bytes,
                "scratch_current_component_count": 0,
            },
        }

    def _residual_only(
        self,
        stage: GenericCompiledStageBlueprint,
        ranges: tuple[tuple[int, int], ...],
    ) -> CompiledMicrokernelStageLowering:
        return CompiledMicrokernelStageLowering(
            original_stage=stage,
            residual_stage=stage,
            original_chunk_ranges=ranges,
            residual_original_chunk_indices=tuple(range(len(ranges))),
            residual_original_output_indices=tuple(range(stage.output_length)),
            owned_current_ids=(),
            table_calls=(),
            execution_order=tuple(
                {
                    "kind": "residual-leaf",
                    "index": index,
                    "original_chunk_index": index,
                }
                for index in range(len(ranges))
            ),
            selector_partitions=_selector_partitions(
                stage,
                ranges,
                tuple(
                    {
                        "kind": "residual-leaf",
                        "index": index,
                        "original_chunk_index": index,
                    }
                    for index in range(len(ranges))
                ),
            ),
            plane_catalog=(),
            factor_catalog=(),
            semantic_row_bytes=0,
        )

    def _eligible_currents(
        self,
        stage: GenericCompiledStageBlueprint,
        eager_stage: EagerStageTables,
        ranges: tuple[tuple[int, int], ...],
    ) -> tuple[_EligibleCurrent, ...]:
        interactions_by_current: dict[int, list[InteractionNode]] = defaultdict(list)
        stage_interaction_ids = set(stage.interaction_ids)
        for interaction in self.dag.interactions:
            if interaction.id in stage_interaction_ids:
                interactions_by_current[interaction.result_id].append(interaction)
        slots_by_current: dict[int, list[GenericStageOutputSlot]] = defaultdict(list)
        for slot in stage.output_slots:
            slots_by_current[slot.current_id].append(slot)
        finalizers = {row.current_id: row for row in eager_stage.finalizations}
        candidates: list[_EligibleCurrent] = []
        for current_id in sorted(interactions_by_current):
            if current_id not in self._admitted_current_ids:
                continue
            current = self.dag.currents[current_id]
            interactions = interactions_by_current[current_id]
            slots = slots_by_current.get(current_id, ())
            finalization = finalizers.get(current_id)
            if not slots or finalization is None or current.dimension != 2:
                continue
            chunk_ids = {_slot_chunk_index(slot, ranges) for slot in slots}
            if None in chunk_ids or len(chunk_ids) != 1:
                continue
            signatures = {
                (
                    slot.selector_domain_ids,
                    slot.color_selector_domain_ids,
                )
                for slot in slots
            }
            if len(signatures) != 1:
                continue
            vertex_ids: list[int] = []
            structural = True
            for interaction in interactions:
                binding = self._vertex_binding(interaction)
                spec = self._specs[binding.kernel_id]
                if not self._eligible_vertex(
                    interaction,
                    spec,
                    binding,
                ):
                    structural = False
                    break
                vertex_ids.append(spec.kernel_id)
            if not structural or len(set(vertex_ids)) != 1:
                continue
            finalizer_id = (
                None
                if finalization.kernel_id == MISSING_U32
                else int(finalization.kernel_id)
            )
            if finalizer_id is not None:
                spec = self._specs.get(finalizer_id)
                if (
                    spec is None
                    or spec.contract_kind != "propagator"
                    or spec.output_dimension != current.dimension
                    or spec.input_arity > COMPILED_MICROKERNEL_MAX_INPUTS
                    or spec.output_dimension > COMPILED_MICROKERNEL_MAX_OUTPUTS
                    or any(
                        item.role not in {"current", "momentum", "model-parameter"}
                        for item in spec.inputs
                    )
                ):
                    continue
            helicity_domains, color_domains = next(iter(signatures))
            candidates.append(
                _EligibleCurrent(
                    current_id=current_id,
                    dimension=current.dimension,
                    vertex_kernel_ids=tuple(vertex_ids),
                    finalizer_kernel_id=finalizer_id,
                    original_chunk_index=next(iter(chunk_ids)),  # type: ignore[arg-type]
                    helicity_selector_domain_ids=helicity_domains,
                    color_selector_domain_ids=color_domains,
                )
            )
        return tuple(candidates)

    def _profitability_preflight(
        self,
    ) -> tuple[frozenset[int], dict[str, object]]:
        """Admit a process-independent island set before emitting payloads."""

        interactions_by_current: dict[int, list[InteractionNode]] = defaultdict(list)
        active_occurrences: Counter[int] = Counter()
        binding_by_interaction: dict[int, object] = {}
        for interaction in self.dag.interactions:
            binding = self._vertex_binding(interaction)
            binding_by_interaction[interaction.id] = binding
            if self.dag.currents[interaction.result_id].dimension == 2:
                active_occurrences[int(binding.kernel_id)] += 1  # type: ignore[attr-defined]
            interactions_by_current[interaction.result_id].append(interaction)
        repeated_kernel_ids = {
            kernel_id for kernel_id, count in active_occurrences.items() if count > 1
        }
        denominator = sum(
            active_occurrences[kernel_id] for kernel_id in repeated_kernel_ids
        )
        finalizers = {
            int(row.current_id): row
            for stage in self.eager_tables.stages
            for row in stage.finalizations
        }

        candidates: dict[
            int,
            tuple[tuple[int, ...], tuple[int, ...]],
        ] = {}
        for current_id, interactions in sorted(interactions_by_current.items()):
            current = self.dag.currents[current_id]
            finalizer = finalizers.get(current_id)
            if finalizer is None or current.dimension != 2:
                continue
            vertex_ids: list[int] = []
            if any(
                not self._eligible_vertex(
                    interaction,
                    self._specs[
                        int(
                            binding_by_interaction[interaction.id].kernel_id  # type: ignore[attr-defined]
                        )
                    ],
                    binding_by_interaction[interaction.id],
                )
                for interaction in interactions
            ):
                continue
            vertex_ids.extend(
                int(binding_by_interaction[item.id].kernel_id)  # type: ignore[attr-defined]
                for item in interactions
            )
            if len(set(vertex_ids)) != 1 or any(
                kernel_id not in repeated_kernel_ids for kernel_id in vertex_ids
            ):
                continue
            finalizer_id = (
                None if finalizer.kernel_id == MISSING_U32 else int(finalizer.kernel_id)
            )
            if finalizer_id is not None:
                spec = self._specs.get(finalizer_id)
                if (
                    spec is None
                    or spec.contract_kind != "propagator"
                    or spec.output_dimension != current.dimension
                    or spec.input_arity > COMPILED_MICROKERNEL_MAX_INPUTS
                    or spec.output_dimension > COMPILED_MICROKERNEL_MAX_OUTPUTS
                    or any(
                        item.role not in {"current", "momentum", "model-parameter"}
                        for item in spec.inputs
                    )
                ):
                    continue
            has_unpropagated = int(finalizer.unpropagated_value_slot_id) != MISSING_U32
            has_propagated = int(finalizer.propagated_value_slot_id) != MISSING_U32
            if not has_unpropagated and not has_propagated:
                continue
            candidates[current_id] = (
                tuple(vertex_ids),
                (() if finalizer_id is None or not has_propagated else (finalizer_id,)),
            )

        # Retain the prepared-spec census as a source-size proxy. Complete
        # currents compile as composite kernels later; identity finalizers no
        # longer have an independently emitted source.
        changed = True
        while changed and candidates:
            changed = False
            prepared_occurrences: Counter[int] = Counter()
            for vertex_ids, finalizer_ids in candidates.values():
                prepared_occurrences.update(vertex_ids)
                prepared_occurrences.update(finalizer_ids)
            rejected = {
                current_id
                for current_id, (vertex_ids, finalizer_ids) in candidates.items()
                if any(
                    prepared_occurrences[kernel_id] < 2
                    for kernel_id in (*vertex_ids, *finalizer_ids)
                )
            }
            if rejected:
                changed = True
                for current_id in rejected:
                    del candidates[current_id]

        numerator = sum(len(vertex_ids) for vertex_ids, _ in candidates.values())
        coverage_basis_points = (
            0 if denominator == 0 else numerator * 10_000 // denominator
        )
        selected_vertex_ids = {
            kernel_id
            for vertex_ids, _ in candidates.values()
            for kernel_id in vertex_ids
        }
        unique_source_bytes = sum(
            _projected_kernel_source_bytes(self._specs[kernel_id])
            for kernel_id in selected_vertex_ids
        )
        replaced_source_bytes = sum(
            _projected_kernel_source_bytes(self._specs[kernel_id])
            for vertex_ids, _ in candidates.values()
            for kernel_id in vertex_ids
        )
        projected_source_basis_points = (
            10_000
            if replaced_source_bytes == 0
            else unique_source_bytes * 10_000 // replaced_source_bytes
        )
        prepared_kernel_ids = {
            kernel_id
            for vertex_ids, finalizer_ids in candidates.values()
            for kernel_id in (*vertex_ids, *finalizer_ids)
        }
        hard_bounds_pass = len(
            prepared_kernel_ids
        ) <= COMPILED_MICROKERNEL_MAX_IDENTITIES and all(
            self._specs[kernel_id].input_arity <= COMPILED_MICROKERNEL_MAX_INPUTS
            and self._specs[kernel_id].output_dimension
            <= COMPILED_MICROKERNEL_MAX_OUTPUTS
            for kernel_id in prepared_kernel_ids
        )
        admitted = (
            numerator >= COMPILED_MICROKERNEL_MIN_ELIGIBLE_OCCURRENCES
            and coverage_basis_points >= COMPILED_MICROKERNEL_MIN_COVERAGE_BASIS_POINTS
            and projected_source_basis_points
            <= COMPILED_MICROKERNEL_MAX_PROJECTED_SOURCE_BASIS_POINTS
            and hard_bounds_pass
        )
        diagnostics = {
            "contract": ("materialized-active-repeated-prepared-kernel-occurrences-v1"),
            "active_occurrence_count": denominator,
            "eligible_occurrence_count": numerator,
            "coverage_basis_points": coverage_basis_points,
            "unique_projected_source_bytes": unique_source_bytes,
            "replaced_projected_source_bytes": replaced_source_bytes,
            "projected_source_basis_points": projected_source_basis_points,
            "kernel_identity_count": len(prepared_kernel_ids),
            "admitted_current_count": len(candidates) if admitted else 0,
            "admitted": admitted,
        }
        return (
            frozenset(candidates) if admitted else frozenset(),
            diagnostics,
        )

    def _eligible_vertex(
        self,
        interaction: InteractionNode,
        spec: PreparedKernelSpec,
        binding: object,
    ) -> bool:
        if (
            spec.contract_kind != "vertex"
            or spec.input_arity > COMPILED_MICROKERNEL_MAX_INPUTS
            or spec.output_dimension > COMPILED_MICROKERNEL_MAX_OUTPUTS
            or any(
                item.role
                not in {
                    "left-current",
                    "right-current",
                    "left-momentum",
                    "right-momentum",
                    "model-parameter",
                }
                for item in spec.inputs
            )
        ):
            return False
        result_state = binding.result_state  # type: ignore[attr-defined]
        result = self.dag.currents[interaction.result_id]
        if result_state.dimension != result.dimension:
            return False
        return result.dimension == 2 and result_state.basis == "weyl-chiral"

    def _vertex_binding(self, interaction: InteractionNode) -> object:
        left = self.dag.currents[interaction.left_id]
        right = self.dag.currents[interaction.right_id]
        result = self.dag.currents[interaction.result_id]
        key = VertexKernelKey(
            kind=interaction.vertex_kind,
            particles=interaction.vertex_particles,
            left_chirality=left.index.chirality,
            right_chirality=right.index.chirality,
            result_chirality=result.index.chirality,
            coupling=interaction.coupling,
        )
        try:
            return self._vertex_bindings[key]
        except KeyError as error:
            raise ValueError(
                f"prepared catalog has no vertex binding for interaction "
                f"{interaction.id}"
            ) from error

    def _reserve_kernel_signatures(self, signatures: set[str]) -> None:
        for signature in sorted(signatures):
            if signature in self._table_kernel_ids:
                continue
            if len(self._table_kernel_ids) >= COMPILED_MICROKERNEL_MAX_IDENTITIES:
                raise ValueError("compiled microkernel identity bound exceeded")
            self._table_kernel_ids[signature] = len(self._table_kernel_ids)

    def _contribution_input_planes(
        self,
        spec: PreparedKernelSpec,
        invocation: object,
        planes: _PlaneCatalog,
    ) -> tuple[int, ...]:
        current_slots = (
            int(invocation.left_value_slot_id),  # type: ignore[attr-defined]
            int(invocation.right_value_slot_id),  # type: ignore[attr-defined]
        )
        momentum_slots = (
            int(invocation.left_momentum_slot_id),  # type: ignore[attr-defined]
            int(invocation.right_momentum_slot_id),  # type: ignore[attr-defined]
        )
        result: list[int] = []
        for item in spec.inputs:
            role = item.role
            component = item.component
            pair: tuple[int, int]
            if role in {"left-current", "right-current"}:
                operand = 1 if role == "right-current" else 0
                slot = self._value_slots[current_slots[operand]]
                pair = planes.complex_pair(
                    "current",
                    int(slot["component_start"]) + component,
                    current_id=int(slot["current_id"]),
                )
            elif role in {"left-momentum", "right-momentum"}:
                operand = 1 if role == "right-momentum" else 0
                slot = self._momentum_slots[momentum_slots[operand]]
                pair = planes.complex_pair(
                    "momentum",
                    int(slot["component_start"]) + component,
                    real_valued=True,
                )
            elif role == "model-parameter":
                pair = planes.model_parameter_pair(
                    **self._runtime_model_parameter_projection(item)
                )
            else:
                raise ValueError(
                    f"prepared contribution input role {role!r} is not supported"
                )
            result.extend(pair)
        return tuple(result)

    def _runtime_model_parameter_projection(
        self,
        item: PreparedKernelInput,
    ) -> dict[str, int | None]:
        """Resolve one logical prepared input to raw runtime f64 components."""

        name = item.model_parameter_name
        if not name:
            raise ValueError("prepared model-parameter input has no logical name")
        logical = self._logical_model_parameters.get(name)
        if logical is not None:
            if set(logical) != {"real", "imag"}:
                raise ValueError(
                    f"runtime complex model parameter {name!r} is incomplete"
                )
            real_record = logical["real"]
            imag_record = logical["imag"]
            domains = {
                str(record.get("complex_domain", "complex"))
                for record in (real_record, imag_record)
            }
            if len(domains) != 1:
                raise ValueError(
                    f"runtime complex model parameter {name!r} changed domain"
                )
            domain = next(iter(domains))
            if domain not in {"real", "imaginary", "complex"}:
                raise ValueError(
                    f"runtime complex model parameter {name!r} has invalid domain"
                )
            return {
                "real_parameter_index": int(real_record["parameter_index"]),
                "imag_parameter_index": (
                    None if domain == "real" else int(imag_record["parameter_index"])
                ),
            }
        record = self._model_parameters_by_name.get(name)
        if record is None:
            raise ValueError(f"prepared model parameter {name!r} is not runtime-bound")
        if (
            record.get("runtime_name") is not None
            or record.get("complex_component") is not None
        ):
            raise ValueError(
                f"runtime model parameter {name!r} has an ambiguous projection"
            )
        return {
            "real_parameter_index": int(record["parameter_index"]),
            "imag_parameter_index": None,
        }

    def _real_kernel_parameter_indices(
        self,
        spec: PreparedKernelSpec,
    ) -> tuple[int, ...]:
        return tuple(
            index
            for index, item in enumerate(spec.inputs)
            if item.role
            in {
                "left-momentum",
                "right-momentum",
                "momentum",
                "coupling-real",
                "coupling-imag",
            }
            or (
                item.role == "model-parameter"
                and self._runtime_model_parameter_projection(item)[
                    "imag_parameter_index"
                ]
                is None
            )
        )

    def _attachment_factor_projection(
        self,
        attachment_factor: complex,
        invocation: object,
    ) -> _CompositeFactor:
        source = int(invocation.output_factor_source)  # type: ignore[attr-defined]
        if source == EAGER_OUTPUT_FACTOR_NONE:
            return _CompositeFactor(attachment_factor, None, "none")
        coupling = self.eager_tables.couplings[
            int(invocation.coupling_slot_id)  # type: ignore[attr-defined]
        ]
        if source == EAGER_OUTPUT_FACTOR_COUPLING_REAL:
            parameter_id = int(coupling.real_parameter_id)
            constant = float(coupling.constant_real)
            component: Literal["real", "imag"] = "real"
        elif source == EAGER_OUTPUT_FACTOR_COUPLING_IMAG:
            parameter_id = int(coupling.imag_parameter_id)
            constant = float(coupling.constant_imag)
            component = "imag"
        else:
            raise ValueError("prepared invocation has an unsupported factor source")
        if parameter_id == MISSING_U32:
            return _CompositeFactor(
                attachment_factor * constant,
                None,
                "none",
            )
        record = self._model_parameters.get(parameter_id)
        if record is None:
            raise ValueError("prepared coupling factor is not runtime-bound")
        complex_component = record.get("complex_component")
        if complex_component is not None and complex_component != component:
            raise ValueError(
                "prepared coupling factor changed its runtime complex projection"
            )
        coupling_component = record.get("component")
        expected_coupling_component = 0 if component == "real" else 1
        if (
            coupling_component is not None
            and int(coupling_component) != expected_coupling_component
        ):
            raise ValueError(
                "prepared coupling factor changed its runtime scalar projection"
            )
        return _CompositeFactor(
            attachment_factor,
            parameter_id,
            component,
        )

    def _composite_current_record(
        self,
        item: _EligibleCurrent,
        interactions: Sequence[InteractionNode],
        witnesses: Sequence[tuple[object, object]],
        finalization: object,
        *,
        planes: _PlaneCatalog,
    ) -> _CompositeCurrentRecord | None:
        """Build one complete-current source and its one semantic table row."""

        from symbolica import Expression

        if item.dimension != 2:
            raise ValueError(
                "compiled complete-current islands require two-component currents"
            )
        if len(interactions) != len(witnesses) or not interactions:
            raise ValueError("compiled complete-current witnesses are incomplete")
        witness_kernel_ids = tuple(
            int(invocation.kernel_id)  # type: ignore[attr-defined]
            for invocation, _attachment in witnesses
        )
        if witness_kernel_ids != item.vertex_kernel_ids:
            raise ValueError(
                "compiled complete-current prepared-kernel witness changed"
            )

        composite_inputs: list[PreparedKernelInput] = []
        input_planes: list[int] = []
        ordered_sums: list[Any] = []
        factor_contracts: list[dict[str, object]] = []
        for contribution_index, (
            (invocation, attachment),
            _interaction,
        ) in enumerate(zip(witnesses, interactions, strict=True)):
            prepared_kernel_id = int(invocation.kernel_id)  # type: ignore[attr-defined]
            spec = self._specs[prepared_kernel_id]
            if spec.output_dimension != item.dimension:
                raise ValueError("compiled contribution output dimension changed")
            expressions = [Expression.parse(value) for value in spec.exact_expressions]
            for input_index, prepared_input in enumerate(spec.inputs):
                symbol = (
                    "pyamplicol_compiled_microkernel::"
                    f"contribution_{contribution_index}_input_{input_index}"
                )
                replacement = Expression.parse(symbol)
                source = Expression.parse(prepared_input.symbol)
                expressions = [
                    expression.replace(source, replacement)
                    for expression in expressions
                ]
                composite_inputs.append(replace(prepared_input, symbol=symbol))
            input_planes.extend(
                self._contribution_input_planes(spec, invocation, planes)
            )

            factor = self._attachment_factor_projection(
                complex(
                    float(attachment.factor_real),  # type: ignore[attr-defined]
                    float(attachment.factor_imag),  # type: ignore[attr-defined]
                ),
                invocation,
            )
            if not (
                math.isfinite(factor.base.real) and math.isfinite(factor.base.imag)
            ):
                raise ValueError("compiled contribution factor is not finite")
            factor_expression = Expression.parse(
                _complex_binary64_expression(factor.base)
            )
            if factor.model_parameter_index is not None:
                factor_symbol = (
                    "pyamplicol_compiled_microkernel::"
                    f"contribution_{contribution_index}_factor"
                )
                factor_input = PreparedKernelInput(
                    role=(
                        "coupling-real"
                        if factor.parameter_component == "real"
                        else "coupling-imag"
                    ),
                    component=contribution_index,
                    symbol=factor_symbol,
                )
                composite_inputs.append(factor_input)
                factor_expression *= Expression.parse(factor_symbol)
                input_planes.extend(
                    planes.model_parameter_pair(
                        real_parameter_index=factor.model_parameter_index,
                        imag_parameter_index=None,
                    )
                )
            factor_contracts.append(
                {
                    "base": [factor.base.real, factor.base.imag],
                    "mutable": factor.model_parameter_index is not None,
                    "parameter_component": factor.parameter_component,
                }
            )
            scaled = [expression * factor_expression for expression in expressions]
            if not ordered_sums:
                ordered_sums = scaled
            else:
                ordered_sums = [
                    accumulated + contribution
                    for accumulated, contribution in zip(
                        ordered_sums,
                        scaled,
                        strict=True,
                    )
                ]

        finalizer_spec = (
            None
            if item.finalizer_kernel_id is None
            else self._specs[item.finalizer_kernel_id]
        )
        propagated: list[Any] | None = None
        if finalizer_spec is not None:
            propagated = [
                Expression.parse(value) for value in finalizer_spec.exact_expressions
            ]
            for input_index, prepared_input in enumerate(finalizer_spec.inputs):
                source = Expression.parse(prepared_input.symbol)
                if prepared_input.role == "current":
                    try:
                        replacement = ordered_sums[prepared_input.component]
                    except IndexError as error:
                        raise ValueError(
                            "compiled finalizer current component is out of range"
                        ) from error
                else:
                    symbol = (
                        "pyamplicol_compiled_microkernel::"
                        f"finalizer_input_{input_index}"
                    )
                    replacement = Expression.parse(symbol)
                    composite_inputs.append(replace(prepared_input, symbol=symbol))
                    input_planes.extend(
                        self._embedded_propagator_input_planes(
                            prepared_input,
                            finalization,
                            planes,
                        )
                    )
                propagated = [
                    expression.replace(source, replacement) for expression in propagated
                ]

        exact_outputs: list[Any] = []
        output_layout: list[str] = []
        output_planes: list[int] = []
        unpropagated_slot_id = int(  # type: ignore[attr-defined]
            finalization.unpropagated_value_slot_id
        )
        propagated_slot_id = int(  # type: ignore[attr-defined]
            finalization.propagated_value_slot_id
        )
        if unpropagated_slot_id != MISSING_U32:
            exact_outputs.extend(ordered_sums)
            output_layout.extend(
                f"unpropagated-current:{component}"
                for component in range(item.dimension)
            )
            output_planes.extend(
                self._current_output_planes(
                    unpropagated_slot_id,
                    item.current_id,
                    item.dimension,
                    planes,
                )
            )
        if propagated_slot_id != MISSING_U32:
            exact_outputs.extend(ordered_sums if propagated is None else propagated)
            output_layout.extend(
                f"propagated-current:{component}" for component in range(item.dimension)
            )
            output_planes.extend(
                self._current_output_planes(
                    propagated_slot_id,
                    item.current_id,
                    item.dimension,
                    planes,
                )
            )
        if not exact_outputs:
            raise ValueError("compiled complete current has no final destination")
        if (
            len(composite_inputs) > COMPILED_MICROKERNEL_MAX_INPUTS
            or len(exact_outputs) > COMPILED_MICROKERNEL_MAX_OUTPUTS
        ):
            return None

        payload = {
            "kind": "compiled-complete-current-direct-table-v1",
            "dimension": item.dimension,
            "ordered_vertex_signatures": [
                self._specs[
                    int(invocation.kernel_id)  # type: ignore[attr-defined]
                ].canonical_signature
                for invocation, _attachment in witnesses
            ],
            "ordered_factors": factor_contracts,
            "finalizer_signature": (
                None if finalizer_spec is None else finalizer_spec.canonical_signature
            ),
            "inputs": [value.to_dict() for value in composite_inputs],
            "exact_expressions": [
                str(value.to_canonical_string()) for value in exact_outputs
            ],
            "output_layout": output_layout,
        }
        signature = _sha256(_canonical_json(payload))
        spec = PreparedKernelSpec(
            kernel_id=0,
            contract_kind="propagator",
            canonical_signature=signature,
            exact_expressions=tuple(
                str(value.to_canonical_string()) for value in exact_outputs
            ),
            inputs=tuple(composite_inputs),
            output_layout=tuple(output_layout),
            proof_classes=("compiled-complete-current-ordered-contributions-v1",),
        )
        existing = self._composite_specs.get(signature)
        if existing is not None and existing != spec:
            raise ValueError("compiled complete-current signature collision")
        self._composite_specs[signature] = spec
        dependencies = tuple(
            sorted(
                {
                    current_id
                    for interaction in interactions
                    for current_id in (interaction.left_id, interaction.right_id)
                }
            )
        )
        return _CompositeCurrentRecord(
            current_id=item.current_id,
            kernel_signature=signature,
            input_planes=tuple(input_planes),
            output_planes=tuple(output_planes),
            original_chunk_index=item.original_chunk_index,
            helicity_selector_domain_ids=item.helicity_selector_domain_ids,
            color_selector_domain_ids=item.color_selector_domain_ids,
            dependency_current_ids=dependencies,
            interaction_ids=tuple(interaction.id for interaction in interactions),
        )

    def _embedded_propagator_input_planes(
        self,
        item: PreparedKernelInput,
        finalization: object,
        planes: _PlaneCatalog,
    ) -> tuple[int, int]:
        if item.role == "momentum":
            slot = self._momentum_slots[
                int(finalization.momentum_slot_id)  # type: ignore[attr-defined]
            ]
            return planes.complex_pair(
                "momentum",
                int(slot["component_start"]) + item.component,
                real_valued=True,
            )
        if item.role == "model-parameter":
            return planes.model_parameter_pair(
                **self._runtime_model_parameter_projection(item)
            )
        raise ValueError(
            f"compiled complete-current propagator input role "
            f"{item.role!r} is not supported"
        )

    def _current_output_planes(
        self,
        value_slot_id: int,
        current_id: int,
        dimension: int,
        planes: _PlaneCatalog,
    ) -> tuple[int, ...]:
        slot = self._value_slots[value_slot_id]
        if (
            int(slot["current_id"]) != current_id
            or int(slot["component_stop"]) - int(slot["component_start"]) != dimension
        ):
            raise ValueError(
                "compiled complete-current destination changed its current"
            )
        return tuple(
            plane
            for component in range(dimension)
            for plane in planes.complex_pair(
                "current",
                int(slot["component_start"]) + component,
                current_id=current_id,
            )
        )

    def _write_composite_groups(
        self,
        stage: GenericCompiledStageBlueprint,
        records: Sequence[_CompositeCurrentRecord],
        *,
        identity_factor_id: int,
    ) -> tuple[tuple[dict[str, object], ...], int]:
        grouped: dict[
            tuple[str, tuple[int, ...], tuple[int, ...], int],
            list[_CompositeCurrentRecord],
        ] = defaultdict(list)
        for record in records:
            grouped[
                (
                    record.kernel_signature,
                    record.helicity_selector_domain_ids,
                    record.color_selector_domain_ids,
                    record.original_chunk_index,
                )
            ].append(record)

        calls: list[dict[str, object]] = []
        total_bytes = 0
        for call_index, (group_key, rows) in enumerate(grouped.items()):
            kernel_signature, helicity, color, original_chunk_index = group_key
            spec = self._composite_kernel_spec(kernel_signature)
            table_kernel_id = self._table_kernel_id(kernel_signature)
            invocation_rows: list[tuple[int, ...]] = []
            attachment_rows: list[tuple[int, ...]] = []
            current_ids: list[int] = []
            dependency_ids: set[int] = set()
            interaction_ids: list[int] = []
            for row in rows:
                attachment_start = len(attachment_rows)
                attachment_rows.append(
                    (*row.output_planes, identity_factor_id, _OVERWRITE)
                )
                invocation_rows.append((*row.input_planes, attachment_start, 1))
                current_ids.append(row.current_id)
                dependency_ids.update(row.dependency_current_ids)
                interaction_ids.extend(row.interaction_ids)
            if current_ids != sorted(set(current_ids)):
                raise ValueError(
                    "compiled complete-current rows are not ordered by unique owner"
                )
            invocation_width = 2 * spec.input_arity + 2
            attachment_width = 2 * spec.output_dimension + 2
            invocation_payload = _pack_u32_rows(
                invocation_rows,
                width=invocation_width,
                context="compiled complete-current invocation",
            )
            attachment_payload = _pack_u32_rows(
                attachment_rows,
                width=attachment_width,
                context="compiled complete-current attachment",
            )
            prefix = (
                f"compiled-microkernels/stage-{stage.stage_index}/"
                f"table-call-{call_index}"
            )
            invocation_table = self._write_table(
                f"{prefix}-invocations.bin",
                invocation_payload,
                count=len(invocation_rows),
                row_size=invocation_width * _U32.size,
            )
            attachment_table = self._write_table(
                f"{prefix}-attachments.bin",
                attachment_payload,
                count=len(attachment_rows),
                row_size=attachment_width * _U32.size,
            )
            calls.append(
                {
                    "table_kernel_id": table_kernel_id,
                    "original_chunk_index": original_chunk_index,
                    "invocation_rows": invocation_table.to_dict(),
                    "attachment_rows": attachment_table.to_dict(),
                    "owned_current_ids": sorted(current_ids),
                    "dependency_current_ids": sorted(dependency_ids),
                    "dependency_current_components": (
                        self._current_components(dependency_ids)
                    ),
                    "interaction_ids": interaction_ids,
                    "helicity_selector_domain_ids": list(helicity),
                    "color_selector_domain_ids": list(color),
                }
            )
            total_bytes += len(invocation_payload) + len(attachment_payload)
        return tuple(calls), total_bytes

    def _current_components(self, current_ids: Sequence[int]) -> list[int]:
        components: set[int] = set()
        for slot in self._value_slots.values():
            if int(slot["current_id"]) not in current_ids:
                continue
            components.update(
                range(int(slot["component_start"]), int(slot["component_stop"]))
            )
        return sorted(components)

    def _table_kernel_id(
        self,
        signature: str,
    ) -> int:
        try:
            table_kernel_id = self._table_kernel_ids[signature]
        except KeyError as error:
            raise ValueError(
                f"unreserved compiled table kernel {signature!r}"
            ) from error
        if table_kernel_id not in self._kernel_sources:
            self._kernel_sources[table_kernel_id] = self._compile_kernel_source(
                table_kernel_id,
                signature,
            )
        return table_kernel_id

    def _composite_kernel_spec(
        self,
        signature: str,
    ) -> PreparedKernelSpec:
        if not isinstance(signature, str) or not signature:
            raise ValueError("compiled table source must be a composite current")
        try:
            return self._composite_specs[signature]
        except KeyError as error:
            raise ValueError(
                "composite table kernel source is not registered"
            ) from error

    def _compile_kernel_source(
        self,
        table_kernel_id: int,
        signature: str,
    ) -> _KernelSource:
        from symbolica import Expression

        from ..evaluators.symbolica_compile import _compile_symbolica_outputs
        from ..evaluators.symbolica_helpers import (
            _symbolica_evaluator_artifact_manifest,
        )

        spec = self._composite_kernel_spec(signature)
        outputs = tuple(Expression.parse(value) for value in spec.exact_expressions)
        parameters = [Expression.parse(item.symbol) for item in spec.inputs]
        real_parameters = self._real_kernel_parameter_indices(spec)
        settings = replace(
            self.settings,
            jit_direct_translation=False,
            jit_optimization_level=3,
            compiled_output_chunk_size=None,
            output_chunk_strategy="uniform",
            compiled_chunk_compile_workers=1,
        )
        adapter = _compile_symbolica_outputs(
            outputs,
            parameters,
            merge_evaluators_strategy=False,
            verbose_evaluator_build=False,
            real_params=real_parameters,
            symbolica_settings=settings,
            jit_compile=True,
            label=f"compiled_microkernel_{table_kernel_id:02d}",
        )
        manifest = _symbolica_evaluator_artifact_manifest(
            adapter,
            self.artifact_dir,
        )
        expected = {
            "kind": "symjit-application-evaluator",
            "application_abi": SYMJIT_APPLICATION_ABI,
            "optimization_level": 3,
            "input_len": spec.input_arity,
            "output_len": spec.output_dimension,
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise ValueError(
                    f"compiled table kernel {table_kernel_id} {field} changed"
                )
        raw_path = manifest.get("application_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("compiled table kernel has no source application")
        raw_source_path = self.artifact_dir / raw_path
        source = raw_source_path.read_bytes()
        if not source:
            raise ValueError(f"compiled table kernel {table_kernel_id} source is empty")
        if (
            self._total_source_bytes + len(source)
            > COMPILED_MICROKERNEL_MAX_SOURCE_BYTES
        ):
            raise ValueError(
                "compiled table kernel sources exceed the global 64 KiB bound"
            )
        self._total_source_bytes += len(source)
        # Reuse the compiler-emitted application in place. Copying it into the
        # semantic table directory would make recursive artifact publication
        # retain two identical machine-code payloads.
        source_relative = raw_path
        state_path = manifest.get("evaluator_state_path")
        if isinstance(state_path, str) and state_path:
            _remove_unreferenced_generated_payload(
                self.artifact_dir,
                state_path,
            )
        descriptor_builder = self._descriptor_builder
        if descriptor_builder is None:
            from .artifact_writer import _derive_eager_direct_descriptor

            descriptor_builder = _derive_eager_direct_descriptor
        descriptor = descriptor_builder(
            source,
            input_complex_count=spec.input_arity,
            output_complex_count=spec.output_dimension,
        )
        if not isinstance(descriptor, bytes) or not descriptor:
            raise ValueError("compiled table descriptor builder returned no bytes")
        descriptor_relative = (
            f"compiled-microkernels/kernels/kernel-{table_kernel_id:02d}-"
            f"{spec.canonical_signature[:16]}.direct-table.bin"
        )
        descriptor_path = self.artifact_dir / descriptor_relative
        descriptor_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor_path.write_bytes(descriptor)
        return _KernelSource(
            table_kernel_id=table_kernel_id,
            canonical_signature=spec.canonical_signature,
            source_application_path=source_relative,
            source_application_size_bytes=len(source),
            source_application_sha256=_sha256(source),
            descriptor_path=descriptor_relative,
            descriptor_size_bytes=len(descriptor),
            descriptor_sha256=_sha256(descriptor),
            input_complex_count=spec.input_arity,
            output_complex_count=spec.output_dimension,
            input_contracts=tuple(item.to_dict() for item in spec.inputs),
            output_layout=spec.output_layout,
        )

    def _write_table(
        self,
        relative: str,
        payload: bytes,
        *,
        count: int,
        row_size: int,
    ) -> _BinaryTable:
        if len(payload) != count * row_size:
            raise ValueError("compiled microkernel table size is inconsistent")
        path = self.artifact_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return _BinaryTable(
            path=relative,
            size_bytes=len(payload),
            sha256=_sha256(payload),
            count=count,
            row_size=row_size,
        )


def compiled_microkernel_session(
    *,
    dag: GenericDAG,
    model: Model,
    runtime_schema: Mapping[str, object],
    artifact_dir: Path,
    symbolica_settings: Any,
    enabled: bool,
    descriptor_builder: DescriptorBuilder | None = None,
) -> CompiledMicrokernelSession | None:
    """Construct the O3 session only for the explicitly selected lane."""

    if (
        not enabled
        or getattr(symbolica_settings, "backend", None) != "jit"
        or getattr(symbolica_settings, "jit_optimization_level", None) != 3
    ):
        return None
    return CompiledMicrokernelSession(
        dag=dag,
        model=model,
        runtime_schema=runtime_schema,
        artifact_dir=artifact_dir,
        symbolica_settings=symbolica_settings,
        descriptor_builder=descriptor_builder,
    )


def _remove_unreferenced_generated_payload(root: Path, relative: str) -> None:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("generated microkernel payload path is not artifact-relative")
    root_resolved = root.resolve()
    candidate = (root / path).resolve()
    if candidate.parent != root_resolved and root_resolved not in candidate.parents:
        raise ValueError("generated microkernel payload escapes the artifact root")
    if not candidate.is_file():
        raise ValueError("generated microkernel evaluator state is absent")
    candidate.unlink()


def empty_residual_evaluator() -> dict[str, object]:
    return {
        "kind": COMPILED_MICROKERNEL_EMPTY_EVALUATOR_KIND,
        "input_len": 0,
        "output_len": 0,
        "required_runtime_capabilities": [],
    }


def residual_only_stage_plan(
    stage: GenericCompiledStageBlueprint,
    *,
    evaluator: Mapping[str, object],
    leaves: Sequence[Mapping[str, object]],
    output_bindings: Sequence[Mapping[str, object]],
    residual_application_abi: str,
) -> dict[str, object]:
    """Wrap an existing full evaluator in the universal v2 stage plan."""

    if not leaves:
        raise ValueError("residual-only compiled stage has no evaluator leaves")
    if len(output_bindings) != stage.output_length:
        raise ValueError("residual-only compiled stage output binding count changed")
    residual_leaves: list[dict[str, object]] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for leaf_index, leaf in enumerate(leaves):
        output_start = int(leaf["output_start"])
        output_stop = int(leaf["output_stop"])
        if output_start != cursor or output_stop <= output_start:
            raise ValueError("residual-only evaluator leaf ranges are not contiguous")
        ranges.append((output_start, output_stop))
        residual_leaves.append(
            {
                **dict(leaf),
                "residual_leaf_index": leaf_index,
                "original_chunk_index": leaf_index,
            }
        )
        cursor = output_stop
    if cursor != stage.output_length:
        raise ValueError("residual-only evaluator leaves do not cover stage outputs")
    execution_order = tuple(
        {
            "kind": "residual-leaf",
            "index": leaf_index,
            "original_chunk_index": leaf_index,
        }
        for leaf_index in range(len(residual_leaves))
    )
    outputs = [
        {
            **dict(binding),
            "original_output_index": output_index,
        }
        for output_index, binding in enumerate(output_bindings)
    ]
    return {
        "schema_version": 2,
        "kind": "compiled-stage-plan",
        "plan_abi": COMPILED_STAGE_PLAN_ABI,
        "residual_application_abi": residual_application_abi,
        "table_source_application_abi": SYMJIT_APPLICATION_ABI,
        "direct_table_descriptor_abi": EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
        "direct_table_binding_abi": EAGER_DIRECT_TABLE_BINDING_ABI,
        "element_layout": "split-complex-component-major",
        "residual_evaluator": dict(evaluator),
        "input_bindings": _residual_input_bindings(stage),
        "output_bindings": outputs,
        "residual_leaves": residual_leaves,
        "scratch_current_component_count": 0,
        "plane_catalog": [],
        "factor_catalog": [],
        "table_kernels": [],
        "table_calls": [],
        "finalizer_calls": [],
        "execution_order": list(execution_order),
        "selector_partitions": list(
            _selector_partitions(stage, tuple(ranges), execution_order)
        ),
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


def _projected_kernel_source_bytes(spec: PreparedKernelSpec) -> int:
    """Deterministic pre-compile source-size proxy for profitability."""

    return len(
        _canonical_json(
            {
                "exact_expressions": list(spec.exact_expressions),
                "inputs": [item.to_dict() for item in spec.inputs],
                "output_layout": list(spec.output_layout),
            }
        )
    )


def _output_chunk_ranges(
    stage: GenericCompiledStageBlueprint,
    *,
    chunk_size: int | None,
) -> tuple[tuple[int, int], ...]:
    if stage.output_length < 1:
        raise ValueError("compiled stage cannot have zero original outputs")
    partitions = stage.selector_output_partitions or ((0, stage.output_length),)
    ranges: list[tuple[int, int]] = []
    expected = 0
    for start, stop in partitions:
        if start != expected or stop <= start or stop > stage.output_length:
            raise ValueError("compiled stage selector partitions are malformed")
        expected = stop
        if chunk_size is None:
            ranges.append((start, stop))
        else:
            if chunk_size < 1:
                raise ValueError("compiled stage chunk size must be positive")
            ranges.extend(
                (chunk_start, min(chunk_start + chunk_size, stop))
                for chunk_start in range(start, stop, chunk_size)
            )
    if expected != stage.output_length:
        raise ValueError("compiled stage selector partitions are incomplete")
    return tuple(ranges)


def _slot_chunk_index(
    slot: GenericStageOutputSlot,
    ranges: Sequence[tuple[int, int]],
) -> int | None:
    matches = [
        index
        for index, (start, stop) in enumerate(ranges)
        if start <= slot.output_start and slot.output_stop <= stop
    ]
    return matches[0] if len(matches) == 1 else None


def _residual_stage(
    stage: GenericCompiledStageBlueprint,
    *,
    dag: GenericDAG,
    owned_current_ids: set[int],
    original_chunk_ranges: Sequence[tuple[int, int]],
) -> tuple[GenericCompiledStageBlueprint, tuple[int, ...], tuple[int, ...]]:
    slots = tuple(
        sorted(
            stage.output_slots,
            key=lambda slot: (slot.output_start, slot.output_stop),
        )
    )
    expected_output = 0
    for slot in slots:
        if (
            slot.output_start != expected_output
            or slot.output_stop <= slot.output_start
            or slot.output_stop > stage.output_length
            or slot.component_stop - slot.component_start
            != slot.output_stop - slot.output_start
        ):
            raise ValueError(
                "compiled output slots must cover stage outputs exactly once"
            )
        expected_output = slot.output_stop
    if expected_output != stage.output_length:
        raise ValueError("compiled output slots do not cover every stage output")

    residual_outputs: list[object] = []
    residual_slots: list[GenericStageOutputSlot] = []
    original_output_indices: list[int] = []
    residual_chunks: list[int] = []
    partitions: list[tuple[int, int]] = []
    expected_chunk_start = 0
    for chunk_index, (chunk_start, chunk_stop) in enumerate(original_chunk_ranges):
        if (
            chunk_start != expected_chunk_start
            or chunk_stop <= chunk_start
            or chunk_stop > stage.output_length
        ):
            raise ValueError("compiled output chunks do not cover stage outputs")
        expected_chunk_start = chunk_stop
        partition_start = len(residual_outputs)
        for slot in slots:
            source_start = max(chunk_start, slot.output_start)
            source_stop = min(chunk_stop, slot.output_stop)
            if source_start >= source_stop:
                continue
            if slot.current_id in owned_current_ids:
                if source_start != slot.output_start or source_stop != slot.output_stop:
                    raise ValueError(
                        "compiled table-owned output slot crosses an original chunk"
                    )
                continue
            start = len(residual_outputs)
            residual_outputs.extend(stage.output_expressions[source_start:source_stop])
            original_output_indices.extend(range(source_start, source_stop))
            component_offset = source_start - slot.output_start
            residual_slots.append(
                replace(
                    slot,
                    component_start=slot.component_start + component_offset,
                    component_stop=(
                        slot.component_start
                        + component_offset
                        + source_stop
                        - source_start
                    ),
                    output_start=start,
                    output_stop=len(residual_outputs),
                )
            )
        if len(residual_outputs) > partition_start:
            residual_chunks.append(chunk_index)
            partitions.append((partition_start, len(residual_outputs)))
    if expected_chunk_start != stage.output_length:
        raise ValueError("compiled output chunks do not cover every stage output")
    residual_interactions = tuple(
        interaction_id
        for interaction_id in stage.interaction_ids
        if dag.interactions[interaction_id].result_id not in owned_current_ids
    )
    output_slot_ids = tuple(
        dict.fromkeys(slot.value_slot_id for slot in residual_slots)
    )
    residual = _prune_residual_stage_inputs(
        replace(
            stage,
            output_length=len(residual_outputs),
            output_slots=tuple(residual_slots),
            output_value_slot_ids=output_slot_ids,
            interaction_ids=residual_interactions,
            first_output_previews=tuple(
                expression.to_canonical_string()[:512]
                for expression in residual_outputs[:3]
            ),
            evaluation_groups_by_current=tuple(
                item
                for item in stage.evaluation_groups_by_current
                if item[0] not in owned_current_ids
            ),
            selector_output_partitions=tuple(partitions),
            output_expressions=tuple(residual_outputs),
        )
    )
    return (
        residual,
        tuple(residual_chunks),
        tuple(original_output_indices),
    )


def _prune_residual_stage_inputs(
    stage: GenericCompiledStageBlueprint,
) -> GenericCompiledStageBlueprint:
    """Retain only runtime parameters referenced by residual expressions.

    The ordinary chunk compiler performs this projection whenever an evaluator
    has multiple chunks.  Removing complete table-owned currents can leave one
    residual chunk, for which SymJIT intentionally keeps the complete declared
    parameter list.  Projecting the stage first preserves the original chunk's
    exact dependency/liveness contract without changing any expression or its
    contribution order.
    """

    if not stage.output_expressions:
        return replace(
            stage,
            input_value_slot_ids=(),
            input_components=(),
            parameter_count=0,
            value_parameter_count=0,
            momentum_parameter_count=0,
            model_parameter_count=0,
            real_valued_inputs=(),
            parameter_symbols=(),
        )
    if stage.parameter_count == 0:
        if stage.parameter_symbols or stage.input_components:
            raise ValueError(
                "compiled residual stage parameter bindings are inconsistent"
            )
        return stage
    if (
        len(stage.parameter_symbols) != stage.parameter_count
        or len(stage.input_components) != stage.parameter_count
    ):
        raise ValueError("compiled residual stage parameter bindings are incomplete")
    components_by_parameter: list[object | None] = [None] * stage.parameter_count
    for component in stage.input_components:
        parameter_index = int(component.parameter_index)
        if (
            parameter_index < 0
            or parameter_index >= stage.parameter_count
            or components_by_parameter[parameter_index] is not None
        ):
            raise ValueError("compiled residual stage parameter bindings are invalid")
        components_by_parameter[parameter_index] = component
    if any(component is None for component in components_by_parameter):
        raise ValueError("compiled residual stage parameter bindings are incomplete")

    used_symbols: set[object] = set()
    for expression in stage.output_expressions:
        getter = getattr(expression, "get_all_symbols", None)
        if not callable(getter):
            raise ValueError(
                "compiled residual expression lacks structural symbol discovery"
            )
        used_symbols.update(getter(False))
    for _function, _arguments, body in stage.symbolica_functions:
        getter = getattr(body, "get_all_symbols", None)
        if not callable(getter):
            raise ValueError(
                "compiled residual function body lacks structural symbol discovery"
            )
        used_symbols.update(getter(False))

    retained_indices = tuple(
        index
        for index, symbol in enumerate(stage.parameter_symbols)
        if symbol in used_symbols
    )
    old_real_inputs = set(stage.real_valued_inputs)
    retained_components = tuple(
        replace(
            components_by_parameter[old_index],
            parameter_index=new_index,
        )
        for new_index, old_index in enumerate(retained_indices)
    )
    value_components = tuple(
        component for component in retained_components if component.kind == "value"
    )
    momentum_components = tuple(
        component for component in retained_components if component.kind == "momentum"
    )
    model_parameter_components = tuple(
        component
        for component in retained_components
        if component.kind == "model_parameter"
    )
    if len(value_components) + len(momentum_components) + len(
        model_parameter_components
    ) != len(retained_components):
        raise ValueError("compiled residual stage has an unsupported input kind")
    return replace(
        stage,
        input_value_slot_ids=tuple(
            dict.fromkeys(component.source_id for component in value_components)
        ),
        input_components=retained_components,
        parameter_count=len(retained_indices),
        value_parameter_count=len(value_components),
        momentum_parameter_count=len(momentum_components),
        model_parameter_count=len(model_parameter_components),
        real_valued_inputs=tuple(
            new_index
            for new_index, old_index in enumerate(retained_indices)
            if old_index in old_real_inputs
        ),
        parameter_symbols=tuple(
            stage.parameter_symbols[index] for index in retained_indices
        ),
    )


def _stage_interaction_groups(
    dag: GenericDAG,
    stage: GenericCompiledStageBlueprint,
) -> tuple[tuple[InteractionNode, ...], ...]:
    groups: dict[tuple[str, int], list[InteractionNode]] = {}
    for interaction_id in stage.interaction_ids:
        interaction = dag.interactions[interaction_id]
        key = (
            ("group", int(interaction.evaluation_group_id))
            if interaction.evaluation_group_id is not None
            else ("interaction", interaction.id)
        )
        groups.setdefault(key, []).append(interaction)
    return tuple(tuple(group) for group in groups.values())


def _selector_partitions(
    stage: GenericCompiledStageBlueprint,
    ranges: Sequence[tuple[int, int]],
    execution_order: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    units_by_signature: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        list[int],
    ] = defaultdict(list)
    slots_by_chunk: dict[int, list[GenericStageOutputSlot]] = defaultdict(list)
    for slot in stage.output_slots:
        chunks = [
            index
            for index, (start, stop) in enumerate(ranges)
            if start < slot.output_stop and slot.output_start < stop
        ]
        if not chunks:
            raise ValueError("compiled selector output slot has no chunk")
        for chunk in chunks:
            slots_by_chunk[chunk].append(slot)
    for chunk_index in range(len(ranges)):
        signatures = {
            (slot.selector_domain_ids, slot.color_selector_domain_ids)
            for slot in slots_by_chunk[chunk_index]
        }
        if len(signatures) != 1:
            raise ValueError("compiled output chunk crosses selector partitions")
        signature = next(iter(signatures))
        if any(
            int(item["original_chunk_index"]) == chunk_index for item in execution_order
        ):
            units_by_signature[signature].append(chunk_index)
    return tuple(
        {
            "partition_id": partition_id,
            "helicity_selector_domain_ids": list(signature[0]),
            "color_selector_domain_ids": list(signature[1]),
            "original_chunk_indices": units,
        }
        for partition_id, (signature, units) in enumerate(
            sorted(units_by_signature.items())
        )
    )


def _residual_input_bindings(
    stage: GenericCompiledStageBlueprint,
) -> list[dict[str, object]]:
    return [
        {
            "parameter_index": component.parameter_index,
            "kind": component.kind,
            "source_id": component.source_id,
            "component": component.component,
            "global_component": component.global_component,
            "real_valued": component.real_valued,
        }
        for component in stage.input_components
    ]


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{context} must be a sequence")
    return value


def _mapping_sequence(value: object, context: str) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _mapping(item, f"{context}[{index}]")
        for index, item in enumerate(_sequence(value, context))
    )


__all__ = [
    "COMPILED_MICROKERNEL_EMPTY_EVALUATOR_KIND",
    "COMPILED_MICROKERNEL_MAX_IDENTITIES",
    "COMPILED_MICROKERNEL_MAX_INPUTS",
    "COMPILED_MICROKERNEL_MAX_OUTPUTS",
    "CompiledMicrokernelSession",
    "CompiledMicrokernelStageLowering",
    "compiled_microkernel_session",
    "empty_residual_evaluator",
]
