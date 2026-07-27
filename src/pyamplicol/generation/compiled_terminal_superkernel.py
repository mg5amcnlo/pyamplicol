# SPDX-License-Identifier: 0BSD
"""Pure symbolic composition for terminal compiled-DAG superkernel probes.

This module deliberately stops at an ordinary
``GenericCompiledStageBlueprint``.  It does not select a runtime path, write an
artifact, or lower a DirectApplication.  Callers must provide the complete
stage blueprint and an explicit selector/structural-zero proof before a
composition can be produced.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from .._internal.physics.parameters import ParamBuilder
from .stage_parameters import _expression_previews
from .stage_types import (
    GenericCompiledStageBlueprint,
    GenericStageCompilerBlueprint,
    GenericStageInputComponent,
)


class TerminalSuperkernelError(ValueError):
    """Raised when a terminal stage chain cannot be composed exactly."""


@dataclass(frozen=True, order=True)
class TerminalSemanticInput:
    """Canonical identity of one surviving stage-local input component."""

    global_component: int
    kind: str
    source_id: int
    component: int
    real_valued: bool


@dataclass(frozen=True)
class TerminalSuperkernelComposition:
    """One pure pair-only or full-tail symbolic composition."""

    kind: Literal["pair", "full-tail"]
    stage: GenericCompiledStageBlueprint
    semantic_inputs: tuple[TerminalSemanticInput, ...]
    elided_stage_indices: tuple[int, int]
    dependency_components: tuple[int, ...]


@dataclass(frozen=True)
class _OutputBinding:
    semantic_input: TerminalSemanticInput
    expression: Any


def compose_terminal_superkernels(
    blueprint: GenericStageCompilerBlueprint,
    *,
    execution_mode: str,
    backend: str,
    optimization_level: int,
    selector_structural_zero_proven: bool,
) -> tuple[TerminalSuperkernelComposition, TerminalSuperkernelComposition]:
    """Build pair-only and full-tail terminal compositions.

    The returned pair is ordered ``(pair, full_tail)``.  All local parameter
    symbols are first rebound to fresh symbols ordered by full semantic input
    identity.  Producer outputs are then substituted into each consumer with
    one simultaneous ``Expression.replace_multiple`` operation per output.
    """

    _validate_request(
        blueprint,
        execution_mode=execution_mode,
        backend=backend,
        optimization_level=optimization_level,
        selector_structural_zero_proven=selector_structural_zero_proven,
    )
    first, second = blueprint.stages[-2:]
    amplitude = blueprint.amplitude_stage
    _validate_exclusive_terminal_chain(blueprint, first, second, amplitude)

    pair = _compose(
        kind="pair",
        stages=(first, second),
        final_stage=second,
    )
    full_tail = _compose(
        kind="full-tail",
        stages=(first, second, amplitude),
        final_stage=amplitude,
    )
    return pair, full_tail


def _validate_request(
    blueprint: GenericStageCompilerBlueprint,
    *,
    execution_mode: str,
    backend: str,
    optimization_level: int,
    selector_structural_zero_proven: bool,
) -> None:
    blockers: list[str] = []
    if execution_mode != "compiled":
        blockers.append("execution mode is not compiled")
    if backend != "jit":
        blockers.append("backend is not JIT")
    if isinstance(optimization_level, bool) or optimization_level != 3:
        blockers.append("JIT optimization level is not O3")
    if not selector_structural_zero_proven:
        blockers.append("selector and structural-zero proof is incomplete")
    if blueprint.kind != "pyamplicol-generic-stage-compiler-blueprint":
        blockers.append("stage compiler blueprint kind is unsupported")
    if not blueprint.expression_ready or blueprint.blockers:
        blockers.append("stage compiler blueprint is not expression-ready")
    if blueprint.stage_count != len(blueprint.stages) + 1:
        blockers.append("stage compiler blueprint count is inconsistent")
    if len(blueprint.stages) < 2:
        blockers.append("fewer than two current stages are available")
    if blockers:
        raise TerminalSuperkernelError("; ".join(blockers))

    first, second = blueprint.stages[-2:]
    amplitude = blueprint.amplitude_stage
    # Earlier stages may already have released their symbolic expressions in
    # the streaming compiler.  They are relevant only as possible consumers
    # below; composition requires complete expression state for the terminal
    # pair and amplitude only.
    for stage in (first, second, amplitude):
        _validate_stage(stage)
    if first.stage_index + 1 != second.stage_index:
        raise TerminalSuperkernelError("terminal current stages are not adjacent")
    if first.stage_kind.startswith("amplitude") or second.stage_kind.startswith(
        "amplitude"
    ):
        raise TerminalSuperkernelError(
            "terminal current-stage candidate contains an amplitude stage"
        )
    if not amplitude.stage_kind.startswith("amplitude"):
        raise TerminalSuperkernelError("terminal amplitude stage kind is invalid")
    if amplitude.output_length > 512:
        raise TerminalSuperkernelError(
            "terminal amplitude output count exceeds the 512-output cap"
        )
    layouts = {stage.parameter_layout for stage in (first, second, amplitude)}
    if layouts != {"stage-local-value-momentum"}:
        raise TerminalSuperkernelError(
            "terminal stages do not use the stage-local parameter layout"
        )

    signatures = {
        _uniform_selector_signature(stage) for stage in (first, second, amplitude)
    }
    if len(signatures) != 1:
        raise TerminalSuperkernelError(
            "terminal stages have incompatible selector-domain signatures"
        )


def _validate_stage(stage: GenericCompiledStageBlueprint) -> None:
    label = stage.evaluator_label
    blockers: list[str] = []
    if not stage.expression_ready or stage.blockers:
        blockers.append("is not expression-ready")
    if stage.output_length < 1:
        blockers.append("has no outputs")
    if stage.output_length != len(stage.output_expressions):
        blockers.append("has inconsistent output expressions")
    if stage.parameter_count != len(stage.parameter_symbols):
        blockers.append("has inconsistent parameter symbols")
    if stage.parameter_count != len(stage.input_components):
        blockers.append("has inconsistent input components")
    if stage.fanout_chunk_size is not None:
        blockers.append("has already been prepared for output chunking")
    if stage.selector_output_partitions not in {
        (),
        ((0, stage.output_length),),
    }:
        blockers.append("has more than one selector output partition")

    indices = tuple(component.parameter_index for component in stage.input_components)
    if tuple(sorted(indices)) != tuple(range(stage.parameter_count)):
        blockers.append("does not have dense parameter indices")
    ordered_components = tuple(
        sorted(stage.input_components, key=lambda item: item.parameter_index)
    )
    real_indices = tuple(
        component.parameter_index
        for component in ordered_components
        if component.real_valued
    )
    if real_indices != stage.real_valued_inputs:
        blockers.append("has inconsistent real-valued parameter indices")

    kind_counts = {
        kind: sum(component.kind == kind for component in ordered_components)
        for kind in ("value", "momentum", "model_parameter")
    }
    if any(component.kind not in kind_counts for component in ordered_components):
        blockers.append("has an unsupported input kind")
    if kind_counts["value"] != stage.value_parameter_count:
        blockers.append("has an inconsistent value-parameter count")
    if kind_counts["momentum"] != stage.momentum_parameter_count:
        blockers.append("has an inconsistent momentum-parameter count")
    if kind_counts["model_parameter"] != stage.model_parameter_count:
        blockers.append("has an inconsistent model-parameter count")

    input_semantics = [_semantic_input(item) for item in ordered_components]
    if len(set(input_semantics)) != len(input_semantics):
        blockers.append("has duplicate semantic input identities")
    if _has_conflicting_aliases(input_semantics):
        blockers.append("has conflicting semantic input aliases")
    if _has_duplicate_objects(stage.parameter_symbols):
        blockers.append("has duplicate local parameter symbols")

    try:
        _output_bindings(stage, stage.output_expressions)
        _validate_symbolica_functions(stage)
    except TerminalSuperkernelError as error:
        blockers.append(str(error))
    if blockers:
        raise TerminalSuperkernelError(f"{label!r} " + "; ".join(blockers))


def _validate_exclusive_terminal_chain(
    blueprint: GenericStageCompilerBlueprint,
    first: GenericCompiledStageBlueprint,
    second: GenericCompiledStageBlueprint,
    amplitude: GenericCompiledStageBlueprint,
) -> None:
    first_outputs = _output_bindings(first, first.output_expressions)
    second_outputs = _output_bindings(second, second.output_expressions)
    first_keys = set(first_outputs)
    second_keys = set(second_outputs)

    consumers_by_component: dict[int, list[tuple[int, TerminalSemanticInput]]] = {}
    for stage in (*blueprint.stages, amplitude):
        for component in stage.input_components:
            semantic = _semantic_input(component)
            consumers_by_component.setdefault(semantic.global_component, []).append(
                (stage.stage_index, semantic)
            )

    _require_exact_consumer(
        producer=first,
        consumer=second,
        output_bindings=first_outputs,
    )
    _require_exact_consumer(
        producer=second,
        consumer=amplitude,
        output_bindings=second_outputs,
    )
    for global_component, expected_consumer in (
        *((component, second.stage_index) for component in first_keys),
        *((component, amplitude.stage_index) for component in second_keys),
    ):
        actual_consumers = consumers_by_component.get(global_component, [])
        if len(actual_consumers) != 1:
            raise TerminalSuperkernelError(
                "terminal producer output has an external or duplicate consumer: "
                f"global_component={global_component}, "
                f"consumers={actual_consumers!r}"
            )
        actual_stage, actual_semantic = actual_consumers[0]
        expected_binding = (
            first_outputs.get(global_component) or second_outputs[global_component]
        )
        if (
            actual_stage != expected_consumer
            or actual_semantic != expected_binding.semantic_input
        ):
            raise TerminalSuperkernelError(
                "terminal producer output consumer identity is inconsistent: "
                f"global_component={global_component}"
            )


def _require_exact_consumer(
    *,
    producer: GenericCompiledStageBlueprint,
    consumer: GenericCompiledStageBlueprint,
    output_bindings: dict[int, _OutputBinding],
) -> None:
    consumed: dict[int, TerminalSemanticInput] = {}
    for component in consumer.input_components:
        semantic = _semantic_input(component)
        if semantic.global_component not in output_bindings:
            continue
        if semantic.global_component in consumed:
            raise TerminalSuperkernelError(
                "terminal producer output is bound more than once by its consumer"
            )
        consumed[semantic.global_component] = semantic
    if set(consumed) != set(output_bindings):
        missing = tuple(sorted(set(output_bindings) - set(consumed)))
        raise TerminalSuperkernelError(
            f"stage {consumer.stage_index} does not consume every output of "
            f"stage {producer.stage_index}: missing={missing!r}"
        )
    for global_component, semantic in consumed.items():
        if semantic != output_bindings[global_component].semantic_input:
            raise TerminalSuperkernelError(
                "terminal producer-output slot/component identity does not "
                f"match its consumer: global_component={global_component}"
            )


def _compose(
    *,
    kind: Literal["pair", "full-tail"],
    stages: tuple[GenericCompiledStageBlueprint, ...],
    final_stage: GenericCompiledStageBlueprint,
) -> TerminalSuperkernelComposition:
    producer_outputs: dict[int, _OutputBinding] = {}
    external_components: list[GenericStageInputComponent] = []
    for stage in stages:
        for component in stage.input_components:
            semantic = _semantic_input(component)
            binding = producer_outputs.get(semantic.global_component)
            if binding is None:
                external_components.append(component)
            elif semantic != binding.semantic_input:
                raise TerminalSuperkernelError(
                    "terminal internal binding changed semantic identity: "
                    f"global_component={semantic.global_component}"
                )
        if stage is not final_stage:
            # The expressions are rebound below after the complete external
            # semantic union and its fresh symbols are known.
            producer_outputs.update(_output_bindings(stage, stage.output_expressions))

    semantic_inputs = _canonical_semantic_union(external_components)
    canonical_symbols = _fresh_parameter_symbols(
        kind,
        stages[0].stage_index,
        stages[-1].stage_index,
        len(semantic_inputs),
    )
    symbols_by_semantic = dict(zip(semantic_inputs, canonical_symbols, strict=True))

    rebound_outputs: dict[int, _OutputBinding] = {}
    final_expressions: tuple[Any, ...] = ()
    for stage in stages:
        expressions = _substitute_stage(
            stage,
            canonical_symbols=symbols_by_semantic,
            producer_outputs=rebound_outputs,
        )
        bindings = _output_bindings(stage, expressions)
        if stage is final_stage:
            final_expressions = expressions
        else:
            overlap = set(rebound_outputs) & set(bindings)
            if overlap:
                raise TerminalSuperkernelError(
                    "terminal producer stages have overlapping output components: "
                    f"{tuple(sorted(overlap))!r}"
                )
            rebound_outputs.update(bindings)

    if len(final_expressions) != final_stage.output_length:
        raise TerminalSuperkernelError(
            "terminal composition did not preserve final output order"
        )
    input_components = tuple(
        GenericStageInputComponent(
            kind=semantic.kind,
            source_id=semantic.source_id,
            component=semantic.component,
            global_component=semantic.global_component,
            parameter_index=index,
            real_valued=semantic.real_valued,
        )
        for index, semantic in enumerate(semantic_inputs)
    )
    real_valued_inputs = tuple(
        index for index, semantic in enumerate(semantic_inputs) if semantic.real_valued
    )
    stage = replace(
        final_stage,
        evaluator_label=(
            "generic_terminal_stage_pair_superkernel"
            if kind == "pair"
            else "generic_terminal_full_tail_superkernel"
        ),
        input_value_slot_ids=tuple(
            sorted(
                {
                    semantic.source_id
                    for semantic in semantic_inputs
                    if semantic.kind == "value"
                }
            )
        ),
        interaction_ids=tuple(
            dict.fromkeys(
                interaction_id
                for source_stage in stages
                for interaction_id in source_stage.interaction_ids
            )
        ),
        input_components=input_components,
        parameter_count=len(semantic_inputs),
        value_parameter_count=sum(
            semantic.kind == "value" for semantic in semantic_inputs
        ),
        momentum_parameter_count=sum(
            semantic.kind == "momentum" for semantic in semantic_inputs
        ),
        model_parameter_count=sum(
            semantic.kind == "model_parameter" for semantic in semantic_inputs
        ),
        real_valued_inputs=real_valued_inputs,
        expression_ready=True,
        blockers=(),
        first_output_previews=_expression_previews(final_expressions),
        fanout_chunk_size=None,
        fanout_evaluation_occurrences_before=None,
        fanout_evaluation_occurrences_after=None,
        parameter_symbols=canonical_symbols,
        output_expressions=final_expressions,
        symbolica_functions=_merged_symbolica_functions(stages),
    )
    _validate_stage(stage)

    first, second = stages[:2]
    first_components = tuple(sorted(_output_bindings(first, first.output_expressions)))
    dependency_components = first_components
    if kind == "full-tail":
        dependency_components += tuple(
            sorted(_output_bindings(second, second.output_expressions))
        )
    return TerminalSuperkernelComposition(
        kind=kind,
        stage=stage,
        semantic_inputs=semantic_inputs,
        elided_stage_indices=(first.stage_index, second.stage_index),
        dependency_components=dependency_components,
    )


def _substitute_stage(
    stage: GenericCompiledStageBlueprint,
    *,
    canonical_symbols: dict[TerminalSemanticInput, Any],
    producer_outputs: dict[int, _OutputBinding],
) -> tuple[Any, ...]:
    from symbolica import Replacement

    replacements: list[Any] = []
    for component in sorted(
        stage.input_components, key=lambda item: item.parameter_index
    ):
        semantic = _semantic_input(component)
        producer = producer_outputs.get(semantic.global_component)
        if producer is not None:
            if semantic != producer.semantic_input:
                raise TerminalSuperkernelError(
                    "terminal producer substitution has a semantic alias conflict"
                )
            replacement = producer.expression
        else:
            try:
                replacement = canonical_symbols[semantic]
            except KeyError as error:
                raise TerminalSuperkernelError(
                    "terminal external input is absent from the canonical union"
                ) from error
        replacements.append(
            Replacement(
                stage.parameter_symbols[component.parameter_index],
                replacement,
            )
        )

    rewritten: list[Any] = []
    for expression in stage.output_expressions:
        replace_multiple = getattr(expression, "replace_multiple", None)
        if not callable(replace_multiple):
            raise TerminalSuperkernelError(
                f"stage {stage.evaluator_label!r} output does not support "
                "simultaneous replacement"
            )
        rewritten.append(replace_multiple(replacements))
    result = tuple(rewritten)
    _reject_retained_local_symbols(stage, result)
    return result


def _reject_retained_local_symbols(
    stage: GenericCompiledStageBlueprint,
    expressions: Sequence[Any],
) -> None:
    local_symbols = set(stage.parameter_symbols)
    for expression in expressions:
        getter = getattr(expression, "get_all_symbols", None)
        if not callable(getter):
            raise TerminalSuperkernelError(
                "composed expression does not expose its symbol closure"
            )
        if local_symbols & set(getter(True)):
            raise TerminalSuperkernelError(
                f"stage {stage.evaluator_label!r} retained a local parameter "
                "after canonical rebinding"
            )


def _canonical_semantic_union(
    components: Iterable[GenericStageInputComponent],
) -> tuple[TerminalSemanticInput, ...]:
    semantics = tuple(_semantic_input(component) for component in components)
    if _has_conflicting_aliases(semantics):
        raise TerminalSuperkernelError(
            "terminal external inputs have conflicting semantic aliases"
        )
    return tuple(sorted(set(semantics)))


def _semantic_input(
    component: GenericStageInputComponent,
) -> TerminalSemanticInput:
    return TerminalSemanticInput(
        global_component=int(component.global_component),
        kind=str(component.kind),
        source_id=int(component.source_id),
        component=int(component.component),
        real_valued=bool(component.real_valued),
    )


def _has_conflicting_aliases(
    semantics: Iterable[TerminalSemanticInput],
) -> bool:
    by_global: dict[int, TerminalSemanticInput] = {}
    for semantic in semantics:
        previous = by_global.setdefault(semantic.global_component, semantic)
        if previous != semantic:
            return True
    return False


def _output_bindings(
    stage: GenericCompiledStageBlueprint,
    expressions: Sequence[Any],
) -> dict[int, _OutputBinding]:
    bindings: dict[int, _OutputBinding] = {}
    covered_outputs: set[int] = set()
    for slot in stage.output_slots:
        output_length = slot.output_stop - slot.output_start
        component_length = slot.component_stop - slot.component_start
        if (
            output_length <= 0
            or output_length != component_length
            or slot.output_start < 0
            or slot.output_stop > len(expressions)
        ):
            raise TerminalSuperkernelError(
                "stage output slot has an invalid component/output span"
            )
        for offset in range(output_length):
            output_index = slot.output_start + offset
            global_component = slot.component_start + offset
            if output_index in covered_outputs:
                raise TerminalSuperkernelError(
                    "stage output slots overlap in output order"
                )
            if global_component in bindings:
                raise TerminalSuperkernelError(
                    "stage output slots alias a global component"
                )
            covered_outputs.add(output_index)
            semantic = TerminalSemanticInput(
                global_component=global_component,
                kind="value",
                source_id=int(slot.value_slot_id),
                component=offset,
                real_valued=False,
            )
            bindings[global_component] = _OutputBinding(
                semantic_input=semantic,
                expression=expressions[output_index],
            )
    if covered_outputs != set(range(len(expressions))):
        raise TerminalSuperkernelError(
            "stage output slots do not cover exact output order"
        )
    return bindings


def _uniform_selector_signature(
    stage: GenericCompiledStageBlueprint,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    signatures = {
        (slot.selector_domain_ids, slot.color_selector_domain_ids)
        for slot in stage.output_slots
    }
    if len(signatures) != 1:
        raise TerminalSuperkernelError(
            f"stage {stage.evaluator_label!r} has non-uniform selector domains"
        )
    return next(iter(signatures))


def _fresh_parameter_symbols(
    kind: str,
    first_stage_index: int,
    last_stage_index: int,
    count: int,
) -> tuple[Any, ...]:
    if count < 1:
        raise TerminalSuperkernelError(
            "terminal composition has no surviving external inputs"
        )
    builder = ParamBuilder()
    return builder.add_parameter_list(
        (
            "artifact_schema_v3_terminal_superkernel",
            kind,
            f"{first_stage_index}_{last_stage_index}",
        ),
        count,
        role="generic_terminal_superkernel_input",
    )


def _merged_symbolica_functions(
    stages: Sequence[GenericCompiledStageBlueprint],
) -> tuple[tuple[Any, tuple[Any, ...], Any], ...]:
    merged: dict[
        tuple[str, tuple[str, ...]],
        tuple[Any, tuple[Any, ...], Any],
    ] = {}
    body_by_key: dict[tuple[str, tuple[str, ...]], str] = {}
    for stage in stages:
        for function, arguments, body in stage.symbolica_functions:
            key = (
                _canonical_text(function),
                tuple(_canonical_text(argument) for argument in arguments),
            )
            body_text = _canonical_text(body)
            previous = body_by_key.setdefault(key, body_text)
            if previous != body_text:
                raise TerminalSuperkernelError(
                    "terminal stages define one symbolic function incompatibly"
                )
            merged.setdefault(key, (function, tuple(arguments), body))
    return tuple(merged[key] for key in sorted(merged))


def _validate_symbolica_functions(
    stage: GenericCompiledStageBlueprint,
) -> None:
    local_symbols = set(stage.parameter_symbols)
    for _function, _arguments, body in stage.symbolica_functions:
        getter = getattr(body, "get_all_symbols", None)
        if not callable(getter):
            raise TerminalSuperkernelError(
                "symbolic function body does not expose its symbol closure"
            )
        if local_symbols & set(getter(True)):
            raise TerminalSuperkernelError(
                "symbolic function body captures a stage-local parameter"
            )


def _canonical_text(value: Any) -> str:
    canonical = getattr(value, "to_canonical_string", None)
    return str(canonical()) if callable(canonical) else repr(value)


def _has_duplicate_objects(values: Sequence[Any]) -> bool:
    for index, value in enumerate(values):
        for previous in values[:index]:
            try:
                if bool(value == previous):
                    return True
            except Exception as error:
                raise TerminalSuperkernelError(
                    "could not compare local parameter symbols"
                ) from error
    return False


__all__ = [
    "TerminalSemanticInput",
    "TerminalSuperkernelComposition",
    "TerminalSuperkernelError",
    "compose_terminal_superkernels",
]
