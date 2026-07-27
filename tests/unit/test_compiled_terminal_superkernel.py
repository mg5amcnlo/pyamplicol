# SPDX-License-Identifier: 0BSD
"""Exact contracts for pure terminal compiled-stage composition."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pyamplicol._internal.physics.parameters import ParamBuilder
from pyamplicol.generation.compiled_terminal_superkernel import (
    TerminalSuperkernelError,
    compose_terminal_superkernels,
)
from pyamplicol.generation.stage_types import (
    GenericCompiledStageBlueprint,
    GenericStageCompilerBlueprint,
    GenericStageInputComponent,
    GenericStageOutputSlot,
)

_SELECTOR_DOMAINS = (7,)
_COLOR_DOMAINS = (0,)


def _local_symbols(count: int) -> tuple[object, ...]:
    builder = ParamBuilder()
    return builder.add_parameter_list(
        ("terminal_superkernel_test", "local"),
        count,
        role="test_local_parameter",
    )


def _input(
    *,
    source_id: int,
    component: int,
    global_component: int,
    parameter_index: int,
) -> GenericStageInputComponent:
    return GenericStageInputComponent(
        kind="value",
        source_id=source_id,
        component=component,
        global_component=global_component,
        parameter_index=parameter_index,
        real_valued=False,
    )


def _real_input(
    *,
    kind: str,
    source_id: int,
    global_component: int,
    parameter_index: int,
) -> GenericStageInputComponent:
    return GenericStageInputComponent(
        kind=kind,
        source_id=source_id,
        component=0,
        global_component=global_component,
        parameter_index=parameter_index,
        real_valued=True,
    )


def _slot(
    *,
    value_slot_id: int,
    current_id: int,
    component_start: int,
    component_stop: int,
    output_start: int,
    output_stop: int,
    amplitude: bool = False,
    selector_domains: tuple[int, ...] = _SELECTOR_DOMAINS,
) -> GenericStageOutputSlot:
    return GenericStageOutputSlot(
        value_slot_id=-1 if amplitude else value_slot_id,
        current_id=-1 if amplitude else current_id,
        variant="amplitude-root" if amplitude else "propagated",
        component_start=component_start,
        component_stop=component_stop,
        output_start=output_start,
        output_stop=output_stop,
        selector_domain_ids=selector_domains,
        color_selector_domain_ids=_COLOR_DOMAINS,
    )


def _stage(
    *,
    stage_index: int,
    label: str,
    inputs: tuple[GenericStageInputComponent, ...],
    outputs: tuple[object, ...],
    slots: tuple[GenericStageOutputSlot, ...],
    input_value_slot_ids: tuple[int, ...],
    output_value_slot_ids: tuple[int, ...],
    amplitude: bool = False,
) -> GenericCompiledStageBlueprint:
    symbols = _local_symbols(len(inputs))
    return GenericCompiledStageBlueprint(
        stage_index=stage_index,
        stage_kind="amplitude-roots" if amplitude else "current-combine",
        subset_size=None if amplitude else stage_index + 1,
        evaluator_label=label,
        parameter_layout="stage-local-value-momentum",
        output_length=len(outputs),
        output_slots=slots,
        input_value_slot_ids=input_value_slot_ids,
        output_value_slot_ids=output_value_slot_ids,
        interaction_ids=() if amplitude else (stage_index,),
        input_components=inputs,
        parameter_count=len(inputs),
        value_parameter_count=len(inputs),
        momentum_parameter_count=0,
        model_parameter_count=0,
        real_valued_inputs=(),
        expression_ready=True,
        blockers=(),
        first_output_previews=tuple(str(output) for output in outputs[:4]),
        parameter_symbols=symbols,
        output_expressions=outputs,
    )


def _terminal_blueprint() -> GenericStageCompilerBlueprint:
    first_inputs = (
        _input(source_id=0, component=0, global_component=0, parameter_index=0),
        _input(source_id=1, component=0, global_component=1, parameter_index=1),
    )
    first_symbols = _local_symbols(2)
    first = _stage(
        stage_index=6,
        label="terminal-first",
        inputs=first_inputs,
        outputs=(
            first_symbols[0] + first_symbols[1],
            first_symbols[0] * first_symbols[1],
        ),
        slots=(
            _slot(
                value_slot_id=10,
                current_id=10,
                component_start=10,
                component_stop=12,
                output_start=0,
                output_stop=2,
            ),
        ),
        input_value_slot_ids=(0, 1),
        output_value_slot_ids=(10,),
    )
    # _stage constructs the same local symbol head intentionally.  Replace the
    # expressions' symbols with the exact symbols retained by the blueprint.
    first = replace(
        first,
        output_expressions=(
            first.parameter_symbols[0] + first.parameter_symbols[1],
            first.parameter_symbols[0] * first.parameter_symbols[1],
        ),
    )

    second_inputs = (
        _input(source_id=10, component=0, global_component=10, parameter_index=0),
        _input(source_id=10, component=1, global_component=11, parameter_index=1),
        _input(source_id=2, component=0, global_component=2, parameter_index=2),
    )
    second = _stage(
        stage_index=7,
        label="terminal-second",
        inputs=second_inputs,
        outputs=(first_symbols[0], first_symbols[1]),
        slots=(
            _slot(
                value_slot_id=20,
                current_id=20,
                component_start=20,
                component_stop=22,
                output_start=0,
                output_stop=2,
            ),
        ),
        input_value_slot_ids=(2, 10),
        output_value_slot_ids=(20,),
    )
    second = replace(
        second,
        output_expressions=(
            second.parameter_symbols[0] * second.parameter_symbols[2]
            + second.parameter_symbols[1],
            second.parameter_symbols[0] - second.parameter_symbols[1],
        ),
    )

    amplitude_inputs = (
        _input(source_id=20, component=0, global_component=20, parameter_index=0),
        _input(source_id=20, component=1, global_component=21, parameter_index=1),
        _input(source_id=3, component=0, global_component=3, parameter_index=2),
    )
    amplitude = _stage(
        stage_index=0,
        label="terminal-amplitude",
        inputs=amplitude_inputs,
        outputs=(first_symbols[0], first_symbols[1]),
        slots=(
            _slot(
                value_slot_id=-1,
                current_id=-1,
                component_start=0,
                component_stop=1,
                output_start=0,
                output_stop=1,
                amplitude=True,
            ),
            _slot(
                value_slot_id=-1,
                current_id=-1,
                component_start=1,
                component_stop=2,
                output_start=1,
                output_stop=2,
                amplitude=True,
            ),
        ),
        input_value_slot_ids=(3, 20),
        output_value_slot_ids=(),
        amplitude=True,
    )
    amplitude = replace(
        amplitude,
        output_expressions=(
            amplitude.parameter_symbols[0] + amplitude.parameter_symbols[2],
            amplitude.parameter_symbols[1] * amplitude.parameter_symbols[2],
        ),
    )
    return GenericStageCompilerBlueprint(
        kind="pyamplicol-generic-stage-compiler-blueprint",
        runtime_available=True,
        parameter_count=0,
        value_parameter_count=0,
        momentum_parameter_count=0,
        model_parameter_count=0,
        real_valued_inputs=(),
        stage_count=3,
        stages=(first, second),
        amplitude_stage=amplitude,
        expression_ready=True,
        blockers=(),
    )


def _compose(
    blueprint: GenericStageCompilerBlueprint | None = None,
):
    return compose_terminal_superkernels(
        _terminal_blueprint() if blueprint is None else blueprint,
        execution_mode="compiled",
        backend="jit",
        optimization_level=3,
        selector_structural_zero_proven=True,
    )


def _evaluate(stage: GenericCompiledStageBlueprint, values: tuple[int, ...]):
    substitutions = dict(zip(stage.parameter_symbols, values, strict=True))
    return tuple(output.evaluate(substitutions) for output in stage.output_expressions)


def test_composition_rebinds_colliding_local_symbols_and_preserves_order() -> None:
    blueprint = _terminal_blueprint()
    first, second = blueprint.stages
    assert first.parameter_symbols[0] == second.parameter_symbols[0]

    pair, full = _compose(blueprint)

    assert pair.kind == "pair"
    assert pair.elided_stage_indices == (6, 7)
    assert pair.dependency_components == (10, 11)
    assert tuple(item.global_component for item in pair.semantic_inputs) == (0, 1, 2)
    assert _evaluate(pair.stage, (2, 3, 5)) == pytest.approx((31, -1))

    assert full.kind == "full-tail"
    assert full.elided_stage_indices == (6, 7)
    assert full.dependency_components == (10, 11, 20, 21)
    assert tuple(item.global_component for item in full.semantic_inputs) == (
        0,
        1,
        2,
        3,
    )
    assert _evaluate(full.stage, (2, 3, 5, 7)) == pytest.approx((38, -7))
    assert full.stage.output_slots == blueprint.amplitude_stage.output_slots


def test_composition_allows_earlier_streamed_expressions_to_be_released() -> None:
    blueprint = _terminal_blueprint()
    first, second = blueprint.stages
    released = replace(
        first,
        stage_index=5,
        evaluator_label="released-earlier-stage",
        parameter_symbols=(),
        output_expressions=(),
    )
    blueprint = replace(
        blueprint,
        stage_count=4,
        stages=(released, first, second),
    )

    pair, full = _compose(blueprint)

    assert _evaluate(pair.stage, (2, 3, 5)) == pytest.approx((31, -1))
    assert _evaluate(full.stage, (2, 3, 5, 7)) == pytest.approx((38, -7))


def test_full_tail_preserves_mixed_planes_and_real_scalars() -> None:
    blueprint = _terminal_blueprint()
    amplitude = blueprint.amplitude_stage
    inputs = (
        *amplitude.input_components,
        _real_input(
            kind="momentum",
            source_id=30,
            global_component=30,
            parameter_index=3,
        ),
        _real_input(
            kind="model_parameter",
            source_id=0,
            global_component=40,
            parameter_index=4,
        ),
    )
    symbols = _local_symbols(len(inputs))
    amplitude = replace(
        amplitude,
        input_components=inputs,
        parameter_count=5,
        value_parameter_count=3,
        momentum_parameter_count=1,
        model_parameter_count=1,
        real_valued_inputs=(3, 4),
        parameter_symbols=symbols,
        output_expressions=(
            symbols[0] + symbols[2] + symbols[3] * symbols[4],
            symbols[1] * symbols[2],
        ),
    )

    _pair, full = _compose(replace(blueprint, amplitude_stage=amplitude))

    assert tuple(
        (item.global_component, item.kind, item.real_valued)
        for item in full.semantic_inputs
    ) == (
        (0, "value", False),
        (1, "value", False),
        (2, "value", False),
        (3, "value", False),
        (30, "momentum", True),
        (40, "model_parameter", True),
    )
    assert full.stage.value_parameter_count == 4
    assert full.stage.momentum_parameter_count == 1
    assert full.stage.model_parameter_count == 1
    assert full.stage.real_valued_inputs == (4, 5)
    assert _evaluate(full.stage, (2, 3, 5, 7, 11, 13)) == pytest.approx((181, -7))


@pytest.mark.parametrize(
    ("override", "match"),
    (
        ({"execution_mode": "eager"}, "not compiled"),
        ({"backend": "compiled-complex"}, "not JIT"),
        ({"optimization_level": 2}, "not O3"),
        (
            {"selector_structural_zero_proven": False},
            "proof is incomplete",
        ),
    ),
)
def test_composition_rejects_ineligible_execution_settings(
    override: dict[str, object],
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "execution_mode": "compiled",
        "backend": "jit",
        "optimization_level": 3,
        "selector_structural_zero_proven": True,
    }
    arguments.update(override)
    with pytest.raises(TerminalSuperkernelError, match=match):
        compose_terminal_superkernels(_terminal_blueprint(), **arguments)  # type: ignore[arg-type]


def test_composition_rejects_an_unavailable_runtime_blueprint() -> None:
    with pytest.raises(TerminalSuperkernelError, match="runtime is unavailable"):
        _compose(replace(_terminal_blueprint(), runtime_available=False))


def test_composition_rejects_a_complex_momentum_input() -> None:
    blueprint = _terminal_blueprint()
    amplitude = blueprint.amplitude_stage
    momentum = GenericStageInputComponent(
        kind="momentum",
        source_id=30,
        component=0,
        global_component=30,
        parameter_index=3,
        real_valued=False,
    )
    symbols = _local_symbols(4)
    amplitude = replace(
        amplitude,
        input_components=(*amplitude.input_components, momentum),
        parameter_count=4,
        value_parameter_count=3,
        momentum_parameter_count=1,
        parameter_symbols=symbols,
        output_expressions=(
            symbols[0] + symbols[2] + symbols[3],
            symbols[1] * symbols[2],
        ),
    )

    with pytest.raises(TerminalSuperkernelError, match="complex-valued momentum"):
        _compose(replace(blueprint, amplitude_stage=amplitude))


def test_composition_rejects_an_external_terminal_output_consumer() -> None:
    blueprint = _terminal_blueprint()
    amplitude = blueprint.amplitude_stage
    inputs = list(amplitude.input_components)
    inputs[2] = _input(
        source_id=10,
        component=0,
        global_component=10,
        parameter_index=2,
    )
    amplitude = replace(
        amplitude,
        input_components=tuple(inputs),
        input_value_slot_ids=(10, 20),
    )
    with pytest.raises(TerminalSuperkernelError, match="external or duplicate"):
        _compose(replace(blueprint, amplitude_stage=amplitude))


def test_composition_rejects_conflicting_external_semantic_aliases() -> None:
    blueprint = _terminal_blueprint()
    first, second = blueprint.stages
    inputs = list(second.input_components)
    inputs[2] = _input(
        source_id=2,
        component=0,
        global_component=0,
        parameter_index=2,
    )
    second = replace(second, input_components=tuple(inputs))
    with pytest.raises(TerminalSuperkernelError, match="conflicting semantic"):
        _compose(replace(blueprint, stages=(first, second)))


def test_composition_rejects_selector_domain_mismatch() -> None:
    blueprint = _terminal_blueprint()
    amplitude = blueprint.amplitude_stage
    slots = tuple(
        replace(slot, selector_domain_ids=(99,)) for slot in amplitude.output_slots
    )
    with pytest.raises(TerminalSuperkernelError, match="selector-domain"):
        _compose(
            replace(
                blueprint,
                amplitude_stage=replace(amplitude, output_slots=slots),
            )
        )


def test_composition_rejects_non_simultaneous_expression_objects() -> None:
    blueprint = _terminal_blueprint()
    second = replace(
        blueprint.stages[-1],
        output_expressions=("not-an-expression",) * 2,
    )
    with pytest.raises(TerminalSuperkernelError, match="simultaneous replacement"):
        _compose(replace(blueprint, stages=(blueprint.stages[0], second)))
