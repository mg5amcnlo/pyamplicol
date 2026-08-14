# SPDX-License-Identifier: 0BSD
"""End-to-end graph-backed coverage for one massive Yukawa line."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from pyamplicol import ModelSource, ProcessRequest, Runtime
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RunConfig,
)
from tools.developer.generation_slice import GenerationSlice, generate_slice

_PROCESS_ID = "g_h_to_t_tbar"
_FLOW = "flow:3,1,4"


def _config() -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc", lc_flow_layout="topology-replay"),
        process=ProcessConfig(
            coupling_order_policy="explicit",
            max_coupling_orders={"QCD": 1, "QED": 1},
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="compiled",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


def _point(top_mass: float) -> tuple[tuple[float, ...], ...]:
    center_of_mass_energy = 1_000.0
    higgs_mass = 125.0
    invariant_mass = center_of_mass_energy**2
    incoming_momentum = (invariant_mass - higgs_mass**2) / (2.0 * center_of_mass_energy)
    outgoing_energy = center_of_mass_energy / 2.0
    outgoing_momentum = math.sqrt(outgoing_energy**2 - top_mass**2)
    outgoing_z = 0.2 * outgoing_momentum
    outgoing_x = math.sqrt(outgoing_momentum**2 - outgoing_z**2)
    return (
        (incoming_momentum, 0.0, 0.0, incoming_momentum),
        (
            center_of_mass_energy - incoming_momentum,
            0.0,
            0.0,
            -incoming_momentum,
        ),
        (outgoing_energy, outgoing_x, 0.0, outgoing_z),
        (outgoing_energy, -outgoing_x, 0.0, -outgoing_z),
    )


def _value(runtime: Runtime, point: object) -> complex:
    return complex(runtime.evaluate((point,), color_flows=(_FLOW,))[0])


def test_graph_spinor_massive_yukawa_line(tmp_path: Path) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    if importlib.util.find_spec("symbolica") is None:
        pytest.skip("Symbolica is unavailable")

    config = _config()
    prepared = ModelSource.built_in_sm().compile(
        cache_dir=tmp_path / "model-cache",
        use_cache=False,
        prepared_output=tmp_path / "built-in-sm-jit-o2.pyamplicol-model",
        evaluator=config.evaluator,
    )
    selection = GenerationSlice(
        reference_color_order=(3, 1, 4),
        selected_color_sector_ids=(0,),
    )
    artifacts: dict[str, Path] = {}
    for mode in ("spinor", "component"):
        artifact = tmp_path / mode
        generate_slice(
            ProcessRequest.parse("g h > t t~", name=_PROCESS_ID),
            artifact,
            selection=GenerationSlice(
                reference_color_order=selection.reference_color_order,
                selected_color_sector_ids=selection.selected_color_sector_ids,
                experimental_spinor_dag=mode == "spinor",
            ),
            config=config,
            model=prepared,
        )
        artifacts[mode] = artifact

    execution = json.loads(
        (artifacts["spinor"] / "processes" / _PROCESS_ID / "execution.json").read_text(
            encoding="utf-8"
        )
    )
    assert execution["graph_payload"] == {
        "abi": "pyamplicol-spinor-dag-binary-v2",
        "path": "spinor-dag-v2.bin",
    }
    assert "process_family" not in execution

    candidate = Runtime.load(artifacts["spinor"], process=_PROCESS_ID)
    reference = Runtime.load(artifacts["component"], process=_PROCESS_ID)
    assert candidate.execution_mode == "spinor"
    assert reference.execution_mode == "compiled"

    default_point = _point(173.0)
    reference_default = _value(reference, default_point)
    assert reference_default == pytest.approx(1.3859590047142938, rel=2.0e-13)
    assert _value(candidate, default_point) == pytest.approx(
        reference_default,
        rel=2.0e-12,
    )

    yukawa_name = "coupling.16.6_25_6.component_0"
    yukawa_default = 2.281602450356579
    for runtime in (candidate, reference):
        runtime.set_model_parameters({yukawa_name: 2.0 * yukawa_default})
    assert _value(reference, default_point) == pytest.approx(
        4.0 * reference_default,
        rel=2.0e-13,
    )
    assert _value(candidate, default_point) == pytest.approx(
        _value(reference, default_point),
        rel=2.0e-12,
    )
    for runtime in (candidate, reference):
        runtime.set_model_parameters({yukawa_name: yukawa_default})

    for runtime in (candidate, reference):
        runtime.set_model_parameters({"particle.6.width": 0.0})
    stable_reference = _value(reference, default_point)
    assert stable_reference != pytest.approx(reference_default, rel=1.0e-9)
    assert _value(candidate, default_point) == pytest.approx(
        stable_reference,
        rel=2.0e-12,
    )

    for runtime in (candidate, reference):
        runtime.set_model_parameters(
            {"particle.6.mass": 180.0, "particle.6.width": 1.4915}
        )
    shifted_point = _point(180.0)
    assert _value(candidate, shifted_point) == pytest.approx(
        _value(reference, shifted_point),
        rel=2.0e-12,
    )
