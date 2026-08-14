# SPDX-License-Identifier: 0BSD
"""End-to-end graph-backed coverage for two massless fermion lines."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _FlowCase:
    color_order: tuple[int, ...]
    color_flow: str
    tracked_value: float


_POINT = (
    (500.0, 0.0, 0.0, 500.0),
    (500.0, 0.0, 0.0, -500.0),
    (500.0, 349.55548503752317, 162.29632791789103, -318.54491806423516),
    (500.0, -349.55548503752317, -162.29632791789103, 318.54491806423516),
)
_FLOWS = (
    _FlowCase((2, 1, 3, 4), "flow:2,1,3,4", 0.08586784491130668),
    _FlowCase((2, 4, 3, 1), "flow:2,4,3,1", 0.7728106042017603),
)
_TRACKED_TOTAL = 0.858678449113067


def _config() -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc", lc_flow_layout="topology-replay"),
        process=ProcessConfig(
            coupling_order_policy="explicit",
            max_coupling_orders={"QCD": 2, "QED": 0},
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


def _value(runtime: Runtime, color_flow: str) -> float:
    return complex(runtime.evaluate((_POINT,), color_flows=(color_flow,))[0]).real


def test_graph_spinor_two_massless_fermion_lines(tmp_path: Path) -> None:
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

    candidate_total = 0.0
    for index, flow in enumerate(_FLOWS):
        runtimes: dict[str, Runtime] = {}
        for mode in ("spinor", "component"):
            artifact = tmp_path / f"ddbar-uubar-flow-{index}-{mode}"
            generate_slice(
                ProcessRequest.parse("d d~ > u u~", name="d_dbar_to_u_ubar"),
                artifact,
                selection=GenerationSlice(
                    reference_color_order=flow.color_order,
                    selected_color_sector_ids=(0,),
                    experimental_spinor_dag=mode == "spinor",
                ),
                config=config,
                model=prepared,
            )
            if mode == "spinor":
                execution = json.loads(
                    (
                        artifact / "processes" / "d_dbar_to_u_ubar" / "execution.json"
                    ).read_text(encoding="utf-8")
                )
                assert execution["graph_payload"] == {
                    "abi": "pyamplicol-spinor-dag-binary-v2",
                    "path": "spinor-dag-v2.bin",
                }
                assert "process_family" not in execution
            runtimes[mode] = Runtime.load(artifact, process="d_dbar_to_u_ubar")

        candidate = runtimes["spinor"]
        reference = runtimes["component"]
        assert candidate.execution_mode == "spinor"
        assert reference.execution_mode == "compiled"
        reference_value = _value(reference, flow.color_flow)
        candidate_value = _value(candidate, flow.color_flow)
        assert candidate_value == pytest.approx(
            reference_value,
            rel=2.0e-12,
            abs=1.0e-15,
        )
        assert reference_value == pytest.approx(
            flow.tracked_value,
            rel=2.0e-12,
            abs=1.0e-15,
        )
        candidate_total += candidate_value

    assert candidate_total == pytest.approx(
        _TRACKED_TOTAL,
        rel=2.0e-12,
        abs=1.0e-15,
    )
