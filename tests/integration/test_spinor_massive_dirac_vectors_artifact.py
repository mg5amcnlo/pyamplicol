# SPDX-License-Identifier: 0BSD
"""Graph-backed massive Dirac line with several massless vectors."""

from __future__ import annotations

import importlib.util
import json
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

_PROCESS_ID = "g_g_to_t_tbar_g"
_FLOW_WORD = (3, 1, 2, 5, 4)
_FLOW_ID = "flow:3,1,2,5,4"
_POINT = (
    (500.0, 0.0, 0.0, 500.0),
    (500.0, 0.0, 0.0, -500.0),
    (
        327.78016913065488,
        -180.39074377820512,
        113.24869646396822,
        179.28957466534038,
    ),
    (
        486.19081801910795,
        233.31345741167567,
        -192.5480392979517,
        -339.03184906320229,
    ),
    (
        186.02901285023736,
        -52.92271363347062,
        79.29934283398349,
        159.74227439786188,
    ),
)


def _config() -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc", lc_flow_layout="topology-replay"),
        process=ProcessConfig(
            coupling_order_policy="explicit",
            max_coupling_orders={"QCD": 3, "QED": 0},
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


def _value(runtime: Runtime) -> complex:
    return complex(runtime.evaluate((_POINT,), color_flows=(_FLOW_ID,))[0])


def test_graph_spinor_massive_dirac_line_uses_per_vector_temporal_references(
    tmp_path: Path,
) -> None:
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
    artifacts: dict[str, Path] = {}
    for mode in ("spinor", "component"):
        artifact = tmp_path / mode
        generate_slice(
            ProcessRequest.parse("g g > t t~ g", name=_PROCESS_ID),
            artifact,
            selection=GenerationSlice(
                reference_color_order=_FLOW_WORD,
                selected_color_sector_ids=(0,),
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
        "abi": "pyamplicol-spinor-dag-binary-v3",
        "path": "spinor-dag-v3.bin",
    }
    assert "process_family" not in execution

    candidate = Runtime.load(artifacts["spinor"], process=_PROCESS_ID)
    reference = Runtime.load(artifacts["component"], process=_PROCESS_ID)
    assert candidate.execution_mode == "spinor"
    assert reference.execution_mode == "compiled"

    baseline = _value(reference)
    assert _value(candidate) == pytest.approx(baseline, rel=2.0e-12, abs=1.0e-15)

    candidate_batch = candidate.evaluate((_POINT, _POINT), color_flows=(_FLOW_ID,))
    assert candidate_batch == pytest.approx((baseline, baseline), rel=2.0e-12)

    for runtime in (candidate, reference):
        runtime.set_model_parameters({"particle.6.width": 20.0})
    shifted = _value(reference)
    assert shifted != pytest.approx(baseline, rel=1.0e-9, abs=1.0e-15)
    assert _value(candidate) == pytest.approx(shifted, rel=2.0e-12, abs=1.0e-15)
