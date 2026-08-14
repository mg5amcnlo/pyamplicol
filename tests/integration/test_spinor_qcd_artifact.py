# SPDX-License-Identifier: 0BSD
"""End-to-end graph-backed spinor coverage for one mixed QCD process."""

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

_POINT = (
    (500.0, 0.0, 0.0, 500.0),
    (500.0, 0.0, 0.0, -500.0),
    (
        500.00000000000176,
        272.72063884871818,
        -293.34745598362332,
        -299.28368350761281,
    ),
    (
        499.99999999999829,
        -272.72063884871545,
        293.34745598361877,
        299.283683507614,
    ),
)
_COMPONENT_RECURRENCE_ORACLE = 4.45988346008312


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


def test_graph_spinor_u_ubar_to_two_gluons(tmp_path: Path) -> None:
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
    artifact = tmp_path / "uugg-spinor"
    generate_slice(
        ProcessRequest.parse("u u~ > g g", name="u_ubar_to_g_g"),
        artifact,
        selection=GenerationSlice(
            selected_color_sector_ids=(0,),
            experimental_spinor_dag=True,
        ),
        config=config,
        model=prepared,
    )

    execution = json.loads(
        (artifact / "processes/u_ubar_to_g_g/execution.json").read_text(
            encoding="utf-8"
        )
    )
    assert execution["graph_payload"] == {
        "abi": "pyamplicol-spinor-dag-binary-v2",
        "path": "spinor-dag-v2.bin",
    }
    assert "process_family" not in execution
    assert "kernel_pack" not in execution

    runtime = Runtime.load(artifact, process="u_ubar_to_g_g")
    assert runtime.execution_mode == "spinor"
    value = complex(runtime.evaluate((_POINT,))[0])
    assert value == pytest.approx(_COMPONENT_RECURRENCE_ORACLE, rel=2.0e-13)

    runtime.set_model_parameters({"normalization.alpha_s_me_check": 0.236})
    assert complex(runtime.evaluate((_POINT,))[0]) == pytest.approx(
        4.0 * _COMPONENT_RECURRENCE_ORACLE,
        rel=2.0e-13,
    )
