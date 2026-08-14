# SPDX-License-Identifier: 0BSD
"""End-to-end graph-backed spinor coverage for external scalar contacts."""

from __future__ import annotations

import importlib.util
import json
import math
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

_MODEL = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "scalars"
    / "scalars.json"
)
_S = 1_000_000.0
_FINAL_ONE_ENERGY = (_S + 1.0 - 4.0) / 2_000.0
_FINAL_TWO_ENERGY = (_S + 4.0 - 1.0) / 2_000.0
_FINAL_MOMENTUM = math.sqrt(_FINAL_ONE_ENERGY**2 - 1.0)
_POINT_0012 = (
    (500.0, 0.0, 0.0, 500.0),
    (500.0, 0.0, 0.0, -500.0),
    (_FINAL_ONE_ENERGY, 0.0, 0.0, _FINAL_MOMENTUM),
    (_FINAL_TWO_ENERGY, 0.0, 0.0, -_FINAL_MOMENTUM),
)
_POINT_0000 = (
    (500.0, 0.0, 0.0, 500.0),
    (500.0, 0.0, 0.0, -500.0),
    (500.0, 0.0, 0.0, 500.0),
    (500.0, 0.0, 0.0, -500.0),
)


@dataclass(frozen=True)
class _ScalarCase:
    expression: str
    process_id: str
    point: tuple[tuple[float, ...], ...]
    normalization: float


_CASES = (
    _ScalarCase(
        "scalar_0 scalar_0 > scalar_0 scalar_0",
        "scalars_0000",
        _POINT_0000,
        0.5,
    ),
    _ScalarCase(
        "scalar_0 scalar_0 > scalar_1 scalar_2",
        "scalars_0012",
        _POINT_0012,
        1.0,
    ),
)


def _config() -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc", lc_flow_layout="topology-replay"),
        process=ProcessConfig(
            coupling_order_policy="explicit",
            max_coupling_orders={"QCD": 1, "QED": 0},
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


def _resolved_total(runtime: Runtime, point: object) -> float:
    resolved = runtime.evaluate_resolved((point,))
    assert resolved.helicity_ids == ("h:sum",)
    assert resolved.color_ids == ("flow:singlet",)
    value = complex(resolved.total()[0])
    assert value.imag == pytest.approx(0.0, abs=1.0e-15)
    return value.real


def test_graph_spinor_scalar_contacts_and_lambda_refresh(tmp_path: Path) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    if importlib.util.find_spec("symbolica") is None:
        pytest.skip("Symbolica is unavailable")

    config = _config()
    prepared_path = tmp_path / "scalar-jit-o2.pyamplicol-model"
    prepared = ModelSource.from_path(_MODEL).compile(
        cache_dir=tmp_path / "model-cache",
        use_cache=False,
        prepared_output=prepared_path,
        evaluator=config.evaluator,
    )

    for case in _CASES:
        artifact = tmp_path / case.process_id
        generate_slice(
            ProcessRequest.parse(case.expression, name=case.process_id),
            artifact,
            selection=GenerationSlice(experimental_spinor_dag=True),
            config=config,
            model=prepared,
        )
        execution = json.loads(
            (artifact / "processes" / case.process_id / "execution.json").read_text(
                encoding="utf-8"
            )
        )
        assert execution["graph_payload"] == {
            "abi": "pyamplicol-spinor-dag-binary-v3",
            "path": "spinor-dag-v3.bin",
        }
        assert "process_family" not in execution

        runtime = Runtime.load(artifact, process=case.process_id)
        assert runtime.execution_mode == "spinor"
        assert _resolved_total(runtime, case.point) == pytest.approx(
            case.normalization
        )
        runtime.set_model_parameters({"lam": 3.0})
        assert _resolved_total(runtime, case.point) == pytest.approx(
            9.0 * case.normalization
        )
