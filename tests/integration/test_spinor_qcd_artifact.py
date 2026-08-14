# SPDX-License-Identifier: 0BSD
"""End-to-end graph-backed spinor coverage for one mixed QCD process."""

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

_UUGG_POINT = (
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
_UUGGG_POINT = (
    (500.0, 0.0, 0.0, 500.0),
    (500.0, 0.0, 0.0, -500.0),
    (
        112.87279513068559,
        -54.9176420897714,
        61.060949618307056,
        -77.43307368059243,
    ),
    (
        393.36301586823254,
        -277.3830888376279,
        264.9224724711745,
        -87.23054430420437,
    ),
    (
        493.7641890010819,
        332.30073092739934,
        -325.9834220894817,
        164.6636179847968,
    ),
)
_GGGG_POINT = (
    (500.0, 0.0, 0.0, 500.0),
    (500.0, 0.0, 0.0, -500.0),
    (
        500.0,
        -454.42735730865138,
        -32.064220250936174,
        206.07683690598216,
    ),
    (
        500.0,
        454.42735730865138,
        32.064220250936174,
        -206.07683690598216,
    ),
)


@dataclass(frozen=True)
class _QcdCase:
    expression: str
    process_id: str
    color_order: tuple[int, ...]
    qcd_order: int
    point: tuple[tuple[float, ...], ...]
    component_recurrence_oracle: float


# The first two totals were retained from selected-flow component-recurrence
# artifacts at seed 12345. The four-gluon value is the tracked
# case:sm_gg_gg:lc / generic-1 / flow:1,2,3,4 authority.
_CASES = (
    _QcdCase(
        expression="u u~ > g g",
        process_id="u_ubar_to_g_g",
        color_order=(2, 3, 4, 1),
        qcd_order=2,
        point=_UUGG_POINT,
        component_recurrence_oracle=4.45988346008312,
    ),
    _QcdCase(
        expression="u u~ > g g g",
        process_id="u_ubar_to_g_g_g",
        color_order=(2, 3, 4, 5, 1),
        qcd_order=3,
        point=_UUGGG_POINT,
        component_recurrence_oracle=0.00359195380772719,
    ),
    _QcdCase(
        expression="g g > g g",
        process_id="g_g_to_g_g",
        color_order=(1, 2, 3, 4),
        qcd_order=2,
        point=_GGGG_POINT,
        component_recurrence_oracle=1.7527418719125202,
    ),
)


def _config(qcd_order: int) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc", lc_flow_layout="topology-replay"),
        process=ProcessConfig(
            coupling_order_policy="explicit",
            max_coupling_orders={"QCD": qcd_order, "QED": 0},
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


def test_graph_spinor_qcd_recurrences(tmp_path: Path) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    if importlib.util.find_spec("symbolica") is None:
        pytest.skip("Symbolica is unavailable")

    compile_config = _config(max(case.qcd_order for case in _CASES))
    prepared = ModelSource.built_in_sm().compile(
        cache_dir=tmp_path / "model-cache",
        use_cache=False,
        prepared_output=tmp_path / "built-in-sm-jit-o2.pyamplicol-model",
        evaluator=compile_config.evaluator,
    )
    for case in _CASES:
        artifact = tmp_path / f"{case.process_id}-spinor"
        generate_slice(
            ProcessRequest.parse(case.expression, name=case.process_id),
            artifact,
            selection=GenerationSlice(
                reference_color_order=case.color_order,
                selected_color_sector_ids=(0,),
                experimental_spinor_dag=True,
            ),
            config=_config(case.qcd_order),
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
        assert "kernel_pack" not in execution

        runtime = Runtime.load(artifact, process=case.process_id)
        assert runtime.execution_mode == "spinor"
        value = complex(runtime.evaluate((case.point,))[0])
        assert value == pytest.approx(
            case.component_recurrence_oracle,
            rel=2.0e-12,
        )

        runtime.set_model_parameters({"normalization.alpha_s_me_check": 0.236})
        assert complex(runtime.evaluate((case.point,))[0]) == pytest.approx(
            2.0**case.qcd_order * case.component_recurrence_oracle,
            rel=2.0e-12,
        )
