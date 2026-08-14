# SPDX-License-Identifier: 0BSD
"""External-model width ownership in graph-backed spinor artifacts."""

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

_UFO_SM_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)


@dataclass(frozen=True)
class _WidthCase:
    expression: str
    process_id: str
    color_order: tuple[int, ...]
    color_flow: str
    qcd_order: int
    qed_order: int
    width_name: str
    point: tuple[tuple[float, ...], ...]


_CASES = (
    _WidthCase(
        expression="g g > t t~",
        process_id="ufo_g_g_to_t_tbar",
        color_order=(3, 1, 2, 4),
        color_flow="flow:3,1,2,4",
        qcd_order=2,
        qed_order=0,
        width_name="WT",
        point=(
            (500.0, 0.0, 0.0, 500.0),
            (500.0, 0.0, 0.0, -500.0),
            (
                500.00000000000006,
                -287.7174640413195,
                197.5084177690694,
                313.4965482999627,
            ),
            (
                500.00000000000006,
                287.7174640413195,
                -197.5084177690694,
                -313.4965482999627,
            ),
        ),
    ),
    _WidthCase(
        expression="u d~ > e+ ve",
        process_id="ufo_u_dbar_to_positron_nue",
        color_order=(2, 1),
        color_flow="flow:2,1",
        qcd_order=0,
        qed_order=2,
        width_name="WW",
        point=(
            (500.0, 0.0, 0.0, 500.0),
            (500.0, 0.0, 0.0, -500.0),
            (
                499.99999999999994,
                -306.65836769058797,
                210.51071473894038,
                334.1345305493651,
            ),
            (
                499.99999999999994,
                306.65836769058797,
                -210.51071473894038,
                -334.1345305493651,
            ),
        ),
    ),
)


def _config(case: _WidthCase) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc", lc_flow_layout="topology-replay"),
        process=ProcessConfig(
            coupling_order_policy="explicit",
            max_coupling_orders={"QCD": case.qcd_order, "QED": case.qed_order},
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


def _value(runtime: Runtime, case: _WidthCase) -> complex:
    return complex(runtime.evaluate((case.point,), color_flows=(case.color_flow,))[0])


def test_graph_spinor_authenticates_external_width_owners(tmp_path: Path) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    if importlib.util.find_spec("symbolica") is None:
        pytest.skip("Symbolica is unavailable")

    compile_config = _config(_CASES[0])
    prepared = ModelSource.from_path(
        _UFO_SM_ROOT / "sm.json",
        restriction=_UFO_SM_ROOT / "restrict_default.json",
    ).compile(
        cache_dir=tmp_path / "model-cache",
        use_cache=False,
        prepared_output=tmp_path / "ufo-sm-jit-o2.pyamplicol-model",
        evaluator=compile_config.evaluator,
    )

    for case in _CASES:
        runtimes: dict[str, Runtime] = {}
        for mode in ("spinor", "component"):
            artifact = tmp_path / f"{case.process_id}-{mode}"
            generate_slice(
                ProcessRequest.parse(case.expression, name=case.process_id),
                artifact,
                selection=GenerationSlice(
                    reference_color_order=case.color_order,
                    selected_color_sector_ids=(0,),
                    experimental_spinor_dag=mode == "spinor",
                ),
                config=_config(case),
                model=prepared,
            )
            if mode == "spinor":
                execution = json.loads(
                    (
                        artifact
                        / "processes"
                        / case.process_id
                        / "execution.json"
                    ).read_text(encoding="utf-8")
                )
                assert execution["graph_payload"] == {
                    "abi": "pyamplicol-spinor-dag-binary-v3",
                    "path": "spinor-dag-v3.bin",
                }
                assert "process_family" not in execution
            runtimes[mode] = Runtime.load(artifact, process=case.process_id)

        candidate = runtimes["spinor"]
        reference = runtimes["component"]
        assert candidate.execution_mode == "spinor"
        assert reference.execution_mode == "compiled"

        baseline = _value(reference, case)
        assert _value(candidate, case) == pytest.approx(
            baseline,
            rel=2.0e-12,
            abs=1.0e-15,
        )

        for runtime in (candidate, reference):
            runtime.set_model_parameters({case.width_name: 20.0})
        shifted = _value(reference, case)
        assert shifted != pytest.approx(baseline, rel=1.0e-9, abs=1.0e-15)
        assert _value(candidate, case) == pytest.approx(
            shifted,
            rel=2.0e-12,
            abs=1.0e-15,
        )
