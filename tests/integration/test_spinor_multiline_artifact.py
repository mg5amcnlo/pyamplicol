# SPDX-License-Identifier: 0BSD
"""End-to-end graph-backed coverage for two massless fermion lines."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from pyamplicol import CompiledModel, ModelSource, ProcessRequest, Runtime
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


@dataclass(frozen=True)
class _ProcessCase:
    expression: str
    process_id: str
    point: tuple[tuple[float, float, float, float], ...]
    flows: tuple[_FlowCase, ...]
    tracked_total: float


_CASES = (
    _ProcessCase(
        expression="d d~ > u u~",
        process_id="d_dbar_to_u_ubar",
        point=(
            (500.0, 0.0, 0.0, 500.0),
            (500.0, 0.0, 0.0, -500.0),
            (500.0, 349.55548503752317, 162.29632791789103, -318.54491806423516),
            (500.0, -349.55548503752317, -162.29632791789103, 318.54491806423516),
        ),
        flows=(
            _FlowCase((2, 1, 3, 4), "flow:2,1,3,4", 0.08586784491130668),
            _FlowCase((2, 4, 3, 1), "flow:2,4,3,1", 0.7728106042017603),
        ),
        tracked_total=0.858678449113067,
    ),
    _ProcessCase(
        expression="d d~ > d d~",
        process_id="d_dbar_to_d_dbar",
        point=(
            (500.0, 0.0, 0.0, 500.0),
            (500.0, 0.0, 0.0, -500.0),
            (500.0, -92.37565236398261, -439.14510003273597, 220.49562346578892),
            (500.0, 92.37565236398261, 439.14510003273597, -220.49562346578892),
        ),
        flows=(
            _FlowCase((2, 1, 3, 4), "flow:2,1,3,4", 22.812191406806715),
            _FlowCase((2, 4, 3, 1), "flow:2,4,3,1", 4.393177047511702),
        ),
        tracked_total=27.205368454318417,
    ),
)

_FULL_DIRAC_PAIR_PROCESS = "t t~ > d d~"
_FULL_DIRAC_PAIR_PROCESS_ID = "t_tbar_to_d_dbar"
_FULL_DIRAC_PAIR_FLOW = "flow:2,4,3,1"
_FULL_DIRAC_PAIR_ORDER = (2, 4, 3, 1)
# Time reversal of catalog:dd_tt_jets:n2 in the checked-in seed-101
# numerical-acceptance fixture. The selected crossed LC flow carries nine
# tenths of the tracked two-flow value, as fixed by the distinct-quark colour
# decomposition.
_FULL_DIRAC_PAIR_POINT = (
    (
        500.00000000000006,
        -287.71746404131949,
        197.50841776906941,
        313.49654829996268,
    ),
    (
        500.00000000000006,
        287.71746404131949,
        -197.50841776906941,
        -313.49654829996268,
    ),
    (500.0, 0.0, 0.0, 500.0),
    (500.0, 0.0, 0.0, -500.0),
)
_FULL_DIRAC_PAIR_TRACKED_VALUE = 0.8316023356715473


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


def _value(
    runtime: Runtime,
    point: tuple[tuple[float, float, float, float], ...],
    color_flow: str,
) -> float:
    return complex(runtime.evaluate((point,), color_flows=(color_flow,))[0]).real


@pytest.fixture(scope="module")
def prepared_model(tmp_path_factory: pytest.TempPathFactory) -> CompiledModel:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    if importlib.util.find_spec("symbolica") is None:
        pytest.skip("Symbolica is unavailable")

    config = _config()
    root = tmp_path_factory.mktemp("spinor-multiline-built-in-sm")
    return ModelSource.built_in_sm().compile(
        cache_dir=root / "model-cache",
        use_cache=False,
        prepared_output=root / "built-in-sm-jit-o2.pyamplicol-model",
        evaluator=config.evaluator,
    )


def test_graph_spinor_two_massless_fermion_lines(
    tmp_path: Path,
    prepared_model: CompiledModel,
) -> None:
    config = _config()
    for case in _CASES:
        candidate_total = 0.0
        for index, flow in enumerate(case.flows):
            runtimes: dict[str, Runtime] = {}
            for mode in ("spinor", "component"):
                artifact = tmp_path / f"{case.process_id}-flow-{index}-{mode}"
                generate_slice(
                    ProcessRequest.parse(case.expression, name=case.process_id),
                    artifact,
                    selection=GenerationSlice(
                        reference_color_order=flow.color_order,
                        selected_color_sector_ids=(0,),
                        experimental_spinor_dag=mode == "spinor",
                    ),
                    config=config,
                    model=prepared_model,
                )
                if mode == "spinor":
                    execution = json.loads(
                        (
                            artifact / "processes" / case.process_id / "execution.json"
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
            reference_value = _value(reference, case.point, flow.color_flow)
            candidate_value = _value(candidate, case.point, flow.color_flow)
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
            case.tracked_total,
            rel=2.0e-12,
            abs=1.0e-15,
        )


def test_graph_spinor_full_dirac_pair_to_vector(
    tmp_path: Path,
    prepared_model: CompiledModel,
) -> None:
    # Crossing the tracked ddbar -> ttbar point makes the closure source the
    # final massless dbar. The incoming massive top pair must therefore form
    # the internal gluon through the full four-component Dirac bilinear.
    runtimes: dict[str, Runtime] = {}
    for mode in ("spinor", "component"):
        artifact = tmp_path / f"{_FULL_DIRAC_PAIR_PROCESS_ID}-{mode}"
        generate_slice(
            ProcessRequest.parse(
                _FULL_DIRAC_PAIR_PROCESS,
                name=_FULL_DIRAC_PAIR_PROCESS_ID,
            ),
            artifact,
            selection=GenerationSlice(
                reference_color_order=_FULL_DIRAC_PAIR_ORDER,
                selected_color_sector_ids=(0,),
                experimental_spinor_dag=mode == "spinor",
            ),
            config=_config(),
            model=prepared_model,
        )
        runtimes[mode] = Runtime.load(
            artifact,
            process=_FULL_DIRAC_PAIR_PROCESS_ID,
        )

    candidate = runtimes["spinor"]
    reference = runtimes["component"]
    assert candidate.execution_mode == "spinor"
    assert reference.execution_mode == "compiled"

    baseline = _value(
        reference,
        _FULL_DIRAC_PAIR_POINT,
        _FULL_DIRAC_PAIR_FLOW,
    )
    assert baseline == pytest.approx(
        _FULL_DIRAC_PAIR_TRACKED_VALUE,
        rel=2.0e-12,
        abs=1.0e-15,
    )
    assert _value(
        candidate,
        _FULL_DIRAC_PAIR_POINT,
        _FULL_DIRAC_PAIR_FLOW,
    ) == pytest.approx(baseline, rel=2.0e-12, abs=1.0e-15)

    # This QCD primitive has one fixed vector coupling rather than independent
    # chiral owners. Doubling alpha_s is therefore the minimal live-parameter
    # discriminator; a QCD-order-two matrix element must increase fourfold.
    for runtime in (candidate, reference):
        runtime.set_model_parameters({"normalization.alpha_s_me_check": 0.236})
    shifted_reference = _value(
        reference,
        _FULL_DIRAC_PAIR_POINT,
        _FULL_DIRAC_PAIR_FLOW,
    )
    assert shifted_reference == pytest.approx(
        4.0 * _FULL_DIRAC_PAIR_TRACKED_VALUE,
        rel=2.0e-12,
        abs=1.0e-15,
    )
    assert _value(
        candidate,
        _FULL_DIRAC_PAIR_POINT,
        _FULL_DIRAC_PAIR_FLOW,
    ) == pytest.approx(shifted_reference, rel=2.0e-12, abs=1.0e-15)
