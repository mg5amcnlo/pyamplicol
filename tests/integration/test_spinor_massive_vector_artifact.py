# SPDX-License-Identifier: 0BSD
"""End-to-end graph-backed coverage for one massive vector source."""

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


@dataclass(frozen=True)
class _MassiveVectorCase:
    expression: str
    process_id: str
    color_order: tuple[int, ...]
    color_flow: str
    qcd_order: int
    qed_order: int
    point: tuple[tuple[float, ...], ...]
    tracked_oracle: float | None = None
    tracked_alpha_ew: float | None = None


_CASES = (
    _MassiveVectorCase(
        expression="d d~ > z",
        process_id="d_dbar_to_z",
        color_order=(2, 1),
        color_flow="flow:2,1",
        qcd_order=0,
        qed_order=1,
        point=(
            (45.594, 0.0, 0.0, 45.594),
            (45.594, 0.0, 0.0, -45.594),
            (91.188, 0.0, 0.0, 0.0),
        ),
    ),
    _MassiveVectorCase(
        expression="d d~ > z g",
        process_id="d_dbar_to_z_g",
        color_order=(2, 4, 1),
        color_flow="flow:2,4,1",
        qcd_order=1,
        qed_order=1,
        point=(
            (500.0, 0.0, 0.0, 500.0),
            (500.0, 0.0, 0.0, -500.0),
            (
                504.15762567199999,
                -438.65194577979531,
                1.0884904775248876,
                -231.17730388450397,
            ),
            (
                495.84237432800001,
                438.65194577979531,
                -1.0884904775248876,
                231.17730388450397,
            ),
        ),
    ),
    _MassiveVectorCase(
        expression="u d~ > w+",
        process_id="u_dbar_to_wplus",
        color_order=(2, 1),
        color_flow="flow:2,1",
        qcd_order=0,
        qed_order=1,
        point=(
            (40.20950122287808, 0.0, 0.0, 40.20950122287808),
            (40.20950122287808, 0.0, 0.0, -40.20950122287808),
            (80.41900244575616, 0.0, 0.0, 0.0),
        ),
        tracked_oracle=229.9705676139197,
        tracked_alpha_ew=0.007546771114,
    ),
    _MassiveVectorCase(
        expression="u d~ > e+ ve",
        process_id="u_dbar_to_positron_nue",
        color_order=(2, 1),
        color_flow="flow:2,1",
        qcd_order=0,
        qed_order=2,
        point=(
            (500.0, 0.0, 0.0, 500.0),
            (500.0, 0.0, 0.0, -500.0),
            (
                499.99999999999994,
                -306.65836769058797,
                210.51071473894038,
                334.13453054936508,
            ),
            (
                499.99999999999994,
                306.65836769058797,
                -210.51071473894038,
                -334.13453054936508,
            ),
        ),
        tracked_oracle=0.000422900873179271016676804851172109,
    ),
)


def _config(qcd_order: int, qed_order: int) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc", lc_flow_layout="topology-replay"),
        process=ProcessConfig(
            coupling_order_policy="explicit",
            max_coupling_orders={"QCD": qcd_order, "QED": qed_order},
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


def _two_body_z_gluon_point(
    z_mass: float,
) -> tuple[tuple[float, ...], ...]:
    center_of_mass_energy = 1_000.0
    invariant_mass = center_of_mass_energy**2
    momentum = (invariant_mass - z_mass**2) / (2.0 * center_of_mass_energy)
    z_energy = (invariant_mass + z_mass**2) / (2.0 * center_of_mass_energy)
    z_component = 0.3 * momentum
    x_component = math.sqrt(momentum**2 - z_component**2)
    return (
        (500.0, 0.0, 0.0, 500.0),
        (500.0, 0.0, 0.0, -500.0),
        (z_energy, -x_component, 0.0, -z_component),
        (momentum, x_component, 0.0, z_component),
    )


def _value(runtime: Runtime, case: _MassiveVectorCase, point: object) -> complex:
    return complex(runtime.evaluate((point,), color_flows=(case.color_flow,))[0])


def test_graph_spinor_massive_vector_processes(tmp_path: Path) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    if importlib.util.find_spec("symbolica") is None:
        pytest.skip("Symbolica is unavailable")

    compile_config = _config(
        max(case.qcd_order for case in _CASES),
        max(case.qed_order for case in _CASES),
    )
    prepared = ModelSource.built_in_sm().compile(
        cache_dir=tmp_path / "model-cache",
        use_cache=False,
        prepared_output=tmp_path / "built-in-sm-jit-o2.pyamplicol-model",
        evaluator=compile_config.evaluator,
    )

    runtimes: dict[str, tuple[Runtime, Runtime]] = {}
    for case in _CASES:
        artifacts: dict[str, Path] = {}
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
                config=_config(case.qcd_order, case.qed_order),
                model=prepared,
            )
            artifacts[mode] = artifact

        execution = json.loads(
            (
                artifacts["spinor"] / "processes" / case.process_id / "execution.json"
            ).read_text(encoding="utf-8")
        )
        assert execution["graph_payload"] == {
            "abi": "pyamplicol-spinor-dag-binary-v2",
            "path": "spinor-dag-v2.bin",
        }
        assert "process_family" not in execution

        candidate = Runtime.load(artifacts["spinor"], process=case.process_id)
        reference = Runtime.load(artifacts["component"], process=case.process_id)
        assert candidate.execution_mode == "spinor"
        assert reference.execution_mode == "compiled"
        if case.tracked_alpha_ew is not None:
            for runtime in (candidate, reference):
                runtime.set_model_parameters(
                    {"normalization.alpha_ew": case.tracked_alpha_ew}
                )
        reference_value = _value(reference, case, case.point)
        assert _value(candidate, case, case.point) == pytest.approx(
            reference_value,
            rel=2.0e-12,
            abs=1.0e-15,
        )
        if case.tracked_oracle is not None:
            assert reference_value == pytest.approx(
                case.tracked_oracle,
                rel=2.0e-12,
                abs=1.0e-15,
            )
        runtimes[case.process_id] = (candidate, reference)

    coupling_name = "coupling.10.1_23_1.component_0"
    z_gluon_case = _CASES[1]
    candidate, reference = runtimes[z_gluon_case.process_id]
    for runtime in (candidate, reference):
        runtime.set_model_parameters({coupling_name: 0.0})
    assert _value(candidate, z_gluon_case, z_gluon_case.point) == pytest.approx(
        _value(reference, z_gluon_case, z_gluon_case.point),
        rel=2.0e-12,
        abs=1.0e-15,
    )

    shifted_mass = 100.0
    shifted_point = _two_body_z_gluon_point(shifted_mass)
    for runtime in (candidate, reference):
        runtime.set_model_parameters({"particle.23.mass": shifted_mass})
    assert _value(candidate, z_gluon_case, shifted_point) == pytest.approx(
        _value(reference, z_gluon_case, shifted_point),
        rel=2.0e-12,
        abs=1.0e-15,
    )

    internal_w_case = _CASES[3]
    candidate, reference = runtimes[internal_w_case.process_id]
    baseline = _value(reference, internal_w_case, internal_w_case.point)
    for update in (
        {"particle.24.width": 20.0},
        {"particle.24.mass": 100.0},
    ):
        for runtime in (candidate, reference):
            runtime.set_model_parameters(update)
        shifted_reference = _value(reference, internal_w_case, internal_w_case.point)
        shifted_candidate = _value(
            candidate,
            internal_w_case,
            internal_w_case.point,
        )
        assert shifted_candidate == pytest.approx(
            shifted_reference,
            rel=2.0e-12,
            abs=1.0e-15,
        )
        assert shifted_reference != pytest.approx(
            baseline,
            rel=1.0e-9,
            abs=1.0e-15,
        )
