# SPDX-License-Identifier: 0BSD
"""Strict public metadata for the developer-only spinor execution lane."""

from __future__ import annotations

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Literal

from ..models.base import Model
from ..processes.ir import CanonicalProcessIR
from .contracts import runtime_coupling_parameter_names

_NORMALIZATION_EXTENSION_KEYS = (
    "color_accuracy",
    "color_factor",
    "average_factor",
    "identical_factor",
    "global_coupling_factor",
    "qcd_coupling_power",
    "electroweak_coupling_power",
    "couplings_in_stage_evaluators",
    "coupling_policy",
)

SpinorProcessFamily = Literal[
    "pure-gluon",
    "single-massless-quark-line",
    "single-massive-quark-line",
    "single-massless-quark-line-massive-neutral-vector",
]

_MASSIVE_QUARK_FAMILY = "single-massive-quark-line"
_MASSIVE_NEUTRAL_VECTOR_FAMILY = "single-massless-quark-line-massive-neutral-vector"


def spinor_graph_parameters(
    process: CanonicalProcessIR,
    model: Model,
    *,
    process_family: SpinorProcessFamily,
    ordered_source_labels: Sequence[int],
) -> tuple[tuple[str, float], ...]:
    """Return the ordered model inputs consumed directly by the spinor DAG."""

    if process_family == _MASSIVE_QUARK_FAMILY:
        source_order = tuple(int(label) for label in ordered_source_labels)
        by_label = {int(leg.label): leg for leg in process.legs}
        ordered_legs = tuple(by_label.get(label) for label in source_order)
        if (
            len(process.legs) != 4
            or sorted(source_order) != [1, 2, 3, 4]
            or any(leg is None for leg in ordered_legs)
            or tuple(
                int(leg.outgoing_pdg or 0)
                for leg in ordered_legs
                if leg is not None
            )
            != (6, 21, 21, -6)
        ):
            raise ValueError(
                "massive-quark spinor parameters require the complete "
                "outgoing t-gluon-gluon-tbar traversal"
            )
        mass = float(model.mass(6))
        width = float(model.width(6))
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(
                "massive-quark spinor parameters require positive top mass"
            )
        if not math.isfinite(width) or width < 0.0:
            raise ValueError(
                "massive-quark spinor parameters require nonnegative top width"
            )
        return (
            ("particle.6.mass", mass),
            ("particle.6.width", width),
        )
    if process_family != _MASSIVE_NEUTRAL_VECTOR_FAMILY:
        return ()
    source_order = tuple(int(label) for label in ordered_source_labels)
    if len(source_order) != len(process.legs) or len(source_order) < 3:
        raise ValueError("massive-vector spinor parameters require a full traversal")
    by_label = {int(leg.label): leg for leg in process.legs}
    quark = by_label.get(source_order[0])
    vector = by_label.get(source_order[-1])
    if quark is None or vector is None:
        raise ValueError("massive-vector spinor traversal references an absent source")
    quark_pdg = int(quark.outgoing_pdg or 0)
    if quark_pdg not in range(1, 6) or int(vector.outgoing_pdg or 0) != 23:
        raise ValueError("massive-vector spinor parameters require outgoing q and Z")
    coupling_provider = getattr(model, "z_fermion_coupling", None)
    if not callable(coupling_provider):
        raise ValueError("the selected model does not expose a Z-fermion coupling")
    coupling = tuple(float(value) for value in coupling_provider(quark_pdg))
    names = runtime_coupling_parameter_names(
        10,
        (quark_pdg, 23, quark_pdg),
        coupling,
        model=model,
    )
    if len(coupling) != 2 or len(names) != 2 or any(name is None for name in names):
        raise ValueError(
            "the Z-fermion spinor graph requires two named chiral couplings"
        )
    return tuple(
        (str(name), value)
        for name, value in zip(names, coupling, strict=True)
        if name is not None
    )


def build_spinor_physics(
    process: CanonicalProcessIR,
    model: Model,
    *,
    process_id: str,
    fixed_color_order: Sequence[int],
    process_family: SpinorProcessFamily = "pure-gluon",
    ordered_source_labels: Sequence[int] | None = None,
) -> dict[str, object]:
    """Describe the singleton aggregate axes evaluated by the spinor graph."""

    order = tuple(int(label) for label in fixed_color_order)
    external_count = len(process.legs)
    if process.color_accuracy != "lc":
        raise ValueError("spinor physics requires LC")
    if process_family == "pure-gluon":
        if (
            external_count not in {4, 5, 6}
            or sorted(order) != list(range(1, external_count + 1))
            or any(leg.pdg != 21 or leg.outgoing_pdg != 21 for leg in process.legs)
        ):
            raise ValueError("pure-gluon spinor physics requires four to six gluons")
    elif process_family == "single-massless-quark-line":
        source_order = tuple(int(label) for label in (ordered_source_labels or ()))
        by_label = {int(leg.label): leg for leg in process.legs}
        ordered_legs = tuple(by_label.get(label) for label in source_order)
        if (
            external_count not in {4, 5, 6}
            or source_order != order
            or sorted(source_order) != list(range(1, external_count + 1))
            or any(leg is None for leg in ordered_legs)
        ):
            raise ValueError(
                "quark-line spinor physics requires two to four gluons and "
                "one complete open-line traversal"
            )
        concrete_legs = tuple(leg for leg in ordered_legs if leg is not None)
        outgoing_pdgs = tuple(int(leg.outgoing_pdg or 0) for leg in concrete_legs)
        if (
            concrete_legs[0].color_role != "fundamental"
            or concrete_legs[0].wavefunction_family != "fermion"
            or concrete_legs[-1].color_role != "antifundamental"
            or concrete_legs[-1].wavefunction_family != "fermion"
            or outgoing_pdgs[0] not in range(1, 6)
            or outgoing_pdgs[-1] != -outgoing_pdgs[0]
            or model.mass(outgoing_pdgs[0]) != 0.0
            or model.mass(outgoing_pdgs[-1]) != 0.0
            or any(
                leg.color_role != "adjoint"
                or leg.wavefunction_family != "vector"
                or pdg != 21
                for leg, pdg in zip(
                    concrete_legs[1:-1],
                    outgoing_pdgs[1:-1],
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "quark-line spinor physics requires outgoing q, two to four "
                "gluons, and the matching outgoing antiquark"
            )
    elif process_family == _MASSIVE_QUARK_FAMILY:
        source_order = tuple(int(label) for label in (ordered_source_labels or ()))
        by_label = {int(leg.label): leg for leg in process.legs}
        ordered_legs = tuple(by_label.get(label) for label in source_order)
        if (
            external_count != 4
            or source_order != order
            or sorted(source_order) != [1, 2, 3, 4]
            or tuple(int(leg.pdg) for leg in process.legs) != (21, 21, 6, -6)
            or tuple(bool(leg.is_initial) for leg in process.legs)
            != (True, True, False, False)
            or any(leg is None for leg in ordered_legs)
        ):
            raise ValueError(
                "massive-quark spinor physics requires g g > t tbar and one "
                "complete fixed open-line traversal"
            )
        concrete_legs = tuple(leg for leg in ordered_legs if leg is not None)
        outgoing_pdgs = tuple(int(leg.outgoing_pdg or 0) for leg in concrete_legs)
        if (
            outgoing_pdgs != (6, 21, 21, -6)
            or concrete_legs[0].color_role != "fundamental"
            or concrete_legs[0].wavefunction_family != "fermion"
            or concrete_legs[-1].color_role != "antifundamental"
            or concrete_legs[-1].wavefunction_family != "fermion"
            or any(
                leg.color_role != "adjoint"
                or leg.wavefunction_family != "vector"
                for leg in concrete_legs[1:-1]
            )
            or not math.isfinite(float(model.mass(6)))
            or model.mass(6) <= 0.0
            or not math.isfinite(float(model.width(6)))
            or model.width(6) < 0.0
        ):
            raise ValueError(
                "massive-quark spinor physics requires outgoing t, two gluons, "
                "and the matching outgoing tbar"
            )
    elif process_family == _MASSIVE_NEUTRAL_VECTOR_FAMILY:
        source_order = tuple(int(label) for label in (ordered_source_labels or ()))
        by_label = {int(leg.label): leg for leg in process.legs}
        ordered_legs = tuple(by_label.get(label) for label in source_order)
        if (
            external_count not in {3, 4, 5}
            or len(order) != external_count - 1
            or sorted(source_order) != list(range(1, external_count + 1))
            or source_order[:-1] != order
            or any(leg is None for leg in ordered_legs)
        ):
            raise ValueError(
                "massive-vector spinor physics requires a complete open-line "
                "color word followed by one singlet source"
            )
        concrete_legs = tuple(leg for leg in ordered_legs if leg is not None)
        outgoing_pdgs = tuple(int(leg.outgoing_pdg or 0) for leg in concrete_legs)
        if (
            concrete_legs[0].color_role != "fundamental"
            or concrete_legs[0].wavefunction_family != "fermion"
            or concrete_legs[-2].color_role != "antifundamental"
            or concrete_legs[-2].wavefunction_family != "fermion"
            or outgoing_pdgs[0] not in range(1, 6)
            or outgoing_pdgs[-2] != -outgoing_pdgs[0]
            or model.mass(outgoing_pdgs[0]) != 0.0
            or model.mass(outgoing_pdgs[-2]) != 0.0
            or any(
                leg.color_role != "adjoint"
                or leg.wavefunction_family != "vector"
                or pdg != 21
                for leg, pdg in zip(
                    concrete_legs[1:-2],
                    outgoing_pdgs[1:-2],
                    strict=True,
                )
            )
            or concrete_legs[-1].color_role != "singlet"
            or concrete_legs[-1].wavefunction_family != "vector"
            or outgoing_pdgs[-1] != 23
            or model.mass(23) <= 0.0
        ):
            raise ValueError(
                "massive-vector spinor physics requires outgoing q, zero to two "
                "gluons, matching outgoing antiquark, and one massive Z"
            )
    else:
        raise ValueError(f"unsupported spinor process family: {process_family}")
    normalization = dict(
        model.runtime_normalization_payload(SimpleNamespace(process=process))
    )
    graph_parameters = spinor_graph_parameters(
        process,
        model,
        process_family=process_family,
        ordered_source_labels=tuple(
            int(label) for label in (ordered_source_labels or order)
        ),
    )
    if process_family == _MASSIVE_NEUTRAL_VECTOR_FAMILY:
        normalization["couplings_in_stage_evaluators"] = True
        normalization["coupling_policy"] = (
            "spinor DAG with local chiral couplings and one global reduction factor"
        )
    else:
        normalization["couplings_in_stage_evaluators"] = False
        normalization["coupling_policy"] = (
            "coupling-stripped spinor DAG with one global reduction factor"
        )
    parameters = [
        {
            "name": str(name),
            "kind": "normalization",
            "default_real": float(value),
            "default_imaginary": 0.0,
            "mutable": True,
        }
        for name, value in sorted(
            model.runtime_normalization_parameter_defaults().items()
        )
    ]
    if process_family == _MASSIVE_NEUTRAL_VECTOR_FAMILY:
        parameters.append(
            {
                "name": "particle.23.mass",
                "kind": "mass",
                "default_real": float(model.mass(23)),
                "default_imaginary": 0.0,
                "mutable": True,
            }
        )
        parameters.extend(
            {
                "name": name,
                "kind": "coupling",
                "default_real": value,
                "default_imaginary": 0.0,
                "mutable": True,
            }
            for name, value in graph_parameters
        )
    elif process_family == _MASSIVE_QUARK_FAMILY:
        parameters.extend(
            {
                "name": name,
                "kind": kind,
                "default_real": value,
                "default_imaginary": 0.0,
                "mutable": True,
            }
            for (name, value), kind in zip(
                graph_parameters,
                ("mass", "width"),
                strict=True,
            )
        )
    color_id = "flow:" + ",".join(str(label) for label in order)
    spinor_extension: dict[str, object] = {
        "helicity_axis": "always-summed-aggregate",
        "fixed_color_order": list(order),
    }
    if process_family != "pure-gluon":
        spinor_extension.update(
            {
                "process_family": process_family,
                "ordered_source_labels": list(source_order),
            }
        )
    if graph_parameters:
        spinor_extension["spinor_parameter_names"] = [
            name for name, _value in graph_parameters
        ]
    return {
        "schema_version": 1,
        "kind": "pyamplicol-resolved-physics",
        "process_id": process_id,
        "process": process.process,
        "color_accuracy": "lc",
        "coverage": {
            "helicities": "complete",
            "color": "selected",
            "color_kind": "physical-lc-flows",
            "structural_zero_helicity_count": 0,
        },
        "external_particles": [
            {
                "index": index,
                "label": int(leg.label),
                "particle": str(leg.particle),
                "pdg": int(leg.pdg),
                "role": "initial" if leg.is_initial else "final",
                "momentum_slot": index,
                "momentum_components": ["E", "px", "py", "pz"],
            }
            for index, leg in enumerate(process.legs)
        ],
        # This is deliberately an aggregate sentinel, not a physical helicity.
        "helicities": [
            {
                "id": "h:sum",
                "index": 0,
                "values": [0] * external_count,
                "computed": True,
                "structural_zero": False,
                "representative_id": "h:sum",
                "coefficient": 1.0,
            }
        ],
        "color_components": [
            {
                "kind": "lc-flow",
                "id": color_id,
                "index": 0,
                "word": list(order),
                "computed": True,
                "representative_id": color_id,
                "coefficient": 1.0,
            }
        ],
        "reduction": {
            "kind": "lc-diagonal",
            "groups": [
                {
                    # Runtime reduction IDs retain the evaluator's numeric group
                    # identifier even though this lane owns a single aggregate.
                    "id": "reduction:0",
                    "representative_helicity_id": "h:sum",
                    "representative_color_id": color_id,
                    "physical_helicity_ids": ["h:sum"],
                    "physical_color_ids": [color_id],
                }
            ],
        },
        "model_parameters": parameters,
        "selectors": {
            "helicity": False,
            "color_flow": True,
            "contracted_color": False,
        },
        "extensions": {
            "process_key": process.key,
            "normalization": {
                key: normalization[key]
                for key in _NORMALIZATION_EXTENSION_KEYS
                if key in normalization
            },
            "spinor_dag": spinor_extension,
        },
    }


__all__ = [
    "SpinorProcessFamily",
    "build_spinor_physics",
    "spinor_graph_parameters",
]
