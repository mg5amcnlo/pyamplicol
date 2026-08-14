// SPDX-License-Identifier: 0BSD

//! Fail-closed lowering of authenticated recurrences to spinor DAG v3.

use std::collections::{BTreeMap, BTreeSet};

use super::template::{
    CurrentOrientation, EvaluatorCallableKind, EvaluatorContractKind, MISSING_U32,
    OutputFactorSource, ParameterKind, ParameterValueType, ParticleStatistics,
    ValidatedRecurrenceTemplateInput,
};
use super::{
    AuthenticatedRecurrenceBuilderInput, CurrentSourceBinding, DirectExecutorRole,
    ExactComplexRational, ExactRational, PreparedDirectExecutorCatalog, RecurrenceProgram,
    RecurrenceStrategy, SourceStateAssignment,
};
use crate::spinor::{
    BispinorExpression, BivectorExpression, DiracExpression, LinearWeylExpression, SpinorChirality,
    SpinorDagBuilder, SpinorDagPayloadV3, SpinorKinematicScalar, SpinorPreparedParameterBinding,
    SpinorSourceInputBinding, SpinorSourceInputKind, bispinor_dot_expression, bispinor_scale,
    bispinor_sum, bivector_scale, bivector_sum, bivector_vector_expression,
    bivector_wedge_expression, dirac_bilinear, dirac_half_scale, dirac_propagator_numerator,
    dirac_scalar_expression, dirac_scale, dirac_sum, dirac_vector_expression,
    external_polarization_expression, external_polarization_expression_with_reference,
    linear_weyl_scale, linear_weyl_sum, massive_dirac_propagator_denominator,
    massive_dirac_source_expression, massive_vector_longitudinal_polarization_expression,
    massive_vector_polarization_expression, massive_vector_propagator_expression,
    quark_vector_weyl_bilinear, quark_vector_weyl_numerator_with_momentum,
    signed_momentum_expression, three_vector_bispinor_expression, weyl_pair_vector_expression,
};
use crate::{RusticolError, RusticolResult};

const SCALAR_PRODUCT_TEMPLATE: &str = "rusticol.recurrence-intrinsic.scalar-product.v1";
const SCALAR_PRODUCT_CONTRACT: &str =
    "8c336af6df4d381f8d8aba881fb1c438e81f9698a31362acb38d3ee5c628fa6c";
const SCALAR_SOURCE_PREFIX: &str = "rusticol.source-fill.scalar.v1:";
const CLOSURE_PREFIX: &str = "rusticol.closure-reduce.v1:";
const IDENTITY_FINALIZER: &str = "rusticol.identity-finalize-in-place.v1";

const THREE_VECTOR_TEMPLATE: &str = "rusticol.recurrence-intrinsic.color-ordered-three-vector.v1";
const THREE_VECTOR_CONTRACT: &str =
    "5fcffbd8137bb0bb892c7347693bf865d8a45279f13dcf10f70d93f1b7660beb";
const ANTISYMMETRIC_TENSOR_VECTOR_TEMPLATE: &str =
    "rusticol.recurrence-intrinsic.antisymmetric-tensor-vector.v1";
const ANTISYMMETRIC_TENSOR_VECTOR_CONTRACT: &str =
    "c4ba66d6b6a2a9bc0d0ccdb500ded6fb71fe3be6f8887a2468d86159ca6e2ffe";
const VECTOR_WEDGE_VECTOR_TEMPLATE: &str = "rusticol.recurrence-intrinsic.vector-wedge-vector.v1";
const VECTOR_WEDGE_VECTOR_CONTRACT: &str =
    "484328b1e11d0e294f512e11cc34797fda35d24c3724a83ffda3a7cbf73cb895";
const WEYL_VECTOR_A_TEMPLATE: &str = "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-a.v1";
const WEYL_VECTOR_A_CONTRACT: &str =
    "cefa8f1afe99611314d099742ee08d6014e16d6cf5cb12f06a4c07c82e1df4b2";
const WEYL_VECTOR_B_TEMPLATE: &str = "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-b.v1";
const WEYL_VECTOR_B_CONTRACT: &str =
    "488de507671a00baeb23979e51303f1a77e7c4747b733ee51c95b00705ca393b";
const WEYL_PAIR_VECTOR_A_TEMPLATE: &str = "rusticol.recurrence-intrinsic.weyl-pair-to-vector-a.v1";
const WEYL_PAIR_VECTOR_A_CONTRACT: &str =
    "4ba229a983d630393867793ae53d0a6acb9d503e4767a43e0c804cb3cf43bf7a";
const WEYL_PAIR_VECTOR_B_TEMPLATE: &str = "rusticol.recurrence-intrinsic.weyl-pair-to-vector-b.v1";
const WEYL_PAIR_VECTOR_B_CONTRACT: &str =
    "83760c08f5e0af401e2d9667af182884da31123a94b25304ec7f8d4530bf83c7";
const WEYL_PROPAGATOR_A_TEMPLATE: &str = "rusticol.recurrence-intrinsic.weyl-propagator-a.v1";
const WEYL_PROPAGATOR_A_CONTRACT: &str =
    "2ca86441343f898281b2848144810275e23b01a032e6bedbdaa6bbdd75d22b88";
const WEYL_PROPAGATOR_B_TEMPLATE: &str = "rusticol.recurrence-intrinsic.weyl-propagator-b.v1";
const WEYL_PROPAGATOR_B_CONTRACT: &str =
    "3c0a6569e86eb94d115561e286491eb85d7210480efa2706f9f8ae5cd63f0888";
const VECTOR_PROPAGATOR_TEMPLATE: &str =
    "rusticol.recurrence-intrinsic.vector-propagator-feynman.v1";
const VECTOR_PROPAGATOR_CONTRACT: &str =
    "785a8b23feff16ec18f5f77d5164d1e3da3ef85cce984a7076f367a96b16c5a9";
const VECTOR_SOURCE_PREFIX: &str = "rusticol.source-fill.vector.v1:";
const FERMION_SOURCE_PREFIX: &str = "rusticol.source-fill.fermion.v1:";
const GLOBAL_FLIP_EXTENSION: &str = "helicity-equivalence:global-flip-v1";

const DIRAC_VECTOR_PARTICLE_TEMPLATE: &str =
    "rusticol.recurrence-intrinsic.dirac-vector-to-dirac-particle.v1";
const DIRAC_VECTOR_PARTICLE_CONTRACT: &str =
    "3f4b28f08fba4d08c56d9a1bac9e9ac563a11462feb8f95c7262f27847738661";
const DIRAC_VECTOR_ANTIPARTICLE_TEMPLATE: &str =
    "rusticol.recurrence-intrinsic.dirac-vector-to-dirac-antiparticle.v1";
const DIRAC_VECTOR_ANTIPARTICLE_CONTRACT: &str =
    "46391ed42113ed52b1960215b03fad8470f6542fbc928e56b9d7b426e66ab9ab";
const CHIRAL_DIRAC_VECTOR_PARTICLE_TEMPLATE: &str =
    "rusticol.recurrence-intrinsic.dirac-vector-to-dirac-chiral-particle.v1";
const CHIRAL_DIRAC_VECTOR_PARTICLE_CONTRACT: &str =
    "2287796f888111348c3eda616eb98e3a69c116a1449c29546c89ed871f43a517";
const CHIRAL_DIRAC_VECTOR_ANTIPARTICLE_TEMPLATE: &str =
    "rusticol.recurrence-intrinsic.dirac-vector-to-dirac-chiral-antiparticle.v1";
const CHIRAL_DIRAC_VECTOR_ANTIPARTICLE_CONTRACT: &str =
    "ff5d75dc8549684287b9eb9c801ead3e60d031ea841754e59c3ee3c841050975";
const DIRAC_SCALAR_TEMPLATE: &str = "rusticol.recurrence-intrinsic.dirac-scalar-to-dirac.v1";
const DIRAC_SCALAR_CONTRACT: &str =
    "d9c7dbc51561cdc2b2a7daf3d97ea24283d6c690ae5da2d775e86c80a3b4886f";
const VECTOR_PAIR_SCALAR_TEMPLATE: &str = "rusticol.recurrence-intrinsic.vector-pair-to-scalar.v1";
const VECTOR_PAIR_SCALAR_CONTRACT: &str =
    "261b7f122671c1afc5ce3e430c82eb907cbc9873c91da3dfcbcb2bbaea048ad9";
const MASSIVE_DIRAC_PARTICLE_TEMPLATE: &str =
    "rusticol.recurrence-intrinsic.massive-dirac-propagator-particle.v1";
const MASSIVE_DIRAC_PARTICLE_CONTRACT: &str =
    "d5b795b8fb0c5b487658d9977623e771f03f4a5c2d1cabf2cd955e542c8ccbef";
const MASSIVE_DIRAC_ANTIPARTICLE_TEMPLATE: &str =
    "rusticol.recurrence-intrinsic.massive-dirac-propagator-antiparticle.v1";
const MASSIVE_DIRAC_ANTIPARTICLE_CONTRACT: &str =
    "7174d14153ebd3028b9e963538bb5255468eeb00665f3a2114dd97206bc0a28c";
const MASSIVE_VECTOR_UNITARY_TEMPLATE: &str =
    "rusticol.recurrence-intrinsic.massive-vector-propagator-unitary.v1";
const MASSIVE_VECTOR_UNITARY_CONTRACT: &str =
    "4293b6a7a8a7433fc598e2031a031c353491ba76404fa984f9a803daad9cfb40";
const MASSIVE_SCALAR_TEMPLATE: &str = "rusticol.recurrence-intrinsic.massive-scalar-propagator.v1";
const MASSIVE_SCALAR_CONTRACT: &str =
    "d90a205a4542718e1f253057502ccc3e4e3eab33030323490bbea128a6a81c38";
// SHA-256 of the canonical exact parameter expression `0`.
const ZERO_PARAMETER_EXPRESSION_DIGEST: [u8; 32] = [
    0x5f, 0xec, 0xeb, 0x66, 0xff, 0xc8, 0x6f, 0x38, 0xd9, 0x52, 0x78, 0x6c, 0x6d, 0x69, 0x6c, 0x79,
    0xc2, 0xdb, 0xc2, 0x39, 0xdd, 0x4e, 0x91, 0xb4, 0x67, 0x29, 0xd7, 0x3a, 0x27, 0xfb, 0x57, 0xe9,
];

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!(
        "authenticated recurrence spinor lowering: {}",
        message.into()
    ))
}

/// Lower an authenticated recurrence without consulting particle IDs or
/// process-family labels.
///
/// Eligibility is established solely by the validated one-component source,
/// transition, closure and prepared-direct intrinsic contracts. Anything
/// outside this first executable primitive set is rejected.
pub fn lower_authenticated_recurrence_to_spinor_payload_v3(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    direct: &PreparedDirectExecutorCatalog,
) -> RusticolResult<SpinorDagPayloadV3> {
    let program = authenticated.build()?;
    let templates = authenticated.template();
    let source_count = usize::try_from(authenticated.process().summary().external_leg_count())
        .map_err(|_| invalid("external source count exceeds usize"))?;
    let scalar_shape = program.currents().iter().all(|current| {
        require_scalar_state(templates, current.key().current_state_template_id()).is_ok()
    });
    if scalar_shape {
        lower_scalar_program(
            &program,
            templates,
            direct,
            source_count,
            templates.summary().parameter_count,
        )
    } else {
        lower_qcd_program(
            authenticated,
            &program,
            templates,
            direct,
            source_count,
            templates.summary().parameter_count,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QcdStateKind {
    Scalar {
        mass_parameter_id: u32,
        mass_prepared_slot: Option<u32>,
        width_parameter_id: u32,
        width_prepared_slot: Option<u32>,
    },
    Vector,
    U1SubtractionVector,
    MassiveVector {
        orientation: CurrentOrientation,
        species_string_id: u32,
        mass_parameter_id: u32,
        mass_prepared_slot: u32,
        width_parameter_id: u32,
        width_prepared_slot: Option<u32>,
    },
    Bivector,
    Weyl {
        chirality: SpinorChirality,
        orientation: CurrentOrientation,
        mass_parameter_id: u32,
    },
    Dirac {
        orientation: CurrentOrientation,
        width_parameter_id: u32,
        width_prepared_slot: Option<u32>,
    },
}

#[derive(Clone, Debug)]
enum QcdCurrent {
    Scalar(u32),
    Vector(BispinorExpression),
    Bivector(BivectorExpression),
    Weyl {
        chirality: SpinorChirality,
        orientation: CurrentOrientation,
        /// `None` is the deliberately unmaterialized final vertex of a Weyl
        /// line. Its action is fused into the authenticated closure.
        value: Option<LinearWeylExpression>,
    },
    Dirac {
        orientation: CurrentOrientation,
        value: DiracExpression,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QcdContributionKind {
    ThreeVector,
    AntisymmetricTensorVector,
    VectorWedgeVector,
    WeylVector(SpinorChirality),
    WeylPairVector(SpinorChirality),
    DiracVector(CurrentOrientation),
    ChiralDiracVector(CurrentOrientation),
    DiracScalar,
    VectorPairScalar,
}

#[derive(Clone, Debug)]
struct QcdSourceLayout {
    input_kinds: Vec<SpinorSourceInputKind>,
    massive_sources: Vec<u16>,
    vector_sources: Vec<u16>,
    source_parameter_slots: Vec<u32>,
    dirac_mass_prepared_slot: Option<u32>,
    dirac_width_prepared_slot: Option<u32>,
}

fn lower_qcd_program(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    source_count: usize,
    prepared_parameter_count: u32,
) -> RusticolResult<SpinorDagPayloadV3> {
    if program.strategy() != RecurrenceStrategy::TopologyReplay {
        return Err(invalid("the QCD slice requires topology replay"));
    }
    if source_count < 3 {
        return Err(invalid("the QCD slice requires at least three sources"));
    }
    let representative_signs = qcd_representative_signs(program, source_count)?;
    let source_layout = qcd_source_layout(program, templates, source_count)?;
    let prepared_slots = qcd_parameter_slots(
        program,
        templates,
        direct,
        &source_layout.source_parameter_slots,
        source_layout.dirac_mass_prepared_slot,
        source_layout.dirac_width_prepared_slot,
    )?;
    let dense_parameter_slots = prepared_slots
        .iter()
        .copied()
        .enumerate()
        .map(|(dense, prepared)| {
            u16::try_from(dense)
                .map(|dense| (prepared, dense))
                .map_err(|_| invalid("dense graph parameter index exceeds u16"))
        })
        .collect::<RusticolResult<BTreeMap<_, _>>>()?;
    let momentum_count = u16::try_from(source_count)
        .map_err(|_| invalid("external source count exceeds the spinor DAG domain"))?;
    let parameter_count = u16::try_from(prepared_slots.len())
        .map_err(|_| invalid("dense graph parameter count exceeds u16"))?;
    let mut builder = SpinorDagBuilder::new_with_parameters_and_massive_sources(
        momentum_count,
        parameter_count,
        &source_layout.massive_sources,
    )?;
    let mut vector_reference_atoms = BTreeMap::new();
    if source_layout.dirac_mass_prepared_slot.is_some() {
        for source in source_layout.vector_sources.iter().copied() {
            vector_reference_atoms.insert(source, builder.temporal_reference_atom(source)?);
        }
    }
    let mut current_values = vec![None; program.currents().len()];

    for current in program.currents() {
        let index = usize::try_from(current.id())
            .map_err(|_| invalid("semantic current ID exceeds usize"))?;
        if index >= current_values.len() {
            return Err(invalid("semantic current ID is out of bounds"));
        }
        let state = qcd_state_kind(templates, current.key().current_state_template_id())?;
        let value = if current.is_source() {
            lower_qcd_source(
                current,
                state,
                templates,
                direct,
                &dense_parameter_slots,
                &vector_reference_atoms,
                &mut builder,
            )?
        } else {
            lower_qcd_current(
                current,
                state,
                program,
                templates,
                direct,
                &dense_parameter_slots,
                source_layout.dirac_mass_prepared_slot,
                source_layout.dirac_width_prepared_slot,
                &representative_signs,
                &current_values,
                &mut builder,
            )?
        };
        if current_values[index].replace(value).is_some() {
            return Err(invalid(format!(
                "semantic current {} was lowered more than once",
                current.id()
            )));
        }
    }
    let current_values = current_values
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| invalid("not every semantic current was lowered"))?;
    let destination_nodes = lower_qcd_closures(
        program,
        templates,
        direct,
        &dense_parameter_slots,
        &representative_signs,
        &current_values,
        &mut builder,
    )?;
    add_qcd_roots(
        authenticated,
        program,
        source_count,
        &destination_nodes,
        &mut builder,
    )?;
    let dag = builder.finish()?;

    let (permutation, replay_signs) = match program.replay_targets() {
        [] => (
            (0..source_count)
                .map(|slot| u32::try_from(slot).map_err(|_| invalid("source slot exceeds u32")))
                .collect::<RusticolResult<Vec<_>>>()?,
            vec![1_i32; source_count],
        ),
        [replay] => (
            replay.source_slot_permutation().to_vec(),
            replay.source_momentum_signs().to_vec(),
        ),
        _ => {
            return Err(invalid(
                "one fixed-flow QCD payload supports at most one replay target",
            ));
        }
    };
    if permutation.len() != source_count || replay_signs.len() != source_count {
        return Err(invalid("QCD source replay mapping has the wrong width"));
    }
    let source_inputs = permutation
        .into_iter()
        .zip(replay_signs)
        .zip(representative_signs)
        .enumerate()
        .map(
            |(graph_slot, ((public_slot, replay_sign), representative_sign))| {
                let public_slot = u16::try_from(public_slot)
                    .map_err(|_| invalid("public source slot exceeds u16"))?;
                let momentum_sign = representative_sign
                    .checked_mul(replay_sign)
                    .and_then(|sign| i8::try_from(sign).ok())
                    .ok_or_else(|| invalid("source momentum sign exceeds i8"))?;
                SpinorSourceInputBinding::new(
                    public_slot,
                    momentum_sign,
                    source_layout.input_kinds[graph_slot],
                )
            },
        )
        .collect::<RusticolResult<Vec<_>>>()?;
    let parameter_bindings = prepared_slots
        .into_iter()
        .map(SpinorPreparedParameterBinding::new)
        .collect();
    SpinorDagPayloadV3::new(
        dag,
        source_inputs,
        prepared_parameter_count,
        parameter_bindings,
    )
}

fn qcd_state_kind(
    templates: &ValidatedRecurrenceTemplateInput,
    state_id: u32,
) -> RusticolResult<QcdStateKind> {
    let state = templates
        .input()
        .current_states
        .get(state_id as usize)
        .ok_or_else(|| invalid(format!("current-state template {state_id} is absent")))?;
    if state.id != state_id {
        return Err(invalid(format!(
            "current-state template {state_id} has a noncanonical ID"
        )));
    }
    let basis = template_string(templates, state.basis_string_id, "current-state basis")?;
    let auxiliary_kind = if state.auxiliary_kind_string_id == MISSING_U32 {
        None
    } else {
        Some(template_string(
            templates,
            state.auxiliary_kind_string_id,
            "current-state auxiliary kind",
        )?)
    };
    let statistics = ParticleStatistics::try_from(state.statistics)?;
    let orientation = CurrentOrientation::try_from(state.orientation)?;
    let kind = if is_u1_subtraction_vector_state(state, basis, auxiliary_kind, templates)? {
        QcdStateKind::U1SubtractionVector
    } else {
        match (
            statistics,
            basis,
            state.dimension,
            state.chirality,
            orientation,
            auxiliary_kind,
        ) {
            (
                ParticleStatistics::Boson,
                "scalar",
                1,
                0,
                CurrentOrientation::SelfConjugate,
                None,
            ) if state.particle_id == state.anti_particle_id => {
                let mass_prepared_slot = optional_prepared_real_parameter_slot(
                    templates,
                    state.mass_parameter_id,
                    "scalar state mass",
                )?;
                let width_prepared_slot = optional_prepared_real_parameter_slot(
                    templates,
                    state.width_parameter_id,
                    "scalar state width",
                )?;
                if width_prepared_slot.is_some() && mass_prepared_slot.is_none() {
                    return Err(invalid(format!(
                        "scalar current-state template {state_id} owns a width without a mass"
                    )));
                }
                QcdStateKind::Scalar {
                    mass_parameter_id: state.mass_parameter_id,
                    mass_prepared_slot,
                    width_parameter_id: state.width_parameter_id,
                    width_prepared_slot,
                }
            }
            (
                ParticleStatistics::Boson,
                "lorentz-vector",
                4,
                0,
                CurrentOrientation::SelfConjugate,
                None,
            ) if state.mass_parameter_id == MISSING_U32
                && state.width_parameter_id == MISSING_U32 =>
            {
                QcdStateKind::Vector
            }
            (ParticleStatistics::Boson, "lorentz-vector", 4, 0, orientation, None)
                if state.mass_parameter_id != MISSING_U32 =>
            {
                let orientation_is_authenticated = match orientation {
                    CurrentOrientation::SelfConjugate => {
                        state.particle_id == state.anti_particle_id
                    }
                    CurrentOrientation::Particle | CurrentOrientation::Antiparticle => {
                        state.particle_id != state.anti_particle_id
                            && massive_vector_conjugate_state_count(
                                &templates.input().current_states,
                                state,
                            ) == 1
                    }
                };
                if !orientation_is_authenticated {
                    return Err(invalid(format!(
                        "massive-vector current-state template {state_id} has no unique authenticated charge conjugate"
                    )));
                }
                let mass_prepared_slot = prepared_real_parameter_slot(
                    templates,
                    state.mass_parameter_id,
                    "massive-vector state mass",
                )?;
                let width_prepared_slot = optional_prepared_real_parameter_slot(
                    templates,
                    state.width_parameter_id,
                    "massive-vector state width",
                )?;
                QcdStateKind::MassiveVector {
                    orientation,
                    species_string_id: state.species_string_id,
                    mass_parameter_id: state.mass_parameter_id,
                    mass_prepared_slot,
                    width_parameter_id: state.width_parameter_id,
                    width_prepared_slot,
                }
            }
            (
                ParticleStatistics::Boson,
                "auxiliary:antisymmetric-tensor",
                6,
                0,
                CurrentOrientation::SelfConjugate,
                Some("antisymmetric-tensor"),
            ) if state.mass_parameter_id == MISSING_U32
                && state.width_parameter_id == MISSING_U32 =>
            {
                QcdStateKind::Bivector
            }
            (
                ParticleStatistics::Fermion,
                "weyl-chiral",
                2,
                chirality @ (-1 | 1),
                orientation,
                None,
            ) if matches!(
                orientation,
                CurrentOrientation::Particle | CurrentOrientation::Antiparticle
            ) && state.width_parameter_id == MISSING_U32 =>
            {
                validate_optional_exact_zero_mass_owner(
                    templates,
                    state.mass_parameter_id,
                    "massless Weyl state",
                )?;
                QcdStateKind::Weyl {
                    chirality: if chirality == 1 {
                        SpinorChirality::Positive
                    } else {
                        SpinorChirality::Negative
                    },
                    orientation,
                    mass_parameter_id: state.mass_parameter_id,
                }
            }
            (ParticleStatistics::Fermion, "dirac", 4, 0, orientation, None)
                if matches!(
                    orientation,
                    CurrentOrientation::Particle | CurrentOrientation::Antiparticle
                ) && state.mass_parameter_id != MISSING_U32 =>
            {
                QcdStateKind::Dirac {
                    orientation,
                    width_parameter_id: state.width_parameter_id,
                    width_prepared_slot: optional_prepared_real_parameter_slot(
                        templates,
                        state.width_parameter_id,
                        "massive Dirac state width",
                    )?,
                }
            }
            _ => {
                return Err(invalid(format!(
                    "current-state template {state_id} is outside the scalar/vector/bivector/fermion QCD slice"
                )));
            }
        }
    };
    let ordering = template_u32_sequence(
        templates,
        state.tensor_ordering_sequence_id,
        "QCD current tensor ordering",
    )?;
    if ordering.len() != state.dimension as usize {
        return Err(invalid(format!(
            "current-state template {state_id} has the wrong tensor-ordering width"
        )));
    }
    for (component, string_id) in ordering.iter().copied().enumerate() {
        let actual = template_string(templates, string_id, "QCD current tensor component")?;
        let expected = format!("{basis}:c{component}");
        if actual != expected {
            return Err(invalid(format!(
                "current-state template {state_id} uses unsupported tensor component {actual:?} at slot {component}"
            )));
        }
    }
    Ok(kind)
}

fn is_u1_subtraction_vector_state(
    state: &super::template::CurrentStateRow,
    basis: &str,
    auxiliary_kind: Option<&str>,
    templates: &ValidatedRecurrenceTemplateInput,
) -> RusticolResult<bool> {
    Ok(
        ParticleStatistics::try_from(state.statistics)? == ParticleStatistics::Boson
            && CurrentOrientation::try_from(state.orientation)?
                == CurrentOrientation::SelfConjugate
            && state.particle_id == state.anti_particle_id
            && state.color_representation == 1
            && basis == "auxiliary:u1-subtraction-color-flow-vector"
            && state.dimension == 4
            && state.chirality == 0
            && template_string(
                templates,
                state.lc_color_shape_string_id,
                "U(1)-subtraction vector LC color shape",
            )? == "singlet-forest"
            && auxiliary_kind == Some("u1-subtraction-color-flow-vector")
            && state.mass_parameter_id == MISSING_U32
            && state.width_parameter_id == MISSING_U32,
    )
}

fn massive_vector_conjugate_state_count(
    states: &[super::template::CurrentStateRow],
    state: &super::template::CurrentStateRow,
) -> usize {
    states
        .iter()
        .filter(|candidate| massive_vector_states_are_mutually_conjugate(state, candidate))
        .count()
}

fn massive_vector_states_are_mutually_conjugate(
    state: &super::template::CurrentStateRow,
    candidate: &super::template::CurrentStateRow,
) -> bool {
    let opposite_orientation = match CurrentOrientation::try_from(state.orientation) {
        Ok(CurrentOrientation::Particle) => CurrentOrientation::Antiparticle,
        Ok(CurrentOrientation::Antiparticle) => CurrentOrientation::Particle,
        Ok(CurrentOrientation::SelfConjugate) | Err(_) => return false,
    };
    candidate.id != state.id
        && candidate.particle_id == state.anti_particle_id
        && candidate.anti_particle_id == state.particle_id
        && candidate.species_string_id == state.species_string_id
        && candidate.orientation == opposite_orientation as u8
        && candidate.statistics == state.statistics
        && candidate.color_representation == state.color_representation
        && candidate.basis_string_id == state.basis_string_id
        && candidate.tensor_ordering_sequence_id == state.tensor_ordering_sequence_id
        && candidate.dimension == state.dimension
        && candidate.chirality == state.chirality
        && candidate.lc_color_shape_string_id == state.lc_color_shape_string_id
        && candidate.auxiliary_kind_string_id == state.auxiliary_kind_string_id
        && candidate.mass_parameter_id == state.mass_parameter_id
        && candidate.width_parameter_id == state.width_parameter_id
}

fn qcd_representative_signs(
    program: &RecurrenceProgram,
    source_count: usize,
) -> RusticolResult<Vec<i32>> {
    let mut signs = vec![None; source_count];
    for current in program
        .currents()
        .iter()
        .filter(|current| current.is_source())
    {
        let [support] = current.key().support_source_slots() else {
            return Err(invalid(format!(
                "source current {} has non-singleton support",
                current.id()
            )));
        };
        let [momentum] = current.key().momentum().terms() else {
            return Err(invalid(format!(
                "source current {} has non-elementary momentum",
                current.id()
            )));
        };
        if momentum.source_slot != *support || !matches!(momentum.coefficient, -1 | 1) {
            return Err(invalid(format!(
                "source current {} has an invalid signed momentum binding",
                current.id()
            )));
        }
        let slot = usize::try_from(*support).map_err(|_| invalid("source slot exceeds usize"))?;
        let destination = signs
            .get_mut(slot)
            .ok_or_else(|| invalid("source slot lies outside the public source domain"))?;
        if destination.is_some_and(|previous| previous != momentum.coefficient) {
            return Err(invalid(format!(
                "source slot {support} has inconsistent representative momentum signs"
            )));
        }
        *destination = Some(momentum.coefficient);
    }
    signs
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| invalid("source currents do not cover every graph source"))
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct DiracSourceEndpoint {
    source_slot: u32,
    state_id: u32,
    particle_id: i32,
    anti_particle_id: i32,
    species_string_id: u32,
    mass_prepared_slot: u32,
    width_parameter_id: u32,
    width_prepared_slot: Option<u32>,
}

fn dirac_source_endpoints_are_mutually_conjugate(
    particle: &DiracSourceEndpoint,
    antiparticle: &DiracSourceEndpoint,
) -> bool {
    particle.source_slot != antiparticle.source_slot
        && particle.particle_id == antiparticle.anti_particle_id
        && particle.anti_particle_id == antiparticle.particle_id
        && particle.species_string_id == antiparticle.species_string_id
        && particle.mass_prepared_slot == antiparticle.mass_prepared_slot
        && particle.width_parameter_id == antiparticle.width_parameter_id
        && particle.width_prepared_slot == antiparticle.width_prepared_slot
}

fn qcd_source_layout(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    source_count: usize,
) -> RusticolResult<QcdSourceLayout> {
    let mut input_kinds = vec![None; source_count];
    let mut massive_sources = BTreeSet::new();
    let mut vector_sources = BTreeSet::new();
    let mut source_parameter_slots = BTreeSet::new();
    let mut particle_endpoints = BTreeSet::new();
    let mut antiparticle_endpoints = BTreeSet::new();

    for current in program
        .currents()
        .iter()
        .filter(|current| current.is_source())
    {
        let [source_slot] = current.key().support_source_slots() else {
            return Err(invalid(format!(
                "source current {} has non-singleton support",
                current.id()
            )));
        };
        let slot =
            usize::try_from(*source_slot).map_err(|_| invalid("source slot exceeds usize"))?;
        let destination = input_kinds
            .get_mut(slot)
            .ok_or_else(|| invalid("source slot lies outside the graph source domain"))?;
        let state_id = current.key().current_state_template_id();
        let state = templates
            .input()
            .current_states
            .get(state_id as usize)
            .ok_or_else(|| invalid(format!("current-state template {state_id} is absent")))?;
        let source = qcd_source_row(current, templates)?;
        let (kind, endpoint) = match qcd_state_kind(templates, state_id)? {
            QcdStateKind::Scalar {
                mass_parameter_id,
                mass_prepared_slot,
                width_parameter_id,
                width_prepared_slot,
            } => {
                if mass_parameter_id == MISSING_U32
                    || mass_prepared_slot.is_none()
                    || source.mass_parameter_id != mass_parameter_id
                    || source.width_parameter_id != width_parameter_id
                {
                    return Err(invalid(format!(
                        "massive scalar source template {} does not own its current-state mass binding",
                        source.id
                    )));
                }
                // Authenticate the explicit scalar mass owner even though the
                // scalar wavefunction itself is the unit expression.  Its
                // momentum must be decomposed when it enters a Dirac
                // propagator's signed support.
                let source_state = qcd_state_kind(templates, source.state_template_id)?;
                if source_state
                    != (QcdStateKind::Scalar {
                        mass_parameter_id,
                        mass_prepared_slot,
                        width_parameter_id,
                        width_prepared_slot,
                    })
                {
                    return Err(invalid(format!(
                        "scalar source template {} disagrees with its authenticated state",
                        source.id
                    )));
                }
                massive_sources.insert(
                    u16::try_from(*source_slot)
                        .map_err(|_| invalid("massive scalar source slot exceeds u16"))?,
                );
                (SpinorSourceInputKind::MassiveSpinorPair, None)
            }
            QcdStateKind::Dirac {
                orientation,
                width_parameter_id,
                width_prepared_slot,
            } => {
                if source.mass_parameter_id != state.mass_parameter_id
                    || source.width_parameter_id != width_parameter_id
                {
                    return Err(invalid(format!(
                        "massive Dirac source template {} disagrees with its current-state mass binding",
                        source.id
                    )));
                }
                let mass_prepared_slot = prepared_real_parameter_slot(
                    templates,
                    source.mass_parameter_id,
                    "massive Dirac source mass",
                )?;
                massive_sources.insert(
                    u16::try_from(*source_slot)
                        .map_err(|_| invalid("massive Dirac source slot exceeds u16"))?,
                );
                (
                    SpinorSourceInputKind::MassiveSpinorPair,
                    Some((
                        orientation,
                        DiracSourceEndpoint {
                            source_slot: *source_slot,
                            state_id,
                            particle_id: state.particle_id,
                            anti_particle_id: state.anti_particle_id,
                            species_string_id: state.species_string_id,
                            mass_prepared_slot,
                            width_parameter_id,
                            width_prepared_slot,
                        },
                    )),
                )
            }
            QcdStateKind::Vector => {
                vector_sources.insert(
                    u16::try_from(*source_slot)
                        .map_err(|_| invalid("vector source slot exceeds u16"))?,
                );
                (SpinorSourceInputKind::NullSpinor, None)
            }
            QcdStateKind::U1SubtractionVector => {
                return Err(invalid(
                    "an auxiliary U(1)-subtraction vector cannot be a QCD source",
                ));
            }
            QcdStateKind::MassiveVector {
                orientation,
                species_string_id,
                mass_parameter_id,
                mass_prepared_slot,
                width_parameter_id,
                width_prepared_slot,
            } => {
                let source_state = qcd_state_kind(templates, source.state_template_id)?;
                if source_state
                    != (QcdStateKind::MassiveVector {
                        orientation,
                        species_string_id,
                        mass_parameter_id,
                        mass_prepared_slot,
                        width_parameter_id,
                        width_prepared_slot,
                    })
                    || source.mass_parameter_id != mass_parameter_id
                    || source.width_parameter_id != width_parameter_id
                {
                    return Err(invalid(format!(
                        "massive-vector source template {} disagrees with its authenticated oriented state",
                        source.id
                    )));
                }
                if source.helicity == 0 && source.spin_state == 0 {
                    source_parameter_slots.insert(mass_prepared_slot);
                }
                massive_sources.insert(
                    u16::try_from(*source_slot)
                        .map_err(|_| invalid("massive-vector source slot exceeds u16"))?,
                );
                (SpinorSourceInputKind::MassiveSpinorPair, None)
            }
            QcdStateKind::Weyl { .. } => (SpinorSourceInputKind::NullSpinor, None),
            QcdStateKind::Bivector => {
                return Err(invalid(
                    "an auxiliary antisymmetric tensor cannot be a QCD source",
                ));
            }
        };
        if destination.is_some_and(|previous| previous != kind) {
            return Err(invalid(format!(
                "graph source slot {source_slot} has incompatible source representations"
            )));
        }
        *destination = Some(kind);
        if let Some((orientation, endpoint)) = endpoint {
            match orientation {
                CurrentOrientation::Particle => {
                    particle_endpoints.insert(endpoint);
                }
                CurrentOrientation::Antiparticle => {
                    antiparticle_endpoints.insert(endpoint);
                }
                CurrentOrientation::SelfConjugate => unreachable!("Dirac state checked above"),
            }
        }
    }

    let input_kinds = input_kinds
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| invalid("QCD source currents do not cover every graph source"))?;
    let particle_endpoints = particle_endpoints.into_iter().collect::<Vec<_>>();
    let antiparticle_endpoints = antiparticle_endpoints.into_iter().collect::<Vec<_>>();
    let dirac_owners = match (
        particle_endpoints.as_slice(),
        antiparticle_endpoints.as_slice(),
    ) {
        ([], []) => None,
        ([particle], [antiparticle])
            if dirac_source_endpoints_are_mutually_conjugate(particle, antiparticle) =>
        {
            Some((particle.mass_prepared_slot, particle.width_prepared_slot))
        }
        _ => {
            return Err(invalid(
                "the massive Dirac slice requires exactly one mutually conjugate particle/antiparticle source pair with authenticated mass/width ownership",
            ));
        }
    };
    let (dirac_mass_prepared_slot, dirac_width_prepared_slot) = dirac_owners
        .map(|(mass, width)| (Some(mass), width))
        .unwrap_or((None, None));
    Ok(QcdSourceLayout {
        input_kinds,
        massive_sources: massive_sources.into_iter().collect(),
        vector_sources: vector_sources.into_iter().collect(),
        source_parameter_slots: source_parameter_slots.into_iter().collect(),
        dirac_mass_prepared_slot,
        dirac_width_prepared_slot,
    })
}

fn qcd_source_row<'a>(
    current: &super::RecurrenceCurrent,
    templates: &'a ValidatedRecurrenceTemplateInput,
) -> RusticolResult<&'a super::template::SourceRow> {
    let source_template_id = match current.key().source_binding() {
        CurrentSourceBinding::FixedTemplate(id) => *id,
        _ => return Err(invalid("QCD source does not use one fixed template")),
    };
    let source = templates
        .input()
        .sources
        .get(source_template_id as usize)
        .ok_or_else(|| invalid(format!("source template {source_template_id} is absent")))?;
    if source.id != source_template_id {
        return Err(invalid(format!(
            "source template {source_template_id} has a noncanonical ID"
        )));
    }
    Ok(source)
}

fn prepared_real_parameter_slot(
    templates: &ValidatedRecurrenceTemplateInput,
    parameter_id: u32,
    label: &str,
) -> RusticolResult<u32> {
    if parameter_id == MISSING_U32 {
        return Err(invalid(format!("{label} parameter is missing")));
    }
    let parameter = templates
        .input()
        .parameters
        .get(parameter_id as usize)
        .ok_or_else(|| {
            invalid(format!(
                "{label} parameter template {parameter_id} is absent"
            ))
        })?;
    if parameter.id != parameter_id
        || ParameterValueType::try_from(parameter.value_type)? != ParameterValueType::Real
        || parameter.prepared_parameter_id == MISSING_U32
    {
        return Err(invalid(format!(
            "{label} parameter template {parameter_id} is not an authenticated prepared real slot"
        )));
    }
    Ok(parameter.prepared_parameter_id)
}

fn optional_prepared_real_parameter_slot(
    templates: &ValidatedRecurrenceTemplateInput,
    parameter_id: u32,
    label: &str,
) -> RusticolResult<Option<u32>> {
    if parameter_id == MISSING_U32 {
        Ok(None)
    } else {
        prepared_real_parameter_slot(templates, parameter_id, label).map(Some)
    }
}

fn validate_optional_exact_zero_mass_owner(
    templates: &ValidatedRecurrenceTemplateInput,
    parameter_id: u32,
    label: &str,
) -> RusticolResult<()> {
    if parameter_id == MISSING_U32 {
        return Ok(());
    }
    let parameter = templates
        .input()
        .parameters
        .get(parameter_id as usize)
        .ok_or_else(|| {
            invalid(format!(
                "{label} parameter template {parameter_id} is absent"
            ))
        })?;
    let expression_digest = templates
        .input()
        .digest_catalog
        .get(parameter.exact_expression_digest_id as usize)
        .ok_or_else(|| invalid(format!("{label} exact-zero expression digest is absent")))?;
    if parameter.id != parameter_id
        || !parameter_row_is_immutable_exact_zero(parameter, expression_digest)?
    {
        return Err(invalid(format!(
            "{label} is neither absent nor an authenticated immutable exact-zero real parameter"
        )));
    }
    match ParameterKind::try_from(parameter.kind)? {
        ParameterKind::Derived => {
            validate_derived_zero_parameter_evaluator(templates, parameter, label)?;
            if parameter.default_factor_id != MISSING_U32
                && template_exact_factor(templates, parameter.default_factor_id, label)?
                    != ExactComplexRational::ZERO
            {
                return Err(invalid(format!("{label} has a nonzero default")));
            }
        }
        ParameterKind::Constant => {
            if parameter.default_factor_id == MISSING_U32
                || template_exact_factor(templates, parameter.default_factor_id, label)?
                    != ExactComplexRational::ZERO
            {
                return Err(invalid(format!(
                    "{label} constant does not own an exact-zero default"
                )));
            }
        }
        ParameterKind::External => unreachable!("parameter kind checked above"),
    }
    Ok(())
}

fn validate_derived_zero_parameter_evaluator(
    templates: &ValidatedRecurrenceTemplateInput,
    parameter: &super::template::ParameterRow,
    label: &str,
) -> RusticolResult<()> {
    let mut owners = Vec::new();
    for evaluator in &templates.input().evaluator_bindings {
        let semantic_templates = template_u32_sequence(
            templates,
            evaluator.semantic_template_sequence_id,
            "model-parameter evaluator semantic templates",
        )?;
        if semantic_templates.contains(&parameter.template_string_id) {
            owners.push(evaluator);
        }
    }
    let [evaluator] = owners.as_slice() else {
        return Err(invalid(format!(
            "{label} has {} model-parameter evaluator owners, expected one",
            owners.len()
        )));
    };
    if EvaluatorContractKind::try_from(evaluator.contract_kind)?
        != EvaluatorContractKind::ModelParameter
        || EvaluatorCallableKind::try_from(evaluator.callable_kind)?
            != EvaluatorCallableKind::PreparedKernel
    {
        return Err(invalid(format!(
            "{label} is not owned by one prepared model-parameter evaluator"
        )));
    }
    let output_layout = template_u32_sequence(
        templates,
        evaluator.output_layout_sequence_id,
        "model-parameter evaluator output layout",
    )?;
    let exact_digests = template_u32_sequence(
        templates,
        evaluator.exact_expression_digest_sequence_id,
        "model-parameter evaluator exact expressions",
    )?;
    let parameter_name = template_string(
        templates,
        parameter.name_string_id,
        "exact-zero parameter name",
    )?;
    let expected_output = format!("model-parameter:{parameter_name}");
    let mut matching_outputs = Vec::new();
    for (index, string_id) in output_layout.iter().copied().enumerate() {
        let actual = template_string(templates, string_id, "model-parameter evaluator output")?;
        if actual == expected_output {
            matching_outputs.push(index);
        }
    }
    let [output_index] = matching_outputs.as_slice() else {
        return Err(invalid(format!(
            "{label} has {} aligned evaluator outputs, expected one",
            matching_outputs.len()
        )));
    };
    let exact_digest_id = *exact_digests
        .get(*output_index)
        .ok_or_else(|| invalid(format!("{label} evaluator output has no exact digest")))?;
    if exact_digest_id != parameter.exact_expression_digest_id {
        return Err(invalid(format!(
            "{label} parameter/evaluator exact-zero digests disagree"
        )));
    }
    Ok(())
}

fn parameter_row_is_immutable_exact_zero(
    parameter: &super::template::ParameterRow,
    expression_digest: &super::template::DigestCatalogRow,
) -> RusticolResult<bool> {
    let kind = ParameterKind::try_from(parameter.kind)?;
    Ok(
        ParameterValueType::try_from(parameter.value_type)? == ParameterValueType::Real
            && matches!(kind, ParameterKind::Derived | ParameterKind::Constant)
            && parameter.mutable == 0
            && parameter.prepared_parameter_id != MISSING_U32
            && expression_digest.id == parameter.exact_expression_digest_id
            && expression_digest.value == ZERO_PARAMETER_EXPRESSION_DIGEST,
    )
}

fn prepared_real_parameter_owner(
    templates: &ValidatedRecurrenceTemplateInput,
    prepared_slot: u32,
    label: &str,
) -> RusticolResult<u32> {
    let owners = templates
        .input()
        .parameters
        .iter()
        .filter(|parameter| parameter.prepared_parameter_id == prepared_slot)
        .collect::<Vec<_>>();
    let [owner] = owners.as_slice() else {
        return Err(invalid(format!(
            "{label} prepared slot {prepared_slot} has {} semantic owners, expected one",
            owners.len()
        )));
    };
    if ParameterValueType::try_from(owner.value_type)? != ParameterValueType::Real {
        return Err(invalid(format!(
            "{label} prepared slot {prepared_slot} is not owned by a real parameter"
        )));
    }
    Ok(owner.id)
}

fn lower_qcd_source(
    current: &super::RecurrenceCurrent,
    state: QcdStateKind,
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    dense_parameter_slots: &BTreeMap<u32, u16>,
    vector_reference_atoms: &BTreeMap<u16, u16>,
    builder: &mut SpinorDagBuilder,
) -> RusticolResult<QcdCurrent> {
    let source = qcd_source_row(current, templates)?;
    let source_template_id = source.id;
    // The recurrence builder has already authenticated the source template's
    // canonical state against the effective current state, including the
    // initial-state crossing map.  Those IDs are deliberately different for
    // a crossed massless fermion, so requiring literal equality here would
    // reject the very crossing contract we consumed above.
    let family = template_string(
        templates,
        source.wavefunction_family_string_id,
        "source wavefunction family",
    )?;
    let evaluator = evaluator_row(
        templates,
        source.evaluator_binding_id,
        EvaluatorContractKind::Source,
    )?;
    let descriptor = direct
        .intrinsic_descriptor(DirectExecutorRole::Source, evaluator.id)
        .ok_or_else(|| invalid("QCD source has no authenticated direct intrinsic"))?;
    validate_intrinsic_runtime_binding(templates, evaluator, descriptor, "source")?;
    if descriptor.contract_digest().is_none()
        || descriptor.scale().is_some()
        || descriptor.chiral_dirac_vector().is_some()
        || descriptor.massive_dirac_finalizer().is_some()
        || descriptor.massive_vector_finalizer().is_some()
        || descriptor.massive_scalar_finalizer().is_some()
    {
        return Err(invalid("QCD source intrinsic descriptor is malformed"));
    }
    let [source_slot] = current.key().support_source_slots() else {
        return Err(invalid("QCD source support is not singleton"));
    };
    let atom = u16::try_from(*source_slot).map_err(|_| invalid("source slot exceeds u16"))?;
    let source_factor = builder.constant(
        current
            .source_exact_factor()
            .ok_or_else(|| invalid("QCD source current has no exact source factor"))?,
    )?;
    match state {
        QcdStateKind::Scalar {
            mass_parameter_id,
            width_parameter_id,
            ..
        } => {
            if family != "scalar"
                || !descriptor
                    .runtime_template()
                    .starts_with(SCALAR_SOURCE_PREFIX)
                || source.helicity != 0
                || source.spin_state != 0
                || source.mass_parameter_id != mass_parameter_id
                || source.width_parameter_id != width_parameter_id
            {
                return Err(invalid(format!(
                    "source template {source_template_id} is not a scalar source"
                )));
            }
            Ok(QcdCurrent::Scalar(source_factor))
        }
        QcdStateKind::Vector => {
            if family != "vector"
                || !descriptor
                    .runtime_template()
                    .starts_with(VECTOR_SOURCE_PREFIX)
                || !matches!(source.helicity, -1 | 1)
                || source.spin_state != source.helicity
                || source.mass_parameter_id != MISSING_U32
                || source.width_parameter_id != MISSING_U32
            {
                return Err(invalid(format!(
                    "source template {source_template_id} is not a transverse vector source"
                )));
            }
            let helicity = i8::try_from(source.helicity)
                .map_err(|_| invalid("vector source helicity exceeds i8"))?;
            let polarization = if let Some(reference) = vector_reference_atoms.get(&atom) {
                external_polarization_expression_with_reference(
                    builder, atom, *reference, helicity,
                )?
            } else {
                external_polarization_expression(builder, atom, helicity)?
            };
            Ok(QcdCurrent::Vector(bispinor_scale(
                builder,
                source_factor,
                &polarization,
            )?))
        }
        QcdStateKind::U1SubtractionVector => Err(invalid(
            "an auxiliary U(1)-subtraction vector cannot be an external QCD source",
        )),
        QcdStateKind::MassiveVector {
            mass_parameter_id,
            mass_prepared_slot,
            width_parameter_id,
            ..
        } => {
            if family != "vector"
                || !descriptor
                    .runtime_template()
                    .starts_with(VECTOR_SOURCE_PREFIX)
                || !matches!(source.helicity, -1 | 0 | 1)
                || source.spin_state != source.helicity
                || source.mass_parameter_id != mass_parameter_id
                || source.width_parameter_id != width_parameter_id
            {
                return Err(invalid(format!(
                    "source template {source_template_id} is not an authenticated massive-vector source"
                )));
            }
            let helicity = i8::try_from(source.helicity)
                .map_err(|_| invalid("massive-vector source helicity exceeds i8"))?;
            let polarization = if helicity == 0 {
                let dense_mass = dense_parameter_slots
                    .get(&mass_prepared_slot)
                    .copied()
                    .ok_or_else(|| {
                        invalid("longitudinal massive-vector source mass has no graph binding")
                    })?;
                let mass = builder.parameter(dense_mass)?;
                massive_vector_longitudinal_polarization_expression(builder, atom, mass)?
            } else {
                massive_vector_polarization_expression(builder, atom, helicity)?
            };
            Ok(QcdCurrent::Vector(bispinor_scale(
                builder,
                source_factor,
                &polarization,
            )?))
        }
        QcdStateKind::Bivector => Err(invalid(
            "an auxiliary antisymmetric tensor cannot be an external QCD source",
        )),
        QcdStateKind::Weyl {
            chirality,
            orientation,
            mass_parameter_id,
        } => {
            if family != "fermion"
                || !descriptor
                    .runtime_template()
                    .starts_with(FERMION_SOURCE_PREFIX)
                || !matches!(source.helicity, -1 | 1)
                || source.spin_state != source.helicity
                || source.mass_parameter_id != mass_parameter_id
                || source.width_parameter_id != MISSING_U32
            {
                return Err(invalid(format!(
                    "source template {source_template_id} is not a massless Weyl source"
                )));
            }
            Ok(QcdCurrent::Weyl {
                chirality,
                orientation,
                value: Some(LinearWeylExpression::atom(atom, source_factor)),
            })
        }
        QcdStateKind::Dirac {
            orientation,
            width_parameter_id,
            ..
        } => {
            if family != "fermion"
                || !descriptor
                    .runtime_template()
                    .starts_with(FERMION_SOURCE_PREFIX)
                || !matches!(source.helicity, -1 | 1)
                || source.spin_state != source.helicity
                || source.mass_parameter_id == MISSING_U32
                || source.width_parameter_id != width_parameter_id
            {
                return Err(invalid(format!(
                    "source template {source_template_id} is not a massive Dirac source"
                )));
            }
            let prepared_mass = prepared_real_parameter_slot(
                templates,
                source.mass_parameter_id,
                "massive Dirac source mass",
            )?;
            let dense_mass = dense_parameter_slots
                .get(&prepared_mass)
                .copied()
                .ok_or_else(|| invalid("massive Dirac source mass has no graph binding"))?;
            let mass = builder.parameter(dense_mass)?;
            let source_value = massive_dirac_source_expression(
                builder,
                atom,
                mass,
                i8::try_from(source.spin_state)
                    .map_err(|_| invalid("massive Dirac spin state exceeds i8"))?,
            )?;
            Ok(QcdCurrent::Dirac {
                orientation,
                value: dirac_scale(builder, source_factor, &source_value)?,
            })
        }
    }
}

fn qcd_parameter_slots(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    source_parameter_slots: &[u32],
    dirac_mass_prepared_slot: Option<u32>,
    dirac_width_prepared_slot: Option<u32>,
) -> RusticolResult<Vec<u32>> {
    let mut slots = source_parameter_slots
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if let Some(slot) = dirac_mass_prepared_slot {
        slots.insert(slot);
    }
    for contribution in program.contributions() {
        let transition = transition_row(templates, contribution.key().transition_template_id())?;
        let descriptor = direct
            .intrinsic_descriptor(
                DirectExecutorRole::Contribution,
                transition.evaluator_binding_id,
            )
            .ok_or_else(|| invalid("QCD transition has no authenticated intrinsic descriptor"))?;
        qcd_contribution_kind(descriptor)?;
        validate_transition_parameter_owner(templates, transition, descriptor)?;
        if let Some(slot) = descriptor
            .scale()
            .and_then(|scale| scale.prepared_parameter_slot())
        {
            slots.insert(slot);
        }
        if let Some(chiral) = descriptor.chiral_dirac_vector() {
            for scale in [chiral.left_scale(), chiral.right_scale()] {
                if let Some(slot) = scale.prepared_parameter_slot() {
                    slots.insert(slot);
                }
            }
        }
    }
    for current in program
        .currents()
        .iter()
        .filter(|current| !current.is_source())
    {
        if qcd_optional_finalization(current, program)?
            .and_then(|finalization| finalization.propagator_template_id())
            .is_none()
        {
            continue;
        }
        match qcd_state_kind(templates, current.key().current_state_template_id())? {
            QcdStateKind::Scalar {
                mass_parameter_id,
                mass_prepared_slot: Some(mass_prepared_slot),
                ..
            } => {
                let finalizer = qcd_massive_scalar_finalizer_contract(
                    current,
                    mass_parameter_id,
                    mass_prepared_slot,
                    program,
                    templates,
                    direct,
                )?;
                slots.insert(finalizer.mass_prepared_parameter_slot());
                slots.insert(finalizer.width_prepared_parameter_slot());
            }
            QcdStateKind::Scalar {
                mass_prepared_slot: None,
                ..
            } => {
                return Err(invalid(
                    "active scalar finalization has no authenticated mass owner",
                ));
            }
            QcdStateKind::Dirac { orientation, .. } => {
                let source_mass = dirac_mass_prepared_slot.ok_or_else(|| {
                    invalid("massive Dirac finalization has no authenticated source-mass owner")
                })?;
                let finalizer = qcd_massive_finalizer_contract(
                    current,
                    orientation,
                    program,
                    templates,
                    direct,
                    source_mass,
                    dirac_width_prepared_slot,
                )?;
                slots.insert(finalizer.mass_prepared_parameter_slot());
                slots.insert(finalizer.width_prepared_parameter_slot());
            }
            QcdStateKind::MassiveVector {
                mass_parameter_id,
                mass_prepared_slot,
                ..
            } => {
                let finalizer = qcd_massive_vector_finalizer_contract(
                    current,
                    mass_parameter_id,
                    mass_prepared_slot,
                    program,
                    templates,
                    direct,
                )?;
                slots.insert(finalizer.mass_prepared_parameter_slot());
                slots.insert(finalizer.width_prepared_parameter_slot());
            }
            _ => {}
        }
    }
    Ok(slots.into_iter().collect())
}

fn qcd_contribution_kind(
    descriptor: &super::PreparedDirectIntrinsicDescriptor,
) -> RusticolResult<QcdContributionKind> {
    let expected_digest = match descriptor.runtime_template() {
        THREE_VECTOR_TEMPLATE => THREE_VECTOR_CONTRACT,
        ANTISYMMETRIC_TENSOR_VECTOR_TEMPLATE => ANTISYMMETRIC_TENSOR_VECTOR_CONTRACT,
        VECTOR_WEDGE_VECTOR_TEMPLATE => VECTOR_WEDGE_VECTOR_CONTRACT,
        WEYL_VECTOR_A_TEMPLATE => WEYL_VECTOR_A_CONTRACT,
        WEYL_VECTOR_B_TEMPLATE => WEYL_VECTOR_B_CONTRACT,
        WEYL_PAIR_VECTOR_A_TEMPLATE => WEYL_PAIR_VECTOR_A_CONTRACT,
        WEYL_PAIR_VECTOR_B_TEMPLATE => WEYL_PAIR_VECTOR_B_CONTRACT,
        DIRAC_VECTOR_PARTICLE_TEMPLATE => DIRAC_VECTOR_PARTICLE_CONTRACT,
        DIRAC_VECTOR_ANTIPARTICLE_TEMPLATE => DIRAC_VECTOR_ANTIPARTICLE_CONTRACT,
        CHIRAL_DIRAC_VECTOR_PARTICLE_TEMPLATE => CHIRAL_DIRAC_VECTOR_PARTICLE_CONTRACT,
        CHIRAL_DIRAC_VECTOR_ANTIPARTICLE_TEMPLATE => CHIRAL_DIRAC_VECTOR_ANTIPARTICLE_CONTRACT,
        DIRAC_SCALAR_TEMPLATE => DIRAC_SCALAR_CONTRACT,
        VECTOR_PAIR_SCALAR_TEMPLATE => VECTOR_PAIR_SCALAR_CONTRACT,
        other => {
            return Err(invalid(format!(
                "unsupported QCD contribution primitive {other:?}"
            )));
        }
    };
    let chiral = matches!(
        descriptor.runtime_template(),
        CHIRAL_DIRAC_VECTOR_PARTICLE_TEMPLATE | CHIRAL_DIRAC_VECTOR_ANTIPARTICLE_TEMPLATE
    );
    if descriptor.massive_dirac_finalizer().is_some()
        || descriptor.massive_vector_finalizer().is_some()
        || descriptor.massive_scalar_finalizer().is_some()
        || descriptor
            .contract_digest()
            .is_none_or(|digest| digest.to_string() != expected_digest)
        || if chiral {
            descriptor.scale().is_some() || descriptor.chiral_dirac_vector().is_none()
        } else {
            descriptor.scale().is_none() || descriptor.chiral_dirac_vector().is_some()
        }
    {
        return Err(invalid(format!(
            "QCD contribution descriptor {:?} has the wrong authenticated contract",
            descriptor.runtime_template()
        )));
    }
    Ok(match descriptor.runtime_template() {
        THREE_VECTOR_TEMPLATE => QcdContributionKind::ThreeVector,
        ANTISYMMETRIC_TENSOR_VECTOR_TEMPLATE => QcdContributionKind::AntisymmetricTensorVector,
        VECTOR_WEDGE_VECTOR_TEMPLATE => QcdContributionKind::VectorWedgeVector,
        WEYL_VECTOR_A_TEMPLATE => QcdContributionKind::WeylVector(SpinorChirality::Positive),
        WEYL_VECTOR_B_TEMPLATE => QcdContributionKind::WeylVector(SpinorChirality::Negative),
        WEYL_PAIR_VECTOR_A_TEMPLATE => {
            QcdContributionKind::WeylPairVector(SpinorChirality::Negative)
        }
        WEYL_PAIR_VECTOR_B_TEMPLATE => {
            QcdContributionKind::WeylPairVector(SpinorChirality::Positive)
        }
        DIRAC_VECTOR_PARTICLE_TEMPLATE => {
            QcdContributionKind::DiracVector(CurrentOrientation::Particle)
        }
        DIRAC_VECTOR_ANTIPARTICLE_TEMPLATE => {
            QcdContributionKind::DiracVector(CurrentOrientation::Antiparticle)
        }
        CHIRAL_DIRAC_VECTOR_PARTICLE_TEMPLATE => {
            QcdContributionKind::ChiralDiracVector(CurrentOrientation::Particle)
        }
        CHIRAL_DIRAC_VECTOR_ANTIPARTICLE_TEMPLATE => {
            QcdContributionKind::ChiralDiracVector(CurrentOrientation::Antiparticle)
        }
        DIRAC_SCALAR_TEMPLATE => QcdContributionKind::DiracScalar,
        VECTOR_PAIR_SCALAR_TEMPLATE => QcdContributionKind::VectorPairScalar,
        _ => unreachable!("runtime template checked above"),
    })
}

fn intrinsic_scale_node(
    descriptor: &super::PreparedDirectIntrinsicDescriptor,
    dense_parameter_slots: &BTreeMap<u32, u16>,
    builder: &mut SpinorDagBuilder,
) -> RusticolResult<u32> {
    let scale = descriptor
        .scale()
        .ok_or_else(|| invalid("intrinsic descriptor has no exact scale"))?;
    intrinsic_scale_value_node(scale, dense_parameter_slots, builder)
}

fn intrinsic_scale_value_node(
    scale: super::PreparedDirectIntrinsicScale,
    dense_parameter_slots: &BTreeMap<u32, u16>,
    builder: &mut SpinorDagBuilder,
) -> RusticolResult<u32> {
    let constant = ExactComplexRational::new(
        ExactRational::from_f64_exact(f64::from_bits(scale.constant_real_bits()))?,
        ExactRational::from_f64_exact(f64::from_bits(scale.constant_imag_bits()))?,
    );
    let constant = builder.constant(constant)?;
    if let Some(prepared_slot) = scale.prepared_parameter_slot() {
        let dense = dense_parameter_slots
            .get(&prepared_slot)
            .copied()
            .ok_or_else(|| invalid("prepared parameter has no dense graph binding"))?;
        let parameter = builder.parameter(dense)?;
        builder.product([constant, parameter])
    } else {
        Ok(constant)
    }
}

fn oriented_chiral_half_scales(
    orientation: CurrentOrientation,
    left: u32,
    right: u32,
) -> RusticolResult<(u32, u32)> {
    match orientation {
        CurrentOrientation::Particle => Ok((left, right)),
        CurrentOrientation::Antiparticle => Ok((right, left)),
        CurrentOrientation::SelfConjugate => {
            Err(invalid("a chiral Dirac current cannot be self-conjugate"))
        }
    }
}

fn qcd_current_momentum(
    current: &super::RecurrenceCurrent,
    representative_signs: &[i32],
    builder: &mut SpinorDagBuilder,
) -> RusticolResult<(BispinorExpression, Vec<(u16, u32)>)> {
    let mut terms = Vec::with_capacity(current.key().momentum().terms().len());
    for term in current.key().momentum().terms() {
        let source = usize::try_from(term.source_slot)
            .map_err(|_| invalid("momentum source slot exceeds usize"))?;
        let representative_sign = representative_signs
            .get(source)
            .copied()
            .ok_or_else(|| invalid("momentum source slot is out of bounds"))?;
        let coefficient = term
            .coefficient
            .checked_mul(representative_sign)
            .ok_or_else(|| invalid("momentum coefficient overflows i32"))?;
        let coefficient = builder.constant(ExactComplexRational::new(
            ExactRational::new(i128::from(coefficient), 1)?,
            ExactRational::ZERO,
        ))?;
        let source = u16::try_from(term.source_slot)
            .map_err(|_| invalid("momentum source slot exceeds u16"))?;
        for atom in builder.source_momentum_atoms(source)? {
            terms.push((atom, coefficient));
        }
    }
    if terms.is_empty() {
        return Err(invalid(format!(
            "current {} has empty momentum support",
            current.id()
        )));
    }
    Ok((signed_momentum_expression(&terms), terms))
}

fn qcd_optional_finalization<'a>(
    current: &super::RecurrenceCurrent,
    program: &'a RecurrenceProgram,
) -> RusticolResult<Option<&'a super::RecurrenceFinalization>> {
    let Some(id) = current.finalization_id() else {
        return Ok(None);
    };
    let finalization = program
        .finalizations()
        .get(id as usize)
        .ok_or_else(|| invalid(format!("finalization {id} is absent")))?;
    if finalization.id() != id || finalization.current_id() != current.id() {
        return Err(invalid(format!(
            "finalization {id} does not canonically own current {}",
            current.id()
        )));
    }
    Ok(Some(finalization))
}

fn qcd_finalization<'a>(
    current: &super::RecurrenceCurrent,
    program: &'a RecurrenceProgram,
) -> RusticolResult<&'a super::RecurrenceFinalization> {
    qcd_optional_finalization(current, program)?.ok_or_else(|| {
        invalid(format!(
            "current {} has no active finalization",
            current.id()
        ))
    })
}

fn qcd_massive_finalizer_contract(
    current: &super::RecurrenceCurrent,
    orientation: CurrentOrientation,
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    source_mass_prepared_slot: u32,
    source_width_prepared_slot: Option<u32>,
) -> RusticolResult<super::PreparedDirectMassiveDiracFinalizer> {
    let finalization = qcd_finalization(current, program)?;
    let propagator_id = finalization
        .propagator_template_id()
        .ok_or_else(|| invalid("active massive Dirac current has an identity finalization"))?;
    let propagator = templates
        .input()
        .propagators
        .get(propagator_id as usize)
        .ok_or_else(|| invalid(format!("propagator template {propagator_id} is absent")))?;
    let state_id = current.key().current_state_template_id();
    let state = templates
        .input()
        .current_states
        .get(state_id as usize)
        .ok_or_else(|| invalid(format!("current-state template {state_id} is absent")))?;
    if propagator.id != propagator_id
        || propagator.applies_propagator != 1
        || propagator.state_template_id != state_id
    {
        return Err(invalid(format!(
            "propagator template {propagator_id} is not the current's authenticated massive Dirac propagator"
        )));
    }
    let evaluator = evaluator_row(
        templates,
        propagator.evaluator_binding_id,
        EvaluatorContractKind::Propagator,
    )?;
    direct.resolve_evaluator(DirectExecutorRole::Finalization, evaluator.id)?;
    let descriptor = direct
        .intrinsic_descriptor(DirectExecutorRole::Finalization, evaluator.id)
        .ok_or_else(|| {
            invalid("massive Dirac propagator has no authenticated intrinsic descriptor")
        })?;
    validate_intrinsic_runtime_binding(
        templates,
        evaluator,
        descriptor,
        "massive Dirac finalization",
    )?;
    let (runtime_template, contract_digest) = match orientation {
        CurrentOrientation::Particle => (
            MASSIVE_DIRAC_PARTICLE_TEMPLATE,
            MASSIVE_DIRAC_PARTICLE_CONTRACT,
        ),
        CurrentOrientation::Antiparticle => (
            MASSIVE_DIRAC_ANTIPARTICLE_TEMPLATE,
            MASSIVE_DIRAC_ANTIPARTICLE_CONTRACT,
        ),
        CurrentOrientation::SelfConjugate => {
            return Err(invalid(
                "massive Dirac finalization has self-conjugate orientation",
            ));
        }
    };
    let typed = descriptor
        .massive_dirac_finalizer()
        .ok_or_else(|| invalid("massive Dirac finalization has no authenticated typed operands"))?;
    if descriptor.runtime_template() != runtime_template
        || descriptor
            .contract_digest()
            .is_none_or(|digest| digest.to_string() != contract_digest)
        || descriptor.scale().is_some()
        || descriptor.chiral_dirac_vector().is_some()
        || descriptor.massive_vector_finalizer().is_some()
        || descriptor.massive_scalar_finalizer().is_some()
        || typed.orientation() != orientation
    {
        return Err(invalid(format!(
            "unsupported massive Dirac finalization primitive {:?}",
            descriptor.runtime_template()
        )));
    }
    if typed.mass_prepared_parameter_slot() != source_mass_prepared_slot {
        return Err(invalid(
            "massive Dirac finalizer mass disagrees with authenticated source ownership",
        ));
    }
    if source_width_prepared_slot
        .is_some_and(|source_width| source_width != typed.width_prepared_parameter_slot())
    {
        return Err(invalid(
            "massive Dirac finalizer width disagrees with authenticated source ownership",
        ));
    }
    if propagator.mass_parameter_id != MISSING_U32 {
        let propagator_mass = prepared_real_parameter_slot(
            templates,
            propagator.mass_parameter_id,
            "massive Dirac propagator mass",
        )?;
        if propagator.mass_parameter_id != state.mass_parameter_id
            || propagator_mass != typed.mass_prepared_parameter_slot()
        {
            return Err(invalid(
                "massive Dirac propagator mass disagrees with state/finalizer ownership",
            ));
        }
    }
    let width_owner = prepared_real_parameter_owner(
        templates,
        typed.width_prepared_parameter_slot(),
        "massive Dirac finalizer width",
    )?;
    validate_optional_width_owner(state.width_parameter_id, width_owner, "massive Dirac state")?;
    validate_optional_width_owner(
        propagator.width_parameter_id,
        width_owner,
        "massive Dirac propagator",
    )?;
    Ok(typed)
}

fn qcd_massive_vector_finalizer_contract(
    current: &super::RecurrenceCurrent,
    state_mass_parameter_id: u32,
    state_mass_prepared_slot: u32,
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
) -> RusticolResult<super::PreparedDirectMassiveVectorFinalizer> {
    let finalization = qcd_finalization(current, program)?;
    let propagator_id = finalization
        .propagator_template_id()
        .ok_or_else(|| invalid("active massive-vector current has an identity finalization"))?;
    let propagator = templates
        .input()
        .propagators
        .get(propagator_id as usize)
        .ok_or_else(|| invalid(format!("propagator template {propagator_id} is absent")))?;
    let state_id = current.key().current_state_template_id();
    let state = templates
        .input()
        .current_states
        .get(state_id as usize)
        .ok_or_else(|| invalid(format!("current-state template {state_id} is absent")))?;
    if propagator.id != propagator_id
        || propagator.applies_propagator != 1
        || propagator.state_template_id != state_id
        || state.mass_parameter_id != state_mass_parameter_id
    {
        return Err(invalid(format!(
            "propagator template {propagator_id} is not the current's authenticated massive-vector propagator"
        )));
    }
    let evaluator = evaluator_row(
        templates,
        propagator.evaluator_binding_id,
        EvaluatorContractKind::Propagator,
    )?;
    direct.resolve_evaluator(DirectExecutorRole::Finalization, evaluator.id)?;
    let descriptor = direct
        .intrinsic_descriptor(DirectExecutorRole::Finalization, evaluator.id)
        .ok_or_else(|| {
            invalid("massive-vector propagator has no authenticated intrinsic descriptor")
        })?;
    validate_intrinsic_runtime_binding(
        templates,
        evaluator,
        descriptor,
        "massive-vector finalization",
    )?;
    let typed = descriptor.massive_vector_finalizer().ok_or_else(|| {
        invalid("massive-vector finalization has no authenticated typed operands")
    })?;
    if descriptor.runtime_template() != MASSIVE_VECTOR_UNITARY_TEMPLATE
        || descriptor
            .contract_digest()
            .is_none_or(|digest| digest.to_string() != MASSIVE_VECTOR_UNITARY_CONTRACT)
        || descriptor.scale().is_some()
        || descriptor.chiral_dirac_vector().is_some()
        || descriptor.massive_dirac_finalizer().is_some()
        || descriptor.massive_scalar_finalizer().is_some()
        || typed.constant_real_bits() != 0.0_f64.to_bits()
        || typed.constant_imag_bits() != (-1.0_f64).to_bits()
    {
        return Err(invalid(format!(
            "unsupported massive-vector finalization primitive {:?}",
            descriptor.runtime_template()
        )));
    }
    if typed.mass_prepared_parameter_slot() != state_mass_prepared_slot {
        return Err(invalid(
            "massive-vector finalizer mass disagrees with authenticated state ownership",
        ));
    }
    if propagator.mass_parameter_id != MISSING_U32 {
        let propagator_mass = prepared_real_parameter_slot(
            templates,
            propagator.mass_parameter_id,
            "massive-vector propagator mass",
        )?;
        if propagator.mass_parameter_id != state_mass_parameter_id
            || propagator_mass != typed.mass_prepared_parameter_slot()
        {
            return Err(invalid(
                "massive-vector propagator mass disagrees with state/finalizer ownership",
            ));
        }
    }
    let width_owner = prepared_real_parameter_owner(
        templates,
        typed.width_prepared_parameter_slot(),
        "massive-vector finalizer width",
    )?;
    validate_optional_width_owner(
        state.width_parameter_id,
        width_owner,
        "massive-vector state",
    )?;
    validate_optional_width_owner(
        propagator.width_parameter_id,
        width_owner,
        "massive-vector propagator",
    )?;
    Ok(typed)
}

fn qcd_massive_scalar_finalizer_contract(
    current: &super::RecurrenceCurrent,
    state_mass_parameter_id: u32,
    state_mass_prepared_slot: u32,
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
) -> RusticolResult<super::PreparedDirectMassiveScalarFinalizer> {
    let finalization = qcd_finalization(current, program)?;
    let propagator_id = finalization
        .propagator_template_id()
        .ok_or_else(|| invalid("active massive-scalar current has an identity finalization"))?;
    let propagator = templates
        .input()
        .propagators
        .get(propagator_id as usize)
        .ok_or_else(|| invalid(format!("propagator template {propagator_id} is absent")))?;
    let state_id = current.key().current_state_template_id();
    let state = templates
        .input()
        .current_states
        .get(state_id as usize)
        .ok_or_else(|| invalid(format!("current-state template {state_id} is absent")))?;
    if propagator.id != propagator_id
        || propagator.applies_propagator != 1
        || propagator.state_template_id != state_id
        || state.mass_parameter_id != state_mass_parameter_id
    {
        return Err(invalid(format!(
            "propagator template {propagator_id} is not the current's authenticated massive-scalar propagator"
        )));
    }
    let evaluator = evaluator_row(
        templates,
        propagator.evaluator_binding_id,
        EvaluatorContractKind::Propagator,
    )?;
    direct.resolve_evaluator(DirectExecutorRole::Finalization, evaluator.id)?;
    let descriptor = direct
        .intrinsic_descriptor(DirectExecutorRole::Finalization, evaluator.id)
        .ok_or_else(|| {
            invalid("massive-scalar propagator has no authenticated intrinsic descriptor")
        })?;
    validate_intrinsic_runtime_binding(
        templates,
        evaluator,
        descriptor,
        "massive-scalar finalization",
    )?;
    let typed = descriptor.massive_scalar_finalizer().ok_or_else(|| {
        invalid("massive-scalar finalization has no authenticated typed operands")
    })?;
    if descriptor.runtime_template() != MASSIVE_SCALAR_TEMPLATE
        || descriptor
            .contract_digest()
            .is_none_or(|digest| digest.to_string() != MASSIVE_SCALAR_CONTRACT)
        || descriptor.scale().is_some()
        || descriptor.chiral_dirac_vector().is_some()
        || descriptor.massive_dirac_finalizer().is_some()
        || descriptor.massive_vector_finalizer().is_some()
        || typed.constant_real_bits() != 0.0_f64.to_bits()
        || typed.constant_imag_bits() != 1.0_f64.to_bits()
    {
        return Err(invalid(format!(
            "unsupported massive-scalar finalization primitive {:?}",
            descriptor.runtime_template()
        )));
    }
    if typed.mass_prepared_parameter_slot() != state_mass_prepared_slot {
        return Err(invalid(
            "massive-scalar finalizer mass disagrees with authenticated state ownership",
        ));
    }
    if propagator.mass_parameter_id != MISSING_U32 {
        let propagator_mass = prepared_real_parameter_slot(
            templates,
            propagator.mass_parameter_id,
            "massive-scalar propagator mass",
        )?;
        if propagator.mass_parameter_id != state_mass_parameter_id
            || propagator_mass != typed.mass_prepared_parameter_slot()
        {
            return Err(invalid(
                "massive-scalar propagator mass disagrees with state/finalizer ownership",
            ));
        }
    }
    let width_owner = prepared_real_parameter_owner(
        templates,
        typed.width_prepared_parameter_slot(),
        "massive-scalar finalizer width",
    )?;
    validate_optional_width_owner(
        state.width_parameter_id,
        width_owner,
        "massive-scalar state",
    )?;
    validate_optional_width_owner(
        propagator.width_parameter_id,
        width_owner,
        "massive-scalar propagator",
    )?;
    Ok(typed)
}

fn validate_optional_width_owner(
    parameter_id: u32,
    authenticated_owner: u32,
    label: &str,
) -> RusticolResult<()> {
    if parameter_id != MISSING_U32 && parameter_id != authenticated_owner {
        return Err(invalid(format!(
            "{label} width disagrees with typed finalizer ownership"
        )));
    }
    Ok(())
}

fn require_identity_finalizer(direct: &PreparedDirectExecutorCatalog) -> RusticolResult<()> {
    direct.resolve_identity_finalizer()?;
    let descriptor = direct
        .intrinsic_descriptor_by_key(super::PreparedDirectExecutorKey::IdentityFinalizer)
        .ok_or_else(|| invalid("authenticated identity finalizer is absent"))?;
    if descriptor.runtime_template() != IDENTITY_FINALIZER
        || descriptor.contract_digest().is_some()
        || descriptor.scale().is_some()
        || descriptor.chiral_dirac_vector().is_some()
        || descriptor.massive_dirac_finalizer().is_some()
        || descriptor.massive_vector_finalizer().is_some()
        || descriptor.massive_scalar_finalizer().is_some()
    {
        return Err(invalid(
            "authenticated identity-finalizer descriptor is malformed",
        ));
    }
    Ok(())
}

fn qcd_finalization_scale(
    current: &super::RecurrenceCurrent,
    state: QcdStateKind,
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    dense_parameter_slots: &BTreeMap<u32, u16>,
    representative_signs: &[i32],
    builder: &mut SpinorDagBuilder,
) -> RusticolResult<u32> {
    let finalization = qcd_finalization(current, program)?;
    let propagator_id = finalization
        .propagator_template_id()
        .ok_or_else(|| invalid("active QCD current has an identity finalization"))?;
    let propagator = templates
        .input()
        .propagators
        .get(propagator_id as usize)
        .ok_or_else(|| invalid(format!("propagator template {propagator_id} is absent")))?;
    let mass_owner_matches = match state {
        QcdStateKind::Weyl {
            mass_parameter_id, ..
        } => {
            propagator.mass_parameter_id == MISSING_U32
                || propagator.mass_parameter_id == mass_parameter_id
        }
        _ => propagator.mass_parameter_id == MISSING_U32,
    };
    if propagator.id != propagator_id
        || propagator.applies_propagator != 1
        || propagator.state_template_id != current.key().current_state_template_id()
        || !mass_owner_matches
        || propagator.width_parameter_id != MISSING_U32
    {
        return Err(invalid(format!(
            "propagator template {propagator_id} is not the current's massless propagator"
        )));
    }
    let evaluator = evaluator_row(
        templates,
        propagator.evaluator_binding_id,
        EvaluatorContractKind::Propagator,
    )?;
    direct.resolve_evaluator(DirectExecutorRole::Finalization, evaluator.id)?;
    let descriptor = direct
        .intrinsic_descriptor(DirectExecutorRole::Finalization, evaluator.id)
        .ok_or_else(|| invalid("QCD propagator has no authenticated intrinsic descriptor"))?;
    let (runtime_template, contract_digest, negate) = match state {
        QcdStateKind::Scalar { .. } => {
            return Err(invalid("a scalar insertion cannot have a QCD finalizer"));
        }
        QcdStateKind::Vector | QcdStateKind::U1SubtractionVector => (
            VECTOR_PROPAGATOR_TEMPLATE,
            VECTOR_PROPAGATOR_CONTRACT,
            false,
        ),
        QcdStateKind::MassiveVector { .. } => {
            return Err(invalid(
                "an internal massive vector requires an authenticated massive-vector propagator",
            ));
        }
        QcdStateKind::Bivector => {
            return Err(invalid(
                "an auxiliary antisymmetric tensor cannot have an active propagator",
            ));
        }
        QcdStateKind::Weyl {
            chirality: SpinorChirality::Positive,
            ..
        } => (WEYL_PROPAGATOR_B_TEMPLATE, WEYL_PROPAGATOR_B_CONTRACT, true),
        QcdStateKind::Weyl {
            chirality: SpinorChirality::Negative,
            ..
        } => (WEYL_PROPAGATOR_A_TEMPLATE, WEYL_PROPAGATOR_A_CONTRACT, true),
        QcdStateKind::Dirac { .. } => {
            return Err(invalid(
                "massive Dirac finalization must use its typed operand contract",
            ));
        }
    };
    if descriptor.runtime_template() != runtime_template
        || descriptor
            .contract_digest()
            .is_none_or(|digest| digest.to_string() != contract_digest)
        || descriptor
            .scale()
            .is_none_or(|scale| scale.prepared_parameter_slot().is_some())
        || descriptor.chiral_dirac_vector().is_some()
        || descriptor.massive_dirac_finalizer().is_some()
        || descriptor.massive_vector_finalizer().is_some()
        || descriptor.massive_scalar_finalizer().is_some()
    {
        return Err(invalid(format!(
            "unsupported QCD finalization primitive {:?}",
            descriptor.runtime_template()
        )));
    }
    let mut factors = vec![
        intrinsic_scale_node(descriptor, dense_parameter_slots, builder)?,
        builder.constant(finalization.exact_factor())?,
    ];
    if negate {
        factors.push(builder.constant(ExactComplexRational::new(
            ExactRational::new(-1, 1)?,
            ExactRational::ZERO,
        ))?);
    }
    let (momentum, _) = qcd_current_momentum(current, representative_signs, builder)?;
    let mass_squared = bispinor_dot_expression(builder, &momentum, &momentum)?;
    factors.push(builder.reciprocal(mass_squared)?);
    builder.product(factors)
}

fn qcd_contribution_contract<'a>(
    contribution: &super::RecurrenceContribution,
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &'a PreparedDirectExecutorCatalog,
) -> RusticolResult<(
    QcdContributionKind,
    [u32; 2],
    &'a super::PreparedDirectIntrinsicDescriptor,
)> {
    let [semantic_left, semantic_right] = contribution.parent_current_ids() else {
        return Err(invalid(format!(
            "QCD contribution {} is not binary",
            contribution.id()
        )));
    };
    if *semantic_left >= contribution.result_current_id()
        || *semantic_right >= contribution.result_current_id()
    {
        return Err(invalid(format!(
            "QCD contribution {} is not topologically ordered",
            contribution.id()
        )));
    }
    let transition = transition_row(templates, contribution.key().transition_template_id())?;
    if transition.result_state_template_id
        != program.currents()[contribution.result_current_id() as usize]
            .key()
            .current_state_template_id()
    {
        return Err(invalid(format!(
            "QCD transition {} has the wrong result state",
            transition.id
        )));
    }
    let evaluator = evaluator_row(
        templates,
        transition.evaluator_binding_id,
        EvaluatorContractKind::Vertex,
    )?;
    let descriptor = direct
        .intrinsic_descriptor(DirectExecutorRole::Contribution, evaluator.id)
        .ok_or_else(|| invalid("QCD contribution has no authenticated intrinsic descriptor"))?;
    validate_intrinsic_runtime_binding(templates, evaluator, descriptor, "QCD contribution")?;
    let kind = qcd_contribution_kind(descriptor)?;
    validate_transition_parameter_owner(templates, transition, descriptor)?;
    direct.resolve_contribution(evaluator.id)?;
    let permutation = descriptor.parent_permutation();
    let semantic = [*semantic_left, *semantic_right];
    let parents = match permutation {
        [0, 1] => semantic,
        [1, 0] => [semantic[1], semantic[0]],
        _ => {
            return Err(invalid(
                "QCD contribution has an invalid parent permutation",
            ));
        }
    };
    // Authenticated recurrence construction has already validated the
    // transition's concrete parent states against its canonical input order
    // and any proven exchange. The graph descriptor's parent permutation is
    // a separate evaluator-algebra contract, so it must not be compared to
    // the transition's semantic state sequence here.
    Ok((kind, parents, descriptor))
}

#[allow(clippy::too_many_arguments)]
fn lower_qcd_current(
    current: &super::RecurrenceCurrent,
    state: QcdStateKind,
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    dense_parameter_slots: &BTreeMap<u32, u16>,
    dirac_mass_prepared_slot: Option<u32>,
    dirac_width_prepared_slot: Option<u32>,
    representative_signs: &[i32],
    current_values: &[Option<QcdCurrent>],
    builder: &mut SpinorDagBuilder,
) -> RusticolResult<QcdCurrent> {
    let range = current
        .contribution_range()
        .as_usize_range(program.contributions().len(), "QCD current contributions")?;
    if range.is_empty() {
        return Err(invalid(format!(
            "non-source QCD current {} has no contributions",
            current.id()
        )));
    }
    let finalization = qcd_optional_finalization(current, program)?;
    match state {
        QcdStateKind::Scalar {
            mass_parameter_id,
            mass_prepared_slot,
            ..
        } => {
            let mut terms = Vec::new();
            for contribution in &program.contributions()[range] {
                if contribution.result_current_id() != current.id() {
                    return Err(invalid("scalar contribution belongs to the wrong current"));
                }
                let (kind, parents, descriptor) =
                    qcd_contribution_contract(contribution, program, templates, direct)?;
                if kind != QcdContributionKind::VectorPairScalar {
                    return Err(invalid(
                        "scalar current uses a primitive other than vector-pair-to-scalar",
                    ));
                }
                let left = required_qcd_vector(current_values, parents[0])?;
                let right = required_qcd_vector(current_values, parents[1])?;
                let contraction = bispinor_dot_expression(builder, left, right)?;
                // Both sparse vector expressions represent V/sqrt(2), so
                // their dot product is half the authenticated component
                // contraction.
                let two = builder.constant(ExactComplexRational::new(
                    ExactRational::new(2, 1)?,
                    ExactRational::ZERO,
                ))?;
                let intrinsic = intrinsic_scale_node(descriptor, dense_parameter_slots, builder)?;
                let exact = qcd_contribution_exact_node(contribution, templates, builder)?;
                terms.push(builder.product([two, contraction, intrinsic, exact])?);
            }
            let numerator = builder.sum(terms)?;
            let value = if finalization
                .and_then(|finalization| finalization.propagator_template_id())
                .is_some()
            {
                let mass_prepared_slot = mass_prepared_slot.ok_or_else(|| {
                    invalid("massive-scalar finalization has no authenticated state mass")
                })?;
                let typed = qcd_massive_scalar_finalizer_contract(
                    current,
                    mass_parameter_id,
                    mass_prepared_slot,
                    program,
                    templates,
                    direct,
                )?;
                let dense_mass = dense_parameter_slots
                    .get(&typed.mass_prepared_parameter_slot())
                    .copied()
                    .ok_or_else(|| invalid("massive-scalar mass has no graph binding"))?;
                let dense_width = dense_parameter_slots
                    .get(&typed.width_prepared_parameter_slot())
                    .copied()
                    .ok_or_else(|| invalid("massive-scalar width has no graph binding"))?;
                let mass = builder.parameter(dense_mass)?;
                let width = builder.parameter(dense_width)?;
                let (momentum, _) = qcd_current_momentum(current, representative_signs, builder)?;
                let denominator =
                    massive_dirac_propagator_denominator(builder, &momentum, mass, width)?;
                let inverse = builder.reciprocal(denominator)?;
                let runtime_scale = builder.constant(exact_binary64_scale(
                    typed.constant_real_bits(),
                    typed.constant_imag_bits(),
                )?)?;
                let final_exact = builder.constant(
                    finalization
                        .ok_or_else(|| invalid("massive-scalar finalization is absent"))?
                        .exact_factor(),
                )?;
                builder.product([numerator, runtime_scale, final_exact, inverse])?
            } else {
                require_identity_finalizer(direct)?;
                let final_exact = builder.constant(
                    finalization
                        .map(super::RecurrenceFinalization::exact_factor)
                        .unwrap_or(ExactComplexRational::ONE),
                )?;
                builder.product([numerator, final_exact])?
            };
            Ok(QcdCurrent::Scalar(value))
        }
        state @ (QcdStateKind::Vector
        | QcdStateKind::U1SubtractionVector
        | QcdStateKind::MassiveVector { .. }) => {
            let mut terms = Vec::new();
            for contribution in &program.contributions()[range] {
                if contribution.result_current_id() != current.id() {
                    return Err(invalid("QCD contribution belongs to the wrong current"));
                }
                let (kind, parents, descriptor) =
                    qcd_contribution_contract(contribution, program, templates, direct)?;
                let scale = intrinsic_scale_node(descriptor, dense_parameter_slots, builder)?;
                let exact = qcd_contribution_exact_node(contribution, templates, builder)?;
                let (numerator, scale) = match kind {
                    QcdContributionKind::ThreeVector => {
                        let left = required_qcd_vector(current_values, parents[0])?;
                        let right = required_qcd_vector(current_values, parents[1])?;
                        let (left_momentum, _) = qcd_current_momentum(
                            &program.currents()[parents[0] as usize],
                            representative_signs,
                            builder,
                        )?;
                        let (right_momentum, _) = qcd_current_momentum(
                            &program.currents()[parents[1] as usize],
                            representative_signs,
                            builder,
                        )?;
                        let numerator = three_vector_bispinor_expression(
                            builder,
                            left,
                            &left_momentum,
                            right,
                            &right_momentum,
                        )?;
                        let sqrt_two = builder.kinematic(SpinorKinematicScalar::SqrtTwo)?;
                        (numerator, builder.product([sqrt_two, scale, exact])?)
                    }
                    QcdContributionKind::AntisymmetricTensorVector => {
                        let tensor = required_qcd_bivector(current_values, parents[0])?;
                        let vector = required_qcd_vector(current_values, parents[1])?;
                        let numerator = bivector_vector_expression(builder, tensor, vector)?;
                        // Sparse vectors represent V/sqrt(2), while sparse
                        // bivectors represent B/2.  The certified component
                        // primitive therefore needs n_B*n_V/n_V = 2 here.
                        let two = builder.constant(ExactComplexRational::new(
                            ExactRational::new(2, 1)?,
                            ExactRational::ZERO,
                        ))?;
                        (numerator, builder.product([two, scale, exact])?)
                    }
                    QcdContributionKind::WeylPairVector(particle_chirality) => {
                        let (particle, antiparticle) =
                            required_qcd_weyl_pair(current_values, parents, particle_chirality)?;
                        let numerator = weyl_pair_vector_expression(
                            builder,
                            particle_chirality,
                            particle,
                            antiparticle,
                        )?;
                        // The component witness returns a Cartesian vector,
                        // while a sparse dyad is that vector divided by two.
                        // Combining the DAG's V/sqrt(2) convention with the
                        // authenticated component scale therefore contributes
                        // one explicit sqrt(2).
                        let sqrt_two = builder.kinematic(SpinorKinematicScalar::SqrtTwo)?;
                        (numerator, builder.product([sqrt_two, scale, exact])?)
                    }
                    _ => {
                        return Err(invalid(
                            "vector current uses a primitive with the wrong result type",
                        ));
                    }
                };
                terms.push(bispinor_scale(builder, scale, &numerator)?);
            }
            let numerator = bispinor_sum(builder, terms)?;
            let propagated = if finalization
                .and_then(|finalization| finalization.propagator_template_id())
                .is_some()
            {
                match state {
                    QcdStateKind::MassiveVector {
                        mass_parameter_id,
                        mass_prepared_slot,
                        ..
                    } => {
                        let typed = qcd_massive_vector_finalizer_contract(
                            current,
                            mass_parameter_id,
                            mass_prepared_slot,
                            program,
                            templates,
                            direct,
                        )?;
                        let dense_mass = dense_parameter_slots
                            .get(&typed.mass_prepared_parameter_slot())
                            .copied()
                            .ok_or_else(|| invalid("massive-vector mass has no graph binding"))?;
                        let dense_width = dense_parameter_slots
                            .get(&typed.width_prepared_parameter_slot())
                            .copied()
                            .ok_or_else(|| invalid("massive-vector width has no graph binding"))?;
                        let mass = builder.parameter(dense_mass)?;
                        let width = builder.parameter(dense_width)?;
                        let runtime_scale = builder.constant(exact_binary64_scale(
                            typed.constant_real_bits(),
                            typed.constant_imag_bits(),
                        )?)?;
                        let final_exact = builder.constant(
                            finalization
                                .ok_or_else(|| invalid("massive-vector finalization is absent"))?
                                .exact_factor(),
                        )?;
                        let (momentum, _) =
                            qcd_current_momentum(current, representative_signs, builder)?;
                        massive_vector_propagator_expression(
                            builder,
                            &numerator,
                            &momentum,
                            mass,
                            width,
                            runtime_scale,
                            final_exact,
                        )?
                    }
                    _ => {
                        let scale = qcd_finalization_scale(
                            current,
                            state,
                            program,
                            templates,
                            direct,
                            dense_parameter_slots,
                            representative_signs,
                            builder,
                        )?;
                        bispinor_scale(builder, scale, &numerator)?
                    }
                }
            } else {
                require_identity_finalizer(direct)?;
                if program.contributions().iter().any(|contribution| {
                    contribution.result_current_id() > current.id()
                        && contribution.parent_current_ids().contains(&current.id())
                }) {
                    return Err(invalid(
                        "an identity-finalized vector current is reused as a contribution parent",
                    ));
                }
                let scale = builder.constant(
                    finalization
                        .map(super::RecurrenceFinalization::exact_factor)
                        .unwrap_or(ExactComplexRational::ONE),
                )?;
                bispinor_scale(builder, scale, &numerator)?
            };
            Ok(QcdCurrent::Vector(propagated))
        }
        QcdStateKind::Bivector => {
            if finalization
                .and_then(|finalization| finalization.propagator_template_id())
                .is_some()
            {
                return Err(invalid(
                    "an auxiliary antisymmetric tensor has an active propagator",
                ));
            }
            require_identity_finalizer(direct)?;
            let mut terms = Vec::new();
            for contribution in &program.contributions()[range] {
                if contribution.result_current_id() != current.id() {
                    return Err(invalid("QCD contribution belongs to the wrong current"));
                }
                let (kind, parents, descriptor) =
                    qcd_contribution_contract(contribution, program, templates, direct)?;
                if kind != QcdContributionKind::VectorWedgeVector {
                    return Err(invalid(
                        "bivector current uses a non-vector-wedge-vector primitive",
                    ));
                }
                let left = required_qcd_vector(current_values, parents[0])?;
                let right = required_qcd_vector(current_values, parents[1])?;
                let wedge = bivector_wedge_expression(builder, left, right);
                let intrinsic = intrinsic_scale_node(descriptor, dense_parameter_slots, builder)?;
                let exact = qcd_contribution_exact_node(contribution, templates, builder)?;
                let scale = builder.product([intrinsic, exact])?;
                // n_V*n_V/n_B = sqrt(2)^2/2 = 1.
                terms.push(bivector_scale(builder, scale, &wedge)?);
            }
            let tensor = bivector_sum(builder, terms);
            let finalization_scale = builder.constant(
                finalization
                    .map(super::RecurrenceFinalization::exact_factor)
                    .unwrap_or(ExactComplexRational::ONE),
            )?;
            Ok(QcdCurrent::Bivector(bivector_scale(
                builder,
                finalization_scale,
                &tensor,
            )?))
        }
        QcdStateKind::Weyl {
            chirality,
            orientation,
            ..
        } => {
            if finalization
                .and_then(|finalization| finalization.propagator_template_id())
                .is_none()
            {
                require_identity_finalizer(direct)?;
                for contribution in &program.contributions()[range] {
                    let (kind, parents, _) =
                        qcd_contribution_contract(contribution, program, templates, direct)?;
                    if kind != QcdContributionKind::WeylVector(chirality) {
                        return Err(invalid(
                            "terminal Weyl current uses the wrong chiral vertex primitive",
                        ));
                    }
                    let (parent_chirality, parent_orientation, _) =
                        required_qcd_weyl(current_values, parents[0])?;
                    if parent_chirality != chirality || parent_orientation != orientation {
                        return Err(invalid(
                            "terminal Weyl contribution changes chirality or orientation",
                        ));
                    }
                    required_qcd_vector(current_values, parents[1])?;
                }
                if program.contributions().iter().any(|contribution| {
                    contribution.result_current_id() > current.id()
                        && contribution.parent_current_ids().contains(&current.id())
                }) {
                    return Err(invalid(
                        "an unpropagated terminal Weyl current is reused as a contribution parent",
                    ));
                }
                return Ok(QcdCurrent::Weyl {
                    chirality,
                    orientation,
                    value: None,
                });
            }
            let mut terms = Vec::new();
            for contribution in &program.contributions()[range] {
                let (kind, parents, descriptor) =
                    qcd_contribution_contract(contribution, program, templates, direct)?;
                if kind != QcdContributionKind::WeylVector(chirality) {
                    return Err(invalid(
                        "propagated Weyl current uses the wrong chiral vertex primitive",
                    ));
                }
                let (parent_chirality, parent_orientation, quark) =
                    required_qcd_weyl(current_values, parents[0])?;
                if parent_chirality != chirality || parent_orientation != orientation {
                    return Err(invalid(
                        "Weyl contribution changes chirality or line orientation",
                    ));
                }
                let vector = required_qcd_vector(current_values, parents[1])?;
                let (_, momentum_terms) =
                    qcd_current_momentum(current, representative_signs, builder)?;
                let numerator = quark_vector_weyl_numerator_with_momentum(
                    builder,
                    chirality,
                    quark,
                    vector,
                    &momentum_terms,
                )?;
                let sqrt_two = builder.kinematic(SpinorKinematicScalar::SqrtTwo)?;
                let scale = intrinsic_scale_node(descriptor, dense_parameter_slots, builder)?;
                let exact = qcd_contribution_exact_node(contribution, templates, builder)?;
                let scale = builder.product([sqrt_two, scale, exact])?;
                terms.push(linear_weyl_scale(builder, scale, &numerator)?);
            }
            let numerator = linear_weyl_sum(builder, terms)?;
            let scale = qcd_finalization_scale(
                current,
                state,
                program,
                templates,
                direct,
                dense_parameter_slots,
                representative_signs,
                builder,
            )?;
            Ok(QcdCurrent::Weyl {
                chirality,
                orientation,
                value: Some(linear_weyl_scale(builder, scale, &numerator)?),
            })
        }
        QcdStateKind::Dirac { orientation, .. } => {
            let mut terms = Vec::new();
            for contribution in &program.contributions()[range] {
                if contribution.result_current_id() != current.id() {
                    return Err(invalid("Dirac contribution belongs to the wrong current"));
                }
                let (kind, parents, descriptor) =
                    qcd_contribution_contract(contribution, program, templates, direct)?;
                let (parent_orientation, parent) = required_qcd_dirac(current_values, parents[0])?;
                if parent_orientation != orientation {
                    return Err(invalid(
                        "massive Dirac contribution changes line orientation",
                    ));
                }
                let exact = qcd_contribution_exact_node(contribution, templates, builder)?;
                let (numerator, scale) = match kind {
                    QcdContributionKind::DiracVector(certified_orientation) => {
                        if certified_orientation != orientation {
                            return Err(invalid(
                                "massive Dirac vector primitive has the wrong orientation",
                            ));
                        }
                        let vector = required_qcd_vector(current_values, parents[1])?;
                        let numerator = dirac_vector_expression(builder, parent, vector)?;
                        // Sparse vectors represent V/sqrt(2), whereas the
                        // authenticated component primitive consumes V.
                        let sqrt_two = builder.kinematic(SpinorKinematicScalar::SqrtTwo)?;
                        let intrinsic =
                            intrinsic_scale_node(descriptor, dense_parameter_slots, builder)?;
                        (numerator, builder.product([sqrt_two, intrinsic, exact])?)
                    }
                    QcdContributionKind::ChiralDiracVector(certified_orientation) => {
                        if certified_orientation != orientation {
                            return Err(invalid(
                                "chiral massive Dirac vector primitive has the wrong orientation",
                            ));
                        }
                        let vector = required_qcd_vector(current_values, parents[1])?;
                        let numerator = dirac_vector_expression(builder, parent, vector)?;
                        let chiral = descriptor.chiral_dirac_vector().ok_or_else(|| {
                            invalid("chiral Dirac-vector descriptor has no typed scales")
                        })?;
                        let left = intrinsic_scale_value_node(
                            chiral.left_scale(),
                            dense_parameter_slots,
                            builder,
                        )?;
                        let right = intrinsic_scale_value_node(
                            chiral.right_scale(),
                            dense_parameter_slots,
                            builder,
                        )?;
                        let (undotted, dotted) =
                            oriented_chiral_half_scales(certified_orientation, left, right)?;
                        let numerator = dirac_half_scale(builder, undotted, dotted, &numerator)?;
                        // Sparse vectors represent V/sqrt(2), whereas the
                        // authenticated component primitive consumes V.
                        let sqrt_two = builder.kinematic(SpinorKinematicScalar::SqrtTwo)?;
                        (numerator, builder.product([sqrt_two, exact])?)
                    }
                    QcdContributionKind::DiracScalar => {
                        let scalar = required_qcd_scalar(current_values, parents[1])?;
                        let intrinsic =
                            intrinsic_scale_node(descriptor, dense_parameter_slots, builder)?;
                        (
                            dirac_scalar_expression(builder, parent, scalar)?,
                            builder.product([intrinsic, exact])?,
                        )
                    }
                    _ => {
                        return Err(invalid(
                            "massive Dirac current uses a primitive with the wrong result type",
                        ));
                    }
                };
                terms.push(dirac_scale(builder, scale, &numerator)?);
            }
            let numerator = dirac_sum(builder, terms)?;
            if finalization
                .and_then(|finalization| finalization.propagator_template_id())
                .is_some()
            {
                let source_mass = dirac_mass_prepared_slot.ok_or_else(|| {
                    invalid("massive Dirac current has no authenticated source-mass owner")
                })?;
                let typed = qcd_massive_finalizer_contract(
                    current,
                    orientation,
                    program,
                    templates,
                    direct,
                    source_mass,
                    dirac_width_prepared_slot,
                )?;
                let dense_mass = dense_parameter_slots
                    .get(&typed.mass_prepared_parameter_slot())
                    .copied()
                    .ok_or_else(|| invalid("massive Dirac mass has no graph binding"))?;
                let dense_width = dense_parameter_slots
                    .get(&typed.width_prepared_parameter_slot())
                    .copied()
                    .ok_or_else(|| invalid("massive Dirac width has no graph binding"))?;
                let mass = builder.parameter(dense_mass)?;
                let width = builder.parameter(dense_width)?;
                let (momentum, _) = qcd_current_momentum(current, representative_signs, builder)?;
                // Particle and antiparticle component templates are dual
                // under the antisymmetric Weyl bilinear. In the raised sparse
                // representation both authenticated slash numerators use this
                // same typed action; their distinct descriptors authenticate
                // that orientation transport.
                let propagated = dirac_propagator_numerator(builder, &numerator, &momentum, mass)?;
                let denominator =
                    massive_dirac_propagator_denominator(builder, &momentum, mass, width)?;
                let reciprocal = builder.reciprocal(denominator)?;
                let runtime_scale = builder.constant(exact_binary64_scale(
                    typed.constant_real_bits(),
                    typed.constant_imag_bits(),
                )?)?;
                let final_exact = builder.constant(
                    finalization
                        .ok_or_else(|| invalid("massive Dirac finalization is absent"))?
                        .exact_factor(),
                )?;
                let scale = builder.product([reciprocal, runtime_scale, final_exact])?;
                Ok(QcdCurrent::Dirac {
                    orientation,
                    value: dirac_scale(builder, scale, &propagated)?,
                })
            } else {
                require_identity_finalizer(direct)?;
                if program.contributions().iter().any(|contribution| {
                    contribution.result_current_id() > current.id()
                        && contribution.parent_current_ids().contains(&current.id())
                }) {
                    return Err(invalid(
                        "an identity-finalized Dirac current is reused as a contribution parent",
                    ));
                }
                let exact = builder.constant(
                    finalization
                        .map(super::RecurrenceFinalization::exact_factor)
                        .unwrap_or(ExactComplexRational::ONE),
                )?;
                Ok(QcdCurrent::Dirac {
                    orientation,
                    value: dirac_scale(builder, exact, &numerator)?,
                })
            }
        }
    }
}

fn required_qcd_vector(
    values: &[Option<QcdCurrent>],
    id: u32,
) -> RusticolResult<&BispinorExpression> {
    match values.get(id as usize).and_then(Option::as_ref) {
        Some(QcdCurrent::Vector(value)) => Ok(value),
        _ => Err(invalid(format!(
            "parent current {id} is not a lowered vector current"
        ))),
    }
}

fn required_qcd_scalar(values: &[Option<QcdCurrent>], id: u32) -> RusticolResult<u32> {
    match values.get(id as usize).and_then(Option::as_ref) {
        Some(QcdCurrent::Scalar(value)) => Ok(*value),
        _ => Err(invalid(format!(
            "parent current {id} is not a lowered scalar current"
        ))),
    }
}

fn required_qcd_dirac(
    values: &[Option<QcdCurrent>],
    id: u32,
) -> RusticolResult<(CurrentOrientation, &DiracExpression)> {
    match values.get(id as usize).and_then(Option::as_ref) {
        Some(QcdCurrent::Dirac { orientation, value }) => Ok((*orientation, value)),
        _ => Err(invalid(format!(
            "parent current {id} is not a lowered massive Dirac current"
        ))),
    }
}

fn required_qcd_bivector(
    values: &[Option<QcdCurrent>],
    id: u32,
) -> RusticolResult<&BivectorExpression> {
    match values.get(id as usize).and_then(Option::as_ref) {
        Some(QcdCurrent::Bivector(value)) => Ok(value),
        _ => Err(invalid(format!(
            "parent current {id} is not a lowered antisymmetric-tensor current"
        ))),
    }
}

fn required_qcd_weyl(
    values: &[Option<QcdCurrent>],
    id: u32,
) -> RusticolResult<(SpinorChirality, CurrentOrientation, &LinearWeylExpression)> {
    match values.get(id as usize).and_then(Option::as_ref) {
        Some(QcdCurrent::Weyl {
            chirality,
            orientation,
            value: Some(value),
        }) => Ok((*chirality, *orientation, value)),
        _ => Err(invalid(format!(
            "parent current {id} is not a propagated/source Weyl current"
        ))),
    }
}

fn required_qcd_weyl_pair(
    values: &[Option<QcdCurrent>],
    parents: [u32; 2],
    particle_chirality: SpinorChirality,
) -> RusticolResult<(&LinearWeylExpression, &LinearWeylExpression)> {
    let (left_chirality, left_orientation, particle) = required_qcd_weyl(values, parents[0])?;
    let (right_chirality, right_orientation, antiparticle) = required_qcd_weyl(values, parents[1])?;
    let antiparticle_chirality = match particle_chirality {
        SpinorChirality::Positive => SpinorChirality::Negative,
        SpinorChirality::Negative => SpinorChirality::Positive,
    };
    if left_orientation != CurrentOrientation::Particle
        || right_orientation != CurrentOrientation::Antiparticle
        || left_chirality != particle_chirality
        || right_chirality != antiparticle_chirality
    {
        return Err(invalid(
            "Weyl-pair vector primitive does not have its certified particle/antiparticle chirality order",
        ));
    }
    Ok((particle, antiparticle))
}

#[allow(clippy::too_many_arguments)]
fn lower_qcd_closures(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    dense_parameter_slots: &BTreeMap<u32, u16>,
    representative_signs: &[i32],
    current_values: &[QcdCurrent],
    builder: &mut SpinorDagBuilder,
) -> RusticolResult<BTreeMap<u32, u32>> {
    let mut terms_by_destination = BTreeMap::<u32, Vec<u32>>::new();
    for closure in program.closure_terms() {
        let [left_id, right_id] = closure.parent_current_ids() else {
            return Err(invalid(format!(
                "QCD closure {} is not binary",
                closure.id()
            )));
        };
        if closure.quantum_flow_template_id().is_some() {
            return Err(invalid(format!(
                "QCD closure {} unexpectedly carries a quantum-flow evaluator",
                closure.id()
            )));
        }
        let row = templates
            .input()
            .closures
            .get(closure.closure_template_id() as usize)
            .ok_or_else(|| invalid("QCD closure template is absent"))?;
        if row.id != closure.closure_template_id() {
            return Err(invalid("QCD closure template has a noncanonical ID"));
        }
        if !template_u32_sequence(
            templates,
            row.coupling_parameter_sequence_id,
            "QCD closure coupling parameters",
        )?
        .is_empty()
        {
            return Err(invalid(
                "QCD closure unexpectedly owns a coupling parameter",
            ));
        }
        let expected_states = template_u32_sequence(
            templates,
            row.input_state_sequence_id,
            "QCD closure input states",
        )?;
        if expected_states.len() != 2
            || expected_states[0]
                != program.currents()[*left_id as usize]
                    .key()
                    .current_state_template_id()
            || expected_states[1]
                != program.currents()[*right_id as usize]
                    .key()
                    .current_state_template_id()
        {
            return Err(invalid(
                "QCD closure state order disagrees with its parents",
            ));
        }
        let evaluator = evaluator_row(
            templates,
            row.evaluator_binding_id,
            EvaluatorContractKind::Closure,
        )?;
        let descriptor = direct
            .intrinsic_descriptor(DirectExecutorRole::Closure, evaluator.id)
            .ok_or_else(|| invalid("QCD closure has no authenticated intrinsic descriptor"))?;
        validate_intrinsic_runtime_binding(templates, evaluator, descriptor, "QCD closure")?;
        if !descriptor.runtime_template().starts_with(CLOSURE_PREFIX)
            || descriptor.contract_digest().is_none()
            || descriptor.scale().is_some()
            || descriptor.chiral_dirac_vector().is_some()
            || descriptor.massive_dirac_finalizer().is_some()
            || descriptor.massive_vector_finalizer().is_some()
            || descriptor.massive_scalar_finalizer().is_some()
        {
            return Err(invalid("QCD closure intrinsic descriptor is malformed"));
        }
        let coefficients = templates.closure_component_coefficients(row.id)?;
        let chirality_relation = template_string(
            templates,
            row.chirality_relation_string_id,
            "QCD closure chirality relation",
        )?;
        let metric_signature = template_optional_string(
            templates,
            row.metric_signature_string_id,
            "QCD closure metric signature",
        )?;
        let left = current_values
            .get(*left_id as usize)
            .ok_or_else(|| invalid("QCD closure left current is absent"))?;
        let right = current_values
            .get(*right_id as usize)
            .ok_or_else(|| invalid("QCD closure right current is absent"))?;
        if let (QcdCurrent::Vector(left), QcdCurrent::Vector(right)) = (left, right) {
            if coefficients.as_slice()
                != [
                    ExactComplexRational::ONE,
                    ExactComplexRational::new(ExactRational::new(-1, 1)?, ExactRational::ZERO),
                    ExactComplexRational::new(ExactRational::new(-1, 1)?, ExactRational::ZERO),
                    ExactComplexRational::new(ExactRational::new(-1, 1)?, ExactRational::ZERO),
                ]
                || chirality_relation != "any"
                || metric_signature != Some("mostly-minus")
            {
                return Err(invalid(
                    "QCD vector closure is not the certified mostly-minus Lorentz contraction",
                ));
            }
            let contraction = bispinor_dot_expression(builder, left, right)?;
            // Both sparse vector expressions represent V/sqrt(2).
            let two = builder.constant(ExactComplexRational::new(
                ExactRational::new(2, 1)?,
                ExactRational::ZERO,
            ))?;
            let exact = builder.constant(closure.exact_factor())?;
            let node = builder.product([two, contraction, exact])?;
            terms_by_destination
                .entry(closure.target_destination_id())
                .or_default()
                .push(node);
            continue;
        }
        if let (
            QcdCurrent::Dirac {
                orientation: left_orientation,
                value: left_value,
            },
            QcdCurrent::Dirac {
                orientation: right_orientation,
                value: right_value,
            },
        ) = (left, right)
        {
            if coefficients.as_slice()
                != [
                    ExactComplexRational::ONE,
                    ExactComplexRational::ONE,
                    ExactComplexRational::ONE,
                    ExactComplexRational::ONE,
                ]
                || chirality_relation != "any"
                || metric_signature.is_some()
                || *left_orientation == *right_orientation
            {
                return Err(invalid(
                    "QCD Dirac closure is not the certified direct particle/antiparticle contraction",
                ));
            }
            let left_current = program
                .currents()
                .get(*left_id as usize)
                .ok_or_else(|| invalid("QCD Dirac closure left current is absent"))?;
            let right_current = program
                .currents()
                .get(*right_id as usize)
                .ok_or_else(|| invalid("QCD Dirac closure right current is absent"))?;
            let (terminal, particle, antiparticle) = match (
                left_current.is_source(),
                right_current.is_source(),
                *left_orientation,
                *right_orientation,
            ) {
                (false, true, CurrentOrientation::Particle, CurrentOrientation::Antiparticle) => {
                    (left_current, left_value, right_value)
                }
                (false, true, CurrentOrientation::Antiparticle, CurrentOrientation::Particle) => {
                    (left_current, right_value, left_value)
                }
                (true, false, CurrentOrientation::Particle, CurrentOrientation::Antiparticle) => {
                    (right_current, left_value, right_value)
                }
                (true, false, CurrentOrientation::Antiparticle, CurrentOrientation::Particle) => {
                    (right_current, right_value, left_value)
                }
                _ => {
                    return Err(invalid(
                        "QCD Dirac closure must join one terminal line current to its opposite-orientation source",
                    ));
                }
            };
            if qcd_optional_finalization(terminal, program)?
                .and_then(|finalization| finalization.propagator_template_id())
                .is_some()
            {
                return Err(invalid(
                    "terminal QCD Dirac current is unexpectedly propagated",
                ));
            }
            let contraction = dirac_bilinear(builder, particle, antiparticle)?;
            let exact = builder.constant(closure.exact_factor())?;
            let node = builder.product([contraction, exact])?;
            let _ = qcd_current_momentum(terminal, representative_signs, builder)?;
            terms_by_destination
                .entry(closure.target_destination_id())
                .or_default()
                .push(node);
            continue;
        }
        if coefficients.as_slice() != [ExactComplexRational::ONE, ExactComplexRational::ONE]
            || chirality_relation != "opposite"
            || metric_signature.is_some()
        {
            return Err(invalid(
                "QCD closure is not a certified vector or opposite-Weyl contraction",
            ));
        }
        let (
            terminal_id,
            terminal_chirality,
            terminal_orientation,
            source_id,
            source_chirality,
            source_orientation,
            source_value,
        ) = match (left, right) {
            (
                QcdCurrent::Weyl {
                    chirality: terminal_chirality,
                    orientation: terminal_orientation,
                    value: None,
                },
                QcdCurrent::Weyl {
                    chirality: source_chirality,
                    orientation: source_orientation,
                    value: Some(source_value),
                },
            ) => (
                *left_id,
                *terminal_chirality,
                *terminal_orientation,
                *right_id,
                *source_chirality,
                *source_orientation,
                source_value,
            ),
            (
                QcdCurrent::Weyl {
                    chirality: source_chirality,
                    orientation: source_orientation,
                    value: Some(source_value),
                },
                QcdCurrent::Weyl {
                    chirality: terminal_chirality,
                    orientation: terminal_orientation,
                    value: None,
                },
            ) => (
                *right_id,
                *terminal_chirality,
                *terminal_orientation,
                *left_id,
                *source_chirality,
                *source_orientation,
                source_value,
            ),
            _ => {
                return Err(invalid(
                    "initial QCD closure requires one terminal and one source Weyl current",
                ));
            }
        };
        if terminal_chirality == source_chirality
            || terminal_orientation != CurrentOrientation::Particle
            || source_orientation != CurrentOrientation::Antiparticle
            || !program.currents()[source_id as usize].is_source()
        {
            return Err(invalid(
                "QCD closure does not terminate one particle-oriented Weyl line on its antiparticle source",
            ));
        }
        let terminal = &program.currents()[terminal_id as usize];
        let terminal_finalization = qcd_optional_finalization(terminal, program)?;
        if terminal_finalization
            .and_then(|finalization| finalization.propagator_template_id())
            .is_some()
        {
            return Err(invalid(
                "terminal QCD Weyl current is unexpectedly propagated",
            ));
        }
        let range = terminal.contribution_range().as_usize_range(
            program.contributions().len(),
            "terminal QCD Weyl contributions",
        )?;
        let mut terminal_terms = Vec::new();
        for contribution in &program.contributions()[range] {
            let (kind, parents, contribution_descriptor) =
                qcd_contribution_contract(contribution, program, templates, direct)?;
            if kind != QcdContributionKind::WeylVector(terminal_chirality) {
                return Err(invalid("terminal QCD line uses the wrong Weyl vertex"));
            }
            let (parent_chirality, parent_orientation, quark) =
                required_qcd_weyl_option(current_values, parents[0])?;
            if parent_chirality != terminal_chirality || parent_orientation != terminal_orientation
            {
                return Err(invalid(
                    "terminal QCD vertex changes the Weyl line contract",
                ));
            }
            let vector = required_qcd_vector_option(current_values, parents[1])?;
            let bilinear = quark_vector_weyl_bilinear(
                builder,
                terminal_chirality,
                quark,
                vector,
                source_value,
            )?;
            let imaginary_unit = builder.constant(ExactComplexRational::new(
                ExactRational::ZERO,
                ExactRational::ONE,
            ))?;
            let sqrt_two = builder.kinematic(SpinorKinematicScalar::SqrtTwo)?;
            let intrinsic =
                intrinsic_scale_node(contribution_descriptor, dense_parameter_slots, builder)?;
            let exact = qcd_contribution_exact_node(contribution, templates, builder)?;
            terminal_terms.push(builder.product([
                imaginary_unit,
                sqrt_two,
                intrinsic,
                exact,
                bilinear,
            ])?);
        }
        let terminal_sum = builder.sum(terminal_terms)?;
        let terminal_exact = builder.constant(
            terminal_finalization
                .map(super::RecurrenceFinalization::exact_factor)
                .unwrap_or(ExactComplexRational::ONE),
        )?;
        let closure_exact = builder.constant(closure.exact_factor())?;
        let node = builder.product([terminal_sum, terminal_exact, closure_exact])?;
        // Force momentum authentication for the terminal even though the
        // identity finalizer does not otherwise consume it.
        let _ = qcd_current_momentum(terminal, representative_signs, builder)?;
        terms_by_destination
            .entry(closure.target_destination_id())
            .or_default()
            .push(node);
    }
    terms_by_destination
        .into_iter()
        .map(|(destination, terms)| Ok((destination, builder.sum(terms)?)))
        .collect()
}

fn required_qcd_vector_option(
    values: &[QcdCurrent],
    id: u32,
) -> RusticolResult<&BispinorExpression> {
    match values.get(id as usize) {
        Some(QcdCurrent::Vector(value)) => Ok(value),
        _ => Err(invalid(format!("parent current {id} is not a vector"))),
    }
}

fn required_qcd_weyl_option(
    values: &[QcdCurrent],
    id: u32,
) -> RusticolResult<(SpinorChirality, CurrentOrientation, &LinearWeylExpression)> {
    match values.get(id as usize) {
        Some(QcdCurrent::Weyl {
            chirality,
            orientation,
            value: Some(value),
        }) => Ok((*chirality, *orientation, value)),
        _ => Err(invalid(format!(
            "parent current {id} is not a materialized Weyl current"
        ))),
    }
}

fn add_qcd_roots(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    program: &RecurrenceProgram,
    source_count: usize,
    destination_nodes: &BTreeMap<u32, u32>,
    builder: &mut SpinorDagBuilder,
) -> RusticolResult<()> {
    let replay = match program.replay_targets() {
        [] => None,
        [replay] => Some(replay),
        _ => {
            return Err(invalid(
                "one QCD payload cannot contain multiple replay targets",
            ));
        }
    };
    if replay.is_none() {
        let sectors = program
            .amplitude_destinations()
            .iter()
            .map(|destination| destination.target_sector_id())
            .collect::<BTreeSet<_>>();
        if sectors.len() != 1 {
            return Err(invalid(
                "zero-target QCD replay is only unambiguous for one selected sector",
            ));
        }
    }
    let resolved_by_id = program
        .resolved_helicities()
        .iter()
        .map(|helicity| (helicity.id(), helicity))
        .collect::<BTreeMap<_, _>>();
    let transported = if let Some(replay) = replay {
        qcd_replay_helicity_map(program, replay, source_count)?
    } else {
        program
            .resolved_helicities()
            .iter()
            .map(|helicity| (helicity.id(), helicity.id()))
            .collect()
    };
    let multiplicity = if authenticated
        .process()
        .semantic_identity()
        .extension_digests()
        .contains_key(GLOBAL_FLIP_EXTENSION)
    {
        2
    } else {
        1
    };
    for destination in program.amplitude_destinations() {
        if let Some(replay) = replay
            && destination.target_sector_id() != replay.materialized_sector_id()
        {
            return Err(invalid(
                "QCD amplitude destination does not belong to the replay representative",
            ));
        }
        let representative_helicity = destination
            .target_helicity_id()
            .ok_or_else(|| invalid("QCD amplitude destination has no resolved helicity"))?;
        let public_helicity_id = transported
            .get(&representative_helicity)
            .copied()
            .ok_or_else(|| invalid("QCD replay helicity transport is incomplete"))?;
        let public_helicity = resolved_by_id
            .get(&public_helicity_id)
            .copied()
            .ok_or_else(|| invalid("transported QCD resolved helicity is absent"))?;
        if public_helicity.public_helicities().len() != source_count {
            return Err(invalid("QCD resolved helicity has the wrong source width"));
        }
        let helicities = qcd_graph_helicities(
            public_helicity.public_helicities(),
            replay.map(|replay| replay.source_slot_permutation()),
        )?;
        let amplitude = destination_nodes
            .get(&destination.id())
            .copied()
            .ok_or_else(|| invalid("QCD amplitude destination expression is absent"))?;
        let amplitude = if let Some(replay) = replay {
            let phase = builder.constant(replay.amplitude_factor())?;
            builder.product([phase, amplitude])?
        } else {
            amplitude
        };
        builder.add_root_with_multiplicity(helicities, amplitude, multiplicity)?;
    }
    if destination_nodes.len() != program.amplitude_destinations().len() {
        return Err(invalid(
            "QCD closure destinations are not covered exactly once",
        ));
    }
    Ok(())
}

fn qcd_graph_helicities(
    public_helicities: &[i32],
    representative_to_public: Option<&[u32]>,
) -> RusticolResult<Vec<i8>> {
    let graph_helicities = if let Some(permutation) = representative_to_public {
        if permutation.len() != public_helicities.len() {
            return Err(invalid(
                "QCD replay helicity permutation has the wrong width",
            ));
        }
        permutation
            .iter()
            .copied()
            .map(|public_slot| {
                public_helicities
                    .get(public_slot as usize)
                    .copied()
                    .ok_or_else(|| invalid("QCD replay public helicity slot is out of bounds"))
            })
            .collect::<RusticolResult<Vec<_>>>()?
    } else {
        public_helicities.to_vec()
    };
    graph_helicities
        .into_iter()
        .map(|helicity| {
            i8::try_from(helicity)
                .map_err(|_| invalid("QCD public helicity exceeds the spinor DAG domain"))
        })
        .collect()
}

fn qcd_replay_helicity_map(
    program: &RecurrenceProgram,
    replay: &super::RecurrenceReplayTarget,
    source_count: usize,
) -> RusticolResult<BTreeMap<u32, u32>> {
    if replay.source_slot_permutation().len() != source_count {
        return Err(invalid("QCD replay source permutation has the wrong width"));
    }
    let mut source_template_by_state = BTreeMap::<(u32, u32), u32>::new();
    let mut source_state_by_template = BTreeMap::<(u32, u32), u32>::new();
    for current in program
        .currents()
        .iter()
        .filter(|current| current.is_source())
    {
        let [slot] = current.key().support_source_slots() else {
            return Err(invalid("QCD replay source has non-singleton ancestry"));
        };
        let template = match current.key().source_binding() {
            CurrentSourceBinding::FixedTemplate(id) => *id,
            _ => return Err(invalid("QCD replay source has no fixed template")),
        };
        let [assignment] = current.key().helicity_identity().local_source_states() else {
            return Err(invalid(
                "QCD replay source has invalid local helicity ancestry",
            ));
        };
        if assignment.source_slot() != *slot {
            return Err(invalid("QCD replay source ancestry uses the wrong slot"));
        }
        source_template_by_state.insert((*slot, assignment.state_index()), template);
        source_state_by_template.insert((*slot, template), assignment.state_index());
    }
    let resolved_by_states = program
        .resolved_helicities()
        .iter()
        .map(|helicity| (helicity.source_states().to_vec(), helicity.id()))
        .collect::<BTreeMap<_, _>>();
    let mut result = BTreeMap::new();
    for helicity in program.resolved_helicities() {
        let mut mapped = vec![None; source_count];
        for assignment in helicity.source_states().iter().copied() {
            let representative_slot = assignment.source_slot();
            let template = source_template_by_state
                .get(&(representative_slot, assignment.state_index()))
                .copied()
                .ok_or_else(|| invalid("QCD replay source state has no fixed template"))?;
            let target_slot = replay.source_slot_permutation()[representative_slot as usize];
            let target_state = source_state_by_template
                .get(&(target_slot, template))
                .copied()
                .ok_or_else(|| invalid("QCD replay cannot transport a source template"))?;
            let target = mapped
                .get_mut(target_slot as usize)
                .ok_or_else(|| invalid("QCD replay target slot is out of bounds"))?;
            if target
                .replace(SourceStateAssignment::new(target_slot, target_state))
                .is_some()
            {
                return Err(invalid("QCD replay source mapping is not bijective"));
            }
        }
        let mapped = mapped
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid("QCD replay does not map every source state"))?;
        let mapped_id = resolved_by_states
            .get(&mapped)
            .copied()
            .ok_or_else(|| invalid("QCD replay maps outside resolved helicity coverage"))?;
        result.insert(helicity.id(), mapped_id);
    }
    Ok(result)
}

fn lower_scalar_program(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    source_count: usize,
    prepared_parameter_count: u32,
) -> RusticolResult<SpinorDagPayloadV3> {
    if program.strategy() != RecurrenceStrategy::TopologyReplay {
        return Err(invalid("the scalar slice requires topology replay"));
    }
    validate_identity_finalizations(program, direct)?;

    let momentum_count = u16::try_from(source_count)
        .map_err(|_| invalid("external source count exceeds the spinor DAG domain"))?;
    if source_count < 2 {
        return Err(invalid("the scalar graph requires at least two sources"));
    }

    let prepared_slots = authenticated_parameter_slots(&program, templates, direct)?;
    let parameter_count = u16::try_from(prepared_slots.len())
        .map_err(|_| invalid("dense graph parameter count exceeds u16"))?;
    let dense_parameter_slots = prepared_slots
        .iter()
        .copied()
        .enumerate()
        .map(|(dense, prepared)| {
            u16::try_from(dense)
                .map(|dense| (prepared, dense))
                .map_err(|_| invalid("dense graph parameter index exceeds u16"))
        })
        .collect::<RusticolResult<BTreeMap<_, _>>>()?;
    let mut builder = SpinorDagBuilder::new_with_parameters(momentum_count, parameter_count)?;
    let mut current_nodes = vec![None; program.currents().len()];
    let mut representative_signs = vec![None; source_count];

    for current in program.currents() {
        let current_index = usize::try_from(current.id())
            .map_err(|_| invalid("semantic current ID exceeds usize"))?;
        if current_index >= current_nodes.len() {
            return Err(invalid("semantic current ID is out of bounds"));
        }
        require_scalar_state(templates, current.key().current_state_template_id())?;
        let node = if current.is_source() {
            lower_scalar_source(
                current,
                templates,
                direct,
                &mut builder,
                &mut representative_signs,
            )?
        } else {
            let range = current.contribution_range().as_usize_range(
                program.contributions().len(),
                "scalar current contributions",
            )?;
            if range.is_empty() {
                return Err(invalid(format!(
                    "non-source current {} has no contributions",
                    current.id()
                )));
            }
            let terms = program.contributions()[range]
                .iter()
                .map(|contribution| {
                    if contribution.result_current_id() != current.id() {
                        return Err(invalid(format!(
                            "contribution {} does not belong to current {}",
                            contribution.id(),
                            current.id()
                        )));
                    }
                    lower_scalar_contribution(
                        contribution,
                        program,
                        templates,
                        direct,
                        &dense_parameter_slots,
                        &current_nodes,
                        &mut builder,
                    )
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            builder.sum(terms)?
        };
        if current_nodes[current_index].replace(node).is_some() {
            return Err(invalid(format!(
                "semantic current {} was lowered more than once",
                current.id()
            )));
        }
    }
    if current_nodes.iter().any(Option::is_none) {
        return Err(invalid("not every semantic current was lowered"));
    }
    let representative_signs = representative_signs
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| invalid("source currents do not cover every graph source exactly once"))?;

    let destination_nodes =
        lower_scalar_closures(program, templates, direct, &current_nodes, &mut builder)?;
    let [replay] = program.replay_targets() else {
        return Err(invalid(
            "the scalar slice requires exactly one authenticated replay target",
        ));
    };
    let destination = program
        .amplitude_destinations()
        .iter()
        .find(|destination| destination.target_sector_id() == replay.materialized_sector_id())
        .ok_or_else(|| invalid("replay representative has no amplitude destination"))?;
    if program.amplitude_destinations().len() != 1 || destination_nodes.len() != 1 {
        return Err(invalid(
            "the scalar slice requires exactly one amplitude destination",
        ));
    }
    let amplitude = *destination_nodes
        .get(&destination.id())
        .ok_or_else(|| invalid("amplitude destination expression is absent"))?;
    let phase = builder.constant(replay.amplitude_factor())?;
    let amplitude = builder.product([phase, amplitude])?;

    let [resolved] = program.resolved_helicities() else {
        return Err(invalid(
            "the scalar slice requires exactly one resolved helicity",
        ));
    };
    if resolved.public_helicities().len() != source_count
        || resolved.public_helicities().iter().any(|value| *value != 0)
    {
        return Err(invalid(
            "MomentumOnly scalar sources require public helicity zero",
        ));
    }
    if destination.target_helicity_id() != Some(resolved.id()) {
        return Err(invalid(
            "amplitude destination does not own the sole resolved scalar helicity",
        ));
    }
    builder.add_root(vec![0_i8; source_count], amplitude)?;
    let dag = builder.finish()?;

    if replay.source_slot_permutation().len() != source_count
        || replay.source_momentum_signs().len() != source_count
    {
        return Err(invalid("replay source mapping has the wrong width"));
    }
    let source_inputs = replay
        .source_slot_permutation()
        .iter()
        .copied()
        .zip(replay.source_momentum_signs().iter().copied())
        .zip(representative_signs)
        .map(|((public_slot, replay_sign), representative_sign)| {
            let public_slot = u16::try_from(public_slot)
                .map_err(|_| invalid("public source slot exceeds u16"))?;
            let sign = representative_sign
                .checked_mul(replay_sign)
                .ok_or_else(|| invalid("source momentum sign overflows i32"))?;
            let sign =
                i8::try_from(sign).map_err(|_| invalid("source momentum sign exceeds i8"))?;
            SpinorSourceInputBinding::new(public_slot, sign, SpinorSourceInputKind::MomentumOnly)
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let parameter_bindings = prepared_slots
        .into_iter()
        .map(SpinorPreparedParameterBinding::new)
        .collect();
    SpinorDagPayloadV3::new(
        dag,
        source_inputs,
        prepared_parameter_count,
        parameter_bindings,
    )
}

fn authenticated_parameter_slots(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
) -> RusticolResult<Vec<u32>> {
    let mut slots = BTreeSet::new();
    for contribution in program.contributions() {
        let transition = transition_row(templates, contribution.key().transition_template_id())?;
        let descriptor = direct
            .intrinsic_descriptor(
                DirectExecutorRole::Contribution,
                transition.evaluator_binding_id,
            )
            .ok_or_else(|| {
                invalid(format!(
                    "transition {} has no authenticated intrinsic descriptor",
                    transition.id
                ))
            })?;
        require_scalar_product_descriptor(descriptor)?;
        if let Some(slot) = descriptor
            .scale()
            .and_then(|scale| scale.prepared_parameter_slot())
        {
            slots.insert(slot);
        }
        validate_transition_parameter_owner(templates, transition, descriptor)?;
    }
    Ok(slots.into_iter().collect())
}

fn validate_identity_finalizations(
    program: &RecurrenceProgram,
    direct: &PreparedDirectExecutorCatalog,
) -> RusticolResult<()> {
    direct.resolve_identity_finalizer()?;
    let descriptor = direct
        .intrinsic_descriptor_by_key(super::PreparedDirectExecutorKey::IdentityFinalizer)
        .ok_or_else(|| invalid("scalar currents have no authenticated identity finalizer"))?;
    if descriptor.runtime_template() != IDENTITY_FINALIZER
        || descriptor.contract_digest().is_some()
        || descriptor.scale().is_some()
        || descriptor.chiral_dirac_vector().is_some()
        || descriptor.massive_dirac_finalizer().is_some()
        || descriptor.massive_vector_finalizer().is_some()
        || descriptor.massive_scalar_finalizer().is_some()
    {
        return Err(invalid("scalar identity-finalizer descriptor is malformed"));
    }
    for finalization in program.finalizations() {
        if finalization.propagator_template_id().is_some()
            || finalization.exact_factor() != ExactComplexRational::ONE
        {
            return Err(invalid(format!(
                "finalization {} is not an exact identity",
                finalization.id()
            )));
        }
    }
    Ok(())
}

fn lower_scalar_source(
    current: &super::RecurrenceCurrent,
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    builder: &mut SpinorDagBuilder,
    representative_signs: &mut [Option<i32>],
) -> RusticolResult<u32> {
    let source_template_id = match current.key().source_binding() {
        CurrentSourceBinding::FixedTemplate(id) => *id,
        _ => {
            return Err(invalid(format!(
                "source current {} does not use one fixed template",
                current.id()
            )));
        }
    };
    let source = templates
        .input()
        .sources
        .get(source_template_id as usize)
        .ok_or_else(|| invalid(format!("source template {source_template_id} is absent")))?;
    if source.id != source_template_id || source.helicity != 0 || source.spin_state != 0 {
        return Err(invalid(format!(
            "source template {source_template_id} is not a scalar helicity-zero source"
        )));
    }
    require_scalar_state(templates, source.state_template_id)?;
    let family = template_string(
        templates,
        source.wavefunction_family_string_id,
        "source wavefunction family",
    )?;
    if family != "scalar" {
        return Err(invalid(format!(
            "source template {source_template_id} uses unsupported family {family:?}"
        )));
    }
    let evaluator = evaluator_row(
        templates,
        source.evaluator_binding_id,
        EvaluatorContractKind::Source,
    )?;
    let descriptor = direct
        .intrinsic_descriptor(DirectExecutorRole::Source, evaluator.id)
        .ok_or_else(|| invalid("scalar source has no authenticated direct intrinsic"))?;
    if !descriptor
        .runtime_template()
        .starts_with(SCALAR_SOURCE_PREFIX)
        || descriptor.contract_digest().is_none()
        || descriptor.scale().is_some()
        || descriptor.chiral_dirac_vector().is_some()
        || descriptor.massive_dirac_finalizer().is_some()
        || descriptor.massive_vector_finalizer().is_some()
        || descriptor.massive_scalar_finalizer().is_some()
    {
        return Err(invalid(format!(
            "source evaluator {} is not the authenticated scalar source primitive",
            evaluator.id
        )));
    }

    let [support_slot] = current.key().support_source_slots() else {
        return Err(invalid(format!(
            "source current {} has non-singleton source support",
            current.id()
        )));
    };
    let [momentum] = current.key().momentum().terms() else {
        return Err(invalid(format!(
            "source current {} has a non-elementary momentum",
            current.id()
        )));
    };
    if momentum.source_slot != *support_slot || !matches!(momentum.coefficient, -1 | 1) {
        return Err(invalid(format!(
            "source current {} has an invalid signed momentum binding",
            current.id()
        )));
    }
    let slot = usize::try_from(*support_slot).map_err(|_| invalid("source slot exceeds usize"))?;
    let destination = representative_signs
        .get_mut(slot)
        .ok_or_else(|| invalid("source slot is outside the public source domain"))?;
    if destination.replace(momentum.coefficient).is_some() {
        return Err(invalid(format!(
            "source slot {support_slot} has more than one scalar source current"
        )));
    }
    builder.constant(
        current
            .source_exact_factor()
            .ok_or_else(|| invalid("scalar source current has no exact source factor"))?,
    )
}

#[allow(clippy::too_many_arguments)]
fn lower_scalar_contribution(
    contribution: &super::RecurrenceContribution,
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    dense_parameter_slots: &BTreeMap<u32, u16>,
    current_nodes: &[Option<u32>],
    builder: &mut SpinorDagBuilder,
) -> RusticolResult<u32> {
    let [left_id, right_id] = contribution.parent_current_ids() else {
        return Err(invalid(format!(
            "contribution {} is not binary",
            contribution.id()
        )));
    };
    let transition = transition_row(templates, contribution.key().transition_template_id())?;
    let descriptor = direct
        .intrinsic_descriptor(
            DirectExecutorRole::Contribution,
            transition.evaluator_binding_id,
        )
        .ok_or_else(|| invalid("scalar contribution has no authenticated intrinsic"))?;
    require_scalar_product_descriptor(descriptor)?;
    validate_transition_parameter_owner(templates, transition, descriptor)?;
    let expected_states = template_u32_sequence(
        templates,
        transition.input_state_sequence_id,
        "transition input states",
    )?;
    if expected_states.len() != 2
        || expected_states
            .iter()
            .copied()
            .any(|state| require_scalar_state(templates, state).is_err())
    {
        return Err(invalid(format!(
            "transition {} does not consume two scalar states",
            transition.id
        )));
    }
    require_scalar_state(templates, transition.result_state_template_id)?;

    let left = required_current_node(current_nodes, *left_id)?;
    let right = required_current_node(current_nodes, *right_id)?;
    if program.currents()[*left_id as usize].id() >= contribution.result_current_id()
        || program.currents()[*right_id as usize].id() >= contribution.result_current_id()
    {
        return Err(invalid("scalar contribution is not topologically ordered"));
    }
    let scale = descriptor
        .scale()
        .ok_or_else(|| invalid("scalar product descriptor has no exact scale"))?;
    let constant = exact_binary64_scale(scale.constant_real_bits(), scale.constant_imag_bits())?;
    let mut factors = vec![left, right, builder.constant(constant)?];
    if let Some(prepared_slot) = scale.prepared_parameter_slot() {
        let dense = dense_parameter_slots
            .get(&prepared_slot)
            .copied()
            .ok_or_else(|| invalid("prepared parameter has no dense graph binding"))?;
        factors.push(builder.parameter(dense)?);
    }
    factors.push(builder.constant(normalized_contribution_exact_factor(
        contribution,
        templates,
    )?)?);
    builder.product(factors)
}

fn lower_scalar_closures(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    current_nodes: &[Option<u32>],
    builder: &mut SpinorDagBuilder,
) -> RusticolResult<BTreeMap<u32, u32>> {
    let mut terms_by_destination = BTreeMap::<u32, Vec<u32>>::new();
    for closure in program.closure_terms() {
        let [left_id, right_id] = closure.parent_current_ids() else {
            return Err(invalid(format!("closure {} is not binary", closure.id())));
        };
        if closure.quantum_flow_template_id().is_some() {
            return Err(invalid(format!(
                "closure {} unexpectedly carries a quantum-flow evaluator",
                closure.id()
            )));
        }
        let row = templates
            .input()
            .closures
            .get(closure.closure_template_id() as usize)
            .ok_or_else(|| invalid("closure template is absent"))?;
        if row.id != closure.closure_template_id() {
            return Err(invalid("closure template has a noncanonical ID"));
        }
        if !template_u32_sequence(
            templates,
            row.coupling_parameter_sequence_id,
            "closure coupling parameters",
        )?
        .is_empty()
        {
            return Err(invalid(format!(
                "closure {} unexpectedly owns coupling parameters",
                closure.id()
            )));
        }
        let evaluator = evaluator_row(
            templates,
            row.evaluator_binding_id,
            EvaluatorContractKind::Closure,
        )?;
        let descriptor = direct
            .intrinsic_descriptor(DirectExecutorRole::Closure, evaluator.id)
            .ok_or_else(|| invalid("scalar closure has no authenticated intrinsic"))?;
        if !descriptor.runtime_template().starts_with(CLOSURE_PREFIX)
            || descriptor.contract_digest().is_none()
            || descriptor.scale().is_some()
            || descriptor.chiral_dirac_vector().is_some()
            || descriptor.massive_dirac_finalizer().is_some()
            || descriptor.massive_vector_finalizer().is_some()
            || descriptor.massive_scalar_finalizer().is_some()
        {
            return Err(invalid(format!(
                "closure evaluator {} is not the authenticated scalar reduction primitive",
                evaluator.id
            )));
        }
        let input_states = template_u32_sequence(
            templates,
            row.input_state_sequence_id,
            "closure input states",
        )?;
        if input_states.len() != 2 {
            return Err(invalid(format!(
                "closure {} does not consume two currents",
                closure.id()
            )));
        }
        for state in input_states {
            require_scalar_state(templates, *state)?;
        }
        let component_factors =
            templates.closure_component_coefficients(closure.closure_template_id())?;
        let [component_factor] = component_factors.as_slice() else {
            return Err(invalid(format!(
                "closure {} does not have one scalar component coefficient",
                closure.id()
            )));
        };
        let factors = [
            required_current_node(current_nodes, *left_id)?,
            required_current_node(current_nodes, *right_id)?,
            builder.constant(closure.exact_factor())?,
            builder.constant(*component_factor)?,
        ];
        let node = builder.product(factors)?;
        terms_by_destination
            .entry(closure.target_destination_id())
            .or_default()
            .push(node);
    }
    terms_by_destination
        .into_iter()
        .map(|(destination, terms)| Ok((destination, builder.sum(terms)?)))
        .collect()
}

fn validate_transition_parameter_owner(
    templates: &ValidatedRecurrenceTemplateInput,
    transition: &super::template::TransitionRow,
    descriptor: &super::PreparedDirectIntrinsicDescriptor,
) -> RusticolResult<()> {
    let parameters = template_u32_sequence(
        templates,
        transition.coupling_parameter_sequence_id,
        "transition coupling parameters",
    )?;
    let output_factor_source = OutputFactorSource::try_from(transition.output_factor_source)?;
    if let Some(chiral) = descriptor.chiral_dirac_vector() {
        let expected_slots = [chiral.left_scale(), chiral.right_scale()]
            .into_iter()
            .filter_map(|scale| scale.prepared_parameter_slot())
            .collect::<BTreeSet<_>>();
        if parameters.len() != expected_slots.len() {
            return Err(invalid(format!(
                "transition {} coupling ownership disagrees with its authenticated chiral scales",
                transition.id
            )));
        }
        let mut actual_slots = BTreeSet::new();
        for template_id in parameters.iter().copied() {
            let parameter = templates
                .input()
                .parameters
                .get(template_id as usize)
                .ok_or_else(|| invalid("transition parameter template is absent"))?;
            if parameter.id != template_id
                || !expected_slots.contains(&parameter.prepared_parameter_id)
                || !actual_slots.insert(parameter.prepared_parameter_id)
            {
                return Err(invalid(format!(
                    "transition {} does not uniquely own its authenticated chiral prepared slots",
                    transition.id
                )));
            }
            match output_factor_source {
                OutputFactorSource::None => {
                    let _ = ParameterValueType::try_from(parameter.value_type)?;
                }
                OutputFactorSource::CouplingReal | OutputFactorSource::CouplingImag => {
                    validate_transition_output_factor_parameter(
                        templates,
                        transition,
                        parameter,
                        output_factor_source,
                    )?;
                }
            }
        }
        if actual_slots != expected_slots
            || (output_factor_source != OutputFactorSource::None && parameters.len() != 1)
        {
            return Err(invalid(format!(
                "transition {} coupling ownership disagrees with its authenticated chiral scales",
                transition.id
            )));
        }
        return Ok(());
    }
    match descriptor
        .scale()
        .and_then(|scale| scale.prepared_parameter_slot())
    {
        None if parameters.is_empty() && output_factor_source == OutputFactorSource::None => Ok(()),
        Some(prepared_slot) if parameters.len() == 1 => {
            let template_id = parameters[0];
            let parameter = templates
                .input()
                .parameters
                .get(template_id as usize)
                .ok_or_else(|| invalid("transition parameter template is absent"))?;
            if parameter.id != template_id || parameter.prepared_parameter_id != prepared_slot {
                return Err(invalid(format!(
                    "transition {} does not own authenticated prepared slot {prepared_slot}",
                    transition.id
                )));
            }
            match output_factor_source {
                OutputFactorSource::None => {
                    // Prepared real parameters occupy the real plane of the
                    // same split-complex runtime domain.  The authenticated
                    // intrinsic scale may therefore bind either a real or a
                    // complex prepared slot without changing the DAG
                    // operation.  Parsing the enum still rejects every
                    // unsupported value type at this boundary.
                    let _ = ParameterValueType::try_from(parameter.value_type)?;
                }
                OutputFactorSource::CouplingReal | OutputFactorSource::CouplingImag => {
                    validate_transition_output_factor_parameter(
                        templates,
                        transition,
                        parameter,
                        output_factor_source,
                    )?;
                }
            }
            Ok(())
        }
        _ => Err(invalid(format!(
            "transition {} coupling ownership disagrees with its authenticated intrinsic scale",
            transition.id
        ))),
    }
}

fn validate_transition_output_factor_parameter(
    templates: &ValidatedRecurrenceTemplateInput,
    transition: &super::template::TransitionRow,
    parameter: &super::template::ParameterRow,
    output_factor_source: OutputFactorSource,
) -> RusticolResult<()> {
    if ParameterValueType::try_from(parameter.value_type)? != ParameterValueType::Real
        || ParameterKind::try_from(parameter.kind)? != ParameterKind::External
        || parameter.mutable != 1
    {
        return Err(invalid(format!(
            "transition {} output-factor component is not one mutable external real parameter",
            transition.id
        )));
    }
    let binding = template_exact_factor(
        templates,
        transition.binding_coupling_factor_id,
        "transition binding coupling",
    )?;
    let component = match output_factor_source {
        OutputFactorSource::CouplingReal => binding.real(),
        OutputFactorSource::CouplingImag => binding.imag(),
        OutputFactorSource::None => {
            return Err(invalid(
                "transition output-factor validation received no output factor",
            ));
        }
    };
    if component == ExactRational::ZERO
        || template_exact_factor(
            templates,
            parameter.default_factor_id,
            "output-factor parameter default",
        )? != ExactComplexRational::new(component, ExactRational::ZERO)
    {
        return Err(invalid(format!(
            "transition {} output-factor parameter default disagrees with its authenticated binding component",
            transition.id
        )));
    }
    Ok(())
}

fn normalized_contribution_exact_factor(
    contribution: &super::RecurrenceContribution,
    templates: &ValidatedRecurrenceTemplateInput,
) -> RusticolResult<ExactComplexRational> {
    let transition = transition_row(templates, contribution.key().transition_template_id())?;
    let binding = template_exact_factor(
        templates,
        transition.binding_coupling_factor_id,
        "transition binding coupling",
    )?;
    let component = match OutputFactorSource::try_from(transition.output_factor_source)? {
        OutputFactorSource::None => return Ok(contribution.exact_factor()),
        OutputFactorSource::CouplingReal => binding.real(),
        OutputFactorSource::CouplingImag => binding.imag(),
    };
    contribution
        .exact_factor()
        .checked_div(ExactComplexRational::new(component, ExactRational::ZERO))
}

fn qcd_contribution_exact_node(
    contribution: &super::RecurrenceContribution,
    templates: &ValidatedRecurrenceTemplateInput,
    builder: &mut SpinorDagBuilder,
) -> RusticolResult<u32> {
    builder.constant(normalized_contribution_exact_factor(
        contribution,
        templates,
    )?)
}

fn template_exact_factor(
    templates: &ValidatedRecurrenceTemplateInput,
    factor_id: u32,
    label: &str,
) -> RusticolResult<ExactComplexRational> {
    if factor_id == MISSING_U32 {
        return Err(invalid(format!("{label} is missing")));
    }
    let factor = templates
        .input()
        .exact_factors
        .get(factor_id as usize)
        .ok_or_else(|| invalid(format!("{label} factor {factor_id} is absent")))?;
    if factor.id != factor_id {
        return Err(invalid(format!(
            "{label} factor {factor_id} has a noncanonical ID"
        )));
    }
    let parse = |string_id, part: &str| -> RusticolResult<i128> {
        template_string(templates, string_id, &format!("{label} {part}"))?
            .parse::<i128>()
            .map_err(|error| invalid(format!("{label} {part} is not an exact integer: {error}")))
    };
    Ok(ExactComplexRational::new(
        ExactRational::new(
            parse(factor.real_numerator_string_id, "real numerator")?,
            parse(factor.real_denominator_string_id, "real denominator")?,
        )?,
        ExactRational::new(
            parse(factor.imag_numerator_string_id, "imaginary numerator")?,
            parse(factor.imag_denominator_string_id, "imaginary denominator")?,
        )?,
    ))
}

fn require_scalar_product_descriptor(
    descriptor: &super::PreparedDirectIntrinsicDescriptor,
) -> RusticolResult<()> {
    if descriptor.runtime_template() != SCALAR_PRODUCT_TEMPLATE
        || descriptor
            .contract_digest()
            .is_none_or(|digest| digest.to_string() != SCALAR_PRODUCT_CONTRACT)
        || descriptor.scale().is_none()
        || descriptor.chiral_dirac_vector().is_some()
        || descriptor.massive_dirac_finalizer().is_some()
        || descriptor.massive_vector_finalizer().is_some()
        || descriptor.massive_scalar_finalizer().is_some()
    {
        return Err(invalid(format!(
            "unsupported contribution primitive {:?}",
            descriptor.runtime_template()
        )));
    }
    Ok(())
}

fn exact_binary64_scale(real_bits: u64, imag_bits: u64) -> RusticolResult<ExactComplexRational> {
    let real = f64::from_bits(real_bits);
    let imag = f64::from_bits(imag_bits);
    if real == 0.0 && imag == 0.0 {
        return Err(invalid("authenticated intrinsic scale is zero"));
    }
    Ok(ExactComplexRational::new(
        ExactRational::from_f64_exact(real)?,
        ExactRational::from_f64_exact(imag)?,
    ))
}

fn required_current_node(current_nodes: &[Option<u32>], current_id: u32) -> RusticolResult<u32> {
    current_nodes
        .get(current_id as usize)
        .copied()
        .flatten()
        .ok_or_else(|| invalid(format!("parent current {current_id} has not been lowered")))
}

fn transition_row(
    templates: &ValidatedRecurrenceTemplateInput,
    transition_id: u32,
) -> RusticolResult<&super::template::TransitionRow> {
    let transition = templates
        .input()
        .transitions
        .get(transition_id as usize)
        .ok_or_else(|| invalid(format!("transition template {transition_id} is absent")))?;
    if transition.id != transition_id {
        return Err(invalid(format!(
            "transition template {transition_id} has noncanonical ID {}",
            transition.id
        )));
    }
    Ok(transition)
}

fn require_scalar_state(
    templates: &ValidatedRecurrenceTemplateInput,
    state_id: u32,
) -> RusticolResult<()> {
    let state = templates
        .input()
        .current_states
        .get(state_id as usize)
        .ok_or_else(|| invalid(format!("current-state template {state_id} is absent")))?;
    if state.id != state_id
        || state.dimension != 1
        || ParticleStatistics::try_from(state.statistics)? != ParticleStatistics::Boson
    {
        return Err(invalid(format!(
            "current-state template {state_id} is not a one-component boson"
        )));
    }
    Ok(())
}

fn evaluator_row(
    templates: &ValidatedRecurrenceTemplateInput,
    evaluator_id: u32,
    expected_contract: EvaluatorContractKind,
) -> RusticolResult<&super::template::EvaluatorBindingRow> {
    if evaluator_id == MISSING_U32 {
        return Err(invalid("intrinsic evaluator binding is missing"));
    }
    let evaluator = templates
        .input()
        .evaluator_bindings
        .get(evaluator_id as usize)
        .ok_or_else(|| invalid(format!("evaluator binding {evaluator_id} is absent")))?;
    if evaluator.id != evaluator_id
        || EvaluatorContractKind::try_from(evaluator.contract_kind)? != expected_contract
    {
        return Err(invalid(format!(
            "evaluator binding {evaluator_id} has the wrong contract"
        )));
    }
    Ok(evaluator)
}

fn validate_intrinsic_runtime_binding(
    templates: &ValidatedRecurrenceTemplateInput,
    evaluator: &super::template::EvaluatorBindingRow,
    descriptor: &super::PreparedDirectIntrinsicDescriptor,
    label: &str,
) -> RusticolResult<()> {
    // A model-owned prepared kernel deliberately has no runtime-template
    // string in the semantic catalog.  Its exact-algebra certification and
    // chosen Rust primitive live in the prepared direct descriptor keyed by
    // this evaluator.  Compiler-owned Rusticol callables retain their string
    // on both sides and must agree.
    if EvaluatorCallableKind::try_from(evaluator.callable_kind)?
        == EvaluatorCallableKind::RusticolTemplate
        && template_string(
            templates,
            evaluator.runtime_template_string_id,
            &format!("{label} runtime template"),
        )? != descriptor.runtime_template()
    {
        return Err(invalid(format!(
            "{label} evaluator {} runtime template disagrees with its prepared intrinsic",
            evaluator.id
        )));
    }
    Ok(())
}

fn template_u32_sequence<'a>(
    templates: &'a ValidatedRecurrenceTemplateInput,
    sequence_id: u32,
    label: &str,
) -> RusticolResult<&'a [u32]> {
    let input = templates.input();
    let range = input
        .u32_sequence_ranges
        .get(sequence_id as usize)
        .ok_or_else(|| invalid(format!("{label} sequence {sequence_id} is absent")))?;
    if range.id != sequence_id {
        return Err(invalid(format!(
            "{label} sequence {sequence_id} has noncanonical ID {}",
            range.id
        )));
    }
    let range = range
        .range
        .as_usize_range(input.u32_sequence_values.len(), label)?;
    Ok(&input.u32_sequence_values[range])
}

fn template_string<'a>(
    templates: &'a ValidatedRecurrenceTemplateInput,
    string_id: u32,
    label: &str,
) -> RusticolResult<&'a str> {
    let input = templates.input();
    let range = input
        .string_ranges
        .get(string_id as usize)
        .ok_or_else(|| invalid(format!("{label} string {string_id} is absent")))?;
    let range = range.as_usize_range(input.string_bytes.len(), label)?;
    std::str::from_utf8(&input.string_bytes[range])
        .map_err(|error| invalid(format!("{label} is not UTF-8: {error}")))
}

fn template_optional_string<'a>(
    templates: &'a ValidatedRecurrenceTemplateInput,
    string_id: u32,
    label: &str,
) -> RusticolResult<Option<&'a str>> {
    if string_id == MISSING_U32 {
        Ok(None)
    } else {
        template_string(templates, string_id, label).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::{
        CanonicalMomentumLinearForm, CheckedTableRange, ContributionKey, CurrentCoreKey,
        CurrentHelicityIdentity, DynamicLCColorState, DynamicLCColorStateId, LCColorWitnessTermId,
        MomentumTerm, PreparedDirectExecutorBinding, PreparedDirectExecutorKey,
        PreparedDirectIntrinsicDescriptor, PreparedDirectIntrinsicScale,
        RecurrenceAmplitudeDestination, RecurrenceClosureTerm, RecurrenceContribution,
        RecurrenceCurrent, RecurrenceFinalization, RecurrenceNodeKind, RecurrenceReplayTarget,
        RecurrenceResolvedHelicity, SemanticDigest, SourceStateAssignment,
        validated_template_fixture,
    };
    use crate::spinor::SpinorNode;

    fn digest(seed: u8) -> SemanticDigest {
        SemanticDigest::new([seed; 32]).unwrap()
    }

    fn scalar_contract_digest() -> SemanticDigest {
        let text = SCALAR_PRODUCT_CONTRACT.as_bytes();
        let mut bytes = [0_u8; 32];
        for (index, pair) in text.chunks_exact(2).enumerate() {
            bytes[index] = (hex(pair[0]) << 4) | hex(pair[1]);
        }
        SemanticDigest::new(bytes).unwrap()
    }

    fn hex(value: u8) -> u8 {
        match value {
            b'0'..=b'9' => value - b'0',
            b'a'..=b'f' => value - b'a' + 10,
            _ => panic!("invalid test hex"),
        }
    }

    fn signed_momentum(slots: &[(u32, i32)]) -> CanonicalMomentumLinearForm {
        CanonicalMomentumLinearForm::new(
            slots
                .iter()
                .map(|(source_slot, coefficient)| MomentumTerm {
                    source_slot: *source_slot,
                    coefficient: *coefficient,
                })
                .collect(),
        )
        .unwrap()
    }

    fn current_key(
        catalog: SemanticDigest,
        id: u32,
        support: &[u32],
        signed_slots: &[(u32, i32)],
        source: bool,
    ) -> CurrentCoreKey {
        CurrentCoreKey::new(
            catalog,
            if source {
                RecurrenceNodeKind::Source
            } else {
                RecurrenceNodeKind::Current
            },
            0,
            DynamicLCColorStateId::from_interner(id),
            support.to_vec(),
            signed_momentum(signed_slots),
            CurrentHelicityIdentity::topology_replay(
                0,
                support
                    .iter()
                    .map(|slot| SourceStateAssignment::new(*slot, 0))
                    .collect(),
            )
            .unwrap(),
            vec![1],
            0,
            vec![],
            if source {
                CurrentSourceBinding::FixedTemplate(0)
            } else {
                CurrentSourceBinding::None
            },
            None,
        )
        .unwrap()
    }

    fn scalar_program(templates: &ValidatedRecurrenceTemplateInput) -> RecurrenceProgram {
        let catalog = templates.summary().catalog_digest;
        let signed = [(0, -1), (1, -1), (2, 1), (3, 1)];
        let keys = vec![
            current_key(catalog, 0, &[0], &signed[0..1], true),
            current_key(catalog, 1, &[1], &signed[1..2], true),
            current_key(catalog, 2, &[2], &signed[2..3], true),
            current_key(catalog, 3, &[3], &signed[3..4], true),
            current_key(catalog, 4, &[0, 1], &signed[0..2], false),
            current_key(catalog, 5, &[2, 3], &signed[2..4], false),
        ];
        let transition = templates.input().transitions[0];
        let contribution = |id: u32, result: u32, parents: [u32; 2]| {
            RecurrenceContribution::new(
                id,
                result,
                parents.to_vec(),
                ContributionKey::new(
                    0,
                    parents.to_vec(),
                    vec![0, 0],
                    parents
                        .iter()
                        .map(|parent| keys[*parent as usize].momentum().clone())
                        .collect(),
                    0,
                    0,
                    LCColorWitnessTermId::new(0, 0),
                    digest(6),
                    transition.output_projection_string_id,
                )
                .unwrap(),
                ExactComplexRational::ONE,
            )
            .unwrap()
        };
        let contributions = vec![contribution(0, 4, [0, 1]), contribution(1, 5, [2, 3])];
        let currents = keys
            .into_iter()
            .enumerate()
            .map(|(id, key)| {
                RecurrenceCurrent::new(
                    id as u32,
                    key,
                    (id < 4).then_some(ExactComplexRational::ONE),
                    if id < 4 {
                        CheckedTableRange::new(0, 0)
                    } else {
                        CheckedTableRange::new((id - 4) as u64, 1)
                    },
                    if id >= 4 { Some((id - 4) as u32) } else { None },
                )
                .unwrap()
            })
            .collect();
        RecurrenceProgram::new(
            RecurrenceStrategy::TopologyReplay,
            1,
            1,
            (0..6)
                .map(|id| DynamicLCColorState::new(id, None, vec![]).unwrap())
                .collect(),
            currents,
            contributions,
            vec![
                RecurrenceFinalization::new(0, 4, None, ExactComplexRational::ONE).unwrap(),
                RecurrenceFinalization::new(1, 5, None, ExactComplexRational::ONE).unwrap(),
            ],
            vec![
                RecurrenceReplayTarget::new(
                    0,
                    0,
                    0,
                    vec![0, 1, 2, 3],
                    vec![1, 1, 1, 1],
                    ExactComplexRational::ONE,
                )
                .unwrap(),
            ],
            vec![
                RecurrenceResolvedHelicity::new(
                    0,
                    (0..4)
                        .map(|slot| SourceStateAssignment::new(slot, 0))
                        .collect(),
                    vec![0, 0, 0, 0],
                )
                .unwrap(),
            ],
            vec![
                RecurrenceAmplitudeDestination::new(0, 0, Some(0), CheckedTableRange::new(0, 1))
                    .unwrap(),
            ],
            vec![
                RecurrenceClosureTerm::new(0, 0, 0, None, vec![4, 5], ExactComplexRational::ONE)
                    .unwrap(),
            ],
        )
        .unwrap()
    }

    fn scalar_catalog(contribution_template: &str) -> PreparedDirectExecutorCatalog {
        let source_key = PreparedDirectExecutorKey::Evaluator {
            role: DirectExecutorRole::Source,
            evaluator_binding_id: 0,
        };
        let contribution_key = PreparedDirectExecutorKey::Evaluator {
            role: DirectExecutorRole::Contribution,
            evaluator_binding_id: 1,
        };
        let closure_key = PreparedDirectExecutorKey::Evaluator {
            role: DirectExecutorRole::Closure,
            evaluator_binding_id: 3,
        };
        PreparedDirectExecutorCatalog::new_with_intrinsics(
            digest(30),
            vec![
                PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Source, 0, 0),
                PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Contribution, 1, 1),
                PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Closure, 3, 2),
                PreparedDirectExecutorBinding::identity_finalizer(3),
            ],
            vec![
                PreparedDirectIntrinsicDescriptor::new(
                    source_key,
                    "rusticol.source-fill.scalar.v1:000000000000000000000000".to_owned(),
                    Some(digest(20)),
                    None,
                ),
                PreparedDirectIntrinsicDescriptor::new(
                    contribution_key,
                    contribution_template.to_owned(),
                    Some(scalar_contract_digest()),
                    Some(PreparedDirectIntrinsicScale::new(
                        1.0_f64.to_bits(),
                        0.0_f64.to_bits(),
                        None,
                    )),
                ),
                PreparedDirectIntrinsicDescriptor::new(
                    closure_key,
                    "rusticol.closure-reduce.v1:000000000000000000000000".to_owned(),
                    Some(digest(23)),
                    None,
                ),
                PreparedDirectIntrinsicDescriptor::new(
                    PreparedDirectExecutorKey::IdentityFinalizer,
                    IDENTITY_FINALIZER.to_owned(),
                    None,
                    None,
                ),
            ],
        )
        .unwrap()
    }

    #[test]
    fn authenticated_scalar_primitive_lowers_to_momentum_only_constant_amplitude() {
        let mut input = validated_template_fixture().into_input();
        input.sources[0].spin_state = 0;
        let templates = input.validate().unwrap();
        let program = scalar_program(&templates);
        let payload = lower_scalar_program(
            &program,
            &templates,
            &scalar_catalog(SCALAR_PRODUCT_TEMPLATE),
            4,
            0,
        )
        .unwrap();

        assert_eq!(payload.prepared_parameter_count(), 0);
        assert!(payload.parameter_bindings().is_empty());
        assert_eq!(
            payload
                .source_inputs()
                .iter()
                .map(|binding| binding.momentum_sign())
                .collect::<Vec<_>>(),
            [-1, -1, 1, 1]
        );
        assert!(
            payload
                .source_inputs()
                .iter()
                .all(|binding| binding.kind() == SpinorSourceInputKind::MomentumOnly)
        );
        assert_eq!(payload.dag().census().brackets, 0);
        assert_eq!(payload.dag().roots()[0].helicities(), [0, 0, 0, 0]);
        let amplitude = payload.dag().roots()[0].amplitude() as usize;
        assert_eq!(
            payload.dag().nodes()[amplitude],
            SpinorNode::Constant(ExactComplexRational::ONE)
        );
    }

    #[test]
    fn unsupported_scalar_primitive_id_fails_closed() {
        let mut input = validated_template_fixture().into_input();
        input.sources[0].spin_state = 0;
        let templates = input.validate().unwrap();
        let error = lower_scalar_program(
            &scalar_program(&templates),
            &templates,
            &scalar_catalog("rusticol.recurrence-intrinsic.unknown.v1"),
            4,
            0,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported contribution primitive")
        );
    }

    #[test]
    fn qcd_replay_helicities_are_pulled_back_to_representative_atoms() {
        assert_eq!(
            qcd_graph_helicities(&[-1, 1, -1, 1], Some(&[1, 0, 3, 2])).unwrap(),
            [1, -1, 1, -1]
        );
        assert_eq!(
            qcd_graph_helicities(&[-1, 1, -1, 1], None).unwrap(),
            [-1, 1, -1, 1]
        );
        assert!(qcd_graph_helicities(&[-1, 1], Some(&[0])).is_err());
    }

    #[test]
    fn chiral_dirac_couplings_follow_fermion_line_orientation() {
        assert_eq!(
            oriented_chiral_half_scales(CurrentOrientation::Particle, 11, 22).unwrap(),
            (11, 22)
        );
        assert_eq!(
            oriented_chiral_half_scales(CurrentOrientation::Antiparticle, 11, 22).unwrap(),
            (22, 11)
        );
        assert!(oriented_chiral_half_scales(CurrentOrientation::SelfConjugate, 11, 22).is_err());
    }

    #[test]
    fn charged_massive_vector_requires_one_exact_conjugate_state() {
        let templates = validated_template_fixture();
        let mut particle = templates.input().current_states[0];
        particle.id = 10;
        particle.particle_id = 701;
        particle.anti_particle_id = -909;
        particle.orientation = CurrentOrientation::Particle as u8;
        particle.statistics = ParticleStatistics::Boson as u8;
        particle.dimension = 4;
        particle.chirality = 0;
        particle.mass_parameter_id = 7;
        particle.width_parameter_id = 9;

        let mut antiparticle = particle;
        antiparticle.id = 11;
        antiparticle.particle_id = particle.anti_particle_id;
        antiparticle.anti_particle_id = particle.particle_id;
        antiparticle.orientation = CurrentOrientation::Antiparticle as u8;
        assert!(massive_vector_states_are_mutually_conjugate(
            &particle,
            &antiparticle
        ));
        assert!(massive_vector_states_are_mutually_conjugate(
            &antiparticle,
            &particle
        ));
        assert_eq!(
            massive_vector_conjugate_state_count(&[particle, antiparticle], &particle),
            1
        );

        let mut wrong_species = antiparticle;
        wrong_species.species_string_id = wrong_species.species_string_id.wrapping_add(1);
        assert!(!massive_vector_states_are_mutually_conjugate(
            &particle,
            &wrong_species
        ));
        let mut wrong_mass = antiparticle;
        wrong_mass.mass_parameter_id = wrong_mass.mass_parameter_id.wrapping_add(1);
        assert!(!massive_vector_states_are_mutually_conjugate(
            &particle,
            &wrong_mass
        ));
        let mut wrong_width = antiparticle;
        wrong_width.width_parameter_id = 10;
        assert!(!massive_vector_states_are_mutually_conjugate(
            &particle,
            &wrong_width
        ));
        let mut wrong_orientation = antiparticle;
        wrong_orientation.orientation = CurrentOrientation::Particle as u8;
        assert!(!massive_vector_states_are_mutually_conjugate(
            &particle,
            &wrong_orientation
        ));
        let mut duplicate = antiparticle;
        duplicate.id = 12;
        assert_eq!(
            massive_vector_conjugate_state_count(&[particle, antiparticle, duplicate], &particle),
            2
        );
    }

    #[test]
    fn massive_dirac_source_pair_requires_one_width_owner() {
        let particle = DiracSourceEndpoint {
            source_slot: 0,
            state_id: 7,
            particle_id: 6,
            anti_particle_id: -6,
            species_string_id: 19,
            mass_prepared_slot: 33,
            width_parameter_id: 12,
            width_prepared_slot: Some(39),
        };
        let antiparticle = DiracSourceEndpoint {
            source_slot: 1,
            state_id: 8,
            particle_id: -6,
            anti_particle_id: 6,
            ..particle
        };
        assert!(dirac_source_endpoints_are_mutually_conjugate(
            &particle,
            &antiparticle
        ));

        let wrong_width = DiracSourceEndpoint {
            width_parameter_id: 13,
            width_prepared_slot: Some(40),
            ..antiparticle
        };
        assert!(!dirac_source_endpoints_are_mutually_conjugate(
            &particle,
            &wrong_width
        ));
    }

    #[test]
    fn optional_state_width_must_match_typed_finalizer_owner() {
        assert!(validate_optional_width_owner(MISSING_U32, 12, "state").is_ok());
        assert!(validate_optional_width_owner(12, 12, "state").is_ok());
        assert!(
            validate_optional_width_owner(13, 12, "state")
                .unwrap_err()
                .to_string()
                .contains("typed finalizer ownership")
        );
    }

    #[test]
    fn massless_owner_requires_one_immutable_exact_zero_real_parameter() {
        let parameter = super::super::template::ParameterRow {
            id: 4,
            template_string_id: 0,
            name_string_id: 0,
            kind: ParameterKind::Derived as u8,
            value_type: ParameterValueType::Real as u8,
            mutable: 0,
            default_factor_id: MISSING_U32,
            exact_expression_digest_id: 7,
            dependency_sequence_id: 0,
            prepared_parameter_id: 9,
            semantic_digest_id: 8,
        };
        let zero_digest = super::super::template::DigestCatalogRow {
            id: 7,
            value: ZERO_PARAMETER_EXPRESSION_DIGEST,
        };
        assert!(parameter_row_is_immutable_exact_zero(&parameter, &zero_digest).unwrap());

        let mutable = super::super::template::ParameterRow {
            mutable: 1,
            ..parameter
        };
        assert!(!parameter_row_is_immutable_exact_zero(&mutable, &zero_digest).unwrap());
        let nonzero = super::super::template::DigestCatalogRow {
            value: [1; 32],
            ..zero_digest
        };
        assert!(!parameter_row_is_immutable_exact_zero(&parameter, &nonzero).unwrap());
    }

    #[test]
    fn u1_subtraction_vector_requires_its_singlet_auxiliary_contract() {
        let templates = validated_template_fixture();
        let mut state = templates.input().current_states[0];
        state.particle_id = 701;
        state.anti_particle_id = 701;
        state.orientation = CurrentOrientation::SelfConjugate as u8;
        state.statistics = ParticleStatistics::Boson as u8;
        state.color_representation = 1;
        state.dimension = 4;
        state.chirality = 0;
        state.mass_parameter_id = MISSING_U32;
        state.width_parameter_id = MISSING_U32;

        let classify = |candidate: &crate::recurrence::template::CurrentStateRow,
                        basis: &str,
                        auxiliary: Option<&str>| {
            is_u1_subtraction_vector_state(candidate, basis, auxiliary, &templates).unwrap()
        };
        let basis = "auxiliary:u1-subtraction-color-flow-vector";
        let auxiliary = Some("u1-subtraction-color-flow-vector");
        assert!(classify(&state, basis, auxiliary));
        assert!(!classify(&state, "lorentz-vector", auxiliary));
        assert!(!classify(&state, basis, Some("antisymmetric-tensor")));

        let mut colored = state;
        colored.color_representation = 8;
        assert!(!classify(&colored, basis, auxiliary));
        let mut massive = state;
        massive.mass_parameter_id = 0;
        assert!(!classify(&massive, basis, auxiliary));
        let mut oriented = state;
        oriented.orientation = CurrentOrientation::Particle as u8;
        assert!(!classify(&oriented, basis, auxiliary));
    }

    #[test]
    fn weyl_pair_vector_requires_canonical_particle_antiparticle_states() {
        let builder = SpinorDagBuilder::new(4).unwrap();
        let particle = QcdCurrent::Weyl {
            chirality: SpinorChirality::Negative,
            orientation: CurrentOrientation::Particle,
            value: Some(LinearWeylExpression::atom(0, builder.one())),
        };
        let antiparticle = QcdCurrent::Weyl {
            chirality: SpinorChirality::Positive,
            orientation: CurrentOrientation::Antiparticle,
            value: Some(LinearWeylExpression::atom(1, builder.one())),
        };
        let canonical = vec![Some(particle.clone()), Some(antiparticle.clone())];
        assert!(required_qcd_weyl_pair(&canonical, [0, 1], SpinorChirality::Negative).is_ok());
        assert!(required_qcd_weyl_pair(&canonical, [0, 1], SpinorChirality::Positive).is_err());

        let raw_antiparticle_first = vec![Some(antiparticle), Some(particle)];
        assert!(
            required_qcd_weyl_pair(&raw_antiparticle_first, [1, 0], SpinorChirality::Negative,)
                .is_ok()
        );
        assert!(
            required_qcd_weyl_pair(&raw_antiparticle_first, [0, 1], SpinorChirality::Negative,)
                .is_err()
        );
    }

    #[test]
    fn authenticated_nonunit_binary64_scale_remains_exact() {
        let value = 1.0_f64 / 3.0;
        let scale = exact_binary64_scale(value.to_bits(), 0.0_f64.to_bits()).unwrap();

        assert_eq!(scale.real(), ExactRational::from_f64_exact(value).unwrap());
        assert_eq!(scale.imag(), ExactRational::ZERO);
        assert!(exact_binary64_scale(0.0_f64.to_bits(), 0.0_f64.to_bits()).is_err());
    }
}
