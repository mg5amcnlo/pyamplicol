// SPDX-License-Identifier: 0BSD

//! Deterministic construction of a spinor lane from authenticated semantics.

use super::spinor_lane::SpinorNativeRuntime;
use super::spinor_manifest::{
    SpinorExecutionManifest, SpinorProcessFamily, TOP_MASS_PARAMETER_NAME, TOP_WIDTH_PARAMETER_NAME,
};
use super::*;
use crate::spinor::{
    build_helicity_summed_massive_quark_two_gluon_spinor_dag,
    build_helicity_summed_quark_gluon_bg_spinor_dag,
    build_helicity_summed_quark_z_gluon_spinor_dag,
    build_optimized_helicity_summed_gluon_spinor_dag, decode_spinor_dag_v2,
};
use serde::Deserialize;

const Z_MASS_PARAMETER_NAME: &str = "particle.23.mass";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpinorNormalization {
    color_accuracy: String,
    color_factor: f64,
    average_factor: f64,
    identical_factor: f64,
    global_coupling_factor: f64,
    qcd_coupling_power: usize,
    electroweak_coupling_power: usize,
    couplings_in_stage_evaluators: bool,
    coupling_policy: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpinorPhysicsExtension {
    helicity_axis: String,
    fixed_color_order: Vec<usize>,
    #[serde(default)]
    process_family: Option<SpinorProcessFamily>,
    #[serde(default)]
    ordered_source_labels: Option<Vec<usize>>,
    #[serde(default)]
    spinor_parameter_names: Option<Vec<String>>,
}

pub(super) struct LoadedSpinorRuntime {
    pub(super) common: ExecutionRuntime,
    pub(super) lane: SpinorNativeRuntime,
}

pub(super) fn load_spinor_native_runtime_from_artifact(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &SpinorExecutionManifest,
    physics: &ProcessPhysicsV1,
) -> RusticolResult<LoadedSpinorRuntime> {
    if !manifest.is_graph_backed() {
        return load_spinor_native_runtime(manifest, physics);
    }
    load_graph_spinor_native_runtime(artifact, evaluator_root, manifest, physics)
}

fn load_graph_spinor_native_runtime(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &SpinorExecutionManifest,
    physics: &ProcessPhysicsV1,
) -> RusticolResult<LoadedSpinorRuntime> {
    let graph = manifest.graph_payload.as_ref().ok_or_else(|| {
        RusticolError::integrity("graph-backed spinor manifest has no graph payload")
    })?;
    let relative_root = evaluator_root.strip_prefix(artifact.root()).map_err(|_| {
        RusticolError::security("spinor process root escapes the verified artifact")
    })?;
    let graph_path = relative_root.join(confined_internal_path(
        &graph.path,
        "spinor graph payload path",
    )?);
    let graph_path = graph_path
        .to_str()
        .ok_or_else(|| RusticolError::security("spinor graph payload path is not UTF-8"))?;
    let record = artifact.payload(graph_path)?;
    if record.role != PayloadRole::EvaluatorState
        || record.media_type != "application/octet-stream"
        || record.process_id.as_deref() != Some(manifest.key.as_str())
        || record.executable
    {
        return Err(RusticolError::integrity(
            "spinor graph payload is not process-owned evaluator state",
        ));
    }
    let payload = decode_spinor_dag_v2(&artifact.read_payload(graph_path)?)?;
    if usize::from(payload.dag().momentum_count()) != manifest.external_pdg_order.len() {
        return Err(RusticolError::integrity(
            "spinor graph source count disagrees with the execution manifest",
        ));
    }

    let metadata = manifest.runtime_metadata.as_ref().ok_or_else(|| {
        RusticolError::integrity("graph-backed spinor manifest has no runtime metadata")
    })?;
    if physics.external_particles.len() != manifest.external_pdg_order.len()
        || physics
            .external_particles
            .iter()
            .map(|particle| particle.pdg)
            .ne(manifest.external_pdg_order.iter().copied())
        || physics.helicities.len() != 1
        || physics.helicities[0].id != "h:sum"
        || physics.helicities[0].values.iter().any(|value| *value != 0)
        || physics.selectors.helicity
        || physics.color_components.len() != 1
        || physics.color_components[0].id() != metadata.public_color_flows[0].public_id
    {
        return Err(RusticolError::integrity(
            "graph-backed spinor physics disagrees with its external, helicity, or color contract",
        ));
    }
    let crate::ColorComponent::LcFlow(flow) = &physics.color_components[0] else {
        return Err(RusticolError::artifact(
            "graph-backed spinor physics requires one LC color component",
        ));
    };
    if !flow.computed || flow.coefficient != 1.0 {
        return Err(RusticolError::integrity(
            "graph-backed spinor color component is not the computed representative",
        ));
    }

    let prepared_pack = manifest
        .kernel_pack
        .as_ref()
        .map(|pack_reference| {
            let pack_record = artifact.payload(&pack_reference.manifest_path)?;
            if pack_record.role != PayloadRole::EvaluatorManifest {
                return Err(RusticolError::security(
                    "spinor prepared kernel pack is not evaluator-manifest payload",
                ));
            }
            let pack_bytes = artifact.read_payload(&pack_reference.manifest_path)?;
            let pack: super::eager_manifest::PreparedKernelPackManifest =
                serde_json::from_slice(&pack_bytes).map_err(|error| {
                    RusticolError::serialization(format!(
                        "could not parse spinor prepared kernel pack: {error}"
                    ))
                })?;
            pack.validate()?;
            super::recurrence_load::validate_recurrence_prepared_pack_outer_target(
                &artifact.manifest().producer.target,
                &pack,
            )?;
            let payload_root = artifact.root().join(confined_internal_path(
                &pack_reference.payload_root,
                "spinor prepared kernel payload root",
            )?);
            super::eager_load::validate_prepared_kernel_references(artifact, &payload_root, &pack)?;
            Ok((pack, payload_root))
        })
        .transpose()?;
    if prepared_pack.is_none() && !payload.parameter_bindings().is_empty() {
        return Err(RusticolError::integrity(
            "a parameterized spinor graph has no prepared kernel pack",
        ));
    }

    let runtime_parameters = super::recurrence_load::recurrence_runtime_parameters(metadata);
    let model_parameter_runtime_slots =
        super::recurrence_load::runtime_parameter_slots(&runtime_parameters)?;
    let model_parameter_values_f64 = runtime_parameters
        .iter()
        .map(|parameter| parameter.default)
        .collect::<Vec<_>>();
    let model_parameter_name_to_index = runtime_parameters
        .iter()
        .map(|parameter| (parameter.name.clone(), parameter.parameter_index))
        .collect::<BTreeMap<_, _>>();
    let parameter_defaults = metadata
        .prepared_parameter_defaults
        .iter()
        .map(|[real, imaginary]| crate::EagerComplex64::new(*real, *imaginary))
        .collect::<Vec<_>>();
    if payload.prepared_parameter_count() as usize != parameter_defaults.len() {
        return Err(RusticolError::integrity(
            "spinor payload prepared-parameter count disagrees with authoritative runtime metadata",
        ));
    }
    let parameter_projection =
        super::recurrence_load::recurrence_prepared_parameter_projection(metadata)?;
    let normalization = super::recurrence_load::recurrence_normalization_values(metadata)?;
    let dag_census = payload.dag().census();
    let current_count = payload.dag().nodes().len();
    let amplitude_output_count = payload.dag().roots().len();
    let lane = SpinorNativeRuntime::new_payload(payload, parameter_defaults, parameter_projection)?;
    let particle_masses = metadata
        .particle_masses
        .iter()
        .map(|row| (row.outgoing_pdg, row.mass))
        .collect::<BTreeMap<_, _>>();
    let mut particle_mass_parameter_names = BTreeMap::new();
    for source in &metadata.source_templates {
        let Some(name) = source.source_ir.mass_parameter.as_ref() else {
            continue;
        };
        particle_mass_parameter_names.insert(source.source_ir.identity.pdg_label, name.clone());
        particle_mass_parameter_names
            .insert(source.source_ir.identity.anti_pdg_label, name.clone());
    }
    let mut common = ExecutionRuntime {
        process: manifest.process.clone(),
        key: manifest.key.clone(),
        color_accuracy: manifest.color_accuracy.clone(),
        external_pdg_order: manifest.external_pdg_order.clone(),
        external_count: manifest.external_pdg_order.len(),
        parameter_count: runtime_parameters.len(),
        value_parameter_count: 0,
        momentum_parameter_count: 0,
        current_count,
        source_count: manifest.external_pdg_order.len(),
        interaction_count: dag_census.estimated_complex_arithmetic(),
        stage_count: 1,
        amplitude_output_count,
        lc_topology_replay_enabled: false,
        lc_topology_replay_mappings: Arc::new(Vec::new()),
        lc_topology_replay_public_mappings: Vec::new(),
        lc_topology_replay_routes: Vec::new(),
        lc_topology_replay_materialized_sector_ids: BTreeSet::new(),
        lc_resolved_replay_plan: None,
        lc_resolved_replay_selection_cache: None,
        lc_replay_flat_momenta_scratch: Vec::new(),
        lc_replay_target_components_scratch: Vec::new(),
        color_topology_replay_enabled: false,
        color_topology_replay_mappings: Arc::new(Vec::new()),
        color_replay_flat_momenta_scratch: Vec::new(),
        helicity_recurrence: None,
        compiled_helicity_execution_plan: None,
        compiled_color_execution_plan: None,
        compiled_direct_runtime: None,
        compiled_direct_color_schedules: BTreeMap::new(),
        compiled_direct_helicity_schedules: BTreeMap::new(),
        helicity_sum_runtime: None,
        helicity_selector_runtimes: Vec::new(),
        helicity_selector_runtime_schedule_modes: Vec::new(),
        helicity_selector_lane_by_domain: BTreeMap::new(),
        color_selector_runtimes: BTreeMap::new(),
        runtime_unavailable_message: None,
        sources: Vec::new(),
        momentum_slots: Vec::new(),
        external_is_initial: physics
            .external_particles
            .iter()
            .map(|particle| particle.role == crate::ParticleRole::Initial)
            .collect(),
        particle_masses,
        particle_mass_parameter_names,
        normalization_factor: normalization.factor,
        normalization_color_factor: normalization.color_factor,
        normalization_average_factor: metadata.normalization.average_factor,
        normalization_identical_factor: metadata.normalization.identical_factor,
        normalization_qcd_coupling_power: metadata.normalization.qcd_coupling_power.unwrap_or(0)
            as usize,
        normalization_electroweak_coupling_power: metadata
            .normalization
            .electroweak_coupling_power
            .unwrap_or(0) as usize,
        model_parameters: runtime_parameters,
        model_parameter_name_to_index,
        model_parameter_runtime_slots,
        model_parameter_values_f64,
        model_parameter_evaluator: None,
        physics_reduction_override: None,
        physics: None,
        stages: None,
        amplitude_stage: None,
        state_scratch_f64: Vec::new(),
        state_scratch_f64_requires_clear: false,
        values_scratch_f64: Vec::new(),
    };
    if let Some((pack, payload_root)) = prepared_pack {
        let kernel_payloads = artifact.evaluator_payload_store(&payload_root)?;
        common.model_parameter_evaluator =
            super::eager_load::load_prepared_model_parameter_evaluator_for_runtime(
                &pack,
                &common.model_parameters,
                &kernel_payloads,
            )?;
    }
    common.refresh_derived_model_parameters()?;
    Ok(LoadedSpinorRuntime { common, lane })
}

pub(super) fn load_spinor_native_runtime(
    manifest: &SpinorExecutionManifest,
    physics: &ProcessPhysicsV1,
) -> RusticolResult<LoadedSpinorRuntime> {
    let external_count = usize::from(manifest.external_count);
    let process_family = manifest.process_family();
    let ordered_source_labels = manifest.ordered_source_labels();
    if physics.external_particles.len() != external_count
        || physics
            .external_particles
            .iter()
            .map(|particle| particle.pdg)
            .ne(manifest.external_pdg_order.iter().copied())
        || (process_family == SpinorProcessFamily::PureGluon
            && physics
                .external_particles
                .iter()
                .any(|particle| particle.pdg != 21))
        || physics.helicities.len() != 1
        || physics.helicities[0].id != "h:sum"
        || physics.selectors.helicity
        || physics.color_components.len() != 1
    {
        return Err(RusticolError::artifact(
            "spinor runtime physics must match the external process and describe one helicity-sum axis and one fixed color flow",
        ));
    }
    if process_family == SpinorProcessFamily::SingleMassiveQuarkLine
        && physics
            .external_particles
            .iter()
            .map(|particle| particle.role)
            .ne([
                crate::ParticleRole::Initial,
                crate::ParticleRole::Initial,
                crate::ParticleRole::Final,
                crate::ParticleRole::Final,
            ])
    {
        return Err(RusticolError::artifact(
            "massive-quark spinor physics requires two initial gluons and final t tbar",
        ));
    }
    let ordered_outgoing_pdgs = ordered_source_labels
        .iter()
        .map(|label| {
            let particle = physics.external_particles.get(label - 1).ok_or_else(|| {
                RusticolError::integrity(
                    "spinor source traversal references an absent external label",
                )
            })?;
            if particle.role == crate::ParticleRole::Initial
                && (1..=5).contains(&particle.pdg.checked_abs().unwrap_or_default())
            {
                particle
                    .pdg
                    .checked_neg()
                    .ok_or_else(|| RusticolError::artifact("spinor external PDG cannot be crossed"))
            } else {
                Ok(particle.pdg)
            }
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let ordered_colored_pdgs = match process_family {
        SpinorProcessFamily::PureGluon => ordered_outgoing_pdgs.as_slice(),
        SpinorProcessFamily::SingleMasslessQuarkLine => ordered_outgoing_pdgs.as_slice(),
        SpinorProcessFamily::SingleMassiveQuarkLine => ordered_outgoing_pdgs.as_slice(),
        SpinorProcessFamily::SingleMasslessQuarkLineMassiveNeutralVector => {
            let Some((&z_pdg, colored_pdgs)) = ordered_outgoing_pdgs.split_last() else {
                return Err(RusticolError::artifact(
                    "the q-Z spinor traversal has no massive neutral-vector source",
                ));
            };
            if z_pdg != 23 {
                return Err(RusticolError::artifact(
                    "the q-Z spinor traversal must end with its Z source",
                ));
            }
            colored_pdgs
        }
    };
    match process_family {
        SpinorProcessFamily::PureGluon => {}
        SpinorProcessFamily::SingleMassiveQuarkLine => {
            if ordered_colored_pdgs != [6, 21, 21, -6] {
                return Err(RusticolError::artifact(
                    "the massive open-quark-line traversal must be t, gluon, gluon, tbar",
                ));
            }
        }
        SpinorProcessFamily::SingleMasslessQuarkLine
        | SpinorProcessFamily::SingleMasslessQuarkLineMassiveNeutralVector => {
            let Some((&quark_pdg, middle_and_antiquark)) = ordered_colored_pdgs.split_first()
            else {
                return Err(RusticolError::artifact(
                    "the massless open-quark-line traversal has no quark endpoint",
                ));
            };
            let Some((&antiquark_pdg, gluon_pdgs)) = middle_and_antiquark.split_last() else {
                return Err(RusticolError::artifact(
                    "the massless open-quark-line traversal has no antiquark endpoint",
                ));
            };
            if !(1..=5).contains(&quark_pdg)
                || antiquark_pdg != -quark_pdg
                || gluon_pdgs.iter().any(|pdg| *pdg != 21)
            {
                return Err(RusticolError::artifact(
                    "the massless open-quark-line traversal must be q, gluons..., qbar with one light flavour",
                ));
            }
        }
    }
    let spinor_value = physics.extensions.get("spinor_dag").ok_or_else(|| {
        RusticolError::artifact("spinor runtime physics has no spinor_dag extension")
    })?;
    let spinor: SpinorPhysicsExtension =
        serde_json::from_value(spinor_value.clone()).map_err(|error| {
            RusticolError::serialization(format!(
                "could not parse spinor runtime physics extension: {error}"
            ))
        })?;
    let extension_family = spinor
        .process_family
        .unwrap_or(SpinorProcessFamily::PureGluon);
    let extension_ordered_source_labels = spinor
        .ordered_source_labels
        .as_deref()
        .unwrap_or(&spinor.fixed_color_order);
    let extension_parameter_names = spinor.spinor_parameter_names.as_deref().unwrap_or(&[]);
    if process_family == SpinorProcessFamily::SingleMasslessQuarkLine
        && (spinor.process_family.is_none() || spinor.ordered_source_labels.is_none())
    {
        return Err(RusticolError::integrity(
            "massless open-quark-line spinor physics omits its process family or source traversal",
        ));
    }
    if process_family == SpinorProcessFamily::SingleMassiveQuarkLine
        && (spinor.process_family.is_none()
            || spinor.ordered_source_labels.is_none()
            || spinor.spinor_parameter_names.is_none())
    {
        return Err(RusticolError::integrity(
            "massive open-quark-line spinor physics omits its process family, source traversal, or mass/width binding",
        ));
    }
    if process_family == SpinorProcessFamily::SingleMasslessQuarkLineMassiveNeutralVector
        && (spinor.process_family.is_none()
            || spinor.ordered_source_labels.is_none()
            || spinor.spinor_parameter_names.is_none())
    {
        return Err(RusticolError::integrity(
            "q-Z spinor physics omits its process family, source traversal, or local parameter binding",
        ));
    }
    if spinor.helicity_axis != "always-summed-aggregate"
        || spinor.fixed_color_order != manifest.fixed_color_order
        || extension_family != process_family
        || extension_ordered_source_labels != ordered_source_labels
        || extension_parameter_names != manifest.spinor_parameter_names()
    {
        return Err(RusticolError::integrity(
            "spinor runtime physics does not authenticate its aggregate helicity axis, process family, and source traversal",
        ));
    }
    let crate::ColorComponent::LcFlow(flow) = &physics.color_components[0] else {
        return Err(RusticolError::artifact(
            "spinor runtime physics requires one LC color flow",
        ));
    };
    if flow.word != manifest.fixed_color_order || !flow.computed || flow.coefficient != 1.0 {
        return Err(RusticolError::integrity(
            "spinor runtime physics fixed color flow does not match its execution manifest",
        ));
    }
    let normalization_value = physics.extensions.get("normalization").ok_or_else(|| {
        RusticolError::artifact("spinor runtime physics has no normalization extension")
    })?;
    let normalization: SpinorNormalization = serde_json::from_value(normalization_value.clone())
        .map_err(|error| {
            RusticolError::serialization(format!(
                "could not parse spinor runtime normalization: {error}"
            ))
        })?;
    let (expected_qcd_power, expected_electroweak_power, local_couplings, coupling_policy) =
        match process_family {
            SpinorProcessFamily::PureGluon
            | SpinorProcessFamily::SingleMasslessQuarkLine
            | SpinorProcessFamily::SingleMassiveQuarkLine => (
                external_count - 2,
                0,
                false,
                "coupling-stripped spinor DAG with one global reduction factor",
            ),
            SpinorProcessFamily::SingleMasslessQuarkLineMassiveNeutralVector => (
                external_count - 3,
                1,
                true,
                "spinor DAG with local chiral couplings and one global reduction factor",
            ),
        };
    if normalization.color_accuracy != "lc"
        || normalization.qcd_coupling_power != expected_qcd_power
        || normalization.electroweak_coupling_power != expected_electroweak_power
        || normalization.couplings_in_stage_evaluators != local_couplings
        || normalization.coupling_policy != coupling_policy
        || [
            normalization.color_factor,
            normalization.average_factor,
            normalization.identical_factor,
            normalization.global_coupling_factor,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(RusticolError::artifact(
            "spinor runtime physics has an incompatible normalization contract",
        ));
    }
    let normalization_factor = normalization.color_factor * normalization.global_coupling_factor
        / (normalization.average_factor * normalization.identical_factor);
    if !normalization_factor.is_finite() {
        return Err(RusticolError::integrity(
            "spinor runtime normalization factor is not finite",
        ));
    }

    let mut model_parameters = Vec::with_capacity(physics.model_parameters.len());
    let mut model_parameter_name_to_index = BTreeMap::new();
    let mut model_parameter_runtime_slots = BTreeMap::new();
    let mut model_parameter_values_f64 = Vec::with_capacity(physics.model_parameters.len());
    let spinor_parameter_names = manifest.spinor_parameter_names();
    let spinor_parameter_name_set = spinor_parameter_names
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut z_mass_parameter_index = None;
    let mut top_mass_parameter_index = None;
    let mut top_width_parameter_index = None;
    for (parameter_index, parameter) in physics.model_parameters.iter().enumerate() {
        let (kind, pdg) = match parameter.kind {
            crate::ParameterKind::Normalization => ("normalization", None),
            crate::ParameterKind::Coupling
                if process_family
                    == SpinorProcessFamily::SingleMasslessQuarkLineMassiveNeutralVector
                    && spinor_parameter_name_set.contains(parameter.name.as_str()) =>
            {
                ("coupling", None)
            }
            crate::ParameterKind::Mass
                if process_family
                    == SpinorProcessFamily::SingleMasslessQuarkLineMassiveNeutralVector
                    && parameter.name == Z_MASS_PARAMETER_NAME
                    && parameter.default_real > 0.0 =>
            {
                z_mass_parameter_index = Some(parameter_index);
                ("particle_mass", Some(23))
            }
            crate::ParameterKind::Mass
                if process_family == SpinorProcessFamily::SingleMassiveQuarkLine
                    && parameter.name == TOP_MASS_PARAMETER_NAME
                    && parameter.default_real.is_finite()
                    && parameter.default_real > 0.0 =>
            {
                top_mass_parameter_index = Some(parameter_index);
                ("particle_mass", Some(6))
            }
            crate::ParameterKind::Width
                if process_family == SpinorProcessFamily::SingleMassiveQuarkLine
                    && parameter.name == TOP_WIDTH_PARAMETER_NAME
                    && parameter.default_real.is_finite()
                    && parameter.default_real >= 0.0 =>
            {
                top_width_parameter_index = Some(parameter_index);
                ("particle_width", Some(6))
            }
            _ => {
                return Err(RusticolError::artifact(
                    "spinor runtime physics contains an unsupported model parameter",
                ));
            }
        };
        if !parameter.mutable
            || parameter.default_imaginary != 0.0
            || model_parameter_name_to_index
                .insert(parameter.name.clone(), parameter_index)
                .is_some()
        {
            return Err(RusticolError::artifact(
                "spinor runtime supports only distinct mutable real normalization and bound graph parameters",
            ));
        }
        model_parameter_runtime_slots.insert(
            parameter.name.clone(),
            RuntimeParameterSlots {
                real: parameter_index,
                imaginary: None,
            },
        );
        model_parameter_values_f64.push(parameter.default_real);
        model_parameters.push(GenericRuntimeModelParameterManifest {
            name: parameter.name.clone(),
            kind: kind.to_string(),
            parameter_index,
            default: parameter.default_real,
            pdg,
            runtime_name: None,
            complex_component: None,
        });
    }
    for required in ["normalization.alpha_s_me_check", "normalization.alpha_ew"] {
        if !model_parameter_name_to_index.contains_key(required) {
            return Err(RusticolError::artifact(format!(
                "spinor runtime physics is missing normalization parameter {required:?}"
            )));
        }
    }
    let spinor_parameter_indices = spinor_parameter_names
        .iter()
        .enumerate()
        .map(|(graph_parameter_index, name)| {
            let parameter_index = model_parameter_name_to_index
                .get(name)
                .copied()
                .ok_or_else(|| {
                    RusticolError::artifact(format!(
                        "spinor runtime physics is missing bound graph parameter {name:?}"
                    ))
                })?;
            let expected_kind = match process_family {
                SpinorProcessFamily::SingleMasslessQuarkLineMassiveNeutralVector => {
                    crate::ParameterKind::Coupling
                }
                SpinorProcessFamily::SingleMassiveQuarkLine => {
                    *[crate::ParameterKind::Mass, crate::ParameterKind::Width]
                        .get(graph_parameter_index)
                        .ok_or_else(|| {
                            RusticolError::integrity(
                                "massive-quark spinor physics declares too many graph parameters",
                            )
                        })?
                }
                SpinorProcessFamily::PureGluon | SpinorProcessFamily::SingleMasslessQuarkLine => {
                    return Err(RusticolError::integrity(
                        "a parameter-free spinor family declares graph parameters",
                    ));
                }
            };
            if physics.model_parameters[parameter_index].kind != expected_kind {
                let problem = if expected_kind == crate::ParameterKind::Coupling {
                    "is not declared as a coupling"
                } else {
                    "has the wrong physics kind"
                };
                return Err(RusticolError::artifact(format!(
                    "spinor DAG parameter {name:?} {problem}"
                )));
            }
            Ok(parameter_index)
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    if process_family == SpinorProcessFamily::SingleMasslessQuarkLineMassiveNeutralVector
        && z_mass_parameter_index.is_none()
    {
        return Err(RusticolError::artifact(format!(
            "q-Z spinor runtime physics is missing the positive mass parameter {Z_MASS_PARAMETER_NAME:?}"
        )));
    }
    if process_family == SpinorProcessFamily::SingleMassiveQuarkLine
        && (top_mass_parameter_index.is_none() || top_width_parameter_index.is_none())
    {
        return Err(RusticolError::artifact(format!(
            "massive-quark spinor runtime physics requires ordered parameters {TOP_MASS_PARAMETER_NAME:?} and {TOP_WIDTH_PARAMETER_NAME:?}"
        )));
    }

    let dag = match process_family {
        SpinorProcessFamily::PureGluon => {
            build_optimized_helicity_summed_gluon_spinor_dag(manifest.external_count)?
        }
        SpinorProcessFamily::SingleMasslessQuarkLine => {
            let graph_source_slots = (0..manifest.external_count).collect::<Vec<_>>();
            build_helicity_summed_quark_gluon_bg_spinor_dag(&graph_source_slots)?
        }
        SpinorProcessFamily::SingleMassiveQuarkLine => {
            build_helicity_summed_massive_quark_two_gluon_spinor_dag(&[0, 1, 2, 3])?
        }
        SpinorProcessFamily::SingleMasslessQuarkLineMassiveNeutralVector => {
            let colored_count = u16::try_from(manifest.fixed_color_order.len()).map_err(|_| {
                RusticolError::artifact("q-Z spinor colored source count exceeds u16")
            })?;
            let graph_colored_slots = (0..colored_count).collect::<Vec<_>>();
            build_helicity_summed_quark_z_gluon_spinor_dag(&graph_colored_slots, colored_count)?
        }
    };
    let census = dag.census();
    let current_count = dag.nodes().len();
    let amplitude_output_count = dag.roots().len();
    let lane = SpinorNativeRuntime::new(
        dag,
        ordered_source_labels.to_vec(),
        spinor_parameter_indices,
    )?;
    let mut particle_masses = BTreeMap::from([(21, 0.0)]);
    for pdg in manifest
        .external_pdg_order
        .iter()
        .chain(&ordered_outgoing_pdgs)
    {
        if *pdg != 23 {
            particle_masses.insert(*pdg, 0.0);
        }
    }
    let mut particle_mass_parameter_names = BTreeMap::new();
    if let Some(parameter_index) = z_mass_parameter_index {
        particle_masses.insert(23, model_parameter_values_f64[parameter_index]);
        particle_mass_parameter_names.insert(23, Z_MASS_PARAMETER_NAME.to_string());
    }
    if let Some(parameter_index) = top_mass_parameter_index {
        let top_mass = model_parameter_values_f64[parameter_index];
        particle_masses.insert(6, top_mass);
        particle_masses.insert(-6, top_mass);
        particle_mass_parameter_names.insert(6, TOP_MASS_PARAMETER_NAME.to_string());
        particle_mass_parameter_names.insert(-6, TOP_MASS_PARAMETER_NAME.to_string());
    }
    let mut common = ExecutionRuntime {
        process: manifest.process.clone(),
        key: manifest.key.clone(),
        color_accuracy: manifest.color_accuracy.clone(),
        external_pdg_order: manifest.external_pdg_order.clone(),
        external_count,
        parameter_count: model_parameters.len(),
        value_parameter_count: 0,
        momentum_parameter_count: 0,
        current_count,
        source_count: external_count,
        interaction_count: census.estimated_complex_arithmetic(),
        stage_count: 1,
        amplitude_output_count,
        lc_topology_replay_enabled: false,
        lc_topology_replay_mappings: Arc::new(Vec::new()),
        lc_topology_replay_public_mappings: Vec::new(),
        lc_topology_replay_routes: Vec::new(),
        lc_topology_replay_materialized_sector_ids: BTreeSet::new(),
        lc_resolved_replay_plan: None,
        lc_resolved_replay_selection_cache: None,
        lc_replay_flat_momenta_scratch: Vec::new(),
        lc_replay_target_components_scratch: Vec::new(),
        color_topology_replay_enabled: false,
        color_topology_replay_mappings: Arc::new(Vec::new()),
        color_replay_flat_momenta_scratch: Vec::new(),
        helicity_recurrence: None,
        compiled_helicity_execution_plan: None,
        compiled_color_execution_plan: None,
        compiled_direct_runtime: None,
        compiled_direct_color_schedules: BTreeMap::new(),
        compiled_direct_helicity_schedules: BTreeMap::new(),
        helicity_sum_runtime: None,
        helicity_selector_runtimes: Vec::new(),
        helicity_selector_runtime_schedule_modes: Vec::new(),
        helicity_selector_lane_by_domain: BTreeMap::new(),
        color_selector_runtimes: BTreeMap::new(),
        runtime_unavailable_message: None,
        sources: Vec::new(),
        momentum_slots: Vec::new(),
        external_is_initial: physics
            .external_particles
            .iter()
            .map(|particle| particle.role == crate::ParticleRole::Initial)
            .collect(),
        particle_masses,
        particle_mass_parameter_names,
        normalization_factor,
        normalization_color_factor: normalization.color_factor,
        normalization_average_factor: normalization.average_factor,
        normalization_identical_factor: normalization.identical_factor,
        normalization_qcd_coupling_power: normalization.qcd_coupling_power,
        normalization_electroweak_coupling_power: normalization.electroweak_coupling_power,
        model_parameters,
        model_parameter_name_to_index,
        model_parameter_runtime_slots,
        model_parameter_values_f64,
        model_parameter_evaluator: None,
        physics_reduction_override: None,
        physics: None,
        stages: None,
        amplitude_stage: None,
        state_scratch_f64: Vec::new(),
        state_scratch_f64_requires_clear: false,
        values_scratch_f64: Vec::new(),
    };
    let authenticated_normalization_factor = common.normalization_factor;
    common.refresh_normalization_factor();
    let scale = authenticated_normalization_factor
        .abs()
        .max(common.normalization_factor.abs())
        .max(1.0);
    if (common.normalization_factor - authenticated_normalization_factor).abs()
        > 64.0 * f64::EPSILON * scale
    {
        return Err(RusticolError::integrity(
            "spinor runtime normalization defaults do not reproduce the authenticated coupling factor",
        ));
    }
    Ok(LoadedSpinorRuntime { common, lane })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spinor::SPINOR_DAG_ABI;
    use serde_json::json;

    fn manifest() -> SpinorExecutionManifest {
        SpinorExecutionManifest {
            schema_version: PROCESS_ARTIFACT_SCHEMA_VERSION,
            kind: super::super::spinor_manifest::SPINOR_EXECUTION_KIND.to_string(),
            required_runtime_capabilities: vec![SPINOR_DAG_RUNTIME_CAPABILITY.to_string()],
            process: "g g > g g".to_string(),
            key: "gg".to_string(),
            color_accuracy: "lc".to_string(),
            external_pdg_order: vec![21; 4],
            spinor_dag_abi: SPINOR_DAG_ABI.to_string(),
            external_count: 4,
            fixed_color_order: vec![1, 2, 3, 4],
            process_family: None,
            ordered_source_labels: None,
            spinor_parameter_names: None,
            helicity_reduction: "complete-incoherent-sum".to_string(),
            coupling_stripped: true,
            graph_payload: None,
            kernel_pack: None,
            runtime_metadata: None,
        }
    }

    fn physics_payload() -> serde_json::Value {
        let alpha_s = 0.118_f64;
        let global = (4.0 * std::f64::consts::PI * alpha_s).powi(2);
        json!({
            "schema_version": 1,
            "kind": "pyamplicol-resolved-physics",
            "process_id": "gg",
            "process": "g g > g g",
            "color_accuracy": "lc",
            "coverage": {
                "helicities": "complete",
                "color": "selected",
                "color_kind": "physical-lc-flows",
                "structural_zero_helicity_count": 0
            },
            "external_particles": (0..4).map(|index| json!({
                "index": index,
                "label": index + 1,
                "particle": "g",
                "pdg": 21,
                "role": if index < 2 { "initial" } else { "final" },
                "momentum_slot": index,
                "momentum_components": ["E", "px", "py", "pz"]
            })).collect::<Vec<_>>(),
            "helicities": [{
                "id": "h:sum",
                "index": 0,
                "values": [0, 0, 0, 0],
                "computed": true,
                "structural_zero": false,
                "representative_id": "h:sum",
                "coefficient": 1.0
            }],
            "color_components": [{
                "kind": "lc-flow",
                "id": "flow:1,2,3,4",
                "index": 0,
                "word": [1, 2, 3, 4],
                "computed": true,
                "representative_id": "flow:1,2,3,4",
                "coefficient": 1.0
            }],
            "reduction": {
                "kind": "lc-diagonal",
                "groups": [{
                    "id": "reduction:0",
                    "representative_helicity_id": "h:sum",
                    "representative_color_id": "flow:1,2,3,4",
                    "physical_helicity_ids": ["h:sum"],
                    "physical_color_ids": ["flow:1,2,3,4"]
                }]
            },
            "model_parameters": [
                {
                    "name": "normalization.alpha_ew",
                    "kind": "normalization",
                    "default_real": 0.0078125,
                    "default_imaginary": 0.0,
                    "mutable": true
                },
                {
                    "name": "normalization.alpha_s_me_check",
                    "kind": "normalization",
                    "default_real": alpha_s,
                    "default_imaginary": 0.0,
                    "mutable": true
                }
            ],
            "selectors": {
                "helicity": false,
                "color_flow": true,
                "contracted_color": false
            },
            "extensions": {
                "normalization": {
                    "color_accuracy": "lc",
                    "color_factor": 81.0,
                    "average_factor": 256.0,
                    "identical_factor": 2.0,
                    "global_coupling_factor": global,
                    "qcd_coupling_power": 2,
                    "electroweak_coupling_power": 0,
                    "couplings_in_stage_evaluators": false,
                    "coupling_policy": "coupling-stripped spinor DAG with one global reduction factor"
                },
                "spinor_dag": {
                    "helicity_axis": "always-summed-aggregate",
                    "fixed_color_order": [1, 2, 3, 4]
                }
            }
        })
    }

    fn physics() -> ProcessPhysicsV1 {
        let payload = physics_payload();
        ProcessPhysicsV1::from_json(&serde_json::to_vec(&payload).unwrap(), "physics.json").unwrap()
    }

    fn quark_manifest() -> SpinorExecutionManifest {
        SpinorExecutionManifest {
            schema_version: PROCESS_ARTIFACT_SCHEMA_VERSION,
            kind: super::super::spinor_manifest::SPINOR_EXECUTION_KIND.to_string(),
            required_runtime_capabilities: vec![SPINOR_DAG_RUNTIME_CAPABILITY.to_string()],
            process: "u u~ > g g".to_string(),
            key: "uugg".to_string(),
            color_accuracy: "lc".to_string(),
            external_pdg_order: vec![2, -2, 21, 21],
            spinor_dag_abi: SPINOR_DAG_ABI.to_string(),
            external_count: 4,
            fixed_color_order: vec![2, 3, 4, 1],
            process_family: Some(SpinorProcessFamily::SingleMasslessQuarkLine),
            ordered_source_labels: Some(vec![2, 3, 4, 1]),
            spinor_parameter_names: None,
            helicity_reduction: "complete-incoherent-sum".to_string(),
            coupling_stripped: true,
            graph_payload: None,
            kernel_pack: None,
            runtime_metadata: None,
        }
    }

    fn quark_physics() -> ProcessPhysicsV1 {
        let mut payload = physics_payload();
        payload["process_id"] = json!("uugg");
        payload["process"] = json!("u u~ > g g");
        payload["external_particles"] = json!([
            {
                "index": 0,
                "label": 1,
                "particle": "u",
                "pdg": 2,
                "role": "initial",
                "momentum_slot": 0,
                "momentum_components": ["E", "px", "py", "pz"]
            },
            {
                "index": 1,
                "label": 2,
                "particle": "u~",
                "pdg": -2,
                "role": "initial",
                "momentum_slot": 1,
                "momentum_components": ["E", "px", "py", "pz"]
            },
            {
                "index": 2,
                "label": 3,
                "particle": "g",
                "pdg": 21,
                "role": "final",
                "momentum_slot": 2,
                "momentum_components": ["E", "px", "py", "pz"]
            },
            {
                "index": 3,
                "label": 4,
                "particle": "g",
                "pdg": 21,
                "role": "final",
                "momentum_slot": 3,
                "momentum_components": ["E", "px", "py", "pz"]
            }
        ]);
        payload["color_components"][0]["id"] = json!("flow:2,3,4,1");
        payload["color_components"][0]["word"] = json!([2, 3, 4, 1]);
        payload["color_components"][0]["representative_id"] = json!("flow:2,3,4,1");
        payload["reduction"]["groups"][0]["representative_color_id"] = json!("flow:2,3,4,1");
        payload["reduction"]["groups"][0]["physical_color_ids"] = json!(["flow:2,3,4,1"]);
        payload["extensions"]["spinor_dag"] = json!({
            "helicity_axis": "always-summed-aggregate",
            "fixed_color_order": [2, 3, 4, 1],
            "process_family": "single-massless-quark-line",
            "ordered_source_labels": [2, 3, 4, 1]
        });
        ProcessPhysicsV1::from_json(&serde_json::to_vec(&payload).unwrap(), "physics.json").unwrap()
    }

    fn q_z_manifest() -> SpinorExecutionManifest {
        SpinorExecutionManifest {
            schema_version: PROCESS_ARTIFACT_SCHEMA_VERSION,
            kind: super::super::spinor_manifest::SPINOR_EXECUTION_KIND.to_string(),
            required_runtime_capabilities: vec![SPINOR_DAG_RUNTIME_CAPABILITY.to_string()],
            process: "d d~ > z".to_string(),
            key: "ddz".to_string(),
            color_accuracy: "lc".to_string(),
            external_pdg_order: vec![1, -1, 23],
            spinor_dag_abi: SPINOR_DAG_ABI.to_string(),
            external_count: 3,
            fixed_color_order: vec![2, 1],
            process_family: Some(SpinorProcessFamily::SingleMasslessQuarkLineMassiveNeutralVector),
            ordered_source_labels: Some(vec![2, 1, 3]),
            spinor_parameter_names: Some(vec![
                "coupling.10.1_23_1.component_0".to_string(),
                "coupling.10.1_23_1.component_1".to_string(),
            ]),
            helicity_reduction: "complete-incoherent-sum".to_string(),
            coupling_stripped: false,
            graph_payload: None,
            kernel_pack: None,
            runtime_metadata: None,
        }
    }

    fn q_z_physics() -> ProcessPhysicsV1 {
        let alpha_ew = 0.0078125_f64;
        let global = 2.0 * 4.0 * std::f64::consts::PI * alpha_ew;
        let sqrt_two = 2.0_f64.sqrt();
        let mut payload = physics_payload();
        payload["process_id"] = json!("ddz");
        payload["process"] = json!("d d~ > z");
        payload["external_particles"] = json!([
            {
                "index": 0,
                "label": 1,
                "particle": "d",
                "pdg": 1,
                "role": "initial",
                "momentum_slot": 0,
                "momentum_components": ["E", "px", "py", "pz"]
            },
            {
                "index": 1,
                "label": 2,
                "particle": "d~",
                "pdg": -1,
                "role": "initial",
                "momentum_slot": 1,
                "momentum_components": ["E", "px", "py", "pz"]
            },
            {
                "index": 2,
                "label": 3,
                "particle": "z",
                "pdg": 23,
                "role": "final",
                "momentum_slot": 2,
                "momentum_components": ["E", "px", "py", "pz"]
            }
        ]);
        payload["helicities"][0]["values"] = json!([0, 0, 0]);
        payload["color_components"][0]["id"] = json!("flow:2,1");
        payload["color_components"][0]["word"] = json!([2, 1]);
        payload["color_components"][0]["representative_id"] = json!("flow:2,1");
        payload["reduction"]["groups"][0]["representative_color_id"] = json!("flow:2,1");
        payload["reduction"]["groups"][0]["physical_color_ids"] = json!(["flow:2,1"]);
        // Deliberately store the chiral couplings opposite to graph-slot order.
        payload["model_parameters"] = json!([
            {
                "name": "normalization.alpha_ew",
                "kind": "normalization",
                "default_real": alpha_ew,
                "default_imaginary": 0.0,
                "mutable": true
            },
            {
                "name": "normalization.alpha_s_me_check",
                "kind": "normalization",
                "default_real": 0.118,
                "default_imaginary": 0.0,
                "mutable": true
            },
            {
                "name": "particle.23.mass",
                "kind": "mass",
                "default_real": 10.0,
                "default_imaginary": 0.0,
                "mutable": true
            },
            {
                "name": "coupling.10.1_23_1.component_1",
                "kind": "coupling",
                "default_real": sqrt_two * 3.0,
                "default_imaginary": 0.0,
                "mutable": true
            },
            {
                "name": "coupling.10.1_23_1.component_0",
                "kind": "coupling",
                "default_real": sqrt_two * 2.0,
                "default_imaginary": 0.0,
                "mutable": true
            }
        ]);
        payload["extensions"]["normalization"] = json!({
            "color_accuracy": "lc",
            "color_factor": 3.0,
            "average_factor": 36.0,
            "identical_factor": 1.0,
            "global_coupling_factor": global,
            "qcd_coupling_power": 0,
            "electroweak_coupling_power": 1,
            "couplings_in_stage_evaluators": true,
            "coupling_policy": "spinor DAG with local chiral couplings and one global reduction factor"
        });
        payload["extensions"]["spinor_dag"] = json!({
            "helicity_axis": "always-summed-aggregate",
            "fixed_color_order": [2, 1],
            "process_family": "single-massless-quark-line-massive-neutral-vector",
            "ordered_source_labels": [2, 1, 3],
            "spinor_parameter_names": [
                "coupling.10.1_23_1.component_0",
                "coupling.10.1_23_1.component_1"
            ]
        });
        ProcessPhysicsV1::from_json(&serde_json::to_vec(&payload).unwrap(), "physics.json").unwrap()
    }

    fn massive_quark_manifest() -> SpinorExecutionManifest {
        SpinorExecutionManifest {
            schema_version: PROCESS_ARTIFACT_SCHEMA_VERSION,
            kind: super::super::spinor_manifest::SPINOR_EXECUTION_KIND.to_string(),
            required_runtime_capabilities: vec![SPINOR_DAG_RUNTIME_CAPABILITY.to_string()],
            process: "g g > t t~".to_string(),
            key: "ggtt".to_string(),
            color_accuracy: "lc".to_string(),
            external_pdg_order: vec![21, 21, 6, -6],
            spinor_dag_abi: SPINOR_DAG_ABI.to_string(),
            external_count: 4,
            fixed_color_order: vec![3, 1, 2, 4],
            process_family: Some(SpinorProcessFamily::SingleMassiveQuarkLine),
            ordered_source_labels: Some(vec![3, 1, 2, 4]),
            spinor_parameter_names: Some(vec![
                TOP_MASS_PARAMETER_NAME.to_string(),
                TOP_WIDTH_PARAMETER_NAME.to_string(),
            ]),
            helicity_reduction: "complete-incoherent-sum".to_string(),
            coupling_stripped: true,
            graph_payload: None,
            kernel_pack: None,
            runtime_metadata: None,
        }
    }

    fn massive_quark_physics() -> ProcessPhysicsV1 {
        let mut payload = physics_payload();
        payload["process_id"] = json!("ggtt");
        payload["process"] = json!("g g > t t~");
        payload["external_particles"] = json!([
            {
                "index": 0, "label": 1, "particle": "g", "pdg": 21,
                "role": "initial", "momentum_slot": 0,
                "momentum_components": ["E", "px", "py", "pz"]
            },
            {
                "index": 1, "label": 2, "particle": "g", "pdg": 21,
                "role": "initial", "momentum_slot": 1,
                "momentum_components": ["E", "px", "py", "pz"]
            },
            {
                "index": 2, "label": 3, "particle": "t", "pdg": 6,
                "role": "final", "momentum_slot": 2,
                "momentum_components": ["E", "px", "py", "pz"]
            },
            {
                "index": 3, "label": 4, "particle": "t~", "pdg": -6,
                "role": "final", "momentum_slot": 3,
                "momentum_components": ["E", "px", "py", "pz"]
            }
        ]);
        payload["color_components"][0]["id"] = json!("flow:3,1,2,4");
        payload["color_components"][0]["word"] = json!([3, 1, 2, 4]);
        payload["color_components"][0]["representative_id"] = json!("flow:3,1,2,4");
        payload["reduction"]["groups"][0]["representative_color_id"] = json!("flow:3,1,2,4");
        payload["reduction"]["groups"][0]["physical_color_ids"] = json!(["flow:3,1,2,4"]);
        // Deliberately store width before mass; graph binding is by name.
        payload["model_parameters"] = json!([
            {
                "name": "normalization.alpha_ew", "kind": "normalization",
                "default_real": 0.0078125, "default_imaginary": 0.0,
                "mutable": true
            },
            {
                "name": "normalization.alpha_s_me_check", "kind": "normalization",
                "default_real": 0.118, "default_imaginary": 0.0,
                "mutable": true
            },
            {
                "name": TOP_WIDTH_PARAMETER_NAME, "kind": "width",
                "default_real": 1.4915, "default_imaginary": 0.0,
                "mutable": true
            },
            {
                "name": TOP_MASS_PARAMETER_NAME, "kind": "mass",
                "default_real": 173.0, "default_imaginary": 0.0,
                "mutable": true
            }
        ]);
        payload["extensions"]["normalization"]["color_factor"] = json!(27.0);
        payload["extensions"]["normalization"]["identical_factor"] = json!(1.0);
        payload["extensions"]["spinor_dag"] = json!({
            "helicity_axis": "always-summed-aggregate",
            "fixed_color_order": [3, 1, 2, 4],
            "process_family": "single-massive-quark-line",
            "ordered_source_labels": [3, 1, 2, 4],
            "spinor_parameter_names": [TOP_MASS_PARAMETER_NAME, TOP_WIDTH_PARAMETER_NAME]
        });
        ProcessPhysicsV1::from_json(&serde_json::to_vec(&payload).unwrap(), "physics.json").unwrap()
    }

    #[test]
    fn mutable_alpha_s_updates_spinor_normalization() {
        let mut loaded = load_spinor_native_runtime(&manifest(), &physics()).unwrap();
        let initial = loaded.common.normalization_factor;
        loaded
            .common
            .apply_model_parameter_overrides(&BTreeMap::from([(
                "normalization.alpha_s_me_check".to_string(),
                (0.236, 0.0),
            )]))
            .unwrap();
        assert!((loaded.common.normalization_factor / initial - 4.0).abs() <= 1.0e-13);
    }

    #[test]
    fn loads_single_massless_quark_line_from_authenticated_traversal() {
        let loaded = load_spinor_native_runtime(&quark_manifest(), &quark_physics()).unwrap();
        let expected = build_helicity_summed_quark_gluon_bg_spinor_dag(&[0, 1, 2, 3]).unwrap();
        assert_eq!(loaded.common.external_pdg_order, [2, -2, 21, 21]);
        assert_eq!(loaded.common.amplitude_output_count, expected.roots().len());
        assert_eq!(loaded.common.particle_masses.get(&2), Some(&0.0));
        assert_eq!(loaded.common.particle_masses.get(&-2), Some(&0.0));
    }

    #[test]
    fn loads_massive_quark_flow_and_mutates_name_bound_mass_and_width() {
        let LoadedSpinorRuntime {
            mut common,
            mut lane,
        } = load_spinor_native_runtime(&massive_quark_manifest(), &massive_quark_physics())
            .unwrap();
        assert_eq!(common.particle_masses.get(&6), Some(&173.0));
        assert_eq!(common.particle_masses.get(&-6), Some(&173.0));
        assert_eq!(
            common
                .particle_mass_parameter_names
                .get(&-6)
                .map(String::as_str),
            Some(TOP_MASS_PARAMETER_NAME)
        );

        let physical = [
            [500.0, 0.0, 0.0, 500.0],
            [500.0, 0.0, 0.0, -500.0],
            [
                500.0,
                319.12027591281498,
                6.6740914282666104,
                343.78584322938013,
            ],
            [
                500.0,
                -319.12027591281498,
                -6.6740914282666104,
                -343.78584322938013,
            ],
        ];
        let ordered_outgoing = [
            physical[2],
            physical[0].map(|value| -value),
            physical[1].map(|value| -value),
            physical[3],
        ];
        let direct =
            build_helicity_summed_massive_quark_two_gluon_spinor_dag(&[0, 1, 2, 3]).unwrap();
        let expected = direct
            .evaluate_with_parameters(
                &ordered_outgoing,
                &[
                    crate::EagerComplex64::new(173.0, 0.0),
                    crate::EagerComplex64::new(1.4915, 0.0),
                ],
            )
            .unwrap()
            .helicity_sum()
            * common.normalization_factor;
        let flat = physical
            .iter()
            .flat_map(|momentum| momentum.iter().copied())
            .collect::<Vec<_>>();
        let batch = F64MomentumBatchView::from_contiguous_prevalidated(&flat, 1, 4, None).unwrap();
        let mut output = [0.0];
        lane.run_total_into_unprofiled(&common, batch, None, None, &mut output)
            .unwrap();
        assert!((output[0] - expected).abs() <= 1.0e-12 * expected.abs().max(1.0));

        common
            .apply_model_parameter_overrides(&BTreeMap::from([(
                TOP_WIDTH_PARAMETER_NAME.to_string(),
                (0.0, 0.0),
            )]))
            .unwrap();
        let stable = direct
            .evaluate_with_parameters(
                &ordered_outgoing,
                &[
                    crate::EagerComplex64::new(173.0, 0.0),
                    crate::EagerComplex64::new(0.0, 0.0),
                ],
            )
            .unwrap()
            .helicity_sum()
            * common.normalization_factor;
        lane.run_total_into_unprofiled(&common, batch, None, None, &mut output)
            .unwrap();
        assert!((output[0] - stable).abs() <= 1.0e-12 * stable.abs().max(1.0));
        assert_ne!(stable, expected);

        common
            .apply_model_parameter_overrides(&BTreeMap::from([(
                TOP_MASS_PARAMETER_NAME.to_string(),
                (174.0, 0.0),
            )]))
            .unwrap();
        assert_eq!(common.particle_masses.get(&6), Some(&174.0));
        assert_eq!(common.particle_masses.get(&-6), Some(&174.0));
        let mutated = direct
            .evaluate_with_parameters(
                &ordered_outgoing,
                &[
                    crate::EagerComplex64::new(174.0, 0.0),
                    crate::EagerComplex64::new(0.0, 0.0),
                ],
            )
            .unwrap()
            .helicity_sum()
            * common.normalization_factor;
        lane.run_total_into_unprofiled(&common, batch, None, None, &mut output)
            .unwrap();
        assert!((output[0] - mutated).abs() <= 1.0e-12 * mutated.abs().max(1.0));
        assert_ne!(mutated, expected);
    }

    #[test]
    fn loads_q_z_family_with_common_mass_and_name_bound_chiral_couplings() {
        let LoadedSpinorRuntime {
            mut common,
            mut lane,
        } = load_spinor_native_runtime(&q_z_manifest(), &q_z_physics()).unwrap();
        assert_eq!(common.particle_masses.get(&23), Some(&10.0));
        assert_eq!(
            common
                .particle_mass_parameter_names
                .get(&23)
                .map(String::as_str),
            Some("particle.23.mass")
        );

        let physical = [
            [5.0, 0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0, -5.0],
            [10.0, 0.0, 0.0, 0.0],
        ];
        let flat = physical
            .iter()
            .flat_map(|momentum| momentum.iter().copied())
            .collect::<Vec<_>>();
        let mut output = [0.0];
        lane.run_total_into_unprofiled(
            &common,
            F64MomentumBatchView::from_contiguous_prevalidated(&flat, 1, 3, None).unwrap(),
            None,
            None,
            &mut output,
        )
        .unwrap();
        assert!((output[0] - 2600.0 * common.normalization_factor).abs() <= 1.0e-11);

        common
            .apply_model_parameter_overrides(&BTreeMap::from([
                ("coupling.10.1_23_1.component_0".to_string(), (0.0, 0.0)),
                ("particle.23.mass".to_string(), (11.0, 0.0)),
            ]))
            .unwrap();
        assert_eq!(common.particle_masses.get(&23), Some(&11.0));
        lane.run_total_into_unprofiled(
            &common,
            F64MomentumBatchView::from_contiguous_prevalidated(&flat, 1, 3, None).unwrap(),
            None,
            None,
            &mut output,
        )
        .unwrap();
        assert!((output[0] - 1800.0 * common.normalization_factor).abs() <= 1.0e-11);
    }

    #[test]
    fn rejects_q_z_graph_parameter_bound_to_a_non_coupling_record() {
        let manifest = q_z_manifest();
        let mut payload = q_z_physics();
        payload
            .model_parameters
            .iter_mut()
            .find(|parameter| parameter.name == "coupling.10.1_23_1.component_0")
            .unwrap()
            .kind = crate::ParameterKind::Normalization;

        let error = match load_spinor_native_runtime(&manifest, &payload) {
            Ok(_) => panic!("non-coupling graph parameter should be rejected"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("not declared as a coupling"));
    }
}
