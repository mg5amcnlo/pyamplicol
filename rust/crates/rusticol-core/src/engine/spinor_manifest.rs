// SPDX-License-Identifier: 0BSD

//! Compact semantic manifest for the developer-only fixed-color spinor lane.

use super::*;
use crate::ArtifactProcess;
use crate::spinor::{SPINOR_DAG_ABI, SPINOR_DAG_BINARY_ABI};
use serde::Deserialize;
use std::collections::BTreeSet;

pub(super) const SPINOR_EXECUTION_KIND: &str = "pyamplicol-runtime-spinor-dag-execution";
pub(super) const TOP_MASS_PARAMETER_NAME: &str = "particle.6.mass";
pub(super) const TOP_WIDTH_PARAMETER_NAME: &str = "particle.6.width";
const SPINOR_HELICITY_REDUCTION: &str = "complete-incoherent-sum";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum SpinorProcessFamily {
    PureGluon,
    SingleMasslessQuarkLine,
    SingleMassiveQuarkLine,
    SingleMasslessQuarkLineMassiveNeutralVector,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SpinorExecutionManifest {
    pub(super) schema_version: u32,
    pub(super) kind: String,
    pub(super) required_runtime_capabilities: Vec<String>,
    pub(super) process: String,
    pub(super) key: String,
    pub(super) color_accuracy: String,
    pub(super) external_pdg_order: Vec<i32>,
    pub(super) spinor_dag_abi: String,
    pub(super) external_count: u16,
    #[serde(default)]
    pub(super) fixed_color_order: Vec<usize>,
    #[serde(default)]
    pub(super) process_family: Option<SpinorProcessFamily>,
    #[serde(default)]
    pub(super) ordered_source_labels: Option<Vec<usize>>,
    #[serde(default)]
    pub(super) spinor_parameter_names: Option<Vec<String>>,
    pub(super) helicity_reduction: String,
    pub(super) coupling_stripped: bool,
    #[serde(default)]
    pub(super) graph_payload: Option<SpinorGraphPayloadReference>,
    #[serde(default)]
    pub(super) kernel_pack: Option<RecurrenceKernelPackReference>,
    #[serde(default)]
    pub(super) runtime_metadata: Option<RecurrenceRuntimeMetadata>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SpinorGraphPayloadReference {
    pub(super) abi: String,
    pub(super) path: String,
}

impl SpinorExecutionManifest {
    pub(super) fn process_family(&self) -> SpinorProcessFamily {
        self.process_family
            .unwrap_or(SpinorProcessFamily::PureGluon)
    }

    pub(super) fn ordered_source_labels(&self) -> &[usize] {
        self.ordered_source_labels
            .as_deref()
            .unwrap_or(&self.fixed_color_order)
    }

    pub(super) fn spinor_parameter_names(&self) -> &[String] {
        self.spinor_parameter_names.as_deref().unwrap_or(&[])
    }

    pub(super) const fn is_graph_backed(&self) -> bool {
        self.graph_payload.is_some()
    }

    fn validate(&self, outer: &ArtifactProcess) -> RusticolResult<()> {
        if self.schema_version != PROCESS_ARTIFACT_SCHEMA_VERSION
            || self.kind != SPINOR_EXECUTION_KIND
        {
            return Err(RusticolError::compatibility(format!(
                "unsupported spinor execution kind {:?} schema {}; regenerate the artifact",
                self.kind, self.schema_version
            )));
        }
        if self.process != outer.expression
            || self.key != outer.id
            || self.color_accuracy != outer.color_accuracy
            || self.external_pdg_order != outer.external_pdgs
        {
            return Err(RusticolError::integrity(format!(
                "spinor execution manifest does not match outer process {:?}",
                outer.id
            )));
        }
        if self.required_runtime_capabilities != [SPINOR_DAG_RUNTIME_CAPABILITY]
            || self.color_accuracy != "lc"
            || self.spinor_dag_abi != SPINOR_DAG_ABI
            || self.helicity_reduction != SPINOR_HELICITY_REDUCTION
        {
            return Err(RusticolError::compatibility(
                "spinor execution requires the fixed-color f64 ABI and complete helicity sum",
            ));
        }
        let external_count = usize::from(self.external_count);
        if self.external_pdg_order.len() != external_count {
            return Err(RusticolError::artifact(
                "spinor execution external count does not match its PDG order",
            ));
        }
        if let Some(graph) = &self.graph_payload {
            if graph.abi != SPINOR_DAG_BINARY_ABI
                || confined_internal_path(&graph.path, "spinor graph payload path").is_err()
            {
                return Err(RusticolError::compatibility(
                    "spinor graph payload uses an unsupported binary contract",
                ));
            }
            if external_count < 2
                || self.process_family.is_some()
                || !self.fixed_color_order.is_empty()
                || self.ordered_source_labels.is_some()
                || self.spinor_parameter_names.is_some()
                || self.coupling_stripped
            {
                return Err(RusticolError::artifact(
                    "graph-backed spinor execution must use payload-owned generic source and parameter bindings",
                ));
            }
            if let Some(kernel_pack) = &self.kernel_pack {
                if kernel_pack.manifest_path != RECURRENCE_KERNEL_PACK_MANIFEST_PATH
                    || kernel_pack.payload_root != RECURRENCE_KERNEL_PAYLOAD_ROOT
                {
                    return Err(RusticolError::compatibility(
                        "graph-backed spinor execution uses an unsupported prepared-kernel location",
                    ));
                }
            }
            let metadata = self.runtime_metadata.as_ref().ok_or_else(|| {
                RusticolError::artifact("graph-backed spinor execution has no runtime metadata")
            })?;
            if metadata.color_contraction.is_some()
                || metadata.public_color_flows.len() != 1
                || !metadata.normalization.couplings_in_stage_evaluators
            {
                return Err(RusticolError::artifact(
                    "graph-backed spinor execution requires one retained LC component and local coupling ownership",
                ));
            }
            let flow = &metadata.public_color_flows[0];
            if flow.public_id.is_empty()
                || flow.public_id.trim() != flow.public_id
                || flow.construction_sector_id != 0
                || flow.target_sector_id != 0
            {
                return Err(RusticolError::integrity(
                    "graph-backed spinor color component must have a canonical ID and bind the sole construction and target sector",
                ));
            }
            metadata
                .validate_spinor_runtime_domain(&self.external_pdg_order, &self.color_accuracy)?;
            return Ok(());
        }
        if self.kernel_pack.is_some() || self.runtime_metadata.is_some() {
            return Err(RusticolError::artifact(
                "legacy spinor execution must not carry graph runtime metadata",
            ));
        }
        match self.process_family() {
            SpinorProcessFamily::PureGluon => {
                if !(4..=6).contains(&external_count)
                    || self.external_pdg_order.iter().any(|pdg| *pdg != 21)
                    || !self.coupling_stripped
                    || self.spinor_parameter_names.is_some()
                {
                    return Err(RusticolError::artifact(
                        "the pure-gluon spinor family requires four to six gluons and no local parameters",
                    ));
                }
                self.validate_complete_colored_traversal(external_count)?;
            }
            SpinorProcessFamily::SingleMasslessQuarkLine => {
                if !(4..=6).contains(&external_count)
                    || self.process_family.is_none()
                    || self.ordered_source_labels.is_none()
                    || !self.coupling_stripped
                    || self.spinor_parameter_names.is_some()
                {
                    return Err(RusticolError::artifact(
                        "the massless open-quark-line spinor family requires four to six particles, an explicit traversal, and no local parameters",
                    ));
                }
                self.validate_complete_colored_traversal(external_count)?;
                let quarks = self
                    .external_pdg_order
                    .iter()
                    .copied()
                    .filter(|pdg| *pdg != 21)
                    .collect::<Vec<_>>();
                let flavour = quarks.first().and_then(|pdg| pdg.checked_abs());
                if quarks.len() != 2
                    || !flavour.is_some_and(|pdg| (1..=5).contains(&pdg))
                    || quarks[1].checked_abs() != flavour
                    || quarks[0].checked_neg() != Some(quarks[1])
                {
                    return Err(RusticolError::artifact(
                        "the massless open-quark-line spinor family requires one light quark pair and gluons",
                    ));
                }
            }
            SpinorProcessFamily::SingleMassiveQuarkLine => {
                let parameter_names = self.spinor_parameter_names();
                if external_count != 4
                    || self.process_family.is_none()
                    || self.ordered_source_labels.is_none()
                    || self.spinor_parameter_names.is_none()
                    || !self.coupling_stripped
                    || self.external_pdg_order != [21, 21, 6, -6]
                    || parameter_names != [TOP_MASS_PARAMETER_NAME, TOP_WIDTH_PARAMETER_NAME]
                {
                    return Err(RusticolError::artifact(
                        "the massive open-quark-line spinor family requires g g > t tbar and ordered top mass/width graph parameters",
                    ));
                }
                self.validate_complete_colored_traversal(external_count)?;
                if self.fixed_color_order != [3, 1, 2, 4] && self.fixed_color_order != [3, 2, 1, 4]
                {
                    return Err(RusticolError::artifact(
                        "the massive open-quark-line spinor family requires one of the two t-gluon-gluon-tbar flows",
                    ));
                }
            }
            SpinorProcessFamily::SingleMasslessQuarkLineMassiveNeutralVector => {
                let z_labels = self
                    .external_pdg_order
                    .iter()
                    .enumerate()
                    .filter_map(|(index, pdg)| (*pdg == 23).then_some(index + 1))
                    .collect::<Vec<_>>();
                let quarks = self
                    .external_pdg_order
                    .iter()
                    .copied()
                    .filter(|pdg| *pdg != 21 && *pdg != 23)
                    .collect::<Vec<_>>();
                let flavour = quarks.first().and_then(|pdg| pdg.checked_abs());
                let parameter_names = self.spinor_parameter_names();
                if !(3..=5).contains(&external_count)
                    || self.process_family.is_none()
                    || self.ordered_source_labels.is_none()
                    || self.spinor_parameter_names.is_none()
                    || self.coupling_stripped
                    || z_labels.len() != 1
                    || self
                        .external_pdg_order
                        .iter()
                        .filter(|pdg| **pdg == 21)
                        .count()
                        != external_count - 3
                    || quarks.len() != 2
                    || !flavour.is_some_and(|pdg| (1..=5).contains(&pdg))
                    || quarks[1].checked_abs() != flavour
                    || quarks[0].checked_neg() != Some(quarks[1])
                    || parameter_names.len() != 2
                    || parameter_names.iter().any(String::is_empty)
                    || parameter_names[0] == parameter_names[1]
                {
                    return Err(RusticolError::artifact(
                        "the q-Z spinor family requires one light quark pair, one Z, zero to two gluons, and two ordered local chiral couplings",
                    ));
                }
                let z_label = z_labels[0];
                let colored_labels = (1..=external_count)
                    .filter(|label| *label != z_label)
                    .collect::<BTreeSet<_>>();
                let fixed_labels = self
                    .fixed_color_order
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>();
                let ordered_source_labels = self.ordered_source_labels();
                let all_source_labels = ordered_source_labels
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>();
                if self.fixed_color_order.len() != external_count - 1
                    || fixed_labels != colored_labels
                    || ordered_source_labels.len() != external_count
                    || all_source_labels != (1..=external_count).collect()
                    || !ordered_source_labels.starts_with(&self.fixed_color_order)
                    || ordered_source_labels.last() != Some(&z_label)
                {
                    return Err(RusticolError::artifact(
                        "the q-Z spinor source traversal must be the complete colored open string followed by Z",
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_complete_colored_traversal(&self, external_count: usize) -> RusticolResult<()> {
        let fixed_order = self
            .fixed_color_order
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let ordered_source_labels = self.ordered_source_labels();
        let source_order = ordered_source_labels
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if self.fixed_color_order.len() != external_count
            || fixed_order != (1..=external_count).collect()
            || ordered_source_labels.len() != external_count
            || source_order != (1..=external_count).collect()
            || ordered_source_labels != self.fixed_color_order
        {
            return Err(RusticolError::artifact(
                "spinor source traversal must equal the complete fixed color order",
            ));
        }
        Ok(())
    }
}

pub(super) fn parse_spinor_execution_manifest(
    bytes: &[u8],
    outer: &ArtifactProcess,
) -> RusticolResult<SpinorExecutionManifest> {
    let manifest: SpinorExecutionManifest = serde_json::from_slice(bytes).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse spinor execution manifest for process {:?}: {error}",
            outer.id
        ))
    })?;
    manifest.validate(outer)?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn scalar_graph_manifest() -> Value {
        let crossing = json!({
            "momentum_transform": "identity",
            "helicity_factor": 1,
            "chirality_factor": 1,
            "spin_state_factor": 1,
            "phase": [1.0, 0.0]
        });
        json!({
            "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
            "kind": SPINOR_EXECUTION_KIND,
            "required_runtime_capabilities": [SPINOR_DAG_RUNTIME_CAPABILITY],
            "process": "scalar_0 scalar_0 > scalar_0 scalar_0",
            "key": "scalars_0000",
            "color_accuracy": "lc",
            "external_pdg_order": [1000, 1000, 1000, 1000],
            "spinor_dag_abi": SPINOR_DAG_ABI,
            "external_count": 4,
            "fixed_color_order": [],
            "helicity_reduction": SPINOR_HELICITY_REDUCTION,
            "coupling_stripped": false,
            "graph_payload": {
                "abi": SPINOR_DAG_BINARY_ABI,
                "path": "spinor-dag-v3.bin"
            },
            "kernel_pack": {
                "manifest_path": RECURRENCE_KERNEL_PACK_MANIFEST_PATH,
                "payload_root": RECURRENCE_KERNEL_PAYLOAD_ROOT
            },
            "runtime_metadata": {
                "public_color_flows": [{
                    "public_id": "flow:singlet",
                    "construction_sector_id": 0,
                    "target_sector_id": 0
                }],
                "runtime_parameters": [],
                "prepared_parameter_defaults": [],
                "parameter_projection": [],
                "source_templates": [{
                    "source_template_id": 0,
                    "current_state_template_id": 0,
                    "dimension": 1,
                    "helicity": 0,
                    "chirality": 0,
                    "spin_state": 0,
                    "source_ir": {
                        "identity": {
                            "canonical_id": "model:scalar_0",
                            "species_id": "model:scalar_0",
                            "anti_canonical_id": "model:scalar_0",
                            "display_name": "scalar_0",
                            "anti_display_name": "scalar_0",
                            "pdg_label": 1000,
                            "anti_pdg_label": 1000,
                            "orientation": "self-conjugate",
                            "self_conjugate": true
                        },
                        "statistics": "boson",
                        "wavefunction_family": "scalar",
                        "component_dimension": 1,
                        "states": [{"helicity": 0, "chirality": 0, "spin_state": 0}],
                        "crossing": crossing,
                        "basis": "scalar",
                        "mass_parameter": null,
                        "width_parameter": null
                    },
                    "crossing": crossing
                }],
                "external_legs": [
                    {"source_slot": 0, "public_label": 1, "physical_pdg": 1000,
                     "outgoing_pdg": 1000, "is_initial": true},
                    {"source_slot": 1, "public_label": 2, "physical_pdg": 1000,
                     "outgoing_pdg": 1000, "is_initial": true},
                    {"source_slot": 2, "public_label": 3, "physical_pdg": 1000,
                     "outgoing_pdg": 1000, "is_initial": false},
                    {"source_slot": 3, "public_label": 4, "physical_pdg": 1000,
                     "outgoing_pdg": 1000, "is_initial": false}
                ],
                "particle_masses": [{"outgoing_pdg": 1000, "mass": 0.0}],
                "normalization": {
                    "color_accuracy": "lc",
                    "color_factor": 1.0,
                    "average_factor": 1.0,
                    "identical_factor": 2.0,
                    "global_coupling_factor": 1.0,
                    "qcd_coupling_power": 0,
                    "electroweak_coupling_power": 0,
                    "couplings_in_stage_evaluators": true,
                    "coupling_policy": "authenticated local scalar coupling"
                },
                "color_contraction": null
            }
        })
    }

    fn scalar_outer() -> ArtifactProcess {
        ArtifactProcess {
            id: "scalars_0000".to_string(),
            expression: "scalar_0 scalar_0 > scalar_0 scalar_0".to_string(),
            color_accuracy: "lc".to_string(),
            external_pdgs: vec![1000; 4],
            physics_path: "processes/scalars_0000/physics.json".to_string(),
            required_runtime_capabilities: vec![SPINOR_DAG_RUNTIME_CAPABILITY.to_string()],
            aliases: Vec::new(),
        }
    }

    #[test]
    fn graph_backed_scalar_manifest_is_family_free_and_binds_singlet_flow() {
        let payload = scalar_graph_manifest();
        let parsed = parse_spinor_execution_manifest(
            &serde_json::to_vec(&payload).unwrap(),
            &scalar_outer(),
        )
        .unwrap();
        assert!(parsed.is_graph_backed());
        assert!(parsed.process_family.is_none());
        assert_eq!(
            parsed.runtime_metadata.unwrap().public_color_flows[0].public_id,
            "flow:singlet"
        );

        let mut wrong_flow = payload.clone();
        wrong_flow["runtime_metadata"]["public_color_flows"][0]["target_sector_id"] = json!(1);
        assert!(
            parse_spinor_execution_manifest(
                &serde_json::to_vec(&wrong_flow).unwrap(),
                &scalar_outer(),
            )
            .is_err()
        );

        let mut family_tagged = payload;
        family_tagged["process_family"] = json!("pure-gluon");
        assert!(
            parse_spinor_execution_manifest(
                &serde_json::to_vec(&family_tagged).unwrap(),
                &scalar_outer(),
            )
            .is_err()
        );
    }

    #[test]
    fn fixed_color_order_must_be_a_permutation() {
        let manifest = SpinorExecutionManifest {
            schema_version: PROCESS_ARTIFACT_SCHEMA_VERSION,
            kind: SPINOR_EXECUTION_KIND.to_string(),
            required_runtime_capabilities: vec![SPINOR_DAG_RUNTIME_CAPABILITY.to_string()],
            process: "g g > g g".to_string(),
            key: "gg".to_string(),
            color_accuracy: "lc".to_string(),
            external_pdg_order: vec![21; 4],
            spinor_dag_abi: SPINOR_DAG_ABI.to_string(),
            external_count: 4,
            fixed_color_order: vec![1, 2, 2, 4],
            process_family: None,
            ordered_source_labels: None,
            spinor_parameter_names: None,
            helicity_reduction: SPINOR_HELICITY_REDUCTION.to_string(),
            coupling_stripped: true,
            graph_payload: None,
            kernel_pack: None,
            runtime_metadata: None,
        };
        let outer = ArtifactProcess {
            id: "gg".to_string(),
            expression: "g g > g g".to_string(),
            color_accuracy: "lc".to_string(),
            external_pdgs: vec![21; 4],
            physics_path: "processes/gg/physics.json".to_string(),
            required_runtime_capabilities: vec![SPINOR_DAG_RUNTIME_CAPABILITY.to_string()],
            aliases: Vec::new(),
        };
        assert!(manifest.validate(&outer).is_err());
    }

    #[test]
    fn single_massless_quark_line_authenticates_its_traversal() {
        let manifest = SpinorExecutionManifest {
            schema_version: PROCESS_ARTIFACT_SCHEMA_VERSION,
            kind: SPINOR_EXECUTION_KIND.to_string(),
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
            helicity_reduction: SPINOR_HELICITY_REDUCTION.to_string(),
            coupling_stripped: true,
            graph_payload: None,
            kernel_pack: None,
            runtime_metadata: None,
        };
        let outer = ArtifactProcess {
            id: "uugg".to_string(),
            expression: "u u~ > g g".to_string(),
            color_accuracy: "lc".to_string(),
            external_pdgs: vec![2, -2, 21, 21],
            physics_path: "processes/uugg/physics.json".to_string(),
            required_runtime_capabilities: vec![SPINOR_DAG_RUNTIME_CAPABILITY.to_string()],
            aliases: Vec::new(),
        };
        manifest.validate(&outer).unwrap();

        let mut wrong_label = manifest;
        wrong_label.ordered_source_labels = Some(vec![1, 3, 4, 2]);
        assert!(wrong_label.validate(&outer).is_err());
    }

    #[test]
    fn massive_quark_line_authenticates_flow_and_parameter_order() {
        let manifest = SpinorExecutionManifest {
            schema_version: PROCESS_ARTIFACT_SCHEMA_VERSION,
            kind: SPINOR_EXECUTION_KIND.to_string(),
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
            helicity_reduction: SPINOR_HELICITY_REDUCTION.to_string(),
            coupling_stripped: true,
            graph_payload: None,
            kernel_pack: None,
            runtime_metadata: None,
        };
        let outer = ArtifactProcess {
            id: "ggtt".to_string(),
            expression: "g g > t t~".to_string(),
            color_accuracy: "lc".to_string(),
            external_pdgs: vec![21, 21, 6, -6],
            physics_path: "processes/ggtt/physics.json".to_string(),
            required_runtime_capabilities: vec![SPINOR_DAG_RUNTIME_CAPABILITY.to_string()],
            aliases: Vec::new(),
        };
        manifest.validate(&outer).unwrap();

        let mut wrong_order = manifest.clone();
        wrong_order.spinor_parameter_names = Some(vec![
            TOP_WIDTH_PARAMETER_NAME.to_string(),
            TOP_MASS_PARAMETER_NAME.to_string(),
        ]);
        assert!(wrong_order.validate(&outer).is_err());

        let mut unsupported_flow = manifest;
        unsupported_flow.fixed_color_order = vec![3, 1, 4, 2];
        unsupported_flow.ordered_source_labels = Some(vec![3, 1, 4, 2]);
        assert!(unsupported_flow.validate(&outer).is_err());
    }

    #[test]
    fn q_z_family_separates_colored_and_full_source_orders() {
        let manifest = SpinorExecutionManifest {
            schema_version: PROCESS_ARTIFACT_SCHEMA_VERSION,
            kind: SPINOR_EXECUTION_KIND.to_string(),
            required_runtime_capabilities: vec![SPINOR_DAG_RUNTIME_CAPABILITY.to_string()],
            process: "d d~ > z g".to_string(),
            key: "ddzg".to_string(),
            color_accuracy: "lc".to_string(),
            external_pdg_order: vec![1, -1, 23, 21],
            spinor_dag_abi: SPINOR_DAG_ABI.to_string(),
            external_count: 4,
            fixed_color_order: vec![2, 4, 1],
            process_family: Some(SpinorProcessFamily::SingleMasslessQuarkLineMassiveNeutralVector),
            ordered_source_labels: Some(vec![2, 4, 1, 3]),
            spinor_parameter_names: Some(vec![
                "coupling.10.1_23_1.component_0".to_string(),
                "coupling.10.1_23_1.component_1".to_string(),
            ]),
            helicity_reduction: SPINOR_HELICITY_REDUCTION.to_string(),
            coupling_stripped: false,
            graph_payload: None,
            kernel_pack: None,
            runtime_metadata: None,
        };
        let outer = ArtifactProcess {
            id: "ddzg".to_string(),
            expression: "d d~ > z g".to_string(),
            color_accuracy: "lc".to_string(),
            external_pdgs: vec![1, -1, 23, 21],
            physics_path: "processes/ddzg/physics.json".to_string(),
            required_runtime_capabilities: vec![SPINOR_DAG_RUNTIME_CAPABILITY.to_string()],
            aliases: Vec::new(),
        };
        manifest.validate(&outer).unwrap();

        let mut z_in_color_word = manifest.clone();
        z_in_color_word.fixed_color_order = vec![2, 3, 1];
        assert!(z_in_color_word.validate(&outer).is_err());

        let mut duplicate_parameter = manifest;
        duplicate_parameter.spinor_parameter_names = Some(vec!["gL".into(), "gL".into()]);
        assert!(duplicate_parameter.validate(&outer).is_err());
    }
}
