// SPDX-License-Identifier: 0BSD

//! Native fixed-color, always-helicity-summed spinor evaluation lane.

use super::recurrence_lane::{
    PreparedParameterProjectionEntry, initialize_prepared_parameter_planes,
};
use super::*;
use crate::spinor::{
    SpinorBatchWorkspace, SpinorDag, SpinorDagPayloadV3, SpinorSourceInputBinding,
    SpinorSourceInputKind,
};

const SPINOR_POINT_TILE_SIZE: usize = 1024;

pub(super) struct SpinorNativeRuntime {
    dag: SpinorDag,
    workspace: SpinorBatchWorkspace,
    sources: SpinorSourceLayout,
    parameters: SpinorParameterLayout,
    parameter_values: Vec<crate::EagerComplex64>,
    prepared_parameter_re: Vec<f64>,
    prepared_parameter_im: Vec<f64>,
    flat_momenta: Vec<f64>,
}

enum SpinorSourceLayout {
    Legacy(Box<[usize]>),
    Payload {
        bindings: Box<[SpinorSourceInputBinding]>,
        kinds: Box<[SpinorSourceInputKind]>,
    },
}

impl SpinorSourceLayout {
    fn len(&self) -> usize {
        match self {
            Self::Legacy(labels) => labels.len(),
            Self::Payload { bindings, .. } => bindings.len(),
        }
    }
}

enum SpinorParameterLayout {
    Legacy(Box<[usize]>),
    Prepared {
        defaults: Box<[crate::EagerComplex64]>,
        projection: Box<[PreparedParameterProjectionEntry]>,
        dag_prepared_slots: Box<[usize]>,
    },
}

impl SpinorNativeRuntime {
    pub(super) fn new(
        dag: SpinorDag,
        ordered_source_labels: Vec<usize>,
        parameter_indices: Vec<usize>,
    ) -> RusticolResult<Self> {
        if parameter_indices.len() != usize::from(dag.parameter_count()) {
            return Err(RusticolError::integrity(
                "spinor DAG parameter bindings do not match its parameter slots",
            ));
        }
        let workspace = dag.batch_workspace(SPINOR_POINT_TILE_SIZE)?;
        Ok(Self {
            dag,
            workspace,
            sources: SpinorSourceLayout::Legacy(ordered_source_labels.into_boxed_slice()),
            parameters: SpinorParameterLayout::Legacy(parameter_indices.into_boxed_slice()),
            parameter_values: Vec::new(),
            prepared_parameter_re: Vec::new(),
            prepared_parameter_im: Vec::new(),
            flat_momenta: Vec::new(),
        })
    }

    pub(super) fn new_payload(
        payload: SpinorDagPayloadV3,
        prepared_parameter_defaults: Vec<crate::EagerComplex64>,
        parameter_projection: Vec<PreparedParameterProjectionEntry>,
    ) -> RusticolResult<Self> {
        let (dag, source_inputs, prepared_parameter_count, parameter_bindings) =
            payload.into_parts();
        let prepared_parameter_count = usize::try_from(prepared_parameter_count).map_err(|_| {
            RusticolError::artifact("spinor prepared-parameter count exceeds usize")
        })?;
        if prepared_parameter_count != prepared_parameter_defaults.len() {
            return Err(RusticolError::integrity(
                "spinor payload prepared-parameter count does not match the authoritative runtime domain",
            ));
        }
        if parameter_projection
            .iter()
            .any(|entry| entry.prepared_slot >= prepared_parameter_defaults.len())
        {
            return Err(RusticolError::integrity(
                "spinor runtime parameter projection exceeds the prepared domain",
            ));
        }
        let dag_prepared_slots = parameter_bindings
            .iter()
            .map(|binding| {
                usize::try_from(binding.prepared_parameter_slot()).map_err(|_| {
                    RusticolError::artifact("spinor prepared parameter slot exceeds usize")
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        if dag_prepared_slots
            .iter()
            .any(|slot| *slot >= prepared_parameter_defaults.len())
        {
            return Err(RusticolError::integrity(
                "spinor DAG parameter binding exceeds the authoritative prepared domain",
            ));
        }
        let kinds = source_inputs
            .iter()
            .map(|binding| binding.kind())
            .collect::<Vec<_>>();
        let workspace = dag.batch_workspace_with_source_kinds(SPINOR_POINT_TILE_SIZE, &kinds)?;
        Ok(Self {
            dag,
            workspace,
            sources: SpinorSourceLayout::Payload {
                bindings: source_inputs,
                kinds: kinds.into_boxed_slice(),
            },
            parameters: SpinorParameterLayout::Prepared {
                defaults: prepared_parameter_defaults.into_boxed_slice(),
                projection: parameter_projection.into_boxed_slice(),
                dag_prepared_slots: dag_prepared_slots.into_boxed_slice(),
            },
            parameter_values: Vec::new(),
            prepared_parameter_re: vec![0.0; prepared_parameter_count],
            prepared_parameter_im: vec![0.0; prepared_parameter_count],
            flat_momenta: Vec::new(),
        })
    }

    fn validate_selection(
        &self,
        common: &ExecutionRuntime,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<()> {
        if selected_helicities.is_some() {
            return Err(RusticolError::selector(
                "spinor DAG execution is always helicity-summed",
            ));
        }
        if let Some(colors) = selected_colors {
            let physics = common.physics.as_ref().ok_or_else(|| {
                RusticolError::artifact("spinor execution has no runtime physics metadata")
            })?;
            let expected = physics
                .manifest
                .color_components
                .first()
                .map(|color| color.id())
                .ok_or_else(|| RusticolError::artifact("spinor execution has no fixed color"))?;
            if colors.len() != 1 || !colors.contains(expected) {
                return Err(RusticolError::selector(format!(
                    "spinor execution contains only fixed color component {expected:?}"
                )));
            }
        }
        Ok(())
    }

    pub(super) fn run_total_into_unprofiled(
        &mut self,
        common: &ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        self.validate_selection(common, selected_helicities, selected_colors)?;
        self.evaluate_batch(
            &common.external_is_initial,
            &common.model_parameter_values_f64,
            common.normalization_factor,
            batch,
            output,
        )
    }

    fn evaluate_batch(
        &mut self,
        external_is_initial: &[bool],
        model_parameter_values: &[f64],
        normalization_factor: f64,
        batch: F64MomentumBatchView<'_>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        if batch.external_count() != self.sources.len()
            || external_is_initial.len() != self.sources.len()
            || output.len() != batch.point_count()
        {
            return Err(RusticolError::invalid_argument(
                "spinor momentum or output shape does not match the execution manifest",
            ));
        }
        self.parameter_values.clear();
        match &self.parameters {
            SpinorParameterLayout::Legacy(parameter_indices) => {
                for parameter_index in parameter_indices.iter().copied() {
                    let value = model_parameter_values.get(parameter_index).ok_or_else(|| {
                        RusticolError::integrity(
                            "spinor DAG parameter binding references an absent runtime value",
                        )
                    })?;
                    self.parameter_values
                        .push(crate::EagerComplex64::new(*value, 0.0));
                }
            }
            SpinorParameterLayout::Prepared {
                defaults,
                projection,
                dag_prepared_slots,
            } => {
                initialize_prepared_parameter_planes(
                    defaults,
                    projection,
                    model_parameter_values,
                    &mut self.prepared_parameter_re,
                    &mut self.prepared_parameter_im,
                )?;
                for slot in dag_prepared_slots.iter().copied() {
                    self.parameter_values.push(crate::EagerComplex64::new(
                        self.prepared_parameter_re[slot],
                        self.prepared_parameter_im[slot],
                    ));
                }
            }
        }
        let point_width = self.sources.len() * 4;
        for tile_start in (0..batch.point_count()).step_by(SPINOR_POINT_TILE_SIZE) {
            let tile_stop = (tile_start + SPINOR_POINT_TILE_SIZE).min(batch.point_count());
            let tile = batch.subview(tile_start, tile_stop)?;
            let tile_count = tile.point_count();
            self.flat_momenta.resize(tile_count * point_width, 0.0);
            for point_index in 0..tile_count {
                let point = tile.point(point_index);
                for ordered_index in 0..self.sources.len() {
                    let (external_index, sign) = match &self.sources {
                        SpinorSourceLayout::Legacy(labels) => {
                            let external_index = labels[ordered_index] - 1;
                            let sign = if external_is_initial[external_index] {
                                -1.0
                            } else {
                                1.0
                            };
                            (external_index, sign)
                        }
                        SpinorSourceLayout::Payload { bindings, .. } => {
                            let binding = bindings[ordered_index];
                            (
                                usize::from(binding.public_source_slot()),
                                f64::from(binding.momentum_sign()),
                            )
                        }
                    };
                    let momentum = point.momentum(external_index).ok_or_else(|| {
                        RusticolError::integrity("spinor source traversal references an absent leg")
                    })?;
                    let start = point_index * point_width + ordered_index * 4;
                    for component in 0..4 {
                        self.flat_momenta[start + component] = sign * momentum[component];
                    }
                }
            }
            match &self.sources {
                SpinorSourceLayout::Legacy(_) => {
                    self.dag.evaluate_sum_batch_into_with_parameters(
                        &self.flat_momenta,
                        tile_count,
                        &self.parameter_values,
                        &mut self.workspace,
                        &mut output[tile_start..tile_stop],
                    )?;
                }
                SpinorSourceLayout::Payload { kinds, .. } => {
                    self.dag
                        .evaluate_sum_batch_into_with_source_kinds_and_parameters(
                            &self.flat_momenta,
                            tile_count,
                            kinds,
                            &self.parameter_values,
                            &mut self.workspace,
                            &mut output[tile_start..tile_stop],
                        )?;
                }
            }
            for value in &mut output[tile_start..tile_stop] {
                *value *= normalization_factor;
            }
        }
        Ok(())
    }

    pub(super) fn run_total_into(
        &mut self,
        common: &ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<RuntimeProfile> {
        let started = Instant::now();
        self.run_total_into_unprofiled(
            common,
            batch,
            selected_helicities,
            selected_colors,
            output,
        )?;
        let total_s = started.elapsed().as_secs_f64();
        Ok(RuntimeProfile {
            orchestration_s: total_s,
            total_s,
            total_materialized_value_count: output.len() as u64,
            ..RuntimeProfile::default()
        })
    }

    pub(super) fn run_resolved(
        &mut self,
        common: &ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<ResolvedValues<f64>> {
        self.validate_selection(common, selected_helicities, selected_colors)?;
        let mut values = vec![0.0; batch.point_count()];
        self.run_total_into_unprofiled(common, batch, None, selected_colors, &mut values)?;
        Ok(ResolvedValues {
            values,
            point_count: batch.point_count(),
            helicity_indices: vec![0],
            color_indices: vec![0],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spinor::{
        SpinorDagBuilder, SpinorPreparedParameterBinding,
        build_helicity_summed_quark_gluon_bg_spinor_dag,
        build_helicity_summed_quark_z_gluon_spinor_dag,
        build_optimized_helicity_summed_gluon_spinor_dag,
    };

    fn scalar_contact_payload(prepared_parameter_count: u32) -> SpinorDagPayloadV3 {
        let mut builder = SpinorDagBuilder::new_with_parameters(4, 1).unwrap();
        let coupling = builder.parameter(0).unwrap();
        builder.add_root(vec![0_i8; 4], coupling).unwrap();
        SpinorDagPayloadV3::new(
            builder.finish().unwrap(),
            vec![
                SpinorSourceInputBinding::new(0, -1, SpinorSourceInputKind::MomentumOnly).unwrap(),
                SpinorSourceInputBinding::new(1, -1, SpinorSourceInputKind::MomentumOnly).unwrap(),
                SpinorSourceInputBinding::new(2, 1, SpinorSourceInputKind::MomentumOnly).unwrap(),
                SpinorSourceInputBinding::new(3, 1, SpinorSourceInputKind::MomentumOnly).unwrap(),
            ],
            prepared_parameter_count,
            vec![SpinorPreparedParameterBinding::new(2)],
        )
        .unwrap()
    }

    #[test]
    fn scalar_payload_projects_complex_coupling_and_refreshes_lambda() {
        let payload = scalar_contact_payload(3);
        let momenta = [
            [500.0, 0.0, 0.0, 500.0],
            [500.0, 0.0, 0.0, -500.0],
            [499.9985, 0.0, 0.0, 499.9975],
            [500.0015, 0.0, 0.0, -499.9975],
        ];
        let kinds = [SpinorSourceInputKind::MomentumOnly; 4];
        let mut workspace = payload.dag().workspace_with_source_kinds(&kinds).unwrap();
        let raw_sum = payload
            .dag()
            .evaluate_into_workspace_with_source_kinds_and_parameters(
                &momenta,
                &kinds,
                &[crate::EagerComplex64::new(0.0, 1.0)],
                &mut workspace,
            )
            .unwrap();
        assert_eq!(
            workspace.amplitudes(),
            [crate::EagerComplex64::new(0.0, 1.0)]
        );
        assert_eq!(raw_sum, 1.0);

        let defaults = vec![crate::EagerComplex64::new(0.0, 0.0); 3];
        let projection = vec![PreparedParameterProjectionEntry {
            runtime_slot: 0,
            prepared_slot: 2,
            component: 1,
        }];
        let mut lane = SpinorNativeRuntime::new_payload(payload, defaults, projection).unwrap();
        let flat = momenta
            .iter()
            .flat_map(|momentum| momentum.iter().copied())
            .collect::<Vec<_>>();
        let batch = F64MomentumBatchView::from_contiguous_prevalidated(&flat, 1, 4, None).unwrap();
        let mut output = [0.0];

        lane.evaluate_batch(&[true, true, false, false], &[1.0], 0.5, batch, &mut output)
            .unwrap();
        assert_eq!(output, [0.5]);
        lane.evaluate_batch(&[true, true, false, false], &[3.0], 1.0, batch, &mut output)
            .unwrap();
        assert_eq!(output, [9.0]);
    }

    #[test]
    fn scalar_payload_requires_authoritative_prepared_parameter_count() {
        let result = SpinorNativeRuntime::new_payload(
            scalar_contact_payload(3),
            vec![crate::EagerComplex64::new(0.0, 0.0); 2],
            vec![],
        );
        let error = match result {
            Ok(_) => panic!("payload count mismatch must fail closed"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("authoritative runtime domain"));
    }

    #[test]
    fn lane_crosses_initial_momenta_and_applies_normalization() {
        let physical = [
            [5.0, 0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0, -5.0],
            [5.0, 3.0, 0.0, 4.0],
            [5.0, -3.0, 0.0, -4.0],
        ];
        let all_outgoing = [
            [-5.0, 0.0, 0.0, -5.0],
            [-5.0, 0.0, 0.0, 5.0],
            physical[2],
            physical[3],
        ];
        let expected = build_optimized_helicity_summed_gluon_spinor_dag(4)
            .unwrap()
            .evaluate(&all_outgoing)
            .unwrap()
            .helicity_sum();
        let flat = physical
            .iter()
            .flat_map(|momentum| momentum.iter().copied())
            .collect::<Vec<_>>();
        let batch = F64MomentumBatchView::from_contiguous_prevalidated(&flat, 1, 4, None).unwrap();
        let mut lane = SpinorNativeRuntime::new(
            build_optimized_helicity_summed_gluon_spinor_dag(4).unwrap(),
            vec![1, 2, 3, 4],
            Vec::new(),
        )
        .unwrap();
        let mut output = [0.0];
        lane.evaluate_batch(&[true, true, false, false], &[], 2.5, batch, &mut output)
            .unwrap();
        assert!((output[0] - 2.5 * expected).abs() <= 1.0e-11 * expected.abs().max(1.0));
    }

    #[test]
    fn quark_lane_reorders_sources_before_crossing_initial_momenta() {
        let physical = [
            [5.0, 0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0, -5.0],
            [5.0, 3.0, 0.0, 4.0],
            [5.0, -3.0, 0.0, -4.0],
        ];
        // Public u, ubar, g, g labels become the all-outgoing open string
        // q(label 2), g(label 3), g(label 4), qbar(label 1).
        let all_outgoing_traversal = [
            [-5.0, 0.0, 0.0, 5.0],
            physical[2],
            physical[3],
            [-5.0, 0.0, 0.0, -5.0],
        ];
        let dag = build_helicity_summed_quark_gluon_bg_spinor_dag(&[0, 1, 2, 3]).unwrap();
        let expected = dag
            .evaluate(&all_outgoing_traversal)
            .unwrap()
            .helicity_sum();
        let flat = physical
            .iter()
            .flat_map(|momentum| momentum.iter().copied())
            .collect::<Vec<_>>();
        let batch = F64MomentumBatchView::from_contiguous_prevalidated(&flat, 1, 4, None).unwrap();
        let mut lane = SpinorNativeRuntime::new(dag, vec![2, 3, 4, 1], Vec::new()).unwrap();
        let mut output = [0.0];
        lane.evaluate_batch(&[true, true, false, false], &[], 3.0, batch, &mut output)
            .unwrap();
        assert!(expected > 0.0);
        assert!((output[0] - 3.0 * expected).abs() <= 1.0e-11 * expected.abs().max(1.0));
    }

    #[test]
    fn q_z_lane_reorders_sources_and_projects_common_chiral_couplings() {
        let physical = [
            [5.0, 0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0, -5.0],
            [10.0, 0.0, 0.0, 0.0],
        ];
        let all_outgoing_graph_order = [[-5.0, 0.0, 0.0, 5.0], [-5.0, 0.0, 0.0, -5.0], physical[2]];
        let sqrt_two = 2.0_f64.sqrt();
        let dag = build_helicity_summed_quark_z_gluon_spinor_dag(&[0, 1], 2).unwrap();
        let expected = dag
            .evaluate_with_parameters(
                &all_outgoing_graph_order,
                &[
                    crate::EagerComplex64::new(sqrt_two * 2.0, 0.0),
                    crate::EagerComplex64::new(sqrt_two * 3.0, 0.0),
                ],
            )
            .unwrap()
            .helicity_sum();
        let flat = physical
            .iter()
            .flat_map(|momentum| momentum.iter().copied())
            .collect::<Vec<_>>();
        let batch = F64MomentumBatchView::from_contiguous_prevalidated(&flat, 1, 3, None).unwrap();
        let mut lane = SpinorNativeRuntime::new(dag, vec![2, 1, 3], vec![2, 1]).unwrap();
        let common_values = [99.0, sqrt_two * 3.0, sqrt_two * 2.0];
        let mut output = [0.0];
        lane.evaluate_batch(
            &[true, true, false],
            &common_values,
            2.5,
            batch,
            &mut output,
        )
        .unwrap();
        assert!((expected - 2600.0).abs() <= 1.0e-11 * 2600.0);
        assert!((output[0] - 2.5 * expected).abs() <= 1.0e-11 * expected);
    }
}
