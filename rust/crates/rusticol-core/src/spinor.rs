// SPDX-License-Identifier: 0BSD

//! Experimental typed spinor-helicity DAG and two-component algebra.
//!
//! This is deliberately a sibling of the component recurrence program.  It
//! preserves dotted/undotted spinor structure instead of attempting to infer
//! it after model kernels have already been expanded into Cartesian planes.

mod codec;

pub use codec::{
    SPINOR_DAG_BINARY_ABI, SpinorDagPayloadV2, SpinorPreparedParameterBinding,
    SpinorSourceInputBinding, SpinorSourceInputKind, decode_spinor_dag_v2, encode_spinor_dag_v2,
};

use crate::recurrence::{ExactComplexRational, ExactRational};
use crate::{RusticolError, RusticolResult};
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
use num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet};

pub const SPINOR_DAG_ABI: &str = "pyamplicol-spinor-dag-v1";

pub type SpinorNodeId = u32;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("spinor DAG: {}", message.into()))
}

/// The two inequivalent antisymmetric contractions in four dimensions.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SpinorBracketKind {
    /// Undotted contraction, conventionally written `<ij>`.
    Angle,
    /// Dotted contraction, conventionally written `[ij]`.
    Square,
}

/// Chirality of a gamma-current contraction before a Fierz rewrite.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SpinorChirality {
    Positive,
    Negative,
}

/// Runtime scalar leaves which depend on the kinematic point but not on a
/// spinor phase convention.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SpinorKinematicScalar {
    SqrtTwo,
    InverseMass { source: u16 },
}

/// One globally interned scalar expression.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SpinorNode {
    Constant(ExactComplexRational),
    Parameter(u16),
    Kinematic(SpinorKinematicScalar),
    Bracket {
        kind: SpinorBracketKind,
        left: u16,
        right: u16,
    },
    Sum(Box<[SpinorNodeId]>),
    Product(Box<[SpinorNodeId]>),
    Reciprocal(SpinorNodeId),
}

/// Accounting for exact identities applied while constructing the DAG.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SpinorRewriteStats {
    pub antisymmetry_rewrites: u64,
    pub self_product_zeros: u64,
    pub constant_folds: u64,
    pub sum_factorizations: u64,
    pub reciprocal_cancellations: u64,
    pub schouten_rewrites: u64,
    pub fierz_rewrites: u64,
    pub structural_zero_roots: u64,
    pub dead_nodes_pruned: u64,
}

/// Live scalar-DAG shape after exact rewriting and dead-code elimination.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SpinorDagCensus {
    pub constants: usize,
    pub parameters: usize,
    pub kinematic_scalars: usize,
    pub brackets: usize,
    pub sums: usize,
    pub sum_operands: usize,
    pub products: usize,
    pub product_operands: usize,
    pub reciprocals: usize,
}

impl SpinorDagCensus {
    pub const fn estimated_complex_arithmetic(self) -> usize {
        3 * self.brackets
            + self.kinematic_scalars
            + self.sum_operands
            + self.product_operands
            + self.reciprocals
    }
}

/// The two null spinor atoms used for one massive momentum `p = k + r`.
/// They are derived deterministically at evaluation time and therefore do not
/// enlarge the physical momentum input. The same representation serves a
/// massive vector or either end of a massive Dirac line.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MassiveSpinorSource {
    source: u16,
    k_atom: u16,
    r_atom: u16,
}

/// One physical-helicity amplitude. Roots remain distinct until after
/// squaring, so the sum cannot introduce cross-helicity interference.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpinorAmplitudeRoot {
    helicities: Box<[i8]>,
    amplitude: SpinorNodeId,
    structural_zero: bool,
    multiplicity: u16,
}

impl SpinorAmplitudeRoot {
    pub fn helicities(&self) -> &[i8] {
        &self.helicities
    }

    pub const fn amplitude(&self) -> SpinorNodeId {
        self.amplitude
    }

    pub const fn is_structural_zero(&self) -> bool {
        self.structural_zero
    }

    pub const fn multiplicity(&self) -> u16 {
        self.multiplicity
    }
}

/// Immutable, helicity-summed-ready scalar spinor graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpinorDag {
    momentum_count: u16,
    spinor_atom_count: u16,
    parameter_count: u16,
    massive_sources: Box<[MassiveSpinorSource]>,
    uses_reference_atom: bool,
    nodes: Box<[SpinorNode]>,
    roots: Box<[SpinorAmplitudeRoot]>,
    rewrite_stats: SpinorRewriteStats,
}

impl SpinorDag {
    pub const fn momentum_count(&self) -> u16 {
        self.momentum_count
    }

    pub const fn parameter_count(&self) -> u16 {
        self.parameter_count
    }

    pub const fn uses_reference_atom(&self) -> bool {
        self.uses_reference_atom
    }

    pub fn nodes(&self) -> &[SpinorNode] {
        &self.nodes
    }

    pub fn roots(&self) -> &[SpinorAmplitudeRoot] {
        &self.roots
    }

    pub const fn rewrite_stats(&self) -> SpinorRewriteStats {
        self.rewrite_stats
    }

    pub fn census(&self) -> SpinorDagCensus {
        let mut census = SpinorDagCensus::default();
        for node in &self.nodes {
            match node {
                SpinorNode::Constant(_) => census.constants += 1,
                SpinorNode::Parameter(_) => census.parameters += 1,
                SpinorNode::Kinematic(_) => census.kinematic_scalars += 1,
                SpinorNode::Bracket { .. } => census.brackets += 1,
                SpinorNode::Sum(operands) => {
                    census.sums += 1;
                    census.sum_operands += operands.len();
                }
                SpinorNode::Product(operands) => {
                    census.products += 1;
                    census.product_operands += operands.len();
                }
                SpinorNode::Reciprocal(_) => census.reciprocals += 1,
            }
        }
        census
    }

    /// Evaluate every helicity root on one all-outgoing massless point and
    /// perform the incoherent helicity sum only after all amplitudes exist.
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub fn evaluate(&self, momenta: &[[f64; 4]]) -> RusticolResult<SpinorDagEvaluation> {
        self.evaluate_with_parameters(momenta, &[])
    }

    /// Evaluate a graph with its runtime scalar parameters.  Parameter order
    /// is defined by the builder which produced the graph; the q-Z builder
    /// below uses `[g_left, g_right]`.
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub fn evaluate_with_parameters(
        &self,
        momenta: &[[f64; 4]],
        parameters: &[Complex64],
    ) -> RusticolResult<SpinorDagEvaluation> {
        let mut workspace = self.workspace();
        let helicity_sum =
            self.evaluate_into_workspace_with_parameters(momenta, parameters, &mut workspace)?;
        Ok(SpinorDagEvaluation {
            amplitudes: workspace.amplitudes.clone(),
            helicity_sum,
        })
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub fn workspace(&self) -> SpinorWorkspace {
        SpinorWorkspace::new(self)
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub fn workspace_with_source_kinds(
        &self,
        source_kinds: &[SpinorSourceInputKind],
    ) -> RusticolResult<SpinorWorkspace> {
        let spinor_slot_count = self.required_spinor_slot_count(source_kinds)?;
        Ok(SpinorWorkspace::new_with_spinor_slots(
            self,
            spinor_slot_count,
        ))
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub fn batch_workspace(&self, point_capacity: usize) -> RusticolResult<SpinorBatchWorkspace> {
        SpinorBatchWorkspace::new(self, point_capacity)
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub fn batch_workspace_with_source_kinds(
        &self,
        point_capacity: usize,
        source_kinds: &[SpinorSourceInputKind],
    ) -> RusticolResult<SpinorBatchWorkspace> {
        let spinor_slot_count = self.required_spinor_slot_count(source_kinds)?;
        SpinorBatchWorkspace::new_with_spinor_slots(self, point_capacity, spinor_slot_count)
    }

    /// Node-major batch evaluation. Node dispatch occurs once per tile and
    /// each scalar operation runs over a contiguous point plane.
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub fn evaluate_sum_batch_into(
        &self,
        momenta: &[f64],
        point_count: usize,
        workspace: &mut SpinorBatchWorkspace,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        self.evaluate_sum_batch_into_with_parameters(momenta, point_count, &[], workspace, output)
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub fn evaluate_sum_batch_into_with_parameters(
        &self,
        momenta: &[f64],
        point_count: usize,
        parameters: &[Complex64],
        workspace: &mut SpinorBatchWorkspace,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let source_kinds = self.legacy_source_kinds();
        self.evaluate_sum_batch_into_with_source_kinds_and_parameters(
            momenta,
            point_count,
            &source_kinds,
            parameters,
            workspace,
            output,
        )
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub fn evaluate_sum_batch_into_with_source_kinds_and_parameters(
        &self,
        momenta: &[f64],
        point_count: usize,
        source_kinds: &[SpinorSourceInputKind],
        parameters: &[Complex64],
        workspace: &mut SpinorBatchWorkspace,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let required_spinor_slots = self.required_spinor_slot_count(source_kinds)?;
        workspace.validate_for_with_spinor_slots(self, point_count, required_spinor_slots)?;
        self.validate_parameters(parameters)?;
        if output.len() != point_count {
            return Err(invalid(format!(
                "batch output contains {} values, expected {point_count}",
                output.len()
            )));
        }
        let point_width = usize::from(self.momentum_count)
            .checked_mul(4)
            .ok_or_else(|| invalid("point momentum width overflows"))?;
        if momenta.len() != point_count.saturating_mul(point_width) {
            return Err(invalid(format!(
                "batch momentum storage contains {} scalars, expected {}",
                momenta.len(),
                point_count.saturating_mul(point_width)
            )));
        }
        let stride = workspace.point_capacity;
        for (point_index, point) in momenta.chunks_exact(point_width).enumerate() {
            for (leg, components) in point.chunks_exact(4).enumerate() {
                if source_kinds[leg] != SpinorSourceInputKind::NullSpinor
                    || leg >= required_spinor_slots
                {
                    continue;
                }
                let momentum = [components[0], components[1], components[2], components[3]];
                workspace.spinors[leg * stride + point_index] =
                    MasslessSpinors::from_momentum(momentum).map_err(|error| {
                        invalid(format!(
                            "point {point_index} momentum {leg} cannot be factorized: {error}"
                        ))
                    })?;
            }
            for massive in self.massive_sources.iter().copied() {
                if usize::from(massive.k_atom) >= required_spinor_slots
                    && usize::from(massive.r_atom) >= required_spinor_slots
                {
                    continue;
                }
                let offset = usize::from(massive.source) * 4;
                let momentum = [
                    point[offset],
                    point[offset + 1],
                    point[offset + 2],
                    point[offset + 3],
                ];
                let (k, r, _) = decompose_massive_momentum(momentum)?;
                workspace.spinors[usize::from(massive.k_atom) * stride + point_index] = k;
                workspace.spinors[usize::from(massive.r_atom) * stride + point_index] = r;
            }
            if self.uses_reference_atom
                && usize::from(self.spinor_atom_count) < required_spinor_slots
            {
                workspace.spinors[usize::from(self.spinor_atom_count) * stride + point_index] =
                    MasslessSpinors::from_momentum(select_common_reference_flat(point)?)?;
            }
        }

        for (node_index, node) in self.nodes.iter().enumerate() {
            let destination_start = workspace.node_slots[node_index] * stride;
            match node {
                SpinorNode::Constant(exact) => {
                    let destination =
                        &mut workspace.values[destination_start..destination_start + point_count];
                    destination.fill(exact_complex_to_f64(*exact));
                }
                SpinorNode::Parameter(index) => {
                    let parameter = parameters
                        .get(usize::from(*index))
                        .copied()
                        .ok_or_else(|| invalid("parameter node is outside the runtime input"))?;
                    workspace.values[destination_start..destination_start + point_count]
                        .fill(parameter);
                }
                SpinorNode::Kinematic(scalar) => {
                    let destination =
                        &mut workspace.values[destination_start..destination_start + point_count];
                    for (point_index, destination) in destination.iter_mut().enumerate() {
                        let point =
                            &momenta[point_index * point_width..(point_index + 1) * point_width];
                        *destination = evaluate_kinematic_flat(*scalar, point)?;
                    }
                }
                SpinorNode::Bracket { kind, left, right } => {
                    let left_start = usize::from(*left) * stride;
                    let right_start = usize::from(*right) * stride;
                    let left_plane = &workspace.spinors[left_start..left_start + point_count];
                    let right_plane = &workspace.spinors[right_start..right_start + point_count];
                    let destination =
                        &mut workspace.values[destination_start..destination_start + point_count];
                    for ((destination, left), right) in
                        destination.iter_mut().zip(left_plane).zip(right_plane)
                    {
                        *destination = match kind {
                            SpinorBracketKind::Angle => angle(left.undotted, right.undotted),
                            SpinorBracketKind::Square => square(left.dotted, right.dotted),
                        };
                    }
                }
                SpinorNode::Sum(operands) => {
                    workspace.values[destination_start..destination_start + point_count]
                        .fill(Complex64::new(0.0, 0.0));
                    for operand in operands.iter().copied() {
                        let source_start = workspace.node_slot(operand)? * stride;
                        let (destination, source) = disjoint_value_planes(
                            &mut workspace.values,
                            destination_start,
                            source_start,
                            point_count,
                        )?;
                        for (destination, source) in destination.iter_mut().zip(source) {
                            *destination += *source;
                        }
                    }
                }
                SpinorNode::Product(operands) => {
                    workspace.values[destination_start..destination_start + point_count]
                        .fill(Complex64::new(1.0, 0.0));
                    for operand in operands.iter().copied() {
                        let source_start = workspace.node_slot(operand)? * stride;
                        let (destination, source) = disjoint_value_planes(
                            &mut workspace.values,
                            destination_start,
                            source_start,
                            point_count,
                        )?;
                        for (destination, source) in destination.iter_mut().zip(source) {
                            *destination *= *source;
                        }
                    }
                }
                SpinorNode::Reciprocal(operand) => {
                    let source_start = workspace.node_slot(*operand)? * stride;
                    let (destination, source) = disjoint_value_planes(
                        &mut workspace.values,
                        destination_start,
                        source_start,
                        point_count,
                    )?;
                    for (destination, source) in destination.iter_mut().zip(source) {
                        if *source == Complex64::new(0.0, 0.0) {
                            return Err(invalid("encountered a singular spinor denominator"));
                        }
                        *destination = Complex64::new(1.0, 0.0) / *source;
                    }
                }
            }
        }

        output.fill(0.0);
        workspace.compensation[..point_count].fill(0.0);
        for root in self.roots.iter().filter(|root| !root.structural_zero) {
            let source_start = workspace.node_slot(root.amplitude)? * stride;
            let source = &workspace.values[source_start..source_start + point_count];
            for ((sum, compensation), amplitude) in output
                .iter_mut()
                .zip(&mut workspace.compensation[..point_count])
                .zip(source)
            {
                let term = amplitude.norm_sqr() * f64::from(root.multiplicity);
                let next = *sum + term;
                *compensation += if sum.abs() >= term.abs() {
                    (*sum - next) + term
                } else {
                    (term - next) + *sum
                };
                *sum = next;
            }
        }
        for (sum, compensation) in output
            .iter_mut()
            .zip(&workspace.compensation[..point_count])
        {
            *sum += *compensation;
            if !sum.is_finite() {
                return Err(invalid("spinor batch helicity sum is non-finite"));
            }
        }
        Ok(())
    }

    /// Evaluate one point without allocating after workspace construction.
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub fn evaluate_into_workspace(
        &self,
        momenta: &[[f64; 4]],
        workspace: &mut SpinorWorkspace,
    ) -> RusticolResult<f64> {
        self.evaluate_into_workspace_with_parameters(momenta, &[], workspace)
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub fn evaluate_into_workspace_with_parameters(
        &self,
        momenta: &[[f64; 4]],
        parameters: &[Complex64],
        workspace: &mut SpinorWorkspace,
    ) -> RusticolResult<f64> {
        let source_kinds = self.legacy_source_kinds();
        self.evaluate_into_workspace_with_source_kinds_and_parameters(
            momenta,
            &source_kinds,
            parameters,
            workspace,
        )
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub fn evaluate_into_workspace_with_source_kinds_and_parameters(
        &self,
        momenta: &[[f64; 4]],
        source_kinds: &[SpinorSourceInputKind],
        parameters: &[Complex64],
        workspace: &mut SpinorWorkspace,
    ) -> RusticolResult<f64> {
        if momenta.len() != usize::from(self.momentum_count) {
            return Err(invalid(format!(
                "received {} momenta, expected {}",
                momenta.len(),
                self.momentum_count
            )));
        }
        let required_spinor_slots = self.required_spinor_slot_count(source_kinds)?;
        workspace.validate_for_with_spinor_slots(self, required_spinor_slots)?;
        self.validate_parameters(parameters)?;
        for (index, momentum) in momenta.iter().copied().enumerate() {
            if source_kinds[index] != SpinorSourceInputKind::NullSpinor
                || index >= required_spinor_slots
            {
                continue;
            }
            workspace.spinors[index] =
                MasslessSpinors::from_momentum(momentum).map_err(|error| {
                    invalid(format!("momentum {index} cannot be factorized: {error}"))
                })?;
        }
        for massive in self.massive_sources.iter().copied() {
            if usize::from(massive.k_atom) >= required_spinor_slots
                && usize::from(massive.r_atom) >= required_spinor_slots
            {
                continue;
            }
            let (k, r, _) = decompose_massive_momentum(momenta[usize::from(massive.source)])?;
            workspace.spinors[usize::from(massive.k_atom)] = k;
            workspace.spinors[usize::from(massive.r_atom)] = r;
        }
        if self.uses_reference_atom && usize::from(self.spinor_atom_count) < required_spinor_slots {
            workspace.spinors[usize::from(self.spinor_atom_count)] =
                MasslessSpinors::from_momentum(select_common_reference_momentum(momenta)?)?;
        }
        for (node_index, node) in self.nodes.iter().enumerate() {
            let node_value = match node {
                SpinorNode::Constant(value) => exact_complex_to_f64(*value),
                SpinorNode::Parameter(index) => parameters
                    .get(usize::from(*index))
                    .copied()
                    .ok_or_else(|| invalid("parameter node is outside the runtime input"))?,
                SpinorNode::Kinematic(scalar) => evaluate_kinematic(*scalar, momenta)?,
                SpinorNode::Bracket { kind, left, right } => {
                    let left = workspace.spinors.get(usize::from(*left)).ok_or_else(|| {
                        invalid("bracket left momentum is outside the input domain")
                    })?;
                    let right = workspace.spinors.get(usize::from(*right)).ok_or_else(|| {
                        invalid("bracket right momentum is outside the input domain")
                    })?;
                    match kind {
                        SpinorBracketKind::Angle => angle(left.undotted, right.undotted),
                        SpinorBracketKind::Square => square(left.dotted, right.dotted),
                    }
                }
                SpinorNode::Sum(operands) => {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for operand in operands.iter().copied() {
                        sum += value(&workspace.values, operand)?;
                    }
                    sum
                }
                SpinorNode::Product(operands) => {
                    let mut product = Complex64::new(1.0, 0.0);
                    for operand in operands.iter().copied() {
                        product *= value(&workspace.values, operand)?;
                    }
                    product
                }
                SpinorNode::Reciprocal(operand) => {
                    let denominator = value(&workspace.values, *operand)?;
                    if denominator == Complex64::new(0.0, 0.0) {
                        return Err(invalid("encountered a singular spinor denominator"));
                    }
                    Complex64::new(1.0, 0.0) / denominator
                }
            };
            if !node_value.re.is_finite() || !node_value.im.is_finite() {
                return Err(invalid("spinor DAG produced a non-finite value"));
            }
            workspace.values[node_index] = node_value;
        }
        let mut helicity_sum = 0.0;
        let mut compensation = 0.0;
        for (root_index, root) in self.roots.iter().enumerate() {
            let amplitude = if root.structural_zero {
                Complex64::new(0.0, 0.0)
            } else {
                value(&workspace.values, root.amplitude)?
            };
            workspace.amplitudes[root_index] = amplitude;
            let term = amplitude.norm_sqr() * f64::from(root.multiplicity);
            let next = helicity_sum + term;
            compensation += if helicity_sum.abs() >= term.abs() {
                (helicity_sum - next) + term
            } else {
                (term - next) + helicity_sum
            };
            helicity_sum = next;
        }
        helicity_sum += compensation;
        if !helicity_sum.is_finite() {
            return Err(invalid("spinor helicity sum is non-finite"));
        }
        Ok(helicity_sum)
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    fn validate_parameters(&self, parameters: &[Complex64]) -> RusticolResult<()> {
        if parameters.len() != usize::from(self.parameter_count) {
            return Err(invalid(format!(
                "received {} parameters, expected {}",
                parameters.len(),
                self.parameter_count
            )));
        }
        if parameters
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(invalid("runtime parameters must be finite"));
        }
        Ok(())
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    fn legacy_source_kinds(&self) -> Vec<SpinorSourceInputKind> {
        (0..self.momentum_count)
            .map(|source| {
                if self
                    .massive_sources
                    .iter()
                    .any(|entry| entry.source == source)
                {
                    SpinorSourceInputKind::MassiveSpinorPair
                } else {
                    SpinorSourceInputKind::NullSpinor
                }
            })
            .collect()
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    fn required_spinor_slot_count(
        &self,
        source_kinds: &[SpinorSourceInputKind],
    ) -> RusticolResult<usize> {
        if source_kinds.len() != usize::from(self.momentum_count) {
            return Err(invalid(format!(
                "received {} source kinds, expected {}",
                source_kinds.len(),
                self.momentum_count
            )));
        }
        for (source, kind) in source_kinds.iter().copied().enumerate() {
            let is_massive = self
                .massive_sources
                .iter()
                .any(|entry| usize::from(entry.source) == source);
            if is_massive != (kind == SpinorSourceInputKind::MassiveSpinorPair) {
                return Err(invalid(format!(
                    "source {source} kind does not match the graph massive-source layout"
                )));
            }
        }
        for root in &self.roots {
            for (source, kind) in source_kinds.iter().copied().enumerate() {
                if kind == SpinorSourceInputKind::MomentumOnly
                    && root.helicities.get(source).copied() != Some(0)
                {
                    return Err(invalid(format!(
                        "momentum-only source {source} has a nonzero root helicity"
                    )));
                }
            }
        }
        let mut required = 0usize;
        for node in &self.nodes {
            if let SpinorNode::Bracket { left, right, .. } = node {
                for atom in [*left, *right] {
                    let atom = usize::from(atom);
                    if atom < source_kinds.len()
                        && source_kinds[atom] != SpinorSourceInputKind::NullSpinor
                    {
                        return Err(invalid(format!(
                            "bracket directly references non-null-spinor source {atom}"
                        )));
                    }
                    required = required.max(atom + 1);
                }
            }
        }
        Ok(required)
    }
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[derive(Clone, Debug)]
pub struct SpinorWorkspace {
    spinors: Vec<MasslessSpinors>,
    values: Vec<Complex64>,
    amplitudes: Vec<Complex64>,
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[derive(Clone, Debug)]
pub struct SpinorBatchWorkspace {
    point_capacity: usize,
    node_slots: Vec<usize>,
    value_slot_count: usize,
    spinors: Vec<MasslessSpinors>,
    values: Vec<Complex64>,
    compensation: Vec<f64>,
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
impl SpinorBatchWorkspace {
    fn new(dag: &SpinorDag, point_capacity: usize) -> RusticolResult<Self> {
        Self::new_with_spinor_slots(dag, point_capacity, usize::from(dag.spinor_atom_count) + 1)
    }

    fn new_with_spinor_slots(
        dag: &SpinorDag,
        point_capacity: usize,
        spinor_slot_count: usize,
    ) -> RusticolResult<Self> {
        if point_capacity == 0 {
            return Err(invalid("batch workspace point capacity must be positive"));
        }
        let zero = Complex64::new(0.0, 0.0);
        let zero_spinor = MasslessSpinors {
            undotted: [zero; 2],
            dotted: [zero; 2],
        };
        let spinor_count = spinor_slot_count
            .checked_mul(point_capacity)
            .ok_or_else(|| invalid("batch spinor workspace size overflows"))?;
        let (node_slots, value_slot_count) = batch_value_slot_layout(dag)?;
        let value_count = value_slot_count
            .checked_mul(point_capacity)
            .ok_or_else(|| invalid("batch scalar workspace size overflows"))?;
        Ok(Self {
            point_capacity,
            node_slots,
            value_slot_count,
            spinors: vec![zero_spinor; spinor_count],
            values: vec![zero; value_count],
            compensation: vec![0.0; point_capacity],
        })
    }

    fn validate_for(&self, dag: &SpinorDag, point_count: usize) -> RusticolResult<()> {
        self.validate_for_with_spinor_slots(
            dag,
            point_count,
            usize::from(dag.spinor_atom_count) + 1,
        )
    }

    fn validate_for_with_spinor_slots(
        &self,
        dag: &SpinorDag,
        point_count: usize,
        required_spinor_slots: usize,
    ) -> RusticolResult<()> {
        if point_count == 0 || point_count > self.point_capacity {
            return Err(invalid(format!(
                "batch point count {point_count} is outside workspace capacity {}",
                self.point_capacity
            )));
        }
        if self.spinors.len() < required_spinor_slots * self.point_capacity
            || self.spinors.len() > (usize::from(dag.spinor_atom_count) + 1) * self.point_capacity
            || self.spinors.len() % self.point_capacity != 0
            || self.node_slots.len() != dag.nodes.len()
            || self.values.len() != self.value_slot_count * self.point_capacity
            || self.compensation.len() != self.point_capacity
        {
            return Err(invalid("batch workspace belongs to a different graph"));
        }
        Ok(())
    }

    pub fn allocated_bytes(&self) -> usize {
        self.spinors.len() * std::mem::size_of::<MasslessSpinors>()
            + self.values.len() * std::mem::size_of::<Complex64>()
            + self.compensation.len() * std::mem::size_of::<f64>()
            + self.node_slots.len() * std::mem::size_of::<usize>()
    }

    pub const fn value_slot_count(&self) -> usize {
        self.value_slot_count
    }

    fn node_slot(&self, node: SpinorNodeId) -> RusticolResult<usize> {
        self.node_slots
            .get(usize::try_from(node).map_err(|_| invalid("node ID exceeds usize"))?)
            .copied()
            .ok_or_else(|| invalid("node ID is outside the batch layout"))
    }
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn batch_value_slot_layout(dag: &SpinorDag) -> RusticolResult<(Vec<usize>, usize)> {
    let node_count = dag.nodes.len();
    let mut last_use = (0..node_count).collect::<Vec<_>>();
    for (consumer, node) in dag.nodes.iter().enumerate() {
        let mut observe = |operand: SpinorNodeId| -> RusticolResult<()> {
            let operand = usize::try_from(operand).map_err(|_| invalid("node ID exceeds usize"))?;
            let Some(last) = last_use.get_mut(operand) else {
                return Err(invalid("batch layout operand is outside the graph"));
            };
            if operand >= consumer {
                return Err(invalid("batch layout graph is not topologically ordered"));
            }
            *last = (*last).max(consumer);
            Ok(())
        };
        match node {
            SpinorNode::Sum(operands) | SpinorNode::Product(operands) => {
                for operand in operands.iter().copied() {
                    observe(operand)?;
                }
            }
            SpinorNode::Reciprocal(operand) => observe(*operand)?,
            SpinorNode::Constant(_)
            | SpinorNode::Parameter(_)
            | SpinorNode::Kinematic(_)
            | SpinorNode::Bracket { .. } => {}
        }
    }
    for root in &dag.roots {
        let root =
            usize::try_from(root.amplitude).map_err(|_| invalid("root node ID exceeds usize"))?;
        let Some(last) = last_use.get_mut(root) else {
            return Err(invalid("batch layout root is outside the graph"));
        };
        *last = node_count;
    }
    let mut release_after = vec![Vec::new(); node_count];
    for (node, last) in last_use.iter().copied().enumerate() {
        if last < node_count {
            release_after[last].push(node);
        }
    }
    let mut node_slots = Vec::with_capacity(node_count);
    let mut free_slots = Vec::new();
    let mut slot_count = 0;
    for node in 0..node_count {
        if node > 0 {
            for released in &release_after[node - 1] {
                free_slots.push(node_slots[*released]);
            }
        }
        let slot = free_slots.pop().unwrap_or_else(|| {
            let slot = slot_count;
            slot_count += 1;
            slot
        });
        node_slots.push(slot);
    }
    Ok((node_slots, slot_count))
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn disjoint_value_planes(
    values: &mut [Complex64],
    destination_start: usize,
    source_start: usize,
    length: usize,
) -> RusticolResult<(&mut [Complex64], &[Complex64])> {
    if destination_start == source_start {
        return Err(invalid(
            "batch liveness layout aliased an input with its output",
        ));
    }
    if destination_start < source_start {
        let (before_source, source_and_after) = values.split_at_mut(source_start);
        let destination = before_source
            .get_mut(destination_start..destination_start + length)
            .ok_or_else(|| invalid("batch destination plane is outside the workspace"))?;
        let source = source_and_after
            .get(..length)
            .ok_or_else(|| invalid("batch source plane is outside the workspace"))?;
        return Ok((destination, source));
    }
    let (before_destination, destination_and_after) = values.split_at_mut(destination_start);
    let source = before_destination
        .get(source_start..source_start + length)
        .ok_or_else(|| invalid("batch source plane is outside the workspace"))?;
    let destination = destination_and_after
        .get_mut(..length)
        .ok_or_else(|| invalid("batch destination plane is outside the workspace"))?;
    Ok((destination, source))
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
impl SpinorWorkspace {
    fn new(dag: &SpinorDag) -> Self {
        Self::new_with_spinor_slots(dag, usize::from(dag.spinor_atom_count) + 1)
    }

    fn new_with_spinor_slots(dag: &SpinorDag, spinor_slot_count: usize) -> Self {
        let zero = Complex64::new(0.0, 0.0);
        let zero_spinor = MasslessSpinors {
            undotted: [zero; 2],
            dotted: [zero; 2],
        };
        Self {
            spinors: vec![zero_spinor; spinor_slot_count],
            values: vec![zero; dag.nodes.len()],
            amplitudes: vec![zero; dag.roots.len()],
        }
    }

    fn validate_for(&self, dag: &SpinorDag) -> RusticolResult<()> {
        self.validate_for_with_spinor_slots(dag, usize::from(dag.spinor_atom_count) + 1)
    }

    fn validate_for_with_spinor_slots(
        &self,
        dag: &SpinorDag,
        required_spinor_slots: usize,
    ) -> RusticolResult<()> {
        if self.spinors.len() < required_spinor_slots
            || self.spinors.len() > usize::from(dag.spinor_atom_count) + 1
            || self.values.len() != dag.nodes.len()
            || self.amplitudes.len() != dag.roots.len()
        {
            return Err(invalid("evaluation workspace belongs to a different graph"));
        }
        Ok(())
    }

    pub fn amplitudes(&self) -> &[Complex64] {
        &self.amplitudes
    }

    pub fn allocated_bytes(&self) -> usize {
        self.spinors.len() * std::mem::size_of::<MasslessSpinors>()
            + (self.values.len() + self.amplitudes.len()) * std::mem::size_of::<Complex64>()
    }
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorDagEvaluation {
    amplitudes: Vec<Complex64>,
    helicity_sum: f64,
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
impl SpinorDagEvaluation {
    pub fn amplitudes(&self) -> &[Complex64] {
        &self.amplitudes
    }

    pub const fn helicity_sum(&self) -> f64 {
        self.helicity_sum
    }
}

/// Content-addressed builder with exact local simplification.
pub struct SpinorDagBuilder {
    momentum_count: u16,
    spinor_atom_count: u16,
    parameter_count: u16,
    massive_sources: Vec<MassiveSpinorSource>,
    nodes: Vec<SpinorNode>,
    interner: BTreeMap<SpinorNode, SpinorNodeId>,
    roots: Vec<SpinorAmplitudeRoot>,
    rewrite_stats: SpinorRewriteStats,
    zero: SpinorNodeId,
    one: SpinorNodeId,
}

impl SpinorDagBuilder {
    pub fn new(momentum_count: u16) -> RusticolResult<Self> {
        Self::new_impl(momentum_count, 0, &[])
    }

    /// Create a graph with caller-owned complex scalar parameters and no
    /// implicit massive source layout.
    pub fn new_with_parameters(momentum_count: u16, parameter_count: u16) -> RusticolResult<Self> {
        Self::new_impl(momentum_count, parameter_count, &[])
    }

    /// Create a graph with one on-shell massive-vector source.  The source
    /// occupies its ordinary momentum slot, while two extra null spinor atoms
    /// are derived from it at evaluation time. Parameters are complex scalar
    /// leaves supplied in a stable zero-based order by the caller.
    pub fn new_with_massive_vector(
        momentum_count: u16,
        massive_source: u16,
        parameter_count: u16,
    ) -> RusticolResult<Self> {
        Self::new_impl(momentum_count, parameter_count, &[massive_source])
    }

    /// Create a graph with two on-shell massive-fermion sources. Runtime
    /// parameters remain caller-defined scalar leaves; the massive-quark
    /// builder below binds them as `[mass, width]`.
    fn new_with_massive_fermion_pair(
        momentum_count: u16,
        fermion_source: u16,
        antifermion_source: u16,
        parameter_count: u16,
    ) -> RusticolResult<Self> {
        if fermion_source == antifermion_source {
            return Err(invalid(
                "massive fermion and antifermion sources must differ",
            ));
        }
        Self::new_impl(
            momentum_count,
            parameter_count,
            &[fermion_source, antifermion_source],
        )
    }

    fn new_impl(
        momentum_count: u16,
        parameter_count: u16,
        massive_source_slots: &[u16],
    ) -> RusticolResult<Self> {
        if momentum_count < 2 {
            return Err(invalid("at least two momentum sources are required"));
        }
        let mut massive_sources = Vec::with_capacity(massive_source_slots.len());
        let mut observed_sources = BTreeSet::new();
        let mut spinor_atom_count = momentum_count;
        for source in massive_source_slots.iter().copied() {
            if source >= momentum_count {
                return Err(invalid(format!(
                    "massive source {source} is outside a {momentum_count}-source graph"
                )));
            }
            if !observed_sources.insert(source) {
                return Err(invalid(format!("massive source {source} is repeated")));
            }
            let k_atom = spinor_atom_count;
            let r_atom = spinor_atom_count
                .checked_add(1)
                .ok_or_else(|| invalid("massive spinor atom count overflows u16"))?;
            spinor_atom_count = spinor_atom_count
                .checked_add(2)
                .ok_or_else(|| invalid("massive spinor atom count overflows u16"))?;
            massive_sources.push(MassiveSpinorSource {
                source,
                k_atom,
                r_atom,
            });
        }
        let mut builder = Self {
            momentum_count,
            spinor_atom_count,
            parameter_count,
            massive_sources,
            nodes: Vec::new(),
            interner: BTreeMap::new(),
            roots: Vec::new(),
            rewrite_stats: SpinorRewriteStats::default(),
            zero: 0,
            one: 0,
        };
        builder.zero = builder.intern(SpinorNode::Constant(ExactComplexRational::ZERO))?;
        builder.one = builder.intern(SpinorNode::Constant(ExactComplexRational::ONE))?;
        Ok(builder)
    }

    pub const fn zero(&self) -> SpinorNodeId {
        self.zero
    }

    pub const fn one(&self) -> SpinorNodeId {
        self.one
    }

    /// The one shared null reference spinor used by external polarizations.
    /// It is deliberately outside the physical momentum-source domain.
    pub const fn reference_atom(&self) -> u16 {
        self.spinor_atom_count
    }

    pub fn constant(&mut self, value: ExactComplexRational) -> RusticolResult<SpinorNodeId> {
        self.intern(SpinorNode::Constant(value))
    }

    pub fn parameter(&mut self, index: u16) -> RusticolResult<SpinorNodeId> {
        if index >= self.parameter_count {
            return Err(invalid(format!(
                "parameter {index} is outside a {}-parameter graph",
                self.parameter_count
            )));
        }
        self.intern(SpinorNode::Parameter(index))
    }

    pub fn kinematic(&mut self, scalar: SpinorKinematicScalar) -> RusticolResult<SpinorNodeId> {
        if let SpinorKinematicScalar::InverseMass { source } = scalar {
            if !self
                .massive_sources
                .iter()
                .any(|entry| entry.source == source)
            {
                return Err(invalid(format!(
                    "inverse-mass source {source} is not a massive source in this graph"
                )));
            }
        }
        self.intern(SpinorNode::Kinematic(scalar))
    }

    fn massive_vector_atoms(&self, source: u16) -> RusticolResult<(u16, u16)> {
        let massive = self
            .massive_sources
            .iter()
            .find(|entry| entry.source == source)
            .ok_or_else(|| invalid(format!("source {source} is not massive")))?;
        Ok((massive.k_atom, massive.r_atom))
    }

    pub fn bracket(
        &mut self,
        kind: SpinorBracketKind,
        left: u16,
        right: u16,
    ) -> RusticolResult<SpinorNodeId> {
        self.validate_spinor_atom(left)?;
        self.validate_spinor_atom(right)?;
        if left == right {
            self.rewrite_stats.self_product_zeros += 1;
            return Ok(self.zero);
        }
        if left < right {
            return self.intern(SpinorNode::Bracket { kind, left, right });
        }
        self.rewrite_stats.antisymmetry_rewrites += 1;
        let bracket = self.intern(SpinorNode::Bracket {
            kind,
            left: right,
            right: left,
        })?;
        self.negate(bracket)
    }

    pub fn angle(&mut self, left: u16, right: u16) -> RusticolResult<SpinorNodeId> {
        self.bracket(SpinorBracketKind::Angle, left, right)
    }

    pub fn square(&mut self, left: u16, right: u16) -> RusticolResult<SpinorNodeId> {
        self.bracket(SpinorBracketKind::Square, left, right)
    }

    pub fn sum(
        &mut self,
        operands: impl IntoIterator<Item = SpinorNodeId>,
    ) -> RusticolResult<SpinorNodeId> {
        let mut pending = Vec::new();
        for operand in operands {
            match self.node(operand)?.clone() {
                SpinorNode::Sum(nested) => pending.extend(nested.iter().copied()),
                _ => pending.push(operand),
            }
        }
        let mut collected = BTreeMap::<Option<SpinorNodeId>, ExactComplexRational>::new();
        for operand in pending {
            let (coefficient, residual) = match self.node(operand)?.clone() {
                SpinorNode::Constant(value) => {
                    self.rewrite_stats.constant_folds += 1;
                    (value, None)
                }
                SpinorNode::Product(factors) => {
                    let mut coefficient = ExactComplexRational::ONE;
                    let mut residual_factors = Vec::new();
                    for factor in factors.iter().copied() {
                        if let SpinorNode::Constant(value) = self.node(factor)? {
                            coefficient = coefficient.checked_mul(*value)?;
                            self.rewrite_stats.constant_folds += 1;
                        } else {
                            residual_factors.push(factor);
                        }
                    }
                    let residual = match residual_factors.as_slice() {
                        [] => None,
                        [single] => Some(*single),
                        _ => Some(
                            self.intern(SpinorNode::Product(residual_factors.into_boxed_slice()))?,
                        ),
                    };
                    (coefficient, residual)
                }
                _ => (ExactComplexRational::ONE, Some(operand)),
            };
            let previous = collected
                .get(&residual)
                .copied()
                .unwrap_or(ExactComplexRational::ZERO);
            collected.insert(residual, previous.checked_add(coefficient)?);
        }
        let mut terms = Vec::with_capacity(collected.len());
        for (residual, coefficient) in collected {
            if coefficient.is_zero() {
                self.rewrite_stats.constant_folds += 1;
                continue;
            }
            let term = match residual {
                None => self.constant(coefficient)?,
                Some(residual) if coefficient == ExactComplexRational::ONE => residual,
                Some(residual) => {
                    let coefficient = self.constant(coefficient)?;
                    self.product([coefficient, residual])?
                }
            };
            terms.push(term);
        }
        terms.sort_unstable();
        match terms.as_slice() {
            [] => Ok(self.zero),
            [single] => Ok(*single),
            _ => {
                let common_factors = self.common_nonconstant_factors(&terms)?;
                if common_factors.is_empty() {
                    return self.intern(SpinorNode::Sum(terms.into_boxed_slice()));
                }
                let mut residuals = Vec::with_capacity(terms.len());
                for term in terms {
                    let mut factors = match self.node(term)?.clone() {
                        SpinorNode::Product(factors) => factors.into_vec(),
                        _ => vec![term],
                    };
                    for common in &common_factors {
                        let position = factors.binary_search(common).map_err(|_| {
                            invalid("sum common-factor accounting became inconsistent")
                        })?;
                        factors.remove(position);
                    }
                    residuals.push(self.product(factors)?);
                }
                let residual = self.sum(residuals)?;
                self.rewrite_stats.sum_factorizations += 1;
                self.product(common_factors.into_iter().chain([residual]))
            }
        }
    }

    fn common_nonconstant_factors(
        &self,
        terms: &[SpinorNodeId],
    ) -> RusticolResult<Vec<SpinorNodeId>> {
        let factor_counts = |term: SpinorNodeId| -> RusticolResult<BTreeMap<SpinorNodeId, usize>> {
            let factors = match self.node(term)? {
                SpinorNode::Product(factors) => factors.as_ref(),
                _ => std::slice::from_ref(&term),
            };
            let mut counts = BTreeMap::new();
            for factor in factors {
                if matches!(self.node(*factor)?, SpinorNode::Constant(_)) {
                    continue;
                }
                *counts.entry(*factor).or_insert(0) += 1;
            }
            Ok(counts)
        };
        let mut common = factor_counts(terms[0])?;
        for term in &terms[1..] {
            let counts = factor_counts(*term)?;
            common.retain(|factor, multiplicity| {
                *multiplicity = (*multiplicity).min(counts.get(factor).copied().unwrap_or(0));
                *multiplicity != 0
            });
            if common.is_empty() {
                break;
            }
        }
        Ok(common
            .into_iter()
            .flat_map(|(factor, multiplicity)| std::iter::repeat_n(factor, multiplicity))
            .collect())
    }

    pub fn product(
        &mut self,
        operands: impl IntoIterator<Item = SpinorNodeId>,
    ) -> RusticolResult<SpinorNodeId> {
        let mut flattened = Vec::new();
        let mut constant = ExactComplexRational::ONE;
        for operand in operands {
            match self.node(operand)?.clone() {
                SpinorNode::Constant(value) => {
                    if value.is_zero() {
                        self.rewrite_stats.constant_folds += 1;
                        return Ok(self.zero);
                    }
                    constant = constant.checked_mul(value)?;
                    self.rewrite_stats.constant_folds += 1;
                }
                SpinorNode::Product(nested) => flattened.extend(nested.iter().copied()),
                _ => flattened.push(operand),
            }
        }
        if constant.is_zero() {
            return Ok(self.zero);
        }
        if constant != ExactComplexRational::ONE {
            flattened.push(self.constant(constant)?);
        }
        flattened.sort_unstable();
        let mut removed = vec![false; flattened.len()];
        for reciprocal_index in 0..flattened.len() {
            let SpinorNode::Reciprocal(base) = self.node(flattened[reciprocal_index])? else {
                continue;
            };
            let base = *base;
            let Some(base_index) = flattened.iter().enumerate().find_map(|(index, candidate)| {
                (!removed[index] && index != reciprocal_index && *candidate == base)
                    .then_some(index)
            }) else {
                continue;
            };
            removed[base_index] = true;
            removed[reciprocal_index] = true;
            self.rewrite_stats.reciprocal_cancellations += 1;
        }
        if removed.iter().any(|removed| *removed) {
            flattened = flattened
                .into_iter()
                .enumerate()
                .filter_map(|(index, node)| (!removed[index]).then_some(node))
                .collect();
        }
        match flattened.as_slice() {
            [] => Ok(self.one),
            [single] => Ok(*single),
            _ => self.intern(SpinorNode::Product(flattened.into_boxed_slice())),
        }
    }

    pub fn negate(&mut self, node: SpinorNodeId) -> RusticolResult<SpinorNodeId> {
        let minus_one = self.constant(ExactComplexRational::ONE.checked_neg()?)?;
        self.product([minus_one, node])
    }

    pub fn reciprocal(&mut self, node: SpinorNodeId) -> RusticolResult<SpinorNodeId> {
        match self.node(node)?.clone() {
            SpinorNode::Constant(value) => {
                if value.is_zero() {
                    return Err(invalid("cannot take the reciprocal of exact zero"));
                }
                self.rewrite_stats.constant_folds += 1;
                self.constant(ExactComplexRational::ONE.checked_div(value)?)
            }
            SpinorNode::Reciprocal(inner) => Ok(inner),
            _ => self.intern(SpinorNode::Reciprocal(node)),
        }
    }

    pub fn quotient(
        &mut self,
        numerator: SpinorNodeId,
        denominator: SpinorNodeId,
    ) -> RusticolResult<SpinorNodeId> {
        let reciprocal = self.reciprocal(denominator)?;
        self.product([numerator, reciprocal])
    }

    pub fn pow(&mut self, base: SpinorNodeId, exponent: u16) -> RusticolResult<SpinorNodeId> {
        if exponent == 0 {
            return Ok(self.one);
        }
        self.product(std::iter::repeat_n(base, usize::from(exponent)))
    }

    /// Replace a two-term Pluecker relation by the third bracket product.
    /// The matcher is orientation-independent and uses exact coefficients.
    pub fn simplify_schouten(&mut self, expression: SpinorNodeId) -> RusticolResult<SpinorNodeId> {
        let SpinorNode::Sum(terms) = self.node(expression)?.clone() else {
            return Ok(expression);
        };
        if terms.len() != 2 {
            return Ok(expression);
        }
        let Some(first) = self.bracket_monomial(terms[0])? else {
            return Ok(expression);
        };
        let Some(second) = self.bracket_monomial(terms[1])? else {
            return Ok(expression);
        };
        if first.factors.len() != 2
            || second.factors.len() != 2
            || first.factors[0].0 != first.factors[1].0
            || second.factors[0].0 != second.factors[1].0
            || first.factors[0].0 != second.factors[0].0
        {
            return Ok(expression);
        }
        let kind = first.factors[0].0;
        let labels = first
            .factors
            .iter()
            .chain(&second.factors)
            .flat_map(|(_, left, right)| [*left, *right])
            .collect::<BTreeSet<_>>();
        if labels.len() != 4 {
            return Ok(expression);
        }
        let labels = labels.into_iter().collect::<Vec<_>>();
        for &i in &labels {
            for &j in &labels {
                for &k in &labels {
                    for &l in &labels {
                        if [i, j, k, l].into_iter().collect::<BTreeSet<_>>().len() != 4 {
                            continue;
                        }
                        let Some((first_sign, first_key)) =
                            bracket_product_key(kind, [(i, k), (j, l)])
                        else {
                            continue;
                        };
                        let Some((second_sign, second_key)) =
                            bracket_product_key(kind, [(i, l), (k, j)])
                        else {
                            continue;
                        };
                        let Some((target_sign, target_key)) =
                            bracket_product_key(kind, [(i, j), (k, l)])
                        else {
                            continue;
                        };
                        let candidates = [(&first, &second), (&second, &first)];
                        for (left, right) in candidates {
                            if left.factors != first_key || right.factors != second_key {
                                continue;
                            }
                            let scale = signed_exact(left.coefficient, first_sign)?;
                            if right.coefficient != signed_exact(scale, second_sign)? {
                                continue;
                            }
                            let coefficient = signed_exact(scale, target_sign)?;
                            let mut factors = Vec::with_capacity(3);
                            if coefficient != ExactComplexRational::ONE {
                                factors.push(self.constant(coefficient)?);
                            }
                            for (_, left, right) in target_key {
                                factors.push(self.intern(SpinorNode::Bracket {
                                    kind,
                                    left,
                                    right,
                                })?);
                            }
                            self.rewrite_stats.schouten_rewrites += 1;
                            return self.product(factors);
                        }
                    }
                }
            }
        }
        Ok(expression)
    }

    /// Apply the two-component Fierz identity at construction time.  No
    /// Lorentz-vector current node enters the resulting graph.
    pub fn fierz_current_contraction(
        &mut self,
        chirality: SpinorChirality,
        i: u16,
        j: u16,
        k: u16,
        l: u16,
    ) -> RusticolResult<SpinorNodeId> {
        let two = self.constant(ExactComplexRational::new(
            ExactRational::new(2, 1)?,
            ExactRational::ZERO,
        ))?;
        let (first, second) = match chirality {
            SpinorChirality::Positive => (self.square(i, k)?, self.angle(l, j)?),
            SpinorChirality::Negative => (self.angle(i, k)?, self.square(l, j)?),
        };
        self.rewrite_stats.fierz_rewrites += 1;
        self.product([two, first, second])
    }

    pub fn add_root(
        &mut self,
        helicities: impl Into<Box<[i8]>>,
        amplitude: SpinorNodeId,
    ) -> RusticolResult<()> {
        self.add_root_with_multiplicity(helicities, amplitude, 1)
    }

    pub fn add_root_with_multiplicity(
        &mut self,
        helicities: impl Into<Box<[i8]>>,
        amplitude: SpinorNodeId,
        multiplicity: u16,
    ) -> RusticolResult<()> {
        self.node(amplitude)?;
        if multiplicity == 0 {
            return Err(invalid("amplitude root multiplicity must be positive"));
        }
        let helicities = helicities.into();
        if helicities.len() != usize::from(self.momentum_count)
            || helicities
                .iter()
                .any(|helicity| !matches!(helicity, -1 | 0 | 1))
        {
            return Err(invalid(
                "amplitude root helicities must contain one -1/0/+1 entry per momentum",
            ));
        }
        if self
            .roots
            .iter()
            .any(|root| root.helicities.as_ref() == helicities.as_ref())
        {
            return Err(invalid("amplitude root repeats a helicity configuration"));
        }
        let structural_zero = amplitude == self.zero;
        if structural_zero {
            self.rewrite_stats.structural_zero_roots += 1;
        }
        self.roots.push(SpinorAmplitudeRoot {
            helicities,
            amplitude,
            structural_zero,
            multiplicity,
        });
        Ok(())
    }

    pub fn finish(mut self) -> RusticolResult<SpinorDag> {
        if self.roots.is_empty() {
            return Err(invalid("at least one amplitude root is required"));
        }
        self.roots
            .sort_unstable_by(|left, right| left.helicities.cmp(&right.helicities));
        self.prune_dead_nodes()?;
        let reference_atom = self.reference_atom();
        let uses_reference_atom = self.nodes.iter().any(|node| {
            matches!(
                node,
                SpinorNode::Bracket { left, right, .. }
                    if *left == reference_atom || *right == reference_atom
            )
        });
        Ok(SpinorDag {
            momentum_count: self.momentum_count,
            spinor_atom_count: self.spinor_atom_count,
            parameter_count: self.parameter_count,
            massive_sources: self.massive_sources.into_boxed_slice(),
            uses_reference_atom,
            nodes: self.nodes.into_boxed_slice(),
            roots: self.roots.into_boxed_slice(),
            rewrite_stats: self.rewrite_stats,
        })
    }

    fn validate_spinor_atom(&self, atom: u16) -> RusticolResult<()> {
        if atom > self.reference_atom() {
            return Err(invalid(format!(
                "spinor atom {atom} is outside a graph with {} physical sources, {} derived atoms, and one reference atom",
                self.momentum_count,
                self.spinor_atom_count - self.momentum_count,
            )));
        }
        Ok(())
    }

    fn node(&self, id: SpinorNodeId) -> RusticolResult<&SpinorNode> {
        self.nodes
            .get(usize::try_from(id).map_err(|_| invalid("node ID exceeds usize"))?)
            .ok_or_else(|| invalid(format!("node ID {id} is outside the graph")))
    }

    fn intern(&mut self, node: SpinorNode) -> RusticolResult<SpinorNodeId> {
        if let Some(id) = self.interner.get(&node) {
            return Ok(*id);
        }
        let id = u32::try_from(self.nodes.len())
            .map_err(|_| invalid("node count exceeds the u32 domain"))?;
        self.nodes.push(node.clone());
        self.interner.insert(node, id);
        Ok(id)
    }

    fn prune_dead_nodes(&mut self) -> RusticolResult<()> {
        let mut live = vec![false; self.nodes.len()];
        let mut pending = self
            .roots
            .iter()
            .map(SpinorAmplitudeRoot::amplitude)
            .collect::<Vec<_>>();
        while let Some(id) = pending.pop() {
            let index = usize::try_from(id).map_err(|_| invalid("node ID exceeds usize"))?;
            let Some(is_live) = live.get_mut(index) else {
                return Err(invalid(format!("node ID {id} is outside the graph")));
            };
            if *is_live {
                continue;
            }
            *is_live = true;
            match self.node(id)? {
                SpinorNode::Sum(operands) | SpinorNode::Product(operands) => {
                    pending.extend(operands.iter().copied());
                }
                SpinorNode::Reciprocal(operand) => pending.push(*operand),
                SpinorNode::Constant(_)
                | SpinorNode::Parameter(_)
                | SpinorNode::Kinematic(_)
                | SpinorNode::Bracket { .. } => {}
            }
        }

        let old_count = self.nodes.len();
        let old_nodes = std::mem::take(&mut self.nodes);
        let mut remap = vec![None; old_nodes.len()];
        let mut compact = Vec::with_capacity(live.iter().filter(|entry| **entry).count());
        for (old_index, node) in old_nodes.into_iter().enumerate() {
            if !live[old_index] {
                continue;
            }
            let new_id = u32::try_from(compact.len())
                .map_err(|_| invalid("node count exceeds the u32 domain"))?;
            let remap_operand = |operand: SpinorNodeId| -> RusticolResult<SpinorNodeId> {
                remap
                    .get(usize::try_from(operand).map_err(|_| invalid("node ID exceeds usize"))?)
                    .copied()
                    .flatten()
                    .ok_or_else(|| invalid("live node depends on an unavailable node"))
            };
            let node = match node {
                SpinorNode::Sum(operands) => SpinorNode::Sum(
                    operands
                        .iter()
                        .copied()
                        .map(remap_operand)
                        .collect::<RusticolResult<Vec<_>>>()?
                        .into_boxed_slice(),
                ),
                SpinorNode::Product(operands) => SpinorNode::Product(
                    operands
                        .iter()
                        .copied()
                        .map(remap_operand)
                        .collect::<RusticolResult<Vec<_>>>()?
                        .into_boxed_slice(),
                ),
                SpinorNode::Reciprocal(operand) => SpinorNode::Reciprocal(remap_operand(operand)?),
                leaf => leaf,
            };
            remap[old_index] = Some(new_id);
            compact.push(node);
        }
        for root in &mut self.roots {
            root.amplitude = remap
                .get(
                    usize::try_from(root.amplitude)
                        .map_err(|_| invalid("root node ID exceeds usize"))?,
                )
                .copied()
                .flatten()
                .ok_or_else(|| invalid("amplitude root was removed as dead"))?;
        }
        self.rewrite_stats.dead_nodes_pruned += u64::try_from(old_count - compact.len())
            .map_err(|_| invalid("dead-node count exceeds u64"))?;
        self.nodes = compact;
        Ok(())
    }

    fn bracket_monomial(&self, node: SpinorNodeId) -> RusticolResult<Option<BracketMonomial>> {
        let factors = match self.node(node)? {
            SpinorNode::Product(factors) => factors.as_ref(),
            SpinorNode::Bracket { .. } | SpinorNode::Constant(_) => std::slice::from_ref(&node),
            _ => return Ok(None),
        };
        let mut coefficient = ExactComplexRational::ONE;
        let mut brackets = Vec::new();
        for factor in factors {
            match self.node(*factor)? {
                SpinorNode::Constant(value) => coefficient = coefficient.checked_mul(*value)?,
                SpinorNode::Bracket { kind, left, right } => {
                    brackets.push((*kind, *left, *right));
                }
                _ => return Ok(None),
            }
        }
        brackets.sort_unstable();
        Ok(Some(BracketMonomial {
            coefficient,
            factors: brackets,
        }))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BracketMonomial {
    coefficient: ExactComplexRational,
    factors: Vec<(SpinorBracketKind, u16, u16)>,
}

fn signed_exact(value: ExactComplexRational, sign: i8) -> RusticolResult<ExactComplexRational> {
    match sign {
        1 => Ok(value),
        -1 => value.checked_neg(),
        _ => Err(invalid("internal bracket sign is not +/-1")),
    }
}

type BracketProductKey = Vec<(SpinorBracketKind, u16, u16)>;

fn bracket_product_key<const N: usize>(
    kind: SpinorBracketKind,
    pairs: [(u16, u16); N],
) -> Option<(i8, BracketProductKey)> {
    let mut sign = 1_i8;
    let mut result = Vec::with_capacity(N);
    for (left, right) in pairs {
        if left == right {
            return None;
        }
        if left < right {
            result.push((kind, left, right));
        } else {
            sign = -sign;
            result.push((kind, right, left));
        }
    }
    result.sort_unstable();
    Some((sign, result))
}

fn four_point_bracket_via_schouten(
    builder: &mut SpinorDagBuilder,
    kind: SpinorBracketKind,
    left: u16,
    right: u16,
) -> RusticolResult<SpinorNodeId> {
    let direct = builder.bracket(kind, left, right)?;
    let auxiliary = match (left, right) {
        (0, 2) => (1, 3),
        (1, 3) => (0, 2),
        _ => return Ok(direct),
    };
    let first_left = builder.bracket(kind, 0, 1)?;
    let first_right = builder.bracket(kind, 2, 3)?;
    let first = builder.product([first_left, first_right])?;
    let second_left = builder.bracket(kind, 0, 3)?;
    let second_right = builder.bracket(kind, 1, 2)?;
    let second = builder.product([second_left, second_right])?;
    let expanded = builder.sum([first, second])?;
    let reduced = builder.simplify_schouten(expanded)?;
    let auxiliary = builder.bracket(kind, auxiliary.0, auxiliary.1)?;
    let recovered = builder.quotient(reduced, auxiliary)?;
    if recovered != direct {
        return Err(invalid(
            "Schouten reduction did not recover the requested four-point bracket",
        ));
    }
    Ok(recovered)
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct SpinorDyad {
    undotted: u16,
    dotted: u16,
}

/// A bispinor kept as a canonical sparse sum of rank-one spinor dyads.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct BispinorExpression {
    terms: BTreeMap<SpinorDyad, SpinorNodeId>,
}

impl BispinorExpression {
    pub(crate) fn dyad(undotted: u16, dotted: u16, coefficient: SpinorNodeId) -> Self {
        Self {
            terms: BTreeMap::from([(SpinorDyad { undotted, dotted }, coefficient)]),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BivectorTerm {
    coefficient: SpinorNodeId,
    left: BispinorExpression,
    right: BispinorExpression,
}

/// An antisymmetric Lorentz tensor kept as a sparse sum of decomposable
/// bivectors.  A term `(coefficient, left, right)` denotes
/// `coefficient * (left ∧ right)` in the authenticated component ordering
/// `(01, 02, 03, 12, 13, 23)`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct BivectorExpression {
    terms: Vec<BivectorTerm>,
}

pub(crate) fn bivector_wedge_expression(
    builder: &SpinorDagBuilder,
    left: &BispinorExpression,
    right: &BispinorExpression,
) -> BivectorExpression {
    BivectorExpression {
        terms: vec![BivectorTerm {
            coefficient: builder.one(),
            left: left.clone(),
            right: right.clone(),
        }],
    }
}

pub(crate) fn bivector_sum(
    builder: &SpinorDagBuilder,
    expressions: impl IntoIterator<Item = BivectorExpression>,
) -> BivectorExpression {
    BivectorExpression {
        terms: expressions
            .into_iter()
            .flat_map(|expression| expression.terms)
            .filter(|term| term.coefficient != builder.zero())
            .collect(),
    }
}

pub(crate) fn bivector_scale(
    builder: &mut SpinorDagBuilder,
    scalar: SpinorNodeId,
    expression: &BivectorExpression,
) -> RusticolResult<BivectorExpression> {
    if scalar == builder.zero() {
        return Ok(BivectorExpression::default());
    }
    let mut terms = Vec::with_capacity(expression.terms.len());
    for term in &expression.terms {
        let coefficient = builder.product([scalar, term.coefficient])?;
        if coefficient != builder.zero() {
            terms.push(BivectorTerm {
                coefficient,
                left: term.left.clone(),
                right: term.right.clone(),
            });
        }
    }
    Ok(BivectorExpression { terms })
}

/// Apply an antisymmetric tensor to a vector without materializing tensor
/// components.  For the certified direct primitive,
/// `(left ∧ right) vector = right (left·vector) - left (right·vector)`.
pub(crate) fn bivector_vector_expression(
    builder: &mut SpinorDagBuilder,
    tensor: &BivectorExpression,
    vector: &BispinorExpression,
) -> RusticolResult<BispinorExpression> {
    let mut terms = Vec::with_capacity(tensor.terms.len() * 2);
    for term in &tensor.terms {
        let left_dot_vector = bispinor_dot_expression(builder, &term.left, vector)?;
        let right_dot_vector = bispinor_dot_expression(builder, &term.right, vector)?;
        let right_scale = builder.product([term.coefficient, left_dot_vector])?;
        terms.push(bispinor_scale(builder, right_scale, &term.right)?);
        let negative_coefficient = builder.negate(term.coefficient)?;
        let left_scale = builder.product([negative_coefficient, right_dot_vector])?;
        terms.push(bispinor_scale(builder, left_scale, &term.left)?);
    }
    bispinor_sum(builder, terms)
}

fn exact_real(numerator: i128, denominator: i128) -> RusticolResult<ExactComplexRational> {
    Ok(ExactComplexRational::new(
        ExactRational::new(numerator, denominator)?,
        ExactRational::ZERO,
    ))
}

pub(crate) fn bispinor_sum(
    builder: &mut SpinorDagBuilder,
    expressions: impl IntoIterator<Item = BispinorExpression>,
) -> RusticolResult<BispinorExpression> {
    let mut terms = BTreeMap::<SpinorDyad, Vec<SpinorNodeId>>::new();
    for expression in expressions {
        for (dyad, coefficient) in expression.terms {
            terms.entry(dyad).or_default().push(coefficient);
        }
    }
    let mut result = BTreeMap::new();
    for (dyad, coefficients) in terms {
        let coefficient = builder.sum(coefficients)?;
        let coefficient = builder.simplify_schouten(coefficient)?;
        if coefficient != builder.zero() {
            result.insert(dyad, coefficient);
        }
    }
    Ok(BispinorExpression { terms: result })
}

pub(crate) fn bispinor_scale(
    builder: &mut SpinorDagBuilder,
    scalar: SpinorNodeId,
    expression: &BispinorExpression,
) -> RusticolResult<BispinorExpression> {
    if scalar == builder.zero() {
        return Ok(BispinorExpression::default());
    }
    let mut result = BTreeMap::new();
    for (dyad, coefficient) in &expression.terms {
        let coefficient = builder.product([scalar, *coefficient])?;
        if coefficient != builder.zero() {
            result.insert(*dyad, coefficient);
        }
    }
    Ok(BispinorExpression { terms: result })
}

/// Immediately lower a Lorentz contraction of dyads with the two-component
/// Fierz identity. No four-vector component node survives this boundary.
pub(crate) fn bispinor_dot_expression(
    builder: &mut SpinorDagBuilder,
    left: &BispinorExpression,
    right: &BispinorExpression,
) -> RusticolResult<SpinorNodeId> {
    let minus_half = builder.constant(exact_real(-1, 2)?)?;
    let mut terms = Vec::with_capacity(left.terms.len() * right.terms.len());
    for (left_dyad, left_coefficient) in &left.terms {
        for (right_dyad, right_coefficient) in &right.terms {
            let angle = builder.angle(left_dyad.undotted, right_dyad.undotted)?;
            let square = builder.square(left_dyad.dotted, right_dyad.dotted)?;
            builder.rewrite_stats.fierz_rewrites += 1;
            terms.push(builder.product([
                minus_half,
                *left_coefficient,
                *right_coefficient,
                angle,
                square,
            ])?);
        }
    }
    let result = builder.sum(terms)?;
    builder.simplify_schouten(result)
}

pub(crate) fn external_polarization_expression(
    builder: &mut SpinorDagBuilder,
    leg: u16,
    helicity: i8,
) -> RusticolResult<BispinorExpression> {
    let reference = builder.reference_atom();
    external_polarization_expression_with_reference(builder, leg, reference, helicity)
}

fn external_polarization_expression_with_reference(
    builder: &mut SpinorDagBuilder,
    leg: u16,
    reference: u16,
    helicity: i8,
) -> RusticolResult<BispinorExpression> {
    match helicity {
        1 => {
            let denominator = builder.angle(reference, leg)?;
            let coefficient = builder.reciprocal(denominator)?;
            Ok(BispinorExpression::dyad(reference, leg, coefficient))
        }
        -1 => {
            let denominator = builder.square(leg, reference)?;
            let coefficient = builder.reciprocal(denominator)?;
            Ok(BispinorExpression::dyad(leg, reference, coefficient))
        }
        _ => Err(invalid("gluon helicity must be -1 or +1")),
    }
}

fn massive_vector_polarization_expression(
    builder: &mut SpinorDagBuilder,
    source: u16,
    helicity: i8,
) -> RusticolResult<BispinorExpression> {
    let (k, r) = builder.massive_vector_atoms(source)?;
    match helicity {
        1 => {
            let denominator = builder.angle(r, k)?;
            let coefficient = builder.reciprocal(denominator)?;
            Ok(BispinorExpression::dyad(r, k, coefficient))
        }
        -1 => {
            let denominator = builder.square(k, r)?;
            let coefficient = builder.reciprocal(denominator)?;
            Ok(BispinorExpression::dyad(k, r, coefficient))
        }
        0 => {
            let inverse_mass = builder.kinematic(SpinorKinematicScalar::InverseMass { source })?;
            let sqrt_two = builder.kinematic(SpinorKinematicScalar::SqrtTwo)?;
            let inverse_sqrt_two = builder.reciprocal(sqrt_two)?;
            let coefficient = builder.product([inverse_mass, inverse_sqrt_two])?;
            let minus_coefficient = builder.negate(coefficient)?;
            Ok(BispinorExpression {
                terms: BTreeMap::from([
                    (
                        SpinorDyad {
                            undotted: k,
                            dotted: k,
                        },
                        coefficient,
                    ),
                    (
                        SpinorDyad {
                            undotted: r,
                            dotted: r,
                        },
                        minus_coefficient,
                    ),
                ]),
            })
        }
        _ => Err(invalid("massive-vector helicity must be -1, 0, or +1")),
    }
}

fn interval_momentum_expression(start: u16, end: u16, one: SpinorNodeId) -> BispinorExpression {
    BispinorExpression {
        terms: (start..=end)
            .map(|leg| {
                (
                    SpinorDyad {
                        undotted: leg,
                        dotted: leg,
                    },
                    one,
                )
            })
            .collect(),
    }
}

fn interval_mass_squared(
    builder: &mut SpinorDagBuilder,
    start: u16,
    end: u16,
) -> RusticolResult<SpinorNodeId> {
    let mut terms = Vec::new();
    for left in start..end {
        for right in (left + 1)..=end {
            let angle = builder.angle(left, right)?;
            let square = builder.square(right, left)?;
            terms.push(builder.product([angle, square])?);
        }
    }
    builder.sum(terms)
}

fn ordered_momentum_expression(atoms: &[u16], one: SpinorNodeId) -> BispinorExpression {
    BispinorExpression {
        terms: atoms
            .iter()
            .copied()
            .map(|atom| {
                (
                    SpinorDyad {
                        undotted: atom,
                        dotted: atom,
                    },
                    one,
                )
            })
            .collect(),
    }
}

/// Build a massless momentum bispinor from authenticated signed source
/// coefficients.  The coefficient nodes are kept symbolic so recurrence
/// exact factors remain owned by the lowering boundary.
pub(crate) fn signed_momentum_expression(terms: &[(u16, SpinorNodeId)]) -> BispinorExpression {
    BispinorExpression {
        terms: terms
            .iter()
            .copied()
            .map(|(atom, coefficient)| {
                (
                    SpinorDyad {
                        undotted: atom,
                        dotted: atom,
                    },
                    coefficient,
                )
            })
            .collect(),
    }
}

fn ordered_mass_squared(
    builder: &mut SpinorDagBuilder,
    atoms: &[u16],
) -> RusticolResult<SpinorNodeId> {
    let mut terms = Vec::new();
    for (left_index, left) in atoms.iter().copied().enumerate() {
        for right in atoms[(left_index + 1)..].iter().copied() {
            let angle = builder.angle(left, right)?;
            let square = builder.square(right, left)?;
            terms.push(builder.product([angle, square])?);
        }
    }
    builder.sum(terms)
}

/// A Weyl spinor whose two components are never expanded: only its linear
/// combination of external spinor atoms is retained. This is sufficient for
/// BCFW-shifted and factorized internal momenta because the scalar DAG only
/// observes spinors through antisymmetric brackets.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct LinearWeylExpression {
    terms: BTreeMap<u16, SpinorNodeId>,
}

impl LinearWeylExpression {
    pub(crate) fn atom(atom: u16, one: SpinorNodeId) -> Self {
        Self {
            terms: BTreeMap::from([(atom, one)]),
        }
    }
}

pub(crate) fn linear_weyl_sum(
    builder: &mut SpinorDagBuilder,
    expressions: impl IntoIterator<Item = LinearWeylExpression>,
) -> RusticolResult<LinearWeylExpression> {
    let mut terms = BTreeMap::<u16, Vec<SpinorNodeId>>::new();
    for expression in expressions {
        for (atom, coefficient) in expression.terms {
            terms.entry(atom).or_default().push(coefficient);
        }
    }
    let mut result = BTreeMap::new();
    for (atom, coefficients) in terms {
        let coefficient = builder.sum(coefficients)?;
        let coefficient = builder.simplify_schouten(coefficient)?;
        if coefficient != builder.zero() {
            result.insert(atom, coefficient);
        }
    }
    Ok(LinearWeylExpression { terms: result })
}

pub(crate) fn linear_weyl_scale(
    builder: &mut SpinorDagBuilder,
    scalar: SpinorNodeId,
    expression: &LinearWeylExpression,
) -> RusticolResult<LinearWeylExpression> {
    if scalar == builder.zero() {
        return Ok(LinearWeylExpression::default());
    }
    let mut terms = BTreeMap::new();
    for (atom, coefficient) in &expression.terms {
        let coefficient = builder.product([scalar, *coefficient])?;
        if coefficient != builder.zero() {
            terms.insert(*atom, coefficient);
        }
    }
    Ok(LinearWeylExpression { terms })
}

fn linear_weyl_bracket(
    builder: &mut SpinorDagBuilder,
    kind: SpinorBracketKind,
    left: &LinearWeylExpression,
    right: &LinearWeylExpression,
) -> RusticolResult<SpinorNodeId> {
    let mut terms = Vec::with_capacity(left.terms.len() * right.terms.len());
    for (left_atom, left_coefficient) in &left.terms {
        for (right_atom, right_coefficient) in &right.terms {
            let bracket = builder.bracket(kind, *left_atom, *right_atom)?;
            terms.push(builder.product([*left_coefficient, *right_coefficient, bracket])?);
        }
    }
    let result = builder.sum(terms)?;
    builder.simplify_schouten(result)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct OnShellSpinorLeg {
    undotted: LinearWeylExpression,
    dotted: LinearWeylExpression,
    helicity: i8,
}

impl OnShellSpinorLeg {
    fn external(atom: u16, helicity: i8, one: SpinorNodeId) -> Self {
        Self {
            undotted: LinearWeylExpression::atom(atom, one),
            dotted: LinearWeylExpression::atom(atom, one),
            helicity,
        }
    }
}

fn parke_taylor_linear_spinors(
    builder: &mut SpinorDagBuilder,
    legs: &[OnShellSpinorLeg],
    required_three_point_branch: Option<SpinorBracketKind>,
    imaginary_unit: SpinorNodeId,
) -> RusticolResult<Option<SpinorNodeId>> {
    if !(3..=5).contains(&legs.len()) {
        return Err(invalid(
            "a BCFW child must contain between three and five gluons",
        ));
    }
    if legs.iter().any(|leg| !matches!(leg.helicity, -1 | 1)) {
        return Err(invalid("a BCFW child contains an invalid helicity"));
    }
    let negative = legs
        .iter()
        .enumerate()
        .filter_map(|(index, leg)| (leg.helicity == -1).then_some(index))
        .collect::<Vec<_>>();
    let positive = legs
        .iter()
        .enumerate()
        .filter_map(|(index, leg)| (leg.helicity == 1).then_some(index))
        .collect::<Vec<_>>();
    let (kind, numerator_legs) = if negative.len() == 2 {
        (SpinorBracketKind::Angle, &negative)
    } else if positive.len() == 2 {
        (SpinorBracketKind::Square, &positive)
    } else {
        return Ok(None);
    };
    if legs.len() == 3 && required_three_point_branch != Some(kind) {
        // At the BCFW pole one of the two three-point complex branches is
        // identically singular. Do not construct its structural 0/0 form.
        return Ok(None);
    }

    let numerator_bracket = match kind {
        SpinorBracketKind::Angle => linear_weyl_bracket(
            builder,
            kind,
            &legs[numerator_legs[0]].undotted,
            &legs[numerator_legs[1]].undotted,
        )?,
        SpinorBracketKind::Square => linear_weyl_bracket(
            builder,
            kind,
            &legs[numerator_legs[0]].dotted,
            &legs[numerator_legs[1]].dotted,
        )?,
    };
    let numerator = builder.pow(numerator_bracket, 4)?;
    let mut denominator_factors = Vec::with_capacity(legs.len());
    for left in 0..legs.len() {
        let right = (left + 1) % legs.len();
        denominator_factors.push(match kind {
            SpinorBracketKind::Angle => {
                linear_weyl_bracket(builder, kind, &legs[left].undotted, &legs[right].undotted)?
            }
            SpinorBracketKind::Square => {
                linear_weyl_bracket(builder, kind, &legs[left].dotted, &legs[right].dotted)?
            }
        });
    }
    let denominator = builder.product(denominator_factors)?;
    let quotient = builder.quotient(numerator, denominator)?;
    let phase = if kind == SpinorBracketKind::Square && legs.len() % 2 == 1 {
        builder.negate(imaginary_unit)?
    } else {
        imaginary_unit
    };
    Ok(Some(builder.product([phase, quotient])?))
}

fn momentum_sum_mass_squared(
    builder: &mut SpinorDagBuilder,
    legs: &[u16],
) -> RusticolResult<SpinorNodeId> {
    let mut terms = Vec::new();
    for (left_index, left) in legs.iter().copied().enumerate() {
        for right in legs[(left_index + 1)..].iter().copied() {
            let angle = builder.angle(left, right)?;
            let square = builder.square(right, left)?;
            terms.push(builder.product([angle, square])?);
        }
    }
    builder.sum(terms)
}

fn momentum_sandwich(
    builder: &mut SpinorDagBuilder,
    undotted: u16,
    momentum_legs: &[u16],
    dotted: u16,
) -> RusticolResult<SpinorNodeId> {
    let mut terms = Vec::with_capacity(momentum_legs.len());
    for momentum_leg in momentum_legs {
        let angle = builder.angle(undotted, *momentum_leg)?;
        let square = builder.square(*momentum_leg, dotted)?;
        terms.push(builder.product([angle, square])?);
    }
    builder.sum(terms)
}

/// Lower one six-gluon NMHV root with an adjacent `[a,b>` shift, choosing a
/// negative `a` whose cyclic predecessor `b` is positive. Every factorization
/// child is at most five point and is therefore exactly Parke--Taylor. The raw
/// bispinor Berends--Giele graph remains the independent construction oracle.
fn six_gluon_nmhv_bcfw_root(
    builder: &mut SpinorDagBuilder,
    helicities: &[i8],
    imaginary_unit: SpinorNodeId,
) -> RusticolResult<SpinorNodeId> {
    if helicities.len() != 6
        || helicities
            .iter()
            .filter(|helicity| **helicity == -1)
            .count()
            != 3
    {
        return Err(invalid(
            "six-gluon BCFW lowering requires an NMHV helicity root",
        ));
    }
    let shift_start = (0..helicities.len())
        .find(|index| {
            helicities[*index] == -1
                && helicities[(*index + helicities.len() - 1) % helicities.len()] == 1
        })
        .ok_or_else(|| invalid("NMHV root has no positive-to-negative cyclic boundary"))?;
    let labels = (0..helicities.len())
        .map(|offset| ((shift_start + offset) % helicities.len()) as u16)
        .collect::<Vec<_>>();
    let a = labels[0];
    let b = labels[labels.len() - 1];
    let one = builder.one();
    let mut contributions = Vec::new();

    // Both factorization children need at least three legs, leaving precisely
    // the three adjacent channels below at six points.
    for split in 1..=(labels.len() - 3) {
        let momentum_legs = &labels[..=split];
        let mass_squared = momentum_sum_mass_squared(builder, momentum_legs)?;
        let sandwich = momentum_sandwich(builder, a, momentum_legs, b)?;
        let z = builder.quotient(mass_squared, sandwich)?;
        let minus_z = builder.negate(z)?;

        let mut shifted_a = OnShellSpinorLeg::external(a, helicities[usize::from(a)], one);
        let shifted_a_term =
            linear_weyl_scale(builder, minus_z, &LinearWeylExpression::atom(b, one))?;
        shifted_a.dotted = linear_weyl_sum(
            builder,
            [LinearWeylExpression::atom(a, one), shifted_a_term],
        )?;
        let mut shifted_b = OnShellSpinorLeg::external(b, helicities[usize::from(b)], one);
        let shifted_b_term = linear_weyl_scale(builder, z, &LinearWeylExpression::atom(a, one))?;
        shifted_b.undotted = linear_weyl_sum(
            builder,
            [LinearWeylExpression::atom(b, one), shifted_b_term],
        )?;

        // P_hat = |U>[V| with U=P|b] and V=<a|P/<a|P|b].
        let mut internal_undotted_terms = BTreeMap::new();
        let mut internal_dotted_numerator_terms = BTreeMap::new();
        for momentum_leg in momentum_legs {
            let undotted_coefficient = builder.square(*momentum_leg, b)?;
            if undotted_coefficient != builder.zero() {
                internal_undotted_terms.insert(*momentum_leg, undotted_coefficient);
            }
            let dotted_coefficient = builder.angle(a, *momentum_leg)?;
            if dotted_coefficient != builder.zero() {
                internal_dotted_numerator_terms.insert(*momentum_leg, dotted_coefficient);
            }
        }
        let internal_undotted = LinearWeylExpression {
            terms: internal_undotted_terms,
        };
        let inverse_sandwich = builder.reciprocal(sandwich)?;
        let internal_dotted = linear_weyl_scale(
            builder,
            inverse_sandwich,
            &LinearWeylExpression {
                terms: internal_dotted_numerator_terms,
            },
        )?;
        // Crossing to -P_hat multiplies both Weyl spinors by i in the
        // repository's negative-energy factorization convention.
        let negative_internal_undotted =
            linear_weyl_scale(builder, imaginary_unit, &internal_undotted)?;
        let negative_internal_dotted =
            linear_weyl_scale(builder, imaginary_unit, &internal_dotted)?;
        let inverse_mass_squared = builder.reciprocal(mass_squared)?;
        let propagator = builder.product([imaginary_unit, inverse_mass_squared])?;

        for internal_helicity in [-1_i8, 1_i8] {
            let mut left = Vec::with_capacity(split + 2);
            left.push(shifted_a.clone());
            for label in &labels[1..=split] {
                left.push(OnShellSpinorLeg::external(
                    *label,
                    helicities[usize::from(*label)],
                    one,
                ));
            }
            left.push(OnShellSpinorLeg {
                undotted: negative_internal_undotted.clone(),
                dotted: negative_internal_dotted.clone(),
                helicity: internal_helicity,
            });

            let mut right = Vec::with_capacity(labels.len() - split + 1);
            right.push(OnShellSpinorLeg {
                undotted: internal_undotted.clone(),
                dotted: internal_dotted.clone(),
                helicity: -internal_helicity,
            });
            for label in &labels[(split + 1)..(labels.len() - 1)] {
                right.push(OnShellSpinorLeg::external(
                    *label,
                    helicities[usize::from(*label)],
                    one,
                ));
            }
            right.push(shifted_b.clone());

            let left_branch = (left.len() == 3).then_some(SpinorBracketKind::Angle);
            let right_branch = (right.len() == 3).then_some(SpinorBracketKind::Square);
            let Some(left_amplitude) =
                parke_taylor_linear_spinors(builder, &left, left_branch, imaginary_unit)?
            else {
                continue;
            };
            let Some(right_amplitude) =
                parke_taylor_linear_spinors(builder, &right, right_branch, imaginary_unit)?
            else {
                continue;
            };
            contributions.push(builder.product([left_amplitude, propagator, right_amplitude])?);
        }
    }
    if contributions.is_empty() {
        return Err(invalid("six-gluon NMHV BCFW lowering produced no terms"));
    }
    builder.sum(contributions)
}

pub(crate) fn three_vector_bispinor_expression(
    builder: &mut SpinorDagBuilder,
    left: &BispinorExpression,
    left_momentum: &BispinorExpression,
    right: &BispinorExpression,
    right_momentum: &BispinorExpression,
) -> RusticolResult<BispinorExpression> {
    let two = builder.constant(exact_real(2, 1)?)?;
    let minus_two = builder.constant(exact_real(-2, 1)?)?;
    let left_dot_right = bispinor_dot_expression(builder, left, right)?;
    let left_dot_right_momentum = bispinor_dot_expression(builder, left, right_momentum)?;
    let right_dot_left_momentum = bispinor_dot_expression(builder, right, left_momentum)?;
    let minus_one = builder.constant(exact_real(-1, 1)?)?;
    let negative_right_momentum = bispinor_scale(builder, minus_one, right_momentum)?;
    let momentum_difference =
        bispinor_sum(builder, [left_momentum.clone(), negative_right_momentum])?;
    let momentum_term = bispinor_scale(builder, left_dot_right, &momentum_difference)?;
    let right_scale = builder.product([two, left_dot_right_momentum])?;
    let right_term = bispinor_scale(builder, right_scale, right)?;
    let left_scale = builder.product([minus_two, right_dot_left_momentum])?;
    let left_term = bispinor_scale(builder, left_scale, left)?;
    bispinor_sum(builder, [momentum_term, right_term, left_term])
}

fn four_vector_bispinor_expression(
    builder: &mut SpinorDagBuilder,
    left: &BispinorExpression,
    middle: &BispinorExpression,
    right: &BispinorExpression,
) -> RusticolResult<BispinorExpression> {
    let two = builder.constant(exact_real(2, 1)?)?;
    let minus_one = builder.constant(exact_real(-1, 1)?)?;
    let left_dot_right = bispinor_dot_expression(builder, left, right)?;
    let left_dot_middle = bispinor_dot_expression(builder, left, middle)?;
    let middle_dot_right = bispinor_dot_expression(builder, middle, right)?;
    let middle_scale = builder.product([two, left_dot_right])?;
    let middle_term = bispinor_scale(builder, middle_scale, middle)?;
    let left_scale = builder.product([minus_one, middle_dot_right])?;
    let left_term = bispinor_scale(builder, left_scale, left)?;
    let right_scale = builder.product([minus_one, left_dot_middle])?;
    let right_term = bispinor_scale(builder, right_scale, right)?;
    bispinor_sum(builder, [middle_term, left_term, right_term])
}

fn interval_lane<'a>(
    currents: &'a BTreeMap<(u16, u16), Box<[BispinorExpression]>>,
    start: u16,
    end: u16,
    mask: usize,
) -> RusticolResult<&'a BispinorExpression> {
    currents
        .get(&(start, end))
        .and_then(|lanes| lanes.get(mask))
        .ok_or_else(|| {
            invalid(format!(
                "missing helicity lane {mask} for interval {start}..={end}"
            ))
        })
}

fn build_ordered_gluon_current_table(
    builder: &mut SpinorDagBuilder,
    gluon_atoms: &[u16],
) -> RusticolResult<BTreeMap<(u16, u16), Box<[BispinorExpression]>>> {
    build_ordered_gluon_current_table_with_references(builder, gluon_atoms, None)
}

fn build_ordered_gluon_current_table_with_references(
    builder: &mut SpinorDagBuilder,
    gluon_atoms: &[u16],
    reference_atoms: Option<&[u16]>,
) -> RusticolResult<BTreeMap<(u16, u16), Box<[BispinorExpression]>>> {
    if gluon_atoms.is_empty() {
        return Err(invalid("a quark line requires at least one attached gluon"));
    }
    let gluon_count =
        u16::try_from(gluon_atoms.len()).map_err(|_| invalid("gluon count exceeds u16"))?;
    let mut momenta = BTreeMap::new();
    for start in 0..gluon_count {
        for end in start..gluon_count {
            momenta.insert(
                (start, end),
                ordered_momentum_expression(
                    &gluon_atoms[usize::from(start)..=usize::from(end)],
                    builder.one(),
                ),
            );
        }
    }

    let mut currents = BTreeMap::<(u16, u16), Box<[BispinorExpression]>>::new();
    if reference_atoms.is_some_and(|references| references.len() != gluon_atoms.len()) {
        return Err(invalid(
            "gluon reference count does not match the gluon count",
        ));
    }
    for (position, atom) in gluon_atoms.iter().copied().enumerate() {
        let polarizations = if let Some(references) = reference_atoms {
            vec![
                external_polarization_expression_with_reference(
                    builder,
                    atom,
                    references[position],
                    -1,
                )?,
                external_polarization_expression_with_reference(
                    builder,
                    atom,
                    references[position],
                    1,
                )?,
            ]
        } else {
            vec![
                external_polarization_expression(builder, atom, -1)?,
                external_polarization_expression(builder, atom, 1)?,
            ]
        };
        let position =
            u16::try_from(position).map_err(|_| invalid("gluon position exceeds u16"))?;
        currents.insert((position, position), polarizations.into_boxed_slice());
    }

    for length in 2..=gluon_count {
        for start in 0..=(gluon_count - length) {
            let end = start + length - 1;
            let lane_count = 1_usize
                .checked_shl(u32::from(length))
                .ok_or_else(|| invalid("gluon interval helicity-lane count overflows"))?;
            let mut numerators = Vec::with_capacity(lane_count);
            for mask in 0..lane_count {
                let mut contributions = Vec::new();
                for left_end in start..end {
                    let left_length = left_end - start + 1;
                    let left_lane_mask = mask & ((1_usize << left_length) - 1);
                    let right_lane_mask = mask >> left_length;
                    contributions.push(three_vector_bispinor_expression(
                        builder,
                        interval_lane(&currents, start, left_end, left_lane_mask)?,
                        &momenta[&(start, left_end)],
                        interval_lane(&currents, left_end + 1, end, right_lane_mask)?,
                        &momenta[&(left_end + 1, end)],
                    )?);
                }
                for left_end in start..(end - 1) {
                    for middle_end in (left_end + 1)..end {
                        let left_length = left_end - start + 1;
                        let middle_length = middle_end - left_end;
                        let left_lane_mask = mask & ((1_usize << left_length) - 1);
                        let middle_lane_mask =
                            (mask >> left_length) & ((1_usize << middle_length) - 1);
                        let right_lane_mask = mask >> (left_length + middle_length);
                        contributions.push(four_vector_bispinor_expression(
                            builder,
                            interval_lane(&currents, start, left_end, left_lane_mask)?,
                            interval_lane(&currents, left_end + 1, middle_end, middle_lane_mask)?,
                            interval_lane(&currents, middle_end + 1, end, right_lane_mask)?,
                        )?);
                    }
                }
                numerators.push(bispinor_sum(builder, contributions)?);
            }
            let mass_squared =
                ordered_mass_squared(builder, &gluon_atoms[usize::from(start)..=usize::from(end)])?;
            let propagator = builder.reciprocal(mass_squared)?;
            let propagated = numerators
                .iter()
                .map(|numerator| bispinor_scale(builder, propagator, numerator))
                .collect::<RusticolResult<Vec<_>>>()?;
            currents.insert((start, end), propagated.into_boxed_slice());
        }
    }
    Ok(currents)
}

fn quark_vector_weyl_numerator(
    builder: &mut SpinorDagBuilder,
    chirality: SpinorChirality,
    quark: &LinearWeylExpression,
    vector: &BispinorExpression,
    output_atoms: &[u16],
) -> RusticolResult<LinearWeylExpression> {
    let one = builder.one();
    let output_momentum = output_atoms
        .iter()
        .copied()
        .map(|atom| (atom, one))
        .collect::<Vec<_>>();
    quark_vector_weyl_numerator_with_momentum(builder, chirality, quark, vector, &output_momentum)
}

/// Apply a vector vertex followed by a massless Weyl numerator whose momentum
/// is an authenticated signed sum of external null momenta.
pub(crate) fn quark_vector_weyl_numerator_with_momentum(
    builder: &mut SpinorDagBuilder,
    chirality: SpinorChirality,
    quark: &LinearWeylExpression,
    vector: &BispinorExpression,
    output_momentum: &[(u16, SpinorNodeId)],
) -> RusticolResult<LinearWeylExpression> {
    let mut output = BTreeMap::new();
    for (output_atom, momentum_coefficient) in output_momentum.iter().copied() {
        let mut terms = Vec::new();
        for (quark_atom, quark_coefficient) in &quark.terms {
            for (dyad, vector_coefficient) in &vector.terms {
                let (first, second) = match chirality {
                    SpinorChirality::Positive => (
                        builder.square(*quark_atom, dyad.dotted)?,
                        builder.angle(dyad.undotted, output_atom)?,
                    ),
                    SpinorChirality::Negative => (
                        builder.angle(*quark_atom, dyad.undotted)?,
                        builder.square(dyad.dotted, output_atom)?,
                    ),
                };
                terms.push(builder.product([
                    momentum_coefficient,
                    *quark_coefficient,
                    *vector_coefficient,
                    first,
                    second,
                ])?);
            }
        }
        let coefficient = builder.sum(terms)?;
        let coefficient = builder.simplify_schouten(coefficient)?;
        if coefficient != builder.zero() {
            output.insert(output_atom, coefficient);
        }
    }
    Ok(LinearWeylExpression { terms: output })
}

fn quark_vector_weyl_closure(
    builder: &mut SpinorDagBuilder,
    chirality: SpinorChirality,
    quark: &LinearWeylExpression,
    vector: &BispinorExpression,
    antiquark_atom: u16,
) -> RusticolResult<SpinorNodeId> {
    let mut terms = Vec::new();
    for (quark_atom, quark_coefficient) in &quark.terms {
        for (dyad, vector_coefficient) in &vector.terms {
            let (first, second) = match chirality {
                SpinorChirality::Positive => (
                    builder.square(*quark_atom, dyad.dotted)?,
                    builder.angle(dyad.undotted, antiquark_atom)?,
                ),
                SpinorChirality::Negative => (
                    builder.angle(*quark_atom, dyad.undotted)?,
                    builder.square(dyad.dotted, antiquark_atom)?,
                ),
            };
            terms.push(builder.product([
                *quark_coefficient,
                *vector_coefficient,
                first,
                second,
            ])?);
        }
    }
    let result = builder.sum(terms)?;
    builder.simplify_schouten(result)
}

/// A full Dirac current retained as its two Weyl halves. `undotted` is the
/// upper two-component spinor and `dotted` the lower one in the built-in
/// model's chiral Dirac basis.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct DiracExpression {
    undotted: LinearWeylExpression,
    dotted: LinearWeylExpression,
}

fn weyl_vector_action(
    builder: &mut SpinorDagBuilder,
    chirality: SpinorChirality,
    spinor: &LinearWeylExpression,
    vector: &BispinorExpression,
) -> RusticolResult<LinearWeylExpression> {
    let mut terms = BTreeMap::<u16, Vec<SpinorNodeId>>::new();
    for (spinor_atom, spinor_coefficient) in &spinor.terms {
        for (dyad, vector_coefficient) in &vector.terms {
            let (output_atom, contraction) = match chirality {
                SpinorChirality::Positive => {
                    (dyad.undotted, builder.square(*spinor_atom, dyad.dotted)?)
                }
                SpinorChirality::Negative => {
                    (dyad.dotted, builder.angle(*spinor_atom, dyad.undotted)?)
                }
            };
            terms.entry(output_atom).or_default().push(builder.product([
                *spinor_coefficient,
                *vector_coefficient,
                contraction,
            ])?);
        }
    }
    let mut output = BTreeMap::new();
    for (atom, coefficients) in terms {
        let coefficient = builder.sum(coefficients)?;
        let coefficient = builder.simplify_schouten(coefficient)?;
        if coefficient != builder.zero() {
            output.insert(atom, coefficient);
        }
    }
    Ok(LinearWeylExpression { terms: output })
}

pub(crate) fn quark_vector_weyl_bilinear(
    builder: &mut SpinorDagBuilder,
    chirality: SpinorChirality,
    quark: &LinearWeylExpression,
    vector: &BispinorExpression,
    antiquark: &LinearWeylExpression,
) -> RusticolResult<SpinorNodeId> {
    let mut terms = Vec::new();
    for (quark_atom, quark_coefficient) in &quark.terms {
        for (dyad, vector_coefficient) in &vector.terms {
            for (antiquark_atom, antiquark_coefficient) in &antiquark.terms {
                let (first, second) = match chirality {
                    SpinorChirality::Positive => (
                        builder.square(*quark_atom, dyad.dotted)?,
                        builder.angle(dyad.undotted, *antiquark_atom)?,
                    ),
                    SpinorChirality::Negative => (
                        builder.angle(*quark_atom, dyad.undotted)?,
                        builder.square(dyad.dotted, *antiquark_atom)?,
                    ),
                };
                terms.push(builder.product([
                    *quark_coefficient,
                    *vector_coefficient,
                    *antiquark_coefficient,
                    first,
                    second,
                ])?);
            }
        }
    }
    let result = builder.sum(terms)?;
    builder.simplify_schouten(result)
}

fn massive_dirac_spin_states(
    builder: &mut SpinorDagBuilder,
    source: u16,
    mass: SpinorNodeId,
) -> RusticolResult<[DiracExpression; 2]> {
    let (k, r) = builder.massive_vector_atoms(source)?;
    let inverse_mass = builder.reciprocal(mass)?;
    let r_k = builder.angle(r, k)?;
    let k_r = builder.angle(k, r)?;
    let negative_r_k = builder.negate(r_k)?;
    let negative_k_r = builder.negate(k_r)?;
    let first_dotted = builder.product([negative_r_k, inverse_mass])?;
    let second_dotted = builder.product([negative_k_r, inverse_mass])?;
    Ok([
        DiracExpression {
            undotted: LinearWeylExpression::atom(k, builder.one()),
            dotted: LinearWeylExpression::atom(r, first_dotted),
        },
        DiracExpression {
            undotted: LinearWeylExpression::atom(r, builder.one()),
            dotted: LinearWeylExpression::atom(k, second_dotted),
        },
    ])
}

fn massive_dirac_vector_numerator(
    builder: &mut SpinorDagBuilder,
    quark: &DiracExpression,
    vector: &BispinorExpression,
    output_atoms: &[u16],
    mass: SpinorNodeId,
) -> RusticolResult<DiracExpression> {
    let undotted_momentum = quark_vector_weyl_numerator(
        builder,
        SpinorChirality::Negative,
        &quark.undotted,
        vector,
        output_atoms,
    )?;
    let undotted_mass =
        weyl_vector_action(builder, SpinorChirality::Positive, &quark.dotted, vector)?;
    let dotted_momentum = quark_vector_weyl_numerator(
        builder,
        SpinorChirality::Positive,
        &quark.dotted,
        vector,
        output_atoms,
    )?;
    let dotted_mass =
        weyl_vector_action(builder, SpinorChirality::Negative, &quark.undotted, vector)?;
    let undotted_mass = linear_weyl_scale(builder, mass, &undotted_mass)?;
    let dotted_mass = linear_weyl_scale(builder, mass, &dotted_mass)?;
    let undotted = linear_weyl_sum(builder, [undotted_momentum, undotted_mass])?;
    let dotted = linear_weyl_sum(builder, [dotted_momentum, dotted_mass])?;
    Ok(DiracExpression { undotted, dotted })
}

fn massive_dirac_scale(
    builder: &mut SpinorDagBuilder,
    scalar: SpinorNodeId,
    current: &DiracExpression,
) -> RusticolResult<DiracExpression> {
    Ok(DiracExpression {
        undotted: linear_weyl_scale(builder, scalar, &current.undotted)?,
        dotted: linear_weyl_scale(builder, scalar, &current.dotted)?,
    })
}

fn massive_dirac_vector_closure(
    builder: &mut SpinorDagBuilder,
    quark: &DiracExpression,
    vector: &BispinorExpression,
    antiquark: &DiracExpression,
) -> RusticolResult<SpinorNodeId> {
    let positive = quark_vector_weyl_bilinear(
        builder,
        SpinorChirality::Positive,
        &quark.dotted,
        vector,
        &antiquark.undotted,
    )?;
    let negative = quark_vector_weyl_bilinear(
        builder,
        SpinorChirality::Negative,
        &quark.undotted,
        vector,
        &antiquark.dotted,
    )?;
    builder.sum([positive, negative])
}

fn massive_quark_propagator_denominator(
    builder: &mut SpinorDagBuilder,
    support: &[u16],
    mass: SpinorNodeId,
    width: SpinorNodeId,
) -> RusticolResult<SpinorNodeId> {
    let momentum_squared = ordered_mass_squared(builder, support)?;
    let mass_squared = builder.product([mass, mass])?;
    let negative_mass_squared = builder.negate(mass_squared)?;
    let imaginary_unit = builder.constant(ExactComplexRational::new(
        ExactRational::ZERO,
        ExactRational::ONE,
    ))?;
    let width_term = builder.product([imaginary_unit, mass, width])?;
    builder.sum([momentum_squared, negative_mass_squared, width_term])
}

fn build_quark_prefix_current_table(
    builder: &mut SpinorDagBuilder,
    chirality: SpinorChirality,
    quark_atom: u16,
    gluon_atoms: &[u16],
    gluon_currents: &BTreeMap<(u16, u16), Box<[BispinorExpression]>>,
) -> RusticolResult<BTreeMap<u16, Box<[LinearWeylExpression]>>> {
    let mut currents = BTreeMap::from([(
        0_u16,
        vec![LinearWeylExpression::atom(quark_atom, builder.one())].into_boxed_slice(),
    )]);
    let gluon_count =
        u16::try_from(gluon_atoms.len()).map_err(|_| invalid("gluon count exceeds u16"))?;
    for prefix_count in 1..gluon_count {
        let lane_count = 1_usize
            .checked_shl(u32::from(prefix_count))
            .ok_or_else(|| invalid("quark-current helicity-lane count overflows"))?;
        let mut lanes = Vec::with_capacity(lane_count);
        let support = std::iter::once(quark_atom)
            .chain(gluon_atoms[..usize::from(prefix_count)].iter().copied())
            .collect::<Vec<_>>();
        let mass_squared = ordered_mass_squared(builder, &support)?;
        let propagator = builder.reciprocal(mass_squared)?;
        let negative_propagator = builder.negate(propagator)?;
        for mask in 0..lane_count {
            let mut contributions = Vec::new();
            for left_prefix_count in 0..prefix_count {
                let left_lane_mask = if left_prefix_count == 0 {
                    0
                } else {
                    mask & ((1_usize << left_prefix_count) - 1)
                };
                let quark = currents
                    .get(&left_prefix_count)
                    .and_then(|entries| entries.get(left_lane_mask))
                    .ok_or_else(|| {
                        invalid(format!(
                            "missing quark helicity lane {left_lane_mask} for prefix {left_prefix_count}"
                        ))
                    })?;
                let gluon_start = left_prefix_count;
                let gluon_end = prefix_count - 1;
                let gluon_lane_mask = mask >> left_prefix_count;
                let vector =
                    interval_lane(gluon_currents, gluon_start, gluon_end, gluon_lane_mask)?;
                contributions.push(quark_vector_weyl_numerator(
                    builder, chirality, quark, vector, &support,
                )?);
            }
            let numerator = linear_weyl_sum(builder, contributions)?;
            lanes.push(linear_weyl_scale(builder, negative_propagator, &numerator)?);
        }
        currents.insert(prefix_count, lanes.into_boxed_slice());
    }
    Ok(currents)
}

fn quark_gluon_root(
    builder: &mut SpinorDagBuilder,
    chirality: SpinorChirality,
    gluon_mask: usize,
    gluon_atoms: &[u16],
    antiquark_atom: u16,
    gluon_currents: &BTreeMap<(u16, u16), Box<[BispinorExpression]>>,
    quark_currents: &BTreeMap<u16, Box<[LinearWeylExpression]>>,
    imaginary_unit: SpinorNodeId,
) -> RusticolResult<SpinorNodeId> {
    let gluon_count =
        u16::try_from(gluon_atoms.len()).map_err(|_| invalid("gluon count exceeds u16"))?;
    let mut contributions = Vec::new();
    for prefix_count in 0..gluon_count {
        let quark_lane_mask = if prefix_count == 0 {
            0
        } else {
            gluon_mask & ((1_usize << prefix_count) - 1)
        };
        let quark = quark_currents
            .get(&prefix_count)
            .and_then(|entries| entries.get(quark_lane_mask))
            .ok_or_else(|| {
                invalid(format!(
                    "missing root quark lane {quark_lane_mask} for prefix {prefix_count}"
                ))
            })?;
        let vector = interval_lane(
            gluon_currents,
            prefix_count,
            gluon_count - 1,
            gluon_mask >> prefix_count,
        )?;
        contributions.push(quark_vector_weyl_closure(
            builder,
            chirality,
            quark,
            vector,
            antiquark_atom,
        )?);
    }
    let numerator = builder.sum(contributions)?;
    let numerator = builder.simplify_schouten(numerator)?;
    builder.product([imaginary_unit, numerator])
}

/// Build the coupling-stripped, fixed-colour, always-helicity-summed tree
/// graph for one massive Dirac fundamental line and exactly two ordered
/// massless gluons. `ordered_source_slots` is the graph traversal
/// `[quark, gluon_1, gluon_2, antiquark]` and must be a permutation of the four
/// physical momentum slots. Runtime parameter order is `[mass, width]`; the
/// internal fermion denominator is `P^2-mass^2+i*mass*width`.
/// Each gluon uses the other as its polarization reference. For the physical
/// back-to-back incoming pair this is the spinor form of the temporal gauge
/// used by the retained component recurrence, including at nonzero width.
pub fn build_helicity_summed_massive_quark_two_gluon_spinor_dag(
    ordered_source_slots: &[u16],
) -> RusticolResult<SpinorDag> {
    if ordered_source_slots.len() != 4 {
        return Err(invalid(
            "the massive-quark slice requires [quark, gluon, gluon, antiquark]",
        ));
    }
    if ordered_source_slots
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        != (0..4).collect()
    {
        return Err(invalid(
            "the massive-quark open string must contain every source exactly once",
        ));
    }
    let quark_source = ordered_source_slots[0];
    let gluon_atoms = [ordered_source_slots[1], ordered_source_slots[2]];
    let antiquark_source = ordered_source_slots[3];
    let mut builder =
        SpinorDagBuilder::new_with_massive_fermion_pair(4, quark_source, antiquark_source, 2)?;
    let mass = builder.parameter(0)?;
    let width = builder.parameter(1)?;
    let imaginary_unit = builder.constant(ExactComplexRational::new(
        ExactRational::ZERO,
        ExactRational::ONE,
    ))?;
    let quark_states = massive_dirac_spin_states(&mut builder, quark_source, mass)?;
    let antiquark_states = massive_dirac_spin_states(&mut builder, antiquark_source, mass)?;
    let gluon_references = [gluon_atoms[1], gluon_atoms[0]];
    let gluon_currents = build_ordered_gluon_current_table_with_references(
        &mut builder,
        &gluon_atoms,
        Some(&gluon_references),
    )?;
    let (quark_k, quark_r) = builder.massive_vector_atoms(quark_source)?;
    let propagator_support = [quark_k, quark_r, gluon_atoms[0]];
    let denominator =
        massive_quark_propagator_denominator(&mut builder, &propagator_support, mass, width)?;
    let propagator = builder.reciprocal(denominator)?;
    let negative_propagator = builder.negate(propagator)?;

    for gluon_mask in 0..4_usize {
        let first_vector = interval_lane(&gluon_currents, 0, 0, gluon_mask & 1)?;
        let second_vector = interval_lane(&gluon_currents, 1, 1, gluon_mask >> 1)?;
        let combined_vector = interval_lane(&gluon_currents, 0, 1, gluon_mask)?;
        for (quark_index, quark) in quark_states.iter().enumerate() {
            let numerator = massive_dirac_vector_numerator(
                &mut builder,
                quark,
                first_vector,
                &propagator_support,
                mass,
            )?;
            let propagated = massive_dirac_scale(&mut builder, negative_propagator, &numerator)?;
            for (antiquark_index, antiquark) in antiquark_states.iter().enumerate() {
                let combined =
                    massive_dirac_vector_closure(&mut builder, quark, combined_vector, antiquark)?;
                let sequential = massive_dirac_vector_closure(
                    &mut builder,
                    &propagated,
                    second_vector,
                    antiquark,
                )?;
                let numerator = builder.sum([combined, sequential])?;
                let amplitude = builder.product([imaginary_unit, numerator])?;
                let mut helicities = vec![0_i8; 4];
                helicities[usize::from(quark_source)] = if quark_index == 0 { -1 } else { 1 };
                helicities[usize::from(gluon_atoms[0])] = if gluon_mask & 1 == 0 { -1 } else { 1 };
                helicities[usize::from(gluon_atoms[1])] = if gluon_mask & 2 == 0 { -1 } else { 1 };
                helicities[usize::from(antiquark_source)] =
                    if antiquark_index == 0 { -1 } else { 1 };
                builder.add_root(helicities.into_boxed_slice(), amplitude)?;
            }
        }
    }
    builder.finish()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QuarkZVectorItem {
    Gluon { position: u16, atom: u16 },
    Z,
}

fn quark_z_vector_interval(
    gluon_currents: &BTreeMap<(u16, u16), Box<[BispinorExpression]>>,
    z_polarization: &BispinorExpression,
    items: &[QuarkZVectorItem],
    start: usize,
    end: usize,
    gluon_mask: usize,
) -> RusticolResult<Option<BispinorExpression>> {
    let interval = &items[start..=end];
    if interval == [QuarkZVectorItem::Z] {
        return Ok(Some(z_polarization.clone()));
    }
    let mut gluon_positions = Vec::with_capacity(interval.len());
    for item in interval {
        let QuarkZVectorItem::Gluon { position, .. } = item else {
            return Ok(None);
        };
        gluon_positions.push(*position);
    }
    let Some(first) = gluon_positions.first().copied() else {
        return Ok(None);
    };
    let last = *gluon_positions
        .last()
        .ok_or_else(|| invalid("empty gluon interval"))?;
    if gluon_positions.iter().copied().ne(first..=last) {
        return Err(invalid("q-Z traversal disrupted gluon ordering"));
    }
    let length = u32::from(last - first + 1);
    let local_mask = (gluon_mask >> first) & ((1_usize << length) - 1);
    Ok(Some(
        interval_lane(gluon_currents, first, last, local_mask)?.clone(),
    ))
}

fn quark_z_prefix_spinor_support(
    builder: &SpinorDagBuilder,
    quark_atom: u16,
    z_source: u16,
    prefix: &[QuarkZVectorItem],
) -> RusticolResult<Vec<u16>> {
    let mut support = vec![quark_atom];
    for item in prefix {
        match item {
            QuarkZVectorItem::Gluon { atom, .. } => support.push(*atom),
            QuarkZVectorItem::Z => {
                let (k, r) = builder.massive_vector_atoms(z_source)?;
                support.extend([k, r]);
            }
        }
    }
    Ok(support)
}

fn quark_z_prefix_mass_squared(
    builder: &mut SpinorDagBuilder,
    quark_atom: u16,
    antiquark_atom: u16,
    items: &[QuarkZVectorItem],
    prefix_count: usize,
) -> RusticolResult<SpinorNodeId> {
    let prefix = &items[..prefix_count];
    if prefix.contains(&QuarkZVectorItem::Z) {
        // Momentum conservation rewrites the massive prefix into the
        // complementary, entirely massless suffix plus the antiquark.  This
        // keeps the propagator an exact bracket polynomial.
        let complement = items[prefix_count..]
            .iter()
            .filter_map(|item| match item {
                QuarkZVectorItem::Gluon { atom, .. } => Some(*atom),
                QuarkZVectorItem::Z => None,
            })
            .chain([antiquark_atom])
            .collect::<Vec<_>>();
        return ordered_mass_squared(builder, &complement);
    }
    let support = std::iter::once(quark_atom)
        .chain(prefix.iter().filter_map(|item| match item {
            QuarkZVectorItem::Gluon { atom, .. } => Some(*atom),
            QuarkZVectorItem::Z => None,
        }))
        .collect::<Vec<_>>();
    ordered_mass_squared(builder, &support)
}

fn quark_z_ordered_root(
    builder: &mut SpinorDagBuilder,
    chirality: SpinorChirality,
    quark_atom: u16,
    antiquark_atom: u16,
    z_source: u16,
    items: &[QuarkZVectorItem],
    gluon_mask: usize,
    gluon_currents: &BTreeMap<(u16, u16), Box<[BispinorExpression]>>,
    z_polarization: &BispinorExpression,
    imaginary_unit: SpinorNodeId,
) -> RusticolResult<SpinorNodeId> {
    let mut quark_currents = vec![LinearWeylExpression::atom(quark_atom, builder.one())];
    for prefix_count in 1..items.len() {
        let mut contributions = Vec::new();
        for left_prefix_count in 0..prefix_count {
            let Some(vector) = quark_z_vector_interval(
                gluon_currents,
                z_polarization,
                items,
                left_prefix_count,
                prefix_count - 1,
                gluon_mask,
            )?
            else {
                continue;
            };
            let output_atoms = quark_z_prefix_spinor_support(
                builder,
                quark_atom,
                z_source,
                &items[..prefix_count],
            )?;
            contributions.push(quark_vector_weyl_numerator(
                builder,
                chirality,
                &quark_currents[left_prefix_count],
                &vector,
                &output_atoms,
            )?);
        }
        if contributions.is_empty() {
            return Err(invalid("q-Z ordered current has no valid attachment"));
        }
        let numerator = linear_weyl_sum(builder, contributions)?;
        let mass_squared =
            quark_z_prefix_mass_squared(builder, quark_atom, antiquark_atom, items, prefix_count)?;
        let propagator = builder.reciprocal(mass_squared)?;
        let negative_propagator = builder.negate(propagator)?;
        quark_currents.push(linear_weyl_scale(builder, negative_propagator, &numerator)?);
    }

    let mut root_contributions = Vec::new();
    for prefix_count in 0..items.len() {
        let Some(vector) = quark_z_vector_interval(
            gluon_currents,
            z_polarization,
            items,
            prefix_count,
            items.len() - 1,
            gluon_mask,
        )?
        else {
            continue;
        };
        root_contributions.push(quark_vector_weyl_closure(
            builder,
            chirality,
            &quark_currents[prefix_count],
            &vector,
            antiquark_atom,
        )?);
    }
    if root_contributions.is_empty() {
        return Err(invalid("q-Z ordered root has no valid attachment"));
    }
    let numerator = builder.sum(root_contributions)?;
    let numerator = builder.simplify_schouten(numerator)?;
    builder.product([imaginary_unit, numerator])
}

/// Build the fixed-colour, always-helicity-summed tree graph for one massless
/// open quark line, one on-shell massive neutral vector, and zero to two
/// ordered gluons. `ordered_colored_source_slots` is `q, gluons..., qbar`;
/// `z_source_slot` is the remaining physical momentum slot. Runtime parameter
/// order is exactly the repository's local `[g_left, g_right]` vector-current
/// coupling (whose wavefunction convention is `epsilon/sqrt(2)`). The graph
/// sums every insertion of the colour-singlet vector along the ordered quark
/// line while retaining its three physical polarizations as separate roots.
pub fn build_helicity_summed_quark_z_gluon_spinor_dag(
    ordered_colored_source_slots: &[u16],
    z_source_slot: u16,
) -> RusticolResult<SpinorDag> {
    if !(2..=4).contains(&ordered_colored_source_slots.len()) {
        return Err(invalid(
            "the q-Z-gluon slice supports an open quark line with zero to two gluons",
        ));
    }
    let external_count = u16::try_from(ordered_colored_source_slots.len() + 1)
        .map_err(|_| invalid("external count exceeds u16"))?;
    let observed = ordered_colored_source_slots
        .iter()
        .copied()
        .chain([z_source_slot])
        .collect::<BTreeSet<_>>();
    if observed != (0..external_count).collect() {
        return Err(invalid(
            "the q-Z open string and Z slot must contain every source exactly once",
        ));
    }
    let quark_atom = ordered_colored_source_slots[0];
    let antiquark_atom = *ordered_colored_source_slots
        .last()
        .ok_or_else(|| invalid("q-Z open string is empty"))?;
    let gluon_atoms = &ordered_colored_source_slots[1..ordered_colored_source_slots.len() - 1];
    let mut builder = SpinorDagBuilder::new_with_massive_vector(external_count, z_source_slot, 2)?;
    let imaginary_unit = builder.constant(ExactComplexRational::new(
        ExactRational::ZERO,
        ExactRational::ONE,
    ))?;
    let gluon_currents = if gluon_atoms.is_empty() {
        BTreeMap::new()
    } else {
        build_ordered_gluon_current_table(&mut builder, gluon_atoms)?
    };
    let gluon_configuration_count = 1_usize
        .checked_shl(
            u32::try_from(gluon_atoms.len()).map_err(|_| invalid("gluon count exceeds u32"))?,
        )
        .ok_or_else(|| invalid("gluon helicity configuration count overflows"))?;

    for quark_helicity in [-1_i8, 1_i8] {
        let (chirality, coupling_parameter) = if quark_helicity == 1 {
            (SpinorChirality::Positive, 1_u16)
        } else {
            (SpinorChirality::Negative, 0_u16)
        };
        let coupling = builder.parameter(coupling_parameter)?;
        for gluon_mask in 0..gluon_configuration_count {
            for z_helicity in [-1_i8, 0_i8, 1_i8] {
                let z_polarization = massive_vector_polarization_expression(
                    &mut builder,
                    z_source_slot,
                    z_helicity,
                )?;
                let mut insertion_amplitudes = Vec::with_capacity(gluon_atoms.len() + 1);
                for insertion in 0..=gluon_atoms.len() {
                    let mut items = Vec::with_capacity(gluon_atoms.len() + 1);
                    for (position, atom) in gluon_atoms.iter().copied().enumerate() {
                        if position == insertion {
                            items.push(QuarkZVectorItem::Z);
                        }
                        items.push(QuarkZVectorItem::Gluon {
                            position: u16::try_from(position)
                                .map_err(|_| invalid("gluon position exceeds u16"))?,
                            atom,
                        });
                    }
                    if insertion == gluon_atoms.len() {
                        items.push(QuarkZVectorItem::Z);
                    }
                    insertion_amplitudes.push(quark_z_ordered_root(
                        &mut builder,
                        chirality,
                        quark_atom,
                        antiquark_atom,
                        z_source_slot,
                        &items,
                        gluon_mask,
                        &gluon_currents,
                        &z_polarization,
                        imaginary_unit,
                    )?);
                }
                let amplitude = builder.sum(insertion_amplitudes)?;
                let amplitude = builder.product([coupling, amplitude])?;
                for antiquark_helicity in [-1_i8, 1_i8] {
                    let mut helicities = vec![-1_i8; usize::from(external_count)];
                    helicities[usize::from(quark_atom)] = quark_helicity;
                    helicities[usize::from(antiquark_atom)] = antiquark_helicity;
                    helicities[usize::from(z_source_slot)] = z_helicity;
                    for (position, atom) in gluon_atoms.iter().copied().enumerate() {
                        helicities[usize::from(atom)] = if gluon_mask & (1_usize << position) == 0 {
                            -1
                        } else {
                            1
                        };
                    }
                    builder.add_root(
                        helicities.into_boxed_slice(),
                        if antiquark_helicity == -quark_helicity {
                            amplitude
                        } else {
                            builder.zero()
                        },
                    )?;
                }
            }
        }
    }
    builder.finish()
}

/// Construct the complete fixed-order massless QCD graph for one open quark
/// line. `ordered_source_slots` gives the spinor-atom order
/// `q, gluons..., qbar` relative to the momentum array supplied at evaluation.
/// A runtime that first arranges momenta in LC open-string order therefore
/// passes the identity permutation here.
pub fn build_helicity_summed_quark_gluon_bg_spinor_dag(
    ordered_source_slots: &[u16],
) -> RusticolResult<SpinorDag> {
    let external_count = u16::try_from(ordered_source_slots.len())
        .map_err(|_| invalid("external count exceeds u16"))?;
    if !(4..=6).contains(&external_count) {
        return Err(invalid(
            "the massless quark-gluon slice currently supports four to six external particles",
        ));
    }
    let observed = ordered_source_slots
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if observed != (0..external_count).collect() {
        return Err(invalid(
            "the quark-gluon open string must contain every source slot exactly once",
        ));
    }
    let quark_atom = ordered_source_slots[0];
    let antiquark_atom = ordered_source_slots[usize::from(external_count - 1)];
    let gluon_atoms = &ordered_source_slots[1..usize::from(external_count - 1)];
    let mut builder = SpinorDagBuilder::new(external_count)?;
    let imaginary_unit = builder.constant(ExactComplexRational::new(
        ExactRational::ZERO,
        ExactRational::ONE,
    ))?;
    let gluon_currents = build_ordered_gluon_current_table(&mut builder, gluon_atoms)?;
    let positive_currents = build_quark_prefix_current_table(
        &mut builder,
        SpinorChirality::Positive,
        quark_atom,
        gluon_atoms,
        &gluon_currents,
    )?;
    let negative_currents = build_quark_prefix_current_table(
        &mut builder,
        SpinorChirality::Negative,
        quark_atom,
        gluon_atoms,
        &gluon_currents,
    )?;
    let gluon_configuration_count = 1_usize
        .checked_shl(
            u32::try_from(gluon_atoms.len()).map_err(|_| invalid("gluon count exceeds u32"))?,
        )
        .ok_or_else(|| invalid("gluon helicity configuration count overflows"))?;

    for quark_helicity in [-1_i8, 1_i8] {
        let chirality = if quark_helicity == 1 {
            SpinorChirality::Positive
        } else {
            SpinorChirality::Negative
        };
        let quark_currents = if chirality == SpinorChirality::Positive {
            &positive_currents
        } else {
            &negative_currents
        };
        for gluon_mask in 0..gluon_configuration_count {
            let amplitude = quark_gluon_root(
                &mut builder,
                chirality,
                gluon_mask,
                gluon_atoms,
                antiquark_atom,
                &gluon_currents,
                quark_currents,
                imaginary_unit,
            )?;
            for antiquark_helicity in [-1_i8, 1_i8] {
                let mut helicities = vec![-1_i8; usize::from(external_count)];
                helicities[usize::from(quark_atom)] = quark_helicity;
                helicities[usize::from(antiquark_atom)] = antiquark_helicity;
                for (position, atom) in gluon_atoms.iter().copied().enumerate() {
                    helicities[usize::from(atom)] = if gluon_mask & (1_usize << position) == 0 {
                        -1
                    } else {
                        1
                    };
                }
                builder.add_root(
                    helicities.into_boxed_slice(),
                    if antiquark_helicity == -quark_helicity {
                        amplitude
                    } else {
                        builder.zero()
                    },
                )?;
            }
        }
    }
    builder.finish()
}

/// Construct one color-ordered Berends--Giele graph in a sparse bispinor
/// representation. The interval topology is shared by every helicity lane;
/// helicities remain distinct only at sources and amplitude roots.
pub fn build_gluon_bg_spinor_dag(external_count: u16) -> RusticolResult<SpinorDag> {
    build_gluon_bg_spinor_dag_impl(external_count, false)
}

/// Construct the always-helicity-summed pure-gluon graph. Global helicity
/// flips have equal norms for real tree-level Yang--Mills momenta, so one root
/// represents each parity pair with multiplicity two.
pub fn build_helicity_summed_gluon_bg_spinor_dag(external_count: u16) -> RusticolResult<SpinorDag> {
    build_gluon_bg_spinor_dag_impl(external_count, true)
}

/// Construct the compact always-helicity-summed graph directly. Four- and
/// five-point roots use Parke--Taylor expressions; six-point NMHV roots use
/// one adjacent BCFW step. The raw off-shell recurrence remains available as
/// an independent oracle, but is not built and discarded on every load.
pub fn build_optimized_helicity_summed_gluon_spinor_dag(
    external_count: u16,
) -> RusticolResult<SpinorDag> {
    build_compact_gluon_tree_spinor_dag_impl(external_count, true)
}

fn build_gluon_bg_spinor_dag_impl(
    external_count: u16,
    helicity_flip_reduction: bool,
) -> RusticolResult<SpinorDag> {
    if !(4..=6).contains(&external_count) {
        return Err(invalid(
            "the bispinor Berends-Giele slice currently supports 4, 5, or 6 gluons",
        ));
    }
    let mut builder = SpinorDagBuilder::new(external_count)?;
    let complement_count = external_count - 1;
    let mut momenta = BTreeMap::new();
    for start in 0..complement_count {
        for end in start..complement_count {
            momenta.insert(
                (start, end),
                interval_momentum_expression(start, end, builder.one()),
            );
        }
    }

    let mut currents = BTreeMap::<(u16, u16), Box<[BispinorExpression]>>::new();
    for leg in 0..complement_count {
        currents.insert(
            (leg, leg),
            vec![
                external_polarization_expression(&mut builder, leg, -1)?,
                external_polarization_expression(&mut builder, leg, 1)?,
            ]
            .into_boxed_slice(),
        );
    }

    let mut root_numerators = None;
    for length in 2..=complement_count {
        for start in 0..=(complement_count - length) {
            let end = start + length - 1;
            let lane_count = 1_usize
                .checked_shl(u32::from(length))
                .ok_or_else(|| invalid("interval helicity-lane count overflows"))?;
            let mut numerators = Vec::with_capacity(lane_count);
            for mask in 0..lane_count {
                let mut contributions = Vec::new();
                for left_end in start..end {
                    let left_length = left_end - start + 1;
                    let left_lane_mask = mask & ((1_usize << left_length) - 1);
                    let right_lane_mask = mask >> left_length;
                    contributions.push(three_vector_bispinor_expression(
                        &mut builder,
                        interval_lane(&currents, start, left_end, left_lane_mask)?,
                        &momenta[&(start, left_end)],
                        interval_lane(&currents, left_end + 1, end, right_lane_mask)?,
                        &momenta[&(left_end + 1, end)],
                    )?);
                }
                for left_end in start..(end - 1) {
                    for middle_end in (left_end + 1)..end {
                        let left_length = left_end - start + 1;
                        let middle_length = middle_end - left_end;
                        let left_lane_mask = mask & ((1_usize << left_length) - 1);
                        let middle_lane_mask =
                            (mask >> left_length) & ((1_usize << middle_length) - 1);
                        let right_lane_mask = mask >> (left_length + middle_length);
                        contributions.push(four_vector_bispinor_expression(
                            &mut builder,
                            interval_lane(&currents, start, left_end, left_lane_mask)?,
                            interval_lane(&currents, left_end + 1, middle_end, middle_lane_mask)?,
                            interval_lane(&currents, middle_end + 1, end, right_lane_mask)?,
                        )?);
                    }
                }
                numerators.push(bispinor_sum(&mut builder, contributions)?);
            }
            if start == 0 && end + 1 == complement_count {
                root_numerators = Some(numerators.into_boxed_slice());
                continue;
            }
            let mass_squared = interval_mass_squared(&mut builder, start, end)?;
            let propagator = builder.reciprocal(mass_squared)?;
            let propagated = numerators
                .iter()
                .map(|numerator| bispinor_scale(&mut builder, propagator, numerator))
                .collect::<RusticolResult<Vec<_>>>()?;
            currents.insert((start, end), propagated.into_boxed_slice());
        }
    }

    let root_numerators = root_numerators.ok_or_else(|| invalid("root interval was not built"))?;
    let twice_i = builder.constant(ExactComplexRational::new(
        ExactRational::ZERO,
        ExactRational::new(2, 1)?,
    ))?;
    let configuration_count = 1_u32
        .checked_shl(u32::from(external_count))
        .ok_or_else(|| invalid("helicity configuration count overflows"))?;
    for mask in 0..configuration_count {
        if helicity_flip_reduction && mask & (1_u32 << (external_count - 1)) != 0 {
            continue;
        }
        let helicities = (0..external_count)
            .map(|index| if mask & (1_u32 << index) == 0 { -1 } else { 1 })
            .collect::<Vec<_>>();
        let negative_count = helicities
            .iter()
            .filter(|helicity| **helicity == -1)
            .count();
        let multiplicity = if helicity_flip_reduction { 2 } else { 1 };
        if negative_count < 2 || negative_count + 2 > usize::from(external_count) {
            builder.add_root_with_multiplicity(
                helicities.into_boxed_slice(),
                builder.zero(),
                multiplicity,
            )?;
            continue;
        }
        let complement_mask = usize::try_from(mask & ((1_u32 << complement_count) - 1))
            .map_err(|_| invalid("complement helicity mask exceeds usize"))?;
        let anchor = external_polarization_expression(
            &mut builder,
            external_count - 1,
            helicities[usize::from(external_count - 1)],
        )?;
        let root =
            bispinor_dot_expression(&mut builder, &anchor, &root_numerators[complement_mask])?;
        let root = builder.product([twice_i, root])?;
        builder.add_root_with_multiplicity(helicities.into_boxed_slice(), root, multiplicity)?;
    }
    builder.finish()
}

fn build_compact_gluon_tree_spinor_dag_impl(
    external_count: u16,
    helicity_flip_reduction: bool,
) -> RusticolResult<SpinorDag> {
    if !(4..=6).contains(&external_count) {
        return Err(invalid(
            "the compact spinor-helicity slice currently supports 4, 5, or 6 gluons",
        ));
    }
    let mut builder = SpinorDagBuilder::new(external_count)?;
    let imaginary_unit = builder.constant(ExactComplexRational::new(
        ExactRational::ZERO,
        ExactRational::ONE,
    ))?;
    let mut angle_cycle = Vec::with_capacity(usize::from(external_count));
    let mut square_cycle = Vec::with_capacity(usize::from(external_count));
    for left in 0..external_count {
        let right = (left + 1) % external_count;
        angle_cycle.push(builder.angle(left, right)?);
        square_cycle.push(builder.square(left, right)?);
    }
    let angle_denominator = builder.product(angle_cycle)?;
    let square_denominator = builder.product(square_cycle)?;
    let angle_reciprocal = builder.reciprocal(angle_denominator)?;
    let square_reciprocal = builder.reciprocal(square_denominator)?;
    let configuration_count = 1_u32
        .checked_shl(u32::from(external_count))
        .ok_or_else(|| invalid("helicity configuration count overflows"))?;
    for mask in 0..configuration_count {
        if helicity_flip_reduction && mask & (1_u32 << (external_count - 1)) != 0 {
            continue;
        }
        let helicities = (0..external_count)
            .map(|index| if mask & (1_u32 << index) == 0 { -1 } else { 1 })
            .collect::<Vec<_>>();
        let negative = helicities
            .iter()
            .enumerate()
            .filter_map(|(index, helicity)| (*helicity == -1).then_some(index as u16))
            .collect::<Vec<_>>();
        let positive = helicities
            .iter()
            .enumerate()
            .filter_map(|(index, helicity)| (*helicity == 1).then_some(index as u16))
            .collect::<Vec<_>>();
        let amplitude = if negative.len() == 2 {
            let bracket = if external_count == 4 {
                four_point_bracket_via_schouten(
                    &mut builder,
                    SpinorBracketKind::Angle,
                    negative[0],
                    negative[1],
                )?
            } else {
                builder.angle(negative[0], negative[1])?
            };
            let numerator = builder.pow(bracket, 4)?;
            builder.product([imaginary_unit, numerator, angle_reciprocal])?
        } else if positive.len() == 2 {
            let bracket = if external_count == 4 {
                four_point_bracket_via_schouten(
                    &mut builder,
                    SpinorBracketKind::Square,
                    positive[0],
                    positive[1],
                )?
            } else {
                builder.square(positive[0], positive[1])?
            };
            let numerator = builder.pow(bracket, 4)?;
            let phase = if external_count % 2 == 0 {
                imaginary_unit
            } else {
                builder.negate(imaginary_unit)?
            };
            builder.product([phase, numerator, square_reciprocal])?
        } else if external_count == 6 && negative.len() == 3 {
            six_gluon_nmhv_bcfw_root(&mut builder, &helicities, imaginary_unit)?
        } else {
            builder.zero()
        };
        builder.add_root_with_multiplicity(
            helicities.into_boxed_slice(),
            amplitude,
            if helicity_flip_reduction { 2 } else { 1 },
        )?;
    }
    builder.finish()
}

/// Construct the complete tree-level fixed-order gluon helicity DAG where
/// MHV and anti-MHV sectors exhaust the answer. This is exact for four and
/// five external gluons; six-point NMHV requires the next construction stage.
pub fn build_complete_gluon_tree_spinor_dag(external_count: u16) -> RusticolResult<SpinorDag> {
    if !matches!(external_count, 4 | 5) {
        return Err(invalid(
            "the complete Parke-Taylor vertical slice currently supports 4 or 5 gluons",
        ));
    }
    build_compact_gluon_tree_spinor_dag_impl(external_count, false)
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
pub type WeylSpinor = [Complex64; 2];
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
pub type Bispinor = [Complex64; 4];
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
pub type LorentzVector = [Complex64; 4];

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
const HALF: Complex64 = Complex64::new(0.5, 0.0);
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
const I: Complex64 = Complex64::new(0.0, 1.0);

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MasslessSpinors {
    pub undotted: WeylSpinor,
    pub dotted: WeylSpinor,
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
impl MasslessSpinors {
    /// Factor a real, possibly negative-energy, null momentum as
    /// `p[a,adot] = lambda[a] tilde_lambda[adot]`.
    pub fn from_momentum(momentum: [f64; 4]) -> RusticolResult<Self> {
        if momentum.iter().any(|component| !component.is_finite()) {
            return Err(invalid("massless momentum components must be finite"));
        }
        let [energy, px, py, pz] = momentum;
        if energy == 0.0 {
            return Err(invalid("massless momentum energy must be nonzero"));
        }
        let norm_scale = energy.abs().max(px.abs()).max(py.abs()).max(pz.abs());
        let mass_squared = energy * energy - px * px - py * py - pz * pz;
        if mass_squared.abs() > 128.0 * f64::EPSILON * norm_scale * norm_scale {
            return Err(invalid(format!(
                "momentum is not null (p^2={mass_squared:.17e})"
            )));
        }
        let sign = energy.signum();
        let q0 = sign * energy;
        let qx = sign * px;
        let qy = sign * py;
        let qz = sign * pz;
        let plus = (q0 + qz).max(0.0);
        let minus = (q0 - qz).max(0.0);
        let mut undotted = if plus >= minus && plus > 0.0 {
            let root = plus.sqrt();
            [Complex64::new(root, 0.0), Complex64::new(qx, qy) / root]
        } else if minus > 0.0 {
            let root = minus.sqrt();
            [Complex64::new(qx, -qy) / root, Complex64::new(root, 0.0)]
        } else {
            return Err(invalid("massless momentum cannot be factorized"));
        };
        if sign < 0.0 {
            for component in &mut undotted {
                *component *= I;
            }
        }
        let dotted = undotted.map(|component| sign * component.conj());
        Ok(Self { undotted, dotted })
    }

    pub fn bispinor(self) -> Bispinor {
        outer(self.undotted, self.dotted)
    }
}

/// Split a real timelike momentum into two real null momenta `p = k + r`.
/// A fixed finite set of null directions makes the choice deterministic; the
/// resulting little-group phases are immaterial to helicity-summed norms.
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn decompose_massive_momentum(
    momentum: [f64; 4],
) -> RusticolResult<(MasslessSpinors, MasslessSpinors, f64)> {
    if momentum.iter().any(|component| !component.is_finite()) {
        return Err(invalid("massive momentum components must be finite"));
    }
    let [energy, px, py, pz] = momentum;
    let mass_squared = energy * energy - px * px - py * py - pz * pz;
    let scale = energy.abs().max(px.abs()).max(py.abs()).max(pz.abs());
    if scale == 0.0 || mass_squared <= 256.0 * f64::EPSILON * scale * scale {
        return Err(invalid(format!(
            "massive-vector source must be timelike (p^2={mass_squared:.17e})"
        )));
    }
    const DIRECTIONS: [[f64; 4]; 6] = [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, -1.0],
    ];
    let direction = DIRECTIONS
        .into_iter()
        .max_by(|left, right| {
            let left_dot = energy * left[0] - px * left[1] - py * left[2] - pz * left[3];
            let right_dot = energy * right[0] - px * right[1] - py * right[2] - pz * right[3];
            left_dot.abs().total_cmp(&right_dot.abs())
        })
        .ok_or_else(|| invalid("massive decomposition direction set is empty"))?;
    let dot = energy * direction[0] - px * direction[1] - py * direction[2] - pz * direction[3];
    if dot.abs() <= 1024.0 * f64::EPSILON * scale {
        return Err(invalid("massive decomposition reference is singular"));
    }
    let coefficient = mass_squared / (2.0 * dot);
    let r_momentum = direction.map(|component| coefficient * component);
    let k_momentum = std::array::from_fn(|index| momentum[index] - r_momentum[index]);
    let k = MasslessSpinors::from_momentum(k_momentum)?;
    let r = MasslessSpinors::from_momentum(r_momentum)?;
    Ok((k, r, mass_squared.sqrt()))
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn evaluate_kinematic(
    scalar: SpinorKinematicScalar,
    momenta: &[[f64; 4]],
) -> RusticolResult<Complex64> {
    match scalar {
        SpinorKinematicScalar::SqrtTwo => Ok(Complex64::new(2.0_f64.sqrt(), 0.0)),
        SpinorKinematicScalar::InverseMass { source } => {
            let momentum = momenta
                .get(usize::from(source))
                .copied()
                .ok_or_else(|| invalid("inverse-mass source is outside the momentum input"))?;
            let (_, _, mass) = decompose_massive_momentum(momentum)?;
            Ok(Complex64::new(1.0 / mass, 0.0))
        }
    }
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn evaluate_kinematic_flat(
    scalar: SpinorKinematicScalar,
    point: &[f64],
) -> RusticolResult<Complex64> {
    match scalar {
        SpinorKinematicScalar::SqrtTwo => Ok(Complex64::new(2.0_f64.sqrt(), 0.0)),
        SpinorKinematicScalar::InverseMass { source } => {
            let offset = usize::from(source)
                .checked_mul(4)
                .ok_or_else(|| invalid("inverse-mass source offset overflows"))?;
            let components = point
                .get(offset..offset + 4)
                .ok_or_else(|| invalid("inverse-mass source is outside the flat point"))?;
            let momentum = [components[0], components[1], components[2], components[3]];
            let (_, _, mass) = decompose_massive_momentum(momentum)?;
            Ok(Complex64::new(1.0 / mass, 0.0))
        }
    }
}

/// Choose one deterministic, well-separated null reference for every gluon
/// source at a point. A common reference exposes the largest exact
/// same-helicity cancellations in the bispinor graph.
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn select_common_reference_momentum(momenta: &[[f64; 4]]) -> RusticolResult<[f64; 4]> {
    const CANDIDATES: [[f64; 4]; 6] = [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, -1.0],
    ];
    let mut best = None;
    for candidate in CANDIDATES {
        let score = momenta.iter().try_fold(f64::INFINITY, |score, momentum| {
            let scale = momentum
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0_f64, f64::max);
            if !scale.is_finite() || scale == 0.0 {
                return Err(invalid("cannot select a reference for a zero momentum"));
            }
            let dot = candidate[0] * momentum[0]
                - candidate[1] * momentum[1]
                - candidate[2] * momentum[2]
                - candidate[3] * momentum[3];
            Ok(score.min(dot.abs() / scale))
        })?;
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((candidate, score));
        }
    }
    let (candidate, score) = best.ok_or_else(|| invalid("reference candidate set is empty"))?;
    if score <= 1024.0 * f64::EPSILON {
        return Err(invalid(
            "all deterministic reference spinors are numerically collinear",
        ));
    }
    Ok(candidate)
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn select_common_reference_flat(point: &[f64]) -> RusticolResult<[f64; 4]> {
    const CANDIDATES: [[f64; 4]; 6] = [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, -1.0],
    ];
    if point.is_empty() || point.len() % 4 != 0 {
        return Err(invalid(
            "flat reference selection requires one or more four-component momenta",
        ));
    }
    let mut best = None;
    for candidate in CANDIDATES {
        let score = point.chunks_exact(4).try_fold(
            f64::INFINITY,
            |score, momentum| -> RusticolResult<f64> {
                let scale = momentum.iter().copied().map(f64::abs).fold(0.0, f64::max);
                if !scale.is_finite() || scale == 0.0 {
                    return Err(invalid("cannot select a reference for a zero momentum"));
                }
                let dot = candidate[0] * momentum[0]
                    - candidate[1] * momentum[1]
                    - candidate[2] * momentum[2]
                    - candidate[3] * momentum[3];
                Ok(score.min(dot.abs() / scale))
            },
        )?;
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((candidate, score));
        }
    }
    let (candidate, score) = best.ok_or_else(|| invalid("reference candidate set is empty"))?;
    if score <= 1024.0 * f64::EPSILON {
        return Err(invalid(
            "all deterministic reference spinors are numerically collinear",
        ));
    }
    Ok(candidate)
}

/// Convert `(v0,v1,v2,v3)` to `v_mu sigma^mu`, in row-major order.
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[inline]
pub fn vector_to_bispinor(vector: LorentzVector) -> Bispinor {
    let [v0, v1, v2, v3] = vector;
    [v0 + v3, v1 - I * v2, v1 + I * v2, v0 - v3]
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[inline]
pub fn bispinor_to_vector(matrix: Bispinor) -> LorentzVector {
    let [m00, m01, m10, m11] = matrix;
    [
        HALF * (m00 + m11),
        HALF * (m01 + m10),
        (m10 - m01) / (I + I),
        HALF * (m00 - m11),
    ]
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[inline]
pub fn adjugate(matrix: Bispinor) -> Bispinor {
    let [m00, m01, m10, m11] = matrix;
    [m11, -m01, -m10, m00]
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[inline]
pub fn bispinor_dot(left: Bispinor, right: Bispinor) -> Complex64 {
    HALF * (left[0] * right[3] + left[3] * right[0] - left[1] * right[2] - left[2] * right[1])
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[inline]
pub fn angle(left: WeylSpinor, right: WeylSpinor) -> Complex64 {
    left[0] * right[1] - left[1] * right[0]
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[inline]
pub fn square(left: WeylSpinor, right: WeylSpinor) -> Complex64 {
    left[1] * right[0] - left[0] * right[1]
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[inline]
pub fn outer(left: WeylSpinor, right: WeylSpinor) -> Bispinor {
    [
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    ]
}

/// Reference-spinor decomposition of an external massless polarization.
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
pub fn massless_polarization_bispinor(
    momentum: MasslessSpinors,
    reference: MasslessSpinors,
    helicity: i8,
) -> RusticolResult<Bispinor> {
    let sqrt_two = 2.0_f64.sqrt();
    let (numerator, denominator) = match helicity {
        1 => (
            outer(reference.undotted, momentum.dotted),
            angle(reference.undotted, momentum.undotted),
        ),
        -1 => (
            outer(momentum.undotted, reference.dotted),
            square(momentum.dotted, reference.dotted),
        ),
        _ => return Err(invalid("massless polarization helicity must be -1 or +1")),
    };
    if denominator.norm_sqr() <= 64.0 * f64::MIN_POSITIVE {
        return Err(invalid(
            "reference spinor is collinear with the polarization momentum",
        ));
    }
    Ok(numerator.map(|component| component * sqrt_two / denominator))
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[inline]
pub fn weyl_vector_positive(spinor: WeylSpinor, vector: Bispinor) -> WeylSpinor {
    let [l0, l1] = spinor;
    let [v00, v01, v10, v11] = vector;
    [I * (l0 * v11 - l1 * v10), I * (-l0 * v01 + l1 * v00)]
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[inline]
pub fn weyl_vector_negative(spinor: WeylSpinor, vector: Bispinor) -> WeylSpinor {
    let [l0, l1] = spinor;
    let [v00, v01, v10, v11] = vector;
    [I * (l0 * v00 + l1 * v10), I * (l0 * v01 + l1 * v11)]
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[inline]
pub fn color_ordered_three_vector(
    left: Bispinor,
    right: Bispinor,
    left_momentum: Bispinor,
    right_momentum: Bispinor,
) -> Bispinor {
    let lr = bispinor_dot(left, right);
    let lq = bispinor_dot(left, right_momentum);
    let rp = bispinor_dot(right, left_momentum);
    std::array::from_fn(|component| {
        lr * (left_momentum[component] - right_momentum[component])
            + (lq * right[component] - rp * left[component]) * 2.0
    })
}

/// Fused color-ordered four-vector contact after eliminating the auxiliary
/// antisymmetric tensor used by the component recurrence.
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[inline]
pub fn color_ordered_four_vector_contact(
    left: Bispinor,
    middle: Bispinor,
    right: Bispinor,
) -> Bispinor {
    let left_dot_right = bispinor_dot(left, right);
    let left_dot_middle = bispinor_dot(left, middle);
    let middle_dot_right = bispinor_dot(middle, right);
    std::array::from_fn(|component| {
        2.0 * left_dot_right * middle[component]
            - middle_dot_right * left[component]
            - left_dot_middle * right[component]
    })
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn exact_complex_to_f64(value: ExactComplexRational) -> Complex64 {
    Complex64::new(exact_to_f64(value.real()), exact_to_f64(value.imag()))
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn exact_to_f64(value: ExactRational) -> f64 {
    value.numerator() as f64 / value.denominator() as f64
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn value(values: &[Complex64], id: SpinorNodeId) -> RusticolResult<Complex64> {
    values
        .get(usize::try_from(id).map_err(|_| invalid("node ID exceeds usize"))?)
        .copied()
        .ok_or_else(|| invalid(format!("node ID {id} is not topologically available")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    const TOLERANCE: f64 = 4.0e-12;

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    fn assert_close(left: Complex64, right: Complex64) {
        let scale = 1.0 + left.norm().max(right.norm());
        assert!(
            (left - right).norm() <= TOLERANCE * scale,
            "{left:?} != {right:?}"
        );
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    fn assert_array_close<const N: usize>(left: [Complex64; N], right: [Complex64; N]) {
        for (left, right) in left.into_iter().zip(right) {
            assert_close(left, right);
        }
    }

    #[test]
    fn bracket_nodes_are_antisymmetric_and_self_products_vanish() {
        let mut builder = SpinorDagBuilder::new(4).unwrap();
        let forward = builder.angle(0, 3).unwrap();
        let reverse = builder.angle(3, 0).unwrap();
        let zero = builder.angle(2, 2).unwrap();
        assert_ne!(forward, reverse);
        assert_eq!(zero, builder.zero());
        assert_eq!(builder.rewrite_stats.antisymmetry_rewrites, 1);
        assert_eq!(builder.rewrite_stats.self_product_zeros, 1);
    }

    #[test]
    fn schouten_reduces_the_nonadjacent_four_gluon_numerator() {
        let mut builder = SpinorDagBuilder::new(4).unwrap();
        let a = builder.angle(0, 1).unwrap();
        let b = builder.angle(2, 3).unwrap();
        let first = builder.product([a, b]).unwrap();
        let c = builder.angle(0, 3).unwrap();
        let d = builder.angle(1, 2).unwrap();
        let second = builder.product([c, d]).unwrap();
        let expanded = builder.sum([first, second]).unwrap();
        let reduced = builder.simplify_schouten(expanded).unwrap();
        let expected_left = builder.angle(0, 2).unwrap();
        let expected_right = builder.angle(1, 3).unwrap();
        let expected = builder.product([expected_left, expected_right]).unwrap();
        assert_eq!(reduced, expected);
        assert_eq!(builder.rewrite_stats.schouten_rewrites, 1);
    }

    #[test]
    fn fierz_rewrite_contains_no_vector_node() {
        let mut builder = SpinorDagBuilder::new(4).unwrap();
        let rewritten = builder
            .fierz_current_contraction(SpinorChirality::Positive, 0, 1, 2, 3)
            .unwrap();
        assert!(matches!(
            builder.node(rewritten).unwrap(),
            SpinorNode::Product(_)
        ));
        assert_eq!(builder.rewrite_stats.fierz_rewrites, 1);
        assert!(builder.nodes.iter().all(|node| matches!(
            node,
            SpinorNode::Constant(_)
                | SpinorNode::Bracket { .. }
                | SpinorNode::Sum(_)
                | SpinorNode::Product(_)
                | SpinorNode::Reciprocal(_)
        )));
    }

    #[test]
    fn sparse_bivector_action_has_the_certified_tensor_vector_orientation() {
        let mut builder = SpinorDagBuilder::new(4).unwrap();
        let one = builder.one();
        let left = BispinorExpression::dyad(0, 1, one);
        let right = BispinorExpression::dyad(2, 3, one);
        let vector = BispinorExpression::dyad(1, 2, one);

        let tensor = bivector_wedge_expression(&builder, &left, &right);
        let actual = bivector_vector_expression(&mut builder, &tensor, &vector).unwrap();

        let left_dot_vector = bispinor_dot_expression(&mut builder, &left, &vector).unwrap();
        let right_dot_vector = bispinor_dot_expression(&mut builder, &right, &vector).unwrap();
        let right_term = bispinor_scale(&mut builder, left_dot_vector, &right).unwrap();
        let negative_right_dot_vector = builder.negate(right_dot_vector).unwrap();
        let left_term = bispinor_scale(&mut builder, negative_right_dot_vector, &left).unwrap();
        let expected = bispinor_sum(&mut builder, [right_term, left_term]).unwrap();

        assert_ne!(actual, BispinorExpression::default());
        assert_eq!(actual, expected);
    }

    #[test]
    fn four_and_five_gluon_graphs_cover_each_helicity_once() {
        for count in [4_u16, 5] {
            let dag = build_complete_gluon_tree_spinor_dag(count).unwrap();
            assert_eq!(dag.roots.len(), 1_usize << count);
            let unique = dag
                .roots
                .iter()
                .map(|root| root.helicities.clone())
                .collect::<BTreeSet<_>>();
            assert_eq!(unique.len(), dag.roots.len());
            let expected_zeros = if count == 4 { 10 } else { 12 };
            assert_eq!(dag.rewrite_stats.structural_zero_roots, expected_zeros);
            assert!(dag.nodes.len() < dag.roots.len() * usize::from(count));
        }
    }

    #[test]
    fn quark_gluon_graphs_cover_all_helicities_and_chirality_zeros() {
        for count in [4_u16, 5, 6] {
            let order = (0..count).collect::<Vec<_>>();
            let dag = build_helicity_summed_quark_gluon_bg_spinor_dag(&order).unwrap();
            assert_eq!(dag.roots.len(), 1_usize << count);
            let unique = dag
                .roots
                .iter()
                .map(|root| root.helicities.clone())
                .collect::<BTreeSet<_>>();
            assert_eq!(unique.len(), dag.roots.len());
            assert_eq!(
                dag.roots.iter().filter(|root| root.structural_zero).count(),
                1_usize << (count - 1),
            );
        }
    }

    #[test]
    fn massive_quark_two_gluon_graph_has_complete_spin_and_parameter_census() {
        let dag = build_helicity_summed_massive_quark_two_gluon_spinor_dag(&[2, 0, 1, 3]).unwrap();
        assert_eq!(dag.momentum_count(), 4);
        assert_eq!(dag.parameter_count(), 2);
        assert_eq!(dag.roots().len(), 16);
        assert!(dag.roots().iter().all(|root| !root.is_structural_zero()));
        assert_eq!(
            dag.roots()
                .iter()
                .map(|root| root.helicities().to_vec())
                .collect::<BTreeSet<_>>()
                .len(),
            16,
        );
        let census = dag.census();
        assert_eq!(
            census,
            SpinorDagCensus {
                constants: 4,
                parameters: 2,
                kinematic_scalars: 0,
                brackets: 21,
                sums: 57,
                sum_operands: 129,
                products: 146,
                product_operands: 512,
                reciprocals: 7,
            }
        );
        assert_eq!(dag.nodes().len(), 237);
        assert!(!dag.uses_reference_atom());
        assert!(build_helicity_summed_massive_quark_two_gluon_spinor_dag(&[2, 0, 0, 3]).is_err());
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn massive_quark_two_gluon_norms_match_all_saved_lc_points_and_both_flows() {
        // `physics-v2` and its independent `legacy-fortran-v2` evidence carry
        // the same four widthful per-flow oracles. Divide their normalized
        // cells by 27*g_s^4/256 to obtain these coupling-stripped sums.
        let probes = [
            (
                [
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
                ],
                [8.954647716017921, 0.3069205428921417],
            ),
            (
                [
                    [500.0, 0.0, 0.0, 500.0],
                    [500.0, 0.0, 0.0, -500.0],
                    [
                        500.0,
                        -413.22510427682994,
                        -179.07952861163048,
                        -131.32606606321244,
                    ],
                    [
                        500.0,
                        413.22510427682994,
                        179.07952861163048,
                        131.32606606321244,
                    ],
                ],
                [0.7460947045512392, 2.1878442037697472],
            ),
            (
                [
                    [500.0, 0.0, 0.0, 500.0],
                    [500.0, 0.0, 0.0, -500.0],
                    [
                        500.0,
                        212.31242292963711,
                        -237.17089689689013,
                        344.59309443874758,
                    ],
                    [
                        500.0,
                        -212.31242292963711,
                        237.17089689689013,
                        -344.59309443874758,
                    ],
                ],
                [9.020602496253207, 0.3054093126849537],
            ),
            (
                [
                    [173.000173, 0.0, 0.0, 173.000173],
                    [173.000173, 0.0, 0.0, -173.000173],
                    [173.000173, 0.14679540447316463, 0.0, 0.1957272059642195],
                    [173.000173, -0.14679540447316463, 0.0, -0.1957272059642195],
                ],
                [1.0022679867918768, 0.9977425727766228],
            ),
        ];
        let parameters = [c(173.0, 0.0), c(1.4915, 0.0)];
        for (point, expected_flows) in probes {
            let momenta = [
                point[0].map(|component| -component),
                point[1].map(|component| -component),
                point[2],
                point[3],
            ];
            for (order, expected) in [
                ([2, 0, 1, 3], expected_flows[0]),
                ([2, 1, 0, 3], expected_flows[1]),
            ] {
                let dag = build_helicity_summed_massive_quark_two_gluon_spinor_dag(&order).unwrap();
                let actual = dag
                    .evaluate_with_parameters(&momenta, &parameters)
                    .unwrap()
                    .helicity_sum();
                assert!(
                    (actual - expected).abs() <= 2.0e-12 * expected,
                    "flow {order:?}: {actual} != {expected}"
                );
            }
        }
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn massive_quark_two_gluon_binds_mass_and_width_and_matches_batch() {
        let point = [
            [-500.0, 0.0, 0.0, -500.0],
            [-500.0, 0.0, 0.0, 500.0],
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
        let dag = build_helicity_summed_massive_quark_two_gluon_spinor_dag(&[2, 0, 1, 3]).unwrap();
        let parameters = [c(173.0, 0.0), c(1.4915, 0.0)];
        let expected = dag
            .evaluate_with_parameters(&point, &parameters)
            .unwrap()
            .helicity_sum();
        let zero_width = dag
            .evaluate_with_parameters(&point, &[c(173.0, 0.0), c(0.0, 0.0)])
            .unwrap()
            .helicity_sum();
        let shifted_mass = dag
            .evaluate_with_parameters(&point, &[c(172.0, 0.0), c(1.4915, 0.0)])
            .unwrap()
            .helicity_sum();
        assert!((zero_width - expected).abs() > 1.0e-6);
        assert!((shifted_mass - expected).abs() > 1.0e-3);
        assert!(dag.evaluate(&point).is_err());

        let flat = point.into_iter().flatten().collect::<Vec<_>>();
        let mut workspace = dag.batch_workspace(1).unwrap();
        let mut output = [0.0];
        dag.evaluate_sum_batch_into_with_parameters(
            &flat,
            1,
            &parameters,
            &mut workspace,
            &mut output,
        )
        .unwrap();
        assert!((output[0] - expected).abs() <= 2.0e-13 * expected);
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn five_point_open_quark_line_matches_the_saved_component_oracle() {
        // u u~ > g g g, flow 2,3,4,5,1.  The retained component-recurrence
        // fixture evaluates the selected all-outgoing helicity
        // (-,+,-,-,+) to 1e-6 before color/coupling normalization.
        let physical = [
            [500.0, 0.0, 0.0, 500.0],
            [500.0, 0.0, 0.0, -500.0],
            [333.3333333333333, 333.3333333333333, 0.0, 0.0],
            [
                333.3333333333333,
                -166.66666666666666,
                288.67513459481285,
                0.0,
            ],
            [
                333.3333333333333,
                -166.66666666666666,
                -288.67513459481285,
                0.0,
            ],
        ];
        let momenta = [
            physical[1].map(|component| -component),
            physical[2],
            physical[3],
            physical[4],
            physical[0].map(|component| -component),
        ];
        let dag = build_helicity_summed_quark_gluon_bg_spinor_dag(&[0, 1, 2, 3, 4]).unwrap();
        let evaluation = dag.evaluate(&momenta).unwrap();
        let root = dag
            .roots()
            .iter()
            .position(|root| root.helicities() == [-1, -1, -1, 1, 1])
            .unwrap();
        let actual = evaluation.amplitudes()[root].norm_sqr();
        assert!(
            (actual - 1.0e-6).abs() <= 2.0e-13 * 1.0e-6,
            "{actual} != 1e-6"
        );
        let summed = evaluation.helicity_sum();
        assert!(
            (summed - 1.2e-5).abs() <= 2.0e-13 * 1.2e-5,
            "{summed} != 1.2e-5"
        );
    }

    #[test]
    fn quark_z_gluon_graphs_keep_three_vector_polarizations_and_chiral_zeros() {
        for (colored_order, z_slot) in [
            (vec![1_u16, 0], 2_u16),
            (vec![1_u16, 3, 0], 2_u16),
            (vec![1_u16, 3, 4, 0], 2_u16),
        ] {
            let gluon_count = colored_order.len() - 2;
            let dag =
                build_helicity_summed_quark_z_gluon_spinor_dag(&colored_order, z_slot).unwrap();
            assert_eq!(dag.parameter_count(), 2);
            assert_eq!(dag.roots().len(), 12_usize << gluon_count);
            assert_eq!(
                dag.roots()
                    .iter()
                    .filter(|root| root.is_structural_zero())
                    .count(),
                6_usize << gluon_count,
            );
            let z_helicities = dag
                .roots()
                .iter()
                .map(|root| root.helicities()[usize::from(z_slot)])
                .collect::<BTreeSet<_>>();
            assert_eq!(z_helicities, BTreeSet::from([-1_i8, 0, 1]));
        }
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn quark_z_three_point_norm_matches_the_chiral_current_oracle() {
        // The independent Decimal oracle `chiral_current_2to1` gives
        // 2*s*g_L^2 = 800 and 2*s*g_R^2 = 1800 for this point. The massive
        // spinor basis is allowed to redistribute those norms among its three
        // polarization roots, so only each chirality sum is compared.
        let momenta = [
            [-5.0, 0.0, 0.0, -5.0],
            [-5.0, 0.0, 0.0, 5.0],
            [10.0, 0.0, 0.0, 0.0],
        ];
        let dag = build_helicity_summed_quark_z_gluon_spinor_dag(&[1, 0], 2).unwrap();
        let sqrt_two = 2.0_f64.sqrt();
        let evaluation = dag
            .evaluate_with_parameters(&momenta, &[c(sqrt_two * 2.0, 0.0), c(sqrt_two * 3.0, 0.0)])
            .unwrap();
        let mut left = 0.0;
        let mut right = 0.0;
        for (root, amplitude) in dag.roots().iter().zip(evaluation.amplitudes()) {
            match root.helicities() {
                [1, -1, _] => left += amplitude.norm_sqr(),
                [-1, 1, _] => right += amplitude.norm_sqr(),
                _ => assert_eq!(*amplitude, c(0.0, 0.0)),
            }
        }
        assert!((left - 800.0).abs() <= TOLERANCE * 800.0, "{left} != 800");
        assert!(
            (right - 1800.0).abs() <= TOLERANCE * 1800.0,
            "{right} != 1800"
        );
        assert!(
            (evaluation.helicity_sum() - 2600.0).abs() <= TOLERANCE * 2600.0,
            "{} != 2600",
            evaluation.helicity_sum()
        );
        assert!(dag.evaluate(&momenta).is_err());
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn quark_z_one_gluon_norm_matches_the_clean_component_oracle() {
        let physical = [
            [500.0, 0.0, 0.0, 500.0],
            [500.0, 0.0, 0.0, -500.0],
            [
                504.1576256720017,
                270.45289818999487,
                -290.90819815599644,
                -296.79506445608888,
            ],
            [
                495.8423743279983,
                -270.45289818999214,
                290.90819815599195,
                296.79506445609007,
            ],
        ];
        let momenta = [
            physical[0].map(|component| -component),
            physical[1].map(|component| -component),
            physical[2],
            physical[3],
        ];
        let dag = build_helicity_summed_quark_z_gluon_spinor_dag(&[1, 3, 0], 2).unwrap();
        let raw = dag
            .evaluate_with_parameters(
                &momenta,
                &[c(-1.0244420275940371, 0.0), c(0.17818666745287456, 0.0)],
            )
            .unwrap()
            .helicity_sum();
        let expected = 4.691124292986115;
        assert!(
            (raw - expected).abs() <= 2.0e-10 * expected,
            "{raw} != {expected}"
        );
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn quark_z_two_gluon_norms_match_both_clean_component_flows() {
        let physical = [
            [500.0, 0.0, 0.0, 500.0],
            [500.0, 0.0, 0.0, -500.0],
            [
                142.22002371924205,
                -53.100900829773103,
                59.040980400311128,
                -74.871495024064103,
            ],
            [
                380.35009699751708,
                -268.20692461897238,
                256.15852033986499,
                -84.344853605566854,
            ],
            [
                477.42987928324095,
                321.30782544874552,
                -315.19950074017623,
                159.21634862963094,
            ],
        ];
        let momenta = [
            physical[0].map(|component| -component),
            physical[1].map(|component| -component),
            physical[2],
            physical[3],
            physical[4],
        ];
        let parameters = [c(-1.0244420275940371, 0.0), c(0.17818666745287456, 0.0)];
        for (order, expected) in [
            ([1_u16, 3, 4, 0], 1.9944892649210398e-4),
            ([1_u16, 4, 3, 0], 6.27628015878202e-5),
        ] {
            let dag = build_helicity_summed_quark_z_gluon_spinor_dag(&order, 2).unwrap();
            let actual = dag
                .evaluate_with_parameters(&momenta, &parameters)
                .unwrap()
                .helicity_sum();
            assert!(
                (actual - expected).abs() <= 3.0e-10 * expected,
                "flow {order:?}: {actual} != {expected}"
            );
        }
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn four_and_six_point_open_quark_lines_match_clean_component_artifacts() {
        // These independent totals were evaluated with the clean-HEAD
        // component recurrence for the first LC flow, then divided by its
        // authenticated color/coupling/average/symmetry normalization.
        let four_point = [
            [-500.0, 0.0, 0.0, 500.0],
            [
                499.99999999999994,
                -306.65836769058797,
                210.51071473894038,
                334.13453054936508,
            ],
            [
                499.99999999999994,
                306.65836769058797,
                -210.51071473894038,
                -334.13453054936508,
            ],
            [-500.0, 0.0, 0.0, -500.0],
        ];
        let four = build_helicity_summed_quark_gluon_bg_spinor_dag(&[0, 1, 2, 3])
            .unwrap()
            .evaluate(&four_point)
            .unwrap()
            .helicity_sum();
        let expected_four = 0.2876493525277244;
        assert!(
            (four - expected_four).abs() <= 2.0e-12 * expected_four,
            "{four} != {expected_four}"
        );

        let six_point = [
            [-500.0, 0.0, 0.0, 500.0],
            [
                290.31075577423769,
                -213.36929162671257,
                88.965338665326172,
                175.61050317417747,
            ],
            [
                362.257846224096,
                153.7528027329422,
                -190.28771533698082,
                -267.17299301111615,
            ],
            [
                186.58961514320916,
                -82.776424293003245,
                62.11635039741472,
                155.25883895566528,
            ],
            [
                160.84178285845738,
                142.39291318677365,
                39.206026274239925,
                -63.696349118726602,
            ],
            [-500.0, 0.0, 0.0, -500.0],
        ];
        let six = build_helicity_summed_quark_gluon_bg_spinor_dag(&[0, 1, 2, 3, 4, 5])
            .unwrap()
            .evaluate(&six_point)
            .unwrap()
            .helicity_sum();
        let expected_six = 8.59278215817057e-11;
        assert!(
            (six - expected_six).abs() <= 3.0e-11 * expected_six,
            "{six} != {expected_six}"
        );
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn dyad_dot_is_the_repository_fierz_contraction() {
        let a = [c(1.0, 0.5), c(-2.0, 1.0)];
        let b = [c(0.25, -1.0), c(3.0, 0.75)];
        let d = [c(-0.5, 2.0), c(1.25, -0.25)];
        let e = [c(2.5, -1.5), c(-0.75, 0.5)];
        let component = bispinor_dot(outer(a, b), outer(d, e));
        let fierz = -0.5 * angle(a, d) * square(b, e);
        assert_close(component, fierz);
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn fused_four_vector_contact_matches_both_auxiliary_orientations() {
        let left = [c(1.0, 0.2), c(-0.5, 1.0), c(2.0, -0.3), c(0.75, 0.4)];
        let middle = [c(-0.4, 0.8), c(1.5, -0.2), c(0.6, 1.1), c(-2.0, 0.5)];
        let right = [c(0.3, -0.7), c(-1.2, 0.9), c(0.5, -1.4), c(1.1, 0.6)];
        let wedge = |a: LorentzVector, b: LorentzVector| {
            [
                a[0] * b[1] - a[1] * b[0],
                a[0] * b[2] - a[2] * b[0],
                a[0] * b[3] - a[3] * b[0],
                a[1] * b[2] - a[2] * b[1],
                a[1] * b[3] - a[3] * b[1],
                a[2] * b[3] - a[3] * b[2],
            ]
        };
        let tensor_vector = |tensor: [Complex64; 6], vector: LorentzVector| {
            let [t0, t1, t2, t3, t4, t5] = tensor;
            let [v0, v1, v2, v3] = vector;
            [
                t0 * v1 + t1 * v2 + t2 * v3,
                t0 * v0 + t3 * v2 + t4 * v3,
                t1 * v0 - t3 * v1 + t5 * v3,
                t2 * v0 - t4 * v1 - t5 * v2,
            ]
        };
        let vector_tensor = |vector: LorentzVector, tensor: [Complex64; 6]| {
            let [v0, v1, v2, v3] = vector;
            let [t0, t1, t2, t3, t4, t5] = tensor;
            [
                -v1 * t0 - v2 * t1 - v3 * t2,
                -v0 * t0 - v2 * t3 - v3 * t4,
                -v0 * t1 + v1 * t3 - v3 * t5,
                -v0 * t2 + v1 * t4 + v2 * t5,
            ]
        };
        let left_tensor = tensor_vector(wedge(left, middle), right);
        let right_tensor = vector_tensor(left, wedge(middle, right));
        let auxiliary_sum = std::array::from_fn(|index| left_tensor[index] + right_tensor[index]);
        let fused = bispinor_to_vector(color_ordered_four_vector_contact(
            vector_to_bispinor(left),
            vector_to_bispinor(middle),
            vector_to_bispinor(right),
        ));
        assert_array_close(auxiliary_sum, fused);
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn bispinor_bg_matches_parke_taylor_for_every_four_and_five_gluon_root() {
        let points = [
            vec![
                [-5.0, 0.0, 0.0, -5.0],
                [-5.0, 0.0, 0.0, 5.0],
                [5.0, 3.0, 0.0, 4.0],
                [5.0, -3.0, 0.0, -4.0],
            ],
            vec![
                [-3.0, 0.0, 0.0, -3.0],
                [-3.0, 0.0, 0.0, 3.0],
                [2.0, 2.0, 0.0, 0.0],
                [2.0, -1.0, 3.0_f64.sqrt(), 0.0],
                [2.0, -1.0, -3.0_f64.sqrt(), 0.0],
            ],
        ];
        for point in points {
            let count = u16::try_from(point.len()).unwrap();
            let bg = build_gluon_bg_spinor_dag(count).unwrap();
            let oracle = build_complete_gluon_tree_spinor_dag(count).unwrap();
            assert_eq!(bg.roots().len(), oracle.roots().len());
            assert!(bg.rewrite_stats().fierz_rewrites > 0);
            let actual = bg.evaluate(&point).unwrap();
            let expected = oracle.evaluate(&point).unwrap();
            for ((actual_root, expected_root), (actual, expected)) in bg
                .roots()
                .iter()
                .zip(oracle.roots())
                .zip(actual.amplitudes().iter().zip(expected.amplitudes()))
            {
                assert_eq!(actual_root.helicities(), expected_root.helicities());
                assert_close(*actual, *expected);
            }
            assert!((actual.helicity_sum() - expected.helicity_sum()).abs() <= TOLERANCE);
        }
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn node_major_batch_matches_scalar_helicity_sum() {
        let dag = build_gluon_bg_spinor_dag(4).unwrap();
        let point = [
            [-5.0, 0.0, 0.0, -5.0],
            [-5.0, 0.0, 0.0, 5.0],
            [5.0, 3.0, 0.0, 4.0],
            [5.0, -3.0, 0.0, -4.0],
        ];
        let flat = point
            .iter()
            .flat_map(|momentum| momentum.iter().copied())
            .chain(point.iter().flat_map(|momentum| momentum.iter().copied()))
            .collect::<Vec<_>>();
        let expected = dag.evaluate(&point).unwrap().helicity_sum();
        let mut workspace = dag.batch_workspace(2).unwrap();
        let mut output = [0.0; 2];
        dag.evaluate_sum_batch_into(&flat, 2, &mut workspace, &mut output)
            .unwrap();
        assert!((output[0] - expected).abs() <= TOLERANCE);
        assert_eq!(output[0], output[1]);
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn always_summed_graph_reduces_global_helicity_flip_pairs() {
        let point = [
            [-5.0, 0.0, 0.0, -5.0],
            [-5.0, 0.0, 0.0, 5.0],
            [5.0, 3.0, 0.0, 4.0],
            [5.0, -3.0, 0.0, -4.0],
        ];
        let complete = build_gluon_bg_spinor_dag(4).unwrap();
        let summed = build_helicity_summed_gluon_bg_spinor_dag(4).unwrap();
        assert_eq!(complete.roots().len(), 16);
        assert_eq!(summed.roots().len(), 8);
        assert!(summed.roots().iter().all(|root| root.multiplicity() == 2));
        assert!(summed.nodes().len() < complete.nodes().len());
        let complete_value = complete.evaluate(&point).unwrap().helicity_sum();
        let summed_value = summed.evaluate(&point).unwrap().helicity_sum();
        assert!((complete_value - summed_value).abs() <= TOLERANCE);
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn mhv_lowering_compacts_the_validated_bispinor_graph() {
        let points = [
            vec![
                [-5.0, 0.0, 0.0, -5.0],
                [-5.0, 0.0, 0.0, 5.0],
                [5.0, 3.0, 0.0, 4.0],
                [5.0, -3.0, 0.0, -4.0],
            ],
            vec![
                [-3.0, 0.0, 0.0, -3.0],
                [-3.0, 0.0, 0.0, 3.0],
                [2.0, 2.0, 0.0, 0.0],
                [2.0, -1.0, 3.0_f64.sqrt(), 0.0],
                [2.0, -1.0, -3.0_f64.sqrt(), 0.0],
            ],
        ];
        for point in points {
            let count = u16::try_from(point.len()).unwrap();
            let raw = build_helicity_summed_gluon_bg_spinor_dag(count).unwrap();
            let optimized = build_optimized_helicity_summed_gluon_spinor_dag(count).unwrap();
            assert_eq!(raw.roots().len(), optimized.roots().len());
            assert!(optimized.nodes().len() < raw.nodes().len());
            let compact_node_budget = if count == 4 { 16 } else { 40 };
            assert!(optimized.nodes().len() <= compact_node_budget);
            assert!(!optimized.uses_reference_atom());
            assert!(raw.rewrite_stats().fierz_rewrites > 0);
            if count == 4 {
                assert!(optimized.rewrite_stats().schouten_rewrites > 0);
            }
            let raw_value = raw.evaluate(&point).unwrap().helicity_sum();
            let optimized_value = optimized.evaluate(&point).unwrap().helicity_sum();
            assert!((raw_value - optimized_value).abs() <= TOLERANCE);
        }
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn six_gluon_nmhv_bcfw_roots_match_raw_bg_root_by_root() {
        // All-outgoing form of numerical_acceptance/ufo-sm-seed101-v1.json,
        // case `catalog:gg_gluons:n4`.
        let point = [
            [-500.0, 0.0, 0.0, -500.0],
            [-500.0, 0.0, 0.0, 500.0],
            [
                290.31075577423769,
                -213.36929162671257,
                88.965338665326172,
                175.61050317417747,
            ],
            [
                362.257846224096,
                153.7528027329422,
                -190.28771533698082,
                -267.17299301111615,
            ],
            [
                186.58961514320916,
                -82.776424293003245,
                62.11635039741472,
                155.25883895566528,
            ],
            [
                160.84178285845738,
                142.39291318677365,
                39.206026274239925,
                -63.696349118726602,
            ],
        ];
        let raw = build_helicity_summed_gluon_bg_spinor_dag(6).unwrap();
        let optimized = build_optimized_helicity_summed_gluon_spinor_dag(6).unwrap();
        assert_eq!(raw.roots().len(), 32);
        assert_eq!(optimized.roots().len(), 32);
        assert_eq!(
            optimized
                .roots()
                .iter()
                .filter(|root| root.is_structural_zero())
                .count(),
            7
        );
        assert!(optimized.nodes().len() < raw.nodes().len());
        assert!(optimized.nodes().len() <= 450);
        assert!(optimized.census().estimated_complex_arithmetic() <= 1_600);
        assert!(optimized.batch_workspace(1).unwrap().value_slot_count() <= 110);
        assert!(!optimized.uses_reference_atom());

        let raw_evaluation = raw.evaluate(&point).unwrap();
        let optimized_evaluation = optimized.evaluate(&point).unwrap();
        let mut nmhv_roots = 0;
        for (((raw_root, optimized_root), raw_amplitude), optimized_amplitude) in raw
            .roots()
            .iter()
            .zip(optimized.roots())
            .zip(raw_evaluation.amplitudes())
            .zip(optimized_evaluation.amplitudes())
        {
            assert_eq!(raw_root.helicities(), optimized_root.helicities());
            if raw_root
                .helicities()
                .iter()
                .filter(|helicity| **helicity == -1)
                .count()
                != 3
            {
                continue;
            }
            nmhv_roots += 1;
            let scale = raw_amplitude
                .norm()
                .max(optimized_amplitude.norm())
                .max(f64::MIN_POSITIVE);
            assert!(
                (*raw_amplitude - *optimized_amplitude).norm() <= 5.0e-10 * scale,
                "NMHV root {:?}: raw={raw_amplitude:?}, BCFW={optimized_amplitude:?}",
                raw_root.helicities()
            );
        }
        assert_eq!(nmhv_roots, 10);
        assert!(
            (raw_evaluation.helicity_sum() - optimized_evaluation.helicity_sum()).abs()
                <= TOLERANCE * raw_evaluation.helicity_sum().abs().max(1.0)
        );
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn vector_bispinor_round_trip_and_metric() {
        let left = [c(1.0, 2.0), c(-3.0, 0.5), c(4.0, -2.0), c(0.25, 1.5)];
        let right = [c(-2.0, 1.0), c(0.75, -4.0), c(2.0, 3.0), c(-1.0, 0.5)];
        let dot = |a: LorentzVector, b: LorentzVector| {
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
        };
        let left_matrix = vector_to_bispinor(left);
        let right_matrix = vector_to_bispinor(right);
        assert_array_close(bispinor_to_vector(left_matrix), left);
        assert_close(bispinor_dot(left_matrix, right_matrix), dot(left, right));
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn momentum_factorization_and_bracket_identity_include_crossing() {
        let momenta = [
            [-5.0, 0.0, 0.0, -5.0],
            [-5.0, 0.0, 0.0, 5.0],
            [5.0, 3.0, 0.0, 4.0],
            [5.0, -3.0, 0.0, -4.0],
        ];
        let spinors = momenta.map(|momentum| MasslessSpinors::from_momentum(momentum).unwrap());
        for (momentum, spinor) in momenta.into_iter().zip(spinors) {
            assert_array_close(
                spinor.bispinor(),
                vector_to_bispinor(momentum.map(|component| c(component, 0.0))),
            );
        }
        for i in 0..4 {
            for j in 0..4 {
                let lhs = angle(spinors[i].undotted, spinors[j].undotted)
                    * square(spinors[j].dotted, spinors[i].dotted);
                let dot = momenta[i][0] * momenta[j][0]
                    - momenta[i][1] * momenta[j][1]
                    - momenta[i][2] * momenta[j][2]
                    - momenta[i][3] * momenta[j][3];
                assert_close(lhs, c(2.0 * dot, 0.0));
            }
        }
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn polarization_is_rank_one_transverse_and_normalized() {
        let momentum = MasslessSpinors::from_momentum([5.0, 3.0, 0.0, 4.0]).unwrap();
        let reference = MasslessSpinors::from_momentum([5.0, -3.0, 0.0, -4.0]).unwrap();
        let plus = massless_polarization_bispinor(momentum, reference, 1).unwrap();
        let minus = massless_polarization_bispinor(momentum, reference, -1).unwrap();
        assert_close(bispinor_dot(plus, momentum.bispinor()), c(0.0, 0.0));
        assert_close(bispinor_dot(minus, momentum.bispinor()), c(0.0, 0.0));
        assert_close(bispinor_dot(plus, plus), c(0.0, 0.0));
        assert_close(bispinor_dot(minus, minus), c(0.0, 0.0));
        assert_close(bispinor_dot(plus, minus), c(-1.0, 0.0));
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn weyl_vector_vertices_match_component_witnesses() {
        let spinor = [c(1.25, -0.5), c(-2.0, 3.0)];
        let vector = [c(0.75, 1.0), c(-4.0, 0.25), c(2.5, -3.0), c(1.5, 2.0)];
        let [l0, l1] = spinor;
        let [r0, r1, r2, r3] = vector;
        let positive_reference = [
            -I * l0 * r3 - I * l1 * r1 + l1 * r2 + I * l0 * r0,
            -l0 * r2 - I * l0 * r1 + I * l1 * r0 + I * l1 * r3,
        ];
        let negative_reference = [
            -l1 * r2 + I * l0 * r0 + I * l0 * r3 + I * l1 * r1,
            -I * l1 * r3 + l0 * r2 + I * l0 * r1 + I * l1 * r0,
        ];
        let matrix = vector_to_bispinor(vector);
        assert_array_close(weyl_vector_positive(spinor, matrix), positive_reference);
        assert_array_close(weyl_vector_negative(spinor, matrix), negative_reference);
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn four_gluon_dag_sums_only_the_six_mhv_roots() {
        let dag = build_complete_gluon_tree_spinor_dag(4).unwrap();
        let point = [
            [-5.0, 0.0, 0.0, -5.0],
            [-5.0, 0.0, 0.0, 5.0],
            [5.0, 3.0, 0.0, 4.0],
            [5.0, -3.0, 0.0, -4.0],
        ];
        let evaluation = dag.evaluate(&point).unwrap();
        assert!(evaluation.helicity_sum > 0.0);
        assert_eq!(
            evaluation
                .amplitudes
                .iter()
                .filter(|amplitude| amplitude.norm_sqr() > 1.0e-24)
                .count(),
            6
        );
    }
}
