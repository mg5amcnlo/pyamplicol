// SPDX-License-Identifier: 0BSD

//! Deterministic executable-payload codec for spinor DAG v2.
//!
//! The payload owns the two pieces of runtime information which cannot be
//! recovered from the scalar expression graph: the mapping from graph source
//! slots to public momentum inputs and the mapping from dense DAG parameter
//! slots to the authenticated prepared-parameter domain.  Artifact identity is
//! the ordinary hash of these bytes; this format deliberately carries no
//! second digest or provenance manifest.

use super::{
    MassiveSpinorSource, SpinorAmplitudeRoot, SpinorBracketKind, SpinorDag, SpinorKinematicScalar,
    SpinorNode, SpinorNodeId, SpinorRewriteStats,
};
use crate::recurrence::{ExactComplexRational, ExactRational};
use crate::{RusticolError, RusticolResult};
use std::collections::HashSet;

const MAGIC: &[u8; 8] = b"PACSPDG2";
const VERSION: u32 = 2;
const HEADER_BYTES: usize = 30;
const SOURCE_BINDING_BYTES: usize = 4;
const PARAMETER_BINDING_BYTES: usize = 4;
const MINIMUM_NODE_BYTES: usize = 2;
const MAX_PAYLOAD_BYTES: usize = 256 * 1024 * 1024;
// At the native lane's 1,024-point tile size this permits at most 4 GiB of
// complex node planes.  The independent u16 spinor-atom domain permits at
// most another 4 GiB, leaving headroom below the 10 GiB execution bound.
const MAX_NODE_COUNT: usize = 1 << 18;
const MAX_OPERAND_COUNT: usize = 4 * MAX_NODE_COUNT;
const MAX_ROOT_COUNT: usize = 1 << 20;

/// Semantic ABI of the graph-backed spinor execution payload.
pub const SPINOR_DAG_BINARY_ABI: &str = "pyamplicol-spinor-dag-binary-v2";

/// Runtime representation required by one graph momentum source.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum SpinorSourceInputKind {
    /// A null momentum which must be factorized into one spinor pair.
    NullSpinor = 0,
    /// A timelike momentum represented by two deterministically derived null
    /// spinor pairs.
    MassiveSpinorPair = 1,
    /// A momentum-only source.  No spinor factorization is permitted or
    /// required; this is the source layout used by external scalars.
    MomentumOnly = 2,
}

impl SpinorSourceInputKind {
    fn decode(value: u8) -> RusticolResult<Self> {
        match value {
            0 => Ok(Self::NullSpinor),
            1 => Ok(Self::MassiveSpinorPair),
            2 => Ok(Self::MomentumOnly),
            _ => Err(artifact(format!(
                "source input kind {value} is outside the v2 domain"
            ))),
        }
    }
}

/// Input binding for one graph source slot.
///
/// The row's position in [`SpinorDagPayloadV2::source_inputs`] is the dense
/// graph source slot. `public_source_slot` addresses the caller-visible
/// momentum input. `momentum_sign` converts that input into the graph's
/// all-outgoing convention.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SpinorSourceInputBinding {
    public_source_slot: u16,
    momentum_sign: i8,
    kind: SpinorSourceInputKind,
}

impl SpinorSourceInputBinding {
    pub fn new(
        public_source_slot: u16,
        momentum_sign: i8,
        kind: SpinorSourceInputKind,
    ) -> RusticolResult<Self> {
        if !matches!(momentum_sign, -1 | 1) {
            return Err(input(format!(
                "source momentum sign must be -1 or +1, received {momentum_sign}"
            )));
        }
        Ok(Self {
            public_source_slot,
            momentum_sign,
            kind,
        })
    }

    pub const fn public_source_slot(self) -> u16 {
        self.public_source_slot
    }

    pub const fn momentum_sign(self) -> i8 {
        self.momentum_sign
    }

    pub const fn kind(self) -> SpinorSourceInputKind {
        self.kind
    }
}

/// Binding from one dense DAG parameter slot to a prepared runtime slot.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SpinorPreparedParameterBinding {
    prepared_parameter_slot: u32,
}

impl SpinorPreparedParameterBinding {
    pub const fn new(prepared_parameter_slot: u32) -> Self {
        Self {
            prepared_parameter_slot,
        }
    }

    pub const fn prepared_parameter_slot(self) -> u32 {
        self.prepared_parameter_slot
    }
}

/// A self-contained executable graph and its authenticated-runtime bindings.
#[derive(Clone, Debug)]
pub struct SpinorDagPayloadV2 {
    dag: SpinorDag,
    source_inputs: Box<[SpinorSourceInputBinding]>,
    prepared_parameter_count: u32,
    parameter_bindings: Box<[SpinorPreparedParameterBinding]>,
}

impl SpinorDagPayloadV2 {
    pub fn new(
        dag: SpinorDag,
        source_inputs: Vec<SpinorSourceInputBinding>,
        prepared_parameter_count: u32,
        parameter_bindings: Vec<SpinorPreparedParameterBinding>,
    ) -> RusticolResult<Self> {
        let payload = Self {
            dag,
            source_inputs: source_inputs.into_boxed_slice(),
            prepared_parameter_count,
            parameter_bindings: parameter_bindings.into_boxed_slice(),
        };
        validate_payload(&payload, ValidationBoundary::Input)?;
        Ok(payload)
    }

    pub const fn dag(&self) -> &SpinorDag {
        &self.dag
    }

    pub fn source_inputs(&self) -> &[SpinorSourceInputBinding] {
        &self.source_inputs
    }

    pub const fn prepared_parameter_count(&self) -> u32 {
        self.prepared_parameter_count
    }

    pub fn parameter_bindings(&self) -> &[SpinorPreparedParameterBinding] {
        &self.parameter_bindings
    }

    pub fn validate(&self) -> RusticolResult<()> {
        validate_payload(self, ValidationBoundary::Input).map(|_| ())
    }

    pub fn into_parts(
        self,
    ) -> (
        SpinorDag,
        Box<[SpinorSourceInputBinding]>,
        u32,
        Box<[SpinorPreparedParameterBinding]>,
    ) {
        (
            self.dag,
            self.source_inputs,
            self.prepared_parameter_count,
            self.parameter_bindings,
        )
    }
}

/// Encode one validated payload into its canonical little-endian v2 bytes.
pub fn encode_spinor_dag_v2(payload: &SpinorDagPayloadV2) -> RusticolResult<Vec<u8>> {
    let encoded_bytes = validate_payload(payload, ValidationBoundary::Input)?;

    let node_count = u32::try_from(payload.dag.nodes.len())
        .map_err(|_| input("node count exceeds the u32 ID domain"))?;
    let root_count = u32::try_from(payload.dag.roots.len())
        .map_err(|_| input("root count exceeds the u32 format domain"))?;
    let mut writer = Writer::new(encoded_bytes)?;
    writer.raw(MAGIC);
    writer.u32(VERSION);
    writer.u16(payload.dag.momentum_count);
    writer.u16(payload.dag.parameter_count);
    writer.u32(payload.prepared_parameter_count);
    writer.u32(node_count);
    writer.u32(root_count);
    writer.u16(payload.dag.temporal_reference_source.unwrap_or(u16::MAX));

    for binding in payload.source_inputs.iter().copied() {
        writer.u16(binding.public_source_slot);
        writer.i8(binding.momentum_sign);
        writer.u8(binding.kind as u8);
    }
    for binding in payload.parameter_bindings.iter().copied() {
        writer.u32(binding.prepared_parameter_slot);
    }
    for node in payload.dag.nodes.iter() {
        encode_node(&mut writer, node)?;
    }
    for root in payload.dag.roots.iter() {
        for helicity in root.helicities.iter().copied() {
            writer.i8(helicity);
        }
        writer.u32(root.amplitude);
        writer.u16(root.multiplicity);
    }
    writer.finish()
}

/// Decode and fully validate one v2 executable spinor payload.
pub fn decode_spinor_dag_v2(bytes: &[u8]) -> RusticolResult<SpinorDagPayloadV2> {
    if bytes.len() > MAX_PAYLOAD_BYTES {
        return Err(artifact(format!(
            "payload contains {} bytes, exceeding the {MAX_PAYLOAD_BYTES}-byte execution limit",
            bytes.len()
        )));
    }
    let mut reader = Reader::new(bytes);
    if reader.take(8, "magic")? != MAGIC {
        return Err(artifact(
            "unsupported spinor payload magic; regenerate the v2 artifact",
        ));
    }
    if reader.u32("version")? != VERSION {
        return Err(artifact(
            "unsupported spinor payload version; regenerate the v2 artifact",
        ));
    }
    let momentum_count = reader.u16("momentum source count")?;
    let parameter_count = reader.u16("DAG parameter count")?;
    let prepared_parameter_count = reader.u32("prepared parameter count")?;
    let node_count = usize::try_from(reader.u32("node count")?)
        .map_err(|_| artifact("node count exceeds usize"))?;
    let root_count = usize::try_from(reader.u32("root count")?)
        .map_err(|_| artifact("root count exceeds usize"))?;
    let temporal_reference_source = match reader.u16("temporal reference source")? {
        u16::MAX => None,
        source => Some(source),
    };

    validate_count_limits(node_count, root_count, ValidationBoundary::Artifact)?;
    if momentum_count < 2 {
        return Err(artifact(
            "spinor DAG requires at least two momentum sources",
        ));
    }
    if root_count == 0 {
        return Err(artifact("spinor DAG requires at least one amplitude root"));
    }

    let source_bytes = usize::from(momentum_count)
        .checked_mul(SOURCE_BINDING_BYTES)
        .ok_or_else(|| artifact("source binding byte count overflows usize"))?;
    let parameter_bytes = usize::from(parameter_count)
        .checked_mul(PARAMETER_BINDING_BYTES)
        .ok_or_else(|| artifact("parameter binding byte count overflows usize"))?;
    let root_row_bytes = usize::from(momentum_count)
        .checked_add(6)
        .ok_or_else(|| artifact("root row byte count overflows usize"))?;
    let root_bytes = root_count
        .checked_mul(root_row_bytes)
        .ok_or_else(|| artifact("root byte count overflows usize"))?;
    let minimum_node_bytes = node_count
        .checked_mul(MINIMUM_NODE_BYTES)
        .ok_or_else(|| artifact("minimum node byte count overflows usize"))?;
    let minimum_remaining = source_bytes
        .checked_add(parameter_bytes)
        .and_then(|value| value.checked_add(root_bytes))
        .and_then(|value| value.checked_add(minimum_node_bytes))
        .ok_or_else(|| artifact("payload section byte count overflows usize"))?;
    // This check happens before any root or per-root helicity allocation.  In
    // particular, a large `momentum_count * root_count` must fit in the
    // already capped input byte slice before either root container is made.
    if minimum_remaining > reader.remaining() {
        return Err(artifact(
            "declared source, parameter, node, and root counts cannot fit in the payload",
        ));
    }

    let mut source_inputs = Vec::new();
    reserve(
        &mut source_inputs,
        usize::from(momentum_count),
        "source bindings",
    )?;
    for _ in 0..momentum_count {
        source_inputs.push(SpinorSourceInputBinding {
            public_source_slot: reader.u16("public source slot")?,
            momentum_sign: reader.i8("source momentum sign")?,
            kind: SpinorSourceInputKind::decode(reader.u8("source input kind")?)?,
        });
    }

    let mut parameter_bindings = Vec::new();
    reserve(
        &mut parameter_bindings,
        usize::from(parameter_count),
        "parameter bindings",
    )?;
    for _ in 0..parameter_count {
        parameter_bindings.push(SpinorPreparedParameterBinding {
            prepared_parameter_slot: reader.u32("prepared parameter slot")?,
        });
    }

    // Source validation precedes node allocation because the source layout
    // determines the derived massive-spinor atom domain.
    validate_source_inputs(
        momentum_count,
        &source_inputs,
        temporal_reference_source,
        ValidationBoundary::Artifact,
    )?;
    validate_root_count(root_count, &source_inputs, ValidationBoundary::Artifact)?;
    validate_parameter_bindings(
        parameter_count,
        prepared_parameter_count,
        &parameter_bindings,
        ValidationBoundary::Artifact,
    )?;
    let (massive_sources, spinor_atom_count) =
        derive_massive_sources(momentum_count, &source_inputs, ValidationBoundary::Artifact)?;

    let mut nodes = Vec::new();
    reserve(&mut nodes, node_count, "nodes")?;
    let mut remaining_operands = MAX_OPERAND_COUNT;
    for node_index in 0..node_count {
        nodes.push(decode_node(
            &mut reader,
            node_index,
            &mut remaining_operands,
        )?);
    }

    let mut roots = Vec::new();
    reserve(&mut roots, root_count, "amplitude roots")?;
    for _ in 0..root_count {
        let helicity_bytes = reader.take(usize::from(momentum_count), "root helicities")?;
        let mut helicities = Vec::new();
        reserve(
            &mut helicities,
            usize::from(momentum_count),
            "root helicities",
        )?;
        helicities.extend(helicity_bytes.iter().copied().map(|value| value as i8));
        let helicities = helicities.into_boxed_slice();
        let amplitude = reader.u32("root amplitude")?;
        let multiplicity = reader.u16("root multiplicity")?;
        let structural_zero = nodes
            .get(usize::try_from(amplitude).unwrap_or(usize::MAX))
            .is_some_and(|node| matches!(node, SpinorNode::Constant(value) if value.is_zero()));
        roots.push(SpinorAmplitudeRoot {
            helicities,
            amplitude,
            structural_zero,
            multiplicity,
        });
    }
    reader.finish()?;

    let uses_reference_atom = nodes.iter().any(|node| {
        matches!(
            node,
            SpinorNode::Bracket { left, right, .. }
                if *left == spinor_atom_count || *right == spinor_atom_count
        )
    });
    let payload = SpinorDagPayloadV2 {
        dag: SpinorDag {
            momentum_count,
            spinor_atom_count,
            parameter_count,
            massive_sources: massive_sources.into_boxed_slice(),
            uses_reference_atom,
            temporal_reference_source,
            nodes: nodes.into_boxed_slice(),
            roots: roots.into_boxed_slice(),
            // Rewrite counters describe generation, not execution.  They are
            // intentionally absent from v2 artifact identity.
            rewrite_stats: SpinorRewriteStats::default(),
        },
        source_inputs: source_inputs.into_boxed_slice(),
        prepared_parameter_count,
        parameter_bindings: parameter_bindings.into_boxed_slice(),
    };
    let encoded_bytes = validate_payload(&payload, ValidationBoundary::Artifact)?;
    if encoded_bytes != bytes.len() {
        return Err(artifact(
            "decoded graph size does not match its canonical binary representation",
        ));
    }
    Ok(payload)
}

fn encode_node(writer: &mut Writer, node: &SpinorNode) -> RusticolResult<()> {
    match node {
        SpinorNode::Constant(value) => {
            writer.u8(0);
            writer.exact(*value);
        }
        SpinorNode::Parameter(index) => {
            writer.u8(1);
            writer.u16(*index);
        }
        SpinorNode::Kinematic(SpinorKinematicScalar::SqrtTwo) => {
            writer.u8(2);
            writer.u8(0);
        }
        SpinorNode::Kinematic(SpinorKinematicScalar::InverseMass { source }) => {
            writer.u8(2);
            writer.u8(1);
            writer.u16(*source);
        }
        SpinorNode::Bracket { kind, left, right } => {
            writer.u8(3);
            writer.u8(match kind {
                SpinorBracketKind::Angle => 0,
                SpinorBracketKind::Square => 1,
            });
            writer.u16(*left);
            writer.u16(*right);
        }
        SpinorNode::Sum(operands) => {
            writer.u8(4);
            writer.node_ids("sum operands", operands)?;
        }
        SpinorNode::Product(operands) => {
            writer.u8(5);
            writer.node_ids("product operands", operands)?;
        }
        SpinorNode::Reciprocal(operand) => {
            writer.u8(6);
            writer.u32(*operand);
        }
    }
    Ok(())
}

fn decode_node(
    reader: &mut Reader<'_>,
    node_index: usize,
    remaining_operands: &mut usize,
) -> RusticolResult<SpinorNode> {
    match reader.u8("node tag")? {
        0 => Ok(SpinorNode::Constant(reader.exact("constant node")?)),
        1 => Ok(SpinorNode::Parameter(reader.u16("parameter node")?)),
        2 => match reader.u8("kinematic scalar tag")? {
            0 => Ok(SpinorNode::Kinematic(SpinorKinematicScalar::SqrtTwo)),
            1 => Ok(SpinorNode::Kinematic(SpinorKinematicScalar::InverseMass {
                source: reader.u16("inverse-mass source")?,
            })),
            tag => Err(artifact(format!(
                "kinematic scalar tag {tag} is outside the v2 domain"
            ))),
        },
        3 => {
            let kind = match reader.u8("bracket kind")? {
                0 => SpinorBracketKind::Angle,
                1 => SpinorBracketKind::Square,
                value => {
                    return Err(artifact(format!(
                        "bracket kind {value} is outside the v2 domain"
                    )));
                }
            };
            Ok(SpinorNode::Bracket {
                kind,
                left: reader.u16("bracket left atom")?,
                right: reader.u16("bracket right atom")?,
            })
        }
        4 => Ok(SpinorNode::Sum(reader.node_ids(
            "sum operands",
            node_index,
            remaining_operands,
        )?)),
        5 => Ok(SpinorNode::Product(reader.node_ids(
            "product operands",
            node_index,
            remaining_operands,
        )?)),
        6 => Ok(SpinorNode::Reciprocal(reader.u32("reciprocal operand")?)),
        tag => Err(artifact(format!("node tag {tag} is outside the v2 domain"))),
    }
}

#[derive(Clone, Copy)]
enum ValidationBoundary {
    Input,
    Artifact,
}

impl ValidationBoundary {
    fn error(self, message: impl Into<String>) -> RusticolError {
        match self {
            Self::Input => input(message),
            Self::Artifact => artifact(message),
        }
    }
}

fn validate_payload(
    payload: &SpinorDagPayloadV2,
    boundary: ValidationBoundary,
) -> RusticolResult<usize> {
    let dag = &payload.dag;
    if dag.momentum_count < 2 {
        return Err(boundary.error("spinor DAG requires at least two momentum sources"));
    }
    validate_count_limits(dag.nodes.len(), dag.roots.len(), boundary)?;

    validate_source_inputs(
        dag.momentum_count,
        &payload.source_inputs,
        dag.temporal_reference_source,
        boundary,
    )?;
    validate_root_count(dag.roots.len(), &payload.source_inputs, boundary)?;
    validate_parameter_bindings(
        dag.parameter_count,
        payload.prepared_parameter_count,
        &payload.parameter_bindings,
        boundary,
    )?;
    let (expected_massive_sources, expected_spinor_atom_count) =
        derive_massive_sources(dag.momentum_count, &payload.source_inputs, boundary)?;
    if dag.spinor_atom_count != expected_spinor_atom_count
        || dag.massive_sources.as_ref() != expected_massive_sources.as_slice()
    {
        return Err(
            boundary.error("DAG massive-spinor atoms do not match its executable source layout")
        );
    }

    validate_nodes(
        dag,
        &payload.source_inputs,
        expected_spinor_atom_count,
        boundary,
    )?;
    validate_roots(dag, &payload.source_inputs, boundary)?;
    validate_all_nodes_live(dag, boundary)?;
    encoded_payload_size(payload, boundary)
}

fn validate_count_limits(
    node_count: usize,
    root_count: usize,
    boundary: ValidationBoundary,
) -> RusticolResult<()> {
    if node_count > MAX_NODE_COUNT {
        return Err(boundary.error(format!(
            "node count {node_count} exceeds the {MAX_NODE_COUNT}-node execution limit"
        )));
    }
    if root_count > MAX_ROOT_COUNT {
        return Err(boundary.error(format!(
            "root count {root_count} exceeds the {MAX_ROOT_COUNT}-root execution limit"
        )));
    }
    Ok(())
}

fn validate_root_count(
    root_count: usize,
    source_inputs: &[SpinorSourceInputBinding],
    boundary: ValidationBoundary,
) -> RusticolResult<()> {
    let maximum = source_inputs.iter().fold(1_usize, |count, binding| {
        let source_states = match binding.kind {
            SpinorSourceInputKind::NullSpinor => 2,
            SpinorSourceInputKind::MassiveSpinorPair => 3,
            SpinorSourceInputKind::MomentumOnly => 1,
        };
        count.saturating_mul(source_states).min(MAX_ROOT_COUNT)
    });
    if root_count > maximum {
        return Err(boundary.error(format!(
            "root count {root_count} exceeds the {maximum}-configuration source-helicity domain"
        )));
    }
    Ok(())
}

fn validate_source_inputs(
    momentum_count: u16,
    source_inputs: &[SpinorSourceInputBinding],
    temporal_reference_source: Option<u16>,
    boundary: ValidationBoundary,
) -> RusticolResult<()> {
    if source_inputs.len() != usize::from(momentum_count) {
        return Err(boundary.error(format!(
            "source layout contains {} bindings for {momentum_count} graph sources",
            source_inputs.len()
        )));
    }
    let mut seen = vec![false; usize::from(momentum_count)];
    for (graph_slot, binding) in source_inputs.iter().copied().enumerate() {
        if !matches!(binding.momentum_sign, -1 | 1) {
            return Err(boundary.error(format!(
                "graph source {graph_slot} has momentum sign {}, expected -1 or +1",
                binding.momentum_sign
            )));
        }
        let public_slot = usize::from(binding.public_source_slot);
        let Some(observed) = seen.get_mut(public_slot) else {
            return Err(boundary.error(format!(
                "graph source {graph_slot} references public source {} outside the {momentum_count}-source domain",
                binding.public_source_slot
            )));
        };
        if *observed {
            return Err(boundary.error(format!(
                "public source {} is bound more than once",
                binding.public_source_slot
            )));
        }
        *observed = true;
    }
    // In-range uniqueness over exactly `momentum_count` rows proves that the
    // public source slots form the complete permutation.
    if let Some(source) = temporal_reference_source {
        let Some(binding) = source_inputs.get(usize::from(source)) else {
            return Err(boundary.error(format!(
                "temporal reference source {source} is outside the {momentum_count}-source graph"
            )));
        };
        if binding.kind != SpinorSourceInputKind::NullSpinor {
            return Err(boundary.error(format!(
                "temporal reference source {source} must be a null-spinor source"
            )));
        }
    }
    Ok(())
}

fn validate_parameter_bindings(
    parameter_count: u16,
    prepared_parameter_count: u32,
    parameter_bindings: &[SpinorPreparedParameterBinding],
    boundary: ValidationBoundary,
) -> RusticolResult<()> {
    if parameter_bindings.len() != usize::from(parameter_count) {
        return Err(boundary.error(format!(
            "parameter layout contains {} bindings for {parameter_count} dense DAG parameters",
            parameter_bindings.len()
        )));
    }
    let mut observed_slots = HashSet::new();
    observed_slots
        .try_reserve(parameter_bindings.len())
        .map_err(|error| {
            boundary.error(format!(
                "could not allocate prepared-parameter validation set: {error}"
            ))
        })?;
    for (dag_slot, binding) in parameter_bindings.iter().copied().enumerate() {
        if binding.prepared_parameter_slot >= prepared_parameter_count {
            return Err(boundary.error(format!(
                "DAG parameter {dag_slot} references prepared parameter {} outside the {prepared_parameter_count}-slot domain",
                binding.prepared_parameter_slot
            )));
        }
        if !observed_slots.insert(binding.prepared_parameter_slot) {
            return Err(boundary.error(format!(
                "prepared parameter {} is bound by more than one dense DAG parameter",
                binding.prepared_parameter_slot
            )));
        }
    }
    Ok(())
}

fn derive_massive_sources(
    momentum_count: u16,
    source_inputs: &[SpinorSourceInputBinding],
    boundary: ValidationBoundary,
) -> RusticolResult<(Vec<MassiveSpinorSource>, u16)> {
    let mut atom_count = momentum_count;
    let mut massive_sources = Vec::new();
    massive_sources
        .try_reserve_exact(
            source_inputs
                .iter()
                .filter(|binding| binding.kind == SpinorSourceInputKind::MassiveSpinorPair)
                .count(),
        )
        .map_err(|error| {
            boundary.error(format!("could not allocate massive-source layout: {error}"))
        })?;
    for (graph_source, binding) in source_inputs.iter().copied().enumerate() {
        if binding.kind != SpinorSourceInputKind::MassiveSpinorPair {
            continue;
        }
        let source = u16::try_from(graph_source)
            .map_err(|_| boundary.error("graph source slot exceeds u16"))?;
        let k_atom = atom_count;
        let r_atom = atom_count
            .checked_add(1)
            .ok_or_else(|| boundary.error("massive spinor atom count overflows u16"))?;
        atom_count = atom_count
            .checked_add(2)
            .ok_or_else(|| boundary.error("massive spinor atom count overflows u16"))?;
        massive_sources.push(MassiveSpinorSource {
            source,
            k_atom,
            r_atom,
        });
    }
    Ok((massive_sources, atom_count))
}

fn validate_nodes(
    dag: &SpinorDag,
    source_inputs: &[SpinorSourceInputBinding],
    spinor_atom_count: u16,
    boundary: ValidationBoundary,
) -> RusticolResult<()> {
    let reference_atom = spinor_atom_count;
    let mut observed_reference = false;
    let mut observed_parameters = vec![false; usize::from(dag.parameter_count)];
    let mut operand_count = 0_usize;
    let mut observed_nodes = HashSet::new();
    observed_nodes
        .try_reserve(dag.nodes.len())
        .map_err(|error| {
            boundary.error(format!(
                "could not allocate canonical-node validation set: {error}"
            ))
        })?;
    for (node_index, node) in dag.nodes.iter().enumerate() {
        if !observed_nodes.insert(node) {
            return Err(boundary.error(format!(
                "node {node_index} duplicates an earlier interned node"
            )));
        }
        let current =
            u32::try_from(node_index).map_err(|_| boundary.error("node index exceeds u32"))?;
        match node {
            SpinorNode::Constant(_) => {}
            SpinorNode::Kinematic(SpinorKinematicScalar::SqrtTwo) => {}
            SpinorNode::Parameter(parameter) => {
                if *parameter >= dag.parameter_count {
                    return Err(boundary.error(format!(
                        "node {node_index} references DAG parameter {parameter} outside the {}-slot domain",
                        dag.parameter_count
                    )));
                }
                observed_parameters[usize::from(*parameter)] = true;
            }
            SpinorNode::Kinematic(SpinorKinematicScalar::InverseMass { source }) => {
                let kind = source_inputs.get(usize::from(*source)).map(|row| row.kind);
                if kind != Some(SpinorSourceInputKind::MassiveSpinorPair) {
                    return Err(boundary.error(format!(
                        "node {node_index} inverse-mass source {source} is not a massive-spinor source"
                    )));
                }
            }
            SpinorNode::Bracket { left, right, .. } => {
                validate_bracket_atom(
                    *left,
                    node_index,
                    "left",
                    source_inputs,
                    reference_atom,
                    boundary,
                )?;
                validate_bracket_atom(
                    *right,
                    node_index,
                    "right",
                    source_inputs,
                    reference_atom,
                    boundary,
                )?;
                if left >= right {
                    return Err(boundary.error(format!(
                        "node {node_index} bracket atoms are not in canonical increasing order"
                    )));
                }
                observed_reference |= *left == reference_atom || *right == reference_atom;
            }
            SpinorNode::Sum(operands) | SpinorNode::Product(operands) => {
                operand_count = operand_count
                    .checked_add(operands.len())
                    .ok_or_else(|| boundary.error("aggregate operand count overflows usize"))?;
                if operand_count > MAX_OPERAND_COUNT {
                    return Err(boundary.error(format!(
                        "aggregate operand count exceeds the {MAX_OPERAND_COUNT}-operand execution limit"
                    )));
                }
                if operands.len() < 2 {
                    return Err(boundary.error(format!(
                        "node {node_index} aggregate has fewer than two operands"
                    )));
                }
                for operand in operands.iter().copied() {
                    validate_prior_node(operand, current, node_index, boundary)?;
                }
                if operands.windows(2).any(|pair| pair[0] > pair[1]) {
                    return Err(boundary.error(format!(
                        "node {node_index} aggregate operands are not canonically ordered"
                    )));
                }
                if matches!(node, SpinorNode::Sum(_))
                    && operands.windows(2).any(|pair| pair[0] == pair[1])
                {
                    return Err(boundary
                        .error(format!("node {node_index} sum contains a repeated operand")));
                }
                let mut constant_count = 0_usize;
                for operand in operands.iter().copied() {
                    let child = dag
                        .nodes
                        .get(usize::try_from(operand).unwrap_or(usize::MAX));
                    match (node, child) {
                        (SpinorNode::Sum(_), Some(SpinorNode::Sum(_)))
                        | (SpinorNode::Product(_), Some(SpinorNode::Product(_))) => {
                            return Err(boundary.error(format!(
                                "node {node_index} contains a non-flattened aggregate operand {operand}"
                            )));
                        }
                        _ => {}
                    }
                    if let Some(SpinorNode::Constant(value)) = child {
                        constant_count += 1;
                        if value.is_zero()
                            || (matches!(node, SpinorNode::Product(_))
                                && *value == ExactComplexRational::ONE)
                        {
                            return Err(boundary.error(format!(
                                "node {node_index} contains a folded aggregate constant"
                            )));
                        }
                    }
                    if let (SpinorNode::Product(_), Some(SpinorNode::Reciprocal(base))) =
                        (node, child)
                        && operands.binary_search(base).is_ok()
                    {
                        return Err(boundary.error(format!(
                            "node {node_index} contains an uncancelled reciprocal pair"
                        )));
                    }
                }
                if constant_count > 1 {
                    return Err(boundary.error(format!(
                        "node {node_index} contains more than one unfolded constant"
                    )));
                }
            }
            SpinorNode::Reciprocal(operand) => {
                validate_prior_node(*operand, current, node_index, boundary)?;
                if matches!(
                    dag.nodes
                        .get(usize::try_from(*operand).unwrap_or(usize::MAX)),
                    Some(SpinorNode::Constant(_) | SpinorNode::Reciprocal(_))
                ) {
                    return Err(boundary.error(format!(
                        "node {node_index} reciprocal operand was not canonically folded"
                    )));
                }
            }
        }
    }
    if dag.uses_reference_atom != observed_reference {
        return Err(boundary.error("DAG reference-spinor flag does not match its bracket nodes"));
    }
    if dag.temporal_reference_source.is_some() && !observed_reference {
        return Err(boundary
            .error("DAG declares a temporal reference source without using the reference atom"));
    }
    if let Some(parameter) = observed_parameters.iter().position(|observed| !observed) {
        return Err(boundary.error(format!(
            "dense DAG parameter {parameter} has no live parameter node"
        )));
    }
    Ok(())
}

fn validate_bracket_atom(
    atom: u16,
    node_index: usize,
    side: &str,
    source_inputs: &[SpinorSourceInputBinding],
    reference_atom: u16,
    boundary: ValidationBoundary,
) -> RusticolResult<()> {
    if atom > reference_atom {
        return Err(boundary.error(format!(
            "node {node_index} bracket {side} atom {atom} exceeds reference atom {reference_atom}"
        )));
    }
    if let Some(binding) = source_inputs.get(usize::from(atom)) {
        if binding.kind != SpinorSourceInputKind::NullSpinor {
            return Err(boundary.error(format!(
                "node {node_index} bracket {side} atom {atom} addresses a {:?} source",
                binding.kind
            )));
        }
    }
    Ok(())
}

fn validate_prior_node(
    operand: SpinorNodeId,
    current: SpinorNodeId,
    node_index: usize,
    boundary: ValidationBoundary,
) -> RusticolResult<()> {
    if operand >= current {
        return Err(boundary.error(format!(
            "node {node_index} operand {operand} is not topologically prior"
        )));
    }
    Ok(())
}

fn validate_roots(
    dag: &SpinorDag,
    source_inputs: &[SpinorSourceInputBinding],
    boundary: ValidationBoundary,
) -> RusticolResult<()> {
    if dag.roots.is_empty() {
        return Err(boundary.error("spinor DAG requires at least one amplitude root"));
    }
    let mut previous_helicities: Option<&[i8]> = None;
    for (root_index, root) in dag.roots.iter().enumerate() {
        if root.helicities.len() != usize::from(dag.momentum_count) {
            return Err(boundary.error(format!(
                "root {root_index} helicities must contain one entry per source"
            )));
        }
        for (source, (helicity, binding)) in root
            .helicities
            .iter()
            .copied()
            .zip(source_inputs.iter().copied())
            .enumerate()
        {
            let valid = match binding.kind {
                SpinorSourceInputKind::NullSpinor => matches!(helicity, -1 | 1),
                SpinorSourceInputKind::MassiveSpinorPair => matches!(helicity, -1 | 0 | 1),
                SpinorSourceInputKind::MomentumOnly => helicity == 0,
            };
            if !valid {
                return Err(boundary.error(format!(
                    "root {root_index} helicity {helicity} is invalid for {:?} source {source}",
                    binding.kind
                )));
            }
        }
        if let Some(previous) = previous_helicities {
            if previous >= root.helicities.as_ref() {
                return Err(
                    boundary.error("amplitude roots are not in unique canonical helicity order")
                );
            }
        }
        previous_helicities = Some(root.helicities.as_ref());
        let Some(amplitude) = dag
            .nodes
            .get(usize::try_from(root.amplitude).unwrap_or(usize::MAX))
        else {
            return Err(boundary.error(format!(
                "root {root_index} amplitude {} is outside the {}-node graph",
                root.amplitude,
                dag.nodes.len()
            )));
        };
        if root.multiplicity == 0 {
            return Err(boundary.error(format!("root {root_index} multiplicity must be positive")));
        }
        let expected_structural_zero =
            matches!(amplitude, SpinorNode::Constant(value) if value.is_zero());
        if root.structural_zero != expected_structural_zero {
            return Err(boundary.error(format!(
                "root {root_index} structural-zero flag does not match its amplitude"
            )));
        }
    }
    Ok(())
}

fn validate_all_nodes_live(dag: &SpinorDag, boundary: ValidationBoundary) -> RusticolResult<()> {
    let mut live = Vec::new();
    live.try_reserve_exact(dag.nodes.len()).map_err(|error| {
        boundary.error(format!("could not allocate node-liveness bitmap: {error}"))
    })?;
    live.resize(dag.nodes.len(), false);
    for root in &dag.roots {
        let index = usize::try_from(root.amplitude)
            .map_err(|_| boundary.error("root node ID exceeds usize"))?;
        let Some(observed) = live.get_mut(index) else {
            return Err(
                boundary.error(format!("root node {} is outside the graph", root.amplitude))
            );
        };
        *observed = true;
    }
    // Operand IDs are strictly smaller than their consumer.  A reverse scan
    // therefore propagates reachability without an attacker-sized work stack.
    for index in (0..dag.nodes.len()).rev() {
        if !live[index] {
            continue;
        }
        match &dag.nodes[index] {
            SpinorNode::Sum(operands) | SpinorNode::Product(operands) => {
                for operand in operands.iter().copied() {
                    let operand = usize::try_from(operand)
                        .map_err(|_| boundary.error("live operand ID exceeds usize"))?;
                    let Some(observed) = live.get_mut(operand) else {
                        return Err(
                            boundary.error(format!("live operand {operand} is outside the graph"))
                        );
                    };
                    *observed = true;
                }
            }
            SpinorNode::Reciprocal(operand) => {
                let operand = usize::try_from(*operand)
                    .map_err(|_| boundary.error("live operand ID exceeds usize"))?;
                let Some(observed) = live.get_mut(operand) else {
                    return Err(
                        boundary.error(format!("live operand {operand} is outside the graph"))
                    );
                };
                *observed = true;
            }
            SpinorNode::Constant(_)
            | SpinorNode::Parameter(_)
            | SpinorNode::Kinematic(_)
            | SpinorNode::Bracket { .. } => {}
        }
    }
    if let Some(dead) = live.iter().position(|entry| !entry) {
        return Err(boundary.error(format!(
            "node {dead} is unreachable from every amplitude root"
        )));
    }
    Ok(())
}

fn encoded_payload_size(
    payload: &SpinorDagPayloadV2,
    boundary: ValidationBoundary,
) -> RusticolResult<usize> {
    let mut size = HEADER_BYTES;
    let mut add = |bytes: usize, label: &str| -> RusticolResult<()> {
        size = size
            .checked_add(bytes)
            .ok_or_else(|| boundary.error(format!("{label} byte count overflows usize")))?;
        if size > MAX_PAYLOAD_BYTES {
            return Err(boundary.error(format!(
                "canonical payload size exceeds the {MAX_PAYLOAD_BYTES}-byte execution limit"
            )));
        }
        Ok(())
    };
    add(
        payload
            .source_inputs
            .len()
            .checked_mul(SOURCE_BINDING_BYTES)
            .ok_or_else(|| boundary.error("source binding byte count overflows usize"))?,
        "source bindings",
    )?;
    add(
        payload
            .parameter_bindings
            .len()
            .checked_mul(PARAMETER_BINDING_BYTES)
            .ok_or_else(|| boundary.error("parameter binding byte count overflows usize"))?,
        "parameter bindings",
    )?;
    for node in payload.dag.nodes.iter() {
        let node_bytes = match node {
            SpinorNode::Constant(_) => 65,
            SpinorNode::Parameter(_) => 3,
            SpinorNode::Kinematic(SpinorKinematicScalar::SqrtTwo) => 2,
            SpinorNode::Kinematic(SpinorKinematicScalar::InverseMass { .. }) => 4,
            SpinorNode::Bracket { .. } => 6,
            SpinorNode::Sum(operands) | SpinorNode::Product(operands) => 5_usize
                .checked_add(
                    operands
                        .len()
                        .checked_mul(4)
                        .ok_or_else(|| boundary.error("aggregate operand bytes overflow usize"))?,
                )
                .ok_or_else(|| boundary.error("aggregate node bytes overflow usize"))?,
            SpinorNode::Reciprocal(_) => 5,
        };
        add(node_bytes, "nodes")?;
    }
    let root_row_bytes = usize::from(payload.dag.momentum_count)
        .checked_add(6)
        .ok_or_else(|| boundary.error("root row byte count overflows usize"))?;
    add(
        payload
            .dag
            .roots
            .len()
            .checked_mul(root_row_bytes)
            .ok_or_else(|| boundary.error("root byte count overflows usize"))?,
        "roots",
    )?;
    Ok(size)
}

fn read_rational(reader: &mut Reader<'_>, label: &str) -> RusticolResult<ExactRational> {
    let numerator = reader.i128(&format!("{label} numerator"))?;
    let denominator = reader.i128(&format!("{label} denominator"))?;
    let value = ExactRational::new(numerator, denominator)
        .map_err(|error| artifact(format!("invalid {label}: {}", error.message())))?;
    if value.numerator() != numerator || value.denominator() != denominator {
        return Err(artifact(format!(
            "{label} is not in canonical reduced form"
        )));
    }
    Ok(value)
}

fn reserve<T>(values: &mut Vec<T>, count: usize, label: &str) -> RusticolResult<()> {
    values
        .try_reserve_exact(count)
        .map_err(|error| artifact(format!("could not allocate {label}: {error}")))
}

fn input(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("spinor DAG v2: {}", message.into()))
}

fn artifact(message: impl Into<String>) -> RusticolError {
    RusticolError::artifact(format!("spinor DAG v2: {}", message.into()))
}

struct Writer {
    bytes: Vec<u8>,
    expected_bytes: usize,
}

impl Writer {
    fn new(expected_bytes: usize) -> RusticolResult<Self> {
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(expected_bytes).map_err(|error| {
            input(format!(
                "could not allocate {expected_bytes} bytes for encoded payload: {error}"
            ))
        })?;
        Ok(Self {
            bytes,
            expected_bytes,
        })
    }

    fn finish(self) -> RusticolResult<Vec<u8>> {
        if self.bytes.len() != self.expected_bytes {
            return Err(RusticolError::internal(format!(
                "spinor DAG v2 encoder produced {} bytes, expected {}",
                self.bytes.len(),
                self.expected_bytes
            )));
        }
        Ok(self.bytes)
    }

    fn raw(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn i8(&mut self, value: i8) {
        self.u8(value as u8);
    }

    fn u16(&mut self, value: u16) {
        self.raw(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    fn i128(&mut self, value: i128) {
        self.raw(&value.to_le_bytes());
    }

    fn exact(&mut self, value: ExactComplexRational) {
        for rational in [value.real(), value.imag()] {
            self.i128(rational.numerator());
            self.i128(rational.denominator());
        }
    }

    fn node_ids(&mut self, label: &str, values: &[SpinorNodeId]) -> RusticolResult<()> {
        let count =
            u32::try_from(values.len()).map_err(|_| input(format!("{label} count exceeds u32")))?;
        self.u32(count);
        for value in values.iter().copied() {
            self.u32(value);
        }
        Ok(())
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn take(&mut self, count: usize, label: &str) -> RusticolResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| artifact(format!("{label} offset overflows usize")))?;
        let bytes = self.bytes.get(self.offset..end).ok_or_else(|| {
            artifact(format!(
                "truncated {label} at byte {}: need {count}, have {}",
                self.offset,
                self.remaining()
            ))
        })?;
        self.offset = end;
        Ok(bytes)
    }

    fn u8(&mut self, label: &str) -> RusticolResult<u8> {
        Ok(self.take(1, label)?[0])
    }

    fn i8(&mut self, label: &str) -> RusticolResult<i8> {
        Ok(self.u8(label)? as i8)
    }

    fn u16(&mut self, label: &str) -> RusticolResult<u16> {
        Ok(u16::from_le_bytes(
            self.take(2, label)?.try_into().expect("checked read"),
        ))
    }

    fn u32(&mut self, label: &str) -> RusticolResult<u32> {
        Ok(u32::from_le_bytes(
            self.take(4, label)?.try_into().expect("checked read"),
        ))
    }

    fn i128(&mut self, label: &str) -> RusticolResult<i128> {
        Ok(i128::from_le_bytes(
            self.take(16, label)?.try_into().expect("checked read"),
        ))
    }

    fn exact(&mut self, label: &str) -> RusticolResult<ExactComplexRational> {
        Ok(ExactComplexRational::new(
            read_rational(self, &format!("{label} real"))?,
            read_rational(self, &format!("{label} imaginary"))?,
        ))
    }

    fn node_ids(
        &mut self,
        label: &str,
        node_index: usize,
        remaining_operands: &mut usize,
    ) -> RusticolResult<Box<[SpinorNodeId]>> {
        let count = usize::try_from(self.u32(&format!("{label} count"))?)
            .map_err(|_| artifact(format!("{label} count exceeds usize")))?;
        if count > self.remaining() / 4 {
            return Err(artifact(format!(
                "{label} count cannot fit in the remaining payload"
            )));
        }
        if count < 2 {
            return Err(artifact(format!(
                "node {node_index} {label} has fewer than two entries"
            )));
        }
        *remaining_operands = remaining_operands.checked_sub(count).ok_or_else(|| {
            artifact(format!(
                "aggregate operand count exceeds the {MAX_OPERAND_COUNT}-operand execution limit"
            ))
        })?;
        let mut values = Vec::new();
        reserve(&mut values, count, label)?;
        for _ in 0..count {
            values.push(self.u32(label)?);
        }
        Ok(values.into_boxed_slice())
    }

    fn finish(self) -> RusticolResult<()> {
        if self.offset != self.bytes.len() {
            return Err(artifact(format!(
                "payload contains {} trailing bytes",
                self.bytes.len() - self.offset
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spinor::{
        SpinorDagBuilder, build_helicity_summed_quark_z_gluon_spinor_dag,
        build_optimized_helicity_summed_gluon_spinor_dag,
    };

    fn binding(
        public_source_slot: u16,
        momentum_sign: i8,
        kind: SpinorSourceInputKind,
    ) -> SpinorSourceInputBinding {
        SpinorSourceInputBinding::new(public_source_slot, momentum_sign, kind).unwrap()
    }

    fn gluon_payload() -> SpinorDagPayloadV2 {
        let dag = build_optimized_helicity_summed_gluon_spinor_dag(4).unwrap();
        SpinorDagPayloadV2::new(
            dag,
            vec![
                binding(2, 1, SpinorSourceInputKind::NullSpinor),
                binding(0, -1, SpinorSourceInputKind::NullSpinor),
                binding(3, 1, SpinorSourceInputKind::NullSpinor),
                binding(1, -1, SpinorSourceInputKind::NullSpinor),
            ],
            3,
            Vec::new(),
        )
        .unwrap()
    }

    fn q_z_payload() -> SpinorDagPayloadV2 {
        let dag = build_helicity_summed_quark_z_gluon_spinor_dag(&[0, 1], 2).unwrap();
        SpinorDagPayloadV2::new(
            dag,
            vec![
                binding(1, 1, SpinorSourceInputKind::NullSpinor),
                binding(2, 1, SpinorSourceInputKind::NullSpinor),
                binding(0, -1, SpinorSourceInputKind::MassiveSpinorPair),
            ],
            7,
            vec![
                SpinorPreparedParameterBinding::new(6),
                SpinorPreparedParameterBinding::new(2),
            ],
        )
        .unwrap()
    }

    fn momentum_only_payload() -> SpinorDagPayloadV2 {
        let mut builder = SpinorDagBuilder::new(2).unwrap();
        let one = builder.one();
        builder
            .add_root(vec![0_i8, 0_i8].into_boxed_slice(), one)
            .unwrap();
        SpinorDagPayloadV2::new(
            builder.finish().unwrap(),
            vec![
                binding(0, -1, SpinorSourceInputKind::MomentumOnly),
                binding(1, 1, SpinorSourceInputKind::MomentumOnly),
            ],
            0,
            Vec::new(),
        )
        .unwrap()
    }

    fn null_payload_without_reference() -> SpinorDagPayloadV2 {
        let mut builder = SpinorDagBuilder::new(2).unwrap();
        let one = builder.one();
        builder
            .add_root(vec![-1_i8, -1_i8].into_boxed_slice(), one)
            .unwrap();
        SpinorDagPayloadV2::new(
            builder.finish().unwrap(),
            vec![
                binding(0, -1, SpinorSourceInputKind::NullSpinor),
                binding(1, 1, SpinorSourceInputKind::NullSpinor),
            ],
            0,
            Vec::new(),
        )
        .unwrap()
    }

    fn temporal_reference_payload() -> SpinorDagPayloadV2 {
        let mut builder = SpinorDagBuilder::new(2).unwrap();
        builder.use_temporal_reference_source(1).unwrap();
        let amplitude = builder.angle(0, builder.reference_atom()).unwrap();
        builder
            .add_root(vec![-1_i8, -1_i8].into_boxed_slice(), amplitude)
            .unwrap();
        SpinorDagPayloadV2::new(
            builder.finish().unwrap(),
            vec![
                binding(0, -1, SpinorSourceInputKind::NullSpinor),
                binding(1, 1, SpinorSourceInputKind::NullSpinor),
            ],
            0,
            Vec::new(),
        )
        .unwrap()
    }

    #[test]
    fn v2_encoding_is_deterministic_and_round_trips_executable_semantics() {
        let payload = q_z_payload();
        let first = encode_spinor_dag_v2(&payload).unwrap();
        let second = encode_spinor_dag_v2(&payload).unwrap();
        assert_eq!(first, second);

        let decoded = decode_spinor_dag_v2(&first).unwrap();
        assert_eq!(decoded.source_inputs(), payload.source_inputs());
        assert_eq!(
            decoded.prepared_parameter_count(),
            payload.prepared_parameter_count()
        );
        assert_eq!(decoded.parameter_bindings(), payload.parameter_bindings());
        assert_eq!(decoded.dag().momentum_count, payload.dag().momentum_count);
        assert_eq!(decoded.dag().parameter_count, payload.dag().parameter_count);
        assert_eq!(
            decoded.dag().spinor_atom_count,
            payload.dag().spinor_atom_count
        );
        assert_eq!(decoded.dag().massive_sources, payload.dag().massive_sources);
        assert_eq!(
            decoded.dag().uses_reference_atom,
            payload.dag().uses_reference_atom
        );
        assert_eq!(
            decoded.dag().temporal_reference_source,
            payload.dag().temporal_reference_source
        );
        assert_eq!(decoded.dag().nodes, payload.dag().nodes);
        assert_eq!(decoded.dag().roots, payload.dag().roots);
        assert_eq!(decoded.dag().census(), payload.dag().census());
        assert_eq!(decoded.dag().rewrite_stats(), SpinorRewriteStats::default());
    }

    #[test]
    fn temporal_reference_source_round_trips_in_the_canonical_header() {
        let payload = temporal_reference_payload();
        assert!(payload.dag().uses_reference_atom());

        let encoded = encode_spinor_dag_v2(&payload).unwrap();
        assert_eq!(&encoded[28..HEADER_BYTES], &1_u16.to_le_bytes());
        let decoded = decode_spinor_dag_v2(&encoded).unwrap();
        assert_eq!(decoded.dag().temporal_reference_source, Some(1));
        assert_eq!(encode_spinor_dag_v2(&decoded).unwrap(), encoded);
    }

    #[test]
    fn decoder_rejects_invalid_temporal_reference_sources() {
        let mut encoded = encode_spinor_dag_v2(&temporal_reference_payload()).unwrap();
        encoded[28..HEADER_BYTES].copy_from_slice(&2_u16.to_le_bytes());
        let error = decode_spinor_dag_v2(&encoded).unwrap_err();
        assert!(error.message().contains("outside the 2-source graph"));

        encoded[28..HEADER_BYTES].copy_from_slice(&1_u16.to_le_bytes());
        encoded[HEADER_BYTES + SOURCE_BINDING_BYTES + 3] =
            SpinorSourceInputKind::MomentumOnly as u8;
        let error = decode_spinor_dag_v2(&encoded).unwrap_err();
        assert!(error.message().contains("must be a null-spinor source"));

        let mut unused = encode_spinor_dag_v2(&null_payload_without_reference()).unwrap();
        unused[28..HEADER_BYTES].copy_from_slice(&0_u16.to_le_bytes());
        let error = decode_spinor_dag_v2(&unused).unwrap_err();
        assert!(error.message().contains("without using the reference atom"));
    }

    #[test]
    fn decoder_rejects_truncation_and_trailing_bytes() {
        let encoded = encode_spinor_dag_v2(&gluon_payload()).unwrap();
        assert!(decode_spinor_dag_v2(&encoded[..encoded.len() - 1]).is_err());

        let mut trailing = encoded;
        trailing.push(0);
        assert!(decode_spinor_dag_v2(&trailing).is_err());
    }

    #[test]
    fn decoder_rejects_invalid_source_permutation_and_kind() {
        let payload = q_z_payload();
        let encoded = encode_spinor_dag_v2(&payload).unwrap();

        let mut repeated = encoded.clone();
        // Header is followed by fixed four-byte source rows.  Repeat the
        // first public source slot in the second row.
        let repeated_slot = repeated[HEADER_BYTES..HEADER_BYTES + 2].to_vec();
        repeated[HEADER_BYTES + SOURCE_BINDING_BYTES..HEADER_BYTES + SOURCE_BINDING_BYTES + 2]
            .copy_from_slice(&repeated_slot);
        assert!(decode_spinor_dag_v2(&repeated).is_err());

        let mut wrong_kind = encoded;
        wrong_kind[HEADER_BYTES + 2 * SOURCE_BINDING_BYTES + 3] =
            SpinorSourceInputKind::MomentumOnly as u8;
        assert!(decode_spinor_dag_v2(&wrong_kind).is_err());
    }

    #[test]
    fn decoder_rejects_parameter_binding_outside_prepared_domain() {
        let payload = q_z_payload();
        let mut encoded = encode_spinor_dag_v2(&payload).unwrap();
        let first_parameter = HEADER_BYTES + payload.source_inputs().len() * SOURCE_BINDING_BYTES;
        encoded[first_parameter..first_parameter + 4]
            .copy_from_slice(&payload.prepared_parameter_count().to_le_bytes());
        assert!(decode_spinor_dag_v2(&encoded).is_err());
    }

    #[test]
    fn decoder_rejects_noncanonical_exact_rationals() {
        let payload = gluon_payload();
        let mut encoded = encode_spinor_dag_v2(&payload).unwrap();
        let constant = first_constant_start(&encoded, &payload);
        encoded[constant + 1..constant + 17].copy_from_slice(&2_i128.to_le_bytes());
        encoded[constant + 17..constant + 33].copy_from_slice(&2_i128.to_le_bytes());
        let error = decode_spinor_dag_v2(&encoded).unwrap_err();
        assert!(error.message().contains("canonical reduced form"));
    }

    #[test]
    fn decoder_rejects_duplicate_prepared_parameter_bindings() {
        let payload = q_z_payload();
        let mut encoded = encode_spinor_dag_v2(&payload).unwrap();
        let first_parameter = HEADER_BYTES + payload.source_inputs().len() * SOURCE_BINDING_BYTES;
        let first_binding = encoded[first_parameter..first_parameter + 4].to_vec();
        encoded[first_parameter + 4..first_parameter + 8].copy_from_slice(&first_binding);
        let error = decode_spinor_dag_v2(&encoded).unwrap_err();
        assert!(
            error
                .message()
                .contains("more than one dense DAG parameter")
        );
    }

    #[test]
    fn decoder_rejects_counts_above_execution_limits_before_allocating() {
        let encoded = encode_spinor_dag_v2(&gluon_payload()).unwrap();

        let mut excessive_nodes = encoded.clone();
        excessive_nodes[20..24]
            .copy_from_slice(&(u32::try_from(MAX_NODE_COUNT).unwrap() + 1).to_le_bytes());
        let error = decode_spinor_dag_v2(&excessive_nodes).unwrap_err();
        assert!(error.message().contains("node execution limit"));

        let mut excessive_roots = encoded;
        excessive_roots[24..28]
            .copy_from_slice(&(u32::try_from(MAX_ROOT_COUNT).unwrap() + 1).to_le_bytes());
        let error = decode_spinor_dag_v2(&excessive_roots).unwrap_err();
        assert!(error.message().contains("root execution limit"));
    }

    #[test]
    fn decoder_enforces_source_kind_helicity_domains() {
        let null_payload = gluon_payload();
        let mut null_encoded = encode_spinor_dag_v2(&null_payload).unwrap();
        let root_start = roots_start(&null_encoded, &null_payload);
        null_encoded[root_start] = 0;
        let error = decode_spinor_dag_v2(&null_encoded).unwrap_err();
        assert!(error.message().contains("invalid for NullSpinor"));

        let momentum_payload = momentum_only_payload();
        let mut momentum_encoded = encode_spinor_dag_v2(&momentum_payload).unwrap();
        let root_start = roots_start(&momentum_encoded, &momentum_payload);
        momentum_encoded[root_start] = 1;
        let error = decode_spinor_dag_v2(&momentum_encoded).unwrap_err();
        assert!(error.message().contains("invalid for MomentumOnly"));
    }

    #[test]
    fn payload_rejects_duplicate_interned_nodes() {
        let dag = SpinorDag {
            momentum_count: 2,
            spinor_atom_count: 2,
            parameter_count: 0,
            massive_sources: Vec::new().into_boxed_slice(),
            uses_reference_atom: false,
            temporal_reference_source: None,
            nodes: vec![
                SpinorNode::Constant(ExactComplexRational::ZERO),
                SpinorNode::Constant(ExactComplexRational::ZERO),
            ]
            .into_boxed_slice(),
            roots: vec![
                SpinorAmplitudeRoot {
                    helicities: vec![-1, -1].into_boxed_slice(),
                    amplitude: 0,
                    structural_zero: true,
                    multiplicity: 1,
                },
                SpinorAmplitudeRoot {
                    helicities: vec![-1, 1].into_boxed_slice(),
                    amplitude: 1,
                    structural_zero: true,
                    multiplicity: 1,
                },
            ]
            .into_boxed_slice(),
            rewrite_stats: SpinorRewriteStats::default(),
        };
        let error = SpinorDagPayloadV2::new(
            dag,
            vec![
                binding(0, 1, SpinorSourceInputKind::NullSpinor),
                binding(1, 1, SpinorSourceInputKind::NullSpinor),
            ],
            0,
            Vec::new(),
        )
        .unwrap_err();
        assert!(
            error
                .message()
                .contains("duplicates an earlier interned node")
        );
    }

    #[test]
    fn decoder_rejects_non_topological_operand() {
        let payload = gluon_payload();
        let mut encoded = encode_spinor_dag_v2(&payload).unwrap();
        let node_start = HEADER_BYTES
            + payload.source_inputs().len() * SOURCE_BINDING_BYTES
            + payload.parameter_bindings().len() * PARAMETER_BINDING_BYTES;
        let (operand_offset, node_index) = first_aggregate_operand(&encoded, node_start);
        encoded[operand_offset..operand_offset + 4]
            .copy_from_slice(&u32::try_from(node_index).unwrap().to_le_bytes());
        assert!(decode_spinor_dag_v2(&encoded).is_err());
    }

    fn first_aggregate_operand(bytes: &[u8], mut offset: usize) -> (usize, usize) {
        let node_count = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
        for node_index in 0..node_count {
            let tag = bytes[offset];
            offset += 1;
            match tag {
                0 => offset += 64,
                1 => offset += 2,
                2 => {
                    let scalar = bytes[offset];
                    offset += if scalar == 0 { 1 } else { 3 };
                }
                3 => offset += 5,
                4 | 5 => {
                    let count =
                        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
                    offset += 4;
                    if count != 0 {
                        return (offset, node_index);
                    }
                    offset += count * 4;
                }
                6 => offset += 4,
                _ => panic!("encoder emitted an unknown node tag"),
            }
        }
        panic!("test graph contains no aggregate node")
    }

    fn roots_start(bytes: &[u8], payload: &SpinorDagPayloadV2) -> usize {
        let mut offset = HEADER_BYTES
            + payload.source_inputs().len() * SOURCE_BINDING_BYTES
            + payload.parameter_bindings().len() * PARAMETER_BINDING_BYTES;
        let node_count = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
        for _ in 0..node_count {
            let tag = bytes[offset];
            offset += 1;
            match tag {
                0 => offset += 64,
                1 => offset += 2,
                2 => {
                    let scalar = bytes[offset];
                    offset += if scalar == 0 { 1 } else { 3 };
                }
                3 => offset += 5,
                4 | 5 => {
                    let count =
                        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
                    offset += 4 + count * 4;
                }
                6 => offset += 4,
                _ => panic!("encoder emitted an unknown node tag"),
            }
        }
        offset
    }

    fn first_constant_start(bytes: &[u8], payload: &SpinorDagPayloadV2) -> usize {
        let mut offset = HEADER_BYTES
            + payload.source_inputs().len() * SOURCE_BINDING_BYTES
            + payload.parameter_bindings().len() * PARAMETER_BINDING_BYTES;
        let node_count = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
        for _ in 0..node_count {
            let start = offset;
            let tag = bytes[offset];
            offset += 1;
            match tag {
                0 => return start,
                1 => offset += 2,
                2 => {
                    let scalar = bytes[offset];
                    offset += if scalar == 0 { 1 } else { 3 };
                }
                3 => offset += 5,
                4 | 5 => {
                    let count =
                        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
                    offset += 4 + count * 4;
                }
                6 => offset += 4,
                _ => panic!("encoder emitted an unknown node tag"),
            }
        }
        panic!("test graph contains no constant node")
    }
}
