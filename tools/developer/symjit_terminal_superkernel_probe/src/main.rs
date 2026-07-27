// SPDX-License-Identifier: 0BSD
//! Disposable direct-execution harness for the compiled terminal superkernel.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{anyhow, bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use symjit::{
    Application, Config, Defuns, DirectApplication, DirectApplicationMetadata, DirectCallable,
    DirectDestinationOperation, DirectInputBinding, DirectInputSnapshot, DirectOutputScale,
    DirectPlane, DirectScalar, Storage, DIRECT_APPLICATION_STORAGE_ABI, DIRECT_NO_ALIAS,
    DIRECT_STATUS_OK,
};

const REQUEST_KIND: &str = "pyamplicol-terminal-direct-runner-request";
const RESULT_KIND: &str = "pyamplicol-terminal-direct-runner-result";
const LEAF_BUNDLE_KIND: &str = "pyamplicol-terminal-superkernel-leaf-bundle";
const SOURCE_APPLICATION_ABI: &str = "symjit-application-storage-v3";
const SCHEMA_VERSION: u32 = 1;
const STACK_LIMIT_BYTES: usize = 1 << 20;
const EXPECTED_BATCHES: [usize; 2] = [128, 1024];
const EXPECTED_STAGE_ORDINALS: [usize; 13] = [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7];

struct AllocationCounter;

static TRACK_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static TRACKED_ALLOCATION_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for AllocationCounter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            TRACKED_ALLOCATION_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            TRACKED_ALLOCATION_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, old: Layout, new_size: usize) -> *mut u8 {
        if TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            TRACKED_ALLOCATION_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.realloc(ptr, old, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: AllocationCounter = AllocationCounter;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum CandidateKind {
    Pair,
    FullTail,
}

impl CandidateKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Pair => "pair",
            Self::FullTail => "full-tail",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum InputKind {
    Value,
    Momentum,
    ModelParameter,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
enum OutputArena {
    Current,
    Amplitude,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Request {
    kind: String,
    schema_version: u32,
    process: String,
    selected_flow: String,
    batches: Vec<usize>,
    tile_size: usize,
    samples: usize,
    sample_seconds: f64,
    rtol: f64,
    atol: f64,
    baseline: LeafBundle,
    arena_shape: ArenaShape,
    schedules: Schedules,
    candidates: CandidateRecords,
    content_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ArenaShape {
    value_component_count: usize,
    current_component_count: usize,
    amplitude_component_count: usize,
    momentum_scalar_component_count: usize,
    momentum_form_count: usize,
    model_parameter_count: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LeafBundle {
    kind: String,
    schema_version: u32,
    process: String,
    process_id: String,
    selected_flow: String,
    source_application_abi: String,
    direct_application_abi: String,
    selected_lane_proof: SelectedLaneProof,
    arena_shape: ArenaShape,
    baseline_leaf_count: usize,
    baseline_leaves: Vec<BaselineLeaf>,
    content_sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SelectedLaneProof {
    status: String,
    materialized_sector_id: usize,
    selected_flow: String,
    reduction_group_count: usize,
    all_groups_exact_selected_flow: bool,
    runtime_selector_boundary_in_tail: bool,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BaselineLeaf {
    leaf_index: usize,
    stage_ordinal: usize,
    stage_index: usize,
    stage_kind: String,
    stage_leaf_index: usize,
    source_application: BaselineSource,
    logical_inputs: Vec<BaselineInput>,
    outputs: Vec<BaselineOutput>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BaselineSource {
    path: String,
    logical_path: String,
    sha256: String,
    size_bytes: usize,
    abi: String,
    optimization_level: u8,
    direct_codegen_optimization_level: u8,
    direct_application_abi: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BaselineInput {
    leaf_parameter_index: usize,
    stage_parameter_index: usize,
    kind: InputKind,
    source_id: usize,
    component: usize,
    global_component: usize,
    real_valued: bool,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BaselineOutput {
    leaf_output_index: usize,
    stage_output_index: usize,
    arena: OutputArena,
    component: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CandidateRecords {
    pair: CandidateRecord,
    #[serde(rename = "full-tail")]
    full_tail: CandidateRecord,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CandidateRecord {
    kind: CandidateKind,
    source_application: CandidateSource,
    logical_inputs: Vec<CandidateInput>,
    outputs: Vec<CandidateOutput>,
    elided_stage_indices: Vec<usize>,
    dependency_components: Vec<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CandidateSource {
    path: String,
    sha256: String,
    size_bytes: usize,
    abi: String,
    optimization_level: u8,
    direct_application_abi: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CandidateInput {
    parameter_index: usize,
    kind: InputKind,
    source_id: usize,
    component: usize,
    global_component: usize,
    real_valued: bool,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CandidateOutput {
    output_index: usize,
    arena: OutputArena,
    component: usize,
    value_slot_id: i64,
    current_id: i64,
    variant: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Schedules {
    baseline: Vec<ScheduleRef>,
    pair: Vec<ScheduleRef>,
    #[serde(rename = "full-tail")]
    full_tail: Vec<ScheduleRef>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "source", deny_unknown_fields)]
enum ScheduleRef {
    #[serde(rename = "baseline")]
    Baseline { leaf_index: usize },
    #[serde(rename = "candidate")]
    Candidate { kind: CandidateKind },
}

#[derive(Clone, Debug)]
struct LogicalInput {
    kind: InputKind,
    source_id: usize,
    global_component: usize,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct LogicalOutput {
    arena: OutputArena,
    component: usize,
}

#[derive(Clone, Debug)]
struct SourceRecord {
    path: PathBuf,
    sha256: String,
    size_bytes: usize,
    optimization_level: u8,
}

#[derive(Clone, Debug)]
struct KernelSpec {
    label: String,
    source: SourceRecord,
    inputs: Vec<LogicalInput>,
    outputs: Vec<LogicalOutput>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum PlaneBinding {
    CurrentReal(usize),
    CurrentImaginary(usize),
    Momentum(usize),
    Zero,
    AmplitudeReal(usize),
    AmplitudeImaginary(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScalarBinding {
    ModelReal(usize),
    ModelImaginary(usize),
}

#[derive(Debug)]
struct KernelBindingPlan {
    input_planes: Box<[PlaneBinding]>,
    input_scalars: Box<[ScalarBinding]>,
    output_planes: Box<[PlaneBinding]>,
    logical_input_count: usize,
}

struct Kernel {
    callable: DirectCallable,
    plan: KernelBindingPlan,
    source_stack_bytes: usize,
    configured_stack_limit_bytes: usize,
    simd_lane_width: usize,
}

struct KernelCatalog {
    baseline: Vec<Kernel>,
    pair: Kernel,
    full_tail: Kernel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CallId {
    Baseline(usize),
    Pair,
    FullTail,
}

#[derive(Debug)]
struct ValidatedSchedules {
    baseline: Box<[CallId]>,
    pair: Box<[CallId]>,
    full_tail: Box<[CallId]>,
}

#[derive(Debug)]
struct Arena {
    logical_batch: usize,
    point_stride: usize,
    current_real: Box<[f64]>,
    current_imaginary: Box<[f64]>,
    amplitude_real: Box<[f64]>,
    amplitude_imaginary: Box<[f64]>,
    momenta: Box<[f64]>,
    zero: Box<[f64]>,
    model_real: Box<[f64]>,
    model_imaginary: Box<[f64]>,
}

#[derive(Debug)]
struct BoundKernel {
    planes: Box<[DirectPlane]>,
    scalars: Box<[DirectScalar]>,
}

#[derive(Debug)]
struct BoundCatalog {
    baseline: Vec<BoundKernel>,
    pair: BoundKernel,
    full_tail: BoundKernel,
}

struct BatchRuntime {
    arena: Arena,
    bound: BoundCatalog,
    tile_size: usize,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct ResultBody {
    kind: &'static str,
    schema_version: u32,
    request_content_sha256: String,
    candidates: CandidateResults,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct CandidateResults {
    pair: CandidateEvidence,
    #[serde(rename = "full-tail")]
    full_tail: CandidateEvidence,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct CandidateEvidence {
    lowering: LoweringEvidence,
    numerical: NumericalEvidence,
    benchmarks: BTreeMap<String, BenchmarkEvidence>,
    projection: ProjectionEvidence,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct LoweringEvidence {
    status: &'static str,
    source_stack_bytes: usize,
    lowered_stack_bytes: Option<usize>,
    configured_stack_limit_bytes: usize,
    stack_limit_enforced: bool,
    warmed_arena_allocation_bytes: usize,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct NumericalEvidence {
    status: &'static str,
    point_count: usize,
    max_absolute_difference: f64,
    max_relative_difference: f64,
    rtol: f64,
    atol: f64,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct BenchmarkEvidence {
    baseline_samples_seconds_per_point: Vec<f64>,
    candidate_samples_seconds_per_point: Vec<f64>,
    baseline_median_seconds_per_point: f64,
    candidate_median_seconds_per_point: f64,
    speedup_fraction: f64,
    alternating_order: Vec<&'static str>,
    baseline_iterations: Vec<u64>,
    candidate_iterations: Vec<u64>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct ProjectionEvidence {
    baseline_call_count: usize,
    candidate_call_count: usize,
    baseline_input_plane_exposures: usize,
    candidate_input_plane_exposures: usize,
    baseline_output_plane_stores: usize,
    candidate_output_plane_stores: usize,
    baseline_logical_input_exposures: usize,
    candidate_logical_input_exposures: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let (request_path, output_path) = arguments()?;
    let raw = fs::read(&request_path)
        .with_context(|| format!("cannot read request {}", request_path.display()))?;
    let request: Request =
        serde_json::from_slice(&raw).context("request is not strict schema-v1 JSON")?;
    validate_request_digest(&request)?;
    let schedules = validate_request(&request)?;
    let specs = kernel_specs(&request)?;
    let kernels = load_kernels(&specs)?;
    let mut runtimes = EXPECTED_BATCHES
        .iter()
        .copied()
        .map(|batch| BatchRuntime::new(batch, request.tile_size, &request.arena_shape, &kernels))
        .collect::<Result<Vec<_>>>()?;

    let pair = assess_candidate(
        CandidateKind::Pair,
        &kernels,
        &schedules,
        &mut runtimes,
        &request,
    )?;
    let full_tail = assess_candidate(
        CandidateKind::FullTail,
        &kernels,
        &schedules,
        &mut runtimes,
        &request,
    )?;
    let body = ResultBody {
        kind: RESULT_KIND,
        schema_version: SCHEMA_VERSION,
        request_content_sha256: request.content_sha256.clone(),
        candidates: CandidateResults { pair, full_tail },
    };
    write_result(&output_path, &body)
}

fn arguments() -> Result<(PathBuf, PathBuf)> {
    let mut request = None;
    let mut output = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(argument) = args.next() {
        match argument.to_str() {
            Some("--request") => {
                ensure!(request.is_none(), "--request was supplied more than once");
                request = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow!("--request needs a path"))?,
                ));
            }
            Some("--output") => {
                ensure!(output.is_none(), "--output was supplied more than once");
                output = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow!("--output needs a path"))?,
                ));
            }
            _ => bail!("usage: symjit-terminal-superkernel-probe --request PATH --output PATH"),
        }
    }
    Ok((
        request.ok_or_else(|| anyhow!("--request is required"))?,
        output.ok_or_else(|| anyhow!("--output is required"))?,
    ))
}

fn validate_request_digest(request: &Request) -> Result<()> {
    validate_sha256(&request.content_sha256, "request content_sha256")?;
    let mut value = serde_json::to_value(request)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| anyhow!("serialized request is not an object"))?;
    object.remove("content_sha256");
    ensure!(
        sha256_hex(&python_canonical_json(&value)?) == request.content_sha256,
        "request content_sha256 mismatch"
    );

    validate_sha256(
        &request.baseline.content_sha256,
        "baseline bundle content_sha256",
    )?;
    let mut baseline = serde_json::to_value(&request.baseline)?;
    baseline
        .as_object_mut()
        .ok_or_else(|| anyhow!("serialized baseline is not an object"))?
        .remove("content_sha256");
    ensure!(
        sha256_hex(&python_canonical_json(&baseline)?) == request.baseline.content_sha256,
        "baseline bundle content_sha256 mismatch"
    );
    Ok(())
}

fn python_canonical_json(value: &Value) -> Result<Vec<u8>> {
    fn append(value: &Value, output: &mut Vec<u8>) -> Result<()> {
        match value {
            Value::Null => output.extend_from_slice(b"null"),
            Value::Bool(true) => output.extend_from_slice(b"true"),
            Value::Bool(false) => output.extend_from_slice(b"false"),
            Value::Number(number) => {
                if let Some(value) = number.as_i64() {
                    output.extend_from_slice(value.to_string().as_bytes());
                } else if let Some(value) = number.as_u64() {
                    output.extend_from_slice(value.to_string().as_bytes());
                } else {
                    let value = number
                        .as_f64()
                        .filter(|value| value.is_finite())
                        .ok_or_else(|| anyhow!("canonical JSON contains a non-finite number"))?;
                    output.extend_from_slice(python_float(value).as_bytes());
                }
            }
            Value::String(text) => serde_json::to_writer(output, text)?,
            Value::Array(values) => {
                output.push(b'[');
                for (index, item) in values.iter().enumerate() {
                    if index != 0 {
                        output.push(b',');
                    }
                    append(item, output)?;
                }
                output.push(b']');
            }
            Value::Object(values) => {
                output.push(b'{');
                let mut keys = values.keys().collect::<Vec<_>>();
                keys.sort_unstable();
                for (index, key) in keys.into_iter().enumerate() {
                    if index != 0 {
                        output.push(b',');
                    }
                    serde_json::to_writer(&mut *output, key)?;
                    output.push(b':');
                    append(&values[key], output)?;
                }
                output.push(b'}');
            }
        }
        Ok(())
    }

    let mut output = Vec::new();
    append(value, &mut output)?;
    Ok(output)
}

fn python_float(value: f64) -> String {
    let text = format!("{value:?}");
    let Some((mantissa, exponent)) = text.split_once('e') else {
        return text;
    };
    let parsed = exponent
        .parse::<i32>()
        .expect("Rust emitted a valid f64 exponent");
    format!("{mantissa}e{parsed:+03}")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{label} must be a lowercase SHA-256 digest"
    );
    Ok(())
}

fn validate_request(request: &Request) -> Result<ValidatedSchedules> {
    ensure!(request.kind == REQUEST_KIND, "request kind is incompatible");
    ensure!(
        request.schema_version == SCHEMA_VERSION,
        "request schema version is incompatible"
    );
    ensure!(!request.process.is_empty(), "request process is empty");
    ensure!(
        request.selected_flow.starts_with("flow:"),
        "request selected_flow is invalid"
    );
    ensure!(
        request.batches == EXPECTED_BATCHES,
        "probe batches must be exactly [128,1024]"
    );
    ensure!(request.tile_size == 32, "probe tile_size must be 32");
    ensure!(request.samples == 9, "probe samples must be 9");
    ensure!(
        request.sample_seconds.is_finite() && request.sample_seconds > 0.0,
        "sample_seconds must be finite and positive"
    );
    ensure!(
        request.rtol.is_finite()
            && request.atol.is_finite()
            && request.rtol >= 0.0
            && request.atol >= 0.0,
        "numerical tolerances must be finite and non-negative"
    );
    ensure!(
        request.rtol == 1.0e-12 && request.atol == 1.0e-15,
        "probe numerical tolerances differ from the fixed contract"
    );
    for batch in &request.batches {
        ensure!(
            batch % request.tile_size == 0,
            "batch {batch} is not divisible by tile_size"
        );
    }

    let shape = &request.arena_shape;
    ensure!(
        shape.value_component_count > 0
            && shape.current_component_count >= shape.value_component_count
            && shape.amplitude_component_count > 0,
        "arena component counts are inconsistent"
    );
    ensure!(
        shape.momentum_scalar_component_count > 0
            && shape.momentum_scalar_component_count % 4 == 0
            && shape.momentum_form_count == shape.momentum_scalar_component_count / 4,
        "momentum arena is not a dense collection of four-vectors"
    );

    let baseline = &request.baseline;
    ensure!(
        baseline.kind == LEAF_BUNDLE_KIND && baseline.schema_version == SCHEMA_VERSION,
        "baseline leaf bundle contract is incompatible"
    );
    ensure!(
        baseline.process == request.process
            && baseline.selected_flow == request.selected_flow
            && baseline.arena_shape == request.arena_shape,
        "baseline identity or arena shape disagrees with the request"
    );
    ensure!(
        baseline.source_application_abi == SOURCE_APPLICATION_ABI
            && baseline.direct_application_abi == DIRECT_APPLICATION_STORAGE_ABI,
        "baseline application ABI is incompatible"
    );
    ensure!(
        baseline.selected_lane_proof.status == "proven"
            && baseline.selected_lane_proof.materialized_sector_id == 0
            && baseline.selected_lane_proof.selected_flow == request.selected_flow
            && baseline.selected_lane_proof.reduction_group_count > 0
            && baseline.selected_lane_proof.all_groups_exact_selected_flow
            && !baseline
                .selected_lane_proof
                .runtime_selector_boundary_in_tail,
        "selected materialized-lane proof is incomplete"
    );
    ensure!(
        baseline.baseline_leaf_count == EXPECTED_STAGE_ORDINALS.len()
            && baseline.baseline_leaves.len() == EXPECTED_STAGE_ORDINALS.len(),
        "baseline must contain exactly thirteen leaves"
    );
    let mut stage_leaf_cursors = BTreeMap::<usize, usize>::new();
    for (index, leaf) in baseline.baseline_leaves.iter().enumerate() {
        ensure!(
            leaf.leaf_index == index,
            "baseline leaf indices are not dense"
        );
        ensure!(
            leaf.stage_ordinal == EXPECTED_STAGE_ORDINALS[index],
            "baseline stage ordinal mismatch at leaf {index}"
        );
        let cursor = stage_leaf_cursors.entry(leaf.stage_ordinal).or_default();
        ensure!(
            leaf.stage_leaf_index == *cursor,
            "baseline stage-local leaf indices are not dense"
        );
        *cursor += 1;
        ensure!(!leaf.stage_kind.is_empty(), "baseline stage kind is empty");
        validate_baseline_source(&leaf.source_application)?;
        validate_dense(
            leaf.logical_inputs
                .iter()
                .map(|input| input.leaf_parameter_index),
            leaf.logical_inputs.len(),
            "baseline leaf parameter indices",
        )?;
        let mut stage_parameters = BTreeSet::new();
        for input in &leaf.logical_inputs {
            ensure!(
                stage_parameters.insert(input.stage_parameter_index),
                "baseline leaf repeats a stage parameter"
            );
            validate_logical_input(
                input.kind,
                input.source_id,
                input.component,
                input.global_component,
                input.real_valued,
                shape,
            )?;
        }
        validate_dense(
            leaf.outputs.iter().map(|output| output.leaf_output_index),
            leaf.outputs.len(),
            "baseline leaf output indices",
        )?;
        for output in &leaf.outputs {
            validate_logical_output(output.arena, output.component, shape)?;
        }
    }

    validate_candidate(&request.candidates.pair, CandidateKind::Pair, shape)?;
    validate_candidate(
        &request.candidates.full_tail,
        CandidateKind::FullTail,
        shape,
    )?;
    ensure!(
        request.candidates.pair.elided_stage_indices
            == request.candidates.full_tail.elided_stage_indices,
        "pair and full-tail candidates disagree on elided stages"
    );
    ensure!(
        request.candidates.pair.elided_stage_indices == [6, 7],
        "candidates do not elide the exact terminal stage pair [6,7]"
    );

    let baseline_expected = (0..13).map(CallId::Baseline).collect::<Vec<_>>();
    let pair_expected = (0..8)
        .map(CallId::Baseline)
        .chain([CallId::Pair, CallId::Baseline(12)])
        .collect::<Vec<_>>();
    let full_expected = (0..8)
        .map(CallId::Baseline)
        .chain([CallId::FullTail])
        .collect::<Vec<_>>();
    let baseline_calls = schedule_ids(&request.schedules.baseline)?;
    let pair_calls = schedule_ids(&request.schedules.pair)?;
    let full_calls = schedule_ids(&request.schedules.full_tail)?;
    ensure!(
        baseline_calls == baseline_expected,
        "baseline schedule membership or order is not canonical"
    );
    ensure!(
        pair_calls == pair_expected,
        "pair schedule membership or order is not canonical"
    );
    ensure!(
        full_calls == full_expected,
        "full-tail schedule membership or order is not canonical"
    );
    Ok(ValidatedSchedules {
        baseline: baseline_calls.into_boxed_slice(),
        pair: pair_calls.into_boxed_slice(),
        full_tail: full_calls.into_boxed_slice(),
    })
}

fn validate_baseline_source(source: &BaselineSource) -> Result<()> {
    validate_source_fields(
        &source.path,
        &source.sha256,
        source.size_bytes,
        &source.abi,
        source.optimization_level,
        &source.direct_application_abi,
    )?;
    ensure!(
        !source.logical_path.is_empty(),
        "baseline logical path is empty"
    );
    ensure!(
        source.direct_codegen_optimization_level == 3,
        "baseline DirectApplication code generation is not O3"
    );
    Ok(())
}

fn validate_candidate(
    candidate: &CandidateRecord,
    expected: CandidateKind,
    shape: &ArenaShape,
) -> Result<()> {
    ensure!(
        candidate.kind == expected,
        "candidate map key and kind disagree"
    );
    validate_source_fields(
        &candidate.source_application.path,
        &candidate.source_application.sha256,
        candidate.source_application.size_bytes,
        &candidate.source_application.abi,
        candidate.source_application.optimization_level,
        &candidate.source_application.direct_application_abi,
    )?;
    validate_dense(
        candidate
            .logical_inputs
            .iter()
            .map(|input| input.parameter_index),
        candidate.logical_inputs.len(),
        "candidate parameter indices",
    )?;
    for input in &candidate.logical_inputs {
        validate_logical_input(
            input.kind,
            input.source_id,
            input.component,
            input.global_component,
            input.real_valued,
            shape,
        )?;
    }
    validate_dense(
        candidate.outputs.iter().map(|output| output.output_index),
        candidate.outputs.len(),
        "candidate output indices",
    )?;
    ensure!(!candidate.outputs.is_empty(), "candidate has no outputs");
    for output in &candidate.outputs {
        validate_logical_output(output.arena, output.component, shape)?;
        ensure!(
            !output.variant.is_empty(),
            "candidate output variant is empty"
        );
        match expected {
            CandidateKind::Pair => ensure!(
                output.arena == OutputArena::Current
                    && output.value_slot_id >= 0
                    && output.current_id >= 0,
                "pair candidate output is not a current component"
            ),
            CandidateKind::FullTail => ensure!(
                output.arena == OutputArena::Amplitude
                    && output.value_slot_id == -1
                    && output.current_id == -1,
                "full-tail candidate output is not an amplitude component"
            ),
        }
    }
    validate_sorted_unique(
        &candidate.elided_stage_indices,
        "candidate elided stage indices",
    )?;
    validate_sorted_unique(
        &candidate.dependency_components,
        "candidate dependency components",
    )?;
    ensure!(
        candidate
            .dependency_components
            .iter()
            .all(|component| *component < shape.value_component_count),
        "candidate dependency component is outside the value domain"
    );
    Ok(())
}

fn validate_source_fields(
    path: &str,
    sha256: &str,
    size_bytes: usize,
    abi: &str,
    optimization_level: u8,
    direct_application_abi: &str,
) -> Result<()> {
    ensure!(
        Path::new(path).is_absolute(),
        "source application path is not absolute"
    );
    ensure!(size_bytes > 0, "source application is empty");
    validate_sha256(sha256, "source application sha256")?;
    ensure!(
        abi == SOURCE_APPLICATION_ABI,
        "source application ABI is incompatible"
    );
    ensure!(
        optimization_level <= 3,
        "source optimization level is not O0 through O3"
    );
    ensure!(
        direct_application_abi == DIRECT_APPLICATION_STORAGE_ABI,
        "DirectApplication ABI is incompatible"
    );
    Ok(())
}

fn validate_logical_input(
    kind: InputKind,
    source_id: usize,
    component: usize,
    global_component: usize,
    real_valued: bool,
    shape: &ArenaShape,
) -> Result<()> {
    match kind {
        InputKind::Value => {
            ensure!(
                global_component < shape.value_component_count,
                "value input is outside the current component domain"
            );
        }
        InputKind::Momentum => {
            let local = global_component
                .checked_sub(shape.value_component_count)
                .filter(|local| *local < shape.momentum_scalar_component_count)
                .ok_or_else(|| anyhow!("momentum input is outside the momentum domain"))?;
            ensure!(
                local % 4 == component % 4,
                "momentum logical component disagrees with its global component"
            );
            ensure!(real_valued, "momentum input is not declared real-valued");
        }
        InputKind::ModelParameter => {
            ensure!(
                source_id < shape.model_parameter_count,
                "model parameter source_id is out of bounds"
            );
            let expected = shape
                .value_component_count
                .checked_add(shape.momentum_scalar_component_count)
                .and_then(|offset| offset.checked_add(source_id))
                .ok_or_else(|| anyhow!("model parameter global index overflows"))?;
            ensure!(
                global_component == expected && component == 0,
                "model parameter semantic identity is inconsistent"
            );
            ensure!(
                real_valued,
                "model parameter input is not declared real-valued"
            );
        }
    }
    Ok(())
}

fn validate_logical_output(arena: OutputArena, component: usize, shape: &ArenaShape) -> Result<()> {
    let count = match arena {
        OutputArena::Current => shape.current_component_count,
        OutputArena::Amplitude => shape.amplitude_component_count,
    };
    ensure!(component < count, "output component is outside its arena");
    Ok(())
}

fn validate_dense(values: impl Iterator<Item = usize>, length: usize, label: &str) -> Result<()> {
    ensure!(
        values.eq(0..length),
        "{label} must be dense, ordered, and zero-based"
    );
    Ok(())
}

fn validate_sorted_unique(values: &[usize], label: &str) -> Result<()> {
    ensure!(
        values.windows(2).all(|pair| pair[0] < pair[1]),
        "{label} must be sorted and unique"
    );
    Ok(())
}

fn schedule_ids(schedule: &[ScheduleRef]) -> Result<Vec<CallId>> {
    schedule
        .iter()
        .map(|entry| {
            Ok(match *entry {
                ScheduleRef::Baseline { leaf_index } => CallId::Baseline(leaf_index),
                ScheduleRef::Candidate {
                    kind: CandidateKind::Pair,
                } => CallId::Pair,
                ScheduleRef::Candidate {
                    kind: CandidateKind::FullTail,
                } => CallId::FullTail,
            })
        })
        .collect()
}

struct AllKernelSpecs {
    baseline: Vec<KernelSpec>,
    pair: KernelSpec,
    full_tail: KernelSpec,
}

fn kernel_specs(request: &Request) -> Result<AllKernelSpecs> {
    let baseline = request
        .baseline
        .baseline_leaves
        .iter()
        .map(|leaf| {
            Ok(KernelSpec {
                label: format!("baseline leaf {}", leaf.leaf_index),
                source: source_record(
                    &leaf.source_application.path,
                    &leaf.source_application.sha256,
                    leaf.source_application.size_bytes,
                    leaf.source_application.optimization_level,
                )?,
                inputs: leaf
                    .logical_inputs
                    .iter()
                    .map(|input| LogicalInput {
                        kind: input.kind,
                        source_id: input.source_id,
                        global_component: input.global_component,
                    })
                    .collect(),
                outputs: leaf
                    .outputs
                    .iter()
                    .map(|output| LogicalOutput {
                        arena: output.arena,
                        component: output.component,
                    })
                    .collect(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(AllKernelSpecs {
        baseline,
        pair: candidate_spec(&request.candidates.pair)?,
        full_tail: candidate_spec(&request.candidates.full_tail)?,
    })
}

fn candidate_spec(candidate: &CandidateRecord) -> Result<KernelSpec> {
    Ok(KernelSpec {
        label: format!("{} candidate", candidate.kind.label()),
        source: source_record(
            &candidate.source_application.path,
            &candidate.source_application.sha256,
            candidate.source_application.size_bytes,
            candidate.source_application.optimization_level,
        )?,
        inputs: candidate
            .logical_inputs
            .iter()
            .map(|input| LogicalInput {
                kind: input.kind,
                source_id: input.source_id,
                global_component: input.global_component,
            })
            .collect(),
        outputs: candidate
            .outputs
            .iter()
            .map(|output| LogicalOutput {
                arena: output.arena,
                component: output.component,
            })
            .collect(),
    })
}

fn source_record(
    path: &str,
    expected_sha256: &str,
    expected_size: usize,
    optimization_level: u8,
) -> Result<SourceRecord> {
    let path = PathBuf::from(path);
    let bytes = fs::read(&path)
        .with_context(|| format!("cannot read source application {}", path.display()))?;
    ensure!(
        bytes.len() == expected_size,
        "source application {} size mismatch",
        path.display()
    );
    ensure!(
        sha256_hex(&bytes) == expected_sha256,
        "source application {} digest mismatch",
        path.display()
    );
    Ok(SourceRecord {
        path,
        sha256: expected_sha256.to_owned(),
        size_bytes: expected_size,
        optimization_level,
    })
}

fn load_kernels(specs: &AllKernelSpecs) -> Result<KernelCatalog> {
    Ok(KernelCatalog {
        baseline: specs
            .baseline
            .iter()
            .map(load_kernel)
            .collect::<Result<Vec<_>>>()?,
        pair: load_kernel(&specs.pair)?,
        full_tail: load_kernel(&specs.full_tail)?,
    })
}

fn load_kernel(spec: &KernelSpec) -> Result<Kernel> {
    let bytes = fs::read(&spec.source.path).with_context(|| {
        format!(
            "cannot reload {} source {}",
            spec.label,
            spec.source.path.display()
        )
    })?;
    ensure!(
        bytes.len() == spec.source.size_bytes && sha256_hex(&bytes) == spec.source.sha256,
        "{} source changed after request validation",
        spec.label
    );
    let plan = binding_plan(spec)?;
    let mut parameter_bindings =
        Vec::with_capacity(plan.input_planes.len() + plan.input_scalars.len());
    let mut plane_index = 0_u32;
    let mut scalar_index = 0_u32;
    for input in &spec.inputs {
        match input.kind {
            InputKind::Value | InputKind::Momentum => {
                parameter_bindings.push(DirectInputBinding::Plane(plane_index));
                plane_index = plane_index
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("{} plane binding count overflows u32", spec.label))?;
                parameter_bindings.push(DirectInputBinding::Plane(plane_index));
                plane_index = plane_index
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("{} plane binding count overflows u32", spec.label))?;
            }
            InputKind::ModelParameter => {
                parameter_bindings.push(DirectInputBinding::Scalar(scalar_index));
                scalar_index = scalar_index
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("{} scalar binding count overflows u32", spec.label))?;
                parameter_bindings.push(DirectInputBinding::Scalar(scalar_index));
                scalar_index = scalar_index
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("{} scalar binding count overflows u32", spec.label))?;
            }
        }
    }
    ensure!(
        plane_index as usize == plan.input_planes.len()
            && scalar_index as usize == plan.input_scalars.len(),
        "{} binding expansion is inconsistent",
        spec.label
    );
    let output_count = spec
        .outputs
        .len()
        .checked_mul(2)
        .ok_or_else(|| anyhow!("{} output plane count overflows", spec.label))?;
    let metadata = DirectApplicationMetadata::new(
        DirectDestinationOperation::Overwrite,
        DirectInputSnapshot::Live,
        DirectOutputScale::Identity,
        Vec::new(),
        parameter_bindings,
        plane_index,
        scalar_index,
        vec![DIRECT_NO_ALIAS; output_count],
    )
    .with_context(|| format!("cannot describe {}", spec.label))?;

    let mut loader_config = Config::default();
    loader_config.set_defuns(Defuns::new());
    let mut input = bytes.as_slice();
    let mut source = Application::load(&mut input, &loader_config)
        .with_context(|| format!("cannot load {}", spec.label))?;
    ensure!(input.is_empty(), "{} source has trailing bytes", spec.label);
    ensure!(
        source.config.opt_level() == spec.source.optimization_level,
        "{} declared O{} but stores O{}",
        spec.label,
        spec.source.optimization_level,
        source.config.opt_level()
    );
    let source_stack_slots = source.prog.builder.stack_size();
    let source_stack_bytes = source_stack_slots
        .checked_mul(2 * std::mem::size_of::<f64>())
        .ok_or_else(|| anyhow!("{} source stack byte count overflows", spec.label))?;
    let configured_stack_limit_bytes = source.config.stack_limit();
    ensure!(
        configured_stack_limit_bytes <= STACK_LIMIT_BYTES,
        "{} source stack limit exceeds 1 MiB",
        spec.label
    );

    let mut direct = DirectApplication::new(source, metadata)
        .with_context(|| format!("cannot lower {} through DirectApplication", spec.label))?;
    ensure!(
        direct.source_optimization_level() == spec.source.optimization_level,
        "{} DirectApplication source level changed",
        spec.label
    );
    direct.prepare_simd();
    let applet = direct
        .seal()
        .with_context(|| format!("cannot seal {}", spec.label))?;
    let simd_lane_width = applet.simd_lane_width();
    ensure!(
        simd_lane_width > 0,
        "{} reports a zero SIMD lane width",
        spec.label
    );
    Ok(Kernel {
        callable: applet.into_callable(),
        plan,
        source_stack_bytes,
        configured_stack_limit_bytes,
        simd_lane_width,
    })
}

fn binding_plan(spec: &KernelSpec) -> Result<KernelBindingPlan> {
    let mut input_planes = Vec::new();
    let mut input_scalars = Vec::new();
    for input in &spec.inputs {
        match input.kind {
            InputKind::Value => {
                input_planes.push(PlaneBinding::CurrentReal(input.global_component));
                input_planes.push(PlaneBinding::CurrentImaginary(input.global_component));
            }
            InputKind::Momentum => {
                // This exactly mirrors rusticol-core's SymJIT
                // append_component_bindings: the complex source ABI retains an
                // imaginary slot even for a real momentum component.
                input_planes.push(PlaneBinding::Momentum(input.global_component));
                input_planes.push(PlaneBinding::Zero);
            }
            InputKind::ModelParameter => {
                // Likewise, source_id is the model scalar index and both
                // complex slots remain present regardless of real_valued.
                input_scalars.push(ScalarBinding::ModelReal(input.source_id));
                input_scalars.push(ScalarBinding::ModelImaginary(input.source_id));
            }
        }
    }
    let mut output_planes = Vec::new();
    let mut seen_outputs = BTreeSet::new();
    for output in &spec.outputs {
        ensure!(
            seen_outputs.insert(*output),
            "{} repeats an output component",
            spec.label
        );
        match output.arena {
            OutputArena::Current => {
                ensure!(
                    !input_planes.contains(&PlaneBinding::CurrentReal(output.component))
                        && !input_planes
                            .contains(&PlaneBinding::CurrentImaginary(output.component)),
                    "{} aliases a current input and output",
                    spec.label
                );
                output_planes.push(PlaneBinding::CurrentReal(output.component));
                output_planes.push(PlaneBinding::CurrentImaginary(output.component));
            }
            OutputArena::Amplitude => {
                output_planes.push(PlaneBinding::AmplitudeReal(output.component));
                output_planes.push(PlaneBinding::AmplitudeImaginary(output.component));
            }
        }
    }
    ensure!(
        !output_planes.is_empty(),
        "{} has no output planes",
        spec.label
    );
    Ok(KernelBindingPlan {
        input_planes: input_planes.into_boxed_slice(),
        input_scalars: input_scalars.into_boxed_slice(),
        output_planes: output_planes.into_boxed_slice(),
        logical_input_count: spec.inputs.len(),
    })
}

impl Arena {
    fn new(logical_batch: usize, point_stride: usize, shape: &ArenaShape) -> Result<Self> {
        fn zeroed(component_count: usize, point_stride: usize, label: &str) -> Result<Box<[f64]>> {
            let length = component_count
                .checked_mul(point_stride)
                .ok_or_else(|| anyhow!("{label} arena length overflows"))?;
            Ok(vec![0.0; length].into_boxed_slice())
        }

        ensure!(
            point_stride == 32
                && logical_batch >= point_stride
                && logical_batch % point_stride == 0,
            "persistent arena requires a 32-point stride and whole logical tiles"
        );
        let mut arena = Self {
            logical_batch,
            point_stride,
            current_real: zeroed(shape.current_component_count, point_stride, "current real")?,
            current_imaginary: zeroed(
                shape.current_component_count,
                point_stride,
                "current imaginary",
            )?,
            amplitude_real: zeroed(
                shape.amplitude_component_count,
                point_stride,
                "amplitude real",
            )?,
            amplitude_imaginary: zeroed(
                shape.amplitude_component_count,
                point_stride,
                "amplitude imaginary",
            )?,
            momenta: zeroed(
                shape.momentum_scalar_component_count,
                point_stride,
                "momentum",
            )?,
            zero: vec![0.0; point_stride].into_boxed_slice(),
            model_real: vec![0.0; shape.model_parameter_count].into_boxed_slice(),
            model_imaginary: vec![0.0; shape.model_parameter_count].into_boxed_slice(),
        };
        arena.reset_deterministic();
        Ok(arena)
    }

    fn reset_deterministic(&mut self) {
        fill_planes(&mut self.current_real, self.point_stride, 0x1000, 0.75, 0.5);
        fill_planes(
            &mut self.current_imaginary,
            self.point_stride,
            0x2000,
            -0.25,
            0.5,
        );
        fill_planes(
            &mut self.amplitude_real,
            self.point_stride,
            0x3000,
            -0.5,
            1.0,
        );
        fill_planes(
            &mut self.amplitude_imaginary,
            self.point_stride,
            0x4000,
            -0.5,
            1.0,
        );
        fill_planes(&mut self.momenta, self.point_stride, 0x5000, 0.75, 0.5);
        self.zero.fill(0.0);
        for (index, value) in self.model_real.iter_mut().enumerate() {
            *value = deterministic_unit(0x6000, index, 0) * 0.5 + 0.75;
        }
        // The selected contract declares model parameters real, while the
        // complex source ABI still binds the fixed imaginary scalar slots.
        self.model_imaginary.fill(0.0);
    }

    fn plane(&mut self, binding: PlaneBinding) -> Result<DirectPlane> {
        let (storage, component) = match binding {
            PlaneBinding::CurrentReal(component) => (&mut self.current_real, component),
            PlaneBinding::CurrentImaginary(component) => (&mut self.current_imaginary, component),
            PlaneBinding::Momentum(global_component) => {
                // The caller normalized the global momentum component before
                // reaching this resolver.
                (&mut self.momenta, global_component)
            }
            PlaneBinding::Zero => {
                return Ok(unsafe {
                    DirectPlane::from_raw_parts(self.zero.as_mut_ptr(), self.zero.len())
                });
            }
            PlaneBinding::AmplitudeReal(component) => (&mut self.amplitude_real, component),
            PlaneBinding::AmplitudeImaginary(component) => {
                (&mut self.amplitude_imaginary, component)
            }
        };
        let offset = component
            .checked_mul(self.point_stride)
            .ok_or_else(|| anyhow!("plane offset overflows"))?;
        ensure!(
            offset
                .checked_add(self.point_stride)
                .is_some_and(|stop| stop <= storage.len()),
            "plane binding is outside persistent storage"
        );
        // SAFETY: the checked component-major range covers exactly one fixed
        // 32-point tile plane, and every backing allocation remains pinned.
        Ok(unsafe {
            DirectPlane::from_raw_parts(storage.as_mut_ptr().add(offset), self.point_stride)
        })
    }

    fn scalar(&mut self, binding: ScalarBinding) -> Result<DirectScalar> {
        let (storage, index) = match binding {
            ScalarBinding::ModelReal(index) => (&mut self.model_real, index),
            ScalarBinding::ModelImaginary(index) => (&mut self.model_imaginary, index),
        };
        let value = storage
            .get(index)
            .ok_or_else(|| anyhow!("model scalar binding is out of bounds"))?;
        Ok(unsafe { DirectScalar::from_raw(std::ptr::from_ref(value)) })
    }

    const fn invocation_shape(&self) -> (usize, usize, usize) {
        (self.logical_batch / self.point_stride, 0, self.point_stride)
    }
}

fn fill_planes(storage: &mut [f64], point_stride: usize, salt: u64, offset: f64, scale: f64) {
    for (flat, value) in storage.iter_mut().enumerate() {
        let component = flat / point_stride;
        let point = flat % point_stride;
        *value = offset + scale * deterministic_unit(salt, component, point);
    }
}

fn deterministic_unit(salt: u64, component: usize, point: usize) -> f64 {
    let mut value = salt
        ^ (component as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (point as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    ((value >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

impl BoundKernel {
    fn new(arena: &mut Arena, plan: &KernelBindingPlan, shape: &ArenaShape) -> Result<Self> {
        let mut planes = Vec::with_capacity(plan.input_planes.len() + plan.output_planes.len());
        for binding in plan.input_planes.iter().copied() {
            let normalized = match binding {
                PlaneBinding::Momentum(global) => {
                    let local = global
                        .checked_sub(shape.value_component_count)
                        .filter(|local| *local < shape.momentum_scalar_component_count)
                        .ok_or_else(|| anyhow!("momentum plane binding is out of bounds"))?;
                    PlaneBinding::Momentum(local)
                }
                other => other,
            };
            planes.push(arena.plane(normalized)?);
        }
        for binding in plan.output_planes.iter().copied() {
            planes.push(arena.plane(binding)?);
        }
        let scalars = plan
            .input_scalars
            .iter()
            .copied()
            .map(|binding| arena.scalar(binding))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            planes: planes.into_boxed_slice(),
            scalars: scalars.into_boxed_slice(),
        })
    }
}

impl BatchRuntime {
    fn new(
        batch: usize,
        tile_size: usize,
        shape: &ArenaShape,
        kernels: &KernelCatalog,
    ) -> Result<Self> {
        let mut arena = Arena::new(batch, tile_size, shape)?;
        let baseline = kernels
            .baseline
            .iter()
            .map(|kernel| BoundKernel::new(&mut arena, &kernel.plan, shape))
            .collect::<Result<Vec<_>>>()?;
        let pair = BoundKernel::new(&mut arena, &kernels.pair.plan, shape)?;
        let full_tail = BoundKernel::new(&mut arena, &kernels.full_tail.plan, shape)?;
        Ok(Self {
            arena,
            bound: BoundCatalog {
                baseline,
                pair,
                full_tail,
            },
            tile_size,
        })
    }
}

fn execute_schedule(
    runtime: &BatchRuntime,
    kernels: &KernelCatalog,
    calls: &[CallId],
) -> Result<()> {
    let (tile_count, point_start, point_count) = runtime.arena.invocation_shape();
    ensure!(
        runtime.tile_size == point_count && point_start == 0,
        "runtime invocation geometry drifted from the fixed 32-point arena"
    );
    for _ in 0..tile_count {
        for call in calls {
            let (kernel, bound) =
                match *call {
                    CallId::Baseline(index) => (
                        kernels
                            .baseline
                            .get(index)
                            .ok_or_else(|| anyhow!("baseline kernel index is out of bounds"))?,
                        runtime.bound.baseline.get(index).ok_or_else(|| {
                            anyhow!("bound baseline kernel index is out of bounds")
                        })?,
                    ),
                    CallId::Pair => (&kernels.pair, &runtime.bound.pair),
                    CallId::FullTail => (&kernels.full_tail, &runtime.bound.full_tail),
                };
            ensure!(
                point_start % kernel.simd_lane_width == 0,
                "tile start is not aligned to a kernel SIMD lane"
            );
            // SAFETY: every descriptor was authenticated against a fixed
            // binding plan, all allocations remain pinned inside runtime, and
            // this nonempty tile is inside every declared plane.
            let status = unsafe {
                kernel.callable.invoke_unchecked(
                    &bound.planes,
                    &bound.scalars,
                    point_start,
                    point_count,
                )
            };
            ensure!(
                status == DIRECT_STATUS_OK,
                "DirectApplication call returned status {status}"
            );
        }
    }
    Ok(())
}

fn assess_candidate(
    kind: CandidateKind,
    kernels: &KernelCatalog,
    schedules: &ValidatedSchedules,
    runtimes: &mut [BatchRuntime],
    request: &Request,
) -> Result<CandidateEvidence> {
    let candidate_calls = match kind {
        CandidateKind::Pair => &schedules.pair,
        CandidateKind::FullTail => &schedules.full_tail,
    };
    let candidate_kernel = match kind {
        CandidateKind::Pair => &kernels.pair,
        CandidateKind::FullTail => &kernels.full_tail,
    };

    let numerical = numerical_evidence(
        kind,
        kernels,
        &schedules.baseline,
        candidate_calls,
        runtimes,
        request.rtol,
        request.atol,
    )?;

    let mut warmed_arena_allocation_bytes = 0_usize;
    for runtime in runtimes.iter_mut() {
        runtime.arena.reset_deterministic();
        execute_schedule(runtime, kernels, &schedules.baseline)?;
        runtime.arena.reset_deterministic();
        execute_schedule(runtime, kernels, candidate_calls)?;
        runtime.arena.reset_deterministic();
        let (result, allocated) = track_allocations(|| {
            execute_schedule(runtime, kernels, &schedules.baseline)?;
            execute_schedule(runtime, kernels, candidate_calls)
        });
        result?;
        warmed_arena_allocation_bytes = warmed_arena_allocation_bytes
            .checked_add(allocated)
            .ok_or_else(|| anyhow!("warmed allocation count overflows"))?;
    }

    let mut benchmarks = BTreeMap::new();
    for runtime in runtimes.iter_mut() {
        runtime.arena.reset_deterministic();
        let benchmark = benchmark_alternating(
            runtime,
            kernels,
            &schedules.baseline,
            candidate_calls,
            request.samples,
            request.sample_seconds,
        )?;
        benchmarks.insert(runtime.arena.logical_batch.to_string(), benchmark);
    }
    let projection = projection(kernels, &schedules.baseline, candidate_calls)?;
    Ok(CandidateEvidence {
        lowering: LoweringEvidence {
            status: "ok",
            source_stack_bytes: candidate_kernel.source_stack_bytes,
            // The pinned public DirectApplication API does not expose its
            // privately lowered frame. Successful lowering and SIMD
            // preparation above did enforce the configured 1 MiB ceiling.
            lowered_stack_bytes: None,
            configured_stack_limit_bytes: candidate_kernel.configured_stack_limit_bytes,
            stack_limit_enforced: candidate_kernel.configured_stack_limit_bytes
                <= STACK_LIMIT_BYTES,
            warmed_arena_allocation_bytes,
        },
        numerical,
        benchmarks,
        projection,
    })
}

fn track_allocations<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    TRACKED_ALLOCATION_BYTES.store(0, Ordering::SeqCst);
    TRACK_ALLOCATIONS.store(true, Ordering::SeqCst);
    let result = operation();
    TRACK_ALLOCATIONS.store(false, Ordering::SeqCst);
    let bytes = TRACKED_ALLOCATION_BYTES.load(Ordering::SeqCst);
    (result, bytes)
}

fn numerical_evidence(
    kind: CandidateKind,
    kernels: &KernelCatalog,
    baseline_calls: &[CallId],
    candidate_calls: &[CallId],
    runtimes: &mut [BatchRuntime],
    rtol: f64,
    atol: f64,
) -> Result<NumericalEvidence> {
    let candidate_kernel = match kind {
        CandidateKind::Pair => &kernels.pair,
        CandidateKind::FullTail => &kernels.full_tail,
    };
    let mut bindings = candidate_kernel.plan.output_planes.to_vec();
    if kind == CandidateKind::Pair {
        for binding in kernels.baseline[12].plan.output_planes.iter().copied() {
            if !bindings.contains(&binding) {
                bindings.push(binding);
            }
        }
    }

    let mut point_count = 0_usize;
    let mut max_absolute_difference = 0.0_f64;
    let mut max_relative_difference = 0.0_f64;
    for runtime in runtimes.iter_mut() {
        runtime.arena.reset_deterministic();
        execute_schedule(runtime, kernels, baseline_calls)?;
        let mut expected = Vec::with_capacity(bindings.len() * runtime.arena.point_stride);
        for binding in &bindings {
            for point in 0..runtime.arena.point_stride {
                expected.push(runtime.arena.value(*binding, point)?);
            }
        }

        runtime.arena.reset_deterministic();
        execute_schedule(runtime, kernels, candidate_calls)?;
        let mut cursor = 0_usize;
        for binding in &bindings {
            for point in 0..runtime.arena.point_stride {
                let reference = expected[cursor];
                let actual = runtime.arena.value(*binding, point)?;
                cursor += 1;
                ensure!(
                    reference.is_finite() && actual.is_finite(),
                    "{} candidate produced a non-finite numerical output",
                    kind.label()
                );
                let absolute = (actual - reference).abs();
                let relative = if reference == 0.0 {
                    if absolute == 0.0 {
                        0.0
                    } else {
                        f64::INFINITY
                    }
                } else {
                    absolute / reference.abs()
                };
                max_absolute_difference = max_absolute_difference.max(absolute);
                max_relative_difference = max_relative_difference.max(relative);
                let tolerance = numerical_tolerance(reference, rtol, atol);
                ensure!(
                    absolute <= tolerance,
                    "{} numerical mismatch at batch {}, point {}, plane {:?}: \
                     actual={actual:.17e}, reference={reference:.17e}, \
                     absolute={absolute:.17e}, tolerance={tolerance:.17e}, \
                     rtol={rtol:.17e}, atol={atol:.17e}",
                    kind.label(),
                    runtime.arena.logical_batch,
                    point,
                    binding
                );
            }
        }
        point_count = point_count
            .checked_add(runtime.arena.point_stride)
            .ok_or_else(|| anyhow!("numerical point count overflows"))?;
    }
    Ok(NumericalEvidence {
        status: "ok",
        point_count,
        max_absolute_difference,
        max_relative_difference,
        rtol,
        atol,
    })
}

fn numerical_tolerance(reference: f64, rtol: f64, atol: f64) -> f64 {
    atol + rtol * reference.abs()
}

impl Arena {
    fn value(&self, binding: PlaneBinding, point: usize) -> Result<f64> {
        ensure!(
            point < self.point_stride,
            "point index is outside the arena"
        );
        let (storage, component) = match binding {
            PlaneBinding::CurrentReal(component) => (&self.current_real, component),
            PlaneBinding::CurrentImaginary(component) => (&self.current_imaginary, component),
            PlaneBinding::AmplitudeReal(component) => (&self.amplitude_real, component),
            PlaneBinding::AmplitudeImaginary(component) => (&self.amplitude_imaginary, component),
            PlaneBinding::Momentum(component) => (&self.momenta, component),
            PlaneBinding::Zero => return Ok(0.0),
        };
        let index = component
            .checked_mul(self.point_stride)
            .and_then(|offset| offset.checked_add(point))
            .ok_or_else(|| anyhow!("arena value index overflows"))?;
        storage
            .get(index)
            .copied()
            .ok_or_else(|| anyhow!("arena value binding is out of bounds"))
    }
}

fn benchmark_alternating(
    runtime: &BatchRuntime,
    kernels: &KernelCatalog,
    baseline_calls: &[CallId],
    candidate_calls: &[CallId],
    samples: usize,
    sample_seconds: f64,
) -> Result<BenchmarkEvidence> {
    execute_schedule(runtime, kernels, baseline_calls)?;
    execute_schedule(runtime, kernels, candidate_calls)?;
    let mut baseline_samples = Vec::with_capacity(samples);
    let mut candidate_samples = Vec::with_capacity(samples);
    let mut alternating_order = Vec::with_capacity(samples);
    let mut baseline_iterations = Vec::with_capacity(samples);
    let mut candidate_iterations = Vec::with_capacity(samples);
    for sample in 0..samples {
        let baseline_first = sample % 2 == 0;
        alternating_order.push(if baseline_first {
            "baseline-first"
        } else {
            "candidate-first"
        });
        let ((baseline_seconds, baseline_count), (candidate_seconds, candidate_count)) =
            if baseline_first {
                (
                    time_schedule(runtime, kernels, baseline_calls, sample_seconds)?,
                    time_schedule(runtime, kernels, candidate_calls, sample_seconds)?,
                )
            } else {
                let candidate = time_schedule(runtime, kernels, candidate_calls, sample_seconds)?;
                let baseline = time_schedule(runtime, kernels, baseline_calls, sample_seconds)?;
                (baseline, candidate)
            };
        baseline_samples
            .push(baseline_seconds / (baseline_count as f64 * runtime.arena.logical_batch as f64));
        candidate_samples.push(
            candidate_seconds / (candidate_count as f64 * runtime.arena.logical_batch as f64),
        );
        baseline_iterations.push(baseline_count);
        candidate_iterations.push(candidate_count);
    }
    let baseline_median = median(&baseline_samples)?;
    let candidate_median = median(&candidate_samples)?;
    ensure!(
        baseline_median > 0.0 && candidate_median > 0.0,
        "timing medians must be positive"
    );
    Ok(BenchmarkEvidence {
        baseline_samples_seconds_per_point: baseline_samples,
        candidate_samples_seconds_per_point: candidate_samples,
        baseline_median_seconds_per_point: baseline_median,
        candidate_median_seconds_per_point: candidate_median,
        speedup_fraction: 1.0 - candidate_median / baseline_median,
        alternating_order,
        baseline_iterations,
        candidate_iterations,
    })
}

fn time_schedule(
    runtime: &BatchRuntime,
    kernels: &KernelCatalog,
    calls: &[CallId],
    sample_seconds: f64,
) -> Result<(f64, u64)> {
    let started = Instant::now();
    let mut iterations = 0_u64;
    loop {
        execute_schedule(runtime, kernels, calls)?;
        iterations = iterations
            .checked_add(1)
            .ok_or_else(|| anyhow!("timing iteration count overflows"))?;
        let elapsed = started.elapsed().as_secs_f64();
        if elapsed >= sample_seconds {
            return Ok((elapsed, iterations));
        }
    }
}

fn median(values: &[f64]) -> Result<f64> {
    ensure!(!values.is_empty(), "cannot compute an empty median");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "timing sample is non-finite"
    );
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let middle = ordered.len() / 2;
    Ok(if ordered.len() % 2 == 0 {
        (ordered[middle - 1] + ordered[middle]) * 0.5
    } else {
        ordered[middle]
    })
}

fn projection(
    kernels: &KernelCatalog,
    baseline_calls: &[CallId],
    candidate_calls: &[CallId],
) -> Result<ProjectionEvidence> {
    fn sum(
        kernels: &KernelCatalog,
        calls: &[CallId],
        select: impl Fn(&KernelBindingPlan) -> usize,
    ) -> Result<usize> {
        let mut result = 0_usize;
        for call in calls {
            let kernel = match *call {
                CallId::Baseline(index) => kernels
                    .baseline
                    .get(index)
                    .ok_or_else(|| anyhow!("projection baseline index is out of bounds"))?,
                CallId::Pair => &kernels.pair,
                CallId::FullTail => &kernels.full_tail,
            };
            result = result
                .checked_add(select(&kernel.plan))
                .ok_or_else(|| anyhow!("projection count overflows"))?;
        }
        Ok(result)
    }
    Ok(ProjectionEvidence {
        baseline_call_count: baseline_calls.len(),
        candidate_call_count: candidate_calls.len(),
        baseline_input_plane_exposures: sum(kernels, baseline_calls, |plan| {
            plan.input_planes.len()
        })?,
        candidate_input_plane_exposures: sum(kernels, candidate_calls, |plan| {
            plan.input_planes.len()
        })?,
        baseline_output_plane_stores: sum(kernels, baseline_calls, |plan| {
            plan.output_planes.len()
        })?,
        candidate_output_plane_stores: sum(kernels, candidate_calls, |plan| {
            plan.output_planes.len()
        })?,
        baseline_logical_input_exposures: sum(kernels, baseline_calls, |plan| {
            plan.logical_input_count
        })?,
        candidate_logical_input_exposures: sum(kernels, candidate_calls, |plan| {
            plan.logical_input_count
        })?,
    })
}

fn write_result(path: &Path, body: &ResultBody) -> Result<()> {
    ensure!(
        !path.exists(),
        "refusing to overwrite result {}",
        path.display()
    );
    let mut value = serde_json::to_value(body)?;
    let digest = sha256_hex(&python_canonical_json(&value)?);
    value
        .as_object_mut()
        .ok_or_else(|| anyhow!("serialized result is not an object"))?
        .insert("content_sha256".to_owned(), Value::String(digest));
    let mut raw = python_canonical_json(&value)?;
    raw.push(b'\n');
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("result path has no parent"))?;
    ensure!(
        parent.is_dir(),
        "result parent does not exist: {}",
        parent.display()
    );
    let temporary = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow!("result file name is not UTF-8"))?,
        std::process::id()
    ));
    ensure!(!temporary.exists(), "temporary result path already exists");
    fs::write(&temporary, raw)
        .with_context(|| format!("cannot write temporary result {}", temporary.display()))?;
    fs::rename(&temporary, path)
        .with_context(|| format!("cannot publish result {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_canonical_json_sorts_keys_and_formats_exponents() {
        let value = serde_json::json!({
            "z": 1.0e-7,
            "a": ["μ", 1.0, -0.0],
        });
        assert_eq!(
            String::from_utf8(python_canonical_json(&value).unwrap()).unwrap(),
            r#"{"a":["μ",1.0,-0.0],"z":1e-07}"#
        );
    }

    #[test]
    fn symjit_complex_source_binding_expansion_matches_runtime_adapter() {
        let spec = KernelSpec {
            label: "contract fixture".to_owned(),
            source: SourceRecord {
                path: PathBuf::from("/unused"),
                sha256: "0".repeat(64),
                size_bytes: 1,
                optimization_level: 3,
            },
            inputs: vec![
                LogicalInput {
                    kind: InputKind::Value,
                    source_id: 4,
                    global_component: 12,
                },
                LogicalInput {
                    kind: InputKind::Momentum,
                    source_id: 8,
                    global_component: 103,
                },
                LogicalInput {
                    kind: InputKind::ModelParameter,
                    source_id: 5,
                    global_component: 205,
                },
            ],
            outputs: vec![LogicalOutput {
                arena: OutputArena::Amplitude,
                component: 7,
            }],
        };
        let plan = binding_plan(&spec).unwrap();
        assert_eq!(
            &*plan.input_planes,
            &[
                PlaneBinding::CurrentReal(12),
                PlaneBinding::CurrentImaginary(12),
                PlaneBinding::Momentum(103),
                PlaneBinding::Zero,
            ]
        );
        assert_eq!(
            &*plan.input_scalars,
            &[
                ScalarBinding::ModelReal(5),
                ScalarBinding::ModelImaginary(5),
            ]
        );
        assert_eq!(
            &*plan.output_planes,
            &[
                PlaneBinding::AmplitudeReal(7),
                PlaneBinding::AmplitudeImaginary(7),
            ]
        );
    }

    #[test]
    fn explicit_schedule_contract_rejects_inferred_or_reordered_membership() {
        let canonical = (0..8)
            .map(|leaf_index| ScheduleRef::Baseline { leaf_index })
            .chain([
                ScheduleRef::Candidate {
                    kind: CandidateKind::Pair,
                },
                ScheduleRef::Baseline { leaf_index: 12 },
            ])
            .collect::<Vec<_>>();
        let ids = schedule_ids(&canonical).unwrap();
        assert_eq!(ids[8], CallId::Pair);
        assert_eq!(ids[9], CallId::Baseline(12));
        let mut reordered = ids.clone();
        reordered.swap(8, 9);
        assert_ne!(ids, reordered);
    }

    #[test]
    fn strict_structs_reject_duplicate_and_unknown_json_fields() {
        let duplicate = r#"{
            "value_component_count":2,
            "value_component_count":3,
            "current_component_count":3,
            "amplitude_component_count":1,
            "momentum_scalar_component_count":4,
            "momentum_form_count":1,
            "model_parameter_count":2
        }"#;
        assert!(serde_json::from_str::<ArenaShape>(duplicate).is_err());
        let unknown = r#"{
            "value_component_count":2,
            "current_component_count":3,
            "amplitude_component_count":1,
            "momentum_scalar_component_count":4,
            "momentum_form_count":1,
            "model_parameter_count":2,
            "extra":0
        }"#;
        assert!(serde_json::from_str::<ArenaShape>(unknown).is_err());
    }

    #[test]
    fn fixed_tile_planes_are_reproducible_and_zero_plane_stays_zero() {
        let shape = ArenaShape {
            value_component_count: 2,
            current_component_count: 3,
            amplitude_component_count: 1,
            momentum_scalar_component_count: 4,
            momentum_form_count: 1,
            model_parameter_count: 2,
        };
        let mut arena = Arena::new(128, 32, &shape).unwrap();
        assert_eq!(arena.invocation_shape(), (4, 0, 32));
        assert_eq!(arena.current_real.len(), 3 * 32);
        assert_eq!(arena.amplitude_real.len(), 32);
        let initial = arena.current_real.to_vec();
        arena.current_real.fill(f64::NAN);
        arena.zero.fill(1.0);
        arena.reset_deterministic();
        assert_eq!(&*arena.current_real, initial);
        assert!(arena.zero.iter().all(|value| *value == 0.0));
        assert!(arena.model_imaginary.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn numerical_tolerance_accepts_observed_o3_reassociation() {
        let actual = 9.939_574_824_707_947_f64;
        let reference = 9.939_574_824_707_945_f64;
        let absolute = (actual - reference).abs();
        let tolerance = numerical_tolerance(reference, 1.0e-12, 1.0e-15);

        assert!(absolute <= tolerance);
        assert!(absolute > 0.0);
        assert!((reference + tolerance * 2.0 - reference).abs() > tolerance);
    }
}
