# Compiled-DAG O3 non-union DirectTable microkernels

## Objective and immutable inputs

This feature targets the compiled JIT O3 non-union execution lane. It must
improve the complete-artifact, runtime-selected `u u~ > Z+6g` workload by at
least 10% at both batch 128 and batch 1024 without turning the compiled DAG
into recurrence execution.

The executable baseline starts from pyAmpliCol
`f4606fa9be52355b4a66efcfa2b7072d489205eb`; implementation is rebased onto
the corrected-main descendant
`2b359bc0f50f724ec45f5ca4e71c458b3ce4f03e`. Its dependency contract is:

- SymJIT repository:
  `https://github.com/ValentinHirschi/symjit_changes_for_pyamplicol.git`;
- branch: `pyamplicol-generic-direct-apis`;
- revision: `89efdb806e7fcd9ac68a9d38f3f2880adf1987d2`;
- archive SHA-256:
  `070ff7fc04d5cdc5ab769d7a47b3da04cbc2b97d87136d303180c95b9eb380cd`;
- source-tree SHA-256:
  `e42d648d995c61881e560aefc50f80a995e86fb24a67ed9b0f0b5a80d6773fcf`;
- configured candidate-tree SHA-256:
  `fdf06a56cffe301df93b7e08a85f6d5cf956842959fc9a5a95fa9bc61c43246d`;
- dependency candidate fingerprint: `c2b7cc28699b`;
- DirectApplication storage ABI: `symjit-direct-application-storage-v1`;
- DirectTable binding ABI: `symjit-direct-table-binding-v1`;
- DirectTable descriptor ABI: `symjit-direct-table-descriptor-v1`.

This optimization adds no local SymJIT patch. Later `main` revisions contain
small recurrence-support patches owned by the report-support lane; they are
baseline inputs, not part of this optimization. This feature may use the
pinned DirectApplication and DirectTable APIs but must not expand that
dependency delta. If those APIs cannot represent the required kernel, the
feature stops.

The historical same-host complete-artifact selected-flow reference is
`54.024 +/- 0.226 us/point` at batch 128 and
`56.880 +/- 0.361 us/point` at batch 1024. Its result SHA-256 is
`6ca4cfa0038d7e2d8dab6b21e0106fadd2572234fef1fbd760eb2476fad242ab`.
These values predate the exact fork-pinned source revision and are diagnostic
only. Before candidate performance is judged, a fresh immutable baseline
artifact and runtime identity must be generated from the source revision
above and recorded in this document.

### Fresh `f4606fa` baseline

The first exact-source baseline capture was generated on 2026-07-26 from the
clean source revision above. Its authenticated identities are:

- candidate native build-input SHA-256:
  `e84ab5ea52d1f523c5338e0adb7178fdb4a8cb8dbd41343fe701809ea2a84873`;
- native extension SHA-256:
  `f71d40debcb612f34da2e038a4b251a66ef010a27026f90946dee59f37d2ae7b`;
- artifact ID:
  `bce2e798113ebe0ff2677c96b0abcfffc65bd58bec065d135747cbc7862a1d8b`;
- artifact semantic-identity SHA-256:
  `403cec4e225634be1660d61a821d4ca0052181ae193ad2841b6ae37ffb836caf`;
- artifact tree SHA-256:
  `38ab1a323e22393e263b5ce01937d3e0614e8e0021fbc2931445f977dc503f0a`.

Generation took `10.979621 s`, reported a `399,654,912` byte process high-water
mark, and produced a `23,608,791` byte artifact, including a `7,801,720` byte
evaluator-state payload. The executable materialized schedule contains seven
current stages, 55 non-source current destinations, and 203 active
interactions. Its destination dimensions are 26 two-component, 15
four-component, and 14 six-component currents. The retained
1,425-current/8,338-interaction helicity recurrence structure is proof-only
metadata and is forbidden as the profitability denominator.

The first seven-sample five-second timing pass measured:

| Batch | Median us/point | MAD us/point |
|---:|---:|---:|
| 1 | 96.121 | 0.874 |
| 127 | 42.774 | 0.112 |
| 128 | 42.993 | 1.245 |
| 129 | 44.719 | 1.723 |
| 1023 | 42.575 | 0.811 |
| 1024 | 42.376 | 0.911 |
| 1025 | 42.809 | 0.309 |

An unrelated native link overlapped early samples in this pass. The identities,
artifact metrics, numerical validation, selector contract, and active-schedule
census are authoritative; these timing values are provisional diagnostic
anchors. Landing acceptance uses a new quiet, alternating baseline/candidate
capture and does not reuse these medians.

### Mandatory pre-code census outcome

This subsection records the census used by the first, contribution-level
candidate. That formulation was measured and rejected below. Its
contribution/finalizer counts and eight-identity/16-input bounds are
historical evidence, not the contract of the later complete-current
formulation.

The exact materialized `f4606fa` `u u~ > Z+6g` schedule was censused on
2026-07-26 before integrating the candidate. An initial audit incorrectly
treated one fully fused current as the source-motif unit. A second independent
audit corrected that interpretation: the approved design tableizes canonical
interaction contributions while retaining all-or-nothing ownership of each
selected current. The first contribution overwrites, later contributions
accumulate in original order, and the ordinary propagator/finalizer runs last.
No selected destination is split between table and residual execution.

The user subsequently authorized the profitability denominator to follow the
actual repeated source unit: each active occurrence of a prepared-kernel
identity that repeats in the 203-interaction materialized schedule. This
replaces the empty `evaluation_group_id` identity denominator. Those IDs
identify concrete execution rows and all have fanout one; they do not identify
shared generated source.

The frozen global prepared-kernel occurrence distribution is:

| Prepared kernel | Active occurrences |
|---:|---:|
| 4 | 35 |
| 7 | 49 |
| 24 | 49 |
| 36 | 40 |
| 41 | 30 |
| **Total** | **203** |

All 26 two-component destinations are wholly eligible. Their 98 contributions
collapse to two canonical vector--Weyl vertex kernels, used 49 times per
chirality:

| Prepared kernel | Chirality | Calls | Complex inputs | Complex outputs | Source bytes |
|---:|---:|---:|---:|---:|---:|
| 7 | +1 | 49 | 6 | 2 | 1,634 |
| 24 | -1 | 49 | 6 | 2 | 1,632 |

The contribution-count distribution is four destinations at each width one
through six and two destinations at width seven. It yields 26 overwrite and
72 ordered-accumulate attachments, followed by four propagator/identity
finalizer classes. The corrected slice is therefore structurally compact and
fits every hard arity, output, kernel-count, and source-size bound.

The initial implementation also includes the smallest structural extension:
five singleton four-component gluon currents, current IDs
`15,17,19,21,23`. They all use prepared three-vector kernel 4 and vector
propagator 37. Eligibility is structural: each current is homogeneous, owns
exactly one contribution, and has no residual contribution. The prepared
vertex has 16 complex inputs; embedding its propagator adds four momentum
inputs, so the complete composite has 20 complex inputs and four complete
complex outputs. Kernel 4 has canonical signature
`120acbff47e08ce698a108c3c8b0758555d5292f24c6b05c4504a977714ebd8c`
and 20,317 projected expression bytes in the current pinned catalog.
Propagator 37 has canonical signature
`b04f53f0c0046718e3090d041bea2f7d31dd363f02d1f7fc614ce620a7e7cacd`.

This authorized slice covers `103/203 = 50.739%` of the global
prepared-kernel occurrences (`5073` basis points under the runtime's
deliberate floor) and therefore passes the 50% gate. It uses three vertex
identities and six known vertex/finalizer identities. The initial prepared
vertex inputs remain capped at 16; the complete-current extension is capped
at 64 and observes 43 overall. Outputs remain two for vector--Weyl kernels or
four for the singleton three-vector family. Its current prepared-source proxy
is `26,233/391,469 = 6.70%`, and the five-row complete-current table adds
exactly 1,040 semantic bytes. Four-output kernels outside this exact
structural family remain ineligible.

The captured census JSON is
`COMPILED_DAG_O3_NON_UNION_MICROKERNEL_CENSUS.json`, with SHA-256
`1afeebdd661064bd13aedb249948b53042d6a9f37e10f47814bcce236c6708ab`.
It was produced from the clean generation worktree by constructing the
existing unit-test `u u~ > Z g g g g g g` evaluator process and calling
the materialized-stage and prepared-kernel catalog builders directly; no
pyAmpliCol native rebuild participated in the structural census.

| Census quantity | Required | Observed |
|---|---:|---:|
| Global repeated prepared-kernel occurrences | denominator | 203 |
| Wholly owned vector--Weyl destinations | exactly 26 | 26 |
| Structurally homogeneous singleton three-vector destinations | exactly 5 | 5 |
| Eligible occurrences | at least 50% | 103 / 203 (50.739%) |
| Canonical vertex kernel identities | at most 8 | 3 |
| Initial prepared-kernel complex inputs | at most 16 | 16 |
| Complete-current complex inputs | at most 64 | 43 |
| Vector--Weyl outputs | exactly 2 | 2 |
| Singleton three-vector outputs | exactly 4 | 4 |
| Prepared vertex source ratio | at most 25% | 26,233 / 391,469 (6.70%) |
| Projected semantic rows | at most 4 MiB | 1,040 bytes for the extension |

A standalone Rust probe compiled four patchless SymJIT O3 source classes
(propagated/unpropagated crossed with chirality) using only DirectTable
binding-v1 and descriptor-v1. Their scalar plus SIMD DirectTable code was
11,584 bytes, versus 92,536 bytes for 26 isolated complete-current O3
applications, a projected 87.482% reduction. Descriptor payload was 1,104
bytes and invocation plus attachment rows were estimated at 10,528 bytes.
The emitted table kernels contained no branch-and-link calls, gathers, or
scatters. For this upper-bound probe the linear massless finalizer was
distributed into each contribution source; that is code-size evidence only,
not authorization to change the approved post-accumulation finalizer order.

The probe covers the dominant 98-call vector--Weyl core. No exact
three-vector machine-code number is inferred from source or descriptor bytes.
The complete candidate records exact selected loaded machine-code evidence
when the pinned API exposes it. This evidence is diagnostic rather than a
landing prerequisite.

The pinned SymJIT API creates an observability constraint.
`DirectTableCallable` exposes exact scalar and SIMD code shapes, but
`DirectApplication` privately lowers its source to a distinct `Application`
and neither it nor the resulting `DirectApplet` exposes the lowered
`MachineCode::size`. The public source-application machine-code size is not
the executed lowered DirectApplication body. The independent auditor therefore
labels only an exact
`executed-selected-machine-code-scalar-plus-simd-v1` value as machine-code
evidence. An unavailable exact metric is reported as unavailable and does not
fail an otherwise correct and faster candidate. DirectTable-only bytes and
portable source-Application bytes remain diagnostic and are never treated as
machine code.

Implementation proceeds with the authorized 103-occurrence slice. No further
authorization pause is required while a suitable in-scope improvement remains.
This does not relax landing: the candidate is deleted or withheld unless it
passes every numerical, performance, resource, cutover, and regression gate
below.

## Design

The compiled schedule remains the owner of topology, helicity, selector,
dependency, and arena policy. After the contribution-level implementation was
rejected, the implementation was cut over to complete-current islands:

1. Select a two-component current, or one of the structurally proven
   singleton three-vector four-component currents, only when every
   interaction contributing to it, its selector partition, its finalizer,
   and all destination slots are structurally proven.
2. Build one O3 source containing the current's complete ordered contribution
   sum and its finalizer substitution.
3. Bind current, momentum, mutable-model-parameter, and coupling-component
   inputs through one DirectTable invocation row.
4. Write every canonical output of that current once through one identity,
   overwrite attachment. There is no contribution accumulation table, scratch
   current, or separate finalizer call.
5. Remove the current's whole output from the residual stage. Four-component
   and otherwise ineligible currents remain ordinary compressed O3
   DirectApplication residual leaves.

One current may not be split between table and residual execution. The
generator constructs the symbolic sum in original interaction order and
records those interaction IDs in the plan. Symbolica canonicalizes algebraic
addition and O3 may reassociate floating-point operations, so bitwise
contribution order is not claimed; acceptance is the explicit
`rtol=1e-12`, `atol=1e-15` numerical contract.

The exact `qq_Z6g` selected schedule admits 26 two-component currents plus
five singleton four-component currents. It emits 27 complete-current kernel
identities and calls: 26 one-row vector--Weyl kernels plus one shared
five-row three-vector kernel. There are 31 invocation rows, 31 overwrite
attachments, zero scratch components, and zero finalizer calls. Maximum
complex input arity is 43; vector--Weyl kernels have two complex outputs and
the one shared three-vector kernel has four. All other four-component
currents remain residual.

The complete-current extension is bounded to 64 kernel identities, 64 complex
inputs, four complex outputs, 64 KiB aggregate source-application payload,
and 4 MiB semantic row data per artifact. These bounds stay within the APIs
already present in the pinned SymJIT fork and add no dependency patch.

The outer compiled stage is still generated exactly once. A residual leaf
reuses an already compiled outer chunk only when the chunk is retained in
full, its DirectApplication binding is byte-for-byte consistent with the
outer evaluator, and every residual input has an exact semantic and canonical
symbol projection. Partial or unproven chunks fall back to ordinary residual
compilation. This removes the duplicate residual compile responsible for the
first candidate's 110-second generation time without introducing a second
execution route.

Landing still requires the complete-current candidate to improve both primary
batches by at least 10%. No whole-schedule superkernel, generic MIR outlining,
additional stage fusion, or recurrence-style execution is in scope.

## Artifact cutover and diagnostics

Every newly generated compiled artifact uses compiled-stage-plan v2. It
contains residual DirectApplication leaves and zero or more DirectTable
islands, with the existing payload locators and ABIs, canonical motif
identities, selector partitions, invocation and attachment rows, factor
catalogs, plane bindings, and explicit dependency/order data. Existing generic
artifact payload hashes remain intact, but this slice adds no second
authentication protocol, execution certificate, or binary-code attestation.

The loader rejects compiled-stage-plan v1 and incompatible direct ABIs with an
actionable regenerate-artifact error. There is no v1 reader, converter,
dual-run production path, hidden fallback, or backward-compatibility toggle.
A v2 stage with no profitable islands is valid and executes its residual
leaves; that is the ordinary current representation, not legacy loading.
Malformed declared islands fail closed instead of silently becoming
residuals.

Inspection records expose island, kernel, invocation, and attachment counts;
semantic-row bytes; arena bytes; warmed allocation counts; and table/residual
code sizes when the underlying API exposes them exactly. Public Python APIs,
CLI modes, selectors, and configuration remain unchanged. Eager retains its
existing O2 DirectTable plan and is a non-regression lane, not a source of the
claimed gain.

## Acceptance contract

The primary artifact is complete topology-replay `u u~ > Z+6g`, with physical
flow `flow:2,4,5,6,7,8,9,1` selected at runtime and a helicity sum. Baseline
and candidate use the same source, model, points, selector, runtime, and host.
Measurements cover batches 1, 128, and 1024 plus odd tails 127/129 and
1023/1025. At least seven alternating subprocess samples run for five seconds
each.

Landing requires:

- at least 10% lower median wall time at both batch 128 and batch 1024, with
  the paired change larger than three MAD;
- batch-1 regression no greater than 5%;
- exact executed selected scalar-plus-SIMD loaded machine-code change is
  reported when available; unavailable evidence is explicitly marked
  unavailable, and neither source bytes nor DirectTable-only bytes substitute
  for it;
- no warmed execution allocations;
- generation time, artifact bytes, load time, and peak RSS no more than 10%
  worse;
- pointwise totals and resolved contributions within `rtol=1e-12` and
  `atol=1e-15`, with `evaluate()` agreeing with
  `evaluate_resolved().total()`.

Regression gates cover the `Z+6g` LC union workload; NLC and full-colour
`g g > t t~ +3g`; residual-only `d d~ > Z+3g`; built-in and UFO-SM models;
one existing fixture with four independent quark lines in compiled LC, NLC,
and full colour; mutable parameters; structural zeros; global and per-point
selectors; and malformed plans. Non-target execution and eager may regress by
no more than 2% or three MAD.

Native AArch64 validation runs under the 30 GiB watchdog. x86-64 compile and
test evidence comes from the existing CI path. No command requests elevated
permissions: blocked operations are replaced with workspace-local/offline
routes or reported as external blockers.

## Work ownership

The parent lane owns exact-source synchronization, integration, immutable
baseline and candidate builds, benchmarks, and the final main-token handoff.
Independent subagents own:

- generation, motif classification, and census;
- Rust DirectTable loading, validation, and stage-plan v2;
- adversarial correctness, cross-platform review, and performance audit.

Each lane uses isolated ownership and sends commits to the parent for review.
Only fully gated milestones may be merged. The implementation goal is complete
only after the accepted candidate is pushed to `main` and its landing SHA,
build/runtime identity, numerical evidence, and performance record are handed
off.

## First formulation: measured outcome and rejection

The complete candidate was rejected on 2026-07-27 and must not be merged into
`main`. The final measured source was
`cf68e57c1fbc7dff640b563d3e540bfd0db57429`, built as candidate wheel
SHA-256
`91ee1b682a0dd2b147c2ccb06cf4a37cba62476092d5fa8f6e9f1b87dec3796c`.
Its installed runtime reported native build-input SHA-256
`5c90765bce354b10cd3fcb4ad68d36f896e50257efd8f60f10e2f0d6389c40ce`
and native-module SHA-256
`bc7b67df0c85043b37c22310017850f31664ba058465c711d692eb45f4750016`.
The comparator remained exact `f4606fa9be52355b4a66efcfa2b7072d489205eb`
with native-module SHA-256
`f71d40debcb612f34da2e038a4b251a66ef010a27026f90946dee59f37d2ae7b`.

Fresh complete artifacts were generated for the required built-in-SM process,
topology-replay layout, and selected physical flow. Artifact generation
completed successfully after fixing a stale serializer assertion that had
compared projected residual bindings to the original full stage width.
The resulting artifact evidence was:

| Quantity | `f4606fa` baseline | `cf68e57` candidate | Candidate / baseline |
|---|---:|---:|---:|
| Artifact ID | `a234484fbaabfd58a1555ad34c8003587bd1458bca40f208b6e7970373833583` | `ef86100a62d2099e0b58aceece571f9ff22510ae7f93f27326f6b19a40075ea5` | — |
| Core generation | 5.273607 s | 110.595564 s | 20.971 |
| Generation command wall time | 6.845435 s | 112.642975 s | 16.455 |
| Artifact tree | 23,606,943 bytes | 31,965,120 bytes | 1.354 |
| Material payload | 23,595,658 bytes | 31,868,225 bytes | 1.351 |
| Generation peak RSS | 0.300 GiB | 0.327 GiB | 1.090 |

The first full seven-pair run was stopped before accepting any measurement
because its precision-32 diagnostic spent more than five minutes in the
process-level Decimal oracle before producing a warmed sample. The decisive
go/no-go runtime evidence therefore used short, independent five-block Arena
samples from the already authenticated read-only artifacts. These samples are
diagnostic rejection evidence, not a substitute for the seven-pair acceptance
campaign:

| Batch | Baseline us/point | Candidate us/point | Candidate / baseline | Claimed gain |
|---:|---:|---:|---:|---:|
| 128 | 42.0984 | 56.0708 | 1.33190 | -33.19% |
| 1024 | 41.7545 | 55.4654 | 1.32837 | -32.84% |

At batch 1024 the eight sampled totals had maximum elementwise relative
difference `1.98345e-14` and maximum absolute difference `6.04742e-32`.
This is useful diagnostic correctness evidence, but the full numerical
acceptance gate was deliberately not completed after both runtime targets had
already failed decisively.

The slowdown is structural. The selected-flow baseline executes 13 fused O3
DirectApplication leaves. The candidate executes 42 ordered operations:
17 contribution DirectTable calls, 16 finalizer DirectTable calls, eight
residual leaves, and one amplitude leaf. Those tables expand to 5,918 unique
contribution rows plus 1,034 finalizer rows. All contribution input rows are
unique, so DirectTable fan-out cannot remove this work. The candidate also
materializes 2,108 complex scratch components: contributions write
overwrite/accumulate planes, then finalizers reread scratch and write the
canonical currents. The baseline fused programs instead retain sums,
propagation, common subexpressions, and register locality inside one compiled
stage.

Removing a few table-call boundaries, scalar parameter fills, or identity
finalizers cannot recover the roughly 48% candidate throughput improvement
needed to move from 56 us/point to the required 10% gain over a 42 us/point
baseline. A credible repair would have to compile complete current or stage
regions together through schedule superkernels, MIR outlining/fusion, or a
recurrence-like executor. Those approaches are explicitly outside this
plan. Adding more four-component islands would amplify the same row dispatch
and scratch-materialization costs.

The contribution-level implementation therefore followed the mandatory
failure branch of the acceptance contract and was not landed.

## Complete-current formulation: measured outcome and rejection

Continued optimization work reopened the feature with the complete-current
design above. This is a replacement, not a compatibility path: the generator
no longer emits contribution tables, scratch-current bindings, identity
finalizer kernels, accumulate attachments, or separate finalizer calls.
The runtime rejects artifacts retaining those shapes and requires regenerated
compiled-stage-plan v2 artifacts.

The source-level `qq_Z6g` census currently reports:

| Quantity | Complete-current candidate |
|---|---:|
| Eligible two-component currents | 26 |
| Eligible singleton four-component currents | 5 |
| Table kernel identities / calls | 27 |
| Invocation rows | 31 |
| Overwrite attachment rows | 31 |
| Scratch current components | 0 |
| Separate finalizer calls | 0 |
| Maximum complex inputs | 43 |
| Complex outputs per kernel | 2 or 4 |

An exact pinned-fork probe compiled representative composite sources in
milliseconds and produced roughly 1.3--1.7 KiB source applications, indicating
that the first candidate's 110-second generation was duplicate residual
compilation rather than composite-source cost. Exact artifact generation,
aggregate source bytes, numerical comparison, and alternating runtime
measurements were therefore required before any landing.

The completed formulation was rejected on 2026-07-27 and must not be merged
into `main`. The exact candidate source was
`2fd961286cbb7854a93e4b59f6bdec237a98ff6f`, built as candidate wheel
SHA-256
`a03d8a415517cbe715b88c2a3d6e15326801f255d9db229d7d1f3004c406c7c0`.
Its installed runtime reported native build-input SHA-256
`61f9029a0c2789ab88c6d6c346a4bbe2a4006feb2aabfff63832aea4c6e8a76b`
and native-module SHA-256
`8c9dbbf04104d64e4368bc4a3c0ae153a11e160cfaaf9c4f853eca6ec6a4bfaf`.
The matched comparator was
`2b359bc0f50f724ec45f5ca4e71c458b3ce4f03e`, with native build-input
SHA-256
`c60c078033c7f66eebf21c6bc0f8215cefb3448575ee3f2cb5e4c722d301ad54`
and native-module SHA-256
`bf2da2b8f6383eefa2268a9c946a11b0ad357d979f3a22119ae2a7855c559279`.

Fresh complete artifacts used identical built-in-SM process, O3 JIT, LC
topology-replay, worker, validation, and selected-flow inputs:

| Quantity | `2b359bc` baseline | `2fd9612` candidate | Candidate / baseline |
|---|---:|---:|---:|
| Generation command wall time | 5.9197 s | 13.5269 s | 2.285 |
| JIT generation time | 3.8309 s | 10.4950 s | 2.739 |
| Artifact tree | 23,611,060 bytes | 30,745,408 bytes | 1.302 |
| Generation peak RSS | 0.303 GiB | 0.342 GiB | 1.129 |

The exact cross-runtime eight-point selected-flow comparison passed with
maximum absolute difference `6.201494420926899e-32`, maximum relative
difference `6.2014944209269e-17`, and
`evaluate() == evaluate_resolved().total()` under `rtol=1e-12` and
`atol=1e-15`.

The decisive go/no-go used five alternating outer subprocess samples per lane
and batch. Each subprocess collected five native Arena headline blocks from
the same deterministic timing batch. The raw JSON evidence aggregate has
SHA-256
`bb9f486ded98f98deb9481853f24475dd32d6ce600d4da359bf193e15da9fc45`.

| Batch | Baseline us/point (median +/- MAD) | Candidate us/point (median +/- MAD) | Headline gain | Paired gain (median +/- MAD) |
|---:|---:|---:|---:|---:|
| 128 | 42.7410 +/- 1.0198 | 53.0954 +/- 0.7583 | -24.23% | -22.15% +/- 5.56% |
| 1024 | 42.6835 +/- 0.7590 | 52.8326 +/- 0.3117 | -23.78% | -25.24% +/- 0.74% |

The regression is structural rather than a cold-load or allocation effect.
For the selected-flow helicity-sum path, the baseline executes 13 fused O3
DirectApplication leaves per helicity pass. The candidate executes 28
DirectTable groups plus nine residual leaves. Four helicity passes therefore
produce exactly the observed increase from 52 to 148 Arena calls per
top-level evaluation. The 28 table groups contain 1,034 invocation rows, and
their 2,108 tableized complex outputs reload their inputs and traverse generic
factor, operation, attachment, and destination handling. Logical source
inputs rise from 9,305 in the fused baseline to 43,392 in the candidate, a
4.66x increase.

The generation and size failures have the same fragmentation cause. The
baseline PACBIN contains 63 source and 63 state members. The candidate retains
all of them, then adds 81 DirectTable sources and 17 residual source/state
compiles. Cross-lane source memoization could remove at most 54 table
compilations and roughly 3.67 seconds, leaving generation about 66% slower.
Raw content deduplication could save at most about 0.4 MiB, far short of the
4.77 MiB needed to satisfy the artifact limit. The v2 plan's repeated plane
catalogs and bindings account for most of the JSON growth.

Reaching a 10% gain from the measured candidate would require a further
roughly 27.5% runtime reduction, in addition to independent generation and
schema redesigns. A specialized identity-overwrite DirectTable ABI plus
coarse current fusion might remove some envelope work, but it requires a new
SymJIT API and still loses the fused baseline's shared-input CSE and register
locality. It is not a credible bounded repair under this plan.

The complete-current implementation therefore followed the mandatory failure
branch of the acceptance contract. The DirectTable microkernel family is
abandoned for compiled non-union-flow execution; future work should optimize
the already-fused DirectApplication schedule rather than tableizing its
currents.
