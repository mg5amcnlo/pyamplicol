---
title: "Always-Summed Spinor and Bispinor DAG Study"
nav_order: 4
parent: "Profiling and Benchmarking"
---
<!-- SPDX-License-Identifier: 0BSD -->

# Always-summed spinor and bispinor DAG study

## Result

The experimental spinor lane is no longer restricted to all-gluon processes.
It now has two complementary construction paths:

- a model-driven recurrence lowerer which consumes the authenticated semantic
  recurrence and prepared-kernel catalogue. It currently publishes generic
  scalar-contact graphs, `u u~ > g g`, `u u~ > g g g`, and selected-flow
  `g g > g g` without a process-family tag or a PDG-specific lowering switch;
- compact specialized builders for four through six external gluons, one
  massless quark line plus two through four gluons, a massless quark line plus
  `Z + 0, 1, 2 gluons`, and `g g > t t~` in either LC ordering.

All of these evaluators expose one aggregate `h:sum` axis and run through the
normal pyAmpliCol `Runtime`. The graph-backed path is an artifact format and
runtime architecture, not merely a private numerical prototype.

The first model-driven mixed-QCD cell is faster than the corresponding
selected-flow component recurrence. The newly covered contact and five-point
cells are correct but not yet competitive with the component evaluator:

| Process and fixed LC flow | Spinor us/point (RSE) | Component us/point (RSE) | Paired speedup (RSE) |
| --- | ---: | ---: | ---: |
| `u u~ > g g`, `flow:2,3,4,1` | 0.4025 (0.93%) | 0.5291 (0.68%) | 1.315x (1.14%) |
| `g g > g g`, `flow:1,2,3,4` | 0.7542 (0.54%) | 0.6169 (1.50%) | 0.818x (1.54%) |
| `u u~ > g g g`, `flow:2,3,4,5,1` | 1.9359 (1.11%) | 1.2619 (0.67%) | 0.652x (1.30%) |

The earlier optimized all-gluon builders remain substantially faster:

| External gluons | Spinor us/point (RSE) | Component us/point (RSE) | Paired speedup (RSE) |
| ---: | ---: | ---: | ---: |
| 4 | 0.1120 (1.72%) | 0.6473 (0.38%) | 5.79x (1.59%) |
| 5 | 0.2038 (1.15%) | 2.8108 (0.95%) | 13.80x (0.51%) |
| 6 | 1.5107 (1.60%) | 6.9406 (0.49%) | 4.60x (1.36%) |

These are warmed native Runtime measurements. The timed spinor boundary
includes crossing, source permutation, spinor factorization, DAG evaluation,
the complete incoherent helicity sum, normalization, and one real output.
Artifact loading and workspace construction are outside both timers.

This remains an experimental, f64, leading-colour, fixed-flow evaluator. The
generic lowerer is deliberately fail-closed and does not yet cover every model
primitive; the exact supported boundary is recorded below.

## Model-driven lowering

The generic path starts from `AuthenticatedRecurrenceBuilderInput::build()`.
The resulting semantic recurrence retains source states, current momenta,
parent current IDs, transition and finalizer templates, exact factors,
closures, resolved helicities, and topology-replay transport. That is the
authoritative graph skeleton; particle names and process-pattern matching are
not used to infer its algebra.

Every executable operation must have an exact prepared-catalogue certificate.
The initial primitive set includes:

- scalar product/contact transitions and scalar closure;
- colour-ordered three-vector transitions;
- vector-wedge-vector and antisymmetric-tensor-vector transitions, represented
  as sparse sums of decomposable bivectors rather than six dense components;
- both massless Weyl-vector transitions;
- massless Weyl and Feynman-vector propagators;
- metric-vector and opposite-chirality Weyl closures, including
  identity-finalized terminal vectors;
- scalar, vector, and fermion source contracts with authenticated crossing.

The certificate retains the runtime primitive, exact contract digest, parent
permutation, and exact constant or prepared-parameter scale. An unknown state,
kernel, scale, source contract, finalizer, or closure rejects lowering instead
of silently falling back to a guessed spinor identity. Compiler-emitted
binary64 common scales are accepted only when their expanded canonical
expressions agree; a perturbed coefficient remains rejected.

The resulting `PACSPDG2` payload stores the simplified semantic DAG, explicit
source permutation/sign/kind rows, and dense DAG-to-prepared-parameter
bindings. It does not store workspace layout or a second identity. The normal
artifact payload checksum authenticates the bytes.

Three source representations are available:

- `NullSpinor` for a massless source;
- `MassiveSpinorPair` for a timelike source decomposed into two null pairs;
- `MomentumOnly` for a scalar source which requires no spinor factorization.

A parameter-free graph, such as the current mixed-QCD cell, needs no prepared
kernel pack. A graph that reads a prepared parameter must retain the existing
prepared model-parameter evaluator. The loader checks this once at the
authoritative payload boundary; the scalar regression verifies that changing
public `lam` refreshes the derived complex coupling used by the DAG.

## Spinor representation and sharing

Each massless momentum is factorized as

`p_(alpha,dotalpha) = lambda_alpha * lambdatilde_dotalpha`.

Massless vector polarizations are rank-one bispinors. After extracting the
repository's common `sqrt(2)` convention, the two helicities are represented
by `|q><i|/<qi>` and `|i><q|/[iq]`. Massless quark lines use linear Weyl
expressions. The massive-vector slice decomposes all three Z polarizations
into bispinors, while the top slice uses a full two-Weyl Dirac source and the
widthful propagator

`i (slash(P) + m) / (P^2 - m^2 + i m Gamma)`.

The graph builder applies bracket antisymmetry, self-zero rules, exact
coefficient combination, Schouten rewriting, common-factor extraction, and
dead-node elimination. The component/bispinor identity

`D(a,b) . D(c,d) = -1/2 <ac>[bd]`

is used for vector contractions. Real tree-level parity or authenticated
global-helicity-flip pairs share one root with multiplicity two where the
proof is available. The batch evaluator uses a node-major tiled workspace,
preallocates output and scratch storage, and performs no per-point graph
construction.

For the pure-gluon oracle, the normalized Berends--Giele recurrence is

```text
C(A,B;p,q) = (A.B)(p-q) + 2(A.q)B - 2(B.p)A
Q(A,B,C)   = 2B(A.C) - A(B.C) - C(A.B)
j_I        = (sum C + sum Q) / P_I^2
```

The optimized four- and five-point graphs use shared Parke--Taylor factors.
Six-point NMHV roots use one adjacent BCFW step. This reduced the six-gluon
execution graph from the 7,482-node raw BG oracle to 415 live semantic nodes.

## Correctness

The graph-backed QCD candidates and component recurrence use the same saved
seed-12345 momentum points and fixed flows:

| Process and fixed flow | Spinor DAG | Component recurrence | Relative difference |
| --- | ---: | ---: | ---: |
| `u u~ > g g`, `flow:2,3,4,1` | 4.459883460083079 | 4.459883460083116 | 8.36e-15 |
| `g g > g g`, `flow:1,2,3,4` | 24.344900673179623 | 24.344900673179900 | 1.14e-14 |
| `u u~ > g g g`, `flow:2,3,4,5,1` | 0.0035919538077271934 | 0.0035919538077272050 | 3.26e-15 |

A single integration test regenerates the prepared model and all three graph
artifacts, checks the retained component authorities (including the tracked
four-gluon LC fixture value `1.7527418719125202` at its generic-1 point), and
verifies that doubling `alpha_s` multiplies each result by `2**QCD_power`.

The generic scalar integration covers two independently normalized processes:

| Process | `lam = 1` | `lam = 3` |
| --- | ---: | ---: |
| `scalar_0 scalar_0 > scalar_0 scalar_0` | 0.5 | 4.5 |
| `scalar_0 scalar_0 > scalar_1 scalar_2` | 1.0 | 9.0 |

The broader specialized builders have independent component or analytic
checks as well:

- four-, five-, and six-point open massless quark lines agree with clean
  component artifacts; the five-point selected root has stripped norm
  `1e-6` and the summed value is `1.2e-5`;
- `d d~ > Z`, `Zg`, and both `Zgg` LC orderings retain all three massive-Z
  polarizations and agree with chiral-current or clean component oracles;
- both `g g > t t~` LC orderings agree at four saved points, including a
  near-threshold point, to `2e-12` relative tolerance. The graph binds both
  `m_t = 173` and `Gamma_t = 1.4915`; setting the width to zero is explicitly
  checked not to reproduce the default oracle;
- the pure-gluon comparison covers the artifact validation point and eight
  deterministic massless RAMBO points at each multiplicity, with maximum
  relative differences of `2.00e-14`, `1.65e-14`, and `1.09e-14` for four,
  five, and six external gluons.

Focused algebra tests additionally cover incoming crossing, momentum
factorization, massive decomposition, polarization transversality and
normalization, Schouten, Fierz, three- and four-vector vertices, chirality
zeros, parity multiplicities, scalar/batch parity, and liveness slot reuse.

## Structural size

The model-driven QCD payloads remain compact, although the unfused generic DAG
grows faster than the component recurrence:

| Process | Spinor payload bytes | Component payload bytes | DAG nodes | Recurrence currents | DAG operands / contribution rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `u u~ > g g` | 431,458 | 7,266,670 | 69 | 16 | 202 / 14 |
| `g g > g g` | 430,354 | 725,823 | 189 | 9 | 612 / 8 |
| `u u~ > g g g` | 442,084 | 975,928 | 531 | 13 | 1,963 / 14 |

The operation counts are structural proxies, not machine-instruction counts.
The two-gluon quark graph's greater scalar-node count still wins at runtime
because its nodes are simple shared bracket/scalar operations and avoid
repeated four-component current kernels. The current generic contact and
five-point graphs need further common-subexpression and liveness lowering;
their structural growth is consistent with the measured slowdown. Most
candidate bytes are ordinary compiled-model and physics metadata rather than
workspace layout.

For comparison, the specialized all-gluon graphs contain 13, 33, and 415
nodes for four, five, and six external gluons, with 8, 16, and 32 stored roots.

## Measurement setup

- Host: Intel Core i7-8700K, 6 cores / 12 threads, Linux x86_64,
  Python 3.12.3.
- Model: prepared built-in SM, leading colour, selected sector 0.
- Processes: `u u~ > g g`, `g g > g g`, and `u u~ > g g g`, each with one
  explicit fixed flow.
- Input: each artifact's deterministic seed-12345 validation point, cycled to
  batch 128.
- Sampling: two warmups and five paired/interleaved blocks per evaluator;
  each block was calibrated to approximately one second.
- Timer: native `Runtime._benchmark_f64_wall_time` for both implementations.
- Candidate RSE range: 0.54--1.11%; reference RSE range: 0.67--1.50%.

Memory was guarded throughout the current mixed/scalar checkpoint:

| Operation | Peak process-tree RSS |
| --- | ---: |
| Prepared built-in model | 0.189 GiB |
| Coordinated release restage plus focused Rust test | 3.389 GiB |
| Fresh three-process QCD integration test | 0.241 GiB |
| Four candidate/reference artifact generations | 0.210 GiB |
| Fresh parameterized-scalar integration test | 1.293 GiB |
| Paired mixed benchmark | 0.136 GiB |

The largest focused debug build observed during the generic-lowering work was
5.486 GiB. Every measured step remained below the 10 GiB design limit.

## Reproducing the mixed graph

The opt-in deliberately remains outside the public configuration surface:

```python
from pathlib import Path

from pyamplicol import ModelSource, ProcessRequest
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    GenerationConfig,
    ProcessConfig,
    RunConfig,
)
from tools.developer.generation_slice import GenerationSlice, generate_slice

config = RunConfig(
    action="generate",
    color=ColorConfig(accuracy="lc", lc_flow_layout="topology-replay"),
    process=ProcessConfig(
        coupling_order_policy="explicit",
        max_coupling_orders={"QCD": 2, "QED": 0},
    ),
    generation=GenerationConfig(workers=1, emit_api_bundle=False),
    evaluator=EvaluatorConfig(execution_mode="compiled"),
)
prepared = ModelSource.built_in_sm().compile(
    prepared_output=Path(".artifacts/built-in-sm.pyamplicol-model"),
    evaluator=config.evaluator,
)
generate_slice(
    ProcessRequest.parse("u u~ > g g", name="u_ubar_to_g_g"),
    Path(".artifacts/uugg-spinor"),
    selection=GenerationSlice(
        reference_color_order=(2, 3, 4, 1),
        selected_color_sector_ids=(0,),
        experimental_spinor_dag=True,
    ),
    config=config,
    model=prepared,
)
```

Loading and evaluating uses the public Runtime:

```python
from pyamplicol import Runtime

runtime = Runtime.load(".artifacts/uugg-spinor", process="u_ubar_to_g_g")
assert runtime.execution_mode == "spinor"
values = runtime.evaluate(points, color_flows=("flow:2,3,4,1",))
```

The paired benchmark used for this report is reproducible with:

```console
python tools/developer/spinor_dag_mixed_benchmark.py \
  --candidate .artifacts/uugg-spinor \
  --reference .artifacts/uugg-component \
  --candidate-process u_ubar_to_g_g \
  --reference-process u_ubar_to_g_g \
  --flow flow:2,3,4,1 \
  --blocks 5 --target-seconds 1 \
  --output .artifacts/uugg-spinor-benchmark.json
```

## Remaining boundary

The result establishes a model-driven mixed-process execution path, but not a
universal replacement for the component recurrence. The next generic
boundaries are:

1. scalar--Dirac/Yukawa and Dirac-vector transitions plus general
   massive-Dirac propagation;
2. arbitrary massive-vector insertions rather than the current certified Z
   slice;
3. fermion-pair-to-vector transitions and more than one open quark line;
4. more than one retained colour flow in one graph payload;
5. common-subexpression, fusion, and liveness lowering for generic contact and
   higher-multiplicity graphs, which are currently slower than the component
   evaluator;
6. a numerically safe dynamic BCFW shift or fallback for the specialized
   pure-gluon graphs.

Until those are implemented and measured, unsupported recurrences continue to
fail closed or use the existing component evaluator. No public configuration
surface promises broader spinor coverage.
