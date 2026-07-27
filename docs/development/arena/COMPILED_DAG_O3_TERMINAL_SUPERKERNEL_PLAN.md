# Compiled-DAG O3 terminal DirectApplication superkernel

## Objective and pinned inputs

This slice replaces the rejected DirectTable microkernel experiment with a
coarse, ordinary DirectApplication fusion of the final two current stages,
optionally incorporating amplitude when the disposable probe selects it. It
targets non-union-flow compiled JIT O3 execution and must improve
`u u~ > Z + 6g`, selected flow
`flow:2,4,5,6,7,8,9,1`, by at least 10% at both batch 128 and batch 1024.

Implementation started from exact executable pyAmpliCol source
`a4082a6c93c30ee34019b8fa0f29acfe2fd2fc4f` and is synchronized with latest
`main` report-only descendant
`fb233054502aaee5e39b1c9287efddd9c56a3913` on branch
`codex/dag_hit_compiled_terminal_superkernel`. The dependency closure remains
candidate fingerprint `c2b7cc28699b`, pinned SymJIT fork revision `89efdb8`,
and configured SymJIT tree
`fdf06a56cffe301df93b7e08a85f6d5cf956842959fc9a5a95fa9bc61c43246d`.
No dependency patch or new SymJIT API is permitted.

The provisional executable comparator is the byte-equivalent compiled runtime
at `2b359bc0f50f724ec45f5ca4e71c458b3ce4f03e`; `a4082a6` changes only
benchmark sampling and its focused test. Final acceptance will rebuild both
baseline and candidate from the final synchronized source.

## Evidence for proceeding

The exact selected-flow baseline has 13 O3 DirectApplication leaves. The last
two current stages and amplitude account for five of them:

- stage 6 has 640 complex outputs, all consumed only by stage 7;
- stage 7 has 768 complex outputs, all consumed only by amplitude;
- amplitude has 384 complex outputs.

The current tail exposes 9,460 split-plane inputs plus five scalar inputs and
writes 3,584 split-plane outputs. A composed tail has 1,498 unique logical
inputs, corresponding to 2,888 split planes plus two scalar inputs, and writes
only the 384 complex amplitude outputs. Projected whole-schedule geometry is:

| Quantity | Current | Tail fused | Change |
|---|---:|---:|---:|
| DirectApplication calls | 13 | 9 | -30.8% |
| Input-plane exposures | 19,652 | 13,080 | -33.4% |
| Output-plane stores | 8,616 | 5,800 | -32.7% |
| Logical input exposures | 10,077 | 6,834 | -32.2% |

The expected runtime gain is uncertain but still capable of clearing 10%.
Independent reviews place stage-6-to-stage-7 fusion at 5--10% centrally and
the full tail at roughly 7--18%, depending on how well O3 recovers sharing.
The hypothesis is not merely fewer calls: fusion removes one or two complete
intermediate materializations and lets O3 recover common subexpressions and
register locality across the exclusive tail.

A configuration-only 512-to-1024 output-chunk experiment already ruled out
simple leaf coalescing. It reduced the selected schedule from 13 calls to
eight, but regressed batch 128 by 1.9% and batch 1024 by 3.5%. The fused tail
must therefore demonstrate a materialization/CSE gain; call-count reduction
alone is not acceptance evidence.

Retained evaluator IR also exposes the main risk:

| Evaluator | Optimized IR instructions | SymJIT source |
|---|---:|---:|
| Stage 6 | 14,018 | 185,204 bytes |
| Stage 7 | 15,559 | 221,101 bytes |
| Amplitude | 1,152 | 12,044 bytes |
| Full tail | 30,729 | 418,349 bytes |

The exact retained baseline applications replaced by pair fusion total
405,792 bytes; including the amplitude leaf gives 418,040 bytes. The
disposable decision rejects a candidate whose stored source exceeds its
corresponding replaced payload, independently of the broader 768 KiB cap.

Every stage-6 complex output is referenced four or six times by stage 7,
giving 3,072 references to 640 values. Fusion therefore depends on CSE
recovering that sharing rather than expanding it. In contrast, each stage-7
component is consumed exactly once by amplitude, so adding amplitude to a
successful stage-6-to-stage-7 composition is structurally cheap. A disposable
composition/direct-execution probe must choose pair-only, full-tail, or
neither before production integration.

## Eligibility and compilation

Fusion is fail-closed and applies only when all of the following are proven:

- execution mode is compiled, backend is JIT, and optimization level is O3;
- the candidate tail consists of exactly two adjacent current stages followed
  by amplitude;
- every output of the first tail stage is consumed only by the second, and
  every output of the second is consumed only by amplitude;
- selector memberships and structural-zero domains are complete and do not
  require a boundary inside the fused tail;
- there are no external or residual consumers of either elided stage;
- parameter mutation, real/complex parameter typing, and output contribution
  order can be represented exactly;
- the amplitude output count is at most the existing 512-output chunk;
- projected and actual source/resource caps are satisfied.

Ineligible schedules retain the ordinary current compiled-DAG path. This is
not a compatibility reader: every artifact is regenerated under the new
runtime schema, and the runtime rejects the old compiled contract.

The generator will:

1. Build canonical semantic input identities for the two current stages and
   amplitude.
2. Build both a stage-6-to-stage-7 candidate and a full-tail candidate by
   substituting outputs with simultaneous `Expression.replace_multiple`
   operations.
3. Preserve the exact amplitude output and contribution order.
4. Deduplicate only semantically identical surviving inputs.
5. Compile the composed expressions as one ordinary O3 SymJIT application.
6. Lower each probe candidate through the existing identity-overwrite
   DirectApplication path.
7. Emit an explicit certificate naming the elided stage indices, source and
   destination bindings, dependency closure, expression/source digests, and
   contribution-order digest.

The runtime will validate the certificate, omit the two materialized current
stages, and execute the fused DirectApplication as the amplitude evaluator.
It will reject bad stage indices, non-exclusive consumers, selector-domain
mismatches, input/output aliases, parameter kinds, digests, or contribution
order. There is no v1 reader, converter, dual execution, hidden toggle, or
fallback after an artifact claims eligibility.

Eager remains architecturally unchanged. Shared helpers may be reused only
when behavior-neutral.

## Development sequence

1. Add a pure composition builder with exact unit tests on small symbolic
   tails.
2. In a disposable feature-only probe, compile both pair-only and full-tail
   applications and run them through the existing persistent arena schedule.
   No compatibility or production schema work is allowed before this probe
   passes.
3. Reject a probe on source expansion, DirectApplication lowering failure,
   stack use above 1 MiB, payload above 768 KiB, compile/RSS explosion, or
   numerical mismatch.
4. Prefer full-tail only when its payload is at most 1.25x pair-only, compile
   time is at most 1.5x pair-only, and it improves the full schedule by at
   least three percentage points more. The disposable composition process
   uses a process-lifetime RSS counter, which cannot honestly attribute a
   separate peak to the second candidate; it therefore enforces the absolute
   30 GiB watchdog but defers the relative RSS gate to isolated production
   artifact generation. Require the selected candidate to project at least
   12% whole-schedule gain at both batch 128 and 1024; otherwise delete both
   probes and stop.
5. Only after a probe passes, add the artifact cutover, certificate, and Rust
   validation/load path.
6. Generate a complete `qq_Z6g` artifact and stop immediately if generation
   time, artifact size, load time, or RSS violates the resource gates.
7. Run exact numerical and short alternating batch 128/1024 go/no-go timing.
8. Continue to the full matrix only if both target batches improve by at least
   10%. Delete the candidate and do not land if either target misses.

## Acceptance

The final baseline and candidate must be freshly generated from the same exact
source/runtime and selected flow. Performance requires at least seven
alternating five-second samples:

- median gain at least 10% at batch 128 and batch 1024;
- separation greater than three MAD;
- batch 1 regression no worse than 5%;
- odd-tail batches 127/129 and 1023/1025 remain correct.

Numerical comparison requires compiled baseline versus candidate totals and
resolved contributions with `rtol=1e-12`, `atol=1e-15`, and
`evaluate() == evaluate_resolved().total()`.

Regression coverage includes:

- `qq_Z6g` union-flow;
- NLC and full-color `g g > t t~ + 3g`;
- residual-only `d d~ > Z + 3g`;
- built-in and UFO-SM models;
- a process with four independent quark lines in LC, NLC, and full color;
- parameter mutation, selectors, structural zeros, and odd batches;
- eager and other non-target modes, with no regression beyond 2% or three
  MAD.

Warmed execution must allocate nothing. Generation time, artifact size, load
time, and peak RSS may not worsen by more than 10%. Native AArch64 gates run
under the 30 GiB watchdog; x86-64 compile/tests use existing CI.

Only an accepted, fully gated candidate may be merged and pushed to `main`.
The landing handoff must include the exact SHA, build/runtime identity,
numerical tolerances, regression gates, and raw benchmark evidence.

## Working protocol

Generation/composition, runtime/certificate validation, and independent
correctness/performance review use isolated subagent ownership. No command may
request escalation. Table-filling support remains in its dedicated parallel
task and does not own or block optimization source work.
