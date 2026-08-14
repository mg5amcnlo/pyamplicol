---
title: "Berends-Giele Closure-Anchor Study"
nav_order: 3
parent: "Profiling and Benchmarking"
---
<!-- SPDX-License-Identifier: 0BSD -->

# Berends-Giele closure-anchor study

## Result

For the tested leading-colour processes
`d d~ > Z + 2, 3, 4, 5 gluons`, the existing **right/terminal closure
anchor is the best default**.

- In the topology-replay workload (one colour flow, all helicities), right was
  12.5--25.1% faster than left in the one-second campaign. A five-second
  Z+5g confirmation measured 27.50 us/point for right and 31.78 us/point for
  left, a 15.6% difference.
- In the all-flow-union workload (all colour flows, one helicity), right and
  left have identical structural work. Right was slightly faster at the larger
  multiplicities. The mixed-endpoint policy was never best: it enlarged the
  union DAG and was 7.3% slower than right in the five-second Z+5g
  confirmation.
- Generating and summing both rootings of the same colour-ordered amplitude is
  not a valid optimization: it duplicates the physical closure. A useful
  "both" experiment must choose one endpoint per physical flow and share the
  resulting currents. That is the mixed policy measured here.

The result is process- and ordering-specific evidence, not a proof that the
terminal endpoint wins for every model. A general optimizer should first
minimize unique currents and contributions per replay-equivalence class, then
use a short warmed runtime trial only when the structural candidates tie.

## What left, right, and both mean

The production recurrence already enumerates every disjoint binary split of a
current. Reversing that subset loop is therefore not a distinct left-to-right
or right-to-left Berends-Giele recurrence. The meaningful choice is the
singleton external source used to close the complementary `(n-1)`-source
current:

- **right**: close with the terminal endpoint of the canonical open colour
  line; this is the unchanged production default;
- **left**: close with the initial endpoint of the same line;
- **both/mixed**: in an all-flow union, compare each interior gluon word with
  its reversal and select one endpoint for that physical sector. Each physical
  amplitude is still closed exactly once.

Topology replay has one materialized representative per exact permutation
orbit. Its anchor must map to every replay target under the authenticated
external permutation. For these Z+gluons processes all flows belong to one
such orbit, so a per-flow mixed choice has no distinct topology-replay
evaluator. It is reported as not applicable rather than timing a duplicate of
left or right.

## Measurement setup

- Host: Intel Core i7-8700K, 12 logical CPUs, Linux x86_64, Python 3.12.3.
- Model/backend: packaged built-in SM JIT O2 model, leading colour, native
  recurrence evaluator, one generation worker.
- Processes: `d d~ > Z + n gluons`, `n = 2..5`.
- Workloads: topology replay selects one computed flow and all helicities;
  all-flow union selects one computed nonzero helicity and all flows.
- Sampling: batch 128, two warmups, at least five independent blocks, one
  second target per primary cell. Z+5g confirmation used a five-second target
  and at least seven blocks.
- Each policy at a fixed multiplicity used byte-identical deterministic
  validation momenta. Numerical-current relation discovery was disabled so
  the comparison isolates recurrence topology.
- Source identity: detached experimental snapshot
  `1a66704edc9c88418fffedd9cdf192d1700a7b8b`; native build-input digest
  `e68abcc5f991b3c2fcbbd2e82bc2808c9874cf927ad497aeb821ab7bef3d2022`.
- The complete 20-cell campaign took 540.8 seconds and peaked at 669,667,328
  bytes (0.624 GiB) process-tree RSS.

The wall rows below are ordinary warmed native runtime measurements. Parent
pair visits, emitted rows, and serialized bytes are structural work proxies;
they are not claimed to be exact FLOP counts. RSE is the relative standard
error of the mean.

## Topology-replay results

Right and left emitted exactly the same number of currents, contributions,
finalizations, closures, and plan bytes. Their runtime difference therefore
comes from schedule orientation/order rather than fewer algebraic rows.

| Final-state gluons | Right wall us/point (RSE) | Left wall us/point (RSE) | Left overhead | Currents / contributions | Plan bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.717 (1.37%) | 1.933 (0.39%) | 12.5% | 69 / 126 | 30,088 |
| 3 | 4.674 (1.25%) | 5.418 (0.36%) | 15.9% | 155 / 410 | 69,384 |
| 4 | 10.918 (0.28%) | 13.188 (0.29%) | 20.8% | 333 / 1,202 | 164,488 |
| 5 | 27.513 (1.10%) | 34.410 (1.79%) | 25.1% | 695 / 3,258 | 446,792 |

## All-flow-union results

The one-sided policies again have identical structural counts. Mixed/both
retains currents demanded by opposite anchors in different sectors, which
reduces rather than improves cross-flow sharing for this process family.

| Final-state gluons | Right wall us/point (RSE) | Left wall us/point (RSE) | Mixed wall us/point (RSE) | Left vs right | Mixed vs right |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.844 (2.14%) | 0.823 (1.16%) | 0.988 (2.66%) | -2.5% | +17.1% |
| 3 | 2.794 (0.17%) | 2.867 (0.85%) | 3.176 (2.54%) | +2.6% | +13.7% |
| 4 | 12.653 (3.09%) | 13.318 (1.90%) | 12.999 (0.90%) | +5.3% | +2.7% |
| 5 | 73.267 (0.67%) | 75.562 (0.58%) | 84.381 (1.14%) | +3.1% | +15.2% |

| Final-state gluons | Currents right -> mixed | Contributions right -> mixed | Parent-pair visits right -> mixed | Plan bytes right -> mixed |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 15 -> 16 | 18 -> 19 | 31 -> 55 | 13,576 -> 13,960 |
| 3 | 46 -> 49 | 100 -> 104 | 136 -> 272 | 32,264 -> 33,160 |
| 4 | 184 -> 197 | 625 -> 652 | 710 -> 1,527 | 99,016 -> 102,024 |
| 5 | 919 -> 986 | 4,336 -> 4,536 | 4,369 -> 9,840 | 419,016 -> 432,712 |

At Z+5g, mixed has 7.3% more currents, 4.6% more contributions,
125% more visited parent pairs, and a 3.3% larger serialized plan than either
one-sided policy.

## Longer Z+5g confirmation

| Layout | Right wall us/point (RSE) | Left wall us/point (RSE) | Mixed wall us/point (RSE) |
| --- | ---: | ---: | ---: |
| Topology replay | 27.50 (1.04%) | 31.78 (0.46%) | not applicable |
| All-flow union | 69.21 (0.21%) | 70.32 (0.69%) | 74.24 (0.22%) |

This confirmation makes the conclusion independent of the noisier individual
one-second cells: topology right is clearly preferable, all-flow right and
left are close, and mixed is clearly slower.

## Generation cost

The direction-sensitive recurrence-construction phase remained below one
second in every cell. Total artifact-generation wall time was mostly 20--23
seconds and was dominated by invariant model loading, validation, and native
preparation, so it is not a useful anchor-policy discriminator.

| Final-state gluons | Topology construction s, right / left | All-flow construction s, right / left / mixed |
| ---: | ---: | ---: |
| 2 | 0.283 / 0.300 | 0.244 / 0.264 / 0.292 |
| 3 | 0.345 / 0.374 | 0.280 / 0.288 / 0.300 |
| 4 | 0.468 / 0.456 | 0.312 / 0.344 / 0.346 |
| 5 | 0.722 / 0.736 | 0.679 / 0.695 / 0.693 |

Because mixed was never faster at evaluation, it has no amortization
break-even point in this campaign.

## Correctness

For each multiplicity and layout, the complete resolved helicity-by-colour
component array from left and mixed was compared with right at the same saved
phase-space point. The arrays contained 96, 576, 4,608, and 46,080 components
for Z+2g through Z+5g. Every comparison passed. Across the campaign, the
largest absolute residual was `3.79e-19` and the largest relative residual was
`2.13e-12`, below the configured `1e-11` tolerance.

The production default remains byte-identical: omitting the experimental
policy and explicitly selecting right produce the same canonical recurrence
input digest.

## Recommended generation rule

1. Keep exactly one closure anchor per physical amplitude and per topology
   replay orbit.
2. Compare candidate anchors using unique emitted currents/contributions,
   live workspace, and serialized plan size before compiling an evaluator.
3. Reject a mixed union when it grows those counts, as it does here.
4. If one-sided candidates tie structurally, retain the canonical right anchor
   or run one short warmed native profile. Static operation counts cannot see
   the schedule-order effect measured for topology replay.

For the current Z+gluons workload, this rule selects the existing right anchor;
no production default change is justified.
