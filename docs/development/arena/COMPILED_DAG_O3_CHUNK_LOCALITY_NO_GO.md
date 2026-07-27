# Compiled-DAG O3 Chunk-Locality Experiment: No-Go

## Decision

Do not change the production compiled-DAG output chunk policy.

Reducing the static output chunk from 512 to 256 improved selected-flow
`u u~ > Z + 6g` execution by only 3.40% at batch 128 and 4.18% at batch
1024. Reducing it again to 128 was slower than 256 and improved batch 128
by only 2.63% relative to 512. These results are well below the required
10% improvement at both primary batches.

The experiment changed no production source. It must not be merged to
`main`.

## Exact Identity

- Source revision:
  `6c07f5d689cc618f9436f43d9ad373ef6b5ec169`
- Feature branch: `codex/dag_hit_compiled_chunk_locality`
- Candidate fingerprint: `c2b7cc28699b`
- Native build-input digest:
  `77f8c739ec14968af61450249fbc035561269b4b01f30f722d15a026425e1b65`
- Native module digest:
  `af81d41f8ad8f24cbce4c2ac09357ce772aa5baf78804d96749fd8888d1b5bda`
- Host: Apple silicon MacBook M3, AArch64
- Model: built-in Standard Model
- Process: `u u~ > Z + 6g`
- Selector: `flow:2,4,5,6,7,8,9,1`
- Execution: compiled DAG, JIT, O3, LC selected flow,
  topology-replay compression
- Workers and generation cores: one

The three artifacts differ only in `generation.compiled.output_chunk_size`.
They were generated and timed with the same installed exact-source runtime.

## Schedule Census

The selected process has stage output counts

```text
220, 360, 560, 800, 576, 640, 768
```

and 384 amplitude outputs. The static policies therefore produce:

| Output chunk | Compiled leaves |
|---:|---:|
| 512 | 13 |
| 256 | 21 |
| 128 | 36 |

## Benchmark Protocol

Measurements used `tools/developer/compiled_mode_sample.py` and the native
`_benchmark_f64_wall_time` path. Each result contains five warmed samples
with a one-second target per sample and two warmups. Times are complete
evaluation wall time per phase-space point. The 512 and 256 artifacts were
measured at batches 128 and 1024. The 128 artifact was stopped after batch
128 because it was already slower than 256 and could no longer satisfy the
required both-batch gate.

This was a bounded go/no-go experiment, not publication timing. It is
sufficient to reject the candidate by a wide margin from the 10% threshold.

## Timing Results

| Policy | Batch | Median (µs/point) | Gain vs 512 |
|---|---:|---:|---:|
| 512 | 128 | 41.2422 | control |
| 256 | 128 | 39.8405 | 3.40% |
| 128 | 128 | 40.1591 | 2.63% |
| 512 | 1024 | 41.3413 | control |
| 256 | 1024 | 39.6151 | 4.18% |
| 128 | 1024 | not run | rejected at batch 128 |

The exact samples, in seconds per point, were:

```text
512, batch 128:
4.124223991935484e-05
4.255235635080645e-05
4.1131520917338714e-05
4.140167162298387e-05
4.1113984879032255e-05

256, batch 128:
4.0202799609375e-05
4.0244409179687504e-05
3.9548331640625004e-05
3.9840519140625004e-05
3.949755859375e-05

128, batch 128:
4.0599210546875e-05
3.99556234375e-05
4.028818359375e-05
4.0159057617187496e-05
4.00686361328125e-05

512, batch 1024:
4.1952416992187505e-05
4.1125008203125e-05
4.1341259765625e-05
4.106943359375e-05
4.762568359375e-05

256, batch 1024:
4.0648966406249995e-05
3.96615640625e-05
3.96006591796875e-05
3.9615087890625e-05
3.959197578125e-05
```

The isolated high 512/batch-1024 sample makes the 4.18% median comparison
slightly favorable to the candidate. Excluding it does not produce anything
close to a 10% improvement.

## Numerical Gate

Every timing run completed the evaluator's internal
`evaluate() == evaluate_resolved().total()` validation. The three artifacts
returned the same physical totals, with only a few last-bit differences from
the changed partitioning/order. The observed differences were within the
compiled-mode numerical contract (`rtol=1e-12`, `atol=1e-15`).

No numerical failure caused the performance rejection.

## Artifact Cost

| Policy | Artifact disk use | Evaluator PACBIN | Execution JSON |
|---|---:|---:|---:|
| 512 | 23,076 KiB | 7,801,592 B | 15,064,838 B |
| 256 | 23,208 KiB | 7,877,480 B | 15,122,104 B |
| 128 | 23,444 KiB | 8,041,336 B | 15,199,118 B |

Smaller chunks increase leaf count and artifact size. The modest locality
benefit at 256 is therefore partly offset by duplicated expressions and more
application boundaries. At 128 that balance has already turned negative.

## Why This Closes the Tractable Slice

Independent profiling and review show:

- roughly 94–95% of full execution wall time is already inside the generated
  evaluator schedule;
- about 97% of profiler samples land in JIT mappings;
- roughly 86% of those JIT samples are concentrated in six large executable
  regions;
- arena binding, dispatch, parameter preparation, tails, and other runtime
  plumbing together have an absolute ceiling below the required 10%;
- reducing calls and exposed planes by about 32% in the rejected terminal
  superkernel experiment improved execution by only 0.47% at batch 128 and
  0.21% at batch 1024;
- prior DirectTable variants regressed by 24–33%;
- increasing chunks from 512 to 1024 regressed, while this experiment proves
  that reducing them to 256 or 128 also cannot reach 10%.

A stage-local mix could recover only an incremental subset of the measured
256 benefit. With uniform 256 delivering 3–4% and 128 already reversing,
there is no evidence that a static stage mix could cross 10% at both primary
batches.

The remaining technically credible double-digit direction is true
cross-application shared-code or MIR outlining to reduce the repeated JIT
instruction footprint. That requires a substantial SymJIT/backend project
and a new compiled-code ABI. It is outside the requested tractable
pyAmpliCol-only slice and conflicts with the constraint to avoid significant
new dependency work.

The optimization frontier under the current constraints is therefore closed:
no candidate from this experiment should land.
