# SymJIT terminal-superkernel probe

This is a disposable, standalone DirectApplication schedule harness. It reads
the strict JSON request written by `compiled_terminal_superkernel_probe.py`,
lowers the thirteen baseline leaves and both candidate leaves through the
pinned factor-free identity-overwrite DirectApplication API, binds the same
persistent 32-point split-plane tile used by production, repeats that tile for
logical batches 128 and 1024, checks deterministic numerical parity, and
records alternating schedule timings.

Bind an explicit configured checkout before invoking Cargo:

```console
SYMJIT_PATH=/absolute/path/to/configured/symjit ./configure.py
cargo run --release -- --request /absolute/request.json --output /absolute/result.json
```

The package deliberately has no fallback registry SymJIT dependency. The
generated `.symjit-path` symlink and Cargo build output are ignored. A warmed
schedule call is allocation-counted, and any nonzero allocation or numerical
mismatch is reported fail-closed.
