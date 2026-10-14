# Grok-1 Triton RoPE validation

This directory contains an **optional PyTorch/Triton reference implementation**
of the rotary embedding math used by `model.py::RotaryEmbedding`.
It does **not** modify the JAX runtime, `model.py`, or core requirements.

## Current status

- RoPE frequency layout corrected to match Grok-1's JAX implementation.
- Triton launch geometry corrected.
- Correctness coverage includes FP16, BF16, contiguous, non-contiguous,
  in-place, out-of-place, offsets, multiple head counts, and neighboring dims.
- A complete H100 benchmark matrix runner is included.
- **No performance number from the pre-fix revision should be used.**
  The original `2.81 ms -> 0.97 ms / 2.9x` result was measured before the
  frequency-layout correction and is therefore invalid for the current code.
- Fresh H100 results must be generated from the current branch before publishing
  a replacement performance claim.

## Reviewer quick start

Install optional benchmark dependencies in a CUDA-capable environment:

```bash
python -m venv .venv-triton
source .venv-triton/bin/activate
pip install torch triton
```

Run the correctness matrix:

```bash
python examples/triton_rope_fused.py --check
```

Run the complete H100 validation/benchmark matrix:

```bash
python examples/run_triton_rope_h100_matrix.py \
  --output-dir benchmark_results/h100
```

The matrix covers:

- dtypes: FP16 and BF16
- heads: 8 and 48
- head dimension: 128
- sequence lengths: 512, 1024, 2048, 4096, 8192, 16384
- layouts: contiguous and non-contiguous
- correctness check before every timed sequence length
- latency, speedup, and maximum absolute error in JSON output

The runner also writes `manifest.json` with the Git commit, GPU name, compute
capability, GPU memory, Python, PyTorch, CUDA runtime, Triton version, warmup,
and repetition count.

## Reviewer acceptance checklist

- [ ] `python examples/triton_rope_fused.py --check` passes.
- [ ] FP16 cases pass with the encoded FP16 tolerance.
- [ ] BF16 cases pass with the encoded BF16 tolerance.
- [ ] 8-head `D=128` runs complete through 8192 tokens.
- [ ] 48-head `D=128` runs complete through 8192 tokens.
- [ ] 16K stress cases complete where memory permits.
- [ ] Non-contiguous cases pass correctness before timing.
- [ ] In-place output matches the reference and reuses input storage.
- [ ] Each result JSON reports latency, speedup, and max absolute error.
- [ ] `manifest.json` records the exact software/hardware environment and commit.

## Grok-1 RoPE convention

The benchmark uses the same frequency layout as `model.py::RotaryEmbedding`:

```python
phase = jnp.einsum("bi,j->bij", t, inv_freq)
phase = jnp.tile(phase, reps=(1, 2))[:, :, None, :]
x = x * jnp.cos(phase) + rotate_half(x) * jnp.sin(phase)
```

The PyTorch reference therefore concatenates the half-width phase vector with
itself. It does not use `repeat_interleave`.

## Measurement policy

- Every timed case performs a correctness comparison first.
- Timing uses `triton.testing.do_bench` rather than Python wall-clock timing.
- Cos/sin cache construction is excluded from both timed paths.
- The baseline is the eager PyTorch translation of the same RoPE operation.
- Kernel microbenchmark speedup is **not** an end-to-end Grok/JAX speedup claim.
- Fresh results should be tied to the exact Git commit and hardware manifest.

## Numerical tolerances

The current harness uses:

- FP16: `atol=2e-3`, `rtol=2e-3`
- BF16: `atol=2e-2`, `rtol=2e-2`

These tolerances reflect the lower precision of the respective formats while
remaining strict enough to catch material RoPE-layout or indexing errors.
Maximum absolute error is recorded separately for every benchmarked shape.

## Tuning policy

The current kernel uses a simple launch heuristic: four warps for feature blocks
up to 256 values and eight warps above that. It does not claim that this is the
optimal H100 configuration.

Autotune changes should be driven by the fresh H100 matrix, especially the
48-head and non-contiguous cases. Any new configuration should be retained only
if it improves measured latency without changing numerical results.

## Scope

This PR is intentionally limited to an optional PyTorch/Triton reference and
microbenchmark for Grok-1-style RoPE. It does not modify the existing JAX
runtime or `model.py`, and it does not claim an equivalent full-model speedup.
