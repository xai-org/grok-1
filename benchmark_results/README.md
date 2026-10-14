# Benchmark results

Fresh benchmark artifacts for PR #434 should be generated from the corrected
branch with:

```bash
python examples/run_triton_rope_h100_matrix.py \
  --output-dir benchmark_results/h100
```

Do not copy performance numbers from pre-fix revisions.

A complete H100 result set contains:

- `manifest.json`
- `rope_h100_fp16_h8_contiguous.json`
- `rope_h100_fp16_h8_noncontiguous.json`
- `rope_h100_fp16_h48_contiguous.json`
- `rope_h100_fp16_h48_noncontiguous.json`
- `rope_h100_bf16_h8_contiguous.json`
- `rope_h100_bf16_h8_noncontiguous.json`
- `rope_h100_bf16_h48_contiguous.json`
- `rope_h100_bf16_h48_noncontiguous.json`

Each per-case JSON must come directly from `examples/triton_rope_fused.py` and
include every requested sequence length. `manifest.json` ties the result set to
the exact Git commit and hardware/software environment.

Before committing results, verify:

- all correctness checks completed successfully;
- no result files were hand-edited;
- the manifest commit matches the tested branch head;
- the GPU is identified as the actual test device;
- latency/speedup claims in the PR are copied from these fresh files only.
