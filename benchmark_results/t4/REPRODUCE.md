# Reproduce PR #434 T4 evidence

Measured at PR commit: `2e9d5df0fc982efd91982cb50c303059f8be0498` (xai-org/grok-1#434).
Harness: `examples/colab_t4_rope_eval.py` sha256 `fc8671986e4b396c18e2a1dae881d09ee80261601eb2c124e0118ec67616cdc4` (the file that produced `results.json` and `manifest.json` in this directory).

## Colab (free tier, Tesla T4)
1. Runtime -> Change runtime type -> T4 GPU.
2. `!nvidia-smi --query-gpu=name,compute_cap --format=csv` -> expect `Tesla T4, 7.5`.
3. Upload `examples/colab_t4_rope_eval.py` (files pane or `from google.colab import files; files.upload()`).
4. `!sha256sum colab_t4_rope_eval.py` -> must equal `fc8671986e4b396c18e2a1dae881d09ee80261601eb2c124e0118ec67616cdc4`.
5. Quick (~minutes): `!python colab_t4_rope_eval.py --mode all --quick --out /content/rope_t4_v3q`
6. Full: `!python colab_t4_rope_eval.py --mode all --out /content/rope_t4_v3`
7. Do NOT pass `--with-sincos` expecting a pass: that variant failed the gate on T4 (see the PR comment).
8. Download `results.json` and `manifest.json`. A run is valid only if `verdict == "GATE_PASSED"`, `gpu == "Tesla T4"`, and `script_sha256` matches.
9. Compare your `results.json` and `manifest.json` with `benchmark_results/t4/results.json` and `benchmark_results/t4/manifest.json` in this PR.

Original run environment (manifest): torch 2.11.0+cu130, Triton 3.6.0, CUDA 13.0, driver 580.82.07. Other versions may give different timings.

## PR file
`examples/triton_rope_fused.py` sha256 `bb954b02af919446a973e22111089941edc6889e0c9bfcdc8b41c772870b95c8`. Kernel/launcher/half-table code of v2 is byte-identical to the measured harness. On a CUDA box:
`python examples/triton_rope_fused.py --check` then `python examples/run_triton_rope_h100_matrix.py --output-dir benchmark_results/h100` (H100 non-regression: OPEN).
