#!/usr/bin/env python3
"""Run the complete Grok-1 Triton RoPE validation/benchmark matrix.

This driver intentionally does not contain expected performance numbers. It runs
all required H100 validation cases against the current kernel and writes one JSON
file per case so results are attributable to dtype/head-count/layout.

Run from the repository root:

    python examples/run_triton_rope_h100_matrix.py --output-dir benchmark_results/h100

Requires CUDA, PyTorch, and Triton. The underlying benchmark performs a
correctness check before timing every sequence length.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton


SEQUENCE_LENGTHS = (512, 1024, 2048, 4096, 8192, 16384)
DTYPES = ("fp16", "bf16")
HEAD_COUNTS = (8, 48)
LAYOUTS = ((False, "contiguous"), (True, "noncontiguous"))


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_case(
    *,
    dtype: str,
    heads: int,
    noncontiguous: bool,
    output_path: Path,
    warmup: int,
    rep: int,
) -> None:
    command = [
        sys.executable,
        "examples/triton_rope_fused.py",
        "--benchmark",
        "--sequence-lengths",
        *map(str, SEQUENCE_LENGTHS),
        "--heads",
        str(heads),
        "--dim",
        "128",
        "--dtype",
        dtype,
        "--warmup",
        str(warmup),
        "--rep",
        str(rep),
        "--json",
        str(output_path),
    ]
    if noncontiguous:
        command.append("--noncontiguous")

    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/h100"),
        help="directory for per-case JSON outputs and manifest",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run the broader edge-case matrix once before performance measurements.
    print("$", sys.executable, "examples/triton_rope_fused.py --check", flush=True)
    subprocess.run(
        [sys.executable, "examples/triton_rope_fused.py", "--check"], check=True
    )

    output_files: list[str] = []
    for dtype in DTYPES:
        for heads in HEAD_COUNTS:
            for noncontiguous, layout in LAYOUTS:
                name = f"rope_h100_{dtype}_h{heads}_{layout}.json"
                path = args.output_dir / name
                run_case(
                    dtype=dtype,
                    heads=heads,
                    noncontiguous=noncontiguous,
                    output_path=path,
                    warmup=args.warmup,
                    rep=args.rep,
                )
                output_files.append(name)

    props = torch.cuda.get_device_properties(0)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "host": platform.platform(),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": f"{props.major}.{props.minor}",
        "gpu_total_memory_bytes": props.total_memory,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "sequence_lengths": list(SEQUENCE_LENGTHS),
        "dtypes": list(DTYPES),
        "head_counts": list(HEAD_COUNTS),
        "layouts": [name for _, name in LAYOUTS],
        "head_dim": 128,
        "warmup": args.warmup,
        "rep": args.rep,
        "result_files": output_files,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nComplete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
