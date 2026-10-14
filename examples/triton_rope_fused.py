# Copyright 2024 X.AI Corp.
# Copyright 2026 MiMindMendinc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused Triton implementation of Grok-1 style rotary embeddings.

This is an optional PyTorch/Triton reference and microbenchmark. It intentionally
lives under ``examples/`` and does not add PyTorch/Triton to Grok-1's JAX runtime
requirements.

The math matches ``model.py::RotaryEmbedding`` exactly:

    phase = tile(phase_half, reps=(1, 2))
    out = x * cos(phase) + rotate_half(x) * sin(phase)

Typical use:

    python examples/triton_rope_fused.py --check
    python examples/triton_rope_fused.py --benchmark --json results.json

Requires a CUDA GPU plus PyTorch and Triton.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import triton
import triton.language as tl


def _interpreting() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


@triton.jit
def _fused_rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    stride_x_b,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_o_b,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_cos_t,
    stride_cos_d,
    stride_sin_t,
    stride_sin_d,
    T,
    H,
    HALF: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    """One program owns BLOCK_H complete feature rows of one (b, t) position.

    Grid is ``(B * T, cdiv(H, BLOCK_H))``. Each program:

    * loads the two halves of each row as separate contiguous vectors, so no
      gather (``paired_d``) and no sign-select are needed;
    * loads the cos/sin row for position ``t`` once and broadcasts it across
      all BLOCK_H heads (the PR head reloads it once per head);
    * computes in fp32 and casts back on store.

    A program writes exactly the rows it read, so in-place use cannot race
    another program reading the paired half of the same row.
    """
    bt = tl.program_id(0)
    hb = tl.program_id(1)
    t = bt % T
    b = bt // T

    h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    d = tl.arange(0, BLOCK_HALF)
    h_mask = h < H
    d_mask = d < HALF
    mask = h_mask[:, None] & d_mask[None, :]

    x_row = x_ptr + b * stride_x_b + t * stride_x_t + h[:, None] * stride_x_h
    o_row = out_ptr + b * stride_o_b + t * stride_o_t + h[:, None] * stride_o_h
    x1_ptrs = x_row + d[None, :] * stride_x_d
    x2_ptrs = x_row + (d[None, :] + HALF) * stride_x_d
    o1_ptrs = o_row + d[None, :] * stride_o_d
    o2_ptrs = o_row + (d[None, :] + HALF) * stride_o_d

    x1 = tl.load(x1_ptrs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptrs, mask=mask, other=0.0).to(tl.float32)

    c_row = cos_ptr + t * stride_cos_t
    s_row = sin_ptr + t * stride_sin_t
    c1 = tl.load(c_row + d * stride_cos_d, mask=d_mask, other=0.0).to(tl.float32)
    c2 = tl.load(c_row + (d + HALF) * stride_cos_d, mask=d_mask, other=0.0).to(tl.float32)
    s1 = tl.load(s_row + d * stride_sin_d, mask=d_mask, other=0.0).to(tl.float32)
    s2 = tl.load(s_row + (d + HALF) * stride_sin_d, mask=d_mask, other=0.0).to(tl.float32)

    # rotate_half(x) = cat(-x2, x1)
    out1 = x1 * c1[None, :] - x2 * s1[None, :]
    out2 = x2 * c2[None, :] + x1 * s2[None, :]

    tl.store(o1_ptrs, out1.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(o2_ptrs, out2.to(out_ptr.dtype.element_ty), mask=mask)


def rotate_half_reference(x: torch.Tensor) -> torch.Tensor:
    """PyTorch equivalent of Grok-1 ``model.py::rotate_half``."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def make_grok_rope_cache(
    sequence_length: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    base_exponent: float = 10000.0,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin tables with the same frequency layout as Grok-1 JAX.

    Grok-1 computes frequencies for dimensions ``0, 2, ...`` and then tiles the
    half-width phase vector twice. ``torch.cat((phase, phase), dim=-1)`` is the
    PyTorch equivalent. This is deliberately *not* ``repeat_interleave``.
    """
    if dim <= 0 or dim % 2:
        raise ValueError("dim must be a positive even integer")

    exponents = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base_exponent ** (exponents / dim))
    positions = torch.arange(
        offset,
        offset + sequence_length,
        device=device,
        dtype=torch.float32,
    )
    phase_half = torch.outer(positions, inv_freq)
    phase = torch.cat((phase_half, phase_half), dim=-1)
    return torch.cos(phase).to(dtype), torch.sin(phase).to(dtype)


def pytorch_rope_reference(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Literal PyTorch translation of Grok-1 RotaryEmbedding math."""
    return x * cos[None, :, None, :] + rotate_half_reference(x) * sin[None, :, None, :]


def default_launch_config(heads: int, dim: int) -> tuple[int, int, int]:
    """Return ``(block_h, num_warps, num_stages)`` for a (heads, dim) shape.

    Heuristic, not autotune: ``triton.autotune`` re-runs the kernel while
    benchmarking candidates, which corrupts data on the in-place path unless
    every config restores its inputs. Fixed configs keep in-place safe and
    make results reproducible across GPUs. Sizes target ~2K elements per
    program at D=128 (16 heads x 128), down from 128 elements in the PR head.
    """
    block_half = triton.next_power_of_2(dim // 2)
    block_h = min(triton.next_power_of_2(heads), max(1, 1024 // block_half))
    elems = block_h * block_half * 2
    num_warps = 2 if elems <= 512 else (4 if elems <= 2048 else 8)
    return block_h, num_warps, 1


def fused_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    inplace: bool = False,
    block_h: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """Apply Grok-style RoPE to ``x`` shaped ``(B, T, H, D)``.

    Supports contiguous and non-contiguous input layouts through explicit
    strides. ``cos`` and ``sin`` must be CUDA tensors shaped ``(T, D)``.
    ``block_h``/``num_warps``/``num_stages`` override the launch heuristic
    (used by tuning sweeps); they never change the math.
    """
    if not _interpreting() and (not x.is_cuda or not cos.is_cuda or not sin.is_cuda):
        raise ValueError("x, cos, and sin must be CUDA tensors")
    if x.ndim != 4:
        raise ValueError(f"x must have shape (B, T, H, D); got {tuple(x.shape)}")

    B, T, H, D = x.shape
    if D % 2:
        raise ValueError("RoPE feature dimension must be even")
    if cos.shape != (T, D) or sin.shape != (T, D):
        raise ValueError(
            f"cos/sin must both have shape {(T, D)}; got {tuple(cos.shape)} and {tuple(sin.shape)}"
        )
    if cos.dtype != x.dtype or sin.dtype != x.dtype:
        raise ValueError("x, cos, and sin must use the same dtype")

    out = x if inplace else torch.empty_like(x, memory_format=torch.preserve_format)
    half = D // 2
    block_half = triton.next_power_of_2(half)
    if block_half > 32768:
        raise ValueError(f"unsupported feature dimension D={D}")

    d_bh, d_nw, d_ns = default_launch_config(H, D)
    block_h = d_bh if block_h is None else block_h
    num_warps = d_nw if num_warps is None else num_warps
    num_stages = d_ns if num_stages is None else num_stages
    if block_h < 1 or block_h & (block_h - 1):
        raise ValueError("block_h must be a positive power of two")

    grid = (B * T, triton.cdiv(H, block_h))
    _fused_rope_kernel[grid](
        x,
        cos,
        sin,
        out,
        *x.stride(),
        *out.stride(),
        *cos.stride(),
        *sin.stride(),
        T,
        H,
        HALF=half,
        BLOCK_H=block_h,
        BLOCK_HALF=block_half,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


@triton.jit
def _rope_v2_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    inv_freq_ptr,
    out_ptr,
    stride_x_b,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_o_b,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_c_t,
    stride_c_d,
    stride_s_t,
    stride_s_d,
    T,
    H,
    pos_offset,
    HALF: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    SINCOS: tl.constexpr,
):
    """2D tile over (tokens, heads) using a HALF-width cos/sin table.

    Grid: ``(B * cdiv(T, BLOCK_T), cdiv(H, BLOCK_H))``. Each program owns
    BLOCK_T x BLOCK_H complete rows and writes only what it read (in-place safe).

    Grok-1's phase is ``tile(phase_half, (1, 2))`` so cos[:, :D/2] == cos[:, D/2:].
    The table is therefore ``(T, D/2)``: half the table bytes of the full layout,
    loaded once per tile and broadcast across BLOCK_H heads.

    SINCOS=True (optional) skips the table and evaluates cos/sin in fp32 from
    ``(t + pos_offset) * inv_freq``; zero table traffic, different rounding.
    """
    pid_bt = tl.program_id(0)
    pid_h = tl.program_id(1)
    n_tb = tl.cdiv(T, BLOCK_T)
    b = pid_bt // n_tb
    tb = pid_bt % n_tb

    t = tb * BLOCK_T + tl.arange(0, BLOCK_T)
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d = tl.arange(0, BLOCK_HALF)
    t_m = t < T
    h_m = h < H
    d_m = d < HALF

    t3 = t[:, None, None]
    h3 = h[None, :, None]
    d3 = d[None, None, :]
    m3 = t_m[:, None, None] & h_m[None, :, None] & d_m[None, None, :]

    xb = x_ptr + b * stride_x_b + t3 * stride_x_t + h3 * stride_x_h
    ob = out_ptr + b * stride_o_b + t3 * stride_o_t + h3 * stride_o_h
    x1 = tl.load(xb + d3 * stride_x_d, mask=m3, other=0.0).to(tl.float32)
    x2 = tl.load(xb + (d3 + HALF) * stride_x_d, mask=m3, other=0.0).to(tl.float32)

    m2 = t_m[:, None] & d_m[None, :]
    if SINCOS:
        inv = tl.load(inv_freq_ptr + d, mask=d_m, other=0.0)
        ph = (t + pos_offset).to(tl.float32)[:, None] * inv[None, :]
        c = tl.cos(ph)
        s = tl.sin(ph)
    else:
        c = tl.load(cos_ptr + t[:, None] * stride_c_t + d[None, :] * stride_c_d, mask=m2, other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + t[:, None] * stride_s_t + d[None, :] * stride_s_d, mask=m2, other=0.0).to(tl.float32)
    c3 = c[:, None, :]
    s3 = s[:, None, :]

    out1 = x1 * c3 - x2 * s3
    out2 = x2 * c3 + x1 * s3
    tl.store(ob + d3 * stride_o_d, out1.to(out_ptr.dtype.element_ty), mask=m3)
    tl.store(ob + (d3 + HALF) * stride_o_d, out2.to(out_ptr.dtype.element_ty), mask=m3)


def make_grok_rope_cache_half(
    sequence_length: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    base_exponent: float = 10000.0,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Half-width ``(T, D/2)`` cos/sin; equals the first half of the full Grok table."""
    if dim <= 0 or dim % 2:
        raise ValueError("dim must be a positive even integer")
    exponents = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base_exponent ** (exponents / dim))
    positions = torch.arange(offset, offset + sequence_length, device=device, dtype=torch.float32)
    phase_half = torch.outer(positions, inv_freq)
    return torch.cos(phase_half).to(dtype), torch.sin(phase_half).to(dtype)


def grok_inv_freq(dim: int, *, device: torch.device, base_exponent: float = 10000.0) -> torch.Tensor:
    exponents = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    return 1.0 / (base_exponent ** (exponents / dim))


def default_launch_config_v2(heads: int, dim: int) -> tuple[int, int, int, int]:
    """Return ``(block_t, block_h, num_warps, num_stages)``; ~4K elements/program at D=128."""
    block_half = triton.next_power_of_2(dim // 2)
    block_h = min(triton.next_power_of_2(heads), max(1, 1024 // block_half))
    block_t = max(1, 2048 // (block_h * block_half))
    elems = block_t * block_h * block_half * 2
    num_warps = 4 if elems <= 2048 else 8
    return block_t, block_h, num_warps, 1


def fused_rope_v2(
    x: torch.Tensor,
    cos_half: torch.Tensor | None,
    sin_half: torch.Tensor | None,
    *,
    inplace: bool = False,
    sincos: bool = False,
    inv_freq: torch.Tensor | None = None,
    pos_offset: int = 0,
    block_t: int | None = None,
    block_h: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """Grok-style RoPE on ``x`` (B, T, H, D) with half-width ``(T, D/2)`` tables.

    Default: table path. ``sincos=True`` requires fp32 ``inv_freq`` of shape (D/2,)
    and computes cos/sin in-kernel (optional; see accuracy note in docs).
    """
    if x.ndim != 4:
        raise ValueError(f"x must have shape (B, T, H, D); got {tuple(x.shape)}")
    B, T, H, D = x.shape
    if D % 2:
        raise ValueError("RoPE feature dimension must be even")
    half = D // 2
    tensors = [x]
    if sincos:
        if inv_freq is None or inv_freq.shape != (half,) or inv_freq.dtype != torch.float32:
            raise ValueError(f"sincos=True needs fp32 inv_freq of shape {(half,)}")
        tensors.append(inv_freq)
        cos_half = sin_half = inv_freq  # unused pointers; never dereferenced
        cs = (0, 0, 0, 0)
    else:
        if cos_half is None or sin_half is None:
            raise ValueError("cos_half and sin_half are required unless sincos=True")
        if cos_half.shape != (T, half) or sin_half.shape != (T, half):
            raise ValueError(f"cos_half/sin_half must have shape {(T, half)}")
        if cos_half.dtype != x.dtype or sin_half.dtype != x.dtype:
            raise ValueError("x, cos_half, and sin_half must use the same dtype")
        tensors += [cos_half, sin_half]
        cs = (*cos_half.stride(), *sin_half.stride())
        inv_freq = cos_half
    if not _interpreting() and not all(t_.is_cuda for t_ in tensors):
        raise ValueError("all tensors must be CUDA tensors")

    out = x if inplace else torch.empty_like(x, memory_format=torch.preserve_format)
    block_half = triton.next_power_of_2(half)
    d_bt, d_bh, d_nw, d_ns = default_launch_config_v2(H, D)
    block_t = d_bt if block_t is None else block_t
    block_h = d_bh if block_h is None else block_h
    num_warps = d_nw if num_warps is None else num_warps
    num_stages = d_ns if num_stages is None else num_stages
    for name, v in (("block_t", block_t), ("block_h", block_h)):
        if v < 1 or v & (v - 1):
            raise ValueError(f"{name} must be a positive power of two")
    grid = (B * triton.cdiv(T, block_t), triton.cdiv(H, block_h))
    _rope_v2_kernel[grid](
        x, cos_half, sin_half, inv_freq, out,
        *x.stride(), *out.stride(), *cs,
        T, H, pos_offset,
        HALF=half, BLOCK_T=block_t, BLOCK_H=block_h, BLOCK_HALF=block_half, SINCOS=sincos,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def _make_x(
    batch: int,
    sequence_length: int,
    heads: int,
    dim: int,
    *,
    dtype: torch.dtype,
    noncontiguous: bool,
) -> torch.Tensor:
    if noncontiguous:
        src = torch.randn(
            batch,
            heads,
            sequence_length,
            dim,
            device="cuda",
            dtype=dtype,
        )
        return src.transpose(1, 2)
    return torch.randn(
        batch,
        sequence_length,
        heads,
        dim,
        device="cuda",
        dtype=dtype,
    )


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 2e-3, 2e-3
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 1e-5, 1e-5


def check_correctness() -> None:
    """Exercise shape, dtype, layout, offset, and in-place correctness."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton correctness suite")

    cases: Iterable[tuple[int, int, int, int, torch.dtype, bool, int]] = (
        (1, 1, 1, 128, torch.float16, False, 0),
        (1, 17, 8, 128, torch.float16, False, 0),
        (1, 257, 8, 128, torch.float16, True, 0),
        (2, 129, 8, 128, torch.float16, True, 37),
        (1, 257, 8, 128, torch.bfloat16, False, 0),
        (1, 257, 48, 128, torch.bfloat16, True, 19),
        (1, 129, 8, 64, torch.float16, False, 0),
        (1, 129, 8, 256, torch.float16, False, 0),
    )

    for B, T, H, D, dtype, noncontiguous, offset in cases:
        x = _make_x(B, T, H, D, dtype=dtype, noncontiguous=noncontiguous)
        cos, sin = make_grok_rope_cache(
            T,
            D,
            device=x.device,
            dtype=dtype,
            offset=offset,
        )
        expected = pytorch_rope_reference(x, cos, sin)
        actual = fused_rope(x, cos, sin)
        atol, rtol = _tolerances(dtype)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

        x_inplace = x.clone(memory_format=torch.preserve_format)
        expected_inplace = pytorch_rope_reference(x_inplace, cos, sin)
        actual_inplace = fused_rope(x_inplace, cos, sin, inplace=True)
        torch.testing.assert_close(actual_inplace, expected_inplace, atol=atol, rtol=rtol)
        if actual_inplace.data_ptr() != x_inplace.data_ptr():
            raise AssertionError("in-place path did not reuse input storage")

        cos_h, sin_h = make_grok_rope_cache_half(T, D, device=x.device, dtype=dtype, offset=offset)
        if not (torch.equal(cos_h, cos[:, : D // 2]) and torch.equal(cos_h, cos[:, D // 2 :])):
            raise AssertionError("half-width table does not match Grok tile layout")
        torch.testing.assert_close(fused_rope_v2(x, cos_h, sin_h), expected, atol=atol, rtol=rtol)
        x_v2 = x.clone(memory_format=torch.preserve_format)
        got_v2 = fused_rope_v2(x_v2, cos_h, sin_h, inplace=True)
        torch.testing.assert_close(got_v2, expected, atol=atol, rtol=rtol)
        if got_v2.data_ptr() != x_v2.data_ptr():
            raise AssertionError("v2 in-place path did not reuse input storage")

        print(
            "PASS",
            f"B={B} T={T} H={H} D={D}",
            str(dtype).removeprefix("torch."),
            f"noncontiguous={noncontiguous}",
            f"offset={offset}",
        )


@dataclass
class BenchmarkResult:
    batch: int
    sequence_length: int
    heads: int
    dim: int
    dtype: str
    noncontiguous: bool
    pytorch_ms: float
    triton_ms: float
    speedup: float
    max_abs_error: float
    triton_v2_ms: float
    speedup_v2: float
    max_abs_error_v2: float


def _bench_ms(fn, *, warmup: int = 25, rep: int = 100) -> float:
    """Benchmark with Triton's CUDA-event based harness."""
    return float(
        triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            return_mode="median",
        )
    )


def benchmark_case(
    *,
    batch: int,
    sequence_length: int,
    heads: int,
    dim: int,
    dtype: torch.dtype,
    noncontiguous: bool,
    warmup: int,
    rep: int,
) -> BenchmarkResult:
    x = _make_x(
        batch,
        sequence_length,
        heads,
        dim,
        dtype=dtype,
        noncontiguous=noncontiguous,
    )
    cos, sin = make_grok_rope_cache(
        sequence_length,
        dim,
        device=x.device,
        dtype=dtype,
    )

    expected = pytorch_rope_reference(x, cos, sin)
    actual = fused_rope(x, cos, sin)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    max_abs_error = float((actual - expected).abs().max().float().item())
    cos_h, sin_h = make_grok_rope_cache_half(sequence_length, dim, device=x.device, dtype=dtype)
    actual_v2 = fused_rope_v2(x, cos_h, sin_h)
    torch.testing.assert_close(actual_v2, expected, atol=atol, rtol=rtol)
    max_abs_error_v2 = float((actual_v2 - expected).abs().max().float().item())

    pytorch_ms = _bench_ms(
        lambda: pytorch_rope_reference(x, cos, sin),
        warmup=warmup,
        rep=rep,
    )
    triton_ms = _bench_ms(
        lambda: fused_rope(x, cos, sin),
        warmup=warmup,
        rep=rep,
    )

    triton_v2_ms = _bench_ms(lambda: fused_rope_v2(x, cos_h, sin_h), warmup=warmup, rep=rep)

    return BenchmarkResult(
        batch=batch,
        sequence_length=sequence_length,
        heads=heads,
        dim=dim,
        dtype=str(dtype).removeprefix("torch."),
        noncontiguous=noncontiguous,
        pytorch_ms=pytorch_ms,
        triton_ms=triton_ms,
        speedup=pytorch_ms / triton_ms,
        max_abs_error=max_abs_error,
        triton_v2_ms=triton_v2_ms,
        speedup_v2=pytorch_ms / triton_v2_ms,
        max_abs_error_v2=max_abs_error_v2,
    )


def run_benchmarks(args: argparse.Namespace) -> list[BenchmarkResult]:
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    results = []
    for T in args.sequence_lengths:
        result = benchmark_case(
            batch=args.batch,
            sequence_length=T,
            heads=args.heads,
            dim=args.dim,
            dtype=dtype,
            noncontiguous=args.noncontiguous,
            warmup=args.warmup,
            rep=args.rep,
        )
        results.append(result)
        print(
            f"T={T:>6}  PyTorch={result.pytorch_ms:>9.4f} ms  "
            f"Triton={result.triton_ms:>9.4f} ms  "
            f"speedup={result.speedup:>6.2f}x  "
            f"TritonV2={result.triton_v2_ms:>9.4f} ms  "
            f"max_err={result.max_abs_error:.6g}"
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness matrix")
    parser.add_argument("--benchmark", action="store_true", help="run latency sweep")
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096, 8192, 16384],
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--noncontiguous", action="store_true")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--json", type=Path, default=None, help="write machine-readable results")
    args = parser.parse_args()
    if not args.check and not args.benchmark:
        args.check = True
        args.benchmark = True
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")

    if args.check:
        check_correctness()

    results: list[BenchmarkResult] = []
    if args.benchmark:
        results = run_benchmarks(args)

    if args.json is not None:
        payload = {
            "gpu": torch.cuda.get_device_name(),
            "pytorch": torch.__version__,
            "triton": triton.__version__,
            "results": [asdict(r) for r in results],
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
