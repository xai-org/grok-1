# colab_t4_rope_eval.py -- Grok-1 PR #434 fused Triton RoPE: T4 evidence generator.
#
# Colab: Runtime -> Change runtime type -> T4 GPU. Colab already ships torch+triton.
# Optional pin (record whatever you use; the manifest captures versions anyway):
#   !pip -q install --upgrade torch triton
# Run (paste whole file in one cell, or save and run):
#   !python colab_t4_rope_eval.py --mode all --out /content/rope_t4
# Modes: gate (correctness only) | bench (gate + timings) | sweep (gate + config sweep) | all
# CPU smoke test without a GPU (tiny shapes, no timings):
#   TRITON_INTERPRET=1 python colab_t4_rope_eval.py --mode gate --smoke
#
# Fail-closed: if any correctness case FAILS, nothing is timed, results.json records
# the failures, and the process exits 1. BF16 that Triton cannot compile on this GPU is
# recorded as SKIPPED with the reason, never as PASS.
#
# baseline_triton = PR #434 head 2e9d5df0fc982efd91982cb50c303059f8be0498,
#   examples/triton_rope_fused.py kernel + launcher copied verbatim. Only changes:
#   identifiers renamed (_fused_rope_kernel->_baseline_rope_kernel, fused_rope->baseline_triton)
#   and the CUDA-only guard relaxed when TRITON_INTERPRET=1.
# improved_triton = v1 proposal (head-blocked, half-split loads), measured on T4 run 1.
# improved_v2 = half-width (T, D/2) cos/sin table + 2D (tokens x heads) tile.
# improved_v2_sincos = optional (--with-sincos): in-kernel fp32 cos/sin, no table.
# improved_v3 = one program per token over all heads, half table loaded once, 128-bit-friendly
#   contiguous D/2 loads (hints + alignment check, strided fallback), eviction hints if exposed,
#   optional persistent grid-stride; fused_rope_qk_v3 rotates q AND k (Hq != Hkv, packed QKV views)
#   in ONE launch. Also: CUDA-graph replay timing (small shapes), torch.compile and Liger baselines.
# Every bench row also times a torch copy_ of x (same layout) as the attainable-bandwidth roofline.
# Full run:  !python colab_t4_rope_eval.py --mode all --out /content/rope_t4_v3
# Quick run: !python colab_t4_rope_eval.py --mode all --quick --out /content/rope_t4_v3q

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.language as tl

PR_SHA = "2e9d5df0fc982efd91982cb50c303059f8be0498"
SCRIPT_VERSION = "3"
TOL = {torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2)}


def _interpreting() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


# ----------------------------------------------------------------- baseline (PR head)
@triton.jit
def _baseline_rope_kernel(
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
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One Triton program owns one complete (B, T, H) feature row.

    Flattening B*T*H into a single launch dimension avoids relying on a fourth
    program-id axis and guarantees that an in-place store cannot race another
    program reading the paired half of the same row.
    """
    row = tl.program_id(0)
    h = row % H
    tmp = row // H
    t = tmp % T
    b = tmp // T

    d = tl.arange(0, BLOCK_D)
    mask = d < D
    half = D // 2
    paired_d = tl.where(d < half, d + half, d - half)

    x_base = b * stride_x_b + t * stride_x_t + h * stride_x_h
    o_base = b * stride_o_b + t * stride_o_t + h * stride_o_h

    x = tl.load(x_ptr + x_base + d * stride_x_d, mask=mask, other=0.0)
    paired = tl.load(
        x_ptr + x_base + paired_d * stride_x_d,
        mask=mask,
        other=0.0,
    )
    rotated = paired * tl.where(d < half, -1.0, 1.0)

    cos_v = tl.load(
        cos_ptr + t * stride_cos_t + d * stride_cos_d,
        mask=mask,
        other=0.0,
    )
    sin_v = tl.load(
        sin_ptr + t * stride_sin_t + d * stride_sin_d,
        mask=mask,
        other=0.0,
    )

    out = x * cos_v + rotated * sin_v
    tl.store(out_ptr + o_base + d * stride_o_d, out, mask=mask)


def baseline_triton(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """Apply Grok-style RoPE to ``x`` shaped ``(B, T, H, D)``.

    Supports contiguous and non-contiguous input layouts through explicit
    strides. ``cos`` and ``sin`` must be CUDA tensors shaped ``(T, D)``.
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
    block_d = triton.next_power_of_2(D)
    if block_d > 65536:
        raise ValueError(f"unsupported feature dimension D={D}")

    # Grok-1 uses D=128. Four warps is a strong default at this width while
    # retaining reasonable occupancy for neighboring test dimensions.
    num_warps = 4 if block_d <= 256 else 8
    grid = (B * T * H,)

    _baseline_rope_kernel[grid](
        x,
        cos,
        sin,
        out,
        *x.stride(),
        *out.stride(),
        *cos.stride(),
        *sin.stride(),
        T=T,
        H=H,
        D=D,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out


# ----------------------------------------------------------------- improved
@triton.jit
def _improved_rope_kernel(
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


def improved_triton(
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
    _improved_rope_kernel[grid](
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


# ----------------------------------------------------------------- improved_v2 (half-width table, 2D tile)
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


# ----------------------------------------------------------------- improved_v3 (per-token, fused QK, persistent)
def _load_supports(name: str) -> bool:
    """Feature-detect tl.load/tl.store keyword support in this Triton build."""
    import inspect as _inspect

    try:
        return name in _inspect.signature(tl.load).parameters and name in _inspect.signature(tl.store).parameters
    except (TypeError, ValueError):
        return False


V3_HAS_EVICTION = _load_supports("eviction_policy")
V3_HAS_CACHE_MODIFIER = _load_supports("cache_modifier")


@triton.jit
def _v3_rotate_heads(
    x_ptr,
    o_ptr,
    sxb,
    sxt,
    sxh,
    sxd,
    sob,
    sot,
    soh,
    sod,
    b,
    t,
    c,
    s,
    H: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    D_CONTIG: tl.constexpr,
    ALIGNED: tl.constexpr,
    EVICT: tl.constexpr,
):
    """Rotate all H heads of one token, BLOCK_H heads per static-unrolled chunk."""
    d = tl.arange(0, BLOCK_HALF)
    d = tl.max_contiguous(tl.multiple_of(d, BLOCK_HALF), BLOCK_HALF)
    d_m = d < HALF
    if D_CONTIG:
        dx1 = d[None, :]
        do1 = d[None, :]
        dx2 = d[None, :] + HALF
        do2 = d[None, :] + HALF
    else:
        dx1 = d[None, :] * sxd
        do1 = d[None, :] * sod
        dx2 = (d[None, :] + HALF) * sxd
        do2 = (d[None, :] + HALF) * sod
    xb = b * sxb + t * sxt
    ob = b * sob + t * sot
    for hc in tl.static_range(0, H, BLOCK_H):
        h = hc + tl.arange(0, BLOCK_H)
        m = (h < H)[:, None] & d_m[None, :]
        xr = xb + h[:, None] * sxh
        orow = ob + h[:, None] * soh
        if ALIGNED:
            xr = tl.multiple_of(xr, (8, 1))
            orow = tl.multiple_of(orow, (8, 1))
        if EVICT:
            x1 = tl.load(x_ptr + xr + dx1, mask=m, other=0.0, eviction_policy="evict_first").to(tl.float32)
            x2 = tl.load(x_ptr + xr + dx2, mask=m, other=0.0, eviction_policy="evict_first").to(tl.float32)
        else:
            x1 = tl.load(x_ptr + xr + dx1, mask=m, other=0.0).to(tl.float32)
            x2 = tl.load(x_ptr + xr + dx2, mask=m, other=0.0).to(tl.float32)
        out1 = x1 * c[None, :] - x2 * s[None, :]
        out2 = x2 * c[None, :] + x1 * s[None, :]
        if EVICT:
            tl.store(o_ptr + orow + do1, out1.to(o_ptr.dtype.element_ty), mask=m, eviction_policy="evict_first")
            tl.store(o_ptr + orow + do2, out2.to(o_ptr.dtype.element_ty), mask=m, eviction_policy="evict_first")
        else:
            tl.store(o_ptr + orow + do1, out1.to(o_ptr.dtype.element_ty), mask=m)
            tl.store(o_ptr + orow + do2, out2.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _rope_v3_kernel(
    q_ptr, qo_ptr, k_ptr, ko_ptr, cos_ptr, sin_ptr,
    sqb, sqt, sqh, sqd, sqob, sqot, sqoh, sqod,
    skb, skt, skh, skd, skob, skot, skoh, skod,
    sct, sst,
    T,
    n_tokens,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    D_CONTIG: tl.constexpr,
    ALIGNED: tl.constexpr,
    EVICT: tl.constexpr,
    PERSISTENT: tl.constexpr,
):
    """One program per token (b, t): loads the half-width cos/sin row ONCE, then
    rotates every q head and (if HK > 0) every k head of that token.

    Program id -> token with t fastest, so neighboring programs touch adjacent
    cos/sin rows and adjacent x rows. PERSISTENT=True: grid-stride loop over
    tokens with a fixed grid (k * SM count). Each program writes only rows it
    read, so in-place and disjoint packed-QKV views are race-free.
    """
    d = tl.arange(0, BLOCK_HALF)
    d_m = d < HALF
    pid = tl.program_id(0)
    if PERSISTENT:
        nprog = tl.num_programs(0)
        for tok in range(pid, n_tokens, nprog):
            b = tok // T
            t = tok % T
            if EVICT:
                c = tl.load(cos_ptr + t * sct + d, mask=d_m, other=0.0, eviction_policy="evict_last").to(tl.float32)
                s = tl.load(sin_ptr + t * sst + d, mask=d_m, other=0.0, eviction_policy="evict_last").to(tl.float32)
            else:
                c = tl.load(cos_ptr + t * sct + d, mask=d_m, other=0.0).to(tl.float32)
                s = tl.load(sin_ptr + t * sst + d, mask=d_m, other=0.0).to(tl.float32)
            _v3_rotate_heads(q_ptr, qo_ptr, sqb, sqt, sqh, sqd, sqob, sqot, sqoh, sqod, b, t, c, s,
                             HQ, HALF, BLOCK_H, BLOCK_HALF, D_CONTIG, ALIGNED, EVICT)
            if HK > 0:
                _v3_rotate_heads(k_ptr, ko_ptr, skb, skt, skh, skd, skob, skot, skoh, skod, b, t, c, s,
                                 HK, HALF, BLOCK_H, BLOCK_HALF, D_CONTIG, ALIGNED, EVICT)
    else:
        b = pid // T
        t = pid % T
        if EVICT:
            c = tl.load(cos_ptr + t * sct + d, mask=d_m, other=0.0, eviction_policy="evict_last").to(tl.float32)
            s = tl.load(sin_ptr + t * sst + d, mask=d_m, other=0.0, eviction_policy="evict_last").to(tl.float32)
        else:
            c = tl.load(cos_ptr + t * sct + d, mask=d_m, other=0.0).to(tl.float32)
            s = tl.load(sin_ptr + t * sst + d, mask=d_m, other=0.0).to(tl.float32)
        _v3_rotate_heads(q_ptr, qo_ptr, sqb, sqt, sqh, sqd, sqob, sqot, sqoh, sqod, b, t, c, s,
                         HQ, HALF, BLOCK_H, BLOCK_HALF, D_CONTIG, ALIGNED, EVICT)
        if HK > 0:
            _v3_rotate_heads(k_ptr, ko_ptr, skb, skt, skh, skd, skob, skot, skoh, skod, b, t, c, s,
                             HK, HALF, BLOCK_H, BLOCK_HALF, D_CONTIG, ALIGNED, EVICT)


def _v3_sm_count(device: torch.device) -> int:
    if _interpreting() or device.type != "cuda":
        return 2  # interpreter stand-in; only affects grid size, never math
    return torch.cuda.get_device_properties(device).multi_processor_count


def _v3_aligned(*tensors: torch.Tensor) -> bool:
    """True when every tensor is 16-byte aligned with D-contiguous rows whose
    b/t/h strides are multiples of 8 elements (128-bit vector friendly)."""
    for x in tensors:
        if x.data_ptr() % 16 or x.stride(-1) != 1:
            return False
        if any(st % 8 for st in x.stride()[:-1]):
            return False
    return True


def default_launch_config_v3(total_heads: int, dim: int) -> tuple[int, int]:
    """Return ``(block_h, num_warps)`` for the per-token kernel."""
    block_half = triton.next_power_of_2(dim // 2)
    block_h = min(triton.next_power_of_2(max(total_heads, 1)), max(1, 1024 // block_half))
    elems = block_h * block_half * 2
    return block_h, (4 if elems <= 2048 else 8)


def fused_rope_qk_v3(
    q: torch.Tensor,
    k: torch.Tensor | None,
    cos_half: torch.Tensor,
    sin_half: torch.Tensor,
    *,
    inplace: bool = False,
    block_h: int | None = None,
    num_warps: int | None = None,
    num_stages: int = 1,
    persistent_k: int = 0,
    evict: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Grok RoPE on q (B, T, Hq, D) and optionally k (B, T, Hkv, D) in ONE launch.

    Hq != Hkv is supported. q/k may be strided views (e.g. slices of a packed
    QKV buffer). ``cos_half``/``sin_half`` are contiguous (T, D/2) tables.
    ``persistent_k > 0`` launches ``persistent_k * SM_count`` programs with a
    grid-stride loop. ``evict=None`` enables eviction hints when this Triton
    exposes ``eviction_policy``.
    """
    if q.ndim != 4 or (k is not None and k.ndim != 4):
        raise ValueError("q and k must have shape (B, T, H, D)")
    B, T, HQ, D = q.shape
    if D % 2:
        raise ValueError("RoPE feature dimension must be even")
    half = D // 2
    if k is not None and (k.shape[0] != B or k.shape[1] != T or k.shape[3] != D or k.dtype != q.dtype):
        raise ValueError("k must match q in B, T, D, and dtype")
    if cos_half.shape != (T, half) or sin_half.shape != (T, half):
        raise ValueError(f"cos_half/sin_half must have shape {(T, half)}")
    if cos_half.dtype != q.dtype or sin_half.dtype != q.dtype:
        raise ValueError("tables must match q dtype")
    if cos_half.stride(-1) != 1 or sin_half.stride(-1) != 1:
        raise ValueError("cos_half/sin_half must have unit stride on the last dim")
    tensors = [q, cos_half, sin_half] + ([k] if k is not None else [])
    if not _interpreting() and not all(t_.is_cuda for t_ in tensors):
        raise ValueError("all tensors must be CUDA tensors")

    qo = q if inplace else torch.empty_like(q, memory_format=torch.preserve_format)
    HK = 0
    if k is not None:
        HK = k.shape[2]
        ko = k if inplace else torch.empty_like(k, memory_format=torch.preserve_format)
    else:
        k, ko = q, qo  # dummies, never dereferenced when HK == 0
    io = [q, qo] + ([k, ko] if HK else [])
    d_contig = all(t_.stride(-1) == 1 for t_ in io)
    aligned = _v3_aligned(*io)
    d_bh, d_nw = default_launch_config_v3(max(HQ, HK), D)
    block_h = d_bh if block_h is None else block_h
    num_warps = d_nw if num_warps is None else num_warps
    if block_h < 1 or block_h & (block_h - 1):
        raise ValueError("block_h must be a positive power of two")
    use_evict = V3_HAS_EVICTION if evict is None else (bool(evict) and V3_HAS_EVICTION)
    n_tokens = B * T
    if persistent_k > 0:
        grid = (max(1, min(n_tokens, persistent_k * _v3_sm_count(q.device))),)
    else:
        grid = (n_tokens,)
    _rope_v3_kernel[grid](
        q, qo, k, ko, cos_half, sin_half,
        *q.stride(), *qo.stride(), *k.stride(), *ko.stride(),
        cos_half.stride(0), sin_half.stride(0),
        T, n_tokens,
        HQ=HQ, HK=HK, HALF=half, BLOCK_H=block_h, BLOCK_HALF=triton.next_power_of_2(half),
        D_CONTIG=d_contig, ALIGNED=aligned, EVICT=use_evict, PERSISTENT=persistent_k > 0,
        num_warps=num_warps, num_stages=num_stages,
    )
    return qo, (ko if HK else None)


def fused_rope_v3(x, cos_half, sin_half, **kw):
    """Single-tensor per-token v3 (one program per token, all heads)."""
    return fused_rope_qk_v3(x, None, cos_half, sin_half, **kw)[0]


# ----------------------------------------------------------------- eager reference
def make_grok_rope_cache(T, D, *, device, dtype, base_exponent=10000.0, offset=0):
    """Grok-1 JAX layout: phase = tile(phase_half, (1, 2)) -> cat, NOT repeat_interleave."""
    exponents = torch.arange(0, D, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base_exponent ** (exponents / D))
    pos = torch.arange(offset, offset + T, device=device, dtype=torch.float32)
    ph = torch.outer(pos, inv_freq)
    ph = torch.cat((ph, ph), dim=-1)
    return torch.cos(ph).to(dtype), torch.sin(ph).to(dtype)


def eager_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos[None, :, None, :] + rot * sin[None, :, None, :]


def eager_rope_fp32(x, cos, sin):
    """Ground-truth in fp32 (used only to report error; gate uses eager_rope)."""
    return eager_rope(x.float(), cos.float(), sin.float())


def _v2_from_full(x, c, s, **kw):
    """Gate adapter: derive the half table from the full Grok table (verified tiled)."""
    half = x.shape[-1] // 2
    if not (torch.equal(c[:, :half], c[:, half:]) and torch.equal(s[:, :half], s[:, half:])):
        raise AssertionError("cos/sin table is not Grok tile layout; half-width path invalid")
    return fused_rope_v2(x, c[:, :half], s[:, :half], **kw)


def _v2_sincos_from_full(x, c, s, offset=0, **kw):
    inv = grok_inv_freq(x.shape[-1], device=x.device)
    return fused_rope_v2(x, None, None, sincos=True, inv_freq=inv, pos_offset=offset, **kw)


def _v3_from_full(x, c, s, **kw):
    half = x.shape[-1] // 2
    if not (torch.equal(c[:, :half], c[:, half:]) and torch.equal(s[:, :half], s[:, half:])):
        raise AssertionError("cos/sin table is not Grok tile layout; half-width path invalid")
    return fused_rope_v3(x, c[:, :half].contiguous(), s[:, :half].contiguous(), **kw)


IMPLS = {"baseline_triton": baseline_triton, "improved_triton": improved_triton,
         "improved_v2": _v2_from_full, "improved_v3": _v3_from_full,
         "improved_v3_persistent": lambda x, c, s, **kw: _v3_from_full(x, c, s, persistent_k=1, **kw)}


# ----------------------------------------------------------------- helpers
def device():
    return torch.device("cpu") if _interpreting() else torch.device("cuda")


def make_x(B, T, H, D, dtype, layout, gen):
    dev = device()
    if layout == "contiguous":
        return torch.randn(B, T, H, D, generator=gen, dtype=torch.float32).to(dev, dtype)
    if layout == "transposed":  # (B,H,T,D) storage viewed as (B,T,H,D)
        return torch.randn(B, H, T, D, generator=gen, dtype=torch.float32).to(dev, dtype).transpose(1, 2)
    if layout == "strided":  # every other token of a 2T buffer, and a padded D
        src = torch.randn(B, 2 * T, H, D + 64, generator=gen, dtype=torch.float32).to(dev, dtype)
        return src[:, ::2, :, :D]
    raise ValueError(layout)


def bf16_status():
    """Probe whether Triton can compile/run a bf16 kernel here. Returns (ok, reason)."""
    if _interpreting():
        return False, "TRITON_INTERPRET=1 smoke mode: bf16 not exercised"
    try:
        x = torch.randn(1, 2, 2, 128, device="cuda", dtype=torch.bfloat16)
        c, s = make_grok_rope_cache(2, 128, device=x.device, dtype=torch.bfloat16)
        baseline_triton(x, c, s)
        improved_triton(x, c, s)
        fused_rope_v2(x, c[:, :64], s[:, :64])
        fused_rope_v3(x, c[:, :64].contiguous(), s[:, :64].contiguous())
        torch.cuda.synchronize()
        return True, "compiled and ran"
    except Exception as e:  # recorded verbatim, never treated as PASS
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"


def gate_cases(smoke):
    if smoke:
        return [(1, 4, 2, 128, torch.float16, l, 0) for l in ("contiguous", "transposed", "strided")] + [
            (1, 3, 3, 128, torch.float16, "contiguous", 5), (2, 5, 3, 128, torch.float16, "strided", 8190)]
    cases = []
    for dtype in (torch.float16, torch.bfloat16):
        for H in (8, 48):
            for T in (128, 512, 2048, 8192):
                for layout in ("contiguous", "transposed", "strided"):
                    B = 2 if T <= 512 else 1
                    cases.append((B, T, H, 128, dtype, layout, 0 if T != 512 else 37))
        # odd edge shapes: T=1, non-power-of-2 T and H
        cases += [(1, 1, 1, 128, dtype, "contiguous", 0), (1, 257, 5, 128, dtype, "transposed", 19),
                  (3, 129, 48, 128, dtype, "strided", 0), (1, 4097, 48, 128, dtype, "contiguous", 0),
                  (1, 130, 8, 128, dtype, "transposed", 131072)]  # odd T tail; large absolute positions
    return cases


def run_gate(smoke, bf16_ok, bf16_reason, with_sincos=False):
    gen = torch.Generator().manual_seed(1234)
    records, failed = [], False
    for (B, T, H, D, dtype, layout, off) in gate_cases(smoke):
        dname = str(dtype).removeprefix("torch.")
        impls = dict(IMPLS)
        if with_sincos:
            impls["improved_v2_sincos"] = (lambda x, c, s, _o=off, **kw: _v2_sincos_from_full(x, c, s, offset=_o, **kw))
        for impl_name, fn in impls.items():
            for mode in ("out_of_place", "in_place"):
                rec = dict(impl=impl_name, B=B, T=T, H=H, D=D, dtype=dname, layout=layout,
                           offset=off, mode=mode)
                if dtype == torch.bfloat16 and not bf16_ok:
                    rec.update(status="SKIPPED", reason=bf16_reason)
                    records.append(rec)
                    continue
                try:
                    x = make_x(B, T, H, D, dtype, layout, gen)
                    c, s = make_grok_rope_cache(T, D, device=x.device, dtype=dtype, offset=off)
                    exp = eager_rope(x, c, s)
                    ref32 = eager_rope_fp32(x, c, s)
                    if mode == "in_place":
                        xi = x.clone(memory_format=torch.preserve_format)
                        if layout == "strided":  # keep the strided view (clone would compact it)
                            buf = torch.zeros(B, 2 * T, H, D + 64, device=x.device, dtype=dtype)
                            xi = buf[:, ::2, :, :D]
                            xi.copy_(x)
                        ptr, strides = xi.data_ptr(), xi.stride()
                        got = fn(xi, c, s, inplace=True)
                        if got.data_ptr() != ptr or got.stride() != strides:
                            raise AssertionError("in-place path did not reuse input storage")
                        if layout == "strided":  # padding/skipped tokens must be untouched
                            if buf[:, 1::2].abs().sum().item() != 0 or buf[..., D:].abs().sum().item() != 0:
                                raise AssertionError("in-place wrote outside the view")
                    else:
                        got = fn(x, c, s)
                        if got.stride() != x.stride() and layout != "strided":
                            pass  # empty_like(preserve_format) may densify; values still checked
                    atol, rtol = TOL[dtype]
                    torch.testing.assert_close(got, exp, atol=atol, rtol=rtol)
                    rec.update(status="PASS",
                               max_abs_err_vs_eager=float((got.float() - exp.float()).abs().max()),
                               max_abs_err_vs_fp32=float((got.float() - ref32).abs().max()))
                except Exception as e:
                    failed = True
                    rec.update(status="FAIL", error=f"{type(e).__name__}: {e}"[:2000])
                records.append(rec)
                print(f"[{rec['status']:7}] {impl_name:15} {dname:8} B={B} T={T:5} H={H:2} {layout:10} {mode:12}"
                      + (f" err={rec.get('max_abs_err_vs_eager', 0):.3g}" if rec["status"] == "PASS" else
                         f" {rec.get('reason', rec.get('error', ''))[:120]}"), flush=True)
    return records, failed


def bench(fn, warmup_ms, rep_ms):
    q = triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms, quantiles=[0.5, 0.1, 0.9])
    return dict(median_ms=float(q[0]), p10_ms=float(q[1]), p90_ms=float(q[2]))


def bytes_moved(B, T, H, D, elt):
    # MINIMUM bytes: read x + write out + read the half-width cos and sin once (T*D/2 each).
    return (2 * B * T * H * D + T * D) * elt


def bytes_copy(B, T, H, D, elt):
    return 2 * B * T * H * D * elt


# ----------------------------------------------------------------- external baselines
EXTERNAL = {}


def torch_compile_status():
    """Compile the eager reference once; record failure honestly and skip it."""
    if "torch_compile" in EXTERNAL:
        return EXTERNAL["torch_compile"]
    if _interpreting():
        EXTERNAL["torch_compile"] = dict(available=False, reason="TRITON_INTERPRET=1 smoke mode")
        return EXTERNAL["torch_compile"]
    try:
        # Many static shapes are timed; raise dynamo's recompile cap so later shapes are not
        # silently run in eager fallback under the torch_compile label.
        import torch._dynamo as _dynamo
        for _attr in ("recompile_limit", "cache_size_limit"):
            if hasattr(_dynamo.config, _attr):
                setattr(_dynamo.config, _attr, max(getattr(_dynamo.config, _attr), 128))
        fn = torch.compile(eager_rope, dynamic=False)
        x = torch.randn(1, 16, 8, 128, device="cuda", dtype=torch.float16)
        c, s = make_grok_rope_cache(16, 128, device=x.device, dtype=x.dtype)
        torch.testing.assert_close(fn(x, c, s), eager_rope(x, c, s), atol=2e-3, rtol=2e-3)
        EXTERNAL["torch_compile"] = dict(available=True, fn=fn, reason="compiled and matched eager on probe")
    except Exception as e:
        EXTERNAL["torch_compile"] = dict(available=False, reason=f"{type(e).__name__}: {str(e)[:300]}")
    return EXTERNAL["torch_compile"]


def liger_status():
    """Liger Kernel rope: used only if already importable. Never pip-installed by this script.
    API assumed (UNVERIFIED): LigerRopeFunction.apply(q, k, cos, sin) with q/k (B, H, T, D),
    cos/sin (1, T, D), HF rotate_half convention (same as Grok tile layout)."""
    if "liger" in EXTERNAL:
        return EXTERNAL["liger"]
    if _interpreting():
        EXTERNAL["liger"] = dict(available=False, reason="TRITON_INTERPRET=1 smoke mode")
        return EXTERNAL["liger"]
    try:
        from liger_kernel.ops.rope import LigerRopeFunction
        try:
            from importlib.metadata import version as _v
            ver = _v("liger-kernel")
        except Exception:
            ver = "unknown"
        EXTERNAL["liger"] = dict(available=True, fn=LigerRopeFunction.apply, version=ver,
                                 reason="imported liger_kernel.ops.rope.LigerRopeFunction")
    except Exception as e:
        EXTERNAL["liger"] = dict(available=False, reason=f"not available: {type(e).__name__}: {str(e)[:200]}")
    return EXTERNAL["liger"]


def _public_status(d):
    return {k: v for k, v in d.items() if k != "fn"}


# ----------------------------------------------------------------- fused QK gate
def make_qk(B, T, HQ, HKV, D, dtype, layout, gen):
    """Returns (q, k, v_or_None, base_or_None). 'packed' = slices of one [B,T,HQ+2HKV,D] buffer."""
    dev = device()
    r = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float32).to(dev, dtype)
    if layout == "separate_contiguous":
        return r(B, T, HQ, D), r(B, T, HKV, D), None, None
    if layout == "separate_transposed":
        return r(B, HQ, T, D).transpose(1, 2), r(B, HKV, T, D).transpose(1, 2), None, None
    if layout == "packed_qkv":
        base = r(B, T, HQ + 2 * HKV, D)
        return base[:, :, :HQ], base[:, :, HQ:HQ + HKV], base[:, :, HQ + HKV:], base
    raise ValueError(layout)


def qk_gate_cases(smoke):
    if smoke:
        return [(1, 5, 6, 2, 128, torch.float16, l, o) for l in ("separate_contiguous", "separate_transposed", "packed_qkv")
                for o in (0,)] + [(2, 3, 3, 3, 128, torch.float16, "packed_qkv", 8190)]
    cases = []
    for dtype in (torch.float16, torch.bfloat16):
        for (HQ, HKV) in ((48, 8), (8, 8)):
            for T in (128, 2048, 4097):
                for layout in ("separate_contiguous", "separate_transposed", "packed_qkv"):
                    cases.append((1, T, HQ, HKV, 128, dtype, layout, 0))
        cases += [(2, 129, 48, 8, 128, dtype, "packed_qkv", 131072), (1, 1, 5, 3, 128, dtype, "separate_transposed", 7)]
    return cases


QK_IMPLS = {
    "improved_v3_fused_qk": dict(persistent_k=0),
    "improved_v3_fused_qk_persistent": dict(persistent_k=1),
}


def run_gate_qk(smoke, bf16_ok, bf16_reason):
    gen = torch.Generator().manual_seed(4321)
    records, failed = [], False
    for (B, T, HQ, HKV, D, dtype, layout, off) in qk_gate_cases(smoke):
        dname = str(dtype).removeprefix("torch.")
        for impl, kw in QK_IMPLS.items():
            for mode in ("out_of_place", "in_place"):
                rec = dict(impl=impl, B=B, T=T, HQ=HQ, HKV=HKV, D=D, dtype=dname, layout=layout, offset=off, mode=mode)
                if dtype == torch.bfloat16 and not bf16_ok:
                    rec.update(status="SKIPPED", reason=bf16_reason)
                    records.append(rec)
                    continue
                try:
                    q, k, v, base = make_qk(B, T, HQ, HKV, D, dtype, layout, gen)
                    c, s = make_grok_rope_cache(T, D, device=q.device, dtype=dtype, offset=off)
                    ch, sh = make_grok_rope_cache_half(T, D, device=q.device, dtype=dtype, offset=off)
                    if not (torch.equal(ch, c[:, :D // 2]) and torch.equal(ch, c[:, D // 2:])):
                        raise AssertionError("half table != Grok tile layout")
                    eq, ek = eager_rope(q, c, s), eager_rope(k, c, s)
                    v_before = v.clone() if v is not None else None
                    if mode == "in_place":
                        qp, kp = q.data_ptr(), k.data_ptr()
                        oq, ok_ = fused_rope_qk_v3(q, k, ch, sh, inplace=True, **kw)
                        if oq.data_ptr() != qp or ok_.data_ptr() != kp:
                            raise AssertionError("in-place did not reuse q/k storage")
                    else:
                        oq, ok_ = fused_rope_qk_v3(q, k, ch, sh, **kw)
                    atol, rtol = TOL[dtype]
                    torch.testing.assert_close(oq, eq, atol=atol, rtol=rtol)
                    torch.testing.assert_close(ok_, ek, atol=atol, rtol=rtol)
                    if v is not None and not torch.equal(v, v_before):
                        raise AssertionError("V slice of packed QKV was modified")
                    rec.update(status="PASS", max_abs_err_q=float((oq.float() - eq.float()).abs().max()),
                               max_abs_err_k=float((ok_.float() - ek.float()).abs().max()))
                except Exception as e:
                    failed = True
                    rec.update(status="FAIL", error=f"{type(e).__name__}: {e}"[:2000])
                records.append(rec)
                print(f"[{rec['status']:7}] {impl:32} {dname:8} B={B} T={T:5} Hq={HQ:2} Hkv={HKV} {layout:20} {mode:12}"
                      + ("" if rec["status"] == "PASS" else f" {rec.get('reason', rec.get('error', ''))[:120]}"), flush=True)
    return records, failed


# ----------------------------------------------------------------- CUDA graphs
def capture_graph(fn):
    """Capture fn() (out-of-place) in a CUDA graph. Returns (graph, static_output)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def graph_impls(x, c, s, ch, sh):
    return {
        "baseline_triton": lambda: baseline_triton(x, c, s),
        "improved_triton": lambda: improved_triton(x, c, s),
        "improved_v2": lambda: fused_rope_v2(x, ch, sh),
        "improved_v3": lambda: fused_rope_v3(x, ch, sh),
    }


def run_gate_graphs(smoke):
    """Graph-replayed output must be bit-identical to a direct call (out-of-place)."""
    records, failed = [], False
    if _interpreting() or not torch.cuda.is_available():
        return [dict(impl="*", status="SKIPPED", reason="CUDA graphs need a CUDA device")], False
    gen = torch.Generator().manual_seed(5)
    for (T, H) in ((128, 8), (512, 48)):
        x = make_x(1, T, H, 128, torch.float16, "contiguous", gen)
        c, s = make_grok_rope_cache(T, 128, device=x.device, dtype=x.dtype)
        ch, sh = c[:, :64].contiguous(), s[:, :64].contiguous()
        for name, fn in graph_impls(x, c, s, ch, sh).items():
            rec = dict(impl=name, kind="cuda_graph_replay_equality", T=T, H=H, dtype="float16")
            try:
                g, out = capture_graph(fn)
                g.replay()
                torch.cuda.synchronize()
                if not torch.equal(out, fn()):
                    raise AssertionError("graph replay output != direct call output")
                rec["status"] = "PASS"
            except Exception as e:
                failed = True
                rec.update(status="FAIL", error=f"{type(e).__name__}: {e}"[:1000])
            records.append(rec)
            print(f"[{rec['status']:7}] graph-replay {name:16} T={T} H={H}", flush=True)
    return records, failed


# ----------------------------------------------------------------- benchmarks
def _row_finish(row, nb, copy_gbps):
    for name, r in list(row["impls"].items()):
        if "median_ms" in r:
            r["effective_GBps"] = nb / (r["median_ms"] * 1e-3) / 1e9
            r["pct_of_copy_roofline"] = 100.0 * r["effective_GBps"] / copy_gbps


def _print_row(prefix, row):
    parts = [f"{n}={r['pct_of_copy_roofline']:.0f}%({r['median_ms']:.4f}ms)"
             for n, r in row["impls"].items() if "pct_of_copy_roofline" in r]
    print(f"{prefix} copy={row['copy_roofline']['effective_GBps']:.0f}GB/s | " + " ".join(parts), flush=True)


def run_bench(args, bf16_ok):
    out = []
    gen = torch.Generator().manual_seed(99)
    tc = torch_compile_status()
    dtypes = [torch.float16] + ([torch.bfloat16] if bf16_ok else [])
    sm = _v3_sm_count(torch.device("cuda"))
    for dtype in dtypes:
        for H in (8, 48):
            for T in args.seq_lens:
                for layout in ("contiguous", "transposed"):
                    x = make_x(1, T, H, 128, dtype, layout, gen)
                    c, s = make_grok_rope_cache(T, 128, device=x.device, dtype=dtype)
                    ch, sh = c[:, :64].contiguous(), s[:, :64].contiguous()
                    exp = eager_rope(x, c, s)
                    y = torch.empty_like(x, memory_format=torch.preserve_format)
                    fns = {
                        "eager": lambda: eager_rope(x, c, s),
                        "baseline_triton": lambda: baseline_triton(x, c, s),
                        "improved_triton": lambda: improved_triton(x, c, s),
                        "improved_v2": lambda: fused_rope_v2(x, ch, sh),
                        "improved_v3": lambda: fused_rope_v3(x, ch, sh),
                        "improved_v3_persistent": lambda: fused_rope_v3(x, ch, sh, persistent_k=args.persistent_k),
                    }
                    if args.with_sincos:
                        inv = grok_inv_freq(128, device=x.device)
                        fns["improved_v2_sincos"] = lambda: fused_rope_v2(x, None, None, sincos=True, inv_freq=inv)
                    row = dict(kind="single", B=1, T=T, H=H, D=128, dtype=str(dtype).removeprefix("torch."),
                               layout=layout, impls={})
                    if tc["available"] and T >= 2048:
                        fns["torch_compile_eager"] = lambda: tc["fn"](x, c, s)
                    elif tc["available"]:
                        row["impls"]["torch_compile_eager"] = dict(skipped="not timed for T<2048 (recompile budget)")
                    else:
                        row["impls"]["torch_compile_eager"] = dict(skipped=tc["reason"])
                    atol, rtol = TOL[dtype]
                    for name, fn in fns.items():
                        try:  # pre-timing correctness, fail closed
                            torch.testing.assert_close(fn(), exp, atol=atol, rtol=rtol)
                        except Exception as e:
                            raise SystemExit(f"bench pre-check FAILED for {name} {row}: {e}")
                        row["impls"][name] = bench(fn, args.warmup_ms, args.rep_ms)
                    row["copy_roofline"] = bench(lambda: y.copy_(x), args.warmup_ms, args.rep_ms)
                    copy_gbps = bytes_copy(1, T, H, 128, x.element_size()) / (row["copy_roofline"]["median_ms"] * 1e-3) / 1e9
                    row["copy_roofline"]["effective_GBps"] = copy_gbps
                    row["bytes_min"] = bytes_moved(1, T, H, 128, x.element_size())
                    row["persistent_programs"] = min(T, args.persistent_k * sm)
                    _row_finish(row, row["bytes_min"], copy_gbps)
                    out.append(row)
                    _print_row(f"{row['dtype']:8} H={H:2} T={T:5} {layout:10}", row)
    return out


def run_bench_qk(args, bf16_ok):
    """Fused Q+K (Grok-1: Hq=48, Hkv=8). Baselines: PR kernel x2, eager x2, torch.compile x2, Liger."""
    out = []
    gen = torch.Generator().manual_seed(11)
    tc, lg = torch_compile_status(), liger_status()
    HQ, HKV = 48, 8
    dtypes = [torch.float16] + ([torch.bfloat16] if bf16_ok else [])
    for dtype in dtypes:
        for T in args.seq_lens:
            for layout in ("separate_contiguous", "packed_qkv"):
                q, k, v, base = make_qk(1, T, HQ, HKV, 128, dtype, layout, gen)
                c, s = make_grok_rope_cache(T, 128, device=q.device, dtype=dtype)
                ch, sh = c[:, :64].contiguous(), s[:, :64].contiguous()
                eq, ek = eager_rope(q, c, s), eager_rope(k, c, s)
                row = dict(kind="fused_qk", B=1, T=T, HQ=HQ, HKV=HKV, D=128, dtype=str(dtype).removeprefix("torch."),
                           layout=layout, impls={})
                fns = {
                    "eager_x2": lambda: (eager_rope(q, c, s), eager_rope(k, c, s)),
                    "baseline_triton_x2": lambda: (baseline_triton(q, c, s), baseline_triton(k, c, s)),
                    "improved_v2_x2": lambda: (fused_rope_v2(q, ch, sh), fused_rope_v2(k, ch, sh)),
                    "improved_v3_fused_qk": lambda: fused_rope_qk_v3(q, k, ch, sh),
                    "improved_v3_fused_qk_persistent": lambda: fused_rope_qk_v3(q, k, ch, sh, persistent_k=args.persistent_k),
                }
                if tc["available"] and T >= 2048:
                    fns["torch_compile_eager_x2"] = lambda: (tc["fn"](q, c, s), tc["fn"](k, c, s))
                else:
                    row["impls"]["torch_compile_eager_x2"] = dict(
                        skipped=tc["reason"] if not tc["available"] else "not timed for T<2048 (recompile budget)")
                if lg["available"]:
                    qt, kt = q.transpose(1, 2), k.transpose(1, 2)
                    c3, s3 = c[None], s[None]

                    def _liger(qt=qt, kt=kt, c3=c3, s3=s3):
                        a, b_ = lg["fn"](qt.clone(), kt.clone(), c3, s3)
                        return a.transpose(1, 2), b_.transpose(1, 2)
                    try:
                        lq, lk = _liger()
                        torch.testing.assert_close(lq, eq, atol=TOL[dtype][0], rtol=TOL[dtype][1])
                        torch.testing.assert_close(lk, ek, atol=TOL[dtype][0], rtol=TOL[dtype][1])
                        fns["liger_rope_incl_clone"] = _liger
                    except Exception as e:
                        row["impls"]["liger_rope_incl_clone"] = dict(skipped=f"excluded, failed check: {type(e).__name__}: {str(e)[:200]}")
                else:
                    row["impls"]["liger_rope_incl_clone"] = dict(skipped=lg["reason"])
                atol, rtol = TOL[dtype]
                for name, fn in fns.items():
                    try:
                        a, b_ = fn()
                        torch.testing.assert_close(a, eq, atol=atol, rtol=rtol)
                        torch.testing.assert_close(b_, ek, atol=atol, rtol=rtol)
                    except Exception as e:
                        raise SystemExit(f"bench pre-check FAILED for {name} {row}: {e}")
                    row["impls"][name] = bench(fn, args.warmup_ms, args.rep_ms)
                src = torch.empty(1, T, HQ + HKV, 128, device=q.device, dtype=dtype)
                dst = torch.empty_like(src)
                row["copy_roofline"] = bench(lambda: dst.copy_(src), args.warmup_ms, args.rep_ms)
                copy_gbps = bytes_copy(1, T, HQ + HKV, 128, q.element_size()) / (row["copy_roofline"]["median_ms"] * 1e-3) / 1e9
                row["copy_roofline"]["effective_GBps"] = copy_gbps
                row["bytes_min"] = bytes_moved(1, T, HQ + HKV, 128, q.element_size())
                _row_finish(row, row["bytes_min"], copy_gbps)
                out.append(row)
                _print_row(f"QK {row['dtype']:8} Hq={HQ} Hkv={HKV} T={T:5} {layout:19}", row)
    return out


def run_bench_graphs(args):
    """CUDA-graph replay timing, small shapes, fp16 contiguous, out-of-place. Reported separately."""
    out = []
    gen = torch.Generator().manual_seed(6)
    for T in (128, 512):
        for H in (8, 48):
            x = make_x(1, T, H, 128, torch.float16, "contiguous", gen)
            c, s = make_grok_rope_cache(T, 128, device=x.device, dtype=x.dtype)
            ch, sh = c[:, :64].contiguous(), s[:, :64].contiguous()
            row = dict(kind="cuda_graph_replay", T=T, H=H, dtype="float16", layout="contiguous", impls={})
            for name, fn in graph_impls(x, c, s, ch, sh).items():
                g, outp = capture_graph(fn)
                row["impls"][name] = dict(direct=bench(fn, args.warmup_ms, args.rep_ms),
                                          graph_replay=bench(g.replay, args.warmup_ms, args.rep_ms))
            out.append(row)
            print(f"GRAPH T={T} H={H} " + " ".join(
                f"{n}: direct {r['direct']['median_ms']:.4f} / replay {r['graph_replay']['median_ms']:.4f} ms"
                for n, r in row["impls"].items()), flush=True)
    return out


# ----------------------------------------------------------------- sweep (bounded)
def run_sweep(args):
    """fp16 only. Each config passes the correctness check before timing."""
    res = []
    gen = torch.Generator().manual_seed(7)
    shapes = [(48, 8192, "contiguous")] if args.quick else [(48, 8192, "contiguous"), (8, 8192, "contiguous"),
                                                            (48, 2048, "transposed")]

    def _try(rec, fn, exp_list):
        try:
            got = fn()
            got = got if isinstance(got, tuple) else (got,)
            for g_, e_ in zip(got, exp_list):
                torch.testing.assert_close(g_, e_, atol=2e-3, rtol=2e-3)
            rec.update(status="PASS", **bench(fn, args.warmup_ms, args.rep_ms))
        except Exception as e:
            rec.update(status="FAIL", error=f"{type(e).__name__}: {e}"[:300])
        res.append(rec)

    for (H, T, layout) in shapes:
        x = make_x(1, T, H, 128, torch.float16, layout, gen)
        c, s = make_grok_rope_cache(T, 128, device=x.device, dtype=x.dtype)
        ch, sh = c[:, :64].contiguous(), s[:, :64].contiguous()
        exp = [eager_rope(x, c, s)]
        sh_ = dict(H=H, T=T, layout=layout)
        for bh in (8, 16, 64):
            for nw in (4, 8):
                if bh <= triton.next_power_of_2(H):
                    _try(dict(impl="improved_triton", block_h=bh, num_warps=nw, num_stages=1, **sh_),
                         lambda: improved_triton(x, c, s, block_h=bh, num_warps=nw, num_stages=1), exp)
        for bt in ((1, 2, 4) if args.quick else (1, 2, 4, 8)):
            for bh in (8, 16):
                for nw in (4, 8):
                    _try(dict(impl="improved_v2", block_t=bt, block_h=bh, num_warps=nw, num_stages=1, **sh_),
                         lambda: fused_rope_v2(x, ch, sh, block_t=bt, block_h=bh, num_warps=nw, num_stages=1), exp)
        for bh in (8, 16, 32):
            for nw in (2, 4, 8):
                for pk in ((0, 4) if args.quick else (0, 2, 4, 8)):
                    for ev in ((None,) if args.quick else (True, False)):
                        _try(dict(impl="improved_v3", block_h=bh, num_warps=nw, persistent_k=pk,
                                  evict=V3_HAS_EVICTION if ev is None else ev and V3_HAS_EVICTION, **sh_),
                             lambda: fused_rope_v3(x, ch, sh, block_h=bh, num_warps=nw, persistent_k=pk, evict=ev), exp)
    # fused QK sweep on Grok-1 shape
    for T in ((8192,) if args.quick else (2048, 8192)):
        q, k, _, _ = make_qk(1, T, 48, 8, 128, torch.float16, "separate_contiguous", gen)
        c, s = make_grok_rope_cache(T, 128, device=q.device, dtype=q.dtype)
        ch, sh = c[:, :64].contiguous(), s[:, :64].contiguous()
        exp = [eager_rope(q, c, s), eager_rope(k, c, s)]
        for bh in (8, 16, 32):
            for nw in (4, 8):
                for pk in ((0, 4) if args.quick else (0, 2, 4, 8)):
                    _try(dict(impl="improved_v3_fused_qk", H="48+8", T=T, layout="separate_contiguous",
                              block_h=bh, num_warps=nw, persistent_k=pk),
                         lambda: fused_rope_qk_v3(q, k, ch, sh, block_h=bh, num_warps=nw, persistent_k=pk), exp)
    best = {}
    for r in res:
        if r["status"] != "PASS":
            continue
        key = (r["impl"], r["H"], r["T"], r["layout"])
        if key not in best or r["median_ms"] < best[key]["median_ms"]:
            best[key] = r
    for key, r in sorted(best.items(), key=str):
        knobs = {k_: r[k_] for k_ in ("block_t", "block_h", "num_warps", "num_stages", "persistent_k", "evict") if k_ in r}
        print(f"sweep best {key}: {r['median_ms']:.4f} ms {knobs}", flush=True)
    return res


def print_summary(results):
    print("\n==== SUMMARY: % of measured copy roofline (median ms) ====")
    for row in results.get("bench", []) + results.get("bench_qk", []):
        tag = (f"{row['dtype'][:4]} H={row['H']}" if row["kind"] == "single" else f"{row['dtype'][:4]} QK48+8")
        cells = [f"{n}:{r['pct_of_copy_roofline']:.0f}%" for n, r in row["impls"].items() if "pct_of_copy_roofline" in r]
        print(f"{tag:12} T={row['T']:5} {row['layout'][:12]:12} " + " ".join(cells))


def sha256_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def manifest(args, bf16_ok, bf16_reason):
    m = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        pr="xai-org/grok-1#434", pr_head_sha=PR_SHA, script_version=SCRIPT_VERSION,
        with_sincos=args.with_sincos, quick=args.quick, persistent_k=args.persistent_k,
        python=platform.python_version(), host=platform.platform(),
        torch=torch.__version__, triton=triton.__version__, cuda_runtime=torch.version.cuda,
        triton_interpret=_interpreting(), argv=sys.argv, mode=args.mode, smoke=args.smoke,
        timing=dict(method="triton.testing.do_bench (CUDA events, L2 flushed per rep)",
                    warmup_ms=args.warmup_ms, rep_ms=args.rep_ms, quantiles=[0.5, 0.1, 0.9],
                    seq_lens=args.seq_lens),
        tolerances=dict(fp16=[2e-3, 2e-3], bf16=[2e-2, 2e-2]),
        bf16_supported=bf16_ok, bf16_reason=bf16_reason,
        triton_features=dict(eviction_policy=V3_HAS_EVICTION, cache_modifier=V3_HAS_CACHE_MODIFIER,
                             cache_modifier_used=False,
                             note="eviction hints used on x/out (evict_first) and cos/sin (evict_last) only if exposed; "
                                  "cache_modifier detected but not used"),
        default_launch_config_D128=dict(h8=default_launch_config(8, 128), h48=default_launch_config(48, 128)),
        default_launch_config_v2_D128=dict(h8=default_launch_config_v2(8, 128), h48=default_launch_config_v2(48, 128)),
        default_launch_config_v3_D128=dict(h8=default_launch_config_v3(8, 128), h48=default_launch_config_v3(48, 128)),
        bytes_formula="min_bytes = (2*B*T*H*D + T*D) * elem_size  [x read + out write + half cos + half sin]; "
                      "copy = 2*B*T*H*D*elem; fused QK uses H = Hq + Hkv",
        not_tested=["vLLM rotary_embedding", "flash-attn rotary"],
    )
    try:
        m["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        m["script_sha256"] = None  # pasted into a cell: no __file__
    for name, obj in [("baseline_kernel", _baseline_rope_kernel.fn), ("baseline_launcher", baseline_triton),
                      ("improved_kernel", _improved_rope_kernel.fn), ("improved_launcher", improved_triton),
                      ("improved_v2_kernel", _rope_v2_kernel.fn), ("improved_v2_launcher", fused_rope_v2),
                      ("improved_v3_kernel", _rope_v3_kernel.fn), ("improved_v3_rows", _v3_rotate_heads.fn),
                      ("improved_v3_launcher", fused_rope_qk_v3),
                      ("rope_cache_half", make_grok_rope_cache_half),
                      ("eager_reference", eager_rope), ("rope_cache", make_grok_rope_cache)]:
        try:
            m[f"sha256_{name}"] = sha256_text(inspect.getsource(obj))
        except Exception as e:
            m[f"sha256_{name}"] = f"unavailable: {e}"
    if torch.cuda.is_available() and not _interpreting():
        p = torch.cuda.get_device_properties(0)
        m.update(gpu=torch.cuda.get_device_name(0), compute_capability=f"{p.major}.{p.minor}",
                 gpu_total_memory_bytes=p.total_memory, sm_count=p.multi_processor_count)
        try:
            m["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
        except Exception as e:
            m["driver"] = f"unavailable: {e}"
    else:
        m["gpu"] = None
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("gate", "bench", "sweep", "all"), default="all")
    ap.add_argument("--out", type=Path, default=Path("rope_t4_out"))
    ap.add_argument("--smoke", action="store_true", help="tiny shapes (for TRITON_INTERPRET=1)")
    ap.add_argument("--quick", action="store_true", help="reduced bench/sweep (~few min on T4)")
    ap.add_argument("--warmup-ms", type=int, default=25)
    ap.add_argument("--rep-ms", type=int, default=100)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=None)
    ap.add_argument("--persistent-k", type=int, default=4, help="programs per SM for persistent variants in bench")
    ap.add_argument("--with-sincos", action="store_true", help="also gate+bench optional in-kernel sincos v2")
    args, _ = ap.parse_known_args()  # tolerate Colab/Jupyter's injected -f arg
    if args.seq_lens is None:
        args.seq_lens = [2048, 8192] if args.quick else [128, 512, 2048, 4096, 8192]
    if args.quick:
        args.rep_ms = min(args.rep_ms, 50)

    if not _interpreting() and not torch.cuda.is_available():
        print("FATAL: no CUDA GPU. In Colab: Runtime > Change runtime type > T4 GPU.")
        sys.exit(2)
    args.out.mkdir(parents=True, exist_ok=True)
    bf16_ok, bf16_reason = bf16_status()
    man = manifest(args, bf16_ok, bf16_reason)
    man["external_baselines"] = dict(torch_compile=_public_status(torch_compile_status()),
                                     liger=_public_status(liger_status()))
    (args.out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps({k: man.get(k) for k in ("gpu", "compute_capability", "torch", "triton", "cuda_runtime",
                                               "driver", "bf16_supported", "bf16_reason", "triton_features",
                                               "external_baselines")}, indent=2))

    results = dict(manifest_file="manifest.json")
    gate, f1 = run_gate(args.smoke, bf16_ok, bf16_reason, args.with_sincos)
    gate_qk, f2 = run_gate_qk(args.smoke, bf16_ok, bf16_reason)
    gate_g, f3 = run_gate_graphs(args.smoke)
    allc = gate + gate_qk + gate_g
    failed = f1 or f2 or f3
    results["gate"] = dict(passed=not failed,
                           counts={s: sum(r["status"] == s for r in allc) for s in ("PASS", "FAIL", "SKIPPED")},
                           cases=gate, cases_fused_qk=gate_qk, cases_cuda_graph=gate_g)
    if failed:
        results["verdict"] = "GATE_FAILED_NO_TIMINGS"
        (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print("GATE FAILED -> no timings. See results.json", flush=True)
        sys.exit(1)
    timed = not args.smoke and not _interpreting()
    if args.mode in ("bench", "all") and timed:
        results["bench"] = run_bench(args, bf16_ok)
        results["bench_qk"] = run_bench_qk(args, bf16_ok)
        results["bench_cuda_graph"] = run_bench_graphs(args)
    if args.mode in ("sweep", "all") and timed:
        results["sweep"] = run_sweep(args)
    results["verdict"] = "GATE_PASSED"
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print_summary(results)
    print(f"GATE PASSED {results['gate']['counts']}. Wrote {args.out}/results.json and manifest.json")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
