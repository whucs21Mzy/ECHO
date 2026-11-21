# Shared arithmetic for the recurrent model families.
"""Layout-specific Raven RoPE fusion with released rounding boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _huginn_rope(
    q,
    k,
    positions,
    trig,
    bias,
    output,
    N,
    QH: tl.constexpr,
    KH: tl.constexpr,
    D: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TILE: tl.constexpr,
):
    pairs = tl.program_id(0) * TILE + tl.arange(0, TILE)
    per_row = (QH + KH) * (D // 2)
    row = pairs // per_row
    head = pairs % per_row // (D // 2)
    dim = pairs % (D // 2)
    valid = row < N
    is_q = head < QH
    local_head = tl.where(is_q, head, head - QH)
    # Both source pointers are valid objects, but only the selected one is
    # dereferenced. Each flattened matrix retains its original packed stride.
    qa = tl.load(
        q + row * QS + local_head * D + 2 * dim, mask=valid & is_q, other=0
    ).to(tl.float32)
    qb = tl.load(
        q + row * QS + local_head * D + 2 * dim + 1, mask=valid & is_q, other=0
    ).to(tl.float32)
    ka = tl.load(
        k + row * KS + local_head * D + 2 * dim, mask=valid & ~is_q, other=0
    ).to(tl.float32)
    kb = tl.load(
        k + row * KS + local_head * D + 2 * dim + 1, mask=valid & ~is_q, other=0
    ).to(tl.float32)
    # Selection, rather than adding a masked zero, preserves signed zeros.
    a, b = tl.where(is_q, qa, ka), tl.where(is_q, qb, kb)
    if HAS_BIAS:
        ba = tl.load(bias + head * D + 2 * dim, mask=valid, other=0).to(tl.float32)
        bb = tl.load(bias + head * D + 2 * dim + 1, mask=valid, other=0).to(tl.float32)
        # The released Q/K bias addition rounds to activation dtype BEFORE
        # rotation; do not silently absorb that rounding into the final cast.
        a = (a + ba).to(q.dtype.element_ty).to(tl.float32)
        b = (b + bb).to(q.dtype.element_ty).to(tl.float32)
    pos = tl.load(positions + row, mask=valid, other=0)
    cos = tl.load(trig + pos * D + dim, mask=valid, other=0)
    sin = tl.load(trig + pos * D + D // 2 + dim, mask=valid, other=0)
    real, imag = a * cos - b * sin, b * cos + a * sin
    tl.store(output + 2 * pairs, real, mask=valid)
    tl.store(output + 2 * pairs + 1, imag, mask=valid)


def huginn_rope(positions, query, key, trig, head_dim, bias=None):
    """One kernel, FP32 separate products, shared Q/K storage, no padding."""
    if (
        query.ndim != 2
        or key.ndim != 2
        or not query.is_cuda
        or query.device != key.device
        or query.dtype != key.dtype
        or query.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or query.shape[0] != key.shape[0]
        or positions.numel() != query.shape[0]
        or positions.device != query.device
        or not positions.is_contiguous()
        or positions.dtype not in (torch.int32, torch.int64)
        or query.stride(1) != 1
        or key.stride(1) != 1
        or head_dim % 2
        or query.shape[1] % head_dim
        or key.shape[1] % head_dim
        or trig.device != query.device
        or trig.ndim != 2
        or trig.shape[1] != head_dim
        or trig.dtype != torch.float32
        or not trig.is_contiguous()
    ):
        raise ValueError(
            "fused Huginn RoPE requires packed CUDA Q/K and an FP32 trig table"
        )
    rows = query.shape[0]
    qh, kh = query.shape[1] // head_dim, key.shape[1] // head_dim
    if bias is not None and (
        qh != kh
        or bias.shape != (2, 1, qh, head_dim)
        or not bias.is_contiguous()
        or bias.dtype != query.dtype
        or bias.device != query.device
    ):
        raise ValueError("fused Huginn Q/K bias shape/dtype is incompatible")
    result = torch.empty(
        (rows, qh + kh, head_dim), dtype=query.dtype, device=query.device
    )
    if rows:
        _huginn_rope[(triton.cdiv(result.numel() // 2, 256),)](
            query,
            key,
            positions,
            trig,
            bias if bias is not None else query,
            result,
            rows,
            qh,
            kh,
            head_dim,
            query.stride(0),
            key.stride(0),
            bias is not None,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    q, k = result.split((qh, kh), dim=1)
    return q.reshape(query.shape), k.reshape(key.shape)


@triton.jit(do_not_specialize=["N"])
def _olmo_rope(
    q,
    k,
    positions,
    trig,
    out_q,
    out_k,
    N,
    QH: tl.constexpr,
    KH: tl.constexpr,
    D: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    TILE: tl.constexpr,
):
    pairs = tl.program_id(0) * TILE + tl.arange(0, TILE)
    per_row = (QH + KH) * (D // 2)
    row = pairs // per_row
    head = pairs % per_row // (D // 2)
    dim = pairs % (D // 2)
    valid = row < N
    is_q = head < QH
    local_head = tl.where(is_q, head, head - QH)
    qa = tl.load(q + row * QS + local_head * D + dim, mask=valid & is_q, other=0).to(
        tl.float32
    )
    qb = tl.load(
        q + row * QS + local_head * D + dim + D // 2, mask=valid & is_q, other=0
    ).to(tl.float32)
    ka = tl.load(k + row * KS + local_head * D + dim, mask=valid & ~is_q, other=0).to(
        tl.float32
    )
    kb = tl.load(
        k + row * KS + local_head * D + dim + D // 2, mask=valid & ~is_q, other=0
    ).to(tl.float32)
    a, b = tl.where(is_q, qa, ka), tl.where(is_q, qb, kb)
    pos = tl.load(positions + row, mask=valid, other=0)
    cos = tl.load(trig + pos * D + dim, mask=valid, other=0)
    sin = tl.load(trig + pos * D + dim + D // 2, mask=valid, other=0)
    # No FP32 FMA and no narrow cos/sin: match OlmoRotaryEmbedding exactly.
    first, second = a * cos - b * sin, b * cos + a * sin
    qo = row * QH * D + local_head * D + dim
    ko = row * KH * D + local_head * D + dim
    tl.store(out_q + qo, first, mask=valid & is_q)
    tl.store(out_q + qo + D // 2, second, mask=valid & is_q)
    tl.store(out_k + ko, first, mask=valid & ~is_q)
    tl.store(out_k + ko + D // 2, second, mask=valid & ~is_q)


def olmo_rope(positions, query, key, trig, head_dim):
    """One FP32 half-pair kernel; preserve OLMo's separate contiguous Q/K."""
    if (
        head_dim < 2
        or head_dim % 2
        or query.ndim != 2
        or key.ndim != 2
        or not query.is_cuda
        or query.device != key.device
        or query.dtype != key.dtype
        or query.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or query.shape[0] != key.shape[0]
        or positions.numel() != query.shape[0]
        or positions.device != query.device
        or not positions.is_contiguous()
        or positions.dtype not in (torch.int32, torch.int64)
        or query.stride(1) != 1
        or key.stride(1) != 1
        or query.shape[1] < head_dim
        or key.shape[1] < head_dim
        or query.shape[1] % head_dim
        or key.shape[1] % head_dim
        or trig.device != query.device
        or trig.ndim != 2
        or trig.shape[1] != head_dim
        or trig.dtype != torch.float32
        or not trig.is_contiguous()
    ):
        raise ValueError(
            "fused OLMo RoPE requires packed CUDA Q/K and an FP32 trig table"
        )
    rows = query.shape[0]
    qh, kh = query.shape[1] // head_dim, key.shape[1] // head_dim
    out_q = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    out_k = torch.empty(key.shape, dtype=key.dtype, device=key.device)
    if rows:
        _olmo_rope[(triton.cdiv(rows * (qh + kh) * (head_dim // 2), 256),)](
            query,
            key,
            positions,
            trig,
            out_q,
            out_k,
            rows,
            qh,
            kh,
            head_dim,
            query.stride(0),
            key.stride(0),
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out_q, out_k
