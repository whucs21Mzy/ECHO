# Shared arithmetic for the recurrent model families.
"""Ouro NeoX RoPE fusion preserving the native narrow-product boundaries.

This replaces the environment's PyTorch RoPE fallback, not its frequency
table or attention algorithm. Q/K products round BEFORE the add/subtract,
as in SGLang's native apply_rotary_emb. No FP32 FMA is permitted.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ROWS"])
def _neox_rope(
    Q,
    K,
    POS,
    CACHE,
    OQ,
    OK,
    V,
    LOC,
    CK,
    CV,
    ROWS,
    QH: tl.constexpr,
    KH: tl.constexpr,
    D: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    STORE: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    per_row = (QH + KH) * (D // 2)
    row, local = pair // per_row, pair % per_row
    head, dim = local // (D // 2), local % (D // 2)
    valid, is_q = row < ROWS, head < QH
    h = tl.where(is_q, head, head - QH)
    qa = tl.load(Q + row * QS + h * D + dim, valid & is_q, 0).to(tl.float32)
    qb = tl.load(Q + row * QS + h * D + D // 2 + dim, valid & is_q, 0).to(tl.float32)
    ka = tl.load(K + row * KS + h * D + dim, valid & ~is_q, 0).to(tl.float32)
    kb = tl.load(K + row * KS + h * D + D // 2 + dim, valid & ~is_q, 0).to(tl.float32)
    a, b = tl.where(is_q, qa, ka), tl.where(is_q, qb, kb)
    position = tl.load(POS + row, valid, 0)
    cos = (
        tl.load(CACHE + position * D + dim, valid, 0)
        .to(Q.dtype.element_ty)
        .to(tl.float32)
    )
    sin = (
        tl.load(CACHE + position * D + D // 2 + dim, valid, 0)
        .to(Q.dtype.element_ty)
        .to(tl.float32)
    )
    ac = (a * cos).to(Q.dtype.element_ty).to(tl.float32)
    bs = (b * sin).to(Q.dtype.element_ty).to(tl.float32)
    bc = (b * cos).to(Q.dtype.element_ty).to(tl.float32)
    ass = (a * sin).to(Q.dtype.element_ty).to(tl.float32)
    first, second = ac - bs, bc + ass
    tl.store(OQ + row * QH * D + h * D + dim, first, valid & is_q)
    tl.store(OQ + row * QH * D + h * D + D // 2 + dim, second, valid & is_q)
    tl.store(OK + row * KH * D + h * D + dim, first, valid & ~is_q)
    tl.store(OK + row * KH * D + h * D + D // 2 + dim, second, valid & ~is_q)
    if STORE:
        dest = tl.load(LOC + row, valid, 0)
        # The normal branch allocator owns and validates these locations.
        # Retain a device-side assertion for sanitizer/TRITON_DEBUG runs.
        tl.device_assert(
            (~valid) | ((dest >= 0) & (dest < CAPACITY)), "Ouro KV slot out of bounds"
        )
        writable = valid & ~is_q & (dest >= 0) & (dest < CAPACITY)
        tl.store(CK + dest * KH * D + h * D + dim, first, writable)
        tl.store(CK + dest * KH * D + h * D + D // 2 + dim, second, writable)
        va = tl.load(V + row * VS + h * D + dim, valid & ~is_q, 0)
        vb = tl.load(V + row * VS + h * D + D // 2 + dim, valid & ~is_q, 0)
        tl.store(CV + dest * KH * D + h * D + dim, va, writable)
        tl.store(CV + dest * KH * D + h * D + D // 2 + dim, vb, writable)


def supported(positions, query, key, cache, head_dim):
    return (
        query.ndim == key.ndim == 2
        and query.is_cuda
        and query.device == key.device == positions.device == cache.device
        and query.dtype == key.dtype
        and query.dtype in (torch.bfloat16, torch.float16)
        and query.shape[0] == key.shape[0] == positions.numel()
        and positions.is_contiguous()
        and positions.dtype in (torch.int32, torch.int64)
        and head_dim > 0
        and head_dim % 2 == 0
        and query.shape[1] % head_dim == key.shape[1] % head_dim == 0
        and query.stride(1) == key.stride(1) == 1
        and cache.ndim == 2
        and cache.shape[1] == head_dim
        and cache.is_contiguous()
        and cache.dtype in (torch.float32, torch.bfloat16, torch.float16)
    )


def ouro_neox_rope(positions, query, key, cache, head_dim, *, kv=None):
    if not supported(positions, query, key, cache, head_dim):
        raise ValueError(
            "Ouro fused RoPE requires CUDA packed Q/K, positions and a NeoX cache"
        )
    # Separate contiguous destinations retain the native path's output layout;
    # never mutate packed QKV inputs or share storage with the V slice.
    oq, ok = (
        torch.empty_like(query, memory_format=torch.contiguous_format),
        torch.empty_like(key, memory_format=torch.contiguous_format),
    )
    rows = query.shape[0]
    qh, kh = query.shape[1] // head_dim, key.shape[1] // head_dim
    if kv is None:
        v, loc, ck, cv, capacity = query, positions, query, query, 0
    else:
        v, loc, ck, cv = kv
        if (
            v.shape != key.shape
            or v.dtype != key.dtype
            or v.device != key.device
            or v.stride(1) != 1
            or loc.ndim != 1
            or loc.numel() != rows
            or not loc.is_contiguous()
            or loc.device != key.device
            or loc.dtype not in (torch.int32, torch.int64)
            or ck.ndim != 3
            or cv.shape != ck.shape
            or tuple(ck.shape[1:]) != (kh, head_dim)
            or ck.dtype != cv.dtype
            or ck.dtype != key.dtype
            or ck.device != cv.device
            or ck.device != key.device
            or not ck.is_contiguous()
            or not cv.is_contiguous()
        ):
            raise ValueError(
                "Ouro fused KV store requires exact unscaled NHD caches and packed V"
            )
        capacity = ck.shape[0]
    pairs = rows * (qh + kh) * (head_dim // 2)
    if pairs:
        _neox_rope[(triton.cdiv(pairs, 256),)](
            query,
            key,
            positions,
            cache,
            oq,
            ok,
            v,
            loc,
            ck,
            cv,
            rows,
            qh,
            kh,
            head_dim,
            query.stride(0),
            key.stride(0),
            v.stride(0),
            kv is not None,
            capacity,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return oq, ok


def native_compatible_dispatch(module, original):
    def forward(positions, query, key, offsets=None, fused_set_kv_buffer_arg=None):
        cache = module.cos_sin_cache
        if (
            offsets is None
            and fused_set_kv_buffer_arg is None
            and module.is_neox_style
            and module.rotary_dim == module.head_size
            and not hasattr(module, "sin_cos_cache")
            and supported(positions, query, key, cache, module.head_size)
        ):
            return ouro_neox_rope(positions, query, key, cache, module.head_size)
        return original(positions, query, key, offsets, fused_set_kv_buffer_arg)

    return forward


def install_ouro_rope(module):
    flag = os.environ.get("ECHO_OURO_FUSED_ROPE", "1")
    if flag not in ("0", "1"):
        raise ValueError("Ouro fused RoPE flag must be 0/1")
    if flag == "0":
        return module
    if getattr(module, "_echo_ouro_rope_installed", False):
        return module
    if os.environ.get("ECHO_FORCE_NATIVE_ROPE", "1") != "1":
        raise ValueError(
            "Ouro native-compatible RoPE requires ECHO_FORCE_NATIVE_ROPE=1"
        )
    module._forward_method = native_compatible_dispatch(module, module._forward_method)
    module._echo_ouro_rope_installed = True
    return module
