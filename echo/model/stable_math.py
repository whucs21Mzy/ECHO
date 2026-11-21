"""Row-invariant BF16 arithmetic for the standalone Torch inference backend.

Linear layers keep one K reduction and tile layout for every query count.
Attention visits only visible KV rows, in sequence order, in fixed-size blocks.
Tree padding and unrelated siblings therefore cannot change the reduction.
These kernels use Torch tensors and Triton directly; they do not import SGLang.
"""

import weakref
from types import MethodType

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_original_sdpa = F.scaled_dot_product_attention
_mask_cache = None


@triton.jit(do_not_specialize=["M"])
def _linear(
    X,
    W,
    B,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 64,
    BK: tl.constexpr = 64,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + inner
        a = tl.load(
            X + rows[:, None] * K + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = tl.load(
            W + cols[None, :] * K + k[:, None],
            (cols[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        accumulator = tl.dot(a, b, accumulator)
    if HAS_BIAS:
        bias = tl.load(B + cols, cols < N, other=0).to(tl.float32)
        accumulator += bias[None, :]
    tl.store(
        Y + rows[:, None] * N + cols[None, :],
        accumulator,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def linear(input, weight, bias=None):
    if (
        input.device.type != "cuda"
        or input.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
    ):
        return F.linear(input, weight, bias)
    x, w = input.contiguous(), weight.contiguous()
    m, n, k = x.numel() // x.shape[-1], w.shape[0], w.shape[1]
    output = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
    _linear[(triton.cdiv(m, 16), triton.cdiv(n, 64))](
        x,
        w,
        bias if bias is not None else x,
        output,
        m,
        n,
        k,
        bias is not None,
        num_warps=4,
    )
    return output


@triton.jit
def _rms(X, W, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + cols, cols < N, other=0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / N
    normalized = (
        (values * tl.rsqrt(variance + EPS)).to(X.dtype.element_ty).to(tl.float32)
    )
    weight = tl.load(W + cols, cols < N, other=0).to(tl.float32)
    tl.store(Y + row * N + cols, normalized * weight, cols < N)


def rmsnorm(input, weight, eps):
    """Released Ouro/Raven ordering: normalize, narrow, multiply weight."""
    x = input.contiguous()
    output = torch.empty_like(x)
    n = x.shape[-1]
    _rms[(x.numel() // n,)](
        x,
        weight,
        output,
        n,
        eps,
        triton.next_power_of_2(n),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


@triton.jit(do_not_specialize=["Q", "K"])
def _visible_indices(
    Mask,
    Indices,
    Lengths,
    Q,
    K,
    SMQ,
    SMK,
    HAS_MASK: tl.constexpr,
    BOOL_MASK: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    query = tl.program_id(0)
    keys = tl.arange(0, BLOCK)
    valid = keys < K
    if HAS_MASK:
        value = tl.load(Mask + query * SMQ + keys * SMK, valid, other=0)
        if BOOL_MASK:
            valid &= value
        else:
            valid &= value == 0
    elif CAUSAL:
        valid &= keys <= query
    offsets = tl.cumsum(valid.to(tl.int32)) - 1
    tl.store(Indices + query * K + offsets, keys, valid)
    tl.store(Lengths + query, tl.sum(valid.to(tl.int32)))


@triton.jit(do_not_specialize=["Q", "K"])
def _attention(
    Query,
    Key,
    Value,
    Indices,
    Lengths,
    Output,
    Q,
    K,
    SQH,
    SQQ,
    SQD,
    SKH,
    SKK,
    SKD,
    SVH,
    SVK,
    SVD,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BD: tl.constexpr,
    BK: tl.constexpr = 16,
):
    row, head = tl.program_id(0), tl.program_id(1)
    kv_head = head // GROUP
    dims = tl.arange(0, BD)
    query = tl.load(Query + head * SQH + row * SQQ + dims * SQD, dims < D, other=0).to(
        tl.float32
    )
    length = tl.load(Lengths + row)
    maximum = -float("inf")
    denominator = 0.0
    accumulator = tl.zeros((BD,), tl.float32)
    for block in range(tl.cdiv(length, BK)):
        offsets = block * BK + tl.arange(0, BK)
        indices = tl.load(Indices + row * K + offsets, offsets < length, other=0)
        keys = tl.load(
            Key + kv_head * SKH + indices[:, None] * SKK + dims[None, :] * SKD,
            (offsets[:, None] < length) & (dims[None, :] < D),
            other=0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        scores = tl.where(offsets < length, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
        rescale = tl.exp(maximum - next_maximum)
        probabilities = tl.exp(scores - next_maximum)
        values = tl.load(
            Value + kv_head * SVH + indices[:, None] * SVK + dims[None, :] * SVD,
            (offsets[:, None] < length) & (dims[None, :] < D),
            other=0,
        ).to(tl.float32)
        accumulator = accumulator * rescale + tl.sum(
            probabilities[:, None] * values, axis=0
        )
        denominator = denominator * rescale + tl.sum(probabilities, axis=0)
        maximum = next_maximum
    output = accumulator / tl.maximum(denominator, 1.0e-20)
    tl.store(Output + (head * Q + row) * D + dims, output, dims < D)


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    enable_gqa=False,
):
    global _mask_cache
    if (
        query.device.type != "cuda"
        or query.dtype != torch.bfloat16
        or query.ndim != 4
        or query.shape[0] != 1
        or dropout_p != 0
    ):
        return _original_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    heads, q, dim = query.shape[1:]
    kv_heads, k = key.shape[1:3]
    if key.dtype != query.dtype or value.dtype != query.dtype or heads % kv_heads:
        raise ValueError("Unsupported deterministic Torch attention dtype/head layout")
    # The recurrent backends construct broadcast-head masks containing only
    # zero/-inf (or booleans), with no additive position bias.
    if attn_mask is not None:
        if (
            attn_mask.shape[-2] not in (1, q)
            or attn_mask.shape[-1] != k
            or attn_mask.numel() not in (k, q * k)
        ):
            raise ValueError("Expected a shared causal/tree mask")
        mask = attn_mask.reshape(attn_mask.shape[-2], k)
        smq = mask.stride(0) if mask.shape[0] != 1 else 0
        smk = mask.stride(1)
        version = None if attn_mask.is_inference() else attn_mask._version
        signature = (q, k, smq, smk, mask.dtype, version)
    else:
        mask, smq, smk = None, 0, 0
        signature = (q, k, is_causal, query.device)
    cached = _mask_cache
    if (
        cached is not None
        and cached[0] == signature
        and (
            (attn_mask is None and cached[1] is None)
            or (
                attn_mask is not None
                and cached[1] is not None
                and cached[1]() is attn_mask
            )
        )
    ):
        indices, lengths = cached[2:]
    else:
        indices = torch.empty((q, k), device=query.device, dtype=torch.int32)
        lengths = torch.empty((q,), device=query.device, dtype=torch.int32)
        _visible_indices[(q,)](
            mask if mask is not None else indices,
            indices,
            lengths,
            q,
            k,
            smq,
            smk,
            mask is not None,
            mask is not None and mask.dtype == torch.bool,
            is_causal,
            triton.next_power_of_2(k),
            num_warps=4,
        )
        _mask_cache = (
            signature,
            weakref.ref(attn_mask) if attn_mask is not None else None,
            indices,
            lengths,
        )
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    _attention[(q, heads)](
        query,
        key,
        value,
        indices,
        lengths,
        output,
        q,
        k,
        *query.stride()[1:],
        *key.stride()[1:],
        *value.stride()[1:],
        heads // kv_heads,
        dim,
        scale if scale is not None else dim**-0.5,
        triton.next_power_of_2(dim),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def _ouro_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    is_causal=None,
    **kwargs,
):
    del kwargs
    if is_causal is None:
        is_causal = (
            query.shape[2] > 1
            and attention_mask is None
            and getattr(module, "is_causal", True)
        )
    output = scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        is_causal=is_causal,
        scale=scaling,
        enable_gqa=query.shape[1] != key.shape[1],
    )
    return output.transpose(1, 2).contiguous(), None


def install(model):
    """Use identical arithmetic for own-engine AR and ECHO in this process."""
    count, norms = 0, 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and not getattr(
            module, "_echo_stable_linear", False
        ):

            def forward(self, input):
                return linear(input, self.weight, self.bias)

            module.forward = MethodType(forward, module)
            module._echo_stable_linear = True
            count += 1
        elif (
            type(module).__name__ in {"RMSNorm", "OuroRMSNorm"}
            and (hasattr(module, "_norm") or hasattr(module, "variance_epsilon"))
            and not getattr(module, "_echo_stable_norm", False)
        ):
            module._echo_original_norm_forward = module.forward

            def norm_forward(self, input):
                if input.is_cuda and input.dtype == self.weight.dtype == torch.bfloat16:
                    eps = (
                        self.variance_epsilon
                        if hasattr(self, "variance_epsilon")
                        else self.eps
                    )
                    return rmsnorm(input, self.weight, float(eps))
                return self._echo_original_norm_forward(input)

            module.forward = MethodType(norm_forward, module)
            module._echo_stable_norm = True
            norms += 1
    if model.config.model_type == "ouro":
        from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
        from transformers.modeling_utils import AttentionInterface

        AttentionInterface.register("echo_row_invariant", _ouro_attention)
        AttentionMaskInterface.register("echo_row_invariant", sdpa_mask)
        model.config._attn_implementation = "echo_row_invariant"
    else:
        model._echo_sdpa = scaled_dot_product_attention
    return dict(
        linear="fixed 16x64x64 tiles and FP32 accumulation",
        normalization="one fixed FP32 row reduction; narrow before weight multiply",
        attention="visible KV sequence, fixed blocks of 16 and FP32 accumulation",
        wrapped_linear_modules=count,
        wrapped_norm_modules=norms,
    )
