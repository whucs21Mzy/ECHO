# Shared arithmetic for the recurrent model families.
"""Partially fused Huginn RMSNorm, retaining the deterministic mean verbatim.

Only replaces the deterministic ``cast_x_before_out_mul`` no-residual path.
The 1024-column batch-invariant mean remains unchanged. Cast/square and
epsilon/rsqrt/scale/narrow-cast/weight multiplication are fused separately. Narrow
input fusion can change Triton's reduction association and is not used here.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ROWS"])
def _raven_square_kernel(
    X, SQUARES, N: tl.constexpr, ROWS, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + index // N * STRIDE + index % N, index < N * ROWS, 0).to(
        tl.float32
    )
    tl.store(SQUARES + index, value * value, index < N * ROWS)


@triton.jit
def _raven_rms_kernel(X, W, V, Y, N, STRIDE, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    variance = tl.load(V + row)
    inverse = tl.rsqrt(variance + EPS)
    for start in range(0, N, BLOCK):
        cols = start + offsets
        values = tl.load(X + row * STRIDE + cols, cols < N, 0).to(tl.float32)
        normalized = (values * inverse).to(X.dtype.element_ty).to(tl.float32)
        weight = tl.load(W + cols, cols < N, 0).to(tl.float32)
        tl.store(Y + row * N + cols, weight * normalized, cols < N)


def raven_rmsnorm(x, weight, eps):
    """No-residual RMSNorm; caller must use the deterministic numerical target."""
    if (
        x.ndim != 2
        or weight.ndim != 1
        or x.shape[1] != weight.numel()
        or x.shape[1] == 0
        or not x.is_cuda
        or x.device != weight.device
        or x.dtype != weight.dtype
        or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or x.stride(1) != 1
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "Raven fused RMS requires matching CUDA floating rows/weight with unit column stride"
        )
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if x.shape[0]:
        # Preserve the upstream FP32 mean kernel verbatim: Triton may choose
        # a different reduction layout when its source is a fused narrow load.
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import mean_dim

        squares = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        _raven_square_kernel[(triton.cdiv(x.numel(), 1024),)](
            x,
            squares,
            x.shape[1],
            x.shape[0],
            x.stride(0),
            BLOCK=1024,
            enable_fp_fusion=False,
        )
        variance = mean_dim(squares, -1, True)
        _raven_rms_kernel[(x.shape[0],)](
            x,
            weight,
            variance,
            out,
            x.shape[1],
            x.stride(0),
            float(eps),
            BLOCK=1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


def install_raven_rmsnorm(norm):
    """Opt in one existing norm, preserving its parameter/weight-loader identity.

    No global/upstream patch. Unsupported arithmetic (residual, mixed dtype,
    variance override, non-deterministic execution) retains the original path.
    """
    if os.environ.get("ECHO_RAVEN_FUSED_RMS", "1") != "1":
        return norm
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    original = norm._forward_method

    def forward(x, residual=None, post_residual_addition=None):
        if (
            is_batch_invariant_mode_enabled()
            and x.is_cuda
            and x.ndim == 2
            and residual is None
            and post_residual_addition is None
            and norm.cast_x_before_out_mul
            and norm.variance_size_override is None
            and norm.override_orig_dtype is None
            and x.shape[-1] == norm.hidden_size
            and x.dtype == norm.weight.dtype
            and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and x.stride(-1) == 1
            and norm.weight.is_contiguous()
        ):
            return raven_rmsnorm(x, norm.weight, norm.variance_epsilon)
        return original(x, residual, post_residual_addition)

    norm._forward_method = forward
    return norm
