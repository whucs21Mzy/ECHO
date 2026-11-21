"""Canonical ancestor-ordered KV rows for native SGLang tree attention."""

from types import SimpleNamespace

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["PREFIX", "WIDTH"])
def _tree_kv_indices(
    locations,
    tree_mask,
    kv_indptr,
    kv_indices,
    PREFIX,
    WIDTH,
    TREE_TILE: tl.constexpr,
    PREFIX_TILE: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.load(kv_indptr + row)
    offsets = tl.arange(0, PREFIX_TILE)
    for block in range(tl.cdiv(PREFIX, PREFIX_TILE)):
        position = block * PREFIX_TILE + offsets
        values = tl.load(locations + position, mask=position < PREFIX, other=0)
        tl.store(kv_indices + start + position, values, mask=position < PREFIX)
    nodes = tl.arange(0, TREE_TILE)
    visible = tl.load(tree_mask + row * WIDTH + nodes, mask=nodes < WIDTH, other=0).to(
        tl.int32
    )
    rank = tl.cumsum(visible) - 1
    physical = tl.load(locations + PREFIX + nodes, mask=nodes < WIDTH, other=0)
    tl.store(
        kv_indices + start + PREFIX + rank,
        physical,
        mask=(nodes < WIDTH) & (visible != 0),
    )


def make_forward_batch(backend, cache, prefix, ids, positions, tree_mask):
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    width = ids.numel()
    if prefix + width > cache.locations.numel():
        raise ValueError("Flattened tree exceeds reserved cache capacity")
    if int(positions.max()) >= backend.context_length:
        raise ValueError("Token positions exceed the configured context")
    common = dict(
        input_ids=ids,
        positions=positions,
        out_cache_loc=cache.locations[prefix : prefix + width],
        spec_algorithm=SpeculativeAlgorithm.NONE,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    # Native FP32 extend attention with IEEE dot products exceeds A800's
    # shared-memory limit. Evaluate these diagnostic prefill queries using
    # the same native decode kernel as tree/linear queries, with causal KV
    # rows. Normal BF16 inference retains native extend attention.
    if tree_mask is None and prefix == 0 and backend.dtype != torch.float32:
        batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            req_pool_indices=torch.tensor(
                [cache.request_index], dtype=torch.int32, device=backend.device
            ),
            seq_lens=torch.tensor([width], dtype=torch.int32, device=backend.device),
            seq_lens_sum=width,
            seq_lens_cpu=torch.tensor([width], dtype=torch.int32),
            extend_num_tokens=width,
            extend_seq_lens=torch.tensor(
                [width], dtype=torch.int32, device=backend.device
            ),
            extend_prefix_lens=torch.zeros(1, dtype=torch.int32, device=backend.device),
            extend_start_loc=torch.zeros(1, dtype=torch.int32, device=backend.device),
            extend_seq_lens_cpu=[width],
            extend_prefix_lens_cpu=[0],
            **common,
        )
    else:
        mask = (
            torch.ones((width, width), dtype=torch.bool, device=backend.device).tril()
            if tree_mask is None
            else tree_mask.to(device=backend.device, dtype=torch.bool).contiguous()
        )
        lengths = mask.sum(1, dtype=torch.int32) + prefix
        indptr = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=backend.device),
                lengths.cumsum(0, dtype=torch.int32),
            )
        )
        indices = torch.empty(
            width * (prefix + width), dtype=torch.int64, device=backend.device
        )
        _tree_kv_indices[(width,)](
            cache.locations,
            mask,
            indptr,
            indices,
            prefix,
            width,
            triton.next_power_of_2(width),
            256,
        )
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=width,
            req_pool_indices=torch.full(
                (width,), cache.request_index, dtype=torch.int32, device=backend.device
            ),
            seq_lens=lengths,
            seq_lens_sum=int(indptr[-1]),
            spec_info=SimpleNamespace(kv_indptr=indptr, kv_indices=indices),
            **common,
        )
    batch.global_forward_mode = batch.forward_mode
    batch.attn_backend = backend.runner.attn_backend
    batch.req_to_token_pool = backend.runner.req_to_token_pool
    batch.token_to_kv_pool = backend.runner.token_to_kv_pool
    batch.attn_backend.init_forward_metadata(batch)
    return batch
