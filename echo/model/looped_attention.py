"""Causal and tree attention masks for Torch recurrent execution."""

import torch


def _attention_bias(
    allowed: torch.Tensor,
    dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    """Convert a 2-D allowed-attention matrix to a 4-D additive bias."""
    if allowed.dtype != torch.bool:
        allowed = allowed.to(torch.bool)
    bias = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    bias.masked_fill_(~allowed, torch.finfo(dtype).min)
    return bias[None, None, :, :].expand(batch_size, 1, -1, -1)


def linear_attention_bias(
    batch_size: int,
    query_length: int,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    prefix = torch.ones((query_length, past_length), dtype=torch.bool, device=device)
    causal = torch.ones(
        (query_length, query_length), dtype=torch.bool, device=device
    ).tril()
    return _attention_bias(torch.cat((prefix, causal), dim=1), dtype, batch_size)


def tree_attention_bias(
    tree_mask: torch.Tensor,
    past_length: int,
    dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    if tree_mask.ndim != 2 or tree_mask.shape[0] != tree_mask.shape[1]:
        raise ValueError("tree_mask must be a square [tree_nodes, tree_nodes] matrix")
    prefix = torch.ones(
        (tree_mask.shape[0], past_length),
        dtype=torch.bool,
        device=tree_mask.device,
    )
    allowed = torch.cat((prefix, tree_mask.to(torch.bool)), dim=1)
    return _attention_bias(allowed, dtype, batch_size)
