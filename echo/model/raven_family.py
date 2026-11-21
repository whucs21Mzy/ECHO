"""Raven-family checkpoint layouts and recurrent-state initialization."""

import json
from pathlib import Path

import torch
from torch import nn


def initial_state_bank(config, capacity, *, dtype, device, seed=0):
    """Create deterministic recurrent states indexed by token position."""
    if float(getattr(config, "test_time_noise", 0)) != 0:
        raise ValueError(
            "ECHO's fixed-recurrence Raven path requires test_time_noise=0"
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bank = torch.empty((capacity, config.hidden_size), dtype=dtype, device="cpu")
    std = float(config.init_values["std"])
    if std > 0:
        nn.init.trunc_normal_(
            bank, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator
        )
    else:
        bank.zero_()
    bank.mul_(float(config.init_values["embed_scale"]))
    return bank.to(device)


def checkpoint_layout(path):
    """Infer norm/rotary layout from checkpoint structure, not its filename."""
    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.is_file():
        keys = set(json.loads(index.read_text())["weight_map"])
    else:
        from safetensors import safe_open

        with safe_open(
            path / "model.safetensors", framework="pt", device="cpu"
        ) as stream:
            keys = set(stream.keys())
    if any(".norm_3.weight" in key for key in keys):
        if not any(".norm_4.weight" in key for key in keys):
            raise ValueError("Incomplete Huginn sandwich normalization weights")
        return "huginn"
    return "olmo" if any(".attn.q_norm.weight" in key for key in keys) else "llama"
