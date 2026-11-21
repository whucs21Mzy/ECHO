"""Independent Torch ECHO execution for the released Raven/Huginn checkpoints.

Shallow execution owns prelude, the first K recurrent iterations and a q1
coda. Deep execution owns the remaining iterations and the target coda. The
accepted state retains both the raw recurrent latent and prelude injection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .looped_attention import linear_attention_bias, tree_attention_bias
from .looped_cache import LoopedEchoCache, LoopedForwardOutput
from .raven_family import initial_state_bank


def load_released_model(path, *, dtype, device):
    """Load every tensor into the original HF class without changing forward."""
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    path = Path(path)
    config = AutoConfig.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True
    )
    if config.model_type != "huginn_raven":
        raise ValueError("Expected a released Raven/Huginn checkpoint")
    cls = get_class_from_dynamic_module(
        "raven_modeling_minimal.RavenForCausalLM",
        str(path),
        local_files_only=True,
    )
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        model = cls(config)
    finally:
        torch.set_default_dtype(previous)
    index = path / "model.safetensors.index.json"
    mapping = json.loads(index.read_text())["weight_map"] if index.is_file() else None
    shards = (
        sorted(set(mapping.values())) if mapping is not None else ["model.safetensors"]
    )
    expected, loaded = set(model.state_dict()), set()
    for shard in shards:
        values = load_file(str(path / shard), device="cpu")
        if loaded & values.keys():
            raise ValueError("Duplicate checkpoint tensor")
        if mapping is not None and any(mapping.get(key) != shard for key in values):
            raise ValueError("Checkpoint index disagrees with its shard")
        result = model.load_state_dict(values, strict=False)
        if result.unexpected_keys:
            raise ValueError(
                f"Unexpected checkpoint tensors: {result.unexpected_keys[:8]}"
            )
        loaded.update(values)
        del values
    if loaded != expected or (mapping is not None and loaded != set(mapping)):
        raise ValueError(
            f"Checkpoint coverage: missing={sorted(expected - loaded)[:8]}, extra={sorted(loaded - expected)[:8]}"
        )
    # The released module disables CUDA math SDPA at import time; custom tree
    # masks with GQA need that documented Torch fallback to remain available.
    torch.backends.cuda.enable_math_sdp(True)
    return model.eval().to(device), sys.modules[cls.__module__]


class RavenLoopedEchoBackend(nn.Module):
    """Torch tree attention over prelude/core and independent q1/target codas."""

    def __init__(
        self, model, module, *, shallow_steps, context_length=2048, initial_seed=0
    ):
        super().__init__()
        self.base_model = model
        self.source_module = module
        self.config = model.config
        self.total_steps = int(self.config.mean_recurrence)
        self.shallow_steps = int(shallow_steps)
        if not 0 < self.shallow_steps < self.total_steps:
            raise ValueError("K must precede the full S-iteration target")
        self.num_layers = int(self.config.n_layers_in_recurrent_block)
        self.pre_count = int(self.config.n_layers_in_prelude)
        self.post_count = int(self.config.n_layers_in_coda)
        self.context_length = context_length
        sample = model.transformer.core_block[0]
        self.layout = (
            "huginn"
            if hasattr(sample, "norm_3")
            else "olmo"
            if hasattr(sample.attn, "q_norm")
            else "llama"
        )
        self.initial_seed = initial_seed
        self.register_buffer(
            "initial_hidden_by_position",
            initial_state_bank(
                self.config,
                context_length,
                dtype=self.dtype,
                device=self.device,
                seed=initial_seed,
            ),
            persistent=False,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        *,
        shallow_steps,
        torch_dtype=torch.bfloat16,
        device="cuda:0",
        context_length=2048,
        initial_seed=0,
    ):
        model, module = load_released_model(
            model_path, dtype=torch_dtype, device=device
        )
        return cls(
            model,
            module,
            shallow_steps=shallow_steps,
            context_length=context_length,
            initial_seed=initial_seed,
        )

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    @property
    def dtype(self):
        return next(self.base_model.parameters()).dtype

    @property
    def vocab_size(self):
        return int(self.config.vocab_size)

    @property
    def shallow_slot_end(self):
        return self.pre_count + self.shallow_steps * self.num_layers + self.post_count

    @property
    def num_cache_slots(self):
        return self.pre_count + self.total_steps * self.num_layers + 2 * self.post_count

    def core_offset(self, step):
        return (
            self.pre_count
            + step * self.num_layers
            + (self.post_count if step >= self.shallow_steps else 0)
        )

    def new_cache(self):
        return LoopedEchoCache(self.num_cache_slots)

    def _frequencies(self, hidden, positions):
        if self.layout == "huginn":
            return self.base_model.freqs_cis.index_select(1, positions.reshape(-1))
        return self.base_model.rotary_emb(hidden, positions)

    def _attention(self, attention, hidden, frequencies, slot, bias, cache):
        batch, width, _ = hidden.shape
        q, k, v = attention.Wqkv(hidden).split(attention.chunks, dim=-1)
        if self.layout == "olmo":
            q, k = attention.q_norm(q), attention.k_norm(k)
        q = q.view(batch, width, attention.n_head, attention.head_dim)
        k = k.view(batch, width, attention.n_kv_heads, attention.head_dim)
        v = v.view(batch, width, attention.n_kv_heads, attention.head_dim)
        if self.config.qk_bias:
            q_bias, k_bias = attention.qk_bias.split(1, dim=0)
            q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(k.dtype)
        if self.layout == "huginn":
            q, k = self.source_module.apply_rotary_emb_complex_like(
                q, k, freqs_cis=frequencies
            )
            q, k = q.transpose(1, 2), k.transpose(1, 2)
        else:
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            q, k = self.source_module.apply_rotary_pos_emb(q, k, *frequencies)
        v = v.transpose(1, 2)
        k, v = cache.update(k, v, slot)
        attention_fn = getattr(
            self.base_model, "_echo_sdpa", F.scaled_dot_product_attention
        )
        out = attention_fn(
            q,
            k,
            v,
            attn_mask=bias,
            dropout_p=0.0,
            enable_gqa=attention.n_head != attention.n_kv_heads,
        )
        out = (
            out.transpose(1, 2)
            .reshape(batch, width, attention.n_head * attention.head_dim)
            .contiguous()
        )
        return attention.proj(out)

    def _blocks(self, blocks, hidden, frequencies, offset, bias, cache):
        for index, block in enumerate(blocks):
            slot = offset + index
            if self.layout == "huginn":
                out = self._attention(
                    block.attn, block.norm_1(hidden), frequencies, slot, bias, cache
                )
                hidden = block.norm_2(hidden + out)
                hidden = block.norm_4(hidden + block.mlp(block.norm_3(hidden)))
            elif self.layout == "olmo":
                hidden = hidden + block.norm_1(
                    self._attention(block.attn, hidden, frequencies, slot, bias, cache)
                )
                hidden = hidden + block.norm_2(block.mlp(hidden))
            else:
                hidden = hidden + self._attention(
                    block.attn, block.norm_1(hidden), frequencies, slot, bias, cache
                )
                hidden = hidden + block.mlp(block.norm_2(hidden))
        return hidden

    def _post(self, hidden, frequencies, offset, bias, cache, *, logits=True):
        if self.layout == "huginn":
            hidden = self.base_model.transformer.ln_f(hidden)
        hidden = self._blocks(
            self.base_model.transformer.coda, hidden, frequencies, offset, bias, cache
        )
        hidden = self.base_model.transformer.ln_f(hidden)
        return self.base_model.lm_head(hidden) if logits else None

    @torch.inference_mode()
    def forward_steps(
        self,
        *,
        step_start,
        step_end,
        cache,
        position_ids,
        input_ids=None,
        hidden_states=None,
        tree_mask=None,
        fill_shallow=True,
    ):
        if not 0 <= step_start < step_end <= self.total_steps:
            raise ValueError("Invalid recurrent iteration range")
        if (input_ids is None) == (hidden_states is None):
            raise ValueError(
                "Provide either token IDs or a packed latent/injection state"
            )
        positions = position_ids.reshape(1, -1).to(device=self.device, dtype=torch.long)
        if int(positions.max()) >= self.context_length:
            raise ValueError("Token positions exceed the configured context")
        start = 0 if step_start == 0 else self.core_offset(step_start)
        prefix = cache.get_seq_length(start)
        if input_ids is not None:
            if step_start != 0:
                raise ValueError("Token IDs enter the prelude before iteration zero")
            embedded = self.base_model.transformer.wte(
                input_ids.reshape(1, -1).to(self.device)
            )
            embedded = (
                embedded * self.base_model.emb_scale
                if self.base_model.emb_scale != 1
                else embedded
            )
            frequencies = self._frequencies(embedded, positions)
            latent = self.initial_hidden_by_position.index_select(0, positions[0])[None]
        else:
            latent, embedded = hidden_states.to(self.device).split(
                self.config.hidden_size, dim=-1
            )
            frequencies = self._frequencies(latent, positions)
        bias = (
            linear_attention_bias(1, positions.numel(), prefix, self.dtype, self.device)
            if tree_mask is None
            else tree_attention_bias(tree_mask, prefix, self.dtype, 1)
        )
        if input_ids is not None:
            embedded = self._blocks(
                self.base_model.transformer.prelude,
                embedded,
                frequencies,
                0,
                bias,
                cache,
            )
        for step in range(step_start, step_end):
            latent = self.base_model.transformer.adapter(
                torch.cat((latent, embedded), dim=-1)
            )
            latent = self._blocks(
                self.base_model.transformer.core_block,
                latent,
                frequencies,
                self.core_offset(step),
                bias,
                cache,
            )
            if (
                step + 1 == self.shallow_steps
                and step_end > self.shallow_steps
                and fill_shallow
            ):
                self._post(
                    latent,
                    frequencies,
                    self.pre_count + self.shallow_steps * self.num_layers,
                    bias,
                    cache,
                    logits=False,
                )
        offset = (
            self.pre_count + self.shallow_steps * self.num_layers
            if step_end == self.shallow_steps
            else self.pre_count + self.total_steps * self.num_layers + self.post_count
        )
        logits = self._post(latent, frequencies, offset, bias, cache)
        return LoopedForwardOutput(torch.cat((latent, embedded), dim=-1), logits)

    def prefill(self, input_ids, cache=None, *, fill_shallow=True):
        cache = self.new_cache() if cache is None else cache
        output = self.forward_steps(
            step_start=0,
            step_end=self.total_steps,
            cache=cache,
            input_ids=input_ids,
            position_ids=torch.arange(input_ids.numel(), device=self.device),
            fill_shallow=fill_shallow,
        )
        return cache, output

    def prefill_ar(self, input_ids, cache=None):
        return self.prefill(input_ids, cache=cache, fill_shallow=False)

    def forward_shallow_tree(
        self, tree_tokens, tree_mask, tree_position_ids, prefix_length, cache
    ):
        return self.forward_steps(
            step_start=0,
            step_end=self.shallow_steps,
            cache=cache,
            input_ids=tree_tokens,
            position_ids=tree_position_ids + prefix_length,
            tree_mask=tree_mask,
        )

    def forward_deep(self, shallow_hidden_states, prefix_length, cache):
        positions = torch.arange(
            prefix_length,
            prefix_length + shallow_hidden_states.shape[1],
            device=self.device,
        )
        return self.forward_steps(
            step_start=self.shallow_steps,
            step_end=self.total_steps,
            cache=cache,
            hidden_states=shallow_hidden_states,
            position_ids=positions,
        )

    def forward_token_ids(self, input_ids, prefix_length, cache, *, fill_shallow=True):
        positions = torch.arange(
            prefix_length, prefix_length + input_ids.numel(), device=self.device
        )
        return self.forward_steps(
            step_start=0,
            step_end=self.total_steps,
            cache=cache,
            input_ids=input_ids,
            position_ids=positions,
            fill_shallow=fill_shallow,
        )

    def forward_ar_token_ids(self, input_ids, prefix_length, cache):
        return self.forward_token_ids(
            input_ids, prefix_length, cache, fill_shallow=False
        )
