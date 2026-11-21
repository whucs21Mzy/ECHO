"""Shared-weight Ouro blocks with one SGLang KV layer per logical UT layer."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gemma2 import Gemma2DecoderLayer
from sglang.srt.models.llama import LlamaAttention, LlamaMLP
from sglang.srt.utils import add_prefix
from torch import nn
from transformers import PreTrainedConfig


class EchoOuroConfig(PreTrainedConfig):
    """Load checkpoint metadata without importing its HF model implementation."""

    model_type = "ouro"


def map_checkpoint_weight(name):
    roots = (
        ("model.embed_tokens.", "embed_tokens."),
        ("model.layers.", "layers."),
        ("model.norm.", "step_norm."),
        ("model.early_exit_gate.", "early_exit_gate."),
        ("lm_head.", "lm_head."),
    )
    target = None
    for original, replacement in roots:
        if name.startswith(original):
            target = replacement + name[len(original) :]
            break
    if target is None:
        return None, None
    # Apply one norm rename to the checkpoint name, never to an earlier rename.
    for original, replacement in (
        (".input_layernorm_2.", ".post_attention_layernorm."),
        (".post_attention_layernorm_2.", ".post_feedforward_layernorm."),
        (".post_attention_layernorm.", ".pre_feedforward_layernorm."),
    ):
        if original in target:
            target = target.replace(original, replacement)
            break
    for original, replacement, shard in (
        (".q_proj.", ".qkv_proj.", "q"),
        (".k_proj.", ".qkv_proj.", "k"),
        (".v_proj.", ".qkv_proj.", "v"),
        (".gate_proj.", ".gate_up_proj.", 0),
        (".up_proj.", ".gate_up_proj.", 1),
    ):
        if original in target:
            return target.replace(original, replacement), shard
    return target, None


class OuroDecoderLayer(Gemma2DecoderLayer):
    """Ouro's four RMS norms and SwiGLU with native SGLang attention."""

    def __init__(self, config, layer_id, quant_config, prefix):
        nn.Module.__init__(self)
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
            rope_is_neox_style=True,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            bias=False,
        )
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            setattr(self, name, RMSNorm(config.hidden_size, eps=config.rms_norm_eps))

    def forward(self, positions, hidden_states, forward_batch):
        hidden, residual = super().forward(
            positions, hidden_states, forward_batch, residual=None
        )
        return hidden + residual


class EchoOuroForCausalLM(nn.Module):
    """Expose complete and recurrence-range execution over the same weights.

    ModelRunner owns weight loading, packed native projections and the KV pool.
    The ECHO backend owns tree masks and the early/deep verification schedule.
    """

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if getattr(config, "model_type", None) != "ouro":
            raise ValueError("EchoOuroForCausalLM requires an Ouro checkpoint")
        if any(kind != "full_attention" for kind in getattr(config, "layer_types", [])):
            raise NotImplementedError(
                "The Ouro ECHO backend requires full-attention layers"
            )
        self.config = config
        self.total_steps = int(config.total_ut_steps)
        self.num_layers = int(config.num_hidden_layers)
        self.start_layer = 0
        self.end_layer = self.total_steps * self.num_layers
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            OuroDecoderLayer(config, i, quant_config, add_prefix(f"layers.{i}", prefix))
            for i in range(self.num_layers)
        )
        self.step_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Retain the released head for complete checkpoint coverage. Full-depth
        # target verification does not invoke its adaptive-exit policy.
        self.early_exit_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("early_exit_gate", prefix),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        if config.tie_word_embeddings:
            self.lm_head.tie_weights(self.embed_tokens)
        self.logits_processor = LogitsProcessor(config)
        self.requires_grad_(False)

    def forward_hidden_steps(
        self,
        positions,
        forward_batch,
        *,
        step_start,
        step_end,
        input_ids=None,
        hidden_states=None,
    ):
        if (input_ids is None) == (hidden_states is None):
            raise ValueError("Provide exactly one of input_ids and hidden_states")
        hidden = (
            self.embed_tokens(input_ids) if hidden_states is None else hidden_states
        )
        for step in range(step_start, step_end):
            for index, layer in enumerate(self.layers):
                layer.self_attn.attn.layer_id = step * self.num_layers + index
                hidden = layer(positions, hidden, forward_batch)
            hidden = self.step_norm(hidden)
        return hidden

    def all_logits(self, hidden, forward_batch):
        # Preserve native SGLang projection, scaling and logits precision while
        # retaining the rows ordinary prefill would discard.
        metadata = LogitsMetadata.from_forward_batch(forward_batch)
        return self.logits_processor._get_logits(hidden, self.lm_head, metadata)

    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        hidden = self.forward_hidden_steps(
            positions,
            forward_batch,
            step_start=0,
            step_end=self.total_steps,
            input_ids=input_ids if input_embeds is None else None,
            hidden_states=input_embeds,
        )
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        parameters = dict(self.named_parameters(remove_duplicate=False))
        loaded, shards, unmatched = set(), {}, []
        for name, value in weights:
            if name.endswith("rotary_emb.inv_freq"):
                continue
            target, shard = map_checkpoint_weight(name)
            parameter = parameters.get(target)
            if parameter is None:
                unmatched.append(name)
                continue
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            if shard is None:
                loader(parameter, value)
            else:
                loader(parameter, value, shard)
            loaded.add(id(parameter))
            shards.setdefault(id(parameter), set()).add(shard)
        missing, incomplete = [], []
        for name, parameter in self.named_parameters():
            if id(parameter) not in loaded:
                missing.append(name)
                continue
            seen = shards[id(parameter)]
            expected = (
                {"q", "k", "v"}
                if ".qkv_proj." in name
                else {0, 1}
                if ".gate_up_proj." in name
                else set()
            )
            if None not in seen and not expected <= seen:
                incomplete.append(name)
        if missing or incomplete or unmatched:
            raise ValueError(
                f"Incomplete Ouro checkpoint: missing={missing[:8]}, packed={incomplete[:8]}, unmatched={unmatched[:8]}"
            )


EntryClass = EchoOuroForCausalLM
