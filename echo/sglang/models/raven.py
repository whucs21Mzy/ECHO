"""Native SGLang Raven/Huginn blocks for independent hierarchical ECHO."""

from __future__ import annotations

import os

import torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaDecoderLayer
from sglang.srt.models.olmo2 import Olmo2DecoderLayer
from sglang.srt.utils import add_prefix
from torch import nn
from transformers import PreTrainedConfig

from echo.model.raven_family import checkpoint_layout, initial_state_bank


class EchoRavenConfig(PreTrainedConfig):
    model_type = "huginn_raven"

    def __init__(self, **kwargs):
        kwargs.setdefault("hidden_size", kwargs.get("n_embd", 5280))
        kwargs.setdefault("num_attention_heads", kwargs.get("n_heads", 55))
        kwargs.setdefault("num_key_value_heads", kwargs["num_attention_heads"])
        kwargs.setdefault("num_hidden_layers", kwargs.get("n_layers", 8))
        kwargs.setdefault(
            "head_dim", kwargs["hidden_size"] // kwargs["num_attention_heads"]
        )
        kwargs.setdefault("max_position_embeddings", kwargs.get("block_size", 4096))
        kwargs.setdefault("vocab_size", kwargs.get("padded_vocab_size", 65536))
        kwargs.setdefault("rms_norm_eps", kwargs.get("norm_eps", 1e-6))
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("rope_theta", kwargs.get("rope_base", 50000.0))
        rope = dict(kwargs.pop("rope_scaling", None) or {})
        rope.update(kwargs.get("rope_parameters") or {})
        rope.setdefault("rope_type", "default")
        rope.setdefault("rope_theta", kwargs["rope_theta"])
        kwargs["rope_parameters"] = rope
        kwargs.setdefault("tie_word_embeddings", kwargs.get("tie_embeddings", False))
        kwargs.setdefault("attention_bias", False)
        super().__init__(**kwargs)


class AdjacentRotaryEmbedding(nn.Module):
    """Huginn's released FP32 adjacent-pair rotary products and table."""

    def __init__(self, head_dim, capacity, base):
        super().__init__()
        self.head_dim = head_dim
        inv = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        angle = torch.outer(torch.arange(capacity, dtype=torch.float32), inv)
        self.register_buffer(
            "cos_sin_cache", torch.cat((angle.cos(), angle.sin()), -1), persistent=False
        )

    def _apply(self, fn, recurse=True):
        original = self.cos_sin_cache
        super()._apply(fn, recurse=recurse)
        self.cos_sin_cache = original.to(device=self.cos_sin_cache.device)
        return self

    def load_frequencies(self, frequencies):
        if (
            frequencies.ndim != 5
            or frequencies.shape[0] != 1
            or frequencies.shape[-2:] != (self.head_dim // 2, 2)
        ):
            raise ValueError("Unexpected released Huginn frequency table")
        paired = frequencies[0, :, 0].float().to(self.cos_sin_cache.device)
        self.cos_sin_cache = torch.cat((paired[..., 0], paired[..., 1]), -1)

    def forward(self, positions, q, k, bias=None):
        if q.is_cuda and os.environ.get("ECHO_RAVEN_FUSED_ROPE", "1") == "1":
            from echo.sglang.ops.raven_rope import huginn_rope

            return huginn_rope(positions, q, k, self.cos_sin_cache, self.head_dim, bias)
        if bias is not None:
            q, k = q + bias[0].reshape(1, -1), k + bias[1].reshape(1, -1)
        trig = self.cos_sin_cache.index_select(0, positions.reshape(-1))
        cos, sin = (x.unsqueeze(1) for x in trig.chunk(2, -1))
        q_heads = q.shape[-1] // self.head_dim
        values = torch.cat(
            (
                q.reshape(len(positions), -1, self.head_dim),
                k.reshape(len(positions), -1, self.head_dim),
            ),
            dim=1,
        )
        pairs = values.reshape(len(positions), -1, self.head_dim // 2, 2).float()
        real, imag = pairs[..., 0], pairs[..., 1]
        rotated = (
            torch.stack((real * cos - imag * sin, imag * cos + real * sin), -1)
            .flatten(-2)
            .to(q.dtype)
        )
        q_out, k_out = rotated.split((q_heads, k.shape[-1] // self.head_dim), dim=1)
        return q_out.reshape(q.shape), k_out.reshape(k.shape)


class OlmoRotaryEmbedding(nn.Module):
    """OLMo's FP32 half-pair rotary products with CPU-derived frequencies."""

    def __init__(self, head_dim, capacity, base):
        super().__init__()
        self.head_dim = head_dim
        inv = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, device="cpu", dtype=torch.float32)
                / head_dim
            )
        )
        inv = inv.to(device=torch.empty(0).device)
        angle = torch.outer(
            torch.arange(capacity, device=inv.device, dtype=torch.float32), inv
        )
        self.register_buffer(
            "cos_sin_cache", torch.cat((angle.cos(), angle.sin()), -1), persistent=False
        )

    def _apply(self, fn, recurse=True):
        original = self.cos_sin_cache
        super()._apply(fn, recurse=recurse)
        self.cos_sin_cache = original.to(device=self.cos_sin_cache.device)
        return self

    def forward(self, positions, q, k):
        if q.is_cuda and os.environ.get("ECHO_RAVEN_FUSED_ROPE", "1") == "1":
            from echo.sglang.ops.raven_rope import olmo_rope

            return olmo_rope(positions, q, k, self.cos_sin_cache, self.head_dim)
        trig = self.cos_sin_cache.index_select(0, positions.reshape(-1))
        cos, sin = (x.unsqueeze(1) for x in trig.chunk(2, -1))

        def rotate(x):
            first, second = (
                x.reshape(len(positions), -1, self.head_dim).float().chunk(2, -1)
            )
            return (
                torch.cat((first * cos - second * sin, second * cos + first * sin), -1)
                .to(x.dtype)
                .reshape(x.shape)
            )

        return rotate(q), rotate(k)


class RavenLlamaLayer(LlamaDecoderLayer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.norm_eps, cast_x_before_out_mul=True
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.norm_eps, cast_x_before_out_mul=True
        )
        self.mlp.act_fn._forward_method = self.mlp.act_fn.forward_native

    def forward(self, positions, hidden, batch):
        hidden = hidden + self.self_attn(positions, self.input_layernorm(hidden), batch)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class RavenOlmoLayer(Olmo2DecoderLayer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.norm_eps, cast_x_before_out_mul=True
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.norm_eps, cast_x_before_out_mul=True
        )
        self.self_attn.q_norm = RMSNorm(
            config.num_attention_heads * config.head_dim,
            eps=config.norm_eps,
            cast_x_before_out_mul=True,
        )
        self.self_attn.k_norm = RMSNorm(
            config.num_key_value_heads * config.head_dim,
            eps=config.norm_eps,
            cast_x_before_out_mul=True,
        )
        self.self_attn.rotary_emb = OlmoRotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.self_attn._apply_qk_norm = lambda q, k: (
            self.self_attn.q_norm(q),
            self.self_attn.k_norm(k),
        )
        self.mlp.act_fn._forward_method = self.mlp.act_fn.forward_native


class HuginnLayer(LlamaDecoderLayer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        del self.input_layernorm, self.post_attention_layernorm
        for name in ("norm_1", "norm_2", "norm_3", "norm_4"):
            setattr(
                self,
                name,
                RMSNorm(
                    config.hidden_size, eps=config.norm_eps, cast_x_before_out_mul=True
                ),
            )
        self.mlp.act_fn._forward_method = self.mlp.act_fn.forward_native
        self.self_attn.rotary_emb = AdjacentRotaryEmbedding(
            config.head_dim, config.block_size, config.rope_theta
        )
        if config.qk_bias:
            if config.num_attention_heads != config.num_key_value_heads:
                raise ValueError(
                    "Huginn trained Q/K bias requires equal Q/K head counts"
                )
            self.self_attn.qk_bias = nn.Parameter(
                torch.zeros(2, 1, config.num_attention_heads, config.head_dim)
            )

    def forward(self, positions, hidden, batch):
        attention = self.self_attn
        qkv, _ = attention.qkv_proj(self.norm_1(hidden))
        q, k, v = qkv.split(
            (attention.q_size, attention.kv_size, attention.kv_size), -1
        )
        q, k = attention.rotary_emb(
            positions, q, k, getattr(attention, "qk_bias", None)
        )
        out, _ = attention.o_proj(attention.attn(q, k, v, batch))
        hidden = self.norm_2(hidden + out)
        return self.norm_4(hidden + self.mlp(self.norm_3(hidden)))


def map_weight(name, layout):
    roots = (
        ("transformer.prelude.", "pre_layers."),
        ("transformer.core_block.", "core_layers."),
        ("transformer.coda.", "post_layers."),
        ("transformer.wte.", "embed_tokens."),
        ("transformer.adapter.", "adapter."),
        ("transformer.ln_f.", "final_norm."),
        ("lm_head.", "lm_head."),
    )
    for old, new in roots:
        if name.startswith(old):
            target = new + name[len(old) :]
            break
    else:
        return None
    for old, new in (
        (".attn.Wqkv.", ".self_attn.qkv_proj."),
        (".attn.proj.", ".self_attn.o_proj."),
        (".attn.q_norm.", ".self_attn.q_norm."),
        (".attn.k_norm.", ".self_attn.k_norm."),
        (".attn.qk_bias", ".self_attn.qk_bias"),
        (".mlp.fc.", ".mlp.gate_up_proj."),
        (".mlp.proj.", ".mlp.down_proj."),
    ):
        target = target.replace(old, new)
    names = (
        {".norm_1.": ".input_layernorm.", ".norm_2.": ".post_attention_layernorm."}
        if layout == "llama"
        else {
            ".norm_1.": ".post_attention_layernorm.",
            ".norm_2.": ".post_feedforward_layernorm.",
        }
        if layout == "olmo"
        else {}
    )
    for old, new in names.items():
        target = target.replace(old, new)
    return target


class EchoRavenForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if (
            float(getattr(config, "test_time_noise", 0)) != 0
            or config.injection_type != "linear"
        ):
            raise ValueError(
                "Raven ECHO requires a noise-free fixed recurrence and linear injection"
            )
        self.config = config
        self.layout = checkpoint_layout(config._name_or_path)
        self.total_steps = int(config.mean_recurrence)
        self.num_layers = int(config.n_layers_in_recurrent_block)
        self.pre_count, self.post_count = (
            int(config.n_layers_in_prelude),
            int(config.n_layers_in_coda),
        )
        self.shallow_steps = 1
        self.start_layer = 0
        self.end_layer = (
            self.pre_count + self.total_steps * self.num_layers + 2 * self.post_count
        )
        layer_cls = {
            "llama": RavenLlamaLayer,
            "olmo": RavenOlmoLayer,
            "huginn": HuginnLayer,
        }[self.layout]
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.pre_layers = nn.ModuleList(
            layer_cls(
                config,
                layer_id=i,
                quant_config=quant_config,
                prefix=add_prefix(f"pre_layers.{i}", prefix),
            )
            for i in range(self.pre_count)
        )
        self.core_layers = nn.ModuleList(
            layer_cls(
                config,
                layer_id=self.pre_count + i,
                quant_config=quant_config,
                prefix=add_prefix(f"core_layers.{i}", prefix),
            )
            for i in range(self.num_layers)
        )
        self.post_layers = nn.ModuleList(
            layer_cls(
                config,
                layer_id=self.pre_count + self.num_layers + i,
                quant_config=quant_config,
                prefix=add_prefix(f"post_layers.{i}", prefix),
            )
            for i in range(self.post_count)
        )
        self.adapter = ReplicatedLinear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=config.bias,
            quant_config=quant_config,
            prefix=add_prefix("adapter", prefix),
        )
        self.final_norm = RMSNorm(
            config.hidden_size, eps=config.norm_eps, cast_x_before_out_mul=True
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        if config.tie_embeddings:
            self.lm_head.tie_weights(self.embed_tokens)
        self.logits_processor = LogitsProcessor(config)
        self.register_buffer("initial_hidden_by_position", None, persistent=False)
        self.requires_grad_(False)

    @property
    def shallow_slot_end(self):
        return self.pre_count + self.shallow_steps * self.num_layers + self.post_count

    def configure_split(self, shallow_steps, context_length, initial_seed):
        self.shallow_steps = shallow_steps
        self.initial_hidden_by_position = initial_state_bank(
            self.config,
            context_length,
            dtype=self.embed_tokens.weight.dtype,
            device=self.embed_tokens.weight.device,
            seed=initial_seed,
        )

    def core_offset(self, step):
        return (
            self.pre_count
            + step * self.num_layers
            + (self.post_count if step >= self.shallow_steps else 0)
        )

    @staticmethod
    def blocks(layers, hidden, positions, batch, offset):
        for i, layer in enumerate(layers):
            layer.self_attn.attn.layer_id = offset + i
            hidden = layer(positions, hidden, batch)
        return hidden

    def post(self, hidden, positions, batch, offset):
        if self.layout == "huginn":
            hidden = self.final_norm(hidden)
        hidden = self.blocks(self.post_layers, hidden, positions, batch, offset)
        return self.final_norm(hidden)

    def execute_steps(
        self,
        positions,
        batch,
        *,
        step_start,
        step_end,
        input_ids=None,
        packed_hidden=None,
        fill_shallow=True,
    ):
        if input_ids is not None:
            injected = self.embed_tokens(input_ids)
            scale = float(self.config.init_values["embed_scale"])
            injected = injected * scale if scale != 1 else injected
            injected = self.blocks(self.pre_layers, injected, positions, batch, 0)
            latent = self.initial_hidden_by_position.index_select(0, positions.long())
        else:
            latent, injected = packed_hidden.split(self.config.hidden_size, dim=-1)
        for step in range(step_start, step_end):
            latent, _ = self.adapter(torch.cat((latent, injected), dim=-1))
            latent = self.blocks(
                self.core_layers, latent, positions, batch, self.core_offset(step)
            )
            if (
                step + 1 == self.shallow_steps
                and step_end > self.shallow_steps
                and fill_shallow
            ):
                self.post(
                    latent,
                    positions,
                    batch,
                    self.pre_count + self.shallow_steps * self.num_layers,
                )
        offset = (
            self.pre_count + self.shallow_steps * self.num_layers
            if step_end == self.shallow_steps
            else self.pre_count + self.total_steps * self.num_layers + self.post_count
        )
        head_hidden = self.post(latent, positions, batch, offset)
        return torch.cat((latent, injected), -1), head_hidden

    def all_logits(self, hidden, batch):
        return self.logits_processor._get_logits(
            hidden, self.lm_head, LogitsMetadata.from_forward_batch(batch)
        )

    def forward(self, input_ids, positions, forward_batch, **kwargs):
        _, head = self.execute_steps(
            positions,
            forward_batch,
            step_start=0,
            step_end=self.total_steps,
            input_ids=input_ids,
            fill_shallow=False,
        )
        return self.logits_processor(input_ids, head, self.lm_head, forward_batch)

    def load_weights(self, weights):
        parameters = dict(self.named_parameters(remove_duplicate=False))
        loaded, unmatched = set(), []
        frequencies = self.layout != "huginn"
        for name, value in weights:
            if name == "freqs_cis" and self.layout == "huginn":
                for layer in (*self.pre_layers, *self.core_layers, *self.post_layers):
                    layer.self_attn.rotary_emb.load_frequencies(value)
                frequencies = True
                continue
            if name.endswith("rotary_emb.inv_freq"):
                continue
            target = map_weight(name, self.layout)
            parameter = parameters.get(target)
            if parameter is None:
                unmatched.append(name)
                continue
            getattr(parameter, "weight_loader", default_weight_loader)(parameter, value)
            loaded.add(id(parameter))
        missing = [
            name
            for name, parameter in self.named_parameters()
            if id(parameter) not in loaded
        ]
        if missing or unmatched or not frequencies:
            raise ValueError(
                f"Raven checkpoint coverage: missing={missing[:8]}, unmatched={unmatched[:8]}, frequencies={frequencies}"
            )


EntryClass = EchoRavenForCausalLM
