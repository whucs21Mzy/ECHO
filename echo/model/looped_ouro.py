"""Torch Ouro execution with one logical KV slot per recurrent layer."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .looped_attention import linear_attention_bias, tree_attention_bias
from .looped_cache import LoopedEchoCache, LoopedForwardOutput


class OuroLoopedEchoBackend(nn.Module):
    """Execute an Ouro model over an explicit range of recurrent steps."""

    def __init__(self, base_model: nn.Module, shallow_steps: int = 2):
        super().__init__()
        self.base_model = base_model
        self.decoder = base_model.model
        self.config = base_model.config
        self.num_layers = int(self.config.num_hidden_layers)
        self.total_steps = int(getattr(self.config, "total_ut_steps", 1))
        if not 0 < shallow_steps < self.total_steps:
            raise ValueError(
                f"shallow_steps must be in [1, {self.total_steps - 1}], got {shallow_steps}"
            )
        self.shallow_steps = int(shallow_steps)
        self.context_length = int(self.config.max_position_embeddings)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        shallow_steps: int = 2,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict = "auto",
        attn_implementation: str = "sdpa",
        **kwargs,
    ) -> "OuroLoopedEchoBackend":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            **kwargs,
        )
        if getattr(model.config, "model_type", None) != "ouro":
            raise ValueError(
                f"expected an Ouro checkpoint, got model_type={model.config.model_type!r}"
            )
        model.eval()
        return cls(model, shallow_steps=shallow_steps)

    @property
    def device(self) -> torch.device:
        return next(self.base_model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.base_model.parameters()).dtype

    @property
    def vocab_size(self) -> int:
        return int(self.config.vocab_size)

    @property
    def shallow_slot_end(self) -> int:
        return self.shallow_steps * self.num_layers

    @property
    def num_cache_slots(self) -> int:
        return self.total_steps * self.num_layers

    def new_cache(self) -> LoopedEchoCache:
        return LoopedEchoCache(self.num_cache_slots)

    @torch.inference_mode()
    def forward_steps(
        self,
        *,
        step_start: int,
        step_end: int,
        cache: LoopedEchoCache,
        position_ids: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        tree_mask: Optional[torch.Tensor] = None,
    ) -> LoopedForwardOutput:
        """Run the shared decoder stack for ``[step_start, step_end)``."""
        if (input_ids is None) == (hidden_states is None):
            raise ValueError("provide exactly one of input_ids or hidden_states")
        if not 0 <= step_start < step_end <= self.total_steps:
            raise ValueError(
                f"invalid recurrent step range [{step_start}, {step_end}) for {self.total_steps} steps"
            )
        if hidden_states is None:
            hidden_states = self.decoder.embed_tokens(input_ids)
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                f"position_ids shape {tuple(position_ids.shape)} does not match hidden states "
                f"{tuple(hidden_states.shape[:2])}"
            )

        first_slot = step_start * self.num_layers
        past_length = cache.get_seq_length(first_slot)
        query_length = hidden_states.shape[1]
        if tree_mask is None:
            attention_bias = linear_attention_bias(
                hidden_states.shape[0],
                query_length,
                past_length,
                hidden_states.dtype,
                hidden_states.device,
            )
        else:
            attention_bias = tree_attention_bias(
                tree_mask,
                past_length,
                hidden_states.dtype,
                hidden_states.shape[0],
            )

        position_embeddings = self.decoder.rotary_emb(hidden_states, position_ids)
        mask_mapping = {"full_attention": attention_bias}
        if getattr(self.decoder, "has_sliding_layers", False):
            # Ouro-1.4B is all-full-attention.  Refuse to silently apply an
            # incorrect tree mask if a future checkpoint enables SWA.
            if tree_mask is not None:
                raise NotImplementedError(
                    "tree attention with Ouro sliding-window layers"
                )
            mask_mapping["sliding_attention"] = attention_bias

        for current_ut in range(step_start, step_end):
            expected_length = cache.get_seq_length(current_ut * self.num_layers)
            if expected_length != past_length:
                raise RuntimeError(
                    f"UT step {current_ut} has cache length {expected_length}, "
                    f"but step {step_start} has {past_length}"
                )
            for decoder_layer in self.decoder.layers[: self.num_layers]:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_value=cache,
                    use_cache=True,
                    cache_position=position_ids[0],
                    position_embeddings=position_embeddings,
                    current_ut=current_ut,
                )
            hidden_states = self.decoder.norm(hidden_states)

        return LoopedForwardOutput(
            hidden_states=hidden_states,
            logits=self.base_model.lm_head(hidden_states),
        )

    def prefill(
        self, input_ids: torch.Tensor, cache: Optional[LoopedEchoCache] = None
    ) -> tuple[LoopedEchoCache, LoopedForwardOutput]:
        cache = self.new_cache() if cache is None else cache
        position_ids = torch.arange(
            input_ids.shape[1], device=input_ids.device, dtype=torch.long
        ).unsqueeze(0)
        output = self.forward_steps(
            input_ids=input_ids,
            step_start=0,
            step_end=self.total_steps,
            cache=cache,
            position_ids=position_ids,
        )
        return cache, output

    def forward_shallow_tree(
        self,
        tree_tokens: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        prefix_length: int,
        cache: LoopedEchoCache,
    ) -> LoopedForwardOutput:
        if tree_tokens.ndim == 1:
            tree_tokens = tree_tokens.unsqueeze(0)
        if tree_position_ids.ndim == 1:
            tree_position_ids = tree_position_ids.unsqueeze(0)
        return self.forward_steps(
            input_ids=tree_tokens,
            step_start=0,
            step_end=self.shallow_steps,
            cache=cache,
            position_ids=tree_position_ids + prefix_length,
            tree_mask=tree_mask,
        )

    def forward_deep(
        self,
        shallow_hidden_states: torch.Tensor,
        prefix_length: int,
        cache: LoopedEchoCache,
    ) -> LoopedForwardOutput:
        positions = torch.arange(
            prefix_length,
            prefix_length + shallow_hidden_states.shape[1],
            device=shallow_hidden_states.device,
            dtype=torch.long,
        ).unsqueeze(0)
        return self.forward_steps(
            hidden_states=shallow_hidden_states,
            step_start=self.shallow_steps,
            step_end=self.total_steps,
            cache=cache,
            position_ids=positions,
        )

    def forward_token_ids(
        self,
        input_ids: torch.Tensor,
        prefix_length: int,
        cache: LoopedEchoCache,
    ) -> LoopedForwardOutput:
        """Run one or more linear tokens through every recurrent step."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        positions = torch.arange(
            prefix_length,
            prefix_length + input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)
        return self.forward_steps(
            input_ids=input_ids,
            step_start=0,
            step_end=self.total_steps,
            cache=cache,
            position_ids=positions,
        )

    def prefill_ar(self, input_ids, cache=None):
        return self.prefill(input_ids, cache=cache)

    def forward_ar_token_ids(self, input_ids, prefix_length, cache):
        return self.forward_token_ids(input_ids, prefix_length, cache)
