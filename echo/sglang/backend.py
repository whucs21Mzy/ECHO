"""Split execution on SGLang; family classes define state and KV roles."""

from abc import ABC, abstractmethod

import torch

from echo.model.looped_cache import LoopedForwardOutput

from .cache import SGLangLoopedCache
from .ops.tree_attention import make_forward_batch
from .runtime import load_runner


class SGLangLoopedBackend(ABC):
    def __init__(
        self, runner, shallow_steps, context_length, max_tree_nodes, initial_seed=0
    ):
        self.runner = runner
        self.model = runner.model
        self.config = self.model.config
        self.total_steps = self.model.total_steps
        self.num_layers = self.model.num_layers
        if not 0 < shallow_steps < self.total_steps:
            raise ValueError("The shallow recurrence must precede the full target")
        self.shallow_steps = shallow_steps
        self.context_length = context_length
        self.max_tree_nodes = max_tree_nodes
        self.initial_seed = initial_seed

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        *,
        shallow_steps=1,
        torch_dtype=torch.bfloat16,
        context_length=2048,
        max_tree_nodes=64,
        mem_fraction_static=None,
        deterministic=True,
        initial_seed=0,
    ):
        runner = load_runner(
            model_path,
            torch_dtype=torch_dtype,
            context_length=context_length,
            max_tree_nodes=max_tree_nodes,
            mem_fraction_static=mem_fraction_static,
            deterministic=deterministic,
        )
        return cls(runner, shallow_steps, context_length, max_tree_nodes, initial_seed)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def vocab_size(self):
        return int(self.config.vocab_size)

    @property
    @abstractmethod
    def shallow_slot_end(self):
        """Exclusive end of the early-stage logical KV slots."""

    @property
    @abstractmethod
    def num_cache_slots(self):
        """Total logical KV slots, including independent readout slots."""

    @abstractmethod
    def _cache_ranges(self, step_start, step_end, *, from_tokens, fill_shallow):
        """Return (offset, count) for each KV range touched by this execution."""

    @abstractmethod
    def _execute_steps(
        self, positions, batch, step_start, step_end, ids, hidden, fill_shallow
    ):
        """Return the continuation state and the state passed to the output head."""

    def new_cache(self):
        return SGLangLoopedCache(self)

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
        last_logits_only=False,
    ):
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        if (input_ids is None) == (hidden_states is None):
            raise ValueError("Provide exactly one of input_ids and hidden_states")
        if not 0 <= step_start < step_end <= self.total_steps:
            raise ValueError("Invalid recurrent range")
        if input_ids is not None and step_start != 0:
            raise ValueError("Token IDs enter before recurrent iteration zero")
        positions = position_ids.reshape(-1).to(device=self.device, dtype=torch.long)
        width = positions.numel()
        ids = (
            torch.zeros(width, dtype=torch.long, device=self.device)
            if input_ids is None
            else input_ids.reshape(-1).to(self.device)
        )
        ranges = self._cache_ranges(
            step_start,
            step_end,
            from_tokens=input_ids is not None,
            fill_shallow=fill_shallow,
        )
        prefix = cache.get_seq_length(ranges[0][0])
        for offset, count in ranges:
            cache.assert_lengths(prefix, offset, offset + count)
        batch = make_forward_batch(self, cache, prefix, ids, positions, tree_mask)
        with forward_context(ForwardContext(attn_backend=batch.attn_backend)):
            state, head = self._execute_steps(
                positions,
                batch,
                step_start,
                step_end,
                ids if input_ids is not None else None,
                None if hidden_states is None else hidden_states.reshape(width, -1),
                fill_shallow,
            )
            logits = self.model.all_logits(head[-1:] if last_logits_only else head, batch)
        for offset, count in ranges:
            cache.lengths[offset : offset + count] = [prefix + width] * count
        return LoopedForwardOutput(state[None], logits[None])

    def prefill(
        self, input_ids, cache=None, *, fill_shallow=True, last_logits_only=False
    ):
        cache = self.new_cache() if cache is None else cache
        result = self.forward_steps(
            step_start=0,
            step_end=self.total_steps,
            cache=cache,
            position_ids=torch.arange(input_ids.numel(), device=self.device),
            input_ids=input_ids,
            fill_shallow=fill_shallow,
            last_logits_only=last_logits_only,
        )
        return cache, result

    def forward_shallow_tree(
        self, tree_tokens, tree_mask, tree_position_ids, prefix_length, cache
    ):
        if tree_tokens.numel() > self.max_tree_nodes:
            raise ValueError("The candidate tree exceeds its configured node budget")
        return self.forward_steps(
            step_start=0,
            step_end=self.shallow_steps,
            cache=cache,
            position_ids=tree_position_ids + prefix_length,
            input_ids=tree_tokens,
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
            position_ids=positions,
            hidden_states=shallow_hidden_states,
        )

    def forward_token_ids(self, input_ids, prefix_length, cache, *, fill_shallow=True):
        positions = torch.arange(
            prefix_length, prefix_length + input_ids.numel(), device=self.device
        )
        return self.forward_steps(
            step_start=0,
            step_end=self.total_steps,
            cache=cache,
            position_ids=positions,
            input_ids=input_ids,
            fill_shallow=fill_shallow,
        )

    def prefill_ar(self, input_ids, cache=None):
        return self.prefill(input_ids, cache=cache, fill_shallow=False)

    def forward_ar_token_ids(self, input_ids, prefix_length, cache):
        return self.forward_token_ids(
            input_ids, prefix_length, cache, fill_shallow=False
        )


class LayerSkipSGLangBackend(SGLangLoopedBackend):
    """Split the ordinary Llama layer stack without rerunning early layers."""

    @property
    def shallow_slot_end(self):
        return self.shallow_steps

    @property
    def num_cache_slots(self):
        return self.total_steps

    def prefill_ar(self, input_ids, cache=None):
        return self.prefill(
            input_ids,
            cache=cache,
            fill_shallow=False,
            last_logits_only=True,
        )

    def _cache_ranges(self, step_start, step_end, *, from_tokens, fill_shallow):
        return [(step_start, step_end - step_start)]

    def _execute_steps(
        self, positions, batch, step_start, step_end, ids, hidden, fill_shallow
    ):
        return self.model.forward_hidden_steps(
            positions,
            batch,
            step_start=step_start,
            step_end=step_end,
            input_ids=ids,
            hidden_states=hidden,
        )


class OuroSGLangBackend(SGLangLoopedBackend):
    """Ouro has one KV range for each recurrent iteration."""

    @property
    def shallow_slot_end(self):
        return self.shallow_steps * self.num_layers

    @property
    def num_cache_slots(self):
        return self.total_steps * self.num_layers

    def _cache_ranges(self, step_start, step_end, *, from_tokens, fill_shallow):
        return [
            (step_start * self.num_layers, (step_end - step_start) * self.num_layers)
        ]

    def _execute_steps(
        self, positions, batch, step_start, step_end, ids, hidden, fill_shallow
    ):
        state = self.model.forward_hidden_steps(
            positions,
            batch,
            step_start=step_start,
            step_end=step_end,
            input_ids=ids,
            hidden_states=hidden,
        )
        return state, state


class RavenSGLangBackend(SGLangLoopedBackend):
    """Prelude/core execution with independent early and final coda KV."""

    def __init__(
        self, runner, shallow_steps, context_length, max_tree_nodes, initial_seed=0
    ):
        super().__init__(
            runner, shallow_steps, context_length, max_tree_nodes, initial_seed
        )
        self.model.configure_split(shallow_steps, context_length, initial_seed)

    @property
    def shallow_slot_end(self):
        return self.model.shallow_slot_end

    @property
    def num_cache_slots(self):
        return self.model.end_layer

    def _cache_ranges(self, step_start, step_end, *, from_tokens, fill_shallow):
        model = self.model
        ranges = [(0, model.pre_count)] if from_tokens else []
        ranges.extend(
            (model.core_offset(step), self.num_layers)
            for step in range(step_start, step_end)
        )
        if step_start < self.shallow_steps < step_end and fill_shallow:
            ranges.append(
                (
                    model.pre_count + self.shallow_steps * self.num_layers,
                    model.post_count,
                )
            )
        post_offset = (
            model.pre_count + self.shallow_steps * self.num_layers
            if step_end == self.shallow_steps
            else model.pre_count + self.total_steps * self.num_layers + model.post_count
        )
        ranges.append((post_offset, model.post_count))
        return ranges

    def _execute_steps(
        self, positions, batch, step_start, step_end, ids, hidden, fill_shallow
    ):
        return self.model.execute_steps(
            positions,
            batch,
            step_start=step_start,
            step_end=step_end,
            input_ids=ids,
            packed_hidden=hidden,
            fill_shallow=fill_shallow,
        )
