"""Greedy AR over the same full recurrent backend used by ECHO."""

from contextlib import closing

import torch

from .looped_common import (
    GenerationResult,
    GenerationStats,
    prepare_input,
    synchronized_time,
)
from .looped_stop import GenerationStops


class LoopedARGenerator:
    def __init__(self, backend, tokenizer):
        self.backend = backend
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        *,
        max_new_tokens=128,
        stop_token_sequences=(),
        eos_token_ids=None,
        stop_strings=(),
        include_stop_tokens=False,
    ) -> GenerationResult:
        input_ids = prepare_input(input_ids, self.backend, max_new_tokens)
        prompt_length = input_ids.shape[1]
        stops = GenerationStops(
            self.tokenizer,
            eos_token_ids=eos_token_ids,
            token_sequences=stop_token_sequences,
            strings=stop_strings,
        )
        finish_reason = dict(type="length", length=max_new_tokens)
        prefill_started = synchronized_time(self.backend.device)
        with closing(self.backend.new_cache()) as cache:
            _, output = self.backend.prefill_ar(input_ids, cache=cache)
            prefill_seconds = synchronized_time(self.backend.device) - prefill_started
            committed = input_ids.clone()
            logits = output.logits[:, -1:, :]
            decode_started = synchronized_time(self.backend.device)
            for position in range(max_new_tokens):
                token = logits[:, -1, :].argmax(dim=-1)
                prefix_length = committed.shape[1]
                committed = torch.cat((committed, token[:, None]), dim=1)
                stop = stops.observe(committed[0, prompt_length:])
                if stop is not None:
                    keep = (
                        stop.generated_count
                        if include_stop_tokens
                        else stop.visible_count
                    )
                    committed = committed[:, : prompt_length + keep]
                    finish_reason = stop.reason
                    break
                if position + 1 < max_new_tokens:
                    output = self.backend.forward_ar_token_ids(
                        token, prefix_length, cache
                    )
                    logits = output.logits[:, -1:, :]
            decode_seconds = synchronized_time(self.backend.device) - decode_started

        count = committed.shape[1] - prompt_length
        stats = GenerationStats(
            outer_steps=count,
            target_accepted=count,
            target_accept_lengths=[1] * count,
            stage_timings=dict(
                prefill_seconds=prefill_seconds, decode_seconds=decode_seconds
            ),
        )
        return GenerationResult(committed, stats, finish_reason)
