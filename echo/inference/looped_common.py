"""Request validation, timing and results shared by recurrent AR and ECHO."""

import time
from dataclasses import dataclass, field

import torch


@dataclass
class GenerationStats:
    outer_steps: int = 0
    deep_calls: int = 0
    full_token_calls: int = 0
    deferred_corrections: int = 0
    draft_trees: int = 0
    drafted_nodes: int = 0
    shallow_accepted: int = 0
    target_accepted: int = 0
    corrections: int = 0
    target_accept_lengths: list[int] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)

    @property
    def mean_target_accept(self) -> float:
        lengths = self.target_accept_lengths
        return sum(lengths) / len(lengths) if lengths else 0.0


@dataclass
class GenerationResult:
    sequences: torch.Tensor
    stats: GenerationStats
    finish_reason: dict


def prepare_input(input_ids, backend, max_new_tokens):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("Provide one nonempty tokenized prompt with shape [1, length]")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if input_ids.shape[1] + max_new_tokens > backend.context_length:
        raise ValueError("Prompt and generation limit exceed context_length")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Token IDs must have an integer dtype")
    if input_ids.min().item() < 0 or input_ids.max().item() >= backend.vocab_size:
        raise ValueError("Token IDs are outside the target vocabulary")
    return input_ids.to(device=backend.device, dtype=torch.long)


def synchronized_time(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()
