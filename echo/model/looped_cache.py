"""Tensor KV storage and forward results shared by recurrent model families."""

from dataclasses import dataclass
from typing import Optional

import torch


class LoopedEchoCache:
    """A manipulable dynamic cache with one slot per logical recurrent layer.

    Hugging Face's dynamic Ouro cache can append, but ECHO additionally needs to
    compact a verified tree path and roll back a rejected tentative suffix.
    Keeping those operations here avoids depending on private cache internals.
    """

    def __init__(self, num_slots: int):
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        self.key_cache: list[Optional[torch.Tensor]] = [None] * num_slots
        self.value_cache: list[Optional[torch.Tensor]] = [None] * num_slots

    def close(self) -> None:
        self.key_cache[:] = [None] * len(self.key_cache)
        self.value_cache[:] = [None] * len(self.value_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if layer_idx < 0 or layer_idx >= len(self.key_cache):
            raise IndexError(
                f"logical cache slot {layer_idx} is outside [0, {len(self.key_cache)})"
            )
        old_key = self.key_cache[layer_idx]
        old_value = self.value_cache[layer_idx]
        if old_key is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            if old_value is None:
                raise RuntimeError("key/value cache became inconsistent")
            self.key_cache[layer_idx] = torch.cat((old_key, key_states), dim=2)
            self.value_cache[layer_idx] = torch.cat((old_value, value_states), dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]  # type: ignore[return-value]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        entry = self.key_cache[layer_idx]
        return 0 if entry is None else int(entry.shape[2])

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        del new_seq_length
        return self.get_seq_length(layer_idx)

    def get_max_length(self) -> None:
        return None

    @property
    def is_compileable(self) -> bool:
        return False

    def truncate(
        self,
        length: int,
        slot_start: int = 0,
        slot_end: Optional[int] = None,
    ) -> None:
        """Drop cache entries after ``length`` in the selected logical slots."""
        if length < 0:
            raise ValueError("cache length must be non-negative")
        end = len(self.key_cache) if slot_end is None else slot_end
        for slot in range(slot_start, end):
            key = self.key_cache[slot]
            value = self.value_cache[slot]
            if key is None:
                continue
            if value is None:
                raise RuntimeError("key/value cache became inconsistent")
            if length > key.shape[2]:
                raise ValueError(
                    f"cannot extend slot {slot} from {key.shape[2]} to {length} by truncation"
                )
            self.key_cache[slot] = key[:, :, :length, :].contiguous()
            self.value_cache[slot] = value[:, :, :length, :].contiguous()

    def compact_tree(
        self,
        prefix_length: int,
        node_indices: torch.Tensor,
        slot_start: int,
        slot_end: int,
    ) -> None:
        """Replace a flattened tree suffix with one root-to-leaf path.

        ``node_indices`` are zero-based within the just-appended tree.  The
        prefix remains in place and selected nodes become a contiguous suffix.
        """
        node_indices = node_indices.to(dtype=torch.long).reshape(-1)
        for slot in range(slot_start, slot_end):
            key = self.key_cache[slot]
            value = self.value_cache[slot]
            if key is None or value is None:
                raise RuntimeError(
                    f"logical slot {slot} was not populated by tree forward"
                )
            src = node_indices.to(key.device) + prefix_length
            prefix_key = key[:, :, :prefix_length, :]
            prefix_value = value[:, :, :prefix_length, :]
            path_key = key.index_select(2, src)
            path_value = value.index_select(2, src)
            self.key_cache[slot] = torch.cat((prefix_key, path_key), dim=2)
            self.value_cache[slot] = torch.cat((prefix_value, path_value), dim=2)

    def assert_lengths(self, expected: int, slot_start: int, slot_end: int) -> None:
        lengths = {self.get_seq_length(slot) for slot in range(slot_start, slot_end)}
        if lengths != {expected}:
            raise RuntimeError(
                f"logical cache slots [{slot_start}, {slot_end}) have lengths "
                f"{sorted(lengths)}, expected {expected}"
            )


@dataclass
class LoopedForwardOutput:
    hidden_states: torch.Tensor
    logits: torch.Tensor
