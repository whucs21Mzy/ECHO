"""Request-scoped paged KV allocation and recurrent-layer cache lengths."""

import weakref
from types import SimpleNamespace

import torch


def _release_cache(allocator, request_pool, allocated, request):
    allocator.free(allocated)
    request_pool.free(request)


class SGLangLoopedCache:
    """Logical UT lengths over a request's native SGLang physical token slots.

    Tree compaction permutes free-suffix addresses. It never copies KV over an
    overlapping source, and never changes the committed deep prefix.
    """

    def __init__(self, backend):
        self.backend = backend
        self.lengths = [0] * backend.num_cache_slots
        runner = backend.runner
        self.request = SimpleNamespace(req_pool_idx=None)
        rows = runner.req_to_token_pool.alloc([self.request])
        if rows is None:
            raise MemoryError("SGLang request pool is exhausted")
        self.request_index = rows[0]
        capacity = backend.context_length + backend.max_tree_nodes
        allocated = runner.token_to_kv_pool_allocator.alloc(capacity)
        if allocated is None:
            runner.req_to_token_pool.free(self.request)
            raise MemoryError("SGLang token pool is exhausted")
        self.locations = allocated.clone()
        self._finalizer = weakref.finalize(
            self,
            _release_cache,
            runner.token_to_kv_pool_allocator,
            runner.req_to_token_pool,
            allocated,
            self.request,
        )
        self._write_mapping(0, self.locations.numel())

    def close(self):
        self._finalizer()

    def _write_mapping(self, start, end):
        table = self.backend.runner.req_to_token_pool.req_to_token
        table[self.request_index, start:end].copy_(self.locations[start:end])

    def get_seq_length(self, layer_idx=0):
        return self.lengths[layer_idx]

    def truncate(self, length, slot_start=0, slot_end=None):
        end = len(self.lengths) if slot_end is None else slot_end
        if length < 0 or any(length > value for value in self.lengths[slot_start:end]):
            raise ValueError("Cache truncation cannot extend a logical UT layer")
        self.lengths[slot_start:end] = [length] * (end - slot_start)

    def compact_tree(self, prefix_length, node_indices, slot_start, slot_end):
        lengths = set(self.lengths[slot_start:slot_end])
        if len(lengths) != 1:
            raise RuntimeError("Tree layers have inconsistent cache lengths")
        width = lengths.pop() - prefix_length
        if any(
            value > prefix_length
            for i, value in enumerate(self.lengths)
            if not slot_start <= i < slot_end
        ):
            raise RuntimeError(
                "Cannot permute a suffix already committed by another UT range"
            )
        selected = node_indices.to(
            device=self.locations.device, dtype=torch.long
        ).reshape(-1)
        if (
            selected.numel() == 0
            or int(selected.min()) < 0
            or int(selected.max()) >= width
        ):
            raise ValueError("Invalid accepted tree path")
        if selected.unique().numel() != selected.numel():
            raise ValueError("An accepted path cannot repeat a tree node")
        segment = self.locations[prefix_length : prefix_length + width].clone()
        remaining = torch.ones(width, dtype=torch.bool, device=segment.device)
        remaining[selected] = False
        self.locations[prefix_length : prefix_length + width] = torch.cat(
            (segment.index_select(0, selected), segment[remaining])
        )
        self._write_mapping(prefix_length, prefix_length + width)
        accepted_length = prefix_length + selected.numel()
        self.lengths[slot_start:slot_end] = [accepted_length] * (slot_end - slot_start)

    def assert_lengths(self, expected, slot_start, slot_end):
        if set(self.lengths[slot_start:slot_end]) != {expected}:
            raise RuntimeError("Logical UT cache length invariant failed")
