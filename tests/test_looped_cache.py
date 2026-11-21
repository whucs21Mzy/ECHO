import unittest
from types import SimpleNamespace

import torch

from echo.model.looped_cache import LoopedEchoCache
from echo.sglang.cache import SGLangLoopedCache


class LoopedCacheTest(unittest.TestCase):
    def test_sibling_paths_compact_only_selected_roles_then_roll_back(self):
        cache = LoopedEchoCache(2)
        original = torch.arange(7).reshape(1, 1, 7, 1).float()
        cache.update(original, original + 100, 0)
        cache.update(original[:, :, :3], original[:, :, :3] + 100, 1)
        cache.compact_tree(3, torch.tensor([0, 2]), 0, 1)
        self.assertEqual(cache.key_cache[0].flatten().tolist(), [0, 1, 2, 3, 5])
        self.assertEqual(
            cache.value_cache[0].flatten().tolist(), [100, 101, 102, 103, 105]
        )
        self.assertEqual(cache.get_seq_length(1), 3)
        cache.truncate(3)
        cache.assert_lengths(3, 0, 2)

    def test_truncation_cannot_extend_a_populated_role(self):
        cache = LoopedEchoCache(1)
        values = torch.zeros((1, 1, 2, 1))
        cache.update(values, values, 0)
        with self.assertRaisesRegex(ValueError, "cannot extend"):
            cache.truncate(3)


class RequestPool:
    def __init__(self):
        self.available = [0]
        self.req_to_token = torch.zeros((1, 16), dtype=torch.int32)

    def alloc(self, requests):
        if not self.available:
            return None
        requests[0].req_pool_idx = self.available.pop()
        return [requests[0].req_pool_idx]

    def free(self, request):
        self.available.append(request.req_pool_idx)


class TokenPool:
    def __init__(self, capacity):
        self.free_tokens = set(range(capacity))

    def alloc(self, count):
        if count > len(self.free_tokens):
            return None
        values = sorted(self.free_tokens)[:count]
        self.free_tokens.difference_update(values)
        return torch.tensor(values)

    def free(self, values):
        tokens = set(values.tolist())
        if self.free_tokens & tokens:
            raise AssertionError("A KV slot was released twice")
        self.free_tokens.update(tokens)


class SGLangCacheOwnershipTest(unittest.TestCase):
    def backend(self, capacity=16):
        return SimpleNamespace(
            num_cache_slots=2,
            context_length=8,
            max_tree_nodes=4,
            runner=SimpleNamespace(
                req_to_token_pool=RequestPool(),
                token_to_kv_pool_allocator=TokenPool(capacity),
            ),
        )

    def test_compacted_allocation_is_released_once_and_can_be_reused(self):
        backend = self.backend()
        cache = SGLangLoopedCache(backend)
        cache.lengths[:] = [7, 3]
        cache.compact_tree(3, torch.tensor([0, 2]), 0, 1)
        self.assertEqual(cache.locations[:7].tolist(), [0, 1, 2, 3, 5, 4, 6])
        cache.assert_lengths(5, 0, 1)
        cache.assert_lengths(3, 1, 2)
        cache.close()
        cache.close()
        self.assertEqual(backend.runner.req_to_token_pool.available, [0])
        self.assertEqual(len(backend.runner.token_to_kv_pool_allocator.free_tokens), 16)
        reused = SGLangLoopedCache(backend)
        reused.close()

    def test_token_allocation_failure_releases_the_reserved_request(self):
        backend = self.backend(capacity=4)
        with self.assertRaisesRegex(MemoryError, "token pool"):
            SGLangLoopedCache(backend)
        self.assertEqual(backend.runner.req_to_token_pool.available, [0])


if __name__ == "__main__":
    unittest.main()
