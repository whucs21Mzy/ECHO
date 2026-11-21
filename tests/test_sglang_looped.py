"""Opt-in native SGLang integration tests; no GPU is loaded during CPU discovery."""

import os
import unittest

import torch


@unittest.skipUnless(
    os.environ.get("ECHO_TEST_SGLANG_MODEL"),
    "set ECHO_TEST_SGLANG_MODEL to run GPU integration tests",
)
class SGLangLoopedEchoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from echo.model.looped_backend import load_looped_backend

        cls.backend = load_looped_backend(
            os.environ["ECHO_TEST_SGLANG_MODEL"],
            backend="sglang",
            shallow_steps=int(os.environ.get("ECHO_TEST_SHALLOW_STEPS", "1")),
            dtype=torch.float32,
            context_length=256,
            max_tree_nodes=16,
        )

    @classmethod
    def tearDownClass(cls):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    @torch.inference_mode()
    def test_tree_branches_compaction_and_deep_continuation_match_causal_model(self):
        backend = self.backend
        pool = backend.runner.req_to_token_pool
        allocator = backend.runner.token_to_kv_pool_allocator
        initial = (pool.available_size(), allocator.available_size())
        prefix = torch.tensor([[2, 5, 7]], device=backend.device)
        tokens = torch.tensor([11, 13, 17, 19, 23], device=backend.device)
        positions = torch.tensor([0, 1, 1, 2, 2], device=backend.device)
        mask = torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [1, 1, 0, 1, 0],
                [1, 0, 1, 0, 1],
            ],
            dtype=torch.bool,
            device=backend.device,
        )
        cache, _ = backend.prefill(prefix)
        try:
            tree = backend.forward_shallow_tree(tokens, mask, positions, 3, cache)
            for path, nodes in (([11, 13, 19], [0, 1, 3]), ([11, 17, 23], [0, 2, 4])):
                reference_cache, _ = backend.prefill(prefix)
                try:
                    reference = backend.forward_steps(
                        step_start=0,
                        step_end=backend.shallow_steps,
                        cache=reference_cache,
                        position_ids=torch.arange(3, 6, device=backend.device),
                        input_ids=torch.tensor([path], device=backend.device),
                    )
                    torch.testing.assert_close(
                        tree.hidden_states[:, nodes],
                        reference.hidden_states,
                        atol=1e-4,
                        rtol=1e-4,
                    )
                    torch.testing.assert_close(
                        tree.logits[:, nodes], reference.logits, atol=1e-3, rtol=1e-3
                    )
                finally:
                    reference_cache.close()
            cache.compact_tree(
                3,
                torch.tensor([0, 2, 4], device=backend.device),
                0,
                backend.shallow_slot_end,
            )
            cache.assert_lengths(6, 0, backend.shallow_slot_end)
            cache.assert_lengths(3, backend.shallow_slot_end, backend.num_cache_slots)
            continued = backend.forward_deep(tree.hidden_states[:, [0, 2, 4]], 3, cache)
            reference_cache, _ = backend.prefill(prefix)
            try:
                full = backend.forward_token_ids(
                    torch.tensor([[11, 17, 23]], device=backend.device),
                    3,
                    reference_cache,
                )
                torch.testing.assert_close(
                    continued.logits, full.logits, atol=1e-3, rtol=1e-3
                )
                cache.assert_lengths(6, 0, backend.num_cache_slots)
                cache.truncate(4)
                cache.assert_lengths(4, 0, backend.num_cache_slots)
            finally:
                reference_cache.close()
        finally:
            cache.close()
        self.assertEqual(initial, (pool.available_size(), allocator.available_size()))


if __name__ == "__main__":
    unittest.main()
