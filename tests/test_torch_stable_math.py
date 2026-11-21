"""Arithmetic must stay unchanged when a query acquires unrelated tree rows."""

import os
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(
    0,
    os.environ.get("ECHO_TEST_BACKEND_ROOT", str(Path(__file__).resolve().parents[1])),
)


@unittest.skipUnless(
    os.environ.get("ECHO_TEST_CUDA_OPS") == "1" and torch.cuda.is_available(),
    "opt-in CUDA arithmetic tests",
)
class StableMathTests(unittest.TestCase):
    def setUp(self):
        from echo.model import stable_math

        self.ops = stable_math
        torch.manual_seed(17)

    @torch.inference_mode()
    def test_linear_query_count_and_fp64_reference(self):
        x = torch.randn((19, 128), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((160, 128), device="cuda", dtype=torch.bfloat16)
        bias = torch.randn((160,), device="cuda", dtype=torch.bfloat16)
        output = self.ops.linear(x, weight, bias)
        one = torch.cat([self.ops.linear(row[None], weight, bias) for row in x])
        self.assertTrue(torch.equal(output, one))
        reference = torch.nn.functional.linear(
            x.double(), weight.double(), bias.double()
        ).bfloat16()
        torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)

    @torch.inference_mode()
    def test_tree_attention_equals_each_compacted_prefix(self):
        q = torch.randn((1, 4, 3, 96), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((1, 2, 9, 96), device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        paths = [[0, 1, 3, 4], [0, 1, 3, 7], [0, 1, 3, 4, 8]]
        mask = torch.zeros((1, 1, 3, 9), device="cuda", dtype=torch.bool)
        for i, path in enumerate(paths):
            mask[0, 0, i, path] = True
        output = self.ops.scaled_dot_product_attention(q, k, v, mask, enable_gqa=True)
        for i, path in enumerate(paths):
            indices = torch.tensor(path, device="cuda")
            compact_k, compact_v = (
                k.index_select(2, indices),
                v.index_select(2, indices),
            )
            one = self.ops.scaled_dot_product_attention(
                q[:, :, i : i + 1], compact_k, compact_v, enable_gqa=True
            )
            self.assertTrue(torch.equal(output[:, :, i : i + 1], one))
            repeated_k, repeated_v = (
                compact_k.double().repeat_interleave(2, dim=1),
                compact_v.double().repeat_interleave(2, dim=1),
            )
            scores = (q[:, :, i : i + 1].double() @ repeated_k.transpose(-1, -2)) * (
                96**-0.5
            )
            reference = (scores.softmax(-1) @ repeated_v).bfloat16()
            torch.testing.assert_close(one, reference, atol=0.01, rtol=0.01)

    @torch.inference_mode()
    def test_causal_mask_and_unrelated_tail(self):
        q = torch.randn((1, 2, 17, 64), device="cuda", dtype=torch.bfloat16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        causal = torch.ones((17, 17), device="cuda", dtype=torch.bool).tril()
        explicit = self.ops.scaled_dot_product_attention(q, k, v, causal)
        implicit = self.ops.scaled_dot_product_attention(q, k, v, is_causal=True)
        self.assertTrue(torch.equal(explicit, implicit))
        for i in (0, 4, 15, 16):
            compact = self.ops.scaled_dot_product_attention(
                q[:, :, i : i + 1], k[:, :, : i + 1], v[:, :, : i + 1]
            )
            self.assertTrue(torch.equal(explicit[:, :, i : i + 1], compact))

    @torch.inference_mode()
    def test_norm_query_count_and_narrow_weight_boundary(self):
        for width in (2048, 5280):
            x = torch.randn((65, width), device="cuda", dtype=torch.bfloat16)
            weight = torch.randn((width,), device="cuda", dtype=torch.bfloat16)
            output = self.ops.rmsnorm(x, weight, 1e-5)
            for row in (0, 17, 63, 64):
                one = self.ops.rmsnorm(x[row : row + 1], weight, 1e-5)
                self.assertTrue(torch.equal(output[row : row + 1], one))
            reference = (
                x.double()
                * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-5)
            ).bfloat16() * weight
            torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)


@unittest.skipUnless(
    os.environ.get("ECHO_TEST_TORCH_MODEL") and torch.cuda.is_available(),
    "opt-in real-checkpoint tree test",
)
class TorchCheckpointTreeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from echo.model.looped_backend import load_looped_backend

        cls.backend = load_looped_backend(
            os.environ["ECHO_TEST_TORCH_MODEL"],
            backend="torch",
            shallow_steps=int(os.environ["ECHO_TEST_SHALLOW_STEPS"]),
            dtype=torch.bfloat16,
            context_length=256,
            max_tree_nodes=16,
            deterministic=True,
        )

    @torch.inference_mode()
    def test_both_branches_compaction_and_rollback_equal_full_causal_forward(self):
        backend = self.backend
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
        tree = backend.forward_shallow_tree(tokens, mask, positions, 3, cache)
        for path, nodes in (([11, 13, 19], [0, 1, 3]), ([11, 17, 23], [0, 2, 4])):
            reference_cache, _ = backend.prefill(prefix)
            reference = backend.forward_steps(
                step_start=0,
                step_end=backend.shallow_steps,
                cache=reference_cache,
                position_ids=torch.arange(3, 6, device=backend.device),
                input_ids=torch.tensor([path], device=backend.device),
            )
            torch.testing.assert_close(
                tree.hidden_states[:, nodes], reference.hidden_states, atol=0, rtol=0
            )
            torch.testing.assert_close(
                tree.logits[:, nodes], reference.logits, atol=0, rtol=0
            )
            del reference_cache
        selected = torch.tensor([0, 2, 4], device=backend.device)
        cache.compact_tree(3, selected, 0, backend.shallow_slot_end)
        cache.assert_lengths(6, 0, backend.shallow_slot_end)
        cache.assert_lengths(3, backend.shallow_slot_end, backend.num_cache_slots)
        continued = backend.forward_deep(tree.hidden_states[:, selected], 3, cache)
        full_cache, full = backend.prefill(
            torch.tensor([[2, 5, 7, 11, 17, 23]], device=backend.device)
        )
        torch.testing.assert_close(
            continued.hidden_states, full.hidden_states[:, 3:], atol=0, rtol=0
        )
        torch.testing.assert_close(continued.logits, full.logits[:, 3:], atol=0, rtol=0)
        cache.assert_lengths(6, 0, backend.num_cache_slots)
        del full_cache
        cache.truncate(4)
        cache.assert_lengths(4, 0, backend.num_cache_slots)
        corrected = backend.forward_token_ids(
            torch.tensor([[29]], device=backend.device), 4, cache
        )
        full_cache, full = backend.prefill(
            torch.tensor([[2, 5, 7, 11, 29]], device=backend.device)
        )
        torch.testing.assert_close(
            corrected.logits, full.logits[:, -1:], atol=0, rtol=0
        )
        cache.assert_lengths(5, 0, backend.num_cache_slots)


if __name__ == "__main__":
    unittest.main()
