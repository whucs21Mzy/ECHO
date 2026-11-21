"""Opt-in checks for the shared arithmetic used by the SGLang baseline."""

import os
import unittest
from unittest.mock import patch

import torch


@unittest.skipUnless(
    os.environ.get("ECHO_TEST_CUDA_OPS") == "1" and torch.cuda.is_available(),
    "set ECHO_TEST_CUDA_OPS=1 for GPU arithmetic checks",
)
class CommonPointwiseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

        enable_batch_invariant_mode()

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.batch_invariant_ops import disable_batch_invariant_mode

        disable_batch_invariant_mode()

    @torch.inference_mode()
    def test_rms_preserves_the_released_narrow_cast_boundary(self):
        from echo.sglang.ops.raven_norm import raven_rmsnorm

        for width in (512, 2048, 5280):
            for rows in (1, 3, 16, 65):
                x = torch.randn((rows, width), dtype=torch.bfloat16, device="cuda")
                weight = torch.randn(width, dtype=x.dtype, device=x.device)
                expected = (
                    x.float()
                    * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
                ).to(x.dtype) * weight
                torch.testing.assert_close(
                    raven_rmsnorm(x, weight, 1e-5), expected, atol=0, rtol=0
                )

    @torch.inference_mode()
    def test_huginn_and_olmo_rope_match_separate_torch_products(self):
        from echo.sglang.models.raven import (
            AdjacentRotaryEmbedding,
            OlmoRotaryEmbedding,
        )
        from echo.sglang.ops.raven_rope import huginn_rope, olmo_rope

        for dtype in (torch.bfloat16, torch.float32):
            positions = torch.tensor([2, 7, 13], device="cuda")
            packed = torch.randn((3, 4 * 96 + 2 * 2 * 96), device="cuda", dtype=dtype)
            q, k, _ = packed.split((4 * 96, 2 * 96, 2 * 96), -1)
            adjacent = AdjacentRotaryEmbedding(96, 256, 50000.0).to("cuda")
            half = OlmoRotaryEmbedding(96, 256, 50000.0).to("cuda")
            with patch.dict(os.environ, {"ECHO_RAVEN_FUSED_ROPE": "0"}):
                expected_h = adjacent(positions, q, k)
                expected_o = half(positions, q, k)
            actual_h = huginn_rope(positions, q, k, adjacent.cos_sin_cache, 96)
            actual_o = olmo_rope(positions, q, k, half.cos_sin_cache, 96)
            for actual, expected in zip(
                (*actual_h, *actual_o), (*expected_h, *expected_o)
            ):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            packed_equal = torch.randn((3, 3 * 4 * 96), device="cuda", dtype=dtype)
            qb, kb, _ = packed_equal.chunk(3, -1)
            bias = torch.randn((2, 1, 4, 96), device="cuda", dtype=dtype)
            with patch.dict(os.environ, {"ECHO_RAVEN_FUSED_ROPE": "0"}):
                expected_bias = adjacent(positions, qb, kb, bias)
            actual_bias = huginn_rope(
                positions, qb, kb, adjacent.cos_sin_cache, 96, bias
            )
            for actual, expected in zip(actual_bias, expected_bias):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @torch.inference_mode()
    def test_ouro_rope_retains_each_narrow_multiplication(self):
        from echo.sglang.ops.ouro_rope import ouro_neox_rope

        positions = torch.tensor([1, 3, 7], device="cuda")
        q = torch.randn((3, 4 * 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((3, 2 * 128), device="cuda", dtype=torch.bfloat16)
        trig = torch.randn((256, 128), device="cuda", dtype=torch.float32)
        cos, sin = trig[positions].to(q.dtype).chunk(2, -1)

        def reference(x):
            first, second = x.reshape(3, -1, 128).chunk(2, -1)
            return torch.cat(
                (
                    first * cos[:, None] - second * sin[:, None],
                    second * cos[:, None] + first * sin[:, None],
                ),
                -1,
            ).reshape(x.shape)

        actual = ouro_neox_rope(positions, q, k, trig, 128)
        for a, x in zip(actual, (q, k)):
            torch.testing.assert_close(a, reference(x), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
