"""Residual continuity at LayerSkip's early/final verification boundary."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from echo.sglang.backend import LayerSkipSGLangBackend


class LayerSkipDispatchTest(unittest.TestCase):
    def test_early_and_deep_ranges_cover_each_kv_layer_once(self):
        model = SimpleNamespace(config=None, total_steps=32, num_layers=1)
        backend = LayerSkipSGLangBackend(SimpleNamespace(model=model), 10, 64, 8)
        self.assertEqual(backend.shallow_slot_end, 10)
        self.assertEqual(backend.num_cache_slots, 32)
        self.assertEqual(
            backend._cache_ranges(0, 10, from_tokens=True, fill_shallow=True),
            [(0, 10)],
        )
        self.assertEqual(
            backend._cache_ranges(10, 32, from_tokens=False, fill_shallow=True),
            [(10, 22)],
        )

    def test_invalid_early_exit_fails_before_allocating_a_runner(self):
        import json

        from echo.model.looped_backend import load_looped_backend

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "config.json").write_text(
                json.dumps(dict(model_type="llama", num_hidden_layers=4))
            )
            with patch.object(LayerSkipSGLangBackend, "from_pretrained") as load:
                for layer in (0, 4, 5):
                    with self.subTest(layer=layer), self.assertRaises(ValueError):
                        load_looped_backend(path, backend="sglang", shallow_steps=layer)
                load.assert_not_called()


class MutatingNorm(nn.Module):
    """Exercise the in-place contract of native fused add/RMSNorm."""

    def forward(self, hidden, residual):
        residual.add_(hidden)
        value = residual.float()
        hidden.copy_(value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5))
        return hidden, residual


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)
        self.calls = 0

    def forward(self, positions, hidden, batch, residual):
        if not hidden.is_contiguous() or (
            residual is not None and not residual.is_contiguous()
        ):
            raise AssertionError("Native kernels need contiguous residual streams")
        self.calls += 1
        residual = hidden if residual is None else residual + hidden
        return self.proj(torch.tanh(residual)), residual


@unittest.skipUnless(importlib.util.find_spec("sglang"), "requires the SGLang environment")
class LayerSkipResidualTest(unittest.TestCase):
    @torch.inference_mode()
    def test_early_readout_preserves_raw_states_and_never_repeats_layers(self):
        from echo.sglang.models.llama import EchoLlamaForCausalLM

        for dtype, width in (
            (torch.float32, 1), (torch.float32, 4),
            (torch.bfloat16, 1), (torch.bfloat16, 4),
        ):
            with self.subTest(dtype=dtype, width=width):
                torch.manual_seed(7)
                model = nn.Module()
                model.embed_tokens = nn.Embedding(32, 16)
                model.layers = nn.ModuleList(ResidualBlock(16) for _ in range(4))
                model.norm = MutatingNorm()
                model.to(dtype=dtype)
                backend_model = SimpleNamespace(
                    model=model, config=SimpleNamespace(hidden_size=16), total_steps=4
                )
                execute = EchoLlamaForCausalLM.forward_hidden_steps
                ids = torch.tensor([2, 3, 5, 7])[:width]
                positions = torch.arange(ids.numel())

                # Independent uninterrupted execution, with no intermediate head.
                hidden, residual = model.embed_tokens(ids), None
                for layer in model.layers:
                    hidden, residual = layer(positions, hidden, None, residual)
                expected, _ = model.norm(hidden, residual)
                for layer in model.layers:
                    layer.calls = 0

                state, _ = execute(
                    backend_model, positions, None,
                    step_start=0, step_end=2, input_ids=ids,
                )
                saved = state.clone()
                _, actual = execute(
                    backend_model, positions, None,
                    step_start=2, step_end=4, hidden_states=state,
                )
                torch.testing.assert_close(state, saved, atol=0, rtol=0)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertEqual([layer.calls for layer in model.layers], [1] * 4)

    def test_normalized_readout_cannot_be_used_as_continuation(self):
        from echo.sglang.models.llama import EchoLlamaForCausalLM

        model = SimpleNamespace(config=SimpleNamespace(hidden_size=16))
        with self.assertRaisesRegex(ValueError, "both residual streams"):
            EchoLlamaForCausalLM.forward_hidden_steps(
                model, torch.arange(2), None,
                step_start=2, step_end=4, hidden_states=torch.zeros(2, 16),
            )


if __name__ == "__main__":
    unittest.main()
