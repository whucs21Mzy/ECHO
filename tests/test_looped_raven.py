"""Compare independent ECHO execution with each original released block layout."""

import os
import sys
import unittest
from pathlib import Path

import torch
from test_looped_ouro import DummyTokenizer, FakeAutomaton
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from echo.inference.looped_echo import EchoConfig, LoopedEchoGenerator
from echo.model.looped_raven import RavenLoopedEchoBackend

MODELS = Path(os.environ.get("ECHO_TEST_MODEL_ROOT", "checkpoints"))
FAMILIES = (
    "huginn-0125",
    "Raven-Llama-3.2",
    "Recurrent-OLMo-2-0425-train-recurrence-32",
    "Recurrent-TinyLlama-3T-train-recurrence-32",
)


def tiny_backend(name):
    path = str(MODELS / name)
    config = AutoConfig.from_pretrained(
        path, trust_remote_code=True, local_files_only=True
    )
    config.n_embd = 32
    config.intermediate_size = 64
    config.n_heads = 4
    config.num_key_value_heads = 4 if name == "huginn-0125" else 2
    config.head_dim = 8
    config.vocab_size = config.padded_vocab_size = 97
    config.n_layers_in_prelude = config.n_layers_in_coda = 1
    config.n_layers_in_recurrent_block = 2
    config.mean_recurrence = 3
    config.n_layers = 4
    config.block_size = config.max_position_embeddings = 128
    config.rope_scaling = None
    config.tie_embeddings = config.tie_word_embeddings = False
    cls = get_class_from_dynamic_module(
        "raven_modeling_minimal.RavenForCausalLM", path, local_files_only=True
    )
    model = cls(config).eval()
    return RavenLoopedEchoBackend(
        model, sys.modules[cls.__module__], shallow_steps=1, context_length=128
    )


@unittest.skipUnless(
    all((MODELS / name / "config.json").exists() for name in FAMILIES),
    "set ECHO_TEST_MODEL_ROOT for released-model CPU checks",
)
class RavenLoopedEchoTest(unittest.TestCase):
    @torch.inference_mode()
    def test_full_split_tree_and_compacted_continuation_match_released_model(self):
        for name in FAMILIES:
            with self.subTest(family=name):
                torch.manual_seed(7)
                backend = tiny_backend(name)
                tokens = torch.tensor([[2, 5, 7, 11, 17, 23]])
                positions = torch.arange(6).reshape(1, -1)
                native = backend.base_model(
                    tokens,
                    position_ids=positions,
                    input_states=backend.initial_hidden_by_position[:6][None],
                    num_steps=backend.total_steps,
                    use_cache=False,
                ).logits
                full_cache, full = backend.prefill(tokens)
                torch.testing.assert_close(full.logits, native, atol=2e-5, rtol=2e-5)
                full_cache.assert_lengths(6, 0, backend.num_cache_slots)
                cache, _ = backend.prefill(tokens[:, :3])
                tree_tokens = torch.tensor([11, 13, 17, 19, 23])
                tree_positions = torch.tensor([0, 1, 1, 2, 2])
                mask = torch.tensor(
                    [
                        [1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0],
                        [1, 1, 0, 1, 0],
                        [1, 0, 1, 0, 1],
                    ],
                    dtype=torch.bool,
                )
                tree = backend.forward_shallow_tree(
                    tree_tokens, mask, tree_positions, 3, cache
                )
                for indices in ([0, 1, 3], [0, 2, 4]):
                    linear_cache, _ = backend.prefill(tokens[:, :3])
                    linear = backend.forward_steps(
                        step_start=0,
                        step_end=1,
                        cache=linear_cache,
                        input_ids=tree_tokens[indices],
                        position_ids=torch.arange(3, 6),
                    )
                    torch.testing.assert_close(
                        tree.hidden_states[:, indices],
                        linear.hidden_states,
                        atol=2e-5,
                        rtol=2e-5,
                    )
                    torch.testing.assert_close(
                        tree.logits[:, indices], linear.logits, atol=2e-5, rtol=2e-5
                    )
                cache.compact_tree(
                    3, torch.tensor([0, 2, 4]), 0, backend.shallow_slot_end
                )
                continued = backend.forward_deep(
                    tree.hidden_states[:, [0, 2, 4]], 3, cache
                )
                torch.testing.assert_close(
                    continued.logits, native[:, 3:], atol=2e-5, rtol=2e-5
                )
                cache.assert_lengths(6, 0, backend.num_cache_slots)
                cache.truncate(4)
                cache.assert_lengths(4, 0, backend.num_cache_slots)

    @torch.inference_mode()
    def test_echo_greedy_ids_match_ar_and_ar_does_not_execute_q1_coda(self):
        from echo.inference.looped_ar import LoopedARGenerator

        for name in FAMILIES:
            with self.subTest(family=name):
                torch.manual_seed(17)
                backend = tiny_backend(name)
                ids = torch.tensor([[2, 5, 7]])
                ar = LoopedARGenerator(backend, DummyTokenizer()).generate(
                    ids, max_new_tokens=12
                )
                echo = LoopedEchoGenerator(
                    backend,
                    DummyTokenizer(),
                    FakeAutomaton(),
                    config=EchoConfig(
                        verification_threshold=4,
                        max_num_draft=8,
                        max_breadth=2,
                        ngram=2,
                    ),
                ).generate(ids, max_new_tokens=12)
                self.assertTrue(torch.equal(ar.sequences, echo.sequences))
                cache, _ = backend.prefill_ar(ids)
                q1_start = (
                    backend.pre_count + backend.shallow_steps * backend.num_layers
                )
                cache.assert_lengths(0, q1_start, backend.shallow_slot_end)
                self.assertEqual(cache.get_seq_length(backend.num_cache_slots - 1), 3)


if __name__ == "__main__":
    unittest.main()
