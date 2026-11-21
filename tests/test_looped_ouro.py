import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from echo.inference.looped_echo import (
    EchoConfig,
    LoopedEchoGenerator,
    build_draft_tree,
)
from echo.model.looped_ouro import LoopedForwardOutput, OuroLoopedEchoBackend

MODEL_PATH = os.environ.get("ECHO_TEST_OURO_MODEL", "")


def tiny_ouro():
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config.pad_token_id = 0
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.head_dim = 8
    config.vocab_size = 97
    config.total_ut_steps = 3
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    config.max_position_embeddings = 128
    config._attn_implementation = "eager"
    config.use_cache = False
    return AutoModelForCausalLM.from_config(config, trust_remote_code=True).eval()


class FakeAutomaton:
    """Small adjacency-backed two-branch trie for an end-to-end CPU test."""

    def __init__(self):
        self.adjacency = {}

    def reset_to_root(self):
        pass

    def insert(self, pattern):
        pass

    def build(self):
        pass

    def trans_tokens(self, tokens):
        pass

    def update(self, tokens, adjacency):
        for token, children in zip(tokens, adjacency):
            self.adjacency[int(token)] = [int(child) for child in children]

    def retrieve(self, root, max_num_draft):
        del max_num_draft
        children = self.adjacency.get(int(root), [0, 1])[:2]
        if len(children) < 2:
            children = children + [0] * (2 - len(children))
        return SimpleNamespace(
            tree_candidates=[int(root), *children],
            position_ids=[0, 1, 1],
            candidates=[[int(root), children[0]], [int(root), children[1]]],
            attn_mask=[[1, 0, 0], [1, 1, 0], [1, 0, 1]],
            retrieve_indices=[[0, 1], [0, 2]],
        )


class DummyTokenizer:
    eos_token_id = None
    pad_token_id = 0


class DraftTreeTest(unittest.TestCase):
    def test_last_position_tree_is_root_only(self):
        tree = build_draft_tree(
            torch.tensor([[0.0, 2.0, 1.0]]),
            FakeAutomaton(),
            pad_token_id=0,
            max_num_draft=8,
            device=torch.device("cpu"),
            max_depth=1,
        )
        self.assertEqual(tree.tree_tokens.tolist(), [1])
        self.assertEqual(tree.position_ids.tolist(), [0])
        self.assertEqual(tree.retrieve_indices.tolist(), [[0]])


class GreedyTerminalTest(unittest.TestCase):
    def test_invalid_token_ids_fail_before_prefill(self):
        from echo.inference.looped_ar import LoopedARGenerator

        for ids in (
            torch.tensor([[1.5]]),
            torch.tensor([[-1]]),
            torch.tensor([[97]]),
        ):
            for method in ("ar", "echo"):
                with self.subTest(ids=ids, method=method):
                    backend = SimpleNamespace(
                        device=torch.device("cpu"),
                        context_length=128,
                        vocab_size=97,
                        new_cache=Mock(),
                    )
                    engine = (
                        LoopedARGenerator(backend, DummyTokenizer())
                        if method == "ar"
                        else LoopedEchoGenerator(
                            backend, DummyTokenizer(), FakeAutomaton()
                        )
                    )
                    with self.assertRaises(ValueError):
                        engine.generate(ids, max_new_tokens=4)
                    backend.new_cache.assert_not_called()

    def test_failed_prefill_releases_the_request_cache_in_both_methods(self):
        from echo.inference.looped_ar import LoopedARGenerator

        for method in ("ar", "echo"):
            with self.subTest(method=method):
                cache = Mock()
                backend = SimpleNamespace(
                    device=torch.device("cpu"),
                    context_length=128,
                    vocab_size=97,
                    new_cache=Mock(return_value=cache),
                    prefill=Mock(side_effect=RuntimeError("prefill failed")),
                    prefill_ar=Mock(side_effect=RuntimeError("prefill failed")),
                )
                engine = (
                    LoopedARGenerator(backend, DummyTokenizer())
                    if method == "ar"
                    else LoopedEchoGenerator(backend, DummyTokenizer(), FakeAutomaton())
                )
                with self.assertRaisesRegex(RuntimeError, "prefill failed"):
                    engine.generate(torch.tensor([[2]]), max_new_tokens=4)
                cache.close.assert_called_once_with()

    def test_terminal_tokens_do_not_trigger_unused_target_forward(self):
        from echo.inference.looped_ar import LoopedARGenerator

        class CountingBackend:
            device = torch.device("cpu")
            context_length = 128
            vocab_size = 3

            def __init__(self):
                self.calls = 0

            def new_cache(self):
                return SimpleNamespace(close=lambda: None)

            def prefill_ar(self, ids, cache=None):
                return cache, SimpleNamespace(logits=torch.tensor([[[0.0, 2.0, 1.0]]]))

            def forward_ar_token_ids(self, ids, prefix, cache):
                self.calls += 1
                return SimpleNamespace(logits=torch.tensor([[[0.0, 2.0, 1.0]]]))

        prompt = torch.tensor([[2]])
        for eos, limit, stops, output_length, calls in (
            (1, 8, (), 1, 0),
            (None, 1, (), 1, 0),
            (None, 3, (), 3, 2),
            (None, 8, ((1,),), 0, 0),
        ):
            with self.subTest(eos=eos, limit=limit, stops=stops):
                backend = CountingBackend()
                result = LoopedARGenerator(
                    backend, SimpleNamespace(eos_token_id=eos)
                ).generate(
                    prompt,
                    max_new_tokens=limit,
                    stop_token_sequences=stops,
                )
                self.assertEqual(
                    result.sequences.shape[1] - prompt.shape[1], output_length
                )
                self.assertEqual(backend.calls, calls)


@unittest.skipUnless(
    MODEL_PATH and os.path.isdir(MODEL_PATH),
    "set ECHO_TEST_OURO_MODEL for released-model CPU checks",
)
class OuroLoopedEchoBackendTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = tiny_ouro()
        self.backend = OuroLoopedEchoBackend(self.model, shallow_steps=1)

    def test_stable_kernels_are_scoped_to_the_loaded_model(self):
        from echo.model.stable_math import install

        native_attention = torch.nn.functional.scaled_dot_product_attention
        native_model = tiny_ouro()
        native_implementation = native_model.config._attn_implementation
        install(self.model)
        self.assertIs(
            torch.nn.functional.scaled_dot_product_attention, native_attention
        )
        self.assertEqual(
            native_model.config._attn_implementation, native_implementation
        )
        self.assertEqual(self.model.config._attn_implementation, "echo_row_invariant")

    def test_prefill_and_incremental_match_native_forward(self):
        ids = torch.tensor([[2, 5, 7, 11, 13]])
        with torch.inference_mode():
            native = self.model(ids, use_cache=False, exit_threshold=None).logits
        cache, prefill = self.backend.prefill(ids)
        torch.testing.assert_close(prefill.logits, native, atol=0, rtol=0)
        cache.assert_lengths(5, 0, self.backend.num_cache_slots)

        prefix = ids[:, :3]
        suffix = ids[:, 3:]
        cache, _ = self.backend.prefill(prefix)
        incremental = self.backend.forward_token_ids(suffix, 3, cache)
        torch.testing.assert_close(incremental.logits, native[:, -2:], atol=0, rtol=0)

    def test_recurrent_split_matches_full_execution(self):
        ids = torch.tensor([[2, 5, 7, 11, 13]])
        with torch.inference_mode():
            native = self.model(ids, use_cache=False, exit_threshold=None).logits
        cache, _ = self.backend.prefill(ids[:, :3])
        positions = torch.arange(3, 5).unsqueeze(0)
        shallow = self.backend.forward_steps(
            input_ids=ids[:, 3:],
            step_start=0,
            step_end=1,
            cache=cache,
            position_ids=positions,
        )
        deep = self.backend.forward_deep(shallow.hidden_states, 3, cache)
        torch.testing.assert_close(deep.logits, native[:, -2:], atol=0, rtol=0)

    def test_tree_attention_matches_each_linear_branch(self):
        prefix = torch.tensor([[2, 5, 7]])
        tree_tokens = torch.tensor([11, 13, 17, 19, 23])
        tree_positions = torch.tensor([0, 1, 1, 2, 2])
        tree_mask = torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [1, 1, 0, 1, 0],
                [1, 0, 1, 0, 1],
            ],
            dtype=torch.bool,
        )
        tree_cache, _ = self.backend.prefill(prefix)
        tree = self.backend.forward_shallow_tree(
            tree_tokens, tree_mask, tree_positions, 3, tree_cache
        )

        for path, node_indices in (
            ([11, 13, 19], [0, 1, 3]),
            ([11, 17, 23], [0, 2, 4]),
        ):
            branch_cache, _ = self.backend.prefill(prefix)
            branch = self.backend.forward_steps(
                input_ids=torch.tensor([path]),
                step_start=0,
                step_end=1,
                cache=branch_cache,
                position_ids=torch.arange(3, 6).unsqueeze(0),
            )
            torch.testing.assert_close(
                tree.hidden_states[:, node_indices],
                branch.hidden_states,
                atol=0,
                rtol=0,
            )

        tree_cache.compact_tree(
            3,
            torch.tensor([0, 2, 4]),
            0,
            self.backend.shallow_slot_end,
        )
        tree_cache.assert_lengths(6, 0, self.backend.shallow_slot_end)
        tree_cache.assert_lengths(
            3, self.backend.shallow_slot_end, self.backend.num_cache_slots
        )

    def test_end_to_end_echo_is_lossless_against_greedy_target(self):
        prompt = torch.tensor([[2, 5, 7]])
        baseline_cache, baseline_prefill = self.backend.prefill(prompt)
        baseline = prompt.clone()
        logits = baseline_prefill.logits[:, -1:, :]
        for _ in range(12):
            token = logits[:, -1, :].argmax(dim=-1)
            output = self.backend.forward_token_ids(
                token, baseline.shape[1], baseline_cache
            )
            baseline = torch.cat((baseline, token[:, None]), dim=1)
            logits = output.logits[:, -1:, :]

        result = LoopedEchoGenerator(
            self.backend,
            DummyTokenizer(),
            FakeAutomaton(),
            config=EchoConfig(
                verification_threshold=4, max_num_draft=8, max_breadth=2, ngram=2
            ),
        ).generate(prompt, max_new_tokens=12)
        self.assertTrue(torch.equal(result.sequences, baseline))
        self.assertGreater(result.stats.draft_trees, 0)

    def test_rejected_suffix_defers_correction_without_an_extra_target_forward(self):
        from echo.inference.looped_ar import LoopedARGenerator

        prompt = torch.tensor([[2, 5, 7]])
        reference = LoopedARGenerator(self.backend, DummyTokenizer()).generate(
            prompt, max_new_tokens=12
        )
        wrong = (
            int(reference.sequences[0, prompt.shape[1] + 1]) + 1
        ) % self.backend.vocab_size
        original = self.backend.forward_shallow_tree

        def inaccurate_shallow(*args, **kwargs):
            result = original(*args, **kwargs)
            logits = torch.full_like(result.logits, -1e9)
            logits[..., wrong] = 0
            return LoopedForwardOutput(result.hidden_states, logits)

        with (
            patch.object(
                self.backend, "forward_shallow_tree", side_effect=inaccurate_shallow
            ),
            patch.object(
                self.backend,
                "forward_token_ids",
                side_effect=AssertionError("extra full-target correction"),
            ),
        ):
            result = LoopedEchoGenerator(
                self.backend,
                DummyTokenizer(),
                FakeAutomaton(),
                config=EchoConfig(
                    verification_threshold=4, max_num_draft=8, max_breadth=2, ngram=2
                ),
            ).generate(prompt, max_new_tokens=12)
        self.assertTrue(torch.equal(result.sequences, reference.sequences))
        self.assertGreater(result.stats.deferred_corrections, 0)
        self.assertEqual(result.stats.full_token_calls, 0)

    def test_python_and_benchmark_interfaces_share_history_and_acceptance(self):
        from evaluation.looped_common import generator

        config = EchoConfig(max_num_draft=8, max_breadth=2, ngram=2)
        direct = LoopedEchoGenerator(self.backend, DummyTokenizer(), config=config)
        options = SimpleNamespace(
            verification_threshold=config.verification_threshold,
            max_num_draft=config.max_num_draft,
            max_breadth=config.max_breadth,
            ngram=config.ngram,
            max_nodes=config.automaton_capacity,
        )
        benchmark = generator(self.backend, DummyTokenizer(), options, "echo")
        for prompt in ([2, 5, 7], [2, 5, 7, 11, 13], [2, 5, 7]):
            ids = torch.tensor([prompt])
            left = direct.generate(ids, max_new_tokens=12)
            right = benchmark.generate(ids, max_new_tokens=12)
            self.assertTrue(torch.equal(left.sequences, right.sequences))
            self.assertEqual(left.stats.drafted_nodes, right.stats.drafted_nodes)
            self.assertEqual(
                left.stats.target_accept_lengths, right.stats.target_accept_lengths
            )


if __name__ == "__main__":
    unittest.main()
