import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation import benchmark_looped as benchmark
from evaluation.looped_common import aggregate, prepare_request


class Tokenizer:
    eos_token_id = 9

    def __call__(self, text, **kwargs):
        raise AssertionError("Frozen token IDs must bypass prompt rendering")


class MetricsAndInputsTest(unittest.TestCase):
    def test_rendered_prompt_keeps_one_bos_marker(self):
        class RenderedTokenizer:
            eos_token_id = 9

            def __call__(self, text, add_special_tokens=True):
                vocabulary = {"<s>": 1, "hello": 2}
                ids = [vocabulary[word] for word in text.split()]
                return {"input_ids": ([1] if add_special_tokens else []) + ids}

        args = SimpleNamespace(seed=1234, max_new_tokens=4, context_length=16, stop=[])
        request = prepare_request(
            dict(prompt="<s> hello"), 0, RenderedTokenizer(), args, 10
        )
        self.assertEqual(request["input_ids"], [1, 2])

    def test_rates_use_sums_and_mat_includes_terminal_tokens(self):
        rows = [
            dict(
                completion_tokens=11,
                decode_tokens=10,
                decode_seconds=1.0,
                stats=dict(deep_calls=2, full_token_calls=0),
            ),
            dict(
                completion_tokens=1,
                decode_tokens=0,
                decode_seconds=0.25,
                stats=dict(deep_calls=1, full_token_calls=0),
            ),
        ]
        result = aggregate(rows)
        self.assertEqual(result["tok_s"], 8.0)
        self.assertEqual(result["mat"], 4.0)

    def test_frozen_input_retains_ids_seed_eos_and_stops(self):
        args = SimpleNamespace(seed=1234, max_new_tokens=4, context_length=16, stop=[])
        row = dict(
            input_ids=[1, 2, 3],
            prompt_id=17,
            doc_id="original",
            eos_token_ids=[8, 9],
            until=["END"],
        )
        request = prepare_request(row, 0, Tokenizer(), args, 10)
        self.assertEqual(request["input_ids"], [1, 2, 3])
        self.assertEqual(request["sampling_seed"], 1251)
        self.assertEqual(request["eos_token_ids"], [8, 9])
        self.assertEqual(request["stop_strings"], ["END"])

    def test_context_overflow_is_rejected_without_truncating_input(self):
        args = SimpleNamespace(seed=1234, max_new_tokens=14, context_length=16, stop=[])
        with self.assertRaisesRegex(ValueError, "context-length"):
            prepare_request(dict(input_ids=[1, 2, 3]), 0, Tokenizer(), args, 10)


class BenchmarkResumeTest(unittest.TestCase):
    def test_resume_without_metadata_is_rejected_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, output = root / "inputs.json", root / "answers.jsonl"
            inputs.write_text(json.dumps([dict(input_ids=[1, 2])]))
            output.write_text('{"request_id":0}\n')
            command = [
                "benchmark",
                "--model-path",
                str(root),
                "--shallow-steps",
                "1",
                "--input-file",
                str(inputs),
                "--output-file",
                str(output),
                "--resume",
            ]
            with (
                patch("sys.argv", command),
                patch.object(benchmark, "load") as load,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                benchmark.main()
            load.assert_not_called()

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, output = root / "inputs.json", root / "answers.jsonl"
            inputs.write_text(json.dumps([dict(input_ids=[1, 2])]))
            output.write_text("existing measurement\n")
            command = [
                "benchmark",
                "--model-path",
                str(root),
                "--shallow-steps",
                "1",
                "--input-file",
                str(inputs),
                "--output-file",
                str(output),
            ]
            with (
                patch("sys.argv", command),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                benchmark.main()
            self.assertEqual(output.read_text(), "existing measurement\n")

    def test_interrupted_pair_replays_echo_history_and_retains_measured_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, output = root / "inputs.json", root / "answers.jsonl"
            inputs.write_text(
                json.dumps([dict(input_ids=[1, 2]), dict(input_ids=[3, 4])])
            )
            args = [
                "benchmark",
                "--model-path",
                str(root),
                "--shallow-steps",
                "1",
                "--input-file",
                str(inputs),
                "--output-file",
                str(output),
                "--max-new-tokens",
                "4",
            ]
            fail = [True]

            def factory(backend, tokenizer, options, method):
                return SimpleNamespace(method=method, history=[])

            def generate(engine, tokenizer, request, options, cap=None):
                if engine.method == "echo" and cap is None:
                    if request["request_id"] == 1:
                        self.assertEqual(engine.history, [0])
                        if fail[0]:
                            fail[0] = False
                            raise RuntimeError("simulated interruption")
                    engine.history.append(request["request_id"])
                return dict(
                    output_ids=[5, 6, 9],
                    finish_reason=dict(type="stop", matched=9),
                    completion_tokens=3,
                    decode_tokens=2,
                    decode_seconds=1.0 if engine.method == "ar" else 0.5,
                    stats=dict(
                        deep_calls=0 if engine.method == "ar" else 1, full_token_calls=0
                    ),
                )

            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        benchmark,
                        "load",
                        return_value=(SimpleNamespace(vocab_size=10), Tokenizer()),
                    )
                )
                stack.enter_context(
                    patch.object(benchmark, "generator", side_effect=factory)
                )
                stack.enter_context(
                    patch.object(benchmark, "run_request", side_effect=generate)
                )
                stack.enter_context(
                    patch.object(
                        benchmark,
                        "source_identity",
                        return_value={"source": "same snapshot"},
                    )
                )
                stack.enter_context(patch.object(benchmark, "warmup"))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                with (
                    patch("sys.argv", args),
                    self.assertRaisesRegex(RuntimeError, "simulated interruption"),
                ):
                    benchmark.main()
                original = output.read_text()
                self.assertEqual(len(original.splitlines()), 1)
                interrupted_tail = '{"request_id":1,"echo":'
                with output.open("a") as stream:
                    stream.write(interrupted_tail)
                with patch("sys.argv", [*args, "--resume"]):
                    benchmark.main()
                backups = list(root.glob("answers.jsonl.interrupted-*"))
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups[0].read_text(), interrupted_tail)
                self.assertTrue(output.read_text().startswith(original))
                self.assertEqual(len(output.read_text().splitlines()), 2)
                summary = json.loads(output.with_suffix(".summary.json").read_text())
                self.assertEqual(summary["speedup"], 2.0)
                self.assertEqual(summary["echo"]["mat"], 3.0)
                self.assertEqual(summary["exact_matches"], 2)
                complete = output.read_text()
                output.with_suffix(".summary.json").unlink()
                with patch("sys.argv", [*args, "--resume"]):
                    benchmark.main()
                self.assertEqual(output.read_text(), complete)
                self.assertTrue(output.with_suffix(".summary.json").exists())
                corrupted = [json.loads(line) for line in complete.splitlines()]
                corrupted[0]["input_hash"] = "another request"
                output.write_text("".join(json.dumps(row) + "\n" for row in corrupted))
                with (
                    patch("sys.argv", [*args, "--resume"]),
                    self.assertRaisesRegex(RuntimeError, "frozen requests"),
                ):
                    benchmark.main()
                output.write_text(complete)
                inputs.write_text(
                    json.dumps([dict(input_ids=[1, 7]), dict(input_ids=[3, 4])])
                )
                with (
                    patch("sys.argv", [*args, "--resume"]),
                    self.assertRaisesRegex(RuntimeError, "changed"),
                ):
                    benchmark.main()


class ResultRecoveryTest(unittest.TestCase):
    def test_complete_final_record_receives_a_newline_before_appending(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "answers.jsonl"
            path.write_text('{"request_id":0}')
            self.assertEqual(benchmark.read_saved_rows(path), [dict(request_id=0)])
            self.assertEqual(path.read_bytes(), b'{"request_id":0}\n')

    def test_complete_but_invalid_record_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "answers.jsonl"
            original = b'{"request_id":0}\n{"invalid":\n{"request_id":1}\n'
            path.write_bytes(original)
            with self.assertRaises(json.JSONDecodeError):
                benchmark.read_saved_rows(path)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
