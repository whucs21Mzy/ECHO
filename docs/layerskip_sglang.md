# LayerSkip checkpoints on SGLang

ECHO runs on the released LayerSkip Llama 2, Llama 3/3.2, and CodeLlama
checkpoints using SGLang's native Llama layers, projections, attention, and
paged KV cache. AR uses the same weights and kernels. This is ECHO's
hierarchical decoder on LayerSkip weights.

## Setup

Use the SGLang environment from the README:

```bash
python3.11 -m venv .venv-sglang
.venv-sglang/bin/pip install -r requirements-looped-sglang.txt
.venv-sglang/bin/pip install -e ./echo/automaton
```

Download a [LayerSkip checkpoint](https://github.com/facebookresearch/LayerSkip)
to a local directory. The backend accepts unquantized Llama checkpoints on
one GPU. Generation is greedy, in BF16 by default.

## Paired evaluation

Supply JSON/JSONL requests with rendered `prompt` text or frozen `input_ids`.
For text, include the checkpoint's BOS/chat template in the rendered prompt.
Optional `eos_token_ids` and `until` fields control EOS and task stops.

```json
{"id":"example-0","prompt":"<|begin_of_text|>Write a Python function that adds two numbers.","until":["\n\nQ:"]}
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv-sglang/bin/python -m evaluation.benchmark_layerskip \
  --model-path ./checkpoints/layerskip-llama3-8B \
  --intermediate-layer 10 --mode paired \
  --input-file ./prompts.jsonl --output-file ./outputs/layerskip-sglang.jsonl \
  --seed 1234 --max-new-tokens 1024 --context-length 2048 \
  --verification-threshold 5 --max-num-draft 64 --max-breadth 8 --ngram 10
```

`--intermediate-layer L` means the first L decoder layers, with
`0 < L < num_hidden_layers`. Set `--mode ar` or `--mode echo` to measure one
method. Each benchmark file can use the same entry point. The existing
Torch commands in the README remain available.

Paired mode checks greedy AR/ECHO agreement before measurement. Raw token
IDs, stops, timings, and verification counts are saved to JSONL. The adjacent
`.summary.json` reports decode throughput, ECHO/AR speedup, and MAT. Decode
timing excludes prefill and its first output token. MAT includes terminal
tokens and counts final verification calls. `--resume` validates the runtime
and inputs, then replays draft history before continuing.

For a single prompt:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-sglang/bin/python -m evaluation.inference_layerskip \
  --model-path ./checkpoints/layerskip-llama3-8B --intermediate-layer 10 \
  --prompt '<|begin_of_text|>The capital of France is' --max-new-tokens 64
```

## Execution

The adapter in `echo/sglang/models/llama.py` pauses the native layer stack at
the early exit. It preserves both residual streams before applying the
shared final norm and LM head for early logits. Final verification resumes
from these raw states, without repeating early layers. Tree compaction and
rejected-token rollback use the same request cache as recurrent ECHO.

The implementation uses eager native SGLang execution with CUDA Graphs
disabled. The reference configuration is SGLang 0.5.14.

## Checks

```bash
.venv-sglang/bin/python -m unittest discover -s tests -q

ECHO_TEST_SGLANG_MODEL=./checkpoints/layerskip-llama3.2-1B \
ECHO_TEST_SHALLOW_STEPS=4 \
  .venv-sglang/bin/python -m unittest discover -s tests -p test_sglang_looped.py -v
```

The GPU check uses FP32 and verifies both tree branches, accepted-path KV
compaction, deep continuation, and request-pool recovery. Use a checkpoint
that fits the available GPU memory in FP32.
