# ECHO for looped transformers

ECHO verifies drafts with the first **K recurrent iterations**, then runs the
remaining iterations for final verification. Both Torch and SGLang support:

| Model | Full recurrence S | Early readout K |
| --- | ---: | ---: |
| [Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B), [Ouro-2.6B](https://huggingface.co/ByteDance/Ouro-2.6B) | 4 | 1 |
| [Huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125) | 32 | 8 |
| [Raven-Llama-3.2](https://huggingface.co/smcleish/Recurrent-Llama-3.2-train-recurrence-32) | 32 | 4 |
| [Recurrent-TinyLlama-3T](https://huggingface.co/smcleish/Recurrent-TinyLlama-3T-train-recurrence-32) | 32 | 4 |
| [Recurrent-OLMo-2-0425](https://huggingface.co/smcleish/Recurrent-OLMo-2-0425-train-recurrence-32) | 32 | 4 |

Each recurrent iteration has its own KV slots. Raven-family models also keep
independent early and final coda slots. The decoder compacts accepted paths
and rolls back rejected suffixes before continuing.

## Installation

Use separate Python 3.11 environments and CUDA-enabled PyTorch builds.
From the repository root, install one backend and build the automaton:

```bash
pip install -r requirements-looped.txt          # Torch / Transformers 4
# Or: pip install -r requirements-looped-sglang.txt
pip install -e ./echo/automaton
```

Both backends use deterministic BF16 arithmetic and eager execution.
SGLang uses native model layers, Triton attention and paged KV. Torch uses
row-invariant linear, RMSNorm and attention kernels. Use `--dtype float32`
for numerical checks.

## Generate

```bash
CUDA_VISIBLE_DEVICES=0 python -m evaluation.inference_looped \
  --model-path ./checkpoints/Ouro-1.4B \
  --backend torch --shallow-steps 1 \
  --prompt 'Solve: 17 * 23 =' --max-new-tokens 128
```

Choose `--backend sglang` in its environment; use `--mode ar` for the
same-backend AR control. Supply rendered prompts, including any chat template.

## Benchmark

Provide a JSON list or JSONL of rendered prompts:

```json
{"id":"example-0","prompt":"Solve: 17 * 23 =","stop":["\n\nQ:"]}
```

Text prompts are tokenized as supplied; include the model's BOS and chat
markers in the rendered input.

Frozen inputs may instead specify `input_ids`, `prompt_id`, `doc_id`,
`eos_token_ids` and `until`. Input order is preserved; each request uses seed
`1234 + prompt_id`. The runner checks prompt and generation limits before timing.

```bash
CUDA_VISIBLE_DEVICES=0 python -m evaluation.benchmark_looped \
  --model-path ./checkpoints/Ouro-1.4B \
  --backend torch --shallow-steps 1 --mode paired \
  --input-file ./prompts.jsonl --output-file ./outputs/ouro-torch.jsonl \
  --max-new-tokens 1024 --context-length 2048
```

Paired mode warms the kernels and checks AR/ECHO outputs, then measures both
methods on the same inputs. Use `--mode echo` or `--mode ar` for one method.
Defaults: 64 draft nodes including the root, breadth 8, n-gram length 10,
verification threshold 5 and automaton capacity 30,000.

Local outputs include raw token IDs, stop reasons, timings and a
`.metadata.json` file recording the code, inputs and configuration.
EOS and explicit task stops end generation. The summary reports:

- **tok/s:** total (completion tokens − 1) / total decode seconds. Prefill
  time is measured separately.
- **MAT:** total completion tokens / ECHO deep-verification calls, including
  root and terminal tokens.
- **Speedup:** ECHO tok/s / same-backend AR tok/s.
- Exact-match counts and the first divergent token of each mismatched pair.

Use `--resume` to continue an interrupted run. The runner checks input and
source fingerprints, then replays completed ECHO requests to restore its
history. A truncated final record is backed up and retried; completed records
are retained. Each input file starts a fresh automaton after warmup and checks.

## Python API

```python
from echo.inference.looped_echo import EchoConfig, LoopedEchoGenerator
from echo.model.looped_backend import load_looped_backend, load_looped_tokenizer

path = "./checkpoints/Ouro-1.4B"
tokenizer = load_looped_tokenizer(path, backend="torch")
backend = load_looped_backend(path, backend="torch", shallow_steps=1)
engine = LoopedEchoGenerator(backend, tokenizer, config=EchoConfig())
ids = tokenizer("Solve: 17 * 23 =", return_tensors="pt").input_ids
result = engine.generate(ids, max_new_tokens=128, include_stop_tokens=True)
```

Pass `config=EchoConfig(...)` to customize draft settings. AR uses
`LoopedARGenerator(backend, tokenizer)` with the same `generate` arguments.
The API and command-line entry points use the same history policy.

The shared algorithm is in `echo/inference/looped_echo.py`; Torch model
execution is in `echo/model/looped_ouro.py` and `looped_raven.py`. SGLang
separates model execution (`backend.py`), initialization (`runtime.py`), KV
ownership (`cache.py`) and attention (`ops/tree_attention.py`).

## Tests

```bash
python -m unittest discover -s tests -v
```

Released-model CPU tests instantiate small models from checkpoint code.
Set `ECHO_TEST_OURO_MODEL` to an Ouro checkpoint and `ECHO_TEST_MODEL_ROOT`
to the directory containing Raven-family checkpoints.

GPU tests are opt-in: set `ECHO_TEST_SGLANG_MODEL` for native SGLang checks,
`ECHO_TEST_CUDA_OPS=1` for arithmetic kernels, or `ECHO_TEST_TORCH_MODEL` and
`ECHO_TEST_SHALLOW_STEPS` for the real-checkpoint Torch tree check.
