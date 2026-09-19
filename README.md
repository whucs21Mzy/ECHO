# ECHO

Skip-layer speculative decoding for official Facebook LayerSkip base checkpoints.

Minimal inference checkout: KV Llama/Qwen + Aho–Corasick automaton, one decode entry, Spec-Bench / HumanEval / GSM8K questions.

## Layout

```
ECHO/
  evaluation/inference_echo.py   # skip-layer decode
  evaluation/eval.py             # prompt wrapping + jsonl writer
  evaluation/speed.py            # tok/s and mean accept length
  echo/                          # KV Llama/Qwen + draft automaton
    automaton/                   # Aho–Corasick trees, adapted from RACER
                                 # https://github.com/hkr04/RACER
  data/{spec_bench,human_eval,gsm8k,mgsm}/question.jsonl
```

## Install

```bash
conda create -n echo python=3.10 -y
conda activate echo
cd /data/mazy/ECHO
pip install -r requirements.txt
# fschat 0.2.31 still declares pydantic<2; overlay to match logitspec
pip install pydantic==2.12.5 pydantic_core==2.41.5 fastapi==0.120.3 uvicorn==0.38.0
pip install -e ./echo/automaton
```

Pinned to the working `logitspec` stack (Python 3.10, `torch==2.7.1+cu118`). System `g++` is required to build the automaton. Rebuild it if you change Python version.

## Prompts

| Bench | Wrap |
| --- | --- |
| `spec_bench` | LayerSkip: FastChat `one_shot`. Other models: Vicuna. |
| `human_eval` | Raw HumanEval prefix (function + docstring). No chat template. The Spec-Bench `"Implement the following code."` line is stripped. |
| `gsm8k` / `mgsm` | Dataset `Q:` / `A:` few-shot as-is. No chat template. |

Stop is eos only. Official LayerSkip weights are continued pretrain on **base** models, not Instruct.

## Run

From the repo root. Default decode is greedy (`temperature=0`).

```bash
# Spec-Bench, LayerSkip Llama-3-8B, L=10 T=3
CUDA_VISIBLE_DEVICES=0 python evaluation/inference_echo.py \
  --model-path /data/mazy/layerskip-llama3-8B \
  --model-id layerskip-llama3-8B \
  --bench-name spec_bench \
  --intermediate-layer 10 \
  --verification-threshold 3 \
  --max-nodes 30000 \
  --no-bottleneck-log

# HumanEval (raw completion)
CUDA_VISIBLE_DEVICES=0 python evaluation/inference_echo.py \
  --model-path /data/mazy/layerskip-llama3-8B \
  --model-id layerskip-llama3-8B \
  --bench-name human_eval \
  --intermediate-layer 10 \
  --verification-threshold 3 \
  --max-nodes 30000 \
  --no-bottleneck-log

# GSM8K (raw few-shot completion)
CUDA_VISIBLE_DEVICES=0 python evaluation/inference_echo.py \
  --model-path /data/mazy/layerskip-llama3-8B \
  --model-id layerskip-llama3-8B \
  --bench-name gsm8k \
  --intermediate-layer 10 \
  --verification-threshold 3 \
  --max-nodes 30000 \
  --no-bottleneck-log
```

Answers go to `data/<bench>/model_answer/<model-id>-echo-L{L}-T{T}_<timestamp>.jsonl`.

Vanilla greedy AR (same `eval.py` prompts, tokenizer, dtype, `max_new_tokens=512`, eos-only stop; no draft tree):

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/inference_vanilla.py \
  --model-path /data/mazy/layerskip-llama3-8B \
  --model-id layerskip-llama3-8B \
  --bench-name spec_bench
```

Answers go to `data/<bench>/model_answer/<model-id>-vanilla_<timestamp>.jsonl`. After both runs:

```bash
python evaluation/compare_answers.py \
  data/spec_bench/model_answer/<echo>.jsonl \
  data/spec_bench/model_answer/<vanilla>.jsonl
python evaluation/speed.py data/spec_bench/model_answer/<echo>.jsonl \
  --base-path data/spec_bench/model_answer/<vanilla>.jsonl
```
