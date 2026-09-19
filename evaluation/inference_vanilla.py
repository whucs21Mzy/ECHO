"""Vanilla greedy AR on the same EchoModel / eval.py path as skip-layer decode.

Same prompts, tokenizer, dtype, max_new_tokens, and eos-only stop as
evaluation/inference_echo.py. Token choice is argmax of the full-stack
base_model logit (temperature=0, no repetition penalty, no draft tree).
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from evaluation.eval import run_eval, reorg_answer_file
from fastchat.utils import str_to_torch_dtype
from echo.model.utils import top_p_filtering
from echo.model.racer_model import RacerModel as EchoModel
from echo.model.kv_cache import initialize_past_key_values


def vanilla_forward(inputs, model, tokenizer, max_new_tokens, max_tokens, temperature=0.0, top_p=0.0):
    input_ids = inputs.input_ids
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    input_ids = input_ids.clone()

    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        current_length_data = model.current_length_data
        current_length_data.zero_()
    else:
        past_key_values, past_key_values_data_list, current_length_data = initialize_past_key_values(
            model.base_model
        )
        model.past_key_values = past_key_values
        model.past_key_values_data_list = past_key_values_data_list
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    model.set_tree_mask(None)
    outputs = model.base_model(input_ids, past_key_values=past_key_values, use_cache=True)

    steps = 0
    for steps in range(max_new_tokens):
        if top_p > 0:
            assert top_p < 1, "top_p should between 0.0 and 1"
            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.0)
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            input_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            input_id = input_id.view(input_id.shape[0], 1)
        else:
            input_id = outputs.logits[:, -1:].argmax(dim=-1)
        outputs = model.base_model(input_id, use_cache=True, past_key_values=past_key_values)
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        if tokenizer.eos_token_id in input_ids[0, input_len:]:
            break
        if input_ids.size(-1) + 1 > max_tokens:
            break

    n_new = steps + 1
    accept_length_list = [1] * n_new
    return input_ids, n_new, n_new, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--bench-name", type=str, default="spec_bench")
    parser.add_argument("--question-begin", type=int)
    parser.add_argument("--question-end", type=int)
    parser.add_argument("--answer-file", type=str)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()

    if args.temperature == 0:
        args.top_p = 0.0

    args.model_id = f"{args.model_id}-vanilla"
    question_file = f"data/{args.bench_name}/question.jsonl"
    if not args.answer_file:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        args.answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}_{timestamp}.jsonl"

    model = EchoModel.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        device_map="auto",
    )
    model.eval()
    tokenizer = model.get_tokenizer()

    if args.max_tokens is None:
        args.max_tokens = model.base_model.config.max_position_embeddings

    print(f"Output to {args.answer_file}")
    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=vanilla_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=args.answer_file,
        max_new_tokens=args.max_new_tokens,
        max_tokens=args.max_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=1,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    reorg_answer_file(args.answer_file)
