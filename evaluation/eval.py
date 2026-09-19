"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
# adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import inspect
import json
import os
import time
import torch
import numpy as np
import shortuuid

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from fastchat.conversation import get_conv_template
from tqdm import tqdm


def _model_blob(model_id=None, tokenizer=None):
    return " ".join(
        filter(
            None,
            [
                str(model_id or ""),
                str(getattr(tokenizer, "name_or_path", "") or ""),
            ],
        )
    ).lower()


def _is_layerskip_base(model_id=None, tokenizer=None):
    """Official facebook/layerskip-* are continued-pretrain on BASE, not Instruct."""
    return "layerskip" in _model_blob(model_id, tokenizer)


def _model_family(model_id=None, tokenizer=None):
    blob = _model_blob(model_id, tokenizer)
    if _is_layerskip_base(model_id, tokenizer):
        return "layerskip"
    if "codellama" in blob or "code-llama" in blob:
        return "codellama"
    if "llama3" in blob or "llama-3" in blob:
        return "llama3"
    if "llama2" in blob or "llama-2" in blob:
        return "llama2"
    if "vicuna" in blob:
        return "vicuna"
    if "qwen" in blob:
        return "qwen"
    return "unknown"


def _token_id(tokenizer, token):
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    tok_id = convert(token)
    unk = getattr(tokenizer, "unk_token_id", None)
    if not isinstance(tok_id, int) or tok_id < 0 or tok_id == unk:
        return None
    return tok_id


def _collect_stop_ids(tokenizer, extra_tokens=()):
    stop_ids = []
    if tokenizer is None:
        return stop_ids
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        stop_ids.append(eos_id)
    for token in extra_tokens:
        tok_id = _token_id(tokenizer, token)
        if tok_id is not None and tok_id not in stop_ids:
            stop_ids.append(tok_id)
    return stop_ids


def _fastchat_conversation(template_name, tokenizer, stop_str, extra_stop_tokens=()):
    conv = get_conversation_template(template_name)
    if conv.name == "one_shot":
        raise ValueError(
            f"FastChat template {template_name!r} fell through to one_shot; "
            "LayerSkip must call get_conv_template('one_shot') explicitly"
        )
    if stop_str is not None:
        conv.stop_str = stop_str
    stop_ids = list(conv.stop_token_ids or [])
    for tok_id in _collect_stop_ids(tokenizer, extra_stop_tokens):
        if tok_id not in stop_ids:
            stop_ids.append(tok_id)
    if stop_ids:
        conv.stop_token_ids = stop_ids
    return conv


def _one_shot_conversation(tokenizer):
    """FastChat default one_shot (few-shot + ### Human / ### Assistant).

    Do not use get_conversation_template('llama2'): that name is not an adapter
    and only accidentally resolved to one_shot.
    """
    conv = get_conv_template("one_shot")
    conv.stop_str = None
    eos_id = tokenizer.eos_token_id if tokenizer is not None else None
    conv.stop_token_ids = [eos_id] if eos_id is not None else None
    return conv


def _spec_bench_conversation(tokenizer=None, model_id=None):
    if _model_family(model_id, tokenizer) == "layerskip":
        return _one_shot_conversation(tokenizer)
    extra = ("<|end_of_text|>", "<|eot_id|>")
    return _fastchat_conversation("vicuna", tokenizer, "</s>", extra)


_HUMANEVAL_CHAT_PREFIX = "Implement the following code.\n"
_RAW_COMPLETION_BENCHES = ("human_eval", "humaneval", "gsm8k", "mgsm")


def _is_raw_completion_bench(question_file=None):
    """HumanEval / GSM8K are prefix-completion, not chat. Do not wrap FastChat."""
    blob = (question_file or "").replace("\\", "/").lower()
    return any(name in blob for name in _RAW_COMPLETION_BENCHES)


class _RawCompletionConversation:
    """Dataset prefix as-is. Stop only on eos, matching official HumanEval/GSM8K."""

    def __init__(self, tokenizer, name="raw-completion"):
        self.name = name
        self.roles = ("user", "assistant")
        self.messages = []
        self.stop_str = None
        eos_id = tokenizer.eos_token_id if tokenizer is not None else None
        self.stop_token_ids = [eos_id] if eos_id is not None else None

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_prompt(self):
        for _role, content in reversed(self.messages):
            if content is not None:
                return content
        return ""


def _eval_conversation(tokenizer=None, model_id=None, question_file=None):
    if _is_raw_completion_bench(question_file):
        return _RawCompletionConversation(tokenizer)
    return _spec_bench_conversation(tokenizer, model_id)


def _eval_user_text(qs, question_file=None):
    blob = (question_file or "").replace("\\", "/").lower()
    if "human_eval" in blob or "humaneval" in blob:
        if qs.startswith(_HUMANEVAL_CHAT_PREFIX):
            return qs[len(_HUMANEVAL_CHAT_PREFIX):]
    return qs


def _call_forward(forward_func, inputs, model, tokenizer, max_new_tokens, call_kwargs, profile_meta=None):
    """Pass profile_meta only if the forward function accepts it."""
    kw = dict(call_kwargs)
    if profile_meta is not None:
        try:
            sig = inspect.signature(forward_func)
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()) or (
                "profile_meta" in sig.parameters
            ):
                kw["profile_meta"] = profile_meta
        except (TypeError, ValueError):
            pass
    return forward_func(inputs, model, tokenizer, max_new_tokens, **kw)


def run_eval(
        model,
        tokenizer,
        forward_func,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_tokens,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        **kwargs,
):
    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model,
                tokenizer,
                forward_func,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_tokens,
                num_choices,
                question_file=question_file,
                **kwargs,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        forward_func,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        num_choices,
        **kwargs,
):

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]
    question_file = kwargs.pop("question_file", "") or ""

    _probe = _eval_conversation(tokenizer, model_id, question_file)
    print(
        f"Conversation family={_model_family(model_id, tokenizer)}  "
        f"template={_probe.name}  stop_str={_probe.stop_str!r}  "
        f"stop_token_ids={_probe.stop_token_ids}  "
        f"question_file={question_file}"
    )

    # warmup
    for _ in range(3):
        torch.manual_seed(0)
        conv = _eval_conversation(tokenizer, model_id, question_file)
        turns = []
        steps = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = _eval_user_text(question["turns"][j], question_file)
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            if _ == 0 and j == 0:
                print("Prompt preview:", repr(prompt[:180]))
            # # ==========================================
            # # DEBUG 核心输出：查看 FastChat 模板内容
            # # ==========================================
            # print("\n" + ">" * 30 + f" [DEBUG] Question {question['question_id']} Turn {j+1} " + "<" * 30)
            # # 使用 repr() 打印可以看清 \n, \t 和前后的空格
            # print(f"RAW PROMPT REPRESENTATION:\n{repr(prompt)}") 
            # print("-" * 20)
            # # 打印前 150 个字符和最后 100 个字符，观察 INST 闭合
            # print(f"PROMPT START: {prompt[:150]}")
            # print(f"PROMPT END  : {prompt[-100:]}")
            # print(">" * 80 + "\n")
            # # ==========================================
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids
            try:
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, step, accept_length_tree = _call_forward(
                    forward_func,
                    inputs,
                    model,
                    tokenizer,
                    max_new_tokens,
                    kwargs,
                    profile_meta={
                        "warmup": True,
                        "question_id": question["question_id"],
                        "category": question.get("category"),
                        "turn": j,
                        "choice": 0,
                    },
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]):]
                # be consistent with the template's stop_token_ids
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
            # ================= [位置 1: Warmup 报错捕获] =================
            except Exception as e:
                print("\n" + "!"*20 + " WARMUP CRITICAL ERROR " + "!"*20)
                print(f"Question ID: {question['question_id']}")
                print(f"Error Type: {type(e).__name__}")
                print(f"Error Message: {e}")
                import traceback
                traceback.print_exc() # 打印详细堆栈
                print("!"*60 + "\n")
                output = "ERROR"
                step, new_token, total_time = 0, 0, 0
                accept_length_tree = []
            # ==========================================================

            turns.append(output)
            steps.append(int(step))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            conv.messages[-1][-1] = output
    print('Warmup done')

    accept_lengths_tree = []
    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            cur_accept_lengths_tree = []
            torch.manual_seed(i)
            conv = _eval_conversation(tokenizer, model_id, question_file)
            turns = []
            steps = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = _eval_user_text(question["turns"][j], question_file)
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
                input_ids = inputs.input_ids
                try:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    output_ids, new_token, step, accept_length_tree = _call_forward(
                        forward_func,
                        inputs,
                        model,
                        tokenizer,
                        max_new_tokens,
                        kwargs,
                        profile_meta={
                            "warmup": False,
                            "question_id": question["question_id"],
                            "category": question.get("category"),
                            "turn": j,
                            "choice": i,
                        },
                    )
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    accept_lengths_tree.extend(accept_length_tree)
                    output_ids = output_ids[0][len(input_ids[0]):]

                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                # ================= [位置 2: 实际评测报错捕获] =================
                except Exception as e:
                    print("\n" + "!"*20 + " EVAL CRITICAL ERROR " + "!"*20)
                    print(f"Question ID: {question['question_id']} (Index: {i}, Turn: {j})")
                    print(f"Error Type: {type(e).__name__}")
                    print(f"Error Message: {e}")
                    import traceback
                    traceback.print_exc()
                    print("!"*60 + "\n")
                    output = "ERROR"
                    step, new_token, total_time = 0, 0, 0
                    accept_length_tree = []
                # ==========================================================

                turns.append(output)
                steps.append(int(step))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                cur_accept_lengths_tree.extend(accept_length_tree)
                conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                            "accept_lengths": cur_accept_lengths_tree})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")
    print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])

