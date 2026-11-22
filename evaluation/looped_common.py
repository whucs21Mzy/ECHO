"""Shared loading, request handling and metrics for the looped entry points."""

import hashlib
import importlib.metadata
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path


def add_model_arguments(parser, *, default_backend="torch"):
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--backend", choices=("torch", "sglang"), default=default_backend)
    parser.add_argument(
        "--shallow-steps",
        "--intermediate-layer",
        type=int,
        required=True,
        help="Early readout after K recurrent iterations or L LayerSkip layers",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--initial-state-seed", type=int, default=0)
    parser.add_argument("--verification-threshold", type=int, default=5)
    parser.add_argument(
        "--max-num-draft",
        type=int,
        default=64,
        help="Candidate nodes, including the root",
    )
    parser.add_argument("--max-breadth", type=int, default=8)
    parser.add_argument("--ngram", type=int, default=10)
    parser.add_argument(
        "--max-nodes", type=int, default=30000, help="Automaton storage capacity"
    )
    parser.add_argument(
        "--stop",
        action="append",
        default=[],
        help="Repeat for multiple decoded-text stops",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        help="SGLang memory fraction; defaults to SGLang's GPU-specific setting",
    )


def load(args):
    import torch

    from echo.model.looped_backend import load_looped_backend, load_looped_tokenizer

    positive = (
        args.shallow_steps,
        args.max_new_tokens,
        args.context_length,
        args.verification_threshold,
        args.max_num_draft,
        args.max_breadth,
        args.max_nodes,
    )
    if min(positive) < 1:
        raise ValueError("Loop and generation parameters must be positive")
    if args.ngram < 0:
        raise ValueError("ngram must be non-negative")
    if args.mem_fraction_static is not None and not 0 < args.mem_fraction_static < 1:
        raise ValueError("mem-fraction-static must lie between zero and one")
    tokenizer = load_looped_tokenizer(args.model_path, backend=args.backend)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    backend = load_looped_backend(
        args.model_path,
        backend=args.backend,
        shallow_steps=args.shallow_steps,
        dtype=getattr(torch, args.dtype),
        context_length=args.context_length,
        max_tree_nodes=args.max_num_draft,
        initial_seed=args.initial_state_seed,
        deterministic=True,
        mem_fraction_static=args.mem_fraction_static,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if args.backend == "torch":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    return backend, tokenizer


def generator(backend, tokenizer, args, method):
    if method == "ar":
        from echo.inference.looped_ar import LoopedARGenerator

        return LoopedARGenerator(backend, tokenizer)
    from echo.inference.looped_echo import EchoConfig, LoopedEchoGenerator

    config = EchoConfig(
        verification_threshold=args.verification_threshold,
        max_num_draft=args.max_num_draft,
        max_breadth=args.max_breadth,
        ngram=args.ngram,
        automaton_capacity=args.max_nodes,
    )
    return LoopedEchoGenerator(backend, tokenizer, config=config)


def prepare_request(row, index, tokenizer, args, vocab_size):
    text = row.get("prompt", row.get("text"))
    ids = row.get("input_ids")
    if ids is None:
        if not isinstance(text, str):
            raise ValueError("Each request needs input_ids or a rendered prompt/text")
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(x) is not int or not 0 <= x < vocab_size for x in ids)
    ):
        raise ValueError(
            "input_ids must be a nonempty list of target-vocabulary token IDs"
        )
    if len(ids) + args.max_new_tokens > args.context_length:
        raise ValueError(
            f"Request {index}: input and generation limit exceed context-length"
        )
    prompt_id = row.get("prompt_id", index)
    if type(prompt_id) is not int:
        raise ValueError("prompt_id must be an integer for reproducible request seeds")
    eos = row.get(
        "eos_token_ids",
        [] if tokenizer.eos_token_id is None else [tokenizer.eos_token_id],
    )
    stops = row.get("until", row.get("stop", args.stop))
    if isinstance(stops, str):
        stops = [stops]
    if not isinstance(stops, list) or any(
        not isinstance(s, str) or not s for s in stops
    ):
        raise ValueError("Stops must be nonempty strings")
    if not isinstance(eos, list) or any(
        type(x) is not int or not 0 <= x < vocab_size for x in eos
    ):
        raise ValueError("Invalid EOS token IDs")
    return dict(
        request_id=index,
        prompt_id=prompt_id,
        doc_id=row.get("doc_id", row.get("id", index)),
        input_ids=ids,
        eos_token_ids=eos,
        stop_strings=stops,
        sampling_seed=args.seed + prompt_id,
    )


def run_request(engine, tokenizer, request, args, cap=None):
    import torch

    seed = request["sampling_seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if engine.backend.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    inputs = torch.tensor([request["input_ids"]], dtype=torch.long)
    runner = getattr(engine.backend, "runner", None)

    def pool_state():
        if runner is None:
            return None
        return (
            runner.req_to_token_pool.available_size(),
            runner.token_to_kv_pool_allocator.available_size(),
        )

    initial_pool = pool_state()
    result = engine.generate(
        inputs,
        max_new_tokens=cap or args.max_new_tokens,
        eos_token_ids=request["eos_token_ids"],
        stop_strings=request["stop_strings"],
        include_stop_tokens=True,
    )
    if pool_state() != initial_pool:
        raise RuntimeError("A request leaked SGLang request/KV pool capacity")
    output = result.sequences[0, inputs.shape[1] :].cpu().tolist()
    text = tokenizer.decode(
        output, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    matched = result.finish_reason.get("matched")
    if isinstance(matched, str):
        text = text.split(matched, 1)[0]
    stats = asdict(result.stats)
    return dict(
        output_ids=output,
        text=text,
        completion_tokens=len(output),
        finish_reason=result.finish_reason,
        decode_tokens=max(len(output) - 1, 0),
        decode_seconds=stats["stage_timings"]["decode_seconds"],
        prefill_seconds=stats["stage_timings"]["prefill_seconds"],
        stats=stats,
    )


def aggregate(records):
    tokens = sum(r["decode_tokens"] for r in records)
    seconds = sum(r["decode_seconds"] for r in records)
    completions = sum(r["completion_tokens"] for r in records)
    verifies = sum(r["stats"]["deep_calls"] for r in records)
    return dict(
        requests=len(records),
        completion_tokens=completions,
        decode_tokens=tokens,
        decode_seconds=seconds,
        tok_s=tokens / seconds if seconds else None,
        verification_calls=verifies,
        mat=completions / verifies if verifies else None,
        full_token_calls=sum(r["stats"]["full_token_calls"] for r in records),
    )


def first_divergence(left, right):
    return next(
        (i for i, (a, b) in enumerate(zip(left, right)) if a != b),
        min(len(left), len(right)) if len(left) != len(right) else None,
    )


def source_identity(args, backend):
    import torch

    root = Path(__file__).resolve().parents[1]
    paths = [
        *(root / "echo").rglob("*.py"),
        root / "echo/automaton/src/automaton.cpp",
        Path(__file__),
        root / "evaluation/benchmark_looped.py",
    ]
    paths.extend((root / "echo/automaton").glob("*.so"))
    versions = {}
    for name in ("torch", "transformers", "triton", "sglang"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    hardware = dict(device=str(backend.device), cuda=torch.version.cuda)
    if backend.device.type == "cuda":
        properties = torch.cuda.get_device_properties(backend.device)
        hardware.update(
            gpu=properties.name,
            uuid=str(properties.uuid),
            compute_capability=[properties.major, properties.minor],
            total_memory=properties.total_memory,
        )
    runner = getattr(backend, "runner", None)
    if runner is not None:
        hardware.update(
            mem_fraction_static=runner.server_args.mem_fraction_static,
            kv_pool_tokens=runner.token_to_kv_pool_allocator.available_size(),
        )
    return dict(
        python=sys.version,
        versions=versions,
        hardware=hardware,
        source_hashes={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths))
        },
        model_config_sha256=hashlib.sha256(
            (args.model_path / "config.json").read_bytes()
        ).hexdigest(),
        checkpoint_weight_files={
            p.name: dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(args.model_path.glob("*.safetensors"))
        },
        checkpoint_code_hashes={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(
                [
                    *args.model_path.glob("*.py"),
                    *args.model_path.glob("*.index.json"),
                    *args.model_path.glob("*tokenizer*.json"),
                    *args.model_path.glob("generation_config.json"),
                ]
            )
        },
        common_kernel_flags={
            name: os.environ.get(name, "1")
            for name in (
                "ECHO_RAVEN_FUSED_RMS",
                "ECHO_RAVEN_FUSED_ROPE",
                "ECHO_OURO_FUSED_ROPE",
            )
        },
        cuda_graphs=False,
    )


def cleanup():
    torch = sys.modules.get("torch")
    if torch is not None and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
