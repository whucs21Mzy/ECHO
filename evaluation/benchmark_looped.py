"""Resumable, fixed-prefix AR/ECHO throughput with raw output IDs and MAT."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

from .looped_common import (
    add_model_arguments,
    aggregate,
    cleanup,
    first_divergence,
    generator,
    load,
    prepare_request,
    run_request,
    source_identity,
)


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_rows(path):
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of request objects")
        return data
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_saved_rows(path):
    """Keep complete records and preserve an interrupted final write separately."""
    rows = []
    with path.open("rb+") as stream:
        while line := stream.readline():
            start = stream.tell() - len(line)
            try:
                rows.append(json.loads(line))
            except (ValueError, UnicodeError):
                if line.endswith(b"\n"):
                    raise
                digest = hashlib.sha256(line).hexdigest()[:16]
                backup = path.with_suffix(path.suffix + f".interrupted-{digest}")
                if not backup.exists():
                    with backup.open("xb") as saved_tail:
                        saved_tail.write(line)
                        saved_tail.flush()
                        os.fsync(saved_tail.fileno())
                stream.truncate(start)
                break
            if not line.endswith(b"\n"):
                stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return rows


def warmup(backend, max_nodes):
    import torch

    prefix_length = min(3, backend.context_length - 1)
    prefix = torch.tensor(
        [[min(n, backend.vocab_size - 1) for n in (2, 5, 7)[:prefix_length]]],
        device=backend.device,
    )
    limit = min(max_nodes, backend.context_length - prefix_length)
    with torch.inference_mode():
        for width in sorted({min(n, limit) for n in (1, 2, 3, 4, 8, 16, 32, 64)}):
            cache, _ = backend.prefill(prefix)
            try:
                ids = torch.full(
                    (width,),
                    min(11, backend.vocab_size - 1),
                    dtype=torch.long,
                    device=backend.device,
                )
                mask = torch.ones(
                    (width, width), dtype=torch.bool, device=backend.device
                ).tril()
                shallow = backend.forward_shallow_tree(
                    ids,
                    mask,
                    torch.arange(width, device=backend.device),
                    prefix_length,
                    cache,
                )
                backend.forward_deep(shallow.hidden_states, prefix_length, cache)
            finally:
                close = getattr(cache, "close", None)
                if close is not None:
                    close()
    if backend.device.type == "cuda":
        torch.cuda.synchronize(backend.device)


def summarize(records, methods):
    result = {name: aggregate([r[name] for r in records]) for name in methods}
    if len(methods) == 2:
        ar, echo = result["ar"], result["echo"]
        result.update(
            speedup=echo["tok_s"] / ar["tok_s"] if ar["tok_s"] else None,
            exact_matches=sum(r["equal"] for r in records),
            mismatches=sum(not r["equal"] for r in records),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--mode", choices=("paired", "echo", "ar"), default="paired")
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="JSON list or JSONL of fixed rendered prompts/token IDs",
    )
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, help="Use the first N requests, preserving input order"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    source = read_rows(args.input_file)
    if args.limit is not None:
        source = source[: args.limit]
    if not source:
        parser.error("Input contains no requests")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.with_suffix(".lock").open("a") as output_lock:
        fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.output_file.exists() and not args.resume:
            parser.error("Output exists; select a new file or use --resume")
        if (
            args.output_file.exists()
            and args.output_file.stat().st_size
            and not args.output_file.with_suffix(".metadata.json").exists()
        ):
            parser.error("Cannot resume nonempty results without their metadata file")
        measure(args, source)


def measure(args, source):
    backend, tokenizer = load(args)
    requests = [
        prepare_request(row, i, tokenizer, args, backend.vocab_size)
        for i, row in enumerate(source)
    ]
    config = {
        k: str(v.resolve()) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k != "resume"
    }
    identity = dict(
        config=config,
        source=source_identity(args, backend),
        input_sha256=hashlib.sha256(args.input_file.read_bytes()).hexdigest(),
        request_hashes=[
            hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
            for r in requests
        ],
        history="Fresh automaton after warmup and preflight; persistent in input order; replayed when resuming",
    )
    metadata = args.output_file.with_suffix(".metadata.json")
    if metadata.exists() and json.loads(metadata.read_text()) != identity:
        raise RuntimeError("Runtime, inputs or settings changed since the saved run")
    save(metadata, identity)
    saved = read_saved_rows(args.output_file) if args.output_file.exists() else []
    if [r["request_id"] for r in saved] != list(range(len(saved))) or len(saved) > len(
        requests
    ):
        raise RuntimeError("Saved rows are not an ordered unique prefix")
    methods = ("ar", "echo") if args.mode == "paired" else (args.mode,)
    for row, expected_hash in zip(saved, identity["request_hashes"]):
        if row.get("input_hash") != expected_hash or any(
            name not in row for name in methods
        ):
            raise RuntimeError("Saved rows do not match the frozen requests or methods")
    warmup(backend, args.max_num_draft)
    if args.mode != "ar":
        controls = []
        ar = generator(backend, tokenizer, args, "ar")
        echo = generator(backend, tokenizer, args, "echo")
        for request in requests[:2]:
            reference = run_request(
                ar, tokenizer, request, args, cap=min(args.max_new_tokens, 128)
            )
            actual = run_request(
                echo, tokenizer, request, args, cap=min(args.max_new_tokens, 128)
            )
            passed = (
                reference["output_ids"] == actual["output_ids"]
                and reference["finish_reason"] == actual["finish_reason"]
            )
            controls.append(
                dict(
                    request_id=request["request_id"],
                    passed=passed,
                    ar=reference,
                    echo=actual,
                )
            )
            save(
                args.output_file.with_suffix(".preflight.json"),
                dict(passed=all(c["passed"] for c in controls), checks=controls),
            )
            if not passed:
                raise RuntimeError(
                    "Paired AR/ECHO preflight failed; see the saved control outputs"
                )
    engines = {name: generator(backend, tokenizer, args, name) for name in methods}
    for request in requests:
        index = request["request_id"]
        if index < len(saved):
            if "echo" in engines:
                replay = run_request(engines["echo"], tokenizer, request, args)
                previous = saved[index]["echo"]
                if (
                    replay["output_ids"] != previous["output_ids"]
                    or replay["finish_reason"] != previous["finish_reason"]
                ):
                    raise RuntimeError("Automaton replay changed a saved output")
            continue
        record = dict(
            request_id=index,
            prompt_id=request["prompt_id"],
            doc_id=request["doc_id"],
            sampling_seed=request["sampling_seed"],
            input_hash=identity["request_hashes"][index],
        )
        for name in methods:
            record[name] = run_request(engines[name], tokenizer, request, args)
        if len(methods) == 2:
            a, b = record["ar"], record["echo"]
            record.update(
                equal=a["output_ids"] == b["output_ids"]
                and a["finish_reason"] == b["finish_reason"],
                first_divergence=first_divergence(a["output_ids"], b["output_ids"]),
            )
        with args.output_file.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        saved.append(record)
        summary = summarize(saved, methods)
        save(args.output_file.with_suffix(".summary.json"), summary)
        print(
            json.dumps(dict(completed=len(saved), expected=len(requests), **summary)),
            flush=True,
        )
    save(args.output_file.with_suffix(".summary.json"), summarize(saved, methods))


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
