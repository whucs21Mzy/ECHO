"""Generate one response with recurrent ECHO or its same-backend AR control."""

import argparse
import json

from .looped_common import (
    add_model_arguments,
    aggregate,
    cleanup,
    generator,
    load,
    prepare_request,
    run_request,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--mode", choices=("echo", "ar"), default="echo")
    parser.add_argument(
        "--prompt",
        required=True,
        help="Rendered model input, without an extra chat wrapper",
    )
    args = parser.parse_args()
    backend, tokenizer = load(args)
    request = prepare_request(
        dict(prompt=args.prompt), 0, tokenizer, args, backend.vocab_size
    )
    record = run_request(
        generator(backend, tokenizer, args, args.mode), tokenizer, request, args
    )
    print(
        json.dumps(
            dict(
                method=args.mode,
                backend=args.backend,
                **record,
                metrics=aggregate([record]),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
