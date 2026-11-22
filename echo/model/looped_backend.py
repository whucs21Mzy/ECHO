"""Model-family dispatch for Torch and native SGLang ECHO execution."""

import json
from pathlib import Path

import torch


def load_looped_tokenizer(model_path, *, backend="torch"):
    from transformers import AutoTokenizer

    options = {}
    if backend == "sglang":
        # AutoTokenizer otherwise imports the checkpoint's HF4-only Raven
        # config before the dedicated SGLang model-config parser can run.
        data = json.loads((Path(model_path) / "config.json").read_text())
        if data.get("model_type") == "huginn_raven":
            from echo.sglang.models.raven import EchoRavenConfig

            config = EchoRavenConfig.from_dict(data)
        elif data.get("model_type") == "ouro":
            from echo.sglang.models.ouro import EchoOuroConfig

            config = EchoOuroConfig.from_dict(data)
        elif data.get("model_type") == "llama":
            from transformers import LlamaConfig

            config = LlamaConfig.from_dict(data)
        else:
            raise ValueError(f"Unsupported ECHO model type: {data.get('model_type')!r}")
        config._name_or_path = str(model_path)
        options["config"] = config
    return AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True, **options
    )


def load_looped_backend(
    model_path,
    *,
    backend="torch",
    shallow_steps,
    dtype=torch.bfloat16,
    context_length=2048,
    max_tree_nodes=64,
    initial_seed=0,
    device_map=None,
    deterministic=True,
    mem_fraction_static=None,
    attn_implementation="sdpa",
):
    config = json.loads((Path(model_path) / "config.json").read_text())
    kind = config.get("model_type")
    if kind not in {"ouro", "huginn_raven", "llama"}:
        raise ValueError(f"Unsupported ECHO model type: {kind!r}")
    if kind == "llama":
        if backend != "sglang":
            raise ValueError(
                "LayerSkip uses the SGLang entry point here; "
                "for Torch use evaluation/inference_echo.py"
            )
        if not 0 < shallow_steps < config["num_hidden_layers"]:
            raise ValueError("The early-exit layer must precede the final Llama layer")
    if backend == "sglang":
        from echo.sglang.backend import (
            LayerSkipSGLangBackend,
            OuroSGLangBackend,
            RavenSGLangBackend,
        )

        cls = {
            "llama": LayerSkipSGLangBackend,
            "ouro": OuroSGLangBackend,
            "huginn_raven": RavenSGLangBackend,
        }[kind]
        return cls.from_pretrained(
            model_path,
            shallow_steps=shallow_steps,
            torch_dtype=dtype,
            context_length=context_length,
            max_tree_nodes=max_tree_nodes,
            initial_seed=initial_seed,
            deterministic=deterministic,
            mem_fraction_static=mem_fraction_static,
        )
    if backend != "torch":
        raise ValueError(f"Unsupported ECHO backend: {backend!r}")
    if kind == "huginn_raven":
        from .looped_raven import RavenLoopedEchoBackend

        result = RavenLoopedEchoBackend.from_pretrained(
            model_path,
            shallow_steps=shallow_steps,
            torch_dtype=dtype,
            context_length=context_length,
            initial_seed=initial_seed,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
    else:
        from .looped_ouro import OuroLoopedEchoBackend

        result = OuroLoopedEchoBackend.from_pretrained(
            model_path,
            shallow_steps=shallow_steps,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            local_files_only=True,
        )
        if device_map is None and torch.cuda.is_available():
            result.to("cuda:0")
    if deterministic and result.device.type == "cuda" and dtype == torch.bfloat16:
        from .stable_math import install

        result.stable_math = install(result.base_model)
    result.context_length = context_length
    return result
