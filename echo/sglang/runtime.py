"""Register ECHO model families and initialize an eager SGLang runner."""

import json
import os
import socket
from pathlib import Path

import torch
import triton

_registered = False


def _register_model():
    global _registered
    if _registered:
        return
    from sglang.srt.configs.model_config_parser_registry import (
        ModelConfigParserBase,
        register_model_config_parser,
    )
    from sglang.srt.models.registry import ModelRegistry
    from transformers import LlamaConfig

    from echo.sglang.models.ouro import EchoOuroConfig
    from echo.sglang.models.raven import EchoRavenConfig

    @register_model_config_parser("echo_looped_transformer")
    class EchoLoopedParser(ModelConfigParserBase):
        def parse(self, model, trust_remote_code=False, revision=None, **kwargs):
            path = Path(model) / "config.json"
            if not path.is_file():
                raise ValueError(
                    "The ECHO SGLang backend currently loads local checkpoints"
                )
            data = json.loads(path.read_text())
            if data.get("model_type") == "ouro":
                config = EchoOuroConfig.from_dict(data)
                config.architectures = ["EchoOuroForCausalLM"]
            elif data.get("model_type") == "huginn_raven":
                config = EchoRavenConfig.from_dict(data)
                config.architectures = ["EchoRavenForCausalLM"]
            elif data.get("model_type") == "llama":
                config = LlamaConfig.from_dict(data)
                config.architectures = ["EchoLlamaForCausalLM"]
            else:
                raise ValueError(
                    "ECHO's SGLang backend supports Llama, Ouro, and Raven checkpoints"
                )
            config._name_or_path = str(model)
            return config

    ModelRegistry.register("echo.sglang.models", overwrite=True, strict=True)
    _registered = True


def _local_port():
    # SGLang also uses port + 10000 for initialization.
    for port in range(38080, 38180):
        with socket.socket() as listener:
            try:
                listener.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No free local initialization port")


def load_runner(
    model_path,
    *,
    torch_dtype,
    context_length,
    max_tree_nodes,
    mem_fraction_static,
    deterministic,
):
    if torch_dtype == torch.float32:
        # Torch's allow_tf32 flag does not govern Triton dot products.
        # Use IEEE dot products for independent FP32 target checks too.
        os.environ["TRITON_F32_DEFAULT"] = "ieee"
        if hasattr(triton, "knobs"):
            triton.knobs.language.fp32_default = "ieee"
    _register_model()
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import PortArgs, ServerArgs

    args = ServerArgs(
        model_path=str(model_path),
        model_config_parser="echo_looped_transformer",
        trust_remote_code=True,
        dtype=str(torch_dtype).split(".")[-1],
        attention_backend="triton",
        page_size=1,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        max_prefill_tokens=context_length,
        max_running_requests=max(
            128, max_tree_nodes, context_length if torch_dtype == torch.float32 else 0
        ),
        context_length=context_length,
        mem_fraction_static=mem_fraction_static,
        max_total_tokens=2 * (context_length + max_tree_nodes) + 256,
        triton_attention_num_kv_splits=1,
        random_seed=1234,
        host="127.0.0.1",
        port=_local_port(),
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        enable_deterministic_inference=deterministic,
    )
    runner = ModelRunner(
        model_config=ModelConfig.from_server_args(args),
        mem_fraction_static=args.mem_fraction_static,
        gpu_id=0,
        nccl_port=PortArgs.init_new(args).nccl_port,
        server_args=args,
        tp_rank=0,
        tp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
    )
    runner.alloc_memory_pool()
    # Reserve flattened tree columns without changing the serving context.
    pool = runner.req_to_token_pool
    required = context_length + max_tree_nodes
    if pool.req_to_token.shape[1] < required:
        old = pool.req_to_token
        pool.req_to_token = torch.zeros(
            (old.shape[0], required), dtype=old.dtype, device=old.device
        )
        pool.req_to_token[:, : old.shape[1]].copy_(old)
        pool.max_context_len = required
    runner.init_attention_backends()
    runner.model.eval()
    _configure_pointwise(runner.model, torch_dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    return runner


def _configure_pointwise(model, torch_dtype):
    # Ordinary LayerSkip Llama uses SGLang's fused CUDA kernels. The native
    # fallbacks below preserve recurrent-checkpoint arithmetic and FP32 checks.
    if model.config.model_type == "llama" and torch_dtype != torch.float32:
        return
    if model.config.model_type == "huginn_raven":
        from sglang.srt.layers.layernorm import RMSNorm

        from echo.sglang.ops.raven_norm import install_raven_rmsnorm

        for module in model.modules():
            if isinstance(module, RMSNorm):
                install_raven_rmsnorm(module)
    # SGLang's fused RMSNorm/SwiGLU CUDA extensions target FP16/BF16.
    # Its own native fallbacks support FP32 correctness diagnostics.
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    for module in model.modules():
        if (torch_dtype == torch.float32 and isinstance(module, RMSNorm)) or isinstance(
            module, (SiluAndMul, RotaryEmbedding)
        ):
            module._forward_method = module.forward_native
    if model.config.model_type == "ouro":
        from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

        from echo.sglang.ops.ouro_rope import install_ouro_rope

        for module in model.modules():
            if isinstance(module, RotaryEmbedding):
                install_ouro_rope(module)
