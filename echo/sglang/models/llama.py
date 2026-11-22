"""LayerSkip execution ranges over SGLang's native Llama model."""

import torch
from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.models.llama import LlamaForCausalLM


class EchoLlamaForCausalLM(LlamaForCausalLM):
    """Keep both fused residual streams when pausing at the early exit."""

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.total_steps = int(config.num_hidden_layers)
        # Each execution step is one ordinary decoder layer.
        self.num_layers = 1

    def forward_hidden_steps(
        self,
        positions,
        forward_batch,
        *,
        step_start,
        step_end,
        input_ids=None,
        hidden_states=None,
    ):
        if (input_ids is None) == (hidden_states is None):
            raise ValueError("Provide exactly one of input_ids and hidden_states")
        if input_ids is not None:
            hidden = self.model.embed_tokens(input_ids)
            residual = None
        else:
            if hidden_states.shape[-1] != 2 * self.config.hidden_size:
                raise ValueError("LayerSkip continuation needs both residual streams")
            hidden, residual = (
                part.clone()
                for part in hidden_states.split(self.config.hidden_size, dim=-1)
            )
        for index in range(step_start, step_end):
            hidden, residual = self.model.layers[index](
                positions, hidden, forward_batch, residual
            )
        # Native fused RMSNorm can mutate both inputs. Save the raw streams
        # before the early readout, so that readout never changes continuation.
        state = (
            torch.cat((hidden, residual), dim=-1)
            if step_end < self.total_steps
            else None
        )
        head, _ = self.model.norm(hidden, residual)
        return (head if state is None else state), head

    def all_logits(self, hidden, forward_batch):
        metadata = LogitsMetadata.from_forward_batch(forward_batch)
        return self.logits_processor._get_logits(hidden, self.lm_head, metadata)


EntryClass = EchoLlamaForCausalLM
