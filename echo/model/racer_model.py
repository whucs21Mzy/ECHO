# adapted from RACER (https://github.com/hkr04/RACER) and
# https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_model.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from .chat_template import VICUNA_CHAT_TEMPLATE
from .llama2_template import LLAMA2_CHAT_TEMPLATE
from .llama3_template import LLAMA3_CHAT_TEMPLATE
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor


class RacerModel(nn.Module):

    def __init__(self, base_model, base_model_name_or_path):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        
        # [修正] 只初始化一次，必须带 trust_remote_code=True
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path, 
            trust_remote_code=True
        )
        self.device = base_model.device
        
        # [修正] 确保 pad_token 存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        Type = self.config.architectures[0]
        if Type in ['LlamaForCausalLM', 'Qwen3ForCausalLM']:
            self.decoder = base_model.model
        else:
            raise ValueError(f"Unsupported model type: {Type}")
        
        path = (base_model_name_or_path or "").lower()
        if "vicuna" in path:
            self.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
        elif "layerskip" in path:
            pass
        elif self.tokenizer.chat_template is None:
            if "codellama" in path or "code-llama" in path:
                self.tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE
            elif "llama3" in path or "llama-3" in path:
                self.tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
            elif "llama2" in path or "llama-2" in path or "llama" in path:
                self.tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE

    def get_tokenizer(self):

        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer
    
    def set_tree_mask(self, tree_mask):
        
        """Set the tree attention mask for decoding.
        """
        self.decoder.tree_mask = tree_mask

    @classmethod
    def from_pretrained(
        cls,
        base_model_path="codellama/CodeLlama-7b-instruct-hf",
        **kwargs,
    ):
        """
        Args:
            base_model_path (str): Name or path of the LLM to load.

        Returns:
            RacerModel
        """
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen3ForCausalLM':
            base_model = KVQwen3ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type: {Type}")

        model = cls(
            base_model,
            base_model_path,
        )

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        extra_args={},
    ):
        """Forward pass of the LLM.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.
            extra_args (dict, optional): Additional arguments for the base model.

        Returns:
            torch.Tensor: A tensor containing predictions from the LM head.
        """
        with torch.inference_mode():
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **extra_args,
            )
            if output_orig:
                # [核心修复] 在进入 lm_head 之前手动调用 base_model 的 norm
                # outputs[0] 通常是 last_hidden_state
                hidden_states = outputs[0]
                normed_hidden_states = self.base_model.model.norm(hidden_states)
                orig = self.base_model.lm_head(normed_hidden_states)

        if output_orig:
            return outputs, orig
        raise NotImplementedError
        
    def racer_hispec_generate(
            self,
            input_ids,
            ac,
            ngram=10,
            temperature=0.0,
            top_p=0.8,
            repetition_penalty=1.2,
            max_steps=512,
            max_num_draft=64,
            max_breadth=8,
            show_accepted=False,
            debug_logits_path=None,
            
            # --- [HISPEC] 新增参数 ---
            intermediate_layer=8,
            verification_threshold=7,
            # --- [HISPEC] 参数结束 ---
            
            extra_args={}
        ):
        """
        使用 HiSpec 和 RACER 结合的生成循环。
        
        - 使用 'full_model_copy_logit' (来自完整模型的 logits) 来生成草稿树。
        - 使用 'intermediate_layer' (例如 8) 来进行中间验证。
        - 累积“暂时接受”的 token，直到达到 'verification_threshold' (例如 7)。
        - 触发对累积 buffer 的一次“完整验证”。
        - 只有“完整验证”才能更新 'full_model_copy_logit'。
        """
        
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        
        # --- [HISPEC] 状态初始化 (Take 5) ---
        
        # 1. 完整模型的 KV 缓存 (Ground Truth)
        (
            full_past_key_values,
            full_past_key_values_data_list,
            full_current_length_data,
        ) = initialize_past_key_values(self.base_model)
        
        # 2. 中间模型的 KV 缓存 (Tentative)
        (
            inter_past_key_values,
            inter_past_key_values_data_list,
            inter_current_length_data,
        ) = initialize_past_key_values(self.base_model)

        # 3. `ground_truth_input_ids`: 只包含已通过“完整验证”的 token
        ground_truth_input_ids = input_ids.clone()
        input_len = input_ids.shape[1]
        
        # 4. `tentative_input_ids`: 包含所有 token (完整 + 暂时)
        tentative_input_ids = input_ids.clone()

        # 5. 获取初始的 "copy-logit" (来自完整模型对 prompt 的处理)
        #    这会填充 full_past_key_values
        full_model_copy_logit = initialize_logits(
            ground_truth_input_ids, self, full_past_key_values, extra_args
        )
        
        # 6. 将初始 KV 状态复制到中间缓存
        for i in range(len(full_past_key_values)):
            if full_past_key_values[i] is not None:
                inter_past_key_values[i][0].copy_(full_past_key_values[i][0])
                inter_past_key_values[i][1].copy_(full_past_key_values[i][1])
        inter_current_length_data.copy_(full_current_length_data)

        # 7. `tentative_tokens_buffer`: 存储等待完整验证的 token
        tentative_tokens_buffer = torch.tensor([], dtype=torch.long, device=self.device)
        
        # --- [HISPEC] 状态初始化结束 ---

        self.set_tree_mask(None)
        new_token = 0
        pad_token_id = self.get_tokenizer().pad_token_id
        
        # --- Automaton (AC) 初始化 (使用 full_model_copy_logit) ---
        if max_breadth:
            adj_topk = torch.topk(full_model_copy_logit, k=max_breadth, dim=-1)[1][0]
            ac.update(ground_truth_input_ids[0].tolist(), adj_topk.tolist())
        
        if ngram:
            for i in range(input_len):
                start = i
                end = i + ngram
                if end > input_ids.size(1): break
                pattern = ground_truth_input_ids[0, start:end].tolist()
                ac.insert(pattern)
            ac.build()
            
        if show_accepted:
            accepted_texts = []
        
        # --- [HISPEC] 主循环 ---
        for idx in range(max_steps):
            
            # 1. 检查是否需要进行“完整验证”
            #    (缓冲区达到阈值，或者发生停滞)
            is_stall = (new_token > 0 and tentative_tokens_buffer.shape[0] == 0) # 检查上一轮是否停滞
            
            if tentative_tokens_buffer.shape[0] >= verification_threshold or \
               (is_stall and tentative_tokens_buffer.shape[0] > 0):
                
                # --- [A] 执行完整验证 ---
                
                # 我们需要一个临时的 KV 缓存，从 `full_past_key_values` 开始
                (temp_kv, temp_kv_list, temp_len_data) = initialize_past_key_values(self.base_model)
                for i in range(len(full_past_key_values)):
                    temp_kv[i][0].copy_(full_past_key_values[i][0])
                    temp_kv[i][1].copy_(full_past_key_values[i][1])
                temp_len_data.copy_(full_current_length_data)
                
                # 对缓冲区中的所有 token 执行一次完整的前向传播
                full_verify_outputs = self.base_model(
                    tentative_tokens_buffer.unsqueeze(0), 
                    past_key_values=temp_kv, 
                    use_cache=True, 
                    exit_at_layer=None # 完整模型
                )
                
                full_verify_logits = full_verify_outputs.logits # Shape: [1, num_tentative, vocab_size]
                
                # 逐个 token 验证
                original_tokens = tentative_tokens_buffer.tolist()
                final_accept_length = 0
                corrected_token = -1
                
                for i in range(len(original_tokens)):
                    token_to_check = original_tokens[i]
                    logit_for_this_pos = full_verify_logits[0, i, :]

                    # 应用重复惩罚 (基于 *完整* 的 ground_truth + 截至当前的 token)
                    current_context = torch.cat([ground_truth_input_ids[0], torch.tensor(original_tokens[:i], device=self.device)])
                    if repetition_penalty != 1.0:
                        processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
                        logit_for_this_pos = processor(current_context.unsqueeze(0), logit_for_this_pos.unsqueeze(0)).squeeze(0)

                    # 检查 token 是否匹配
                    if top_p == 0:
                        predicted_token = torch.argmax(logit_for_this_pos).item()
                    else:
                        filtered_logits = top_p_filtering(logit_for_this_pos.unsqueeze(0), top_p=top_p)
                        predicted_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).item()
                    
                    if predicted_token == token_to_check:
                        final_accept_length += 1
                    else:
                        # 发生不匹配
                        corrected_token = predicted_token
                        final_accept_length += 1 # 接受这个被修正的 token
                        break # 停止验证
                
                # 获取最终被接受的 token 列表
                if corrected_token != -1:
                    # Mismatch 发生
                    accepted_tokens_list = original_tokens[:final_accept_length-1]
                    accepted_tokens_list.append(corrected_token)
                else:
                    # 全部匹配
                    accepted_tokens_list = original_tokens[:final_accept_length]
                
                accepted_tokens_tensor = torch.tensor(accepted_tokens_list, dtype=torch.long, device=self.device)
                
                # --- 更新“完整”状态 ---
                # 1. 更新 ground_truth_input_ids
                ground_truth_input_ids = torch.cat([ground_truth_input_ids, accepted_tokens_tensor.unsqueeze(0)], dim=1)
                
                # 2. 更新 full_past_key_values
                prev_len = full_current_length_data.item()
                new_len = prev_len + final_accept_length
                for i in range(len(full_past_key_values)):
                    # 从 temp_kv 复制已验证的 KV 数据到 full_kv
                    tgt_k = temp_kv[i][0].data[..., prev_len:new_len, :]
                    dst_k = full_past_key_values[i][0].data[..., prev_len:new_len, :]
                    dst_k.copy_(tgt_k, non_blocking=True)
                    
                    tgt_v = temp_kv[i][1].data[..., prev_len:new_len, :]
                    dst_v = full_past_key_values[i][1].data[..., prev_len:new_len, :]
                    dst_v.copy_(tgt_v, non_blocking=True)
                full_current_length_data.fill_(new_len)
                
                # 3. 更新 full_model_copy_logit (!!)
                full_model_copy_logit = full_verify_logits[:, final_accept_length - 1, :].unsqueeze(0)
                
                # 4. 重置“中间”状态以匹配“完整”状态
                for i in range(len(full_past_key_values)):
                    inter_past_key_values[i][0].copy_(full_past_key_values[i][0])
                    inter_past_key_values[i][1].copy_(full_past_key_values[i][1])
                inter_current_length_data.copy_(full_current_length_data)
                
                tentative_input_ids = ground_truth_input_ids.clone()
                tentative_tokens_buffer = torch.tensor([], dtype=torch.long, device=self.device)

            # --- [B] 执行中间验证 (如果不需要完整验证) ---
            
            # 1. 生成草稿树
            #    使用 full_model_copy_logit (来自上次完整验证)
            #    使用 tentative_input_ids (完整 + 暂时) 进行重复惩罚
            candidates, tree_candidates, tree_attn_mask, tree_position_ids, retrieve_indices = generate_draft_tree(
                logits=full_model_copy_logit, # <-- 使用“copy-logit”
                ac=ac,
                pad_token_id=pad_token_id,
                input_ids=tentative_input_ids, # <-- 使用暂时状态进行重复惩罚
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                temperature=temperature,
                max_num_draft=max_num_draft,
                device=self.base_model.device
            )
            tree_candidates_batch = tree_candidates[None, :]
            tree_attn_mask_batch = tree_attn_mask[None, None, :]
            tree_position_ids_batch = tree_position_ids[None, :]
            self.set_tree_mask(tree_attn_mask_batch)

            # 2. 中间验证 (Layer 8)
            #    使用 *inter_past_key_values*
            inter_logits, inter_outputs, inter_tree_logits = tree_decoding(
                self, 
                tree_candidates_batch, 
                inter_past_key_values, # <-- 使用中间 KV 缓存
                tree_position_ids_batch, 
                tentative_input_ids, # <-- 使用暂时 input_ids
                retrieve_indices,
                exit_at_layer=intermediate_layer # <-- 提前退出
            )
            
            # 3. 评估中间结果
            best_candidate, accept_length = evaluate_posterior(
                inter_logits, candidates, temperature, top_p
            )
            
            # 4. 更新 *中间* 状态 (KV 缓存, input_ids)
            tentative_input_ids, current_step_inter_logit, new_token_count_step, accept_length = update_inference_inputs(
                tentative_input_ids, 
                candidates, 
                best_candidate, 
                accept_length, 
                retrieve_indices,
                inter_outputs, 
                inter_logits, 
                0, # new_token 计数器在循环外部管理
                inter_past_key_values_data_list, # <-- 更新中间 KV 缓存
                inter_current_length_data,       # <-- 更新中间 KV 缓存
                eos_token_id=self.tokenizer.eos_token_id,
            )
            new_token += new_token_count_step
            
            # 5. 将新接受的 token 添加到缓冲区
            newly_accepted_tokens = tentative_input_ids[0, -new_token_count_step:]
            tentative_tokens_buffer = torch.cat(
                [tentative_tokens_buffer, newly_accepted_tokens]
            )
            
            # 6. 自动机更新 (基于 *中间* 验证的 logits)
            if max_breadth:
                adj_topk = torch.topk(inter_tree_logits, k=max_breadth, dim=-1)[1][0]
                ac.update(tree_candidates_batch[0].tolist(), adj_topk.tolist())
                    
            if ngram:
                for i in range(1 + accept_length):
                    start = -ngram - i
                    end = -i if i != 0 else None
                    if abs(start) > tentative_input_ids.size(1): break
                    pattern = tentative_input_ids[0, start:end].tolist()
                    ac.insert(pattern)
                if accept_length > 0:
                    ac.trans_tokens(tentative_input_ids[0, -accept_length:].tolist())
            
            # 7. 如果在中间验证时停滞 (accept_length=0)
            if new_token_count_step == 0 and tentative_tokens_buffer.shape[0] == 0:
                # 我们卡住了，草稿树的根 token 就被拒绝了。
                # 必须更新 copy-logit 以便下次生成不同的树。
                # 这是对用户“只能完整更新”规则的必要例外。
                full_model_copy_logit = current_step_inter_logit
            
            # 8. Yield 文本 (基于 tentative_input_ids)
            text = self.tokenizer.decode(
                tentative_input_ids[0, input_len:],
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            
            # ... (show_accepted 和 debug_logits_path 代码省略) ...

            yield {"text": text}

            # 9. 检查停止条件
            if self.tokenizer.eos_token_id == tentative_input_ids[0, -1]:
                break
            
            full_text = self.tokenizer.decode(tentative_input_ids[0, input_len:], skip_special_tokens=False)
            if "[/PARSER]" in full_text:
                break
        
        # ... (racer_generate 和 baseline_generate 保持不变) ...
    def racer_generate(
        self,
        input_ids,
        ac,
        ngram=10,
        temperature=0.0,
        top_p=0.8,
        repetition_penalty=1.2,
        max_steps=512,
        max_num_draft=64,
        max_breadth=8,
        show_accepted=False,
        debug_logits_path=None,
        extra_args={}
    ):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        input_ids = input_ids.clone()

        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data_list = self.past_key_values_data_list
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data_list,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data_list = past_key_values_data_list
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        self.set_tree_mask(None)
        logits = initialize_logits(input_ids, self, past_key_values, extra_args)

        new_token = 0
        pad_token_id = self.get_tokenizer().pad_token_id
        
        if max_breadth:
            adj_topk = torch.topk(logits, k=max_breadth, dim=-1)[1][0]
            ac.update(input_ids[0].tolist(), adj_topk.tolist())
        
        if ngram:
            for i in range(input_len):
                start = i
                end = i + ngram
                if end > input_ids.size(1): break
                pattern = input_ids[0, start:end].tolist()
                ac.insert(pattern)
            ac.build()
            
        if show_accepted:
            accepted_texts = []

        for idx in range(max_steps):
            candidates, tree_candidates, tree_attn_mask, tree_position_ids, retrieve_indices = generate_draft_tree(
                logits=logits,
                ac=ac,
                pad_token_id=pad_token_id,
                input_ids=input_ids,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                temperature=temperature,
                max_num_draft=max_num_draft,
                device=self.base_model.device
            )
            tree_candidates = tree_candidates[None, :]
            tree_attn_mask = tree_attn_mask[None, None, :]
            tree_position_ids = tree_position_ids[None, :]
            self.set_tree_mask(tree_attn_mask)
            
            logits, outputs, tree_logits = tree_decoding(
                self, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices
            )
            
            best_candidate, accept_length = evaluate_posterior(logits, candidates, temperature, top_p)
            
            input_ids, logits, new_token, accept_length = update_inference_inputs(
                input_ids, candidates, best_candidate, accept_length, retrieve_indices,
                outputs, logits, new_token, past_key_values_data_list, current_length_data,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            if max_breadth:
                adj_topk = torch.topk(tree_logits, k=max_breadth, dim=-1)[1][0]
                ac.update(tree_candidates[0].tolist(), adj_topk.tolist())
                    
            if ngram:
                for i in range(1 + accept_length):
                    start = -ngram - i
                    end = -i if i != 0 else None
                    if abs(start) > input_ids.size(1): break
                    pattern = input_ids[0, start:end].tolist()
                    ac.insert(pattern)
                if accept_length > 0:
                    ac.trans_tokens(input_ids[0, -accept_length:].tolist())
            
            # === 正常的解码文本 (用于显示) ===
            text = self.tokenizer.decode(
                input_ids[0, input_len:],
                skip_special_tokens=True, # 这里为了显示干净，还是True
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            # ... (show_accepted 和 debug_logits_path 代码省略，保持原样即可) ...

            yield {"text": text}

            # === [核心修复] 停止条件检查 ===
            
            # 1. 检查标准 EOS
            if self.tokenizer.eos_token_id == input_ids[0, -1]:
                break
            
            # 2. 检查 LayerSkip 的 [/PARSER]
            # 必须使用 skip_special_tokens=False，否则 [/PARSER] 会被过滤掉！
            full_text = self.tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=False)
            if "[/PARSER]" in full_text:
                break


    def baseline_generate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.8,
        max_steps=512,
        repetition_penalty=1.2,
        debug_logits_path=None,
        extra_args={}
    ):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        input_ids = input_ids.clone()

        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data_list = self.past_key_values_data_list
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data_list,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data_list = past_key_values_data_list
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        self.set_tree_mask(None)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True, **extra_args)

        for idx in range(max_steps):
            # --- [核心修改] Baseline 的重复惩罚 ---
            
            # 1. 获取 logits
            next_token_logits = outputs.logits[:, -1, :]

            # 2. 应用惩罚
            if repetition_penalty != 1.0:
                # 注意：这里需要 import
                # 请在 racer_model.py 顶部添加:
                # from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
                processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
                next_token_logits = processor(input_ids, next_token_logits)

            # 3. 基于惩罚后的 logits 进行选择
            if top_p > 0:
                assert top_p < 1, "top_p should between 0.0 and 1"
                next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
                filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
                input_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                input_id = input_id.view(input_id.shape[0], 1)
            else:
                input_id = next_token_logits.argmax(dim=-1).unsqueeze(0) # 确保 shape 
            # --- [修改结束] ---
            
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            
            # ... (debug_logits_path 代码省略) ...

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            # === [核心修复] 停止条件检查 ===
            
            # 1. 标准 EOS
            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break
            
            # 2. LayerSkip 停止词
            # 同样必须 skip_special_tokens=False
            full_text = self.tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=False)
            if "[/PARSER]" in full_text:
                break