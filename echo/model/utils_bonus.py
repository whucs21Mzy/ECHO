# adapted from https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/utils.py

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from . import MODEL_CLASS_MAP
import importlib
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
from typing import Optional

def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))



def initialize_logits(input_ids, model, past_key_values, extra_args={}):
    """
    Forward pass through the model to obtain the model outputs, and logits.


    Args:
    - input_ids (torch.Tensor): The input tensor containing token ids.
    - model: The LLM for generation.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.
    - extra_args (dict): Additional arguments to be passed to the model.

    Returns:
    - logits (torch.Tensor): logits from the LLM.
    """
    outputs, logits = model(
        input_ids, past_key_values=past_key_values, output_orig=True, extra_args=extra_args
    )
    return logits


def reset_past_key_values(passed_key_values):
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_draft_tree(logits, ac, pad_token_id, input_ids, repetition_penalty=1.0, top_p=0., temperature=1., max_num_draft=64, device="cuda"):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - logits (torch.Tensor): Original logits.
    - ac (Automaton): Used for retrieving candidates.
    - temperature (float): Softmax temperature for probability scaling.
    - top_p (float): Nucleus sampling threshold. If 0, greedy decoding is used.
    - max_num_draft (int): Maximum number of draft candidates to generate.
    - device (str): Device to run the computation on (default is "cuda").
    
    Returns:
    - dict: Returns candidates (paths of the tree), tree_candidates (BFS sequence of candidates),
             tree_attn_mask (attention mask for the BFS sequence), tree_position_ids (positional IDs for the BFS sequence),
             retrieve_indices (indices for reordering the logits, mapping each prefix to a BFS index).
    """

    # --- [核心修改] 应用重复惩罚 ---
    if logits.dim() == 3:
        # [batch, seq, vocab] -> [batch, vocab]
        next_token_logits = logits[:, -1]
    elif logits.dim() == 2:
        # 已经是 [batch, vocab]
        next_token_logits = logits
    else:
        raise ValueError(f"Expected logits to be 2D or 3D, but got {logits.dim()}D shape: {logits.shape}")

    # 2. 应用重复惩罚
    if repetition_penalty != 1.0:
        processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
        next_token_logits = processor(input_ids, next_token_logits)
    
    # 3. 基于被惩罚过的 logits 进行选择
    if top_p == 0:
        # 贪婪解码
        next_token = torch.argmax(next_token_logits).unsqueeze(0)
    else:
        # 采样
        # assert top_p < 1, "top_p should between 0.0 and 1" # 假设 top_p_filtering 内部处理了
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
        # 假设 top_p_filtering 和 F.softmax 已正确导入
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(0)
        
    # --- [修改结束] ---
        
    buf = ac.retrieve(next_token.item(), max_num_draft)
    
    candidates = buf.candidates
    tree_candidates = buf.tree_candidates
    tree_attn_mask = buf.attn_mask
    tree_position_ids = buf.position_ids # ⚠️ WARNING: This contains LINEAR indices!
    retrieve_indices = buf.retrieve_indices
    
    # [DEBUG TREE]
    # print(f"\n[DEBUG Tree Gen] Start Token: {next_token.item()} | Num Candidates: {len(candidates)}")
    # if tree_position_ids and isinstance(tree_position_ids, list) and len(tree_position_ids) > 1:
    #     print(f"[DEBUG Tree Gen] Raw Pos IDs (Sample 10): {tree_position_ids[:10]} | Total Len: {len(tree_position_ids)}")
    
    # Make sure length of BFS seq is at least 2
    if len(tree_candidates) <= 1:
        candidates = [[next_token.item(), pad_token_id]]
        tree_candidates = [next_token.item(), pad_token_id]
        tree_attn_mask = [[1, 0], [1, 1]]
        tree_position_ids = [0, 1]
        retrieve_indices = [[0, 1]]
    
    # Pad the candidates to the maximum depth with 0
    max_depth = max(len(candidate) for candidate in candidates)
    candidates = [pad_path(candidate, max_depth, pad_token_id) for candidate in candidates]
    
    # Pad the retrieved candidates to the maximum depth with -1
    seq_len = max(len(indices) for indices in retrieve_indices)
    retrieve_indices = [pad_path(indices, seq_len, -1) for indices in retrieve_indices]
    
    # === [开始] 强力清洗逻辑 v2 ===
    if candidates is None: candidates = []
    
    def sanitize_data(data):
        if data is None: return 0
        if isinstance(data, list): return [sanitize_data(x) for x in data]
        if isinstance(data, tuple): return tuple(sanitize_data(x) for x in data)
        return data

    candidates = sanitize_data(candidates)
    
    # === [针对 tree_candidates 的清洗逻辑] ===
    if tree_candidates is None: tree_candidates = []
    def sanitize_tree_data(data):
        if data is None: return 0
        if isinstance(data, list): return [sanitize_tree_data(x) for x in data]
        if isinstance(data, tuple): return tuple(sanitize_tree_data(x) for x in data)
        return data

    tree_candidates = sanitize_tree_data(tree_candidates)
    
    # Convert lists to tensors
    candidates = torch.tensor(candidates, device=device, dtype=torch.long)
    tree_candidates = torch.tensor(tree_candidates, device=device, dtype=torch.long)
    # =======================================
    tree_attn_mask = torch.tensor(tree_attn_mask, device=device)
    tree_position_ids = torch.tensor(tree_position_ids, device=device)
    retrieve_indices = torch.tensor(retrieve_indices, device=device)
    
    return candidates, tree_candidates, tree_attn_mask, tree_position_ids, retrieve_indices


def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    tree_position_ids,
    input_ids,
    retrieve_indices,
    exit_at_layer: Optional[int] = None  # <--- [HISPEC] 新增参数
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.
    
    Parameters:
    # ... (other params) ...
    - exit_at_layer (Optional[int]): Layer to exit at for intermediate verification.
    
    Returns:
    - tuple: Returns logits, and other outputs from the model.
    """

    # Compute new position IDs by adding the draft position IDs to the length of the input sequence.
    position_ids = tree_position_ids + input_ids.shape[1]

    # --- [HISPEC] 准备 extra_args ---
    extra_args = {}
    if exit_at_layer is not None:
        extra_args["exit_at_layer"] = exit_at_layer
    # --- [HISPEC] 修改结束 ---

    # --- [!!!] 开始修复 [!!!] ---
    # model() 返回 (BaseModelOutput, orig_logits)，我们必须解包
    model_outputs, tree_logits = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
        extra_args=extra_args,
    )
    # --- [!!!] 修复结束 [!!!] ---

    # Reorder the obtained logits based on the retrieve_indices...
    logits = tree_logits[0, retrieve_indices]

    # --- [!!!] 开始修复 [!!!] ---
    # 返回正确的 'model_outputs' 对象，而不是错误的 'outputs' 元组
    return logits, model_outputs, tree_logits
    # --- [!!!] 修复结束 [!!!] ---

def get_nucleus_posterior_mask(logits, candidates, temperature, top_p):

    # adapted from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79

    # Apply temperature
    logits = logits[:, :-1] / temperature

    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)

    # Convert to probabilities (softmax)
    probs = F.softmax(logits, dim=-1)
    # Sort the probabilities
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cum_probs = torch.cumsum(sorted_logits, dim=-1)

    # Create mask for the top-p nucleus
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)


    # Remove low-probability tokens
    logits[indices_to_remove] = float('-inf')

    # Sample from the remaining tokens
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    # Create a mask for selected tokens
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()

    return posterior_mask

def evaluate_posterior(logits, candidates, temperature, top_p=0.8, is_bonus_step=False):
    """
    逻辑：logits[:, i] 预测的是 candidates[:, i+1]
    candidates[:, 0] 是草稿树的根节点（Bonus Token）。
    """
    if temperature == 0:
        # 保持原版：只验证索引 1 之后的 token
        posterior_mask = (candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        
        # 核心逻辑：如果是 bonus_step，accept_length 至少为 0 (代表接受了根节点)
        # 即使 mask 全为 0，返回 (0, 0) 也会让后续逻辑把 candidates[0] 存入 buffer
        return best_candidate, accept_length

    elif top_p > 0:
        posterior_mask = get_nucleus_posterior_mask(logits, candidates, temperature, top_p)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        
        return best_candidate, accept_length
    else:
        raise NotImplementedError

def evaluate_posterior(logits, candidates, temperature, top_p=0.8, is_bonus_step=False):
    """
    逻辑：logits[:, i] 预测的是 candidates[:, i+1]
    candidates[:, 0] 是草稿树的根节点（Bonus Token）。
    """
    if temperature == 0:
        # 保持原版：只验证索引 1 之后的 token
        posterior_mask = (candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        
        # 核心逻辑：如果是 bonus_step，accept_length 至少为 0 (代表接受了根节点)
        # 即使 mask 全为 0，返回 (0, 0) 也会让后续逻辑把 candidates[0] 存入 buffer
        return best_candidate, accept_length

    elif top_p > 0:
        posterior_mask = get_nucleus_posterior_mask(logits, candidates, temperature, top_p)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        
        return best_candidate, accept_length
    else:
        raise NotImplementedError


def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    outputs,
    logits,
    new_token,
    past_key_values_data_list,
    current_length_data,
    eos_token_id,
):
    """
    Update the input sequences and relevant tensors based on the selected best candidate from the inference results.

    Args:
    - input_ids (torch.Tensor): Current input token sequences.
    - candidates (torch.Tensor): Candidate token sequences generated in the current step.
    - best_candidate (int): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - retrieve_indices (torch.Tensor): Indices to map tree to a cartesian product.
    - outputs, logits (torch.Tensor): Model's outputs from the previous inference step.
    - new_token (int): Counter for the new tokens added during inference.
    - past_key_values_data_list (list): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.
    - eos_token_id (int): End-of-sequence token ID, used to determine when generation should stop.

    Returns:
    - input_ids (torch.Tensor): Updated input token sequences.
    - logits (torch.Tensor): Updated logits.
    - new_token (int): Updated counter for the new tokens added.
    - accept_length (int): Real accept length after possible truncation by eos_token_id
    """
    # Check if eos_token_id is in the accepted candidate sequence
    # if so, truncate the sequence at the eos_token_id
    eos_index = [
        i
        for i, id in enumerate(candidates[best_candidate, : accept_length + 1])
        if id == eos_token_id
    ]

    if eos_index:
        accept_length = eos_index[0]
    
    # Calculate the starting position for new tokens based on the previous input length
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [
            input_ids,
            candidates[None, best_candidate, : accept_length + 1].to(input_ids.device),
        ],
        dim=-1,
    )
    
    # --- [CRITICAL FIX: KV Cache 长度更新] ---
    
    # [DEBUG KV] Print info before KV copy
    # print(f"\n[DEBUG KV Update] Prev Len: {prev_input_len}, Accepted Len: {accept_length + 1}")
    # print(f"[DEBUG KV Update] Select Indices (Source): {select_indices.tolist()}")

    # Update the past key values based on the selected tokens
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[
            ..., select_indices.to(past_key_values_data.device), :
        ]
        # Destination tensor where the relevant past information will be stored
        # 这里 prev_input_len 实际上是 KV Cache 中要写入的起始位置
        dst = past_key_values_data[
            ..., prev_input_len : prev_input_len + tgt.shape[-2], :
        ]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # [CRITICAL FIX: Off-by-One in Length]
    # new length = previous committed length + new accepted tokens length
    new_total_len = prev_input_len + accept_length + 1
    
    # Update the current length tensor (currently only support batch size is 1)
    # [FIX] 确保 current_length_data 被正确更新，而不是用 tgt.shape[-2]
    current_length_data.fill_(new_total_len)
    
    # [DEBUG KV] Print info after KV update
    # print(f"[DEBUG KV Update] New Total Len (Data): {new_total_len} | Updated Length Data: {current_length_data.item()}")
    # ----------------------------------------
    
    # Extract logits for the accepted tokens
    logits = logits[None, best_candidate, accept_length : accept_length + 1]

    # Update the new token counter
    new_token += accept_length + 1

    return input_ids, logits, new_token, accept_length


def top_p_filtering(logits, top_p=0.0, filter_value=float('-inf')):
    # from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79


    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
    return logits