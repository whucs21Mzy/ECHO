"""ECHO skip-layer speculative decoding.

Early-exit draft trees accumulate to T, then remaining layers verify the chain.

One outer step:
1. Seed a 64-wide draft tree from the frozen target bonus logit (copy-logit).
2. Verify the tree with layers 0..L-1 (early-exit). Accept the speculative path.
3. If the tentative buffer is still shorter than T, take the layer-L logit at the
   last accepted position as the early bonus and grow another 64-wide tree.
   Repeat until len(buffer) >= T.
4. Remaining layers L..31 run on [last_committed_hs, buffer HS] from position
   pos-1. logits[:, i] checks buffer[i]; logits[:, accepted_count] is the bonus
   / correction copy-logit for the next outer step's first draft tree.
   Remaining is linear on the L1-accepted chain: tokens the early layers reject
   are almost always rejected by later layers as well, so remaining does not
   re-score the 64-tree.
5. Early automaton update: after each L1 tree, write adjacency from every tree
   node (original LM head + RMSNorm) so the next inner-loop retrieve can see
   them. Remaining-layer accept overwrites adjacency with target logits.
   Ngrams: insert the tentative L1 suffix after a tree and do not insert again
   at remaining.
6. Retrieval cursor: retrieve()+trans_tokens, no reset_to_root at sample start.
   Rewind to the committed suffix only if remaining rejects.
7. Remaining layers and L1 early-exit (layers 0..L-1) both go through
   transformers LlamaDecoderLayer / LlamaAttention. torch.compile wraps those
   layer stacks unless ECHO_COMPILE_REMAINING=0.
"""
import argparse
import os
import sys
import time
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from evaluation.eval import run_eval, reorg_answer_file
from evaluation.echo_bottleneck import EchoBottleneckLog, NullBottleneckLog, format_summary, load_records, summarize
from fastchat.utils import str_to_torch_dtype
from echo.model.utils import evaluate_posterior, generate_draft_tree
from echo.model.racer_model import RacerModel as EchoModel
from echo.model.kv_cache import initialize_past_key_values
from echo.automaton import Automaton

_BLOG = NullBottleneckLog()


def set_kv_lengths(length_data, length, layer_start=0, layer_end=None):
    """CPU-side length tensor for all (or a range of) layers. No per-layer Python loop."""
    if length_data is None:
        return False
    n_layers = length_data.numel() // 2
    if layer_end is None:
        layer_end = n_layers
    length_data[layer_start * 2 : layer_end * 2] = int(length)
    return True


def sync_kv_bulk(src_shards, dst_shards, dst_len, start_idx, length, n_layers, layer_start=0):
    """One slice copy per GPU shard instead of 2*n_layers Python copy_ launches."""
    if length <= 0 or not src_shards or not dst_shards:
        return False
    end_idx = start_idx + length
    row0 = layer_start * 2
    row1 = n_layers * 2
    consumed = 0
    with torch.no_grad():
        for src, dst in zip(src_shards, dst_shards):
            shard_rows = src.shape[0]
            g0, g1 = consumed, consumed + shard_rows
            a = max(g0, row0)
            b = min(g1, row1)
            if b > a:
                sl = slice(a - consumed, b - consumed)
                dst[sl, :, :, start_idx:end_idx, :].copy_(
                    src[sl, :, :, start_idx:end_idx, :], non_blocking=True
                )
            consumed = g1
            if consumed >= row1:
                break
    if dst_len is not None:
        dst_len[row0:row1] = end_idx
    return True


def sync_kv_segment(source_kv, dest_kv, start_idx, length, num_layers, src_shards=None, dst_shards=None, dst_len=None, layer_start=0):
    if length <= 0:
        return
    if src_shards is not None and dst_shards is not None:
        if sync_kv_bulk(src_shards, dst_shards, dst_len, start_idx, length, num_layers, layer_start=layer_start):
            return
    end_idx = start_idx + length
    with torch.no_grad():
        for i in range(layer_start, num_layers):
            src_k, src_v = source_kv[i]
            dst_k, dst_v = dest_kv[i]
            dst_k.data[..., start_idx:end_idx, :].copy_(src_k.data[..., start_idx:end_idx, :], non_blocking=True)
            dst_v.data[..., start_idx:end_idx, :].copy_(src_v.data[..., start_idx:end_idx, :], non_blocking=True)
            dst_k.current_length.fill_(end_idx)
            dst_v.current_length.fill_(end_idx)


def truncate_kv_cache(past_key_values, length, length_data=None, layer_start=0, layer_end=None):
    if set_kv_lengths(length_data, length, layer_start=layer_start, layer_end=layer_end):
        return
    if past_key_values is None:
        return
    if hasattr(past_key_values, "current_length"):
        layers = [past_key_values]
    elif (
        isinstance(past_key_values, (list, tuple))
        and len(past_key_values) == 2
        and hasattr(past_key_values[0], "current_length")
    ):
        layers = [past_key_values]
    else:
        layers = past_key_values
    start = layer_start
    end = layer_end if layer_end is not None else len(layers)
    for i in range(start, end):
        layer_kv = layers[i]
        for cache_obj in layer_kv:
            if hasattr(cache_obj, "current_length"):
                cache_obj.current_length.fill_(length)


def _retrieve_2d(retrieve_indices):
    if retrieve_indices.dim() == 3:
        return retrieve_indices[0]
    return retrieve_indices


def _valid_path_indices(retrieve_indices, best, num_accepted, max_index):
    """Drop -1 padding; CUDA index_select does not accept negative indices."""
    retrieve_indices = _retrieve_2d(retrieve_indices)
    path_indices = retrieve_indices[best, :num_accepted]
    if path_indices.numel() == 0:
        return path_indices, 0
    # One small D2H instead of GPU .any()/.item() on the accepted path.
    path_list = path_indices.detach().cpu().tolist()
    out = []
    for p in path_list:
        p = int(p)
        if p < 0:
            break
        if max_index is not None and p > max_index:
            p = max_index
        out.append(p)
    if not out:
        return path_indices[:0], 0
    return (
        torch.tensor(out, device=path_indices.device, dtype=path_indices.dtype),
        len(out),
    )


def manual_kv_reorder_segment(
    kv_list,
    retrieve_indices,
    best_cand_idx,
    accept_len,
    start_len,
    num_layers,
    shards=None,
    path_indices=None,
):
    if accept_len <= 0:
        return
    if path_indices is None:
        path_indices, accept_len = _valid_path_indices(retrieve_indices, best_cand_idx, accept_len, None)
    if accept_len <= 0:
        return
    src_indices = start_len + path_indices
    dst_indices = torch.arange(start_len, start_len + accept_len, device=src_indices.device)
    row_end = num_layers * 2
    with torch.no_grad():
        if shards:
            consumed = 0
            for shard in shards:
                g0, g1 = consumed, consumed + shard.shape[0]
                a = max(g0, 0)
                b = min(g1, row_end)
                if b > a:
                    view = shard[a - consumed : b - consumed]
                    view.index_copy_(3, dst_indices, view.index_select(3, src_indices))
                consumed = g1
                if consumed >= row_end:
                    break
            return
        for i in range(num_layers):
            if kv_list[i] is None:
                continue
            k, v = kv_list[i]
            k.data.index_copy_(2, dst_indices, k.data.index_select(2, src_indices))
            v.data.index_copy_(2, dst_indices, v.data.index_select(2, src_indices))


def tree_decoding_with_exit(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices, exit_layer):
    extra_args = {"exit_at_layer": exit_layer}
    real_position_ids = tree_position_ids + input_ids.shape[1]
    outputs, tree_logits = model(
        input_ids=tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=real_position_ids,
        extra_args=extra_args,
    )
    retrieve_indices = _retrieve_2d(retrieve_indices)
    # Advanced indexing: -1 padding wraps to the last tree node.
    inter_logits = tree_logits[0, retrieve_indices]
    return inter_logits, outputs.last_hidden_state, tree_logits


def run_partial_forward_target(model, hidden_states, start_layer, past_key_values, current_pos, length_data=None):
    """Continue layers [start_layer, 32) from layer-L hidden states. Linear causal mask."""
    if not set_kv_lengths(length_data, current_pos, layer_start=start_layer):
        for i in range(start_layer, len(past_key_values)):
            for cache in past_key_values[i]:
                cache.current_length.fill_(current_pos)
    position_ids = torch.arange(
        current_pos, current_pos + hidden_states.shape[1], dtype=torch.long, device=hidden_states.device
    ).unsqueeze(0)
    model.set_tree_mask(None)
    outputs = model.base_model(
        input_ids=None,
        attention_mask=None,
        past_key_values=past_key_values,
        position_ids=position_ids,
        use_cache=True,
        start_at_layer=start_layer,
        intermediate_hidden_states=hidden_states,
    )
    return outputs.logits


def _as_1d_token_list(token_ids):
    if isinstance(token_ids, torch.Tensor):
        return token_ids.detach().reshape(-1).tolist()
    if isinstance(token_ids, (list, tuple)):
        out = []
        for x in token_ids:
            out.extend(_as_1d_token_list(x) if isinstance(x, (list, tuple, torch.Tensor)) else [int(x)])
        return out
    return [int(token_ids)]


def _argmax_token(logits):
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    return int(torch.argmax(logits, dim=-1).reshape(-1)[0].item())


def update_automaton(ac, token_ids, next_logits, max_breadth):
    if max_breadth <= 0 or ac is None:
        return
    tokens = _as_1d_token_list(token_ids)
    if not tokens:
        return
    if next_logits.dim() == 3:
        next_logits = next_logits.reshape(-1, next_logits.shape[-1])
    elif next_logits.dim() == 1:
        next_logits = next_logits.unsqueeze(0)
    adj_vectors = torch.topk(next_logits, k=max_breadth, dim=-1).indices
    n = min(len(tokens), adj_vectors.shape[0])
    ac.update(tokens[:n], adj_vectors[:n].tolist())


def insert_recent_ngrams(ac, seq_ids, ngram, num_new):
    """Insert the last `num_new` ngram windows along the accepted path."""
    if ngram <= 0 or num_new <= 0 or ac is None:
        return
    tokens = _as_1d_token_list(seq_ids)
    seq_len = len(tokens)
    for i in range(num_new):
        end = seq_len - i
        start = end - ngram
        if start < 0:
            break
        ac.insert(tokens[start:end])


def rewind_automaton_to_committed(ac, generated_tokens):
    """Restore retrieve cursor after remaining-layer rejects a speculative suffix.

    Inner-loop retrieve()/trans_tokens() walk the tentative buffer. If later
    layers reject a suffix, the cursor is past the committed token. Rewind is
    needed because retrieve already walked the tentative buffer. When remaining
    accepts the whole buffer, leave the cursor where retrieve+trans put it.
    """
    if ac is None or not hasattr(ac, "reset_to_root"):
        return
    ac.reset_to_root()
    if generated_tokens:
        ac.trans_tokens(generated_tokens)


def sequential_forward_tokens(model, token_ids, past_key_values, start_len, output_hidden_states=False, length_data=None):
    """Linear full-model forward of 1+ tokens. Writes KV from start_len."""
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    model.set_tree_mask(None)
    truncate_kv_cache(past_key_values, start_len, length_data=length_data)
    return model.base_model(
        input_ids=token_ids,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=output_hidden_states,
    )


def _eos_truncated_accept(candidates, best_cand_idx, accept_len, eos_id):
    if isinstance(best_cand_idx, torch.Tensor) or isinstance(accept_len, torch.Tensor):
        device = best_cand_idx.device if isinstance(best_cand_idx, torch.Tensor) else accept_len.device
        packed = torch.stack(
            (
                torch.as_tensor(best_cand_idx, device=device, dtype=torch.long).reshape(()),
                torch.as_tensor(accept_len, device=device, dtype=torch.long).reshape(()),
            )
        )
        best, accept_len = (int(x) for x in packed.detach().cpu().tolist())
    else:
        best = int(best_cand_idx)
        accept_len = int(accept_len)
    num_accepted = accept_len + 1
    if eos_id is not None and num_accepted > 0:
        path_list = candidates[best, :num_accepted].detach().cpu().tolist()
        for i, tok in enumerate(path_list):
            if int(tok) == eos_id:
                accept_len = i
                num_accepted = i + 1
                break
    return best, accept_len, num_accepted


def accept_early_layer_tree(
    model,
    tokenizer,
    ac,
    draft_logit,
    inter_input_ids,
    inter_kv,
    inter_kv_list,
    max_num_draft,
    max_tokens,
    repetition_penalty,
    top_p,
    temperature,
    exit_layer,
    eos_id,
    max_breadth,
    ngram,
    tree_idx=0,
    from_copy_logit=True,
):
    """One 64-wide draft tree, verified at layers 0..L-1.

    Returns accepted tokens, their layer-L hidden states, and the layer-L
    correction/bonus logit at the last accepted position (seeds the next tree).
    Remaining later re-checks this chain linearly; early-layer reject already
    drops tokens later layers would reject.
    """
    if inter_input_ids.shape[1] + max_num_draft >= max_tokens:
        return [], None, draft_logit

    use_cuda_times = getattr(_BLOG, "cuda_times", False)
    t0 = time.perf_counter()
    candidates, tree_candidates, tree_attn_mask, tree_position_ids, retrieve_indices = generate_draft_tree(
        logits=draft_logit,
        ac=ac,
        pad_token_id=tokenizer.pad_token_id,
        input_ids=inter_input_ids,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        temperature=temperature,
        max_num_draft=max_num_draft,
        device=model.device,
    )
    retrieve_ms = (time.perf_counter() - t0) * 1000.0

    if use_cuda_times:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    model.set_tree_mask(tree_attn_mask[None, None, :])
    inter_logits, inter_hidden, tree_logits = tree_decoding_with_exit(
        model,
        tree_candidates[None, :],
        inter_kv,
        tree_position_ids[None, :],
        inter_input_ids,
        retrieve_indices,
        exit_layer=exit_layer,
    )
    if use_cuda_times:
        torch.cuda.synchronize()
    l1_fwd_ms = (time.perf_counter() - t1) * 1000.0

    best_cand_idx, accept_len = evaluate_posterior(inter_logits, candidates, temperature, top_p)
    best, accept_len, num_accepted = _eos_truncated_accept(candidates, best_cand_idx, accept_len, eos_id)
    path_indices, num_accepted = _valid_path_indices(
        retrieve_indices, best, num_accepted, inter_hidden.shape[1] - 1
    )
    # Copy-logit / correction root is already the previous target (or L-layer)
    # decision. Early-exit must not drop it; always keep the root.
    if num_accepted <= 0:
        path_indices, num_accepted = _valid_path_indices(
            retrieve_indices, 0, 1, inter_hidden.shape[1] - 1
        )
        best = 0
    if num_accepted <= 0:
        _BLOG.add_l1_tree(
            tree_idx=tree_idx,
            n_nodes=int(tree_candidates.numel()),
            n_paths=int(candidates.shape[0]) if candidates.dim() > 1 else 1,
            l1_accept=0,
            retrieve_ms=retrieve_ms,
            l1_fwd_ms=l1_fwd_ms,
            l1_update_ms=0.0,
            from_copy_logit=from_copy_logit,
        )
        return [], None, draft_logit
    accept_len = num_accepted - 1
    path_hs = inter_hidden.index_select(1, path_indices)
    new_tokens = candidates[best, :num_accepted]

    start_pos_in_cache = inter_input_ids.shape[1]
    manual_kv_reorder_segment(
        inter_kv_list,
        retrieve_indices,
        best,
        num_accepted,
        start_pos_in_cache,
        exit_layer,
        shards=getattr(model, "inter_kv_data_list", None),
        path_indices=path_indices,
    )
    truncate_kv_cache(
        inter_kv,
        start_pos_in_cache + num_accepted,
        length_data=getattr(model, "inter_len_data", None),
        layer_end=exit_layer,
    )

    # Layer-L logit after the last accepted token: correction if the tree
    # stopped early, bonus continuation otherwise. Next draft tree is grown
    # from this logit (its argmax becomes the next tree root).
    correction_logit = inter_logits[best, num_accepted - 1 : num_accepted, :]
    # Adjacency from every tree node, ngrams from the accepted
    # suffix. retrieve() already consumed the tree root.
    t2 = time.perf_counter()
    if max_breadth > 0:
        adj_topk = torch.topk(tree_logits, k=max_breadth, dim=-1)[1][0]
        tree_tokens = tree_candidates.tolist() if tree_candidates.dim() == 1 else tree_candidates[0].tolist()
        ac.update(tree_tokens, adj_topk.tolist())
    insert_recent_ngrams(ac, torch.cat([inter_input_ids[0], new_tokens], dim=0), ngram, num_accepted)
    if ngram > 0 and num_accepted > 1:
        ac.trans_tokens(_as_1d_token_list(new_tokens[1:]))
    l1_update_ms = (time.perf_counter() - t2) * 1000.0
    _BLOG.add_l1_tree(
        tree_idx=tree_idx,
        n_nodes=int(tree_candidates.numel()),
        n_paths=int(candidates.shape[0]) if candidates.dim() > 1 else 1,
        l1_accept=int(num_accepted),
        retrieve_ms=retrieve_ms,
        l1_fwd_ms=l1_fwd_ms,
        l1_update_ms=l1_update_ms,
        from_copy_logit=from_copy_logit,
    )
    return new_tokens.tolist(), path_hs, correction_logit


def commit_one_token(
    model,
    token_id,
    full_kv,
    inter_kv,
    pos,
    num_layers,
    ac,
    max_breadth,
    ngram,
    context_ids,
    intermediate_layer,
):
    new_token = torch.tensor([[token_id]], device=model.device, dtype=torch.long)
    out = sequential_forward_tokens(
        model,
        new_token,
        full_kv,
        pos,
        output_hidden_states=True,
        length_data=getattr(model, "full_len_data", None),
    )
    last_logit = out.logits[:, -1, :].unsqueeze(1)
    last_committed_hs = out.hidden_states[intermediate_layer][:, -1:, :]
    sync_kv_segment(
        full_kv,
        inter_kv,
        pos,
        1,
        num_layers,
        src_shards=getattr(model, "full_kv_data_list", None),
        dst_shards=getattr(model, "inter_kv_data_list", None),
        dst_len=getattr(model, "inter_len_data", None),
    )
    update_automaton(ac, new_token, out.logits[:, -1, :], max_breadth)
    insert_recent_ngrams(ac, torch.cat([context_ids[0], new_token[0]], dim=0), ngram, 1)
    if ngram > 0:
        ac.trans_tokens([token_id])
    return new_token, last_logit, last_committed_hs


def echo_forward(
    inputs,
    model,
    tokenizer,
    max_new_tokens,
    max_tokens,
    temperature,
    top_p,
    ac,
    max_num_draft,
    max_breadth,
    intermediate_layer,
    verification_threshold,
    ngram=10,
    repetition_penalty=1.0,
    profile_meta=None,
):
    INTERMEDIATE_LAYER = intermediate_layer
    VERIFICATION_THRESHOLD = verification_threshold
    num_layers = len(model.base_model.model.layers)

    if max_tokens is None:
        max_tokens = model.base_model.config.max_position_embeddings

    prompt_ids = inputs.input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    if not hasattr(model, "full_kv"):
        full_kv, full_kv_data_list, full_len_data = initialize_past_key_values(model.base_model)
        inter_kv, inter_kv_data_list, inter_len_data = initialize_past_key_values(model.base_model)
        model.full_kv = full_kv
        model.inter_kv = inter_kv
        model.full_kv_data_list = full_kv_data_list
        model.inter_kv_data_list = inter_kv_data_list
        model.full_len_data = full_len_data
        model.inter_len_data = inter_len_data
    else:
        full_kv = model.full_kv
        inter_kv = model.inter_kv
        full_kv_data_list = model.full_kv_data_list
        inter_kv_data_list = model.inter_kv_data_list
        full_len_data = model.full_len_data
        inter_len_data = model.inter_len_data
        full_len_data.zero_()
        inter_len_data.zero_()

    full_kv_data_list = model.full_kv_data_list
    inter_kv_data_list = model.inter_kv_data_list
    full_len_data = model.full_len_data
    inter_len_data = model.inter_len_data

    inter_kv_list = list(inter_kv)

    committed_ids = prompt_ids.clone()
    generated_tokens = []
    last_token_id = None
    accept_length_list = []
    step_count = 0
    _BLOG.begin_sample(profile_meta)

    model.set_tree_mask(None)

    # Keep the retrieve cursor across samples (warmup leftover included).
    # reset_to_root here is why ECHO's first-turn retrieve is weaker.

    if ngram > 0:
        for i in range(prompt_len):
            end = i + ngram
            if end > prompt_ids.size(1):
                break
            ac.insert(prompt_ids[0, i:end].tolist())
        ac.build()

    # Warmup logits must come from base_model (same head as remaining-layer
    # skip_logits). EchoModel.forward applies an extra RMSNorm and would make
    # argmax(copy-logit) disagree with skip_logits[0], so remaining rejects the
    # bonus token every step and the outer loop never commits.
    outputs = model.base_model(
        input_ids=committed_ids,
        past_key_values=full_kv,
        use_cache=True,
        output_hidden_states=True,
    )
    prompt_logits = outputs.logits
    if max_breadth > 0:
        adj_topk = torch.topk(prompt_logits, k=max_breadth, dim=-1)[1][0]
        ac.update(committed_ids[0].tolist(), adj_topk.tolist())
    sync_kv_segment(
        full_kv,
        inter_kv,
        0,
        committed_ids.shape[1],
        num_layers,
        src_shards=full_kv_data_list,
        dst_shards=inter_kv_data_list,
        dst_len=inter_len_data,
    )

    last_logit_for_draft = prompt_logits[:, -1:, :]
    last_committed_hs = outputs.hidden_states[INTERMEDIATE_LAYER][:, -1:, :]

    eos_id = tokenizer.eos_token_id

    while (committed_ids.shape[1] - prompt_len) < max_new_tokens:
        step_count += 1
        if step_count > max_new_tokens + 2:
            break
        pos = committed_ids.shape[1]
        frozen_bonus = _argmax_token(last_logit_for_draft)
        _BLOG.begin_outer(pos, frozen_bonus, VERIFICATION_THRESHOLD)

        inter_accepted_token_ids = []
        hidden_states_buffer = []
        inter_input_ids = committed_ids
        truncate_kv_cache(inter_kv, inter_input_ids.shape[1], length_data=inter_len_data)

        # First tree: remaining copy-logit. Later trees: L1 bonus/correction
        # logit at the last accepted node (tree-root index).
        draft_logit = last_logit_for_draft
        l1_tree_idx = 0
        while (
            len(inter_accepted_token_ids) < VERIFICATION_THRESHOLD
            and (inter_input_ids.shape[1] - prompt_len) < max_new_tokens
            and inter_input_ids.shape[1] + max_num_draft < max_tokens
            and (not inter_accepted_token_ids or inter_accepted_token_ids[-1] != eos_id)
        ):
            new_tokens, path_hs, draft_logit = accept_early_layer_tree(
                model,
                tokenizer,
                ac,
                draft_logit,
                inter_input_ids,
                inter_kv,
                inter_kv_list,
                max_num_draft,
                max_tokens,
                repetition_penalty,
                top_p,
                temperature,
                INTERMEDIATE_LAYER,
                eos_id,
                max_breadth,
                ngram,
                tree_idx=l1_tree_idx,
                from_copy_logit=(l1_tree_idx == 0),
            )
            l1_tree_idx += 1
            if not new_tokens:
                break
            hidden_states_buffer.append(path_hs)
            inter_input_ids = torch.cat(
                [inter_input_ids, torch.tensor([new_tokens], device=model.device, dtype=torch.long)],
                dim=1,
            )
            inter_accepted_token_ids.extend(new_tokens)
            if eos_id in new_tokens:
                break

        _BLOG.mark_l1_block_done()
        model.set_tree_mask(None)
        remaining = max_new_tokens - (committed_ids.shape[1] - prompt_len)
        if remaining <= 0:
            _BLOG.end_outer(
                buffer_len=len(inter_accepted_token_ids),
                remaining_accept=0,
                frozen_forced=False,
                rewind=False,
                empty_l1=len(inter_accepted_token_ids) == 0,
                commit_one=False,
                reject_at=None,
            )
            break
        if hidden_states_buffer:
            hs_buffer_stacked = torch.cat(hidden_states_buffer, dim=1)
        else:
            hs_buffer_stacked = None
        if len(inter_accepted_token_ids) > remaining:
            inter_accepted_token_ids = inter_accepted_token_ids[:remaining]
            hs_buffer_stacked = hs_buffer_stacked[:, :remaining, :]
            truncate_kv_cache(inter_kv, pos + remaining, length_data=inter_len_data)

        if len(inter_accepted_token_ids) == 0:
            true_first = frozen_bonus

            def _do_commit():
                return commit_one_token(
                    model,
                    true_first,
                    full_kv,
                    inter_kv,
                    pos,
                    num_layers,
                    ac,
                    max_breadth,
                    ngram,
                    committed_ids,
                    INTERMEDIATE_LAYER,
                )

            new_token, last_logit_for_draft, last_committed_hs = _BLOG.time_commit(_do_commit)
            committed_ids = torch.cat([committed_ids, new_token], dim=1)
            generated_tokens.append(int(true_first))
            last_token_id = int(true_first)
            _BLOG.time_rewind(lambda: rewind_automaton_to_committed(ac, generated_tokens))
            accept_length_list.append(0)
            _BLOG.end_outer(
                buffer_len=0,
                remaining_accept=0,
                frozen_forced=False,
                rewind=True,
                empty_l1=True,
                commit_one=True,
                reject_at=0,
            )
        else:
            buffer_tensor = torch.tensor(inter_accepted_token_ids, device=model.device, dtype=torch.long)
            buffered_len = buffer_tensor.numel()
            hs_buffer_stacked = hs_buffer_stacked[:, :buffered_len, :]

            # Stitch last committed layer-L HS so remaining layers recompute the
            # bonus position: logits[:, i] verifies buffer[i], logits[:, k] is
            # the copy-logit after k accepted tokens.
            set_kv_lengths(full_len_data, pos, layer_end=INTERMEDIATE_LAYER)
            set_kv_lengths(full_len_data, pos - 1, layer_start=INTERMEDIATE_LAYER)
            stitched_hs = torch.cat([last_committed_hs, hs_buffer_stacked], dim=1)
            skip_logits = _BLOG.time_remaining(
                lambda: run_partial_forward_target(
                    model,
                    stitched_hs,
                    INTERMEDIATE_LAYER,
                    full_kv,
                    pos - 1,
                    length_data=full_len_data,
                )
            )

            greedy_match = skip_logits[0, :buffered_len].argmax(dim=-1) == buffer_tensor
            # buffer[0] is already a Python int; do not .item() it off GPU.
            root_is_bonus = inter_accepted_token_ids[0] == frozen_bonus
            if root_is_bonus:
                matched = greedy_match.clone()
                matched[0] = True
            else:
                matched = greedy_match
            prefix_hits = matched.to(torch.int32).cumprod(dim=0).sum()
            if root_is_bonus:
                # One D2H for accept length + whether remaining would have dropped the root.
                stats = torch.stack((prefix_hits, (~greedy_match[0]).to(prefix_hits.dtype)))
                accepted_count, frozen_i = stats.detach().cpu().tolist()
                accepted_count = int(accepted_count)
                frozen_forced = bool(frozen_i)
            else:
                accepted_count = int(prefix_hits.item())
                frozen_forced = False
            bonus_idx = min(accepted_count, skip_logits.shape[1] - 1)
            last_logit_for_draft = skip_logits[:, bonus_idx, :].unsqueeze(1)
            commit_one = False

            if accepted_count > 0:
                matched_tokens = buffer_tensor[:accepted_count]
                committed_ids = torch.cat([committed_ids, matched_tokens.unsqueeze(0)], dim=1)
                generated_tokens.extend(inter_accepted_token_ids[:accepted_count])
                last_token_id = inter_accepted_token_ids[accepted_count - 1]
                last_committed_hs = hs_buffer_stacked[:, accepted_count - 1 : accepted_count, :]
                _BLOG.time_kv(
                    lambda: sync_kv_segment(
                        inter_kv,
                        full_kv,
                        pos,
                        accepted_count,
                        INTERMEDIATE_LAYER,
                        src_shards=inter_kv_data_list,
                        dst_shards=full_kv_data_list,
                        dst_len=full_len_data,
                    )
                )
                ac_tokens = torch.cat([committed_ids[0, pos - 1 : pos], matched_tokens], dim=0)
                ac_logits = skip_logits[0, : min(accepted_count + 1, skip_logits.shape[1]), :]
                update_automaton(ac, ac_tokens[: ac_logits.shape[0]], ac_logits, max_breadth)
            else:
                true_first = _argmax_token(skip_logits[:, :1, :])
                commit_one = True

                def _do_commit_corr():
                    return commit_one_token(
                        model,
                        true_first,
                        full_kv,
                        inter_kv,
                        pos,
                        num_layers,
                        ac,
                        max_breadth,
                        ngram,
                        committed_ids,
                        INTERMEDIATE_LAYER,
                    )

                new_token, last_logit_for_draft, last_committed_hs = _BLOG.time_commit(_do_commit_corr)
                committed_ids = torch.cat([committed_ids, new_token], dim=1)
                generated_tokens.append(int(true_first))
                last_token_id = int(true_first)
            did_rewind = accepted_count < buffered_len
            if did_rewind:
                _BLOG.time_rewind(lambda: rewind_automaton_to_committed(ac, generated_tokens))
            accept_length_list.append(accepted_count)
            reject_at = None if accepted_count >= buffered_len else accepted_count
            _BLOG.end_outer(
                buffer_len=buffered_len,
                remaining_accept=accepted_count,
                frozen_forced=frozen_forced,
                rewind=did_rewind,
                empty_l1=False,
                commit_one=commit_one,
                reject_at=reject_at,
            )

        truncate_kv_cache(full_kv, committed_ids.shape[1], length_data=full_len_data)
        truncate_kv_cache(inter_kv, committed_ids.shape[1], length_data=inter_len_data)

        if last_token_id == eos_id:
            break
        if committed_ids.shape[1] >= max_tokens:
            break

    model.set_tree_mask(None)
    accept_length_list = [int(x) if not isinstance(x, torch.Tensor) else int(x.item()) for x in accept_length_list]
    new_token_count = committed_ids.shape[1] - prompt_len
    _BLOG.end_sample(new_token_count, step_count)
    return committed_ids, new_token_count, step_count, accept_length_list


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
    parser.add_argument("--intermediate-layer", type=int, default=12)
    parser.add_argument("--verification-threshold", type=int, default=5)
    parser.add_argument("--ngram", type=int, default=10)
    parser.add_argument("--max-num-draft", type=int, default=64)
    parser.add_argument("--max-breadth", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=30000)
    parser.add_argument(
        "--bottleneck-log",
        type=str,
        default=None,
        help="Sidecar JSONL for skip-layer bottleneck stats. Default: <answer-file> with suffix .bottleneck.jsonl",
    )
    parser.add_argument(
        "--no-bottleneck-log",
        action="store_true",
        help="Disable sidecar logging (18-25-22 path, no extra CUDA sync).",
    )
    parser.add_argument(
        "--bottleneck-counts-only",
        action="store_true",
        help="Log tree/reject counts without extra CUDA synchronize so tok/s stays closer to a normal run.",
    )
    parser.add_argument(
        "--bottleneck-cuda-times",
        action="store_true",
        help="Insert torch.cuda.synchronize around timed regions (slows tok/s; for timing only).",
    )

    args = parser.parse_args()

    args.model_id = f"{args.model_id}-echo-L{args.intermediate_layer}-T{args.verification_threshold}"
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

    ac = Automaton(args.max_nodes)
    ac.init_logits(tokenizer.vocab_size, args.max_breadth)

    if args.max_tokens is None:
        args.max_tokens = model.base_model.config.max_position_embeddings

    blog_path = None
    if args.no_bottleneck_log:
        _BLOG = NullBottleneckLog()
    else:
        blog_path = args.bottleneck_log
        if not blog_path:
            if args.answer_file.endswith(".jsonl"):
                blog_path = args.answer_file[:-6] + ".bottleneck.jsonl"
            else:
                blog_path = args.answer_file + ".bottleneck.jsonl"
        _BLOG = EchoBottleneckLog(
            blog_path,
            cuda_times=bool(args.bottleneck_cuda_times) and not args.bottleneck_counts_only,
        )
        print(
            f"Bottleneck log: {blog_path}  cuda_times={_BLOG.cuda_times}  "
            "(profiled tok/s is not comparable to 18-25-22 if cuda_times=True)"
        )

    try:
        run_eval(
            model=model,
            tokenizer=tokenizer,
            forward_func=echo_forward,
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
            ac=ac,
            ngram=args.ngram,
            max_num_draft=args.max_num_draft,
            max_breadth=args.max_breadth,
            intermediate_layer=args.intermediate_layer,
            verification_threshold=args.verification_threshold,
        )
        reorg_answer_file(args.answer_file)
        if blog_path:
            recs = load_records(blog_path, drop_warmup=True)
            print("\n===== ECHO bottleneck summary (warmup dropped) =====")
            print(format_summary(summarize(recs)))
    finally:
        _BLOG.close()
