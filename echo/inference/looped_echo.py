"""Hierarchical ECHO decoding: shallow drafts, target verification, commit."""

from contextlib import closing
from dataclasses import dataclass

import torch

from .looped_common import (
    GenerationResult,
    GenerationStats,
    prepare_input,
    synchronized_time,
)
from .looped_stop import GenerationStops


@dataclass(frozen=True)
class EchoConfig:
    verification_threshold: int = 5
    max_num_draft: int = 64
    max_breadth: int = 8
    ngram: int = 10
    automaton_capacity: int = 30000

    def __post_init__(self):
        if (
            min(
                self.verification_threshold,
                self.max_num_draft,
                self.max_breadth,
                self.automaton_capacity,
            )
            < 1
        ):
            raise ValueError(
                "Verification threshold, draft nodes, breadth and automaton capacity must be positive"
            )
        if self.ngram < 0:
            raise ValueError("ngram must be non-negative")


@dataclass
class DraftTree:
    candidates: torch.Tensor
    tree_tokens: torch.Tensor
    tree_mask: torch.Tensor
    position_ids: torch.Tensor
    retrieve_indices: torch.Tensor


@dataclass
class DraftBlock:
    tokens: torch.Tensor
    hidden_states: torch.Tensor


def _pad(rows: list[list[int]], value: int) -> list[list[int]]:
    width = max(len(row) for row in rows)
    return [row + [value] * (width - len(row)) for row in rows]


def build_draft_tree(
    logits: torch.Tensor,
    automaton,
    *,
    pad_token_id: int,
    max_num_draft: int,
    device: torch.device,
    max_depth: int | None = None,
) -> DraftTree:
    """Retrieve ECHO's shared-prefix multi-token tree from its automaton."""
    if logits.ndim == 3:
        logits = logits[:, -1, :]
    root_token = int(logits.argmax(dim=-1)[0].item())
    buf = automaton.retrieve(root_token, max_num_draft)
    candidates = [list(row) for row in buf.candidates]
    tree_tokens = list(buf.tree_candidates)
    tree_mask = [list(row) for row in buf.attn_mask]
    position_ids = list(buf.position_ids)
    retrieve_indices = [list(row) for row in buf.retrieve_indices]

    # A root-only trie cannot produce parent logits for posterior checking.
    # Match the original ECHO fallback by adding one harmless padded child.
    if len(tree_tokens) <= 1 and max_num_draft > 1 and max_depth != 1:
        candidates = [[root_token, pad_token_id]]
        tree_tokens = [root_token, pad_token_id]
        tree_mask = [[1, 0], [1, 1]]
        position_ids = [0, 1]
        retrieve_indices = [[0, 1]]

    if max_depth is not None:
        if max_depth < 1:
            raise ValueError("The draft must have room for its root")
        keep = [i for i, position in enumerate(position_ids) if position < max_depth]
        mapping = {old: new for new, old in enumerate(keep)}
        paths = []
        for row in retrieve_indices:
            path = tuple(i for i in row if i in mapping)
            if path and path not in paths:
                paths.append(path)
        candidates = [[tree_tokens[i] for i in path] for path in paths]
        retrieve_indices = [[mapping[i] for i in path] for path in paths]
        tree_tokens = [tree_tokens[i] for i in keep]
        tree_mask = [[tree_mask[i][j] for j in keep] for i in keep]
        position_ids = [position_ids[i] for i in keep]

    candidates = _pad(candidates, pad_token_id)
    retrieve_indices = _pad(retrieve_indices, -1)
    return DraftTree(
        candidates=torch.tensor(candidates, dtype=torch.long, device=device),
        tree_tokens=torch.tensor(tree_tokens, dtype=torch.long, device=device),
        tree_mask=torch.tensor(tree_mask, dtype=torch.bool, device=device),
        position_ids=torch.tensor(position_ids, dtype=torch.long, device=device),
        retrieve_indices=torch.tensor(
            retrieve_indices, dtype=torch.long, device=device
        ),
    )


def select_greedy_path(
    path_logits: torch.Tensor, candidates: torch.Tensor
) -> tuple[int, int]:
    """Return ``(best_path, accepted_tokens_including_root)``."""
    child_matches = candidates[:, 1:] == path_logits[:, :-1, :].argmax(dim=-1)
    lengths = child_matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    best = int(lengths.argmax().item())
    return best, int(lengths[best].item()) + 1


def _valid_path_indices(
    retrieve_indices: torch.Tensor, best: int, count: int
) -> torch.Tensor:
    values = retrieve_indices[best, :count]
    return values[values >= 0]


def _update_adjacency(
    automaton, tokens: torch.Tensor, logits: torch.Tensor, top_k: int
) -> None:
    if top_k <= 0 or tokens.numel() == 0:
        return
    flat_logits = logits.reshape(-1, logits.shape[-1])
    count = min(tokens.numel(), flat_logits.shape[0])
    adjacency = flat_logits[:count].topk(top_k, dim=-1).indices.detach().cpu().tolist()
    automaton.update(tokens.reshape(-1)[:count].detach().cpu().tolist(), adjacency)


def _insert_recent_ngrams(
    automaton, ids: torch.Tensor, ngram: int, num_new: int
) -> None:
    if ngram <= 0:
        return
    values = ids.reshape(-1).detach().cpu().tolist()
    for offset in range(num_new):
        end = len(values) - offset
        start = end - ngram
        if start < 0:
            break
        automaton.insert(values[start:end])


def _rewind_automaton(automaton, ids: torch.Tensor) -> None:
    automaton.reset_to_root()
    automaton.trans_tokens(ids.reshape(-1).detach().cpu().tolist())


class LoopedEchoGenerator:
    """ECHO's automaton and decoding policy, shared by both execution backends."""

    def __init__(
        self, backend, tokenizer, automaton=None, *, config: EchoConfig | None = None
    ):
        self.backend = backend
        self.tokenizer = tokenizer
        self.config = config or EchoConfig()
        if self.config.max_breadth > backend.vocab_size:
            raise ValueError("Draft breadth exceeds the target vocabulary")
        if automaton is None:
            from echo.automaton import Automaton

            automaton = Automaton(self.config.automaton_capacity)
            automaton.init_logits(backend.vocab_size, self.config.max_breadth)
        self.automaton = automaton

    def _prepare_automaton(self, input_ids, prompt_logits):
        values = input_ids[0].detach().cpu().tolist()
        ngram = self.config.ngram
        if ngram > 0:
            for start in range(max(0, len(values) - ngram + 1)):
                self.automaton.insert(values[start : start + ngram])
        self.automaton.build()
        _update_adjacency(
            self.automaton, input_ids[0], prompt_logits[0], self.config.max_breadth
        )

    def _draft(self, committed, target_logits, cache, remaining, eos, stats):
        """Extend the shallow frontier, retaining the accepted path's KV and state."""
        config = self.config
        prefix = committed.shape[1]
        tokens, hidden = [], []
        count = 0
        logits = target_logits
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        while count < min(config.verification_threshold, remaining):
            tree = build_draft_tree(
                logits,
                self.automaton,
                pad_token_id=pad_id,
                max_num_draft=config.max_num_draft,
                device=self.backend.device,
                max_depth=remaining - count,
            )
            tree_prefix = prefix + count
            output = self.backend.forward_shallow_tree(
                tree.tree_tokens, tree.tree_mask, tree.position_ids, tree_prefix, cache
            )
            path_logits = output.logits[0, tree.retrieve_indices.clamp_min(0)]
            best, accepted = select_greedy_path(path_logits, tree.candidates)
            path = _valid_path_indices(
                tree.retrieve_indices, best, min(accepted, remaining - count)
            )
            accepted = path.numel()
            if accepted == 0:
                cache.truncate(
                    tree_prefix, slot_start=0, slot_end=self.backend.shallow_slot_end
                )
                break
            cache.compact_tree(
                tree_prefix, path, slot_start=0, slot_end=self.backend.shallow_slot_end
            )
            path_tokens = tree.candidates[best, :accepted]
            tokens.append(path_tokens)
            hidden.append(output.hidden_states.index_select(1, path))
            count += accepted
            stats.draft_trees += 1
            stats.drafted_nodes += tree.tree_tokens.numel()
            stats.shallow_accepted += accepted

            _update_adjacency(
                self.automaton, tree.tree_tokens, output.logits[0], config.max_breadth
            )
            tentative = torch.cat((committed[0], *tokens))
            _insert_recent_ngrams(self.automaton, tentative, config.ngram, accepted)
            # retrieve() consumed the root; advance over the rest of the path.
            if accepted > 1:
                self.automaton.trans_tokens(path_tokens[1:].detach().cpu().tolist())
            logits = path_logits[best, accepted - 1 : accepted]
            if eos.numel() and torch.isin(path_tokens, eos).any().item():
                break

        if not tokens:
            return None
        block = DraftBlock(torch.cat(tokens), torch.cat(hidden, dim=1))
        # Keep the first EOS for final verification; discard later shallow tokens.
        eos_positions = torch.isin(block.tokens, eos).nonzero(as_tuple=False)
        if eos_positions.numel():
            keep = int(eos_positions[0, 0]) + 1
            block = DraftBlock(block.tokens[:keep], block.hidden_states[:, :keep])
            cache.truncate(
                prefix + keep, slot_start=0, slot_end=self.backend.shallow_slot_end
            )
        return block

    def _verify(self, block, target_logits, prefix, cache):
        """The frozen target root is safe; verify the following shallow tokens."""
        if block.tokens[0].item() != target_logits.argmax(dim=-1).item():
            raise RuntimeError("First ECHO root diverged from the frozen target logit")
        output = self.backend.forward_deep(block.hidden_states, prefix, cache)
        matches = output.logits[0, :-1].argmax(dim=-1) == block.tokens[1:]
        accepted = 1 + int(matches.to(torch.int32).cumprod(dim=0).sum())
        return output, accepted

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        *,
        max_new_tokens=128,
        stop_token_sequences=(),
        eos_token_ids=None,
        stop_strings=(),
        include_stop_tokens=False,
    ) -> GenerationResult:
        input_ids = prepare_input(input_ids, self.backend, max_new_tokens)
        prompt_length = input_ids.shape[1]
        stops = GenerationStops(
            self.tokenizer,
            eos_token_ids=eos_token_ids,
            token_sequences=stop_token_sequences,
            strings=stop_strings,
        )
        eos = torch.tensor(
            sorted(stops.eos), dtype=torch.long, device=self.backend.device
        )
        stats = GenerationStats()
        finish_reason = dict(type="length", length=max_new_tokens)
        prefill_started = synchronized_time(self.backend.device)
        with closing(self.backend.new_cache()) as cache:
            _, output = self.backend.prefill(input_ids, cache=cache)
            committed = input_ids.clone()
            target_logits = output.logits[:, -1:, :]
            self._prepare_automaton(committed, output.logits)
            prefill_seconds = synchronized_time(self.backend.device) - prefill_started
            decode_started = synchronized_time(self.backend.device)

            while committed.shape[1] - prompt_length < max_new_tokens:
                stats.outer_steps += 1
                prefix = committed.shape[1]
                remaining = max_new_tokens - (prefix - prompt_length)
                block = self._draft(
                    committed, target_logits, cache, remaining, eos, stats
                )
                if block is None:
                    tokens = target_logits[:, -1].argmax(dim=-1)
                    output = self.backend.forward_token_ids(tokens, prefix, cache)
                    accepted = 1
                    stats.full_token_calls += 1
                    stats.corrections += 1
                else:
                    output, accepted = self._verify(block, target_logits, prefix, cache)
                    tokens = block.tokens[:accepted]
                    stats.deep_calls += 1

                committed = torch.cat((committed, tokens[None]), dim=1)
                stop = stops.observe(committed[0, prompt_length:])
                if block is not None:
                    count = (
                        accepted
                        if stop is None
                        else stop.generated_count - (prefix - prompt_length)
                    )
                    stats.target_accepted += count
                    stats.target_accept_lengths.append(count)
                if stop is not None:
                    keep = (
                        stop.generated_count
                        if include_stop_tokens
                        else stop.visible_count
                    )
                    committed = committed[:, : prompt_length + keep]
                    finish_reason = stop.reason
                    break

                _update_adjacency(
                    self.automaton,
                    tokens,
                    output.logits[0, :accepted],
                    self.config.max_breadth,
                )
                target_logits = output.logits[:, accepted - 1 : accepted]
                if block is None:
                    _insert_recent_ngrams(
                        self.automaton, committed[0], self.config.ngram, 1
                    )
                    _rewind_automaton(self.automaton, committed[:, prompt_length:])
                elif accepted < block.tokens.numel():
                    # Defer the target correction to the next tree's frozen root.
                    # It then shares that tree's deep verification.
                    cache.truncate(committed.shape[1])
                    stats.deferred_corrections += 1
                    _rewind_automaton(self.automaton, committed[:, prompt_length:])

            decode_seconds = synchronized_time(self.backend.device) - decode_started
        stats.stage_timings = dict(
            prefill_seconds=prefill_seconds, decode_seconds=decode_seconds
        )
        return GenerationResult(committed, stats, finish_reason)
