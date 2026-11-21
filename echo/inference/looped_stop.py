"""EOS and decoded-text stops shared by the paired ECHO/AR harness."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StopDecision:
    generated_count: int
    visible_count: int
    reason: dict


class GenerationStops:
    def __init__(
        self, tokenizer, *, eos_token_ids=None, token_sequences=(), strings=()
    ):
        self.tokenizer = tokenizer
        if eos_token_ids is None:
            eos_token_ids = (
                () if tokenizer.eos_token_id is None else (tokenizer.eos_token_id,)
            )
        self.eos = set(int(x) for x in eos_token_ids)
        self.sequences = [list(x) for x in token_sequences if x]
        self.strings = [x for x in strings if x]
        self.tokens = []
        self.string_overlap = max(
            64, 2 * max((len(x) for x in self.strings), default=0) + 8
        )

    def _decode(self, ids):
        return self.tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def observe(self, completion):
        if not (self.eos or self.sequences or self.strings):
            return None
        previous = len(self.tokens)
        values = completion.reshape(-1)
        self.tokens.extend(values[previous:].detach().cpu().tolist())
        decisions = []
        for i in range(previous, len(self.tokens)):
            if self.tokens[i] in self.eos:
                decisions.append(
                    StopDecision(
                        i + 1, i + 1, dict(type="stop", matched=self.tokens[i])
                    )
                )
                break
        for stop in self.sequences:
            for i in range(
                max(0, previous - len(stop) + 1), len(self.tokens) - len(stop) + 1
            ):
                if self.tokens[i : i + len(stop)] == stop:
                    decisions.append(
                        StopDecision(i + len(stop), i, dict(type="stop", matched=stop))
                    )
                    break
        if self.strings:
            start = max(0, previous - self.string_overlap)
            tail = self.tokens[start:]
            text = self._decode(tail)
            for stop in self.strings:
                found = text.find(stop)
                if found < 0:
                    continue
                # Find the first token completing the string, including stops
                # whose boundary lies inside a BPE token or crosses a block.
                low, high = max(1, previous - start), len(tail)
                while low < high:
                    middle = (low + high) // 2
                    if stop in self._decode(tail[:middle]):
                        high = middle
                    else:
                        low = middle + 1
                end = start + low
                # Token sequences are the legacy API's exact stripping path;
                # string stops retain the terminal token and trim visible text.
                decisions.append(
                    StopDecision(end, end, dict(type="stop", matched=stop))
                )
        return min(decisions, key=lambda x: x.generated_count) if decisions else None
