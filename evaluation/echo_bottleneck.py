"""Per-step bottleneck recorder for ECHO skip-layer decode.

Records the skip-layer cost structure:
  - how many L1 64-trees grew before remaining
  - L1 proposed tokens vs remaining keep
  - where remaining first rejected
  - frozen-bonus force-keep
  - rewind / commit_one_token
  - CPU/GPU time split (retrieve, L1 forward, remaining, KV sync)

Sidecar JSONL is one object per sample (one eval turn). Summarize with:
  python evaluation/summarize_echo_bottleneck.py <file>.bottleneck.jsonl
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict


def _ms(t0):
    return (time.perf_counter() - t0) * 1000.0


class EchoBottleneckLog:
    def __init__(self, path, cuda_times=False, print_every=1, full_every=10):
        self.path = path
        self.cuda_times = bool(cuda_times)
        self.print_every = int(print_every)
        self.full_every = int(full_every)
        self.sample_idx = 0
        self._eval_n = 0
        self._eval_records = []
        self._sample = None
        self._outer = None
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _sync(self):
        if self.cuda_times:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def begin_sample(self, profile_meta=None):
        meta = dict(profile_meta or {})
        self._sample = {
            "sample_idx": self.sample_idx,
            "warmup": bool(meta.get("warmup", False)),
            "question_id": meta.get("question_id"),
            "category": meta.get("category"),
            "turn": meta.get("turn"),
            "choice": meta.get("choice"),
            "t0": time.perf_counter(),
            "outers": [],
            "n_l1_trees": 0,
            "n_rewind": 0,
            "n_commit_one": 0,
            "n_empty_l1": 0,
            "n_frozen_forced": 0,
            "n_extra_tree_then_ones": 0,
            "l1_tokens": 0,
            "remaining_tokens": 0,
            "wasted_l1_tokens": 0,
            "ms_retrieve": 0.0,
            "ms_l1_fwd": 0.0,
            "ms_l1_update": 0.0,
            "ms_remaining": 0.0,
            "ms_kv": 0.0,
            "ms_rewind": 0.0,
            "ms_commit": 0.0,
        }
        self.sample_idx += 1

    def begin_outer(self, prefix_len, frozen_bonus, T):
        self._sync()
        self._outer = {
            "prefix_len": int(prefix_len),
            "frozen_bonus": int(frozen_bonus),
            "T": int(T),
            "t0": time.perf_counter(),
            "t_l1_block0": time.perf_counter(),
            "l1": [],
            "ms_retrieve": 0.0,
            "ms_l1_fwd": 0.0,
            "ms_l1_update": 0.0,
        }

    def add_l1_tree(
        self,
        tree_idx,
        n_nodes,
        n_paths,
        l1_accept,
        retrieve_ms,
        l1_fwd_ms,
        l1_update_ms,
        from_copy_logit,
    ):
        rec = {
            "tree_idx": int(tree_idx),
            "from_copy_logit": bool(from_copy_logit),
            "n_nodes": int(n_nodes),
            "n_paths": int(n_paths),
            "l1_accept": int(l1_accept),
            "retrieve_ms": float(retrieve_ms),
            "l1_fwd_ms": float(l1_fwd_ms),
            "l1_update_ms": float(l1_update_ms),
        }
        self._outer["l1"].append(rec)
        self._outer["ms_retrieve"] += rec["retrieve_ms"]
        self._outer["ms_l1_fwd"] += rec["l1_fwd_ms"]
        self._outer["ms_l1_update"] += rec["l1_update_ms"]

    def mark_l1_block_done(self):
        self._sync()
        self._outer["ms_l1_block"] = _ms(self._outer["t_l1_block0"])

    def time_remaining(self, fn):
        self._sync()
        t0 = time.perf_counter()
        out = fn()
        self._sync()
        self._outer["ms_remaining"] = _ms(t0)
        return out

    def time_kv(self, fn):
        self._sync()
        t0 = time.perf_counter()
        out = fn()
        self._sync()
        self._outer["ms_kv"] = self._outer.get("ms_kv", 0.0) + _ms(t0)
        return out

    def time_rewind(self, fn):
        t0 = time.perf_counter()
        out = fn()
        self._outer["ms_rewind"] = _ms(t0)
        return out

    def time_commit(self, fn):
        self._sync()
        t0 = time.perf_counter()
        out = fn()
        self._sync()
        self._outer["ms_commit"] = _ms(t0)
        return out

    def end_outer(
        self,
        buffer_len,
        remaining_accept,
        frozen_forced,
        rewind,
        empty_l1,
        commit_one,
        reject_at,
    ):
        o = self._outer
        l1_accepts = [x["l1_accept"] for x in o["l1"]]
        n_trees = len(o["l1"])
        l1_tokens = int(sum(l1_accepts))
        rem = int(remaining_accept)
        wasted = max(0, l1_tokens - rem)
        extra_then_ones = n_trees > 1 and rem == 1
        rec = {
            "n_l1_trees": n_trees,
            "l1_accepts": l1_accepts,
            "l1_tokens": l1_tokens,
            "buffer_len": int(buffer_len),
            "remaining_accept": rem,
            "reject_at": None if reject_at is None else int(reject_at),
            "frozen_forced": bool(frozen_forced),
            "rewind": bool(rewind),
            "empty_l1": bool(empty_l1),
            "commit_one": bool(commit_one),
            "wasted_l1_tokens": wasted,
            "extra_tree_then_ones": bool(extra_then_ones),
            "prefix_len": o["prefix_len"],
            "frozen_bonus": o["frozen_bonus"],
            "ms_retrieve": o["ms_retrieve"],
            "ms_l1_fwd": o["ms_l1_fwd"],
            "ms_l1_update": o["ms_l1_update"],
            "ms_l1_block": o.get("ms_l1_block", o["ms_retrieve"] + o["ms_l1_fwd"] + o["ms_l1_update"]),
            "ms_remaining": o.get("ms_remaining", 0.0),
            "ms_kv": o.get("ms_kv", 0.0),
            "ms_rewind": o.get("ms_rewind", 0.0),
            "ms_commit": o.get("ms_commit", 0.0),
            "ms_outer": _ms(o["t0"]),
            "l1": o["l1"],
        }
        s = self._sample
        s["outers"].append(rec)
        s["n_l1_trees"] += n_trees
        s["l1_tokens"] += l1_tokens
        s["remaining_tokens"] += rem
        s["wasted_l1_tokens"] += wasted
        s["n_rewind"] += int(rewind)
        s["n_commit_one"] += int(commit_one)
        s["n_empty_l1"] += int(empty_l1)
        s["n_frozen_forced"] += int(frozen_forced)
        s["n_extra_tree_then_ones"] += int(extra_then_ones)
        s["ms_retrieve"] += rec["ms_retrieve"]
        s["ms_l1_fwd"] += rec["ms_l1_fwd"]
        s["ms_l1_update"] += rec["ms_l1_update"]
        s["ms_remaining"] += rec["ms_remaining"]
        s["ms_kv"] += rec["ms_kv"]
        s["ms_rewind"] += rec["ms_rewind"]
        s["ms_commit"] += rec["ms_commit"]
        self._outer = None

    def end_sample(self, new_tokens, n_outer_reported):
        s = self._sample
        s["wall_ms"] = _ms(s["t0"])
        s["new_tokens"] = int(new_tokens)
        s["n_outer"] = len(s["outers"])
        s["n_outer_reported"] = int(n_outer_reported)
        s.pop("t0", None)
        # Keep per-outer l1 detail; file size is acceptable for Spec-Bench.
        self._fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        self._fh.flush()
        if not s.get("warmup"):
            self._eval_n += 1
            rec = {k: v for k, v in s.items() if k != "outers"}
            rec["outers"] = [{k: v for k, v in o.items() if k != "l1"} for o in s["outers"]]
            self._eval_records.append(rec)
            summary = summarize(self._eval_records)
            if self.print_every > 0 and self._eval_n % self.print_every == 0:
                print(
                    f"[bottleneck q={s.get('question_id')} t={s.get('turn')} {s.get('category')}] "
                    f"{format_oneline(summary)}",
                    flush=True,
                )
            if self.full_every > 0 and self._eval_n % self.full_every == 0:
                print(
                    f"\n===== bottleneck so far ({summary.get('n_samples', 0)} eval samples) =====",
                    flush=True,
                )
                print(format_summary(summary), flush=True)
        self._sample = None


class NullBottleneckLog:
    cuda_times = False

    def begin_sample(self, profile_meta=None):
        return None

    def begin_outer(self, *a, **k):
        return None

    def add_l1_tree(self, *a, **k):
        return None

    def mark_l1_block_done(self):
        return None

    def time_remaining(self, fn):
        return fn()

    def time_kv(self, fn):
        return fn()

    def time_rewind(self, fn):
        return fn()

    def time_commit(self, fn):
        return fn()

    def end_outer(self, *a, **k):
        return None

    def end_sample(self, *a, **k):
        return None

    def close(self):
        return None


def load_records(path, drop_warmup=True):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if drop_warmup and rec.get("warmup"):
                continue
            rows.append(rec)
    return rows


def _pct(n, d):
    return 0.0 if d == 0 else 100.0 * n / d


def summarize(records):
    outers = []
    for rec in records:
        for o in rec.get("outers", []):
            o = dict(o)
            o["_category"] = rec.get("category")
            o["_turn"] = rec.get("turn")
            outers.append(o)

    n_o = len(outers)
    n_s = len(records)
    if n_o == 0:
        return {"n_samples": n_s, "n_outers": 0}

    tree_hist = Counter(o["n_l1_trees"] for o in outers)
    rem_hist = Counter(min(int(o["remaining_accept"]), 12) for o in outers)
    reject_hist = Counter(
        o["reject_at"] if o["reject_at"] is not None else "full" for o in outers
    )
    l1_first = [o["l1_accepts"][0] for o in outers if o["l1_accepts"]]
    l1_later = [a for o in outers for a in o["l1_accepts"][1:]]

    def mean(xs):
        return 0.0 if not xs else sum(xs) / len(xs)

    ms_keys = [
        "ms_retrieve",
        "ms_l1_fwd",
        "ms_l1_update",
        "ms_l1_block",
        "ms_remaining",
        "ms_kv",
        "ms_rewind",
        "ms_commit",
        "ms_outer",
    ]
    ms_sum = {k: sum(o.get(k, 0.0) for o in outers) for k in ms_keys}
    wall = sum(r.get("wall_ms", 0.0) for r in records)
    toks = sum(r.get("new_tokens", 0) for r in records)

    extra_ones = sum(1 for o in outers if o.get("extra_tree_then_ones"))
    n_ones = sum(1 for o in outers if o["remaining_accept"] == 1)
    n_rewind = sum(1 for o in outers if o.get("rewind"))
    n_frozen = sum(1 for o in outers if o.get("frozen_forced"))
    n_commit = sum(1 for o in outers if o.get("commit_one"))
    n_empty = sum(1 for o in outers if o.get("empty_l1"))
    wasted = sum(o.get("wasted_l1_tokens", 0) for o in outers)
    l1_tok = sum(o.get("l1_tokens", 0) for o in outers)
    rem_tok = sum(o.get("remaining_accept", 0) for o in outers)

    by_cat = defaultdict(lambda: dict(n=0, trees=0, rem=0, extra_ones=0, ones=0, wasted=0, l1=0, ms=0.0))
    for o in outers:
        d = by_cat[o.get("_category") or "?"]
        d["n"] += 1
        d["trees"] += o["n_l1_trees"]
        d["rem"] += o["remaining_accept"]
        d["l1"] += o["l1_tokens"]
        d["wasted"] += o.get("wasted_l1_tokens", 0)
        d["extra_ones"] += int(o.get("extra_tree_then_ones"))
        d["ones"] += int(o["remaining_accept"] == 1)
        d["ms"] += o.get("ms_outer", 0.0)

    by_trees_then_rem = Counter(
        (o["n_l1_trees"], min(int(o["remaining_accept"]), 8)) for o in outers
    )

    return {
        "n_samples": n_s,
        "n_outers": n_o,
        "new_tokens": toks,
        "tok_s": 0.0 if wall <= 0 else 1000.0 * toks / wall,
        "mean_l1_trees_per_outer": mean([o["n_l1_trees"] for o in outers]),
        "mean_l1_tokens_per_outer": mean([o["l1_tokens"] for o in outers]),
        "mean_remaining_accept": mean([o["remaining_accept"] for o in outers]),
        "mean_buffer_len": mean([o["buffer_len"] for o in outers]),
        "mean_wasted_l1_tokens": mean([o.get("wasted_l1_tokens", 0) for o in outers]),
        "l1_keep_frac": 0.0 if l1_tok == 0 else rem_tok / l1_tok,
        "pct_ones": _pct(n_ones, n_o),
        "pct_extra_tree_then_ones": _pct(extra_ones, n_o),
        "pct_ones_that_had_extra_tree": _pct(extra_ones, n_ones),
        "pct_rewind": _pct(n_rewind, n_o),
        "pct_frozen_forced": _pct(n_frozen, n_o),
        "pct_commit_one": _pct(n_commit, n_o),
        "pct_empty_l1": _pct(n_empty, n_o),
        "mean_first_l1_accept": mean(l1_first),
        "mean_later_l1_accept": mean(l1_later),
        "n_later_l1_trees": len(l1_later),
        "tree_hist": dict(sorted(tree_hist.items())),
        "remaining_hist": dict(sorted((str(k), v) for k, v in rem_hist.items())),
        "reject_at_hist": {str(k): v for k, v in sorted(reject_hist.items(), key=lambda x: str(x[0]))},
        "trees_x_remaining_top": [
            {"n_l1_trees": a, "remaining_accept_clip8": b, "n": n, "pct": _pct(n, n_o)}
            for (a, b), n in by_trees_then_rem.most_common(20)
        ],
        "ms_sum": ms_sum,
        "ms_frac_of_outer": {
            k: _pct(ms_sum[k], ms_sum["ms_outer"]) for k in ms_keys if k != "ms_outer"
        },
        "ms_per_outer": {k: (ms_sum[k] / n_o) for k in ms_keys},
        "by_category": {
            cat: {
                "n_outers": d["n"],
                "mean_l1_trees": d["trees"] / d["n"],
                "mean_remaining": d["rem"] / d["n"],
                "mean_l1_tokens": d["l1"] / d["n"],
                "keep_frac": 0.0 if d["l1"] == 0 else d["rem"] / d["l1"],
                "pct_ones": _pct(d["ones"], d["n"]),
                "pct_extra_tree_then_ones": _pct(d["extra_ones"], d["n"]),
                "ms_per_outer": d["ms"] / d["n"],
            }
            for cat, d in sorted(by_cat.items(), key=lambda kv: kv[0] or "")
        },
        "wasted_l1_tokens_total": wasted,
        "l1_tokens_total": l1_tok,
        "remaining_tokens_total": rem_tok,
    }


def format_oneline(s):
    if s.get("n_outers", 0) == 0:
        return "no outer steps"
    frac = s.get("ms_frac_of_outer") or {}
    return (
        f"n={s['n_samples']} out={s['n_outers']} "
        f"trees/out={s['mean_l1_trees_per_outer']:.2f} "
        f"L1tok={s['mean_l1_tokens_per_outer']:.2f} rem={s['mean_remaining_accept']:.2f} "
        f"keep={100 * s['l1_keep_frac']:.0f}% ones={s['pct_ones']:.0f}% "
        f"xT+1={s['pct_extra_tree_then_ones']:.0f}% rewind={s['pct_rewind']:.0f}% "
        f"L1blk={frac.get('ms_l1_block', 0):.0f}% remain={frac.get('ms_remaining', 0):.0f}% "
        f"kv={frac.get('ms_kv', 0):.0f}%"
    )


def format_summary(s):
    if s.get("n_outers", 0) == 0:
        return "no outer steps recorded"
    lines = []
    a = lines.append
    a(f"samples {s['n_samples']}  outers {s['n_outers']}  tok/s(profiled) {s['tok_s']:.2f}")
    a(
        f"L1 trees/outer {s['mean_l1_trees_per_outer']:.3f}  "
        f"L1 tokens/outer {s['mean_l1_tokens_per_outer']:.3f}  "
        f"remaining accept {s['mean_remaining_accept']:.3f}  "
        f"buffer {s['mean_buffer_len']:.3f}"
    )
    a(
        f"remaining keeps {100 * s['l1_keep_frac']:.1f}% of L1-proposed tokens  "
        f"wasted L1 tokens {s['wasted_l1_tokens_total']} "
        f"({s['mean_wasted_l1_tokens']:.3f}/outer)"
    )
    a(
        f"ones {s['pct_ones']:.1f}%  extra-tree-then-ones {s['pct_extra_tree_then_ones']:.1f}%  "
        f"(of ones, {s['pct_ones_that_had_extra_tree']:.1f}% grew >1 L1 tree)"
    )
    a(
        f"rewind {s['pct_rewind']:.1f}%  frozen-forced {s['pct_frozen_forced']:.1f}%  "
        f"commit_one {s['pct_commit_one']:.1f}%  empty_L1 {s['pct_empty_l1']:.1f}%"
    )
    a(
        f"first L1 accept {s['mean_first_l1_accept']:.3f}  "
        f"later L1 accept {s['mean_later_l1_accept']:.3f}  "
        f"(n_later_trees {s['n_later_l1_trees']})"
    )
    a("L1 tree-count hist: " + ", ".join(f"{k}:{v}" for k, v in s["tree_hist"].items()))
    a("remaining-accept hist: " + ", ".join(f"{k}:{v}" for k, v in s["remaining_hist"].items()))
    a("reject_at hist: " + ", ".join(f"{k}:{v}" for k, v in s["reject_at_hist"].items()))
    a("time per outer (ms) / fraction of outer:")
    for k in [
        "ms_retrieve",
        "ms_l1_fwd",
        "ms_l1_update",
        "ms_l1_block",
        "ms_remaining",
        "ms_kv",
        "ms_rewind",
        "ms_commit",
        "ms_outer",
    ]:
        frac = s["ms_frac_of_outer"].get(k)
        frac_s = "" if frac is None else f"  ({frac:.1f}% of outer)"
        a(f"  {k:16s} {s['ms_per_outer'][k]:7.2f}{frac_s}")
    a("n_l1_trees x remaining_accept (top):")
    for row in s["trees_x_remaining_top"][:12]:
        a(
            f"  trees={row['n_l1_trees']} rem={row['remaining_accept_clip8']}  "
            f"n={row['n']} ({row['pct']:.1f}%)"
        )
    a("per category:")
    a(
        f"  {'cat':16s} {'trees/out':>9} {'L1tok':>7} {'rem':>7} {'keep':>6} "
        f"{'ones':>6} {'xT+1':>6} {'ms':>8}"
    )
    for cat, d in s["by_category"].items():
        a(
            f"  {str(cat):16s} {d['mean_l1_trees']:9.3f} {d['mean_l1_tokens']:7.3f} "
            f"{d['mean_remaining']:7.3f} {100 * d['keep_frac']:5.1f}% "
            f"{d['pct_ones']:5.1f}% {d['pct_extra_tree_then_ones']:5.1f}% "
            f"{d['ms_per_outer']:8.2f}"
        )
    return "\n".join(lines)
