"""Compare decoded turns of two Spec-Bench jsonl files by question_id."""
import argparse
import json
import sys


def _load(path):
    by_id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_id[row["question_id"]] = row
    return by_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("echo_file")
    parser.add_argument("vanilla_file")
    parser.add_argument("--max-print", type=int, default=8)
    args = parser.parse_args()

    echo = _load(args.echo_file)
    vanilla = _load(args.vanilla_file)
    shared = sorted(set(echo) & set(vanilla))
    if not shared:
        print("no overlapping question_id")
        sys.exit(1)

    n_turns = 0
    n_match = 0
    mismatches = []
    for qid in shared:
        e_turns = echo[qid]["choices"][0]["turns"]
        v_turns = vanilla[qid]["choices"][0]["turns"]
        n = min(len(e_turns), len(v_turns))
        if len(e_turns) != len(v_turns):
            mismatches.append((qid, "turn_count", len(e_turns), len(v_turns)))
        for t in range(n):
            n_turns += 1
            if e_turns[t] == v_turns[t]:
                n_match += 1
            else:
                mismatches.append((qid, t, e_turns[t][:120], v_turns[t][:120]))

    print(f"overlap questions: {len(shared)}  echo={len(echo)}  vanilla={len(vanilla)}")
    print(f"matching turns: {n_match}/{n_turns}")
    print(f"mismatched turns: {len(mismatches)}")
    for item in mismatches[: args.max_print]:
        print(item)
    sys.exit(0 if not mismatches else 1)


if __name__ == "__main__":
    main()
