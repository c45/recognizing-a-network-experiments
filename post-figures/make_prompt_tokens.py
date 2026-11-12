"""Prompt size in tokens for the depth and transformer tests, as each model counted it.

Usage: python3 -I make_prompt_tokens.py   (writes prompt_tokens.json next to this file)

Reads the first usage record of each saved no-tools call (events.jsonl, kept on this machine and not in git), so
a count comes from the model's own tokenizer and includes its harness's instructions. Medians per depth; the
transformer row pools the 16-network and mirror reader sets.
"""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEPTH = HERE.parent / "studies" / "depth-benchmark"
DEPTHS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16)


def first_input(events, arm):
    for line in events.open(errors="replace"):
        if '"usage"' not in line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = event.get("usage") or (event.get("message") or {}).get("usage")
        if not usage:
            continue
        if arm.startswith("astra"):
            return usage.get("input_tokens")
        return sum(usage.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    return None


def counts(set_dirs):
    out = {}
    for arm in ("astra_xhigh", "opus_xhigh"):
        tokens = [t for d in set_dirs for events in sorted((d / "replies" / arm).glob("*/events.jsonl"))
                  if (t := first_input(events, arm))]
        name = arm.split("_")[0]
        out[name], out[name + "_n"] = round(statistics.median(tokens)), len(tokens)
    return out


def main():
    scale = DEPTH / "ladder" / "scale"
    depth = [{"d": d, **counts([DEPTH / "ladder" / "l2" / "readers_d"] if d == 2
                               else sorted(p for p in scale.glob(f"readers_d{d}*") if p.name[len(f"readers_d{d}"):].isalpha() or p.name == f"readers_d{d}"))}
             for d in DEPTHS]
    out = {"depth": depth, "transformer": counts([DEPTH / "realistic" / "readers_r1b", DEPTH / "realistic" / "readers_r1c"])}
    (HERE / "prompt_tokens.json").write_text(json.dumps(out, indent=1) + "\n")
    for row in depth:
        print(row)
    print("transformer", out["transformer"])


if __name__ == "__main__":
    main()
