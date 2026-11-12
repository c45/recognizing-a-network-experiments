"""Score the camouflage suite: game identity (keywords, as pre-registered) and stated orientation.

Usage: python3 -I score_suite.py <suite dir> [--sides sides.json]
Without --sides: writes orientation_review.md, every answer's sentences that mention the paddle/player/agent/goal with
a side word (left, right, top, bottom, up, down), listed by run id only, for blind hand-coding into sides.json as
{"<arm>_<id>": "left" | "right" | "top" | "bottom" | "none"}. With --sides: joins the key and prints the comparison.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I leaves the script folder off the path
from models import MODELS, effort, valid_answer  # noqa: E402

GAME_RULES = [("breakout", r"breakout|arkanoid|brick"), ("pacman", r"pac-?\s?man|maze"),
              ("freeway", r"freeway|frogger|road|traffic|crossing|cross"), ("pong", r"\bpong\b|paddle")]
SIDE = re.compile(r"\b(left|right|top|bottom|upper|lower|upward|downward)\b", re.I)
SUBJ = re.compile(r"paddle|player|agent|goal|destination|start", re.I)


def game_of(answer):
    m = re.search(r"(?mi)^[\W_]*Game[\W_]*:[\W_]*(.+)", answer) or re.search(r"(?mi)^[#\s]*Game\s*$\n+(.+)", answer)
    line = (m.group(1) if m else answer[:300]).lower()
    for g, pat in GAME_RULES:
        if re.search(pat, line):
            return g, line[:120]
    return "other", line[:120]


def answers(suite):
    out = {}
    for d in sorted((suite / "toolfree").glob("*_*")):
        t = valid_answer(d) if d.is_dir() else None
        if t:
            out[d.name] = ("toolfree",) + tuple(d.name.split("_", 1)) + (t,)
    for d in sorted((suite / "analyst").glob("*_*")):
        f = d / "final_answer.md"
        if f.exists():
            out["analyst_" + d.name] = ("analyst",) + tuple(d.name.split("_", 1)) + (f.read_text(),)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("--sides")
    a = ap.parse_args()
    suite = Path(a.suite).resolve()
    key = json.loads((suite / "key.json").read_text())
    ans = answers(suite)
    if not a.sides:
        lines = ["# Orientation review (blind: run ids only)", ""]
        for rid, (_, model, nid, text) in sorted(ans.items(), key=lambda kv: kv[0][::-1]):
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text) if SIDE.search(s) and SUBJ.search(s)]
            lines += [f"## {rid}", *[f"- {s[:300]}" for s in sents[:8]], ""]
        (suite / "orientation_review.md").write_text("\n".join(lines))
        print(f"wrote orientation_review.md for {len(ans)} answers")
        return
    sides = json.loads(Path(a.sides).read_text())
    want = {"paddle on the left edge": "left", "paddle on the right edge": "right", "paddle on the bottom row": "bottom",
            "paddle on the top row": "top", "start at the bottom, goal at the top": "top",
            "start at the top, goal at the bottom": "bottom"}
    rows = []
    for rid, (arm, model, nid, text) in ans.items():
        k = key[nid]
        g, line = game_of(text)
        truth_side = want.get(k["orientation"])
        stated = sides.get(rid, "none")
        rows.append({"run": rid, "arm": arm, "model": model, "id": nid, **k,
                     "effort": effort(suite / arm / f"{model}_{nid}"), "said_game": g, "game_line": line,
                     "game_correct": g == k["game"], "stated_side": stated, "true_side": truth_side,
                     "side_correct": None if truth_side is None or stated == "none" else stated == truth_side})
    (suite / "scores.json").write_text(json.dumps(rows, indent=1))
    for arm in ("toolfree", "analyst"):
        for model in MODELS:
            for cond in ("plain", "camo"):
                sel = [r for r in rows if r["arm"] == arm and r["model"] == model and r["condition"] == cond]
                if not sel:
                    continue
                gc = sum(r["game_correct"] for r in sel)
                sd = [r for r in sel if r["true_side"] is not None]
                sc = sum(r["side_correct"] is True for r in sd)
                sw = sum(r["side_correct"] is False for r in sd)
                lad = sum(r["effort"] not in ("max", "high") if model != "astra" else False for r in sel)
                print(f"{arm:8s} {model:5s} {cond:5s} game {gc}/{len(sel)}   side right {sc}, wrong {sw}, "
                      f"not stated {len(sd) - sc - sw} (of {len(sd)})" + (f"   [{lad} below max effort]" if lad else ""))


if __name__ == "__main__":
    main()
