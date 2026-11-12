"""Paired comparison of tool-free Opus vs Astra on camouflaged networks, per architecture and pooled; output tokens.

Usage: python3 -I arch_stats.py   (reads suite_v2, suite_relu24x3, suite_tanh48x1 scores.json and run files)
ReLU Breakout: the camouflaged networks of suite_relu24x3 are replaced by suite_relu24x3_bfix (camouflage rebuilt with
unused inputs from the whole training record; the originals overwrote 40 inputs the final policy uses). The original
numbers are kept by arch_stats_v1.py -> arch_stats_v1.json.
Adjudication: suite_tanh48x1 opus_16d4a089 (plain Breakout) answered Pong; the keyword rule matched "closest
alternative is Breakout" on its Game line, so it is counted wrong here (camouflaged results are unaffected).
"""
import json
import sys
from math import comb
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I leaves the script folder off the path
from models import CLAUDE, MODELS  # noqa: E402

SUITES = {"tanh32x2": "suite_v2", "relu24x3": "suite_relu24x3", "tanh48x1": "suite_tanh48x1"}
OVERRIDE = {("suite_tanh48x1", "opus_16d4a089"): False}


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n) if n else 1.0


def out_tokens(S, model, nid):
    d = Path(S) / "toolfree" / f"{model}_{nid}"
    if model in CLAUDE:
        u = json.loads((d / "run.json").read_text()).get("usage") or {}
        return u.get("output_tokens")
    ev = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if '"turn.completed"' in l]
    return ev[-1]["usage"].get("output_tokens") if ev else None


res, pooled = {}, {"b": 0, "c": 0, "o": 0, "a": 0, "n": 0}
for arch, S in SUITES.items():
    rows = [{**r, "suite": S} for r in json.loads((Path(S) / "scores.json").read_text())]
    if arch == "relu24x3":
        rows = [r for r in rows if not (r["game"] == "breakout" and r["condition"] == "camo")]
        rows += [{**r, "suite": "suite_relu24x3_bfix"}
                 for r in json.loads(Path("suite_relu24x3_bfix/scores.json").read_text())]
    sof = {r["id"]: r["suite"] for r in rows}
    for r in rows:
        if (S, r["run"]) in OVERRIDE:
            r["game_correct"] = OVERRIDE[(S, r["run"])]
    tf = {(r["model"], r["id"]): r for r in rows if r["arm"] == "toolfree"}
    camo = sorted({r["id"] for r in rows if r["condition"] == "camo"})
    o = [tf[("opus", n)]["game_correct"] for n in camo]
    a = [tf[("astra", n)]["game_correct"] for n in camo]
    b = sum(x and not y for x, y in zip(o, a))
    c = sum(y and not x for x, y in zip(o, a))
    cell = {}
    for arm in ("toolfree", "analyst"):
        for model in MODELS:
            for cond in ("plain", "camo"):
                sel = [r for r in rows if r["arm"] == arm and r["model"] == model and r["condition"] == cond]
                if sel:
                    cell[f"{arm}/{model}/{cond}"] = f"{sum(r['game_correct'] for r in sel)}/{len(sel)}"
    by_game = {}
    for g in ("pong", "breakout", "pacman", "freeway"):
        ids = [n for n in camo if tf[("astra", n)]["game"] == g]
        by_game[g] = {m: sum(tf[(m, n)]["game_correct"] for n in ids) for m in ("astra", "opus")}
    toks = {m: median([t for t in (out_tokens(sof[n], m, n) for n in camo if (m, n) in tf) if t] or [0]) for m in MODELS}
    res[arch] = {"cells": cell, "opus_only": b, "astra_only": c, "mcnemar_p": round(mcnemar(b, c), 4),
                 "camo_by_game": by_game, "median_output_tokens_camo": toks,
                 "astra_wrong_said": [tf[("astra", n)]["said_game"] + ":" + tf[("astra", n)]["game"] for n in camo
                                      if not tf[("astra", n)]["game_correct"]]}
    pooled["b"] += b
    pooled["c"] += c
    pooled["o"] += sum(o)
    pooled["a"] += sum(a)
    pooled["n"] += len(camo)
res["pooled_new_archs"] = {k: res["relu24x3"][k] + res["tanh48x1"][k] for k in ("opus_only", "astra_only")}
res["pooled_new_archs"]["mcnemar_p"] = mcnemar(res["pooled_new_archs"]["opus_only"], res["pooled_new_archs"]["astra_only"])
res["pooled_all"] = {**pooled, "mcnemar_p": mcnemar(pooled["b"], pooled["c"])}
Path("arch_stats.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
