"""Blind packets and scores for the open and hint answers on the rebuilt twin games (V2GAMES_PREREG.md).

Usage: python3 -I pack_v2games.py pack     new answers -> v2games_packets/batch<k>/<code>.md (shuffled; at most 45 per
                                           batch) and v2games_packets_mapping.json (outside the packet folder)
       python3 -I pack_v2games.py score    join every batch's codes.json with the keys -> v2games_scores.json + summary
Arms: astra_max (toolfree_max/astra_*), astra (toolfree/astra_*, high), opus (toolfree/). Coder brief:
CODER_BRIEF_TWINS.md. The coder sees only the answer text.
"""
import json
import random
import secrets
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import effort, valid_answer  # noqa: E402

CONDS = {"open": "", "hint1": "_hint1", "hint2": "_hint2"}
BASES = ("suite_twins_v2games", "suite_twins2_v2games")
P, MAP = Path("v2games_packets"), Path("v2games_packets_mapping.json")


def answers():
    for base in BASES:
        key = json.loads(Path(base, "key.json").read_text())
        for cond, sfx in CONDS.items():
            s = Path(base + sfx)
            for folder in ("toolfree_max", "toolfree"):
                for d in sorted((s / folder).glob("*_*")):
                    model, nid = d.name.split("_", 1)
                    t = valid_answer(d)
                    if not t or nid not in key or (folder == "toolfree_max" and model != "astra"):
                        continue
                    arm = "astra_max" if folder == "toolfree_max" else model
                    yield (base, cond, arm, nid), t, d, key[nid]


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


if sys.argv[1] == "pack":
    mp = json.loads(MAP.read_text()) if MAP.exists() else {}
    have = {tuple(v["run"]) for v in mp.values()}
    new = [(k, t) for k, t, _, _ in answers() if k not in have]
    random.Random(len(mp) + 31).shuffle(new)
    nb = max([v["batch"] for v in mp.values()], default=0)
    for i in range(0, len(new), 45):
        nb += 1
        (P / f"batch{nb}").mkdir(parents=True, exist_ok=True)
        for k, t in new[i:i + 45]:
            c = secrets.token_hex(3)
            mp[c] = {"run": list(k), "batch": nb}
            (P / f"batch{nb}" / f"{c}.md").write_text(t)
        print(f"batch{nb}: {len(new[i:i + 45])} answers")
    MAP.write_text(json.dumps(mp, indent=1))
    print(len(new), "packed;", len(mp), "in mapping")
else:
    mp = json.loads(MAP.read_text())
    codes = {}
    for f in sorted(P.glob("batch*/codes.json")):
        codes.update(json.loads(f.read_text()))
    meta = {k: (d, kk) for k, _, d, kk in answers()}
    rows = []
    for c, v in mp.items():
        k = tuple(v["run"])
        if c not in codes or k not in meta:
            continue
        d, kk = meta[k]
        cd = codes[c]
        sc = cd["stance"] == kk["stance"]
        rows.append({"suite": k[0], "question": k[1], "arm": k[2], "id": k[3], "game": kk["game"], "twin": kk["twin"],
                     "condition": kk["condition"], "stance": kk["stance"],
                     "effort": "max" if k[2] == "astra_max" else effort(d),
                     **{f"coded_{a}": b for a, b in cd.items()}, "stance_correct": sc,
                     "game_correct": sc and cd["setting"] == kk["setting"] and not cd["asserts_opposite"]})
    Path("v2games_scores.json").write_text(json.dumps(rows, indent=1))
    print(len(rows), "scored of", len(mp), "packed")
    for arm in ("astra_max", "astra", "opus"):
        for q in CONDS:
            sel = [r for r in rows if r["arm"] == arm and r["question"] == q]
            if not sel:
                continue
            line = f"{arm:9s} {q:5s}"
            for g in ("cartag_v2", "hunter_v2", "freeway_v2"):
                s = [r for r in sel if r["game"] == g]
                line += f"  {g} {sum(r['stance_correct'] for r in s)}/{len(s)}"
            tw = [r for r in sel if r["twin"]]
            print(line, f" | twins {sum(r['stance_correct'] for r in tw)}/{len(tw)}"
                        f" (game {sum(r['game_correct'] for r in tw)}/{len(tw)})")
    by = defaultdict(dict)
    for r in rows:
        if r["twin"]:
            by[(r["arm"], r["suite"], r["id"])][r["question"]] = r["stance_correct"]
    for arm in ("astra_max", "astra", "opus"):
        for q in ("hint1", "hint2"):
            pairs = [v for (a, _, _), v in by.items() if a == arm and "open" in v and q in v]
            if pairs:
                b = sum(v[q] and not v["open"] for v in pairs)
                c = sum(v["open"] and not v[q] for v in pairs)
                print(f"{arm} twins {q} vs open on {len(pairs)} paired: {b}-{c}, McNemar p={mcnemar(b, c):.4f}")
