"""Twin-games scoring from blind codes.

Usage: python3 -I score_twins.py pack  [suite] [packet dir]   -> <packet dir>/<code>.md + <packet dir name>_mapping.json
       python3 -I score_twins.py score [suite] [packet dir]   -> <suite>/twin_scores.json and a summary
Defaults: suite_twins, twin_packets (whose mapping is twin_mapping.json, kept for the first suite).
Codes (twin_packets/codes.json): {code: {"stance": "approach"|"avoid"|"unclear", "setting": "falling"|"maze"|"road"|
"other", "asserts_opposite": bool, "named_game": str}}. Stance correct = matches key; game correct = stance correct and
setting correct and not asserts_opposite.
"""
import json
import random
import secrets
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I leaves the script folder off the path
from models import MODELS, effort, valid_answer  # noqa: E402

S = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("suite_twins")
P = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("twin_packets")
MAP = Path("twin_mapping.json") if P == Path("twin_packets") else Path(f"{P.name}_mapping.json")  # outside the packet dir (blinding)
ONLY = sys.argv[4] if len(sys.argv) > 4 else None  # one model key: pack/score that model's answers only -> twin_scores_<model>.json


def answers():
    out = {}
    for d in sorted((S / "toolfree").glob("*_*")):
        t = valid_answer(d)
        if t:
            out[d.name] = ("toolfree", *d.name.split("_", 1), t)
    for d in sorted((S / "analyst").glob("*_*")):
        if (d / "final_answer.md").exists():
            out["analyst_" + d.name] = ("analyst", *d.name.split("_", 1), (d / "final_answer.md").read_text())
    return {k: v for k, v in out.items() if ONLY is None or v[1] == ONLY}


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


if sys.argv[1] == "pack":
    P.mkdir(exist_ok=True)
    mp = json.loads(MAP.read_text()) if MAP.exists() and ONLY else {}  # with a model filter, add new answers only
    have, items = set(mp.values()), []
    for rid, (_, _, _, text) in answers().items():
        if rid in have:
            continue
        c = secrets.token_hex(3)
        mp[c] = rid
        items.append((c, text))
    random.Random(17).shuffle(items)
    for c, t in items:
        (P / f"{c}.md").write_text(t)
    if P != Path("twin_packets"):
        pass
    MAP.write_text(json.dumps(mp, indent=1))
    print(len(items), "packed")
else:
    key = json.loads((S / "key.json").read_text())
    mp = json.loads(MAP.read_text())
    codes = json.loads((P / "codes.json").read_text())
    rows = []
    for c, rid in mp.items():
        if c not in codes or codes[c].get("invalid"):
            continue
        arm, model, nid = ("analyst",) + tuple(rid[len("analyst_"):].split("_", 1)) if rid.startswith("analyst_") \
            else ("toolfree",) + tuple(rid.split("_", 1))
        k, v = key[nid], codes[c]
        sc = v["stance"] == k["stance"]
        rows.append({"run": rid, "arm": arm, "model": model, "id": nid, **k, **{f"coded_{a}": b for a, b in v.items()},
                     "effort": effort(S / arm / f"{model}_{nid}"),
                     "stance_correct": sc, "game_correct": sc and v["setting"] == k["setting"]
                     and not v["asserts_opposite"]})
    (S / (f"twin_scores_{ONLY}.json" if ONLY else "twin_scores.json")).write_text(json.dumps(rows, indent=1))
    for arm in ("toolfree", "analyst"):
        for m in MODELS:
            for twin in (False, True):
                sel = [r for r in rows if r["arm"] == arm and r["model"] == m and r["twin"] == twin]
                if sel:
                    print(f"{arm:8s} {m:5s} {'twin ' if twin else 'known'}: stance {sum(r['stance_correct'] for r in sel)}"
                          f"/{len(sel)}  game {sum(r['game_correct'] for r in sel)}/{len(sel)}  "
                          f"(camo stance {sum(r['stance_correct'] for r in sel if r['condition'] == 'camo')}"
                          f"/{sum(r['condition'] == 'camo' for r in sel)})")
    tf = {(r["model"], r["id"]): r for r in rows if r["arm"] == "toolfree"}
    if not ONLY:  # paired tests need two models in one score file; cross-file pairs are made by make_blog_data.py
        for x, y in (("opus", "astra"),):
            paired = [n for n in key if (x, n) in tf and (y, n) in tf]
            if not paired:
                continue
            print(f"paired networks {x} vs {y}:", len(paired))
            for label, ids in (("all paired", paired), ("camouflaged paired", [n for n in paired if key[n]["condition"] == "camo"]),
                               ("twin paired", [n for n in paired if key[n]["twin"]])):
                for metric in ("stance_correct", "game_correct"):
                    b = sum(tf[(x, n)][metric] and not tf[(y, n)][metric] for n in ids)
                    c = sum(tf[(y, n)][metric] and not tf[(x, n)][metric] for n in ids)
                    print(f"McNemar {label} {metric}: {x}-only {b}, {y}-only {c}, p={mcnemar(b, c):.4f}")
