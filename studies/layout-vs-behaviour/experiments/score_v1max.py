"""Astra at max on the unchanged Dodge, Catch and Pac-Man networks (V2GAMES_PREREG.md, Amendment 2).

Usage: python3 -I score_v1max.py pack    new hint answers -> v1max_packets/batch<k>/<code>.md + v1max_packets_mapping.json
       python3 -I score_v1max.py score   codes + forced-choice letters -> v1max_scores.json and a summary
Rows: suite, question (hint1 | hint2 | forced), id, game, twin, condition, run (0 | 1), and stance_correct
(hints, blind-coded with CODER_BRIEF_TWINS.md) or correct (forced choice, the Answer letter).
"""
import json
import random
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import effort, valid_answer  # noqa: E402
from score_bridge import ANS  # noqa: E402

GAMES = ("dodge", "catch", "pacman")
HINTS = {"suite_twins_hint1": ("hint1", 0), "suite_twins_hint2": ("hint2", 0),
         "suite_twins2_hint1": ("hint1", 1), "suite_twins2_hint2": ("hint2", 1)}
# Opus 5.5 (max, cap ladder) on the same unchanged networks: its hint replies and its set-2 open-question replies.
OPUS = dict(HINTS, suite_twins2=("open", 1))
P, MAP = Path("v1max_packets"), Path("v1max_packets_mapping.json")


def hint_answers():
    """((suite, id[, arm]), text, key, question, run, dir). Astra keys keep their original two-part form."""
    for arm, folder, table in (("astra_max", "toolfree_max", HINTS), ("opus", "toolfree", OPUS)):
        for s, (q, run) in table.items():
            key = json.loads(Path(s, "key.json").read_text())
            for nid, k in key.items():
                d = Path(s, folder, f"{arm.split('_')[0]}_{nid}")
                t = valid_answer(d)
                if t and k["game"] in GAMES:
                    yield ((s, nid) if arm == "astra_max" else (s, nid, arm)), t, k, q, run, d


if sys.argv[1] == "pack":
    mp = json.loads(MAP.read_text()) if MAP.exists() else {}
    have = {tuple(v["run"]) for v in mp.values()}
    new = [(k, t) for k, t, *_ in hint_answers() if k not in have]
    random.Random(len(mp) + 7).shuffle(new)
    nb = max([v["batch"] for v in mp.values()], default=0) + 1
    if new:
        (P / f"batch{nb}").mkdir(parents=True, exist_ok=True)
        for k, t in new:
            c = secrets.token_hex(3)
            mp[c] = {"run": list(k), "batch": nb}
            (P / f"batch{nb}" / f"{c}.md").write_text(t)
        MAP.write_text(json.dumps(mp, indent=1))
    print(len(new), "packed into batch", nb, ";", len(mp), "in mapping")
else:
    mp = json.loads(MAP.read_text()) if MAP.exists() else {}
    codes = {}
    for f in sorted(P.glob("batch*/codes.json")):
        codes.update(json.loads(f.read_text()))
    by = {tuple(v["run"]): c for c, v in mp.items()}
    rows = []
    for k, _, kk, q, run, d in hint_answers():
        c = by.get(k)
        if c not in codes:
            continue
        cd = codes[c]
        sc = cd["stance"] == kk["stance"]
        arm = k[2] if len(k) > 2 else "astra_max"
        rows.append({"suite": k[0], "question": q, "id": k[1], "arm": arm, "game": kk["game"], "twin": kk["twin"],
                     "condition": kk["condition"], "run": run,
                     "effort": "max" if arm == "astra_max" else effort(d),
                     **{f"coded_{a}": b for a, b in cd.items()}, "stance_correct": sc,
                     "game_correct": sc and cd["setting"] == kk["setting"] and not cd["asserts_opposite"]})
    key = json.loads(Path("suite_twins2_fc", "key.json").read_text())
    for nid, k in key.items():
        d = Path("suite_twins2_fc", "toolfree_max", f"astra_{nid}")
        t = valid_answer(d)
        if t and k["game"] in GAMES:
            m = ANS.search(t)
            rows.append({"suite": "suite_twins2_fc", "question": "forced", "id": nid, "arm": "astra_max", "game": k["game"], "twin": k["twin"],
                         "condition": k["condition"], "run": 1, "effort": "max",
                         "said": m.group(1) if m else None, "truth": k["truth_letter"],
                         "correct": bool(m) and m.group(1) == k["truth_letter"]})
    Path("v1max_scores.json").write_text(json.dumps(rows, indent=1))
    for arm in ("astra_max", "opus"):
        for q, f in (("open", "stance_correct"), ("hint1", "stance_correct"), ("hint2", "stance_correct"), ("forced", "correct")):
            s = [r for r in rows if r["question"] == q and r["arm"] == arm]
            if s:
                print(arm, q, {g: f"{sum(r[f] for r in s if r['game'] == g)}/{sum(r['game'] == g for r in s)}" for g in GAMES},
                      "below max:", sum(r["effort"] != "max" for r in s))
