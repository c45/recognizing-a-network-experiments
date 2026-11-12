"""Forced-choice scores on the rebuilt twin games (V2GAMES_PREREG.md): the Answer letter against truth_letter.

Usage: python3 -I score_v2games_fc.py   -> v2games_fc_scores.json and a summary. Unparseable answers are wrong.
Rows: suite, arm (astra_max, astra, opus), id, game, twin, condition, effort, said, truth, correct.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import effort, valid_answer  # noqa: E402
from score_bridge import ANS  # noqa: E402

rows = []
for suite in ("suite_twins_v2games_fc", "suite_twins2_v2games_fc"):
    key = json.loads(Path(suite, "key.json").read_text())
    for folder, only in (("toolfree_max", "astra"), ("toolfree", None)):
        for d in sorted(Path(suite, folder).glob("*_*")):
            model, nid = d.name.split("_", 1)
            t = valid_answer(d)
            if not t or nid not in key or (only and model != only):
                continue
            m = ANS.search(t)
            k = key[nid]
            arm = "astra_max" if folder == "toolfree_max" else model
            rows.append({"suite": suite, "arm": arm, "id": nid, "game": k["game"], "twin": k["twin"],
                         "condition": k["condition"], "effort": "max" if arm == "astra_max" else effort(d),
                         "said": m.group(1) if m else None, "truth": k["truth_letter"],
                         "correct": bool(m) and m.group(1) == k["truth_letter"]})
Path("v2games_fc_scores.json").write_text(json.dumps(rows, indent=1))
for arm in ("astra_max", "astra", "opus"):
    sel = [r for r in rows if r["arm"] == arm]
    if not sel:
        continue
    line = f"{arm:9s}"
    for g in ("cartag_v2", "hunter_v2", "freeway_v2"):
        s = [r for r in sel if r["game"] == g]
        line += f"  {g} {sum(r['correct'] for r in s)}/{len(s)}"
    tw = [r for r in sel if r["twin"]]
    print(line, f" | twins {sum(r['correct'] for r in tw)}/{len(tw)}", "unparsed", sum(r["said"] is None for r in sel))
