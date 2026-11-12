"""Score the minimum bridge (BRIDGE_PREREG.md).

Usage: python3 -I score_bridge.py fc            forced-choice twins: parse "Answer: A|B", compare with the open condition
       python3 -I score_bridge.py pack          open benchmark answers -> bridge_open_packets/<code>.md (blind) and
                                                bridge_open_mapping.json (kept outside the packet folder)
       python3 -I score_bridge.py open          join bridge_open_packets/codes.json with the key
Writes suite_bridge_fc/bridge_scores.json and suite_bridge_open/bridge_scores.json.
"""
import json
import random
import re
import secrets
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I leaves the script folder off the path
from models import MODELS, effort, valid_answer  # noqa: E402

ANS = re.compile(r"(?mi)^[\W_]*Answer[\W_]*:[\W_]*\(?([AB])\b")


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def answers(suite):
    out = {}
    for d in sorted((Path(suite) / "toolfree").glob("*_*")):
        t = valid_answer(d)  # skips archived attempts (*.capped_<effort>_<time>) and replies that ended at the output cap
        if t:
            model, nid = d.name.split("_", 1)
            out[(model, nid)] = t
    return out


def fc():
    key = json.loads(Path("suite_bridge_fc/key.json").read_text())
    open_rows = {(r["model"], r["id"]): r for f in sorted(Path("suite_twins").glob("twin_scores*.json"))
                 for r in json.loads(f.read_text()) if r["arm"] == "toolfree"}
    rows = []
    for (model, nid), text in answers("suite_bridge_fc").items():
        m = ANS.search(text)
        said = m.group(1) if m else None
        k = key[nid]
        o = open_rows.get((model, nid))
        rows.append({"model": model, "id": nid, "game": k["game"], "twin": k["twin"], "condition": k["condition"],
                     "effort": effort(Path("suite_bridge_fc/toolfree") / f"{model}_{nid}"),
                     "said": said, "truth": k["truth_letter"], "correct": said == k["truth_letter"],
                     "open_correct": o["stance_correct"] if o else None})
    Path("suite_bridge_fc/bridge_scores.json").write_text(json.dumps(rows, indent=1))
    for model in MODELS:
        for twin in (True, False):
            sel = [r for r in rows if r["model"] == model and r["twin"] == twin]
            if not sel:
                continue
            paired = [r for r in sel if r["open_correct"] is not None]
            b = sum(r["correct"] and not r["open_correct"] for r in paired)
            c = sum(r["open_correct"] and not r["correct"] for r in paired)
            print(f"{model:5s} {'twin ' if twin else 'known'}: forced choice {sum(r['correct'] for r in sel)}/{len(sel)}"
                  f"  open (same networks) {sum(r['open_correct'] for r in paired)}/{len(paired)}"
                  f"  forced-only {b}, open-only {c}, McNemar p={mcnemar(b, c):.4f}"
                  f"  unparsed {sum(r['said'] is None for r in sel)}"
                  f"  efforts {dict((e, sum(r['effort'] == e for r in sel)) for e in sorted({r['effort'] for r in sel}))}")


def pack():
    out = Path("bridge_open_packets")
    out.mkdir(exist_ok=True)
    mp = json.loads(Path("bridge_open_mapping.json").read_text()) if Path("bridge_open_mapping.json").exists() else {}
    have = set(mp.values())
    items = []
    for (model, nid), text in answers("suite_bridge_open").items():
        rid = f"{model}_{nid}"
        if rid in have:
            continue
        c = secrets.token_hex(3)
        mp[c] = rid
        items.append((c, text))
    random.Random(31).shuffle(items)
    for c, t in items:
        (out / f"{c}.md").write_text(t)
    Path("bridge_open_mapping.json").write_text(json.dumps(mp, indent=1))
    print(len(items), "packed")


def score_open():
    key = json.loads(Path("suite_bridge_open/key.json").read_text())
    mp = json.loads(Path("bridge_open_mapping.json").read_text())
    codes = json.loads(Path("bridge_open_packets/codes.json").read_text())
    rows = []
    for c, rid in mp.items():
        if c not in codes:
            continue
        model, nid = rid.split("_", 1)
        k, v = key[nid], codes[c]
        said = {"toward": "right-toward", "away": "right-away"}.get(v["moving_right"])
        rows.append({"model": model, "id": nid, "truth": k["truth"], "moving_right": v["moving_right"],
                     "effort": effort(Path("suite_bridge_open/toolfree") / rid),
                     "direction_dependent": v["direction_dependent"], "correct": said == k["truth"]})
    Path("suite_bridge_open/bridge_scores.json").write_text(json.dumps(rows, indent=1))
    for model in MODELS:
        sel = [r for r in rows if r["model"] == model]
        if sel:
            print(f"{model:5s} open: correct {sum(r['correct'] for r in sel)}/{len(sel)}  said toward "
                  f"{sum(r['moving_right'] == 'toward' for r in sel)}, away {sum(r['moving_right'] == 'away' for r in sel)}"
                  f", not stated {sum(r['moving_right'] == 'not_stated' for r in sel)}; direction-dependent rule "
                  f"{sum(bool(r['direction_dependent']) for r in sel)}/{len(sel)}")


if __name__ == "__main__":
    {"fc": fc, "pack": pack, "open": score_open}[sys.argv[1]]()
