"""Pack and score the three Astra arms of LADDER2_PREREG.md.

Usage: python3 -I score_ladder2.py pack     new answers -> ladder2_packets/<code>.md (all blind-coded conditions in one
                                             shuffled set) and ladder2_packets_mapping.json (outside the packet folder)
       python3 -I score_ladder2.py score    join ladder2_packets/codes.json with the keys -> ladder2_scores.json
       python3 -I score_ladder2.py baseline the existing replies only (run before any call, to check the baselines)
Codes are those of score_twins.py: {code: {"stance": "approach"|"avoid"|"unclear", "setting": ..., "asserts_opposite":
bool, "named_game": str}}. Stance correct = the coded stance matches the key. The forced choice is scored from its
Answer line with the pattern of score_bridge.py. A network without a finished reply counts as wrong.
"""
import json
import random
import secrets
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I leaves the script folder off the path
from models import valid_answer  # noqa: E402
from score_bridge import ANS  # noqa: E402

P, MAP = Path("ladder2_packets"), Path("ladder2_packets_mapping.json")
SETS = {"1": "suite_twins", "2": "suite_twins2"}
KEY = {s: json.loads(Path(d, "key.json").read_text()) for s, d in SETS.items()}
# blind-coded condition -> (set, folder of astra_<id> reply folders, answer file)
CODED = {"hint1_s2": ("2", "suite_twins2_hint1/toolfree", "answer.md"),
         "hint2_s2": ("2", "suite_twins2_hint2/toolfree", "answer.md"),
         "max_s1": ("1", "suite_twins/toolfree_max", "answer.md"),
         "max_s2": ("2", "suite_twins2/toolfree_max", "answer.md"),
         "high2_s1": ("1", "suite_twins/toolfree_high2", "answer.md"),
         "high2_s2": ("2", "suite_twins2/toolfree_high2", "answer.md"),
         "pystance": (None, "suite_twins_pystance/analyst", "final_answer.md")}
GAME_OF = {n: k for s in KEY for n, k in KEY[s].items()}


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def new_answers():
    out = {}
    for cond, (_, folder, name) in CODED.items():
        for d in sorted(Path(folder).glob("astra_*")):
            t = valid_answer(d, name)
            if t:
                out[(cond, d.name.split("_", 1)[1])] = t
    return out


def existing():
    """Stance correct per network for the Astra replies collected before this run: {condition: {id: bool}}."""
    base = {}
    for s, d in SETS.items():
        rows = json.loads(Path(d, "twin_scores.json").read_text())
        base[f"open_s{s}"] = {r["id"]: r["stance_correct"] for r in rows if (r["arm"], r["model"]) == ("toolfree", "astra")}
        base.setdefault("pyopen", {}).update({r["id"]: r["stance_correct"] for r in rows
                                              if (r["arm"], r["model"]) == ("analyst", "astra")})
    for h in ("1", "2"):
        rows = json.loads(Path(f"suite_twins_hint{h}/twin_scores.json").read_text())
        base[f"hint{h}_s1"] = {r["id"]: r["stance_correct"] for r in rows if r["model"] == "astra"}
    base["fc_s1"] = {r["id"]: r["correct"] for r in json.loads(Path("suite_bridge_fc/bridge_scores.json").read_text())
                     if r["model"] == "astra"}
    mp = json.loads(Path("twin_packets_ax_mapping.json").read_text())
    codes = json.loads(Path("twin_packets_ax/codes.json").read_text())
    for c, v in mp.items():
        if v["run"].startswith("analyst_astra_") and c in codes:
            nid = v["run"].rsplit("_", 1)[1]
            base["pyopen"][nid] = codes[c]["stance"] == GAME_OF[nid]["stance"]
    return base


def forced_choice():
    key = json.loads(Path("suite_twins2_fc/key.json").read_text())
    out = {}
    for nid, k in key.items():
        t = valid_answer(Path("suite_twins2_fc/toolfree") / f"astra_{nid}")
        m = ANS.search(t) if t else None
        out[nid] = bool(m) and m.group(1) == k["truth_letter"]
    return out


def line(label, got, ids, against=None, against_label=""):
    twins = [n for n in ids if GAME_OF[n]["twin"]]
    known = [n for n in ids if not GAME_OF[n]["twin"]]
    by = {g: f"{sum(got.get(n, False) for n in twins if GAME_OF[n]['game'] == g)}/"
             f"{sum(GAME_OF[n]['game'] == g for n in twins)}" for g in ("dodge", "hunter", "cartag")}
    s = (f"{label:34s} answered {sum(n in got for n in ids):2d}/{len(ids):2d}  known "
         f"{sum(got.get(n, False) for n in known)}/{len(known)}  twins {sum(got.get(n, False) for n in twins)}/{len(twins)}"
         f"  (Dodge {by['dodge']}, Hunter {by['hunter']}, Car-tag {by['cartag']})")
    if against is not None:
        b = sum(got.get(n, False) and not against.get(n, False) for n in twins)
        c = sum(against.get(n, False) and not got.get(n, False) for n in twins)
        s += f"  twins vs {against_label}: {b}-{c}, McNemar p={mcnemar(b, c):.4f}"
    print(s)


def report(cond):
    base = existing()
    ids = {s: list(KEY[s]) for s in KEY}
    both = ids["1"] + ids["2"]
    camo = [n for n in both if GAME_OF[n]["condition"] == "camo"]
    print("Existing Astra replies (baselines)")
    for c, s in (("open_s1", "1"), ("open_s2", "2"), ("hint1_s1", "1"), ("hint2_s1", "1"), ("fc_s1", "1")):
        line(c, base[c], ids[s])
    line("pyopen (Python, open question)", base["pyopen"], camo)
    if cond is None:
        return
    cond["fc_s2"] = forced_choice()
    print("\nA. Question ladder, set 2 (primary: each rung against the open question on the same 9 twins)")
    for c in ("hint1_s2", "hint2_s2", "fc_s2"):
        line(c, cond[c], ids["2"], base["open_s2"], "open")
    print("   pooled over both sets (18 twins)")
    open_both = {**base["open_s1"], **base["open_s2"]}
    pooled = {}
    for r in ("hint1", "hint2", "fc"):
        pooled[r] = {**base[f"{r}_s1"], **cond[f"{r}_s2"]}
        line(f"{r} both sets", pooled[r], both, open_both, "open")
    line("hint2 against hint1, both sets", pooled["hint2"], both, pooled["hint1"], "hint1")
    line("fc against hint2, both sets", pooled["fc"], both, pooled["hint2"], "hint2")
    print("\nB. Open question at effort max, both sets (primary: max against the concurrent high replies)")
    mx, h2 = {**cond["max_s1"], **cond["max_s2"]}, {**cond["high2_s1"], **cond["high2_s2"]}
    line("max", mx, both, h2, "high2")
    line("max (against the earlier high)", mx, both, open_both, "open")
    line("high2 (against the earlier high)", h2, both, open_both, "open")
    print("\nC. Python with the behaviour question, 24 camouflaged networks (primary: against Python, open question)")
    line("pystance", cond["pystance"], camo, base["pyopen"], "pyopen")
    for d in sorted(Path("suite_twins_pystance/analyst").glob("astra_*")):
        calls = d / "tool_calls.jsonl"
        runs = len(calls.read_text().splitlines()) if calls.exists() else 0
        if (d / "final_answer.md").exists() and runs == 0:
            print(f"   {d.name}: final answer with 0 code runs")


if sys.argv[1] == "pack":
    P.mkdir(exist_ok=True)
    mp = json.loads(MAP.read_text()) if MAP.exists() else {}
    have, items = set(map(tuple, mp.values())), []
    for (cond, nid), text in new_answers().items():
        if (cond, nid) not in have:
            c = secrets.token_hex(3)
            mp[c] = [cond, nid]
            items.append((c, text))
    random.Random(29).shuffle(items)
    for c, t in items:
        (P / f"{c}.md").write_text(t)
    MAP.write_text(json.dumps(mp, indent=1))
    print(len(items), "packed;", len(mp), "in mapping")
elif sys.argv[1] == "baseline":
    report(None)
else:
    mp, codes = json.loads(MAP.read_text()), json.loads((P / "codes.json").read_text())
    cond = {c: {} for c in CODED}
    rows = []
    for c, (cn, nid) in mp.items():
        if c in codes and not codes[c].get("invalid"):
            k, v = GAME_OF[nid], codes[c]
            cond[cn][nid] = v["stance"] == k["stance"]
            rows.append({"arm": cn, "id": nid, **k, **{f"coded_{a}": b for a, b in v.items()},
                         "stance_correct": cond[cn][nid]})
    report(cond)
    rows += [{"arm": "fc_s2", "id": n, **GAME_OF[n], "stance_correct": ok} for n, ok in cond["fc_s2"].items()]
    Path("ladder2_scores.json").write_text(json.dumps(rows, indent=1))
