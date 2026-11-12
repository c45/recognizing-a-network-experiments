"""Pack and score the Python-at-max arm (PYMAX_PREREG.md). Self-contained: imports nothing from the other scorers.

Usage: python3 -I score_pymax.py status          finished / failed / missing conversations per job
       python3 -I score_pymax.py pack [twins]    final answers -> pymax_packets/<code>.md (answer text only, one shuffle
                                                 over both questions and all suites) + pymax_packets_mapping.json, which
                                                 stays outside the packet folder. Refuses unless every job (or, with
                                                 "twins", every twin job) is finished or has failed 3 times. Answers
                                                 already packed are not packed again.
       python3 -I score_pymax.py score           pymax_packets/codes.json + key.json -> pymax_scores.json and tables
Codes (pymax_packets/codes.json, written by the blind coder): {code: {"stance": "approach"|"avoid"|"unclear",
"setting": str, "asserts_opposite": bool, "named_game": str}}. Stance correct = the coded stance equals key.json
"stance". A conversation is finished when state.json says done and final_answer.md is not empty (the queue's rule).
Nothing is written inside the suites.
"""
import json
import random
import secrets
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = [l.split() for l in (HERE / "pymax_jobs.txt").read_text().splitlines() if l.strip()]
ARM = {"open": "analyst_max", "stance": "analyst_stance_max"}
P, MAP, OUT = HERE / "pymax_packets", HERE / "pymax_packets_mapping.json", HERE / "pymax_scores.json"
KEYS = {s: json.loads((HERE / s / "key.json").read_text()) for s in sorted({j[0] for j in JOBS})}


def folder(suite, nid, q):
    return HERE / suite / ARM[q] / f"astra_{nid}"


def finished(d):
    f, s = d / "final_answer.md", d / "state.json"
    return f.exists() and f.read_text().strip() != "" and s.exists() and json.loads(s.read_text()).get("done") is True


def attempts(suite, nid, q):
    return len(list((HERE / suite / f"{ARM[q]}_failed").glob(f"astra_{nid}.attempt*")))


def state(job):
    d = folder(*job)
    return "finished" if finished(d) else "failed" if attempts(*job) >= 3 else "missing"


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        for job in JOBS:
            k = KEYS[job[0]][job[1]]
            print(f"{job[0]:22} {job[1]} {job[2]:6} {k['game']:11} {'twin ' if k['twin'] else 'known'} {state(job)}"
                  f" (failed attempts {attempts(*job)})")
        c = [state(j) for j in JOBS]
        print({s: c.count(s) for s in ("finished", "failed", "missing")})
    elif cmd == "pack":
        only_twins = len(sys.argv) > 2 and sys.argv[2] == "twins"
        jobs = [j for j in JOBS if KEYS[j[0]][j[1]]["twin"] or not only_twins]
        missing = [j for j in jobs if state(j) == "missing"]
        if missing:
            sys.exit(f"{len(missing)} conversations neither finished nor failed 3 times: not packed")
        mp = json.loads(MAP.read_text()) if MAP.exists() else {}
        have = {(v["suite"], v["id"], v["question"]) for v in mp.values()}
        items = []
        for job in jobs:
            if tuple(job) in have or state(job) != "finished":
                continue
            code = secrets.token_hex(3)
            while code in mp:
                code = secrets.token_hex(3)
            mp[code] = {"suite": job[0], "id": job[1], "question": job[2]}
            items.append((code, (folder(*job) / "final_answer.md").read_text()))
        random.Random(20261110).shuffle(items)
        P.mkdir(exist_ok=True)
        for code, text in items:
            (P / f"{code}.md").write_text(text)
        MAP.write_text(json.dumps(mp, indent=1))
        print(len(items), "packed;", len(mp), "in mapping")
    elif cmd == "score":
        mp = json.loads(MAP.read_text())
        codes = json.loads((P / "codes.json").read_text())
        by = {(v["suite"], v["id"], v["question"]): c for c, v in mp.items()}
        rows = []
        for job in JOBS:
            suite, nid, q = job
            k, d = KEYS[suite][nid], folder(*job)
            row = {"suite": suite, "id": nid, "game": k["game"], "twin": k["twin"], "condition": k["condition"],
                   "train_seed": k.get("train_seed", 1), "stance": k["stance"], "question": q, "effort": None,
                   "state": state(job), "runs": None, "turns": None, "tool_items": [], "coded_stance": None,
                   "coded_named_game": None, "coded_setting": None, "coded_asserts_opposite": None, "stance_correct": False}
            if row["state"] == "finished":
                st = json.loads((d / "state.json").read_text())
                row.update(runs=st.get("runs"), turns=st.get("turns"),
                           effort=(d / "effort.txt").read_text().strip() if (d / "effort.txt").exists() else None)
                for t in sorted((d / "turns").glob("t*/run.json")):
                    row["tool_items"] += json.loads(t.read_text()).get("tool_items") or []
                c = codes.get(by.get(tuple(job)))
                if c:
                    row.update(coded_stance=c["stance"], coded_named_game=c.get("named_game"), coded_setting=c.get("setting"),
                               coded_asserts_opposite=c.get("asserts_opposite"), stance_correct=c["stance"] == k["stance"])
                else:
                    row["state"] = "finished, not coded"
            rows.append(row)
        OUT.write_text(json.dumps(rows, indent=1))
        ok = lambda rs: f"{sum(r['stance_correct'] for r in rs)}/{len(rs)}"  # noqa: E731
        for q in ("open", "stance"):
            tw = [r for r in rows if r["twin"] and r["question"] == q]
            kn = [r for r in rows if not r["twin"] and r["question"] == q]
            print(f"{q:6} twins {ok(tw)}  " + "  ".join(f"{g} {ok([r for r in tw if r['game'] == g])}" for g in sorted({r['game'] for r in tw}))
                  + f"  | known {ok(kn)}  " + "  ".join(f"{g} {ok([r for r in kn if r['game'] == g])}" for g in sorted({r['game'] for r in kn}))
                  + f"  | known without freeway_v2 {ok([r for r in kn if r['game'] != 'freeway_v2'])}")
        pair = {}
        for r in rows:
            if r["twin"]:
                pair.setdefault((r["suite"], r["id"]), {})[r["question"]] = r["stance_correct"]
        b = sum(v["stance"] and not v["open"] for v in pair.values())
        c = sum(v["open"] and not v["stance"] for v in pair.values())
        print(f"primary, 12 twins: behaviour-only {b}, open-only {c}, exact McNemar p = {mcnemar(b, c):.4f}"
              f" ({'behaviour question ahead' if b > c else 'not ahead'})")
        for r in rows:
            if r["state"] != "finished":
                print("NOT FINISHED OR NOT CODED:", r["suite"], r["id"], r["question"], r["state"])
            elif r["runs"] == 0:
                print("0 CODE RUNS:", r["suite"], r["id"], r["question"], "stance_correct", r["stance_correct"])
            if r["tool_items"]:
                print("TOOL ITEMS:", r["suite"], r["id"], r["question"], r["tool_items"])
            if r["state"] == "finished" and r["effort"] != "max":
                print("EFFORT NOT RECORDED AS MAX:", r["suite"], r["id"], r["question"], r["effort"])
        z = [r for r in rows if r["twin"] and r["state"] == "finished" and r["runs"] == 0]
        if z:
            keep = [r for r in rows if r["twin"] and not (r["state"] == "finished" and r["runs"] == 0)]
            print("twins without 0-run conversations:", {q: ok([r for r in keep if r["question"] == q]) for q in ("open", "stance")})
        print("wrote", OUT.name)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
