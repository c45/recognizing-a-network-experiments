"""The post's `ladder` and `twins` data blocks for the current twin set (V2GAMES_PREREG.md, Amendments 1 and 2).

Usage: python3 -I make_ladder_v2.py [data.json to patch in place]   (prints the blocks; patches the file if given)
Current set: twins = Dodge (v1, unchanged) + Hunter v2 + Car-tag v2; famous games = Catch, Pac-Man (v1) + Freeway v2;
two training runs each. Cells are [right, scored]; a network with no finished reply is left out. `astra_max` is
Astra at max, `opus` is Opus 5.5 at max; Opus replies that hit the output cap at max and answered on the xhigh rerun
count, and are tallied in `opus.xhigh` / `opus.xhigh_by`.
Sources (v1max_scores.json also holds Opus set-2 forced choice and late set-1 open replies, Nov 11): ladder2_scores.json (Astra max, open, unchanged games), suite_bridge_fc/fc_parity_scores.json (Astra max,
forced choice, run 1), v1max_scores.json (Astra max on the unchanged games: hints and run-2 forced choice),
v2games_scores.json and v2games_fc_scores.json (rebuilt games), suite_twins/twin_scores.json and
suite_bridge_fc/bridge_scores.json (Opus on the unchanged games, run 1), pymax_scores.json (Astra max with Python),
pymax_opus_scores.json (Opus max with Python).
"""
import json
import sys
from pathlib import Path

O = Path(__file__).resolve().parent
KEEP = {"dodge": "dodge", "catch": None, "pacman": None, "hunter_v2": "hunter", "cartag_v2": "cartag", "freeway_v2": None}
RUN = {"suite_twins": 0, "suite_twins2": 1, "suite_twins_v2games": 0, "suite_twins2_v2games": 1}


def load(name):
    return json.loads((O / name).read_text())


def cell(rows, runs=False):
    """rows: dicts with game, twin, ok, run. Only games of the current set count."""
    out = {"twin": [0, 0], "known": [0, 0], "games": {g: [0, 0] for g in ("dodge", "hunter", "cartag")}}
    per = [{"twin": [0, 0], "known": [0, 0]}, {"twin": [0, 0], "known": [0, 0]}]
    for r in rows:
        if r["game"] not in KEEP:
            continue
        kind = "twin" if r["twin"] else "known"
        for t in (out[kind], out["games"].get(KEEP[r["game"]]), per[r["run"]][kind]):
            if t is not None:
                t[0] += bool(r["ok"])
                t[1] += 1
    if runs:
        out["runs"] = per
    return out


def rows(src, ok, run=None, **where):
    return [dict(r, ok=r[ok], run=run if run is not None else RUN[r["suite"].split("_fc")[0].split("_hint")[0]])
            for r in src if all(r.get(k) == v for k, v in where.items())]


def main():
    l2, par = load("ladder2_scores.json"), load("suite_bridge_fc/fc_parity_scores.json")
    v2, v2fc, v1 = load("v2games_scores.json"), load("v2games_fc_scores.json"), load("v1max_scores.json")
    py = load("pymax_scores.json")
    opus_open = [r for r in load("suite_twins/twin_scores.json") if r["arm"] == "toolfree" and r["model"] == "opus"]
    opus_fc = [r for r in load("suite_bridge_fc/bridge_scores.json") if r["model"] == "opus"]
    # Opus with Python on the current set's 24 camouflaged networks (OPUS_PYMAX_PREREG.md), when it exists.
    opus_pymax = load("pymax_opus_scores.json") if (O / "pymax_opus_scores.json").exists() else []
    amax = {
        "open": rows(l2, "stance_correct", 0, arm="max_s1") + rows(l2, "stance_correct", 1, arm="max_s2")
        + rows(v2, "stance_correct", arm="astra_max", question="open"),
        "hint1": rows(v1, "stance_correct", question="hint1", arm="astra_max") + rows(v2, "stance_correct", arm="astra_max", question="hint1"),
        "hint2": rows(v1, "stance_correct", question="hint2", arm="astra_max") + rows(v2, "stance_correct", arm="astra_max", question="hint2"),
        "forced": rows(par, "correct", 0, arm="astra_max") + rows(v1, "correct", question="forced", arm="astra_max")
        + rows(v2fc, "correct", arm="astra_max"),
        "python": rows(py, "stance_correct", question="open"),
        "python_stance": rows(py, "stance_correct", question="stance"),
    }
    opus = {
        "open": rows(opus_open, "stance_correct", 0) + rows(v1, "stance_correct", question="open", arm="opus")
        + rows(v2, "stance_correct", arm="opus", question="open"),
        "hint1": rows(v1, "stance_correct", question="hint1", arm="opus") + rows(v2, "stance_correct", arm="opus", question="hint1"),
        "hint2": rows(v1, "stance_correct", question="hint2", arm="opus") + rows(v2, "stance_correct", arm="opus", question="hint2"),
        "forced": rows(opus_fc, "correct", 0) + rows(v1, "correct", question="forced", arm="opus")
        + rows(v2fc, "correct", arm="opus"),
        "python": rows(opus_pymax, "stance_correct", question="open"),
        "python_stance": rows(opus_pymax, "stance_correct", question="stance"),
    }
    ladder = []
    for step in ("open", "hint1", "hint2", "forced", "python", "python_stance"):
        s = {"id": step, "astra_max": cell(amax[step], runs=step in ("open", "hint1", "hint2", "forced"))}
        # Opus runs at max. Replies that hit the output cap at max and were answered one step lower (xhigh) count,
        # and are tallied in `xhigh` / `xhigh_by` so the figures can mark them.
        o = [r for r in opus[step] if r["game"] in KEEP]
        if o:
            s["opus"] = cell(o)
            low = [r for r in o if r.get("effort") not in (None, "max")]
            assert all(r.get("effort") == "xhigh" for r in low), {r.get("effort") for r in low}
            s["opus"]["xhigh"] = len(low)
            # The same, split as the figure shows it: by kind and, for twins, by game.
            s["opus"]["xhigh_by"] = {"twin": sum(r["twin"] for r in low), "known": sum(not r["twin"] for r in low),
                                     "games": {g: sum(KEEP[r["game"]] == g for r in low)
                                               for g in ("dodge", "hunter", "cartag")}}
            if step not in ("open", "forced"):  # the scoreboard reads every game in those two rows
                s["opus"]["games"] = {g: c for g, c in s["opus"]["games"].items() if c[1]}
        ladder.append(s)
    probe = [r for s in ("suite_twins", "suite_twins2", "suite_twins_v2games", "suite_twins2_v2games")
             for r in load(f"{s}/scripted_baseline.json").values() if r["game"] in ("dodge", "hunter_v2", "cartag_v2")]
    twins = {q: {m: {k: s[m][k] for k in ("known", "twin")} for m in ("astra_max", "opus") if m in s}
             for q, s in (("open", ladder[0]), ("forced", ladder[3]), ("python", ladder[4]))}
    twins["probe_twins"] = [sum(r["pred_locked"] == r["truth"] for r in probe), len(probe)]
    for s in ladder:
        print(json.dumps(s))
    print(json.dumps(twins))
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        d = json.loads(p.read_text())
        d["ladder"], d["twins"] = ladder, twins
        p.write_text(json.dumps(d, separators=(",", ":")) if "\n" not in p.read_text().strip() else json.dumps(d, indent="\t") + "\n")
        print("patched", p)


if __name__ == "__main__":
    main()
