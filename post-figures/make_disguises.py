"""Pictures, shapes and scores for the post's combined "disguises" figure (camouflage and the heavier rewrites).

Usage: uv run --with numpy python -I make_disguises.py [recognition dir] [website data.json]
The rewritten networks are the recognition study's stage 5 and stage 6 suites
(default ../studies/footprints-and-function/experiments: build_stage5.py, build_stage6.py, suite_stage5,
suite_stage6). One row per rewrite; each row is four networks, one per game.

Writes disguises.json next to this file:
  rows[i].maps    for the Pong network of that row: first-layer weight size per board cell for the player,
                  moving-object, reward-target and obstacle channels (L2 norm of the input's first-layer weights, divided by the channel's largest),
                  with inputs taken in printed order, so the mixed-input rewrite shows what the prompt shows
  rows[i].layers  the hidden layers as drawn: units, whether the layer was inserted, twins, cancelling pairs
  rows[i].astra / .opus   {"named": [k, n], "games": {game: 1 named, 0 another game, -1 no answer}}
Astra's scores are its max-effort runs when the site data has them for all nine rows (see main), else its high-effort
runs; "astra_effort" in the output says which. Opus is always at max.
Scores use the strict rule of the published rewrites figure (the first game named on the "Game:" line must be right).
The seven rewrite rows are copied from the website's data.json ("rewrites"); the untouched and copy-camouflage rows are
read from suite_stage5/scores.json, where every answer names a single game first, so the strict and keyword rules agree
(checked below against the stored keyword score).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "studies" / "footprints-and-function" / "experiments"
SITE = Path(sys.argv[2]) if len(sys.argv) > 2 else None
GAMES = ["pong", "breakout", "pacman", "freeway"]
# First game named on the Game line of each stage-5 answer for the two baseline rows (read from scores.json game_line;
# "frogger" counts as Freeway, as in the naming figure).
FIRST_NAMED = {
    ("astra", "plain"): {"pong": "pong", "breakout": "pong", "pacman": "pacman", "freeway": "freeway"},
    ("astra", "camo_v2"): {"pong": "pong", "breakout": "pong", "pacman": "freeway", "freeway": "freeway"},
    ("opus", "plain"): {g: g for g in GAMES},
    ("opus", "camo_v2"): {g: g for g in GAMES},
}


def maps(w1):
    norm = np.linalg.norm(w1.astype(np.float64), axis=1).reshape(10, 10, 5)
    return [(norm[:, :, ch] / norm[:, :, ch].max()).round(2).tolist() for ch in (0, 1, 2, 3)]


def pong_net(suite, key, condition):
    ids = [i for i, v in key.items() if v["game"] == "pong" and v["condition"] == condition]
    assert len(ids) == 1, (condition, ids)
    return np.load(REC / suite / "nets" / ids[0] / "actor.npz")


def widths(z):
    n = len([k for k in z.files if k.startswith("W")])
    return [int(z[f"W{k}"].shape[1]) for k in range(1, n)]


def baseline(scores, condition):
    out = {}
    for model in ("astra", "opus"):
        rows = {r["game"]: r for r in scores if r["model"] == model and r["condition"] == condition}
        assert sorted(rows) == sorted(GAMES), (model, condition, sorted(rows))
        first = FIRST_NAMED[(model, condition)]
        games = {g: int(first[g] == g) for g in GAMES}
        assert all(bool(rows[g]["correct"]) == bool(games[g]) and rows[g]["said"] == first[g] for g in GAMES), (model, condition)
        out[model] = {"named": [sum(games.values()), len(games)], "games": games}
    return out


def main():
    k5 = json.loads((REC / "suite_stage5" / "key.json").read_text())
    k6 = json.loads((REC / "suite_stage6" / "key.json").read_text())
    scores = json.loads((REC / "suite_stage5" / "scores.json").read_text())
    published, baseline_max = {}, {}
    if SITE:
        site = json.loads(SITE.read_text())
        published = {r["id"]: r for r in site["rewrites"]}
        # Matched-effort runs, when the site data carries them: "astra_max" on every rewrite row and a top-level
        # "rewrites_baseline" with rows "plain" and "camo" (our "copy"). Only when all nine rows have one are the
        # figure's Astra scores switched to max; a partial set is ignored so the figure never mixes settings.
        baseline_max = {{"camo": "copy"}.get(r["id"], r["id"]): r for r in site.get("rewrites_baseline", [])}
    use_max = (len(published) == 7 and all("astra_max" in r for r in published.values())
               and all("astra_max" in baseline_max.get(k, {}) for k in ("plain", "copy")))

    real = lambda n: {"n": n}  # noqa: E731
    added = lambda n: {"n": n, "added": True}  # noqa: E731
    spec = [
        ("plain", "tanh", "suite_stage5", k5, "plain", lambda w: [real(w[0]), real(w[1])]),
        ("copy", "tanh", "suite_stage5", k5, "camo_v2", lambda w: [real(w[0]), real(w[1])]),
        ("synth", "tanh", "suite_stage5", k5, "synth", lambda w: [real(w[0]), real(w[1])]),
        ("pairs", "tanh", "suite_stage5", k5, "pairs", lambda w: [real(w[0]), {"n": 32, "pairs": (w[1] - 32) // 2}]),
        ("basis", "tanh", "suite_stage5", k5, "basis", lambda w: [real(w[0]), real(w[1])]),
        ("split", "tanh", "suite_stage6", k6, "split", lambda w: [{"n": w[0] // 2, "twins": True}, {"n": w[1] // 2, "twins": True}]),
        ("relu0", "relu", "suite_stage6", k6, "deep0", lambda w: [real(x) for x in w]),
        ("relu3", "relu", "suite_stage6", k6, "deep3", lambda w: [real(x) if i % 2 == 0 else added(x) for i, x in enumerate(w)]),
        ("relu6", "relu", "suite_stage6", k6, "deep6", lambda w: [real(x) if i % 3 == 0 else added(x) for i, x in enumerate(w)]),
    ]
    rows = []
    for rid, group, suite, key, condition, layers in spec:
        z = pong_net(suite, key, condition)
        w = widths(z)
        row = {"id": rid, "group": group, "maps": maps(z["W1"]), "layers": layers(w)}
        assert sum(l["n"] * (2 if l.get("twins") else 1) + 2 * l.get("pairs", 0) for l in row["layers"]) == sum(w), (rid, w)
        if use_max:
            src = baseline_max[rid] if rid in ("plain", "copy") else published[rid]
            row.update({"astra": src["astra_max"], "opus": src["opus"]})
        elif rid in ("plain", "copy"):
            row.update(baseline(scores, condition))
        elif published:
            row.update({m: published[rid][m] for m in ("astra", "opus")})
        rows.append(row)
        print(rid, group, w, {m: row[m]["named"] for m in ("astra", "opus") if m in row})

    untouched = {
        "tanh": np.load(HERE.parent / "studies" / "layout-vs-behaviour" / "experiments" / "train" / "g_pong" / "actor.npz"),
        "relu": np.load(HERE.parent / "studies" / "layout-vs-behaviour" / "experiments" / "train" / "g_pong_relu24x3" / "actor.npz"),
    }
    # The suite stores each network as printed (3 decimals), so compare with that tolerance.
    assert np.allclose(untouched["tanh"]["W1"], pong_net("suite_stage5", k5, "plain")["W1"], atol=1e-3)
    # The untouched tanh picture is the "plain" row's own (the suite's network as printed), so the two are identical.
    before = {"tanh": {"maps": rows[0]["maps"]}, "relu": {"maps": maps(untouched["relu"]["W1"])}}
    out = {"astra_effort": "max" if use_max else "high", "before": before, "rows": rows}
    (HERE / "disguises.json").write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print("disguises.json", (HERE / "disguises.json").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
