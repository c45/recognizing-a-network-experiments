"""The post's `rewrites` and `rewrites_baseline` blocks (the "Rewriting the network" figure) from the stored replies.

Usage: python3 -I make_rewrites.py <data.json to patch in place>
Runs ../studies/footprints-and-function/experiments/score_max_parity.py (strict rule: the first game named on the
reply's Game line must be right) and groups its per-network table into one cell per row and arm:
{"named": [right, 4], "games": {game: 1 right, 0 another game, -1 no answer}}.
Arms: `astra` (Astra at high), `astra_max` (Astra at max, the figure's setting) and `opus` (Opus 5.5 at max).
Rows: the two baselines (`plain`, `camo`) go to `rewrites_baseline`, the seven rewrites to `rewrites`.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REC = HERE.parent / "studies" / "footprints-and-function" / "experiments"
GAMES = ("pong", "breakout", "pacman", "freeway")
BASELINE = ("plain", "camo")
REWRITES = ("synth", "pairs", "basis", "split", "relu0", "relu3", "relu6")
GROUP = {"synth": "tanh", "pairs": "tanh", "basis": "tanh", "split": "tanh", "relu0": "relu", "relu3": "relu", "relu6": "relu"}
# MAX_PARITY_PREREG.md, addendum 1: the scorer credits this reply ("Dodge / Avoider ... similar to evading the ghosts in
# Pac-Man") because Dodge is not one of its keywords; the figure does not, because the line leads with another game.
STRICT_OVERRIDES = {("relu3", "opus", "pacman"): 0}


def cell(rows, row, arm):
    games = {}
    for r in rows:
        v = r[arm]
        games[r["game"]] = STRICT_OVERRIDES.get((row, arm, r["game"]), -1 if v is None else v)
    games = {g: games[g] for g in GAMES}
    return {"named": [sum(v == 1 for v in games.values()), len(games)], "games": games}


def main():
    site_path = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "table.json"
        subprocess.run([sys.executable, "-I", str(REC / "score_max_parity.py"), "--json", str(out)], check=True,
                       cwd=REC, stdout=subprocess.DEVNULL)
        table = json.loads(out.read_text())
    by_row = {}
    for r in table:
        by_row.setdefault(r["row"], []).append(r)
    site = json.loads(site_path.read_text())
    site["rewrites"] = [{"id": row, "group": GROUP[row], **{arm: cell(by_row[row], row, arm) for arm in ("astra", "opus", "astra_max")}}
                        for row in REWRITES]
    site["rewrites_baseline"] = [{"id": row, **{arm: cell(by_row[row], row, arm) for arm in ("astra", "astra_max", "opus")}}
                                 for row in BASELINE]
    site_path.write_text(json.dumps(site, separators=(",", ":")) + "\n")
    for r in site["rewrites_baseline"] + site["rewrites"]:
        print(f"{r['id']:6}", {arm: r[arm]["named"] for arm in ("astra", "astra_max", "opus")})


if __name__ == "__main__":
    main()
