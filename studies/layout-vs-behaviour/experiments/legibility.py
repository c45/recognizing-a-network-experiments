"""How much of each game's layout the human-legible readers can still see, scored against the truth.

Usage: python3 -I legibility.py <suite dir> <train dir> [train suffix, e.g. _relu24x3]
For every network in the suite: each battery map (size, centered, neighbour, agent~prev) for channels 0-2 is
correlated with the game's true occupancy map for that channel (which cells the input is ever on, from the trained
states; flipped to match the network). High |r| means the layout is still legible from that reader.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from battery import buttons, load, maps  # noqa: E402


def occupancy(states, flip):
    occ = (states.max(0) > 0).reshape(10, 10, 5).astype(float)
    if flip == "lr":
        occ = occ[:, ::-1]
    elif flip == "ud":
        occ = occ[::-1]
    return occ


def main(suite, train, suffix=""):
    suite, train = Path(suite), Path(train)
    key = json.loads((suite / "key.json").read_text())
    occ_cache, rows = {}, {}
    for nid, k in key.items():
        if k["game"] == "untrained":
            continue
        g = k["game"]
        if g not in occ_cache:
            occ_cache[g] = np.load(train / f"g_{g}{suffix}" / "states.npy")
        occ = occupancy(occ_cache[g], k["flip"])
        w = load(suite / "nets" / nid / "actor.txt")
        M = maps(w["W1"])
        best = {}
        for name, m in M.items():
            for ch in (0, 1, 2):
                truth = occ[1:9, 1:9, ch].ravel()
                if truth.std() == 0:
                    continue
                r = float(np.corrcoef(m[1:9, 1:9, ch].ravel(), truth)[0, 1])
                best[f"{name}/ch{ch}"] = round(r, 2)
        top = max(best.items(), key=lambda kv: abs(kv[1]))
        rows[nid] = {**k, "max_abs_r": top, "all_r": best, "buttons": buttons(w)["closest_pairs"][:2]}
    (suite / "legibility.json").write_text(json.dumps(rows, indent=1))
    for nid, v in sorted(rows.items(), key=lambda kv: (kv[1]["game"], kv[1]["condition"], str(kv[1]["flip"]))):
        print(f"{v['game']:9s} {v['condition']:6s} flip={str(v['flip']):4s} seed={v['seed']}  strongest layout "
              f"correlation {v['max_abs_r'][0]:16s} r={v['max_abs_r'][1]:+.2f}   closest buttons {v['buttons']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *sys.argv[3:4])
