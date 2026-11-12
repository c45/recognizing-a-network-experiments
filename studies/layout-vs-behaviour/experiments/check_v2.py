"""Leak check for camouflage v2 on flipped twins: do truth-free readers still point to the true side?

Usage: python3 -I check_v2.py <train dir> <work dir>
For Pong (left-right), Breakout and Freeway (top-bottom): builds seed 0 and 1 in both orientations with
camouflage_v2.py, then for each reader and channel compares the two edges (columns 1 vs 8, or rows 1 vs 8) and
records which edge scores higher. A reader leaks when its pick follows the true orientation in all four networks.
Readers: size, centered size, neighbour similarity, agent~prev, sign alternation along the edge (adjacent rows or
columns pointing in opposite directions), linearized action-effect size. Also reports the strongest layout
correlation (legibility.py maps) and the within-channel detector AUC.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from battery import load, maps  # noqa: E402
from optimize_fill import effects  # noqa: E402

GAMES = {"pong": "lr", "breakout": "ud", "freeway": "ud"}
JITTER = "0"


def alternation(W1, axis):
    R = W1.reshape(10, 10, 5, 32)
    U = R / np.maximum(np.linalg.norm(R, axis=-1, keepdims=True), 1e-9)
    if axis == "lr":
        s = -(U[:-1] * U[1:]).sum(-1)
        out = np.zeros((10, 10, 5))
        out[:-1] += s
        out[1:] += s
        return out / 2
    s = -(U[:, :-1] * U[:, 1:]).sum(-1)
    out = np.zeros((10, 10, 5))
    out[:, :-1] += s
    out[:, 1:] += s
    return out / 2


def edge_pick(m, axis, ch):
    a = m[1:9, 1, ch].mean() if axis == "lr" else m[1, 1:9, ch].mean()
    b = m[1:9, 8, ch].mean() if axis == "lr" else m[8, 1:9, ch].mean()
    if axis == "lr":
        return "left" if a > b else "right"
    return "top" if a > b else "bottom"


def main(train, work):
    import torch
    train, work = Path(train), Path(work)
    res = {}
    for game, axis in GAMES.items():
        g = train / f"g_{game}"
        truth = {("none", 0): None, ("none", 1): None, (axis, 0): None, (axis, 1): None}
        picks = {}
        for flip, seed in truth:
            d = work / f"{game}_{flip}_{seed}"
            subprocess.run([sys.executable, "-I", str(HERE / "camouflage_v2.py"), str(g / "actor.npz"),
                            str(g / "states.npy"), str(d), "--seed", str(seed), "--flip", flip, "--mirror", axis,
                            "--jitter", JITTER],
                           check=True, capture_output=True)
            w = load(d / "actor.npz")
            M = maps(w["W1"])
            M["alternation"] = alternation(w["W1"], axis)
            E = effects(torch.tensor(w["W1"]), w).numpy().reshape(10, 10, 5, 5)
            M["effect_size"] = np.linalg.norm(E, axis=-1)
            orig_side = {"pong": "left", "breakout": "bottom", "freeway": "top"}[game]
            flipped_side = {"left": "right", "bottom": "top", "top": "bottom"}[orig_side]
            true_side = orig_side if flip == "none" else flipped_side
            for name, m in M.items():
                for ch in range(5):
                    picks.setdefault(f"{name}/ch{ch}", []).append(edge_pick(m, axis, ch) == true_side)
        leaks = {k: v for k, v in picks.items() if all(v) or not any(v)}
        res[game] = {"readers_tracking_truth_all_4": sorted(k for k, v in leaks.items() if all(v)),
                     "readers_anti_tracking_all_4": sorted(k for k, v in leaks.items() if not any(v)),
                     "n_readers": len(picks)}
        print(game, json.dumps(res[game]))
    (work / "check_v2.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
