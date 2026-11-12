"""Dominant-path tracing, as the tool-free answers describe it, run as a fixed weight-only script at every depth.

Usage: python3 -I greedy_paths.py <out json>
For each network: q = q_ball - q_prev (curvature_recipe.py) per first-layer unit. From the m first-layer units with
the largest |q|, follow the single strongest outgoing weight (largest |W_k[u, v]|) layer by layer to the last hidden
layer, then take the right-minus-left output weight. Each path contributes q_u * product of the signed weights along
it; the readout is their sum. Also a beam variant that keeps the b strongest partial paths per layer. Accuracy uses
the sign rule of curvature_recipe.py. Compares how a few dominant paths fare against the full product as depth grows.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curvature_recipe import SETS  # noqa: E402


def q_units(w):
    A = w["W1"].astype(float).reshape(10, 10, 5, -1)
    c = np.arange(1, 9, dtype=float) - 4.5
    pinv = np.linalg.pinv(np.column_stack([np.ones(8), c, c * c]))
    return (pinv @ A[4:7, 1:9, 1].mean(axis=0))[2] - (pinv @ A[4:7, 1:9, 4].mean(axis=0))[2]


def trace(w, m=3, beam=1):
    L = max(int(k[1:]) for k in w if k.startswith("W"))
    q = q_units(w)
    total = 0.0
    for u in np.argsort(-np.abs(q))[:m]:
        paths = [(q[u], u)]
        for k in range(2, L):
            Wk = w[f"W{k}"].astype(float)
            cand = [(val * Wk[unit, v], v) for val, unit in paths for v in range(Wk.shape[1])]
            cand.sort(key=lambda t: -abs(t[0]))
            paths = cand[:beam]
        out = w[f"W{L}"].astype(float)
        total += sum(val * (out[unit, 4] - out[unit, 3]) for val, unit in paths)
    return total


def accuracy(rows):
    y = np.array([r[0] for r in rows], int)
    s = np.array([r[1] for r in rows])
    sign_toward = np.sign(np.median(s[y == 1])) or 1.0
    return round(float(((np.sign(s) == sign_toward).astype(int) == y).mean()), 3)


def main(out):
    res = {}
    for depth, nets in SETS.items():
        ws = []
        for f in sorted(nets.glob("*.npz")):
            if json.loads(f.with_suffix(".json").read_text())["heldout_acc"] >= 0.95:
                ws.append((f.stem.startswith("right_toward"), dict(np.load(f))))
        res[depth] = {"n": len(ws)}
        for m, beam in [(1, 1), (3, 1), (3, 3), (5, 5)]:
            res[depth][f"m{m}_beam{beam}"] = accuracy([(y, trace(w, m, beam)) for y, w in ws])
        print(depth, res[depth], flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
