"""The analysts' curvature contraction, run as a fixed weight-only script on every network of every depth.

Usage: python3 -I curvature_recipe.py <out json>
For each network: q_ball, q_prev = quadratic coefficients (per first-layer unit) of the column profile of the
current-ball (channel 1) and previous-frame (channel 4) W1 rows, averaged over rows 4-6; d = right-minus-left
column of W2 @ ... @ W{L} (tanh ignored). Readout s = (q_ball - q_prev) . d; the sign is mapped to a class by
leave-one-seed-out majority, and reported with |s| relative to its spread. No input is evaluated.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
L2 = HERE.parent / "l2"
SETS = {2: L2 / "nets_d", 3: HERE / "nets_d3", 4: HERE / "nets_d4", 6: HERE / "nets_d6", 8: HERE / "nets_d8"}


def readout(w):
    L = max(int(k[1:]) for k in w if k.startswith("W"))
    A = w["W1"].astype(float).reshape(10, 10, 5, -1)
    c = np.arange(1, 9, dtype=float) - 4.5
    B = np.column_stack([np.ones(8), c, c * c])
    pinv = np.linalg.pinv(B)
    q_ball = (pinv @ A[4:7, 1:9, 1].mean(axis=0))[2]
    q_prev = (pinv @ A[4:7, 1:9, 4].mean(axis=0))[2]
    T = np.eye(A.shape[-1])
    for k in range(2, L + 1):
        T = T @ w[f"W{k}"].astype(float)
    d = T[:, 4] - T[:, 3]
    return float((q_ball - q_prev) @ d)


def main(out):
    res = {}
    for depth, nets in SETS.items():
        rows = []
        for f in sorted(nets.glob("*.npz")):
            meta = json.loads(f.with_suffix(".json").read_text())
            if meta["heldout_acc"] < 0.95:
                continue
            rows.append((f.stem.startswith("right_toward"), readout(dict(np.load(f)))))
        y = np.array([r[0] for r in rows], int)
        s = np.array([r[1] for r in rows])
        sign_toward = np.sign(np.median(s[y == 1]))
        acc = float(((np.sign(s) == sign_toward).astype(int) == y).mean())
        sep = float(abs(np.median(s[y == 1]) - np.median(s[y == 0])) / (s.std() + 1e-12))
        res[depth] = {"n": len(rows), "accuracy_sign_rule": round(acc, 3), "median_gap_over_sd": round(sep, 3),
                      "median_toward": round(float(np.median(s[y == 1])), 4), "median_away": round(float(np.median(s[y == 0])), 4)}
        print(depth, res[depth])
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
