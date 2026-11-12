"""L4-scale baselines for any depth: B1 W1 row norms; B2 sorted W1 row norms + singular values of every weight
matrix + output biases; B3 output biases; B4 linear reading (product of all weight matrices, tanh ignored; the
pilot's ball-slope sign, positive -> right_toward). Leave-one-seed-out, as in pilot F1.

Usage: python3 -I baselines_scale.py <nets dir> <depth> <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_f1"))
from baselines import ci, idx  # noqa: E402


def main(nets_dir, depth, out):
    files = sorted(Path(nets_dir).glob("*.npz"))
    W = [dict(np.load(f)) for f in files]
    y = np.array([f.stem.startswith("right_toward") for f in files], int)
    seeds = np.array([int(f.stem.rsplit("_s", 1)[1]) for f in files])
    L = depth + 1
    feats = {"B1": [], "B2": [], "B3": []}
    slopes = []
    for w in W:
        norms = np.linalg.norm(w["W1"], axis=1)
        sv = [np.linalg.svd(w[f"W{k}"], compute_uv=False) for k in range(1, L + 1)]
        feats["B1"].append(norms)
        feats["B2"].append(np.concatenate([np.sort(norms)[::-1], *sv, w[f"b{L}"]]))
        feats["B3"].append(w[f"b{L}"])
        P = w["W1"]
        for k in range(2, L + 1):
            P = P @ w[f"W{k}"]
        diff = P[:, 4] - P[:, 3]
        cols = np.arange(1, 9)
        ball = np.array([np.mean([diff[idx(r, c, 1)] for r in range(1, 8)]) for c in cols])
        slopes.append(np.polyfit(cols, ball, 1)[0])
    res = {"n_networks": len(files)}
    for b, X in feats.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        pred = cross_val_predict(clf, np.array(X), y, cv=LeaveOneGroupOut(), groups=seeds)
        res[b] = ci(int((pred == y).sum()), len(y))
    res["B4_linear_reading"] = ci(int(((np.array(slopes) > 0).astype(int) == y).sum()), len(y))
    Path(out).write_text(json.dumps(res, indent=1))
    print(nets_dir, {k: (v["correct"], v["ci95"]) for k, v in res.items() if k != "n_networks"})


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
