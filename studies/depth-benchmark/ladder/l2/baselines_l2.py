"""Ladder L2 baselines B1-B4 (PREREG.md): B1-B3 as in pilot F1; B4 = the pilot's direction-blind linear reading,
mapping a positive ball slope to fall_toward.

Usage: python3 -I baselines_l2.py <nets dir> <states.npz> <out json>
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
from baselines import ci, feats, linear_slope  # noqa: E402


def main(nets_dir, states, out):
    files = sorted(Path(nets_dir).glob("*.npz"))
    W = [dict(np.load(f)) for f in files]
    y = np.array([f.stem.startswith(("fall_toward", "right_toward")) for f in files], int)
    seeds = np.array([int(f.stem.rsplit("_s", 1)[1]) for f in files])
    norms = np.array([np.linalg.norm(w["W1"], axis=1) for w in W])
    res = {"n_networks": len(files), "inputs_ever_on": int((np.load(states)["obs"].max(0) > 0).sum()),
           "mean_rownorm_map_corr": round(float(np.corrcoef(norms[y == 1].mean(0), norms[y == 0].mean(0))[0, 1]), 4)}
    F = [feats(w) for w in W]
    for b in ("B1", "B2", "B3"):
        X = np.array([f[b] for f in F])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        pred = cross_val_predict(clf, X, y, cv=LeaveOneGroupOut(), groups=seeds)
        res[b] = ci(int((pred == y).sum()), len(y))
    slopes = np.array([linear_slope(w)[0] for w in W])
    res["B4_linear_reading"] = ci(int(((slopes > 0).astype(int) == y).sum()), len(y))
    res["B4_slopes"] = {"fall_toward": np.round(slopes[y == 1], 3).tolist(), "fall_away": np.round(slopes[y == 0], 3).tolist()}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "B4_slopes"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
