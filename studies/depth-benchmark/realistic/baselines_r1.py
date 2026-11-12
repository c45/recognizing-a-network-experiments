"""Pre-registered R1 baselines: orientation (protected), condition and offset pair (reported), leave-one-group-out.

Usage: python3 -I baselines_r1.py <nets dir> <out json>
A group is (unordered offset pair, condition); both orientations and both seeds of a group are held out together.
Writes per-network B1 orientation decisions to b1_orientation.json (for counterbalancing, if needed).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

H, DH, L = 4, 10, 3


def ci(c, n, p0=0.5):
    lo, hi = binomtest(c, n).proportion_ci(0.95, method="exact")
    return {"correct": int(c), "n": int(n), "acc": round(c / n, 3), "ci95": [round(lo, 3), round(hi, 3)], "chance": round(p0, 3)}


def features(w):
    E, P, U = (w[k].astype(float) for k in ("E", "P", "U"))
    b1 = np.r_[np.linalg.norm(E, axis=1), np.linalg.norm(U, axis=0)]
    b2 = np.concatenate([np.sort(np.linalg.norm(E, axis=1)), np.sort(np.linalg.norm(P, axis=1)),
                         *[np.linalg.svd(w[k], compute_uv=False) for k in sorted(w)]])
    b3 = np.linalg.norm(P, axis=1)
    b4 = [np.mean(np.diag(E @ U))]
    for l in range(1, L + 1):
        for h in range(H):
            s = slice(h * DH, (h + 1) * DH)
            b4.append(np.mean(np.diag(E @ w[f"Wv{l}"][:, s] @ w[f"Wo{l}"][s] @ U)))
            S = (P @ w[f"Wq{l}"][:, s]) @ (P @ w[f"Wk{l}"][:, s]).T
            b4 += [np.mean([S[t, t + o] for t in range(16) if t + o >= 0]) for o in (0, -1, -2, -3)]
    return b1, b2, b3, np.array(b4)


def main(nets, out):
    files, metas = [], []
    for f in sorted(Path(nets).glob("*.npz")):
        m = json.loads(f.with_suffix(".json").read_text())
        if min(m["heldout_acc"].values()) >= 0.95:
            files.append(f)
            metas.append(m)
    W = [dict(np.load(f)) for f in files]
    feats = list(zip(*[features(w) for w in W]))
    group = np.array([f"{min(m['a'], m['b'])}_{max(m['a'], m['b'])}_{m['C']}" for m in metas])
    targets = {
        "orientation": (np.array([m["a"] > m["b"] for m in metas], int), 0.5),
        "condition": (np.array([m["C"] for m in metas]), 0.25),
        "offset_pair": (np.array([f"{min(m['a'], m['b'])}_{max(m['a'], m['b'])}" for m in metas]), 1 / 15),
    }
    res = {"n_networks": len(files)}
    b1_orient = None
    for tname, (y, p0) in targets.items():
        res[tname] = {}
        for bname, X in zip(["B1_token_footprint", "B2_order_free", "B3_position_footprint", "B4_single_layer_circuits"], feats):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
            pred = cross_val_predict(clf, np.array(X), y, cv=LeaveOneGroupOut(), groups=group)
            res[tname][bname] = ci(int((pred == y).sum()), len(y), p0)
            if tname == "orientation" and bname.startswith("B1"):
                b1_orient = {f.stem: int(p) for f, p in zip(files, pred)}
    Path(out).write_text(json.dumps(res, indent=1))
    Path(out).with_name("b1_orientation.json").write_text(json.dumps(b1_orient, indent=1, sort_keys=True))
    for t in targets:
        print(t, {b: f"{v['correct']}/{v['n']} {v['ci95']} (chance {v['chance']})" for b, v in res[t].items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
