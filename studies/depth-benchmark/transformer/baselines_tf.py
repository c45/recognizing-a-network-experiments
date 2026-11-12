"""Pre-registered baselines B1-B4 and the reference K-composition script for transformer family T1.

Usage: python3 -I baselines_tf.py <nets dir> <out json>
Leave-one-seed-out logistic regression for B1-B4; the reference script is a fixed sign rule with no fitting.
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


def ci(c, n):
    lo, hi = binomtest(c, n).proportion_ci(0.95, method="exact")
    return {"correct": int(c), "n": int(n), "acc": round(c / n, 3), "ci95": [round(lo, 3), round(hi, 3)]}


def layers(w):
    return sorted(int(k[1:]) for k in w if k.startswith("Q"))


def offset_scores(P, Q, K):
    S = (P @ Q) @ (P @ K).T
    return [np.mean([S[t, t + o] for t in range(S.shape[0]) if 0 <= t + o]) for o in (0, -1, -2)]


def features(w):
    E, P, U = w["E"], w["P"], w["U"]
    L = layers(w)
    b1 = np.r_[np.linalg.norm(E, axis=1), np.linalg.norm(U, axis=0)]
    sv = [np.linalg.svd(w[k], compute_uv=False) for k in sorted(w)]
    b2 = np.concatenate([np.sort(np.linalg.norm(E, axis=1)), np.sort(np.linalg.norm(P, axis=1)), *sv])
    b3 = np.linalg.norm(P, axis=1)
    b4 = [np.mean(np.diag(E @ U))]
    for l in L:
        b4.append(np.mean(np.diag(E @ w[f"V{l}"] @ w[f"O{l}"] @ U)))
        b4 += offset_scores(P, w[f"Q{l}"], w[f"K{l}"])
    return b1, b2, b3, np.array(b4)


def reference(w, tokens=None):
    """Mean diagonal of the layer-2 K-composed match minus the direct token match, over the given query tokens."""
    E = w["E"].astype(float)
    q = E @ w["Q2"]
    idx = np.arange(E.shape[0]) if tokens is None else np.asarray(tokens)
    prev_info = np.diag(q @ (E @ w["V1"] @ w["O1"] @ w["K2"]).T)[idx].mean()
    same_tok = np.diag(q @ (E @ w["K2"]).T)[idx].mean()
    return float(prev_info - same_tok)


def main(nets, out):
    files = sorted(f for f in Path(nets).glob("*.npz")
                   if json.loads(f.with_suffix(".json").read_text())["heldout_acc"] >= 0.95)
    W = [dict(np.load(f)) for f in files]
    y = np.array([f.stem.startswith(("next_", "evennext_")) for f in files], int)
    interaction = files[0].stem.startswith(("evennext_", "evenprev_"))
    seeds = np.array([int(f.stem.rsplit("_s", 1)[1]) for f in files])
    feats = list(zip(*[features(w) for w in W]))
    res = {"n_networks": len(files)}
    for name, X in zip(["B1_token_footprint", "B2_order_free", "B3_position_footprint", "B4_single_layer_circuits"], feats):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        pred = cross_val_predict(clf, np.array(X), y, cv=LeaveOneGroupOut(), groups=seeds)
        res[name] = ci(int((pred == y).sum()), len(y))
    s = np.array([reference(w, np.arange(0, 16, 2) if interaction else None) for w in W])
    res["reference_kcomp_sign_rule"] = ci(int(((s > 0).astype(int) == y).sum()), len(y))
    res["reference_kcomp_median"] = {"next": round(float(np.median(s[y == 1])), 3), "prev": round(float(np.median(s[y == 0])), 3)}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
