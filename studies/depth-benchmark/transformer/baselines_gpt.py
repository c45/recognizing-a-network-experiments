"""Pre-registered baselines B1-B4 and the reference K-composition script for the GPT-mini family G2.

Usage: python3 -I baselines_gpt.py <nets dir> <out json>
Leave-one-seed-out logistic regression for B1-B4; the reference script is a fixed sign rule (LayerNorm gains folded
into the adjacent weights, normalization ignored).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines_tf import ci  # noqa: E402

H, DH = 4, 8


def nlayers(w):
    return max(int(k[2:]) for k in w if k.startswith("Wq"))


def head(M, h, rows=False):
    return M[h * DH:(h + 1) * DH] if rows else M[:, h * DH:(h + 1) * DH]


def offsets(S):
    return [np.mean([S[t, t + o] for t in range(S.shape[0]) if t + o >= 0]) for o in (0, -1, -2)]


def features(w):
    E, P, U = w["E"].astype(float), w["P"].astype(float), w["U"].astype(float)
    L = nlayers(w)
    b1 = np.r_[np.linalg.norm(E, axis=1), np.linalg.norm(U, axis=0), w["c"]]
    mats = [k for k in sorted(w) if w[k].ndim == 2]
    vecs = [k for k in sorted(w) if w[k].ndim == 1]
    b2 = np.concatenate([np.sort(np.linalg.norm(E, axis=1)), np.sort(np.linalg.norm(P, axis=1)),
                         *[np.linalg.svd(w[k], compute_uv=False) for k in mats], *[np.sort(w[k]) for k in vecs]])
    b3 = np.linalg.norm(P, axis=1)
    b4 = [np.mean(np.diag(E @ U))]
    for l in range(1, L + 1):
        for h in range(H):
            b4.append(np.mean(np.diag(E @ head(w[f"Wv{l}"], h) @ head(w[f"Wo{l}"], h, rows=True) @ U)))
            b4 += offsets((P @ head(w[f"Wq{l}"], h)) @ (P @ head(w[f"Wk{l}"], h)).T)
    return b1, b2, b3, np.array(b4)


def reference(w, tokens):
    """Sum over layer-2 heads of the K-composed match minus the direct token match, over the given query tokens."""
    E = w["E"].astype(float)
    g1, g2 = w["ln1_g1"].astype(float), w["ln1_g2"].astype(float)
    ov1 = sum(head(w["Wv1"], h) @ head(w["Wo1"], h, rows=True) for h in range(H))
    total = 0.0
    for h in range(H):
        q = (E * g2) @ head(w["Wq2"], h)
        kc = ((E * g1) @ ov1 * g2) @ head(w["Wk2"], h)
        kd = (E * g2) @ head(w["Wk2"], h)
        total += np.diag(q @ kc.T)[tokens].mean() - np.diag(q @ kd.T)[tokens].mean()
    return float(total)


def main(nets, out):
    files = sorted(f for f in Path(nets).glob("*.npz")
                   if json.loads(f.with_suffix(".json").read_text())["heldout_acc"] >= 0.95)
    W = [dict(np.load(f)) for f in files]
    y = np.array([f.stem.startswith("evennext_") for f in files], int)
    seeds = np.array([int(f.stem.rsplit("_s", 1)[1]) for f in files])
    feats = list(zip(*[features(w) for w in W]))
    res = {"n_networks": len(files)}
    for name, X in zip(["B1_token_footprint", "B2_order_free", "B3_position_footprint", "B4_single_layer_circuits"], feats):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        pred = cross_val_predict(clf, np.array(X), y, cv=LeaveOneGroupOut(), groups=seeds)
        res[name] = ci(int((pred == y).sum()), len(y))
    s = np.array([reference(w, np.arange(0, 16, 2)) for w in W])
    res["reference_kcomp_sign_rule"] = ci(int(((s > 0).astype(int) == y).sum()), len(y))
    res["reference_kcomp_sign_flipped"] = ci(int(((s < 0).astype(int) == y).sum()), len(y))
    res["reference_kcomp_median"] = {"evennext": round(float(np.median(s[y == 1])), 3),
                                     "evenprev": round(float(np.median(s[y == 0])), 3)}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) or "ci95" not in v else f"{v['correct']}/{v['n']} {v['ci95']}")
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
