"""Optimize the never-used rows of a camouflaged network so filled rows look like real ones to a broad set of readers.

Usage: python3 -I optimize_fill.py <camouflaged actor.npz> <states.npy> <flip none|lr|ud> <out actor.npz>
                                   [--steps 3000] [--seed 0]
Only rows of inputs that are never on in the reachable states change, so play is unchanged.
Optimized features (per row): size, largest entry, sum, spread, similarity to b1, similarity to grid neighbours at
distance 1 (straight and diagonal) and 2, similarity to the 4 other channels in the same cell, and the row's
linearized effect on the 5 action logits (at zero input, centred across actions). For each feature and channel the loss
drives the correlation with the true occupancy map to zero and matches the mean and spread over used vs unused cells.
Held-out check (never optimized), in evaluate(): a gradient-boosted classifier, 5-fold cross-validated, trying to tell
used from unused rows from different features (sorted absolute values in 8 quantiles, projections on the top 8
principal directions of W1, distance to the 3 nearest other rows, linearized action effects at the mean reachable
state); plus the layout correlation of each of those features.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weights_text import to_text  # noqa: E402

KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")


def occupancy(states, flip):
    occ = (states.max(0) > 0).reshape(10, 10, 5).astype(np.float64)
    if flip == "lr":
        occ = occ[:, ::-1]
    elif flip == "ud":
        occ = occ[::-1]
    return np.ascontiguousarray(occ)


def unit(R):
    return R / R.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def neigh(U, offsets):
    out = torch.zeros(U.shape[:3], dtype=U.dtype)
    cnt = torch.zeros(U.shape[:3], dtype=U.dtype)
    for dr, dc in offsets:
        a = U[max(dr, 0):10 + min(dr, 0), max(dc, 0):10 + min(dc, 0)]
        b = U[max(-dr, 0):10 + min(-dr, 0), max(-dc, 0):10 + min(-dc, 0)]
        sim = (a * b).sum(-1).abs()
        out[max(-dr, 0):10 + min(-dr, 0), max(-dc, 0):10 + min(-dc, 0)] += sim
        cnt[max(-dr, 0):10 + min(-dr, 0), max(-dc, 0):10 + min(-dc, 0)] += 1
    return out / cnt


def effects(W1, w, x0=None):
    """Linearized change of the 5 action logits per unit of each input, around input x0 (default: zero)."""
    b1, W2, b2, W3 = (torch.as_tensor(w[k]) for k in ("b1", "W2", "b2", "W3"))
    pre1 = b1 if x0 is None else torch.as_tensor(x0) @ W1 + b1
    d1 = 1 - torch.tanh(pre1) ** 2
    d2 = 1 - torch.tanh(torch.tanh(pre1) @ W2 + b2) ** 2
    e = (W1 * d1) @ (W2 * d2) @ W3
    return e - e.mean(-1, keepdim=True)


def features(W1, w):
    R = W1.reshape(10, 10, 5, 32)
    U = unit(R)
    f = {"size": R.norm(dim=-1), "max": R.abs().amax(-1), "sum": R.sum(-1), "spread": R.std(-1),
         "to_bias": (U * unit(torch.as_tensor(w["b1"]))).sum(-1).abs(),
         "n_straight": neigh(U, [(-1, 0), (1, 0), (0, -1), (0, 1)]),
         "n_diag": neigh(U, [(-1, -1), (-1, 1), (1, -1), (1, 1)]),
         "n_dist2": neigh(U, [(-2, 0), (2, 0), (0, -2), (0, 2)])}
    for k in range(1, 5):
        f[f"cross{k}"] = (U * U.roll(k, dims=2)).sum(-1).abs()
    E = effects(W1, w).reshape(10, 10, 5, 5)
    for a in range(5):
        f[f"effect{a}"] = E[..., a]
    f["effect_size"] = E.norm(dim=-1)
    return f


def corr(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-12)


def evaluate(W1, w, occ, states, flip, seed):
    """Held-out readers: a cross-validated classifier and layout correlations on features never optimized."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    Wn = W1.numpy() if torch.is_tensor(W1) else W1
    q = np.quantile(np.abs(Wn), np.linspace(0, 1, 8), axis=1).T
    _, _, vt = np.linalg.svd(Wn - Wn.mean(0), full_matrices=False)
    pcs = Wn @ vt[:8].T
    d = np.linalg.norm(Wn[:, None] - Wn[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    near = np.sort(d, axis=1)[:, :3]
    xm = states.mean(0)
    if flip == "lr":
        xm = xm.reshape(10, 10, 5)[:, ::-1].reshape(500)
    elif flip == "ud":
        xm = xm.reshape(10, 10, 5)[::-1].reshape(500)
    e2 = effects(torch.as_tensor(Wn), w, x0=np.ascontiguousarray(xm)).numpy()
    X = np.hstack([q, pcs, near, e2])
    y = occ.reshape(500).astype(int)
    interior = np.array([1 <= (i // 50) <= 8 and 1 <= (i // 5) % 10 <= 8 for i in range(500)])
    res = {}
    try:
        auc = cross_val_score(HistGradientBoostingClassifier(max_iter=200, random_state=seed), X[interior],
                              y[interior], cv=5, scoring="roc_auc")
        res["detector_auc_mean"] = round(float(auc.mean()), 3)
    except ValueError:
        res["detector_auc_mean"] = None
    rs = {}
    names = [f"q{k}" for k in range(8)] + [f"pc{k}" for k in range(8)] + [f"near{k}" for k in range(3)] + \
        [f"effect_mean_state{k}" for k in range(5)]
    for j, n in enumerate(names):
        col = X[:, j].reshape(10, 10, 5)
        for ch in range(5):
            t = occ[1:9, 1:9, ch].ravel()
            if t.std() == 0:
                continue
            v = col[1:9, 1:9, ch].ravel()
            if v.std() == 0:
                continue
            rs[f"{n}/ch{ch}"] = round(float(np.corrcoef(v, t)[0, 1]), 3)
    res["heldout_max_layout_r"] = max(rs.items(), key=lambda kv: abs(kv[1])) if rs else None
    return res


def layout_scores(f, occ):
    out = {}
    for name, m in f.items():
        for ch in range(5):
            t = occ[1:9, 1:9, ch].reshape(-1)
            if t.std() == 0:
                continue
            out[f"{name}/ch{ch}"] = float(corr(m[1:9, 1:9, ch].reshape(-1), t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("states")
    ap.add_argument("flip")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    z = np.load(a.actor)
    w = {k: z[k].astype(np.float64) for k in KEYS}
    states = np.load(a.states)
    occ_np = occupancy(states, a.flip)
    occ = torch.tensor(occ_np)
    dead = torch.tensor(occ_np.reshape(500) == 0)
    W1 = torch.tensor(w["W1"])
    free = W1[dead].clone().requires_grad_(True)
    opt = torch.optim.Adam([free], lr=0.01 * float(W1.norm(dim=1).mean()))
    top = lambda d: max(d.items(), key=lambda kv: abs(kv[1]))  # noqa: E731
    before_opt = top(layout_scores(features(W1, w), occ))
    before_held = evaluate(W1, w, occ_np, states, a.flip, a.seed)
    for step in range(a.steps):
        Wc = W1.clone()
        Wc[dead] = free
        loss = 0.0
        for name, m in features(Wc, w).items():
            for ch in range(5):
                t = occ[1:9, 1:9, ch].reshape(-1)
                if t.std() == 0:
                    continue
                x = m[1:9, 1:9, ch].reshape(-1)
                on, off = x[t > 0], x[t == 0]
                sd = x.std().clamp_min(1e-9)
                loss = loss + corr(x, t) ** 2 + ((on.mean() - off.mean()) / sd) ** 2 + \
                    ((on.std() - off.std()) / sd) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    Wc = W1.clone()
    Wc[dead] = free.detach()
    w["W1"] = Wc.numpy()
    after_opt = layout_scores(features(Wc, w), occ)
    after_held = evaluate(Wc, w, occ_np, states, a.flip, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: v.astype(np.float32) for k, v in w.items()})
    (out.parent / "actor.txt").write_text(to_text(w))
    rep = {"steps": a.steps, "final_loss": float(loss.detach()), "rows_optimized": int(dead.sum()),
           "optimized_before_max_r": before_opt, "optimized_after_max_r": top(after_opt),
           "heldout_before": before_held, "heldout_after": after_held}
    (out.parent / "optimize_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
