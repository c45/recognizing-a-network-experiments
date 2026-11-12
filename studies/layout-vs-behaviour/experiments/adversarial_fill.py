"""Adversarial fill: train the never-used rows against a learned discriminator until real and filled rows look alike.

Usage: python3 -I adversarial_fill.py <camouflaged actor.npz> <states.npy> <flip none|lr|ud> <out actor.npz>
                                      [--rounds 60] [--seed 0]
Only never-used rows change, so play is unchanged. For each input cell the discriminator sees: its row, the mean of its
4 grid neighbours in the same channel, the rows of the other 4 channels in the same cell, and the row's linearized
effect on the 5 actions (centred). It is trained to label used vs unused rows (balanced loss); the unused rows are then
updated to make it output "used", together with the fixed-feature loss of optimize_fill.py. Each round restarts the
discriminator part of the time, so the filled rows cannot overfit one detector. The final check (in optimize_fill's
evaluate) uses a separate gradient-boosted detector on features never trained against, plus a fresh discriminator
of this architecture trained from scratch with 5-fold cross-validation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_fill import KEYS, corr, effects, evaluate, features, layout_scores, occupancy  # noqa: E402
from weights_text import to_text  # noqa: E402


def cell_inputs(W1, w):
    R = W1.reshape(10, 10, 5, 32)
    pad = torch.nn.functional.pad(R.permute(2, 3, 0, 1), (1, 1, 1, 1), mode="replicate").permute(2, 3, 0, 1)
    nb = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]) / 4
    others = torch.cat([R.roll(k, dims=2) for k in range(1, 5)], dim=-1)
    E = effects(W1, w).reshape(10, 10, 5, 5)
    x = torch.cat([R, nb, others, E], dim=-1).reshape(500, -1)
    return x / x.std(0, keepdim=True).clamp_min(1e-6)


def disc(d_in):
    return nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))


def balanced_bce(logit, y):
    w = torch.where(y > 0, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    return nn.functional.binary_cross_entropy_with_logits(logit, y, weight=w)


def fresh_disc_auc(W1, w, y, interior, seed):
    from sklearn.metrics import roc_auc_score
    X = cell_inputs(W1, w).detach()
    idx = np.where(interior)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, 5)
    aucs = []
    for k in range(5):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(5) if j != k])
        D = disc(X.shape[1])
        opt = torch.optim.Adam(D.parameters(), lr=3e-3, weight_decay=1e-3)
        yt = torch.tensor(y[train], dtype=torch.float64).float()
        for _ in range(400):
            opt.zero_grad()
            balanced_bce(D(X[train].float()).squeeze(-1), yt).backward()
            opt.step()
        with torch.no_grad():
            p = D(X[test].float()).squeeze(-1).numpy()
        if len(set(y[test])) == 2:
            aucs.append(roc_auc_score(y[test], p))
    return round(float(np.mean(aucs)), 3) if aucs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("states")
    ap.add_argument("flip")
    ap.add_argument("out")
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    z = np.load(a.actor)
    w = {k: z[k].astype(np.float64) for k in KEYS}
    states = np.load(a.states)
    occ_np = occupancy(states, a.flip)
    occ = torch.tensor(occ_np)
    y_np = occ_np.reshape(500).astype(int)
    interior = np.array([1 <= (i // 50) <= 8 and 1 <= (i // 5) % 10 <= 8 for i in range(500)])
    dead = torch.tensor(y_np == 0)
    y = torch.tensor(y_np, dtype=torch.float32)
    W1 = torch.tensor(w["W1"])
    free = W1[dead].clone().requires_grad_(True)
    g_opt = torch.optim.Adam([free], lr=0.005 * float(W1.norm(dim=1).mean()))
    top = lambda d: max(d.items(), key=lambda kv: abs(kv[1]))  # noqa: E731
    before = {"heldout": evaluate(W1, w, occ_np, states, a.flip, a.seed),
              "fresh_disc_auc": fresh_disc_auc(W1, w, y_np, interior, a.seed)}
    D = None
    for rnd in range(a.rounds):
        Wc = W1.clone()
        Wc[dead] = free
        if D is None or rnd % 3 == 0:
            D = disc(cell_inputs(Wc, w).shape[1])
            d_opt = torch.optim.Adam(D.parameters(), lr=3e-3, weight_decay=1e-3)
        X = cell_inputs(Wc, w).detach().float()
        for _ in range(150):
            d_opt.zero_grad()
            balanced_bce(D(X).squeeze(-1), y).backward()
            d_opt.step()
        for _ in range(40):
            Wc = W1.clone()
            Wc[dead] = free
            X = cell_inputs(Wc, w).float()
            fool = nn.functional.binary_cross_entropy_with_logits(D(X).squeeze(-1)[dead], torch.ones(int(dead.sum())))
            fixed = 0.0
            for name, m in features(Wc, w).items():
                for ch in range(5):
                    t = occ[1:9, 1:9, ch].reshape(-1)
                    if t.std() == 0:
                        continue
                    fixed = fixed + corr(m[1:9, 1:9, ch].reshape(-1), t) ** 2
            loss = fool + 0.2 * fixed
            g_opt.zero_grad()
            loss.backward()
            g_opt.step()
    Wc = W1.clone()
    Wc[dead] = free.detach()
    w["W1"] = Wc.numpy()
    after = {"heldout": evaluate(Wc, w, occ_np, states, a.flip, a.seed),
             "fresh_disc_auc": fresh_disc_auc(Wc, w, y_np, interior, a.seed),
             "optimized_features_max_r": top(layout_scores(features(Wc, w), occ))}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: v.astype(np.float32) for k, v in w.items()})
    (out.parent / "actor.txt").write_text(to_text(w))
    rep = {"rounds": a.rounds, "before": before, "after": after}
    (out.parent / "adversarial_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, default=str))


if __name__ == "__main__":
    main()
