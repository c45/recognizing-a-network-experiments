"""Exact camouflage of a trained policy: change how the weights look, not what the network does on reachable states.

Usage: python3 -I camouflage.py <actor.npz> <states.npy> <out dir> [--seed 0] [--strength 3.0] [--flip none|lr|ud]
Steps (each preserves the logits on every reachable state, up to float error):
  0. optional flip: relabel the grid left-right (lr) or top-bottom (ud) and swap the matching actions; the result is
     the same policy for the mirrored game (used as a side/orientation control)
  1. null-space noise: with A = [states, 1] (reachable observations plus the constant bias input), every vector n with
     A n = 0 can be added to [W1; b1] as n v^T without changing any hidden pre-activation. Never-used inputs, always-on
     inputs and conserved counts (one ball, a two-cell paddle, ...) all lie in this space. We add random noise in it,
     scaled so that every W1 row is several times the size of the largest trained row.
  2. hidden-unit shuffle and sign flips in both hidden layers (tanh is odd)
  3. output shift: the same random vector added to every W3 column and the same constant to every b3 entry
     (softmax is unchanged)
The null space is fitted on 80% of the states; the report checks the other 20%, and the logit change of the network as
printed (3 decimals) on all states. Writes actor.npz, actor.txt and camouflage_report.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weights_text import to_text  # noqa: E402

KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")


def logits(w, X):
    h = np.tanh(X @ w["W1"] + w["b1"])
    h = np.tanh(h @ w["W2"] + w["b2"])
    return h @ w["W3"] + w["b3"]


def flip_inputs(kind):
    """Permutation p with new input p[i] = old input i under the grid flip."""
    p = np.zeros(500, int)
    for r in range(10):
        for c in range(10):
            nr, nc = (r, 9 - c) if kind == "lr" else (9 - r, c)
            for ch in range(5):
                p[(r * 10 + c) * 5 + ch] = (nr * 10 + nc) * 5 + ch
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("states")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strength", type=float, default=3.0)
    ap.add_argument("--flip", default="none", choices=["none", "lr", "ud"])
    ap.add_argument("--jitter", type=float, default=0.15, help="relative noise on copied never-used rows")
    a = ap.parse_args()
    z = np.load(a.actor)
    w = {k: z[k].astype(np.float64) for k in KEYS}
    X = np.load(a.states).astype(np.float64)
    X = np.unique(X, axis=0)
    rng = np.random.default_rng(a.seed)
    orig = {k: v.copy() for k, v in w.items()}

    if a.flip != "none":
        p = flip_inputs(a.flip)
        W1 = np.zeros_like(w["W1"])
        W1[p] = w["W1"]
        w["W1"] = W1
        X2 = np.zeros_like(X)
        X2[:, p] = X
        X = X2
        swap = [0, 1, 2, 4, 3] if a.flip == "lr" else [0, 2, 1, 3, 4]
        w["W3"] = w["W3"][:, swap]
        w["b3"] = w["b3"][swap]
    flipped = {k: v.copy() for k, v in w.items()}

    perm = rng.permutation(len(X))
    fit, hold = X[perm[: int(0.8 * len(X))]], X[perm[int(0.8 * len(X)):]]
    dead = fit.max(0) == 0
    live = np.where(~dead)[0]
    A = np.hstack([fit[:, live], np.ones((len(fit), 1))])
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    rank = int((s > s[0] * 1e-9).sum())
    Nsp = vt[rank:].T
    trained_max = float(np.linalg.norm(w["W1"], axis=1).max())
    noise = Nsp @ rng.standard_normal((Nsp.shape[1], 32))
    scale = a.strength * trained_max / max(float(np.sqrt((np.linalg.norm(noise[:-1], axis=1) ** 2).mean())), 1e-12)
    Wl = np.vstack([w["W1"][live], w["b1"][None, :]]) + scale * noise
    W1 = w["W1"].copy()
    W1[live] = Wl[:-1]
    w["b1"] = Wl[-1]
    # never-used rows: a lightly perturbed copy of the nearest used row in the same channel (same grid row first,
    # then same column, then anywhere), so shared structure from conserved counts spreads over the whole grid
    def cell(i):
        return (i // 5) // 10, (i // 5) % 10

    for ch in range(5):
        live_ch = [i for i in live if i % 5 == ch] or list(live)
        for i in [j for j in np.where(dead)[0] if j % 5 == ch]:
            r, c = cell(i)
            donor = min(live_ch, key=lambda j: (0 if cell(j)[0] == r else 1 if cell(j)[1] == c else 2,
                                                abs(cell(j)[0] - r) + abs(cell(j)[1] - c), rng.random()))
            W1[i] = W1[donor] + a.jitter * np.linalg.norm(W1[donor]) / np.sqrt(32) * rng.standard_normal(32)
    w["W1"] = W1

    for layer, (Win, bin_, Wout) in enumerate([("W1", "b1", "W2"), ("W2", "b2", "W3")]):
        pi = rng.permutation(32)
        sg = rng.choice([-1.0, 1.0], 32)
        w[Win] = w[Win][:, pi] * sg
        w[bin_] = w[bin_][pi] * sg
        w[Wout] = w[Wout][pi, :] * sg[:, None]

    u = rng.standard_normal(32) * a.strength * float(np.linalg.norm(w["W3"], axis=0).max())
    w["W3"] = w["W3"] + u[:, None]
    w["b3"] = w["b3"] + rng.normal() * 3

    rounded = {k: np.round(v, 3) for k, v in w.items()}
    ref_rounded = {k: np.round(v, 3) for k, v in flipped.items()}

    def softmax(l):
        e = np.exp(l - l.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    Xall = np.vstack([fit, hold])
    rep = {
        "source": a.actor, "flip": a.flip, "seed": a.seed, "strength": a.strength,
        "unique_states": int(len(X)), "never_used_inputs": int(dead.sum()), "null_space_dim_live": int(Nsp.shape[1]),
        "rank": rank, "holdout_inputs_never_used_but_on": int(hold[:, dead].sum()),
        "holdout_residual_max": float(np.abs(np.hstack([hold[:, live], np.ones((len(hold), 1))]) @ Nsp).max()),
        "max_prob_change_holdout_exact": float(np.abs(softmax(logits(w, hold)) - softmax(logits(flipped, hold))).max()),
        "max_prob_change_all_as_printed": float(np.abs(softmax(logits(rounded, Xall)) -
                                                       softmax(logits(ref_rounded, Xall))).max()),
        "row_norm_median_before": float(np.median(np.linalg.norm(flipped["W1"], axis=1))),
        "row_norm_median_after": float(np.median(np.linalg.norm(w["W1"], axis=1))),
        "lines_with_number_ge_1_after": int((np.abs(rounded["W1"]) >= 1).any(1).sum()),
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "actor.npz", **{k: v.astype(np.float32) for k, v in w.items()})
    np.savez(out / "reference_flipped.npz", **{k: v.astype(np.float32) for k, v in flipped.items()})
    (out / "actor.txt").write_text(to_text(w))
    (out / "camouflage_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
