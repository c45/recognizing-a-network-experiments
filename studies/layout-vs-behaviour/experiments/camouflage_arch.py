"""Camouflage v2 (Level 2) for any architecture in arch.py: exact on reachable states.

Usage: python3 -I camouflage_arch.py <actor.npz> <states.npy> <out dir> --arch A [--seed 0] [--flip none|lr|ud]
                                      [--mirror lr|ud|none]
Same steps as camouflage_v2.py: optional flip (relabel the grid, swap the matching actions); every never-used W1 row
becomes an exact copy of its mirror-image cell's row when that cell is used, otherwise of a random used row of the
same channel; every hidden layer's units are shuffled, with sign flips for tanh (odd) or positive rescaling by
exp(U(-0.5, 0.5)) for ReLU (positively homogeneous); one random vector is added to every output column and one
constant to every output bias (softmax-invariant). Writes actor.npz, actor.txt, reference_flipped.npz,
camouflage_report.json (exact change on held-out states, change as printed at 3 decimals on all states).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch import ARCHS, keys, logits, to_text  # noqa: E402
from camouflage import flip_inputs  # noqa: E402


def cell(i):
    return (i // 5) // 10, (i // 5) % 10, i % 5


def idx(r, c, ch):
    return (r * 10 + c) * 5 + ch


def softmax(l):
    e = np.exp(l - l.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("states")
    ap.add_argument("out")
    ap.add_argument("--arch", required=True, choices=sorted(ARCHS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flip", default="none", choices=["none", "lr", "ud"])
    ap.add_argument("--mirror", default="lr", choices=["none", "lr", "ud"])
    ap.add_argument("--never-on", help="occupancy_never_on.npy from training: define unused inputs by the whole "
                    "training record instead of the state sample")
    a = ap.parse_args()
    act, hidden = ARCHS[a.arch]
    n = len(hidden) + 1
    z = np.load(a.actor)
    w = {k: z[k].astype(np.float64) for k in keys(a.arch)}
    X = np.unique(np.load(a.states).astype(np.float64), axis=0)
    rng = np.random.default_rng(a.seed)
    if a.flip != "none":
        p = flip_inputs(a.flip)
        W1 = np.zeros_like(w["W1"])
        W1[p] = w["W1"]
        w["W1"] = W1
        X2 = np.zeros_like(X)
        X2[:, p] = X
        X = X2
        swap = [0, 1, 2, 4, 3] if a.flip == "lr" else [0, 2, 1, 3, 4]
        w[f"W{n}"], w[f"b{n}"] = w[f"W{n}"][:, swap], w[f"b{n}"][swap]
    flipped = {k: v.copy() for k, v in w.items()}
    perm = rng.permutation(len(X))
    fit, hold = X[perm[: int(0.8 * len(X))]], X[perm[int(0.8 * len(X)):]]
    dead = fit.max(0) == 0
    if a.never_on:
        dead = np.load(a.never_on).astype(bool)
        if a.flip != "none":
            d2 = np.zeros_like(dead)
            d2[flip_inputs(a.flip)] = dead
            dead = d2
    live = [i for i in range(500) if not dead[i]]
    W1 = w["W1"].copy()
    kinds = {"mirror": 0, "random": 0}
    for i in np.where(dead)[0]:
        r, c, ch = cell(i)
        m = idx(r, 9 - c, ch) if a.mirror == "lr" else idx(9 - r, c, ch) if a.mirror == "ud" else None
        if m is not None and not dead[m]:
            donor = m
            kinds["mirror"] += 1
        else:
            pool = [j for j in live if j % 5 == ch] or live
            donor = pool[rng.integers(len(pool))]
            kinds["random"] += 1
        W1[i] = w["W1"][donor]
    w["W1"] = W1
    for k in range(1, n):
        width = hidden[k - 1]
        pi = rng.permutation(width)
        s = rng.choice([-1.0, 1.0], width) if act == "tanh" else np.exp(rng.uniform(-0.5, 0.5, width))
        w[f"W{k}"] = w[f"W{k}"][:, pi] * s
        w[f"b{k}"] = w[f"b{k}"][pi] * s
        w[f"W{k + 1}"] = w[f"W{k + 1}"][pi, :] / s[:, None]
    u = rng.standard_normal(hidden[-1]) * 3 * float(np.linalg.norm(w[f"W{n}"], axis=0).max())
    w[f"W{n}"] = w[f"W{n}"] + u[:, None]
    w[f"b{n}"] = w[f"b{n}"] + rng.normal() * 3
    Xall = np.vstack([fit, hold])
    r3 = lambda d: {k: np.round(v, 3) for k, v in d.items()}  # noqa: E731
    rep = {"source": a.actor, "arch": a.arch, "flip": a.flip, "mirror": a.mirror, "seed": a.seed,
           "unique_states": int(len(X)), "never_used_inputs": int(dead.sum()), "fill_kinds": kinds,
           "holdout_states_using_never_used_input": int(hold[:, dead].sum()), "dead_from": a.never_on or "state sample",
           "max_prob_change_holdout_exact": float(np.abs(softmax(logits(w, hold, a.arch)) -
                                                         softmax(logits(flipped, hold, a.arch))).max()),
           "max_prob_change_all_as_printed": float(np.abs(softmax(logits(r3(w), Xall, a.arch)) -
                                                          softmax(logits(r3(flipped), Xall, a.arch))).max()),
           "max_prob_change_plain_as_printed": float(np.abs(softmax(logits(r3(flipped), Xall, a.arch)) -
                                                            softmax(logits(flipped, Xall, a.arch))).max())}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "actor.npz", **{k: v.astype(np.float32) for k, v in w.items()})
    np.savez(out / "reference_flipped.npz", **{k: v.astype(np.float32) for k, v in flipped.items()})
    (out / "actor.txt").write_text(to_text(w, a.arch))
    (out / "camouflage_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
