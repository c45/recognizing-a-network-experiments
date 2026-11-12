"""Camouflage v2: exact on reachable states, without null-space noise, with mirror decoys in never-used rows.

Usage: python3 -I camouflage_v2.py <actor.npz> <states.npy> <out dir> [--seed 0] [--flip none|lr|ud]
                                    [--mirror lr|ud|none] [--jitter 0.15]
Differences from camouflage.py: used rows keep their trained values (no noise along conserved input combinations,
which had left a readable sign pattern and let constructed valid states cancel the noise). Every never-used row gets
a lightly perturbed copy of a used row from the same channel: its mirror-image cell's row when that cell is used
(left-right or top-bottom, the axis of the flipped twins), otherwise a random used row of that channel. The hidden-unit shuffle,
sign flips and the shared output shift are as before. Writes actor.npz, actor.txt, reference_flipped.npz,
camouflage_report.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camouflage import KEYS, flip_inputs, logits  # noqa: E402
from weights_text import to_text  # noqa: E402


def cell(i):
    return (i // 5) // 10, (i // 5) % 10, i % 5


def idx(r, c, ch):
    return (r * 10 + c) * 5 + ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("states")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flip", default="none", choices=["none", "lr", "ud"])
    ap.add_argument("--mirror", default="lr", choices=["none", "lr", "ud"])
    ap.add_argument("--jitter", type=float, default=0.0)
    a = ap.parse_args()
    z = np.load(a.actor)
    w = {k: z[k].astype(np.float64) for k in KEYS}
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
        w["W3"], w["b3"] = w["W3"][:, swap], w["b3"][swap]
    flipped = {k: v.copy() for k, v in w.items()}
    perm = rng.permutation(len(X))
    fit, hold = X[perm[: int(0.8 * len(X))]], X[perm[int(0.8 * len(X)):]]
    dead = fit.max(0) == 0
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
        W1[i] = w["W1"][donor] + a.jitter * np.linalg.norm(w["W1"][donor]) / np.sqrt(32) * rng.standard_normal(32)
    w["W1"] = W1
    for Win, bin_, Wout in [("W1", "b1", "W2"), ("W2", "b2", "W3")]:
        pi = rng.permutation(32)
        sg = rng.choice([-1.0, 1.0], 32)
        w[Win] = w[Win][:, pi] * sg
        w[bin_] = w[bin_][pi] * sg
        w[Wout] = w[Wout][pi, :] * sg[:, None]
    u = rng.standard_normal(32) * 3 * float(np.linalg.norm(w["W3"], axis=0).max())
    w["W3"] = w["W3"] + u[:, None]
    w["b3"] = w["b3"] + rng.normal() * 3

    def softmax(l):
        e = np.exp(l - l.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    Xall = np.vstack([fit, hold])
    rep = {"source": a.actor, "flip": a.flip, "mirror": a.mirror, "seed": a.seed, "unique_states": int(len(X)),
           "never_used_inputs": int(dead.sum()), "fill_kinds": kinds,
           "holdout_states_using_never_used_input": int(hold[:, dead].sum()),
           "max_prob_change_holdout_exact": float(np.abs(softmax(logits(w, hold)) - softmax(logits(flipped, hold))).max()),
           "max_prob_change_all_as_printed": float(np.abs(
               softmax(logits({k: np.round(v, 3) for k, v in w.items()}, Xall)) -
               softmax(logits({k: np.round(v, 3) for k, v in flipped.items()}, Xall))).max())}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "actor.npz", **{k: v.astype(np.float32) for k, v in w.items()})
    np.savez(out / "reference_flipped.npz", **{k: v.astype(np.float32) for k, v in flipped.items()})
    (out / "actor.txt").write_text(to_text(w))
    (out / "camouflage_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
