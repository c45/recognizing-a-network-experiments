"""Is the catch/dodge pair matched on the footprint? And do simple weight statistics still tell the two apart?

Usage: python3 -I pair_check.py <runs dir> <out json>
For the four networks (catch s0/s1, dodge s0/s1):
- never-on input sets over the whole training run (occupancy_never_on.npy): sizes and pairwise Jaccard similarity;
- simple statistics a reader could compute: the first-layer row-norm map (Spearman correlation between networks),
  output biases b3, and the output-layer button similarity;
- the policy's behaviour: how often the agent ends a step in the falling object's column, from 64 envs x 1,000 steps.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from games_new import ALL  # noqa: E402
from legibility_fade import buttons  # noqa: E402

NETS = ["g_catch_s0", "g_catch_s1", "g_dodge_s0", "g_dodge_s1"]


def rank(a):
    return np.argsort(np.argsort(a))


def under_object(game, w, n=64, steps=1000, seed=0):
    rng = np.random.default_rng(seed)
    env = ALL[game](n, rng)
    same, total = 0, 0
    for _ in range(steps):
        x = env.obs()
        h = np.tanh(x @ w["W1"] + w["b1"])
        h = np.tanh(h @ w["W2"] + w["b2"])
        z = h @ w["W3"] + w["b3"]
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        a = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
        r, d = env.step(a)
        same += int((env.ac == env.ocol).sum())
        total += n
        env.reset(d)
    return same / total


def main(runs, out):
    runs = Path(runs)
    w = {n: dict(np.load(runs / n / "actor.npz")) for n in NETS}
    never = {n: np.load(runs / n / "occupancy_never_on.npy") for n in NETS}
    norms = {n: np.linalg.norm(w[n]["W1"], axis=1) for n in NETS}
    rep = {"never_on": {n: int(never[n].sum()) for n in NETS}, "jaccard_used": {}, "norm_spearman": {},
           "b3": {n: np.round(w[n]["b3"], 3).tolist() for n in NETS}, "buttons": {n: buttons(w[n]) for n in NETS},
           "fraction_steps_in_object_column": {n: round(under_object(n.split("_")[1], w[n]), 3) for n in NETS}}
    for a, b in itertools.combinations(NETS, 2):
        ua, ub = ~never[a], ~never[b]
        rep["jaccard_used"][f"{a}~{b}"] = round(float((ua & ub).sum() / (ua | ub).sum()), 4)
        rep["norm_spearman"][f"{a}~{b}"] = round(float(np.corrcoef(rank(norms[a]), rank(norms[b]))[0, 1]), 3)
    rep["never_on_sets_identical"] = all((never[a] == never[NETS[0]]).all() for a in NETS)
    rep["never_on_union_diff"] = int((never[NETS[0]] != never[NETS[2]]).sum())
    Path(out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "buttons"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
