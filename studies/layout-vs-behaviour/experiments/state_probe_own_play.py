"""Behaviour check of each twin-study policy on states from its own final-policy play.

Usage: python3 state_probe_own_play.py [out.json]      (default state_probe_own_play.json)
For each of the six twin-study games and both training seeds, plays the trained policy (64 environments, 400 steps,
actions sampled from the policy) and reports the expected step along the agent -> nearest-mover direction over states
with the nearest mover within 3 cells: positive = moves toward movers, negative = away. Where the sampled training
states (train/g_<game>/states.npy, not stored in git; see DATA.md) are present, the same number is also reported on a
20,000-state sample of them ("training_sample").
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "train"))
from games import GAMES  # noqa: E402

MOVES = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}  # action -> (row, column) step


def forward(w, X):
    n = len([k for k in w if k[0] == "W"])
    h = X
    for k in range(1, n):
        h = np.tanh(h @ w[f"W{k}"] + w[f"b{k}"])
    return h @ w[f"W{n}"] + w[f"b{n}"]


def softmax(l):
    e = np.exp(l - l.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def state_probe(w, X):
    G = X.reshape(-1, 10, 10, 5)
    p = softmax(forward(w, X.astype(np.float64)))
    step = p @ np.array([MOVES[a] for a in range(5)], float)
    res = []
    for i in range(len(X)):
        ag = np.argwhere(G[i, :, :, 0] > 0)
        ms = np.argwhere(G[i, :, :, 1] > 0)
        if len(ag) == 0 or len(ms) == 0:
            continue
        pair = np.abs(ms[:, None, :] - ag[None, :, :]).sum(2)    # movers x agent cells (paddles span 2 cells)
        j, k = np.unravel_index(int(pair.argmin()), pair.shape)
        if pair[j, k] == 0 or pair[j, k] > 3:
            continue
        v = (ms[j] - ag[k]) / np.linalg.norm(ms[j] - ag[k])
        res.append(float(step[i] @ v))
    return float(np.mean(res)) if res else 0.0


def main(out_path):
    out = {}
    for game in ("catch", "dodge", "pacman", "hunter", "freeway", "cartag"):
        for suf in ("", "_s2"):
            d = HERE / "train" / f"g_{game}{suf}"
            w = dict(np.load(d / "actor.npz"))
            rng = np.random.default_rng(5)
            env = GAMES[game](64, rng)
            X = []
            for _ in range(400):
                o = env.obs().astype(np.float64)
                X.append(o.copy())
                p = softmax(forward(w, o))
                a = (p.cumsum(1) > rng.random((64, 1))).argmax(1)
                _, done = env.step(a)
                env.reset(done)
            out[f"{game}{suf}"] = {"own_play": round(state_probe(w, np.concatenate(X).astype(np.float32)), 4)}
            if (d / "states.npy").exists():
                early = np.load(d / "states.npy", mmap_mode="r")
                sel = np.sort(np.random.default_rng(0).choice(len(early), 20000, replace=False))
                out[f"{game}{suf}"]["training_sample"] = round(state_probe(w, np.asarray(early[sel], np.float32)), 4)
            print(game + suf, out[f"{game}{suf}"], flush=True)
    Path(out_path).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "state_probe_own_play.json")
