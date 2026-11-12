"""How much does each policy's action distribution depend on the movers? On states from its own play: mean total
variation distance between the policy's action probabilities on the real observation and on the same observation with
the movers deleted (blind) or swapped for another game's movers (decoy). Also the share of first-layer weight norm on
the mover channel. Writes tv_check.json.

Usage: python -I tv_check.py [game[:suffix] ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_lib as A  # noqa: E402

names = sys.argv[1:] or [g + s for g in ("catch", "dodge", "pacman", "hunter", "freeway", "cartag") for s in ("", ":_s2")]
out = {}
for name in names:
    game, _, suf = name.partition(":")
    w = A.load(game, suf)
    n, steps = 256, 400
    rng = np.random.default_rng(5)
    env = A.games.GAMES[game](n, rng)
    prev0, prev1 = np.zeros((n, 100), np.float32), np.zeros((n, 100), np.float32)
    tv = {"blind": [], "decoy": []}
    flip = {"blind": [], "decoy": []}
    for s in range(steps):
        f = env.frame()
        f0, f1 = f[:, :, 0].copy(), f[:, :, 1].copy()
        obs = {"true": f.copy(), "blind": f.copy(), "decoy": f.copy()}
        obs["true"][:, :, 4] = np.clip(prev0 + prev1, 0, 1)
        obs["blind"][:, :, 1] = 0
        obs["blind"][:, :, 4] = prev0
        obs["decoy"][:, :, 1] = np.roll(f1, 1, 0)
        obs["decoy"][:, :, 4] = np.clip(prev0 + np.roll(prev1, 1, 0), 0, 1)
        p = {k: A.softmax(A.forward(w, v.reshape(n, 500).astype(np.float64))) for k, v in obs.items()}
        for k in tv:
            tv[k].append(0.5 * np.abs(p["true"] - p[k]).sum(1))
            flip[k].append(p["true"].argmax(1) != p[k].argmax(1))
        act = (p["true"].cumsum(1) > rng.random((n, 1))).argmax(1)
        _, d = env.step(act)
        prev0, prev1 = f0, f1
        env.reset(d)
        prev0[d] = 0
        prev1[d] = 0
    W1 = w["W1"].reshape(100, 5, -1)
    norm = np.sqrt((W1 ** 2).sum((0, 2)))
    out[game + suf] = {"tv_blind": round(float(np.mean(tv["blind"])), 4), "tv_decoy": round(float(np.mean(tv["decoy"])), 4),
                       "top_action_changes_blind": round(float(np.mean(flip["blind"])), 4),
                       "top_action_changes_decoy": round(float(np.mean(flip["decoy"])), 4),
                       "w1_norm_share_by_channel": [round(float(x), 3) for x in norm / norm.sum()]}
    print(game + suf, out[game + suf], flush=True)
if not sys.argv[1:]:
    (HERE / "tv_check.json").write_text(json.dumps(out, indent=1))
