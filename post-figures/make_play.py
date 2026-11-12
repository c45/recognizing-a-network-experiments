"""Recorded play of the four 32-32 tanh game policies, for the post's first figure.

Usage: uv run --with numpy python -I make_play.py
Reads ../studies/layout-vs-behaviour/experiments/train/g_<game>/actor.npz and plays each policy in its own game
(train/games.py, our reconstruction of the Exp 1 games) with actions sampled from the softmax, as in training.
Writes play.json next to this file: per game, FRAMES consecutive steps from a fixed seed, and the first ROWS rows of
the first-layer weights as printed in the prompts (3 decimals). make_website_data.py copies it into the post's data.

A frame is [agent cells, moving-object cells, reward-target cells or 0, obstacle cells or 0, action, reward, done]:
cells are indices row*10+col, and 0 in place of a list means "same as the previous frame".
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / "studies" / "layout-vs-behaviour" / "experiments" / "train"
sys.path.insert(0, str(TRAIN))
from games import GAMES  # noqa: E402

FRAMES, ROWS = 120, 24
SEEDS = {"pong": 0, "breakout": 3, "pacman": 0, "freeway": 2}
KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")


def probs(w, o):
    h = np.tanh(o @ w[0] + w[1])
    h = np.tanh(h @ w[2] + w[3])
    l = h @ w[4] + w[5]
    e = np.exp(l - l.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


out = {}
for game, seed in SEEDS.items():
    z = np.load(TRAIN / f"g_{game}" / "actor.npz")
    w = [z[k].astype(np.float64) for k in KEYS]
    assert w[0].shape == (500, 32) and sum(x.size for x in w) == 17253
    rng = np.random.default_rng(seed)
    env = GAMES[game](1, rng)
    frames, last, total, episodes = [], [None, None], 0.0, 0
    for _ in range(FRAMES):
        o = env.obs().astype(np.float64)
        cells = [np.flatnonzero(o[0].reshape(100, 5)[:, ch]).tolist() for ch in range(4)]
        a = int((probs(w, o).cumsum(1) > rng.random((1, 1))).argmax(1)[0])
        r, d = env.step(np.array([a]))
        static = [0 if cells[ch] == last[ch - 2] else cells[ch] for ch in (2, 3)]
        last = [cells[2], cells[3]]
        frames.append([cells[0], cells[1], static[0], static[1], a, int(r[0]), int(d[0])])
        total += float(r[0])
        episodes += int(d[0])
        env.reset(d)
    # The prompts print a rounded zero without its sign (weights_text.py), so do the same here.
    rows = [" ".join(f"{v:.3f}".replace("-0.000", "0.000") for v in row) for row in w[0][:ROWS]]
    out[game] = {"seed": seed, "frames": frames, "rows": rows, "reward": total, "episodes_ended": episodes}
    print(game, "reward", total, "episodes ended", episodes)

sizes = [["W1", 16000], ["b1", 32], ["W2", 1024], ["b2", 32], ["W3", 160], ["b3", 5]]
assert sum(n for _, n in sizes) == 17253
(HERE / "play.json").write_text(json.dumps({"games": out, "sizes": sizes}, separators=(",", ":")) + "\n")
print("play.json", (HERE / "play.json").stat().st_size, "bytes")
