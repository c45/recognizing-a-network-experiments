"""Files for the post's first figure, which runs the four 32-32 tanh game policies in the browser.

Usage: uv run --with numpy python -I make_live_play.py <post folder>
Reads ../studies/layout-vs-behaviour/experiments/train/g_<game>/actor.npz and train/games.py, and writes into the post folder:
  policies/<game>.bin         the 17,253 weights as little-endian float32, in the order W1 (500 x 32, row by row),
                              b1, W2 (32 x 32), b2, W3 (32 x 5), b3
  figures/games.fixture.json  a trace of games.py and of the forward pass, which figures/games.test.ts replays
                              against the TypeScript port
  data.json                   only its "play" key is replaced: the layer sizes and, per game, the first ROWS rows of
                              the first-layer weights as printed in the prompts (3 decimals)

The trace, per game: STEPS steps from a fixed seed, the first half with moves sampled from the policy and the second
half with uniformly random moves, so losses and restarts are covered. "draws" is every random number the game drew,
in order (a choice from [-1, 1] is logged as its index); "steps" is [observation, move, reward, ended] with the 500
inputs packed into bits (first input in the top bit) as base64; "probs" is the policy's five move probabilities at
every PROBS_EVERY-th step; "hidden" is both hidden layers at every HIDDEN_EVERY-th step.
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / "studies" / "layout-vs-behaviour" / "experiments" / "train"
sys.path.insert(0, str(TRAIN))
from games import GAMES  # noqa: E402

STEPS, PROBS_EVERY, HIDDEN_EVERY, ROWS = 240, 4, 40, 24
SEEDS = {"pong": 0, "breakout": 3, "pacman": 0, "freeway": 2}
KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")
SIZES = [["W1", 16000], ["b1", 32], ["W2", 1024], ["b2", 32], ["W3", 160], ["b3", 5]]


class Recorder:
    """A numpy generator that logs every number it hands out."""

    def __init__(self, rng):
        self.rng, self.log = rng, []

    def integers(self, low, high=None, size=None):
        v = self.rng.integers(low, high, size)
        self.log += np.atleast_1d(v).ravel().tolist()
        return v

    def random(self, size=None):
        v = self.rng.random(size)
        self.log += np.atleast_1d(v).ravel().tolist()
        return v

    def choice(self, a, size=None, replace=True):
        v = self.rng.choice(a, size, replace=replace)
        flat = np.atleast_1d(v).ravel().tolist()
        self.log += [a.index(x) for x in flat] if isinstance(a, list) else flat
        return v


def forward(w, o):
    h1 = np.tanh(o @ w[0] + w[1])
    h2 = np.tanh(h1 @ w[2] + w[3])
    l = h2 @ w[4] + w[5]
    e = np.exp(l - l.max(1, keepdims=True))
    return h1, h2, e / e.sum(1, keepdims=True)


def main():
    post = Path(sys.argv[1])
    (post / "policies").mkdir(exist_ok=True)
    rows, trace = {}, {}
    for game, seed in SEEDS.items():
        z = np.load(TRAIN / f"g_{game}" / "actor.npz")
        assert all(z[k].dtype == np.float32 for k in KEYS)
        flat = np.concatenate([z[k].ravel() for k in KEYS]).astype("<f4")
        assert z["W1"].shape == (500, 32) and flat.size == 17253 == sum(n for _, n in SIZES)
        (post / "policies" / f"{game}.bin").write_bytes(flat.tobytes())
        w = [z[k].astype(np.float64) for k in KEYS]

        rec = Recorder(np.random.default_rng(seed))
        moves = np.random.default_rng(seed + 1000)
        env = GAMES[game](1, rec)
        steps, probs, hidden, total, ended = [], [], [], 0.0, 0
        for t in range(STEPS):
            o = env.obs().astype(np.float64)
            h1, h2, p = forward(w, o)
            if t < STEPS // 2:
                a = int((p.cumsum(1) > moves.random((1, 1))).argmax(1)[0])
            else:
                a = int(moves.integers(5))
            r, d = env.step(np.array([a]))
            packed = np.packbits(o[0].astype(np.uint8)).tobytes()
            steps.append([base64.b64encode(packed).decode(), a, int(r[0]), int(d[0])])
            if t % PROBS_EVERY == 0:
                probs.append([t, [round(float(x), 12) for x in p[0]]])
            if t % HIDDEN_EVERY == 0:
                hidden.append([t, [round(float(x), 12) for x in h1[0]], [round(float(x), 12) for x in h2[0]]])
            total += float(r[0])
            ended += int(d[0])
            env.reset(d)
        trace[game] = {"seed": seed, "draws": rec.log, "steps": steps, "probs": probs, "hidden": hidden}
        # The prompts print a rounded zero without its sign (weights_text.py), so do the same here.
        rows[game] = {"rows": [" ".join(f"{v:.3f}".replace("-0.000", "0.000") for v in row) for row in w[0][:ROWS]]}
        print(game, "reward", total, "games ended", ended, "draws", len(rec.log))

    fixture = post / "figures" / "games.fixture.json"
    fixture.write_text(json.dumps(trace, separators=(",", ":")) + "\n")
    data = json.loads((post / "data.json").read_text())
    data["play"] = {"sizes": SIZES, "games": rows}
    (post / "data.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print("fixture", fixture.stat().st_size, "bytes | data.json", (post / "data.json").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
