"""Blind the counterbalanced L2d reader set and write one prompt per network (ladder PREREG.md, L2d).

Usage: python3 -I make_prompts_l2d.py <nets dir> <out dir> [--width 32] [--nosim] [--seed 20261107]
--width: hidden width of the networks; the description, the weight-block headers and the parameter count are
rescaled. --nosim: adds the post's Exp 2 instruction against forward-pass calculation, adapted (ladder PREREG.md,
L2d-nosim).
"""
import argparse
import hashlib
import json
import secrets
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "pilot_f1"))
from make_prompts import DESC, NEW, OLD, to_text  # noqa: E402

NOSIM = ("Do not do any forward-pass calculation: do not run the network on any input (in your head or on paper), "
         "including partial or approximate sums for a specific ball and paddle position, and do not simulate play. "
         "Read what the weights themselves show. ")


def rescale(text, width):
    if width == 32:
        return text
    params = 500 * width + width + width * width + width + width * 5 + 5
    for a, b in [("two hidden layers of 32 units", f"two hidden layers of {width} units"),
                 ("W1: 500 x 32, b1: 32", f"W1: 500 x {width}, b1: {width}"),
                 ("W2: 32 x 32,  b2: 32", f"W2: {width} x {width},  b2: {width}"),
                 ("W3: 32 x 5,   b3: 5", f"W3: {width} x 5,   b3: 5"),
                 ("the 32 hidden units", f"the {width} hidden units"),
                 ("17,253 parameters", f"{params:,} parameters"),
                 ("shape 500 x 32: one line per row (row i = input i), 32 values per line",
                  f"shape 500 x {width}: one line per row (row i = input i), {width} values per line"),
                 ("b1 (layer 1 biases), shape 32:", f"b1 (layer 1 biases), shape {width}:"),
                 ("W2 (layer 2 weights), shape 32 x 32:", f"W2 (layer 2 weights), shape {width} x {width}:"),
                 ("b2 (layer 2 biases), shape 32:", f"b2 (layer 2 biases), shape {width}:"),
                 ("W3 (output layer weights), shape 32 x 5:", f"W3 (output layer weights), shape {width} x 5:")]:
        assert a in text, a
        text = text.replace(a, b)
    return text

GAME = """THE GAME
A Breakout-like game on this grid. The agent is a paddle two cells wide on row 8 (channel 0); it moves one cell left (action 3) or right (action 4) per step, and actions 1 (up) and 2 (down) do nothing. A ball (channel 1) moves one cell diagonally per step and bounces off the walls, the bricks and the paddle. Bricks (channel 2, reward targets) fill rows 1-3. Channel 3 is the wall border; channel 4 is the previous frame of channels 0 and 1, so comparing channel 1 with channel 4 shows which way the ball is moving."""
QUESTION = """QUESTION. The weights below belong to one policy network trained on this game. The rule it imitates depends on which way the ball is moving horizontally: when the ball moves in one direction the paddle moves TOWARD the ball's column, and when it moves in the other direction the paddle moves AWAY from it (it does nothing when the ball is directly above the paddle). When the ball is moving RIGHT, does this network move the paddle toward the ball or away from it? Work from the weights. You have no tools: no code execution, no calculator, nothing to run. This is a single response.

Answer in this order:
Observations: <what you see in the weights that bears on the question; quote the weights you rely on>
Reasoning: <how those observations decide the question>
Answer: <right-toward | right-away>
Confidence: <50-100>%"""


def main(nets_dir, out, width=32, nosim=False, seed=20261107, same_as=None):
    out = Path(out)
    rng = np.random.default_rng(seed)
    question = QUESTION.replace("Work from the weights. ", "Work from the weights. " + NOSIM) if nosim else QUESTION
    pick = []
    for cls in ("right_toward", "right_away"):
        pos, neg = [], []
        for s in range(20):
            w = np.load(Path(nets_dir) / f"{cls}_s{s}.npz")
            (pos if w["b3"][4] > w["b3"][3] else neg).append(f"{cls}_s{s}")
        assert len(pos) >= 3 and len(neg) >= 3, (cls, len(pos), len(neg))
        pick += list(rng.choice(pos, 3, replace=False)) + list(rng.choice(neg, 3, replace=False))
    if same_as:
        pick = sorted(json.loads(Path(same_as).read_text()).values())
    order = list(rng.permutation(pick))
    desc = DESC.read_text().replace(OLD, NEW)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    key = {}
    for name in order:
        w = dict(np.load(Path(nets_dir) / f"{name}.npz"))
        bid = secrets.token_hex(4)
        key[bid] = str(name)
        (out / "prompts" / f"{bid}.txt").write_text(rescale(
            desc.rstrip("\n") + "\n\n" + GAME + "\n\n" + question + "\n\nTHE NETWORK'S WEIGHTS\n" + to_text(w), width))
    kj = json.dumps(key, indent=1, sort_keys=True)
    (out / "key.json").write_text(kj)
    (out / "PREREG_KEY_HASH.txt").write_text(hashlib.sha256(kj.encode()).hexdigest() + "  key.json\n")
    print("prompts", len(key), "hash", hashlib.sha256(kj.encode()).hexdigest()[:16])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("nets")
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--nosim", action="store_true")
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--same-as", help="key.json of an earlier set: reuse exactly its networks (new IDs and order)")
    a = ap.parse_args()
    main(a.nets, a.out, a.width, a.nosim, a.seed, a.same_as)
