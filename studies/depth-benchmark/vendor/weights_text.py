"""Write an actor network in the Exp 1 prompt's weight-text format, and build a full Exp 1 prompt around it.

Usage (check): python3 -I weights_text.py <exp1_which_game dir>
Format copied from networks/<id>_actor.txt: a header line, a blank line, then for each of W1, b1, W2, b2, W3, b3 a
"<name> (...), shape ...:" line followed by its values to 3 decimals; blocks separated by blank lines.
"""
import sys
from pathlib import Path

import numpy as np

HEADER = ("Policy network weights. Forward pass: h1 = tanh(x @ W1 + b1); h2 = tanh(h1 @ W2 + b2); "
          "logits = h2 @ W3 + b3. Values rounded to 3 decimals.")
BLOCKS = [
    ("W1", "W1 (layer 1 weights), shape 500 x 32: one line per row (row i = input i), 32 values per line"),
    ("b1", "b1 (layer 1 biases), shape 32:"),
    ("W2", "W2 (layer 2 weights), shape 32 x 32: one line per row"),
    ("b2", "b2 (layer 2 biases), shape 32:"),
    ("W3", "W3 (output layer weights), shape 32 x 5: one line per row"),
    ("b3", "b3 (output layer biases), shape 5:"),
]


def fmt(x):
    s = f"{x:.3f}"
    return "0.000" if s == "-0.000" else s


def to_text(w):
    """w: dict with W1 (500x32), b1, W2, b2, W3 (32x5), b3 (in-by-out kernels, as in the released npz)."""
    out = [HEADER, ""]
    for k, (name, head) in enumerate(BLOCKS):
        a = np.asarray(w[name], dtype=np.float64)
        out.append(head)
        rows = a if a.ndim == 2 else a[None, :]
        out += [" ".join(fmt(v) for v in row) for row in rows]
        if k < len(BLOCKS) - 1:
            out.append("")
    return "\n".join(out) + "\n"


def released_actor(npz_path):
    z = np.load(npz_path)
    return {"W1": z["actor/dense1/kernel"], "b1": z["actor/dense1/bias"], "W2": z["actor/dense2/kernel"],
            "b2": z["actor/dense2/bias"], "W3": z["actor/logits/kernel"], "b3": z["actor/logits/bias"]}


def build_prompt(template_prompt, weights_text):
    """The released prompt up to and including the 'THE NETWORK'S WEIGHTS' line, then the new weight text."""
    t = Path(template_prompt).read_text()
    k = t.index("THE NETWORK'S WEIGHTS\n") + len("THE NETWORK'S WEIGHTS\n")
    return t[:k] + weights_text


if __name__ == "__main__":
    exp = Path(sys.argv[1])
    for nid, game in [("dfc6cb65", "pong"), ("a9f535cc", "breakout")]:
        mine = to_text(released_actor(exp / "networks" / f"{nid}.npz"))
        theirs = (exp / "networks" / f"{nid}_actor.txt").read_text()
        ml, tl = mine.splitlines(), theirs.splitlines()
        diff = [k for k, (a, b) in enumerate(zip(ml, tl)) if a != b]
        print(nid, "lines", len(ml), len(tl), "differing lines", len(diff), diff[:5])
        prompt = build_prompt(exp / "runs" / game / "prompt.txt", theirs)
        print("  rebuilt prompt identical:", prompt == (exp / "runs" / game / "prompt.txt").read_text())
