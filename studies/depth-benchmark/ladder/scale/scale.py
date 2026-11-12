"""Ladder L4-scale: train L2d-family networks of any depth/width, and write blinded prompts for them.

Usage:
  python3 -I scale.py train <states.npz> <right_toward|right_away> <seed> <out.npz> --depth D --width H [--epochs E]
  python3 -I scale.py prompts <nets dir> <out dir> --depth D --width H [--seed S]
Networks: 500 -> H (x D hidden layers, tanh) -> 5, orthogonal init (gain sqrt(2) hidden, 0.01 output), Adam 1e-3,
batch 512, as in pilot F1. Saved as W1..W{D+1}, b1..b{D+1} (in-by-out). Prompts follow make_prompts_l2d.py with the
description, formula and weight blocks written for the actual depth and width; reader sets are counterbalanced on
the output-bias sign (3 + 3 per class).
"""
import argparse
import hashlib
import json
import math
import secrets
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "l2"))
sys.path.insert(0, str(HERE.parents[2] / "pilot_f1"))

HEADER_PREFIX = "Policy network weights. Forward pass: "


def description(depth, width):
    from make_prompts import DESC, NEW, OLD
    d = DESC.read_text().replace(OLD, NEW)
    start = d.index("A policy network (multilayer perceptron)")
    end = d.index("Row i of W1 holds")
    params = 500 * width + width + (depth - 1) * (width * width + width) + width * 5 + 5
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
             10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
             20: "twenty", 24: "twenty-four"}[depth]
    lines = [f"A policy network (multilayer perceptron) with 500 inputs, {words} hidden layers of {width} units with tanh "
             "activations, and 5 outputs:"]
    lines.append(f"  h1 = tanh(x @ W1 + b1)          W1: 500 x {width}, b1: {width}")
    for k in range(2, depth + 1):
        lines.append(f"  h{k} = tanh(h{k - 1} @ W{k} + b{k})         W{k}: {width} x {width},  b{k}: {width}")
    lines.append(f"  logits = h{depth} @ W{depth + 1} + b{depth + 1}           W{depth + 1}: {width} x 5,   b{depth + 1}: 5")
    tail = d[end:].replace("the 32 hidden units", f"the {width} hidden units").replace("17,253 parameters", f"{params:,} parameters")
    return d[:start] + "\n".join(lines) + "\n" + tail


def weights_text(w, depth, width):
    fmt = lambda x: "0.000" if f"{x:.3f}" == "-0.000" else f"{x:.3f}"  # noqa: E731
    formula = "; ".join([f"h1 = tanh(x @ W1 + b1)"] + [f"h{k} = tanh(h{k - 1} @ W{k} + b{k})" for k in range(2, depth + 1)]
                        + [f"logits = h{depth} @ W{depth + 1} + b{depth + 1}"])
    out = [HEADER_PREFIX + formula + ". Values rounded to 3 decimals.", ""]
    blocks = [("W1", f"W1 (layer 1 weights), shape 500 x {width}: one line per row (row i = input i), {width} values per line"),
              ("b1", f"b1 (layer 1 biases), shape {width}:")]
    for k in range(2, depth + 1):
        blocks += [(f"W{k}", f"W{k} (layer {k} weights), shape {width} x {width}: one line per row"),
                   (f"b{k}", f"b{k} (layer {k} biases), shape {width}:")]
    blocks += [(f"W{depth + 1}", f"W{depth + 1} (output layer weights), shape {width} x 5: one line per row"),
               (f"b{depth + 1}", f"b{depth + 1} (output layer biases), shape 5:")]
    for i, (name, head) in enumerate(blocks):
        a = np.asarray(w[name], dtype=np.float64)
        out.append(head)
        out += [" ".join(fmt(v) for v in row) for row in (a if a.ndim == 2 else a[None, :])]
        if i < len(blocks) - 1:
            out.append("")
    return "\n".join(out) + "\n"


def train(states, rule, seed, out, depth, width, epochs):
    import torch
    import torch.nn as nn
    from train_l2 import labels
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    z = np.load(states)
    x = torch.from_numpy(z["obs"].astype(np.float32))
    y = torch.from_numpy(labels(z, rule).astype(np.int64))
    tr, te = z["split"] == 0, z["split"] == 1
    layers, dims = [], [500] + [width] * depth
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.Tanh()]
    layers.append(nn.Linear(width, 5))
    net = nn.Sequential(*layers)
    lin = [m for m in net if isinstance(m, nn.Linear)]
    for k, m in enumerate(lin):
        nn.init.orthogonal_(m.weight, 0.01 if k == len(lin) - 1 else math.sqrt(2))
        nn.init.zeros_(m.bias)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    xtr, ytr = x[tr], y[tr]
    for _ in range(epochs):
        perm = torch.randperm(len(xtr), generator=gen)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            loss = nn.functional.cross_entropy(net(xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        acc = float((net(x[te]).argmax(1) == y[te]).float().mean())
    W = {}
    for k, m in enumerate(lin, start=1):
        W[f"W{k}"] = m.weight.detach().numpy().T.astype(np.float32)
        W[f"b{k}"] = m.bias.detach().numpy().astype(np.float32)
    np.savez(out, **W)
    Path(out).with_suffix(".json").write_text(json.dumps({"rule": rule, "seed": seed, "depth": depth, "width": width,
                                                          "epochs": epochs, "heldout_acc": round(acc, 4)}))
    print(f"{rule} s{seed} d{depth} w{width} acc {acc:.4f}", flush=True)


def prompts(nets_dir, out, depth, width, seed, exclude=None):
    from make_prompts_l2d import GAME, QUESTION
    out = Path(out)
    rng = np.random.default_rng(seed)
    last = f"b{depth + 1}"
    used = set()
    for e in (exclude.split(",") if exclude else []):
        used |= set(json.loads(Path(e).read_text()).values())
    pick = []
    for cls in ("right_toward", "right_away"):
        pos, neg = [], []
        for f in sorted(Path(nets_dir).glob(f"{cls}_s*.npz"), key=lambda p: int(p.stem.rsplit("_s", 1)[1])):
            if f.stem in used or json.loads(f.with_suffix(".json").read_text())["heldout_acc"] < 0.95:
                continue
            w = np.load(f)
            (pos if w[last][4] > w[last][3] else neg).append(f.stem)
        assert len(pos) >= 3 and len(neg) >= 3, (cls, len(pos), len(neg))
        pick += list(rng.choice(pos, 3, replace=False)) + list(rng.choice(neg, 3, replace=False))
    desc = description(depth, width)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    key = {}
    for name in rng.permutation(pick):
        w = dict(np.load(Path(nets_dir) / f"{name}.npz"))
        bid = secrets.token_hex(4)
        key[bid] = str(name)
        (out / "prompts" / f"{bid}.txt").write_text(desc.rstrip("\n") + "\n\n" + GAME + "\n\n" + QUESTION +
                                                    "\n\nTHE NETWORK'S WEIGHTS\n" + weights_text(w, depth, width))
    kj = json.dumps(key, indent=1, sort_keys=True)
    (out / "key.json").write_text(kj)
    (out / "PREREG_KEY_HASH.txt").write_text(hashlib.sha256(kj.encode()).hexdigest() + "  key.json\n")
    print("prompts", len(key), "hash", hashlib.sha256(kj.encode()).hexdigest()[:16])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "prompts"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--exclude", help="key.json of an earlier reader set whose networks must not be reused")
    a = ap.parse_args()
    if a.mode == "train":
        states, rule, seed, out = a.args
        train(states, rule, int(seed), out, a.depth, a.width, a.epochs)
    else:
        prompts(a.args[0], a.args[1], a.depth, a.width, a.seed, a.exclude)
