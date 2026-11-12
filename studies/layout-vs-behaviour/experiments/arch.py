"""Architectures of the policy networks: forward pass, weight-text format and the prompt's network description.

tanh32x2 is the post's 500-32-32-5 tanh MLP; relu24x3 (500-24-24-24-5, ReLU) and tanh48x1 (500-48-5, tanh) are the
replication architectures. Weights are dicts W1, b1, ..., Wk, bk of in-by-out kernels.
"""
import re
from pathlib import Path

import numpy as np

ARCHS = {"tanh32x2": ("tanh", [32, 32]), "tanh64x2": ("tanh", [64, 64]), "relu24x3": ("relu", [24, 24, 24]), "tanh48x1": ("tanh", [48])}
NUM = {1: "one", 2: "two", 3: "three"}


def keys(arch):
    n = len(ARCHS[arch][1]) + 1
    return [f"{p}{k}" for k in range(1, n + 1) for p in ("W", "b")]


def sizes(arch):
    return [500] + ARCHS[arch][1] + [5]


def act(arch):
    return np.tanh if ARCHS[arch][0] == "tanh" else (lambda z: np.maximum(z, 0.0))


def logits(w, X, arch):
    f, n = act(arch), len(ARCHS[arch][1]) + 1
    h = X
    for k in range(1, n):
        h = f(h @ w[f"W{k}"] + w[f"b{k}"])
    return h @ w[f"W{n}"] + w[f"b{n}"]


def header(arch):
    a, n = ARCHS[arch][0], len(ARCHS[arch][1]) + 1
    parts = [f"h1 = {a}(x @ W1 + b1)"] + [f"h{k} = {a}(h{k - 1} @ W{k} + b{k})" for k in range(2, n)]
    parts.append(f"logits = h{n - 1} @ W{n} + b{n}")
    return "Policy network weights. Forward pass: " + "; ".join(parts) + ". Values rounded to 3 decimals."


def blocks(arch):
    s, n = sizes(arch), len(ARCHS[arch][1]) + 1
    out = []
    for k in range(1, n + 1):
        kind = "output layer" if k == n else f"layer {k}"
        if k == 1:
            out.append((f"W{k}", f"W{k} ({kind} weights), shape 500 x {s[1]}: one line per row (row i = input i), "
                        f"{s[1]} values per line"))
        else:
            out.append((f"W{k}", f"W{k} ({kind} weights), shape {s[k - 1]} x {s[k]}: one line per row"))
        out.append((f"b{k}", f"b{k} ({kind} biases), shape {s[k]}:"))
    return out


def fmt(x):
    s = f"{x:.3f}"
    return "0.000" if s == "-0.000" else s


def to_text(w, arch):
    out, bl = [header(arch), ""], blocks(arch)
    for k, (name, head) in enumerate(bl):
        a = np.asarray(w[name], dtype=np.float64)
        out.append(head)
        rows = a if a.ndim == 2 else a[None, :]
        out += [" ".join(fmt(v) for v in row) for row in rows]
        if k < len(bl) - 1:
            out.append("")
    return "\n".join(out) + "\n"


def n_params(arch):
    s = sizes(arch)
    return sum(s[k] * s[k + 1] + s[k + 1] for k in range(len(s) - 1))


def network_section(arch):
    a, hidden = ARCHS[arch]
    n, s = len(hidden) + 1, sizes(arch)
    width = hidden[0]
    actname = "tanh activations" if a == "tanh" else "ReLU activations (relu(z) = max(z, 0))"
    layer_word = "layer" if len(hidden) == 1 else "layers"
    lines = [f"A policy network (multilayer perceptron) with 500 inputs, {NUM[len(hidden)]} hidden {layer_word} of "
             f"{width} units with {actname}, and 5 outputs:"]
    for k in range(1, n + 1):
        src = "x" if k == 1 else f"h{k - 1}"
        lhs = f"logits = {src} @ W{k} + b{k}" if k == n else f"h{k} = {a}({src} @ W{k} + b{k})"
        shp = f"W{k}: {s[k - 1]} x {s[k]},"
        lines.append(f"  {lhs:<32}{shp:<14}b{k}: {s[k]}")
    lines.append(f"Row i of W1 holds the weights from input i to the {width} hidden units; column j of a weight matrix "
                 f"belongs to unit j of the next layer. {n_params(arch):,} parameters in total. It was trained with "
                 f"reinforcement learning (PPO) to maximise the reward of a game played on this grid.")
    return "\n".join(lines)


SECTION = re.compile(r"A policy network \(multilayer perceptron\).*?maximise the reward of a game played on this grid\.",
                     re.S)


def describe(text, arch):
    """The post's description/prompt text with its network section rewritten for arch (unchanged for tanh32x2)."""
    assert SECTION.search(text)
    return SECTION.sub(lambda m: network_section(arch), text, count=1)


def build_prompt(template_prompt, weights_text, arch):
    t = describe(Path(template_prompt).read_text(), arch)
    k = t.index("THE NETWORK'S WEIGHTS\n") + len("THE NETWORK'S WEIGHTS\n")
    return t[:k] + weights_text


def parse_text(path):
    blocks_, cur = {}, None
    for ln in Path(path).read_text().splitlines():
        head = ln.split(" ", 1)[0]
        if re.fullmatch(r"[Wb]\d", head) and "shape" in ln:
            cur = head
            blocks_[cur] = []
            continue
        if cur and ln.strip():
            try:
                blocks_[cur].append([float(x) for x in ln.split()])
            except ValueError:
                pass
    w = {k: np.array(v) for k, v in blocks_.items()}
    for k in w:
        if k.startswith("b"):
            w[k] = w[k].ravel()
    return w


def arch_of(w):
    n = len([k for k in w if k.startswith("W")])
    width = w["W1"].shape[1]
    for name, (_, hidden) in ARCHS.items():
        if len(hidden) + 1 == n and hidden[0] == width:
            return name
    raise ValueError("unknown architecture")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from weights_text import HEADER, BLOCKS
    assert header("tanh32x2") == HEADER and blocks("tanh32x2") == BLOCKS
    from paths import PROMPT_TEMPLATE as tpl
    t = tpl.read_text()
    assert describe(t, "tanh32x2") == t, "tanh32x2 section must reproduce the post's text"
    for a in ARCHS:
        print(network_section(a), "\n", header(a), "\n")
