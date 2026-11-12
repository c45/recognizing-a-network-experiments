"""Stage 6: disguised splits (tanh) and behaviour-exact depth inflation (ReLU), as blinded prompt suites.

Usage: python3 -I build_stage6.py <out dir>
Base networks start from synthetic-row camouflage (stage 5's `synth`: unused first-layer rows replaced by new rows fitted
per channel to the used rows, then hidden units shuffled and output rows offset), so there are no duplicate rows.
  split    tanh 32-32 policies -> 64-64. Layer 1: every unit j gets a twin whose input column equals unit j's on every
           input the game uses and is random on inputs it never uses, so on visited states the twin's activation equals
           unit j's. Each layer-2 weight from unit j is divided between j and its twin with random proportions.
           Layer 2: every unit k is split in two whose input columns are different random redistributions over the
           (j, twin) pairs (so equal on visited states) and whose output rows are r_k + z and -z for a random large z.
           No column or row is an exact copy or negation of another; the output layer's button statistics are scrambled.
  deep0    ReLU 24-24-24 policies, synth only (3 hidden layers).
  deep3    deep0 with one extra ReLU layer (width 32) after each hidden layer: 6 hidden layers.
  deep6    deep0 with two extra ReLU layers after each hidden layer: 9 hidden layers.
           Each inserted layer is non-negative (a scaled permutation plus dense uniform noise of up to 0.10, chosen to
           keep 3-decimal printing error near the base network's) with small positive biases, so on non-negative inputs ReLU is the
           identity; the next real layer absorbs the composite map through its pseudo-inverse. Exact up to float.
Every network is checked on visited states (a fresh rollout of the base policy plus its training state sample): the
largest action-probability change, exact and as printed to 3 decimals.
Writes prompts/<id>.txt, nets/<id>/{actor.npz,actor.txt,head.txt}, key.json, order.txt, checks.json.
"""
import json
import secrets
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE))
from ff_paths import LAYOUT  # noqa: E402
sys.path.insert(0, str(LAYOUT))
from build_prompts import head, TEMPLATE  # noqa: E402
from build_stage5 import fill, NET_POST  # noqa: E402
from games_new import ALL  # noqa: E402
from weights_text import fmt  # noqa: E402

GAMES = ["pong", "breakout", "pacman", "freeway"]
TRAIN = LAYOUT / "train"


def nlayers(w):
    return len([k for k in w if k.startswith("W")])


def forward(w, X, act):
    f = np.tanh if act == "tanh" else (lambda z: np.maximum(z, 0.0))
    h, n = X, nlayers(w)
    for k in range(1, n):
        h = f(h @ w[f"W{k}"] + w[f"b{k}"])
    return h @ w[f"W{n}"] + w[f"b{n}"]


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def visited(game, w, act, n_env=64, steps=4000, seed=0):
    rng = np.random.default_rng(seed)
    env = ALL[game](n_env, rng)
    S = []
    for _ in range(steps):
        o = env.obs()
        S.append(o.copy())
        p = softmax(forward(w, o.astype(np.float64), act))
        a = (p.cumsum(1) > rng.random((n_env, 1))).argmax(1)
        r, d = env.step(a)
        env.reset(d)
    return np.unique(np.concatenate(S).astype(np.uint8), axis=0).astype(np.float64)


def shuffle(w, rng, act):
    w, n = dict(w), nlayers(w)
    for k in range(1, n):
        m = w[f"W{k}"].shape[1]
        pi = rng.permutation(m)
        s = rng.choice([-1.0, 1.0], m) if act == "tanh" else np.exp(rng.uniform(-0.5, 0.5, m))
        w[f"W{k}"], w[f"b{k}"] = w[f"W{k}"][:, pi] * s, w[f"b{k}"][pi] * s
        w[f"W{k + 1}"] = w[f"W{k + 1}"][pi, :] / (s[:, None] if act == "relu" else 1) * (s[:, None] if act == "tanh" else 1)
    return w


def offset(w, rng):
    n = nlayers(w)
    W = w[f"W{n}"]
    u = rng.standard_normal(W.shape[0]) * 3 * float(np.linalg.norm(W, axis=0).max())
    return dict(w, **{f"W{n}": W + u[:, None], f"b{n}": w[f"b{n}"] + rng.normal() * 3})


def split(w, unused, rng):
    W1, b1, W2, b2, W3, b3 = (w[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
    sd = W1[unused].std()
    T = W1.copy()
    T[unused] = rng.normal(0, sd, (int(unused.sum()), W1.shape[1]))
    W1n, b1n = np.hstack([W1, T]), np.concatenate([b1, b1])
    a = rng.uniform(-1.0, 2.0, W2.shape)
    W2n = np.vstack([a * W2, (1 - a) * W2])
    b = rng.uniform(-1.0, 2.0, W2.shape)
    W2b = np.vstack([b * W2, (1 - b) * W2])
    W2s = np.hstack([W2n, W2b])
    b2s = np.concatenate([b2, b2])
    z = rng.standard_normal(W3.shape) * 2.0 * np.abs(W3 - W3.mean(1, keepdims=True)).max()
    W3s = np.vstack([W3 + z, -z])
    out = {"W1": W1n, "b1": b1n, "W2": W2s, "b2": b2s, "W3": W3s, "b3": b3}
    return shuffle(out, rng, "tanh")


def inflate(w, per_gap, rng, width=32):
    if per_gap == 0:
        return dict(w)
    n = nlayers(w)
    Ws = [w[f"W{k}"] for k in range(1, n + 1)]
    bs = [w[f"b{k}"] for k in range(1, n + 1)]
    newW, newb = [Ws[0]], [bs[0]]
    for g in range(1, n):
        d_in = newW[-1].shape[1]
        P, q = np.eye(d_in), np.zeros(d_in)
        for t in range(per_gap):
            m_in = P.shape[1]
            A = np.zeros((m_in, width))
            idx = rng.permutation(width)[:m_in] if m_in <= width else rng.permutation(m_in)[:width]
            for i in range(m_in):
                A[i, idx[i % len(idx)]] = rng.uniform(0.6, 1.4)
            M = A + 0.10 * rng.uniform(0, 1, (m_in, width))
            c = rng.uniform(0, 0.1, width)
            newW.append(M)
            newb.append(c)
            P, q = P @ M, q @ M + c
        Pp = np.linalg.pinv(P)
        Wn = Pp @ Ws[g]
        newW.append(Wn)
        newb.append(bs[g] - q @ Wn)
    out = {}
    for k, (Wk, bk) in enumerate(zip(newW, newb), start=1):
        out[f"W{k}"], out[f"b{k}"] = Wk, bk
    return out


def weights_text(w, act):
    n = nlayers(w)
    parts = [f"h1 = {act}(x @ W1 + b1)"] + [f"h{k} = {act}(h{k - 1} @ W{k} + b{k})" for k in range(2, n)]
    parts.append(f"logits = h{n - 1} @ W{n} + b{n}")
    out = ["Policy network weights. Forward pass: " + "; ".join(parts) + ". Values rounded to 3 decimals.", ""]
    for k in range(1, n + 1):
        Wk, bk = w[f"W{k}"], w[f"b{k}"]
        kind = "output layer" if k == n else f"layer {k}"
        if k == 1:
            out.append(f"W1 (layer 1 weights), shape 500 x {Wk.shape[1]}: one line per row (row i = input i), "
                       f"{Wk.shape[1]} values per line")
        else:
            out.append(f"W{k} ({kind} weights), shape {Wk.shape[0]} x {Wk.shape[1]}: one line per row")
        out += [" ".join(fmt(v) for v in row) for row in Wk]
        out += ["", f"b{k} ({kind} biases), shape {bk.shape[0]}:", " ".join(fmt(v) for v in bk)]
        if k < n:
            out.append("")
    return "\n".join(out) + "\n"


def network_section(w, act):
    n = nlayers(w)
    sizes = [500] + [w[f"W{k}"].shape[1] for k in range(1, n + 1)]
    hidden = sizes[1:-1]
    actname = "tanh activations" if act == "tanh" else "ReLU activations (relu(z) = max(z, 0))"
    widths = (f"{len(hidden)} hidden layers of {hidden[0]} units" if len(set(hidden)) == 1 else
              f"{len(hidden)} hidden layers (widths " + ", ".join(str(h) for h in hidden) + ")")
    lines = [f"A policy network (multilayer perceptron) with 500 inputs, {widths} with {actname}, and 5 outputs:"]
    for k in range(1, n + 1):
        src = "x" if k == 1 else f"h{k - 1}"
        lhs = f"logits = {src} @ W{k} + b{k}" if k == n else f"h{k} = {act}({src} @ W{k} + b{k})"
        shp = f"W{k}: {sizes[k - 1]} x {sizes[k]},"
        lines.append(f"  {lhs:<34}{shp:<15}b{k}: {sizes[k]}")
    npar = sum(sizes[k] * sizes[k + 1] + sizes[k + 1] for k in range(n))
    lines.append(f"Row i of W1 holds the weights from input i to the {sizes[1]} units of the first hidden layer; column j "
                 f"of a weight matrix belongs to unit j of the next layer. {npar:,} parameters in total.")
    return "\n".join(lines)


def main(out):
    out = Path(out)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int.from_bytes(secrets.token_bytes(4), "little"))
    h0 = head(TEMPLATE)
    assert NET_POST in h0
    key, checks = {}, {}
    for game in GAMES:
        for act, src in (("tanh", f"g_{game}"), ("relu", f"g_{game}_relu24x3")):
            w0 = {k: v.astype(np.float64) for k, v in dict(np.load(TRAIN / src / "actor.npz")).items()}
            unused = np.load(TRAIN / src / "occupancy_never_on.npy")
            X = visited(game, w0, act)
            S = np.unique(np.load(TRAIN / src / "states.npy").astype(np.float64), axis=0)
            X = np.unique(np.vstack([X, S]), axis=0)
            assert X[:, unused].max() == 0, f"{src}: an 'unused' input is on in a visited state"
            p0 = softmax(forward(w0, X, act))
            p0r = softmax(forward({k: np.round(v, 3) for k, v in w0.items()}, X, act))
            base = offset(shuffle(fill(w0, unused, game, rng, True), rng, act), rng)
            nets = {"split": offset(split(base, unused, rng), rng)} if act == "tanh" else {
                f"deep{3 * g}": offset(shuffle(inflate(base, g, rng), rng, act), rng) if g else base for g in (0, 1, 2)}
            for cond, w in nets.items():
                p = softmax(forward(w, X, act))
                pr = softmax(forward({k: np.round(v, 3) for k, v in w.items()}, X, act))
                checks[f"{game}/{cond}"] = {"visited_states": int(len(X)), "hidden_layers": nlayers(w) - 1,
                                            "max_dp_exact": float(np.abs(p - p0).max()),
                                            "max_dp_printed_vs_printed_base": float(np.abs(pr - p0r).max())}
                pid = secrets.token_hex(4)
                nd = out / "nets" / pid
                nd.mkdir(parents=True)
                h = h0.replace(NET_POST, network_section(w, act))
                text = weights_text(w, act)
                np.savez(nd / "actor.npz", **{k: np.round(v, 3) for k, v in w.items()})
                (nd / "actor.txt").write_text(text)
                (nd / "head.txt").write_text(h)
                (out / "prompts" / f"{pid}.txt").write_text(h + text)
                key[pid] = {"game": game, "condition": cond, "arch": act}
                print(game, cond, {k: round(v, 6) if isinstance(v, float) else v for k, v in checks[f"{game}/{cond}"].items()},
                      flush=True)
    order = list(key)
    rng.shuffle(order)
    (out / "key.json").write_text(json.dumps(key, indent=1))
    (out / "order.txt").write_text("\n".join(order) + "\n")
    (out / "checks.json").write_text(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
