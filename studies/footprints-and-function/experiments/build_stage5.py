"""Stage 5: three behaviour-preserving obfuscations of our trained policies, plus controls, as blinded prompt suites.

Usage: python3 -I build_stage5.py <out dir>
Sources: obscured/train/g_<game>/actor.npz (tanh 32-32, the camouflage sources) for pong, breakout, pacman, freeway.
"Unused" inputs = never on during the whole training run (occupancy_never_on.npy); checked against a fresh rollout of
the final policy (64 envs x 4,000 steps): none of them is ever on.
Conditions (one network per game each):
  plain     the trained network
  camo_v2   unused rows overwritten with exact copies of used rows (mirror cell, else a random used row of the same
            channel), hidden units shuffled and sign-flipped, a common offset added to each output-layer row
  synth     as camo_v2, but unused rows are new rows: per channel, drawn from a Gaussian fitted to the used rows,
            rescaled to the norm of a random used row of that channel (no row is a copy)
  pairs     camo_v2 plus 8 cancelling pairs of layer-2 units (width 32 -> 48): unit a has input column c and bias b,
            unit b has -c and -b (tanh is odd, so h_b = -h_a), and both send the same random output row up to a
            constant across actions, so their sum adds nothing but a softmax-invariant constant. The added rows
            scramble the output layer's button similarity.
  basis     the plain network, with its first layer re-expressed for a re-encoded input: the 500 inputs are split
            into 50 random groups of 10 and each group is rotated by a random 10x10 orthogonal matrix Q_g
            (y_g = x_g Q_g, W1_y[g] = Q_g^-1 W1[g], with Q_g as printed to 3 decimals). The prompt states the encoding and lists every group and Q_g.
Each condition is checked on the visited states: the largest action-probability change, exact and as printed
(3 decimals). Writes prompts/<id>.txt, nets/<id>/{actor.npz,actor.txt,head.txt,extra.npz}, key.json, order.txt,
checks.json.
"""
import json
import secrets
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ff_paths import LAYOUT  # noqa: E402
sys.path.insert(0, str(LAYOUT))
from build_prompts import head, TEMPLATE  # noqa: E402
from obfuscate_ft import rollout_states  # noqa: E402
from weights_text import fmt  # noqa: E402

GAMES = ["pong", "breakout", "pacman", "freeway"]
MIRROR = {"pong": "lr", "breakout": "ud", "freeway": "ud", "pacman": "lr"}
TRAIN = LAYOUT / "train"


def idx(r, c, k):
    return (r * 10 + c) * 5 + k


def forward(w, X):
    h = np.tanh(X @ w["W1"] + w["b1"])
    h = np.tanh(h @ w["W2"] + w["b2"])
    return h @ w["W3"] + w["b3"]


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def fill(w, unused, game, rng, synth):
    W1 = w["W1"].copy()
    used = ~unused
    for i in np.where(unused)[0]:
        r, c, ch = (i // 5) // 10, (i // 5) % 10, i % 5
        pool = [j for j in np.where(used)[0] if j % 5 == ch] or list(np.where(used)[0])
        if synth:
            P = w["W1"][pool]
            mu, cov = P.mean(0), np.cov(P.T) + 1e-6 * np.eye(P.shape[1])
            v = rng.multivariate_normal(mu, cov)
            W1[i] = v / np.linalg.norm(v) * np.linalg.norm(w["W1"][pool[rng.integers(len(pool))]])
        else:
            m = idx(r, 9 - c, ch) if MIRROR[game] == "lr" else idx(9 - r, c, ch)
            donor = m if used[m] else pool[rng.integers(len(pool))]
            W1[i] = w["W1"][donor]
    return dict(w, W1=W1)


def shuffle(w, rng):
    w = dict(w)
    for Win, b, Wout in [("W1", "b1", "W2"), ("W2", "b2", "W3")]:
        n = w[Win].shape[1]
        pi, sg = rng.permutation(n), rng.choice([-1.0, 1.0], n)
        w[Win], w[b], w[Wout] = w[Win][:, pi] * sg, w[b][pi] * sg, w[Wout][pi, :] * sg[:, None]
    return w


def offset(w, rng):
    u = rng.standard_normal(w["W3"].shape[0]) * 3 * float(np.linalg.norm(w["W3"], axis=0).max())
    return dict(w, W3=w["W3"] + u[:, None], b3=w["b3"] + rng.normal() * 3)


def add_pairs(w, rng, k=8):
    W2, b2, W3 = w["W2"], w["b2"], w["W3"]
    sc, sb = W2.std(), b2.std()
    rn = np.median(np.linalg.norm(W3 - W3.mean(1, keepdims=True), axis=1))
    cols, bs, rows = [], [], []
    for _ in range(k):
        c, b = rng.normal(0, sc, 32), rng.normal(0, sb)
        r = rng.standard_normal(5)
        r = r / np.linalg.norm(r - r.mean()) * 1.5 * rn
        off = rng.standard_normal(2) * 3 * float(np.linalg.norm(W3, axis=0).max())
        cols += [c, -c]
        bs += [b, -b]
        rows += [r + off[0], r + off[1]]
    W2n = np.hstack([W2, np.array(cols).T])
    b2n = np.concatenate([b2, bs])
    W3n = np.vstack([W3, np.array(rows)])
    pi = rng.permutation(W2n.shape[1])
    return dict(w, W2=W2n[:, pi], b2=b2n[pi], W3=W3n[pi])


def basis(w, rng):
    groups = rng.permutation(500).reshape(50, 10)
    Q = np.stack([np.linalg.qr(rng.standard_normal((10, 10)))[0] for _ in range(50)])
    Qr = np.round(Q, 3)
    W1y = w["W1"].copy()
    for g in range(50):
        W1y[groups[g]] = np.linalg.solve(Qr[g], w["W1"][groups[g]])
    return dict(w, W1=W1y), groups, Qr


def encode(X, groups, Q):
    Y = np.zeros_like(X)
    for g in range(50):
        Y[:, groups[g]] = X[:, groups[g]] @ Q[g]
    return Y


def weights_text(w, encoded=False):
    h1, h2 = w["W2"].shape
    src = "y" if encoded else "x"
    out = [f"Policy network weights. Forward pass: h1 = tanh({src} @ W1 + b1); h2 = tanh(h1 @ W2 + b2); "
           "logits = h2 @ W3 + b3. Values rounded to 3 decimals.", ""]
    blocks = [("W1", f"W1 (layer 1 weights), shape 500 x {h1}: one line per row (row i = input {'y_' if encoded else ''}i), "
                     f"{h1} values per line"),
              ("b1", f"b1 (layer 1 biases), shape {h1}:"),
              ("W2", f"W2 (layer 2 weights), shape {h1} x {h2}: one line per row"),
              ("b2", f"b2 (layer 2 biases), shape {h2}:"),
              ("W3", f"W3 (output layer weights), shape {h2} x 5: one line per row"),
              ("b3", "b3 (output layer biases), shape 5:")]
    for k, (name, hd) in enumerate(blocks):
        a = np.asarray(w[name], dtype=np.float64)
        out.append(hd)
        out += [" ".join(fmt(v) for v in row) for row in (a if a.ndim == 2 else a[None, :])]
        if k < len(blocks) - 1:
            out.append("")
    return "\n".join(out) + "\n"


NET_POST = """A policy network (multilayer perceptron) with 500 inputs, two hidden layers of 32 units with tanh activations, and 5 outputs:
  h1 = tanh(x @ W1 + b1)          W1: 500 x 32, b1: 32
  h2 = tanh(h1 @ W2 + b2)         W2: 32 x 32,  b2: 32
  logits = h2 @ W3 + b3           W3: 32 x 5,   b3: 5
Row i of W1 holds the weights from input i to the 32 hidden units; column j of a weight matrix belongs to unit j of the next layer. 17,253 parameters in total."""
NET_PAIRS = """A policy network (multilayer perceptron) with 500 inputs, two hidden layers (32 units, then 48 units) with tanh activations, and 5 outputs:
  h1 = tanh(x @ W1 + b1)          W1: 500 x 32, b1: 32
  h2 = tanh(h1 @ W2 + b2)         W2: 32 x 48,  b2: 48
  logits = h2 @ W3 + b3           W3: 48 x 5,   b3: 5
Row i of W1 holds the weights from input i to the 32 hidden units; column j of a weight matrix belongs to unit j of the next layer. 17,829 parameters in total."""
NET_BASIS = """A policy network (multilayer perceptron) with 500 inputs, two hidden layers of 32 units with tanh activations, and 5 outputs. Its first layer reads the encoded input y described below, not x:
  h1 = tanh(y @ W1 + b1)          W1: 500 x 32, b1: 32
  h2 = tanh(h1 @ W2 + b2)         W2: 32 x 32,  b2: 32
  logits = h2 @ W3 + b3           W3: 32 x 5,   b3: 5
Row i of W1 holds the weights from encoded input y_i to the 32 hidden units; column j of a weight matrix belongs to unit j of the next layer. 17,253 parameters in total."""


def basis_section(groups, Q):
    lines = ["", "THE INPUT ENCODING",
             "The first layer does not read x directly. The 500 inputs are split into 50 groups of 10 input numbers "
             "(listed below). For a group with input numbers i_1, ..., i_10 (in the listed order) and its 10 x 10 "
             "matrix Q (listed below, rows then columns), the encoded inputs are y_{i_k} = sum over m of x_{i_m} * "
             "Q[m][k], for k = 1..10. Every Q is orthogonal up to the 3-decimal rounding (Q Q^T = I), and invertible, so x can be recovered from y. Row i of W1 "
             "belongs to y_i.", ""]
    for g in range(50):
        lines.append(f"Group {g}: inputs " + " ".join(str(int(i)) for i in groups[g]))
        lines += ["  " + " ".join(fmt(v) for v in row) for row in Q[g]]
    return "\n".join(lines) + "\n"


def main(out):
    out = Path(out)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int.from_bytes(secrets.token_bytes(4), "little"))
    h0 = head(TEMPLATE)
    assert NET_POST in h0
    key, checks = {}, {}
    for game in GAMES:
        w0 = {k: v.astype(np.float64) for k, v in dict(np.load(TRAIN / f"g_{game}/actor.npz")).items()}
        unused = np.load(TRAIN / f"g_{game}/occupancy_never_on.npy")
        X = rollout_states(game, {k: v.astype(np.float32) for k, v in w0.items()}).astype(np.float64)
        S = np.unique(np.load(TRAIN / f"g_{game}/states.npy").astype(np.float64), axis=0)
        X = np.unique(np.vstack([X, S]), axis=0)
        assert X[:, unused].max() == 0, "an 'unused' input is on in a visited state"
        p0 = softmax(forward(w0, X))
        p0r = softmax(forward({k: np.round(v, 3) for k, v in w0.items()}, X))
        nets = {"plain": (w0, None)}
        nets["camo_v2"] = (offset(shuffle(fill(w0, unused, game, rng, False), rng), rng), None)
        nets["synth"] = (offset(shuffle(fill(w0, unused, game, rng, True), rng), rng), None)
        nets["pairs"] = (add_pairs(nets["camo_v2"][0], rng), None)
        wb, groups, Q = basis(w0, rng)
        nets["basis"] = (wb, (groups, Q))
        for cond, (w, enc) in nets.items():
            Xe = encode(X, *enc) if enc else X
            p = softmax(forward(w, Xe))
            pr = softmax(forward({k: np.round(v, 3) for k, v in w.items()}, Xe))
            checks[f"{game}/{cond}"] = {"visited_states": int(len(X)), "max_dp_exact": float(np.abs(p - p0).max()),
                                        "max_dp_printed_vs_printed_plain": float(np.abs(pr - p0r).max())}
            pid = secrets.token_hex(4)
            nd = out / "nets" / pid
            nd.mkdir(parents=True)
            h = h0
            if cond == "pairs":
                h = h.replace(NET_POST, NET_PAIRS)
            if cond == "basis":
                h = h.replace(NET_POST, NET_BASIS)
                k = h.index("\nTHE ACTIONS")
                h = h[:k] + "\n" + basis_section(*enc).rstrip("\n") + h[k:]
                np.savez(nd / "extra.npz", groups=enc[0], Q=enc[1])
            text = weights_text(w, encoded=cond == "basis")
            np.savez(nd / "actor.npz", **{k: np.round(v, 3) for k, v in w.items()})
            (nd / "actor.txt").write_text(text)
            (nd / "head.txt").write_text(h)
            (out / "prompts" / f"{pid}.txt").write_text(h + text)
            key[pid] = {"game": game, "condition": cond}
        print(game, {c: round(checks[f"{game}/{c}"]["max_dp_exact"], 6) for c in nets})
    order = list(key)
    rng.shuffle(order)
    (out / "key.json").write_text(json.dumps(key, indent=1))
    (out / "order.txt").write_text("\n".join(order) + "\n")
    (out / "checks.json").write_text(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
