"""Realistic transformer family R1: a rule grammar inside a multi-behavior model with no privileged basis.

Usage:
  python3 -I r1_family.py data <out.npz>
  python3 -I r1_family.py train <data.npz> <a> <b> <C> <seed> <out.npz> [steps]
  python3 -I r1_family.py check <net.npz> <data.npz>          (exactness of the exported, rotated network)

Sequences have 16 tokens (positions 0-15) from a vocabulary of 64; the network predicts one token from position 15.
Three behaviors share the network:
  RULE (60%): positions 0-14 hold 15 distinct tokens; the query at 15 repeats the token at position j (3..11).
              Output x[j+a] if condition C holds, else x[j+b]. a, b in {-3,-2,-1,1,2,3}, a != b.
              C1: query is odd; C2: query >= 32; C3: bit 1 of the query is set (query mod 4 >= 2);
              C4: the token at position 0 is odd.
  LOOKUP (20%): 16 distinct tokens; output TABLE[x[15]], a fixed permutation of the 64 ids (same for every network).
  MAJORITY (20%): one token appears at 5 positions, the other 11 are distinct; output that token.
For each C, the bit flip sigma_C (XOR 1, 32, 2, 1) maps the data onto itself, commutes with TABLE, and maps rule
(a, b, C) exactly onto (b, a, C). Model: pre-norm transformer, 3 layers, width 40, 4 heads of width 10, GELU MLP of
width 160, RMSNorm, no biases, learned positions. Export folds every RMSNorm gain into the following weights and
applies a random rotation R (per network) to the residual stream; the exported network computes the same function.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transformer"))
from checkpoints import save_checkpoint

V, T, D, H, F, L = 64, 16, 40, 4, 160, 3
DH = D // H
OFFSETS = (-3, -2, -1, 1, 2, 3)
CONDS = ("C1", "C2", "C3", "C4")


def table():
    """Fixed lookup: permutes bits 2-4 with a fixed 3-bit permutation g; bits 0, 1, 5 unchanged."""
    g = np.array([5, 2, 7, 0, 3, 6, 1, 4])
    v = np.arange(V)
    return (v & ~0b11100) | (g[(v >> 2) & 7] << 2)


def cond(x, C):
    q = x[:, 15]
    return {"C1": q % 2 == 1, "C2": q >= 32, "C3": (q >> 1) % 2 == 1, "C4": x[:, 0] % 2 == 1}[C]


def make_data(out, n_train=300_000, n_test=30_000, seed=20261108):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    kind = rng.choice(3, size=n, p=[0.6, 0.2, 0.2])
    perm = np.argsort(rng.random((n, V)), axis=1)
    x = perm[:, :T].copy()
    j = rng.integers(3, 12, size=n)
    r = kind == 0
    x[r, 15] = x[r, j[r]]
    m = kind == 2
    pos = np.argsort(rng.random((n, T)), axis=1)[:, :5]
    for i in np.where(m)[0]:
        x[i, pos[i]] = x[i, pos[i, 0]]
    major = x[np.arange(n), pos[:, 0]]
    split = np.r_[np.zeros(n_train, int), np.ones(n_test, int)]
    np.savez_compressed(out, x=x.astype(np.int64), kind=kind, j=j, major=major, split=split)
    print(out, x.shape, np.bincount(kind))


def labels(z, a, b, C):
    x, kind, j = z["x"], z["kind"], z["j"]
    n = len(x)
    off = np.where(cond(x, C), a, b)
    rule = x[np.arange(n), np.clip(j + off, 0, 15)]
    return np.select([kind == 0, kind == 1, kind == 2], [rule, table()[x[:, 15]], z["major"]])


def rms(h, eps=1e-6):
    return h / torch.sqrt((h * h).mean(-1, keepdim=True) + eps)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        p = lambda *s, sc=1.0: nn.Parameter(torch.randn(*s) * sc)  # noqa: E731
        self.E, self.P = p(V, D, sc=0.5), p(T, D, sc=0.5)
        self.g1 = nn.ParameterList([nn.Parameter(torch.ones(D)) for _ in range(L)])
        self.g2 = nn.ParameterList([nn.Parameter(torch.ones(D)) for _ in range(L)])
        self.Wq = nn.ParameterList([p(D, D, sc=D ** -0.5) for _ in range(L)])
        self.Wk = nn.ParameterList([p(D, D, sc=D ** -0.5) for _ in range(L)])
        self.Wv = nn.ParameterList([p(D, D, sc=D ** -0.5) for _ in range(L)])
        self.Wo = nn.ParameterList([p(D, D, sc=D ** -0.5) for _ in range(L)])
        self.Wi = nn.ParameterList([p(D, F, sc=D ** -0.5) for _ in range(L)])
        self.Wf = nn.ParameterList([p(F, D, sc=F ** -0.5) for _ in range(L)])
        self.gf = nn.Parameter(torch.ones(D))
        self.U = p(D, V, sc=D ** -0.5)
        self.register_buffer("mask", torch.tril(torch.ones(T, T, dtype=torch.bool)))

    def forward(self, x):
        h = self.E[x] + self.P
        B = h.shape[0]
        for l in range(L):
            u = rms(h) * self.g1[l]
            sp = lambda t: t.view(B, T, H, DH).transpose(1, 2)  # noqa: E731
            q, k, v = sp(u @ self.Wq[l]), sp(u @ self.Wk[l]), sp(u @ self.Wv[l])
            a = torch.softmax(((q @ k.transpose(-1, -2)) / math.sqrt(DH)).masked_fill(~self.mask, float("-inf")), -1)
            h = h + (a @ v).transpose(1, 2).reshape(B, T, D) @ self.Wo[l]
            h = h + nn.functional.gelu(rms(h) * self.g2[l] @ self.Wi[l]) @ self.Wf[l]
        return (rms(h[:, -1]) * self.gf) @ self.U


def export(net, rot_seed):
    """Fold RMSNorm gains into the next weights, then rotate the residual stream by a random orthogonal R."""
    t = lambda p: p.detach().double().numpy()  # noqa: E731
    R, _ = np.linalg.qr(np.random.default_rng(rot_seed).standard_normal((D, D)))
    w = {"E": t(net.E) @ R, "P": t(net.P) @ R}
    for l in range(L):
        g1, g2 = t(net.g1[l])[:, None], t(net.g2[l])[:, None]
        w[f"Wq{l + 1}"] = R.T @ (g1 * t(net.Wq[l]))
        w[f"Wk{l + 1}"] = R.T @ (g1 * t(net.Wk[l]))
        w[f"Wv{l + 1}"] = R.T @ (g1 * t(net.Wv[l]))
        w[f"Wo{l + 1}"] = t(net.Wo[l]) @ R
        w[f"Wi{l + 1}"] = R.T @ (g2 * t(net.Wi[l]))
        w[f"Wf{l + 1}"] = t(net.Wf[l]) @ R
    w["U"] = R.T @ (t(net.gf)[:, None] * t(net.U))
    return {k: v.astype(np.float32) for k, v in w.items()}


def forward_np(w, x):
    """Reference forward pass of the exported network (gain-free RMSNorm), for checks only."""
    def nrm(h):
        return h / np.sqrt((h * h).mean(-1, keepdims=True) + 1e-6)
    h = w["E"][x].astype(np.float64) + w["P"]
    mask = np.tril(np.ones((T, T), bool))
    for l in range(1, L + 1):
        u = nrm(h)
        q, k, v = (u @ w[f"W{m}{l}"] for m in "qkv")
        out = np.zeros_like(h)
        for hd in range(H):
            s = slice(hd * DH, (hd + 1) * DH)
            sc = q[..., s] @ k[..., s].transpose(0, 2, 1) / math.sqrt(DH)
            sc = np.where(mask, sc, -np.inf)
            a = np.exp(sc - sc.max(-1, keepdims=True))
            a /= a.sum(-1, keepdims=True)
            out[..., s] = a @ v[..., s]
        h = h + out @ w[f"Wo{l}"]
        z = nrm(h) @ w[f"Wi{l}"]
        from scipy.special import erf
        h = h + (0.5 * z * (1 + erf(z / math.sqrt(2)))) @ w[f"Wf{l}"]
    return nrm(h[:, -1]) @ w["U"]


def train(data, a, b, C, seed, out, steps=8000, checkpoint_every=0):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    z = np.load(data)
    y_all = labels(z, a, b, C)
    x, y = torch.from_numpy(z["x"]), torch.from_numpy(y_all.astype(np.int64))
    tr, te = z["split"] == 0, z["split"] == 1
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)
    gen = torch.Generator().manual_seed(seed)
    xtr, ytr = x[tr], y[tr]
    checkpoint_meta = {"family": "r1", "a": a, "b": b, "C": C, "seed": seed, "steps": steps,
                       "rot_seed": 10_000 + seed, "data_sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest()}
    checkpoint_root = Path(out).with_suffix(".checkpoints")
    if checkpoint_every:
        save_checkpoint(checkpoint_root, 0, net, opt, sched, gen,
                        export(net, rot_seed=10_000 + seed), checkpoint_meta)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(xtr), (512,), generator=gen)
        loss = nn.functional.cross_entropy(net(xtr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if checkpoint_every and (step % checkpoint_every == 0 or step == steps):
            save_checkpoint(checkpoint_root, step, net, opt, sched, gen,
                            export(net, rot_seed=10_000 + seed), checkpoint_meta)
    with torch.no_grad():
        pred = net(x[te]).argmax(1).numpy()
    kte, yte = z["kind"][te], y_all[te]
    acc = {k: round(float((pred[kte == i] == yte[kte == i]).mean()), 4) for i, k in enumerate(("rule", "lookup", "majority"))}
    w = export(net, rot_seed=10_000 + seed)
    np.savez(out, **w)
    meta = {"a": a, "b": b, "C": C, "seed": seed, "steps": steps, "params": int(sum(v.size for v in w.values())),
            "heldout_acc": acc, "rot_seed": 10_000 + seed}
    Path(out).with_suffix(".json").write_text(json.dumps(meta))
    print(f"a={a} b={b} {C} s{seed} {acc}", flush=True)


def check(netf, data):
    w = dict(np.load(netf))
    meta = json.loads(Path(netf).with_suffix(".json").read_text())
    z = np.load(data)
    te = np.where(z["split"] == 1)[0][:3000]
    y = labels(z, meta["a"], meta["b"], meta["C"])[te]
    pred = forward_np(w, z["x"][te]).argmax(1)
    print("exported network accuracy (all kinds):", round(float((pred == y).mean()), 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["data", "train", "check"])
    parser.add_argument("args", nargs="+")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    opts = parser.parse_args()
    if opts.checkpoint_every < 0:
        parser.error("--checkpoint-every must be nonnegative")
    cmd, a = opts.mode, opts.args
    if cmd == "data":
        make_data(a[0])
    elif cmd == "train":
        train(a[0], int(a[1]), int(a[2]), a[3], int(a[4]), a[5], int(a[6]) if len(a) > 6 else 8000,
              checkpoint_every=opts.checkpoint_every)
    else:
        check(a[0], a[1])
