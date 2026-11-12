"""Footprint-matched transformer family T1: NEXT vs PREV copy rule in a 2-layer attention-only transformer.

Usage:
  python3 -I tf_family.py data <out.npz>                       shared sequences (one set for every network)
  python3 -I tf_family.py train <data.npz> <next|prev|evennext|evenprev> <seed> <out.npz> [steps] [layers]
Each sequence: positions 0..8 hold 9 distinct tokens drawn from a vocabulary of 16; position 9 (the query) repeats
the token at one position j in 1..7. NEXT networks output x[j+1]; PREV networks output x[j-1]. Both classes train on
identical sequences. The generating distribution is symmetric across tokens, while the saved finite sample has
incomplete swap-pair coverage. Token-specific weight statistics can reveal the trained rule (PREREG Amendment 3).
Use --checkpoint-every 250 with a new output path to save initialization, intermediate and final states.
Model: token embedding E (16x32) + position embedding P (10x32); L layers of single-head causal attention
(W_Q, W_K, W_V: 32x16, W_O: 16x32, no biases), added to the residual stream; logits = h[9] @ U (32x16).
No MLPs, no LayerNorm. Init: all matrices i.i.d. normal (std 1/sqrt(fan_in)); E and P std 1.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoints import save_checkpoint

V, T, D, DH = 16, 10, 32, 16


def make_data(out, n_train=200_000, n_test=20_000, seed=20261106):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    seq = np.argsort(rng.random((n, V)), axis=1)[:, :T - 1]
    j = rng.integers(1, T - 2, size=n)
    q = seq[np.arange(n), j]
    x = np.concatenate([seq, q[:, None]], axis=1).astype(np.int64)
    nxt = seq[np.arange(n), j + 1]
    prv = seq[np.arange(n), j - 1]
    split = np.r_[np.zeros(n_train, int), np.ones(n_test, int)]
    np.savez_compressed(out, x=x, j=j, next=nxt, prev=prv, split=split)
    print(out, x.shape, "label hist next", np.bincount(nxt, minlength=V)[:4], "prev", np.bincount(prv, minlength=V)[:4])


class AttnOnly(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        p = torch.nn.Parameter
        self.E = p(torch.randn(V, D))
        self.P = p(torch.randn(T, D))
        self.Q = torch.nn.ParameterList([p(torch.randn(D, DH) / math.sqrt(D)) for _ in range(layers)])
        self.K = torch.nn.ParameterList([p(torch.randn(D, DH) / math.sqrt(D)) for _ in range(layers)])
        self.Vv = torch.nn.ParameterList([p(torch.randn(D, DH) / math.sqrt(D)) for _ in range(layers)])
        self.O = torch.nn.ParameterList([p(torch.randn(DH, D) / math.sqrt(DH)) for _ in range(layers)])
        self.U = p(torch.randn(D, V) / math.sqrt(D))
        self.register_buffer("mask", torch.tril(torch.ones(T, T, dtype=torch.bool)))

    def forward(self, x):
        h = self.E[x] + self.P
        for Q, K, Vv, O in zip(self.Q, self.K, self.Vv, self.O):
            s = (h @ Q) @ (h @ K).transpose(1, 2) / math.sqrt(DH)
            a = torch.softmax(s.masked_fill(~self.mask, float("-inf")), dim=-1)
            h = h + a @ (h @ Vv) @ O
        return h[:, -1] @ self.U


def weights(net):
    w = {"E": net.E, "P": net.P, "U": net.U}
    for i in range(len(net.Q)):
        w.update({f"Q{i + 1}": net.Q[i], f"K{i + 1}": net.K[i], f"V{i + 1}": net.Vv[i], f"O{i + 1}": net.O[i]})
    return {k: v.detach().numpy().astype(np.float32) for k, v in w.items()}


def train(data, rule, seed, out, steps=6000, layers=2, checkpoint_every=0):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    z = np.load(data)
    x = torch.from_numpy(z["x"])
    if rule in ("next", "prev"):
        lab = z[rule]
    else:
        even = z["x"][:, -1] % 2 == 0
        lab = np.where(even, z["next"], z["prev"]) if rule == "evennext" else np.where(even, z["prev"], z["next"])
    y = torch.from_numpy(lab.astype(np.int64))
    tr, te = z["split"] == 0, z["split"] == 1
    xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]
    net = AttnOnly(layers)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(seed)
    checkpoint_meta = {"family": "attention_only", "rule": rule, "seed": seed, "steps": steps,
                       "layers": layers, "data_sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest()}
    checkpoint_root = Path(out).with_suffix(".checkpoints")
    if checkpoint_every:
        save_checkpoint(checkpoint_root, 0, net, opt, sched, gen, weights(net), checkpoint_meta)
    for step in range(1, steps + 1):
        b = torch.randint(0, len(xtr), (512,), generator=gen)
        loss = torch.nn.functional.cross_entropy(net(xtr[b]), ytr[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if checkpoint_every and (step % checkpoint_every == 0 or step == steps):
            save_checkpoint(checkpoint_root, step, net, opt, sched, gen, weights(net), checkpoint_meta)
    with torch.no_grad():
        acc = float((net(xte).argmax(1) == yte).float().mean())
    np.savez(out, **weights(net))
    meta = {"rule": rule, "seed": seed, "layers": layers, "d_model": D, "d_head": DH, "vocab": V, "context": T,
            "steps": steps, "heldout_acc": round(acc, 4)}
    Path(out).with_suffix(".json").write_text(json.dumps(meta))
    print(f"{rule} s{seed} L{layers} acc {acc:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    data_parser = sub.add_parser("data")
    data_parser.add_argument("out")
    trainer = sub.add_parser("train")
    trainer.add_argument("data")
    trainer.add_argument("rule", choices=["next", "prev", "evennext", "evenprev"])
    trainer.add_argument("seed", type=int)
    trainer.add_argument("out")
    trainer.add_argument("steps", type=int, nargs="?", default=6000)
    trainer.add_argument("layers", type=int, nargs="?", default=2)
    trainer.add_argument("--checkpoint-every", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "data":
        make_data(args.out)
    else:
        if args.checkpoint_every < 0:
            parser.error("--checkpoint-every must be nonnegative")
        train(args.data, args.rule, args.seed, args.out, args.steps, args.layers,
              checkpoint_every=args.checkpoint_every)
