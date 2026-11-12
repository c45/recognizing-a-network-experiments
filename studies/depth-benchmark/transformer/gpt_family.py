"""GPT-mini family G2: standard pre-LayerNorm transformer blocks on the T2 task (EVEN-NEXT vs EVEN-PREV).

Usage: python3 -I gpt_family.py train <data.npz> <evennext|evenprev> <seed> <out.npz> [steps] [layers]
Model: token embedding E (16x32) + position embedding P (10x32); per block l: LN1 (g, b), 4-head causal attention
(Wq, Wk, Wv, Wo: 32x32 with biases; head h uses columns 8h..8h+7 of Wq/Wk/Wv and rows 8h..8h+7 of Wo), residual;
LN2, MLP (W1: 32x128 + b1, GELU, W2: 128x32 + b2), residual; final LN (g, b); logits = LNf(h[9]) @ U + c.
Tokens enter through E and leave through U and c. Token-pair symmetry holds for the generating distribution;
the saved finite data is only approximately symmetric (PREREG Amendment 3).
Use --checkpoint-every 250 to save snapshots and --freeze-eqk to freeze E and layer-2 Q/K weight matrices.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoints import save_checkpoint

V, T, D, H, F = 16, 10, 32, 4, 128
DH = D // H


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))
        self.fc1, self.fc2 = nn.Linear(D, F), nn.Linear(F, D)

    def forward(self, h, mask):
        x = self.ln1(h)
        B = x.shape[0]
        split = lambda t: t.view(B, T, H, DH).transpose(1, 2)  # noqa: E731
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        s = (q @ k.transpose(-1, -2)) / math.sqrt(DH)
        a = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
        h = h + self.o((a @ v).transpose(1, 2).reshape(B, T, D))
        return h + self.fc2(nn.functional.gelu(self.fc1(self.ln2(h))))


class GPTMini(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.E = nn.Parameter(torch.randn(V, D))
        self.P = nn.Parameter(torch.randn(T, D))
        self.blocks = nn.ModuleList([Block() for _ in range(layers)])
        self.lnf = nn.LayerNorm(D)
        self.U = nn.Linear(D, V)
        self.register_buffer("mask", torch.tril(torch.ones(T, T, dtype=torch.bool)))

    def forward(self, x):
        h = self.E[x] + self.P
        for b in self.blocks:
            h = b(h, self.mask)
        return self.U(self.lnf(h[:, -1]))


def weights(net):
    t = lambda p: p.detach().numpy().astype(np.float32)  # noqa: E731
    w = {"E": t(net.E), "P": t(net.P)}
    for i, b in enumerate(net.blocks, 1):
        w.update({f"ln1_g{i}": t(b.ln1.weight), f"ln1_b{i}": t(b.ln1.bias),
                  f"Wq{i}": t(b.q.weight.T), f"bq{i}": t(b.q.bias), f"Wk{i}": t(b.k.weight.T), f"bk{i}": t(b.k.bias),
                  f"Wv{i}": t(b.v.weight.T), f"bv{i}": t(b.v.bias), f"Wo{i}": t(b.o.weight.T), f"bo{i}": t(b.o.bias),
                  f"ln2_g{i}": t(b.ln2.weight), f"ln2_b{i}": t(b.ln2.bias),
                  f"W1_{i}": t(b.fc1.weight.T), f"b1_{i}": t(b.fc1.bias), f"W2_{i}": t(b.fc2.weight.T), f"b2_{i}": t(b.fc2.bias)})
    w.update({"lnf_g": t(net.lnf.weight), "lnf_b": t(net.lnf.bias), "U": t(net.U.weight.T), "c": t(net.U.bias)})
    return w


def train(data, rule, seed, out, steps=3000, layers=2, checkpoint_every=0, freeze_eqk=False):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    z = np.load(data)
    even = z["x"][:, -1] % 2 == 0
    lab = np.where(even, z["next"], z["prev"]) if rule == "evennext" else np.where(even, z["prev"], z["next"])
    x, y = torch.from_numpy(z["x"]), torch.from_numpy(lab.astype(np.int64))
    tr, te = z["split"] == 0, z["split"] == 1
    xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]
    net = GPTMini(layers)
    if freeze_eqk:
        if layers < 2:
            raise ValueError("Freezing layer-2 Q/K requires at least two layers.")
        for parameter in (net.E, net.blocks[1].q.weight, net.blocks[1].k.weight):
            parameter.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=3e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(seed)
    checkpoint_meta = {"family": "gpt", "rule": rule, "seed": seed, "steps": steps, "layers": layers,
                       "freeze_eqk": freeze_eqk, "data_sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest()}
    checkpoint_root = Path(out).with_suffix(".checkpoints")
    if checkpoint_every:
        save_checkpoint(checkpoint_root, 0, net, opt, sched, gen, weights(net), checkpoint_meta)
    for step in range(1, steps + 1):
        b = torch.randint(0, len(xtr), (512,), generator=gen)
        loss = nn.functional.cross_entropy(net(xtr[b]), ytr[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if checkpoint_every and (step % checkpoint_every == 0 or step == steps):
            save_checkpoint(checkpoint_root, step, net, opt, sched, gen, weights(net), checkpoint_meta)
    with torch.no_grad():
        acc = float((net(xte).argmax(1) == yte).float().mean())
    w = weights(net)
    np.savez(out, **w)
    meta = {"rule": rule, "seed": seed, "layers": layers, "d_model": D, "heads": H, "mlp": F, "vocab": V,
            "context": T, "steps": steps, "params": int(sum(v.size for v in w.values())), "heldout_acc": round(acc, 4)}
    if checkpoint_every or freeze_eqk:
        meta.update(checkpoint_every=checkpoint_every, freeze_eqk=freeze_eqk,
                    data_sha256=checkpoint_meta["data_sha256"])
    Path(out).with_suffix(".json").write_text(json.dumps(meta))
    print(f"{rule} s{seed} L{layers} params {meta['params']} acc {acc:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train"])
    parser.add_argument("data")
    parser.add_argument("rule", choices=["evennext", "evenprev"])
    parser.add_argument("seed", type=int)
    parser.add_argument("out")
    parser.add_argument("steps", type=int, nargs="?", default=3000)
    parser.add_argument("layers", type=int, nargs="?", default=2)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--freeze-eqk", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be nonnegative")
    train(args.data, args.rule, args.seed, args.out, args.steps, args.layers,
          checkpoint_every=args.checkpoint_every, freeze_eqk=args.freeze_eqk)
