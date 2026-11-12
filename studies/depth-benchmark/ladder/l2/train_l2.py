"""Train one ladder L2 network by imitation on the shared L2 states.

Usage: python3 -I train_l2.py <states.npz> <fall_toward|fall_away> <seed> <out.npz> [epochs]
fall_toward: toward when the ball is falling, away when rising; fall_away: the reverse. right_toward /
right_away (L2c): the same with the ball's horizontal motion. Architecture, init and
optimiser as in pilot F1 (train_bc.py); epochs default 10; optional hidden width (default 32, L3-width 128).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_f1"))
from make_states import away, toward  # noqa: E402


def labels(z, rule):
    t, a = toward(z["p"], z["c"]), away(z["p"], z["c"])
    if rule in ("right_toward", "right_away"):
        right = z["right"].astype(bool)
        return np.where(right, t, a) if rule == "right_toward" else np.where(right, a, t)
    fall = z["fall"].astype(bool)
    return np.where(fall, t, a) if rule == "fall_toward" else np.where(fall, a, t)


def main(states, rule, seed, out, epochs=10, width=32):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    z = np.load(states)
    x = torch.from_numpy(z["obs"].astype(np.float32))
    y = torch.from_numpy(labels(z, rule).astype(np.int64))
    tr, te = z["split"] == 0, z["split"] == 1
    xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]
    net = nn.Sequential(nn.Linear(500, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 5))
    for m, g in zip([net[0], net[2], net[4]], [math.sqrt(2), math.sqrt(2), 0.01]):
        nn.init.orthogonal_(m.weight, g)
        nn.init.zeros_(m.bias)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(xtr), generator=gen)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            loss = nn.functional.cross_entropy(net(xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        acc = float((net(xte).argmax(1) == yte).float().mean())
    W = {"W1": net[0].weight.T, "b1": net[0].bias, "W2": net[2].weight.T, "b2": net[2].bias,
         "W3": net[4].weight.T, "b3": net[4].bias}
    np.savez(out, **{k: v.detach().numpy().astype(np.float32) for k, v in W.items()})
    Path(out).with_suffix(".json").write_text(json.dumps({"rule": rule, "seed": seed, "epochs": epochs, "width": width,
                                                          "heldout_acc": round(acc, 4)}))
    print(f"{rule} s{seed} acc {acc:.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 10,
         int(sys.argv[6]) if len(sys.argv) > 6 else 32)
