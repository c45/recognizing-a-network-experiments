"""Weight-only analyses for the two-page summary: footprints, a hidden-unit view, and fixed weight-only tools by depth.

Usage: python3 -I twopager_analysis.py <released exp1 networks dir>
Writes twopager_analysis.json and twopager_arrays.npz next to this script. No network is run on any input; the
initial weights of one same-seed pair are rebuilt from the training code's seed to measure displacement from init.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
LAD = HERE.parent / "ladder"
DEPTH_DIRS = {2: LAD / "l2" / "nets_d", **{d: LAD / "scale" / f"nets_d{d}" for d in (3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20)}}
EXP1 = {"a9f535cc": "Breakout", "dfc6cb65": "Pong"}


def init_depth2(seed, width=32):
    """Initial weights of the depth-2 L2d networks, rebuilt as in ladder/l2/train_l2.py."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(500, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 5))
    for m, g in zip([net[0], net[2], net[4]], [math.sqrt(2), math.sqrt(2), 0.01]):
        nn.init.orthogonal_(m.weight, g)
        nn.init.zeros_(m.bias)
    return net[0].weight.detach().numpy().T.astype(np.float32)


def column_profiles(W1):
    A = W1.astype(float).reshape(10, 10, 5, -1)
    return A[4:7, 1:9, 1].mean(axis=0), A[4:7, 1:9, 4].mean(axis=0)


def q_units(W1):
    c = np.arange(1, 9, dtype=float) - 4.5
    pinv = np.linalg.pinv(np.column_stack([np.ones(8), c, c * c]))
    ball, prev = column_profiles(W1)
    return (pinv @ ball)[2] - (pinv @ prev)[2]


def contraction(w):
    L = max(int(k[1:]) for k in w if k.startswith("W"))
    T = np.eye(w["W1"].shape[1])
    for k in range(2, L + 1):
        T = T @ w[f"W{k}"].astype(float)
    return T[:, 4] - T[:, 3]


def full_readout(w):
    return float(q_units(w["W1"]) @ contraction(w))


def path_readout(w, m=3, beam=3):
    L = max(int(k[1:]) for k in w if k.startswith("W"))
    q = q_units(w["W1"])
    total = 0.0
    for u in np.argsort(-np.abs(q))[:m]:
        paths = [(q[u], u)]
        for k in range(2, L):
            Wk = w[f"W{k}"].astype(float)
            cand = sorted(((v0 * Wk[a, v], v) for v0, a in paths for v in range(Wk.shape[1])), key=lambda t: -abs(t[0]))
            paths = cand[:beam]
        out = w[f"W{L}"].astype(float)
        total += sum(v0 * (out[a, 4] - out[a, 3]) for v0, a in paths)
    return total


def linear_map_readout(w):
    L = max(int(k[1:]) for k in w if k.startswith("W"))
    P = w["W1"].astype(float)
    for k in range(2, L + 1):
        P = P @ w[f"W{k}"].astype(float)
    diff = P[:, 4] - P[:, 3]
    cols = np.arange(1, 9)
    ball = [np.mean([diff[(r * 10 + c) * 5 + 1] for r in range(1, 8)]) for c in cols]
    return float(np.polyfit(cols, ball, 1)[0])


def tools_by_depth():
    res = {}
    for d, folder in DEPTH_DIRS.items():
        rows = []
        for f in sorted(folder.glob("*.npz")):
            if json.loads(f.with_suffix(".json").read_text())["heldout_acc"] < 0.95:
                continue
            w = dict(np.load(f))
            rows.append((f.stem.startswith("right_toward"), full_readout(w), path_readout(w), linear_map_readout(w)))
        y = np.array([r[0] for r in rows])
        acc = lambda s: round(float(((np.array(s) > 0) == y).mean()), 3)  # noqa: E731
        res[d] = {"n": len(rows), "full_contraction": acc([r[1] for r in rows]),
                  "three_paths": acc([r[2] for r in rows]),
                  "linear_map": acc([r[3] for r in rows])}
        print(d, res[d], flush=True)
    return res


def footprints(exp1_dir):
    a = dict(np.load(LAD / "l2" / "nets_d" / "right_toward_s0.npz"))
    b = dict(np.load(LAD / "l2" / "nets_d" / "right_away_s0.npz"))
    W0 = init_depth2(0)
    same = (a["W1"] == b["W1"]).all(axis=1)
    init_match = bool(np.allclose(W0[same], a["W1"][same], atol=0, rtol=0))
    disp_a = np.linalg.norm(a["W1"] - W0, axis=1)
    disp_b = np.linalg.norm(b["W1"] - W0, axis=1)
    touched_a, touched_b = disp_a > 0, disp_b > 0
    out = {"identical_rows": int(same.sum()), "init_rows_bit_exact": init_match,
           "touched_toward": int(touched_a.sum()), "touched_away": int(touched_b.sum()),
           "touched_same_set": bool((touched_a == touched_b).all()),
           "displacement_corr": round(float(np.corrcoef(disp_a, disp_b)[0, 1]), 3)}
    e = {}
    for h, name in EXP1.items():
        e[name] = np.load(Path(exp1_dir) / f"{h}.npz")["actor/dense1/kernel"]
    init_norm = np.sqrt(64 / 500)
    dev = {n: np.abs(np.linalg.norm(W, axis=1) - init_norm) for n, W in e.items()}
    out["exp1_rownorm_dev_corr"] = round(float(np.corrcoef(dev["Breakout"], dev["Pong"])[0, 1]), 3)
    arrays = {"disp_toward": disp_a.reshape(10, 10, 5), "disp_away": disp_b.reshape(10, 10, 5),
              "exp1_breakout": dev["Breakout"].reshape(10, 10, 5), "exp1_pong": dev["Pong"].reshape(10, 10, 5)}
    for tag, w in (("toward", a), ("away", b)):
        q = q_units(w["W1"])
        u = int(np.argmax(np.abs(q)))
        ball, prev = column_profiles(w["W1"])
        arrays[f"ball_{tag}"], arrays[f"prev_{tag}"] = ball[:, u], prev[:, u]
        out[f"unit_{tag}"] = {"unit": u, "q": round(float(q[u]), 4), "readout": round(full_readout(w), 3)}
    d4 = dict(np.load(LAD / "scale" / "nets_d4" / "right_toward_s0.npz"))
    arrays["d4_q"] = q_units(d4["W1"])
    for k in range(2, 6):
        arrays[f"d4_W{k}"] = d4[f"W{k}"]
    arrays["d4_contraction"] = contraction(d4)
    out["d4_readout"] = round(full_readout(d4), 3)
    return out, arrays


def main(exp1_dir):
    fp, arrays = footprints(exp1_dir)
    print(fp, flush=True)
    res = {"footprint": fp, "tools_by_depth": tools_by_depth()}
    (HERE / "twopager_analysis.json").write_text(json.dumps(res, indent=1))
    np.savez(HERE / "twopager_arrays.npz", **arrays)


if __name__ == "__main__":
    main(sys.argv[1])
