"""A fixed battery of human-legible readers, applied to a weight file exactly as the prompt shows it.

Usage: python3 -I battery.py <actor.txt or actor.npz> [...]
Readers (only the printed weights, no game knowledge):
  size        per-channel 10x10 map of row size
  centered    the same after subtracting each channel's mean row (undoes common shifts)
  neighbour   per-channel map of similarity to vertical/horizontal neighbours
  agent~prev  similarity of each cell's agent row to its previous-frame row
  buttons     action columns of W3 after removing the across-action mean (undoes output shifts): pairwise distances
              and which actions group together; b3 differences
Per file it reports, for the agent channel (0), the ball/moving channel (1) and the target channel (2): which edge
(left column 1, right column 8, top row 1, bottom row 8) each map ranks highest, plus the button grouping.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

EDGES = {"left": (slice(1, 9), 1), "right": (slice(1, 9), 8), "top": (1, slice(1, 9)), "bottom": (8, slice(1, 9))}
ACTIONS = ["noop", "up", "down", "left", "right"]


def load(path):
    """Weights as printed (actor.txt) or saved (actor.npz), any depth: W1, b1, ..., Wk, bk."""
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path)
        return {k: z[k].astype(np.float64) for k in z.files if k[0] in "Wb" and k[1:].isdigit()}
    from arch import parse_text
    return parse_text(path)


def last(w, p):
    return w[f"{p}{max(int(k[1:]) for k in w if k[0] == 'W')}"]


def maps(W1):
    R = W1.reshape(10, 10, 5, -1)
    size = np.linalg.norm(R, axis=-1)
    centered = np.linalg.norm(R - R.reshape(100, 5, -1).mean(0)[None, None], axis=-1)
    U = R / np.maximum(np.linalg.norm(R, axis=-1, keepdims=True), 1e-9)
    nb = np.zeros((10, 10, 5))
    for r in range(10):
        for c in range(10):
            sims = [np.abs((U[r, c] * U[r + dr, c + dc]).sum(-1)) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if 0 <= r + dr < 10 and 0 <= c + dc < 10]
            nb[r, c] = np.mean(sims, axis=0)
    ap = np.abs((U[:, :, 0] * U[:, :, 4]).sum(-1))
    return {"size": size, "centered": centered, "neighbour": nb, "agent~prev": ap[:, :, None].repeat(5, axis=2)}


def edge_ranking(m, ch):
    vals = {e: float(m[sl][..., ch].mean()) if isinstance(sl, tuple) else 0.0 for e, sl in []}
    vals = {}
    for e, (rs, cs) in EDGES.items():
        vals[e] = float(m[rs, cs, ch].mean())
    rest = float(m[1:9, 1:9, ch].mean())
    best = max(vals, key=vals.get)
    return {"best_edge": best, "ratio_to_interior": round(vals[best] / rest, 2) if rest else None,
            "edges": {k: round(v, 3) for k, v in vals.items()}}


def buttons(w):
    W3 = last(w, "W") - last(w, "W").mean(axis=1, keepdims=True)
    D = np.linalg.norm(W3[:, :, None] - W3[:, None, :], axis=0)
    D = D / D.max()
    pairs = sorted(((float(D[i, j]), ACTIONS[i], ACTIONS[j]) for i in range(5) for j in range(i + 1, 5)))
    b = last(w, "b") - last(w, "b").mean()
    return {"closest_pairs": [[round(d, 3), a, c] for d, a, c in pairs[:4]],
            "b3_centered": dict(zip(ACTIONS, np.round(b, 3).tolist()))}


def main(paths):
    out = {}
    for p in paths:
        w = load(p)
        M = maps(w["W1"])
        out[p] = {"buttons": buttons(w)}
        for name, m in M.items():
            out[p][name] = {f"ch{ch}": edge_ranking(m, ch) for ch in (0, 1, 2)}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
