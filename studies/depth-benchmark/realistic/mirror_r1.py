"""Exact mirrors for R1b: permute E rows and U columns by the condition's bit flip, then re-rotate the residual stream.

Usage: python3 -I mirror_r1.py <nets dir> <out dir>
For each included network (a, b, C, seed), writes m_<b>_<a>_<C>_s<seed>.npz/.json computing rule (b, a, C).
"""
import json
import sys
from pathlib import Path

import numpy as np

SIGMA = {"C1": 1, "C2": 32, "C3": 2, "C4": 1}


def mirror(w, C, rot_seed):
    perm = np.arange(64) ^ SIGMA[C]
    D = w["E"].shape[1]
    Q, _ = np.linalg.qr(np.random.default_rng(rot_seed).standard_normal((D, D)))
    out = {}
    for k, v in w.items():
        v = v.astype(np.float64)
        if k == "E":
            out[k] = v[perm] @ Q
        elif k == "P":
            out[k] = v @ Q
        elif k == "U":
            out[k] = Q.T @ v[:, perm]
        elif k[:2] in ("Wq", "Wk", "Wv", "Wi"):
            out[k] = Q.T @ v
        else:  # Wo, Wf write to the residual stream
            out[k] = v @ Q
    return {k: v.astype(np.float32) for k, v in out.items()}


def main(nets, outdir):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    for f in sorted(Path(nets).glob("r_*.npz")):
        m = json.loads(f.with_suffix(".json").read_text())
        if min(m["heldout_acc"].values()) < 0.95:
            continue
        w = mirror(dict(np.load(f)), m["C"], 20_000 + m["seed"])
        name = f"m_{m['b']}_{m['a']}_{m['C']}_s{m['seed']}"
        np.savez(outdir / f"{name}.npz", **w)
        meta = dict(m, a=m["b"], b=m["a"], mirror_of=f.stem, rot_seed=20_000 + m["seed"])
        (outdir / f"{name}.json").write_text(json.dumps(meta))
    print("mirrors written")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
