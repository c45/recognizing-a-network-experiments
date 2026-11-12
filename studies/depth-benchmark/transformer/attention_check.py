"""Mechanism check for T2 networks: layer-1 previous-token attention and layer-2 routing by query parity.

Usage: python3 -I attention_check.py <nets dir> <out json>
Runs each included network on 4,000 held-out sequences (analysis only; readers never get forward passes) and reports
mean layer-1 attention to the previous position and layer-2 attention from the query to positions j+1 and j,
separately for queries the rule routes to NEXT and to PREV.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def attn(w, x):
    h = w["E"][x] + w["P"]
    mask = np.tril(np.ones((10, 10), bool))
    pats = []
    for l in (1, 2):
        s = (h @ w[f"Q{l}"]) @ (h @ w[f"K{l}"]).transpose(0, 2, 1) / 4
        s = np.where(mask, s, -1e9)
        a = np.exp(s - s.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        pats.append(a)
        h = h + a @ (h @ w[f"V{l}"]) @ w[f"O{l}"]
    return pats


def main(nets, out):
    z = np.load(HERE / "data" / "seqs.npz")
    te = z["split"] == 1
    x, j = z["x"][te][:4000], z["j"][te][:4000]
    n = np.arange(len(x))
    rows = []
    for f in sorted(Path(nets).glob("*.npz")):
        meta = json.loads(f.with_suffix(".json").read_text())
        if meta["heldout_acc"] < 0.95:
            continue
        a1, a2 = attn(dict(np.load(f)), x)
        even = x[:, 9] % 2 == 0
        nexty = even if meta["rule"] == "evennext" else ~even
        rows.append({"net": f.stem, "prev_token_l1": float(np.mean([a1[:, t, t - 1].mean() for t in range(1, 10)])),
                     "next_routed_to_j1": float(a2[n, 9, j + 1][nexty].mean()), "next_routed_to_j": float(a2[n, 9, j][nexty].mean()),
                     "prev_routed_to_j1": float(a2[n, 9, j + 1][~nexty].mean()), "prev_routed_to_j": float(a2[n, 9, j][~nexty].mean())})
    r = {k: [row[k] for row in rows] for k in rows[0] if k != "net"}
    summary = {"n": len(rows), "prev_token_l1_mean": round(float(np.mean(r["prev_token_l1"])), 2),
               "prev_token_l1_min": round(float(np.min(r["prev_token_l1"])), 2),
               "next_routed_to_j1": round(float(np.mean(r["next_routed_to_j1"])), 2),
               "prev_routed_to_j": round(float(np.mean(r["prev_routed_to_j"])), 2),
               "n_follow_routing": int(sum(a > b and c > d for a, b, c, d in zip(r["next_routed_to_j1"], r["next_routed_to_j"],
                                                                               r["prev_routed_to_j"], r["prev_routed_to_j1"])))}
    Path(out).write_text(json.dumps({"summary": summary, "networks": rows}, indent=1))
    print(summary)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
