"""Shared Breakout-grid states for ladder L2 (direction-dependent rule), with the ball's observed vertical motion.

Usage: python3 -I make_states_l2.py <out dir> [--n 330000]
Same behaviour mix as pilot F1 (40% toward, 40% away, 20% random). Keeps only states after the first step of an
episode whose ball row changed since the previous frame, so the motion is visible from channels 1 and 4.
fall = 1 when the ball moved down (row increased), 0 when it moved up. States are balanced to 50% falling by
subsampling the majority direction. --per-row (L2b): only ball rows 5 and 6, the rows where both directions are
common in this environment, each balanced to 50% falling. Writes states.npz (obs, p, c, fall, split).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_f1"))
from make_states import Breakout, away, toward  # noqa: E402


def main(out, n, per_row=False):
    rng = np.random.default_rng(1106)
    env = Breakout(256, rng)
    prev_r = env.br.copy()
    fresh = np.ones(256, bool)
    obs, ps, cs, fs, rs = [], [], [], [], []
    total = 0
    while total < (8 if per_row else 3) * n:
        o, p, c, r = env.obs(), env.p.copy(), env.bc.copy(), env.br.copy()
        keep = ~fresh & (r != prev_r)
        obs.append(o[keep].astype(np.uint8))
        ps.append(p[keep])
        cs.append(c[keep])
        fs.append((r[keep] > prev_r[keep]).astype(np.int8))
        rs.append(r[keep])
        total += int(keep.sum())
        u = rng.random(len(p))
        act = np.where(u < 0.4, toward(p, c), np.where(u < 0.8, away(p, c), rng.integers(0, 5, len(p))))
        prev_r = env.br.copy()
        _, done = env.step(act)
        fresh = done.copy()
        env.reset(done)
        prev_r[done] = env.br[done]
    obs, p, c, f, rows = (np.concatenate(a) for a in (obs, ps, cs, fs, rs))
    if per_row:
        cells = [(np.where((rows == r) & (f == 0))[0], np.where((rows == r) & (f == 1))[0]) for r in (5, 6)]
        q = n // 4
        assert all(min(len(u), len(d)) >= q for u, d in cells), [(len(u), len(d)) for u, d in cells]
        pick = np.concatenate([np.concatenate([rng.choice(u, q, replace=False), rng.choice(d, q, replace=False)])
                               for u, d in cells])
        n = len(pick)
    else:
        up, down = np.where(f == 0)[0], np.where(f == 1)[0]
        pick = np.concatenate([rng.choice(up, n // 2, replace=False), rng.choice(down, n - n // 2, replace=False)])
    pick = pick[rng.permutation(len(pick))]
    obs, p, c, f = obs[pick], p[pick], c[pick], f[pick]
    split = np.zeros(n, np.int8)
    split[rng.permutation(n)[:n - 300_000]] = 1
    Path(out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(out) / "states.npz", obs=obs, p=p, c=c, fall=f, split=split)
    rows = rows[pick] if per_row else None
    by_row = {int(r): round(float(f[rows == r].mean()), 3) for r in (5, 6)} if per_row else None
    print(f"{n:,} states; falling share {f.mean():.3f}; by row {by_row}; inputs ever on {int((obs.max(0) > 0).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=330_000)
    ap.add_argument("--per-row", action="store_true", help="balance falling/rising within each ball row (L2b)")
    a = ap.parse_args()
    main(a.out, a.n, a.per_row)
