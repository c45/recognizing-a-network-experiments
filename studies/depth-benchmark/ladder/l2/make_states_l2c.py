"""Shared mirror-symmetric Breakout states for ladder L2c (rule depends on the ball's horizontal motion).

Usage: python3 -I make_states_l2c.py <out dir> [--n 165000]
Same behaviour mix as pilot F1. Keeps states after an episode's first step whose ball column changed since the
previous frame. right = 1 when the ball moved right. Every kept state is stored with its left-right mirror (columns
reversed in every channel; paddle p -> 8 - p; ball column c -> 9 - c; right -> 1 - right). Pairs stay together in the
train / held-out split. --balance-side (L2d): before mirroring, equal counts in the four direction x ball-side
cells (37% of states each pair) plus no-op states (13%) split equally by direction. Writes states.npz (obs, p, c,
right, split).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_f1"))
from make_states import Breakout, away, toward  # noqa: E402


def mirror(obs):
    return obs.reshape(-1, 10, 10, 5)[:, :, ::-1, :].reshape(-1, 500)


def main(out, n, balance_side=False):
    rng = np.random.default_rng(1107)
    env = Breakout(256, rng)
    prev_c = env.bc.copy()
    fresh = np.ones(256, bool)
    obs, ps, cs, rs = [], [], [], []
    total = 0
    while total < (4 if balance_side else 1) * n:
        o, p, c = env.obs(), env.p.copy(), env.bc.copy()
        keep = ~fresh & (c != prev_c)
        obs.append(o[keep].astype(np.uint8))
        ps.append(p[keep])
        cs.append(c[keep])
        rs.append((c[keep] > prev_c[keep]).astype(np.int8))
        total += int(keep.sum())
        u = rng.random(len(p))
        act = np.where(u < 0.4, toward(p, c), np.where(u < 0.8, away(p, c), rng.integers(0, 5, len(p))))
        prev_c = env.bc.copy()
        _, done = env.step(act)
        fresh = done.copy()
        env.reset(done)
        prev_c[done] = env.bc[done]
    obs, p, c, r = (np.concatenate(a) for a in (obs, ps, cs, rs))
    if balance_side:
        side = np.where(c < p, 0, np.where(c > p + 1, 1, 2))
        q = int(n * 0.37)
        noop = int(n * 0.13)
        pick = []
        for d in (0, 1):
            for s_ in (0, 1):
                idx_ = np.where((r == d) & (side == s_))[0]
                pick.append(rng.choice(idx_, q // 2, replace=False))
            idx_ = np.where((r == d) & (side == 2))[0]
            pick.append(rng.choice(idx_, noop // 2, replace=False))
        pick = np.concatenate(pick)
        pick = pick[rng.permutation(len(pick))]
        obs, p, c, r = obs[pick], p[pick], c[pick], r[pick]
        n = len(pick)
    else:
        obs, p, c, r = obs[:n], p[:n], c[:n], r[:n]
    pair_test = np.zeros(n, bool)
    pair_test[rng.permutation(n)[:n // 11]] = True
    obs2 = np.concatenate([obs, mirror(obs)])
    p2, c2, r2 = np.concatenate([p, 8 - p]), np.concatenate([c, 9 - c]), np.concatenate([r, 1 - r])
    split = np.concatenate([pair_test, pair_test]).astype(np.int8)
    m = mirror(obs2[:3])
    assert (mirror(m) == obs2[:3]).all()
    Path(out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(out) / "states.npz", obs=obs2, p=p2, c=c2, right=r2, split=split)
    t = toward(p2, c2)
    side2 = np.where(c2 < p2, "L", np.where(c2 > p2 + 1, "R", "0"))
    cells = {f"{d}{s_}": int(((r2 == d) & (side2 == s_)).sum()) for d in (0, 1) for s_ in "LR0"}
    print(f"{len(obs2):,} states; moving-right share {r2.mean():.3f}; direction x side {cells}; "
          f"toward-label counts {np.bincount(t, minlength=5).tolist()}; "
          f"inputs ever on {int((obs2.max(0) > 0).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=165_000)
    ap.add_argument("--balance-side", action="store_true", help="L2d: balance direction x ball side before mirroring")
    a = ap.parse_args()
    main(a.out, a.n, a.balance_side)
