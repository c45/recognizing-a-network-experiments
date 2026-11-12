"""Five new 10x10 grid games in the Exp 1 observation format, for separating reading from recognition.

Same channels and actions as obscured/train/games.py (ch0 agent, ch1 moving objects, ch2 reward targets, ch3
obstacles incl. the wall border, ch4 previous frame; actions 0 no-op, 1 up, 2 down, 3 left, 4 right; 200-step limit).

Matched pair (same footprint, opposite rules): the agent is one cell in row 8 and moves left/right; one object (ch1)
falls one row per step from a random column of row 1. When the object is in row 8:
  catch   +1 if it is in the agent's column, then a new object; otherwise a new object, no reward
  dodge   -1 and the episode ends if it is in the agent's column; otherwise +1 and a new object
Unnamed games (no well-known name):
  shepherd     the agent moves in four directions; a sheep (ch1) wanders, and steps directly away from the agent when
               the agent is next to it; +1 when the sheep enters the 3x3 pen (ch2) in the bottom-right corner, then a
               new sheep appears outside the pen
  escort       the agent moves in four directions; a wanderer (ch1) random-walks; +0.1 on every step the agent is
               next to it (including diagonally)
  lamplighter  the agent is one cell in row 8 and moves left/right; one unlit lamp (ch2) sits in a random column of
               row 1; pressing up lights it if the agent is directly below it (+1, a new lamp appears), otherwise -0.05
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ff_paths import TRAIN  # noqa: E402
sys.path.insert(0, str(TRAIN))
from games import GAMES, Game, border_mask  # noqa: E402

BORDER = border_mask().ravel()


class Fall(Game):
    CATCH = True

    def __init__(self, n, rng):
        super().__init__(n, rng)
        self.ac, self.orow, self.ocol = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
        self.prev_cells = np.zeros((n, 100), np.float32)
        self.reset(np.ones(n, bool))

    def _spawn(self, m):
        self.orow[m] = 1
        self.ocol[m] = self.rng.integers(1, 9, int(m.sum()))

    def _reset(self, m):
        self.ac[m] = self.rng.integers(1, 9, int(m.sum()))
        self._spawn(m)

    def frame(self):
        f = np.zeros((self.n, 100, 5), np.float32)
        a = np.arange(self.n)
        f[a, 80 + self.ac, 0] = 1
        f[a, self.orow * 10 + self.ocol, 1] = 1
        f[:, BORDER, 3] = 1
        return f

    def _step(self, act):
        self.ac = np.clip(self.ac - (act == 3) + (act == 4), 1, 8)
        rew, done = np.zeros(self.n), np.zeros(self.n, bool)
        land = self.orow == 8
        hit = land & (self.ocol == self.ac)
        if self.CATCH:
            rew[hit] = 1
        else:
            rew[hit] = -1
            done = hit
            rew[land & ~hit] = 1
        self.orow[~land] += 1
        self._spawn(land)
        return rew, done


class Catch(Fall):
    CATCH = True


class Dodge(Fall):
    CATCH = False


def move4(r, c, act):
    r = np.clip(r - (act == 1) + (act == 2), 1, 8)
    c = np.clip(c - (act == 3) + (act == 4), 1, 8)
    return r, c


class Shepherd(Game):
    PEN = (6, 8)

    def __init__(self, n, rng):
        super().__init__(n, rng)
        self.ar, self.ac, self.sr, self.sc = (np.zeros(n, int) for _ in range(4))
        self.prev_cells = np.zeros((n, 100), np.float32)
        self.reset(np.ones(n, bool))

    def _in_pen(self, r, c):
        return (r >= self.PEN[0]) & (r <= self.PEN[1]) & (c >= self.PEN[0]) & (c <= self.PEN[1])

    def _spawn_sheep(self, m):
        for k in np.where(m)[0]:
            while True:
                r, c = self.rng.integers(1, 9, 2)
                if not self._in_pen(r, c) and not (r == self.ar[k] and c == self.ac[k]):
                    self.sr[k], self.sc[k] = r, c
                    break

    def _reset(self, m):
        k = int(m.sum())
        self.ar[m], self.ac[m] = self.rng.integers(1, 9, k), self.rng.integers(1, 9, k)
        self._spawn_sheep(m)

    def frame(self):
        f = np.zeros((self.n, 100, 5), np.float32)
        a = np.arange(self.n)
        f[a, self.ar * 10 + self.ac, 0] = 1
        f[a, self.sr * 10 + self.sc, 1] = 1
        for r in range(self.PEN[0], self.PEN[1] + 1):
            for c in range(self.PEN[0], self.PEN[1] + 1):
                f[:, r * 10 + c, 2] = 1
        f[:, BORDER, 3] = 1
        return f

    def _step(self, act):
        nr, nc = move4(self.ar, self.ac, act)
        blocked = (nr == self.sr) & (nc == self.sc)
        self.ar, self.ac = np.where(blocked, self.ar, nr), np.where(blocked, self.ac, nc)
        near = (np.abs(self.sr - self.ar) <= 1) & (np.abs(self.sc - self.ac) <= 1)
        flee_r = np.clip(self.sr + np.sign(self.sr - self.ar), 1, 8)
        flee_c = np.clip(self.sc + np.sign(self.sc - self.ac), 1, 8)
        wander = self.rng.random(self.n) < 0.3
        wa = self.rng.integers(1, 5, self.n)
        wr, wc = move4(self.sr, self.sc, wa)
        tr = np.where(near, flee_r, np.where(wander, wr, self.sr))
        tc = np.where(near, flee_c, np.where(wander, wc, self.sc))
        ok = ~((tr == self.ar) & (tc == self.ac))
        self.sr, self.sc = np.where(ok, tr, self.sr), np.where(ok, tc, self.sc)
        penned = self._in_pen(self.sr, self.sc)
        rew = penned.astype(float)
        self._spawn_sheep(penned)
        return rew, np.zeros(self.n, bool)


class Escort(Game):
    def __init__(self, n, rng):
        super().__init__(n, rng)
        self.ar, self.ac, self.wr, self.wc = (np.zeros(n, int) for _ in range(4))
        self.prev_cells = np.zeros((n, 100), np.float32)
        self.reset(np.ones(n, bool))

    def _reset(self, m):
        k = int(m.sum())
        self.ar[m], self.ac[m] = self.rng.integers(1, 9, k), self.rng.integers(1, 9, k)
        self.wr[m], self.wc[m] = self.rng.integers(1, 9, k), self.rng.integers(1, 9, k)
        same = m & (self.wr == self.ar) & (self.wc == self.ac)
        self.wc[same] = np.where(self.wc[same] < 8, self.wc[same] + 1, self.wc[same] - 1)

    def frame(self):
        f = np.zeros((self.n, 100, 5), np.float32)
        a = np.arange(self.n)
        f[a, self.ar * 10 + self.ac, 0] = 1
        f[a, self.wr * 10 + self.wc, 1] = 1
        f[:, BORDER, 3] = 1
        return f

    def _step(self, act):
        nr, nc = move4(self.ar, self.ac, act)
        blocked = (nr == self.wr) & (nc == self.wc)
        self.ar, self.ac = np.where(blocked, self.ar, nr), np.where(blocked, self.ac, nc)
        wa = np.where(self.rng.random(self.n) < 0.2, 0, self.rng.integers(1, 5, self.n))
        tr, tc = move4(self.wr, self.wc, wa)
        ok = ~((tr == self.ar) & (tc == self.ac))
        self.wr, self.wc = np.where(ok, tr, self.wr), np.where(ok, tc, self.wc)
        near = (np.abs(self.wr - self.ar) <= 1) & (np.abs(self.wc - self.ac) <= 1)
        return 0.1 * near, np.zeros(self.n, bool)


class Lamplighter(Game):
    def __init__(self, n, rng):
        super().__init__(n, rng)
        self.ac, self.lc = np.zeros(n, int), np.zeros(n, int)
        self.prev_cells = np.zeros((n, 100), np.float32)
        self.reset(np.ones(n, bool))

    def _reset(self, m):
        k = int(m.sum())
        self.ac[m], self.lc[m] = self.rng.integers(1, 9, k), self.rng.integers(1, 9, k)

    def frame(self):
        f = np.zeros((self.n, 100, 5), np.float32)
        a = np.arange(self.n)
        f[a, 80 + self.ac, 0] = 1
        f[a, 10 + self.lc, 2] = 1
        f[:, BORDER, 3] = 1
        return f

    def _step(self, act):
        self.ac = np.clip(self.ac - (act == 3) + (act == 4), 1, 8)
        press = act == 1
        lit = press & (self.lc == self.ac)
        rew = np.where(lit, 1.0, np.where(press, -0.05, 0.0))
        k = int(lit.sum())
        if k:
            new = self.rng.integers(1, 8, k)
            self.lc[lit] = np.where(new >= self.lc[lit], new + 1, new)
        return rew, np.zeros(self.n, bool)


NEW = {"catch": Catch, "dodge": Dodge, "shepherd": Shepherd, "escort": Escort, "lamplighter": Lamplighter}
ALL = {**GAMES, **NEW}
