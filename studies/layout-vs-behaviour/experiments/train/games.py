"""Four 10x10 grid games in the Exp 1 observation format (our reconstructions; the post's game code is not released).

Observation: 100 cells x 5 binary channels, input i = (row*10+col)*5+channel. ch0 agent, ch1 moving objects,
ch2 reward targets, ch3 obstacles (always includes the wall border), ch4 previous frame (ch0 and ch1 cells of the
previous step). Actions: 0 no-op, 1 up, 2 down, 3 left, 4 right. Episodes end after 200 steps or on a loss.
Layouts follow the footprints of the released networks:
  pong      paddle 2 cells tall in column 1; ball bounces in the court; right edge (column 9) = reward targets
  breakout  paddle 2 cells wide in row 8; ball in rows 1-7; bricks = reward targets in rows 1-3
  pacman    symmetric 8x8 maze; agent moves along corridors; 2 ghosts; pellets = reward targets
  freeway   agent starts on row 8 and crosses 6 lanes of traffic (rows 2-7) to the goal row 1
"""
import numpy as np

N = 10


def idx(r, c, ch):
    return (r * 10 + c) * 5 + ch


def border_mask():
    m = np.zeros((N, N), bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return m


class Game:
    def __init__(self, n, rng):
        self.n, self.rng = n, rng
        self.t = np.zeros(n, int)
        self.prev = np.zeros((n, 500), np.float32)

    def frame(self):
        """(n, 100, 5) array of the current channels 0-3."""
        raise NotImplementedError

    def obs(self):
        f = self.frame()
        f[:, :, 4] = self.prev_cells
        return f.reshape(self.n, 500)

    def remember(self):
        f = self.frame()
        self.prev_cells = np.clip(f[:, :, 0] + f[:, :, 1], 0, 1)

    def step(self, action):
        self.remember()
        r, d = self._step(action)
        self.t += 1
        d = d | (self.t >= 200)
        return r.astype(np.float32), d

    def reset(self, m):
        if m.any():
            self.t[m] = 0
            self._reset(m)
            self.prev_cells[m] = 0


class Pong(Game):
    def __init__(self, n, rng):
        super().__init__(n, rng)
        self.p, self.br, self.bc = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
        self.dr, self.dc = np.zeros(n, int), np.zeros(n, int)
        self.prev_cells = np.zeros((n, 100), np.float32)
        self.reset(np.ones(n, bool))

    def _reset(self, m):
        k = int(m.sum())
        self.p[m] = self.rng.integers(1, 8, k)
        self.br[m] = self.rng.integers(1, 9, k)
        self.bc[m] = self.rng.integers(5, 9, k)
        self.dr[m] = self.rng.choice([-1, 1], k)
        self.dc[m] = -1

    def frame(self):
        f = np.zeros((self.n, 100, 5), np.float32)
        a = np.arange(self.n)
        f[a, self.p * 10 + 1, 0] = 1
        f[a, (self.p + 1) * 10 + 1, 0] = 1
        f[a, self.br * 10 + self.bc, 1] = 1
        for r in range(1, 9):
            f[:, r * 10 + 9, 2] = 1
        f[:, border_mask().ravel(), 3] = 1
        return f

    def _step(self, act):
        self.p = np.clip(self.p - (act == 1) + (act == 2), 1, 7)
        nr = self.br + self.dr
        flip = (nr < 1) | (nr > 8)
        self.dr[flip] *= -1
        nr = self.br + self.dr
        nc = self.bc + self.dc
        rew = np.zeros(self.n)
        right = nc >= 9
        rew[right] += 1
        self.dc[right] = -1
        nc[right] = self.bc[right] - 1
        left = nc <= 1
        hit = left & (nr >= self.p) & (nr <= self.p + 1)
        miss = left & ~hit
        self.dc[hit] = 1
        nc[hit] = 2
        rew[miss] -= 1
        self.br, self.bc = nr, np.clip(nc, 1, 8)
        return rew, miss


class Breakout(Game):
    def __init__(self, n, rng):
        super().__init__(n, rng)
        self.p, self.br, self.bc = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
        self.dr, self.dc = np.zeros(n, int), np.zeros(n, int)
        self.bricks = np.zeros((n, N, N), bool)
        self.prev_cells = np.zeros((n, 100), np.float32)
        self.reset(np.ones(n, bool))

    def _fill_bricks(self, m):
        self.bricks[m] = False
        self.bricks[np.ix_(np.where(m)[0], np.arange(1, 4), np.arange(1, 9))] = True

    def _reset(self, m):
        k = int(m.sum())
        self.p[m] = self.rng.integers(1, 8, k)
        self.br[m] = self.rng.integers(4, 7, k)
        self.bc[m] = self.rng.integers(1, 9, k)
        self.dr[m] = 1
        self.dc[m] = self.rng.choice([-1, 1], k)
        self._fill_bricks(m)

    def frame(self):
        f = np.zeros((self.n, 100, 5), np.float32)
        a = np.arange(self.n)
        f[a, 80 + self.p, 0] = 1
        f[a, 80 + self.p + 1, 0] = 1
        f[a, self.br * 10 + self.bc, 1] = 1
        f[:, :, 2] = self.bricks.reshape(self.n, 100)
        f[:, border_mask().ravel(), 3] = 1
        return f

    def _step(self, act):
        a = np.arange(self.n)
        self.p = np.clip(self.p - (act == 3) + (act == 4), 1, 7)
        nc = self.bc + self.dc
        side = (nc < 1) | (nc > 8)
        self.dc[side] *= -1
        nc = self.bc + self.dc
        nr = self.br + self.dr
        top = nr < 1
        self.dr[top] = 1
        nr[top] = 1
        rew = np.zeros(self.n)
        hitb = self.bricks[a, np.clip(nr, 0, 9), np.clip(nc, 0, 9)] & (nr <= 3)
        rew[hitb] += 1
        self.bricks[a[hitb], nr[hitb], nc[hitb]] = False
        self.dr[hitb] *= -1
        nr[hitb] = self.br[hitb]
        low = nr >= 8
        caught = low & (nc >= self.p) & (nc <= self.p + 1)
        miss = low & ~caught
        self.dr[caught] = -1
        nr[caught] = 7
        rew[miss] -= 1
        self.br, self.bc = np.clip(nr, 1, 7), nc
        empty = ~self.bricks.reshape(self.n, -1).any(1)
        if empty.any():
            self._fill_bricks(empty)
        return rew, miss


MAZE_WALLS = {2: (2, 3, 6, 7), 3: (2, 7), 4: (4, 5), 5: (4, 5), 6: (2, 7), 7: (2, 3, 6, 7)}


class PacMan(Game):
    def __init__(self, n, rng):
        super().__init__(n, rng)
        wall = border_mask()
        for r, cs in MAZE_WALLS.items():
            for c in cs:
                wall[r, c] = True
        self.wall = wall
        self.open = [(r, c) for r in range(N) for c in range(N) if not wall[r, c]]
        self.ar, self.ac = np.zeros(n, int), np.zeros(n, int)
        self.gr, self.gc = np.zeros((n, 2), int), np.zeros((n, 2), int)
        self.pellets = np.zeros((n, N, N), bool)
        self.prev_cells = np.zeros((n, 100), np.float32)
        self.reset(np.ones(n, bool))

    def _reset(self, m):
        for k in np.where(m)[0]:
            cells = self.rng.choice(len(self.open), 3, replace=False)
            (self.ar[k], self.ac[k]), g1, g2 = (self.open[j] for j in cells)
            self.gr[k], self.gc[k] = (g1[0], g2[0]), (g1[1], g2[1])
            self.pellets[k] = ~self.wall
            self.pellets[k, self.ar[k], self.ac[k]] = False

    def frame(self):
        f = np.zeros((self.n, 100, 5), np.float32)
        a = np.arange(self.n)
        f[a, self.ar * 10 + self.ac, 0] = 1
        for g in range(2):
            f[a, self.gr[:, g] * 10 + self.gc[:, g], 1] = 1
        f[:, :, 2] = self.pellets.reshape(self.n, 100)
        f[:, self.wall.ravel(), 3] = 1
        return f

    def _move(self, r, c, act):
        dr = np.select([act == 1, act == 2], [-1, 1], 0)
        dc = np.select([act == 3, act == 4], [-1, 1], 0)
        nr, nc = r + dr, c + dc
        blocked = self.wall[nr, nc]
        return np.where(blocked, r, nr), np.where(blocked, c, nc)

    def _step(self, act):
        a = np.arange(self.n)
        self.ar, self.ac = self._move(self.ar, self.ac, act)
        rew = np.zeros(self.n)
        eat = self.pellets[a, self.ar, self.ac]
        rew[eat] += 1
        self.pellets[a[eat], self.ar[eat], self.ac[eat]] = False
        for g in range(2):
            ga = self.rng.integers(1, 5, self.n)
            chase = self.rng.random(self.n) < 0.5
            toward = np.where(np.abs(self.ar - self.gr[:, g]) >= np.abs(self.ac - self.gc[:, g]),
                              np.where(self.ar < self.gr[:, g], 1, 2), np.where(self.ac < self.gc[:, g], 3, 4))
            self.gr[:, g], self.gc[:, g] = self._move(self.gr[:, g], self.gc[:, g], np.where(chase, toward, ga))
        caught = ((self.gr[:, 0] == self.ar) & (self.gc[:, 0] == self.ac)) | (
            (self.gr[:, 1] == self.ar) & (self.gc[:, 1] == self.ac))
        rew[caught] -= 1
        empty = ~self.pellets.reshape(self.n, -1).any(1)
        for k in np.where(empty)[0]:
            self.pellets[k] = ~self.wall
        return rew, caught


class Freeway(Game):
    def __init__(self, n, rng):
        super().__init__(n, rng)
        self.ar, self.ac = np.zeros(n, int), np.zeros(n, int)
        self.lane_dir = np.array([1, -1, 1, -1, 1, -1])
        self.lane_period = np.array([1, 2, 1, 2, 1, 2])
        self.cars = np.zeros((n, 6), int)
        self.prev_cells = np.zeros((n, 100), np.float32)
        self.reset(np.ones(n, bool))

    def _reset(self, m):
        k = int(m.sum())
        self.ar[m] = 8
        self.ac[m] = self.rng.integers(1, 9, k)
        self.cars[m] = self.rng.integers(1, 9, (k, 6))

    def frame(self):
        f = np.zeros((self.n, 100, 5), np.float32)
        a = np.arange(self.n)
        f[a, self.ar * 10 + self.ac, 0] = 1
        for L in range(6):
            f[a, (L + 2) * 10 + self.cars[:, L], 1] = 1
        for c in range(1, 9):
            f[:, 10 + c, 2] = 1
        f[:, border_mask().ravel(), 3] = 1
        return f

    def _step(self, act):
        self.ar = np.clip(self.ar - (act == 1) + (act == 2), 1, 8)
        self.ac = np.clip(self.ac - (act == 3) + (act == 4), 1, 8)
        move = (self.t[:, None] % self.lane_period[None, :]) == 0
        self.cars = np.where(move, (self.cars - 1 + self.lane_dir[None, :]) % 8 + 1, self.cars)
        rew = np.zeros(self.n)
        lane = self.ar - 2
        in_lane = (lane >= 0) & (lane < 6)
        hit = in_lane & (self.cars[np.arange(self.n), np.clip(lane, 0, 5)] == self.ac)
        self.ar[hit] = 8
        goal = self.ar == 1
        rew[goal] += 1
        self.ar[goal] = 8
        return rew, np.zeros(self.n, bool)


class FallingObjects(Game):
    """Agent: one cell on row 8, moves left/right. Two objects fall one row per step from row 1 (staggered) and respawn
    at row 1 in a random column after reaching row 8. CONTACT is the reward event (sign set by the subclass)."""
    CONTACT = 0.0
    CONTACT_ENDS = False

    def __init__(self, n, rng):
        super().__init__(n, rng)
        self.ac = np.zeros(n, int)
        self.orow, self.ocol = np.zeros((n, 2), int), np.zeros((n, 2), int)
        self.prev_cells = np.zeros((n, 100), np.float32)
        self.reset(np.ones(n, bool))

    def _reset(self, m):
        k = int(m.sum())
        self.ac[m] = self.rng.integers(1, 9, k)
        self.orow[m] = np.array([1, 5])[None, :]
        self.ocol[m] = self.rng.integers(1, 9, (k, 2))

    def frame(self):
        f = np.zeros((self.n, 100, 5), np.float32)
        a = np.arange(self.n)
        f[a, 80 + self.ac, 0] = 1
        for o in range(2):
            f[a, self.orow[:, o] * 10 + self.ocol[:, o], 1] = 1
        f[:, border_mask().ravel(), 3] = 1
        return f

    def _step(self, act):
        self.ac = np.clip(self.ac - (act == 3) + (act == 4), 1, 8)
        rew, done = np.zeros(self.n), np.zeros(self.n, bool)
        for o in range(2):
            land = self.orow[:, o] == 8
            hit = land & (self.ocol[:, o] == self.ac)
            rew[hit] += self.CONTACT
            if self.CONTACT_ENDS:
                done |= hit
            self.ocol[land, o] = self.rng.integers(1, 9, int(land.sum()))
            self.orow[:, o] = np.where(land, 1, self.orow[:, o] + 1)
        return rew, done


class Catch(FallingObjects):
    """Catch the falling objects: +1 when one lands on the agent."""
    CONTACT = 1.0


class Dodge(FallingObjects):
    """Dodge the falling objects: -1 and the episode ends when one lands on the agent."""
    CONTACT = -1.0
    CONTACT_ENDS = True


class Hunter(PacMan):
    """Pac-Man's maze and pellets (+1), but the two 'ghosts' are prey: they flee from the agent half the time (random
    otherwise), touching one gives +3 and it reappears on a random open cell; nothing ends the episode early."""

    def _step(self, act):
        a = np.arange(self.n)
        self.ar, self.ac = self._move(self.ar, self.ac, act)
        rew = np.zeros(self.n)
        eat = self.pellets[a, self.ar, self.ac]
        rew[eat] += 1
        self.pellets[a[eat], self.ar[eat], self.ac[eat]] = False
        for g in range(2):
            hit = (self.gr[:, g] == self.ar) & (self.gc[:, g] == self.ac)
            ga = self.rng.integers(1, 5, self.n)
            flee = self.rng.random(self.n) < 0.5
            away = np.where(np.abs(self.ar - self.gr[:, g]) >= np.abs(self.ac - self.gc[:, g]),
                            np.where(self.ar < self.gr[:, g], 2, 1), np.where(self.ac < self.gc[:, g], 4, 3))
            self.gr[:, g], self.gc[:, g] = self._move(self.gr[:, g], self.gc[:, g], np.where(flee, away, ga))
            hit |= (self.gr[:, g] == self.ar) & (self.gc[:, g] == self.ac)
            rew[hit] += 3
            for k in np.where(hit)[0]:
                cell = self.open[self.rng.integers(len(self.open))]
                self.gr[k, g], self.gc[k, g] = cell
        empty = ~self.pellets.reshape(self.n, -1).any(1)
        for k in np.where(empty)[0]:
            self.pellets[k] = ~self.wall
        return rew, np.zeros(self.n, bool)


class CarTag(Freeway):
    """Freeway's road, cars and displayed goal row, but touching a car gives +1 (the car jumps to a random column of its
    lane) and the goal row gives nothing; the agent is never reset."""

    def _step(self, act):
        self.ar = np.clip(self.ar - (act == 1) + (act == 2), 1, 8)
        self.ac = np.clip(self.ac - (act == 3) + (act == 4), 1, 8)
        move = (self.t[:, None] % self.lane_period[None, :]) == 0
        self.cars = np.where(move, (self.cars - 1 + self.lane_dir[None, :]) % 8 + 1, self.cars)
        rew = np.zeros(self.n)
        lane = self.ar - 2
        in_lane = (lane >= 0) & (lane < 6)
        a = np.arange(self.n)
        hit = in_lane & (self.cars[a, np.clip(lane, 0, 5)] == self.ac)
        rew[hit] += 1
        for k in np.where(hit)[0]:
            self.cars[k, lane[k]] = self.rng.integers(1, 9)
        return rew, np.zeros(self.n, bool)


GAMES = {"pong": Pong, "breakout": Breakout, "pacman": PacMan, "freeway": Freeway, "catch": Catch, "dodge": Dodge,
         "hunter": Hunter, "cartag": CarTag}
