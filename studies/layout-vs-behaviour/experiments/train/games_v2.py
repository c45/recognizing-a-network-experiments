"""Second versions of three twin-study games (November 10 behaviour audit, ../twin_audit_20261110/AUDIT.md).

The first versions in games.py are unchanged; importing this module adds the new ones to games.GAMES. Observation
format, board layouts and mover dynamics are those of the first versions. What changes is the reward rule, so that the
trained policy's moves have to depend on the moving objects:

  cartag_v2    A car is tagged (+1) only when the agent's own move takes it onto the cell the car occupies, as shown,
               before the cars move. Standing still never scores. The goal row is shown but the agent cannot enter
               it. (v1 scored whenever agent and car shared a cell after both moved, so cars driving into the agent
               paid; its agent could also stand on the goal row, which Freeway's never does, so 16 inputs were used
               in Car-tag and never in Freeway.)
  cartag_v2a   cartag_v2 with an open goal row (first draft; the 16-input difference from Freeway remains).
  cartag_v2r   cartag_v2 where reaching the goal row sends the agent back to the start row, as in Freeway (second
               draft; one seed then walks to the goal row 2.5 times per 200 steps, which looks like crossing).
  hunter_v2    Catching a prey gives +1. Pellets are shown and disappear when walked over, as in Pac-Man, but give
               nothing. (v1 paid +1 per pellet and +3 per prey; a fixed pellet route that never looks at the prey
               was optimal.)
  freeway_v2   A collision ends the episode, as Pac-Man's ghosts and Dodge's objects do. (v1 only sent the agent back
               to the start row, so walking straight up was near-optimal.)
  freeway_v2p  As freeway_v2 with -1 for the collision. Not used: in a 6M-step pilot the policy never left the start
               row (random play scores about -12 per 200 steps, staying put scores 0).
"""
import numpy as np

import games
from games import Freeway, PacMan


class CarTagV2(Freeway):
    GOAL = "block"      # "block": the agent cannot enter the goal row; "open": it may stand on it; "return":
                        # reaching it sends the agent to the start row

    def _dest(self, a):
        nr = np.clip(self.ar - (a == 1) + (a == 2), 2 if self.GOAL == "block" else 1, 8)
        nc = np.clip(self.ac - (a == 3) + (a == 4), 1, 8)
        return nr, nc

    def contact_actions(self):
        """(n, 5) bool: action a tags a car this step."""
        idx = np.arange(self.n)
        H = np.zeros((self.n, 5), bool)
        for a in range(1, 5):
            nr, nc = self._dest(a)
            lane = nr - 2
            H[:, a] = (((nr != self.ar) | (nc != self.ac)) & (lane >= 0) & (lane < 6)
                       & (self.cars[idx, np.clip(lane, 0, 5)] == nc))
        return H

    def _step(self, act):
        idx = np.arange(self.n)
        hit = self.contact_actions()[idx, act]
        self.ar, self.ac = self._dest(act)
        k = int(hit.sum())
        if k:
            self.cars[idx[hit], self.ar[hit] - 2] = self.rng.integers(1, 9, k)
        move = (self.t[:, None] % self.lane_period[None, :]) == 0
        self.cars = np.where(move, (self.cars - 1 + self.lane_dir[None, :]) % 8 + 1, self.cars)
        if self.GOAL == "return":
            self.ar[self.ar == 1] = 8
        return hit.astype(float), np.zeros(self.n, bool)


class CarTagV2A(CarTagV2):
    GOAL = "open"


class CarTagV2R(CarTagV2):
    GOAL = "return"


class FreewayV2(Freeway):
    HIT = 0.0

    def _step(self, act):
        self.ar = np.clip(self.ar - (act == 1) + (act == 2), 1, 8)
        self.ac = np.clip(self.ac - (act == 3) + (act == 4), 1, 8)
        move = (self.t[:, None] % self.lane_period[None, :]) == 0
        self.cars = np.where(move, (self.cars - 1 + self.lane_dir[None, :]) % 8 + 1, self.cars)
        rew = np.zeros(self.n)
        lane = self.ar - 2
        hit = (lane >= 0) & (lane < 6) & (self.cars[np.arange(self.n), np.clip(lane, 0, 5)] == self.ac)
        rew[hit] += self.HIT
        goal = self.ar == 1
        rew[goal] += 1
        self.ar[goal] = 8
        return rew, hit


class FreewayV2P(FreewayV2):
    HIT = -1.0


class HunterV2(PacMan):
    def __init__(self, n, rng):
        self._open = None
        super().__init__(n, rng)

    def _step(self, act):
        if self._open is None:
            self._open = np.array(self.open)
        idx = np.arange(self.n)
        self.ar, self.ac = self._move(self.ar, self.ac, act)
        rew = np.zeros(self.n)
        self.pellets[idx, self.ar, self.ac] = False
        for g in range(2):
            hit = (self.gr[:, g] == self.ar) & (self.gc[:, g] == self.ac)
            ga = self.rng.integers(1, 5, self.n)
            flee = self.rng.random(self.n) < 0.5
            away = np.where(np.abs(self.ar - self.gr[:, g]) >= np.abs(self.ac - self.gc[:, g]),
                            np.where(self.ar < self.gr[:, g], 2, 1), np.where(self.ac < self.gc[:, g], 4, 3))
            self.gr[:, g], self.gc[:, g] = self._move(self.gr[:, g], self.gc[:, g], np.where(flee, away, ga))
            hit |= (self.gr[:, g] == self.ar) & (self.gc[:, g] == self.ac)
            rew[hit] += 1
            k = int(hit.sum())
            if k:
                cells = self._open[self.rng.integers(len(self._open), size=k)]
                self.gr[hit, g], self.gc[hit, g] = cells[:, 0], cells[:, 1]
        empty = ~self.pellets.reshape(self.n, -1).any(1)
        for k in np.where(empty)[0]:
            self.pellets[k] = ~self.wall
        return rew, np.zeros(self.n, bool)


V2 = {"cartag_v2": CarTagV2, "cartag_v2a": CarTagV2A, "cartag_v2r": CarTagV2R, "freeway_v2": FreewayV2, "freeway_v2p": FreewayV2P, "hunter_v2": HunterV2}
games.GAMES.update(V2)
