"""Behaviour checks for E1 labels: a causal mover-dependence probe and a game-play check (audit-20261108, M1/M4).

causal_probe(w, X)   For each state whose nearest mover is within 3 cells of the agent, remove that mover (its
                     channel-1 cell and its previous-frame trace) and measure how the expected step along the
                     agent -> mover direction changes: delta = step(with mover) - step(without). > 0 means the mover
                     pulls the agent toward it (approach), < 0 pushes it away (avoid), ~0 means its moves do not depend
                     on the mover (ignore). twingen.state_probe instead correlates the step with the mover's direction
                     on a state distribution, so a policy that ignores movers can still score > 0 when the states
                     put movers ahead of its path (the generic "ignore" networks: +0.03..+0.18).
causal_label(delta)  approach / avoid / ignore with the margins used for acceptance (CAUSAL_BAR, IGNORE_BAR).
play(policy, game)   Mean return per 200 steps, and for paddle games the share of ball landings returned, of a
                     policy (label function or network) in a game of obscured/train/games.py or new_games.py.
"""
import sys
from pathlib import Path

import numpy as np

RL = Path(__file__).resolve().parent  # vendored: twingen.py sits next to this file
sys.path.insert(0, str(RL))
from twingen import MOVES, forward, softmax  # noqa: E402

MV = np.array([MOVES[a] for a in range(5)], float)
CAUSAL_BAR = 0.06    # approach: delta > +0.06, avoid: delta < -0.06 (main-build networks: |delta| >= 0.15)
IGNORE_BAR = 0.04    # ignore: |delta| < 0.04 (main-build ignore networks: |delta| <= 0.03)


def _remove_nearest(X):
    """Copies of the states with the agent's nearest mover (Manhattan distance 1-3) removed, the unit direction
    agent -> mover, and the indices of the states used."""
    G = X.reshape(-1, 10, 10, 5)
    Xr, D, keep = [], [], []
    for i in range(len(X)):
        ag = np.argwhere(G[i, :, :, 0] > 0)
        ms = np.argwhere(G[i, :, :, 1] > 0)
        if len(ag) == 0 or len(ms) == 0:
            continue
        pair = np.abs(ms[:, None, :] - ag[None, :, :]).sum(2)
        j, k = np.unravel_index(int(pair.argmin()), pair.shape)
        if pair[j, k] == 0 or pair[j, k] > 3:
            continue
        m, a = ms[j], ag[k]
        g = G[i].copy()
        g[m[0], m[1], 1] = 0
        # the mover's previous-frame trace: ch4 cells next to it that belong to neither the agent nor another mover
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            r, c = m[0] + dr, m[1] + dc
            if not (0 <= r < 10 and 0 <= c < 10) or g[r, c, 4] == 0:
                continue
            if np.abs(ag - [r, c]).sum(1).min() <= 1 or g[r, c, 1] > 0:
                continue
            g[r, c, 4] = 0
        Xr.append(g.reshape(500))
        D.append((m - a) / np.linalg.norm(m - a))
        keep.append(i)
    return np.array(Xr), np.array(D), np.array(keep, int)


def _steps(policy, X):
    if isinstance(policy, dict):
        return softmax(forward(policy, X)) @ MV
    return MV[np.array([policy(x) for x in X], int)]


def causal_probe(policy, X, max_states=20000):
    """Mean change of the expected step toward the nearest mover when that mover is removed (see module doc).
    policy: a weights dict (softmax policy) or a callable obs(500,) -> action (e.g. a scripted label rule)."""
    X = np.asarray(X[:max_states], np.float64)
    Xr, D, keep = _remove_nearest(X)
    if not len(keep):
        return 0.0
    s_with = _steps(policy, X[keep])
    s_without = _steps(policy, Xr)
    return float((((s_with - s_without) * D).sum(1)).mean())


def causal_label(delta):
    if delta > CAUSAL_BAR:
        return "approach"
    if delta < -CAUSAL_BAR:
        return "avoid"
    if abs(delta) < IGNORE_BAR:
        return "ignore"
    return "unclear"


def behaves(stance, delta):
    """Acceptance test for a network labelled `stance` (approach / avoid / ignore)."""
    return causal_label(delta) == stance


def play(policy, game, steps=600, n=48, seed=0, axis=None):
    """policy: a callable obs(500,) -> action, or a weights dict (argmax of the network). Returns mean return per 200
    steps and, for pong / breakout layouts (incl. the dodge games), the share of ball landings returned."""
    sys.path.insert(0, str(RL.parents[1] / "train"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import games
    import new_games
    new_games.register()
    rng = np.random.default_rng(seed)
    env = games.GAMES[game](n, rng)
    tot, land, ended = 0.0, 0, 0
    paddle = "breakout" if "breakout" in game else "pong" if "pong" in game else None
    for _ in range(steps):
        obs = env.obs()
        if isinstance(policy, dict):
            act = forward(policy, obs.astype(np.float64)).argmax(1)
        else:
            act = np.array([policy(obs[i]) for i in range(n)])
        if paddle == "breakout":
            will = (env.br + env.dr >= 8) & (env.dr > 0)
        elif paddle == "pong":
            will = env.bc + env.dc <= 1
        r, done = env.step(act)
        if paddle:   # a landing ends the episode on a miss (pong, breakout) or on a touch (the dodge games)
            land += int(will.sum())
            ended += int((will & done).sum())
        tot += float(r.sum())
        env.reset(done)
    out = {"return_per_200_steps": round(tot / (n * steps / 200), 3)}
    if paddle:
        key = "touched_share" if "dodge" in game else "returned_share"
        out[key] = round((ended if "dodge" in game else land - ended) / max(land, 1), 3)
    return out


# (layout, stance) -> (game the behaviour should play, metric, comparison, bar). Bars sit well past a random
# policy (audit-20261108/a8_play_rules.json: random returns 24-30% of Pong/Breakout landings, touches 25-28% in the
# dodge games, crosses Freeway 0.17 times per 200 steps, touches 17.6 Car-tag cars, catches 6.2 Catch balls).
PLAY_BARS = {
    ("breakout", "approach"): ("breakout", "returned_share", ">=", 0.80),
    ("breakout", "avoid"): ("breakoutdodge", "touched_share", "<=", 0.10),
    ("pong", "approach"): ("pong", "returned_share", ">=", 0.80),
    ("pong", "avoid"): ("pongdodge", "touched_share", "<=", 0.10),
    ("freeway", "avoid"): ("freeway", "return_per_200_steps", ">=", 5.0),
    ("freeway", "approach"): ("cartag", "return_per_200_steps", ">=", 22.0),
    ("catch", "approach"): ("catch", "return_per_200_steps", ">=", 20.0),
    ("catch", "avoid"): ("dodge", "return_per_200_steps", ">=", -1.0),
}


def plays_its_game(policy, layout, stance, steps=600, n=48):
    """(ok, measurement) for layouts in PLAY_BARS; (True, None) for layouts without a bar."""
    if (layout, stance) not in PLAY_BARS:
        return True, None
    game, metric, op, bar = PLAY_BARS[(layout, stance)]
    m = play(policy, game, steps=steps, n=n)
    v = m[metric]
    return (v >= bar if op == ">=" else v <= bar), {"game": game, metric: v, "bar": f"{op} {bar}"}
