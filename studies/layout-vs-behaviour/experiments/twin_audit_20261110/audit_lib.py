"""Play-level behaviour measurements for the twin-study policies (obscured/train/g_<game>[_s2]).

run(game, policy, obs_mode) plays a policy in a game of train/games.py and counts what happens:
  return per 200 steps, contacts with moving objects, goal-row arrivals (road), pellets (maze),
  the expected step toward the nearest mover within 3 cells (twingen.state_probe's quantity, on the true board),
  and a contact index: over states where some but not all actions put the agent on a mover this step, the
  probability the policy gives to those actions, beside the share of such actions (what a uniform policy gives).
obs_mode: "true" the real observation; "decoy" the movers (and their previous-frame trace) are replaced by those of
  another parallel game, so the policy sees plausible movers that carry no information; "blind" movers are deleted.
A policy is a weights dict (sampled from the softmax, as in training) or a function (env, rng) -> actions.
"""
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
import games  # noqa: E402
import games_v2  # noqa: E402,F401  (adds the v2 games to games.GAMES)
from twingen import MOVES, forward, softmax  # noqa: E402

MV = np.array([MOVES[a] for a in range(5)], float)
KIND = {"catch": "falling", "dodge": "falling", "pacman": "maze", "hunter": "maze", "freeway": "road",
        "cartag": "road"}


def kind_of(game):
    return KIND.get(game) or ("road" if "cartag" in game or "freeway" in game else
                             "maze" if "hunter" in game or "pacman" in game else "falling")


def load(game, suffix=""):
    z = np.load(EXP / "train" / f"g_{game}{suffix}" / "actor.npz")
    return {k: z[k].astype(np.float64) for k in z.files}


def road_after(env, a):
    """Agent cell and car columns after action a (array or scalar) and this step's car movement."""
    ar = np.clip(env.ar - (a == 1) + (a == 2), 1, 8)
    ac = np.clip(env.ac - (a == 3) + (a == 4), 1, 8)
    move = (env.t[:, None] % env.lane_period[None, :]) == 0
    cars = np.where(move, (env.cars - 1 + env.lane_dir[None, :]) % 8 + 1, env.cars)
    return ar, ac, cars


def contact_actions(env, kind):
    """(n, 5) bool: action a puts the agent on a mover this step (road, falling: exactly, by the game's own contact
    rule at the time of writing; maze: the destination cell holds a mover now)."""
    n = env.n
    H = np.zeros((n, 5), bool)
    idx = np.arange(n)
    for a in range(5):
        if kind == "road":
            if hasattr(env, "contact_actions"):
                return env.contact_actions()
            ar, ac, cars = road_after(env, a)
            lane = ar - 2
            H[:, a] = (lane >= 0) & (lane < 6) & (cars[idx, np.clip(lane, 0, 5)] == ac)
        elif kind == "maze":
            r, c = env._move(env.ar, env.ac, np.full(n, a))
            H[:, a] = ((env.gr == r[:, None]) & (env.gc == c[:, None])).any(1)
        else:
            ac = np.clip(env.ac - (a == 3) + (a == 4), 1, 8)
            H[:, a] = ((env.orow == 8) & (env.ocol == ac[:, None])).any(1)
    return H


def toward_nearest(true_obs, p, max_states=20000, seed=0):
    """Expected step along agent -> nearest mover (Manhattan distance 1-3) under action probabilities p."""
    sel = np.arange(len(true_obs))
    if len(sel) > max_states:
        sel = np.sort(np.random.default_rng(seed).choice(len(sel), max_states, replace=False))
    G = true_obs[sel].reshape(-1, 10, 10, 5)
    step = p[sel] @ MV
    out = []
    for i in range(len(sel)):
        ag = np.argwhere(G[i, :, :, 0] > 0)
        ms = np.argwhere(G[i, :, :, 1] > 0)
        if len(ag) == 0 or len(ms) == 0:
            continue
        pair = np.abs(ms[:, None, :] - ag[None, :, :]).sum(2)
        j, k = np.unravel_index(int(pair.argmin()), pair.shape)
        if pair[j, k] == 0 or pair[j, k] > 3:
            continue
        v = (ms[j] - ag[k]) / np.linalg.norm(ms[j] - ag[k])
        out.append(float(step[i] @ v))
    return float(np.mean(out)) if out else 0.0, len(out)


def run(game, policy, obs_mode="true", n=256, steps=600, seed=5, argmax=False, keep_states=False):
    rng = np.random.default_rng(seed)
    env = games.GAMES[game](n, rng)
    kind = kind_of(game)
    idx = np.arange(n)
    prev0, prev1 = np.zeros((n, 100), np.float32), np.zeros((n, 100), np.float32)
    tot = dict(ret=0.0, contact=0.0, goal=0.0, pellet=0.0, still_contact=0.0)
    acts = np.zeros(5)
    rows = np.zeros(10)
    hp, hu, hn = 0.0, 0.0, 0
    Xs, Ps = [], []
    for s in range(steps):
        f = env.frame()
        f0, f1 = f[:, :, 0].copy(), f[:, :, 1].copy()
        true = f.copy()
        true[:, :, 4] = np.clip(prev0 + prev1, 0, 1)
        if s < 3:
            assert np.array_equal(true.reshape(n, 500), env.obs())
        o = true
        if obs_mode == "decoy":
            o = f.copy()
            o[:, :, 1] = np.roll(f1, 1, 0)
            o[:, :, 4] = np.clip(prev0 + np.roll(prev1, 1, 0), 0, 1)
        elif obs_mode == "blind":
            o = f.copy()
            o[:, :, 1] = 0
            o[:, :, 4] = prev0
        if isinstance(policy, dict):
            p = softmax(forward(policy, o.reshape(n, 500).astype(np.float64)))
            act = p.argmax(1) if argmax else (p.cumsum(1) > rng.random((n, 1))).argmax(1)
        else:
            act = policy(env, rng)
            p = np.eye(5)[act]
        H = contact_actions(env, kind)
        some = H.any(1) & ~H.all(1)
        hp += float((p * H).sum(1)[some].sum())
        hu += float(H.mean(1)[some].sum())
        hn += int(some.sum())
        if kind == "road":
            rows += np.bincount(env.ar, minlength=10)
            ar0, ac0 = env.ar.copy(), env.ac.copy()
            ar1, ac1, _ = road_after(env, act)
        if kind == "maze":
            pel0 = env.pellets.copy()
        acts += np.bincount(act, minlength=5)
        Xs.append(true.reshape(n, 500).astype(np.uint8))
        Ps.append(p.astype(np.float32))
        r, d = env.step(act)
        tot["ret"] += float(r.sum())
        if kind == "road":
            hit = H[idx, act]
            tot["contact"] += float(hit.sum())
            tot["still_contact"] += float((hit & (ar1 == ar0) & (ac1 == ac0)).sum())
            tot["goal"] += float(((ar1 == 1) & (ar0 == 2)).sum())
        elif kind == "maze":
            eat = pel0[idx, env.ar, env.ac].astype(float)
            tot["pellet"] += float(eat.sum())
            if game == "hunter_v2":      # pellets pay nothing, each prey +1
                tot["contact"] += float(r.sum())
            elif game == "hunter":       # pellets +1, each prey +3
                tot["contact"] += float(((r - eat) / 3).sum())
            else:                        # pacman: pellets +1, a ghost contact -1
                tot["contact"] += float((eat - r).sum())
        else:
            tot["contact"] += float(np.abs(r).sum())
        prev0, prev1 = f0, f1
        env.reset(d)
        prev0[d] = 0
        prev1[d] = 0
    X, P = np.concatenate(Xs), np.concatenate(Ps)
    per = 200.0 / (n * steps)
    tw, tn = toward_nearest(X, P)
    out = {"return": round(tot["ret"] * per, 2), "contacts": round(tot["contact"] * per, 2),
           "toward_nearest": round(tw, 3), "toward_n": tn,
           "contact_index": {"policy": round(hp / max(hn, 1), 3), "uniform": round(hu / max(hn, 1), 3), "states": hn},
           "actions": [round(float(x), 3) for x in acts / acts.sum()]}
    if kind == "road":
        out["goal_arrivals"] = round(tot["goal"] * per, 2)
        out["contacts_while_still"] = round(tot["still_contact"] * per, 2)
        out["row_share"] = [round(float(x), 3) for x in rows[1:9] / rows.sum()]
    if kind == "maze":
        out["pellets"] = round(tot["pellet"] * per, 2)
    if keep_states:
        out["_X"], out["_P"] = X, P
    return out


def random_policy(env, rng):
    return rng.integers(0, 5, env.n)


def const_policy(a):
    return lambda env, rng: np.full(env.n, a)


def camp_policy(row, sweep=False):
    """Road: walk up to `row`, then stand still (or keep stepping against that lane's traffic)."""
    def f(env, rng):
        lane = row - 2
        against = 3 if env.lane_dir[lane] > 0 else 4
        return np.where(env.ar > row, 1, np.where(env.ar < row, 2, against if sweep else 0))
    return f
