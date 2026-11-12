"""Play audit of redesigned-game policies. Usage: python -I audit_v2.py <out json> <game>:<train dir name> ...

For each policy: play in its own game with the real observation, with decoy movers and with movers deleted, greedy
play, the causal probe of rl_pilot/behaviour_checks.py, and the share of states where deleting the movers changes
the top action. Also random play in each game.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))
import audit_lib as A  # noqa: E402
from behaviour_checks import causal_probe  # noqa: E402

TRAIN = HERE.parent / "train"


def show(tag, m):
    ci = m["contact_index"]
    extra = "".join(f" {k}={m[k]}" for k in ("goal_arrivals", "contacts_while_still", "pellets", "causal",
                                               "top_action_changes_blind") if k in m)
    print(f"{tag:36s} ret {m['return']:8.2f}  contacts {m['contacts']:7.2f}  toward {m['toward_nearest']:+.3f}"
          f"  contact idx {ci['policy']:.3f} vs {ci['uniform']:.3f}{extra}", flush=True)


def top_action_changes(game, w, n=256, steps=300):
    rng = np.random.default_rng(7)
    env = A.games.GAMES[game](n, rng)
    prev0 = np.zeros((n, 100), np.float32)
    prev1 = np.zeros((n, 100), np.float32)
    ch = []
    for _ in range(steps):
        f = env.frame()
        f0, f1 = f[:, :, 0].copy(), f[:, :, 1].copy()
        t, b = f.copy(), f.copy()
        t[:, :, 4] = np.clip(prev0 + prev1, 0, 1)
        b[:, :, 1] = 0
        b[:, :, 4] = prev0
        pt = A.softmax(A.forward(w, t.reshape(n, 500).astype(np.float64)))
        pb = A.softmax(A.forward(w, b.reshape(n, 500).astype(np.float64)))
        ch.append(pt.argmax(1) != pb.argmax(1))
        _, d = env.step((pt.cumsum(1) > rng.random((n, 1))).argmax(1))
        prev0, prev1 = f0, f1
        env.reset(d)
        prev0[d] = 0
        prev1[d] = 0
    return round(float(np.mean(ch)), 4)


out = {}
seen = set()
for spec in sys.argv[2:]:
    game, d = spec.split(":")
    z = np.load(TRAIN / d / "actor.npz")
    w = {k: z[k].astype(np.float64) for k in z.files}
    m = A.run(game, w, keep_states=True)
    m["causal"] = round(causal_probe(w, m.pop("_X").astype(float)), 3)
    m.pop("_P")
    m["top_action_changes_blind"] = top_action_changes(game, w)
    out[f"{d}|true"] = m
    show(f"{d}", m)
    for mode in ("decoy", "blind"):
        out[f"{d}|{mode}"] = A.run(game, w, obs_mode=mode)
        show(f"{d} {mode}", out[f"{d}|{mode}"])
    out[f"{d}|argmax"] = A.run(game, w, argmax=True)
    show(f"{d} argmax", out[f"{d}|argmax"])
    if game not in seen:
        seen.add(game)
        out[f"random|{game}"] = A.run(game, A.random_policy)
        show(f"random in {game}", out[f"random|{game}"])
        if "freeway" in game:
            out[f"always_up|{game}"] = A.run(game, A.const_policy(1))
            show(f"always up in {game}", out[f"always_up|{game}"])
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
