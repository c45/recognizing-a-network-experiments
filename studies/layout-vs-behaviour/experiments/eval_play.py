"""Play every suite network in its game, exactly as printed (3 decimals), and compare with random play.

Usage: python3 -I eval_play.py <suite dir> [...]   -> <suite>/play.json
Flipped networks play the flipped game: the observation is mirrored before the network sees it and the network's
action is mapped back (left/right or up/down swapped). 64 parallel games x 4,000 steps, actions sampled from the
softmax (as in training) and also greedy (argmax). For camouflaged networks, also the largest action-probability change
versus the uncamouflaged reference (reference_flipped.npz) on every state visited, and how often an input the
camouflage overwrote was active.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "train"))
from arch import arch_of, logits, parse_text  # noqa: E402
from camouflage import flip_inputs  # noqa: E402
from games import GAMES  # noqa: E402

NE, STEPS = 64, 4000
SWAP = {"lr": [0, 1, 2, 4, 3], "ud": [0, 2, 1, 3, 4]}


def softmax(l):
    e = np.exp(l - l.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def play(game, policy, seed, flip=None, ref=None):
    rng = np.random.default_rng(seed)
    env = GAMES[game](NE, rng)
    p = flip_inputs(flip) if flip else None
    ret, rets, maxdp, argdiff, n = np.zeros(NE), [], 0.0, 0, 0
    for _ in range(STEPS):
        o = env.obs().astype(np.float64)
        if p is not None:
            o2 = np.zeros_like(o)
            o2[:, p] = o
            o = o2
        pr = policy(o)
        if ref is not None:
            pref = ref(o)
            maxdp = max(maxdp, float(np.abs(pr - pref).max()))
            argdiff += int((pr.argmax(1) != pref.argmax(1)).sum())
            n += len(o)
        a = (pr.cumsum(1) > rng.random((NE, 1))).argmax(1)
        if flip:
            a = np.array(SWAP[flip])[a]
        r, d = env.step(a)
        ret += r
        for k in np.where(d)[0]:
            rets.append(ret[k])
            ret[k] = 0
        env.reset(d)
    out = {"episodes": len(rets), "mean_return": round(float(np.mean(rets)), 3) if rets else None}
    if ref is not None:
        out.update(max_prob_change_vs_reference=round(maxdp, 4), argmax_differs=round(argdiff / n, 5))
    return out


def greedy(policy):
    def g(o):
        pr = policy(o)
        h = np.zeros_like(pr)
        h[np.arange(len(pr)), pr.argmax(1)] = 1.0
        return h
    return g


def main():
    for s in sys.argv[1:]:
        s = Path(s)
        key = json.loads((s / "key.json").read_text())
        res = {}
        for game in ("pong", "breakout", "pacman", "freeway"):
            res[f"random_{game}"] = {"game": game, "condition": "random",
                                     "sampled": play(game, lambda o: np.full((len(o), 5), 0.2), 1)}
        for nid, k in sorted(key.items(), key=lambda kv: (kv[1]["game"], kv[1]["condition"])):
            w = parse_text(s / "nets" / nid / "actor.txt")
            arch = arch_of(w)
            pol = lambda o, w=w, arch=arch: softmax(logits(w, o, arch))  # noqa: E731
            games = [k["game"]] if k["game"] != "untrained" else ["pong", "breakout", "pacman", "freeway"]
            for g in games:
                ref = None
                refp = s / "nets" / nid / "reference_flipped.npz"
                if refp.exists():
                    z = np.load(refp)
                    wr = {kk: z[kk].astype(np.float64) for kk in z.files}
                    ref = lambda o, wr=wr, arch=arch: softmax(logits(wr, o, arch))  # noqa: E731
                tag = nid if k["game"] != "untrained" else f"{nid}_{g}"
                res[tag] = {"game": g, "condition": k["condition"], "flip": k["flip"], "seed": k["seed"], "arch": arch,
                            "sampled": play(g, pol, 1, k["flip"], ref), "greedy": play(g, greedy(pol), 2, k["flip"])}
                print(s.name, g, k["condition"], k["flip"], k["seed"], res[tag]["sampled"], "greedy",
                      res[tag]["greedy"]["mean_return"], flush=True)
        (s / "play.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
