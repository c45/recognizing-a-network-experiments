"""First-layer footprint maps of the four networks released with "Sparks of Weight-Space Programming".

Usage: uv run --with numpy python -I make_pictures.py [networks dir]
Default networks dir: ../studies/layout-vs-behaviour/external/lm_understand-logs/exp1_which_game/networks (fetch it with
layout-vs-behaviour/external/fetch_upstream.sh; third-party, RikiyaT/lm_understand-logs at commit 4d09ba6, CC BY 4.0).
Writes pictures.json next to this file: per game, five 10 x 10 maps (one per input channel) of the L2 norm of each
input's 32 first-layer weights, divided by the channel's largest. make_website_data.py copies it into the post's data.

"camo" holds the same maps after copy camouflage, as an illustration. The game code of these networks is not released,
so the unused inputs are inferred from the weights: an input counts as unused when its first-layer weights still have
their starting size (L2 norm under UNUSED_BELOW; the untrained rows of all four networks lie between 0.22 and 0.49, the
trained ones above 0.5, with a narrow gap in Breakout and Pac-Man). Each such input gets a copy of a trained input's weights: its mirror-image cell in the same
channel if that is trained, else a random trained input of that channel (the rule of the recognition study's camo_v2).
Whether this edit leaves these four policies' behaviour unchanged is NOT checked; that check exists only for our own
policies, whose unused inputs are known from the training record.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NETS = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    HERE.parent / "studies" / "layout-vs-behaviour" / "external" / "lm_understand-logs" / "exp1_which_game" / "networks")

UNUSED_BELOW = 0.5
MIRROR = {"pong": "lr", "breakout": "ud", "freeway": "ud", "pacman": "lr"}


def as_maps(w1):
    norm = np.linalg.norm(w1.astype(np.float64).reshape(10, 10, 5, -1), axis=-1)
    return [(norm[:, :, ch] / norm[:, :, ch].max()).round(2).tolist() for ch in range(5)]


def camouflage(w1, game, rng):
    w1 = w1.astype(np.float64)
    unused = np.linalg.norm(w1, axis=1) < UNUSED_BELOW
    out = w1.copy()
    for i in np.where(unused)[0]:
        r, c, ch = (i // 5) // 10, (i // 5) % 10, i % 5
        # A channel with no trained input at all (Pac-Man's obstacles) borrows from any trained input, as camo_v2 does.
        pool = [j for j in np.where(~unused)[0] if j % 5 == ch] or list(np.where(~unused)[0])
        m = (r * 10 + 9 - c) * 5 + ch if MIRROR[game] == "lr" else ((9 - r) * 10 + c) * 5 + ch
        out[i] = w1[m if not unused[m] else pool[rng.integers(len(pool))]]
    return out, int(unused.sum())


out = {}
for f in sorted(NETS.glob("*.npz")):
    meta = json.loads((NETS / "meta" / f"{f.stem}.json").read_text())
    w1 = np.load(f)["actor/dense1/kernel"]
    assert w1.shape == (500, 32) and meta["actor_parameters"] == 17253
    camo, n_unused = camouflage(w1, meta["game"], np.random.default_rng(0))
    out[meta["game"]] = {"id": f.stem, "env_steps": meta["env_steps"], "maps": as_maps(w1), "camo": as_maps(camo),
                         "unused_inputs": n_unused}
assert sorted(out) == ["breakout", "freeway", "pacman", "pong"], sorted(out)
(HERE / "pictures.json").write_text(json.dumps(out, separators=(",", ":")) + "\n")
print({g: (v["id"], v["unused_inputs"]) for g, v in out.items()})
