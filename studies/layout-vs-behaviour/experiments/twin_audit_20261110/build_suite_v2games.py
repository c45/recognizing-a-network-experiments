"""Suite of the redesigned games, in the format of build_suite_twins.py: per game 1 plain + 2 Level-2 camouflaged
networks (seeds 10, 11, no flips), 500-32-32-5, the post's Exp 1 prompt.

Usage: python -I build_suite_v2games.py <out dir> [train suffix, e.g. _s2]
Writes key.json (game, twin pair, stance), prompts/, nets/, order.txt, analyst.txt, camouflage_reports.json.
"""
import json
import secrets
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OBS = HERE.parent
sys.path.insert(0, str(OBS))
from arch import build_prompt, to_text  # noqa: E402
from build_suite import TEMPLATE  # noqa: E402

GAMES = {"hunter_v2": ("maze", "approach"), "freeway_v2": ("road", "avoid"), "cartag_v2": ("road", "approach")}
TWINS = {"hunter_v2", "cartag_v2"}

train, out = OBS / "train", Path(sys.argv[1])
SUF = sys.argv[2] if len(sys.argv) > 2 else ""
(out / "prompts").mkdir(parents=True, exist_ok=True)
(out / "nets").mkdir(exist_ok=True)
key, reports, analyst = {}, {}, []
for game, (setting, stance) in GAMES.items():
    g = train / f"g_{game}{SUF}"
    for cond, seed in [("plain", 0), ("camo", 10), ("camo", 11)]:
        nid = secrets.token_hex(4)
        d = out / "nets" / nid
        if cond == "plain":
            z = np.load(g / "actor.npz")
            d.mkdir(parents=True)
            (d / "actor.txt").write_text(to_text({k: z[k] for k in z.files}, "tanh32x2"))
        else:
            r = subprocess.run([sys.executable, "-I", str(OBS / "camouflage_arch.py"), str(g / "actor.npz"),
                                str(g / "states.npy"), str(d), "--arch", "tanh32x2", "--seed", str(seed),
                                "--flip", "none", "--mirror", "lr", "--never-on", str(g / "occupancy_never_on.npy")],
                               capture_output=True, text=True, check=True)
            reports[nid] = json.loads(r.stdout.strip().splitlines()[-1])
            if seed == 10 and game in TWINS:
                analyst.append(nid)
        (out / "prompts" / f"{nid}.txt").write_text(build_prompt(TEMPLATE, (d / "actor.txt").read_text(), "tanh32x2"))
        key[nid] = {"game": game, "setting": setting, "stance": stance, "twin": game in TWINS, "condition": cond,
                    "seed": seed, "flip": None, "orientation": None, "arch": "tanh32x2",
                    "train_seed": 2 if SUF == "_s2" else 1}
(out / "key.json").write_text(json.dumps(key, indent=1))
(out / "camouflage_reports.json").write_text(json.dumps(reports, indent=1))
(out / "analyst.txt").write_text("\n".join(analyst) + "\n")
order = list(key)
np.random.default_rng(13).shuffle(order)
(out / "order.txt").write_text("\n".join(order) + "\n")
print(len(key), "networks; max exact holdout change", max(r["max_prob_change_holdout_exact"] for r in reports.values()),
      "; as printed", max(r["max_prob_change_all_as_printed"] for r in reports.values()))
