"""Rebuild the four camouflaged ReLU Breakout networks with unused inputs defined by the whole training record.

Usage: python3 -I build_fix_relu_breakout.py <out dir>
Same variants as build_suite_arch.py (seeds 0-1, unflipped and top-bottom flipped), new random ids. Writes key.json,
prompts/, nets/, order.txt, analyst.txt (flipped seed 0), camouflage_reports.json. Replaces suite_relu24x3's
Breakout camouflaged networks, whose camouflage overwrote 40 inputs the final policy uses.
"""
import json
import secrets
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arch import build_prompt  # noqa: E402
from build_suite import ORIENT, TEMPLATE  # noqa: E402

out = Path(sys.argv[1])
(out / "prompts").mkdir(parents=True, exist_ok=True)
(out / "nets").mkdir(exist_ok=True)
g = HERE / "train/g_breakout_relu24x3"
key, reports, analyst = {}, {}, []
for seed, flip in [(0, None), (1, None), (0, "ud"), (1, "ud")]:
    nid = secrets.token_hex(4)
    d = out / "nets" / nid
    r = subprocess.run([sys.executable, "-I", str(HERE / "camouflage_arch.py"), str(g / "actor.npz"),
                        str(g / "states.npy"), str(d), "--arch", "relu24x3", "--seed", str(seed),
                        "--flip", flip or "none", "--mirror", "ud", "--never-on", str(g / "occupancy_never_on.npy")],
                       capture_output=True, text=True, check=True)
    reports[nid] = json.loads(r.stdout.strip().splitlines()[-1])
    (out / "prompts" / f"{nid}.txt").write_text(build_prompt(TEMPLATE, (d / "actor.txt").read_text(), "relu24x3"))
    key[nid] = {"game": "breakout", "arch": "relu24x3", "condition": "camo", "seed": seed, "flip": flip,
                "orientation": ORIENT["breakout"][1 if flip else 0], "replaces_suite": "suite_relu24x3"}
    if seed == 0 and flip:
        analyst.append(nid)
(out / "key.json").write_text(json.dumps(key, indent=1))
(out / "camouflage_reports.json").write_text(json.dumps(reports, indent=1))
(out / "analyst.txt").write_text("\n".join(analyst) + "\n")
order = list(key)
np.random.default_rng(11).shuffle(order)
(out / "order.txt").write_text("\n".join(order) + "\n")
for nid, rp in reports.items():
    print(nid, rp["never_used_inputs"], rp["fill_kinds"], rp["max_prob_change_holdout_exact"], rp["max_prob_change_all_as_printed"])
