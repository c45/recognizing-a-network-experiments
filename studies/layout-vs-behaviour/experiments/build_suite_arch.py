"""Build a blinded Level-2 camouflage suite for one architecture (same design as build_suite.py with camouflage_v2).

Usage: python3 -I build_suite_arch.py <train dir> <arch> <out dir>
Networks come from <train dir>/g_<game>_<arch> (or g_<game> for tanh32x2). Per game: plain, 4 camouflaged copies
(Pong left-right, Breakout and Freeway top-bottom flips: seeds 0-1 unflipped and flipped; Pac-Man seeds 0-3), plus one
untrained network of the architecture. Prompts are the post's Exp 1 prompt with the network section and weights
rewritten for the architecture. Writes key.json, prompts/, nets/, order.txt, camouflage_reports.json, analyst.txt
(one flipped or seed-0 camouflaged network per game, for the analyst arm).
"""
import json
import math
import secrets
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arch import ARCHS, build_prompt, sizes, to_text  # noqa: E402
from build_suite import FLIP, ORIENT, TEMPLATE  # noqa: E402


def untrained(arch, seed):
    rng = np.random.default_rng(seed)
    s, w = sizes(arch), {}
    for k in range(1, len(s)):
        n_in, n_out = s[k - 1], s[k]
        q, r = np.linalg.qr(rng.standard_normal((max(n_in, n_out), min(n_in, n_out))))
        q = q * np.sign(np.diag(r))
        w[f"W{k}"] = (q if n_in >= n_out else q.T) * (0.01 if k == len(s) - 1 else math.sqrt(2))
        w[f"b{k}"] = np.zeros(n_out)
    return w


def main(train_dir, arch, out_dir):
    train_dir, out = Path(train_dir), Path(out_dir)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    (out / "nets").mkdir(exist_ok=True)
    key, reports, analyst = {}, {}, []
    for game in ("pong", "breakout", "pacman", "freeway"):
        g = train_dir / (f"g_{game}" if arch == "tanh32x2" else f"g_{game}_{arch}")
        variants = [("plain", 0, None)]
        if FLIP[game]:
            variants += [("camo", s, None) for s in (0, 1)] + [("camo", s, FLIP[game]) for s in (0, 1)]
        else:
            variants += [("camo", s, None) for s in (0, 1, 2, 3)]
        for cond, seed, flip in variants:
            nid = secrets.token_hex(4)
            d = out / "nets" / nid
            if cond == "plain":
                z = np.load(g / "actor.npz")
                d.mkdir(parents=True)
                (d / "actor.txt").write_text(to_text({k: z[k] for k in z.files}, arch))
            else:
                r = subprocess.run([sys.executable, "-I", str(HERE / "camouflage_arch.py"), str(g / "actor.npz"),
                                    str(g / "states.npy"), str(d), "--arch", arch, "--seed", str(seed),
                                    "--flip", flip or "none", "--mirror", FLIP[game] or "lr"],
                                   capture_output=True, text=True, check=True)
                reports[nid] = json.loads(r.stdout.strip().splitlines()[-1])
                if seed == 0 and (flip or not FLIP[game]):
                    analyst.append(nid)
            (out / "prompts" / f"{nid}.txt").write_text(build_prompt(TEMPLATE, (d / "actor.txt").read_text(), arch))
            key[nid] = {"game": game, "arch": arch, "condition": cond, "seed": seed, "flip": flip,
                        "orientation": ORIENT[game][1 if flip else 0]}
    nid = secrets.token_hex(4)
    d = out / "nets" / nid
    d.mkdir(parents=True)
    (d / "actor.txt").write_text(to_text(untrained(arch, 7), arch))
    (out / "prompts" / f"{nid}.txt").write_text(build_prompt(TEMPLATE, (d / "actor.txt").read_text(), arch))
    key[nid] = {"game": "untrained", "arch": arch, "condition": "untrained", "seed": 7, "flip": None,
                "orientation": None}
    (out / "key.json").write_text(json.dumps(key, indent=1))
    (out / "camouflage_reports.json").write_text(json.dumps(reports, indent=1))
    (out / "analyst.txt").write_text("\n".join(analyst) + "\n")
    order = list(key)
    np.random.default_rng(11).shuffle(order)
    (out / "order.txt").write_text("\n".join(order) + "\n")
    print(arch, len(key), "prompts; max prob change as printed:",
          max(r["max_prob_change_all_as_printed"] for r in reports.values()),
          "; max exact holdout change:", max(r["max_prob_change_holdout_exact"] for r in reports.values()),
          "; plain as printed:", max(r["max_prob_change_plain_as_printed"] for r in reports.values()))


if __name__ == "__main__":
    assert sys.argv[2] in ARCHS
    main(*sys.argv[1:4])
