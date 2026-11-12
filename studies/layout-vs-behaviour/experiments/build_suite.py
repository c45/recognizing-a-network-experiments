"""Build the blinded camouflage suite: prompts for every network, a private key, and battery results.

Usage: python3 -I build_suite.py <train dir> <out dir> [camouflage script, default camouflage.py]
Per game (trained network <train dir>/g_<game>/actor.npz and states.npy):
  plain    the trained network as is (control)
  camo     4 camouflaged copies: seeds 0-1 in the trained orientation and seeds 0-1 flipped (Pong left-right,
           Breakout and Freeway top-bottom); Pac-Man's maze is symmetric, so it gets seeds 0-3 unflipped
Plus one untrained network (orthogonal init, gain sqrt 2, output 0.01, as in PPO) showing default guesses.
Every prompt is the post's Exp 1 prompt with the weights replaced. IDs are random; key.json holds the truth:
game, condition, seed, flip, and the agent's side/orientation after the flip.
"""
import json
import secrets
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from weights_text import build_prompt, to_text  # noqa: E402

sys.path.insert(0, str(HERE))
from paths import PROMPT_TEMPLATE as TEMPLATE  # noqa: E402
PY = sys.executable
CAMO = "camouflage.py"
FLIP = {"pong": "lr", "breakout": "ud", "freeway": "ud", "pacman": None}
ORIENT = {  # (trained orientation, flipped orientation)
    "pong": ("paddle on the left edge", "paddle on the right edge"),
    "breakout": ("paddle on the bottom row", "paddle on the top row"),
    "freeway": ("start at the bottom, goal at the top", "start at the top, goal at the bottom"),
    "pacman": ("maze", "maze"),
}


def untrained(seed):
    rng = np.random.default_rng(seed)

    def orth(n_in, n_out, gain):
        q, r = np.linalg.qr(rng.standard_normal((max(n_in, n_out), min(n_in, n_out))))
        q = q * np.sign(np.diag(r))
        return (q if n_in >= n_out else q.T) * gain
    return {"W1": orth(500, 32, np.sqrt(2)), "b1": np.zeros(32), "W2": orth(32, 32, np.sqrt(2)), "b2": np.zeros(32),
            "W3": orth(32, 5, 0.01), "b3": np.zeros(5)}


def main(train_dir, out_dir):
    train_dir, out = Path(train_dir), Path(out_dir)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    (out / "nets").mkdir(exist_ok=True)
    key, reports = {}, {}
    for game in ("pong", "breakout", "pacman", "freeway"):
        g = train_dir / f"g_{game}"
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
                (d / "actor.txt").write_text(to_text({k: z[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}))
            else:
                extra = ["--mirror", FLIP[game] or "lr"] if CAMO == "camouflage_v2.py" else []
                r = subprocess.run([PY, "-I", str(HERE / CAMO), str(g / "actor.npz"), str(g / "states.npy"),
                                    str(d), "--seed", str(seed), "--flip", flip or "none", *extra],
                                   capture_output=True, text=True, check=True)
                reports[nid] = json.loads(r.stdout.strip().splitlines()[-1])
            (out / "prompts" / f"{nid}.txt").write_text(build_prompt(TEMPLATE, (d / "actor.txt").read_text()))
            key[nid] = {"game": game, "condition": cond, "seed": seed, "flip": flip,
                        "orientation": ORIENT[game][1 if flip else 0]}
    nid = secrets.token_hex(4)
    d = out / "nets" / nid
    d.mkdir(parents=True)
    (d / "actor.txt").write_text(to_text(untrained(7)))
    (out / "prompts" / f"{nid}.txt").write_text(build_prompt(TEMPLATE, (d / "actor.txt").read_text()))
    key[nid] = {"game": "untrained", "condition": "untrained", "seed": 7, "flip": None, "orientation": None}
    (out / "key.json").write_text(json.dumps(key, indent=1))
    (out / "camouflage_reports.json").write_text(json.dumps(reports, indent=1))
    order = list(key)
    np.random.default_rng(11).shuffle(order)
    (out / "order.txt").write_text("\n".join(order) + "\n")
    print(len(key), "prompts;", "max prob change as printed:",
          max(r["max_prob_change_all_as_printed"] for r in reports.values()),
          "; max exact holdout change:", max(r["max_prob_change_holdout_exact"] for r in reports.values()))


if __name__ == "__main__":
    if len(sys.argv) > 3:
        CAMO = sys.argv[3]
    main(sys.argv[1], sys.argv[2])
