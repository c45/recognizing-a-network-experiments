"""Train the L4-scale variants (30 seeds x 2 classes each; seeds 20-29 only complete counterbalancing), 6 processes at a time; skips finished networks.

Usage: python3 -I train_variants.py <states.npz> <variant:depth:width> [...]   e.g. d4:4:32 d8:8:32 w256:2:256
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = int(__import__("os").environ.get("SEEDS", "30"))


def job(args):
    states, name, depth, width, rule, seed = args
    out = HERE / f"nets_{name}" / f"{rule}_s{seed}.npz"
    if out.exists():
        return
    subprocess.run([sys.executable, "-I", str(HERE / "scale.py"), "train", states, rule, str(seed), str(out),
                    "--depth", str(depth), "--width", str(width), "--epochs", os.environ.get("EPOCHS", "10")],
                   check=True, capture_output=True)


def main(states, variants):
    jobs = []
    for v in variants:
        name, depth, width = v.split(":")
        (HERE / f"nets_{name}").mkdir(exist_ok=True)
        jobs += [(states, name, int(depth), int(width), r, s) for r in ("right_toward", "right_away") for s in range(SEEDS)]
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(job, jobs))
    for v in variants:
        name = v.split(":")[0]
        acc = [json.loads(p.read_text())["heldout_acc"] for p in (HERE / f"nets_{name}").glob("*.json")]
        print(name, "n", len(acc), "min acc", min(acc), "below .95", sum(a < 0.95 for a in acc), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
