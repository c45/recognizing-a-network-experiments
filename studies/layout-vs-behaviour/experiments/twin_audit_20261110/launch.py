"""Start detached training runs. Usage: python3 launch.py <game>:<out dir name>:<seed>:<steps> ..."""
import os
import subprocess
import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1] / "train"
for spec in sys.argv[1:]:
    game, out, seed, steps = spec.split(":")
    log = open(TRAIN / f"log_{out[2:] if out.startswith('g_') else out}.txt", "w")
    cmd = ["uv", "run", "--offline", "--no-project", "--with", "torch", "--with", "numpy", "python", "-I",
           str(TRAIN / "train_v2.py"), game, str(TRAIN / out), "--seed", seed, "--steps", steps, "--threads", "2"]
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                         cwd=str(TRAIN), env={**os.environ, "OMP_NUM_THREADS": "2"})
    print(spec, "pid", p.pid)
