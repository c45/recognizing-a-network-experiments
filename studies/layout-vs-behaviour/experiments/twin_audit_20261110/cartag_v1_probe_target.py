"""Car-tag v1: the probe's 'step toward the nearest car' measured against where the car is now (the published
probe) and against where it will be after this step's car movement (v1 scores contact after the cars move).
Also splits the probe by distance to the nearest car. Writes cartag_v1_probe_target.json."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_lib as A  # noqa: E402

out = {}
for suf in ("", "_s2"):
    w = A.load("cartag", suf)
    n = 256
    rng = np.random.default_rng(5)
    env = A.games.GAMES["cartag"](n, rng)
    res = {"now": {1: [], 2: [], 3: []}, "next": {1: [], 2: [], 3: []}}
    for s in range(400):
        p = A.softmax(A.forward(w, env.obs().astype(np.float64)))
        step = p @ A.MV
        _, _, cars1 = A.road_after(env, np.zeros(n, int))
        for key, cars in (("now", env.cars), ("next", cars1)):
            dr = (np.arange(2, 8)[None, :] - env.ar[:, None]).astype(float)
            dc = (cars - env.ac[:, None]).astype(float)
            dist = np.abs(dr) + np.abs(dc)
            j = dist.argmin(1)
            i = np.arange(n)
            d = dist[i, j]
            ok = (d >= 1) & (d <= 3)
            v = np.stack([dr[i, j], dc[i, j]], 1)
            v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            proj = (step * v).sum(1)
            for k in (1, 2, 3):
                res[key][k] += list(proj[ok & (d == k)])
        act = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
        _, d_ = env.step(act)
        env.reset(d_)
    out["cartag" + suf] = {key: {"all": round(float(np.mean(sum(v.values(), []))), 3),
                                 **{f"dist{k}": round(float(np.mean(x)), 3) for k, x in v.items()},
                                 **{f"n{k}": len(x) for k, x in v.items()}} for key, v in res.items()}
    print("cartag" + suf, out["cartag" + suf], flush=True)
(HERE / "cartag_v1_probe_target.json").write_text(json.dumps(out, indent=1))
