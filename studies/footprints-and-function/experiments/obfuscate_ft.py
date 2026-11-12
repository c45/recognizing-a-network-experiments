"""Continued training that tries to hide the first-layer footprint while keeping the policy's behaviour.

Usage: python3 -I obfuscate_ft.py <game> <actor.npz> <out dir> --decoy q --flat lam [--profile mu] [--steps 20000]
1. Roll out the trained policy (64 envs x 4,000 steps) and keep every clean visited observation; inputs never on in
   them are the "unused" set.
2. Fine-tune a copy by distillation: minimise KL(original(x) || new(x~)) on visited states x, where x~ switches on
   each unused input with probability q ("decoys", so unused rows receive gradient), plus lam * the variance of the
   500 first-layer row norms (pushes every row toward the same size), plus mu * the mean squared gap between each
   row's sorted |values| and the average sorted |values| profile (every row gets the same value distribution, so
   order-free row statistics such as the largest value or the leading-digit count stop separating rows).
   --profile-unused mu applies that penalty to unused rows only, toward the used rows' average profile.
Writes actor.npz, unused.npy and ft.json: the largest action-probability change on clean visited states, KL, and the
row-norm spread before and after.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from games_new import ALL  # noqa: E402


def forward(w, x):
    h = torch.tanh(x @ w["W1"] + w["b1"])
    h = torch.tanh(h @ w["W2"] + w["b2"])
    return h @ w["W3"] + w["b3"]


def rollout_states(game, w, n_env=64, steps=4000, seed=0):
    rng = np.random.default_rng(seed)
    env = ALL[game](n_env, rng)
    wt = {k: torch.tensor(v, dtype=torch.float32) for k, v in w.items()}
    states = []
    for _ in range(steps):
        o = env.obs()
        states.append(o.copy())
        with torch.no_grad():
            p = torch.softmax(forward(wt, torch.from_numpy(o)), -1).numpy()
        a = np.array([rng.choice(5, p=q / q.sum()) for q in p])
        r, d = env.step(a)
        env.reset(d)
    return np.unique(np.concatenate(states).astype(np.uint8), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("actor")
    ap.add_argument("out")
    ap.add_argument("--decoy", type=float, default=0.0)
    ap.add_argument("--flat", type=float, default=0.0)
    ap.add_argument("--profile", type=float, default=0.0)
    ap.add_argument("--profile-unused", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    w0 = {k: v.astype(np.float32) for k, v in dict(np.load(a.actor)).items()}
    X = rollout_states(a.game, w0)
    unused = X.max(0) == 0
    np.save(out / "unused.npy", unused)
    Xt = torch.from_numpy(X.astype(np.float32))
    t0 = {k: torch.tensor(v) for k, v in w0.items()}
    with torch.no_grad():
        p0 = torch.softmax(forward(t0, Xt), -1)
    w = {k: torch.tensor(v, requires_grad=True) for k, v in w0.items()}
    opt = torch.optim.Adam(w.values(), lr=1e-3)
    um = torch.from_numpy(unused.astype(np.float32))
    rng = torch.Generator().manual_seed(a.seed)
    for s in range(a.steps):
        j = torch.randint(0, len(Xt), (1024,), generator=rng)
        x = Xt[j]
        if a.decoy > 0:
            x = torch.maximum(x, (torch.rand(x.shape, generator=rng) < a.decoy).float() * um)
        logq = torch.log_softmax(forward(w, x), -1)
        kl = (p0[j] * (torch.log(p0[j] + 1e-12) - logq)).sum(-1).mean()
        norms = w["W1"].norm(dim=1)
        loss = kl + a.flat * norms.var()
        if a.profile_unused > 0:
            prof = w["W1"].abs().sort(dim=1).values
            used_prof = prof[~torch.from_numpy(unused)].mean(0).detach()
            loss = loss + a.profile_unused * ((prof[torch.from_numpy(unused)] - used_prof) ** 2).mean() * 32
        if a.profile > 0:
            prof = w["W1"].abs().sort(dim=1).values
            loss = loss + a.profile * ((prof - prof.mean(0, keepdim=True).detach()) ** 2).mean() * 32
        opt.zero_grad()
        loss.backward()
        opt.step()
    wf = {k: v.detach() for k, v in w.items()}
    with torch.no_grad():
        p1 = torch.softmax(forward(wf, Xt), -1)
    n0, n1 = np.linalg.norm(w0["W1"], axis=1), wf["W1"].norm(dim=1).numpy()
    np.savez(out / "actor.npz", **{k: v.numpy() for k, v in wf.items()})
    rep = {"game": a.game, "source": str(Path(a.actor).resolve()), "decoy": a.decoy, "flat": a.flat,
           "profile": a.profile, "profile_unused": a.profile_unused, "steps": a.steps,
           "visited_states": int(len(X)), "unused_inputs": int(unused.sum()),
           "max_abs_dp_clean": float((p1 - p0).abs().max()), "mean_kl_clean": float(
               (p0 * (torch.log(p0 + 1e-12) - torch.log(p1 + 1e-12))).sum(-1).mean()),
           "row_norm_unused_mean": [float(n0[unused].mean()), float(n1[unused].mean())],
           "row_norm_used_mean": [float(n0[~unused].mean()), float(n1[~unused].mean())]}
    (out / "ft.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
