"""A 10x10 grid Pong in the format of Exp 1, and PPO training of the 500-32-32-5 tanh policy.

Usage: python3 -I pong_ppo.py <out dir> [--steps 30000000] [--seed 0] [--input-noise 0.0]
Environment (our reconstruction; the post's game code is not released):
  observation: 10x10 cells x 5 binary channels, input i = (row*10+col)*5+channel
    ch0 agent: the paddle, 2 cells tall in column 1 (rows p, p+1, p in 1..7)
    ch1 moving objects: the ball (rows 1-8, columns 2-8)
    ch2 reward targets: the right edge, column 9 rows 1-8 (always shown)
    ch3 obstacles: the wall border (always on)
    ch4 previous frame: cells of channels 0 and 1 on the previous step
  actions: 0 no-op, 1 up, 2 down, 3 left, 4 right (left/right do nothing)
  ball moves one cell diagonally per step, bounces off rows 1/8; reaching column 9 gives +1 and bounces back;
  reaching column 1 bounces off the paddle if it covers that row, otherwise -1 and the episode ends; 200-step limit.
PPO settings follow the post's meta files (64 envs x 128 steps, 4 epochs, 8 minibatches, lr 5e-4 annealed, clip 0.2,
entropy 0.01, gamma 0.99, GAE 0.95, grad-norm 0.5, orthogonal init, actor output gain 0.01).
--input-noise p: during training each of the 500 inputs is also switched on with probability p (every row gets
gradient). Writes actor.npz (W1,b1,W2,b2,W3,b3 as in-by-out kernels), occupancy.npy (True = input never on in clean
observations), states.npy (a sample of clean visited observations) and train_log.json.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def idx(r, c, ch):
    return (r * 10 + c) * 5 + ch


class Pong:
    def __init__(self, n, rng):
        self.n, self.rng = n, rng
        self.static = np.zeros(500, np.float32)
        for r in range(10):
            for c in range(10):
                if r in (0, 9) or c in (0, 9):
                    self.static[idx(r, c, 3)] = 1
        for r in range(1, 9):
            self.static[idx(r, 9, 2)] = 1
        self.p = np.zeros(n, int)
        self.br = np.zeros(n, int)
        self.bc = np.zeros(n, int)
        self.dr = np.zeros(n, int)
        self.dc = np.zeros(n, int)
        self.t = np.zeros(n, int)
        self.prev = np.zeros((n, 500), np.float32)
        self.reset(np.ones(n, bool))

    def reset(self, m):
        k = int(m.sum())
        if k == 0:
            return
        self.p[m] = self.rng.integers(1, 8, k)
        self.br[m] = self.rng.integers(1, 9, k)
        self.bc[m] = self.rng.integers(5, 9, k)
        self.dr[m] = self.rng.choice([-1, 1], k)
        self.dc[m] = -1
        self.t[m] = 0
        self.prev[m] = 0

    def cells(self):
        cur = np.zeros((self.n, 500), np.float32)
        a = np.arange(self.n)
        cur[a, idx(self.p, 1, 0)] = 1
        cur[a, idx(self.p + 1, 1, 0)] = 1
        cur[a, idx(self.br, self.bc, 1)] = 1
        return cur

    def obs(self):
        cur = self.cells()
        o = cur + self.static[None, :]
        prev_cells = self.prev.reshape(self.n, 100, 5)
        o4 = (prev_cells[:, :, 0] + prev_cells[:, :, 1]).clip(0, 1)
        o.reshape(self.n, 100, 5)[:, :, 4] = o4
        return o.clip(0, 1)

    def step(self, action):
        self.prev = self.cells()
        self.p = np.clip(self.p - (action == 1) + (action == 2), 1, 7)
        nr = self.br + self.dr
        flip = (nr < 1) | (nr > 8)
        self.dr[flip] *= -1
        nr = self.br + self.dr
        nc = self.bc + self.dc
        reward = np.zeros(self.n, np.float32)
        done = np.zeros(self.n, bool)
        right = nc >= 9
        reward[right] += 1
        self.dc[right] = -1
        nc[right] = self.bc[right] - 1
        left = nc <= 1
        hit = left & (nr >= self.p) & (nr <= self.p + 1)
        miss = left & ~hit
        self.dc[hit] = 1
        nc[hit] = 2
        reward[miss] -= 1
        done |= miss
        self.br, self.bc = nr, np.clip(nc, 1, 8)
        self.t += 1
        done |= self.t >= 200
        return reward, done


def mlp(out, gain_out):
    net = nn.Sequential(nn.Linear(500, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, out))
    for k, m in enumerate([net[0], net[2], net[4]]):
        nn.init.orthogonal_(m.weight, gain=gain_out if k == 2 else math.sqrt(2))
        nn.init.zeros_(m.bias)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=30_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input-noise", type=float, default=0.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    rng = np.random.default_rng(a.seed)
    NE, NS, EPOCHS, MB = 64, 128, 4, 8
    actor, critic = mlp(5, 0.01), mlp(1, 1.0)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=5e-4, eps=1e-5)
    env = Pong(NE, rng)
    seen = np.zeros(500, bool)
    sample_states = []
    updates = a.steps // (NE * NS)
    ep_ret, ep_rets, log = np.zeros(NE), [], []
    obs = env.obs()
    t0 = time.time()
    for u in range(updates):
        for g in opt.param_groups:
            g["lr"] = 5e-4 * (1 - u / updates)
        bo, ba, blp, bv, br, bd = [], [], [], [], [], []
        for _ in range(NS):
            seen |= obs.max(0) > 0
            if len(sample_states) < 400 and rng.random() < 0.02:
                sample_states.append(obs[: 8].copy())
            o = obs
            if a.input_noise > 0:
                o = np.maximum(obs, (rng.random(obs.shape) < a.input_noise).astype(np.float32))
            ot = torch.from_numpy(o)
            with torch.no_grad():
                logits = actor(ot)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                v = critic(ot).squeeze(-1)
            r, d = env.step(act.numpy())
            ep_ret += r
            for k in np.where(d)[0]:
                ep_rets.append(ep_ret[k])
                ep_ret[k] = 0
            env.reset(d)
            bo.append(ot)
            ba.append(act)
            blp.append(dist.log_prob(act))
            bv.append(v)
            br.append(torch.from_numpy(r))
            bd.append(torch.from_numpy(d.astype(np.float32)))
            obs = env.obs()
        o = obs
        if a.input_noise > 0:
            o = np.maximum(obs, (rng.random(obs.shape) < a.input_noise).astype(np.float32))
        with torch.no_grad():
            nv = critic(torch.from_numpy(o)).squeeze(-1)
        adv = torch.zeros(NS, NE)
        last = torch.zeros(NE)
        for s in reversed(range(NS)):
            nxt = nv if s == NS - 1 else bv[s + 1]
            delta = br[s] + 0.99 * nxt * (1 - bd[s]) - bv[s]
            last = delta + 0.99 * 0.95 * (1 - bd[s]) * last
            adv[s] = last
        ret = adv + torch.stack(bv)
        O, A, LP, ADV, RET = (torch.cat(bo), torch.cat(ba), torch.cat(blp), adv.reshape(-1), ret.reshape(-1))
        n = O.shape[0]
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for k in range(MB):
                j = perm[k * n // MB:(k + 1) * n // MB]
                dist = torch.distributions.Categorical(logits=actor(O[j]))
                ratio = torch.exp(dist.log_prob(A[j]) - LP[j])
                adv_j = (ADV[j] - ADV[j].mean()) / (ADV[j].std() + 1e-8)
                pl = -torch.min(ratio * adv_j, ratio.clamp(0.8, 1.2) * adv_j).mean()
                vl = 0.5 * ((critic(O[j]).squeeze(-1) - RET[j]) ** 2).mean()
                loss = pl + 0.5 * vl - 0.01 * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)
                opt.step()
        if u % 200 == 0 or u == updates - 1:
            mr = float(np.mean(ep_rets[-200:])) if ep_rets else 0.0
            log.append({"update": u, "steps": (u + 1) * NE * NS, "mean_return": round(mr, 3), "s": round(time.time() - t0)})
            print(log[-1], flush=True)
    sd = {k: v.detach().numpy() for k, v in actor.state_dict().items()}
    np.savez(out / "actor.npz", W1=sd["0.weight"].T, b1=sd["0.bias"], W2=sd["2.weight"].T, b2=sd["2.bias"],
             W3=sd["4.weight"].T, b3=sd["4.bias"])
    np.save(out / "occupancy_never_on.npy", ~seen)
    np.save(out / "states.npy", np.concatenate(sample_states))
    (out / "train_log.json").write_text(json.dumps({"args": vars(a), "never_on_inputs": int((~seen).sum()),
                                                     "log": log}, indent=1))


if __name__ == "__main__":
    main()
