"""Procedural twin-game generator: policies whose layout is shared and whose stance toward moving objects differs.

Usage: python3 -I twingen.py <out dir> [--settings falling,arena,road,maze] [--seeds 4] [--states 120000]
                             [--hidden 32,32] [--camo 1]
For every (setting, seed) one shared state set is collected by rolling out a mixture of all stance policies (plus
random actions), so every stance sees exactly the same inputs: footprints are identical by construction. Each stance
labels the same states with its scripted action, and a tanh MLP (500 -> hidden -> 5) is imitation-trained on them.
Training reweights each class toward the mean action marginal of its family (removes the class-wide action prior).
A network is kept only at >= 0.80 held-out accuracy AND
if its behaviour on held-out states confirms its stance (state_probe: expected step toward the nearest mover within
3 cells is > 0.02 for approach and < -0.02 for avoid; for coupled pairs, split by that mover's sideways motion). Each kept network is exported plain and with --camo exact
camouflage copies (exact on the shared state set), as tasks with ground truth for grader.py.

Observation (as in the Exp 1 prompt): 10x10 grid, 5 binary channels, input (row*10+col)*5+ch. ch0 agent, ch1 moving
objects, ch2 static targets, ch3 walls (border always), ch4 previous frame (agent and movers one step earlier).
Actions: 0 no-op, 1 up, 2 down, 3 left, 4 right.
Settings:
  falling  agent on row 8 moving left/right; two movers fall one row per step from row 1
  arena    open field; agent moves freely; two movers random-walk; two static targets
  road     six lanes (rows 2-7) of horizontally moving cars; agent moves freely; goal row 1 shown in ch2
  maze     fixed symmetric maze; agent moves along corridors; two movers random-walk; pellets in ch2
  arena_c movers keep a sideways direction (bouncing off walls), with a random vertical step 25% of the time
  arena_c, road_c, maze_c  coupled versions of the free settings: approach_right approaches movers that moved right
           last step and avoids those that moved left (approach_left the reverse); road_c draws lane directions per
           episode so the data are mirror-closed. Idle behaviour (no relevant mover / no target) is a mirror-symmetric
           patrol that ignores movers.
  drift    as falling, but each mover drifts one column left or right per step (visible against the previous frame);
           the two classes are coupled by mirror symmetry: approach_right catches right-drifting movers and dodges
           left-drifting ones, approach_left the reverse (same input usage, opposite interaction sign)
Stances (toward channel-1 movers):
  approach  step that minimises the distance to the nearest mover
  avoid     when the nearest mover is within 3 cells, step that maximises the distance to it; otherwise as ignore
  ignore    go for the nearest target if any, otherwise no-op (movers play no role)
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

N = 10
MOVES = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
STANCES = ("approach", "avoid", "ignore")
COUPLED = ("approach_right", "approach_left")  # drift setting: catch right-drifting, dodge left-drifting (or reverse)


def stances_for(setting):
    return COUPLED if setting == "drift" or setting.endswith("_c") else STANCES


def base_of(setting):
    """Geometry of a setting: coupled variants (name_c) reuse the base layout; drift reuses falling."""
    return "falling" if setting == "drift" else setting[:-2] if setting.endswith("_c") else setting
EPOCHS = None
MAZE = {2: (2, 3, 6, 7), 3: (2, 7), 4: (4, 5), 5: (4, 5), 6: (2, 7), 7: (2, 3, 6, 7)}


def idx(r, c, ch):
    return (r * 10 + c) * 5 + ch


class World:
    """One episode of a setting; small and deterministic given the rng."""

    def __init__(self, setting, rng):
        self.s, self.rng = setting, rng
        self.base = base_of(setting)
        self.wall = np.zeros((N, N), bool)
        self.wall[0, :] = self.wall[-1, :] = self.wall[:, 0] = self.wall[:, -1] = True
        if self.base == "maze":
            for r, cs in MAZE.items():
                for c in cs:
                    self.wall[r, c] = True
        self.open = [(r, c) for r in range(1, 9) for c in range(1, 9) if not self.wall[r, c]]
        self.reset()

    def reset(self):
        rng, s = self.rng, self.base
        if s == "falling":
            self.agent = (8, int(rng.integers(1, 9)))
            self.movers = [(1, int(rng.integers(1, 9))), (5, int(rng.integers(1, 9)))]
            self.drift = [int(rng.choice([-1, 1])) for _ in range(2)]
            self.targets = set()
        elif s == "road":
            self.agent = (8, int(rng.integers(1, 9)))
            self.cars = [int(c) for c in rng.integers(1, 9, 6)]
            # coupled road: lane directions random per episode, so the data are closed under left-right mirroring
            self.dirs = ([int(d) for d in rng.choice([-1, 1], 6)] if self.s == "road_c" else [1, -1, 1, -1, 1, -1])
            self.movers = [(L + 2, self.cars[L]) for L in range(6)]
            self.targets = {(1, c) for c in range(1, 9)}
        else:
            cells = [self.open[i] for i in rng.choice(len(self.open), 5 if s == "arena" else 3, replace=False)]
            self.agent, self.movers = cells[0], cells[1:3]
            self.vel = [int(v) for v in rng.choice([-1, 1], 2)]
            self.targets = set(cells[3:]) if s == "arena" else {c for c in self.open if c != self.agent}
        self.prev = [self.agent] + list(self.movers)
        self.t = 0

    @property
    def setting(self):
        return self.s

    def obs(self):
        x = np.zeros(500, np.float32)
        x[[idx(r, c, 3) for r in range(N) for c in range(N) if self.wall[r, c]]] = 1
        x[idx(*self.agent, 0)] = 1
        for m in self.movers:
            x[idx(*m, 1)] = 1
        for t in self.targets:
            x[idx(*t, 2)] = 1
        for p in self.prev:
            x[idx(*p, 4)] = 1
        return x

    def allowed(self, a):
        if self.base == "falling" and a in (1, 2):
            return False
        r, c = self.agent
        dr, dc = MOVES[a]
        return not self.wall[r + dr, c + dc]

    def step(self, a):
        self.prev = [self.agent] + list(self.movers)
        if self.allowed(a):
            r, c = self.agent
            dr, dc = MOVES[a]
            self.agent = (r + dr, c + dc)
        if self.base != "road":
            self.targets.discard(self.agent)
        if self.s == "falling":
            self.movers = [(1, int(self.rng.integers(1, 9))) if r == 8 else (r + 1, c) for r, c in self.movers]
        elif self.s == "drift":
            new = []
            for k, (r, c) in enumerate(self.movers):
                if r == 8:
                    self.drift[k] = int(self.rng.choice([-1, 1]))
                    new.append((1, int(self.rng.integers(1, 9))))
                    continue
                if not 1 <= c + self.drift[k] <= 8:
                    self.drift[k] = -self.drift[k]
                new.append((r + 1, c + self.drift[k]))
            self.movers = new
        elif self.base == "road":
            self.cars = [(c - 1 + self.dirs[L]) % 8 + 1 for L, c in enumerate(self.cars)]
            self.movers = [(L + 2, self.cars[L]) for L in range(6)]
        elif self.s == "arena_c":
            # persistent sideways motion (bounce off walls); 25% of steps a random vertical step instead
            new = []
            for k, (r, c) in enumerate(self.movers):
                if self.rng.random() < 0.25:
                    dr = int(self.rng.choice([-1, 1]))
                    new.append((r + dr, c) if not self.wall[r + dr, c] else (r, c))
                    continue
                if self.wall[r, c + self.vel[k]]:
                    self.vel[k] = -self.vel[k]
                new.append((r, c + self.vel[k]) if not self.wall[r, c + self.vel[k]] else (r, c))
            self.movers = new
        else:
            new = []
            for r, c in self.movers:
                dr, dc = MOVES[int(self.rng.integers(0, 5))]
                new.append((r + dr, c + dc) if not self.wall[r + dr, c + dc] else (r, c))
            self.movers = new
        if self.base == "maze" and not self.targets:
            self.targets = {c for c in self.open if c != self.agent}
        if self.base == "arena" and not self.targets:
            self.targets = {self.open[i] for i in self.rng.choice(len(self.open), 2, replace=False)}
        self.t += 1
        if self.t >= 50:
            self.reset()


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def pick(w, acts, key):
    """Best action by key (smaller is better) with a mirror-symmetric tie-break: no-op first, then the move toward the
    board centre (so left-right mirrored states get mirrored actions)."""
    best = min(key(a) for a in acts)
    tied = [a for a in acts if key(a) == best]
    if len(tied) == 1 or 0 in tied:
        return tied[0] if len(tied) == 1 else 0
    r, c = w.agent

    def centre(a):
        dr, dc = MOVES[a]
        return abs(r + dr - 4.5) + abs(c + dc - 4.5)
    tied.sort(key=centre)
    return tied[0]


def hvel(w, k):
    """Horizontal velocity of mover k from the previous frame (cars wrap 8 -> 1)."""
    dc = w.movers[k][1] - w.prev[1 + k][1]
    return -int(np.sign(dc)) if abs(dc) > 1 else int(dc)


def patrol(w, acts):
    """Idle behaviour that ignores movers, moves almost every step and is mirror-symmetric: keep moving the way the
    agent moved last step (from the previous frame); if blocked or stationary, move horizontally toward the farther
    side wall."""
    r, c = w.agent
    pr, pc = w.prev[0]
    d = (r - pr, c - pc)
    for a, mv in MOVES.items():
        if a and mv == d and a in acts:
            return a
    far = 4 if c <= 4 else 3
    if far in acts:
        return far
    other = 3 if far == 4 else 4
    return other if other in acts else 0


def scripted(w, stance):
    """The scripted action of a stance in world w (mirror-symmetric tie-break, see pick)."""
    acts = [a for a in range(5) if a == 0 or w.allowed(a)]
    r, c = w.agent

    def after(a):
        dr, dc = MOVES[a]
        return (r + dr, c + dc) if w.allowed(a) else (r, c)

    movers = w.movers
    near = min(movers, key=lambda m: dist(w.agent, m))
    if stance in COUPLED:
        k = movers.index(near)
        v = w.drift[k] if w.s == "drift" else hvel(w, k)
        stance = "neutral" if v == 0 else ("approach" if (v == 1) == (stance == "approach_right") else "avoid")
    tgt = min(w.targets, key=lambda t: dist(w.agent, t)) if w.targets else None
    if stance == "approach":
        return pick(w, acts, lambda a: dist(after(a), near))
    if stance == "avoid" and dist(w.agent, near) <= 3:
        return pick(w, acts, lambda a: -dist(after(a), near))
    if tgt is not None and w.base != "road":
        return pick(w, acts, lambda a: dist(after(a), tgt))
    return patrol(w, acts)


def collect(setting, seed, n_states):
    """Shared states from a mixture of all stance policies plus 20% random actions."""
    rng = np.random.default_rng(seed)
    w = World(setting, rng)
    classes = stances_for(setting)
    X, labels = [], {s: [] for s in classes}
    behave = classes[int(rng.integers(len(classes)))]
    while len(X) < n_states:
        if w.t == 0:
            behave = classes[int(rng.integers(len(classes)))]
        X.append(w.obs())
        for s in classes:
            labels[s].append(scripted(w, s))
        a = int(rng.integers(5)) if rng.random() < 0.2 else labels[behave][-1]
        w.step(a)
    return np.array(X), {s: np.array(v) for s, v in labels.items()}


def train(X, y, hidden, seed, epochs=30, sw=None, out_bias=True):
    epochs = EPOCHS or epochs
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    sizes = [500] + hidden + [5]
    layers = []
    for k in range(len(sizes) - 1):
        lin = nn.Linear(sizes[k], sizes[k + 1])
        nn.init.orthogonal_(lin.weight, gain=1.0 if k == len(sizes) - 2 else math.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin] + ([nn.Tanh()] if k < len(sizes) - 2 else [])
    net = nn.Sequential(*layers)
    if not out_bias:  # output biases frozen at 0: no class-wide action prior can be stored there
        layers[-1].bias.requires_grad_(False)
    n = len(X)
    perm = np.random.default_rng(seed).permutation(n)
    tr, te = perm[: int(0.85 * n)], perm[int(0.85 * n):]
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).long()
    swt = torch.ones(len(y)) if sw is None else torch.from_numpy(sw.astype(np.float32))
    opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad], lr=1e-3)
    for _ in range(epochs):
        for b in np.array_split(np.random.default_rng(seed + _).permutation(tr), max(1, len(tr) // 256)):
            ce = nn.functional.cross_entropy(net(Xt[b]), yt[b], reduction="none")
            loss = (ce * swt[b]).sum() / swt[b].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        acc = float((net(Xt[te]).argmax(1) == yt[te]).float().mean())
    lin = [m for m in net if isinstance(m, nn.Linear)]
    w = {}
    for k, m in enumerate(lin, 1):
        w[f"W{k}"] = m.weight.detach().numpy().T.astype(np.float64)
        w[f"b{k}"] = m.bias.detach().numpy().astype(np.float64)
    return w, acc


def forward(w, X):
    n = len([k for k in w if k[0] == "W"])
    h = X
    for k in range(1, n):
        h = np.tanh(h @ w[f"W{k}"] + w[f"b{k}"])
    return h @ w[f"W{n}"] + w[f"b{n}"]


def camouflage(w, X, rng):
    """Exact on X: never-active rows -> copies of used rows of the same channel; unit shuffle + sign flips; output
    shift. Returns new weights and the largest probability change on X."""
    w = {k: v.copy() for k, v in w.items()}
    n = len([k for k in w if k[0] == "W"])
    dead = X.max(0) == 0
    live = np.where(~dead)[0]
    for i in np.where(dead)[0]:
        pool = [j for j in live if j % 5 == i % 5] or list(live)
        w["W1"][i] = w["W1"][pool[rng.integers(len(pool))]]
    for k in range(1, n):
        m = w[f"W{k}"].shape[1]
        pi, sg = rng.permutation(m), rng.choice([-1.0, 1.0], m)
        w[f"W{k}"] = w[f"W{k}"][:, pi] * sg
        w[f"b{k}"] = w[f"b{k}"][pi] * sg
        w[f"W{k + 1}"] = w[f"W{k + 1}"][pi, :] * sg[:, None]
    u = rng.standard_normal(w[f"W{n}"].shape[0]) * 3 * float(np.abs(w[f"W{n}"]).max())
    w[f"W{n}"] = w[f"W{n}"] + u[:, None]
    w[f"b{n}"] = w[f"b{n}"] + rng.normal() * 3
    return w


def coupled_probe(w):
    """Drift setting: expected step toward a mover within 3 cells whose previous-frame cell shows a right (+1) or left
    (-1) drift. Returns (score for right-drifting, score for left-drifting)."""
    out = []
    for dc in (1, -1):
        X, D = [], []
        for ac in range(1, 9):
            for mr in range(5, 8):
                for mc in range(1, 9):
                    pc = mc - dc
                    if not 1 <= pc <= 8 or abs(mr - 8) + abs(mc - ac) > 3:
                        continue
                    x = np.zeros(500)
                    x[[idx(r, c, 3) for r in range(N) for c in range(N) if r in (0, 9) or c in (0, 9)]] = 1
                    x[[idx(8, ac, 0), idx(mr, mc, 1), idx(8, ac, 4), idx(mr - 1, pc, 4)]] = 1
                    v = np.array([0.0, float(mc + dc * (8 - mr) - ac)])
                    if v[1] == 0:
                        continue
                    X.append(x)
                    D.append(v / np.linalg.norm(v))
        p = softmax(forward(w, np.array(X)))
        mv = np.array([MOVES[a] for a in range(5)], float)
        out.append(float(((p @ mv) * np.array(D)).sum(1).mean()))
    return out


def coupled_probe_free(w, setting):
    """Coupled free settings: expected step toward a mover within 3 cells that moved one column right (+1) or left
    (-1) last step (previous-frame cell beside it); agent stationary. Returns (score right-moving, score left-moving)."""
    wall = World(setting, np.random.default_rng(0)).wall
    out = []
    for dc in (1, -1):
        X, D = [], []
        for ar in range(1, 9):
            for ac in range(1, 9):
                if wall[ar, ac]:
                    continue
                for mr in (range(2, 8) if base_of(setting) == "road" else range(1, 9)):
                    for mc in range(1, 9):
                        pc = mc - dc
                        dd = abs(mr - ar) + abs(mc - ac)
                        if wall[mr, mc] or not 1 <= pc <= 8 or wall[mr, pc] or dd == 0 or dd > 3 or (mr, pc) == (ar, ac):
                            continue
                        x = np.zeros(500)
                        x[[idx(r, c, 3) for r in range(N) for c in range(N) if wall[r, c]]] = 1
                        x[[idx(ar, ac, 0), idx(mr, mc, 1), idx(ar, ac, 4), idx(mr, pc, 4)]] = 1
                        v = np.array([mr - ar, mc - ac], float)
                        X.append(x)
                        D.append(v / np.linalg.norm(v))
        p = softmax(forward(w, np.array(X)))
        mv = np.array([MOVES[a] for a in range(5)], float)
        out.append(float(((p @ mv) * np.array(D)).sum(1).mean()))
    return out


def state_probe(w, X, coupled=False):
    """On-distribution behaviour check on real (held-out) states: expected step along the agent -> nearest-mover
    direction, over states with the nearest mover within 3 cells. coupled: split by that mover's horizontal motion
    read from the previous-frame channel; returns (right-moving, left-moving)."""
    G = X.reshape(-1, 10, 10, 5)
    p = softmax(forward(w, X.astype(np.float64)))
    mv = np.array([MOVES[a] for a in range(5)], float)
    step = p @ mv
    res = {1: [], -1: [], 0: []}
    for i in range(len(X)):
        ag = np.argwhere(G[i, :, :, 0] > 0)
        ms = np.argwhere(G[i, :, :, 1] > 0)
        if len(ag) == 0 or len(ms) == 0:
            continue
        pair = np.abs(ms[:, None, :] - ag[None, :, :]).sum(2)    # movers x agent cells (paddles span 2 cells)
        j, k = np.unravel_index(int(pair.argmin()), pair.shape)
        if pair[j, k] == 0 or pair[j, k] > 3:
            continue
        a, m = ag[k], ms[j]
        v = (m - a) / np.linalg.norm(m - a)
        key = 0
        if coupled:
            left = m[1] - 1 >= 0 and G[i, m[0], m[1] - 1, 4] > 0
            right = m[1] + 1 <= 9 and G[i, m[0], m[1] + 1, 4] > 0
            key = 1 if left and not right else -1 if right and not left else None
            if key is None:
                continue
        res[key].append(float(step[i] @ v))
    if coupled:
        return [float(np.mean(res[1])) if res[1] else 0.0, float(np.mean(res[-1])) if res[-1] else 0.0]
    return float(np.mean(res[0])) if res[0] else 0.0


def stance_probe(w, setting):
    """Behavioural check (forward passes on built boards): expected step toward a single mover within 3 cells of the
    agent, walls on, no targets, previous frame = current cells. > 0 approach, < 0 avoid."""
    wall = World(setting, np.random.default_rng(0)).wall
    rows = [8] if base_of(setting) == "falling" else range(1, 9)
    X, D = [], []
    for ar in rows:
        for ac in range(1, 9):
            if wall[ar, ac]:
                continue
            for mr in (range(2, 8) if base_of(setting) == "road" else range(1, 9)):
                for mc in range(1, 9):
                    d = abs(mr - ar) + abs(mc - ac)
                    if wall[mr, mc] or d == 0 or d > 3:
                        continue
                    x = np.zeros(500)
                    x[[idx(r, c, 3) for r in range(N) for c in range(N) if wall[r, c]]] = 1
                    x[[idx(ar, ac, 0), idx(mr, mc, 1), idx(ar, ac, 4), idx(mr, mc, 4)]] = 1
                    v = np.array([mr - ar, mc - ac], float)
                    X.append(x)
                    D.append(v / np.linalg.norm(v))
    p = softmax(forward(w, np.array(X)))
    mv = np.array([MOVES[a] for a in range(5)], float)
    return float(((p @ mv) * np.array(D)).sum(1).mean())


def softmax(l):
    e = np.exp(l - l.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--settings", default="falling,arena,road,maze")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--states", type=int, default=120000)
    ap.add_argument("--hidden", default="32,32")
    ap.add_argument("--camo", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min-acc", type=float, default=0.80, help="held-out accuracy bar (the behaviour check is the "
                    "label-validity test; coupled free settings at 64-64 are accepted at 0.70)")
    ap.add_argument("--no-out-bias", action="store_true", help="freeze output biases at 0 (camouflage then adds a "
                    "shared random constant), removing the output-bias leak")
    a = ap.parse_args()
    global EPOCHS
    EPOCHS = a.epochs
    out = Path(a.out)
    (out / "nets").mkdir(parents=True, exist_ok=True)
    hidden = [int(h) for h in a.hidden.split(",")]
    index = []
    for setting in a.settings.split(","):
        for seed in range(a.seeds):
            X, labels = collect(setting, 1000 * seed + 7, a.states)
            Xu = np.unique(X, axis=0)
            margs = {st: np.bincount(labels[st], minlength=5) / len(X) for st in stances_for(setting)}
            target = np.mean(list(margs.values()), axis=0)
            for stance in stances_for(setting):
                # action-marginal matching: reweight so every class trains toward the same action marginal, which
                # removes the class-wide action prior (output-bias leak) found in pilot v1
                aw = np.clip(target / np.maximum(margs[stance], 1e-6), 0.2, 5.0)
                w, acc = train(X, labels[stance], hidden, seed, sw=aw[labels[stance]], out_bias=not a.no_out_bias)
                Xh = X[int(0.85 * len(X)):][:20000]
                if stance in COUPLED:
                    pr, pl = state_probe(w, Xh, coupled=True)
                    probe = [round(pr, 4), round(pl, 4)]
                    behaves = (pr > 0.02 and pl < -0.02) if stance == "approach_right" else (pr < -0.02 and pl > 0.02)
                else:
                    probe = state_probe(w, Xh)
                    behaves = {"approach": probe > 0.02, "avoid": probe < -0.02, "ignore": True}[stance]
                rec = {"setting": setting, "seed": seed, "stance": stance, "heldout_acc": round(acc, 4),
                       "hidden": hidden, "stance_probe": probe if isinstance(probe, list) else round(probe, 4), "behaves_as_labelled": bool(behaves),
                       "action_marginal": [round(float(v), 3) for v in margs[stance]],
                       "kept": acc >= a.min_acc and behaves}
                if not rec["kept"]:
                    index.append(rec)
                    print(json.dumps(rec), flush=True)
                    continue
                rng = np.random.default_rng(10_000 + seed)
                variants = [("plain", w)] + [(f"camo{k}", camouflage(w, Xu, rng)) for k in range(a.camo)]
                ref = softmax(forward(w, Xu))
                for cond, wv in variants:
                    nid = f"{setting}-{seed}-{stance}-{cond}"
                    np.savez(out / "nets" / f"{nid}.npz", **wv)
                    dp = float(np.abs(softmax(forward(wv, Xu)) - ref).max())
                    index.append({**rec, "id": nid, "condition": cond, "max_prob_change_on_states": dp,
                                  "never_active_inputs": int((Xu.max(0) == 0).sum())})
                print(json.dumps(rec), flush=True)
            np.save(out / f"states_{setting}_{seed}.npy", Xu.astype(np.uint8))
    (out / "index.json").write_text(json.dumps(index, indent=1))
    kept = [r for r in index if r.get("id")]
    print(f"{len(kept)} networks exported; dropped {sum(not r['kept'] for r in index)} (accuracy or behaviour check)")


if __name__ == "__main__":
    main()
