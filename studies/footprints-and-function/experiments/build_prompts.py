"""Build blinded prompt suites for the recognition study.

Usage:
  python3 -I build_prompts.py released <out dir>       prompt changes and layout changes on the 4 released networks
  python3 -I build_prompts.py scrambled <out dir>      stage 1b: all 100 cells permuted and channel/action meanings removed
  python3 -I build_prompts.py nets <out dir> <name=actor.npz> ...   the post's prompt around our own networks

released conditions (every one keeps the network's behaviour, or the prompt's facts, exactly as stated below):
  base        the released prompt, unchanged
  unnamed     channel and action meanings removed from the description (grid geometry kept)
  chswap      the descriptions of channel 0 (agent) and channel 2 (reward targets) exchanged
  actswap     the descriptions of up/down and left/right exchanged (action 1 said to be left, 2 right, 3 up, 4 down)
  shuffled    W1 printed in a random line order, each line prefixed with its input number
  transposed  the network rewritten for the transposed game: cell (r, c) becomes (c, r) and the actions are relabelled
              to match (up<->left, down<->right), so it is exactly the same policy on the transposed grid
  permuted    the 64 interior cells relabelled by a fixed random permutation (same for all channels; the border is
              fixed), so it is exactly the same policy on a scrambled grid
IDs are random; key.json holds the truth. order.txt is a random run order.
"""
import json
import secrets
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ff_paths import LAYOUT, EXP1  # noqa: E402
sys.path.insert(0, str(LAYOUT))
from weights_text import HEADER, BLOCKS, fmt, released_actor, to_text  # noqa: E402

EXP = EXP1
RELEASED = {"pong": "dfc6cb65", "breakout": "a9f535cc", "pacman": "bdf99279", "freeway": "0bbcc3ea"}
TEMPLATE = EXP / "runs/pong/prompt.txt"

CH_LINES = ["  channel 0  agent: the cells the player controls",
            "  channel 1  moving objects: things other than the agent that move",
            "  channel 2  reward targets: things that give reward",
            "  channel 3  obstacles: cells the agent cannot enter or that end the episode (the wall border is always here)",
            "  channel 4  previous frame: the cells that were in channel 0 or channel 1 on the previous step"]
ACT_LINES = ["  action 0  no-op", "  action 1  up (toward row 0)", "  action 2  down (toward row 9)",
             "  action 3  left (toward column 0)", "  action 4  right (toward column 9)"]


def idx(r, c, k):
    return (r * 10 + c) * 5 + k


def head(template):
    t = Path(template).read_text()
    k = t.index("THE NETWORK'S WEIGHTS\n") + len("THE NETWORK'S WEIGHTS\n")
    return t[:k]


def edit_description(h, cond):
    for line in CH_LINES + ACT_LINES:
        assert line in h, line
    if cond == "unnamed":
        h = h.replace("Each cell has 5 binary channels (0 or 1):",
                      "Each cell has 5 binary channels (0 or 1), numbered 0 to 4. What each channel represents is not given:")
        for k, line in enumerate(CH_LINES):
            h = h.replace(line, f"  channel {k}")
        h = h.replace("one per action; the action is sampled from the softmax of these logits:",
                      "one per action; the action is sampled from the softmax of these logits. What each action does is "
                      "not given:")
        for k, line in enumerate(ACT_LINES):
            h = h.replace(line, f"  action {k}")
    elif cond == "chswap":
        h = h.replace(CH_LINES[0], "@@0@@").replace(CH_LINES[2], CH_LINES[0].replace("channel 0", "channel 2"))
        h = h.replace("@@0@@", CH_LINES[2].replace("channel 2", "channel 0"))
    elif cond == "actswap":
        new = ["  action 0  no-op", "  action 1  left (toward column 0)", "  action 2  right (toward column 9)",
               "  action 3  up (toward row 0)", "  action 4  down (toward row 9)"]
        block = "\n".join(ACT_LINES)
        assert block in h
        h = h.replace(block, "\n".join(new))
    return h


def shuffled_text(w, rng):
    out = [HEADER, ""]
    order = rng.permutation(500)
    for k, (name, hd) in enumerate(BLOCKS):
        a = np.asarray(w[name], dtype=np.float64)
        if name == "W1":
            out.append("W1 (layer 1 weights), shape 500 x 32: one line per row (row i = input i), 32 values per line. "
                       "The lines are in shuffled order; each line starts with its row number i and a colon.")
            out += [f"{i}: " + " ".join(fmt(v) for v in a[i]) for i in order]
        else:
            out.append(hd)
            rows = a if a.ndim == 2 else a[None, :]
            out += [" ".join(fmt(v) for v in row) for row in rows]
        if k < len(BLOCKS) - 1:
            out.append("")
    return "\n".join(out) + "\n"


def transposed(w):
    W1 = np.zeros_like(w["W1"])
    for r in range(10):
        for c in range(10):
            for k in range(5):
                W1[idx(r, c, k)] = w["W1"][idx(c, r, k)]
    perm = [0, 3, 4, 1, 2]  # new action j takes old action perm[j]: new up = old left, new left = old up
    return dict(w, W1=W1, W3=w["W3"][:, perm], b3=w["b3"][perm])


def permuted(w, rng):
    inner = [(r, c) for r in range(1, 9) for c in range(1, 9)]
    p = rng.permutation(64)
    W1 = w["W1"].copy()
    for j, (r, c) in enumerate(inner):
        nr, nc = inner[p[j]]
        for k in range(5):
            W1[idx(nr, nc, k)] = w["W1"][idx(r, c, k)]
    return dict(w, W1=W1), {f"{r},{c}": list(inner[p[j]]) for j, (r, c) in enumerate(inner)}


def scrambled(w, rng):
    cells = [(r, c) for r in range(10) for c in range(10)]
    p = rng.permutation(100)
    W1 = np.zeros_like(w["W1"])
    for j, (r, c) in enumerate(cells):
        nr, nc = cells[p[j]]
        for k in range(5):
            W1[idx(nr, nc, k)] = w["W1"][idx(r, c, k)]
    return dict(w, W1=W1), {f"{r},{c}": list(cells[p[j]]) for j, (r, c) in enumerate(cells)}


def released_scrambled(out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(exist_ok=True)
    rng = np.random.default_rng(202611051)
    h = edit_description(head(TEMPLATE), "unnamed")
    geo = ("The network sees a 10 x 10 grid. Rows are numbered 0 (top) to 9 (bottom) and columns 0 (left) to 9 (right). "
           "The outer ring of cells (row 0, row 9, column 0 and column 9) is a wall border; the playable area is the "
           "inner 8 x 8 cells (rows 1-8, columns 1-8).")
    assert geo in h
    h = h.replace(geo, "The network sees 100 cells, indexed by a row number 0-9 and a column number 0-9. The indexing "
                       "has been scrambled: it is for bookkeeping only and does not reflect where cells are in the "
                       "game, which cells are neighbours, or which are walls.")
    key, checks = {}, {}
    for game, nid in RELEASED.items():
        w = released_actor(EXP / "networks" / f"{nid}.npz")
        w2, cellmap = scrambled(w, rng)
        checks[f"{game}/scrambled"] = check_equivalent(w, w2, "permuted", cellmap)
        pid = secrets.token_hex(4)
        (out / "prompts" / f"{pid}.txt").write_text(h + to_text(w2))
        key[pid] = {"game": game, "condition": "scrambled", "source": nid, "cellmap": cellmap}
    order = list(key)
    rng.shuffle(order)
    (out / "key.json").write_text(json.dumps(key, indent=1))
    (out / "order.txt").write_text("\n".join(order) + "\n")
    (out / "equivalence_checks.json").write_text(json.dumps(checks, indent=1))
    print(len(key), "prompts; max |logit change|:", max(checks.values()))


def check_equivalent(w, w2, kind, cellmap=None, n=2000, seed=0):
    """Same action probabilities on random binary observations mapped into the new layout."""
    rng = np.random.default_rng(seed)
    x = (rng.random((n, 500)) < 0.05).astype(float)
    x2 = np.zeros_like(x)
    for r in range(10):
        for c in range(10):
            if kind == "transposed":
                nr, nc = c, r
            else:
                nr, nc = cellmap.get(f"{r},{c}", (r, c))
            for k in range(5):
                x2[:, idx(nr, nc, k)] = x[:, idx(r, c, k)]

    def logits(w, x):
        h = np.tanh(x @ w["W1"] + w["b1"])
        h = np.tanh(h @ w["W2"] + w["b2"])
        return h @ w["W3"] + w["b3"]
    z, z2 = logits(w, x), logits(w2, x2)
    if kind == "transposed":
        z = z[:, [0, 3, 4, 1, 2]]
    return float(np.abs(z - z2).max())


def released(out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(exist_ok=True)
    rng = np.random.default_rng(20261105)
    h0 = head(TEMPLATE)
    key, checks = {}, {}
    for game, nid in RELEASED.items():
        w = released_actor(EXP / "networks" / f"{nid}.npz")
        orig = (EXP / "networks" / f"{nid}_actor.txt").read_text()
        assert to_text(w) == orig
        for cond in ("base", "unnamed", "chswap", "actswap", "shuffled", "transposed", "permuted"):
            extra = {}
            if cond in ("base", "unnamed", "chswap", "actswap"):
                text = h0 if cond == "base" else edit_description(h0, cond)
                text += orig
                if cond == "base":
                    assert text == (EXP / "runs" / game / "prompt.txt").read_text()
            elif cond == "shuffled":
                text = h0 + shuffled_text(w, rng)
            elif cond == "transposed":
                w2 = transposed(w)
                checks[f"{game}/transposed"] = check_equivalent(w, w2, "transposed")
                text = h0 + to_text(w2)
            else:
                w2, cellmap = permuted(w, rng)
                checks[f"{game}/permuted"] = check_equivalent(w, w2, "permuted", cellmap)
                extra = {"cellmap": cellmap}
                text = h0 + to_text(w2)
            pid = secrets.token_hex(4)
            (out / "prompts" / f"{pid}.txt").write_text(text)
            key[pid] = {"game": game, "condition": cond, "source": nid, **extra}
    order = list(key)
    rng.shuffle(order)
    (out / "key.json").write_text(json.dumps(key, indent=1))
    (out / "order.txt").write_text("\n".join(order) + "\n")
    (out / "equivalence_checks.json").write_text(json.dumps(checks, indent=1))
    print(len(key), "prompts; max |logit change| after relabelling:", max(checks.values()))


def nets(out, specs):
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(exist_ok=True)
    rng = np.random.default_rng(int.from_bytes(secrets.token_bytes(4), "little"))
    h0 = head(TEMPLATE)
    key = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        w = dict(np.load(path))
        pid = secrets.token_hex(4)
        (out / "prompts" / f"{pid}.txt").write_text(h0 + to_text(w))
        key[pid] = {"name": name, "source": str(Path(path).resolve())}
    order = list(key)
    rng.shuffle(order)
    (out / "key.json").write_text(json.dumps(key, indent=1))
    (out / "order.txt").write_text("\n".join(order) + "\n")
    print(len(key), "prompts")


if __name__ == "__main__":
    if sys.argv[1] == "released":
        released(Path(sys.argv[2]))
    elif sys.argv[1] == "scrambled":
        released_scrambled(Path(sys.argv[2]))
    else:
        nets(Path(sys.argv[2]), sys.argv[3:])
