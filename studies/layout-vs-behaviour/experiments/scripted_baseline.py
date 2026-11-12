"""Scripted behaviour baseline: approach vs avoid toward the moving objects, from the printed weights alone.

Usage: python3 -I scripted_baseline.py <suite dir> [...]   -> <suite>/scripted_baseline.json and a summary
Uses only what the Exp 1 prompt states (channel meanings, action meanings, input layout): no game code, no game states.
For every placement of one agent cell (channel 0) and one moving object (channel 1) on different interior cells of an
otherwise empty board (wall border on in channel 3, previous frame = the same two cells), it runs the network as
printed and takes the expected movement along the agent-to-object direction:
    score = sum_a p(a) * (move(a) . d / |d|),  move: up (-1,0), down (+1,0), left (0,-1), right (0,+1), no-op 0.
The network's stance is the sign of the mean score over all placements (positive = approach, negative = avoid).
Variant "row8": the agent only on row 8 (for paddle-like layouts the prompt cannot reveal; reported, not used).
Variant "near" (post hoc, added after seeing the primary results): only placements with the object within Manhattan
distance 2 of the agent, where avoidance matters.
LOCKED RULE for held-out suites (fixed on Nov 6 before suite_twins2 existed): stance = sign of the "near" score,
"unclear" when |score| < 0.005.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arch import arch_of, logits, parse_text  # noqa: E402

MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], float)
STANCE = {"pong": "approach", "breakout": "approach", "catch": "approach", "hunter": "approach", "cartag": "approach",
          "pacman": "avoid", "freeway": "avoid", "dodge": "avoid"}


def idx(r, c, ch):
    return (r * 10 + c) * 5 + ch


def placements(agent_rows=range(1, 9), maxdist=99):
    cells = [(r, c) for r in range(1, 9) for c in range(1, 9)]
    X, D = [], []
    border = [idx(r, c, 3) for r in range(10) for c in range(10) if r in (0, 9) or c in (0, 9)]
    for ar in agent_rows:
        for ac in range(1, 9):
            for orow, oc in cells:
                if (orow, oc) == (ar, ac) or abs(orow - ar) + abs(oc - ac) > maxdist:
                    continue
                x = np.zeros(500)
                x[border] = 1
                x[[idx(ar, ac, 0), idx(orow, oc, 1), idx(ar, ac, 4), idx(orow, oc, 4)]] = 1
                d = np.array([orow - ar, oc - ac], float)
                X.append(x)
                D.append(d / np.linalg.norm(d))
    return np.array(X), np.array(D)


def score(w, arch, X, D):
    l = logits(w, X, arch)
    p = np.exp(l - l.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return float(((p @ MOVES) * D).sum(1).mean())


def main():
    Xa, Da = placements()
    X8, D8 = placements([8])
    Xn, Dn = placements(maxdist=2)
    for s in sys.argv[1:]:
        s = Path(s)
        key = json.loads((s / "key.json").read_text())
        rows = {}
        for nid, k in key.items():
            if k["game"] not in STANCE:
                continue
            w = parse_text(s / "nets" / nid / "actor.txt")
            arch = arch_of(w)
            sa, s8, sn = score(w, arch, Xa, Da), score(w, arch, X8, D8), score(w, arch, Xn, Dn)
            pred = "approach" if sa > 0 else "avoid"
            rows[nid] = {**k, "score_all": round(sa, 4), "score_row8": round(s8, 4), "score_near": round(sn, 4),
                         "pred_near": "approach" if sn > 0 else "avoid", "pred": pred,
                         "pred_locked": "unclear" if abs(sn) < 0.005 else ("approach" if sn > 0 else "avoid"),
                         "truth": STANCE[k["game"]], "correct": pred == STANCE[k["game"]]}
        (s / "scripted_baseline.json").write_text(json.dumps(rows, indent=1))
        for cond in ("plain", "camo"):
            sel = [r for r in rows.values() if r["condition"] == cond]
            print(f"{s.name:20s} {cond:5s} correct {sum(r['correct'] for r in sel)}/{len(sel)}   near (post hoc) "
                  f"{sum(r['pred_near'] == r['truth'] for r in sel)}/{len(sel)}   locked rule: right "
                  f"{sum(r['pred_locked'] == r['truth'] for r in sel)}, unclear "
                  f"{sum(r['pred_locked'] == 'unclear' for r in sel)}, wrong "
                  f"{sum(r['pred_locked'] not in (r['truth'], 'unclear') for r in sel)}")
        for g in sorted({r["game"] for r in rows.values()}):
            sel = [r for r in rows.values() if r["game"] == g]
            print(f"   {g:9s} truth {STANCE[g]:8s} scores " +
                  " ".join(f"{r['score_all']:+.3f}{'' if r['correct'] else '(x)'}" for r in sel) + "  | near " +
                  " ".join(f"{r['score_near']:+.3f}{'' if r['pred_near'] == r['truth'] else '(x)'}" for r in sel))


if __name__ == "__main__":
    main()
