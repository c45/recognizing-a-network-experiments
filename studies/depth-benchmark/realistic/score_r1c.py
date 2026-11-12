"""Score the R1c paired-mirror test for one or more arms.

Usage: python3 -I score_r1c.py [tool-free label ...]   (default: astra_xhigh; the with-Python arms are read from
analyst/runs/r1c_<id>/final_answer.md, reported as "python" (GPT-6 Astra), and analyst/runs/opusx_r1c_<id>/, reported
as "opus_python" (Claude Opus 5.5);
only finished conversations are counted). A conversation listed for its arm in
readers_r1c/rulings.json was ruled a violation of the computing rules and is scored as unanswered.
For each arm: exact / condition / pair accuracy; orientation accuracy on trained vs mirror networks (among replies with
the right pair and condition); and the pre-registered pair classification: structure-consistent pairs that are
"opposite" (orientations differ) or "same" (orientations agree). Also agreement of answered orientations with the
trained-pair B2 classifier's leave-one-group-out prediction (mirrors inherit their source's prediction).
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANS = re.compile(r"answer\s*:?\s*\**\s*a\s*=\s*([+-]?\d)\s*,?\s*b\s*=\s*([+-]?\d)\s*,?\s*C\s*=\s*(C[1-4])", re.I)
INFRA = re.compile(r"hit your|reached your [A-Za-z0-9. ]*limit|Can't reach|Connection refused|at capacity|Not logged in|usage limit")


def parse(text):
    text = text.replace("−", "-").replace("–", "-")
    m = ANS.findall(text)
    return (int(m[-1][0]), int(m[-1][1]), m[-1][2].upper()) if m else None


def b2_predictions():
    """Leave-one-group-out B2 orientation predictions on the trained pool (as in baselines_r1.json)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0, str(HERE))
    from baselines_r1 import features
    files, metas = [], []
    for f in sorted((HERE / "nets").glob("r_*.npz")):
        m = json.loads(f.with_suffix(".json").read_text())
        if min(m["heldout_acc"].values()) >= 0.95:
            files.append(f)
            metas.append(m)
    X = np.array([features(dict(np.load(f)))[1] for f in files])
    y = np.array([m["a"] > m["b"] for m in metas], int)
    g = np.array([f"{min(m['a'], m['b'])}_{max(m['a'], m['b'])}_{m['C']}" for m in metas])
    pred = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)), X, y,
                             cv=LeaveOneGroupOut(), groups=g)
    return {f.stem: int(p) for f, p in zip(files, pred)}


def main(labels):
    d = HERE / "readers_r1c"
    kj, pj = (d / "key.json").read_text(), (d / "pairs.json").read_text()
    assert hashlib.sha256((kj + pj).encode()).hexdigest() == (d / "PREREG_KEY_HASH.txt").read_text().split()[0]
    key, pairs = json.loads(kj), json.loads(pj)
    b2 = b2_predictions()
    arms = {lab: (lambda bid, lab=lab: d / "replies" / lab / bid / "answer.md") for lab in labels}
    arms["python"] = lambda bid: HERE.parent / "analyst" / "runs" / f"r1c_{bid}" / "final_answer.md"
    if any((HERE.parent / "analyst" / "runs").glob("opusx_r1c_*")):   # Claude Opus 5.5 with Python (Amendment 5)
        arms["opus_python"] = lambda bid: HERE.parent / "analyst" / "runs" / f"opusx_r1c_{bid}" / "final_answer.md"
    rulings = json.loads((d / "rulings.json").read_text()) if (d / "rulings.json").exists() else {}
    res = {}
    for arm, path in arms.items():
        ans, ruled = {}, 0
        for bid in key:
            f = path(bid)
            if f.exists() and not INFRA.search(f.read_text()):
                ans[bid] = parse(f.read_text())
                if bid in rulings.get(arm, {}):   # ruled a violation of the computing rules: scored as unanswered
                    ans[bid], ruled = None, ruled + 1
        rows = {"answered": sum(v is not None for v in ans.values()), "n": len(ans), "ruled_violations": ruled}
        ex = co = pa = 0
        orient = {"trained": [0, 0], "mirror": [0, 0]}
        cue_agree = truth_agree = cue_n = 0
        for bid, a in ans.items():
            if a is None:
                continue
            _, ta, tb, tC, s = key[bid].split("_")
            ta, tb = int(ta), int(tb)
            ex += a == (ta, tb, tC)
            co += a[2] == tC
            pa += {a[0], a[1]} == {ta, tb}
            src = key[bid] if key[bid].startswith("r_") else next(k for k, ids in pairs.items() if bid in ids)
            if a[2] == tC and {a[0], a[1]} == {ta, tb}:
                kind = "trained" if key[bid].startswith("r_") else "mirror"
                orient[kind][1] += 1
                orient[kind][0] += a == (ta, tb, tC)
            if a[0] != a[1]:
                cue_n += 1
                cue_agree += int(a[0] > a[1]) == b2[src]
                truth_agree += int(a[0] > a[1]) == int(ta > tb)
        opposite = same = 0
        for src, (i, j) in pairs.items():
            ai, aj = ans.get(i), ans.get(j)
            if ai and aj and {ai[0], ai[1]} == {aj[0], aj[1]} and ai[2] == aj[2] and ai[0] != ai[1]:
                if (ai[0], ai[1]) == (aj[0], aj[1]):
                    same += 1
                else:
                    opposite += 1
        rows.update(exact=ex, condition=co, pair=pa,
                    orientation_trained=f"{orient['trained'][0]}/{orient['trained'][1]}",
                    orientation_mirror=f"{orient['mirror'][0]}/{orient['mirror'][1]}",
                    structure_consistent_pairs=opposite + same, opposite=opposite, same=same,
                    orientation_agrees_with_B2_cue=f"{cue_agree}/{cue_n}", orientation_agrees_with_truth=f"{truth_agree}/{cue_n}")
        res[arm] = rows
        print(arm, rows)
    (d / "score_r1c.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["astra_xhigh"])
