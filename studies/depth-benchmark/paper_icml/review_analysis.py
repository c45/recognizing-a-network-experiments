"""Post hoc analyses for the paper, recomputed from saved prompts, replies and weights.

Usage: python3 -I review_analysis.py (after threshold/analyze.py and threshold/twopager_analysis.py)
Needs the trained networks described in DATA.md. Writes review_analysis.json. Makes no reader calls.
Paired comparisons report discordant counts with a nominal exact two-sided McNemar p, uncorrected for
multiplicity or shared initialization seeds.
"""
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "threshold"))
import analyze  # noqa: E402
from twopager_analysis import DEPTH_DIRS, full_readout, linear_map_readout, path_readout, q_units  # noqa: E402

READERS = ("astra_xhigh", "opus_xhigh")
TOOLFREE_GRID = (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16)  # threshold/PREREG.md, Amendments 2 and 9
# Amendment 9 gave Opus no second set at depths 14 and 16; the update to Amendment 10 has both models read every set.
NOT_PLANNED = {}
INFRA = re.compile(r"hit your|reached your [A-Za-z0-9. ]*limit|Can't reach|Connection refused|at capacity|Not logged in|usage limit")
BASELINES = ("B1_token_footprint", "B2_order_free", "B3_position_footprint", "B4_single_layer_circuits")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def interval(k, n):
    ci = binomtest(k, n).proportion_ci()
    return {"correct": int(k), "n": int(n), "ci95": [float(ci.low), float(ci.high)]}


def paired(a, b):
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    p = binomtest(only_a, only_a + only_b, .5).pvalue if only_a + only_b else 1.0
    return {"n": len(a), "a_correct": int(sum(a)), "b_correct": int(sum(b)), "a_only": only_a, "b_only": only_b,
            "p_mcnemar_two_sided": float(p)}


def prompt_matrix(text, name):
    block = next(part for part in text.split("\n\n") if part.startswith(name + " ("))
    return np.array([list(map(float, row.split())) for row in block.splitlines()[1:]])


def score(embedding, query, key):
    direct = (embedding @ query) @ (embedding @ key).T
    diagonal = direct.diagonal()
    return float(diagonal[1] - diagonal[2]), float(diagonal[1::2].mean() - diagonal[::2].mean())


def transformer_replies(folder, key):
    """Per-network correctness of each reader's tool-free reply; unanswered counts as wrong."""
    ans = re.compile(r"answer\s*:?\s*\**\s*(even-next|even-prev)", re.I)
    out = {}
    for reader in READERS:
        rows = {}
        for bid, name in key.items():
            path = folder / "replies" / reader / bid / "answer.md"
            if not path.exists() or INFRA.search(path.read_text()):
                continue
            found = ans.findall(path.read_text())
            rows[bid] = bool(found) and found[-1].lower() == name.rsplit("_s", 1)[0].replace("_", "-")
        out[reader] = rows
    return out


def transformer_family(family, names, prompt_names):
    folder = ROOT / "transformer" / f"readers_{family}"
    assert sha(folder / "key.json") == (folder / "PREREG_KEY_HASH.txt").read_text().split()[0]
    key = json.loads((folder / "key.json").read_text())
    rows = []
    for bid, name in key.items():
        path = folder / "prompts" / f"{bid}.txt"
        matrices = [prompt_matrix(path.read_text(), k) for k in prompt_names]
        two, parity = score(*matrices)
        truth = name.startswith("even_next")
        rows.append({"id": bid, "network": name, "two_token_score": two, "parity_score": parity,
                     "correct": bool((two > 0) == truth), "parity_correct": bool((parity > 0) == truth),
                     "prompt_sha256": sha(path)})
    pool = []
    for path in sorted((ROOT / "transformer" / f"nets_{family}").glob("*.npz")):
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["heldout_acc"] < 0.95:
            continue
        with np.load(path) as weights:
            two, parity = score(*(weights[k].astype(float) for k in names))
        truth = meta["rule"] == "evennext"
        pool.append({"network": path.stem, "correct": bool((two > 0) == truth),
                     "parity_correct": bool((parity > 0) == truth), "sha256": sha(path)})
    replies = transformer_replies(folder, key)
    shared = [bid for bid in key if all(bid in replies[r] for r in READERS)]
    return {"reader_n": len(rows), "reader_correct": sum(r["correct"] for r in rows),
            "pool_n": len(pool), "pool_correct": sum(r["correct"] for r in pool),
            "astra_vs_opus": paired([replies["astra_xhigh"][b] for b in shared],
                                    [replies["opus_xhigh"][b] for b in shared]),
            "reader_rows": rows, "pool_rows": pool}


def with_python_answer(depth, bid, name, prefix):
    """Correctness of one with-Python run, following threshold/analyze.py; None if it is not counted."""
    run = ROOT / "analyst" / "runs" / f"{prefix}d{depth}_{bid}"
    state = json.loads((run / "state.json").read_text())
    if not state.get("done"):
        return False if state.get("error") in ("output overflow", "empty answer") else None
    return bool(analyze.score((run / "final_answer.md").read_text(), name)[0])


def matched_scripts():
    results = {}
    for depth in (4, 8, 12, 16, 20):
        key = json.loads((ROOT / "analyst" / f"key_d{depth}.json").read_text())
        rows = []
        for bid, name in key.items():
            model = ROOT / "analyst" / "models" / bid / "blocks.npz"
            with np.load(model) as z:
                weights = dict(z)
            truth = name.startswith("right_toward")
            script_scores = {"curvature": full_readout(weights), "three_paths": path_readout(weights),
                             "ball_slope": linear_map_readout(weights)}
            product = weights["W1"].astype(float)
            for i in range(2, depth + 2):
                product = product @ weights[f"W{i}"].astype(float)
            equivalent = float(q_units(product[:, [4, 3]]) @ np.array([1., -1.]))
            assert np.isclose(equivalent, script_scores["curvature"], rtol=1e-10, atol=1e-8)
            answer = ROOT / "analyst" / "runs" / f"d{depth}_{bid}" / "final_answer.md"
            rows.append({"id": bid, "network": name, "scores": script_scores,
                         **{k: bool((v > 0) == truth) for k, v in script_scores.items()},
                         "astra": with_python_answer(depth, bid, name, ""),
                         "opus": with_python_answer(depth, bid, name, "opusx_"),
                         "weights_sha256": sha(model), "answer_sha256": sha(answer)})
        assert all(r["astra"] is not None and r["opus"] is not None for r in rows), depth
        summary = {"n": len(rows), "rows": rows}
        for method in ("curvature", "three_paths", "ball_slope", "astra", "opus"):
            summary[method] = interval(sum(r[method] for r in rows), len(rows))
        summary["script_only"] = sum(r["curvature"] and not r["astra"] for r in rows)
        summary["astra_only"] = sum(r["astra"] and not r["curvature"] for r in rows)
        summary["script_only_opus"] = sum(r["curvature"] and not r["opus"] for r in rows)
        summary["opus_only"] = sum(r["opus"] and not r["curvature"] for r in rows)
        results[depth] = summary
    rows = [r for v in results.values() for r in v["rows"]]
    totals = {"script_astra": paired([r["curvature"] for r in rows], [r["astra"] for r in rows]),
              "script_opus": paired([r["curvature"] for r in rows], [r["opus"] for r in rows]),
              "astra_opus": paired([r["astra"] for r in rows], [r["opus"] for r in rows])}
    return results, totals


def toolfree():
    """Planned versus saved tool-free replies, and reader and script accuracy on the same reader-set networks."""
    inventory = {r: {"planned": 0, "saved": 0, "missing": []} for r in READERS}
    unanswered = {r: [] for r in READERS}
    by_depth = {}
    for depth in TOOLFREE_GRID:
        rows = []
        for relative in analyze.TOOLFREE[depth]:
            folder = ROOT / relative
            if not (folder / "key.json").exists():
                continue
            for bid, name in analyze.check_key(folder).items():
                with np.load(DEPTH_DIRS[depth] / f"{name}.npz") as z:
                    weights = dict(z)
                truth = name.startswith("right_toward")
                row = {"set": relative, "id": bid, "network": name,
                       "curvature": bool((full_readout(weights) > 0) == truth),
                       "three_paths": bool((path_readout(weights) > 0) == truth)}
                for reader in READERS:
                    if relative in NOT_PLANNED.get(reader, ()):
                        row[reader] = None
                        continue
                    inventory[reader]["planned"] += 1
                    path = folder / "replies" / reader / bid / "answer.md"
                    if not path.exists() or INFRA.search(path.read_text()):
                        inventory[reader]["missing"].append({"depth": depth, "set": relative, "id": bid})
                        row[reader] = None
                        continue
                    inventory[reader]["saved"] += 1
                    text = path.read_text()
                    row[reader] = bool(analyze.score(text, name)[0])
                    if analyze.score(text, name)[0] is None:
                        kind = ("output_limit" if "output token maximum" in text else
                                "withheld" if re.search(r"safety|reasoning_extraction", text) else "other")
                        unanswered[reader].append({"depth": depth, "set": relative, "id": bid, "kind": kind})
                rows.append(row)
        entry = {"n": len(rows), "rows": rows,
                 **{m: interval(sum(r[m] for r in rows), len(rows)) for m in ("curvature", "three_paths")}}
        for reader in READERS:
            mine = [r for r in rows if r[reader] is not None]
            entry[f"{reader}_vs_three_paths"] = paired([r[reader] for r in mine], [r["three_paths"] for r in mine])
        both = [r for r in rows if all(r[x] is not None for x in READERS)]
        entry["astra_vs_opus"] = paired([r["astra_xhigh"] for r in both], [r["opus_xhigh"] for r in both])
        by_depth[depth] = entry
    for reader in READERS:
        missing = inventory[reader]["missing"]
        inventory[reader]["missing_by_set"] = {s: sum(m["set"] == s for m in missing)
                                               for s in sorted({m["set"] for m in missing})}
        inventory[reader]["unanswered"] = unanswered[reader]
        inventory[reader]["unanswered_by_kind"] = {k: sum(u["kind"] == k for u in unanswered[reader])
                                                   for k in sorted({u["kind"] for u in unanswered[reader]})}
        inventory[reader]["unanswered_by_depth"] = {d: sum(u["depth"] == d for u in unanswered[reader])
                                                    for d in sorted({u["depth"] for u in unanswered[reader]})}
    both = [r for v in by_depth.values() for r in v["rows"] if all(r[x] is not None for x in READERS)]
    return {"grid": list(TOOLFREE_GRID), "inventory": inventory, "by_depth": by_depth,
            "astra_vs_opus": paired([r["astra_xhigh"] for r in both], [r["opus_xhigh"] for r in both])}


def deep_confidence(depths=(14, 16)):
    """Post hoc: accuracy of the tool-free replies at the deepest depths, split by the confidence each reply states."""
    out = {}
    for reader in READERS:
        bands = {b: {"n": 0, "correct": 0, "right_away": 0} for b in ("even", "between", "committed")}
        unanswered = 0
        for depth in depths:
            for relative in analyze.TOOLFREE[depth]:
                folder = ROOT / relative
                if not (folder / "key.json").exists() or relative in NOT_PLANNED.get(reader, ()):
                    continue
                for bid, name in analyze.check_key(folder).items():
                    path = folder / "replies" / reader / bid / "answer.md"
                    if not path.exists() or INFRA.search(path.read_text()):
                        continue
                    text = path.read_text()
                    answers = analyze.ANS.findall(text)
                    if not answers:
                        unanswered += 1
                        continue
                    stated = analyze.CONF.findall(text)
                    conf = int(stated[-1]) if stated else 50
                    band = bands["even" if conf <= 50 else "between" if conf < 60 else "committed"]
                    band["n"] += 1
                    band["correct"] += bool(analyze.score(text, name)[0])
                    band["right_away"] += answers[-1].lower() == "right-away"
        out[reader] = {"depths": list(depths), "unanswered": unanswered, **bands}
    return out


def r1b():
    """Reader components and the four R1b baselines on the same 16 reader-set networks."""
    scorer = module("r1_scorer", ROOT / "realistic/score_r1.py")
    baseline = module("r1_baselines", ROOT / "realistic/baselines_r1.py")
    folder = ROOT / "realistic/readers_r1b"
    assert sha(folder / "key.json") == (folder / "PREREG_KEY_HASH.txt").read_text().split()[0]
    key = json.loads((folder / "key.json").read_text())
    readers = {}
    for reader in READERS:
        rows = []
        for bid, name in key.items():
            a, b, condition = name.split("_")[1:4]
            a, b = int(a), int(b)
            path = folder / "replies" / reader / bid / "answer.md"
            text = path.read_text().replace("−", "-").replace("–", "-") if path.exists() else ""
            found = scorer.ANS.findall(text)
            pred = [int(found[-1][0]), int(found[-1][1]), found[-1][2].upper()] if found else None
            right_condition = pred is not None and pred[2] == condition
            right_pair = pred is not None and set(pred[:2]) == {a, b}
            rows.append({"id": bid, "network": name, "predicted": pred, "truth": [a, b, condition],
                         "exact": pred == [a, b, condition], "condition": right_condition,
                         "offset_pair": right_pair, "eligible_orientation": right_condition and right_pair})
        readers[reader] = rows
    files, metas = [], []
    for path in sorted((ROOT / "realistic/nets_r1b").glob("*.npz")):
        meta = json.loads(path.with_suffix(".json").read_text())
        if min(meta["heldout_acc"].values()) >= .95:
            files.append(path)
            metas.append(meta)
    features = list(zip(*[baseline.features(dict(np.load(path))) for path in files]))
    groups = np.array([f"{min(m['a'], m['b'])}_{max(m['a'], m['b'])}_{m['C']}" for m in metas])
    targets = {"orientation": np.array([int(m["a"] > m["b"]) for m in metas]),
               "condition": np.array([m["C"] for m in metas]),
               "offset_pair": np.array([f"{min(m['a'], m['b'])}_{max(m['a'], m['b'])}" for m in metas])}
    lookup = {p.stem: i for i, p in enumerate(files)}
    selected = np.array([lookup[name] for name in key.values()])
    astra = readers["astra_xhigh"]
    eligible = np.array([lookup[r["network"]] for r in astra if r["eligible_orientation"]])
    published = json.loads((ROOT / "realistic/baselines_r1b.json").read_text())
    baselines = {}
    for target, y in targets.items():
        baselines[target] = {}
        for method, X in zip(BASELINES, features):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
            correct = cross_val_predict(clf, np.array(X), y, cv=LeaveOneGroupOut(), groups=groups) == y
            assert int(correct.sum()) == published[target][method]["correct"], (target, method)
            entry = {"pool_correct": int(correct.sum()), "pool_n": len(y),
                     "matched_correct": int(correct[selected].sum()), "matched_n": len(selected)}
            if target == "orientation":
                entry["astra_eligible_correct"] = int(correct[eligible].sum())
                entry["astra_eligible_n"] = len(eligible)
            else:
                entry["astra_paired"] = paired([r[target] for r in astra], [bool(v) for v in correct[selected]])
            baselines[target][method] = entry
    summary = {}
    for reader, rows in readers.items():
        k = sum(r["exact"] for r in rows if r["eligible_orientation"])
        n = sum(r["eligible_orientation"] for r in rows)
        summary[reader] = {"n": len(rows), **{t: sum(r[t] for r in rows) for t in ("exact", "condition", "offset_pair")},
                           "orientation": {"correct": k, "n": n,
                                           **({"ci95": interval(k, n)["ci95"],
                                               "p_greater_than_half": float(binomtest(k, n, .5, alternative="greater").pvalue)}
                                              if n else {})}}
    summary["opus_xhigh"]["p_condition_vs_quarter"] = float(
        binomtest(summary["opus_xhigh"]["condition"], len(key), .25, alternative="greater").pvalue)
    return {"readers": summary, "baselines": baselines, "reader_rows": readers}


def main():
    prompt = ROOT / "transformer/readers_t2/prompts/2881672b.txt"
    answer = ROOT / "transformer/readers_t2/replies/astra_xhigh/2881672b/answer.md"
    embedding, query, key = [prompt_matrix(prompt.read_text(), k) for k in ("E", "Q2", "K2")]
    direct = (embedding @ query) @ (embedding @ key).T
    estimates = re.search(r"D_\{11\}\\approx(\d+).*?D_\{12\}\\approx(-?\d+).*?"
                          r"D_\{22\}\\approx(\d+).*?D_\{21\}\\approx(-?\d+)", answer.read_text(), re.S)
    if estimates is None:
        raise ValueError("The retained Astra arithmetic example no longer matches the expected format.")
    positions = [(1, 1), (1, 2), (2, 2), (2, 1)]
    arithmetic = {"id": "2881672b", "prompt_sha256": sha(prompt), "answer_sha256": sha(answer),
                  "rows": [{"a": a, "b": b, "astra": int(estimate), "recomputed": float(direct[a, b])}
                           for (a, b), estimate in zip(positions, estimates.groups())]}
    z = np.load(ROOT / "transformer/data/seqs.npz")
    symmetry = {}
    for split in (0, 1):
        x = z["x"][z["split"] == split]
        members = {row.tobytes() for row in x}
        symmetry[split] = {"n": len(x), "partner_present": sum((row ^ 1).tobytes() in members for row in x)}
    d24 = [json.loads(p.read_text()) for p in (ROOT / "ladder/scale/nets_d24").glob("*.json")]
    matched, matched_totals = matched_scripts()
    result = {"analysis_kind": "post_hoc", "mlp_analysis": "threshold/breaking_depth.json from threshold/analyze.py",
              "arithmetic": arithmetic, "matched": matched, "matched_totals": matched_totals,
              "toolfree": toolfree(), "toolfree_deep_confidence": deep_confidence(), "symmetry": symmetry,
              "depth24": {"trained": len(d24), "qualified": sum(r["heldout_acc"] >= .95 for r in d24)},
              "t2": transformer_family("t2", ("E", "Q2", "K2"), ("E", "Q2", "K2")),
              "g2": transformer_family("g2", ("E", "Wq2", "Wk2"), ("E", "Wq_2", "Wk_2")),
              "r1b": r1b()}
    (ROOT / "paper_icml/review_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"arithmetic": arithmetic["rows"], "depth24": result["depth24"], "symmetry": symmetry,
                      "matched_totals": matched_totals,
                      "toolfree_saved": {r: (v["saved"], v["planned"]) for r, v in result["toolfree"]["inventory"].items()},
                      "r1b": result["r1b"]["readers"]}, indent=2))


if __name__ == "__main__":
    main()
