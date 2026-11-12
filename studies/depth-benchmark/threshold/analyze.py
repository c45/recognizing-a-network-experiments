"""Breaking depth per reader x arm (threshold/PREREG.md): per-depth pass/fail, fitted D50 with bootstrap intervals,
and the two figures (tool-free, with tools).

Usage: python3 -I analyze.py <out dir>
Tool-free sets: ladder/l2/readers_d (depth 2) and ladder/scale/readers_d<D>[b]; the answer is parsed from each
replies/<reader>/<id>/answer.md. Primary arms are astra_xhigh / opus_xhigh; astra (high) and opus (max)
are reported separately. With tools: analyst/runs/<prefix>d<D>_<id>/final_answer.md, prefix "" for Astra, "opusx_" for
Opus; a run with state.json done=false and error "output overflow" or "empty answer" is unanswered.
Unanswered replies count as wrong (primary) or are dropped (sensitivity). Replies still running or that failed for
service reasons are not counted.
"""
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binomtest

R = Path(__file__).resolve().parents[1]
N_BOOT = 2000
ANS = re.compile(r"answer\s*:?\s*\**\s*(right-toward|right-away)", re.I)
TOOLFREE = {2: ["ladder/l2/readers_d"]}
for d in (3, 4, 5, 6, 7, 8, 10, 12, 14, 16):
    TOOLFREE[d] = [f"ladder/scale/readers_d{d}{sfx}" for sfx in ("", "b", "c", "d", "e")]
CONF = re.compile(r"confidence\s*:?\s*\**\s*(\d{1,3})\s*%", re.I)


def score(text, name):
    """(correct or None if unanswered, probability placed on the correct answer)."""
    a = ANS.findall(text)
    if not a:
        return None, 0.5
    c = CONF.findall(text)
    conf = min(max(int(c[-1]), 50), 100) / 100 if c else 0.5
    ok = a[-1].lower() == truth(name)
    return ok, conf if ok else 1 - conf


def truth(name):
    return name.rsplit("_s", 1)[0].replace("_", "-")


def check_key(d):
    raw = (d / "key.json").read_text()
    assert hashlib.sha256(raw.encode()).hexdigest() == (d / "PREREG_KEY_HASH.txt").read_text().split()[0], d
    return json.loads(raw)


def toolfree():
    out = defaultdict(list)
    for depth, sets in TOOLFREE.items():
        for s in sets:
            d = R / s
            if not (d / "key.json").exists():
                continue
            key = check_key(d)
            for reader in ("astra_xhigh", "opus_xhigh", "astra", "opus"):
                for bid, name in key.items():
                    f = d / "replies" / reader / bid / "answer.md"
                    if not f.exists():
                        continue
                    text = f.read_text()
                    if re.search(r"hit your|reached your [A-Za-z0-9. ]*limit|Can't reach|Connection refused|at capacity|Not logged in|usage limit", text):
                        continue
                    out[(reader, "tool-free")].append((depth, *score(text, name)))
    return out


def with_tools():
    out = defaultdict(list)
    A = R / "analyst"
    for kf in sorted(A.glob("key_d*.json")):
        depth = int(re.match(r"key_d(\d+)", kf.name).group(1))
        key = json.loads(kf.read_text())
        for reader, prefix in (("astra_xhigh", ""), ("opus_xhigh", "opusx_")):
            for bid, name in key.items():
                rd = A / "runs" / f"{prefix}d{depth}_{bid}"
                st = rd / "state.json"
                if not st.exists():
                    continue
                state = json.loads(st.read_text())
                if not state.get("done"):
                    if state.get("error") in ("output overflow", "empty answer"):
                        out[(reader, "with tools")].append((depth, None, 0.5))
                    continue
                out[(reader, "with tools")].append((depth, *score((rd / "final_answer.md").read_text(), name)))
    return out


def fit_d50(depths, y, start=None):
    def nll(p):
        b, d50 = p
        q = 0.5 + 0.5 / (1 + np.exp(np.clip(b * (depths - d50), -50, 50)))
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))
    starts = [start] if start is not None else [[b0, d0] for b0 in (0.5, 1.0, 2.0) for d0 in (4, 8, 12, 16)]
    best = min((minimize(nll, s0, method="Nelder-Mead") for s0 in starts), key=lambda r: r.fun)
    return best.x


def summarize(rows, rng):
    by = defaultdict(list)
    for d, c, p in rows:
        by[d].append((c, p))
    per = {}
    rules = {12: (10, 8), 24: (18, 15), 48: (33, 29)}
    for d in sorted(by):
        cs = [c for c, _ in by[d]]
        ps = np.array([p for _, p in by[d]])
        primary = [bool(c) for c in cs]
        answered = [c for c in cs if c is not None]
        k, n = sum(primary), len(primary)
        lo, hi = binomtest(k, n).proportion_ci(0.95) if n else (None, None)
        if n in rules:
            verdict = "pass" if k >= rules[n][0] else "fail" if k <= rules[n][1] else "inconclusive"
        else:
            verdict = "partial"
        boot = [rng.choice(ps, len(ps)).mean() for _ in range(2000)]
        per[d] = {"correct": k, "n": n, "unanswered": sum(c is None for c in cs), "ci95": [round(lo, 3), round(hi, 3)] if n else None,
                  "answered_correct": sum(answered), "answered_n": len(answered), "verdict": verdict,
                  "mean_p_correct": round(float(ps.mean()), 3),
                  "mean_p_ci95": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
                  "brier": round(float(((1 - ps) ** 2).mean()), 4)}
    depths = np.array([d for d, c, p in rows], float)
    y = np.array([bool(c) for d, c, p in rows], float)
    res = {"per_depth": per}
    if len(set(depths)) >= 2 and 0 < y.mean() < 1:
        b, d50 = fit_d50(depths, y)
        boots = []
        for _ in range(N_BOOT):
            idx = np.concatenate([rng.choice(np.where(depths == d)[0], int((depths == d).sum())) for d in np.unique(depths)])
            bb, dd = fit_d50(depths[idx], y[idx], start=[b, d50])
            boots.append(dd)
        res.update(slope=round(float(b), 3), D50=round(float(d50), 2),
                   D50_ci95=[round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)])
    passes = [d for d, v in per.items() if v["verdict"] == "pass"]
    fails = [d for d, v in per.items() if v["verdict"] == "fail"]
    first_fail = min((d for d in fails if all(per[e]["verdict"] == "fail" for e in per if e > d)), default=None)
    res["deepest_pass"] = max((d for d in passes if first_fail is None or d < first_fail), default=None)
    res["breaks_at"] = first_fail
    return res


COMMON = ("Each network is a tanh MLP (500 inputs, D hidden layers of 32 units, 5 action logits) trained by imitation "
          "to play a Breakout paddle under one of two mirror-image rules: when the ball moves right, the paddle moves "
          "toward the ball's column (right-toward) or away from it (right-away). Both classes are trained on the same "
          "mirror-balanced states and differ only in their labels, so weight statistics that ignore the rule (row norms, "
          "order-free statistics, the end-to-end linear map) classify at chance; reader sets are counterbalanced on the "
          "output-bias sign. Per depth, every model sees the same networks under blinded IDs and answers right-toward or "
          "right-away with a confidence (chance = 0.5). All run at reasoning effort xhigh. ")
ARM_TEXT = {
    "tool-free": ("Figure 1. Depth at which frontier models stop reading a trained network's rule from its weights, "
                  "without tools. ", "The model receives the architecture, the observation and action encoding, and every "
                  "weight as text (3 decimals), and replies once with no code execution. "),
    "with tools": ("Figure 2. Depth at which frontier models stop reading a trained network's rule from its weights, "
                   "with a Python sandbox. ", "The model reads the float32 weights only through Python (numpy, scipy, "
                   "PyTorch; up to 100 runs, one code block per turn) and may use any linear algebra on them, but no "
                   "forward pass: running the network or any layer on any input is forbidden. "),
}
SUPTITLE = {"tool-free": "Reading a network's rule from its weights alone (no tools)",
            "with tools": "Reading a network's rule from its weights with Python (no forward pass)"}


def caption(arm, series_counts):
    head, setup = ARM_TEXT[arm]
    prog = "; ".join(f"{n} {g}/{w}" for n, g, w in series_counts)
    return (head + setup + COMMON +
            "(a) Fraction correct per depth (unanswered replies count as wrong); numbers under the points are "
            "correct/n per depth. Curves are logistic fits that fall from 1 to chance; D50 is the depth at which the fit "
            "reaches 0.75 (95% bootstrap interval in the legend). (b) Partial credit: mean probability the model put on "
            "the correct rule (stated confidence if right, 1 - confidence if wrong; unanswered = 0.5). "
            f"Replies: {prog}.")


def figures(results, out):
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ASTRA, OPUS = "#d9480f", "#2a6fdb"
    planned_toolfree = {2: 12, 3: 12, 4: 24, 5: 12, 6: 48, 7: 48, 8: 48, 10: 48, 12: 12, 14: 12, 16: 12}
    series = {
        "tool-free": [("astra_xhigh", "GPT-6 Astra", ASTRA, -0.12), ("opus_xhigh", "Claude Opus 5.5", OPUS, 0.12)],
        "with tools": [("astra_xhigh", "GPT-6 Astra + Python", ASTRA, -0.15),
                       ("opus_xhigh", "Claude Opus 5.5 + Python", OPUS, 0.15)],
    }
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlelocation": "left"})
    captions = []
    for arm, fname in (("tool-free", "fig_breaking_depth_toolfree.png"), ("with tools", "fig_breaking_depth_tools.png")):
        fig, (ax, axp) = plt.subplots(2, 1, figsize=(7.6, 9.4), sharex=True,
                                      gridspec_kw={"height_ratios": [1.15, 1]})
        counts = []
        for si, (reader, name, color, off) in enumerate(series[arm]):
            r = results.get(f"{reader}|{arm}")
            want = sum(planned_toolfree.values()) if arm == "tool-free" else 12 * 5
            if arm == "tool-free" and reader == "astra_xhigh":
                want += 24 + 84  # Amendments 9 and 10: further sets at depths 12, 14 and 16, read by Astra only
            if not r or not r["per_depth"]:
                ax.plot([], [], color=color, marker="o", ls="none", label=f"{name}: not started")
                counts.append((name, 0, want))
                continue
            per = r["per_depth"]
            ds = sorted(per)
            acc = [per[d]["correct"] / per[d]["n"] for d in ds]
            got = sum(per[d]["n"] for d in ds)
            counts.append((name, got, want))
            label = name
            lo95, hi95 = r.get("D50_ci95", [None, None])
            degenerate = "D50" in r and (hi95 - lo95 < 1e-6)
            if "D50" in r and len(ds) >= 3 and not degenerate:
                if hi95 > 2 * max(ds):
                    label += f": no break up to {max(ds)} layers (D50 > {lo95:.1f})"
                else:
                    label += f": D50 = {r['D50']:.1f} layers [{lo95:.1f}, {hi95:.1f}]"
                xs = np.linspace(min(ds), max(ds), 200)
                ax.plot(xs, 0.5 + 0.5 / (1 + np.exp(r["slope"] * (xs - r["D50"]))), color=color, lw=1.4, alpha=0.6)
            elif got < want:
                label += ": too few depths to fit"
            ax.plot([d + off for d in ds], acc, color=color, marker="o", ms=6, ls="none", label=label)
            pm = [per[d]["mean_p_correct"] for d in ds]
            axp.plot([d + off for d in ds], pm, color=color, marker="o", ms=6, ls="-", lw=1.2)
            row = 0.13 - 0.05 * si
            for d in ds:
                ax.text(d, row, f"{per[d]['correct']}/{per[d]['n']}", ha="center", va="center", fontsize=7,
                        color=color)
        for a in (ax, axp):
            a.axhline(0.5, color="#8c959f", lw=0.9, ls="--")
            a.text(a.get_xlim()[1] if False else 1.0, 0.5, " chance", transform=a.get_yaxis_transform(),
                   va="center", ha="left", fontsize=7, color="#59636e")
        ax.axhline(0.75, color="#d0d7de", lw=0.8, ls=":")
        ax.set_ylim(0, 1.05)
        axp.set_ylim(0.3, 1.0)
        axp.set_xlabel("hidden layers D (width 32)")
        axp.set_ylabel("mean probability on the correct rule")
        axp.set_title("(b) Partial credit", fontsize=9.5, fontweight="bold")
        ax.set_ylabel("fraction correct")
        ax.set_title("(a) Accuracy", fontsize=9.5, fontweight="bold")
        ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.17))
        fig.suptitle(SUPTITLE[arm], fontsize=11.5, fontweight="bold", x=0.02, ha="left")
        cap = caption(arm, counts)
        captions.append(cap)
        fig.text(0.02, 0.01, "\n".join(textwrap.wrap(cap, 142)), fontsize=6.6, ha="left", va="bottom",
                 color="#24292f", linespacing=1.35)
        fig.tight_layout(rect=(0, 0.2, 1, 0.97))
        fig.savefig(out / fname, dpi=200)
        plt.close(fig)
    (out / "figure_captions.md").write_text("\n\n".join(captions) + "\n")


def main(out):
    out = Path(out)
    rng = np.random.default_rng(1)
    data = {**toolfree(), **with_tools()}
    results = {}
    for (reader, arm), rows in sorted(data.items()):
        results[f"{reader}|{arm}"] = summarize(rows, rng)
        sens = [(d, c, p) for d, c, p in rows if c is not None]
        results[f"{reader}|{arm}|answered-only"] = summarize(sens, rng)
        r = results[f"{reader}|{arm}"]
        print(f"{reader:6s} {arm:10s} breaks_at={r['breaks_at']} deepest_pass={r['deepest_pass']} "
              f"D50={r.get('D50')} {r.get('D50_ci95')}")
        for d, v in r["per_depth"].items():
            print(f"    depth {d:2d}: {v['correct']}/{v['n']} (unanswered {v['unanswered']}) {v['verdict']}")
    (out / "breaking_depth.json").write_text(json.dumps(results, indent=1))
    figures(results, out)


if __name__ == "__main__":
    main(sys.argv[1])
