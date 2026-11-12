"""Build the figure data for the post "Recognizing a network is not understanding it".

Usage: python3 -I make_website_data.py [output path]   (default: website_data.json next to this file)
Run make_blog_data.py first: the layout-study blocks are taken from intermediate/layout_blocks.json.

Inputs are result files only: depth-benchmark threshold/breaking_depth.json, paper_icml/review_analysis.json and
numbers.tex, realistic/readers_r1c (key, pairs, replies), the analyst runs r1c_<id>, and the layout study's
scores.json files and arch_stats.json. Readers are GPT-6 Astra and Claude Opus 5.5.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEPTH = HERE.parent / "studies" / "depth-benchmark"
LAYOUT = HERE.parent / "studies" / "layout-vs-behaviour" / "experiments"
READERS = {"astra": "astra_xhigh", "opus": "opus_xhigh"}
ARCHS = [("32-32 tanh", "tanh32x2", ["suite_v2"]), ("24-24-24 ReLU", "relu24x3", ["suite_relu24x3", "suite_relu24x3_bfix"]),
         ("48 tanh", "tanh48x1", ["suite_tanh48x1"])]
GAMES = ["pong", "breakout", "pacman", "freeway"]
ANS = re.compile(r"answer\s*:?\s*\**\s*a\s*=\s*([+-]?\d)\s*,?\s*b\s*=\s*([+-]?\d)\s*,?\s*C\s*=\s*(C[1-4])", re.I)


def load(p):
    return json.loads(Path(p).read_text())


def depth():
    bd = load(DEPTH / "threshold" / "breaking_depth.json")
    ra = load(DEPTH / "paper_icml" / "review_analysis.json")
    mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", (DEPTH / "paper_icml" / "numbers.tex").read_text(), re.M))
    words = {"Two": 2, "Three": 3, "Four": 4, "Five": 5, "Six": 6, "Seven": 7, "Eight": 8, "Ten": 10, "Twelve": 12,
             "Fourteen": 14, "Sixteen": 16, "Twenty": 20}

    def arm(label, name):
        a = bd[f"{label}|{name}"]
        return {"points": [{"d": int(d), "k": v["correct"], "n": v["n"], "unanswered": v["unanswered"],
                            "lo": v["ci95"][0], "hi": v["ci95"][1]} for d, v in sorted(a["per_depth"].items(), key=lambda x: int(x[0]))],
                "d50": a["D50"], "ci": a["D50_ci95"], "slope": a["slope"], "deepest_pass": a["deepest_pass"]}

    def script(prefix):
        return sorted(({"d": d, "p": float(mac[prefix + w].replace("\\%", "")) / 100} for w, d in words.items()
                       if prefix + w in mac), key=lambda x: x["d"])

    out = {"toolfree": {m: arm(lab, "tool-free") for m, lab in READERS.items()},
           "python": {m: arm(lab, "with tools") for m, lab in READERS.items()},
           "scripts": {"curvature": script("FullC"), "beam": script("ThreeP")},
           # by_depth splits the paired total for the Scoreboard figure; its rows add up to the total's counts.
           "paired_toolfree": {**ra["toolfree"]["astra_vs_opus"],
                               "by_depth": [{"d": int(d), "n": v["astra_vs_opus"]["n"], "astra": v["astra_vs_opus"]["a_correct"],
                                             "opus": v["astra_vs_opus"]["b_correct"]}
                                            for d, v in sorted(ra["toolfree"]["by_depth"].items(), key=lambda x: int(x[0]))]},
           "matched": [{"d": int(d), "n": v["n"], "astra": v["astra"]["correct"], "opus": v["opus"]["correct"],
                        "curvature": v["curvature"]["correct"], "beam": v["three_paths"]["correct"]}
                       for d, v in sorted(ra["matched"].items(), key=lambda x: int(x[0]))],
           "matched_totals": {k: {"a": v["a_correct"], "b": v["b_correct"], "n": v["n"]} for k, v in ra["matched_totals"].items()},
           "nets_total": int(mac["NetsTotal"].replace("{,}", "")),
           "opus_unanswered": {"total": int(mac["OpusTFUnans"]), "output_limit": int(mac["OpusTFOutputLimit"]),
                               "withheld": int(mac["OpusTFWithheld"])},
           "astra_missing": int(mac["AstraTFMissing"])}
    # Prompt size per depth for the no-tools chart; rebuild with make_prompt_tokens.py.
    out["prompt_tokens"] = [{k: row[k] for k in ("d", "astra", "opus")} for row in load(HERE / "prompt_tokens.json")["depth"]]
    d8 = ra["toolfree"]["by_depth"]["8"]["rows"]
    out["depth8_beam"] = {"n": len(d8), "beam": sum(r["three_paths"] for r in d8),
                          "astra": sum(bool(r.get("astra_xhigh")) for r in d8),
                          "opus": sum(bool(r.get("opus_xhigh")) for r in d8)}
    return out, ra


def transformers(ra):
    out = {}
    for fam in ("t2", "g2"):
        p = ra[fam]["astra_vs_opus"]
        out[fam] = {"astra": [p["a_correct"], p["n"]], "opus": [p["b_correct"], p["n"]],
                    "script_pool": [ra[fam]["pool_correct"], ra[fam]["pool_n"]]}
    out["arithmetic"] = [{"a": r["a"], "b": r["b"], "astra": r["astra"], "recomputed": round(r["recomputed"], 2)}
                         for r in ra["arithmetic"]["rows"]]
    out["arithmetic_reply"] = {"id": ra["arithmetic"]["id"], "confidence": stated_confidence(
        DEPTH / "transformer" / "readers_t2" / "replies" / "astra_xhigh" / ra["arithmetic"]["id"] / "answer.md")}
    return out


CONF = re.compile(r"confidence\s*:?\s*\**\s*(\d+)\s*%", re.I)


def stated_confidence(path):
    """The last 'Confidence: N%' a reply states, or None."""
    m = CONF.findall(path.read_text()) if path.exists() else []
    return int(m[-1]) if m else None


def r1b(ra):
    rows = {m: {r["id"]: r for r in ra["r1b"]["reader_rows"][lab]} for m, lab in READERS.items()}
    replies = DEPTH / "realistic" / "readers_r1b" / "replies"
    out = []
    for bid, a in rows["astra"].items():
        o = rows["opus"][bid]
        out.append({"id": bid, "mirror": a["network"].startswith("m_"), "truth": a["truth"],
                    "astra": a["predicted"], "opus": o["predicted"],
                    "conf": {m: stated_confidence(replies / lab / bid / "answer.md") for m, lab in READERS.items()}})
    tot = {m: {k: ra["r1b"]["readers"][lab][k] for k in ("exact", "condition", "offset_pair")} for m, lab in READERS.items()}
    return {"rows": out, "totals": tot}


def parse(text):
    m = ANS.findall(text.replace("−", "-").replace("–", "-"))
    return [int(m[-1][0]), int(m[-1][1]), m[-1][2].upper()] if m else None


def r1c():
    d = DEPTH / "realistic" / "readers_r1c"
    key, pairs, score = load(d / "key.json"), load(d / "pairs.json"), load(d / "score_r1c.json")

    def answer(path):
        return parse(path.read_text()) if path.exists() else None

    def member(bid):
        _, a, b, c, _ = key[bid].split("_")
        return {"id": bid, "mirror": key[bid].startswith("m_"), "truth": [int(a), int(b), c],
                "toolfree": answer(d / "replies" / "astra_xhigh" / bid / "answer.md"),
                "python": answer(DEPTH / "analyst" / "runs" / f"r1c_{bid}" / "final_answer.md")}

    out = []
    for src, ids in sorted(pairs.items()):
        ms = sorted((member(i) for i in ids), key=lambda m: m["mirror"])
        out.append({"source": src, "trained": ms[0], "mirror": ms[1]})
    return {"pairs": out, "totals": {"toolfree": score["astra_xhigh"], "python": score["python"]}}


def naming(blog):
    st = load(LAYOUT / "arch_stats.json")
    me = load(LAYOUT / "matched_effort_scores.json")
    matched = {r["id"]: r for r in me["rows"]}
    cells = []
    for label, key, suites in ARCHS:
        rows = load(LAYOUT / suites[0] / "scores.json")
        if len(suites) > 1:  # ReLU Breakout camouflage was rebuilt (see arch_stats.py)
            rows = [r for r in rows if not (r["game"] == "breakout" and r["condition"] == "camo")]
            rows += load(LAYOUT / suites[1] / "scores.json")
        rows = [r for r in rows if r["arm"] == "toolfree" and r["condition"] == "camo"]
        for g in GAMES:
            cell = {"arch": label, "game": g}
            for m in ("astra", "opus"):
                rs = sorted((r for r in rows if r["model"] == m and r["game"] == g), key=lambda r: r["id"])
                cell[m] = [bool(r["game_correct"]) for r in rs]
                cell[m + "_said"] = [r["said_game"] for r in rs]
            # The matched-effort arm (MATCHED_EFFORT_RESULTS.md): Astra at max, and its concurrent rerun at high.
            ids = sorted(r["id"] for r in rows if r["model"] == "astra" and r["game"] == g)
            for arm in ("astra_max", "astra_high2"):
                cell[arm] = [bool(matched[i][arm]["correct"]) for i in ids]
                cell[arm + "_said"] = [matched[i][arm]["said"] for i in ids]
            assert cell["astra"] == [bool(matched[i]["astra_high"]["correct"]) for i in ids]
            cells.append(cell)
        assert f"{sum(sum(c['astra']) for c in cells if c['arch'] == label)}/16" == blog["naming"][key]["astra"]
        assert f"{sum(sum(c['opus']) for c in cells if c['arch'] == label)}/16" == blog["naming"][key]["opus"]
    camo, plain = me["camouflaged"], me["plain"]
    control = camo["high_control"]
    return {"cells": cells, "pooled": blog["naming_pooled"],
            "median_tokens": {a: {m: st[a]["median_output_tokens_camo"][m] for m in ("astra", "opus")}
                              for _, a, _ in ARCHS},
            # Astra rerun at Opus's effort setting (max), with a concurrent rerun at high as the control.
            "matched": {"n": camo["n"], "astra_max": camo["correct"]["astra_max"], "astra_high": camo["correct"]["astra_high"],
                        "astra_high2": camo["correct"]["astra_high2"], "opus_max": camo["correct"]["opus_max"],
                        "opus_only": camo["tests"]["opus_max_vs_astra_max"]["only_opus_max"],
                        "astra_only": camo["tests"]["opus_max_vs_astra_max"]["only_astra_max"],
                        "mcnemar_p": camo["tests"]["opus_max_vs_astra_max"]["mcnemar_p"],
                        "max_gained_over_high2": control["astra_max_vs_astra_high2"]["only_astra_max"],
                        "max_lost_to_high2": control["astra_max_vs_astra_high2"]["only_astra_high2"],
                        "high_runs_agree": control["agree"],
                        "policies_opus_ahead": camo["policy_level"]["opus_ahead"],
                        "policy_sign_p": camo["policy_level"]["sign_test_p"],
                        "without_tanh_breakout": [camo["without_tanh_breakout"]["correct"]["astra_max"],
                                                  camo["without_tanh_breakout"]["correct"]["opus_max"],
                                                  camo["without_tanh_breakout"]["n"]],
                        "astra_max_median_output_tokens": camo["astra_max_median_output_tokens"],
                        "plain": {k: plain["correct"][k] for k in ("astra_max", "astra_high", "opus_max")}}}


def twins(blog):
    def tally(rows):
        out = {}
        for r in rows:
            k, c = r["model"], "twin" if r["twin"] else "known"
            t = out.setdefault(k, {"known": [0, 0], "twin": [0, 0]})[c]
            t[0] += bool(r["correct"])
            t[1] += 1
        return out

    tw = [dict(r, correct=r["stance_correct"]) for r in blog["twins"] if r["model"] in ("astra", "opus")]
    probe = [r for r in blog["probe"] if r["suite"] == "suite_twins2" and r["game"] in ("dodge", "hunter", "cartag")]
    return {"open": tally(r for r in tw if r["arm"] == "toolfree"),
            "forced": tally(r for r in blog["bridge"]["fc"] if r["model"] in ("astra", "opus")),
            # Forced choice at matched effort (FC_PARITY_PREREG.md): Astra at max beside Opus (max, xhigh after a cap).
            "forced_matched": tally(dict(r, model=r["arm"].split("_")[0]) for r in
                                    load(LAYOUT / "suite_bridge_fc" / "fc_parity_scores.json")
                                    if r["arm"] in ("astra_max", "opus") and r["answered"]),
            "python": tally(r for r in tw if r["arm"] == "analyst"),
            "probe_heldout_twins": [sum(r["pred"] == r["truth"] for r in probe), len(probe)],
            "open_rule": {"astra": [sum(r["correct"] for r in blog["bridge"]["open"] if r["model"] == "astra"),
                                    sum(r["model"] == "astra" for r in blog["bridge"]["open"])]}}


def pictures():
    """Footprint maps of the four released networks (built by make_pictures.py; needs the upstream networks)."""
    return {g: {"maps": v["maps"], "camo": v["camo"]} for g, v in load(HERE / "pictures.json").items()}


def play():
    """Layer sizes and first printed weight rows of the four 32-32 tanh policies (play.json, built by make_play.py).
    The first figure runs the games live since kevin-website#15, so the recorded frames are no longer copied; its
    weight files and test trace come from make_live_play.py."""
    d = load(HERE / "play.json")
    return {"sizes": d["sizes"], "games": {g: {"rows": v["rows"]} for g, v in d["games"].items()}}


def ladder(blog):
    """The twin networks under each question. Astra's tool-free rows pool both training runs (the second run's
    pointed questions are in ladder2_scores.json, LADDER2_PREREG.md) and carry the split by run; Opus has the
    first run only. The two Python rows use the 24 camouflaged networks of both runs."""
    twin_games = ("dodge", "hunter", "cartag")

    def cell(rows, correct):
        out = {"twin": [0, 0], "known": [0, 0], "games": {g: [0, 0] for g in twin_games}}
        for r in rows:
            for t in (out["twin" if r["twin"] else "known"], out["games"].get(r["game"])):
                if t is not None:
                    t[0] += bool(r[correct])
                    t[1] += 1
        return out

    def by_model(rows, correct):
        return {m: cell([r for r in rows if r["model"] == m], correct) for m in ("astra", "opus")
                if any(r["model"] == m for r in rows)}

    first = lambda p: [r for r in load(p) if r["arm"] == "toolfree"]  # noqa: E731
    second = load(LAYOUT / "ladder2_scores.json")
    run2 = lambda arm: [dict(r, model="astra", correct=r["stance_correct"]) for r in second if r["arm"] == arm]  # noqa: E731
    open2 = [r for r in first(LAYOUT / "suite_twins2" / "twin_scores.json") if r["model"] == "astra"]
    steps = [("open", first(LAYOUT / "suite_twins" / "twin_scores.json"), open2, "stance_correct"),
             ("hint1", first(LAYOUT / "suite_twins_hint1" / "twin_scores.json"), run2("hint1_s2"), "stance_correct"),
             ("hint2", first(LAYOUT / "suite_twins_hint2" / "twin_scores.json"), run2("hint2_s2"), "stance_correct"),
             ("forced", load(LAYOUT / "suite_bridge_fc" / "bridge_scores.json"), run2("fc_s2"), "correct"),
             ("python", [r for r in blog["twins"] if r["arm"] == "analyst"], [], "stance_correct"),
             ("python_stance", [], run2("pystance"), "stance_correct")]
    out = []
    for name, run1, more, correct in steps:
        step = {"id": name, **by_model(run1 + more, correct)}
        if run1 and more:
            parts = (cell([r for r in run1 if r["model"] == "astra"], correct), cell(more, correct))
            step["astra"]["runs"] = [{k: c[k] for k in ("twin", "known")} for c in parts]
        out.append(step)
    # Astra rerun at effort max, the setting Opus used: the open question on both runs (LADDER2_PREREG.md, arm B) and
    # the forced choice on the first run (FC_PARITY_PREREG.md). The other rows have no max replies.
    parity = load(LAYOUT / "suite_bridge_fc" / "fc_parity_scores.json")
    maxed = {"open": [[dict(r, correct=r["stance_correct"]) for r in second if r["arm"] == a] for a in ("max_s1", "max_s2")],
             "forced": [[r for r in parity if r["arm"] == "astra_max"]]}
    for step in out:
        runs = maxed.get(step["id"])
        if runs:
            step["astra_max"] = cell([r for run in runs for r in run], "correct")
            if len(runs) > 1:
                step["astra_max"]["runs"] = [{k: cell(run, "correct")[k] for k in ("twin", "known")} for run in runs]
    return out


# Two coders agreed on "through the network" against "first and last only" for 23 of 24 re-read replies, but not on
# whether a claimed trace covered every layer or summarised the middle (17 of 24), so those two are merged here.
REACH = {"all layers": "through", "some middle layers skipped or summarised": "through", "first and last only": "ends",
         "unclear": "none"}


def replies(ra):
    """Coded methods of the depth-test replies (depth_reply_codes.json), joined to correctness."""
    codes = load(HERE / "depth_reply_codes.json")
    right = {}
    for d, v in ra["toolfree"]["by_depth"].items():
        for row in v["rows"]:
            if row["set"].endswith(f"readers_d{d}"):
                right[(int(d), row["id"])] = {"astra": bool(row.get("astra_xhigh")), "opus": bool(row.get("opus_xhigh"))}
    toolfree = [{"model": r["model"], "depth": r["depth"], "method": r["method"] if r["answered"] else "no answer",
                 "reach": REACH[r["layers"]] if r["answered"] else "none", "right": right[(r["depth"], r["id"])][r["model"]],
                 "hedges": r["hedges"], "confidence": r["confidence"] if r["confidence"] >= 0 else None,
                 "quote": r["quote"].replace("**", "").replace("`", "") if r["answered"] else ""} for r in codes["toolfree"]]
    python = {}
    for m in ("astra", "opus"):
        rows = [r for r in codes["python"] if r["model"] == m]
        ran = sorted(r["code_runs"] for r in rows if r["code_runs"])
        python[m] = {"n": len(rows), "ran_code": len(ran), "product": sum(r["method"] == "product of weight matrices" for r in rows),
                     "median_runs": ran[len(ran) // 2], "max_runs": ran[-1],
                     "unrun_analysis": sum(r["unrun_analysis"] for r in rows),
                     "applies_activation": sum(r["applies_activation_or_evaluates_layer"] for r in rows)}
    return {"toolfree": toolfree, "python": python}


def main():
    blog = load(HERE / "intermediate" / "layout_blocks.json")
    dp, ra = depth()
    out = {"pictures": pictures(), "depth": dp,
           "transformers": transformers(ra), "r1b": r1b(ra), "r1c": r1c(), "naming": naming(blog),
           "twins": twins(blog), "ladder": ladder(blog)}
    out["replies"] = replies(ra)
    out["play"] = play()
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "website_data.json"
    dest.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(dest, round(dest.stat().st_size / 1000, 1), "KB")
    print("depth D50", {m: (v["d50"], v["ci"]) for m, v in dp["toolfree"].items()}, "| paired", dp["paired_toolfree"])
    print("matched", dp["matched_totals"], "| depth 8 beam", dp["depth8_beam"])
    print("transformers", out["transformers"])
    print("r1b", out["r1b"]["totals"], "| r1c", {k: (v["exact"], v["condition"], v["pair"], v["opposite"], v["same"])
                                                  for k, v in out["r1c"]["totals"].items()})
    print("naming pooled", out["naming"]["pooled"], out["naming"]["median_tokens"])
    print("twins", out["twins"])
    print("ladder", [(s["id"], {m: (v["twin"], v["known"]) for m, v in s.items() if m != "id"}) for s in out["ladder"]])


if __name__ == "__main__":
    main()
