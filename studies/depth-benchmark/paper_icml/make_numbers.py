"""Write numbers.tex: LaTeX macros for every result quoted in the paper, from the saved analysis outputs.

Usage: python3 -I make_numbers.py
Reads ../threshold/breaking_depth.json, ../threshold/twopager_analysis.json, review_analysis.json, the transformer
and realistic score and baseline files, and ../transformer/frozen_pair/verification.json.
"""
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = HERE.parent / "threshold"
TF = HERE.parent / "transformer"
RL = HERE.parent / "realistic"
res = json.loads((T / "breaking_depth.json").read_text())
an = json.loads((T / "twopager_analysis.json").read_text())
review = json.loads((HERE / "review_analysis.json").read_text())
frozen = json.loads((TF / "frozen_pair/verification.json").read_text())
WORDS = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 10: "Ten", 12: "Twelve",
         14: "Fourteen", 16: "Sixteen", 20: "Twenty"}
SHORT = {"B1_token_footprint": "One", "B2_order_free": "Two", "B3_position_footprint": "Three",
         "B4_single_layer_circuits": "Four"}
macros = {}


def per(key):
    return {int(d): v for d, v in res.get(key, {"per_depth": {}})["per_depth"].items()}


def pct(x):
    return f"{round(100 * x)}\\%"


def pval(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}".rstrip("0")


def frac(k, n):
    return f"{k}/{n}"


def series(values):
    values = [str(v) for v in values]
    return values[0] if len(values) == 1 else ", ".join(values[:-1]) + " and " + values[-1]


for tag, key in (("AstraTF", "astra_xhigh|tool-free"), ("OpusTF", "opus_xhigh|tool-free"),
                 ("AstraPy", "astra_xhigh|with tools"), ("OpusPy", "opus_xhigh|with tools")):
    p = per(key)
    for d, v in p.items():
        macros[f"{tag}Cnt{WORDS[d]}"] = f"{v['correct']}/{v['n']}"
        macros[f"{tag}Pc{WORDS[d]}"] = f"{v['mean_p_correct']:.2f}"
    macros[f"{tag}N"] = str(sum(v["n"] for v in p.values()))
    macros[f"{tag}Unans"] = str(sum(v["unanswered"] for v in p.values()))
    r = res.get(key, {})
    if "D50" in r:
        macros[f"{tag}Dfifty"] = f"{r['D50']:.1f}"
        macros[f"{tag}DfiftyLo"] = f"{r['D50_ci95'][0]:.1f}"
        macros[f"{tag}DfiftyHi"] = f"{r['D50_ci95'][1]:.1f}"
    macros[f"{tag}DeepestPass"] = str(r["deepest_pass"])
    macros[f"{tag}MaxN"] = str(max(v["n"] for v in p.values()))
    macros[f"{tag}MinN"] = str(min(v["n"] for v in p.values()))
for tag, key in (("Astra", "astra_xhigh"), ("Opus", "opus_xhigh")):
    item = res[f"{key}|tool-free|answered-only"]
    macros[f"{tag}AnsweredDfifty"] = f"{item['D50']:.1f}"
    macros[f"{tag}AnsweredLo"] = f"{item['D50_ci95'][0]:.1f}"
    macros[f"{tag}AnsweredHi"] = f"{item['D50_ci95'][1]:.1f}"

tb = {int(k): v for k, v in an["tools_by_depth"].items()}
for d, v in tb.items():
    macros[f"FullC{WORDS[d]}"] = pct(v["full_contraction"])
    macros[f"ThreeP{WORDS[d]}"] = pct(v["three_paths"])
    macros[f"LinM{WORDS[d]}"] = pct(v["linear_map"])
    macros[f"NetsN{WORDS[d]}"] = str(v["n"])
macros["FullCMinToTwelve"] = pct(min(v["full_contraction"] for d, v in tb.items() if d <= 12))
macros["NetsTotal"] = f"{sum(v['n'] for v in tb.values()):,}".replace(",", "{,}")
fp = an["footprint"]
macros.update({"FpIdentical": str(fp["identical_rows"]), "FpTouched": str(fp["touched_toward"]),
               "FpCorr": f"{fp['displacement_corr']:.3f}", "FpExpCorr": f"{fp['exp1_rownorm_dev_corr']:.2f}",
               "FpUnit": str(fp["unit_toward"]["unit"]), "FpQToward": f"{fp['unit_toward']['q']:+.2f}",
               "FpQAway": f"{fp['unit_away']['q']:+.2f}", "FpDFourReadout": f"{fp['d4_readout']:.0f}"})

# Tool-free inventory, unanswered replies and matched comparisons on the reader-set networks.
inv = review["toolfree"]["inventory"]
for tag, key in (("Astra", "astra_xhigh"), ("Opus", "opus_xhigh")):
    v = inv[key]
    macros[f"{tag}TFPlanned"] = str(v["planned"])
    macros[f"{tag}TFSaved"] = str(v["saved"])
    macros[f"{tag}TFMissing"] = str(v["planned"] - v["saved"])
    kinds = v["unanswered_by_kind"]
    macros[f"{tag}TFOutputLimit"] = str(kinds.get("output_limit", 0))
    macros[f"{tag}TFWithheld"] = str(kinds.get("withheld", 0))
    depths = v["unanswered_by_depth"]
    if depths:
        macros[f"{tag}TFUnansDepths"] = series(sorted(depths, key=int))
        macros[f"{tag}TFUnansCounts"] = series(depths[d] for d in sorted(depths, key=int))
    missing = v["missing_by_set"]
    if missing:
        parts = [f"{n} at depth {re.search(r'readers_d(\d+)', s).group(1)}" for s, n in missing.items()]
        macros[f"{tag}TFMissingWhere"] = series(sorted(parts, key=lambda x: int(x.rsplit(" ", 1)[1])))
for tag, key in (("Astra", "astra_xhigh"), ("Opus", "opus_xhigh")):
    # Deepest passing depth below the first failing depth, and the fitted accuracy at the deepest tool-free depth.
    p = per(f"{key}|tool-free")
    first_fail = min(d for d, v in p.items() if v["verdict"] == "fail")
    macros[f"{tag}TFPassBeforeFail"] = str(max(d for d, v in p.items() if v["verdict"] == "pass" and d < first_fail))
    r = res[f"{key}|tool-free"]
    macros[f"{tag}TFFitDeepest"] = pct(0.5 + 0.5 / (1 + math.exp(r["slope"] * (max(p) - r["D50"]))))
    macros["TFDeepest"] = str(max(p))
    deep = review["toolfree_deep_confidence"][key]
    for band, name in (("committed", "Commit"), ("between", "Mid"), ("even", "Even")):
        macros[f"{tag}Deep{name}N"] = str(deep[band]["n"])
        macros[f"{tag}Deep{name}Right"] = str(deep[band]["correct"])
    macros[f"{tag}DeepEvenAway"] = str(deep["even"]["right_away"])
    macros[f"{tag}DeepN"] = str(sum(deep[b]["n"] for b in ("committed", "between", "even")) + deep["unanswered"])
pt = review["toolfree"]["astra_vs_opus"]
macros.update({"TFPairedN": str(pt["n"]), "TFPairedAstraOnly": str(pt["a_only"]),
               "TFPairedOpusOnly": str(pt["b_only"])})
for d, v in review["toolfree"]["by_depth"].items():
    w = WORDS[int(d)]
    macros[f"TFBeam{w}"] = frac(v["three_paths"]["correct"], v["n"])
    macros[f"TFCurv{w}"] = frac(v["curvature"]["correct"], v["n"])
    for tag, key in (("Astra", "astra_xhigh"), ("Opus", "opus_xhigh")):
        q = v[f"{key}_vs_three_paths"]
        macros[f"{tag}TFBeamPairN{w}"] = str(q["n"])
        macros[f"{tag}TFOverBeam{w}"] = str(q["a_only"])
        macros[f"{tag}TFUnderBeam{w}"] = str(q["b_only"])
        macros[f"{tag}TFBeamPairP{w}"] = pval(q["p_mcnemar_two_sided"])

# With-Python arm: readers and fixed scripts on the same blinded networks.
matched = review["matched"]
macros["MatchedTotal"] = str(sum(v["n"] for v in matched.values()))
macros["MatchedScriptTotal"] = str(sum(v["curvature"]["correct"] for v in matched.values()))
macros["MatchedBeamTotal"] = str(sum(v["three_paths"]["correct"] for v in matched.values()))
macros["MatchedSlopeTotal"] = str(sum(v["ball_slope"]["correct"] for v in matched.values()))
macros["MatchedAstraTotal"] = str(sum(v["astra"]["correct"] for v in matched.values()))
macros["MatchedOpusTotal"] = str(sum(v["opus"]["correct"] for v in matched.values()))
mt = review["matched_totals"]
macros.update({"MatchedScriptOnly": str(mt["script_astra"]["a_only"]), "MatchedAstraOnly": str(mt["script_astra"]["b_only"]),
               "MatchedScriptAstraP": pval(mt["script_astra"]["p_mcnemar_two_sided"]),
               "MatchedScriptOnlyOpus": str(mt["script_opus"]["a_only"]), "MatchedOpusOnly": str(mt["script_opus"]["b_only"]),
               "MatchedScriptOpusP": pval(mt["script_opus"]["p_mcnemar_two_sided"]),
               "MatchedAstraNotOpus": str(mt["astra_opus"]["a_only"]), "MatchedOpusNotAstra": str(mt["astra_opus"]["b_only"])})
macros["MatchedTableRows"] = "\n".join(
    f"{d} & {v['n']} & {v['astra']['correct']} & {v['opus']['correct']} & {v['curvature']['correct']} & "
    f"{v['three_paths']['correct']} & {v['ball_slope']['correct']} " + r"\\"
    for d, v in matched.items())

macros["DepthTwentyFourQualified"] = str(review["depth24"]["qualified"])
macros["DepthTwentyFourTrained"] = str(review["depth24"]["trained"])
baseline_six = json.loads((HERE.parent / "ladder/scale/baselines_d6.json").read_text())["B2"]
macros["DepthSixOrderFree"] = f"{baseline_six['correct']}/{baseline_six['n']}"
macros["AmendmentCount"] = str(len(re.findall(r"^## Amendment", (T / "PREREG.md").read_text(), re.M)))

# Transformers T1, T2 and G2.
for tag, d in (("TT", "readers_t2"), ("TG", "readers_g2")):
    sc = json.loads((TF / d / "score.json").read_text())
    for who, lab in (("Astra", "astra_xhigh"), ("Opus", "opus_xhigh")):
        v = sc[lab]
        macros[f"{tag}{who}"] = f"{v['correct']}/{v['n']}"
        macros[f"{tag}{who}P"] = f"{v['p_one_sided']:.4f}".rstrip("0").rstrip(".")
        macros[f"{tag}{who}Pc"] = f"{v['partial_credit']:.2f}"
for tag, f in (("BTone", "baselines_t1.json"), ("BTtwo", "baselines_t2.json"), ("BGtwo", "baselines_g2.json")):
    d = json.loads((TF / f).read_text())
    for k, short in (*SHORT.items(), ("reference_kcomp_sign_rule", "Ref")):
        macros[f"{tag}{short}"] = f"{d[k]['correct']}/{d[k]['n']}"
ac = json.loads((TF / "attention_check_t2.json").read_text())["summary"]
macros.update({"AttnPrevMean": f"{ac['prev_token_l1_mean']:.2f}", "AttnNextJone": f"{ac['next_routed_to_j1']:.2f}",
               "AttnPrevJ": f"{ac['prev_routed_to_j']:.2f}", "AttnFollow": f"{ac['n_follow_routing']}/{ac['n']}"})
for tag, key in (("TTwo", "t2"), ("GTwo", "g2")):
    item = review[key]
    macros[f"{tag}ShortcutPool"] = f"{item['pool_correct']}/{item['pool_n']}"
    macros[f"{tag}ShortcutReader"] = f"{item['reader_correct']}/{item['reader_n']}"
    q = item["astra_vs_opus"]
    macros[f"{tag}AstraNotOpus"] = str(q["a_only"])
    macros[f"{tag}OpusNotAstra"] = str(q["b_only"])
    macros[f"{tag}PairedP"] = pval(q["p_mcnemar_two_sided"])
macros["SymmetryTrainPartners"] = str(review["symmetry"]["0"]["partner_present"])
macros["SymmetryTestPartners"] = str(review["symmetry"]["1"]["partner_present"])
macros["ArithmeticTableRows"] = "\n".join(
    f"$D_{{{v['a']}{v['b']}}}$ & ${v['astra']}$ & ${v['recomputed']:.2f}$ " + r"\\"
    for v in review["arithmetic"]["rows"])

# Realistic transformer R1b: reader components, pool baselines and baselines on the same reader networks.
r = json.loads((RL / "readers_r1b" / "score.json").read_text())
for who, lab in (("Astra", "astra_xhigh"), ("Opus", "opus_xhigh")):
    v = r[lab]
    macros[f"R{who}Exact"] = f"{v['exact']}/{v['n']}"
    macros[f"R{who}Cond"] = f"{v['condition']}/{v['n']}"
    macros[f"R{who}Pair"] = f"{v['offset_pair']}/{v['n']}"
    macros[f"R{who}Orient"] = v["orientation"]
    macros[f"R{who}Answered"] = str(v["answered"])
b1 = json.loads((RL / "baselines_r1.json").read_text())
b1b = json.loads((RL / "baselines_r1b.json").read_text())
for tag, d in (("RBtrain", b1), ("RBmirror", b1b)):
    for t, tt in (("orientation", "Orient"), ("condition", "Cond"), ("offset_pair", "Pair")):
        for k, short in SHORT.items():
            macros[f"{tag}{tt}{short}"] = f"{d[t][k]['correct']}/{d[t][k]['n']}"
rr = review["r1b"]
ro = rr["readers"]["astra_xhigh"]["orientation"]
macros.update({"RAstraOrientLo": f"{ro['ci95'][0]:.2f}", "RAstraOrientHi": f"{ro['ci95'][1]:.2f}",
               "RAstraOrientP": pval(ro["p_greater_than_half"]),
               "ROpusCondP": pval(rr["readers"]["opus_xhigh"]["p_condition_vs_quarter"])})
for t, tt in (("orientation", "Orient"), ("condition", "Cond"), ("offset_pair", "Pair")):
    for k, short in SHORT.items():
        e = rr["baselines"][t][k]
        macros[f"RMatched{tt}{short}"] = frac(e["matched_correct"], e["matched_n"])
        if t == "orientation":
            macros[f"RMatchedEligible{short}"] = frac(e["astra_eligible_correct"], e["astra_eligible_n"])
        else:
            q = e["astra_paired"]
            macros[f"RPaired{tt}{short}AstraOnly"] = str(q["a_only"])
            macros[f"RPaired{tt}{short}BaseOnly"] = str(q["b_only"])
            macros[f"RPaired{tt}{short}P"] = pval(q["p_mcnemar_two_sided"])

# Frozen-tensor pair and replayed trajectories.
frozen_runs = {v["rule"]: v for k, v in frozen["runs"].items() if k.startswith("nets_3000/")}
macros["FrozenNextAcc"] = f"{100 * frozen_runs['evennext']['accuracy']:.1f}\\%"
macros["FrozenPrevAcc"] = f"{100 * frozen_runs['evenprev']['accuracy']:.1f}\\%"
macros["FrozenHeldoutN"] = f"{frozen['heldout_n']:,}".replace(",", "{,}")
macros["FrozenShortcutScore"] = f"{frozen_runs['evennext']['shortcut_score']:+.4f}"
macros["FrozenSnapshotN"] = str(frozen_runs['evennext']['checkpoints'])
macros["FrozenDisagreements"] = f"{frozen['frozen_pair_disagreements']:,}".replace(",", "{,}")
macros["SavedSnapshotN"] = str(frozen["snapshot_count"])
replay_runs = [v for k, v in frozen["runs"].items() if k.startswith("replay/")]
macros["ReplayLowestAcc"] = f"{100 * min(v['accuracy'] for v in replay_runs):.1f}\\%"

RLc = HERE.parent / "realistic"
rc = json.loads((RLc / "readers_r1c" / "score_r1c.json").read_text())
for tag, arm in (("RcTF", "astra_xhigh"), ("RcPy", "python")):
    v = rc[arm]
    macros[f"{tag}Exact"] = f"{v['exact']}/{v['n']}"
    macros[f"{tag}Cond"] = f"{v['condition']}/{v['n']}"
    macros[f"{tag}Pair"] = f"{v['pair']}/{v['n']}"
    macros[f"{tag}OrientTrained"] = v["orientation_trained"]
    macros[f"{tag}OrientMirror"] = v["orientation_mirror"]
    ot = [int(x) for x in v["orientation_trained"].split("/")]; om = [int(x) for x in v["orientation_mirror"].split("/")]
    macros[f"{tag}Orient"] = f"{ot[0] + om[0]}/{ot[1] + om[1]}"
    macros[f"{tag}PairsConsistent"] = str(v["structure_consistent_pairs"])
    macros[f"{tag}Opposite"] = str(v["opposite"])
    macros[f"{tag}Same"] = str(v["same"])
    macros[f"{tag}CueAgree"] = v["orientation_agrees_with_B2_cue"]
    macros[f"{tag}TruthAgree"] = v["orientation_agrees_with_truth"]
import statistics as _st
_key = json.loads((RLc / "readers_r1c" / "key.json").read_text())
_runs = [json.loads((HERE.parent / "analyst" / "runs" / f"r1c_{b}" / "state.json").read_text())["runs"] for b in _key]
macros["RcPyRunsMedian"] = str(int(_st.median(_runs)))
macros["RcPyRunsMin"], macros["RcPyRunsMax"] = str(min(_runs)), str(max(_runs))
import re as _re
_ANS = _re.compile(r"answer\s*:?\s*\**\s*a\s*=\s*([+-]?\d)\s*,?\s*b\s*=\s*([+-]?\d)\s*,?\s*C\s*=\s*(C[1-4])", _re.I)
_cmiss = _c4c1 = 0
for _b, _name in _key.items():
    _t = (HERE.parent / "analyst" / "runs" / f"r1c_{_b}" / "final_answer.md").read_text().replace("−", "-")
    _m = _ANS.findall(_t)[-1]
    if _m[2] != _name.split("_")[3]:
        _cmiss += 1; _c4c1 += (_name.split("_")[3], _m[2]) == ("C4", "C1")
macros["RcPyCondMiss"], macros["RcPyCondMissCFourCOne"] = str(_cmiss), str(_c4c1)
_au = json.loads((RLc / "readers_r1c" / "forward_pass_audit.json").read_text())
macros["RcAuditCodeRuns"] = str(_au["code_runs"])
macros["RcAuditForward"] = str(_au["forward_passes_found"])
_tc = json.loads((RLc / "readers_r1b" / "transcript_coding.json").read_text())
macros["TxAstraGuessN"], macros["TxAstraGuessExact"] = str(_tc["astra_guess_n"]), str(_tc["astra_guess_exact"])
macros["TxAstraDerivedN"] = str(_tc["astra_derived_n"])
macros["TxOpusAnswered"], macros["TxOpusDefault"] = str(_tc["opus_answered"]), str(_tc["opus_default_plus1_minus1"])
lines = ["% Generated by make_numbers.py; do not edit."] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
(HERE / "numbers.tex").write_text("\n".join(lines) + "\n")
print(len(macros), "macros")
