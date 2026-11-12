"""Matched-effort naming arm: score GPT-6 Astra at effort max against Astra at high and Opus at max.

Usage: python3 -I score_matched_effort.py --jobs   writes matched_effort_jobs.txt, lines "<suite> <id> <arm>": the pilot
                                                    network, then the other 47 camouflaged networks in a seeded random
                                                    order, each at max and then at high (the concurrent control), then
                                                    the 12 plain networks at max
       python3 -I score_matched_effort.py           scores what exists and writes matched_effort_scores.json
Scoring follows MATCHED_EFFORT_PREREG.md. The game is the one named first on the reply's Game line (game_first); the
unchanged keyword rule of score_suite.game_of is reported beside it. ReLU Breakout uses the rebuilt camouflaged networks
of suite_relu24x3_bfix, as arch_stats.py does. Tests are printed as final only when every camouflaged network has a
max-effort reply or 3 failed attempts; until then the output is labelled interim.
"""
import json
import random
import re
import sys
from math import comb
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # python -I leaves the script folder off the path
from models import valid_answer  # noqa: E402
from score_suite import GAME_RULES, game_of  # noqa: E402

ARCHS = [("tanh32x2", ["suite_v2"]), ("relu24x3", ["suite_relu24x3", "suite_relu24x3_bfix"]), ("tanh48x1", ["suite_tanh48x1"])]
GAMES = ["pong", "breakout", "pacman", "freeway"]
PILOT = ("suite_v2", "ae1d1ba0")
ORDER_SEED = 20261109
VERSION = "codex-cli 0.166.0-alpha.5.1"
# arm label -> (reply folder, run prefix, needs rc 0 in run.json, effort that run.json must record)
ARMS = {"astra_high": ("toolfree", "astra", False, None), "astra_high2": ("toolfree_high2", "astra", True, None),
        "astra_max": ("toolfree_max", "astra", True, "max"), "opus_max": ("toolfree", "opus", False, None)}
GAME_LINE = (r"(?mi)^[\W_]*Game[\W_]*:[\W_]*(.+)", r"(?mi)^[#\s]*Game\s*$\n+(.+)")  # the two patterns of game_of


def game_first(answer):
    """The game whose keyword comes first on the Game line; "other" if none, or if there is no Game line."""
    m = re.search(GAME_LINE[0], answer) or re.search(GAME_LINE[1], answer)
    if not m:
        return "other", "", []
    line = m.group(1).lower()
    hits = sorted((h.start(), g) for g, pat in GAME_RULES for h in [re.search(pat, line)] if h)
    return (hits[0][1] if hits else "other"), line[:160], [g for _, g in hits]


def networks():
    """(arch, suite, id, game, condition): the 48 camouflaged networks, then the 12 plain ones, in suite order."""
    rows = []
    for arch, suites in ARCHS:
        for s in suites:
            key = json.loads((HERE / s / "key.json").read_text())
            order = (HERE / s / "order.txt").read_text().split() if (HERE / s / "order.txt").exists() else sorted(key)
            for n in order:
                v = key[n]
                if v["condition"] not in ("camo", "plain"):
                    continue
                if s == "suite_relu24x3" and v["game"] == "breakout" and v["condition"] == "camo":
                    continue  # superseded by suite_relu24x3_bfix
                rows.append((arch, s, n, v["game"], v["condition"]))
    return [r for r in rows if r[4] == "camo"] + [r for r in rows if r[4] == "plain"]


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n) if n else 1.0


def usage(d):
    """Token usage of a codex reply: the turn.completed event (events.jsonl, or the usage.json copy the queue keeps)."""
    out = None
    for f in (d / "events.jsonl", d / "usage.json"):
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") == "turn.completed" and "usage" in e:
                out = e["usage"]
        if out:
            break
    return out


def reply(suite, label, n):
    folder, prefix, need_rc, need_effort = ARMS[label]
    d = HERE / suite / folder / f"{prefix}_{n}"
    t = valid_answer(d)
    if t is not None and need_rc:
        meta = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
        if meta.get("rc") != 0 or (need_effort and meta.get("effort") != need_effort):
            t = None
    return d, t


def main():
    nets = networks()
    if "--jobs" in sys.argv:
        camo = [r for r in nets if r[4] == "camo" and (r[1], r[2]) != PILOT]
        random.Random(ORDER_SEED).shuffle(camo)
        lines = [f"{PILOT[0]} {PILOT[1]} max", f"{PILOT[0]} {PILOT[1]} high2"]
        for _, s, n, _, _ in camo:
            lines += [f"{s} {n} max", f"{s} {n} high2"]
        lines += [f"{s} {n} max" for _, s, n, _, c in nets if c == "plain"]
        (HERE / "matched_effort_jobs.txt").write_text("\n".join(lines) + "\n")
        print(len(lines), "jobs:", sum(x.endswith(" max") for x in lines), "at max,", sum(x.endswith(" high2") for x in lines), "at high")
        return

    rows = []
    for arch, s, n, game, cond in nets:
        row = {"arch": arch, "suite": s, "id": n, "game": game, "condition": cond}
        for label in ARMS:
            d, t = reply(s, label, n)
            if t is None:
                row[label] = {"said": None, "correct": None}
            else:
                g, line, named = game_first(t)
                row[label] = {"said": g, "correct": g == game, "line": line, "named": named,
                              "unchanged_rule_correct": game_of(t)[0] == game}
            if label in ("astra_max", "astra_high2"):
                failed = len(list((HERE / s / (ARMS[label][0] + "_failed")).glob(f"astra_{n}.attempt*")))
                row[label]["status"] = "answered" if t is not None else "unanswered" if failed >= 3 else "not_run"
                u = usage(d)
                if t is not None and u:
                    row[label].update(input_tokens=u.get("input_tokens"), output_tokens=u.get("output_tokens"),
                                      reasoning_tokens=u.get("reasoning_output_tokens"))
                if t is not None and (d / "run.json").exists():
                    row[label]["wall_s"] = json.loads((d / "run.json").read_text()).get("wall_s")
                if (d / "codex_version.txt").exists():
                    row[label]["codex_version"] = (d / "codex_version.txt").read_text().strip()
        rows.append(row)

    ok = lambda r, k: bool(r[k]["correct"])  # noqa: E731  (a network with 3 failed attempts counts as wrong)

    def paired(sel, a, b, field="correct"):
        only_a = sum(bool(r[a].get(field)) and not bool(r[b].get(field)) for r in sel)
        only_b = sum(bool(r[b].get(field)) and not bool(r[a].get(field)) for r in sel)
        return {"only_" + a: only_a, "only_" + b: only_b, "mcnemar_p": mcnemar(only_a, only_b)}

    def block(cond):
        rs = [r for r in rows if r["condition"] == cond]
        out = {"n": len(rs)}
        out["max_status"] = {k: sum(r["astra_max"]["status"] == k for r in rs) for k in ("answered", "unanswered", "not_run")}
        out["complete"] = out["max_status"]["not_run"] == 0
        out["not_run"] = [f"{r['suite']}/{r['id']}" for r in rs if r["astra_max"]["status"] == "not_run"]
        out["unanswered"] = [f"{r['suite']}/{r['id']}" for r in rs if r["astra_max"]["status"] == "unanswered"]
        out["correct"] = {k: sum(ok(r, k) for r in rs) for k in ARMS}
        out["answered"] = {k: sum(r[k]["correct"] is not None for r in rs) for k in ARMS}
        out["correct_unchanged_rule"] = {k: sum(bool(r[k].get("unchanged_rule_correct")) for r in rs) for k in ARMS}
        out["tests"] = {"opus_max_vs_astra_max": paired(rs, "opus_max", "astra_max"),
                        "astra_max_vs_astra_high": paired(rs, "astra_max", "astra_high")}
        both_high = [r for r in rs if r["astra_high2"]["correct"] is not None]
        out["high_control"] = {"n": len(both_high), "astra_high2_correct": sum(ok(r, "astra_high2") for r in both_high),
                               "astra_high_correct_same_networks": sum(ok(r, "astra_high") for r in both_high),
                               "agree": sum(ok(r, "astra_high") == ok(r, "astra_high2") for r in both_high),
                               "astra_high2_vs_astra_high": paired(both_high, "astra_high2", "astra_high"),
                               "astra_max_vs_astra_high2": paired(both_high, "astra_max", "astra_high2")}
        if cond == "camo":
            out["share_of_gap_closed"] = (out["correct"]["astra_max"] - out["correct"]["astra_high"]) / max(
                1, out["correct"]["opus_max"] - out["correct"]["astra_high"])
            # Sensitivity (a): the 12 policies, each scored by its camouflaged copies named; sign test over policies.
            pol = {}
            for r in rs:
                p = pol.setdefault(f"{r['arch']}/{r['game']}", {"astra_max": 0, "opus_max": 0, "astra_high": 0, "n": 0})
                p["n"] += 1
                for k in ("astra_max", "opus_max", "astra_high"):
                    p[k] += ok(r, k)
            up = sum(p["opus_max"] > p["astra_max"] for p in pol.values())
            down = sum(p["astra_max"] > p["opus_max"] for p in pol.values())
            out["policy_level"] = {"policies": pol, "opus_ahead": up, "astra_ahead": down, "sign_test_p": mcnemar(up, down)}
            # Sensitivity (b): without the two tanh Breakout policies, which barely play Breakout.
            rest = [r for r in rs if not (r["game"] == "breakout" and r["arch"] in ("tanh32x2", "tanh48x1"))]
            out["without_tanh_breakout"] = {"n": len(rest), "correct": {k: sum(ok(r, k) for r in rest) for k in ARMS},
                                            "opus_max_vs_astra_max": paired(rest, "opus_max", "astra_max")}
        out["by_arch_game"] = {f"{arch}/{g}": {k: f"{sum(ok(r, k) for r in rs if r['arch'] == arch and r['game'] == g)}/"
                                                   f"{sum(r['arch'] == arch and r['game'] == g for r in rs)}" for k in ARMS}
                               for arch, _ in ARCHS for g in GAMES}
        out["astra_max_wrong_labels"] = sorted(f"{r['arch']}/{r['game']}->{r['astra_max']['said']}" for r in rs
                                               if r["astra_max"]["correct"] is False)
        out["multi_keyword_game_lines"] = [{"run": f"{k}:{r['suite']}/{r['id']}", "truth": r["game"], "named": r[k]["named"],
                                            "line": r[k]["line"]} for r in rs for k in ARMS if len(r[k].get("named") or []) > 1]
        for f in ("output_tokens", "reasoning_tokens", "wall_s"):
            v = [r["astra_max"][f] for r in rs if r["astra_max"].get(f)]
            out[f"astra_max_median_{f}"] = median(v) if v else None
        vers = sorted({r[k]["codex_version"] for r in rs for k in ("astra_max", "astra_high2") if r[k].get("codex_version")})
        out["codex_versions"] = vers
        out["codex_version_deviation"] = bool(vers) and vers != [VERSION]
        return out

    res = {"camouflaged": block("camo"), "plain": block("plain"), "rows": rows}
    (HERE / "matched_effort_scores.json").write_text(json.dumps(res, indent=1))
    for name in ("camouflaged", "plain"):
        b = res[name]
        tag = "FINAL" if b["complete"] else "INTERIM (no test or reading applies yet)"
        print(f"{name} [{tag}] n {b['n']} | max status {b['max_status']} | correct {b['correct']}")
        print("  unchanged keyword rule:", b["correct_unchanged_rule"])
        print("  tests:", b["tests"])
        print("  high control:", b["high_control"])
        if name == "camouflaged":
            print("  share of gap closed:", round(b["share_of_gap_closed"], 3), "| policy level:", b["policy_level"]["opus_ahead"],
                  "policies Opus ahead,", b["policy_level"]["astra_ahead"], "Astra ahead, p =", b["policy_level"]["sign_test_p"])
            print("  without tanh Breakout:", b["without_tanh_breakout"])
        print("  Astra max medians: output", b["astra_max_median_output_tokens"], "reasoning", b["astra_max_median_reasoning_tokens"],
              "wall s", b["astra_max_median_wall_s"], "| versions", b["codex_versions"], "| wrong labels", b["astra_max_wrong_labels"])
        if b["multi_keyword_game_lines"]:
            print("  Game lines naming more than one game:", [(m["run"], m["named"]) for m in b["multi_keyword_game_lines"]])


if __name__ == "__main__":
    main()
