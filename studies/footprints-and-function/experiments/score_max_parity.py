"""Score the heavier-rewrites networks at matched effort (MAX_PARITY_PREREG.md).

Usage: python3 -I score_max_parity.py [--json out.json]
Rule: the first game named on the reply's Game line has to be the right one (game_first of
../obscured/score_matched_effort.py). A reply with no answer file, or an Opus reply that is the output-cap error,
names no game. Arms: Astra high (toolfree/), Astra high rerun (toolfree_high2/), Astra max (toolfree_max/), Opus max
(toolfree/); with Python on the 9-layer networks: analyst/ (Astra high, Opus max) and analyst_max/ (Astra max).
Prints counts per row, the paired tests and token medians, and optionally writes the per-network table.
"""
import json
import statistics
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ff_paths import LAYOUT  # noqa: E402
sys.path.insert(0, str(LAYOUT))
from score_matched_effort import game_first  # noqa: E402

ROWS = [("plain", "suite_stage5", "plain"), ("camo", "suite_stage5", "camo_v2"),  # the two baseline rows (addendum 2)
        ("synth", "suite_stage5", "synth"), ("pairs", "suite_stage5", "pairs"), ("basis", "suite_stage5", "basis"),
        ("split", "suite_stage6", "split"), ("relu0", "suite_stage6", "deep0"), ("relu3", "suite_stage6", "deep3"),
        ("relu6", "suite_stage6", "deep6")]
GROUP = {"plain": "base", "camo": "base", "synth": "tanh", "pairs": "tanh", "basis": "tanh", "split": "tanh", "relu0": "relu", "relu3": "relu",
         "relu6": "relu"}
ARMS = [("astra", "toolfree", "astra"), ("astra_high2", "toolfree_high2", "astra"), ("astra_max", "toolfree_max", "astra"),
        ("opus", "toolfree", "opus")]
GAMES = ("pong", "breakout", "pacman", "freeway")


def outcome(folder, truth):
    """1 named, 0 another game, -1 no answer, None not run."""
    f = folder / "answer.md"
    if not folder.exists():
        return None, ""
    text = f.read_text(errors="ignore").strip() if f.exists() else ""
    if not text or (text.lower().startswith("api error") and "output token maximum" in text):
        return -1, ""
    first, line, _ = game_first(text)
    return int(first == truth), line


def usage(folder):
    ev = folder / "events.jsonl"
    if not ev.exists():
        return None
    for line in reversed(ev.read_text(errors="ignore").splitlines()):
        if '"turn.completed"' in line:
            return json.loads(line).get("usage")
    return None


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    table = []
    for row, suite, cond in ROWS:
        key = json.loads((HERE / suite / "key.json").read_text())
        for nid, v in key.items():
            if v["condition"] != cond:
                continue
            rec = {"row": row, "group": GROUP[row], "suite": suite, "id": nid, "game": v["game"]}
            for arm, sub, who in ARMS:
                rec[arm], rec[arm + "_line"] = outcome(HERE / suite / sub / f"{who}_{nid}", v["game"])
            if row == "relu6":
                for arm, sub, who in (("py_astra", "analyst", "astra"), ("py_astra_max", "analyst_max", "astra"),
                                      ("py_opus", "analyst", "opus")):
                    d = HERE / suite / sub / f"{who}_{nid}"
                    f = d / "final_answer.md"
                    if f.exists():
                        first, line, _ = game_first(f.read_text(errors="ignore"))
                        state = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else {}
                        rec[arm], rec[arm + "_line"], rec[arm + "_runs"] = int(first == v["game"]), line, state.get("runs")
                    else:
                        rec[arm] = None
            table.append(rec)
    everything, table = table, [r for r in table if r["group"] != "base"]   # tests and totals: the 28 rewritten networks
    arms = [a for a, _, _ in ARMS]
    for row in ("plain", "camo"):
        rs = [r for r in everything if r["row"] == row]
        print(f"{row:7}" + "".join(f"{sum(r[a] == 1 for r in rs):>9}/{sum(r[a] is not None for r in rs):<3}" for a in arms) + "   (baseline)")
        for r in rs:
            if r["astra_max"] is not None:
                print(f"        {r['id']} {r['game']:9} max {r['astra_max']} <- {r['astra_max_line'][:90]!r}")
    print(f"{'row':7}" + "".join(f"{a:>13}" for a in arms))
    for row, _, _ in ROWS[2:]:
        rs = [r for r in table if r["row"] == row]
        print(f"{row:7}" + "".join(f"{sum(r[a] == 1 for r in rs):>9}/{sum(r[a] is not None for r in rs):<3}" for a in arms))
    for label, rs in (("tanh", [r for r in table if r["group"] == "tanh"]), ("relu", [r for r in table if r["group"] == "relu"]),
                      ("all", table)):
        print(f"{label:7}" + "".join(f"{sum(r[a] == 1 for r in rs):>9}/{sum(r[a] is not None for r in rs):<3}" for a in arms))
    done = [r for r in table if r["astra_max"] is not None and r["astra_high2"] is not None]
    print(f"\nnetworks with both a max and a high-rerun reply: {len(done)} of {len(table)}")
    for a, b in (("astra_max", "astra_high2"), ("astra_max", "astra"), ("astra_high2", "astra"), ("opus", "astra_max")):
        rs = [r for r in table if r[a] is not None and r[b] is not None]
        only_a = sum(r[a] == 1 and r[b] != 1 for r in rs)
        only_b = sum(r[b] == 1 and r[a] != 1 for r in rs)
        agree = sum((r[a] == 1) == (r[b] == 1) for r in rs)
        print(f"{a} vs {b}: n={len(rs)}, {a}-only {only_a}, {b}-only {only_b}, agree {agree}, McNemar p = {mcnemar(only_a, only_b):.4f}")
    print("\nper game, Astra max / high rerun / high / Opus:")
    for g in GAMES:
        rs = [r for r in table if r["game"] == g]
        print(f"  {g:9}" + "  ".join(f"{sum(r[a] == 1 for r in rs)}/{sum(r[a] is not None for r in rs)}"
                                      for a in ("astra_max", "astra_high2", "astra", "opus")))
    print("\nmisses at max (first game on the Game line):")
    for r in table:
        if r["astra_max"] == 0 or r["astra_max"] == -1:
            print(f"  {r['row']:6} {r['id']} {r['game']:9} -> {r['astra_max_line'][:90]!r}")
    # The figure reads Opus's lines strictly: a line that leads with another game ("Dodge ... similar to Pac-Man") is
    # not credited even though the keyword rule finds the right game first. Every credited line is listed so the same
    # reading can be applied to Astra.
    print("\ncredited at max, Game lines (check that each leads with the game):")
    for r in table:
        if r["astra_max"] == 1:
            print(f"  {r['row']:6} {r['id']} {r['game']:9} <- {r['astra_max_line'][:110]!r}")
    for arm, sub in (("astra_max", "toolfree_max"), ("astra_high2", "toolfree_high2"), ("astra", "toolfree")):
        us = [usage(HERE / r["suite"] / sub / f"astra_{r['id']}") for r in table]
        us = [u for u in us if u]
        if us:
            print(f"{arm}: median output tokens {statistics.median(u['output_tokens'] for u in us):,.0f}, "
                  f"reasoning {statistics.median(u.get('reasoning_output_tokens', 0) for u in us):,.0f} (n={len(us)})")
    py = [r for r in table if r["row"] == "relu6"]
    print("\nwith Python, 9-layer networks:")
    for arm in ("py_astra", "py_astra_max", "py_opus"):
        print(f"  {arm:13} {sum(r.get(arm) == 1 for r in py)}/{sum(r.get(arm) is not None for r in py)}  "
              + "; ".join(f"{r['game']}: {r.get(arm + '_line', '')[:50]!r} ({r.get(arm + '_runs')} runs)" for r in py if r.get(arm) is not None))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(everything, indent=1))


if __name__ == "__main__":
    main()
