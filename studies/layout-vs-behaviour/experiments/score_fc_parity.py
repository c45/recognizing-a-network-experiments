"""Score the matched-effort forced-choice arm (FC_PARITY_PREREG.md).

Usage: python3 -I score_fc_parity.py            writes suite_bridge_fc/fc_parity_scores.json and prints the counts
       python3 -I score_fc_parity.py --jobs     writes fc_parity_jobs.txt (random order, each network at max then high2)
Arms on the 18 networks of suite_bridge_fc: Astra at max (toolfree_max), the concurrent Astra high control
(toolfree_high2), and the existing Astra high and Opus replies (toolfree). The Answer line is parsed with the pattern
of score_bridge.py; an unparseable answer counts as wrong, an unanswered network is left out and listed.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I leaves the script folder off the path
from models import effort, valid_answer  # noqa: E402
from score_bridge import ANS, mcnemar  # noqa: E402

S = Path("suite_bridge_fc")
ARMS = [("astra_max", "toolfree_max", "astra"), ("astra_high2", "toolfree_high2", "astra"),
        ("astra_high", "toolfree", "astra"), ("opus", "toolfree", "opus")]


def jobs():
    order = S.joinpath("order.txt").read_text().split()
    random.Random(20261110).shuffle(order)
    Path("fc_parity_jobs.txt").write_text("".join(f"{S} {n} {arm}\n" for n in order for arm in ("max", "high2")))
    print("fc_parity_jobs.txt", 2 * len(order), "jobs")


def main():
    key = json.loads(S.joinpath("key.json").read_text())
    rows = []
    for arm, folder, model in ARMS:
        for nid, k in key.items():
            d = S / folder / f"{model}_{nid}"
            text = valid_answer(d)
            m = ANS.search(text) if text else None
            said = m.group(1) if m else None
            rows.append({"arm": arm, "id": nid, "game": k["game"], "twin": k["twin"], "condition": k["condition"],
                         "answered": text is not None, "effort": effort(d) if text else None, "said": said,
                         "truth": k["truth_letter"], "correct": said == k["truth_letter"] if text else None})
    S.joinpath("fc_parity_scores.json").write_text(json.dumps(rows, indent=1))
    by = {(r["arm"], r["id"]): r for r in rows if r["answered"]}
    for arm, _, _ in ARMS:
        for twin in (True, False):
            sel = [r for r in rows if r["arm"] == arm and r["twin"] == twin]
            ans = [r for r in sel if r["answered"]]
            eff = {e: sum(r["effort"] == e for r in ans) for e in sorted({r["effort"] for r in ans})}
            print(f"{arm:11s} {'twin ' if twin else 'known'}: {sum(r['correct'] for r in ans)}/{len(ans)}"
                  f"  unanswered {[r['id'] for r in sel if not r['answered']]}"
                  f"  unparsed {sum(r['said'] is None for r in ans)}  efforts {eff}")
    for a, b in (("astra_max", "opus"), ("astra_max", "astra_high2"), ("astra_max", "astra_high"), ("astra_high2", "astra_high")):
        for twin in (True, False):
            ids = [n for n, k in key.items() if k["twin"] == twin and (a, n) in by and (b, n) in by]
            x = sum(by[(a, n)]["correct"] and not by[(b, n)]["correct"] for n in ids)
            y = sum(by[(b, n)]["correct"] and not by[(a, n)]["correct"] for n in ids)
            print(f"{a} vs {b} {'twin ' if twin else 'known'}: paired n={len(ids)}  {a}-only {x}, {b}-only {y}, "
                  f"McNemar p={mcnemar(x, y):.4f}  same letter {sum(by[(a, n)]['said'] == by[(b, n)]['said'] for n in ids)}")


if __name__ == "__main__":
    jobs() if "--jobs" in sys.argv else main()
