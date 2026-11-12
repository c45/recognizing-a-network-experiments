"""Score analyst-arm conversations (ladder PREREG.md, analyst arm).

Usage: python3 -I score_analyst.py <family> <nets dir>
Checks key_<family>.json against PREREG_KEY_HASH_<family>.txt, parses each runs/<family>_<id>/final_answer.md, and
flags code runs that may evaluate the network (tanh, softmax, an input vector pushed through W1, or the word
'forward') for reading by hand.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
FLAGS = [r"np\.tanh|torch\.tanh|\.tanh\(", r"softmax", r"forward", r"@\s*W\[['\"]W1['\"]\]", r"x\s*@\s*W"]


def main(fam, nets):
    raw = (HERE / f"key_{fam}.json").read_text()
    assert hashlib.sha256(raw.encode()).hexdigest() == (HERE / f"PREREG_KEY_HASH_{fam}.txt").read_text().split()[0]
    key = json.loads(raw)
    rows = []
    for bid, name in sorted(key.items()):
        rd = HERE / "runs" / f"{fam}_{bid}"
        fa = rd / "final_answer.md"
        text = fa.read_text() if fa.exists() else ""
        a = re.findall(r"answer\s*:?\s*\**\s*(right-toward|right-away)", text, re.I)
        c = re.findall(r"confidence\s*:?\s*\**\s*(\d{1,3})\s*%", text, re.I)
        tools = [json.loads(l) for l in (rd / "tool_calls.jsonl").read_text().splitlines()] if (rd / "tool_calls.jsonl").exists() else []
        flagged = [t["run"] for t in tools if any(re.search(f, t["code"]) for f in FLAGS)]
        audits = [json.loads(p.read_text()) for p in sorted((rd / "turns").glob("t*/run.json"))]
        truth = name.rsplit("_s", 1)[0].replace("_", "-")
        rows.append({"id": bid, "network": name, "answer": a[-1].lower() if a else None,
                     "confidence": int(c[-1]) if c else None, "correct": bool(a) and a[-1].lower() == truth,
                     "runs": len(tools), "flagged_runs": flagged, "tool_free": all(x.get("tool_free") for x in audits)})
    scored = [r for r in rows if r["answer"]]
    k, n = sum(r["correct"] for r in scored), len(scored)
    res = {"family": fam, "correct": k, "n": n,
           "p_one_sided": round(binomtest(k, n, 0.5, alternative="greater").pvalue, 4) if n else None,
           "rows": rows}
    (HERE / f"score_{fam}.json").write_text(json.dumps(res, indent=1))
    print(f"{fam}: {k}/{n}, p={res['p_one_sided']}")
    for r in rows:
        print("  ", r["id"], r["network"], r["answer"], r["confidence"], r["correct"], "runs", r["runs"], "flagged", r["flagged_runs"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
