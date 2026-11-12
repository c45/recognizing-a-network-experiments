"""Score tool-free replies on an R1 reader set: exact rule, condition, offset pair, orientation.

Usage: python3 -I score_r1.py <reader set dir> [label ...]
Checks the key hash, skips infrastructure failures, and reports exact matches (chance 1/120), condition (1/4),
unordered offset pair (1/15) and orientation given correct pair and condition (1/2), with one-sided binomial p-values.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

from scipy.stats import binomtest

ANS = re.compile(r"answer\s*:?\s*\**\s*a\s*=\s*([+-]?\d)\s*,?\s*b\s*=\s*([+-]?\d)\s*,?\s*C\s*=\s*(C[1-4])", re.I)
CONF = re.compile(r"confidence\s*:?\s*\**\s*(\d{1,3})\s*%", re.I)
INFRA = re.compile(r"hit your|reached your [A-Za-z0-9. ]*limit|Can't reach|Connection refused|at capacity|Not logged in|usage limit")


def p_at_least(c, n, p0):
    return round(binomtest(c, n, p0, alternative="greater").pvalue, 5) if n else None


def main(d, labels):
    d = Path(d)
    raw = (d / "key.json").read_text()
    assert hashlib.sha256(raw.encode()).hexdigest() == (d / "PREREG_KEY_HASH.txt").read_text().split()[0]
    key = json.loads(raw)
    res = {}
    for lab in labels or sorted(p.name for p in (d / "replies").iterdir()):
        n = answered = exact = cond = pair = orient_n = orient_ok = 0
        confs = []
        for bid, name in key.items():
            f = d / "replies" / lab / bid / "answer.md"
            if not f.exists() or INFRA.search(f.read_text()):
                continue
            a, b, C = name.split("_")[1:4]
            a, b = int(a), int(b)
            n += 1
            text = f.read_text().replace("\u2212", "-").replace("\u2013", "-")
            m = ANS.findall(text)
            if not m:
                continue
            answered += 1
            pa, pb, pC = int(m[-1][0]), int(m[-1][1]), m[-1][2].upper()
            c = CONF.findall(text)
            ok = (pa, pb, pC) == (a, b, C)
            exact += ok
            cond += pC == C
            pair += {pa, pb} == {a, b}
            if pC == C and {pa, pb} == {a, b}:
                orient_n += 1
                orient_ok += ok
            confs.append((int(c[-1]) if c else None, ok))
        res[lab] = {"n": n, "of": len(key), "answered": answered,
                    "exact": exact, "p_exact": p_at_least(exact, n, 1 / 120),
                    "condition": cond, "p_condition": p_at_least(cond, n, 0.25),
                    "offset_pair": pair, "p_pair": p_at_least(pair, n, 1 / 15),
                    "orientation": f"{orient_ok}/{orient_n}", "confidences": confs}
        print(lab, {k: v for k, v in res[lab].items() if k != "confidences"})
    (d / "score.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
