"""Score tool-free replies on a transformer reader set (even-next vs even-prev).

Usage: python3 -I score_tf.py <reader set dir> [label ...]
Checks the key against PREREG_KEY_HASH.txt, skips infrastructure failures, and reports correct/answered/n, the
exact binomial p (one-sided, against 0.5) and partial credit (mean probability on the correct answer; unanswered
counts 0.5).
"""
import hashlib
import json
import re
import sys
from pathlib import Path

from scipy.stats import binomtest

ANS = re.compile(r"answer\s*:?\s*\**\s*(even-next|even-prev)", re.I)
CONF = re.compile(r"confidence\s*:?\s*\**\s*(\d{1,3})\s*%", re.I)
INFRA = re.compile(r"hit your|reached your [A-Za-z0-9. ]*limit|Can't reach|Connection refused|at capacity|Not logged in|usage limit")


def main(d, labels):
    d = Path(d)
    raw = (d / "key.json").read_text()
    assert hashlib.sha256(raw.encode()).hexdigest() == (d / "PREREG_KEY_HASH.txt").read_text().split()[0]
    key = json.loads(raw)
    res = {}
    for lab in labels or sorted(p.name for p in (d / "replies").iterdir()):
        c = n = answered = 0
        pc = 0.0
        for bid, name in key.items():
            f = d / "replies" / lab / bid / "answer.md"
            if not f.exists() or INFRA.search(f.read_text()):
                continue
            text = f.read_text()
            n += 1
            a = ANS.findall(text)
            if not a:
                pc += 0.5
                continue
            answered += 1
            conf = CONF.findall(text)
            p = min(max(int(conf[-1]), 50), 100) / 100 if conf else 0.5
            ok = a[-1].lower() == name.rsplit("_s", 1)[0].replace("_", "-")
            c += ok
            pc += p if ok else 1 - p
        res[lab] = {"correct": c, "answered": answered, "n": n, "of": len(key),
                    "p_one_sided": round(binomtest(c, n, 0.5, alternative="greater").pvalue, 4) if n else None,
                    "partial_credit": round(pc / n, 3) if n else None}
        print(lab, res[lab])
    (d / "score.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
