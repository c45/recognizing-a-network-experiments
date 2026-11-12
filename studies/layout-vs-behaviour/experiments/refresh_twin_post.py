"""Refresh the post's twin numbers in place, on the text as it is after PR 32 (use this for every later refresh).

Usage: python3 -I refresh_twin_post.py <post dir>
Regenerates the `ladder` / `twins` blocks (make_ladder_v2.py: Astra at max, Opus at max only; Opus replies answered
only below max are left out and counted), then rewrites the sentences and Setup items that quote twin counts.
Each pattern must match exactly once, so a text change on main makes this fail loudly instead of silently.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

post, here = Path(sys.argv[1]), Path(__file__).resolve().parent
subprocess.run([sys.executable, "-I", str(here / "make_ladder_v2.py"), str(post / "data.json")], check=True,
               stdout=subprocess.DEVNULL)
L = {s["id"]: s for s in json.loads((post / "data.json").read_text())["ladder"]}
A = lambda q: L[q]["astra_max"]["twin"]  # noqa: E731
O = lambda q: L[q]["opus"]["twin"]  # noqa: E731
of = lambda c: f"{c[0]} of {c[1]}"  # noqa: E731
allof = lambda c: f"all {c[1]}" if c[0] == c[1] else of(c)  # noqa: E731
left = {q: L[q].get("opus", {}).get("left_out", 0) for q in L}
left_total, left_fc = sum(left.values()), left["forced"]
fc_dodge = L["forced"]["opus"]["games"]["dodge"]


def sub(text, pattern, new):
    n = len(re.findall(pattern, text))
    assert n == 1, (n, pattern[:70])
    return re.sub(pattern, lambda m: new, text)


def maybe(text, pattern, new):
    """Like sub, for a sentence a later text rewrite may have removed: 0 matches is allowed, more than 1 is not."""
    n = len(re.findall(pattern, text))
    assert n <= 1, (n, pattern[:70])
    return re.sub(pattern, lambda m: new, text) if n else text


N = r"\d+"
md = (post / "index.md").read_text()
md = maybe(md, rf"Opus describes what it does for {N} of {N}, Astra for {N} of {N}\. Define the two candidate games and ask which one it is, and Astra gets {N} of {N}, Opus {N} of {N}\.",
         f"Opus describes what it does for {of(O('open'))}, Astra for {of(A('open'))}. Define the two candidate games and ask which one it is, and Astra gets {of(A('forced'))}, Opus {of(O('forced'))}.")
md = sub(md, rf"Opus for {N} of the {N} it has answered so far\. The skill that took Opus to 48 of 48 on the camouflaged networks takes it to {N} of {N} here\.",
         f"Opus for {O('open')[0]} of the {O('open')[1]} it has answered so far. The skill that took Opus to 48 of 48 on the camouflaged networks takes it to {of(O('open'))} here.")
od = L["open"]["opus"]["games"]["dodge"]
dodge_clause = ("and Opus calls Dodge Catch" if od[0] == 0 else
                f"and Opus calls {od[1] - od[0]} of its {od[1]} Dodge networks Catch" if od[0] < od[1] else "")
assert dodge_clause, "Opus reads every Dodge: rewrite the trap sentence by hand"
md = sub(md, r"and Opus calls (?:Dodge Catch|\d+ of its \d+ Dodge networks Catch)", dodge_clause)
md = sub(md, rf"Astra again gets all 18, Opus (?:{N} of {N}|all {N})(?: it has answered so far)?\.",
         f"Astra again gets all 18, Opus {allof(O('forced'))}" + (" it has answered so far." if O("forced")[1] < 18 else "."))
md = maybe(md, rf"Given a forced choice between the two games, Astra gets {N} of {N} and Opus {N} of {N}\.",
           f"Given a forced choice between the two games, Astra gets {of(A('forced'))} and Opus {of(O('forced'))}.")
md = sub(md, r"Opus's answers on the twins are still coming in, so its counts there are smaller than Astra's\.(?: [^.]*output at max[^.]*\.)?",
         "Opus's answers on the twins are still coming in, so its counts there are smaller than Astra's. When Opus ran out of "
         "output at max and answered only on a rerun one step lower, I leave that answer out, so every twin count compares "
         "the two models at max.")
(post / "index.md").write_text(md)

cp = (post / "copy.json").read_text()
c = json.loads(cp)
item = lambda block, label: next(i["text"] for i in c[block]["setup"] if i["label"] == label)  # noqa: E731


def setj(text, old, new):
    o = json.dumps(old, ensure_ascii=False)
    assert text.count(o) == 1, old[:60]
    return text.replace(o, json.dumps(new, ensure_ascii=False))


dodge_note = ("" if fc_dodge[1] else "; in row 4 that is all three Dodge twins, so Opus has no Dodge count there yet")
r14 = item("ladder", "Rows 1 to 4")
cp = setj(cp, r14, re.sub(r" (In row 4 its 3 Dodge answers are reruns at xhigh\.|Opus answers given only at xhigh[^.]*\.)$",
                          f" Opus answers given only at xhigh, after running out of output at max, are left out{dodge_note}.", r14))
r56 = item("ladder", "Rows 5 and 6")
py_o = L["python"].get("opus", {}).get("twin", [0, 0])
cp = setj(cp, r56, re.sub(r"Astra at effort high, Opus at max\.|Astra at effort max(, Opus too)?\.", "Astra at effort max, Opus too.",
                          re.sub(r" (Opus has run one of these networks, a Dodge twin in row 5\.|Opus has run [^.]*\.)$",
                                 "" if py_o[1] >= 12 else f" Opus has run {py_o[1]} of the 12 twins in row 5.", r56)))
cp = setj(cp, item("ladder", "Counts"),
          f"a network with no finished reply at max is left out. That includes Opus replies that ran out of output at max "
          f"and answered only on a rerun at xhigh: so far {left_total}, {left_fc} of them in row 4.")
eff = item("scoreboard", "Effort")
cp = setj(cp, eff, re.sub(r" ?(In the forced-choice row 3 of Opus's replies hit the output limit at max and were rerun at xhigh\.|Opus replies answered only at xhigh[^.]*\.)$",
                          " Opus replies answered only at xhigh, after hitting the output limit at max, are left out of the twin rows"
                          + ("" if fc_dodge[1] else ", so Opus has no Dodge twin in the forced-choice row yet") + ".", eff))
(post / "copy.json").write_text(cp)
json.loads(cp)
print("ok", {q: (A(q), L[q].get("opus", {}).get("twin")) for q in L}, "left out", left, "words", len(md.split()))
