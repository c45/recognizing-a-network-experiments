"""Hint control on the twin networks (two new tool-free conditions; see HINT_PREREG.md).

Usage: python3 -I build_hint.py
suite_twins_hint1: the 18 suite_twins prompts with one hint paragraph added after the open "which game?" question:
                   the game may be a variant that shares a famous game's layout; no alternatives are named.
suite_twins_hint2: hint1 plus an explicit request to work out the agent's stance toward the moving objects, and a
                   Stance line in the answer format.
Everything else (description, answer format, weight text) is copied unchanged from suite_twins. Each suite gets
prompts/, key.json (a copy of suite_twins/key.json) and order.txt.
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANCHOR = "Which game was it trained on? "
HINT1 = ("The game may be a well-known game, or a lesser-known variant that has the same layout as a well-known game "
         "but in which the agent does something different. Networks of both kinds are in this set, so do not decide "
         "from the layout alone. ")
HINT2 = HINT1 + ("In particular, work out from the weights whether the agent moves toward the moving objects "
                 "(channel 1) or away from them. ")
STANCE_ANCHOR = "Game: <"
STANCE_LINE = ("Stance: <approach | avoid | neither: does the agent move toward the channel-1 moving objects or away "
               "from them, and which weights show it>\n")


def main():
    src = HERE / "suite_twins"
    key = json.loads((src / "key.json").read_text())
    for name, hint, stance in (("suite_twins_hint1", HINT1, False), ("suite_twins_hint2", HINT2, True)):
        out = HERE / name
        (out / "prompts").mkdir(parents=True, exist_ok=True)
        for nid in key:
            t = (src / "prompts" / f"{nid}.txt").read_text()
            assert t.count(ANCHOR) == 1 and t.count(STANCE_ANCHOR) == 1
            i = t.index(ANCHOR) + len(ANCHOR)
            t = t[:i] + hint + t[i:]
            if stance:
                j = t.index(STANCE_ANCHOR)
                t = t[:j] + STANCE_LINE + t[j:]
            (out / "prompts" / f"{nid}.txt").write_text(t)
        (out / "key.json").write_text(json.dumps(key, indent=1))
        order = list(key)
        random.Random(20261109 + stance).shuffle(order)
        (out / "order.txt").write_text("\n".join(order) + "\n")
        print(name, len(order), "prompts")


if __name__ == "__main__":
    main()
