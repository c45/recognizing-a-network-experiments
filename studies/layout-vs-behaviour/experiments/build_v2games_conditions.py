"""Question conditions for the rebuilt twin games (V2GAMES_PREREG.md).

Usage: python3 -I build_v2games_conditions.py
For each rebuilt suite (suite_twins_v2games, suite_twins2_v2games; open "which game?" prompts already in prompts/):
  <suite>_fc     forced choice between the two games of the pair (question text of build_bridge.py, new rule sentences)
  <suite>_hint1  open question + variant warning (text of build_hint.py)
  <suite>_hint2  hint1 + stance question and Stance line (text of build_hint.py)
Weight text is copied unchanged. Each new folder gets prompts/, key.json and order.txt.
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_bridge import FC_QUESTION, GAMES as V1, replace_question  # noqa: E402
from build_hint import ANCHOR, HINT1, HINT2, STANCE_ANCHOR, STANCE_LINE  # noqa: E402

# Rule sentences from the rebuild's hand-off (twin_audit_20261110/AUDIT.md); Pac-Man is the unchanged v1 sentence.
RULES = {
    "cartag_v2": "Car-tag: the agent moves among six lanes of moving cars; stepping onto a car gives +1 (that car then "
                 "jumps elsewhere in its lane); the goal row shown at the top cannot be entered and gives nothing.",
    "hunter_v2": "Hunter: the agent moves through a fixed maze; two creatures move through the maze, often fleeing "
                 "the agent, and touching one gives +1 (it then reappears elsewhere); the pellets shown disappear when "
                 "walked over and give nothing.",
    "freeway_v2": "Freeway: the agent starts at the bottom and crosses six lanes of moving cars to reach the goal row "
                  "at the top (+1, then it starts again); a car hitting the agent ends the episode.",
    "pacman": V1["pacman"],
}
PAIR = {"cartag_v2": "freeway_v2", "freeway_v2": "cartag_v2", "hunter_v2": "pacman"}


def main():
    for si, suite in enumerate(("suite_twins_v2games", "suite_twins2_v2games")):
        src = HERE / suite
        key = json.loads((src / "key.json").read_text())
        rng = random.Random(20261110 + si)
        fc = {}
        out = HERE / f"{suite}_fc"
        (out / "prompts").mkdir(parents=True, exist_ok=True)
        for nid, k in key.items():
            g, other = k["game"], PAIR[k["game"]]
            first = rng.random() < 0.5
            a, b = (g, other) if first else (other, g)
            q = FC_QUESTION.format(a=RULES[a], b=RULES[b])
            (out / "prompts" / f"{nid}.txt").write_text(replace_question((src / "prompts" / f"{nid}.txt").read_text(), q))
            fc[nid] = {**k, "option_A": a, "option_B": b, "truth_letter": "A" if first else "B"}
        (out / "key.json").write_text(json.dumps(fc, indent=1))
        order = list(key)
        rng.shuffle(order)
        (out / "order.txt").write_text("\n".join(order) + "\n")
        for name, hint, stance in (("hint1", HINT1, False), ("hint2", HINT2, True)):
            out = HERE / f"{suite}_{name}"
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
            rng.shuffle(order)
            (out / "order.txt").write_text("\n".join(order) + "\n")
        print(suite, len(key), "networks;", sum(v["truth_letter"] == "A" for v in fc.values()), "A /",
              sum(v["truth_letter"] == "B" for v in fc.values()), "B")


if __name__ == "__main__":
    main()
