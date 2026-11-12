"""Minimum bridge between the twin-game study and the depth benchmark (two new tool-free conditions).

Usage: python3 -I build_bridge.py <depth-benchmark ladder/l2/readers_d dir>
A. suite_bridge_fc: the 18 suite_twins networks with the open "which game?" question replaced by a forced choice
   between the two games of the network's twin pair (rules defined; option order randomised per network).
B. suite_bridge_open: the depth benchmark's 12 depth-2 L2d networks with its forced-choice rule question replaced by an
   open question about how the paddle responds to the ball (the game description is kept; the rule hint is removed).
Each suite gets prompts/, key.json, order.txt. The networks' weight text is copied unchanged.
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GAMES = {
    "catch": "Catch: objects fall one row per step toward an agent that moves left and right along the bottom row; "
             "the agent gets +1 for each object that lands on it.",
    "dodge": "Dodge: objects fall one row per step toward an agent that moves left and right along the bottom row; "
             "an object landing on the agent gives -1 and ends the episode.",
    "pacman": "Pac-Man: the agent moves through a fixed maze collecting pellets (+1 each); two ghosts move through the "
              "maze, often chasing the agent, and touching a ghost gives -1 and ends the episode.",
    "hunter": "Hunter: the agent moves through a fixed maze collecting pellets (+1 each); two creatures move through "
              "the maze, often fleeing the agent, and touching one gives +3 (it then reappears elsewhere).",
    "freeway": "Freeway: the agent starts at the bottom and crosses six lanes of moving cars to reach the goal row at "
               "the top (+1, then it starts again); a car hitting the agent sends it back to the start.",
    "cartag": "Car-tag: the agent moves among six lanes of moving cars; touching a car gives +1 (that car then jumps "
              "elsewhere in its lane); the goal row shown at the top gives nothing.",
}
PAIR = {"catch": "dodge", "dodge": "catch", "pacman": "hunter", "hunter": "pacman", "freeway": "cartag",
        "cartag": "freeway"}

FC_QUESTION = """QUESTION. The weights below belong to one policy network that was trained with reinforcement learning to play one of these two games on this grid:
  (A) {a}
  (B) {b}
The two games look alike in which grid cells and channels they use; they differ in what the agent does. Which of the two games was this network trained on? Work from the weights. You have no tools: no code execution, no calculator, nothing to run. This is a single response.

Answer in this order:
Observations: <what you see in the weights that bears on the question; quote the weights you rely on>
Reasoning: <how those observations decide between (A) and (B)>
Answer: <A | B>
Confidence: <50-100>%

"""

OPEN_QUESTION = """QUESTION. The weights below belong to one policy network trained on this game. It imitates a fixed rule for moving the paddle in response to the ball. What is that rule? Describe precisely how the paddle moves in response to the ball. Work from the weights. You have no tools: no code execution, no calculator, nothing to run. This is a single response.

Answer in this order:
Observations: <what you see in the weights that bears on the question; quote the weights you rely on>
Reasoning: <how those observations determine the rule>
Rule: <the rule, stated precisely: what the paddle does in each situation>
Confidence: <0-100>%

"""


def replace_question(text, new):
    i = text.index("QUESTION.")
    j = text.index("THE NETWORK'S WEIGHTS")
    return text[:i] + new + text[j:]


def main(readers_d):
    rng = random.Random(20261108)
    src = HERE / "suite_twins"
    out = HERE / "suite_bridge_fc"
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    key = json.loads((src / "key.json").read_text())
    fc = {}
    for nid, k in key.items():
        g, other = k["game"], PAIR[k["game"]]
        first = rng.random() < 0.5
        a, b = (g, other) if first else (other, g)
        q = FC_QUESTION.format(a=GAMES[a], b=GAMES[b])
        (out / "prompts" / f"{nid}.txt").write_text(replace_question((src / "prompts" / f"{nid}.txt").read_text(), q))
        fc[nid] = {**k, "option_A": a, "option_B": b, "truth_letter": "A" if first else "B"}
    (out / "key.json").write_text(json.dumps(fc, indent=1))
    order = list(fc)
    rng.shuffle(order)
    (out / "order.txt").write_text("\n".join(order) + "\n")

    rd = Path(readers_d)
    out = HERE / "suite_bridge_open"
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    bkey = json.loads((rd / "key.json").read_text())
    ok = {}
    for nid, net in bkey.items():
        t = (rd / "prompts" / f"{nid}.txt").read_text()
        assert "When the ball is moving RIGHT" in t
        (out / "prompts" / f"{nid}.txt").write_text(replace_question(t, OPEN_QUESTION))
        ok[nid] = {"network": net, "truth": "right-toward" if net.startswith("right_toward") else "right-away",
                   "source": str(rd / "prompts" / f"{nid}.txt")}
    (out / "key.json").write_text(json.dumps(ok, indent=1))
    order = list(ok)
    rng.shuffle(order)
    (out / "order.txt").write_text("\n".join(order) + "\n")
    print(len(fc), "forced-choice twin prompts;", len(ok), "open benchmark prompts")


if __name__ == "__main__":
    main(sys.argv[1])
