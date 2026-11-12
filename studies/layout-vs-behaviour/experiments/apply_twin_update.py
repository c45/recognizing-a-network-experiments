"""Apply the rebuilt-games twin update to the post (index.md, copy.json, data.json). Usage: python3 -I apply_twin_update.py <post dir> <obscured dir>"""
import json
import subprocess
import sys
from pathlib import Path

post, obs = Path(sys.argv[1]), Path(sys.argv[2])
subprocess.run([sys.executable, "-I", str(obs / "make_ladder_v2.py"), str(post / "data.json")], check=True, stdout=subprocess.DEVNULL)
d = json.loads((post / "data.json").read_text())
L = {s["id"]: s for s in d["ladder"]}
a, o = (lambda q, k="twin": L[q]["astra_max"][k]), (lambda q, k="twin": L[q]["opus"][k])
W = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
of = lambda c: f"{c[0]} of {c[1]}"
all_or = lambda c: f"all {c[1]}" if c[0] == c[1] else of(c)
ox = L["forced"]["opus"]["xhigh"]
og = lambda q, g: L[q]["opus"]["games"][g]
ag = lambda q, g: L[q]["astra_max"]["games"][g]
assert a("open") == [9, 18] and ag("open", "dodge") == [6, 6] and ag("open", "hunter") == [3, 6] and ag("open", "cartag") == [0, 6]
assert a("hint1")[1] == 18 and a("hint2") == [18, 18] and a("forced") == [18, 18]
assert og("open", "cartag")[0] == 0 and og("open", "dodge") == [0, 2] and og("forced", "dodge") == [1, 3]
assert o("forced")[1] - o("forced")[0] == 2, "text says Opus's two forced-choice misses are both Dodge"
assert L["python"]["astra_max"]["games"] == {"dodge": [4, 4], "hunter": [4, 4], "cartag": [0, 4]} and L["python_stance"]["astra_max"]["twin"] == [12, 12]
assert L["python"]["opus"]["twin"] == [1, 1]

md = (post / "index.md").read_text()
cp = (post / "copy.json").read_text()


def rep(text, old, new, n=1):
    assert text.count(old) == n, (text.count(old), old[:70])
    return text.replace(old, new)


def repj(text, old, new):  # a JSON string value in copy.json
    return rep(text, json.dumps(old, ensure_ascii=False), json.dumps(new, ensure_ascii=False))


# ---- index.md (text as simplified on main, PR 31)
md = rep(md, "Opus describes what it does for 0 of 8, Astra for 6 of 18. Define the two candidate games and ask which one it is, and Astra gets 9 of 9, Opus 7.",
         f"Opus describes what it does for {of(o('open'))}, Astra for {of(a('open'))}. Define the two candidate games and ask which one it is, and Astra gets {of(a('forced'))}, Opus {of(o('forced'))}.")
md = rep(md, "Car-tag touches the cars that Freeway avoids.\n",
         "Car-tag touches the cars that Freeway avoids. (Hunter, Car-tag and Freeway are second versions, retrained after my first Hunter policies turned out never to chase.)\n")
md = rep(md, "Both models walk straight into the trap. Both call Hunter Pac-Man and Car-tag Frogger, Freeway's better-known cousin, and Opus calls Dodge Catch. Astra describes the right behaviour for 6 of the 18 twins, Opus for 0 of 8. The skill that took Opus to 48 of 48 on the camouflaged networks takes it to 0 of 8 here.",
         f"Both models walk into the trap. Both call Car-tag Frogger, Freeway's better-known cousin, every time, and Opus calls Dodge Catch. Astra reads Hunter as a chase about half the time. Astra describes the right behaviour for {a('open')[0]} of the {a('open')[1]} twins, Opus for {o('open')[0]} of the {o('open')[1]} it has answered so far. The skill that took Opus to 48 of 48 on the camouflaged networks takes it to {of(o('open'))} here.")
md = rep(md, "Warn Astra, at its high setting, that the game may be a look-alike and it goes from 1 right of 18 to 11. Ask as well whether the player moves towards the moving objects or away from them, and it gets 13. Replace the open question with a _forced choice_ between the two games and Astra gets all 9, Opus 7.",
         f"Warn Astra that the game may be a look-alike and it goes from {a('open')[0]} right of 18 to {a('hint1')[0]}. Ask as well whether the player moves towards the moving objects or away from them, and it gets all 18. Replace the open question with a _forced choice_ between the two games and Astra again gets all 18, Opus {of(o('forced'))}.")
md = rep(md, "Python rescues Dodge (0 of 4 to 4 of 4) and nothing else: Hunter and Car-tag stay at 0 of 8.",
         "Python lifts Hunter (2 of 4 to 4 of 4) and nothing else: Car-tag stays at 0 of 4, still called Frogger.")
md = rep(md, "The warning question, the behaviour question and the mirror test ran on Astra only.",
         "Opus's answers on the twins are still coming in, so its counts there are smaller than Astra's. The mirror test ran on Astra only.")
(post / "index.md").write_text(md)

# ---- copy.json
c = json.loads(cp)
lad, sb = c["ladder"], c["scoreboard"]
item = lambda block, label: next(i["text"] for i in block["setup"] if i["label"] == label)
cp = repj(cp, item(lad, "Games"), "three pairs sharing a board: Catch / Dodge (falling objects), Pac-Man / Hunter (the creatures flee; touching one pays +1 and the pellets pay nothing), Freeway / Car-tag (stepping onto a car pays +1; the goal row cannot be entered). The second of each pair is the twin. Hunter, Car-tag and Freeway are second versions: in the first, the Hunter policies ignored the creatures and one Freeway policy only ever moved up, so I changed their rewards and retrained them. Dodge, Catch and Pac-Man are unchanged.")
cp = repj(cp, item(lad, "Rows 1 to 4"), "18 networks per training run, the same in every row: 9 twins and 9 famous-game networks. No tools, both models at effort max. Astra's bars are both training runs. Opus's are still filling in, mostly from the first run; the count beside each bar says how many it has answered. In row 4 its 3 Dodge answers are reruns at xhigh.")
cp = repj(cp, item(lad, "Row 4"), "the question is replaced by a choice between the pair's two games, each defined in one sentence. The order of the two options varies from network to network: the right answer is A for 11 of the 18 twins.")
cp = repj(cp, item(lad, "Rows 5 and 6"), "the 12 camouflaged twins of both training runs (and the 12 camouflaged famous-game networks), Astra at effort max. The printed weights in a sandbox, up to 30 code runs, any analysis, including running the network on inputs the model builds. Row 5 asks the open question; row 6 adds row 3's warning and behaviour question, in separate conversations. Opus has run one of these networks, a Dodge twin in row 5.")
cp = repj(cp, item(lad, "Counts"), "a network with no finished reply is left out. Opus replies that hit the output limit at max are rerun at xhigh and count once they answer: so far 3 Dodge twins and one Catch network, all in row 4.")
cp = repj(cp, lad["notes"]["twin"], "Opus's bars are still filling in: read the counts beside them. Point at a bar for the split by game.")
cp = repj(cp, lad["notes"]["known"], "The same questions on the famous-game networks. Pointed questions barely hurt. Some replies are still running.")
cp = repj(cp, item(sb, "Effort"), "xhigh for both models in the paddle-rule and transformer rows; max for both in the game-naming row and the two twin rows. In the forced-choice row 3 of Opus's replies hit the output limit at max and were rerun at xhigh.")
cp = repj(cp, item(sb, "Counts"), "networks right out of each model's scored replies; the no-tools paddle row uses the 242 networks both models were given. Opus's twin replies are still coming in, so it has fewer twins than Astra in both twin rows.")
old_b = item(sb, "Breakdowns")
frag = "Opus has 2 Dodge twins in the open question because one reply was never finished"
assert old_b.count(frag) == 1
cp = repj(cp, old_b, old_b.replace(frag, "in the twin tests Opus has fewer networks per game than Astra because its replies are still coming in"))
(post / "copy.json").write_text(cp)
json.loads(cp)
print("ok", {q: (a(q), L[q].get("opus", {}).get("twin")) for q in L}, "words:", len(md.split()))
