"""Three Astra arms on the twin networks (see LADDER2_PREREG.md).

Usage: python3 -I build_ladder2.py
A. Question ladder on the second twin set: suite_twins2_hint1, suite_twins2_hint2 (the hint paragraphs of
   build_hint.py) and suite_twins2_fc (the forced choice of build_bridge.py), each the 18 suite_twins2 prompts with
   only the question changed.
B. Open question at effort max: no new prompts; the 36 open prompts of suite_twins and suite_twins2.
C. Python with the behaviour question: suite_twins_pystance holds the 24 camouflaged networks of both twin sets
   (nets/<id> links, merged key.json) and analyst/analyst_loop_stance.py is analyst_loop.py with the hint2 question.
Writes ladder2_jobs.txt, one "<suite> <id> <arm>" line per call, in running order.
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # python -I leaves the script folder off the path
from build_bridge import FC_QUESTION, GAMES, PAIR, replace_question  # noqa: E402
from build_hint import ANCHOR, HINT1, HINT2, STANCE_ANCHOR, STANCE_LINE  # noqa: E402

SEED = 20261110
LIMIT_WAIT = """            if LIMIT.search(err) and attempt < 60:
                print(f"{rd.name}: usage limit, waiting 10 min", flush=True)
                time.sleep(600)
                continue
"""
LIMIT_STOP = """            if LIMIT.search(err):
                break
"""


def add_hint(text, hint, stance):
    assert text.count(ANCHOR) == 1 and text.count(STANCE_ANCHOR) == 1
    i = text.index(ANCHOR) + len(ANCHOR)
    text = text[:i] + hint + text[i:]
    if stance:
        j = text.index(STANCE_ANCHOR)
        text = text[:j] + STANCE_LINE + text[j:]
    return text


def write_suite(name, key, prompts, rng):
    out = HERE / name
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    for nid, text in prompts.items():
        (out / "prompts" / f"{nid}.txt").write_text(text)
    (out / "key.json").write_text(json.dumps(key, indent=1))
    order = list(key)
    rng.shuffle(order)
    (out / "order.txt").write_text("\n".join(order) + "\n")
    return order


def main():
    rng = random.Random(SEED)
    src = HERE / "suite_twins2"
    key = json.loads((src / "key.json").read_text())
    text = {nid: (src / "prompts" / f"{nid}.txt").read_text() for nid in key}

    # A. ladder on set 2
    orders = {}
    for name, hint, stance in (("suite_twins2_hint1", HINT1, False), ("suite_twins2_hint2", HINT2, True)):
        orders[name] = write_suite(name, key, {n: add_hint(t, hint, stance) for n, t in text.items()}, rng)
    # Option order is balanced so that a constant letter scores chance: truth is A on 4 of the 9 twins and 5 of the
    # 9 known games, and every game has both letters.
    while True:
        letters = {}
        for twin, n_a in ((True, 4), (False, 5)):
            ids = [n for n, k in key.items() if k["twin"] == twin]
            rng.shuffle(ids)
            letters.update({n: i < n_a for i, n in enumerate(ids)})
        if all(len({letters[n] for n, k in key.items() if k["game"] == g}) == 2 for g in PAIR):
            break
    fc_key, fc_prompts = {}, {}
    for nid, k in key.items():
        g, other = k["game"], PAIR[k["game"]]
        first = letters[nid]
        a, b = (g, other) if first else (other, g)
        fc_prompts[nid] = replace_question(text[nid], FC_QUESTION.format(a=GAMES[a], b=GAMES[b]))
        fc_key[nid] = {**k, "option_A": a, "option_B": b, "truth_letter": "A" if first else "B"}
    orders["suite_twins2_fc"] = write_suite("suite_twins2_fc", fc_key, fc_prompts, rng)
    ladder = [f"{s} {n} high" for trio in zip(*orders.values()) for s, n in zip(orders, trio)]

    # C. Python with the behaviour question, on the camouflaged networks of both sets
    py = HERE / "suite_twins_pystance"
    (py / "nets").mkdir(parents=True, exist_ok=True)
    pkey = {}
    for s in ("suite_twins", "suite_twins2"):
        for nid, k in json.loads((HERE / s / "key.json").read_text()).items():
            if k["condition"] == "camo":
                pkey[nid] = {**k, "source_suite": s}
                link = py / "nets" / nid
                if not link.exists():
                    link.symlink_to(Path("..") / ".." / s / "nets" / nid)
                assert (link / "actor.txt").exists()
    (py / "key.json").write_text(json.dumps(pkey, indent=1))
    twins = [n for n, k in pkey.items() if k["twin"]]
    known = [n for n, k in pkey.items() if not k["twin"]]
    rng.shuffle(twins)
    rng.shuffle(known)
    (py / "order.txt").write_text("\n".join(twins + known) + "\n")
    loop = (HERE / "analyst" / "analyst_loop.py").read_text()
    for old in (ANCHOR, STANCE_ANCHOR, LIMIT_WAIT):
        assert loop.count(old) == 1, old
    loop = add_hint(loop, HINT2, True).replace(LIMIT_WAIT, LIMIT_STOP)
    (HERE / "analyst" / "analyst_loop_stance.py").write_text(
        "# analyst_loop.py with the hint2 question of build_hint.py; a usage limit ends the conversation. "
        "Built by build_ladder2.py.\n" + loop)
    pystance = [f"suite_twins_pystance {n} pystance" for n in twins + known]

    # B. open question at max, with a high-effort reply on the same network run beside it
    nets = [(s, n) for s in ("suite_twins", "suite_twins2") for n in json.loads((HERE / s / "key.json").read_text())]
    rng.shuffle(nets)
    effort = [f"{s} {n} {arm}" for s, n in nets for arm in ("max", "high2")]

    jobs = ladder + pystance + effort
    (HERE / "ladder2_jobs.txt").write_text("\n".join(jobs) + "\n")
    print(len(ladder), "ladder,", len(pystance), "python + stance,", len(effort), "open max/high2 jobs;",
          sum(v["truth_letter"] == "A" for v in fc_key.values()), "of", len(fc_key), "forced choices have truth A")


if __name__ == "__main__":
    main()
