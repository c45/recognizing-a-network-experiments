"""Blind packets for the hint control: both conditions in one shuffled packet set.

Usage: python3 -I pack_hint.py pack    -> hint_packets/<code>.md and hint_packets_mapping.json (outside the packet dir)
       python3 -I pack_hint.py split   -> twin_packets_hint{1,2}/codes.json + twin_packets_hint{1,2}_mapping.json,
                                          the layout score_twins.py reads (score <suite> twin_packets_hint<k>)
"""
import json
import random
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import valid_answer  # noqa: E402

SUITES = {"1": Path("suite_twins_hint1"), "2": Path("suite_twins_hint2")}
P, MAP = Path("hint_packets"), Path("hint_packets_mapping.json")

if sys.argv[1] == "pack":
    P.mkdir(exist_ok=True)
    mp = json.loads(MAP.read_text()) if MAP.exists() else {}
    have, items = set(map(tuple, mp.values())), []
    for k, s in SUITES.items():
        for d in sorted((s / "toolfree").glob("astra_*")):
            t = valid_answer(d)
            if t and (k, d.name) not in have:
                c = secrets.token_hex(3)
                mp[c] = [k, d.name]
                items.append((c, t))
    random.Random(23).shuffle(items)
    for c, t in items:
        (P / f"{c}.md").write_text(t)
    MAP.write_text(json.dumps(mp, indent=1))
    print(len(items), "packed;", len(mp), "in mapping")
else:
    mp, codes = json.loads(MAP.read_text()), json.loads((P / "codes.json").read_text())
    for k in SUITES:
        d = Path(f"twin_packets_hint{k}")
        d.mkdir(exist_ok=True)
        sel = {c: rid for c, (kk, rid) in mp.items() if kk == k}
        (d / "codes.json").write_text(json.dumps({c: codes[c] for c in sel if c in codes}, indent=1))
        Path(f"twin_packets_hint{k}_mapping.json").write_text(json.dumps(sel, indent=1))
        print(f"hint{k}: {len(sel)} answers, {sum(c in codes for c in sel)} coded")
