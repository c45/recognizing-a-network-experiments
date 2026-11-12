"""Audit of the 12 twin-study policies in play. Writes audit_existing.json and prints a table.

Usage: python -I audit_existing.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))
from audit_lib import camp_policy, const_policy, load, random_policy, run  # noqa: E402
from behaviour_checks import causal_probe  # noqa: E402

PAIRS = [("catch", "dodge"), ("pacman", "hunter"), ("freeway", "cartag")]
out = {}


def show(tag, m):
    ci = m["contact_index"]
    extra = "".join(f" {k}={m[k]}" for k in ("goal_arrivals", "contacts_while_still", "pellets", "causal") if k in m)
    print(f"{tag:34s} ret {m['return']:8.2f}  contacts {m['contacts']:7.2f}  toward {m['toward_nearest']:+.3f}"
          f"  contact idx {ci['policy']:.3f} vs {ci['uniform']:.3f}{extra}", flush=True)


for famous, twin in PAIRS:
    for game, other in ((famous, twin), (twin, famous)):
        for suf in ("", "_s2"):
            w = load(game, suf)
            name = game + suf
            m = run(game, w, keep_states=True)
            m["causal"] = round(causal_probe(w, m.pop("_X").astype(float)), 3)
            m.pop("_P")
            out[f"{name}|own|true"] = m
            show(f"{name} own", m)
            for mode in ("decoy", "blind"):
                out[f"{name}|own|{mode}"] = run(game, w, obs_mode=mode)
                show(f"{name} own {mode}", out[f"{name}|own|{mode}"])
            out[f"{name}|own|argmax"] = run(game, w, argmax=True)
            show(f"{name} own argmax", out[f"{name}|own|argmax"])
            out[f"{name}|in_{other}|true"] = run(other, w)
            show(f"{name} in {other}", out[f"{name}|in_{other}|true"])
        out[f"random|{game}"] = run(game, random_policy)
        show(f"random in {game}", out[f"random|{game}"])
out["always_up|freeway"] = run("freeway", const_policy(1))
show("always up in freeway", out["always_up|freeway"])
for row in range(2, 8):
    for sweep in (False, True):
        k = f"camp_row{row}{'_sweep' if sweep else ''}|cartag"
        out[k] = run("cartag", camp_policy(row, sweep))
        show(k, out[k])
(HERE / "audit_existing.json").write_text(json.dumps(out, indent=1))
