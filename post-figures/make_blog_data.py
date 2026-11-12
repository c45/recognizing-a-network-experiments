"""Build intermediate/layout_blocks.json (the layout study's blocks) from the result files in ../studies.

Usage: python3 -I make_blog_data.py [experiments dir] [depth numbers.tex]
Defaults: ../studies/layout-vs-behaviour/experiments and ../studies/depth-benchmark/paper_icml/numbers.tex.
make_website_data.py reads the output.

Readers are GPT-6 Astra and Claude Opus 5.5. A reader appears in a block only when its result file has rows for it.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / "studies" / "layout-vs-behaviour" / "experiments"
NUMBERS = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "studies" / "depth-benchmark" / "paper_icml" / "numbers.tex"
sys.path.insert(0, str(EXP))  # python -I leaves it off the path
from arch import parse_text  # noqa: E402

MODELS = ("astra", "opus")
ARCHS = [("32-32 tanh", "suite_v2", "tanh32x2"), ("24-24-24 ReLU", "suite_relu24x3", "relu24x3"),
         ("48 tanh", "suite_tanh48x1", "tanh48x1")]
DEPTH_WORDS = {"Two": 2, "Three": 3, "Four": 4, "Five": 5, "Six": 6, "Seven": 7, "Eight": 8, "Ten": 10, "Twelve": 12,
               "Sixteen": 16, "Twenty": 20}


def load(p):
    return json.loads(Path(p).read_text())


def rownorm_maps(w):
    """Per channel, the size of each grid cell's first-layer row, scaled to the channel's largest."""
    n = np.linalg.norm(w["W1"].reshape(10, 10, 5, -1), axis=-1)
    return {ch: (n[:, :, ch] / max(n[:, :, ch].max(), 1e-9)).round(3).tolist() for ch in (0, 1, 2)}


def pick(suite, game, cond):
    key = load(EXP / suite / "key.json")
    return next(n for n, v in key.items() if v["game"] == game and v["condition"] == cond and v["seed"] == 0
                and not v["flip"])


def footprint():
    return {lab: {c: rownorm_maps(parse_text(EXP / s / "nets" / pick(s, "pong", c) / "actor.txt"))
                  for c in ("plain", "camo")} for lab, s, _ in ARCHS}


def demo_nets():
    out = {}
    for g in ("catch", "dodge"):
        w = parse_text(EXP / "suite_twins" / "nets" / pick("suite_twins", g, "plain") / "actor.txt")
        out[g] = {k: np.round(v, 3).tolist() for k, v in w.items()}
        out[g]["maps"] = rownorm_maps(w)
    return out


def depth():
    mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", NUMBERS.read_text()))

    def series(prefix):
        pts = []
        for word, d in DEPTH_WORDS.items():
            v = mac.get(prefix + word)
            if v is None:
                continue
            if "/" in v:
                a, b = map(int, v.split("/"))
                pts.append({"d": d, "k": a, "n": b, "p": a / b})
            else:
                pts.append({"d": d, "p": float(v.replace("\\%", "")) / 100})
        return sorted(pts, key=lambda x: x["d"])

    out = {"script_full": series("FullC"), "script_3paths": series("ThreeP"), "d50": {}}
    for m, name in (("astra", "Astra"), ("opus", "Opus")):
        for arm, tag in (("tf", "TF"), ("py", "Py")):
            out[f"{m}_{arm}"] = series(f"{name}{tag}Cnt")
        if f"{name}TFDfifty" in mac:
            out["d50"][f"{m}_tf"] = [float(mac[f"{name}TFDfifty{x}"]) for x in ("", "Lo", "Hi")]
    return out


def naming():
    st = load(EXP / "arch_stats.json")
    out = {}
    for _, _, a in ARCHS:
        cells = st[a]["cells"]
        row = {"astra": cells["toolfree/astra/camo"], "opus": cells["toolfree/opus/camo"], "p": st[a]["mcnemar_p"],
               "astra_by_game": {g: v["astra"] for g, v in st[a]["camo_by_game"].items()}}
        out[a] = row
    return out, st["pooled_all"], st.get("pooled_pairs")


def twins():
    keep = ("arm", "model", "game", "twin", "condition", "stance_correct", "coded_named_game", "coded_stance")
    rows = []
    for s in ("suite_twins", "suite_twins2"):
        seen = set()
        for f in sorted((EXP / s).glob("twin_scores*.json")):
            for r in load(f):
                k = (r["arm"], r["model"], r["id"])
                if k in seen:
                    continue
                seen.add(k)
                rows.append({k: r[k] for k in keep} | {"suite": s} | ({"effort": r["effort"]} if r.get("effort") else {}))
    # analysts on the remaining camouflaged networks (round 8), coded in a separate blind pass
    extra = EXP / "analyst_extra_scores.json"
    if extra.exists():
        rows += [{k: r[k] for k in keep + ("suite",)} for r in load(extra)]
    return rows


def probe():
    return [{"suite": s, "game": r["game"], "condition": r["condition"], "score": r["score_near"], "pred": r["pred_locked"],
             "truth": r["truth"]} for s in ("suite_twins", "suite_twins2")
            for r in load(EXP / s / "scripted_baseline.json").values()]


def main():
    nm, pooled, pooled_pairs = naming()
    out = {"footprint": footprint(), "nets": demo_nets(), "depth": depth(), "naming": nm, "naming_pooled": pooled,
           "twins": twins(), "probe": probe(),
           "bridge": {"fc": load(EXP / "suite_bridge_fc" / "bridge_scores.json"),
                      "open": load(EXP / "suite_bridge_open" / "bridge_scores.json")}}
    if pooled_pairs:
        out["naming_pooled_pairs"] = pooled_pairs
    (HERE / "intermediate" / "layout_blocks.json").write_text(json.dumps(out, separators=(",", ":")))
    for m in MODELS:
        print(m, "twin rows", sum(r["model"] == m for r in out["twins"]),
              "| bridge fc", sum(r["model"] == m for r in out["bridge"]["fc"]),
              "| bridge open", sum(r["model"] == m for r in out["bridge"]["open"]),
              "| naming", [out["naming"][a].get(m) for _, _, a in ARCHS])


if __name__ == "__main__":
    main()
