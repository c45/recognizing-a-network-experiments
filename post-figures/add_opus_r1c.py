"""Add Claude Opus 5.5 to the post's `r1c` block (transformer mirror test).

Usage: python3 -I add_opus_r1c.py apply <data.json>
       python3 -I add_opus_r1c.py extract <depth-benchmark folder with the replies> <score_r1c.json with opus arms>
`apply` adds, per pair member, `opus_toolfree` and `opus_python`, and totals `opus_toolfree` and `opus_python`, from
opus_r1c.json, leaving the Astra keys unchanged. `extract` rebuilds opus_r1c.json from the Opus replies
(readers_r1c/replies/opus_xhigh and analyst/runs/opusx_r1c_*, effort xhigh), which are not in this repository,
with the same answer parser as make_website_data.py.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_website_data import parse  # noqa: E402

RESULTS = HERE / "opus_r1c.json"


def extract(depth, score_path):
    depth, score = Path(depth), json.loads(Path(score_path).read_text())
    key = json.loads((depth / "realistic" / "readers_r1c" / "key.json").read_text())

    def answer(path):
        return parse(path.read_text()) if path.exists() else None
    answers = {bid: {"toolfree": answer(depth / "realistic" / "readers_r1c" / "replies" / "opus_xhigh" / bid / "answer.md"),
                     "python": answer(depth / "analyst" / "runs" / f"opusx_r1c_{bid}" / "final_answer.md")}
               for bid in sorted(key)}
    RESULTS.write_text(json.dumps({"answers": answers, "totals": {"toolfree": score["opus_xhigh"],
                                                                  "python": score["opus_python"]}}, indent=1) + "\n")


def apply(data_path):
    data, res = json.loads(Path(data_path).read_text()), json.loads(RESULTS.read_text())
    for pair in data["r1c"]["pairs"]:
        for member in (pair["trained"], pair["mirror"]):
            member["opus_toolfree"] = res["answers"][member["id"]]["toolfree"]
            member["opus_python"] = res["answers"][member["id"]]["python"]
    data["r1c"]["totals"]["opus_toolfree"] = res["totals"]["toolfree"]
    data["r1c"]["totals"]["opus_python"] = res["totals"]["python"]
    Path(data_path).write_text(json.dumps(data, separators=(",", ":")))


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(*sys.argv[2:4])
    else:
        apply(sys.argv[2])
