"""Rebuild the post's figure data from ../studies and check it against the published copy.

Usage: uv run --with numpy python -I build_post_data.py [--write]
Steps, each a script in this repository:
  1. make_blog_data.py                 layout-study blocks -> intermediate/layout_blocks.json
  2. make_website_data.py              pictures, play, depth, transformers, r1b, r1c, naming, replies -> data.json
  3. make_ladder_v2.py (studies/layout-vs-behaviour/experiments)   twin blocks `ladder` and `twins`
  4. make_rewrites.py                  `rewrites` and `rewrites_baseline`
  5. make_disguises.py                 `disguises` (pictures and layer shapes of the rewritten networks)
  6. add_opus_r1c.py apply             Opus answers on the mirror test (`r1c`), from opus_r1c.json
Writes intermediate/data.json, then compares every top-level block with published_data.json (the data.json the
live post serves) and exits 1 on any difference. With --write it also replaces published_data.json.
pictures.json, play.json and prompt_tokens.json are inputs: make_pictures.py, make_play.py and make_prompt_tokens.py
rebuild them, but need the fetched upstream networks or the raw event streams (see ../DATA.md).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDIES = HERE.parent / "studies"
OUT = HERE / "intermediate" / "data.json"
PUBLISHED = HERE / "published_data.json"


def run(script, *args, cwd=HERE):
    print(f"$ {Path(script).name} {' '.join(map(str, args))}")
    subprocess.run([sys.executable, "-I", str(script), *map(str, args)], check=True, cwd=cwd, stdout=subprocess.DEVNULL)


def main():
    OUT.parent.mkdir(exist_ok=True)
    run(HERE / "make_blog_data.py")
    run(HERE / "make_website_data.py", OUT)
    run(STUDIES / "layout-vs-behaviour" / "experiments" / "make_ladder_v2.py", OUT)
    run(HERE / "make_rewrites.py", OUT)
    run(HERE / "make_disguises.py", STUDIES / "footprints-and-function" / "experiments", OUT)
    data = json.loads(OUT.read_text())
    data["disguises"] = json.loads((HERE / "disguises.json").read_text())
    (HERE / "disguises.json").replace(HERE / "intermediate" / "disguises.json")
    OUT.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    run(HERE / "add_opus_r1c.py", "apply", OUT)   # Opus on the mirror test, from opus_r1c.json
    data = json.loads(OUT.read_text())

    published = json.loads(PUBLISHED.read_text())
    bad = []
    for key in sorted(set(data) | set(published)):
        same = data.get(key) == published.get(key)
        state = "same" if same else ("missing from rebuild" if key not in data else
                                     "not in published" if key not in published else "DIFFERENT")
        print(f"  {key:18} {state}")
        if not same:
            bad.append(key)
    if "--write" in sys.argv:
        shutil.copyfile(OUT, PUBLISHED)
        print("published_data.json replaced")
    elif bad:
        print(f"{len(bad)} block(s) differ from the published data: {', '.join(bad)}")
        sys.exit(1)
    else:
        print(f"all {len(data)} blocks match the published data")


if __name__ == "__main__":
    main()
