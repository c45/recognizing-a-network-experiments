# Footprints and function: the rewrite suites

This folder holds the exact rewrites of the game policies behind experiment
[2, *Rewriting the network*](../../experiments/2-rewriting-the-network) of the post. Start from that README. The
wider study it came from (relabellings, scrambled grids, new games, footprint fade, a scripted baseline and a third
reader) is not part of the post and is not included.

## Layout

```
experiments/          scripts expect this layout
  ff_paths.py         locations of the layout study and the upstream material (environment overrides)
  suite_stage5/       untouched, copy camouflage, random fill (EFC-Synth), cancelling pairs, mixed inputs (basis re-encoding)
  suite_stage6/       twin and split, ReLU random fill, ReLU + 3 layers, ReLU + 6 layers (depth inflation)
external/fetch_upstream.sh   fetches the original post's released networks and prompts (third-party)
```

Each suite has `key.json` (row and game of each network id), `order.txt`, `prompts/`, `nets/<id>/` (weights as
printed and as npz) and `checks.json` (exactness). Model replies are not included.

## Main scripts (in `experiments/`)

| Step | Script |
|---|---|
| Build the suites | `build_stage5.py`, `build_stage6.py` (with `build_prompts.py`, `obfuscate_ft.py`, `games_new.py`, `pair_check.py`) |
| Readers, no tools | `one_reply.sh` → `run_astra.sh` / `run_astra_max.sh`, `run_claude.sh` |
| Readers, Python sandbox | `analyst_loop_r.py`, `analyst_loop_r_max.py`, `one_analyst.sh` |
| Scoring at matched effort | `score_max_parity.py` (strict first-game rule; `MAX_PARITY_PREREG.md`, `MAX_PARITY_RESULTS.md`) |

Setup: `pip install -r requirements.txt`; run `external/fetch_upstream.sh`; create the analyst sandbox venv at
`.venv-sandbox/` or set `WR_SANDBOX_PYTHON`. The layout study's experiments folder is found at
`../layout-vs-behaviour/experiments` (override with `WR_LAYOUT_EXPERIMENTS`).
Run Python that reads model output or downloaded data with `python3 -I`.

`experiments/PREREG.md` is the whole study's pre-registration, kept as written; stages 5 and 6 are the parts used
here. What is not stored is listed in [`DATA.md`](DATA.md).
