# Layout vs behaviour: what frontier models read in camouflaged policy weights

Small grid-game policies (500 inputs = 10×10 cells × 5 channels, 5 actions) are shown to GPT-6 Astra and Claude
Opus 5.5 as raw weights, in the format of the Exp 1 prompt from *Sparks of Weight-Space Programming*. The study asks
what the readers actually use.

- **Exact camouflage** removes the first-layer "picture" of the layout and leaves play unchanged. Without tools, Opus
  still names the game on 48/48 camouflaged networks across three architectures; Astra manages 26/48.
- **Twin games** keep a known game's layout and dynamics but reverse the agent's stance toward the moving objects:
  Dodge vs Catch, Hunter vs Pac-Man, Car-tag vs Freeway. Tool-free, both models answer with the famous game, so twin
  answers are almost all wrong. With Python, or with a short scripted probe, the reversed behaviour is recovered.

In this repository the folder backs experiments [1](../../experiments/1-recognizing-the-game) and [3](../../experiments/3-twin-games)
of the post; start from those READMEs. The study's draft paper and round-by-round notebook are not included.

## Layout

```
experiments/            code, trained networks and every reader set (scripts expect this layout)
  train/                games.py (Pong, Breakout, Pac-Man, Freeway, Catch, Dodge, Hunter, Car-tag), train_game.py,
                        g_<game>[_<arch>|_s2]/ trained policies (actor.npz, occupancy_never_on.npy, train_log.json)
  suite/                Level 1 camouflage (null-space noise), 500-32-32-5 tanh
  suite_v2/             Level 2 camouflage (exact mirror copies), 500-32-32-5 tanh
  suite_relu24x3/       Level 2, 500-24-24-24-5 ReLU (Breakout camouflage superseded by suite_relu24x3_bfix/)
  suite_relu24x3_bfix/  rebuilt ReLU Breakout networks (unused inputs from the whole training record)
  suite_tanh48x1/       Level 2, 500-48-5 tanh
  suite_twins/          twin games, training seed 1 (6 games x plain + 2 camouflaged)
  suite_twins2/         twin games, training seed 2 (held-out set for the scripted probe)
  suite_bridge_fc/      round 8: the suite_twins networks with a forced choice between the pair's two games
  suite_bridge_open/    round 8: the depth benchmark's 12 depth-2 networks with an open rule question
  analyst/              analyst_loop.py: the with-Python arm (sandboxed code runs, up to 30)
  coding_packets/, side_packets*/, twin_packets*/   blind-coding inputs and codes (mappings kept outside the packets)
external/fetch_upstream.sh   fetches the post's prompt template and released networks (third-party)
```

Each suite has `PREREG.md` (where pre-registered), `key.json` (the truth, kept away from readers), `prompts/`,
`nets/<id>/actor.txt` (weights as printed). Model replies and scores are not included.

## Main scripts (in `experiments/`)

| Step | Script |
|---|---|
| Train a policy | `train/train_game.py <game> <out> --arch tanh32x2\|relu24x3\|tanh48x1 --seed N` |
| Architectures, prompt text, weight format | `arch.py`, `weights_text.py` |
| Exact camouflage (Level 2, any architecture) | `camouflage_arch.py` (`--never-on` takes unused inputs from training) |
| Build a blinded suite | `build_suite_arch.py`, `build_suite_twins.py`, `build_fix_relu_breakout.py` |
| Check exactness in play | `eval_play.py` |
| Layout legibility of row statistics | `battery.py`, `legibility.py` |
| Readers | `run_astra.sh`, `run_claude.sh`, `analyst/analyst_loop.py`; `job_arch.sh <tf\|an> <suite> <astra\|opus> <id>`; `run_astra_max.sh` and the `*_max` analyst loops for the max-effort arms |
| Scoring | `score_suite.py` (game names), `score_twins.py` (blind-coded stance), `arch_stats.py`, `analyst_grounding.py`; `models.py` (model keys; skips replies that ended at the output cap) |
| Stance labels checked in play | `state_probe_own_play.py` → `state_probe_own_play.json` |
| Scripted probe | `scripted_baseline.py` (locked rule for held-out suites) |
| Bridge to the depth benchmark (round 8) | `build_bridge.py`, `score_bridge.py`; `BRIDGE_PREREG.md`; Astra xhigh via `ASTRA_SCRIPT=./run_astra_xhigh.sh` |
| Post figure data for the twin rows | `make_ladder_v2.py` |

Setup: `pip install -r requirements.txt`; run `external/fetch_upstream.sh`; create the analyst sandbox venv at
`.venv-sandbox/` or set `WR_SANDBOX_PYTHON`. Input paths are in `experiments/paths.py`; set `WR_PYTHON` for the shell
scripts. Run Python that reads model output or downloaded data with `python3 -I`.

What is not stored is listed in [`DATA.md`](DATA.md) and in the repository's [`DATA.md`](../../DATA.md).
