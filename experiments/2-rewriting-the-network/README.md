# Experiment 2: Rewriting the network

Post section: *Rewriting the network*. Figure: **Disguises** (figure 3).

## Question

Does recognition survive rewrites that change how the network is written down but not what it does? Each rewrite
removes or scrambles a different cue the models might be using.

## Setup

- **Policies.** One per game (Pong, Breakout, Pac-Man, Freeway). The first six rows use the 500 → 32 → 32 → 5 tanh
  policies, the last three the 500 → 24 → 24 → 24 → 5 ReLU policies, all from
  [experiment 1](../1-recognizing-the-game/). Each row is four networks, one per game: **36 networks**.
- **Rows** (figure 3, top to bottom):

  | Row id | Rewrite |
  |---|---|
  | `plain` | untouched |
  | `camo` | copy camouflage, as in experiment 1 |
  | `synth` | random fill: unused inputs get new weights drawn to match the used inputs of the same channel, so no two printed lines are equal |
  | `pairs` | copy camouflage plus eight cancelling pairs of second-layer units |
  | `basis` | mixed inputs: the 500 inputs mixed in 50 groups of 10 by orthogonal matrices the prompt lists |
  | `split` | random fill, then every first-layer unit gets a twin and every second-layer unit is split in two (64 units per layer) |
  | `relu0` | ReLU policies, random fill |
  | `relu3` | ReLU, random fill, plus one extra ReLU layer of 32 after each hidden layer (6 hidden layers) |
  | `relu6` | ReLU, random fill, plus two extra layers after each hidden layer (9 hidden layers) |

  Before printing, no rewrite changes a move probability by more than 10⁻¹¹ on boards met in play.
- **Prompt.** The open question from experiment 1 ("Which game was it trained on?"), every weight to 3 decimals.
- **Models.** Astra and Opus at effort max, no tools, one reply each. A reply that ran out of output was rerun once.
- **Scoring (strict).** The first game named on the `Game:` line has to be the right one ("frogger" counts as
  Freeway). One reply is scored by hand under this rule: Opus on `relu3` Pac-Man says "Dodge / Avoider ... similar to
  evading the ghosts in Pac-Man", which leads with another game and so is not credited (`MAX_PARITY_PREREG.md`,
  addendum 1; applied in `post-figures/make_rewrites.py`).
- **With Python.** The four `relu6` networks again, in a sandbox with up to 30 code runs and any analysis, including
  running the network on inputs the model builds. Both models at max.

## Results (games named, of 4)

These are the numbers reported in the post. The model replies behind them are not included in this repository;
running the readers below on the stored prompts produces new ones.

| Row | Astra (max) | Opus (max) |
|---|---|---|
| untouched | 4 | 4 |
| copy camouflage | 4 | 4 |
| random fill | 3 | 2 |
| cancelling pairs | 3 | 4 |
| mixed inputs | 3 | 4 |
| twin and split | 4 | 4 |
| ReLU, random fill | 1 | 2 |
| ReLU + 3 layers | 1 | 2 |
| ReLU + 6 layers | 1 | 2 |
| **ReLU + 6 layers, with Python** | **2** | **3** |

- On the tanh rows the switch from copy camouflage to random fill costs Astra only Breakout. On the ReLU policies it
  drops both models to 1–2 of 4, and the added layers change nothing further.
- At effort high, Astra recognized 15 of the 28 networks in the seven rewrite rows, 16 in a fresh rerun at high and
  16 at max (max against the rerun: one network changed each way).
- With Python: one Opus conversation (Breakout) reported code runs it never made. It was discarded and rerun once,
  and the rerun is counted. Astra's conversations at high, four days earlier, recognized 1 of 4.

## Files

| What | Where |
|---|---|
| Building the rewrites | [`build_stage5.py`](../../studies/footprints-and-function/experiments/build_stage5.py) (rows `plain` to `basis`), [`build_stage6.py`](../../studies/footprints-and-function/experiments/build_stage6.py) (rows `split` to `relu6`), [`obfuscate_ft.py`](../../studies/footprints-and-function/experiments/obfuscate_ft.py); exactness checks in each suite's `checks.json` |
| Suites | [`suite_stage5/`](../../studies/footprints-and-function/experiments/suite_stage5), [`suite_stage6/`](../../studies/footprints-and-function/experiments/suite_stage6): `key.json` (row and game of each id), `nets/`, `prompts/` |
| Running the readers | [`one_reply.sh`](../../studies/footprints-and-function/experiments/one_reply.sh) → [`run_astra_max.sh`](../../studies/footprints-and-function/experiments/run_astra_max.sh) / [`run_claude.sh`](../../studies/footprints-and-function/experiments/run_claude.sh); with Python: [`analyst_loop_r_max.py`](../../studies/footprints-and-function/experiments/analyst_loop_r_max.py) (Astra, max), [`analyst_loop_r.py`](../../studies/footprints-and-function/experiments/analyst_loop_r.py) (Opus) |
| Scoring | [`score_max_parity.py`](../../studies/footprints-and-function/experiments/score_max_parity.py) (uses `game_first` from [`score_matched_effort.py`](../../studies/layout-vs-behaviour/experiments/score_matched_effort.py)) |
| Pre-registration | [`MAX_PARITY_PREREG.md`](../../studies/footprints-and-function/experiments/MAX_PARITY_PREREG.md); stages 5 and 6 in [`PREREG.md`](../../studies/footprints-and-function/experiments/PREREG.md) |
| Figure data | [`post-figures/make_rewrites.py`](../../post-figures/make_rewrites.py) (scores), [`make_disguises.py`](../../post-figures/make_disguises.py) (pictures and layer shapes) |
