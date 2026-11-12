# Experiment 1: Recognizing the game from the weights

Post sections: *The original weight-reading experiment*, *The weights contain a picture of the board*,
*Camouflaging the input layer*. Figures: **PlayAndRead** (figure 1), **BoardPictures** (figure 2).

## Question

Given only the printed weights of a small game-playing policy, can a model say which game it was trained on? And
does it still manage once the most obvious cue, a picture of the board in the first layer, is hidden?

## Setup

- **Networks.** Policies that look at a 10 × 10 board with five on-off channels per cell (player, moving objects,
  reward targets, obstacles, previous frame), 500 inputs in all, and pick one of five moves. Four games: Pong,
  Breakout, Pac-Man, Freeway. Three shapes, trained here with PPO for 50 million steps:
  500 → 32 → 32 → 5 tanh (17,253 weights, the original post's shape), 500 → 24 → 24 → 24 → 5 ReLU, and 500 → 48 → 5 tanh.
- **Why the first layer is a picture.** An input that is never on during training gets no gradient, so its
  first-layer weights keep their starting size. Folded back onto the board, the first layer shows where the game's
  pieces can be (figure 2, drawn on the four networks released with *Sparks of Weight-Space Programming*).
- **Camouflage.** Every unused input gets an exact copy of a used input's first-layer weights (its mirror-image cell
  if that is used, else a random used input in the same channel). Hidden units are then shuffled and sign-flipped
  (tanh) or rescaled (ReLU), and the output layer gets a shift the softmax ignores. Play is unchanged. Each of the
  12 policies (4 games × 3 shapes) gets four camouflaged copies (two seeds, each also with the grid mirrored), so
  there are **48 camouflaged networks**.
- **Prompt.** The original post's prompt, with the network description rewritten for the two new shapes, and every
  weight printed to 3 decimals. No tools.
- **Models.** GPT-6 Astra (Codex CLI) and Claude Opus 5.5 (Claude Code), both at effort max, one reply per network.
- **Scoring.** Keywords on the reply's `Game:` line ("brick" counts as Breakout, "frogger" as Freeway). If the line
  names two games, the first counts.

## Results

These are the numbers reported in the post. The model replies behind them are not included in this repository;
running the readers below on the stored prompts produces new ones.

| | Astra (max) | Opus (max) |
|---|---|---|
| 48 camouflaged networks | **43 / 48** | **48 / 48** |
| 12 untouched policies | 12 / 12 | 9 / 12 (misses Breakout in all three shapes) |

- The two models differ on 5 networks from 3 policies, all in Opus's favour (exact McNemar p = 0.0625, sign test
  over policies p = 0.25). Astra calls all five Pong.
- At the original post's lower setting (effort high), Astra recognized 26 of the 48, and 27 in a rerun alongside
  the max run. The max run was pre-registered (`MATCHED_EFFORT_PREREG.md`).
- Median reply length: Astra about 15,800 output tokens, Opus about 123,000.
- Caveats: the four copies of a policy are not independent. The two tanh Breakout policies play barely better than
  random. After camouflage, Pac-Man's maze stays partly legible in other first-layer statistics.

## Files

| What | Where |
|---|---|
| Games and training | [`train/games.py`](../../studies/layout-vs-behaviour/experiments/train/games.py), [`train/train_game.py`](../../studies/layout-vs-behaviour/experiments/train/train_game.py); trained policies in `train/g_<game>[_relu24x3\|_tanh48x1]/actor.npz` |
| Architectures, prompt text, weight printing | [`arch.py`](../../studies/layout-vs-behaviour/experiments/arch.py), [`weights_text.py`](../../studies/layout-vs-behaviour/experiments/weights_text.py) |
| Exact camouflage | [`camouflage_arch.py`](../../studies/layout-vs-behaviour/experiments/camouflage_arch.py); exactness in play: [`eval_play.py`](../../studies/layout-vs-behaviour/experiments/eval_play.py), [`exactness_audit.json`](../../studies/layout-vs-behaviour/experiments/exactness_audit.json) |
| Blinded suites (networks, prompts, answer key) | [`suite_v2/`](../../studies/layout-vs-behaviour/experiments/suite_v2) (32-32 tanh), [`suite_relu24x3/`](../../studies/layout-vs-behaviour/experiments/suite_relu24x3) + [`suite_relu24x3_bfix/`](../../studies/layout-vs-behaviour/experiments/suite_relu24x3_bfix) (ReLU; Breakout rebuilt with unused inputs from the whole training record), [`suite_tanh48x1/`](../../studies/layout-vs-behaviour/experiments/suite_tanh48x1); each has `nets/<id>/`, `prompts/<id>.txt`, `key.json`, `PREREG.md` |
| Running the readers | [`run_astra_max.sh`](../../studies/layout-vs-behaviour/experiments/run_astra_max.sh) (Astra, max), [`run_claude.sh`](../../studies/layout-vs-behaviour/experiments/run_claude.sh) (Opus, max), or [`job_arch.sh`](../../studies/layout-vs-behaviour/experiments/job_arch.sh) for one network with retries and the output-cap rerun rule |
| Scoring | [`score_suite.py`](../../studies/layout-vs-behaviour/experiments/score_suite.py), [`arch_stats.py`](../../studies/layout-vs-behaviour/experiments/arch_stats.py), [`score_matched_effort.py`](../../studies/layout-vs-behaviour/experiments/score_matched_effort.py) |
| Pre-registration of the max run | [`MATCHED_EFFORT_PREREG.md`](../../studies/layout-vs-behaviour/experiments/MATCHED_EFFORT_PREREG.md) |
| Figure 1 (live policy) | [`post-figures/make_play.py`](../../post-figures/make_play.py), [`make_live_play.py`](../../post-figures/make_live_play.py) |
| Figure 2 (board pictures of the released networks) | [`post-figures/make_pictures.py`](../../post-figures/make_pictures.py); needs the upstream networks: [`studies/layout-vs-behaviour/external/fetch_upstream.sh`](../../studies/layout-vs-behaviour/external/fetch_upstream.sh) |
