# Reading vs recognition: pre-registration (November 5, 2026)

Written before any model call in this folder. Later stages are added below, each before its own calls. The
answers must not change these rules. Deviations go in a dated "Deviations" section.

## Question

When a model names the game from a policy's printed weights, is it reading the network's function, or recognising a
picture (the first-layer footprint) plus labels and matching them to games it already knows?

## Readers (all stages)

- **GPT-6 Astra**, effort high, via `run_astra.sh`: Codex exec, every tool disabled, read-only sandbox.
- **Claude Opus 5.5**, effort max, via `run_claude.sh`: `claude -p` with no tools, MCP, memory or CLAUDE.md.
- Both are tool-free with one reply, using the system text in `instructions.md`.
- One reply per prompt and reader. A run that returns an empty answer is retried once; no other reruns.

## Stage 1: prompt and layout changes on the four released networks (`suite_released/`)

**Networks:** the post's four Exp 1 networks (Pong dfc6cb65, Breakout a9f535cc, Pac-Man bdf99279, Freeway 0bbcc3ea).

**Conditions** (7, so 28 prompts per reader; built by `build_prompts.py released`):

| Condition | What changes | What it tests |
|---|---|---|
| base | nothing; byte-identical to the released prompt (checked) | baseline for these readers today |
| unnamed | channel and action meanings removed; grid geometry kept | does naming depend on the labels plus world knowledge? |
| chswap | channel 0 (agent) and channel 2 (reward targets) descriptions exchanged | do readers follow labels or the weights? |
| actswap | action labels: 1 = left, 2 = right, 3 = up, 4 = down | do the buttons or the picture decide the movement axis? |
| shuffled | W1 lines in random order, each prefixed "i:" | do readers rely on counting line positions? |
| transposed | exactly equivalent policy on the transposed grid (cell (r, c) becomes (c, r); up/left and down/right relabelled) | same rules, different footprint |
| permuted | exactly equivalent policy with the 64 interior cells relabelled by a fixed random permutation (border fixed) | same rules, no spatial picture |

- Equivalence of transposed and permuted was checked on 2,000 random binary observations: max |logit change| 0.0
  (`equivalence_checks.json`).
- In chswap, channel 4's text still says "cells that were in channel 0 or channel 1"; the labels are deliberately
  wrong.

**Primary score:** the game named on the "Game:" line, mapped by the keyword rule of `obscured/score_suite.py`
(breakout|arkanoid|brick, pac-man|maze, freeway|frogger|road|traffic|crossing|cross, pong|paddle; first match in that
order; no match = other). The correct answer is always the true game, including under transposed and permuted.

**Secondary, hand-coded blind to condition:**
- For actswap and transposed: the agent's movement axis the answer states (vertical / horizontal / none).
- For chswap: which channel the answer treats as the player.

**Planned comparisons:**
- Accuracy per condition, per reader and pooled (n = 4 per reader, 8 pooled), against base.
- Pre-stated readings:
  - permuted close to base means naming survives without a spatial picture;
  - permuted near zero while base is high means naming relies on the picture;
  - unnamed far below base means naming relies on labels plus world knowledge;
  - in actswap, an answer whose stated movement axis follows the labels rather than the layout counts as
    "label-following".
- With n = 4 per cell these are descriptive. The only test is an exact sign test of base vs each condition, pooled
  over readers (paired by network and reader). It is reported, not used as a gate.

**Hashes:** key.json sha256 `7b433be01e221ba9670d85ba8eaff49c8d541c1f1ebabb11cacaf51c7388fddc`; concatenated
prompts sha256 `4875e2f672974a4d064e6555fa1bae4df6904d2cd8718436bf7a6a0dfa030f18`. Claude Code 2.1.314; Codex
release local-722d08b4.

## Stages 2–3 (added before their calls)

- **Matched pair:** catch vs dodge.
- **Unnamed games:** shepherd, escort, lamplighter.
- **Footprint fade:** training variants.
- Their games and training are defined in `games_new.py` and `train_variant.py`, and are running now.

## Deviations

- **Nov 5, stage 1:** the first runner's job limit did not work under zsh, so all 56 stage 1 replies ran
  concurrently instead of 2 Astra and 3 Opus at a time. This changes no prompt, setting or scoring. Replies that come
  back empty because of rate limits or capacity are rerun, as the one-retry rule allows. Later stages use the
  `xargs -P` runner.

## Stage 1b: no picture, no labels (`suite_scrambled/`), added Nov 5 before its calls

**Why:** in stage 1, Astra named all four released networks under every single change, including permuted and
unnamed. Its permuted answers cite non-spatial cues: "vertical control" (button grouping), "a fixed right-side
scoring boundary" (the border, which permuted leaves fixed) and "a pronounced upward preference" (output bias). This
stage removes all of them at once.

**Condition `scrambled`:** each of the four released networks is rewritten with all 100 cells relabelled by a fixed
random permutation (border included) and checked exactly equivalent (max |logit change| 0.0).
- The prompt removes channel and action meanings, as in unnamed.
- It replaces the grid geometry sentence with a true statement that the indexing is scrambled and does not reflect
  position, neighbours or walls.
- What remains: how many inputs of each channel are used, which (unlabelled) actions behave alike, the output biases,
  and how channels interact through the hidden layers.

**Score:** the same keyword rule; correct = the true game. 4 prompts × 2 readers.

**Pre-stated reading:**
- Near zero means naming needs either a spatial picture or labels.
- Accuracy well above zero would be evidence of identification from structure alone. We will then read the answers
  to see which structure they cite.

**Hashes:** key.json sha256 `2e05a5f05f38aa75deb436081e49db7531069bec913202628ecfbbe5397b7906`; concatenated prompts
sha256 `65e349c40d37829111e4b0a0b179b86cb1e9b39f77a5df5a412cc5025342adec`.
- **Nov 5, stage 1:** three Opus replies on permuted prompts (Breakout, Pong, Freeway) returned no answer. Each ended
  with "API Error: Claude's response exceeded the 128000 output token maximum". They are kept in
  `suite_released/invalid/` and rerun once, by the same rule as empty answers. Both the first-attempt tally (1/4, with
  3 invalid) and the rerun tally are reported.

## Stage 5: three new obfuscations (`suite_stage5/`), added Nov 5 before its calls

Built by `build_stage5.py`. Sources are our trained tanh 32-32 policies for Pong, Breakout, Pac-Man and Freeway.
- **Unused inputs:** never on over the whole training run; none is on in a fresh rollout of the final policy.
- **Conditions:** one network per game for each of five conditions, 20 prompts in all.

| Condition | Change | Exact change on visited states (as printed) |
|---|---|---|
| plain | none (control) | 0 |
| camo_v2 | unused rows = exact copies of used rows; hidden shuffle and sign flips; output offsets (control) | ≤ 2e-14 (≤ 0.002) |
| synth | as camo_v2, but unused rows are new Gaussian rows fitted per channel to used rows, norm-matched (no copies) | ≤ 2e-14 (≤ 0.002) |
| pairs | camo_v2 plus 8 cancelling layer-2 unit pairs (width 32 → 48) that scramble the output layer's button similarity; prompt states the 32/48 widths | ≤ 3e-13 (≤ 0.002) |
| basis | plain network re-expressed for a stated input encoding (50 random groups of 10 inputs, each mixed by a listed 10×10 orthogonal Q to 3 decimals; W1_y = Q⁻¹W1) | ≤ 2e-14 (≤ 0.052, Pac-Man) |

**Arms:**
- Tool-free: all 20 prompts × 2 readers, settings as in stage 1.
- Analyst: pairs and basis networks (8) × 2 readers via `analyst_loop_r.py`.
  - The sandbox is the same as the camouflage study's: 30 code runs, numpy/scipy/sklearn/torch, no game environment.
  - The description is the same as tool-free, including the stated encoding; Q and groups are loaded as arrays.

**Score:** the keyword rule as in stage 1; correct = the true game.

**Planned comparisons** (paired by game and reader; descriptive, with pooled sign tests reported, not gated):
- **synth vs camo_v2:** does removing exact duplicates change accuracy?
  - Both readers flagged copies in nearly every camouflaged reply, and most then ignored the copied rows.
  - A drop means duplicates were helping readers discount fake rows.
- **pairs vs camo_v2:** a drop means the button-grouping cue was read from output-layer surface statistics.
  - The answers are also checked for any mention of paired or negated units, which would mean the cancellation was
    found.
- **basis tool-free vs plain tool-free:** does legibility depend on the input basis?
- **basis analyst vs basis tool-free:** can code undo it?

**No-LLM yardstick** (`detector.py`): a leave-one-game-out gradient-boosted detector of unused rows, using within-channel
row statistics. It is reported next to the best single hand statistic, for every condition.

**Hashes:** key.json sha256 `a65a0b5a9c5d2848326be2861cbd3fc04f2cd403629c5c5329e15d060e41d6a9`; concatenated prompts
sha256 `3ffb2decd379d334734e2c815f6df272cb622adbf55b8ace6b8dfa02ca6e6b3a`.
- **Nov 5, stage 1 outcome of the reruns:** all three permuted Opus replies failed again the same way, each after
  about 73 minutes and 512,000 output tokens. Under the one-retry rule they are scored as no answer, so Opus is 1/4 on
  permuted. The failures themselves are reported as a result: on the permuted prompt, which still describes a real
  grid geometry, Opus did not converge.

## Stages 2–3: matched pair and unnamed games (`suite_newgames/`), added Nov 5 before its calls

**Networks:** 10 policies trained by `train_variant.py` (tanh 32-32, 50M steps, the post's PPO settings): catch and dodge
(seeds 0, 1), and shepherd, escort and lamplighter (seeds 0, 1). Games are defined in `games_new.py`. Final training
returns:

| Game | Return (seed 0, seed 1) | Notes |
|---|---|---|
| catch | 24.97, 24.96 | max 25 |
| dodge | 24.92, 24.64 | max 25 |
| shepherd | 12.3, 12.4 | |
| escort | 19.4, 19.4 | max 20 |
| lamplighter | 49.7, 49.8 | |

**Matched footprint (`pair_check.py` → `runs/pair_check.json`):**
- catch and dodge have identical never-on input sets over training (328 each, Jaccard 1.00 for all 6 pairs).
- The row-norm maps correlate by seed, not by game: Spearman 0.98 for the same seed across games, 0.62 across seeds
  within a game.
- What differs is behaviour: catch agents end 17–21% of steps in the object's column, dodge 11–12% (chance 12.5%).
  Dodge output biases lean left or right; catch's are flat.

**Prompt:** the post's prompt, unchanged, around each network (`build_prompts.py nets`). For the unnamed games no
answer can be "correct" by name.

**Scoring:** `grade_rubric.py` (sha256 `154d5c2aba0dab103874a8feaf259bccbb2f96a839227260edf6d17bf9b41c84`).
- **Grader:** Claude Sonnet 5.5, effort low, no tools.
- **What the grader sees:** only the rubric for the true game and the answer's Game, Description and Reasoning
  text. It never sees the reader's identity.
- **Primary score, "gist":**
  - catch: says the agent seeks / catches the object;
  - dodge: says it avoids it;
  - shepherd: pushes a moving object, and reward comes when that object reaches a fixed region;
  - escort: reward for staying near a moving object;
  - lamplighter: scores by pressing up while lined up beneath a target.
- **Secondary:** the other rubric items.
- I hand-check every grader decision on the pair and report disagreements; the grader's call stays primary.

**Pre-stated readings:**
- **catch/dodge:** both games share the footprint, so correct seek/avoid calls must come from beyond the footprint
  (hidden layers, output biases).
  - Chance is 50% per network; n = 8 replies (4 networks × 2 readers).
  - The Nov 5 benchmark found 12/12 on an imitation-trained toward/away family. This is the RL-trained,
    open-question version.
- **unnamed games:** gist rates per game and reader, descriptive (n = 4 per game).

**Hashes:** key.json sha256 `c646f6f7970ffe422d5764c121408dd4eaa5e83b5b39afeb46436f3d4cc43a34`; concatenated prompts
sha256 `198f868c77d38c1b85ae67cad48423a1bfe791746dbc053aeacf9e1cab71a103`.

## Footprint fade results (no model calls)

These were measured before any stage that uses them.
- From-scratch Pong variants: `fade/scratch_pong.json`.
- Continued training: `fade/ft_pong.json`, `fade/ft_profU.json`.
- Exact camouflage: `fade/camo_pong.json`.

Stage 4, a reader test on the most obfuscated *training-based* variant, is **not run**. The only behaviour-preserving
training-based variant with reduced legibility is the unused-row profile match (`ft_pong_profU10`, best statistic
0.84). Stage 5's `synth` condition is its post-hoc analogue and is read by both models.
- **Nov 5, ~16:00:** Opus `claude -p` calls hit a usage limit.
  - Opus replies that are only the limit notice are not answers: 10 in suite_newgames and 19 in suite_stage5. They are
    kept in `<suite>/invalid_limit/` and rerun.
  - `one_reply.sh` now waits out a limit notice instead of saving it as an answer. The analyst loop already does.
  - No prompt, setting or scoring changes.
- **Nov 5, ~22:30:** three more Opus replies ended with the 128,000-output-token API error after ~1 hour and 512,000
  output tokens: dodge_s1 (stages 2–3), and Pac-Man basis and Pac-Man synth (stage 5). As in stage 1, each is moved
  to `<suite>/invalid/` and rerun once. A second failure is scored as no answer.

## Stage 6: disguised splits and depth inflation (`suite_stage6/`), added Nov 6 before its calls

Built by `build_stage6.py` (sha256 `43c56c8d5ea6f00d7d32ef728ea6b4ac92fb0009685d820732c68537e6e0231f`). Every base is the
synthetic-row camouflage (no duplicate rows).

| Condition | Source | Change | Exact on visited states | Printed vs printed base |
|---|---|---|---|---|
| split | tanh 32-32 | layer-1 twins (equal on used inputs, random on unused); layer-2 units split with random redistributions and output rows r+z / −z; width 64-64 | 0 | ≤ 0.029 |
| deep0 | ReLU 24-24-24 | synth only (3 hidden layers) | 0 | ≤ 0.090 |
| deep3 | ReLU | + 1 behaviour-exact ReLU layer per gap (6 hidden layers) | 0 | ≤ 0.099 |
| deep6 | ReLU | + 2 per gap (9 hidden layers) | 0 | ≤ 0.178 (Pac-Man), 0.166 (Breakout), ≤ 0.047 otherwise |

**Why these settings:** inserted layers are scaled permutations plus dense non-negative noise of up to 0.10. A setting
test on Pac-Man before any call showed printing error at deep6 of 0.21 with noise 0.25, 0.12 with 0.10, and 0.10 with
0.05. Scaling inserted layers up or down made it worse.

**Arms:**
- Tool-free: 16 prompts × Astra (high), Opus (max).
- Analyst: split and deep6 networks (8) × Astra and Opus.

**Score:** the keyword rule via `answer_parse.py`.

**Pre-stated readings:**
- **split vs stage 5 synth** (same tanh sources): a drop means readers needed visible exact relations (copies,
  negations), or an unscrambled button grouping.
- **deep0 → deep3 → deep6:** a monotone drop means tool-free reading is limited by depth on real policies (the Nov 5
  benchmark found a break at depth 8). Analysts on deep6 test whether code restores it.

**Hashes:** key.json sha256 `240411d79141608599b8031636d4eb7f6e8109e661e4c1428ff9d01c4af6845b`; concatenated prompts
sha256 `350a690d2b6e417148694a2826ab5197bba0d02f4456fdd1cedad2540bb94d97`.

## No-LLM game namer (`namer.py`), frozen Nov 8 before any test network was scored

- **Inputs:** weights only, with no cell positions and no labels.
- **Rule:**
  - freeway if the output-bias gap is above 1.5 or the action groups split 2/3;
  - else pacman if there are 4 or more action groups (centred cosine above 0.35, complete linkage);
  - else breakout if the reward channel (role inferred from counts) has 12 or more used inputs;
  - else pong.
- **Development set:** the 4 released networks and our 4 plain tanh 32-32 networks; 7/8 correct, with our weakly
  trained tanh Breakout called pong.
- **Test set:** every other network the readers saw, scored without further changes. BR networks are decoded with
  their stated Q, and CP's negated unit pairs are dropped, as an analyst would.
- **Hash:** namer.py sha256 `f3b4c9a9f6a9af6ed1d96244ef7be98f68e6da7b76a81fbdae0a147524f94927`.
