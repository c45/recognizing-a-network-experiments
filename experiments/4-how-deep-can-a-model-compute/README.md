# Experiment 4: How deep a network a model can follow

Post sections: *How much can a model compute in its head?*, *Astra starts at the input layer and gets further*,
*With a calculator*. Figures: **DepthCurve** (figure 5), **DepthReplies** (figure 6), and the two paddle rows of
**Scoreboard**.

## Question

Experiment 3 shows a model can get a network's behaviour right when pushed. How deep a network can it follow, from
input to output, with no tools? And how much does a calculator help?

## Setup

- **Task.** Breakout on a 10 × 10 grid: 500 binary inputs, 5 action logits. Two opposite rules: move the paddle
  toward the ball when it travels right and away when it travels left, or the reverse.
- **Networks.** tanh MLP 500 → 32 × D → 5, D = 2 to 20 hidden layers, trained by imitation of one rule on a shared,
  mirror-balanced set of 261,000 states (cross-entropy, Adam, learning rate 0.001, batch 512, 10 epochs; 30 for most
  networks at depth 16 and all at 20), kept at held-out accuracy of 0.95 or more. Both rules train on the same
  boards, so the first layer has the same footprint for both and recognition cannot help.
- **Test sets.** 12 networks per set, 6 per rule, balanced on the sign of the right-minus-left output bias, random
  ids. Answer keys were hashed before any model call. Depths 12, 14 and 16 have four sets each (48 networks).
- **Prompt.** Both rules, the architecture and every weight to 3 decimals; the answer is right-toward or right-away
  with a confidence.
- **Models.** Astra and Opus at effort xhigh, no tools, one reply each. No answer counts as wrong.
- **With Python.** Weights reachable only through code, up to 100 runs (numpy, scipy, PyTorch and others). Any
  linear algebra, but no running the network or any layer on any input. 12 networks at each of 4, 8, 12, 16, 20 layers.
- **Curve.** p(D) = 0.5 + 0.5 / (1 + e^(b(D − D50))), maximum likelihood over a model's replies. D50 is the depth
  where accuracy crosses 75%, halfway between chance and perfect.

## Results

These are the numbers reported in the post. The model replies behind them are not included in this repository;
running the readers below on the stored prompts produces new ones.

| | Astra | Opus |
|---|---|---|
| No tools: D50 (95% CI), depths 2–16 | **11.1** layers (9.8–12.8) | **6.1** layers (5.5–6.8) |
| No tools: the 374 networks both were given | 303 right | 227 right (McNemar p = 2 × 10⁻¹⁰) |
| No tools: depth 8 (48 networks) | 45 | 30 |
| No tools: depths 12–16 (144 networks) | 90 | 68 (a coin flip) |
| With Python, 4 to 20 layers (60 networks) | 53 | 53 |

- A fixed weight-only script (curvature projection) gets 55 of the same 60; a three-path script gets 27 of 48 at
  depth 8.
- Prompt size at D50: about 115,000 tokens for Astra and 100,000 for Opus (each model's own tokenizer).
- **How the replies work** (figure 6; 120 no-tools replies at 4, 6, 8, 10 and 12 layers, coded by a separate model
  that did not see the key or scores). Astra starts at the input with a concrete board in 46 of its 60 replies.
  Opus more often starts at the output and pushes the right-minus-left score back through every matrix; where it
  finishes, it is right 21 times in 23, but from 8 layers on it tries in only 4 of 35 replies.
- Coverage: Opus has every planned no-tools reply (396). Astra has 374 of 396, because 22 calls failed on a
  connection error and were not rerun; the paired comparison uses the 374 networks both models have. Two Opus
  with-Python conversations reported results from analyses they never ran (see the figure note).

## Files

Paths are in [`studies/depth-benchmark/`](../../studies/depth-benchmark). Each reader set holds `prompts/<id>.txt` (exactly what
the model saw), `key.json` and `PREREG_KEY_HASH.txt`.

| What | Where |
|---|---|
| States and training | [`ladder/l2/make_states_l2c.py`](../../studies/depth-benchmark/ladder/l2/make_states_l2c.py), [`ladder/l2/train_l2.py`](../../studies/depth-benchmark/ladder/l2/train_l2.py), [`ladder/scale/scale.py`](../../studies/depth-benchmark/ladder/scale/scale.py), [`ladder/scale/train_variants.py`](../../studies/depth-benchmark/ladder/scale/train_variants.py); trained networks are not included; the scripts rebuild them ([`DATA.md`](../../studies/depth-benchmark/DATA.md)) |
| No-tools reader sets | depth 2: [`ladder/l2/readers_d/`](../../studies/depth-benchmark/ladder/l2/readers_d); depths 3–16: [`ladder/scale/readers_d<D>[b\|c\|d\|e]/`](../../studies/depth-benchmark/ladder/scale); run with each set's `run_astra.sh` and `run_claude.sh` |
| With Python | [`analyst/`](../../studies/depth-benchmark/analyst): `astra_loop.py`, per-depth `key_d<D>.json` and `message_d<D>.txt`; weights as arrays in `models/<id>/blocks.npz` |
| Weight-only scripts | [`ladder/scale/curvature_recipe.py`](../../studies/depth-benchmark/ladder/scale/curvature_recipe.py), [`ladder/scale/greedy_paths.py`](../../studies/depth-benchmark/ladder/scale/greedy_paths.py) |
| Analysis | [`threshold/analyze.py`](../../studies/depth-benchmark/threshold/analyze.py) (per-depth scores, D50 fits), [`paper_icml/review_analysis.py`](../../studies/depth-benchmark/paper_icml/review_analysis.py), [`paper_icml/make_numbers.py`](../../studies/depth-benchmark/paper_icml/make_numbers.py) |
| Pre-registration (with amendments 9 and 10 for depths 14 and 16) | [`threshold/PREREG.md`](../../studies/depth-benchmark/threshold/PREREG.md), [`ladder/PREREG.md`](../../studies/depth-benchmark/ladder/PREREG.md) |
| Prompt sizes | [`post-figures/make_prompt_tokens.py`](../../post-figures/make_prompt_tokens.py) (reads the readers' raw event streams) |
