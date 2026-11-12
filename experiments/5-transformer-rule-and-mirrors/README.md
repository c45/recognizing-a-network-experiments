# Experiment 5: An unfamiliar rule in a scrambled transformer

Post section: *An unfamiliar rule in a scrambled transformer*. Figure: **MirrorPairs** (figure 7), and the two
transformer rows of **Scoreboard**.

## Question

Can a model work out a rule it has never seen from the weights of a larger network, a transformer whose internal
coordinates have been scrambled so that no single weight gives the rule away? And is a right answer really read
from the mechanism, or partly guessed?

## Setup

- **Task.** Sequences of 16 tokens, each a number from 0 to 63. The first 15 are all different; the last repeats
  one of positions 3 to 11. If the last token repeats the token at position j, the network outputs the token at
  j + a when a yes-or-no condition C holds, else at j + b. a and b differ, each one of ±1, ±2, ±3. Four conditions:
  C1 the last token is odd; C2 it is 32 or more; C3 bit 1 of it is set; C4 the first token is odd. That gives 120
  possible rules (chance of guessing the whole rule: 1 in 120). The same network also serves a fixed lookup table
  (20% of sequences) and the majority token (20%).
- **Network.** 3-layer pre-norm transformer, width 40, 4 heads, GELU MLPs, RMSNorm, no biases: 63,360 weights,
  about 260,000–290,000 prompt tokens. AdamW, 8,000 steps, kept at held-out accuracy of 0.95 or more on each
  behaviour.
- **Scrambling.** Before printing, normalization gains are folded into the next weights and the residual stream is
  rotated by a random orthogonal matrix. This spreads every meaning across many weights without changing what the
  network computes.
- **Mirrors.** A mirror is the same network with token-embedding rows and unembedding columns permuted by the
  condition's bit flip (token id XOR 1, 32 or 2), then rotated again. It computes the same rule with a and b swapped.
  To any statistic that ignores which token is which, a network and its mirror look identical.
- **Three tests:**

  | Test | Networks | Models | Tools |
  |---|---|---|---|
  | Warm-up (scoreboard only) | 24 small transformers on 10-token sequences, two possible rules; 12 attention-only (`t2`), 12 GPT-style (`g2`) | Astra, Opus, xhigh | none |
  | 16-network test (`r1b`) | 16 networks of this family: 4 per condition, 9 trained and 7 mirrors | Astra, Opus, xhigh | none |
  | Mirror test (`r1c`) | 12 trained networks (3 per condition, none used above) and their 12 mirrors, shuffled; the model is not told mirrors exist | Astra, Opus, xhigh | none, and separately Python: weights as arrays, up to 100 code runs, any linear algebra, no forward pass on any input |

  Every key was hashed before any model call.

## Results

These are the numbers reported in the post. The model replies behind them are not included in this repository;
running the readers below on the stored prompts produces new ones.

| | Astra | Opus |
|---|---|---|
| Warm-up, attention-only | 12 / 12 | 12 / 12 |
| Warm-up, GPT-style | 12 / 12 | 9 / 12 |
| 16-network test, whole rule, no tools | **6 / 16** | **0 / 16** |
| 16-network test, condition right / offset pair right | 13 / 16, 13 / 16 | 8 / 16, 1 / 16 |
| Mirror test, whole rule, no tools | 6 / 24 | 1 / 24 |
| Mirror test, whole rule, with Python | **19 / 24** | **18 / 24** |
| Mirror pairs given opposite directions, with Python | **11 of the 11 comparable pairs** | **9 of the 9 comparable pairs** |
| Mirror pairs given opposite directions, no tools | 3 of the 4 comparable pairs | 0 of the 5 comparable pairs (same answer for both) |

- A pair is comparable when both answers name the same condition and the same two offsets. Only a model that follows
  the mechanism can give a network and its mirror opposite directions, since the two look the same to anything that
  ignores token identity.
- A simple statistic of the weights classifies every warm-up network, so the warm-up does not need reading across
  layers.
- One Opus reply in the 16-network test was blocked by a safety filter and counts as wrong. All 135 of Astra's Python
  code runs in the mirror test were audited for forward passes.
- Opus with Python: three conversations reported code runs that never happened (one turn, nothing executed). Each was
  rerun once; one did it again and counts as wrong. Every flagged Opus code block was read by hand: none runs the given
  network or trains a reference network. Five compute attention patterns from weight rows (mean embeddings and
  positions, no token sequences); without those five, Opus gets 14 of 19. Opus's answers and totals are in
  [`post-figures/opus_r1c.json`](../../post-figures/opus_r1c.json).

## Files

Paths are in [`studies/depth-benchmark/`](../../studies/depth-benchmark).

| What | Where |
|---|---|
| Warm-up families | [`transformer/tf_family.py`](../../studies/depth-benchmark/transformer/tf_family.py) (attention-only), [`transformer/gpt_family.py`](../../studies/depth-benchmark/transformer/gpt_family.py) (GPT-style), prompts [`make_prompts_tf.py`](../../studies/depth-benchmark/transformer/make_prompts_tf.py), [`make_prompts_gpt.py`](../../studies/depth-benchmark/transformer/make_prompts_gpt.py); reader sets [`transformer/readers_t2/`](../../studies/depth-benchmark/transformer/readers_t2), [`transformer/readers_g2/`](../../studies/depth-benchmark/transformer/readers_g2); scoring [`score_tf.py`](../../studies/depth-benchmark/transformer/score_tf.py); [`PREREG.md`](../../studies/depth-benchmark/transformer/PREREG.md) |
| Rule family and mirrors | [`realistic/r1_family.py`](../../studies/depth-benchmark/realistic/r1_family.py), [`realistic/mirror_r1.py`](../../studies/depth-benchmark/realistic/mirror_r1.py), [`realistic/make_prompts_r1.py`](../../studies/depth-benchmark/realistic/make_prompts_r1.py), [`realistic/make_pairs_r1c.py`](../../studies/depth-benchmark/realistic/make_pairs_r1c.py) |
| 16-network test | [`realistic/readers_r1b/`](../../studies/depth-benchmark/realistic/readers_r1b) (prompts, key), scored by [`realistic/score_r1.py`](../../studies/depth-benchmark/realistic/score_r1.py) |
| Mirror test | [`realistic/readers_r1c/`](../../studies/depth-benchmark/realistic/readers_r1c) (prompts, `key.json`, `pairs.json`), scored by [`realistic/score_r1c.py`](../../studies/depth-benchmark/realistic/score_r1c.py); with Python: [`analyst/astra_loop.py`](../../studies/depth-benchmark/analyst/astra_loop.py) with the task message [`analyst/message_r1c.txt`](../../studies/depth-benchmark/analyst/message_r1c.txt); forward-pass audit [`realistic/audit_r1c_forward.py`](../../studies/depth-benchmark/realistic/audit_r1c_forward.py) |
| Pre-registration | [`realistic/PREREG.md`](../../studies/depth-benchmark/realistic/PREREG.md) |
| Figure data | `transformers`, `r1b` and `r1c` blocks in [`post-figures/make_website_data.py`](../../post-figures/make_website_data.py) |
