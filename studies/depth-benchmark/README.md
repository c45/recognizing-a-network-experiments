# Reading deep neural networks from their weights

Code, prompts and answer keys for the depth benchmark: footprint-matched networks whose rule a model must read
from the weights, at increasing depth, plus the transformer families.

**Question.** Can a frontier language model tell what a trained network computes by reading its raw weights, once the
training-data *footprint* is removed? And at what depth does that reading break, with and without tools?

**Status:** reader collection for the pre-registered sets ended on November 7; tool-free depths 14 and 16 were added
on November 10 and both models were taken to 48 networks at depths 12, 14 and 16 (threshold/PREREG.md, Amendments 9
and 10). Opus has every planned tool-free reply (396); Astra has 374 of 396, because 22 calls failed on a connection
error and were not rerun.

> **In this repository** this folder backs experiments [4](../../experiments/4-how-deep-can-a-model-compute) and
> [5](../../experiments/5-transformer-rule-and-mirrors) of the post; start from those READMEs. The pilot family, the
> sign-flip and ReLU-rescale ladders, the width variants, the paper and all model replies are not included.

## Headline results

| | GPT-6 Astra | Claude Opus 5.5 |
|---|---|---|
| Weights as text, no tools: fitted 75%-accuracy depth D50 (95% CI), depths 2 to 16 | 11.1 layers (9.8–12.8) | 6.1 layers (5.5–6.8) |
| Weights as text, no tools: depth 12 / 14 / 16 (48 networks each) | 29/48 / 29/48 / 32/48 | 18/48 / 26/48 / 24/48 |
| Python sandbox, no forward pass | no break up to 20 layers (10/12 at depth 20) | no break up to 20 layers (11/12 at depth 20) |
| Same 60 with-Python networks | 53/60 | 53/60 |
| Fixed curvature-projection script, same 60 networks | 55/60 | |
| Transformers T2 / G2, no tools | 12/12 / 12/12 | 12/12 / 9/12 |
| Transformer R1b (120 rules): condition, offset pair, orientation | 13/16, 13/16, 6/10 (inconclusive) | 8/16, 1/16, n/a |

Reading without tools is beyond what a person can do unaided; with tools, the models are comparable to a fixed
weight-only script but do not beat it. The human comparison is reasoned, not measured. A two-token QK contrast
identifies every eligible T2 and G2 transformer, and an opposite-rule G2 pair that freezes those tensors identically
reaches 100% held-out accuracy in both members; reader evaluation on that pair is pending.

## Repository map

| Path | What it is |
|---|---|
| `paper_icml/` | Analysis scripts: `review_analysis.py` (matched comparisons, baselines) and `make_numbers.py` |
| `DESIGN.md`, `RESULTS.md` | Benchmark design and the full results log for the earlier stages |
| `ladder/l2/` | The interaction family, versions L2 → L2d (each fixing a shortcut a baseline caught) |
| `ladder/scale/` | Depth (2–24) scaling: training, baselines, blinded reader sets `readers_d*` |
| `transformer/` | Footprint-matched 2-layer transformer families (T1 confounded, T2 and G2 used) |
| `transformer/frozen_pair/` | Opposite-rule G2 pair with frozen E/Q2/K2, replayed trajectories, checkpoint manifest and verifier |
| `realistic/` | R1: rule grammar (120 rules) in a multi-behavior, rotated RMSNorm transformer (63k params) |
| `ladder/PREREG.md` | Pre-registration for the ladder families |
| `analyst/` | With-Python arm: conversation loop (`astra_loop.py`), sandbox, per-depth keys and messages, weights as arrays |
| `threshold/` | Breaking-depth study: `PREREG.md` (with amendments), `analyze.py`, weight-only scripts, figures |
| `vendor/` | Our Breakout simulator and weight-to-text formatter, used by the pilot scripts |
| `scripts/` | Environment setup, fetching third-party data, checkpoint checks |

Inside every reader set (`*/readers*/`): `prompts/<id>.txt` (exactly what the model saw), `key.json` (answers),
and `PREREG_KEY_HASH.txt` (SHA-256 of the key, recorded before any call). Model replies are not included.

## Setup

```bash
./scripts/setup_env.sh          # creates .venv from requirements.txt and links the analyst sandbox to it
./scripts/fetch_upstream.sh     # optional: released networks from the original post (footprint comparison)
```

Python 3.12. The analyst sandbox uses macOS `sandbox-exec`. Readers need the Codex CLI (GPT-6 Astra) and Claude
Code CLI (Opus 5.5); neither is needed to reproduce the analysis from the saved
replies.

## Analysis (after collecting replies)

Write replies to `replies/<reader>/<id>/` in each reader set (see below), then:

```bash
source .venv/bin/activate
cd threshold && python -I analyze.py . && cd ..          # per-depth scores, D50 fits
python -I paper_icml/review_analysis.py                  # matched comparisons, R1b baselines
python -I paper_icml/make_numbers.py
```

`twopager_analysis.py`, `review_analysis.py` and the baseline scripts also need the trained networks
(`ladder/**/nets*`, `transformer/nets_*`, `realistic/nets_r1b`), which the training scripts rebuild; see
[DATA.md](DATA.md).

## Running readers (model calls)

- Tool-free: each reader set's `run_astra.sh <prompt file> <out dir>` (GPT-6 Astra through `codex exec`) and
  `run_claude.sh <prompt file> <out dir> [model] [effort]` (Claude Opus 5.5 through `claude -p`), one reply per
  prompt, written to `replies/<reader>/<id>/`.
- With Python: `analyst/astra_loop.py <id> --reader astra|opus`.

## Working rules for collaborators

1. **Pre-registration first.** New families, depths or reader sets get a `PREREG.md` entry, including the key's
   SHA-256, *before* any model call. Deviations are appended as numbered amendments; earlier text is never edited.
2. **Keys are read-only.** Never edit `key.json` or a prompt after its hash is recorded. Make a new set instead.
3. **Infrastructure failures are not answers.** A reply that is a service-error message is set aside and rerun;
   `analyze.py` skips such texts.
4. **Numbers flow one way:** replies → `analyze.py` / `twopager_analysis.py` → `review_analysis.py` →
   `make_numbers.py`.

## License

Code is MIT-licensed (see [LICENSE](LICENSE)); data are licensed under CC BY 4.0 (see the repository's [LICENSE-DATA](../../LICENSE-DATA)).
