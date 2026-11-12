# Footprint-matched weight-reading benchmark: design (November 5, 2026)

## Question

Can an LLM read **learned function** from raw weights, beyond footprints of the training data?

Earlier checks (`../weight-space-exp1-2-shortcuts-20261103/`: README, interp/, exp3/, obscured/; Exp 4 and Exp 8
audits) show that the post's reading demos are explained by data footprints:
- which inputs were ever on;
- which pixels vary;
- how many letters are used;
- how often the text repeats.

Exact camouflage cannot hide those footprints (`obscured/RESULTS.md`). So this benchmark **holds the footprint fixed by
construction and varies only the function**.

## Core design: same inputs, different function

Each family trains networks by supervised imitation on **one shared set of input states**. Classes differ only in the
labels. Which inputs are ever active, and how often, is then identical for every network. Any difference in the
weights comes from the function the network computes.

| Family | Shared inputs | Classes (function) | What a reader must recover |
|---|---|---|---|
| F1 (pilot) | Breakout-grid states (paddle row 8, ball rows 1-7) | move the paddle **toward** the ball vs **away** | the sign linking ball column and paddle column to left/right |
| F2 | Pong-grid states | intercept the ball vs dodge it | the same sign question, on a vertical paddle |
| F3 | Breakout-grid states | track the ball's column now vs where it will land (uses the previous frame) | whether the previous frame is used |
| F4 | character LMs, same corpus | a positive control: different corpus with a matched alphabet and repetition rate | language, with the known cues removed |

Positive control: the post's own Exp 1 family, where footprints differ. Readers are known to succeed there.

## Readers, baselines and arms

- **Tool-free readers** (the post's protocol: weights as text, one reply):
  - GPT-6 Astra (`codex exec`, effort high);
  - Claude Opus 5.5 (`claude -p`, effort max, no tools);
  - later, open models (Qwen 3.8, GLM-5.3) for full reasoning traces.
- **Analyst readers:** Python with weights preloaded, linear algebra allowed, no forward pass (the Exp 3 rule).
- **Baselines:**
  1. a footprint classifier: logistic regression on the 500 first-layer row norms;
  2. a weight-statistics classifier on features that don't depend on hidden-unit order (sorted row norms, singular
     values of each layer, output biases);
  3. a classifier given the full weights;
  4. chance.

  All baselines are cross-validated across seeds.
- **Humans:** a notepad-and-pencil protocol. Drafted here; participants are recruited separately.

## Statistics and blinding

- Every stage is pre-registered before its networks are generated or read.
- Accuracy is reported per reader with exact binomial tests and 95% intervals. The full benchmark adds a mixed-effects
  logistic regression (reader × family, with network as a random effect) and Brier scores.
- Blinding:
  - networks get random IDs;
  - the key's SHA-256 is recorded in the pre-registration before any reader call;
  - the key is not opened until scoring.

## Interpretation, fixed in advance

- **Readers above chance on matched families, beating baselines 1–2:** evidence of reading learned function. This is
  the ambitious positive result.
- **Readers at chance while the baselines are also at chance:** the demos show footprint reading only, and function
  reading is not demonstrated. The strong negative result.
- **Readers above chance but no better than the weight-statistics baseline:** they read simple statistics that happen
  to differ with function. Report which statistics.
