# Transformer family T1: NEXT vs PREV copy rule (pre-registration, November 6, 2026)

Written after two trial networks (seed 0, one per class, in `trial/`, both at held-out accuracy 1.0) and before the
family is trained, before any baseline is computed, and before any model call.

## Question

Can GPT-6 Astra and Claude Opus 5.5 read which copy rule a small attention-only transformer learned, from its raw
weights, when both rules are trained on identical inputs with identical token statistics?

## Networks

- **Data (one shared set for every network):** `data/seqs.npz`, 200,000 training and 20,000 held-out sequences.
  - Positions 0–8 hold 9 distinct tokens from a vocabulary of 16.
  - Position 9 (the query) repeats the token at one position j, with j uniform in 1–7.
- **Classes:**
  - **NEXT:** output the token right after the earlier occurrence, x[j+1];
  - **PREV:** output the token right before it, x[j−1].
- **Why the footprint is matched:** both classes see the same sequences. Every token is equally likely at every
  input position and as a label, and initialization is i.i.d. across tokens. So any statistic of the token
  embedding or unembedding has the same distribution in both classes.
- **Architecture:** attention-only, 2 layers, 1 head per layer, d_model 32, d_head 16, causal mask, no MLPs,
  no LayerNorm, no biases. Logits are read at position 9. 5,184 parameters.
- **Training:** Adam (lr 3e-3, cosine decay), 3,000 steps of batch 512; seed s sets init and batch order.
- **Family:** 40 seeds per class (seeds 0–39). Inclusion requires held-out accuracy of 0.95 or more.

## Baselines (leave-one-seed-out logistic regression; prediction: all at chance, 95% interval includes 0.5)

- **B1, token footprint:** row norms of E (16) and column norms of U (16).
- **B2, order-free statistics:** sorted row norms of E and P, plus the singular values of every matrix.
- **B3, position footprint:** row norms of P (10).
- **B4, single-layer circuit summaries (no composition):**
  - mean diagonal of the direct path E·U;
  - mean diagonal of each layer's OV path E·V_l·O_l·U;
  - each layer's positional attention preference: the mean score P·Q_l·(P·K_l)ᵀ at offsets 0, −1 and −2.

**Confound rule:** if any baseline's interval excludes 0.5, T1 is reported as confounded. No reader result is then
claimed as reading. The family is fixed (re-matched or counterbalanced, recorded here) before any reader call.

**Reference script (expected to succeed; reported, not a decision criterion):** layer-2 K-composition versus
direct token matching. Compare the mean diagonal of (E·Q2)(E·V1·O1·K2)ᵀ, which matches the query to the
previous-token information written by layer 1, against the mean diagonal of (E·Q2)(E·K2)ᵀ, which matches it to the
current token. NEXT predicts the first is larger.

## Readers

- **Reader set:** 12 included networks, 6 per class, drawn with a fixed seed under fresh blinded IDs. The key's
  SHA-256 is recorded below before any call.
- **Tool-free** (as in the breaking-depth study): GPT-6 Astra via `codex exec` and Claude Opus 5.5 via
  `claude -p --tools ""`, both at effort xhigh, one reply each. The prompt describes the sequences, the
  architecture and the formula, then gives every weight to 3 decimals.
- **Decision rule:** at least 10/12 is evidence of reading the rule; 8/12 or fewer is no evidence; 9 is
  inconclusive. Partial credit (mean probability on the correct class) is also reported.
- Infrastructure failures (usage limit, authentication, network) are rerun and never scored.

## Amendment 1 (November 6, before any model call): T1 is confounded; T2 replaces it

**T1 baselines** (`baselines_t1.json`, 80 networks, all at held-out accuracy 1.0):

| Baseline | Correct | 95% interval |
|---|---|---|
| B1 token footprint | 67/80 | [0.74, 0.91] |
| B2 order-free statistics | 78/80 | [0.91, 1.00] |
| B3 position footprint | 44/80 | [0.44, 0.66] |
| B4 single-layer circuits | 79/80 | [0.93, 1.00] |

The reference K-composition script scored 80/80. By the confound rule, T1 is not used for readers.
- **Why it is confounded:** NEXT and PREV use different mechanisms. NEXT matches the query to the previous-token
  information written by layer 1 (K-composition). PREV matches the current token and copies the previous-token
  information (V-composition).
- **Consequence:** the two mechanisms leave different spectra and different single-layer traces, so a classifier
  separates them without reading the circuit.

**T2: interaction rule matched by exact symmetry.** Same sequences, architecture and training. Two classes:
- **EVEN-NEXT:** NEXT when the query token is even, PREV when it is odd;
- **EVEN-PREV:** the reverse.

**The symmetry:** let σ swap each token with its partner, v ↔ v XOR 1. σ maps the shared sequence set onto itself,
and A(σ(s)) = σ(B(s)) for every sequence s.
- So EVEN-PREV networks are distributed exactly as EVEN-NEXT networks with the rows of E and the columns of U
  permuted by σ. Initialization is i.i.d. across tokens.
- Every network needs both mechanisms, routed by query parity.
- Any statistic invariant under that joint permutation has the same distribution in both classes: singular values,
  sorted norms, everything about P, and traces such as the B4 diagonals.

**T2 baselines:** the same B1–B4, plus the reference script (the T1 K-composition contrast, restricted to even
query tokens).
- **Predictions:** B2, B3 and B4 at chance by the symmetry. B1 (token norms in index order) is not invariant under
  σ and may lean.
- **If B1's interval excludes 0.5:** the reader set is counterbalanced on B1's leave-one-seed-out decision value
  (half of each class on each side), as with the L2d output-bias lean.
- **Any other baseline outside chance:** T2 is confounded.

**Family:** 40 seeds per class (EVEN-NEXT seeds 0–39, EVEN-PREV seeds 0–39), with inclusion at held-out accuracy
of 0.95 or more. Steps may be increased up to 12,000 if 3,000 do not reach the bar, the same for both classes,
and reported.

**Readers and decision rule:** as above. The question asks which rule applies to even query tokens. Answer
`even-next` or `even-prev`, with a confidence.

## T2 reader set key hash (recorded before any call)

`readers_t2/key.json`: `429114325d27533a0521b2e9cc2bf9172148d6b2232ad46172d012c1df51ad0a`

Baselines (`baselines_t2.json`, 79 included networks; evenprev s32 excluded at 0.9485): B1 47/79 [0.48, 0.70], B2 45/79 [0.45, 0.68], B3 39/79 [0.38, 0.61], B4 40/79 [0.39, 0.62]; all include 0.5, so no counterbalancing is applied. Reference K-composition script (even queries): 79/79.

## Amendment 2 (November 6, before any G2 baseline or model call): GPT-mini family G2

Written after four trial networks (`trial_g2/`, seeds 0–1 per class, all at held-out accuracy 1.0) and before the
T2 reader results are known. **G2 readers launch only if tool-free Astra or Opus reaches 10/12 or more on T2.**
Otherwise G2 is reported as built but not run.

- **Task and data:** T2 unchanged (EVEN-NEXT vs EVEN-PREV, `data/seqs.npz`).
- **Architecture (`gpt_family.py`):** standard pre-LayerNorm GPT blocks.
  - 2 layers, width 32, 4 attention heads of width 8, with biases.
  - MLP 32 → 128 → 32 with GELU; final LayerNorm.
  - Untied unembedding U (32 × 16) with bias c.
  - 26,832 parameters, about 170,000 characters as text: the size of the depth-10 MLP prompts.
- **Training:** AdamW (lr 3e-3, no weight decay, cosine decay), 3,000 steps of batch 512. 40 seeds per class;
  inclusion at held-out accuracy of 0.95 or more.
- **Symmetry:** tokens enter only through E and leave only through U and c, all initialized i.i.d. across tokens.
  So the T2 token-pair symmetry is exact: EVEN-PREV networks are distributed as EVEN-NEXT networks with E rows,
  U columns and c entries permuted by σ.
- **Baselines** (as for T2, extended to the new parameters; prediction B2–B4 at chance, with B1 handled as in
  Amendment 1):
  - B1: E row norms, U column norms and c;
  - B2: sorted norms, every matrix's singular values, and sorted biases;
  - B3: P row norms;
  - B4: the direct-path and per-head OV diagonals, plus per-head positional offset scores (LayerNorm ignored).
- **Reference script (reported only):** the T2 K-composition contrast summed over heads, with LayerNorm gains folded
  into the adjacent weights and normalization ignored.
- **Readers:** a blinded set of 12, tool-free Astra and Opus at effort xhigh. The prompt states every formula,
  including LayerNorm, GELU and the head split. Same decision rule.

## G2 baselines and reader set (recorded before any G2 call)

Baselines (`baselines_g2.json`, 80 networks, all at held-out accuracy 1.0): B1 53/80 [0.55, 0.76]; B2 49/80 [0.50, 0.72]; B3 41/80 [0.40, 0.63]; B4 43/80 [0.42, 0.65]. Reference K-composition script: 80/80. B1 leans, so as pre-registered (Amendment 1) the reader set is counterbalanced on B1's leave-one-seed-out decision (`b1_decisions_g2.json`): within each class, 3 networks on each side, so a B1-only reader scores 6/12. B2's interval touches 0.5 (lower bound 0.497) and is reported as a borderline caveat.

`readers_g2/key.json`: `9e62b3478e49086d94733533c958addb4173e5fdabb627a9e50d9fa0dc0c18ce`

## T2 tool-free Astra result and G2 launch (November 6)

Tool-free GPT-6 Astra (effort xhigh) on `readers_t2`: **12/12** (one-sided p = 0.0002, partial credit 0.937). This
meets the Amendment 2 gate, so G2 readers (Astra and Opus, tool-free, xhigh) were launched. Opus on T2 is still
pending (usage limits).

GPT-mini tool-free Astra (effort xhigh) on `readers_g2`: **12/12** (one-sided p = 0.0002, partial credit 0.822).

## Opus results (Nov 7, 01:30)

- **T2:** Opus 5.5 tool-free (xhigh) scored 11/11 valid replies.
- **G2:** 5/7 valid replies correct.
- **Network failures:** the remaining T2 reply and 5 G2 replies were 17-second network failures (empty answers,
  stderr: "Check your network connection"). These are infrastructure failures, not answers, and are being rerun.
  An earlier interim note counted them as unanswered; that was wrong.
- Astra: 12/12 on both.

## Amendment 3 (November 7): post hoc local cue and prospective frozen-tensor pilot

The two-token contrast (E[1]Q2).(E[1]K2) - (E[2]Q2).(E[2]K2) classifies 79/79 eligible T2 and 80/80 eligible
G2 networks. It also classifies 12/12 in each blinded set from the rounded prompt text. This post hoc analysis
shows that the single-layer baseline suite missed a token-specific QK cue. The earlier B1-B4 results concern
those particular features. Full-population classification accuracy supports rule recognition while an inference
of cross-layer reading requires additional controls.

The token-swap argument above applies to the generating distribution. The saved finite data has only 2/200,000
training rows with their XOR-1 partners present, and 0/20,000 held-out rows. Future claims of exact finite-data
closure require explicit pairing within each split. The original dataset and completed cohorts retain their
recorded provenance.

Draft revisions, checkpoint saving and a frozen-tensor pair were added on November 7. The new pilot's
prospective plan and source/data hashes are in frozen_pair/PREREG.md and frozen_pair/registration.json. Both G2
rules are trained from seed 1001 with E, layer-2 query weights and layer-2 key weights frozen. Intermediate
checkpoints are saved for that pair and four reruns of the original G2 recipe. Reader testing is pending.
Numerical outcomes and replay differences are recorded by frozen_pair/verify.py.
