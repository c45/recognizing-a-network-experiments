# Realistic transformer family R1: pre-registration (November 6, 2026)

Written after four trial networks (`trial/`, one per condition) and before the network pool is trained, before any
baseline is computed, and before any model call. Code: `r1_family.py`.

## Question

Can GPT-6 Astra and Claude Opus 5.5 recover an unfamiliar rule from the raw weights of a transformer? The network has
the structural features that make real LLM interpretability hard:
- several behaviors share the weights;
- the vocabulary is larger than the model width (superposition);
- RMSNorm and multi-head attention;
- no privileged basis.

Everything needed to answer is in the prompt: the full forward pass, the encoding, the complete rule grammar, the
background behaviors and every weight.

## Networks

- **Data (shared by every network):** `data/seqs.npz`, 300,000 training and 30,000 held-out sequences of 16 tokens
  from a vocabulary of 64. There are three behaviors:
  - **RULE (60%):** positions 0–14 hold distinct tokens, and the query at position 15 repeats the token at
    position j (3–11). The output is x[j+a] if condition C holds, else x[j+b], with a ≠ b drawn from
    {−3, −2, −1, +1, +2, +3}. The conditions:
    - C1: the query is odd;
    - C2: the query is 32 or more;
    - C3: bit 1 of the query is set;
    - C4: the token at position 0 is odd.
  - **LOOKUP (20%):** 16 distinct tokens. The output is TABLE[x[15]], a fixed permutation (bits 2–4 permuted, bits 0,
    1 and 5 kept), the same for every network.
  - **MAJORITY (20%):** one token appears 5 times, and the output is that token.
- **Classes:** a network is defined by its rule (a, b, C), giving 30 ordered offset pairs × 4 conditions = 120
  classes.
- **Exact symmetry:** for each C, a bit flip σ_C (XOR 1 for C1 and C4, XOR 32 for C2, XOR 2 for C3) maps the data onto
  itself, commutes with TABLE, and maps rule (a, b, C) onto (b, a, C).
  - Initialization is i.i.d. across tokens. So (b, a, C) networks are distributed exactly as (a, b, C) networks with
    their embedding rows and unembedding columns permuted by σ_C.
  - **The orientation**, which offset goes with "C holds", is therefore invisible to every statistic invariant under
    that permutation.
- **Architecture:** pre-norm transformer, 3 layers, width 40, 4 heads of width 10, GELU MLP of width 160, RMSNorm, no
  biases, learned position embeddings. 63,360 parameters.
- **Export:** every RMSNorm gain is folded into the following weights, and the residual stream is rotated by a
  random orthogonal matrix (seed 10,000 + network seed). The exported network computes the same function: checked
  with an independent forward pass on held-out sequences (accuracy 0.9993–1.0 on the trials).
- **Training:** AdamW (lr 2e-3, weight decay 0.01, one-cycle), 8,000 steps of batch 512.
- **Inclusion:** held-out accuracy of 0.95 or more on each of the three behaviors.

## Network pool

- **Groups:** 20 (unordered offset pair, condition) groups, 5 per condition, with the pairs drawn by seed 20261108
  from the 15 unordered pairs.
- **Per group:** both orientations × seeds 0 and 1, so 80 networks.

## Baselines (leave-one-group-out logistic regression; features as in the T2 family, extended to this architecture)

- **Features:**
  - B1, token footprint: E row norms and U column norms;
  - B2, order-free statistics: sorted norms and every matrix's singular values;
  - B3, position footprint: P row norms;
  - B4, single-layer circuit summaries: direct-path and per-head OV diagonals, plus per-head positional offset scores.
- **Orientation** (the protected component; label: "the offset used when C holds is the larger one"):
  - **prediction:** B2, B3 and B4 at chance by the symmetry;
  - **B1** is not invariant under σ_C and may lean. If its interval excludes 0.5, the reader set is counterbalanced
    on B1's orientation decision;
  - **any other baseline outside chance:** R1's orientation result is confounded.
- **Condition (4-way) and offset pair (15-way), reported only, with no prediction:** these may be readable by
  supervised classifiers. The readers get no labeled examples, so the baselines show what simple statistics carry,
  not what a zero-shot reader can use.

## Readers

- **Reader set:** 16 networks from 16 different groups (4 per condition), one network per group (seed drawn at
  random), with 8 networks in each orientation. Each network's rule is therefore a different class. Networks get
  fresh blinded IDs, and the key's SHA-256 is recorded below before any call.
- **Tool-free:** GPT-6 Astra and Claude Opus 5.5 at effort xhigh, one reply each. The answer gives a, b and C, plus a
  confidence for the whole rule.
- **Scoring:**
  - **Primary:** exact matches of (a, b, C) out of 16, against chance 1/120.
  - **Decision rule:** 3 or more exact is evidence of reading the full rule (one-sided binomial p < 0.001 at chance
    1/120); 1–2 exact is weak; 0 is none.
  - **Secondary:** condition accuracy (chance 1/4) and unordered offset-pair accuracy (chance 1/15), with exact
    binomial tests.
  - **Orientation:** accuracy among replies with the correct pair and condition (chance 1/2).
- Infrastructure failures (usage limit, authentication, network) are rerun and never scored.

## Amendment 1 (November 6, before any model call): trained-pair orientation confounded; R1b uses exact mirrors

**Pool:** 80 trained, 78 included (excluded: `r_-1_-2_C4_s1` and `r_-3_-2_C4_s1`, below 0.95 on at least one
behavior). Baselines (`baselines_r1.json`, leave-one-group-out):

| Target | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| Orientation (chance 0.5) | 51/78 [0.54, 0.76] | **50/78 [0.52, 0.75]** | 29/78 [0.27, 0.49] | 36/78 [0.35, 0.58] |
| Condition (chance 0.25) | 22/78 | 29/78 | 26/78 | 17/78 |
| Offset pair (chance 0.067) | 16/78 | 30/78 | 25/78 | 31/78 |

**Ruling:** B2's interval excludes 0.5, so by the confound rule the orientation of *independently trained*
networks is reported as confounded.
- **Variance caveat:** B3 lands below chance although positions are untouched by σ. So leave-one-group-out intervals
  on these closely related networks are probably too narrow.
- **Likely cause:** the symmetry holds only in distribution. The finite training set and the batch order are not
  exactly σ-closed.

**R1b (used for readers): exact mirrors.**
- **Construction:** for every included trained network n with rule (a, b, C), its mirror m(n) permutes the rows of
  E and the columns of U by σ_C, then applies a fresh random rotation (seed 20,000 + network seed).
- **Why it is exact:** m(n)(x) = σ_C(n(σ_C(x))). That computes rule (b, a, C) on RULE sequences, commutes with TABLE
  and keeps MAJORITY. Each mirror's behavior is checked with the independent forward pass, with inclusion at 0.95
  or more as before.
- **Consequence:** in the R1b pool (78 trained plus 78 mirrors), every statistic invariant under σ_C (B2, B3, B4) is
  identical between a network and its mirror. Orientation is therefore at exactly 50% for those baselines by
  construction.
- **B1** is index-ordered and is computed on the R1b pool. If it leans, the reader set is counterbalanced on B1's
  decision, as pre-registered.
- **Everything else is unchanged:** selection (16 groups, 4 per condition, 8 per orientation; each network drawn
  from that class's trained and mirror networks), prompts, readers and decision rules. The prompt describes the
  network as trained. That is accurate: a mirror is exactly the network mirrored training produces.

## R1b baselines and reader set (recorded before any model call)

- **Mirror check:** all 78 mirrors pass the independent forward-pass check (lowest behavior accuracy 0.992).
- **R1b pool** (`baselines_r1b.json`, 156 networks):
  - Orientation: B1 92/156 [0.51, 0.67]; B2, B3 and B4 each exactly 78/156, as expected by construction.
  - Condition: B1 87/156, B2 58/156, B3 48/156, B4 34/156 (chance 0.25).
  - Offset pair: B1 22/156, B2 60/156, B3 48/156, B4 62/156 (chance 0.067).
- **Interpretation:** supervised classifiers on simple statistics carry information about the condition (token norms)
  and about the offset pair (spectra, positional attention). Orientation is invisible to every σ-invariant statistic.
- **Counterbalancing:** B1 leans on orientation, so the reader set is counterbalanced on B1's decision
  (`b1_orientation.json`). Within each orientation, 4 networks have B1 right and 4 have it wrong, so a B1-only
  reader scores 8/16 on orientation.
- **Reader set:** 4 networks per condition, 8 per orientation; 7 mirrors and 9 trained networks.

`readers_r1b/key.json`: `636c82e0cfb122d1845cc0249056ba48f60890d6c8b81cab174dd5aa09501c13`

**Launch (Nov 6, night):** tool-free GPT-6 Astra and Claude Opus 5.5, effort xhigh, one reply per network.

## Interim results (Nov 7, 01:30; collection continuing)

- **Scorer fix:** replies often write offsets with a Unicode minus (−). The parser in `score_r1.py` now normalizes
  − and – to "-" before matching. This is a parsing fix only; no scoring rule changes.
- **Infrastructure:** all runs stopped when their controlling process ended (Nov 6, 23:19). They were relaunched as
  detached processes with the same scripts and settings.

| Reader | Replies | Exact (chance 1/120) | Condition (1/4) | Offset pair (1/15) | Orientation given pair and condition (1/2) |
|---|---|---|---|---|---|
| Astra, tool-free, xhigh | 15/16 | **5/15** (p < 1e-4) | 12/15 | 12/15 | 5/9 |
| Opus, tool-free, xhigh | 10/16 | 0/10 | 5/10 | 1/10 | – |

- **Astra meets the decision rule** (3 or more exact). Its exact hits come from recovering the offsets and condition
  (12/15 each, far above the supervised baselines: condition at most 87/156, offset pair at most 62/156) combined
  with orientation at chance (5/9). The symmetry-protected component is not read.
- **Opus:** one reply says an earlier attempt was "cut off by a safety classifier" and gives no answer. It is
  counted as unanswered.

## Amendment 2 (November 8, before any R1c call): paired-mirror test (R1c)

**Motivating observation (exploratory, from the R1b transcripts; recorded before any R1c call).**
- **Orientation by network type:** among the 10 Astra replies with the correct offset pair and condition,
  orientation is right on 5 of 6 trained networks but only 1 of 4 mirrors.
- **Why that matters:** a mirror shares every σ-invariant statistic with a trained network of the opposite
  orientation. So a reader that takes orientation from such a statistic would be right on trained networks and wrong
  on mirrors.
- **Stated confidence:** 14 of 16 Astra replies call themselves guesses (confidence 1–10%), yet 5 of those 14 are
  exact. Only 2 replies show an explicit derivation (one fully correct at 85%, one with orientation reversed at 45%).

**Design.**
- **Pairs:** 12 trained R1 networks not used in R1b (3 per condition, 6 per orientation, `make_pairs_r1c.py`, seed
  20261110), each shown alongside its exact mirror.
- **Prompts:** 24 prompts under fresh blinded IDs, shuffled, identical in format to R1b.
- **Arms:**
  - tool-free GPT-6 Astra, effort xhigh;
  - GPT-6 Astra with Python (the analyst loop, effort xhigh, no forward pass: `analyst/message_r1c.txt`), on the
    same 24 networks with the same IDs.

**Primary measure (each arm).**
- **Structure-consistent pairs:** pairs where both members are answered with the same unordered offset pair and the
  same condition.
- **Classification:** each such pair is *opposite* (the two orientations differ, as circuit reading predicts) or
  *same* (they agree, as following σ-invariant cues predicts).
- **Decision rule:** at least 75% opposite is evidence of reading direction; at least 75% same is evidence of
  cue-following; anything else is mixed. Fewer than 4 structure-consistent pairs is uninformative.

**Secondary measures.**
- Orientation accuracy on trained networks versus mirrors, among replies with the correct pair and condition.
- Exact, condition and pair accuracy, as in R1b.
- **Agreement with the σ-invariant cue:** whether the answered orientation agrees with the trained-pair B2
  classifier's leave-one-group-out prediction for that network's source (mirrors inherit their source's
  prediction), relative to agreement with the truth.

**Opus is not run** on R1c. Tool-free Opus read nothing in R1b. A decision on Opus
with Python follows the Astra results.

`readers_r1c` key.json + pairs.json hash: `1467d23964140767ac515eb653fa89767ab3468588242deafab3c8226d3b66e0`

## R1c results (November 8, 2026)

All 24 networks answered in both arms. Scorer: `score_r1c.py`; output: `readers_r1c/score_r1c.json`.

| Arm | Exact (chance 1/120) | Condition | Offset pair | Orientation: trained / mirror | Structure-consistent pairs: opposite / same | Orientation agrees with B2 cue / with truth |
|---|---|---|---|---|---|---|
| Astra, tool-free | 6/24 | 14/24 | 13/24 | 4/5 / 2/3 | 3 / 1 | 11/24 / 15/24 |
| Astra + Python | **19/24** | 21/24 | 22/24 | **9/9 / 10/10** | **11 / 0** | 12/24 / 22/24 |

- **With Python, Astra reads orientation.**
  - All 11 structure-consistent pairs get opposite orientations, which meets the pre-registered rule for reading
    direction.
  - Orientation is right on every network with the correct pair and condition, trained and mirror alike.
  - The B2 cue predicts its answers at chance (12/24).
- **Without tools, the result is uninformative.** There are exactly 4 structure-consistent pairs (3 opposite, 1 same),
  which is the minimum; it meets the 75%-opposite rule only formally and is too few to rely on.
  - The R1b trained/mirror split does not replicate (mirrors 2/3 here). Answered orientations agree with the B2 cue
    no more than chance (11/24).
  - The σ-invariant-cue hypothesis from the R1b transcripts is not supported.
- **Error types with Python:**
  - all 3 condition misses read C4 (first-token parity) as C1 (query parity), with the parity-to-offset mapping
    itself correct;
  - both offset misses are one network and its mirror, each reading +2 as +1 with orientation consistent.
- **Confidence with Python:** 96–99% when right, 92–97% when wrong; median 5 code runs per network (range 3–11).
- **Forward-pass audit:** every code run (135) was swept for nonlinearities, normalization of constructed inputs,
  attention probabilities and sequence construction. The 8 pattern hits and 171 flagged lines were all read by hand.
  All are linear algebra on weights (bilinear forms, correlations, argmax over score matrices, eigen and pseudo-inverse
  operations). There were no forward passes.

## Rules tightened for later with-Python runs (November 10, 2026)

One with-Python conversation on the R1c prompts, from a reader outside this write-up, wrote the task's data
generator and architecture from the prompt, trained reference networks on rules it chose and compared them with the
given weights. That was ruled a violation (scored as unanswered), and the rules were tightened for any later run:
- `analyst/message_r1c_v2.txt` adds: no building, training, fitting or running of any other neural network, no
  generated task sequences, no files outside the working directory. `message_r1c.txt` is unchanged for the record.
- The sandbox in `analyst/astra_loop.py` now allows writes only in the run's working directory and denies reads
  of `/private/tmp` and the system temp folder. The earlier profile let code read and write `/private/tmp`, which
  persisted between runs and was shared between conversations.
- Conversations may be resumed: `analyst/astra_loop.py` continues an interrupted conversation from its saved
  transcript (`--resume-run`); no turn that already had a reply is re-asked.

## Amendment 5 (November 10, 2026, before any Opus call on an R1c prompt): Claude Opus 5.5 on the paired-mirror test

Claude Opus 5.5 is added to R1c in both arms (xhigh against xhigh), so that the mirror
test has the same two readers as R1b. Amendment 2 had left Opus out pending the Astra results.
- **Tool-free arm (label `opus_xhigh`).** The 24 R1c prompts, unchanged, one reply each, through the R1b Opus call
  (`pilot_f1/readers/run_claude.sh`: `claude -p`, effort xhigh, tools off) on the pinned Claude Code 2.1.314, the
  version every R1b Opus reply used. Replies in `readers_r1c/replies/opus_xhigh/`.
- **With-Python arm (run names `opusx_r1c_<id>`).** The analyst loop (`analyst/astra_loop.py --reader opus`, effort
  xhigh, 100-run limit) on the same 24 networks and IDs, Claude Code 2.1.314.
  It uses `analyst/message_r1c_v2.txt` and the tightened sandbox (see the November 10 section above); the Astra arm used
  `message_r1c.txt` and the earlier sandbox. That difference is reported with the results.
- **Measures, rules and audit.** Unchanged from Amendment 2 (key and pairs hash above reused). A reply or
  conversation with no answer counts as unanswered. Every with-Python code run is audited for forward passes and for
  the practices named above (other networks, generated sequences).
