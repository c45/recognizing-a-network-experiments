# Difficulty ladder (pre-registration, November 5, 2026)

Context: pilot F1 (toward vs away) tool-free Astra scored 12/12 at 98–99% confidence, with all 71 weight values it
cited by index exact. Footprint baselines were at chance and a linear reading (W1·W2·W3) was 40/40. The answers
describe explicit sign tracing: input → hidden unit → second-layer unit → left/right output. The pilot's Opus replies
were still running when this was written.

The ladder asks where tool-free reading stops. Every rung keeps the footprint matched: one shared set of input
states, with classes differing only in labels.

## L1-flip: a function-preserving transform of the pilot networks

- **Networks:** the same 12 blinded pilot networks.
- **Transform:**
  - each first-layer hidden unit's sign is flipped with probability 0.5: its W1 column, its b1 entry and its W2 row
    are negated;
  - the same for second-layer units: the W2 column, b2 entry and W3 row;
  - then both hidden layers are randomly permuted.

  Because tanh is odd, the network computes exactly the same function. Checked: the maximum logit difference on the
  held-out states must be < 1e-4.
- **Blinding:** new random IDs, a new key, and its hash recorded before any call. Same prompt and readers as the
  pilot. Astra first; Opus only if Opus scores ≥ 10/12 on the pilot.
- **Prediction:** a reader that multiplies signs along paths scores as before (≥ 10/12). A drop to ≤ 8/12 means the
  pilot answers relied on sign conventions that happened to hold, not on path products.

## L2: interaction (direction-dependent rule)

- **Inputs:** a new shared set of 330,000 Breakout states from the same behaviour mix. It also records the ball's
  vertical direction d (+1 falling toward the paddle, −1 rising), and only includes states whose previous frame
  shows the ball, so direction can be read from channels 1 and 4. *Added before any L2 network was trained:* the
  raw states are 78% falling, which would let a direction-blind reading succeed, so states are subsampled to 50%
  falling.
- **Classes:**
  - **fall-toward:** toward when falling, away when rising;
  - **fall-away:** away when falling, toward when rising.

  "Toward" and "away" are as in F1, with no-op when the ball is above the paddle.
- **Networks:** 20 seeds per class, same architecture and training; inclusion requires held-out accuracy ≥ 0.95.
  More epochs (up to 30) may be used if accuracy at 10 epochs is below 0.95, the same for both classes, reported.
- **Baselines** (leave-one-seed-out): B1 row norms, B2 order-free statistics, B3 output biases, and **B4 the linear
  reading from the pilot**, which ignores direction. **Prediction: B4 at chance.** If B4 is above chance, L2 does
  not test nonlinear reading, and that is reported first.
- **Question to readers:** "When the ball is FALLING (moving down toward the paddle), does this policy move the paddle
  TOWARD the ball's column or AWAY from it? (When the ball is rising it does the opposite.)" Answer `fall-toward` or
  `fall-away`, with a confidence of 50–100%. The game description adds that the direction is visible from channel 4
  (the previous frame).
- **Readers:** 12 blinded networks, 6 per class. Tool-free Astra (effort high) and tool-free Opus (effort max),
  one reply each.
- **Decision rule:** ≥ 10/12 is evidence of reading a nonlinear interaction; ≤ 8/12 is no evidence; 9 is
  inconclusive.

## Analyst arm (after L2's tool-free results)

The same 12 L2 networks, with Python, linear algebra allowed and no forward pass (the Exp 3 harness). Astra effort
high. This separates "the information is recoverable by explicit analysis without running the network" from "a
tool-free reader can do it".

## L1-flip key hash (recorded before any call)

`f88cc9ee91a867d3bf45c520a32c24f277009baf406aa138e3c4f9e82f6389c2`

## L1-flip result (scored November 5, 2026)

Tool-free Astra scored 12/12 (98–99% confidence), the same as on the untransformed networks.

## L2 deviation and L2b (written after L2 baselines, before any L2 reader call)

On L2 as built (50% falling overall), **B4 scored 0/40**. The direction-blind linear reading separates the classes
perfectly, but with the sign inverted. The cause is that in Breakout the ball's row correlates with its direction,
so overall balance leaves a row-dependent cue. L2 therefore fails its own condition ("B4 at chance") and is not used
as the interaction test. Its networks are kept in `l2/nets`.

**L2b:** the same as L2, except the shared states are balanced to 50% falling *within each ball row*, by subsampling.
In this environment the ball is almost never above row 4, and rows 4 and 7 are nearly one-directional (row counts:
row 4 25,854 rising vs 290 falling; row 7 0 vs 127,218). So L2b keeps only states with the ball in rows 5 and 6,
each balanced 50/50. Everything else follows the L2 entry: classes, training, inclusion, baselines B1–B4 with their prediction of chance,
question, readers and decision rule. If B4 is again far from chance (95% interval excludes 0.5), L2b is reported as
confounded, and no reader result on it is claimed as evidence of interaction reading.

## L2b result and L2c (written after L2b baselines, before any L2 reader call)

**L2b baselines:** B1 22/40, B2 22/40, B3 18/40, **B4 39/40**. L2b is confounded by its own rule, so no reader
result on it is claimed as interaction reading. The paddle sits at the bottom, so falling and rising are not
symmetric. Networks implement the two classes differently, and their linearization reveals the class.

**L2c: matched by exact symmetry.** The rule depends on the ball's *horizontal* direction of motion (observed from
channels 1 and 4).
- **right-toward:** toward when the ball moves right, away when it moves left.
- **right-away:** the reverse.

The shared states are made mirror-symmetric by adding the left-right mirror of every state. Mirroring maps one
class's labels exactly onto the other's. Network inits are invariant in distribution under the mirror's input
permutation and the left/right output swap. So any weight statistic that is unchanged by mirroring has the same
distribution in both classes. That includes the B4 linear slope, B2, B3 up to the left/right swap, and the sign-chain
reading the pilot answers used.

- **Sizes:** 165,000 states plus their mirrors, 330,000 in total. Held-out states are split by pairs, so a state and
  its mirror are in the same split.
- **Networks:** 20 seeds per class, trained as in L2; inclusion ≥ 0.95.
- **Baselines:** B1–B4, predicted at chance (95% interval includes 0.5). If B4 or B2 is not at chance, L2c is reported
  as confounded.
- **Question:** "When the ball is moving RIGHT, does this policy move the paddle TOWARD the ball's column or AWAY from
  it? (When it moves left it does the opposite.)" Answer `right-toward` or `right-away`. The game description notes
  that direction is visible from channel 4.
- **Readers and decision rule:** as in L2 (12 blinded networks, 6 per class; tool-free Astra effort high and Opus
  effort max; ≥ 10/12 is evidence of reading the interaction, ≤ 8/12 is none).

## L2c result and L2d (written after L2c baselines, before any L2 reader call)

**L2c baselines:** B1 15/40, **B2 30/40**, **B3 37/40**, B4 22/40. Confounded by its own rule. A ball moving right is
more often to the right of the paddle (79,920 vs 43,105 states), so right-toward networks see about 160k "right"
labels against 86k "left", and the output biases record that action frequency.

**L2d:** L2c with the shared states also balanced over the four combinations of motion direction (right/left) and
ball side (left of the paddle / right of it), by subsampling before mirroring. States with the ball above the paddle
(no-op) are kept in the same proportion within each direction. Both classes then have equal left and right label
counts. Everything else follows L2c, including the rule that a baseline outside chance means confounded.

## L2d result and reader set (written after L2d baselines, before any L2 reader call)

**L2d baselines:** B1 25/40 [0.46, 0.77], B2 24/40 [0.43, 0.75], **B3 33/40 [0.67, 0.93]**, B4 24/40 [0.43, 0.75].
Label counts are now equal in both classes (left 122,100 = right 122,100). The output-bias cue is a small,
consistent left/right bias difference: mean right−left −0.071 for right-toward and +0.084 for right-away. 16/20 and
17/20 networks lean the expected way. Mirror symmetry makes the two classes' biases mirror images, but does not stop
this sign from tracking the class.

**Counterbalanced reader set:** 6 networks per class. In each class, 3 have b3[right] > b3[left] and 3 have
b3[right] ≤ b3[left], drawn with numpy default_rng(20261107) from the eligible networks. In this set the bias sign is
uncorrelated with class, so a bias-only reader scores exactly 6/12.

- **Primary outcome:** correct out of 12 per reader. The decision rule is as in L2: ≥ 10 is evidence of reading the
  interaction, ≤ 8 is none.
- **Secondary:** accuracy on bias-congruent vs bias-incongruent networks (6 each). Reading the interaction predicts
  similar accuracy on both; following the bias predicts congruent correct and incongruent wrong.

## L2d key hash (recorded before any call)

`833df1d52f150f6293b04a2545231f49976449576eb697443a8db27602412479`

## L2d tool-free Astra result (scored November 5, 2026)

12/12 correct (91–96% confidence). Bias-congruent 6/6 and incongruent 6/6. All 34 weight values cited by index are
exact. The answers describe a **hand-simulated forward pass on a constructed example**: the ball moving from one cell
to the next, with the paddle placed; dominant first-layer sums are pushed through W2 to the output weights.
("a dominant-term inference, not an exact numerical evaluation"). The prompt did not forbid mental simulation (the
post's Exp 1 did not either; its Exp 2 did).

## L2d-nosim: same networks, mental simulation forbidden (written before any call)

- **Networks:** the same 12 L2d networks with new IDs and a new key, its hash recorded before any call.
- **Prompt:** the L2d prompt plus the post's Exp 2 instruction, adapted: "Do not do any forward-pass calculation: do
  not run the network on any input (in your head or on paper), including partial or approximate sums for a specific
  ball and paddle position, and do not simulate play. Read what the weights themselves show."
- **Readers:** tool-free Astra (effort high); tool-free Opus (effort max) if Opus scores ≥ 10/12 on the pilot.
- **Compliance:** each reply is read by hand and judged non-compliant if it evaluates the network on a specific input
  (sums over a constructed example). Non-compliant replies are reported and not scored.
- **Decision rule:**
  - **≥ 10/12 compliant-correct:** the interaction is read from weight structure without simulation.
  - **≤ 8/12:** simulation was the route on this family.

## L3-width: the same interaction family at larger width (written before training)

- **Networks:** L2d states, classes and training, but with hidden width 128 (500 → 128 → 128 → 5). 20 seeds per class;
  inclusion ≥ 0.95. Hypothesis: dominant-term hand simulation degrades as computation spreads over more units.
- **Baselines:** B1–B4 as in L2d, with the same confound rule. If B3 again tracks the class, the reader set is
  counterbalanced on the bias sign as in L2d.
- **Prompt:** the L2d prompt with the description's width changed to 128 and the parameter count updated. Mental
  simulation is not forbidden, which matches L2d and isolates width.
- **Readers:** 12 networks (6 per class), tool-free Astra (effort high). Opus only if its context fits the prompt
  (about 200k tokens).
- **Decision rule:** ≥ 10/12 means reading survives 4× width; ≤ 8/12 means it does not.

## readers_nosim key hash (recorded before any call)

Superseded before any call: the first build drew a new counterbalanced set instead of reusing the L2d networks as this pre-registration requires (hash 10d0e23b…). It was rebuilt from L2d's key with new IDs. Current hash: `419dd6a6c6533d54c66f9ca6805406492d37c25651b35049ddc49c1e71cad49b`

## readers_w128 key hash (recorded before any call)

`88f3b96e66b65fc212980faec6f902ad978b773fd42af91a36d72c21a8c6b3dd`

## Results: L2d-nosim and L3-width (scored November 5, 2026)

**L2d-nosim (tool-free Astra):** **11/12** (p = 0.003). All 12 replies were read by hand and all are compliant: each
states it evaluated no input, and none sums over a constructed example. Three replies declare a guess at 50%
confidence (all three happened to be correct); counting only committed answers, **8/9**. Confidence was 50–78% (L2d:
91–96%) and the Brier score 0.161 (L2d: 0.004). All 34 quoted weights are exact. Method: the replies find hidden
units whose current-ball and previous-frame weight profiles are shifted copies (motion-direction filters), combine
them with the ball-vs-paddle sign, and follow the routing downstream.

**L3-width, 128 units (tool-free Astra, simulation allowed):** **11/12** (p = 0.003), confidence 58–86%, Brier 0.097.
45 of 48 quoted weights are exact, with 3 wrong W2 values: the first transcription errors in this benchmark.

## L4-scale: depth and width (written before training)

- **Family:** L2d (states, classes, training rule).
- **Variants:**
  - depth 4 (four hidden layers of 32);
  - depth 8 (eight hidden layers of 32);
  - width 256 (two hidden layers of 256).
- **Networks:** 20 seeds per class per variant. More epochs (up to 30) are allowed if needed to reach the inclusion
  bar of 0.95, the same for both classes, reported.
- **Baselines:** B1–B4 per variant. Reader sets are counterbalanced on the output-bias sign whenever B3 tracks the
  class (95% interval excluding 0.5), as in L2d.
- **Prompt:** the L2d prompt, with the description, formula and weight blocks generated for the actual depth or
  width. Mental simulation is not forbidden, matching L2d and L3-width.
- **Readers:** 12 networks per variant (6 per class), tool-free Astra (effort high). The width-256 prompt is about
  320k tokens and runs on Astra only.
- **Decision rule per variant:** ≥ 10/12 means reading survives; ≤ 8/12 means it does not.
- **Secondary, reported as a trend over size:** confidence, Brier score, and the share of quoted weights that are exact.

**L4-scale baselines** (seeds 0–19):

| Variant | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| depth 4 | 24/40 | 22/40 | 36/40 | 23/40 |
| depth 8 | 18/40 | 24/40 | 34/40 | 16/40 |
| width 256 | 18/40 | 19/40 | 29/40 | 22/40 |

B3 tracks the class in all three, so all reader sets are counterbalanced. Lowest held-out accuracy: depth 4 0.9991,
depth 8 0.9523, width 256 1.0. **Deviation, before any L4 call:** in depth 4, only 2 right-away networks lean the
minority way, too few for 3 + 3 counterbalancing. 10 more seeds per class (20–29) are trained for every variant, used
only to complete the counterbalanced reader sets. Baselines stay as reported on seeds 0–19.

## scale/readers_d4 key hash (recorded before any call)

`4abb44874011df1234d4c369afacf6893748160affcfa47d153ee4f291dd5f32`

## scale/readers_d8 key hash (recorded before any call)

`94e7ebe50d37d884e5c78c9662fe52e39a03a19c56dcefa15fd4e674477b0ca4`

## scale/readers_w256 key hash (recorded before any call)

`bfa085964f1575b282baa62b271449864befae1b4be2180310dc3bc74ef769b1`

## L4-scale results (scored November 5, 2026; tool-free Astra, effort high)

| Variant | Correct | p (one-sided) | Confidence | Brier | Quoted weights exact |
|---|---|---|---|---|---|
| depth 4 | **9/12** (inconclusive) | 0.073 | 78–90% | 0.190 | 75/75 |
| depth 8 | **7/12** (no evidence) | 0.387 | 50–88%, mostly 50–65% | 0.208 | 64/65 |
| width 256 | not runnable | – | – | – | – |

The width-256 prompts (1,272,573 characters) exceed Codex's per-input limit of 1,048,576 characters. Codex rejected
them before they reached the model (`input_too_large`), so they are excluded and not scored.

**Reading:** tool-free accuracy holds at width 128 (11/12) but falls with depth: depth 2 12/12, depth 4 9/12, depth 8
7/12. At depth 4 the answers stay confident while 3 are wrong, so it is miscalibrated. At depth 8 confidence mostly
falls to near 50%.

## Analyst arm at the breaking point (written before any analyst call)

- **Networks:** the same 12 depth-8 networks and the same 12 depth-4 networks (same blinded IDs and keys as their
  tool-free sets).
- **Reader:** GPT-6 Astra (effort high) with Python, using the Exp 3 loop. Weights are preloaded as `W` (W1…W{L},
  b1…b{L}, in-by-out), linear algebra is allowed, a forward pass is not, at most 100 runs, macOS sandbox. The message
  is the tool-free prompt's description, game and question, with the weights replaced by the computing rules. Depth 8
  runs first.
- **Compliance:** every code run is read for forward passes (running the network or any layer, tanh included, on an
  input). A conversation with a forward pass is reported and not scored.
- **Decision rule per depth:**
  - **≥ 10/12:** explicit analysis without forward passes recovers what tool-free reading lost. The tool-free limit
    is working memory or arithmetic.
  - **≤ 8/12:** the interaction is not recovered from weight structure alone at this depth.

## Analyst-arm results (scored November 5, 2026)

| Depth | Analyst Astra (Python, no forward pass) | Tool-free Astra | Analyst confidence | Code runs per conversation |
|---|---|---|---|---|
| 4 | **12/12** (p = 0.0002) | 9/12 | 88–97% | 2–5 |
| 8 | **11/12** (p = 0.003) | 7/12 | 89–97% | 3–5 |

**Compliance:** the keyword check flagged 2 code runs. Read by hand, both use "forward" only in comments ("not …
forward passes"). No conversation applies tanh or evaluates the network on an input.

**Methods seen in the code:**
- contracting W2…W{L} into a right-minus-left readout direction;
- quadratic fits to the column profiles of the current-ball and previous-frame W1 rows;
- a robustness check that inserts random positive diagonal gains between layers, as stand-ins for unknown tanh
  slopes, and counts how often the sign survives.

The last is a gray zone: no input is evaluated, but it approximates Jacobians at arbitrary operating points. It
follows the stated rule ("applying one weight matrix to directions taken from other weights").

**Reading:** the interaction is recoverable from weight structure by explicit linear algebra at depth 8. The
tool-free decline with depth is a limit on composing the computation in one pass, not missing information.

## L4b: denser depth curve (written before training)

- **Purpose:** tighten the tool-free depth curve.
- **New networks:** depths 3 and 6 (width 32), 30 seeds per class each, trained and counterbalanced exactly like L4.
- **Reader sets (tool-free Astra, effort high):**
  - depths 3 and 6: 12 networks each, 6 per class, counterbalanced on the bias sign;
  - depths 4 and 8: 12 more networks each, drawn from those not used before, with the same counterbalancing.
- **Reporting:** accuracy per depth with exact 95% intervals, and a logistic regression of correctness on depth. The
  pre-specified test is that the depth coefficient is negative (one-sided Wald test, α = 0.05).

L4b baselines (all 60 networks per depth): depth 3 B1 38/60, B2 36/60, B3 45/60, B4 29/60; depth 6 B1 32/60,
**B2 41/60 [0.55, 0.80]**, B3 51/60, B4 25/60. Reader sets are counterbalanced on the bias sign only. The depth-6 B2
signal, which includes the output biases, is reported as a caveat.

## scale/readers_d3 key hash (recorded before any call)

`3e8b81daefe2314161a26a0aa1337cb9aff80c484950c4d67a7905f94eba9412`

## scale/readers_d6 key hash (recorded before any call)

`8ceefe1e58aa68717116329e994c5c4fb97829245b1c116649f94e2b79ed6988`

**Deviation, before any L4b call:** after excluding the first depth-4 set, no minority-lean right-toward networks remain. Seeds 30–59 are trained for depths 4 and 8, used only to complete the second counterbalanced sets.

Depth-8 seeds 30–59: 4 networks below the 0.95 inclusion bar (lowest 0.424) are excluded from reader sets.

## scale/readers_d4b key hash (recorded before any call)

`39adf5a1ba4ea3257c920d706da679d3cbaf5236ee4c7c8e30fb3affca887b09`

## scale/readers_d8b key hash (recorded before any call)

`3abfe474fbb49f8587da417ca4a2025d690308dd475f07b2d0d9d09c44d872fd`

Depth-8 seeds 30–59: 4 networks below the 0.95 inclusion bar (lowest 0.424) are excluded from reader sets. Depth 4 needed seeds 60–119 as well to find 3 minority-lean right-toward networks.

## L4b results (scored November 5, 2026)

- **Tool-free Astra:** depth 3 **10/12** (one reply failed with "model at capacity" and was rerun once), depth 6
  **12/12**, depth 4 set 2 **12/12**, depth 8 set 2 **6/12**.
- **Pooled by depth:** 2: 12/12; 3: 10/12; 4: 21/24; 6: 12/12; 8: 13/24.
- **Pre-registered trend test:** logistic slope on depth −0.53 (SE 0.16), z = −3.22, one-sided p = 0.0006 (on the 11
  depth-3 replies scored at the time; the rerun adds one wrong answer, which strengthens the trend).
- **Post-hoc, labelled as such:** a fixed weight-only curvature recipe classifies 97–100% at every depth
  (`scale/curvature_recipe.json`). Greedy dominant-path tracing decays with depth (`scale/greedy_paths.json`).

## Opus at depth 8 (written before any call, November 5, 2026)

- **Networks and prompts:** the same 12 depth-8 networks and prompts as `scale/readers_d8` (same key).
- **Reader:** Opus 5.5, tool-free, effort max, one reply each.
- **Decision rule:**
  - **≥ 10/12:** Opus's reading survives depth 8 where tool-free Astra's did not (7/12 on this set).
  - **≤ 8/12:** the depth limit holds across both models.
