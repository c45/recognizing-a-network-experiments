# Pre-registration: reader reruns on the rebuilt twin games (written 2026-11-10 10:05 PST, before any model call on these prompts)

**Why.** The November 10 behaviour audit (`twin_audit_20261110/AUDIT.md`) found that v1 Hunter policies ignore the
prey, v1 Freeway seed 1 is "always up", and v1 Car-tag reads as approach only at the car's next cell. Car-tag, Hunter
and Freeway were rebuilt (`train/games_v2.py`) so that each policy's moves depend on the moving objects. Catch, Dodge
and Pac-Man are unchanged and keep their v1 results.

**Networks.** `suite_twins_v2games` (seed 1) and `suite_twins2_v2games` (seed 2): per game one plain and two
camouflaged networks, 18 in all; 12 twins (Hunter v2, Car-tag v2: approach) and 6 known-game controls (Freeway v2).

**Conditions** (`build_v2games_conditions.py`; weight text unchanged; tool-free; one reply per network):
1. open "which game?" (the suites' own prompts, the post's Exp 1 wording);
2. forced choice between the two games of the pair (Car-tag v2 vs Freeway v2; Hunter v2 vs Pac-Man), question text
   of `build_bridge.py` with the rebuild's rule sentences, option order randomised;
3. hint1: variant warning; 4. hint2: warning plus stance question (texts of `build_hint.py`).

**Readers.** Astra at high effort via `run_astra.sh` (run now). Opus 5.5 at max effort with the cap
ladder, through the twin study queue; reported separately as they arrive.

**Scoring.** As for v1. Open and hint answers: a separate coder model (Claude Sonnet), blind to model, condition, game version and key,
codes stance (approach / avoid / unclear), setting, asserts_opposite and named game with the existing brief;
`score_twins.py` scores against `key.json`. Forced choice: the `Answer:` letter against `truth_letter`; unparseable
is wrong. Primary outcome: stance correct on the 12 twins under the open question; secondary: each other condition
against the open question on the same networks (exact McNemar). Missing replies count as wrong only in a final table;
interim counts are labelled interim.

**Fixed in advance.**
- Freeway v2 is a known-game control labelled "avoid" by its objective (contact ends the episode). Its policies still
  step toward nearby cars because the goal lies beyond them (AUDIT.md), so an answer that reports that movement may be
  coded approach or unclear and scored wrong; Freeway is reported separately and no twin claim rests on it.
- The Hunter v2 forced choice differs from Pac-Man in two stated ways (creatures pay, pellets do not), so a reader
  can choose correctly from pellet-seeking alone. It is reported, but the open and hint conditions are the test of
  stance reading for Hunter.
- v1 results are kept and reported beside v2. For v1 Hunter the audit's behaviour-correct answer is "ignores the
  creatures", so v1 Hunter "errors" are not evidence about stance reading; the paper will say so.

**Predictions.** If the v1 pattern was real and not an artefact of weak policies: open question still near zero on the
12 twins (famous game named); forced choice and hint2 high. If v1 failures on Hunter and Car-tag came from the
policies not expressing the stance, the open question may now succeed on v2 twins.

**Order and stopping.** Seed 1 before seed 2; within a seed: open, hint2, forced choice, hint1. The Astra queue stops
on a Codex usage-limit event and is not retried against the limit.

## Amendment 1 (2026-11-10 10:08 PST; before any max-effort call; no reply to the prompts above had been read)

- **Astra at effort max is added and becomes the primary Astra arm.** The rule for the post's figures is matched
  effort: Astra at max against Opus at max. Every prompt of the eight
  suites is also run with `run_astra_max.sh` (unchanged) into `<suite>/toolfree_max/astra_<id>/`; the high replies
  already queued (`toolfree/`) are the concurrent control and the link to the v1 high-effort results. Queue
  `run_queue_v2games_max.sh`, order: seed 1 open, forced choice, hint2, hint1, then seed 2. Same stop on a Codex
  usage-limit event. Blind coding pools max and high answers; the coder does not see the effort.
- **The swap is confirmed** (about 10:08 PT): Hunter v2 and Freeway v2 are used as well as Car-tag
  v2; the v1 Hunter and Car-tag jobs still waiting in the Opus queue are removed; the post is updated
  once Astra and Opus 5.5 are complete on the rebuilt networks.

## Amendment 2 (2026-11-10 15:58 PST; before any of these calls)

- **Astra at max on the unchanged games, to complete the matched-effort ladder.** The post's twin set becomes Dodge
  (v1, unchanged) with Hunter v2 and Car-tag v2, and its famous-game set Catch and Pac-Man (v1) with Freeway v2.
  Astra at max exists on the unchanged networks for the open question (both sets) and the set-1 forced choice only.
  Added: `run_astra_max.sh` (unchanged) on the Dodge, Catch and Pac-Man networks of `suite_twins_hint1`,
  `suite_twins_hint2`, `suite_twins2_hint1`, `suite_twins2_hint2` and `suite_twins2_fc` (existing prompts, 45
  replies), into `<suite>/toolfree_max/astra_<id>/`; Dodge twins first. Old Hunter, Car-tag and Freeway networks
  are not run. Coding and scoring as above (hint answers join the blind packets; forced choice by letter).
- **Codex configuration.** Because of a usage limit, these 45 replies run under the second Codex configuration
  folder of `PYMAX_PREREG.md` (Amendment 1). Each reply folder records the configuration. It gives the model a
  shorter built-in skills list (no config, rules or saved prefixes), as noted there.
- **About 15:57 PT:** the post's twin section is updated with Astra complete and Opus 5.5 marked
  partial (seed 1, replies answered so far), and refreshed when Opus finishes.
