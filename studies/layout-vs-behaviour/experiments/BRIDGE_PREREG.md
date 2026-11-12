# Pre-registration: minimum bridge (written 2026-11-08 12:27 EST, before any model call on these prompts)

**Question.** Is the gap between the depth benchmark (forced choice, 12/12 tool-free at depth 2) and the twin games
(open "which game?", 1 of 26 twin answers right) due to the question asked?

**A. suite_bridge_fc** (`build_bridge.py`): the 18 suite_twins networks (unchanged weight text). The open question is
replaced by a forced choice between the two games of the network's twin pair, each defined in one sentence; option
order is randomised per network (8 A / 10 B). Readers: Astra (high), Opus 5.5 (max), one reply each, as in the open
twin runs. Scoring: the Answer line (A/B) against key.json; unparseable = wrong. Reported separately for twin and
known games and against the open condition on the same networks (paired).

**B. suite_bridge_open**: the depth benchmark's 12 depth-2 L2d networks (`ladder/l2/readers_d`, weight text
unchanged; 6 right-toward, 6 right-away). The forced-choice rule question is replaced by "What is that rule? Describe
precisely how the paddle moves in response to the ball"; the game description is kept and the rule hint removed.
Readers: Astra (xhigh, as in the benchmark), Opus 5.5 (max). Scoring, blind-coded by a separate model with model
names hidden: (i) what the answer says the paddle does when the ball moves right: toward / away / not stated (an
unconditional "tracks the ball" counts as toward); correct = matches the class; (ii) whether the stated rule depends
on the ball's horizontal direction. A reader defaulting to "tracks the ball" scores 6/12.

**Predictions** (question hypothesis): A well above the open condition on twins (approaching the benchmark's level); B
near 6/12 with few direction-dependent rules. If A stays low, the networks (RL training, camouflage), not the
question, drive the gap.
