# Experiment 3: Twin games

Post sections: *Recognizing a game is not understanding the network*, *A pointed question gets the behavior*,
*Python alone does not*. Figure: **QuestionLadder** (figure 4), and the two twin rows of **Scoreboard**.

## Question

When a model names the game, is it reading what the network does, or recognizing a familiar layout? A *twin game*
keeps a famous game's board and pieces but rewards the opposite behaviour, so only reading the behaviour gets it
right.

## Setup

- **Games.** Three pairs that share a board. The second of each pair is the twin:

  | Famous game | Twin | The player... |
  |---|---|---|
  | Catch | Dodge | collects falling objects / avoids them |
  | Pac-Man | Hunter | flees the creatures / chases them (touching one pays +1, pellets pay nothing) |
  | Freeway | Car-tag | avoids the cars / steps onto them (+1; the goal row cannot be entered) |

  Hunter, Car-tag and Freeway are second versions (`_v2`). In the first versions the Hunter policies ignored the
  creatures and one Freeway policy only ever moved up, so their rewards were changed and the policies retrained. The
  behaviour audit that found this is [`twin_audit_20261110/AUDIT.md`](../../studies/layout-vs-behaviour/experiments/twin_audit_20261110/AUDIT.md).
  Dodge, Catch and Pac-Man are unchanged.
- **Networks.** 500 → 32 → 32 → 5 tanh, PPO, 50 million steps, two training runs. Each policy is tested as trained
  and in 2 camouflaged copies (experiment 1's copy camouflage): 18 networks per run, 9 of them twins.
- **Questions** (rows of figure 4), no tools unless stated, both models at effort max, one reply per network:

  | Row | Question |
  |---|---|
  | 1 `open` | the original open question, "Which game was it trained on?" |
  | 2 `hint1` | adds: "the game may be a well-known game, or a lesser-known variant that has the same layout … Networks of both kinds are in this set, so do not decide from the layout alone" |
  | 3 `hint2` | row 2's warning plus "work out from the weights whether the agent moves toward the moving objects (channel 1) or away from them", and a Stance line in the answer |
  | 4 `forced` | a choice between the pair's two games, each defined in one sentence; option order varies (A is right for 11 of the 18 twins) |
  | 5 `python` | row 1's open question, on the 12 camouflaged twins and 12 camouflaged famous-game networks, with a sandbox: up to 30 code runs, any analysis, including running the network on inputs the model builds |
  | 6 `python_stance` | row 3's warning and behaviour question, with the same sandbox, in separate conversations |

- **Scoring.** A separate grader model, blind to which model wrote the answer, codes whether the answer says the
  player approaches or avoids the moving objects (unclear counts as wrong). In row 4 the answer is the letter.
- **Opus replies that ran out of output at max** were rerun at xhigh and count, marked with an asterisk in the
  figure (36 in all, 9 of them in row 4). The post, the figure and the table below include them.

## Results (twins whose behaviour is described right)

These are the numbers reported in the post. The model replies behind them are not included in this repository;
running the readers below on the stored prompts produces new ones.

| Row | Astra (max) | Opus (max, xhigh reruns counted) | Famous games, Astra / Opus |
|---|---|---|---|
| 1 open question | 9 / 18 | 7 / 18 (4 at xhigh) | 18 / 18, 18 / 18 |
| 2 + look-alike warning | 16 / 18 | 15 / 18 (10 at xhigh) | 18 / 18, 16 / 18 |
| 3 + behaviour question | 18 / 18 | 17 / 18 (7 at xhigh) | 18 / 18, 16 / 18 |
| 4 forced choice | 18 / 18 | 16 / 18 (6 at xhigh) | 18 / 18, 18 / 18 |
| 5 open question, Python | 8 / 12 | 8 / 12 | 12 / 12, 12 / 12 |
| 6 behaviour question, Python | 12 / 12 | 11 / 12 | 12 / 12, 12 / 12 |

- On the open question both models call Car-tag Frogger every time; Astra describes Hunter as a chase about half the
  time; Opus calls 2 of its 6 Dodge networks Catch.
- On the camouflaged twins, Python lifts Astra on Hunter (2 of 4 to 4 of 4) and nothing else; Car-tag stays at 0 of 4.

## Files

| What | Where |
|---|---|
| Games and training | [`train/games.py`](../../studies/layout-vs-behaviour/experiments/train/games.py) (Catch, Dodge, Pac-Man), [`train/games_v2.py`](../../studies/layout-vs-behaviour/experiments/train/games_v2.py) and [`train/train_v2.py`](../../studies/layout-vs-behaviour/experiments/train/train_v2.py) (Hunter, Car-tag, Freeway v2); policies `train/g_<game>[_s2]/`, `train/g_<game>_v2[_s2]/` (`_s2` = second training run) |
| Behaviour audit and rebuild | [`twin_audit_20261110/`](../../studies/layout-vs-behaviour/experiments/twin_audit_20261110) (`AUDIT.md`, `audit_v2.py`, `build_suite_v2games.py`, `eval_play_v2games.py`; helpers in `vendor/`); stance checked in each policy's own play: [`state_probe_own_play.py`](../../studies/layout-vs-behaviour/experiments/state_probe_own_play.py) |
| Row 1 suites (networks, prompts, key) | Dodge, Catch, Pac-Man: [`suite_twins/`](../../studies/layout-vs-behaviour/experiments/suite_twins) (run 1), [`suite_twins2/`](../../studies/layout-vs-behaviour/experiments/suite_twins2) (run 2); Hunter, Car-tag, Freeway v2: [`suite_twins_v2games/`](../../studies/layout-vs-behaviour/experiments/suite_twins_v2games), [`suite_twins2_v2games/`](../../studies/layout-vs-behaviour/experiments/suite_twins2_v2games) |
| Rows 2–4 (same networks, new prompts) | `suite_twins[2]_hint1/`, `suite_twins[2]_hint2/`, forced choice `suite_bridge_fc/` (run 1) and `suite_twins2_fc/` (run 2); the v2 games in `suite_twins[2]_v2games_{hint1,hint2,fc}/`. Built by [`build_hint.py`](../../studies/layout-vs-behaviour/experiments/build_hint.py), [`build_bridge.py`](../../studies/layout-vs-behaviour/experiments/build_bridge.py), [`build_ladder2.py`](../../studies/layout-vs-behaviour/experiments/build_ladder2.py), [`build_v2games_conditions.py`](../../studies/layout-vs-behaviour/experiments/build_v2games_conditions.py) |
| Rows 5–6 (Python) | [`analyst/analyst_loop_max.py`](../../studies/layout-vs-behaviour/experiments/analyst/analyst_loop_max.py) (open question), [`analyst/analyst_loop_stance_max.py`](../../studies/layout-vs-behaviour/experiments/analyst/analyst_loop_stance_max.py) (behaviour question); `suite_twins_pystance/` holds the networks |
| Running the readers | [`job_arch.sh`](../../studies/layout-vs-behaviour/experiments/job_arch.sh) (`tf` no tools, `an` / `as` with Python), [`job_hint.sh`](../../studies/layout-vs-behaviour/experiments/job_hint.sh), [`run_astra_max.sh`](../../studies/layout-vs-behaviour/experiments/run_astra_max.sh), [`run_claude.sh`](../../studies/layout-vs-behaviour/experiments/run_claude.sh) |
| Blind coding | coder brief [`CODER_BRIEF_TWINS.md`](../../studies/layout-vs-behaviour/experiments/CODER_BRIEF_TWINS.md); packers [`pack_hint.py`](../../studies/layout-vs-behaviour/experiments/pack_hint.py), [`pack_v2games.py`](../../studies/layout-vs-behaviour/experiments/pack_v2games.py) |
| Scoring | [`score_twins.py`](../../studies/layout-vs-behaviour/experiments/score_twins.py), [`score_ladder2.py`](../../studies/layout-vs-behaviour/experiments/score_ladder2.py), [`score_v1max.py`](../../studies/layout-vs-behaviour/experiments/score_v1max.py), [`score_v2games_fc.py`](../../studies/layout-vs-behaviour/experiments/score_v2games_fc.py), [`score_fc_parity.py`](../../studies/layout-vs-behaviour/experiments/score_fc_parity.py), [`score_pymax.py`](../../studies/layout-vs-behaviour/experiments/score_pymax.py), [`score_pymax_opus.py`](../../studies/layout-vs-behaviour/experiments/score_pymax_opus.py) |
| Pre-registrations | [`HINT_PREREG.md`](../../studies/layout-vs-behaviour/experiments/HINT_PREREG.md), [`LADDER2_PREREG.md`](../../studies/layout-vs-behaviour/experiments/LADDER2_PREREG.md), [`BRIDGE_PREREG.md`](../../studies/layout-vs-behaviour/experiments/BRIDGE_PREREG.md), [`FC_PARITY_PREREG.md`](../../studies/layout-vs-behaviour/experiments/FC_PARITY_PREREG.md), [`V2GAMES_PREREG.md`](../../studies/layout-vs-behaviour/experiments/V2GAMES_PREREG.md), [`PYMAX_PREREG.md`](../../studies/layout-vs-behaviour/experiments/PYMAX_PREREG.md), [`OPUS_PYMAX_PREREG.md`](../../studies/layout-vs-behaviour/experiments/OPUS_PYMAX_PREREG.md), [`EFFORT_MATCH_PREREG.md`](../../studies/layout-vs-behaviour/experiments/EFFORT_MATCH_PREREG.md) |
| Figure data | [`make_ladder_v2.py`](../../studies/layout-vs-behaviour/experiments/make_ladder_v2.py) writes the `ladder` and `twins` blocks from the score files |
