# Heavier rewrites at matched effort: GPT-6 Astra at max (pre-registration, November 10, 2026, before any call)

## Why

The blog figure "Game named after heavier rewrites" compares Astra at effort `high` with Opus 5.5 at `max` on 28
exactly rewritten networks. On the camouflaged networks of the naming test, running Astra at `max` took it from 26 to
43 of 48 (`../obscured/MATCHED_EFFORT_RESULTS.md`). Every Astra-against-Opus figure is to be shown
max against max. This arm reruns Astra at `max` on these 28 networks.

## Networks

The 28 networks of the figure, nothing rebuilt: `suite_stage5` conditions `synth`, `pairs`, `basis` (12 networks,
tanh 500-32-32-5) and `suite_stage6` conditions `split` (4, tanh) and `deep0`, `deep3`, `deep6` (12, ReLU; 3, 6 and 9
hidden layers). One trained policy per game in each row. Prompts are the existing `<suite>/prompts/<id>.txt`, byte for
byte the ones Astra (high) and Opus (max) read. Job list `max_parity_jobs.txt` (random order, seed 20261110, each
network at max and then at high), SHA-256 `12772f18f43d6280a27e412d3346ca6c2bed055e33e6f801917dee2bf0fd0233`.

## Arms

- **Astra at max.** `run_astra_max.sh` (SHA-256 `547520fa3755…`, the script of the matched-effort naming arm, byte
  for byte): `run_astra.sh` with `model_reasoning_effort="max"`. Tools off, one reply per network, into
  `<suite>/toolfree_max/astra_<id>/`.
- **Concurrent high control.** Each network once more at `high` with the unchanged `run_astra.sh`, into
  `<suite>/toolfree_high2/astra_<id>/`.
- Queue `run_queue_astra_max.sh` (SHA-256 `4861db4e5ae6…`, unchanged), 3 replies at a time, run from this folder,
  where the Astra (high) replies ran. `codex-cli 0.166.0-alpha.5.1`; a usage-limit event stops the
  queue and it is resumed later. Nothing in `toolfree/` is touched.

## Scoring (fixed now)

- The figure's rule: the first game named on the reply's Game line has to be the right one
  (`game_first` of `../obscured/score_matched_effort.py`); a reply with no Game line names no game. Applied to the
  existing Astra (high) replies before any call it reproduces the published Astra counts in all 7 rows (3, 3, 2, 4, 1,
  1, 1 of 4).
- Failed calls, attempts and unanswered networks as in the matched-effort naming arm: a completed reply is always
  kept; a failed call is repeated, up to 3 attempts in total; 3 failed attempts count as wrong and are listed.
- No hand-coding and no model grader.

## Outcomes and how they will be read

- **Primary:** Astra (max) against Astra at high (the concurrent control) on the 28 networks, paired exact two-sided
  McNemar. **Effort changes the picture** if p < 0.05; otherwise **no detectable effect of effort**, with both counts.
- **Against Opus (max):** counts per row and in the two groups (16 tanh, 12 ReLU), and the same paired test over the
  28. With one policy per game per row the networks within a game are not independent; counts are the result.
- **Secondary:** the control against the earlier high replies (agreement); which games are missed and what they are
  called; median output tokens, reasoning tokens and wall time per reply.
- The Python count quoted in the post (row 7, Astra 1 of 4 at high) is not rerun.

## Limits known in advance

- One reply per network; 4 networks per row, one per game.
- Parity is by effort label. Opus wrote far more per reply than Astra does at max.
- These are the footprints study's networks; its own analysis and paper are not changed by this arm.

## Addendum: the Python count at max (November 10, 08:34 PST, before any analyst call at max)

Written while the tool-free queue above was running, with its first replies in and none scored. It replaces the
last outcome line ("the Python count ... is not rerun"): the post compares Astra (high) with Opus (max) on that count
too, so it is rerun at max.

- **Networks:** the four 9-layer networks of row 7 (`suite_stage6`, condition `deep6`): `0ac437bc` (Pong), `df216717`
  (Breakout), `489c08d3` (Pac-Man), `dc3cf4b6` (Freeway).
- **Arm:** `analyst_loop_r_max.py` (SHA-256 `f56d45c68632…`): `analyst_loop_r.py` with two changes,
  `model_reasoning_effort="max"` and the output folder `suite_stage6/analyst_max/`. Same question, sandbox, 30-run
  limit, instructions file and Codex binary (`codex-cli 0.166.0-alpha.5.1`) as the high conversations of November 6.
  Runner `run_analyst_max.sh` (SHA-256 `324b80c61db1…`), two conversations at a time. One conversation per network.
- **Scoring:** `game_first` on `final_answer.md`. On the existing conversations it gives Astra (high) 1 of 4 and Opus
  (max) 3 of 4, the counts in the post. A conversation that answers after 0 code runs while describing runs that never
  happened is set aside and rerun once, the rule already applied to Opus.
- **Reading:** counts only, 4 networks. No test.
- **No concurrent high control** for this arm: the high conversations are four days older, which is a limit.
- **Scorer** (written November 10, 08:36 PST, with 4 of 56 tool-free replies in; no max reply had been read):
  `score_max_parity.py`, SHA-256 `89c60c66fc52…`. On the existing replies it reproduces every published cell of the
  figure except one: Opus, row 6, Pac-Man, whose line is "Dodge / Avoider ... similar to evading the ghosts in
  Pac-Man". `game_first` credits it (Dodge is not one of its keywords); the figure does not, because the line leads
  with another game. The same strict reading applies to Astra: the scorer lists every credited Game line, and a line
  that leads with a different game is not credited in the figure. Both counts are reported if they differ.

## Addendum 2: the two baseline rows (November 10, 08:53 PST, before any call on these networks at max)

The figure's note compares the rewrites with the untouched policies and with copy camouflage ("Untouched, Astra names
3 of the 4 tanh policies and Opus all 4"), and a combined figure in preparation shows both as rows. Those Astra
replies are at `high`. So the 8 networks of `suite_stage5` conditions `plain` and `camo_v2` (the four tanh policies,
untouched and copy-camouflaged) are also run at `max`, each with a concurrent `high` control, by the same scripts,
scoring and failure rules. Job list `max_parity_jobs2.txt` (SHA-256 `5a6bd03c9bcf…`, random order, seed 20261110), started by
`run_max_parity2.sh` (SHA-256 `614488a36503…`) when the first queue has ended. Written during a network outage that began at
about 08:37, with 5 of the first queue's 56 replies in and none read. Counts per row only; the primary test stays on
the 28 rewritten networks.
- Scorer extended to the two baseline rows at 08:55 PST (SHA-256 `b17bdf168f77…`): rows `plain` and `camo` are printed and
  written to the table but are in no test or total. No scoring rule changed.
