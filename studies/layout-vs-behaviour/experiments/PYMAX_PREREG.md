# Twins with Python at matched effort: GPT-6 Astra at max (pre-registration, November 10, 2026, before any call)

Reviewed before launch by three independent readers (queue, parity, design); this is the text after their fixes.
No model call of this arm had been made when it was written.

## Why

Rows 5 and 6 of the post's twin figure ("Open question, with Python", "Behaviour question, with Python") show Astra at
effort `high`; Opus runs at `max`. Every figure is to show Astra at max against Opus at max, and on
November 10 these two rows were added, to be run with Astra at max on the rebuilt networks. The four no-tools questions
on the rebuilt networks are a separate run (`V2GAMES_PREREG.md`); this arm adds the two Python rows.

## Networks

The 24 camouflaged networks of the twin set as it stands after the rebuild (`twin_audit_20261110/AUDIT.md`), both
training runs, two camouflaged copies per policy:

- **12 twins:** Dodge (unchanged, `suite_twins`, `suite_twins2`), Hunter v2 and Car-tag v2 (`suite_twins_v2games`,
  `suite_twins2_v2games`).
- **12 known games:** Catch and Pac-Man (unchanged) and Freeway v2.

Nothing is rebuilt for this arm: the scripts read `<suite>/nets/<id>/actor.txt` of those four suites. Job list
`pymax_jobs.txt`, 48 lines, built from the four `key.json` files: the camouflaged twins shuffled with Python's
`random.Random(20261110)`, each with the open question and then the behaviour question; then the camouflaged known
games shuffled with the same generator, the same way.

## Arms

- **Open question, Python, max.** `analyst/analyst_loop_max.py`: `analyst/analyst_loop.py` with three changes,
  `model_reasoning_effort="max"`, the output folder `<suite>/analyst_max/`, and a usage limit ends the conversation
  instead of waiting (as `analyst_loop_stance.py` already does).
- **Behaviour question, Python, max.** `analyst/analyst_loop_stance_max.py`: `analyst/analyst_loop_stance.py` with
  two changes, the effort and the output folder `<suite>/analyst_stance_max/`. The two max scripts differ from each
  other in the question (the warning and behaviour paragraph and the `Stance:` line of `build_hint.py`) and the folder.
- Same message, sandbox, 30-run limit, instructions file and Codex binary as the `high` conversations
  (`codex-cli 0.166.0-alpha.5.1`). The queue checks the binary's version before every conversation, writes it to
  `codex_version.txt` in the conversation folder with `effort.txt`, and stops if it differs. One conversation per
  network and question. Model `gpt-6-astra`; running the network on inputs the model builds is allowed, as before.
- **Tools.** All Codex tools are disabled, as before. Codex still advertises a custom tool `exec` whose host is
  disabled (a call returns "code-mode host is disabled"; 5 of the 48 `high` conversations called it once). A
  conversation with any tool item is kept and listed.
- Queue `run_queue_pymax.sh`, 2 conversations at a time, from this folder (`obscured`), where the `high`
  conversations ran.
- **What the model sees that differs from the `high` conversations:** the effort; the working-folder path, which
  ends `<suite>/analyst_max/astra_<id>/cwd` or `<suite>/analyst_stance_max/astra_<id>/cwd` (the `high` behaviour
  conversations ran under `suite_twins_pystance/analyst/`; for the rebuilt games the suite name carries `_v2games`);
  and the current date, which Codex sends with the path. The path names no game. Codex also sends context it builds
  from `~/.codex` (its installed-skills list, saved command prefixes on resumed turns); it is the same mechanism as
  in the `high` conversations and is not edited for this run. After the run the Codex session record of every
  conversation is checked: model `gpt-6-astra` and effort `max` on every turn, CLI version, whether the skills block
  equals that of the `high` conversations of November 10, and whether any message was sent twice. Differences are
  reported with the results; no conversation is discarded for them.
- **A failed turn is not re-sent.** It ends the conversation, and the queue starts that conversation again from the
  first message (the behaviour of the `high` conversations of November 10).
- **No concurrent `high` control.** For Dodge, Catch and Pac-Man the existing `high`
  conversations are on the same networks (open question November 5, 6 and 8; behaviour question November 10); for
  the rebuilt games there are none.
- **No pilot.** The same loop, binary and effort flag ran four conversations at max on other networks this morning
  (`../recognition/suite_stage6/analyst_max/`, effort `max` confirmed in their session records).

## Usage limits

- A usage-limit event from Codex stops this queue (`STOP_PYMAX`); see "Incomplete runs" below.

## Failed and interrupted conversations

- A conversation is finished when `state.json` says done and `final_answer.md` is not empty. A finished
  conversation is always kept, whatever it says.
- A conversation that ends unfinished is a failed attempt: it is moved to `<suite>/<arm>_failed/astra_<id>.attemptN_*`
  and started again, up to 3 attempts per network and question over all runs of the queue. A network with 3 failed
  attempts counts as wrong and is listed.
- **Not attempts**, each listed with the reason: a conversation cut off by a usage limit (`.limit_*`), by the stop
  file (`.stopped_*`), or from outside, found unfinished at the next start (`.interrupted_*`). The queue is stopped
  by hand only for usage-limit or CLI-version reasons, never after reading a conversation, and no unfinished
  conversation is opened before the arm is complete.
- Three failed attempts in a row stop the queue for a person to look, so that a fault of the service does not use
  up every network's attempts.
- A conversation that gives a final answer after 0 code runs is listed, and counts are given with and without it
  (the rule of `LADDER2_PREREG.md`). The primary test uses every finished conversation as coded.

## Scoring (fixed now)

- `score_pymax.py` (self-contained). `pack` writes each final answer under a random code into `pymax_packets/`, in
  one shuffle over both questions and all four suites, with the mapping outside the packet folder. It packs only
  when all 48 conversations are finished or have failed three times; if a usage limit stops the queue, the 24 twin
  answers are packed as one set (`pack twins`) and the known games later as a second.
- The coder is a fresh instance of a separate coder model (Claude Sonnet) that sees only the packet files and the brief `CODER_BRIEF_TWINS.md`, used
  verbatim (the rebuilt-games run's brief; SHA-256 below). It returns stance (approach / avoid / unclear),
  setting, asserts_opposite, named game and a supporting quote. **Stance correct** = the coded stance equals
  `key.json` "stance"; unclear is wrong. These answers are coded once, by this arm; if another run also codes
  them, this arm's codes are used and disagreements are listed. No hand-coding.
- `score` writes `pymax_scores.json` (suite, id, game, twin, condition, question, effort, coded stance, coded named
  game, stance_correct, code runs) and prints the tables below.

## Outcomes and how they will be read

- **Row 5:** stance correct on the 12 camouflaged twins, open question. **Row 6:** the same with the behaviour
  question. Both also per twin game (4 networks each).
- **Known games:** all 12, and without Freeway v2 (Catch and Pac-Man, 8 networks). Freeway v2 is labelled "avoid" by
  its objective although its policies step towards nearby cars (`V2GAMES_PREREG.md`); its coded stances are listed
  and no claim rests on it.
- **Primary:** behaviour question against open question at max on the 12 twins, paired exact two-sided McNemar.
  **With Python the question, not the tool, was missing, at max too** if the behaviour question is ahead and
  p < 0.05; otherwise counts with no claim.
- For reference at `high` on the earlier twin set (`score_ladder2.py baseline`): open 4 of 12 twins (Dodge 4, Hunter
  0, Car-tag 0), behaviour 12 of 12; known games 11 and 10 of 12. On the networks reused here: Dodge 4 of 4 and 4 of
  4, Catch 4 and 4, Pac-Man 3 and 4.
- **Not tested:** max against `high`. Dodge, Catch and Pac-Man are described side by side with their `high`
  conversations. Hunter v2, Car-tag v2 and Freeway v2 are new networks; their counts are not compared with the v1
  `high` counts as an effect of effort.
- **Opus.** No Astra-against-Opus test is made in this arm. Opus's row 5 is whatever open-question Python
  conversations the twin queue completes on these networks (today: one of the 24); it has none for row 6.
- **Secondary:** code runs per conversation; whether the conversation ran the network on inputs it built (read from
  the code, descriptive); reasoning tokens per conversation; which game each answer names; a scan of the code for
  file access outside its work folder (the sandbox can read `/private/tmp`), listed.
- **Incomplete runs.** If a usage limit stops the queue, it is resumed later with the same scripts and job list.
  No test is applied before all 24 twin conversations are finished or failed three times; earlier counts are
  labelled interim and list what is missing. If the arm is abandoned incomplete, counts are reported without a test
  or a reading.

## Limits known in advance

- One conversation per network and question; 4 networks per game, 2 camouflaged copies of each of 2 policies.
- Parity with Opus is by effort label, and Opus has almost no Python conversations on these networks.
- No `high` control on the rebuilt networks.
- Behaviour-question answers carry a `Stance:` line, so the coder can tell the two questions apart.
- An answer file left by a failed call of a turn would be read as that turn's reply (inherited from the `high`
  loop; with no re-sending of turns it cannot arise within a conversation).
- This file and the scripts are not committed; the hashes below record what was fixed before the first call.

## SHA-256

| File | SHA-256 |
|---|---|
| `analyst/analyst_loop_max.py` | `b99cb9917544fd9ace69bf717b377be317d7ecd542459473b8b653b0b312d8f2` |
| `analyst/analyst_loop_stance_max.py` | `dc7a421163b5b73ecdb513a72aab0dfe24041e29f06fc46463c71b3edc6484ee` |
| `analyst/analyst_loop.py` | `5424b40116b96cc9c3752cd051227ba30a6a572161a074955bba3485a97a74a3` |
| `analyst/analyst_loop_stance.py` | `394c6208521da6e9bffd5b72b73f6203c9ce35fae626e4977faf3a2bb5ee7160` |
| `analyst/instructions.md` | `b33fe0624137cd6f190e42a8659046baf16261f3c29cd6455734523ba473afeb` |
| `arch.py` | `41476c18c02d15a25a267688129a53e2e298b4ebb84784ee5f5458202e78961b` |
| `../../weight-space-exp4-repro-20261102/upstream/exp1_which_game/networks/description.txt` | `2361ff107c85c3dafa85420e79503cc01747eed462a8216f7c06ef711d8d93bd` |
| `run_queue_pymax.sh` | `22ff134a2906b690d07f24df29baf6913ea9628a192d2a9afaede8e54668ff68` |
| `pymax_jobs.txt` | `bdfcfda8cee18856c237dc2e7251cb16d357c0a30fec21a653f9aeda9421ff13` |
| `score_pymax.py` | `806a168bed7d9371bfc9952cde0c7921db6b06fb326adb30758f84e1f35ef915` |
| `CODER_BRIEF_TWINS.md` | `be1ce6a7c7d0b1edf5f14f086efa9dc81f7a111008fb5b07a609ce5140f22f32` |
| `suite_twins/key.json` | `a9b89baf70e1f9eb1af72c533717f054617b365f339186e86d214c73f510ac21` |
| `suite_twins2/key.json` | `fdc9183e250d697746fa832bf8fcf282a94be77fae757d9aa052d9337fffedf5` |
| `suite_twins_v2games/key.json` | `9cf0753071536fec74a095a4b6f48025d07896ef8b0b1072c46efe5ea5eaeac3` |
| `suite_twins2_v2games/key.json` | `5a452b4800332fa703b4bc1906e19b55c8802c63e50452210bea94bd3de59a05` |
| The 24 `actor.txt` files | `05ac5199b73b433fd440c1402b721ca261601b7dbebfcae81e0ae76e6ef8c8a5` |

The last value is the SHA-256 of the lines `<sha256>  <suite>/nets/<id>/actor.txt` for the 24 networks of the job
list, sorted by suite and id.

## Amendment 1: second Codex configuration (November 10, 12:47 PST, before any call on it; no conversation had been read)

- **Why.** With 13 of the 48 conversations finished, a usage limit meant the remaining conversations had to run under
  a second Codex configuration folder.
- **What changes.** The queue was stopped with its stop file at 12:44; the two conversations under way finished
  (queue end 12:47; its log and stop file are kept). The remaining 35 conversations run under the second
  configuration: same Codex binary and version, scripts, job list, order, question and effort. Finished
  conversations are not rerun. Every conversation folder records which configuration it ran under.
- **What the model sees that differs between the two configurations.** Codex builds part of its context from its
  configuration folder. The second folder has only the five built-in skills (the first also lists 19 user-installed skills),
  no `config.toml`, no `rules` and no saved command prefixes, so the skills list is shorter and resumed turns carry no
  user-config context. Neither folder has text in `AGENTS.md`. Nothing is copied between the folders. The post-run
  check of the session records reports the difference as observed.
- **Reading.** The outcomes, primary test and readings are unchanged. Counts are also given by configuration, as
  description: the first 13 are 7 open-question and 6 behaviour-question conversations on 7 twin networks; all 12
  known-game networks and the other twin conversations run under the second configuration. If a question's two
  conversations on one network ran under different configurations, that is listed.
- **New hashes.** `run_queue_pymax.sh` `1bc85e6cfff7fb96b66f8844511470dfd23e34eea6838585a15cabf18013b002` (adds the
  configuration record, the configuration check and the start line). The analyst scripts, job list, scorer and brief
  are unchanged.
