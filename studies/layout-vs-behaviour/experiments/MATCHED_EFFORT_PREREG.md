# Matched-effort naming arm: GPT-6 Astra at effort max (pre-registration, November 9, 2026, before any call)

## Why

On the 48 camouflaged networks of the naming test, Opus 5.5 named the game on 48 and Astra on 26. The two readers did
not run at the same reasoning effort: Astra ran at `high` (the setting of the original post) and Opus at `max`. At the
median Astra wrote about 3,500 output tokens per reply and Opus about 123,000. A rerun of Astra at `xhigh` on its 9
misses in `suite_v2` recovered 5. This arm asks how much of the 48 to 26 gap remains when Astra runs at its own highest
single-agent setting.

## Networks

The same networks as the published comparison, with nothing rebuilt:

- **Primary, 48 camouflaged networks:** `suite_v2` (16), `suite_relu24x3` (12; its 4 camouflaged Breakout networks are
  superseded), `suite_relu24x3_bfix` (4 rebuilt ReLU Breakout networks), `suite_tanh48x1` (16). They are 4 camouflaged
  copies of each of 12 trained policies (4 games in 3 architectures).
- **Secondary, 12 plain networks:** the 4 uncamouflaged networks of `suite_v2`, `suite_relu24x3` and `suite_tanh48x1`.

Prompts are the existing files `<suite>/prompts/<id>.txt`, byte for byte the ones Astra (high) and Opus (max) read.
Answer keys are unchanged. SHA-256 values recorded now:

| File | SHA-256 |
|---|---|
| `suite_v2/key.json` | `29786913a2669fec61e865896a883c1b6c4f84d77b347c4628f49f4d03ef5980` |
| `suite_relu24x3/key.json` | `6627ffa8c779ba1a9e88a22a5153f6c30eca0aa0d6d30b7755279034c854a1e9` |
| `suite_relu24x3_bfix/key.json` | `e256c5a52b42784afe2e601e0aa73e6d0c77f6811e0612f60e48a812e107270e` |
| `suite_tanh48x1/key.json` | `b0cb827c99bc209e7c9788660887bc9432dcb1e2ea1a7861eff020615dab0fbe` |
| `matched_effort_jobs.txt` | `47a6fbd8f484b8778e6ede1698adb45e420139e8038638ad86414f6ae293b014` |
| `run_astra_max.sh` | `547520fa375504a6927cf224094e0cbaad47255f3113356db9671e3fa367a3ce` |
| `run_astra.sh` (unchanged) | `7f0e68f145735f3c550edaa288ea4d1d0eead591c997de68ea47a4d5695f564f` |
| `run_queue_astra_max.sh` | `4861db4e5ae69601d4222c35ba378dd913b7c6da0acd53b1ea7eeb69be334fb6` |
| `score_matched_effort.py` | `9dc4e7ae990cf3ea265e9350cea613af276c20aaf14a532e18f7d21b6848aeba` |
| `score_suite.py` | `e7ac5d730f6201dbf557ffeb1d085d59d7de1a81429e2eac593a2f08481d3ec4` |
| `models.py` | `3ca2e8a5d3e173af7bdcf15bbae4faf1f2bf115dfe1d59d702f32e2ce9af3c92` |
| The 60 prompts | `e3e96aa5a703c633a888165efd59ae998bf67d6ac784b57f10edf26c92bbab5b` |

The prompt-set value is the SHA-256 of the output of `shasum -a 256 <suite>/prompts/<id>.txt` over the 60 distinct
networks of the job list, sorted by suite and id.

## Arms

- **Astra at max (the arm).** GPT-6 Astra through `codex exec` with `run_astra_max.sh`. It is `run_astra.sh` with one
  change: `model_reasoning_effort="max"` in place of `"high"` (it also writes `effort.txt` and records the effort in
  `run.json`). Model `gpt-6-astra`, built-in tools disabled, web search disabled, read-only sandbox, the same one-line
  system instruction, one reply per network. All 60 networks.
- `max` is the highest effort the Codex model list gives for `gpt-6-astra` that keeps a single agent ("Maximum
  reasoning depth for the hardest problems"). The list also has `ultra` ("Maximum reasoning with automatic task
  delegation"), which is not used.
- **Concurrent high-effort control.** The 48 camouflaged networks are also run once more at `high` with the unchanged
  `run_astra.sh`, into `<suite>/toolfree_high2/`, interleaved with the max replies. It gives the agreement between two
  high-effort replies on the same network and a same-week baseline for the effort comparison.
- **Where the calls run.** From the original working folder (`weight-space-exp1-2-shortcuts-20261103/obscured`), where
  the Astra (high) replies ran, because Codex includes the working-directory path in the model's context (each xhigh
  reply read exactly 4 more input tokens than its high reply on the same prompt). The scripts, job list, keys and
  prompts there are byte-identical copies of this folder's. The only model-visible differences from the high arm are
  the effort and `toolfree_max` (or `toolfree_high2`) in place of `toolfree` in that path.
- **CLI version.** `codex-cli 0.166.0-alpha.5.1`, the version of the earlier Astra replies, recorded per reply in
  `codex_version.txt`. The queue stops if a reply reports another version.
- Replies go to `<suite>/toolfree_max/astra_<id>/` and `<suite>/toolfree_high2/astra_<id>/`. Nothing in `toolfree/`
  is touched.
- **Order.** `matched_effort_jobs.txt`: the pilot network first, then the other 47 camouflaged networks in a random
  order (seed 20261109), each at max and then at high, then the 12 plain networks at max. Queue:
  `run_queue_astra_max.sh`, 3 replies at a time.

## Pilot

The first call is `suite_v2 ae1d1ba0` at max, alone. It is kept as that network's one reply whatever it says, and its
Game line is not read before the queue starts. The queue starts only if:

- `run.json` has `rc` 0 and `events.jsonl` ends with `turn.completed`;
- `codex_version.txt` is `codex-cli 0.166.0-alpha.5.1`;
- input tokens are within 10 of the high reply's 76,279 for this network (same prompt, nothing added);
- reasoning tokens exceed the xhigh reply's 11,912 for this network (high: 4,142).

If `~/.codex/logs_2.sqlite` has rows for the reply's thread id, they must show the effort `max`. If the
reasoning-token check fails, `077ece9d` and `dbc67670` (xhigh 17,084 and 11,328 reasoning tokens) are run at max and
kept, and the arm goes ahead only if at least 2 of the 3 exceed their xhigh reply. Otherwise the arm stops and is
reported as "max not distinguishable from xhigh through this CLI".

## Scoring (fixed now)

- **Game.** The reply's Game line is found with the two patterns of `score_suite.game_of` and mapped with its keyword
  rules. Two cases are fixed in advance: if the Game line names more than one game, the first one named counts; a reply
  with no Game line is scored as naming no game. `score_matched_effort.py` applies this to the new replies and to the
  existing Astra (high) and Opus (max) replies. Run on the existing replies before any call, it gives Astra 26 of 48
  and Opus 48 of 48 camouflaged, and Astra 10 of 12 and Opus 9 of 12 plain, the published counts (it reproduces the one
  hand adjudication of `arch_stats.py` without an override). The count under the unchanged `game_of` rule is reported
  beside it, and every Game line that names more than one game is listed.
- No hand-coding and no model grader.
- **Failed calls.** A call has failed when `codex exec` exits non-zero, reports a `turn.failed` event, or leaves no
  `answer.md`. The attempt folder is moved to `<suite>/toolfree_max_failed/` (or `toolfree_high2_failed/`) and the call
  is repeated, up to 3 attempts in total per network over all runs of the queue. A reply that `codex exec` completed is
  always kept, whatever its text: no reply is discarded or asked again because of its content. Failed calls are not
  answers and are never scored.
- **Unanswered.** A network with 3 failed attempts counts as wrong in the primary count and is listed.
- **Usage.** A usage-limit event from codex stops the whole queue (`STOP_ASTRA_MAX`) and is not an attempt. The
  queue is then resumed with the same scripts, job list and CLI version, and again if needed, until each network has
  a reply or 3 failed attempts. No test and no reading below is applied before all 48
  camouflaged networks are in that state; any earlier count is labelled interim and lists the networks not yet run. If
  the arm is abandoned incomplete, counts are reported without a test or a reading.

## Outcomes and how they will be read

- **Primary:** Astra (max) correct out of 48 camouflaged networks, against Opus (max) on the same networks, by a paired
  exact two-sided McNemar test.
  - **The gap persists** if p < 0.05. With Opus at 48 of 48 this means Astra (max) at 42 or fewer.
  - **No detectable gap at this sample size** if Astra (max) is at 43 or more. This is reported with the count and the
    share of the gap closed; it is not by itself read as "effort accounts for the gap".
  - **Effort is credited with closing the gap** only if, in addition, Astra (max) beats the concurrent Astra (high)
    control on the same 48 at p < 0.05 by the same test.
- **Sensitivity analyses, fixed now and reported beside the primary whatever it shows:**
  - Policy level: each of the 12 trained policies scored by its number of camouflaged copies named (0 to 4), exact
    two-sided sign test over the policies where the two models differ.
  - The 40 networks without the two tanh Breakout policies, which play barely better than random (McNemar; p < 0.05
    means Astra (max) at 34 or fewer of 40).
  - If the primary test and the policy-level test disagree, the result is reported as "a gap confined to k policies",
    not under either headline.
- **Secondary:**
  - The high control against the November 4 and 5 high replies: agreement, and McNemar. If the two high arms differ at
    p < 0.05, drift or noise is reported as unresolved.
  - Astra (max) against each high arm; share of the gap closed, (Astra max − 26) / 22.
  - Counts by game and architecture, and the labels the misses get (the high-effort misses called tanh Breakout "pong"
    and ReLU Pac-Man "other").
  - The 12 plain networks, descriptively.
  - Median output tokens, reasoning tokens and wall time per reply at `max`.
- The 9 xhigh replies of `suite_v2` are used in no test. The high, xhigh and max replies on those 9 networks may be
  shown side by side as description only; the 9 were chosen because Astra missed them at high.

## Limits known in advance

- One reply per network, and the 48 networks are 4 copies each of 12 policies, so they are not 48 independent trials.
- Effort names are each vendor's own. `max` for both models matches the label, not the token budget: Claude Code let an
  Opus reply continue past its 128,000-token output limit, and Astra's replies may be far shorter. Token counts are
  reported so the reader can judge.
- The Opus and Astra (high) replies date from November 4 and 5. This arm starts on November 9 or 10 and may finish
  on a later day. The model served under `gpt-6-astra` could have changed in
  between; the concurrent high control is there to show it.
- The two tanh Breakout policies play barely better than random, so "Breakout" may not be recoverable from their
  behaviour at any effort.
- This file and the scripts are not committed yet; the hashes above are the record of what was fixed before the first
  call.
