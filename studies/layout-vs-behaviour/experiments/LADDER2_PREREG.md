# Pre-registration: three Astra arms on the twin networks (written November 9, 2026, 23:46 PST, before any call)

Reader in every arm: GPT-6 Astra through `codex exec` (`codex-cli 0.166.0-alpha.5.1`). Built by
`build_ladder2.py`; run by `run_queue_ladder2.sh`, 3 calls at a time, after the matched-effort naming queue
(`MATCHED_EFFORT_PREREG.md`) has ended; scored by `score_ladder2.py`. No network is retrained or rebuilt, and no
existing reply folder is touched.

## Why

The twin-game results rest on one training run and one effort setting, and one cell of the design was never run.

- The question ladder (open 0/9, warning 4/9, warning + stance question 7/9, forced choice 8/9 twins) was run on the
  nine twin networks of `suite_twins` only. `suite_twins2` (second training run) has the open answers only (1/9).
- Every Astra twin answer is at effort `high`. "The open question gets recognition" may depend on that.
- With Python, the open question got 4 of 12 camouflaged twins. Python with the stance question was never run.

## Arm A: question ladder on the second twin set (54 replies)

- `suite_twins2_hint1`, `suite_twins2_hint2`: the 18 `suite_twins2` prompts with the hint paragraphs of
  `build_hint.py` (imported, not retyped). The text before the weights is identical to the set-1 hint prompts.
- `suite_twins2_fc`: the same prompts with the forced-choice question of `build_bridge.py` (imported).
  - **Deviation from set 1:** option order is balanced (truth is A on 4 of the 9 twins and 5 of the 9 known games,
    and every game has both letters), so a constant letter scores chance. Set 1 used independent coin flips (8 A, 10 B).
- Tool-free, effort `high`, `run_astra.sh` unchanged, one reply per network, into `<suite>/toolfree/astra_<id>/`.
- **Primary:** stance correct on the 9 twins of set 2, each rung against the open question on the same networks
  (existing replies, 1/9), exact two-sided McNemar.
- **Pooled:** set 1 (existing) and set 2 together, 18 twins per rung, the same test. Also hint2 against hint1 and
  forced choice against hint2 on the 18.
- **Reading, fixed now.** A rung **replicates** if its set-2 count is above the open count and the pooled test has
  p < 0.05. The stance question **adds to the warning** only if hint2 beats hint1 on the 18 at p < 0.05; otherwise
  the two are reported as not separated. Known-game counts are reported beside every rung (the cost of the question).

## Arm B: open question at effort max (36 replies, and 36 at high beside them)

- The 36 open prompts of `suite_twins` and `suite_twins2`, byte for byte, with `run_astra_max.sh` (effort `max`,
  the script of the matched-effort arm), into `<suite>/toolfree_max/astra_<id>/`.
- **Concurrent control:** each network is also asked once more at `high` with `run_astra.sh`, into
  `<suite>/toolfree_high2/astra_<id>/`, right after its max reply is started.
- **Primary:** stance correct on the 18 twins at max against the concurrent high replies, exact two-sided McNemar.
- **Reading, fixed now.** **Effort alone makes Astra read the behaviour** if max beats the concurrent high at
  p < 0.05. Otherwise **no detectable effect of effort**, reported with both counts. Not read as "effort cannot help".
- **Secondary:** max and high2 each against the November 5 to 7 high replies (1/18 twins); agreement between the two
  high replies; known games at max (18); per twin game; reasoning tokens, output tokens and wall time per reply.

## Arm C: Python with the stance question (24 conversations)

- The 24 camouflaged networks of both twin sets (12 twins, 12 known games), the networks of the existing Astra
  analyst arm. `suite_twins_pystance/nets/<id>` links to the original network folders.
- `analyst/analyst_loop_stance.py` is `analyst/analyst_loop.py` (hash below) with two changes: the hint2 paragraph
  and `Stance:` line of `build_hint.py` are added to the question, and a usage limit ends the conversation instead of
  waiting. The message differs from the earlier analyst message in those two places only. Effort `high`, at most 30
  code runs, the same sandbox; running the network on constructed inputs is allowed, as before.
- The 12 twins run first, then the 12 known games.
- **Primary:** stance correct on the 12 camouflaged twins against Python with the open question on the same networks
  (existing conversations, 4/12), exact two-sided McNemar.
- **Reading, fixed now.** **With Python the question, not the tool, was missing** if the stance question beats the
  open question at p < 0.05. If it does not, the result is reported as counts with no claim.
- **Secondary:** known games (existing 11/12); per twin game (Car-tag and Hunter were 0/4 each); code runs per
  conversation; whether each conversation ran the network on inputs it built (read from the code, descriptive);
  tool-free hint2 on the same 12 twins.
- A conversation with a final answer and 0 code runs is listed, and counts are given with and without it.
  `analyst_grounding.py` is run on the folder.

## Scoring (all arms)

- **Forced choice:** the `Answer:` line, with the pattern of `score_bridge.py`. Unparseable counts as wrong.
- **Everything else:** one shuffled packet set over all conditions (`score_ladder2.py pack`), coded by a separate
  coder model (Claude Sonnet) that sees only the answer text (no condition, model, network or key), with the codes of `score_twins.py`
  (stance approach / avoid / unclear toward the channel-1 objects; setting; asserts_opposite; named game). Stance
  correct = the coded stance matches `key.json`.
- `score_ladder2.py baseline`, run on the existing replies before any call, gives open 0/9 and 1/9 twins, hint1 4/9,
  hint2 7/9, forced choice 8/9, and Python with the open question 4/12 twins and 11/12 known: the published counts.
- **Failed calls.** As in the matched-effort arm: a call that exits non-zero, reports a failed turn or leaves no
  answer is moved to `<arm folder>_failed/` and repeated, up to 3 attempts per network and arm over all runs of the
  queue. Each attempt first waits for the network. A completed reply is always kept, whatever it says. A network with
  3 failed attempts counts as wrong and is listed.
- **Usage.** A usage-limit event stops the queue (`STOP_LADDER2`) and is not an attempt. The queue is resumed with the
  same scripts, job list and CLI version. No test or reading is applied to an arm before every network in it has a
  reply or 3 failed attempts; earlier counts are labelled interim. The queue also stops if a reply reports another
  CLI version.
- **Order.** `ladder2_jobs.txt`: arm A (the three rungs interleaved, random order), arm C, arm B (random order, each
  network at max then high). Seed 20261110.

## Limits known in advance

- One reply per network and condition. Each training run gives three twin policies, each as trained and in two
  camouflaged copies, so 9 twin networks are not 9 independent trials.
- Answers to the stance question carry a `Stance:` line, so the coder can tell those conditions from the open ones.
- Freeway (known game) is labelled "avoid" from its objective, but its policies step toward nearby cars on net
  (RESULTS.md round 9). Known-game counts under the stance question are also given without Freeway.
- The existing replies date from November 5 to 9. Arm B has a concurrent control for this; arms A and C compare
  against the earlier replies, as round 11 did.
- Astra only. `max` is the highest single-agent effort in the Codex model list; `ultra` is not used.
- This file and the scripts are not committed yet; the hashes below are the record of what was fixed before the first
  call.

## SHA-256

| File | SHA-256 |
|---|---|
| `build_ladder2.py` | `694b65542cd36c6a5e74f52d40ac65d253e3ac1fe4f3a4172ce5f560db18bd44` |
| `score_ladder2.py` | `b49e585d5746de562a28682b443247a4d2e4d1fb854638ba9f1e415de56dd8d2` |
| `run_queue_ladder2.sh` | `379b1d65c8e39d883b2fab0feda6577fa31dba53334eae9372468c428e8678ba` |
| `ladder2_jobs.txt` | `d71992f0ca022c73473188ce23b5bc082d30d99e400daf0d38e3d6a7b59b9ccd` |
| `analyst/analyst_loop_stance.py` | `394c6208521da6e9bffd5b72b73f6203c9ce35fae626e4977faf3a2bb5ee7160` |
| `analyst/analyst_loop.py` | `5424b40116b96cc9c3752cd051227ba30a6a572161a074955bba3485a97a74a3` |
| `analyst/instructions.md` | `b33fe0624137cd6f190e42a8659046baf16261f3c29cd6455734523ba473afeb` |
| `instructions.md` | `b33fe0624137cd6f190e42a8659046baf16261f3c29cd6455734523ba473afeb` |
| `run_astra.sh` | `7f0e68f145735f3c550edaa288ea4d1d0eead591c997de68ea47a4d5695f564f` |
| `run_astra_max.sh` | `547520fa375504a6927cf224094e0cbaad47255f3113356db9671e3fa367a3ce` |
| `build_hint.py` | `818d571938ba2a774d415f3d207376396a0f1f97cabe79dade19779fbdedd8d7` |
| `build_bridge.py` | `a8867bfc4878a60bdd61f08bf1032875cc2286694cd2c57a8a21358f34b60079` |
| `score_bridge.py` | `b3c67fe9a7ea8bf4721a3bedf562211c9dc6e5fbe6b2588f8e2e46a401db85d1` |
| `score_twins.py` | `b5f9cc3108550ec991c88c00ccf6748a2a838fe882ad77c40508f8ed04207eb1` |
| `models.py` | `3ca2e8a5d3e173af7bdcf15bbae4faf1f2bf115dfe1d59d702f32e2ce9af3c92` |
| `arch.py` | `41476c18c02d15a25a267688129a53e2e298b4ebb84784ee5f5458202e78961b` |
| `suite_twins/key.json` | `a9b89baf70e1f9eb1af72c533717f054617b365f339186e86d214c73f510ac21` |
| `suite_twins2/key.json` | `fdc9183e250d697746fa832bf8fcf282a94be77fae757d9aa052d9337fffedf5` |
| `suite_twins2_hint1/key.json` | `fdc9183e250d697746fa832bf8fcf282a94be77fae757d9aa052d9337fffedf5` |
| `suite_twins2_hint2/key.json` | `fdc9183e250d697746fa832bf8fcf282a94be77fae757d9aa052d9337fffedf5` |
| `suite_twins2_fc/key.json` | `72c599560599efcd04ce09bb35ea9a3b0553da03e6d3ac53372d374d6f8e816d` |
| `suite_twins_pystance/key.json` | `f845dfe036aaf260d1a77e9a5d5660e7f60955d0ab034d0bb7c9e7cd4286ed1a` |
| The 18 prompts of `suite_twins2_hint1` | `2e51591aba3b1dcc6c914b9832a65589e4140044edbf2ff1a56ac2094390861d` |
| The 18 prompts of `suite_twins2_hint2` | `21cb69d255866c9e18e32e6f607391e0de3d175d8d7ca43f44ed38e202d3e40b` |
| The 18 prompts of `suite_twins2_fc` | `0d5467895f895631ec899e5dfc4672a3ea598a08983792fb2a80a26d5050e45b` |
| The 18 prompts of `suite_twins` | `509bab1835636116616cfed5010d5d5fc19b38853935815c957ced61eb2392a9` |
| The 18 prompts of `suite_twins2` | `1f09fab77a3baa0431c83c7c8c8bb05617dbe2f78d9681b2d100398ff2ac3ccc` |
| The 24 `actor.txt` files of `suite_twins_pystance` | `16b758d8427903e8ad6209638621ece8a3309a09a1cfd9cf411ff3500b39ae4d` |

A prompt-set value is the SHA-256 of the output of `shasum -a 256 *.txt` run inside the suite's `prompts` folder
(for the last row, `shasum -a 256 */actor.txt` inside `suite_twins_pystance/nets`).
