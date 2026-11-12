# Breaking depth: at what depth can each model no longer read the function? (pre-registration, November 5, 2026)

## Question

For GPT-6 Astra and Claude Opus 5.5, each **with tools** (Python, linear algebra only, no forward pass) and
**without tools** (weights as text, one reply), find the depth at which they can no longer read the L2d
interaction rule ("when the ball moves right, does the paddle move toward or away?").

## Networks

The L2d family (`../ladder/l2/data_d/`): mirror-matched states, imitation training, width 32, hidden depth D. Readers
see the description, formula and weights generated for the actual depth (`../ladder/scale/scale.py`).

- **Existing depths:** 2, 3, 4, 6 and 8.
- **New depths:** 5, 7, 10, 12 and 16, each with 60 seeds per class.
- **Inclusion:** held-out accuracy ≥ 0.95.
- **Reader sets:** 12 networks per depth, 6 per class, counterbalanced on the output-bias sign as in L2d.
- **Same networks for both models:** within a depth, Astra and Opus see the same 12 networks, under new IDs for each
  model's set where a set is reused. The comparison is paired.

## Readers and arms

| Arm | Astra | Opus 5.5 |
|---|---|---|
| **Tool-free** | `codex exec`, effort high, one reply | `claude -p`, effort max, no tools, one reply |
| **With tools** | Exp 3 loop (`../analyst/astra_loop.py`), effort high | the same loop on `claude -p --resume`, effort max, no built-in tools; code runs in the same sandbox |

## Depths per arm (adaptive, rule fixed now)

1. **Initial depths:**
   - Tool-free: depths 2–8, plus 10 and 12 for Astra.
   - With tools: 8, 12 and 16.
2. **Already available** (reused, not rerun):
   - Astra tool-free: depths 2, 3, 4, 6, 8.
   - Opus tool-free: depths 2 and 8.
   - Astra with tools: depths 4 and 8.
3. **Order:**
   - Opus tool-free: depths 4 and 6 first, then 3, 5 and 7 where they help bracket the break.
   - With-tools arms: depth 16 first, then bisect between the deepest passing and the shallowest failing depth.
4. **Stopping:** an arm stops when it has a passing depth and a failing depth that are adjacent among the tested
   depths, and every deeper tested depth also fails. Or it stops when the deepest tested depth still passes, in
   which case the break is reported as "beyond D".

**Pass / fail per depth (12 networks):**
- **Pass:** ≥ 10/12 (one-sided binomial p ≤ 0.019 against 0.5).
- **Fail:** ≤ 8/12.
- 9/12 is inconclusive: 12 more networks from the same depth are added, and the 24 are judged on ≥ 18 pass, ≤ 15
  fail.

## Scoring

- **Unanswered replies count as wrong** (primary). These are output overflow, or a reply with no `Answer:` line,
  after one rerun. Service failures (rate limits, network, capacity) are rerun and not counted.
- **Sensitivity analysis:** the same results with unanswered replies excluded.

## Breaking depth

- **Definition:** for each reader × arm, the shallowest tested depth that fails, with every deeper tested depth also
  failing. Reported with the deepest passing depth: "breaks between D_pass and D_fail".
- **Model-based estimate:** fit P(correct | depth) = 0.5 + 0.5 / (1 + exp(b·(depth − D50))) by maximum likelihood on
  all replies. D50 is where accuracy is midway between perfect and chance (0.75). Report D50 with a 95% bootstrap
  interval (2,000 resamples of networks within each depth).

## Figures

Two figures, accuracy vs depth with 95% intervals and the fitted curve:
1. **Tool-free:** Astra vs Opus.
2. **With tools:** Astra vs Opus.

**Correction (recorded before any call of this study):** the analyst loop copied from the Exp 3 harness runs Astra at
effort **xhigh**, not high. The earlier depth-4 and depth-8 analyst results (12/12, 11/12) were at xhigh, contrary to
the ladder pre-registration's "effort high". The with-tools Astra arm keeps xhigh for consistency, and RESULTS.md
is corrected.

**Note:** within a depth, Opus tool-free reuses the exact prompt files and blinded IDs of Astra's set. Opus never sees
the key, so blinding is unaffected, and the comparison is exactly paired.

## scale/readers_d7 key hash (recorded before any call)

`4c5c74e83d91e02413fd8f653fabca400c75af9cb5f6b02a058a526898e1370c`

**Training outcome (recorded before any call at these depths):** included networks (held-out ≥ 0.95): depth 5
120/120, depth 7 120/120, depth 10 106/120, depth 12 77/120, depth 16 20/120. At depth 16 only one included
right-away network leans left, too few for 3 + 3 counterbalancing. **Deviation:** 120 more depth-16 seeds (60–179)
are trained at 30 epochs, used only to complete the depth-16 set. The depth-16 set may therefore mix networks
trained for 10 and 30 epochs, and this is reported.

**Deviation (before any depth-5 call):** depth 5 has only 2 minority-lean right-toward networks, so seeds 60–179 are trained (10 epochs) to complete counterbalancing.

## scale/readers_d10 key hash (recorded before any call)

`6dbb654f4c865314aa5ae26f9e85f6e304eeefc5a740c9b3029db8d1e1502ec7`

## scale/readers_d12 key hash (recorded before any call)

`ad1bededd48d88f05ee833b08f32118148fc92033f4b709e6cf657d5b7fc52cf`

## Amendment (November 5, 2026): Opus at effort high

At effort max, tool-free Opus replies at depth 4 ran 62–77 minutes and 480k–617k output tokens. 10 of 12 overflowed
the CLI's output limit without an answer (the other 2 were correct). Depth-8 replies behaved similarly. Opus is
switched to **effort high** for this study.

- **Opus arms (tool-free and with tools):** effort high from here on. Astra stays at effort high tool-free and xhigh
  with tools, as before.
- **Max-effort results are kept and reported separately**, not mixed into the high-effort curves:
  - depth 2: 12/12;
  - depth 4: 2 answered (both correct) and 10 overflows;
  - depth 8: 9 resolved (2/7 answered correctly and 2 overflows); 1 attempt flagged by Opus's safeguards
    (`db52b080`); the last reply (`fc564ba5`) was stopped mid-run.

  The 12 depth-6 max replies were stopped mid-run when the queue was halted and are kept in `replies_stopped/`.
- **Opus-high tool-free:** depths 2 (`ladder/l2/readers_d`), 4, 6 and 8 first, on the same networks as Astra, saved
  under `replies/opus_high/`. Then 3, 5 or 7 as needed to bracket the break.
- **Opus-high with tools:** depths 8 and 12, then 16 or bisection.
- **Queue fix:** the queue's parallel limit did not work (its job count was always 0, so all 12 started at once). It
  now waits on process IDs; 2 at a time.

## scale/readers_d5 key hash (recorded before any call)

`eda8cd7cc69ac94ae05771c4249ca2c9b49d2078e92779f9ee42f2e3373c3146`

## scale/readers_d16 key hash (recorded before any call)

`ab798dfc3b1b19339f90362f3cf1ca9b06d09a28df82d3c95691d4b480a68002`

Depth-16 reader set: 2 networks trained 10 epochs, 10 trained 30 epochs.

## Amendment 2 (November 5, 2026): every arm at effort xhigh; equal sets and depth grid

Every arm runs at **xhigh** (`claude --effort xhigh` exists in Claude Code 2.1.314), Opus also reads the
second sets, and the depth grid is equal.

- **Arms (all xhigh):**
  - Astra tool-free: replies under `replies/astra_xhigh/`;
  - Opus tool-free: replies under `replies/opus_xhigh/`;
  - Astra with Python: run names `d<D>_<id>`, already xhigh;
  - Opus with Python: run names `opusx_d<D>_<id>`.
- **Earlier arms, kept and reported separately, not mixed in:** Astra tool-free at high (depths 2–12) and Opus at max
  (depths 2, 4, 8). The Opus-high queue produced no answers (both attempts hit the usage limit) and is withdrawn.
- **Tool-free grid, identical for both models:** depths 2, 3, 4, 5, 6, 7, 8, 10, 12. The second sets at depths 4 and 8
  (`readers_d4b`, `readers_d8b`) are included for both, so n = 24 at depths 4 and 8 and 12 elsewhere. Every
  depth is run regardless of earlier results, with no adaptive stopping.
- **With-Python grid, identical for both models:** depths 4, 8, 12, 16 (the `key_d4`, `key_d8`, `key_d12`,
  `key_d16` sets, 12 networks each).
- **Opus order:** tool-free 2, 4, 6, 8, 4b, 8b, 3, 5, 7, 10, 12; then with Python 8, 12, 4, 16.
- **Concurrency:** Opus 2 at a time; Astra 3 at a time.
- **Covariates recorded per reply:** output and reasoning tokens, and wall time.
- **Added analysis:** a paired McNemar test per depth on the shared networks.

## Deeper networks for the with-Python arms (November 5, 2026, before any call at these depths)

Astra with Python (xhigh) still passes at depth 16 (10/12), so depths 20 and 24 are trained to find the
with-tools break. The recipe is L2d's, width 32, 30 epochs (needed at depth 16), seeds 0–59 per class, with the same
0.95 inclusion bar and counterbalancing. Reader sets and with-Python sets are built and hashed before any call.

## Amendment 3 (November 5, 2026): precision near the break, and benchmark-style reporting

- **More networks:** depths 6, 7, 8 and 10 go to 48 networks each, for both models in the tool-free arm at xhigh.
  New 12-network sets are counterbalanced like the others and exclude every network already used at that depth:
  - depth 6: readers_d6c, d6d, d6e (36 more);
  - depth 7: readers_d7c, d7d, d7e (36 more);
  - depth 8: readers_d8c, d8d (24 more);
  - depth 10: readers_d10c, d10d, d10e (36 more).

  Spare seeds are trained as needed (L2d recipe, 10 epochs). Depths 2, 3, 4, 5 and 12 stay as planned.
- **Reporting, added alongside the pass/fail decision rule:**
  - accuracy per depth with 95% intervals;
  - **mean probability on the correct answer** (stated confidence / 100 if correct, 1 − that if wrong; unanswered =
    0.5), with 95% intervals from resampling networks within each depth;
  - **Brier score** per depth;
  - D50 from the pooled fit.

  The decision rule for "breaks at" uses n = 48 at these depths: pass ≥ 33/48, fail ≤ 29/48 (one-sided binomial
  α ≈ 0.01 against 0.5 for a pass; anything between is inconclusive).

**Depths 20 and 24 (training outcome, before any call):** at 30 epochs, depth 20 included 24/120 networks and depth
24 included 1/120. Depth 20 gets seeds 60–119 (30 epochs) to complete counterbalancing. Depth 24 gets a feasibility
test: 20 seeds per class at 60 epochs. If too few train, depth 20 is the deepest with-Python depth, and this is
reported as a limit of the network family, not of the readers.

## scale/readers_d6c key hash (recorded before any call)

`38ad2af68e9c1203eff73af3ed875aa7524a480ca59a0a0777925bd31aa2c98d`

## scale/readers_d6d key hash (recorded before any call)

`9eff5393d584be6080830e033c63e014c15e52f703cd7869544f11d01a2edd80`

## scale/readers_d6e key hash (recorded before any call)

`abf693b3fa37ee51b77b7a0d0b55d4268636b682c7c6e3e5bf02909e3cb4c4a1`

## scale/readers_d7c key hash (recorded before any call)

`981cee29dac1af9f160c6a9d2b58bc016836b7d282f5b0742d725db86f2b8f22`

## scale/readers_d7d key hash (recorded before any call)

`f0b949d04ad199aa5433b65a8645a0a088e763ee24ea76197040d29f45a42daa`

## scale/readers_d7e key hash (recorded before any call)

`c0cd4846e581fab4d1d7178d36596bbbc0247f434f0cf71e3b02ecc907d8d24d`

## scale/readers_d8c key hash (recorded before any call)

`2b8be4e80c41ed9ae9a148cdd8166b277ace0978bafe5789090695843bc492b8`

## scale/readers_d8d key hash (recorded before any call)

`15194a58fb7b930336d6af8195ae1940adeb8e920e9d278b9a5999ae6abc8a91`

## scale/readers_d10c key hash (recorded before any call)

`98bfe4296883ad59d0f5c3c29da00da4ff340c8b3eb72e9c5ba70bb00b197b20`

## scale/readers_d10d key hash (recorded before any call)

`6d7687ffe4b81b54fb623e32ddf86b0f92590f3b31f3f209bc40f14c4e9d4ae9`

## scale/readers_d10e key hash (recorded before any call)

`bef1275521cba566b0d3bc30f8b57588ced9012f794cc5b25659adedac36fea1`

## Amendment 4 (November 5, 2026): run scheduling for Opus

Amendment 4: operational (run scheduling); no change to method. Model, effort and prompts are unchanged. The
tool-free queues were stopped and restarted; finished replies are kept and unfinished ones are rerun.

**Depth 24 infeasible:** 2/160 networks reached 0.95 (30 and 60 epochs), so depth 20 is the deepest with-Python depth (a limit of the network family, not the readers). Depth 20: 41/240 included.

## scale/readers_d20 key hash (recorded before any call)

`84c72cacf2e2028bd2dfd7d7609a12284eeca8e987974df80facaf6bb21705cc`

## Amendment 5 (November 6, 2026): Astra reruns after a usage limit

Every extra-set Astra reply failed immediately with "usage limit", a service usage limit. The failed extra-set Astra
replies (depths 6, 7, 8, 10, sets c–e) are rerun with the same model, effort (xhigh) and prompts. Astra's main grid
and with-Python runs were completed before the limit.

**Update to Amendment 5:** operational (run scheduling); no change to method.

## Amendment 6 (Nov 6): Opus configuration bug, and reruns

A configuration error in the Opus run script made the CLI report "Not logged in" for part of the calls. Every Opus call affected returned that message and no answer. Affected:
- 167 tool-free opus_xhigh replies. Those sets were depths 3, 5, 7, 10 and 12, plus the d6d/d6e, d7d/d7e, d8c (5 of 12), d8d and d10c–e sets.
- All 32 Opus-with-Python runs (opusx_d4/d8/d12).

None of these is a model response. The tool-free replies are moved to `replies_failed_queue/` and rerun on the same prompts. The analyst runs are moved to `analyst/runs_failed_login/` and rerun from the start. The scorer treats any text containing "Not logged in" as an infrastructure failure, never as an answer. Every valid reply (105 tool-free) is kept unchanged. The configuration error is fixed. Prompts, keys, effort (xhigh) and model are unchanged.

## Amendment 7 (Nov 6, evening): run scheduling for the remaining Opus calls

Amendment 7: operational (run scheduling); no change to method.
- Nothing scientific changes: models, effort (xhigh), prompts, keys and decision rules are the same.
- No tool-free call was in flight when the change was made. Two with-Python conversations already running were
  left to finish.

## Amendment 9 (November 10, 2026, before any training or call at these depths): tool-free depths 14 and 16

At depth 12, the deepest tool-free depth, Astra's fitted curve has not reached the floor (7/12 right, fitted about
0.71), so the tool-free figure ends before the curve does. The tool-free arms are therefore extended
to depths 14 and 16 for GPT-6 Astra and Claude Opus 5.5.

- **Depth 16 networks: nothing new is trained.** The set `scale/readers_d16` is used as it stands: 12 networks,
  key hash recorded above on November 5, the same 12 networks as the with-Python set `analyst/key_d16.json`, and
  no tool-free reply has ever been requested on it.
- **Depth 14 networks (new).** The L2d recipe, width 32: seeds 0 to 59 per class at 10 epochs, inclusion at
  held-out accuracy ≥ 0.95. If any class × output-bias-sign cell then has fewer than 3 included networks, seeds 60
  to 179 are trained at 30 epochs (the depth-16 deviation) and the set is drawn from all included networks. The
  reader set `scale/readers_d14` is 12 networks, 3 + 3 per class, built by `scale.py prompts` with seed 20261340
  (20261200 + 14 × 10); the training outcome and the key hash are recorded below before any call.
- **Readers.** Tool-free only, one reply per network, the calls of Amendment 2: Astra through `codex exec` at
  effort xhigh (`run_astra_effort.sh`, label `astra_xhigh`), Opus 5.5 through `claude -p` at effort xhigh on
  Claude Code 2.1.314 (a pinned copy selected with `CLAUDE_BIN`, since the main CLI is now 2.1.321; label
  `opus_xhigh`). Both models read the same 12 prompts per depth.
- **Not run.** The with-Python arms (depth 16 and 20 already exist there; depth 14 is not added).
- **Analysis.** Unchanged rules: n = 12 per depth (pass ≥ 10, fail ≤ 8), unanswered counts as wrong, service
  failures are rerun. The floor-logistic fit and D50 are refitted on depths 2 to 16; the earlier fit on depths 2
  to 12 (Astra 11.3, Opus 6.1) is kept and reported beside it. No depth is added or dropped after seeing these
  replies. The two fixed scripts (`curvature_recipe.py`, `greedy_paths.py`) are run on every included network at
  depths 14 and 16.
- **Disclosed difference: folder path.** The study moved on November 8, so each call's working folder is now
  under `weight-reading-combined-20261108/depth-benchmark/` where the depth 2 to 12 calls ran under
  `weight-space-benchmark-20261105/`. Both CLIs pass the working-folder path to the model; it names the reader
  set and the blinded id, as before, and nothing about the rule.

**Depth 14 (training outcome, recorded before any call at this depth):** at 10 epochs, 51 of 120 networks are
included (held-out ≥ 0.95): right-toward 24 (7 with the output bias leaning right, 17 leaning left), right-away 27
(18 and 9). Every cell has at least 3, so no 30-epoch seeds are trained and the whole depth-14 set is at 10 epochs.

## scale/readers_d14 key hash (recorded before any call)

`ded769ac1b0ac970e2a5e501bb7ddc19b0ab868c87c6a959dd39401ef382cc9f`

**Amendment 9, inconclusive rule applied (November 10, 10:52, before any call on the new sets).** Astra's 24
replies are in: 9/12 at depth 14 and 9/12 at depth 16. The standing rule of this pre-registration ("Pass / fail
per depth") makes 9/12 inconclusive and adds 12 more networks from the same depth, judged on the 24 (pass ≥ 18,
fail ≤ 15). It is applied as written:
- **New sets:** `scale/readers_d14b` and `scale/readers_d16b`, 12 networks each, 3 + 3 per class, drawn by
  `scale.py prompts` from the included networks not in the first set (`--exclude`), with seeds 20261341 and 20261361
  (the extra-set convention of Amendment 3: 20261200 + 10 × depth + 1). No network is trained for them.
- **Astra** reads both new sets (effort xhigh, as above).
- **Opus** has 12 of its 24 first-set replies so far (2/4 at depth 14, 4/8 at depth 16). The rule is applied per
  reader: Opus reads a depth's second set only if its first 12 at that depth end at exactly 9/12. Otherwise its
  point at that depth stays at 12 networks, and the figure and the paired comparison use the networks each model
  answered.
- The second sets were not planned in Amendment 9's first text ("n = 12 per depth"); they follow from the standing
  rule, which that text left unchanged, and from nothing else in the results.

## scale/readers_d14b key hash (recorded before any call)

`0e5d5ba228414b3cff5f9fffec5d952792ef6bfb56597899698a1f3650420624`

## scale/readers_d16b key hash (recorded before any call)

`7458c467e8d1bf9be6d7367a848ed7f52fa5bc35f814c6ebb14ed11225c592f7`

**Amendment 9, outcome (November 10, 12:40).** All 72 calls returned a reply on the first attempt (48 Astra, 24
Opus); no call was rerun. Astra: depth 14, 9/12 then 6/12, 15/24 = fail; depth 16,
9/12 then 9/12, 18/24 = pass. Opus: depth 14, 7/12 = fail; depth 16, 4/12 = fail (one reply withheld by a safety
classifier, counted as wrong). Opus ended at 9/12 at neither depth, so it read no second set. With a pass at depth
16 above fails at 12 and 14, Astra has no breaking depth under the definition above ("every deeper tested depth
also failing"); its fitted D50 on depths 2 to 16 is 12.2 [9.9, 15.8] (11.3 [9.7, 17.1] on depths 2 to 12, kept in
`breaking_depth_depths2to12.json`). Opus: 6.1 [5.4, 6.8], unchanged.

## Amendment 10 (November 10, 2026, before any training or call for it): Astra to 48 networks at depths 12, 14 and 16

After Amendment 9, Astra's tool-free points beyond depth 10 rest on 12 networks at depth 12 (7/12) and 24 at depths
14 and 16 (15/24 and 18/24). The depth-16 point was judged too high to publish on 24 networks; a post hoc
split showed it owes about 1.5 networks to how uncommitted guesses fell (9 of 15 right). More
networks are therefore read, **by Astra only for now**.

- **Sets.** Astra goes to 48 networks at each of depths 12, 14 and 16, as at depths 6 to 10 (Amendment 3):
  - depth 12: `readers_d12b`, `readers_d12c`, `readers_d12d` (36 more), seeds 20261321 to 20261323;
  - depth 14: `readers_d14c`, `readers_d14d` (24 more), seeds 20261342 and 20261343;
  - depth 16: `readers_d16c`, `readers_d16d` (24 more), seeds 20261362 and 20261363.

  Each is 12 networks, 3 + 3 per class, built by `scale.py prompts`, excluding every network already in a reader
  set at that depth. Key hashes are recorded below before any call on a set.
- **Networks.** Depth 16 has enough included networks. Depths 12 and 14 do not (smallest unused cell: 5 at depth 12,
  1 at depth 14), so spare seeds are trained with the base recipe (L2d, width 32, 10 epochs): seeds 60 to 119 per
  class at depth 12 and 60 to 179 at depth 14; if a depth-12 cell is still short, seeds 120 to 179 there. Both
  depths stay at 10 epochs throughout.
- **Reader.** Astra, tool-free, effort xhigh, one reply per network, through the run script (the call of
  Amendment 2).
- **Opus is not run on these sets** (it may be added later on the same sets). Its points at these
  depths stay at 12 networks, and the paired comparison uses the networks both models answered.
- **Analysis.** All 84 replies are run and counted whatever they show; nothing is dropped. At 48 networks the rule
  of Amendment 3 applies (pass ≥ 33, fail ≤ 29). D50 is refitted on all replies. The partial-credit measure (mean
  probability on the correct answer), which scores a 50% guess at 0.5 however it lands, is reported beside accuracy
  at these depths.
- **Disclosure.** This amendment was written after seeing Astra's depth 12, 14 and 16 results and because one of
  them looked too high. It adds networks; it changes no rule and removes no reply.

## scale/readers_d16c key hash (recorded before any call)

`d37f7aa388e4c00f10030274d2956a32e5511e45c20e7ae24f983b91318649e6`

## scale/readers_d16d key hash (recorded before any call)

`a7c0c22b7c40ef8d1732085abc2a1b7ef279fd8251bbeea625074314382fe33a`

**Amendment 10, training outcome (recorded before any call on the depth-12 and depth-14 sets):** at 10 epochs,
depth 12 now has 157 of 240 networks included (seeds 0 to 119) and depth 14 has 152 of 360 (seeds 0 to 179). Unused
included networks per class × bias-sign cell: depth 12: 18, 54, 53 and 20; depth 14: 17, 44, 49 and 18. Every cell
covers the new sets, so seeds 120 to 179 are not trained at depth 12.

## scale/readers_d12b key hash (recorded before any call)

`d488fd8a544e6bbd94b4d89252d68a468a52656d6ef1325bd247322d60545699`

## scale/readers_d12c key hash (recorded before any call)

`e4a6e11c114033cda64b3a7c862b5d6aaa74987198ef60916d0a0467926aa786`

## scale/readers_d12d key hash (recorded before any call)

`827eeb87a237e436ea2881c327e08c6e5d9e8296c74dbdf789e562ad989ac983`

## scale/readers_d14c key hash (recorded before any call)

`34d8c0a613d00f3062a9c4bb01111990a861fd0b99e84127285ac9f7b618d039`

## scale/readers_d14d key hash (recorded before any call)

`07b4361e14ac25e9954cf3a2e513fed8a4e92494322e467641b44da9222ba215`

**Update to Amendment 10 (November 10, 16:25, before any Astra reply on the Amendment 10 sets was
scored): Opus reads the extra sets too.** These results are now plotted against reply length as well as depth,
and Opus reads the same networks, so both models reach 48 networks at depths 12, 14 and 16.
- **Sets:** the nine sets Opus has not read: `readers_d12b`, `d12c`, `d12d`, `readers_d14b`, `d14c`, `d14d`,
  `readers_d16b`, `d16c`, `d16d` (108 replies), the same prompts and blinded ids as Astra. This supersedes the
  per-reader second-set rule applied in Amendment 9 (under which Opus skipped `d14b` and `d16b`): every set at
  these depths is now read by both models, as in Amendment 2.
- **Call:** unchanged (Opus 5.5, `claude -p`, effort xhigh, tools off, Claude Code 2.1.314).
- **Order:** one set per depth at a time (b sets, then c, then d; depth 16, 14, 12 within each round), so that an
  interrupted run leaves the depths with equal numbers of networks.
- **Interruptions:** if not all 108 replies can be run, the analysis reports which sets are in.
- **Analysis:** as in Amendment 10; all replies are counted whatever they show.

**Amendment 10, outcome (November 10, 21:45).** Astra: all 84 replies returned on the first attempt. Opus: all 108
replies are in; four calls hit a network outage at about 20:20 ("Can't reach the API server"), were set aside as
infrastructure failures and rerun. At 48 networks per depth:
- Astra: 29/48 at depth 12 (fail), 29/48 at depth 14 (fail), 32/48 at depth 16 (undecided: between 29 and 33).
  Partial credit 0.59, 0.57 and 0.58. D50 on all replies: 11.1 [9.8, 12.8].
- Opus: 18/48, 26/48 and 24/48 (all fail; one depth-16 reply withheld by a safety classifier, counted as wrong).
  Partial credit 0.48, 0.49 and 0.50. D50: 6.1 [5.5, 6.8].
