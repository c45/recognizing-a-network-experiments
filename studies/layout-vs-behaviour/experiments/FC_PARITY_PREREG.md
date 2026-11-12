# Pre-registration: forced choice at matched effort (written November 10, 2026, 00:02 PST, before any call)

## Why

On the forced choice between a twin pair's two games (`suite_bridge_fc`, BRIDGE_PREREG.md), Astra ran at effort `high`
and Opus 5.5 at `max`. The two readers are to be at the same effort on this row. As in the naming test
(MATCHED_EFFORT_PREREG.md), parity is reached by running Astra at `max`, its highest single-agent setting.

## Arms

- **Astra at max (the arm).** The 18 prompts of `suite_bridge_fc`, byte for byte, with `run_astra_max.sh` (unchanged),
  one reply per network, into `suite_bridge_fc/toolfree_max/astra_<id>/`.
- **Concurrent high control.** Each network is also asked once more at `high` with `run_astra.sh` (unchanged), into
  `suite_bridge_fc/toolfree_high2/astra_<id>/`, right after its max reply is started.
- **Opus 5.5 (existing design).** Effort `max`, with the pre-registered fallback to `xhigh` when a reply ends at the
  128,000-token output cap. At this time 13 of 18 are answered (10 at max, 3 at xhigh) and the last five are running
  at xhigh (`run_queue_opus_fc.sh`). No Opus reply is rerun for this arm.
- Queue `run_queue_fc_parity.sh`, 2 replies at a time, job list `fc_parity_jobs.txt` (random order, seed 20261110,
  each network at max then high2). `codex-cli 0.166.0-alpha.5.1`; own stop file `STOP_FC_PARITY`.
  Nothing in `suite_bridge_fc/toolfree/` is touched.

## Scoring (fixed now)

- `score_fc_parity.py`: the `Answer:` line with the pattern of `score_bridge.py`, against `key.json`. Unparseable
  counts as wrong. Run on the existing replies before any call it gives Astra (high) 8/9 twins and 9/9 known games,
  and Opus 4/5 twins and 8/8 known: the published counts.
- Failed calls, attempts and usage limits are handled as in the matched-effort naming arm: a completed reply is always
  kept; a failed call is repeated up to 3 times; a network with 3 failed attempts counts as wrong and is listed.

## Outcomes and how they will be read

- **Primary:** Astra (max) against Opus on the 9 twins, paired, exact two-sided McNemar, once all 18 Astra (max)
  replies and all Opus replies are in (or exhausted on the ladder). **The readers differ at matched effort** if
  p < 0.05; otherwise **no detectable difference**, reported with both counts. With 9 networks the test can only
  detect a large gap; counts are the result.
- **Secondary:** Astra (max) against the concurrent high control and against the November 8 high replies (does
  effort change the forced choice?); agreement between the two high replies; the 9 known games; per twin game;
  reasoning tokens, output tokens and wall time per reply.

## Limits known in advance

- "Matched" means the vendors' effort labels match. Opus at `max` wrote 100,000 to 512,000 output tokens per reply
  here; Astra at `max` wrote about 16,000 on the naming test.
- Up to 8 of the 18 Opus answers are at `xhigh`, because the reply at `max` ended at the output cap. Astra is not
  rerun at `xhigh` on those networks.
- One reply per network; the 9 twins are 3 policies, each as trained and in two camouflaged copies. Option order is
  not balanced in this suite (8 A, 10 B).
- The Astra (high) and most Opus replies date from November 8 and 9.

## SHA-256

| File | SHA-256 |
|---|---|
| `run_astra_max.sh` | `547520fa375504a6927cf224094e0cbaad47255f3113356db9671e3fa367a3ce` |
| `run_astra.sh` | `7f0e68f145735f3c550edaa288ea4d1d0eead591c997de68ea47a4d5695f564f` |
| `run_queue_fc_parity.sh` | `83ff17eb49b2da6d898950104b0796b339edbb08dbdf46e4af1f6a234033bf89` |
| `score_fc_parity.py` | `12b21646a3b671ade9efc193453d75e80f94484f4790ed9825ee6065c01da6f3` |
| `fc_parity_jobs.txt` | `33b7bfa62e94a8d2dc4a5f8fbefdd2d1e7e342f8a4af9adc8f2df9a2bb84df0a` |
| `score_bridge.py` | `b3c67fe9a7ea8bf4721a3bedf562211c9dc6e5fbe6b2588f8e2e46a401db85d1` |
| `models.py` | `3ca2e8a5d3e173af7bdcf15bbae4faf1f2bf115dfe1d59d702f32e2ce9af3c92` |
| `suite_bridge_fc/key.json` | `964bb5869106c09856ed6ea58c1aa52ac17b8b5270b5ea25905dcd2b461815d6` |
| The 18 prompts of `suite_bridge_fc` | `31a0a00870eb89b6d8ffd616d68e9255441bfd629f78d6d10d609e496837e7d6` |

The prompt-set value is the SHA-256 of the output of `shasum -a 256 *.txt` run inside `suite_bridge_fc/prompts`.
