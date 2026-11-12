# Pre-registration: hint control on the twin networks (written 2026-11-09 17:34 PST, before any model call on these prompts)

**Question.** On twin games, readers asked "which game?" name the famous game (Astra 0/9 twins, stance) but pick the
right game when both are named (Astra 8/9). Is the open-question failure a failure to *check* behaviour (a hint
fixes it) or a failure to *read* it without named alternatives (a hint does not)?

**Conditions** (`build_hint.py`; the 18 `suite_twins` networks, weight text and answer format unchanged; tool-free,
one reply per network, Astra at high effort via `run_astra.sh`, as in the open twin runs):

- **hint1** (`suite_twins_hint1`, primary). One paragraph added after "Which game was it trained on?": the game may
  be a well-known game or a lesser-known variant with the same layout in which the agent does something different;
  both kinds are in the set; do not decide from the layout alone. No game, alternative or behaviour is named.
- **hint2** (`suite_twins_hint2`, secondary). hint1 plus "work out from the weights whether the agent moves toward
  the moving objects (channel 1) or away from them" and a `Stance:` line in the answer format.

**Scoring.** Identical to the open twin runs: a separate coder model (Claude Sonnet), blind to condition, model and key, codes each answer
(stance approach / avoid / unclear toward channel-1 objects; setting; asserts_opposite; named game), and
`score_twins.py` scores it against `key.json`. Primary outcome: stance correct on the 9 twin networks, compared with
the open question on the same networks (exact McNemar). Reported too: known-game stance (9 networks; a hint could make
readers over-call variants), game correct, and camouflaged vs plain. A missing or unparseable reply counts as wrong.

**Caveat fixed in advance.** Freeway (known game) is labelled "avoid" from its objective, but its policies step toward
nearby cars on net (RESULTS.md round 9). A reader that computes the stance may answer "approach" for Freeway and be
scored wrong; Freeway is therefore also reported separately. No twin label has this problem (Car-tag is weakly
approach, Dodge avoid, Hunter approach in own play).

**Predictions.** If the failure is not checking: hint1 twins well above 0/9 (toward the forced choice's 8/9), known
games stay high. If hint1 stays low and hint2 is high: an explicit behavioural question is needed, alternatives are
not. If both stay low: naming the alternatives is what makes the forced choice work, and "asked for" in the paper's
claim must be narrowed to "asked as a choice between named games".

**Not in this run.** Opus 5.5; repeat samples.
