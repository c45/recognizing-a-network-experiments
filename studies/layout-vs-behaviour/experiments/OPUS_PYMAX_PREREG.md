# Twins with Python: Claude Opus 5.5 (pre-registration, November 10, 2026, 17:25 PST, before any call of this arm)

**Why.** Rows 5 and 6 of the post's twin figure ("Open question, with Python", "Behaviour question, with Python") have
GPT-6 Astra at max on the rebuilt twin set (`PYMAX_PREREG.md`, `PYMAX_RESULTS.md`) and no Opus cells. On Nov 10 the
same two rows were queued for Opus 5.5.

**Networks.** The 24 camouflaged networks of `pymax_jobs.txt`, unchanged and in its order: 12 twins (Dodge in
`suite_twins` / `suite_twins2`; Hunter v2 and Car-tag v2 in `suite_twins_v2games` / `suite_twins2_v2games`), then 12
known games (Catch, Pac-Man, Freeway v2). One conversation per network and question, 48 in all. One open-question
conversation already exists from round 6 (`suite_twins/analyst/opus_99e63c62`) and is kept, so 47 are new.

**Arms.**
- Open question: `analyst/analyst_loop.py opus <id> ../<suite>` → `<suite>/analyst/opus_<id>/` (the script and folder
  of every earlier Opus analyst).
- Behaviour question: `analyst/analyst_loop_stance_max.py opus <id> ../<suite>` →
  `<suite>/analyst_stance_max/opus_<id>/`, beside Astra's folders. Neither `*_max.py` file is changed.
- Both: `claude -p` / `--resume` with `claude-opus-5-5` on Claude Code 2.1.314 (`cli-2.1.314/claude`), effort max,
  all tools off, the Python sandbox of the earlier analysts, at most 30 code runs. Running the network on inputs the
  model builds is allowed, as before.

**Queue.** After the no-tools replies on the rebuilt games; twins before known games (the file's order). Job kinds
`an` (open) and `as` (behaviour) in `job_arch.sh`. A conversation interrupted by a usage limit is started again from
the beginning.

**Scoring.** Final answers are coded for stance (approach / avoid / unclear), setting, asserts_opposite and named game
by a separate coder model (Claude Sonnet) with `CODER_BRIEF_TWINS.md` verbatim, blind to model, question, game version and key; Opus and
any other model's answers awaiting coding go in one packet set. Stance correct = coded stance matches `key.json`.
Reported per row as twins (of 12) and known games (of 12), with Freeway v2 shown separately, and beside Astra's row.
Conversations that report an analysis without running any code are flagged with `analyst_grounding.py` and reported.
Missing conversations count as wrong only in a final table; interim counts are labelled interim.
