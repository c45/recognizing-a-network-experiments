# Twins: Astra replies that match Opus's lower-effort reruns (pre-registration, November 10, 2026, 17:33 PST, before any call)

## Why

The rule for the post is matched effort in every Astra-against-Opus comparison (set
November 10). An audit of every counted reply (five independent readers, from run records)
found the post matched everywhere except the twin figure's no-tools rows: Opus's queue reruns a reply that hit the
128,000-token output cap one step lower (max, then xhigh, then high), and the rerun counts. Five counted or
soon-to-be-counted Opus replies are at xhigh today (forced choice: Dodge 9728b43c, 99e63c62, ac71abe0, Catch
c5cd72ce, Car-tag v2 0d500420), with up to nine more in rerun; Astra's replies on those networks are at max.

## What is run

- For every network and question of the current twin set where Opus's finished reply has `effort.txt` = xhigh, one
  GPT-6 Astra reply at xhigh on the same prompt file: `run_astra_xhigh.sh` (unchanged; `run_astra.sh` with the effort
  string xhigh), into `<suite>/toolfree_xhigh/astra_<id>/`. Where Opus's reply is at high, Astra's existing high
  reply (`<suite>/toolfree/astra_<id>`) is the match and nothing is run.
- The list is derived from the reply folders by `effort_match_jobs.py`, never by hand, and the queue
  (`run_queue_effort_match.sh`) is rerun as the Opus queue adds lower-effort replies, until Opus is complete.
- The second Codex configuration folder of `PYMAX_PREREG.md` (Amendment 1), `codex-cli 0.166.0-alpha.5.1`, one reply per cell, up to 3
  attempts, a completed reply always kept. A usage-limit event stops the queue.

## How the replies are used

- A **matched pair** is an Astra reply and an Opus reply on the same network and question at the same effort label.
  For a cell where Opus counts at xhigh, the matched Astra reply is the xhigh one; elsewhere it is the max one.
- Scoring is unchanged: forced choice by the `Answer:` letter against the key; open and hinted questions by the
  blind stance coding of `CODER_BRIEF_TWINS.md`, done in the coding run for the twin figure, pooled with its
  other answers.
- Reported for each affected cell: Astra at max, Astra at the matched effort, Opus. How the figure shows them
  (Astra's matched reply in those cells, or Opus counted at max only) is decided separately; both tables are given.
- No test. These are a handful of cells; the point is that no comparison in the figure mixes effort labels.

## Limits

Parity is by label: Opus reaches xhigh only after failing to finish at max, Astra is simply run at xhigh. Astra's
forced-choice answers have not depended on effort so far (first run: 9 of 9 at max, 9 of 9 and 8 of 9 at high).

## SHA-256

| File | SHA-256 |
|---|---|
| `effort_match_jobs.py` | `101cb94fafca050353735c1775e4d9a7b5ea9e0873cf45225270f876cd683a36` |
| `run_queue_effort_match.sh` | `b66d76f1cc96a9d345bb06ae7579afa9c4785ebe17f01f9b069add76997f5b0b` |
| `run_astra_xhigh.sh` | `5a03538daac06e3a4b75a27b69062508a9a37b4dbd7ee9f6f4fd8c8cf53040cb` |
- Watcher `watch_effort_match.sh` (SHA-256 `c512def460edea958ccc541c6464c4ce39666b6716516171294de320915f48e7`, started 17:55 PST): reruns the queue every 10 minutes when the list has a cell without an Astra xhigh reply, for up to 48 hours.
