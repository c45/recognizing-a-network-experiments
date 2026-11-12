# Camouflaged games: tool-free readers vs analysts (pre-registration, November 5, 2026, before any call)

## Networks

21 blinded prompts (`key.json` holds the truth; models never see it):
- **plain:** 4 trained networks (Breakout, Pong, Pac-Man, Freeway) from our reconstructions of the games.
- **camouflaged:** 16 networks, 4 per game. Pong is flipped left-right and Breakout and Freeway top-bottom; Pac-Man,
  whose maze is symmetric, gets 4 seeds instead.
  - Camouflage is exact on reachable states: [../camouflage.py](../camouflage.py) applies null-space noise along
    conserved input combinations, a fill of never-used rows by perturbed nearest copies, a hidden-unit shuffle with sign
    flips, and an output shift.
  - Max probability change on held-out states is ≤ 1e-9; as printed (3 decimals), ≤ 4%.
- **untrained:** 1 network, which shows each model's default guess.

## Known residual clues

Within a channel, used and unused rows are at chance for a row-value detector (AUC 0.49–0.55). But which channels are
busy inside the court remains detectable (AUC 0.66–0.98), and the button grouping (which actions behave alike) is
unchanged. These clues are available to both arms below.

## Arms

- **Tool-free reader:** the post's exact Exp 1 prompt, one reply.
  - GPT-6 Astra via `codex exec`, effort high, tools off (as in the post).
  - Claude Opus 5.5 via `claude -p`, effort max, tools off.
  - All 21 networks.
- **Analyst:** the same models and efforts. The weights are not printed; they are loaded in a Python sandbox, as the
  exact printed values (3 decimals).
  - The analyst may do any analysis, including running the network on inputs it constructs. One code block per turn;
    at most 30 runs.
  - Same answer format.
  - 12 networks: the 4 plain ones, plus camouflaged seed 0 of Pong, Breakout and Freeway in both orientations and
    Pac-Man seeds 0 and 1.

## Scoring (fixed now)

- **Game:** the `Game:` line mapped to breakout / pong / pacman / freeway by keywords:
  - brick, Breakout, Arkanoid → breakout;
  - Pong, paddle with up/down → pong;
  - Pac-Man, maze → pacman;
  - Freeway, Frogger, road/traffic crossing → freeway;
  - anything else → other.
- **Orientation:** read from the whole answer by hand, blind to the key:
  - Pong: paddle left or right;
  - Breakout: paddle bottom or top;
  - Freeway: goal top or bottom.
  - Scored correct, wrong, or not stated.
- **Primary comparison:** game accuracy and orientation accuracy on the camouflaged networks, tool-free vs analyst,
  per model.
- **Evidence of reading beyond explicit analysis:** the tool-free reader beats the analyst of the same model on the
  camouflaged networks by at least 2 networks out of 8 on game identity, or by at least 2 on orientation. Anything
  smaller is a tie at this sample size.

---

## v2 addendum (November 4, 2026, 21:55, before any v2 call)

This suite uses [../camouflage_v2.py](../camouflage_v2.py) instead of camouflage.py, after round 1 showed that the
null-space noise was itself readable:
- Astra decoded its sign alternation as a two-cell paddle.
- The analyst tested layouts by checking which states cancel the noise.

**Changes:**
- No null-space noise; used rows keep their trained values.
- Never-used rows are exact copies of the mirror-image cell's row when that cell is used, otherwise of a random used
  row in the same channel.
- Hidden-unit shuffle, sign flips and the output shift are kept.

**Exactness:** max probability change on held-out states 1.7e-14; as printed, 0.24%.

**Legibility (strongest reader correlation with the true layout):** Pong 0.17–0.21, Breakout 0.15–0.25,
Freeway 0.28, Pac-Man 0.50–0.70.

**Unchanged:** arms, models, efforts, network selection rule, scoring and the decision rule.

**Analyst networks** (`analyst/analyst_networks_v2.txt`): the 4 plain networks, plus camouflaged seed 0 of Pong,
Breakout and Freeway in both orientations, and Pac-Man seeds 0 and 1.
