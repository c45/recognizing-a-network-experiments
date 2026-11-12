# Pre-registration: twin games (written 2026-11-05 13:42 EST, before any model call on this suite)

**Question.** Do the readers identify what the policy DOES, or fall back on the famous game a layout suggests?
Three twin pairs share layout, channels and dynamics and differ only in the stance toward the moving objects:

| Pair | Known game (stance) | Twin (stance) |
|---|---|---|
| Falling objects, agent on row 8 | Catch: +1 per object caught (approach) | Dodge: -1 and episode ends on contact (avoid) |
| Pac-Man maze, pellets, 2 moving objects | Pac-Man: ghosts chase, contact -1 and ends (avoid) | Hunter: objects flee, contact +3 (approach) |
| Freeway road, cars, goal row shown | Freeway: cross to goal row, cars reset you (avoid) | Car-tag: touching a car +1, goal row gives nothing (approach) |

The famous name predicts approach / avoid / avoid whatever the truth, so a prior-only reader scores 3 of 6 games
(and 50% of networks). Games: train/games.py (Catch, Dodge, Hunter, CarTag new). Architecture: 500-32-32-5 tanh
(the post's). Pac-Man and Freeway reuse train/g_pacman, g_freeway; the four new games are trained the same way
(PPO, 50M steps, seed 1).

**Suite.** Per game: 1 plain + 2 Level-2 camouflaged networks (camouflage_arch.py, seeds 10 and 11, no flips, unused
inputs from occupancy_never_on.npy), 18 networks, random ids, the post's Exp 1 prompt unchanged. Exactness checked
in play (eval_play.py) before any model call.

**Arms.** Tool-free Astra (high) and Opus 5.5 (max), one reply per network. Analysts (Python, 30 runs), both models,
on camouflaged seed 10 of Dodge, Hunter, Car-tag.

**Scoring (primary).** Blind-coded from the full answer, model names hidden: the stance the answer attributes to the
agent toward the moving objects (channel 1): approach/catch/collect/chase, avoid/dodge/evade, or unclear/none.
Correct = matches the truth. Secondary: the game is right as a whole: correct stance AND the right setting
(falling objects with an agent on the bottom row / maze / road with lanes), AND for the twins the answer does not
assert the opposite reward (e.g. "Pac-Man: avoid ghosts").

**Tests.**
1. Per model: stance accuracy on the 9 twin-game networks (Dodge, Hunter, Car-tag) vs the 9 known-game networks.
   A reader that relies on the famous game is at ceiling on known games and near 0 on twins.
2. Opus vs Astra, paired exact McNemar on stance over all 18 networks and over the 12 camouflaged ones.
3. Analysts vs tool-free on the same networks (no reading beyond analysis if analysts are never worse).
