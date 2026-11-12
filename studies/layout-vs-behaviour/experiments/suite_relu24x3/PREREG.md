# Pre-registration: architecture replication, relu24x3 (written 2026-11-05 00:40 EST, before any model call)

Suite: 4 games x (1 plain + 4 Level-2 camouflaged) + 1 untrained = 21 networks of architecture relu24x3, trained exactly
as the tanh32x2 networks in suite_v2 (PPO, 50M steps, seed 1) except for the architecture. Camouflage:
camouflage_arch.py (= camouflage_v2 procedure; ReLU hidden units get positive rescaling instead of sign flips).
Prompts: the post's Exp 1 prompt with the network section rewritten for the architecture.

Arms (unchanged from suite_v2): tool-free Astra (gpt-6-astra, effort high, codex exec, tools off) and Opus 5.5
(claude -p, effort max, tools off), one reply per network; analysts (Python sandbox, 30 runs) for both models on the
4 networks in analyst.txt (one camouflaged network per game, flipped where the game has a flip).

Scoring (unchanged): score_suite.py keyword mapping on the "Game:" line; sides hand-coded blind from
orientation_review.md.

Questions:
1. Does Opus tool-free > Astra tool-free on camouflaged networks replicate? Test: exact McNemar on the 16 camouflaged
   networks of this architecture, and pooled over both new architectures (32).
2. Are analysts >= tool-free for both models (no reading beyond explicit analysis)?
3. Does accuracy change with architecture (compare with suite_v2's 16/16 vs 7/16)?
