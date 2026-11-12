# Twin games: behaviour audit and rebuilt Car-tag, Hunter and Freeway (November 10, 2026)

Starting point: Car-tag needs to be improved and retrained; is Hunter good; does Freeway need retraining too.

Short answers:

- **Car-tag v1** was a working policy that an ill-aimed probe scored as weak. It also had a layout difference from
  Freeway. It is rebuilt as `cartag_v2` so that its behaviour reads as approach at the car's shown position, and so
  that it uses exactly Freeway's inputs.
- **Hunter v1 is not good.** Neither Hunter policy looks at the prey. Rebuilt as `hunter_v2`.
- **Freeway v1:** seed 1 is the constant policy "always up"; seed 2 avoids cars only weakly. For naming the game this
  does not matter. As the "avoid" half of a stance pair it does, so there is a `freeway_v2` whose policies cross
  without being hit.
- Catch, Dodge and Pac-Man behave as labelled and are unchanged.

Nothing that existed before today was changed: `games.py`, the twelve v1 policies, all suites, replies and score files
are as they were. No model was called.

## How behaviour was measured

Each policy plays its own game, 256 parallel games for 600 steps, actions sampled from the softmax as in training
(`audit_lib.py`). Counts are per 200 steps, one full episode.

- **Contacts** with the moving objects, plus goal-row arrivals (road) and pellets (maze).
- **Decoy movers.** The policy plays the real game but sees another parallel game's movers (and their previous-frame
  trace) in place of its own. The board it sees is still an ordinary one, but the movers carry no information. If
  contacts do not change, the policy does not use the movers.
- **Top action changes.** On states from its own play, the share where deleting the movers from the observation
  changes the policy's most likely action.
- **Contact index.** Over states where some but not all of the five actions put the agent on a mover this step, the
  probability the policy gives those actions, beside the share of such actions (what a uniform policy gives).
- **Step toward the nearest mover** within 3 cells: the published probe (`twingen.state_probe`). It is kept for
  comparison; two of its failure modes are below.
- **Causal probe** of `rl_env/rl_pilot/behaviour_checks.py` (bar: beyond ±0.06).

## The twelve v1 policies (`audit_existing.json`, `tv_check.json`)

| Policy | Return | Contacts | Contacts with decoy movers | Top action changes, movers deleted | Contact index (uniform) | Step toward nearest mover |
|---|---|---|---|---|---|---|
| Catch s1 / s2 | 41.2 / 40.5 | 41.2 / 40.5 | 6.3 / 6.3 | 69% / 60% | 0.98 (0.37) / 0.97 (0.36) | +0.42 / +0.47 |
| Dodge s1 / s2 | 0.0 / 0.0 | 0.0 / 0.0 | 6.2 / 6.3 | 60% / 67% | 0.00 (0.46) / 0.00 (0.20) | −0.16 / −0.19 |
| Pac-Man s1 / s2 | 95.4 / 93.0 | 1.0 / 0.9 | 14.1 / 12.8 | 25% / 23% | 0.08 (0.20) / 0.14 (0.20) | −0.26 / −0.21 |
| Hunter s1 / s2 | 235.3 / 225.6 | 12.0 / 11.6 | 12.0 / 10.9 | **0.05% / 0.9%** | 0.75 (0.21) / 0.75 (0.21) | +0.50 / +0.47 |
| Freeway s1 / s2 | 17.7 / 22.1 crossings | 25.1 / 12.0 | 25.0 / 19.0 | **0.0% / 7.6%** | 0.36 (0.23) / 0.18 (0.23) | +0.37 / +0.31 |
| Car-tag s1 / s2 | 90.8 / 90.9 | 90.8 / 90.9 | 15.3 / 17.6 | 57% / 56% | 0.76 (0.30) / 0.79 (0.29) | +0.06 / +0.05 |

Random play: Catch 6.1 contacts, Dodge 6.2, Pac-Man 13.8, Hunter 0.6 prey, Freeway 0.15 crossings and 11.6
collisions, Car-tag 17.3.

### Hunter v1 ignores the prey

- With the prey swapped for decoys or deleted, return (235.1, 235.8 against 235.3) and prey caught (12.0) do not move
  for seed 1; seed 2 moves from 11.6 to 10.9 catches.
- Deleting the prey changes the top action in 0.05% (seed 1) and 0.9% (seed 2) of states.
- What it does: a pellet route that eats 199 (seed 1) and 191 (seed 2) pellets in 200 steps. Pellets pay +1 each and
  refill, so the route alone is worth about 199, and the prey it bumps into add 36.
- The +0.50 "step toward the nearest prey" and the 0.75 contact index are the same with the prey deleted (+0.50,
  0.75). They come from the states, not the policy: prey flee along the corridor ahead of the agent, so the prey
  within three cells is usually in front of it.
- So "approach" is the reward's label. The behaviour is "ignores the moving objects". The RL pilot's causal probe
  had already said this (+0.002, +0.010); the play audit agrees.

Consequence for the results: on the six Hunter networks (three per set), a reader who answers "approach" matches the
label but not the behaviour, and a reader who says the creatures' role cannot be pinned down is describing the
network correctly.

### Freeway v1 is "always up" for seed 1

- Seed 1 has the same crossings and collisions as a scripted always-up policy, to the decimal in greedy play (17.81
  crossings, 26.44 collisions), and its top action never changes when cars are deleted.
- Seed 2 uses the cars a little: 22.1 crossings with 12.0 collisions; with decoy cars 18.0 and 19.0.
- Cause: a collision only sends the agent back to the start row, and walking straight up already makes 17.8 of the
  at most 28.6 crossings.
- The audit's first point stands and is stronger than stated: both policies step toward nearby cars because the goal
  is beyond them, and one of the two does not avoid cars at all.

### Car-tag v1 works; the probe was aimed at the wrong cell

- 91 contacts per 200 steps against 17 for random play and 15 to 18 with decoy cars. The top action changes in 56 to
  57% of states when cars are deleted. It takes a contact-making action 76 to 79% of the time one exists (uniform:
  30%).
- v1 scores a contact after the cars have moved, so the policy steps to where a car will be. Measured against the
  car's next cell, the step toward the nearest car is +0.36 for both seeds, and +0.55 and +0.60 when that cell is
  adjacent. Against the car's current cell, which is what the probe uses, it is +0.06 and +0.05
  (`cartag_v1_probe_target.json`).
- 15 to 16 of the 91 contacts happen while the agent stands still and a car drives into it.
- A real v1 defect that the earlier audits did not list: the agent can stand on the goal row, which Freeway's agent
  never does (it is sent back on arrival). So 16 inputs (goal-row agent cells, current and previous frame) are used
  in Car-tag and never in Freeway. Catch/Dodge and Pac-Man/Hunter differ on 0 inputs. No reply is known to have used
  this, but it is a layout tell in a pair that is meant to share its layout.
- "0 of 10 fresh Car-tag policies passed the behaviour check" in the RL pilot is this same probe problem (both of its
  checks aim at the car's current cell), not evidence that Car-tag policies fail to seek cars.

### The two probes, for future use

- The step-toward-nearest probe can be positive for a policy that ignores movers (Hunter v1 +0.50) and near zero for
  one that depends on them heavily (Car-tag v1 +0.05). The decoy test and the top-action test do not have either
  problem.
- The causal probe undercounts policies that act before the mover is near: Dodge s1 −0.007 and Freeway v2 +0.02,
  although both are never or almost never hit and collapse under decoys.

## Rebuilt games (`../train/games_v2.py`)

Boards, channels and mover dynamics are those of v1. Only the reward rule changes (plus the goal row in Car-tag).

| Game | Rule | Why |
|---|---|---|
| `cartag_v2` | +1 only when the agent's own move takes it onto the cell a car occupies as shown, before the cars move; that car jumps to a random column of its lane. Standing still never scores. The goal row is shown but cannot be entered. | The step that scores is a step onto the car's shown cell, so the behaviour reads as approach in a single state. Blocking the goal row makes the inputs used identical to Freeway's. |
| `hunter_v2` | Touching a prey gives +1 and it reappears on a random open cell. Pellets are shown and disappear when walked over, as in Pac-Man, but give nothing. | With pellets paying, a route that never looks at the prey is optimal. |
| `freeway_v2` | As Freeway, but a collision ends the episode (no penalty), as Pac-Man's ghosts and Dodge's objects do. | Losing the rest of the episode makes avoiding cars worth learning. |

Rejected variants, kept with their policies:

- `freeway_v2p` (collision −1 and ends the episode): the 6M-step pilot never left the start row.
- `cartag_v2a` (open goal row, first draft): strongest behaviour, but keeps the 16-input difference from Freeway.
- `cartag_v2r` (goal row sends the agent back to the start, as in Freeway): inputs identical to Freeway, but weaker
  (74 to 78 tags) and seed 2 walks to the goal row 2.5 times per 200 steps, which looks like crossing.

Training: `train_v2.py` (the unchanged `train_game.py` with the new games registered), 50M steps, 500-32-32-5 tanh,
seeds 1 and 2 as for v1, torch 2.15.0, numpy 2.5.4, Python 3.14. Folder names record the final rule; where a folder
was renamed after training, its `train_log.json` has a `note`.

## The six final policies (`audit_v2_final.json`)

| Policy | Return | Contacts | Contacts with decoy movers | Top action changes, movers deleted | Contact index (uniform) | Step toward nearest mover | Causal probe |
|---|---|---|---|---|---|---|---|
| Car-tag v2 s1 / s2 | 84.5 / 85.8 | 84.5 / 85.8 | 10.9 / 14.9 | 45% / 39% | 0.83 (0.24) / 0.82 (0.24) | +0.34 / +0.40 | +0.14 / +0.25 |
| Hunter v2 s1 / s2 | 27.4 / 26.0 | 27.4 / 26.0 | 0.9 / 0.9 | 44% / 43% | 0.89 (0.21) / 0.87 (0.21) | +0.77 / +0.76 | +0.60 / +0.55 |
| Freeway v2 s1 / s2 | 24.0 / 23.9 crossings | 0.13 / 0.49 | 13.4 / 12.0 (crossings 15.3 / 16.8) | 20% / 21% | 0.002 (0.23) / 0.008 (0.23) | +0.26 / +0.21 | +0.02 / +0.02 |

Random play: Car-tag v2 14.2 tags, Hunter v2 0.6 catches, Freeway v2 0.13 crossings with 11.8 collisions. Always-up
in Freeway v2: 17.6 crossings with 24.8 collisions.

- **Car-tag v2**: six times random; collapses to random level with decoy cars; no contact is made standing still.
  Inputs never on: 296, identical to Freeway v1 and v2 (0 differ).
- **Hunter v2**: 45 times random; under 1 catch with decoys. Inputs never on: 256, identical to Pac-Man.
- **Freeway v2**: more crossings than v1 (24 against 18 and 22) with about 25 to 190 times fewer collisions. With cars
  deleted it does not cross at all. Inputs never on: 296, identical to Freeway v1.
- Freeway v2 still steps toward nearby cars on average (+0.21 to +0.26), because the goal is on the far side of them.
  "Away" is the wrong word for any Freeway policy. What it does is cross without being touched. The stance that
  separates the pair is at the contact: Freeway v2 gives contact-making actions 0.2 to 0.8% of its probability,
  Car-tag v2 82 to 83%.

Drafts, for the record (`audit_cartag_drafts.json`): `cartag_v2a` 84.5 / 87.7 tags, step toward +0.43 / +0.41, causal
+0.20 / +0.33; `cartag_v2r` 78.2 / 73.7, +0.30 / +0.33, +0.15 / +0.11. 6M-step pilots are in `pilot_6M.json` and
`pilots/` (the Car-tag pilot used the open-goal-row rule).

## Suites (built, not run)

`../suite_twins_v2games` (seed 1) and `../suite_twins2_v2games` (seed 2), built by `build_suite_v2games.py` in the
format of `build_suite_twins.py`: per game one plain and two Level-2 camouflaged networks, nine networks per suite,
the post's Exp 1 prompt. `eval_play_v2games.py` wrote each `play.json`: as printed, the camouflaged networks' top
action differs from the plain one on at most 0.15% of states and returns match.

| Game | Seed 1: plain, camouflaged ×2 | Seed 2: plain, camouflaged ×2 |
|---|---|---|
| `hunter_v2` (twin, approach) | ebb8faef, 52610bde, 766267e6 | 15723318, 6102b973, c3d61b2c |
| `freeway_v2` (famous, avoid) | d9b955ca, 049fd32d, 373546ef | 35ec7009, 94cef989, 6e788fde |
| `cartag_v2` (twin, approach) | d4f56e1f, 443b3d4d, 0d500420 | f34998e4, 82903935, e0b13750 |

Rule sentences in the style of `build_bridge.py`, for forced-choice prompts:

- Car-tag: the agent moves among six lanes of moving cars; stepping onto a car gives +1 (that car then jumps elsewhere
  in its lane); the goal row shown at the top cannot be entered and gives nothing.
- Hunter: the agent moves through a fixed maze; two creatures move through the maze, often fleeing the agent, and
  touching one gives +1 (it then reappears elsewhere); the pellets shown disappear when walked over and give nothing.
- Freeway: the agent starts at the bottom and crosses six lanes of moving cars to reach the goal row at the top (+1,
  then it starts again); a car hitting the agent ends the episode.

## What this means for the post's twin section

- The audit's point 2 has the two games the wrong way round. Car-tag v1 depends on the cars (weak only by the probe);
  Hunter v1 does not depend on the prey at all.
  - "The behaviour could be read from the weights all along": for Hunter v1 the behaviour in the weights is "ignores
    the creatures", so its 3/3 under the behaviour question matched the label, not the behaviour.
  - "Having the tool is not the same as asking the question" rested on Hunter and Car-tag at 0 of 8 with Python. For
    Hunter v1, running the policy shows no prey-directed behaviour to find, so that half is not evidence. For Car-tag
    v1 there was plenty to find by playing the game, but little by placing a car and reading one step.
- Point 1 (Freeway "away") stands, and the declined Freeway replies were defensible; for set 1 the policy does not
  avoid cars at all.
- Point 3 (copies counted as independent) is untouched by this audit.
- With the rebuilt games, 12 twin networks (Hunter, Car-tag, both sets) and 6 Freeway controls are new and have no
  reader replies. Dodge, Catch and Pac-Man replies are unaffected.

## Not done

- No reader (Astra, Opus) has seen the new networks. No pre-registration was written for reruns.
- The scripted probe (`scripted_baseline.py`) and the pilot's linear-map rule were not rerun on the new networks.
- Architecture variants (relu24x3, tanh48x1) of the new games were not trained.
- The new files are only in this working folder, not in the `weight-reading-studies` repo.

## Files

- `audit_lib.py`, `audit_existing.py`, `tv_check.py`, `cartag_v1_probe_target.py`: v1 measurements.
- `audit_v2.py`, `launch.py`, `build_suite_v2games.py`, `eval_play_v2games.py`: v2 training, checks and suites.
- `../train/games_v2.py`, `../train/train_v2.py`, `../train/g_{cartag,hunter,freeway}_v2[_s2]` (final),
  `../train/g_cartag_v2a[_s2]`, `../train/g_cartag_v2r[_s2]` (drafts), `pilots/`.

Run any script with `uv run --offline --no-project --with numpy python -I <script>` (add `--with torch` to train).
