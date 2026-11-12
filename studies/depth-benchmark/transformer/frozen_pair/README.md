# Frozen-tensor pair and saved trajectories

The November 7 pilot trained a G2 pair with opposite parity rules and identical frozen E, Wq2 and Wk2 matrices.
Both networks achieved 20,000/20,000 correct on the saved held-out set. They made different predictions on all
20,000 sequences. Their two-token shortcut score was identical at +8.259646, so the shortcut selects EVEN-NEXT
for both. The three frozen arrays match bit for bit across the two networks and all 13 snapshots per network.
Reader evaluation on this pair is pending.

Four additional G2 runs reproduce the original training recipe for seeds 0 and 1 under each rule. All four reach
100% held-out accuracy. Their final weights differ from the historical saved weights by maximum absolute entry
errors of 0.00777, 0.00931, 0.03057 and 0.01449, depending on the run. An unmodified-trainer control also differs
from its historical checkpoint. These are fresh trajectories collected under the original recipe. The exact cause
of cross-run numerical differences remains unresolved.

There are 78 snapshots across the six runs. Each snapshot contains `weights.npz`, `training.pt` and
`metadata.json`. The native state includes optimizer/scheduler state, minibatch RNG, Torch RNG and the frozen
parameter names. Checkpoint publication uses a temporary directory and atomic rename. An existing step directory
causes an error before publication. A verified restoration of the frozen EVEN-NEXT model from step 250 reproduced
step 500 bit for bit.

## Files

- `PREREG.md` and `registration.json` record the prospective training design and input hashes.
- `source_at_registration.zip` retains the exact registered source before explanatory docstring corrections.
- `nets_3000/` holds the two pilot networks and their `.checkpoints/` directories.
- `replay/` holds the four original-recipe reruns and their `.checkpoints/` directories.
- `verification.json` gives independent full-set evaluation, fixed-tensor checks and diagnostic trajectory accuracy.
- `checkpoint_manifest.json` binds the saved arrays and native state by SHA-256.
- `checkpoint_control.json` records the separate unmodified-trainer comparison.
- `implementation_checks.json` records small-run checks for all three trainers and checkpoint-write invariants.

The training arrays and native state are local artifacts excluded from git. Source, metadata and verification
summaries are versioned. The historical network banks remain available through the existing data-release route.

## Reproduction

Run from the repository root, with the pinned research dependencies installed. Use a new destination for each run.
For the frozen pair, run each rule with the same seed and add both options.

```bash
mkdir -p transformer/frozen_pair/nets_3000
python -B transformer/gpt_family.py train transformer/data/seqs.npz evennext 1001 transformer/frozen_pair/nets_3000/evennext_s1001.npz 3000 2 --checkpoint-every 250 --freeze-eqk
python -B transformer/gpt_family.py train transformer/data/seqs.npz evenprev 1001 transformer/frozen_pair/nets_3000/evenprev_s1001.npz 3000 2 --checkpoint-every 250 --freeze-eqk
```

For original-recipe trajectories, use seeds 0 and 1, both rules, a destination under `replay/` and
`--checkpoint-every 250`. Omit `--freeze-eqk`. After both collections are present, run the verifier.

```bash
python -B transformer/frozen_pair/verify.py --include-replays
```

The verifier evaluates all final networks directly from exported arrays and evaluates a fixed 2,048-sequence
subset at intermediate steps. Its tensor-equality and snapshot-hash checks are exhaustive. The test inputs are
used for offline reporting after training. Training uses the original training split and fixed step schedule.
