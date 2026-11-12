# Frozen E/Q2/K2 pilot and checkpoint recovery, November 7, 2026

Registered before the following training runs. This is a feasibility pilot following the post hoc discovery that
the sign of (E[1]Q2).(E[1]K2) - (E[2]Q2).(E[2]K2) separates every eligible T2 and G2 network.

## Frozen pair

Use GPT-mini G2, with its existing shared data and training recipe. Initialize both networks with seed 1001.
Train one on EVEN-NEXT and the other on EVEN-PREV, with the embedding E and layer 2's query and key weight
matrices frozen at their common initialization. All remaining parameters train as usual, including query/key
biases and LayerNorm parameters. These three frozen arrays exactly determine the discovered shortcut.

Train both for 3,000 steps, batch 512, AdamW at 3e-3, zero weight decay, cosine schedule. Save initial state,
every 250 steps, and final state. A usable pair requires both final held-out accuracies >= 0.95 and bitwise
equality of E, Wq2 and Wk2 across all checkpoints and across the two networks. Report each accuracy and the
unchanged shortcut score. If either fails the accuracy bar, retain the failed pilot and repeat both from the same
initialization for 12,000 steps in a separate directory. Stop after that attempt and report the outcome.

The same-split data is reused to preserve the original G2 task. Its generating distribution is symmetric under
XOR 1, while the finite sampled dataset is not exactly closed under that permutation. Paired finite-sample
symmetry is a distinct future design. This pilot controls the three shortcut tensors directly.

Reader evaluation is pending. Training success alone establishes availability of the pair. Any later reader
experiment requires its own prompts, frozen key hash and scoring plan before calls.

## Intermediate checkpoint recovery

The earlier T2, G2 and R1 training processes have completed and their trainers saved final weights only. Recreate
G2 trajectories for seeds 0 and 1, both rules, using the saved data and original 3,000-step recipe. Save every 250
steps, including 0 and 3,000. Compare every final array against the original saved checkpoints. Label these as
replayed trajectories even when the final arrays match bit for bit. This checks that checkpoint writes preserve
the training trajectory and supplies intermediate networks for future behavioral-disagreement tests.

Each checkpoint holds exported float32 weights, native model state, optimizer and scheduler state, minibatch RNG
and Torch RNG state, metadata and hashes. Raw checkpoints remain local, with their manifest and verification
results retained with the source. Additional training requires a new output path.

## Completed pilot and replay deviation, November 7

Both frozen models reached 20,000/20,000 held-out accuracy at the first 3,000-step endpoint, with all three
controlled tensors bit-identical at every checkpoint. The 12,000-step fallback was unnecessary. Reader testing
is pending.

All four original-recipe reruns reached 100% held-out accuracy, but their arrays differ from the historical
saved networks. Exact original-array reproduction was therefore not established. An unmodified-trainer control
also differed from its historical final network. We retain the reruns as newly collected trajectories and record
maximum entry differences in verification.json and checkpoint_control.json. Short-run numerical differences are
also preserved in implementation_checks.json. A direct checkpoint-write check preserved model tensors and both
random-generator states exactly, and restoring the frozen model at step 250 reproduced step 500 bit for bit.

The registration hashes bind the source used at training launch. Subsequent changes to trainer module docstrings
correct the scope of the token-swap statement and document the new options. The exact launch source is saved in source_at_registration.zip. Each archived file matches its hash in registration.json.
