# Data not stored in git

## Trained networks (`ladder/**/nets*/`), about 600 MB unpacked

Every network the study trained: one `.npz` (weights) and one `.json` (rule, seed, depth, width, epochs, held-out
accuracy) per network. They are not included. To rebuild them (deterministic per seed on CPU with the pinned
versions in `requirements.txt`):

- interaction family: `ladder/l2/make_states_l2c.py`, then `ladder/l2/train_l2.py`;
- depth scaling: `ladder/scale/scale.py train ...` and `ladder/scale/train_variants.py`, for example
  `SEEDS=180 EPOCHS=10 python -I ladder/scale/train_variants.py ladder/l2/data_d/states.npz d14:14:32`.

`twopager_analysis.py` checks one network's rebuilt initialization bit for bit against the saved weights, and
`NETWORKS_SHA256SUMS.txt` lists the checksum of every original network file. Networks below 0.95 held-out accuracy
were kept; the analysis scripts exclude them using `heldout_acc`.

The networks that models actually read are also in git: as weight text inside each reader set's `prompts/`, and as
`analyst/models/<id>/blocks.npz` for the with-Python arm.

## Released networks from the original post (`external/`)

The footprint comparison in Figure 1c uses the "which game" networks released with *Sparks of Weight-Space
Programming* (`RikiyaT/lm_understand-logs`, commit `4d09ba6`). They are third-party material and are not
redistributed here; `scripts/fetch_upstream.sh` fetches them.

## Model output

Model replies, with-Python transcripts, call records and event streams are not included.

## Transformer networks (`transformer/nets*`, `realistic/nets*`), about 54 MB

The T1, T2, G2, R1 and R1b networks are not included. Regenerate them with
`transformer/tf_family.py` (T1, T2), `transformer/gpt_family.py` (G2), and `realistic/r1_family.py` followed by
`realistic/mirror_r1.py` (R1, R1b). The networks that models read are in git as weight text in each reader set's
`prompts/`. `paper_icml/review_analysis.py` needs `transformer/nets_t2`, `transformer/nets_g2` and
`realistic/nets_r1b`.

## November 7 checkpoints

The new training pilot and original-recipe reruns are described in
[the frozen-pair collection](transformer/frozen_pair/README.md). They contain 78 snapshots from six G2 runs.
Each step includes exported weights, native optimizer/model/random state and hashed metadata. The arrays are
excluded from git. `transformer/frozen_pair/checkpoint_manifest.json` records their hashes. These reruns are
separately collected trajectories, and their weight differences from the original final networks are reported.
