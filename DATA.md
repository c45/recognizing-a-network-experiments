# What is not stored here

- **The original post's released networks and prompt** (*Sparks of Weight-Space Programming*,
  `RikiyaT/lm_understand-logs`, commit `4d09ba6`, CC BY 4.0). Fetch them with
  `studies/layout-vs-behaviour/external/fetch_upstream.sh` (also in the other two study folders). Experiment 1's
  board pictures (figure 2) and the Exp 1 prompt template are built from them.
- **Trained networks of the depth benchmark** (`studies/depth-benchmark/ladder/**/nets*`, about 600 MB) and its
  transformer networks (about 54 MB). They can be regenerated deterministically on CPU with the training scripts;
  see [`studies/depth-benchmark/DATA.md`](studies/depth-benchmark/DATA.md). Every network a model actually read is
  stored anyway, as the weight text in each reader set's `prompts/` and as `analyst/models/<id>/blocks.npz` for the
  with-Python arm. The trained game policies of experiments 1–3 (`train/g_*/actor.npz`) are stored.
- **Sampled training states** of the game policies (`train/*/states.npy`, about 3 GB). Training is stochastic, so
  new samples will differ; `occupancy_never_on.npy`, the record of unused inputs that camouflage uses, is stored.
- **All frontier-model output**: no-tools replies, with-Python transcripts and code runs, call records, event
  streams, blind-coding packets and codes, and the per-network score files built from them. The prompts that
  produced them are stored, so running the readers produces new replies; the counts the post reports are in each
  experiment's README and in `post-figures/published_data.json`.
- **Other readers and arms.** The same suites were also given to other models. Pre-registrations are kept, so some
  mention those arms.
- **Material outside the post**: side studies of the same projects (level-1 camouflage, scrambled grids, new games,
  footprint fade, the depth pilot and width variants, policy-training pilots) and the papers written from them.
