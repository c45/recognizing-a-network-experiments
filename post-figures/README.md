# Post figures

`published_data.json` is the `data.json` the post serves (the data behind all eight figures), with the post's text
in `post_index.md`, both at the revision in `published_rev.txt`.

The scripts here build that file. Only the first three run from what is stored; the rest need scored model
replies, which are not included (run the readers in `../studies` to produce them).

| Script | Builds | Inputs |
|---|---|---|
| `make_play.py`, `make_live_play.py <post folder>` | `play` (figure 1) and the in-browser policies | trained policies in `studies/layout-vs-behaviour/experiments/train/` |
| `make_pictures.py` | `pictures.json` (figure 2) | the original post's released networks (`studies/layout-vs-behaviour/external/fetch_upstream.sh`) |
| `make_disguises.py` | `disguises` pictures and layer shapes (figure 3) | the rewrite suites; scores from `rewrites` |
| `add_opus_r1c.py` | Opus answers and totals in `r1c` (mirror test, figure 7) | `opus_r1c.json` (extracted from the Opus replies, not included) |
| `make_rewrites.py` | `rewrites`, `rewrites_baseline` (figure 3) | rewrite replies, via `score_max_parity.py` |
| `make_blog_data.py` → `make_website_data.py` | `naming`, `depth`, `replies`, `transformers`, `r1b`, `r1c` | score files of experiments 1, 4 and 5, the depth analysis outputs, reply coding and prompt sizes |
| `studies/layout-vs-behaviour/experiments/make_ladder_v2.py` | `ladder`, `twins` (figure 4) | twin score files |
| `make_prompt_tokens.py` | prompt sizes per depth | the readers' raw event streams |
| `build_post_data.py` | all of the above, compared block by block with `published_data.json` | everything above |

`pictures.json` and `play.json` are stored outputs of the first two rows.
