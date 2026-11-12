# Transformer family: reading attention-circuit composition from weights

Pre-registration with amendments and key hash: [PREREG.md](PREREG.md).

## Task

Sequences of 10 tokens from a vocabulary of 16. Positions 0–8 hold distinct tokens; the query at position 9 repeats
the token at one position j (1–7). The networks are 2-layer, attention-only transformers (one head per layer,
d_model 32, d_head 16, no MLPs, no LayerNorm, 5,184 parameters), trained by imitation on one shared set of
sequences.

## T1 (confounded, not used for readers)

- **Classes:** NEXT (output x[j+1]) vs PREV (output x[j−1]).
- **Baselines:** order-free weight statistics classify 78/80 networks, and single-layer circuit summaries 79/80.
- **Why:** the two rules need different mechanisms, which leave different spectra. Layer 2 matches the query against
  layer 1's previous-token output for NEXT (K-composition), and against the raw token for PREV.

## T2: an interaction rule with a local QK cue

- **Classes:** EVEN-NEXT (NEXT for even query tokens, PREV for odd ones) vs EVEN-PREV (the reverse).
- **Symmetry:** swapping each token with its partner (v ↔ v XOR 1) maps one class onto the other exactly. The generating distribution has this symmetry. The finite sampled dataset has incomplete
  swap-pair coverage, as documented in Amendment 3.
- **Registered baselines (79 networks):** each interval includes chance. Token footprint 47/79, order-free statistics 45/79, position footprint
  39/79, single-layer circuits 40/79.
- **Reference script:** a fixed weight-only K-composition contrast restricted to even query tokens classifies 79/79.
- **Readers:** a blinded set of 12 (`readers_t2/`), tool-free GPT-6 Astra and Claude Opus 5.5 at effort xhigh.

## Files

| File | Purpose |
|---|---|
| `tf_family.py` | Shared sequences (`data`) and training (`train`, rules `next`, `prev`, `evennext`, `evenprev`) |
| `baselines_tf.py` | Pre-registered baselines B1–B4 and the reference script |
| `make_prompts_tf.py` | Blinded tool-free reader set, key and key hash |
| `score_tf.py` | Scores replies: correct, p-value, partial credit |
| `baselines_t1.json`, `baselines_t2.json` | Baseline results |
| `readers_t2/` | Prompts, key, key hash, replies |

Trained networks (`nets/`, `nets_t2/`, `trial*/`) are excluded from git. Regenerate them with `tf_family.py`; each
network takes about 13 s on a CPU.

## November 7 additions

The post hoc two-token QK contrast classifies all 79 eligible T2 and all 80 eligible G2 networks. The paper
includes a four-number Astra example and a new frozen-tensor G2 pair. See [the pilot](frozen_pair/README.md).

To save initialization, every 250 steps and the final state in a new run, use a new output path and append
`--checkpoint-every 250` to either `tf_family.py train ...` or `gpt_family.py train ...`. R1 supports the same
option through `realistic/r1_family.py train ...`. Each step directory holds exported weights, restart state
and hashed metadata. `--freeze-eqk` on the G2 trainer freezes E and the layer-2 query/key weight matrices.
