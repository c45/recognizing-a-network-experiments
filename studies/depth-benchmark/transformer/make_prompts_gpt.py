"""Blinded tool-free reader set for the GPT-mini family G2 (EVEN-NEXT vs EVEN-PREV).

Usage: python3 -I make_prompts_gpt.py <nets dir> <out dir> [seed]
Same selection, key format and hashing as make_prompts_tf.py; the network description and weight blocks follow
gpt_family.py.
"""
import hashlib
import json
import secrets
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_prompts_tf import QUESTION, TASK  # noqa: E402

NETWORK = """THE NETWORK
A GPT-style transformer with 2 layers, model width 32, 4 attention heads of width 8 per layer, an MLP of width 128 per layer, LayerNorm before each sublayer and at the end, and biases. For a sequence x[0..9]:
  h[t] = E[x[t]] + P[t]                         E: 16 x 32 (row i = token i), P: 10 x 32 (row t = position t)
  LN(v; g, b) = g * (v - mean(v)) / sqrt(var(v) + 1e-5) + b      (mean and variance over the 32 entries of v)
  for layer l = 1, 2:
    u = LN(h[t]; ln1_g_l, ln1_b_l) at every position t
    q = u @ Wq_l + bq_l,  k = u @ Wk_l + bk_l,  v = u @ Wv_l + bv_l          Wq_l, Wk_l, Wv_l: 32 x 32
    head n (n = 0..3) uses entries 8n..8n+7 of q, k and v:
      a_n[t, s] = softmax over s <= t of (q_n[t] . k_n[s]) / sqrt(8)       (causal: s <= t)
      z_n[t] = sum over s of a_n[t, s] v_n[s]
    h[t] = h[t] + concat(z_0[t], z_1[t], z_2[t], z_3[t]) @ Wo_l + bo_l        Wo_l: 32 x 32
    m = LN(h[t]; ln2_g_l, ln2_b_l)
    h[t] = h[t] + GELU(m @ W1_l + b1_l) @ W2_l + b2_l                         W1_l: 32 x 128, W2_l: 128 x 32
  logits = LN(h[9]; lnf_g, lnf_b) @ U + c                                     U: 32 x 16 (column i = token i), c: 16
GELU(z) = z * Phi(z), with Phi the standard normal CDF. 26,832 parameters in total."""


def blocks(L=2):
    out = [("E", "E (token embedding), shape 16 x 32: one line per row (row i = token i)"),
           ("P", "P (position embedding), shape 10 x 32: one line per row (row t = position t)")]
    for l in range(1, L + 1):
        out += [(f"ln1_g{l}", f"ln1_g_{l} (layer {l} attention LayerNorm gain), 32 values"),
                (f"ln1_b{l}", f"ln1_b_{l} (layer {l} attention LayerNorm bias), 32 values")]
        for m, name in (("q", "query"), ("k", "key"), ("v", "value")):
            out += [(f"W{m}{l}", f"W{m}_{l} (layer {l} {name} weights), shape 32 x 32: one line per row"),
                    (f"b{m}{l}", f"b{m}_{l} (layer {l} {name} bias), 32 values")]
        out += [(f"Wo{l}", f"Wo_{l} (layer {l} attention output weights), shape 32 x 32: one line per row"),
                (f"bo{l}", f"bo_{l} (layer {l} attention output bias), 32 values"),
                (f"ln2_g{l}", f"ln2_g_{l} (layer {l} MLP LayerNorm gain), 32 values"),
                (f"ln2_b{l}", f"ln2_b_{l} (layer {l} MLP LayerNorm bias), 32 values"),
                (f"W1_{l}", f"W1_{l} (layer {l} MLP input weights), shape 32 x 128: one line per row"),
                (f"b1_{l}", f"b1_{l} (layer {l} MLP input bias), 128 values"),
                (f"W2_{l}", f"W2_{l} (layer {l} MLP output weights), shape 128 x 32: one line per row"),
                (f"b2_{l}", f"b2_{l} (layer {l} MLP output bias), 32 values")]
    return out + [("lnf_g", "lnf_g (final LayerNorm gain), 32 values"), ("lnf_b", "lnf_b (final LayerNorm bias), 32 values"),
                  ("U", "U (unembedding), shape 32 x 16: one line per row (column i = token i)"),
                  ("c", "c (output bias), 16 values (entry i = token i)")]


def weights_text(w):
    out = ["Transformer weights, values rounded to 3 decimals.", ""]
    for k, title in blocks():
        out.append(title)
        out += [" ".join(f"{v:.3f}" for v in row) for row in np.atleast_2d(w[k])]
        out.append("")
    return "\n".join(out)


def main(nets, out, seed=20261107):
    rng = np.random.default_rng(seed)
    out = Path(out)
    dec_file = Path(nets).parent / "b1_decisions_g2.json"
    dec = json.loads(dec_file.read_text()) if dec_file.exists() else None
    pick = []
    for cls in ("evennext", "evenprev"):
        ok = [f.stem for f in sorted(Path(nets).glob(f"{cls}_s*.npz"), key=lambda p: int(p.stem.rsplit("_s", 1)[1]))
              if json.loads(f.with_suffix(".json").read_text())["heldout_acc"] >= 0.95]
        if dec is None:
            pick += list(rng.choice(ok, 6, replace=False))
        else:  # counterbalanced on the B1 leave-one-seed-out decision (Amendment 1 rule)
            for side in ("evennext", "evenprev"):
                pick += list(rng.choice([n for n in ok if dec[n] == side], 3, replace=False))
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    key = {}
    for name in rng.permutation(pick):
        w = dict(np.load(Path(nets) / f"{name}.npz"))
        bid = secrets.token_hex(4)
        key[bid] = str(name).replace("evennext", "even_next").replace("evenprev", "even_prev")
        (out / "prompts" / f"{bid}.txt").write_text(
            "\n\n".join([TASK, NETWORK, QUESTION, "THE NETWORK'S WEIGHTS\n" + weights_text(w)]))
    kj = json.dumps(key, indent=1, sort_keys=True)
    (out / "key.json").write_text(kj)
    h = hashlib.sha256(kj.encode()).hexdigest()
    (out / "PREREG_KEY_HASH.txt").write_text(h + "  key.json\n")
    print("prompts", len(key), "hash", h)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
