"""Blinded tool-free reader set for transformer family T2 (EVEN-NEXT vs EVEN-PREV).

Usage: python3 -I make_prompts_tf.py <nets dir> <out dir> [seed]
Draws 6 included networks per class with a fixed seed, writes prompts/<id>.txt, key.json and PREREG_KEY_HASH.txt.
Key values are even_next_s<seed> / even_prev_s<seed>, so the expected answers are even-next / even-prev.
"""
import hashlib
import json
import secrets
import sys
from pathlib import Path

import numpy as np

TASK = """THE TASK
The network reads a sequence of 10 tokens at positions 0 to 9, from a vocabulary of 16 tokens with ids 0 to 15. Positions 0-8 always hold 9 distinct tokens. The last token (position 9, the query) is a repeat of exactly one earlier token, found at some position j between 1 and 7. The network outputs one token: logits over the 16 ids, read at position 9. It was trained by supervised learning to imitate a fixed rule on such sequences, and it follows that rule on more than 95% of held-out sequences.

The rule depends on whether the query token's id is even or odd. For one parity the network outputs the token right after the earlier occurrence (the token at position j+1, "NEXT"); for the other parity it outputs the token right before the earlier occurrence (the token at position j-1, "PREV")."""

NETWORK = """THE NETWORK
An attention-only transformer with 2 layers and one attention head per layer, model width 32 and head width 16. There are no MLPs, no LayerNorm and no biases. For a sequence x[0..9]:
  h0[t] = E[x[t]] + P[t]                       E: 16 x 32 (row i = token i), P: 10 x 32 (row t = position t)
  for layer l = 1, 2 (causal: position t attends only to positions s <= t):
    q = h @ Ql,  k = h @ Kl,  v = h @ Vl       Ql, Kl, Vl: 32 x 16
    a[t, s] = softmax over s <= t of (q[t] . k[s]) / 4
    hl[t] = h(l-1)[t] + (sum over s of a[t, s] v[s]) @ Ol      Ol: 16 x 32
  logits = h2[9] @ U                            U: 32 x 16 (column i = token i)
5,184 parameters in total."""

QUESTION = """QUESTION. The weights below belong to one network trained on this task. When the query token's id is EVEN, does the network output NEXT (the token at position j+1) or PREV (the token at position j-1)? Work from the weights. You have no tools: no code execution, no calculator, nothing to run. This is a single response.

Answer in this order:
Observations: <what you see in the weights that bears on the question; quote the weights you rely on>
Reasoning: <how those observations decide the question>
Answer: <even-next | even-prev>
Confidence: <50-100>%"""

BLOCKS = [("E", "E (token embedding), shape 16 x 32: one line per row (row i = token i)"),
          ("P", "P (position embedding), shape 10 x 32: one line per row (row t = position t)"),
          ("Q1", "Q1 (layer 1 query), shape 32 x 16: one line per row"),
          ("K1", "K1 (layer 1 key), shape 32 x 16: one line per row"),
          ("V1", "V1 (layer 1 value), shape 32 x 16: one line per row"),
          ("O1", "O1 (layer 1 output), shape 16 x 32: one line per row"),
          ("Q2", "Q2 (layer 2 query), shape 32 x 16: one line per row"),
          ("K2", "K2 (layer 2 key), shape 32 x 16: one line per row"),
          ("V2", "V2 (layer 2 value), shape 32 x 16: one line per row"),
          ("O2", "O2 (layer 2 output), shape 16 x 32: one line per row"),
          ("U", "U (unembedding), shape 32 x 16: one line per row (column i = token i)")]


def weights_text(w):
    out = ["Transformer weights, values rounded to 3 decimals.", ""]
    for k, title in BLOCKS:
        out.append(title)
        out += [" ".join(f"{v:.3f}" for v in row) for row in np.atleast_2d(w[k])]
        out.append("")
    return "\n".join(out)


def main(nets, out, seed=20261106):
    rng = np.random.default_rng(seed)
    out = Path(out)
    pick = []
    for cls in ("evennext", "evenprev"):
        ok = [f.stem for f in sorted(Path(nets).glob(f"{cls}_s*.npz"), key=lambda p: int(p.stem.rsplit("_s", 1)[1]))
              if json.loads(f.with_suffix(".json").read_text())["heldout_acc"] >= 0.95]
        pick += list(rng.choice(ok, 6, replace=False))
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
