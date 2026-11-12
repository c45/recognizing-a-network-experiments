"""Blinded tool-free reader set for R1 (rule grammar, multi-behavior, rotated residual stream).

Usage: python3 -I make_prompts_r1.py <nets dir> <out dir> [seed] [--counterbalance-b1]
Picks 16 groups (4 per condition), one network per group, 8 per orientation (optionally balanced on the B1
orientation decision), writes prompts/<id>.txt, key.json ({id: network name}) and PREREG_KEY_HASH.txt.
"""
import hashlib
import json
import secrets
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_family import table  # noqa: E402

TAB = table()

TASK = f"""THE TASK
The network reads a sequence of 16 tokens x[0..15] (ids 0-63) and outputs logits over the 64 ids, read at position 15. It was trained by supervised learning on a mixture of three kinds of sequences, and it gets more than 95% of held-out sequences of each kind right:

1. RULE sequences (60% of training): positions 0-14 hold 15 distinct tokens; the last token x[15] (the query) is a repeat of exactly one earlier token, found at position j, with j between 3 and 11. The network outputs the token at position j+a if condition C holds, and the token at position j+b otherwise. a and b are two different offsets from {{-3, -2, -1, +1, +2, +3}}, and C is one of:
   C1: the query x[15] is odd
   C2: the query x[15] is 32 or more
   C3: bit 1 of the query is set (x[15] mod 4 is 2 or 3)
   C4: the token at position 0, x[0], is odd
   Every network uses one fixed rule (a, b, C); there are 120 possible rules.
2. LOOKUP sequences (20%): all 16 tokens distinct (no repeat). The network outputs TABLE[x[15]], where TABLE is the same fixed permutation for every network: {' '.join(f'{i}->{TAB[i]}' for i in range(64))}.
3. MAJORITY sequences (20%): one token appears at 5 positions and the other 11 tokens are distinct. The network outputs the token that appears 5 times."""

NETWORK = """THE NETWORK
A pre-norm transformer with 3 layers, model width 40, 4 attention heads of width 10 per layer, a GELU MLP of width 160 per layer, RMS normalization, no biases and learned position embeddings. Every normalization gain has been folded into the weights that follow it, and the residual stream has been expressed in a random orthonormal basis (this does not change what the network computes; no coordinate of the residual stream has a special meaning). For a sequence x[0..15]:
  h[t] = E[x[t]] + P[t]                               E: 64 x 40 (row i = token i), P: 16 x 40 (row t = position t)
  N(v) = v / sqrt(mean(v^2) + 1e-6)                   (mean over the 40 entries of v)
  for layer l = 1, 2, 3:
    u[t] = N(h[t]);  q = u @ Wq_l,  k = u @ Wk_l,  v = u @ Wv_l          Wq_l, Wk_l, Wv_l: 40 x 40
    head n (n = 0..3) uses entries 10n..10n+9 of q, k and v:
      a_n[t, s] = softmax over s <= t of (q_n[t] . k_n[s]) / sqrt(10)    (causal: s <= t)
      z_n[t] = sum over s of a_n[t, s] v_n[s]
    h[t] = h[t] + concat(z_0[t], z_1[t], z_2[t], z_3[t]) @ Wo_l          Wo_l: 40 x 40
    h[t] = h[t] + GELU(N(h[t]) @ Wi_l) @ Wf_l                            Wi_l: 40 x 160, Wf_l: 160 x 40
  logits = N(h[15]) @ U                                                  U: 40 x 64 (column i = token i)
GELU(z) = z * Phi(z), with Phi the standard normal CDF. 63,360 parameters in total."""

QUESTION = """QUESTION. The weights below belong to one network trained on this mixture. Which RULE did it learn: what are a, b and C? Work from the weights. You have no tools: no code execution, no calculator, nothing to run. This is a single response.

Answer in this order:
Observations: <what you see in the weights that bears on the question; quote the weights you rely on>
Reasoning: <how those observations decide a, b and C>
Answer: a=<offset>, b=<offset>, C=<C1|C2|C3|C4>
Confidence: <1-100>% that the whole rule (a, b and C) is right"""


def blocks():
    out = [("E", "E (token embedding), shape 64 x 40: one line per row (row i = token i)"),
           ("P", "P (position embedding), shape 16 x 40: one line per row (row t = position t)")]
    for l in (1, 2, 3):
        out += [(f"Wq{l}", f"Wq_{l}, shape 40 x 40: one line per row"), (f"Wk{l}", f"Wk_{l}, shape 40 x 40: one line per row"),
                (f"Wv{l}", f"Wv_{l}, shape 40 x 40: one line per row"), (f"Wo{l}", f"Wo_{l}, shape 40 x 40: one line per row"),
                (f"Wi{l}", f"Wi_{l}, shape 40 x 160: one line per row"), (f"Wf{l}", f"Wf_{l}, shape 160 x 40: one line per row")]
    return out + [("U", "U (unembedding), shape 40 x 64: one line per row (column i = token i)")]


def weights_text(w):
    out = ["Network weights, values rounded to 3 decimals.", ""]
    for k, title in blocks():
        out.append(title)
        out += [" ".join(f"{v:.3f}" for v in row) for row in w[k]]
        out.append("")
    return "\n".join(out)


def main(nets, outdir, seed=20261109, counterbalance=False):
    rng = np.random.default_rng(seed)
    nets, outdir = Path(nets), Path(outdir)
    metas = {}
    for f in nets.glob("*.npz"):
        m = json.loads(f.with_suffix(".json").read_text())
        if min(m["heldout_acc"].values()) >= 0.95:
            metas[f.stem] = m
    b1 = json.loads((nets.parent / "b1_orientation.json").read_text()) if counterbalance else None
    pick, want_correct = [], {0: 0, 1: 0}
    for C in ("C1", "C2", "C3", "C4"):
        groups = sorted({(min(m["a"], m["b"]), max(m["a"], m["b"])) for m in metas.values() if m["C"] == C})
        chosen = [groups[i] for i in rng.choice(len(groups), 4, replace=False)]
        orient = rng.permutation([1, 1, 0, 0])
        for (lo, hi), o in zip(chosen, orient):
            a, b = (hi, lo) if o else (lo, hi)
            cands = sorted(k for k, m in metas.items() if (m["a"], m["b"], m["C"]) == (a, b, C))
            if b1 is not None:  # alternate, within each orientation, networks B1 gets right and wrong (4 and 4)
                target = want_correct[int(o)] % 2 == 0
                pref = [k for k in cands if (b1[k] == int(o)) == target]
                want_correct[int(o)] += 1
                cands = pref or cands
            pick.append(str(rng.choice(cands)))
    (outdir / "prompts").mkdir(parents=True, exist_ok=True)
    key = {}
    for name in rng.permutation(pick):
        w = dict(np.load(nets / f"{name}.npz"))
        bid = secrets.token_hex(4)
        m = metas[name]
        key[bid] = name  # r_<a>_<b>_<C>_s<seed> (trained) or m_<a>_<b>_<C>_s<seed> (exact mirror)
        (outdir / "prompts" / f"{bid}.txt").write_text("\n\n".join([TASK, NETWORK, QUESTION, "THE NETWORK'S WEIGHTS\n" + weights_text(w)]))
    kj = json.dumps(key, indent=1, sort_keys=True)
    (outdir / "key.json").write_text(kj)
    h = hashlib.sha256(kj.encode()).hexdigest()
    (outdir / "PREREG_KEY_HASH.txt").write_text(h + "  key.json\n")
    print("prompts", len(key), "hash", h)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], args[1], *(int(a) for a in args[2:]), counterbalance="--counterbalance-b1" in sys.argv)
