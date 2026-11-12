"""R1c paired-mirror reader set: 12 trained R1 networks and their exact mirrors, blinded and shuffled.

Usage: python3 -I make_pairs_r1c.py [seed]
Picks 12 included trained networks not used in readers_r1b (3 per condition, 6 per trained orientation, distinct
offset pairs within a condition), adds each one's mirror from nets_mirror/, and writes:
  readers_r1c/prompts/<id>.txt, key.json ({id: network name}), pairs.json ({source: [trained id, mirror id]}),
  PREREG_KEY_HASH.txt (SHA-256 of key.json + pairs.json);
  ../analyst/models/<id>/blocks.npz and ../analyst/message_r1c.txt for the with-Python arm (same ids).
"""
import hashlib
import json
import secrets
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_prompts_r1 import NETWORK, QUESTION, TASK, weights_text  # noqa: E402

COMPUTE = """WHAT YOU CAN USE. The network's weights, exactly as exported (float32). You may use any computation on these weights, with any method you like: multiply them, decompose them, compare, cluster or fit them, write any analysis you want. One thing is NOT allowed: a forward pass. Do not run the network or any of its parts (attention, MLP, normalization N, or any layer) on any input, whether a real sequence, a random one or a constructed one, in code, in your head or on paper. Treating the weight matrices as linear-algebra objects (products, eigenvalues, singular vectors, applying one weight matrix to rows or directions taken from other weights, such as embedding or position rows) is fine; simulating what the network does to an input sequence is not.

HOW YOU COMPUTE. The weights are not written out here: you read them by running Python code. To run code, reply with exactly one code block that starts with ```python and ends with ```, and nothing else. We run it and reply with its printed output. Available: numpy, scipy, scikit-learn, PyTorch (CPU), pandas, sympy, networkx. Before your code runs, numpy is imported as np and W is a dict of the weight arrays, named and shaped as in the formula above: W['E'] (64 x 40), W['P'] (16 x 40), W['Wq1'], W['Wk1'], W['Wv1'], W['Wo1'] (40 x 40), W['Wi1'] (40 x 160), W['Wf1'] (160 x 40), the same for layers 2 and 3, and W['U'] (40 x 64). Each run is a fresh process, may take up to 600 seconds, and its output is cut at 20,000 characters. You may run code up to 100 times. When you are done, reply WITHOUT a code block, with your final answer in the order given in the question."""


def main(seed=20261110):
    rng = np.random.default_rng(seed)
    r1b = json.loads((HERE / "readers_r1b" / "key.json").read_text()).values()
    used = set()
    for name in r1b:
        meta = json.loads((HERE / "nets_r1b" / f"{name}.json").read_text())
        used.add(meta.get("mirror_of", name))
    trained = {}
    for f in sorted((HERE / "nets").glob("r_*.npz")):
        m = json.loads(f.with_suffix(".json").read_text())
        if min(m["heldout_acc"].values()) >= 0.95 and f.stem not in used:
            trained[f.stem] = m
    pick, patterns = [], [[1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0]]
    for C, pat in zip(("C1", "C2", "C3", "C4"), patterns):
        groups = sorted({(min(m["a"], m["b"]), max(m["a"], m["b"])) for m in trained.values() if m["C"] == C})
        chosen = [groups[i] for i in rng.choice(len(groups), 3, replace=False)]
        for (lo, hi), o in zip(chosen, pat):
            a, b = (hi, lo) if o else (lo, hi)
            cands = sorted(k for k, m in trained.items() if (m["a"], m["b"], m["C"]) == (a, b, C))
            pick.append(str(rng.choice(cands)))
    out = HERE / "readers_r1c"
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    amodels = HERE.parent / "analyst" / "models"
    entries = []
    for src in pick:
        m = trained[src]
        mname = f"m_{m['b']}_{m['a']}_{m['C']}_s{m['seed']}"
        entries += [(src, src, HERE / "nets" / f"{src}.npz"), (src, mname, HERE / "nets_mirror" / f"{mname}.npz")]
    key, pairs = {}, {}
    for i in rng.permutation(len(entries)):
        src, name, path = entries[i]
        w = dict(np.load(path))
        bid = secrets.token_hex(4)
        key[bid] = name
        pairs.setdefault(src, []).append(bid)
        (out / "prompts" / f"{bid}.txt").write_text("\n\n".join([TASK, NETWORK, QUESTION, "THE NETWORK'S WEIGHTS\n" + weights_text(w)]))
        (amodels / bid).mkdir(parents=True, exist_ok=True)
        np.savez(amodels / bid / "blocks.npz", **w)
    question = QUESTION.replace("You have no tools: no code execution, no calculator, nothing to run. This is a single response.",
                                "Use the computing rules below.")
    (HERE.parent / "analyst" / "message_r1c.txt").write_text("\n\n".join([TASK, NETWORK, COMPUTE, question]) + "\n")
    kj = json.dumps(key, indent=1, sort_keys=True)
    pj = json.dumps(pairs, indent=1, sort_keys=True)
    (out / "key.json").write_text(kj)
    (out / "pairs.json").write_text(pj)
    h = hashlib.sha256((kj + pj).encode()).hexdigest()
    (out / "PREREG_KEY_HASH.txt").write_text(h + "  key.json + pairs.json\n")
    print("pairs", len(pairs), "prompts", len(key), "hash", h)
    print("trained orientations:", [int(trained[s]["a"] > trained[s]["b"]) for s in pick])


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
