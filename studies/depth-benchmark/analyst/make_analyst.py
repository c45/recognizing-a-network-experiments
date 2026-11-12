"""Analyst arm: Python-enabled reader set built from a blinded tool-free reader set (same networks, same IDs).

Usage: python3 -I make_analyst.py <readers dir> <nets dir> <family: pilot|l2d|<name>> [depth] [width]
Any family other than "pilot" uses the L2d game and question; depth and width (default 2, 32) set the description.
Writes models/<id>/blocks.npz next to this script (W1, b1, W2, b2, W3, b3 as in-by-out kernels; astra_loop.py reads
models/ and runs/ beside itself), message_<family>.txt, and copies the key and its hash as key_<family>.json and
PREREG_KEY_HASH_<family>.txt. The message is the tool-free prompt's description, game and question, with the weights replaced by
the Exp 3 rules for computing: one Python block per turn, weights preloaded as W, linear algebra allowed, no forward
pass.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "pilot_f1"))
sys.path.insert(0, str(HERE.parents[1] / "ladder/l2"))
sys.path.insert(0, str(HERE.parents[1] / "ladder/scale"))
import make_prompts as P  # noqa: E402
import make_prompts_l2d as L  # noqa: E402

COMPUTE = """WHAT YOU CAN USE. The network's weights, exactly as trained (float32). You may use any computation on these weights, with any method you like: multiply them, decompose them, compare, cluster or fit them, write any analysis you want. One thing is NOT allowed: a forward pass. Do not run the network or any of its layers (including tanh) on any input, whether real, random or constructed, in code, in your head or on paper. Treating the weight matrices as linear-algebra objects (products, eigenvalues, singular vectors, applying one weight matrix to directions taken from other weights) is fine; simulating what the network does to an input is not.

HOW YOU COMPUTE. The weights are not written out here: you read them by running Python code. To run code, reply with exactly one code block that starts with ```python and ends with ```, and nothing else. We run it and reply with its printed output. Available: numpy, scipy, scikit-learn, PyTorch (CPU), pandas, sympy, networkx. Before your code runs, numpy is imported as np and W is a dict of the weight arrays, named and shaped as in the formula above: W['W1'] is 500 x {width} (row i = input i), W['b1'] has {width} entries, and so on up to the output layer W['W{L}'] ({width} x 5) and W['b{L}'] (5). Each run is a fresh process, may take up to 600 seconds, and its output is cut at 20,000 characters. You may run code up to 100 times. When you are done, reply WITHOUT a code block, with your final answer in the order given in the question."""


def main(readers, nets, family, depth=2, width=32):
    from scale import description
    readers, out = Path(readers), HERE.parent
    game, question = (P.GAME, P.QUESTION) if family == "pilot" else (L.GAME, L.QUESTION)
    question = question.replace("You have no tools: no code execution, no calculator, nothing to run. This is a single response.",
                                "Use the computing rules below.")
    desc = description(depth, width)
    compute = COMPUTE.replace("{width}", str(width)).replace("{L}", str(depth + 1))
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / f"message_{family}.txt").write_text(desc.rstrip("\n") + "\n\n" + game + "\n\n" + compute + "\n\n" + question + "\n")
    key = json.loads((readers / "key.json").read_text())
    for bid, name in key.items():
        (out / "models" / bid).mkdir(exist_ok=True)
        np.savez(out / "models" / bid / "blocks.npz", **dict(np.load(Path(nets) / f"{name}.npz")))
    shutil.copy(readers / "key.json", out / f"key_{family}.json")
    shutil.copy(readers / "PREREG_KEY_HASH.txt", out / f"PREREG_KEY_HASH_{family}.txt")
    print("analyst set", len(key), "->", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], *(int(x) for x in sys.argv[4:6]))
