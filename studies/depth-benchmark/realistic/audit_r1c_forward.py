"""Forward-pass audit of the R1c with-Python runs: flag code that could evaluate the network on an input.

Usage: python3 -I audit_r1c_forward.py
Scans every code run in ../analyst/runs/r1c_*/tool_calls.jsonl for two pattern sets: (1) direct signs of a forward
pass (softmax/exp, GELU/erf/tanh, normalization functions, sequence construction, the words forward/simulate) and
(2) a broader sweep (maximum, sigmoid, stats CDFs, square-root normalizations, argmax, torch). Every flagged block was
read by hand; MANUAL_VERDICT records that review. Writes readers_r1c/forward_pass_audit.json.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIRECT = re.compile(r"softmax|np\.exp\(|\.exp\(|gelu|erf\(|tanh|def\s+N\b|def\s+norm|def\s+rms|randint|"
                    r"random\.(choice|permutation)|seq\s*=|sequence|forward|simulate", re.I)
BROAD = re.compile(r"np\.maximum|sigmoid|relu|norm\.cdf|Phi\(|scipy\.stats|/\s*np\.sqrt\(|argmax\(|torch\.", re.I)
MANUAL_VERDICT = ("All flagged blocks were read by hand. Direct-pattern hits are print statements asserting that no "
                  "softmax, normalization or sequence is used, the phrase 'forward versus backward target', and random "
                  "resampling of weight positions. Broad-sweep hits are query-key bilinear forms scaled by sqrt(10), "
                  "correlation denominators, argmax over weight-product matrices, and eigenvalue clipping. No code "
                  "applies attention probabilities, GELU or the normalization to an input, or constructs a token sequence.")


def main():
    runs = sorted((HERE.parent / "analyst" / "runs").glob("r1c_*"))
    n_code = direct_blocks = broad_lines = 0
    for r in runs:
        for line in (r / "tool_calls.jsonl").read_text().splitlines():
            code = json.loads(line)["code"]
            n_code += 1
            direct_blocks += bool(DIRECT.search(code))
            broad_lines += sum(bool(BROAD.search(l)) for l in code.splitlines())
    out = {"runs": len(runs), "code_runs": n_code, "direct_pattern_blocks": direct_blocks,
           "broad_pattern_lines": broad_lines, "forward_passes_found": 0, "manual_verdict": MANUAL_VERDICT}
    (HERE / "readers_r1c" / "forward_pass_audit.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "manual_verdict"})


if __name__ == "__main__":
    main()
