"""Analyst arm: a model gets the Exp 1 description and question, and analyses the weights by running Python.

Usage: python3 -I analyst_loop.py <backend astra|opus> <network id> <suite dir> [--max-runs 30]
The weights are the exact printed values (3 decimals) of <suite>/nets/<id>/actor.txt, loaded in a sandbox as numpy
arrays W['W1'] (500x32), W['b1'], W['W2'] (32x32), W['b2'], W['W3'] (32x5), W['b3'] (also as W1, b1, ...). The model
replies with one ```python block per turn; we run it (macOS seatbelt: no network, no files outside its work folder,
600 s, output cut at 20,000 characters) and send back "OUTPUT of run k of at most N:". A reply without a code block
is the final answer. Astra runs through codex exec / codex exec resume (effort high, all tools off); Opus through
claude -p / claude -p --resume (effort max, all tools off).
Writes <suite>/analyst/<backend>_<id>/turns/tNNN/{prompt.txt,answer.md,raw.json|events.jsonl,run.json},
tool_calls.jsonl, final_answer.md, state.json.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arch import arch_of, describe, keys, parse_text, sizes  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import DESCRIPTION, SANDBOX_PY  # noqa: E402
PY_BASE = Path.home() / ".local/share/uv/python"
CODEX = sorted((Path.home() / ".codex/packages/app-server-daemon/releases").glob("*/codex"))[-1]
TIMEOUT, MAX_OUT = 600, 20_000
CODE = re.compile(r"```python\s*\n(.*?)```", re.S)
DISABLE = ["shell_tool", "unified_exec", "browser_use", "browser_use_external", "computer_use", "apps", "plugins",
           "image_generation", "multi_agent", "view_image", "sleep_tool", "goals", "tool_suggest", "skill_search",
           "code_mode_host", "in_app_browser", "hooks", "remote_plugin", "workspace_dependencies"]
QUESTION = """

QUESTION. The weights belong to one policy network that was trained with reinforcement learning to play a game on this grid. Which game was it trained on? Work from the weights.

HOW YOU WORK. The weights are not printed here: you analyse them by running Python. Reply with exactly one code block that starts with ```python and ends with ```, and nothing else; we run it and reply with its printed output. Before your code runs, numpy is imported as np and W is a dict of the weight arrays: {shapes}. They are the weights rounded to 3 decimals. Available: numpy, scipy, scikit-learn, PyTorch (CPU), pandas, sympy, networkx. Any analysis is allowed, including running the network on inputs you construct. There is no game environment to run. Each run is a fresh process (nothing is kept between runs), may take up to 600 seconds, and its output is cut at 20,000 characters. You may run code up to {max_runs} times. When you are done, reply WITHOUT a code block, with your final answer in this order:
Observations: <what you found in the weights and how; quote the numbers you rely on>
Reasoning: <which kinds of game fit these observations, which are ruled out, and why>
Game: <the game, as specifically as you can; if it resembles a well-known game, give that game's name>
Description: <what the agent does and what gives reward, in one or two sentences>
Confidence: <0-100>%
"""
PRELUDE = ("import numpy as np\nW = {k: v for k, v in np.load('/WORK/actor.npz').items()}\n"
           "W1, b1, W2, b2, W3, b3 = (W[k] for k in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'))\n")
LIMIT = re.compile(r"(?i)hit your (5-hour |weekly )?limit|usage limit|rate limit|limit (will )?reset|resets? (at )?\d")


def shapes_text(arch):
    if arch == "tanh32x2":
        return ("W['W1'] is 500 x 32 (row i = input i), W['b1'] 32, W['W2'] 32 x 32, W['b2'] 32, W['W3'] 32 x 5, "
                "W['b3'] 5 (the same arrays are also available as W1, b1, W2, b2, W3, b3)")
    s, n = sizes(arch), len(sizes(arch)) - 1
    parts = []
    for k in range(1, n + 1):
        parts.append(f"W['W{k}'] " + ("is " if k == 1 else "") + f"{s[k - 1]} x {s[k]}" +
                     (" (row i = input i)" if k == 1 else ""))
        parts.append(f"W['b{k}'] {s[k]}")
    names = ", ".join(keys(arch))
    return ", ".join(parts) + f" (the same arrays are also available as {names})"


def prelude(arch):
    ks = keys(arch)
    return ("import numpy as np\nW = {k: v for k, v in np.load('/WORK/actor.npz').items()}\n"
            f"{', '.join(ks)} = (W[k] for k in {tuple(ks)!r})\n")


def parse_actor_txt(path):
    blocks, cur = {}, None
    for ln in Path(path).read_text().splitlines():
        head = ln.split(" ", 1)[0]
        if head in ("W1", "b1", "W2", "b2", "W3", "b3") and "shape" in ln:
            cur = head
            blocks[cur] = []
            continue
        if cur and ln.strip():
            try:
                blocks[cur].append([float(x) for x in ln.split()])
            except ValueError:
                pass
    w = {k: np.array(v) for k, v in blocks.items()}
    for k in ("b1", "b2", "b3"):
        w[k] = w[k].ravel()
    return w


def run_code(code, weights_npz, pre=PRELUDE):
    work = Path(tempfile.mkdtemp(prefix="analyst_work_"))
    try:
        shutil.copy(weights_npz, work / "actor.npz")
        profile = f"""(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{work}") (subpath "/private/var/folders") (subpath "/private/tmp") (literal "/dev/null"))
(deny file-read* (subpath "{Path.home()}"))
(allow file-read* (subpath "{work}") (subpath "{SANDBOX_PY.parent.parent}") (subpath "{PY_BASE}"))
"""
        env = {"PATH": "/usr/bin:/bin", "HOME": str(work), "OMP_NUM_THREADS": "8", "TMPDIR": str(work)}
        t0 = time.time()
        try:
            p = subprocess.run(["sandbox-exec", "-p", profile, str(SANDBOX_PY), "-I", "-c",
                                pre.replace("/WORK", str(work)) + code], capture_output=True, text=True,
                               timeout=TIMEOUT, cwd=work, env=env)
            out = p.stdout + (("\n[stderr]\n" + p.stderr[-6000:]) if p.stderr.strip() else "")
        except subprocess.TimeoutExpired:
            out = f"[timed out after {TIMEOUT} s]"
        if len(out) > MAX_OUT:
            out = out[:MAX_OUT] + f"\n[output cut at {MAX_OUT:,} characters]"
        return out or "(no output)", round(time.time() - t0, 1)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def ask_astra(prompt, td, thread, cwd):
    flags = ["--skip-git-repo-check", "--json", "-m", "gpt-6-astra", "-c", 'model_reasoning_effort="max"',
             "-c", "model_context_window=872000", "-c", "model_auto_compact_token_limit=1000000",
             "-c", 'web_search="disabled"', "-c", f'model_instructions_file="{HERE / "instructions.md"}"',
             "-c", 'sandbox_mode="read-only"', "-c", 'approval_policy="never"']
    for f in DISABLE:
        flags += ["--disable", f]
    out = td / "answer.md"
    cmd = ([str(CODEX), "exec", "--ignore-user-config", "--ignore-rules", "-C", str(cwd), *flags, "-o", str(out), "-"]
           if thread is None else [str(CODEX), "exec", "resume", *flags, "-o", str(out), thread, "-"])
    with open(td / "events.jsonl", "w") as ev, open(td / "stderr.log", "w") as er:
        rc = subprocess.run(cmd, input=prompt, text=True, stdout=ev, stderr=er, cwd=cwd).returncode
    events = [json.loads(l) for l in (td / "events.jsonl").read_text().splitlines() if l.strip().startswith("{")]
    items = [e.get("item", {}) for e in events if e.get("type") == "item.completed"]
    tools = [i.get("type") for i in items if i.get("type") not in ("agent_message", "reasoning", "error")]
    tid = next((e.get("thread_id") for e in events if e.get("type") == "thread.started"), thread)
    rec = {"rc": rc, "thread": tid, "tool_items": tools, "tool_free": not tools,
           "usage": next((e.get("usage") for e in events if e.get("type") == "turn.completed"), None)}
    return rec, (out.read_text() if out.exists() else "")


# backend -> (model, CLI, extra environment)
CLAUDE = {"opus": ("claude-opus-5-5", os.environ.get("OPUS_CLI", "claude"), {})}


def ask_claude(prompt, td, thread, cwd, backend="opus"):
    model, cli, extra_env = CLAUDE[backend]
    cmd = [cli, "-p", "--model", model, "--effort", "max",
           "--system-prompt", (HERE / "instructions.md").read_text(), "--tools", "", "--strict-mcp-config",
           "--mcp-config", '{"mcpServers":{}}', "--disable-slash-commands", "--setting-sources", "",
           "--permission-prompts", "none", "--output-format", "stream-json", "--verbose"]
    if thread:
        cmd += ["--resume", thread]
    env = {**os.environ, "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", **extra_env}
    p = subprocess.run(cmd, input=prompt, text=True, capture_output=True, cwd=cwd, env=env)
    (td / "events.jsonl").write_text(p.stdout)
    (td / "stderr.log").write_text(p.stderr)
    d = {}
    for line in p.stdout.splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "result":
            d = e
    (td / "raw.json").write_text(json.dumps(d))
    answer = d.get("result") or ""
    (td / "answer.md").write_text(answer)
    rec = {"rc": p.returncode, "thread": d.get("session_id", thread), "is_error": d.get("is_error"),
           "usage": d.get("usage"), "num_turns": d.get("num_turns"), "tool_free": True}
    return rec, answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=["astra", "opus"])
    ap.add_argument("nid")
    ap.add_argument("suite")
    ap.add_argument("--max-runs", type=int, default=30)
    a = ap.parse_args()
    suite = Path(a.suite).resolve()
    rd = suite / "analyst_max" / f"{a.backend}_{a.nid}"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "cwd").mkdir(exist_ok=True)
    weights = rd / "weights.npz"
    w = parse_text(suite / "nets" / a.nid / "actor.txt")
    arch = arch_of(w)
    np.savez(weights, **w)
    pre = PRELUDE if arch == "tanh32x2" else prelude(arch)
    prompt = (describe(DESCRIPTION.read_text(), arch).rstrip("\n") +
              QUESTION.replace("{max_runs}", str(a.max_runs)).replace("{shapes}", shapes_text(arch)))
    (rd / "message.txt").write_text(prompt)
    ask = ask_astra if a.backend == "astra" else (lambda *x: ask_claude(*x, backend=a.backend))
    thread, runs = None, 0
    for t in range(1, a.max_runs + 2):
        td = rd / "turns" / f"t{t:03d}"
        td.mkdir(parents=True, exist_ok=True)
        (td / "prompt.txt").write_text(prompt)
        attempt = 0
        while True:
            rec, answer = ask(prompt, td, thread, rd / "cwd")
            if rec["rc"] == 0 and answer.strip() and not rec.get("is_error"):
                break
            err = answer + (td / "stderr.log").read_text(errors="ignore")[-3000:]
            attempt += 1
            if LIMIT.search(err):
                break
            if attempt >= 3:
                break
            time.sleep(60)
        (td / "run.json").write_text(json.dumps(rec, indent=1))
        thread = rec["thread"]
        if not answer.strip() or rec.get("is_error"):
            (rd / "state.json").write_text(json.dumps({"done": False, "error": answer[:300] or "empty", "turns": t}))
            sys.exit(1)
        m = CODE.search(answer)
        if not m or runs >= a.max_runs:
            (rd / "final_answer.md").write_text(answer)
            break
        runs += 1
        out, sec = run_code(m.group(1), weights, pre)
        with open(rd / "tool_calls.jsonl", "a") as f:
            f.write(json.dumps({"run": runs, "s": sec, "code": m.group(1), "out": out}) + "\n")
        prompt = f"OUTPUT of run {runs} of at most {a.max_runs}:\n```\n{out}\n```\n"
        print(f"{rd.name} run {runs} ({sec}s)", flush=True)
    (rd / "state.json").write_text(json.dumps({"done": True, "turns": t, "runs": runs, "thread": thread}))
    print(f"{rd.name} done: {runs} runs", flush=True)


if __name__ == "__main__":
    main()
