"""Analyst arm for the recognition study (copy of obscured/analyst/analyst_loop.py, generalised).

Usage: python3 -I analyst_loop_r.py <backend astra|opus> <network id> <suite dir> [--max-runs 30]
Differences from the original: the description is the suite's own nets/<id>/head.txt (the tool-free prompt text before
its QUESTION), so it includes any stated input encoding or changed widths; the weights are nets/<id>/actor.npz with any
shapes; nets/<id>/extra.npz arrays (e.g. groups, Q) are also loaded into W.
Original description follows.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ff_paths import SANDBOX_PY  # noqa: E402
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
LIMIT = re.compile(r"(?i)hit your (5-hour |weekly )?limit|usage limit|rate limit|limit (will )?reset|resets? (at )?\d")


def shapes_text(w):
    parts = [f"W['{k}'] " + " x ".join(str(d) for d in w[k].shape) for k in w]
    return ", ".join(parts) + " (the same arrays are also available under those names as variables)"


def prelude(names):
    return ("import numpy as np\nW = {k: v for k, v in np.load('/WORK/actor.npz').items()}\n"
            f"{', '.join(names)} = (W[k] for k in {tuple(names)!r})\n")


def run_code(code, weights_npz, pre):
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
    flags = ["--skip-git-repo-check", "--json", "-m", "gpt-6-astra", "-c", 'model_reasoning_effort="high"',
             "-c", "model_context_window=872000", "-c", "model_auto_compact_token_limit=1000000",
             "-c", 'web_search="disabled"', "-c", f'model_instructions_file="{HERE / "analyst_instructions.md"}"',
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


CLAUDE = {"opus": ("claude-opus-5-5", {})}


def ask_claude(prompt, td, thread, cwd, backend="opus"):
    model, extra_env = CLAUDE[backend]
    cmd = ["claude", "-p", "--model", model, "--effort", "max",
           "--system-prompt", (HERE / "analyst_instructions.md").read_text(), "--tools", "", "--strict-mcp-config",
           "--mcp-config", '{"mcpServers":{}}', "--disable-slash-commands", "--setting-sources", "",
           "--permission-prompts", "none", "--output-format", "stream-json", "--verbose"]
    if thread:
        cmd += ["--resume", thread]
    env = {**os.environ, "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", **extra_env}
    with open(td / "events.jsonl", "w") as ev, open(td / "stderr.log", "w") as er:
        rc = subprocess.run(cmd, input=prompt, text=True, stdout=ev, stderr=er, cwd=cwd, env=env).returncode
    d = {}
    for line in (td / "events.jsonl").read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "result":
            d = e
    (td / "raw.json").write_text(json.dumps(d))
    answer = d.get("result") or ""
    (td / "answer.md").write_text(answer)
    rec = {"rc": rc, "thread": d.get("session_id", thread), "is_error": d.get("is_error"),
           "usage": d.get("usage"), "model_usage": d.get("modelUsage"), "num_turns": d.get("num_turns"), "tool_free": True}
    return rec, answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=["astra", "opus"])
    ap.add_argument("nid")
    ap.add_argument("suite")
    ap.add_argument("--max-runs", type=int, default=30)
    a = ap.parse_args()
    suite = Path(a.suite).resolve()
    rd = suite / "analyst" / f"{a.backend}_{a.nid}"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "cwd").mkdir(exist_ok=True)
    weights = rd / "weights.npz"
    nd = suite / "nets" / a.nid
    w = {k: v for k, v in np.load(nd / "actor.npz").items()}
    if (nd / "extra.npz").exists():
        w.update({k: v for k, v in np.load(nd / "extra.npz").items()})
    np.savez(weights, **w)
    pre = prelude(list(w))
    desc = (nd / "head.txt").read_text()
    desc = desc[:desc.index("QUESTION.")].rstrip("\n")
    prompt = desc + QUESTION.replace("{max_runs}", str(a.max_runs)).replace("{shapes}", shapes_text(w))
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
            if LIMIT.search(err) and attempt < 60:
                print(f"{rd.name}: usage limit, waiting 10 min", flush=True)
                time.sleep(600)
                continue
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
