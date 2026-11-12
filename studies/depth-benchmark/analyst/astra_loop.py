"""Analyst conversation loop: the reader replies with one Python block per turn; the output goes back in the same
conversation (Astra: codex exec resume, effort xhigh; Opus: claude -p --resume, effort from OPUS_EFFORT, default high, no built-in tools).
--resume-run continues an interrupted conversation from its saved turns.

Usage: python3 -I astra_loop.py <model id> [--max-runs 100] [--reader astra|opus] [--message f] [--run-name n]
Opus turns that hit a usage limit wait 20 minutes and retry; a reply over the 128k output-token limit ends the
conversation unanswered.
Copied from the Exp 3 re-ask harness; the sandbox venv path is resolved through its symlink.
Writes runs/<id>/turns/tNNN/{prompt.txt,answer.md,events.jsonl,stderr.log,run.json}, tool_calls.jsonl,
final_answer.md and state.json. Code runs in a macOS seatbelt sandbox with W preloaded from models/<id>/blocks.npz.
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

HERE = Path(__file__).resolve().parent
CODEX = sorted((Path.home() / ".codex/packages/app-server-daemon/releases").glob("*/codex"))[-1]
SANDBOX_PY = (HERE / ".venv-sandbox").resolve() / "bin/python"
PY_BASE = Path.home() / ".local/share/uv/python"
TIMEOUT, MAX_OUT = 600, 20_000
DISABLE = ["shell_tool", "unified_exec", "browser_use", "browser_use_external", "computer_use", "apps", "plugins",
           "image_generation", "multi_agent", "view_image", "sleep_tool", "goals", "tool_suggest", "skill_search",
           "code_mode_host", "in_app_browser", "hooks", "remote_plugin", "workspace_dependencies"]
PRELUDE = "import numpy as np\nW = {k: v for k, v in np.load('/WORK/blocks.npz').items()}\n"
CODE = re.compile(r"```python\s*\n(.*?)```", re.S)


def codex(prompt, turn_dir, thread, cwd):
    flags = ["--skip-git-repo-check", "--json", "-m", "gpt-6-astra", "-c", 'model_reasoning_effort="xhigh"',
             "-c", "model_context_window=872000", "-c", "model_auto_compact_token_limit=1000000",
             "-c", 'web_search="disabled"', "-c", f'model_instructions_file="{HERE / "instructions.md"}"',
             "-c", 'sandbox_mode="read-only"', "-c", 'approval_policy="never"']
    for f in DISABLE:
        flags += ["--disable", f]
    out = turn_dir / "answer.md"
    if thread is None:
        cmd = [str(CODEX), "exec", "--ignore-user-config", "--ignore-rules", "-C", str(cwd), *flags, "-o", str(out), "-"]
    else:
        cmd = [str(CODEX), "exec", "resume", *flags, "-o", str(out), thread, "-"]
    t0 = time.time()
    with open(turn_dir / "events.jsonl", "w") as ev, open(turn_dir / "stderr.log", "w") as er:
        rc = subprocess.run(cmd, input=prompt, text=True, stdout=ev, stderr=er, cwd=cwd).returncode
    events = [json.loads(l) for l in (turn_dir / "events.jsonl").read_text().splitlines() if l.strip().startswith("{")]
    items = [e.get("item", {}) for e in events if e.get("type") == "item.completed"]
    item_types = sorted({i.get("type") for i in items})
    tool_items = [i.get("type") for i in items if i.get("type") not in ("agent_message", "reasoning", "error")]
    tid = next((e.get("thread_id") for e in events if e.get("type") == "thread.started"), thread)
    usage = next((e.get("usage") for e in events if e.get("type") == "turn.completed"), None)
    rec = {"rc": rc, "wall_s": round(time.time() - t0, 1), "thread_id": tid, "item_types": item_types,
           "tool_items": tool_items, "tool_free": not tool_items, "usage": usage, "cmd": cmd}
    (turn_dir / "run.json").write_text(json.dumps(rec, indent=1))
    return rec, (out.read_text() if out.exists() else "")


# CLAUDE_MODEL / CLAUDE_BIN select another Claude model or Claude Code binary.
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
CLAUDE_FLAGS = ["--model", os.environ.get("CLAUDE_MODEL", "claude-opus-5-5"), "--effort", os.environ.get("OPUS_EFFORT", "high"), "--tools", "", "--strict-mcp-config",
                "--mcp-config", '{"mcpServers":{}}', "--disable-slash-commands", "--setting-sources", "",
                "--permission-prompts", "none", "--output-format", "stream-json", "--verbose"]


def claude_turn(prompt, turn_dir, session, cwd):
    cmd = [CLAUDE_BIN, "-p", "--system-prompt", (HERE / "instructions.md").read_text(), *CLAUDE_FLAGS]
    if session:
        cmd += ["--resume", session]
    for wait in range(60):
        env = {**os.environ, "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1"}
        t0 = time.time()
        with open(turn_dir / "events.jsonl", "w") as ev, open(turn_dir / "stderr.log", "w") as er:
            rc = subprocess.run(cmd, input=prompt, text=True, stdout=ev, stderr=er, cwd=cwd, env=env).returncode
        events = [json.loads(l) for l in (turn_dir / "events.jsonl").read_text().splitlines() if l.strip().startswith("{")]
        result = next((e for e in events if e.get("type") == "result"), {}) or {}
        text = result.get("result", "") or ""
        if ("limit" in text and ("hit your" in text or "reached your" in text or "usage" in text.lower())) or "Not logged in" in text:
            time.sleep(1200)
            continue
        break
    sid = next((e.get("session_id") for e in events if e.get("type") == "system" and e.get("subtype") == "init"), session)
    tool_uses = [b.get("name") for e in events if e.get("type") == "assistant"
                 for b in e.get("message", {}).get("content", []) if b.get("type") == "tool_use"]
    overflow = "output token maximum" in text
    rec = {"rc": rc, "wall_s": round(time.time() - t0, 1), "thread_id": sid, "tool_items": tool_uses,
           "tool_free": not tool_uses, "usage": result.get("usage"), "is_error": result.get("is_error"),
           "overflow": overflow, "limit_waits": wait,
           "cmd": cmd[:2] + ["--system-prompt", "<instructions.md>"] + cmd[4:]}
    (turn_dir / "run.json").write_text(json.dumps(rec, indent=1))
    answer = "" if (result.get("is_error") or overflow) else text
    (turn_dir / "answer.md").write_text(text)
    return rec, answer


def run_code(code, model_dir):
    work = Path(tempfile.mkdtemp(prefix="astra_work_")).resolve()
    try:
        shutil.copy(model_dir / "blocks.npz", work / "blocks.npz")
        profile = f"""(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{work}") (literal "/dev/null"))
(deny file-read* (subpath "{Path.home()}") (subpath "/private/tmp") (subpath "{Path(tempfile.gettempdir()).resolve()}"))
(allow file-read* (subpath "{work}") (subpath "{(HERE / '.venv-sandbox').resolve()}") (subpath "{PY_BASE}"))
"""
        src = PRELUDE.replace("/WORK", str(work)) + code
        env = {"PATH": "/usr/bin:/bin", "HOME": str(work), "OMP_NUM_THREADS": "8", "TMPDIR": str(work)}
        t0 = time.time()
        try:
            p = subprocess.run(["sandbox-exec", "-p", profile, str(SANDBOX_PY), "-I", "-c", src], capture_output=True,
                               text=True, timeout=TIMEOUT, cwd=work, env=env)
            out = p.stdout + (("\n[stderr]\n" + p.stderr[-6000:]) if p.stderr.strip() else "")
        except subprocess.TimeoutExpired:
            out = f"[timed out after {TIMEOUT} s]"
        if len(out) > MAX_OUT:
            out = out[:MAX_OUT] + f"\n[output cut at {MAX_OUT:,} characters]"
        return out or "(no output)", round(time.time() - t0, 1)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--max-runs", type=int, default=100)
    ap.add_argument("--message", default="message.txt")
    ap.add_argument("--run-name")
    ap.add_argument("--reader", choices=["astra", "opus"], default="astra")
    ap.add_argument("--resume-run", action="store_true",
                    help="continue an interrupted Claude conversation from its saved turns instead of starting over")
    a = ap.parse_args()
    turn = codex if a.reader == "astra" else claude_turn
    model_dir = HERE / "models" / a.model
    rd = HERE / "runs" / (a.run_name or a.model)
    rd.mkdir(parents=True, exist_ok=True)
    cwd = rd / "cwd"
    cwd.mkdir(exist_ok=True)
    (rd / "message.txt").write_text((HERE / a.message).read_text())
    prompt, thread, runs, start = (HERE / a.message).read_text(), None, 0, 1
    if a.resume_run and a.reader == "opus":
        # Last turn with a saved, valid reply; later turn folders hold at most a prompt and are redone.
        k = 0
        for d in sorted((rd / "turns").glob("t*")) if (rd / "turns").exists() else []:
            rj, am = d / "run.json", d / "answer.md"
            ok = rj.exists() and am.exists() and am.read_text().strip()
            if ok:
                r0 = json.loads(rj.read_text())
                ok = r0.get("rc") == 0 and not r0.get("is_error") and not r0.get("overflow")
            if not ok:
                break
            k = int(d.name[1:])
        if k:
            r0 = json.loads((rd / "turns" / f"t{k:03d}" / "run.json").read_text())
            thread = r0["thread_id"]
            answer = (rd / "turns" / f"t{k:03d}" / "answer.md").read_text()
            calls = [json.loads(l) for l in (rd / "tool_calls.jsonl").read_text().splitlines()] if (rd / "tool_calls.jsonl").exists() else []
            runs = len(calls)
            m = CODE.search(answer)
            if not m or runs >= a.max_runs and runs >= k:
                (rd / "final_answer.md").write_text(answer)
                (rd / "state.json").write_text(json.dumps({"done": True, "turns": k, "runs": runs, "thread": thread,
                                                           "reader": a.reader}))
                return
            if runs < k:   # the reply of turn k was saved but its code had not been run yet
                runs += 1
                out, sec = run_code(m.group(1), model_dir)
                with open(rd / "tool_calls.jsonl", "a") as f:
                    f.write(json.dumps({"t": time.time(), "run": runs, "s": sec, "code": m.group(1), "out": out}) + "\n")
            else:
                out = calls[-1]["out"]
            prompt = f"OUTPUT of run {runs} of at most {a.max_runs}:\n```\n{out}\n```\n"
            start = k + 1
            print(f"{rd.name} resuming at turn {start} (run {runs})", flush=True)
    for t in range(start, a.max_runs + 2):
        td = rd / "turns" / f"t{t:03d}"
        td.mkdir(parents=True, exist_ok=True)
        (td / "prompt.txt").write_text(prompt)
        for attempt in range(3):
            if a.reader == "opus":
                rec, answer = claude_turn(prompt, td, thread, cwd)
            else:
                rec, answer = turn(prompt, td, thread, cwd)
            if rec.get("overflow"):
                break
            if rec["rc"] == 0 and answer.strip():
                break
            time.sleep(60)
        thread = rec["thread_id"]
        if not answer.strip():
            err = "output overflow" if rec.get("overflow") else "empty answer"
            (rd / "state.json").write_text(json.dumps({"done": False, "error": err, "turns": t, "runs": runs,
                                                       "reader": a.reader}))
            sys.exit(1)
        m = CODE.search(answer)
        if not m or runs >= a.max_runs:
            (rd / "final_answer.md").write_text(answer)
            break
        runs += 1
        out, sec = run_code(m.group(1), model_dir)
        with open(rd / "tool_calls.jsonl", "a") as f:
            f.write(json.dumps({"t": time.time(), "run": runs, "s": sec, "code": m.group(1), "out": out}) + "\n")
        prompt = f"OUTPUT of run {runs} of at most {a.max_runs}:\n```\n{out}\n```\n"
        print(f"{rd.name} turn {t} run {runs} ({sec}s)", flush=True)
    (rd / "state.json").write_text(json.dumps({"done": True, "turns": t, "runs": runs, "thread": thread,
                                               "reader": a.reader}))
    print(f"{rd.name} done: {t} turns, {runs} runs", flush=True)


if __name__ == "__main__":
    main()
