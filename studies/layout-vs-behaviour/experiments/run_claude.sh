#!/bin/bash
# One tool-free Claude reply via claude -p. Usage: run_claude.sh <prompt file> <out dir> [model] [effort]
# CLAUDE_BIN names the CLI to run (default: claude on PATH).
set -euo pipefail
PROMPT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; mkdir -p "$2"; OUT="$(cd "$2" && pwd)"
MODEL="${3:-claude-opus-5-5}"; EFFORT="${4:-max}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT/cwd"
cp "$HERE/instructions.md" "$OUT/instructions.md"
start=$(date +%s)
"${CLAUDE_BIN:-claude}" --version > "$OUT/claude_version.txt"
(cd "$OUT/cwd" && CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 CLAUDE_CODE_DISABLE_CLAUDE_MDS=1 "${CLAUDE_BIN:-claude}" -p \
  --model "$MODEL" --effort "$EFFORT" \
  --system-prompt "$(cat "$OUT/instructions.md")" \
  --tools "" --strict-mcp-config --mcp-config '{"mcpServers":{}}' --disable-slash-commands \
  --setting-sources "" --no-session-persistence --permission-prompts none \
  --output-format stream-json --verbose \
  < "$PROMPT" > "$OUT/events.jsonl" 2> "$OUT/stderr.log") && rc=0 || rc=$?
python3 -I - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
tool_uses, text, result, init = [], [], None, None
for line in (out / "events.jsonl").read_text().splitlines():
    e = json.loads(line)
    if e.get("type") == "system" and e.get("subtype") == "init":
        init = {k: e.get(k) for k in ("model", "tools", "mcp_servers", "slash_commands", "skills", "plugins", "agents")}
    if e.get("type") == "assistant":
        for block in e["message"].get("content", []):
            if block.get("type") == "tool_use":
                tool_uses.append(block.get("name"))
            if block.get("type") == "text":
                text.append(block["text"])
    if e.get("type") == "result":
        result = e
answer = result.get("result", "") if result else "\n".join(text)
# When a reply reaches the output limit the CLI continues it in a new turn, and the result holds only the last
# turn's text. If visible text was written before the cut, keep all of it.
if result and not result.get("is_error") and len([t for t in text if t.strip()]) > 1:
    answer = "".join(text)
(out / "answer.md").write_text(answer)
run = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else {}
run.update(init=init, tool_uses=tool_uses, tool_free=not tool_uses,
           usage=result.get("usage") if result else None, model_usage=result.get("modelUsage") if result else None,
           is_error=result.get("is_error") if result else None, num_turns=result.get("num_turns") if result else None)
(out / "run.json").write_text(json.dumps(run, indent=1))
EOF
python3 -I - "$OUT" "$rc" "$(( $(date +%s) - start ))" <<'EOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "run.json"
run = json.loads(p.read_text())
run.update(rc=int(sys.argv[2]), wall_s=int(sys.argv[3]))
p.write_text(json.dumps(run, indent=1))
EOF
exit $rc
