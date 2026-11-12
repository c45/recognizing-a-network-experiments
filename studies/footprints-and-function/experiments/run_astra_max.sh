#!/bin/bash
# One tool-free gpt-6-astra reply via codex exec at reasoning effort max (the matched-effort arm; see
# MATCHED_EFFORT_PREREG.md). Identical to run_astra.sh apart from the effort. Usage: run_astra_max.sh <prompt file> <out dir>
set -euo pipefail
PROMPT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; mkdir -p "$2"; OUT="$(cd "$2" && pwd)"
CODEX="$(ls -d "$HOME"/.codex/packages/app-server-daemon/releases/*/ | tail -1)codex"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT/cwd"
cp "$HERE/instructions.md" "$OUT/instructions.md"
DISABLE=()
for f in shell_tool unified_exec browser_use browser_use_external computer_use apps plugins image_generation \
         multi_agent view_image sleep_tool goals tool_suggest skill_search code_mode_host in_app_browser hooks \
         remote_plugin workspace_dependencies; do DISABLE+=(--disable "$f"); done
start=$(date +%s)
"$CODEX" --version > "$OUT/codex_version.txt"
"$CODEX" exec --ignore-user-config --ignore-rules --ephemeral --skip-git-repo-check --json \
  -C "$OUT/cwd" -s read-only "${DISABLE[@]}" \
  -m gpt-6-astra -c 'model_reasoning_effort="max"' \
  -c 'model_context_window=872000' -c 'model_auto_compact_token_limit=1000000' \
  -c 'web_search="disabled"' \
  -c "model_instructions_file=\"$OUT/instructions.md\"" \
  -o "$OUT/answer.md" - < "$PROMPT" > "$OUT/events.jsonl" 2> "$OUT/stderr.log" && rc=0 || rc=$?
echo max > "$OUT/effort.txt"
echo "{\"rc\": $rc, \"wall_s\": $(( $(date +%s) - start )), \"effort\": \"max\"}" > "$OUT/run.json"
exit $rc
