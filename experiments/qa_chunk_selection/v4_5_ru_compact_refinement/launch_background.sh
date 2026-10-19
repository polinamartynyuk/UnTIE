#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="$PROJECT_ROOT/artifacts/qa_chunk_selection_v4_5_ru/run_$STAMP"
LOG="$PROJECT_ROOT/logs/qa_chunk_selection_v4_5_ru_$STAMP.log"
mkdir -p "$PROJECT_ROOT/logs"

cd "$PROJECT_ROOT"
nohup "$PROJECT_ROOT/untie_venv/bin/python" \
  -m experiments.qa_chunk_selection.v4_5_ru_compact_refinement \
  --phase all \
  --datasets russerrc,scimdix \
  --output-dir "$OUTPUT" \
  "$@" >"$LOG" 2>&1 &
PID=$!
printf '%s\n' "$PID" >"$LOG.pid"
echo "PID: $PID"
echo "Log: $LOG"
echo "Output: $OUTPUT"
echo "Progress: tail -f '$LOG'"

