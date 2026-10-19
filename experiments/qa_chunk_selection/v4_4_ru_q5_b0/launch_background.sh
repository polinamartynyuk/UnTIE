#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

BASE_ARTIFACT="${BASE_ARTIFACT:-$PROJECT_ROOT/artifacts/qa_chunk_selection_v4_4_ru_data_20260910_173020}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/artifacts/qa_chunk_selection_v4_4_ru_q5_b0_$STAMP}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/qa_chunk_selection_v4_4_ru_q5_b0_$STAMP.log}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/untie_venv/bin/python}"
DEVICE="${DEVICE:-auto}"

mkdir -p "$PROJECT_ROOT/logs"
nohup "$PYTHON_BIN" -u -m experiments.qa_chunk_selection.v4_4_ru_q5_b0 \
  --base-artifact "$BASE_ARTIFACT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  "$@" >"$LOG_FILE" 2>&1 &
PID=$!
printf '%s\n' "$PID" >"$LOG_FILE.pid"
printf 'PID=%s\nOUTPUT_DIR=%s\nLOG_FILE=%s\n' "$PID" "$OUTPUT_DIR" "$LOG_FILE"

