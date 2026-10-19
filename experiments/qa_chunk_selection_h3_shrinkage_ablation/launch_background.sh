#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/aiserver/POLINA/UnTIE_project"
PYTHON="$PROJECT_ROOT/untie_venv/bin/python3"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/artifacts/qa_chunk_selection_h3_shrinkage_ablation/run_$STAMP}"
LOG_DIR="$PROJECT_ROOT/logs/qa_chunk_selection_h3_shrinkage_ablation"
LOG_FILE="${LOG_FILE:-$LOG_DIR/run_$STAMP.log}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment is missing: $PYTHON" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

nohup "$PYTHON" -m experiments.qa_chunk_selection_h3_shrinkage_ablation \
  --run-dir "$RUN_DIR" "$@" >"$LOG_FILE" 2>&1 &
PID=$!
printf '%s\n' "$PID" >"$LOG_FILE.pid"

echo "Started PID=$PID"
echo "Run directory: $RUN_DIR"
echo "Log: $LOG_FILE"
echo "Progress: $RUN_DIR/progress.jsonl"
