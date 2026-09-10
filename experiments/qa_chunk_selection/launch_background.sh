#!/usr/bin/env bash
# Background launcher for isolated SciREX chunk-selection experiments A–D.
# Reads frozen Phase 1 Q5 labels. Writes only to artifacts/qa_chunk_selection/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/untie_venv/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/qa_chunk_selection}"
DEVICE="${DEVICE:-auto}"
STAGES="${STAGES:-all}"
CANDIDATE_MODE="${CANDIDATE_MODE:-all}"
SMOKE="${SMOKE:-0}"
SKIP_QA="${SKIP_QA:-0}"
RESUME="${RESUME:-0}"
LIMIT_DOCS="${LIMIT_DOCS:-0}"
SEED="${SEED:-42}"
LOG="${LOG:-$LOG_DIR/qa_chunk_selection.log}"

mkdir -p "$LOG_DIR" "$OUT_DIR"

EXTRA=()
if [[ "$SMOKE" == "1" ]]; then
  EXTRA+=(--smoke)
fi
if [[ "$SKIP_QA" == "1" ]]; then
  EXTRA+=(--skip-qa)
fi
if [[ "$RESUME" == "1" ]]; then
  EXTRA+=(--resume)
fi
if [[ "$LIMIT_DOCS" != "0" ]]; then
  EXTRA+=(--limit-docs "$LIMIT_DOCS")
fi

nohup "$PYTHON" -m experiments.qa_chunk_selection \
  --output-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --stages "$STAGES" \
  --candidate-mode "$CANDIDATE_MODE" \
  --seed "$SEED" \
  --skip-bertscore \
  "${EXTRA[@]}" \
  > "$LOG" 2>&1 &

PID=$!
echo "PID=$PID"
echo "log=$LOG"
echo "output=$OUT_DIR"
echo "progress=$OUT_DIR/progress.jsonl"
echo "tail -f $LOG"
