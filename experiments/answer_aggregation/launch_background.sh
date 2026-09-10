#!/usr/bin/env bash
# Background launcher for question-conditioned answer aggregation (merge + query filter).
# Does not write into hierarchical / KeyBERT artifact trees. Does not re-run RoBERTa QA.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/untie_venv/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/answer_aggregation/en}"
DATASET="${DATASET:-$ROOT/datasets/scirex_structured.json}"
EVIDENCE_DIR="${EVIDENCE_DIR:-$ROOT/artifacts/keybert_topic_clustering/en_full/evidence}"
DEVICE="${DEVICE:-auto}"
QUICK="${QUICK:-0}"
SKIP_BERTSCORE="${SKIP_BERTSCORE:-0}"
QUESTION="${QUESTION:-Which task was solved?}"
LOG="${LOG:-$LOG_DIR/answer_aggregation.log}"

mkdir -p "$LOG_DIR" "$OUT_DIR"

EXTRA=()
if [[ "$QUICK" == "1" ]]; then
  EXTRA+=(--quick --skip-bertscore)
fi
if [[ "$SKIP_BERTSCORE" == "1" ]]; then
  EXTRA+=(--skip-bertscore)
fi

nohup "$PYTHON" "$ROOT/experiments/answer_aggregation/run.py" \
  --dataset "$DATASET" \
  --output-dir "$OUT_DIR" \
  --evidence-dir "$EVIDENCE_DIR" \
  --language en \
  --device "$DEVICE" \
  --question "$QUESTION" \
  "${EXTRA[@]}" \
  > "$LOG" 2>&1 &

PID=$!
echo "PID=$PID"
echo "log=$LOG"
echo "output=$OUT_DIR"
echo "tail -f $LOG"
