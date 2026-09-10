#!/usr/bin/env bash
# Background launcher for the isolated KeyBERT + extractive QA experiment (hyp3).
# Does not write into artifacts/hierarchical_topic_optimized/ or
# artifacts/keybert_topic_clustering/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/untie_venv/bin/python}"
PIP="${PIP:-$ROOT/untie_venv/bin/pip}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/keybert_qa_integration/en}"
DATASET="${DATASET:-$ROOT/datasets/scirex_structured.json}"
STATIC_MODEL="${STATIC_MODEL:-$ROOT/experiments/analysis_results/keyword_tuning_task/en/full_v5/scart_tuned_model.json}"
EVIDENCE_DIR="${EVIDENCE_DIR:-$ROOT/artifacts/keybert_topic_clustering/en_full/evidence}"
DEVICE="${DEVICE:-auto}"
QUICK="${QUICK:-0}"
INCLUDE_CHUNK="${INCLUDE_CHUNK:-1}"
SKIP_BERTSCORE="${SKIP_BERTSCORE:-0}"
LOG="${LOG:-$LOG_DIR/keybert_qa_integration.log}"

mkdir -p "$LOG_DIR" "$OUT_DIR"

if ! "$PYTHON" -c "import keybert" >/dev/null 2>&1; then
  echo "Installing keybert==0.9.0 into the project venv..."
  "$PIP" install "keybert==0.9.0"
fi

EXTRA=()
if [[ "$QUICK" == "1" ]]; then
  EXTRA+=(--quick --skip-bertscore)
  INCLUDE_CHUNK=0
fi
if [[ "$SKIP_BERTSCORE" == "1" ]]; then
  EXTRA+=(--skip-bertscore)
fi
if [[ "$INCLUDE_CHUNK" == "1" ]]; then
  EXTRA+=(--include-chunk-keybert)
fi

nohup "$PYTHON" "$ROOT/experiments/keybert_qa_integration/run.py" \
  --dataset "$DATASET" \
  --output-dir "$OUT_DIR" \
  --static-model "$STATIC_MODEL" \
  --evidence-dir "$EVIDENCE_DIR" \
  --language en \
  --device "$DEVICE" \
  --question "Which task was solved?" \
  --field-id 1 \
  "${EXTRA[@]}" \
  > "$LOG" 2>&1 &

PID=$!
echo "PID=$PID"
echo "log=$LOG"
echo "output=$OUT_DIR"
echo "tail -f $LOG"
