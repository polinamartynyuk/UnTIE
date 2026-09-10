#!/usr/bin/env bash
# Background launcher for the isolated KeyBERT topic-clustering experiment.
# Does not write into artifacts/hierarchical_topic_optimized/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/untie_venv/bin/python}"
PIP="${PIP:-$ROOT/untie_venv/bin/pip}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/keybert_topic_clustering/en}"
DATASET="${DATASET:-$ROOT/datasets/scirex_structured.json}"
STATIC_MODEL="${STATIC_MODEL:-$ROOT/experiments/analysis_results/keyword_tuning_task/en/full_v5/scart_tuned_model.json}"
DEVICE="${DEVICE:-auto}"
STAGE="${STAGE:-all}"
QUICK="${QUICK:-0}"
LOG="${LOG:-$LOG_DIR/keybert_topic_clustering.log}"

mkdir -p "$LOG_DIR" "$OUT_DIR"

if ! "$PYTHON" -c "import keybert" >/dev/null 2>&1; then
  echo "Installing keybert==0.9.0 into the project venv..."
  "$PIP" install "keybert==0.9.0"
fi

EXTRA=()
if [[ "$QUICK" == "1" ]]; then
  EXTRA+=(--quick --skip-bertscore)
fi

nohup "$PYTHON" "$ROOT/experiments/keybert_topic_clustering/run.py" \
  --dataset "$DATASET" \
  --output-dir "$OUT_DIR" \
  --static-model "$STATIC_MODEL" \
  --language en \
  --device "$DEVICE" \
  --question "Which task was solved?" \
  --field-id 1 \
  --stage "$STAGE" \
  "${EXTRA[@]}" \
  > "$LOG" 2>&1 &

PID=$!
echo "PID=$PID"
echo "log=$LOG"
echo "output=$OUT_DIR"
echo "tail -f $LOG"
