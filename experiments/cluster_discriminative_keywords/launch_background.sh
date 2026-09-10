#!/usr/bin/env bash
# Background launcher for cluster-specific discriminative QA keywords (hyp4).
# Frozen clustering: C_keybert. Cached RoBERTa evidence. Does not write into
# hierarchical / KeyBERT / answer_aggregation artifact trees.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/untie_venv/bin/python}"
PIP="${PIP:-$ROOT/untie_venv/bin/pip}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/cluster_discriminative_keywords/en}"
DATASET="${DATASET:-$ROOT/datasets/scirex_structured.json}"
STATIC_MODEL="${STATIC_MODEL:-$ROOT/experiments/analysis_results/keyword_tuning_task/en/full_v5/scart_tuned_model.json}"
EVIDENCE_DIR="${EVIDENCE_DIR:-$ROOT/artifacts/keybert_topic_clustering/en_full/evidence}"
CLUSTER_DIR="${CLUSTER_DIR:-$ROOT/artifacts/keybert_topic_clustering/en_full}"
DEVICE="${DEVICE:-auto}"
QUICK="${QUICK:-0}"
SKIP_BERTSCORE="${SKIP_BERTSCORE:-0}"
RESUME="${RESUME:-0}"
QUESTION="${QUESTION:-Which task was solved?}"
LOG="${LOG:-$LOG_DIR/cluster_discriminative_keywords.log}"

mkdir -p "$LOG_DIR" "$OUT_DIR"

if ! "$PYTHON" -c "import keybert" >/dev/null 2>&1; then
  echo "Installing keybert==0.9.0 into the project venv..."
  "$PIP" install "keybert==0.9.0"
fi

EXTRA=()
if [[ "$QUICK" == "1" ]]; then
  EXTRA+=(--quick --skip-bertscore)
fi
if [[ "$SKIP_BERTSCORE" == "1" ]]; then
  EXTRA+=(--skip-bertscore)
fi
if [[ "$RESUME" == "1" ]]; then
  EXTRA+=(--resume)
fi

nohup "$PYTHON" "$ROOT/experiments/cluster_discriminative_keywords/run.py" \
  --dataset "$DATASET" \
  --output-dir "$OUT_DIR" \
  --evidence-dir "$EVIDENCE_DIR" \
  --cluster-dir "$CLUSTER_DIR" \
  --static-model "$STATIC_MODEL" \
  --language en \
  --device "$DEVICE" \
  --question "$QUESTION" \
  --field-id 1 \
  "${EXTRA[@]}" \
  > "$LOG" 2>&1 &

PID=$!
echo "PID=$PID"
echo "log=$LOG"
echo "output=$OUT_DIR"
echo "tail -f $LOG"
