#!/usr/bin/env bash
# Background launcher for isolated D4 oracle + D5 audit.
# Writes only to artifacts/qa_chunk_selection_v3_d45/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/untie_venv/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/qa_chunk_selection_v3_d45}"
V3_DIR="${V3_DIR:-$ROOT/artifacts/qa_chunk_selection_v3}"
V2_DIR="${V2_DIR:-$ROOT/artifacts/qa_chunk_selection_v2}"
E0_DIR="${E0_DIR:-$ROOT/artifacts/qa_chunk_selection}"
PREP_DIR="${PREP_DIR:-$ROOT/artifacts/qa_chunk_selection_v3_1_prep}"
AUTO_DIR="${AUTO_DIR:-$ROOT/artifacts/qa_chunk_selection_v3_1_auto}"
RANK_DIR="${RANK_DIR:-$ROOT/artifacts/qa_chunk_selection_v3_1}"
STAGES="${STAGES:-all}"
SMOKE="${SMOKE:-0}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
LOG="${LOG:-$LOG_DIR/$(basename "$OUT_DIR").log}"

V4_DIR="${V4_DIR:-$ROOT/artifacts/qa_chunk_selection_v4}"
for label_dir in "E0:$E0_DIR" "V2:$V2_DIR" "V3:$V3_DIR" "manual V3.1 prep:$PREP_DIR" "automated silver:$AUTO_DIR" "V3.1 ranking:$RANK_DIR" "V4:$V4_DIR"; do
  label="${label_dir%%:*}"
  path="${label_dir#*:}"
  if [[ "$(realpath -m "$OUT_DIR")" == "$(realpath -m "$path")" ]]; then
    echo "Refusing OUT_DIR=$OUT_DIR because it is the $label artifact tree." >&2
    exit 2
  fi
done

mkdir -p "$LOG_DIR" "$OUT_DIR"
EXTRA=()
if [[ "$SMOKE" == "1" ]]; then EXTRA+=(--smoke); fi

nohup "$PYTHON" -m experiments.qa_chunk_selection.v3_d45 \
  --output-dir "$OUT_DIR" \
  --v3-artifact-dir "$V3_DIR" \
  --stages "$STAGES" \
  --device "$DEVICE" \
  --seed "$SEED" \
  "${EXTRA[@]}" \
  > "$LOG" 2>&1 &

PID=$!
echo "PID=$PID"
echo "log=$LOG"
echo "output=$OUT_DIR"
echo "progress=$OUT_DIR/progress.jsonl"
echo "V3 artifacts (read-only)=$V3_DIR"
echo "tail -f $LOG"
