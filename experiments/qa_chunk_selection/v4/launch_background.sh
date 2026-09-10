#!/usr/bin/env bash
# Background launcher for isolated V4 (7 strategies).
# Writes only to artifacts/qa_chunk_selection_v4/ by default.
# Never overwrites E0, V2, V3, V3.1, or D4/D5 trees.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/untie_venv/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/qa_chunk_selection_v4}"
V3_DIR="${V3_DIR:-$ROOT/artifacts/qa_chunk_selection_v3}"
V2_DIR="${V2_DIR:-$ROOT/artifacts/qa_chunk_selection_v2}"
E0_DIR="${E0_DIR:-$ROOT/artifacts/qa_chunk_selection}"
PREP_DIR="${PREP_DIR:-$ROOT/artifacts/qa_chunk_selection_v3_1_prep}"
AUTO_DIR="${AUTO_DIR:-$ROOT/artifacts/qa_chunk_selection_v3_1_auto}"
AUTO_V2_DIR="${AUTO_V2_DIR:-$ROOT/artifacts/qa_chunk_selection_v3_1_auto_v2}"
RANK_DIR="${RANK_DIR:-$ROOT/artifacts/qa_chunk_selection_v3_1}"
D45_DIR="${D45_DIR:-$ROOT/artifacts/qa_chunk_selection_v3_d45}"
STAGES="${STAGES:-phase1}"
SMOKE="${SMOKE:-0}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
SKIP_QA="${SKIP_QA:-0}"
ALLOW_EXISTING_OUT="${ALLOW_EXISTING_OUT:-0}"
LOG="${LOG:-$LOG_DIR/qa_chunk_selection_v4.log}"

for label_dir in \
  "E0:$E0_DIR" \
  "V2:$V2_DIR" \
  "V3:$V3_DIR" \
  "manual V3.1 prep:$PREP_DIR" \
  "automated silver:$AUTO_DIR" \
  "automated silver v2:$AUTO_V2_DIR" \
  "V3.1 ranking:$RANK_DIR" \
  "D4/D5:$D45_DIR"
do
  label="${label_dir%%:*}"
  path="${label_dir#*:}"
  if [[ "$(realpath -m "$OUT_DIR")" == "$(realpath -m "$path")" ]]; then
    echo "Refusing OUT_DIR=$OUT_DIR because it is the $label artifact tree." >&2
    exit 2
  fi
done

if [[ ( -f "$OUT_DIR/analysis/compare.json" || -f "$OUT_DIR/reports/results_v4.md" ) && "$ALLOW_EXISTING_OUT" != "1" ]]; then
  echo "OUT_DIR=$OUT_DIR already has a completed V4 run." >&2
  echo "Set a new OUT_DIR so the previous run is not overwritten, or ALLOW_EXISTING_OUT=1." >&2
  exit 2
fi

mkdir -p "$LOG_DIR" "$OUT_DIR"
EXTRA=()
if [[ "$SMOKE" == "1" ]]; then EXTRA+=(--smoke); fi
if [[ "$SKIP_QA" == "1" ]]; then EXTRA+=(--skip-qa); fi
if [[ "$ALLOW_EXISTING_OUT" == "1" ]]; then EXTRA+=(--allow-existing-output); fi

nohup "$PYTHON" -m experiments.qa_chunk_selection.v4 \
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
echo "stages=$STAGES"
echo "tail -f $LOG"
