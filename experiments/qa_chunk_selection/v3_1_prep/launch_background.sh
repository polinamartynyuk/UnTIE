#!/usr/bin/env bash
# Background launcher for isolated V3.1 overlay data preparation.
# Writes only to artifacts/qa_chunk_selection_v3_1_prep/.
# Never overwrites E0, V2, or V3 experiment artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/untie_venv/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/qa_chunk_selection_v3_1_prep}"
V3_DIR="${V3_DIR:-$ROOT/artifacts/qa_chunk_selection_v3}"
V2_DIR="${V2_DIR:-$ROOT/artifacts/qa_chunk_selection_v2}"
E0_DIR="${E0_DIR:-$ROOT/artifacts/qa_chunk_selection}"
STAGES="${STAGES:-all}"
SMOKE="${SMOKE:-0}"
SEED="${SEED:-42}"
ANNOTATOR_ID="${ANNOTATOR_ID:-annotator_1}"
COVERAGE_N="${COVERAGE_COMPLETION_N:-0}"
ALLOW_PARTIAL_FREEZE="${ALLOW_PARTIAL_FREEZE:-0}"
LOG="${LOG:-$LOG_DIR/qa_chunk_selection_v3_1_prep.log}"

if [[ "$(realpath -m "$OUT_DIR")" == "$(realpath -m "$E0_DIR")" ]]; then
  echo "Refusing OUT_DIR=$OUT_DIR because it is the E0 artifact tree." >&2
  exit 2
fi
if [[ "$(realpath -m "$OUT_DIR")" == "$(realpath -m "$V2_DIR")" ]]; then
  echo "Refusing OUT_DIR=$OUT_DIR because it is the V2 artifact tree." >&2
  exit 2
fi
if [[ "$(realpath -m "$OUT_DIR")" == "$(realpath -m "$V3_DIR")" ]]; then
  echo "Refusing OUT_DIR=$OUT_DIR because it is the V3 artifact tree." >&2
  exit 2
fi
if [[ "$(realpath -m "$OUT_DIR")" == "$(realpath -m "$ROOT/artifacts/qa_chunk_selection_v4")" ]]; then
  echo "Refusing OUT_DIR=$OUT_DIR because it is the V4 artifact tree." >&2
  exit 2
fi

mkdir -p "$LOG_DIR" "$OUT_DIR"

EXTRA=()
if [[ "$SMOKE" == "1" ]]; then
  EXTRA+=(--smoke)
fi
if [[ "$COVERAGE_N" != "0" ]]; then
  EXTRA+=(--coverage-completion-n "$COVERAGE_N")
fi
if [[ "$ALLOW_PARTIAL_FREEZE" == "1" ]]; then
  EXTRA+=(--allow-partial-freeze)
fi

nohup "$PYTHON" -m experiments.qa_chunk_selection.v3_1_prep \
  --output-dir "$OUT_DIR" \
  --v3-artifact-dir "$V3_DIR" \
  --stages "$STAGES" \
  --seed "$SEED" \
  --annotator-id "$ANNOTATOR_ID" \
  "${EXTRA[@]}" \
  > "$LOG" 2>&1 &

PID=$!
echo "PID=$PID"
echo "log=$LOG"
echo "output=$OUT_DIR"
echo "progress=$OUT_DIR/progress.jsonl"
echo "V3 artifacts (read-only)=$V3_DIR"
echo "tail -f $LOG"
