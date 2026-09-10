#!/usr/bin/env bash
# Background launcher for isolated SciREX chunk-selection V3.
# Writes only to artifacts/qa_chunk_selection_v3/. Never overwrites E0 or V2.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/untie_venv/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/qa_chunk_selection_v3}"
V2_DIR="${V2_DIR:-$ROOT/artifacts/qa_chunk_selection_v2}"
E0_DIR="${E0_DIR:-$ROOT/artifacts/qa_chunk_selection}"
DEVICE="${DEVICE:-auto}"
STAGES="${STAGES:-all}"
SMOKE="${SMOKE:-0}"
SKIP_QA="${SKIP_QA:-0}"
RESUME="${RESUME:-0}"
LIMIT_DOCS="${LIMIT_DOCS:-0}"
SEED="${SEED:-42}"
ALLOW_INTERNAL_TEST="${ALLOW_INTERNAL_TEST:-0}"
LOG="${LOG:-$LOG_DIR/qa_chunk_selection_v3.log}"

if [[ "$(realpath -m "$OUT_DIR")" == "$(realpath -m "$E0_DIR")" ]]; then
  echo "Refusing OUT_DIR=$OUT_DIR because it is the E0 artifact tree." >&2
  exit 2
fi
if [[ "$(realpath -m "$OUT_DIR")" == "$(realpath -m "$V2_DIR")" ]]; then
  echo "Refusing OUT_DIR=$OUT_DIR because it is the V2 artifact tree." >&2
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
if [[ "$SKIP_QA" == "1" ]]; then
  EXTRA+=(--skip-qa)
fi
if [[ "$RESUME" == "1" ]]; then
  EXTRA+=(--resume)
fi
if [[ "$LIMIT_DOCS" != "0" ]]; then
  EXTRA+=(--limit-docs "$LIMIT_DOCS")
fi
if [[ "$ALLOW_INTERNAL_TEST" == "1" ]]; then
  EXTRA+=(--allow-internal-test)
fi

nohup "$PYTHON" -m experiments.qa_chunk_selection.v3 \
  --output-dir "$OUT_DIR" \
  --v2-artifact-dir "$V2_DIR" \
  --e0-artifact-dir "$E0_DIR" \
  --device "$DEVICE" \
  --stages "$STAGES" \
  --seed "$SEED" \
  "${EXTRA[@]}" \
  > "$LOG" 2>&1 &

PID=$!
echo "PID=$PID"
echo "log=$LOG"
echo "output=$OUT_DIR"
echo "progress=$OUT_DIR/progress.jsonl"
echo "V2 artifacts (read-only)=$V2_DIR"
echo "E0 artifacts (read-only)=$E0_DIR"
echo "tail -f $LOG"
