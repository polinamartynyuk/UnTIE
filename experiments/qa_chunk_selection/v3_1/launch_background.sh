#!/usr/bin/env bash
# Background launcher for isolated V3.1-Silver ranking.
# Writes only to artifacts/qa_chunk_selection_v3_1/.
# Requires a frozen automated silver overlay unless SMOKE=1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/untie_venv/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/qa_chunk_selection_v3_1}"
V3_DIR="${V3_DIR:-$ROOT/artifacts/qa_chunk_selection_v3}"
V2_DIR="${V2_DIR:-$ROOT/artifacts/qa_chunk_selection_v2}"
E0_DIR="${E0_DIR:-$ROOT/artifacts/qa_chunk_selection}"
PREP_DIR="${PREP_DIR:-$ROOT/artifacts/qa_chunk_selection_v3_1_prep}"
SILVER_DIR="${SILVER_DIR:-$ROOT/artifacts/qa_chunk_selection_v3_1_auto}"
DEVICE="${DEVICE:-auto}"
STAGES="${STAGES:-all}"
SMOKE="${SMOKE:-0}"
SKIP_QA="${SKIP_QA:-0}"
RESUME="${RESUME:-0}"
SEED="${SEED:-42}"
LOG="${LOG:-$LOG_DIR/qa_chunk_selection_v3_1.log}"

for label_dir in \
  "E0:$E0_DIR" \
  "V2:$V2_DIR" \
  "V3:$V3_DIR" \
  "manual V3.1 prep:$PREP_DIR" \
  "automated silver:$SILVER_DIR"
do
  label="${label_dir%%:*}"
  path="${label_dir#*:}"
  if [[ "$(realpath -m "$OUT_DIR")" == "$(realpath -m "$path")" ]]; then
    echo "Refusing OUT_DIR=$OUT_DIR because it is the $label artifact tree." >&2
    exit 2
  fi
done

if [[ "$SMOKE" != "1" ]]; then
  FROZEN="$SILVER_DIR/annotation_qa_silver/silver_manifest.json"
  if [[ ! -f "$FROZEN" ]]; then
    echo "Frozen QA-silver overlay missing: $FROZEN" >&2
    echo "Run v3_1_auto_prep with STAGES=full first." >&2
    exit 2
  fi
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

nohup "$PYTHON" -m experiments.qa_chunk_selection.v3_1 \
  --output-dir "$OUT_DIR" \
  --v3-artifact-dir "$V3_DIR" \
  --v2-artifact-dir "$V2_DIR" \
  --silver-dir "$SILVER_DIR" \
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
echo "V3 artifacts (read-only)=$V3_DIR"
echo "silver overlay (read-only)=$SILVER_DIR"
echo "tail -f $LOG"
