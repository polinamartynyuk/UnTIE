#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_DIR"
PYTHON="${PYTHON:-$PROJECT_DIR/untie_venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/artifacts/qa_chunk_selection_v4_4_ru_data_$STAMP}"
LOG="${LOG:-$PROJECT_DIR/logs/qa_chunk_selection_v4_4_ru_data_$STAMP.log}"
DEVICE="${DEVICE:-auto}"

if [[ -e "$OUT_DIR" || -e "$LOG" || -e "$LOG.pid" ]]; then
  printf 'Refusing to overwrite an existing output, log, or PID file.\n' >&2
  exit 2
fi
case "$OUT_DIR" in "$PROJECT_DIR"/artifacts/*) ;; *) printf 'OUT_DIR must be below project/artifacts\n' >&2; exit 2;; esac
case "$LOG" in "$PROJECT_DIR"/logs/*.log) ;; *) printf 'LOG must be a .log file below project/logs\n' >&2; exit 2;; esac

mkdir -p -- "$(dirname -- "$LOG")"
set -o noclobber
exec 3> "$LOG"
nohup "$PYTHON" -B -u -m experiments.qa_chunk_selection.v4_4_ru_data \
  --structured-source "$PROJECT_DIR/datasets/ruserrc_structured.csv" \
  --output-dir "$OUT_DIR" --device "$DEVICE" >&3 2>&1 < /dev/null &
RUN_PID=$!
exec 3>&-
printf '%s\n' "$RUN_PID" > "$LOG.pid"
printf 'Started PID=%s\nOutput: %s\nLog: %s\nWatch: tail -f %q\n' \
  "$RUN_PID" "$OUT_DIR" "$LOG" "$LOG"
