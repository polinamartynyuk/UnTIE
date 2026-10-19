#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
stamp="$(date +%Y%m%d_%H%M%S)"
output="${OUTPUT_DIR:-artifacts/qa_chunk_selection_v4_4_external_${stamp}}"
log="${LOG_FILE:-logs/qa_chunk_selection_v4_4_external_${stamp}.log}"
mkdir -p logs
read -r -a datasets <<< "${DATASETS:-scimdix scier_384}"
args=(--datasets "${datasets[@]}" --output-dir "$output" --device "${DEVICE:-auto}")
nohup untie_venv/bin/python -m experiments.qa_chunk_selection.v4_4_external "${args[@]}" >"$log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"${log}.pid"
printf 'PID=%s\nOUTPUT=%s\nLOG=%s\n' "$pid" "$output" "$log"
