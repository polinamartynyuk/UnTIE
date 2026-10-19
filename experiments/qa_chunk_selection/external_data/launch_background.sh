#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
stamp="$(date +%Y%m%d_%H%M%S)"
output="${OUTPUT_DIR:-artifacts/qa_chunk_selection_external_data_${stamp}}"
log="${LOG_FILE:-logs/qa_chunk_selection_external_data_${stamp}.log}"
mkdir -p logs
args=(--output-dir "$output" --device "${DEVICE:-auto}")
if [[ -n "${SCIMDIX_SOURCE:-}" && -n "${SCIER_SOURCE:-}" ]]; then
  args+=(--no-download --scimdix-source "$SCIMDIX_SOURCE" --scier-source "$SCIER_SOURCE")
fi
nohup untie_venv/bin/python -m experiments.qa_chunk_selection.external_data "${args[@]}" >"$log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"${log}.pid"
printf 'PID=%s\nOUTPUT=%s\nLOG=%s\n' "$pid" "$output" "$log"
