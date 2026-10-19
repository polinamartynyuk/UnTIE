#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
stamp="$(date +%Y%m%d_%H%M%S)"
output="${1:-artifacts/qa_chunk_selection_v4_4_ru_${stamp}}"
mkdir -p logs
log="logs/v4_4_ru_${stamp}.log"

nohup python3 -u -m experiments.qa_chunk_selection.v4_4_ru \
  --output-dir "$output" \
  >"$log" 2>&1 &

pid=$!
echo "PID=$pid"
echo "LOG=$log"
echo "OUTPUT=$output"
