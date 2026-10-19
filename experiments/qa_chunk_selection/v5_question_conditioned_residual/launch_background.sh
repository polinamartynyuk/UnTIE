#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${ROOT}/artifacts/qa_chunk_selection_v5/run_${STAMP}"
LOG="${ROOT}/logs/qa_chunk_selection_v5_${STAMP}.log"
mkdir -p "$(dirname "${OUT}")" "${ROOT}/logs"
cd "${ROOT}"
nohup env PYTHONUNBUFFERED=1 python3 -m experiments.qa_chunk_selection.v5_question_conditioned_residual \
  --output-dir "${OUT}" --phase all "$@" >"${LOG}" 2>&1 &
PID=$!
printf '%s\n' "${PID}" > "${LOG}.pid"
printf 'PID=%s\nOUTPUT=%s\nLOG=%s\n' "${PID}" "${OUT}" "${LOG}"

