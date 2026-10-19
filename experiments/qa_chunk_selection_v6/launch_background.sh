#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${ROOT}/artifacts/qa_chunk_selection_v6_universal/run_${STAMP}"
LOG_DIR="${ROOT}/logs/qa_chunk_selection_v6"
LOG="${LOG_DIR}/run_${STAMP}.log"
mkdir -p "${LOG_DIR}"
cd "${ROOT}"
nohup env PYTHONUNBUFFERED=1 "${ROOT}/untie_venv/bin/python3" -m experiments.qa_chunk_selection_v6.run_dev \
  --config "${ROOT}/configs/qa_chunk_selection/v6_universal/main.yaml" \
  --datasets scirex scier_384 scier_128 russerrc scimdix \
  --seeds 42 43 44 --outer-folds 5 --inner-meta-folds 4 \
  --run-dir "${OUT}" "$@" >"${LOG}" 2>&1 &
PID=$!
printf '%s\n' "${PID}" >"${LOG}.pid"
printf 'PID=%s\nOUTPUT=%s\nLOG=%s\n' "${PID}" "${OUT}" "${LOG}"

