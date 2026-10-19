#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/untie_venv/bin/python3"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${PROJECT_ROOT}/artifacts/qa_chunk_selection_v9_h2_acs/run_${STAMP}"
LOG_DIR="${PROJECT_ROOT}/logs/qa_chunk_selection_v9"
LOG_FILE="${LOG_DIR}/run_${STAMP}.log"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python environment not found: %s\n' "${PYTHON_BIN}" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
cd "${PROJECT_ROOT}"
nohup "${PYTHON_BIN}" -u -m experiments.qa_chunk_selection_v9 \
  --run-dir "${RUN_DIR}" --phase all \
  >"${LOG_FILE}" 2>&1 &
PID=$!
printf '%s\n' "${PID}" >"${LOG_FILE}.pid"
printf 'PID=%s\nRUN_DIR=%s\nLOG=%s\n' "${PID}" "${RUN_DIR}" "${LOG_FILE}"

