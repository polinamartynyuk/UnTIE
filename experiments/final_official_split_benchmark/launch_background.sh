#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/untie_venv/bin/python3"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${1:-${PROJECT_ROOT}/artifacts/final_official_split_benchmark/run_${STAMP}}"
LOG_DIR="${PROJECT_ROOT}/logs/final_official_split_benchmark"
LOG_FILE="${LOG_DIR}/$(basename "${RUN_DIR}")_pretest.log"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"
cd "${PROJECT_ROOT}"
nohup "${PYTHON_BIN}" -m experiments.final_official_split_benchmark \
  --run-dir "${RUN_DIR}" \
  --phase all \
  --allow-downloads \
  >"${LOG_FILE}" 2>&1 &

echo "PID=$!"
echo "RUN_DIR=${RUN_DIR}"
echo "LOG=${LOG_FILE}"
echo "The pre-TEST run stops at READY_FOR_TEST. Resume with --unlock-test only after reviewing frozen/ and refit/."
