#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/untie_venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
  printf 'Python environment not found: %s\n' "${PYTHON}" >&2
  exit 2
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${ROOT}/artifacts/qa_chunk_selection_v7_rapr/run_${STAMP}"
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--run-dir" && $((i+1)) -lt ${#ARGS[@]} ]]; then
    OUT="${ARGS[$((i+1))]}"
  elif [[ "${ARGS[$i]}" == --run-dir=* ]]; then
    OUT="${ARGS[$i]#--run-dir=}"
  fi
done
LOG_DIR="${ROOT}/logs/qa_chunk_selection_v7"
LOG="${LOG_DIR}/run_${STAMP}.log"
mkdir -p "${LOG_DIR}"
cd "${ROOT}"
nohup env PYTHONUNBUFFERED=1 "${PYTHON}" -m experiments.qa_chunk_selection_v7.run \
  --config "${ROOT}/configs/qa_chunk_selection/v7_rapr/main.yaml" \
  --cells scirex scier_384 scier_128 russerrc scimdix \
  --seeds 42 43 44 --outer-folds 0 1 2 3 4 --inner-folds 4 \
  --run-dir "${OUT}" "${ARGS[@]}" >"${LOG}" 2>&1 &
PID=$!
printf '%s\n' "${PID}" >"${LOG}.pid"
printf 'PID=%s\nOUTPUT=%s\nLOG=%s\n' "${PID}" "${OUT}" "${LOG}"
