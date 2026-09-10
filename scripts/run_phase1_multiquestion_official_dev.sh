#!/usr/bin/env bash
# Official SciREX DEV multi-question (Q5 then Q3, select thresholds).
# Does not touch legacy hash-split DEV artifacts or frozen_thresholds.json.
# Does not run TEST.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -x "$ROOT/untie_venv/bin/python" ]]; then
  PY="$ROOT/untie_venv/bin/python"
else
  PY="python3"
fi
exec "$PY" -m experiments.phase1_multiquestion.official_dev "$@"
