#!/usr/bin/env bash
# Label remaining official SciREX TRAIN documents.
# Does not rewrite preprocessing, hash-train artifacts, official DEV/TEST, or TEST.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -x "$ROOT/untie_venv/bin/python" ]]; then
  PY="$ROOT/untie_venv/bin/python"
else
  PY="python3"
fi
exec "$PY" -m experiments.phase1_multiquestion.official_train_remainder "$@"
