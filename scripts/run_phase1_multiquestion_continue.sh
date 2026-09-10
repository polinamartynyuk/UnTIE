#!/usr/bin/env bash
# Continue SciREX multi-question Phase 1 from the last finished step.
# Does not rewrite preprocessing, does not run TEST.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec ./untie_venv/bin/python -m experiments.phase1_multiquestion.continue "$@"
