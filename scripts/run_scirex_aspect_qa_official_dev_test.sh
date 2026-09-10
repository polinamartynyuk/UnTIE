#!/usr/bin/env bash
# Aspect-QA preprocessing for official SciREX dev and test.
# Does not touch datasets/scirex_structured.json or datasets/scirex_aspect_qa_preprocessing/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -x "$ROOT/untie_venv/bin/python" ]]; then
  PY="$ROOT/untie_venv/bin/python"
else
  PY="python3"
fi

"$PY" -m untie.aspect_qa \
  --input datasets/scirex_dev_structured.json \
  --output datasets/scirex_aspect_qa_preprocessing_dev \
  --fixed-split dev \
  --tokenizer qa_model \
  --device cpu

"$PY" -m untie.aspect_qa \
  --input datasets/scirex_test_structured.json \
  --output datasets/scirex_aspect_qa_preprocessing_test \
  --fixed-split test \
  --tokenizer qa_model \
  --device cpu
