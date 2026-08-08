#!/usr/bin/env bash
# EN full tuning profile for a larger keyword dictionary (v3).
# Uses cached evidence from artifacts/keyword_tuning_notebooks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source untie_venv/bin/activate

OUT_DIR="experiments/analysis_results/keyword_tuning_task/en/full_large_v3"
mkdir -p "$OUT_DIR"

python scripts/05_Tune_model_keywords.py \
  --language en \
  --field-id 1 \
  --cache-dir artifacts/keyword_tuning_notebooks \
  --max-keywords 8 \
  --min-keywords 4 \
  --max-rescue-keywords 8 \
  --selection-policy union \
  --stability-threshold 0.2 \
  --screen-top-k 60 \
  --min-activation-rate 0.15 \
  --use-conditional-gain \
  --inactive-fallback-penalty 0.04 \
  --activation-weight 0.12 \
  --win-rate-weight 0.08 \
  --harm-cap 0.12 \
  --min-enriched-support 6 \
  --no-enrich-train-references \
  --require-non-empty \
  --output "$OUT_DIR/scart_tuned_model.json" \
  --trace "$OUT_DIR/scart_tuned_model.tuning.json" \
  2>&1 | tee "$OUT_DIR/retune.log"

echo "Artifacts: $OUT_DIR/scart_tuned_model.json"
