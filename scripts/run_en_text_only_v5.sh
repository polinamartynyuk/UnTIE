#!/usr/bin/env bash
# EN tuning v5: text-only, relaxed filtering for a larger final dictionary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source untie_venv/bin/activate

OUT_DIR="experiments/analysis_results/keyword_tuning_task/en/full_v5"
mkdir -p "$OUT_DIR"

python scripts/05_Tune_model_keywords.py \
  --language en \
  --field-id 1 \
  --cache-dir artifacts/keyword_tuning_notebooks \
  --no-enrich-train-references \
  --min-document-support 1 \
  --max-candidates 200 \
  --max-keywords 10 \
  --min-keywords 6 \
  --max-rescue-keywords 12 \
  --selection-policy union \
  --stability-threshold 0.15 \
  --screen-top-k 80 \
  --min-activation-rate 0.10 \
  --use-conditional-gain \
  --inactive-fallback-penalty 0.03 \
  --activation-weight 0.10 \
  --win-rate-weight 0.06 \
  --size-penalty 0.001 \
  --harm-cap 0.18 \
  --require-non-empty \
  --output "$OUT_DIR/scart_tuned_model.json" \
  --trace "$OUT_DIR/scart_tuned_model.tuning.json" \
  2>&1 | tee "$OUT_DIR/retune.log"

echo "Artifacts: $OUT_DIR/scart_tuned_model.json"
