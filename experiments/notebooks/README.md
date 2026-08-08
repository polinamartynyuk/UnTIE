# Рабочая область для ноутбуков

Размещайте здесь новые исследовательские ноутбуки. Сгенерированные результаты не коммитьте; переиспользуемую логику выносите в `untie/` вместе с тестами.

Существующие ноутбуки остаются в `scripts/` как замороженные legacy-эксперименты, потому что их ячейки содержат абсолютные и относительные пути, которые нужно мигрировать по одному.

## Анализ refactored-результатов

- `04_Analysis_refactored_results.ipynb` — анализ JSON из `artifacts/results_keys_refactored.json`
  и `artifacts/results_keys_rus_refactored.json`; артефакты пишутся в `experiments/analysis_results/`.
- `05_Extraction_metrics_strategy_comparison.ipynb` — оценка качества извлечения
  (`char_f1`, `token_f1`, `rouge_l_f1`, опционально BERTScore) и сравнение стратегий;
  артефакты: `experiments/analysis_results/extraction_metrics/`.
- Вспомогательные функции: `untie/results_analysis.py`, `untie/extraction_metrics.py`.
