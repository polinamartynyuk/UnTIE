# Рабочая область для ноутбуков

Размещайте здесь новые исследовательские ноутбуки. Сгенерированные результаты не коммитьте; переиспользуемую логику выносите в `untie/` вместе с тестами.

Существующие ноутбуки остаются в `scripts/` как замороженные legacy-эксперименты, потому что их ячейки содержат абсолютные и относительные пути, которые нужно мигрировать по одному.

## Анализ refactored-результатов

- `04_Analysis_refactored_results.ipynb` — анализ JSON из `artifacts/results_keys_refactored.json`
  и `artifacts/results_keys_rus_refactored.json`; артефакты пишутся в `experiments/analysis_results/`.
- Вспомогательные функции: `untie/results_analysis.py` (импорт: `from untie.results_analysis import ...`).
