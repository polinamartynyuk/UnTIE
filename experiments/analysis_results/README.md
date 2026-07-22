# Результаты анализа экспериментов

Сюда сохраняются артефакты из ноутбука `experiments/notebooks/04_Analysis_refactored_results.ipynb`:

- `metrics_summary_<lang>.csv` — сводка по 27 комбинациям стратегий
- `metrics_summary_evaluable_<lang>.csv` — та же сводка только для строк,
  где было не менее двух различных кандидатов ответа
- `strategy_stats_<lang>.csv` — агрегаты по отдельным типам стратегий
- `figures/` — PNG-графики (стратегии, heatmap, топ комбинаций)
- `statistical_tests.csv` — результаты непараметрических тестов (если доступны зависимости)
- `en_ru_strategy_comparison.csv` — сравнение EN и RU по комбинациям стратегий

Файлы генерируются локально и не коммитятся в Git.
