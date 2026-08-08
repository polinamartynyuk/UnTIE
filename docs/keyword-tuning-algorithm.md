# Настройка статических ключевых слов для извлечения информации

## 1. Назначение

Алгоритм подбирает для тематического аспекта небольшой общий словарь ключевых
слов, который повышает качество extractive QA на коллекции документов. После
настройки словарь и выбранная стратегия переранжирования сохраняются в JSON
модели. При обычном извлечении эталонный ответ и attention-анализ конкретного
документа уже не нужны.

Поддерживаются три режима:

- `baseline` — QA по всем чанкам без ключевых слов;
- `attention` — ключевые слова строятся для каждого документа с использованием
  эталонного ответа;
- `static-keywords` — чанки переранжируются по заранее настроенному словарю.

Настройка отделяет дорогой сбор evidence от дешёвого перебора словарей. QA и
attention выполняются один раз и кэшируются; последующие запуски поиска
используют кэш ответов.

## 2. Общая схема

```mermaid
flowchart TD
    Dataset[Датасет с текстами и эталонами] --> Split[Детерминированный split 60/20/20]
    InitModel[Начальная модель аспекта] --> Evidence
    Split --> Evidence[Сбор evidence и QA-кэш]
    Evidence --> TrainCandidates[Кандидаты только из train]
    TrainCandidates --> Pool[Агрегация и фильтрация пула]
    Pool --> Strategies[Сетка стратегий reranking]
    Strategies --> SFFS[Multi-fidelity SFFS]
    SFFS --> Stability[Stability selection]
    Stability --> DevChoice[Выбор по dev]
    DevChoice --> TestEval[Однократная оценка на test]
    TestEval --> TunedModel[Tuned model JSON]
    TestEval --> Trace[Trace JSON и strategies CSV]
```

Основная точка входа — `scripts/05_Tune_model_keywords.py`. Реализация
распределена по модулям:

- `untie.keyword_evidence` — сбор и хранение evidence, кэш метрик, cached
  reranking;
- `untie.keyword_tuning` — split, objective, панели и SFFS;
- `untie.keyword_training` — стратегии, stability selection и оркестрация;
- `untie.extraction_metrics` — метрики извлечения;
- `untie.model_params` — схема и атомарное сохранение модели.

## 3. Входные данные

### 3.1. Английский язык

По умолчанию используется `datasets/scirex_structured.json`:

- `doc_id` — идентификатор документа;
- `original_text` — исходный текст;
- `tasks` или `tasks_cleaned` — один или несколько эталонных ответов.

Подчёркивания в `tasks` заменяются пробелами. Параметры аспекта берутся из
`model_params/scart_init_model.json`; текущий аспект — `task`, вопрос:
`Which task was solved?`.

### 3.2. Русский язык

По умолчанию используется `datasets/ruserrc_structured.csv` с разделителем
`;`:

- `id` или `doc_id`;
- `text_clean`, `original_text` или `text`;
- `Task_aspects` или `tasks_cleaned`.

Строковое представление списка эталонов разбирается через `ast.literal_eval`.
Параметры аспекта берутся из `model_params/ruserrc_init_model.json`; вопрос:
`Какая задача была решена?`.

Строки без эталонного ответа исключаются. Сейчас один запуск настраивает одно
поле модели и использует первый вопрос этого поля.

## 4. Детерминированное разделение

`deterministic_document_split` формирует train/dev/test в пропорции 60/20/20.
Документы сортируются по SHA-256 от `seed` и `doc_id`, поэтому результат:

- не зависит от порядка строк датасета;
- повторяется при том же seed;
- не допускает пересечений частей.

Назначение частей:

- **train** — единственный источник кандидатов и их морфологических форм;
- **dev** — поиск словаря и выбор стратегии;
- **test** — только финальная оценка выбранного решения.

Это принципиальная защита от leakage: слова из dev/test не могут попасть в пул
кандидатов, а test не участвует в выборе конфигурации.

## 5. Сбор evidence

Для каждого документа `collect_document_evidence` выполняет:

1. разбиение на предложения и перекрывающиеся чанки;
2. extractive QA для каждого чанка;
3. агрегацию baseline-ответа;
4. сохранение ответа, confidence и позиций span для каждого чанка;
5. вычисление матрицы семантического сходства ответов.

Только для train-документов дополнительно выполняется mining кандидатов:

1. ответы сравниваются со всеми эталонами;
2. из валидных чанков извлекаются слова с высоким attention;
3. динамический IDF-фильтр удаляет равномерно распределённые слова;
4. дубликаты объединяются по максимальному attention weight;
5. удаляются stop words;
6. сохраняются слова, которые семантически ближе к описанию аспекта, чем к
   эталонному ответу;
7. для каждого слова записываются lemma, stem, attention weight,
   score difference и индексы поддерживающих чанков.

### 5.1. Кэш и fingerprint

Evidence хранится в:

```text
{cache_dir}/{language}/field-{field_id}/evidence/
```

Fingerprint включает текст, идентификатор аспекта, вопрос, эталоны, признак
сбора кандидатов, модели, chunking, `attention_top_k` и версию пайплайна.
Изменение любого из этих параметров приводит к безопасному cache miss.

QA-кэш позволяет проверять тысячи подмножеств слов без повторного запуска
transformer-модели. Кэш не является результатом настройки и может быть удалён:
он будет построен заново.

## 6. Формирование пула кандидатов

`aggregate_candidate_pool` работает только с train evidence:

- surface forms нормализуются (`casefold`, пробелы, пунктуация);
- кандидаты ниже `min_document_support` удаляются;
- attention и score difference агрегируются медианой по документам, что
  уменьшает влияние выбросов;
- кандидаты сортируются по document support, score difference, attention и
  имени;
- размер ограничивается `max_candidates`.

Для итогового `WeightedKeyword` наиболее частая пара `(lemma, stem)` выбирается
голосованием по train-документам.

## 7. Переранжирование и стратегии

`score_chunks` ищет lemma/stem ключевых слов в чанках. Оценка учитывает:

- комбинированный вес слова;
- позицию совпадения;
- частоту совпадений;
- долю уникальных совпавших ключей.

Вес слова задаётся формулой:

```text
keyword_weight =
    attention_weight × weight_ratio
    + score_difference × (1 − weight_ratio)
```

Настройка проверяет 27 комбинаций:

1. скоринг чанков:
   - `only_score_diff` (`weight_ratio = 0`);
   - `only_weight` (`weight_ratio = 1`);
   - `equal_weight_score_diff` (`weight_ratio = 0.5`);
2. выбор кластера ответов:
   - `highest_avg_score`;
   - `weighted_score`;
   - `highest_cohesion`;
3. выбор ответа внутри кластера:
   - `highest_chunk_score`;
   - `highest_similarity`;
   - `combined_score`.

Если ни один чанк не содержит ключей, используется baseline-ответ и отмечается
`fallback=True`.

## 8. Метрики качества

Для prediction и набора эталонов вычисляются:

- `char_f1` — посимвольный F1, диапазон `[0, 1]`;
- `token_f1` — SQuAD token F1, диапазон `[0, 100]`;
- `rouge_l_f1` — ROUGE-L F1, диапазон `[0, 1]`;
- `bertscore_p`, `bertscore_r`, `bertscore_f1` — опциональные BERTScore.

При нескольких эталонах берётся максимальный результат. Внутренний composite:

```text
quality = Σ normalized_metric_weight × normalized_metric
```

`token_f1` перед объединением делится на 100. Без BERTScore три веса
перенормируются до суммы 1; с BERTScore участвуют четыре метрики.

## 9. Целевая функция

Для каждого документа вычисляется:

```text
delta = tuned_quality − baseline_quality
```

Далее:

```text
objective =
    mean_gain
    + confidence_weight × bootstrap_lower_bound
    − downside_penalty × mean(max(0, −delta))
    − harm_penalty × harm_rate
    − fallback_penalty × fallback_rate
    − size_penalty × number_of_keywords
```

По умолчанию orchestration использует:

- `downside_penalty = 0.75`;
- `harm_penalty = 0.5`;
- `fallback_penalty = 0.1`;
- `size_penalty = 0.002`;
- `harm_threshold = 0.01`;
- `confidence_weight = 0.25`.

Таким образом, оптимизируется не только среднее улучшение: отдельно штрафуются
ухудшения, нестабильность, частый fallback и слишком большой словарь.

## 10. Critical и guard документы

На dev выделяются:

- **critical** — baseline quality ниже 0.65, документ содержит несколько
  чанков и хотя бы один кандидат;
- **guard** — детерминированная подвыборка хорошо обрабатываемых документов.

Critical-документы направляют поиск на реальные ошибки baseline, а guard
защищают уже правильные ответы от регрессии.

## 11. Multi-fidelity SFFS

Для каждой стратегии запускается Sequential Forward Floating Selection:

1. forward-шаг добавляет лучший кандидат;
2. successive halving продвигает лучшие варианты от малой панели к полной;
3. floating backward удаляет ставшие вредными слова;
4. поиск завершается по `max_keywords`, `evaluation_budget`, `patience` или
   сходимости.

Панели составляют примерно 25%, 50% и 100% текущей stability-подвыборки dev.
Это позволяет быстро отбрасывать слабые варианты, но подтверждать финальные
решения на большем числе документов.

Checkpoint каждого запуска сохраняется в:

```text
{cache_dir}/{language}/field-{field_id}/checkpoints/
```

Checkpoint используется только при совпадении fingerprint.

## 12. Stability selection и политики финального отбора

SFFS повторяется `stability_runs` раз на детерминированных 80%-подвыборках dev.
После SFFS применяется политика финального отбора (`--selection-policy`):

| Политика | Поведение |
|---|---|
| `strict` | Частота выбора не ниже `stability_threshold`, затем backward pruning |
| `relaxed` | То же с более низким порогом (по умолчанию 0.4) |
| `union` | Объединение всех run-selections, затем pruning |
| `best_run` | Лучший одиночный stability run по dev objective |
| `frequency_top_k` | Top-k слов по частоте выбора в runs |

**Non-empty rescue guard** (`--require-non-empty`, по умолчанию включён):
если stability даёт пустой набор, но существует непустой кандидат с
`mean_gain > 0` и `harm_rate <= harm_cap`, он сохраняется вместо пустого
словаря.

## 12.1 Prescreening кандидатов

Перед SFFS каждый терм из pool оценивается по одиночке на full dev
(`--screen-top-k`, по умолчанию 40). Это сокращает перебор generic-слов и
оставляет бюджет для комбинаций из 2–3 ключевых слов.

## 12.2 Objective, fallback и activation

Штраф `fallback_penalty` применяется только к **вредному fallback** (delta < 0).
Для непустого subset дополнительно штрафуется **idle fallback** — когда keywords
не матчат чанки (`inactive_fallback_penalty * fallback_rate`).

Activation-aware objective (balanced defaults):

```text
gain = mean_gain_active   # средний delta только на docs с fallback=False
objective = gain
  + activation_weight * activation_rate
  + win_rate_weight * win_rate
  - inactive_fallback_penalty * fallback_rate
  - harm_penalty * harm_rate
  - ...
```

Hard gate: если `activation_rate < min_activation_rate` (default 0.20) для
непустого subset, objective = −∞. Это не даёт выбирать «редкие безопасные»
keywords с fallback ~84%.

При `--use-conditional-gain` (default ON) tuning оптимизирует **реальное
улучшение на active docs**, а не mean_gain, размазанный по tie/fallback.

## 12.3 Candidate pool (только текст документов)

По умолчанию pool строится **только** из attention/QA evidence, собранного из
текста train-документов. Gold task labels **не** добавляются в pool.

- Фильтрация generic-глаголов (`strive`, `formulate`, `hypotheses`, …)
- `chunk_support_rate > 0` — терм должен матчить хотя бы один чанк
- Опциональный relaxed contrast при сборе evidence (`--relaxed-contrast`)

Опционально (не рекомендуется): `--enrich-train-references` добавляет n-grams из
train gold references. Это подмешивает task labels, а не текстовые сигналы.

После stability selection выполняется backward pruning на полном dev. Лучшая
стратегия выбирается лексикографически по `mean_gain_active`, `activation_rate`,
меньшему `harm_rate`, затем `objective` и имени стратегии.

## 13. Финальная test-оценка и ablation

После выбора словаря и стратегии test используется один раз. Сравниваются:

- `empty_baseline` — пустой словарь;
- `frequency_only` — верхние слова по частоте/support;
- `floating_tuned` — результат SFFS и stability selection.

`release_recommended` устанавливается, если:

```text
test_mean_gain > 0
and test_confidence_lower_bound >= 0
and test_harm_rate <= 0.1
```

Флаг является автоматическим gate, а не гарантией production-качества.

## 14. Сохранение модели

Начальный JSON не изменяется. Создаётся глубокая копия, где у выбранного поля:

- `keywords` — упорядоченный список слов;
- `keyword_metadata` — lemma, stem, attention weight, score difference,
  document support, selection frequency и marginal gain;
- `tuning_metadata` — стратегия, dev/test показатели, stability, split hash,
  ablations, fingerprint и release flag.

`save_model_params_atomic` пишет временный файл и заменяет destination только
после успешной сериализации. Неизвестные vendor-поля модели сохраняются.
Legacy-ключи в виде объектов также поддерживаются при чтении.

Дополнительные артефакты:

- `*.tuning.json` — полный trace настройки;
- `*.tuning.strategies.csv` — сравнение стратегий;
- `extraction_metrics.json` — кэш метрик prediction/reference.

## 15. Запуск

Полный английский прогон:

```bash
python scripts/05_Tune_model_keywords.py \
  --language en \
  --include-bertscore
```

Полный русский прогон:

```bash
python scripts/05_Tune_model_keywords.py \
  --language ru \
  --include-bertscore
```

Быстрая проверка одной стратегии на подвыборке:

```bash
python scripts/05_Tune_model_keywords.py \
  --language en \
  --limit 20 \
  --max-candidates 20 \
  --max-keywords 5 \
  --evaluation-budget 30 \
  --stability-runs 2 \
  --strategy equal_weight_score_diff weighted_score combined_score \
  --output artifacts/scart_tuned_demo.json
```

`--strategy` можно повторить несколько раз. В этом случае настройка проверит
только перечисленные комбинации и выберет лучшую по dev objective:

```bash
python scripts/05_Tune_model_keywords.py \
  --language en \
  --strategy equal_weight_score_diff weighted_score combined_score \
  --strategy only_score_diff highest_avg_score highest_similarity
```

Inference после настройки:

```bash
python -m untie.cli article.txt \
  --language en \
  --mode static-keywords \
  --model-params model_params/scart_tuned_model.json \
  --question "Which task was solved?"
```

## 16. Основные параметры

| Параметр | Default | Значение |
|---|---:|---|
| `--seed` | 42 | split, bootstrap, панели |
| `--attention-top-k` | 100 | attention-кандидаты из чанка |
| `--min-document-support` | 2 | минимум train-документов |
| `--max-candidates` | 150 | максимальный пул |
| `--max-keywords` | 20 | максимальный словарь |
| `--evaluation-budget` | 250 | evaluations на SFFS run |
| `--patience` | 2 | шаги без улучшения |
| `--beam-width` | 5 | ширина forward beam |
| `--stability-runs` | 5 | число повторов |
| `--stability-threshold` | 0.4 | минимальная частота выбора |
| `--selection-policy` | relaxed | политика финального отбора |
| `--require-non-empty` | true | не возвращать пустой словарь, если есть лучший кандидат |
| `--harm-cap` | 0.12 | максимальный harm rate для rescue и release gate |
| `--activation-weight` | 0.15 | бонус за activation rate (1 − fallback) |
| `--min-activation-rate` | 0.20 | hard gate: минимальная доля active docs |
| `--inactive-fallback-penalty` | 0.05 | штраф idle fallback при непустом subset |
| `--use-conditional-gain` | true | mean_gain только на active docs |
| `--win-rate-weight` | 0.10 | бонус за долю docs с delta > harm_threshold |
| `--min-enriched-support` | 8 | минимальный document_support для enriched terms |
| `--max-rescue-keywords` | 5 | максимум терминов в rescue/frequency fallback |
| `--min-keywords` | 1 | нижняя граница backward pruning (для `full_large` — 4) |
| `--screen-top-k` | 40 | число кандидатов после prescreening |
| `--tuning-exact-match` | true | exact-match веса для tuning objective |
| `--enrich-train-references` | false | добавлять train gold n-grams в pool (не рекомендуется) |
| `--relaxed-contrast` | true | мягкий contrast при сборе evidence |
| `--include-bertscore` | false | включить BERTScore в objective |

Для RU по умолчанию используются чанки 128/24, для EN — 384/50.

## 17. Ограничения и диагностика

- Полный grid `27 × stability_runs × evaluation_budget` может выполняться
  много часов; начинать следует с sample-режима.
- BERTScore заметно увеличивает время и требует optional-зависимостей.
- RU tuning требует `pymorphy3`; sentence encoder может потребовать заранее
  скачанные веса.
- При `Candidate pool is empty` следует увеличить выборку, снизить
  `min_document_support` или проверить attention/валидацию.
- `stop_reason=evaluation_budget` означает корректную, но потенциально
  незавершённую оптимизацию.
- Изменение модели, вопроса, chunking или датасета инвалидирует evidence-кэш.
- `--limit` берёт первые строки до split и может менять распределение данных.
- Regex matching lemma/stem ограниченно работает для дефисных и составных
  терминов.

## 18. Демонстрационные ноутбуки

- `experiments/notebooks/06_Keyword_tuning_task_en.ipynb`;
- `experiments/notebooks/07_Keyword_tuning_task_ru.ipynb`;
- `experiments/notebooks/08_Keyword_tuning_diagnostics_en.ipynb` — пошаговая диагностика pool/prescreen/SFFS/stability по сохранённому trace (без повторного tuning).

Каждый ноутбук имеет режимы `sample` и `full`, показывает progress bars,
диагностику настройки, сохраняет отдельную tuned-модель и сравнивает baseline
с `static-keywords` на held-out test по всем доступным метрикам. Результаты
записываются в `experiments/analysis_results/keyword_tuning_task/{en,ru}`.
