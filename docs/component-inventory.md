# Инвентаризация компонентов

## Поддерживаемое ядро

- `untie.config` — валидируемые EN/RU-профили и настройки пайплайна.
- `untie.models` — ленивая загрузка tokenizer, QA-модели, pipeline и sentence encoder.
- `untie.text`, `untie.chunking` — разбиение на предложения и перекрывающиеся чанки.
- `untie.qa` — извлечение, валидация, агрегация и консенсус ответов.
- `untie.keywords`, `untie.ranking`, `untie.attention` — этапы переранжирования.
- `untie.topics` — versioned train-only topic artifact, hierarchy, soft routing,
  aspect/topic scoring, dev config selection и retrieval/compute metrics.
- `untie.pipelines` — сценарии baseline и attention.
- `untie.cli` — поддерживаемая точка входа из командной строки.
- `untie.results_analysis` — анализ JSON-результатов batch-экспериментов.
- `untie.extraction_metrics` — метрики качества извлечения (char/token F1, ROUGE-L, BERTScore).
- `untie.model_params` — типизированный контракт init/tuned JSON и атомарная запись.
- `untie.keyword_evidence` — train-only evidence и кэши chunks/QA/metrics.
- `untie.keyword_tuning`, `untie.keyword_training` — objective, SFFS,
  stability selection и сборка глобального словаря.
- `untie.topic_keyword_training` — leakage-safe SFFS-профили словарей для
  train-derived topic clusters и nearest-leaf assignments.
- `untie.keyword_diagnostics` — аудит pool/prescreen/SFFS/stability по
  сохранённым evidence и trace JSON.
- `untie.cross_encoder_ranking` — document-level ranking чанков MiniLM
  cross-encoder и смесь со static-словарём без drop; веса смеси только на
  train/dev gold.
- `untie.ce_topic_shortlist` — CE-запрос с leaf `representative_terms`,
  shortlist top-k и финальный `AnswerAggregator`; k на dev; без SFFS-словарей.

Документация: [`information-extraction-concepts.md`](information-extraction-concepts.md)
(концепции и формулы), [`keyword-tuning-algorithm.md`](keyword-tuning-algorithm.md)
(операционное описание настройки),
[`sredstva-otbora-relevantnyh-chankov.md`](sredstva-otbora-relevantnyh-chankov.md)
(каталог средств выявления релевантных чанков).

## Слой совместимости

- `scripts/run_baseline.py` и `scripts/run_attention.py` проксируют вызовы в CLI пакета.
- Нумерованные скрипты и модули ниже `scripts/` сохранены для совместимости с ноутбуками,
  но больше не являются каноническим API.
- `scripts/03_Keywords_with_attention_refactored.py` — refactored EN batch через `untie`.
- `scripts/03_Keywords_with_attention_refactored_ru.py` — refactored RU batch для RusErrC.
- `scripts/05_Tune_model_keywords.py` — leakage-safe сборка EN/RU tuned-модели.
- `scripts/06_Build_topic_model.py` — train-only сборка и диагностика
  иерархического topic artifact.
- `scripts/07_Tune_topic_keywords.py` — добавление cluster-specific словарей
  в topic artifact поверх train/dev evidence.
- `scripts/10_Refit_hierarchical_keywords.py` — финальный train+dev refit
  keywords при замороженных cluster/routing; по умолчанию без gold-contrast и
  со scoring `only_weight`; пол `min_keywords` на этом шаге снижен до 4;
  пишет отдельный `topic_model.refit.json` в `refit_no_gold_contrast/`.
- `scripts/11_Evaluate_hierarchical_test.py` — test-only оценка hierarchical
  модели; пишет в отдельный каталог и может переиспользовать предыдущие
  режимы из `test_predictions_all_metrics.csv`. `--soft-rerank` отвечает на
  всех чанках и ранжирует ответы, не затирая `evaluation/` и прошлые refit.
- `scripts/12_Evaluate_cross_encoder.py` — изолированная оценка
  `cross-encoder/ms-marco-MiniLM-L6-v2` по всем чанкам, затем смесь со
  static-словарём; веса смеси подбираются на dev; test gold в ranking не
  используется; прошлые `evaluation/` не затираются.
- `scripts/13_Evaluate_ce_topic_shortlist.py` — CE shortlist с
  topic-conditioned query (leaf representative terms) и baseline
  `AnswerAggregator`; k на dev; без cluster SFFS и без overwrite прошлых eval.
- `scripts/run_en_text_only_v4.sh`, `scripts/run_en_text_only_v5.sh`,
  `scripts/run_en_large_dict_v3.sh` — готовые shell-профили полного EN tuning.

## Сохранённые эксперименты

- Нумерованные ноутбуки: историческое исследование chunking, QA, attention, RAG,
  русских датасетов, Qwen и анализа результатов.
- `06_Keyword_tuning_task_en.ipynb`, `07_Keyword_tuning_task_ru.ipynb` —
  полный tuning pipeline; `08_Keyword_tuning_diagnostics_en.ipynb` —
  диагностика по сохранённым артефактам.
- `scripts/models_processing/save_model_scripts` — заменено на `tools/model_download/download.py`.
- Сгенерированные CSV, JSON, PNG и log-файлы — локальные артефакты.

## Неактивные legacy-компоненты

- `FilterMode.BY_QUESTION` не имеет реализации в legacy-оркестраторе.
- Legacy NER getters — заглушки.
- `EnhancedAnswer`, `filter_characteristic_words`, несинглтоновый `SentenceTokenizer`
  и несколько вариантов consensus не имеют production-вызовов.
- LangChain/FAISS и KeyBERT встречаются только в экспериментальном коде или дампах зависимостей.

Ни один сохранённый legacy-компонент не импортируется новым пакетом `untie`.
