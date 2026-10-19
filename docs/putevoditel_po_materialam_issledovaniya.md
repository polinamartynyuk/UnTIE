# Путеводитель по материалам исследования для автореферата и диссертации

Дата составления: 18 октября 2026 г.

## 1. Как пользоваться материалами

В проекте сохранены три уровня материалов:

1. **Итоговые источники** — финальная архитектура, официальный эксперимент и рукописи статей. Их следует использовать для основных формулировок, таблиц и численных результатов.
2. **Исследовательские досье** — связное объяснение постановки задачи, развития гипотез, отрицательных результатов и ограничений. Они удобны для глав диссертации и автореферата, но числа из них следует сверять с первичными итоговыми отчётами.
3. **Исторические протоколы и результаты** — последовательность V1–V10 и абляции. Они нужны для обоснования того, как развивался метод, но не должны заменять итоговую оценку.

Главная пара источников:

- [финальная архитектура метода](../hypothesis/final_method_architecture.md);
- [обобщённое сравнение вариантов и базовых систем](../hypothesis/final_method_all_variants_and_baselines_comparison.md).

Готовая визуальная схема верхнего уровня находится в файле [architecture.png](../articles/method_article/architecture.png). Схема развития метода находится в [method_development.png](../articles/method_article/method_development.png).

## 2. Гипотезы и развитие метода

### Основные материалы

- [Досье статьи 1: разработка метода UnTIE](../hypothesis/article_1_untie_method_experiments_dossier.md) — постановка задачи, хронология V1–V10, отрицательные результаты, финальная архитектура, статистика и ограничения доказательности.
- [Досье статьи 2: многоязычные данные и сравнительные испытания](../hypothesis/article_2_multilingual_dataset_benchmark_experiments_dossier.md) — гипотезы о переносимости между корпусами и языками, подготовка данных и межкорпусное сравнение.
- [Финальная архитектура метода](../hypothesis/final_method_architecture.md) — точное определение H2, H3 и consistency gate.
- [Обобщённое сравнение всех основных вариантов](../hypothesis/final_method_all_variants_and_baselines_comparison.md) — связь поколений метода, базовых систем и итоговых результатов.

### Исходные и промежуточные гипотезы

- [Иерархическое тематико-аспектное повторное ранжирование](../hypothesis/hyp1_hierarchical-topic-aspect-reranking.md).
- [Исходная гипотеза 1](../hypothesis/hyp1_prompt1.md), [гипотеза 2](../hypothesis/hyp2_prompt2.md), [гипотеза 3](../hypothesis/hyp3_prompt3.md), [гипотеза 4](../hypothesis/hyp4_prompt4.md).
- [План семи стратегий V4](../hypothesis/v4_experiment_plan_7_strategies.md).
- [Протокол reader-centered V4.1](../hypothesis/v4_1_reader_centered_continuation_protocol.md).
- [Протокол semantic reader agreement V4.3](../hypothesis/v4_3_semantic_reader_agreement_protocol.md).
- [Протокол adaptive reader consensus V4.4](../hypothesis/v4_4_adaptive_reader_consensus_protocol.md).
- [Протокол universal experiment V6](../hypothesis/v6_uar_g_universal_experiment_protocol.md).
- [Протокол risk-aware V7](../hypothesis/v7_rapr_progressive_risk_aware_protocol.md).
- [Протокол dual-branch V8](../hypothesis/v8_dual_branch_hrr_and_v44_adaptive_fusion_protocol.md).
- [Протокол adaptive component selection V9](../hypothesis/v9_h2_adaptive_component_selection_protocol.md).
- [Протокол switch-rank-risk V10](../hypothesis/v10_srr_switch_rank_risk_protocol.md).

Для описания развития метода лучше начинать с двух досье, а к отдельным протоколам обращаться только за деталями конкретного этапа.

## 3. Подготовка данных

### Методика и техническая инструкция

- [Подготовка исходных данных SciREX](metodika-podgotovki-dannyh-scirex.md) — последовательность предобработки, разбиение документов, формирование фрагментов, нормализация, разметка и контроль корректности.
- [Предобработка SciREX для аспектного извлечения ответа](scirex-aspect-qa-preprocessing.md) — точный контракт входов, разбиений, фрагментов, эталонных ответов и выходных файлов.
- [Multi-question расширение Phase 1](scirex-phase1-multiquestion.md) — формулировки вопросов Q0/Q3/Q5, согласованность ответов, классы положительных и отрицательных примеров, выбор порогов и выходные артефакты.
- [Протокол подготовки данных V3.1](../hypothesis/v3_1_data_preparation_protocol.md) и [автоматизированный протокол V3.1](../hypothesis/v3_1_automated_data_preparation_protocol.md) — историческое развитие процедуры подготовки примеров.

### Проверка подготовленных данных

- [Аудит подготовки обучающих примеров](../articles/data_article/data_preparation_audit/data_preparation_audit.md) — финальные объёмы, классы примеров и проверки внутренней непротиворечивости.
- [Сводка аудита JSON](../articles/data_article/data_preparation_audit/audit_summary.json).
- [Численность классов CSV](../articles/data_article/data_preparation_audit/class_counts.csv).
- [Показатели качества CSV](../articles/data_article/data_preparation_audit/quality_metrics.csv).
- [Сводка обучающих примеров CSV](../articles/data_article/data_preparation_audit/training_examples_summary.csv).
- [График статистики подготовки данных](../articles/data_article/data_preparation_audit/data_preparation_statistics.png).

SciER-128 и SciER-384 построены из одних исходных документов. Их нельзя считать двумя независимыми корпусами или суммировать при подсчёте числа документов.

## 4. Результаты экспериментов

### Основной заранее заданный эксперимент

Каталог основного запуска: `artifacts/final_official_split_benchmark/run_20260930_175541`.

- [Итоговый отчёт official-split benchmark](../artifacts/final_official_split_benchmark/run_20260930_175541/reports/final_benchmark_results.md) — основная таблица по SciREX, SciER-128, SciER-384, RuSERRC и SciMDIX.
- [Первичные метрики JSON](../artifacts/final_official_split_benchmark/run_20260930_175541/test_analysis/primary_metrics.json).
- [Метрики по вопросам JSON](../artifacts/final_official_split_benchmark/run_20260930_175541/test_analysis/question_metrics.json).
- [Парные сравнения JSON](../artifacts/final_official_split_benchmark/run_20260930_175541/test_analysis/primary_contrasts.json).
- [Статистический анализ](../artifacts/final_official_split_benchmark/run_20260930_175541/reports/statistical_analysis.md).
- [Воспроизводимость](../artifacts/final_official_split_benchmark/run_20260930_175541/reports/reproducibility.md).
- [Анализ вычислительных ресурсов](../artifacts/final_official_split_benchmark/run_20260930_175541/reports/resource_analysis.md).
- [Статус завершения](../artifacts/final_official_split_benchmark/run_20260930_175541/COMPLETED.json).

Запуск имеет итоговый статус `COMPLETED`. Сохранённый рядом `FAILED.json` относится к ошибке одного из предыдущих этапов запуска и не отменяет последующее успешное завершение; для статуса следует использовать `status.json` и `COMPLETED.json`.

### Финальная архитектура после абляций H3

- [Анализ consistency gate](../hypothesis/h3_consistency_gate_results_analysis.md) — русскоязычное объяснение гипотезы, решений по корпусам, итоговых метрик и границ доказательности.
- [Первичный отчёт consistency gate](../artifacts/qa_chunk_selection_h3_consistency_gate/run_20261008_01/reports/consistency_gate_results.md).
- [Первичные метрики consistency gate](../artifacts/qa_chunk_selection_h3_consistency_gate/run_20261008_01/test_analysis/primary_metrics.json).
- [Финальное сравнение вариантов и базовых систем](../hypothesis/final_method_all_variants_and_baselines_comparison.md).

Финальная архитектура `H3-CONSISTENCY-GATE-3` получила следующий Token F1:

| Набор данных | Token F1 |
|---|---:|
| SciREX | 26,0253 |
| SciER-384 | 72,0436 |
| SciER-128 | 72,7105 |
| RuSERRC | 66,2114 |
| SciMDIX | 62,8274 |

При семейной агрегации две конфигурации SciER сначала усредняются, затем четыре семейства получают одинаковый вес. Результат `ALL_STANDARDIZED` равен **56,8603**.

Consistency gate был сформулирован после раскрытия official TEST. Поэтому `56,8603` является результатом финальной замороженной post-hoc архитектуры. Последняя версия, полностью включённая в заранее заданный official-split benchmark, — H3/P4 с `ALL_STANDARDIZED = 55,9674`. В диссертации следует явно разделять эти два уровня доказательности.

### Числа, использованные в статье о методе

В основном межкорпусном сравнении статьи оставлена только конфигурация SciER-384. Для четырёх наборов данных использованы:

- базовая система B2: `(24,0979 + 63,6773 + 65,6714 + 52,6723) / 4 = 51,53`;
- финальный метод: `(26,0253 + 72,0436 + 66,2114 + 62,8274) / 4 = 56,78`;
- разница: `56,78 − 51,53 = 5,25` пункта Token F1.

Эти показатели корректны для представления SciER-384 в статье. Их нельзя называть `ALL_STANDARDIZED`, поскольку полный стандартизованный показатель объединяет SciER-128 и SciER-384 в одно семейство.

## 5. Заготовки и текущие версии статей

### Статья о методе

- [Метод извлечения информации из неструктурированных текстовых документов](../articles/method_article/metod_izvlecheniya_informacii_iz_nestrukturirovannyh_tekstovyh_dokumentov.docx) — текущая рукопись.
- [Сравнительная схема метода](../articles/method_article/architecture.png).
- [Схема развития метода](../articles/method_article/method_development.png).

### Статья о данных

- [Программный конвейер подготовки данных, редакция 2.3.5](../articles/data_article/programmnyy_konveyer_podgotovki_dannyh_redakciya_2_3_5.docx) — наиболее поздняя и полная версия по дате и размеру файла.
- [Многоязычные данные для выбора фрагментов научных текстов](../articles/data_article/mnogoyazychnye_dannye_dlya_vybora_fragmentov_nauchnyh_tekstov.docx) — альтернативная рукопись.
- [Версия без оракульных оценок](../articles/data_article/mnogoyazychnye_dannye_dlya_vybora_fragmentov_nauchnyh_tekstov_bez_orakulnyh_ocenok.docx) — вариант для подачи без оракульных показателей.

Остальные файлы DOCX в `articles/data_article` являются промежуточными редакциями. При подготовке диссертации следует начинать с наиболее поздней версии и сверять описание данных с исследовательским досье статьи 2 и аудитом подготовки примеров.

## 6. Русскоязычная терминология

- [Терминология статьи 1: метод UnTIE](../hypothesis/article_1_untie_method_terminology_ru.md) — рекомендуемые переводы терминов метода, моделей, признаков, метрик и статистических процедур.
- [Терминология статьи 2: данные и сравнительная оценка](../hypothesis/article_2_multilingual_dataset_benchmark_terminology_ru.md) — термины для корпусов, разметки, разбиений, предобработки и сравнительных испытаний.
- [Основные понятия извлечения информации](information-extraction-concepts.md) — более общий понятийный материал.
- [Средства отбора релевантных фрагментов](sredstva-otbora-relevantnyh-chankov.md) — обзор компонентов и ранних подходов.

Для диссертации рекомендуется последовательно использовать «фрагмент», «ранжирующая модель», «модель извлечения ответа», «повторное ранжирование», «набор данных» и «конвейер обработки». Кальки «чанк», «ранкер», «ридер», «реранкер», «датасет» и «пайплайн» лучше оставлять только в кодовых именах или техническом приложении.

## 7. Рекомендуемый порядок подготовки диссертации

1. Постановка задачи и научная новизна: досье статьи 1 и финальная архитектура.
2. Обзор и терминология: два терминологических файла и `information-extraction-concepts.md`.
3. Данные и методика их подготовки: досье статьи 2, методика SciREX, Phase 1 и аудит данных.
4. Развитие метода: хронология V1–V10 из досье статьи 1 с обращением к отдельным протоколам только для нужных деталей.
5. Основной эксперимент: отчёт official-split benchmark и статистический анализ.
6. Финальная модификация: consistency gate с явной пометкой post-hoc статуса.
7. Обсуждение ограничений: exposure RuSERRC, общие документы SciER-128/384, answerable-only фильтрация, зависимость от модели извлечения ответа и необходимость новой закрытой проверки consistency gate.

## 8. Что считать первичным источником при расхождении

При расхождении формулировок использовать следующий приоритет:

1. JSON-метрики завершённого экспериментального запуска;
2. отчёт, автоматически созданный тем же запуском;
3. файлы `final_method_architecture.md` и `final_method_all_variants_and_baselines_comparison.md`;
4. исследовательские досье статей;
5. тексты DOCX;
6. исторические протоколы и промежуточные отчёты.

Такой порядок позволяет сохранить воспроизводимость чисел и одновременно использовать связное объяснение развития метода из досье и статей.
