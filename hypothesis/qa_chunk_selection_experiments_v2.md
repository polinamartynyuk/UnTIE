# Инструкция агенту по разработке кода для продолжения экспериментов по отбору evidence и extractive QA

## 0. Контекст и цель

Необходимо **продолжить уже выполненную серию экспериментов**, а не переписывать проект с нуля.

Текущая экспериментальная реализация:

```text
experiments/qa_chunk_selection/
```

Пилотные артефакты E0:

```text
artifacts/qa_chunk_selection/
```

Исходный протокол:

```text
hypothesis/qa_chunk_selection_experiments_v1.md
```

Данные и метки:

```text
artifacts/phase1_multiquestion/official_train_q5
artifacts/phase1_multiquestion/official_dev_q5
datasets/scirex_aspect_qa_preprocessing
datasets/scirex_aspect_qa_preprocessing_dev
```

Канонические вопросы аспектов:

```text
untie.aspect_qa.config.ASPECT_QUESTIONS
```

Главная цель нового этапа — проверить гипотезу:

> Для устойчивого aspect-oriented extractive QA недостаточно выбирать тематически релевантный или формально answerable chunk. Необходимо учитывать, насколько evidence пригодно для извлечения конкретным downstream extractive reader, а при неуверенности ranking — адаптивно использовать несколько candidates.

Пилотный прогон E0 считать **разведочным экспериментом**. Его код и артефакты нельзя перезаписывать.

---

# 1. Что обязательно сохранить из E0

Агент перед изменениями должен прочитать существующий код, `qa_chunk_selection_experiments_v1.md` и отчет E0.

На official SciREX DEV было:

```text
105 answerable вопросов
57 документов
candidate_mode = all
~19.6 чанка на вопрос
seed = 42
```

## 1.1. A0 — dense cosine

```text
Hit@1 = 10.5%
Hit@5 = 48.6%
HN@1 = 12.4%
Token F1 = 11.37
```

## 1.2. A — generic relevance CE

Модель:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

без дообучения на SciREX.

DEV:

```text
Hit@1 = 22.9%
Hit@5 = 64.8%
HN@1 = 12.4%
Pairwise Pos>HN = 48.8%
Token F1 = 22.90
```

## 1.3. B — answerability CE

Тот же MiniLM-backbone, отдельная classification-head.

В E0 использовалось:

```text
loss = BCEWithLogits
epochs = 3
batch = 16
lr = 2e-5
max_length = 512
pos:HN:easy = 1:3:1
```

DEV:

```text
Hit@1 = 30.5%
Hit@5 = 81.0%
HN@1 = 4.8%
Pairwise Pos>HN = 79.1%
Token F1 = 22.14
```

Особенно:

```text
при 2–3 HN:
A pairwise = 49.4%
B pairwise = 87.2%
```

Значит, `relevance != answerability` — рабочая гипотеза.

Но B не улучшил итоговый QA.

Критический результат:

```text
F1 | selected positive:

A = 77.1
B = 56.6
```

B выбирает больше positive, но эти positive хуже для frozen reader.

## 1.4. C

E0 C:

```text
B ranking
→ top-5
→ QA отдельно по каждому passage
→ выбор span по max(start_logit + end_logit)
```

DEV:

```text
B Token F1 = 22.14
C Token F1 = 25.81
```

TRAIN: C хуже B.

Следовательно, multi-passage помогает преимущественно при ненадежном top-1.

**Важно:** текущий C не называть полноценным Multi-passage BERT. Это raw-logit multi-passage baseline.

## 1.5. D1

```text
dense + BM25 → RRF
```

показал ухудшение:

```text
HN@1 = 20%
Pairwise = 41.9%
metric: Hit@1 = 0/16, F1 = 0
```

D1 **не включать в новый default pipeline**.

## 1.6. D2

E0:

```text
0.5 * RRF(dense, BM25)
+ 0.3 * P_B(answerable)
+ 0.2 * section signal
```

DEV:

```text
Hit@1 = 34.3%
Token F1 = 28.89
EM = 23/105
```

Но:

```text
HN@1 = 12.4%
Pairwise = 59.5%
Hit@5 хуже B
~70% section_title пустые
веса не тюнились
```

D2 считать перспективной, но **не подтвержденной** конфигурацией.

---

# 2. Ключевые вопросы V2

Код должен позволить ответить на вопросы:

1. Почему B-positive хуже извлекаются reader-ом, чем A-positive?
2. Насколько `ambiguous` является главным оставшимся distractor?
3. Можно ли ранжировать chunks по сочетанию:
   - relevance;
   - answerability;
   - reader suitability for the frozen QA model?
4. Как зависит QA от `M ∈ {1,2,3,5,8}`?
5. Можно ли выбирать M адаптивно по уверенности ranking?
6. Какие компоненты D2 реально полезны?
7. Помогает ли sentence-level evidence selection?
8. Помогает ли answer aggregation после очистки evidence?
9. Нужен ли вообще D3 graph expansion?

Не выполнять D3 автоматически.

---

# 3. Общие правила разработки

## 3.1. E0 неизменяем

Нельзя перезаписывать:

```text
artifacts/qa_chunk_selection/
```

Новые артефакты писать в:

```text
artifacts/qa_chunk_selection_v2/
```

E0 должен оставаться полностью воспроизводимым отдельным конфигом.

## 3.2. Не менять production-поведение без необходимости

По возможности не менять:

```text
untie/
datasets/
Phase 1 artifacts
KeyBERT
hierarchical
answer_aggregation
cluster_discriminative_keywords
```

Расширять `experiments/qa_chunk_selection/` либо добавить внутри него `v2/`.

Если общий код требуется изменить ради повторного использования, изменение должно быть обратно совместимым и покрыто regression-тестом E0.

## 3.3. Строго разделить генерацию predictions и gold-based evaluation

Архитектурно разделить два слоя:

```text
prediction generation
    ↓
raw predictions / raw chunk scores
    ↓
gold-based evaluation/enrichment
```

Runtime scorer/ranker **не должен получать объект, содержащий gold-поля**, если эти поля ему не нужны.

На inference для `DEVELOPMENT`, `INTERNAL_TEST` и `seen_exploratory_dev` запрещено использовать:

```text
gold_answer
gold_span
positive label
hard_negative label
ambiguous subtype
```

для:

- ranking;
- filtering;
- top-K/top-M selection;
- adaptive M;
- reader-aware scoring;
- aggregation;
- stopping rules.

Gold допустим:

```text
TRAIN:
    только для обучения trainable-компонентов

CALIBRATION:
    только для early stopping, score/probability calibration,
    выбора заранее объявленных численных thresholds/weights

DEVELOPMENT:
    только для выбора архитектурного варианта и анализа ошибок

INTERNAL_TEST:
    только для финального post-hoc evaluation после freeze
```

После первого открытия `INTERNAL_TEST` запрещено менять метод и повторно называть тот же split независимым тестом.

Oracle stages маркировать:

```text
oracle = true
diagnostic_only = true
```

Oracle-результаты не участвуют в model selection.

## 3.4. Extractive traceability

Каждый непустой ответ должен сохранять **две системы координат**:

```text
chunk-local:
    qa_chunk_start_char
    qa_chunk_end_char

source/document-level:
    source_start_char
    source_end_char
```

и выполняться проверки:

```text
chunk_text[qa_chunk_start_char:qa_chunk_end_char] == answer_text
source_text[source_start_char:source_end_char] == answer_text
```

Если исходное хранилище не имеет document-level offsets, агент должен явно зафиксировать доступный канонический source representation и не называть chunk-local offsets document-level координатами.

## 3.5. Нельзя подменять понятие `reader extractability` сырым QA-score

`start_logit + end_logit`, margin spans и другие внутренние оценки frozen QA — это только:

```text
QA confidence proxy
```

Они **не являются доказательством**, что ответ правильный или что чанк действительно extractable.

Во всех именах полей/отчётах использовать формулировки вроде:

```text
qa_confidence_proxy
qa_span_score
qa_span_margin
```

а `reader_success` определять только post-hoc через gold на допустимом split либо через отдельную обученную модель, если такой эксперимент будет явно добавлен позже.

# 4. Чистая схема данных V2

Official SciREX DEV уже использовался для выбора смеси B, просмотра результатов и формирования V2. Его считать:

```text
seen_exploratory_dev
```

а не чистым TEST.

## 4.1. Создать закрытый INTERNAL_TEST из official TRAIN

Одного `TRAIN/CALIBRATION/DEVELOPMENT` недостаточно: V2 предполагает последовательный выбор B2, reader-aware схемы, M, adaptive policy и D2-абляций. Повторное использование одного DEVELOPMENT для всех решений неизбежно делает его частью разработки.

Поэтому split формировать **по `document_id` до answerable-фильтрации**:

```text
TRAIN         60%
CALIBRATION   15%
DEVELOPMENT   15%
INTERNAL_TEST 10%
```

Это рекомендуемый default. Проценты должны задаваться конфигом.

Seed split:

```text
42
```

Сохранить:

```text
artifacts/qa_chunk_selection_v2/data/splits_v2.json
```

## 4.2. INTERNAL_TEST должен быть технически заблокирован

До фиксации финальной конфигурации stages не должны читать gold/метрики `INTERNAL_TEST`.

Рекомендуемый механизм:

```text
--allow-internal-test
```

по умолчанию `false`.

Перед первым запуском `final-eval` создать:

```text
artifacts/qa_chunk_selection_v2/final_freeze_manifest.json
```

с:

```text
git commit
resolved config hash
selected B2 variant
selected reader-aware variant
selected M/adaptive policy
selected D2/D4/D5 components
seeds
model identifiers
```

`final-eval` должен проверить наличие freeze-manifest и его hash.

## 4.3. Альтернатива при слишком малом INTERNAL_TEST

Если после document-level split в `INTERNAL_TEST` оказывается слишком мало answerable документов/вопросов для осмысленной оценки, **не увеличивать test за счет просмотра результатов и ручного перестроения split**.

Поддержать дополнительный режим:

```text
grouped_nested_cv
```

или grouped repeated CV на official TRAIN с полностью закрытыми outer folds.

Default можно оставить fixed holdout; nested-CV — опциональный строгий режим для итоговой диссертационной проверки.

## 4.4. Стратификационный sanity-check

Разбиение остаётся group-by-document. После split проверить распределение:

```text
n_documents
n_questions
answerable / no-robust-positive
method/task/metric/material
positive count
HN count
easy count
ambiguous count
HN buckets: 0, 1, 2–3, 4+
```

Если распределение резко несбалансировано, допускается **детерминированный group-stratified split**, использующий только заранее известные метаданные/labels, но алгоритм и seed должны быть зафиксированы до model evaluation.

Ни один `document_id` не может встречаться в двух ролях.

## 4.5. Seen official DEV

`seen_exploratory_dev` использовать только для:

- regression E0;
- воспроизведения пилотного вывода;
- проверки, сохраняется ли направление эффекта после того, как V2 уже выбран на внутренних splits.

Нельзя:

- подбирать по нему новые параметры;
- выбирать архитектуру;
- возвращаться после его просмотра и менять финальную конфигурацию, сохраняя статус независимой оценки.

# 5. Regression E0

Добавить stage:

```text
e0-regression
```

Цель — убедиться, что V2 не изменяет поведение уже зафиксированного E0.

## 5.1. Regression должен иметь два уровня

### E0-R1 — artifact/metric regression

Без повторного обучения:

- прочитать сохраненные E0 `predictions.jsonl` / `metrics.json`;
- пересчитать метрики текущими функциями;
- сравнить с сохраненными E0 results.

Этот уровень должен проходить строго, кроме заранее объяснимой floating-point tolerance.

### E0-R2 — inference replay

Переиспользовать **сохраненные E0 checkpoints**, включая выбранный B, а не переобучать B заново.

На старом config, data и seed=42 повторить inference A/B/C/D1/D2 и сравнить outputs.

Это проверяет, что новые adapters/cache/schema не изменили старый pipeline.

## 5.2. Что сравнивать

Минимум:

```text
A Hit@1, Token F1
B Hit@1, HN@1, Pairwise, Token F1
C Token F1
D1 Hit@1
D2 Hit@1, Token F1
```

Для integer counts / selected IDs при детерминированном inference требовать exact match.

Для floating scores использовать явно заданную tolerance, например:

```text
atol / rtol
```

и сохранить ее в regression config.

## 5.3. Не использовать retrain как regression criterion

Переобучение E0-B с seed=42 может отличаться из-за CUDA/library nondeterminism. Это отдельная reproducibility диагностика, но не должно блокировать V2, если сохраненный checkpoint replay полностью совпадает.

Если artifact regression или checkpoint replay не проходит — завершить run с ошибкой и вывести структурированный diff.

# 6. R1 — Reader Oracle

Добавить stage:

```text
reader-oracle
```

Это первый обязательный диагностический этап, но **не production method**.

Использовать текущий frozen reader из resolved config. Не хардкодить путь, если модель уже описана профилем/конфигом; фактический identifier записать в manifest.

## 6.1. На каких splits запускать до freeze

До финальной фиксации разрешено выполнять R1 на:

```text
TRAIN
CALIBRATION
DEVELOPMENT
seen_exploratory_dev
```

Не выполнять gold-based reader oracle на `INTERNAL_TEST` до `final_freeze_manifest`.

После freeze его можно посчитать на INTERNAL_TEST только как post-hoc ceiling diagnostic; результат не должен менять pipeline.

## 6.2. Per-positive

Для каждого вопроса взять все `positive` chunks и прогнать QA на каждом отдельно.

Сохранить:

```json
{
  "document_id": "...",
  "question_id": "...",
  "aspect": "...",
  "chunk_id": "...",
  "chunk_length_tokens": 0,

  "predicted_answer": "...",
  "qa_start_logit": 0.0,
  "qa_end_logit": 0.0,
  "qa_span_score": 0.0,

  "qa_chunk_start_char": 0,
  "qa_chunk_end_char": 0,
  "source_start_char": 0,
  "source_end_char": 0,

  "em": 0,
  "token_f1": 0.0,
  "char_f1": 0.0,
  "rouge_l_f1": 0.0,

  "matched_gold": "...",
  "gold_string_present": true,
  "gold_occurrence_count": 0,
  "gold_relative_position": 0.0
}
```

## 6.3. Oracle diagnostics

### R1a — per-positive quality

Распределение QA-качества по всем positive chunks.

### R1b — best-positive oracle

Для каждого вопроса выбрать positive с максимальным gold-based QA F1.

Это **строго diagnostic-only**. Не использовать выбранный oracle-positive для обучения reader-aware weights или model selection.

### R1c — reader-solvable rate

Для каждого вопроса определить, существует ли хотя бы один positive, где:

```text
EM = 1
Token F1 > 0
Token F1 >= 50
Token F1 >= 80
```

Помнить: Token F1 в текущей реализации имеет шкалу `0..100`.

## 6.4. A-positive vs B/B2-positive

Для вопросов, где A/B/B2 выбрали positive, сравнить:

```text
chunk length
gold relative position
section
gold occurrence count
A score
B/B2 score
QA span score (confidence proxy)
EM
Token F1
```

Группы:

```text
A_positive_only
B_positive_only
same_positive
different_positive
```

Цель — установить, **какие свойства отличают positive, удобные reader-у, от positive, которые answerability ranker поднимает, но reader обрабатывает хуже**.

## 6.5. Дополнительная важная декомпозиция

Посчитать для каждого вопроса:

```text
positive_retrievable_by_A
positive_retrievable_by_B/B2
reader_solvable_given_any_positive
reader_solvable_given_topK_B/B2
```

Это разделяет ошибки:

```text
ranking failure
vs
reader failure despite correct evidence
```

## 6.6. Выход

```text
analysis/reader_oracle/per_positive.jsonl
analysis/reader_oracle/per_question.jsonl
analysis/reader_oracle/a_vs_b_positive.csv
analysis/reader_oracle/by_aspect.csv
analysis/reader_oracle/summary.json
```

# 7. R2 — Ambiguous Audit

Добавить параметризованный stage:

```text
ambiguous-audit
```

В E0:

```text
B top-1 ambiguous = 59/105
C QA source ambiguous = 53/105
```

Поэтому ambiguous анализировать раньше D3.

## 7.1. Два момента запуска

### R2-E0 — ранний audit

До обучения B2 использовать уже существующие:

```text
A
B_E0
frozen QA
```

Это позволяет сразу объяснить ошибки E0.

### R2-B2 — повторный audit после выбора B2

После `b2-eval` повторно пересчитать только B2-зависимые scores/selection на **том же frozen наборе chunks**.

QA outputs переиспользовать из cache.

Не использовать изменения R2-B2 для relabeling DEVELOPMENT.

## 7.2. Сохранять subtype

Не сводить в один класс:

```text
question_sensitive
gold_supporting_robust_negative
false_negative_candidate
confirmed_false_negative
```

## 7.3. Для каждого ambiguous chunk

До gold-evaluation вычислить:

```text
dense_score
A relevance score
selected answerability score: B_E0 или B2
QA best span
QA start/end logits
QA span score (confidence proxy)
```

После inference evaluation-layer может добавить:

```text
gold_string_present
normalized_gold_present
EM
Token F1
Char F1
ROUGE-L
```

## 7.4. Отчет по subtype

```text
count
mean/median A score
mean/median B/B2 score
mean/median QA confidence proxy
A top1 frequency
B/B2 top1 frequency
B/B2 top5 frequency
C qa-source frequency
EM
Token F1
```

## 7.5. Manual-review export

Сформировать:

```text
analysis/ambiguous_audit/e0/top_errors.jsonl
analysis/ambiguous_audit/e0/top_errors.md
analysis/ambiguous_audit/b2/top_errors.jsonl
analysis/ambiguous_audit/b2/top_errors.md
```

В Markdown вывести:

- document_id;
- aspect/question;
- gold;
- subtype;
- chunk text;
- A/B/B2 scores;
- QA answer/confidence proxy;
- EM/F1.

Сортировки:

1. highest answerability among ambiguous;
2. top-1 ambiguous;
3. C QA-source ambiguous;
4. non-zero F1 ambiguous;
5. gold-string-present ambiguous.

**Не переписывать labels автоматически.**

До final freeze `ambiguous-audit` не должен раскрывать gold-based top-errors для INTERNAL_TEST.

# 8. B2 — уменьшение переобучения answerability model

Текущий B сохранить как:

```text
b_oversample_e0
```

Новые trainable варианты запускать на seeds:

```text
42
43
44
```

## 8.1. Сначала ввести `training_eligible_hn`

В E0 были HN, для которых нормализованный gold встречался в тексте чанка. Это не доказывает автоматически, что label неверный, но делает такой пример потенциально шумным strong negative.

**Не переписывать исходный label.**

Добавить отдельные флаги:

```text
hn_gold_string_warning
training_eligible_hn
```

Default:

```text
training_eligible_hn =
    label == hard_negative
    AND hn_gold_string_warning == false
```

Для воспроизводимости E0 сохранить возможность `include_hn_gold_warning=true`, но V2-default — `false`.

В evaluation считать отдельно:

```text
pairwise_all_hn
pairwise_clean_hn
HN@K_all
HN@K_clean
```

Основные E0-совместимые метрики по всем исходным labels не удалять.

## 8.2. Не давать вопросам с большим числом чанков доминировать training loss

Простой список всех `(question, chunk)` пар может неявно сильнее взвешивать вопросы с большим количеством negatives.

Реализовать question-balanced training как default:

- каждый `question_id` получает примерно одинаковый суммарный вес;
- внутри вопроса positive/HN/easy балансируются согласно варианту обучения;
- логировать effective weight / число samples на вопрос.

Если используется обычный flat sampler, оставить его только как абляцию.

## 8.3. B2a — question-balanced weighted BCE

Использовать уникальные training pairs без статического дублирования HN до `3 × positives`.

Компенсацию классов задавать через class/sample weights, рассчитанные **только по TRAIN**.

Рекомендуемый default:

```text
question-balanced weights
+
class weighting
```

Формулу weights сохранить в resolved config/report.

## 8.4. B2b — dynamic negative sampling

На каждой эпохе:

```text
positives
+
training_eligible_hn sampled per question
+
при необходимости easy negatives
```

Не создавать один статический dataset с большим количеством физических копий одного HN.

Если negatives мало, менять выбор между эпохами и логировать:

```text
unique HN used
usage count per HN
usage distribution
```

## 8.5. B2c — pairwise loss

Train units:

```text
(question, positive, training_eligible_hard_negative)
```

Не строить полный Cartesian product всех positives × HN, если это приводит к доминированию отдельных вопросов.

Использовать question-balanced triplet sampling.

Базовый loss:

```text
-log sigmoid(score_pos - score_neg)
```

или margin ranking — конкретный вариант должен быть зафиксирован.

Не смешивать с BCE в основном B2c.

## 8.6. Early stopping

Не фиксировать безусловно 3 epochs.

Weights обновляются только по TRAIN.

Checkpoint выбирать по CALIBRATION.

Поскольку основная гипотеза B — `positive > HN`, заранее зафиксировать primary selection metric:

```text
macro pairwise_clean_hn
```

Tie-breaker:

```text
PR-AUC
```

`Hit@1/HN@1` сохранять как diagnostics.

QA F1 **не использовать для выбора B2**, иначе answerability-модель начнет косвенно оптимизироваться под downstream reader и сравнение B→R3 станет менее чистым.

## 8.7. Model selection

```text
TRAIN:
    weights

CALIBRATION:
    early stopping
    probability calibration при необходимости

DEVELOPMENT:
    выбрать B2a / B2b / B2c
```

После выбора вариант B2 фиксируется до reader-aware/C/D2.

Для INTERNAL_TEST использовать **точно те frozen checkpoints и fusion/calibration artifacts, которые были выбраны до freeze**. Не переобучать B2 после просмотра DEVELOPMENT перед INTERNAL_TEST: это изменит downstream score distribution и потребует повторной калибровки R3/C/adaptive policy. Если позже нужен retrain на большем объеме данных, это отдельный эксперимент с новым независимым evaluation protocol.

## 8.8. Отчет

Для каждого seed и варианта:

```text
TRAIN/CALIBRATION/DEVELOPMENT:
pairwise_all_hn
pairwise_clean_hn
PR-AUC
Hit@1
Hit@5
HN@1
MRR
QA EM/F1 top-1
```

Отдельно:

```text
train-development generalization gap
mean ± std across seeds
```

# 9. R3 — Reader-Aware Reranking

Добавить stage:

```text
reader-aware
```

Цель — проверить, можно ли выбирать evidence, учитывая одновременно:

- generic relevance;
- answerability;
- **confidence proxy текущего frozen QA reader**.

Не называть сырой QA-score истинной `extractability`.

## 9.1. Inference-only признаки

Для каждого рассматриваемого candidate chunk без gold:

### A

```text
generic relevance score
```

### B2

```text
answerability logit
calibrated probability, если calibration выполнена
```

### QA confidence proxy

Прогнать frozen reader:

```text
best_start_logit
best_end_logit
qa_span_score = start + end
predicted_span_length
```

Если надежно доступно:

```text
second_best_span_score
qa_span_margin = best - second_best
```

Raw QA logits могут зависеть от длины/содержимого контекста и не считаются калиброванной вероятностью правильности.

## 9.2. Два режима стоимости

### R3-full — diagnostic upper-cost mode

Посчитать QA confidence proxy для **всех chunks** документа.

Нужен для проверки самой гипотезы, но не считать его автоматически практическим финальным методом: он делает QA forward-pass по каждому чанку.

### R3-topK — практический основной вариант

```text
all chunks
→ B2 ranking
→ top-K_pre
→ QA confidence только для top-K_pre
→ reader-aware rerank
```

Проверить компактно:

```text
K_pre ∈ {3, 5, 8}
```

При выборе K обязательно учитывать:

```text
Hit@K_pre
Positive Coverage@K_pre
reader forward passes
reader tokens
```

Если prefilter теряет весь positive evidence, reader-aware стадия не может исправить ошибку.

## 9.3. Нельзя напрямую складывать raw каналы

Реализовать минимум две схемы.

### R3a — rank fusion

```text
rank_A
rank_B
rank_QA_proxy
```

затем RRF / weighted rank fusion.

### R3b — normalized fusion

Внутри candidate set нормализовать каналы:

```text
min-max
или z-score
```

и только потом:

```text
wA * A_norm
+ wB * B_norm
+ wQ * QA_norm
```

Нормализация должна быть детерминированной. Обработать отдельно случаи zero variance.

## 9.4. Обязательные абляции

```text
R3_0 = B2
R3_1 = B2 + A
R3_2 = B2 + QA confidence proxy
R3_3 = B2 + A + QA confidence proxy
```

Для `R3_2/R3_3` основной practically relevant вариант — `topK`.

`R3-full` оставить как диагностическую верхнюю по стоимости проверку.

## 9.5. Настройка

```text
CALIBRATION:
    weights / K_pre / normalization parameters

DEVELOPMENT:
    выбрать архитектурный вариант R3_0..R3_3
```

Не использовать `seen_exploratory_dev`.

Не выбирать вариант по INTERNAL_TEST.

## 9.6. Главные метрики

```text
Hit@1
Hit@5
Coverage@5
HN@1
Pairwise
MRR
EM
Token F1
Char F1
ROUGE-L
F1 | selected positive
```

И обязательно resource metrics:

```text
QA forwards/question
reader tokens/question
latency
```

Критический вопрос:

> сохраняет ли reader-aware ranking низкий HN@1 B2 и повышает ли downstream QA без недопустимого увеличения числа reader-forward-pass?

## 9.7. Опциональная, но НЕ default следующая гипотеза

Если unsupervised QA-confidence proxy не помогает, разрешено позже добавить отдельный `reader-success meta-ranker`, обучаемый **только на TRAIN** предсказывать успех frozen reader по gold-derived target.

Не реализовывать его в default V2 до результатов R1/R3, чтобы не смешивать две разные гипотезы.

# 10. C2 — Sweep по числу passages

Добавить:

```text
c-sweep
```

Проверить:

```text
M ∈ {1, 2, 3, 5, 8}
```

минимум для:

- frozen B2 ranking;
- выбранного reader-aware ranking.

## 10.1. Сохранить current C как baseline

Название:

```text
c_rawlogit
```

Для top-M:

- QA на каждом passage;
- лучший span внутри passage;
- глобальный выбор по raw `start+end`.

Не называть это обученным Multi-passage BERT и не интерпретировать raw score как калиброванную вероятность.

## 10.2. Разделить tuning и architecture evaluation

Для каждого уже выбранного ranking method:

```text
CALIBRATION:
    оценить M-grid и выбрать M_fixed* либо Pareto-кандидаты

DEVELOPMENT:
    проверить выбранный fixed-M без повторного подбора M
```

Не выбирать M по DEVELOPMENT, а затем тем же DEVELOPMENT доказывать его преимущество.

Если требуется сравнить несколько policy families, family выбирается на DEVELOPMENT, численные M/thresholds — на CALIBRATION.

## 10.3. Для каждого M

Считать:

```text
Hit@M
Coverage@M
label финального QA-source
EM
Token F1
Char F1
ROUGE-L
reader_tokens
QA forward passes
latency
```

Отдельно по:

```text
method
task
metric
material
```

И обязательно сохранить распределение:

```text
qa-source positive / HN / easy / ambiguous
```

## 10.4. Cost accounting

Если reader-aware уже вычислил QA для части top-M chunks и cache переиспользуется, scientific cost report должен считать **логическое число required QA forwards**, а не только фактический cached wall-clock.

# 11. C3 — Adaptive M

Добавить:

```text
c-adaptive
```

Гипотеза:

```text
если ranking top-1 надежен → читать 1 passage
если ranking не уверен → читать несколько passages
```

## 11.1. Confidence features должны быть сопоставимыми

Нельзя применять один абсолютный `tau_conf` к произвольному RRF/raw-score, если шкала зависит от числа кандидатов или конкретного fusion.

Для каждого ranking method явно определить confidence representation.

Предпочтительные признаки:

```text
calibrated B2 top1 probability
normalized top1 score
normalized top1-top2 margin
candidate_count
```

Опционально:

```text
entropy
```

только если score → probability преобразование формально определено и калибровано.

## 11.2. Базовая deterministic policy

Например:

```text
if confidence >= tau_conf and margin >= tau_margin:
    M = 1
else:
    M = M_uncertain
```

```text
M_uncertain ∈ {3, 5}
```

`tau_conf`, `tau_margin`, `M_uncertain` подбирать только на CALIBRATION.

Политика должна обрабатывать:

```text
candidate_count < M
candidate_count = 1
ties
zero-variance normalized scores
```

## 11.3. Более чистый вариант политики

Если абсолютные scores нестабильны, использовать только ranker-confidence признаки, не зависящие от raw scale, например нормализованный margin.

Не вводить обученный policy-classifier в основной V2, пока не проверена простая deterministic policy.

## 11.4. DEVELOPMENT comparison

```text
fixed M=1
fixed M=2
fixed M=3
fixed M=5
fixed M=8
best fixed M
adaptive M
```

Основной критерий adaptive M — quality/cost trade-off:

```text
QA quality
reader_tokens
QA forward passes
latency
```

Adaptive M полезен, если при сопоставимом качестве уменьшает стоимость либо при сопоставимой стоимости повышает качество.

# 12. D2-v2 — абляции старой смеси

Добавить:

```text
d2-ablate
```

Цель D2-v2 — разобрать **старые relevance/lexical/structure channels**. QA confidence proxy относится к R3 и не должен дублироваться в D2.

Минимальные варианты:

```text
D2a = B2
D2b = B2 + old RRF(dense, BM25)
D2c = B2 + section
D2d = B2 + old RRF(dense, BM25) + section   # ближайшая архитектура к E0-D2
D2e = B2 + A relevance CE
D2f = B2 + A relevance CE + section
```

Это позволяет отдельно проверить:

```text
эффект RRF
эффект section
их совместный эффект
можно ли заменить проблемный RRF сильным generic CE
```

## 12.1. Все fusion-каналы должны быть сопоставлены корректно

Нельзя складывать raw scores разных моделей без нормализации.

Для D2 использовать одну заранее выбранную схему:

```text
rank fusion
или
within-candidate-set normalization + weighted sum
```

Схема должна быть одинаково реализована во всех сравнимых D2 variants.

Численные weights подбирать на CALIBRATION.

Архитектурный вариант D2a..D2f выбирать на DEVELOPMENT.

## 12.2. Section features

Не смешивать:

```text
section_title_score
section_id_score
section_title_missing
```

Если title пуст, fallback `section_id` — отдельный канал, а не значение того же `section_title_score`.

Дополнительно отчитать результаты:

```text
section_title_present
section_title_missing
```

чтобы понять, приносит ли signal пользу только там, где title реально существует.

## 12.3. BM25

D1 не default.

BM25 участвует только внутри контролируемых D2b/D2d абляций.

Обязательно отдельный отчет для `metric`, где E0 показал систематический провал lexical retrieval.

## 12.4. Не смешивать D2 и R3

Reader-aware QA confidence proxy сравнивается в секции R3. Если финальный pipeline позже объединит лучший D2 structural signal с R3, это должен быть **новый отдельный ablation после того, как полезность каждого компонента показана отдельно**.

# 13. D4-v2 — sentence-level evidence

Код реализовать, но не включать в `DEFAULT_STAGES_V2`.

Stages:

```text
d4-train
d4
```

## 13.1. Когда запускать

D4 запускать после R1/R2/R3/C, если есть подтвержденный failure mode:

```text
positive chunk содержит gold evidence,
но reader часто берет неправильный span из-за шумного контекста.
```

## 13.2. Sentence offsets

Для каждого предложения хранить:

```text
sentence_id
chunk_start_char
chunk_end_char
source_start_char
source_end_char
```

Sentence splitter должен быть детерминированным и его версия фиксируется.

## 13.3. TRAIN labels

### Positive evidence

Предложение **полностью содержит** размеченный gold span.

Если gold span пересекает границу двух предложений:

- не обрезать gold;
- пометить single-sentence example как `cross_sentence_gold`;
- исключить его из D4a training либо сформировать отдельный двухпредложенческий unit по заранее описанному правилу.

### Strong negative evidence

Предложения только из:

```text
training_eligible_hn == true
```

Не использовать HN с `hn_gold_string_warning=true` как strong sentence negatives по default.

### Ignore/unlabeled

Не считать автоматически negative:

- другие предложения positive chunk;
- ambiguous chunks;
- соседние предложения gold sentence;
- flagged/noisy HN.

## 13.4. Модель и split protocol

Первый вариант:

```text
(question, sentence) -> evidence score
```

малый Cross-Encoder.

```text
TRAIN: model weights
CALIBRATION: checkpoint / numeric threshold
DEVELOPMENT: D4a vs D4b architecture comparison
```

Не добавлять complex contrastive objective в primary D4.

## 13.5. Два режима

```text
D4a = selected sentence
D4b = selected sentence ± 1 neighbor
```

Для D4b соседние предложения возвращать в исходном document order.

## 13.6. Offset mapping

При сборке compact context хранить mapping каждого блока:

```text
assembled_start
assembled_end
source_chunk_id
source_start_char
source_end_char
```

После QA source projection должен быть однозначным.

Unit tests:

```text
assembled span -> source span round-trip
source_text[source_start:source_end] == predicted_answer
```

## 13.7. Сравнение

```text
best pre-D4
vs D4a
vs D4b
```

Считать не только QA, но:

```text
reader tokens
latency
source projection errors
```

# 14. D5-v2 — Answer Aggregation

Запускать явно после того, как candidate quality приемлемо проверено:

```text
d5
```

D5 не должен маскировать плохой ranking путем голосования шумных spans.

## 14.1. На входе

Для нескольких candidate spans хранить:

```text
answer_text
normalized_answer
candidate_score
qa_span_score
ranking_score
source_chunk_id
source_start_char
source_end_char
```

## 14.2. Сначала de-dup overlapping evidence

Если один исходный source span повторился в нескольких sliding-window chunks, это одно evidence.

Сгруппировать по исходным координатам / сильному source-overlap до подсчета поддержки.

## 14.3. Exact normalization — первый вариант

```text
NFKC
casefold
whitespace normalization
dash normalization
punctuation normalization
```

Semantic clustering не включать в primary D5.

## 14.4. Нельзя агрегировать сырые QA logits как будто они сопоставимы

Не использовать по default:

```text
logsumexp(raw start_logit + end_logit)
```

между passages.

Это противоречит самой мотивации исследования о несопоставимости локальных scores.

Для aggregation использовать либо:

- calibrated/normalized final candidate score предыдущего pipeline;
- rank-based score;
- явно откалиброванный span score, если его сопоставимость была отдельно проверена.

## 14.5. Обязательные D5-абляции

### D5a — no aggregation control

Лучший одиночный candidate.

### D5b — exact-answer grouping + max

Для каждого normalized answer:

```text
group_score = max(calibrated candidate score over unique supports)
```

Это не дает частоте автоматического преимущества.

### D5c — support-aware aggregation

Только как отдельная абляция:

```text
group_score = max_score + lambda * log(1 + n_unique_support)
```

`lambda` подбирать на CALIBRATION; `lambda=0` является обязательным control.

На DEVELOPMENT сравнить D5a/D5b/D5c без дополнительного подбора lambda.

Частота/число supports — слабый дополнительный признак, а не доказательство истинности.

## 14.6. Provenance

Финальный answer обязан сохранять все unique supporting source spans.

# 15. D3 — отложить

Существующий D3 не удалять, но:

```text
DEFAULT_STAGES_V2: D3 отсутствует
```

Возвращаться к graph expansion только если error analysis подтвердит заметный класс случаев:

```text
одного chunk/sentence недостаточно,
а соседний/связанный chunk содержит необходимое supporting evidence.
```

До этого:

```text
experimental_optional = true
default = false
```

---

# 16. Unanswerable Audit

В E0 исключались инстансы без `robust_positive`.

Нельзя автоматически считать их true no-answer.

Добавить:

```text
unanswerable-audit
```

## 16.1. До final freeze

Gold-based audit выполнять только на:

```text
TRAIN
CALIBRATION
DEVELOPMENT
seen_exploratory_dev
```

Не раскрывать INTERNAL_TEST no-answer/gold analysis до freeze.

## 16.2. Для каждого исключенного instance

Сохранить:

```text
document_id
question_id
aspect
question
gold_answers
n_chunks
gold_string_anywhere_in_document
gold_string_anywhere_in_chunks
ambiguous_count
HN_count
```

Если исходный полный document text недоступен, `gold_string_anywhere_in_document` не подменять поиском по chunks; пометить поле как unavailable.

## 16.3. Heuristic audit statuses

Можно вычислить:

```text
true_no_answer_candidate
answer_exists_but_no_robust_positive
unknown
```

Но в schema явно записать:

```text
status_is_heuristic = true
```

Эти статусы не являются новой gold-разметкой.

No-answer model пока не обучать.

# 17. Порядок стадий

Рекомендуемые имена:

```text
prepare-v2
split-v2
e0-regression

reader-oracle
ambiguous-audit

b2-train
b2-eval
ambiguous-audit --answerability-model=b2

reader-aware
c-sweep
c-adaptive
d2-ablate

compare-v2
report-v2

freeze-final
final-eval

# optional later:
d4-train
d4
d5
unanswerable-audit
d3
```

## 17.1. DEFAULT_STAGES_V2

До final freeze:

```text
prepare-v2
split-v2
e0-regression
reader-oracle
ambiguous-audit
b2-train
b2-eval
ambiguous-audit --answerability-model=b2
reader-aware
c-sweep
c-adaptive
d2-ablate
compare-v2
report-v2
```

Не включать default:

```text
final-eval
d3
d4
d5
full no-answer modeling
trained multi-passage reader
```

## 17.2. Final evaluation запускается отдельно

После выбора pipeline:

```text
freeze-final
```

создает `final_freeze_manifest.json`.

Только затем явная команда:

```text
final-eval --allow-internal-test
```

После `final-eval` pipeline считается закрытым для этого INTERNAL_TEST. Любые последующие изменения — новая версия эксперимента и новый независимый evaluation protocol.

# 18. Seeds и reproducibility

Для trainable stages использовать минимум:

```text
42
43
44
```

Один и тот же набор seeds для сравниваемых trainable конфигураций.

Явно seed:

- Python;
- NumPy;
- PyTorch;
- CUDA;
- samplers;
- dynamic negative sampling;
- bootstrap;
- deterministic tie-breaking, если есть randomized component.

Split seed хранить отдельно от model seed.

Сохранять:

```text
split_seed
model_seed
bootstrap_seed
git commit
resolved config hash
model identifiers/tokenizer identifiers
text/chunk data version hash
torch
transformers
sentence-transformers
CUDA
GPU
```

Если полная детерминированность CUDA недостижима без существенного замедления, это явно зафиксировать; seeds все равно должны быть установлены.

# 19. Метрики

Сохранить E0-совместимые определения:

```text
Hit@1
Hit@5
Coverage@1
Coverage@5
HN@1
HN@5
MRR
Pairwise Pos>HN
EM
Token F1
Char F1
ROUGE-L
QA coverage
```

## 19.1. HN-метрики после введения clean flag

Сообщать оба варианта:

```text
Pairwise_all_HN        # совместимость с E0
Pairwise_clean_HN      # cleaner supervision/evaluation diagnostic
HN@K_all
HN@K_clean
```

Нельзя молча заменить старые labels новым subset и сравнивать число напрямую с E0 без пометки.

## 19.2. Reader diagnostics

```text
reader_solvable_EM
reader_solvable_F1>0
reader_solvable_F1>=50
reader_solvable_F1>=80
F1 | selected positive
```

## 19.3. Ambiguous diagnostics

```text
top1 ambiguous rate
top5 ambiguous rate
qa-source ambiguous rate
```

и отдельно по subtype.

## 19.4. Efficiency

```text
QA forward passes/question
reader tokens/question
unique chunks scored
mean/p50/p95 latency
peak GPU memory
peak RAM
cache hit rate
```

### Timing protocol

Для CUDA timing:

- сделать warm-up;
- `torch.cuda.synchronize()` перед стартом и после измеряемого блока;
- не смешивать data loading/preprocessing и pure model latency без явного указания;
- отдельно измерять `cold/uncached` и `cached/reuse` режим, если cache влияет на stage.

Для peak GPU memory:

```text
torch.cuda.reset_peak_memory_stats()
...
torch.cuda.max_memory_allocated()
```

или эквивалентный корректный profiler.

Не использовать одно постоянное значение CUDA context как peak memory метода.

## 19.5. Metric scales

Фиксировать метаданные шкал:

```text
Token F1: 0..100
Char F1: 0..1
ROUGE-L: 0..1
EM: 0..1
```

# 20. Статистика

Все сравнения парные на одних и тех же вопросах.

Bootstrap clusters:

```text
document_id
```

Рекомендуемо:

```text
n_bootstrap = 5000
```

configurable.

## 20.1. Multi-seed нельзя превращать в псевдорепликацию

Seeds одной модели не являются независимыми документами/вопросами.

Для trainable methods сообщать:

```text
metric mean ± std across seeds
```

Для paired bootstrap primary V2 comparison использовать один из заранее зафиксированных вариантов:

### Рекомендуемый вариант

Сначала для каждого `(document_id, question_id)` усреднить per-question metric/correctness по seeds, затем выполнить clustered bootstrap по `document_id`.

Детерминированный baseline A повторяется как одно и то же значение для каждого seed.

### Дополнительно

Показывать CI отдельно для каждого seed, чтобы видеть нестабильность направления эффекта.

Не объединять все `question × seed` строки как независимые observations.

## 20.2. Обязательные пары до final freeze

```text
A vs B2
B2 vs reader-aware
B2 top1 vs best fixed-M
best fixed-M vs adaptive-M
```

D2 ablations:

```text
B2 vs B2+RRF
B2 vs B2+section
B2 vs B2+RRF+section
B2 vs B2+A
B2 vs B2+A+section
```

R3 reader-aware comparisons сохранять отдельно, не смешивая их с D2 decomposition.

После явного запуска D4/D5:

```text
best pre-D4 vs D4
best pre-D5 vs D5
```

## 20.3. INTERNAL_TEST

На INTERNAL_TEST выполнить только **заранее замороженные comparisons**, описанные в `final_freeze_manifest.json`.

Не добавлять новые comparison pairs после просмотра test-результатов и не выбирать по нему «лучший» вариант из многих.

## 20.4. Сохранять

```text
observed delta
bootstrap mean
95% CI
n_questions
n_documents
number_of_seeds
aggregation_across_seeds
```

Точечное улучшение без CI описывать как направленный сигнал, а не доказанное преимущество.

# 21. Анализ по аспектам

Для каждого метода:

```text
method
task
metric
material
```

Минимум:

```text
n
Hit@1
Hit@5
HN@1
Pairwise
EM
Token F1
```

Особое внимание:
- `metric` — BM25 regressions;
- `task` — multi-passage gains.

На DEVELOPMENT aspect-level анализ разрешен для разработки. На INTERNAL_TEST после freeze aspect-level таблицы считаются post-hoc и не используются для изменения pipeline.

---

# 22. Артефакты

```text
artifacts/qa_chunk_selection_v2/
├── contract.json
├── config_resolved.json
├── progress.jsonl
├── final_freeze_manifest.json          # появляется только перед final-eval
├── data/
│   ├── dataset.jsonl
│   ├── splits_v2.json
│   ├── candidate_sets.jsonl
│   ├── data_summary.json
│   └── hn_quality_flags.jsonl
├── regression/
│   └── e0_regression.json
├── cache/
│   └── ...
├── models/
│   ├── b2/
│   └── sentence_evidence/
├── runs/
│   ├── b2/
│   ├── reader_aware_full/
│   ├── reader_aware_topk/
│   ├── c_m1/
│   ├── c_m2/
│   ├── c_m3/
│   ├── c_m5/
│   ├── c_m8/
│   ├── c_adaptive/
│   ├── d2_ablation/
│   ├── d4/
│   ├── d5/
│   └── final_internal_test/
├── analysis/
│   ├── reader_oracle/
│   ├── ambiguous_audit/
│   ├── unanswerable_audit/
│   ├── by_aspect/
│   ├── paired_comparisons.json
│   ├── ablation_table.csv
│   ├── efficiency.csv
│   └── error_analysis.csv
└── reports/
    ├── results_v2.md
    ├── final_internal_test.md
    └── experiment_manifest.md
```

# 23. Predictions schema

Сохранять **два файла**, чтобы gold физически не находился в runtime artifact scorer-а.

## 23.1. `predictions_raw.jsonl`

Генерируется inference-кодом без gold labels/answers:

```json
{
  "document_id": "...",
  "question_id": "...",
  "aspect": "...",
  "question": "...",

  "ranking_method": "...",
  "candidate_count": 0,
  "selected_chunk_id": "...",
  "qa_chunk_id": "...",

  "qa_answer": "...",
  "qa_chunk_start_char": 0,
  "qa_chunk_end_char": 0,
  "source_start_char": 0,
  "source_end_char": 0,

  "a_score": 0.0,
  "b_score": 0.0,
  "qa_confidence_proxy": 0.0,
  "final_ranking_score": 0.0,
  "used_m": 1
}
```

## 23.2. `predictions_scored.jsonl`

Создается отдельным evaluation-слоем и может добавлять:

```json
{
  "selected_chunk_label": "...",
  "qa_chunk_label": "...",
  "gold_answers": ["..."],
  "em": 0,
  "token_f1": 0.0,
  "char_f1": 0.0,
  "rouge_l_f1": 0.0
}
```

Не использовать `predictions_scored.jsonl` как input следующего inference-stage, если stage теоретически мог бы прочитать gold fields.

# 24. Chunk score schema

Аналогично разделить raw и evaluation-enriched artifacts.

## 24.1. Raw chunk scores

```json
{
  "document_id": "...",
  "question_id": "...",
  "chunk_id": "...",

  "dense_score": 0.0,
  "a_relevance_score": 0.0,
  "b_logit": 0.0,
  "b_probability": 0.0,

  "qa_best_answer": "...",
  "qa_start_logit": 0.0,
  "qa_end_logit": 0.0,
  "qa_span_score": 0.0,
  "qa_span_margin": null,

  "section_title_score": 0.0,
  "section_id_score": 0.0,
  "section_title_missing": false,

  "rank_dense": 0,
  "rank_a": 0,
  "rank_b": 0,
  "rank_qa_proxy": 0,

  "final_score": 0.0
}
```

## 24.2. Evaluation enrichment

Отдельный слой добавляет:

```text
label
primary_class
hn_gold_string_warning
training_eligible_hn
gold overlap / metrics
```

Gold label не должен присутствовать в object, передаваемом scorer/ranker.

# 25. Cache

Не прогонять одинаковый `(question, chunk, model, config)` повторно без необходимости.

Cache:

```text
dense
A
B2
QA raw outputs
sentence boundaries
section features
BM25
```

## 25.1. Cache key

Минимум:

```text
document_id
question_id
chunk_id
question_text_hash
chunk_text_hash
model_identifier/model_revision
model_config_hash
tokenizer_identifier/tokenizer_revision
inference_config_hash
```

Одних IDs недостаточно: если text/preprocessing изменились при тех же IDs, cache должен инвалидироваться.

## 25.2. QA cache

QA cache обязателен для R1/R3/C/D4, но resource reporting должен различать:

```text
algorithmic model work
vs
wall-clock with cache reuse
```

Для сравнения вычислительной стоимости методов выполнять отдельный `uncached benchmark` либо считать число forward passes/tokens из логической схемы, чтобы cache не создавал ложное впечатление, что метод дешевле.

# 26. Tests

Обязательные unit/integration tests.

## 26.1. Leakage

- runtime scorer/ranker не принимает gold fields;
- `predictions_raw` не содержит gold;
- INTERNAL_TEST inaccessible без explicit allow flag;
- oracle output не используется model-selection функциями.

## 26.2. Split

Нет пересечений `document_id` между:

```text
TRAIN
CALIBRATION
DEVELOPMENT
INTERNAL_TEST
```

## 26.3. Determinism

Одинаковый seed → одинаковые:

- split;
- negative sampling sequence;
- tie-breaking;
- adaptive-M decisions.

## 26.4. E0 regression

Старые результаты воспроизводятся в допустимой tolerance.

## 26.5. Source offsets

```text
chunk_text[chunk_start:chunk_end] == answer
source_text[source_start:source_end] == answer
```

## 26.6. D4 cross-sentence gold

Gold span, пересекающий sentence boundary, не должен молча обрезаться или получать неверный single-sentence label.

## 26.7. Overlap de-duplication

Один source span в двух overlapping chunks = одно evidence.

## 26.8. Adaptive M

```text
M в допустимом наборе
M <= candidate_count
candidate_count=1 обрабатывается
confidence scale соответствует ranking method
```

## 26.9. B2 question balancing

Проверить, что вопрос с 20 negatives не получает автоматически в 10 раз больший суммарный training weight, чем вопрос с 2 negatives, если включен question-balanced mode.

## 26.10. Noisy HN

HN с `hn_gold_string_warning=true` не попадает в default strong-negative training.

## 26.11. Cache invalidation

Изменение `chunk_text`, question, model revision или inference config меняет cache key.

## 26.12. Metric scales

```text
Token F1: 0..100
Char F1: 0..1
ROUGE-L: 0..1
```

## 26.13. Timing

CUDA benchmark synchronizes device and peak-memory reset выполняется перед измерением.

# 27. Smoke mode

Каждая новая stage должна иметь smoke mode:
- несколько документов;
- 1 seed;
- минимальное обучение;
- проверка schema/cache/offsets/metrics.

Smoke результаты не включать в scientific report.

---

# 28. Логи

`progress.jsonl`:

```text
timestamp
stage
seed
split
model
config_hash
n_documents
n_questions
elapsed
status
```

Train:

```text
epoch
train_loss
calibration_metric
checkpoint
early_stopping
```

Adaptive M:

```text
count M=1
count M=3
count M=5
...
```

Reader-aware:

```text
features
normalization
weights
```

---

# 29. Автоматический `results_v2.md`

Отчет генерировать из JSON/JSONL/CSV, не переписывать числа вручную.

Разделы:

```text
1. Configuration and code/data versions
2. Splits and INTERNAL_TEST lock status
3. E0 regression
4. Reader oracle
5. Ambiguous audit
6. HN quality warnings
7. B2 variants and seed stability
8. Reader-aware: full-cost vs topK
9. M sweep
10. Adaptive M
11. D2 ablations
12. D4, если запускался
13. D5, если запускался
14. By aspect
15. By HN bucket
16. Efficiency: cached and uncached/logical cost
17. Multi-seed statistics and clustered bootstrap
18. Error analysis
19. Limitations
20. Confirmed findings
21. Open hypotheses
22. Seen exploratory DEV replication
23. INTERNAL_TEST results — только после final-eval
```

До `final-eval` раздел 23 должен явно писать:

```text
INTERNAL_TEST NOT OPENED
```

# 30. Gating: когда переходить дальше

## После R1

Если best-positive oracle низок:

```text
reader limitation dominates
```

Тогда чистое усложнение ranker-а имеет ограниченный потолок.

Если почти всегда существует reader-solvable positive, но ranker его не выбирает:

```text
specific-positive selection dominates
```

и reader-aware направление особенно оправдано.

## После ambiguous-audit

Если ambiguous остается главным distractor:

```text
priority = evidence sufficiency / ambiguous discrimination
```

выше graph expansion.

Но не relabel автоматически.

## После B2

Не переходить к R3 с моделью, у которой train pairwise≈1 и большой development gap, если более стабильный B2 вариант дает сопоставимый development result.

## После reader-aware

Перспективный эффект должен оцениваться одновременно по:

```text
HN@1
Hit@1
F1 | selected positive
final QA
QA forwards/tokens
```

Если `R3-full` помогает, а `R3-topK` нет, это сигнал о цене prefilter recall; не объявлять full режим практическим без ресурсного анализа.

## После M-sweep

Если multi-passage помогает преимущественно при низкой ranking confidence — переходить к adaptive M.

## После adaptive M

Сравнить Pareto trade-off quality/cost, а не только максимальный F1.

## После D4

Если D4 не улучшает качество и не дает ресурсной экономии — не включать его в final pipeline.

## После D5

Если support-aware D5 выигрывает только за счет дублированных/частых occurrences, а после source de-dup эффект исчезает — не считать aggregation подтвержденной.

## D3

Только при доказанном distributed-evidence failure mode.

## Перед INTERNAL_TEST

Все архитектурные решения, thresholds, seeds, components и comparison pairs должны быть записаны в `final_freeze_manifest.json`.

# 31. Что запрещено

1. Перезаписывать E0.
2. Использовать seen official DEV как чистый TEST.
3. Использовать DEVELOPMENT как финальный независимый результат после многократного выбора архитектуры.
4. Открывать INTERNAL_TEST до final freeze.
5. Выбирать метод по одному seed.
6. Использовать gold в runtime inference.
7. Автоматически relabel ambiguous.
8. Автоматически считать все no-robust-positive true no-answer.
9. Включать D1 в default.
10. Считать D2 доказанным победителем по E0.
11. Автоматически запускать D3.
12. Называть current raw-logit C полноценным Multi-passage BERT.
13. Называть raw QA span score истинной extractability/probability correctness.
14. Складывать raw A/B/QA scores без normalization/rank fusion.
15. Использовать `logsumexp` сырых локальных QA logits в D5 без доказанной/выполненной calibration.
16. Считать overlapping chunks независимыми evidence.
17. Подбирать новые параметры по `seen_exploratory_dev`.
18. Одновременно менять reader и ranker без отдельной абляции.
19. Удалять неудобные HN/ambiguous после просмотра DEVELOPMENT/INTERNAL_TEST.
20. Использовать TRAIN как доказательство обобщения.
21. Использовать flagged HN с gold-string warning как default strong negatives без отдельной sensitivity-абляции.
22. Считать cached wall-clock честным сравнением вычислительной стоимости без учета логического числа model forward passes.
23. Считать seed×question независимыми observations при статистическом тестировании.

# 32. Предпочтительная модульная структура

Если это согласуется с текущим кодом:

```text
experiments/qa_chunk_selection/
└── v2/
    ├── split.py
    ├── regression.py
    ├── reader_oracle.py
    ├── ambiguous_audit.py
    ├── answerability_v2.py
    ├── reader_aware.py
    ├── multipassage_sweep.py
    ├── adaptive_m.py
    ├── d2_ablation.py
    ├── sentence_evidence.py
    ├── answer_aggregation_v2.py
    ├── unanswerable_audit.py
    ├── statistics.py
    ├── report.py
    └── schemas.py
```

Если существующая архитектура лучше — расширять ее, не дублируя корректные E0-функции.

---

# 33. Приоритет реализации

## Priority 0
```text
inspect code
freeze E0
v2 artifact root
clean split
e0 regression
```

## Priority 1
```text
reader-oracle
ambiguous-audit
```

## Priority 2
```text
B2 variants
3 seeds
early stopping
```

## Priority 3
```text
reader-aware
```

## Priority 4
```text
M sweep
adaptive M
```

## Priority 5
```text
D2 ablations
```

## Priority 6
```text
D4
```

## Priority 7
```text
D5
```

Deferred:
```text
D3
trained multi-passage reader
full no-answer model
```

---

# 34. Финальный self-check агента

После реализации проверить:

- [ ] E0 не изменен.
- [ ] E0 regression проходит.
- [ ] Split строго document-level.
- [ ] Есть TRAIN/CALIBRATION/DEVELOPMENT/INTERNAL_TEST.
- [ ] INTERNAL_TEST закрыт explicit flag до freeze.
- [ ] `final_freeze_manifest.json` фиксирует финальную конфигурацию.
- [ ] Official DEV = `seen_exploratory_dev`.
- [ ] Raw inference artifacts не содержат gold.
- [ ] Reader oracle diagnostic-only и не используется для selection.
- [ ] Ambiguous не relabeled автоматически.
- [ ] HN с gold-string warning имеет отдельный flag.
- [ ] Default B2 не использует flagged HN как strong negatives.
- [ ] B2 использует seeds 42/43/44.
- [ ] B2 question-balanced либо явно аблирован против flat sampling.
- [ ] Есть B2 без статического oversampling HN.
- [ ] Reader-aware называет QA score confidence proxy, а не ground-truth extractability.
- [ ] Есть `R3-full` и practically relevant `R3-topK` либо документированная причина не реализовывать один из них.
- [ ] Raw channels не складываются без normalization/ranks.
- [ ] C реально проверяет M=1/2/3/5/8.
- [ ] Adaptive M использует scale-valid inference features.
- [ ] D1 отсутствует в DEFAULT_STAGES_V2.
- [ ] D2 разбирает RRF/section/A отдельно и не дублирует R3 QA-proxy.
- [ ] D3 не default.
- [ ] D4 обрабатывает cross-sentence gold и сохраняет source mapping.
- [ ] D5 не агрегирует raw QA logits как сопоставимые confidence.
- [ ] D5 de-duplicates overlap до support counting.
- [ ] Unanswerable audit не делает автоматический no-answer gold.
- [ ] Metric definitions совместимы с E0 и scale сохранен.
- [ ] Clustered bootstrap по document_id.
- [ ] Multi-seed statistics не используют seed×question как независимые rows.
- [ ] CUDA timing синхронизирован; peak memory reset/read корректен.
- [ ] Cache key включает hashes текста и revisions моделей.
- [ ] Efficiency report различает cache reuse и logical/uncached cost.
- [ ] `results_v2.md` строится из artifacts.
- [ ] Smoke mode работает.
- [ ] Полный run воспроизводим через data/config/model/seed/commit hashes.

# 35. Требуемый итог работы

После выполнения должны существовать:

1. Код V2 без изменения E0 artifacts.
2. Конфиги V2.
3. Regression E0.
4. Document-level `TRAIN/CALIBRATION/DEVELOPMENT/INTERNAL_TEST` split.
5. INTERNAL_TEST lock + `final_freeze_manifest.json`.
6. Reader oracle.
7. Ambiguous audit.
8. HN quality flags без автоматического relabeling.
9. B2 variants с question-balanced training и 3 seeds.
10. Reader-aware reranking с full-cost diagnostic и практическим top-K вариантом.
11. M sweep.
12. Adaptive M.
13. D2 ablations.
14. Реализованные, но не default D4/D5.
15. D3 optional/deferred.
16. Unanswerable audit infrastructure.
17. Tests на leakage, offsets, balancing, cache, timing и reproducibility.
18. Воспроизводимые raw/scored JSONL, JSON, CSV.
19. Автоматический `results_v2.md`.
20. Отдельный `final_internal_test.md` после явного frozen evaluation.

Главный принцип:

> Код должен не максимизировать F1 любой ценой, а разделять источники ошибок и экспериментально проверять, что для качественного extractive QA нужен выбор evidence, одновременно релевантного, достаточного и **пригодного для текущего reader**, при этом внутренние QA logits рассматриваются лишь как confidence proxy; при неопределенности ranking допускается адаптивное использование нескольких candidates, а любой прирост оценивается вместе с вычислительной стоимостью и на закрытом после разработки holdout.

