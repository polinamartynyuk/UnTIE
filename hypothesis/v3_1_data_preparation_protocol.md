# Протокол подготовки дополнительных данных для V3.1: target-status overlay

## 0. Назначение документа

Этот документ описывает **отдельный обязательный этап подготовки данных**, который необходимо завершить до продолжения экспериментов V3/V3.1 по отбору evidence-чанков для aspect-oriented extractive QA.

Подготовка данных не означает создание нового корпуса и не изменяет исходный SciREX / Phase 1 dataset.

Цель этапа:

> получить проверенный дополнительный слой разметки (`V3 target-status overlay`) для исходных `ambiguous`-чанков, чтобы различать:
>
> - evidence, относящийся к целевой работе;
> - правдоподобную аспектную информацию, относящуюся к другому объекту;
> - частичное evidence;
> - случаи, которые невозможно надёжно классифицировать.

Именно отсутствие этого overlay не позволило в текущем V3 полноценно проверить главную гипотезу `target evidence vs target mismatch`.

---

# 1. Исходное состояние после текущего V3

Текущий эксперимент уже сформировал очередь для экспертной разметки:

```text
annotation/queue.jsonl
```

Текущий размер:

```text
400 чанков
```

Структура выборки:

```text
hard_case       = 200
representative  = 200
```

По исходным ambiguous subtype:

```text
gold_supporting_robust_negative = 192
false_negative_candidate        = 105
question_sensitive              = 103
```

Файл:

```text
annotation/target_status.jsonl
```

на текущий момент пуст.

Следовательно:

```text
verified_target_mismatch = 0
verified_target_evidence = 0
```

и в текущем V3:

```text
P_v3_target == P_original
TargetMismatch@1 = unavailable
pairwise P_v3_target > TM = unavailable
```

Это означает, что B1/B2/B3 не получили нового target-mismatch supervision.

---

# 2. Источники данных

Использовать только уже разблокированную часть V3.

Для **primary overlay V3.1** основной universe должен совпадать с текущим grouped-CV:

```text
453 answerable document–aspect
240 документов
```

То есть вопросы, для которых существует хотя бы один исходный `robust_positive`.

Это подмножество происходит из:

```text
V2 train
+
V2 calibration
+
V2 development
```

Из 268 unlocked документов 28 документов без `robust_positive`
не входят в primary V3.1 CV.

Если в будущем для них будет обнаружено потенциальное evidence,
сохранять его как отдельный diagnostic false-negative/no-answer audit,
но не добавлять задним числом в primary V3.1 universe.

INTERNAL_TEST:

```text
31 документ
54 answerable-вопроса
```

остаётся полностью закрытым.

Запрещено:

```text
просматривать INTERNAL_TEST для разметки;
строить по нему taxonomy;
искать там target mismatch;
использовать его для примеров;
использовать его для проверки annotation guideline.
```

Official SciREX DEV (`seen_exploratory_dev`) также не используется для формирования primary V3 overlay.

---

# 3. Единица разметки

Единица экспертного решения:

```text
(document_id, aspect/question, chunk_id)
```

Аннотатор решает не общую тематическую релевантность чанка, а:

> является ли данный чанк достаточным или частичным evidence для **целевого аспекта именно текущей статьи**.

Разметка должна учитывать:

```text
текущий document_id;
аспект;
канонический aspect question;
текст чанка;
при необходимости локальный/документный контекст.
```

---

# 4. Исходная разметка не изменяется

Исходные поля Phase 1 / SciREX-derived labels являются неизменяемыми.

Запрещено переписывать:

```text
source_label
source_ambiguous_subtype
primary_class
```

Новая разметка записывается только отдельным overlay.

Минимальная схема записи:

```text
document_id
question_id / aspect_id
aspect
question
chunk_id

source_label
source_ambiguous_subtype

v3_target_status
v3_target_value
v3_gold_relation
v3_mismatch_type
v3_owner_text
v3_support_scope
v3_evidence_sentence_ids
v3_evidence_char_offsets
v3_annotation_confidence
v3_annotation_reason

annotation_sampling_stratum
annotation_sampling_frame
sampling_probability
annotation_version
annotator_id
annotation_timestamp
```

Дополнительно хранить, когда применимо:

```text
source_sentence
target_entity_or_value
mismatch_entity_or_value
context_used
context_source_ids / offsets
```

Локализация evidence особенно желательна для `target_evidence`, `target_mismatch` и `partial_support`:

```text
v3_evidence_sentence_ids
v3_evidence_char_offsets
```

Если точные char offsets разметить надёжно нельзя, сохранить хотя бы sentence id / sentence text.
Эти поля позже позволят анализировать discourse attribution и D4 без повторной ручной разметки.

---

# 5. Основной статус `v3_target_status`

Допустимы ровно четыре основных значения:

```text
target_evidence
target_mismatch
partial_support
unresolved
```

## 5.1. `target_evidence`

Назначать, если чанк содержит evidence, достаточное для ответа на аспектный вопрос **относительно текущей статьи**.

Практический критерий:

> если бы этот фрагмент был основным доступным evidence, можно ли на его основании корректно утверждать значение аспекта именно для данной статьи?

Примеры типов утверждений:

```text
авторы предлагают метод X;
в работе используется датасет Y;
статья решает задачу Z;
результаты оцениваются метрикой M.
```

## 5.2. `target_mismatch`

Назначать, если чанк содержит правдоподобную информацию нужного аспектного типа, но она относится **не к целевому объекту исследования**.

Типовые случаи:

```text
другой метод;
другая работа;
related work;
baseline;
сравниваемая система;
другой датасет;
другой benchmark;
другой эксперимент;
чужой результат;
внешняя цитируемая работа.
```

Важно:

```text
семантически подходящая сущность
≠
target evidence.
```

## 5.3. `partial_support`

Назначать, если чанк относится к текущей работе и действительно содержит полезную часть evidence, но **сам по себе недостаточен** для уверенного ответа.

Например:

```text
значение раскрывается только вместе с соседним чанком;
упоминается сокращение без расшифровки;
описан компонент метода, но не сам целевой метод;
указана часть набора данных без достаточной идентификации.
```

`partial_support` по умолчанию:

```text
не positive;
не strong negative.
```

Для него желательно заполнить `v3_support_scope`:

```text
same_sentence
neighbor_sentence
neighbor_chunk
document_wide
unclear
```

Это поле не используется как runtime feature в V3.1 автоматически; оно нужно для D3/D4 diagnostics.

## 5.4. `unresolved`

Назначать, если по доступному контексту нельзя надёжно решить:

```text
target evidence это или нет;
принадлежит ли значение текущей работе;
достаточно ли evidence;
является ли исходная разметка ошибочной.
```

Правило:

> лучше `unresolved`, чем искусственно создать неправильный positive или negative.

`unresolved` не использовать как strong supervised label.

---

# 6. Дополнительная taxonomy для `target_mismatch`

Для `v3_target_status = target_mismatch` обязательно, если возможно, заполнять:

```text
v3_mismatch_type
```

Рекомендуемые значения:

```text
related_work
external_cited_work
baseline
compared_method
other_dataset
other_benchmark
other_experiment
other_metric_context
background_definition
historical_result
other
unclear_mismatch_type
```

Дополнительно сохранять:

```text
v3_owner_text       # кому/чему фактически принадлежит значение, если это можно установить
v3_annotation_reason
```

Пример логики:

```text
В чанке упоминается CIFAR-10,
но он используется сравниваемым baseline,
а не целевой системой статьи.
```

Цель `mismatch_type`:

1. анализировать реальные причины ошибок B0;
2. понять, какой discourse/attribution signal нужен следующему reranker;
3. не сводить все target mismatch к одной неинтерпретируемой группе.

---

# 7. Поле `v3_target_value`

Для `target_evidence` и `target_mismatch` желательно извлечь само аспектное значение:

```text
v3_target_value
```

Примеры:

```text
method   → название метода / модели
task     → решаемая задача
material → используемый датасет / материал
metric   → метрика оценки
```

Для `target_mismatch` это значение означает:

> какое правдоподобное, но нецелевое аспектное значение содержится в чанке.

Это понадобится для:

```text
ошибочного target attribution;
анализа конкурирующих сущностей;
D5/aggregation audit;
reader/span error analysis.
```

---

# 8. Отношение к исходному gold: `v3_gold_relation`

`v3_gold_relation` описывает **отношение извлечённого аспектного значения к существующему gold**, а не принадлежность evidence.

Заполнять его для всех случаев, где выделено `v3_target_value`:

```text
matches_existing_gold
semantically_equivalent_to_existing_gold
new_valid_value_candidate
no_gold_match_expected
unclear
```

Важно: `target_status` и `gold_relation` независимы. Например, `target_mismatch` может содержать ту же строку, что и gold, но относить её к другому объекту. Тогда допустимо:

```text
v3_target_status = target_mismatch
v3_gold_relation = matches_existing_gold
```

Для `target_evidence` значение `no_gold_match_expected` считать несогласованным: если evidence действительно целевое, но его значения нет в gold, использовать `new_valid_value_candidate` и отправить случай в audit.

## 8.1. `matches_existing_gold`

Чанк поддерживает существующее gold-значение.

## 8.2. `semantically_equivalent_to_existing_gold`

Формулировка отличается, но аспектное значение эквивалентно существующему gold.

## 8.3. `new_valid_value_candidate`

Эксперт считает значение корректным для текущей статьи, но его нет в существующем gold.

Такое значение:

```text
НЕ добавлять автоматически в primary gold;
НЕ считать автоматически новым official positive;
сохранить в отдельный audit.
```

## 8.4. Primary V3 positive overlay

В primary V3 comparison в дополнительный positive set включать только:

```text
v3_target_status = target_evidence

AND

v3_gold_relation ∈ {
    matches_existing_gold,
    semantically_equivalent_to_existing_gold
}
```

`new_valid_value_candidate` анализировать отдельно:

```text
augmented-gold sensitivity analysis
```

и не смешивать с primary V2-comparable evaluation.

---

# 9. Формирование множеств после разметки

После annotation freeze определить:

```text
P_original =
    исходные robust_positive
```

```text
P_v3_target =
    P_original
    ∪ verified overlay target_evidence,
      для которого gold_relation допустим для primary V3
```

```text
TM_verified =
    verified overlay target_mismatch
```

Не включать в `P_v3_target`:

```text
partial_support
unresolved
new_valid_value_candidate, если не зафиксирована отдельная augmented-gold версия
```

Не включать в `TM_verified`:

```text
весь исходный ambiguous автоматически.
```

---

# 10. Sampling: две обязательные части и optional coverage-completion

Текущую схему сохранять:

```text
hard_case
+
representative
```

## 10.1. Hard-case stratum

Приоритетные случаи:

```text
B0 top-1 ambiguous;
B0 top-5 ambiguous;
высокий B0 answerability score;
GSRN;
question_sensitive;
high-score false_negative_candidate.
```

Назначение:

> получить обучающие примеры именно вокруг действующей границы ошибок B0.

## 10.2. Representative stratum

Выборка должна охватывать:

```text
aspect;
source ambiguous subtype;
B0 score quantile;
позицию в документе;
section;
длину chunk.
```

Назначение:

> не обучить новую модель только на наиболее трудных / model-selected ошибках и получить более честное представление о структуре ambiguous.

## 10.3. Coverage-completion stratum — только до freeze

Если после первых 400 примеров coverage недостаточен, разрешён дополнительный проход, сформированный **только по уже существующему frozen B0 OOF**, а не по результатам новых V3.1-моделей.

Приоритет:

```text
ambiguous в B0 top-1;
ambiguous в B0 top-5;
при необходимости — union B0 top-10.
```

Цель:

> повысить покрытие именно того candidate space, внутри которого будет работать V3.1 target reranker.

Такие примеры маркировать:

```text
annotation_sampling_stratum = coverage_completion
```

Их нельзя смешивать с representative stratum при оценке prevalence.

Для representative sampling сохранять `sampling_probability`, если она известна. Для deterministic hard-case/coverage-completion допускается `sampling_probability = null`, но sampling rule должен быть записан в manifest.

---

# 11. Нельзя показывать аннотатору model score

Файл для аннотатора не должен содержать:

```text
B0 score
B1/B2 score
attention score
model rank
predicted class новой модели
```

Model score допустим только в отдельном:

```text
sampling_metadata
```

для формирования очереди.

Это уменьшает confirmation bias.

---

# 12. Какой контекст разрешено показывать аннотатору

Минимально:

```text
aspect
question
chunk text
```

Если этого недостаточно, разрешено раскрывать:

```text
section title;
предыдущее предложение;
следующее предложение;
соседний chunk;
локальный fragment документа.
```

При использовании дополнительного контекста сохранить:

```text
context_used = ...
```

Не следует сразу показывать весь документ, если локального контекста достаточно.

Причины:

```text
снизить время разметки;
не создавать лишнюю субъективность;
фиксировать, сколько контекста реально требуется для attribution.
```

---

# 13. Рекомендуемый пошаговый алгоритм аннотации одного чанка

## Шаг 1. Определить аспект

```text
task / method / material / metric
```

## Шаг 2. Найти аспектное значение в чанке

Если явного значения нет:

```text
часто partial_support / unresolved
```

но решение зависит от контекста.

## Шаг 3. Определить принадлежность значения

Задать вопрос:

> это значение относится к текущей работе или к другому объекту?

Если к другому объекту:

```text
target_mismatch
```

## Шаг 4. Проверить достаточность

Если относится к текущей работе:

```text
достаточно → target_evidence
недостаточно → partial_support
```

## Шаг 5. Проверить отношение к gold

Заполнить:

```text
v3_gold_relation
```

## Шаг 6. Для mismatch определить subtype

Заполнить:

```text
v3_mismatch_type
```

## Шаг 7. Записать reason

Краткое, проверяемое объяснение.

## Шаг 8. Если остаётся существенная неопределённость

```text
unresolved
```

---

# 14. Confidence аннотации

Рекомендуемая шкала:

```text
high
medium
low
```

Интерпретация:

### `high`

Принадлежность evidence однозначна.

### `medium`

Есть небольшая неопределённость, но решение достаточно обосновано.

### `low`

Решение зависит от неоднозначного контекста.

Правило для primary verified overlay:

```text
high + medium
```

можно использовать после QC.

`low`:

```text
направлять на adjudication
или
переводить в unresolved.
```

---

# 15. Вторичная проверка / adjudication

Если возможно, минимум:

```text
10–20% verified examples
```

должен независимо разметить второй эксперт.

Особенно приоритетно проверять:

```text
target_mismatch;
target_evidence из исходного ambiguous;
new_valid_value_candidate;
low confidence;
method;
metric.
```

Оценить:

```text
raw agreement
confusion matrix
agreement отдельно по target_evidence/target_mismatch
Cohen's kappa для двух аннотаторов, если применимо
Krippendorff's alpha / иной устойчивый коэффициент, если схема разметки этого требует
```

При сильном дисбалансе классов не интерпретировать один kappa без confusion matrix и per-class agreement.

При расхождении:

```text
adjudicated_status
adjudication_reason
```

Финальный overlay должен хранить именно adjudicated label.

---

# 16. Обязательный audit исходной разметки

Отдельно собрать случаи:

```text
source_label = ambiguous
AND
v3_target_status = target_evidence
```

Это потенциальные false negatives исходной разметки.

Разделить:

```text
поддерживает существующий gold;
семантически эквивалентен gold;
содержит новое возможное gold-значение.
```

Не переписывать исходный `source_label`.

---

# 17. Контроль качества после разметки

Перед использованием overlay проверить:

## 17.1. Schema validation

Нет неизвестных значений taxonomy.

## 17.2. Completeness

Для каждого аннотированного чанка заполнены обязательные поля.

## 17.3. Consistency

Например:

```text
target_mismatch
→ mismatch_type заполнен или explicitly unclear
```

```text
target_evidence
→ gold_relation заполнен
```

## 17.4. Duplicate consistency

Одинаковый chunk не должен иметь конфликтующие финальные labels.

## 17.5. Document/aspect consistency

Сходные чанки одного document–aspect проверить на логические противоречия.

## 17.6. Leakage check

Ни один INTERNAL_TEST id не присутствует в overlay.

---

# 18. Проверка распределения overlay

После завершения разметки построить таблицы:

```text
target_status × aspect
target_status × source subtype
target_status × sampling stratum
target_status × B0 score quantile
target_status × section
mismatch_type × aspect
mismatch_type × source subtype
confidence × target_status
```

Отдельно показать:

```text
hard_case
representative
coverage_completion, если использовался
```

Не объединять их сырые доли для оценки prevalence.

Для оценки распространённости `target_mismatch` во всей ambiguous-популяции использовать:

```text
representative stratum
```

или заранее корректно рассчитанные sampling weights / inverse-probability weights. Hard-case и coverage-completion намеренно смещены к трудным примерам.

---

# 19. Coverage и effective target role

Новая overlay-разметка существует только для части исходных `ambiguous`, поэтому необходимо отличать **target role** от исходного `source_label`.

Для оценки сформировать вычисляемое поле:

```text
effective_target_role =
    target_evidence_original      # source robust_positive
    target_evidence_overlay       # verified overlay target_evidence, допустимый gold_relation
    target_mismatch_verified      # verified overlay target_mismatch
    known_other_negative          # source HN/easy; не считать это verified TM
    partial_support
    unresolved
    unknown_ambiguous             # ambiguous без adjudicated overlay
```

`known_other_negative` участвует в общей известности роли кандидата, но **не** добавляется в `TM_verified`.

Если размечена не вся совокупность ambiguous, считать минимум:

```text
TargetStatusCoverage@1
TargetStatusSlotCoverage@5/@10
TargetStatusCompleteCoverage@5/@10
```

Для top-1:

\[
Coverage^{target}_{@1}
=
\frac{
\#\{q:\ effective\_target\_role(top1_q)\neq unknown\_ambiguous\}
}{N}.
\]

Для K>1:

```text
SlotCoverage@K =
число top-K slots с известным effective_target_role /
общее число top-K slots
```

```text
CompleteCoverage@K =
доля вопросов, у которых все top-K slots имеют известный effective_target_role
```

При неполной разметке выводить:

```text
TargetMismatch@1_lower_bound
TargetMismatchRate@1_known_role
```

где:

```text
TargetMismatch@1_lower_bound =
verified TM top1 / all questions
```

```text
TargetMismatchRate@1_known_role =
verified TM top1 /
top1 questions с известным effective_target_role
```

Эта величина не является несмещённой оценкой prevalence, если annotation set формировался преимущественно hard-case sampling. Всегда показывать stratum composition.

Дополнительно для исходных ambiguous top-1 можно считать:

```text
TargetMismatchRate@1_overlay_adjudicated =
verified TM среди adjudicated original-ambiguous top1 /
все adjudicated original-ambiguous top1
```

Она характеризует только adjudicated ambiguous subset и также должна сопровождаться sampling-stratum composition.

Для `TargetEvidenceHit@K_v3` при неполном overlay результат является **консервативной нижней оценкой expanded target evidence recall**: непроверенный ambiguous не превращать автоматически ни в positive, ни в negative.

Для финального QA-source считать:

```text
WrongTargetAnswerCoverage =
число вопросов, где source финального ответа имеет известный effective_target_role / N
```

```text
WrongTargetAnswerRate_lower_bound =
число вопросов, где финальный QA source = target_mismatch_verified / N
```

```text
WrongTargetAnswerRate_known_role =
число вопросов, где финальный QA source = target_mismatch_verified /
число вопросов с известным effective_target_role финального QA source
```

`GoldMatchFromNonOriginalPositive` трактовать только как diagnostic discrepancy: исходный ambiguous/HN source теоретически может быть false negative исходной разметки, поэтому эта метрика сама по себе не доказывает wrong-target error.

TargetMismatch нельзя использовать как hard safeguard, если coverage недостаточен.

Сохраняется ранее принятый ориентир:

```text
minimum TargetStatusCoverage@1 = 0.95
```

для использования TM@1 как полноценного model-selection safeguard.

---

# 20. Баланс по document–aspect и защита от дублирования

Перед freeze проверить, что overlay не состоит из большого числа почти одинаковых чанков нескольких вопросов.

Обязательно сохранить:

```text
unique document_id
unique document–aspect
annotation count per question
overlap / duplicate cluster id, если доступен
```

При обучении следующего B1 нельзя позволять одному вопросу с большим числом размеченных TM
доминировать над остальными.

Следовательно, supervision должен оставаться:

```text
question-balanced / document–aspect balanced
```

а overlapping duplicate chunks одного evidence не должны создавать искусственно много независимых отрицательных наблюдений.

---

# 21. Критерий достаточности данных для запуска V3.1

Недостаточно просто «разметить 400 строк».

V3.1 разрешено запускать только если выполнено всё:

- [ ] taxonomy зафиксирована;
- [ ] 400 текущих примеров обработаны или явно задокументирована причина неполной разметки;
- [ ] все `low confidence` прошли adjudication либо переведены в `unresolved`;
- [ ] `source_label` нигде не переписан;
- [ ] INTERNAL_TEST отсутствует в overlay;
- [ ] `verified_target_mismatch > 0`;
- [ ] `verified_target_evidence` посчитан;
- [ ] для каждого outer CV-run посчитаны `n_train_TM_chunks` и `n_train_questions_with_TM`;
- [ ] до обучения зафиксирован единый `min_train_questions_with_TM` для признания B1 подтверждающим, а не только exploratory;
- [ ] каждый outer TRAIN удовлетворяет этому минимуму либо эксперимент явно маркируется как недостаточно обеспеченный TM-supervision;
- [ ] для основной pairwise TM-метрики посчитано число document–aspect случаев с обеими сторонами: `P_v3_target` и `TM_verified`;
- [ ] до model evaluation зафиксирован `min_eval_questions_with_target_and_TM` для интерпретации pairwise TM как устойчивой метрики;
- [ ] построены distributions по aspect/subtype/stratum;
- [ ] coverage посчитан;
- [ ] annotation snapshot заморожен;
- [ ] сохранён hash snapshot;
- [ ] annotation guideline/version записаны;
- [ ] проверка схемы и дубликатов пройдена.

Если в каком-либо CV-run training недостаточно уникальных document–aspect с TM по заранее зафиксированному минимуму:

> выборку нужно расширить **до запуска V3.1**, а не менять folds и не компенсировать проблему многократным oversampling нескольких TM.

---

# 22. Когда расширять выборку сверх 400

Расширить representative/hard-case sampling, если после первой волны:

```text
verified TM слишком мало для всех CV-runs;
какой-либо аспект практически не представлен;
один subtype доминирует почти полностью;
TargetStatusCoverage@1 остаётся слишком низким;
pairwise P_v3_target > TM нельзя посчитать на значимой части вопросов;
```

Расширение должно выполняться тем же frozen sampling protocol.

Не выбирать новые примеры после просмотра результатов B1 на EVAL folds.

---

# 23. Freeze данных

После QC создать immutable snapshot, например:

```text
annotation/target_status_v3_1_frozen.jsonl
annotation/annotation_summary_v3_1.json
annotation/annotation_manifest_v3_1.json
```

Manifest должен содержать:

```text
taxonomy_version
annotation_guideline_version
n_annotated
n_target_evidence
n_target_mismatch
n_partial_support
n_unresolved
n_by_aspect
n_by_subtype
n_by_sampling_stratum
n_train_questions_with_TM_by_outer_run
n_eval_questions_with_target_and_TM_by_outer_run
coverage
sampling frame / sampling probabilities
evidence localization coverage
annotator information / anonymized ids
adjudication information
source dataset hash
CV split hash
overlay file hash
freeze timestamp
```

После freeze:

```text
не исправлять labels незаметно;
не добавлять примеры в тот же эксперимент;
не менять taxonomy.
```

Любая следующая правка создаёт новую версию overlay.

После freeze запрещено:

```text
просмотреть ошибки B1 на EVAL;
добавить именно эти EVAL-чанки в overlay;
переобучить B1;
снова назвать результат на том же EVAL подтверждающим.
```

Если EVAL error analysis выявляет новый класс ошибок,
он переносится в следующую версию протокола/overlay.

---

# 24. Какие данные получит следующий эксперимент

После freeze V3.1 получает:

### Основные positives

```text
P_original
P_v3_target
```

### Основные negatives

```text
training_eligible_hn
TM_verified
limited easy_negative
```

### Неиспользуемые как strong labels

```text
partial_support
unresolved
low-confidence unresolved
```

### Дополнительные аналитические данные

```text
mismatch_type
gold_relation
annotation_reason
confidence
sampling stratum
```

---

# 25. Что категорически запрещено

1. Переписывать исходный SciREX/Phase 1 `source_label`.
2. Считать весь `ambiguous` класс `target_mismatch`.
3. Считать весь GSRN автоматически target mismatch.
4. Превращать `partial_support` в strong negative.
5. Превращать `unresolved` в negative.
6. Добавлять `new_valid_value_candidate` в primary gold без отдельного audit.
7. Показывать model score аннотатору как часть решения.
8. Использовать INTERNAL_TEST.
9. Менять CV folds после появления overlay.
10. Расширять sampling после просмотра EVAL-выигрыша конкретной модели и считать тот же EVAL подтверждающим.
11. Смешивать hard-case и representative без раздельного отчёта.
12. Считать отсутствие annotation эквивалентом `not target_mismatch`.

---

# 26. Итог этапа подготовки данных

Результатом должен быть **не просто файл с ручными метками**, а проверенный и замороженный dataset overlay, который позволяет впервые честно проверить:

\[
P(\text{target evidence} \mid q,c,D)
\]

и сравнить:

```text
target evidence
vs
target mismatch
```

при сохранении исходной разметки, document-level CV и полностью закрытого INTERNAL_TEST.

Только после выполнения критериев готовности из §21 разрешено переходить к протоколу продолжения экспериментов V3.1.
