# Протокол автоматизированной подготовки данных V3.1 на основе текущей frozen QA-модели

## 0. Статус документа

Этот документ **заменяет предыдущую версию подготовки данных с отдельной teacher-моделью**.

В новой схеме:

```text
НЕ используется генеративная LLM;
НЕ требуется ручная разметка;
НЕ вводится отдельная тяжёлая модель-аннотатор.
```

Используется **та же frozen extractive QA-модель**, которая уже входит в текущий pipeline и применялась при анализе positive / hard-negative случаев.

Основная идея подготовки данных:

> использовать принцип, аналогичный hard-negative mining:
>
> сначала автоматически найти среди `ambiguous` те чанки, которые текущая QA-модель считает правдоподобными источниками ответа, а затем среди этих «опасных» кандидатов определить, относится ли извлечённый ответ к текущей статье или к другому объекту.

Высокоуровневая схема:

```text
all ambiguous chunks
↓
canonical QA pass
↓
QA-plausible ambiguous candidates
↓
target-attribution probes той же QA-моделью
+
детерминированные contextual/discourse признаки
↓
consensus / contradiction checks
↓
silver_target_evidence
silver_target_mismatch
silver_partial_support
silver_unresolved
↓
freeze
↓
V3.1
```

Автоматические метки называются **silver**, а не `verified` и не `gold`.

---

# 1. Почему используется именно текущая QA-модель

Одна из целей исследования — снижение вычислительных затрат относительно генеративного подхода.

Поэтому подготовка данных не должна скрыто требовать:

```text
большой генеративной LLM
→ разметка
→ обучение компактного encoder-подхода.
```

Вместо этого используется уже имеющаяся frozen QA-модель.

Преимущества:

```text
не требуется новая тяжёлая модель;
не требуется дополнительное обучение на этапе разметки;
используется уже существующий inference pipeline;
можно батчировать вычисления;
можно переиспользовать QA-cache;
все признаки остаются связанными с extractive QA постановкой.
```

Главное ограничение:

> QA-модель нельзя считать независимым истинным судьёй `target_evidence / target_mismatch`.

Поэтому её выход используется для формирования **silver weak supervision**, а не нового gold standard.

---

# 2. Связь с текущим принципом поиска HN

Логика подготовки максимально приближена к hard-negative mining.

Для HN концептуально используется:

```text
не-positive chunks
↓
найти наиболее правдоподобные для вопроса
↓
отделить трудные negatives
```

Для нового target-attribution этапа:

```text
ambiguous chunks
↓
найти те, из которых QA уверенно извлекает правдоподобный аспектный ответ
↓
определить принадлежность найденного ответа
↓
current paper      → silver_target_evidence
other object/work  → silver_target_mismatch
неясно             → silver_unresolved
```

Ключевое различие:

```text
HN:
похож на ответ, но evidence недостаточен / отрицателен

Target mismatch:
answerable-looking chunk содержит правдоподобный аспектный ответ,
но ответ относится не к тому объекту.
```

---

# 3. Primary universe

Использовать тот же unlocked primary universe, что в V3:

```text
453 answerable document–aspect
240 документов
```

Источники:

```text
V2 train
+
V2 calibration
+
V2 development
```

Не использовать:

```text
INTERNAL_TEST
seen_exploratory_dev
внешние документы
web search
сторонние сведения о статье
```

Primary candidate universe для автоматической подготовки:

```text
все source_label = ambiguous
```

во всех 453 answerable document–aspect.

Ограничение старой ручной очередью из 400 чанков больше не требуется.

Очередь 400 сохранить только как:

```text
historical hard-case sample
+
representative diagnostic sample.
```

---

# 4. Исходные labels нельзя переписывать

Не изменять:

```text
source_label
source_ambiguous_subtype
primary_class
original gold
```

Новый слой хранить отдельно:

```text
source_label = ambiguous
source_ambiguous_subtype = ...

silver_target_status = ...
```

Это позволяет сохранять одновременно:

```text
старую V2/V3 taxonomy;
новую target-attribution taxonomy;
анализ расхождений между ними.
```

---

# 5. Итоговые silver-классы

Основная taxonomy:

```text
silver_target_evidence
silver_target_mismatch
silver_partial_support
silver_unresolved
```

## 5.1. `silver_target_evidence`

Чанк автоматически признан достаточным evidence аспекта **для текущей статьи**.

## 5.2. `silver_target_mismatch`

Чанк выглядит answerable и содержит правдоподобное аспектное значение, но автоматические attribution-сигналы указывают, что значение относится:

```text
к внешней работе;
baseline;
сравниваемому методу;
другому датасету;
другому эксперименту;
другому объекту.
```

## 5.3. `silver_partial_support`

Evidence связано с текущей статьёй, но само по себе недостаточно для уверенного ответа.

## 5.4. `silver_unresolved`

Сигналы недостаточны или противоречат друг другу.

Правило:

> при сомнении — `silver_unresolved`.

---

# 6. Двухступенчатая схема ради вычислительной экономии

Не запускать несколько QA-вопросов по каждому ambiguous chunk сразу.

Использовать два этапа.

## Stage A — candidate mining

Один canonical QA pass:

```text
1 вопрос
×
все ambiguous chunks
```

Цель:

> найти только те ambiguous chunks, которые действительно опасны для текущего QA pipeline.

## Stage B — attribution probing

Дополнительные QA-проходы запускать только для Stage-A candidates.

Обычно:

```text
2–4 дополнительных вопроса
×
только QA-plausible ambiguous candidates
```

Таким образом дорогая часть не масштабируется на весь document corpus.

---

# 7. Stage A — canonical QA mining

Для каждого `(document_id, aspect, ambiguous_chunk)` использовать тот же canonical aspect question.

Например:

```text
method:
What method or model is used or proposed in the paper?

task:
What task is addressed in the paper?

material:
What dataset or material is used in the paper?

metric:
What evaluation metric is used in the paper?
```

Frozen QA возвращает минимум:

```text
predicted_span
start_index
end_index
start_logit
end_logit
raw_span_score
```

Если текущая реализация уже вычисляет дополнительные reader outputs, их можно сохранить, но нельзя менять веса reader-а.

---

# 8. Нельзя использовать raw QA score как абсолютную probability

`start_logit + end_logit`:

```text
не является P(correct);
не является глобально откалиброванной probability между разными passages.
```

Поэтому Stage-A threshold должен строиться **относительно распределений на уже известных source classes**, а не задаваться как произвольное число.

---

# 9. Калибровка QA-plausibility

Для каждого outer CV-run threshold определяется без EVAL leakage.

Использовать TRAIN/CALIBRATION:

### Reference positive distribution

```text
original robust_positive
```

### Reference negative controls

```text
training_eligible_hn
easy_negative
```

Из QA outputs построить признаки, например:

```text
raw_span_score
top1–top2 span margin
answer length
answer stability, если доступна дешёвая повторная оценка
```

Основной mining score должен быть заранее определён.

В primary варианте **не обучать отдельный калибратор**.

Использовать только детерминированное преобразование уже имеющегося QA-score, например:

```text
qa_plausibility_score =
empirical percentile / robust z-score raw_span_score
относительно TRAIN distribution
```

Допустимые TRAIN-derived параметры:

```text
median
IQR
mean/std
empirical quantiles / empirical CDF
```

Threshold выбирается только на CALIBRATION по заранее заданному критерию.

Если позднее захочется проверить trainable calibration, это должна быть отдельная ablation следующего этапа, а не часть primary подготовки данных.

---

# 10. Требование к Stage-A threshold

Stage A должен быть **recall-oriented**.

Цель:

> не пропустить потенциальный target mismatch.

Поэтому threshold выбирается не для максимальной общей accuracy, а для высокой полноты известных positives / трудных answerable-like cases.

До EVAL зафиксировать критерий, например:

```text
минимальный threshold,
при котором recall original robust_positive на CAL >= R_min
```

где `R_min` задаётся заранее.

Дополнительно ограничить:

```text
максимальную долю easy_negative,
проходящую threshold,
```

чтобы shortlist не стал практически всем ambiguous pool.

Конкретные `R_min` и допустимую negative pass-rate зафиксировать до EVAL.

---

# 11. Stage-A output

Каждый ambiguous chunk получает:

```text
qa_predicted_span
qa_start
qa_end
qa_raw_span_score
qa_margin
qa_plausibility_score
qa_plausible = true / false
```

Только:

```text
qa_plausible = true
```

переходит в attribution Stage B.

`qa_plausible = false` не превращается автоматически в negative.

Его статус:

```text
silver_unresolved_noncandidate
```

или общий:

```text
silver_unresolved
```

в зависимости от выбранной схемы хранения.

---

# 12. Gold relation как offline diagnostic signal

Для unlocked TRAIN/CAL/EVAL разрешено автоматически сравнивать predicted span с существующим gold.

Сохранить:

```text
qa_exact_match_gold
qa_token_f1_gold
qa_char_overlap_gold
qa_normalized_match_gold
```

Но:

> совпадение строки с gold не доказывает target ownership.

Один и тот же термин может встречаться:

```text
в related work;
baseline;
сравниваемом методе;
другом эксперименте.
```

Поэтому gold relation — только один из сигналов.

---

# 13. Stage B — target-attribution probing той же QA-моделью

Для QA-plausible candidate запускаются дополнительные вопросы той же frozen QA.

Они делятся на два типа:

```text
target-focused paraphrases
contrast / external-attribution probes
```

---

# 14. Target-focused paraphrases

Для каждого аспекта заранее фиксируется небольшой набор близких вопросов.

Пример `method`:

```text
Q0:
What method or model is used or proposed in the paper?

Q1:
What method do the authors propose or use?

Q2:
What model is used in this work?

Q3:
What approach is introduced in the paper?
```

Пример `task`:

```text
What task is addressed in the paper?
What problem do the authors solve?
What task is studied in this work?
```

Аналогично определить наборы для:

```text
material
metric
```

Количество paraphrases фиксируется до запуска.

Все вопросы выполняются **той же самой frozen QA-моделью**; никакая другая языковая модель для их обработки не подключается.

Не подбирать вопросы после просмотра downstream EVAL.

---

# 15. Что измеряется по paraphrase probes

Для каждого question variant сохранять:

```text
predicted_span
span offsets
raw span score
normalized score
```

Затем считать:

```text
span_text_agreement
span_overlap_agreement
answer_value_agreement
score_stability
position_stability
```

Например:

```text
n_same_normalized_answer / n_questions
```

и:

```text
mean pairwise Token F1 между predicted spans
```

Высокая стабильность означает:

> QA устойчиво считает этот fragment источником одного и того же аспектного значения.

Но это всё ещё не определяет ownership.

---

# 16. Contrast / ownership probes

Для Stage-A candidate использовать небольшой заранее фиксированный набор вопросов на принадлежность.

Идея:

```text
target question:
что используют / предлагают авторы текущей статьи?

external question:
что используется / предлагается другой или предыдущей работой?
```

Пример для `method`:

```text
Q_target:
What method do the authors of this paper use or propose?

Q_external:
What method from previous, cited, baseline, or compared work is mentioned?
```

Для `material`:

```text
Q_target:
What dataset or material is used in this work?

Q_external:
What dataset or material is mentioned only for another work, baseline, or comparison?
```

Для `task` и `metric` формируются аналогичные пары.

Важно:

> эти вопросы являются QA-probes, а не новой генеративной задачей.

---

# 17. Ограничение ownership probes

Текущая QA-модель могла не обучаться специально на вопросах принадлежности.

Поэтому:

```text
ownership QA score
```

нельзя использовать как единственный label source.

Он работает только совместно с:

```text
canonical QA;
paraphrase stability;
deterministic discourse features;
gold relation;
local context checks.
```

Если ownership probe нестабилен:

```text
abstain → silver_unresolved.
```

---

# 18. Детерминированные contextual/discourse признаки

Для sentence/window вокруг predicted span автоматически вычислить признаки.

## Current-paper markers

Например:

```text
we
our
in this work
we propose
we introduce
we use
we evaluate
our method
our model
our experiments
```

## External / comparison markers

Например:

```text
et al.
previous work
prior work
baseline
compared with
following X
proposed by
reported by
according to
```

## Structural signals

```text
citation marker near span
section title
relative document position
sentence position
distance to citation marker
```

Эти признаки:

```text
не являются сами по себе финальными правилами;
не используются как единственный источник label.
```

---

# 19. Citation marker не равен target mismatch

Запрещено правило:

```text
есть citation → target_mismatch
```

Причины:

```text
авторы могут цитировать собственный компонент;
citation может стоять рядом с target evidence;
описание текущего метода может включать ссылку на основу.
```

Citation используется только как один из attribution signals.

---

# 20. Формирование компактного attribution signature

Для каждого Stage-B candidate сформировать:

```text
z_attr =
[
  canonical QA score,
  QA margin,
  gold relation features,
  paraphrase agreement,
  target-probe score,
  external-probe score,
  target-vs-external score margin,
  current-paper marker features,
  external marker features,
  citation-distance features,
  section/position features
]
```

Все numeric features, требующие scaling:

```text
fit scaler только TRAIN;
apply CAL/EVAL без refit.
```

---

# 21. Как получать silver label без новой trainable model

На этапе подготовки данных не обучать новый classifier на target-status.

Использовать:

```text
детерминированную consensus policy
```

над `z_attr`.

Причина:

> новая trainable target classifier уже является частью следующего эксперимента B1/R1 и не должна незаметно появляться внутри подготовки данных.

В primary data-preparation pipeline запрещены дополнительные модели:

```text
generative LLM
NLI model
semantic encoder
Cross-Encoder classifier
learned calibration model
```

Разрешены только:

```text
текущая frozen QA-модель
+
детерминированные операции над её outputs
+
детерминированные признаки текста/структуры.
```

---

# 22. `silver_target_evidence`

Strong silver label назначается только при согласии нескольких сигналов.

Минимальная логика:

```text
qa_plausible = true
AND
paraphrase consistency >= τ_consistency
AND
target ownership signal > external ownership signal + δ
AND
нет сильного external contradiction
AND
predicted answer имеет допустимую relation к existing gold
```

Допустимая gold relation для primary определяется **без дополнительной semantic-модели**:

```text
exact normalized match
или
Token F1 / Char F1 выше заранее заданного CAL threshold
```

Normalization должна быть детерминированной и фиксированной.

Не использовать отдельный semantic encoder / NLI / generative judge для определения gold relation.

Если QA извлекает потенциально новое корректное значение вне gold:

```text
new_value_candidate
→ не включать в primary silver_target_evidence.
```

---

# 23. `silver_target_mismatch`

Strong silver mismatch назначается, если:

```text
qa_plausible = true
AND
QA устойчиво извлекает аспектное значение
AND
external ownership signal > target ownership signal + δ
AND
есть минимум один независимый contextual attribution signal
AND
нет sufficient current-paper signal для того же value
```

Примеры независимого attribution signal:

```text
external probe указывает на тот же span/value;
span находится в явной baseline/previous-work конструкции;
владелец значения связан с cited work;
target-focused paraphrases не поддерживают current-paper ownership.
```

Gold string может:

```text
совпадать
или
не совпадать
```

с extracted value.

Совпадение с gold **не запрещает** mismatch, если ownership явно другой.

---

# 24. `silver_partial_support`

Назначать консервативно.

Например:

```text
qa_plausible = true
AND
есть current-paper attribution
AND
target probes согласованы частично
AND
answer/sufficiency нестабильны
AND
нет сильного external ownership.
```

Primary B1 training:

```text
не использовать partial_support как strong positive;
не использовать как strong negative.
```

---

# 25. `silver_unresolved`

Назначать всегда, когда:

```text
target/external probes конфликтуют;
paraphrases дают разные значения;
QA score около threshold;
gold relation неясна;
mixed context содержит current и external values;
ownership зависит от широкого контекста;
discourse signals противоречат друг другу.
```

Это основной механизм автоматического abstention.

---

# 26. Mixed-value chunks

Особый случай:

```text
один chunk содержит target value
+
baseline/external value.
```

Не считать его mismatch автоматически.

Если QA/probes показывают sufficient current-paper value:

```text
silver_target_evidence
contains_competing_values = true
```

Если модель извлекает то target, то external value:

```text
silver_unresolved
```

Если стабильно извлекается только external value:

```text
silver_target_mismatch
```

при выполнении остальных strong-label условий.

---

# 27. Confidence tiers

Каждому автоматически назначенному status сопоставить:

```text
high
medium
low
```

## High

```text
Stage-A score далеко от threshold;
paraphrases согласованы;
ownership margin устойчив;
deterministic signals не противоречат;
gold relation валидна там, где требуется.
```

## Medium

```text
основной consensus есть,
но margin или contextual support умеренные.
```

## Low

```text
есть конфликт или слабый signal.
```

Primary strong supervision:

```text
только high-confidence.
```

Medium:

```text
отдельная sensitivity ablation.
```

Low:

```text
silver_unresolved.
```

---

# 28. Thresholds нельзя подбирать по downstream EVAL

Все thresholds:

```text
τ_QA
τ_consistency
δ_ownership
confidence boundaries
```

настраиваются только:

```text
TRAIN
+
CALIBRATION
```

по уже существующим source controls и заранее заданным критериям.

Запрещено:

```text
посмотреть B1/R1 EVAL Token F1
→ изменить silver thresholds
→ пересоздать labels
→ снова посчитать тот же EVAL.
```

---

# 29. Автоматические control sets

Поскольку ручного annotator нет, использовать уже известные source classes как sanity controls.

## Positive controls

```text
original robust_positive
```

QA должна часто:

```text
проходить Stage-A;
давать стабильный aspect answer;
показывать target-oriented signal.
```

## HN controls

```text
training_eligible_hn
```

Они позволяют проверить, что новая target-attribution схема не разрушает уже решённое HN separation.

## Easy controls

```text
easy_negative
```

Большая часть не должна проходить Stage-A mining.

---

# 30. Основные QC-метрики подготовки

Считать минимум:

```text
StageA_PositiveRecall
StageA_HNPassRate
StageA_EasyPassRate

ParaphraseAgreement_Positive
ParaphraseAgreement_HN

TargetProbeMargin_Positive
TargetProbeMargin_HN

SilverTargetEvidenceRate
SilverTargetMismatchRate
SilverPartialRate
SilverUnresolvedRate

HighConfidenceRate
MediumConfidenceRate

InvalidSpanOffsetRate
```

Дополнительно по aspect:

```text
task
method
material
metric
```

---

# 31. Stage-A mining quality gate

До построения final silver overlay зафиксировать минимальный acceptable Stage-A behavior.

Например, до EVAL определить:

```text
min_positive_recall
max_easy_pass_rate
max_invalid_span_rate
```

Если gate не пройден:

```text
не продолжать Stage B;
не менять downstream models;
```

а исправить:

```text
threshold calibration
или
QA extraction implementation
```

на TRAIN/CAL.

---

# 32. Attribution QC без human labels

Истинного `target_mismatch` gold нет.

Поэтому attribution quality нельзя доказать напрямую.

Вместо этого проверять:

```text
consistency;
control behavior;
abstention;
downstream usefulness.
```

Обязательные diagnostics:

```text
target-probe agreement на robust_positive;
external-probe false activation на robust_positive;
ownership conflict rate;
paraphrase consistency;
silver label stability при небольших threshold perturbations.
```

---

# 33. Stability analysis silver labels

До freeze проверить чувствительность к небольшим изменениям thresholds на CAL.

Например:

```text
τ ± small_delta
δ ± small_delta
```

Считать:

```text
label flip rate
high-confidence flip rate
```

Strong silver data должны иметь низкий flip rate.

Если high-confidence labels нестабильны:

```text
confidence rule слишком агрессивно.
```

---

# 34. Deduplication overlapping chunks

Перед использованием silver labels в обучении:

```text
определить overlap / source interval clusters.
```

Сохранить:

```text
evidence_cluster_id
```

Несколько overlapping chunks одного и того же source region не должны:

```text
многократно увеличивать вес одного TM;
создавать искусственно много независимых positives.
```

Training остаётся:

```text
question-balanced
+
cluster-aware.
```

---

# 35. Минимальная структура output

Для каждого ambiguous candidate:

```text
document_id
question_id
aspect
question
chunk_id

source_label
source_ambiguous_subtype

qa_predicted_span
qa_span_start
qa_span_end
qa_raw_span_score
qa_plausibility_score
qa_plausible

qa_gold_relation
qa_paraphrase_agreement

qa_target_probe_answer
qa_target_probe_score
qa_external_probe_answer
qa_external_probe_score
qa_ownership_margin

current_paper_marker_score
external_marker_score
citation_distance
section_type
relative_document_position

silver_target_status
silver_confidence
contains_competing_values
evidence_cluster_id

silver_rule_version
qa_model_hash
question_set_version
threshold_config_hash
```

---

# 36. Рекомендуемая файловая структура

```text
annotation_qa_silver/
    qa_model_manifest.json
    aspect_questions.json
    attribution_questions.json
    thresholds.json
    rules.json

    stage_a_all_ambiguous.jsonl
    stage_a_candidates.jsonl

    stage_b_paraphrases.jsonl
    stage_b_ownership.jsonl
    discourse_features.jsonl

    silver_target_status_all.jsonl
    silver_target_status_high.jsonl
    silver_target_status_medium.jsonl

    qc_controls.jsonl
    qc_summary.json
    silver_summary.json
    silver_manifest.json
```

---

# 37. Воспроизводимость

Зафиксировать:

```text
QA model path / checkpoint hash
tokenizer hash/version
canonical questions
paraphrase question sets
ownership question sets
normalization rules
Stage-A calibration rule
all thresholds
discourse feature rules
silver mapping rules
code commit hash
CV split hash
source dataset hash
```

Frozen QA:

```text
eval mode
dropout off
weights unchanged
```

---

# 38. Вычислительная стоимость

Отдельно посчитать offline preparation cost:

```text
n Stage-A QA calls
n Stage-B QA calls
total reader tokens
GPU time
peak GPU memory
cache hit rate
```

Главный efficiency показатель:

```text
Stage-B calls / all ambiguous chunks
```

Желательно, чтобы attribution probing запускался только на небольшой доле ambiguous.

Это и есть преимущество HN-like mining:

```text
сначала дешёвый фильтр
→ потом подробный анализ только hard candidates.
```

---

# 39. Не использовать attention в primary silver preparation

Текущий V3 не показал сильного общего attention signal.

Поэтому primary automated labeling строить без attention.

Разрешено сохранить attention как:

```text
отдельную будущую ablation
```

но не включать его в обязательную Stage-A/Stage-B разметку.

Это:

```text
снижает вычислительную стоимость;
уменьшает circularity;
сохраняет простоту базовой схемы.
```

---

# 40. Gold и leakage

Gold разрешено использовать:

```text
для offline TRAIN/CAL silver relation;
для диагностической классификации predicted spans;
для downstream QA evaluation.
```

Но нельзя:

```text
использовать EVAL gold для настройки thresholds;
использовать INTERNAL_TEST gold до final evaluation.
```

Silver overlay EVAL создаётся frozen procedure без адаптации по EVAL.

---

# 41. CV policy

Для каждого outer run:

```text
TRAIN:
fit только детерминированные QA-score normalization statistics,
fit feature scalers, если они вообще нужны для отчётных numeric features,
derive any train-derived statistics

CAL:
select τ_QA,
τ_consistency,
δ_ownership,
confidence thresholds

EVAL:
apply frozen Stage-A + Stage-B + silver mapping
```

Важное следствие:

> если thresholds зависят от fold calibration, silver labels являются fold-specific runtime artifacts для V3.1 OOF.

Поэтому сохранять:

```text
outer_run_id
threshold_config_id
```

для каждого silver prediction.

---

# 42. Global precomputed features vs fold-specific labels

Можно один раз глобально вычислить **сырой frozen QA output** для всех unlocked chunks:

```text
canonical QA outputs
paraphrase QA outputs
ownership probe outputs
deterministic text features
```

потому что веса QA frozen и эти outputs не зависят от train labels.

Но нельзя глобально fit:

```text
score normalization
thresholds
feature scaling
confidence mapping
```

по всем unlocked данным.

Они вычисляются fold-честно:

```text
TRAIN/CAL only.
```

Это позволяет одновременно:

```text
сократить compute через cache
и
избежать leakage.
```

---

# 43. Критерий готовности silver data к V3.1

V3.1 разрешено запускать, если:

- [ ] frozen QA regression check пройден;
- [ ] все ambiguous получили Stage-A output;
- [ ] Stage-A QC gate пройден;
- [ ] Stage-B выполнен для всех QA-plausible candidates;
- [ ] invalid offsets обработаны;
- [ ] thresholds получены только TRAIN/CAL;
- [ ] high-confidence silver labels сформированы;
- [ ] unresolved сохранены, а не принудительно размечены;
- [ ] evidence-cluster dedup выполнен;
- [ ] per-aspect distributions построены;
- [ ] per-fold `silver_target_mismatch` counts посчитаны;
- [ ] per-fold `silver_target_evidence` counts посчитаны;
- [ ] label stability audit выполнен;
- [ ] QA model/questions/rules/hashes зафиксированы;
- [ ] INTERNAL_TEST не открыт.

---

# 44. Если silver target mismatch слишком мало

Не снижать thresholds автоматически до получения желаемого количества.

Сначала проверить:

```text
Stage-A recall;
ownership probe behavior;
distribution по aspect;
сколько ambiguous вообще является QA-plausible.
```

Если strong mismatch действительно редок:

```text
это самостоятельный результат.
```

Только отдельной заранее объявленной sensitivity-version разрешено добавить:

```text
medium-confidence silver labels.
```

---

# 45. Если silver target evidence слишком много

Проверить:

```text
не превращается ли lexical gold match автоматически в target ownership;
не слишком ли сильны current-paper markers;
не игнорируются ли citations/baseline contexts;
не попадают ли mixed chunks в target_evidence без проверки.
```

Высокая доля `silver_target_evidence` среди GSRN может означать:

```text
реальные false negatives исходной разметки
или
слишком мягкую automatic attribution policy.
```

Эти варианты нельзя смешивать без audit.

---

# 46. Новые valid-value candidates

Если QA устойчиво извлекает current-paper value, которого нет в original gold:

```text
new_valid_value_candidate = true
```

Но:

```text
не добавлять автоматически в original gold;
не включать в primary P_silver_target;
не использовать для primary QA metric.
```

Сохранить отдельно:

```text
new_value_candidates.jsonl
```

для будущего анализа качества исходной разметки.

---

# 47. INTERNAL_TEST

До final V3.1 freeze:

```text
INTERNAL_TEST не обрабатывать silver-labeling pipeline.
```

После final freeze:

```text
primary INTERNAL_TEST evaluation
→ original QA gold.
```

Silver test diagnostics допустимы только если заранее записаны в final manifest и:

```text
не используются для model selection;
не приводят к изменению thresholds/rules/model.
```

---

# 48. Что запрещено

1. Использовать отдельную генеративную teacher-модель.
2. Использовать web/external knowledge.
3. Переписывать source labels.
4. Считать весь ambiguous target mismatch.
5. Считать QA answerability достаточным доказательством target ownership.
6. Считать gold string match доказательством target evidence.
7. Считать citation marker доказательством mismatch.
8. Использовать один ownership QA probe как единственный label source.
9. Использовать EVAL для настройки silver thresholds.
10. Принудительно размечать конфликтные случаи вместо abstention.
11. Многократно oversample overlapping TM chunks.
12. Изменять QA weights во время подготовки.
13. Использовать INTERNAL_TEST до final freeze.

---

# 49. Итоговая схема подготовки

```text
AMBIGUOUS CHUNKS
        ↓
STAGE A
canonical frozen QA
        ↓
QA-plausibility normalization / thresholding
        ↓
hard ambiguous candidates
        ↓
STAGE B
same frozen QA:
    target paraphrases
    ownership / contrast probes
        +
deterministic discourse features
        +
gold relation diagnostics
        ↓
CONSENSUS / ABSTENTION
        ↓
high-confidence:
    silver_target_evidence
    silver_target_mismatch

non-strong:
    silver_partial_support
    silver_unresolved
        ↓
cluster-aware dedup
        ↓
freeze
        ↓
V3.1 B1 / target reranking
```

---

# 50. Научная интерпретация

Корректная формулировка метода:

> дополнительные target-attribution данные формируются автоматически той же frozen extractive QA-моделью, которая используется в основном подходе. Сначала по принципу hard-negative mining выделяются ambiguous-чанки, из которых модель уверенно извлекает правдоподобный аспектный ответ; затем для них с помощью нескольких QA-проб и детерминированных контекстных признаков оценивается принадлежность извлечённого значения текущей статье. Неоднозначные случаи исключаются из сильной разметки посредством abstention.

Это позволяет:

```text
не использовать генеративную LLM;
не вводить ручную разметку;
сохранить вычислительную идеологию encoder-based подхода;
автоматически получить weak-supervision target-attribution labels.
```

Primary доказательство полезности такой подготовки:

```text
улучшение downstream QA
(Token F1 / EM по original gold)
```

а silver target-attribution metrics используются как диагностические.
