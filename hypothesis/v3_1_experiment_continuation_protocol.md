# Протокол продолжения экспериментов V3.1 после подготовки target-status overlay

## 0. Назначение документа

Этот документ описывает **продолжение экспериментов после завершения и freeze дополнительной target-status разметки**.

Он не заменяет текущий V3 и не начинает исследование заново.

Текущий V3 уже установил:

```text
B0 — сильный воспроизводимый baseline;
HN больше не является главным top-1 distractor;
главная проблема — ambiguous / GSRN;
правильный original positive уже находится в top-5 для ~79% вопросов
и в top-10 для ~92%;
reader oracle существенно выше живого pipeline;
raw attention и текущий SpanRanker B0 не улучшили;
D4/D5 gates открыты.
```

Главная цель V3.1:

> впервые полноценно проверить target-attribution hypothesis на verified target-mismatch данных, а затем — только если этот сигнал подтверждается — исследовать узкий reranking shortlist, специализированные attention-признаки, D4 и D5.

---

# 1. Текущее состояние, которое считается baseline

Текущий OOF V3:

```text
240 документов
453 answerable-вопроса
5 document-level folds
3 seeds для текстовых ranker
```

B0 pooled:

```text
Token F1       = 27.36
EM             = 20.75%
Hit@1          = 36.72%
Hit@5          = 79.03%
Hit@10         = 92.20%
HN@1           = 4.86%
pairwise_clean = 75.14%
Ambiguous@1    = 50.85%
GSRN@1         = 33.92%
F1 | positive  = 61.70
```

Reader oracle:

```text
best-positive Token F1 = 73.01
question with ≥1 positive F1≥50 = 76.38%
```

Вывод:

> главный резерв — ranking / target attribution, а не поиск любого positive и не замена frozen reader.

---

# 2. Что текущий V3 НЕ проверил

Из-за пустого verified overlay не были полноценно проверены:

```text
H3.1 target-mismatch supervision
H3.2 target-vs-mismatch pairwise/listwise ranking
TargetMismatch@K
Pairwise P_v3_target > TM
attention: successful target vs real TM
target-aware span correctness
```

B1/B2/B3 текущего V3 нельзя трактовать как отрицательный результат против target-aware подхода.

---

# 3. Preconditions

V3.1 запрещено запускать до выполнения отдельного протокола подготовки данных.

Обязательные входные артефакты:

```text
frozen target-status overlay
annotation manifest
annotation summary
CV split file
CV split hash
overlay hash
taxonomy version
```

Минимальные условия:

```text
verified_target_mismatch > 0;
каждый outer TRAIN удовлетворяет frozen `min_train_questions_with_TM`;
есть достаточное число `P_v3_target vs TM` document–aspect cases по frozen `min_eval_questions_with_target_and_TM`;
source labels не переписаны;
overlay hash и sampling manifest зафиксированы;
INTERNAL_TEST закрыт.
```

---

# 4. Схема оценки остаётся той же

Не создавать новые outer folds.

Использовать замороженные:

```text
5 document-level folds
```

с теми же ролями:

```text
3 folds TRAIN
1 fold CALIBRATION
1 fold EVAL
```

Для всех новых систем:

```text
одинаковые EVAL_FOLD;
одинаковый document universe;
одинаковый reader;
одинаковые primary QA metrics.
```

INTERNAL_TEST:

```text
не открывать.
```

---

# 5. Этап V3.1-A — audit overlay до обучения

До запуска ranker построить:

```text
TM count by fold
TM count by aspect
TM count by source subtype
TM count by mismatch_type
target_evidence overlay count
partial/unresolved count
TargetStatusCoverage@1, SlotCoverage@5/@10 и CompleteCoverage@5/@10 для B0
```

Отдельно вычислить:

```text
B0 top-1:
positive / verified TM / unresolved ambiguous / other
```

и:

```text
B0 top-5:
сколько вопросов содержат:
- P_original
- P_v3_target
- TM_verified
- и positive, и TM одновременно
```

Это определит фактическую сложность target-attribution задачи.

---

# 6. Этап V3.1-B0 — saturation check и фиксация нового reference

В текущем V3 у B0:

```text
15 checkpoints
mean best epoch = 4.27
epoch 5 выбран 8 раз
max_epochs = 5
```

Это указывает, что лимит `5` мог обрезать CAL-selected training trajectory.

Чтобы не выбирать baseline постфактум по тем же EVAL-folds, V3.1 заранее определяет новый reference recipe:

```text
тот же B0 recipe
те же outer folds
те же seeds 42/43/44
тот же loss
тот же negative composition
тот же learning rate
тот же batch size
тот же CAL criterion
max_epochs = 8
patience = 2
```

В каждом outer run конкретная epoch выбирается **только по CALIBRATION**, как и раньше.

Старый `B0-max5` остаётся историческим diagnostic baseline. Нельзя после просмотра OOF EVAL выбрать между `max5` и `max8` тот, который дал больший EVAL Token F1.

Если при `max_epochs=8` CAL best epoch снова часто достигает 8, это фиксируется как диагностическое наблюдение; дальнейшее увеличение до 10 относится уже к следующей заранее объявленной версии training protocol и не делается адаптивно после просмотра EVAL.

## 6.1. Reference для всех V3.1 сравнений

Все B1/R1/B2/B3 trainable text models получают **тот же training budget policy**, что и reference, если архитектура не требует другого режима по заранее указанной причине.

Основной reference V3.1:

```text
B0-max8, CAL-selected epoch, seeds 42/43/44
```

Старый B0-max5 показывать только для continuity с V3.

---

# 7. Этап V3.1-B1 — первый основной эксперимент

Это **главный обязательный эксперимент после overlay**.

Меняется только supervision.

## 7.1. Reference

```text
B0:
P_original
vs
training_eligible HN + limited easy
```

## 7.2. B1-TM

Positive:

```text
P_v3_target
```

Negative:

```text
training_eligible HN
+
TM_verified
+
limited easy
```

Не использовать:

```text
partial_support
unresolved
```

как strong negatives.

Supervision должен оставаться question-balanced.

Если у одного document–aspect много verified TM:

```text
не дублировать его вес пропорционально числу overlapping chunks;
не делать неконтролируемый oversampling одних и тех же TM;
по возможности учитывать overlap/duplicate clusters.
```

## 7.3. Архитектура

Та же, что у V3.1 reference B0:

```text
MiniLM cross-encoder
question-balanced weighted BCE
max_epochs = 8
patience = 2
CAL-only early stopping
```

Не менять одновременно model family.

## 7.4. Seeds

Минимум:

```text
42
43
44
```

для каждого CV-run.

---

# 8. Метрики B1-TM

Обязательные:

### QA

```text
Token F1
EM
Char F1
ROUGE-L
```

### Ranking

```text
Hit@1_original
Hit@5_original
Hit@10_original
TargetEvidenceHit@1_v3
TargetEvidenceHit@5_v3
TargetEvidenceHit@10_v3
MRR
```

### HN safeguards

```text
HN@1
pairwise_clean_hn
```

### Target attribution

```text
TargetStatusCoverage@1
TargetStatusSlotCoverage@5/@10
TargetStatusCompleteCoverage@5/@10
TargetMismatch@1_lower_bound
TargetMismatchRate@1_adjudicated
Pairwise P_v3_target > TM_verified
GSRN@1_original
Ambiguous@1_original
```

### Source attribution

```text
WrongTargetAnswerRate_lower_bound
WrongTargetAnswerRate_known_role
WrongTargetAnswerCoverage
GoldMatchFromNonOriginalPositive
```

---

# 9. Model-selection safeguards для B1

Сохранить существующие hard safeguards:

```text
HN@1_candidate <= HN@1_B0 + 0.02
pairwise_clean_candidate >= pairwise_clean_B0 - 0.03
```

TargetMismatch использовать как hard safeguard только если:

```text
TargetStatusCoverage@1 >= 0.95
```

Если coverage ниже:

```text
TM metrics = diagnostic
```

Primary criterion среди допустимых систем:

```text
document–aspect Token F1
```

Но для решения о продолжении target-attribution ветки смотреть также:

```text
Pairwise P_v3_target > TM
TM@1
TargetEvidenceHit@1_v3
```

---

# 10. Paired statistical comparison B1 vs B0

Обязательно использовать:

```text
clustered paired bootstrap by document_id
```

Рекомендуется:

```text
5000 replications
```

Считать paired deltas:

```text
ΔTokenF1
ΔEM
ΔHit@1
ΔHN@1
ΔTargetEvidenceHit@1_v3
ΔTargetMismatch
ΔPairwiseTM
```

Не сравнивать:

```text
single-seed candidate
vs
three-seed pooled B0
```

Сравнение должно быть на одинаковом наборе predictions.

---

# 11. Gate G1 — как интерпретировать B1-TM

B1-TM является первым тестом новой supervision, но **не единственным возможным тестом target-attribution hypothesis**.

Если решение о запуске R1 принимается после просмотра **OOF EVAL** B1, дальнейшую ветку маркировать как следующую заранее замороженную итерацию (`V3.1b` или новый protocol version). Её результаты на тех же outer folds остаются model-selection evidence, а не независимым подтверждением.

Если требуется сохранить более чистую последовательность внутри одной версии, gate принимать по заранее определённым TRAIN/CALIBRATION критериям и overlay audit, а EVAL использовать только для оценки уже выбранных вариантов.

### G1-A: B1 показывает target signal

Продолжать к R1, если:

1. B1 проходит HN safeguards;
2. verified TM separation лучше B0 хотя бы по заранее объявленной TM-метрике;
3. Token F1 не имеет устойчивого практически значимого ухудшения;
4. эффект не держится только на одном outer fold.

### G1-B: B1 не выигрывает, но target-attribution задача реально присутствует

Даже если full-document B1 не улучшил QA/TM separation, разрешено один раз проверить специализированный R1-text внутри B0 shortlist, если overlay audit показывает:

```text
достаточное число B0 top-K случаев, где одновременно есть P_v3_target и TM_verified;
TM не является редким единичным классом;
разметка прошла QC.
```

Причина: B1 решает full-document answerability + attribution одной функцией, тогда как R1 решает более узкую задачу перестановки уже релевантного shortlist.

Перед R1 в этом случае обязателен label/mismatch-type audit на TRAIN/CALIBRATION, но не изобретение features по EVAL.

### G1-STOP

Остановить target-attribution ветку в текущей версии, если:

```text
TM слишком редок / плохо покрыт;
нет достаточного числа target-vs-TM конкурирующих shortlist cases;
качество разметки недостаточно;
```

а не просто потому, что B1 не выиграл общий Token F1.

---

# 12. Этап V3.1-R1 — target attribution reranker внутри B0 shortlist

Текущий B0 уже имеет:

```text
Hit@5 ≈ 79%
Hit@10 ≈ 92%
```

Поэтому следующая архитектура должна **не заменять B0**, а специализироваться на перестановке его shortlist.

Pipeline:

```text
all chunks
↓
B0
↓
top-K
↓
Target Attribution Reranker
↓
top-1 target evidence
↓
frozen reader
```

Проверить:

```text
K = 5
K = 10
```

K подбирать только на CALIBRATION.

---

# 13. Критически важно: cross-fitted shortlist для обучения второго этапа

Нельзя обучать second-stage reranker только на shortlist, полученном B0, который уже обучался на тех же TRAIN examples, и считать это честным распределением.

Для каждого outer CV-run:

### CALIBRATION / EVAL

Shortlist строит:

```text
B0 trained only on TRAIN_FOLDS
```

### TRAIN examples для Target Attribution Reranker

Использовать:

```text
inner grouped cross-fitting внутри TRAIN_FOLDS
```

чтобы каждый TRAIN document получил B0 score/shortlist от модели, которая его не видела при обучении.

Сохранить:

```text
shortlist_source_model_id
crossfit_fold_id
B0_score
B0_rank
```

Это защищает второй этап от train-distribution leakage.

---

# 14. Target Attribution Reranker: базовые варианты

Сначала проверять простые варианты.

## R1-a. Text only

Вход:

```text
[ASPECT]
...

[QUESTION]
...

[CHUNK]
...
```

## R1-b. Local discourse context

Добавить:

```text
section title
previous sentence
chunk
next sentence
```

Но не повторять прежнюю ошибку «просто длиннее input».

Контекст должен использоваться именно для attribution.

При `max_length=512` candidate chunk имеет приоритет. Truncation policy фиксируется заранее:

```text
1. сохранить candidate chunk и его ключевую область;
2. оставшийся token budget распределить на section / prev / next;
3. не допускать, чтобы добавление контекста систематически обрезало candidate chunk;
4. логировать долю и место truncation.
```

Иначе R1-b будет сравнивать не attribution context, а разные степени потери исходного чанка.

## R1-c. Explicit discourse markers

Можно добавить автоматически извлекаемые признаки:

```text
citation marker near value
we / our
propose / use / evaluate / train
baseline
previous work
compared with
et al.
reported by
section type
relative document position
```

Эти признаки:

```text
не являются ручной финальной rule-system;
используются как features/markers для supervised reranker.
```

Список feature families фиксируется до EVAL.

Новые discourse markers, обнаруженные после ручного просмотра ошибок EVAL,
не добавлять в тот же confirmatory OOF run;
для них требуется следующая версия протокола.

---

# 15. Обязательная ошибка, которую должен решать R1

Главный target:

```text
P_v3_target
vs
TM_verified
```

HN остаются важным safeguard, но R1 не должен заново решать всю full-document relevance задачу.

В shortlist training можно использовать:

```text
positive = P_v3_target
hard attribution negative = TM_verified
control negative = HN/easy, ограниченно
```

Кандидаты:

```text
partial_support
unresolved
unknown_ambiguous
```

не использовать как strong supervised negatives. Они могут присутствовать в runtime shortlist и должны сохраняться для coverage/error analysis.

---

# 16. Этап V3.1-B2 — pairwise только после G1

Pairwise objective возвращать только если B1 **или специализированный R1-text** показал воспроизводимый target-vs-TM signal.

Training pairs:

```text
P_v3_target > TM_verified
P_v3_target > training_eligible HN
```

Loss:

\[
L=-\log\sigma(s^+-s^-)
\]

Separate sampling:

```text
p_TM
p_HN
p_easy
```

фиксировать на CALIBRATION.

Не заставлять разные valid positives конкурировать друг с другом.

Overlapping positives / TM из одного source evidence cluster
не должны считаться большим числом независимых пар без контроля веса.

---

# 17. Этап V3.1-B3 — listwise внутри shortlist, а не всего документа

Текущий full-document listwise B3 не улучшил B0.

Следующая проверка допускается только после подтверждения TM-signal в B1 или R1.

Candidate set:

```text
B0 top-5
или
B0 top-10
```

Target:

```text
масса вероятности на всех P_v3_target внутри shortlist
```

Loss:

\[
L=
-\log
\frac{
\sum_{c\in P}\exp(s(q,c)/T)
}{
\sum_{c'\in C_{shortlist,labeled}}\exp(s(q,c')/T)
}
\]

`partial_support` и `unresolved`:

```text
mask from supervised denominator
```

T выбирать только CALIBRATION.

Если shortlist содержит несколько overlapping chunks одного evidence,
протокол должен либо:

```text
deduplicate clusters до loss
```

либо:

```text
нормировать их суммарный вес,
чтобы один source region не доминировал denominator.
```

---

# 18. Attention: новая роль после текущего отрицательного результата

Текущий attention-only:

```text
не является хорошим универсальным ranker;
HN@1 резко ухудшился;
mean attention AUROC vs HN ≈ chance.
```

Поэтому **не повторять**:

```text
attention as full-document ranker
0.5 * B0 + 0.5 * attention
```

Новая гипотеза:

> attention может быть специализированным признаком `target evidence vs target mismatch`, а не универсальным relevance score.

Важно: текущие attention features зависят от forward pass frozen QA reader и predicted span. Поэтому attention-ветка **не является дешёвым pre-reader фильтром**. Для shortlist-кандидатов нужно сначала запустить reader с `output_attentions=True`, после чего использовать сигнатуру для reranking; QA-ответ выбранного кандидата можно переиспользовать из этого же forward pass.

Стоимость должна входить в сравнение с text-only R1.

---

# 19. Этап V3.1-AT1 — честный attention audit после overlay

Исправить ошибки текущего AT1/AT3:

```text
head/feature analysis только TRAIN_FOLDS;
CALIBRATION для N_heads / thresholds;
EVAL не участвует в head selection.
```

Основная задача:

```text
P_v3_target
vs
TM_verified
```

Контрольные задачи:

```text
P_v3_target vs HN
reader-success vs reader-failure
```

Gate для дальнейшего развития attention принимать по TRAIN/CALIBRATION согласно заранее заданному критерию.

Если attention не разделяет TM на TRAIN/CALIBRATION:

```text
не развивать attention branch дальше.
```

EVAL не использовать для изобретения новой attention feature family в той же версии протокола.

---

# 20. Attention features

Использовать компактные inference-compatible признаки:

```text
normalized Q→predicted-span attention
normalized span→Q
normalized entropy
top-k attention concentration
competing-region margin
head agreement
layer dynamics
head dispersion
```

Обязательные controls:

```text
question length
context length
predicted span length
predicted span position
B0 rank
raw QA span score
```

Сравнения:

```text
CTRL
ATT
CTRL + ATT
```

---

# 21. Head selection

Head selection:

```text
TRAIN only
```

Желательно через:

```text
inner grouped resampling
```

Проверить:

```text
all-head mean
last-layer mean
middle-layer mean
selected heads
```

N_heads:

```text
{1,2,4,8,16}
```

если позволяет архитектура.

Выбирать N только на CALIBRATION.

Не предполагать заранее, что retrieval heads decoder LLM переносятся на RoBERTa.

---

# 22. Attention fusion

Если AT показывает signal на `P_v3_target vs TM`:

проверить:

```text
text-only target reranker
attention-only TM discriminator
text + attention
```

Fusion weight:

```text
НЕ фиксировать 0.5/0.5
```

Подбирать только CALIBRATION.

Лучше сначала:

```text
linear/logistic fusion
```

и только затем small nonlinear model.

---

# 23. Внутренние reader-signals: controls, distillation и causal validation

Только если attention показывает заранее определённый TRAIN/CALIBRATION signal на `P_v3_target vs TM_verified`; OOF EVAL затем используется только для оценки уже замороженной ветки.

## 23.1. Hidden-state control

Чтобы понять, специфичен ли эффект именно для attention, сравнить компактные reader representations:

```text
pooled question hidden state
pooled predicted-span hidden state
Q/span cosine
|Q-S|
Q*S
```

Сравнения:

```text
attention only
hidden-state only
attention + hidden-state
```

Все pooling/scaling параметры fit только TRAIN.

## 23.2. Attention × value norm — optional

Если реализация позволяет без изменения reader-а, проверить отдельную ablation:

```text
attention weight × value-vector norm
```

Не смешивать с raw attention без отдельной колонки результатов.

## 23.3. Gradient × attention — diagnostic only

Разрешено только как high-cost diagnostic branch:

```text
grad(predicted span score) × attention
```

Требует backward pass и не считается предпочтительным runtime feature. Если не превосходит forward-only сигнал, ветку закрыть.

## 23.4. Attention distillation — только после доказанного signal

Если direct attention полезен, но дорог, разрешить:

```text
reader attention-derived teacher score
→ lightweight question+chunk student
```

Teacher targets генерируются только для TRAIN/cross-fit данных. CAL выбирает distillation hyperparameters. EVAL/test не создают teacher supervision.

## 23.5. Causal head masking — diagnostic only

Если head selection даёт небольшой устойчивый набор heads по folds/seeds, сравнить:

```text
normal reader
mask selected heads
mask matched random heads
```

на TRAIN/CAL diagnostic set; при заранее замороженной процедуре можно оценить frozen ablation на EVAL как diagnostic result.

Causal masking не используется для выбора heads по EVAL и не является отдельным production pipeline.

---

# 24. Отдельный method-attention audit

Текущий exploratory результат показал необычное поведение attention на `method`:

```text
резко выше Hit@1/F1
одновременно резко выше HN@1
```

Провести paired transition analysis:

```text
B0 top1 → attention top1
```

Категории:

```text
GSRN → positive
other ambiguous → positive
positive → HN
GSRN → HN
positive → positive
```

Вручную проанализировать representative subset переходов
только на TRAIN/CALIBRATION материалах.

EVAL transitions можно посчитать как заранее определённую diagnostic metric,
но не использовать для изобретения feature в том же run.

Цель:

> понять, действительно ли attention помогает отличать method target от ambiguous или наблюдение является prototype artifact.

Не вводить aspect-specific production model без воспроизводимости по folds.

---

# 25. SpanRanker: текущую ветку временно остановить

Текущий K=3 SpanRanker:

```text
Token F1 ниже B0;
часто покидает правильный B0 top-1;
qa source positive выбирается реже.
```

Поэтому:

```text
не тюнить текущий SpanRanker;
не перебирать K=2/3/5 сейчас;
не добавлять attention в текущую span-модель.
```

Возвращаться к span-level ranking только после улучшения evidence shortlist.

---

# 26. Gate G2 для возврата SpanRanker

После final target reranker построить candidate spans на:

```text
top-5
или
top-10
```

Сначала посчитать:

```text
raw candidate span oracle
attribution-aware candidate span oracle
```

Attribution-aware correctness:

```text
span_correct_target = 1
только если source chunk ∈ P_v3_target
и predicted span соответствует допустимому gold.
```

Если gold-like строка извлечена из `TM_verified`, это:

```text
wrong_target = 1
span_correct_target = 0
```

Если oracle:

```text
существенно выше top-1 QA
```

только тогда обучать новый SpanRanker.

Если oracle почти не даёт запаса:

```text
SpanRanker stop.
```

---

# 27. D4 — сначала oracle diagnostic

D4 gate в текущем V3 прошёл:

```text
374 hard/failed positives
```

Но перед обучением sentence selector выполнить:

```text
D4-oracle diagnostic
```

Для gate-разработки внутри каждого outer run использовать только:

```text
TRAIN_FOLDS + CALIBRATION_FOLD
```

и сравнить:

```text
reader(full chunk)
reader(gold-containing sentence)
reader(gold sentence ± 1)
```

Gold используется только в diagnostic oracle и никогда в runtime.

Для gold spans, пересекающих границу предложений, отдельно считать:

```text
minimal gold-support window
sentence ±1
```

и не принуждать их к искусственному single-sentence oracle. Использовать только случаи с валидными source offsets.

EVAL_FOLD не использовать для решения «строить D4 или нет» в этом же run.

Считать:

```text
ΔTokenF1
ΔEM
по aspect
по chunk length
по answer position
```

---

# 28. Gate G3 для D4

Обучать sentence-level evidence selector только если oracle diagnostic показывает:

```text
заметный и устойчивый запас
особенно на reader-hard/failed positives.
```

Если gold sentence ±1 почти не улучшает reader:

```text
D4 не развивать.
```

---

# 29. D4 runtime model при прохождении gate

Candidate:

```text
target-reranked positive candidate
```

Sentence selector выбирает:

```text
sentence
или
sentence ±1
```

Требования:

```text
TRAIN-only supervision;
source offsets preserved;
cross-sentence answer support handled;
full-chunk fallback;
не терять provenance.
```

Сравнить:

```text
full chunk
selected sentence
selected sentence ±1
```

---

# 30. D5 — сначала проверить независимость evidence

Текущий gate:

```text
307 / 453 вопросов имеют >=2 original positives
```

Но это может быть результат overlapping chunks.

До D5 выполнить gate-analysis на TRAIN_FOLDS + CALIBRATION_FOLD:

1. получить source intervals positives;
2. объединить overlapping / near-duplicate chunks;
3. получить independent evidence clusters;
4. проверить, сколько вопросов имеют ≥2 независимых clusters;
5. проверить, приводят ли clusters к одному correct answer value.

EVAL_FOLD не использовать для принятия решения «разрабатывать D5 или нет» в том же run.

---

# 31. Gate G4 для D5

Развивать aggregation только если после de-dup остаётся достаточно случаев:

```text
≥2 независимых evidence sources
```

для одного document–aspect.

Не использовать простой count overlapping chunks как support.

---

# 32. D5 aggregation при прохождении gate

Pipeline:

```text
candidate answers
↓
normalize answer string
↓
group exact-normalized candidates
↓
deduplicate overlapping evidence
↓
aggregate calibrated candidate scores
```

Не использовать:

```text
raw start/end logits как межпассажную probability.
```

Сначала простой вариант:

```text
group score = max calibrated candidate score
```

Затем optional:

```text
support-aware term
```

только после CAL calibration.

Semantic/equivalent-answer clustering не включать в первую D5-версию. Его можно добавить отдельной заранее объявленной ablation только при наличии валидированной equivalence rule.

---

# 33. D3 остаётся deferred

Не запускать D3 автоматически.

Возвращаться только если error analysis показывает:

```text
правильный target требует распределённого evidence;
локального chunk + neighbor недостаточно;
D4 не решает проблему.
```

---

# 34. Aspect-specific analysis обязателен

Отдельно:

```text
task
method
material
metric
```

Текущая рабочая гипотеза:

### `method`

```text
основной bottleneck = ranking / target attribution
```

### `material`

```text
ranking относительно сильный,
reader failures заметны
→ D4 особенно интересен
```

### `metric`

```text
есть и ranking, и reader bottleneck
```

### `task`

```text
reader относительно сильный
→ приоритет reranking
```

Это не разные production pipelines заранее, а обязательный error analysis.

---

# 35. Fair comparison по seeds

Текущий V3 выявил ошибку интерпретации:

```text
attention/span n=453
vs
B0 pooled n=1359
```

В V3.1 запрещено сравнивать разные seed aggregation.

Допустимые схемы:

### Схема A

Все trainable кандидаты:

```text
3 seeds
```

сравниваются с B0 тех же seeds.

### Схема B

Детерминированная ветка:

```text
сравнивается с matching B0 prediction set,
на котором построены её candidates.
```

В отчёте всегда указывать:

```text
unique questions
seed count
row count
comparison universe
```

Seeds **не являются независимыми наблюдениями**. Для primary bootstrap у trainable 3-seed систем сначала получить одно question-level outcome, усреднив соответствующую метрику по seeds для каждого `(document_id, question)`, и только затем bootstrap cluster `document_id`.

Per-seed результаты показывать дополнительно как stability analysis.

Для детерминированной/single-seed ветки использовать matching B0 seed и не смешивать её с pooled 3-seed строками.

---

# 36. Статистика

Для каждого ключевого paired comparison:

```text
cluster by document_id
5000 bootstrap replications
```

Обязательные пары:

```text
B1-TM vs B0
TargetReranker vs B0
pairwise shortlist vs best BCE target reranker
text+attention vs text-only
D4 vs pre-D4
D5 vs pre-D5
```

Выводить:

```text
mean paired delta
95% CI
win / lose / tie
```

OOF CI:

```text
model-selection evidence
```

не независимый final confirmatory test.

До финального OOF comparison зафиксировать:

```text
delta_F1_min          # минимально практически значимый выигрыш
runtime_ratio_max
memory_ratio_max
```

и, при достаточной TM coverage:

```text
epsilon_TM_max
```

Если по множеству secondary endpoints выполняются формальные hypothesis tests, заранее использовать одну correction procedure, например Holm. Иначе secondary results трактовать как effect size + CI, а не как набор независимых p-value claims.

---

# 37. Error taxonomy для V3.1

Считать как минимум:

```text
E1  no P_v3_target in shortlist
E2  HN outranks target
E3  verified TM outranks target
E4  GSRN outranks target
E5  unresolved ambiguous selected
E6  easy negative selected
E7  correct target selected, reader wrong
E8  worse target-positive selected among several
E9  wrong-target answer extracted
E10 correct candidate span exists, final selector wrong
E11 target lost by shortlist
E12 D4 removed required context
E13 D5 merged different values
E14 duplicate overlapping evidence counted multiple times
E15 attention favors TM
E16 attention fixes TM but breaks HN
```

---

# 38. Resource metrics

Для каждого живого pipeline:

```text
mean latency / document–aspect
p50
p95
peak GPU
peak RAM
candidate chunks scored
B0 calls
target-reranker calls
reader calls
reader tokens
attention-enabled calls
D4 calls
D5 candidates
cache hit rate
```

Особенно сравнить:

```text
B0
B0 + target reranker
B0 + target reranker + attention
B0 + target reranker + D4
final candidate
```

---

# 39. Правило использования EVAL_FOLD

EVAL_FOLD используется для оценки **заранее определённой** конфигурации.

После просмотра EVAL запрещено в той же итерации:

```text
добавлять feature;
менять taxonomy;
менять shortlist K;
менять loss;
менять attention heads;
менять D4/D5 rule;
и повторно считать тот же EVAL как независимое подтверждение.
```

Новые идеи из EVAL error analysis оформляются как следующая версия эксперимента.

Если сам факт запуска следующей ветки определяется по OOF EVAL предыдущей ветки, следующую ветку оформлять как новую frozen protocol version. Её оценка на тех же folds остаётся model-selection evidence; независимое подтверждение даёт только locked INTERNAL_TEST.

---

# 40. Решение о победителе V3.1

Primary:

```text
document–aspect Token F1
```

Hard safeguards:

```text
HN@1
pairwise_clean_hn
```

Target-attribution metrics:

```text
TargetEvidenceHit@1_v3
TargetMismatch metrics
Pairwise P_v3_target > TM
WrongTargetAnswerRate
```

TargetMismatch как hard safeguard только при достаточной coverage.

До выбора финального кандидата должны быть уже заморожены:

```text
delta_F1_min
runtime_ratio_max
memory_ratio_max
epsilon_TM_max, если TM coverage достаточен
```

Не выбирать систему только по:

```text
pairwise HN
или
Token F1 без attribution analysis.
```

---

# 41. Рекомендуемый порядок запуска

## Phase 0

```text
overlay frozen
↓
overlay audit
```

## Phase 1

```text
B0 saturation check
↓
freeze reference B0
```

## Phase 2 — обязательная

```text
B0
vs
B1-TM
↓
paired bootstrap
↓
Gate G1
```

## Phase 3 — по решению G1-A или G1-B

```text
cross-fitted B0 top-5/top-10
↓
Target Attribution Reranker
↓
text/discourse variants
```

## Phase 4 — только после подтверждённого TM signal

```text
pairwise shortlist
listwise shortlist
```

## Phase 5 — optional specialist internal signal

```text
TRAIN-only attention audit
↓
TM discrimination
↓
text vs attention vs fusion
```

## Phase 6 — reader-side diagnostics

```text
D4 oracle
D5 independent-evidence audit
```

## Phase 7 — только по gates

```text
D4 runtime
D5 aggregation
SpanRanker revisited only with sufficient candidate-span oracle
```

## Phase 8

```text
OOF final comparison
paired bootstrap
Pareto/cost analysis
```

## Phase 9

```text
choose one final V3.1 pipeline
final refit
freeze
INTERNAL_TEST once
```

---

# 42. Final refit

До refit зафиксировать:

```text
architecture
hyperparameters
shortlist K
loss
fusion rule
attention heads/features, если используются
D4/D5 decision
seed policy
epoch policy
calibration policy
```

Final refit зависит от архитектуры.

### Одностадийный B0/B1

Train final model на всех unlocked answerable documents по frozen epoch/hyperparameter policy.

### Двухстадийный B0 → Target Reranker

Нельзя обучать final second-stage reranker на in-sample shortlist финального B0. Выполнить:

```text
1. cross-fitted B0 predictions/shortlists по всем unlocked documents;
2. обучить final target reranker на этих cross-fitted candidates;
3. отдельно обучить final B0 на всех unlocked documents;
4. на INTERNAL_TEST: final B0 → shortlist → frozen final target reranker.
```

Если downstream trainable D4/SpanRanker зависит от upstream predictions, для его training inputs использовать тот же принцип cross-fitted upstream generation.

Если final target reranker использует attention/hidden-state признаки frozen reader, эти признаки для training candidates извлекать из **cross-fitted upstream shortlist**; сам frozen reader может быть общим, поскольку его веса в V3.1 не обучаются на этих folds.

Threshold/calibration/fusion параметры получать из CAL/OOF cross-fitted predictions, а не из in-sample scores.

Если разные outer CAL выбирали разные hyperparameters, заранее использовать deterministic aggregation rule, например:

```text
categorical → mode
ordered numeric → median
continuous fusion/calibration → frozen aggregate-CAL rule
```

INTERNAL_TEST не используется для:

```text
early stopping
head selection
thresholds
calibration
epoch selection
feature selection.
```

Для final epoch использовать заранее зафиксированную CV-derived policy, например:

```text
median best epoch across CV runs
```

или другую deterministic rule, зафиксированную до refit.

---

# 43. Final freeze manifest

Создать:

```text
final_freeze_manifest_v3_1.json
```

Сохранить:

```text
data hashes
overlay hash
CV split hash
model architecture
model checkpoint/hash
B0 reference
shortlist K
target reranker config
attention config
D4/D5 config
scalers
calibrators
seed policy
epoch policy
primary endpoint
safeguards
runtime config
```

---

# 44. INTERNAL_TEST

Открывать только один раз после freeze.

Primary final comparison:

```text
final V3.1
vs
final B0 reference
```

На INTERNAL_TEST не запускать альтернативные V3.1-кандидаты для выбора победителя. Дополнительные замороженные ablations допустимы только если они были заранее объявлены как secondary diagnostic и не влияют на выбор final pipeline.

Primary endpoint:

```text
Token F1
```

Secondary:

```text
EM
Hit@1_original
HN@1
Ambiguous@1_original
GSRN@1_original
cost
```

Если INTERNAL_TEST не имеет заранее подготовленного target-status overlay:

```text
не создавать его после unlock;
не заявлять test TargetMismatch@1;
target-attribution conclusion остаётся OOF/adjudicated unlocked result.
```

---

# 45. Что не запускать сейчас

До B1-TM и Gate G1 не запускать:

```text
full-document B2/B3 снова;
отдельный B4 как ещё одну дублирующую стадию (его роль покрывает R1 shortlist reranker);
новый generic structural mix;
сложный attention ranker;
attention distillation;
AT5 causal masking;
новый SpanRanker;
D3;
final test.
```

Текущий raw multi-passage / max raw QA logits не возвращать как основной метод.

---

# 46. Наиболее вероятная итоговая архитектура, которую должен проверить V3.1

```text
DOCUMENT
↓
B0 answerability
↓
top-5 / top-10
↓
Target Attribution Reranker (text/discourse)
↓
optional reader-aware branch:
    frozen QA on shortlist → predicted span + attention/hidden features
    → specialist TM reranking
↓
target-evidence candidates + cached QA outputs
↓
optional D4 for reader-hard cases
↓
frozen extractive QA / reuse cached QA where applicable
↓
optional D5 independent-evidence aggregation
↓
FINAL ASPECT VALUE
```

Ключевой научный переход:

```text
semantic relevance
↓
answerability
↓
target attribution
↓
evidence-conditioned extraction
```

---

# 47. Критерий содержательного результата

V3.1 считается научно полезным даже без общего F1-выигрыша, если надёжно установлено хотя бы одно:

1. verified target mismatch действительно составляет отдельный систематический класс ошибок;
2. B1-TM снижает target-mismatch ошибки без разрушения HN separation;
3. shortlist target reranking лучше full-document re-ranking;
4. pairwise/listwise полезны только после появления настоящего TM supervision;
5. attention помогает именно `target vs mismatch`, хотя не работает как общий ranker;
6. D4 существенно повышает reader success на hard/failed positives;
7. после de-dup D5 действительно имеет независимое multi-evidence основание;
8. bottleneck различается по аспектам;
9. дальнейший рост ограничен не моделью QA вообще, а конкретной стадией target attribution / evidence conditioning.

---

# 48. Краткое решение-дерево

```text
Есть frozen verified overlay?
    нет → STOP, подготовить данные
    да
     ↓
B1-TM улучшает TM separation и проходит HN safeguards?
    да → перейти к target reranking
    нет → проверить: TM достаточно и часто конкурирует с target в B0 top-K?
             нет → STOP target branch / расширить данные
             да  → один специализированный R1-text после TRAIN/CAL audit
     ↓
Target reranking top-5/top-10 улучшает QA / TM separation?
    нет → анализ ошибок / D4 diagnostics
    да
     ↓
Attention добавляет сигнал поверх text-only?
    нет → удалить attention из final candidate
    да → оставить как specialist TM feature
     ↓
D4 oracle большой?
    нет → D4 stop
    да → train D4
     ↓
Есть ≥2 independent evidence clusters после de-dup?
    нет → D5 stop
    да → test D5
     ↓
final OOF comparison
↓
final refit
↓
freeze
↓
INTERNAL_TEST once
```
