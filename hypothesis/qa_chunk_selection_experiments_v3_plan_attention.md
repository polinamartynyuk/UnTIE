# План дальнейшего эксперимента V3: target-evidence ranking и ранжирование кандидатных ответов для aspect-oriented extractive QA

> **Редакция после повторной методической проверки.** В этой версии дополнительно разведены исходные SciREX-метки и V3 target-status overlay, уточнены метрики при неполной ручной разметке, исправлены attribution-aware labels для SpanRanker, нормировка attention-признаков, grouped-CV/final-refit, а также правила единственного INTERNAL_TEST.

## 0. Назначение эксперимента

Эксперимент V3 является продолжением V2 и должен проверять гипотезы, непосредственно вытекающие из наблюдений V2.

V3 **не должен повторять V2 целиком** и не должен механически продолжать исходную лестницу `B → C → D1 → D2 → D3 → D4 → D5`.

Цель V3 — проверить, можно ли повысить качество aspect-oriented extractive QA за счёт перехода от простой оценки `answerability` к более точному различению:

```text
целевой evidence текущего исследования
vs
правдоподобный, но относящийся к другому объекту / эксперименту / датасету / методу контекст
```

и затем — за счёт отдельного ранжирования уже извлечённых candidate spans.

---

# 1. Исходные результаты V2, которые считаются отправной точкой

## 1.1. Основной рабочий baseline

В V2 честным выбранным кандидатом стал:

```text
B2 = b2a
seed = 44
ranking mode
M = 1
без R3
без C
без D2 fusion
```

На DEVELOPMENT:

```text
Hit@1 = 37.7%
HN@1 = 3.9%
pairwise_clean_hn = 74.6%
Token F1 = 25.71
EM = 19.5%
F1 | selected positive = 59.89
top-1 ambiguous = 54.5%
```

Этот результат использовать как основной ориентир.

## 1.2. Reader oracle

На DEVELOPMENT:

```text
best-positive oracle Token F1 = 71.86
reader-solvable F1 >= 50 = 76.6%
```

Следовательно, основной резерв качества находится **до финального извлечения span**, прежде всего в выборе evidence.

При этом:

```text
B2 F1 | positive = 59.89
oracle best-positive F1 = 71.86
```

значит существует также отдельная задача:

```text
выбор подходящего positive среди нескольких допустимых positives.
```

## 1.3. Hard negatives больше не являются главным источником ошибок

На DEVELOPMENT у B2:

```text
positive       = 29
ambiguous      = 42
easy_negative  = 3
hard_negative  = 3
```

То есть среди неправильных top-1 большая часть относится к `ambiguous`, а не к классическим hard negatives.

Главный subtype:

```text
gold_supporting_robust_negative
```

Он часто содержит правдоподобное значение аспекта, но относится не к целевому объекту исследования.

Типичные случаи:

```text
другой датасет;
другой benchmark;
baseline;
сравниваемый метод;
соседний эксперимент;
описание Related Work;
другая работа, упомянутая в текущей статье.
```

## 1.4. Старые направления, которые не следует продолжать как default

### Generic CE / BM25 / RRF

Не показали устойчивого улучшения поверх B2.

### Raw-logit multi-passage C

CALIBRATION-выигрыш не перенёсся на DEVELOPMENT.

При B2:

```text
M = 1:
Token F1 = 25.71
F1 | positive = 59.89

M = 3:
Token F1 = 20.51
F1 | positive = 30.50
```

Следовательно, текущий способ:

```text
выбрать maximum raw start_logit + end_logit между passages
```

не должен быть основным multi-passage механизмом V3.

### Adaptive M

В V2 фактически выродился почти в always-M=3 и не улучшил B2.

### D2 lexical/structural mix

Не воспроизвёл улучшение E0. Section signal в текущем виде слабый, RRF/BM25 повышает HN@1.

---

# 2. Центральная исследовательская гипотеза V3

## H3.1. Target-evidence hypothesis

После подавления классических hard negatives основной ошибкой выбора контекста становятся фрагменты, содержащие корректную по смыслу аспектную информацию, но относящуюся не к целевому объекту исследования.

Следовательно:

> Для дальнейшего повышения качества недостаточно оценивать `P(answerable | question, chunk)`. Необходимо оценивать, является ли chunk **целевым evidence именно для текущего document–aspect**.

Формально требуется приблизить оценку к:

\[
P(\text{target evidence} \mid q, c, D)
\]

а не только:

\[
P(\text{answerable} \mid q, c).
\]

---

# 3. Дополнительные гипотезы

## H3.2. Document-level ranking hypothesis

Независимая бинарная классификация `(question, chunk)` не полностью соответствует реальной задаче.

Реальная задача:

```text
для одного document–aspect
ранжировать все chunks одного документа
и поднять хотя бы один допустимый target-positive выше distractors.
```

Поэтому listwise/document-level ranking objective может быть эффективнее независимого BCE.

## H3.3. Positive-quality hypothesis

Не все gold-positive chunks одинаково полезны downstream reader.

Даже среди positives требуется предпочитать те, из которых reader способен извлечь корректный span.

## H3.4. Span-ranking hypothesis

Сырые QA logits разных passages недостаточно надёжны для прямого выбора финального ответа.

Лучше:

```text
сначала получить candidate span из нескольких shortlisted chunks,
затем отдельно ранжировать candidate answers.
```

## H3.5. Sentence-evidence hypothesis

D4 имеет смысл только если будет установлено, что часть reader-ошибок внутри positive вызвана локальным шумом / конкурирующими сущностями / избыточным контекстом.

## H3.6. Evidence-aggregation hypothesis

D5 имеет смысл только если анализ покажет, что один и тот же корректный аспектный ответ часто независимо подтверждается несколькими различными source spans.


## H3.7. Attention-signature hypothesis

Внутреннее распределение внимания frozen Transformer-reader может содержать информацию,
которая не выражается явно через:

```text
B-score;
generic relevance;
raw start/end logits;
section;
lexical overlap.
```

Предполагается, что корректное target evidence и успешное извлечение ответа могут сопровождаться
устойчивыми паттернами взаимодействия:

```text
question tokens
↔
answer-bearing region / predicted span
```

в отдельных слоях и attention heads.

Поэтому требуется проверить:

> существуют ли inference-compatible attention-derived признаки, которые позволяют
> отличить target-positive evidence от target-mismatch ambiguous и/или
> reader-success от reader-failure.

Важно: attention в данном эксперименте рассматривается **не как объяснение решения модели**,
а как источник внутренних диагностических и ranking-признаков.

## H3.8. Specialized-head hypothesis

Усреднение attention по всем heads может уничтожать полезный сигнал.

Следует проверить гипотезу, что:

```text
небольшая подгруппа layer×head
```

содержит существенно более сильный сигнал для:

```text
target evidence attribution;
reader success;
passage reranking.
```

Выбор таких heads должен выполняться только на TRAIN_FOLDS,
а размер набора / схема агрегации — только на CALIBRATION_FOLD.

## H3.9. Successful-attention-pattern hypothesis

На TRAIN_FOLDS можно построить эталонные attention signatures для групп:

```text
target-positive + reader-success;
target-positive + reader-failure;
verified target-mismatch;
hard negative;
easy negative.
```

Следует проверить, обладает ли candidate chunk меньшим расстоянием
до профиля успешного target evidence, чем до профилей distractors.

Эта ветка должна быть проверена как отдельный простой baseline,
например через:

```text
cosine distance;
regularized Mahalanobis distance;
linear discriminant / logistic probe.
```

Не начинать сразу со сложной нейронной модели.

## H3.10. Internal-signal complementarity hypothesis

Если attention-derived признаки полезны, необходимо проверить,
дают ли они **новую** информацию поверх текстового target-evidence ranker.

Основные сравнения:

```text
text-only;
attention-only;
text + attention;
text + attention + reader hidden-state features.
```

Если улучшение возникает только при объединении сигналов,
это будет свидетельствовать об их комплементарности, а не о замене B/target-evidence ветки.


---

# 4. Главное методологическое изменение: не использовать текущий DEVELOPMENT как единственный селекционный набор V3

Текущий DEVELOPMENT уже многократно использовался:

```text
для выбора B2 variant;
для выбора семейства R3;
для выбора D2;
для анализа R3_3;
для формирования гипотез V3.
```

Поэтому нельзя строить V3 так:

```text
новая модель
→ проверить на том же DEVELOPMENT
→ выбрать лучшую
→ считать это подтверждением.
```

Это приведёт к post-hoc оптимизации.

---

# 5. Данные V3

## 5.1. INTERNAL_TEST остаётся закрытым

Существующий:

```text
internal_test:
31 документ
54 answerable-вопроса
```

не открывать.

Запрещено использовать его:

```text
для taxonomy;
для ручного анализа ошибок;
для threshold tuning;
для выбора model family;
для model selection;
для выбора loss;
для выбора weights;
для выбора D4/D5;
для отбора аспектов;
для определения practical-success criteria.
```

## 5.2. seen_exploratory_dev

Official SciREX DEV сохранить как:

```text
seen_exploratory_dev
```

Его можно использовать только как:

```text
дополнительный post-hoc sanity check направления эффекта.
```

Не использовать для model selection V3.

## 5.3. Основной исследовательский пул V3

Использовать только разблокированные документы official TRAIN:

```text
train
+
calibration
+
development
```

из V2.

Всего:

```text
180 + 44 + 44 = 268 документов
```

INTERNAL_TEST = 31 документов остаётся вне этого пула.


## 5.4. Primary scope V3 — только answerable document–aspect

Основные ranking/QA эксперименты V3 выполняются на вопросах, для которых в исходной разметке
есть хотя бы один `robust_positive`.

Сохраняется логика:

```text
include_unanswerable = false
```

для primary V3 comparison.

Вопросы без `robust_positive`:

```text
не считать автоматически no-answer;
не использовать как negative examples;
не включать в target-mismatch supervision;
не использовать для обучения null/no-answer classifier в V3.
```

Они остаются отдельным будущим направлением / audit-задачей.

Если ручной аудит обнаружит потенциальный target evidence среди ранее `no-robust-positive`,
такой случай сохранить как diagnostic false-negative candidate,
но не менять primary V3 universe задним числом.

---

# 6. Новая схема валидации: grouped cross-validation до финального TEST

Чтобы не зависеть от одного уже просмотренного DEVELOPMENT, создать:

```text
V3_CV_SPLITS
```

по `document_id`.

Рекомендуемый вариант:

```text
5 document-level folds
```

с фиксированным seed.

Для каждого outer fold:

```text
1 fold → EVAL_FOLD
1 fold → CALIBRATION_FOLD
3 fold → TRAIN_FOLDS
```

Роли циклически переставлять так, чтобы каждый fold:

```text
один раз был EVAL
один раз был CALIBRATION
несколько раз входил в TRAIN.
```

Пример:

```text
run 0:
train = F0,F1,F2
cal   = F3
eval  = F4

run 1:
train = F1,F2,F3
cal   = F4
eval  = F0

run 2:
train = F2,F3,F4
cal   = F0
eval  = F1

run 3:
train = F3,F4,F0
cal   = F1
eval  = F2

run 4:
train = F4,F0,F1
cal   = F2
eval  = F3
```

Таким образом получить **out-of-fold predictions** для всех 268 unlocked документов.

## 6.1. Почему это необходимо

V3 содержит новые гипотезы, сформированные после просмотра старого DEVELOPMENT.

Поэтому повторная оценка на новых document-level folds нужна для снижения риска подгонки под 44 уже изученных документа.


## 6.2. Как формировать folds

Folds создаются **до V3 target-status annotation** и до обучения новых моделей.

Единица группировки:

```text
document_id
```

Ни один документ не может одновременно находиться в TRAIN / CALIBRATION / EVAL одного run.

При распределении документов желательно балансировать folds по исходным, уже доступным характеристикам:

```text
число answerable-вопросов;
аспекты task/method/material/metric;
число исходных robust_positive;
HN bucket;
число исходных ambiguous.
```

Не балансировать folds по будущим V3 `target_mismatch` labels, поскольку они появляются позже.

После создания сохранить:

```text
v3_cv_splits.json
split_seed
алгоритм group assignment
per-fold data summary
hash списка document_id
```

и больше не менять folds по результатам моделей.

Если обычный `GroupKFold` даёт заметный дисбаланс аспектов,
разрешена детерминированная grouped-stratification процедура,
но она должна быть реализована и зафиксирована **до** model evaluation.

---

# 7. Фаза A — расширение анализа ambiguous и создание новой target-mismatch taxonomy

Это обязательный этап перед обучением новой модели.

## 7.1. Не считать все ambiguous negatives

Исходные subtypes:

```text
gold_supporting_robust_negative
false_negative_candidate
question_sensitive
confirmed_false_negative
```

не переводить автоматически в один negative-класс.

## 7.2. Создать новый аналитический статус

Для каждого ambiguous chunk определить:

```text
target_match_status =
    target_evidence
    target_mismatch
    partial_support
    unresolved
```

### target_evidence

Chunk содержит допустимое evidence именно для целевого аспекта текущей статьи.

### target_mismatch

Chunk содержит правдоподобную аспектную информацию, но она относится к другому объекту:

```text
другой метод;
другой датасет;
другой benchmark;
baseline;
related work;
сравниваемая система;
другой эксперимент;
чужой результат.
```

### partial_support

Chunk относится к целевой работе, но содержит только часть необходимого evidence / вспомогательный контекст.

### unresolved

Невозможно надежно решить без дополнительной экспертизы.

## 7.3. Разметка должна делаться только на unlocked data

Не смотреть INTERNAL_TEST.

## 7.4. Приоритет ручной проверки

В первую очередь:

```text
B2 top-1 ambiguous
B2 top-5 ambiguous
highest B2 score ambiguous
gold_supporting_robust_negative
question_sensitive
false_negative_candidate с высоким score
```

## 7.5. Требуемый минимальный результат

Не обязательно переразмечать все ambiguous.

Достаточно получить надёжную обучающую выборку:

```text
verified_target_mismatch
```

и отдельно:

```text
verified_target_evidence
```

Неуверенные случаи сохранять `unresolved`.

## 7.6. Inter-annotator agreement

Если возможно привлечь второго эксперта хотя бы к части выборки:

```text
10–20% verified examples
```

оценить согласие.

Если второй эксперт недоступен — сохранить:

```text
annotation_reason
source sentence
target entity/value
mismatch entity/value
confidence
```

для последующего аудита.


## 7.7. Разделить разметку и разработку модели

Чтобы grouped CV оставался честным:

- taxonomy и annotation guideline должны быть зафиксированы **до** OOF model comparison;
- verified labels могут быть подготовлены для всех unlocked folds, но в конкретном CV-run:
  - TRAIN labels разрешены для обучения;
  - CALIBRATION labels разрешены для настройки;
  - EVAL labels используются только для метрик;
- запрещено после ручного просмотра ошибок конкретного EVAL_FOLD добавлять новый feature/rule и повторно считать его на том же EVAL как подтверждающий результат;
- такой новый feature переносится в **новую заранее объявленную итерацию протокола**.

Если моделирующий исследователь сам выполняет разметку, желательно заранее сохранить:

```text
annotation_snapshot_hash
taxonomy_version
annotation_date
```

и отделить фазу annotation freeze от model-selection phase.


## 7.8. Исходные labels и V3 target-status — разные слои разметки

Исходный Phase 1 / SciREX-derived label **никогда не переписывать**.

Для каждого chunk хранить отдельно:

```text
source_label
source_ambiguous_subtype
v3_target_status
v3_target_value
v3_gold_relation
v3_annotation_confidence
v3_annotation_reason
```

`v3_gold_relation`:

```text
matches_existing_gold
semantically_equivalent_to_existing_gold
new_valid_value_candidate
no_gold_match_expected
unclear
```

Нельзя молча расширять официальный gold набор после обнаружения нового правдоподобного значения.

Если эксперт считает, что исходный gold неполон:

```text
v3_gold_relation = new_valid_value_candidate
```

и такой случай идёт в отдельный audit.

Для **primary V3 comparison** дополнительный `verified target_evidence`
включать в расширенный positive set только если его value:

```text
matches_existing_gold
или
semantically_equivalent_to_existing_gold
```

Новые валидные значения можно анализировать отдельно как `augmented-gold sensitivity analysis`,
но нельзя смешивать с primary V2-comparable evaluation без отдельной заранее зафиксированной версии gold.

Пример:

```text
source_label = ambiguous
source_ambiguous_subtype = false_negative_candidate
v3_target_status = target_evidence
```

означает, что исходная метка сохранена, но V3-аудит считает этот chunk допустимым target evidence.

Это необходимо, чтобы одновременно иметь:

```text
сопоставимость с V2 / исходной разметкой
и
расширенную V3 target-attribution оценку.
```

Определить два множества positives:

```text
P_original =
    исходные robust_positive

P_v3_target =
    исходные robust_positive
    ∪ verified target_evidence из V3 overlay,
      для которого v3_gold_relation ∈
      {matches_existing_gold, semantically_equivalent_to_existing_gold}
```

`partial_support` и `unresolved` не входят в `P_v3_target`.

Аналогично:

```text
TM_verified =
    только chunks с v3_target_status = target_mismatch
```

Нельзя считать весь исходный `ambiguous` класс target mismatch.

## 7.9. Sampling ручной разметки: hard-case + representative

Если нет ресурсов разметить все ambiguous, нельзя аннотировать только B2 top-errors.

Обязательны два strata:

### Hard-case stratum

```text
B0/B2 top-1 ambiguous;
top-5 ambiguous;
высокие answerability scores;
GSRN;
question_sensitive;
high-score false_negative_candidate.
```

### Representative stratum

Случайная/стратифицированная выборка ambiguous по:

```text
aspect;
original subtype;
B-score quantile;
позиция в документе;
section;
длина chunk.
```

Это требуется, чтобы новый target-evidence classifier не обучался только на наиболее трудных примерах
и чтобы оценка prevalence target mismatch не была смещена selection bias.

Сохранить:

```text
annotation_sampling_stratum
sampling_probability, если известна
```

и отдельно отчёт по каждому stratum.


## 7.9.1. По возможности скрыть score модели от аннотатора

Model score можно использовать для формирования очереди hard cases,
но при принятии решения `target_evidence / target_mismatch / partial / unresolved`
аннотатору желательно не показывать:

```text
B0/B2 score
attention score
predicted class новой модели
model rank
```

чтобы снизить confirmation bias.

Аннотатор должен опираться на:

```text
aspect question;
текст chunk;
необходимый document context;
исходный gold/value только в рамках зафиксированного annotation guideline.
```

Sampling metadata сохраняется отдельно и может быть раскрыта после annotation freeze.

## 7.10. Неполная разметка должна быть видна в метриках

Если не все selected ambiguous получили `v3_target_status`,
любая target-mismatch метрика должна сопровождаться coverage.

Минимум:

```text
TargetStatusCoverage@1
TargetStatusCoverage@5
```

Для top-1:

\[
Coverage^{target}_{@1}
=
\frac{
\#\{q:\ top1_q\ имеет\ известный\ V3\ target\ status\}
}{
N
}.
\]

Для `robust_positive`, `hard_negative` и `easy_negative`
статус source-класса известен; проблема coverage прежде всего относится к исходным `ambiguous`.

Если coverage неполный, выводить две величины:

```text
TargetMismatch@1_lower_bound
=
# verified target-mismatch top1 / N

TargetMismatchRate@1_adjudicated
=
# verified target-mismatch top1 /
# top1 cases с adjudicated target-status.
```

Не использовать `TargetMismatch@1` как жёсткий model-selection safeguard,
если coverage ниже заранее заданного порога.

Рекомендуемый default:

```text
minimum TargetStatusCoverage@1 for safeguard = 0.95
```

Если этот порог не достигнут,
`TargetMismatch` остаётся diagnostic metric,
а универсальными safeguards служат исходные HN / ambiguous / GSRN показатели и QA.

---

# 8. Новые метрики V3

Старые метрики сохранить:

```text
Hit@1
Hit@5
Coverage@K
MRR
HN@1
pairwise Pos>HN
Token F1
EM
Char F1
ROUGE-L
F1 | selected positive
```

Но этого уже недостаточно.

## 8.1. Ambiguous@1_original

\[
Amb@1_{orig}=P(source\_label(top1)=ambiguous)
\]

Это **метрика исходной разметки**, полезная для сопоставления с V2.

Она не означает автоматически, что выбранный chunk ошибочен:
часть исходных ambiguous после V3-аудита может стать `verified target_evidence`.

## 8.2. TargetMismatch@1

Использовать только verified V3 overlay.

При полной target-status coverage:

\[
TM@1=P(v3\_target\_status(top1)=target\_mismatch).
\]

При неполной разметке обязательно выводить:

```text
TargetStatusCoverage@1
TargetMismatch@1_lower_bound
TargetMismatchRate@1_adjudicated
```

по определениям §7.10.

## 8.3. GSRN@1_original

Отдельно для исходного subtype:

```text
gold_supporting_robust_negative
```

Это важный **универсально доступный diagnostic proxy**,
но не эквивалент `verified target_mismatch`:
часть GSRN теоретически может оказаться ошибкой исходной разметки.

## 8.4. Pairwise P_original > Ambiguous_original

Диагностическая метрика V2-comparability.

На вопросах, где есть:

```text
исходный robust_positive
и
исходный ambiguous.
```

Не интерпретировать как чистый target-attribution показатель.

## 8.5. Pairwise P_v3_target > TargetMismatch_verified

Основная новая target-attribution ranking-метрика.

Считать только на вопросах, где существуют обе проверенные стороны.

## 8.6. Pairwise P_original > GSRN_original

Отдельный диагностический показатель для сопоставления с V2 taxonomy.

## 8.7. TargetEvidenceHit@K_v3

Использовать:

```text
P_v3_target
```

из §7.8.

Считать как отдельную расширенную метрику.

Одновременно всегда сохранять:

```text
Hit@K_original
```

по исходным `robust_positive`,
чтобы результаты V3 оставались сопоставимыми с V2.

Если V3 annotation неполная, `TargetEvidenceHit@K_v3`
является conservative expanded-gold estimate:
непроверенный ambiguous не считается target evidence и не должен молча становиться negative.

## 8.8. WrongTargetAnswerRate

Эта метрика требует source attribution.

На unlocked/adjudicated data считать случай `wrong_target`, если:

```text
финальный answer span выбран из verified target_mismatch chunk
```

даже если текст span случайно совпадает с одной из gold-строк.

При неполной target-status annotation обязательно выводить coverage.

Дополнительно для всех splits с исходными chunk labels считать универсальный diagnostic:

```text
GoldMatchFromNonOriginalPositive
```

то есть долю случаев, где финальная строка совпала с gold,
но source chunk не является исходным `robust_positive`.

Эта метрика не доказывает target mismatch:
non-original-positive source может быть настоящим false negative исходной разметки.
Она только сигнализирует, что строковая QA-метрика и source-attribution расходятся.

---

# 9. Новый baseline V3-B0

Для каждого fold переобучить B2 по тому же рецепту, который победил в V2:

```text
question-balanced weighted BCE
без статического oversampling
только training-eligible HN
early stopping на CALIBRATION_FOLD
```

Это:

```text
V3-B0
```

Не использовать старый checkpoint seed 44 как единственный baseline для CV.

В каждом fold B0 должен обучаться только на TRAIN_FOLDS.

Это необходимо для честного paired comparison.

---

# 10. Эксперимент V3-B1 — расширенная answerability с verified target-mismatch negatives

Цель:

проверить, достаточно ли просто добавить новый тип сложных отрицательных примеров.

## 10.1. Train classes

Positive (`P_v3_target`):

```text
исходный robust_positive
+
verified target_evidence из V3 overlay
с допустимым `v3_gold_relation` для primary V3
```

Negative:

```text
training_eligible_hard_negative
+
verified target_mismatch из V3 overlay
+
ограниченная доля easy negatives
```

Исходный `source_label` при этом не менять.

Не включать:

```text
partial_support
unresolved
```

как сильные negatives.

## 10.2. Loss

Начать с того же:

```text
question-balanced weighted BCE
```

чтобы менять только training target, а не одновременно loss.

## 10.3. Сравнение

```text
B0
vs
B1
```

Главное:

```text
TM@1 ↓
GSRN@1 ↓
Ambiguous@1 ↓
Hit@1 ↑
Token F1 ↑
```

при сохранении:

```text
HN@1
pairwise Pos>HN.
```

---

# 11. Эксперимент V3-B2 — pairwise target-evidence ranking

Если B1 даёт положительный сигнал, проверить pairwise objective.

Training triplets:

```text
(question, positive, target_mismatch)
(question, positive, hard_negative)
```

Не смешивать автоматически все negative types с одинаковым весом.

Поддержать отдельные sampling probabilities:

```text
p_target_mismatch
p_hard_negative
p_easy
```

## 11.1. Loss

Например:

\[
L=-\log\sigma(s^+-s^-)
\]

или margin ranking.

## 11.2. Вопросы с несколькими positives

Не заставлять positives конкурировать друг с другом.

Для negative sampling положительный контекст выбирать из множества допустимых positives.

---

# 12. Эксперимент V3-B3 — listwise / document-level target-evidence ranking

Это основной новый архитектурный эксперимент.

Для одного `(document, aspect)`:

```text
C = все candidate chunks документа
P = множество допустимых target positives
```

Модель выдаёт:

\[
s(q,c)
\]

для каждого chunk.

Обучать через document-level normalization:

\[
L=
-\log
\frac{
\sum_{c\in P}\exp(s(q,c)/T)
}{
\sum_{c'\in C_{labeled}}\exp(s(q,c')/T)
}
\]

где supervised denominator `C_labeled` включает только кандидатов с допустимой обучающей ролью:

```text
P_v3_target
verified target mismatch
training-eligible HN
ограниченное число easy
```

`partial_support` и `unresolved` маскировать из supervised loss.

Так как в этой задаче число chunks на документ умеренное,
предпочтительно включать **все доступные labeled candidates документа**,
а не имитировать document-level objective одним случайным negative.

Если по памяти используется sampled denominator,
sampling-policy должна быть одинаковой для сравниваемых моделей и зафиксирована заранее.

## 12.1. Температура T

Если используется:

```text
T
```

подбирать только на CALIBRATION_FOLD.

## 12.2. Несколько positives

Loss должен учитывать сумму вероятностей всех допустимых positives.

Нельзя случайно назначить один единственный positive, если evidence корректно встречается в нескольких чанках.

## 12.3. Главный вопрос

Даёт ли document-level objective более прямое улучшение:

```text
Hit@1
TargetMismatch@1
Token F1
```

чем independent BCE.

---

# 13. Эксперимент V3-B4 — hierarchical / two-stage target evidence discrimination

Если target-mismatch negatives слишком сложны для одной binary head, проверить:

```text
Stage 1:
B0 answerability

Stage 2:
target-evidence discriminator
```

Pipeline:

```text
all chunks
    ↓
B0
    ↓
top-K shortlist
    ↓
TargetEvidenceCE
    ↓
ranking
```

`K` проверить:

```text
3
5
8
```

Критерий:

не терять evidence recall на Stage 1.

Перед Stage 2 обязательно считать:

```text
TargetEvidenceHit@K
```

Если правильный target evidence не входит в shortlist, Stage 2 не может исправить ошибку.

---

# 14. Новая роль структурного контекста

Не возвращать старый D2 `section_title lexical overlap`.

Вместо этого структура должна помогать определить:

```text
кто является субъектом утверждения
и к какому эксперименту/методу/датасету относится значение.
```

## 14.1. Простые признаки, допустимые для первой проверки

Без сложного knowledge graph:

```text
section type
section position
sentence position
previous/next sentence
наличие citation marker рядом
наличие comparative markers
```

Например признаки/паттерны:

```text
"we use"
"we evaluate"
"our model"
"our experiments"
"we train"
"we propose"

vs

"previous work"
"baseline"
"compared with"
"X et al."
"reported"
"uses"
```

Важно:

не реализовывать это как ручную rule-based финальную систему.

Эти признаки могут использоваться:

```text
как дополнительные features;
как context для Cross-Encoder;
как error-analysis categories.
```

## 14.2. Контекстный вход

Проверить:

```text
question + chunk
```

против:

```text
question + section title + previous sentence + chunk + next sentence
```

при одинаковом model backbone.

Это более содержательная structural-гипотеза, чем старый scalar section-overlap.

---

# 15. Анализ positive-quality: какой positive удобен reader-у

До создания нового reader-aware ranker провести отдельный анализ.

Для каждого positive сохранить:

```text
reader Token F1
reader EM
chunk length
gold span position
gold answer occurrence count
число competing entities
section
distance to section start
neighbor sentences
```

## 15.1. Основные сравнения

```text
reader-easy positives:
F1 >= 80

reader-medium:
50 <= F1 < 80

reader-hard:
0 < F1 < 50

reader-failed:
F1 = 0
```

Проверить, какие свойства statistically/empirically связаны с reader success.

## 15.2. Цель

Не использовать raw QA score как proxy автоматически.

Сначала установить:

```text
какие observable inference-time признаки
реально коррелируют с extractability.
```

---


# 16. Attention-based ветка V3-AT: общие принципы

Attention-ветка является **дополнительной** линией экспериментов.

Она не заменяет:

```text
V3-B0;
V3-B1;
V3-B2;
V3-B3;
V3-B4;
структурно-дискурсивную ветку;
positive-quality analysis;
span-level reranking.
```

Необходимо проверить и исходные предложения V3, и новые attention-based варианты
на одинаковых document-level folds.

## 16.1. Какие attention анализируются

Основной reader V3 остаётся frozen RoBERTa extractive QA.

Для RoBERTa это encoder self-attention над объединённой последовательностью,
содержащей вопрос и контекст.

Не называть его decoder cross-attention.

Для каждого слоя `l` и головы `h` имеется attention matrix:

\[
A^{(l,h)} \in \mathbb{R}^{T\times T}.
\]

Полную матрицу нельзя непосредственно использовать как обычный фиксированный feature vector,
поскольку она:

```text
зависит от длины последовательности;
очень велика;
дорога для хранения;
содержит много избыточных heads.
```

Следовательно, основная стратегия:

```text
raw attention
→ компактные агрегированные признаки
→ head selection
→ attention signature
→ ranking / classification.
```


Важно: результаты о `retrieval heads` из long-context decoder LLM рассматриваются только как научный прецедент.

Нельзя предполагать заранее, что:

```text
те же heads;
та же локализация по слоям;
та же causal роль
```

будут существовать в frozen RoBERTa extractive reader.

Для V3 это отдельная проверяемая гипотеза.
Если head-specific сигнал не переносится на RoBERTa, это является валидным отрицательным результатом.

## 16.2. Gold-compatible и inference-compatible признаки должны быть разделены

Разрешены два режима.

### Diagnostic-only

Можно использовать gold span для анализа:

```text
как выглядит attention на настоящий answer span?
отличается ли он от attention на ошибочный predicted span?
```

Такие признаки маркировать:

```text
gold_dependent = true
diagnostic_only = true
```

Они **никогда** не участвуют в живом reranking.

### Inference-compatible

Для реального ranking разрешены только признаки, вычисляемые из:

```text
question;
candidate chunk;
frozen reader forward pass;
predicted span;
attention;
hidden states;
reader logits.
```

Gold answer и gold span недоступны.

## 16.3. Attention нельзя трактовать как готовое explanation

Во всех отчётах использовать термины:

```text
attention-derived feature;
attention signature;
internal reader signal;
selected attention heads.
```

Не писать без отдельной causal validation:

```text
"модель считает эти токены причиной ответа";
"attention объясняет решение";
"высокий attention = важность".
```

Это принципиально важно для корректной научной интерпретации.

---

# 17. V3-AT0 — техническая верификация извлечения attention

До научного анализа убедиться, что включение attention не меняет baseline reader.

## 17.1. Regression check

На фиксированной небольшой выборке выполнить frozen reader:

```text
обычный inference;
inference с возвратом attention.
```

Проверить:

```text
predicted span идентичен;
start logits совпадают в пределах tolerance;
end logits совпадают в пределах tolerance;
EM/F1 идентичны;
dropout выключен;
model.eval() активен.
```

Если backend Transformer при `output_attentions=True` переключает реализацию attention/kernel,
это явно зафиксировать.

## 17.2. Разметка question/context token ranges

Не предполагать наличие корректных `token_type_ids`, поскольку RoBERTa обычно на них не опирается.

Маски:

```text
question_token_mask
context_token_mask
predicted_span_token_mask
special_token_mask
```

строить из фактического tokenizer encoding.

Для fast tokenizer предпочтительно использовать:

```text
BatchEncoding.sequence_ids()
offset_mapping
special_tokens_mask
```

где `sequence_ids()` однозначно разделяет first/second sequence
(вопрос и контекст) независимо от конкретного набора separator tokens.

Ручное восстановление границ по special token ids использовать только как fallback
и покрыть отдельными unit tests.

Добавить unit test на:

```text
короткий вопрос;
длинный вопрос;
контекст с truncation;
predicted span у начала/конца context;
special tokens.
```

## 17.3. Source offsets

Predicted span должен по-прежнему иметь mapping:

```text
token span
→ chunk char offsets
→ document source offsets.
```

Attention instrumentation не должна ломать существующую provenance-схему.

## 17.4. Не сохранять все raw tensors по умолчанию

Полные tensors `layers × heads × T × T` могут быстро занять большой объём.

Default:

```text
attention tensor существует только во время forward;
feature extractor сразу строит compact signature;
на диск сохраняется signature + metadata.
```

Raw matrices разрешено сохранять только:

```text
для ограниченной audit sample;
для визуализаций;
для unit tests;
для нескольких типичных ошибок.
```

## 17.5. Отдельно измерить стоимость

Сохранить:

```text
latency attention off;
latency attention on;
peak GPU memory;
peak RAM;
размер signature cache.
```

---

# 18. V3-AT1 — attention audit: существует ли различимый сигнал

Это **первый научный attention-эксперимент**.

Не обучать attention-ranker до его завершения.

## 18.1. Группы сравнения

На TRAIN_FOLDS сформировать:

```text
G1 = target-positive + reader-success
G2 = target-positive + reader-medium/failure
G3 = verified target-mismatch
G4 = training-eligible hard negative
G5 = easy negative
```

Для reader-success использовать несколько заранее фиксированных диагностических порогов:

```text
F1 >= 80;
F1 >= 50;
F1 = 0.
```

Основная интерпретация не должна зависеть от единственного произвольного threshold.

## 18.2. Два span режима

Для positives:

```text
predicted-span attention — inference-compatible;
gold-span attention — diagnostic-only.
```

Сравнение этих двух режимов позволит выяснить:

```text
reader видит правильный evidence, но выбирает неверный span?
или attention уже концентрируется на другой области?
```

## 18.3. Анализ отдельно по аспектам

Обязательно:

```text
task;
method;
material;
metric.
```

Особенно `metric`, поскольку V2 показал reader limitation даже на gold-positive.

## 18.4. Основной вопрос AT1

Нужно установить:

> есть ли layer/head/features, на которых распределения G1 и G3
> или G1 и G2 заметно разделяются на TRAIN и сохраняют направление
> на CALIBRATION.

Если различимого сигнала нет, сложные attention-rankers не развивать как основную ветку.

---

# 19. V3-AT2 — формирование attention signature

Для каждого `(question, chunk)` строить компактный вектор признаков.

Обозначения:

```text
Q = question tokens
C = context tokens
S = predicted answer span tokens
```

## 19.1. Question → predicted span attention

Для каждой `(l,h)` сначала считать raw attention mass:

\[
m^{(l,h)}_{Q\to S}
=
\sum_{i\in Q}\sum_{j\in S}A^{(l,h)}_{ij}.
\]

Поскольку raw mass механически растёт с длиной predicted span,
основным length-normalized признаком сделать:

\[
\bar A^{(l,h)}_{Q\to S}
=
\frac{
m^{(l,h)}_{Q\to S}
}{
|Q|\cdot |S|
}.
\]

Дополнительно считать долю от всей question→context attention mass:

\[
r^{(l,h)}_{Q\to S}
=
\frac{
m^{(l,h)}_{Q\to S}
}{
\sum_{i\in Q}\sum_{j\in C}A_{ij}^{(l,h)}+\epsilon
}.
\]

Raw mass сохранять только вместе с:

```text
question_length
span_length
context_length
```

чтобы модель не получала скрытый proxy длины без контрольной абляции.

## 19.2. Predicted span → question attention

\[
m^{(l,h)}_{S\to Q}
=
\sum_{i\in S}\sum_{j\in Q}A^{(l,h)}_{ij},
\]

\[
\bar A^{(l,h)}_{S\to Q}
=
\frac{
m^{(l,h)}_{S\to Q}
}{
|S|\cdot |Q|
}.
\]

Основным признаком считать нормированную величину;
raw mass — только secondary/control feature.

Проверить, несёт ли двунаправленная связь:

```text
Q ↔ S
```

больше информации, чем только `Q → S`.

## 19.3. Attention concentration / entropy

Сначала для каждой `(l,h)` построить корректно нормированное распределение по context tokens:

\[
p^{(l,h)}_j
=
\frac{
\sum_{i\in Q}A^{(l,h)}_{ij}
}{
\sum_{k\in C}\sum_{i\in Q}A^{(l,h)}_{ik}+\epsilon
},
\quad j\in C.
\]

Затем вычислить:

\[
H^{(l,h)}=-\sum_{j\in C}p_j\log(p_j+\epsilon).
\]

Для сравнения контекстов разной длины основным считать:

\[
H^{(l,h)}_{norm}
=
\frac{H^{(l,h)}}{\log |C|}
\]

при `|C| > 1`.

Дополнительно:

```text
top-1 token mass;
top-k token mass;
effective support size = exp(H);
по возможности Gini/concentration index.
```

Не предполагать заранее:

```text
correct = low entropy.
```

Направление должно определяться экспериментально.

## 19.4. Competing-region margin

Разбить context на локальные окна / предложения.

Поскольку предложения/окна имеют разную длину,
основной region score должен быть length-normalized:

```text
mean question→region attention per region token
```

Raw region mass сохранять только как secondary feature.

Считать:

```text
normalized score predicted-span region
-
normalized score strongest competing region.
```

Также:

```text
число competing regions выше заданной доли от maximum;
отношение top1/top2 normalized region score.
```

Правило сегментации на предложения/окна фиксировать до EVAL.

Пороговые схемы выбирать только на CALIBRATION.

## 19.5. Head agreement

Для каждого candidate:

```text
доля heads, где хотя бы один token predicted span входит в top-X% context tokens;
доля heads, где mean attention predicted-span region является максимальным среди regions;
доля heads, где Q→S normalized ratio выше threshold.
```

Первый вариант чувствителен к длине span,
поэтому обязательно дополнительно считать control:

```text
expected top-X hit rate при случайном span той же длины
```

или использовать region-level normalized score как основной head-agreement показатель.

Проверить несколько заранее заданных `X`, например:

```text
1%;
5%;
10%.
```

## 19.6. Layer dynamics

Для каждого признака получить:

```text
early-layer summary;
middle-layer summary;
late-layer summary;
late - early;
maximum over layers;
layer index of maximum.
```

Не ограничиваться последним слоем.

## 19.7. Head dispersion

Считать:

```text
mean;
std;
max;
top-k mean
```

по heads внутри layer и по selected heads.

Это позволяет различать:

```text
одиночный случайный высокий head
vs
устойчивое согласование нескольких heads.
```

## 19.8. Special-token признаки

Attention на `<s>` / separator tokens разрешено исследовать только как secondary diagnostic features.

Не приписывать им заранее семантику `CLS relevance`, если архитектура конкретного reader этого явно не гарантирует.


## 19.9. Обязательные control features

Чтобы не принять длину/позицию за attention-эффект,
для каждого candidate сохранить:

```text
question_length
context_length
predicted_span_length
predicted_span_relative_position
chunk_rank_before_attention
raw QA span score
```

В attention-analysis сравнивать как минимум:

```text
CTRL = length + position controls
ATT = attention features
CTRL + ATT
```

Attention считается действительно дополнительным сигналом,
если он улучшает OOF результат относительно соответствующего control baseline.

## 19.10. Масштабирование признаков

Все непрямые distance/probe features стандартизировать:

```text
fit mean/std только на TRAIN_FOLDS
apply без refit к CALIBRATION/EVAL
```

Для каждого fold сохранять scaler parameters.

Запрещено нормировать feature columns по объединённому TRAIN+CAL+EVAL.

---

# 20. V3-AT3 — поиск информативных attention heads

Это отдельный шаг между attention audit и ranker.

## 20.1. Head selection только на TRAIN_FOLDS

Для каждого `(layer, head)` оценить способность признаков различать.

Чтобы уменьшить multiple-comparison overfit по большому числу heads,
предпочтительно получать head-score через **inner grouped resampling внутри TRAIN_FOLDS**,
а не по одному fit/eval на тех же TRAIN examples.

EVAL_FOLD в head selection не участвует.

Задачи:

### Target attribution task

```text
target-positive
vs
verified target-mismatch.
```

### Reader-success task

```text
target-positive reader-success
vs
target-positive reader-failure.
```

Возможные диагностические критерии:

```text
AUROC;
PR-AUC;
standardized effect size;
mutual information;
univariate logistic coefficient.
```

Не выбирать heads по EVAL_FOLD.

## 20.2. Ограничить множество размеров selected-head set

Вместо свободного перебора сотен комбинаций заранее проверить небольшой набор:

```text
N_heads ∈ {1, 2, 4, 8, 16}
```

если архитектура содержит достаточно heads.

Конкретный `N` выбирать на CALIBRATION.

## 20.3. Stability criterion

Для selected heads сохранять:

```text
частоту выбора по folds;
частоту выбора по seeds;
Jaccard overlap top-N sets;
слой/head.
```

Особенно интересны heads, которые повторно выбираются в разных folds.

## 20.4. Baseline all-head average обязателен

Сравнить:

```text
all-head mean attention;
last-layer mean;
middle-layer mean;
selected-head attention.
```

Это позволит доказать, действительно ли head selection нужен.


## 20.5. Shared heads — default

Основной V3 вариант использует общий набор selected heads для всех четырёх аспектов.

Aspect-specific head sets разрешены только как заранее объявленная secondary ablation,
если в каждом TRAIN_FOLD достаточно примеров соответствующего аспекта.

Не вводить четыре отдельных head-набора post-hoc после просмотра EVAL,
поскольку это резко увеличивает пространство гиперпараметров.

---

# 21. V3-AT4 — attention-pattern и attention-signature ranking

Должны быть проверены **несколько простых вариантов**, чтобы не приписать эффект сложной модели.

## 21.1. AT4-P — prototype distance

После TRAIN-only стандартизации attention signature построить прототипы:

\[
\mu_{success},
\mu_{target},
\mu_{mismatch},
\mu_{readerFail}.
\]

Для candidate attention signature `z` считать:

```text
cosine(z, μ_success);
cosine(z, μ_target);
cosine(z, μ_mismatch);
distance-to-success;
distance-to-mismatch;
разность этих расстояний.
```

Дополнительно разрешён regularized Mahalanobis,
если covariance оценивается устойчиво и regularization выбирается без EVAL.

Это прямой тест исходной идеи:

> насколько attention-картина candidate похожа на картину успешного извлечения.

## 21.2. AT4-H — direct selected-head retrieval score

Проверить простой score:

```text
aggregated Q→context / Q→predicted-span attention
по selected heads.
```

Ранжировать shortlist напрямую этим score.

Это closest-to-retrieval-head baseline.

## 21.3. AT4-L — linear attention probe

Обучить:

```text
logistic regression
или
linear ranking model
```

на compact attention signature.

Targets отдельно:

```text
target evidence vs target mismatch;
reader success vs failure.
```

Linear probe должен предшествовать MLP.

## 21.4. AT4-M — small nonlinear attention model

Только если AT4-L подтверждает наличие сигнала.

Использовать небольшой MLP / shallow model.

Не вводить большую новую language model.


## 21.5. AT4-D — optional attention distillation / selector alignment

Только если AT4-H/AT4-L показывают устойчивый OOF-сигнал,
проверить дополнительный вариант, уменьшающий runtime cost.

Идея:

```text
frozen reader attention
→ teacher attention-derived target score
→ lightweight selector/student
```

Student получает только:

```text
question + chunk
```

и обучается воспроизводить:

```text
selected-head attention score;
attention-prototype score;
или pairwise ordering, полученный из attention teacher.
```

Цель:

> проверить, можно ли перенести полезный внутренний reader-signal
> в более дешёвый selector и не запускать attention-enabled reader
> для каждого runtime candidate.

Сравнить:

```text
direct attention reranking;
attention-distilled selector;
обычный text target-evidence ranker.
```

Teacher targets строятся только на TRAIN_FOLDS.
Distillation hyperparameters — только CALIBRATION.

Если direct attention не показывает самостоятельного OOF-сигнала,
AT4-D не запускать.

## 21.6. Attention-only и fusion сравнивать раздельно

Обязательные абляции:

```text
FUS-0 = control-only
FUS-1 = attention only
FUS-2 = B0/B1/B3 score only
FUS-3 = B + attention
FUS-4 = best textual target-ranker + attention
```

Если `FUS-4` лучше соответствующих отдельных каналов,
это поддерживает гипотезу комплементарности.

## 21.7. Где применять attention-ranker

Основной практический режим:

```text
all chunks
↓
B0/B1/B3
↓
top-K pre-shortlist
↓
frozen reader + attention extraction
↓
attention reranking
```

Проверить:

```text
K ∈ {3,5,8}
```

Полный attention forward по всем chunks оставить только как diagnostic cost ceiling,
если вычислительная цена существенна.

---

# 22. V3-AT5 — reader-success, span-level attention и causal validation

Attention необходимо проверить не только для target attribution,
но и для качества уже извлечённого candidate span.

## 22.1. AT5-S — attention-enhanced SpanRanker

В будущий SpanRanker добавить отдельную абляцию:

```text
S-text:
question + chunk + candidate span

S-score:
S-text + evidence scores + QA score

S-att:
S-text + attention signature

S-full:
S-text + evidence scores + QA score + attention signature
```

Это проверяет вторую роль attention:

```text
не "правильный ли chunk",
а
"правильный ли extracted candidate span".
```

## 22.2. Hidden-state comparison

Чтобы понять, специфичен ли эффект именно для attention,
добавить сравнительную ветку внутренних признаков reader-а:

```text
pooled question hidden state;
pooled predicted-span hidden state;
Q/span cosine;
elementwise |Q-S|;
elementwise Q*S;
layer summaries.
```

Сравнить:

```text
attention only;
hidden states only;
attention + hidden states.
```

Не сохранять огромные hidden tensors без необходимости — только compact pooled features.

## 22.3. Attention × value norm — optional

Raw attention weight не учитывает величину передаваемого value representation.

Если текущая реализация позволяет получить это без изменения reader-а,
разрешено проверить дополнительный признак вида:

```text
attention_weight × value_vector_norm
```

как отдельную абляцию.

Не делать обязательным компонентом первой итерации.

## 22.4. Gradient × attention — diagnostic high-cost branch

Можно проверить:

```text
gradient of predicted span score
×
attention
```

только как secondary diagnostic.

Причины:

```text
нужен backward pass;
выше стоимость;
хуже подходит для дешёвого runtime filtering.
```

Если он существенно не превосходит forward-only attention signature,
не включать в production candidate.

## 22.5. Causal head masking

Если AT3 находит устойчивый небольшой набор heads,
выполнить diagnostic causal ablation на unlocked folds:

```text
normal reader;
mask selected heads;
mask такое же число случайных heads;
при возможности mask matched heads из тех же layers.
```

Сравнить:

```text
reader F1 на target-positive;
target-evidence ranking quality;
attention-score separation;
QA EM.
```

Selected heads определяются только на TRAIN/CALIBRATION соответствующего fold.

Random masking повторить несколькими random sets.

Если masking selected heads ухудшает качество сильнее random masking,
это усиливает аргумент о функциональной значимости heads.

## 22.6. Causal ablation не является финальным методом

Head masking — только проверка гипотезы.

Нельзя:

```text
смотреть INTERNAL_TEST;
выбирать heads по test masking effect;
модифицировать финальный reader после test.
```

---


# 23. Эксперимент V3-R — reader-success meta-ranker

Проводить только после анализа positive-quality.

## 23.1. Train target

Для TRAIN_FOLD можно построить training examples:

```text
(question, chunk, frozen reader prediction)
```

и target:

```text
reader_success
```

например:

```text
1, если reader F1 >= 50
0, если reader F1 = 0
ignore/soft target для промежуточных случаев
```

Важно:

gold используется только при создании TRAIN labels.

На inference:

```text
gold недоступен.
```

## 23.2. Вход meta-ranker

Минимум:

```text
question
chunk text
reader predicted span
локальное окружение predicted span
B-target-evidence score
```

Не использовать только raw QA score.

## 23.3. Возможные режимы

### R-a

Ранжировать chunks до финального QA.

### R-b

Ранжировать уже:

```text
(chunk, predicted span)
```

Это предпочтительнее и ближе к следующей фазе span reranking.

---

# 24. Отказ от raw multi-passage C

В V3 не продолжать оптимизацию:

```text
M
+
max raw start/end logits
```

как основную систему.

Сохранить текущий C только как baseline:

```text
C_rawlogit
```

Новые multi-candidate эксперименты строить на **answer candidate reranking**.

---

# 25. Эксперимент V3-S — span-level candidate reranking

Это второй основной новый архитектурный эксперимент после target-evidence ranking.

## 25.1. Candidate generation

Взять top-K от лучшего evidence ranker.

Проверить:

```text
K = 2, 3, 5
```

Frozen reader независимо получает:

```text
answer_i
start_i
end_i
raw_span_score_i
```

для каждого candidate chunk.

## 25.2. Не выбирать max raw QA score

Вместо этого построить:

```text
SpanRanker(question, chunk, predicted_span, local_context)
```

## 25.3. Train labels

На TRAIN_FOLDS label должен учитывать **и строку, и источник evidence**.

Положительный binary label:

```text
span_correct_target = 1
```

только если одновременно:

```text
1. source chunk ∈ P_v3_target;
2. extracted span соответствует одному из допустимых gold answers
   по заранее выбранному правилу (например EM для S1).
```

Если строка совпала с gold, но source chunk является:

```text
verified target_mismatch
или
другим заведомо нецелевым source
```

это **не correct span**:

```text
span_correct_target = 0
wrong_target = 1
```

`partial_support` и `unresolved` по умолчанию исключать из binary SpanRanker training,
если для них невозможно надёжно определить attribution.

Сохранять раздельно:

```text
raw_EM
raw_TokenF1
source_target_status
attribution_aware_EM
attribution_aware_TokenF1
wrong_target
```

Определить:

```text
attribution_aware_TokenF1 =
    raw_TokenF1, если source ∈ P_v3_target
    0,            если source = verified target_mismatch / known non-target
```

Это не позволяет высокой строковой похожести из неправильного source
ошибочно стать положительным training signal.


Можно проверить две постановки:

### S1 — binary span correctness

```text
correct / incorrect
```

### S2 — regression / ordinal

Предсказывать качество:

```text
attribution_aware_TokenF1
```

`raw_TokenF1` хранить только как дополнительную диагностическую цель.

или интервальный класс:

```text
0
(0,50)
[50,80)
>=80
```

Начать с S1.

## 25.4. Features / input

Основной вариант должен быть текстовым Cross-Encoder input, например:

```text
[QUESTION]
...
[CHUNK]
...
[CANDIDATE ANSWER]
...
```

Можно добавить:

```text
evidence ranking score
B0/B1/B3 score
raw QA score
QA margin
chunk rank
```

только после калибровки/нормализации.

---

# 26. Span-ranker evaluation

Сравнить:

```text
top-1 evidence → QA

top-K evidence → max raw QA score

top-K evidence → SpanRanker
```

Главные метрики:

```text
Token F1
EM
WrongTargetAnswerRate
F1 | target evidence present in top-K
```

Отдельно считать два oracle:

```text
raw best-string candidate span among generated top-K
attribution-aware best candidate span among generated top-K
```

Основным для V3 считать **attribution-aware oracle**:
candidate из verified non-target source не может быть oracle-positive
только потому, что его строка совпала с gold.

Это показывает реальный потолок span reranking при фиксированном candidate generation.

---

# 27. D4 sentence-level evidence — только через gate

D4 не запускать автоматически.

## Gate D4

Переходить к D4, только если positive-quality analysis показывает:

```text
1. target-positive выбран правильно;
2. reader ошибается;
3. ошибки связаны с competing entities / длинным контекстом / несколькими локальными фактами;
4. gold-support sentence можно локализовать.
```

Если основная проблема остаётся target attribution — D4 не приоритет.

## 27.1. Если gate пройден

Сравнить:

```text
full positive chunk
gold-independent selected sentence
selected sentence ± 1 neighbor
```

Sentence selector обучать только на TRAIN_FOLDS.

Сохранять source offset mapping.

---

# 28. D5 answer aggregation — только через gate

## Gate D5

До реализации посчитать:

```text
сколько document–aspect случаев
имеют >=2 независимых source spans,
которые приводят к одному корректному answer value.
```

Если таких случаев мало:

```text
D5 отложить.
```

Если достаточно:

реализовать exact-normalized answer aggregation.

Не начинать с semantic clustering.

---

# 29. D3 graph expansion остаётся deferred

Не включать в основной V3.

Возвращаться только если error analysis показывает:

```text
правильный answer требует контекста из нескольких соседних chunks,
и target evidence невозможно определить локально.
```

---

# 30. Новое правило model selection

V2 показал, что:

```text
pairwise_clean_hn как первый lexicographic criterion
может выбрать систему с худшим downstream QA.
```

Поэтому V3 не использовать:

```text
(pairwise_hn, Hit@1, F1|pos, F1)
```

как жёсткий lexicographic key.

## 30.1. Multi-objective constrained selection

Reference:

```text
V3-B0
```

Для candidate configuration считать:

```text
ΔTokenF1
ΔEM
ΔHit@1_original
ΔTargetEvidenceHit@1_v3
ΔHN@1
ΔAmbiguous@1_original
ΔGSRN@1_original
ΔTargetMismatch@1, только при достаточной target-status coverage
ΔPairwiseHN
ΔPairwiseTM, только на adjudicated pairs
```

## 30.2. Ranking safeguards

Hard safeguards по умолчанию строить на наиболее надёжных исходных HN-показателях.

До эксперимента зафиксировать допустимую деградацию:

```text
epsilon_HN
epsilon_pairwise_HN
```

`Ambiguous@1_original` и `GSRN@1_original` использовать как diagnostics,
а не автоматические hard constraints,
поскольку V3 как раз проверяет наличие false negatives внутри ambiguous.

Например концептуально:

```text
HN@1_candidate <= HN@1_B0 + epsilon_HN
pairwise_HN_candidate >= pairwise_HN_B0 - epsilon_pairwise_HN
```

Конкретные epsilon определить **до анализа EVAL folds**, используя:

```text
application needs
CALIBRATION variation
V2 variability
```

## 30.3. Среди допустимых систем

Основным критерием выбрать:

```text
document–aspect Token F1
```

с обязательным анализом:

```text
Hit@1_original
TargetEvidenceHit@1_v3
HN@1
Ambiguous@1_original
GSRN@1_original
TargetMismatch metrics + coverage
WrongTargetAnswerRate + coverage
```

## 30.4. Pareto frontier

Дополнительно вывести системы, которые не доминируются одновременно по:

```text
Token F1 ↑
Hit@1_original ↑
TargetEvidenceHit@1_v3 ↑
HN@1 ↓
GSRN@1_original ↓
TargetMismatch ↓ при достаточной coverage
cost ↓
```

---


## 30.5. Отдельные правила для attention-based вариантов

Attention-based candidate не должен выигрывать только потому, что перебрано много heads/features.

Поэтому:

1. family признаков фиксируется до EVAL;
2. heads выбираются на TRAIN;
3. `N_heads`, thresholds, fusion и shortlist K выбираются на CALIBRATION;
4. EVAL используется только для OOF оценки;
5. после просмотра EVAL нельзя добавлять новый attention feature в тот же comparison run.

Обязательные сравнения:

```text
B0;
best textual target-evidence ranker;
attention-only;
selected-head score;
prototype-distance;
linear attention probe;
text + attention;
best span-ranker;
span-ranker + attention.
```

Primary критерий остаётся document–aspect QA,
а attention-specific ranking metrics являются диагностическими.


# 31. Статистическая проверка V3

Основной статистический объект:

```text
out-of-fold prediction per document–aspect.
```

После 5 fold получить единый набор OOF predictions.

## 31.1. Clustered bootstrap

Ресэмплировать по:

```text
document_id
```

не по отдельным вопросам.

Минимум:

```text
5000 bootstrap replications
```

## 31.2. Основные пары

```text
B0 vs B1
B0 vs B2
B0 vs B3
best evidence ranker vs evidence + structure
best evidence ranker vs span-ranker
raw-logit multi-passage vs span-ranker
pre-D4 vs D4, если D4 запускался
pre-D5 vs D5, если D5 запускался
```


Дополнительно для attention-ветки:

```text
all-head mean vs selected-head score
B0 vs attention-only
best textual target-ranker vs text+attention
prototype-distance vs linear attention probe
SpanRanker vs SpanRanker+attention
selected-head mask vs matched random-head mask
```

Для causal masking random-head comparison выполнять повторно
для нескольких random masks и сохранять распределение эффекта.


## 31.3. Не использовать seed как независимую репликацию вопросов

Если несколько seeds:

- усреднять predictions/metrics по seeds на уровне fold;
- либо строить hierarchical summary;
- не считать `question × seed` независимыми наблюдениями.


## 31.4. OOF statistics — model-selection evidence, не финальный confirmatory test

Поскольку на OOF results выбирается архитектура V3,
bootstrap CI по этим же OOF predictions нельзя выдавать за полностью независимую
подтверждающую оценку уже выбранного метода.

OOF использовать для:

```text
model selection;
effect-size estimation;
robustness analysis;
отсечения явно неработающих гипотез.
```

Финальное confirmatory comparison выполняется только на INTERNAL_TEST.

## 31.5. Ограничить multiplicity

До final refit заранее определить:

```text
1 primary final comparison
final V3 pipeline vs final V3-B0

1 primary endpoint
document–aspect Token F1
```

Дополнительные:

```text
EM
Hit@1_original
HN@1
Ambiguous@1_original
GSRN@1_original
cost
```

считать secondary endpoints.

Если по secondary endpoints выполняются формальные multiple hypothesis tests,
использовать заранее выбранную correction procedure (например Holm).
Иначе представлять effect size + CI как descriptive secondary analysis.

С учётом малого INTERNAL_TEST:

```text
31 document clusters
54 answerable questions
```

не сводить интерпретацию только к `p < 0.05`;
обязательно приводить paired effect size и широкий/узкий CI.

---

# 32. Несколько random seeds

Trainable components:

```text
B0/B1/B2/B3
TargetEvidenceCE
attention linear/nonlinear probe, если обучаемый
attention-enhanced target ranker
reader-success meta-ranker
SpanRanker
attention-enhanced SpanRanker
D4, если используется
```

проверять минимум на:

```text
3 seeds.
```

Если computational budget позволяет:

```text
5 seeds
```

для финальных 1–2 candidate architectures.

---

# 33. Анализ по аспектам

Обязательно отдельно:

```text
task
method
material
metric
```

Потому что V2 показал разные типы ограничений.

## 33.1. task

Reader уже силён при правильном positive.

Основной вопрос:

```text
может ли target-evidence ranking повысить Hit@1?
```

## 33.2. metric

Даже oracle reader заметно слабее.

Отдельно анализировать:

```text
reader span boundaries
несколько метрик в одном контексте
benchmark vs target metric
related-work metrics
```

Не делать общий вывод по macro-average без aspect analysis.

---

# 34. Анализ ошибок V3

Расширить taxonomy.

```text
E1  no target-positive in top-K
E2  hard negative outranks positive
E3  target-mismatch ambiguous outranks positive
E4  GSRN outranks positive
E5  other ambiguous outranks positive
E6  easy negative selected
E7  correct positive selected, wrong reader span
E8  correct positive exists but worse positive selected
E9  wrong target answer extracted
E10 correct candidate span generated but SpanRanker selected wrong one
E11 target evidence lost by shortlist
E12 sentence selector removed required context
E13 aggregation merged different values
E14 duplicate overlap counted as independent evidence
E15 attention signature strongly favors target-mismatch
E16 selected attention heads fail to transfer across fold
E17 attention-ranker removes true target evidence from shortlist
E18 attention span-confidence favors wrong extracted span
E19 attention feature helps ranking but adds unacceptable runtime/memory cost
```

Для каждого метода считать распределение ошибок.

---

# 35. Ресурсные метрики

Для каждого живого pipeline:

```text
mean latency / document–aspect
p50
p95
peak GPU
peak RAM
candidate chunks scored
QA forward passes
reader tokens
target-ranker calls
span-ranker calls
attention-enabled reader calls
attention signature extraction time
attention cache size
selected-head count
hidden-state feature extraction time, если используется
gradient-based feature time, если используется
```

Особенно сравнить:

```text
B0
B3 listwise
two-stage B4
SpanRanker
```

---

# 36. Порядок выполнения V3

Порядок должен позволить одновременно проверить исходные V3-гипотезы
и новую attention-based ветку, не смешивая их эффекты.

## Этап 0 — freeze V2

Сохранить V2 без изменений.

## Этап 1 — создать V3 grouped CV

```text
268 unlocked docs
5 folds
INTERNAL_TEST untouched
```

## Этап 2 — расширить ambiguous audit

Создать:

```text
target_match_status
verified_target_mismatch
verified_target_evidence
partial_support
unresolved
```

## Этап 3 — реализовать новые target-attribution метрики

```text
Ambiguous@1
TargetMismatch@1
GSRN@1
Pairwise Pos>Amb
Pairwise Pos>TM
WrongTargetAnswerRate
TargetStatusCoverage@K
```

## Этап 4 — V3-B0

Переобучить B2 baseline внутри каждого fold.

## Этап 5 — исходная target-evidence ветка

Последовательно проверить:

```text
V3-B1 = BCE + verified target mismatch
V3-B2 = pairwise target-evidence ranking
V3-B3 = listwise/document-level ranking
V3-B4 = two-stage answerability → target evidence, если оправдано
```

Нельзя пропускать baseline B0.

## Этап 6 — structural attribution context

Проверить richer local structural/discourse context
поверх зафиксированного evidence-ranking baseline.

Старый section-overlap не считать основной структурной гипотезой.

## Этап 7 — positive-quality analysis

Получить reader-easy / reader-hard / reader-failed positives.

Это требуется одновременно для:

```text
reader-success meta-ranker;
attention audit;
span-level ranking.
```

## Этап 8 — V3-AT0 instrumentation

Проверить:

```text
attention extraction;
logit regression;
token masks;
offsets;
runtime/memory.
```

До прохождения AT0 attention-results не считать научными.

## Этап 9 — V3-AT1 attention audit

На TRAIN/CALIBRATION сравнить:

```text
target+reader-success
target+reader-failure
target-mismatch
HN
easy
```

Отдельно:

```text
predicted-span attention
vs
gold-span attention diagnostic-only.
```

## Этап 10 — V3-AT2 attention signatures

Реализовать компактные признаки:

```text
Q→span
span→Q
entropy/concentration
competing-region margin
head agreement
layer dynamics
head dispersion
```

## Этап 11 — V3-AT3 head selection

Сравнить:

```text
all-head mean
last-layer mean
middle-layer mean
selected heads
```

Выбор heads только TRAIN,
размер и thresholds только CALIBRATION.

## Этап 12 — V3-AT4 attention ranking

Проверить минимум:

```text
prototype distance
selected-head direct score
linear attention probe
small nonlinear probe, только если linear подтверждён
attention only
B/text ranker + attention
attention distillation, только если direct attention подтвердился
```

## Этап 13 — reader-success meta-ranker V3-R

Проверить исходную reader-success идею
отдельно от attention и затем с attention features.

## Этап 14 — span candidate generation

Top-K лучшего evidence ranker:

```text
text-only candidate
attention-enhanced candidate
```

если attention-ветка прошла OOF проверку.

## Этап 15 — V3-S SpanRanker

Сравнить:

```text
top-1 evidence → QA
raw-logit multi-passage baseline
SpanRanker text-only
SpanRanker + evidence scores
SpanRanker + attention
SpanRanker + attention + hidden-state features
```

Gradient×attention оставить secondary diagnostic.

## Этап 16 — V3-AT5 causal head ablation

Только если есть устойчивые selected heads.

Сравнить:

```text
selected-head masking
vs
matched random-head masking.
```

Не выполнять на INTERNAL_TEST до финального freeze.

## Этап 17 — D4 gate

Решить, нужен ли sentence-level evidence.

D4 разрешён только если positive-reader error analysis подтверждает
локальный контекстный шум как существенную причину.

## Этап 18 — D5 gate

Решить, нужна ли aggregation.

## Этап 19 — D3 gate

Решить, нужен ли graph/inter-chunk expansion.

Default:

```text
deferred.
```

## Этап 20 — grouped CV final comparison

Получить единые OOF predictions для всех candidate families:

```text
B0
best B1/B2/B3/B4
best structural target-ranker
best attention-only
best text+attention
best SpanRanker
best SpanRanker+attention
D4/D5, если прошли gates
```

## Этап 21 — clustered bootstrap и Pareto analysis

Проверить:

```text
QA quality
target mismatch
HN
cost
```

## Этап 22 — выбрать финальный pipeline

Все архитектурные и hyperparameter решения принять по OOF/CALIBRATION
до любого доступа к INTERNAL_TEST.

## Этап 23 — зафиксировать final-refit policy

До финального обучения определить, как использовать все 268 unlocked documents.

Trainable components финальной системы переобучить на всех unlocked documents
**только после** freeze архитектуры и hyperparameters.

Если в разных CV-runs CALIBRATION выбрала разные значения одного hyperparameter,
заранее зафиксировать deterministic aggregation rule, например:

```text
categorical parameter → mode across folds;
ordered numeric parameter → median across folds;
continuous fusion weight → значение, максимизирующее заранее определённый
                           aggregate CAL criterion, без доступа к EVAL labels.
```

Нельзя после просмотра INTERNAL_TEST выбирать «тот hyperparameter,
который чаще помогал на похожих test-примерах».
Нельзя делать early stopping по INTERNAL_TEST.

Для epoch-based моделей использовать заранее зафиксированную CV-derived policy, например:

```text
final_epochs = median(best_epoch across CV runs)
```

или другую функцию от CV best epochs, выбранную до refit.

Для thresholds / score calibration использовать:

```text
OOF predictions
или
предварительно зафиксированную cross-fitted calibration procedure
```

а не predictions той же модели на её train data.

Для attention head selection заранее зафиксировать одну policy:

```text
A. rerun frozen head-selection algorithm на всех 268 unlocked docs
   при уже фиксированном N_heads и criterion;

или

B. использовать заранее определённый consensus-head rule по CV folds.
```

Policy A/B выбрать **до** final refit и записать в manifest.
Нельзя вручную подбирать конкретные heads после просмотра итоговых OOF tables.

Обязательно выполнить такой же final refit для reference `V3-B0`,
чтобы final test comparison был честным.

## Этап 24 — final refit

Получить:

```text
final_v3_model
final_b0_reference
final_scalers
final_calibrators
final_attention_head_set, если используется
```

используя только 268 unlocked documents и frozen refit policy.

## Этап 25 — freeze-final

Создать:

```text
final_freeze_manifest_v3.json
```

В manifest дополнительно фиксировать:

```text
attention enabled/disabled
attention feature family
selected layers/heads
head-selection algorithm
head-finalization policy
N_heads
signature definition
feature scaler
shortlist K
prototype/statistical model
hidden-state features
span-ranker features
final epoch policy
calibration policy
exact checkpoints / hashes
```

## Этап 26 — INTERNAL_TEST

Открыть один раз.

Primary final comparison:

```text
final_v3_model
vs
final_b0_reference
```

на заранее зафиксированном `Token F1`.

Другие V3 variants на INTERNAL_TEST не перебирать.

После открытия:

```text
не менять архитектуру;
не менять heads;
не менять attention features;
не менять thresholds;
не добавлять новый fusion;
не выбирать другой seed;
не менять taxonomy.
```

---

# 37. Условия перехода к INTERNAL_TEST

INTERNAL_TEST разрешено открывать только если выполнены все пункты:

- [ ] target-mismatch taxonomy зафиксирована;
- [ ] CV folds зафиксированы;
- [ ] B0 baseline полностью оценён OOF;
- [ ] B1/B2/B3 проверены;
- [ ] решение о B4 принято до TEST;
- [ ] span reranking проверен;
- [ ] attention instrumentation regression пройден;
- [ ] attention audit завершён;
- [ ] решение attention-only yes/no принято;
- [ ] решение text+attention yes/no принято;
- [ ] selected heads / feature family зафиксированы, если используются;
- [ ] решение attention distillation yes/no принято до TEST;
- [ ] attention-enhanced SpanRanker проверен, если attention-путь перспективен;
- [ ] causal head ablation завершён либо заранее признан необязательным;
- [ ] решение D4 yes/no принято;
- [ ] решение D5 yes/no принято;
- [ ] D3 yes/no принято;
- [ ] practical-success criteria зафиксированы;
- [ ] thresholds зафиксированы;
- [ ] model architecture зафиксирована;
- [ ] random seed policy зафиксирована;
- [ ] source provenance проверено;
- [ ] final-refit policy зафиксирована;
- [ ] final V3 и final B0 переобучены по одной заранее заданной схеме;
- [ ] calibration/scaler fit не использует INTERNAL_TEST;
- [ ] exact checkpoints / hashes записаны;
- [ ] final_freeze_manifest_v3.json создан;
- [ ] код final-eval не использует test gold при inference.

---

# 38. Practical-success criterion

До открытия INTERNAL_TEST зафиксировать.

Primary:

```text
Q_primary = Token F1 document–aspect
```

Для OOF model selection secondary safeguards:

```text
HN@1
Ambiguous@1_original
GSRN@1_original
TargetMismatch metrics + coverage
Hit@1_original
TargetEvidenceHit@1_v3
cost
```

Для INTERNAL_TEST ручная V3 target-status переразметка запрещена.

Confirmatory test должен разделять:

### Hard safeguards

Доступны и достаточно однозначны из исходной разметки:

```text
HN@1
Hit@1_original
EM
cost
```

### Descriptive attribution diagnostics

```text
Ambiguous@1_original
GSRN@1_original
GoldMatchFromNonOriginalPositive
```

Эти показатели **не использовать как жёсткое условие провала/успеха**,
потому что исходный ambiguous/GSRN класс сам может содержать false negatives
или корректный evidence, не признанный исходной разметкой.

Рост `Ambiguous@1_original` при одновременном росте Token F1
не следует автоматически интерпретировать как ухудшение.

`TargetMismatch@1` на INTERNAL_TEST не требовать,
если независимая target-status annotation не существовала **до** test unlock.

Определить:

```text
delta_F1_min
epsilon_HN_max
runtime_ratio_max
memory_ratio_max
```

Для OOF при достаточной annotation coverage дополнительно:

```text
epsilon_TM_max
```

Успех на INTERNAL_TEST:

```text
Token F1 улучшен практически значимо
+
paired 95% CI поддерживает направление эффекта
+
HN@1 не деградирует выше заранее заданного допустимого уровня
+
стоимость не выходит за заранее заданный предел.
```

`Ambiguous@1_original`, `GSRN@1_original`,
`GoldMatchFromNonOriginalPositive` и EM интерпретировать совместно
как secondary diagnostics.

TargetMismatch improvement подтверждается на OOF/adjudicated unlocked data,
но не изображается как измеренный INTERNAL_TEST эффект при отсутствии test target-status labels.


---

# 39. Что нельзя делать в V3

Запрещено:

1. Открывать INTERNAL_TEST до freeze.
2. Использовать старый DEVELOPMENT как единственный новый DEV.
3. Переразмечать INTERNAL_TEST.
4. Считать все ambiguous negatives.
5. Добавлять unresolved ambiguous в strong-negative train.
6. Выбирать модель только по pairwise Pos>HN.
7. Выбирать модель только по Token F1 без HN/TM safeguards.
8. Продолжать raw-logit C как основной multi-passage механизм.
9. Использовать raw QA score как доказанную extractability probability.
10. Возвращать BM25/RRF в default pipeline без нового подтверждения.
11. Считать section-title overlap полноценной document structure model.
12. Запускать D3 автоматически.
13. Запускать D4 до анализа positive-reader failures.
14. Запускать D5 до оценки частоты независимых подтверждений.
15. Считать R3_3 уже доказанным улучшением V2.
16. Использовать post-hoc наблюдения DEVELOPMENT как финальный результат.
17. Менять pipeline после открытия INTERNAL_TEST.
18. Выбирать attention heads по EVAL_FOLD или INTERNAL_TEST.
19. Использовать gold-span attention как runtime feature.
20. Усреднять все heads и без абляции объявлять это «attention signal».
21. Интерпретировать raw attention как причинное объяснение без causal test.
22. Хранить полный attention tensor для всего корпуса без необходимости.
23. Добавлять новые attention features после просмотра OOF-EVAL и сравнивать их в том же протоколе.
24. Смешивать эффект target-evidence supervision и attention без отдельных text-only/attention-only абляций.
25. Переносить выводы о retrieval heads из decoder LLM на RoBERTa без собственной проверки.
26. Использовать attention-distilled teacher targets, рассчитанные по EVAL_FOLD.
27. Переписывать исходный `source_label` новой V3 target-разметкой.
28. Считать непроверенный ambiguous автоматически `not target_mismatch`.
29. Обучать SpanRanker считать gold-string из verified target-mismatch правильным ответом.
30. Настраивать final epoch / calibration / heads по INTERNAL_TEST.
31. Выдавать OOF bootstrap CI после model selection за полностью независимый confirmatory test.

---

# 40. Предпочтительная итоговая архитектура, которую должен проверить V3

Это не заранее выбранная победившая система.

V3 должен сравнить **две параллельные ветки evidence ranking** и их объединение.

```text
DOCUMENT
   ↓
all chunks
   ↓
B0 answerability
   ↓
────────────────────────────────────────────
│                                          │
│ text target-evidence branch              │ internal-reader branch
│ B1/B2/B3/B4 + discourse context          │ frozen reader on shortlist
│                                          │ ↓
│                                          │ attention signature
│                                          │ selected heads
│                                          │ prototype / probe
│                                          │
└───────────────────┬──────────────────────┘
                    ↓
          text-only / attention-only /
          text+attention comparison
                    ↓
          top-K target candidates
                    ↓
       frozen extractive reader
       для каждого candidate
                    ↓
          candidate answer spans
                    ↓
     SpanRanker text-only / +attention /
     +hidden-state internal features
                    ↓
       optional D4 / D5 only by gates
                    ↓
        FINAL DOCUMENT–ASPECT ANSWER
```

Минимально должны существовать независимо оцениваемые системы:

```text
P0 = B0
P1 = best textual target-evidence ranker
P2 = attention-only evidence ranker
P3 = text + attention evidence ranker
P4 = P1 + SpanRanker
P5 = P3 + attention-enhanced SpanRanker
```

Если attention не даёт OOF-прироста или слишком дорог,
финальная система может законно остановиться на P1/P4.

Если attention на grouped-CV OOF повышает QA и улучшает target-attribution metrics
при достаточной annotation coverage и приемлемой стоимости,
он становится кандидатом на включение в финальный метод.

---

# 41. Критерий научного результата V3

V3 считается содержательным даже если итоговый F1 не вырастет, если эксперимент надёжно установит одно из следующего:

1. target-mismatch ambiguous действительно является отдельным типом ошибки, не сводимым к HN;
2. target-aware supervision устойчиво снижает verified target-mismatch errors при достаточной annotation coverage;
3. listwise document-level objective лучше independent BCE;
4. правильный evidence ranking не гарантирует правильный answer candidate;
5. span-level reranking превосходит raw-logit multi-passage;
6. sentence-level evidence необходимо только для определённых аспектов/типов ошибок;
7. дальнейший рост ограничен reader-ом для отдельных аспектов, особенно metric;
8. attention signatures действительно различают successful target evidence и target mismatch;
9. небольшой устойчивый набор attention heads лучше среднего по всем heads;
10. attention-derived признаки дают дополнительный сигнал поверх текстового target-evidence ranker;
11. attention полезен только для span correctness, но не для chunk attribution — или наоборот;
12. causal masking подтверждает либо опровергает функциональную роль selected heads.

Главный ожидаемый научный переход:

```text
semantic relevance
    ↓
answerability
    ↓
target evidence attribution
    ↓
internal reader signals / attention
    ↓
answer candidate ranking
```

Именно эта последовательность наиболее полно соответствует наблюдениям V2.

# 42. Научные источники для attention-based ветки

Эти работы используются как научное основание и ориентир.
V3 не является их буквальной репликацией.


Важно: часть работ ниже исследует decoder-only long-context LLM,
а текущий V3 reader — encoder-only RoBERTa extractive QA.
Поэтому из литературы переносится **исследовательская гипотеза и тип сигнала**,
но не утверждение о существовании тех же layer/head mechanisms в RoBERTa.
Это проверяется экспериментально в AT1–AT5.

## [AT-S1] FAA — fine-grained attention feedback для passage selection

**Zhen Li, Chongyang Tao, Jiazhan Feng, Tao Shen, Dongyan Zhao, Xiubo Geng, Daxin Jiang.**
2023. *FAA: Fine-grained Attention Alignment for Cascade Document Ranking*. pp. 1688–1700.
На момент поиска в Consensus: 5 цитирований.

Источник:
https://consensus.app/papers/faa-finegrained-attention-alignment-for-cascade-document-li-tao/9cdd0a089367565490eb5bcd087b160f/?utm_source=chatgpt

Использование в V3:

```text
attention activations сильной модели как feedback/признак для selector/ranker;
обоснование AT4 fusion.
```

## [AT-S2] Retrieval Heads — причинно значимые attention heads

**Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, Yao Fu.**
2024. *Retrieval Head Mechanistically Explains Long-Context Factuality*.
ArXiv, abs/2404.15574.
На момент поиска: 232 цитирования.

Источник:
https://consensus.app/papers/retrieval-head-mechanistically-explains-longcontext-wu-wang/e3b25e0485d55e448397cf7c5f6a0ee6/?utm_source=chatgpt

Использование:

```text
обоснование specialized-head hypothesis;
обоснование causal masking selected heads vs random heads.
```

## [AT-S3] Query-Focused Retrieval Heads

**Wu Zhang, Fangcong Yin, Howard Yen, Danqi Chen, Xi Ye.**
2025. *Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking*.
ArXiv, abs/2506.09944.
На момент поиска: 33 цитирования.

Источник:
https://consensus.app/papers/queryfocused-retrieval-heads-improve-longcontext-zhang-yin/5adf92a46d3951d49a6684d8bb5b35f8/?utm_source=chatgpt

Использование:

```text
выбор heads по query-context attention;
attention mass selected heads как retrieval/reranking score;
AT3 и AT4-H.
```

## [AT-S4] Attention-score listwise reranking

**Yuqing Li, Jiangnan Li, Mo Yu, Guoxuan Ding, Zheng Lin, Weiping Wang, Jie Zhou.**
2026. *Query-focused and Memory-aware Reranker for Long Context Processing*.
ArXiv, abs/2602.12192.
На момент поиска: 6 цитирований.

Источник:
https://consensus.app/papers/queryfocused-and-memoryaware-reranker-for-long-context-li-li/d47ebe109c2852e696654805f7d25b84/?utm_source=chatgpt

Использование:

```text
attention scores selected heads как ranking signal;
listwise интерпретация candidate ranking;
роль middle-layer heads.
```

## [AT-S5] MuDAF — contrastive focusing of attention heads

**Weihao Liu, Ning Wu, Shiping Yang, Wenbiao Ding, Shining Liang, Ming Gong, Dongmei Zhang.**
2025. *MuDAF: Long-Context Multi-Document Attention Focusing through Contrastive Learning on Attention Heads*.
ArXiv, abs/2502.13963.
На момент поиска: 3 цитирования.

Источник:
https://consensus.app/papers/mudaf-longcontext-multidocument-attention-focusing-liu-wu/6127926a63945272bc8cdd271ff2b2c5/?utm_source=chatgpt

Использование:

```text
обоснование contrastive анализа attention heads;
идея focused vs distracted attention.
```

## [AT-S6] Reader-centered passage selection

**Chunlei Xin, Shuheng Zhou, Xuanang Chen, Yaojie Lu, Huijia Zhu, Weiqiang Wang,
Zhongyi Liu, Xianpei Han, Le Sun.**
2025. *Aligning Retrieval with Reader Needs: Reader-Centered Passage Selection for Open-Domain Question Answering*.
pp. 1000–1012.
На момент поиска: 4 цитирования.

Источник:
https://consensus.app/papers/aligning-retrieval-with-reader-needs-readercentered-xin-zhou/3b8526bd25c351369b9fb825bb91a7d7/?utm_source=chatgpt

Использование:

```text
общая идея reader-derived сигналов для passage selection;
обоснование сравнения attention с prediction-distribution/reader-confidence signals.
```

## [AT-S7] Attention-guided extractive QA

**Peng Xu, Davis Liang, Zhiheng Huang, Bing Xiang.**
2021. *Attention-guided Generative Models for Extractive Question Answering*.
ArXiv, abs/2110.06393.
На момент поиска: 17 цитирований.

Источник:
https://consensus.app/papers/attentionguided-generative-models-for-extractive-xu-liang/e8555d2de4605dd19e39bdd9d6622ffb/?utm_source=chatgpt

Использование:

```text
attention pattern как сигнал локализации extractive span;
связь attention-derived reader signal с reranking.
```

## [AT-S8] Ограничение интерпретации attention

**Michael Neely, Stefan F. Schouten, Maurits J. R. Bleeker, Ana Lucic.**
2022. *A Song of (Dis)agreement: Evaluating the Evaluation of Explainable Artificial Intelligence in Natural Language Processing*.
pp. 60–78.
На момент поиска: 22 цитирования.

Источник:
https://consensus.app/papers/a-song-of-disagreement-evaluating-the-evaluation-of-neely-schouten/f0ac1b95d94b5432aa866478f789627f/?utm_source=chatgpt

Использование:

```text
запрет трактовать raw attention как автоматически faithful explanation;
необходимость empirical и causal validation.
```

---

# 43. Итоговая логика V3 после расширения attention-веткой

Эксперимент должен одновременно ответить на четыре последовательных вопроса:

```text
1. Может ли специальная supervision по target-mismatch
   улучшить отбор evidence по сравнению с B0?

2. Лучше ли pairwise/listwise document-level target ranking
   независимой binary answerability?

3. Содержит ли внутренняя attention-картина frozen reader
   дополнительный сигнал о target attribution и/или reader success?

4. Если несколько candidate spans уже извлечены,
   можно ли надежнее выбрать финальный ответ через SpanRanker,
   в том числе с attention-derived признаками,
   чем через raw QA logits?
```

Таким образом обязательны обе линии:

```text
исходная V3:
target-mismatch supervision
→ pairwise/listwise ranking
→ structural attribution
→ reader-success/span ranking

новая V3-AT:
attention audit
→ head selection
→ attention signatures
→ prototype/head/probe ranking
→ text+attention fusion
→ attention-enhanced span ranking
→ causal head ablation
```

Ни одна из них не должна быть объявлена основной заранее.
Победившая конфигурация определяется только по grouped-CV OOF результатам,
многокритериальным safeguards и стоимости — до открытия INTERNAL_TEST.
