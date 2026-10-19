# V10 / SRR — Switch–Rank–Risk
## Детальный протокол эксперимента для ИИ-агента, реализующего код

**Статус:** новый development-эксперимент после V9 / H2-ACS.  
**Цель:** проверить, является ли главным bottleneck текущего метода именно механизм финального выбора после QA.  
**Основная идея:** вместо регрессии точного `ΔTokenF1` раздельно решать задачи `SWITCH → RANK → ANSWER-RISK + EVIDENCE-RISK`.

**Важно:** V10 не добавляет новый нейросетевой `Q+A+chunk support`-модуль. Сначала проверяется сама новая постановка принятия решения на frozen H2 runtime features. Новый evidence-support signal допускается только как отдельная следующая версия после завершения V10.

**Reviewed revision:** явно обработаны вопросы с `K=1`; pairwise inference сделан антисимметричным; добавлены feasibility/class-count checks для обучаемых модулей; уточнена семантика `EVIDENCE-RISK`; диагностические ablations получают собственную TRAIN-only threshold calibration; зафиксирована seed-level training semantics и требования к confirmation data.

---

# 0. Рабочее название

```text
V10 / SRR
Switch–Rank–Risk
```

Русская формулировка:

> двухэтапный выбор ответа с относительным ранжированием кандидатов и раздельным контролем риска ответа и качества evidence.

---

# 1. Основание для V10

V9 показал, что на большинстве datasets хороший evidence и хороший QA-answer уже находятся внутри top-K, но H2 редко использует этот потенциал.

Диагностические ориентиры:

```text
SciREX:
Hit@K ≈ 79.0%
QA-oracle ≈ 60.4 F1
H2 ≈ 33.3

SciER 384/50:
Hit@K ≈ 93.0%
QA-oracle ≈ 87.7
H2 ≈ 58.8

SciER 128/24:
Hit@K ≈ 94.9%
QA-oracle ≈ 89.5
H2 ≈ 58.3

RuSERRC:
Hit@K ≈ 99.1%
QA-oracle ≈ 88.1
H2 ≈ 61.7

SciMDIX:
Hit@K ≈ 92.3%
QA-oracle ≈ 91.7
H2 ≈ 60.1
```

---

# 2. Главный bottleneck

Для SciER и RU основной разрыв:

```text
top-K already contains a useful answer
        ↓
current policy fails to select it.
```

Positive-candidate gate recall в V9 был особенно низок:

```text
SciER384 ≈ 7.6%
RuSERRC ≈ 5.0%
SciMDIX ≈ 25.7%
```

---

# 3. Почему H2 utility regression меняется

H2 пытается предсказывать:

\[
u_i = TokenF1(A_i,G)-TokenF1(A_0,G)
\]

как непрерывную величину.

Utility имеет:

```text
много нулей
крупные отрицательные значения
редкие крупные положительные значения
```

и плохо подходит простой Ridge regression.

---

# 4. Вторая проблема H2

Один score одновременно решает:

```text
A. стоит ли покидать B0;
B. какой candidate выбрать.
```

V10 разделяет эти задачи.

---

# 5. Третья проблема H2

Текущий harm target:

```text
utility < 0
```

описывает только:

```text
answer-quality harm.
```

Но protocol safety отдельно контролирует:

```text
evidence / HN harm.
```

Это разные события.

---

# 6. Четвёртая проблема H2

Система сильно привязана к исходному B0 top-1.

V10 должен сравнивать top-K candidate answers:

```text
относительно друг друга
```

а не только оценивать каждый candidate как correction B0.

---

# 7. Primary research question

> Может ли декомпозиция финального решения на `SWITCH → PAIRWISE RANK → ANSWER-RISK + EVIDENCE-RISK` существенно увеличить использование уже доступного top-K QA-потенциала при сохранении evidence safety?

---

# 8. Что V10 НЕ меняет

Не менять:

```text
documents
questions
chunking
candidate universe
B0
K policy
QA reader
answer normalization
Token F1 evaluator
outer folds
seeds
```

---

# 9. Что V10 НЕ добавляет

Не добавлять:

```text
Agreement
новый PairLEX
SEM
новый encoder
новый reranker
новый QA
Q+A+chunk neural support
aspect-specific rules
K-specific if rules
language-specific rules
dataset-specific rules
```

---

# 10. Почему Agreement исключён

V9 уже отдельно сравнил:

```text
H2
H2 + Agreement
```

Одновременное изменение decision architecture и добавление Agreement смешало бы две гипотезы.

---

# 11. Почему PairLEX не добавляется

Exact H2 уже содержит PairLEX-derived features:

```text
pairlex
pairlex_x_margin
qarel_x_pairlex
```

Не объявлять PairLEX новым компонентом V10.

---

# 12. Архитектура V10

```text
DOCUMENT
   ↓
B0
   ↓
top-K
   ↓
frozen QA(top-K)
   ↓
exact H2 runtime features
   ↓
SWITCH
"есть ли смысл уходить от B0?"
   ↓ yes
PAIRWISE RANK
"какой alternative лучше?"
   ↓
ANSWER-RISK
+
EVIDENCE-RISK
   ↓
highest-ranked safe candidate
   ↓
final answer
```

---

# 13. Fallback

Если:

```text
SWITCH = no
```

или ни одна alternative не проходит оба risk filters:

```text
return B0.
```

---

# 14. Системы

```text
S0 = canonical B0

S1 = exact frozen H2

S2 = V10-SRR
```

`S2` — primary new system.

---

# 15. Development status

Все текущие пять cells уже использовались в V4.4–V9.

Следовательно:

```text
V10 = development evidence
```

и не является independent confirmation.

---

# 16. Dataset cells

```text
SciREX
SciER 384/50
SciER 128/24
RuSERRC
SciMDIX
```

---

# 17. SciER family

SciER128 и SciER384:

```text
одна corpus family
```

и должны учитываться совместно в family-balanced statistics.

---

# 18. Historical protection

Read-only:

```text
V1–V9
```

включая:

```text
code
configs
artifacts
reports
predictions
caches
frozen contracts
```

---

# 19. Новый namespace

Создать:

```text
experiments/qa_chunk_selection_v10/
configs/qa_chunk_selection/v10_srr/
artifacts/qa_chunk_selection_v10_srr/
logs/qa_chunk_selection_v10/
```

---

# 20. Write guard

Любая запись в historical paths:

```text
raise RuntimeError
```

---

# 21. Git и hash audit

До run сохранить:

```text
git status --porcelain
git rev-parse HEAD
git diff
historical_hashes_before.json
```

После run:

```text
git status --porcelain
git diff
historical_hashes_after.json
```

Требование:

```text
historical hashes BEFORE == AFTER.
```

---

# 22. Outer CV

Использовать exact existing:

```text
5 document-grouped outer folds
seeds 42,43,44
```

---

# 23. Seed semantics

Seeds:

```text
stochastic replications
```

и не являются независимыми statistical observations.

---

# 24. Replay gate

Перед новой моделью exact replay:

```text
S0
S1/H2
```

---

# 25. H2 replay

Должны совпасть:

```text
candidate IDs
QA answers
H2 runtime features
selected candidate
final answer
switch flags
Token F1
```

---

# 26. Replay failure

Если replay H2 fails:

```text
STOP
H2_REPLAY_FAILED
```

Не строить приблизительную H2.

---

# 27. Источник H2 features

Источник истины:

```text
V9 frozen contract
V9 code
V9 canonical feature cache
```

---

# 28. Candidate feature vector

Для candidate `i`:

```text
x_i = exact numeric H2 runtime feature vector
```

Сохранить exact ordered schema:

```text
audit/h2_feature_schema.json
```

---

# 29. Learned H2 outputs исключаются

Primary `x_i` не включает:

```text
H2 predicted utility
H2 predicted harm
H2 final switch
```

V10 должен заменить H2 decision layer.

---

# 30. Разрешённые H2 runtime signals

Только реально существующие frozen features, например:

```text
B0 score / delta / rank / margin
Q-A relevance
FullLEX
PairLEX
validity / empty indicators
existing deterministic H2 interactions
```

Не создавать новые признаки под видом historical.

---

# 31. Forbidden features

Runtime-models не получают:

```text
gold
actual Token F1
actual utility
evidence gold label
oracle rank
oracle candidate
dataset id
language id
aspect id
```

---

# 32. K

Разрешён только общий structural scalar:

```text
candidate_count
```

Запрещены ручные rules вида:

```text
if K == 5 ...
```

## Обязательный special case `K=1`

В canonical RuSERRC существуют вопросы с `K=1`.

Для них alternatives отсутствуют, поэтому:

```text
S2 final candidate = B0
fallback_reason = NO_ALTERNATIVE
```

и **не запускать**:

```text
SWITCH inference
PAIRWISE RANK inference
ANSWER-RISK inference
EVIDENCE-RISK inference
```

Training semantics:

```text
SWITCH:
K=1 question исключается из training,
поскольку SWITCH в runtime для него не вызывается.

RANK:
K=1 question исключается — pairs отсутствуют.

ANSWER-RISK:
K=1 question исключается — alternatives отсутствуют.

EVIDENCE-RISK:
K=1 question исключается — модель применяется только к alternatives.
```

Для evaluation bookkeeping:

```text
positive_opportunity = false
final candidate = B0
```

`K=1` вопросы остаются в итоговых corpus quality/evidence metrics и bootstrap.

---

# 33. Question aggregate vector q

Из `x_i` построить deterministic question-level vector.

Для каждого continuous candidate feature:

```text
max
min
mean
std
range
```

---

# 34. Дополнительные question aggregates

```text
candidate_count
valid_answer_count
empty_answer_count
unique_normalized_answer_count
answer_diversity
largest_exact_answer_cluster_ratio
```

---

# 35. Answer diversity

```text
answer_diversity =
unique_normalized_answer_count
/
max(valid_answer_count,1)
```

Использовать existing canonical answer normalization.

---

# 36. No semantic clustering

Не использовать:

```text
embeddings
semantic answer clustering
new lemmatization
```

для question aggregates.

---

# 37. NaN/Inf

Перед fit:

```text
assert all model features finite.
```

Если feature undefined — использовать exact historical H2 fallback, а не новый imputation rule.

---

# 38. SWITCH target

Для question `q`:

\[
y_q^{switch}=
1\left[
\max_{i\ne0}
TokenF1(A_i,G)
>
TokenF1(A_0,G)
\right].
\]

---

# 39. SWITCH semantics

`1`:

> существует хотя бы одна alternative лучше B0.

`0`:

> ни одна alternative не улучшает B0.

---

# 40. SWITCH input

Только:

```text
q
```

без gold-derived runtime features.

---

# 41. SWITCH model

Primary:

```text
StandardScaler
L2 LogisticRegression
solver=liblinear
class_weight=None
random_state = model seed, если sklearn version поддерживает параметр
```

Перед каждым фактическим fit:
- в каждом `INNER_TRAIN`;
- на полном outer `TRAIN/CAL` перед refit;

проверить среди questions с `K>=2`:

```text
>=10 positive questions
>=10 negative questions
```

Если хотя бы одного класса меньше:

```text
SWITCH_TRAINING_BLOCKED
```

Не использовать constant-class surrogate.

---

# 42. SWITCH C grid

```text
0.01
0.1
1
10
```

---

# 43. SWITCH tuning

На TRAIN-only OOF минимизировать:

```text
binary log-loss.
```

Tie `<0.001`:

```text
smaller C.
```

---

# 44. SWITCH diagnostics

```text
ROC-AUC
PR-AUC
log-loss
Brier
opportunity recall
opportunity precision
```

---

# 45. PAIRWISE RANK target

Только для questions с:

```text
K>=2
```

внутри каждого question построить pairs `(i,j)`.

Label:

\[
y_{ij}=1
\iff
TokenF1(A_i,G)>TokenF1(A_j,G).
\]

---

# 46. Pair tie rule

Primary:

```text
PAIR_MIN_GAP = 1.0 Token F1.
```

Если:

\[
|F1_i-F1_j|<1.0
\]

pair не используется для training.

---

# 47. Symmetric augmentation

Для каждого unordered pair добавить:

```text
(i,j), y
(j,i), 1-y
```

---

# 48. Pair features

\[
z_{ij}=[x_i-x_j,\ q].
\]

Не использовать cross-question pairs.

---

# 49. Pairwise weighting

Каждый question с хотя бы одной usable pair имеет total weight:

```text
1.
```

Если `m` usable unordered pairs:

```text
каждая directed pair получает 1/(2m).
```

Questions без usable pairs после `PAIR_MIN_GAP` не входят в ranker training.

Перед fit проверить минимум:

```text
>=10 unique questions with usable pairs
>=20 usable unordered pairs
```

Иначе:

```text
RANK_TRAINING_BLOCKED
```

---

# 50. Ranker model

Primary:

```text
sklearn HistGradientBoostingClassifier
random_state = model seed
```

Нелинейный, но компактный и не нейросетевой.

Если установленная sklearn version не поддерживает конкретный randomization parameter ожидаемым образом, зафиксировать фактическое API/version в environment artifact и не имитировать seed вручную.

---

# 51. Ranker grid

```text
learning_rate = 0.05

max_leaf_nodes:
7
15

max_iter:
100
200

l2_regularization:
1
10
```

Всего 8 configurations.

---

# 52. Ranker tuning objective

На TRAIN-only OOF, только для questions с `K>=2`, максимизировать:

```text
mean question-level Token F1
of top-ranked alternative.
```

Это tuning metric, а не unbiased scientific estimate.

---

# 53. Ranker tie-break

Если difference `<0.10 F1`:

1. smaller `max_leaf_nodes`;
2. smaller `max_iter`;
3. larger `l2_regularization`.

---

# 54. Pairwise inference

Для каждой unordered pair `(i,j)` вычислить **оба** directed predictions:

\[
p^{raw}_{ij}=P(i>j),
\qquad
p^{raw}_{ji}=P(j>i).
\]

Обычный classifier не гарантирует:

\[
p^{raw}_{ij}=1-p^{raw}_{ji}.
\]

Поэтому обязательна inference-symmetrization:

\[
p_{ij}
=
\frac{
p^{raw}_{ij}
+
(1-p^{raw}_{ji})
}{2},
\]

\[
p_{ji}=1-p_{ij}.
\]

Для candidate `i` score считается относительно **всех остальных доступных candidates, включая исходный B0**:

\[
R_i=
\frac{1}{K-1}
\sum_{j\ne i}p_{ij},
\qquad K\ge2.
\]

После вычисления `R_i` B0 исключается из множества alternatives, но comparison с B0 остаётся частью rank score.

Для `K=1` pairwise inference не выполняется.

---

# 55. Rank ordering

```text
descending R_i
```

Tie-break:

1. higher B0 score;
2. lower original B0 rank;
3. candidate id.

---

# 56. Alternative set

После положительного SWITCH:

1. `R_i` уже рассчитан с учётом pairwise comparisons со всеми candidates, включая B0;
2. для выбора сформировать:

```text
alternatives = all candidates except original B0 top1
```

3. сортировать alternatives по `R_i`.

B0 остаётся fallback.

---

# 57. ANSWER-RISK target

Для alternative `i`:

\[
y_i^{ansharm}
=
1[
TokenF1(A_i,G)<TokenF1(A_0,G)
].
\]

---

# 58. Neutral answer

Если F1 одинаков:

```text
answer_harm = 0.
```

---

# 59. ANSWER-RISK input

\[
a_i=[x_i-x_0,\ q].
\]

---

# 60. ANSWER-RISK model

```text
StandardScaler
L2 LogisticRegression
solver=liblinear
class_weight=None
random_state = model seed, если поддерживается
```

Training rows существуют только для alternatives (`K>=2`).

Перед каждым фактическим fit:
- в каждом `INNER_TRAIN`;
- на полном outer `TRAIN/CAL` перед refit;

проверить:

```text
>=10 harm rows
>=10 non-harm rows
>=5 unique questions represented in each class
```

Иначе:

```text
ANSWER_RISK_TRAINING_BLOCKED
```

Не обучать constant predictor.

---

# 61. ANSWER-RISK grid

```text
C = 0.01, 0.1, 1, 10
```

Tuning metric:

```text
OOF log-loss.
```

Tie:

```text
smaller C.
```

---

# 62. ANSWER-RISK diagnostics

```text
ROC-AUC
PR-AUC
Brier
log-loss
calibration
```

---

# 63. EVIDENCE-RISK purpose

Отдельно оценить:

> является ли candidate chunk **недостаточным evidence** для данного question независимо от того, совпала ли извлечённая строка ответа с gold.

Основной binary target:

```text
SAFE_EVIDENCE
vs
UNSAFE_EVIDENCE
```

`HN` — только один подтип `UNSAFE_EVIDENCE`.

Не смешивать эту цель с answer F1.

---

# 64. Evidence label audit

До full run создать:

```text
audit/evidence_label_mapping.json
```

с exact mapping для каждой cell.

---

# 65. Safe evidence

Canonical safe:

```text
robust_positive
positive
```

или exact project equivalents, если они означают sufficient evidence.

---

# 66. Harmful evidence

Canonical evidence-harm:

```text
hard_negative
easy_negative
explicit insufficient evidence
```

или exact equivalents.

---

# 67. Ambiguous / missing labels

```text
ambiguous
unresolved
GSRN / question-sensitive unresolved
missing/unlabeled
```

не использовать как binary evidence-risk targets.

Для evidence-risk:

```text
sample_weight = 0.
```

Если встречается новый **неизвестный non-null label**, которого нет в frozen mapping:

```text
raise RuntimeError("UNKNOWN_EVIDENCE_LABEL")
```

Не относить его автоматически к safe/unsafe.

---

# 68. Mapping prohibition

Не использовать string guessing.

Mapping должен опираться на existing project semantics и быть frozen до outer EVAL.

---

# 69. Evidence minimum data

Проверять **в каждом training partition, где модель реально fit-ится**:
- каждом `INNER_TRAIN`;
- полном outer `TRAIN/CAL` перед refit.

После exclusion ambiguous/missing требуется:

```text
>=10 SAFE_EVIDENCE rows
>=10 UNSAFE_EVIDENCE rows
```

и минимум:

```text
>=5 unique questions в каждом классе.
```

---

# 70. Evidence-data failure

Если Section 69 не выполнен:

```text
EVIDENCE_RISK_TRAINING_BLOCKED
```

Outer fold не подменять ручным prior.

---

# 71. EVIDENCE-RISK input

\[
e_i=[x_i,\ x_i-x_0,\ q].
\]

---

# 72. EVIDENCE-RISK model

```text
StandardScaler
L2 LogisticRegression
solver=liblinear
class_weight=None
random_state = model seed, если поддерживается
```

---

# 73. EVIDENCE-RISK grid

```text
C = 0.01, 0.1, 1, 10
```

Tuning metric:

```text
OOF log-loss.
```

Tie:

```text
smaller C.
```

---

# 74. Evidence diagnostics

```text
ROC-AUC
PR-AUC
Brier
log-loss

UNSAFE_EVIDENCE recall
UNSAFE_EVIDENCE precision
SAFE_EVIDENCE recall

HN recall
HN precision
```

`HN` metrics являются subgroup diagnostics внутри общего unsafe target.

---

# 75. Candidate-model weighting

Для answer-risk и evidence-risk:

```text
total question weight = 1.
```

Если question имеет `m` eligible candidate rows:

```text
row weight = 1/m.
```

---

# 76. Inner CV

Внутри каждого:

```text
(cell, outer_fold, model_seed)
```

использовать:

```text
4 document-grouped inner folds.
```

Правило V10 фиксировано:

```text
model seeds = 42,43,44
inner split assignment = одинаковый для всех трёх seeds
model fitting = отдельно для каждого seed
hyperparameter selection = отдельно для каждого seed
threshold selection = отдельно для каждого seed
outer prediction = отдельно для каждого seed
```

Если model family фактически deterministic при фиксированных данных, совпадающие результаты seeds допустимы и должны быть честно отражены.

Не менять эту convention в зависимости от V9 implementation.

---

# 77. Same inner splits

Одинаковые document splits для:

```text
SWITCH
RANK
ANSWER-RISK
EVIDENCE-RISK
```

---

# 78. Split determinism

Reuse canonical grouped-splitter family.

Если helper принимает random state:

```text
INNER_SPLIT_SEED=42.
```

Сохранить split manifest и hashes.

---

# 79. Honest OOF

Каждый inner model:

```text
fit INNER_TRAIN
predict INNER_VALID.
```

---

# 80. Required OOF outputs

На outer TRAIN/CAL получить:

```text
p_switch
pairwise probabilities
R_i
p_answer_harm
p_evidence_harm
```

только OOF.

Важно:

OOF используется для **model/threshold selection внутри TRAIN**, поэтому его итоговый F1 после выбора hyperparameters/thresholds является tuning estimate, а не unbiased scientific estimate.

Отдельный третий CV-уровень для threshold calibration в V10 не требуется: hyperparameters и thresholds вместе рассматриваются как части training procedure, а unbiased estimate всей процедуры обеспечивает sealed outer EVAL.

Следовательно:

```text
inner OOF = tuning evidence
outer EVAL = scientific development estimate
```

---

# 81. No in-sample tuning

Запрещено на той же training row:

```text
fit model
и
использовать prediction для threshold selection.
```

---

# 82. Thresholds

После OOF выбрать:

```text
tau_switch
tau_answer_harm
tau_evidence_harm.
```

---

# 83. tau_switch grid

```text
0.30
0.40
0.50
0.60
0.70
```

---

# 84. tau_answer_harm grid

```text
0.20
0.30
0.40
0.50
```

---

# 85. tau_evidence_harm grid

```text
0.10
0.20
0.30
0.40
0.50
```

---

# 86. OOF policy simulation

Для каждого threshold tuple:

```text
if p_switch < tau_switch:
    choose B0
else:
    rank alternatives by R_i
    scan in rank order
    select first candidate satisfying:
        p_answer_harm <= tau_answer_harm
        AND
        p_evidence_harm <= tau_evidence_harm

    if none safe:
        choose B0
```

---

# 87. Threshold objective

Максимизировать:

```text
final OOF Token F1
```

subject to safety.

---

# 88. Absolute evidence safety

Relative to B0:

```text
ΔHit@1 >= -0.50 pp
ΔHN@1 <= +1.00 pp
```

---

# 89. Sufficient-evidence safety

Если canonical metric доступна:

```text
selected sufficient-positive rate
```

не должна снизиться более чем на:

```text
1.0 pp
```

relative B0.

---

# 90. Threshold tie-break

Если F1 differs `<0.10`:

1. lower HN@1;
2. higher sufficient-positive rate;
3. higher final positive capture;
4. lower harmful-switch rate;
5. higher `tau_switch`;
6. lower switch rate.

---

# 91. No safe threshold tuple

Если safe tuple отсутствует:

```text
OUTER_FOLD_POLICY = B0_FALLBACK.
```

---

# 92. Почему не fallback H2

H2 остаётся отдельной comparator S1.

Автоматический H2 fallback внутри S2 скрыл бы failure новой decision architecture.

---

# 93. Refit

После выбора hyperparameters и thresholds:

```text
refit SWITCH
refit RANK
refit ANSWER-RISK
refit EVIDENCE-RISK
```

на полном outer TRAIN/CAL.

---

# 94. Outer EVAL

После refit:

```text
predict outer EVAL once.
```

---

# 95. Outer-EVAL prohibition

После просмотра outer EVAL нельзя менять:

```text
model family
features
PAIR_MIN_GAP
evidence mapping
C grids
ranker grid
threshold grids
safety budgets
```

---

# 96. Ablation ladder

Diagnostic only:

```text
A0 = SWITCH + RANK

A1 = SWITCH + RANK + ANSWER-RISK

A2 = SWITCH + RANK + EVIDENCE-RISK

A3 = SWITCH + RANK + ANSWER-RISK + EVIDENCE-RISK
     = S2 PRIMARY
```

Каждая ablation получает **собственную TRAIN-only threshold calibration** на тех же OOF rows:

```text
A0:
tune tau_switch

A1:
tune tau_switch + tau_answer_harm

A2:
tune tau_switch + tau_evidence_harm

A3:
tune all three thresholds
```

с теми же safety constraints, которые применимы к данной ablation.

Outer EVAL никогда не участвует в threshold tuning.

---

# 97. Ablations cannot replace primary

Даже если A0/A1/A2 outer F1 выше:

```text
diagnostic only.
```

Разности между ablations интерпретировать как:

```text
system-level incremental effects
```

а не как чистые causal effects одного risk head, поскольку thresholds у систем калибруются отдельно.

---

# 98. Primary answer metric

```text
Token F1
```

exact canonical evaluator.

---

# 99. Secondary answer metrics

```text
Char F1
Unicode ROUGE-L
EM
Lemma F1 RU
```

BERTScore только если exact frozen implementation уже существует.

---

# 100. Evidence metrics

```text
Hit@1
Hit@K
HN@1
HN@K

selected SAFE_EVIDENCE rate
selected UNSAFE_EVIDENCE rate
selected sufficient-positive rate

F1|selected positive
Amb/GSRN where historically defined
```

`SAFE/UNSAFE` metrics должны использовать тот же frozen mapping, что EVIDENCE-RISK.

---

# 101. Positive opportunity

Question имеет positive opportunity, если:

\[
\max_{i\ne0}u_i>0.
\]

---

# 102. SWITCH opportunity recall

\[
Recall_{switch}
=
P(
p_{switch}\ge\tau_{switch}
\mid
positive\ opportunity
).
\]

---

# 103. Final Positive Capture

Главный mechanism metric:

> доля positive-opportunity questions, где финальный S2 answer действительно имеет `u>0`.

Primary aggregation:
1. вычислить binary capture per `(question, seed)`;
2. усреднить по seeds внутри question;
3. усреднить question-level values в cell.

Не считать три seeds тремя независимыми questions.

---

# 104. Stay-on-B0 despite opportunity

Отдельно считать:

> долю positive opportunities, где S2 оставил B0.

---

# 105. Harmful switch rate

```text
fraction final switches with u < 0.
```

Primary aggregation также выполняется:
```text
seed → question → cell
```
без трактовки seeds как независимых observations.

---

# 106. Ranker diagnostics

```text
pairwise AUC
top-ranked alternative F1
oracle-best alternative capture
mean rank regret
median rank regret
```

---

# 107. Rank regret

\[
Regret =
\max_{i\ne0}F1_i
-
F1_{\hat i},
\]

где `hat i` — top-ranked alternative.

---

# 108. Oracle-best final capture

Diagnostic only:

> доля positive opportunities, где S2 выбрал actual best-F1 candidate.

Gold oracle не является runtime system.

---

# 109. B0-rank analysis

Post-hoc:

```text
best alternative original rank:
2
3
4
5
```

Для каждого:

```text
H2 capture
S2 capture.
```

Не использовать в policy.

---

# 110. Oracle ceiling

Для каждого cell report:

```text
QA-oracle top-K
available oracle gain
H2 used-oracle-gain %
S2 used-oracle-gain %
```

---

# 111. Selection-bottleneck cells

Для mechanism criterion:

```text
SciER384
SciER128
RuSERRC
SciMDIX
```

SciREX анализируется отдельно, потому что у него также существенны retrieval и reader bottlenecks.

---

# 112. Mechanism success criterion

Основная bottleneck hypothesis поддержана, если:

1. Final Positive Capture растёт минимум на `+10 pp` относительно H2
   хотя бы на `3 из 4` selection-bottleneck cells;

2. harmful-switch rate не ухудшается более чем на `+2 pp`;

3. HN@1 safety проходит.

---

# 113. Overall quality criterion

Для S2:

```text
family-balanced ΔTokenF1(S2-S1) >= +0.50
```

и на каждой cell:

```text
point Δ(S2-S1) >= -0.25.
```

---

# 114. Uncertainty guard

На каждой cell:

```text
LCB95(S2-S1) > -0.50 F1.
```

---

# 115. Strong universal criterion

Желаемый строгий результат:

```text
S2-S1 > 0
```

на всех пяти cells.

---

# 116. Evidence safety

S2 vs B0:

```text
ΔHit@1 >= -0.50 pp
ΔHN@1 <= +1.00 pp
```

на каждой cell.

---

# 117. Incremental evidence preservation

S2 vs H2:

```text
ΔHN@1 <= +0.50 pp
```

и, если доступна:

```text
selected sufficient-positive rate
не хуже более чем на 1.0 pp.
```

---

# 118. Non-degeneracy

S2 отличается от B0 устойчиво минимум на:

```text
max(3, ceil(1% questions))
```

per cell.

Stable:

```text
same question switched in >=2/3 seeds.
```

---

# 119. Final development PASS

`V10_SRR_DEVELOPMENT_CANDIDATE`, если одновременно:

```text
mechanism success PASS
overall quality PASS
uncertainty guard PASS
evidence safety PASS
non-degeneracy PASS
```

---

# 120. Если mechanism улучшен, а F1 нет

Не менять V10.

Разложить failure по:

```text
SWITCH
RANK
ANSWER-RISK
EVIDENCE-RISK
reader ceiling.
```

---

# 121. Если SWITCH слаб

Если opportunity recall/PR-AUC низкие:

следующей версии нужен более информативный:

```text
question-level opportunity signal.
```

---

# 122. Если RANK слаб

Если:

```text
oracle-best capture low
regret high
```

следующей версии нужны более прямые candidate/evidence features.

---

# 123. Если EVIDENCE-RISK слаб

Если:

```text
evidence PR-AUC low
HN unsafe switches remain
```

это прямое основание для отдельного:

```text
Q + Answer + Chunk support model.
```

---

# 124. Если ANSWER-RISK слаб

Если harmful switches остаются высокими:

следующая версия должна улучшать answer-relative risk features, а не retrieval.

---

# 125. Direct support НЕ входит в V10

Не реализовывать:

\[
S_{support}=f(Q,A_i,C_i)
\]

в этой версии.

---

# 126. Условие для V11/support

V11 обоснован только после завершения V10, если module diagnostics показывают:

```text
selection still limited by evidence discrimination.
```

---

# 127. Future support semantics

Если V11 потребуется, signal должен оценивать:

> подтверждает ли конкретный chunk конкретный extracted answer на данный question.

Не только similarity.

---

# 128. SciREX special case

На SciREX дополнительно существуют:

```text
retrieval ceiling
reader ceiling.
```

Даже идеальный SRR не устранит всю потерю.

---

# 129. SciREX future line

Если V10 улучшает selection, но SciREX остаётся слаб:

отдельно исследовать:

```text
retrieval improvement
local context expansion
reader context construction.
```

---

# 130. Statistical unit

Seeds не независимы.

Порядок:

1. per-question per-seed;
2. average over seeds within question;
3. paired document-cluster bootstrap.

---

# 131. Bootstrap

```text
iterations=5000
seed=42
paired=true
cluster=canonical_document_id
```

---

# 132. Required contrasts

```text
S1-S0
S2-S0
S2-S1

A0-S1
A1-S1
A2-S1
```

Ablation contrasts diagnostic.

---

# 133. SciER coupled bootstrap

Для SciER:

```text
resample canonical source document IDs once
apply same sampled ID multiset to 128 and 384
compute two cell effects
average them into one SciER-family effect.
```

---

# 134. Family-balanced global effect

Families:

```text
SciREX
SciER
RuSERRC
SciMDIX
```

равные веса.

---

# 135. Global bootstrap CI

Обязательно:

```text
Global Δ(S1-S0)
Global Δ(S2-S0)
Global Δ(S2-S1)
```

---

# 136. Resource hypothesis

S2 не добавляет neural forwards относительно H2.

---

# 137. Forward equality

Для каждого question:

```text
QA_forward_count(S2) == QA_forward_count(S1)
Q-A_forward_count(S2) == Q-A_forward_count(S1)
```

---

# 138. Pairwise compute ceiling

При `K<=5`:

\[
K(K-1)/2\le10
\]

unordered pairs.

Из-за обязательной symmetrization требуется максимум:

```text
20 directed pair rows
```

на question.

Оба направления следует подавать в один batched `predict_proba`, а не делать отдельный model call на каждую пару.

---

# 139. Resource targets

Relative S2/S1:

```text
median latency <=1.10
P95 latency <=1.15
peak VRAM <=1.05
neural parameter ratio =1.00
```

---

# 140. Resource benchmark

Fresh-process systems:

```text
S0
S1
S2
```

Одинаковые sample IDs.

---

# 141. Benchmark procedure

Предпочтительно:

```text
fold0
cap 100 questions/cell
warmup >=10
3 repeats
CUDA synchronization
```

---

# 142. Qwen/BGE

Если exact fresh runner отсутствует:

```text
BLOCKED
```

Не подставлять старые hardware numbers.

---

# 143. Artifact namespace

```text
artifacts/qa_chunk_selection_v10_srr/
  run_<timestamp>/
```

---

# 144. Artifact structure

```text
run/
  status.json
  manifest.json
  frozen_contract.yaml
  COMPLETED.json

  audit/
    git_before/
    git_after/
    historical_hashes_before.json
    historical_hashes_after.json

    s0_replay.json
    s1_h2_replay.json

    h2_feature_schema.json
    evidence_label_mapping.json

    outer_split_hashes.json
    inner_split_manifests/
    inner_split_hashes.json

    candidate_universe.json
    scier_lineage.json
    cross_corpus_overlap.json
    model_manifests/

  models/
    switch/
    rank/
    answer_risk/
    evidence_risk/

  oof/
    switch/
    rank/
    answer_risk/
    evidence_risk/
    threshold_search/

  predictions/
    S0/
    S1/
    S2/
    A0/
    A1/
    A2/

  analysis/
    primary_metrics.json
    paired_bootstrap.json
    global_family_bootstrap.json

    switch_metrics.json
    rank_metrics.json
    answer_risk_metrics.json
    evidence_risk_metrics.json

    positive_capture.json
    stay_on_b0.json
    harmful_switches.json
    oracle_capture.json

    b0_rank_analysis.json
    aspect_analysis.json
    k_analysis.json

    fold_stability.json
    seed_stability.json
    development_decision.json

  resource_benchmark/
    sample_manifest.json
    raw/
    summary.json
    environment.json

  reports/
    v10_srr_results.md
    v10_module_analysis.md
    reproducibility.md
```

---

# 145. Runtime/evaluation separation

```json
{
  "runtime": {},
  "evaluation": {}
}
```

---

# 146. Runtime fields

```text
cell
document_id
question_id
outer_fold
seed

candidate IDs
B0 scores
QA answers

H2 x_i vectors
q aggregate vector

p_switch

pairwise probabilities
R_i
rank order

p_answer_harm
p_evidence_harm

thresholds

selected candidate
fallback reason
final answer

forward counts
```

---

# 147. Evaluation-only fields

```text
gold answer
actual Token F1
actual utility
evidence label

positive opportunity
oracle candidate
oracle rank

good/bad/neutral switch
```

---

# 148. Fallback enum

```text
NO_ALTERNATIVE
NO_SWITCH
NO_SAFE_CANDIDATE
SELECTED_ALTERNATIVE
OUTER_FOLD_B0_FALLBACK
```

---

# 149. Provenance

Per prediction:

```text
outer fold
seed
inner split hash

switch model hash
rank model hash
answer-risk model hash
evidence-risk model hash

feature schema hash
threshold tuple
```

---

# 150. Cache isolation

V10 writes only in V10 namespace.

Historical cache:

```text
read-only.
```

---

# 151. Immutable run

После bug:

```text
mark run INVALID
create new run.
```

Не перезаписывать completed artifacts.

---

# 152. Unit tests — history/replay

```text
test_history_read_only
test_historical_hashes_unchanged

test_s0_replay
test_h2_candidate_ids
test_h2_answers
test_h2_features
test_h2_final_metrics
```

---

# 153. Unit tests — features

```text
test_runtime_no_gold
test_no_dataset_id
test_no_language_id
test_no_aspect_feature

test_question_aggregates_deterministic
test_features_finite
test_feature_order_stable
```

---

# 154. Unit tests — SWITCH

```text
test_switch_target
test_k1_excluded_from_switch_training
test_k1_skips_switch_inference

test_switch_question_weight
test_switch_class_count_guard
test_switch_oof
test_switch_probability_bounds
```

---

# 155. Unit tests — RANK

```text
test_pair_target
test_pair_min_gap
test_symmetric_pairs
test_pair_question_weight_sum

test_pair_difference_features
test_no_cross_question_pair

test_k1_no_pair_rows
test_rank_training_support_guard

test_pair_probability_symmetrization
test_pair_antisymmetry
test_rank_score_uses_b0_comparison
test_rank_score_bounds
test_rank_tiebreak
```

---

# 156. Unit tests — ANSWER-RISK

```text
test_answer_harm_target
test_neutral_not_harm
test_k1_no_answer_risk_rows
test_answer_risk_class_count_guard
test_answer_risk_question_weight
test_answer_risk_oof
```

---

# 157. Unit tests — EVIDENCE-RISK

```text
test_evidence_mapping_exact
test_no_string_guess_mapping
test_unknown_evidence_label_raises
test_ambiguous_missing_weight_zero

test_k1_no_evidence_risk_rows
test_min_class_count_each_inner_train
test_evidence_unique_question_guard

test_evidence_question_weight
test_evidence_oof
```

---

# 158. Unit tests — nested CV

```text
test_inner_document_grouping
test_same_inner_splits_modules
test_inner_hash_stability

test_no_outer_eval_tuning
test_no_in_sample_threshold_search
```

---

# 159. Unit tests — policy

```text
test_switch_false_b0
test_rank_only_after_switch

test_answer_filter
test_evidence_filter

test_scan_rank_order
test_no_safe_candidate_b0

test_threshold_tiebreak
test_no_safe_tuple_b0_fallback
test_b0_fallback_not_used_for_training_failure

test_ablation_thresholds_tuned_train_only
```

---

# 160. Unit tests — resources/leakage

```text
test_same_qa_forwards_s1_s2
test_same_qarel_forwards_s1_s2
test_same_neural_model_set

test_runtime_no_gold
test_scier_embargo
test_cross_corpus_overlap
```

---

# 161. Smoke test

До full run:

```text
1 EN cell
1 RU cell
1 outer fold
1 seed
small subset
```

Проверить:

```text
replay
SWITCH
pair generation
ranker
two risk heads
OOF thresholds
refit
outer-style prediction
artifact writing
resource counters
```

---

# 162. Smoke cannot tune science

Нельзя менять по smoke quality:

```text
model families
PAIR_MIN_GAP
grids
thresholds
safety budgets
success criteria
```

---

# 163. Frozen contract

До full outer EVAL сохранить:

```text
frozen_contract.yaml
SHA-256
```

---

# 164. Frozen contract contents

```text
canonical hashes
H2 feature schema

SWITCH target/model/grid

PAIR target
PAIR_MIN_GAP
ranker model/grid
directed-pair symmetrization
rank aggregation

ANSWER-RISK target/model/grid

evidence label mapping
EVIDENCE-RISK target/model/grid

module training-support/class-count guards

inner CV
seed convention
seeds

threshold grids
threshold objective
safety constraints
tie-breaks

S0/S1/S2
A0/A1/A2

metrics
bootstrap
success criteria
resource targets
```

---

# 165. Execution order

```text
A. git/history audit
B. hashes BEFORE
C. S0/H2 replay
D. H2 schema audit
E. evidence-label audit
F. overlap/SciER audit

G. unit tests
H. smoke
I. freeze contract

J. for each outer fold:
   1. create inner splits
   2. SWITCH OOF
   3. RANK OOF
   4. ANSWER-RISK OOF
   5. EVIDENCE-RISK OOF
   6. choose hyperparameters
   7. threshold search on OOF
   8. refit all modules
   9. evaluate outer EVAL once

K. aggregate seeds
L. bootstrap
M. family bootstrap
N. module analysis
O. resource benchmark

P. hashes AFTER
Q. history integrity
R. reports
S. development decision
```

---

# 166. Per-fold failure

Если обязательный module нельзя корректно обучить из-за:
- single-class target;
- недостаточного class-count;
- отсутствия usable pairs;
- unknown evidence labels;
- numerical/model fit error;

то:

```text
mark outer fold FAILED
```

Не silently substitute:
```text
constant predictor
H2 decision
manual prior
другой classifier.
```

`OUTER_FOLD_B0_FALLBACK` разрешён только в заранее определённом случае:

```text
models trained successfully,
но ни один threshold tuple не прошёл safety constraints.
```

---

# 167. Whole-run completeness

`COMPLETED` только если:

```text
5 cells
5 outer folds
3 seeds
```

имеют valid predictions или заранее определённый:

```text
OUTER_FOLD_B0_FALLBACK
```

из-за отсутствия safe threshold tuple.

---

# 168. Required primary table

```text
Cell
S0 F1
S1 H2 F1
S2 SRR F1

ΔS1-S0 + CI
ΔS2-S0 + CI
ΔS2-S1 + CI

Hit1
HN1
selected-positive rate

Final Positive Capture
Harmful Switch Rate
Stay-on-B0 despite opportunity
```

---

# 169. Required SWITCH table

```text
Cell
ROC-AUC
PR-AUC
Brier
log-loss
opportunity recall
opportunity precision
```

---

# 170. Required RANK table

```text
Cell
pairwise AUC
top-ranked alternative F1
oracle-best capture
mean regret
median regret
```

---

# 171. Required RISK table

```text
Cell

ANSWER-RISK:
AUC
PR-AUC
Brier
log-loss

EVIDENCE-RISK:
AUC
PR-AUC
Brier
log-loss
HN recall
safe evidence recall
```

---

# 172. Required bottleneck table

```text
Cell
Hit@K
QA-oracle F1

H2 used-oracle-gain %
S2 used-oracle-gain %

H2 positive capture
S2 positive capture

H2 harmful switch rate
S2 harmful switch rate
```

---

# 173. Required stability

```text
cell
fold
seed

ΔS1-S0
ΔS2-S1

switch recall
positive capture
harmful rate
```

---

# 174. Aspect/K diagnostics

Только post-hoc.

Явно пометить:

```text
NOT USED FOR MODEL OR POLICY.
```

---

# 175. Resource table

```text
Cell
System
QA forwards
Q-A forwards
pair comparisons
median latency
P95 latency
throughput
VRAM
RSS
neural params
tabular params
```

---

# 176. Если V10 успешен

Допустимая formulation:

> Разделение финального решения на обнаружение полезного переключения, относительное ранжирование candidate answers и независимую оценку answer/evidence risk позволило лучше использовать уже доступный top-K QA-потенциал без увеличения числа neural forward passes.

---

# 177. Если V10 не успешен

Не делать unrestricted V10.1 на тех же outer results.

Использовать module diagnostics только для новой заранее зарегистрированной гипотезы.

Failure V10:

```text
не отменяет development evidence H2/V9.
```

Если S2 не проходит критерии замены H2:

```text
retain H2 as current frozen reference method
```

и отдельно решить:
- переходить ли с H2 к untouched confirmation;
- или обосновывать заранее зарегистрированный V11 evidence-support experiment.

---

# 178. Independent confirmation

Если development criteria выполнены:

```text
freeze V10 immediately
```

и перейти к:

```text
untouched EN confirmation
untouched RU confirmation.
```

---

# 179. Confirmation must preserve

```text
feature schema
targets
model families
hyperparameter grids
PAIR_MIN_GAP
pairwise symmetrization
K=1 policy
evidence mapping
threshold grids
safety budgets
seed convention
training/CV procedure
metrics
```

Untouched confirmation dataset считается совместимым с exact V10 только если его `CONFIRM_TRAIN/CAL` содержит:
- gold answers для SWITCH/RANK/ANSWER-RISK targets;
- candidate-level evidence labels, которые однозначно отображаются в frozen `SAFE/UNSAFE/EXCLUDE` semantics.

Если evidence labels отсутствуют, exact V10-SRR с `EVIDENCE-RISK` нельзя считать подтверждённым; требуется заранее определённая отдельная variant, а не silent removal evidence head.

---

# 180. Pre-run checklist

```text
[ ] V10 namespace created
[ ] history write guard active
[ ] git BEFORE saved
[ ] hashes BEFORE saved

[ ] S0 replay PASS
[ ] H2 replay PASS

[ ] H2 feature schema frozen
[ ] learned H2 outputs excluded
[ ] q aggregates frozen
[ ] K=1 policy frozen

[ ] SWITCH frozen
[ ] SWITCH class-count guard frozen

[ ] PAIR_MIN_GAP=1.0 frozen
[ ] pairwise model/grid frozen
[ ] pairwise symmetrization frozen
[ ] rank aggregation frozen
[ ] rank training-support guard frozen

[ ] ANSWER-RISK frozen
[ ] ANSWER-RISK class-count guard frozen

[ ] evidence mapping audited
[ ] unknown-label failure policy frozen
[ ] EVIDENCE-RISK frozen
[ ] evidence per-inner-train support guard frozen

[ ] outer folds frozen
[ ] inner folds frozen
[ ] same inner split assignment across seeds verified
[ ] seed-level independent tuning convention frozen
[ ] seeds 42/43/44 frozen

[ ] threshold grids frozen
[ ] safety rules frozen
[ ] tie-breaks frozen

[ ] systems/ablations frozen
[ ] metrics/success criteria frozen
[ ] bootstrap frozen
[ ] resource targets frozen

[ ] SciER embargo PASS
[ ] overlap audit PASS

[ ] unit tests PASS
[ ] smoke PASS

[ ] contract SHA-256 saved
[ ] no full outer EVAL seen before freeze
```

---

# 181. Post-run checklist

```text
[ ] all cells complete
[ ] all outer folds complete
[ ] all seeds complete

[ ] S0/S1/S2 complete
[ ] A0/A1/A2 complete

[ ] SWITCH analysis complete
[ ] RANK analysis complete
[ ] ANSWER-RISK analysis complete
[ ] EVIDENCE-RISK analysis complete

[ ] positive capture complete
[ ] stay-on-B0 complete
[ ] harmful switches complete
[ ] oracle/regret complete

[ ] answer metrics complete
[ ] evidence metrics complete

[ ] paired bootstrap complete
[ ] SciER coupled bootstrap complete
[ ] global family bootstrap complete

[ ] fold/seed stability complete
[ ] B0-rank diagnostic complete
[ ] aspect/K post-hoc complete

[ ] forward equality PASS
[ ] resource benchmark complete

[ ] development decision generated

[ ] hashes AFTER saved
[ ] BEFORE == AFTER
[ ] git AFTER saved

[ ] reports created
[ ] COMPLETED only after validity checks
```

---

# 182. Краткая инструкция ИИ-агенту

```text
Implement NEW V10 / SRR.
DO NOT MODIFY V1–V9.

Goal:
replace H2 final utility/harm decision layer only.

Keep exact:
B0
top-K
QA
H2 runtime feature inputs.

Do NOT add:
Agreement
new PairLEX
SEM
new neural model
Q+A+chunk support
dataset/language/aspect rules.

PRIMARY S2:

Special case:
K=1 -> return B0 directly.
Exclude K=1 from SWITCH/RANK/RISK training.
Do not run SWITCH/RANK/RISK inference.
Keep K=1 in overall quality/evidence metrics.

1. SWITCH
L2 LogisticRegression.
Target:
exists alternative with better Token F1 than B0.

2. RANK
Pairwise HistGradientBoostingClassifier.
Target:
candidate i has higher Token F1 than j.
PAIR_MIN_GAP=1.0.
Symmetric pairs.
Features:
[x_i-x_j, q].
At inference predict both directions for every pair and symmetrize:
p_ij = 0.5 * (p_raw(i>j) + 1 - p_raw(j>i)).
Rank alternatives by mean symmetrized win probability,
including comparisons against B0 in the rank score.

3. ANSWER-RISK
L2 LogisticRegression.
Target:
candidate F1 < B0 F1.

4. EVIDENCE-RISK
L2 LogisticRegression.
Target:
SAFE_EVIDENCE vs UNSAFE_EVIDENCE.
HN is only a subgroup of UNSAFE.
Exclude ambiguous/unresolved/missing labels.
Unknown non-null evidence label -> FAIL.
Require adequate class/question counts in every INNER_TRAIN and outer refit.

Inference:
if p_switch < tau_switch:
    B0
else:
    rank alternatives
    select highest-ranked candidate with:
        p_answer_harm <= tau_answer
        p_evidence_harm <= tau_evidence
    if none:
        B0

Use 4-fold document-grouped inner OOF.
Use identical inner splits across modules and model seeds.
Tune all model hyperparameters on TRAIN only.
Tune thresholds on OOF only.
Treat OOF post-selection performance as tuning evidence, not scientific estimate.

Threshold objective:
maximize final OOF Token F1 subject to:
ΔHit1 vs B0 >= -0.50 pp
ΔHN1 vs B0 <= +1.00 pp
selected-positive loss <=1.0 pp where available.

Refit on full outer TRAIN/CAL.
Evaluate outer EVAL once.

Primary comparison:
S2 vs exact H2.

Key mechanism metric:
Final Positive Capture.

Mechanism target:
+10 pp vs H2 on >=3/4:
SciER384
SciER128
RuSERRC
SciMDIX

Quality target:
family-balanced ΔF1(S2-H2) >= +0.50
no cell point delta < -0.25
LCB95 > -0.50 all cells
evidence safety PASS
non-degeneracy PASS.

No direct evidence-support neural model in V10.
If evidence discrimination remains the bottleneck,
that is a separate frozen V11.
```
