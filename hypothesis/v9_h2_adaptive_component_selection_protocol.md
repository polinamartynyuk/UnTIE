# V9 / H2-ACS — Adaptive Component Selection over H2
## Полный протокол следующего development-эксперимента для ИИ-агента, пишущего код

**Статус:** новая версия после V4.4 → V4.5 → V5 → V6 → V7 → V8  
**Тип:** development experiment  
**Основная цель:** формально проверить `H2 = V7 S2 + FullLEX` как новый универсальный кандидат и безопасно улучшить его за счёт TRAIN-only выбора дополнительных reader-компонентов `Agreement` и `PairLEX`, не используя ручные правила по языку, датасету, аспекту или K.

**Final reviewed revision:** nested selection и tuning полностью разделены; optional configuration выбирается одна на `(cell, outer_fold)` по seed-averaged TRAIN-only `SELECT_OOF`; `SELECT_FOLDS=4` и split manifest общие для всех configs/seeds; exact H2 target/grids и optional feature-group semantics зафиксированы; новые cross-group interactions запрещены; ACS обязан сохранять F1, evidence-safety и non-degeneracy относительно H2; confirmation повторяет ту же трёхseedовую selection procedure.

---

# 0. Рабочее название

```text
V9 / H2-ACS
H2 Adaptive Component Selection
```

Русская формулировка:

> адаптивный выбор дополнительных reader-компонентов поверх универсального H2-ядра.

---

# 1. Основная научная идея

V8 показал, что:

```text
H2 = V7 S2 + FullLEX
```

дал положительный point estimate относительно B0 на всех пяти protocol cells:

```text
SciREX      +5.9568
SciER128    +2.1428
SciER384    +0.0725
RuSERRC     +0.4484
SciMDIX     +2.7878
```

Это первая конфигурация поздней универсальной линии, которая одновременно имеет:

```text
+,+,+,+,+
```

на всех пяти cells.

Однако H2 в V8 был абляцией, а не заранее выбранным primary-кандидатом.

V9 должен исправить именно это.

---

# 2. Второй ключевой результат V8

Добавление `Agreement`:

```text
H3 = H2 + Agreement
```

дало incremental effect:

```text
SciREX      +2.9591
SciER128    +2.6609
SciER384    -0.2436
RuSERRC     -0.2207
SciMDIX     -0.5867
```

Следовательно:

```text
Agreement полезен не всегда.
```

Его нельзя оставлять обязательным universal component.

---

# 3. Почему Agreement нельзя включать вручную по dataset

Запрещено:

```text
if dataset in {SciREX, SciER128}:
    use Agreement
else:
    disable Agreement
```

Это post-hoc dataset-specific assembly.

V9 должен использовать одно правило:

> компонент включается только тогда, когда его добавочная downstream-полезность подтверждена внутри TRAIN/CAL.

---

# 4. Почему PairLEX рассматривается отдельно

V8/V7 показали, что `PairLEX` способен давать дополнительный сигнал, особенно на части RU-наборов.

Но его вклад измерялся в другой архитектурной ветви.

Поэтому V9 не предполагает, что PairLEX обязательно полезен.

Он рассматривается как:

```text
predeclared optional component group.
```

---

# 5. Главная архитектура

```text
DOCUMENT
  ↓
historical B0
  ↓
top-K
  ↓
frozen QA on top-K
  ↓
H2 core:
    exact V7 S2 mechanics
    +
    exact historical FullLEX
  ↓
optional groups:
    Agreement
    PairLEX
  ↓
TRAIN-only component selector
  ↓
selected H2 configuration
  ↓
expected-utility + harm-safe decision
  ↓
final chunk
  ↓
final answer
```

---

# 6. Принцип V9

H2 всегда является допустимым fallback.

Дополнительные компоненты:

```text
не обязаны использоваться.
```

Если TRAIN/CAL не показывает устойчивой пользы:

```text
selected configuration = H2.
```

---

# 7. Что считается одной архитектурой

На всех пяти cells одинаковы:

```text
candidate universe
question policy
K policy
upstream models
QA procedure

H2 core
FullLEX definition

Agreement definition
PairLEX definition

utility target
harm target

component candidate set
component-selection rule

regularization procedure
threshold-selection rule

metrics
success criteria
```

Различаться могут:

```text
fitted model coefficients
selected optional component group
thresholds
```

если они выбраны только по соответствующей TRAIN/CAL части.

---

# 8. Primary mode

```text
LOCAL
```

Selection/fitting выполняется отдельно внутри TRAIN/CAL конкретной cell.

Это:

```text
parameterized architectural universality
```

а не shared-parameter universality.

---

# 9. Dataset cells

Обязательно:

```text
EN:
- SciREX
- SciER 384/50
- SciER 128/24

RU:
- RuSERRC
- SciMDIX
```

---

# 10. SciER family

```text
SciER 384/50
SciER 128/24
```

— две protocol cells одной corpus family.

Это учитывать:

```text
при global summary
при bootstrap family aggregation
при leakage control
```

---

# 11. Development status

Все пять cells уже участвовали в построении гипотез V4.4–V8.

Поэтому:

```text
V9 = DEVELOPMENT
```

Даже если nested CV выполнен без утечек.

После freeze потребуется:

```text
independent untouched confirmation.
```

---

# 12. Historical protection

Строго read-only:

```text
V1
V2
V3
V4
V4.1
V4.2
V4.3
V4.4
V4.5
V5
V6
V7
V8
```

---

# 13. Запрет изменения history

Не изменять:

```text
historical source code
configs
artifacts
caches
checkpoints
reports
frozen contracts
predictions
```

---

# 14. Новый namespace

Создать:

```text
experiments/qa_chunk_selection_v9/
configs/qa_chunk_selection/v9_h2_acs/
artifacts/qa_chunk_selection_v9_h2_acs/
logs/qa_chunk_selection_v9/
```

---

# 15. Write guard

Любая попытка записи в historical paths:

```text
raise RuntimeError
```

---

# 16. Git audit

До coding/run сохранить:

```bash
git status --porcelain
git rev-parse HEAD
git diff
```

После run:

```text
тот же audit.
```

---

# 17. Запрещённые git operations

Не использовать:

```text
git reset
git clean
git checkout -- .
```

---

# 18. Historical hashes

До run:

```text
historical_hashes_before.json
```

После:

```text
historical_hashes_after.json
```

Требование:

```text
before == after
```

Иначе run invalid.

---

# 19. Canonical protocol

V9 использует exact canonical protocol V8-HRR.

Не менять:

```text
B0
question policy
candidate universe
K
chunking
answer normalization
metrics
```

---

# 20. Почему canonical protocol фиксирован

Цель V9:

> проверить H2 и component selection.

Нельзя одновременно менять:

```text
ranker
question
chunks
K
feature logic.
```

---

# 21. Exact H0

```text
H0 = canonical historical B0
```

Exact replay обязателен.

---

# 22. Exact H1

```text
H1 = exact V7 S2 replay
```

Использовать exact V8 implementation/artifacts.

---

# 23. Exact H2

```text
H2 = exact V8 HRR H2
   = H1 + exact historical FullLEX
```

V9 не переопределяет H2.

---

# 24. H2 contract

До coding new features восстановить и зафиксировать:

```text
input features
normalization
interactions
utility target
utility model
harm model
hyperparameter grids
safe-switch rule
tie-break
fallback semantics
```

из V8 artifacts/config/code.

---

# 25. Запрет приблизительного H2

Если exact H2 implementation не воспроизводится:

```text
STOP
H2_REPLAY_FAILED
```

Не создавать «похожий» H2.

---

# 26. H2 replay criterion

На всех пяти cells:

```text
stored V8 H2 aggregate F1
vs
replayed H2 F1
```

должны совпадать exact/within floating tolerance.

Также проверить:

```text
selected candidate IDs
final normalized answers
switch flags
```

---

# 27. H3 replay

Обязательно exact replay:

```text
H3 = H2 + exact V4.4 reader Agreement
```

на canonical V8 protocol.

Это проверяет, что Agreement group восстановлена правильно.

---

# 28. Agreement group

`Agreement` должен быть **ровно тем feature-group delta**, который добавлялся в V8 H3 относительно H2.

Group `A` включает:
- сами Agreement-derived columns;
- только те Agreement-related transformations/interactions, которые уже входили в exact V8 H3.

Не добавлять:
```text
новый agreement signal
новую normalization
новую interaction
```

Иными словами:

```text
CA feature schema = exact H3 feature schema.
```

---

# 29. Agreement audit

Сохранить:

```text
feature names
formula
normalization
candidate-level semantics
question-level semantics
source implementation
```

в:

```text
audit/agreement_group.json
```

---

# 30. PairLEX group

`PairLEX` = exact V7 **PairLEX-derived runtime feature group**, переносимый как дополнительная feature-group в H2.

В group `P` входят только:
- PairLEX value;
- exact PairLEX-derived deterministic transformations, если они были частью frozen V7 feature definition.

В group `P` **не входят**:
```text
pre-QA admission
candidate queue/order policy
progressive stopping
V7 routing logic
```

Не подменять PairLEX через `FullLEX` и не пересчитывать его с другой reference answer.

---

# 31. PairLEX reference semantics

Если V7 PairLEX использовал:

```text
predicted answer of B0 top1
```

как reference,

V9 использует тот же reference.

---

# 32. PairLEX audit

Сохранить:

```text
formula
token normalization
reference answer semantics
empty-answer policy
value range
```

в:

```text
audit/pairlex_group.json
```

---

# 33. No new neural model

V9 не добавляет:

```text
new LM
new reranker
new encoder
new QA reader
new semantic model
```

---

# 34. No new neural forwards

Agreement и PairLEX должны вычисляться из уже имеющихся:

```text
top-K QA outputs
scores
```

V9 не должен увеличивать число QA-forward относительно H2.

---

# 35. Candidate configurations

Ровно четыре:

```text
C0 = H2

CA = H2 + Agreement

CP = H2 + PairLEX

CAP = H2 + Agreement + PairLEX
```

---

# 36. Почему ровно четыре

Это полный факторный набор двух optional groups:

```text
Agreement ∈ {off,on}
PairLEX  ∈ {off,on}
```

Не добавлять дополнительные combinations после просмотра EVAL.

---

# 37. CA identity

```text
CA
```

должен exact-replay V8 H3 при одинаковой meta/hyperparameter процедуре.

Если не replay:

```text
STOP
AGREEMENT_INTEGRATION_MISMATCH
```

---

# 38. CP

Новая конфигурация.

Она добавляет PairLEX к exact H2.

---

# 39. CAP

Новая конфигурация.

Она добавляет к H2 точное объединение feature-groups:

```text
Agreement + PairLEX
```

Правило:

```text
CAP feature columns = H2 ∪ A ∪ P
```

Запрещено автоматически создавать новые cross-group interactions вроде:

```text
Agreement × PairLEX
```

если такой interaction не входил в уже frozen source feature-groups.

Новая cross-group interaction была бы новой гипотезой и требует отдельной версии.

---

# 40. Feature integration principle

Optional groups добавляются в тот же feature/model pipeline H2.

Не создавать:

```text
отдельный heuristic score
отдельный router
отдельный language branch.
```

---

# 41. Utility model family

Использовать exact H2 utility-model family.

Не менять model class.

---

# 42. Harm model family

Использовать exact H2 harm-model family.

Не менять model class.

---

# 43. Hyperparameter grids

Использовать exact H2 grids из frozen V8 contract:

```text
Ridge alpha:
0.1
1
10
100

harm LogisticRegression C:
0.01
0.1
1.0

tau_utility:
0
0.25
0.5
1
2

tau_harm:
0.1
0.2
0.3
0.4
0.5
```

Другие fixed H2 parameters также должны быть восстановлены exact из V8 contract/code.

Не расширять grids после просмотра V8/V9 outer EVAL.

---

# 44. Utility target

Exact H2 target из V8:

для B0 top-1 answer `A0` и candidate answer `Ai`:

\[
u_i =
TokenF1(A_i,G)-TokenF1(A_0,G)
\]

где Token F1 — exact canonical historical evaluator в шкале `0..100`.

Training weighting также exact H2:

если question имеет `m` alternatives:

\[
w_i = 1/m
\]

то есть total training weight каждого question равен `1`.

Не переопределять target, anchor или weighting в V9.

Источник истины для проверки реализации:

```text
V8 frozen contract + code + artifacts
```

---

# 45. Harm target

Exact H2 target:

\[
h_i = 1[u_i < 0]
\]

Harm classifier:

```text
L2 LogisticRegression
solver = liblinear
class_weight = None
```

Использовать exact H2 fitting/degenerate-class behavior из V8.

Не переопределять.

---

# 46. Component selection level

Optional configuration выбирается **один раз для каждого**:

```text
(cell, outer_fold)
```

а не отдельно для каждого seed.

Все три seeds используются только как повторные stochastic realizations внутри TRAIN-only оценки.

После выбора одной configuration она применяется ко всем seeds данного `(cell, outer_fold)`; сами meta-models каждого seed по-прежнему fit независимо.

Это предотвращает случайное изменение состава метода от seed к seed.

---

# 47. Нельзя выбирать component per question

Запрещено:

```text
query-level Agreement gate
query-level PairLEX gate
```

V7 уже показал слабость query-level admission.

V9 использует:

```text
outer-fold/corpus-fit-level component selection.
```

---

# 48. Seed isolation и seed-averaged selection

Seeds:

```text
42
43
44
```

никогда не объединяются как независимые training rows.

Для каждого seed:

```text
upstream/meta models fit separately
predictions generated separately
```

Для **component selection** после получения honest selection-OOF predictions:

1. для каждого question вычислить metric contribution отдельно для seeds `42/43/44`;
2. усреднить contribution внутри question по seeds;
3. только после этого агрегировать по questions/documents.

Таким образом:

```text
seed ≠ independent sample
```

но компонент выбирается устойчиво по среднему поведению трёх stochastic realizations.

---

# 49. Outer CV

Exact canonical V8 folds:

```text
5 document-grouped outer folds.
```

Outer EVAL полностью sealed **для V9 selector/model code** до окончательного выбора component configuration и fit final fold-models.

Важно: эти outer EVAL partitions уже были просмотрены в предыдущих версиях V4.4–V8 на уровне исследовательской программы. Поэтому они не являются independent confirmation data. Здесь `sealed` означает только отсутствие доступа нового V9 training/selection algorithm к соответствующим labels/results.

---

# 50. Три уровня CV внутри V9

V9 требует **трёх логических уровней**:

```text
OUTER CV
    ↓
COMPONENT-SELECTION CV
    ↓
CONFIG-TUNING / META CV
```

Причина:

> нельзя подбирать hyperparameters/thresholds конфигурации и затем на тех же OOF predictions оценивать, должна ли эта конфигурация быть выбрана.

Это создало бы optimistic component-selection bias.

Для всех пяти заранее известных cells фиксировать:

```text
SELECT_FOLDS = 4
```

document-grouped folds.

Fallback `3/2 folds` в V9 **не использовать**: объём document groups во всех пяти cells достаточен, поэтому изменение числа selection folds создало бы ненужную дополнительную степень свободы.

Требования к SELECT splits:

```text
- использовать тот же document-grouped splitter family/helper, что canonical V8 meta-CV;
- n_splits = 4;
- SELECT_SPLIT_SEED = 42, если helper принимает random_state;
- один и тот же split assignment для C0/CA/CP/CAP;
- один и тот же split assignment для model seeds 42/43/44;
- grouping только по canonical document id;
- split создаётся детерминированно до model fitting;
- split manifest и SHA-256 сохраняются в audit;
- SELECT splits строятся только внутри соответствующего outer TRAIN/CAL.
```

Если canonical V8 grouped splitter является детерминированным и не принимает `random_state`, не добавлять искусственную рандомизацию; использовать exact helper и зафиксировать manifest.

---

# 51. CONFIG-TUNING внутри каждого SELECT fold

Для каждого selection fold `s`:

```text
SELECT_TRAIN_s
SELECT_VALID_s
```

На `SELECT_TRAIN_s` каждая configuration:

```text
C0
CA
CP
CAP
```

проходит **полную exact H2 tuning procedure**, включая:

```text
META_FEATURE_OOF
META_PRED_OOF для tuning
utility-model hyperparameters
harm-model hyperparameters
switch thresholds
```

с использованием только документов `SELECT_TRAIN_s`.

После выбора/tuning parameters configuration refit на `SELECT_TRAIN_s` и один раз применяется к `SELECT_VALID_s`.

`SELECT_VALID_s` не участвует в tuning этой configuration.

---

# 52. SELECT_OOF — единственный источник component-selection решения

Предсказания всех held-out `SELECT_VALID_s` объединяются в:

```text
SELECT_OOF
```

для каждой configuration и seed.

Требования:

```text
every SELECT_OOF row was predicted by a configuration
whose model/hyperparameters/thresholds were selected without that row
```

Именно `SELECT_OOF`, а не tuning `META_PRED_OOF`, используется для:

```text
component selection
EPS_SELECT
fold-sign stability
incremental evidence safety
activity criterion
```

---

# 53. Что нельзя использовать для component selection

Нельзя использовать:

```text
outer EVAL labels
outer EVAL Token F1
outer EVAL switch outcomes
outer EVAL aspect analysis
outer EVAL K analysis
```

Также нельзя использовать:

```text
CONFIG-TUNING OOF metrics
```

как итоговую оценку полезности component configuration.

Они служат только для настройки самой configuration внутри `SELECT_TRAIN`.

---

# 54. Component selection objective

Выбор configuration выполняется по:

```text
final downstream Token F1
```

на seed-averaged `SELECT_OOF`.

Не выбирать по:

```text
Ridge MAE
coefficient magnitude
feature AUROC
R²
CONFIG-TUNING score
```

---

# 55. Почему downstream objective

V8 показал:

```text
positive Agreement coefficient
≠
positive final F1 effect.
```

Поэтому component selection должен оптимизировать конечный pipeline outcome на честном held-out TRAIN-only уровне.

Важно для интерпретации:

```text
Δ(CA-C0), Δ(CP-C0), Δ(CAP-C0)
```

являются **system-level incremental effects**. Каждая configuration проходит собственный nested tuning и refit, поэтому эти разницы нельзя трактовать как чистый причинный эффект только одного feature при фиксированных остальных параметрах.

---

# 56. Base comparator внутри TRAIN

Все optional configurations сравниваются с:

```text
C0 = H2
```

на тех же `SELECT_OOF` questions/documents и с теми же seeds.

Все incremental metrics вычислять **paired**.

---

# 57. Minimum incremental gain

Predeclare:

```text
EPS_SELECT = +0.20 Token F1
```

Optional configuration eligible only if:

\[
F1_{SELECT\_OOF}(C_j)-F1_{SELECT\_OOF}(C0) \ge 0.20
\]

после усреднения seed contributions внутри question.

---

# 58. Почему EPS_SELECT = 0.20

Optional component не должен включаться из-за микроскопического шумового выигрыша.

`0.20 F1`:

```text
мал по отношению к крупным useful effects V8
но вводит explicit practical margin над нулём.
```

Это preregistered V9 margin, а не статистическая граница значимости.

В отчёте обязательно привести sensitivity diagnostic для:

```text
EPS = 0.0 / 0.1 / 0.2 / 0.3
```

**только как diagnostic**; primary selection всегда использует `0.20`.

---

# 59. Selection-fold sign stability

Optional configuration eligible only if seed-averaged delta на held-out selection folds имеет знак:

```text
>= 0
```

минимум:

```text
3 из 4 SELECT folds
```

`SELECT_FOLDS=4` фиксированы для всех пяти cells.

Это относится к `SELECT_VALID` predictions, а не к CONFIG-TUNING folds.

---

# 60. Worst-selection-fold guard

Также:

```text
min SELECT-fold ΔF1(Cj-C0) >= -0.50
```

Token F1.

Это защищает от configuration с хорошим средним, но одним резко отрицательным TRAIN-only held-out fold.

---

# 61. Incremental evidence safety budget

Поскольку H2 уже является сильной базой, optional components получают только половину historical safety budget.

Relative to C0 на seed-averaged `SELECT_OOF`:

```text
ΔHit@1 >= -0.25 pp
ΔHN@1 <= +0.50 pp
```

Все evidence deltas paired на одинаковых questions.

---

# 62. Absolute evidence safety

Optional configuration также должна проходить standard safety relative to H0 на `SELECT_OOF`:

```text
ΔHit@1 >= -0.50 pp
ΔHN@1 <= +1.00 pp
```

Если конкретная evidence metric неприменима для cell:

```text
use exact canonical historical fallback / transition definition
```

из H2 contract; не придумывать новую метрику.

---

# 63. Why stricter incremental safety

Optional component:

```text
не является необходимым.
```

Следовательно, нет причины тратить весь historical safety budget ради небольшого incremental gain.

---

# 64. Minimum stable changed-question activity

Optional configuration eligible only if её `SELECT_OOF` final predictions отличаются от C0 минимум на:

```text
max(5 unique questions, ceil(1% TRAIN/CAL questions))
```

Причём вопрос считается `changed` только если configuration отличается от C0 минимум в:

```text
2 из 3 seeds.
```

Это предотвращает выбор component по stochastic single-seed switches.

---

# 65. Почему activity criterion нужен

Он защищает от выбора component по:

```text
1–2 случайным вопросам
```

с большим F1 jump, а также от случайных seed-specific переключений.

---

# 66. Eligible configurations

```text
C0
```

eligible всегда.

Для:

```text
CA
CP
CAP
```

нужно одновременно выполнить Sections 57–64 на `SELECT_OOF`.

---

# 67. Selection rule

Среди eligible optional configurations:

```text
choose maximum seed-averaged SELECT_OOF Token F1.
```

Результат выбора — **одна configuration на `(cell, outer_fold)`**.

---

# 68. Complexity tie-break

Если разница Token F1 между top configurations:

```text
< 0.10 F1
```

выбрать configuration с меньшим числом optional groups.

Complexity:

```text
C0  = 0
CA  = 1
CP  = 1
CAP = 2
```

---

# 69. Equal-complexity tie-break

Если complexity одинаков:

1. lower seed-averaged HN@1;
2. higher seed-averaged Hit@1;
3. lower final switch rate;
4. deterministic lexical order of config id.

---

# 70. No eligible optional configuration

Если:

```text
CA/CP/CAP
```

не проходят eligibility:

```text
SELECT C0.
```

---

# 71. Selected configuration manifest

Для каждого:

```text
cell
outer fold
```

сохранить:

```text
selected_config
seed-averaged SELECT_OOF Token F1
Δ vs C0
per-seed Δ diagnostics
selection-fold sign vector
worst selection-fold Δ
Hit1 Δ
HN1 Δ
stable changed-question count
selection reason
```

---

# 72. Final refit after component selection

После выбора одной configuration для `(cell, outer_fold)`:

1. для каждого seed отдельно применить exact H2 tuning procedure на **всём outer TRAIN/CAL** для выбранной feature configuration;
2. refit models на всём allowed outer TRAIN/CAL;
3. freeze selected configuration, coefficients, hyperparameters и thresholds;
4. evaluate outer EVAL exactly once.

Outer EVAL никак не участвует в configuration selection.

---

# 73. Outer EVAL prohibition и статус evidence

Никакого повторного выбора после outer EVAL.

При этом все V9 outer-CV результаты остаются:

```text
DEVELOPMENT EVIDENCE
```

поскольку сами corpora/folds уже использовались для hypothesis generation в V4.4–V8.

V9 outer EVAL защищён от **algorithmic leakage**, но не является новым independent dataset-level confirmation.

---

# 74. Primary V9 systems

```text
S0 = H0 B0

S1 = FIXED-H2
     exact H2
     eligible final candidate

S2 = H2-ACS
     adaptive component selection
     eligible final candidate
```

---

# 75. Secondary fixed diagnostics

Сохранить также:

```text
D-A  = fixed CA
D-P  = fixed CP
D-AP = fixed CAP
```

Они:

```text
diagnostic only
```

и не могут post-hoc заменить S1/S2.

---

# 76. Статус FIXED-H2

В V8 H2 был secondary ablation.

V9 заранее объявляет его:

```text
formal frozen development candidate / anchor.
```

Но exact replay H2 на тех же canonical folds **не является новой проверкой гипотезы** и не добавляет независимого statistical evidence.

Его функции в V9:

1. reproducibility check;
2. frozen comparator для ACS;
3. возможный final development candidate, если ACS не оправдывает усложнение.

Если итогом выбран S1/H2, следующий научно значимый шаг — untouched confirmation, а не интерпретация V9 replay как новой репликации.

---

# 77. Why ACS must justify complexity

S2 сложнее S1.

Поэтому S2 не становится final candidate только потому, что имеет немного выше mean F1.

---

# 78. Development decision gate для S1 / FIXED-H2

Для целей выбора frozen development candidate S1 считается прошедшим gate, если уже наблюдаемые/replayed canonical metrics удовлетворяют:

```text
ΔTokenF1(S1-S0) > 0
```

на всех пяти cells,

и:

```text
LCB95 > -0.5 F1
```

на всех cells,

и:

```text
evidence safety PASS
```

на всех cells,

и:

```text
NON_DEGENERATE PASS
```

на всех cells.

---

# 79. S1 strict criterion

```text
LCB95(S1-S0) > 0
```

на всех пяти cells.

Это strong development pattern, но не independent confirmation.

---

# 80. Universal gate для S2 / H2-ACS

S2 также должен выполнить:

```text
ΔTokenF1(S2-S0) > 0
```

на всех five cells,

```text
LCB95 > -0.5
```

на всех cells,

```text
evidence safety PASS
```

на всех cells,

и:

```text
NON_DEGENERATE PASS
```

на всех cells.

---

# 81. ACS no-degradation gate relative to H2

Поскольку S1/H2 уже является полноценным candidate, более сложный ACS может заменить его только при **отсутствии point-estimate деградации**:

```text
ΔTokenF1(S2-S1) >= 0.00
```

на каждой из пяти cells.

Дополнительный uncertainty guard:

```text
LCB95(S2-S1) > -0.50 F1
```

на каждой cell.

Кроме answer-quality preservation, ACS обязан сохранить incremental evidence-safety относительно H2 на outer-CV predictions:

```text
ΔHit@1(S2-S1) >= -0.25 pp
ΔHN@1(S2-S1) <= +0.50 pp
```

на каждой applicable cell.

Если хотя бы одна cell имеет отрицательный point estimate S2-S1 или нарушает incremental evidence budget:

```text
ACS_PRESERVATION = FAIL
```

и S2 не может заменить S1 как final development candidate.

---

# 82. ACS added-value gate

Кроме no-degradation, S2 должен давать practically meaningful aggregate gain:

```text
family-balanced mean Δ(S2-S1) >= +0.20 F1
```

и минимум:

```text
2 из 5 cells
```

должны иметь строго положительный point estimate:

```text
S2-S1 > 0.
```

Если S2 только воспроизводит H2 без добавочной пользы:

```text
final candidate = S1
```

при условии, что S1 проходит universal gate.

---

# 83. Strict ACS success

Самый желательный результат:

```text
S2-S1 > 0
```

на всех пяти cells.

---

# 84. Final candidate selection

Predeclared hierarchy:

### Case A
S2 passes:
- universal gate,
- preservation gate,
- added-value gate.

Then:

```text
FINAL DEVELOPMENT CANDIDATE = S2.
```

### Case B
S2 fails added-value/preservation,
but S1 passes universal gate.

Then:

```text
FINAL DEVELOPMENT CANDIDATE = S1.
```

### Case C
Neither passes universal gate:

```text
NO UNIVERSAL DEVELOPMENT CANDIDATE.
```

---

# 85. Нельзя выбрать diagnostic D-A/D-P/D-AP

Даже если одна diagnostic system имеет лучший observed F1.

Если S1 или S2 выбирается как final development candidate по outer-CV результатам, его итоговый outer-CV score является:

```text
selected development estimate
```

а не unbiased confirmatory estimate, поскольку сам выбор между S1/S2 сделан после просмотра этих development results. Независимая оценка переносится только на untouched confirmation.

---

# 86. Non-degeneracy

Для S1/S2 считать stable final switches relative S0.

Stable:

```text
same question differs from B0 in >=2/3 seeds.
```

Minimum:

```text
max(3, ceil(1% questions)).
```

Для объявления S1 или S2 **активным final development candidate** non-degeneracy должна выполняться на каждой cell.

Если quality gate формально проходит, но second stage не достигает minimum activity:

```text
NON_DEGENERATE = FAIL
```

и нельзя утверждать, что добавочный компонент продемонстрировал содержательную работу на этой cell.

Для S2 отдельно сохраняется component-selection activity relative S1.

---

# 87. Component-selection activity

Отдельно report:

```text
how often selector chose C0 / CA / CP / CAP
```

по:

```text
5 outer folds per cell
cells
languages
```

Поскольку configuration выбирается одна на `(cell, outer_fold)`, seeds не считаются отдельными selection events.

---

# 88. Expected qualitative pattern is NOT a rule

Можно анализировать, получился ли pattern:

```text
Agreement чаще EN
PairLEX чаще RU
```

но нельзя использовать такой pattern как training rule.

---

# 89. Coefficients

Для selected configuration сохранять standardized coefficients:

```text
FullLEX
Agreement if selected
PairLEX if selected
other H2 features
```

---

# 90. Coefficient analysis

Report:

```text
median
IQR
sign consistency
```

по folds/seeds.

Не использовать coefficients для component selection.

---

# 91. Primary quality metric

```text
Token F1
```

exact canonical evaluator.

---

# 92. Secondary answer metrics

Сохранить exact historical contract:

```text
Char F1
ROUGE-L
Unicode ROUGE-L
EM
Lemma F1 RU
```

BERTScore:

```text
only if frozen historical contract already supports it.
```

---

# 93. Evidence metrics

```text
Hit@1
Hit@K
HN@1
HN@K
F1|selected positive
Amb@1/GSRN@1 where applicable
```

---

# 94. Switch metrics

```text
switch rate

good
bad
neutral

mean actual utility
median utility

large gain:
u >= +5

large harm:
u <= -5
```

---

# 95. Component incremental metrics

На outer EVAL diagnostic only:

```text
D-A - S1
D-P - S1
D-AP - S1
```

Нельзя использовать их для selection после факта.

---

# 96. Selection correctness diagnostics

Для S2 report:

```text
selected config on TRAIN
best config on EVAL  # oracle diagnostic only
```

и:

```text
selector regret =
EVAL(best diagnostic config) - EVAL(selected config)
```

---

# 97. Oracle diagnostic restriction

Outer-EVAL oracle config:

```text
NEVER eligible
```

Используется только для оценки:

> есть ли headroom у component selector.

---

# 98. Fold-selection stability

Per cell report:

```text
selected config frequency
```

по 5 outer-fold selections.

Дополнительно report seed-level SELECT_OOF deltas, чтобы показать stochastic sensitivity, хотя selection выполняется по seed-averaged metric.

---

# 99. Component selection entropy

Secondary diagnostic:

```text
entropy(selected_config distribution over 5 outer folds)
```

Высокая entropy означает чувствительность component set к составу TRAIN/CAL документов.

Не считать seeds независимыми selection events.

---

# 100. Aspect analysis

Post-hoc only.

Не использовать:

```text
aspect
```

как component-selection feature.

---

# 101. K analysis

Post-hoc only.

Не менять K и не создавать K-dependent selection rule внутри V9.

---

# 102. Почему K-dependent rule пока запрещён

V8 показал интересные K-зависимости, но добавление K-specific policy сейчас увеличит число одновременно меняемых гипотез.

V9 проверяет только:

```text
component selection.
```

---

# 103. POOLED/LOCO

Не являются обязательными primary experiments V9.

Причина:

```text
V8 уже показал слабый перенос shared parameters.
```

---

# 104. Optional POOLED diagnostic

Можно запускать только после completion primary LOCAL и только если предусмотрено ресурсами.

Не влияет на final candidate.

---

# 105. SciER cross-cell embargo

Обязательно для любого combined/pooled diagnostic.

Использовать:

```text
canonical_document_id
full-text document SHA-256
```

---

# 106. Cross-corpus overlap audit

Повторить exact/normalized full-document hash check.

---

# 107. Statistical unit

Seeds не считаются независимыми observations.

Порядок:

1. per-question per-seed;
2. average seeds within question;
3. document-cluster paired bootstrap.

---

# 108. Bootstrap

```text
iterations = 5000
paired = true
cluster = document_id
seed = 42
```

---

# 109. Bootstrap contrasts

Обязательно:

```text
S1 - S0
S2 - S0
S2 - S1
```

---

# 110. SciER bootstrap dependence

SciER 384/128 global family summary должен сохранять shared document lineage.

Не bootstrap-ить их как независимые corpora.

---

# 111. Global family-balanced metric

Families:

```text
SciREX
SciER
RuSERRC
SciMDIX
```

где:

\[
\Delta_{SciER}
=
mean(\Delta_{384},\Delta_{128})
\]

Обязательно вычислить не только point estimate, но и **family-aware bootstrap CI** для:

```text
Global Δ(S1-S0)
Global Δ(S2-S0)
Global Δ(S2-S1)
```

Bootstrap должен:
- resample documents within each corpus family;
- для SciER один раз resample canonical source-document IDs и применить **тот же sampled ID multiset одновременно** к SciER-128 и SciER-384;
- затем вычислить обе cell deltas и их среднее как один SciER-family effect;
- сначала получить family effects;
- затем усреднить четыре families с равными весами.

Не resample SciER-128 и SciER-384 независимо и не объединять все questions пяти cells в один плоский bootstrap.

---

# 112. Resource principle

S1 и S2 должны выполнять:

```text
exact same neural forwards.
```

Agreement/PairLEX не должны добавлять neural model inference.

---

# 113. QA-forward assertion

Для каждого question:

```text
QA_forward_count(S2) == QA_forward_count(S1)
```

Если нет:

```text
IMPLEMENTATION BUG
```

---

# 114. Ranker-forward assertion

Optional components не должны увеличивать B0/Q-A neural forwards, если H2 contract их уже вычисляет.

Если дополнительный forward требуется из-за implementation mistake:

```text
FAIL.
```

---

# 115. Latency resource target relative H2

Engineering target:

```text
median latency S2 <= 1.10 × S1
P95 latency S2 <= 1.15 × S1
```

---

# 116. VRAM target

```text
peak VRAM S2 <= 1.05 × S1
```

---

# 117. Parameters

Trainable meta-parameters Agreement/PairLEX должны быть negligible относительно neural stack.

Report nonetheless.

---

# 118. Qwen comparison

Если exact fresh Qwen runner всё ещё недоступен:

```text
QWEN_FRESH_COMPARISON = BLOCKED
```

Не использовать historical latency с другого hardware как primary numeric comparison.

---

# 119. Strong-ranker RU reference

BGE остаётся contextual reference.

Не менять canonical RU B0 в V9.

---

# 120. Resource benchmark systems

Минимум:

```text
S0 B0
S1 FIXED-H2
S2 H2-ACS
```

Optional:

```text
Qwen
BGE
```

если exact fresh runners доступны.

---

# 121. Fresh-process benchmark

Каждая system:

```text
separate process
```

---

# 122. Benchmark sample

Использовать same frozen sample IDs для S0/S1/S2.

Предпочтительно:

```text
all fold0 questions capped at 100/cell
```

как previous engineering benchmark.

---

# 123. Warm-up

```text
>=10
```

если sample size позволяет.

---

# 124. Repeats

```text
3 timed repeats
```

---

# 125. CUDA synchronization

До/после timed block:

```text
torch.cuda.synchronize()
```

на фактическом device.

---

# 126. Artifact namespace

```text
artifacts/qa_chunk_selection_v9_h2_acs/
  run_<timestamp>/
```

---

# 127. Artifact structure

```text
run/
  status.json
  manifest.json
  COMPLETED.json

  frozen_contract.yaml

  audit/
    git_before/
    git_after/

    historical_hashes_before.json
    historical_hashes_after.json

    h0_replay.json
    h1_replay.json
    h2_replay.json
    h3_replay.json

    fulllex_group.json
    agreement_group.json
    pairlex_group.json

    candidate_universe.json
    split_hashes.json
    select_split_manifests/
    select_split_hashes.json
    model_manifests/

    scier_lineage.json
    cross_corpus_overlap.json

  models/
    S1/
    S2/
    diagnostics/

  selections/
    component_selection_rows.jsonl
    selection_summary.json
    select_fold_predictions/
    tuning_manifests/

  predictions/
    S0/
    S1/
    S2/
    D_A/
    D_P/
    D_AP/

  analysis/
    primary_metrics.json
    paired_bootstrap.json

    component_selection_analysis.json
    selector_regret.json

    switch_analysis.json
    coefficient_analysis.json

    fold_stability.json
    seed_stability.json

    aspect_analysis.json
    k_analysis.json

    global_summary.json
    development_decision.json

  resource_benchmark/
    sample_manifest.json
    raw/
    summary.json
    environment.json

  reports/
    v9_results.md
    component_selection.md
    reproducibility.md
```

---

# 128. Immutable runs

Каждый run immutable.

Bugfix:

```text
new run directory.
```

---

# 129. Atomic writes

Использовать:

```text
tmp file
fsync
atomic rename
```

для critical artifacts.

---

# 130. Cache isolation

V9 writes:

```text
V9 cache only.
```

Historical caches:

```text
read only.
```

---

# 131. Cache key

Минимум:

```text
cell
document
question
candidate
outer fold
inner/meta fold
seed

model hashes
feature version
Agreement version
PairLEX version
```

---

# 132. Runtime/evaluation separation

Prediction row:

```json
{
  "runtime": {},
  "evaluation": {}
}
```

---

# 133. Runtime fields

```text
cell
document_id
canonical_document_id
question_id

outer fold
seed

candidate IDs
B0 scores
QA answers

H2 base features
FullLEX

Agreement values
PairLEX values

selected component config

utility prediction
harm probability
thresholds

final candidate
final answer

forward counts
```

---

# 134. Evaluation-only fields

```text
gold answer
actual Token F1
actual candidate utility
evidence labels
good/bad/neutral
```

---

# 135. Gold leakage test

Runtime cannot contain:

```text
gold
actual utility
oracle config
oracle best candidate
positive evidence label
HN label
```

---

# 136. Selection leakage test

Component selector cannot read:

```text
outer EVAL predictions
outer EVAL metrics
outer EVAL labels.
```

---

# 137. Unit tests — history

```text
test_history_read_only
test_historical_hashes_unchanged
test_no_overwrite
```

---

# 138. Unit tests — replay

```text
test_h0_replay
test_h1_v7_s2_replay
test_h2_v8_replay
test_h3_v8_replay
```

---

# 139. Unit tests — feature groups

```text
test_fulllex_exact_h2
test_agreement_exact_h3
test_agreement_group_contains_only_h3_delta

test_pairlex_exact_v7
test_pairlex_group_excludes_progressive_logic
test_pairlex_reference_semantics

test_CAP_exact_union_no_new_cross_interactions

test_empty_answer_policy
test_feature_orientation
```

---

# 140. Unit tests — candidate configs

```text
test_C0_identity
test_CA_identity_with_h3

test_CP_only_adds_pairlex
test_CAP_adds_exact_union
```

---

# 141. Unit tests — nested OOF

```text
test_outer_eval_never_seen_in_selection

test_select_fold_document_grouping
test_select_folds_exactly_four
test_select_splitter_matches_v8_family
test_select_split_seed_rule
test_select_splits_same_all_configs
test_select_splits_same_all_seeds
test_select_split_hash_stability
test_select_valid_not_used_in_config_tuning

test_config_tuning_meta_feature_oof
test_config_tuning_meta_prediction_oof

test_select_oof_complete_once_per_question
test_select_oof_hyperparams_chosen_without_valid_row

test_seed_isolation
test_seed_averaged_selection_not_seed_concatenation
```

---

# 142. Unit tests — selector

```text
test_selector_uses_select_oof_only
test_selector_one_config_per_cell_outer_fold

test_eps_select
test_selection_fold_sign_rule
test_worst_selection_fold_guard

test_incremental_hit_safety
test_incremental_hn_safety
test_outer_incremental_hit_preservation
test_outer_incremental_hn_preservation

test_stable_changed_questions_across_seeds
test_complexity_tiebreak

test_fallback_to_C0
test_final_candidate_nondegeneracy
```

---

# 143. Unit tests — resource invariance

```text
test_same_qa_forwards_S1_S2
test_same_neural_model_set_S1_S2
test_no_new_neural_model
```

---

# 144. Unit tests — leakage

```text
test_runtime_no_gold
test_selection_no_outer_eval
test_scier_embargo
test_cross_corpus_overlap
```

---

# 145. Smoke test

Before full run:

```text
1 EN cell
1 RU cell
1 outer fold
1 seed
small subset
```

---

# 146. Smoke verification

Check:

```text
H2 replay
Agreement
PairLEX

C0/CA/CP/CAP predictions

component selection
refit
outer-style inference

resource counters
artifact writing
```

---

# 147. Smoke output cannot tune protocol

Нельзя менять:

```text
EPS_SELECT
safety budgets
fold stability rule
candidate configurations
```

по smoke quality.

---

# 148. Frozen contract

До full outer EVAL сохранить:

```text
frozen_contract.yaml
SHA-256
```

---

# 149. Frozen contract content

Обязательно:

```text
dataset cells
B0/questions/K/candidate universe

H2 exact version/hash

Agreement group exact delta schema
PairLEX group exact derived schema
CAP union/no-new-cross-interaction rule

C0/CA/CP/CAP

H2 exact utility target / question weighting
H2 exact harm target
H2 exact hyperparameter grids

EPS_SELECT=0.20
SELECT_FOLDS=4 fixed
SELECT split generation + hashes
CONFIG-TUNING nested procedure
seed-averaged component-selection rule
selection-fold sign rule
worst-selection-fold guard

incremental safety budget
absolute safety budget

activity threshold
tie-breaks

outer/inner folds
seeds

metrics
bootstrap
success criteria

resource targets
```

---

# 150. После freeze запрещено менять

```text
candidate configs
feature groups
selection thresholds
safety thresholds
tie-breaks
model families
hyperparameter grids
success criteria
```

---

# 151. Execution order

```text
A. git/history/hash audit
B. exact H0/H1/H2/H3 replay
C. recover Agreement + PairLEX
D. unit tests
E. smoke
F. freeze contract
G. run fixed S1 / H2 outer pipeline
H. for every outer TRAIN/CAL build nested SELECT_OOF for C0/CA/CP/CAP
I. choose one configuration per (cell, outer_fold), refit each seed, evaluate S2 outer EVAL once
J. run fixed diagnostics D-A/D-P/D-AP under the frozen outer protocol
K. bootstrap/statistics
L. selector-regret analysis
M. resource benchmark
N. final history hash assertion
O. reports
```

---

# 152. H2 replay must precede new training

Если H2 replay fails:

```text
DO NOT CONTINUE TO V9.
```

---

# 153. Technical failure handling

Если bug найден после partial run:

```text
mark run INVALID
preserve it
create new run
rerun all affected outer folds/cells
```

---

# 154. Do not patch one cell only

Если code bug влияет на shared logic:

```text
rerun every affected cell.
```

---

# 155. Development decision report

Автоматически вывести:

```text
S1 universal PASS/FAIL
S2 universal PASS/FAIL

S2 preservation PASS/FAIL
S2 added-value PASS/FAIL

final candidate
reason
```

---

# 156. Required primary table

```text
Cell
H0 F1
S1 H2 F1
Δ S1-H0
CI

S2 ACS F1
Δ S2-H0
CI

Δ S2-S1
CI

Hit1
HN1
stable switches
selected config frequencies
```

---

# 157. Required component-selection table

Per cell:

```text
C0 frequency across 5 outer folds
CA frequency
CP frequency
CAP frequency

mean seed-averaged SELECT_OOF gain when selected
per-seed SELECT_OOF gain diagnostics
outer EVAL gain of selected config

selection regret
```

---

# 158. Required diagnostics table

```text
Cell
D-A Δ vs H2
D-P Δ vs H2
D-AP Δ vs H2
```

Label:

```text
OUTER-EVAL DIAGNOSTIC — NOT ELIGIBLE.
```

---

# 159. Required stability table

```text
Cell
Seed42 Δ
Seed43 Δ
Seed44 Δ

Fold0
Fold1
Fold2
Fold3
Fold4
```

for:

```text
S1
S2
S2-S1
```

---

# 160. Required resource table

```text
Cell
System
QA forwards
ranker forwards
median latency
P95 latency
throughput
VRAM
RSS
parameters
```

---

# 161. Required negative-result transparency

Если ACS не помогает:

не скрывать:

```text
selected optional groups
wrong selections
selector regret
```

---

# 162. Interpretation if selector mostly chooses C0

Допустимый вывод:

> H2 оказался устойчивой конфигурацией, а дополнительные components не давали достаточно стабильной TRAIN-only пользы.

Это не failure of engineering.

---

# 163. Interpretation if Agreement selected mostly EN

Корректно:

> TRAIN-only selection чаще сохраняла Agreement на исследованных EN cells.

Не:

```text
Agreement is an English-language feature.
```

---

# 164. Interpretation if PairLEX selected mostly RU

Корректно:

> PairLEX чаще проходил selection criterion на исследованных RU cells.

Не:

```text
PairLEX is inherently Russian.
```

---

# 165. Independent confirmation trigger

Если final candidate S1 или S2 проходит development decision gate:

```text
freeze architecture immediately.
```

Не интерпретировать V9 как independent confirmation.

Не запускать V10 на тех же cells.

Следующий quality experiment должен использовать untouched confirmation data.

---

# 166. Untouched confirmation

Нужно минимум:

```text
one untouched EN corpus
one untouched RU corpus
```

со схожей thematic extraction постановкой.

---

# 167. Confirmation for LOCAL parameterized architecture

Для каждого untouched corpus:

```text
CONFIRM_TRAIN/CAL
CONFIRM_TEST
```

Fit:

```text
same H2/ACS training procedure
```

только на TRAIN/CAL.

Для ACS confirmation обязательно сохранить:
```text
SELECT_FOLDS = 4
тот же grouped splitter family
тот же SELECT_SPLIT_SEED rule
model seeds = 42,43,44
seed-averaged component selection
тот же EPS_SELECT / stability / safety / activity rules
```

TEST sealed.

---

# 168. No confirmation retuning

На confirmation нельзя менять:

```text
component set
A/P group definitions
SELECT_FOLDS
selection split rule
selection seeds
EPS_SELECT
selection stability rule
safety budgets
activity rule
tie-breaks
model families
hyperparameter grids
```

---

# 169. Dissertation claim if FIXED-H2 wins

Возможная формулировка:

> Разработан единый параметризуемый reader-assisted метод остаточного ранжирования фрагментов, объединяющий ожидаемую полезность вопросно-ответных кандидатов с полным лексическим согласованием извлечённых ответов и ограничением риска ухудшения решения.

---

# 170. Dissertation claim if H2-ACS wins

Возможная формулировка:

> Разработан единый параметризуемый reader-assisted метод, в котором базовый лексически усиленный остаточный механизм дополняется автоматическим выбором reader-компонентов по их добавочной downstream-полезности на обучающей выборке; архитектура и правило выбора компонентов остаются едиными для различных языков и корпусов.

---

# 171. Что нельзя утверждать

До independent confirmation нельзя писать:

```text
universal proven
works on arbitrary scientific corpora
language-independent weights
```

---

# 172. Stop rule

Если:

```text
S1 fails universal gate
AND
S2 fails universal gate
```

не начинать unrestricted V10 feature search на тех же пяти development cells.

---

# 173. Что делать при failure V9

Тогда использовать накопленные результаты для:

```text
applicability-boundary analysis
strong per-corpus evidence
final dissertation method with conditional scope
```

или переходить к новому независимому dataset source.

---

# 174. Почему это должен быть последний широкий development experiment

V4.4–V8 уже многократно использовали те же corpora для hypothesis generation.

Дальнейшее многократное изменение architecture на них увеличивает:

```text
research-program selection bias.
```

V9 должен либо:

```text
freeze candidate
```

либо:

```text
закрыть universal-development search.
```

---

# 175. Pre-run checklist

```text
[ ] V9 namespace created
[ ] history write guard active
[ ] git state saved
[ ] hashes BEFORE saved

[ ] canonical V8 protocol resolved

[ ] H0 replay PASS
[ ] H1 V7-S2 replay PASS
[ ] H2 V8 replay PASS
[ ] H3 V8 replay PASS

[ ] FullLEX exact + audit saved
[ ] Agreement exact H3-delta group
[ ] PairLEX exact derived group
[ ] PairLEX progressive logic excluded
[ ] CAP no-new-cross-interaction rule verified

[ ] C0/CA/CP/CAP frozen

[ ] exact H2 model family frozen
[ ] exact H2 hyperparameter grids frozen

[ ] EPS_SELECT=0.20 frozen
[ ] SELECT_FOLDS=4 frozen
[ ] canonical V8 grouped-splitter family frozen
[ ] SELECT_SPLIT_SEED rule frozen
[ ] SELECT split generation/hashes frozen
[ ] same SELECT splits across configs/seeds verified
[ ] nested CONFIG-TUNING procedure frozen
[ ] seed-averaged selection rule frozen
[ ] selection-fold sign-stability rule frozen
[ ] worst-selection-fold guard frozen

[ ] incremental safety frozen
[ ] absolute safety frozen
[ ] activity threshold frozen
[ ] tie-break frozen

[ ] outer folds frozen
[ ] SELECT folds=4 and split hashes frozen
[ ] config-tuning meta-fold procedure frozen
[ ] seeds frozen

[ ] bootstrap frozen
[ ] universal gates frozen
[ ] final-candidate hierarchy frozen

[ ] resource invariance assertions frozen

[ ] SciER embargo PASS
[ ] cross-corpus overlap audit PASS

[ ] unit tests PASS
[ ] smoke PASS

[ ] frozen contract hash saved
[ ] no full outer-EVAL viewed before freeze
```

---

# 176. Post-run checklist

```text
[ ] all 5 cells complete
[ ] all 5 outer folds complete
[ ] all 3 seeds complete

[ ] S0 complete
[ ] S1 FIXED-H2 complete
[ ] S2 H2-ACS complete

[ ] D-A/D-P/D-AP complete

[ ] component selection manifests complete
[ ] selector regret complete

[ ] Token F1 complete
[ ] answer metrics complete
[ ] evidence metrics complete
[ ] switch metrics complete

[ ] paired bootstrap complete
[ ] SciER coupled family handling verified
[ ] family-aware GlobalDelta CIs complete

[ ] S1/S2 non-degeneracy checked on every cell
[ ] S2 incremental evidence preservation checked
[ ] seed/fold stability complete
[ ] aspect/K diagnostics complete

[ ] resource invariance PASS
[ ] fresh benchmark complete

[ ] development decision generated once

[ ] hashes AFTER saved
[ ] BEFORE == AFTER
[ ] git diff saved

[ ] v9_results.md created
[ ] component_selection.md created
[ ] reproducibility.md created

[ ] COMPLETED only after all validity checks
```

---

# 177. Краткая инструкция ИИ-агенту

```text
Create NEW V9 / H2-ACS.

DO NOT MODIFY V4.4–V8.

Use exact canonical V8 protocol.

First exact-replay:
H0
H1 = V7 S2
H2 = H1 + FullLEX
H3 = H2 + Agreement

If replay fails: STOP.

H2 is now a FORMAL frozen development candidate / comparator.
Its replay is NOT new independent evidence because the same development folds were already seen in V8.

Optional component groups:
A = exact V8 H3-H2 Agreement feature-group delta
P = exact V7 PairLEX-derived feature group ONLY
    (no admission, queue, routing or progressive stopping)

CAP = exact union H2+A+P.
DO NOT create new Agreement×PairLEX cross-interactions.

Candidate configs:
C0  = H2
CA  = H2 + A
CP  = H2 + P
CAP = H2 + A + P

Same model family / target / harm logic / hyperparameter procedure as H2.

For each (cell, outer_fold):
build TRAIN-only nested SELECT_OOF predictions for C0/CA/CP/CAP.

Use exactly 4 document-grouped SELECT folds.
Reuse the canonical V8 grouped-splitter family/helper.
Use SELECT_SPLIT_SEED=42 only if that helper accepts random_state.
The SELECT split assignment must be identical across all configs and seeds
and saved with hashes.

IMPORTANT:
CONFIG-TUNING and COMPONENT-SELECTION must be different held-out levels.
Do NOT tune a config and select that config on the same OOF rows.

For every SELECT fold:
- tune each config only on SELECT_TRAIN using its own nested meta CV
- refit on SELECT_TRAIN
- predict SELECT_VALID once

Concatenate SELECT_VALID predictions into SELECT_OOF.

Keep seeds isolated during fitting.
For selection, average metric contributions within question across seeds.
Choose ONE config per (cell, outer_fold), shared by all three seeds.

Optional config eligible only if:
- seed-averaged SELECT_OOF ΔF1 vs H2 >= +0.20
- >=3/4 SELECT folds have ΔF1 >=0
- worst SELECT-fold ΔF1 >= -0.50
- incremental ΔHit1 >= -0.25 pp
- incremental ΔHN1 <= +0.50 pp
- absolute evidence safety vs B0 passes
- stable changed questions >= max(5,ceil(1% TRAIN questions)), changed in >=2/3 seeds

Choose eligible config with max seed-averaged SELECT_OOF Token F1.
If top configs differ by <0.10 F1:
choose fewer optional groups.

If none eligible:
choose H2.

Refit selected config on all allowed TRAIN/CAL.
Evaluate outer EVAL once.

Primary eligible systems:
S1 = FIXED-H2
S2 = H2-ACS

Diagnostics only:
fixed H2+A
fixed H2+P
fixed H2+A+P

S2 may replace H2 only if:
- universal gate vs B0 passes
- S2-S1 point estimate >=0 on every cell
- LCB95(S2-S1) > -0.50 on every cell
- ΔHit1(S2-S1) >= -0.25 pp on every applicable cell
- ΔHN1(S2-S1) <= +0.50 pp on every applicable cell
- S1/S2 final non-degeneracy passes
- family-balanced gain vs H2 >= +0.20
- at least 2 cells improve strictly vs H2

Otherwise:
if H2 satisfies the development decision gate,
freeze H2 and move directly to untouched confirmation.
Do NOT describe the V9 H2 replay as a new replication.

Do not use dataset names, language, aspects or K
to decide which optional component is active.

Do not add query-level admission/gating.

Agreement/PairLEX must add ZERO neural forwards relative H2.

Treat CA/CP/CAP deltas as SYSTEM-LEVEL incremental effects:
each config is retuned/refit, so do not call them pure causal feature effects.

If neither H2 nor ACS passes:
stop universal-development feature search on these five cells.
```
