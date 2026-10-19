# V8 — Dual-Branch Universal Reader Reranking
## Проверенный план реализации V8-HRR и V8-V44-AF (адаптивное усиление подхода V4.4)

**Статус:** новый development-протокол после V4.4 → V4.5 → V5 → V6 → V7  
**Назначение:** техническая инструкция для ИИ-агента, который будет писать код, запускать эксперимент, формировать артефакты и отчёт  
**Основная цель:** объединить сильные reader-assisted сигналы V4.4 с risk-aware expected-utility логикой V7 S2 и проверить, можно ли получить **положительный прирост на всех пяти protocol cells одной архитектурой**, не вводя новую крупную языковую модель и не изменяя historical upstream models.

**Reviewed version:** исправлено разделение exact V4.4 preservation и нового универсального эксперимента; обе основные ветки теперь сравниваются только в одном canonical protocol. Во второй ветке отдельно моделируются `FullLEX` V4.4 и exact `PairLEX` V7, а различия EN/RU допускаются только через обученные коэффициенты, не через ручные языковые правила.

---

# 0. Центральная идея V8

Результаты V4.4–V7 показывают две разные сильные стороны.

## V4.4

На английских данных reader-assisted reranking давал крупный прирост:

```text
SciREX:
reader/SEM branch >> поздние residual systems

SciER 384/50:
LEX >> поздние systems

SciER 128/24:
SEM/LEX >> поздние systems
```

Следовательно:

```text
в top-K predicted answers содержится сильный сигнал,
который поздние V5–V7 использовали недостаточно полно.
```

## V7

`LOCAL S2` показал положительный point estimate на всех пяти cells:

```text
SciREX      +0.6915
SciER384    +0.2856
SciER128    +0.1629
RuSERRC     +0.2822
SciMDIX     +0.9740
```

Но S2 использовал более узкий набор reader-derived features, чем V4.4.

При этом risk-aware harm model оказался полезным, а progressive admission S3 потерял часть английского выигрыша.

---

# 1. Гипотеза V8

Сильный reader signal V4.4 и safe decision rule V7 решают разные части задачи.

Поэтому:

\[
\boxed{
\text{V4.4 reader evidence signals}
+
\text{V7 expected utility}
+
\text{V7 harm control}
}
\]

может дать более устойчивое улучшение, чем каждая из этих частей отдельно.

Primary V8 не использует pre-QA admission.

---

# 2. Почему admission исключён из primary

V7 показал:

```text
SciREX:
S2 +0.6915
S3 +0.0526

SciER384:
S2 +0.2856
S3 -0.2786
```

Admission AUROC на английских cells был около случайного.

Следовательно:

```text
полезность alternative часто нельзя определить до QA.
```

Primary V8 поэтому:

```text
QA(top-K)
```

и только затем:

```text
reader-assisted scoring
utility/risk decision.
```

---

# 3. Рабочее название

```text
V8-HRR
Hybrid Reader–Risk Reranking
```

Русская формулировка:

> гибридное reader-assisted остаточное ранжирование с оценкой ожидаемой полезности и риска ухудшения.

---

# 4. Primary архитектура

```text
DOCUMENT
   ↓
historical B0 ranker
   ↓
historical top-K
   ↓
frozen QA on all top-K
   ↓
predicted answers A1...AK
   ↓
reader-derived candidate features:
   - B0
   - Q-A
   - V4.4 agreement/consensus
   - V4.4 LEX
   - V4.4 SEM if globally reproducible
   ↓
expected-utility model
   ↓
harm-risk model
   ↓
safe switch decision
   ↓
final chunk
   ↓
final answer
```

---

# 5. Что НЕ делаем в primary V8

Не использовать:

```text
pre-QA admission
progressive candidate queue
aspect interactions
source-support как обязательный signal
attention-derived signals
новую pretrained LM
новый Transformer reranker
новый QA reader
новый semantic encoder
```

---

# 6. Универсальность

Primary universality:

```text
одна архитектура
один feature contract
одни target definitions
одна utility/risk процедура
одни правила выбора thresholds
```

Допускается:

```text
LOCAL fitted coefficients
```

на каждом corpus TRAIN.

Это:

```text
architectural / parameterized universality.
```

---

# 7. Dataset cells

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

SciER 384/50 и 128/24:

```text
one corpus family
```

---

# 8. Development status

Все эти corpora уже использовались в исследовательском цикле.

Поэтому:

```text
V8-A = DEVELOPMENT
```

Даже идеальный nested CV не превращает V8-A в independent confirmation.

---

# 9. Historical protection

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
```

Нельзя изменять:

```text
artifacts
caches
configs
manifests
reports
checkpoints
historical reproduction code
```

---

# 10. Новый namespace

Создать:

```text
experiments/qa_chunk_selection_v8/
configs/qa_chunk_selection/v8_hrr/
artifacts/qa_chunk_selection_v8_hrr/
logs/qa_chunk_selection_v8/
```

---

# 11. History write guard

Реализовать forbidden write prefixes для всех historical experiment paths.

Любая попытка записи:

```text
raise RuntimeError
```

---

# 12. Git protection

До coding/run:

```bash
git status --porcelain
git rev-parse HEAD
git diff
```

Сохранить в audit.

Не использовать:

```text
git reset
git clean
checkout -- .
```

После run повторить snapshot.

---

# 13. Historical hash audit

До:

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

# 14. Historical replay

До V8 training exact replay минимум:

```text
B0 current protocol
V4.4 relevant reader features/scores
V7 LOCAL S2
```

Для отдельной LEX-факторной ветки восстановить также:

```text
EN:
V4.4 B0
V4.4 V6
V4.4 LEX
V4.4 SEM / SEM-FIX where historically present

RU:
V4.5 DiTy B0
V4.5 S3
BGE B0
```

---

# 15. Canonical baseline

Primary V8 использует **тот же B0, question policy, candidate universe и K, что V7**.

Это необходимо, чтобы:

```text
V8 vs V7 S2
```

было чистым сравнением second-stage feature set.

---

# 16. Нельзя возвращать V4.4 B0 в primary V8

V4.4 absolute records используются как historical reference.

Primary V8 не должен одновременно менять:

```text
B0
question
chunking
second-stage features
```

Иначе эффект невозможно атрибутировать.

---

# 17. Candidate universe

Exact V7 historical candidate universe.

Не менять:

```text
documents
chunks
ids
order
overlap
token limits
```

---

# 18. Question policy

Exact V7 policy.

SciER fold-local question selection:

```text
nested-safe exact historical procedure.
```

---

# 19. K policy

Exact V7/historical:

```text
no K search
```

---

# 20. Models

Использовать exact historical:

```text
B0 ranker
QA reader
semantic encoder
```

per cell.

Если exact historical semantic encoder для SEM-feature невозможно воспроизвести на **всех пяти cells**:

```text
SEM primary extension disabled globally.
```

Не включать SEM только на части cells в primary.

---

# 21. Model identity manifest

Для каждой model:

```text
id
revision
local path
SHA-256
tokenizer
formatter
max length
precision
device
```

---

# 22. Feature audit перед freeze

До outer EVAL агент обязан восстановить exact semantics:

```text
V4.4 agreement/majority signal
V4.4 LEX signal
V4.4 SEM signal
V7 Q-A signal
```

Для каждого:

```text
implementation path
input
output
normalization
direction
availability by cell
```

Сохранить:

```text
audit/feature_recovery.json
```

---

# 23. Запрет «примерной» реконструкции historical features

Если exact formula/implementation не доказана:

```text
feature_status = UNRECOVERED
```

Не создавать похожую feature под тем же именем.

---

# 24. Primary common feature requirement

Primary V8 feature group должен быть воспроизводим на всех пяти cells.

Минимально:

```text
B0
Q-A
LEX
reader agreement
```

SEM:

```text
optional global extension
```

только если exact implementation доступна везде.

---

# 25. B0 candidate feature

Для candidate i:

```text
oriented B0 score
within-query z-score
candidate rank
```

Pair difference:

\[
\Delta B0_i=z(B0_i)-z(B0_0)
\]

---

# 26. Q-A feature

\[
QArel_i=Ranker(Q,A_i)
\]

Использовать exact same ranker state, который дал B0 для данного outer/inner row.

Primary:

```text
oriented within-query normalized score
```

\[
\Delta QArel_i=z(QArel_i)-z(QArel_0)
\]

---

# 27. V4.4 reader agreement

Восстановить exact historical V6/read-agreement implementation.

Не использовать слово `majority`, если historical feature формально называлась иначе.

Сохранить два уровня:

```text
candidate-level agreement
question-level consensus confidence
```

только если historical implementation их действительно имела.

---

# 28. Candidate agreement feature

Для каждого candidate:

```text
Agreement_i
```

в exact historical semantics.

Primary difference:

\[
\Delta Agreement_i
=
Agreement_i-Agreement_0
\]

---

# 29. Historical LEX

Использовать **exact full-answer LEX implementation V4.4/V6**, а не V7 PairLEX.

Причина:

```text
V4.4 English gains были связаны именно с full reader answer relations.
```

V7 PairLEX сохраняется только diagnostic reference.

---

# 30. LEX feature

Для каждого candidate:

```text
LEX_i = exact historical candidate lexical agreement/relevance
```

Primary:

```text
within-question normalized LEX
```

\[
\Delta LEX_i=z(LEX_i)-z(LEX_0)
\]

Если historical LEX уже находится в bounded common scale:

raw также сохранить для diagnostics.

---

# 31. SEM feature

Если globally reproducible:

```text
SEM_i = exact historical semantic answer relation
```

Primary extension:

\[
\Delta SEM_i=z(SEM_i)-z(SEM_0)
\]

SEM не является новым encoder.

---

# 32. Source-support

Не включать в primary.

Reason:

```text
V4.5/V6 не показали universal benefit.
```

Сохранить только historical diagnostic.

---

# 33. Aspect

Не использовать как model feature в primary V8.

Только post-hoc analysis.

---

# 34. Primary V8 candidate features

```text
ΔB0
ΔQ-A
ΔAgreement
ΔLEX

candidate rank
effective K

B0 margin12 normalized
B0 entropy normalized

answer_valid_i
answer_valid_0
```

---

# 35. SEM extension features

Если globally available:

```text
ΔSEM
```

Не добавлять новые SEM interactions автоматически.

---

# 36. Explicit interactions

Primary minimal interactions:

```text
ΔLEX × B0 margin
ΔAgreement × B0 margin
ΔQ-A × B0 margin

ΔLEX × ΔAgreement
ΔQ-A × ΔLEX
```

Без aspect interactions.

---

# 37. Почему interactions ограничены

V6 показал риск overfit при широких interactions.

V8 использует только mechanistically justified interactions:

```text
reader signal × baseline uncertainty
reader signal × reader signal
```

---

# 38. Gold downstream utility

Для B0 top1:

```text
A0
```

Для candidate:

```text
Ai
```

\[
u_i=
TokenF1(A_i,G)-TokenF1(A_0,G)
\]

Exact historical Token F1.

---

# 39. Utility model

Primary:

```text
Ridge regression
```

как V7.

Предсказывает:

\[
\hat\mu_i \approx E[u_i\mid x_i]
\]

---

# 40. Ridge alpha

Predeclared:

```text
0.1
1
10
100
```

Добавить `100`, потому что V7 почти всегда выбирал сильную regularization `10`, а новый feature set шире.

Выбор только META_PRED_OOF.

---

# 41. Utility weighting

Каждый question total training weight = 1.

Если m alternatives:

\[
w_i=1/m
\]

---

# 42. Harm target

\[
h_i=1[u_i<0]
\]

---

# 43. Harm classifier

```text
L2 LogisticRegression
solver=liblinear
class_weight=None
```

C grid:

```text
0.01
0.1
1.0
```

V7 стабильно выбирал сильную regularization.

---

# 44. Harm probability

Использовать native logistic probability.

Не добавлять новый calibrator в primary без отдельной необходимости.

Calibration quality report separately.

---

# 45. Safe switch

Для candidate:

```text
mu_i
p_harm_i
```

Safe:

```text
mu_i >= tau_utility
AND
p_harm_i <= tau_harm
```

---

# 46. Threshold grids

```text
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

---

# 47. Candidate selection

Evaluate all historical top-K.

Among safe alternatives:

```text
choose candidate with maximum predicted mu
```

Tie-break:

```text
lower p_harm
better B0 rank
```

---

# 48. Preserve B0

Если no safe candidate:

```text
keep B0 top1
```

---

# 49. Primary systems

Predeclare:

```text
H0 = B0

H1 = exact V7 S2 replay
     baseline universal-positive-sign reference

H2 = H1 + exact historical LEX

H3 = H2 + exact V4.4 reader agreement
     PRIMARY HYBRID

H4 = H3 + exact SEM
     only if SEM globally reproducible before freeze
```

---

# 50. H3 — главный кандидат

Основная гипотеза V8:

> добавление V4.4 LEX и reader-agreement сигналов к V7 S2 risk-aware decision восстанавливает сильный английский reader gain, не разрушая положительный знак на RU.

---

# 51. H4 status

Если SEM доступна на всех cells:

```text
H4 = preregistered secondary extension
```

H4 не заменяет H3 post-hoc, если H3 failed.

Для final development candidate H4 может быть выбран только по заранее заданному selection rule.

---

# 52. Никакого per-dataset feature selection

Нельзя:

```text
SciREX use SEM
SciER use LEX
RU use Q-A
```

Все cells используют одну system definition.

LOCAL означает:

```text
same features
different fitted coefficients.
```

---

# 53. Meta-level OOF

Как V7:

```text
META_FEATURE_OOF
META_PRED_OOF
```

Все:

```text
alpha
C
thresholds
```

выбирать только на META_PRED_OOF.

---

# 54. Outer CV

Exact historical:

```text
5 document-grouped outer folds
```

---

# 55. Inner folds

Default:

```text
4 document-grouped
```

Fallback:

```text
3
2
```

только по pre-run feasibility.

---

# 56. Seeds

```text
42
43
44
```

Каждый fit отдельно.

Seeds не объединять в training rows.

---

# 57. Dataset-trained upstream

Если upstream trained on dataset:

features only from honest inner OOF upstream model.

---

# 58. Fully frozen upstream

No artificial retraining.

---

# 59. Hyperparameter selection order

### Step 1
Ridge alpha via OOF utility MAE.

### Step 2
harm C via OOF log-loss.

### Step 3
joint threshold pair via simulated final Token F1 subject to safety.

---

# 60. Ridge selection

Primary:

```text
weighted MAE
```

Tie within 0.05 F1:

```text
choose larger alpha.
```

---

# 61. Harm C selection

Primary:

```text
log-loss
```

Tie within 0.001:

```text
choose smaller C.
```

---

# 62. Threshold selection objective

Maximize:

```text
OOF final Token F1
```

subject to:

```text
ΔHit1 >= -0.5 pp
ΔHN1 <= +1 pp
```

---

# 63. Threshold safety tie-break

If multiple equal:

1. lower switch rate;
2. lower predicted harm;
3. larger tau_utility.

---

# 64. Degenerate fallback

If no safe threshold tuple:

```text
B0_FALLBACK
```

Record.

---

# 65. Fallback is not method success

Report:

```text
fallback-fold rate
actual switch rate
stable switches
```

---

# 66. Primary mode

```text
LOCAL
```

Same architecture.

Per-cell TRAIN fitting allowed.

---

# 67. POOLED-simple

Secondary transfer diagnostic only.

No aspect.

No dataset id.

Equal corpus-family weight.

---

# 68. LOCO-meta

Secondary diagnostic.

SciER cells leave together.

---

# 69. SciER cross-cell embargo

Mandatory:

```text
canonical_document_id
document hash
```

No same source document across train/eval through alternate chunking.

---

# 70. Cross-corpus overlap audit

Exact full-document hashes.

---

# 71. Primary quality metric

```text
Token F1
```

---

# 72. Answer metrics

```text
Char F1
ROUGE-L
Unicode ROUGE-L
EM
Lemma F1 RU
BERTScore only if historical contract
```

---

# 73. Evidence metrics

```text
Hit@1
Hit@K
HN@1
HN@K
F1|selected positive
Amb@1/GSRN@1 where applicable
```

---

# 74. Switch diagnostics

```text
good
bad
neutral

mean actual u
median u

u >= +5
u <= -5
```

---

# 75. Feature attribution

Predeclared differences:

```text
H2-H1 = LEX contribution
H3-H2 = reader-agreement contribution
H4-H3 = SEM contribution
```

Это attribution contrasts.

---

# 76. Coefficient diagnostics

Report:

```text
median coefficient
IQR
sign stability
```

для:

```text
ΔB0
ΔQ-A
ΔLEX
ΔAgreement
ΔSEM
interactions
```

---

# 77. Primary universal-development success

H3 success if:

```text
ΔTokenF1 > 0
```

on all five cells,

and:

```text
LCB95 > -0.5
```

on every cell,

and evidence safety passes.

---

# 78. Strict success

```text
LCB95 > 0
```

all cells.

---

# 79. Non-degeneracy

Per cell:

```text
stable switches >= max(3, ceil(1% questions))
```

Stable = switched in at least 2/3 seeds.

Diagnostic flag.

Для перехода к confirmation:

```text
non-degenerate required.
```

---

# 80. H4 selection rule

Если H4 globally available:

H4 может заменить H3 как final development candidate только если:

```text
H4 passes all universal criteria
AND
Global family-balanced Δ(H4) >= Δ(H3)+0.2 F1
AND
no cell H4 worse than H3 by >0.5 F1
```

Иначе H3 remains primary.

---

# 81. Resource strategy

Primary V8 intentionally uses full top-K QA.

Не вводить admission до доказательства quality.

---

# 82. Главный resource comparison

Сравнивать не только с B0, но и с:

```text
Qwen generative extraction
BGE strong RU baseline
```

где historical/fresh comparable execution possible.

---

# 83. Почему comparison with Qwen обязателен

Диссертационная цель:

```text
resource reduction vs generative extraction
```

а не:

```text
minimal overhead over B0 at any quality cost.
```

---

# 84. Resource benchmark systems

```text
B0
H1 V7-S2
H3 Hybrid
H4 if globally available
Qwen reference
BGE RU reference
```

---

# 85. Resource metrics

```text
parameters
weights size
peak VRAM
CPU RSS
median latency
P95
throughput
QA forwards
ranker forwards
semantic encoder forwards
FLOPs/MACs where practical
```

---

# 86. Qwen benchmark fairness

Use same:

```text
documents
questions/aspects
answer normalization
hardware
precision if practical
```

Do not compare historical latency from different hardware as primary fresh benchmark.

Historical Qwen memory can remain contextual reference.

---

# 87. Fresh-process benchmark

Each system separate process.

---

# 88. Resource decision

V8 resource success does NOT require:

```text
<=1.25× B0
```

Primary dissertation-oriented condition:

```text
Hybrid materially cheaper than Qwen
```

while quality improves B0.

Also report overhead vs B0 transparently.

---

# 89. Pareto reporting

Report:

```text
Token F1 ↑
VRAM ↓
latency ↓
FLOPs ↓
```

Do not invent one artificial combined score.

---

# 90. Statistical unit

Per-question seed averaging, then document-cluster bootstrap.

---

# 91. Bootstrap

```text
5000
paired
document cluster
seed=42
```

---

# 92. Global family weighting

```text
SciER family effect =
mean(384,128)
```

Global families:

```text
SciREX
SciER
RuSERRC
SciMDIX
```

---

# 93. Historical references table

Report absolute historical records:

```text
SciREX V4.4 SEM-FIX
SciER384 V4.4 LEX
SciER128 V4.4 SEM
RuSERRC BGE
SciMDIX BGE
```

But do not use them as current primary baselines.

---

# 94. V8 result interpretation

Separate:

```text
increment over V7 B0
absolute F1
distance to historical record
```

---

# 95. Stop rule

If H3 fails universal criteria:

do not add new features to V8.

New hypothesis:

```text
V8.1
```

---

# 96. No post-hoc hybrid assembly

Cannot choose:

```text
H4 on SciREX
H2 on SciER
H3 on RU
```

One final system definition only.

---

# 97. Independent confirmation

If H3/H4 passes development gate:

freeze.

Then untouched:

```text
EN confirmation
RU confirmation
```

---

# 98. Confirmation for LOCAL method

Untouched corpus must be split:

```text
CONFIRM_TRAIN/CAL
CONFIRM_TEST
```

Fit local meta-layer only on TRAIN/CAL.

Test sealed.

This validates architecture, not zero-shot shared coefficients.

---

# 99. Dissertation claim if confirmed

Possible:

> Разработан единый параметризуемый метод reader-assisted остаточного ранжирования фрагментов научно-технического документа, объединяющий сигналы согласованности ответов нескольких фрагментов с оценкой ожидаемого изменения качества извлечения и риска ухудшения решения; базовая вопросно-ответная модель при этом не дообучается.

---

# 100. What not to claim

Do not claim:

```text
first reader-aware reranking
universal proven on development
better than BGE everywhere
```

---

# 101. SECOND PRIMARY BRANCH — V8-V44-AF

Рабочее название:

```text
V8-V44-AF
V4.4 Adaptive Fusion
```

Цель:

> взять reader-assisted механизм V4.4 как архитектурную основу и добавить к нему обучаемые компоненты, показавшие полезность в поздних RU-экспериментах, при этом сохранить одну и ту же архитектуру на EN и RU.

Критически важно различать:

```text
historical V4.4 replay
```

и:

```text
new V8-V44-AF experiment.
```

Это не одно и то же.

---

# 102. Три слоя второй ветки

Вторая ветка состоит из трёх строго разделённых слоёв.

## Layer P — exact preservation

Только EN.

Воспроизвести exact historical V4.4 protocol и результаты.

Никаких новых weights/features.

## Layer C — canonical universal experiment

Основной V8-V44-AF.

Все пять cells работают в **одном canonical V8 protocol**:

```text
тот же B0
тот же question policy
тот же candidate universe
тот же K
```

что Branch A / V7.

Именно Layer C участвует в универсальном сравнении Branch A vs Branch B.

## Layer R — historical-protocol EN retention diagnostic

Только EN.

Применить новую adaptive-fusion архитектуру внутри exact historical V4.4 protocol и проверить:

> сохраняет ли новая схема сильный абсолютный результат V4.4, если вернуть historical question/chunking/candidate conditions?

Layer R — diagnostic и не используется для выбора универсальной ветки.

---

# 103. Почему такое разделение обязательно

Без Layer C Branch A и Branch B могли бы иметь:

```text
разные question policies
разные SciER candidate universes
разные B0 semantics
```

и прямое сравнение абсолютного F1 было бы методологически некорректно.

Поэтому:

```text
Branch A vs Branch B
```

сравниваются **только на canonical protocol**.

Exact historical V4.4 results остаются отдельной preservation/reference линией.

---

# 104. Layer P — English V4.4 preservation

Обязательно:

```text
SciREX
SciER 384/50
SciER 128/24
```

Восстановить exact historical:

```text
documents
candidate universe
chunking
outer folds
seeds
question policy
K
B0 ranker
QA reader
semantic encoder
normalization
metric implementation
```

---

# 105. SciREX preservation

Воспроизвести минимум:

```text
V4.4 B0
V4.4 V6-K5 / protocol-selected reader-assisted system
V4.4 LEX / LEX-FIX where stored
V4.4 SEM / SEM-FIX
V4.4 ASEM only as diagnostic if stored
```

Сохранить historical status:

```text
selected
secondary
diagnostic
```

Не переименовывать diagnostic maximum в selected system.

---

# 106. SciER 384/50 preservation

Воспроизвести:

```text
B0
V6
LEX
SEM
```

с exact:

```text
384/50 chunking
question-selection policy
fold definitions
```

---

# 107. SciER 128/24 preservation

Воспроизвести:

```text
B0
V6
LEX
SEM
```

с exact:

```text
128/24 chunking
question-selection policy
fold definitions
```

---

# 108. Preservation pass criterion

Если historical predictions сохранены:

```text
exact metric replay
```

Если выполняется fresh deterministic inference:

```text
same selected candidate IDs
same normalized final answers
aggregate Token F1 abs difference <= 1e-4
```

Если library/hardware nondeterminism меняет только raw logits, но не selections/final answers:

```text
PASS_WITH_NUMERIC_WARNING
```

Материальное изменение final selections:

```text
PRESERVATION_FAILED
```

---

# 109. Preservation report

Отдельная таблица:

```text
cell
historical system
historical status
stored Token F1
replayed Token F1
absolute difference
candidate agreement
answer agreement
PASS/FAIL
```

Layer P не является новым quality experiment.

---

# 110. Layer C — canonical protocol

Для всех пяти cells:

```text
SciREX
SciER 384/50
SciER 128/24
RuSERRC
SciMDIX
```

использовать **тот же canonical protocol, что Branch A**.

То есть Branch C не возвращает historical SciER question policy/candidate universe.

Он возвращает **архитектуру V4.4**, а не весь исторический experimental environment.

Это обеспечивает честное сравнение:

```text
V8-HRR
vs
V8-V44-AF
```

---

# 111. Что значит «V4.4 как основа» в Layer C

V4.4 рассматривается как feature/scoring architecture:

```text
B0 relevance
+
reader answer agreement
+
common V4.4 support component if semantically identical on all cells
```

Это называется:

```text
V44-CORE
```

Не использовать в Layer C разные historical best branches per dataset.

---

# 112. V44-CORE commonality audit

До freeze создать:

```text
audit/v44_common_core.json
```

Для каждого candidate component:

```text
B0
reader agreement
support
```

проверить:

```text
same mathematical semantics
runtime computable
exact implementation recovered
available on all five cells
```

---

# 113. Обязательные и optional backbone components

Обязательно:

```text
B0
reader agreement
```

`support` включается в V44-CORE только если:

```text
same_semantics = true
```

на всех пяти cells.

Если support semantics EN/RU различаются:

```text
exclude support GLOBALLY from Layer C
```

до outer EVAL.

Не использовать:

```text
support on EN
no support on RU
```

в primary Layer C.

---

# 114. Не путать V4.4 `V6` и экспериментальную версию `V6`

В коде и artifacts historical V4.4 reader score называть:

```text
V44_READER_CORE
```

или:

```text
V44_CORE
```

Не использовать ambiguous variable name:

```text
V6
```

без namespace.

---

# 115. Parameter family V44-CORE

Если historical V4.4 core имел форму:

```text
B0 + weighted reader agreement + weighted support
```

восстановить exact scoring family.

Не переносить hard-coded EN weights на RU.

Использовать:

```text
same scoring family
same allowed parameter family
same selection procedure
```

на всех cells.

---

# 116. V44-CORE parameter fitting

Primary Layer C:

```text
LOCAL
```

Параметры V44-CORE fit/select отдельно внутри разрешённой TRAIN/CAL части каждой cell.

Это допускает:

```text
reader agreement weight EN != reader agreement weight RU
```

при одной architecture.

Если historical V4.4 weights были fixed после TRAIN/CAL search:

восстановить именно **parameter search family**, а не просто скопировать найденные EN числа.

---

# 117. Если exact historical parameter-selection procedure не восстановима

Не угадывать.

В этом случае:

```text
V44_CORE_FIT_BLOCKED
```

и использовать только Branch A до исправления audit.

Нельзя создать новый arbitrary grid под именем V4.4.

---

# 118. V44-CORE nested fitting

Любые parameters V44-CORE:

```text
weights
thresholds
routing parameters
```

подбираются только внутри outer TRAIN/CAL.

Outer EVAL sealed.

Для dataset-trained upstream соблюдать honest inner OOF.

---

# 119. Anchor candidate

После fit V44-CORE:

```text
C44 = argmax S44_i
```

где:

```text
S44_i = fitted V44-CORE candidate score.
```

Это anchor новой residual correction.

---

# 120. Anchor answer

```text
A44 = QA(Q,C44)
```

Для alternative:

```text
Ai = QA(Q,Ci)
```

---

# 121. Gold residual utility

Для alternative `i != anchor`:

\[
u_i =
TokenF1(A_i,G)
-
TokenF1(A_{44},G)
\]

Exact historical Token-F1 evaluator данной canonical cell.

Gold используется:

```text
training target
evaluation
```

но никогда runtime feature.

---

# 122. Зачем anchor именно V44-CORE

Цель второй ветки:

> не заменить V4.4 поздним selector-ом, а позволить новым компонентам исправлять V4.4 только когда TRAIN evidence поддерживает correction.

Если residual не уверен:

```text
return C44.
```

---

# 123. Добавляемые компоненты

Primary residual feature family:

```text
FullLEX
PairLEX
Q-A relevance
SEM if globally eligible
```

Harm-risk:

```text
отдельный safety model.
```

---

# 124. FullLEX — V4.4-derived signal

`FullLEX` = exact historical full-answer lexical agreement/relevance family V4.4.

Для каждой cell восстановить одну и ту же mathematical semantics.

Не использовать V7 PairLEX под именем FullLEX.

---

# 125. Почему FullLEX нужен

FullLEX:

```text
сильно проявлялся на EN в V4.4
```

и позднее показывал полезность в отдельных RU анализах.

Он является основным cross-language lexical residual signal.

---

# 126. PairLEX — RU-derived late signal

Отдельно включить exact V7 PairLEX semantics.

Если V7 PairLEX определялся относительно B0 top1 answer:

сохранить именно эту semantics:

\[
PairLEX^{B0}_i
=
TokenF1(A_i,A_{B0})/100
\]

Не менять silently anchor на C44.

---

# 127. Почему PairLEX не заменяется на C44-PairLEX

Польза PairLEX была обнаружена при конкретной V7 semantics.

Изменение reference answer:

```text
A_B0 → A_C44
```

создало бы новый feature.

Поэтому exact V7 PairLEX входит как RU-derived signal.

Дополнительно можно сохранить:

```text
PairLEX_to_C44
```

только как diagnostic, не primary.

---

# 128. SEM

SEM включается в primary residual set только если до EVAL доказано:

```text
same semantic feature meaning
exact historical encoder per language
same pooling/normalization semantics
available all five cells
```

Разные language-specific encoders допустимы, если feature semantics совпадает.

---

# 129. SEM eligibility

Создать:

```text
SEM_COMMON = true/false
```

только из technical audit.

Если false:

```text
disable SEM globally in Layer C primary
```

Не выбирать SEM availability по quality.

---

# 130. Q-A relevance

\[
QArel_i=Ranker(Q,A_i)
\]

Использовать exact same ranker state, который дал B0 scores данного row/fold.

Primary:

```text
oriented
within-query normalized
```

---

# 131. Reader agreement не дублировать

Reader agreement уже находится в V44-CORE.

Не подавать математически идентичный agreement signal второй раз в residual model.

Если есть отдельный historical candidate-level agreement, отличный от core score:

он допускается только как predeclared diagnostic.

---

# 132. Source-support не добавлять повторно

Если common support входит в V44-CORE:

не включать его второй раз residual feature.

Если support globally excluded from V44-CORE:

не возвращать его как RU-only residual.

---

# 133. Residual normalization

Все relevance-like candidate features:

```text
higher = better
```

Для:

```text
S44
FullLEX
SEM
Q-A
```

использовать within-query z-normalization.

If std < 1e-8:

```text
all z=0
```

---

# 134. Residual differences relative to C44

\[
\Delta S44_i=z(S44_i)-z(S44_{44})
\]

\[
\Delta FullLEX_i=z(FullLEX_i)-z(FullLEX_{44})
\]

\[
\Delta SEM_i=z(SEM_i)-z(SEM_{44})
\]

\[
\Delta QA_i=z(QA_i)-z(QA_{44})
\]

PairLEX V7 сохраняется в собственной bounded scale.

---

# 135. PairLEX feature representation

Primary PairLEX feature:

```text
PairLEX_B0_i
```

Дополнительно difference:

```text
PairLEX_B0_i - PairLEX_B0_anchor
```

Использовать один predeclared вариант:

```text
ΔPairLEX = PairLEX_B0_i - PairLEX_B0_anchor
```

Raw сохранить для diagnostics.

---

# 136. Primary residual base features

```text
ΔS44
ΔFullLEX
ΔPairLEX
ΔQ-A

V44 margin
B0 margin
effective K
candidate rank
```

Если SEM_COMMON:

```text
+ ΔSEM
```

---

# 137. Explicit interactions

Минимально:

```text
ΔFullLEX × V44_margin
ΔPairLEX × V44_margin
ΔQ-A × V44_margin

ΔFullLEX × ΔPairLEX
ΔFullLEX × ΔQ-A
```

Если SEM_COMMON:

```text
ΔSEM × V44_margin
ΔFullLEX × ΔSEM
```

No automatic polynomial expansion.

---

# 138. Что исключено

Primary Layer C не использует:

```text
aspect
dataset id
manual language rules
new pretrained models
attention features
RU-only source heuristics
```

---

# 139. Learnable adaptive weights

Residual utility model должен сохранять standardized coefficients.

Это позволяет проверить пользовательскую гипотезу:

> LEX-related weight может обучиться большим на RU и меньшим на EN, при одной архитектуре.

Report:

```text
β_FullLEX
β_PairLEX
β_SEM
β_QA
```

---

# 140. Primary utility model

```text
StandardScaler
+
Ridge regression
```

Target:

\[
\hat\mu_i
\approx E[u_i|x_i]
\]

Ridge coefficients и являются learnable residual weights после standardization.

---

# 141. Ridge regularization

Predeclared:

```text
alpha ∈ {
0.1,
1,
10,
100
}
```

Сильная regularization позволяет:

```text
неполезным signals получать малый effective weight
```

без ручного отключения feature по языку.

---

# 142. Почему не Lasso primary

Не использовать Lasso/ElasticNet как primary:

```text
малые выборки
коррелированные FullLEX/PairLEX/SEM
нестабильный feature selection
```

Ridge лучше соответствует задаче:

```text
smooth adaptive weighting.
```

---

# 143. Question-level weighting

Если question имеет m residual alternatives:

\[
w_i=1/m
\]

Total question training weight = 1.

---

# 144. Harm target

\[
h_i=1[u_i<0]
\]

---

# 145. Harm classifier

```text
L2 LogisticRegression
solver=liblinear
class_weight=None
```

C grid:

```text
0.01
0.1
1
```

---

# 146. Safe switch

Candidate safe if:

```text
mu_i >= tau_utility
AND
p_harm_i <= tau_harm
```

---

# 147. Threshold grids

```text
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

---

# 148. Candidate selection

Evaluate historical canonical top-K.

Among safe candidates choose:

```text
maximum predicted mu
```

Tie-break:

```text
lower p_harm
better V44 rank
better B0 rank
```

---

# 149. Preserve anchor

Если safe candidate отсутствует:

```text
select C44.
```

---

# 150. Layer C system ladder

Predeclare:

```text
F0 = V44-CORE anchor system

F1 = F0 + FullLEX residual

F2 = F1 + exact V7 PairLEX residual

F3 = F2 + Q-A residual

F4 = F3 + SEM residual
     only if SEM_COMMON=true

F5-core = F3 + harm-risk safe switch

F5-sem = F4 + harm-risk safe switch
         only if SEM_COMMON=true
```

---

# 151. Primary F5 definition — без quality-based выбора

До outer EVAL:

```text
if SEM_COMMON=true:
    PRIMARY = F5-sem
else:
    PRIMARY = F5-core
```

Это решается **только технической availability**, а не качеством.

Нельзя после EVAL выбрать F5-core потому, что F5-sem оказался хуже.

---

# 152. Что проверяют ablations

```text
F1-F0:
FullLEX incremental value

F2-F1:
V7 PairLEX additional RU-derived information

F3-F2:
Q-A incremental value

F4-F3:
SEM incremental value

F5-F3/F4:
harm-risk safety value
```

---

# 153. Coefficient hypothesis

Predeclared descriptive hypotheses:

```text
median |β_FullLEX| RU >= median |β_FullLEX| EN
```

и:

```text
median |β_PairLEX| RU >= median |β_PairLEX| EN
```

Если SEM_COMMON:

```text
median |β_SEM| EN >= median |β_SEM| RU
```

Это mechanism diagnostics, не criteria for success.

---

# 154. Coefficient reporting

Для standardized coefficients:

```text
median
IQR
sign consistency
per-fold values
per-seed values
```

Не сравнивать raw unstandardized slopes.

---

# 155. Нельзя задавать веса вручную по языку

Forbidden:

```text
if RU: lambda_LEX = ...
if EN: lambda_LEX = ...
```

Различия появляются только из TRAIN data.

---

# 156. Hyperparameter OOF protocol

Различать:

```text
META_FEATURE_OOF
META_PRED_OOF
```

All:

```text
V44 core parameters
Ridge alpha
harm C
tau_utility
tau_harm
```

выбирать только внутри allowed TRAIN/CAL.

---

# 157. Ridge alpha selection

Primary metric:

```text
question-weighted OOF MAE of residual utility
```

If MAE difference <0.05 F1:

```text
choose larger alpha.
```

---

# 158. Harm C selection

Primary:

```text
OOF log-loss
```

Tie <0.001:

```text
choose smaller C.
```

---

# 159. Threshold selection

На `META_PRED_OOF` maximize:

```text
final Token F1
```

subject to safety relative to F0:

```text
ΔHit1 >= -0.5 pp
ΔHN1 <= +1 pp
```

---

# 160. Degenerate fallback

Если no safe threshold tuple:

```text
fold policy = F0 fallback
```

Report.

Полный fallback не считать активным улучшением.

---

# 161. Layer C primary success — strict universal improvement

PRIMARY F5 passes strong universal development criterion if:

```text
F5 - F0 > 0
```

on all five cells,

and:

```text
LCB95(F5-F0) > -0.5
```

on all cells,

and evidence safety PASS.

Это основной критерий пользовательской цели:

```text
улучшение всех datasets.
```

---

# 162. Preservation-oriented success

Дополнительно report более мягкий criterion:

```text
EN:
F5-F0 >= 0

RU:
F5-F0 > 0
```

на обеих RU cells.

Назвать:

```text
EN-preservation + RU-improvement
```

Не называть universal improvement.

---

# 163. Strict statistical success

```text
LCB95(F5-F0) > 0
```

all cells.

---

# 164. Non-degeneracy

For active-improvement claim:

```text
stable switches >= max(3, ceil(1% questions))
```

per cell.

Stable switch:

```text
same question switched in >=2/3 seeds.
```

---

# 165. Absolute quality reporting

Для каждого cell обязательно:

```text
B0 absolute F1
F0 absolute F1
F5 absolute F1

historical V4.4 selected F1
historical best diagnostic F1 where relevant
```

Но historical numbers помечаются:

```text
different protocol
```

если question/candidate policy отличается от canonical.

---

# 166. Layer R — EN historical-protocol retention diagnostic

На:

```text
SciREX
SciER384
SciER128
```

взять exact historical V4.4 protocol Layer P.

Внутри него применить **ту же residual architecture F5**, что определена до EVAL.

Цель:

> проверить, сохраняет ли adaptive fusion сильный EN результат в исходных условиях V4.4.

---

# 167. Layer R training

Новая residual model fit только на historical protocol TRAIN/CAL.

Outer historical EVAL sealed.

Feature definitions, grids и harm rule:

```text
identical Layer C
```

Нельзя retune feature set для Layer R.

---

# 168. Layer R anchor

Anchor:

```text
exact historical V4.4 common reader-core selected within historical protocol
```

Не historical best-per-dataset LEX/SEM branch.

Это сохраняет одну архитектурную основу.

---

# 169. Layer R report

Для EN:

```text
historical protocol B0
historical V44 core
historical LEX
historical SEM
Layer-R F5

Δ F5 vs V44 core
gap to historical best
```

---

# 170. Layer R interpretation

Если:

```text
Layer-R F5 ≈ historical best
```

это поддерживает тезис:

> одна adaptive-fusion architecture способна приблизиться к вручную различавшимся V4.4 branches.

Если хуже:

не использовать Layer R для universal selection.

---

# 171. Branch A vs Branch B fairness

Final Branch A vs Branch B comparison использует **только canonical Layer C results**.

Нельзя сравнивать:

```text
Branch A canonical
vs
Layer R historical protocol
```

как будто это один benchmark.

---

# 172. Final branch selection hierarchy

Обе ветки frozen before EVAL.

После completion:

1. candidate must pass its own strong universal criterion;
2. если проходит одна → выбрать её;
3. если проходят обе → сравнить family-balanced absolute F1 **в canonical protocol**;
4. если difference <0.2 F1 → lower resource cost;
5. если still tied → simpler feature/model architecture.

---

# 173. Family-balanced absolute F1

SciER contributes once:

\[
F1_{SciER}
=
mean(F1_{384},F1_{128})
\]

Global families:

```text
SciREX
SciER
RuSERRC
SciMDIX
```

equal weight.

---

# 174. POOLED shared-weight diagnostic

Secondary only.

Один shared residual model.

No:

```text
dataset id
aspect
manual language feature weights
```

Equal corpus-family training weight.

---

# 175. POOLED V44-CORE parameters

Для pooled diagnostic:

V44 core parameters также должны быть shared.

Нельзя:

```text
local V44 core
+
pooled residual
```

и называть это fully shared model.

Если full shared core technically infeasible:

```text
POOLED_FULL = BLOCKED
```

и разрешён только:

```text
POOLED_RESIDUAL_ON_LOCAL_CORE
```

как отдельно названный diagnostic.

---

# 176. LOCO-meta

Secondary.

SciER cells leave together.

Если upstream/core parameters local:

называть:

```text
LOCO-residual
```

а не zero-shot whole pipeline.

---

# 177. SciER cross-cell embargo

Для any pooled/LOCO diagnostic:

same source document from another SciER chunking cell нельзя использовать в training.

Use:

```text
canonical_document_id
full-text SHA-256
```

---

# 178. Cross-corpus overlap audit

Exact/normalized full-document hashes.

Same lineage = same leakage group.

---

# 179. Optional RU ranker-strength diagnostic

Дополнительно на:

```text
RuSERRC
SciMDIX
```

повторить Layer C с:

```text
DiTy
BGE
```

если exact BGE historical recipe/checkpoints доступны.

---

# 180. Ranker-strength purpose

Проверить:

\[
Gain_{F5|DiTy}
-
Gain_{F5|BGE}
\]

Если residual gain уменьшается под BGE:

> stronger primary ranking makes reader residual more redundant.

---

# 181. BGE diagnostic не меняет universal primary

Нельзя после результата сделать:

```text
BGE on RU
current EN ranker on EN
```

final method в текущем V8.

Это потребовало бы отдельного preregistered architecture.

---

# 182. Resource benchmark

Canonical systems:

```text
B0
Branch A primary
Branch B F0
Branch B F5

Qwen generative reference
BGE RU reference
```

---

# 183. Layer R resource benchmark

Secondary only.

Сравнить:

```text
historical V44 core
Layer-R F5
historical best branch
```

---

# 184. Resource metrics

```text
parameters
weights size
peak VRAM
CPU RSS

median latency
P95
throughput

QA forwards
Q-A ranker forwards
semantic encoder forwards

FLOPs/MACs where practical
```

---

# 185. Resource interpretation

Branch B uses full top-K QA.

Не требовать near-B0 cost.

Главный dissertation comparison:

```text
extractive adaptive fusion
vs
generative Qwen.
```

Overhead relative F0/B0 report transparently.

---

# 186. Statistical unit

Seeds are not independent observations.

Use:

1. per-question per-seed;
2. average seeds within question;
3. paired document-cluster bootstrap.

---

# 187. Bootstrap

```text
iterations=5000
seed=42
paired=true
cluster=document_id
```

---

# 188. Fold/seed stability

Report:

```text
5 folds
3 seeds
```

for F0-F5.

---

# 189. Feature attribution table

Per cell:

```text
F1-F0 FullLEX
F2-F1 PairLEX
F3-F2 Q-A
F4-F3 SEM if available
F5-previous harm
```

---

# 190. Coefficient table

Per cell:

```text
β_FullLEX
β_PairLEX
β_QA
β_SEM
```

median/IQR/sign consistency.

---

# 191. Key mechanism analysis

Главный механизмный вопрос:

> действительно ли одна training procedure автоматически распределяет веса так, чтобы RU-derived lexical signals были сильнее там, где они полезны, без ручного language branching?

Report descriptive comparisons:

```text
EN coefficient distribution
RU coefficient distribution
```

---

# 192. Не выводить языковую причинность

Даже если RU coefficients выше:

не писать:

> русский язык требует большего LEX.

Корректно:

> на исследованных русскоязычных корпусах обученная модель присваивала LEX-компонентам больший относительный вес.

---

# 193. Aspect analysis

Post-hoc only.

Aspect не feature.

---

# 194. K analysis

Post-hoc only.

No K retuning.

---

# 195. Historical preservation vs canonical result

В final report три разных таблицы:

```text
A. V4.4 preservation
B. canonical Branch A/B comparison
C. Layer R EN retention diagnostic
```

Не объединять их в одну таблицу без protocol label.

---

# 196. Artifact namespace

```text
artifacts/qa_chunk_selection_v8_dual/
  run_<timestamp>/
```

---

# 197. Artifact structure

```text
run/
  status.json
  manifest.json

  contracts/
    branch_hrr_frozen.yaml
    branch_v44_af_frozen.yaml

  audit/
    git_before/
    git_after/

    hashes_before.json
    hashes_after.json

    v44_preservation/
    v44_common_core.json
    v7_replay/

    feature_recovery.json
    candidate_universe.json
    split_hashes.json
    model_manifests/

    scier_lineage.json
    cross_corpus_overlap.json

  branch_hrr/
    models/
    predictions/
    analysis/
    resources/

  branch_v44_af/
    canonical/
      models/
      predictions/
      analysis/
      resources/

    en_historical_retention/
      models/
      predictions/
      analysis/

    ru_ranker_strength/
      models/
      predictions/
      analysis/

  reports/
    v44_preservation.md
    v8_hrr_results.md
    v8_v44_af_results.md
    v8_branch_comparison.md
    reproducibility.md
```

---

# 198. Prediction runtime/evaluation separation

```json
{
  "runtime": {},
  "evaluation": {}
}
```

Runtime cannot contain:

```text
gold answer
actual utility
evidence labels
oracle rank
oracle best candidate
```

---

# 199. Branch B runtime fields

```text
document
question
cell
fold
seed

B0 top-K
B0 scores

V44 core component scores
V44 fitted parameters
S44
C44

QA answers top-K

FullLEX
PairLEX_B0
Q-A
SEM if available

residual feature vector
utility prediction
harm probability
thresholds
selected candidate
final answer
```

---

# 200. Meta provenance

Per training/prediction row:

```text
outer fold
inner upstream fold
meta fold
seed

upstream model hash
training-doc-set hash

V44 core parameter source
feature version

self-excluded upstream
self-excluded meta
```

---

# 201. Unit tests — preservation

```text
test_v44_scirex_replay
test_v44_scier384_replay
test_v44_scier128_replay
test_v44_status_labels
```

---

# 202. Unit tests — common core

```text
test_v44_core_same_schema_all_cells
test_v44_core_parameter_family_exact
test_no_hardcoded_en_weights_on_ru
test_support_global_eligibility
test_anchor_is_v44_core_argmax
```

---

# 203. Unit tests — residual features

```text
test_fulllex_exact
test_pairlex_exact_v7_semantics
test_pairlex_reference_is_b0

test_sem_common_eligibility
test_q_a_same_ranker_state

test_score_orientation
test_within_query_normalization
```

---

# 204. Unit tests — learning/leakage

```text
test_question_weighting
test_meta_feature_oof
test_meta_prediction_oof
test_threshold_oof_only

test_runtime_no_gold
test_utility_relative_to_c44
test_no_manual_language_weights
```

---

# 205. Unit tests — cross-cell protection

```text
test_scier_cross_cell_embargo
test_cross_corpus_overlap
test_seed_isolation
```

---

# 206. Historical protection tests

```text
test_history_read_only
test_no_historical_overwrite
test_hashes_before_after
```

---

# 207. Smoke test

Before full EVAL:

```text
one EN cell
one RU cell
one fold
one seed
```

Verify:

```text
V44 core fitting
anchor selection
FullLEX
PairLEX
SEM if eligible
Q-A
Ridge coefficients
harm switch
artifact isolation
```

No scientific tuning from smoke.

---

# 208. Execution order

```text
A. history/hash/git audit
B. Layer P exact V4.4 preservation
C. recover V44 common core
D. recover FullLEX / PairLEX / SEM / Q-A
E. V7 replay
F. unit tests
G. smoke
H. freeze Branch A + Branch B contracts
I. run Branch A canonical
J. run Branch B Layer C canonical
K. run Layer R EN retention diagnostic
L. optional RU BGE diagnostic
M. analysis
N. resource benchmark
O. final history integrity check
```

---

# 209. Freeze requirement

Before first full outer EVAL store SHA-256 for:

```text
branch_hrr_frozen.yaml
branch_v44_af_frozen.yaml
```

Layer R uses Branch B feature/model contract.

No post-EVAL changes.

---

# 210. Stop rule

If neither Branch A nor Branch B PRIMARY passes strong universal criterion:

do not add more features in V8.

Do not use Layer R or RU BGE diagnostic to patch individual cells.

At that point:

```text
move to independent validation / applicability-boundary formulation
```

or create a separately justified future version.

---

# 211. Dissertation interpretation if Branch B succeeds

Strong formulation:

> Разработан единый параметризуемый reader-assisted метод извлечения тематически заданной информации, в котором общий механизм согласования ответов нескольких фрагментов дополняется регуляризованной остаточной комбинацией лексических, семантических и вопросно-ответных признаков; относительный вклад дополнительных компонентов определяется по обучающей выборке без изменения архитектуры метода между языками и корпусами.

---

# 212. Harm formulation

Если harm component materially contributes:

> Коррекция базового reader-assisted решения выполняется только при прогнозируемой положительной полезности и допустимом риске ухудшения.

---

# 213. If EN weights small / RU weights large

Корректный вывод:

> При общей архитектуре fitted coefficients дополнительных лексических сигналов различались между исследованными корпусами; на части русскоязычных наборов их относительный вклад оказался выше.

Не:

```text
LEX inherently Russian.
```

---

# 214. If Layer R preserves V4.4 best

Допустимо:

> Адаптивная fusion-схема сохранила сильное качество historical V4.4 protocol на английских данных, одновременно обеспечивая возможность дополнительной коррекции в canonical multilingual setting.

Это всё ещё development evidence.

---

# 215. Branch comparison decision

Final development candidate выбирается только из:

```text
Branch A primary
Branch B Layer C primary
```

Layer P/R и RU BGE diagnostics:

```text
not eligible.
```

---

# 216. Pre-run checklist

```text
[ ] new V8 dual namespace created
[ ] history write guard active
[ ] git state saved
[ ] historical hashes BEFORE saved

[ ] V4.4 EN preservation artifacts resolved
[ ] Layer P replay PASS

[ ] canonical V8 protocol frozen
[ ] Branch A baseline replay PASS

[ ] V44 common core schema recovered
[ ] V44 parameter family recovered exactly
[ ] support global eligibility decided technically
[ ] no EN hard-coded weights reused on RU

[ ] FullLEX recovered exactly
[ ] PairLEX exact V7 semantics recovered
[ ] SEM_COMMON decided before EVAL
[ ] Q-A contract recovered

[ ] residual normalization frozen
[ ] interactions frozen

[ ] alpha grid frozen
[ ] harm C grid frozen
[ ] threshold grids frozen

[ ] F0-F5 definitions frozen
[ ] Branch B PRIMARY definition frozen by technical SEM availability

[ ] Layer R diagnostic frozen
[ ] RU BGE diagnostic policy frozen

[ ] outer/inner folds frozen
[ ] meta OOF rules frozen
[ ] question weighting frozen

[ ] success/safety/non-degeneracy criteria frozen
[ ] branch selection hierarchy frozen

[ ] SciER cross-cell embargo PASS
[ ] cross-corpus overlap audit PASS

[ ] resource benchmark frozen

[ ] unit tests PASS
[ ] smoke PASS

[ ] Branch A contract hash saved
[ ] Branch B contract hash saved
[ ] no full outer EVAL viewed before hashes
```

---

# 217. Post-run checklist

```text
[ ] Layer P preservation complete
[ ] Branch A canonical complete
[ ] Branch B Layer C canonical complete
[ ] Layer R EN diagnostic complete

[ ] optional RU BGE diagnostic complete or BLOCKED

[ ] all five canonical cells complete
[ ] all folds/seeds complete

[ ] F0-F5 outputs complete
[ ] SEM status consistent globally

[ ] Token F1 complete
[ ] evidence metrics complete
[ ] bootstrap complete

[ ] feature attribution complete
[ ] coefficient analysis complete
[ ] switch/harm analysis complete
[ ] fold/seed stability complete

[ ] canonical Branch A/B comparison complete
[ ] historical protocol labels clear
[ ] Layer R not mixed into universal selection

[ ] resource benchmark complete

[ ] historical hashes AFTER saved
[ ] BEFORE == AFTER
[ ] git diff saved

[ ] all reports created
[ ] reproducibility report complete
[ ] COMPLETED only after all validity checks
```

---

# 218. Краткая инструкция ИИ-агенту

```text
V8 has TWO PRIMARY branches.

BRANCH A:
V8-HRR from V7 S2 + recovered V4.4 reader signals.
Keep the already frozen canonical design.

BRANCH B:
V8-V44-AF.

IMPORTANT:
separate exact historical V4.4 preservation from the new universal test.

Layer P:
exactly replay V4.4 English historical protocol.
No new training logic.

Layer C:
use the SAME canonical V8/V7 protocol as Branch A on all five cells.
This is the only Branch B result eligible for universal comparison.

Build one common V44-CORE:
B0 + reader agreement + support only if support has identical semantics
on all five cells.

Recover the exact V4.4 scoring/parameter family.
Do not copy English numeric weights onto RU.
Fit/select V44-CORE parameters inside TRAIN/CAL using the same procedure.

V44-CORE selects anchor C44.

Then learn residual utility relative to C44.

Common residual signals:
- FullLEX: exact V4.4 full-answer lexical signal
- PairLEX: exact V7 PairLEX semantics, kept separate
- Q-A relevance
- SEM only if technically common across all five cells

Use the SAME feature schema on EN and RU.

Use Ridge expected-utility regression.
Strong L2 regularization lets unnecessary components shrink.

This is how LEX may get:
large learned weight on RU,
small learned weight on EN,
WITHOUT manual language rules.

Primary ladder:
F0 V44-CORE
F1 + FullLEX
F2 + PairLEX
F3 + Q-A
F4 + SEM if SEM_COMMON
F5 + harm-risk   <-- PRIMARY

If SEM_COMMON=false:
PRIMARY is F3 + harm.
Do not choose based on EVAL quality.

Primary strong objective:
F5-F0 > 0 on ALL FIVE canonical cells
+ evidence safety.

Also report:
EN preservation + RU improvement criterion.

Layer R:
apply the SAME Branch B architecture inside exact historical V4.4 EN
protocol only as a retention diagnostic.
Do not use Layer R for universal branch selection.

Only compare Branch A vs Branch B using canonical Layer C results.

No best-per-dataset assembly.
No manual RU/EN coefficients.
No post-EVAL feature removal.

If neither primary branch passes:
do not continue unrestricted feature search on the same development data.
```
