# Final Official-Split Benchmark / H2 Family Confirmation
## Инструкция для ИИ-агента по реализации финального эксперимента

**Назначение:** финальное стандартизованное сравнение предложенного метода и его наиболее успешных версий с базовыми, сильными внешними и генеративными подходами на официальных `train / validation / test` разделениях используемых датасетов.

**Primary proposed method:** `H2`.

**Secondary proposed/challenger methods:** `V7-S2`, `H2-ACS`, `H3 fixed`.

**Главный принцип:** `test` не используется ни для выбора архитектуры, ни для выбора модели, ни для настройки параметров, ни для выбора порогов, ни для отбора компонентов. Все решения принимаются до первого просмотра test-метрик.

**Reviewed revision:** уточнены режимы official split, единая evaluation population, LOCAL-only обучение, EN-only статус исходного ModernBERT, exact H2-ACS final-fit procedure, test-overlap audit по ID и текстовым hash, resource accounting, статистические claim tiers и требования к train/validation/test label availability.

---

# 0. Рабочее название

```text
FINAL-OFFICIAL-SPLIT-BENCHMARK
H2-FAMILY-CONFIRMATION
```

---

# 1. Цель эксперимента

Проверить на официальных split:

1. превосходит ли `H2` простые extractive QA baselines;
2. превосходит ли `H2` стандартный chunk-ranking pipeline;
3. превосходит ли `H2` сильный внешний reranker;
4. можно ли получить сопоставимое/лучшее качество простой заменой reader на современный encoder QA;
5. сохраняется ли положительный эффект линии `V7-S2 → H2 → H2-ACS`;
6. способен ли `H3 fixed` сохранить высокие F1, близкие к сильным V4.4 development-вариантам;
7. как proposed extractive methods соотносятся с генеративным baseline по quality/resource trade-off.

---

# 2. Научная формулировка

Не использовать формулировку:

> «H2 является абсолютно лучшим методом».

Допустимо:

> «H2 показал лучший / один из лучших результатов среди заранее зафиксированного набора сравниваемых методов на официальном test-разделении».

Если статистического превосходства нет:

> «H2 статистически сопоставим с ...».

---

# 3. Критический аудит независимости test

До любого quality-run сравнить:

```text
official TEST document IDs
vs
all historical V1–V10 manifests / folds / evaluation sets
```

Сохранить результат в:

```text
audit/test_exposure/
```

---

# 4. Статус split и test

Для каждого dataset сначала определить `split_mode`:

```text
OFFICIAL_TRAIN_VAL_TEST
OFFICIAL_TRAIN_TEST_INTERNAL_VAL
NO_OFFICIAL_TEST
OFFICIAL_SPLIT_UNRESOLVED
```

Отдельно для test присвоить:

```text
UNTOUCHED
EXPOSED_TO_RESEARCH_PROGRAM
NOT_APPLICABLE_NO_OFFICIAL_TEST
```

Не смешивать понятия «нет official validation» и «нет official test».

---

# 5. Когда test считается UNTOUCHED

Только если official test documents никогда не использовались в:

```text
training
validation
outer evaluation
diagnostics
hypothesis generation
V1–V10
```

---

# 6. Если official test уже использовался ранее

Эксперимент всё равно выполняется, но его статус:

```text
FINAL_STANDARDIZED_BENCHMARK
NOT_INDEPENDENT_CONFIRMATION
```

Не называть его независимым подтверждением.

---

# 7. Если часть test действительно untouched

Для соответствующих datasets отдельно report:

```text
INDEPENDENT_TEST_CONFIRMATION
```

Не распространять этот статус на остальные datasets.

---

# 8. Датасеты / cells

Использовать:

```text
SciREX
SciER 384/50
SciER 128/24
RuSERRC
SciMDIX
```

---

# 9. SciER lineage

`SciER 384/50` и `SciER 128/24` относятся к одной source corpus family.

Official source-document split обязан быть одинаковым для обеих chunking cells.

---

# 10. Source-of-truth official split

Для каждого dataset определить official split source:

```text
dataset distribution
repository files
official split metadata
```

Сохранить:

```text
audit/official_split_source.json
```

Не называть внутренний split «official».

---

# 11. Если official validation отсутствует

Если источник задаёт:

```text
official TRAIN
official TEST
```

но не задаёт `validation`, использовать:

```text
split_mode = OFFICIAL_TRAIN_TEST_INTERNAL_VAL
```

и создать `INTERNAL_VAL` **только внутри official TRAIN**:

```text
document-grouped
deterministic
seed = 42
до model fitting
```

После выбора параметров:

```text
refit on full official TRAIN
```

то есть `TRAIN + INTERNAL_VAL` означает восстановление полного official TRAIN.

Official TEST остаётся закрыт.

---

## 11.1. Если official TEST отсутствует

Если corpus не имеет официального test-разделения:

```text
split_mode = NO_OFFICIAL_TEST
```

Такой dataset нельзя использовать для утверждения:

```text
official-test confirmation
independent official-test confirmation
```

Разрешён отдельный заранее зафиксированный standardized document split, но его результаты должны идти в отдельный блок:

```text
STANDARDIZED_NONOFFICIAL_HOLDOUT
```

и не смешиваться с official-test confirmatory statistics.

---

## 11.2. Если official split неоднозначен

Если существует несколько competing official variants и source-of-truth нельзя однозначно установить:

```text
OFFICIAL_SPLIT_UNRESOLVED
```

Dataset блокируется до явного решения.

---

# 12. Split audit artifact

Для каждой cell сохранить:

```text
split_mode

train document IDs
val/internal-val document IDs
test document IDs

question counts
document counts

canonical normalized-text SHA-256 per document
DOI/title/source-id where available

source reference
manifest SHA-256
```

Для `NO_OFFICIAL_TEST` хранить separate standardized-holdout manifest.

---

# 13. Split disjointness и duplicate audit

Обязательные assertions по canonical source-document IDs:

```text
train ∩ val = ∅
train ∩ test = ∅
val ∩ test = ∅
```

Дополнительно проверить exact duplicate leakage по:

```text
normalized full-text SHA-256
DOI / canonical publication ID where available
```

Если одинаковый документ имеет разные IDs:

```text
treat as overlap
```

и зафиксировать до quality evaluation.

Near-duplicate detection допускается только как заранее заданный audit и не должна удалять test documents post-hoc после просмотра качества.

---

# 14. Test firewall

До frozen contract запрещено:

```text
читать test labels
считать test F1
считать test confidence distributions
смотреть test errors
подбирать thresholds по test
подбирать K по test
выбирать model/checkpoint по test
```

Допускается только metadata/ID audit без gold-dependent analytics.

---

# 15. Общая последовательность

```text
PHASE A: split/exposure audit
PHASE B: TRAIN development
PHASE C: VAL final selection
PHASE D: freeze
PHASE E: refit TRAIN+VAL
PHASE F: blind TEST prediction
PHASE G: TEST evaluation
PHASE H: statistics/resources/report
```

---

# 16. TRAIN stage

Для `OFFICIAL_TRAIN_VAL_TEST`:
- `TRAIN` = official train;
- `VAL` = official validation.

Для `OFFICIAL_TRAIN_TEST_INTERNAL_VAL`:
- `TRAIN` = deterministic training portion official train;
- `VAL` = internal validation portion official train.

На `TRAIN`:
- обучать trainable components;
- выполнять inner document-grouped CV;
- подбирать internal hyperparameters только в заранее заданных grids;
- не придумывать новые features/branches.

Все trainable proposed/meta systems работают в режиме:

```text
LOCAL
```

то есть **отдельная модель для каждой dataset/cell**.

Запрещено pooled/cross-dataset training в primary benchmark.

---

# 17. VAL stage

На `VAL` разрешено:

- окончательно выбирать hyperparameters;
- выбирать thresholds;
- выбирать epoch;
- выбирать prompt/decode variant;
- выбирать K только если он не frozen исторически.

После просмотра VAL запрещено создавать новую архитектуру или dataset-specific rule.

---

# 18. Freeze

До test зафиксировать:

```text
system list
model IDs
chunking
K
question texts
feature schemas
thresholds
prompt
decoding
random seeds
metrics
statistics
resource procedure
```

в:

```text
frozen_contract.yaml
```

и сохранить SHA-256.

---

# 19. TRAIN+VAL refit

После freeze:

Для `OFFICIAL_TRAIN_VAL_TEST`:

```text
refit/retrain on official TRAIN + official VAL
```

Для `OFFICIAL_TRAIN_TEST_INTERNAL_VAL`:

```text
refit/retrain on full official TRAIN
```

с уже выбранными hyperparameters/threshold procedure.

Для meta-methods, где component selection является частью frozen training algorithm, допускается заново выполнить только **внутреннюю TRAIN+VAL OOF-selection**, не используя TEST.

TEST остаётся закрыт.

---

# 20. TEST

После TRAIN+VAL refit:

```text
generate TEST predictions exactly once
```

Желательно сначала получить все predictions без расчёта quality metrics, затем запустить единый evaluation stage.

---

# 21. Historical protection

Все V1–V10 artifacts/readouts:

```text
read-only
```

---

# 22. Новый namespace

```text
experiments/final_official_split_benchmark/
configs/final_official_split_benchmark/
artifacts/final_official_split_benchmark/
logs/final_official_split_benchmark/
```

---

# 23. Historical hash guard

Создать:

```text
historical_hashes_before.json
historical_hashes_after.json
```

Требование:

```text
before == after
```

---

# 24. Общая предобработка

Где применимо, у всех методов должны совпадать:

```text
document preprocessing
text normalization
canonical chunking
question formulation
answer normalization
evaluation code
```

---

## 24.1. Единая evaluation population

До TEST зафиксировать один и тот же eligibility rule для всех real systems.

Primary population должна совпадать с canonical historical H2 evaluation contract.

Если исторически primary metrics считались только по answerable questions:

```text
PRIMARY = canonical answerable subset
```

для **всех** систем.

Дополнительно можно report all-question/no-answer metrics только как secondary и только если все сравниваемые systems имеют совместимую no-answer semantics.

Запрещено:
- исключать test question из одного метода, но оставлять в другом;
- отбрасывать failed/empty prediction;
- менять denominator между systems.

Empty/no-answer policy должна быть frozen до TEST.

---

# 25. Исключение full-context systems

ModernBERT long-context и Qwen могут использовать другой context regime.

В отчёте обязательно явно пометить:

```text
different context regime
```

---

# 26. Primary metric

```text
Token F1
```

Использовать exact canonical evaluator из V9/V10.

---

# 27. Secondary metrics

Если exact implementation доступен:

```text
Character F1
Unicode ROUGE-L
Exact Match
Lemma F1 RU
```

Не подставлять отсутствующие значения нулями.

---

# 28. Evidence metrics

Где candidate-level evidence labels существуют **или воспроизводятся exact frozen historical labeling pipeline**:

```text
Hit@1
Hit@K
HN@1
selected SAFE_EVIDENCE rate
selected UNSAFE_EVIDENCE rate
F1 | selected positive
```

До TRAIN сохранить:

```text
evidence_label_availability.json
evidence_label_mapping.json
```

Если labels недоступны для dataset:
- не выводить соответствующие evidence metrics;
- не подменять их эвристикой;
- пометить методы, которым такие labels нужны для training/selection, как `BLOCKED_MISSING_REQUIRED_LABELS`.

---

# 29. Resource metrics

Inference — обязательно:

```text
end-to-end median latency/question
P95 latency/question
throughput
peak VRAM
CPU RSS

loaded neural parameter count
trainable parameter count

QA neural forward count
reranker forward count
```

Latency scope:

```text
question + already-preprocessed canonical document/chunks
→ final answer
```

Исключить:
- one-time model loading;
- offline dataset parsing/chunk materialization.

Secondary для trainable systems:

```text
training GPU-hours
training peak VRAM
number of optimization steps
```

Это особенно важно для сравнения frozen-reader H2 с fine-tuned ModernBERT.

---

# 30. System groups

```text
BASELINES
EXTERNAL STRONG BASELINES
PROPOSED FAMILY
GENERATIVE BASELINE
DIAGNOSTIC ORACLES
ABLATIONS
```

---

# 31. B1 — QA-only all chunks

```text
B1 = QA-ALL-CHUNKS-MAXCONF
```

Pipeline:

```text
document
→ all canonical chunks
→ canonical frozen QA reader
→ choose answer with maximum QA confidence
```

Без B0/reranker.

---

# 32. B1 purpose

Показывает обычный extractive QA baseline без специального chunk ranking и post-processing.

---

# 33. B1 candidate universe

Использовать все canonical chunks документа.

Не подменять B1 на B0 top-K.

---

# 34. QA confidence definition

Использовать exact canonical confidence из существующего QA implementation.

Если единого API нет:
- определить deterministic span-confidence formula на TRAIN/VAL;
- заморозить до TEST;
- использовать одинаково в B1/B3/B4b.

Отдельно report:

```text
QA confidence is a ranking proxy, not calibrated P(correct)
```

и не калибровать confidence по TEST.

---

## 34.1. Empty/failure policy for confidence baselines

Для B1/B3/B4b:
- empty answer остаётся prediction;
- model error не приводит к исключению question;
- если все candidate answers empty, вернуть canonical empty answer;
- question остаётся в denominator.

Не разрешать method-specific filtering TEST rows.

---

# 35. B2 — B0 top-1 + QA

```text
B2 = B0-TOP1-QA
```

Pipeline:

```text
B0
→ top-1 chunk
→ canonical frozen QA
→ answer
```

---

# 36. B3 — B0 top-K + max QA confidence

```text
B3 = B0-TOPK-QA-MAXCONF
```

Pipeline:

```text
B0
→ top-K
→ canonical frozen QA on all candidates
→ max QA confidence
```

---

# 37. Почему B3 ключевой

У B3 и H2 совпадают:

```text
candidate pool
top-K
QA reader
основная neural computation
```

Различается final selection algorithm.

---

# 38. K policy

Для B2/B3/proposed family использовать одну canonical K policy.

K должен быть frozen до TEST.

---

# 39. B4 — strong external reranker

Primary external reranker:

```text
BAAI/bge-reranker-v2-m3
```

Это multilingual comparator.

До TEST сохранить:

```text
model revision / commit hash
tokenizer revision
library version
```

Если exact checkpoint недоступен:

```text
BLOCKED_EXTERNAL_RERANKER
```

Не заменять другим reranker после просмотра TEST.

---

# 40. B4a — BGE top-1

```text
question + chunks
→ BGE reranker
→ top-1
→ canonical QA
```

---

# 41. B4b — BGE top-K

```text
question + chunks
→ BGE reranker
→ top-K
→ canonical QA
→ max QA confidence
```

---

# 42. BGE fairness

BGE frozen.

Никакого fine-tuning по TEST.

Query = exact thematic question.

Passage = canonical chunk text.

---

# 43. Optional EN historical reranker

Secondary only:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

Не включать в primary multilingual summary.

---

# 44. B5-EN — ModernBERT-QA

```text
B5-EN = MODERNBERT-QA-EN
```

Современный сильный **англоязычный** reader baseline.

Исходный `answerdotai/ModernBERT-*` pretrained checkpoint относится к English/code pretraining и не должен использоваться как общий RU/EN reader comparator.

---

# 45. B5-EN purpose

На EN cells проверяет:

> можно ли получить сопоставимое качество простой заменой QA encoder без H2 post-processing.

B5-EN оценивается только на:

```text
SciREX
SciER 384/50
SciER 128/24
```

Он **не входит** в universal five-cell family-balanced summary.

---

# 46. ModernBERT checkpoint

До TEST выбрать и заморозить exact:

```text
answerdotai/ModernBERT-base
или
answerdotai/ModernBERT-large
```

через TRAIN/VAL protocol.

Нельзя выбирать base/large по TEST.

Сохранить:

```text
model revision
tokenizer revision
transformers version
```

---

# 47. ModernBERT QA supervision audit

До training проверить, что TRAIN/VAL supervision корректно преобразуется в extractive QA spans.

Сохранить:

```text
total QA examples
exact span-match coverage
multiple-answer handling
unmatched-answer count
no-answer policy
negative-chunk policy
```

Не silently drop unmatched examples.

Если standard QA training требует решения по no-answer/negative chunks, правило выбирается на TRAIN/VAL и frozen до TEST.

---

# 48. ModernBERT training

Если исходный checkpoint не QA-finetuned:

```text
add standard extractive QA head
fine-tune on TRAIN
select epoch/hyperparameters on VAL
freeze
retrain from original checkpoint on TRAIN+VAL
evaluate TEST once
```

Training выполняется LOCAL отдельно по EN dataset/cell.

Это сильный in-domain trainable reader comparator, а не training-cost-matched baseline.

---

# 49. ModernBERT primary comparison regime

Primary B5-EN:

```text
canonical B0 top-K
→ ModernBERT-QA on same chunks
→ max ModernBERT QA confidence
```

Так candidate pool сохраняется, а меняется reader.

---

## 49.1. B5-RU — optional modern RU/multilingual reader

Если до TEST заранее выбран **валидный современный RU/multilingual extractive QA encoder**, создать отдельный system:

```text
B5-RU
```

с exact model/revision и тем же TRAIN→VAL→refit protocol.

Не использовать English ModernBERT как RU baseline.

Если suitable B5-RU не зафиксирован до TEST:

```text
B5-RU = NOT_RUN
```

и не считать это missing result.

---

## 49.2. B5-long optional

`ModernBERT` long-context variant:

```text
B5-EN-LONG
```

secondary only.

Может использовать full document или deterministic maximum context.

Не считать compute-matched baseline и не включать в universal primary contrasts.

---

# 50. G1 — generative baseline

Primary historical reference:

```text
G1 = Qwen/Qwen2.5-3B-Instruct
```

если этот exact checkpoint соответствует ранее зафиксированному generative baseline.

Иначе агент обязан восстановить historical exact model ID из артефактов и сохранить его до TEST.

Model weights frozen.

---

# 51. Qwen prompt/decode

Только TRAIN/VAL используются для выбора:

```text
prompt template
temperature
top_p
max_new_tokens
stop criteria
```

Не создавать dataset-specific prompt после просмотра VAL.

Primary prompt policy:

```text
one EN template shared across EN cells
one semantically equivalent RU template shared across RU cells
```

если historical protocol не задаёт ещё более строгий единый multilingual template.

После freeze TEST immutable.

---

# 52. Qwen context

Использовать full document, если помещается, иначе deterministic truncation/segmentation rule.

Rule frozen до TEST.

---

# 53. P1 — V7-S2

```text
P1 = exact V7-S2
```

Ранний proposed-family вариант с положительным development point estimate на всех пяти cells.

---

# 54. V7 source-of-truth

Не реконструировать по памяти.

Использовать historical V7 code/config/artifacts.

---

# 55. P2 — H2

```text
P2 = exact H2
```

Статус:

```text
PRIMARY PROPOSED METHOD
```

---

# 56. H2 source-of-truth

Exact definition должна совпадать с V9/V10 replay.

Если replay невозможен:

```text
STOP_H2_DEFINITION_UNRESOLVED
```

---

# 57. H2 training

Trainable meta-components:

```text
fit on TRAIN
select hyperparameters/thresholds via TRAIN/VAL protocol
freeze
refit TRAIN+VAL
TEST once
```

---

# 58. P3 — H2-ACS

```text
P3 = exact V9 H2-ACS
```

---

# 59. H2-ACS purpose

Проверяет переносимость adaptive extension на official split.

---

# 60. H2-ACS component universe

Использовать exact final V9 component universe.

Не добавлять новые optional components.

---

# 61. H2-ACS TRAIN+VAL behavior

Final-fit procedure должна быть exact и однозначной:

1. на финальном training pool (`TRAIN+VAL` либо full official TRAIN при internal-val mode) построить frozen V9 `SELECT_OOF`;
2. выбрать **одну ACS configuration для dataset/cell** только по этим OOF predictions;
3. применить exact V9 eligibility/safety/tie-break rules;
4. refit выбранную configuration на всём final training pool;
5. TEST predict once.

TEST не участвует ни в component selection, ни в safety selection.

Configuration может отличаться от TRAIN-only development choice, потому что component selection является частью frozen training algorithm.

---

# 62. P4 — H3 fixed

```text
P4 = exact V8 H3-FIXED
```

То есть:

```text
H2 + fixed Agreement group
```

---

# 63. H3 purpose

Сильный fixed challenger, который лучше сохранял высокие EN F1, близкие к V4.4 development maxima.

---

# 64. H3 status

```text
SECONDARY CHALLENGER
```

Не объявлять universal winner заранее.

---

## 64.1. Method capability matrix

До TRAIN создать:

```text
method_capability_matrix.json
```

Для каждой dataset × system указать:

```text
CAN_RUN
BLOCKED_MISSING_SPLIT
BLOCKED_MISSING_LABELS
BLOCKED_MISSING_MODEL
NOT_APPLICABLE_LANGUAGE
```

Никакой blocked system не заменять другим методом после просмотра TEST.

---

# 65. V4.4 historical maxima

Report только как:

```text
HISTORICAL DEVELOPMENT REFERENCE
```

в appendix.

Не смешивать их с official TEST metrics.

---

# 66. Почему V4.4 best не primary system

Лучшие V4.4 variants различались по dataset/cell и не образовывали одну универсальную frozen system.

Поэтому в final universal benchmark сильную линию представляет H3 fixed.

---

# 67. D1 — QA oracle

```text
D1 = QA-ORACLE-TOPK
```

Для каждого question выбирает top-K QA answer с максимальным gold Token F1.

Diagnostic only.

---

# 68. D2 — positive evidence oracle

```text
D2 = POSITIVE-EVIDENCE-ORACLE
```

Если sufficient-positive evidence присутствует, выбрать best QA answer среди positive chunks.

Diagnostic only.

---

# 69. H2 ablation A1

Обязательная:

```text
A_H2_NO_FULLLEX
```

---

# 70. H2 ablation A2

Если exact removal поддерживается:

```text
A_H2_NO_PAIRLEX
```

---

# 71. H2 ablation A3

Если exact deterministic ablation можно определить:

```text
A_H2_NO_RISK
```

Иначе:

```text
BLOCKED_NOT_EXACT
```

---

# 72. Ablations status

Ablations secondary only.

Не участвуют в выборе primary method.

---

# 73. Primary hierarchy

До TEST frozen:

```text
PRIMARY:
P2 H2

SECONDARY CHALLENGERS:
P1 V7-S2
P3 H2-ACS
P4 H3-FIXED
```

---

# 74. Если challenger лучше H2

Честно report:

```text
best observed official-test system = challenger
```

Не переписывать primary hypothesis post-hoc.

---

# 75. TRAIN inner CV

Для trainable meta systems внутри каждой dataset/cell:

```text
4 document-grouped folds
```

если размер и class support позволяют.

Fallback 3 folds разрешён только если необходимость определяется **по структуре TRAIN до просмотра VAL quality**.

Сохранить:
- split manifest;
- reason;
- class counts.

Не использовать данные другой dataset/cell для компенсации малого TRAIN.

---

# 76. Historical grids

Для V7/H2/H3/H2-ACS использовать frozen historical grids/procedures.

Не расширять search space в final benchmark.

---

# 77. ModernBERT / optional B5-RU grid

До VAL зафиксировать небольшой grid по:

```text
learning_rate
batch size / accumulation
epochs
max context length
doc stride
```

Exact values записать в frozen contract.

---

# 78. BGE training

Primary B4:

```text
frozen model
no fine-tuning
```

---

# 79. Reader fairness

Для B1/B2/B3/B4/P-family использовать один canonical frozen QA reader, если method definition не требует иного.

---

# 80. Fairness matrix

Создать таблицу:

```text
method
applicable languages/cells
training data
training labels
chunking
candidate source
candidate count
reader
reader trainable?
reranker
post-processing
whole-document context?
neural forwards
```

---

# 81. No hidden advantage

Без явного system-specific основания не менять между B2/B3/B4/P-family:

```text
chunk overlap
question wording
answer normalization
max answer length
```

---

# 82. Training seeds

Primary stochastic trainable systems:

```text
42,43,44
```

Если compute объективно не допускает 3 seeds для дорогого external baseline, это решение должно быть принято **до TEST**, отражено в contract и не должно зависеть от quality.

Основной H2/proposed-family benchmark обязан сохранить 3 seeds.

---

# 83. Deterministic systems

Frozen deterministic B1/B2/B3/B4:

```text
one prediction set
```

Не дублировать fake seeds.

---

# 84. ModernBERT seeds

```text
42,43,44
```

---

# 85. Qwen seeds

Предпочтительно deterministic decoding.

Если decoding stochastic — predeclare seeds до TEST.

---

# 86. Prediction schema

Минимум:

```text
dataset
split
document_id
question_id
aspect
system_id
seed
evaluation_eligible
candidate IDs
selected candidate
answer
confidence
fallback/failure reason
runtime metadata
evaluation metadata
```

---

# 87. Runtime/evaluation separation

Runtime pipeline не получает:

```text
gold answer
evidence label
oracle rank
```

---

# 88. TRAIN artifacts

Сохранить:

```text
inner-CV predictions
hyperparameter tables
models
feature schemas
training logs
```

---

# 89. VAL artifacts

Сохранить:

```text
all evaluated configs
VAL metrics
selected config
selected thresholds
epoch choice
prompt choice
```

---

# 90. Frozen contract contents

Минимум:

```text
dataset split hashes
split modes
test exposure status
evaluation population
method capability matrix
evidence label availability/mapping
system registry
model checkpoints
chunking
K
questions
confidence function
V7/H2/H3 schemas
ACS procedure
ModernBERT config
BGE model ID
Qwen prompt/decoding
metrics
seeds
bootstrap
resource procedure
primary contrasts
claim criteria
```

---

# 91. TEST unlock assertion

Перед TEST:

```text
assert frozen_contract exists
assert SHA256 exists
assert no unresolved system config
assert TRAIN+VAL refit complete
```

---

# 92. Blind TEST prediction

Сначала получить все real-system TEST predictions без расчёта aggregate F1.

После completion всех systems запустить evaluation.

---

# 93. Test rerun policy

Разрешён только при:

```text
crash
corrupt artifact
proven implementation bug
```

Не при низком качестве.

---

# 94. Bug policy

Если bug обнаружен после частичного TEST:

1. mark run INVALID;
2. document bug;
3. fix;
4. rerun all affected systems/datasets.

---

# 95. Primary comparisons

Universal five-cell primary contrasts:

```text
H2 vs B1
H2 vs B2
H2 vs B3
H2 vs B4b
```

Separate EN-reader primary contrast:

```text
H2 vs B5-EN
```

только на:

```text
SciREX
SciER384
SciER128
```

Если B5-RU predeclared and run, его comparison — отдельный RU reader contrast.

Не вычислять фиктивный five-cell `H2 vs ModernBERT` contrast.

---

# 96. Secondary comparisons

```text
H2 vs V7-S2
H2 vs H2-ACS
H2 vs H3
H2 vs B4a
H2 vs Qwen
```

---

# 97. Key role B3

Если H2 > B3, выигрыш нельзя объяснить только наличием top-K и тем же QA reader.

---

# 98. Key role B4b

Если H2 > B4b, выигрыш нельзя объяснить только сильным external reranker.

---

# 99. Key role B5-EN

Если H2 > B5-EN на EN-family summary, выигрыш на английских данных нельзя объяснить только заменой reader backbone.

Это не является утверждением о RU.

---

# 100. Qwen interpretation

Главный вопрос:

```text
quality-resource trade-off
```

Не требовать H2 > Qwen по F1 как обязательное условие.

---

# 101. Per-dataset reporting

Каждая cell — отдельная строка.

Не скрывать отрицательные datasets global mean.

---

# 102. Family-balanced metric

Для systems, применимых ко всем пяти cells:

```text
SciREX
SciER
RuSERRC
SciMDIX
```

SciER effect:

```text
mean(SciER128, SciER384)
```

Equal family weights.

Вычислять отдельно:

```text
ALL_STANDARDIZED family-balanced
```

по всем совместимым standardized holdouts,

и, если есть untouched official tests:

```text
UNTOUCHED_CONFIRMATION family-balanced
```

только по untouched official-test families.

Если untouched остаётся меньше двух независимых corpus families:

```text
DO NOT report a global confirmation mean
```

а report только per-family/per-cell confirmation results.

Не смешивать `NO_OFFICIAL_TEST` standardized holdout с official-test confirmation claim.

Каждый superiority/non-inferiority claim должен иметь scope tag:

```text
STANDARDIZED
или
UNTOUCHED_CONFIRMATION
```

Confirmatory language разрешён только для `UNTOUCHED_CONFIRMATION`.

---

# 103. Seed aggregation

Для stochastic methods:

1. metric contribution per question/seed;
2. average seeds within question;
3. compute dataset metric.

Seeds не independent observations.

---

# 104. Bootstrap

Paired document-cluster bootstrap:

```text
iterations = 10000
bootstrap_seed = 42
```

---

# 105. SciER coupled bootstrap

Один resampled multiset canonical source-document IDs применять одновременно к SciER128 и SciER384.

---

# 106. Test CI

Для key contrasts:

```text
ΔTokenF1
95% paired bootstrap CI
```

---

# 107. Multiple comparisons

Primary contrasts frozen заранее.

Universal primary family:

```text
H2 vs B1
H2 vs B2
H2 vs B3
H2 vs B4b
```

EN-reader primary family:

```text
H2 vs B5-EN
```

Если вычисляются p-values:
- Holm correction по четырём universal primary contrasts;
- EN-reader contrast report отдельно как заранее заявленный domain-specific contrast.

Основные выводы всё равно строятся на effect size + CI.

---

# 108. Claim: H2 beats simple baselines

До TEST установить:

```text
CELL_MATERIAL_DROP = -1.0 Token F1
```

Для каждого comparator B1/B2/B3:

`OBSERVED_BETTER`:

```text
family-balanced Δ(H2-comparator) > 0
```

`SUPPORTED_SUPERIORITY`:

```text
95% CI lower bound of family-balanced Δ > 0
AND
no cell point delta < -1.0 F1
```

`CONSISTENT_SUPPORTED_SUPERIORITY`:
дополнительно:

```text
positive point delta on >=4/5 cells
```

Использовать именно эти labels, а не расплывчатое «лучше».

---

# 109. Claim: beyond strong reranker

`SUPPORTED_BEYOND_RERANKER`, если:

```text
family-balanced Δ(H2-B4b) > 0
95% CI lower bound > 0
no cell point delta < -1.0 F1
```

Иначе report observed point difference без superiority claim.

---

# 110. Claim: beyond stronger reader

Для EN:

```text
EN-family Δ(H2-B5-EN)
```

где EN families:

```text
SciREX
SciER
```

Если используется non-inferiority:

```text
LCB95(H2-B5-EN) > NI_MARGIN
```

Ресурсное преимущество report отдельно численным ratio; не использовать неопределённое «заметно дешевле» без заранее заданного resource threshold.

---

# 111. Non-inferiority margin

Secondary predeclared margin:

```text
NI_MARGIN = -1.0 Token F1
```

Non-inferiority supported только если:

```text
LCB95(H2-comparator) > -1.0
```

Margin применяется только к заранее перечисленным NI contrasts:
- H2 vs B5-EN;
- H2 vs Qwen, если quality-resource NI анализ включён в frozen contract.

Не применять margin post-hoc к другим systems.

---

# 112. Proposed-family evolution

Обязательно report:

```text
V7-S2
H2
H2-ACS
H3
```

---

# 113. Desired, но не обязательный pattern

```text
V7-S2 < H2 <= H2-ACS
```

Не подгонять систему под этот результат.

---

# 114. H3 interpretation

Если H3 выигрывает EN, но проигрывает RU/другим cells:

> fixed Agreement remains non-universal.

Если выигрывает global — report как strongest observed challenger без post-test retuning.

---

# 115. Patchwork prohibition

Нельзя после TEST собрать:

```text
H3 для SciREX
H2 для RuSERRC
ACS для SciER128
...
```

и назвать это одним universal method.

---

# 116. Oracle reporting

D1/D2 не участвуют в рейтинге real systems.

Используются только для headroom/ceiling analysis.

---

# 117. Resource hardware

Все systems измерять:

```text
same machine
same GPU
same precision policy
same software environment
```

---

# 118. Fresh process

Каждый method resource benchmark:

```text
fresh process
```

---

# 119. Warmup/timing

```text
warmup >= 10 questions
3 timing repeats
CUDA synchronize before/after timed section
```

---

# 120. Resource sample

Предпочтительно весь TEST.

Если невозможно — frozen sample создаётся до просмотра resource results и одинаков для сравниваемых methods.

---

# 121. Qwen resource extras

Report:

```text
input tokens
output tokens
generated tokens/sec
```

---

# 122. H2 resource decomposition

Отдельно:

```text
B0 cost
QA top-K cost
meta/post-processing cost
```

---

# 123. Main result table

```text
System
Applicability
SciREX F1
SciER384 F1
SciER128 F1
RuSERRC F1
SciMDIX F1
Family-balanced F1 (where defined)
Median latency
Peak VRAM
Loaded params
Trainable params
Training GPU-hours where applicable
```

Для language-inapplicable system ставить:

```text
N/A
```

а не 0 и не пропускать cell из universal average молча.

---

# 124. Statistical table

Для H2 vs each primary comparator:

```text
family-balanced delta
95% CI
per-cell deltas
per-cell CIs
```

---

# 125. Proposed-family table

```text
V7-S2
H2
H2-ACS
H3
```

с:

```text
per-cell F1
global F1
evidence safety
resources
```

---

# 126. Evidence table

Per dataset/system:

```text
Hit@1
HN@1
SAFE rate
UNSAFE rate
```

---

# 127. Resource-quality frontier

Для B3/B4b/H2/Qwen и отдельно B5-EN на EN data построить:

```text
x = latency или VRAM
y = Token F1
```

---

# 128. Pareto interpretation

System Pareto-dominated, если другой:
- не хуже F1;
- не дороже по выбранному ресурсу;
- строго лучше хотя бы по одному критерию.

---

# 129. Ablation table

```text
H2 full
H2 no FullLEX
H2 no PairLEX
H2 no risk (if exact)
```

---

# 130. Component claim

Компонент считается поддержанным, если его removal снижает family-balanced F1 и направление достаточно согласовано по cells.

---

# 131. Exposure table in final report

Обязательно:

```text
Dataset
Official split source
Test exposure status
Independent confirmation? yes/no
```

---

# 132. Если все official tests exposed

Итоговый статус:

```text
FINAL_STANDARDIZED_OFFICIAL_SPLIT_BENCHMARK
```

не `INDEPENDENT_CONFIRMATION`.

Cells с `NO_OFFICIAL_TEST` имеют отдельный статус:

```text
STANDARDIZED_NONOFFICIAL_HOLDOUT
```

и не должны называться official-split results.

---

# 133. Если есть untouched subset

Разделить выводы:

1. standardized benchmark across all datasets;
2. independent confirmation only on untouched subset.

---

# 134. Если ни одного untouched test нет

Настоящее независимое подтверждение требует нового корпуса / новых документов, не использовавшихся в V1–V10.

---

# 135. Artifact root

```text
artifacts/final_official_split_benchmark/run_<timestamp>/
```

---

# 136. Artifact structure

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
    official_split_source.json
    split_manifests/
    split_hashes.json
    test_exposure/
    system_registry.json
    method_capability_matrix.json
    evaluation_population.json
    evidence_label_availability.json
    evidence_label_mapping.json
    preprocessing_hashes.json
    confidence_definition.json

  train/
    inner_cv/
    models/
    logs/

  validation/
    predictions/
    metrics/
    selections/
    final_choices.json

  refit_train_val/
    models/
    logs/
    model_hashes.json

  test_predictions/
    B1/
    B2/
    B3/
    B4a/
    B4b/
    B5/
    B5_long/
    G1/
    P1/
    P2/
    P3/
    P4/
    ablations/
    oracles/

  test_analysis/
    primary_metrics.json
    secondary_metrics.json
    evidence_metrics.json
    paired_bootstrap.json
    family_bootstrap.json
    primary_contrasts.json
    proposed_family.json
    oracle_analysis.json
    ablation_analysis.json

  resources/
    raw/
    summary.json
    environment.json

  reports/
    final_benchmark_results.md
    statistical_analysis.md
    resource_analysis.md
    reproducibility.md
```

---

# 137. System registry

Создать machine-readable:

```text
system_registry.json
```

с exact pipeline definition каждого system ID.

---

# 138. Model provenance

Для trained/refit model сохранить:

```text
config hash
checkpoint hash
training split hash
seed
selected hyperparameters
```

---

# 139. Prediction immutability

После создания TEST predictions:

```text
hash file
never overwrite
```

---

# 140. Unit tests — split/firewall

```text
test_official_split_loaded
test_split_document_disjoint
test_scier_split_shared
test_split_hash_stable
test_split_text_hash_disjoint
test_split_doi_disjoint_where_available
test_test_exposure_audit_complete

test_same_evaluation_population_all_systems
test_empty_answer_policy_frozen

test_no_test_gold_in_train
test_no_test_gold_in_val
test_test_metrics_blocked_before_freeze
test_test_predictions_blocked_before_contract
```

---

# 141. Unit tests — baselines

```text
test_B1_all_chunks
test_B1_max_confidence

test_B2_exact_top1

test_B3_exact_topK
test_B3_same_reader_as_H2

test_B4_BGE_frozen
test_B4_no_test_finetune
```

---

# 142. Unit tests — proposed family

```text
test_V7_exact_definition
test_H2_exact_definition
test_ACS_exact_V9_procedure
test_H3_exact_fixed_agreement
```

---

# 143. Unit tests — ModernBERT/Qwen

```text
test_modernbert_en_only
test_modernbert_train_only
test_modernbert_val_selection
test_modernbert_span_supervision_audit
test_modernbert_refit_train_val
test_modernbert_same_normalization

test_qwen_prompt_frozen
test_qwen_decoding_frozen
test_qwen_no_test_prompt_tuning
```

---

# 144. Unit tests — statistics/resources

```text
test_seed_aggregation
test_document_cluster_bootstrap
test_scier_coupled_bootstrap
test_family_balanced_equal_weights

test_same_hardware_metadata
test_fresh_process_flag
test_peak_vram_reset
test_forward_counts_available
```

---

# 145. Smoke phase

До TEST выполнить smoke только на TRAIN/VAL subset.

Проверить:

```text
all pipelines run
outputs parse
confidence exists
metrics work
checkpoint loading works
resource logger works
artifacts write
```

---

# 146. Smoke cannot alter science

По smoke разрешено исправлять только technical bugs.

Нельзя менять:

```text
system list
model family
comparator definition
metrics
claim criteria
```

---

# 147. Pre-TEST checklist

```text
[ ] split_mode resolved for every dataset
[ ] official/internal split source documented
[ ] ID disjointness PASS
[ ] text-hash duplicate audit PASS/reviewed
[ ] DOI/source-id duplicate audit PASS/reviewed where available
[ ] SciER linkage PASS
[ ] test exposure audit complete

[ ] evaluation population frozen
[ ] empty/no-answer policy frozen
[ ] evidence-label availability frozen
[ ] method capability matrix frozen

[ ] all system definitions frozen
[ ] B1/B2/B3 frozen
[ ] B4 model ID + revision frozen
[ ] B5-EN checkpoint/training frozen
[ ] B5-RU status/model frozen if applicable
[ ] ModernBERT supervision audit complete
[ ] Qwen exact model/prompt/decoding frozen

[ ] V7 exact
[ ] H2 exact
[ ] ACS exact
[ ] H3 exact

[ ] K frozen
[ ] confidence frozen
[ ] normalization frozen

[ ] TRAIN tuning complete
[ ] VAL selection complete
[ ] TRAIN+VAL refit complete
[ ] model hashes saved

[ ] statistics frozen
[ ] resources frozen
[ ] primary contrasts frozen
[ ] claim rules frozen
[ ] STANDARDIZED vs UNTOUCHED_CONFIRMATION scope rules frozen

[ ] frozen_contract.yaml saved
[ ] SHA-256 saved
[ ] no TEST quality metrics viewed
```

---

# 148. TEST checklist

```text
[ ] generate predictions for every real system
[ ] hash predictions
[ ] verify full question coverage
[ ] verify no missing answers
[ ] verify normalization
[ ] generate oracles separately
[ ] only then run evaluation
```

---

# 149. Post-TEST checklist

```text
[ ] per-cell metrics complete
[ ] family-balanced metrics complete
[ ] H2 primary contrasts complete
[ ] proposed-family table complete
[ ] BGE comparison complete
[ ] ModernBERT comparison complete
[ ] Qwen quality/resource comparison complete
[ ] evidence metrics complete
[ ] oracle analysis complete
[ ] ablations complete
[ ] paired bootstrap complete
[ ] SciER coupled bootstrap complete
[ ] family bootstrap complete
[ ] resource benchmark complete
[ ] Pareto analysis complete
[ ] exposure caveat included
[ ] final reports created
[ ] historical hashes unchanged
```

---

# 150. Final conclusions order

Сначала ответить:

1. H2 vs B1/B2/B3;
2. H2 vs BGE;
3. H2 vs ModernBERT-QA on EN;
4. H2 vs optional B5-RU if predeclared;
5. H2 vs Qwen quality/resource;
6. какой proposed-family variant дал лучший observed TEST result.

Затем:

7. V7 → H2;
8. ACS transferability;
9. H3 transferability;
10. H2 ablations;
11. oracle gap.

---

# 151. Если H2 лучший

Допустимая формулировка:

> На заранее зафиксированных holdout-разделениях H2 показал наилучшее family-balanced качество среди заранее зафиксированных extractive comparator systems при указанном resource profile.

Если вывод относится только к untouched official tests, явно написать `UNTOUCHED_CONFIRMATION`.

Если в aggregate входят exposed или non-official holdouts, использовать только `STANDARDIZED` wording и соответствующую оговорку.

---

# 152. Если H2-ACS лучший

Report:

> Adaptive extension показал лучший observed test result среди proposed-family systems.

Не переписывать primary hypothesis задним числом.

---

# 153. Если H3 лучший

Report:

> Fixed Agreement challenger показал максимальный observed F1.

Отдельно проверить переносимость по datasets.

---

# 154. Если BGE лучше

Не скрывать.

Вывод:

> сильный внешний reranker объясняет значительную часть / весь прирост proposed method.

---

# 155. Если ModernBERT лучше на EN

Разделить:

```text
EN reader-quality advantage
training cost
inference resource cost
H2 competitiveness
```

Не переносить этот вывод автоматически на RU.

---

# 156. Если Qwen лучше по F1

Report:

```text
quality gap
latency gap
VRAM gap
parameter gap
```

H2 может оставаться сильным resource-efficient method только если данные это подтверждают.

---

# 157. Если H2 лучше только по resources

Формулировка:

> H2 обеспечивает более выгодный quality/resource trade-off.

Не говорить «лучше по качеству».

---

# 158. Если proposed family проигрывает простым baselines

Это falsifies ключевую value hypothesis.

Не создавать post-hoc rules для спасения результата.

---

# 159. Новый winner после TEST

Если challenger становится новым best observed system:

```text
не переоценивать его как новый primary на том же TEST
```

Его независимая проверка потребует новых untouched data.

---

# 160. Final deliverables

```text
final_benchmark_results.md
statistical_analysis.md
resource_analysis.md
reproducibility.md
system_registry.json
frozen_contract.yaml
```

---

# 161. Краткая инструкция ИИ-агенту

```text
Implement a NEW final official-split benchmark.
DO NOT MODIFY V1–V10 historical artifacts.

FIRST:
resolve split mode for:
SciREX,
SciER384,
SciER128,
RuSERRC,
SciMDIX.

Use one of:
OFFICIAL_TRAIN_VAL_TEST
OFFICIAL_TRAIN_TEST_INTERNAL_VAL
NO_OFFICIAL_TEST
OFFICIAL_SPLIT_UNRESOLVED.

If official VAL is absent but official TEST exists:
make deterministic document-grouped INTERNAL_VAL from official TRAIN only.

If official TEST is absent:
use a separately labelled STANDARDIZED_NONOFFICIAL_HOLDOUT only if predeclared,
and NEVER call it official-test confirmation.

Audit whether official TEST document IDs appeared anywhere in V1–V10.
If yes:
mark TEST as EXPOSED_TO_RESEARCH_PROGRAM.
Still run the standardized benchmark,
but do not call it independent confirmation.

Use:
TRAIN -> internal tuning
VAL -> final hyperparameter/threshold/epoch/prompt selection
FREEZE
REFIT on final training pool:
  official TRAIN+VAL when official VAL exists,
  full official TRAIN when VAL was carved internally from TRAIN
TEST exactly once.

Systems:

B1:
all chunks -> canonical frozen QA -> max QA confidence

B2:
B0 top1 -> canonical frozen QA

B3:
B0 topK -> canonical frozen QA -> max QA confidence

B4a:
BAAI/bge-reranker-v2-m3 top1 -> canonical QA

B4b:
BAAI/bge-reranker-v2-m3 topK -> canonical QA -> max QA confidence

B5-EN:
ModernBERT-QA on the same B0 topK chunks -> max confidence.
EN cells ONLY.
Original ModernBERT is not a RU/EN universal reader.
Audit extractive span supervision before training.
Train only on TRAIN,
select epoch/config on VAL,
refit final training pool,
TEST once.

Optional B5-RU:
only if a valid modern RU/multilingual extractive QA reader is
predeclared before TEST.

Optional B5-EN-LONG:
long-context ModernBERT,
secondary only.

G1:
frozen Qwen generative baseline.
Prompt/decoding selected on TRAIN/VAL only.
Measure quality + resources.

Proposed family:
P1 = exact V7-S2
P2 = exact H2 <-- PRIMARY
P3 = exact V9 H2-ACS
P4 = exact V8 H3 fixed

Diagnostics:
D1 QA oracle topK
D2 positive-evidence oracle

Ablations:
H2 no FullLEX
H2 no PairLEX
H2 no risk only if exact ablation exists.

Primary universal comparisons:
H2 vs B1
H2 vs B2
H2 vs B3
H2 vs B4b

Separate EN primary reader comparison:
H2 vs B5-EN on SciREX + SciER only.

Secondary:
H2 vs V7-S2
H2 vs H2-ACS
H2 vs H3
H2 vs Qwen

Primary metric:
Token F1.

Secondary:
Char F1 / Unicode ROUGE-L / EM / RU Lemma F1 where exact.

Evidence:
Hit1/HN1/SAFE/UNSAFE where labels exist.

Resources:
latency median/P95,
throughput,
VRAM,
RSS,
params,
forward counts.

Statistics:
average stochastic seeds within question,
paired document-cluster bootstrap,
10000 iterations,
SciER128/384 coupled by source document,
family-balanced equal weights over applicable universal families:
SciREX,
SciER,
RuSERRC,
SciMDIX.

Compute ALL_STANDARDIZED and UNTOUCHED_CONFIRMATION summaries separately.
Never mix a NO_OFFICIAL_TEST holdout into an official-test confirmation claim.

Generate ALL TEST predictions before inspecting quality if possible.
Never retune after TEST.

If another challenger beats H2 on TEST:
report it honestly as best observed TEST system,
but do not post-hoc redefine the primary hypothesis.

Final report MUST distinguish:
FINAL_STANDARDIZED_BENCHMARK
from
INDEPENDENT_CONFIRMATION
based on historical test exposure audit.
```
