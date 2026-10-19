# V4.5 — пересмотренный план реализации русскоязычного эксперимента
## Кандидат компактного ранкера + адаптивная коррекция остаточных ошибок

**Версия:** reviewed / corrected  
**Назначение:** техническая инструкция для код-агента  
**Статус:** продолжение линии V4.4, не новая архитектурная ветка

---

# 0. Главный принцип V4.5

V4.5 должен проверить не идею:

> «взять более слабый ранкер, чтобы второй компонент обязательно дал прирост»,

а более строгую гипотезу:

> **можно ли получить практически полезную quality–resource рабочую точку, используя менее тяжёлый русскоязычный ранкер и заранее зафиксированный лёгкий механизм коррекции его остаточных ошибок, при сохранении сильного BGE как обязательного reference.**

Критически важно:

1. выбор DiTy был сделан **после анализа V4.4** и частично мотивирован наблюдаемым поведением второго этапа;
2. поэтому результаты V4.5-A на уже использованных RuSERRC/SciMDIX являются **development characterization**, а не независимым доказательством;
3. итоговое подтверждение возможно только на действительно ранее не использовавшихся данных;
4. если независимых данных нет, это должно быть прямо отражено в диссертации.

---

# 1. Что исправлено относительно предыдущего плана

Предыдущая версия содержала несколько методологических неточностей. В этой версии исправлено следующее.

## 1.1. DiTy больше не называется заранее доказанно «ресурсно-эффективным»

До измерений используется формулировка:

```text
candidate resource-oriented ranker
```

или:

```text
кандидат на более ресурсно-эффективную рабочую точку
```

Утверждение о снижении ресурсоёмкости допускается только после end-to-end benchmark.

## 1.2. BGE не называется «верхней границей» в абсолютном смысле

Корректная формулировка:

```text
strong reference among the tested rankers
```

или:

> сильный проверенный reference среди исследованных ранкеров.

## 1.3. Не фиксируются одинаковые α/γ для разных датасетов без проверки

V4.4 показал dataset dependence.

Поэтому единый метод определяется:

```text
same architecture
+ same allowed parameter family
+ same TRAIN/CAL selection rule
```

а не обязательно одинаковыми численными коэффициентами.

## 1.4. SciMDIX не считается новой независимой репликацией

Положительный DiTy + stage2 эффект на SciMDIX уже участвовал в выборе дальнейшей гипотезы.

Повторный запуск на тех же данных:

```text
reproduction / stability check
```

но не:

```text
independent replication
```

## 1.5. Выбор DiTy не маскируется

Нельзя писать, что DiTy выбран полностью независимо от результатов V4.4.

Корректно:

> DiTy выбран как development-кандидат после анализа V4.4 с учётом размера/архитектуры, русскоязычной направленности, качества и наблюдаемой возможности дополнительной коррекции. Поэтому окончательная доказательная оценка требует новых данных.

## 1.6. Уточнена архитектура stage2

Нельзя заранее переименовывать historical `ASEM-Q5` в «простой source-support router», пока не доказана эквивалентность.

Нужно восстановить фактическую механику:

- V6 branch;
- фактический β;
- source-support γ;
- альтернативную branch;
- confidence/router feature;
- threshold rule;
- роль Q5-veto.

Упрощать архитектуру разрешено только как заранее объявленный новый candidate и только после regression analysis.

## 1.7. Resource benchmark считается end-to-end

Необходимо учитывать, что stage2 может требовать QA на top-K, тогда как BGE B0 может требовать QA только на top-1.

Поэтому меньший ranker **не гарантирует** меньшую стоимость полного pipeline.

---

# 2. Связь с глобальными выводами V4.4

V4.5 строится на следующих уже наблюдавшихся закономерностях.

## 2.1. B0 является критическим компонентом

Качество первичного ranker существенно влияет на downstream Token F1 и evidence retrieval.

## 2.2. Stage2 является residual correction, а не универсальным усилителем

Польза второго этапа зависит от:

- силы B0;
- residual error structure;
- датасета;
- K;
- доступности дополнительного evidence signal.

## 2.3. На сильном BGE stage2 может становиться избыточным

Это не считается failure.

Корректный safety mechanism должен иметь возможность:

```text
fallback → B0
```

## 2.4. На SciMDIX с DiTy найден development-сигнал stage2

Этот результат мотивирует V4.5, но не является новым независимым подтверждением.

---

# 3. Исследовательские вопросы V4.5

## RQ1 — качество первого компонента

Какое качество обеспечивает DiTy B0 на RuSERRC и SciMDIX при строгом воспроизводимом OOF-протоколе?

## RQ2 — остаточная коррекция

Даёт ли заранее зафиксированный stage2 дополнительный прирост поверх DiTy B0?

## RQ3 — междатасетная согласованность

Наблюдается ли одинаковое направление stage2 effect на двух русскоязычных корпусах?

Корректный термин:

```text
cross-dataset consistency
```

а не:

```text
independent replication
```

для V4.5-A.

## RQ4 — ресурсы

Даёт ли полный:

```text
DiTy + stage2
```

практический resource advantage относительно:

```text
BGE B0
```

при сопоставимой постановке задачи?

## RQ5 — residual-error hypothesis

Согласуются ли результаты с гипотезой:

```text
чем меньше полезных остаточных ошибок после B0,
тем меньше потенциальный gain stage2
```

Это только descriptive hypothesis, не строгий закон.

---

# 4. Фазы V4.5

## V4.5-A — development characterization

Используются уже исследованные RuSERRC/SciMDIX.

Цели:

- воспроизвести DiTy;
- реализовать frozen stage2 family;
- проверить поведение на обоих датасетах;
- измерить resource profile;
- выбрать deployable candidate;
- заморозить final configuration.

Все CI этой фазы:

```text
development CI
```

## V4.5-B — prospective independent evaluation

Запускается только если найден действительно untouched test/external set.

До его открытия:

```text
architecture frozen
ranker recipe frozen
parameter selection rule frozen
K frozen
question frozen
QA frozen
runtime pipeline frozen
```

---

# 5. Обязательный аудит независимости test

Перед любым V4.5-B создать:

```text
test_provenance_audit.json
```

Для каждого candidate test source проверить:

- присутствовали ли его document ids в предыдущих experiments;
- использовались ли labels;
- использовались ли texts;
- попадал ли он в model selection;
- использовался ли official test ранее;
- есть ли его predictions в старых artifacts.

Статусы:

```text
untouched
contaminated_for_confirmation
unknown
```

Только:

```text
untouched
```

может использоваться как V4.5-B independent test.

Если статус `unknown`:

```text
НЕ считать независимым.
```

---

# 6. Нельзя создать «новый независимый holdout» из уже просмотренных данных

Если все документы RuSERRC/SciMDIX уже входили в прошлые development experiments:

```text
новое разбиение этих же документов
≠ independent confirmation
```

Такой split разрешён для:

```text
development stability
```

но не для финального подтверждения.

Если untouched RU test отсутствует, варианты:

1. официальный split, который реально не использовался;
2. новый внешний корпус;
3. новая вручную размеченная выборка;
4. честно оставить результат как development evidence.

---

# 7. Полная изоляция V4.5

Все исторические данные и artifacts:

```text
READ ONLY
```

Разрешено:

- чтение;
- hash;
- exact cache reuse;
- использование frozen splits;
- импорт чистых helper functions;
- historical references.

Запрещено:

- overwrite;
- migration in place;
- исправление старого JSON;
- запись внутрь V4.4 directories;
- удаление;
- переименование;
- обновление old cache.

---

# 8. Каталог

Создать:

```text
experiments/qa_chunk_selection/v4_5_ru_compact_refinement/
```

Artifacts:

```text
artifacts/qa_chunk_selection_v4_5_ru/
  run_<timestamp>/
```

Logs:

```text
logs/qa_chunk_selection_v4_5_ru_<timestamp>.log
```

---

# 9. Dataset D1 — RuSERRC

Использовать exact source и exact document-grouped outer folds из V4.4-RU.

Источником истины являются:

```text
historical manifest
frozen split files
```

Не числа из отчёта.

Ожидаемые числа можно использовать только как sanity check.

Нужно сохранить:

```text
dataset_hash
candidate_universe_hash
document_ids_hash
question_ids_hash
fold_membership_hash
```

---

# 10. Dataset D2 — SciMDIX

Использовать exact V4.4 source и frozen document folds.

Создать аналогичные hashes:

```text
dataset_hash
candidate_universe_hash
document_ids_hash
question_ids_hash
fold_membership_hash
```

Если official train/test структуры существуют, нельзя предполагать, что official test untouched.

Проверить по old artifacts.

---

# 11. Candidate universe invariant

Это критическая проверка.

Для каждой модели и системы внутри одного dataset/fold:

```text
candidate chunk ids
```

должны быть одинаковыми до top-K.

Сохранить:

```text
candidate_universe_check.json
```

Проверить:

```text
DiTy candidate universe
=
historical B0 universe
=
BGE quality-reference universe
```

если выполняется парное качество-сравнение.

Если candidate universe отличается:

- paired F1 comparison пометить invalid;
- сначала восстановить причину;
- не смешивать результаты в одной paired таблице.

---

# 12. Chunking invariant

Не менять:

- chunk length;
- overlap;
- sentence boundaries;
- detokenization;
- chunk ids;
- order;
- document segmentation.

Использовать exact historical chunking соответствующего dataset.

---

# 13. Question formulation

Primary:

```text
canonical q0
```

если exact contract подтверждает q0 для dataset.

Не запускать Q5 question search.

Не выбирать вопрос на EVAL.

---

# 14. Primary ranker candidate — DiTy

Использовать exact base model id/revision из historical factorial experiment.

Агент должен получить из старого artifact:

```text
model_id
revision / commit if available
tokenizer id/revision
max sequence length
head type
training recipe
```

Если точный model id невозможно восстановить:

```text
STOP
```

Не угадывать модель по названию.

---

# 15. DiTy является development-selected candidate

В manifest записать:

```text
ranker_selection_status:
  development_selected_after_v4_4
```

Не:

```text
prespecified_independent_candidate
```

Это важно для корректной диссертационной интерпретации.

---

# 16. BGE status

BGE:

```text
BAAI/bge-reranker-v2-m3
```

используется как:

```text
strong tested reference
```

Не использовать термины:

```text
upper bound
oracle
best possible
```

---

# 17. Historical models

Можно использовать как descriptive references:

```text
old B0
ModernBERT
DiTy
BGE
```

но primary V4.5 не должен заново перебирать их все.

---

# 18. Primary V4.5 systems

Для каждого dataset:

## S0 — historical original B0

Context/reference only.

## S1 — DiTy B0

Основной ranker baseline V4.5.

## S2 — DiTy + frozen-family V6-support branch

Показывает raw effect второго этапа.

## S3 — DiTy + adaptive fallback

Основной candidate pipeline.

## S4 — BGE B0

Strong reference.

---

# 19. Не называть S2 заранее «source-support V6», пока не восстановлена формула

Перед реализацией нужно восстановить historical stage2.

Создать:

```text
historical_stage2_contract.json
```

В нём:

```text
score formula
alpha
beta
gamma
normalization
support definition
tie-break
top-K semantics
reader dependencies
question dependencies
routing branch definitions
confidence feature
threshold calibration
Q5-veto role
availability/fallback logic
```

---

# 20. Ключевой принцип parameter freeze

Не переносить слепо один численный α/γ между SciMDIX и RuSERRC.

Primary единый метод должен фиксировать:

```text
FORMULA
ALLOWED PARAMETER GRID
SELECTION OBJECTIVE
TIE-BREAK RULE
CALIBRATION PROCEDURE
```

Параметры могут fold-locally выбираться на TRAIN/CAL.

Это соответствует concept:

```text
one method
different calibrated parameters
```

---

# 21. Frozen parameter family

Код-агент обязан восстановить exact historical RU-safe search space из V4.4.

Например, если historical grid содержал:

```text
alpha ∈ {...}
beta ∈ {...}
gamma ∈ {...}
```

не расширять его.

Для V4.5 primary:

```text
beta = 0
```

может быть зафиксирован только потому, что это заранее принятое post-V4.4 simplification decision.

В manifest:

```text
beta_fixed_from_v4_4_evidence = true
```

---

# 22. Если exact historical grid не восстанавливается

Нельзя создавать «разумный похожий grid» автоматически.

Статус:

```text
BLOCKED_PROTOCOL_RECOVERY
```

Сначала восстановить protocol из:

- config;
- tuning artifacts;
- README;
- source code;
- logs.

---

# 23. Parameter selection objective

Нужно восстановить exact V4.4 TRAIN/CAL objective.

Нельзя в новой реализации заменить его на другой metric для удобства.

Сохранить:

```text
selection_metric
secondary_tie_breaks
fallback_condition
```

Если historical objective был неоднозначен:

```text
STOP
```

до explicit resolution.

---

# 24. Fixed-transfer diagnostic

Дополнительно разрешено провести diagnostic:

```text
SciMDIX winning numerical parameters
→ RuSERRC without retuning
```

Назвать:

```text
FIXED_TRANSFER
```

Это НЕ primary system.

Цель:

> проверить переносимость конкретных параметров.

Не использовать этот diagnostic для выбора primary post-hoc.

---

# 25. Primary architecture vs historical ASEM-Q5

Нельзя автоматически утверждать:

```text
ASEM-Q5 = source-support + fallback B0
```

До regression reconstruction.

Нужно определить:

1. какая branch являлась V6;
2. какая branch являлась alternative reader branch;
3. действительно ли alternative branch была exactly B0;
4. какой signal управлял router;
5. использовался ли Q5-veto в runtime decision;
6. сколько QA forwards требовалось.

---

# 26. Q5 decision

V4.4 показал небольшой самостоятельный Q5 contribution.

Для V4.5 resource-oriented candidate заранее выбирается:

```text
NO Q5 question selection
```

и предпочтительно:

```text
NO Q5-veto
```

НО это означает новую упрощённую конфигурацию.

Поэтому она не может наследовать исторический ASEM-Q5 результат как будто это та же система.

---

# 27. Historical equivalence check Q5

На старых cached predictions сравнить:

```text
ASEM-Q5
vs
ASEM without Q5-veto
```

Сохранить:

```text
q5_ablation_replay.json
```

Поля:

```text
same_top1_count
changed_top1_count
F1_delta
Hit1_delta
HN1_delta
runtime_dependency_delta
```

Это descriptive replay.

---

# 28. Primary stage2 candidate

После V4.4 принят следующий development choice:

```text
S3 = ASEM-style adaptive correction WITHOUT question selection
```

Q5-veto:

- исключить из primary resource-oriented candidate, если реализация без него однозначна;
- historical ASEM-Q5 оставить reference;
- не добавлять Q5 обратно после EVAL.

Если без Q5 историческая механика не может быть корректно восстановлена:

```text
не упрощать самовольно
```

и остановить protocol freeze.

---

# 29. Runtime branch interpretation

Если regression подтверждает:

```text
alternative reader branch == B0
```

то в описании разрешено говорить:

```text
adaptive fallback to B0
```

Если не подтверждает:

использовать точное название branch.

---

# 30. Router feature

Даже если final branch equals B0, router может использовать:

- consensus confidence;
- semantic feature;
- Q5 consistency;
- другой reader-derived signal.

Нельзя называть router «source-support router», если его decision feature другой.

В отчёте отдельно различать:

```text
candidate scoring signal
routing signal
```

---

# 31. Router runtime leakage

Runtime router input не должен содержать:

- gold label;
- gold answer F1;
- evidence class;
- outer-EVAL truth.

CAL может использовать gold utility только для выбора параметров/threshold.

Runtime features должны быть computable на новом документе.

Создать:

```text
router_feature_provenance.json
```

---

# 32. Outer CV

Использовать exact historical 5-fold document-grouped split.

Не создавать новый outer split.

Для каждого fold:

```text
TRAIN
CAL
EVAL
```

---

# 33. Inner CAL

Если exact historical CAL split существует:

```text
reuse exactly
```

Если не существует:

- создать один deterministic document-grouped CAL split внутри TRAIN;
- freeze до model fitting;
- использовать одинаково для S1/S2/S3;
- сохранить hash.

Не создавать разные CAL для разных systems.

---

# 34. Seeds

Development seeds:

```text
42
43
44
```

Не выбирать лучший seed.

Если historical DiTy использовал другие seeds:

- historical result остаётся reference;
- V4.5 использует canonical project seeds;
- различие явно указать.

---

# 35. DiTy training

Переобучить DiTy OOF в V4.5-A рекомендуется для:

- чистого artifact provenance;
- unified seeds;
- reproducibility;
- resource logging.

Training recipe:

```text
exact historical DiTy recipe
```

Не делать новый hyperparameter search.

---

# 36. Training recipe manifest

Сохранить:

```text
optimizer
learning_rate
batch_size
gradient_accumulation
effective_batch_size
max_length
precision
max_epochs
early_stopping
warmup
sampling
class/pair weighting
question weighting
scheduler
```

---

# 37. Early stopping

Использовать только TRAIN/CAL.

Не использовать outer EVAL.

Selection criterion должен быть exact historical criterion.

Не менять его после первого fold.

---

# 38. B0 quality reference reproducibility

Новый S1 не обязан bitwise совпасть с historical DiTy при stochastic training.

Но должен пройти sanity:

```text
same score direction
same candidate universe
same label semantics
same training objective
no catastrophic regression
```

Если aggregate F1 отличается слишком сильно, сначала провести root-cause analysis.

Не задавать post-hoc tolerance после результата.

Рекомендуется до запуска определить допустимую техническую tolerance только для deterministic preprocessing invariants, а не для F1.

---

# 39. QA reader

Заморожен.

Использовать exact reader per dataset из V4.4.

Не менять:

- checkpoint;
- tokenizer;
- decoding;
- answer length;
- normalization;
- preprocessing.

---

# 40. QA dependency graph

До resource benchmark определить:

```text
какие QA forwards реально необходимы каждому system
```

Создать:

```text
runtime_dependency_graph.json
```

Для S1 B0:

```text
обычно QA(final top1)
```

Для S2/S3:

```text
QA(top-K)
```

только если реальные features этого требуют.

Нельзя выполнять ненужные QA forwards только потому, что так делал старый experiment.

---

# 41. QA cache reuse

Проверять exact:

```text
question hash
chunk hash
reader id
reader revision
tokenizer
decoding config
```

Новый cache:

```text
V4.5 directory only
```

Сохранить:

```text
qa_cache_audit.json
```

---

# 42. K freeze

До outer-EVAL:

```text
RuSERRC K = 5
SciMDIX K = 4
```

Эти значения development-selected из V4.4.

Записать:

```text
K_selection_status = development_selected
```

Не заявлять их prespecified independently.

---

# 43. Почему разные K допустимы

Единый метод:

```text
architecture + tuning contract
```

а не:

```text
one universal K
```

V4.4 уже показал dataset dependence.

---

# 44. Primary quality hypothesis

Новая наиболее содержательная development-проверка:

```text
H_stage_RuSERRC:
S3 > S1
```

на RuSERRC с DiTy.

SciMDIX:

```text
S3 > S1
```

является frozen reproduction/stability check, потому что похожий результат уже известен.

---

# 45. Cross-dataset criterion

Если stage2 positive на обоих:

Корректно:

> наблюдается согласованный положительный development signal на двух русскоязычных датасетах.

Не:

> независимо подтверждён эффект на двух датасетах.

---

# 46. Primary metric

```text
Token F1
```

Secondary answer metrics:

```text
Lemma F1
Character F1
Unicode ROUGE-L
Exact Match
```

---

# 47. Ranking/evidence metrics

Обязательно:

```text
MRR
Hit@1
Hit@K
HN@1
HN@K
F1 | selected positive
```

При наличии labels:

```text
Amb@1
GSRN@1
```

---

# 48. Safety margins

Если проект сохраняет historical V4.4 margins:

```text
Hit@1 >= B0 - 0.5 pp
HN@1 <= B0 + 1.0 pp
```

использовать их без изменения.

В contract записать:

```text
safety_margin_origin = inherited_from_v4_4
```

Если эти margins не были частью project-wide freeze:

- не изображать их внешне предзаданными;
- использовать как development safety convention;
- не подбирать новые margins по V4.5 EVAL.

---

# 49. Statistical unit

Rows разных seeds не являются независимыми.

Для paired bootstrap:

1. получить metric contribution per `(question_id, seed)`;
2. усреднить seeds внутри question;
3. resample `document_id` clusters.

---

# 50. Bootstrap

```text
iterations >= 5000
seed = 42
cluster = document_id
```

Сохранять:

```text
observed_mean_delta
bootstrap_mean_delta
CI_low
CI_high
```

Primary reported delta:

```text
observed paired mean
```

CI — uncertainty estimate.

---

# 51. Multiple datasets

V4.5-A не является formal confirmatory family.

Поэтому:

- не делать вид, что development CI решают проблему multiplicity;
- report both datasets transparently;
- не выбирать после результатов только удачный dataset.

---

# 52. Primary comparisons

RuSERRC:

```text
S3 vs S1
```

SciMDIX frozen reproduction:

```text
S3 vs S1
```

Secondary:

```text
S2 vs S1
S3 vs S2
S1 vs historical original B0
S3 vs BGE
FIXED_TRANSFER vs S1
```

---

# 53. Не объявлять S3 vs BGE superiority/non-inferiority без заранее заданного margin

Если formal quality margin к BGE не определён до run:

```text
S3 vs BGE = descriptive trade-off
```

Не придумывать «приемлемую потерю F1» после результатов.

---

# 54. Resource hypothesis

Не утверждать заранее, что S3 дешевле BGE.

Вопрос:

```text
Is S3 more resource-efficient end-to-end?
```

может получить ответ:

```text
yes / mixed / no
```

---

# 55. Primary resource dimensions

Inference / deployment — primary:

```text
total parameter count
model weights size
peak GPU VRAM
peak CPU RAM
ranker latency
full pipeline latency/question
throughput
QA forward count/question
ranker forward count/question
```

Training resources — secondary:

```text
GPU-hours
peak training VRAM
training wall time
```

---

# 56. Ranker-only benchmark

Сравнить DiTy и BGE на:

```text
одинаковых question/chunk pairs
одинаковой sequence-length distribution
одинаковом precision mode
одинаковом device
```

Показать:

```text
latency
throughput
peak VRAM
RAM
```

---

# 57. End-to-end benchmark

Сравнить реальные deployable pipelines.

## S3

Считать все обязательные runtime steps:

```text
DiTy all-candidate scoring
top-K
required QA forwards
stage2 feature computation
router
final selection
final answer
```

## BGE reference

Считать:

```text
BGE all-candidate scoring
top1
QA needed for final extracted answer
final answer
```

Если BGE pipeline в реальности требует больше шагов, использовать exact production-equivalent path.

---

# 58. Не выравнивать искусственно число QA forwards

Resource benchmark должен отражать реальный алгоритм.

Если:

```text
S3 requires K QA forwards
BGE requires 1
```

так и измерять.

Не запускать BGE на K QA только «для симметрии».

---

# 59. Benchmark environment

Зафиксировать:

```text
GPU model
driver
CUDA
PyTorch
Transformers
Python
precision
tokenizer
batching policy
sequence lengths
CPU
RAM
```

---

# 60. Benchmark repetitions

Для каждого mode:

1. warmup >= 10;
2. measurements >= 10, если стоимость позволяет;
3. `torch.cuda.synchronize()` вокруг GPU timing;
4. reset peak memory;
5. primary latency = median;
6. report mean/std/p95 secondary;
7. raw observations сохранить.

---

# 61. Batch policy

Нужны два benchmark режима.

## A. matched operational batch

Одинаковый practically relevant batch policy для сравнения.

## B. best-feasible throughput

Каждой модели разрешён наиболее эффективный batch при фиксированном GPU memory budget.

Не смешивать показатели двух режимов.

---

# 62. Resource conclusion

Основной анализ:

```text
dimension-wise deltas
+
Pareto frontier
```

Не использовать один искусственный scalar.

Предпочтительно report:

```text
VRAM reduction %
latency delta %
parameter reduction %
F1 delta
```

---

# 63. F1/GB и F1/sec

Если рассчитывать:

```text
appendix only
```

Не использовать как основной критерий.

Такие ratios могут быть вводящими в заблуждение.

---

# 64. Pareto rule

Если S3 хуже BGE одновременно по:

```text
Token F1
VRAM
latency
```

то S3 Pareto-dominated и не может называться ресурсно-предпочтительной системой.

Если trade-off смешанный:

> представить как альтернативную рабочую точку, а не победителя.

---

# 65. Switch analysis

S3 vs S1:

```text
same_top1
good_switch
bad_switch
neutral_switch
```

Где:

```text
good: F1 increases
bad: F1 decreases
neutral: F1 equal but chunk changes
```

---

# 66. Evidence transition

Для changed top1:

```text
positive → positive
positive → non-positive
non-positive → positive
non-positive → non-positive
```

Отдельно считать net positive-row change.

---

# 67. Router analysis

Считать:

```text
availability
router branch use rate
actual top1 change rate
fallback rate
good switches
bad switches
neutral switches
useful-switch precision
useful-switch recall
```

---

# 68. Router opportunity definition

`missed opportunity` — только post-hoc diagnostic с gold.

Нельзя использовать его runtime.

Сохранять отдельно от deployable features.

---

# 69. Per-seed stability

Для 42/43/44:

```text
S1 F1
S2 F1
S3 F1
ΔS3-S1
Hit@1
HN@1
```

---

# 70. Per-fold stability

Для каждого outer fold:

```text
ΔF1
ΔHit1
ΔHN1
```

Считать:

```text
positive fold count
negative fold count
range
fraction of total net gain by fold
```

Особенно важно для малой SciMDIX.

---

# 71. Aspect analysis

Только descriptive.

Не вводить:

```text
aspect-specific thresholds
aspect-specific K
```

в V4.5.

Для каждого аспекта:

```text
N
F1
ΔF1
Hit1
HN1
```

Low-N помечать.

---

# 72. Residual-error analysis

Для каждого B0:

```text
Token F1
Hit@1
HN@1
MRR
```

и stage2:

```text
ΔToken F1
ΔHit@1
ΔHN@1
```

---

# 73. B0-strength descriptive table

Использовать historical:

```text
old B0
ModernBERT
DiTy
BGE
```

только если:

- candidate universe comparable;
- metric definitions identical.

Иначе таблицу разбить.

---

# 74. Не строить строгую регрессию на нескольких rankers

Допустимый вывод:

> результаты согласуются с residual-correction hypothesis.

Недопустимый:

> доказана монотонная математическая зависимость между силой B0 и gain stage2.

---

# 75. Historical serialization defect

Не использовать buggy saved protocol scores как источник новых ranking metrics.

В V4.5:

- recompute metrics из authoritative selected chunk/prediction data;
- strict typed schemas;
- finite validation;
- `NaN` → explicit null/missing reason.

---

# 76. Regression tests

До full run обязательны:

1. historical files hash before;
2. exact datasets;
3. exact document folds;
4. no document leakage;
5. candidate universe identity;
6. chunking identity;
7. q0 identity;
8. DiTy base model exact id;
9. DiTy recipe recovery;
10. stage2 formula recovery;
11. parameter-grid recovery;
12. selection-objective recovery;
13. router-feature provenance;
14. Q5 replay;
15. QA model identity;
16. QA cache audit;
17. no old write;
18. serialization strictness;
19. all runtime scores finite;
20. Unicode RU metrics verified.

---

# 77. Smoke run

Только:

```text
1 dataset
1 fold
1 seed
```

Проверить:

```text
training
ranking
QA cache
stage2
router
metrics
resource instrumentation
artifact paths
```

Smoke-result:

```text
NOT SCIENTIFIC RESULT
```

---

# 78. Frozen contract before outer-EVAL inspection

Создать:

```text
contract_frozen.json
```

Включить:

```text
dataset hashes
fold hashes
DiTy model id/revision
training recipe
q0 id/text hash
K per dataset
beta setting
parameter grid
parameter selection objective
tie-break
stage2 formula
router feature
router calibration rule
Q5 policy
QA model
metrics
safety convention
bootstrap
resource benchmark
seed policy
```

Hash contract.

---

# 79. Важное различие: freeze procedure vs freeze numbers

До outer-EVAL обязательно фиксируются:

```text
procedure
grid
selection objective
tie-break
```

Не обязательно фиксировать один α/γ для всех folds.

Fold-local TRAIN/CAL may choose values.

Это не post-hoc, если selection procedure была frozen заранее.

---

# 80. Artifact structure

```text
artifacts/qa_chunk_selection_v4_5_ru/
  run_<timestamp>/

    status.json
    COMPLETED.json
    contract_frozen.json
    config_resolved.json

    inputs/
      ruserrc_manifest.json
      scimdix_manifest.json
      historical_reference_manifest.json
      source_hashes.json
      test_provenance_audit.json

    protocol_recovery/
      historical_stage2_contract.json
      parameter_grid.json
      selection_rule.json
      q5_ablation_replay.json
      router_feature_provenance.json

    regression/
      hash_before.json
      hash_after.json
      split_check.json
      candidate_universe_check.json
      chunking_check.json
      qa_cache_audit.json
      serialization_check.json

    ruserrc/
      fold0/
        seed42/
          dity_b0/
          v6/
          adaptive/
        ...
      oof/

    scimdix/
      fold0/
        ...
      oof/

    resource_benchmark/
      environment.json
      ranker_only/
      end_to_end/
      raw/

    analysis/
      primary_metrics.json
      paired_bootstrap.json
      seed_stability.json
      fold_stability.json
      aspect_metrics.json
      switch_analysis.json
      evidence_transitions.json
      router_analysis.json
      residual_error_analysis.json
      quality_resource_tradeoff.json

    reports/
      results_v4_5_ru.md
```

---

# 81. Completion criteria V4.5-A

```text
[ ] historical artifacts unchanged
[ ] exact datasets resolved
[ ] exact folds resolved
[ ] candidate universe validated
[ ] exact DiTy resolved
[ ] training recipe recovered
[ ] exact historical stage2 reconstructed
[ ] parameter grid frozen
[ ] selection objective frozen
[ ] q0 frozen
[ ] K frozen
[ ] Q5 policy frozen
[ ] router provenance valid
[ ] QA frozen
[ ] QA cache audited
[ ] no outer-EVAL tuning
[ ] all folds complete
[ ] seeds 42/43/44 complete
[ ] bootstrap complete
[ ] resource benchmark complete
[ ] report complete
```

---

# 82. Interpretation after V4.5-A

## Case A — positive on RuSERRC and SciMDIX

Корректно:

> согласованный положительный development signal на двух RU datasets.

Не:

> окончательно подтверждён универсальный RU effect.

## Case B — positive SciMDIX, null RuSERRC

Вывод:

> stage2 utility зависит от residual error structure.

Не retune.

## Case C — negative RuSERRC

Сохранить.

Не ослаблять gate.

## Case D — F1 positive, evidence safety worse

Не принимать как final stage2 under existing safety convention.

## Case E — resource hypothesis fails

Если end-to-end S3 не даёт meaningful resource advantage:

> нельзя использовать ресурсную мотивацию DiTy как подтверждённый результат.

Остаётся только quality/component analysis.

---

# 83. Final candidate selection

После V4.5-A выбирать по заранее объявленной иерархии:

1. runtime validity;
2. evidence safety;
3. stage2 stability;
4. resource trade-off;
5. Token F1;
6. simplicity.

Не выбирать ячейку только по максимальному F1.

---

# 84. V4.5-B freeze

Перед independent test:

```text
final_freeze_manifest.json
```

Содержит:

```text
ranker
training recipe
final parameter-selection/deployment rule
K
question
QA reader
stage2
router
seed policy
runtime preprocessing
metrics
```

Hash file.

---

# 85. Как получить deployable global stage2 parameters

Outer-fold-specific параметры нельзя просто использовать на test.

До V4.5-B заранее определить deployment procedure.

Предпочтительный вариант:

1. собрать только cross-fitted OOF development predictions;
2. применить ту же frozen parameter family;
3. выбрать **один global deployable parameter set** на development OOF;
4. выбрать один global router calibration rule/threshold на development;
5. freeze;
6. затем обучить final B0;
7. test не использовать.

Если historical architecture требует score calibration, убедиться, что параметры переносимы на final model.

Если нет — использовать заранее выделенный development CAL split для final calibration.

Не выбирать способ после test.

---

# 86. Final B0 training

После parameter freeze:

```text
train final DiTy on all allowed development training data
```

Seed:

```text
canonical 42
```

если это project-wide policy.

Не выбирать лучший seed по development F1 в последний момент.

---

# 87. Score-distribution calibration caution

Если router threshold зависит от абсолютных B0 scores:

- OOF threshold может плохо переноситься на final refit;
- нужно использовать normalized/query-relative feature либо dedicated development CAL.

Не игнорировать этот вопрос.

Сохранить calibration diagnostics до test.

---

# 88. Independent test primary systems

На V4.5-B:

```text
Final DiTy B0
Final DiTy + frozen stage2
```

BGE:

```text
optional prespecified strong reference
```

если его final recipe также frozen.

Не открывать исторические alternatives на test.

---

# 89. После test запрещено

- менять K;
- менять α/β/γ;
- менять router;
- менять q0;
- возвращать Q5;
- выбирать другой seed;
- добавлять SEM/LEX;
- менять QA;
- менять preprocessing.

---

# 90. Если independent test отсутствует

Итоговый статус:

```text
development-supported method
```

Не:

```text
independently confirmed method
```

В диссертации:

- раскрыть OOF protocol;
- раскрыть reused-development limitation;
- показать внешние/междатасетные результаты;
- не завышать доказательную силу.

---

# 91. Что делать с V5

V5 reader-aware ranking:

```text
DEFERRED
```

Возвращаться только если:

1. V4.5 stage2 недостаточно устойчив;
2. требуется новый learned mechanism;
3. доступен новый validation/test budget.

Не запускать V5 параллельно с V4.5 и не использовать его для спасения V4.5 post-hoc.

---

# 92. Scientific narrative при успешном V4.5

Корректная формулировка:

> На основе результатов V4.4 была сформирована ресурсно-ориентированная конфигурация метода с русскоязычным ранжировщиком меньшего масштаба и адаптивным этапом коррекции top-K кандидатов. Второй этап рассматривается как механизм коррекции остаточных ошибок и применяется только при наличии дополнительного полезного сигнала. Эффективность оценивается одновременно по качеству ответа, корректности выбора evidence и вычислительным затратам.

---

# 93. Что нельзя писать

Нельзя:

> «DiTy специально слабее, поэтому позволяет показать эффективность второго компонента».

Нельзя:

> «BGE хуже подходит, потому что stage2 с ним ничего не улучшает».

Нельзя:

> «V4.5 независимо подтвердил SciMDIX», если использованы те же данные.

Нельзя:

> «DiTy + stage2 ресурсно эффективнее», пока нет end-to-end measurements.

Нельзя:

> «доказана универсальность stage2 для русского языка» без новых данных.

---

# 94. Короткая инструкция агенту

```text
Implement V4.5 as a NEW isolated RU development experiment.

Never modify V3/V4.x/V4.4 artifacts.

Datasets:
- RuSERRC
- SciMDIX

Use exact historical document-grouped outer folds.

Primary candidate ranker:
- exact DiTy checkpoint/revision recovered from historical artifacts

Strong tested reference:
- BAAI/bge-reranker-v2-m3

Do not assume DiTy is resource-efficient before benchmarking.

Primary systems:
S1 = DiTy B0
S2 = DiTy + frozen-family V6 branch
S3 = DiTy + adaptive correction/fallback
S4 = BGE B0 reference

Before coding S2/S3:
reconstruct the exact historical stage2 contract:
formula, alpha/beta/gamma, support definition,
router feature, branches, threshold rule, Q5 role,
runtime QA dependencies.

Do not blindly transfer one numerical alpha/gamma across datasets.
Freeze the allowed parameter grid and TRAIN/CAL selection procedure.

Primary RU simplification:
beta = 0.
Use q0.
No Q5 question search.
Q5-veto removal is allowed only as an explicitly frozen resource-oriented
development choice after historical replay; do not pretend it is identical
to historical ASEM-Q5 unless top1 equivalence is proven.

Freeze K:
RuSERRC = 5
SciMDIX = 4

TRAIN/CAL only for:
parameter selection
router calibration
early stopping

Outer EVAL only for frozen evaluation.

QA reader remains frozen.

Primary new development question:
Does S3 improve over S1 on RuSERRC?

SciMDIX is a frozen reproduction/stability check, not new confirmation.

Report:
Token F1
Lemma F1
Char F1
Unicode ROUGE-L
EM
MRR
Hit@1/Hit@K
HN@1/HN@K
F1|positive
switches
evidence transitions
router diagnostics

Bootstrap:
average seeds within question,
cluster by document_id,
>=5000 iterations.

Benchmark on the same machine:
DiTy ranker-only vs BGE ranker-only
DiTy+stage2 end-to-end vs BGE B0+final QA end-to-end.

Do not artificially equalize QA forward counts.
Measure the real runtime pipeline.

V4.5-A = development characterization only.

Before V4.5-B:
audit that the test is genuinely untouched,
freeze one deployable global parameter/router configuration,
train final B0,
hash the freeze manifest.

No post-hoc changes after test.
```

---

# 95. Final pre-run checklist

```text
[ ] old artifacts hashed and read-only
[ ] RuSERRC exact source resolved
[ ] SciMDIX exact source resolved
[ ] frozen folds verified
[ ] candidate universes verified
[ ] chunking verified
[ ] q0 verified
[ ] DiTy exact model/revision resolved
[ ] DiTy historical training recipe recovered
[ ] historical V4.4 stage2 contract reconstructed
[ ] actual router feature identified
[ ] actual branch semantics identified
[ ] parameter grid recovered
[ ] selection objective recovered
[ ] beta policy frozen
[ ] K frozen
[ ] Q5 replay completed
[ ] Q5 policy frozen
[ ] QA model frozen
[ ] QA cache audited
[ ] router runtime features proven gold-free
[ ] outer-EVAL tuning impossible by design
[ ] bootstrap frozen
[ ] benchmark environment frozen
[ ] smoke run passed
[ ] test provenance audit created
```

Full run only after all critical items pass.

---

# 96. Итоговая цель V4.5

V4.5 должен дать честный ответ на три разных вопроса, не смешивая их:

### A. Component quality
Повышает ли adaptive second stage качество относительно DiTy B0?

### B. Evidence safety
Не достигается ли этот прирост ценой ухудшения выбора корректного evidence?

### C. Resource trade-off
Действительно ли полный DiTy + stage2 pipeline является более выгодной рабочей точкой по ресурсам, чем сильный BGE reference?

Только совместный ответ на A+B+C позволяет использовать ресурсную мотивацию в диссертации.

Если C не подтверждается, V4.5 всё равно остаётся полезным для анализа компонентов, но тезис о снижении ресурсоёмкости именно относительно BGE должен быть снят или ограничен.
