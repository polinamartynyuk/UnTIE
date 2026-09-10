# Протокол продолжения V3.1 с QA-based silver target-status overlay

## 0. Назначение документа

Этот протокол продолжает текущий V3 после автоматической подготовки target-status данных **той же frozen extractive QA-моделью**, которая уже используется в основном pipeline.

Отдельная генеративная teacher-модель, NLI-модель или semantic encoder для подготовки данных не используются.

Обязательный вход:

```text
annotation_qa_silver/
    qa_model_manifest.json
    aspect_questions.json
    attribution_questions.json
    stage_a_all_ambiguous.jsonl
    stage_b_paraphrases.jsonl
    stage_b_ownership.jsonl
    discourse_features.jsonl
    qc_summary.json
    silver_manifest.json
```

Сырые frozen-QA outputs и детерминированные признаки могут быть вычислены один раз и закэшированы для всего unlocked пула.

Но итоговые OOF silver labels являются **fold-specific**, потому что:

```text
TRAIN:
    score normalization / train-derived statistics

CALIBRATION:
    Stage-A threshold
    paraphrase-consistency threshold
    ownership margin
    confidence thresholds

EVAL:
    только применение frozen правил текущего outer run
```

Поэтому для каждого outer run сохранять отдельный overlay, например:

```text
annotation_qa_silver/oof/run0/silver_target_status.jsonl
...
annotation_qa_silver/oof/run4/silver_target_status.jsonl
```

Терминология:

```text
P_silver_target
TM_silver
SilverTargetMismatch
SilverPairwiseTargetVsMismatch
```

Не использовать термин `verified TM`: это автоматическая weak-supervision разметка, а не независимый human gold.

---

# 1. Что уже известно из текущего V3

Reference V3 показал:

```text
B0 Token F1 ≈ 27.36
Hit@1 ≈ 36.7%
Hit@5 ≈ 79%
Hit@10 ≈ 92%
HN@1 ≈ 4.9%
Ambiguous@1 ≈ 50.9%
GSRN@1 ≈ 33.9%
reader best-positive oracle ≈ 73
```

Следовательно:

> primary bottleneck — перестановка / attribution уже найденных plausible candidates, а не полный поиск evidence с нуля.

---

# 2. Что меняется из-за автоматической разметки

Новый QA-based silver overlay не является независимым ground truth.

Поэтому:

### Primary model-selection target

```text
document–aspect Token F1 по original SciREX gold
```

### Hard safeguards

```text
HN@1
pairwise_clean_hn
```

### Silver diagnostics

```text
SilverTargetMismatch@1
SilverPairwiseTargetVsMismatch
SilverTargetEvidenceHit@K
SilverWrongTargetAnswerRate
```

Silver metrics нельзя использовать как единственное доказательство улучшения.

---

## 2.1. Ограничение вычислительной схемы

На этапе подготовки silver labels запрещено подключать:

```text
генеративную LLM
NLI-модель
semantic encoder
отдельный learned annotator
```

Разрешены только:

```text
текущая frozen extractive QA
+
её spans/logits
+
заранее фиксированные QA-вопросы
+
детерминированные признаки текста/структуры
+
TRAIN/CAL thresholds.
```

Новый trainable target-attribution model появляется только в самом эксперименте (`B1-Silver`, `R1-Silver`), а не внутри подготовки данных.

---

# 3. Preconditions

До запуска downstream experiments:

- [ ] frozen QA regression check пройден;
- [ ] QA checkpoint/tokenizer hash сохранён;
- [ ] canonical/paraphrase/ownership question sets frozen;
- [ ] Stage-A и Stage-B raw outputs закэшированы;
- [ ] discourse-feature rules frozen;
- [ ] silver mapping rules frozen;
- [ ] automatic QC gates пройдены;
- [ ] fold-specific TRAIN/CAL thresholds сформированы без EVAL;
- [ ] TM_silver > 0 хотя бы в достаточном числе document–aspect по заранее заданному минимуму;
- [ ] per-fold silver counts достаточны по заранее заданным минимумам;
- [ ] high-confidence labels отделены от medium/low/unresolved;
- [ ] evidence clusters/dedup готовы;
- [ ] INTERNAL_TEST закрыт.

---

# 4. CV остаётся замороженным

Использовать текущие 5 document-level folds.

В каждом outer run:

```text
3 folds TRAIN
1 fold CALIBRATION
1 fold EVAL
```

Сырые outputs frozen QA не меняются между runs и могут быть общим cache.

Но silver mapping выполняется отдельно в каждом outer run:

```text
TRAIN → train-derived normalization/statistics
CAL   → thresholds
EVAL  → frozen application
```

EVAL silver labels — только диагностический weak-supervision target, полученный тем же frozen QA-based pipeline; это не независимый gold.

---

# 5. Новый reference B0

Для V3.1 заранее использовать:

```text
B0-max8
same MiniLM backbone
question-balanced weighted BCE
max_epochs = 8
patience = 2
CAL-only early stopping
seeds = 42,43,44
```

Старый B0-max5 показывать только исторически.

Нельзя выбирать между max5/max8 по OOF EVAL.

---

# 6. Этап S0 — анализ silver overlay до обучения

Построить:

```text
silver status × aspect
silver status × source ambiguous subtype
silver status × QA ownership-probe outcome
silver status × confidence
silver status × evidence cluster
silver status × original B0 rank
Stage-A pass-rate × aspect/subtype
Stage-B attribution-call rate × aspect/subtype
```

Отдельно:

```text
B0 top-1/top-5/top-10:
сколько high-confidence P_silver_target;
сколько TM_silver;
сколько unresolved;
сколько вопросов имеют одновременно target и TM.
```

Цель:

> проверить, что автоматическая target-attribution задача реально присутствует внутри B0 shortlist.

---

## 6.1. Откуда берутся `P_silver_target` и `TM_silver`

Они формируются не отдельной моделью-аннотатором, а двухступенчатой QA-based процедурой:

```text
Stage A:
canonical frozen QA
→ mining QA-plausible ambiguous

Stage B:
та же frozen QA
→ paraphrase probes
→ target/external ownership probes
+
детерминированные discourse/context признаки
→ consensus / abstention
```

Primary strong labels:

```text
P_silver_target =
P_original
∪ high-confidence silver_target_evidence

TM_silver =
high-confidence silver_target_mismatch
```

Не включать в primary supervision:

```text
medium confidence
silver_partial_support
silver_unresolved
new_valid_value_candidate
```

Для каждого OOF-run использовать только fold-specific labels, полученные его TRAIN/CAL thresholds.

---

# 7. Этап S1 — B1-Silver: минимальное изменение supervision

Reference:

```text
B0:
P_original
vs
training_eligible HN + limited easy
```

Candidate:

```text
B1-Silver:
positive = P_silver_target
negative = HN + TM_silver + limited easy
```

Не использовать в primary:

```text
medium-confidence target/TM
silver_partial_support
silver_unresolved
new_value_candidate
```

Question-balanced weighting сохраняется.

Evidence-cluster duplicates не должны многократно увеличивать вес одного source region.

Для outer run `r`:

```text
B1-Silver TRAIN labels = silver labels, построенные thresholds run r
B1-Silver CAL labels   = те же frozen rules run r
B1-Silver EVAL labels  = те же frozen rules run r
```

Нельзя использовать silver labels, построенные с thresholds другого outer run, если эти thresholds зависели от CAL другого fold.

---

# 8. B1-Silver metrics

### Primary

```text
Token F1
EM
```

### Original-gold ranking

```text
Hit@1_original
Hit@5_original
Hit@10_original
MRR
```

### HN safeguards

```text
HN@1
pairwise_clean_hn
```

### Silver diagnostics

```text
SilverTargetStatusCoverage@1
SilverTargetStatusSlotCoverage@5/@10
SilverTargetMismatch@1_lower_bound
SilverTargetMismatchRate@1_known_silver
SilverPairwiseTargetVsMismatch
SilverTargetEvidenceHit@K
SilverWrongTargetAnswerRate
SilverAbstentionRate
```

Все silver metrics сопровождать:

```text
n labels
n unique document–aspect
confidence composition
QA model hash + question-set version + silver-rule version
```

---

# 9. B1-Silver model-selection rule

Hard safeguards относительно B0:

```text
HN@1_candidate <= HN@1_B0 + epsilon_HN
pairwise_clean_candidate >= pairwise_clean_B0 - epsilon_pairwise
```

Значения epsilon фиксируются до EVAL.

Primary criterion:

```text
Token F1
```

Silver TM improvement считается **поддерживающим**, но не достаточным условием.

Пример допустимого вывода:

> B1-Silver улучшает downstream Token F1 и одновременно сильнее согласуется с frozen QA-based silver target-attribution procedure.

Недопустимый вывод:

> B1-Silver доказанно уменьшает истинные target-mismatch ошибки.

---

# 10. Paired statistics B1-Silver vs B0

Использовать:

```text
clustered paired bootstrap by document_id
>= 5000 replications
```

Для trainable 3-seed systems:

```text
сначала получить question-level mean outcome across seeds
затем bootstrap document clusters
```

Per-seed результаты показывать как stability analysis.

Обязательные deltas:

```text
ΔTokenF1
ΔEM
ΔHit@1
ΔHN@1
ΔSilverTargetMismatch
ΔSilverPairwiseTargetVsMismatch
```

---

# 11. Gate G1

## G1-A — B1-Silver успешен

Продолжать target-attribution branch, если:

```text
проходит HN safeguards;
Token F1 не ухудшается практически значимо;
silver TM diagnostics улучшаются;
эффект не ограничен одним fold.
```

## G1-B — B1 не улучшает F1, но silver TM реально конкурирует внутри shortlist

Разрешить один специализированный second-stage text reranker, если:

```text
в B0 top-K достаточно вопросов с одновременно P_silver_target и TM_silver;
QA-based silver QC высокий;
TM_silver не редкий единичный класс.
```

## G1-STOP

Остановить ветку, если:

```text
QA-based silver QC недостаточен;
TM_silver слишком мало;
B0 shortlist почти не содержит target-vs-TM конкуренции.
```

---

# 12. R1-Silver — специализированный target-attribution reranker

Pipeline:

```text
all chunks
↓
B0
↓
top-K
↓
R1-Silver target attribution
↓
top-1
↓
frozen QA
```

Проверить:

```text
K = 5
K = 10
```

K выбирать только CALIBRATION.

---

# 13. Cross-fitted shortlist — обязательно

Для TRAIN second-stage model нельзя использовать in-sample B0 shortlist.

В каждом outer run:

```text
TRAIN_FOLDS
↓
inner grouped cross-fitting B0
↓
out-of-inner-fold B0 scores/shortlists
↓
R1 training candidates
```

CAL/EVAL shortlist:

```text
B0 trained only on outer TRAIN_FOLDS
```

Сохранять:

```text
shortlist_source_model_id
crossfit_fold_id
B0_score
B0_rank
```

---

# 14. R1-a — text-only attribution

Input:

```text
[ASPECT]
...
[QUESTION]
...
[CHUNK]
...
```

Training target:

```text
P_silver_target
vs
TM_silver
```

Silver labels здесь происходят из текущей frozen QA-based mining/attribution процедуры, а сам R1 остаётся отдельным trainable MiniLM/Cross-Encoder этапом.

Контрольные HN/easy добавлять ограниченно.

Unresolved/partial/medium-confidence не использовать как strong negative.

---

# 15. R1-b — local discourse attribution

Добавить:

```text
section title
previous sentence
candidate chunk
next sentence
```

При `max_length=512` candidate chunk имеет приоритет.

Frozen truncation policy:

```text
1. сохранить candidate chunk;
2. остальной budget → section/prev/next;
3. логировать truncation rate;
4. не позволять context addition обрезать candidate evidence сильнее baseline.
```

---

# 16. R1-c — explicit attribution markers

Автоматически извлекаемые признаки:

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

Они являются features, а не rule-based final classifier.

Feature family фиксируется до EVAL.

---

# 17. Pairwise Silver ranking

Только после подтверждённого target signal в B1 или R1.

Pairs:

```text
P_silver_target > TM_silver
P_silver_target > training_eligible HN
```

Loss:

\[
L=-\log\sigma(s^+-s^-)
\]

Не заставлять valid positives конкурировать друг с другом.

Overlapping clusters контролировать по весу.

---

# 18. Listwise Silver ranking внутри shortlist

Не повторять full-document B3.

Candidate set:

```text
B0 top-5 / top-10
```

Loss:

\[
L=-\log
\frac{
\sum_{c\in P_{silver}}\exp(s_c/T)
}{
\sum_{c'\in C_{shortlist,labeled}}\exp(s_{c'}/T)
}
\]

`silver_unresolved` и `partial_support` маскировать из supervised denominator.

T выбирать только CALIBRATION.

---

# 19. Attention: только specialist signal

Текущий V3 показал, что attention не работает как хороший общий ranker.

Поэтому не повторять:

```text
attention-only full-document ranking
0.5 B0 + 0.5 attention
```

Новая задача:

```text
P_silver_target
vs
TM_silver
```

---

# 20. Честный attention audit

Head/feature analysis:

```text
TRAIN only
```

N_heads/thresholds:

```text
CAL only
```

EVAL:

```text
только frozen diagnostic evaluation
```

Основная задача:

```text
silver target vs silver mismatch
```

Контроли:

```text
silver target vs HN
reader-success vs reader-failure
```

---

# 21. Attention features

Использовать inference-compatible compact features:

```text
normalized Q→predicted-span attention
normalized span→Q
normalized entropy
top-k concentration
competing-region margin
head agreement
layer dynamics
head dispersion
```

Control features:

```text
question length
context length
predicted span length
span position
B0 rank
raw QA span score
```

Сравнить:

```text
CTRL
ATT
CTRL + ATT
```

---

# 22. Reader-aware nature и стоимость attention

Attention features требуют frozen QA forward по shortlist.

Pipeline:

```text
B0 top-K
↓
frozen QA + output_attentions
↓
predicted spans + attention signatures
↓
specialist TM reranker
```

Это не дешёвый pre-reader filter.

QA outputs можно кэшировать и переиспользовать для финального ответа.

Обязательно измерять:

```text
latency
GPU memory
reader calls
reader tokens
attention cache size
```

---

# 23. Head selection

TRAIN-only, желательно inner grouped resampling.

Сравнить:

```text
all-head mean
last-layer mean
middle-layer mean
selected heads
```

N_heads grid:

```text
1,2,4,8,16
```

если архитектура позволяет.

Не переносить retrieval-head conclusions decoder LLM на RoBERTa без собственной проверки.

---

# 24. Internal reader-signal controls

Если attention показывает TRAIN/CAL signal, сравнить:

```text
attention only
hidden-state only
attention + hidden-state
```

Hidden features:

```text
pooled question hidden state
pooled predicted-span state
Q/span cosine
|Q-S|
Q*S
```

Optional diagnostics:

```text
attention × value norm
gradient × attention
```

Gradient branch — diagnostic only из-за backward cost.

---

# 25. Attention distillation

Только если direct attention полезен, но дорог.

```text
frozen reader attention-derived score
↓
lightweight question+chunk student
```

Distillation targets только TRAIN/cross-fit candidates.

CAL выбирает distillation hyperparameters.

EVAL/test не создают supervision.

---

# 26. Causal head masking

Только если selected heads устойчивы по folds/seeds.

Сравнить:

```text
normal reader
mask selected heads
mask matched random heads
```

Это diagnostic causal validation, а не production method.

---

# 27. Method-specific attention audit

Exploratory V3 показал необычное поведение `method`.

Провести TRAIN/CAL transition analysis:

```text
B0 top1 → attention top1
```

Категории:

```text
GSRN → target
ambiguous → target
target → HN
GSRN → HN
target → target
```

Использовать silver status для автоматического анализа переходов.

Не изобретать новые feature families по EVAL transitions.

---

# 28. SpanRanker пока остановлен

Текущий K=3 SpanRanker ухудшил QA.

Не тюнить его до улучшения evidence shortlist.

Возврат разрешён только после вычисления:

```text
raw candidate-span oracle
attribution-aware candidate-span oracle
```

Attribution-aware correctness для silver режима:

```text
span_correct_silver_target = 1
только если source ∈ P_silver_target
и span соответствует original gold.
```

Если source = TM_silver:

```text
wrong_target_silver = 1
```

---

# 29. D4 — сначала oracle diagnostic

Gate уже открыт текущими positive-quality результатами, но обучение selector не начинать сразу.

На TRAIN/CAL original positives сравнить:

```text
reader(full chunk)
reader(gold-containing sentence)
reader(minimal gold-support window)
reader(sentence ±1)
```

Для spans через границу предложений использовать minimal support window.

Gold только diagnostic, не runtime.

---

# 30. D4 gate

Развивать runtime D4 только если oracle даёт устойчивый practically meaningful gain.

Особенно анализировать:

```text
material
metric
reader-hard
reader-failed
```

---

# 31. D4 runtime

При прохождении gate:

```text
candidate chunk
↓
sentence/local-window selector
↓
frozen reader
```

Обязательные требования:

```text
source offsets preserved
full-chunk fallback
cross-sentence support handled
TRAIN-only selector supervision
```

---

# 32. D5 — independent evidence audit

До aggregation:

```text
merge overlapping positive source intervals
↓
independent evidence clusters
```

Проверить, сколько document–aspect имеют ≥2 независимых clusters.

Не считать overlapping chunks независимой поддержкой.

---

# 33. D5 gate

Развивать aggregation только если независимое multi-evidence реально распространено после de-dup.

---

# 34. D5 aggregation

Первая версия:

```text
candidate answers
↓
exact normalization
↓
group exact-normalized answer
↓
deduplicate overlapping evidence
↓
aggregate calibrated scores
```

Не использовать semantic answer clustering в первой версии.

Не использовать raw cross-passage start/end logits как probability.

---

# 35. D3 остаётся deferred

Запускать только если error analysis показывает необходимость распределённого межчанкового evidence, которое D4 не решает.

---

# 36. Aspect-specific analysis

Обязательно:

```text
task
method
material
metric
```

Рабочая гипотеза:

```text
method   → ranking / attribution bottleneck
material → reader/context bottleneck заметнее
metric   → оба bottleneck
task     → преимущественно reranking
```

Это diagnostics, не четыре заранее разные production pipelines.

---

# 37. Seeds и честное сравнение

Trainable text models:

```text
seeds 42/43/44
```

Seeds не независимые наблюдения.

Primary statistics:

```text
1. question-level average across seeds
2. cluster bootstrap by document_id
```

Single-seed/deterministic branch сравнивать с matching B0 seed/universe.

Не сравнивать n=453 branch с n=1359 pooled строкой напрямую.

---

# 38. Статистика

Обязательные paired comparisons:

```text
B1-Silver vs B0
R1-Silver vs B0
pairwise/listwise vs best BCE/text target reranker
text+attention vs text-only
D4 vs pre-D4
D5 vs pre-D5
```

Clustered paired bootstrap:

```text
>= 5000
cluster = document_id
```

Выводить:

```text
mean delta
95% CI
win/lose/tie
```

OOF = model-selection evidence, не independent confirmatory test.

---

# 39. Practical thresholds и multiplicity

До финального OOF comparison зафиксировать:

```text
delta_F1_min
epsilon_HN
runtime_ratio_max
memory_ratio_max
```

Silver metrics могут иметь diagnostic thresholds, но не должны становиться единственным primary success criterion.

Для formal secondary hypothesis tests использовать заранее выбранную correction procedure, например Holm; иначе показывать effect size + CI.

---

# 40. Правило использования EVAL

После просмотра EVAL в той же protocol version запрещено:

```text
добавлять новую feature family
менять canonical/paraphrase/ownership question sets
менять Stage-A/Stage-B thresholds или confidence rule
менять taxonomy
менять shortlist K
менять loss
менять attention heads
```

Если EVAL error analysis порождает новую идею:

```text
новая frozen protocol version
```

На тех же folds это всё равно model-selection evidence, а не независимое подтверждение.

---

# 41. Error taxonomy V3.1-Silver

Минимум:

```text
E1  no original positive in shortlist
E2  HN outranks target
E3  TM_silver outranks P_silver_target
E4  GSRN outranks target
E5  silver_unresolved selected
E6  easy negative selected
E7  target chunk selected, reader wrong
E8  worse positive selected among several
E9  silver wrong-target answer
E10 target lost by shortlist
E11 attention improves TM but breaks HN
E12 D4 removes required context
E13 D5 merges different values
E14 overlap counted multiple times
E15 QA-based silver label нестабилен/противоречит source-control diagnostics
```

---

# 42. Resource metrics

Для каждого pipeline:

```text
mean latency / document–aspect
p50
p95
peak GPU
peak RAM
B0 calls
R1 calls
reader calls
reader tokens
attention calls
D4 calls
D5 candidates
cache hit rate
```

QA-based silver preparation cost учитывать **отдельно как offline data-preparation cost**:

```text
n Stage-A canonical QA calls
n Stage-B paraphrase QA calls
n Stage-B ownership-probe QA calls
total reader tokens
GPU time
cache hit rate
Stage-B calls / all ambiguous
```

Это вычисления той же frozen QA-модели, а не отдельного teacher.

Не смешивать offline preparation cost с runtime inference cost финальной системы.

---

# 43. Рекомендуемый порядок запуска

```text
Phase 0
frozen QA raw feature cache
→ Stage-A QA mining
→ Stage-B attribution probes
→ fold-specific QA-based silver overlay
→ QC
→ freeze rules/hashes

Phase 1
B0-max8 reference

Phase 2
B1-Silver vs B0
→ paired bootstrap
→ G1

Phase 3
R1-Silver top-5/top-10, если разрешено gate

Phase 4
pairwise/listwise shortlist, если target signal подтверждён

Phase 5
attention specialist TM branch, если TRAIN/CAL audit показывает signal

Phase 6
D4 oracle
D5 independent-evidence audit

Phase 7
D4/D5 only by gates
SpanRanker only by candidate-span oracle

Phase 8
final OOF comparison
cost/Pareto analysis

Phase 9
final refit
freeze
INTERNAL_TEST once
```

---

# 44. Final refit

## 44.1. Одностадийный final model

Train на всех unlocked answerable documents по frozen hyperparameter/epoch policy.

## 44.2. Cascade B0 → R1

Нельзя обучать final R1 на in-sample final-B0 shortlist.

Выполнить:

```text
1. cross-fitted B0 shortlist по всем unlocked documents;
2. для cross-fitted training candidates применить frozen QA-based Stage-A/Stage-B procedure
   с заранее зафиксированной final threshold policy;
3. train final R1 на cross-fitted candidates + resulting QA-based silver labels;
4. train separate final B0 на всех unlocked documents;
5. test runtime: final B0 → shortlist → frozen R1.
```

Если R1 использует frozen reader attention:

```text
reader features для R1 training также извлекать из cross-fitted shortlist.
```

Calibration/fusion:

```text
OOF/CAL predictions only
```

---

# 45. Final manifest

Создать:

```text
final_freeze_manifest_v3_1_silver.json
```

Содержимое:

```text
source data hash
CV split hash
silver raw-feature cache hash
QA model/checkpoint hash
canonical/paraphrase/ownership question-set hashes
Stage-A threshold policy
Stage-B ownership/consistency threshold policy
silver mapping rule hash
QA-based silver QC summary hash
B0 config
R1 config
shortlist K
attention config
D4/D5 decisions
scalers/calibrators
seed policy
epoch policy
primary endpoint
hard safeguards
runtime budget
```

---

# 46. INTERNAL_TEST

Открыть один раз после полного freeze.

Primary final comparison:

```text
final V3.1-Silver
vs
final B0
```

Primary endpoint:

```text
Token F1
```

Secondary original metrics:

```text
EM
Hit@1_original
HN@1
Ambiguous@1_original
GSRN@1_original
cost
```

Frozen QA-based silver procedure может построить INTERNAL_TEST silver diagnostics **только если это заранее записано в final manifest**.

Для этого после единственного открытия test применяются без изменений:

```text
те же canonical/paraphrase/ownership questions
тот же QA checkpoint
та же deterministic normalization
заранее зафиксированная final threshold policy
те же silver mapping rules
```

Такие diagnostics:

```text
не являются gold;
не используются для model selection;
не приводят к изменению pipeline.
```

---

# 47. Что не запускать до silver QC/B1

```text
full-document B2/B3
новый generic structural mix
сложный attention ranker
attention distillation
causal head masking
SpanRanker tuning
D3
INTERNAL_TEST
```

---

# 48. Наиболее вероятная архитектура-кандидат

```text
DOCUMENT
↓
B0 answerability
↓
top-5 / top-10
↓
R1-Silver target-attribution text/discourse
    supervision = QA-based silver labels
↓
optional reader-aware specialist:
    frozen QA → span + attention/hidden features
    → silver-TM reranking
↓
target candidates + cached QA outputs
↓
optional D4
↓
frozen QA / cached answer
↓
optional D5
↓
FINAL ASPECT ANSWER
```

---

# 49. Научная интерпретация результата

Если автоматический overlay помогает, корректная формулировка:

> автоматически сформированный текущей frozen QA-моделью silver target-attribution signal улучшил downstream evidence selection / QA относительно answerability baseline.

Некорректная формулировка без human verification:

> модель научилась истинно отличать все target-mismatch случаи.

Главный независимый downstream критерий остаётся:

```text
Token F1 / EM по original gold
```

а silver metrics показывают согласование модели с frozen QA-based attribution procedure.
