# V4.3 — перепроверенный протокол: semantic reader agreement, честный reader-only baseline и adaptive consensus

## 0. Назначение

Этот документ задаёт следующий экспериментальный цикл после V4.2.

Главная цель:

> проверить, даёт ли **семантическая согласованность predicted answers frozen reader** дополнительный сигнал к подтверждённому V6-K5 внутри того же B0 top-5 shortlist.

Вторичная цель:

> честно проверить reader-only ranking, поскольку диагностический CORE3 из V4.2 дал высокий point estimate, но не был отдельно оптимизирован как самостоятельная система.

Третья цель:

> проверить, можно ли по **силе и отделимости consensus** автоматически определить, когда reader-only ranking надёжен, а когда следует оставить V6.

Четвёртая цель:

> проверить локальную evidence-адекватность predicted span без возврата к raw attention.

`INTERNAL_TEST` в V4.3 остаётся закрыт.

---

# 1. Исходная точка

Подтверждённый reference:

```text
V6-K5
Token F1 = 34.62
Hit@1    = 42.90%
HN@1     = 5.45%
Amb@1    = 42.90%
GSRN@1   = 26.64%
```

Selected-head attention V4.2:

```text
ATT1 Token F1 = 34.76
ΔF1 vs V6 = +0.14
95% CI = [-1.24, +1.56]
```

Следовательно:

```text
attention branch = near-null;
V6-K5 остаётся основной подтверждённой системой.
```

---

# 2. Что важно из CORE3 V4.2

Исторический diagnostic CORE3:

```text
Token F1 = 35.44
EM       = 32.08%
Hit@1    = 41.21%
HN@1     = 6.18%
ΔF1 vs V6 = +0.78
95% CI = [-1.76, +3.30]
```

Особенно:

```text
material: 38.84 vs V6 34.47
method:   34.43 vs V6 26.36
metric:   27.47 vs V6 24.24
task:     36.85 vs V6 45.81
```

Это показывает:

> reader-only consensus может быть очень полезен, но его полезность неоднородна между вопросами.

---

# 3. Ключевое исправление: старый CORE3 — не честный оптимизированный reader-only baseline

В V4.2 CORE3 был diagnostic-only:

```text
CORE3 = CORE2 weights
с удалённым B0 term.
```

То есть его коэффициенты не выбирались специально для reader-only задачи.

Поэтому V4.3 **не использует старый CORE3 как единственный baseline для SEMCORE**.

Нужно построить два новых честно сопоставимых reader-only OOF baseline:

```text
LEXCORE = Majority + normalized lexical agreement
SEMCORE = Majority + semantic agreement
```

с одинаковой процедурой tuning.

Исторический CORE3 сохраняется только как reference diagnostic.

---

# 4. Почему нужен semantic agreement

Текущий reader agreement использует:

```text
exact match
majority
Token F1
```

Но короткие научные ответы могут быть семантически близки при неполном token overlap:

```text
graph neural network
graph convolutional network
GCN-based model
```

Основной вопрос:

> улучшает ли frozen semantic representation именно **межкандидатное agreement**, не меняя QA reader и B0 shortlist?

---

# 5. Hypothesis hierarchy

## H1 — primary

```text
SEM1 > V6-K5
```

где SEM1 = semantic agreement как additive correction к V6.

## H2 — secondary independent

```text
SEMCORE > LEXCORE
```

Это честный тест:

```text
semantic reader-only agreement
vs
lexical reader-only agreement
```

## H3 — exploratory / architecture-discovery

```text
ADAPT1 > V6
```

ADAPT1 использует confidence reader consensus для переключения:

```text
SEMCORE ↔ V6.
```

## H4 — conditional secondary

Если H1 успешна:

```text
SEMCTX1 > SEM1
```

где SEMCTX1 добавляет local evidence-fit.

---

# 6. Candidate universe

Primary:

```text
K = 5
```

Для каждого:

```text
(question_id, seed)
```

использовать exact B0 top-5 из V4.1/V4.2.

Не расширять shortlist.

---

# 7. Frozen components

Не ретюнить:

```text
B0
frozen extractive QA reader
V6 per-run weights
document folds
B0 seeds
```

Seeds:

```text
42
43
44
```

---

# 8. Semantic encoder

Primary encoder фиксируется заранее:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Режим:

```text
frozen
eval
no fine-tuning
```

Не делать model search по нескольким encoders в V4.3.

Важно:

> отрицательные результаты старого KeyBERT document-profile не переносятся автоматически на answer-level semantic agreement: объект сравнения здесь другой.

---

# 9. TRAIN-only semantic sanity audit

До primary EVAL сделать диагностическую проверку **без выбора модели**.

На каждом outer TRAIN собрать answer pairs:

### positive-consensus pairs

Predicted answers из двух reader-easy original-positive chunks одного вопроса.

### contrast pairs

Predicted answer reader-easy positive vs predicted answer valid HN того же вопроса.

Посчитать:

```text
cosine distributions
pairwise P(sim_pospos > sim_posHN)
```

Это:

```text
diagnostic only
```

а не hard gate и не новый tuning signal.

Если pairwise близок к 0.5:
```text
это заранее предупреждает, что SEM1 может быть слабым,
но primary OOF всё равно выполняется.
```

---

# 10. Answer normalization

Перед embedding:

1. Unicode normalize;
2. trim;
3. collapse repeated whitespace;
4. сохранить readable text;
5. не выполнять aggressive stemming/lemmatization;
6. не раскрывать acronyms внешним словарём.

Lexical normalization хранить отдельно для Token-F1/exact comparisons.

---

# 11. Valid-answer policy

Если answer пустой или состоит только из служебных символов:

```text
valid_answer = false
```

Primary semantic feature не должен выдумывать ему embedding-consensus.

Если в shortlist меньше:

```text
3 valid answers
```

то:

```text
semantic_available = false
SemRel = 0 для всех candidates
SEM1 == V6
```

Это соответствует логике Top-2 support:

```text
candidate + минимум 2 других valid answers.
```

---

# 12. Answer embeddings

Для valid predicted answer \(a_i\):

\[
e_i = Encoder(a_i)
\]

\[
\hat e_i =
\frac{e_i}{\|e_i\|_2+\epsilon}
\]

Similarity:

\[
sim_{ij}
=
\hat e_i^\top \hat e_j
\]

---

# 13. Self-similarity исключается

Нельзя использовать:

```text
sim(i,i)=1
```

при agreement.

---

# 14. Primary candidate-level semantic feature

Для valid candidate \(i\):

\[
SemTop2_i
=
mean(
top2_{j\ne i,\ valid} sim_{ij}
)
\]

Если у candidate меньше двух valid других answers:

```text
SemTop2_i = missing
```

и весь shortlist получает:

```text
semantic_available = false.
```

---

# 15. Почему Top-2

Top-1 similarity может отражать:

```text
одну случайную пару distractors.
```

Top-2 требует:

```text
candidate + два поддерживающих answer candidates
```

то есть минимально формирует тройку consensus.

---

# 16. SemMean diagnostic

Дополнительно:

\[
SemMean_i
=
mean_{j\ne i}sim_{ij}
\]

Но primary SEM1 использует:

```text
SemTop2.
```

---

# 17. Shortlist-relative normalization

Для каждого `(question, seed)` при `semantic_available=true`:

\[
SemRel_i
=
Z_q(SemTop2_i)
\]

Если variance:

```text
< epsilon
```

то:

```text
SemRel = 0
```

для всех candidates.

---

# 18. Phase A — SEM1 primary

Reference:

```text
V6-K5
```

\[
S_{SEM1}
=
Z_q(S_{V6})
+
\delta_s SemRel
\]

Grid:

```text
δ_s ∈ {0, 0.1, 0.25, 0.5, 1.0}
```

---

# 19. SEM1 CAL selection

Для каждого outer run:

1. V6 weights frozen из V4.1;
2. semantic encoder frozen;
3. CAL выбирает только `δ_s`;
4. `δ_s=0` обязателен;
5. safeguard:

```text
HN@1_SEM1
<=
HN@1_V6_CAL + 0.02
```

6. objective:

```text
mean CAL Token F1 across seeds,
сначала mean по seed внутри question.
```

Tie-break:
1. меньший δ;
2. меньший HN@1.

---

# 20. SEM1 EVAL

На EVAL:

```text
δ_s frozen
```

Merge только 5 EVAL parts:

```text
true OOF SEM1
```

Primary comparison:

```text
SEM1 vs matching V6-K5.
```

---

# 21. SEM1 success gate

SEM1 подтверждён, если:

```text
ΔToken F1 > 0
95% CI excludes 0
HN safeguard passed
```

Дополнительно:
- gain не определяется одним seed;
- shortlist invariants passed.

---

# 22. Lexical-vs-semantic diagnostic

Для всех valid answer pairs сохранить:

```text
Token F1
semantic cosine
exact-match flag
```

Отдельно изучить:

```text
high semantic / low Token F1
high both
low both
high Token F1 / low semantic
```

Цель:

> увидеть, действительно ли semantic encoder добавляет новый тип agreement.

---

# 23. Phase B — честный LEXCORE baseline

Новый reader-only lexical baseline:

\[
T_i =
mean\_token\_f1_i/100
\]

\[
S_{LEXCORE}
=
\lambda_m Majority_i
+
\lambda_t T_i
\]

Grid:

```text
λ_m ∈ {0.25,0.5,1.0,2.0}
λ_t ∈ {0.25,0.5,1.0,2.0}
```

Разрешить `λ_t=0` отдельным control:

```text
Majority-only.
```

То есть фактически:

```text
λ_t ∈ {0,0.25,0.5,1.0,2.0}
```

---

# 24. LEXCORE tuning

Для обычного LEXCORE OOF:

```text
CAL → выбрать (λ_m, λ_t)
EVAL → frozen
```

Это честный secondary OOF baseline.

Не использовать historical CORE3 coefficients как tuning reference.

---

# 25. Phase C — SEMCORE

\[
S_{SEMCORE}
=
\lambda_m Majority_i
+
\lambda_s SemRel_i
\]

Grid:

```text
λ_m ∈ {0.25,0.5,1.0,2.0}
λ_s ∈ {0.25,0.5,1.0,2.0}
```

Разрешить:

```text
λ_s=0
```

как Majority-only fallback.

---

# 26. Reader-only tie-break

LEXCORE/SEMCORE могут иметь одинаковый score у нескольких candidates.

Чтобы результат был deterministic:

```text
primary tie-break = original B0 rank
```

Это **только tie-break**, а не additive B0 feature.

Отдельно считать:

```text
tie_rate
```

чтобы понимать, насколько reader-only система фактически опирается на B0 при равных score.

---

# 27. SEMCORE comparisons

Primary secondary comparison:

```text
SEMCORE vs LEXCORE
```

Также:

```text
SEMCORE vs V6
LEXCORE vs V6
historical CORE3 vs new LEXCORE
```

Последнее — diagnostic, потому что tuning procedures различались.

---

# 28. Что будет считаться успехом SEMCORE

Сильный результат:

```text
SEMCORE > LEXCORE
```

с positive paired delta и CI без 0.

Дополнительно желательно:

```text
method/material benefit сохраняется
task regression уменьшается
HN не ухудшается заметно.
```

---

# 29. Важное ограничение: SEMCORE не может тюниться дважды перед ADAPT

Для ADAPT1 нельзя:

```text
CAL выбрать SEMCORE weights
и
на том же CAL выбрать routing threshold.
```

Это двойная адаптация на ~90 вопросах.

Поэтому ADAPT1 использует отдельную nested схему.

---

# 30. ADAPT1: reader-only weights выбираются на TRAIN

Для каждого outer run:

```text
TRAIN = 3 folds
```

Выбрать SEMCORE weights через leave-one-TRAIN-fold-out:

1. 2 TRAIN folds → candidate weights;
2. held-out TRAIN fold → Token F1;
3. повторить три раза;
4. выбрать weight pair по mean inner-TRAIN F1;
5. tie-break → меньшая суммарная weight magnitude.

После этого:

```text
SEMCORE weights frozen
до CAL.
```

---

# 31. Ключевое исправление: consensus confidence не равен top1-top2 candidate margin

Плохая идея:

\[
SemTop2_{(1)}-SemTop2_{(2)}
\]

Почему:

```text
если три answers образуют сильный cluster,
у двух-трёх его members SemTop2 почти одинаково высокий,
поэтому margin может быть близок к 0
именно при хорошем consensus.
```

Поэтому V4.3 использует cluster-free **best-triple cohesion/separation**.

---

# 32. Best coherent triple

При K=5 перебрать все тройки valid answers.

Для triple \(T\):

\[
Coh(T)
=
mean_{i<j,\ i,j\in T}
sim_{ij}
\]

Выбрать:

\[
T^*
=
argmax_T Coh(T)
\]

---

# 33. Triple separation

Для лучшей тройки:

\[
Between(T^*)
=
mean_{i\in T^*,\ j\notin T^*}
sim_{ij}
\]

Определить:

\[
ConsensusConf
=
Coh(T^*)
-
Between(T^*)
\]

Интерпретация:

```text
высокий Coh
+
низкая похожесть на оставшиеся alternatives
→ чистый consensus cluster.
```

---

# 34. Consensus strength diagnostics

Сохранять отдельно:

```text
Cohesion = Coh(T*)
Separation = ConsensusConf
```

Не смешивать их сразу в новую обучаемую модель.

Primary ADAPT confidence:

```text
ConsensusConf
```

---

# 35. Valid-answer requirements для ADAPT

Для SEM1 достаточно:

```text
MIN_VALID_ANSWERS_SEM = 3
```

Но `ConsensusConf = Cohesion - Between` требует не только coherent triple, но и хотя бы одну альтернативу.

Поэтому для ADAPT1:

```text
MIN_VALID_ANSWERS_ADAPT = 4
```

Если valid answers < 4:

```text
routing_confidence_available = false
fallback → V6
```

Дополнительно confidence считается применимым только если:

```text
SEMCORE selected candidate ∈ T*
```

где `T*` — best coherent triple.

Если SEMCORE top-1 не входит в `T*`:

```text
routing_confidence_available = false
fallback → V6
```

Иначе существует риск доверить SEMCORE из-за сильного consensus-кластера, к которому его собственный selected answer не относится.

# 36. ADAPT1

Hard routing:

```text
if ConsensusConf >= τ:
    use SEMCORE top1
else:
    use V6 top1
```

---

# 37. ADAPT threshold grid

Использовать CAL empirical quantiles:

```text
Q50
Q60
Q70
Q80
Q90
```

Плюс обязательные controls:

```text
NEVER  = +∞  → всегда V6
ALWAYS = -∞  → всегда SEMCORE
```

Это позволяет CAL честно выбрать:

```text
не использовать routing вообще.
```

---

# 38. ADAPT1 CAL

На CAL:

- SEMCORE weights уже frozen из TRAIN;
- вычислить ConsensusConf без gold;
- подобрать только `τ`;
- HN safeguard:

```text
HN@1_ADAPT
<=
HN@1_V6_CAL + 0.02
```

Objective:

```text
mean CAL Token F1 across seeds.
```

Tie-break:
1. `NEVER / более консервативный routing`;
2. больший threshold;
3. lower HN.

---

# 39. ADAPT1 EVAL

На EVAL frozen:

```text
SEMCORE inner-TRAIN weights
τ from CAL
```

Merge EVAL only.

ADAPT1 остаётся exploratory architecture-discovery system в V4.3.

Если он выглядит сильным:
```text
следующий V4.4 должен повторить adaptive architecture
как отдельную primary hypothesis.
```

---

# 40. Почему ADAPT1 лучше aspect routing

Он использует:

```text
структуру ответов конкретного shortlist
```

а не заранее известный:

```text
task/method/material/metric label.
```

Это более общий механизм.

---

# 41. Phase D — local evidence fit

Предыдущий вариант `context-to-context agreement` не является лучшим primary test против GSRN:

```text
related-work contexts тоже могут быть похожи между собой.
```

Поэтому primary context feature должен отвечать:

> насколько local sentence поддерживает именно пару `question + predicted answer`?

---

# 42. Span-local sentence

Для каждого predicted span извлечь:

```text
sentence containing predicted span
```

Fallback:

```text
±64 context tokens вокруг span
```

если sentence segmentation невозможно.

Не добавлять section labels.

---

# 43. Query / query-answer / evidence representations

Для каждого candidate создать:

```text
Query:
<canonical question>

QueryAnswer:
<canonical question> <predicted answer>

LocalEvidence:
<span-containing sentence>
```

Кодировать тем же frozen semantic encoder:

\[
q = Encoder(Query)
\]

\[
qa_i = Encoder(QueryAnswer_i)
\]

\[
l_i = Encoder(LocalEvidence_i)
\]

и L2-normalize.

`Query` embedding можно кэшировать один раз на question.

---

# 44. Почему raw QueryAnswer ↔ Evidence cosine недостаточен

Если local sentence просто тематически релевантна вопросу, то:

```text
cos(QueryAnswer, Evidence)
```

может быть высоким даже для GSRN / related-work distractor.

Нужно отделить:

```text
общую question relevance
```

от:

```text
добавочной поддержки конкретного predicted answer.
```

---

# 45. Primary local evidence feature: AnswerSupportGain

Base relevance:

\[
BaseFit_i
=
cos(q,l_i)
\]

Question+answer fit:

\[
AnswerFit_i
=
cos(qa_i,l_i)
\]

Primary:

\[
SupportGain_i
=
AnswerFit_i
-
BaseFit_i
\]

Интерпретация:

> насколько добавление конкретного predicted answer делает локальный evidence более согласованным по сравнению с одним только вопросом.

Далее:

\[
SupportRel_i
=
Z_q(SupportGain_i)
\]

внутри top-5.

Сохранять diagnostic:

```text
BaseFit
AnswerFit
SupportGain
```

---

# 46. SEMCTX1

Только если SEM1 прошёл H1 gate:

\[
S_{SEMCTX1}
=
Z_q(S_{SEM1})
+
\delta_c SupportRel
\]

Grid:

```text
δ_c ∈ {0,0.1,0.25,0.5}
```

CAL выбирает только `δ_c`.

`δ_s` из SEM1 уже frozen и не ретюнится.

# 47. Context-to-context agreement

Старую идею:

```text
Answer + LocalContext
vs
Answer + LocalContext других candidates
```

сохранить только как:

```text
diagnostic.
```

Она не входит в primary SEMCTX1.

---

# 48. SEMCTX success question

Primary conditional comparison:

```text
SEMCTX1 vs SEM1
```

Особенно:

```text
GSRN@1
Amb@1
positive → ambiguous
GSRN → positive
```

---

# 49. CORE / V6 complementary-error analysis

Для каждого `(question, seed)` сохранить:

```text
V6 correct?
LEXCORE correct?
SEMCORE correct?
ConsensusConf
Cohesion
B0 score margin
aspect
answer strings
```

Buckets:

```text
V6 win / SEMCORE lose
SEMCORE win / V6 lose
оба win
оба lose
```

---

# 50. B0 margin diagnostic

\[
Margin_{B0}
=
score_{rank1}
-
score_{rank2}
\]

Проверить post-hoc:

```text
слабый B0 margin
↔
SEMCORE/reader consensus полезнее?
```

Diagnostic only.

Не добавлять B0 margin в ADAPT1 в V4.3.

---

# 51. Semantic clustering diagnostic

Можно построить explicit answer clusters:

```text
cosine thresholds = {0.70,0.80,0.85,0.90}
```

Только для analysis.

Не использовать эти thresholds в primary SEM1 или ADAPT1.

Сохранять:

```text
cluster sizes
cluster cohesion
cluster separation
```

---

# 52. Aspect analysis

Для:

```text
V6
SEM1
LEXCORE
SEMCORE
ADAPT1
SEMCTX1
```

считать:

```text
Token F1
Hit@1
HN@1
Amb@1
GSRN@1
EM
```

Особенно:

## method/material
Проверить, сохраняется ли reader-only benefit.

## task
Проверить, избегает ли ADAPT1 провала reader-only режима.

## metric
Проверить, даёт ли semantic agreement больше, чем lexical agreement.

---

# 53. Error buckets lexical vs semantic

Отдельно:

```text
LEXCORE wrong → SEMCORE correct
LEXCORE correct → SEMCORE wrong

V6 wrong → SEM1 correct
V6 correct → SEM1 wrong
```

Для каждого bucket сохранить:
- answer strings;
- Token F1 similarity;
- semantic similarity;
- aspect;
- selected source label.

---

# 54. Metrics

Primary endpoint:

```text
Token F1
```

Secondary:

```text
EM
Hit@1
MRR
HN@1
Amb@1
GSRN@1
F1 | selected positive
```

Agreement diagnostics:

```text
SemTop2
SemMean
Cohesion
ConsensusConf
semantic_available rate
reader-only tie rate
```

---

# 55. Statistical evaluation

Paired clustered bootstrap:

```text
cluster = document_id
n_boot >= 5000
```

Pairing:

```text
(question_id, seed)
```

Procedure:

1. delta candidate-reference per seed;
2. mean delta across seeds per question;
3. bootstrap documents.

Seeds не считать independent samples.

---

# 56. Main comparisons

Обязательные:

```text
H1: SEM1 vs V6

H2: SEMCORE vs LEXCORE

diagnostic:
LEXCORE vs V6
SEMCORE vs V6
historical CORE3 vs LEXCORE

H3 exploratory:
ADAPT1 vs V6

H4 conditional:
SEMCTX1 vs SEM1
```

Не добавлять новые primary comparisons после просмотра OOF.

---

# 57. Stop SEM1

Если:

```text
δ_s=0
в большинстве CAL runs
```

или:

```text
ΔF1 SEM1-V6 <= 0
```

или CI явно показывает отсутствие практически значимого gain:

```text
semantic additive correction не подтверждён.
```

SEMCORE всё равно можно оценить как отдельный H2.

---

# 58. Stop SEMCORE

Если:

```text
SEMCORE не лучше LEXCORE
```

и/или:

```text
task/HN профиль остаётся плохим,
```

reader-only semantic branch не выводить в final candidates.

---

# 59. Stop ADAPT1

Если:

```text
NEVER
```

выбирается в большинстве CAL или OOF не лучше V6:

```text
adaptive routing не поддержан.
```

Нельзя затем искать новые confidence formulas по тому же EVAL.

---

# 60. Stop SEMCTX1

Если:

```text
ΔF1 SEMCTX1-SEM1 <= 0
```

или:

```text
GSRN/HN ухудшаются,
```

context correction закрыть.

---

# 61. U80 policy

Не смешивать U80 с V4.3 primary.

Если:
- H1 не прошла;
- H2 не дал нового reader-only gain;
- H3/H4 не дают устойчивого результата,

следующий приоритет:

```text
corrected U80 reader-utility weak supervision.
```

---

# 62. Hidden-state policy

Если output-level semantic agreement исчерпан:

следующая internal-representation ветка:

```text
question hidden representation
↔
predicted-span hidden representation
```

а не raw attention weights.

Не запускать hidden-state learning в V4.3.

---

# 63. Efficiency

Измерить:

```text
answer embedding latency
query-answer embedding latency
local-evidence embedding latency
CPU/GPU
RAM
cache hit rate
```

Marginal cost:

```text
V6
SEM1
SEMCORE
ADAPT1
SEMCTX1
```

---

# 64. Embedding cache

Key:

```text
semantic_model_hash
normalization_version
text_hash
```

Отдельные stores:

```text
answer embeddings
query-answer embeddings
local-evidence embeddings
```

Pure answer embeddings одинаковой строки можно переиспользовать между вопросами.

Query-answer embeddings нельзя переиспользовать без question в key.

---

# 65. Leakage rules

Semantic encoder:

```text
frozen
no SciREX fine-tuning
```

Gold:
- не используется при runtime features;
- используется для CAL selection / EVAL metrics по существующему протоколу;
- TRAIN-only semantic sanity audit может использовать source labels/gold.

ADAPT SEMCORE weights:
```text
TRAIN inner folds only.
```

ADAPT threshold:
```text
CAL only.
```

EVAL:
```text
no tuning.
```

---

# 66. Regression tests

Обязательные:

1. B0 top-5 identity.
2. V6 identity.
3. deterministic embeddings.
4. cosine self-sim ≈ 1.
5. self-sim excluded.
6. identical normalized answer → equal answer embedding.
7. fewer than 3 valid answers → SEM1 falls back exactly to V6.
8. `δ_s=0` → SEM1 exactly V6.
9. zero SemTop2 variance → SemRel=0, no NaN.
10. LEXCORE/SEMCORE tie-break deterministic.
11. historical CORE3 reproduction kept separate from new LEXCORE.
12. fewer than 4 valid answers → ADAPT fallback exactly to V6.
13. SEMCORE top1 outside best coherent triple → ADAPT fallback exactly to V6.
14. ADAPT `NEVER` → exactly V6.
15. ADAPT `ALWAYS` → frozen SEMCORE only when routing confidence is available; otherwise V6 fallback.
16. local sentence contains predicted span.
17. SupportGain = AnswerFit - BaseFit exactly.
18. zero SupportGain variance → SEMCTX correction 0.
19. no EVAL leakage in inner-TRAIN tuning.

---

# 67. Artifacts

```text
artifacts/qa_chunk_selection_v4_3/

contract.json

reference/
    b0_live/
    v6_k5_v4_1/
    core3_v4_2_historical/

semantic_cache/
    answer_embeddings/
    query_answer_embeddings/
    local_evidence_embeddings/
    pairwise_similarity.jsonl

semantic_sanity/
    run0/
    ...
    pair_distributions.json

sem1/
    run0/
    ...
    oof_predictions.jsonl
    metrics.json
    bootstrap_vs_v6.json

lexcore/
    run0/
    ...
    oof_predictions.jsonl

semcore/
    run0/
    ...
    oof_predictions.jsonl
    bootstrap_vs_lexcore.json

adapt1/
    run0/
        inner_semcore_weights.json
        cal_threshold.json
    ...
    oof_predictions.jsonl

semctx1/
    ...

analysis/
    lexical_vs_semantic.json
    consensus_confidence.json
    core_complementarity.json
    aspect_metrics.json
    transitions.json
    efficiency.json

reports/
    results_v4_3.md
```

---

# 68. Contract до первого EVAL

Зафиксировать:

```text
K = 5
seeds = [42,43,44]

SEMANTIC_MODEL =
sentence-transformers/all-MiniLM-L6-v2

SEM_FEATURE =
mean top-2 cosine similarities excluding self

MIN_VALID_ANSWERS_SEM = 3
MIN_VALID_ANSWERS_ADAPT = 4

DELTA_SEM_GRID =
[0,0.1,0.25,0.5,1.0]

LEXCORE_MAJORITY_GRID =
[0.25,0.5,1.0,2.0]

LEXCORE_TOKEN_GRID =
[0,0.25,0.5,1.0,2.0]

SEMCORE_MAJORITY_GRID =
[0.25,0.5,1.0,2.0]

SEMCORE_SEM_GRID =
[0,0.25,0.5,1.0,2.0]

ADAPT_QUANTILES =
[0.50,0.60,0.70,0.80,0.90]
+ NEVER
+ ALWAYS

CONTEXT_DELTA_GRID =
[0,0.1,0.25,0.5]

HN_EPSILON = 0.02
BOOTSTRAP = 5000
```

После первого outer EVAL contract не расширять.

---

# 69. Рекомендуемый порядок запуска

```text
STEP 1
reuse exact V4.1/V4.2 references

STEP 2
build answer embedding cache

STEP 3
run TRAIN-only semantic sanity audit

STEP 4
compute top-5 semantic matrices

STEP 5
run SEM1 true 5-fold OOF

STEP 6
build fair LEXCORE true OOF

STEP 7
build SEMCORE true OOF
compare SEMCORE vs LEXCORE

STEP 8
analyze V6 / LEXCORE / SEMCORE complementary errors

STEP 9
for ADAPT1:
    select SEMCORE weights on inner TRAIN
    select only confidence threshold on CAL
    evaluate EVAL once

STEP 10
if H1 passes:
    extract local evidence
    run SEMCTX1

STEP 11
aspect / transitions / efficiency

STEP 12
choose exactly one next research direction

STEP 13
keep INTERNAL_TEST locked
```

---

# 70. Короткая инструкция агенту

```text
Use exact V4.1 V6-K5.
Do not retune V6.
Keep K=5 and seeds 42/43/44.

Freeze one semantic encoder:
all-MiniLM-L6-v2.

For every top-5:
    normalize predicted answer text;
    require >=3 valid answers;
    embed answers;
    cosine matrix;
    exclude self;
    SemTop2 = mean of two highest valid similarities;
    z-normalize SemTop2 within shortlist.

SEM1:
    z(V6) + delta_sem * SemRel
    choose only delta_sem on CAL.
Primary test = SEM1 vs V6.

Do NOT compare SEMCORE only with historical CORE3.
Create a fair new lexical reader-only baseline:
    LEXCORE = Majority + normalized Token-F1
with the same OOF tuning design.

Create:
    SEMCORE = Majority + semantic agreement.
Secondary test = SEMCORE vs LEXCORE.

For ADAPT1:
    do NOT use top1-top2 SemTop2 margin as confidence.
    Find best coherent triple of answers.
    confidence =
        triple cohesion
        -
        similarity of triple to remaining alternatives.
    Require >=4 valid answers.
    Require SEMCORE selected candidate to belong to the best triple.
    Otherwise fallback to V6.

Select SEMCORE weights using inner TRAIN only.
Select only routing threshold on CAL.
Include NEVER=always V6 and ALWAYS=SEMCORE-when-confidence-available controls.

Context branch:
    do not use context-to-context agreement as primary.
    encode Query, Query+PredictedAnswer, and local evidence sentence.
    BaseFit   = cos(Query, Evidence)
    AnswerFit = cos(Query+Answer, Evidence)
    SupportGain = AnswerFit - BaseFit
    use shortlist-normalized SupportGain as the context correction.
    add this only after SEM1 is frozen.

Do not add attention, K10, U80, hidden-state learning,
aspect routing, or multiple semantic encoders.

Keep INTERNAL_TEST locked.
```

---

# 71. Главная исследовательская формулировка

> После подтверждения reader-centered reranking и отрицательного результата selected-head attention проверить, даёт ли семантическая согласованность ответов frozen reader дополнительный evidence-signal к V6; честно сравнить semantic и lexical reader-only consensus при одинаковой процедуре настройки; затем проверить, можно ли по cohesion/separation согласованной группы ответов автоматически определить, когда reader-only ranking следует предпочесть relevance-aware V6.

---

# 72. Возможные результаты и дальнейшее решение

## A. SEM1 подтверждён

```text
semantic agreement даёт независимый additive gain
```

→ стабилизировать semantic V6.

## B. SEMCORE > LEXCORE, ADAPT1 promising

```text
semantic consensus полезен прежде всего
для выбора режима ranking
```

→ отдельный V4.4 adaptive-routing protocol.

## C. SEMCTX1 подтверждён

```text
question-answer ↔ local evidence fit
помогает против GSRN
```

→ развивать context-aware evidence consistency.

## D. Все ветки near-null

```text
heuristic output-level agreement близок к потолку
```

→ перейти к:
1. corrected U80 reader-utility weak supervision;
или
2. hidden-state/span representation.

---

# 73. INTERNAL_TEST

`INTERNAL_TEST` не открывать в V4.3.

Перед locked test должны остаться:
- одна architecture;
- один hyperparameter contract;
- final B0/refit procedure;
- freeze manifest.

Только затем:

```text
one locked INTERNAL_TEST evaluation.
```
