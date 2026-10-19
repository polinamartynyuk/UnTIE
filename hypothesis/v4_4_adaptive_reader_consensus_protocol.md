# V4.4 — финально перепроверенный протокол: adaptive reader-consensus и выбор кандидата перед locked test

## 0. Назначение

Этот документ задаёт следующий экспериментальный цикл после V4.3.

Главная цель:

> проверить и стабилизировать архитектуру **adaptive reader-consensus**, которая выбирает между relevance-aware V6 и reader-only consensus в зависимости от структуры ответов B0 top-5.

Вторичная цель:

> определить, какой reader-only режим лучше использовать внутри adaptive схемы: **LEXCORE** или **SEMCORE**.

Третья цель:

> при подтверждении adaptive архитектуры проверить лёгкий local-evidence veto для уменьшения GSRN/HN без возврата к attention.

V4.4 — **последний model-selection / stability цикл на уже многократно использованном unlocked-пуле**, но не новое независимое подтверждение. `INTERNAL_TEST` остаётся закрыт до выбора одной полностью deployable архитектуры.

---

# 1. Что установлено V4.3

Подтверждённый reference:

```text
V6-K5
Token F1 = 34.62
Hit@1    = 42.90%
HN@1     = 5.45%
```

SEM1:

```text
Token F1 = 34.69
ΔF1 vs V6 ≈ +0.07
95% CI = [-0.68, +0.82]
```

Следовательно, простое добавление semantic similarity поверх V6 практически не даёт нового сигнала.

---

# 2. Reader-only результаты V4.3

LEXCORE:

```text
Token F1 = 36.58
Hit@1    = 43.41%
HN@1     = 5.81%
ΔF1 vs V6 = +1.92
95% CI = [-0.015, +3.888]
```

SEMCORE:

```text
Token F1 = 37.33
Hit@1    = 42.53%
HN@1     = 6.55%
ΔF1 vs V6 = +2.69
95% CI = [+0.89, +4.46]
```

SEMCORE vs LEXCORE:

```text
ΔF1 = +0.77
95% CI = [-0.90, +2.45]
```

Вывод:

> reader-only consensus перспективен, но semantic agreement пока не доказал превосходство над более простым lexical agreement.

---

# 3. Почему adaptive architecture сейчас главный приоритет

V6 и SEMCORE имеют реальную complementarity:

```text
SEMCORE win / V6 lose = 103
V6 win / SEMCORE lose = 108
```

Exploratory ADAPT1 V4.3:

```text
Token F1 = 36.02
Hit@1    = 43.19%
HN@1     = 6.18%
ΔF1 vs V6 = +1.38
95% CI = [+0.50, +2.30]
```

Это означает, что идея переключаться между V6 и reader-only режимом имеет эмпирическую основу.

---

# 4. Почему V4.4 нужен отдельно

ADAPT1 V4.3 был architecture-discovery, а не заранее зафиксированная primary система.

V4.4 нужен, чтобы:
1. заранее зафиксировать конкретную adaptive architecture;
2. резко сократить tuning freedom;
3. выполнить **fold-respecting cross-fitted stability replay**;
4. проверить механистическую устойчивость routing;
5. выбрать ровно одного кандидата для final freeze.

Важно:

> поскольку Q70, сама adaptive идея и fixed reader weights выбраны после анализа V4.3 на тех же 453 unlocked вопросах, результаты V4.4 нельзя называть новым независимым или «честным confirmatory OOF».

Bootstrap CI V4.4 описывает неопределённость **на development-пуле**, но не устраняет selection bias предыдущих циклов.

Настоящее независимое подтверждение остаётся:

```text
ONE frozen pipeline
→ locked INTERNAL_TEST
```

# 5. Основные исследовательские вопросы

## H1 — primary stability question

> Сохраняется ли преимущество заранее зафиксированного adaptive routing `V6 ↔ SEMCORE`, то есть именно той архитектурной идеи, которая дала положительный exploratory результат ADAPT1 в V4.3?

## H2 — secondary substitution question

> Можно ли заменить SEMCORE внутри router на LEXCORE без потери качества и с более безопасным HN/Hit@1 профилем?

## H3 — exploratory only

> Имеет ли local evidence support потенциал отбрасывать плохие reader-only switches?

H3 в V4.4 **не может заменить H1/H2 winner** и не входит в final selection без отдельного следующего протокола.

# 6. Candidate universe

Primary:

```text
K = 5
```

Использовать exact B0 top-5.

Не запускать в V4.4:

- K=10;
- full-document reranking;
- attention;
- U80;
- hidden-state branch;
- aspect-specific routing.

---

# 7. Frozen components

Не ретюнить:

```text
B0
frozen QA reader
V6 per-run weights из V4.1
folds
seeds
```

Seeds:

```text
42, 43, 44
```

---

# 8. Frozen semantic encoder

Использовать тот же encoder, что в V4.3:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Режим:

```text
frozen
eval
no fine-tuning
```

Не проводить model search по encoders.

---

# 9. Fixed LEX reader-only score

V4.3 показал:

- `λ_m = 0.25` на всех пяти CAL runs;
- `λ_t = 2.0` на трёх runs, 1.0 на одном и 0.5 на одном.

Для заранее фиксированного secondary substitution test V4.4 используется modal pair:

\[
S_{LEX}
=
0.25\,Majority
+
2.0\,T_i
\]

где:

\[
T_i = mean\_token\_f1_i/100.
\]

Обозначение:

```text
LEX-FIX
```

Важно:

> это исторически выбранная development-конфигурация, а не независимо оценённый optimal global weight vector.

Она не ретюнится в V4.4.

# 10. Fixed SEM reader-only score

Для SEMCORE обоснование сильнее: в V4.3 inner-TRAIN ADAPT selection на **всех пяти outer runs** выбрала:

```text
λ_m = 0.5
λ_s = 0.25
```

Поэтому primary reader-only component фиксируется как:

\[
S_{SEM}
=
0.5\,Majority
+
0.25\,SemRel_i.
\]

Обозначение:

```text
SEM-FIX
```

Эти веса не меняются по run и не ретюнятся.

# 11. Tie-break reader-only

При равных score:

```text
original B0 rank
```

Это только deterministic tie-break, а не additive B0 feature.

Отдельно считать:

```text
tie_rate
```

для LEX-FIX и SEM-FIX.

---

# 12. Почему reader-only weights нельзя снова тюнить

Цель V4.4 — проверить уже найденную architecture, а не повторить V4.3 с новой сеткой.

Поэтому reader-only weights:

- одинаковы для всех outer runs;
- не выбираются по CAL;
- не меняются после просмотра EVAL.

---

# 13. Consensus confidence

Использовать ту же best-triple схему из V4.3.

Для каждой тройки valid answers \(T\):

\[
Coh(T)
=
mean_{i<j,\ i,j\in T} sim_{ij}.
\]

Лучшая тройка:

\[
T^* = argmax_T Coh(T).
\]

Связь с альтернативами:

\[
Between(T^*)
=
mean_{i\in T^*,\ j\notin T^*} sim_{ij}.
\]

Confidence:

\[
ConsensusConf
=
Coh(T^*) - Between(T^*).
\]

---

# 14. Routing availability

Routing к reader-only режиму разрешён только если:

```text
valid answers >= 4
```

и:

```text
reader-only selected candidate ∈ T*
```

Иначе:

```text
fallback → V6
```

Это предотвращает использование confidence сильного consensus-кластера для кандидата, который сам в этот кластер не входит.

---

# 15. Routing threshold: Q70 — исторически зафиксированный operating point

В V4.3 adaptive SEMCORE router выбирал:

```text
Q70 / Q70 / ALWAYS / ALWAYS / Q50
```

То есть:
- adaptive routing имел положительный exploratory signal;
- конкретный threshold был нестабилен;
- `Q70` дважды повторился как непредельный режим.

V4.4 фиксирует:

```text
Q70
```

**до нового replay**, но честно помечает его как hyperparameter, выбранный на основании истории V4.3.

Поэтому Q70 нельзя трактовать как независимо предзаданный threshold относительно всего research program.

# 16. Как вычислять Q70 внутри development cross-fit

Для каждого outer run:

1. собрать `ConsensusConf` только на non-EVAL части (`TRAIN+CAL`);
2. использовать только строки, где `routing_available=true`;
3. не использовать gold labels;
4. вычислить:

\[
\tau_{70}^{(run)}
=
quantile_{0.70}
(
ConsensusConf_{TRAIN+CAL,\ available}
).
\]

Если available-строк недостаточно для устойчивой квантили, заранее задать:

```text
MIN_ROUTING_CAL_ROWS = 100
```

и при меньшем числе:

```text
router disabled for this run
→ exact V6.
```

EVAL в распределение threshold не входит.

# 17. Что именно доказывает такой Q70 replay

Использование non-EVAL распределения feature без gold сохраняет **fold isolation текущего расчёта**, но не делает V4.4 независимым подтверждением, потому что:

```text
Q70 как operating point
+
adaptive architecture
+
SEM-FIX
```

были выбраны после просмотра предыдущих результатов того же unlocked-пула.

Поэтому корректное название результата:

```text
cross-fitted development stability estimate
```

а не:

```text
new confirmatory OOF.
```

# 18. Primary architecture ASEM

ASEM = adaptive V6 ↔ SEM-FIX.

```text
if routing_available
and ConsensusConf >= τ70:
    use SEM-FIX top1
else:
    use V6 top1
```

Именно эта схема является фиксированным продолжением exploratory ADAPT1 V4.3.

# 19. Secondary architecture ALEX

ALEX = adaptive V6 ↔ LEX-FIX.

```text
if routing_available
and ConsensusConf >= τ70:
    use LEX-FIX top1
else:
    use V6 top1
```

ALEX — новая substitution-ветка: она не была проверена как adaptive architecture в V4.3.

# 20. Почему ASEM — primary

Положительный adaptive результат V4.3 был получен именно для:

```text
V6 ↔ SEMCORE
```

с `SEMCORE weights = (0.5, 0.25)` во всех inner-TRAIN runs.

Поэтому единственный методологически корректный primary stability question V4.4:

```text
ASEM vs V6.
```

Использовать ALEX как primary означало бы тестировать новую архитектуру, а не стабилизировать найденную.

# 21. Почему ALEX — secondary

LEXCORE V4.3:
- дал высокий Hit@1;
- имел более низкий HN@1, чем SEMCORE;
- был дешевле в reader-only scoring;
- но adaptive `V6 ↔ LEXCORE` ранее не проверялся.

Кроме того:

> ALEX всё равно использует semantic `ConsensusConf` для routing, поэтому он **не устраняет MiniLM cost** целиком.

H2 проверяет прежде всего quality/safety substitution, а не полноценное computational simplification.

# 22. Primary development gate H1

ASEM считается **preferred development candidate** после V4.4, если одновременно:

```text
ΔToken F1(ASEM−V6) > 0
95% development CI excludes 0
HN@1_ASEM <= HN@1_V6 + 1.0 percentage point
Hit@1_ASEM >= Hit@1_V6 - 0.5 percentage point
```

Термин `development CI` обязателен: этот CI не превращает H1 в независимое подтверждение из-за repeated use unlocked data.

Даже при H1 pass финальная гипотеза подтверждается только locked INTERNAL_TEST.

# 23. Secondary comparison H2

Если H1 development gate прошёл, сравнить:

```text
ALEX vs ASEM
```

ALEX может заменить ASEM как preferred final candidate, если выполнено одно из двух:

### superiority

```text
ΔF1(ALEX−ASEM) > 0
и development CI excludes 0
```

при допустимом HN;

### safety-preserving non-inferiority

```text
lower CI(ALEX−ASEM) > -0.5 F1
```

и одновременно:

```text
HN@1_ALEX <= HN@1_ASEM
Hit@1_ALEX >= Hit@1_ASEM - 0.5 п.п.
```

Поскольку semantic routing cost у ALEX остаётся, «простота» сама по себе не является достаточным основанием выбора.

# 24. Controls

Обязательные development outputs:

```text
V6
SEM-FIX always
LEX-FIX always
ASEM Q70
ALEX Q70
```

Historical ADAPT1 V4.3:
```text
reference diagnostic only.
```

Ни один always-control не становится final candidate post-hoc.

# 25. Routing-rate audit

Для ASEM и ALEX считать:

```text
routing_available_rate
reader_only_used_rate
fallback_rate
```

Отдельно:
- по outer run;
- по seed;
- по aspect.

Также фиксировать:

```text
n_threshold_source_rows
tau70
```

чтобы видеть устойчивость unsupervised calibration.

# 26. Анализ только случаев disagreement

Adaptive routing имеет смысл только если:

```text
V6 top1 != reader-only top1.
```

Поэтому отдельно считать:

```text
n_disagreement
n_reader_switch_on_disagreement
reader_better
v6_better
tie
```

---

# 27. Routing precision

Post-EVAL diagnostic:

\[
RoutingPrecision
=
\frac{
\#(switch\ to\ reader\ \land reader\ better\ than\ V6)
}{
\#(switch\ to\ reader\ \land systems\ disagree)
}.
\]

`better` определяется по конечному Token F1 ответа.

Не использовать этот target при tuning V4.4.

---

# 28. Routing recall

\[
RoutingRecall
=
\frac{
\#(switch\ to\ reader\ \land reader\ better)
}{
\#(reader\ better\ than\ V6)
}.
\]

Это показывает, насколько router:

- слишком консервативен;
- или слишком агрессивен.

---

# 29. Confidence calibration diagnostics

После OOF разделить `ConsensusConf` на квинтили:

```text
0–20%
20–40%
40–60%
60–80%
80–100%
```

В каждом bin считать:

```text
P(reader better than V6)
P(V6 better than reader)
mean ΔF1(reader−V6)
n disagreements
```

---

# 30. Confidence monotonicity

Желаемое поведение:

```text
чем выше ConsensusConf,
тем чаще reader-only лучше V6.
```

Если такого тренда нет, hard routing считается механистически слабым даже при небольшом среднем gain.

---

# 31. Aspect diagnostics

Для:

```text
material
method
metric
task
```

считать:

```text
V6
LEX-FIX
SEM-FIX
ALEX
ASEM
```

Router не получает aspect labels.

---

# 32. Главный aspect-вопрос

LEXCORE V4.3:

```text
method: сильный gain
task: regression
```

SEMCORE:

```text
task: восстановление
method: слабее LEXCORE
```

Поэтому ключевая диагностика:

> сохраняет ли ALEX сильный method gain и одновременно избегает task regression за счёт fallback к V6?

---

# 33. B0 margin

Не использовать как routing feature.

V4.3 не показал, что слабый B0 margin связан с большей пользой SEMCORE.

Сохранять только post-hoc diagnostic.

---

# 34. Optional ASOFT diagnostic

Только если ASEM проходит H1 development gate.

Exploratory soft routing:

\[
w = sigmoid(10(ConsensusConf-\tau_{70}))
\]

\[
S = (1-w)Z(V6)+wZ(SEM).
\]

ASOFT:

```text
diagnostic only
```

Он не участвует в final selection V4.4 и не может заменить ASEM/ALEX без отдельного протокола.

# 35. Почему soft routing не primary

Нельзя одновременно снова оптимизировать:

- threshold;
- slope;
- mixture weights;
- routing features.

Это создаст новый широкий model search.

V4.4 primary должен оставаться максимально фиксированным.

---

# 36. Conditional local-evidence diagnostic

Запускать только если H1 ASEM прошёл development gate.

Эта ветка проверяет механизм:

> может ли local evidence support отбрасывать часть плохих reader-only switches?

Но поскольку feature `SupportGain` ещё не был оценён в V4.3, V4.4 рассматривает его только как:

```text
exploratory diagnostic
```

Он **не может стать final freeze candidate** в этой версии.

# 37. Local evidence extraction

Для predicted span reader-only candidate извлечь:

```text
sentence containing predicted span
```

Fallback:

```text
±64 context tokens вокруг span
```

---

# 38. SupportGain

Тем же frozen MiniLM:

\[
BaseFit_i
=
cos(Encoder(Query), Encoder(LocalEvidence_i))
\]

\[
AnswerFit_i
=
cos(Encoder(Query+Answer_i), Encoder(LocalEvidence_i))
\]

\[
SupportGain_i
=
AnswerFit_i - BaseFit_i.
\]

Идея:

> измерить, насколько конкретный predicted answer добавляет поддержку сверх общей релевантности sentence вопросу.

---

# 39. SupportRel

Внутри top-5:

\[
SupportRel_i = Z_q(SupportGain_i).
\]

Если variance≈0:

```text
SupportRel = 0
```

---

# 40. Local support используется как veto

Определить `PREF`:

```text
PREF = ASEM
```

или, если H2 выбрал substitution:

```text
PREF = ALEX.
```

Если PREF хочет перейти на reader-only candidate:

```text
if SupportRel(reader_candidate) < τ_support:
    fallback → V6
else:
    keep reader-only switch
```

Это diagnostic veto, а не новый broad reranker.

# 41. Support threshold

Не тюнить по F1.

Использовать:

```text
τ_support = Q20
```

но квантили считать **только среди потенциальных reader-switch candidates** на `TRAIN+CAL`, где:
- routing available;
- reader-only top1 != V6 top1.

Это соответствует runtime distribution veto.

Gold не используется.

Если таких строк меньше:

```text
MIN_SUPPORT_CAL_ROWS = 100
```

support diagnostic отключается.

# 42. PREF-SUPPORT diagnostic

Обозначение:

```text
PREF-SUPPORT
```

где PREF = выбранная development adaptive система.

Comparison:

```text
PREF-SUPPORT vs PREF
```

Главные diagnostics:
- Token F1;
- Hit@1;
- HN@1;
- GSRN@1;
- veto rate;
- good-veto / bad-veto counts.

Не включать PREF-SUPPORT в final candidate list V4.4.

# 43. Интерпретация local support

Local support интересен, если:

```text
HN/GSRN уменьшаются
при практически неизменном F1
```

или:

```text
F1 растёт
при не худшем HN.
```

Для описания practical non-inferiority можно использовать:

```text
-0.25 F1
```

но только как exploratory diagnostic criterion.

Если signal выглядит полезным:
```text
оформить отдельный следующий protocol amendment
до locked test
```

или отказаться от него ради немедленного freeze основной adaptive системы.

# 44. Не использовать context-to-context similarity как primary

Причина:

```text
related-work / GSRN contexts тоже могут быть похожи между собой.
```

Primary local signal — именно `Question+Answer ↔ LocalEvidence` relative support.

---

# 45. Statistical evaluation

Для OOF:

```text
pairing = (question_id, seed)
cluster = document_id
n_boot >= 5000
```

Порядок:

1. delta per seed;
2. mean delta per question;
3. clustered bootstrap documents.

Seeds не являются independent samples.

---

# 46. Primary comparison hierarchy

Development hierarchy:

```text
H1: ASEM vs V6
H2: ALEX vs ASEM   (только если H1 pass)
H3: PREF-SUPPORT diagnostic only
```

Не добавлять новые final-candidate hypotheses после просмотра development replay.

# 47. Hierarchical development-selection rule

## Если H1 fail

```text
preferred final candidate = V6-K5
```

ALEX и support не могут post-hoc стать winner этой версии.

## Если H1 pass

```text
preferred = ASEM
```

## Затем H2

ALEX заменяет ASEM только по заранее заданному superiority/non-inferiority + safety rule.

## H3

Local support только информирует будущую ветку и не меняет preferred candidate V4.4.

# 48. Near-positive rule

Если:

```text
ΔF1 > 0
CI включает 0
```

то система остаётся exploratory.

Нельзя переносить её на locked test вместо V6.

---

# 49. Efficiency

Измерить:

```text
LEX feature cost
semantic confidence cost
routing overhead
local support embedding cost
total latency
peak RAM / GPU
```

Сравнить:

```text
V6
ALEX
ASEM
PREF-SUPPORT
```

---

# 50. Cache reuse

Переиспользовать:

- QA cache V4.1–V4.3;
- answer embeddings V4.3.

Новые embeddings нужны только для conditional support:

```text
Query
Query+Answer
LocalEvidence
```

---

# 51. Regression tests

Обязательные:

1. B0 top-5 identity.
2. V6 per-run identity.
3. SEM-FIX deterministic.
4. LEX-FIX deterministic.
5. Q70 считается только по non-EVAL `routing_available` rows.
6. Q70 computation не использует gold.
7. недостаточно calibration rows → router disabled → exact V6.
8. routing unavailable → exact V6.
9. ASEM switch → exact SEM-FIX top1.
10. PREF switch → exact LEX-FIX top1.
11. reader-selected candidate outside `T*` → exact V6 fallback.
12. support Q20 считается только по potential-switch rows.
13. SupportGain formula exact.
14. support veto → exact V6.
15. no NaN при zero variance.
16. никаких INTERNAL_TEST reads.
17. final deployable manifest не содержит per-run V6 weights как runtime rule.

# 52. Transition analysis

Считать:

```text
V6 → ASEM
V6 → ALEX
PREF → PREF-SUPPORT
```

Transitions:

```text
ambiguous → positive
GSRN → positive
HN → positive
positive → ambiguous
positive → HN
positive → easy
```

---

# 53. Error buckets

Для каждого switch сохранить:

```text
question_id
seed
aspect
V6 chunk / answer
reader chunk / answer
ConsensusConf
reader candidate ∈ T*
source label V6
source label reader
Token F1 V6
Token F1 reader
```

Buckets:

```text
good switch
bad switch
unnecessary switch
missed opportunity
```

---

# 54. Artifacts

```text
artifacts/qa_chunk_selection_v4_4/

contract.json

reference/
    v6_k5_oof/
    semcore_v4_3/
    lexcore_v4_3/
    adapt1_v4_3_historical/

fixed_readers/
    sem_fix.json
    lex_fix.json

confidence/
    run0/
        tau70.json
        bins.json
    ...

asem/
    run0/
    ...
    crossfit_predictions.jsonl
    metrics.json
    bootstrap_vs_v6.json

alex/
    ...
    bootstrap_vs_asem.json

support_diagnostic/
    embeddings/
    pref_support/

analysis/
    routing_rate.json
    routing_precision_recall.json
    confidence_monotonicity.json
    aspect_metrics.json
    transitions.json
    efficiency.json

final/
    development_selection.json
    deployable_freeze_manifest_draft.json

reports/
    results_v4_4.md
```

# 55. Contract before first EVAL

Зафиксировать:

```text
K = 5
seeds = [42,43,44]

SEM_FIXED:
    lambda_majority = 0.5
    lambda_sem = 0.25

LEX_FIXED:
    lambda_majority = 0.25
    lambda_token = 2.0

ROUTING_CONFIDENCE =
    best-triple cohesion - between similarity

MIN_VALID_ANSWERS_ADAPT = 4
MIN_ROUTING_CAL_ROWS = 100

ROUTING_THRESHOLD =
    Q70 of non-EVAL TRAIN+CAL
    routing_available confidence distribution
    no gold

HN_PRIMARY_MARGIN = +1.0 percentage point
HIT1_PRIMARY_MARGIN = -0.5 percentage point

SUPPORT_DIAGNOSTIC_THRESHOLD =
    Q20 of TRAIN+CAL potential-switch SupportRel
    no gold

MIN_SUPPORT_CAL_ROWS = 100

BOOTSTRAP = 5000

STATUS =
    cross-fitted development stability replay
    NOT independent confirmatory OOF
```

После первого EVAL ничего не менять.

# 56. Что делать с historical ADAPT1

Использовать только как reference:

```text
V4.3 ADAPT1 = exploratory architecture-discovery
```

Не использовать его threshold mix как OOF candidate V4.4.

---

# 57. Почему V4.4 нельзя называть окончательным confirmatory test

Архитектура и operating point V4.4 сформированы после анализа V4.3 на тех же unlocked 453 вопросах.

Поэтому:
- fold isolation текущего replay можно сохранить;
- leakage внутри run можно исключить;
- но **research-level selection bias уже существует**.

Корректная формулировка:

```text
cross-fitted development stability evidence
```

Некорректная:

```text
independent confirmatory OOF evidence
```

Настоящий confirmatory endpoint:

```text
ONE locked INTERNAL_TEST
```

после полного deployable freeze.

# 58. Development candidate selection

В конце V4.4 должен остаться ровно один preferred candidate:

```text
V6-K5
или
ASEM
или
ALEX
```

`PREF-SUPPORT`, ASOFT и always-controls не входят в final candidate list этой версии.

# 59. Deployable final freeze после V4.4

Критически важно:

> исторический V6 OOF использует разные per-run веса и **не является одной deployable системой**.

Перед locked test нужно создать единственную runtime-конфигурацию.

## 59.1. Final V6 weights

Не выбирать новый vector по V4.4 EVAL.

Использовать уже зафиксированный V4.1 global freeze rule:

```text
V6_GLOBAL = (α=0.0, β=0.25, γ=2.0)
```

если он действительно остаётся действующим freeze-manifest проекта.

Если проект решает отказаться от этого global vector, новый способ получения V6-final weights должен быть оформлен **до INTERNAL_TEST** отдельным manifest rule без доступа к test.

Per-run weights запрещены в runtime final pipeline.

## 59.2. Final routing threshold

После выбора ASEM/ALEX вычислить один:

```text
tau70_global
```

на unlocked development predictions без test.

Рекомендуемый источник:

```text
OOF / cross-fitted ConsensusConf
только routing_available rows
```

а не in-sample predictions B0-final.

Threshold затем замораживается и применяется к INTERNAL_TEST без recalibration.

## 59.3. Final B0 seed policy

До final refit явно зафиксировать:

```text
FINAL_B0_SEED_POLICY
```

Нельзя выбрать «лучший seed» по OOF.

Если production предполагает один B0:
```text
выбрать один canonical seed заранее
```

Если предполагается ensemble из 3 seeds:
```text
это должно быть явно частью архитектуры
и вычислительной стоимости.
```

С учётом цели снижения ресурсов предпочтителен один заранее фиксированный seed, а три seeds использовать для development uncertainty, не как скрытый ensemble.

## 59.4. Final refit

После этого:

1. обучить B0-final на всём unlocked по frozen recipe;
2. применить frozen V6_GLOBAL;
3. frozen reader-only weights;
4. frozen `ConsensusConf`;
5. frozen `tau70_global`;
6. frozen seed policy;
7. записать hashes;
8. создать deployable freeze manifest.

Только после этого разрешён INTERNAL_TEST.

# 60. INTERNAL_TEST

После deployable freeze выполнить ровно один primary comparison:

```text
B0-final
vs
ONE frozen final pipeline
```

на 78 locked records.

Primary endpoint:

```text
Token F1
```

Secondary descriptive:
- EM;
- Hit@1;
- HN@1;
- Amb@1;
- GSRN@1.

Paired uncertainty:
```text
cluster bootstrap by document_id
```
если document_id structure доступна на locked split.

Запрещено после открытия test:
- менять threshold;
- менять weights;
- менять seed policy;
- менять architecture;
- менять K;
- переобучать по test result;
- повторно использовать test как tuning split.

# 61. Что не делать в V4.4

Не:

1. ретюнить V6;
2. искать новый semantic encoder;
3. менять K;
4. возвращать attention;
5. добавлять hidden states;
6. запускать U80;
7. route по aspect;
8. добавлять B0 margin в router;
9. подбирать Q50–Q90 по EVAL/F1;
10. открывать INTERNAL_TEST.

---

# 62. Рекомендуемый порядок запуска

```text
STEP 1
load exact V4.1/V4.3 references

STEP 2
freeze SEM-FIX and LEX-FIX

STEP 3
verify ConsensusConf implementation

STEP 4
for each outer run:
    compute non-EVAL unsupervised Q70
    on routing-available rows only

STEP 5
run ASEM cross-fitted development replay

STEP 6
development H1:
    ASEM vs V6

STEP 7
only if H1 pass:
    run ALEX substitution replay
    compare ALEX vs ASEM

STEP 8
confidence monotonicity
routing precision/recall
aspect transitions

STEP 9
optional local-support diagnostic only
(no final selection)

STEP 10
select exactly one preferred candidate:
    V6 / ASEM / ALEX

STEP 11
freeze deployable:
    V6_GLOBAL
    global tau70
    final B0 seed policy
    B0-final recipe
    hashes

STEP 12
only then open INTERNAL_TEST once
```

# 63. Короткая инструкция агенту

```text
Use exact V4.1/V4.3 development references.

Do not call V4.4 an independent confirmatory OOF.
It is a cross-fitted stability replay on reused unlocked data.

Primary fixed reader:
SEM-FIX =
    0.5 * Majority
    + 0.25 * SemRel

Secondary substitution:
LEX-FIX =
    0.25 * Majority
    + 2.0 * normalized Token-F1

Use the same best-triple ConsensusConf.

Routing available only when:
    >=4 valid answers
    reader-selected candidate belongs to T*.

For each development outer run:
    tau = Q70 of non-EVAL TRAIN+CAL
    routing-available ConsensusConf
    no gold.

Primary:
    ASEM = V6 ↔ SEM-FIX

Primary development comparison:
    ASEM vs V6.

Only if H1 passes:
    test ALEX = V6 ↔ LEX-FIX
    as substitution against ASEM.

Do not let support/ASOFT become post-hoc winners.

Before INTERNAL_TEST:
    replace per-run V6 weights with one deployable frozen V6 rule;
    freeze one global tau70;
    freeze B0 seed policy;
    train B0-final;
    hash everything.

Do not add:
    K10
    attention
    U80
    hidden states
    aspect routing
    new encoders.

Open INTERNAL_TEST exactly once after full freeze.
```

# 64. Главная исследовательская формулировка

> После выявления complementarity между relevance-aware V6 и reader-only SEMCORE проверить устойчивость заранее сформулированной adaptive architecture `V6 ↔ SEMCORE`, которая переключается на reader-only режим только при выраженном семантическом consensus среди B0 top-5; затем проверить, может ли lexical reader-only режим заменить semantic reader-only component без ухудшения качества и safety-профиля.

# 65. Критерий перехода к locked test

Переход разрешён только если:

```text
ONE preferred architecture selected
ONE deployable V6 weight vector frozen
ONE routing threshold frozen
ONE B0 seed policy frozen
B0-final recipe frozen
all hashes recorded
no unresolved primary model-selection branch remains
```

Если ASEM не проходит H1 development gate:

```text
preferred final candidate = V6-K5 architecture
```

но для locked test всё равно нужен **deployable global V6 configuration**, а не историческая смесь per-run OOF weights.

# 66. Следующая ветка при отрицательном результате

Если ASEM не показывает устойчивого development improvement:

```text
adaptive heuristic routing branch закрыть
и переходить к final freeze V6
```

если цель текущего цикла — завершить model selection.

Если вместо финализации исследования решено продолжать method development, следующий отдельный приоритет:

```text
corrected U80 reader-utility weak supervision
```

или:

```text
hidden-state / predicted-span representation
```

Но эти направления нельзя добавлять post-hoc к V4.4 и нельзя запускать после открытия INTERNAL_TEST.

