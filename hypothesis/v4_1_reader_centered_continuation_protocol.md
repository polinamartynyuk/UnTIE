# V4.1 — исправленный протокол продолжения экспериментов

## 0. Назначение

Этот документ задаёт следующий экспериментальный цикл после V4.

Цель:

> честно воспроизвести и развить два наиболее обоснованных сигнала V4 — reader-centered reranking (V6) и answer aggregation (V2-A4) — без ручной разметки, без генеративного teacher-а и без открытия `INTERNAL_TEST` до полного freeze.

Основная последовательность:

```text
R0  исправить evaluation infrastructure
↓
R1  честный 5-fold OOF для V6
↓
R2  диагностически разложить V6 на компоненты
↓
R3  проверить V6 + V2-A4
↓
R4  отдельно сравнить fixed K=5 и fixed K=10
↓
R5  вторично перепроверить reader-utility с правильной шкалой F1
↓
F   выбрать один pipeline, определить final hyperparameters, refit, freeze
↓
INTERNAL_TEST один раз
```

Primary-направления V4.1:

```text
V6
V6 + V2-A4
```

Secondary:

```text
corrected reader-utility
K=10 sensitivity
```

Не развивать как primary:

```text
KeyBERT LOSCO ranking
ownership probe
V7 natural-only
raw QA-score ranking
softmax-only calibration
QA-silver Stage A
manual TE/TM overlay
attention-only branch
старый SpanRanker
```

---

# 1. Что установлено V4

## 1.1. Исторический reference

`B0-live`:

```text
Token F1 = 27.36
Hit@1 = 36.72%
Hit@5 = 79.03%
Hit@10 = 92.20%
HN@1 = 4.86%
Amb@1 = 50.85%
GSRN@1 = 33.92%
```

Best-positive oracle:

```text
Token F1 ≈ 73.01
```

Главный резерв остаётся внутри shortlist:

```text
B0 уже часто находит positive в top-K,
но плохо ставит его на top-1.
```

## 1.2. V2-A4

V4:

```text
Token F1 = 30.26
ΔF1 vs B0 = +2.90
95% CI = [1.32, 4.55]
EM = 24.72%
```

V2-A4:
- не меняет состав/порядок B0 ranking;
- использует ответы frozen QA из top-5;
- выбирает final answer по сочетанию strongest evidence и поддержки из нескольких source regions.

## 1.3. V6

V4:

```text
Token F1 = 35.04
ΔF1 vs B0 = +7.69
95% CI = [4.79, 10.59]

Hit@1:
36.72% → 42.60%

HN@1:
4.86% → 5.15%

Amb@1:
50.85% → 44.30%

GSRN@1:
33.92% → 26.64%
```

На run-0 EVAL:

```text
ΔF1 = +6.68
95% CI = [0.75, 12.90]
```

Это самый сильный **перспективный** результат V4, но ещё не полноценный 5-fold OOF: веса V6 были выбраны только на одном CAL-фолде.

## 1.4. Отрицательные / незавершённые ветки

### KeyBERT LOSCO
Не дал устойчивого ranking gain.

### Ownership probe
```text
AUROC ≈ 0.543
pairwise ≈ 0.543
```
Near-random; anchored synthetic branch не развивать.

### V5
Реализация смешивала новые shortlist-scores со старыми B0 scores outside shortlist. Чистого доказательства global calibration нет.

### V7 natural-only
На OOF cal+eval:
```text
ΔF1 ≈ -5.96
95% CI ≈ [-11.61, -0.56]
```
Текущую реализацию закрыть.

### V3 utility
Нельзя интерпретировать из-за:
- in-sample evaluation;
- одного outer run;
- ошибки шкалы (`0.8` при F1 в шкале `0..100`).

---

# 2. Основная гипотеза V4.1

Рабочая гипотеза:

> при высоком recall B0 top-K относительное согласование ответов frozen reader между несколькими candidate passages является более полезным вторичным сигналом, чем абсолютная QA-confidence отдельного passage.

Схема:

```text
question
→ B0 top-K
→ frozen QA по каждому candidate
→ predicted answers
→ cross-candidate agreement/support
→ rerank passages
→ optional answer aggregation
```

---

# 3. Что именно считается новым сигналом

Не:

```text
QA raw score высокий
⇒ passage правильный
```

а:

```text
candidate answer
+
согласованность с ответами других сильных candidates
+
независимая source-region support
+
B0 relevance
```

Формально:

\[
S_i =
\alpha s_{B0,i}
+
\beta S_{agreement,i}
+
\gamma S_{support,i}
\]

V4.1 должен определить:
1. воспроизводится ли эффект;
2. какой компонент agreement реально полезен;
3. добавляет ли answer aggregation прирост поверх V6.

---

# 4. Общие ограничения

Во всех V4.1:

```text
INTERNAL_TEST закрыт;
source_label не переписывается;
production не меняется;
те же 5 document-level folds;
никаких новых outer splits;
gold не используется в runtime ranking;
ambiguous не становится strong negative;
B0-live остаётся historical OOF reference;
```

Unlocked pool:

```text
453 answerable document–aspect
240 documents
```

Seeds для основного three-seed дизайна:

```text
42
43
44
```

---

# 5. R0 — обязательное исправление evaluation infrastructure

Новые эксперименты не запускать до прохождения всех regression tests.

## 5.1. Pairing по seed

Для three-seed систем:

```text
candidate(question_id, seed)
vs
B0(question_id, same seed)
```

Нельзя индексировать baseline только по `question_id`.

Для single-seed exploratory system:

```text
candidate seed 42
vs
B0 seed 42
```

Single-seed system не запрещён автоматически, но:
- должен сравниваться только с matching seed;
- не должен молча конкурировать с three-seed pooled system как будто дизайн одинаков.

## 5.2. Clustered paired bootstrap

Primary uncertainty:

```text
paired clustered bootstrap
cluster = document_id
n_boot >= 5000
```

Для three-seed system:

```text
1. получить delta для каждого (question_id, seed);
2. усреднить delta по seeds внутри question;
3. bootstrap document_id.
```

Seeds не считать независимыми наблюдениями.

## 5.3. Canonical pairwise metric

Найти и переиспользовать **ровно ту canonical реализацию**, которая дала для V3-B0:

```text
pairwise_clean ≈ 0.7514
```

Regression:

```text
recompute(B0-live)
≈ historical V3-B0 pairwise_clean
```

с фиксированной tolerance.

Если canonical pairwise не восстановлен:

```text
pairwise_clean не использовать как hard safeguard
```

и временно опираться на:
```text
HN@1
Hit@1
Token F1
```

## 5.4. Shortlist isolation

Для reranking top-K:

```text
allowed candidates = exact B0 top-K
```

Outside-top-K:
```text
вообще не участвуют в argmax.
```

Invariant:

```text
pure top-5 rerank
⇒
candidate set unchanged
⇒
Hit@5 identical B0
```

## 5.5. Явно зафиксировать шкалы F1

Это обязательная защита от повторения ошибки V3 utility.

### Utility / QA evaluation

По V4 известно, что используемый при utility mining `token_f1()` возвращал:

```text
0..100
```

Поэтому для R5:

```text
exact match → 100
U80 → threshold 80
```

### V6 reader-agreement

**Не предполагать заранее**, что V6 agreement использовал `0..1`.

Сначала прочитать фактическую V4 implementation и определить:

```text
mean_token_f1 range
exact_frequency range
source_support range
```

Затем записать их в:

```text
v6_feature_contract.json
```

Primary R1 обязан сохранить **ровно V4 scale**, какой бы она ни была:

```text
0..1
или
0..100
или count-based
```

Unit tests должны проверять не желаемую шкалу, а совпадение с frozen V4 feature values на контрольных examples.

Только после точного R1 reproduction в R2 разрешена отдельная sensitivity-ablation:

```text
normalized agreement features ∈ [0,1]
```

как новая экспериментальная модификация.

Запрещено молча нормализовать V6 features до воспроизведения старого результата.

## 5.6. Зафиксировать feature contract V6

До R1 создать:

```text
v6_feature_contract.json
```

Для каждого feature записать:
```text
name
definition
range
normalization
V4 implementation location
```

Обязательно:

```text
b0_score
b0_rank
exact_frequency
mean_token_f1
majority_flag
source_cluster_support
```

Для каждого сохранить:

```text
definition
observed range
normalization
dtype
```

Не менять определения или шкалы между folds.

## 5.7. Role-aware reporting

Для trainable branches отчёт обязан отдельно показывать:

```text
TRAIN
CAL
EVAL
CAL+EVAL
ALL
```

`ALL` не использовать для selection, если содержит TRAIN predictions модели.

Автоматический flag:

```text
IN_SAMPLE_CONTAMINATED = true
```

если candidate scored собственные training documents.

## 5.8. Selection comparability

`selection.json` должен проверять:
- evaluation role;
- seed design;
- matching units;
- candidate count;
- shortlist invariants;
- leakage flags.

Не запрещать систему только потому, что `n=453` вместо `1359`; single-seed comparison допустим, если она явно matched к одному B0 seed. Но такая система не может автоматически вытеснить three-seed primary без отдельного правила.

---

# 6. R0 regression suite

До дальнейшего запуска должны пройти:

## 6.1. B0 identity

```text
Token F1
Hit@1
Hit@5
Hit@10
HN@1
Amb@1
GSRN@1
```

совпадают с historical V3-B0.

## 6.2. Identity comparison

Копия B0:

```text
ΔF1 = 0
wins = 0
losses = 0
ties = 1359
```

## 6.3. Pairwise identity

```text
pairwise_clean(B0-live)
≈ 0.7514
```

если canonical implementation восстановлен.

## 6.4. Shortlist invariant

```text
set(reranked_chunk_ids)
==
set(B0_topK_chunk_ids)
```

для каждого question/seed.

## 6.5. V4 V6 reproduction check

До 5-fold запуска на старом run-0 CAL/EVAL должна быть воспроизведена V4 V6 implementation при её старых frozen weights:

```text
α = 2.0
β = 0.25
γ = 0.5
```

с той же feature contract.

Допуск задать заранее, например:
```text
|ΔTokenF1| < 1e-6
```
если используются те же cached outputs.

Если старый V4 результат не воспроизводится, R1 не запускать.

---

# 7. Cache policy

Не пересчитывать дорогие outputs без необходимости.

Разрешено переиспользовать V4 cache, если совпадают:

```text
dataset hash
split hash
B0 prediction hash
QA checkpoint hash
answer normalization version
chunk offsets
```

Иначе пересчитать.

Cache key:

```text
(question_id, chunk_id, qa_model_hash, preprocessing_hash)
```

Для K=5 primary достаточно exact union top-5 по трём B0 seeds.

---

# 8. R1 — честный 5-fold OOF V6

Главный эксперимент V4.1.

## 8.1. Outer roles

Использовать существующую V3 rotation.

Для каждого:

```text
run r = 0..4

TRAIN = 3 folds
CAL   = 1 fold
EVAL  = 1 fold
```

V6 non-trainable; TRAIN нужен только как часть общего split contract. Выбор весов выполняется исключительно на CAL.

## 8.2. Primary K

```text
K = 5
```

K=10 не использовать в R1.

## 8.3. Frozen QA outputs

Для каждого B0 top-5 candidate:

```text
predicted_answer
normalized_answer
start
end
qa_raw_score
qa_margin
```

Frozen QA:

```text
eval mode
dropout off
same checkpoint
```

## 8.4. Source-overlap clusters

Использовать runtime source-overlap clustering только по offsets candidate chunks.

Primary threshold:

```text
ровно тот же overlap threshold, что V4
```

Если V4 использовал `0.6`, зафиксировать `0.6`.

Не использовать gold-positive D5 clusters как runtime feature.

---

# 9. R1 — точное определение V6 features

Первый R1 должен **в точности повторять V4 feature transformations**.

Не переопределять их по памяти.

Обязательно сверить с кодом V4 и записать в `v6_feature_contract.json`.

Conceptually:

```text
S_answer =
exact_frequency
+
mean_token_f1_unit
+
majority_flag
```

и:

```text
S_support =
source-cluster support
для agreeing answers
```

Но фактические детали normalization должны быть взяты из V4 implementation и frozen до запуска.

Формула:

\[
S_i =
\alpha s_{B0,i}
+
\beta S_{answer,i}
+
\gamma S_{support,i}
\]

---

# 10. R1 — weight selection

## 10.1. Grid

Использовать **ровно predeclared grid** из `contract.json`.

Предпочтительно начать с той же `FUSION_LAMBDAS`, которая использовалась V4.

Не расширять grid после просмотра outer EVAL.

## 10.2. Selection unit

Для каждого candidate weight triple на CAL:

1. посчитать результат отдельно для seeds 42/43/44;
2. получить mean Token F1 across seeds per question;
3. затем mean CAL Token F1.

Не считать 3 seed rows независимыми CAL examples.

## 10.3. Constrained selection

Сначала отфильтровать configurations, нарушающие CAL safeguard:

```text
HN@1_candidate
<=
HN@1_B0 + epsilon_HN
```

Primary:

```text
epsilon_HN = 2 percentage points
```

Из допустимых выбрать max:

```text
mean CAL Token F1
```

Tie-break:
1. lower HN@1;
2. higher Hit@1;
3. lower reader-feature contribution (`β + γ`);
4. deterministic lexicographic order.

Это уменьшает гиперпараметрическую свободу.

## 10.4. One weight triple per outer run

Primary:

```text
один (α,β,γ) на run
для всех трёх B0 seeds
```

Per-seed weights не использовать в primary.

---

# 11. R1 — EVAL

Для каждого run:

```text
CAL → select weights
freeze
EVAL → apply once
```

Сохранить:

```text
run{r}/selected_weights.json
run{r}/eval_predictions_seed42.jsonl
run{r}/eval_predictions_seed43.jsonl
run{r}/eval_predictions_seed44.jsonl
```

После 5 runs объединить только EVAL:

```text
true OOF V6
453 questions × 3 seeds = 1359 rows
```

---

# 12. R1 metrics

Primary:

```text
Token F1
```

Secondary:

```text
EM
Char F1
ROUGE-L
Hit@1
MRR
HN@1
Amb@1
GSRN@1
easy@1
F1 | selected original positive
GoldMatchFromNonOriginalPositive
```

Invariant:

```text
Hit@5 V6 == Hit@5 B0
```

если composition top-5 не меняется.

---

# 13. R1 transition analysis

B0 → V6:

```text
positive → positive
positive → ambiguous
positive → HN
positive → easy

ambiguous → positive
GSRN → positive
HN → positive
easy → positive
```

Главные:

```text
GSRN → positive
ambiguous → positive
positive → HN
positive → ambiguous
```

Оценивать:
```text
net_positive_gain
=
(ambiguous→positive + HN→positive + easy→positive)
-
(positive→ambiguous + positive→HN + positive→easy)
```

Отдельно считать GSRN transitions.

---

# 14. R1 statistical comparison

Primary comparison:

```text
V6 true OOF
vs
B0 matching OOF
```

Paired clustered bootstrap:

```text
cluster = document_id
>= 5000
```

Report:

```text
ΔToken F1
95% CI
ΔEM
ΔHit@1
ΔHN@1
per-fold deltas
```

Per-fold direction — diagnostic, не отдельный hard gate.

Не требовать автоматически `4/5 folds positive`: при ~90 questions/fold это слишком шумный критерий.

---

# 15. Gate R1

V6 считается подтверждённым model-selection candidate, если:

```text
ΔToken F1 > 0
95% CI excludes 0
HN safeguard passed
Hit@1 > B0
shortlist invariants passed
```

Если CI включает 0:

```text
V6 остаётся exploratory
```

и primary V4.1 pipeline не открывает INTERNAL_TEST на основании V6.

---

# 16. R2 — диагностическое разложение V6

Цель:

> понять источник V6 gain, а не провести новую широкую model search.

Поэтому `full V6` остаётся primary architecture.

Ablations — diagnostics.

## 16.1. Frozen-weight knockout design

Для каждого outer run взять **уже выбранные R1 weights** и не переоптимизировать их для каждой ablation.

Так измеряется вклад feature block без дополнительного tuning.

### A0
```text
B0
```

### A1
```text
full V6 minus exact-frequency contribution
```

### A2
```text
full V6 minus Token-F1 agreement contribution
```

### A3
```text
full V6 minus majority contribution
```

### A4
```text
full V6 minus source-cluster support
γ = 0
```

### A5
```text
B0 + answer-agreement block
без source support
```

### A6
```text
B0 + source support
без answer-agreement block
```

### A7
```text
reader features only
α = 0
```

Diagnostic only.

## 16.2. Optional Char-F1

Char-F1 agreement не входил в исходный V6.

Поэтому:

```text
Char-F1 feature = exploratory extension
```

Не включать его в primary R1.

Если добавляется, пометить:
```text
R2-exploratory
```
и не заменять full V6 как final architecture в этом же протоколе без нового freeze decision rule.

---

# 17. R2 feature scales

Primary R1 scale:

```text
exactly V4 scale from v6_feature_contract.json
```

Не задавать её по предположению.

Для R2 разрешено диагностически проверить:

```text
normalized exact-frequency
normalized Token-F1 agreement
normalized support
```

но только как **явно новую ablation**.

Если V4 использовал `Token F1 0..100`, а normalized R2 использует `0..1`:

```text
это две разные системы;
их результаты нельзя смешивать.
```

Любая normalization должна быть deterministic и не зависеть от EVAL labels.

---

# 18. R2 diagnostics by source label and aspect

Сравнить distribution:

```text
original positive
GSRN
other ambiguous
HN
easy
```

для:

```text
exact frequency
mean Token-F1 agreement
majority
source support
```

По аспектам:

```text
task
method
material
metric
```

Особенно проверить:
- почему V4 дал большой gain на task;
- сохраняется ли gain на method;
- остаётся ли metric слабым.

Ablation results не использовать как confirmatory p-value family.

---

# 19. R3 — V6 + V2-A4

Запускать после подтверждения R1.

Цель:

> проверить, добавляет ли answer-level evidence aggregation пользу поверх уже улучшенного passage ranking.

## 19.1. Обязательный V2 control regression

Сначала на B0 воспроизвести V4 V2-A4:

```text
Token F1 ≈ 30.26
```

Использовать:
- точную V4 answer normalization;
- точную V4 source-cluster rule;
- точную V4 aggregation coefficient / formula.

Не восстанавливать λ по памяти: прочитать из V4 config/code и записать в:

```text
v2_a4_contract.json
```

## 19.2. Почему нельзя использовать raw V6 score с λ из V2-A4

Raw `V6 score` имеет другую шкалу, чем raw `B0 score`:

\[
s_{V6}
=
\alpha s_{B0}
+
\beta S_{agreement}
+
\gamma S_{support}
\]

Поэтому запрещено:

```text
max raw V6 score
+
старый λ_V4 * support
```

без новой калибровки масштаба.

Иначе relative contribution support определяется произвольной шкалой V6.

## 19.3. Primary combination C1

Сначала внутри каждого question/seed нормировать V6 scores **только внутри top-K**.

Primary normalization:

\[
p_{V6}(c)
=
\frac{\exp(s_{V6}(c))}
{\sum_{j\in topK}\exp(s_{V6}(c_j))}
\]

Softmax здесь используется **не для reranking** — порядок не меняется — а только чтобы получить сопоставимый within-question evidence weight для answer aggregation.

Answer score:

\[
S_{C1}(a)
=
\max_{c\in C_a} p_{V6}(c)
+
\lambda_{agg}\log(1+N_{clusters}(a))
\]

## 19.4. λ selection

Для каждого outer run:

1. V6 weights уже frozen из R1;
2. на CAL выбрать только `λ_agg`;
3. применить frozen `λ_agg` на EVAL.

Grid:

```text
LAMBDA_AGG_GRID
```

записать в `contract.json` до первого EVAL.

Предпочтительно использовать ту же небольшую сетку/диапазон, что применялись для V2 aggregation, если она сохранена в V4 code/config.

Не использовать EVAL для изменения grid.

## 19.5. Mandatory controls

### C0 — exact V2-A4 reproduction

```text
old B0-based V2-A4
```

должен воспроизвести V4.

### C1 — primary

```text
softmax-normalized V6 strength
+
source-cluster support
```

### C2 — secondary diagnostic

\[
S(a)=
\sum_{c\in C_a}p_{V6}(c)
\]

### C3 — secondary diagnostic

```text
C1 + exact answer frequency
```

C2/C3 не заменяют C1 как primary в этом протоколе.

---

# 20. R3 evaluation

В каждом outer run:

```text
использовать V6 weights, frozen в R1 для этого run
```

На CAL разрешено выбрать только:

```text
λ_agg
```

из заранее frozen grid.

Никакого повторного выбора V6 weights под aggregation.

Final object = answer.

Primary:

```text
Token F1
EM
```

Secondary:

```text
Char F1
ROUGE-L
final_answer_has_original_positive_support
GoldMatchFromNonOriginalPositive
n_source_clusters
support_label_mix
```

Сохранять V6 top1 chunk отдельно для provenance.

---

# 21. Gate R3

Primary comparison:

```text
C1 vs V6
```

Если:

```text
ΔToken F1 > 0
и 95% CI excludes 0
```

C1 становится более сильным final candidate.

Если CI включает 0:
- оценить magnitude/runtime;
- если gain мал, предпочесть V6 как более простой;
- не объявлять C1 подтверждённым improvement.

---

# 22. R4 — fixed K=5 vs fixed K=10

Не использовать per-run adaptive K как primary: это смешало бы две системы.

Нужно построить **две отдельные OOF системы**.

## 22.1. K=5 system

Для каждого outer run:
```text
tune V6 weights on CAL using K=5
apply to EVAL K=5
```

Собрать:
```text
OOF_K5
```

## 22.2. K=10 system

Независимо:

```text
tune V6 weights on CAL using K=10
apply to EVAL K=10
```

Собрать:
```text
OOF_K10
```

Если final architecture после R3 = C1:
повторить C1 при K=5 и K=10 с той же frozen aggregation rule.

## 22.3. Primary comparison

```text
fixed K=10 OOF
vs
fixed K=5 OOF
```

Paired по тем же questions/seeds.

Не выбирать K отдельно для каждого outer run в final system.

## 22.4. Strict K sensitivity diagnostic

Дополнительно можно применить K=5-selected weights к K=10 без retuning.

Это показывает эффект самого расширения shortlist.

Diagnostic only.

---

# 23. R4 efficiency

Не писать автоматически:

```text
K10 = ровно 2× cost
```

Считать фактически:

```text
QA calls / question
unique cached calls
tokens
latency
GPU peak
RAM peak
```

На live inference без межзапросного cache reader calls примерно растут с K, но экспериментальный union-cache может давать другую стоимость.

K=10 выбирать для final pipeline только при practically meaningful gain.

---

# 24. R5 — corrected reader-utility branch

Secondary branch.

Не должна блокировать final V6/C1, если до freeze формально исключена из final selection.

## 24.1. Utility F1 scale

Использовать:

```text
token_f1_pct ∈ [0,100]
```

Primary threshold:

```text
U80: F1 >= 80
```

Sensitivity diagnostics:

```text
U50: F1 >= 50
U100: exact/near-exact
```

Для U100 не сравнивать float `== 100` без tolerance.

Использовать:

```text
EM == 1
```

или:
```text
Token F1 >= 100 - eps
```

Primary остаётся **U80**; U50/U100 не выбирать post-hoc по EVAL.

## 24.2. Mining

Только outer TRAIN:

```text
ambiguous chunks
→ frozen QA
→ compare prediction with TRAIN gold
→ utility candidates
```

CAL/EVAL gold не использовать для mining.

## 24.3. New source-region requirement

Primary utility candidate:

```text
does not overlap original-positive source cluster
```

по frozen overlap rule.

Overlapping candidates:
```text
separate diagnostic only
```

## 24.4. Weight

Primary grid:

```text
η ∈ {0.25, 0.5, 0.75}
```

выбирать на CAL.

Utility threshold не тюнить:
```text
U80 fixed primary
```

## 24.5. Objective

\[
L_q =
-\log
\frac{
\sum_{p\in P}\exp(s_p/T)
+
\eta\sum_{u\in U}\exp(s_u/T)
}{
\sum_{p\in P}\exp(s_p/T)
+
\eta\sum_{u\in U}\exp(s_u/T)
+
\sum_{n\in N}\exp(s_n/T)
}
\]

где:

```text
P = original positives
U = U80 utility candidates
N = training-eligible HN + limited easy
other ambiguous = masked
```

Temperature/loss details использовать из заранее зафиксированной B3/V3 recipe; не подбирать заново по EVAL.

---

# 25. R5 seed policy

Предпочтительный secondary дизайн:

```text
seeds = 42, 43, 44
```

для trainable utility ranker.

Если вычислительно выполняется только seed 42:

```text
R5 status = exploratory single-seed
```

и сравнение только:

```text
R5 seed42
vs
B0 seed42
```

Такая ветка не может автоматически стать final winner над three-seed primary без нового решения.

---

# 26. R5 true OOF

Для каждого outer run:

```text
TRAIN:
mine U80 + train model

CAL:
select η / early stop

EVAL:
score once
```

Объединять только EVAL.

Запрещено:
```text
ALL=TRAIN+CAL+EVAL
```
для selection.

Gate:

```text
true OOF ΔF1 > 0
```

с paired CI.

Даже при успехе термин:

```text
reader-utility weak supervision
```

не:
```text
automatic target evidence.
```

---

# 27. Hyperparameter contract

До первого outer EVAL записать:

```text
contract.json
```

Содержит:

```text
V6 weight grid
epsilon_HN
answer normalization
V6 feature scale exactly as V4
optional normalized-R2 feature version
source overlap threshold
V2-A4 aggregation contract
K candidates = {5,10}
utility threshold = 80
ETA grid
seed policy
bootstrap procedure
selection rule
```

После первого EVAL:
```text
никаких расширений grid.
```

---

# 28. Model-selection hierarchy

Primary endpoint:

```text
Token F1
```

Secondary:

```text
EM
Hit@1
HN@1
runtime
```

Правило для unlocked OOF:

1. valid evaluation design;
2. leakage-free;
3. shortlist invariants;
4. HN safeguard;
5. paired ΔToken F1;
6. CI;
7. Pareto F1/runtime;
8. при статистически/практически неразличимых результатах — более простой pipeline.

OOF — model-selection evidence, не independent confirmatory test.

---

# 29. Final architecture selection

К final freeze допускаются только заранее определённые primary candidates:

```text
B0
V6 K5
C1(V6+V2) K5
V6 K10
C1(V6+V2) K10
```

R2 ablations и R5 exploratory variants не входят автоматически в этот список.

Если R5 unexpectedly лучше:
```text
зафиксировать как новый candidate только через явный protocol amendment до INTERNAL_TEST.
```

---

# 30. Как определить final hyperparameters ДО INTERNAL_TEST

Это обязательная часть, которой не хватало в предыдущей версии.

После выбора final **architecture** по OOF:

## 30.1. Final V6 weights

Не выбирать по INTERNAL_TEST.

Выбрать один global weight triple:

```text
argmax mean CAL Token F1
across all 5 outer runs
```

из уже frozen grid, с HN safeguard.

То есть используются только CAL predictions unlocked pool.

Tie-break тот же, что в R1.

## 30.2. Final aggregation coefficient

Если final = C1:

выбрать один global `λ_agg`:

```text
argmax mean CAL Token F1
across all 5 outer runs
```

из frozen `LAMBDA_AGG_GRID`, при уже frozen final V6 weight-selection procedure.

Не переносить raw coefficient V2-A4 на raw V6 score.

Final C1 использует:

```text
within-topK normalized V6 evidence weight
+
global λ_agg
```

## 30.3. Final K

Выбрать:

```text
K=5 или K=10
```

по paired OOF comparison R4.

Это model selection на unlocked pool и выполняется до test.

## 30.4. Final B0 refit

`B0-live` — OOF predictions, а не final test model.

Перед INTERNAL_TEST нужно создать:

```text
B0-final
```

обученный на всех unlocked training documents по **замороженной B0 recipe**.

Epoch policy нельзя определять на test.

Если final training требует фиксированного числа epochs:
- использовать заранее записанное правило из V3/V4.1 fold training reports;
- например median/mode best epoch rule;
- само правило записать в freeze manifest ДО test.

Не изобретать epoch policy после открытия INTERNAL_TEST.

## 30.5. Final seed policy

Предпочтительно final refit:

```text
seeds 42, 43, 44
```

и на INTERNAL_TEST:
- прогнать все три frozen fits один раз;
- report per-seed;
- report mean across seeds.

Не усреднять logits между seeds как ensemble, если ensemble не был заранее объявлен.

---

# 31. Final freeze manifest

До открытия INTERNAL_TEST сохранить:

```text
dataset hash
split hash
B0 recipe hash
B0 final epoch rule
B0 seeds

QA checkpoint hash
answer normalization version
source-overlap rule

final architecture
final K
final V6 weights
V6 feature contract

V2-A4 aggregation contract, если используется

utility branch status:
included / excluded

bootstrap procedure
primary endpoint
HN safeguard
runtime measurement rule
```

После создания manifest:
```text
no config changes.
```

---

# 32. Когда можно открыть INTERNAL_TEST

Только если:

- [ ] R0 regression suite passed;
- [ ] R1 true V6 OOF finished;
- [ ] R2 diagnostics finished или формально skipped;
- [ ] R3 checked, если V6 подтверждён;
- [ ] R4 K decision finished либо K10 formally excluded;
- [ ] R5 completed или **до test** formally excluded from final selection;
- [ ] one final architecture selected;
- [ ] final hyperparameters determined;
- [ ] B0-final refit policy frozen;
- [ ] final models trained;
- [ ] freeze manifest written.

После этого:

```text
INTERNAL_TEST
→ single locked evaluation event
```

Никакого:
```text
test → tweak → test again.
```

---

# 33. Confirmatory INTERNAL_TEST design

Сравнить:

```text
B0-final
vs
ONE frozen V4.1-final pipeline
```

на одинаковых final seeds.

Primary:

```text
Token F1
```

Secondary:

```text
EM
Hit@1
HN@1
Amb@1
GSRN@1
runtime
```

Report:
```text
paired delta
95% CI
per-seed values
mean across seeds
```

Не требовать обязательно, чтобы небольшой locked test дал CI, исключающий 0: статистическая мощность может быть ограниченной.

Сильнейшая интерпретация:
```text
OOF significant positive
+
locked-test effect same direction and practically consistent.
```

Если locked-test CI также исключает 0:
```text
это дополнительное усиление вывода.
```

---

# 34. Aspect analysis

Для каждого OOF primary candidate:

```text
task
method
material
metric
```

Обязательно:

```text
Token F1
Hit@1
HN@1
Amb@1
GSRN@1
```

Вопросы:

## task
Воспроизводится ли крупный V6 gain?

## method
Сохраняется ли improvement ranking/GSRN?

## material
Есть ли дополнительная польза aggregation?

## metric
Остаётся ли самым слабым после V6/C1?

Если metric остаётся слабым:
```text
выделить как отдельную следующую исследовательскую проблему,
а не подстраивать текущий V4.1 post-hoc.
```

---

# 35. Efficiency report

Сравнить:

```text
B0-final/reference
V6 K5
C1 K5
V6 K10
C1 K10
```

Metrics:

```text
QA calls / question
unique QA calls
tokens processed
latency
GPU peak
RAM peak
cache hit rate
```

Отдельно:

```text
offline experimental cache cost
vs
live inference cost
```

Не считать cached QA бесплатным в production interpretation.

---

# 36. Stop rules

## Stop R1
Если:
```text
ΔF1 <= 0
```
или HN safeguard явно нарушен:
не развивать V6 как primary.

Если ΔF1 > 0, но CI включает 0:
V6 exploratory; INTERNAL_TEST не открывать на нём как подтверждённом winner без отдельного решения.

## Stop R3
Если C1 не лучше V6 practically/statistically:
оставить V6.

## Stop K10
Если gain мал относительно дополнительного reader cost:
оставить K5.

## Stop R5
Если true OOF <= B0:
закрыть utility branch.

---

# 37. Что НЕ делать

1. Не доверять старому V4 `selection.json`.
2. Не использовать `ALL` для trainable winner selection.
3. Не сравнивать three-seed candidate с last B0 seed.
4. Не считать seeds независимыми samples.
5. Не смешивать shortlist new scores и outside B0 scores.
6. Не использовать pairwise V4 scale ~0.41 как historical 0.75.
7. Не смешивать F1 scales `[0,1]` и `[0,100]`.
8. Не тюнить reader-agreement ablations по outer EVAL.
9. Не выбирать K отдельно по каждому EVAL fold как final adaptive system.
10. Не использовать gold-positive D5 clusters как runtime clusters.
11. Не считать reader-utility candidate истинным target evidence.
12. Не открывать INTERNAL_TEST до freeze.
13. Не использовать INTERNAL_TEST для epoch/weight/K selection.
14. Не объявлять 3-seed evaluation ensemble, если logits не объединяются.
15. Не менять final config после первого test evaluation.

---

# 38. Рекомендуемая структура артефактов

```text
experiments/qa_chunk_selection/v4_1/

artifacts/qa_chunk_selection_v4_1/
    contract.json

    regression/
        b0_identity.json
        paired_identity.json
        pairwise_identity.json
        v6_run0_reproduction.json
        shortlist_invariants.json
        feature_scale_tests.json

    cache/
        qa_top5/
        qa_top10/
        normalized_answers/
        source_clusters/

    r1_v6_oof/
        run0/
        run1/
        run2/
        run3/
        run4/
        oof_predictions.jsonl
        metrics.json
        bootstrap.json
        transitions.json

    r2_v6_ablation/
        knockout_results.json
        feature_distributions.json
        aspect_diagnostics.json

    r3_v6_v2/
        control_v2_a4.json
        run0/
        ...
        oof_predictions.jsonl
        metrics.json
        bootstrap.json

    r4_k/
        k5/
        k10/
        paired_comparison.json
        efficiency.json

    r5_utility/
        run0/
        ...
        utility_candidates.jsonl
        training_reports/
        oof_predictions.jsonl
        metrics.json

    final/
        model_selection.json
        hyperparameter_selection.json
        b0_final_training_manifest.json
        freeze_manifest.json
```

---

# 39. Experiment IDs

```text
V4.1-R0  evaluation regression
V4.1-R1  V6 honest 5-fold OOF
V4.1-R2  V6 frozen-weight ablations
V4.1-R3  V6 + V2-A4
V4.1-R4  fixed K5 vs fixed K10
V4.1-R5  corrected U80 reader-utility
V4.1-F   final selection/refit/freeze
V4.1-T   one INTERNAL_TEST event
```

---

# 40. Короткая инструкция агенту

```text
1. DO NOT trust V4 selection.json.

2. Fix evaluation first:
   - pair by (question_id, seed)
   - bootstrap by document_id
   - restore canonical pairwise
   - isolate exact B0 top-K
   - fix utility F1 as [0,100]
   - discover and freeze actual V4 V6 agreement scale
   - reproduce old V4 V6 on run0 before any normalization

3. R1:
   for each of 5 outer runs:
       select ONE V6 weight triple on CAL across 3 seeds
       enforce HN safeguard
       freeze
       score EVAL only
   merge 5 EVAL folds → true OOF

4. R2:
   do frozen-weight knockout ablations;
   DO NOT turn them into a new winner search.

5. R3:
   first reproduce V2-A4 exactly;
   then replace max B0 score by max V6 score;
   keep original V2-A4 aggregation coefficient in primary C1.

6. R4:
   build separate OOF_K5 and OOF_K10;
   compare fixed systems;
   DO NOT choose K independently per outer EVAL.

7. R5:
   U80 means F1 >= 80 on 0..100 scale;
   mine utility only on TRAIN;
   3 seeds preferred;
   evaluate EVAL only.

8. Before test:
   choose one architecture;
   choose final V6 weights by mean CAL across 5 runs;
   choose K by OOF;
   train B0-final on all unlocked data using frozen epoch rule;
   write freeze manifest.

9. Open INTERNAL_TEST once:
   B0-final vs ONE final V4.1 pipeline.
```

---

# 41. Ожидаемый итог V4.1

Если гипотеза подтверждается, итоговый ресурсно-эффективный pipeline:

```text
UNSTRUCTURED DOCUMENT
↓
existing chunking
↓
B0 answerability ranking
↓
top-K
↓
frozen extractive QA per candidate
↓
reader-centered cross-candidate agreement
↓
V6 reranking
↓
optional V2-A4-style answer aggregation
↓
FINAL EXTRACTIVE ANSWER
```

Без:

```text
ручной target-mismatch разметки
генеративной teacher-модели
document section labels
full-document LLM inference
```

Главный научный вопрос V4.1:

> переносится ли обнаруженный в V4 reader-centered signal на честный 5-fold OOF и даёт ли evidence aggregation дополнительный прирост поверх него?
