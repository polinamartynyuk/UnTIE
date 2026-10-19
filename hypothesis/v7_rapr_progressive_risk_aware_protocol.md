# V7 / RAPR — Risk-Aware Progressive Residual Reranking
## Полный план реализации следующего эксперимента для ИИ-агента

**Статус:** новый development-протокол после V4.4 → V4.5 → V5 → V6  
**Назначение:** техническая инструкция для ИИ-агента, который должен написать код, провести эксперимент, проверить корректность и сформировать воспроизводимые артефакты  
**Основная цель:** проверить единый параметризуемый метод безопасной последовательной коррекции выбора evidence-фрагмента, который оценивает **ожидаемую величину downstream-полезности**, отдельно контролирует риск ухудшения и запускает QA только для ограниченного числа наиболее перспективных альтернатив.

---

# 0. Главный научный вопрос

V6 показал, что проблема больше не сводится к отсутствию ещё одного признака.

Ключевая проблема:

```text
мы недостаточно хорошо оцениваем
risk/reward конкретного switch
```

V7 проверяет следующий принцип:

\[
\boxed{
\text{switch должен выполняться только тогда, когда}
\; E[\Delta F1 \mid x] > 0
\; \text{и риск вредного switch достаточно мал}
}
\]

при этом:

```text
не нужно запускать QA на всех top-K chunks
```

Если полезная альтернатива может быть найдена после проверки 1–2 дополнительных кандидатов, обработка должна завершаться раньше.

---

# 1. Почему это новая версия, а не post-hoc исправление V6

V6 завершён со статусом:

```text
NO_UNIVERSAL_CANDIDATE
```

Поэтому запрещено:

- исправлять V6 results;
- изменять V6 contract;
- добавлять новый вариант в старый V6 run;
- объявлять `S4-no-aspect`, `S4-lex` или `LOCAL S4` победителем V6 задним числом.

V7 является новым experiment version:

```text
новая гипотеза
новый frozen contract
новый run namespace
новая development evaluation
```

---

# 2. Что именно следует из V6 и используется как основание V7

V6 дал пять наиболее важных результатов.

## 2.1. Binary positive-utility target недостаточен

V6 S2/S3 моделировали в основном:

\[
P(u_i>0)
\]

где:

\[
u_i = F1(A_i,G)-F1(A_0,G)
\]

Но `+0.1 F1` и `+30 F1` оба являются одним классом `1`, а `-0.1 F1` и `-40 F1` — классом `0`.

V7 должен учитывать magnitude.

## 2.2. Gate полезен как safety, но не делает selector сильнее

V6 gate часто:

```text
route=false → preserve B0
```

Это снижает вред, но иногда просто превращает second stage в B0.

Следовательно, необходимо раздельно моделировать:

```text
кандидат заслуживает QA?
```

и:

```text
после QA безопасно ли на него переключаться?
```

## 2.3. LEX — наиболее перспективный дополнительный signal

По V6 LEX-only был полезнее source в большинстве relevant cells.

Поэтому:

```text
LEX входит в primary V7
source НЕ входит в primary V7
```

## 2.4. Aspect interactions убрать из primary

V6 `S4-no-aspect` оказался лучше full pooled S4 на 4/5 cells.

Поэтому primary V7:

```text
NO aspect interactions
NO aspect-specific models
NO aspect-specific thresholds
```

Aspect используется только в post-hoc diagnostic.

## 2.5. Full top-K слишком дорог

V6 gated pipeline всё ещё имел высокий P95 на EN.

Поэтому V7 вводит hard runtime cap:

```text
QA top1 + не более 2 дополнительных candidates
```

То есть:

```text
MAX_QA_PER_QUESTION = 3
```

в primary.

---

# 3. Рабочее название метода

```text
RAPR
Risk-Aware Progressive Residual Reranking
```

Русская формулировка:

> риск-адаптивное последовательное остаточное ранжирование.

---

# 4. Архитектура V7

```text
DOCUMENT
   ↓
historical B0 ranker
   ↓
top-K candidates
   ↓
pre-QA candidate-admission model
   ↓
rank alternatives by expected chance of positive utility
   ↓
QA(top1)
   ↓
candidate queue, max 2 alternatives
   ↓
for candidate in queue:
      QA(candidate)
      ↓
      Q-A relevance
      pairwise lexical agreement
      ↓
      expected-utility regression
      harm-risk classifier
      ↓
      if safe positive switch:
          select candidate
          STOP
      else:
          continue
   ↓
if no candidate accepted:
      preserve B0 top1
```

---

# 5. Универсальность V7

Primary claim:

```text
одна архитектура
одни признаки
одни target definitions
одна training procedure
одна progressive policy
```

Но meta-model coefficients могут обучаться отдельно на TRAIN части конкретного corpus.

Это:

```text
architectural / parameterized universality
```

Primary mode:

```text
LOCAL
```

Потому что V6 показал, что LOCAL оказался устойчивее POOLED как development-режим.

---

# 6. POOLED в V7

POOLED не является primary winner mode.

Он остаётся:

```text
secondary transfer diagnostic
```

и использует ту же feature schema **без aspect interactions**.

Он проверяет:

> переносится ли упрощённая V7 representation лучше, чем V6 POOLED.

---

# 7. LOCO-meta

LOCO-meta остаётся secondary diagnostic.

Не считать его independent confirmation.

Upstream models всё ещё могут быть corpus-specific.

---

# 8. Dataset cells

Использовать те же development cells:

```text
EN:
- SciREX
- SciER 384/50
- SciER 128/24

RU:
- RuSERRC
- SciMDIX
```

SciER cells:

```text
one corpus family
```

---

# 9. Development status

Все corpora уже повлияли на:

```text
feature engineering
method design
threshold logic
hypothesis generation
```

Следовательно:

```text
V7-A = development
```

После freeze нужен:

```text
V7-B untouched confirmation
```

---

# 10. Historical protection — ОБЯЗАТЕЛЬНО

Read-only:

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
```

Запрещено изменять:

```text
historical artifacts
historical caches
historical model checkpoints
historical manifests
historical reports
historical configs
historical frozen contracts
historical source code used for reproduction
```

---

# 11. New namespace

Создать:

```text
experiments/qa_chunk_selection_v7/
configs/qa_chunk_selection/v7_rapr/
artifacts/qa_chunk_selection_v7_rapr/
logs/qa_chunk_selection_v7/
```

---

# 12. Historical code reuse

Разрешено:

```text
import historical pure helpers
wrap historical code
reuse loaders/inference/metrics
```

Не редактировать historical modules.

Если требуется новая логика:

```text
write V7-specific adapter
```

---

# 13. Write guard

Реализовать forbidden write prefixes для всех V4.x/V5/V6 paths.

Любая попытка write:

```text
raise RuntimeError
```

---

# 14. Git safety

До coding/run:

```bash
git status --porcelain
git rev-parse HEAD
git diff
```

сохранить.

Не делать:

```text
git reset
git clean
checkout -- .
```

После run сохранить новый `git diff`/`status`.

---

# 15. Historical hashes

Перед run:

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

# 16. Historical baseline replay

Перед V7 воспроизвести:

```text
B0
V6 LOCAL S4
V6 S4-lex attribution
```

где exact replay технически возможен.

V6 historical results используются только как frozen references.

---

# 17. Candidate universe

Полностью historical:

```text
documents
chunks
chunk ids
chunk text
chunking
order
candidate universe
```

Нельзя менять.

---

# 18. Question policy

Использовать exact historical V6 policy.

RU:

```text
Q0
```

EN:

```text
exact historical question policy
```

SciER fold-local selection nested-safe.

---

# 19. K

Historical:

```text
no new K search
```

Known:

```text
RuSERRC K=5
SciMDIX K=4
```

EN восстановить exact из manifests.

---

# 20. Hard progressive runtime limit

Primary:

```text
MAX_EXTRA_QA = 2
MAX_QA_PER_QUESTION = 3
```

То есть top1 + максимум 2 alternatives.

Это не tuneable primary parameter.

Secondary resource diagnostic:

```text
MAX_EXTRA_QA = 1
```

но он не может стать post-hoc primary.

---

# 21. Почему hard cap = 3

V6 показал высокий P95 при full top-K.

Cap=3:

- ограничивает worst-case neural inference;
- позволяет проверить больше одной альтернативы;
- не превращает метод обратно в full K.

---

# 22. Gold utility definition

Для B0 top1:

```text
C0
A0 = QA(Q,C0)
```

Для candidate `Ci`:

```text
Ai = QA(Q,Ci)
```

\[
u_i = TokenF1(A_i,G)-TokenF1(A_0,G)
\]

Token F1:

```text
exact historical evaluator
0...100
```

Gold только target/evaluation.

---

# 23. Expected utility model

Primary V7 utility model должен оценивать:

\[
\hat{\mu}_i \approx E[u_i \mid x_i]
\]

Это принципиально отличается от V6:

```text
V6 → P(u>0)
V7 → expected magnitude of ΔF1
```

---

# 24. Utility regressor

Primary:

```text
StandardScaler
+
Ridge regression
```

Использовать:

```text
Ridge(fit_intercept=True)
```

Predeclared alpha:

```text
alpha ∈ {0.1, 1.0, 10.0}
```

Выбор только meta-OOF внутри outer TRAIN.

---

# 25. Почему Ridge

- оценивает continuous conditional mean;
- маленькая вычислительная стоимость;
- поддерживает sample weights;
- интерпретируемые coefficients;
- подходит к ограниченному размеру datasets;
- не требует новой LM;
- explicit interactions остаются контролируемыми.

---

# 26. Utility clipping

Gold utility:

```text
-100...+100
```

Training target не clip.

Predicted:

```text
clip(mu_hat, -100, +100)
```

только после model output для runtime stability.

---

# 27. Harm-risk target

Отдельный classifier:

\[
h_i = 1[u_i < 0]
\]

То есть:

> приведёт ли switch к ухудшению Token F1?

---

# 28. Harm classifier

```text
StandardScaler
+
LogisticRegression
```

```text
penalty = L2
solver = liblinear
C ∈ {0.1,1.0,10.0}
max_iter = 3000
class_weight = None
```

C выбирается только inner/meta OOF.

`class_weight=None` выбран намеренно: `p_harm` используется как runtime risk score, а искусственное изменение class prior ухудшило бы вероятностную интерпретацию. Дисбаланс классов компенсируется threshold selection и question-level weights.

---

# 29. Почему отдельный harm classifier

Regression может предсказать положительный mean при существенном negative tail.

Поэтому switch требует одновременно:

```text
positive expected utility
AND
low predicted probability of harm
```

---

# 30. Safe-switch rule

Для candidate i:

```text
mu_i = expected Token-F1 utility
p_harm_i = P(u_i < 0)
```

Candidate считается safe, если:

```text
mu_i >= tau_utility
AND
p_harm_i <= tau_harm
```

---

# 31. tau_utility grid

Predeclared:

```text
0.00
0.25
0.50
1.00
2.00
```

F1 points.

---

# 32. tau_harm grid

Predeclared:

```text
0.20
0.30
0.40
0.50
```

---

# 33. Threshold selection

Выбирать joint pair:

```text
(tau_utility, tau_harm)
```

только по META_PRED_OOF simulation.

Objective:

```text
max Token F1
```

subject to:

```text
ΔHit@1 >= -0.5 pp
ΔHN@1 <= +1.0 pp
```

Tie-break:

1. lower switch rate;
2. larger tau_utility;
3. smaller tau_harm.

---

# 34. Candidate-admission model

До QA alternative candidate определить:

> стоит ли вообще тратить QA-forward на этот candidate?

Для каждого `i>0`:

\[
a_i=1[u_i>0]
\]

---

# 35. Admission runtime features

Только pre-QA:

```text
candidate B0 rank
candidate B0 z-score
ΔB0_norm to top1

effective_k
candidate_count

gap12_norm
gap13_norm
B0 entropy
top1 z-score

language_id
```

Не использовать:

```text
aspect
predicted answer
Q-A
LEX
QA confidence
gold
```

Primary admission model **без aspect**.

---

# 36. Admission model

```text
StandardScaler
+
OneHotEncoder(language)
+
LogisticRegression
```

Fixed:

```text
C = 1
penalty = L2
solver = liblinear
class_weight = None
max_iter = 3000
```

`class_weight=None` используется по той же причине: `p_admit` нужен для сортировки и threshold policy. Question-level weighting сохраняется, а operating threshold выбирается на OOF pipeline simulation.

---

# 37. Admission output

Для каждого alternative:

```text
p_admit_i = P(u_i > 0 | pre-QA features)
```

---

# 38. Candidate queue

Runtime:

1. вычислить `p_admit` для всех `i>0`;
2. оставить `p_admit_i >= tau_admit`;
3. отсортировать по `p_admit desc`;
4. tie-break — better B0 rank;
5. оставить максимум 2 candidates.

---

# 39. tau_admit

Выбирать не по accuracy admission classifier, а по полной META_PRED_OOF pipeline simulation.

Predeclared grid:

```text
0.30
0.40
0.50
0.60
0.70
0.80
```

---

# 40. Admission resource constraint

При выборе tau_admit:

```text
mean QA forwards/question <= 2.0
```

на META_PRED_OOF simulation.

Hard cap гарантирует:

```text
QA forwards/question <= 3
```

на любом runtime question.

---

# 41. Joint policy selection

Не делать один огромный unconstrained grid.

Использовать staged selection.

## Stage 1 — Ridge

Выбрать alpha по OOF utility regression quality.

## Stage 2 — Harm

Выбрать C по OOF probabilistic quality.

## Stage 3 — Runtime policy

С зафиксированными models выбрать:

```text
tau_admit
tau_utility
tau_harm
```

по full-pipeline OOF Token F1 при safety + resource constraints.

Outer EVAL не используется.

---

# 42. Ridge selection metric

Primary:

```text
question-weighted MAE of u_i
```

Tie-break:

```text
if MAE difference < 0.05 F1:
choose larger alpha
```

---

# 43. Harm-model selection metric

Primary:

```text
question-weighted OOF log-loss
```

Tie-break:

```text
if difference < 0.001:
choose smaller C
```

---

# 44. Question-level weighting

Каждый question total weight = 1.

Если question имеет `m_q` alternatives:

\[
w_{qi}=1/m_q
\]

Использовать для:

```text
Ridge
harm classifier
admission classifier
```

---

# 45. POOLED weighting

Secondary POOLED diagnostic:

после question normalization:

```text
equal total weight per corpus family
```

SciER обе cells:

```text
one family total weight
```

Дополнительно ввести `lineage_key`:

```text
(canonical_document_id, canonical_question_or_aspect_id)
```

Если один и тот же исходный SciER question представлен одновременно в 384/50 и 128/24 cells, суммарный вес этого lineage сначала нормируется к 1 и делится между representations, а уже затем выполняется family normalization.

Это запрещает скрытое удвоение веса SciER из-за двух схем чанкинга.

Сохранить `training_weight_audit.json`.

---

# 46. Primary runtime selector features

После QA candidate:

```text
ΔB0_norm
ΔQ-A_norm
PairLEX

candidate B0 rank
effective_k

gap12_norm
B0 entropy

language_id

answer_valid_i
answer_valid_0
both_empty
```

Без:

```text
aspect
source-support
SEM
dataset id
```

---

# 47. Q-A relevance

\[
QArel_i=Ranker(Q,A_i)
\]

Использовать тот же exact ranker state, который дал B0.

Primary:

```text
within-query oriented normalized difference
```

Для progressive runtime normalization Q-A должна использовать только значения **уже вычисленных answers** плюс A0; будущие unevaluated candidate answers запрещены.

---

# 48. Progressive lexical feature

V7 primary LEX не требует QA всех K.

Определить:

\[
PairLEX_i = TokenF1(A_i,A_0)/100
\]

где:

```text
A0 = predicted answer B0 top1
Ai = predicted answer current candidate
```

Это gold-free runtime feature.

---

# 49. PairLEX normalization

`PairLEX` уже:

```text
0...1
```

Не применять corpus-specific normalization.

---

# 50. Связь PairLEX с historical LEX

До full run выполнить audit на вопросах, где historical full LEX доступен.

Report:

```text
correlation
rank agreement
```

PairLEX — новый заранее определённый progressive feature.

Не утверждать, что он идентичен historical LEX.

---

# 51. Primary explicit interactions

Минимально:

```text
ΔQ-A × gap12_norm
PairLEX × gap12_norm
ΔQ-A × PairLEX
ΔB0 × gap12_norm
```

---

# 52. Что исключено из primary

```text
aspect interactions
source-support
SEM
QA confidence
automatic polynomial features
new pretrained models
```

---

# 53. Candidate rank

Добавить:

```text
candidate_rank_norm = (rank-1)/(K-1)
```

Если K=1 alternatives нет.

---

# 54. Feature orientation

Все relevance-like continuous features:

```text
higher = better
```

до difference calculation.

---

# 55. B0 within-query normalization

B0 score нормализуется по historical top-K, потому что эти scores доступны до QA для всех candidates.

Если std < 1e-8:

```text
all z=0
```

---

# 56. Q-A progressive normalization

Нельзя нормализовать `Q-A` по answers, которые runtime ещё не вычислил.

Поэтому primary V7 использует для Q-A один из двух заранее фиксированных вариантов:

```text
A) raw oriented score transformed by TRAIN-fitted scaler
```

или

```text
B) delta to A0 in raw oriented score,
then TRAIN-fitted StandardScaler
```

Primary выбрать **B**:

\[
\Delta QArel_i^{raw}=QArel_i-QArel_0
\]

после чего StandardScaler fit только на training rows.

Это исправляет потенциальную future-information leakage progressive runtime.

Не использовать V6-style full-shortlist Q-A z-score в V7 primary.

---

# 57. Empty QA answer

Если candidate QA возвращает empty/no-answer:

```text
answer_valid=0
```

Q-A:

```text
use exact ranker score for empty text only if historical formatter supports it
```

Иначе:

```text
ΔQ-A = 0
```

с mandatory validity flag.

PairLEX:

```text
0
```

если один answer empty.

Если оба empty:

```text
PairLEX=1
both_empty=1
```

Missing policy зафиксировать до EVAL.

---

# 58. Progressive decision procedure

Runtime exact algorithm:

```text
1. B0 rank all historical candidates.
2. QA(B0 top1) → A0.
3. Compute admission probabilities for all alternatives.
4. Build queue with p_admit >= tau_admit.
5. Keep max 2 queue candidates.
6. For candidate in queue:
       QA(candidate) → Ai
       compute ΔQ-A using A0 and Ai only
       compute PairLEX(Ai,A0)
       compute selector features
       mu = utility_regressor(features)
       p_harm = harm_classifier(features)

       if mu >= tau_utility
          and p_harm <= tau_harm:
              select candidate
              STOP

7. If no accepted candidate:
       select B0 top1.
```

---

# 59. Почему stop после первого safe candidate

Цель V7:

```text
получить безопасный positive correction
```

а не найти oracle-best candidate любой ценой.

Это:
- уменьшает compute;
- снижает multiple-comparison risk;
- делает runtime bounded;
- соответствует dissertation resource objective.

---

# 60. Candidate queue order

Primary:

```text
descending p_admit
```

Tie:

```text
better B0 rank
```

Secondary diagnostic:

```text
pure B0 rank order
```

но diagnostic не становится post-hoc primary.

---

# 61. K=1

Если:

```text
effective_k=1
```

то:

```text
QA top1 only
preserve B0
```

---

# 62. K=2

Одна alternative максимум.

Hard cap candidate set не меняет.

---

# 63. Systems V7

Predeclare:

```text
S0 = B0

S1 = EU-FULL
     expected-utility Ridge
     all historical top-K QA
     no harm guard
     no admission
     choose candidate with max predicted mu if mu >= tau_utility

S2 = RISK-FULL
     S1
     + harm classifier
     all top-K QA

S3 = RAPR
     S2
     + candidate admission
     + max 2 additional QA
     + progressive stop
     PRIMARY

S3-noLEX
     S3 without PairLEX
     diagnostic only

S3-B0-order
     S3 queue ordered by B0 rank
     diagnostic only
```

---

# 64. Why S1

Проверяет отдельно:

> помогает ли continuous expected-utility target относительно V6 binary utility?

---

# 65. Why S2

Проверяет:

> добавляет ли explicit harm-risk guard безопасность?

---

# 66. Why S3

Проверяет:

> можно ли сохранить gain S2 при bounded progressive compute?

---

## Diagnostic-variant tuning rule

`S3-noLEX` и `S3-B0-order` должны проходить **собственный** inner/meta-OOF threshold selection по тем же grids и constraints.

Нельзя использовать thresholds full S3, если feature/queue definition изменилась.

При этом diagnostic systems остаются ineligible для final V7 selection.

---

# 67. Exact candidate-selection semantics for S1/S2/S3

## S1 EU-FULL

Все alternatives top-K получают QA и `mu_hat`.

Выбрать:

```text
best_i = argmax(mu_hat_i)
```

Switch только если:

```text
mu_hat_best >= tau_utility_S1
```

`tau_utility_S1` выбирается по S1-specific META_PRED_OOF simulation.

S1 не использует harm classifier и admission.

## S2 RISK-FULL

Все alternatives top-K получают QA, `mu_hat` и `p_harm`.

Safe set:

```text
mu_hat_i >= tau_utility_S2
AND
p_harm_i <= tau_harm_S2
```

Если safe set пуст:

```text
keep B0
```

Иначе выбрать safe candidate с максимальным `mu_hat`.

Thresholds S2 выбираются только по S2 META_PRED_OOF.

## S3 RAPR

Использует admission queue и выбирает **первого safe candidate в queue**, после чего останавливается.

Thresholds S3 выбираются отдельно по полной S3 META_PRED_OOF simulation.

Нельзя переиспользовать S2 thresholds в S3 без отдельного заранее заданного OOF selection, потому что progressive queue меняет decision distribution.

---

# 68. Historical references

В final table добавить:

```text
V6 LOCAL S4
V6 S4-lex attribution
V6 POOLED S4-no-aspect diagnostic
BGE RU strong reference
```

Но они не участвуют в V7 model selection.

---

# 69. Primary parameterization mode

Primary:

```text
LOCAL
```

Same architecture across all cells.

Separate fitted meta-parameters.

---

# 70. Secondary POOLED-simple

POOLED uses:

```text
same V7 features
NO aspect
NO source
NO dataset id
equal corpus-family weights
shared model and shared thresholds
```

## 70.1. One global model per `(outer_fold, seed)`

Для global outer fold `f` и seed `s` одновременно удерживать EVAL fold `f` **во всех пяти cells**.

Train scope:

```text
all allowed non-EVAL documents from all corpus families
minus cross-cell/cross-corpus embargo documents
```

На этом scope обучается **один** admission model, один utility model, один harm model и один общий threshold tuple.

Нельзя обучать отдельный `POOLED model for SciREX`, `POOLED model for RuSERRC` и затем называть их pooled.

## 70.2. Shared policy selection

META_PRED_OOF from all training families объединяются с equal family weights.

Один общий набор:

```text
alpha
C_harm
tau_admit
tau_utility
tau_harm
```

выбирается для всех target cells данного `(fold,seed)`.

## 70.3. Runtime features

Разрешён `language_id`, потому что язык известен runtime и upstream rankers различаются.

Запрещены:

```text
dataset_id
corpus_family_id
file/source path
```

Это transfer diagnostic; он не заменяет primary LOCAL S3.

---

# 71. Secondary LOCO-meta

Same simplified V7 feature schema.

Для held-out corpus family:

1. полностью исключить её rows из meta-model fitting;
2. полностью исключить её rows из alpha/C/threshold selection;
3. обучить admission/utility/harm только на остальных families;
4. выбрать один threshold tuple только на training families;
5. применить frozen meta-policy к held-out family без local refit.

SciER 384/50 и SciER 128/24 уходят вместе.

Upstream B0/QA held-out family могут оставаться historical corpus-specific, поэтому это:

```text
LOCO-meta transfer
```

а не full zero-shot pipeline.

---

# 72. SciER cross-cell embargo

Обязателен во всех pooled/LOCO training.

Если document находится target EVAL в одной SciER cell, его representation из второй cell запрещено в training.

Использовать:

```text
canonical_document_id
full document SHA-256
```

---

# 73. Cross-corpus overlap audit

Exact/normalized full-document hashes.

Любой overlap:

```text
same leakage group
```

---

# 74. Outer CV

Exact historical:

```text
5 document-grouped folds
```

Не менять.

---

# 75. Seeds

```text
42
43
44
```

Не объединять seeds в один training fit.

---

# 76. Inner meta folds

Default:

```text
4 document-grouped folds
```

Fallback:

```text
3
2
```

только по feasibility.

---

# 77. Upstream provenance

Как V6:

```text
fully_frozen
dataset_trained
mixed
```

Dataset-trained upstream features должны быть OOF.

---

# 78. Two OOF levels

Обязательно различать:

```text
META_FEATURE_OOF
META_PRED_OOF
```

`META_FEATURE_OOF` — honest upstream features.

`META_PRED_OOF` — admission/Ridge/harm predictions на rows, на которых эти meta-models не fit.

Threshold selection только по `META_PRED_OOF`.

---

# 79. Outer workflow

```text
outer TRAIN+CAL
    ↓
honest upstream features
    ↓
meta-model OOF
    ↓
select:
Ridge alpha
harm C
tau_admit
tau_utility
tau_harm
    ↓
refit meta models on all allowed TRAIN+CAL
    ↓
freeze policy
    ↓
outer EVAL once
```

---

# 80. Degenerate Ridge

Если all `u=0`:

```text
utility_status = zero_variance
mu=0
no switch
```

---

# 81. Degenerate harm classifier

Если harm target one-class:

```text
all safe → p_harm=0
all harmful → p_harm=1
```

Не обучать invalid classifier.

---

# 82. Degenerate admission classifier

Если admission target one-class:

```text
all 0 → queue empty
all 1 → p_admit=1 for all alternatives
```

Hard cap сохраняется.

---

# 83. Outer threshold fallback

Если ни один threshold tuple не проходит safety/resource на META_PRED_OOF:

```text
fold_policy = B0_FALLBACK
```

Outer EVAL:

```text
no extra QA
no switch
```

Сохранить причину.

---

# 84. Нельзя считать fallback успехом

В final report:

```text
fallback-fold count
fallback-question rate
```

Нулевой delta при полном fallback не является evidence of equivalence.

---

# 85. Minimum activity / non-degeneracy criterion

Чтобы fallback к B0 не считался «универсальным методом», second stage должен быть фактически активен.

Определить question как `stably_switched`, если финальный candidate отличается от B0 top1 минимум в:

```text
2 из 3 seeds
```

Для каждой protocol cell требуется:

```text
N_stably_switched >= max(3, ceil(0.01 * N_questions))
```

для перехода к финальному dissertation candidate.

Если quality gate выполнен, но activity ниже порога:

```text
status = DEGENERATE_FALLBACK_CANDIDATE
```

Такой результат report, но не считать доказательством дополнительной ценности второго компонента.

Дополнительно report:

```text
any-seed switch rate
2-of-3 stable switch rate
all-3-seed switch rate
```

Этот criterion фиксируется до outer EVAL.

---

# 86. Primary quality metric

```text
Token F1
```

Exact historical evaluator.

---

# 87. Secondary answer metrics

```text
Char F1
Lemma F1 RU
ROUGE-L
Unicode ROUGE-L RU
EM
BERTScore if historical
```

---

# 88. Evidence metrics

```text
Hit@1
Hit@K
HN@1
HN@K
F1 | selected positive
Amb@1
GSRN@1
```

где applicable.

---

# 89. Switch metrics

```text
good
bad
neutral

mean utility switched
median utility switched

large gain: u >= +5
large harm: u <= -5
```

---

# 90. Risk diagnostics

Для accepted switches:

```text
predicted mu
actual u
predicted p_harm
```

Report calibration by bins.

---

# 91. Utility regression diagnostics

OOF:

```text
MAE
RMSE
Spearman
Pearson
R2
```

Scientific success определяется downstream pipeline, не R2.

---

# 92. Harm diagnostics

```text
AUROC
PR-AUC
log-loss
Brier
```

---

# 93. Admission diagnostics

```text
AUROC
PR-AUC
Recall of truly positive-utility alternatives
precision
```

Отдельно:

```text
oracle positive candidate missed by admission
```

---

# 94. Resource metrics

```text
mean QA forwards/question
max QA forwards/question
mean Q-A ranker forwards/question
median latency
P95
P99
throughput
VRAM
RSS
loaded params
```

---

# 95. Hard resource guarantee

Primary S3:

```text
QA forwards <= 3
```

на каждом question.

Если >3:

```text
IMPLEMENTATION BUG
RUN INVALID
```

---

# 96. Mean resource target

TRAIN/CAL policy:

```text
mean QA forwards <= 2.0
```

Outer actual report.

---

# 97. Latency target

Primary engineering:

```text
median <= 1.25 × B0
P95 <= 2.5 × B0
```

---

# 98. Fresh-process benchmark

Каждая system:

```text
fresh process
```

No neural cache timing.

---

# 99. Benchmark sample

Для каждой cell:

```text
all outer-fold-0 questions capped at 100
```

Если >100, stratify by:

```text
aspect
effective K
```

Fixed sample manifest shared across systems.

Benchmark configuration фиксирована:

```text
parameterization_mode = LOCAL
outer_fold = 0
seed = 42
```

Для S1/S2/S3 загружать именно fold-0/seed-42 fitted models и thresholds этой cell.

В summary обязательно записать:

```text
mode
fold
seed
alpha
C_harm
tau_admit
tau_utility
tau_harm
```

Это предотвращает неоднозначность resource report, обнаруженную после V6.

---

# 100. Warm-up / repeats

```text
warmup = min(10, sample size)
repeats = 3
```

GPU synchronize around timing.

---

# 101. Correct CUDA memory

Использовать actual model device.

---

# 102. Statistical unit

Seeds are not samples.

1. per-question per-seed metric;
2. average seeds within question;
3. document-cluster bootstrap.

---

# 103. Bootstrap

```text
5000
paired
cluster=document_id
seed=42
```

---

# 104. SciER family weighting

Global summary:

```text
SciER effect = mean(384,128)
```

Then equal families:

```text
SciREX
SciER
RuSERRC
SciMDIX
```

---

# 105. Primary development quality gate

S3 RAPR проходит quality gate, если одновременно:

```text
ΔTokenF1 > 0
```

на всех пяти protocol cells,

```text
LCB95 > -0.5 F1
```

на каждой,

```text
evidence safety PASS
```

и выполнен non-degeneracy criterion из раздела о minimum activity.

Это development gate, а не independent proof of superiority.

---

# 106. Evidence safety

```text
ΔHit@1 >= -0.5 pp
ΔHN@1 <= +1.0 pp
```

где applicable.

---

# 107. Resource success gate

Resource PASS требует:

```text
hard max QA <= 3  [обязательно, архитектурный invariant]
```

и для benchmark каждой cell:

```text
median latency <= 1.25 × B0
P95 latency <= 2.5 × B0
```

Дополнительно outer-CV report:

```text
mean QA forwards/question
```

с engineering target:

```text
<= 2.0
```

Если quality gate PASS, но resource gate FAIL:

```text
status = QUALITY_ONLY_CANDIDATE
```

Такой вариант научно интересен, но не является финальным ресурсоэффективным диссертационным кандидатом.

---

# 108. Strict scientific success

Strong:

```text
LCB95 > 0
```

на всех cells.

Не требовать для development freeze, но report.

---

# 109. No post-hoc winner substitution

Если S3 fails, нельзя объявить:

```text
S1 winner
S2 winner
S3-noLEX winner
S3-B0-order winner
```

как final V7 post-hoc.

Diagnostics используются только для следующей version.

---

# 110. Development freeze rule

Переход к confirmatory V7-B разрешён только если primary LOCAL S3 одновременно:

```text
QUALITY GATE = PASS
NON-DEGENERACY = PASS
RESOURCE GATE = PASS
```

Тогда:

```text
freeze S3 architecture and training procedure
```

Если quality PASS, resource FAIL:

```text
QUALITY_ONLY_CANDIDATE
```

Если quality/safety/non-degeneracy FAIL:

```text
V7 universal candidate NOT SUPPORTED
```

Не продолжать tuning внутри V7 и не заменять primary S3 диагностической системой.

---

# 111. POOLED transfer criterion

Secondary POOLED-simple считается promising, если:

```text
family-balanced Δ > 0
AND
no cell Δ < -0.5
AND
evidence safe
```

Но он не заменяет primary LOCAL S3.

---

# 112. LOCO-meta interpretation

Только transfer diagnostic.

Не final method selection.

---

# 113. Artifact namespace

```text
artifacts/qa_chunk_selection_v7_rapr/run_<timestamp>/
```

---

# 114. Artifact structure

```text
run_<timestamp>/
  status.json
  COMPLETED.json
  frozen_contract.yaml
  manifest.json

  audit/
    git_before/
    git_after/
    historical_hashes_before.json
    historical_hashes_after.json
    historical_replay/
    candidate_universe.json
    split_hashes.json
    model_manifests/
    question_policy/
    upstream_provenance.json
    scier_lineage.json
    cross_corpus_overlap.json
    regression_summary.json

  cache/
    qa/
    q_a/
    meta_features/

  models/
    admission/
    utility/
    harm/

  predictions/
    local/
    pooled/
    loco_meta/

  analysis/
    primary_metrics.json
    bootstrap.json
    switch_analysis.json
    utility_diagnostics.json
    harm_diagnostics.json
    admission_diagnostics.json
    fold_stability.json
    seed_stability.json
    aspect_analysis.json
    k_analysis.json
    resource_summary.json
    global_summary.json

  resource_benchmark/
    sample_manifest.json
    raw/
    summary.json
    environment.json

  reports/
    results_v7.md
    reproducibility.md
```

---

# 115. No overwrite

Every run immutable.

```text
overwrite=False
```

Bugfix → new run.

---

# 116. Atomic writes

```text
tmp
fsync
atomic rename
```

---

# 117. Cache isolation

V7 writes only V7 cache.

Historical cache:

```text
read only
```

---

# 118. Cache key

Minimum:

```text
dataset cell
document
question
candidate
model hash
ranker state hash
QA hash
outer fold
inner fold
seed
feature version
```

---

# 119. Runtime/evaluation namespace

Prediction rows:

```json
{
  "runtime": {},
  "evaluation": {}
}
```

Gold only evaluation/training target storage.

---

# 120. Runtime row fields

```text
document_id
canonical_document_id
question_id
cell
family
language
aspect
outer fold
seed
B0 top-K
B0 scores
admission probabilities
queue
A0
evaluated candidate answers
Q-A scores
PairLEX
utility prediction
harm probability
thresholds
accepted/rejected reason
selected candidate
QA forward count
Q-A forward count
```

---

# 121. Evaluation row fields

```text
gold answer
actual candidate utilities
final Token F1
evidence labels
switch good/bad/neutral
```

---

# 122. Gold leakage tests

Runtime cannot contain:

```text
gold
actual utility
positive label
HN label
oracle rank
oracle best candidate
```

---

# 123. Progressive state leakage

At first evaluated alternative, features cannot use answers/features of unevaluated candidates.

Добавить explicit regression test.

---

# 124. QA-forward accounting test

Count logical model forwards, not cache reads.

Quality cache may exist.

Resource benchmark executes real model inference.

---

# 125. Regression tests

At minimum:

```text
test_history_read_only
test_v6_replay
test_candidate_universe
test_splits
test_questions
test_k
test_same_ranker_state_q_a
test_score_orientation
test_pairlex_formula
test_pairlex_gold_free
test_question_weights
test_seed_isolation
test_upstream_oof
test_meta_prediction_oof
test_threshold_oof_only
test_admission_pre_qa_only
test_progressive_state_no_future_features
test_max_qa_three
test_early_stop
test_empty_answer_policy
test_scier_embargo
test_cross_corpus_overlap
test_runtime_no_gold
test_no_dataset_id_pooled
test_cuda_device
test_no_historical_changes
test_no_overwrite
```

---

# 126. Smoke test

Before full GPU:

```text
1 cell
1 fold
1 seed
small subset
```

Verify:

```text
admission queue
top1 QA
candidate QA
PairLEX
utility
harm
safe switch
early stop
forward counters
artifacts
```

No scientific interpretation.

---

# 127. Execution order

```text
A. audit/replay
B. unit tests
C. smoke
D. freeze contract
E. full LOCAL S0-S3
F. diagnostics noLEX / B0-order
G. secondary pooled-simple
H. secondary LOCO-meta
I. analysis
J. resource benchmark
K. history hash assertion
```

---

# 128. Frozen contract

Before full outer EVAL store:

```text
dataset cells
historical models
questions
K
MAX_EXTRA_QA=2
feature schema
PairLEX formula
interactions
Ridge alpha grid
harm C grid
tau_admit grid
tau_utility grid
tau_harm grid
weighting
OOF procedure
threshold selection
fallback policy
metrics
safety
resource targets
success criteria
diagnostic variants
pooled/LOCO rules
```

Hash SHA-256.

---

# 129. После freeze запрещено менять

```text
features
PairLEX
interactions
targets
alpha grid
C grid
threshold grids
hard QA cap
queue order
success criterion
resource constraints
```

New idea → V7.1.

---

# 130. Technical bug handling

If bug:

```text
run invalid
preserve artifacts
document bug
new run
rerun all affected cells
```

---

# 131. Final history assertion

Before COMPLETED:

```text
historical_hashes_before == historical_hashes_after
```

Otherwise no COMPLETED marker.

---

# 132. CLI suggestion

Audit:

```bash
python -m experiments.qa_chunk_selection_v7.audit \
  --config configs/qa_chunk_selection/v7_rapr/main.yaml \
  --read-only-history \
  --run-dir artifacts/qa_chunk_selection_v7_rapr/<RUN>
```

Run:

```bash
python -m experiments.qa_chunk_selection_v7.run \
  --config configs/qa_chunk_selection/v7_rapr/main.yaml \
  --cells scirex scier_384_50 scier_128_24 russerrc scimdix \
  --seeds 42 43 44 \
  --outer-folds 5 \
  --inner-folds 4 \
  --run-dir artifacts/qa_chunk_selection_v7_rapr/<RUN>
```

Analyze:

```bash
python -m experiments.qa_chunk_selection_v7.analyze \
  --run-dir artifacts/qa_chunk_selection_v7_rapr/<RUN> \
  --bootstrap 5000
```

---

# 133. Independent confirmation

Если primary LOCAL S3 проходит development gate, подтверждение должно соответствовать LOCAL deployment assumption.

Для каждого нового external corpus требуется **заранее запечатанное document-grouped разделение**:

```text
CONFIRM_TRAIN/CAL
CONFIRM_TEST
```

После freeze V7 architecture:

1. разрешено fit LOCAL admission/utility/harm meta-layer только на `CONFIRM_TRAIN/CAL`;
2. разрешено выбирать alpha/C/thresholds только внутри `CONFIRM_TRAIN/CAL` по тому же nested rule;
3. `CONFIRM_TEST` не открывается до полного freeze fitted local policy;
4. test выполняется один раз.

Нельзя:

```text
использовать CONFIRM_TEST для threshold selection
менять feature schema
менять hard QA cap
менять targets
```

Желательно:

```text
1 independent EN corpus/test
1 independent RU corpus/test
```

Если новый corpus не предоставляет labeled TRAIN/CAL часть, primary LOCAL V7 на нём честно подтвердить нельзя; тогда можно оценивать только заранее frozen POOLED/LOCO-meta policy как отдельный transfer question.

---

# 134. Dissertation claim if V7 succeeds

Possible formulation:

> Разработан единый параметризуемый метод последовательной остаточной коррекции выбора фрагмента научно-технического документа, в котором дополнительные кандидаты обрабатываются адаптивно, а решение об изменении исходного результата принимается на основе прогнозируемой величины улучшения качества и оценки риска ухудшения; число дополнительных обращений к вопросно-ответной модели при этом ограничивается заранее заданным вычислительным бюджетом.

Do not claim:

```text
first reader-aware reranker
universal proven
```

before untouched confirmation.

---

# 135. Weaker defensible claim if V7 fails

> Показано, что оценка ожидаемой величины downstream-полезности и явное ограничение риска позволяют формализовать безопасную остаточную коррекцию выбора evidence; однако переносимый положительный эффект второго этапа зависит от корпуса и не подтверждён как универсальный.

---

# 136. Main scientific questions V7

### Q1
Даёт ли continuous expected-utility regression более стабильный selector, чем binary `P(u>0)`?

### Q2
Снижает ли harm-risk guard число крупных отрицательных переключений?

### Q3
Можно ли получить positive residual gain при hard cap `<=3 QA forwards/question`?

### Q4
Достаточен ли простой progressive PairLEX как переносимый lexical signal?

### Q5
Сохраняется ли одна архитектура на EN/RU при LOCAL parameter fitting?

### Q6
Улучшается ли P95 относительно V6 full/top-K approach?

---

# 137. Краткая инструкция code-agent

```text
Create a NEW V7/RAPR experiment.

Historical V4.4/V4.5/V5/V6 code and artifacts are READ-ONLY.

Primary architecture:
B0
→ candidate admission
→ QA top1
→ at most 2 extra candidates
→ expected-utility Ridge
→ harm-risk logistic
→ safe progressive switch

Hard runtime guarantee:
MAX_QA_PER_QUESTION = 3.

Primary selector target:
continuous
u_i = TokenF1(A_i,gold) - TokenF1(A0,gold)

Primary utility model:
Ridge.

Primary risk target:
u_i < 0.

Primary LEX:
PairLEX = TokenF1(pred_answer_i, pred_answer_top1)/100.

Primary features:
ΔB0
raw-delta Q-A transformed with TRAIN-fitted scaler
PairLEX
candidate rank
effective K
normalized B0 margin
normalized entropy
answer validity

Primary interactions:
ΔQ-A × margin
PairLEX × margin
ΔQ-A × PairLEX
ΔB0 × margin

NO aspect interactions.
NO source-support primary.
NO SEM primary.
NO new pretrained LM.

Admission is PRE-QA only.

Candidate queue:
sort by p_admit,
max 2 candidates.

Progressive:
evaluate candidate,
estimate expected utility and harm risk,
switch and STOP when safe.

Thresholds selected ONLY from META_PRED_OOF.

Mean TRAIN/CAL QA budget <=2.
Hard runtime max=3.

Primary mode:
LOCAL same architecture, separately fitted parameters.

POOLED-simple and LOCO-meta are secondary diagnostics only.

Success requires:
positive Token F1 delta on all five cells
LCB95 > -0.5 on each
evidence safety
and separate resource PASS flag.

Do not replace failed S3 with a post-hoc diagnostic winner.

Freeze one architecture before untouched EN/RU confirmation.

Never overwrite old runs or historical results.
```

---

# 138. Pre-run checklist

```text
[ ] V7 namespace created
[ ] history write guard active
[ ] git state saved
[ ] historical hashes before saved

[ ] V6 exact result/reference paths resolved
[ ] B0 replay PASS
[ ] V6 replay PASS where applicable

[ ] candidate universes verified
[ ] questions verified
[ ] K verified
[ ] model hashes verified

[ ] score direction verified
[ ] Q-A same-ranker-state verified

[ ] PairLEX formula frozen
[ ] utility target frozen
[ ] harm target frozen
[ ] admission target frozen

[ ] feature schema frozen
[ ] interaction schema frozen
[ ] progressive Q-A normalization policy frozen

[ ] Ridge alpha grid frozen
[ ] harm C grid frozen
[ ] tau_admit grid frozen
[ ] tau_utility grid frozen
[ ] tau_harm grid frozen

[ ] question weighting frozen
[ ] OOF procedures frozen
[ ] degenerate fallbacks frozen

[ ] MAX_EXTRA_QA=2 frozen
[ ] hard max QA test PASS

[ ] safety criteria frozen
[ ] resource targets frozen
[ ] success criteria frozen

[ ] SciER embargo PASS
[ ] cross-corpus overlap audit PASS

[ ] unit tests PASS
[ ] smoke PASS

[ ] frozen_contract hash saved
[ ] no full outer-EVAL seen before freeze
```

---

# 139. Post-run checklist

```text
[ ] all five cells complete
[ ] all 3 seeds complete
[ ] all 5 outer folds complete

[ ] S0-S3 complete
[ ] diagnostic noLEX complete
[ ] diagnostic B0-order complete
[ ] pooled-simple complete
[ ] LOCO-meta complete

[ ] utility diagnostics complete
[ ] harm diagnostics complete
[ ] admission diagnostics complete

[ ] Token F1 complete
[ ] evidence metrics complete
[ ] switch metrics complete
[ ] large-harm analysis complete

[ ] mean/max QA forwards verified
[ ] max QA <=3 for every question
[ ] latency benchmark complete
[ ] P95 benchmark complete
[ ] VRAM/RSS complete

[ ] bootstrap complete
[ ] seed/fold stability complete

[ ] historical hashes after saved
[ ] before == after
[ ] git diff saved

[ ] results_v7.md created
[ ] reproducibility.md created
[ ] COMPLETED.json created only after all validity checks
```

---

# 140. Итоговый смысл V7

V7 больше не пытается найти ещё один универсальный heuristic signal.

Он проверяет более общий механизм:

\[
\boxed{
\text{оценивай величину возможного выигрыша,}
\;\text{оценивай риск ущерба,}
\;\text{и трать compute только на перспективные alternatives}
}
\]

Если эта схема даст положительный residual gain на всех development cells при ограниченном числе QA-forward, она станет наиболее обоснованным кандидатом на итоговый диссертационный метод.

Если нет — результат будет означать, что residual chunk correction на основе frozen upstream signals не обладает достаточной переносимостью, и дальнейшее усложнение на тех же development datasets методологически не оправдано.
