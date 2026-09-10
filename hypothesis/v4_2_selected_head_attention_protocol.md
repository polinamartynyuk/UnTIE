# V4.2 — исправленный протокол: устойчивый reader-agreement + selected-head span attention

## 0. Назначение

Этот документ задаёт следующий экспериментальный цикл после V4.1.

Главная цель:

> проверить, даёт ли task-specific self-attention frozen extractive QA-ридера дополнительный сигнал качества evidence **поверх честно подтверждённого V6-K5**, если использовать небольшой поднабор контрастно выбранных attention-heads и анализировать связь `question ↔ predicted answer span` только внутри B0 top-5.

Вторичная цель:

> проверить, можно ли упростить V6 до более устойчивого reader-agreement baseline без слабых и плохо масштабированных признаков.

`INTERNAL_TEST` в V4.2 не открывается.

---

# 1. Исходная точка V4.1

Честный 5-fold OOF:

```text
B0-live:
Token F1 = 27.36
Hit@1    = 36.72%
HN@1     = 4.86%

V6-K5:
Token F1 = 34.62
ΔF1      = +7.26
95% CI   = [4.38, 10.17]
Hit@1    = 42.90%
HN@1     = 5.45%
Amb@1    = 42.90%
GSRN@1   = 26.64%
```

Подтверждённый механизм:

```text
ambiguous/GSRN → positive
```

при умеренной цене обратных переходов.

---

# 2. Что V4.1 говорит о V6

## 2.1. Reader-agreement — основной источник прироста

Без agreement:

```text
F1 ≈ 29.53
```

вместо:

```text
F1 ≈ 34.62.
```

## 2.2. Majority — наиболее убедительный отдельный компонент

Удаление majority ухудшило V6 статистически заметно.

## 2.3. Exact frequency почти ничего не добавляет

Его удаление почти не меняет результат.

## 2.4. Source-cluster support слаб на K=5

Его удаление практически не ухудшает V6.

## 2.5. Mean Token F1 требует новой шкалы

В V4.1 он использовался в шкале:

```text
0..100
```

и мог непропорционально доминировать внутри score.

## 2.6. B0-score не является главным источником прироста

Reader-only knockout был близок к full V6.

## 2.7. K=10 пока не primary

K10 имеет более высокий point estimate, но:
- не отделён статистически от K5;
- дороже;
- неоднороден по аспектам;
- ухудшает Hit@5.

Поэтому:

```text
V4.2 primary K = 5
```

---

# 3. Почему attention имеет смысл перепроверить

Старая V3 attention-ветка была отрицательной:

```text
AT1 AUROC ≈ 0.519
attention-only HN@1 ≈ 19%
text + attention ухудшал Hit@1
```

Но она проверяла более грубую гипотезу:

```text
общая / усреднённая attention signature
→ standalone ranking
```

V4.2 проверяет другую идею:

```text
B0 top-5
→ frozen QA
→ predicted span
→ selected task-specific attention heads
→ relative query-span alignment
→ correction поверх V6
```

---

# 4. Литературная логика

Используются следующие принципы:

1. **Query-Focused Retrieval Heads**  
   Полезный retrieval-signal может быть сосредоточен в небольшом числе heads.

2. **Contrastive Retrieval Heads**  
   Heads следует выбирать по относительному критерию `relevant > irrelevant`, а не по средней активации.

3. **ReAttn**  
   Raw attention может быть смещён в сторону lexical overlap; полезно reweighting по информативности токенов.

4. **RECONSIDER / span-focused QA**  
   При высоком top-K recall полезно фокусировать reranking на candidate answer span.

5. **Listwise / shortlist-relative reranking**  
   Относительный сигнал внутри набора сильных кандидатов может быть полезнее абсолютного pointwise score.

---

# 5. Главная гипотеза V4.2

> Если среди frozen RoBERTa QA heads выбрать на TRAIN те, которые устойчиво сильнее связывают вопрос с корректно извлекаемым positive span, чем с HN span, то относительный query-span attention этих heads может улучшить V6 top-5 reranking.

Primary fusion:

\[
S_{ATT1}(c_i)
=
Z_q(S_{V6}(c_i))
+
\delta\,A_{rel}(c_i)
\]

где:
- \(Z_q\) — z-нормализация только внутри top-5 данного `(question, seed)`;
- \(A_{rel}\) — относительный attention-score selected heads внутри того же top-5;
- `δ` выбирается только на CAL;
- head-set и его размер выбираются только на TRAIN.

---

# 6. Ключевое исправление: head selection должен соответствовать runtime distribution

Attention применяется только к:

```text
B0 top-5.
```

Поэтому primary head selection **не должен** обучаться на всех HN/easy chunks документа.

Для каждого outer TRAIN сначала сформировать:

```text
union of B0 top-5
across seeds 42/43/44
```

и удалить дубликаты:

```text
unique key = (question_id, chunk_id)
```

Head selection выполняется только внутри этого TRAIN shortlist-union.

Это исключает:
- distribution shift `all-document HN → top-5 HN`;
- тройное переучитывание одного и того же чанка из-за трёх B0 seeds.

---

# 7. Frozen folds / seeds / leakage rules

Использовать:
- те же 5 document-level folds;
- ту же outer rotation;
- B0 seeds `42/43/44`;
- тот же frozen QA checkpoint.

Для каждого outer run:

```text
TRAIN → head discovery + N selection
CAL   → δ selection
EVAL  → frozen evaluation
```

Запрещено:
- выбирать heads по CAL/EVAL labels;
- использовать EVAL при выборе N;
- использовать ambiguous/GSRN как negative supervision;
- открывать `INTERNAL_TEST`.

---

# 8. Reference systems

Обязательные:

```text
B0-live
V6-K5 honest OOF из V4.1
```

Для attention primary использовать **ровно per-run V6 weights из V4.1**.

V6 заново не ретюнить.

---

# 9. Phase A — V6-Core simplification

Эта ветка вторичная.

Цель:

> проверить, можно ли получить почти тот же результат более простой и устойчивой reader-agreement формулой.

---

# 10. CORE1 — B0 + majority

\[
S_{CORE1}
=
Z_q(s_{B0})
+
\lambda_m Majority
\]

Grid:

```text
λ_m ∈ {0.25, 0.5, 1.0, 2.0}
```

Для каждого outer run:

```text
CAL → λ_m
EVAL → frozen λ_m
```

Merge только EVAL → true OOF CORE1.

---

# 11. CORE2 — majority + нормализованный Token-F1

\[
T_i =
mean\_token\_f1_i / 100
\]

\[
S_{CORE2}
=
Z_q(s_{B0})
+
\lambda_m Majority
+
\lambda_t T_i
\]

Grid:

```text
λ_m ∈ {0.25,0.5,1.0,2.0}
λ_t ∈ {0,0.25,0.5,1.0}
```

Это новая система.

Нельзя сравнивать её коэффициенты с old V6, где Token F1 был `0..100`.

---

# 12. CORE3 — reader-only diagnostic

```text
Majority + normalized Token-F1
без B0
```

Diagnostic only.

---

# 13. Решение по V6-Core

CORE не заменяет V6 автоматически.

Если нужен критерий упрощения, использовать non-inferiority-style rule:

```text
ΔF1 = CORE - V6
```

CORE можно считать practically non-inferior, если:

```text
нижняя граница 95% CI > -0.5 F1
```

и:
- HN safeguard не хуже;
- формула проще;
- CAL weights стабильнее.

Иначе reference остаётся V6-K5.

---

# 14. Phase B — attention extraction regression

Использовать тот же frozen RoBERTa QA:

```text
output_attentions=True
```

Перед массовым извлечением проверить:

```text
start logits
end logits
predicted start/end
predicted answer text
```

с/без attentions.

Если backend требует перехода на eager attention implementation:

```text
attn_implementation = eager
```

это допустимо только после regression, подтверждающего отсутствие существенного изменения QA outputs.

---

# 15. Token masks

Для attention-features явно определить:

```text
Q = только question content tokens
C = только context content tokens
S = predicted span tokens внутри C
```

Исключать:
- `<s>`;
- `</s>`;
- `<pad>`;
- служебные разделители;
- padding.

Использовать tokenizer offsets / sequence ids, а не эвристику по индексам.

---

# 16. Почему raw Q→span mass нельзя использовать без поправки на длину span

Если просто суммировать attention в span:

\[
\sum_{s\in S} A(q,s)
\]

то длинный span автоматически имеет больше возможностей получить высокий score.

Это особенно опасно, потому что длины ответов различаются по аспектам.

Поэтому primary использует **length-normalized attention density / enrichment**, а raw mass сохраняется только как diagnostic.

---

# 17. Q→span attention density

Для head `h`:

\[
D_h^{Q\rightarrow S}
=
\frac{1}{|Q||S|}
\sum_{q\in Q}
\sum_{s\in S}
A_h(q,s)
\]

Это средняя attention-weight на одну пару `question-token → span-token`.

---

# 18. Q→span enrichment

Дополнительно:

\[
M_h^{Q\rightarrow S}
=
\frac{1}{|Q|}
\sum_{q\in Q}
\sum_{s\in S}
A_h(q,s)
\]

Ожидаемая масса при равномерном распределении attention по context:

\[
E_{uniform}
=
\frac{|S|}{|C|}
\]

Тогда:

\[
E_h^{Q\rightarrow S}
=
\frac{
M_h^{Q\rightarrow S}
}{
|S|/|C|+\epsilon
}
\]

Primary feature-family candidate:

```text
Q→span enrichment
```

потому что он учитывает длину span и context.

---

# 19. Span→Q attention density

\[
D_h^{S\rightarrow Q}
=
\frac{1}{|S||Q|}
\sum_{s\in S}
\sum_{q\in Q}
A_h(s,q)
\]

Raw span→Q mass сохранять diagnostic only.

---

# 20. Bidirectional feature

Не усреднять сырые Q→S и S→Q величины напрямую: их масштабы могут отличаться.

Primary:

1. вычислить `Q→span enrichment`;
2. вычислить `span→Q density`;
3. нормировать каждое направление внутри shortlist;
4. затем:

\[
A_h^{bi}
=
\frac{
Z_q(E_h^{Q\rightarrow S})
+
Z_q(D_h^{S\rightarrow Q})
}{2}
\]

Это делает два направления сопоставимыми.

---

# 21. Что сохранять из attention

Raw tensors на диск не сохранять.

Для каждого:

```text
(question_id, chunk_id, layer, head)
```

сохранить компактно:

```text
q_to_span_mass_raw
q_to_span_density
q_to_span_enrichment
span_to_q_mass_raw
span_to_q_density
bidirectional_feature
entropy_q_to_context
valid_span
span_length_tokens
context_length_tokens
```

---

# 22. Invalid-span policy

Для final runtime candidate invalid predicted span не удаляется.

Но для **head discovery** invalid-span negatives нельзя автоматически превращать в `attention=0`, иначе head selection может научиться отличать:

```text
valid extraction
vs
invalid extraction
```

вместо evidence quality.

Поэтому primary TRAIN head-selection использует только пары, где:

```text
positive valid_span = true
negative valid_span = true
```

Invalid-span HN/easy:
- учитываются только в отдельной диагностике;
- не участвуют в primary contrastive head score.

---

# 23. Phase C — TRAIN candidate sets

В outer TRAIN shortlist-union:

## Positive

```text
source_label = robust_positive
valid_span = true
frozen QA Token F1 >= 80
```

Token F1 считается тем же TRAIN-only gold handling, что в текущем pipeline.

## Negative

Primary:

```text
training_eligible HN
valid_span = true
находится в B0 top-5 union
```

Ambiguous/GSRN:

```text
не использовать как negative labels.
```

---

# 24. Почему positives должны быть reader-easy

Attention анализируется относительно **predicted span**.

Если frozen QA на positive chunk извлёк неправильный span, то высокий/низкий Q↔span attention уже плохо связан с корректным evidence.

Поэтому primary head discovery использует only:

```text
positive + reader-success.
```

---

# 25. Feasibility audit

До выбора heads на каждом outer TRAIN посчитать:

```text
n_questions_with_valid_pos_and_HN
n_unique_positive_chunks
n_unique_HN_chunks
```

Primary minimum:

```text
MIN_PAIRED_QUESTIONS = 30
```

Если меньше:

```text
HN-only head discovery for this run = insufficient
```

и применяется заранее заданный fallback.

---

# 26. Fallback для недостаточного HN coverage

Fallback не должен брать все easy negatives случайно.

Использовать:

```text
high-ranked easy_negative
из того же B0 top-5 TRAIN-union
valid_span = true
```

Негативный набор:

```text
HN + high-ranked easy
```

при этом:
- HN имеет приоритет;
- exact composition сохраняется в артефактах;
- run помечается `negative_fallback=true`.

Если даже после fallback:

```text
paired questions < 30
```

то attention head discovery этого run считается ненадёжным и:

```text
δ forced to 0
```

на CAL/EVAL.

---

# 27. Question-balanced contrastive score

Для каждого question `q` и head `h`:

\[
R_{h,q}
=
mean_{p\in P_q,n\in N_q}
\left[
I(A_h(p)>A_h(n))
+
0.5I(A_h(p)=A_h(n))
\right]
\]

где `A_h` = primary bidirectional feature.

Затем:

\[
R_h = mean_q R_{h,q}
\]

Каждый вопрос имеет одинаковый вес.

---

# 28. Seed deduplication в head discovery

B0 top-5 различается между seeds.

Для head discovery:

```text
union top-5 across seeds
→ deduplicate by (question_id, chunk_id)
```

Один и тот же chunk не должен иметь вес 2–3 раза только потому, что его выбрали несколько B0 seeds.

Для final EVAL ranking, наоборот:

```text
каждый seed использует свой собственный top-5
```

как и V4.1.

---

# 29. Head stability внутри outer TRAIN

Outer TRAIN состоит из трёх document folds.

Для каждой головы считать `R_h`:
- по каждому TRAIN fold отдельно;
- по всем TRAIN folds вместе.

Direction-stable head:

```text
R_h > 0.5
минимум в 2 из 3 TRAIN folds.
```

Сохранять:
```text
overall R_h
per-fold R_h
coverage
```

---

# 30. Выбор числа heads N: только TRAIN

В предыдущей версии `N` и `δ` одновременно выбирались на CAL.

Это создаёт лишнюю model-selection freedom.

Исправление:

```text
N выбирается на TRAIN
δ выбирается на CAL
```

---

# 31. Inner-TRAIN выбор N

Grid:

```text
N ∈ {1,2,4,8}
```

Для каждого N провести leave-one-TRAIN-fold-out:

```text
2 TRAIN folds:
    rank stable heads

held-out TRAIN fold:
    оценить aggregate selected-head pairwise positive>negative
```

Повторить 3 раза.

Выбрать N с максимальным mean inner pairwise.

Tie-break:

```text
меньший N
```

После этого на всех 3 TRAIN folds:
- заново вычислить head ranking;
- взять top-N;
- freeze head set.

CAL не участвует в выборе N.

---

# 32. Aggregation selected heads

После выбора top-N:

\[
A_{raw}(c)
=
\frac{1}{N}
\sum_{h\in H_N}
A_h^{bi}(c)
\]

Primary — uniform mean.

Не обучать отдельный вес для каждого head.

---

# 33. Shortlist-relative attention score

Для каждого `(question, seed)`:

\[
A_{rel}(c_i)
=
\frac{
A_{raw}(c_i)-\mu_q
}{
\sigma_q+\epsilon
}
\]

где `μq, σq` считаются только по текущим 5 candidates.

Если:

```text
σq < epsilon
```

то:

```text
A_rel = 0
```

для всех candidates.

---

# 34. Phase D — ATT1 primary fusion

\[
S_{ATT1}
=
Z_q(S_{V6})
+
\delta A_{rel}
\]

Grid:

```text
δ ∈ {0, 0.1, 0.25, 0.5, 1.0}
```

Теперь CAL выбирает **только δ**.

Это существенно снижает риск переоптимизации на ~90 CAL questions.

---

# 35. CAL selection ATT1

Для каждого outer run:

1. V6 per-run weights frozen из V4.1.
2. Head set + N frozen из TRAIN.
3. CAL перебирает только `δ`.
4. Safeguard:

```text
HN@1_ATT
<=
HN@1_V6_CAL + 0.02
```

5. Выбрать max mean CAL Token F1 across seeds.
6. Tie-break:
   - меньший `δ`;
   - меньший HN@1.

---

# 36. Почему δ=0 обязателен

Если attention не добавляет value:

```text
δ = 0
```

ATT1 точно возвращается к V6.

Regression invariant:

```text
ATT1(δ=0)
==
V6
```

по ranking и QA outcome.

---

# 37. EVAL ATT1

На EVAL frozen:

```text
head set
N
δ
V6 weights
```

Merge только EVAL частей пяти outer runs:

```text
true OOF ATT1
```

Primary comparison:

```text
ATT1 vs matching V6-K5 OOF
```

---

# 38. Primary statistical hypothesis

Единственная primary attention hypothesis V4.2:

\[
H_0:
\Delta F1(ATT1-V6)\le 0
\]

против:

\[
H_1:
\Delta F1(ATT1-V6)>0
\]

Оценка:

```text
paired clustered bootstrap
cluster = document_id
n_boot >= 5000
```

---

# 39. Иерархия secondary attention tests

Чтобы не превратить V4.2 в широкий post-hoc search:

1. ATT1 — primary.
2. ATT2-IDF тестируется как improvement **только если ATT1 прошёл primary gate**.
3. Если ATT1 не прошёл, ATT2 можно посчитать exploratory, но он не становится confirmatory winner в этой же версии протокола.
4. ATT3-veto — safety/diagnostic branch, не новая primary гипотеза.

---

# 40. Phase E — shortlist-IDF ATT2

Термин:

```text
shortlist-IDF
```

а не `cross-document IDF`, потому что все top-5 chunks относятся к одному документу.

Для каждого `(question, seed)`:

\[
df_q(t)
=
\#\{candidate\ chunks\ из\ top5,\ содержащих\ normalized\ word\ t\}
\]

\[
idf_q(t)
=
\log\frac{K+1}{df_q(t)+1}+1
\]

---

# 41. Token-to-word mapping для IDF

RoBERTa использует subword tokens.

Поэтому:
1. восстановить mapping subword → original word по offsets;
2. IDF считать на уровне normalized words;
3. все subwords одного word получают одинаковый `idf(word)`.

Не считать каждый BPE-piece отдельным документным термином.

---

# 42. IDF-weighted Q→span feature

IDF применяется только к **context/span destination tokens**.

\[
M_{h,IDF}^{Q\rightarrow S}
=
\frac{1}{|Q|}
\sum_{q\in Q}
\sum_{s\in S}
A_h(q,s)\,idf(word(s))
\]

Затем выполнить ту же поправку на span length / context length и shortlist-normalization.

Для `span→Q` IDF не применять:
- question одинаков для всех candidates;
- query-token IDF не даёт нужной cross-candidate специфики.

Bidirectional ATT2:

```text
IDF-adjusted Q→span component
+
unweighted span→Q component
```

после отдельной shortlist-normalization направлений.

---

# 43. ATT2

\[
S_{ATT2}
=
Z_q(S_{V6})
+
\delta A_{IDF,rel}
\]

Head set и N:
```text
те же TRAIN-selected,
что ATT1.
```

На CAL можно выбрать только новый `δ_IDF` из той же frozen grid.

Не выбирать заново heads/N под ATT2.

---

# 44. Phase F — ATT3 HN-veto

ATT3 — secondary safety branch.

Идея:

```text
V6 считает candidate сильным,
но selected-head alignment аномально низкий
→ мягкий штраф.
```

\[
Penalty_i
=
\lambda_v\max(0,\tau-A_{rel,i})
\]

\[
S_{ATT3}
=
Z_q(S_{V6})
-
Penalty_i
\]

Grid:

```text
τ ∈ {-1.0,-0.5,0.0}
λ_v ∈ {0.1,0.25,0.5}
```

CAL only.

ATT3 особенно анализируется, если ATT1/ATT2 повышают HN@1.

---

# 45. Attention diagnostics не должны использовать GSRN как supervision

GSRN/ambiguous — главный runtime distractor, но source labels там не дают честного binary target-status supervision.

Поэтому:

```text
GSRN никогда не используется
при выборе heads / N / δ.
```

Он используется только:
- в EVAL diagnostics;
- в transition analysis.

---

# 46. Aspect diagnostics

Считать отдельно:

```text
task
method
material
metric
```

Особенно `method`, потому что старая attention-only ветка давала там сильный, но HN-опасный exploratory signal.

Не делать aspect-specific weights в V4.2.

---

# 47. Metrics

Primary:

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

Attention-specific:

```text
selected head IDs
selected layers
N selected
δ selected
TRAIN R_pos>HN
inner-TRAIN R_pos>HN
CAL pos>HN diagnostic
invalid-span rate
attention variance inside top-5
```

---

# 48. Seed handling

Для system metrics:

```text
candidate(question_id, seed)
vs
reference(question_id, same seed)
```

Затем:
1. delta per seed;
2. mean delta per question;
3. bootstrap by document.

Seeds не считать независимыми samples.

---

# 49. Head stability across outer runs

После OOF:

```text
Jaccard(top-N head sets)
head recurrence count
layer recurrence
```

Это mechanistic diagnostic.

Сильная повторяемость heads:
```text
поддерживает устойчивый task-specific signal.
```

Полностью разные head sets:
```text
указывает на нестабильность / data dependence.
```

---

# 50. Transition analysis

V6 → ATT:

```text
positive → positive
positive → HN
positive → ambiguous
HN → positive
ambiguous → positive
GSRN → positive
```

Главный desirable transition:

```text
GSRN → positive
```

при минимальном:

```text
positive → HN.
```

---

# 51. Primary success gate

ATT1 считается подтверждённо полезным, если:

```text
ΔToken F1 vs V6 > 0
95% CI excludes 0
HN safeguard passed
```

Дополнительно:
- gain наблюдается более чем на одном seed;
- нет технических invariant failures.

---

# 52. Weak positive

Если:

```text
ΔF1 > 0
CI включает 0
HN safeguard passed
```

то:

```text
attention = exploratory promising
```

но не final candidate для locked test.

---

# 53. Negative result

Если:
- `δ=0` выбирается в большинстве outer CAL;
- OOF ATT1 ≈ V6;
- или HN заметно ухудшается,

то primary attention hypothesis не подтверждается.

ATT2 можно сохранить как exploratory literature-driven sensitivity, но не объявлять новым подтверждённым winner в этой же protocol version.

---

# 54. V6-Core interpretation

CORE1/CORE2 — отдельный вопрос:

> можно ли сделать reader-agreement проще и устойчивее?

Если CORE проходит non-inferiority margin `−0.5 F1` и имеет более стабильные параметры:
```text
можно выбрать CORE как future engineering baseline.
```

Но это не доказывает attention.

---

# 55. K=10

В V4.2:

```text
не запускать primary attention на K10.
```

Если ATT1 позже подтвердится:
```text
K10 + attention
```
оформляется отдельным amendment / следующим экспериментом.

---

# 56. U80 utility

Не смешивать с V4.2.

R5/U80 остаётся отдельной secondary research branch после решения вопроса:
```text
V6 vs V6+attention.
```

---

# 57. Efficiency

Измерить реальную live cost, потому что V4.1 работал по cache.

Для:
```text
V6-K5
V6-K5 + attention extraction
```

измерить:

```text
wall latency / question
latency / candidate
peak GPU
peak RAM
tokens processed
```

---

# 58. Attention backend cost

Если `output_attentions=True` отключает flash/SDPA и переводит модель на eager attention:
- это фиксировать отдельно;
- не приписывать overhead только compact feature computation.

Runtime report должен разделять:

```text
base QA forward
attention-enabled QA forward
feature reduction overhead
```

---

# 59. Compact implementation

Raw attention tensors не писать на диск.

Внутри forward:

```text
raw attentions
→ token masks
→ per-head compact features
→ detach/cpu compact features
→ free raw tensors
```

---

# 60. Cache keys

```text
question_id
chunk_id
qa_model_hash
tokenizer_hash
attention_backend
attention_feature_version
```

Head-selection artifact key:

```text
outer_run
train_split_hash
candidate_union_hash
feature_version
```

---

# 61. Regression tests

Обязательные:

1. logits identity with/without attention within tolerance;
2. predicted span identity;
3. B0 top-5 identity;
4. TRAIN seed-union dedup works;
5. no EVAL ids in TRAIN head discovery;
6. invalid-span HN excluded from primary head score;
7. `δ=0` reproduces V6 exactly;
8. zero-variance shortlist → `A_rel=0`, no NaN;
9. span-length synthetic test confirms density/enrichment is not trivially proportional to `|S|`;
10. special/pad tokens excluded from Q/C/S masks.

---

# 62. Artifacts

```text
artifacts/qa_chunk_selection_v4_2/

contract.json

reference/
    v6_k5_v4_1/

regression/
    logits_identity.json
    span_identity.json
    top5_identity.json
    seed_union_dedup.json
    span_length_bias_test.json

core/
    run0/
    ...
    oof_core1.jsonl
    oof_core2.jsonl

attention_cache/
    compact_features.jsonl
    feature_contract.json

head_selection/
    run0/
        feasibility.json
        inner_n_selection.json
        head_scores.json
        selected_heads.json
    ...

att1/
    run0/
    ...
    oof_predictions.jsonl
    metrics.json
    bootstrap_vs_v6.json

att2_idf/
    ...

att3_veto/
    ...

analysis/
    head_stability.json
    transitions.json
    aspect_metrics.json
    efficiency.json

reports/
    results_v4_2.md
```

---

# 63. Contract до запуска

Зафиксировать:

```text
K = 5
seeds = [42,43,44]

READER_EASY_THRESHOLD = 80
MIN_PAIRED_QUESTIONS = 30

N_HEAD_GRID = [1,2,4,8]
DELTA_GRID = [0,0.1,0.25,0.5,1.0]

CORE_MAJORITY_GRID = [0.25,0.5,1.0,2.0]
CORE_TOKEN_GRID = [0,0.25,0.5,1.0]

HN_EPSILON = 0.02
BOOTSTRAP = 5000

primary attention feature =
    bidirectional(
        Q→span enrichment,
        span→Q density
    )

head selection universe =
    unique TRAIN union of B0 top-5 across seeds

primary negative =
    valid-span training-eligible HN

fallback negative =
    valid-span high-ranked easy from same shortlist-union
```

После первого outer EVAL contract не менять.

---

# 64. Рекомендуемый порядок запуска

```text
STEP 1
reuse V4.1 folds, B0 top-5, V6 per-run weights

STEP 2
run attention backend/logits/span regression

STEP 3
extract compact attention features
for unique TRAIN/CAL/EVAL top-5 candidates

STEP 4
for each outer TRAIN:
    build seed-deduplicated top-5 union
    audit valid reader-easy positives and HN
    select stable heads
    choose N by inner-TRAIN folds

STEP 5
CAL:
    choose δ only
    apply HN safeguard

STEP 6
EVAL:
    frozen ATT1

STEP 7
merge true 5-fold OOF
ATT1 vs V6-K5

STEP 8
run CORE simplification branch

STEP 9
if ATT1 passes primary gate:
    run ATT2 shortlist-IDF

STEP 10
if HN problem appears:
    run ATT3 veto

STEP 11
head stability / aspect / transition / efficiency analysis

STEP 12
keep INTERNAL_TEST locked
```

---

# 65. Короткая инструкция агенту

```text
DO NOT retune V6.

Use exact V4.1 per-run V6 weights and each seed's exact B0 top-5.

For TRAIN head discovery:
    union top-5 across seeds
    deduplicate by (question_id, chunk_id)

Enable output_attentions on the same frozen RoBERTa QA.
Verify logits and predicted spans do not change.

Mask tokens explicitly:
    question content
    context content
    predicted context span
    no special/pad tokens.

Do NOT use raw Q→span mass as primary:
    correct for span length.
Primary head feature:
    Q→span enrichment
    +
    span→Q density,
    normalized separately within shortlist.

Head discovery:
    reader-easy original positives (TRAIN F1>=80, valid span)
    vs
    valid-span training-eligible HN
    from the same TRAIN B0-top5 union.

Do not use ambiguous/GSRN as negative supervision.

Choose N on TRAIN using inner-fold pairwise validation.
Choose only δ on CAL.
Evaluate frozen ATT1 on EVAL.

Primary test:
    ATT1 vs V6-K5.

ATT2 shortlist-IDF is hierarchical secondary.
ATT3 is HN-safety diagnostic.

If ATT1 does not beat V6:
    do not turn secondary branches into a post-hoc winner.
Keep INTERNAL_TEST locked.
```

---

# 66. Главная исследовательская формулировка

> После подтверждения reader-centered reranking в V4.1 проверить, содержит ли selected-head self-attention frozen extractive reader независимый дополнительный evidence-signal, если головы выбираются контрастно только на TRAIN, attention анализируется относительно predicted answer span с поправкой на длину span, а итоговый score используется исключительно как shortlist-relative correction поверх V6.

---

# 67. Литература

1. Zhang W., Yin F., Yen H., Chen D., Ye X. (2025). **Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking.**
2. Tran L., Li Y., Florian R., Sun W. (2025). **Contrastive Retrieval Heads Improve Attention-Based Re-Ranking.**
3. Tian Y., Mo F., Zhang W., Qi Y., Nie J.-Y. (2026). **ReAttn: Improving Attention-based Re-ranking via Attention Re-weighting.**
4. Iyer S., Min S., Mehdad Y., Yih W.-t. (2021). **RECONSIDER: Improved Re-Ranking using Span-Focused Cross-Attention for Open Domain Question Answering.**
5. Xu P., Liang D., Huang Z., Xiang B. (2021). **Attention-guided Generative Models for Extractive Question Answering.**
6. Li Y., Li J., Yu M., Ding G., Chen Y., Lin Z., Zhang W., Zhou J. (2026). **Query-focused and Memory-aware Reranker for Long Context Processing.**
