# Новый эксперимент V4: автоматическое развитие chunk selection без ручной разметки

## 0. Назначение документа

Этот документ задаёт новый экспериментальный цикл после E0 → V2 → V3 → QA-silver v1/v2 → D4/D5.

Цель нового цикла:

> улучшить выбор evidence-чанка и итоговый extractive QA **без ручной разметки**, без генеративной teacher-модели и без открытия `INTERNAL_TEST`.

Новый цикл не продолжает неудачный QA-silver Stage A как источник `target_evidence / target_mismatch`.

Основной принцип:

```text
не пытаться автоматически присвоить каждому ambiguous жёсткий TE/TM label;
использовать уже доступные сигналы:
    B0 ranking;
    original positives / HN / easy;
    frozen QA;
    document-level consistency;
    cross-chunk agreement;
    multiple independent evidence;
    automatically constructed synthetic controls.
```

---

# 1. Что уже установлено предыдущими экспериментами

Текущий live baseline:

```text
V3-B0
Token F1 ≈ 27.36
Hit@1 ≈ 36.72%
Hit@5 ≈ 79.03%
Hit@10 ≈ 92.20%
HN@1 ≈ 4.86%
Ambiguous@1 ≈ 50.85%
GSRN@1 ≈ 33.92%
F1 | selected positive ≈ 61.70
```

Best-positive oracle:

```text
Token F1 ≈ 73.01
```

Главный bottleneck:

```text
не top-K recall,
а перестановка кандидатов внутри уже хорошего shortlist.
```

Следовательно, новый эксперимент концентрируется преимущественно на:

```text
B0 top-5 / top-10
→ более информативный second-stage selection / aggregation
```

а не на новом full-document retriever.

QA-silver v1/v2 показал:

```text
raw QA score / margin
не разделяет достаточно хорошо:
positive
vs
ambiguous / HN / easy.
```

Поэтому в V4 запрещено возвращаться к идее:

```text
QA score threshold
→ автоматический TE/TM label
```

как основной supervision.

---

# 2. Научная логика нового цикла

Эксперимент опирается на несколько направлений литературы, но важно различать:
- что прямо следует из литературы;
- что является новой рабочей гипотезой V4.

## L0. KeyBERT как компактное представление текста

KeyBERT извлекает ключевые слова и фразы по близости их эмбеддингов к представлению текста. В V4 это используется **не как доказательство target evidence**, а как дешёвый способ построить компактный профиль содержания документа.

Рабочая гипотеза V4:

```text
candidate chunk,
который согласуется с остальным документом,
может быть более вероятным evidence текущей статьи,
чем тематически правдоподобный, но локально изолированный distractor.
```

Эта гипотеза должна проверяться экспериментально.

## L1. Hard-negative mining и denoising

RocketQA поддерживает использование:
- hard negatives;
- denoising;
- data augmentation;
- защиты от unlabeled positives.

Следствие для V4:

```text
ambiguous нельзя автоматически считать negative;
strong negatives должны иметь надёжное происхождение.
```

## L2. Document-level + passage-level retrieval

Dense Hierarchical Retrieval показывает пользу совместного учёта document-level и passage-level semantics, а также специальных in-document negatives.

Для V4 это мотивирует `document compatibility` signal, но **не означает**, что KeyBERT-профиль сам по себе эквивалентен DHR.

## L3. Multiple valid evidence / false negatives

Retrieval Marginalization показывает, что:
- правильный ответ может поддерживаться несколькими evidence;
- не все полезные contexts размечены positive;
- жёсткое наказание unlabeled passages создаёт false-negative noise.

Поэтому:

```text
ambiguous = unlabeled,
а не strong negative.
```

## L4. QA-informed retrieval data augmentation

Ferguson et al. показывают, что полезные retrieval contexts можно автоматически находить по тому, помогают ли они downstream QA получить правильный ответ.

Это мотивирует новую версию 3:
- не повторять V3-B3;
- использовать frozen reader как TRAIN-only источник `reader-utility` для поиска дополнительных **полезных contexts**, не называя их `target evidence`.

## L5. Evidence aggregation

Evidence Aggregation for Answer Re-Ranking мотивирует:
- strength-based aggregation;
- coverage-based aggregation;
- использование нескольких passages.

Это напрямую относится к версии 2.

## L6. Global normalization

Multi-passage BERT показывает, что independently scored passages могут быть плохо сопоставимы.

Важно:

> простой post-hoc `softmax` над frozen scores не воспроизводит Multi-passage BERT и не меняет ранжирование, если применяется к одному scalar score монотонно.

Поэтому версия 5 использует обучаемый лёгкий cross-passage calibration head над frozen признаками.

## L7. Reader-centered selection

Reader-Centered Passage Selection мотивирует использование:
- predicted answers;
- agreement между candidates;
- reader-oriented passage selection.

Это относится к версии 6.

## L8. High-recall → close-false-positive reranking

RECONSIDER мотивирован ситуацией:
- высокий recall среди top candidates;
- существенно более низкий top-1 result;
- необходимость reranking близких false positives.

V7 является **RECONSIDER-inspired**, а не буквальным воспроизведением RECONSIDER.

---

# 3. Общие ограничения эксперимента

Во всех версиях V4:

```text
INTERNAL_TEST закрыт;
source_label не переписывается;
gold не используется в runtime ranking;
old E0/V2/V3/V3.1 artifacts read-only;
новые outer folds не создаются;
те же document-level 5 folds;
seeds 42/43/44 для trainable моделей;
B0 остаётся reference.
```

Новые результаты считаются:

```text
OOF model-selection evidence
```

а не confirmatory test.

---

# 4. Данные и frozen CV

Primary unlocked pool:

```text
453 answerable document–aspect
240 documents
```

Использовать существующие 5 grouped folds по `document_id`.

В каждом outer run:

```text
3 folds TRAIN
1 fold CALIBRATION
1 fold EVAL
```

Никакие hyperparameters не выбираются по EVAL.

---

# 5. Общий reference B0

Текущий подтверждённый live baseline — именно V3-B0 из завершённого V3:

```text
MiniLM Cross-Encoder
question-balanced weighted BCE
max_epochs = 5
patience = 2
same negative recipe
same frozen CV
seeds = 42, 43, 44
```

Его нельзя молча заменить будущим `B0-max8`, потому что V3.1 ranking с `max8` не был запущен.

Поэтому в V4:

```text
B0-live = завершённый V3-B0 (max5)
```

обязательно остаётся историческим reference.

Если `B0-max8` уже был заранее зафиксирован отдельным письменным протоколом, его разрешено прогнать как:

```text
B0-extended
```

но:
- не выбирать `max5` vs `max8` по EVAL post-hoc;
- не переписывать исторический B0;
- все V4 conclusions показывать как минимум относительно `B0-live`;
- если `B0-extended` становится upstream для новых веток, это должно быть зафиксировано **до** запуска этих веток.

---

# 6. Общие shortlist

Основные candidate sets:

```text
K ∈ {5, 10}
```

`K` выбирается только по CALIBRATION либо фиксируется заранее для конкретной ветки.

Важно:

```text
если система только переставляет уже готовый B0 top-K,
то Hit@K этого же K не меняется.
```

Например, при reranking внутри `B0 top-5`:

```text
Hit@5_after = Hit@5_B0
```

по определению, если состав shortlist не изменён.

Поэтому главные ranking-метрики second-stage reranking:

```text
Hit@1
MRR / rank of first original positive
Token F1 downstream
```

а Hit@5/Hit@10 остаются upstream recall diagnostics.

## 6.1. Cross-fitted shortlist

Все trainable second-stage модели получают TRAIN candidates только из:

```text
out-of-inner-fold B0 predictions
```

с grouped split по `document_id`.

Нельзя обучать second-stage model на in-sample B0 shortlist.

## 6.2. Fair seed policy

Для deterministic downstream веток желательно применять одну и ту же процедуру отдельно к:

```text
B0 seed 42
B0 seed 43
B0 seed 44
```

и затем усреднять question-level outcome across seeds.

Если по вычислительным причинам используется только один B0 seed, он фиксируется заранее и сравнение выполняется только с **matching B0 seed**, а не с pooled B0.

---

# 7. Общие метрики

## 7.1. Primary

```text
Token F1
```

по original SciREX gold.

## 7.2. Secondary QA

```text
EM
Char F1
ROUGE-L F1
```

`BERTScore` — только offline secondary analysis для финальных shortlisted systems, если он уже есть в pipeline. Не использовать его для model selection.

## 7.3. Метрики chunk-selecting systems

Для V1 / V3 / V5 / V6 / V7, когда итогом является выбранный chunk:

```text
Hit@1
MRR
HN@1
Ambiguous@1
GSRN@1
easy_negative@1
pairwise_clean
F1 | selected original positive
```

Hit@5/10 показывать как upstream B0 recall, если candidate set не меняется.

## 7.4. Метрики answer-aggregation systems

Для V2 финальный объект — **answer**, а не обязательно один chunk.

Поэтому нельзя механически приписывать ему обычный `Hit@1`.

Обязательны:

```text
Token F1
EM
final_answer_support_count
n_independent_source_clusters
final_answer_has_original_positive_support
GoldMatchFromNonOriginalPositive
```

Если нужен `selected_source_chunk`, заранее зафиксировать deterministic provenance rule, например:

```text
highest-weight supporting chunk of winning answer
```

и только после этого считать source-label diagnostics.

## 7.5. Reader-conditioned diagnostics

Для chunk-selecting systems:

```text
F1 | selected original positive
reader_easy / medium / hard / failed
```

Для answer aggregation не смешивать эти значения с обычным `F1 | selected positive`.

## 7.6. Efficiency

```text
latency / document–aspect
peak GPU
peak RAM
reader calls
reader tokens
KeyBERT document-preprocessing time
KeyBERT amortized cost / aspect
additional encoder calls
cache hit rate
```

Для нового unseen document KeyBERT extraction не считается «бесплатным offline cache»: отдельно показывать one-time preprocessing cost и amortized cost.

---

# 8. Общие safeguards

До EVAL зафиксировать:

```text
epsilon_HN
epsilon_pairwise
delta_F1_min
runtime_ratio_max
memory_ratio_max
```

Но safeguards применяются только к сопоставимым системам.

## Для chunk rerankers

Обязательны:

```text
HN@1_candidate <= HN@1_B0 + epsilon_HN
pairwise_clean_candidate >= pairwise_clean_B0 - epsilon_pairwise
```

## Для answer aggregation

`HN@1` может быть не определён без единственного selected chunk.

Тогда использовать:

```text
wrong-source support diagnostics
final_answer_has_original_positive_support
GoldMatchFromNonOriginalPositive
```

и не создавать фиктивную `HN@1` метрику.

Главный общий safeguard:

```text
не улучшать Token F1 за счёт очевидного разрушения source-selection behavior.
```

---

# 9. Статистическое сравнение

Для основных OOF comparisons:

```text
paired clustered bootstrap
cluster = document_id
>= 5000 replications
```

Для 3-seed end-to-end systems:

```text
1. outcome для каждого (document_id, question, seed);
2. mean across seeds внутри question;
3. bootstrap document_id.
```

Seeds не являются независимыми наблюдениями.

Для deterministic branch поверх трёх B0 seeds использовать ту же схему.

Если branch запущен только на одном заранее фиксированном B0 seed:

```text
сравнивать только с matching B0 seed.
```

Выводить:

```text
mean delta
95% CI
win / lose / tie
```

Все семь primary variants V1–V7 объявляются заранее этим протоколом. OOF EVAL используется для их model-selection comparison, но **не** для изобретения новых feature families в той же V4.

---

# 10. Версия 1 — KeyBERT document compatibility без структуры секций

## 10.1. Гипотеза

B0 оценивает преимущественно:

```text
question ↔ chunk relevance / answerability
```

Новый сигнал:

> candidate, согласованный с содержанием остального документа, может быть полезнее локально правдоподобного distractor.

Это **рабочая гипотеза**, а не правило target-attribution.

Корректный evidence может быть уникальным и встретиться один раз, поэтому `cross-chunk support` нельзя использовать как hard filter.

## 10.2. KeyBERT backend

Зафиксировать один lightweight embedding backend до эксперимента.

Предпочтительно:
- использовать уже имеющийся совместимый frozen sentence-embedding model, если он есть;
- иначе зафиксировать один KeyBERT backend (например, `all-MiniLM-L6-v2`) и его hash/version.

Нельзя выбирать embedding backend по EVAL.

## 10.3. Per-chunk keyphrases

Для каждого chunk:

```text
KeyBERT(chunk)
→ top-N keyphrases
```

Ограниченная сетка:

```text
N ∈ {5, 10, 15}
ngram_range ∈ {(1,2), (1,3)}
MMR ∈ {off, on}
```

Чтобы не переобучиться на одном CAL:
- KeyBERT extraction hyperparameters выбирать внутри TRAIN через inner grouped CV либо фиксировать заранее;
- CAL оставлять главным образом для fusion weight / final threshold;
- EVAL не использовать.

## 10.4. Почему простого leave-one-chunk-out недостаточно

При overlapping/sliding windows соседний chunk может почти полностью повторять candidate.

Поэтому primary вариант:

```text
leave-one-source-cluster-out (LOSCO)
```

а не только `leave-one-chunk-out`.

Для candidate `c_i`:
1. определить source interval;
2. найти все chunks с overlap выше заранее заданного порога;
3. исключить **весь overlap cluster** из document profile.

Простой LOCO оставить только как абляцию.

## 10.5. LOSCO document profile

\[
K_{D\setminus G(c_i)}
=
Aggregate(
K_c:\ c \notin G(c_i)
)
\]

где `G(c_i)` — source-overlap cluster candidate.

## 10.6. Source-overlap clusters, а не gold evidence clusters

В runtime неизвестно, какие chunks являются gold-positive.

Поэтому для KeyBERT support использовать только:

```text
source-overlap clusters of all chunks
```

построенные по offsets.

Не использовать `independent gold evidence clusters` из D5 как runtime feature.

Сохранять:

```text
phrase
mean KeyBERT score
max KeyBERT score
n_supporting_chunks
n_supporting_source_clusters
corpus_IDF
```

## 10.7. IDF без leakage

`IDF across documents` рассчитывать:

```text
только на outer TRAIN documents
```

и применять frozen к CAL/EVAL.

Для финального refit IDF можно пересчитать на всём unlocked training pool.

## 10.8. Compatibility features

Для candidate:

```text
K_candidate = KeyBERT(candidate)
K_rest      = LOSCO profile
```

### K1 — phrase overlap

\[
S_{overlap}
=
\frac{
\sum_{k\in K_c\cap K_{rest}} w(k)
}{
\sum_{k\in K_c} w(k)
}
\]

### K2 — phrase embedding compatibility

Для каждой candidate phrase найти max similarity к phrases `K_rest` и агрегировать:

```text
mean(max-sim)
top-m mean(max-sim)
KeyBERT-weighted max-sim
```

### K3 — source-cluster support

\[
S_{support}
=
\sum_{k\in K_c}
w(k)\log(1+n_{source\_clusters}(k))
\]

Это мягкий признак, не обязательное условие target evidence.

### K4 — TRAIN-IDF weighted support

\[
w(k)
=
KeyBERTScore(k)\times IDF_{TRAIN}(k)
\]

### K5 — optional aspect-conditioned phrase relevance

Без секций можно оценить:

```text
aspect question embedding ↔ keyphrase embedding
```

тем же frozen KeyBERT backend.

Это отдельная абляция.

## 10.9. Reranking variants

```text
V1-A0 = B0
V1-A1 = B0 + global document profile
V1-A2 = B0 + simple LOCO
V1-A3 = B0 + LOSCO
V1-A4 = B0 + LOSCO + source-cluster support
V1-A5 = B0 + LOSCO + support + TRAIN-IDF
V1-A6 = optional + aspect-conditioned phrase relevance
```

Fusion:

\[
S(c)=s_{B0}(c)+\lambda S_{doc}(c)
\]

или небольшой linear/logistic ranker.

## 10.10. Trainable V1 head

Features:

```text
B0 score / rank
S_overlap
S_embedding
S_support
TRAIN-IDF support
candidate length
relative document position
optional aspect phrase relevance
```

Labels:

```text
positive = original robust_positive
negative = training-eligible HN + limited easy
ambiguous = masked / unlabeled
```

Cross-fitting обязательно.

## 10.11. Критерий успеха

Primary comparison:

```text
V1-A3/A4/A5 vs B0
```

Смотреть:
- Token F1;
- Hit@1;
- MRR;
- HN safeguards;
- method;
- GSRN→positive transitions.

Особенно важная абляция:

```text
LOCO vs LOSCO
```

Если LOSCO заметно слабее LOCO, прежний gain мог быть частично self-overlap artifact.

---

# 11. Версия 2 — answer agreement + evidence aggregation

## 11.1. Гипотеза

B0 top-5 содержит original positive примерно у 79% вопросов.

```text
B0 top-K
→ frozen QA
→ candidate answers
→ deterministic answer grouping
→ source-overlap dedup
→ aggregate support
→ final answer
```

## 11.2. Answer normalization

Primary:

```text
lowercase
whitespace normalization
punctuation normalization
hyphen normalization
тот же article/token normalization, что совместим с текущими QA metrics
```

Не использовать:
```text
LLM
новый semantic encoder
semantic answer clustering
```

в primary V2.

## 11.3. Source-overlap dedup

D5 audit был на gold positives. Runtime V2 должен строить clusters только по:

```text
source offsets B0 top-K candidates
```

и удалять overlapping windows без знания gold.

Называть их:

```text
source-overlap clusters
```

а не `gold evidence clusters`.

## 11.4. Aggregation scores

### A — independent source-cluster count

\[
S(a)=N_{source\_clusters}(a)
\]

### B — B0-weighted support

\[
S(a)=\sum_{c\in C_a}w_{B0}(c)
\]

### C — max + support

\[
S(a)=
\max_{c\in C_a}s_{B0}(c)
+\lambda\log(1+N_{source\_clusters}(a))
\]

### D — relative QA confidence + support

Использовать только relative/calibrated QA feature. Не считать raw logits probability.

## 11.5. Риск

Одинаковый неправильный answer может повторяться в related work / baseline и получить сильный aggregate score.

Поэтому V2 не считается attribution method.

Обязательные diagnostics:

```text
final_answer_has_original_positive_support
GoldMatchFromNonOriginalPositive
support_source_label_mix
n_source_clusters_of_winning_answer
```

## 11.6. Ablations

```text
V2-A0 = B0 top1 answer
V2-A1 = answer frequency
V2-A2 = source-cluster count
V2-A3 = B0-weighted support
V2-A4 = max B0 + source-cluster support
V2-A5 = relative QA confidence + support
```

## 11.7. Критерий успеха

Primary:

```text
Token F1 / EM > B0
```

Diagnostic subset:

```text
B0 top1 wrong
+
B0 top-K contains original positive
```

Этот subset используется для объяснения результата, но не как runtime criterion.

---

# 12. Версия 3 — reader-utility augmentation + marginalization

## 12.1. Почему исходную формулировку нужно изменить

Простая идея:

```text
multiple original positives
+
HN/easy
+
ambiguous masked
→ set-valued listwise loss
```

уже практически совпадает с V3-B3:
- B3 суммировал probability по допустимым positives;
- denominator включал labeled P/HN/easy;
- unlabeled ambiguous не должен был становиться strong negative.

Поэтому повтор такого objective **не является новой V4 гипотезой**.

## 12.2. Новая гипотеза

Использовать frozen downstream QA для автоматического поиска **reader-useful unlabeled contexts** на TRAIN.

Не утверждать:

```text
ambiguous chunk = target evidence
```

Утверждать только:

```text
этот context позволяет frozen reader получить правильный gold answer
на TRAIN.
```

Такие examples называются:

```text
utility_positive_candidate
```

а не `target_evidence`.

## 12.3. TRAIN-only utility signal

Для каждого TRAIN `(question, ambiguous_chunk)`:

```text
run frozen QA
compare predicted answer with TRAIN gold
```

Признаки:

```text
Token F1(pred, gold)
EM(pred, gold)
reader loss / span score, если корректно доступен
```

Candidate utility определяется только на TRAIN/inner-TRAIN.

Нельзя использовать EVAL gold для создания supervision.

## 12.4. Utility-positive set

\[
P_{utility}
=
P_{original}
\cup
U_{reader-useful}
\]

Но:
- `P_original` остаётся strong;
- `U_reader-useful` имеет отдельный более слабый вес;
- same-answer / wrong-ownership cases остаются риском.

Thresholds, например:

```text
F1 ∈ {0.5, 0.8, 1.0}
```

выбирать только внутри TRAIN inner CV либо фиксировать заранее.

## 12.5. Marginalized objective

\[
L_q=
-\log
\frac{
\sum_{p\in P_{original}}\exp(s_p/T)
+
\eta\sum_{u\in U_{reader-useful}}\exp(s_u/T)
}{
\sum_{p\in P_{original}}\exp(s_p/T)
+
\eta\sum_{u\in U_{reader-useful}}\exp(s_u/T)
+
\sum_{n\in N_{strong}}\exp(s_n/T)
}
\]

где:

```text
0 <= eta <= 1
N_strong = training-eligible HN + limited easy
```

Остальные ambiguous masked.

## 12.6. Альтернатива: loss-guided augmentation

Ранжировать ambiguous по тому, насколько context снижает frozen-reader QA loss, и добавлять только top utility candidates.

Отдельная ablation.

## 12.7. Ablations

```text
V3-A0 = historical V3-B3 result
V3-A1 = original B0
V3-A2 = + utility-positive candidates, fixed eta
V3-A3 = utility-weighted marginalization
V3-A4 = loss-guided top utility augmentation
```

## 12.8. Interpretation

Даже при успехе нельзя писать:

```text
мы автоматически разметили true target evidence.
```

Корректно:

> reader-utility based weak supervision помогло retrieval/reranking без ручной разметки.

## 12.9. Gate

Успех:
- Token F1 / Hit@1 растут;
- HN safeguards проходят;
- gain не объясняется одним aspect/fold;
- utility candidates добавляют новые source regions, а не overlapping copies original positives.

---

# 13. Версия 4 — controlled ownership / target-document compatibility diagnostic

## 13.1. Исправление постановки

Synthetic pair:

```text
target document A
+
positive chunk B from document B
```

нельзя корректно использовать с input:

```text
generic aspect question + chunk B
```

потому что canonical aspect question одинаков для документов и модель **не видит идентичность target document A**.

Тогда один и тот же `(question, chunk B)`:
- positive для B;
- negative для A;
- но вход модели идентичен.

Это противоречивая supervision.

Поэтому synthetic wrong-document diagnostic обязан явно давать системе **target-document anchor**.

## 13.2. Target-document anchor

Anchor строится автоматически без секций и без генерации:

```text
KeyBERT LOSCO / full-document profile
```

или другой заранее зафиксированный compact document profile.

Для synthetic candidate B anchor всегда относится к target A.

## 13.3. Controlled frozen-QA probe

QA reader должен извлекать span только из candidate chunk.

Target identity передавать, например, в question prefix:

```text
Target-document keyphrases: k1; k2; ...; kN.
Question: What method or model is used or proposed in the paper?
```

Context:

```text
candidate chunk
```

Это distribution shift относительно исходной QA-модели, поэтому ветка остаётся **diagnostic**.

## 13.4. Controls

Known target:

```text
anchor(A) + positive chunk A
```

Synthetic wrong-document:

```text
anchor(A) + positive chunk B
B != A
same aspect
```

B выбирать hard:

```text
high document-profile similarity
+
answerable-looking by frozen QA/B0
```

## 13.5. Split discipline

Для TRAIN/CAL/EVAL synthetic controls:
- source documents для pairing выбирать внутри соответствующего split;
- matching algorithm, similarity metric и random seed freeze до EVAL;
- EVAL controls не использовать для настройки probe.

## 13.6. Scores

Сравнить:

```text
canonical QA without anchor
anchored target-focused QA
anchored target/external contrast probes
KeyBERT compatibility baseline
```

Metrics:

```text
AUROC
AUPRC
pairwise target > synthetic mismatch
effect size
```

## 13.7. Что доказывает

Если signal есть:

> compact target anchor помогает отличать known own-document evidence от контролируемого чужого evidence.

Он **не доказывает** качество на naturally occurring target mismatch / GSRN.

Если signal нет:

```text
ownership-probe branch STOP
```

и не использовать его в V7.

---

# 14. Версия 5 — lightweight global cross-passage calibration

## 14.1. Исправление постановки

Для frozen scalar scores:

\[
softmax(z_i)
\]

является монотонным преобразованием и **не меняет порядок** candidates.

Поэтому:

```text
independent score
→ post-hoc softmax
```

не может сам по себе улучшить Hit@1.

Multi-passage BERT получает пользу из global normalization **в процессе обучения**, а не из косметического softmax после frozen inference.

## 14.2. Гипотеза V5

Не fine-tune frozen QA.

Обучить маленький cross-passage calibration/ranking head над frozen features, который видит relative candidate context внутри B0 top-K.

## 14.3. Frozen features

```text
B0 score
B0 rank
QA best-span score
QA top1-top2 margin
relative QA score within shortlist
relative B0 score within shortlist
span length
span position
candidate length
```

## 14.4. Trainable global head

\[
z_i=f_\theta(x_i)
\]

\[
P(i|q,C)
=
\frac{\exp z_i}{\sum_{j\in C}\exp z_j}
\]

Train loss — multiple-positive set-valued objective по original positives:

```text
positive = original robust_positive
negative = training-eligible HN + limited easy
ambiguous = masked
```

Head:

```text
linear / logistic
или
small MLP
```

без fine-tuning frozen QA.

## 14.5. Ablations

```text
V5-A0 = independent frozen QA score
V5-A1 = post-hoc softmax sanity check
        # expected identical ranking
V5-A2 = B0 + relative frozen QA features, fixed formula
V5-A3 = learned linear global head
V5-A4 = small MLP global head
```

## 14.6. Cross-fitting

Head обучается только на cross-fitted TRAIN shortlist.

CAL:
```text
architecture / regularization
```

EVAL:
```text
frozen head only
```

## 14.7. Критерий успеха

```text
Hit@1
MRR
Token F1
HN safeguards
```

Не интерпретировать успех как доказательство того, что raw QA logits стали вероятностями.

---

# 15. Версия 6 — reader-centered reranking and answer consistency

## 15.1. Гипотеза

Relevant passage ≠ passage, наиболее полезный frozen reader.

Использовать согласованность predicted answers внутри B0 top-K.

## 15.2. Pipeline

```text
B0 top-K
→ frozen QA
→ predicted answers
→ deterministic answer similarity / grouping
→ reader-centered features
→ rerank passages
```

## 15.3. Primary answer agreement без новой модели

Primary similarity:

```text
exact normalized match
Token F1 between predicted answers
Char F1 between predicted answers
```

Если V1 уже использует KeyBERT embedding backend, разрешена отдельная ablation:

```text
answer embedding similarity
```

тем же frozen backend.

Не подключать ещё одну модель только ради V6.

## 15.4. Reader-centered features

```text
B0 score / rank
relative QA score
QA margin

exact-answer frequency
mean Token-F1 to other candidate answers
mean Char-F1 to other candidate answers
supporting source-overlap cluster count
agreement with majority normalized answer
answer uniqueness

optional V1 document-compatibility score
```

## 15.5. Non-trainable V6

\[
S(c)
=
\alpha s_{B0}(c)
+
\beta S_{answer-agreement}(c)
+
\gamma S_{source-support}(c)
\]

Weights только CAL.

## 15.6. Trainable V6

Small linear/logistic/MLP ranker.

Labels:

```text
positive = original positives
negative = HN + limited easy
ambiguous = masked
```

Cross-fitting обязательно.

## 15.7. Отличие от V2

V2:
```text
выбирает финальный ANSWER.
```

V6:
```text
выбирает финальный PASSAGE.
```

## 15.8. Риск

Несколько related-work chunks могут повторять один и тот же неправильный answer.

Обязательны:

```text
GSRN transitions
wrong-source support diagnostics
```

Комбинация с V1 — только если обе ветки отдельно полезны.

---

# 16. Версия 7 — RECONSIDER-inspired close-false-positive reranker

## 16.1. Гипотеза

```text
B0 top-K recall высокий
B0 top1 слабый
```

мотивирует reranking close false positives внутри high-recall shortlist.

V7 вдохновлён RECONSIDER, но не является его буквальным воспроизведением.

## 16.2. Candidate generation

TRAIN candidates:

```text
cross-fitted B0 top-K only
```

## 16.3. Positives

```text
original positives,
попавшие в B0 top-K
```

## 16.4. Natural strong negatives

```text
N1 = HN в B0 top-K
N2 = high-ranked easy_negative
```

Ambiguous:

```text
masked / unlabeled
```

## 16.5. Synthetic wrong-document negatives — только с target anchor

Без target-document anchor synthetic negative некорректен, потому что generic question одинаков между documents.

Synthetic branch разрешён только если input содержит compact target-document anchor, например успешный V1 KeyBERT profile.

Input:

```text
TARGET_PROFILE(A)
+
QUESTION
+
CANDIDATE_CHUNK
```

При candidate из B:

```text
TARGET_PROFILE(A)
+
QUESTION
+
POSITIVE_CHUNK(B)
```

только тогда модель видит, почему B wrong-target относительно A.

Если V1/V4 anchor не показывает signal, synthetic branch V7-A2+ закрыть.

## 16.6. Hard synthetic sampling

Для target A выбирать B:

```text
same aspect
different document
high TRAIN-safe document-profile similarity
answerable-looking
```

TRAIN synthetic sources — только outer TRAIN documents.

## 16.7. Input variants

Base:

```text
question + candidate chunk
```

для natural HN/easy control.

Anchored:

```text
target profile + question + candidate
```

для synthetic variants.

Не вставлять полный документ.

## 16.8. Span-focused frozen features

```text
predicted span
span position
relative QA score
span margin
```

Это feature augmentation, а не повтор старого SpanRanker K=3.

## 16.9. Objectives

Primary:

```text
BCE
```

Pairwise/listwise — secondary ablation после TRAIN/CAL signal anchored BCE.

## 16.10. Ablations

```text
V7-A0 = B0
V7-A1 = shortlist reranker with natural HN/easy
V7-A2 = anchored reranker + synthetic wrong-document negatives
V7-A3 = V7-A2 + V1 document-compatibility features
V7-A4 = V7-A3 + frozen QA span features
V7-A5 = pairwise anchored reranking
```

## 16.11. Interpretation

Нельзя писать:

```text
synthetic mismatch = true naturally occurring target mismatch
```

Корректно:

> synthetic cross-document controls дали автоматически известный wrong-target supervision signal.

---

# 17. D4 — оставить как вторичную ветку

D4 не входит в 7 основных версий.

Его запускать только после появления более сильного top-1 pipeline.

Причина:

```text
windowing ceiling ≈ +5 F1
после уже правильного positive,
а основной ranking gap намного больше.
```

Runtime D4:

```text
selective,
не always-on.
```

Особенно не применять blindly для:

```text
method
reader_easy.
```

---

# 18. Рекомендуемый порядок запуска

Все семь **primary** веток объявлены заранее этим документом.

Gates управляют:
- дорогими secondary ablations;
- комбинациями;
- продолжением ветки;

но не позволяют post-hoc изобретать новую V4 feature family по EVAL.

## Phase 0 — common cache

```text
freeze B0-live reference
optional predeclared B0-extended
cross-fitted B0 top-5/top-10
frozen QA outputs for union(top-K)
KeyBERT per-chunk phrases / embeddings
source-overlap clusters
normalized predicted answers
```

## Phase 1

```text
V1 KeyBERT LOSCO
V2 answer aggregation
```

## Phase 2

```text
V3 reader-utility augmentation + marginalization
```

## Phase 3

```text
V4 anchored ownership / synthetic control
```

## Phase 4

```text
V5 global calibration
V6 reader-centered reranking
```

## Phase 5

```text
V7 RECONSIDER-inspired
```

Synthetic V7 variants требуют target-document anchor signal из V1/V4.

---

# 19. Основная матрица экспериментов

| ID | Система | Trainable | Ручная разметка | Основной новый сигнал |
|---|---|---:|---:|---|
| B0 | current live reference | да | нет | answerability |
| V1 | KeyBERT LOSCO | нет / small head optional | нет | document compatibility |
| V2 | answer aggregation | нет | нет | multi-source answer support |
| V3 | reader-utility marginalization | да | нет | QA-informed weak utility positives |
| V4 | anchored ownership diagnostic | нет | нет | target-anchor contrast on synthetic controls |
| V5 | global calibration head | small head | нет | cross-passage relative frozen features |
| V6 | reader-centered passage rerank | нет / small head | нет | answer agreement |
| V7 | close-FP reranker | да | нет | high-ranked negatives + anchored synthetic controls |

Важно:

```text
V4 = diagnostic,
не production ranker сам по себе.
```

---

# 20. Основные comparisons

Primary:

```text
V1 vs B0-live
V2 vs B0-live
V3 vs B0-live
V5 vs B0-live
V6 vs B0-live
V7 vs B0-live
```

V4: primary diagnostic:

```text
known-target vs synthetic-wrong-document separation.
```

Internal ablations:

```text
V1: LOCO vs LOSCO
V1: LOSCO vs LOSCO+support
V2: raw frequency vs source-cluster-aware aggregation
V3: historical B3 vs reader-utility augmentation
V4: no-anchor vs target-anchor controls
V5: post-hoc softmax invariance vs learned calibration
V6: answer aggregation signal vs passage reranking
V7: natural HN/easy vs anchored synthetic mismatch
```

---

# 21. Gates и правила развития веток

## G1 — KeyBERT signal

На TRAIN/CAL проверить:

```text
LOSCO / support дают стабильный gain
без HN degradation.
```

## G2 — aggregation signal

V2 продолжать к комбинациям, если:

```text
Token F1/EM растут,
а winning answer не поддерживается преимущественно wrong-source candidates.
```

## G3 — reader-utility signal

V3 продолжать, если:

```text
utility augmentation улучшает CAL/OOF
относительно historical B3/B0
и не разрушает HN control.
```

## G4 — anchored ownership signal

Продолжать ownership features в V7 только если:

```text
anchor(A)+positive(A)
>
anchor(A)+positive(B)
```

устойчиво на synthetic controls.

## G5 — global calibration

Простой softmax не считается успехом.

Продолжать только если:

```text
learned lightweight head
>
frozen-score baselines
```

## G6 — reader-centered signal

Продолжать к комбинации V2/V6, если passage reranking улучшает результат сверх простого answer aggregation.

## G7 — close-FP reranker

Final candidate только если:

```text
practical Token F1 gain
+
HN safeguards
+
acceptable runtime
+
stable folds/seeds.
```

Gates для запуска **новых secondary ablations/combinations** должны опираться на TRAIN/CAL. OOF EVAL сравнивает уже зафиксированные primary variants.

---

# 22. Aspect-specific diagnostics

Обязательные разрезы:

```text
task
method
material
metric
```

Особенно:

## method

Текущий bottleneck:

```text
низкий Hit@1
высокий F1|positive
много GSRN.
```

Поэтому V1/V4/V7 наиболее интересны.

## material

Ranking уже относительно сильный.

Интереснее:

```text
V2
V5
V6
D4 later.
```

## metric

Оба bottleneck.

Проверять:

```text
ranking
+
reader normalization
+
aggregation.
```

## task

Сильный кандидат для reranking / aggregation.

---

# 23. Transition analysis

Для каждого нового reranker считать переходы относительно B0:

```text
positive → positive
positive → ambiguous
positive → HN
ambiguous → positive
GSRN → positive
GSRN → HN
easy → positive
```

Особенно важен:

```text
GSRN → positive
```

без роста:

```text
positive → HN.
```

---

# 24. Error taxonomy V4

Минимум:

```text
E1  no positive in B0 top-K
E2  positive in top-K, final system missed
E3  GSRN remains top1
E4  HN promoted
E5  easy promoted
E6  KeyBERT profile self-bias
E7  document profile too generic
E8  aggregation picks frequent wrong answer
E9  overlapping evidence double-counted
E10 global QA normalization promotes wrong span
E11 reader agreement reinforces shared wrong answer
E12 synthetic mismatch too easy
E13 synthetic mismatch distribution shift
E14 ownership probe non-separating
E15 set-valued objective collapses score spread
```

---

# 25. Synthetic wrong-document construction rules

Synthetic controls используются только в:

```text
V4 diagnostic
V7 anchored training
```

## 25.1. Обязательный target anchor

Synthetic wrong-document candidate разрешён только если model input/feature set содержит target-document identity через compact anchor.

Без anchor такой negative логически некорректен.

## 25.2. Pairing

Для target document A выбрать B:

```text
same aspect
B != A
high document-profile similarity
```

Candidate:

```text
original positive chunk B
```

## 25.3. Split discipline

Outer TRAIN:

```text
A и B только из TRAIN.
```

CAL:

```text
pairing только внутри CAL,
frozen matching rule.
```

EVAL:

```text
pairing только внутри EVAL,
frozen matching rule,
diagnostic only.
```

Не использовать EVAL documents как negative source при TRAIN.

## 25.4. Similarity source

Primary:

```text
KeyBERT document-profile similarity
```

Optional:

```text
existing frozen lightweight embedding
```

Не использовать генеративную модель.

---

# 26. Защита от слишком лёгких synthetic controls

Считать:

```text
document-profile similarity(A,B)
B0/QA answerability of chunk B
lexical overlap with aspect question
```

Synthetic mismatch должен быть:

```text
тематически близким
и
answerable-looking.
```

Hardness threshold нельзя подбирать по EVAL downstream F1.

```text
TRAIN / inner CV
→ freeze
→ CAL check
→ EVAL apply
```

Дополнительно сравнить:

```text
synthetic distribution
vs
natural B0 top-K negative distribution
```

чтобы оценить distribution shift.

---

# 27. Cross-fitting и split-safe feature construction

Обязательно для:

```text
V1 trainable head
V3
V5
V6 trainable
V7
```

TRAIN candidates upstream:

```text
inner grouped cross-fitted B0
```

TRAIN-derived corpus statistics:

```text
KeyBERT IDF
feature scaling
synthetic-hardness thresholds
utility-positive thresholds
```

вычисляются только на соответствующем TRAIN.

Per-document features, доступные в runtime из самого нового документа:

```text
KeyBERT phrases текущего документа
LOSCO profile
source-overlap clusters
```

можно вычислять для CAL/EVAL документа без labels.

Нельзя использовать:

```text
CAL/EVAL corpus-wide IDF
EVAL gold
EVAL-derived thresholds
```

---

# 28. Caching

Общий cache:

```text
B0 scores
B0 ranks
frozen QA spans/logits
KeyBERT phrases
phrase embeddings
document phrase profiles
overlap clusters
normalized answers
```

Все ключи cache должны включать:

```text
document_id
question_id
chunk_id
model_hash
config_hash
```

---

# 29. Вычислительная стоимость

## V1

KeyBERT требует embedding computation для нового документа.

Отчётность:

```text
one-time preprocessing / document
amortized preprocessing / aspect
rerank-only incremental latency
peak memory
```

Не считать experimental cache production runtime cost = 0.

## V2

```text
K frozen QA calls
answer normalization
source-overlap aggregation
```

## V3

```text
training:
    дополнительные frozen-QA utility calls на TRAIN ambiguous
runtime:
    trained ranker / примерно B0-level extra scoring
```

Offline weak-supervision cost учитывать отдельно.

## V4

```text
anchored QA probes on controlled candidates
```

Diagnostic cost.

## V5

```text
K frozen QA calls
+ lightweight global head
```

## V6

```text
K frozen QA calls
+ deterministic agreement
+ optional small ranker
```

## V7

```text
B0
+ second-stage Cross-Encoder over K
+ optional frozen QA features
+ KeyBERT anchor preprocessing, если anchored variant
```

Для всех:

```text
latency
peak GPU
peak RAM
reader tokens
encoder calls
preprocessing amortization
```

---

# 30. Expected Pareto analysis

Финальные кандидаты сравнивать по:

```text
Token F1
vs
latency
vs
GPU memory
```

Важный критерий исследования:

> небольшой прирост F1, требующий большого reader/cross-encoder budget, может быть хуже более дешёвого KeyBERT или set-valued варианта.

---

# 31. Что не делать в V4

1. Не открывать `INTERNAL_TEST`.
2. Не возвращаться к QA-silver v1/v2 Stage-A thresholds как источнику TE/TM.
3. Не мапить ambiguous/GSRN → TM.
4. Не считать KeyBERT similarity доказательством target evidence.
5. Не использовать только exact candidate exclusion при overlapping windows как primary V1; удалять весь source-overlap cluster.
6. Не считать D5 gold-positive clusters runtime feature.
7. Не считать overlapping chunks независимыми votes.
8. Не считать raw QA logits probability across passages.
9. Не считать post-hoc softmax новым ranking method: без learned relative features порядок не меняется.
10. Не использовать ambiguous как strong negative.
11. Не считать `reader-useful ambiguous` автоматически true target evidence.
12. Не создавать synthetic wrong-document negatives без target-document anchor.
13. Не использовать один generic `(question, chunk)` как positive для одного document и negative для другого, если document identity не входит во вход.
14. Не подбирать KeyBERT backend/N/MMR/IDF/weights по EVAL.
15. Не считать corpus IDF по CAL/EVAL в OOF.
16. Не изобретать новые features после просмотра EVAL в той же V4.
17. Не смешивать single-seed downstream systems с pooled 3-seed B0.
18. Не объявлять synthetic mismatch naturally occurring TM.
19. Не заменять original SciREX gold автоматическими labels.
20. Не считать KeyBERT experimental cache бесплатным для новых production documents.

---

# 32. Final model selection

После завершения заранее объявленных primary variants:

1. оставить `B0-live` в таблице независимо от новых результатов;
2. исключить системы, нарушившие applicable safeguards;
3. исключить явно нестабильные системы;
4. сравнить paired Token F1 / EM и ranking metrics;
5. учесть latency, memory и preprocessing cost;
6. V4 diagnostic не рассматривать как production winner сам по себе;
7. выбрать один final V4 pipeline;
8. зафиксировать его полностью;
9. final refit на всех unlocked training documents;
10. freeze manifest;
11. только затем открыть `INTERNAL_TEST` один раз.

OOF V4 остаётся model-selection evidence. Confirmatory conclusion — только по единственному final frozen pipeline на INTERNAL_TEST.

---

# 33. Final freeze manifest

Сохранить:

```text
dataset hash
CV split hash

B0-live config/hash
upstream B0 variant used by final system

KeyBERT backend/hash
KeyBERT N/ngram/MMR
source-overlap threshold
LOSCO rule
TRAIN-IDF rule/hash
optional aspect-phrase rule

answer normalization rule
answer source-overlap aggregation rule

QA checkpoint hash
reader-utility threshold/rule
utility weight eta

ownership anchor construction
ownership question/prefix set
synthetic pairing rule
synthetic hardness rule

global-calibration head config
reader-centered feature config
V7 second-stage config

seed policy
cross-fitting policy
statistical procedure
primary endpoint
applicable safeguards
runtime/preprocessing budget
selected final variant
```

Manifest должен позволять восстановить final pipeline без EVAL-derived ad hoc decisions.

---

# 34. Рекомендуемая последовательность решений

```text
START
↓
B0-live reference
↓
V1 KeyBERT LOSCO document compatibility
↓
V2 source-cluster-aware answer aggregation
↓
V3 reader-utility augmentation + marginalization
↓
V4 anchored ownership synthetic diagnostic
↓
V5 learned global calibration
↓
V6 reader-centered passage reranking
↓
V7 anchored close-FP reranker
↓
paired OOF comparison of predeclared primary variants
↓
Pareto: F1 / latency / memory / preprocessing
↓
select one frozen final pipeline
↓
final refit
↓
INTERNAL_TEST once
```

---

# 35. Возможные комбинации после отдельных веток

Комбинации запускать только если компоненты отдельно прошли TRAIN/CAL gate и показали OOF utility как predeclared primary variants.

## C1

```text
B0
→ V1 KeyBERT LOSCO
→ V2 answer aggregation
```

## C2

```text
B0
→ V1 KeyBERT LOSCO
→ V6 reader-centered reranking
```

## C3

```text
V3 reader-utility trained ranker
→ top-K
→ V2 aggregation
```

## C4

```text
B0
→ V7 anchored close-FP reranker
→ V2 aggregation
```

## C5

```text
B0
→ V5 global calibration
→ V6 reader-centered reranking
```

Нельзя собирать «всё лучшее» post-hoc на том же EVAL без новой frozen protocol version.

---

# 36. Практический приоритет веток

```text
1. V1 KeyBERT LOSCO
2. V2 answer aggregation
3. V3 reader-utility marginalization
4. V5 lightweight global calibration
5. V6 reader-centered
6. V4 anchored ownership diagnostic
7. V7 RECONSIDER-inspired
```

Обоснование:

```text
V1/V2:
    дешёвые и прямо используют high-recall top-K.

V3:
    впервые добавляет новый automatic supervision signal,
    а не повторяет historical B3.

V5/V6:
    используют frozen reader относительно shortlist.

V4:
    проверяет target-anchor signal
    на known-target / wrong-document benchmark.

V7:
    самая сложная и дорогая trainable ветка;
    synthetic variants имеют смысл только при наличии target anchor signal.
```

---

# 37. Основная исследовательская гипотеза V4

> качество extractive QA можно повысить без ручного `target_mismatch` overlay, если после high-recall B0 использовать дополнительные автоматически доступные сигналы: согласованность candidate с остальным документом, reader utility, независимую cross-chunk поддержку, относительное поведение frozen QA среди кандидатов и контролируемые wrong-document contrasts.

V4 **не ставит целью автоматически восстановить истинную TE/TM taxonomy для всех ambiguous**.

---

# 38. Возможный научный вклад

При положительном результате вклад можно формулировать как:

```text
resource-efficient extractive QA pipeline:
1. answerability-oriented shortlist;
2. document-consistency / reader-utility / multi-evidence refinement;
3. optional lightweight reranking.
```

Без:

```text
manual target-mismatch annotation
generative teacher
full-document LLM inference
```

Отдельный отрицательный результат:

```text
frozen QA confidence alone
не является достаточным attribution signal.
```

---

# 39. Литература и связь с вариантами

## [1] KeyBERT

Grootendorst, M. (2020).
**KeyBERT: Minimal keyword extraction with BERT.**
Zenodo.
DOI: `10.5281/zenodo.4461265`

Используется для V1 как lightweight keyphrase extraction mechanism.

Важно: гипотеза `document compatibility → better target ranking` является собственной гипотезой V4.

## [2] RocketQA

Qu, Y., Ding, Y., Liu, J. et al. (2020).
**RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering.**

Связь:
```text
hard-negative mining
denoising
unlabeled positives
data augmentation
```

## [3] Dense Hierarchical Retrieval

Hashimoto, K., Zhou, Y., Yavuz, S., Xiong, C., Yu, P. S. (2021).
**Dense Hierarchical Retrieval for Open-Domain Question Answering.**

Связь:
```text
document-level + passage-level semantics
in-document negatives
```

## [4] Retrieval Marginalization

Ni, A., Gardner, M., Dasigi, P. (2021).
**Mitigating False-Negative Contexts in Multi-document Question Answering with Retrieval Marginalization.**
EMNLP 2021.
DOI: `10.18653/v1/2021.emnlp-main.497`

## [5] QA-informed retrieval augmentation

Ferguson, J., Hajishirzi, H., Dasigi, P., Khot, T. (2022).
**Retrieval Data Augmentation Informed by Downstream Question Answering Performance.**
FEVER 2022.
DOI: `10.18653/v1/2022.fever-1.1`

Основной мотиватор новой V3.

## [6] Evidence Aggregation

Wang, S., Yu, M., Jiang, J. et al. (2017).
**Evidence Aggregation for Answer Re-Ranking in Open-Domain Question Answering.**

## [7] Multi-passage BERT

Wang, Z., Ng, P., Ma, X., Nallapati, R., Xiang, B. (2019).
**Multi-passage BERT: A Globally Normalized BERT Model for Open-domain Question Answering.**

Не трактовать как обоснование простого post-hoc softmax.

## [8] Reader-Centered Passage Selection

Xin, C., Zhou, S., Chen, X. et al. (2025).
**Aligning Retrieval with Reader Needs: Reader-Centered Passage Selection for Open-Domain Question Answering.**

## [9] RECONSIDER

Iyer, S., Min, S., Mehdad, Y., Yih, W.-t. (2021).
**RECONSIDER: Improved Re-Ranking using Span-Focused Cross-Attention for Open Domain Question Answering.**
NAACL 2021.
DOI: `10.18653/v1/2021.naacl-main.100`

V7 — adaptation/inspiration, не точная репликация метода.

---

# 40. Короткая версия исправленного эксперимента

```text
V1:
B0 top-K
→ KeyBERT per chunk
→ remove candidate's entire overlapping source cluster
→ LOSCO document profile
→ document compatibility / support
→ rerank

V2:
B0 top-K
→ frozen QA
→ deterministic answer grouping
→ source-overlap dedup
→ evidence aggregation
→ final answer

V3:
TRAIN ambiguous
→ frozen QA
→ find reader-useful contexts by TRAIN gold-answer utility
→ weak utility-positive set
→ weighted marginalization
→ ranker
# не повтор historical B3

V4:
target-document KeyBERT anchor
+
candidate chunk
→ anchored frozen-QA ownership/contrast diagnostic
→ known own-document vs synthetic wrong-document controls
# diagnostic, not natural TM proof

V5:
B0 top-K
→ frozen QA relative features
→ lightweight learned cross-passage calibration head
→ global listwise normalization
# plain post-hoc softmax = invariance control only

V6:
B0 top-K
→ frozen QA
→ exact/Token-F1/Char-F1 answer agreement
→ reader-centered passage reranking

V7:
cross-fitted B0 top-K
→ close natural negatives
+
anchored synthetic wrong-document negatives
→ RECONSIDER-inspired second-stage reranker
```

Ключевые safeguards:

```text
B0-live remains max5 historical reference;
IDF = TRAIN-only;
KeyBERT primary = LOSCO, not simple LOCO;
runtime clusters = source-overlap, not gold D5 evidence clusters;
V3 uses reader utility and does not duplicate B3;
synthetic wrong-document requires target anchor;
post-hoc softmax alone cannot improve ranking;
answer aggregation and chunk reranking use different metric sets;
INTERNAL_TEST stays locked until one final V4 pipeline is frozen.
```
