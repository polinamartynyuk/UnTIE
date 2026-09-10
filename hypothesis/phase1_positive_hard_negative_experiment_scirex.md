# Эксперимент: формирование выборки `positive / false-negative / hard-negative` для последующего поиска признаков правильного QA-чанка

## 0. Назначение документа

Этот документ описывает **первую часть эксперимента** по улучшению extractive QA на уже подготовленном aspect-level представлении **SciREX**.

На данном этапе **не требуется** тематическая кластеризация документов, построение итогового словаря признаков, изменение RoBERTa QA, обучение нового reranker или окончательная фильтрация чанков при inference.

Предварительная обработка SciREX уже выполняется отдельным этапом. Она формирует:

- один общий набор физических чанков для каждого документа;
- отдельный QA-instance для каждого аспекта:
  - `method`;
  - `task`;
  - `metric`;
  - `material`;
- один фиксированный вопрос на аспект;
- `raw_gold_answers`;
- `canonical_gold_answers`;
- `extractable_gold_answers`;
- `unextractable_gold_answers`;
- `gold_supporting_chunk_ids`;
- статус:
  - `ready`;
  - `partially_extractable`;
  - `no_extractable_gold`.

Поле `score` в основной aspect-level QA-схеме отключено и в этой Phase 1 отдельно не обрабатывается.

**Единица эксперимента теперь не просто документ, а пара:**

\[
(document\_id,\ aspect).
\]

То есть один SciREX-документ может породить до четырех независимых QA-instance, которые используют **одни и те же физические chunks**, но разные:

- `aspect`;
- `question`;
- наборы допустимых `extractable_gold_answers`;
- `gold_supporting_chunk_ids`;
- будущие `positive_chunks`;
- будущие `hard_negative_chunks`.

Цель первой части:

> **для каждого SciREX aspect-level QA-instance `документ + аспект + фиксированный вопрос + набор извлекаемых правильных ответов` построить надежно размеченный набор чанков, разделив их на positive, false-negative candidates и hard negatives, чтобы на следующем этапе искать признаки, действительно отличающие правильные чанки от сложных неправильных.**

Итогом обработки каждого `(document_id, aspect)` должны стать как минимум:

```text
document_id
aspect
question
extractable_gold_answers
chunks
positive_chunks
hard_negative_chunks
```

Дополнительно обязательно сохранить:

```text
gold_supporting_chunks
evidence_positive_qa_failed
false_negative_candidates
easy_negative_chunks
QA predictions / confidence
retrieval scores
cluster/pseudo-label diagnostics
```

Важно:

> текущая Phase 1 должна использовать **baseline source-only preprocessing (`E0`)**.  
> `external_enrichment` и `enrichment_candidates`, если они сохранены предыдущим этапом, не используются для создания positive/hard-negative labels и не вмешиваются в основной эксперимент.


# 1. Ключевая идея

Последующий исследовательский вопрос:

> **Какие лексические, семантические и внутренние признаки QA-модели систематически отличают чанки, из которых extractive QA получает правильный ответ, от тематически или семантически очень похожих чанков, из которых модель получает неправильный ответ?**

На следующем этапе такими признаками могут быть:

- ключевые слова;
- n-граммы и keyphrases;
- признаки из attention QA-модели;
- attention с учетом вопроса;
- attention относительно answer span;
- существующий `diff_score`;
- question-aware аналоги KeyBERT;
- лексическая специфичность positive относительно hard-negative;
- embedding similarity;
- позиционные и структурные признаки;
- комбинации этих сигналов.

Но **до анализа признаков необходимо построить качественную positive/hard-negative выборку**.

---

# 2. Основной научный источник

Основной источник:

**Shen, T., Liu, M., Zhou, M., Xiong, D. (2022).  
“Recovering Gold from Black Sand: Multilingual Dense Passage Retrieval with Hard and False Negative Samples.”  
EMNLP 2022.**

ACL Anthology:

https://aclanthology.org/2022.emnlp-main.730/

PDF:

https://aclanthology.org/2022.emnlp-main.730.pdf

Далее метод обозначается как **mHFN**.

Важно:

> Мы **не воспроизводим весь mHFN**.

Из него заимствуется прежде всего идея **adaptive false negative sample filter**:

```text
passage representations
        ↓
K-means
        ↓
Gap Statistic → выбор K
        ↓
Gumbel-Softmax → pseudo-label passages
        ↓
negative с pseudo-label positive
        ↓
false-negative candidate
```

Hard-negative mining и false-negative filtering необходимо строго различать.

---

# 3. Что на самом деле делают Shen et al.

## 3.1. Dense retrieval

mHFN использует bi-encoder:

\[
E_q(q)
\]

для вопроса и

\[
E_p(p)
\]

для passage.

Авторы определяют similarity как dot product:

\[
sim(q,p)=E_q(q)\cdot E_p(p).
\]

Это Section 3.1 статьи.

---

# 4. Hard negatives у Shen et al.

Это принципиальный момент:

> **K-means + Gap Statistic + Gumbel-Softmax НЕ используется авторами для поиска hard negatives.**

Hard negatives формируются отдельно.

В Section 3.2 авторы используют BM25 и выбирают **top-30 non-positive results** для каждого query.

Базовая схема:

```text
query
  ↓
BM25
  ↓
top-30 passages,
не размеченных как positive
  ↓
hard-negative pool
```

Далее mHFN дополнительно создает synthetic hard-negative representations и использует multilingual sharing, но это не требуется для нашей первой фазы.

### Вывод

Нельзя описывать источник так:

```text
K-means → hard negatives
```

Корректно:

```text
retrieval
   ↓
hard-negative candidates

и отдельно:

candidate negatives
   ↓
false-negative filter
```

---

# 5. False negatives у Shen et al.

В passage retrieval dataset passage, не размеченный как positive, не обязательно действительно нерелевантен.

Могут существовать **unlabeled relevant passages**.

Если такой passage использовать как negative в contrastive learning, модель будет искусственно отталкивать query от фактически релевантного passage.

Такой объект — `false negative`:

\[
false\ negative
=
\text{unlabeled passage, фактически релевантный query}.
\]

Это не то же самое, что hard negative.

---

# 6. Различие hard negative и false negative

## Positive

Passage действительно релевантен вопросу.

## Hard negative

Passage:

- реально нерелевантен / не дает правильного ответа;
- но очень похож на positive;
- поэтому его трудно отличить от positive.

## False negative

Passage был отнесен к negative pool из-за отсутствия разметки, но фактически является релевантным.

```text
                    похож на positive?
                          │
                         да
                          │
             ┌────────────┴────────────┐
             │                         │
         реально                    реально
        релевантен                 нерелевантен
             │                         │
             ▼                         ▼
       FALSE NEGATIVE              HARD NEGATIVE
```

Для нашей задачи особенно важно **не загрязнить hard-negative выборку false negatives**.

---

# 7. Какие negatives фильтруют авторы mHFN

В Section 3.4 для каждого query авторы объединяют:

- positive passages;
- in-batch negatives;
- cached negatives;

в passage set:

\[
\mathcal P.
\]

Именно этот набор используется в adaptive false-negative filter.

После фильтрации оставшиеся true negatives объединяются с original/synthetic hard negatives для обучения.

### Следствие

Оригинальный mHFN **не утверждает**, что BM25 hard negatives пропускаются через этот K-means/Gumbel filter.

Если мы будем так очищать hard-negative candidates, это **наша адаптация**, а не дословное воспроизведение.

---

# 8. K-means в adaptive false-negative filter

Для query \(q_i\) формируется passage set \(\mathcal P\).

Passage representations разбиваются K-means на:

\[
K
\]

clusters с центрами:

\[
c_1,\dots,c_K.
\]

Интуиция:

> passages с близкими dense representations чаще будут иметь один semantic pseudo-label.

Но \(K\) заранее неизвестно.

---

# 9. Gap Statistic

Для автоматического выбора \(K\) mHFN использует:

**Tibshirani, Walther, Hastie (2001),  
“Estimating the Number of Clusters in a Data Set Via the Gap Statistic.”**

DOI:

https://doi.org/10.1111/1467-9868.00293

Основная статистика:

\[
Gap(k)
=
E^*[\log W_k]
-
\log W_k,
\]

где:

- \(W_k\) — within-cluster dispersion при \(k\) clusters;
- \(E^*\) — ожидаемое значение для reference null distribution.

Смысл:

> сравнивается компактность реальной кластеризации с компактностью, которую можно было бы получить на данных без выраженной кластерной структуры.

В оригинальной работе Tibshirani et al. также используется правило выбора наименьшего \(k\), после которого улучшение следующего решения уже укладывается в неопределенность оценки.

---

# 10. Роль Gap Statistic в mHFN

Gap Statistic используется **только для автоматического определения K для K-means**.

Он:

- не определяет positive;
- не определяет hard negative;
- не вычисляет QA relevance;
- не является keyword score.

---

# 11. Similarity к cluster centers

Для passage \(p\) и cluster center \(c_t\) Shen et al. публикуют:

\[
P_{p\in L_t}
=
\frac{sim(p,c_t)}
{\sum_{j=1}^{K}sim(p,c_j)}.
\]

Далее это используется как основа Gumbel-Softmax.

## Техническая оговорка

В Section 3.1 явно определен dot product для \(sim(q,p)\).  
В Section 3.4 используется \(sim(p,c_t)\), но отдельная подробная процедура преобразования passage-centroid similarities в корректное вероятностное распределение не описана.

Поэтому при реализации нельзя молча придумывать отсутствующие детали.

Если точный авторский код недоступен, выбранное преобразование, например:

\[
P_t = softmax(s_t)
\]

от cosine или dot-product similarities, необходимо явно назвать **нашей численно устойчивой адаптацией**.

---

# 12. Gumbel-Softmax

Источник:

**Jang, Gu, Poole (2017),  
“Categorical Reparameterization with Gumbel-Softmax.”**

https://research.google/pubs/categorical-reparameterization-with-gumbel-softmax/

Для каждого label:

\[
g_t\sim Gumbel(0,1).
\]

Стандартная форма:

\[
y_t
=
\frac{
\exp((\log P_t+g_t)/\tau)
}{
\sum_j
\exp((\log P_j+g_j)/\tau)
}.
\]

Pseudo-label:

\[
\hat L(p)=\arg\max_t y_t.
\]

---

# 13. Temperature τ

\(\tau\) управляет жесткостью assignment.

### Высокая τ

- распределение более плавное;
- assignment менее уверенный.

### Низкая τ

- распределение ближе к one-hot;
- assignment более уверенный.

---

# 14. Temperature schedule в mHFN

В начале обучения passage representations еще ненадежны, поэтому mHFN использует более высокую temperature.

По мере обучения representations улучшаются, temperature уменьшается.

В статье приведена схема:

\[
\tau
=
\frac{\tau_0}
{\#TrainingSteps}.
\]

Смысл:

```text
начало training
→ representations ненадежны
→ high τ
→ менее уверенные pseudo-labels

позже
→ representations лучше
→ lower τ
→ более уверенные pseudo-labels
```

---

# 15. Как mHFN определяет false negative

После pseudo-labeling:

1. определяется label positive passage;
2. проверяются negative passages;
3. negative с тем же pseudo-label считается false negative.

\[
\hat L(p^-)=\hat L(p^+)
\]

↓

\[
p^- \rightarrow false\ negative.
\]

Такие passages исключаются из negative set.

---

# 16. Что mHFN делает с false negatives

Авторы затем используют false negatives как positive passages в NCE loss.

Но они начинают делать это только **после 20% training steps**, когда pseudo-label confidence должна быть выше.

Это специфично для training mHFN.

---

# 17. Проверенная схема оригинального метода

После повторной сверки Sections 3.1–3.5 корректно описывать источник так:

```text
                         QUERY
                           │
             ┌─────────────┴─────────────┐
             │                           │
     labeled positive              negative pools
                                        │
                         ┌───────────────┴──────────────┐
                         │                              │
                 BM25 hard negatives            in-batch/cache
                         │                              │
                         │                              ▼
                         │                 Adaptive FN Filter
                         │                     │
                         │           K-means + Gap Statistic
                         │                     │
                         │              Gumbel-Softmax
                         │                     │
                         │               pseudo-labels
                         │                     │
                         │       same pseudo-label as positive?
                         │             │              │
                         │            yes             no
                         │             │              │
                         │             ▼              ▼
                         │       false negative   true negative
                         │
                         └─────────────┬──────────────
                                       │
                              contrastive training
```

---

# 18. Что переносится в наш эксперимент

Мы переносим общий принцип:

> перед тем как считать сложный неправильный chunk настоящим hard negative, нужно попытаться исключить случаи, которые считаются negative из-за неполной или неоднозначной разметки, но по представлению оказываются очень близкими к known positive.

False-negative filtering становится **этапом очистки выборки**.

---

# 19. Что не переносится

На Phase 1 не нужны:

- multilingual translation;
- multilingual hard-negative sharing;
- synthetic hard-negative interpolation;
- multilingual cache queue;
- NCE retriever training;
- обновление retrieval encoder.

---

# 20. Отличие нашей постановки

У mHFN задача:

\[
query \rightarrow relevant\ passage
\]

для corpus-level dense retrieval.

В нашей постановке на SciREX:

\[
(document,\ aspect,\ question)
\rightarrow
correct\ chunk
\rightarrow
extractive\ answer.
\]

Каждый документ заранее разбит на один общий набор chunks.

Для одного документа отдельно рассматриваются aspect-level QA-instance:

```text
(document_id, method)
(document_id, task)
(document_id, metric)
(document_id, material)
```

если соответствующий instance имеет статус:

```text
ready
```

или:

```text
partially_extractable
```

Один и тот же `chunk_id` может быть:

- positive для одного аспекта;
- hard negative для другого аспекта;
- easy negative для третьего аспекта.

Поэтому все метки Phase 1 должны быть **aspect-specific**.

Мы работаем с chunks конкретного SciREX aspect-level QA-instance, а не с огромным corpus retrieval pool.

Следовательно, дальнейшая схема остается **адаптацией mHFN к chunk selection for extractive QA**, но теперь ее единица явно определена как `(document_id, aspect)`.


# 21. Целевой формат SciREX QA-instance

Использовать уже подготовленный aspect-level объект.

Минимальная логическая структура:

```json
{
  "qa_instance_id": "<document_id>::<aspect>",
  "document_id": "...",
  "aspect": "task",

  "question": "What task is addressed in the paper?",

  "raw_gold_answers": [
    "..."
  ],

  "canonical_gold_answers": [
    "..."
  ],

  "extractable_gold_answers": [
    "..."
  ],

  "unextractable_gold_answers": [
    "..."
  ],

  "status": "ready",

  "chunk_ids": [
    0,
    1,
    2
  ],

  "gold_supporting_chunk_ids": [
    1
  ],

  "positive_chunk_ids": [],
  "evidence_positive_qa_failed_ids": [],
  "false_negative_candidate_ids": [],
  "hard_negative_chunk_ids": [],
  "easy_negative_chunk_ids": []
}
```

`qa_instance_id` является **техническим идентификатором эксперимента**, а не новой фактической информацией SciREX.

Рекомендуемое детерминированное значение:

```text
qa_instance_id = document_id + "::" + aspect
```

Не создавать отдельный QA-instance для каждого отдельного элемента `extractable_gold_answers`.

Для одного аспекта остается **один вопрос и множество допустимых правильных ответов**.


# 22. Шаг 1. Использовать результат текущего SciREX preprocessing

Phase 1 должна начинаться **не с повторного парсинга raw SciREX**, а с уже подготовленных артефактов предыдущего этапа.

Ожидаемые входные таблицы:

## `documents.parquet`

Минимум:

```text
document_id
raw_text
clean_text
```

## `chunks.parquet`

Минимум:

```text
document_id
chunk_id
chunk_text
char_start
char_end
token_start
token_end
section_type
section_title
eligible_for_qa
segment_ids
```

## `aspect_instances.parquet`

Минимум:

```text
document_id
aspect
question
status
raw_gold_answers
canonical_gold_answers
extractable_gold_answers
unextractable_gold_answers
chunk_ids
gold_supporting_chunk_ids
```

Дополнительно там могут присутствовать:

```text
external_enrichment
enrichment_candidates
```

но в основном режиме Phase 1 они **не используются**.

## `gold_mentions.parquet`

Минимум:

```text
document_id
aspect
canonical_answer
raw_answer
mention_id
matched_text
char_start
char_end
chunk_ids
```

## `source_metadata.parquet`

Сохраняется для трассируемости исходных:

```text
doc_id
methods
tasks
metrics
materials
score
```

### Какие aspect instances допускать в Phase 1

Использовать только:

```text
aspect ∈ {method, task, metric, material}
```

и:

```text
status ∈ {ready, partially_extractable}
```

Instance со статусом:

```text
no_extractable_gold
```

не включать в основной extractive QA experiment и не превращать его chunks в negative-only examples.

### Поле `score`

Не создавать для `score` отдельный QA-instance в основной Phase 1:

```python
ENABLE_SCORE_ASPECT = False
```

Это соответствует уже принятой схеме preprocessing.

### Enrichment

Основная конфигурация:

```text
enrichment_mode = E0
```

то есть:

```text
source-only
```

Поля:

```text
external_enrichment
enrichment_candidates
```

сохранять для будущих экспериментов, но не использовать:

- при определении правильного ответа;
- при `qa_positive`;
- при hard-negative mining;
- при false-negative filtering.

Это необходимо, чтобы Phase 1 не загрязнялась сторонней информацией и оставалась сопоставимой с baseline preprocessing.


# 23. Шаг 2. Использовать уже сформированные общие chunks

**Не выполнять новый chunking**, если `chunks.parquet` уже создан предыдущим preprocessing.

Для каждого SciREX-документа использовать один и тот же набор:

```text
chunk_0
chunk_1
...
chunk_N
```

во всех аспектах.

То есть:

```text
method   ┐
task     ├→ те же chunk_id
metric   │
material ┘
```

Это критично для последующего честного сравнения.

Для каждого chunk использовать уже сохраненные:

```text
document_id
chunk_id
chunk_text
token_start
token_end
char_start
char_end
eligible_for_qa
```

При overlap сохранить исходные глобальные offsets и `mention_id`, уже подготовленные preprocessing.

Нельзя:

- повторно нарезать документ отдельно для каждого аспекта;
- менять `chunk_size` между аспектами;
- менять `chunk_overlap` между аспектами;
- повторно нормализовать текст так, чтобы изменились offsets.

Если текущие preprocessing-артефакты отсутствуют или не проходят validation, сначала исправить preprocessing, а не создавать альтернативный chunker внутри Phase 1.


# 24. Шаг 3. Frozen QA по каждому `(aspect, chunk)`

Одна и та же frozen extractive QA model применяется к каждому `eligible_for_qa` chunk.

Для одного документа chunk text один и тот же, но вопрос зависит от аспекта.

Пример:

```text
chunk_3 + method question
chunk_3 + task question
chunk_3 + metric question
chunk_3 + material question
```

Это четыре разных QA inference, хотя физический chunk один.

Для каждого `(qa_instance_id, chunk_id)` сохранить:

```text
document_id
aspect
qa_instance_id
question
chunk_id

predicted_answer
answer_start
answer_end
start_logit
end_logit
qa_confidence
```

Дополнительно сохранить:

```text
matched_gold_answer
answer_quality
qa_positive
```

QA-модель не должна меняться в процессе создания positive/hard-negative dataset.

Нельзя использовать `external_enrichment` для подмены вопроса или допустимых ответов в основном режиме `E0`.


# 25. Два независимых признака positive для SciREX

Необходимо хранить два независимых свойства **для каждого `(document_id, aspect, chunk_id)`**.

## Gold-supporting

Этот признак уже получен предыдущим preprocessing.

```text
gold_supporting = true
```

если `chunk_id` входит в:

```text
gold_supporting_chunk_ids
```

данного aspect-instance.

Он означает:

> chunk содержит хотя бы одно подтвержденное текстовое mention из `extractable_gold_answers` данного аспекта.

## QA-positive

Frozen QA model на данном chunk возвращает ответ, соответствующий хотя бы одному:

```text
extractable_gold_answers
```

данного аспекта.

```text
qa_positive = true / false
```

Важно:

```text
gold_supporting
```

и:

```text
qa_positive
```

остаются разными свойствами.


# 26. Почему `gold_supporting` и `qa_positive` нельзя смешивать

Для конкретного `(document_id, aspect, chunk_id)` возможны четыре случая.

### A

```text
gold_supporting = true
qa_positive = true
```

Chunk содержит подтвержденное mention данного аспекта, и QA извлекает допустимый правильный answer.

Это наиболее надежный positive.

### B

```text
gold_supporting = true
qa_positive = false
```

Chunk содержит gold mention, но QA отвечает неправильно.

Это:

```text
evidence_positive_qa_failed
```

и **не hard negative**.

### C

```text
gold_supporting = false
qa_positive = true
```

QA извлекает один из допустимых `extractable_gold_answers`, хотя preprocessing не пометил chunk как `gold_supporting`.

Такой случай необходимо:

1. сохранить;
2. проверить offsets и нормализацию;
3. не превращать автоматически в hard negative.

Это может быть:

- альтернативное textual occurrence;
- пограничный случай нормализации;
- overlap/mapping issue;
- потенциальный unlabeled positive.

### D

```text
gold_supporting = false
qa_positive = false
```

Это исходный negative candidate для данного аспекта.


# 27. Основной positive set

Поскольку цель будущего анализа:

> найти признаки чанков, **которые дают правильный ответ при frozen QA**,

основной positive set строится отдельно для каждого aspect-level QA-instance:

\[
P_{d,a}
=
\{c:
QA(c,q_a)
\approx
G_{d,a}^{extractable}
\},
\]

где:

- \(d\) — SciREX document;
- \(a\) — аспект `method/task/metric/material`;
- \(q_a\) — фиксированный вопрос аспекта;
- \(G_{d,a}^{extractable}\) — множество `extractable_gold_answers`.

То есть:

```text
positive_chunk
=
qa_positive == true
```

при данном `(document_id, aspect)`.

Одновременно обязательно сохранять:

```text
gold_supporting
matched_gold_answer
```

чтобы позднее различать QA-success и наличие заранее подтвержденного mention.


# 28. Критерий корректного QA answer для multi-answer SciREX instance

У одного аспекта может быть несколько допустимых ответов.

Поэтому prediction нельзя сравнивать с одним `gold_answer`.

Необходимо сравнивать:

```text
predicted_answer
```

со всем множеством:

```text
extractable_gold_answers
```

данного `(document_id, aspect)`.

Основной первый вариант:

```text
normalized_exact_match
```

QA chunk считается positive, если:

\[
\exists g
\in
G_{d,a}^{extractable}:
Norm(prediction)=Norm(g).
\]

Сохранить:

```text
matched_gold_answer
```

если match найден.

Дополнительные критерии:

```text
token_f1
semantic criterion
```

можно исследовать позже как отдельный ablation.

Threshold выбирать только на DEV.

### Важное ограничение

`unextractable_gold_answers` не использовать как обязательные extractive targets.

`external_enrichment` и `enrichment_candidates` в режиме `E0` также не использовать как допустимые ответы.

Иначе состав positive set перестанет соответствовать базовому SciREX preprocessing.


# 29. `evidence_positive_qa_failed`

Для каждого аспекта chunks:

```text
gold_supporting = true
qa_positive = false
```

выделить в отдельную категорию:

```text
evidence_positive_qa_failed
```

Они **не являются hard negatives**.

Их необходимо исключить из negative pool независимо от величины:

- BM25;
- QA confidence;
- dense similarity.

Позднее эта группа может использоваться для отдельного исследования:

> почему frozen QA не смогла извлечь допустимый ответ из chunk, который содержит подтвержденное gold mention соответствующего SciREX-аспекта.


# 30. Шаг 4. Negative candidates

Negative pool строится **отдельно для каждого `(document_id, aspect)`**.

Исходный pool:

\[
N_{d,a}
=
\{c:
qa\_positive(c,a)=false
\land
gold\_supporting(c,a)=false
\}.
\]

Иными словами, chunk может быть negative для `task`, но не быть negative для `method`.

Не переносить labels между аспектами.

В pool не должны попадать:

```text
qa_positive
evidence_positive_qa_failed
```

для текущего аспекта.

Также не включать chunks:

```text
eligible_for_qa = false
```

если preprocessing исключил их из QA context.

Большинство объектов \(N_{d,a}\) будут easy negatives.


# 31. Шаг 5. Hard-negative candidates

Source-inspired baseline сохраняется без изменения по сути:

```text
aspect question
   ↓
BM25 over eligible chunks данного документа
   ↓
Top-N non-positive chunks для данного аспекта
```

Формально:

\[
HN_{BM25}^{d,a}
=
TopN_{c\in N_{d,a}}
BM25(q_a,c).
\]

То есть BM25 score необходимо считать заново для каждого аспекта, потому что:

```text
q_method != q_task != q_metric != q_material
```

при одинаковых физических chunks.

mHFN использует top-30 non-positive passages на corpus retrieval.

Внутри одного SciREX-документа число chunks существенно меньше, поэтому проверить на DEV, например:

```text
N = 3 / 5 / 10
```

или использовать:

```text
N = min(configured_top_n, |N_{d,a}|)
```

Не выбирать один и тот же hard-negative set для всех аспектов документа.


# 32. Дополнительные определения hardness

После BM25 baseline можно сравнить:

## HN-QA

Wrong chunks с максимальным QA confidence.

## HN-Dense-Q

Chunks с максимальной dense question similarity.

## HN-Positive

Chunks, embedding которых наиболее похож на positive chunk.

## HN-Hybrid

Rank fusion:

```text
BM25
+
question similarity
+
QA confidence
```

---

# 33. Почему high-confidence wrong chunks полезны

Это случаи, где QA-модель уверенно ошибается.

Они могут содержать:

- похожие лексические конструкции;
- похожие entity types;
- слова вопроса;
- похожий контекст;
- misleading attention patterns.

Именно такие примеры особенно интересны для будущего поиска discriminative features.

---

# 34. Шаг 6. False-negative filtering

Перед фиксацией hard negatives применить адаптированный mHFN filter:

```text
K-means
+
Gap Statistic
+
Gumbel-Softmax
```

---

# 35. Набор для clustering в SciREX

False-negative filtering выполняется **отдельно для каждого aspect-level QA-instance**.

Для:

```text
(document_id = d, aspect = a)
```

формируется:

\[
\mathcal P_{d,a}
=
P_{d,a}
\cup
N_{d,a}^{candidate},
\]

где:

- \(P_{d,a}\) — QA-positive chunks данного аспекта;
- \(N_{d,a}^{candidate}\) — candidate negatives данного аспекта.

Минимальный практический вариант:

```text
positive chunks аспекта
+
BM25 hard-negative candidates аспекта
+
при необходимости несколько обычных negatives этого же аспекта
```

Физические chunks могут быть теми же самыми для разных аспектов, но:

- positive set;
- negative candidate set;
- BM25 ranking;

различаются по вопросу.

Поэтому результаты false-negative filtering также сохраняются с ключом:

```text
document_id
aspect
chunk_id
```

## Важное отличие от mHFN

В оригинале filter работает по positive + in-batch/cache negatives.

Включение BM25 hard-negative candidates в \(\mathcal P_{d,a}\) остается **нашей адаптацией**, а не дословным воспроизведением mHFN.

Это не меняется из-за выбора SciREX.


# 36. Chunk representations

Использовать frozen passage encoder:

\[
e_c=E(c).
\]

Нельзя в одной Phase 1 одновременно переобучать encoder и менять labels.

---

# 37. Включать ли question в embedding

В mHFN K-means работает с passage representations.

Поэтому source-aligned baseline:

\[
e_c=E_p(c).
\]

То есть:

```text
C0 = passage-only
```

Дополнительный будущий ablation:

```text
C1 = question-conditioned representation
```

C1 уже является нашей модификацией.

---

# 38. K-means и диапазон K

Выполнить K-means для диапазона:

\[
K=1,\dots,K_{max}.
\]

\(K_{max}\) сделать зависимым от числа chunks, например:

\[
K_{max}
=
\min(10,\lfloor\sqrt{|\mathcal P_q|}\rfloor).
\]

Это пример нашей конфигурации, не параметр из mHFN.

---

# 39. Gap Statistic procedure

Для каждого \(K\):

1. K-means на observed embeddings.
2. Within-cluster dispersion.
3. Reference null samples.
4. K-means на reference samples.
5. `Gap(K)`.
6. Выбор optimal K.

Параметр:

```text
reference/bootstrap samples B = 10 / 20 / 50
```

должен быть конфигурируемым.

---

# 40. Слишком мало chunks

Если данных недостаточно для meaningful clustering:

```text
fn_filter_status = skipped_insufficient_data
```

Не заставлять K-means/Gumbel работать искусственно.

---

# 41. Edge case K=1

Если Gap Statistic выбирает:

\[
K=1,
\]

то все chunks имеют один label, и правило `same label as positive` объявило бы false negatives почти все negatives.

mHFN отдельно этот edge case не обсуждает.

Поэтому в нашей адаптации:

> при \(K=1\) cluster-based FN filtering не применять.

```text
fn_filter_status = no_cluster_structure
```

---

# 42. Similarity vector

После K-means для chunk \(p\):

\[
s(p)
=
[
sim(e_p,c_1),\ldots,sim(e_p,c_K)
].
\]

---

# 43. Probabilities

Необходимо явно обеспечить:

\[
P_t>0,
\qquad
\sum_tP_t=1.
\]

Если точная реализация авторов не установлена, использовать явно документированную адаптацию:

\[
P_t=softmax(s_t/\tau_s).
\]

В отчете отметить, что это численно устойчивое уточнение нашей реализации.

---

# 44. Gumbel-Softmax pseudo-label

Для каждого chunk:

\[
g_t\sim Gumbel(0,1).
\]

\[
y_t
=
softmax
\left(
\frac{\log P_t+g_t}{\tau_g}
\right).
\]

\[
L(c)=argmax_t y_t.
\]

---

# 45. Важное отличие: frozen embeddings

В mHFN temperature \(\tau\) уменьшается по мере **обучения retriever**, поскольку сами passage representations меняются и, по замыслу авторов, становятся надежнее.

В нашей Phase 1 passage/chunk encoder **заморожен**:

\[
E(c)=const.
\]

Следовательно, качество representations не улучшается от некоторого условного «шага 1» к «шагу 100».

Поэтому:

> **training-step decay temperature из mHFN в основной метод нашего эксперимента не переносится.**

Искусственное уменьшение \(\tau\) при неизменных embeddings создавало бы видимость роста уверенности без соответствующего изменения представлений.

---

# 46. Основной вариант и роль Gumbel-Softmax

## 46.1. Основной вариант Phase 1

Основным способом pseudo-labeling сделать **обычное hard assignment, уже полученное K-means**:

\[
L(c)
=
\arg\min_j
\|e_c-\mu_j\|^2.
\]

То есть основной false-negative filter:

```text
chunk embeddings
      ↓
K-means
      ↓
Gap Statistic выбирает K
      ↓
hard cluster assignment
      ↓
negative находится в том же cluster,
что хотя бы один positive?
```

В основной схеме **temperature вообще отсутствует**.

Это наиболее простой, детерминированный и воспроизводимый вариант для frozen embeddings.

## 46.2. Gumbel-Softmax — только дополнительный ablation

Чтобы проверить, дает ли stochastic pseudo-labeling из mHFN дополнительную пользу в нашей статической постановке, отдельно разрешается сравнить:

```text
K-means + Gap Statistic + fixed-temperature Gumbel-Softmax
```

Но:

- никакого decay по шагам не использовать;
- Gumbel-Softmax не считать обязательной частью метода;
- фиксированную \(\tau\) рассматривать только как параметр дополнительного эксперимента.

При необходимости проверить небольшой набор фиксированных значений на DEV, например:

```text
τ ∈ {0.25, 0.5, 1.0}
```

Это **не адаптация temperature во времени**, а обычный выбор постоянного hyperparameter для ablation.

## 46.3. Optional repeated Gumbel

Если один stochastic assignment оказывается нестабильным, только в дополнительном ablation можно выполнить:

```text
R = 20 / 50
```

Gumbel draws и оценить:

\[
P_{same}(n,p)
=
\frac{
\#runs[L_r(n)=L_r(p)]
}{R}.
\]

Этот consensus-механизм является **нашей offline-адаптацией**, а не частью оригинального mHFN.


# 47. Основной критерий false-negative candidate

В основной конфигурации используется hard K-means assignment.

Для negative chunk \(n\) он считается `false_negative_candidate`, если его cluster совпадает с cluster хотя бы одного positive chunk:

\[
FN(n)=
\mathbb{1}
\left[
\exists p\in P_q:
L(n)=L(p)
\right].
\]

То есть:

```text
same hard cluster as positive
        ↓
false-negative candidate
```

Это наиболее близкая по смыслу адаптация правила mHFN, но без ненужного stochastic слоя.

Если positive chunks несколько, достаточно совпадения с одним из них.

---

# 48. Дополнительный probabilistic diagnostic

Поле:

```text
false_negative_probability
```

**не является обязательным для основной Phase 1**.

Оно появляется только в optional Gumbel ablation.

При repeated Gumbel можно определить:

\[
P_{FN}(n)
=
\max_{p\in P_q}
P_{same}(n,p).
\]

Для основной конфигурации сохранять:

```text
same_cluster_as_positive: true / false
```

а для Gumbel-ablation дополнительно:

```text
false_negative_probability
```

---

# 49. Консервативное использование false-negative candidates

В mHFN найденные false negatives впоследствии могут использоваться как positives в training loss.

В нашей Phase 1 основной режим более консервативный:

```text
negative
  ↓
same hard cluster as positive
  ↓
false_negative_candidate
  ↓
ИСКЛЮЧИТЬ из hard-negative set
```

Но автоматически не переносить такой chunk в primary positive set.

Причина:

> принадлежность одному semantic cluster еще не доказывает, что chunk действительно содержит корректное evidence.

---

# 50. Confirmed false negative

`false_negative_candidate` можно повысить до:

```text
confirmed_false_negative
```

только при независимом подтверждении внутри уже подготовленных SciREX-данных.

Допустимые подтверждения в baseline `E0`:

- chunk входит в `gold_supporting_chunk_ids` данного аспекта;
- в chunk обнаружен подтвержденный `gold_mentions.parquet` mention данного аспекта;
- frozen QA на chunk возвращает один из `extractable_gold_answers`.

Если возникает такое подтверждение после того, как chunk ранее считался negative, необходимо:

1. сохранить диагностическую причину расхождения;
2. исключить chunk из hard-negative set;
3. проверить preprocessing / offsets / answer matching.

**Не использовать для подтверждения** в baseline:

```text
external_enrichment
enrichment_candidates
unextractable_gold_answers
```

Иначе source-only разметка будет загрязнена сторонней информацией.


# 51. Финальный hard-negative set

В основной конфигурации hard negative для `(document_id=d, aspect=a)` должен удовлетворять:

\[
qa\_positive(c,a)=false,
\]

\[
gold\_supporting(c,a)=false,
\]

\[
same\_cluster\_as\_positive(c,a)=false,
\]

и иметь достаточно высокий hardness score для вопроса \(q_a\).

Например:

```text
BM25(question_of_aspect, chunk)
```

или, в отдельном mining ablation:

```text
QA confidence
dense question-chunk similarity
```

Финальная метка всегда aspect-specific:

```text
(document_id, aspect, chunk_id) -> hard_negative
```

Один и тот же физический chunk может иметь другую категорию для другого аспекта.

Если проводится optional Gumbel experiment, дополнительно можно сравнить правило:

\[
P_{FN}(c|d,a)<t_{FN}.
\]

Но это **не основное определение Phase 1**.


# 52. Primary analytical classes

Каждый chunk получает одну основную категорию:

```text
1. qa_positive
2. evidence_positive_qa_failed
3. confirmed_false_negative
4. false_negative_candidate
5. hard_negative
6. easy_negative
```

При этом исходные бинарные flags сохраняются.

---

# 53. Пример итогового объекта SciREX

```json
{
  "qa_instance_id": "doc_001::task",
  "document_id": "doc_001",
  "aspect": "task",
  "question": "What task is addressed in the paper?",

  "extractable_gold_answers": [
    "Sentiment Analysis",
    "Text Classification"
  ],

  "unextractable_gold_answers": [],

  "status": "ready",
  "enrichment_mode": "E0",

  "chunks": [
    {
      "chunk_id": 0,
      "text": "...",

      "gold_supporting": false,

      "qa_positive": false,
      "qa_answer": "...",
      "matched_gold_answer": null,
      "qa_score": 0.42,

      "bm25_score": 4.21,

      "cluster_id": 1,
      "same_cluster_as_positive": false,

      "class": "hard_negative"
    },
    {
      "chunk_id": 1,
      "text": "...",

      "gold_supporting": true,

      "qa_positive": true,
      "qa_answer": "Text Classification",
      "matched_gold_answer": "Text Classification",
      "qa_score": 0.91,

      "cluster_id": 2,
      "same_cluster_as_positive": true,

      "class": "qa_positive"
    }
  ],

  "positive_chunk_ids": [1],
  "evidence_positive_qa_failed_ids": [],
  "false_negative_candidate_ids": [],
  "hard_negative_chunk_ids": [0]
}
```

Важно:

- `extractable_gold_answers` может содержать несколько значений;
- `question` определяется аспектом;
- `chunks` физически общие с другими аспектами того же документа;
- все analytical classes относятся только к текущему `aspect`;
- `external_enrichment` не участвует в данном объекте при `enrichment_mode="E0"`.


# 54. Итог Phase 1 для SciREX

Для каждого допустимого SciREX `(document_id, aspect)` получить:

## Aspect

```text
method / task / metric / material
```

## Question

Фиксированный вопрос этого аспекта.

## Chunks

Общий набор `eligible_for_qa` chunks документа.

## Correct answers

Не один `gold_answer`, а множество:

```text
extractable_gold_answers
```

## Gold-supporting chunks

Chunks из подготовленного:

```text
gold_supporting_chunk_ids
```

## Positive chunks

Chunks, где frozen QA возвращает хотя бы один допустимый `extractable_gold_answer`.

## Evidence-positive QA-failed

Chunks, содержащие подтвержденное gold mention, но не дающие правильный QA prediction.

## False-negative candidates

Negative candidates, которые не следует автоматически считать true negatives по результату false-negative filter.

## Hard negatives

Chunks, которые:

- дают неправильный QA answer для текущего аспекта;
- не являются `gold_supporting` для текущего аспекта;
- достаточно сложны / релевантны вопросу;
- проходят false-negative filtering.

## Easy negatives

Остальные неправильные chunks данного аспекта.

Все наборы должны быть привязаны к:

```text
document_id
aspect
```

а не только к документу.


# 55. Не анализировать ключевые признаки на Phase 1

Phase 1 должна закончиться на построении датасета.

Не использовать будущие исследуемые признаки для определения классов.

Если позже мы хотим проверить attention, `diff_score` или Question-KeyBERT как distinguishing features, нельзя заранее определять hard negatives именно по этим признакам.

Иначе возникнет circularity.

---

# 56. Безопасные hardness signals

Основной baseline:

```text
BM25(question, chunk)
```

Дополнительно:

```text
QA confidence при неправильном ответе
dense question-chunk similarity
```

Attention, `diff_score`, Question-KeyBERT и будущие keyword profiles не использовать для primary class creation.

---

# 57. Audit выборки

До Phase 2 провести audit случайной выборки **aspect-level** примеров.

Для:

```text
N = 50–100 (document_id, aspect)
```

вывести:

```text
document_id
aspect
question

extractable_gold_answers
unextractable_gold_answers
gold_supporting_chunk_ids

positive chunk
matched_gold_answer

hard negative #1
hard negative #2

false negative candidate
```

Обязательно включить в audit примеры:

- `ready`;
- `partially_extractable`;
- разные аспекты;
- документы с несколькими `extractable_gold_answers`.

Не использовать `no_extractable_gold` как отрицательные QA examples.


# 58. Automatic sanity checks

Обязательно проверить:

### Check 1

Каждый обрабатываемый QA-instance имеет:

```text
aspect ∈ {method, task, metric, material}
```

и:

```text
status ∈ {ready, partially_extractable}
```

### Check 2

`extractable_gold_answers` не пуст для каждого включенного instance.

### Check 3

`qa_positive=true` только если prediction matches хотя бы один `extractable_gold_answer`.

### Check 4

Для `qa_positive=true` сохраняется:

```text
matched_gold_answer
```

из того же множества `extractable_gold_answers`.

### Check 5

Hard negative имеет:

```text
qa_positive = false
gold_supporting = false
```

для **того же аспекта**.

### Check 6

`gold_supporting=true` + `qa_positive=false` попадает в:

```text
evidence_positive_qa_failed
```

а не в hard negative.

### Check 7

False-negative candidates исключены из hard-negative pool.

### Check 8

Один `(document_id, aspect, chunk_id)` не имеет взаимоисключающих primary classes.

### Check 9

Один физический `chunk_id` может иметь разные labels между аспектами — это не ошибка.

### Check 10

Все offsets и `chunk_id` совпадают с исходными preprocessing-артефактами; Phase 1 не делает повторный chunking.

### Check 11

`external_enrichment` и `enrichment_candidates` не используются при `enrichment_mode=E0`.

### Check 12

`score` не породил самостоятельных QA-instance в основной Phase 1.

### Check 13

`no_extractable_gold` не превращается в negative-only training example.


# 59. Статистика Phase 1

Сохранять общую статистику и статистику **по каждому аспекту**.

Общий минимум:

```text
documents_total
qa_instances_total
chunks_total
aspect_chunk_pairs_total
```

По каждому:

```text
method
task
metric
material
```

считать:

```text
instances_total
ready_instances
partially_extractable_instances

extractable_gold_answers_total
unextractable_gold_answers_total

positive_chunks_total
positive_per_instance_mean

gold_supporting_chunks_total
evidence_positive_qa_failed_total

hard_negatives_total
hard_negatives_per_instance_mean

false_negative_candidates_total
false_negative_rate

easy_negatives_total
```

Дополнительно сохранить распределение:

```text
number_of_extractable_gold_answers_per_instance
```

поскольку SciREX aspect-level instance может быть multi-answer.


# 60. Диагностика FN filter

Сохранять:

```text
selected_K
gap_values
cluster_sizes
hard_cluster_id
same_cluster_as_positive

# только для optional Gumbel ablation:
gumbel_temperature
gumbel_runs
pseudo_label_distribution
false_negative_probability
```

---

# 61. Ablation первой фазы

Сравнить минимум:

## F0 — без false-negative filtering

```text
BM25 Top-N wrong chunks
```

## F1 — основной метод

```text
K-means
+
Gap Statistic
+
hard cluster assignment
```

Negative с тем же hard cluster label, что и positive, исключается из hard-negative set.

**F1 является основной конфигурацией Phase 1.**

## F2 — optional source-inspired Gumbel ablation

```text
K-means
+
Gap Statistic
+
Gumbel-Softmax с фиксированной τ
```

Никакого temperature decay.

## F3 — optional repeated-Gumbel ablation

```text
K-means
+
Gap Statistic
+
repeated fixed-τ Gumbel-Softmax
+
consensus / P_same
```

Цель F2/F3:

> проверить, дает ли stochastic pseudo-labeling дополнительную пользу относительно обычного hard K-means assignment в нашей frozen/offline постановке.

---

# 62. Критический контроль

В оригинальном mHFN Gumbel-Softmax встроен в динамический training pipeline, где representations изменяются во времени.

В нашей Phase 1 embeddings frozen.

Поэтому:

> **ни Gumbel-Softmax, ни temperature decay не следует считать обязательными компонентами нашей адаптации.**

Основной метод:

\[
\boxed{
KMeans
\rightarrow
GapStatistic
\rightarrow
HardAssignment
}
\]

Если F2/F3 не превосходят F1, они исключаются из дальнейшей архитектуры как ненужное усложнение.

Полностью допустимый результат:

```text
F1 >= F2
```

или:

```text
F1 ≈ F2 ≈ F3
```

В таком случае предпочтение отдается F1 как более простому, быстрому и воспроизводимому варианту.


# 63. Оценка FN filtering

В текущем SciREX preprocessing нет отдельной полной gold-разметки всех возможных false negatives.

Поэтому pseudo-label нельзя считать ground truth.

Использовать доступные source-only проверки:

- доля FN candidates, которые неожиданно входят в `gold_supporting_chunk_ids`;
- доля FN candidates, где повторная проверка QA дает один из `extractable_gold_answers`;
- наличие подтвержденного `gold_mentions.parquet` mention;
- manual audit;
- распределения similarity / cluster distances.

Если обнаруживается:

```text
FN candidate
+
confirmed gold mention
```

это трактуется как диагностическое расхождение и такой chunk исключается из hard-negative pool.

Нельзя подтверждать FN через:

```text
external_enrichment
```

в основном режиме `E0`.

Для future enrichment experiments можно отдельно повторить оценку как E1/E2, не перезаписывая E0.


# 64. Что будет Phase 2

После фиксации positive/hard-negative sets исследовать:

> **что отделяет Positive от HardNegative?**

Candidate features:

```text
keywords
attention
question-conditioned attention
answer-conditioned attention
diff_score
Question-KeyBERT
n-grams
semantic features
structural features
combinations
```

---

# 65. Центральный принцип Phase 2

Не спрашивать только:

> «часто ли feature встречается в positive?»

Нужно оценивать:

> **насколько feature сильнее характерен для positive, чем для наиболее похожих неправильных hard negatives?**

Например:

\[
D(f)
=
\log
\frac{
P(f|Positive)+\epsilon
}{
P(f|HardNegative)+\epsilon
}.
\]

Но конкретные feature scores относятся уже ко второй фазе.

---

# 66. TRAIN / DEV / TEST для SciREX

Использовать **те же split**, которые уже применяются в текущем SciREX preprocessing.

Не выполнять новый случайный split на уровне chunks или aspect instances.

Особенно важно:

> все аспекты и все chunks одного документа должны оставаться в одном исходном split.

Иначе возникнет document leakage.

## TRAIN

Использовать для:

- формирования train positive/HN examples;
- будущего построения feature statistics;
- будущего обучения discriminative models.

## DEV

Использовать для выбора:

- дополнительного QA answer-match threshold, если normalized exact match недостаточен;
- hard-negative Top-N;
- диапазона/параметров false-negative filtering;
- optional Gumbel fixed temperature;
- числа stochastic samples;
- будущих feature thresholds.

## TEST

Использовать только для финальной оценки.

Не подбирать параметры отдельно по TEST aspect instances.

Основной enrichment mode на всех split:

```text
E0 = source-only
```

E1/E2, если будут исследоваться позднее, должны запускаться как отдельные экспериментальные конфигурации при тех же split.


# 67. Reproducibility

Сохранять:

```text
dataset_name = SciREX
preprocessing_version
preprocessing_config_hash

input_documents_path
input_chunks_path
input_aspect_instances_path
input_gold_mentions_path

aspects = [method, task, metric, material]
enable_score_aspect = false
enrichment_mode = E0

random_seed

QA_model_name
QA_model_revision

embedding_model_name
embedding_model_revision

BM25_parameters
hard_negative_top_n

K_range
gap_bootstrap_count

positive_match_method

# optional Gumbel ablation only:
Gumbel_temperature
Gumbel_runs
FN_probability_threshold
```

Не дублировать `chunk_size/chunk_overlap` как независимо изменяемые параметры Phase 1, если они уже зафиксированы preprocessing.

Их можно записать в итоговый config только как **унаследованные preprocessing parameters** для воспроизводимости.


# 68. Выходные файлы

Не перезаписывать исходные preprocessing-артефакты.

Результаты Phase 1 сохранять отдельно, например:

```text
phase1_scirex/
```

Минимальный набор:

```text
experiment_config.json

qa_chunk_predictions.parquet

positive_chunks.parquet
evidence_positive_qa_failed.parquet
false_negative_candidates.parquet
hard_negative_chunks.parquet
easy_negative_chunks.parquet

clustering_diagnostics.parquet
gap_statistics.parquet

# только если запускался optional Gumbel ablation:
gumbel_assignments.parquet

qa_chunk_dataset.parquet

phase1_summary.md
```

Во всех таблицах, где row относится к конкретной QA-задаче, обязательно хранить:

```text
document_id
aspect
qa_instance_id
chunk_id
```

Не использовать только `chunk_id`, поскольку один физический chunk участвует в нескольких аспектах.


# 69. Рекомендуемые колонки `qa_chunk_dataset.parquet`

```text
document_id
aspect
qa_instance_id

question
status
enrichment_mode

extractable_gold_answers
unextractable_gold_answers

chunk_id
chunk_text

gold_supporting

qa_positive
matched_gold_answer
qa_predicted_answer
qa_confidence
answer_quality

bm25_score
dense_question_similarity

selected_K
cluster_id
same_cluster_as_positive

# optional Gumbel:
false_negative_probability

primary_class
```

При необходимости добавить ссылки на preprocessing evidence:

```text
gold_mention_ids
section_type
section_title
char_start
char_end
token_start
token_end
```

Все значения должны быть трассируемы либо к preprocessing artifacts, либо к вычислениям Phase 1.


# 70. Что должно быть в `phase1_summary.md`

## Dataset

Указать:

```text
SciREX
```

и:

- используемые исходные split;
- число документов;
- число aspect-level QA-instance;
- количество instance по `method/task/metric/material`;
- число `ready`;
- число `partially_extractable`;
- число исключенных `no_extractable_gold`;
- среднее chunks/document;
- среднее `extractable_gold_answers`/instance;
- `enrichment_mode=E0`;
- `score aspect disabled`.

## QA model

- модель;
- preprocessing/tokenization;
- answer-match criterion;
- доля QA-positive отдельно по аспектам.

## Hard-negative mining

- BM25 / другой способ;
- Top-N;
- hardness statistics отдельно по аспектам.

## False-negative filtering

- embedding model;
- K-means;
- Gap Statistic;
- K range;
- основной assignment: hard K-means label;
- правило `same_cluster_as_positive`;
- если запускался optional Gumbel ablation:
  - fixed temperature;
  - число stochastic runs.

## Results

Отдельно overall и по каждому aspect:

- positive count;
- evidence-positive QA-failed count;
- hard-negative count;
- FN candidate count;
- FN rate.

## Manual audit

Показать примеры минимум для нескольких аспектов.

## Limitations

Обязательно отметить:

- `gold_supporting` в текущем preprocessing определяется по подтвержденным mentions соответствующего аспекта;
- один aspect-instance может иметь несколько допустимых extractable answers;
- `no_extractable_gold` исключен из основного experiment;
- cluster label не равен истинной relevance annotation;
- схема адаптирует mHFN к within-document aspect-level extractive QA;
- основной режим использует hard K-means assignment;
- Gumbel-Softmax является только optional ablation;
- temperature decay mHFN не переносится, поскольку encoder frozen;
- external enrichment не используется в baseline E0.


# 71. mHFN vs наша схема

| Компонент | mHFN | Наш SciREX experiment |
|---|---|---|
| Основная задача | multilingual dense retrieval | aspect-level extractive QA chunk analysis |
| Единица задачи | query + corpus passages | `(document_id, aspect)` + chunks одного документа |
| Positive | labeled relevant passage | chunk, где frozen QA возвращает один из `extractable_gold_answers` |
| Gold evidence flag | labeled positive passage | `gold_supporting`, унаследованный из SciREX preprocessing |
| Hard negatives | BM25 top-30 non-positive + augmentation | BM25 / high-confidence wrong chunks отдельно по аспекту |
| FN candidate pool | in-batch + cached negatives | candidate chunks внутри `(document_id, aspect)` |
| Clustering | K-means | K-means |
| Выбор K | Gap Statistic | Gap Statistic |
| Pseudo-labels | Gumbel-Softmax | **hard K-means assignment — основной вариант; Gumbel только ablation** |
| Temperature | decays during retriever training | **в основном методе отсутствует; fixed τ только в optional Gumbel ablation** |
| FN treatment | exclude negatives, позже use as positives | exclude from HN; promote only if independently confirmed source-only evidence |
| Training loss | NCE | отсутствует в Phase 1 |
| Multi-answer | не центральная часть схемы | один aspect может иметь несколько `extractable_gold_answers` |
| External enrichment | не относится к схеме | не используется в baseline `E0` |


# 72. Что прямо подтверждается Shen et al.

После проверки Sections 3.1–3.5 статьи подтверждены следующие утверждения:

1. mHFN использует bi-encoder.
2. Query-passage similarity определяется dot product.
3. BM25 top-30 non-positive passages используются как hard negatives.
4. Adaptive FN filter формирует set из positive, in-batch negatives и cached negatives.
5. Passages кластеризуются K-means.
6. Gap Statistic автоматически выбирает K.
7. Gumbel-Softmax используется для pseudo-label assignment.
8. Negative с тем же pseudo-label, что positive, считается false negative.
9. False negative исключается из negative set.
10. False negatives затем используются как positives в training loss.
11. Их начинают использовать как positives после 20% training steps.
12. В ablation удаление FN filter приводит, по сообщению авторов, примерно к 3.0% среднему падению MRR@100 и 4.0% Recall@100.

---

# 73. Что является нашей адаптацией

Следующее **не следует приписывать Shen et al.**:

- применение false-negative filtering внутри одного SciREX-документа;
- использование `(document_id, aspect)` как отдельного QA-instance;
- четыре aspect-level вопроса `method/task/metric/material`;
- multi-answer `extractable_gold_answers`;
- positive по корректности ответа frozen RoBERTa;
- отдельный флаг `gold_supporting`, полученный SciREX preprocessing;
- QA-confidence hard negatives;
- применение FN filter непосредственно к BM25 hard-negative pool;
- hard K-means assignment как основной FN filter вместо обязательного Gumbel;
- optional repeated Gumbel consensus;
- optional `false_negative_probability`;
- fixed temperature только в Gumbel-ablation;
- отказ от temperature decay при frozen embeddings;
- `K=1 → skip`;
- консервативное исключение FN без автоматического promotion;
- исключение `no_extractable_gold` из основной QA-выборки;
- использование `E0 source-only` как базового режима;
- последующий анализ attention / keywords / KeyBERT.

Все эти пункты являются частью **нашего экспериментального дизайна**, а не описанием оригинального mHFN.


# 74. Неоднозначность формулы в PDF

В опубликованной формуле (7) mHFN визуально есть неоднозначность индекса вероятности в знаменателе. Стандартное определение Gumbel-Softmax Jang et al. использует probability/logit соответствующего класса \(j\) в каждом члене суммы.

Поэтому при реализации:

1. сверить доступный source code mHFN, если возможно;
2. использовать стандартную Gumbel-Softmax формулу;
3. не выдавать исправление возможной опечатки статьи за подтвержденную авторскую реализацию.

---

# 75. Полная Phase 1 для SciREX

```text
PREPROCESSED SCIREX DOCUMENT
                │
                ▼
        ONE COMMON CHUNK SET
                │
                ├────────────────────────────────────┐
                │                                    │
                ▼                                    ▼
      ASPECT = method / task / metric / material
                │
                ▼
     fixed question for current aspect
                │
                ▼
      extractable_gold_answers
                │
                ▼
     Frozen RoBERTa QA over every
       eligible physical chunk
                │
                ▼
       ┌────────┴────────┐
       │                 │
   QA-correct         QA-wrong
   for aspect         for aspect
       │                 │
       │          ┌──────┴─────────────┐
       │          │                    │
       │   gold_supporting=true   gold_supporting=false
       │          │                    │
       │          ▼                    ▼
       │  evidence_positive_     Negative candidates
       │      qa_failed                │
       │                               ▼
       │                       BM25 / QA hardness
       │                               │
       │                               ▼
       │                    Hard-negative candidates
       │                               │
       │                               ▼
       │                    False-negative filter
       │                               │
       │                      K-means embeddings
       │                               │
       │                        Gap Statistic
       │                               │
       │                   hard cluster assignment
       │                               │
       │                    ┌──────────┴──────────┐
       │                    │                     │
       │              same cluster as       different cluster
       │                 positive              from positive
       │                    │                     │
       │                    ▼                     ▼
       │               FN candidate        CLEAN HARD NEGATIVE
       │
       ▼
    POSITIVE

[optional ablation:
 fixed-τ Gumbel-Softmax]
```

Эта схема повторяется **независимо для каждого аспекта** одного документа при неизменных физических chunk boundaries.


# 76. Итог Phase 1 для каждого SciREX aspect-instance

Для каждого:

```text
(document_id, aspect)
```

где:

```text
aspect ∈ {method, task, metric, material}
```

и:

```text
status ∈ {ready, partially_extractable}
```

получить:

```text
ASPECT

QUESTION

COMMON DOCUMENT CHUNKS

EXTRACTABLE GOLD ANSWERS
— один или несколько допустимых source-only ответов

GOLD-SUPPORTING CHUNKS
— chunks с подтвержденными mentions данного аспекта

POSITIVE CHUNKS
— frozen QA дает один из extractable_gold_answers

EVIDENCE-POSITIVE QA-FAILED
— gold_supporting=true, но QA отвечает неправильно

FALSE-NEGATIVE CANDIDATES
— negative chunks, попавшие в тот же hard K-means cluster,
  что и positive для текущего аспекта

HARD-NEGATIVE CHUNKS
— сложные/релевантные для вопроса текущего аспекта,
  QA отвечает неправильно,
  gold_supporting=false,
  и hard cluster отличается от cluster positive
  (либо пройден optional Gumbel-filter в отдельном ablation)

EASY NEGATIVES
— остальные неправильные chunks этого аспекта
```

Не создавать основной Phase 1 instance для:

```text
status = no_extractable_gold
```

Не создавать отдельный aspect для:

```text
score
```

при `ENABLE_SCORE_ASPECT=False`.

Не использовать:

```text
external_enrichment
enrichment_candidates
```

для определения классов в baseline `E0`.

После этого начинается Phase 2 — поиск признаков, отделяющих `Positive` от `HardNegative`.


# 77. Главный исследовательский смысл

Phase 1 не должна сама улучшать QA.

Ее задача:

> **создать чистую постановку “почему два очень похожих чанка приводят QA-модель к разным результатам?”**

Только после надежного формирования `positive` и `hard-negative` можно честно проверять attention, question-aware attention, `diff_score`, ключевые слова, Question-KeyBERT и другие признаки как потенциальные механизмы выбора правильного чанка.
