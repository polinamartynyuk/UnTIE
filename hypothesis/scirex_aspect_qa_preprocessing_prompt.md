# Промпт: подготовка SciREX к aspect-level extractive QA и последующему анализу positive / hard-negative chunks

Ты — NLP-разработчик и исследователь.

Необходимо реализовать надежный preprocessing имеющегося представления **SciREX** для последующего эксперимента по выбору правильного чанка в extractive QA.

Основная задача preprocessing:

> Для каждого документа создать единый набор чанков, а затем для каждого извлекаемого тематического аспекта создать отдельный QA-instance с фиксированным вопросом, набором допустимых gold-ответов и информацией о том, какие чанки содержат доступные в тексте gold mentions.

На данном этапе **не искать hard negatives и не анализировать attention/ключевые слова**. Этот этап только формирует корректную основу для последующего запуска frozen QA.

---

# 1. Принцип источниковой чистоты и отдельный enrichment-слой

Базовая gold-разметка должна строиться только из:

1. полей текущей исходной записи SciREX;
2. содержимого `original_text`;
3. заранее заданной конфигурации эксперимента:
   - фиксированные вопросы для аспектов;
   - параметры chunking;
   - параметры нормализации;
   - настройки включения/исключения bibliography;
4. детерминированных преобразований исходных строк, явно описанных ниже.

При этом **допускается отдельно собирать дополнительную внешнюю или эвристически полученную информацию**, но только в независимом экспериментальном слое, который не смешивается с gold.

Внешний enrichment может включать, например:

- внешние синонимы;
- альтернативные названия методов и датасетов;
- расширенные аббревиатуры;
- ontology labels;
- варианты названий из внешних справочников;
- fuzzy/semantic candidates;
- сведения из другой версии статьи или таблиц, если они были специально подключены отдельным экспериментом.

Но такие сведения должны сохраняться только в отдельном поле, например:

```text
external_enrichment
```

или более детально:

```json
{
    "external_enrichment": [
        {
            "value": "...",
            "source_type": "...",
            "source_reference": "...",
            "relation_to_raw_gold": "...",
            "confidence": null,
            "used_for_gold": false
        }
    ]
}
```

Критическое правило:

> **ни одно значение из `external_enrichment` не может автоматически попадать в `canonical_gold_answers`, `extractable_gold_answers`, `gold_mentions` или `gold_supporting_chunk_ids`.**

Базовый режим должен оставаться полностью `source-only`.

Дополнительный enrichment сохраняется для будущего сравнения, например:

```text
E0 = source-only
E1 = source-only + external aliases
E2 = source-only + fuzzy/semantic enrichment
```

Если информация отсутствует в `original_text`, она остается **неизвлекаемой для baseline extractive QA**, даже если аналогичная информация была найдена во внешнем enrichment-слое.

Фиксированные QA-вопросы являются **конфигурацией эксперимента**, а не дополнительной фактической разметкой документа.

---

# 2. Формат исходных данных

Одна запись имеет приблизительно следующий формат:

```json
{
    "doc_id": "...",

    "methods": [...],
    "tasks": [...],
    "metrics": [...],
    "materials": [...],
    "score": [...],

    "original_text": "..."
}
```

Пример структуры:

```json
{
    "methods": [
        "fastText",
        "FastText"
    ],

    "tasks": [
        "Sentiment_Analysis",
        "Text_Classification"
    ],

    "metrics": [
        "Error",
        "Accuracy"
    ],

    "materials": [
        "Yelp_Binary_classification",
        "..."
    ],

    "score": [
        "7.5",
        "36.1",
        "..."
    ]
}
```

`original_text` содержит доступный текст статьи в сериализованном виде, например:

```text
document: ...
section: ...
subsection: ...
paragraph: ...
bibliography: ...
```

Не предполагать наличие каких-либо других полей или скрытой разметки, если их нет во входном объекте.

---

# 3. Критически важная особенность исходного представления

Не предполагать:

```text
каждая gold-сущность из metadata
обязательно встречается в original_text
```

Для текущего эксперимента необходимо различать:

```text
raw_gold_answers
```

и:

```text
extractable_gold_answers
```

## `raw_gold_answers`

Все значения соответствующего поля исходной записи.

## `extractable_gold_answers`

Только те gold entities, для которых удалось найти подтвержденное текстовое упоминание в `original_text`.

Если gold entity отсутствует в тексте, сохранить ее как:

```text
unextractable_gold_answer
```

но **не использовать как expected extractive QA span**.

Нельзя достраивать отсутствующее упоминание из внешних источников.

---

# 4. Какие аспекты использовать

Основные aspect-level QA-задачи:

```text
method
task
metric
material
```

Соответствие полям:

```text
method   <- methods
task     <- tasks
metric   <- metrics
material <- materials
```

## Поле `score`

Не включать `score` в основной эксперимент как обычный независимый тематический аспект.

Причина определяется самой структурой записи: `score` содержит набор числовых значений, но текущий формат не содержит явного однозначного отображения:

```text
score
→ конкретный method
→ конкретный task
→ конкретный metric
→ конкретный material
```

Поэтому вопрос вида:

```text
"What score is reported in the paper?"
```

может иметь несколько конкурирующих числовых ответов, а из текущих входных полей невозможно надежно установить отношение каждого числа к контексту без дополнительной relation-разметки.

Сохранить `score` в metadata, но по умолчанию:

```python
ENABLE_SCORE_ASPECT = False
```

Если позже появится явная relation-разметка, `score` можно обрабатывать отдельно как relation-conditioned QA target.

---

# 5. Фиксированные вопросы QA

Использовать **один фиксированный вопрос для каждого аспекта во всем корпусе**.

Например:

```python
ASPECT_QUESTIONS = {
    "method":
        "What method or model is used or proposed in the paper?",

    "task":
        "What task is addressed in the paper?",

    "metric":
        "What evaluation metric is used in the paper?",

    "material":
        "What dataset or material is used in the paper?"
}
```

Формулировки должны храниться в конфигурации.

Не генерировать новый вопрос отдельно для каждого документа.

Важно:

- вопросы — это экспериментальная конфигурация;
- они не должны использоваться для добавления новых фактических сущностей;
- ответами могут быть только сущности из соответствующего поля исходной записи, которые подтверждены текстом.

---

# 6. Один аспект может иметь несколько правильных ответов

Не предполагать:

```text
aspect → one gold answer
```

Корректная структура:

\[
aspect \rightarrow
\{a_1,a_2,\ldots,a_n\}.
\]

Например, поле `tasks` может содержать несколько значений.

Поэтому структура должна содержать:

```json
{
    "aspect": "task",

    "question":
        "What task is addressed in the paper?",

    "gold_answers": [
        "...",
        "..."
    ]
}
```

---

# 7. Не создавать один QA-instance на каждую сущность

Не делать:

```text
document + task question + answer_1
document + task question + answer_2
```

как независимые QA-example с совершенно одинаковым вопросом.

Это создало бы ситуацию, где одинаковый input искусственно имеет разные единственные answers.

Правильнее:

```text
document
+
one aspect question
+
set of valid answers
```

То есть:

```json
{
    "question": "What task is addressed in the paper?",

    "gold_answers": [
        "answer_1",
        "answer_2"
    ]
}
```

После запуска QA prediction считается корректным, если соответствует **любому допустимому extractable gold answer** данного аспекта.

---

# 8. Нормализация gold entities

Исходные labels могут отличаться от surface form текста только по техническому форматированию.

Разрешена только **детерминированная нормализация**, которая не добавляет новых смысловых сведений.

Минимум:

## Unicode

```python
unicodedata.normalize("NFKC", text)
```

## Underscores

Последовательность:

```text
_+
```

заменять одним пробелом.

Пример:

```text
Sentiment_Analysis
→
Sentiment Analysis
```

```text
Yahoo__Answers
→
Yahoo Answers
```

## Whitespace

Любую последовательность whitespace:

```text
\s+
```

нормализовать в один пробел.

## Dash normalization

Нормализовать эквивалентные unicode dash characters к единому символу.

## Matching case

Для поиска можно использовать:

```python
casefold()
```

но обязательно сохранять исходный surface form.

---

# 9. Что нормализация НЕ должна делать

Запрещено автоматически:

- лемматизировать сущность с изменением ее смысла;
- переводить сущность;
- подбирать синоним;
- расширять аббревиатуру;
- заменять название «более правильным» внешним названием;
- объединять семантически близкие, но разные сущности;
- подменять dataset другим dataset;
- связывать число `score` с объектом только по догадке;
- использовать fuzzy/embedding similarity как доказательство gold identity.

Нормализация должна изменять **форму записи**, но не содержательную идентичность.

---

# 10. Не терять оригинальные названия

Для каждой gold entity хранить:

```json
{
    "canonical": "...",

    "raw_variants": [
        "..."
    ],

    "normalized_variants": [
        "..."
    ]
}
```

Поле `canonical` в базовом `source-only` preprocessing не должно создаваться на основе внешнего справочника.

Его можно определить только из исходных вариантов по детерминированному правилу, например:

- первый исходный вариант;
- вариант с наиболее информативным регистром;
- либо отдельный стабильный normalized identity key.

В любом случае сохранить все `raw_variants`.

---

# 11. Deduplication

Возможны case variants:

```text
fastText
FastText
```

Они не должны автоматически считаться двумя разными gold answers только из-за регистра.

Сначала использовать normalized identity matching:

```python
identity_key =
    normalize_entity(entity).casefold()
```

Например:

```text
fastText
FastText
→
fasttext
```

Но при этом сохранить все surface aliases.

Не использовать semantic/fuzzy deduplication автоматически.

Например:

```text
BERT
SciBERT
```

нельзя объединять только потому, что названия семантически связаны.

---

# 12. Поиск gold mentions в тексте

Для каждого normalized gold entity найти все подтвержденные surface mentions в `original_text`.

Основной способ — детерминированное lexical matching.

Поддержать:

- case-insensitive match;
- `_ → space`;
- multiple whitespace;
- нормализацию эквивалентных дефисов;
- сохранение точных offsets.

Для каждого найденного mention сохранить:

```text
canonical_entity
raw_entity
matched_text
char_start
char_end
```

---

# 13. Fuzzy matching не является gold-разметкой

Допускается вычислять отдельно:

```text
fuzzy_candidate
```

или:

```text
semantic_candidate
```

только для диагностики.

Но такой кандидат **не должен автоматически становиться gold mention**.

Если точное или надежно нормализованное текстовое упоминание отсутствует:

```text
extractable = false
```

Это предпочтительнее, чем искусственно приписывать сущности неправильный span.

Если fuzzy-кандидаты сохраняются, хранить их в отдельном диагностическом/enrichment-поле и не использовать в baseline:

- при разметке `gold_supporting`;
- при определении `extractable_gold_answers`;
- при QA evaluation.

Например:

```json
{
    "enrichment_candidates": [
        {
            "value": "...",
            "method": "fuzzy_match",
            "score": 0.91,
            "used_for_gold": false
        }
    ]
}
```

Это позволит позднее сравнить baseline без enrichment и экспериментальную версию с enrichment, не пересобирая исходные данные.

---

# 14. Результат grounding

Для каждой сущности сохранить:

```json
{
    "canonical": "Accuracy",

    "extractable": true,

    "mentions": [
        {
            "text": "accuracy",
            "start": 123,
            "end": 131
        }
    ]
}
```

или:

```json
{
    "canonical": "Error",

    "extractable": false,

    "mentions": []
}
```

`extractable=true` разрешено выставлять только при наличии подтвержденного mention в текущем `original_text`.

---

# 15. Статус aspect-level QA

Для каждого аспекта определить:

```text
raw_gold_count
extractable_gold_count
```

Если:

\[
extractable\_gold\_count=0,
\]

установить:

```text
status = "no_extractable_gold"
```

Такой QA-instance:

- сохранить для статистики;
- **не использовать в основном extractive QA experiment**;
- не превращать все его chunks в negatives.

Иначе отсутствие данных в тексте искусственно создаст ложные negative examples.

---

# 16. Structural parsing `original_text`

Не удалять исходный `original_text`.

Сохранить:

```text
raw_text
```

без изменений.

Дополнительно построить structural representation, используя только маркеры, которые реально присутствуют в `original_text`.

Если встречаются:

```text
document:
section:
subsection:
paragraph:
bibliography:
```

их можно распознать как структурные markers.

Получить последовательность segments:

```json
[
    {
        "type": "document",
        "text": "...",
        "raw_start": 0,
        "raw_end": 123
    },

    {
        "type": "section",
        "title": "Introduction",
        "text": "...",
        "raw_start": 124,
        "raw_end": 900
    }
]
```

Не создавать section titles или section types, которых нет в исходном тексте.

---

# 17. Не разрушать структуру текста при очистке

Не делать:

```python
text = text.replace("section:", "")
```

без сохранения offsets.

Все преобразования должны либо:

1. сохранять raw offsets;

либо

2. строить explicit mapping:

```text
clean_char_index → raw_char_index
```

Это потребуется для восстановления evidence spans.

---

# 18. Bibliography

Bibliography сохранить в parsed document, если такой segment присутствует во входном тексте.

По умолчанию:

```python
INCLUDE_BIBLIOGRAPHY_IN_QA = False
```

Причина методологическая: bibliography может содержать mentions methods/tasks/datasets, но такие mentions могут относиться к цитируемым работам, а не к содержанию текущей статьи.

Важно:

- не удалять bibliography физически;
- не считать ее mentions основным QA evidence при `INCLUDE_BIBLIOGRAPHY_IN_QA=False`;
- при необходимости позднее анализировать bibliography отдельно.

---

# 19. Chunking выполнять один раз на документ

Не создавать разные физические тексты чанков отдельно для:

```text
method
task
metric
material
```

Вместо этого:

```text
document
    ↓
ONE CHUNKING
    ↓
chunk_0
chunk_1
...
chunk_N
```

А aspect-level instances должны ссылаться на эти же `chunk_id`.

Это обеспечивает идентичные границы context при сравнении разных аспектов.

---

# 20. Использовать существующий chunker проекта

Если в проекте уже есть стандартная процедура:

```text
sentence segmentation
→ tokenizer
→ overlapping token chunks
```

переиспользовать ее.

Не создавать вторую несовместимую реализацию.

Если chunker отсутствует, использовать tokenizer той QA-модели, которая будет применяться далее.

---

# 21. Token budget

Chunk должен гарантированно помещаться в QA model вместе с вопросом:

\[
L_q + L_c + L_{special}
\le
L_{model}.
\]

Чтобы границы chunks были одинаковыми для всех aspects, использовать context budget, рассчитанный по **самому длинному из фиксированных aspect questions**.

Не менять размер chunk в зависимости от конкретного aspect.

---

# 22. Overlap

Использовать фиксированный overlap из конфигурации проекта.

Не оптимизировать:

```text
chunk_size
chunk_overlap
```

одновременно с positive/hard-negative experiment.

Сначала зафиксировать preprocessing.

---

# 23. Для каждого chunk хранить

```json
{
    "document_id": "...",
    "chunk_id": 0,

    "text": "...",

    "token_start": 0,
    "token_end": 350,

    "char_start": 0,
    "char_end": 1600,

    "section_type": "section",
    "section_title": "Introduction",

    "eligible_for_qa": true
}
```

Если chunk пересекает несколько structural segments, не выдумывать единственный section title. В таком случае:

- сохранить список пересекаемых segment IDs;
- либо определить `section_type/section_title` только по явному правилу;
- либо оставить неоднозначное поле пустым.

---

# 24. Определение gold-supporting chunks

После chunking сопоставить найденные gold mentions с chunks.

Для каждого:

```text
aspect
×
chunk
```

определить:

```text
gold_supporting
```

Если chunk содержит хотя бы одно подтвержденное gold mention данного аспекта:

```python
gold_supporting = True
```

Сохранить также:

```text
gold_entities_in_chunk
gold_mentions_in_chunk
```

При этом использовать только подтвержденные `gold_mentions`, а не fuzzy/semantic candidates.

---

# 25. Overlapping chunks

Из-за overlap одна и та же gold mention может оказаться в нескольких chunks.

Это нормально.

Все такие chunks получают:

```text
gold_supporting = true
```

Но сохранить:

```text
mention_id
```

чтобы было понятно, что это одна и та же evidence mention, продублированная overlap.

---

# 26. Не определять `qa_positive` на этапе parsing

На данном preprocessing этапе известно только:

```text
gold_supporting
```

`qa_positive` появляется **только после реального запуска frozen QA model**.

Нельзя считать:

```text
gold_supporting == qa_positive
```

Это разные свойства.

Позже:

```text
gold_supporting = chunk содержит подтвержденное gold mention

qa_positive = QA на chunk вернула допустимый правильный answer
```

---

# 27. Aspect-level instance

Создать структуру вида:

```json
{
    "document_id": "...",

    "aspect": "task",

    "question":
        "What task is addressed in the paper?",

    "raw_gold_answers": [
        "Sentiment_Analysis",
        "Text_Classification"
    ],

    "canonical_gold_answers": [
        "Sentiment Analysis",
        "Text Classification"
    ],

    "extractable_gold_answers": [
        "Sentiment Analysis",
        "Text Classification"
    ],

    "unextractable_gold_answers": [],

    "external_enrichment": [],
    "enrichment_candidates": [],

    "chunk_ids": [
        0,
        1,
        2
    ],

    "gold_supporting_chunk_ids": [
        0,
        4
    ],

    "status": "ready"
}
```

В `canonical_gold_answers` не должно появляться значений, которых нельзя получить из `raw_gold_answers` разрешенной детерминированной нормализацией.

---

# 28. Если часть answers отсутствует

Например:

```text
raw metrics:
Error
Accuracy
```

но найден только:

```text
Accuracy
```

сохранить:

```json
{
    "extractable_gold_answers": [
        "Accuracy"
    ],

    "unextractable_gold_answers": [
        "Error"
    ],

    "status": "partially_extractable"
}
```

Не исключать весь aspect только потому, что один из answers отсутствует.

---

# 29. Статусы

Использовать:

```text
ready
partially_extractable
no_extractable_gold
```

## `ready`

Все raw gold entities данного аспекта после deduplication имеют хотя бы одно подтвержденное mention.

## `partially_extractable`

Хотя бы один gold доступен, но часть отсутствует.

## `no_extractable_gold`

Ни одного extractable answer.

---

# 30. Формат основного результата

Предпочтительно создать отдельные таблицы.

## `documents.parquet`

```text
document_id
raw_text
clean_text
```

## `chunks.parquet`

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

```text
document_id
aspect
question
status
raw_gold_answers
canonical_gold_answers
extractable_gold_answers
unextractable_gold_answers
external_enrichment
enrichment_candidates
chunk_ids
gold_supporting_chunk_ids
```

## `gold_mentions.parquet`

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

## `source_metadata.parquet` или эквивалент

Сохранить без потери исходные:

```text
doc_id
methods
tasks
metrics
materials
score
```


## `enrichment.parquet` — опционально

Сохранять дополнительную информацию отдельно от gold:

```text
document_id
aspect
raw_gold_answer
candidate_value
candidate_type
source_type
source_reference
matching_method
confidence
used_for_gold
```

Для baseline:

```text
used_for_gold = false
```

для всех строк этого файла.

Если enrichment-эксперимент позднее будет включен явно, его результаты должны оцениваться как отдельная конфигурация и не перезаписывать исходные `source-only` поля.

---

# 31. Сохранить исходную metadata

Не выбрасывать:

```text
methods
tasks
metrics
materials
score
```

Даже если часть данных не используется в QA.

Ни одно поле исходной записи не должно быть молча переписано внешними знаниями.

---

# 32. После preprocessing подготовить QA input

Только instances со статусом:

```text
ready
```

или:

```text
partially_extractable
```

должны поступать в основной QA experiment.

Для каждого:

```text
question
+
each eligible chunk
```

передаются frozen QA model.

---

# 33. Критерий QA-positive на следующей фазе

После inference:

```text
predicted_answer
```

сравнивается со всем множеством:

```text
extractable_gold_answers
```

QA chunk считается positive, если его prediction соответствует хотя бы одному допустимому gold answer.

Формально:

\[
qa\_positive(c,a)
=
\mathbb{1}
[
prediction(c,q_a)
\in
Gold_a
].
\]

Конкретную функцию `answer_match` сделать конфигурируемой.

Начать с:

```text
normalized_exact_match
```

Дополнительно позднее можно проверять:

```text
token_f1
```

Но это уже следующая фаза и не должно влиять на текущий parser.

---

# 34. Не использовать unextractable answers в QA evaluation

Если gold answer отсутствует в `original_text`, extractive QA физически не может извлечь его как span.

Поэтому такой:

```text
unextractable_gold_answer
```

не должен:

- использоваться как обязательный extractive span;
- превращать все chunks документа в negatives;
- автоматически считаться ошибкой QA.

Отдельно учитывать:

```text
extractability_rate
```

как характеристику подготовленного корпуса.

---

# 35. Обязательная статистика preprocessing

После обработки train/dev/test вывести по каждому aspect:

```text
documents
raw_gold_entities
unique_gold_entities

extractable_gold_entities
unextractable_gold_entities

ready_instances
partially_extractable_instances
no_extractable_gold_instances

gold_supporting_chunks
```

Отдельно:

\[
ExtractabilityRate
=
\frac{
N_{extractable}
}{
N_{raw\ unique\ gold}
}.
\]

В знаменателе использовать число gold entities после детерминированного deduplication, чтобы регистровые варианты одной сущности не искажали показатель.

---

# 36. Обязательный sanity-check на предоставленной записи

Для текущего примера проверить минимум:

## Methods

В исходном поле присутствуют:

```text
fastText
FastText
```

Они не должны автоматически превращаться в две независимые сущности только из-за регистра.

Должны сохраняться оба исходных варианта.

## Tasks

После `_ → space` корректно обрабатывать:

```text
Sentiment_Analysis
Text_Classification
```

Не добавлять никаких других task labels, которых нет в исходном поле `tasks`.

## Metrics

Проверять каждое значение независимо:

```text
Accuracy
Error
```

Если конкретное normalized mention отсутствует в `original_text`, соответствующая сущность остается `unextractable`.

Не подменять отсутствие сущности семантически похожим словом.

## Materials

Не считать datasets extractable только потому, что они присутствуют в metadata.

Если exact/normalized mention отсутствует:

```text
extractable = false
```

Не искать их во внешних версиях статьи.

## Scores

Не создавать fake spans для чисел, отсутствующих в тексте.

Не связывать score с method/task/metric/material без явной relation-информации во входных данных.

---

# 37. Unit tests

Написать тесты минимум для:

## Test 1 — underscores

```text
Text_Classification
→
Text Classification
```

## Test 2 — duplicate case variants

```text
fastText
FastText
```

должны давать один normalized identity key, сохраняя оба raw variants.

## Test 3 — absent entity

Metadata entity отсутствует в document:

```text
extractable = false
mentions = []
```

## Test 4 — partial extractability

Один aspect содержит два gold, но найден только один:

```text
status = partially_extractable
```

## Test 5 — overlapping chunks

Один gold mention входит в два chunks.

Оба chunks:

```text
gold_supporting = true
```

но используют один `mention_id`.

## Test 6 — offsets

Из:

```text
raw_text[start:end]
```

должен восстанавливаться ожидаемый `matched_text`.

## Test 7 — bibliography exclusion

Bibliography сохраняется, но не становится основным QA context при:

```python
INCLUDE_BIBLIOGRAPHY_IN_QA = False
```

## Test 8 — no gold

`no_extractable_gold` не создает negative-only QA instance.

## Test 9 — no external enrichment

Если входная entity имеет только одно конкретное название, parser не должен автоматически добавлять:

- синоним;
- аббревиатуру;
- перевод;
- внешнее canonical name.

## Test 10 — score isolation

Значения `score` сохраняются в metadata, но не превращаются в независимые QA gold spans при:

```python
ENABLE_SCORE_ASPECT = False
```

---

# 38. Не делать

Не следует:

- считать все metadata values extractive gold;
- использовать fuzzy matching как автоматический gold;
- использовать embeddings как автоматическое доказательство identity;
- создавать отдельный документ для каждого aspect;
- создавать отдельный QA-example для каждого gold entity при одном и том же вопросе;
- считать `gold_supporting` и `qa_positive` одним и тем же;
- превращать отсутствующие gold entities в negatives;
- использовать `score` как обычный independent aspect без relation context;
- терять offsets при очистке текста;
- удалять bibliography без возможности ее восстановить;
- запускать attention/diff/KeyBERT на этапе parsing;
- искать hard negatives до фиксации корректного QA dataset;
- использовать внешние знания для изменения baseline gold-разметки;
- автоматически исправлять baseline-названия сущностей «по смыслу»;
- добавлять entities, которых нет в соответствующем поле исходной записи, в baseline gold;
- смешивать `external_enrichment` с `canonical_gold_answers` или `extractable_gold_answers`;
- использовать значения из одного аспекта как gold другого аспекта;
- связывать `score` с другими полями по позиции массива без явного подтверждения формата данных.

---

# 39. Финальная ручная проверка

После реализации вручную проверить не менее 20 случайных документов.

Для каждого вывести:

```text
DOC_ID

ASPECT
QUESTION

RAW GOLD
NORMALIZED / CANONICAL GOLD
EXTRACTABLE GOLD
UNEXTRACTABLE GOLD

GOLD MENTIONS

GOLD-SUPPORTING CHUNKS
```

Дополнительно выводить:

```text
SOURCE CHECK:
- какие значения пришли из исходного поля;
- какие преобразования были выполнены;
- были ли добавлены новые фактические значения.
```

Для baseline gold-разметки всегда должно выполняться:

```text
external_values_used_for_gold = false
```

При этом дополнительный enrichment может присутствовать:

```text
external_enrichment_count >= 0
```

но он должен быть отделен от baseline и иметь явный provenance.

Если parser присваивает gold span тексту, которого фактически нет в `original_text`, считать это ошибкой preprocessing.

Если parser добавляет gold entity, которой нет в соответствующем исходном поле, считать это ошибкой preprocessing.

---

# 40. Автоматическая проверка отсутствия сторонней фактической информации

Для каждого aspect-instance выполнить invariant:

\[
NormalizedGold
\subseteq
Normalize(RawGold).
\]

То есть каждый `canonical_gold_answer` должен быть трассируем к одному или нескольким значениям `raw_gold_answers`.

Для каждого mention выполнить:

\[
matched\_text
=
raw\_text[char\_start:char\_end].
\]

Для каждого `extractable_gold_answer` должен существовать минимум один подтвержденный `mention_id`.

Для каждого `unextractable_gold_answer`:

```text
confirmed_mentions = 0
```

Никакая внешняя entity не может появиться в baseline-полях:

```text
canonical_gold_answers
extractable_gold_answers
gold_mentions
gold_supporting_chunk_ids
```

без явного изменения экспериментального режима.

Все внешние/эвристические значения должны находиться только в:

```text
external_enrichment
enrichment_candidates
```

и быть трассируемы по полям provenance.

---

# 41. Подготовить возможность будущего сравнения enrichment

Даже если внешний enrichment пока не используется, структура данных должна позволять впоследствии сравнить минимум:

## E0 — baseline source-only

Используются только:

```text
raw_gold_answers
→ deterministic normalization
→ confirmed mentions in original_text
```

## E1 — source + external aliases

Дополнительно разрешается использовать только заранее сохраненные внешние aliases/abbreviation variants.

## E2 — source + fuzzy/semantic candidates

Дополнительно используются кандидаты из:

```text
fuzzy matching
semantic matching
external enrichment
```

Но E1/E2 должны запускаться как **отдельные экспериментальные конфигурации**.

Нельзя перезаписывать исходный baseline dataset.

Для каждого результата QA позднее должно быть возможно сохранить:

```text
enrichment_mode = E0 / E1 / E2
```

и сравнить:

```text
extractability
gold-supporting chunk recall
qa_positive rate
final QA quality
hard-negative composition
```

Таким образом можно проверить эмпирически:

> дает ли внешнее/семантическое расширение реальное улучшение или, наоборот, добавляет шум.

---

# 42. Итоговая цель

После завершения preprocessing данные должны позволять выполнить следующий pipeline:

```text
SciREX document
      ↓
one common chunking
      ↓
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ METHOD QA   │ TASK QA     │ METRIC QA   │ MATERIAL QA │
└─────────────┴─────────────┴─────────────┴─────────────┘
      ↓
same physical chunks,
different aspect question + gold set
      ↓
Frozen RoBERTa QA
      ↓
qa_positive / qa_wrong
      ↓
hard-negative mining
      ↓
false-negative filtering
      ↓
Positive vs HardNegative feature analysis
```

Главный результат текущей задачи — **не QA-метрика**, а корректный, воспроизводимый aspect-level SciREX dataset, где:

1. один документ chunked только один раз;
2. каждый aspect имеет один фиксированный вопрос;
3. aspect может иметь несколько допустимых gold answers;
4. extractive gold существует только при подтвержденном mention в `original_text`;
5. отсутствующая в тексте разметка не превращается в false negative data;
6. parser не добавляет стороннюю фактическую информацию в baseline gold;
7. внешняя/эвристическая информация при необходимости сохраняется отдельно в enrichment-полях и не загрязняет baseline;
8. все нормализованные baseline-значения полностью трассируемы к исходной записи;
9. `gold_supporting` и будущий `qa_positive` остаются разными понятиями;
10. данные позволяют позднее сравнить `source-only` и `source + enrichment` режимы;
11. данные готовы к следующей фазе `positive → hard-negative candidates → false-negative filtering → поиск discriminative features`.
