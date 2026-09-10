# Инструкция по постановке четырех экспериментов для выбора контекста в экстрактивном QA

## 1. Цель исследования

Цель серии экспериментов — определить, какие способы отбора и совместной обработки фрагментов документа позволяют надежнее извлекать ответ в условиях, когда документ заранее разделен на чанки и среди них присутствуют:

- **positive chunks** — чанки, содержащие достаточную информацию для извлечения корректного ответа;
- **hard negatives** — тематически и семантически близкие к вопросу чанки, которые при этом **не содержат достаточной информации для корректного ответа**;
- **easy/random negatives** — нерелевантные или слабо релевантные чанки, которые существенно проще отличить от положительных.

Основная исследовательская гипотеза: для такой постановки недостаточно максимизировать только семантическую близость вопроса и чанка. Более эффективная система должна отдельно учитывать **тематическую релевантность**, **ответность (answerability)**, конкуренцию между несколькими кандидатами и, в наиболее полном варианте, структуру документа и взаимосвязи между фрагментами.

В качестве основной единицы оценки используется пара:

```text
(document_id, question_id)
```

Все чанки, относящиеся к одной такой паре, рассматриваются как единый набор кандидатов. Это принципиально: задача является **внутридокументным выбором контекста**, а не поиском по внешнему корпусу.

---

# 2. Общая организация экспериментов

## 2.1. Общий принцип сравнения

Эксперименты необходимо выстроить последовательно:

```text
A. relevance-based reranking
        ↓
B. hard-negative-aware answerability filtering
        ↓
C. multi-passage extractive QA with global normalization
        ↓
D. hybrid + structure-aware evidence pipeline
```

По возможности каждый следующий эксперимент должен наследовать компоненты предыдущего. Это необходимо, чтобы прирост качества можно было связывать с конкретным изменением, а не с одновременной заменой нескольких частей системы.

**Критическое требование:** нельзя менять одновременно датасет, разбиение, базовую QA-модель, метрики и процедуру оценки. Иначе результаты A–D будут несопоставимы.

## 2.2. Два режима проведения экспериментов

Эксперименты рекомендуется проводить в **двух взаимодополняющих режимах**.

### Режим I — полный pipeline

Используется, если требуется оценить систему так, как она будет работать на новых документах:

```text
все чанки документа
        ↓
retrieval / ranking
        ↓
candidate set
        ↓
answerability / multi-passage / structure-aware processing
        ↓
QA
```

В этом режиме метки `positive`, `hard_negative`, `easy_negative` используются **только для обучения и оценки**, но **никогда не используются для выбора чанков на test во время inference**.

### Режим II — контролируемый candidate-set experiment

Используется, если чанки и их метки уже заранее определены и основная цель — исследовать именно способность метода различать positive и hard negative, не смешивая этот эффект с ошибками initial retrieval.

Для каждого вопроса заранее формируется фиксированный набор:

```text
C_fixed(q) =
    all positives
  + all hard negatives
  + фиксированная выборка easy negatives
```

или, если число чанков в документе невелико:

```text
C_fixed(q) = все чанки текущего документа
```

После этого A–D работают **на одном и том же фиксированном candidate set**.

Ключевое правило:

> gold-метки допустимо использовать для формирования контролируемого набора **до эксперимента**, но не для ранжирования, фильтрации или выбора чанков внутри test inference.

Режим II нужен для ответа на вопрос:

> «Если правильный чанк и hard negatives уже находятся перед моделью, умеет ли конкретный метод выбрать/использовать evidence правильно?»

Режим I нужен для ответа на другой вопрос:

> «Как хорошо работает вся система целиком, включая initial retrieval?»

Оба результата желательно приводить отдельно. Это существенно повышает интерпретируемость исследования.

---

# 3. Подготовка данных

## 3.1. Рекомендуемая структура одной записи

Минимально запись должна содержать:

```json
{
  "document_id": "...",
  "question_id": "...",
  "question": "...",
  "gold_answers": ["..."],
  "chunks": [
    {
      "chunk_id": "...",
      "text": "...",
      "section_id": "...",
      "section_title": "...",
      "position": 0,
      "label": "positive | hard_negative | easy_negative | ambiguous"
    }
  ],
  "gold_spans": [
    {
      "chunk_id": "...",
      "start_char": 0,
      "end_char": 0,
      "text": "..."
    }
  ]
}
```

Если структурные поля `section_id`, `section_title` отсутствуют, эксперименты A–C можно выполнять без них. Для полного эксперимента D рекомендуется восстановить хотя бы:

- порядок чанков;
- принадлежность к разделу;
- заголовок раздела;
- связь с предыдущим и следующим чанком.

## 3.2. Определение positive и hard negative

### Positive chunk

Чанк считается положительным, если из него можно извлечь корректный ответ на вопрос без добавления информации, отсутствующей в данном контексте.

Если корректный ответ встречается более чем в одном чанке, **все такие чанки должны считаться положительными**. Нельзя принудительно выбирать один «единственный» positive, если фактически ответ поддерживается несколькими контекстами.

### Hard negative

Чанк считается hard negative, если выполняются одновременно условия:

1. он тематически близок к вопросу;
2. содержит связанные сущности, термины, методы или формулировки;
3. может получать высокий retrieval/reranking score;
4. при этом не содержит достаточного основания для правильного ответа.

### Ambiguous

Отдельно необходимо завести класс `ambiguous` для случаев, когда:

- чанк содержит частичный ответ;
- присутствует альтернативная корректная формулировка;
- gold-разметка неполна;
- невозможно однозначно решить, должен ли чанк считаться отрицательным.

**Ambiguous-примеры нельзя автоматически использовать как hard negatives при обучении.** Их следует исключить из основного loss либо анализировать отдельно.

Это особенно важно, поскольку исследования hard-negative mining показывают риск ложных отрицательных примеров. RocketQA использует именно *denoised hard negatives*, а работы по false-negative contexts отдельно указывают, что близкие контексты могут быть ошибочно размечены как отрицательные [1, 5].

---

# 4. Контроль качества hard negatives

Если метки чанков **еще не проверены**, перед запуском моделей необходимо провести отдельную проверку разметки hard negatives.

Если же чанки и их метки (`positive`, `hard_negative`, `easy_negative`, при необходимости `ambiguous`) **уже определены и верифицированы в предыдущем этапе работы**, повторный mining или повторная ручная разметка не требуются. В этом случае раздел 4 выполняется как **sanity audit**, а не как повторное создание меток.

Минимальный sanity audit для уже готовой разметки:

1. проверить уникальность `document_id`, `question_id`, `chunk_id`;
2. убедиться, что для каждого answerable-вопроса существует хотя бы один positive;
3. убедиться, что hard negative не совпадает с positive по `chunk_id`;
4. проверить, нет ли буквального или нормализованного gold answer в hard negative;
5. случайно вручную проверить небольшую стратифицированную выборку positive/hard-negative;
6. зафиксировать версию разметки и больше не изменять ее после формирования split.

Если все проверки пройдены, использовать существующие метки напрямую.

## 4.1. Автоматические проверки

Для каждого hard negative сохранить:

```text
question
hard_negative_text
gold_answer
answer_string_present
normalized_answer_present
semantic_answer_similarity
```

Проверка наличия строки ответа полезна, но **не должна быть единственным критерием**. Отсутствие буквального ответа не доказывает неответность, если в тексте присутствует эквивалентная формулировка.

## 4.2. Проверка через extractive QA

До обучения reranker-а можно прогнать базовую QA-модель по размеченным hard negatives и сохранить:

- лучший предсказанный span;
- start logit;
- end logit;
- итоговый span score;
- текст предсказания;
- сходство с gold answer.

Hard negatives, на которых QA уверенно извлекает правильный или практически эквивалентный gold answer, необходимо повторно проверить.

## 4.3. Итоговая маркировка

Рекомендуемые поля:

```text
label_original
label_verified
verification_reason
verification_method
```

В основной обучающий набор B–D включать только `hard_negative_verified=true`.

---

# 5. Разбиение train / validation / test

## 5.1. Разбивать только по документам

Если один документ содержит несколько вопросов или аспектов, все записи этого документа должны находиться **только в одном** подмножестве.

Правильно:

```text
Document 1 → train
Document 2 → train
Document 3 → validation
Document 4 → test
```

Неправильно:

```text
Document 1 / question A → train
Document 1 / question B → test
```

Иначе одинаковые или соседние чанки могут попасть одновременно в обучение и тест, что создаст утечку.

### Если разметка уже существует

Если positive/hard-negative labels были созданы до настоящего эксперимента, это не проблема само по себе. Критично другое: никакие модели/пороги, использовавшие test labels, не должны затем оцениваться на том же test как на независимом наборе.

Если hard negatives были получены алгоритмом, обученным на всей коллекции, необходимо проверить, не использовал ли этот алгоритм test gold answers или test labels. Если использовал — split следует пересобрать на более раннем уровне или повторно сформировать hard negatives только допустимым способом.

## 5.2. Фиксация разбиения

Один раз создать:

```text
splits.json
```

и использовать его во **всех четырех экспериментах**.

После начала экспериментов разбиение не изменять.

## 5.3. Настройка гиперпараметров

Все пороги и параметры выбирать только по validation:

- `top_k`;
- число passages для C;
- threshold answerability;
- коэффициенты hybrid retrieval;
- число seed chunks;
- число соседей графа;
- порог semantic edge;
- число evidence sentences;
- коэффициенты answer aggregation.

Test используется **однократно для финального сравнения выбранных конфигураций**.

---

# 6. Общие модели и компоненты

Для корректного сравнения желательно фиксировать:

1. одну модель начального dense retrieval;
2. один tokenizer;
3. один QA backbone для A, B, C и D;
4. одинаковую максимальную длину входа;
5. одинаковые train/validation/test splits;
6. одинаковый preprocessing;
7. одинаковые random seeds;
8. одинаковый набор gold answers.

Если архитектура конкретного эксперимента требует изменения QA reader-а, это изменение должно быть явно зафиксировано как экспериментальный фактор.

---

# 7. Общая система метрик

Оценивать нужно **не только итоговый QA**, но и каждый уровень pipeline.

## 7.1. Метрики выбора чанков

### Hit@K / Positive Recall@K

Есть ли хотя бы один positive в первых K кандидатах:

\[
Hit@K = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[P_i \cap TopK_i \neq \varnothing].
\]

### Positive Coverage@K

Если positive chunks несколько:

\[
Coverage@K_i = \frac{|P_i \cap TopK_i|}{|P_i|}.
\]

### MRR

Ранг первого положительного чанка:

\[
MRR = \frac{1}{N}\sum_i \frac{1}{rank_i^+}.
\]

Если в конкретной постановке допускаются вопросы без positive chunk, для них reciprocal rank задается равным 0 и такие случаи дополнительно анализируются отдельно как unanswerable.

### Hard-Negative@K

Доля hard negatives среди первых K:

Если число доступных кандидатов всегда не меньше K:

\[
HN@K = \frac{1}{N}\sum_i \frac{|HN_i \cap TopK_i|}{K}.
\]

Если у некоторых документов число кандидатов меньше K, использовать эффективный размер:

\[
K_i^{eff}=\min(K,|C_i|),
\]

\[
HN@K = \frac{1}{N}\sum_i \frac{|HN_i \cap TopK_i|}{K_i^{eff}}.
\]

Чем меньше значение, тем лучше.

### Pairwise positive-vs-hard-negative accuracy

Для каждой пары `(positive, hard_negative)`:

\[
Acc_{pair}=P(s(q,c^+) > s(q,c^-_{hard})).
\]

Эта метрика особенно важна для эксперимента B.

## 7.2. Метрики answerability

Для B и всех последующих систем:

- Precision;
- Recall;
- F1;
- PR-AUC;
- ROC-AUC;
- confusion matrix.

При сильном дисбалансе классов основной метрикой лучше считать **PR-AUC и F1 positive/answerable class**, а не accuracy.

Дополнительно можно оценивать калибровку:

- Brier score;
- Expected Calibration Error.

## 7.3. Метрики итогового QA

Минимальный набор:

- Exact Match;
- Token F1;
- Character F1;
- ROUGE-L F1;
- BERTScore F1;
- доля случаев, где система вернула `no answer`;
- coverage — доля вопросов, для которых вообще был получен ответ.

Если используются несколько допустимых gold answers, оценку выполнять относительно лучшего совпадения с любым допустимым вариантом.

## 7.4. Ресурсные метрики

Для каждого эксперимента сохранять:

- среднее время на один вопрос;
- p50/p95 latency;
- количество обработанных чанков;
- количество токенов, переданных QA reader-у;
- максимальную GPU memory;
- размер дополнительных моделей.

Это позволит оценить не только качество, но и вычислительную цену улучшения.

---

# 8. Эксперимент A — сильный relevance-based baseline

## 8.1. Цель

Построить **реально сильный generic relevance baseline**, который ранжирует чанки по релевантности вопросу, но не обучается на вашей специальной разметке `positive vs hard_negative` как на задаче answerability.

Это важно: baseline нельзя искусственно ослаблять только easy negatives, иначе преимущество B будет частично создано самой постановкой обучения.

Допустимые варианты A:

1. готовый сильный Cross-Encoder reranker, обученный на общей задаче passage relevance;
2. reranker, обученный на внешнем retrieval-датасете;
3. если внешняя модель недоступна — task-specific relevance model, но тогда необходимо отдельно описать, чем ее целевая функция отличается от answerability B.

Предпочтителен вариант 1, поскольку он дает наиболее чистое сравнение:

```text
A = generic relevance
B = task-specific answerability learned from labeled hard negatives
```

## 8.2. Гипотеза

Generic semantic relevance должна хорошо отделять явно нерелевантные чанки, однако может путать positive и hard negatives, поскольку оба типа тематически близки вопросу, а релевантность не тождественна достаточности evidence для ответа.

## 8.3. Pipeline

```text
all chunks of current document
        ↓
dense retrieval
        ↓
top-K candidates
        ↓
Cross-Encoder relevance reranker
        ↓
top-1 chunk
        ↓
extractive QA
        ↓
answer
```

## 8.4. Candidate pool

Поиск выполнять **только среди чанков текущего документа**.

Рекомендуемый начальный диапазон:

```text
K ∈ {10, 20, 30}
```

Выбрать K по validation, ориентируясь прежде всего на `Hit@K`, а затем на вычислительную цену reranking.

## 8.5. Модель relevance reranker

**Основной рекомендуемый вариант:** не дообучать A на вашей разметке чанков вообще, а использовать заранее обученный generic Cross-Encoder relevance reranker.

Это устраняет методическую проблему прежней версии плана, где A предлагалось обучать только на easy negatives: такой baseline был бы заведомо слабее B.

Если требуется обучаемый baseline с тем же backbone, проводить его как отдельную абляцию `A_train`, но не подменять им основной A. В `A_train` необходимо заранее формально определить, что считается relevance, не используя answerability-label напрямую.

Для основной таблицы:

```text
A = generic/off-the-shelf relevance Cross-Encoder
B = тот же или сопоставимый backbone, дообученный на answerability labels
```

Если backbone A и B различаются, дополнительно выполнить абляцию с одинаковым backbone, чтобы исключить влияние размера/архитектуры модели.

## 8.6. Inference

### Полный режим

Использовать `dense → top-K → relevance reranker`.

### Контролируемый режим при уже готовых чанках и метках

Если число чанков одного документа умеренное, initial dense retrieval можно **временно исключить из controlled experiment**:

```text
все chunks документа / C_fixed(q)
        ↓
generic relevance Cross-Encoder
        ↓
ranking
```

Это дает чистую оценку способности relevance-модели ранжировать positive выше hard negatives.

Важно: метки используются только после scoring для вычисления метрик.

Для каждого из candidate chunks Cross-Encoder возвращает:

```text
relevance_score
```

Берется чанк с максимальным score.

QA запускается только на нем.

## 8.7. Что сохранять

Для каждого вопроса:

```text
dense_top_k
reranker_scores
selected_chunk_id
selected_chunk_label
qa_answer
qa_start_logit
qa_end_logit
qa_score
gold_answer
```

## 8.8. Основные результаты A

Нужно получить:

1. Hit@1/5/10 после dense retrieval;
2. Hit@1/5 после reranking;
3. MRR;
4. HN@1 и HN@5;
5. итоговые QA-метрики.

### Ключевой диагностический показатель

```text
P(selected chunk = hard negative)
```

Он покажет, насколько проблема hard negatives действительно ограничивает baseline.

---

# 9. Эксперимент B — answerability reranker, обученный на hard negatives

## 9.1. Цель

Проверить, дает ли явное обучение различению:

```text
answer-bearing positive
vs.
semantically similar but unanswerable hard negative
```

улучшение выбора контекста.

## 9.2. Отличие от A

Архитектуру Cross-Encoder желательно **не менять**.

Меняется смысл задачи и обучающая выборка:

```text
A: relevance
B: answerability / evidence sufficiency
```

Это делает A → B интерпретируемым сравнением.

## 9.3. Pipeline

```text
all chunks of current document
        ↓
the same dense retriever as A
        ↓
the same top-K candidate pool
        ↓
answerability Cross-Encoder
        ↓
P(answerable | question, chunk)
        ↓
top-1 / filtered candidate set
        ↓
extractive QA
```

## 9.4. Обучающие данные

Если чанки и метки уже определены, **hard-negative mining внутри эксперимента B не выполнять**. Использовать готовые labels как supervised dataset.

При этом split формируется сначала по `document_id`, и только затем из train-документов создаются обучающие пары. Нельзя сначала создать пары, а затем случайно делить их на train/test.

Positive:

```text
(question, positive_chunk, 1)
```

Negative:

```text
(question, verified_hard_negative, 0)
```

Easy negatives можно добавлять, но они не должны доминировать.

Рекомендуется проверить на validation несколько соотношений:

```text
positive : hard_negative : easy_negative
1 : 1 : 0
1 : 2 : 1
1 : 3 : 1
```

Выбор выполнять по PR-AUC и pairwise positive-vs-hard-negative accuracy.

## 9.5. Loss

Базовый вариант:

\[
L_{cls}=-y\log p -(1-y)\log(1-p).
\]

Дополнительная абляция — pairwise ranking loss:

\[
L_{pair}=\max(0,m-s^+ + s^-).
\]

Не следует сразу смешивать несколько loss в основной конфигурации. Сначала сравнить:

```text
B1 = binary answerability loss
B2 = pairwise loss
```

и только после этого, при необходимости:

```text
B3 = combined loss
```

## 9.6. False-negative protection

Перед обучением hard negatives должны быть очищены от:

- альтернативных positives;
- контекстов с эквивалентным answer span;
- неоднозначных фрагментов.

Это согласуется с выводами RocketQA о необходимости denoised hard negatives [1] и с работами о false-negative contexts [5].

## 9.7. Thresholding

На validation выбрать threshold:

\[
P(answerable)>\tau.
\]

Однако для сравнения A и B обязательно дополнительно сохранить **ranking mode без threshold**, где выбирается просто максимальный answerability score.

Это позволит разделить:

- качество ранжирования;
- качество решения `answerable / unanswerable`.

## 9.8. Главные сравнения

### A vs B

Сравнить:

- Pairwise Accuracy positive > hard negative;
- HN@1;
- Hit@1;
- MRR;
- итоговый QA F1/EM.

### Отдельно по сложности

Разбить test на:

```text
0 hard negatives
1 hard negative
2–3 hard negatives
4+ hard negatives
```

и проверить, растет ли преимущество B при увеличении числа сложных отрицательных контекстов.

Это особенно важно для проверки основной гипотезы.

---

# 10. Эксперимент C — multi-passage extractive QA с глобальной нормализацией

## 10.1. Цель

Проверить, обязательно ли выбирать один единственный чанк до QA, либо положительный и hard-negative contexts лучше заставить **конкурировать непосредственно на уровне answer spans**.

## 10.2. Основание

Multi-passage BERT был предложен именно из-за того, что при независимой обработке passages оценки ответов из разных passages могут быть несопоставимыми. В работе используется глобальная нормализация answer scores по passages одного вопроса [2].

## 10.3. Справедливое сравнение с B

В полном режиме candidate generator в C должен быть тем же, что в B.

То есть:

```text
same document
  ↓
same dense retriever
  ↓
same answerability model B
  ↓
same ranked candidates
```

Меняется только финальный reader:

```text
B → top-1 chunk → ordinary QA
C → top-M chunks → global multi-passage QA
```

### Контролируемый режим

При заранее известных и размеченных чанках выполнить дополнительное сравнение:

```text
C_fixed(q)
   ↓
A или B scores
   ↓
top-M без использования gold label
   ↓
global multi-passage QA
```

И отдельно — **oracle diagnostic**, где positive принудительно присутствует среди M. Oracle-режим не является рабочей системой и не должен смешиваться с основным результатом.

## 10.4. Число passages

Проверить на validation:

```text
M ∈ {2, 3, 5, 8}
```

Для каждого M обязательно считать:

- candidate Hit@M;
- QA EM/F1;
- latency;
- processed tokens.

Не выбирать M по test.

## 10.5. Критически важное требование реализации

**Нельзя** считать корректной реализацией Multi-passage BERT простой выбор максимального `softmax confidence`, полученного независимо внутри каждого чанка.

Корректный принцип для воспроизведения идеи global normalization:

1. получить **ненормализованные start/end logits** для токенов всех M passages;
2. не выполнять независимую финальную калибровку confidence внутри каждого passage с последующим сравнением этих confidence;
3. задавать общую нормализацию по passages одного вопроса;
4. при обучении использовать согласованную с inference глобальную целевую функцию.

Конкретная формула loss должна соответствовать выбранной реализации Multi-passage reader-а и быть зафиксирована в коде/конфигурации. Ниже приведена допустимая схема, но она является **экспериментальной адаптацией**, а не утверждением, что это единственно возможная реализация исходной статьи.

Условно:

\[
P_s(c,t)=
\frac{\exp z_s(c,t)}
{\sum_{c'\in C_M}\sum_{t'}\exp z_s(c',t')}.
\]

Аналогично для end logits.

Для span:

\[
S(c,s,e)=z_s(c,s)+z_e(c,e).
\]

## 10.6. Несколько корректных gold spans

Если ответ встречается в нескольких passages, нельзя обучать модель только на одном произвольно выбранном span.

Использовать множество допустимых gold spans:

\[
G=\{(c,s,e)_1,\ldots,(c,s,e)_m\}.
\]

Loss должен учитывать суммарную вероятность допустимых вариантов, например через marginal log-likelihood:

\[
L=-\log\sum_{g\in G}P(g).
\]

Это предотвращает наказание модели за выбор другого корректного positive passage.

## 10.7. Unanswerable / no-answer

Если в датасете **каждый вопрос гарантированно имеет ответ хотя бы в одном чанке**, отдельный `no answer` класс в C не нужен.

Если имеются genuinely unanswerable вопросы, multi-passage reader должен иметь явно определенный null candidate (например, специальный null span / CLS-based score), и его score также должен конкурировать в общем пространстве. Нельзя сравнивать no-answer score, вычисленный по другой несопоставимой шкале.

## 10.8. Состав candidate set во время обучения

Каждый training set для вопроса должен по возможности содержать:

```text
≥1 positive
+ hard negatives
+ при необходимости easy negatives
```

Если positives несколько, полезно периодически менять их сочетание с hard negatives между эпохами.

## 10.9. Oracle-анализ

Обязательно посчитать два результата:

### Реальный C

```text
C_actual = candidates from B
```

### Oracle C

```text
C_oracle = candidate set, в котором гарантированно присутствует positive
```

Разница:

```text
C_oracle - C_actual
```

показывает, сколько качества теряется из-за retrieval/filtering, а сколько — из-за reader-а.

## 10.10. Основные сравнения

```text
B(top-1 ordinary QA)
vs.
C(top-M global QA)
```

Особенно анализировать случаи:

- top-1 = hard negative, но positive находится на позициях 2–5;
- несколько positives;
- несколько hard negatives с высоким score;
- разные passages дают разные answer spans.

---

# 11. Эксперимент D — структурно-ориентированная система evidence selection

## 11.1. Цель

Проверить более общую гипотезу: для научно-технического документа лучший контекст можно получать не выбором одного изолированного чанка, а **формированием компактного набора взаимосвязанных evidence units** с учетом:

- dense similarity;
- lexical matching;
- структуры документа;
- соседства чанков;
- sentence-level evidence;
- нескольких конкурирующих answer spans.

## 11.2. Почему D нельзя тестировать только одной конфигурацией

D содержит несколько новых механизмов. Если сразу сравнить:

```text
C vs D_full
```

и D окажется лучше, будет невозможно установить причину прироста.

Поэтому D должен выполняться как **абляционная лестница**.

```text
D0 = C
D1 = D0 + hybrid retrieval
D2 = D1 + structural reranking
D3 = D2 + graph/neighbor expansion
D4 = D3 + sentence-level evidence filtering
D5 = D4 + answer aggregation
```

Каждый шаг тестируется отдельно.

---

# 12. D1 — hybrid dense + lexical retrieval

## 12.1. Мотивация

Dense retrieval хорошо улавливает семантическую близость, однако hard negatives как раз могут быть семантически близкими. Lexical retrieval может дополнительно выделять специфичные термины, обозначения, названия методов и другие точные признаки.

## 12.2. Реализация

Для каждого вопроса независимо получить два ranking:

```text
dense_rank
BM25_rank
```

Предпочтительный первый вариант объединения — **Reciprocal Rank Fusion**, поскольку он не требует напрямую калибровать несовместимые шкалы score:

\[
RRF(c)=\sum_{r\in R}\frac{1}{k+rank_r(c)}.
\]

`k` подобрать на validation или использовать фиксированное стандартное значение, заранее записав его в конфигурацию.

Дополнительная абляция:

\[
S=\alpha S_{dense}+(1-\alpha)S_{lexical},
\]

но только после нормализации шкал.

## 12.3. Сравнение

```text
D0 vs D1
```

Оценивать прежде всего:

- Hit@K;
- HN@K;
- MRR;
- final QA.

---

# 13. D2 — учет структуры документа

## 13.1. Минимальная структура

Для каждого чанка восстановить:

```text
section_id
section_title
chunk_position
previous_chunk_id
next_chunk_id
```

## 13.2. Структурные признаки

Можно использовать:

- relevance вопроса к заголовку раздела;
- принадлежность нескольких сильных кандидатов одному разделу;
- расстояние до другого сильного candidate chunk;
- положение внутри документа;
- соседство с positive-like seed chunk.

Итоговый score, например:

\[
S(c)=w_1S_{retrieval}+w_2S_{answerability}+w_3S_{section}.
\]

Коэффициенты подбирать только на validation.

## 13.3. Обязательная абляция

Сравнить:

```text
retrieval + answerability
vs.
retrieval + answerability + section score
```

Dense Hierarchical Retrieval показывает, что passage ranking может улучшаться при использовании одновременно локальной семантики passage и более общего document/section context [4].

---

# 14. D3 — граф и расширение evidence

## 14.1. Узлы

Каждый chunk — узел графа.

## 14.2. Безопасный минимальный набор ребер

Использовать только отношения, которые можно однозначно получить из документа:

```text
ADJACENT       — соседние chunks
SAME_SECTION   — chunks одного раздела
SEMANTIC       — similarity выше threshold
```

Не начинать с автоматически извлеченных сложных knowledge-graph relations: это добавит дополнительный источник ошибок и усложнит интерпретацию эксперимента.

## 14.3. Seed nodes

Выбрать `r` лучших chunks после D2:

```text
r ∈ {1, 2, 3}
```

## 14.4. Expansion

Для каждого seed рассмотреть:

- предыдущий chunk;
- следующий chunk;
- несколько наиболее близких chunks того же section;
- semantic neighbors выше validation threshold.

После expansion повторно пересчитать answerability/relevance и оставить ограниченное число evidence chunks.

## 14.5. Token budget

Нельзя позволять expansion бесконтрольно увеличивать контекст.

Заранее зафиксировать:

```text
max_evidence_chunks
max_total_tokens
max_graph_hops
```

Для первого эксперимента рекомендуется `max_graph_hops = 1`.

Только если D3 дает устойчивый прирост, проверять 2-hop expansion.

---

# 15. D4 — sentence-level evidence filtering

## 15.1. Цель

Не передавать downstream QA весь отобранный chunk, если полезными являются только 1–3 предложения.

Pipeline:

```text
selected / expanded chunks
        ↓
sentence segmentation
        ↓
(question, sentence) evidence score
        ↓
top evidence sentences
        ↓
compact evidence context
```

## 15.2. Разметка предложений

Если известен gold answer span:

- предложение, содержащее gold span → `positive evidence`;
- предложения из verified hard-negative chunks → `negative evidence`;
- остальные предложения positive chunk → **не считать автоматически жесткими negatives**.

Последний пункт важен: соседнее предложение может содержать необходимое условие или контекст, даже если буквальный answer span находится не в нем.

Такие предложения можно помечать как:

```text
unlabeled / context
```

## 15.3. Модель

Первый вариант — небольшой Cross-Encoder:

```text
(question, sentence) → evidence score
```

Более сложный вариант с contrastive evidence objective выполнять только как дополнительную абляцию. В long-context QA дополнительный contrastive signal для различения supporting evidence и negative sentences показывал улучшение качества [6].

## 15.4. Сборка evidence context и сохранение extractive-свойства

После отбора предложения необходимо вернуть в исходный порядок документа.

Для каждого selected sentence сохранить исходные координаты:

```text
source_chunk_id
source_start_char
source_end_char
```

Если предложения конкатенируются в новый compact context, необходимо хранить отображение:

```text
assembled_offset → source_chunk_id/source_offset
```

чтобы любой финальный answer span можно было однозначно спроецировать обратно в исходный документ.

Иначе система формально перестает быть полностью проверяемым extractive QA pipeline.


Правильно:

```text
s3 → s8 → s11
```

в соответствии с исходной позицией.

Не сортировать финальный текст по relevance score, поскольку это разрушает локальную связность.

## 15.5. Соседний контекст

Проверить две конфигурации:

```text
D4a: только selected sentence
D4b: selected sentence ± 1 соседнее предложение
```

Это позволит установить, достаточно ли точечного evidence или QA нужна локальная связность.

---

# 16. D5 — answer aggregation

## 16.1. Цель

Если несколько evidence contexts приводят к одному и тому же ответу, использовать согласованность нескольких независимых свидетельств как дополнительный сигнал.

Исследования answer re-ranking показывают, что агрегирование evidence из нескольких passages может улучшать окончательный выбор ответа [3].

## 16.2. Базовый вариант

Для каждого candidate span сохранить:

```text
answer_text
normalized_answer_text
span_score
source_chunk_id
source_sentence_ids
```

Сначала группировать ответы только по строгой нормализованной строке:

- lowercase;
- нормализация пробелов;
- нормализация пунктуации;
- при необходимости детерминированная морфологическая нормализация.

## 16.3. Score группы

Перед агрегированием необходимо устранить искусственное многократное подтверждение, возникающее из-за overlapping chunks.

Если один и тот же фрагмент исходного документа попал в несколько перекрывающихся чанков, эти ответы **не считать независимыми evidence**.

Сначала сгруппировать candidate spans по исходным координатам/сильному перекрытию и оставить один вклад на одну исходную evidence region.

После de-duplication, например:

\[
S(a)=\log\sum_{j\in E_{unique}(a)}\exp(s_j).
\]

где `E_unique(a)` — уникальные evidence regions, давшие тот же нормализованный ответ.

Это предотвращает ложный прирост score только потому, что один и тот же ответ оказался продублирован sliding-window chunking.

## 16.4. Семантическая группировка

Ответы вроде:

```text
convolutional neural network
CNN
```

могут быть семантически эквивалентны, но semantic clustering добавляет риск ошибочного объединения.

Поэтому:

```text
D5a = exact normalized aggregation
D5b = semantic aggregation
```

`D5b` считать отдельной абляцией с threshold, выбранным на validation.

---

# 17. Финальная конфигурация D_full

После завершения абляций выбрать только компоненты, которые показали устойчивый положительный эффект на validation.

Пример итоговой схемы:

```text
DOCUMENT
   ↓
all chunks
   ↓
dense + BM25 retrieval
   ↓
rank fusion
   ↓
answerability reranker
   ↓
structural reranking
   ↓
seed chunks
   ↓
1-hop structural/semantic expansion
   ↓
sentence evidence selection
   ↓
compact evidence units
   ↓
globally normalized multi-passage QA
   ↓
answer aggregation
   ↓
FINAL ANSWER
```

**Необязательно**, чтобы D_full включал все элементы D1–D5. Если компонент не дает достоверного улучшения или слишком дорог, его следует исключить.

---

# 18. Единый порядок выполнения экспериментов

## Этап 0. Зафиксировать данные

- проверить positive/hard-negative разметку;
- создать `ambiguous`;
- зафиксировать chunks;
- зафиксировать preprocessing;
- создать document-level split.

## Этап 1. Получить retrieval-only baseline

До A желательно сохранить дополнительный A0:

```text
A0 = dense retrieval → top-1 → QA
```

Он покажет реальную ценность Cross-Encoder reranking.

## Этап 2. Эксперимент A

```text
dense → relevance CE → top-1 → QA
```

## Этап 3. Эксперимент B

```text
dense → answerability CE trained with hard negatives → top-1 → QA
```

## Этап 4. Эксперимент C

```text
dense → B answerability ranking → top-M → globally normalized QA
```

## Этап 5. D1–D5

Добавлять по одному новому компоненту.

## Этап 6. D_full

Собрать финальную систему только из подтвержденных абляциями компонентов.

## Этап 7. Один финальный test run

После выбора конфигураций на validation выполнить test для:

```text
A0
A
B
C
D_full
```

и, если нужно для статьи/диссертации, ключевых D-абляций.

---

# 19. Статистическая проверка результатов

Эксперименты оцениваются на **одних и тех же вопросах**, поэтому сравнение должно быть парным.

## 19.1. Доверительные интервалы

Если один документ содержит только один вопрос, использовать paired bootstrap по `(document_id, question_id)`.

Если на один документ приходится несколько вопросов/аспектов, предпочтительнее **clustered paired bootstrap по `document_id`**, пересэмплируя документы целиком вместе со всеми их вопросами. Это сохраняет внутридокументную зависимость и согласуется с document-level split.

Для разницы метрик:

```text
ΔF1 = F1_B - F1_A
ΔEM = EM_B - EM_A
```

считать 95% bootstrap CI.

## 19.2. Binary outcomes

Для:

- EM correct/incorrect;
- Hit@1 yes/no;
- hard-negative selected yes/no;

можно дополнительно использовать McNemar test.

## 19.3. Random seeds

Для моделей, которые дообучаются, выполнить минимум 3 независимых запуска, используя **один и тот же заранее фиксированный набор seeds для сравниваемых конфигураций**, например:

```text
42
43
44
```

и сообщать:

```text
mean ± std
```

Итоговое преимущество желательно считать убедительным, если оно наблюдается не в одном случайном запуске, а устойчиво по seeds.

---

# 20. Анализ ошибок

Для каждой системы автоматически выделить категории:

```text
1. positive отсутствует в initial top-K
2. positive есть, но reranker выбрал hard negative
3. positive выбран, QA извлек неправильный span
4. hard negative выбран, QA дал правдоподобный ложный ответ
5. несколько positives дали разные ответы
6. answerability отклонил все chunks
7. structure expansion добавил полезный context
8. structure expansion добавил шум
9. sentence filtering удалил необходимый context
10. answer aggregation исправила ошибку
11. answer aggregation создала ошибку
```

Для научного анализа эта таблица зачастую информативнее одной итоговой F1.

---

# 21. Специальный сценарий: чанки и их метки уже полностью определены

Этот сценарий является наиболее удобным для чистой проверки исследовательской гипотезы.

Предполагается, что уже существуют:

```text
document_id
question_id
question
gold_answer / gold_answers
chunk_id
chunk_text
chunk_label = positive | hard_negative | easy_negative | ambiguous
```

и, желательно:

```text
gold answer span
chunk position
section metadata
```

## 21.1. Что в этом случае НЕ нужно делать

Не нужно повторно:

- генерировать чанки;
- искать hard negatives;
- переопределять positive;
- обучать отдельную модель только для mining;
- менять существующую разметку ради удобства конкретного метода;
- использовать retrieval для искусственного построения candidate pool, если все чанки документа помещаются в допустимый вычислительный бюджет.

Вместо этого исходный набор чанков становится фиксированным объектом эксперимента.

## 21.2. Сначала сформировать два представления данных

### `C_all`

Все чанки текущего документа:

```text
C_all(q) = {c1, c2, ..., cn}
```

Использовать, если `n` достаточно мало для scoring всех chunks.

### `C_controlled`

Стратифицированный фиксированный candidate set:

```text
all positives
+ all hard negatives
+ N easy negatives
```

`N` определить один раз по validation/design и затем не менять между A–D.

Если easy negatives очень много, их подвыборка должна быть:

- детерминированной;
- сохраненной в файл;
- одинаковой для всех методов.

Например:

```text
candidate_sets.jsonl
```

## 21.3. Эксперимент A при готовой разметке

```text
C_all или C_controlled
        ↓
generic relevance Cross-Encoder
        ↓
rank all candidates
        ↓
top-1
        ↓
ordinary extractive QA
```

Метки не подаются в модель.

Основной вопрос:

> Насколько обычная relevance-модель ставит positive выше заранее известных hard negatives?

Главные метрики:

```text
Pairwise Pos>HN
MRR
Hit@1
HN@1
QA F1
```

## 21.4. Эксперимент B при готовой разметке

Train:

```text
positive → 1
hard_negative → 0
easy_negative → 0 (дополнительный класс отрицательных)
ambiguous → ignore / separate analysis
```

Test:

```text
тот же C_all / C_controlled
        ↓
answerability Cross-Encoder
        ↓
rank all candidates
```

Ключевое сравнение A vs B проводится **на идентичном candidate set**.

Это наиболее чистая проверка влияния hard-negative-aware supervision.

## 21.5. Эксперимент C при готовой разметке

Основной режим:

```text
C_all / C_controlled
        ↓
B scores
        ↓
top-M candidates
        ↓
globally normalized multi-passage QA
```

Дополнительно выполнить два диагностических режима:

### `C_random-M`

Случайные M из фиксированного candidate set.

Показывает, что дает само наличие нескольких passages без качественного ranking.

### `C_oracle-M`

M passages, где гарантированно присутствует positive.

Используется только как верхняя диагностическая граница reader-а.

Итоговое сравнение:

```text
B top-1
vs
C_random-M
vs
C_actual-M
vs
C_oracle-M
```

## 21.6. Эксперимент D при готовой разметке

Начать не с retrieval, а с фиксированного candidate universe:

```text
C_all / C_controlled
        ↓
generic + answerability scores
        ↓
structural relationships
        ↓
evidence expansion/filtering
        ↓
multi-passage QA
        ↓
answer aggregation
```

D1 (`BM25 + dense`) в таком сценарии можно:

- оставить как отдельную абляцию ranking signal;
- либо пропустить, если исследование уже не касается initial retrieval и все candidates обрабатываются напрямую.

Тогда более чистая лестница будет:

```text
D0 = C
D2 = C + structure
D3 = D2 + inter-chunk expansion
D4 = D3 + sentence evidence filtering
D5 = D4 + answer aggregation
```

Это допустимо и даже предпочтительно, если задача исследования сформулирована именно как **выбор правильного evidence среди заранее известных chunks**, а не как retrieval из большого корпуса.

## 21.7. Что нельзя делать даже при готовых метках

Нельзя на test:

- выбирать top-M по gold label;
- отбрасывать hard negatives, потому что их label известен;
- выбирать только section, в котором заранее находится positive;
- оптимизировать threshold по test;
- использовать gold answer при runtime scoring;
- удалять «неудобные» hard negatives после просмотра ошибок test.

Gold labels нужны только для:

```text
training (train split)
validation/model selection (validation split)
final evaluation/error analysis (test split)
```

## 21.8. Рекомендуемый основной протокол при готовых чанках

Наиболее чистый вариант:

```text
1. document-level split
2. freeze candidate sets
3. A: generic relevance
4. B: answerability trained on labeled HN
5. C: B top-M + global multi-passage reader
6. D: structure/evidence ablations on same candidates
7. final clustered bootstrap by document
```

И отдельно, если нужен end-to-end результат:

```text
8. вернуть initial retrieval и повторить A–D в полном режиме
```

Таким образом исследование будет иметь два уровня доказательства:

```text
controlled evidence-selection result
+
end-to-end system result
```

---

# 22. Рекомендуемая структура файлов проекта

```text
experiments/
│
├── data/
│   ├── dataset.jsonl
│   ├── splits.json
│   └── hard_negative_verification.jsonl
│
├── configs/
│   ├── a0_dense.yaml
│   ├── a_relevance_ce.yaml
│   ├── b_answerability_ce.yaml
│   ├── c_multipassage.yaml
│   ├── d1_hybrid.yaml
│   ├── d2_structure.yaml
│   ├── d3_graph.yaml
│   ├── d4_evidence.yaml
│   ├── d5_aggregation.yaml
│   └── d_full.yaml
│
├── runs/
│   └── <experiment>/<seed>/
│       ├── predictions.jsonl
│       ├── chunk_scores.jsonl
│       ├── metrics.json
│       ├── config_resolved.yaml
│       └── runtime.json
│
├── analysis/
│   ├── paired_comparison.csv
│   ├── error_analysis.csv
│   └── ablation_table.csv
│
└── references/
    └── sources_used.md
```

---

# 23. Фиксация источников

Для каждой использованной научной идеи в `references/sources_used.md` хранить:

```yaml
- source_id: S01
  title: ...
  authors: ...
  year: ...
  claim_used: ...
  verified_excerpt: ...
  url: ...
  used_in: [B, C]
```

В конфигурации каждого эксперимента дополнительно фиксировать:

```yaml
source_refs:
  - S01
  - S02
```

Это позволит впоследствии восстановить, на каком источнике основан каждый компонент экспериментальной схемы.

---

# 24. Использованные и проверенные источники

Ниже перечислены только те работы, из которых непосредственно заимствована или адаптирована идея постановки экспериментов. Формулировки проверены по полученным библиографическим записям/аннотациям.

## [1] RocketQA — hard-negative training и очистка сложных negatives

**Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Xin Zhao, Daxiang Dong, Hua Wu, Haifeng Wang.** 2020. *RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering*. pp. 5835–5847. На момент проверки: 767 цитирований.

Источник:  
https://consensus.app/papers/rocketqa-an-optimized-training-approach-to-dense-passage-qu-ding/6402deea6cdb54dfacbe2e4792ec7de4/?utm_source=chatgpt

Проверенный контекст:

> “cross-batch negatives, denoised hard negatives and data augmentation”

Использовано в данной инструкции для обоснования:

- обучения на hard negatives;
- необходимости проверять/очищать сложные отрицательные примеры;
- различения easy и hard negatives.

## [2] Multi-passage BERT — глобальная нормализация ответов между passages

**Zhiguo Wang, Patrick Ng, Xiaofei Ma, Ramesh Nallapati, Bing Xiang.** 2019. *Multi-passage BERT: A Globally Normalized BERT Model for Open-domain Question Answering*. pp. 5877–5881. На момент проверки: 266 цитирований.

Источник:  
https://consensus.app/papers/multipassage-bert-a-globally-normalized-bert-model-for-wang-ng/2ffeb044b2795a0b9c58f8b51613bd03/?utm_source=chatgpt

Проверенный контекст:

> “viewing passages ... as independent training instances ... may cause incomparable scores”

и:

> “globally normalize answer scores across all passages of the same question”

Использовано для эксперимента C. Существенное методическое следствие: независимые softmax confidence по чанкам нельзя выдавать за faithful multi-passage global normalization.

## [3] Evidence Aggregation — совместное использование evidence из нескольких passages

**Shuohang Wang, Mo Yu, Jing Jiang, Wei Zhang, Xiaoxiao Guo, Shiyu Chang, Zhiguo Wang, Tim Klinger, G. Tesauro, Murray Campbell.** 2017. *Evidence Aggregation for Answer Re-Ranking in Open-Domain Question Answering*. ArXiv abs/1711.05116. На момент проверки: 167 цитирований.

Источник:  
https://consensus.app/papers/evidence-aggregation-for-answer-reranking-in-opendomain-wang-yu/22a493c166f55a9db38843517b762c42/?utm_source=chatgpt

Проверенный контекст:

> “make use of the aggregated evidence from different passages to better determine the answer”

Использовано для D5 — answer aggregation.

## [4] Dense Hierarchical Retrieval — сочетание локального passage context и более общей структуры

**Kazuma Hashimoto, Yingbo Zhou, Semih Yavuz, Caiming Xiong, Philip S. Yu.** 2021. *Dense Hierarchical Retrieval for Open-Domain Question Answering*. pp. 188–200. На момент проверки: 49 цитирований.

Источник:  
https://consensus.app/papers/dense-hierarchical-retrieval-for-opendomain-question-hashimoto-zhou/adb0e59a319c561586507c9b279c24a6/?utm_source=chatgpt

Проверенный контекст:

> “utilizing both macroscopic semantics in the document and microscopic semantics specific to each passage”

В работе также исследуются `In-Doc` и `In-Sec negatives`.

Использовано для D2 и общей идеи структурного ранжирования внутри документа.

## [5] Mitigating False-Negative Contexts — множественные допустимые evidence contexts

**Ansong Ni, Matt Gardner, Pradeep Dasigi.** 2021. *Mitigating False-Negative Contexts in Multi-document Question Answering with Retrieval Marginalization*. ArXiv abs/2103.12235. На момент проверки: 14 цитирований.

Источник:  
https://consensus.app/papers/mitigating-falsenegative-contexts-in-multidocument-ni-gardner/4145ef51c72e5b0ab54113c0a4abeae1/?utm_source=chatgpt

Проверенный контекст:

> “the correct answer can often be obtained from multiple evidence candidates”

Использовано для правил:

- не считать только один чанк единственным positive, если существуют альтернативные evidence contexts;
- не обучаться на потенциальных false negatives;
- учитывать множество допустимых gold spans в C.

## [6] Long Context QA via Supervised Contrastive Learning — evidence discrimination

**Avi Caciularu, Ido Dagan, J. Goldberger, Arman Cohan.** 2021. *Long Context Question Answering via Supervised Contrastive Learning*. pp. 2872–2879. На момент проверки: 28 цитирований.

Источник:  
https://consensus.app/papers/long-context-question-answering-via-supervised-caciularu-dagan/fab86611916158d8886b86badb444102/?utm_source=chatgpt

Проверенный контекст:

> “explicitly discriminate supporting evidence sentences from negative ones”

Использовано как научное основание для дополнительной contrastive абляции sentence-level evidence selector в D4.

## [7] Answerability under high semantic overlap

**Rustam Abdumalikov, Pasquale Minervini, Yova Kementchedjhieva.** 2024. *Answerability in Retrieval-Augmented Open-Domain Question Answering*. ArXiv abs/2403.01461.

Источник:  
https://consensus.app/papers/answerability-in-retrievalaugmented-opendomain-abdumalikov-minervini/3523e4964ed955c391f712e5cb2a22ec/?utm_source=chatgpt

Проверенный контекст:

> “irrelevant text excerpts with high semantic overlap”

Использовано как основание для разделения обычной relevance и answerability в эксперименте B.

---

# 25. Что является адаптацией, а не прямым воспроизведением статьи

Необходимо отдельно зафиксировать, что описанные эксперименты **не являются точной репликацией** перечисленных работ.

В частности:

- работы DPR/RocketQA/Multi-passage BERT в основном рассматривают open-domain QA;
- здесь retrieval ограничивается чанками **одного заранее известного документа**;
- D объединяет несколько концептов в одну внутридокументную pipeline;
- graph expansion, конкретные structural features и схема D1–D5 являются предлагаемой экспериментальной адаптацией;
- exact-normalized answer aggregation — упрощенная воспроизводимая версия общей идеи evidence aggregation;
- конкретные значения K, M, thresholds и token budgets должны определяться только validation-экспериментами.

Эта оговорка обязательна, чтобы не приписывать источникам методы или результаты, которых в них непосредственно не заявлено.

---

# 26. Многократная проверка корректности постановки

Перед запуском план проверен по нескольким независимым критериям.

## Проверка 1. Причинная интерпретируемость сравнения

### A → B

Сохраняются:

- dense retriever;
- candidate pool;
- Cross-Encoder backbone;
- QA reader.

Меняется целевой сигнал:

```text
relevance/easy negatives
→
answerability/hard negatives
```

Следовательно, разницу можно интерпретировать как эффект hard-negative-aware answerability training.

### B → C

Сохраняются candidate generator и answerability ranking.

Меняется:

```text
top-1 independent QA
→
top-M globally normalized QA
```

Следовательно, проверяется именно гипотеза о пользе совместной multi-passage обработки.

### C → D

D разбит на D1–D5, поэтому новые механизмы добавляются по одному.

**Результат проверки:** причинная интерпретируемость сохранена.

## Проверка 2. Отсутствие утечки

Проверены основные потенциальные утечки:

- split выполняется по `document_id`;
- соседние chunks одного документа не расходятся между train/test;
- test не используется для выбора K/M/threshold;
- hard-negative verification не должна превращать test-разметку в обучающие данные;
- preprocessing фиксируется до финального test.

**Результат проверки:** при соблюдении инструкции прямой leakage отсутствует.

## Проверка 3. Соответствие метрик гипотезам

Только итогового F1 недостаточно.

Добавлены отдельные метрики для:

- попадания positive в candidates;
- ранга positive;
- присутствия hard negatives в top-K;
- способности B отличать positive от hard negative;
- качества extraction;
- покрытия;
- вычислительной стоимости.

**Результат проверки:** каждая основная гипотеза имеет наблюдаемую метрику.

## Проверка 4. Корректность multi-passage постановки

Отдельно исключена методическая ошибка:

```text
независимый softmax по каждому passage
→ затем сравнение confidence
```

не считается эквивалентом globally normalized multi-passage QA.

В C требуется глобальная конкуренция raw logits / span scores и соответствующее обучение.

**Результат проверки:** эксперимент C соответствует концептуальной постановке Multi-passage BERT.

## Проверка 5. Корректность hard-negative supervision

Учтено, что hard negatives могут содержать:

- неполную разметку;
- альтернативные correct contexts;
- partial evidence.

Поэтому введены:

```text
verified hard negatives
ambiguous
multiple positives
multiple gold spans
```

**Результат проверки:** риск false-negative training существенно снижен.

## Проверка 6. Корректность сложного эксперимента D

Первоначальный вариант D менял слишком много компонентов одновременно. В окончательной постановке введены D1–D5 и D_full собирается только после абляций.

**Результат проверки:** D можно использовать не только как инженерный прототип, но и как научно интерпретируемый эксперимент.

---

# 27. Минимальный итоговый набор таблиц

## Таблица 1. Retrieval / chunk selection

| Model | Hit@1 | Hit@5 | MRR | HN@1 ↓ | HN@5 ↓ | Pairwise Pos>HN |
|---|---:|---:|---:|---:|---:|---:|
| A0 | | | | | | |
| A | | | | | | |
| B | | | | | | |
| C candidate stage | | | | | | |
| D_full | | | | | | |

## Таблица 2. QA

| Model | EM | Token F1 | Char F1 | ROUGE-L | BERTScore F1 | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| A0 | | | | | | |
| A | | | | | | |
| B | | | | | | |
| C | | | | | | |
| D_full | | | | | | |

## Таблица 3. D ablations

| Variant | Hybrid | Structure | Graph | Sentence evidence | Aggregation | QA F1 | HN@1 | Latency |
|---|---|---|---|---|---|---:|---:|---:|
| D0 | – | – | – | – | – | | | |
| D1 | + | – | – | – | – | | | |
| D2 | + | + | – | – | – | | | |
| D3 | + | + | + | – | – | | | |
| D4 | + | + | + | + | – | | | |
| D5 | + | + | + | + | + | | | |

## Таблица 4. Efficiency

| Model | chunks processed | reader tokens | latency mean | latency p95 | peak GPU memory |
|---|---:|---:|---:|---:|---:|
| A | | | | | |
| B | | | | | |
| C | | | | | |
| D_full | | | | | |

---

# 27.1. Исправления после повторной методической проверки

В текущую редакцию внесены следующие принципиальные исправления:

1. **A больше не обучается только на easy negatives.** Такой baseline был бы искусственно ослаблен. Основной A теперь определяется как сильный generic relevance reranker без специального использования вашей hard-negative answerability-разметки.
2. Введены **два режима эксперимента**: полный pipeline и controlled candidate-set mode.
3. Для уже готовых labels явно указано, что повторный hard-negative mining не нужен.
4. Уточнено, что gold labels нельзя использовать при test inference даже в controlled mode.
5. Для Multi-passage C отдельно отмечено, что независимые passage-level softmax confidence нельзя считать global normalization.
6. Формула marginalization в C обозначена как допустимая адаптация, а не как единственная буквальная реализация исходной статьи.
7. Добавлено корректное обращение с truly unanswerable questions через null candidate.
8. Исправлена `HN@K` для документов, где число кандидатов меньше K.
9. Для sentence-level evidence добавлено обязательное отображение assembled offsets обратно в исходный документ.
10. Для answer aggregation добавлена de-duplication перекрывающихся чанков, чтобы sliding-window duplicates не создавали ложное «множественное подтверждение».
11. Статистический bootstrap уточнен до cluster bootstrap по документам, если один документ содержит несколько вопросов.
12. Для pre-labeled scenario добавлен отдельный полноценный протокол A–D.

После этих исправлений план лучше разделяет:

```text
initial retrieval quality
answerability discrimination
multi-passage reading
structure/evidence contribution
```

и позволяет интерпретировать вклад каждого компонента отдельно.

---

# 28. Основной критерий успешности исследования

Наиболее содержательный результат — не просто ситуация, когда `D_full > A` по F1, а последовательное подтверждение цепочки:

```text
A → показывает проблему hard negatives
B → лучше различает positive и hard negative
C → исправляет часть ошибок, когда positive не top-1, но находится среди top-M
D → дополнительно выигрывает за счет структуры/evidence selection
```

Если такая последовательность наблюдается по retrieval-, answerability- и QA-метрикам и сохраняется при парной статистической проверке, экспериментальная постановка дает сильное основание утверждать, что **для внутридокументного экстрактивного QA выбор evidence следует рассматривать шире, чем простое ранжирование одного наиболее семантически близкого чанка**.
