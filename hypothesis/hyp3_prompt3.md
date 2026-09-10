# Промпт: эксперименты по интеграции KeyBERT с extractive QA на RoBERTa

Ты — NLP-разработчик и исследователь. Необходимо провести серию экспериментов по улучшению существующего extractive QA-пайплайна на базе RoBERTa.

## 1. Исходная концепция

Базовая система решает задачу извлечения краткого фрагмента ответа из научно-технического текста по фиксированному тематическому вопросу.

Основной принцип системы уже определён и **не должен изменяться**:

- основной способ извлечения ответа — **extractive QA на RoBERTa**;
- важным task-specific сигналом являются **ключевые слова, полученные из attention QA-модели**;
- attention-keywords отражают не просто общую тематику документа, а признаки, на которые модель опирается именно при ответе на QA-вопрос;
- существует дополнительный сигнал `diff`, уже используемый в текущем алгоритме;
- KeyBERT должен использоваться только как **дополнительный тематический / семантический сигнал**;
- **KeyBERT не должен заменять attention-keywords**;
- желательно минимально увеличивать вычислительную стоимость online-инференса;
- эксперименты должны быть реализованы модульно, чтобы каждую модификацию можно было независимо включать и выключать.

Главная исследовательская гипотеза:

> KeyBERT может улучшить выбор релевантных фрагментов и фильтрацию признаков, если использовать его как общий тематический prior, а attention RoBERTa — как task-specific сигнал, непосредственно связанный с QA-задачей.

## 2. Общая схема экспериментов

Для всех экспериментов использовать одну и ту же тестовую выборку и одну и ту же базовую QA-модель.

Сравнить минимум следующие варианты:

1. **Baseline QA** — обычный extractive QA без дополнительных модификаций.
2. **QA + attention** — существующий алгоритм с attention-keywords.
3. **QA + attention + diff** — текущая наиболее полная версия.
4. **QA + KeyBERT only** — контрольный эксперимент без attention.
5. **QA + attention + KeyBERT**.
6. **QA + attention + diff + KeyBERT**.

KeyBERT-only нужен только как контроль, а не как целевая архитектура.

## 3. Эксперимент 1. Тематические кластеры KeyBERT + attention-keywords

### Идея

KeyBERT определяет общие темы документа, а attention-keywords показывают, какие из этих тем важны именно для QA-вопроса.

### Что реализовать

1. Получить Top-K KeyBERT keyphrases для документа.
2. Получить embedding каждой keyphrase.
3. Сгруппировать близкие keyphrases в тематические кластеры.
4. Получить существующие attention-keywords.
5. Для каждого attention-keyword найти ближайший тематический кластер.
6. Рассчитать вес кластера по:
   - количеству attention-keywords;
   - суммарному attention-score;
   - semantic similarity между attention-keywords и KeyBERT phrases.
7. Использовать наиболее релевантные кластеры для выбора чанков, изменения chunk score или постобработки ответа.

### Что сравнить

- без кластеров;
- только KeyBERT;
- только attention;
- KeyBERT clusters + attention.

Проверить разные значения:
- `top_k_keybert`;
- числа кластеров;
- semantic similarity threshold.

## 4. Эксперимент 2. Совместный chunk score

### Идея

Добавить к текущему chunk score тематический prior от KeyBERT.

Примеры формул:

```text
chunk_score =
    α * attention_score
  + β * keybert_topic_score
```

или:

```text
chunk_score =
    α * qa_score
  + β * attention_score
  + γ * keybert_topic_score
```

### Варианты KeyBERT topic score

1. Максимальная cosine similarity между chunk embedding и KeyBERT phrase.
2. Среднее Top-N similarity.
3. Взвешенная сумма similarity с весами KeyBERT relevance score.

### Эксперимент

Подобрать `α`, `β`, `γ` через grid search на validation set.

Главный вопрос: может ли небольшой вес KeyBERT улучшить выбор чанка без деградации attention-based подхода.

## 5. Эксперимент 3. Attention + diff + KeyBERT

Добавить KeyBERT как третий независимый сигнал:

```text
score =
    w_att * attention_score
  + w_diff * diff_score
  + w_key * keybert_score
```

Перед объединением нормализовать все score.

Провести ablation:

- attention;
- diff;
- KeyBERT;
- attention + diff;
- attention + KeyBERT;
- diff + KeyBERT;
- attention + diff + KeyBERT.

## 6. Эксперимент 4. KeyBERT-фильтрация шумных attention-keywords

### Идея

Использовать KeyBERT как мягкий тематический фильтр для attention-keywords.

Важно: не удалять attention-keyword только потому, что он отсутствует в списке KeyBERT.

Для каждого attention-keyword:

1. получить embedding;
2. посчитать максимальную similarity с KeyBERT phrases;
3. переоценить attention score.

Пример:

```text
topic_similarity =
max cosine(attention_keyword, keybert_phrase)
```

Мягкое взвешивание:

```text
new_attention_score =
attention_score * (0.5 + 0.5 * topic_similarity)
```

Также проверить threshold-вариант, где score уменьшается, но не обнуляется.

Сравнить:
- исходный attention;
- жёсткую фильтрацию;
- мягкую фильтрацию;
- semantic reweighting.

## 7. Эксперимент 5. Global vs topic-specific attention-keywords

Разделить attention-keywords на:

- **global** — часто появляются для одного и того же QA-вопроса во многих документах;
- **local/topic-specific** — важны для конкретного документа.

Для каждого attention-keyword рассчитать:

```text
global_frequency
```

и:

```text
local_topic_score = similarity_to_keybert
```

Пример итогового веса:

```text
keyword_score =
    attention_score
  * (1 + α * global_score)
  * (1 + β * local_topic_score)
```

Проверить гипотезу, что лучшие признаки одновременно связаны:
- с QA-вопросом;
- с локальной тематикой документа.

## 8. Эксперимент 6. KeyBERT-guided hard negatives

### Идея

Использовать KeyBERT только при подготовке обучающих данных, чтобы не увеличивать online inference cost.

Для каждого документа:

1. найти через KeyBERT тематически близкие фрагменты;
2. выбрать среди них чанки, которые похожи на правильный ответ по теме, но не содержат gold answer span;
3. использовать их как hard negatives.

Сравнить:

### Baseline
Positive + random negatives.

### Experiment
Positive + random negatives + KeyBERT hard negatives.

Проверить долю hard negatives:
- 10%;
- 25%;
- 50%;
- 75%.

Оценивать:
- качество извлечения;
- false positive rate;
- устойчивость на тематически похожих, но неправильных фрагментах.

## 9. Эксперимент 7. Auxiliary alignment KeyBERT phrases ↔ QA evidence

Основная задача остаётся span prediction.

Добавить лёгкий auxiliary loss:

```text
L_total = L_QA + λ * L_alignment
```

где `L_alignment` усиливает близость представления gold answer к релевантным KeyBERT phrases.

Пример:

```text
L_alignment =
1 - cosine(answer_embedding, relevant_keybert_phrase_embedding)
```

Считать KeyBERT phrase релевантной только при достаточной semantic similarity к gold answer.

Проверить:

```text
λ = 0
0.01
0.05
0.1
0.2
```

Основное ограничение: auxiliary loss не должен ухудшать span extraction.

## 10. Эксперимент 8. Answer consensus

Финальный answer span оценивать по трём типам evidence:

1. QA confidence;
2. attention evidence;
3. KeyBERT topic evidence.

Пример:

```text
answer_score =
    α * qa_confidence
  + β * attention_evidence
  + γ * keybert_topic_evidence
```

Использовать score:
- для выбора между несколькими candidate spans;
- для выбора ответа между чанками;
- при необходимости для no-answer decision.

## 11. Document-level vs chunk-level KeyBERT

Проверить два режима.

### Document-level
KeyBERT запускается один раз на документ.

Плюсы:
- минимальная стоимость;
- единая тема документа.

### Chunk-level
KeyBERT запускается для каждого чанка.

Плюсы:
- более точная локальная тематика.

Сравнить по:
- качеству;
- времени inference;
- GPU memory;
- RAM.

Если прирост chunk-level мал, предпочтение отдать document-level.

## 12. Количество KeyBERT keyphrases

Проверить:

```text
Top-5
Top-10
Top-15
Top-20
Top-30
```

Для каждого варианта измерить:
- качество;
- время;
- стабильность результатов.

Найти минимальное K, после которого прирост качества становится незначительным.

## 13. Метрики качества

Использовать основные метрики текущей системы:

- Character-level F1;
- Token-level F1;
- ROUGE-L F1;
- BERTScore Precision;
- BERTScore Recall;
- BERTScore F1.

Также при наличии соответствующей логики считать:
- Coverage / Answer rate;
- Exact Match;
- False Positive Rate.

## 14. Computational cost

Для каждого варианта измерить:

```text
mean inference time per document
mean inference time per chunk
peak GPU memory
peak RAM
```

Также рассчитать:

```text
Δtime %
ΔGPU memory %
ΔRAM %
```

относительно baseline.

Главный критерий: прирост качества должен оправдывать дополнительную стоимость.

## 15. Основная таблица результатов

Сформировать таблицу:

| Method | Char F1 | Token F1 | ROUGE-L | BERTScore F1 | Coverage | Time | GPU Memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| QA baseline | | | | | | | |
| QA + attention | | | | | | | |
| QA + attention + diff | | | | | | | |
| QA + KeyBERT | | | | | | | |
| QA + attention + KeyBERT | | | | | | | |
| QA + attention + diff + KeyBERT | | | | | | | |

Для сложных модификаций добавить отдельные строки.

## 16. Ablation study

Для лучшего варианта последовательно убрать каждый компонент.

Если лучшая система:

```text
QA + attention + diff + KeyBERT
```

то проверить:

```text
-attention
-diff
-KeyBERT
-attention -diff
-attention -KeyBERT
-diff -KeyBERT
```

Цель — количественно определить вклад каждого компонента.

## 17. Статистическая проверка

Если размер тестовой выборки позволяет, выполнить bootstrap, например:

```text
1000 bootstrap samples
```

Для основных метрик получить:
- mean;
- 95% confidence interval.

Особенно сравнить:

```text
QA + attention + diff
```

и:

```text
QA + attention + diff + KeyBERT
```

## 18. Приоритет экспериментов

### Приоритет 1 — дешёвые, без переобучения

1. QA + KeyBERT baseline.
2. Attention + KeyBERT filtering.
3. Joint chunk score.
4. Attention + diff + KeyBERT.
5. Answer consensus.
6. Document-level vs chunk-level KeyBERT.

### Приоритет 2 — анализ признаков

7. Global vs local attention-keywords.
8. KeyBERT thematic clusters + attention.

### Приоритет 3 — обучение

9. KeyBERT-guided hard negatives.
10. Auxiliary alignment loss.

Сначала выяснить, даёт ли KeyBERT прирост без переобучения.

## 19. Требования к коду

Архитектура должна позволять независимо включать компоненты:

```python
use_attention = True
use_diff = True
use_keybert = True
use_keybert_filter = False
use_keybert_chunk_score = False
use_answer_consensus = False
```

Все веса и thresholds вынести в конфигурацию, например:

```python
KEYBERT_TOP_K = 10

ATTENTION_WEIGHT = 1.0
DIFF_WEIGHT = 1.0
KEYBERT_WEIGHT = 0.2

KEYBERT_SIM_THRESHOLD = 0.5
```

Не хардкодить экспериментальные параметры внутри функций.

## 20. Что должно быть результатом каждого эксперимента

Для каждого эксперимента предоставить:

1. описание алгоритма;
2. изменения в коде;
3. конфигурацию параметров;
4. таблицу метрик;
5. время выполнения;
6. потребление памяти;
7. сравнение с baseline;
8. краткий вывод:
   - стало лучше или хуже;
   - на сколько;
   - какой компонент дал прирост;
   - оправдывает ли прирост вычислительную стоимость.

## 21. Финальный вывод

В конце ответить на вопросы:

1. Даёт ли KeyBERT пользу поверх attention-keywords?
2. Какая комбинация даёт лучший результат?
3. Какие компоненты оказываются избыточными?
4. Какой вариант является лучшим компромиссом качества и вычислительной стоимости?
5. Следует ли использовать KeyBERT:
   - только offline;
   - только при обучении;
   - при inference;
   - в обеих фазах?

## 22. Ключевое ограничение исследования

Attention-keywords являются **центральным task-specific механизмом предлагаемого подхода**.

KeyBERT не должен рассматриваться как его замена.

Главный исследовательский вопрос:

> Может ли общий тематический сигнал KeyBERT дополнить task-specific сигнал attention и улучшить extractive QA без существенного увеличения вычислительной сложности?
