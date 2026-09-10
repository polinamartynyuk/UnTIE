# Промпт для код-агента: эксперимент по тематической кластеризации научных статей с KeyBERT и downstream-проверкой в extractive QA

Тебе необходимо провести **отдельный исследовательский эксперимент по тематической кластеризации научных статей с использованием KeyBERT**, а затем проверить, насколько полученные тематические кластеры полезны для основной задачи проекта — выбора корректного фрагмента и ответа в существующем extractive QA pipeline.

Эксперимент не должен сразу изменять production/inference pipeline проекта. Вся новая логика должна находиться в изолированном экспериментальном контуре и максимально переиспользовать существующие компоненты проекта без изменения их поведения.

---

# 1. Главные исследовательские вопросы

Эксперимент должен проверить две группы гипотез.

## 1.1. Гипотезы о тематической кластеризации

**H1. KeyBERT representation**

> Представление документа через наиболее значимые KeyBERT-keyphrases позволяет получать более содержательные и/или устойчивые тематические кластеры, чем embedding всего `title + abstract`.

**H2. Hybrid representation**

> Совместное использование embedding документа и weighted embedding его KeyBERT-keyphrases даёт более качественное тематическое представление, чем каждый из этих сигналов по отдельности.

**H3. Cluster description**

> KeyBERT полезен для автоматического получения `representative_terms` уже построенных тематических кластеров, даже если он не улучшает геометрические метрики кластеризации.

## 1.2. Downstream QA-гипотезы

После выбора нескольких лучших вариантов кластеризации необходимо проверить:

**H4. Cluster-specific attention**

> Ключевые слова, извлечённые из attention и обученные отдельно внутри тематически однородных кластеров, лучше помогают локализовать релевантные QA-чанки, чем отсутствие keyword reranking.

Проверка:

\[
D1 > D0.
\]

**H5. Cluster-specific attention + diff**

> Использование существующего `score_difference` (`diff`) совместно с attention внутри тематических кластеров улучшает настройку ключевых слов.

Проверка:

\[
D2 > D1.
\]

**H6. Topic specialization vs global tuning**

> Тематически специализированные attention+diff словари эффективнее одного глобального словаря текущего `static-keywords`.

Проверка:

\[
D2 > Q1.
\]

**H7. KeyBERT clustering downstream**

> Кластеры, построенные с участием KeyBERT, дают больший downstream QA gain при обучении cluster-specific keywords, чем кластеры, построенные только по обычному document embedding.

---

# 2. Критические ограничения

1. Эксперимент должен быть **изолирован от production**.
2. Не изменяй поведение существующих `baseline`, `attention`, `static-keywords`, QA, reranking и tuning.
3. Не заменяй extractive QA генеративной моделью.
4. Не добавляй LLM в topic clustering, cluster labeling или downstream answer selection.
5. Не используй gold QA answers для построения тематических кластеров.
6. Gold/reference допустим только там, где он уже законно используется существующим train/dev keyword-tuning.
7. Не используй dev/test для формирования train topic vocabulary или centroids.
8. Не считай KeyBERT score эквивалентом `score_difference`.
9. Не используй правило `diff < 0 => topic`.
10. Все сравниваемые downstream-варианты должны использовать один и тот же существующий CE/QA-компонент и одну и ту же зафиксированную кластеризацию внутри конкретного ablation.
11. Не проводи огромный полный grid search без необходимости.
12. Кэшируй embeddings и KeyBERT results, не выполняй повторный Transformer inference без причины.

---

# 3. Сначала изучи проект

До написания экспериментального кода найди и опиши:

- где находятся научные документы;
- формат документов;
- наличие `title`, `abstract`, `section headings`, полного текста;
- языки корпуса;
- train/dev/test split;
- document IDs;
- используемый sentence encoder;
- наличие `sentence-transformers`;
- наличие KeyBERT;
- готовые document embeddings;
- `WeightedKeyword` или фактический аналог;
- где вычисляются `attention_weight` и `score_difference`;
- где выполняется attention keyword mining;
- где формируется candidate pool;
- где реализован `static-keywords`;
- где вычисляется chunk score;
- `minimum_matches`;
- positional/frequency/uniqueness modifiers;
- fallback;
- `AnswerConsensus`;
- SFFS;
- tuning objective;
- существующие метрики;
- existing test infrastructure;
- существующий CE/QA-компонент.

Ничего не предполагай заранее.

Сначала составь краткую карту найденных компонентов и укажи, какие из них можно безопасно переиспользовать в эксперименте.

---

# 4. Уточнение по CE

В downstream-схемах используется обозначение:

```text
CE
```

Под CE необходимо понимать **существующий в проекте компонент, который формирует candidate answers / участвует в текущей цепочке извлечения**.

Перед реализацией:

- найди его фактический класс/функцию;
- опиши вход;
- опиши выход;
- укажи, какие confidence/scores он возвращает;
- укажи, где доступен attention;
- укажи связь с текущим extractive QA pipeline.

Не создавай новый CE и не подменяй существующий компонент собственной интерпретацией.

Если в коде термин `CE` не используется, но соответствующий компонент существует под другим названием, используй фактический компонент и явно зафиксируй соответствие.

---

# 5. Структура экспериментального модуля

Если в проекте уже есть структура для экспериментов, используй её.

Иначе создай изолированную папку, например:

```text
experiments/
    keybert_topic_clustering/
```

или эквивалент в стиле проекта.

Production-код допускается только импортировать и переиспользовать.

Не меняй его поведение без отдельной необходимости и явного обоснования.

---

# 6. Anti-leakage

Если split существует:

## Train

Только train используется для:

- построения topic representations;
- кластеризации;
- вычисления centroids;
- извлечения representative terms;
- обучения cluster-specific attention keywords;
- агрегации cluster-specific `diff`.

## Dev

Dev используется для:

- выбора representation;
- выбора K;
- выбора clustering algorithm;
- `top_n`;
- n-gram range;
- MMR;
- `alpha`;
- routing thresholds;
- hard/soft routing;
- числа cluster-specific keywords;
- `weight_ratio` и других существующих параметров;
- выбора answer-selection strategy.

## Test

Test используется только для финальной оценки выбранных конфигураций.

Запрещено:

- изменять centroids по test;
- добавлять test terms в topic dictionaries;
- обучать attention/diff словари по test;
- выбирать thresholds по test;
- использовать test gold для routing.

---

# 7. Базовое представление научного документа

Основная конфигурация:

```text
title + abstract
```

Если данные позволяют, провести ablation:

```text
R1 = abstract
R2 = title + abstract
R3 = title + abstract + headings
```

Полный текст не использовать как основное topic representation без отдельного обоснования.

---

# 8. Общий embedding backend

Желательно использовать один embedding backend для:

- document embedding;
- KeyBERT;
- keyphrase embeddings.

Предпочтительный дешёвый baseline:

```text
sentence-transformers/all-MiniLM-L6-v2
```

если он подходит языку корпуса.

Если проект уже использует совместимый encoder — сначала проверить возможность его переиспользования.

Сохранять:

```text
model_name
model_revision
embedding_dimension
normalization
```

---

# 9. KeyBERT extraction

Для каждого train-документа извлечь keyphrases.

Проверить:

```text
keyphrase_ngram_range = (1, 1)
keyphrase_ngram_range = (1, 2)
keyphrase_ngram_range = (1, 3)
```

Приоритетный вариант:

```text
(1, 3)
```

Проверить:

```text
top_n = 5
top_n = 10
top_n = 20
```

Опционально:

```text
top_n = 30
```

если вычислительная стоимость приемлема.

---

# 10. Diversity / MMR

Сравнить:

```text
use_mmr = False
```

и:

```text
use_mmr = True
```

Для MMR проверить, например:

```text
diversity = 0.3
diversity = 0.5
diversity = 0.7
```

MaxSum не использовать в основном эксперименте; допускается только маленький дополнительный ablation.

---

# 11. Представления документа для clustering

Сравнить минимум четыре варианта.

## E0 — lexical baseline

```text
TF-IDF
  ↓
TruncatedSVD
  ↓
clustering
```

## E1 — document embedding

```text
title + abstract
  ↓
sentence encoder
  ↓
L2 normalize
  ↓
e_D
```

## E2 — KeyBERT-only representation

KeyBERT возвращает:

\[
(k_i,w_i),\quad i=1,\dots,K.
\]

Для каждой phrase получить:

\[
e_{k_i}.
\]

Построить:

\[
e_K =
\frac{\sum_i w_i e_{k_i}}
{\sum_i w_i}.
\]

Затем L2 normalize.

## E3 — hybrid representation

\[
e_H =
normalize(
\alpha e_D + (1-\alpha)e_K
).
\]

Проверить:

```text
alpha = 0.25
alpha = 0.50
alpha = 0.75
```

`alpha=0` и `alpha=1` можно использовать как sanity check.

---

# 12. Кэширование

Если KeyBERT и document embedding используют один encoder:

- переиспользуй document embeddings;
- кэшируй candidate phrase embeddings;
- кэшируй KeyBERT output;
- используй `extract_embeddings` или эквивалентный API, если это соответствует версии KeyBERT.

Отдельно измеряй:

```text
document_embedding_time
keybert_candidate_time
keybert_ranking_time
clustering_time
```

---

# 13. Clustering algorithms

Основной кандидат:

```text
BisectingKMeans
```

Сравнить с:

```text
KMeans
```

и, если корпус достаточно большой:

```text
MiniBatchKMeans
```

HDBSCAN/BERTopic не использовать как основной pipeline.

UMAP не использовать перед основной кластеризацией без отдельного ablation.

---

# 14. Количество кластеров

Провести coarse search:

```text
K ∈ {4, 8, 16}
```

При необходимости адаптировать к размеру корпуса.

Не выбирать K только по одной внутренней метрике.

Сохранять:

```text
cluster_size_min
cluster_size_max
cluster_size_mean
cluster_size_median
```

---

# 15. Метрики качества clustering

Для каждой конфигурации вычислить:

- Silhouette Score;
- Davies–Bouldin Index;
- Calinski–Harabasz Index;
- cluster size distribution.

Если есть внешние тематические labels, которых эксперимент не использует при обучении, можно дополнительно применять их **только для post-hoc evaluation**.

---

# 16. Stability

Для stochastic algorithms выполнить несколько запусков:

```text
seed = 1
seed = 2
seed = 3
seed = 4
seed = 5
```

Оценить:

- Adjusted Rand Index;
- Normalized Mutual Information;

между запусками.

---

# 17. Representative terms тематических кластеров

После кластеризации для каждого cluster автоматически получить topic description.

Предпочтительная схема:

1. найти документы, ближайшие к centroid;
2. взять их `title + abstract`;
3. использовать KeyBERT;
4. агрегировать phrases;
5. получить top representative terms.

Пример:

```text
cluster_07:
    frost heave
    frozen soil
    freezing front
    ice lens
    permafrost
```

LLM для labels не использовать.

Машинное представление кластера:

```text
centroid + representative_terms
```

---

# 18. Дополнительный эксперимент: clustering самих keyphrases

Отдельно:

```text
all KeyBERT phrases
  ↓
deduplication
  ↓
phrase embeddings
  ↓
clustering
```

Цель — проверить, формируются ли семантические группы научных понятий.

Не смешивать этот эксперимент с основным document clustering.

---

# 19. Согласованность document clusters и keyphrase clusters

Для каждого document cluster определить:

- наиболее представленные keyphrase clusters;
- специфичность keyphrase clusters;
- общие keyphrase groups между соседними document clusters.

Использовать это только как анализ и потенциальную основу будущей topic hierarchy.

---

# 20. Вычислительная эффективность topic clustering

Для E0/E1/E2/E3 измерить:

- preparation time;
- embedding time;
- KeyBERT time;
- clustering time;
- peak RAM;
- GPU memory;
- artifact size;
- среднее время построения topic representation одного нового документа.

---

# 21. Routing simulation

Для dev/test документов, не участвовавших в train clustering:

1. построить выбранное topic representation;
2. сравнить с train cluster centroids;
3. определить top-1 и top-3;
4. сохранить cosine similarities;
5. измерить время.

Сохранять:

```text
document_id
cluster_id
similarity
normalized_weight
```

---

# 22. Hard и soft routing

Проверить:

## Hard routing

```text
document -> top-1 cluster
```

## Soft routing

```text
document -> top-N clusters + normalized weights
```

Routing параметры подбирать только на dev.

---

# 23. Staged search по clustering

Не делать полный Cartesian grid.

## Stage A — representation

При фиксированном разумном K сравнить:

```text
E0 TF-IDF
E1 document embedding
E2 KeyBERT
E3 hybrid
```

## Stage B — KeyBERT parameters

Для лучших KeyBERT representations сравнить:

```text
top_n
ngram_range
MMR
```

## Stage C — K

Для лучших representations сравнить несколько K.

## Stage D — clustering algorithm

Только на лучших representations сравнить:

```text
KMeans
BisectingKMeans
MiniBatchKMeans
```

## Stage E — выбрать ограниченное число лучших clustering configurations

Например:

```text
C_plain = лучший вариант без KeyBERT
C_keybert = лучший вариант с KeyBERT/hybrid
```

Именно их передавать в downstream QA experiment.

---

# 24. Downstream QA experiment

Хороший Silhouette не доказывает полезность topic clustering для основной задачи.

После выбора `C_plain` и `C_keybert` выполнить downstream experiment.

Production pipeline при этом не изменяется: используются экспериментальные wrappers/configs.

---

# 25. Downstream baselines и варианты

Сравнить:

```text
Q0 — текущий QA baseline без тематических кластеров и cluster-specific keywords

Q1 — текущий global static-keywords, если доступен

D0 — clustering -> CE -> frequency/consensus answer

D1 — clustering -> CE
     -> cluster-specific attention keywords
     -> keyword chunk score
     -> score-aware answer selection

D2 — clustering -> CE
     -> cluster-specific attention keywords
     -> cluster-specific attention + diff weighting/filtering
     -> keyword chunk score
     -> score-aware answer selection
```

---

# 26. D0 — кластеризация → CE → наиболее частый ответ

Для каждого документа:

```text
document
  ↓
topic routing
  ↓
chunking
  ↓
CE / existing QA candidate generation
  ↓
candidate answers
  ↓
existing semantic/extractive answer clustering
  ↓
most frequent answer cluster
```

Если в проекте есть `AnswerConsensus`, переиспользовать его.

Для answer cluster \(C_j\):

\[
Score_{D0}(C_j)=|C_j|.
\]

D0 не использует обученные cluster keywords.

---

# 27. D1 — cluster-specific attention keywords

Для каждого train topic cluster \(C_m\) использовать существующий attention-mining pipeline и собрать ключи только по train-документам этого кластера:

\[
K_m^{att}.
\]

Для каждого ключа агрегировать:

\[
w_{k,m}^{att}
=
Agg(w_{k,d}^{att}:d\in C_m).
\]

Предпочтительно переиспользовать существующую aggregation policy проекта.

Получаются отдельные словари:

\[
K_1^{att},K_2^{att},...,K_M^{att}.
\]

Один и тот же keyword может иметь разные веса в разных clusters.

---

# 28. D1 не использует diff

Это обязательный ablation.

D1 должен использовать attention-based cluster-specific keywords, но **не использовать `score_difference`**.

Это нужно, чтобы отдельно измерить вклад `diff` в D2.

---

# 29. D1 chunk scoring

Для чанка \(c_i\) и cluster \(m\):

\[
S_{att}(c_i|m)
=
\sum_{k\in K_m(c_i)}w_{k,m}^{att}.
\]

Если текущий проект уже имеет frequency/position/uniqueness modifiers и `minimum_matches`, максимально переиспользовать существующую формулу, заменив только источник/веса keywords.

Не менять одновременно весь chunk scoring.

---

# 30. D1 hard vs soft topic dictionaries

При hard routing:

```text
использовать словарь top-1 cluster
```

При soft routing:

\[
S_{att}(c_i|D)
=
\sum_m \pi_m S_{att}(c_i|m).
\]

Проверить оба варианта, если бюджет позволяет.

---

# 31. D1 answer selection

В D1 итоговый answer должен учитывать:

- frequency/consensus;
- score чанков.

Если текущий `AnswerConsensus` уже имеет `weighted_score`, `highest_avg_score`, `combined_score` или эквивалент — переиспользовать его.

Не вводить новую формулу без необходимости.

---

# 32. D2 — cluster-specific attention + diff

D2 использует тот же clustering и тот же CE, что D1.

Для каждого cluster-specific keyword агрегировать:

\[
w_{k,m}^{att}
\]

и:

\[
w_{k,m}^{diff}.
\]

Перед реализацией найти точную формулу `score_difference` в проекте.

Зафиксировать:

- какие embeddings используются;
- positive/negative reference;
- обработку нескольких gold/reference;
- полезный знак;
- существующую агрегацию.

---

# 33. Что означает «diff с учётом тематики»

Не вводить новую семантику `diff`.

Сначала для каждого train-документа вычислять existing document-level diff существующим способом.

Затем агрегировать его **только по train-документам одного thematic cluster**:

\[
w_{k,m}^{diff}
=
Agg(diff_{k,d}:d\in C_m).
\]

Тематичность возникает из cluster-specific aggregation.

Запрещено:

```text
negative diff = topic keyword
```

---

# 34. D2 combined keyword weight

Если текущий проект уже использует:

\[
\omega_k
=
\lambda w_k^{att}
+
(1-\lambda)w_k^{diff},
\]

переиспользовать эту формулу.

В D2:

\[
\omega_{k,m}
=
\lambda w_{k,m}^{att}
+
(1-\lambda)w_{k,m}^{diff}.
\]

`lambda` выбирать только на dev.

Если текущая формула отличается — использовать фактическую формулу проекта.

---

# 35. D2 chunk scoring

Для routed cluster использовать существующий keyword chunk scoring с cluster-specific combined weights.

При soft routing:

\[
S_{att+diff}(c_i|D)
=
\sum_m
\pi_m
S_{att+diff}(c_i|m).
\]

Keyword layer только оценивает candidate chunks/answers и не переписывает extractive span.

---

# 36. D0/D1/D2 должны использовать одну кластеризацию

Внутри одного ablation запрещено менять `K` или cluster assignments.

Порядок:

```text
выбрать clustering configuration
  ↓
заморозить train clusters + centroids
  ↓
D0
D1
D2
```

---

# 37. Сравнение двух лучших типов clustering

После выбора лучших вариантов выполнить:

```text
C_plain × {D0, D1, D2}
C_keybert × {D0, D1, D2}
```

Где:

- `C_plain` — лучший вариант без KeyBERT;
- `C_keybert` — лучший KeyBERT-only или hybrid вариант.

---

# 38. Ключевые downstream сравнения

Обязательно посчитать:

\[
D1-D0
\]

— вклад cluster attention;

\[
D2-D1
\]

— вклад `diff`;

\[
D2-Q1
\]

— вклад тематической специализации против global tuning;

\[
D2(C_{keybert})-D2(C_{plain})
\]

— вклад KeyBERT clustering в конечную QA-задачу.

---

# 39. Метрики downstream QA

Для Q0/Q1/D0/D1/D2 использовать одинаковые метрики:

- char F1;
- token F1;
- ROUGE-L F1;
- BERTScore, если уже используется;
- existing composite quality;
- mean delta относительно Q0;
- median delta;
- win rate;
- harm rate;
- bootstrap confidence interval;
- fallback rate.

---

# 40. Retrieval metrics

Для D1/D2 измерить:

- Evidence Recall@1;
- Evidence Recall@3;
- Evidence Recall@5;
- MRR evidence;
- rank gold-containing chunk;
- долю документов, где evidence chunk поднялся;
- долю, где опустился.

---

# 41. Анализ по topic clusters

Для каждого cluster сохранять:

```text
cluster_id
representative_terms
document_count
Q0_quality
Q1_quality
D0_quality
D1_quality
D2_quality
D2_minus_Q1
D2_minus_D1
```

Также анализировать:

- cluster cohesion;
- keyword specificity;
- average routing confidence;
- fallback rate;
- downstream gain.

---

# 42. Вычислительная стоимость downstream

Для Q0/Q1/D0/D1/D2 измерить:

- CE/QA calls;
- QA chunks;
- QA tokens;
- topic routing time;
- keyword matching time;
- total inference time;
- peak RAM;
- GPU memory.

---

# 43. Итоговая downstream-таблица

| Clustering | QA mode | QA quality | Recall@5 | Win | Harm | QA chunks | Time |
|---|---|---:|---:|---:|---:|---:|---:|
| none | Q0 baseline | | | | | | |
| none | Q1 global keywords | | | | | | |
| plain | D0 frequency | | | | | | |
| plain | D1 attention | | | | | | |
| plain | D2 attention+diff | | | | | | |
| KeyBERT/hybrid | D0 frequency | | | | | | |
| KeyBERT/hybrid | D1 attention | | | | | | |
| KeyBERT/hybrid | D2 attention+diff | | | | | | |

---

# 44. Артефакты эксперимента

Сохранить минимум:

```text
experiment_config.json

document_keyphrases.csv
document_embeddings.npy / npz / parquet
keyword_embeddings.npy / npz / parquet

cluster_assignments.csv
cluster_metrics.csv
cluster_representative_terms.csv
routing_simulation.csv
timings.csv

cluster_attention_keywords.csv
cluster_attention_diff_keywords.csv

downstream_predictions.csv
downstream_chunk_scores.csv
downstream_metrics.csv
downstream_metrics_by_cluster.csv
downstream_compute.csv

summary.md
```

Большие embeddings не хранить в JSON.

---

# 45. Что сохранять для cluster-specific keyword

Минимум:

```text
cluster_id
keyword
document_support
attention_weight
score_difference
combined_weight
```

Для D1 поля `score_difference`/`combined_weight` могут быть пустыми или не использоваться.

---

# 46. Что сохранять для candidate answer

Если доступно:

```text
document_id
split
routed_cluster_ids
routing_weights
chunk_id
candidate_answer
CE_confidence
chunk_keyword_score
answer_cluster_id
selected
```

---

# 47. Визуализация

PCA/UMAP разрешены только для визуализации, если отдельно не тестируется preprocessing.

Построить:

- 2D scatter документов;
- color = cluster;
- label = representative terms;
- сравнение E1 vs E3;
- cluster size distribution;
- downstream gain по clusters.

---

# 48. `summary.md`

Отчёт должен содержать:

## Dataset

- размер;
- языки;
- поля;
- splits;
- длины abstracts.

## Environment

- KeyBERT version;
- sentence-transformers version;
- sklearn version;
- encoder name/revision;
- hardware.

## Clustering configurations

Полная таблица.

## Clustering metrics

- Silhouette;
- Davies–Bouldin;
- Calinski–Harabasz;
- stability;
- sizes;
- runtime;
- memory.

## Cluster examples

- representative terms;
- representative papers;
- size;
- nearest clusters.

## KeyBERT effect

Сравнить:

```text
document embedding
vs
KeyBERT-only
vs
hybrid
```

## Downstream configurations

Q0/Q1/D0/D1/D2.

## Downstream metrics

QA quality, retrieval, harm/win, compute.

## Ablations

Обязательно:

```text
D1-D0
D2-D1
D2-Q1
D2(KeyBERT)-D2(plain)
```

## Conclusion

Ответить:

1. Улучшает ли KeyBERT clustering?
2. Улучшает ли KeyBERT только interpretability?
3. Какой `top_n` лучше?
4. Нужен ли MMR?
5. Какой `ngram_range` лучше?
6. Какой clustering algorithm лучше?
7. Какой K лучше?
8. Полезна ли cluster-specific attention настройка?
9. Даёт ли `diff` дополнительный выигрыш?
10. Лучше ли cluster-specific attention+diff текущего global static-keywords?
11. Даёт ли KeyBERT clustering downstream QA gain?
12. Какой вариант находится на лучшей Pareto-точке `quality / compute`?
13. Стоит ли переносить схему в основную algorithm hypothesis?

---

# 49. Интерпретация отрицательных результатов

Не считать эксперимент неудачным автоматически.

### A

KeyBERT улучшил clustering и downstream QA.

→ использовать KeyBERT representation.

### B

Clustering metrics почти не изменились, но representative terms чище.

→ использовать KeyBERT для cluster description/topic keyword extraction.

### C

KeyBERT clusters не лучше геометрически, но D2 на них лучше.

→ downstream QA важнее Silhouette для данной прикладной задачи.

### D

D1 > D0, но D2 ≈ D1.

→ cluster-specific attention полезен, `diff` дополнительного выигрыша не даёт.

### E

D2 > D1 и D2 > Q1.

→ подтверждается гипотеза topic-specific attention+diff tuning.

### F

Q1 >= D2.

→ тематическая кластеризация не даёт достаточного выигрыша относительно global static-keywords.

---

# 50. Что не делать

Не следует:

- изменять production pipeline;
- использовать генеративную модель;
- использовать LLM для cluster labels;
- использовать QA gold для topic clustering;
- обучать clustering на test;
- обучать cluster keywords на dev/test;
- считать KeyBERT score `diff`;
- считать `diff < 0` topic signal;
- использовать разные clusters для D0/D1/D2;
- запускать полный QA grid для каждой clustering configuration;
- использовать UMAP как обязательный preprocessing без ablation;
- заменять весь эксперимент BERTopic;
- переписывать `AnswerConsensus`, если его можно переиспользовать;
- менять одновременно clustering, keyword scoring и answer selection так, что вклад отдельных изменений нельзя измерить.

---

# 51. Рекомендуемый порядок эксперимента

```text
STAGE 0
Repository/data study

        ↓

STAGE 1
Representation:
E0 TF-IDF
E1 document embedding
E2 KeyBERT
E3 hybrid

        ↓

STAGE 2
KeyBERT parameters:
top_n / ngram / MMR

        ↓

STAGE 3
Clustering:
K / KMeans / BisectingKMeans / MiniBatchKMeans

        ↓

STAGE 4
Stability + interpretability + compute

        ↓

STAGE 5
Выбрать:
C_plain
C_keybert

        ↓

STAGE 6
Заморозить train clusters и centroids

        ↓

STAGE 7
Q0 current baseline
Q1 global static-keywords

        ↓

STAGE 8
D0:
cluster + CE + frequency/consensus

        ↓

STAGE 9
D1:
cluster + CE
+ cluster attention keywords
+ chunk scoring
+ score-aware answer selection

        ↓

STAGE 10
D2:
cluster + CE
+ cluster attention
+ cluster-specific diff
+ chunk scoring
+ score-aware answer selection

        ↓

STAGE 11
Сравнить:
D1-D0
D2-D1
D2-Q1
D2(KeyBERT)-D2(plain)

        ↓

STAGE 12
Quality / compute Pareto analysis

        ↓

STAGE 13
summary.md + recommendation
```

---

# 52. Самопроверка

Перед завершением проверь:

- [ ] эксперимент изолирован от production;
- [ ] existing QA/CE переиспользован;
- [ ] train/dev/test leakage отсутствует;
- [ ] KeyBERT output кэшируется;
- [ ] embeddings не вычисляются повторно без причины;
- [ ] E0/E1/E2/E3 сравнены;
- [ ] clustering stability измерена;
- [ ] representative terms построены автоматически;
- [ ] routing simulation выполнена;
- [ ] C_plain и C_keybert выбраны до downstream QA;
- [ ] Q0 реализован;
- [ ] Q1 включён, если доступен;
- [ ] D0 использует CE + frequency/consensus;
- [ ] D1 использует cluster-specific attention keywords и не использует diff;
- [ ] D2 использует тот же clustering и attention + existing diff;
- [ ] diff агрегируется только внутри train clusters;
- [ ] нет правила `negative diff = topic`;
- [ ] hard и/или soft routing проверены;
- [ ] D0/D1/D2 используют одинаковый CE;
- [ ] D0/D1/D2 используют одинаковые test documents;
- [ ] измерен Evidence Recall@K;
- [ ] измерено QA quality;
- [ ] измерен compute;
- [ ] посчитаны D1-D0, D2-D1, D2-Q1;
- [ ] KeyBERT clustering сравнен с plain clustering по downstream QA;
- [ ] production pipeline не изменён;
- [ ] подготовлен `summary.md`.

---

# 53. Главный критерий

Цель эксперимента — не доказать заранее, что KeyBERT лучше.

Нужно объективно определить:

> **какой способ тематической кластеризации научных статей формирует такие группы документов, внутри которых обучение специализированных attention/`diff` ключевых признаков даёт максимальное улучшение выбора evidence-чанков и итогового extractive QA-ответа при минимальной дополнительной вычислительной стоимости.**
