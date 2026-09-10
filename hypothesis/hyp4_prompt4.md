# Промпт для эксперимента: кластерно-зависимые QA-ключевые слова для отбора релевантных чанков

Ты — NLP-разработчик и исследователь. Необходимо провести **отдельный эксперимент по улучшению extractive QA на базе RoBERTa** за счёт тематической кластеризации документов и последующего обучения **кластерно-зависимых ключевых признаков, позволяющих отличать правильный чанк от неправильных**.

Эксперимент должен быть реализован изолированно и не должен ломать текущий production pipeline.

# 1. Основная исследовательская идея

Предыдущие эксперименты с одним глобальным набором ключевых слов не позволили устойчиво превзойти baseline.

Новая гипотеза состоит в следующем:

> Универсальный словарь может быть недостаточно эффективен из-за тематической неоднородности корпуса. Если сначала разделить документы на тематически однородные кластеры, а затем внутри каждого кластера определить признаки, отличающие чанки с правильным ответом от тематически похожих, но неправильных чанков, такие признаки могут быть существенно полезнее для предварительного отбора контекста и выбора итогового extractive QA-ответа.

Ключевой принцип:

**не искать просто «ключевые слова темы».**

Нужно искать:

> **слова и фразы, которые внутри конкретного тематического кластера позволяют отличить правильный QA-чанк от неправильного.**

# 2. Что сохраняется из текущей системы

Необходимо сохранить:

- extractive QA на RoBERTa;
- текущий chunking;
- существующий baseline;
- существующий механизм извлечения ключевых слов из attention;
- существующий `diff_score` / `score_difference`;
- текущую логику QA confidence;
- `AnswerConsensus` или его фактический аналог;
- существующие метрики качества;
- текущий train/dev/test split;
- существующие fallback-механизмы.

Новая система должна быть экспериментальной и отключаемой.

# 3. Общая архитектура эксперимента

Общий pipeline:

```text
TRAIN DOCUMENTS
      ↓
тематическая кластеризация
      ↓
Cluster 1
Cluster 2
...
Cluster M
      ↓
для каждого кластера:
positive chunks
+
hard negative chunks
      ↓
генерация candidate keywords
      ↓
проверка discriminative usefulness
      ↓
cluster-specific QA keyword dictionary
```

Inference:

```text
new document
      ↓
определение тематического кластера
      ↓
cluster-specific keyword dictionary
      ↓
оценка / фильтрация chunks
      ↓
RoBERTa QA
      ↓
QA confidence + chunk evidence
      ↓
final answer
```

# 4. Этап 0. Изучи существующий проект

Перед изменением кода найди:

- реализацию RoBERTa QA;
- chunking;
- `WeightedKeyword` или аналог;
- механизм attention extraction;
- `attention_weight`;
- `score_difference` / `diff_score`;
- текущий static/global keyword pipeline;
- candidate pool;
- chunk score;
- minimum matches;
- position/frequency/uniqueness modifiers;
- AnswerConsensus;
- QA confidence;
- train/dev/test split;
- формат gold answers;
- возможность определить, какой chunk содержит gold answer;
- существующие embeddings;
- текущую тематическую кластеризацию, если она уже реализована;
- existing tests.

Сначала составь краткую архитектурную карту.

Не создавай дублирующие компоненты, если нужная функциональность уже существует.

# 5. Тематическая кластеризация

Используй **наиболее эффективную конфигурацию кластеризации, выбранную в предыдущем эксперименте**.

Если в проекте уже есть результат эксперимента, используй:

- выбранное representation;
- выбранный embedding model;
- выбранный clustering algorithm;
- выбранное `K`;
- сохранённые cluster centroids.

Не переоптимизируй clustering одновременно с keyword-экспериментом.

Это важно для чистоты ablation.

После выбора clustering:

```text
cluster assignments
```

должны быть зафиксированы.

# 6. Anti-leakage

Тематическая модель строится только по TRAIN.

TRAIN используется для:

- кластеризации;
- определения positive chunks;
- поиска hard negatives;
- извлечения attention keywords;
- расчёта diff;
- построения cluster dictionaries.

DEV используется только для:

- выбора thresholds;
- числа keywords;
- коэффициентов;
- Top-K chunks;
- выбора способа фильтрации/reranking.

TEST используется один раз для финальной оценки.

Gold TEST запрещено использовать для:

- выбора кластера;
- построения словаря;
- изменения weights;
- подбора threshold.

# 7. Positive chunks

Для каждого train-документа определить чанки, содержащие правильный gold answer.

Обозначить:

\[
C_m^+
\]

— множество positive chunks внутри тематического кластера \(m\).

Если из-за overlap gold span присутствует в нескольких чанках, допускается считать positive несколько чанков.

Но необходимо сохранить:

```text
document_id
chunk_id
gold_span
global_span_coordinates
cluster_id
```

# 8. Negative chunks

Не использовать только случайные отрицательные чанки.

Разделить negatives на:

## Easy negatives

Случайные чанки документа, явно не содержащие ответ.

## Hard negatives

Чанки, которые:

- не содержат gold answer;
- тематически похожи на positive chunk;
- имеют высокий текущий keyword score;
- или получают высокий QA confidence, но дают неправильный ответ;
- или близки к вопросу / gold по embedding;
- или входят в top текущего retrieval.

Обозначить:

\[
C_m^-
\]

и отдельно:

\[
C_{m,\mathrm{hard}}^-.
\]

Основной discriminative experiment должен использовать hard negatives.

# 9. Зачем нужны hard negatives

Задача метода — отличить:

```text
правильный релевантный чанк
```

от:

```text
похожего и правдоподобного,
но неправильного чанка
```

а не от очевидно нерелевантного текста.

Пример:

Вопрос:

```text
Какая задача решается в работе?
```

Positive:

```text
Основной задачей исследования является разработка метода
прогнозирования морозного пучения...
```

Hard negative:

```text
Для решения задачи определения температурного режима
использовался численный метод...
```

Именно такие пары наиболее информативны.

# 10. Candidate keyword sources

Необходимо независимо проверить четыре основных источника кандидатов.

# K1. Attention keywords

Использовать текущий QA attention mining.

Для positive chunks получить Top-N attention tokens/phrases.

Для кандидата \(k\):

\[
A(k)
\]

— aggregate attention score.

Сначала использовать текущую существующую aggregation policy.

# 11. K2. Attention + diff

Использовать тот же attention candidate pool, но учитывать существующий:

```text
diff_score
```

или:

```text
score_difference
```

Важно:

не переписывать семантику diff.

Сначала найди реальную формулу в проекте.

Сравнить:

```text
attention only
```

против:

```text
attention + diff
```

Это отдельный обязательный ablation.

# 12. K3. Question-aware KeyBERT

Реализовать экспериментальный вариант KeyBERT, ориентированный не просто на содержание документа, а на QA-вопрос.

Обычный KeyBERT использует близость candidate phrase к документу.

Здесь необходимо проверить question-conditioned score.

Для candidate phrase \(k\):

\[
Q(k)=
\cos(e_k,e_q),
\]

где:

- \(e_k\) — embedding candidate phrase;
- \(e_q\) — embedding QA-вопроса.

При необходимости проверить комбинированный вариант:

\[
Q'(k)=
\alpha
\cos(e_k,e_q)
+
(1-\alpha)
\cos(e_k,e_c),
\]

где:

- \(e_c\) — embedding positive chunk.

Проверить:

```text
question only
question + chunk
```

# 13. K4. Cluster keywords

Для каждого тематического кластера получить слова/фразы, характеризующие саму тему.

Допустимые источники:

- KeyBERT;
- c-TF-IDF;
- TF-IDF cluster-vs-rest;
- embedding similarity к cluster centroid;
- метод, показавший лучшие результаты в предыдущем clustering experiment.

Такие слова являются **topic signal**, а не автоматически QA-relevant keywords.

Пример:

```text
frost heave
frozen soil
pore pressure
```

Они не должны автоматически получать высокий QA-weight.

# 14. Комбинации candidate pools

После проверки отдельных источников исследовать объединения.

Минимум:

```text
K1 = Attention

K2 = Attention + diff

K3 = Question-aware KeyBERT

K4 = Cluster keywords

K5 = Attention ∪ Question-KeyBERT

K6 = Attention ∪ Cluster

K7 = Question-KeyBERT ∪ Cluster

K8 = Attention ∪ Question-KeyBERT ∪ Cluster

K9 = Attention + diff ∪ Question-KeyBERT ∪ Cluster
```

Не смешивать их веса сразу.

Сначала использовать их как **разные способы генерации candidate pool**.

# 15. Главное изменение: одинаковая проверка кандидатов

Независимо от того, откуда пришёл keyword, его финальная полезность должна проверяться одинаковым способом.

Для каждого cluster \(m\) и keyword \(k\):

\[
H_m(k)=
\log
\frac{
P(k|C_m^+)+\varepsilon
}{
P(k|C_{m,\mathrm{hard}}^-)+\varepsilon
}.
\]

Интерпретация:

### Если

\[
H_m(k)>0
\]

keyword чаще встречается в правильных чанках.

### Если

\[
H_m(k)\approx0
\]

keyword практически ничего не различает.

### Если

\[
H_m(k)<0
\]

keyword чаще встречается в неправильных hard-negative chunks.

# 16. Разрешить отрицательные признаки

Не обязательно удалять все keywords с:

\[
H_m(k)<0.
\]

Отдельно проверить:

## Positive-only dictionary

Использовать только:

\[
H_m(k)>0.
\]

## Signed dictionary

Разрешить положительные и отрицательные веса.

Например:

```text
objective of this study       +1.8
we propose                    +1.2
prediction                    +0.7

experimental setup            -0.6
sample preparation            -0.8
Table shows                   -1.1
```

Это превращает словарь в:

> **кластерно-зависимый лексический профиль правильного чанка.**

# 17. Document support

Для каждого keyword считать:

\[
support_m(k)=
\frac{
\#documents\ in\ cluster\ m\ containing\ k
}{
\#documents\ in\ cluster\ m
}.
\]

Keyword не должен считаться полезным только потому, что много раз встретился в одном документе.

Проверить thresholds:

```text
minimum_support = 2 documents
minimum_support = 5%
minimum_support = 10%
```

Адаптировать к размеру кластера.

# 18. Итоговый cluster-specific dictionary

Для каждого cluster получить:

\[
K_m^*=
\{(k,w_{k,m})\}.
\]

Минимальный вариант:

\[
w_{k,m}=H_m(k).
\]

Далее можно проверить более сложные варианты.

# 19. Вариант комбинирования сигналов

После эксперимента с чистым \(H_m(k)\) проверить:

\[
w_{k,m}=
\theta_HH_m(k)
+
\theta_AA_m(k)
+
\theta_DD_m(k)
+
\theta_QQ_m(k)
+
\theta_TT_m(k).
\]

Где:

- \(H\) — positive vs hard-negative discrimination;
- \(A\) — attention;
- \(D\) — diff;
- \(Q\) — question similarity;
- \(T\) — cluster topic relevance.

Но этот этап выполнять **только после определения полезности каждого отдельного сигнала**.

# 20. Shrinkage для маленьких кластеров

Если кластер мал, cluster-specific statistics могут быть нестабильны.

Поэтому проверить hierarchical/global shrinkage:

\[
w_{k,m}^{final}
=
\lambda_m w_{k,global}
+
(1-\lambda_m)w_{k,m}.
\]

Где:

- для большого cluster \(\lambda_m\) маленькая;
- для маленького cluster \(\lambda_m\) большая.

Пример:

```text
large cluster:
λ = 0.1

small cluster:
λ = 0.7
```

Можно сделать \(\lambda_m\) функцией числа train-документов.

# 21. Эксперимент A: только ranking правильного чанка

Это должен быть **первый основной эксперимент**.

На этом этапе не запускать полный QA для каждой комбинации.

Главный вопрос:

> способен ли cluster-specific dictionary поднять правильный chunk выше остальных?

Для каждого документа:

```text
all chunks
    ↓
cluster-specific scoring
    ↓
ranking
```

# 22. Chunk score

Базовый score:

\[
S(c|m)=
\sum_{k\in K_m^*\cap c}
w_{k,m}.
\]

Если current project уже использует:

- frequency modifier;
- position modifier;
- uniqueness modifier;

сначала провести experiment без изменения этих механизмов.

Потом отдельно проверить:

```text
pure discriminative score
```

и:

```text
discriminative score + current modifiers
```

# 23. Retrieval metrics

Для каждого candidate source / combination считать:

- Evidence Recall@1;
- Evidence Recall@3;
- Evidence Recall@5;
- Evidence Recall@10;
- MRR;
- Mean rank;
- Median rank;
- долю документов, где correct chunk поднялся;
- долю документов, где correct chunk опустился.

Основная таблица:

| Method | Recall@1 | Recall@3 | Recall@5 | MRR | Median rank |
|---|---:|---:|---:|---:|---:|
| Current global keywords | | | | | |
| Cluster keywords | | | | | |
| Attention | | | | | |
| Attention + diff | | | | | |
| Question-KeyBERT | | | | | |
| Attention + Question-KeyBERT | | | | | |
| Attention + Cluster | | | | | |
| Attention + Question-KeyBERT + Cluster | | | | | |
| + hard-negative discrimination | | | | | |

# 24. Критерий перехода к QA

Полный QA experiment проводить только для 2–4 лучших ranking methods.

Не запускать RoBERTa QA для всех combinations.

Особенно важен:

\[
EvidenceRecall@K.
\]

Если correct chunk часто отсутствует в Top-K, prefilter использовать нельзя.

# 25. Эксперимент B: reranking после QA

Самый безопасный вариант.

Pipeline:

```text
all chunks
    ↓
RoBERTa QA
    ↓
candidate answers
    +
cluster-specific chunk score
    ↓
final answer reranking
```

Для candidate answer \(a\):

\[
R(a)=
\alpha Q_{QA}(a)
+
\beta \hat S(c_a|m).
\]

Где:

- \(Q_{QA}\) — QA confidence;
- \(S\) — cluster-specific chunk score.

Если существующий AnswerConsensus уже умеет учитывать chunk score — переиспользовать его.

# 26. Эксперимент C: prefiltering до QA

Pipeline:

```text
all chunks
    ↓
cluster-specific scoring
    ↓
Top-K
    ↓
RoBERTa QA
```

Проверить:

```text
K = 3
K = 5
K = 10
K = 20
```

Главные метрики:

- Evidence Recall@K;
- final QA quality;
- число QA calls;
- число QA tokens.

# 27. Эксперимент D: hybrid prefilter + final reranking

Наиболее перспективная итоговая схема:

```text
Document
   ↓
topic cluster
   ↓
cluster-specific chunk scoring
   ↓
Top-K chunks
   ↓
RoBERTa QA
   ↓
candidate answers
   ↓
QA confidence
+
cluster-specific chunk score
   ↓
final answer
```

Один и тот же cluster-specific signal используется:

1. до QA;
2. после QA.

# 28. Контрольные QA-варианты

Обязательно сравнить:

```text
Q0 — baseline:
all chunks -> QA -> current answer selection
```

```text
Q1 — current global keyword approach
```

```text
Q2 — cluster keywords only
```

```text
Q3 — cluster Attention
```

```text
Q4 — cluster Attention + diff
```

```text
Q5 — cluster Question-KeyBERT
```

```text
Q6 — cluster Attention + Question-KeyBERT
```

```text
Q7 — cluster Attention + Question-KeyBERT + topic
```

```text
Q8 — best candidate source + hard-negative discriminative weighting
```

# 29. Дополнительный важный ablation

Для лучшего candidate pool отдельно сравнить:

```text
candidate importance only
```

против:

```text
positive vs easy negatives
```

против:

```text
positive vs hard negatives
```

Это покажет, даёт ли hard-negative discrimination основной прирост.

# 30. Hard-negative mining strategies

Проверить несколько способов выбора hard negatives.

## HN1 — QA confidence

Неправильные chunks с высоким QA confidence.

## HN2 — current keyword retrieval

Неправильные chunks с высоким текущим chunk score.

## HN3 — semantic similarity

Неправильные chunks, embedding которых близок к positive chunk.

## HN4 — question similarity

Chunks, наиболее близкие к вопросу.

## HN5 — combined

Объединение Top-N кандидатов HN1–HN4.

Сравнить качество resulting dictionaries.

# 31. Работа с multi-word phrases

Проверить:

```text
unigrams
unigrams + bigrams
unigrams + bigrams + trigrams
```

Особенно для:

- Question-KeyBERT;
- cluster keywords.

Например:

```text
objective
```

хуже, чем:

```text
objective of study
```

или:

```text
frost heave prediction
```

# 32. Hard routing vs soft routing

Основной простой вариант:

```text
document -> one cluster
```

Дополнительно проверить soft routing:

```text
cluster_1  0.65
cluster_2  0.27
cluster_3  0.08
```

Тогда:

\[
S(c|D)=
\sum_m
\pi_mS(c|m).
\]

Особенно полезно для междисциплинарных документов.

# 33. Метрики финального QA

Использовать те же метрики, что в основном проекте:

- Character F1;
- Token F1;
- ROUGE-L F1;
- BERTScore F1;
- BERTScore Precision;
- BERTScore Recall;
- composite metric;
- coverage / answer rate.

При наличии:

- Exact Match;
- false positive rate.

# 34. Computational metrics

Для каждого финального варианта измерить:

- QA calls/document;
- QA chunks/document;
- QA tokens/document;
- routing time;
- keyword matching time;
- total inference time;
- peak GPU memory;
- peak RAM.

Рассчитать:

\[
QAChunkRatio=
\frac{
N_{QA\ chunks}^{method}
}{
N_{QA\ chunks}^{baseline}
}.
\]

И:

\[
QATokenRatio=
\frac{
N_{QA\ tokens}^{method}
}{
N_{QA\ tokens}^{baseline}
}.
\]

# 35. Главная итоговая таблица

| Method | Recall@5 | Char F1 | Token F1 | ROUGE-L | BERTScore | QA chunks ratio | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline QA | | | | | | 1.00 | |
| Global keywords | | | | | | | |
| Cluster topic keywords | | | | | | | |
| Cluster Attention | | | | | | | |
| Cluster Attention+diff | | | | | | | |
| Question-KeyBERT | | | | | | | |
| Attention + Question-KeyBERT | | | | | | | |
| Attention + Topic | | | | | | | |
| Attention + Question-KeyBERT + Topic | | | | | | | |
| Best + hard-negative discrimination | | | | | | | |

# 36. Анализ отдельно по кластерам

Для каждого topic cluster посчитать:

```text
cluster_id
document_count
representative_terms
Recall@5
baseline_F1
new_F1
delta_F1
harm_rate
```

Это необходимо, чтобы понять:

> для каких тематик специализация работает, а для каких нет.

# 37. Связь качества кластера и эффекта

Для каждого cluster сохранить:

- размер;
- cohesion;
- silhouette contribution;
- keyword stability;
- average `H_m(k)`;
- QA gain.

Проверить, существует ли зависимость:

```text
более однородный кластер
→ более качественный cluster-specific dictionary
→ больший QA gain
```

# 38. Statistical testing

Для финальных систем выполнить bootstrap:

```text
1000 samples
```

Получить:

- mean difference;
- 95% confidence interval.

Основные сравнения:

```text
Baseline vs best method
```

```text
Global keywords vs cluster-specific
```

```text
Attention vs Attention + diff
```

```text
Attention vs Attention + Question-KeyBERT
```

```text
candidate importance vs hard-negative discrimination
```

# 39. Порядок выполнения

Не запускать всё сразу.

## Phase 1 — freeze clustering

Использовать лучший clustering из предыдущего эксперимента.

## Phase 2 — construct positives and hard negatives

Проверить корректность evidence labels.

## Phase 3 — candidate sources

Сравнить:

- attention;
- attention+diff;
- Question-KeyBERT;
- cluster topic keywords;
- combinations.

## Phase 4 — discriminative filtering

Посчитать:

\[
H_m(k).
\]

## Phase 5 — chunk ranking only

Выбрать лучшие методы по Recall@K/MRR.

## Phase 6 — QA reranking

Проверить без prefilter.

## Phase 7 — prefilter

Проверить Top-K.

## Phase 8 — hybrid

Prefilter + QA + reranking.

## Phase 9 — computational evaluation

Построить Pareto `quality / compute`.

# 40. Артефакты эксперимента

Сохранять:

```text
cluster_assignments.csv
positive_chunks.csv
hard_negative_chunks.csv

candidate_keywords_attention.csv
candidate_keywords_attention_diff.csv
candidate_keywords_question_keybert.csv
candidate_keywords_topic.csv

cluster_keyword_profiles.csv

chunk_ranking_results.csv
chunk_recall_metrics.csv

qa_predictions.csv
qa_metrics.csv
qa_metrics_by_cluster.csv

compute_metrics.csv

experiment_config.json
summary.md
```

# 41. Формат cluster keyword profile

Желательно сохранять:

```text
cluster_id
keyword
source_attention
source_diff
source_question_keybert
source_topic
attention_score
diff_score
question_score
topic_score
positive_frequency
negative_frequency
hard_negative_frequency
document_support
discriminative_score
final_weight
```

# 42. Главные ablation-гипотезы

Проверить:

### H1

\[
ClusterAttention > GlobalAttention
\]

### H2

\[
ClusterAttentionDiff > ClusterAttention
\]

### H3

\[
QuestionKeyBERT > ClusterTopicKeywords
\]

для QA-релевантного chunk ranking.

### H4

\[
Attention + QuestionKeyBERT > Attention
\]

### H5

\[
HardNegativeDiscrimination > CandidateImportanceOnly
\]

### H6

\[
ClusterSpecific > Global
\]

### H7

\[
HybridPrefilterRerank > Baseline
\]

по quality/compute Pareto.

# 43. Наиболее важный критерий

Главной метрикой первого этапа считать не финальный F1, а:

\[
EvidenceRecall@K
\]

и:

\[
MRR.
\]

Если метод не способен устойчиво поднимать правильный chunk, не переносить его в следующий QA experiment.

# 44. Что не делать

Не следует:

- обучать keywords по TEST;
- выбирать clustering по downstream TEST;
- использовать gold на inference;
- считать cluster keyword автоматически QA keyword;
- считать `diff < 0` topic keyword;
- заменять attention KeyBERT;
- менять сразу clustering, QA model и answer consensus;
- проводить полный grid search всех комбинаций;
- использовать hard filtering до проверки Recall@K;
- делать вывод только по Silhouette;
- делать вывод только по итоговому F1 без анализа chunk ranking.

# 45. Главный исследовательский вопрос

Финальный эксперимент должен ответить:

> **Можно ли за счёт тематической кластеризации научно-технических документов и обучения внутри каждого кластера специализированного лексического профиля, построенного на сочетании QA-attention, `diff`, question-aware KeyBERT, тематических признаков и positive/hard-negative discrimination, повысить точность ранжирования evidence-чанков и качество extractive QA либо сохранить качество при меньшем количестве обрабатываемых QA-чанков?**

Особое внимание уделить не абсолютной сложности метода, а сравнению:

```text
качество
vs
вычислительная стоимость
vs
стабильность
```

Главной целевой архитектурой считать:

```text
document
  ↓
topic routing
  ↓
cluster-specific discriminative keyword profile
  ↓
chunk ranking
  ↓
Top-K
  ↓
RoBERTa extractive QA
  ↓
QA confidence + chunk evidence
  ↓
final answer
```

При этом необходимо экспериментально доказать необходимость каждого компонента, а не предполагать её заранее.
