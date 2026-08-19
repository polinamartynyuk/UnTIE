# Гипотеза: иерархическая тематическая адаптация ключевых слов для reranking чанков в extractive QA

## 0. Назначение документа

**Тип:** исследовательская гипотеза и техническая спецификация экспериментального улучшения.  
**Целевая папка проекта:** `hypothesis/`.  
**Базовый документ:** `information-extraction-concepts.md`.

Цель документа — описать изменение существующего pipeline так, чтобы код-агент мог сначала составить план, а затем реализовать изменение без разрушения текущих режимов.

Критическое ограничение: **extractive QA остаётся основным reader-механизмом; генеративная модель не добавляется в production inference.**

---

# 1. Исходная проблема

Текущий режим `static-keywords` использует один глобальный словарь ключевых слов для заданного тематического аспекта. Один и тот же словарь применяется к документам разных предметных областей.

Это создаёт несколько проблем:

- предметно-специфичные термины «размываются» при глобальной агрегации;
- общие научные слова могут получать вес, хотя плохо локализуют нужный фрагмент;
- одно и то же слово может быть полезно в одной тематике и бесполезно в другой;
- новый документ заранее не имеет пользовательской метки темы;
- ручное назначение тем пользователем противоречит требованию автоматизации.

Предлагается добавить **автоматическую иерархическую тематическую модель**, которая будет использоваться только как дешёвый retrieval/reranking layer перед существующим extractive QA.

---

# 2. Что сохраняется без изменения идеи

Необходимо сохранить:

- `baseline`;
- `attention`;
- существующий `static-keywords`;
- extractive QA reader;
- chunking;
- `WeightedKeyword` или его фактический аналог;
- `attention_weight`;
- `score_difference`;
- train/dev/test split;
- запрет leakage;
- cached QA/attention evidence;
- текущий fallback;
- `AnswerConsensus`;
- текущие метрики качества;
- текущую tuning-логику там, где она не конфликтует с новым слоем.

Новая функциональность должна быть **добавочной и отключаемой**. Рекомендуется отдельный экспериментальный режим/feature flag, например `hierarchical-static-keywords` либо эквивалентное имя в стиле проекта.

При отключённой новой функциональности старый pipeline должен вести себя как прежде.

---

# 3. Центральная идея

Для каждого ключевого слова/фразы рассматриваются два независимых сигнала:

1. **Aspect signal** — насколько признак характерен для места текста, где выражается искомый тематический аспект.
2. **Topic signal** — насколько признак характерен для предметной тематики текущего документа.

Ключ не обязан иметь единственный тип `aspect` или `topic`. Он может иметь оба веса.

Логически:

```text
keyword
├── attention_weight
├── aspect_score
└── topic_weights[node_id]
```

---

# 4. Aspect signal: сохранить существующий `score_difference`

В текущей концепции уже существует:

\[
\Delta_k =
\cos(e_k,e_{\mathrm{aspect}})
-
\cos(e_k,e_{\mathrm{reference}})
\]

(с соответствующей текущей логикой для нескольких gold/reference на tuning).

Этот параметр следует сохранить и использовать как основу **aspect relevance**.

Базовый вариант:

\[
A_k = \max(0,\Delta_k).
\]

Либо с порогом:

\[
A_k =
\begin{cases}
\Delta_k, & \Delta_k > \tau_A,\\
0, & \text{иначе}.
\end{cases}
\]

`tau_A` должен быть конфигурируемым и подбираться на dev.

### Важный запрет

Нельзя использовать правило:

```text
score_difference < 0 => topic keyword
```

Отрицательный `diff` означает только, что слово не прошло аспектный контраст. Это может быть:

- реальный тематический термин;
- слово из содержания gold;
- generic scientific term;
- случайный кандидат.

Topic relevance вычисляется независимо.

---

# 5. Пользователь не задаёт тему

Новый документ не должен требовать:

```text
topic = "geotechnics"
```

или любой другой ручной тематической метки.

Алгоритму не требуется человекочитаемое название темы.

Для каждого тематического узла достаточно:

- `centroid` / `prototype`;
- `representative_terms`;
- `parent_id`;
- `children`;
- служебных параметров.

Человекочитаемая подпись строится автоматически только для диагностики.

---

# 6. Представление документа для определения темы

Предпочтительный источник тематического представления:

```text
title + abstract + section headings
```

Это короче полного текста и лучше концентрирует предметную область.

Fallback при отсутствии полей:

1. `title + abstract + headings`;
2. `title + headings`;
3. `title + abstract`;
4. `abstract`;
5. первые содержательные абзацы/чанки;
6. компактное представление полного текста по имеющейся в проекте стратегии.

Fallback должен быть детерминированным и одинаковым на train и inference.

---

# 7. Document embeddings

Для train-документа:

\[
z_d = E(R_d),
\]

где `R_d` — тематическое представление документа, `E` — существующий sentence/document encoder, если он подходит.

Требования:

- по возможности переиспользовать уже существующий encoder;
- не вводить новый тяжёлый Transformer без необходимости;
- embeddings кэшировать offline;
- сохранять encoder name/revision и способ нормализации в артефакте;
- проверять совместимость при загрузке.

---

# 8. Кластеризовать документы, а не отдельные слова

Основной topic clustering выполняется над `z_d`.

Это принципиально: слова `model`, `network`, `pressure`, `field` неоднозначны без контекста, а документный embedding лучше отражает предметную область.

Допустимые алгоритмы:

- agglomerative clustering;
- k-means;
- HDBSCAN;
- другой алгоритм, подходящий текущему стеку.

Код-агент должен выбрать вариант после изучения зависимостей проекта.

Если есть random initialization:

- фиксировать seed;
- сохранять seed;
- проверять воспроизводимость.

---

# 9. Иерархия тематик

Плоских кластеров может быть недостаточно.

Пример логической структуры:

```text
geotechnics
├── frozen soils
│   ├── frost heave
│   └── permafrost/thawing
├── consolidation
├── filtration
└── strength
```

Пользователь эти названия не вводит.

Рекомендуемая безопасная схема:

1. построить leaf clusters документов;
2. вычислить centroid каждого leaf;
3. выполнить agglomerative clustering centroids;
4. сохранить `parent -> children`;
5. вычислить centroids внутренних узлов.

Для родителя:

\[
\mu_p =
\frac{\sum_j n_j\mu_j}{\sum_j n_j}.
\]

После вычисления centroid нормализовать.

---

# 10. Машинное описание topic node

Главное описание узла:

\[
\mu_m
\]

— его centroid/prototype.

Для нового документа:

\[
sim(D,C_m)=\cos(z_D,\mu_m).
\]

Этого достаточно для routing.

Ни один scoring-компонент не должен зависеть от ручной строки `"frost heave"` или `"NLP"`.

---

# 11. Автоматическое человекочитаемое описание

Для анализа каждый узел получает `representative_terms`, например:

```json
[
  {"term": "frost heave", "weight": 0.92},
  {"term": "frozen soil", "weight": 0.87},
  {"term": "freezing front", "weight": 0.81}
]
```

Опциональная подпись:

```text
frost heave | frozen soil | freezing front
```

строится автоматически из top-N terms.

Генеративная модель для этого не нужна.

---

# 12. Topic candidate terms

Кандидаты извлекаются **только из train-документов текущего topic node**.

Желательно поддержать:

- unigram;
- bigram;
- trigram;

но только если это можно встроить без опасного рефакторинга текущего matcher.

Также учитывать:

- lemma/stem;
- stop words;
- generic scientific words;
- minimum document support;
- минимальную длину;
- объединение морфологических вариантов.

---

# 13. Topic specificity

Тематический термин должен быть характерен для документов узла и нехарактерен для остального train-корпуса.

Базовый score:

\[
T(k,m) =
\log
\frac{P(k|C_m)+\varepsilon}
{P(k|\neg C_m)+\varepsilon}.
\]

Допускается устойчивая log-odds реализация с псевдосчётчиками.

### Document support

\[
support(k,m)=
\frac{|\{d\in C_m:k\in d\}|}{|C_m|}.
\]

Термин должен пройти `min_topic_document_support`.

Высокая частота в одном документе не должна сама по себе делать слово topic keyword.

---

# 14. Термины parent node

Нельзя просто объединять все термины детей.

Для parent полезны слова, которые:

- характерны для родителя относительно внешнего корпуса;
- поддерживаются несколькими дочерними кластерами.

Предлагается:

\[
ParentScore(k,p)
=
T(k,p)\cdot ChildCoverage(k,p),
\]

где:

\[
ChildCoverage(k,p)=
\frac{\#children\ containing\ k}{\#children}.
\]

---

# 15. Хранение topic weights

Topic weight зависит от node:

\[
T(k,m).
\]

Предпочтительная логика:

```text
KeywordLexeme
  text
  lemma
  stem

TopicNode
  term_weights:
      keyword_id -> weight
```

Если текущая архитектура проще поддерживает копии `WeightedKeyword` внутри узла — допустимо, но не следует ломать текущую сериализацию без миграции.

---

# 16. Soft routing нового документа

Новый документ может быть междисциплинарным. Поэтому нельзя жёстко выбирать только один topic leaf.

Для узлов:

\[
s_m=\cos(z_D,\mu_m).
\]

После отбора кандидатов:

\[
\pi_m=
\frac{\exp(s_m/\tau)}
{\sum_j\exp(s_j/\tau)}.
\]

Нужны:

- `top_k`;
- `beam_width`;
- similarity threshold;
- temperature;
- deterministic tie-breaking.

Допускается другая простая нормализация, если она лучше вписывается в проект.

---

# 17. Coarse-to-fine routing

Рекомендуемый inference:

```text
root
 ↓
top broad nodes
 ↓
top children
 ↓
top leaf nodes
```

Углубление зависит от:

- similarity;
- margin между лучшими узлами;
- минимального качества/размера node;
- beam width.

Если уверенность низкая — не идти глубже.

---

# 18. Иерархический fallback

Если leaf плохо подходит:

```text
leaf -> parent -> ancestor -> global topic -> existing static-keywords -> baseline
```

Точная цепочка адаптируется к реальной архитектуре.

Нужны диагностические причины fallback, например:

```text
LOW_TOPIC_CONFIDENCE
NO_TOPIC_MATCHES
INCOMPATIBLE_TOPIC_ARTIFACT
NO_RERANKED_CHUNKS
```

Новая система не должна давать пустой ответ только из-за topic routing.

---

# 19. Topic score чанка

Для node `m`:

\[
T_m(c)=
\sum_{k\in K(c,m)}
w_{k,m}\,g(tf(k,c)).
\]

`g(tf)` должна насыщаться, например:

\[
g(tf)=\log(1+tf).
\]

Линейное бесконечное поощрение повторов нежелательно.

Если routing выбрал несколько тем:

\[
T(c|D)=\sum_m\pi_mT_m(c).
\]

---

# 20. Aspect score чанка

Использовать существующие keyword-механизмы максимально консервативно.

Минимально:

\[
A(c)=\sum_{k\in K_A(c)}w^A_k,
\]

где `w^A_k` использует существующий `score_difference` и, при необходимости, текущий `attention_weight`/`weight_ratio`.

Новый topic layer не должен требовать переписывания aspect mining на первом этапе.


# 21. Взаимодействие Aspect × Topic

Главная гипотеза:

> хороший evidence-чанк одновременно содержит признаки нужного аспекта и признаки тематики текущего документа.

Поэтому новый score должен учитывать не только сумму, но и взаимодействие:

\[
R(c)
=
\theta_A\hat A(c)
+
\theta_T\hat T(c|D)
+
\theta_{AT}\hat A(c)\hat T(c|D).
\]

Где:

- `A` — aspect signal;
- `T` — topic signal;
- `hat` — нормализованные значения;
- `theta` — подбираемые параметры.

Интеракционный член особенно важен: он усиливает чанки, которые одновременно выглядят как место формулировки нужного аспекта и как фрагмент по теме текущего документа.

---

# 22. Интеграция с текущим chunk score

Текущие positional/frequency/uniqueness modifiers не удалять без ablation.

Нужно поддержать минимум одну консервативную стратегию.

### Вариант A

Новый `R(c)` заменяет только базовый keyword score:

\[
S(c)=R(c)\cdot P(c)\cdot F(c)\cdot U(c).
\]

### Вариант B

Текущий score сохраняется, topic добавляется:

\[
S(c)=
S_{legacy}(c)
+
\lambda_T\hat T(c|D)
+
\lambda_{AT}\hat A(c)\hat T(c|D).
\]

На первом этапе выбирать вариант, который минимально вмешивается в существующий код. Финальный выбор делать по dev и ablation.

---

# 23. Нормализация

Aspect и Topic могут иметь разные шкалы.

Перед смешиванием явно нормализовать их:

- min-max внутри документа;
- robust normalization;
- softmax;
- либо train/dev normalization.

Нельзя неявно складывать несопоставимые значения.

Параметры нормализации:

- сериализуются;
- не используют test statistics;
- воспроизводимы.

---

# 24. Фильтрация против reranking

На первом этапе не делать topic layer жёстким фильтром.

Основная опасность:

> удалить чанк, содержащий правильный evidence, только потому что topic confidence оказался низким.

Поэтому первая версия должна прежде всего **переранжировать**, а aggressive pruning включать только после подтверждения высокого `Evidence Recall@K`.

Существующий `minimum_matches` сохранить. При необходимости добавить диагностику:

```text
aspect_matches
topic_matches
total_matches
```

---

# 25. Train/dev/test и leakage

Сохранить текущую дисциплину.

## Train

Только train может использоваться для:

- document clustering;
- hierarchy;
- centroids;
- candidate topic terms;
- topic term weights;
- aspect candidate pool;
- representative terms.

## Dev

Dev используется для выбора:

- числа/глубины кластеров;
- thresholds;
- top-k;
- beam width;
- routing temperature;
- `theta_A`;
- `theta_T`;
- `theta_AT`;
- normalization;
- fallback policy.

Dev не должен добавлять новые production vocabulary terms, если текущая политика candidate pool этого не допускает.

## Test

Test используется только для финальной оценки.

Запрещено использовать test для:

- centroids;
- topic terms;
- thresholds;
- labels;
- routing tuning;
- normalization fitting.

---

# 26. Offline pipeline

```text
TRAIN
  │
  ├─ existing QA/attention evidence
  │    └─ existing score_difference -> aspect signal
  │
  ├─ title + abstract + headings
  │
  ├─ document embeddings
  │
  ├─ leaf document clustering
  │
  ├─ topic hierarchy
  │
  ├─ cluster-vs-rest topic term scoring
  │
  ├─ representative terms
  │
  └─ versioned topic artifact
        │
        ▼
DEV
  │
  ├─ routing parameters
  ├─ aspect/topic mixing
  ├─ normalization
  ├─ fallback
  ├─ Evidence Recall@K
  ├─ QA quality
  └─ compute reduction
        │
        ▼
TEST
  └─ one final evaluation
```

---

# 27. Inference pipeline

```text
Document + Question
        │
        ├── existing chunking
        │
        ├── title + abstract + headings
        │
        ├── document topic embedding
        │
        ├── hierarchical soft routing
        │
        ├── active topic nodes + weights
        │
        ├── aspect matching
        │
        ├── topic matching
        │
        ├── Aspect × Topic reranking
        │
        ├── top chunks
        │
        ├── extractive QA
        │
        └── existing consensus/final answer
```

На inference нет gold/reference.

---

# 28. Вычислительная стоимость

Online допускается:

- один компактный topic embedding документа;
- cosine similarity с centroids;
- sparse keyword/phrase matching;
- дешёвая арифметика scoring;
- extractive QA только на выбранных чанках.

Online не добавлять без отдельного исследования:

- генеративный LLM;
- тяжёлый cross-encoder для каждого чанка;
- многократное perturbation inference;
- отдельное дорогое topic modeling для каждого документа.

Дорогие операции допустимы offline.

---

# 29. Метрики

Нельзя оценивать только финальный QA F1.

## Retrieval

Обязательно:

- `Evidence Recall@K`;
- rank gold/evidence chunk;
- `MRR`, если применимо;
- fallback rate;
- routing coverage.

## QA quality

Сохранить текущие:

- char F1;
- token F1;
- ROUGE-L F1;
- BERTScore, если используется;
- composite quality.

## Вред относительно baseline

Сохранить/расширить:

- mean delta;
- win rate;
- harm rate;
- downside;
- bootstrap lower bound.

## Compute

Добавить:

\[
QAChunkRatio=
\frac{N_{\mathrm{QA\ chunks}}^{new}}
{N_{\mathrm{QA\ chunks}}^{baseline}}
\]

и

\[
QATokenRatio=
\frac{N_{\mathrm{QA\ tokens}}^{new}}
{N_{\mathrm{QA\ tokens}}^{baseline}}.
\]

Желательно также:

- transformer calls/document;
- latency;
- peak memory;
- offline cost отдельно от inference cost.

---

# 30. Целевая постановка

Главная цель:

> повысить или сохранить качество extractive QA при уменьшении количества нерелевантного текста, передаваемого reader.

Предпочтительно анализировать Pareto frontier:

```text
Quality <-> QA compute
```

а не только один сложный scalar objective.

Существующий tuning objective не удалять в первой реализации.

---

# 31. SFFS

Не переписывать SFFS сразу.

Безопасная первая версия:

- SFFS продолжает работать там, где уже работает;
- topic weights строятся независимо;
- dev настраивает routing/mixing;
- затем проводится ablation.

Topic-specific SFFS можно исследовать позже, так как он сильно увеличивает search space.

---

# 32. Multi-word terms

Topic layer особенно выигрывает от:

```text
frost heave
frozen soil
pore pressure
question answering
hydraulic conductivity
```

Архитектура должна допускать phrases.

Но если текущий matcher поддерживает только одиночные слова и n-gram требует большого рефакторинга:

1. не ломать matcher в первой версии;
2. реализовать базовый topic layer на поддерживаемых термах;
3. phrase matching вынести в Phase 2.

---

# 33. Hard negatives — Phase 2

После базовой реализации можно добавить дополнительный contrastive signal.

Формируются:

- positive chunks — содержат gold/evidence;
- hard negatives — неправильные чанки, которые текущий retrieval высоко ранжирует.

\[
D(k)=
\log
\frac{P(k|C^+)+\varepsilon}
{P(k|C^-_{\mathrm{hard}})+\varepsilon}.
\]

Не заменять им автоматически `score_difference`.

Рассматривать как дополнительный сигнал и проверять ablation.

---

# 34. Source-span duplicate bias — отдельная проверка

Нужно проверить текущий `AnswerConsensus`.

Из-за overlapping chunks один и тот же source span может появляться несколько раз и искусственно увеличивать cluster weight.

Проверить, есть ли уже deduplication по глобальным координатам документа.

Если нет, зафиксировать отдельный план:

```text
chunk-local span
 -> global document span
 -> overlap/near-duplicate merge
 -> semantic consensus
```

Не смешивать этот рефакторинг с topic hierarchy в одном крупном изменении, если это увеличивает риск.

---

# 35. Диагностические артефакты

После tuning желательно сохранять:

```text
topic_nodes.json/csv
topic_representative_terms.csv
topic_routing_dev.csv
topic_routing_test.csv
chunk_scores_before_after.csv
fallback_summary.json
compute_summary.json
```

Для topic node:

- node id;
- parent;
- depth;
- document count;
- representative terms;
- support;
- centroid metadata;
- ближайшие sibling nodes.

Для документа:

- top routed topics;
- similarities;
- routing weights;
- выбранная глубина;
- fallback reason;
- число чанков до/после reranking.

---

# 36. Предлагаемый логический формат артефакта

```json
{
  "schema_version": 1,
  "encoder": {
    "name": "...",
    "revision": "...",
    "normalization": "l2"
  },
  "document_representation": {
    "fields": ["title", "abstract", "headings"],
    "fallback_policy": "..."
  },
  "clustering": {
    "algorithm": "...",
    "seed": 42,
    "params": {}
  },
  "routing": {
    "top_k": 3,
    "beam_width": 2,
    "temperature": 0.1,
    "similarity_threshold": 0.0,
    "min_margin": 0.0
  },
  "nodes": [
    {
      "id": "topic_001",
      "parent_id": null,
      "depth": 0,
      "document_count": 120,
      "centroid": [],
      "representative_terms": [
        {
          "term": "frost heave",
          "weight": 0.92,
          "support": 0.61
        }
      ],
      "children": ["topic_004", "topic_005"]
    }
  ]
}
```

Это не обязательная буквальная схема. Агент должен встроиться в фактические dataclasses/Pydantic/config conventions проекта.

---

# 37. Версионирование и совместимость артефактов

Требования:

- новый artifact имеет `schema_version`;
- старые `static-keywords` artifacts продолжают работать старым путём;
- старый JSON нельзя молча трактовать как новый;
- сохраняется encoder/config metadata;
- несовместимость даёт понятную ошибку или разрешённый fallback;
- миграции добавляются только при необходимости.

---

# 38. `WeightedKeyword`

Не менять публичную структуру вслепую.

Предпочтительные варианты:

1. сохранить `WeightedKeyword` как есть, topic weights хранить в `TopicNode`;
2. добавить nullable backward-compatible поля;
3. ввести новый контейнер `TopicNode/TopicTermWeight`, который использует существующий keyword object.

До изменения проверить:

- serialization;
- equality/hash;
- CLI;
- notebooks;
- imports;
- tests;
- downstream code.

---

# 39. Конфигурация

Окончательные названия должны соответствовать стилю проекта.

Логически нужны:

```text
enabled
topic_model_path
topic_representation_fields
clustering_algorithm
clustering_params
hierarchy_enabled
routing_top_k
routing_beam_width
routing_temperature
routing_similarity_threshold
routing_min_margin
min_topic_document_support
topic_term_top_k
aspect_threshold
theta_aspect
theta_topic
theta_interaction
fallback_policy
```

Не превращать CLI в десятки новых аргументов без необходимости. Если есть config-файлы — предпочесть их.

---

# 40. Детерминизм

При одинаковых:

- seed;
- train split;
- encoder;
- config;

должны воспроизводиться:

- embeddings;
- clustering;
- hierarchy;
- representative terms;
- routing;
- artifact.

Все случайные компоненты имеют явный seed.

---

# 41. Backward compatibility

Обязательные условия:

1. `baseline` не меняется.
2. `attention` не меняется, кроме согласованных bugfix.
3. старый `static-keywords` не меняется при выключенном topic layer.
4. старые artifacts читаются старым путём.
5. новые artifacts versioned.
6. существующие scripts/notebooks по возможности продолжают работать.
7. public API не переименовывается без необходимости.
8. новые параметры имеют безопасные defaults.
9. отсутствие topic model не приводит к необоснованному падению.
10. existing test suite проходит.

---

# 42. Тесты

## Unit

Проверить:

- document topic representation;
- metadata fallback;
- centroid;
- cosine similarity;
- topic specificity;
- support;
- parent/child hierarchy;
- routing;
- soft weights;
- thresholds;
- parent fallback;
- serialization;
- schema version;
- aspect/topic scoring;
- normalization;
- deterministic output;
- empty cluster;
- empty keywords;
- zero matches;
- отсутствие NaN/inf.

## Regression

При `feature disabled`:

- baseline predictions не меняются;
- static-keyword ordering не меняется;
- fallback semantics не меняется;
- consensus не ломается.

## Integration

Synthetic corpus минимум из двух тематик:

```text
Topic A: A1 A2 A3
Topic B: B1 B2 B3
```

Проверить:

- A document routes в A;
- B document routes в B;
- mixed document получает оба веса;
- OOD document fallback;
- aspect + topic взаимодействие;
- inference не требует gold.

## Leakage

Проверить:

- dev term не попадает в train topic dictionary из-за dev;
- test term не влияет на centroid;
- test не влияет на thresholds;
- gold не используется router на inference.

---

# 43. Критерии технической приёмки

Первая реализация считается корректной, если:

1. новая функция включается отдельно;
2. старые режимы не сломаны;
3. hierarchy строится только по train;
4. пользователь не вводит topic;
5. документ автоматически получает topic weights;
6. есть parent/global fallback;
7. representative terms строятся автоматически;
8. существующий `score_difference` используется как aspect signal;
9. topic relevance независим;
10. inference не использует gold;
11. artifact сериализуется;
12. есть unit/integration/regression tests;
13. есть diagnostics;
14. измеряются QA chunks/tokens;
15. полный test suite проходит.

---

# 44. Исследовательские ablation

Необходимо сравнить:

```text
A0: baseline QA
A1: current static-keywords
A2: aspect-only
A3: topic-only
A4: aspect + topic
A5: aspect + topic + interaction
A6: hierarchical aspect + topic
A7: hierarchical + hard negatives (Phase 2)
```

Метрики:

- final QA quality;
- Evidence Recall@K;
- QAChunkRatio;
- QATokenRatio;
- harm rate;
- fallback rate.

---

# 45. Формулировка гипотезы

> Признаки, позволяющие локализовать ответ в научно-техническом документе, состоят из общего для тематического аспекта компонента и предметно-зависимого тематического компонента. Автоматическое иерархическое моделирование тематического компонента и его совместное использование с аспектным сигналом позволяет повысить точность предварительного отбора контекста для extractive QA либо сохранить качество при меньшем объёме текста, обрабатываемого QA reader.

---

# 46. Рекомендуемая последовательность реализации

## Phase 0 — Repository study

Ничего не менять. Найти:

- `WeightedKeyword`;
- attention mining;
- `score_difference`;
- static keyword artifact;
- tuning config/CLI;
- chunk reranking;
- fallback;
- consensus;
- encoder;
- document parser;
- split;
- tests.

Составить dependency map.

## Phase 1 — Data model

Добавить topic artifact/node/centroid/representative terms/versioning без изменения inference.

## Phase 2 — Offline topic model

Добавить document representation, embeddings, clustering, hierarchy, representative terms и diagnostics.

## Phase 3 — Routing

Добавить document topic embedding, hierarchical soft routing и fallback. Пока не менять chunk score.

## Phase 4 — Topic-aware reranking

Добавить topic score, aspect score из текущего diff, normalization и interaction под feature flag.

## Phase 5 — Tuning integration

Добавить dev tuning routing/mixing и compute metrics.

## Phase 6 — Regression/performance

Проверить old configs/artifacts, full tests, deterministic run и end-to-end.

## Phase 7 — Research extensions

Отдельно:

- hard negatives;
- phrase matcher;
- source-span deduplication;
- cheap learned ranker;
- compute-constrained objective.

---

# 47. Что не делать в первой версии

Не следует:

- заменять extractive QA генеративной моделью;
- требовать user topic label;
- использовать LLM для topic labels;
- удалять old `static-keywords`;
- переписывать SFFS без необходимости;
- смешивать несколько больших рефакторингов;
- обучать topic model на dev/test;
- использовать gold на inference;
- считать `diff < 0` достаточным признаком topic;
- route документ только в один leaf;
- удалять чанки только из-за низкого topic confidence;
- молча менять JSON schema;
- добавлять тяжёлую зависимость только ради удобства.

---

# 48. Главный принцип

Новый компонент — это:

```text
сложнее offline
+
дёшево online
+
автоматическая тематика
+
иерархический soft routing
+
aspect diff сохраняется
+
независимый topic signal
+
безопасный fallback
+
никакой генерации
```

Главный критерий:

> QA reader должен получать меньше нерелевантных чанков, не теряя правильный evidence и не разрушая текущий pipeline.
