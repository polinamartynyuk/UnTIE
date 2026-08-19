# Промпт для код-агента

Тебе необходимо доработать существующий проект извлечения информации из научно-технических документов.

В репозитории есть текущая реализация extractive QA pipeline и новый документ-гипотеза:

```text
hypothesis/hyp1_hierarchical-topic-aspect-reranking.md
```

## Главная задача

Сначала **полностью изучи текущую архитектуру проекта**, затем составь подробный план изменений по `hypothesis/hyp1_hierarchical-topic-aspect-reranking.md`, проверь этот план на совместимость с реальным кодом и только после этого последовательно реализуй его.

Цель — добавить иерархическую тематическую адаптацию keyword reranking перед существующим extractive QA, сохранив текущую систему работоспособной и обратно совместимой.

---

# Критические ограничения

1. **Extractive QA остаётся.**  
   Не заменяй reader генеративной моделью и не добавляй генеративный inference.

2. **Не ломай существующие режимы.**  
   `baseline`, `attention` и существующий `static-keywords` должны продолжить работать.

3. **Новая функциональность должна быть отключаемой.**  
   Предпочтительно реализовать её отдельным режимом или feature flag.

4. **Не используй gold/reference на inference.**  
   Gold допустим только там, где он законно используется на train/dev tuning.

5. **Не требуй от пользователя тематику документа.**  
   Тема определяется автоматически.

6. **Не считай `score_difference < 0` признаком topic.**  
   Существующий `score_difference` трактуется как aspect signal. Topic relevance вычисляется отдельно.

7. **Не используй LLM для названий topic clusters.**  
   Машинное описание кластера — centroid/prototype. Человекочитаемое — representative terms.

8. **Сохрани train/dev/test anti-leakage.**  
   Topic clusters, centroids и candidate terms строятся только на train.  
   Dev используется для выбора параметров.  
   Test — только для финальной оценки.

9. **Не делай большой рефакторинг без необходимости.**  
   Сначала переиспользуй существующие abstractions, encoder, config, serialization и test infrastructure.

10. **Все случайные алгоритмы должны быть детерминируемыми через seed.**

11. **Не меняй публичные API и JSON schemas молча.**  
    Все изменения должны быть backward-compatible либо versioned.

12. **Не внедряй aggressive pruning до проверки Evidence Recall@K.**  
    Первая версия должна прежде всего rerank'ить, а не терять правильные evidence chunks.

---

# Этап 1. Изучи репозиторий

До изменения кода найди и опиши:

- реализацию `WeightedKeyword` или фактический аналог;
- где вычисляются `attention_weight` и `score_difference`;
- где формируется candidate pool;
- где реализован `static-keywords`;
- где находится chunk scoring;
- `minimum_matches`;
- positional/frequency/uniqueness modifiers;
- fallback;
- `AnswerConsensus`;
- train/dev/test split;
- SFFS;
- tuning objective;
- sentence/document encoder;
- document parsing;
- доступность `title`, `abstract`, `headings`;
- JSON/config schemas;
- CLI;
- notebooks/scripts;
- existing tests.

Не предполагай имена модулей. Найди их в коде.

Составь dependency map и отметь, какие компоненты являются public/stable API.

---

# Этап 2. Составь план до изменения кода

Сначала выдай подробный план.

Для каждого шага укажи:

- какие файлы будут изменены;
- какие классы/функции будут добавлены;
- какие существующие API затрагиваются;
- как обеспечивается backward compatibility;
- какие тесты будут добавлены;
- какие artifacts/JSON изменятся;
- какие новые параметры конфигурации нужны;
- какие потенциальные риски есть;
- как новая функция отключается;
- как выполнить rollback;
- какие части гипотезы нужно адаптировать к реальной архитектуре.

Если гипотеза конфликтует с реальным устройством проекта, не делай слепую реализацию.  
Объясни конфликт и выбери минимальное архитектурно корректное решение, сохраняющее смысл гипотезы.

---

# Этап 3. Реализуй поэтапно

## 3.1. Topic model data layer

Добавь минимальные сущности для:

- topic node;
- centroid;
- parent/children;
- representative terms;
- topic term weights;
- schema version;
- serialization/deserialization.

Максимально переиспользуй существующие структуры.

Старые `static-keywords` artifacts должны продолжать загружаться старым путём.

---

## 3.2. Document topic representation

Реализуй тематическое представление:

```text
title + abstract + headings
```

с безопасным deterministic fallback при отсутствии части структуры.

Используй существующий encoder, если он подходит.

Не добавляй новый тяжёлый encoder без явного обоснования.

---

## 3.3. Offline clustering

Только на train:

- вычисли document embeddings;
- кластеризуй документы;
- вычисли leaf centroids;
- построй hierarchy;
- вычисли centroids внутренних узлов;
- сохрани metadata и seed.

Алгоритм выбирай с учётом:

- текущих зависимостей;
- воспроизводимости;
- возможности получить centroids;
- минимальной сложности интеграции.

---

## 3.4. Topic terms

Для каждого topic node:

- собери candidate terms;
- вычисли cluster-vs-rest topic specificity;
- вычисли document support;
- отфильтруй generic terms;
- сохрани top representative terms.

Базовая идея:

\[
T(k,m)=
\log
\frac{P(k|C_m)+\varepsilon}
{P(k|\neg C_m)+\varepsilon}.
\]

Не используй dev/test для формирования vocabulary.

Для parent node не объединяй слова детей механически: учитывай общую поддержку по дочерним кластерам.

---

## 3.5. Hierarchical routing

На inference:

- построй topic representation нового документа;
- вычисли embedding;
- выполни coarse-to-fine routing;
- поддержи top-k / beam;
- используй soft weights;
- реализуй similarity threshold;
- реализуй margin;
- реализуй parent/global fallback;
- не требуй human label.

Новый документ может относиться к нескольким темам.

---

## 3.6. Aspect/topic chunk scoring

Переиспользуй существующий `score_difference` как aspect signal.

Добавь независимый topic signal.

Реализуй формулу, эквивалентную:

\[
R(c)
=
\theta_A \hat A(c)
+
\theta_T \hat T(c|D)
+
\theta_{AT}\hat A(c)\hat T(c|D).
\]

Не складывай ненормализованные шкалы.

Сохрани существующие modifiers или интегрируй новый score консервативно.

Новая схема должна включаться отдельно.

---

## 3.7. Fallback

Реализуй понятную цепочку, адаптированную к реальному коду:

```text
leaf
→ parent
→ broader/global topic
→ existing static-keywords
→ baseline
```

Fallback должен быть диагностируемым.

Добавь причины вроде:

```text
LOW_TOPIC_CONFIDENCE
NO_TOPIC_MATCHES
INCOMPATIBLE_TOPIC_ARTIFACT
NO_RERANKED_CHUNKS
```

или эквивалентные enum/статусы в стиле проекта.

---

# Этап 4. Не ломай текущую tuning-логику

Не переписывай SFFS и текущий objective без необходимости.

Первая версия должна:

- максимально переиспользовать существующую aspect/static tuning;
- строить topic model отдельно;
- на dev подбирать routing/mixing;
- сохранить существующие метрики;
- добавить compute и retrieval metrics.

Topic-specific SFFS считай Phase 2.

---

# Этап 5. Тестирование

## Unit tests

Добавь тесты на:

- document topic representation;
- missing metadata fallback;
- centroid;
- cosine similarity;
- topic specificity;
- support;
- parent/child hierarchy;
- soft routing;
- thresholds;
- margin;
- parent fallback;
- serialization;
- schema version;
- deterministic clustering;
- aspect/topic scoring;
- normalization;
- empty/zero cases;
- отсутствие NaN/inf.

## Regression tests

При выключенной новой функции:

- existing baseline не должен измениться;
- existing static-keywords не должен измениться;
- existing fallback не должен измениться;
- existing consensus не должен сломаться.

Если exact equality невозможна из-за исправленного ранее существовавшего бага — явно зафиксируй это.

## Integration tests

Создай небольшой synthetic corpus как минимум из двух явно разных тематик.

Проверь:

- корректный routing;
- mixed-topic routing;
- OOD fallback;
- aspect + topic interaction;
- отсутствие gold на inference;
- корректную сериализацию/загрузку topic artifact.

## Leakage tests

Проверь, что:

- dev term не появился в train topic dictionary из-за dev;
- test term не влияет на centroid;
- test не влияет на thresholds;
- gold не используется router на inference.

---

# Этап 6. Метрики и диагностика

Добавь измерение:

- Evidence Recall@K;
- QA quality;
- fallback rate;
- harm/win rates, если они уже есть;
- QA chunks/document;
- QA tokens/document;
- `QAChunkRatio`;
- `QATokenRatio`;
- отношение к baseline.

Сохраняй diagnostics для:

- topic nodes;
- representative terms;
- routing;
- chunk scores before/after;
- fallback;
- compute summary.

Не меняй существующую систему отчётов сильнее, чем требуется.

---

# Этап 7. Проверка совместимости

Перед завершением:

1. запусти existing test suite;
2. запусти новые тесты;
3. проверь старые config/artifacts;
4. проверь новый topic artifact;
5. проверь deterministic run;
6. проверь feature disabled;
7. проверь feature enabled;
8. проверь минимум один end-to-end пример;
9. проверь отсутствие test leakage;
10. проверь, что inference не вызывает генеративные модели;
11. проверь, что пользователь не обязан передавать topic;
12. проверь, что old `static-keywords` можно использовать без нового artifact.

Если тест падает, не скрывай это. Исправь причину либо явно опиши несовместимость.

---

# Этап 8. Не реализовывай Phase 2 раньше времени

Следующие идеи считаются отдельными расширениями, если их ещё нет в коде:

- hard-negative discriminativeness;
- полноценный phrase matcher;
- source-span deduplication;
- learned lightweight ranker;
- compute-constrained objective;
- topic-specific SFFS.

Сначала добейся работающей, тестируемой и отключаемой базовой иерархической topic adaptation.

---

# Итоговый отчёт после реализации

Дай отчёт в следующей структуре.

## 1. Что найдено в исходной архитектуре

Карта реальных компонентов и зависимостей.

## 2. План и отклонения от него

Что было запланировано и что пришлось скорректировать.

## 3. Что изменено

По файлам, классам и функциям.

## 4. Почему выбран такой дизайн

Особенно если он отличается от буквального текста гипотезы.

## 5. Backward compatibility

Как гарантируется работа старых режимов и artifacts.

## 6. Tests

Какие тесты добавлены и результаты полного test suite.

## 7. Новые config/CLI параметры

С defaults и описанием.

## 8. Новый artifact

Формат, `schema_version`, пример структуры.

## 9. Diagnostics и metrics

Что теперь сохраняется и считается.

## 10. Known limitations

Что сознательно осталось на Phase 2.

## 11. Как запустить

Точные команды для:

- offline topic tuning;
- inference;
- evaluation;
- regression check.

---

# Самопроверка перед завершением

Проверь:

- [ ] extractive QA сохранён;
- [ ] генеративный inference не добавлен;
- [ ] old `static-keywords` работает;
- [ ] пользователь не задаёт topic;
- [ ] topic clusters имеют centroids;
- [ ] representative terms строятся автоматически;
- [ ] topic hierarchy формируется только на train;
- [ ] routing soft/hierarchical;
- [ ] есть fallback;
- [ ] `score_difference` используется как aspect signal;
- [ ] topic signal независим;
- [ ] нет правила `diff < 0 => topic`;
- [ ] нет train/dev/test leakage;
- [ ] новая функция отключаемая;
- [ ] artifacts versioned;
- [ ] tests проходят;
- [ ] измеряется compute;
- [ ] документация обновлена;
- [ ] feature disabled сохраняет прежнее поведение.

Главный принцип: **сначала понять текущий проект, затем встроить минимально инвазивное улучшение, и только после этого оптимизировать. Не переписывай работающие части без необходимости.**
