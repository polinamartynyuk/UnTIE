# Концептуальное описание алгоритмов извлечения информации и их настройки

Документ описывает **идею и математическую постановку** методов, реализованных в пакете `untie`.
Операционные детали запуска, CLI-параметры и структура артефактов приведены в
[`keyword-tuning-algorithm.md`](keyword-tuning-algorithm.md).

---

## 1. Постановка задачи

Рассматривается **extractive question answering (извлекающий ответ на вопрос)**:
по документу \(D\) и вопросу \(q\) требуется найти текстовый фрагмент (span),
максимально соответствующий эталонному ответу \(y\) по тематическому аспекту
(например, «какая задача решена в работе»).

Документ предварительно разбивается на множество перекрывающихся **чанков**
(context windows) \(\{c_1, \ldots, c_n\}\). Для каждого чанка transformer-модель
extractive QA возвращает кандидатный ответ \(a_i\) и confidence \(p_i\).
Итоговый ответ выбирается агрегацией или переранжированием кандидатов.

Система поддерживает три режима inference:

| Режим | Ключевые слова | Эталон при inference |
|---|---|---|
| `baseline` | не используются | не нужен |
| `attention` | строятся per-document из attention QA | нужен |
| `static-keywords` | заранее настроенный глобальный словарь | не нужен |

---

## 2. Сегментация документа

Текст \(D\) разбивается на предложения, затем формируются чанки фиксированного
размера в токенах (`chunk_max_tokens`) с перекрытием (`overlap_tokens`).
Перекрытие снижает риск потери ответа на границе двух окон.

Для английского корпуса по умолчанию используются окна 384/50 токенов,
для русского — 128/24. Ограничение: `overlap_tokens < chunk_max_tokens / 2`.

**Обоснование:** extractive QA-модели имеют ограниченный контекст (512
subword-токенов); разбиение позволяет обработать длинные научные тексты без
потери локальной релевантности.

---

## 3. Базовый extractive QA

Для каждого чанка \(c_i\) модель \(f_\theta\) решает задачу span extraction:

\[
(a_i, p_i, s_i, e_i) = f_\theta(q, c_i),
\]

где \(a_i\) — извлечённый текст, \(p_i\) — confidence модели, \((s_i, e_i)\) —
позиции span в чанке.

### 3.1. Baseline-агрегация

В режиме `baseline` все чанки участвуют в QA без фильтрации. Кандидаты
\(\{a_i\}\) кодируются sentence encoder'ом; строится матрица косинусных
сходств \(S_{ij} = \cos(e_i, e_j)\). Итоговый ответ — тот, чей embedding
имеет **максимальное среднее сходство** с остальными:

\[
i^\* = \arg\max_i \frac{1}{n-1}\sum_{j \neq i} S_{ij}.
\]

**Обоснование:** при extractive QA множество чанков часто содержит повторяющиеся
или парафразные ответы; выбор «центрального» кластера в embedding-пространстве
устойчив к локальным ошибкам отдельных окон.

---

## 4. Ключевые слова и переранжирование чанков

### 4.1. Представление ключевого слова

Ключевое слово \(k\) описывается структурой `WeightedKeyword`:

- `word` — surface form;
- `lemma`, `stem` — морфологические формы для matching;
- `attention_weight` \(w_k^{\mathrm{att}}\) — salience из attention QA;
- `score_difference` \(w_k^{\mathrm{diff}}\) — контрастная семантическая
  характеристика (см. §4.2).

### 4.2. Контрастное scoring ключей

Для слова \(k\) вычисляется **score difference** на основе sentence encoder:

\[
\Delta_k = \cos(e_k, e^{+}) - \cos(e_k, e^{-}),
\]

где \(e^{+}\) — embedding описания аспекта (вопроса или названия поля).

**Режим inference (`attention`):** \(e^{-}\) — embedding одного эталонного ответа
\(y\). Слово сохраняется, если \(\Delta_k > 0\).

**Режим сбора evidence (tuning):** \(e^{-} = \max_j \cos(e_k, e_{y^{(j)}})\) по
всем gold-ответам документа. Слово сохраняется, если семантически ближе к
аспекту, чем к **любому** эталону:

\[
\Delta_k = \cos(e_k, e_{\mathrm{aspect}}) - \max_j \cos(e_k, e_{y^{(j)}}) > 0.
\]

**Обоснование:** эталонный ответ часто совпадает с целевым span; без контраста
attention выделяет слова самого ответа, а не характерные маркеры аспекта в
тексте документа.

### 4.3. Скоринг чанка по ключевым словам

Пусть \(K(c)\) — множество ключей, найденных в чанке \(c\) по regex-сопоставлению
lemma/stem. Если \(|K(c)| < \texttt{minimum\_matches}\), чанк отбрасывается.

Комбинированный вес ключа (параметр `weight_ratio` \(\lambda\)):

\[
\omega_k = \lambda \cdot w_k^{\mathrm{att}} + (1 - \lambda) \cdot w_k^{\mathrm{diff}}.
\]

Базовый score:

\[
B(c) = \sum_{k \in K(c)} \omega_k.
\]

Позиционный, частотный и уникальный множители:

\[
\bar{\pi}(c) = \frac{1}{|M|}\sum_{m \in M}\left(1 - \frac{\mathrm{pos}(m)}{|c|}\right),
\qquad
P(c) = 1 + \alpha_\pi \cdot \bar{\pi}(c),
\]

\[
F(c) = 1 + \alpha_f \cdot \ln(1 + |M|),
\qquad
U(c) = 1 + \frac{|K(c)|}{|K(D)|} \cdot 0.5,
\]

где \(M\) — множество совпадений в чанке, \(|K(D)|\) — ключи, встречающиеся
**хотя бы в одном** чанке документа (нормализация только по присутствующим
термам, чтобы отсутствующие ключи не «размывали» score).

Итоговый score чанка:

\[
S(c) = B(c) \cdot P(c) \cdot F(c) \cdot U(c).
\]

По умолчанию \(\alpha_\pi = 0.3\), \(\alpha_f = 0.7\).

**Обоснование:** ранняя позиция и повторяемость терма часто коррелируют с
тематической центральностью фрагмента; uniqueness bonus поощряет чанки,
покрывающие больше различных ключей.

### 4.4. Fallback

Если после фильтрации не осталось чанков с ключами, pipeline возвращает
**baseline-ответ** по всем чанкам и помечает `keyword_fallback = True`.
Это предотвращает пустой результат, но может не дать выигрыша от keywords.

---

## 5. Выбор финального ответа после reranking

Для каждого чанка с \(S(c) > 0\) повторно вызывается QA; ответы сортируются
по `chunk_score`. Финальный span выбирается одной из стратегий **AnswerConsensus**.

### 5.1. Кластеризация ответов

Embeddings ответов группируются **жадным** алгоритмом: ответ \(i\) попадает
в кластер с \(j\), если \(\cos(e_i, e_j) \geq \tau\) (default \(\tau = 0.75\)).

Для кластера \(C\) вычисляются метрики:

\[
\bar{s}_C = \frac{1}{|C|}\sum_{i \in C} s_i^{\mathrm{chunk}},
\qquad
W_C = |C| \cdot \bar{s}_C,
\qquad
H_C = \frac{2}{|C|(|C|-1)} \sum_{i<j,\, i,j \in C} \cos(e_i, e_j).
\]

Выбор кластера (`cluster_strategy`):

- `highest_avg_score` → \(\arg\max \bar{s}_C\);
- `weighted_score` → \(\arg\max W_C\);
- `highest_cohesion` → \(\arg\max H_C\).

Внутри выбранного кластера (`answer_strategy`):

- `highest_chunk_score` → max \(s_i^{\mathrm{chunk}}\);
- `highest_similarity` → max средней similarity с соседями;
- `combined_score` → \(0.5 \cdot s_i^{\mathrm{chunk}} + 0.5 \cdot \bar{\cos}_i\).

**Обоснование:** reranking смещает QA к «keyword-rich» чанкам, но среди их
ответов возможны синонимичные варианты; кластерный консенсус снижает дисперсию
выбора.

---

## 6. Режим attention (per-document)

Pipeline `AttentionRerankingPipeline`:

1. baseline QA по всем чанкам;
2. **валидация** кандидатов относительно эталона \(y\):
   \(\cos(e_{a_i}, e_y) \geq 0.9\) (strict), иначе порог 0.7;
3. attention extraction из пар \((q, c_i)\) для валидных чанков;
4. IDF-фильтрация слов внутри валидных чанков (удаление «равномерно
   распределённых» термов);
5. contrastive scoring (§4.2);
6. reranking и consensus (§4–5).

Attention вес слова \(k\) в контексте чанка — среднее attention по subword-токенам,
восстановленным в слово:

\[
w_k^{\mathrm{att}} = \frac{1}{|T_k|}\sum_{t \in T_k} A_t,
\]

где \(T_k\) — subword-токены слова \(k\), \(A_t\) — усреднённые attention-веса
(по слоям, головам внимания и query-позициям) на context-токены пары \((q, c)\).
В ranking попадают слова длиной \(> 1\) символа, top-\(K\) по весу (default
\(K = 100\)).

**Ограничение:** режим требует эталон при inference и не масштабируется на
batch без разметки; используется как источник сигналов для offline-настройки.

---

## 7. Метрики качества извлечения

Качество предсказания \(\hat{y}\) относительно эталона (или множества эталонов
\(\{y^{(j)}\}\)) оценивается composite-метрикой.

### 7.1. Базовые метрики

- **char F1** — посимвольный F1 после lowercasing; при нескольких эталонах
  берётся maximum;
- **token F1** — SQuAD token-level F1, диапазон \([0, 100]\);
- **ROUGE-L F1** — longest common subsequence F1;
- **BERTScore F1** — опционально, semantic similarity на contextualized
  embeddings.

### 7.2. Composite quality

\[
Q(\hat{y}, y) = \sum_{m \in \mathcal{M}} \tilde{w}_m \cdot \tilde{m}(\hat{y}, y),
\]

где \(\tilde{m}\) — нормализованные значения метрик (\(\mathrm{token\_f1}\)
делится на 100), \(\tilde{w}_m\) — веса, перенормированные до суммы 1.

При настройке по умолчанию (`tuning-exact-match`) используется пресет
char/token F1 = 0.4, ROUGE-L = 0.2; BERTScore в objective tuning не
участвует.

**Обоснование:** комбинация lexical (char/token) и sequence-level (ROUGE-L)
метрик устойчива к морфологическим и перефразировочным вариациям в
научных текстах.

---

## 8. Концепция настройки static-keywords

**Цель настройки:** построить **один глобальный словарь** ключевых слов и
выбрать стратегию reranking/consensus, максимизирующие качество extractive QA
на коллекции документов **без** эталона на этапе inference.

### 8.1. Разделение данных и предотвращение leakage

Документы детерминированно делятся train/dev/test (60/20/20) по hash от
`(seed, doc_id)`:

- **train** — единственный источник кандидатов в pool;
- **dev** — поиск словаря, SFFS, stability selection;
- **test** — однократная финальная оценка.

Это исключает попадание dev/test термов в candidate pool и использование test
при выборе гиперпараметров.

### 8.2. Двухфазная архитектура

**Фаза 1 (дорогая, кэшируемая):** для каждого документа один раз выполняются
QA, attention mining (только train), сохранение chunk-answers и baseline quality.

**Фаза 2 (дешёвая):** перебор подмножеств словаря использует cached QA-ответы
и reranking без повторного inference transformer-модели.

### 8.3. Формирование candidate pool

Pool агрегируется **только из train evidence**, извлечённого из текста
документов (не из gold labels). Для каждого терма:

- медиана `attention_weight` и `score_difference` по поддерживающим документам;
- фильтр `document_support ≥ min_document_support`;
- фильтр `chunk_support_rate > 0` (терм должен матчить хотя бы один чанк);
- удаление stop words и generic-глаголов.

Опциональное `--enrich-train-references` добавляет n-grams из train gold —
не рекомендуется, т.к. подмешивает разметку, а не текстовые сигналы.

### 8.4. Prescreening

Перед combinatorial search каждый терм оценивается **в одиночку** на full dev.
Остаются top-k по `mean_gain_active`, `activation_rate`, `objective`; отсекаются
термы с высоким `harm_rate` или низкой predicted activation.

**Обоснование:** сокращает пространство поиска SFFS, сохраняя бюджет на
комбинации из 2–3 семантически комплементарных ключей.

### 8.5. Multi-fidelity SFFS

Для каждой из 27 комбинаций стратегий (3 × 3 × 3, см. §4.3–5) выполняется
**Sequential Forward Floating Selection**:

- forward: жадное добавление терма с максимальным приростом objective;
- successive halving: оценка на панелях 25% → 50% → 100% dev-подвыборки;
- backward: удаление ставших вредными термов.

Процесс повторяется `stability_runs` раз на 80%-подвыборках dev; затем
применяется политика финального отбора (`union`, `best_run`, frequency-based
и др.) с **non-empty rescue guard**.

---

## 9. Целевая функция настройки

Для подмножества ключей \(K\) и документа \(d\) определяется приращение качества:

\[
\delta_d(K) = Q(\hat{y}_d(K), y_d) - Q(\hat{y}_d(\emptyset), y_d),
\]

где \(\hat{y}_d(\emptyset)\) — baseline prediction без keywords.

Дополнительные агрегаты на dev:

\[
\overline{\Delta}_{\mathrm{active}} = \frac{1}{|D_a|}\sum_{d \in D_a} \delta_d,
\qquad
r_{\mathrm{fall}} = \frac{|\{d : \mathrm{fallback}_d\}|}{|D|},
\qquad
r_{\mathrm{act}} = 1 - r_{\mathrm{fall}},
\]

\[
r_{\mathrm{harm}} = \frac{|\{d : \delta_d < -\varepsilon\}|}{|D|},
\qquad
r_{\mathrm{win}} = \frac{|\{d : \delta_d > \varepsilon\}|}{|D|},
\]

где \(D_a\) — документы без fallback, \(\varepsilon = \texttt{harm\_threshold}\)
(default 0.01).

Базовый gain (при `use_conditional_gain`):

\[
G = \begin{cases}
\overline{\Delta}_{\mathrm{active}}, & |D_a| > 0,\\
\overline{\Delta}, & \text{иначе.}
\end{cases}
\]

Objective для непустого \(K\):

\[
\begin{aligned}
J(K) =\;& G
+ \beta_{\mathrm{act}} \cdot r_{\mathrm{act}}
+ \beta_{\mathrm{win}} \cdot r_{\mathrm{win}}
+ \beta_{\mathrm{conf}} \cdot \mathrm{LB}_{\mathrm{boot}} \\
& - \beta_{\mathrm{down}} \cdot \overline{\max(0, -\delta)}
- \beta_{\mathrm{harm}} \cdot r_{\mathrm{harm}}
- \beta_{\mathrm{fb}} \cdot r_{\mathrm{fb}}^{\mathrm{harm}}
- \beta_{\mathrm{idle}} \cdot r_{\mathrm{fall}}
- \beta_{\mathrm{size}} \cdot |K|,
\end{aligned}
\]

где:

- \(r_{\mathrm{fb}}^{\mathrm{harm}}\) — доля документов с fallback **и**
  \(\delta_d < 0\) (вредный fallback);
- \(\mathrm{LB}_{\mathrm{boot}}\) — нижняя граница bootstrap по active deltas;
- при \(r_{\mathrm{act}} < r_{\mathrm{act}}^{\min}\) objective = \(-\infty\)
  (hard gate против «редких безопасных» словарей).

Default коэффициенты CLI: \(\beta_{\mathrm{down}}=0.75\), \(\beta_{\mathrm{harm}}=0.5\),
\(\beta_{\mathrm{fb}}=0.1\), \(\beta_{\mathrm{idle}}=0.05\), \(\beta_{\mathrm{size}}=0.002\),
\(\beta_{\mathrm{conf}}=0.25\), \(\beta_{\mathrm{act}}=0.15\), \(\beta_{\mathrm{win}}=0.10\),
\(r_{\mathrm{act}}^{\min}=0.20\).

**Обоснование компонентов objective:**

| Компонент | Смысл |
|---|---|
| \(G\) / conditional gain | оптимизировать реальное улучшение там, где keywords активны |
| \(r_{\mathrm{act}}\) | поощрять словари, часто матчащие чанки |
| \(r_{\mathrm{win}}\) | поощрять документы с заметным выигрышем |
| \(\mathrm{LB}_{\mathrm{boot}}\) | штрафовать нестабильные подмножества |
| downside / harm | ограничивать регрессии относительно baseline |
| idle fallback penalty | штрафовать словари, не влияющие на reranking |
| size penalty | Occam's razor — предпочитать компактные словари |

### 9.1. Критерий release

После выбора на dev финальная конфигурация **один раз** оценивается на test.
Флаг `release_recommended` устанавливается, если:

\[
\overline{\Delta}_{\mathrm{test}} > 0,\quad
r_{\mathrm{act}}^{\mathrm{test}} \geq r_{\mathrm{act}}^{\min},\quad
r_{\mathrm{win}}^{\mathrm{test}} \geq r_{\mathrm{loss}}^{\mathrm{test}},\quad
r_{\mathrm{harm}}^{\mathrm{test}} \leq \texttt{harm\_cap},\quad
\mathrm{LB}_{\mathrm{boot}}^{\mathrm{test}} \geq 0.
\]

Это автоматический gate качества, а не гарантия production-пригодности.

---

## 10. Связь режимов inference и настройки

```text
                    ┌─────────────────────────────────────┐
                    │  Offline tuning (train/dev/test)    │
                    │  attention evidence → global pool   │
                    │  SFFS + stability → tuned JSON      │
                    └─────────────────┬───────────────────┘
                                      │ keywords + strategy
                                      ▼
Документ + вопрос ──► static-keywords pipeline ──► extractive answer
                      (без эталона)
```

Режим `attention` использует ту же математику reranking (§4–5), но ключи
строятся **online** с эталоном. Режим `static-keywords` переносит результат
offline-оптимизации в production: один словарь на аспект, один проход QA по
отфильтрованным чанкам.

---

## 11. Ограничения модели

1. **Regex matching** lemma/stem ограниченно работает для дефисных и
   многословных термов.
2. **Extractive QA** не генерирует текст вне документа; качество bounded
   наличием span в чанках.
3. **Attention-based mining** зависит от архитектуры QA-модели (наличие
   `output_attentions`).
4. **Global словарь** не адаптируется к жанру отдельного документа; выигрыш
   возможен, когда аспект маркируется устойчивой лексикой корпуса.

---

## 12. Связанные документы

- [`keyword-tuning-algorithm.md`](keyword-tuning-algorithm.md) — операционное
  описание pipeline настройки, CLI, артефакты, ноутбуки.
- [`component-inventory.md`](component-inventory.md) — карта модулей `untie`.
