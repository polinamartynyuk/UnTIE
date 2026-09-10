# Средства выявления релевантных чанков для extractive QA

Документ описывает **все способы отбора или ранжирования текстовых фрагментов (чанков)**, которые в проекте UnTIE **заявлены** как средства найти чанк, из которого extractive QA должна извлечь ответ. Это каталог идей и реализованных слоёв, а не инструкция запуска.

Связанные материалы: постановка extractive QA и формулы keyword-score — [`information-extraction-concepts.md`](information-extraction-concepts.md); иерархический topic-слой — [`hierarchical-topic-reranking.md`](hierarchical-topic-reranking.md); настройка словарей — [`keyword-tuning-algorithm.md`](keyword-tuning-algorithm.md); подготовка SciREX и метки positive/HN — [`metodika-podgotovki-dannyh-scirex.md`](metodika-podgotovki-dannyh-scirex.md). Инвентарь модулей: [`component-inventory.md`](component-inventory.md).

---

## 0. Общая рамка

Документ \(D\) режется на перекрывающиеся чанки \(\{c_i\}\). Frozen extractive QA (RoBERTa / аналог профиля) по паре \((q, c_i)\) возвращает span и confidence. Релевантность чанка в проекте понимается по-разному:

| Роль | Что делает средство | Когда применяется |
|---|---|---|
| **Фильтр** | отбрасывает чанки до QA | hard keyword / topic-term gate |
| **Ранжирование** | оставляет все чанки, меняет порядок или вес | soft rerank, CE, смесь каналов |
| **Shortlist ответов** | QA на всех чанках, агрегатор видит только top-k | CE topic shortlist |
| **Выбор ответа** | не скорит чанк напрямую, выбирает span среди кандидатов | baseline consensus, AnswerConsensus |
| **Разметка для поиска признаков** | строит positive / hard-negative, но сама не inference | BM25+FN, QAConsistency |

Gold при **inference** запрещён, кроме режима `attention` (он сам по себе требует эталон и служит источником сигналов для offline-настройки). Gold на train/dev законен для tuning словарей, весов смеси и порогов.

Ниже каждое средство: идея, заявленный источник, как скорится чанк, статус в репозитории.

---

## 1. Базовый extractive QA по всем чанкам

**Заявлено в:** ядро пайплайна; `docs/information-extraction-concepts.md` §3; режим CLI `baseline`.

**Идея.** Не отбирать чанки заранее. QA вызывается на каждом окне. Итоговый ответ — «центральный» среди извлечённых span в embedding-пространстве.

**Скоринг чанка.** Явного chunk score нет. Кандидаты \(\{a_i\}\) кодируются sentence encoder; выбирается ответ с максимальным средним косинусным сходством к остальным (`AnswerAggregator`, исторически centroid).

**Статус.** Production. Контроль для всех последующих вмешательств. Fallback, если keyword/topic-фильтр ничего не оставил (`keyword_fallback = True`).

---

## 2. Keyword reranking: `WeightedKeyword` и `score_chunks`

**Заявлено в:** `information-extraction-concepts.md` §4; реализация `untie/ranking.py`.

Это **общий механизм**, на который садятся режимы `attention`, `static-keywords` и иерархические словари.

Ключ \(k\) несёт:

- surface / lemma / stem для matching;
- `attention_weight` — salience из attention QA;
- `score_difference` (`diff`) — контраст «аспект vs эталон».

Чанк проходит, если число совпавших ключей \(\ge\) `minimum_matches`. Иначе в hard-режиме отбрасывается.

Комбинированный вес ключа с `weight_ratio` \(\lambda\):

\[
\omega_k = \lambda\, w_k^{\mathrm{att}} + (1-\lambda)\, w_k^{\mathrm{diff}}.
\]

Стратегии \(\lambda\): `only_weight` (1), `only_score_diff` (0), `equal_weight_score_diff` (0.5).

Базовый score — сумма \(\omega_k\) по совпадениям, затем множители:

- **позиция** — раньше в чанке → выше;
- **частота** — \(\ln(1+|M|)\);
- **уникальность** — доля различных ключей чанка среди ключей, встретившихся хоть в одном чанке документа.

Итого \(S(c) = B(c)\cdot P(c)\cdot F(c)\cdot U(c)\). QA затем идёт по чанкам с \(S(c)>0\); финальный span — `AnswerConsensus` (кластеры ответов по cosine, стратегии `highest_avg_score` / `weighted_score` / `highest_cohesion` и внутрикластерный выбор).

Режим **`keep_unmatched`** (soft): чанки без ключей не дропаются, получают нулевой или нижний score и участвуют в ранжировании.

**Статус.** Production-ядро. Все словарные методы ниже — разные источники \(K\) и \(\omega_k\), не разные формулы matching.

---

## 3. Per-document attention-keywords

**Заявлено в:** `information-extraction-concepts.md` §6; режим `attention`; `untie/attention.py`, `AttentionRerankingPipeline`.

**Идея.** Слова, на которые QA-модель смотрит в парах \((q, c_i)\), маркируют место ответа лучше, чем общая тематика документа.

**Как строится словарь документа.**

1. Baseline QA по всем чанкам.
2. Валидация span относительно эталона (cosine 0.9 / запасной 0.7) — **нужен gold**.
3. Attention по subword-токенам, усреднение в слово, top-\(K\) (обычно 100).
4. IDF-фильтрация термов, равномерно размазанных по валидным чанкам.
5. Contrastive `score_difference` (§4).
6. `score_chunks` + consensus.

**Статус.** Production как режим, но **не для разметки без эталона**. Используется как шахта признаков для offline-сборки глобальных и кластерных словарей.

---

## 4. Контрастный сигнал `score_difference` (aspect vs gold)

**Заявлено в:** `information-extraction-concepts.md` §4.2; гипотеза hyp1 («не считать `diff < 0` признаком topic»).

\[
\Delta_k = \cos(e_k, e^{+}) - \cos(e_k, e^{-}),
\]

где \(e^{+}\) — embedding аспекта/вопроса. На inference в `attention` \(e^{-}\) — эталон документа; при tuning — max по gold-ответам. Сохраняются ключи с \(\Delta_k > 0\): ближе к формулировке аспекта, чем к самому ответу.

**Смысл для чанка.** Diff не скорит чанк отдельно: он входит в \(\omega_k\) и тем самым в \(S(c)\). Заявлен как **aspect signal**, ортогональный topic-специфичности.

**Статус.** Production. Ablation: словари `only_weight` vs `equal_weight_score_diff`; refit hierarchical keywords по умолчанию без gold-contrast (`scripts/10_Refit_hierarchical_keywords.py`).

---

## 5. Глобальный static-keywords (SFFS)

**Заявлено в:** `information-extraction-concepts.md` §8; `keyword-tuning-algorithm.md`; режим `static-keywords`.

**Идея.** Один словарь на аспект, собранный на train (attention evidence + contrast), отобранный Sequential Forward Floating Selection на train/dev **без** gold на test/inference.

**Скоринг чанка.** Тот же `score_chunks`. Hard-фильтр по `minimum_matches`. Нет совпадений → fallback на baseline.

**Статус.** Production. В гипотезах 2–4 заявлен как слабый из-за тематической неоднородности корпуса: глобальные термы «размываются».

---

## 6. Иерархическая тематическая адаптация

**Заявлено в:** `hypothesis/hyp1_hierarchical-topic-aspect-reranking.md`; `docs/hierarchical-topic-reranking.md`; режим `hierarchical-static-keywords`.

**Идея.** У ключа два независимых сигнала: aspect (`score_difference`) и topic (специфичность терма для кластера документа). Тема документа определяется автоматически (centroid nearest leaf), без метки пользователя и без LLM.

### 6.1. Маршрутизация документа

Train-only иерархия (KMeans / дерево листьев). Представление: `title + abstract + headings`, иначе первые непустые абзацы. Dev выбирает параметры routing и смешивания; test только оценка.

### 6.2. Topic-term gate и смесь A/T

Чанк получает aspect-компоненту \(A\) (как в §2) и topic-компоненту \(T\) (сглаженная cluster-vs-rest log-специфичность × document support). После min-max:

\[
\theta_A A + \theta_T T + \theta_{\mathrm{int}} A\cdot T.
\]

По умолчанию чанки без routed `representative_terms` **исключаются** из QA (`filter_unmatched_chunks`). `max_qa_chunks=0` — не резать дальше, пока на dev не проверен Evidence Recall@K.

Fallback: leaf → предок → root → static-keywords → baseline.

### 6.3. Cluster-specific SFFS-словари

На листе — `keyword_profile`, обученный SFFS только на train-документах этого кластера. Inference: nearest leaf → `score_chunks` этим словарём (hard filter). Нет профиля / нет совпадений → подъём к предку, затем глобальный словарь, затем baseline.

**Статус.** Production-экспериментальный режим в `untie.topics` / CLI. Отключается вместе с режимом.

---

## 7. Soft rerank (все чанки отвечаются)

**Заявлено в:** `docs/hierarchical-topic-reranking.md`; `scripts/11_Evaluate_hierarchical_test.py --soft-rerank`; `untie/pipelines.py` (`keep_unmatched=True`).

**Идея.** Hard-drop по словарю теряет evidence. QA на **всех** чанках; ключевые слова, сходство с title/вопросом и routed representative terms только **ранжируют**, какой ответ оставит consensus (`weighted_score` + `combined_score`).

**Скоринг.** `score_chunks(..., keep_unmatched=True)` плюс каналы `chunk_query_similarities` (cosine чанка к запросу). Несовпавшие чанки не обнуляют пул QA.

**Статус.** Реализовано как оценка, пишет отдельный каталог, не затирает hard-filter evaluation.

---

## 8. Cross-encoder ранжирование чанков

**Заявлено в:** `untie/cross_encoder_ranking.py`; `scripts/12_Evaluate_cross_encoder.py`; упоминание в hierarchical-доке.

**Идея.** Попарный MiniLM (`cross-encoder/ms-marco-MiniLM-L6-v2`) скорит \((query, chunk)\). Query — вопрос и/или title-представление документа. QA по-прежнему на всех чанках; CE задаёт `chunk_score`.

**Смесь** со static-словарём (`keep_unmatched=True`):

\[
s = w_{\mathrm{CE}}\,\mathrm{norm}(s_{\mathrm{CE}}) + w_{\mathrm{kw}}\,\mathrm{norm}(s_{\mathrm{kw}}).
\]

Сетка весов на **dev** gold char-F1, на test заморожена. Test gold в ranking не входит. Cluster SFFS-словари **не** используются.

**Статус.** Изолированный эксперимент (`artifacts/hierarchical_topic_optimized/en/cross_encoder/`).

---

## 9. CE shortlist с topic-conditioned query

**Заявлено в:** `untie/ce_topic_shortlist.py`; `scripts/13_Evaluate_ce_topic_shortlist.py`.

**Идея.** Frozen topic-route **не** подставляет SFFS-словарь, а расширяет CE-запрос leaf `representative_terms`. QA на всех чанках; CE выбирает top-\(k\) ответов для baseline `AnswerAggregator`. \(k \in \{8,12,16,\mathrm{all}\}\) на dev.

**Статус.** Изолированный эксперимент (`ce_topic_query/`). Shortlist ответов, не hard-drop чанков.

---

## 10. KeyBERT как тематический prior чанка

**Заявлено в:** `hypothesis/hyp3_prompt3.md`; пакет `experiments/keybert_qa_integration/`.

**Идея.** Attention — task-specific QA-сигнал; KeyBERT — дешёвый тематический prior. KeyBERT **не** заменяет attention-keywords.

Заявленные режимы сравнения: baseline; QA+attention; QA+attention+diff; KeyBERT-only (контроль); attention+KeyBERT; attention+diff+KeyBERT.

**Скоринг чанка** (`keybert_chunk_scores`): cosine embedding чанка к KeyBERT-фразам документа — max / mean Top-N / взвешенная сумма по KeyBERT relevance. Смесь после нормализации:

\[
\alpha\cdot s_{\mathrm{att}} + \beta\cdot s_{\mathrm{diff}} + \gamma\cdot s_{\mathrm{keybert}}
\]

(или с QA confidence). Веса на validation.

Дополнительно в hyp3: кластеры KeyBERT-фраз внутри документа и привязка attention-слов к ближайшему кластеру — вес кластера для выбора чанков.

Hard negatives в этом контуре: чанки с высоким KeyBERT-score, но без лексического gold span (`experiments/keybert_qa_integration/hard_negatives.py`) — для анализа, не как production filter.

**Статус.** Изолированный эксперимент. Production `baseline` / `attention` / `static-keywords` не меняет.

---

## 11. Тематическая кластеризация документов (KeyBERT / hybrid) + cluster keywords

**Заявлено в:** `hypothesis/hyp2_prompt2.md`; `experiments/keybert_topic_clustering/`.

**Идея.** Кластеры по KeyBERT-keyphrases, по `title+abstract` или hybrid embedding устойчивее, чем один глобальный словарь. Downstream: внутри кластера обучить attention / attention+diff словарь и локализовать QA-чанки.

Гипотезы H4–H7: cluster-specific attention лучше отсутствия rerank; +diff лучше; лучше глобального static-keywords; KeyBERT-кластеры лучше чистого document embedding.

**Скоринг чанка на inference.** Документ → кластер → словарь кластера → `score_chunks` (тот же механизм §2).

**Статус.** Изолированный эксперимент; выбранная конфигурация кластеризации переиспользуется в hyp4 и в hierarchical topic artifact.

---

## 12. Кластерно-дискриминативные QA-ключевые слова (positive vs hard-negative)

**Заявлено в:** `hypothesis/hyp4_prompt4.md`; `experiments/cluster_discriminative_keywords/`.

**Идея.** Не «слова темы», а **слова, которые внутри кластера отличают правильный QA-чанк от похожего неправильного**.

Pipeline (заявленный):

```text
train → кластеры → для каждого кластера:
  positive chunks + hard negative chunks
  → кандидаты (attention, diff, question-KeyBERT, topic)
  → H_m(k) = log( (P(k|C+) + ε) / (P(k|C_hard−) + ε) )
  → support + shrinkage к глобальному весу
  → cluster-specific QA dictionary
```

Inference: кластер документа → словарь → оценка/фильтрация чанков → RoBERTa QA → confidence + consensus.

**Статус.** Изолированный эксперимент. Качество меток Pos/HN критично; поэтому отдельно заявлена SciREX Phase 1 (следующие разделы).

---

## 13. Лексический gold-supporting чанк (предобработка)

**Заявлено в:** `hypothesis/scirex_aspect_qa_preprocessing_prompt.md`; `docs/scirex-aspect-qa-preprocessing.md`.

**Идея.** Эталон SciREX — списки сущностей, не span. Чанк **подтверждающий**, если пересекается с лексическим вхождением extractable gold вне блокируемой библиографии (режим E0, `identity_key`).

Это **не** inference-ранжирование и **не** метка «QA угадает». Это улика текста. Чанк с gold, на котором QA молчит, в Phase 1 **не** hard-negative (`gold_supporting_robust_negative` / `evidence_positive_qa_failed`).

**Статус.** Готовые таблицы `datasets/scirex_aspect_qa_preprocessing{,_dev,_test}/`.

---

## 14. BM25 + adaptive false-negative filter (mHFN, адаптация)

**Заявлено в:** `hypothesis/phase1_positive_hard_negative_experiment_scirex.md`; `experiments/phase1_scirex/`; методика §6.

**Идея** (из Shen et al. EMNLP 2022, не полная реплика): трудный отрицательный пассажир — верх BM25, не размеченный как positive; из отрицательных выкидываются те, что в том же K-means кластере, что известный positive (подозрение на неполноту разметки).

Внутри **одного документа**:

1. Positive = чанк, на котором frozen QA извлекла extractable gold (`gold_span_containment`).
2. Пул HN: QA не извлекла gold **и** нет лексического gold.
3. BM25(вопрос, текст чанка), \(k_1=1.5\), \(b=0.75\), top-5.
4. Passage embeddings (C0, без вопроса) → K-means + Gap statistic → hard assignment.
5. Тот же кластер, что positive → `false_negative_candidate`, не HN.
6. Gumbel-Softmax — только абляция, без temperature decay (кодировщик заморожен).

**Скоринг для inference не заявлен.** Это **построение выборки** для последующего поиска признаков (§12, следующий этап SciREX).

**Статус.** Выполнено на официальном SciREX train/dev. TEST не гонялся.

---

## 15. Многовопросная устойчивость (QAConsistency)

**Заявлено в:** расширение Phase 1; `docs/scirex-phase1-multiquestion.md`; `experiments/phase1_multiquestion/`.

**Идея.** Один канонический вопрос смешивает «модель стабильно находит ответ» и «повезло с формулировкой». На тех же чанках и том же gold — 3 или 5 фиксированных парафраз (без фактов документа).

\[
\mathrm{QAConsistency}(c) = N_{\mathrm{correct}}(c) / N_{\mathrm{questions}}.
\]

Порог \(t\) только на DEV: Q5 \(t=0.6\), Q3 \(t=2/3\), Q0 \(t=1\).

| Метка | Условие | Роль для поиска признаков |
|---|---|---|
| `robust_positive` | cons \(\ge t\) | целевой релевантный чанк |
| `question_sensitive` | \(0 <\) cons \(< t\) | успех зависит от парафразы |
| `hard_negative` | cons \(= 0\), не gold, top-5 BM25 после FN | трудный нерелевантный |
| `gold_supporting_robust_negative` | cons \(= 0\), есть gold | не HN |

BM25 агрегируется по вопросам (`mean_normalized`; диагностика RRF / mean rank). Правило «хотя бы один вопрос верен» — только диагностика.

**Статус.** Разметка готова: `artifacts/phase1_multiquestion/official_train_q5` (поиск признаков) и `official_dev_q5` (пороги признаков). TEST не размечать, пока правило признаков не заморожено.

---

## 16. Question-conditioned отбор ответов (не чанков)

**Заявлено в:** `untie/answer_selection.py`; `experiments/answer_aggregation/`.

**Идея.** После QA слить почти дубликаты span и отфильтровать ответы, далёкие от **текста вопроса** (квантиль cosine), без поле-специфичного лексикона («task-like words»).

По умолчанию в ядре **выключено** (исторический centroid всех ответов). Эксперимент включает merge + query filter изолированно.

Это средство **выбора span среди уже извлечённых ответов**, не предварительного отбора чанков. В каталог попало, потому что заявленное улучшение «какого фрагмента ответа оставить» часто путают с chunk retrieval.

**Статус.** Опции `AnswerAggregator`; отдельный evaluation-контур.

---

## 17. Прочие заявленные или упомянутые сигналы

Эти пункты **заявлены как кандидаты признаков** будущего поиска на SciREX Phase 1 или встречаются как legacy, но не являются отдельным inference-режимом ядра.

| Сигнал | Где заявлен | Замечание |
|---|---|---|
| N-граммы и keyphrases | Phase 1 spec §1 | кандидаты признаков, не готовый scorer |
| Attention относительно answer span | Phase 1 spec §1 | не вынесен в production ranking |
| Question-aware KeyBERT | hyp4, Phase 1 | источник кандидатов словаря, не gate |
| Позиция / структура секции | preprocessing + keyword modifiers | секция уже в таблицах чанков SciREX |
| Embedding similarity чанк–вопрос | `chunk_query_similarities`, soft rerank, KeyBERT | канал смеси |
| Dense retrieval bi-encoder (полный mHFN) | Phase 1 spec явно **не** воспроизводится | нет обучения двухкодировщика |
| FAISS / LangChain RAG | ноутбуки, `component-inventory` | не канонический API `untie` |
| NER-фильтры чанков | inventory: legacy stubs | нет production-вызовов |
| `FilterMode.BY_QUESTION` | inventory: нет реализации | — |
| Gumbel-Softmax псевдометки | mHFN; в Phase 1 только абляция FN | не inference |

---

## 18. Сводка: что чем является

```text
документ
  ├─ чанки (общий chunking)
  │
  ├─ [фильтр до QA]
  │     static-keywords | cluster SFFS | topic representative_terms
  │
  ├─ [QA на (под)множестве чанков]
  │     frozen extractive RoBERTa
  │
  ├─ [ранжирование / смесь score]
  │     attention+diff keywords
  │     topic A/T/interaction
  │     KeyBERT prior
  │     MiniLM cross-encoder ± keywords
  │     soft: keywords + cosine(title/question)
  │
  ├─ [shortlist / consensus ответов]
  │     AnswerAggregator (centroid)
  │     AnswerConsensus (кластеры span)
  │     CE top-k
  │     question-conditioned merge/filter
  │
  └─ [offline-разметка для поиска новых признаков]
        lexical gold_supporting
        QA match (containment)
        BM25 top-5 + Gap/K-means FN
        QAConsistency по парафразам
        → robust_positive vs hard_negative
```

Для **текущего этапа поиска признаков на SciREX** рабочий контраст — не inference-фильтр, а таблицы `official_train_q5`: `robust_positive` против `hard_negative` (и при необходимости третий класс `question_sensitive`). Перечисленные выше production- и экспериментальные scorers — как раз те семейства сигналов, которые этот этап должен сравнивать на размеченных чанках.
