# Результаты эксперимента V3: target-evidence ranking и attention signatures

Документ фиксирует **фактические результаты** прогона `experiments/qa_chunk_selection/v3` по протоколу [`qa_chunk_selection_experiments_v3_plan_attention.md`](qa_chunk_selection_experiments_v3_plan_attention.md). Это не автодамп JSON (`artifacts/qa_chunk_selection_v3/reports/results_v3.md`), а полный разбор: что запускали, что реально удалось проверить, какие числа получились и что из этого следует.

- Дата завершения успешного полного прогона: **2026-09-02 18:35:59** (старт `17:26:36`, wall-clock **4163.4 с ≈ 69 мин**, `cuda:0`).
- Config hash: `950c345e0660827a`.
- Артефакты: `artifacts/qa_chunk_selection_v3/`.
- Production (`untie/`, `datasets/`, Phase 1) **не менялся**. Пакеты E0 (`artifacts/qa_chunk_selection/`) и V2 (`artifacts/qa_chunk_selection_v2/`) **не перезаписывались**.
- Gold **никогда** не использовался в ранжировании, фильтрации, shortlist, attention-runtime признаках и агрегации ответа. Gold есть только в scored-метриках и в диагностическом reader / span oracle.

**INTERNAL_TEST не открыт.** Ниже нет ни одного QA-числа с этого сплита, кроме описи состава. Стадии `b4`, `at4-d`, `at5`, `d3`, `d4`, `d5`, `freeze-final`, `final-eval` в default stages не входили и не запускались.

Первый запуск полного пайплайна упал (`AttributeError: PipelineV3 has no attribute '_device'` — метод `_device` был случайно перезаписан при добавлении `_chunk_signature`). После фикса пользователь продолжил с `resume=true`. CV-сплиты к тому моменту уже были заморожены и **не переписывались**. Успешный лог начинается с уже подготовленного датасета и frozen splits (`data/prepare_summary.json` и `data/v3_cv_splits.json` помечены `17:25`).

---

## 0. Короткий итог

На out-of-fold пуле V3 (240 документов, 453 answerable-вопроса, 5 document-level fold, 3 seed для текстовых ранкеров) **победитель — V3-B0**: тот же рецепт, что честный победитель V2 (B2-b2a, question-balanced weighted BCE, только training-eligible HN, early stop по CALIBRATION pairwise_clean). Новые цели обучения, структурный вход, attention-сигнатуры и span-reranking **не обошли** этот baseline по протокольной селекции.

| | n | Token F1 | EM | Hit@1 | HN@1 | pairwise_clean | Amb@1 | GSRN@1 | F1 \| pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0 (выбран)** | 1359 | **27.36** | 20.75% | **36.72%** | 4.86% | **75.14%** | 50.85% | 33.92% | 61.70 |
| B1 | 1359 | 26.83 | 19.50% | 35.98% | **4.49%** | 73.68% | 49.96% | 32.60% | 61.48 |
| B2 pairwise | 1359 | 25.86 | 20.38% | 34.00% | 5.22% | 72.78% | 52.76% | 33.26% | 61.29 |
| B3 listwise | 1359 | 25.35 | 18.84% | 34.58% | 5.15% | 71.84% | 51.43% | 34.36% | 61.09 |
| structure | 1359 | 26.15 | 19.50% | 35.39% | 5.37% | 68.66% | 51.21% | 33.11% | 60.34 |
| attention_only | 453 | 24.17 | 19.21% | 35.54% | **18.98%** | 51.51% | 1.32% | 1.10% | 62.36 |
| text+attention | 453 | 26.02 | 18.32% | 32.23% | 5.74% | 68.79% | 52.76% | 30.24% | 63.48 |
| span_ranker | 453 | 24.40 | 18.76% | 36.20% | 4.86% | 74.31% | 51.21% | 34.44% | 37.32† |

† `F1 | pos` у span_ranker **нельзя** читать так же, как у B0–B3: ranking-id остаётся B0, а ответ берётся из другого чанка. См. §13.

**Главное ограничение прогона.** Очередь ручной разметки собрана (400 чанков), но файл verified overlay `annotation/target_status.jsonl` **пустой**. Поэтому:

- гипотеза H3.1 (учить модель отличать target evidence от правдоподобного чужого объекта) **не была реально проверена**;
- `P_v3_target` совпадает с исходным `robust_positive`;
- `pairwise_pv3_gt_tm` = NaN, `TargetMismatch@1` = null;
- B1 не получил ни одного verified target-mismatch negative и по сути повторил B0 с другим сборщиком пар;
- B2/B3 не получили TM-триплетов;
- TM@1 **нельзя** было использовать как hard safeguard (порог покрытия 0.95 не выполнен).

**Что прогон всё же показал.**

1. Разрыв живого пайплайна и ридера остаётся разрывом **ранжирования**. Оракул «лучший gold-positive» даёт Token F1 **73.01**; у **76.4%** вопросов есть хотя бы один positive с Token F1 ≥ 50. Живой B0 даёт **27.36**.
2. Классический HN больше не главный враг: у B0 HN@1 = 4.86%. Главный top-1 distractor — **ambiguous**, особенно `gold_supporting_robust_negative` (GSRN@1 = 33.92%).
3. Evidence часто уже в top-5 (Hit@5 = 79.0%) и почти всегда в top-10 (Hit@10 = 92.2%). Узкое место — **top-1**.
4. Attention frozen RoBERTa-ридера в этой постановке **не даёт переносимого сигнала** (AT1 AUROC 0.519, `has_signal=false`). Attention-only почти не выбирает ambiguous и зато сваливает HN в top-1 (19%).
5. Span-reranking среди top-3 B0 **не улучшает** Token F1: 24.40 против 26.46 у B0 на том же seed 42.

Бутстреп B0 Token F1 (кластеризация по документу, 5000 реплик): среднее 27.39, 95% ДИ **24.30–30.56**. Парных ДИ «B0 минус другая система» в `analysis/bootstrap.json` нет.

OOF — это **доказательство для выбора модели внутри unlocked пула**, не confirmatory INTERNAL_TEST.

---

## 1. Что это за эксперимент

UnTIE — extractive QA по научным статьям SciREX: для канонического aspect-вопроса (task / method / material / metric) нужно выбрать evidence-чанк текущей статьи и вытащить из него span.

V2 показал:

- frozen reader умеет извлекать ответ с gold-positive (оракул DEVELOPMENT Token F1 71.86);
- живой пайплайн даёт ~25.7, разрыв в выборе чанка;
- честный победитель — answerability B2-b2a, M=1;
- главный distractor — ambiguous / GSRN, не hard negative;
- reader-aware fusion, сырая склейка пассажей и старый structural mix D2 на DEVELOPMENT выигрыш не удержали.

V3 продолжает эту линию на **другой схеме оценки**: не один DEVELOPMENT из 77 вопросов, а grouped 5-fold CV по всем unlocked answerable документам, чтобы старый DEVELOPMENT не был единственным селекционным сетом. Цель — target-evidence ranking (отличать нужный объект от правдоподобного чужого контекста) плюс опциональные attention-сигнатуры и span-reranking.

Рабочий baseline V3-B0 — не старый checkpoint V2 seed 44 «как есть», а **тот же рецепт**, переобученный в каждом fold только на TRAIN_FOLDS этого fold.

Режим кандидатов: **все чанки текущей статьи**. Ридер: frozen Phase 1 `ExtractiveQuestionAnswerer` (RoBERTa). Cross-encoder: `cross-encoder/ms-marco-MiniLM-L6-v2`. BERTScore отключён. `source_label` Phase 1 **никогда не переписывается**; V3 target-status — отдельный overlay-слой.

Запущенные стадии:

```text
prepare, split, overlay-sample,
b0, b1, b2, b3, structure,
positive-quality,
at0, at1, at2, at3, at4,
span-ranker, oof-compare, report
```

Не запускались (не default): `b4`, `at4-d`, `at5`, `d3`, `d4`, `d5`, `freeze-final`, `final-eval`.

Конфиг (стабильная часть + CLI): seed 42, split_seed 42, bootstrap_seed 42, n_bootstrap 5000, 5 fold, model seeds 42/43/44, B-семейство epochs 5 / patience 2 / batch 16 / lr 2e-5 / max_length 512 / pos:hn:easy = 1:3:1, pairwise sampling p_tm:p_hn:p_easy = 0.50:0.40:0.10, listwise T=1.0, ε_HN=0.02, ε_pairwise=0.03, min_target_status_coverage=0.95, overlay 200 hard-case + 200 representative, `include_unanswerable=false`, `allow_internal_test=false`, `skip_qa=false`, `smoke=false`, `resume=true`.

---

## 2. Как читать метрики

Шкалы те же, что в V2, плюс V3-overlay метрики.

| метрика | шкала | смысл |
|---|---|---|
| Token F1 | 0..100 | токенное пересечение предсказанного спана с gold |
| Char F1 | 0..1 | символьное пересечение |
| ROUGE-L F1 | 0..1 | longest common subsequence |
| Exact match (EM) | 0..1 | совпадение нормализованных строк; в таблицах часто как % |
| Hit@K / Hit@K_original | 0..1 | среди top-K есть исходный gold **robust_positive** |
| TargetEvidenceHit@K_v3 | 0..1 | среди top-K есть `P_v3_target`. В этом прогоне **тождественно** Hit@K_original, потому что overlay пуст |
| HN@K | 0..1 | доля слотов в top-K, занятых **hard_negative**: `\|top-K ∩ HN\| / K`. При K=1 это «top-1 есть HN». Из-за деления на K величина **не** обязана расти с K (это не P(хотя бы один HN в top-K)) |
| pairwise_clean_hn | 0..1 | доля пар (pos, HN без `hn_gold_string_warning`), где score(pos) > score(HN); среднее по вопросам, у которых есть обе стороны |
| pairwise_pos_gt_amb_original | 0..1 | то же для исходного ambiguous |
| pairwise_pos_gt_gsrn_original | 0..1 | то же для GSRN |
| pairwise_pv3_gt_tm | 0..1 | pos_v3 vs verified target_mismatch. Здесь **NaN**: verified TM нет |
| Ambiguous@1_original | 0..1 | `source_label` выбранного чанка = ambiguous. Это **не** автоматически «неправильный ответ» |
| GSRN@1_original | 0..1 | primary_class top-1 = `gold_supporting_robust_negative` |
| TargetMismatch@1 | 0..1 или null | доля top-1 с verified TM. Здесь **null**: coverage verified TM = 0 |
| target_status_coverage@1 | 0..1 | доля вопросов, у которых статус top-1 известен. Без overlay известны только исходные pos / HN / easy; ambiguous считается неизвестным ≈ 0.49 у B0 |
| F1 \| pos (`f1_given_selected_positive`) | 0..100 | Token F1, если **selected_chunk_label** (чанк ранжирования) — original positive. Для span_ranker это поле смешано с другим QA-чанком, см. §13 |
| attribution-aware Token F1 | 0..100 | Token F1, обнулённый на TM / known non-target источниках. Без overlay почти совпадает с обычным F1 |
| gold_match_from_non_original_positive | 0..1 | предсказанный спан совпал с gold, хотя выбранный чанк не original positive |
| MRR | 0..1 | mean reciprocal rank первого original positive |

**Ambiguous** — отдельный класс Phase 1, не hard negative. Подтипы: `gold_supporting_robust_negative`, `false_negative_candidate`, `question_sensitive`.

**n в таблицах.** B0–B3 и structure считаются по **1359** строкам = 453 вопроса × 3 seed (42/43/44). Attention и span_ranker считались только на seed 42 → **453** строки. Честное QA-сравнение attention/span с B0 нужно делать по **B0 seed=42** (Token F1 26.46), а не по трёхсидовому среднему 27.36. Автосравнение `oof_compare.json` этого не делает и вычитает attention/span из трёхсидового B0 — дельты там завышены примерно на 0.9 F1.

OOF bootstrap — **model-selection evidence**, не confirmatory INTERNAL_TEST.

---

## 3. Данные и сплиты

### 3.1. Снимок V2 и locked-сплиты

`data/prepare_summary.json`: снимок V2-датасета — **955** instance-записей. `source_label_rewritten=false`. INTERNAL_TEST: 78 записей, **locked**.

Состав снимка по V2-ролям (все вопросы, не только answerable):

| V2-сплит | записей | статей | answerable | роль в V3 |
|---|---:|---:|---:|---|
| train | 477 | 180 | 303 | unlocked, входит в CV |
| calibration | 111 | 44 | 73 | unlocked, входит в CV |
| development | 126 | 44 | 77 | unlocked, входит в CV |
| internal_test | 78 | 31 | 54 | **LOCKED** |
| seen_exploratory_dev | 163 | 65 | 105 | official SciREX DEV; exploratory, не селекция |

Сумма unlocked документов V2: 180+44+44 = **268**. Это число из протокола §5.3.

### 3.2. Почему в CV не 268 документов, а 240

Primary scope V3 — только answerable (`include_unanswerable=false`). Документы, у которых нет ни одного `robust_positive`, в CV не попали. Unlock + answerable = **453 вопроса / 240 документов**. 268 − 240 = **28** unlocked документов без robust_positive. Это не потеря данных и не другой корпус: это ровно те же 453 answerable-вопроса, что train+cal+dev V2 (303+73+77).

Official SciREX DEV целиком остаётся `seen_exploratory_dev` и **не** участвует в подборе.

### 3.3. Grouped 5-fold CV

`data/v3_cv_splits.json`:

- grouping unit: `document_id`;
- seed 42;
- алгоритм: `greedy_grouped_stratification_by_stratum_then_smallest_fold`;
- `n_documents=240`, `n_folds=5`, `frozen=true`;
- `document_id_hash` = `21cf53200c0350ad4176b378b7c56c6eadd6f31cf479560ce1db87e12a05f90d`;
- `split_payload_hash` = `d3647846e7be929791b359a1cb93b6722dcd0df6b8567159c9fd6d50eba2fcb8`.

По fold (документы / answerable q / robust_positives / ambiguous):

| fold | docs | q | pos | amb | task | material | method | metric | HN 0 / 1 / 2–3 / 4+ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 48 | 91 | 267 | 728 | 29 | 25 | 22 | 15 | 11 / 6 / 24 / 7 |
| 1 | 48 | 90 | 276 | 737 | 26 | 25 | 24 | 15 | 9 / 3 / 25 / 11 |
| 2 | 48 | 91 | 224 | 791 | 30 | 25 | 23 | 13 | 10 / 3 / 24 / 11 |
| 3 | 48 | 91 | 229 | 720 | 32 | 26 | 23 | 10 | 10 / 7 / 21 / 10 |
| 4 | 48 | 90 | 257 | 823 | 30 | 25 | 23 | 12 | 7 / 7 / 24 / 10 |
| **сумма** | **240** | **453** | **1253** | 3799 | **147** | **126** | **115** | **65** | 47 / 26 / 118 / 49 |

Аспекты в таблице fold — по answerable-вопросам. Сумма аспектов 147+126+115+65 = 453.

Циклические роли (как в протоколе):

| run | train folds | calibration | eval |
|---:|---|---:|---:|
| 0 | 0, 1, 2 | 3 | 4 |
| 1 | 1, 2, 3 | 4 | 0 |
| 2 | 2, 3, 4 | 0 | 1 |
| 3 | 0, 3, 4 | 1 | 2 |
| 4 | 0, 1, 4 | 2 | 3 |

В логе: run 0/1/2/3 имеют `n_train=272`, run 4 — `n_train=271` (90+91+90); calibration 90 или 91 в зависимости от eval-fold. Каждый текстовый ранкер в каждом run учится на трёх seed → 5×3 = 15 чекпоинтов на семейство.

---

## 4. Overlay: очередь есть, verified labels нет

Таксономия `v3-target-status-1` (`annotation/taxonomy.json`):

- `target_match_status`: `target_evidence` / `target_mismatch` / `partial_support` / `unresolved`;
- `v3_gold_relation`: matches / semantically_equivalent / new_valid_value_candidate / no_gold_match_expected / unclear;
- аннотатор **не должен** видеть `b_score`, `attention_score`, `predicted_class`, `model_rank`.

Сэмплинг (`overlay_hard_n=200`, `overlay_repr_n=200`):

| | count |
|---|---:|
| очередь всего | 400 |
| hard_case | 200 |
| representative | 200 |
| subtype GSRN | 192 |
| subtype false_negative_candidate | 105 |
| subtype question_sensitive | 103 |
| verified `target_status.jsonl` | **0** |

Скоры модели спрятаны от annotator-файла (`queue.jsonl`); метаданные со скорами лежат отдельно в `sampling_metadata.jsonl`. Это соответствует протоколу §7.9–7.9.1.

Пустой overlay — не сбой кода стадии `overlay-sample`. Стадия сделала то, что умеет без человека: собрала очередь. Ручная адъюдикация в этот прогон **не входила**.

Следствия для всех последующих таблиц:

- `P_v3_target == P_original`;
- Hit@K_original ≡ TargetEvidenceHit@K_v3;
- TM-метрики либо null, либо lower_bound = 0;
- `target_status_coverage@1` ≈ 0.49 у текстовых ранкеров, потому что известен статус только pos/HN/easy;
- у attention_only coverage@1 подскакивает до 0.987: система почти не выбирает ambiguous, а pos/HN/easy имеют известный исходный статус. Это **не** покрытие verified TM.

---

## 5. Протокол селекции

Не лексикографический ключ V2 `(pairwise_clean, Hit@1, F1|pos, Token F1)`.

Hard safeguards относительно B0 (из `config_resolved.json` и `analysis/selection.json`):

- HN@1 ≤ HN@1_B0 + 0.02;
- pairwise_clean ≥ pairwise_clean_B0 − 0.03;
- TM@1 как hard safeguard **выключен**, если coverage ниже 0.95. В этом прогоне так и есть: `tm_as_hard_safeguard=false`, пометка *«TM@1 diagnostic only»*.

Среди допустимых систем primary criterion: **document-aspect Token F1**.

Порог pairwise относительно B0: 0.75139 − 0.03 = **0.72139**.
Порог HN@1: 0.04857 + 0.02 = **0.06857**.

| система | HN safeguard | pairwise safeguard | admissible | Token F1 |
|---|---|---|---|---:|
| **b0** | ok | ok | да | **27.36** |
| b1 | ok | ok | да | 26.83 |
| b2 | ok | ok (0.7278 ≥ 0.7214) | да | 25.86 |
| span_ranker | ok | ok | да | 24.40 |
| b3 | ok | **fail** (0.7184 < 0.7214) | нет | 25.35 |
| structure | ok | **fail** (0.6866) | нет | 26.15 |
| text_plus_attention | ok | **fail** (0.6879) | нет | 26.02 |
| attention_only | **fail** (HN@1 0.1898) | **fail** (0.5151) | нет | 24.17 |

Выбрано: **b0**.
Допустимые: `b0`, `b1`, `b2`, `span_ranker`.
Отвергнутые: `b3`, `structure`, `attention_only`, `text_plus_attention`.
Парето по (Token F1 ↑, HN@1 ↓): **b0, b1**. У B1 чуть ниже F1 и чуть ниже HN@1.

---

## 6. Сводная OOF-таблица (полные колонки)

Числа из `analysis/oof_compare.json` / `analysis/selection.json`. Дельты в автофайле — против трёхсидового B0; для систем с n=453 это смешение выборок, см. §2.

| система | n | Token F1 | attr F1 | EM | Char F1 | ROUGE-L | Hit@1 | Hit@5 | Hit@10 | HN@1 | HN@5 | HN@10 | MRR | pairwise_clean | pairwise all HN | pos>amb | pos>GSRN | Amb@1 | GSRN@1 | F1\|pos | gold≠orig pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 1359 | **27.36** | 26.80 | 20.75% | 0.521 | 0.339 | **36.72%** | 79.03% | 92.20% | 4.86% | 6.90% | 8.23% | 0.542 | **75.14%** | 74.96% | 64.51% | 55.47% | 50.85% | 33.92% | 61.70 | 4.64% |
| B1 | 1359 | 26.83 | 26.08 | 19.50% | 0.512 | 0.330 | 35.98% | 77.85% | 92.05% | 4.49% | 6.67% | 8.10% | 0.534 | 73.68% | 73.89% | 64.18% | 55.25% | 49.96% | 32.60% | 61.48 | 4.49% |
| B2 | 1359 | 25.86 | 24.95 | 20.38% | 0.516 | 0.323 | 34.00% | 76.67% | 91.02% | 5.22% | 6.73% | 8.31% | 0.518 | 72.78% | 72.71% | 62.33% | 53.20% | 52.76% | 33.26% | 61.29 | 4.78% |
| B3 | 1359 | 25.35 | 24.78 | 18.84% | 0.509 | 0.318 | 34.58% | 76.53% | 91.32% | 5.15% | 7.03% | 8.51% | 0.527 | 71.84% | 72.16% | 62.83% | 53.84% | 51.43% | 34.36% | 61.09 | 4.19% |
| structure | 1359 | 26.15 | 25.51 | 19.50% | 0.515 | 0.329 | 35.39% | 75.50% | 90.51% | 5.37% | 7.39% | 8.31% | 0.527 | 68.66% | 68.27% | 62.37% | 54.06% | 51.21% | 33.11% | 60.34 | 4.86% |
| att_only | 453 | 24.17 | 22.19 | 19.21% | 0.503 | 0.293 | 35.54% | 73.51% | 83.66% | 18.98% | 14.30% | 10.46% | 0.526 | 51.51% | 51.47% | 67.80% | 67.93% | 1.32% | 1.10% | 62.36 | 1.99% |
| text+att | 453 | 26.02 | 25.00 | 18.32% | 0.514 | 0.311 | 32.23% | 75.50% | 89.85% | 5.74% | 7.51% | 8.05% | 0.510 | 68.79% | 68.71% | 59.14% | 53.59% | 52.76% | 30.24% | 63.48 | 5.52% |
| span | 453 | 24.40 | 23.49 | 18.76% | 0.507 | 0.313 | 36.20% | 80.57% | 92.27% | 4.86% | 7.24% | 8.20% | 0.541 | 74.31% | 73.97% | 64.47% | 55.58% | 51.21% | 34.44% | 37.32† | 5.30% |

У attention_only pairwise pos>amb / pos>GSRN высокие (≈68%), потому что система почти не ставит ambiguous в top-1 и тем самым «выигрывает» эти пары ценой обвала pos>HN. Это не признак хорошего ранжирования.

---

## 7. V3-B0 — честный baseline (рецепт V2-B2 b2a)

### 7.1. Что это

В каждом fold заново учится MiniLM cross-encoder:

- вопрос-балансированный weighted BCE, уникальные пары, без оверсэмпла HN;
- negatives только `training_eligible_hn` + ограниченный easy;
- early stopping по CALIBRATION_FOLD `pairwise_clean_hn`, patience 2, max 5 эпох;
- инференс: сигмоид-вероятность по всем чанкам статьи, top-1 в ридер.

Это **не** загрузка V2 checkpoint seed 44.

### 7.2. Обучение

15 чекпоинтов. Средняя лучшая эпоха **4.27** (гистограмма: epoch 5 — 8 раз, 3 — 4, 4 — 3). Средний CAL pairwise_clean **0.7649** (min 0.6988, max 0.8254).

B0 — единственное семейство, у которого CAL pairwise в среднем выше, чем у B1/B2/B3/structure. Это согласуется с тем, что B0 остался победителем и на OOF QA.

### 7.3. OOF QA, три seed

| seed | n | Token F1 | Hit@1 | HN@1 | Amb@1 | pairwise_clean |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 453 | 26.46 | 36.20% | 4.86% | 51.21% | 74.31% |
| 43 | 453 | 27.94 | 37.97% | 5.96% | 49.89% | 74.33% |
| 44 | 453 | 27.68 | 35.98% | **3.75%** | 51.43% | **76.78%** |
| **pool** | **1359** | **27.36** | **36.72%** | **4.86%** | **50.85%** | **75.14%** |

Seed 44 снова лучший по pairwise (как в V2), но уже не единственный «выбранный checkpoint»: селекция идёт по pooled OOF, не по одному seed.

### 7.4. Разброс по fold (seed 42)

| eval fold | n | Token F1 | Hit@1 | HN@1 |
|---:|---:|---:|---:|---:|
| 0 | 90 | 30.67 | 34.44% | 0.00% |
| 1 | 91 | 22.59 | 34.07% | 7.69% |
| 2 | 90 | 27.11 | 37.78% | 4.44% |
| 3 | 91 | 20.80 | 31.87% | 5.49% |
| 4 | 91 | 31.21 | 42.86% | 6.59% |

Fold 1 и 3 заметно тяжелее fold 0 и 4. Разброс Token F1 ≈ 10 пунктов. Один старый DEVELOPMENT на 77 вопросов легко мог попасть в «лёгкий» или «тяжёлый» карман; именно поэтому V3 ушёл в grouped CV.

### 7.5. Состав top-1 (pool 1359)

| source_label | count | доля |
|---|---:|---:|
| positive | 499 | 36.72% (= Hit@1) |
| ambiguous | 691 | 50.85% |
| easy_negative | 103 | 7.58% |
| hard_negative | 66 | 4.86% |

Primary class внутри top-1:

| primary_class | count | доля |
|---|---:|---:|
| robust_positive | 499 | 36.72% |
| gold_supporting_robust_negative | 461 | 33.92% |
| question_sensitive | 136 | 10.01% |
| easy_negative | 103 | 7.58% |
| false_negative_candidate | 94 | 6.92% |
| hard_negative | 66 | 4.86% |

GSRN почти равен числу positives. Картина V2 на DEVELOPMENT (pos 29 / GSRN 27 из 77) воспроизводится на большом OOF: система путает «чанк, из которого тоже можно вытащить похожую сущность» с нужным evidence.

### 7.6. По аспектам (pool 1359)

| аспект | n | Token F1 | EM | Hit@1 | HN@1 | Amb@1 | GSRN@1 | F1\|pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| task | 441 | 35.54 | 18.59% | 37.19% | 5.44% | 52.38% | 33.79% | 67.87 |
| material | 378 | 30.94 | 32.01% | **51.85%** | 2.38% | 39.95% | 28.57% | 54.82 |
| method | 345 | 18.66 | 16.81% | 23.77% | 6.96% | 57.68% | 40.87% | **73.25** |
| metric | 195 | 17.31 | 10.77% | 29.23% | 4.62% | 56.41% | 32.31% | 51.01 |

Method/metric остаются тяжёлыми. У method F1|pos высокий (73.3): если ранкер всё-таки попал в positive, ридер справляется. Проблема method — **ранжирование** (Hit@1 23.8%, Amb@1 57.7%, GSRN@1 40.9%). У metric и Hit@1, и F1|pos ниже: здесь есть и ранжирование, и потолок ридера (см. §14).

### 7.7. Бутстреп

`analysis/bootstrap.json`, только B0 Token F1, кластеризация по документу, 5000:

- mean 27.39
- 95% CI **24.30 – 30.56**

Парных сравнений B0 vs B1 / vs structure в этом файле нет. По точечным оценкам B1 (−0.53 F1) и structure (−1.21 F1) лежат внутри ширины ДИ B0; это **не** формальный тест на ноль.

---

## 8. V3-B1 — «расширенная answerability» без TM-negatives

Протокол: тот же BCE, но positives = original pos + verified target_evidence, negatives = eligible HN + verified target_mismatch + easy.

Факт прогона: verified overlay пуст → B1 не добавил ни одного TM-negative и ни одного overlay-positive. Отличие от B0 — только реализация сборщика пар / весов, не учебный сигнал H3.1.

OOF (n=1359): Token F1 26.83 (−0.53), Hit@1 35.98% (−0.74 п.п.), HN@1 4.49% (−0.37 п.п.), pairwise 73.68% (−1.46 п.п.), Amb@1 49.96% (−0.88), GSRN@1 32.60% (−1.33).

Top-1: pos 489, amb 679, easy 130, HN 61. B1 чуть чаще уходит в easy и чуть реже в HN/GSRN, но positives тоже меньше.

По seed:

| seed | Token F1 | Hit@1 | HN@1 | pairwise |
|---:|---:|---:|---:|---:|
| 42 | 25.95 | 36.87% | 3.97% | 75.31% |
| 43 | 26.73 | 34.44% | 4.19% | 73.55% |
| 44 | 27.80 | 36.64% | 5.30% | 72.18% |

На seed 42 Hit@1 даже чуть выше B0-s42 (36.87% vs 36.20%), но Token F1 ниже (25.95 vs 26.46) из-за худшего F1|pos (58.86 vs 61.26).

**Вывод.** B1 допустим и лежит на парето по HN@1, но H3.1 не тестировался. Нельзя читать «добавление TM не помогло»: TM не добавляли.

---

## 9. V3-B2 — pairwise ranking без TM-триплетов

Протокол: triplets `(q, pos, TM)` / `(q, pos, HN)` с вероятностями 0.50 / 0.40 / 0.10. Loss `-log σ(s+ − s−)`.

Факт прогона: TM-триплетов нет, остаются pos vs HN/easy. Это ближе к V2-b2c (pairwise loss), который на V2 уже проиграл b2a по DEVELOPMENT.

OOF: Token F1 25.86 (−1.50), Hit@1 34.00% (−2.72 п.п.), HN@1 5.22%, pairwise 72.78% (едва выше порога 72.14%). Amb@1 вырос до 52.76%. Top-1: pos 462, amb 717, easy 109, HN 71.

По seed F1: 42 → 24.80, 43 → 25.78, 44 → 27.00. Даже лучший seed не обходит pooled B0.

CAL pairwise в среднем 0.7468, лучшая эпоха в среднем 3.4 — модель останавливается раньше B0 и на CAL слабее.

**Вывод.** Pairwise objective без нового класса negatives не бьёт question-balanced BCE. Это согласуется с V2-b2c.

---

## 10. V3-B3 — listwise / document-level ranking

Протокол: listwise по документу, температура T=1.0. Смысл H3.2: сравнивать чанки внутри одной статьи, а не независимые пары.

OOF: Token F1 25.35 (−2.01), Hit@1 34.58%, HN@1 5.15%, pairwise **71.84%** — не проходит safeguard. Top-1: pos 470, amb 699, easy 120, HN 70. GSRN@1 34.36% — чуть хуже B0.

По seed F1: 42 → 25.45, 43 → 25.78, 44 → 24.83. Стабильно ниже B0.

Fold 3 на seed 42 провален особенно сильно: Token F1 14.05, Hit@1 21.98%.

**Вывод.** Listwise без TM-разметки не дал document-level преимущества и размыл pos>HN. H3.2 в задуманном виде (ранжировать target vs mismatch внутри документа) не проверялась.

---

## 11. Structure — богатый контекстный вход

Не старый D2 (RRF/section-overlap/generic CE). V3-structure подаёт в тот же CE-рецепт, что B0, расширенный текст:

```text
[SECTION] … [PREV] … [CHUNK] … [NEXT] …
```

OOF: Token F1 26.15 (−1.21), Hit@1 35.39%, HN@1 5.37%, pairwise **68.66%** (обвал относительно 75.14%). F1|pos 60.34. Top-1: pos 481, amb 696, easy 109, HN 73.

По аспектам method ещё хуже, чем у B0: Token F1 14.81 vs 18.66, Hit@1 20.87% vs 23.77%.

CAL pairwise средний 0.7192 — худший среди текстовых семейств.

**Вывод.** Соседний контекст и секция не помогли отделить GSRN от positive и заметно повредили pos>HN. Это другой механизм, чем E0-D2/V2-D2, но практический итог тот же: structural mix поверх answerability не апгрейд.

---

## 12. Attention (H3.7–H3.10)

Гипотеза: у frozen ридера есть внутренний сигнал «я смотрю на нужный объект / успешный экстракционный паттерн», который можно сжать в компактную сигнатуру и использовать для ранжирования.

Прогон: AT0–AT4. Не запускались AT4-D (дистилляция) и AT5 (причинное маскирование).

### 12.1. AT0 — регрессия `output_attentions`

4 контрольных примера. Dropout выключен. Gold в runtime не входил. Raw tensors не сохранялись.

У всех четырёх: `start_max_abs=0.0`, `end_max_abs=0.0`, argmax start/end совпадают с прогоном без attentions. Значит, включение `output_attentions` **не меняет логиты** ридера. Это необходимое условие, чтобы attention-анализ не сдвигал сам QA.

AT0 плюс факт, что `qa_answerer` был загружен (`skip_qa=false`, `smoke=false`), означает: дальше использовался **реальный** ридер, не synthetic smoke-attention. Поле `note` в `at1.json` («smoke uses synthetic attention…») — оставшаяся строка кода, не описание этого прогона.

### 12.2. AT1 — есть ли сигнал в среднем векторе

Протокол просил собирать группы на TRAIN_FOLDS. В коде AT1 цикл идёт по **всему unlocked** (`self.unlocked`, 453 вопроса). Поэтому G1=615 совпадает с глобальным `reader_easy` из positive-quality, а не с train-only срезом.

Фактические определения в коде (`pipeline.stage_at1`):

| группа | как собрано в этом прогоне | n |
|---|---|---:|
| G1 | original positive с Token F1 ≥ 80 (`reader_easy`) | 615 |
| G2 | остальные original positives (medium+hard+failed = 264+138+236) | 638 |
| G3 | `known_target_status == target_mismatch` | **0** |
| G4 | все чанки с `source_label == hard_negative` (не фильтр `training_eligible_hn`) | 714 |
| G5 | `easy_negative` | 3181 |

AUROC скаляра `mean(vector)` для G1 vs G3-или-G4: **0.519**. Порог `has_signal` в коде — 0.55. Результат: `has_signal=false`.

G3 пуст, поэтому сравнение фактически G1 vs G4 (успешный positive vs любой HN). Даже эту грубую оппозицию, которую текстовый B0 решает с pairwise ≈ 75%, средний attention-вектор почти не разделяет.

### 12.3. AT2 — компактные сигнатуры

5148 сигнатур, raw tensors не сохранялись (`signatures.jsonl` ≈ 23 МБ). Это сжатые признаки, не полные attention maps.

### 12.4. AT3 — отбор голов

Поле `selection` в `at3.json` пишет *«TRAIN-only univariate AUROC»*, но код снова идёт по **всему unlocked** и метит positive vs HN/easy. Это не fold-честный отбор голов: eval-вопросы участвуют в AUROC. N взято как второй элемент сетки `n_heads_grid=[1,2,4,8,16]` → **N=2**, без CAL-подбора (вторая половина той же строки, *«CAL in full run»*, в этом прогоне не выполнялась).

Выбранные головы (сохранённые AUROC):

| layer | head | AUROC |
|---:|---:|---:|
| 1 | 4 | 0.552 |
| 11 | 8 | 0.548 |

Сохранённый `all_head_mean_auroc`: **0.485**. Это диагностическое число из JSON; в коде оно считается как AUROC по `mean(q_to_s_matrix)` на префиксе `signatures.jsonl`, а не как аккуратно выровненное среднее всех голов на тех же примерах, что labels. Даже лучшие головы едва выше 0.55. Специализированных «сильных» голов в смысле H3.8 здесь нет.

### 12.5. AT4-P — prototype distance

Прототипы считаются уже **по TRAIN_FOLDS каждого CV-run** (здесь протокол соблюдён). `success` и `target` в коде — все TRAIN positives, не только reader-success; `mismatch` — TRAIN HN + easy. Оценка чанка: `success_minus_mismatch`.

Два ранкера, оба только seed 42, n=453.

**attention_only** — prototype по **всем** чанкам статьи, без текстового shortlist.

| | Token F1 | Hit@1 | HN@1 | pairwise | Amb@1 | GSRN@1 |
|---|---:|---:|---:|---:|---:|---:|
| attention_only | 24.17 | 35.54% | **18.98%** | 51.51% | **1.32%** | 1.10% |
| B0 seed 42 | 26.46 | 36.20% | 4.86% | 74.31% | 51.21% | 34.44% |

Top-1 attention_only: easy 200, positive 161, HN 86, ambiguous **6**. Система почти игнорирует ambiguous и предпочитает «чистые» исходные классы — в том числе HN. Pairwise pos>HN ≈ 0.5: относительно HN это случайный классификатор.

Парно против B0-s42 (453 общих вопроса): средняя дельта Token F1 **−2.29**; win 20.8% / lose 26.5% / tie 52.8%.

По аспектам attention_only ведёт себя иначе, чем B0:

| аспект | n | att F1 | att Hit@1 | att HN@1 | B0-s42 F1 | B0-s42 Hit@1 |
|---|---:|---:|---:|---:|---:|---:|
| method | 115 | **36.36** | **53.91%** | 22.61% | 15.75 | 22.61% |
| material | 126 | 20.41 | 35.71% | 15.87% | 32.24 | 52.38% |
| task | 147 | 22.30 | 27.21% | 23.13% | 34.59 | 36.73% |
| metric | 65 | 14.15 | 21.54% | 9.23% | 15.85 | 27.69% |

На method attention-only резко поднимает Hit@1 и F1 — и одновременно держит HN@1 22.6%. Это любопытный exploratory сигнал, не допустимая система: safeguard HN провален на всём пуле. Нельзя брать «attention лучше на method» как готовый рецепт без отдельной проверки, почему прототип путает method-positive с HN.

**text_plus_attention** — B0 shortlist top-5, на shortlist смесь `0.5 * B0 + 0.5 * prototype`.

| | Token F1 | Hit@1 | HN@1 | pairwise | Amb@1 |
|---|---:|---:|---:|---:|---:|
| text+attention | 26.02 | 32.23% | 5.74% | 68.79% | 52.76% |
| B0 seed 42 | 26.46 | 36.20% | 4.86% | 74.31% | 51.21% |

Hit@1 падает на 4 пункта: fusion сдвигает top-1 внутри shortlist в сторону не-positive. Token F1 почти как у B0-s42 (−0.44), потому что F1|pos чуть выше (63.48). Pairwise 68.8% — safeguard fail.

Парно vs B0-s42: mean ΔF1 **−0.44**; win 16.1% / lose 19.0% / tie 64.9%.

Top-1: amb 239, pos 146, easy 42, HN 26. Positives меньше, чем у B0-s42 (164).

По fold text+att повторяет разброс B0 (fold 0/4 лёгкие, 1/3 тяжёлые), но Hit@1 везде ниже.

**Вывод по вниманию.** H3.7–H3.10 в этой постановке не подтверждены. Средний вектор и две лучшие головы почти случайны на G1 vs HN; G3 (настоящий TM) отсутствует, так что «успешный паттерн vs чужой объект» измерить было нельзя. Attention-only ломает HN-барьер, который текстовый B0 как раз умеет держать. Смесь с B0 не компенсирует это и портит Hit@1.

---

## 13. SpanRanker (H3.4)

Протокол: среди top-K чанков B0 сгенерировать спаны ридера и переранжировать пару (чанк, спан).

Факт реализации полного прогона: `k = 2 if smoke else 3` → **K=3**. Учится и применяется только seed 42. Ranking-id для Hit@K/HN@K/Amb@K остаются **ранжированием B0**; меняется только то, из какого чанка берётся `qa_answer`.

Поэтому Hit@1 / Hit@5 / HN@1 / Amb@1 / pairwise у span_ranker **численно совпадают** с B0 seed 42 (36.20% / 80.57% / 4.86% / 51.21% / 74.31%). Это не независимое ранжирование.

Token F1 при этом **24.40** против 26.46 у B0-s42. Парно: mean ΔF1 **−2.06**; win 16.1% / lose 18.8% / tie 65.1%.

### 13.1. Почему F1|pos = 37.32 — ловушка

`f1_given_selected_positive` смотрит на `selected_chunk_label` (top-1 B0). Ответ уже может быть из **другого** чанка top-3.

| | |
|---|---|
| тот же чанк, что B0 top-1 | **32.2%** (146/453) |
| другой чанк | 67.8% |
| selected_label (B0 top-1): pos / amb / easy / HN | 164 / 232 / 35 / 22 |
| qa_chunk_label (откуда ответ): pos / amb / easy / HN | 143 / 238 / 44 / 28 |
| Token F1, если **QA-чанк** positive | 60.83 |
| Token F1, если **selected** positive (поле F1\|pos) | 37.32 |
| oracle среди span-кандидатов, raw | 48.25 |
| oracle среди span-кандидатов, attribution-aware | 41.58 |

Когда SpanRanker всё-таки отвечает из positive, качество экстракции нормальное (60.83 ≈ B0 F1|pos 61.26). Он просто **реже** отвечает из positive, чем B0 top-1 (143 vs 164), и в 68% случаев уходит с B0-победителя. Поле F1|pos 37.32 смешивает «B0 попал в positive» с «ответ взят из другого чанка» — его нельзя сравнивать с 61.70 у B0.

Oracle 48.25 среди top-3 спанов ниже оракула «любой gold-positive» 73.01: даже идеальный выбор спана внутри B0 top-3 не достаёт потолка ридера. У B0 seed 42 Hit@5 = 80.57%, но SpanRanker смотрит только top-3, и не каждый positive в shortlist даёт хороший спан.

По аспектам span_ranker хуже B0-s42 везде, особенно metric (F1 12.84 vs 15.85; F1|pos 17.39 — снова артефакт метрики).

**Вывод.** H3.4 в реализации K=3 поверх B0 не помог. Узкое место по-прежнему top-1 чанка, а не выбор спана внутри уже неплохого shortlist. Отдельно: метрика F1|pos для этой ветки в текущем коде считается некорректно относительно фактического QA-чанка.

---

## 14. Positive-quality и оракул ридера

`analysis/positive_quality/summary.json`, диагностика на всех gold-positive unlocked answerable (INTERNAL_TEST не трогали).

| | |
|---|---|
| gold-positives | 1253 |
| вопросов | 453 |
| mean Token F1 на positives | 64.66 |
| EM на positives | 55.15% |
| доля positives с F1 ≥ 50 | 70.15% |
| доля positives с F1 ≥ 80 | 49.08% |
| **оракул best-positive Token F1** | **73.01** |
| доля вопросов с хотя бы одним pos F1 ≥ 50 | **76.38%** |

Полосы positives:

| band | n | смысл |
|---|---:|---|
| reader_easy | 615 | Token F1 ≥ 80 |
| reader_medium | 264 | |
| reader_hard | 138 | |
| reader_failed | 236 | |

По аспектам mean F1 на positives: task 70.10, method 67.88, metric 58.76, material 58.23.

Разрез band × aspect:

| аспект | easy | medium | hard | failed |
|---|---:|---:|---:|---:|
| task | 224 | 67 | 58 | 41 |
| method | 183 | 72 | 25 | 69 |
| material | 173 | 89 | 32 | 110 |
| metric | 35 | 36 | 23 | 16 |

Material имеет много reader_failed (110/404 positives) — согласуется с относительно низким F1|pos у B0 на material (54.8) при высоком Hit@1 (51.9%): ранкер material находит, ридер извлекает хуже. Method наоборот: ридер на positive сильный, ранкер слабый.

Сравнение с V2 DEVELOPMENT-оракулом (другой, меньший сет): V2 DEV best-pos F1 71.86, solvable F1≥50 = 76.6%. V3 OOF 73.01 / 76.4% — тот же порядок. Вывод V2 («ридер не главный потолок») на большом unlocked пуле **подтверждается**.

Живой B0 27.36 vs оракул 73.01. Запас по-прежнему в выборе чанка, особенно против ambiguous/GSRN.

### 14.1. Gates D4 / D5

Протокол оставлял sentence-level D4 и aggregation D5 за воротами качества positives.

- **D4 gate passed**: 374 hard/failed positives с присутствующим gold и длинным контекстом.
- **D5 gate passed**: 307 / 453 вопросов имеют ≥ 2 positives.

Оба gate открыты, но стадии D4/D5 **не запускались** (не default). Это разрешение на будущий эксперимент, не результат D4/D5.

---

## 15. Сравнение с V2 — только как ориентир

V2 DEVELOPMENT (77 q, один сплит, B2-b2a seed 44) и V3-B0 OOF (453 q × 3 seed / 240 docs) — **разные схемы**. Нельзя писать «V3 выиграл у V2».

| | V2-B2 DEVELOPMENT | V3-B0 OOF pool | V3-B0 seed 42 |
|---|---:|---:|---:|
| n вопросов (уникальных) | 77 | 453 | 453 |
| Hit@1 | 37.7% | 36.72% | 36.20% |
| HN@1 | 3.9% | 4.86% | 4.86% |
| pairwise_clean | 74.6% | 75.14% | 74.31% |
| Token F1 | 25.71 | 27.36 | 26.46 |
| F1\|pos | 59.89 | 61.70 | 61.26 |
| Amb@1 | 54.5% | 50.85% | 51.21% |
| oracle best-pos F1 | 71.86 | 73.01 | 73.01 |
| solvable F1≥50 | 76.6% | 76.4% | 76.4% |

Порядок величин совпал: Hit@1 середина 30%, HN@1 около 4–5%, Amb@1 около половины, живой F1 середина 20, оракул ≈ 72–73. V3 не опроверг V2 и не заменил INTERNAL_TEST.

---

## 16. Обучение текстовых семейств

75 training_report.json: 5 семейств × 5 run × 3 seed.

| семейство | n | mean best epoch | mean CAL pairwise | min / max CAL pairwise |
|---|---:|---:|---:|---|
| B0 | 15 | 4.27 | **0.7649** | 0.6988 / 0.8254 |
| B1 | 15 | 4.40 | 0.7502 | 0.6822 / 0.8044 |
| B2 | 15 | 3.40 | 0.7468 | 0.6652 / 0.8205 |
| B3 | 15 | 3.73 | 0.7296 | 0.6613 / 0.7904 |
| structure | 15 | 4.13 | 0.7192 | 0.6781 / 0.7578 |

Порядок CAL pairwise совпадает с порядком OOF pairwise и в целом с QA: B0 > B1 > B2 > B3 > structure. Early stopping смотрел на CAL fold, не на eval fold и не на Token F1.

---

## 17. Ресурсы и тайминг

Успешный прогон: 17:26:36 → 18:35:59, **4163.4 с**. QA-cache hit rate в конце **77.37%**.

Границы стадий по `logs/qa_chunk_selection_v3.log` (первая отметка стадии → OOF/конец стадии):

| стадия | к моменту, с | wall стадии (прибл.) |
|---|---:|---:|
| start + models | 6 | 6 |
| overlay-sample | 13 | 7 |
| B0 (15 чекпоинтов + eval) | 807 | 794 |
| B1 | 1326 | 518 |
| B2 | 1879 | 553 |
| B3 | 2462 | 583 |
| structure | 3247 | 786 |
| positive-quality | 3441 | 193 |
| AT0 | 3442 | 1 |
| AT1 | 3691 | 250 |
| AT2 | 3834 | 142 |
| AT3 | 3834 | <1 |
| AT4 (оба ранкера) | 3969 | 135 |
| span-ranker | 4163 | 194 |
| oof-compare + report | 4163 | <1 |

Самые дорогие куски: обучение B0 и structure, затем B3/B2/B1, затем AT1 (групповая статистика по сигнатурам) и span-ranker (ридер по top-3). AT0/AT3 почти бесплатны после AT2.

Prepare/split в этом логе не крутились заново: датасет и frozen splits уже лежали на диске с 17:25.

---

## 18. Что не запускалось и что нельзя называть проверенным

Не в default и не запускалось:

- **INTERNAL_TEST** (31 документ, 54 answerable) — замок `allow_internal_test=false`;
- **B4** — двухстадийный shortlist B0→B1;
- **AT4-D / AT5** — дистилляция и причинное маскирование;
- **D3 / D4 / D5** — sentence-evidence и агрегация (ворота D4/D5 при этом открыты);
- **freeze-final / final-eval**;
- сетка N голов на CAL, сетка span K ∈ {2,3,5}, сетка attention K ∈ {3,5,8} — в коде есть списки, в default прогоне взяты фиксированные N=2, span K=3, attention shortlist K=5.

Не проверено по сути, хотя стадия формально бежала:

- **H3.1** (TM-supervision) — overlay пуст;
- **H3.2** (document-level target vs mismatch) — нет mismatch-меток;
- **H3.8 / H3.9** в полном виде — нет G3, головы слабоваты, N не тюнился; AT1/AT3 считались на всём unlocked, а не только на TRAIN_FOLDS;
- честное сравнение attention/span с трёхсидовым B0 в автоотчёте — разные n.

Автоотчёт `artifacts/qa_chunk_selection_v3/reports/results_v3.md` — сырой дамп. Настоящий разбор — этот файл.

---

## 19. Во что можно верить и чего нельзя утверждать

**Поддержано этим прогоном (OOF unlocked, не INTERNAL_TEST):**

1. Frozen reader по-прежнему не главный потолок: оракул F1 73.01 vs живой B0 27.36; 76% вопросов reader-solvable с какого-то positive.
2. Рецепт V2-B2 b2a, переобученный в grouped CV, остаётся лучшей допустимой системой. Новые losses и structural input его не обошли.
3. Главный top-1 distractor — ambiguous / GSRN, не HN. Hit@5 уже 79%, Hit@10 92%: запас в переранжировании top-5, не в «найти хоть какой-то positive в статье».
4. Method провален ранжированием (Hit@1 23.8% при F1|pos 73.3); metric тяжёл и для ранкера, и для ридера.
5. Компактные attention-сигнатуры frozen RoBERTa в AT1–AT4 не отделяют успешный positive от HN (AUROC 0.52 на полном unlocked, не train-only) и как единственный ранкер ломают HN@1. Даже этот завышенно оптимистичный (с точки зрения утечки) AUROC не проходит порог 0.55.
6. Span-reranking K=3 поверх B0 не поднимает Token F1.
7. Код сплитов заморожен (payload hash выше); INTERNAL_TEST не открывался; `source_label` не переписывался.
8. Ворота D4/D5 открыты: имеет смысл ставить sentence-level / aggregation как **следующий** эксперимент, не как спасение этого прогона.

**Нельзя утверждать:**

- что B0 — победитель на независимом INTERNAL_TEST;
- что гипотеза target-mismatch опровергнута или подтверждена (меток не было);
- что pairwise/listwise «не работают» в принципе — они работали без того класса negatives, ради которого их заводили;
- что attention «бесполезен для method»: exploratory Hit@1 там выше, но с HN@1 23% и без safeguard;
- что V3 улучшил V2 (другие сплиты, другая агрегация по seed);
- что можно менять production ranking.

Если когда-нибудь открывать INTERNAL_TEST, замороженный кандидат по этому прогону: **V3-B0, рецепт b2a, M=1, без B1/B2/B3/structure/attention/span**. Перед этим разумнее закрыть overlay: без verified TM главная гипотеза V3 так и останется не поставленной.

---

## 20. Что делать дальше (из фактов этого прогона, не из желаемого)

1. **Разметить очередь 400** (и по возможности расширить representative). Пока overlay пуст, B1–B3 и TM-метрики бессмысленны.
2. После overlay — повторить **только B1 vs B0** на тех же frozen folds. Не крутить сразу listwise/attention.
3. Смотреть запас **внутри top-5 B0**, не новый full-document ранкер: Hit@5 79% уже содержит большую часть positives.
4. Method: отдельно разобрать, почему attention-only поднимает Hit@1 и одновременно тащит HN. Это может быть артефакт прототипа, а не рабочий сигнал.
5. D4/D5 имеют открытые ворота, но они не заменят target-status overlay.
6. INTERNAL_TEST не открывать, пока overlay и повтор B1 не дадут другую выбранную систему — иначе тест сожжётся на том же B0, который уже известен.

---

## 21. Указатель артефактов

| путь | содержимое |
|---|---|
| `artifacts/qa_chunk_selection_v3/config_resolved.json` | полный CLI+defaults, hash `950c345e0660827a` |
| `data/prepare_summary.json` | 955 / 453 / INTERNAL_TEST locked |
| `data/v3_cv_splits.json` | frozen 5-fold, hashes |
| `data/dataset.jsonl`, `data/hn_quality_flags.jsonl` | снимок V2 |
| `annotation/queue.jsonl` | 400 чанков, скоры скрыты |
| `annotation/sampling_metadata.jsonl` | те же 400 со скорами |
| `annotation/target_status.jsonl` | **пустой** verified overlay |
| `annotation/taxonomy.json` | `v3-target-status-1` |
| `models/{b0,b1,b2,b3,structure}/run{0-4}/{42,43,44}/` | 75 чекпоинтов + `training_report.json` |
| `analysis/oof/{b0,b1,b2,b3,structure,attention_only,text_plus_attention,span_ranker}/` | OOF predictions + metrics |
| `analysis/oof_compare.json` | сводная таблица и дельты vs B0 |
| `analysis/selection.json` | safeguards, admissible, selected=b0, pareto |
| `analysis/bootstrap.json` | только ДИ Token F1 B0 |
| `analysis/positive_quality/summary.json` | оракул, bands, D4/D5 gates |
| `analysis/attention/at{0,1,2,3,4}.json` | AT-стадии |
| `analysis/attention/signatures.jsonl` | 5148 компактных сигнатур |
| `logs/qa_chunk_selection_v3.log` | wall-clock успешного прогона |
| `reports/results_v3.md` | автодамп, не этот документ |

Протокол: [`hypothesis/qa_chunk_selection_experiments_v3_plan_attention.md`](qa_chunk_selection_experiments_v3_plan_attention.md).
Предыдущий разбор V2: [`hypothesis/qa_chunk_selection_experiments_v2_results.md`](qa_chunk_selection_experiments_v2_results.md).
