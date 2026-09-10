# Результаты эксперимента V4: семь автоматических стратегий без ручной разметки

Документ фиксирует **фактические результаты** прогона `experiments/qa_chunk_selection/v4` по протоколу [`v4_experiment_plan_7_strategies.md`](v4_experiment_plan_7_strategies.md). Это не автодамп JSON (`artifacts/qa_chunk_selection_v4/reports/results_v4.md`), а полный разбор: что запускали, что реально удалось проверить, какие числа честные, а какие нельзя читать как OOF.

- Дата завершения: **2026-09-07 02:35:22** (старт `00:50:21`, wall-clock **6301.8 с ≈ 105 мин**, `auto→cuda:0`).
- Артефакты: `artifacts/qa_chunk_selection_v4/`.
- Лог: `logs/qa_chunk_selection_v4.log`.
- Production (`untie/`, `datasets/`, Phase 1) **не менялся**. Деревья E0 / V2 / V3 / V3.1 prep / auto-silver / D4–D5 **не перезаписывались**.
- Gold **не использовался в runtime ranking** V1/V2/V5/V6. У V3 (utility mining) gold Token F1 разрешён протоколом **только на TRAIN** для поиска `utility_positive_candidate`; это не TE/TM-метка. У V4 diagnostic gold не нужен для ранжирования live-пайплайна: ветка диагностическая.

**INTERNAL_TEST не открыт.** Ниже нет ни одного QA-числа с этого сплита, кроме описи: 78 locked records.

Автодамп `reports/results_v4.md` и `analysis/selection.json` объявили победителем **V3-A3**. Это **нельзя** принимать как научный итог: V3-A3 и V7 оценивались на всех 453 вопросах после обучения на train-фолдах **одного** outer run, то есть большая часть «выигрыша» — in-sample. Честный разбор ниже.

---

## 0. Короткий итог

На честном сравнении с историческим **V3-B0 (B0-live, max5, три seed)** автоматический цикл **не дал нового full-document ranker**, который можно ставить вместо B0. Две ветки реально сдвинули downstream QA **без смены состава B0 top-K**:

| Что можно утверждать | Система | n | Token F1 | ΔF1 vs B0 (qid+seed) | 95% CI | Hit@1 |
|---|---|---:|---:|---:|---|---:|
| Живой ranking-reference | **B0-live** | 1359 | **27.36** | — | 24.30–30.56 | **36.72%** |
| Лучшая answer-aggregation | **V2-A4** | 1359 | **30.26** | **+2.90** | 1.32–4.55 | 36.72% (как B0) |
| Лучший shortlist-rerank | **V6** | 1359 | **35.04** | **+7.69** | 4.79–10.59 | **42.60%** |
| Автодамп «победитель» | V3-A3 | 453 | 38.42 | +11.96 **на смеси train+OOF** | — | 55.85% |
| То же V3-A3, только EVAL run 0 | V3-A3 eval | 90 | 29.90 | **−0.78** vs B0-s42 | −8.24–6.48 | 36.67% |

**Главное.**

1. Узкое место по-прежнему top-1 внутри уже хорошего B0 shortlist (Hit@5 B0 = 79.03%). Это V4 и проверял.
2. **KeyBERT / LOSCO (V1)** ranking не улучшил. Несколько абляций схлопнулись в копию B0 (`λ=0` на CAL).
3. **Агрегация ответов (V2-A4)** даёт устойчивый прирост Token F1 и EM **без смены выбранного чанка**: ранжирование остаётся B0, меняется только то, какой span из top-5 считается ответом.
4. **Reader-centered rerank (V6)** — самый большой честный прирост среди трёхсидовых систем: Hit@1 36.72% → 42.60%, Amb@1 и GSRN@1 падают, HN@1 почти не растёт (+0.29 п.п. при `epsilon_HN=2` п.п.). Веса взяты с одного CAL-фолда, но прирост держится и на run-0 EVAL (`ΔF1 +6.68`, CI 0.75–12.90).
5. **V3-A3 нельзя называть победителем.** На 272 train-вопросах run 0 Token F1 = 43.27 (Hit@1 66.5%); на 90 EVAL-вопросах — 29.90 против 30.67 у B0 seed 42. Автоселекция взяла смешанную цифру 38.42.
6. **V4 ownership diagnostic не прошёл gate** (AUROC 0.543, pairwise 0.543 < 0.55). Поэтому V7 **не получил** anchored synthetic negatives и на OOF **хуже** B0 (`ΔF1 −5.96`, CI −11.61…−0.56).
7. **V5 learned head** на полном пуле `ΔF1 +1.97`, но 95% CI включает ноль; на OOF run 0 `+0.55` с CI через ноль. Плюс Hit@5/10 сломались из-за смешения шкал shortlist vs остальных чанков. Softmax-only (V5-A0/A1) не успех.

OOF V4 — это **model-selection evidence на unlocked пуле**, не confirmatory INTERNAL_TEST. B0 остаётся историческим ranking-reference. Кандидаты на следующий цикл — **V6** и более простой **V2-A4**, не V3-A3.

---

## 1. Что это за эксперимент

UnTIE — extractive QA по SciREX: для aspect-вопроса (task / method / material / metric) нужно выбрать evidence-чанк **текущей** статьи и вытащить span. После E0→V2→V3 живой baseline — V3-B0:

```text
Token F1 ≈ 27.36
Hit@1 ≈ 36.72%
Hit@5 ≈ 79.03%
Hit@10 ≈ 92.20%
HN@1 ≈ 4.86%
Ambiguous@1 ≈ 50.85%
GSRN@1 ≈ 33.92%
F1 | selected positive ≈ 61.70
best-positive oracle ≈ 73.01
```

V4 не продолжает неудачный QA-silver как источник `target_evidence / target_mismatch` и **не требует ручной overlay-разметки**. Семь заранее объявленных стратегий используют уже доступные сигналы: B0 ranking, original positives/HN/easy, frozen extractive QA, document-level consistency, cross-chunk agreement, synthetic controls.

Концентрация — second-stage внутри `B0 top-5` (primary K=5; top-10 считался в union QA cache), не новый full-document retriever. Исключение по факту реализации: V3-A3 и V7 скорят **все чанки документа**, не только shortlist.

Ридер: frozen Phase 1 extractive QA. KeyBERT: `sentence-transformers/all-MiniLM-L6-v2`, top-10 фраз, ngram 1–3, без MMR. Cross-encoder trainable веток: `cross-encoder/ms-marco-MiniLM-L6-v2`. BERTScore отключён. `source_label` не переписывается.

---

## 2. Вселенная, сплиты, reference

Те же frozen V3 CV, без новых outer folds.

| Величина | Значение | Где лежит |
|---|---|---|
| Unlocked answerable вопросы | **453** | `source_hashes.json` |
| Документы | **240** | V3 `data/v3_cv_splits.json` |
| INTERNAL_TEST records | 78, locked | `source_hashes.json` |
| Folds | 5, document-level, seed 42 | те же, что V3 |
| `dataset` hash | `3be581a8dfb015ddbc5a6fd02d411a3ab0aa7dadc4ac06c7833618135844dc9b` | совпадает с V3.1 prep |
| `splits` hash | `aa34bb4f4f6c31d318b7f472b166a5e3bda1ca6dd03891d6b07efef211bc0cf7` | то же |
| B0-live | исторический V3-B0 max5 OOF, **не переобучался** | `reference/b0_live/` |
| B0 seeds | 42, 43, 44 | n=1359 = 453×3 |
| B0-extended (max8) | **не запускался** | как и требовал план |

Роли одного outer run (run 0), которыми код **фактически** пользовался для trainable/CAL:

| Роль run 0 | Folds | Документы | Вопросы |
|---|---|---:|---:|
| TRAIN | 0, 1, 2 | 144 | 272 |
| CALIBRATION | 3 | 48 | 91 |
| EVAL | 4 | 48 | 90 |

Протокол требовал **повторить** 3/1/1 для каждого из пяти outer run. Реализация обучала V3/V5/V7 и подбирала λ/веса **только на run 0**, затем применяла модель/формулу ко всему unlocked пулу. Это центральное отклонение, без которого цифры V3-A3 нельзя читать.

B0 по seed (из `reference/b0_live/predictions_scored.jsonl`, это копия V3 OOF):

| Seed | Token F1 | Hit@1 |
|---:|---:|---:|
| 42 | 26.46 | 36.20% |
| 43 | 27.94 | 37.97% |
| 44 | 27.68 | 35.98% |
| pooled | **27.36** | **36.72%** |

Кластерный бутстреп B0 Token F1 (document_id, 5000 реплик): среднее 27.39, 95% ДИ **24.30–30.56** — совпадает с отчётом V3.

B0 по аспектам (pooled 1359):

| Aspect | n | Token F1 | Hit@1 | HN@1 | Amb@1 | GSRN@1 |
|---|---:|---:|---:|---:|---:|---:|
| material | 378 | 30.94 | 51.85% | 2.38% | 39.95% | 28.57% |
| method | 345 | 18.66 | 23.77% | 6.96% | 57.68% | 40.87% |
| metric | 195 | 17.31 | 29.23% | 4.62% | 56.41% | 32.31% |
| task | 441 | 35.54 | 37.19% | 5.44% | 52.38% | 33.79% |

`method` по-прежнему главный ranking-bottleneck.

---

## 3. Что реально запускалось

Стадии успешного прогона (из лога, не default `phase1` лаунчера):

```text
init, freeze-b0, shortlist, clusters, keybert, qa-topk,
v1, v2, v3, v4, v5, v6, v7, compare, report
```

`smoke=False`, `skip_qa=False`, `allow_internal_test=False`, `inner_crossfit=false`, `primary_k=5`.

| ID | Смысл | Trainable | Как считали | n в метриках |
|---|---|---|---|---:|
| B0-live | исторический V3-B0 | нет (заморожен) | 3 seed | 1359 |
| V1-A0…A6 | KeyBERT compatibility на top-5 | нет (`λ` с CAL) | 3 B0 seed | 1359 |
| V2-A0…A5 | агрегация ответов на top-5 | нет | 3 B0 seed | 1359 |
| V3-A3 | listwise + utility positives | да, seed 42 | **все 453, 1 seed** | 453 |
| V4 | ownership diagnostic | нет | 272 train-пары run 0 | — |
| V5-A0…A3 | калибровка shortlist | A3: linear head на run-0 TRAIN | 3 B0 seed | 1359 |
| V6 | reader agreement rerank | нет (`α,β,γ` с CAL) | 3 B0 seed | 1359 |
| V7 | close-FP reranker без synthetics | да, seed 42 | **все 453, 1 seed** | 453 |

Не запускалось, хотя план это допускал/описывал:

- trainable V1 head и inner cross-fit;
- V3-A0/A1/A2/A4 (historical B3, BCE, loss-guided и пр.) — только **V3-A3**;
- V5-A4 MLP;
- trainable V6 head;
- V7-A2+ anchored synthetics (закрыты G4);
- D4 (вторичная ветка, после более сильного top-1);
- B0-extended max8;
- seeds 43/44 для trainable моделей;
- пять outer run для trainable/CAL, не один.

Phase 0 cache: 8947 чанков KeyBERT, 38540 phrase embeddings, frozen QA на **5176** парах `(question, chunk)` = union top-K по трём B0 seed. Cache hit rate в конце: **0.248**.

---

## 4. Критические отклонения от протокола

Их нужно держать в голове **до** любой таблицы.

### 4.1. Trainable модели не OOF

V3-A3 и V7 обучены на `records_for_role(..., run=0, role="train")` (272 вопроса) и CAL early-stop на run-0 CAL (91), затем `score` по **всем** 453 unlocked, включая свои TRAIN документы.

V5 linear head — то же: fit на run-0 TRAIN, apply на весь пул.

Это не «чуть оптимистичный OOF». Для V3-A3 in-sample кусок даёт Hit@1 **66.5%** при B0-s42 **34.6%** на тех же 272 вопросах.

### 4.2. Один outer run вместо пяти

λ для V1-A3/A4/A5, веса V6, IDF KeyBERT, V4-пары, utility mining — всё с run 0. Гиперпараметры не крутились по EVAL, но и не были per-fold frozen procedure из плана.

### 4.3. `inner_crossfit=false`

План требовал inner grouped cross-fitted B0 shortlist для trainable second-stage. Конфиг явно фиксирует: этот цикл берёт исторический V3-B0 OOF (V3.1 ranking не запускался). Для документов run-0 TRAIN B0-скоры всё же OOF относительно **B0**, но не относительно V5/V3/V7 голов.

### 4.4. Автосравнение bootstrap сломано

`paired_delta` индексирует baseline по одному `question_id`. У B0 три seed → в словарь попадает **последний** (seed 44). Каждая из трёх строк кандидата сравнивается с B0-s44. Поэтому даже у **идентичных** систем (V1-A0 = B0) автодамп пишет `mean_delta ≈ −0.32`, win/lose 72/82.

Ниже везде, где сказано «честный ΔF1», сравнение по `(question_id, seed)`. Для V3/V7 — только B0 seed 42.

### 4.5. `pairwise_clean_hn` в V4-метриках ≠ 75.14% из отчёта V3

Исходный V3-B0 jsonl: `pairwise_clean_hn = 0.7514` (это те самые **75.14%**). После V4 `chunk_prediction` / пересчёта scores то же поле у B0-live в `reference/b0_live/metrics.json` становится **0.4068**. Primary QA (Token F1, Hit@k, HN@1) при этом совпадают с V3 до последнего знака.

Safeguard `pairwise_clean ≥ B0 − 0.03` в compare работал на **сломанной** шкале (~0.41). Все системы «прошли». Абсолютные pairwise V4 **нельзя** сопоставлять с 75% из V3. Относительно друг друга внутри V4 поле внутренне согласовано, но как hard gate оно ослаблено.

### 4.6. Автоселекция = max Token F1 среди admissible

Так в `pipeline.stage_compare`. Поэтому выиграла утекающая V3-A3 с n=453 против трёхсидовых n=1359. `selection.json`: `"selected": "V3-A3"`, `"b0_live_kept": true`. B0 не выкинут, но «selected» для следующего цикла брать оттуда нельзя.

### 4.7. Порог reader-utility сравнивался не в той шкале

План задаёт пороги `F1 ∈ {0.5, 0.8, 1.0}` (единичная шкала). Конфиг зафиксировал `utility_f1=0.8`. Функция `token_f1` (SQuAD) возвращает **0–100**, как и Token F1 27.36 в таблицах. Код делает `f1 < 0.8` на шкале 0–100, то есть отсекает только ответы с F1 < 0.8 пункта. В `v3_utility_candidates.jsonl` минимум 4.17, **294/294** кандидатов прошли; при пороге 80 осталось бы **31**. Слабые «utility positives» — не те high-utility contexts, которые описывал план.

---

## 5. Сводная таблица (как посчитал пайплайн)

Это **сырые** `metrics.json`. V3-A3 и V7 здесь **несопоставимы** с остальными по n и по OOF.

| Система | n | Token F1 | EM | Hit@1 | MRR | HN@1 | Amb@1 | GSRN@1 | Hit@5 | F1\|pos | GM-nonpos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0-live** | 1359 | **27.36** | 20.75% | **36.72%** | 0.542 | 4.86% | 50.85% | 33.92% | 79.03% | 61.70 | 4.64% |
| V1-A0 | 1359 | 27.36 | 20.75% | 36.72% | 0.542 | 4.86% | 50.85% | 33.92% | 79.03% | 61.70 | 4.64% |
| V1-A1 | 1359 | 27.36 | 20.75% | 36.72% | 0.542 | 4.86% | 50.85% | 33.92% | 79.03% | 61.70 | 4.64% |
| V1-A2 | 1359 | 27.08 | 20.24% | 35.32% | 0.535 | 5.15% | 52.47% | 33.55% | 79.03% | 62.12 | 5.00% |
| V1-A3 | 1359 | 27.21 | 20.53% | 36.28% | 0.538 | 4.86% | 52.02% | 33.70% | 79.03% | 62.37 | 4.56% |
| V1-A4 | 1359 | 27.36 | 20.75% | 36.72% | 0.542 | 4.86% | 50.85% | 33.92% | 79.03% | 61.70 | 4.64% |
| V1-A5 | 1359 | 27.36 | 20.75% | 36.72% | 0.542 | 4.86% | 50.85% | 33.92% | 79.03% | 61.70 | 4.64% |
| V1-A6 | 1359 | 26.79 | 19.94% | 36.06% | 0.537 | 4.71% | 52.10% | 34.58% | 79.03% | 61.05 | 4.86% |
| V2-A0 | 1359 | 27.36 | 20.75% | 36.72% | 0.542 | 4.86% | 50.85% | 33.92% | 79.03% | 61.70 | 4.64% |
| V2-A1 | 1359 | 28.05 | 23.77% | 36.72% | 0.542 | 4.86% | 50.85% | 33.92% | 79.03% | 41.87† | 3.46% |
| V2-A2 | 1359 | 28.05 | 23.77% | 36.72% | 0.542 | 4.86% | 50.85% | 33.92% | 79.03% | 41.87† | 3.46% |
| V2-A3 | 1359 | 30.11 | 24.58% | 36.72% | 0.542 | 4.86% | 50.85% | 33.92% | 79.03% | 59.09† | 4.86% |
| **V2-A4** | 1359 | **30.26** | **24.72%** | 36.72% | 0.542 | 4.86% | 50.85% | 33.92% | 79.03% | 59.29† | 4.86% |
| V2-A5 | 1359 | 27.75 | 21.63% | 36.72% | 0.542 | 4.86% | 50.85% | 33.92% | 79.03% | 41.35† | 3.90% |
| V3-A3 | 453 | 38.42‡ | 32.23% | 55.85% | 0.693 | 0.66% | 35.98% | 23.84% | **86.75%** | 61.72 | 4.19% |
| V5-A0 | 1359 | 26.19 | 19.87% | 33.41% | 0.520 | 5.96% | 49.67% | 31.42% | 76.09% | 65.47 | 4.05% |
| V5-A1 | 1359 | 27.36 | 20.75% | 36.72% | 0.542 | 4.86% | 50.85% | 33.92% | 79.03% | 61.70 | 4.64% |
| V5-A2 | 1359 | 27.06 | 21.12% | 34.81% | 0.536 | 4.64% | 51.14% | 33.70% | 79.03% | 65.09 | 4.19% |
| V5-A3 | 1359 | 29.33 | 23.99% | 38.26% | 0.544 | 4.93% | 49.45% | 34.58% | 75.06% | 65.49 | 5.00% |
| **V6** | 1359 | **35.04** | **25.46%** | **42.60%** | 0.572 | 5.15% | 44.30% | 26.64% | 79.03% | 66.13 | 7.36% |
| V7 | 453 | 24.80 | 18.76% | 35.54% | 0.528 | 4.64% | 54.75% | 32.45% | 75.72% | 57.07 | 5.08% |

† У V2 `selected_chunk` остаётся B0 top-1 (Hit@1 не меняется), а ответ берётся из группы чанков. `F1|pos` считается по label выбранного B0-чанка и **новому** ответу — это не тот же смысл, что у B0.
‡ Смесь in-sample TRAIN и OOF; см. §9.

Hit@5/10 у shortlist-rerankers V1/V2/V6 совпадают с B0, как и должно быть, если состав top-K не меняется. У V3-A3/V7/V5-A0/V5-A3 Hit@5 **другой**: они либо скорят весь документ, либо смешивают шкалы shortlist и остальных чанков.

Честный Δ Token F1 (match qid+seed; кластерный бутстреп по `document_id`, 5000):

| Система | ΔF1 | 95% CI | Win / Lose / Tie |
|---|---:|---|---:|
| V1-A0/A1/A4/A5, V2-A0, V5-A1 | 0.00 | — | 0 / 0 / 1359 |
| V1-A2 | −0.28 | — | 128 / 125 / 1106 |
| V1-A3 | −0.15 | −1.30–1.01 | 64 / 65 / 1230 |
| V1-A6 | −0.57 | — | 28 / 35 / 1296 |
| V2-A1 = V2-A2 | +0.69 | −2.41–3.80 | 239 / 249 / 871 |
| V2-A3 | +2.75 | 1.18–4.40 | 78 / 49 / 1232 |
| **V2-A4** | **+2.90** | **1.32–4.55** | 79 / 48 / 1232 |
| V2-A5 | +0.39 | — | 252 / 254 / 853 |
| V5-A0 | −1.17 | — | 248 / 270 / 841 |
| V5-A2 | −0.30 | — | 195 / 214 / 950 |
| V5-A3 | +1.97 | **−0.18–4.04** | 145 / 139 / 1075 |
| **V6** | **+7.69** | **4.79–10.59** | 312 / 149 / 898 |

---

## 6. Версия 1 — KeyBERT document compatibility

Гипотеза: чанк, согласованный с остальным документом (LOSCO, не простой LOCO), лучше тематически правдоподобного, но локально изолированного distractor.

CAL (run 0, 91 вопрос, B0 seed 42) выбрал:

```text
V1-A3 λ = 0.25
V1-A4 λ = 0.0
V1-A5 λ = 0.0
```

`λ=0` означает: на CAL добавка LOSCO+support / IDF **не улучшила** Hit@1, формула схлопнулась в B0. Проверка: V1-A4 и V1-A5 — **0** смен top-1 относительно B0. V1-A0 (контроль = B0) и V1-A1 (global profile) тоже 0 смен.

V1-A3 (основной LOSCO) сменил top-1 в **272 / 1359** строках (это 272 предсказания по всем фолдам×seed, не «все 272 train-вопроса run 0» — число совпало случайно) и **чуть проиграл**: Token F1 27.21 vs 27.36, Hit@1 36.28% vs 36.72%, честный ΔF1 −0.15, CI через ноль. На самом CAL-фолде, где подбирали λ, ΔF1 **−2.33** (CI −4.84…−0.01) — отбор λ не помог даже там.

LOCO (V1-A2) хуже LOSCO по F1, но оба ≤ B0. Абляцию «прежний gain был self-overlap artifact» честно поставить нельзя: **LOSCO не дал gain**. Aspect-conditioned V1-A6 ещё хуже (26.79).

Переходы V1-A3 vs B0 (qid+seed): GSRN→positive 28, positive→HN 3, ambiguous→positive 37, positive→ambiguous 50. Нет картины «достаём GSRN без поломки positives».

**G1 — не пройден.** KeyBERT как compact document profile в этой постановке (fusion на B0 logits, K=5, MiniLM фразы) не даёт рабочего ranking-сигнала. Это не доказательство, что document compatibility бесполезен вообще, а отрицательный результат **конкретной** V1-матрицы.

---

## 7. Версия 2 — answer agreement + aggregation

Гипотеза: в B0 top-5 original positive уже есть у ~79% вопросов; можно собрать ответ по согласованным span, не переставляя чанки.

Реализация не меняет B0 ranking: Hit@1/5/10, HN@1, Amb@1, GSRN@1 **идентичны B0**. Меняется `qa_answer`. Provenance rule: supporting chunk с максимальным B0 среди победившей answer-группы. Проверка: V2-A4 `selected_chunk_id` совпал с B0 во **всех 1359** строках. Честные label-transitions нулевые.

| Вариант | Смысл | Token F1 | EM | pos-support | mean support | GM-nonpos |
|---|---|---:|---:|---:|---:|---:|
| V2-A0 | B0 top-1 answer | 27.36 | 20.75% | 36.72% | 1.00 | 4.64% |
| V2-A1 | частота ответов | 28.05 | 23.77% | 35.98% | 1.48 | 3.46% |
| V2-A2 | source-cluster count | 28.05 | 23.77% | 35.98% | 1.48 | 3.46% |
| V2-A3 | сумма B0-весов | 30.11 | 24.58% | 40.84% | 1.47 | 4.86% |
| **V2-A4** | max B0 + log(1+N_clusters) | **30.26** | **24.72%** | **40.77%** | 1.47 | 4.86% |
| V2-A5 | relative QA + support | 27.75 | 21.63% | 35.54% | 1.36 | 3.90% |

V2-A1 и V2-A2 **бит-в-бит одинаковы** (F1 28.047166562294464): при среднем ~1.48 кластера на ответ частота и cluster-count почти не расходятся.

V2-A1/A2 на OOF run 0 (cal+eval) **в минусе** (ΔF1 −2.84, CI через ноль, lose>win). Частотный majority на чужих фолдах вредит — ровно тот риск из плана (related work / baseline повторяет неправильный ответ).

V2-A4 держится:

- весь пул ΔF1 **+2.90**, CI **1.32–4.55**, win/lose 79/48;
- run-0 EVAL ΔF1 **+4.14**, CI 0.88–8.03;
- run-0 cal+eval ΔF1 **+2.55**, CI 0.26–4.99;
- ΔEM **+3.97 п.п.** (71 win / 17 lose);
- Char F1 0.521 → 0.554, ROUGE-L 0.339 → 0.378;
- `final_answer_has_original_positive_support` 36.72% → 40.77% (winning answer чаще опирается на original positive, чем B0 top-1);
- `GoldMatchFromNonOriginalPositive` 4.86% vs B0 4.64% — прирост F1 **не** объясняется массовым «угадыванием gold с чужого чанка»;
- mix support (сумма по 1359): positive 949, ambiguous 817, easy 140, HN 95.

На подмножестве «B0 top-1 не positive» (860 строк) средний Token F1 B0 7.43 → V2-A4 13.41; лучше в 67 случаях. Это diagnostic subset из плана, не runtime-критерий.

По аспектам ΔF1 V2-A4 vs B0 (qid+seed): method **+4.89**, material +3.05, task +1.91, metric +1.33. Агрегация особенно помогает там, где ranking слабый (`method`), не ломая Hit@1.

**G2 — пройден для V2-A4/A3**, не для частотного A1/A2. V2 не attribution method (ранжирование то же). Комбинации с V6 план разрешает после G6; в этом прогоне их не было.

---

## 8. Версия 3 — reader-utility augmentation + listwise

Гипотеза: frozen reader на TRAIN может найти **полезные** ambiguous-контексты (Token F1 ≥ 0.80 vs gold) и добавить их в listwise с весом `η=0.50`, не называя их target evidence.

Сделано только **V3-A3**. Utility mining прошёл по **2256** ambiguous-чанкам run-0 TRAIN (лог `utility_qa`). В файл попали **294** кандидата на 134 вопросах; **196** в новой source-region (`overlaps_original_positive=false`). Средний Token F1 vs train gold **40.2** (min 4.2, max 100). Из-за ошибки шкалы (§4.7) порог `0.8` почти ничего не отсёк: F1≥80 имеют только **31/294**, F1≥50 — 109/294. Это не «high-utility ≥ 0.80» из плана. `n_groups=272` (все train-вопросы run 0). Обучение 5 эпох, best = epoch 5, CAL `pairwise_clean_hn=0.759` (это **train-time** CAL-метрика answerability, не V4 compare-шкала).

Затем модель скорит **все чанки всех 453 вопросов**. Hit@5 86.75% vs 80.57% у B0-s42: это уже не shortlist-rerank, а новый full-document ranker.

Разрез по роли run 0 vs B0 seed 42:

| Срез | n | V3-A3 F1 | B0-s42 F1 | ΔF1 | 95% CI | V3 Hit@1 | B0 Hit@1 |
|---|---:|---:|---:|---:|---|---:|---:|
| TRAIN (in-sample) | 272 | **43.27** | 23.48 | **+19.79** | 14.41–25.13 | **66.54%** | 34.56% |
| CAL | 91 | 32.34 | 31.21 | +1.13 | −5.24–7.42 | 42.86% | 42.86% |
| EVAL | 90 | 29.90 | 30.67 | **−0.78** | −8.24–6.48 | 36.67% | 34.44% |
| OOF cal+eval | 181 | 31.12 | 30.94 | +0.18 | −4.57–4.97 | 39.78% | 38.67% |
| все 453 (то, что в автодампе) | 453 | 38.42 | 26.46 | +11.96 | 8.20–15.90 | 55.85% | 36.20% |

На TRAIN HN@1 = 0% и Amb@1 = 28.3% — модель выучила train-документы. На EVAL HN@1 = 0% при F1 **ниже** B0: «красивый» HN не означает обобщение.

Нельзя писать «мы автоматически разметили target evidence». Даже как «utility weak supervision помогло retrieval» утверждение **не держится на OOF**.

**G3 — не пройден** на честном срезе. Автовыбор V3-A3 — ошибка селекции, не результат.

---

## 9. Версия 4 — ownership diagnostic

Не ranker. 272 пары run-0 TRAIN: same aspect, другой документ, mean KeyBERT-profile similarity **0.722**. Probe: frozen QA на `anchor(A)+chunk`, свой positive vs чужой positive.

| | Значение |
|---|---:|
| n target = n mismatch | 272 |
| AUROC | **0.543** |
| AUPRC | 0.537 |
| pairwise target > mismatch | **0.543** |
| mean anchored target | −7.17 |
| mean anchored mismatch | −7.61 |
| gate (`≥ 0.55`) | **false** |
| plain (без anchor) pairwise | 0.544 |

Anchor **не** создал разделимости: plain и anchored одинаково около случайного. Средний anchored_score сильно отрицательный и у своих, и у чужих — зонд не «ломает» QA-скор в полезную сторону.

**G4 — не пройден.** В логе: `WARN V7 skips anchored synthetics because V4 gate failed or was not run`. Это правильное закрытие V7-A2+.

E14 из таксономии ошибок плана («ownership probe non-separating») — основной исход V4.

---

## 10. Версия 5 — global calibration

Гипотеза: лёгкая голова по **относительным** frozen-признакам внутри top-K лучше, чем независимый QA-скор или post-hoc softmax.

Признаки: `b0_score, b0_rank, qa_raw, qa_margin, rel_qa, rel_b0, span_len, cand_len, rel_pos`. Head: sklearn logistic, `max_iter=200`, в логе **ConvergenceWarning** (не сошёлся). Коэффициенты:

```text
b0_score   +1.687
b0_rank    −0.099
qa_raw     +0.141
qa_margin   0.0      # фактически выключен
rel_qa     −0.411
rel_b0     −0.151
span_len   −0.006
cand_len   +0.0008
rel_pos    +0.069
intercept  −1.910
```

Доминирует B0; relative QA входит с **отрицательным** весом. Softmax-only V5-A1 — идентичен B0 (0 смен top-1), как и задумано invariance control. V5-A0 (сырой QA на shortlist) **хуже** B0 (F1 26.19, Hit@1 33.41%, HN@1 5.96%) и выталкивает top-1 **за пределы B0 top-5 в 173 / 1359** случаях.

Почему Hit@5 у A0/A3 не равен B0: `apply_v5` ставит новые скоры только shortlist, остальным оставляет **исходный B0-скор**. Шкалы смешиваются, не-shortlist может всплыть. У V5-A3 selected вне B0 top-5: **95 / 1359**. Это нарушение «rerank inside top-K ⇒ Hit@5 неизменен».

V5-A3 vs B0:

| Срез | n | ΔF1 | 95% CI | Hit@1 V5 |
|---|---:|---:|---|---:|
| все | 1359 | +1.97 | **−0.18–4.04** | 38.26% |
| run-0 TRAIN (in-sample head) | 816 | +2.92 | 0.06–5.69 | 37.25% |
| run-0 CAL | 273 | −2.18 | −5.65–0.78 | 44.32% |
| run-0 EVAL | 270 | +3.32 | −2.59–8.93 | 35.19% |
| OOF cal+eval | 543 | +0.55 | −2.73–3.78 | 39.78% |

**G5 — не пройден как рабочий ranker.** Learned head > softmax (A3 лучше A0/A1/A2 по F1), но не лучше B0 устойчиво на OOF, и ломает upstream recall.

---

## 11. Версия 6 — reader-centered rerank

Гипотеза: полезный чанк — тот, на котором frozen reader согласуется с другими кандидатами shortlist, а не просто высокий B0.

Нетрансформируемая формула, веса с run-0 CAL (сетка `FUSION_LAMBDAS`):

```text
α = 2.0   # B0 score
β = 0.25  # exact_freq + mean_token_f1 + majority
γ = 0.5   # source-cluster support of agreeing chunks
```

Shortlist K=5; Hit@5/10 = B0. Top-1 сменился в **760 / 1359**. HN@1 5.15% vs 4.86% (дельта **+0.29 п.п.** < `epsilon_HN=2` п.п.).

| Срез | n | V6 F1 | ΔF1 | 95% CI | Hit@1 |
|---|---:|---:|---:|---|---:|
| все (3 seed) | 1359 | **35.04** | **+7.69** | **4.79–10.59** | **42.60%** |
| run-0 TRAIN | 816 | 33.89 | +9.48 | 5.51–13.59 | 42.65% |
| run-0 CAL (там подбирали веса) | 273 | 37.23 | +3.32 | −1.52–8.18 | 45.42% |
| run-0 EVAL | 270 | 36.32 | **+6.68** | **0.75–12.90** | 39.63% |
| OOF cal+eval | 543 | 36.78 | **+4.99** | **1.14–8.96** | 42.54% |

CAL, где крутили веса, **не** самый жирный срез (CI через ноль). EVAL и cal+eval держат плюс. Это не тот паттерн, что у V3-A3.

Ещё QA: EM 20.75% → 25.46% (Δ +4.71 п.п.), Char F1 0.521 → 0.570, ROUGE-L 0.339 → 0.432, attribution-aware F1 26.80 → 33.55, F1\|pos 61.70 → 66.13.

Source-selection: Amb@1 50.85% → 44.30%, GSRN@1 33.92% → 26.64%. Честные переходы vs B0 (qid+seed): **GSRN→positive 124**, ambiguous→positive 183, positive→ambiguous 98, **positive→HN 19** (у идентичных систем 0; у V1-A3 — 3). Есть цена: часть gold-positive уходит в ambiguous/HN, но нет прироста за счёт обвала в HN@1. `GoldMatchFromNonOriginalPositive` 7.36% vs 4.64% — часть QA-gain с не-positive чанков; это нужно помнить, но Hit@1 original positive при этом **вырос**.

По аспектам ΔF1 vs B0: **task +12.11**, method +7.78, metric +4.33, material +4.17. План ждал V6 скорее на material; фактически сильнее всего task/method.

Отличие от V2: V2 не меняет чанк, V6 меняет. V6 F1 35.04 > V2-A4 30.26 на том же пуле.

**G6 — пройден** относительно B0 и относительно V2-A4. Комбинацию V2+V6 этот прогон не строил.

Оговорки: веса с одного CAL-фолда; три B0 seed, но не отдельный trainable seed; pairwise V4-шкалы слабые как gate. Для следующего цикла V6 — главный честный кандидат second-stage, не замена B0 как full-document retriever.

---

## 12. Версия 7 — close-FP reranker

После провала G4 обучен только natural HN/easy shortlist-reranker (V7-A1 по смыслу), **без** anchored synthetics. `training_report.use_anchor = false`. 5 эпох, best epoch 5, CAL pairwise_clean_hn 0.693. Скорит все чанки, n=453, seed 42.

| Срез | n | V7 F1 | B0-s42 F1 | ΔF1 | 95% CI | Hit@1 |
|---|---:|---:|---:|---:|---|---:|
| TRAIN | 272 | 24.69 | 23.48 | +1.21 | −2.58–4.86 | 38.24% |
| CAL | 91 | 23.53 | 31.21 | **−7.68** | −15.27…−0.51 | 34.07% |
| EVAL | 90 | 26.45 | 30.67 | −4.22 | −12.44–3.84 | 28.89% |
| OOF cal+eval | 181 | 24.98 | 30.94 | **−5.96** | **−11.61…−0.56** | 31.49% |
| все 453 | 453 | 24.80 | 26.46 | −1.66 | −4.72–1.45 | 35.54% |

На OOF Hit@5 тоже просел (65.6–70.3% vs ~76% у B0). GSRN→positive всего 20 при 49 positive→ambiguous.

**G7 — не пройден.** Без working target-anchor synthetic branch закрыт правильно; natural-only rerank B0 shortlist не вытянул.

---

## 13. Gates — сводка

| Gate | Вопрос | Итог |
|---|---|---|
| G1 KeyBERT | LOSCO/support без HN-деградации | **нет** |
| G2 aggregation | F1/EM ↑ и не wrong-source majority | **да, V2-A4/A3** (не A1/A2) |
| G3 utility | OOF лучше B0/B3, HN жив | **нет** (in-sample + порог F1 в чужой шкале) |
| G4 ownership | anchor(A)+pos(A) > anchor(A)+pos(B) | **нет** (0.543 < 0.55) |
| G5 calibration | learned head > frozen baselines **и** B0 | softmax нет; vs B0 OOF нет |
| G6 reader-centered | rerank > простая агрегация | **да** (V6 > V2-A4 > B0) |
| G7 close-FP | F1 + HN + runtime + stability | **нет** |

Safeguard HN для трёхсидовых chunk-систем: никто не превысил B0+2 п.п. V6 +0.29 п.п. V5-A0 +1.10 п.п. (на грани по духу, формально < 2).

---

## 14. Аспекты: что сдвинулось

План: method — ranking/GSRN; material — aggregation/V5/V6; metric — оба bottleneck; task — rerank/aggregation.

Факт на честных трёхсидовых системах:

| Aspect | B0 F1 | V2-A4 F1 | V6 F1 | B0 Hit@1 | V6 Hit@1 |
|---|---:|---:|---:|---:|---:|
| material | 30.94 | 33.99 | 35.11 | 51.85% | 55.29% |
| method | 18.66 | 23.55 | 26.44 | 23.77% | 35.07% |
| metric | 17.31 | 18.64 | 21.64 | 29.23% | 33.85% |
| task | 35.54 | 37.44 | 47.65 | 37.19% | 41.50% |

V6 сильнее всего поднимает **task** (+12.1 F1) и **method** (+7.8 F1) — как раз ranking-дыры V3. Metric остаётся слабым (21.64). Material и так лучший B0 Hit@1; там скромнее.

V3-A3 «красивые» aspect-числа (method Hit@1 45%, metric 62%) смешаны с train leak и в селекцию не идут.

---

## 15. Эффективность

Из лога, отдельных peak-GPU/RAM файлов пайплайн не писал.

| Этап | Время (накопленно) | Заметка |
|---|---|---|
| freeze-b0 + shortlist + clusters | ~8 с | 453 документов, overlap 0.6 |
| KeyBERT phrases + embed | ~335 с (до 00:55:56) | 8947 чанков, 38540 фраз, cuda:0 |
| frozen QA union(top-K) | +1200 с (до 01:15:55) | 5176 вызовов ридера |
| V1 все абляции | до 02:16:30 | ~1 час; LOSCO features на 3 seed |
| V2 | ~30 с | детерминированная агрегация |
| V3 utility QA + train | до 02:26:37 | 2256 ambiguous TRAIN + 5 эпох |
| V4 272 probes | ~4 мин | 2 QA на пару |
| V5 | ~20 с | logistic + apply |
| V6 | ~4 мин | CAL grid + apply 3 seed |
| V7 train+eval | ~30 с | 5 эпох, без synthetics |
| **весь прогон** | **6301.8 с** | cache hit rate 0.248 |

Для нового документа KeyBERT не «бесплатный cache»: one-time preprocessing в этом прогоне ≈ 5.5 мин на 240 документов (~1.4 с/док на фразы+эмбеддинги, грубо, без амортизации QA). Амортизация на аспект не считалась отдельно — в артефактах нет per-aspect KeyBERT cost dump.

---

## 16. Что автодамп сделал не так (чтобы не повторить)

1. Выбрал V3-A3 по Token F1 на смеси train+eval.
2. Сравнил n=453 с n=1359 как будто это один дизайн.
3. Bootstrap по `question_id` без seed — даже копии B0 «проигрывают» на −0.32 F1.
4. Объявил все системы admissible по pairwise на сломанной шкале ~0.41.
5. Не построил разрез TRAIN/CAL/EVAL для trainable веток.
6. Transitions для V2 скопировали B0-шум last-seed, хотя ranking V2 не менялся.

Правило для следующего агента: **не брать `analysis/selection.json` как победителя V4.**

---

## 17. Что из этого следует (и чего не утверждать)

Утверждать можно:

- B0-live остаётся ranking-reference V4; его не заменяли max8 и не переобучали.
- Внутри уже готового B0 top-5 есть неиспользованный QA-сигнал: **V2-A4** собирает лучший ответ, **V6** ещё и лучше выбирает чанк.
- KeyBERT-LOSCO, ownership-anchor и close-FP без anchor в этой реализации не сработали.
- Utility-positives на TRAIN модель выучивает; на EVAL того же split это не переносится.
- INTERNAL_TEST по-прежнему закрыт; ни одну систему нельзя «замораживать как финал».

Нельзя писать:

- «V3-A3 победил V4» / «utility labelling нашла target evidence»;
- «KeyBERT доказал document compatibility» или наоборот «KeyBERT бесполезен вообще»;
- «synthetic mismatch = настоящий target mismatch» (ветка даже не включилась);
- «softmax калибрует QA logits» (V5-A1 ничего не меняет, V5-A0 вредит);
- что V6 готов в production без повторного per-fold подбора весов и без проверки на остальных outer run.

Практический следующий шаг по протоколу, если не открывать INTERNAL_TEST: зафиксировать V6 и/или V2-A4 как заранее объявленные second-stage поверх **того же** B0-live, с честным 5× OOF и без обучения full-document ranker на одном run. D4 по-прежнему вторичен: windowing после правильного positive, а ranking gap всё ещё больше. Комбинация V6+V2 разрешена G6, но в этом прогоне не считалась.

---

## 18. Карта артефактов

| Что | Путь |
|---|---|
| Код | `experiments/qa_chunk_selection/v4/` |
| План | `hypothesis/v4_experiment_plan_7_strategies.md` |
| Конфиг / контракт | `artifacts/qa_chunk_selection_v4/config_resolved.json`, `contract.json` |
| B0-live | `reference/b0_live/{manifest,metrics,predictions_scored}.jsonl` |
| Метрики систем | `analysis/<ID>/metrics.json` + `predictions_scored.jsonl` |
| Compare / selection | `analysis/compare.json`, `compare_summary.json`, `selection.json` |
| V1 λ | `analysis/v1_lambdas.json` |
| V2 — в метриках систем, extra поля aggregation | `final_answer_*` в `V2-A*/metrics.json` |
| V3 utility | `analysis/v3_utility_candidates.jsonl`, `v3_utility_summary.json`, `models/v3_utility/seed42/` |
| V4 diagnostic | `analysis/v4_diagnostic.json`, `v4_pairs.jsonl` |
| V5 head | `models/v5_linear.json` |
| V6 веса | `analysis/v6_weights.json` |
| V7 | `models/v7/seed42/`, `training_report.json` (`use_anchor: false`) |
| Автодамп | `reports/results_v4.md` — **не** научный отчёт |
| Лог | `logs/qa_chunk_selection_v4.log` |
| Прогресс | `artifacts/qa_chunk_selection_v4/progress.jsonl` |
