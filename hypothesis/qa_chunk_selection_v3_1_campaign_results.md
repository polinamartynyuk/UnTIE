# Сводный отчёт: отбор чанков E0 → V2 → V3 → авто-silver V3.1 → D4/D5

Документ фиксирует **все фактически полученные результаты** линейки QA chunk selection и кампании 2026-09-06 (автоматический QA-silver и диагностические D4/D5). Это человеческий разбор для решения, **какие изменения протокола и кода делать дальше**. Автодампы JSON→MD (`artifacts/.../reports/results_*.md`) не заменяют этот файл.

Дата сборки: **2026-09-06**.  
INTERNAL_TEST **не открыт**. Gold **никогда** не использовался в ранжировании, фильтрации, shortlist и runtime-агрегации. `source_label` Phase 1 **не переписывался**. Деревья E0 / V2 / V3 / ручной V3.1-prep **не перезаписывались**.

Соседние отчёты (не дублировать слепо; при расхождении приоритет у исходных JSON и этого файла для кампании 6 сентября):

- V2: [`qa_chunk_selection_experiments_v2_results.md`](qa_chunk_selection_experiments_v2_results.md)
- V3: [`qa_chunk_selection_experiments_v3_results.md`](qa_chunk_selection_experiments_v3_results.md)
- Статус на 2026-09-04 **устарел** ([`qa_chunk_selection_current_status.md`](qa_chunk_selection_current_status.md): ещё пишет, что кода ranking V3.1 нет и D4/D5 не запускались).

---

## 0. Короткий итог для решения «что менять»

1. **Живой победитель по-прежнему V3-B0** (рецепт V2-B2a, MiniLM, grouped 5-fold OOF): Token F1 **27.36**, Hit@1 **36.7%**, HN@1 **4.9%**, Amb@1 **50.9%**, GSRN@1 **33.9%**. Ни одна новая семья V3 (B1–B3, structure, attention, span) его не обошла по протокольной селекции.
2. **Главный разрыв — ранжирование top-1**, не ридер. Оракул «лучший gold-positive чанк» даёт Token F1 **73.01**; у **76.4%** вопросов есть хотя бы один positive с F1 ≥ 50. Hit@5 у B0 уже **79.0%**, Hit@10 **92.2%**. Узкое место — отличить нужный evidence от ambiguous / GSRN внутри уже хорошего shortlist.
3. **H3.1 (target-mismatch) ни разу не проверялась.** Ручной overlay 0/400. Авто-silver v1 и v2 дали **0 high-conf TE и 0 TM**. Freeze не состоялся. `v3_1_training_allowed=false`. Код ranking V3.1 написан, **прогон не запускался** (и не должен был).
4. **Авто-silver провален не из-за бага τ=0, а из-за недостаточной разделимости frozen QA.** v2 (raw+margin CDF, честный τ) на всех 5 фолдах имеет пустое пересечение `(R ≥ 0.80) ∩ (easy pass ≤ 0.25)`. При R≈0.80 easy pass = **48–56%**. При easy ≤ 0.25 recall падает до **35–59%**. HN почти неотличим от ambiguous (P(pos>HN) ≈ 0.64, P(amb>HN) ≈ 0.52).
5. **Даже если бы Stage A прошёл, TE/TM всё равно не появились бы без Stage B.** Stage B не запускался (гейт). Ownership-пробы `qa_target_probe_score` / `qa_external_probe_score` = `None`. Mapping без ownership не может выдать `consensus_target` / `consensus_external` и сваливается в `partial_current` или `unresolved`. Все 8927 строк `abstained=true`.
6. **D4 (gold-окно) открывает гейт runtime-селектора, но это потолок, не система.** Minimal gold window: **70.04** vs full chunk **64.66** (**+5.38**). Gold sentence и sentence±1 **хуже** полного чанка. Выигрыш сидит в hard/failed; на reader_easy сжатие окна **вредит** (98.3 → 87.5). Method: minimal window хуже full (−3.1).
7. **D5: мульти-evidence реален, не дубликаты.** 307/453 вопросов имеют ≥2 positive-чанка; после overlap-dedup 0.6 остаётся **306**. Агрегация разрешена, но не заменяет top-1 ranking.
8. **Что из этого следует для следующих изменений.**
   - Не запускать V3.1 ranking на silver.
   - Не ослаблять Stage A post-hoc, чтобы «получить хоть какие-то TE/TM».
   - Не ждать, что runtime D4/D5 закроют разрыв 27→73.
   - Единственная честная проверка H3.1 — **ручная разметка 400-очереди** (уже лежит в `artifacts/qa_chunk_selection_v3_1_prep/`).
   - Если менять протокол авто-разметки — это **новый** протокол с новыми априорными порогами и, скорее всего, другим сигналом, не повтор v1/v2 с теми же `(0.80, 0.25)`.
   - Если делать код без разметки — только вторичные диагностические ветки в **новых** директориях: runtime D4 на TRAIN, D5-агрегатор, Stage B diagnostic на хвосте/400. Не как замена B0 и не как B1-Silver.

---

## 1. Постановка и что считается стратегией

UnTIE — extractive QA по SciREX. На канонический aspect-вопрос (task / method / material / metric) нужно выбрать evidence-чанк **текущей** статьи и вытащить span. Ридер во всей линейке — frozen Phase 1 RoBERTa (`scripts/models_processing/models/bert_eng_qa_baseroberta_model`). Кандидаты — все чанки текущей статьи (Mode II).

Слои меток **нельзя смешивать**:

| Слой | Кто ставит | Значения | Можно ли переписывать |
|---|---|---|---|
| Phase 1 `source_label` | исходная разметка | positive / hard_negative / easy_negative / ambiguous | **нет** |
| V3 `target_status` | человек (overlay) | target_evidence / target_mismatch / partial_support / unresolved | отдельный файл, не `source_label` |
| QA-silver | правила + frozen QA | `silver_*` тех же четырёх статусов | слабая разметка, не gold, не verified |

Ambiguous и GSRN **нельзя** автоматически считать TM. Пустой / unresolved / partial **нельзя** мапить в TM или «точно не TM».

Стратегии, которые реально прогонялись (или были формально запущены и остановлены гейтом):

| # | Стратегия | Тип | Артефакты | Статус |
|---|---|---|---|---|
| E0 | A/B/C/D1/D2 на official DEV | ranking | `artifacts/qa_chunk_selection/` | закрыт |
| V2 | B2a + fusion/C/D2 на CAL/DEV | ranking | `artifacts/qa_chunk_selection_v2/` | закрыт |
| V3-B0 | тот же B2a в grouped CV | ranking | `artifacts/qa_chunk_selection_v3/` | **победитель** |
| V3-B1/B2/B3 | TM / pairwise / listwise | ranking | там же | хуже B0; TM не было |
| V3-structure | секция + соседи в CE | ranking | там же | хуже, pairwise fail |
| V3-attention | AT0–AT4 | ranking | там же | нет сигнала; HN@1 19% |
| V3-span | rerank спанов top-3 B0 | QA поверх B0 | там же | F1 падает |
| V3.1 manual prep | очередь 400 | разметка | `artifacts/qa_chunk_selection_v3_1_prep/` | 0/400 |
| QA-silver v1 | Stage A raw CDF, fallback τ=0 | авто-метки | `artifacts/qa_chunk_selection_v3_1_auto/` | гейт fail, 0 TE/TM |
| QA-silver v2 | raw+margin CDF, честный τ | авто-метки | `artifacts/qa_chunk_selection_v3_1_auto_v2/` | гейт fail, 0 TE/TM |
| Stage B / B1-Silver | ownership → обучение | не бежало | — | заблокировано |
| V3.1 ranking | S0, B0-max8, B1-Silver, G1 | код есть, прогона нет | `experiments/qa_chunk_selection/v3_1/` | не запускать |
| D4 oracle | gold-окна на positives | диагностика | `artifacts/qa_chunk_selection_v3_d45/` | +5.38, гейт open |
| D5 audit | overlap-кластеры positives | диагностика | там же | 306/453 multi-cluster |

---

## 2. Вселенная данных и замки

Снимок V2, который ест V3 и все последующие пакеты: **955** instance-записей. `source_label_rewritten=false`.

| V2-сплит | записей | статей | answerable | роль дальше |
|---|---:|---:|---:|---|
| train | 477 | 180 | 303 | unlocked, в CV |
| calibration | 111 | 44 | 73 | unlocked, в CV |
| development | 126 | 44 | 77 | unlocked, в CV |
| internal_test | 78 | 31 | 54 | **LOCKED**, QA нет |
| seen_exploratory_dev | 163 | 65 | 105 | official SciREX DEV, не селекция |

Unlock + answerable = **453 вопроса / 240 документов**. В протоколе фигурируют 268 unlocked документов; **28** из них не имеют `robust_positive` и в CV не входят. Это те же 303+73+77 answerable, не другой корпус.

Frozen CV (`artifacts/qa_chunk_selection_v3/data/v3_cv_splits.json`):

- grouping: `document_id`, seed 42, 5 fold;
- `document_id_hash` = `21cf53200c0350ad4176b378b7c56c6eadd6f31cf479560ce1db87e12a05f90d`;
- `split_payload_hash` = `d3647846e7be929791b359a1cb93b6722dcd0df6b8567159c9fd6d50eba2fcb8`;
- роли цикла: 3 TRAIN / 1 CAL / 1 EVAL, как в отчёте V3 §3.3.

По fold (документы / вопросы / positives / ambiguous):

| fold | docs | q | pos | amb | task | material | method | metric |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 48 | 91 | 267 | 728 | 29 | 25 | 22 | 15 |
| 1 | 48 | 90 | 276 | 737 | 26 | 25 | 24 | 15 |
| 2 | 48 | 91 | 224 | 791 | 30 | 25 | 23 | 13 |
| 3 | 48 | 91 | 229 | 720 | 32 | 26 | 23 | 10 |
| 4 | 48 | 90 | 257 | 823 | 30 | 25 | 23 | 12 |
| **сумма** | **240** | **453** | **1253** | **3799** | **147** | **126** | **115** | **65** |

Новые outer folds **не создавать**. INTERNAL_TEST не открывать, пока не будет другой выбранной системы, чем уже известный B0.

---

## 3. Как читать метрики (кратко)

| метрика | смысл |
|---|---|
| Token F1 | 0..100, токенное пересечение спана с gold |
| Hit@K | в top-K есть original `robust_positive` |
| HN@K | доля слотов top-K, занятых hard_negative (`|top-K ∩ HN| / K`) |
| pairwise_clean | доля пар (pos, чистый HN) с score(pos) > score(HN) |
| Amb@1 / GSRN@1 | top-1 — ambiguous / `gold_supporting_robust_negative` |
| F1 \| pos | Token F1, если **выбранный чанк ранжирования** — original positive |
| TargetMismatch@1 | verified TM на top-1; без overlay = null |
| silver_* | слабая метка; high-conf TE/TM только если правила и гейт это разрешили |

OOF V3 — **model-selection evidence внутри unlocked**, не confirmatory INTERNAL_TEST.  
D4-числа — **chunk-level на gold-positive**, окно выбрано gold. Это не живой Token F1.  
Silver-числа — качество **правил отбора кандидатов**, не качество ранкера.

---

## 4. E0 и V2 — что уже закрыто и зачем это помнить

### 4.1. E0 (official DEV, 105 q / 57 papers)

Generic relevance путает positive с HN. Answerability (B) — почти единственный сигнал, который их разделяет. Structural mix D2 давал лучший DEV Token F1 в той схеме, но хуже держал HN. Official DEV потом стал `seen_exploratory_dev` и **не** участвует в селекции V2/V3.

### 4.2. V2 (2026-09-02, ≈41 мин, `cuda:0`)

Победитель: **B2-b2a seed 44, M=1**. DEVELOPMENT (77 q):

| | Hit@1 | HN@1 | pairwise | Token F1 | EM | F1\|pos | Amb@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B2-b2a | 37.7% | 3.9% | 74.6% | **25.71** | 19.5% | 59.89 | 54.5% |

Оракул лучшего positive: **71.86**; solvable F1≥50 = **76.6%**.

Что на DEVELOPMENT **не удержалось**:

- QA-confidence fusion (R3_2) выбран ключом селекции, по QA хуже B2;
- R3_3 лучше по QA, отвергнут pairwise;
- сырые multi-passage логиты C: CAL F1 36.12 → DEV **20.51**, F1|pos 59.89 → 30.50;
- adaptive M повторяет провал C;
- D2a = чистый B2; RRF поднимает HN@1 до 9.1%.

**Наблюдение, которое V3 подтвердил на большем пуле:** ридер умеет извлекать, если чанк правильный; живой провал — выбор чанка, особенно ambiguous/GSRN, не HN.

V2 **не перезапускать**. Рецепт b2a переехал в V3-B0 как переобучение на каждом fold, не как checkpoint seed 44.

---

## 5. V3 ranking — полные числа и выводы

Прогон: 2026-09-02 17:26:36 → 18:35:59, **4163 с ≈ 69 мин**, `cuda:0`, config hash `950c345e0660827a`.  
Стадии: prepare, split, overlay-sample, b0–b3, structure, positive-quality, at0–at4, span-ranker, oof-compare, report.  
Не default / не бежали: b4, at4-d, at5, d3, d4, d5, freeze-final, final-eval.

Первый старт упал (`_device` перезаписан `_chunk_signature`). Resume после фикса; CV уже были frozen.

### 5.1. Overlay: очередь есть, меток нет

Таксономия `v3-target-status-1`. Сэмплинг 200 hard + 200 representative = **400**. Подтипы hard: GSRN 192, FN-cand 105, question_sensitive 103. `annotation/target_status.jsonl` **пустой**.

Следствия, которые бьют по всем семьям:

- `P_v3_target == P_original`;
- Hit@K ≡ TargetEvidenceHit@K_v3;
- `pairwise_pv3_gt_tm` = NaN, TargetMismatch@1 = null;
- B1 не получил ни одного TM-negative;
- B2/B3 не получили TM-триплетов;
- TM@1 нельзя было сделать hard safeguard (coverage < 0.95).

**H3.1 не тестировалась.** «B1 хуже B0» ≠ «TM-supervision не работает».

### 5.2. Протокол селекции V3

Hard safeguards vs B0: HN@1 ≤ B0+0.02; pairwise_clean ≥ B0−0.03. Среди допустимых — document-aspect Token F1.

Пороги: pairwise ≥ 0.7214, HN@1 ≤ 0.0686.

| система | HN | pairwise | admissible | Token F1 |
|---|---|---|---|---:|
| **b0** | ok | ok | да | **27.36** |
| b1 | ok | ok | да | 26.83 |
| b2 | ok | ok (0.7278) | да | 25.86 |
| span_ranker | ok | ok | да | 24.40 |
| b3 | ok | fail 0.7184 | нет | 25.35 |
| structure | ok | fail 0.6866 | нет | 26.15 |
| text+attention | ok | fail 0.6879 | нет | 26.02 |
| attention_only | fail 0.1898 | fail 0.5151 | нет | 24.17 |

Выбрано: **b0**. Pareto (F1 ↑, HN@1 ↓): **b0, b1**.  
Честный кандидат на будущий INTERNAL_TEST: **V3-B0, b2a, M=1**. Не B1, не attention, не SpanRanker.

### 5.3. Полная OOF-таблица

B0–B3/structure: n=1359 = 453×3 seed. Attention/span: только seed 42, n=453. Сравнивать их с трёхсидовым B0 27.36 нельзя — честный якорь B0-s42 = **26.46**.

| система | n | Token F1 | EM | Hit@1 | Hit@5 | Hit@10 | HN@1 | pairwise | Amb@1 | GSRN@1 | F1\|pos | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 1359 | **27.36** | 20.75% | **36.72%** | 79.03% | 92.20% | 4.86% | **75.14%** | 50.85% | 33.92% | 61.70 | 0.542 |
| B1 | 1359 | 26.83 | 19.50% | 35.98% | 77.85% | 92.05% | 4.49% | 73.68% | 49.96% | 32.60% | 61.48 | 0.534 |
| B2 | 1359 | 25.86 | 20.38% | 34.00% | 76.67% | 91.02% | 5.22% | 72.78% | 52.76% | 33.26% | 61.29 | 0.518 |
| B3 | 1359 | 25.35 | 18.84% | 34.58% | 76.53% | 91.32% | 5.15% | 71.84% | 51.43% | 34.36% | 61.09 | 0.527 |
| structure | 1359 | 26.15 | 19.50% | 35.39% | 75.50% | 90.51% | 5.37% | 68.66% | 51.21% | 33.11% | 60.34 | 0.527 |
| att_only | 453 | 24.17 | 19.21% | 35.54% | 73.51% | 83.66% | 18.98% | 51.51% | 1.32% | 1.10% | 62.36 | 0.526 |
| text+att | 453 | 26.02 | 18.32% | 32.23% | 75.50% | 89.85% | 5.74% | 68.79% | 52.76% | 30.24% | 63.48 | 0.510 |
| span | 453 | 24.40 | 18.76% | 36.20% | 80.57% | 92.27% | 4.86% | 74.31% | 51.21% | 34.44% | 37.32† | 0.541 |

† F1|pos у span смотрит на B0-чанк, а ответ уже из другого чанка top-3. Реальный F1 при QA-чанке positive = 60.83. Поле 37.32 **не сравнивать** с 61.70.

Дополнительно у B0: attr F1 26.80, Char F1 0.521, ROUGE-L 0.339, pos>amb 64.51%, pos>GSRN 55.47%, gold-match с не-original-positive 4.64%.

### 5.4. V3-B0 подробно

Не checkpoint V2. В каждом fold заново: question-balanced weighted BCE, только `training_eligible_hn` + capped easy, early stop по CAL pairwise_clean, patience 2, max 5 эпох, pos:hn:easy = 1:3:1.

Обучение 15 чекпоинтов: mean best epoch 4.27, mean CAL pairwise **0.7649** (min 0.6988, max 0.8254) — лучший CAL среди текстовых семей.

По seed:

| seed | Token F1 | Hit@1 | HN@1 | Amb@1 | pairwise |
|---:|---:|---:|---:|---:|---:|
| 42 | 26.46 | 36.20% | 4.86% | 51.21% | 74.31% |
| 43 | 27.94 | 37.97% | 5.96% | 49.89% | 74.33% |
| 44 | 27.68 | 35.98% | **3.75%** | 51.43% | **76.78%** |
| pool | **27.36** | **36.72%** | **4.86%** | **50.85%** | **75.14%** |

По eval-fold (seed 42) разброс ≈ 10 F1: fold 0/4 лёгкие (30.67 / 31.21), fold 1/3 тяжёлые (22.59 / 20.80). Именно поэтому V3 ушёл с одного DEVELOPMENT на grouped CV.

Состав top-1 (pool 1359):

| source_label | n | доля |
|---|---:|---:|
| positive | 499 | 36.72% |
| ambiguous | 691 | 50.85% |
| easy_negative | 103 | 7.58% |
| hard_negative | 66 | 4.86% |

Primary class top-1: robust_positive 499, **GSRN 461 (33.92%)**, question_sensitive 136, easy 103, FN-cand 94, HN 66. GSRN почти равен числу positives.

По аспектам (pool 1359):

| аспект | n | Token F1 | EM | Hit@1 | HN@1 | Amb@1 | GSRN@1 | F1\|pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| task | 441 | 35.54 | 18.59% | 37.19% | 5.44% | 52.38% | 33.79% | 67.87 |
| material | 378 | 30.94 | 32.01% | **51.85%** | 2.38% | 39.95% | 28.57% | 54.82 |
| method | 345 | **18.66** | 16.81% | **23.77%** | 6.96% | 57.68% | **40.87%** | **73.25** |
| metric | 195 | **17.31** | 10.77% | 29.23% | 4.62% | 56.41% | 32.31% | 51.01 |

**Наблюдение.** Method: ридер сильный (F1|pos 73.3), ранкер слабый (Hit@1 23.8%, GSRN@1 40.9%). Material: ранкер сильный (Hit@1 51.9%), ридер слабее (F1|pos 54.8). Metric тяжёл и там, и там.

Бутстреп B0 Token F1, кластер по документу, 5000: mean 27.39, 95% ДИ **24.30–30.56**. Парных ДИ B0−B1 нет. Точечные −0.53 / −1.21 лежат внутри ширины ДИ B0 — это не формальный тест.

### 5.5. Остальные семьи V3 — что именно проверялось

**B1.** Тот же BCE, но positives += verified TE, negatives += verified TM. Overlay пуст → только другой сборщик пар. F1 26.83, HN@1 чуть лучше (4.49%). H3.1 не ставилась.

**B2 pairwise.** Должны быть триплеты pos-vs-TM с p=0.50. TM нет → pos vs HN/easy, близко к проигравшему V2-b2c. F1 25.86, Amb@1 вырос до 52.76%. CAL early stop раньше (mean epoch 3.4).

**B3 listwise.** Document-level, T=1.0. H3.2 (target vs mismatch внутри статьи) без mismatch-меток не проверялась. F1 25.35, pairwise 71.84% — safeguard fail. Fold 3 seed 42 провален (F1 14.05, Hit@1 22.0%).

**Structure.** Вход `[SECTION] [PREV] [CHUNK] [NEXT]`. F1 26.15, pairwise **68.66%** (обвал). Method ещё хуже B0 (F1 14.81 vs 18.66). Соседний контекст не отделяет GSRN.

**Attention AT0–AT4.**

- AT0: `output_attentions` не меняет логиты (start/end max abs = 0) — необходимое условие.
- AT1: AUROC mean(vector) G1 vs HN = **0.519**, порог 0.55, `has_signal=false`. G3 (verified TM) = 0. Код AT1/AT3 шёл по **всему unlocked**, не только TRAIN — отклонение от плана, AUROC оптимистично завышен и всё равно не проходит.
- AT3: лучшие головы layer1/head4 AUROC 0.552 и layer11/head8 0.548; `all_head_mean_auroc` 0.485. N=2 без CAL-сетки.
- attention_only: F1 24.17, HN@1 **18.98%**, pairwise 51.5%, Amb@1 1.32%. Почти не берёт ambiguous, сваливает HN. На method exploratory Hit@1 53.9% vs B0-s42 22.6% — **и** HN@1 22.6%. Нельзя брать как рецепт.
- text+attention (0.5 B0 + 0.5 prototype на B0 top-5): F1 26.02 vs B0-s42 26.46, Hit@1 падает до 32.2%, pairwise fail.

**SpanRanker K=3** поверх B0 seed 42. Ranking-id остаются B0 (поэтому Hit@K совпадают с B0-s42). Token F1 **24.40** vs 26.46. В 67.8% случаев уходит с B0 top-1. Oracle среди span-кандидатов 48.25 — даже идеальный выбор спана в top-3 не достаёт оракул любого gold-positive (73.01). H3.4 не подтверждена: узкое место не спан внутри shortlist, а top-1 чанка.

CAL pairwise порядок совпал с OOF: B0 0.765 > B1 0.750 > B2 0.747 > B3 0.730 > structure 0.719.

### 5.6. Positive-quality (1253 gold-positives, 453 вопросов)

| | |
|---|---|
| mean Token F1 на positives | 64.66 |
| EM на positives | 55.15% |
| positives с F1 ≥ 50 / ≥ 80 | 70.15% / 49.08% |
| **оракул best-positive Token F1** | **73.01** |
| вопросы с хотя бы одним pos F1 ≥ 50 | **76.38%** |

Полосы: easy 615, medium 264, hard 138, failed 236.  
Mean F1 на positives: task 70.10, method 67.88, metric 58.76, material 58.23.  
Material: 110/404 failed — согласуется с высоким Hit@1 и низким F1|pos.  
V2 DEV oracle 71.86 / 76.6% solvable — тот же порядок. Вывод «ридер не потолок» на большом пуле подтверждён.

**Ворота V3 (ещё до отдельного пакета D4/D5):** D4 gate passed (374 hard/failed с gold и длинным контекстом); D5 gate passed (307/453 с ≥2 positives). В самом V3 стадии D4/D5 были заглушки и **не бежали**.

### 5.7. Что V3 разрешает утверждать и что нет

Поддержано на OOF unlocked:

- разрыв 27 vs 73 — ranking, не reader;
- B0 допустимый победитель; новые losses/structure/attention/span его не обошли;
- главный distractor — ambiguous/GSRN;
- запас в переранжировании top-5, не в «найти хоть какой-то positive»;
- method провален ранжированием; metric тяжёл для обоих;
- attention-сигнатуры RoBERTa здесь не переносятся;
- span K=3 не поднимает F1;
- D4/D5 гейты открыты как разрешение на *следующий* эксперимент.

Нельзя утверждать:

- B0 победил на INTERNAL_TEST;
- H3.1 опровергнута или подтверждена;
- pairwise/listwise «не работают» в принципе (не было класса negatives, ради которого их заводили);
- attention бесполезен для method (exploratory сигнал есть, safeguard нет);
- V3 улучшил V2 (другие сплиты);
- можно менять production ranking.

---

## 6. Ручная подготовка V3.1 (400-очередь)

Код: `experiments/qa_chunk_selection/v3_1_prep/`.  
Артефакты: `artifacts/qa_chunk_selection_v3_1_prep/`.  
Протокол: [`v3_1_data_preparation_protocol.md`](v3_1_data_preparation_protocol.md).

Annotator pack экспортирован, n=400, скоры скрыты. Писать в `annotator/to_fill.jsonl` или UI `annotator/index.html`. Поля: `v3_target_status`, `v3_annotation_confidence`, `v3_annotation_reason`; при mismatch — `v3_mismatch_type`; при evidence — `v3_gold_relation`.

Ingest пустой разметки 2026-09-04 ≈ 16:27:

- `n_filled=0`;
- все TM-счётчики 0 во всех 5 outer runs;
- `v3_1_training_allowed=false`;
- freeze отказался (правильно), код выхода 2, `analysis/freeze_skipped.json`.

`--allow-partial-freeze` не использовать.  
Пороги готовности: `min_train_questions_with_TM=10`, `min_eval_questions_with_target_and_TM=10`.

**Это по-прежнему единственный честный путь проверить H3.1.** Авто-silver его не заменил.

---

## 7. Автоматический QA-silver: замысел и реализация

Протоколы:

- [`v3_1_automated_data_preparation_protocol.md`](v3_1_automated_data_preparation_protocol.md)
- [`v3_1_experiment_continuation_automated_overlay.md`](v3_1_experiment_continuation_automated_overlay.md)

Критическое решение относительно более раннего черновика: **нет generative teacher / NLI / semantic encoder**. Silver должен получаться из того же frozen extractive QA плюс детерминированные discourse / gold-relation / consensus правила.

Код: `experiments/qa_chunk_selection/v3_1_auto_prep/`.  
Априорные пороги (зафиксированы до прогона, не подбирать по EVAL):

- `R_min = 0.80` — recall gold-positives на CAL, прошедших τ;
- `max_easy_pass = 0.25` — доля easy negatives, прошедших τ;
- `gold_f1_threshold = 0.50`;
- `min_train_TM = 10`;
- гейт Stage A обязан пройти **до** Stage B;
- freeze не перезаписывает готовый каталог.

Пайплайн: init → freeze-qa → build-universe → stage-a → calibrate → stage-a-gate → stage-b → discourse → map-silver → cluster → qc → stability → readiness → freeze.

Universe (оба прогона, 8927 строк):

| роль | n | source_label |
|---|---:|---|
| primary_ambiguous | 3799 | ambiguous |
| control_positive | 1253 | positive |
| control_hn | 694 | hard_negative |
| control_easy | 3181 | easy_negative |

Историческая 400-очередь помечена `in_historical_queue=true` (**400**), но **не** является целевым пулом Stage A: протокол требует scoring **всех** unlocked ambiguous. Все `source_label=ambiguous` остаются ambiguous; 400 — только диагностика.

Модель QA: local RoBERTa, `eval_mode=true`, `weights_unchanged=true`, CE и sentence encoder **не** грузились.  
`question_set_version=v3.1-qa-questions-1`, hash `bcdb39832feba1824858c3d8a88d0b72671316c527db4eb130a6f07f6a8176f6`.

Первый старт ≈18:35 упал: `load_models(..., cross_encoder_model="unused")` пытался скачать HuggingFace `cross-encoder/unused` (401). Исправление: `load_frozen_qa()` в `experiments/qa_chunk_selection/models.py` — только локальный QA.

---

## 8. QA-silver v1 (`v3.1-qa-silver-1`)

Каталог: `artifacts/qa_chunk_selection_v3_1_auto/`.  
Лог: `logs/qa_chunk_selection_v3_1_auto.log`.  
Старт 2026-09-06 18:52:34, конец 19:27:53, wall **≈35.3 мин** (2119 с). Почти всё время — Stage A QA на 8927 строк (`n_uncached_calls=8927`).

Mining score v1: CDF **сырого span score** относительно TRAIN-positives. Если совместный гейт пуст — fallback **τ=0**.

### 8.1. Сырые скоры отделяют классы слабо

`qa_raw_span_score` / `qa_margin` на Stage A:

| класс | n | raw p25 / p50 / p75 | margin p50 | gold Token F1 p50 |
|---|---:|---|---:|---:|
| positive | 1253 | 0.74 / **3.12** / 5.87 | 0.70 | **1.00** |
| ambiguous | 3799 | −1.04 / **1.39** / 4.02 | 0.51 | 0.00 |
| hard_negative | 694 | −1.16 / **1.12** / 3.74 | 0.53 | 0.00 |
| easy_negative | 3181 | −2.60 / **−0.51** / 1.94 | 0.50 | 0.00 |

Mean raw positive = 3.27. Valid span rate = 1.0 (invalid_span_rate в гейте 0.0).

Попарные вероятности на raw (bootstrap-сэмпл пар):

| сравнение | P(>) |
|---|---:|
| pos > easy | **0.762** |
| pos > HN | 0.648 |
| pos > amb | 0.630 |
| amb > easy | 0.645 |
| HN > easy | 0.621 |
| amb > HN | **0.522** |

**Наблюдение.** Pos vs easy — умеренное разделение. Pos vs HN/amb — слабое. Amb и HN почти неразличимы. Frozen QA отвечает «здесь вообще есть какой-то извлекаемый спан», а не «это evidence *этого* объекта, а не соседнего». Это ровно та оппозиция, которую H3.1 хотела закрыть разметкой TM, и ровно тот сигнал, которого у ридера нет.

Gold F1 на controls: у positive медиана 1.0 (ридер на своём чанке умеет), у остальных медиана 0. Отдельный gold-F1 не спасает калибровку τ: гейт смотрит на mining score, не на gold F1 (gold в калибровке порога не должен решать «какой чанк брать в проде»).

### 8.2. Калибровка и гейт

На всех 5 фолдах: `τ=0`, recall=1.0, easy_pass=1.0, `ok=false`. Совместной точки нет, сработал fallback.

`stage_a_gate.json`:

- `min_positive_recall`: true (тривиально при τ=0);
- `max_easy_pass`: **false**;
- `max_invalid_span`: true;
- `passed`: **false**.

Stage B **не запускался**. `stage_b_ownership.jsonl` и `stage_b_paraphrases.jsonl` — пустые файлы.

### 8.3. Mapping: почему 93.6% стали partial

Правило `map_silver` требует для TE: plausible + consistency ≥ 0.5 + target_score > external_score + δ + gold.allowed_for_primary_te + нет paraphrase_conflict.  
Для TM: plausible + consistency ≥ 0.5 + external лучше + independent external context.

После пропуска Stage B:

- `qa_target_probe_score` = `None` на всех 8927;
- `qa_external_probe_score` = `None`;
- `qa_paraphrase_agreement` = 0.0;
- `target_better` / `external_better` = false.

Тогда срабатывает ветка `partial_current`: plausible ∧ current_ctx ∧ ¬external_better ∧ consistency < τ_consistency.  
`current_paper_marker_score > 0` у **8354 / 8927 (93.6%)** строк. При τ=0 plausible=true у всех.

Итог v1:

| статус | n | доля |
|---|---:|---:|
| silver_partial_support | 8354 | 93.58% |
| silver_unresolved | 573 | 6.42% |
| silver_target_evidence | **0** | 0 |
| silver_target_mismatch | **0** | 0 |

Причины mapping: `partial_current` 8354, `insufficient_consensus` 422, `conflict_or_near_threshold` 151.  
Confidence: medium 8354, low 573. **Все 8927 abstained=true.** High-файл пуст (0 байт). Medium-файл = 8354 строки.

Разрез статус × исходный класс:

| source_label | partial | unresolved |
|---|---:|---:|
| ambiguous | 3610 | 189 |
| easy_negative | 2879 | 302 |
| positive | 1213 | 40 |
| hard_negative | 652 | 42 |

**Наблюдение.** Даже gold-positives почти все ушли в partial, не в TE: без Stage B ownership «target лучше external» доказать нельзя. Контрольные easy тоже стали partial в 2879 случаях — это не «модель нашла evidence», это сработало правило «есть маркер текущей статьи и QA не отказала».

По аспектам (из `silver_summary.json`): partial task 2731 / material 2270 / method 2164 / metric 1189; unresolved 165 / 151 / 169 / 88.

### 8.4. QC, inventory, freeze

`qc_controls.json` (осторожно: StageA_* rates записаны как 0.0 — это артефакт QC при проваленном гейте, не «ридер не нашёл ни одного positive»; реальный recall на CAL при τ=0 равен 1.0):

- SilverTargetEvidenceRate 0, SilverTargetMismatchRate 0, HighConfidenceRate 0;
- SilverPartialRate 0.936, SilverUnresolvedRate 0.064;
- controls: 1253 pos / 694 HN / 3181 easy.

`cv_tm_inventory.json`: во всех run `n_train_TM_chunks=0`, `n_train_questions_with_TM_silver=0`. Поле `n_train_silver_target_chunks` (767/729/710/753/800) — это **число TRAIN gold-positives**, не silver TE. Не путать.

`stability.json`: `high_confidence_flip_proxy=null`, `n_runs=0` — flip-тест не на чем считать.  
`validation.json`: 44635 строк, 0 ошибок схемы (universe×fold overlays).  
`readiness.json`: `v3_1_training_allowed=false`. Блокеры: `stage_a_gate`, `TM_silver_gt_0`, `each_outer_train_meets_min_TM_silver`.  
`freeze_skipped.json`: `frozen=false`, те же блокеры.

---

## 9. QA-silver v2 (`v3.1-qa-silver-2`)

Каталог: `artifacts/qa_chunk_selection_v3_1_auto_v2/`.  
Лог: `logs/qa_chunk_selection_v3_1_auto_v2.log`.  
Старт 19:54:47, конец 19:54:56, **≈9.5 с**. Stage A не пересчитывался.

`analysis/reused_stage_a.json`:

- источник: `.../v3_1_auto/annotation_qa_silver/stage_a_all_ambiguous.jsonl`;
- sha256 `4a2be277a5c4e30425ddd34f4ba5ecdf140503fd02c98896fd14d936db5f33b1`;
- `n_imported=8927`;
- `n_uncached_calls=0`.

Изменения калибровки (`SILVER_RULE_VERSION=v3.1-qa-silver-2`, `mining_score_version=raw_margin_mean_cdf_1`):

1. Mining score = среднее CDF(raw) и CDF(top1−top2 margin) по TRAIN-positives.
2. Если joint gate пуст — **самый селективный τ с R ≥ R_min**, никогда fallback τ=0.
3. Запись в уже готовый prep-каталог запрещена (новый `OUT_DIR`).

Смысл v2 был диагностический: отличить «сломанный fallback» от «скор принципиально не тянет гейт».

### 9.1. Mining score по классам

`qa_plausibility_score` (fold-зависимый CDF; сводка по всем mapped rows):

| класс | n | p25 / p50 / p75 | mean |
|---|---:|---|---:|
| positive | 1253 | 0.330 / **0.496** / 0.658 | 0.498 |
| ambiguous | 3799 | 0.216 / **0.377** / 0.544 | 0.390 |
| hard_negative | 694 | 0.203 / **0.377** / 0.542 | 0.388 |
| easy_negative | 3181 | 0.151 / **0.294** / 0.462 | 0.322 |

P(pos>easy)=0.718, P(pos>HN)=0.635, P(pos>amb)=0.635.  
Margin добавил честности порогу, **не** сделал HN отличимым от ambiguous (медианы 0.377 и 0.377).

### 9.2. Пороги по фолдам (точка R ≥ 0.80)

| fold | τ | CAL n_pos / n_easy | recall | easy pass | n_feasible | TRAIN n_pos |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 0.2966 | 229 / 546 | 80.35% | **49.08%** | 0 | 767 |
| 1 | 0.2949 | 257 / 732 | 80.16% | **54.23%** | 0 | 729 |
| 2 | 0.3282 | 267 / 566 | 80.15% | **48.06%** | 0 | 710 |
| 3 | 0.2895 | 276 / 669 | 80.07% | **49.48%** | 0 | 753 |
| 4 | 0.2406 | 224 / 668 | 80.80% | **56.29%** | 0 | 800 |

`reason=no_feasible_tau`, `selection_rule=most_selective_meeting_r_min`.  
Easy pass в **1.92–2.25 раза** выше бюджета 25%. Разброс по фолдам есть (run2 чуть чище, run4 грязнее), но **ни один** fold не проходит.

Гейт: `min_positive_recall=true`, `max_easy_pass=false`, `passed=false`. Stage B снова skipped.

Число Stage A candidates: v1 **3799** (все ambiguous, τ=0), v2 **2697** (часть ambiguous отсеялась). Отсев есть, чистоты нет.

### 9.3. Pareto CAL: вся кривая по фолдам

Доля CAL positives / easy с mining score ≥ τ. Формат `recall% / easy_pass%`.

| τ | run0 | run1 | run2 | run3 | run4 |
|---:|---|---|---|---|---|
| 0.00 | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 |
| 0.05 | 98.3/93.2 | 98.8/95.6 | 98.9/94.3 | 99.3/95.1 | 98.2/93.4 |
| 0.10 | 96.1/82.2 | 97.7/88.3 | 97.0/86.2 | 98.9/85.9 | 96.4/81.0 |
| 0.15 | 91.3/74.0 | 96.1/80.1 | 94.4/78.6 | 95.3/76.5 | 94.2/72.3 |
| 0.20 | 89.1/65.6 | 92.2/70.8 | 90.6/69.8 | 89.5/66.1 | 87.5/63.2 |
| 0.25 | 84.3/56.6 | 86.4/62.2 | 86.9/60.8 | 84.1/55.5 | 79.5/54.9 |
| 0.30 | 78.6/48.2 | 78.6/53.6 | 82.0/53.2 | 78.6/47.4 | 73.2/47.5 |
| 0.35 | 71.6/42.3 | 74.3/45.6 | 78.3/44.2 | 72.1/38.7 | 66.5/38.6 |
| 0.40 | 62.9/35.2 | 66.1/38.3 | 71.9/36.4 | 67.0/30.3 | 62.1/31.9 |
| 0.45 | 55.5/27.7 | 53.7/30.6 | 62.9/30.2 | 59.4/22.9 | 51.3/23.5 |
| 0.50 | 49.3/20.3 | 45.5/25.3 | 54.3/23.0 | 53.6/17.5 | 45.5/17.1 |
| 0.55 | 41.5/15.8 | 35.4/19.3 | 46.4/15.2 | 44.2/12.9 | 38.4/13.0 |
| 0.60 | 33.6/11.4 | 30.0/13.7 | 39.0/11.3 | 35.1/9.0 | 29.0/9.9 |
| 0.70 | 22.7/5.7 | 22.2/6.8 | 23.2/5.3 | 21.7/4.8 | 17.0/4.0 |
| 0.80 | 13.1/2.0 | 9.7/2.6 | 10.9/1.9 | 11.6/1.6 | 12.1/1.2 |
| 1.00 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |

Источник: `analysis/stage_a_pareto.json`.

Если **после** кривой сказать «давайте R_min=0.50 и easy=0.25» — это подбор порога по тем же CAL-данным, на которых гейт ставился. Протокол это запрещает. Даже описательно: при easy≤0.25 лучший recall по фолдам **49.3 / 35.4 / 54.3 / 59.4 / 51.3%**. Половина gold-positives потеряется ещё до Stage B. Такой кандидатный пул не годится как «почти полный» silver TE.

### 9.4. Mapping v2

| статус | n | доля |
|---|---:|---:|
| silver_partial_support | 5051 | 56.58% |
| silver_unresolved | 3876 | 43.42% |
| TE / TM | **0 / 0** | 0 |

Причины: `partial_current` 5051, `stage_a_noncandidate` 3560, `insufficient_consensus` 266, `conflict_or_near_threshold` 50.  
`qa_plausible`: true 5367 / false 3560.  
Снова **все abstained=true**, high-файл пуст, medium=5051.

Plausible × класс:

| | plausible | not |
|---|---:|---:|
| ambiguous | 2376 | 1423 |
| easy_negative | 1579 | 1602 |
| positive | 986 | 267 |
| hard_negative | 426 | 268 |

Recall positives overall 986/1253 = **78.7%** (близко к fold CAL ≈80%). Easy pass overall 1579/3181 = **49.6%**. HN pass 426/694 = **61.4%** — hard negatives проходят чаще easy. Это ещё один аргумент, что скор меряет «извлекаемость», не «правильный объект».

Историческая 400-очередь (все ambiguous): 254 partial / 146 unresolved. Авто-правила не дали по ней ни одного TE/TM.

Discourse на mapped rows (детерминированные признаки, Stage B не нужен): current_marker 93.6%, independent_external_context 43.4%, related_work 3.3%, citation 3.1%. External context есть часто, но без ownership-пробы правило TM не срабатывает.

QC v2: PartialRate 0.566, UnresolvedRate 0.434, TE/TM/high 0. Inventory TM снова нули. Validation 44635/0 errors. Readiness и freeze — те же блокеры, что v1.

### 9.5. Сравнение v1 и v2

| | v1 | v2 |
|---|---|---|
| mining | raw CDF | raw+margin mean CDF |
| пустой joint | τ=0 | самый жёсткий τ с R≥0.80 |
| τ на fold 0 | 0.00 | 0.297 |
| CAL recall | 1.00 | ≈0.80 |
| CAL easy pass | 1.00 | 0.48–0.56 |
| candidates | 3799 | 2697 |
| partial / unresolved | 8354 / 573 | 5051 / 3876 |
| high TE / TM | 0 / 0 | 0 / 0 |
| Stage B | skip | skip |
| freeze | skip | skip |
| вывод | сломанный fallback маскировал провал | провал **подтверждён** на честном τ |

v2 лучше v1 только как диагноз. Как источник training overlay оба равны нулю.

**Что осталось не проверенным даже после v2.** Если бы кто-то прогнал Stage B на хвосте высоких скоров, ownership-пробы *могли бы* разделить часть ambiguous на TE vs TM. Этого эксперимента не было. Протокол его запрещает после failed gate. Делать его можно только как **отдельный diagnostic** в новом каталоге, с явным статусом «не training overlay».

---

## 10. Ranking V3.1 — код есть, прогона нет

Пакет: `experiments/qa_chunk_selection/v3_1/`.  
Целевой каталог артефактов `artifacts/qa_chunk_selection_v3_1/` **не создавался**.

Задумано: preconditions → prepare → S0 audit → B0-max8 reference → B1-Silver → oof-compare → G1 → report. R1 / attention / D4 / D5 / INTERNAL_TEST — gated. Primary metric — original-gold Token F1. Overlay fold-specific: `annotation_qa_silver/oof/run{r}/silver_target_status.jsonl` (нельзя мешать CAL другого run).

Preconditions требуют freeze (`silver_manifest.json`, fold overlays, `v3_1_training_allowed=true`). Freeze никогда не состоялся. Launcher `v3_1/launch_background.sh` не запускался.

**Не запускать.** Нет ни silver TE/TM, ни human overlay.

---

## 11. D4 — gold-window oracle

Пакет: `experiments/qa_chunk_selection/v3_d45/`.  
Артефакты: `artifacts/qa_chunk_selection_v3_d45/`.  
Лог: `logs/qa_chunk_selection_v3_d45.log`.  
Старт 20:02:38, конец 20:09:35, **417 с ≈ 7 мин**. Стадии: init, check-gates, d4-oracle, d5-audit, report.  
`delta_f1_min=1.0`, `min_multi_cluster_rate=0.2`. Smoke=false, skip_qa_models=false.

Предусловия из V3 positive-quality подтверждены локально (`analysis/preconditions.json`): D4 374 hard/failed, D5 307/453 с ≥2 positives.

### 11.1. Что измерялось

Frozen QA на **1253 TRAIN/CAL gold-positive чанках**. Все 1253 локализованы (`n_localized_gold_sentence=1253`). Четыре окна, выбор окна **по gold**:

- `full_chunk` — как сейчас;
- `gold_sentence` — предложение, содержащее gold span;
- `minimal_gold_window` — минимальное окно вокруг gold span;
- `sentence_pm1` — gold sentence ±1.

Это **потолок** sentence/window selector при уже известном gold-positive. Не runtime, не ranking.

### 11.2. Средние Token F1

| окно | mean Token F1 | Δ vs full |
|---|---:|---:|
| full_chunk | 64.66 | 0 |
| gold_sentence | 57.32 | **−7.34** |
| **minimal_gold_window** | **70.04** | **+5.38** |
| sentence_pm1 | 56.20 | **−8.46** |

`runtime_d4_allowed=true` (порог Δ≥1.0). Селектор **не обучался**.

По аспектам:

| аспект | full | gold sent. | minimal | ±1 | Δ minimal |
|---|---:|---:|---:|---:|---:|
| task | 70.10 | 72.80 | **85.88** | 66.65 | **+15.78** |
| metric | 58.76 | 53.72 | **69.85** | 51.25 | **+11.10** |
| material | 58.23 | 50.41 | 59.31 | 52.79 | +1.08 |
| method | **67.88** | 49.16 | 64.81 | 50.05 | **−3.07** |

**Наблюдение.** «Взять gold sentence» — плохая идея в среднем. Нужно узкое окно вокруг спана. Method от сжатия **страдает**: методу, видимо, нужен более широкий контекст чанка. Любой runtime D4 без учёта aspect может ударить по уже слабому method-ранжированию (V3 method Hit@1 23.8%).

Hard/failed (n=374): full **12.03** → best window **42.79**.

### 11.3. Разрез по reader-band полного чанка

Это ключевое наблюдение, которого нет в автодампе.

| band полного чанка | n | full | gold sent. | minimal | ±1 |
|---|---:|---:|---:|---:|---:|
| reader_easy (F1≥80) | 615 | **98.34** | 77.12 | 87.51 | 78.86 |
| reader_medium | 264 | 60.76 | 54.61 | **72.24** | 52.11 |
| reader_hard | 138 | 32.59 | 39.35 | **62.37** | 35.15 |
| reader_failed | 236 | **0.00** | 19.25 | **26.51** | 14.06 |

На already-easy сжатие окна **ломает** почти идеальный ответ (98 → 87). Средний +5.38 — смесь вреда на easy и пользы на medium/hard/failed. Runtime-селектор, который всегда сужает окно, рискует отнять F1 там, где ридер уже справляется.

Построчно (n=1253), с учётом ничьих:

- minimal > full: **334** (26.7%);
- minimal = full: **710** (56.7%);
- minimal < full: **209** (16.7%);
- gold sentence > full: только 143; < full: 274.

«Лучшее» окно при разрыве ничьих в пользу первого max: full 904, minimal 231, gold sentence 114, ±1 4. То есть в большинстве случаев полный чанк уже не хуже. Выигрыш концентрированный, не универсальный.

По аспектам число уникальных побед minimal: task 92/390, material 58/404, method 51/349, metric 30/110 (знаменатели — число positives аспекта из D4 rows).

### 11.4. Что D4 разрешает и что нет

Разрешает: ставить **обучаемый** window selector как вторичный эксперимент (гейт открыт), только TRAIN, в новом каталоге, с проверкой method и reader_easy отдельно.

Не разрешает: объявлять +5.4 живым приростом; заменять B0; ждать закрытия разрыва 46 F1; учить селектор на EVAL или на gold-выбранном окне без честного runtime-признака.

---

## 12. D5 — audit независимых кластеров

Overlap-dedup gold-positive char-интервалов, порог 0.6, код кластеров из `v3_1_auto_prep.cluster.assign_clusters`. Семантики ответа нет, агрегатора нет.

| | |
|---|---|
| вопросы | 453 |
| ≥2 positive-чанка | 307 (67.8%) |
| ≥2 независимых кластера после dedup | **306 (67.5%)** |
| почти нет коллапса | 307 → 306 |
| `aggregation_allowed` | true (порог 0.20) |

Среднее число кластеров по аспектам: material **3.21**, method **3.03**, task **2.65**, metric **1.69**.

Гистограмма `n_independent_clusters` (из `audit.json`, ключи — размер): 1:147, 2:118, 3:64, 4:51, 5:32, 6:16, 7:7, 8:7, 9:3, 10:5, 11:1, 12:2.

**Наблюдение.** У двух третей вопросов несколько genuine evidence-регионов, не дубликаты одного чанка. Агрегация имеет смысл как покрытие нескольких валидных значений / нескольких мест. Она **не** исправляет top-1, если ранкер отдал GSRN. Coverage@K у B0 и Hit@5=79% — более прямой рычаг, чем склейка спанов с неправильного чанка.

V3 D5-гейт смотрел на «≥2 positives» (307). Реальный D5-аудит подтвердил, что это почти то же самое, что «≥2 независимых кластера» (306). Можно проектировать агрегатор, не боясь, что positives — копипаста.

---

## 13. Разложение разрыва Token F1

Пулы родственные, но не тождественные. Не складывать их как одну абляцию.

| точка | пул | Token F1 | что уже «дано» |
|---|---|---:|---|
| живой B0 | OOF вопросы, выбранный чанк | **27.36** | ничего |
| B0 F1 \| selected positive | те OOF, где top-1 = pos | **61.70** | ranking угадал чанк (не обязательно лучший) |
| D4 full chunk | 1253 gold-positive чанка | **64.66** | чанк = какой-то gold-positive |
| D4 minimal gold window | те же чанки, окно по gold | **70.04** | чанк + идеальное окно |
| V3 best-positive oracle | OOF, лучший positive вопроса | **73.01** | идеальный выбор среди positives |

Интерпретация для следующих изменений:

1. **≈46 F1** (73−27) — не тот чанк на top-1. Это главный рычаг. Содержимое: Amb/GSRN, method Hit@1, запас внутри top-5.
2. **≈9 F1** (73−62) — даже при попадании в *какой-то* positive это не всегда лучший positive, плюс ошибка спана. D5 (несколько независимых регионов) и выбор среди top-5 здесь уместнее, чем новый full-document ranker.
3. **≈5 F1** (70−65) — потолок windowing **после** уже известного gold-positive, с вредом на easy и method. Вторично.
4. Silver не сдвинул ни один из этих уровней: training overlay не появился.

---

## 14. Что каждая стратегия исключает и что оставляет открытым

| Стратегия | Исключает | Оставляет открытым |
|---|---|---|
| E0 D2 / V2 C / V2 D2 | «склеить пассажи / добавить RRF / generic CE и получить устойчивый DEV» | — |
| V2 / V3 B0 | «нужен принципиально другой bi-encoder recipe как baseline» | B0 как якорь; max8 как заранее зафиксированный reference |
| V3-B1/B2/B3 без overlay | «эти лоссы уже проверили H3.1» | те же лоссы **после** verified TM |
| V3-structure | «просто добавить секцию и соседей в CE» | другие structural признаки, не этот mix |
| V3-attention AT1–AT4 | «средний вектор / 2 головы frozen RoBERTa как ranker» | узкий method-аудит; другие слои/ридеры — только после overlay |
| V3-span K=3 | «переранжировать спаны top-3 B0» | span на top-5 после overlay не доказан, но приоритет низкий |
| Silver v1 | «τ=0 и массовый partial можно учить как TE» | — |
| Silver v2 | «достаточно починить калибровку / добавить margin» | другой сигнал (не этот QA-score) в **новом** протоколе |
| Stage B (не бежал) | ничего | diagnostic на хвосте/400, не training |
| D4 oracle | «gold sentence / ±1 как окно»; «окно закроет разрыв 46 F1»; «всегда сужать» | aspect-aware / band-aware selector; только TRAIN |
| D5 audit | «positives — дубликаты одного региона» | exact-normalized агрегатор как вторичный |
| Пустой human overlay | «H3.1 закрыта» | разметка 400 |

---

## 15. Рекомендации по следующим изменениям

Ниже — не новый протокол, а следствия из чисел. Любое ослабление гейта или новый сигнал требуют **отдельного** письменного протокола и нового каталога артефактов.

### 15.1. Делать

1. **Разметить 400** в `artifacts/qa_chunk_selection_v3_1_prep/annotator/to_fill.jsonl`. После ingest→validate→audit→coverage→cv-inventory→readiness→freeze, и только если `v3_1_training_allowed=true`, запускать continuation: сначала audit overlay, затем **только B1 vs B0** на тех же frozen folds. Не крутить сразу listwise/attention/span.
2. Смотреть запас **внутри top-5 B0** (Hit@5 79%), не новый full-document ranker вместо B0.
3. Method разбирать отдельно: ranking fail при сильном ридере; attention-only там поднимает Hit@1 ценой HN; D4-окно method вредит.

### 15.2. Не делать

1. Не запускать `experiments/qa_chunk_selection/v3_1/` на v1/v2 silver.
2. Не ставить `R_min=0.50` / `max_easy_pass=0.50` post-hoc на тех же CAL, чтобы получить TE.
3. Не учить модель на `silver_partial_support` как на target_evidence.
4. Не мапить ambiguous/GSRN → TM автоматически.
5. Не открывать INTERNAL_TEST на текущем B0.
6. Не перезаписывать E0/V2/V3/v3_1_prep/v3_1_auto/v3_1_auto_v2/v3_d45.
7. Не менять production (`untie/`, `datasets/`, Phase 1).
8. Не выбирать max5 vs max8 после просмотра OOF F1.

### 15.3. Опциональные вторичные эксперименты (только если нужна работа без аннотатора)

Каждый — **новый** `artifacts/...` каталог, read-only на предыдущих.

| эксперимент | зачем | ограничения |
|---|---|---|
| Runtime D4 selector, TRAIN-only | гейт открыт; hard/failed дают большой локальный прирост | не сужать always-on; проверить method и reader_easy; не ждать +46 F1 |
| D5 exact-normalized aggregation | 67.5% multi-cluster | только после shortlist; не лечит GSRN@1 |
| Stage B diagnostic на 400 или на хвосте score | узнать, разделяет ли ownership-проба TE/TM хотя бы там | **не** training overlay; не обходить Stage A gate молча |
| Остановка на этом | честный STOP аналог G1 на prep | предпочтительно, пока нет человеческих меток |

Новый авто-протокол (другой скор, другой teacher, другие априорные пороги) имеет смысл только если явно признать: **текущий frozen extractive QA как единственный teacher не проходит свой собственный Stage A**. Это отрицательный результат, не недоделанный прогон.

---

## 16. Реализационные факты, чтобы не наступить снова

- `qa_plausibility_score` нельзя считать сырыми логитами как вероятностями; v2 использует эмпирический CDF TRAIN-positives.
- Fold-specific silver: B1 run `r` не должен видеть CAL пороги другого run.
- `load_frozen_qa()` — только локальный QA. Не передавать `cross_encoder_model="unused"`.
- Launch-скрипты должны быть Unix LF (`bash\r` уже ломал V3).
- Launcher auto-prep отказывается писать в каталог, где уже есть `analysis/readiness.json`, если нет `ALLOW_EXISTING_OUT=1`.
- V3 AT1/AT3 считались на полном unlocked — не повторять как «train-only сигнал».
- Span F1|pos в V3 считается от `selected_chunk_label`, не от QA-чанка.
- Поле `n_train_silver_target_chunks` в inventory — счётчик gold-positives TRAIN, не silver TE.
- QC `StageA_PositiveRecall=0.0` при failed gate в v1/v2 не означает нулевой raw recall.
- `hypothesis/qa_chunk_selection_current_status.md` от 2026-09-04 не описывает авто-silver и D4/D5.

---

## 17. Указатель артефактов

| путь | что |
|---|---|
| `artifacts/qa_chunk_selection/` | E0, не трогать |
| `artifacts/qa_chunk_selection_v2/` | V2, не трогать |
| `artifacts/qa_chunk_selection_v3/` | V3 ranking + пустой overlay + positive-quality |
| `artifacts/qa_chunk_selection_v3/analysis/oof_compare.json` | сводная OOF-таблица |
| `artifacts/qa_chunk_selection_v3/analysis/selection.json` | safeguards, selected=b0 |
| `artifacts/qa_chunk_selection_v3/analysis/positive_quality/summary.json` | оракул 73.01, D4/D5 gates |
| `artifacts/qa_chunk_selection_v3/annotation/queue.jsonl` | 400, скоры скрыты |
| `artifacts/qa_chunk_selection_v3_1_prep/` | ручной annotator pack, 0 filled |
| `artifacts/qa_chunk_selection_v3_1_auto/` | silver v1 |
| `artifacts/qa_chunk_selection_v3_1_auto_v2/` | silver v2 + Pareto |
| `artifacts/qa_chunk_selection_v3_1_auto_v2/analysis/stage_a_pareto.json` | кривые (R, easy) |
| `artifacts/qa_chunk_selection_v3_d45/analysis/d4/oracle_summary.json` | сводка окон |
| `artifacts/qa_chunk_selection_v3_d45/analysis/d4/oracle_rows.jsonl` | 1253 построчных окна |
| `artifacts/qa_chunk_selection_v3_d45/analysis/d5/audit.json` | кластеры |
| `experiments/qa_chunk_selection/v3_1/` | ranking-код, без артефактов |
| `logs/qa_chunk_selection_v3.log` | V3 wall-clock |
| `logs/qa_chunk_selection_v3_1_auto.log` | v1 35 мин |
| `logs/qa_chunk_selection_v3_1_auto_v2.log` | v2 9.5 с |
| `logs/qa_chunk_selection_v3_d45.log` | D4/D5 7 мин |
| `logs/qa_chunk_selection_v3_1_prep.log` | пустой freeze 4 сентября |

---

## 18. Одна строка дальше

```text
B0 stays the live ranker (F1 27.36 vs oracle 73.01).
H3.1 still untested: human overlay 0/400; silver v1/v2 = 0 TE/TM (gate fail).
Do not train V3.1-Silver. Do not relax (R=0.80, easy=0.25) post-hoc.
D4/D5 gates open but secondary (+5.4 window ceiling; 67.5% multi-cluster).
Next honest step: annotate the existing 400-queue.
INTERNAL_TEST stays locked. Do not overwrite E0/V2/V3/prep/auto/d45.
```
