# Результаты эксперимента V2: отбор evidence-чанков для extractive QA

Документ фиксирует **фактические результаты** прогона `experiments/qa_chunk_selection/v2` по протоколу [`qa_chunk_selection_experiments_v2.md`](qa_chunk_selection_experiments_v2.md). Это не автодамп JSON (`artifacts/qa_chunk_selection_v2/reports/results_v2.md`), а полный разбор: что запускали, как выбирали модели, какие числа получились и что из этого следует.

- Дата завершения: **2026-09-02 14:05:23** (wall-clock **2483 с ≈ 41 мин**, `cuda:0`).
- Артефакты: `artifacts/qa_chunk_selection_v2/`.
- Production (`untie/`, `datasets/`, Phase 1) **не менялся**. Пакет E0 (`experiments/qa_chunk_selection/` без `v2/`, `artifacts/qa_chunk_selection/`) **не перезаписывался**.
- Gold **никогда** не использовался в ранжировании, фильтрации, top-M, adaptive M, reader-aware fusion и агрегации ответа. Gold есть только в scored-метриках и в диагностическом reader oracle.

**INTERNAL_TEST не открыт.** Ниже нет ни одного числа с этого сплита, кроме описи состава. Freeze-final, D4, D5 и `final-eval` в default stages не входили и не запускались.

---

## 0. Короткий итог

Замороженный extractive reader **умеет** извлекать ответ, если ему отдать настоящий evidence-чанк. На DEVELOPMENT (77 answerable-вопросов) оракул «лучший gold-positive» даёт Token F1 **71.86**; у **76.6%** вопросов есть хотя бы один positive с Token F1 ≥ 50.

Живой пайплайн с выбранным ранкером даёт Token F1 **25.71**. Разрыв почти целиком в **выборе чанка**, а не в экстракции.

Честный победитель прогона — **B2 answerability**, вариант **b2a**, seed **44**, один top-1 чанк:

| | Hit@1 | HN@1 | pairwise_clean | Token F1 | EM | F1 \| pos | top-1 ambiguous |
|---|---:|---:|---:|---:|---:|---:|---:|
| DEVELOPMENT | 37.7% | 3.9% | 74.6% | 25.71 | 19.5% | 59.89 | 54.5% |

Остальные гипотезы на DEVELOPMENT **не удержали** выигрыш CALIBRATION:

- **R3_2** (B2 + QA-confidence proxy) выбран из-за чуть более высокого pairwise_clean (75.6%), но по QA хуже B2 (Hit@1 35.1%, F1 23.28). Бутстреп-ДИ разницы включает ноль.
- **R3_3** по QA лучше B2 (Hit@1 40.3%, F1 28.23), но **не выбран**: pairwise_clean 70.9%. Это артефакт ключа селекции `(pairwise_clean, Hit@1, F1|pos, Token F1)`.
- **C (сырые multi-passage логиты)** на CALIBRATION поднимает F1 до 36.12 (M=3 на B2) и **падает** на DEVELOPMENT до 20.51; F1|pos обрушивается с 59.89 до 30.50.
- **Adaptive M** почти всегда включает M=3 и повторяет тот же провал (DEV F1 21.62).
- **D2:** победитель D2a = чистый B2. RRF поднимает HN@1 до 9.1%. Section на DEV не меняет выбранный top-1. Generic CE A портит Hit@1 и F1.

На уже виденном официальном SciREX DEV (105 q, exploratory) V2-B2 **не бьёт** E0-B и тем более E0-D2. Это ожидаемо: B2 учился на 60% official TRAIN, а E0-D2 подбирали как раз на этом DEV.

---

## 1. Что это за эксперимент

UnTIE — extractive QA по научным статьям SciREX: для канонического aspect-вопроса (task / method / material / metric) нужно выбрать evidence-чанк текущей статьи и вытащить из него span.

E0 (`qa_chunk_selection_experiments_v1.md`) показал:

- generic relevance (A) путает positive с hard negative;
- answerability (B), обученная на robust_positive vs hard_negative, это почти единственный сигнал, который отличает их;
- multi-passage C помогает экстракции на DEV;
- lexical hybrid D1 вредит HN;
- structural mix D2 даёт лучший DEV Token F1, но хуже B по HN.

V2 проверяет это на **чистой схеме сплитов** (закрытый INTERNAL_TEST из official TRAIN), с отдельным CALIBRATION, без подбора параметров на official DEV, и с новыми гипотезами: менее переобученный B2, reader-aware reranking, sweep/adaptive M, абляции D2 без QA-proxy.

Режим кандидатов: **Mode II — все чанки текущей статьи** (`candidate_mode=all`). Ридер: frozen Phase 1 `ExtractiveQuestionAnswerer` (RoBERTa). Dense: MiniLM. Generic CE: `cross-encoder/ms-marco-MiniLM-L6-v2`. BERTScore отключён.

Запущенные стадии:

```text
prepare-v2, split-v2, e0-regression, reader-oracle, ambiguous-audit,
b2-train, b2-eval, ambiguous-audit-b2, reader-aware, c-sweep, c-adaptive,
d2-ablate, compare-v2, report-v2
```

Не запускались (не default): `d4-train`, `d4`, `d5`, `freeze-final`, `final-eval`, `unanswerable-audit`.

Конфиг прогона (стабильная часть + CLI): seed 42, split_seed 42, bootstrap_seed 42, n_bootstrap 5000, B2 epochs 5 / patience 2 / batch 16 / lr 2e-5 / max_length 512 / pos:hn:easy = 1:3:1, R3 K_pre ∈ {3,5,8}, M ∈ {1,2,3,5,8}, `include_hn_gold_warning=false`, `allow_internal_test=false`, `resume=true` (успешный прогон был продолжением после починки E0-R2 replay D2).

---

## 2. Как читать метрики

Шкалы зафиксированы в артефактах (`metric_scales`):

| метрика | шкала | смысл |
|---|---|---|
| Token F1 | 0..100 | токенное пересечение предсказанного спана с gold |
| Char F1 | 0..1 | символьное пересечение |
| ROUGE-L F1 | 0..1 | longest common subsequence |
| Exact match (EM) | 0..1 | совпадение нормализованных строк; в таблицах часто как % |
| Hit@K | 0..1 | среди top-K есть gold **robust_positive** |
| Coverage@K | 0..1 | доля **всех** positives вопроса, попавших в top-K |
| HN@K | 0..1 | доля слотов в top-K, занятых **hard_negative**: `\|top-K ∩ HN\| / K`. При K=1 это «top-1 есть HN». Из-за деления на K величина **не** обязана расти с K (это не P(хотя бы один HN в top-K)) |
| pairwise / pairwise_all_hn | 0..1 | доля пар (pos, HN), где score(pos) > score(HN); среднее по вопросам, у которых есть обе стороны |
| pairwise_clean_hn | 0..1 | то же, но HN без `hn_gold_string_warning` |
| F1 \| pos (`f1_given_selected_positive`) | 0..100 | Token F1 только если **выбранный ранжированием** чанк — positive |
| top-1 ambiguous | 0..1 | label выбранного чанка — ambiguous (не HN и не positive) |
| MRR | 0..1 | mean reciprocal rank первого positive |

**Ambiguous** — отдельный класс Phase 1, не hard negative. Подтипы: `gold_supporting_robust_negative`, `false_negative_candidate`, `question_sensitive` (и `confirmed_false_negative`, в этом прогоне в audit не выделялся как отдельная масса).

**QA-confidence proxy** — сырой span-score frozen reader. Это не P(correct) и не «extractability».

**C в V2** — не обученный Multi-passage BERT, а склейка top-M чанков и выбор спана **сырыми** start/end логитами.

Оракул использует gold **только как диагностический потолок**. Это не runnable-система.

DEVELOPMENT использовался для выбора варианта B2, семейства R3 и варианта D2. Его QA-числа **оптимистичны** относительно будущего INTERNAL_TEST. CALIBRATION использовался для early stopping, K_pre/fusion, M и порогов adaptive M.

---

## 3. Сплиты

Документный сплит official TRAIN, seed 42, **до** фильтра answerable. Official SciREX DEV целиком ушёл в `seen_exploratory_dev` и **не** участвовал в подборе параметров V2.

Счётчики ниже — из `data/data_summary.json`. Поле `aspects` считается по **всем** вопросам сплита (включая no-robust-positive), не только по answerable. Eval (кроме оракула на positives) идёт по answerable.

| сплит | статей | вопросов | answerable | без robust + | pos | HN | easy | amb | роль |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 180 | 477 | 303 | 174 | 824 | 479 | 4731 | 3223 | обучение B2 |
| calibration | 44 | 111 | 73 | 38 | 201 | 110 | 1054 | 813 | stop / K / M / τ |
| development | 44 | 126 | 77 | 49 | 228 | 125 | 1191 | 828 | выбор архитектуры |
| internal_test | 31 | 78 | 54 | 24 | 137 | 70 | 847 | 544 | **LOCKED** |
| seen_exploratory_dev | 65 | 163 | 105 | 58 | 267 | 179 | 1612 | 1049 | exploratory, виден в E0 |

Сумма TRAIN-кусков: 180+44+44+31 = **299** статей. У `seen_exploratory_dev` 65 статей в сплите, из них **57** содержат хотя бы один answerable-вопрос — это те же 57 DEV-бумаг, на которых считался E0.

Аспекты (все вопросы):

| сплит | method | task | metric | material |
|---|---:|---:|---:|---:|
| train | 119 | 135 | 111 | 112 |
| calibration | 29 | 36 | 23 | 23 |
| development | 29 | 33 | 36 | 28 |
| internal_test | 20 | 24 | 16 | 18 |
| seen_exploratory_dev | 40 | 47 | 35 | 41 |

HN-корзины **answerable**-вопросов (сколько HN у вопроса):

| сплит | 0 | 1 | 2–3 | 4+ |
|---|---:|---:|---:|---:|
| train | 115 | 35 | 119 | 34 |
| calibration | 30 | 9 | 24 | 10 |
| development | 27 | 13 | 25 | 12 |
| internal_test | 22 | 10 | 16 | 6 |
| seen_exploratory_dev | 38 | 9 | 44 | 14 |

Много вопросов без HN: pairwise на них не определён (в таблицах `NaN` по корзине 0). DEVELOPMENT answerable по аспектам (оракул): material 22, method 20, task 19, metric 16.

---

## 4. Качество hard negatives

`data/hn_quality_flags.jsonl`: **18 512** строк (question×chunk).

| label | count |
|---|---:|
| easy_negative | 9435 |
| ambiguous | 6457 |
| positive | 1657 |
| hard_negative | 963 |

`hn_gold_string_warning`: **23** HN, в тексте которых встречается нормализованная gold-строка. Исходный label **не переписывался**. Default V2: такие HN **не** `training_eligible_hn`.

`training_eligible_hn`: **940** = 963 − 23.

Предупреждения по сплитам: train 12, development 5, calibration 3, seen_exploratory_dev 3, internal_test 0.

Eligible HN по сплитам: train 467, seen_exploratory_dev 176, development 120, calibration 107, internal_test 70.

---

## 5. Регрессия E0

Два уровня, оба **ok**, mismatches пустые. Сравнение на официальном DEV E0 (105 answerable-вопросов на 57 статьях; в V2 тот же набор вопросов лежит в `seen_exploratory_dev`, вместе с 8 статьями без robust positive).

**E0-R1** — пересчёт метрик по сохранённым E0 predictions. **E0-R2** — replay инференса V2-кодом по E0 ranking mix (для D2: `structural_mix(RRF(dense,BM25), B, section)` с весами 0.5 / 0.3 / 0.2). Первая попытка R2 упала, потому что E0 не сохранил replay D2; после фикса и `RESUME=1` оба уровня сошлись **точно**.

| метод | Hit@1 | HN@1 | pairwise pos>HN | Token F1 | EM |
|---|---:|---:|---:|---:|---:|
| A (generic CE) | 22.86% | 12.38% | 48.81% | 22.90 | 17.1% |
| B (answerability) | 30.48% | 4.76% | 79.13% | 22.14 | 15.2% |
| C (multi-passage на B) | 30.48% | 4.76% | 79.13% | 25.81 | 19.0% |
| D1 (hybrid) | 19.05% | 20.00% | 41.88% | 19.83 | 15.2% |
| D2 (+structure) | 34.29% | 12.38% | 59.52% | 28.89 | 21.9% |

Hit@1 / Token F1 / HN@1(B) / pairwise(B) / D1 Hit@1 / D2 Hit@1+F1 — из V2 `e0_regression.json` (exact match). HN/pairwise/EM для A, C, D1, D2 дополнены из E0 `runs/*/42/dev/metrics.json`. C не меняет ранжирование B, поэтому Hit@1/HN@1/pairwise у C совпадают с B.

Это якорь: код метрик и replay не съехали относительно E0. Заодно видно, что E0-D2 выиграл Token F1 ценой HN@1 как у generic A (12.4%), а не как у B (4.8%).

---

## 6. Reader oracle (только диагностика)

Frozen reader прогоняется по **каждому gold-positive** чанку; для вопроса берётся лучший. INTERNAL_TEST не трогали. Суммарно **558** вопросов (train 303 + cal 73 + DEV 77 + seen DEV 105), 1520 QA cache miss на первом проходе.

Глобально: reader_solvable EM **70.25%**, F1>0 **85.66%**, F1≥50 **77.96%**.

По сплитам:

| сплит | n Q | n pos | best-pos Token F1 | EM | F1>0 | F1≥50 | F1≥80 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 303 | 824 | 72.60 | 71.9% | 83.2% | 74.9% | 63.4% |
| calibration | 73 | 201 | 75.94 | 67.1% | 90.4% | 82.2% | 60.3% |
| development | 77 | 228 | 71.86 | 68.8% | 84.4% | 76.6% | 61.0% |
| seen_exploratory_dev | 105 | 267 | 78.75 | 68.6% | 90.5% | 84.8% | 66.7% |

DEVELOPMENT по аспектам:

| аспект | n | oracle EM | best-pos Token F1 |
|---|---:|---:|---:|
| material | 22 | 90.9% | 65.53 |
| method | 20 | 75.0% | 72.00 |
| task | 19 | 68.4% | 87.02 |
| metric | 16 | 31.3% | 62.40 |

**Вывод.** На ~77% DEVELOPMENT-вопросов ридер уже достаёт F1 ≥ 50 с какого-то positive. Живой B2 попадает в positive только в 37.7% случаев, и тогда F1|pos = 59.89 против оракула 71.86. Основной потолок — **какой** чанк (и какой из нескольких positives). Аспект **metric** тяжёлый даже на gold-positive: это ограничение ридера, не ранкера.

---

## 7. B2 — обучение answerability

Backbone тот же MiniLM cross-encoder, что в E0-B, но:

- negatives только `training_eligible_hn`;
- три рецепта × семена 42/43/44;
- early stopping по CALIBRATION `pairwise_clean_hn`, patience 2, max 5 эпох;
- **вариант** (b2a/b2b/b2c) выбирается по **среднему DEVELOPMENT pairwise_clean** по трём seed;
- **checkpoint** внутри победившего варианта — seed с лучшим CALIBRATION pairwise_clean;
- Token F1 **не** участвует в выборе B2 (чтобы не подгонять answerability под ридер до сравнения с R3).

Рецепты:

- **b2a** — question-balanced weighted BCE, уникальные пары, без оверсэмпла HN. У выбранного seed 44 на TRAIN: 303 вопроса, 3473 пары, вес каждого вопроса одинаковый (`question_weight_spread ≈ 1`).
- **b2b** — BCE, каждый epoch заново сэмплирует eligible HN.
- **b2c** — pairwise loss `-log sigmoid(s_pos − s_neg)`, question-balanced triplets.

### 7.1. Выбор варианта

DEVELOPMENT `pairwise_clean` (n=3 seed, sample std):

| вариант | seed 42 | seed 43 | seed 44 | mean ± std |
|---|---:|---:|---:|---:|
| **b2a** | 0.7324 | 0.8047 | 0.7461 | **0.7611 ± 0.0384** |
| b2b | 0.7514 | 0.7393 | 0.7317 | 0.7408 ± 0.0099 |
| b2c | 0.7161 | 0.7315 | 0.7240 | 0.7239 ± 0.0077 |

Победитель: **b2a**. Среди b2a CALIBRATION pairwise_clean: seed 42 = 0.7556, 43 = 0.7659, **44 = 0.7671** → downstream seed **44**, best epoch **2**.

b2a seed 43 — предостережение: TRAIN pairwise_clean **0.9840** при CAL 0.7659 и DEV 0.8047. Сильный TRAIN-overfit; протокол правильно не взял его по CAL (и не должен был брать по DEV: seed выбирается на CALIBRATION).

b2c: TRAIN pairwise 0.84–0.99, но DEVELOPMENT PR-AUC 0.475–0.506 (у b2a 0.577–0.650). Pairwise loss оптимизирует не калибровку ранжирования.

DEVELOPMENT PR-AUC (для полноты):

| вариант | 42 | 43 | 44 |
|---|---:|---:|---:|
| b2a | 0.618 | 0.650 | 0.577 |
| b2b | 0.600 | 0.605 | 0.593 |
| b2c | 0.506 | 0.475 | 0.486 |

Лучшая epoch / CAL pairwise_clean:

| | 42 | 43 | 44 |
|---|---|---|---|
| b2a | ep 2 / 0.7556 | ep 5 / 0.7659 | **ep 2 / 0.7671** |
| b2b | ep 2 / 0.7712 | ep 2 / 0.7681 | ep 2 / 0.7587 |
| b2c | ep 4 / 0.7722 | ep 2 / 0.7571 | ep 2 / 0.7887 |

Замечание: b2c seed 44 имеет **лучший CAL pairwise среди всех девяти прогонов** (0.7887), но вариант b2c проигрывает по mean DEV, поэтому checkpoint отброшен. Селекция вариант-на-DEV / seed-на-CAL здесь сработала как задумано.

Полная таблица PR-AUC / pairwise / threshold-F1 — `models/b2/selection.json`. Порог в этом файле — диагностический operating point классификатора, **не** используется в ranking mode: в инференсе B2 отдаёт **сигмоид-вероятность** (не сырой logit) и ранжирует все чанки без отсечения.

### 7.2. Downstream QA выбранного B2 (b2a, seed 44)

Ранжирование: score B2 по всем чанкам статьи, top-1 в ридер. `qa_forward_passes=1`, mean `reader_tokens` на DEV ≈ 279, mean `candidate_count` ≈ 19.8, mean `n_positive` ≈ 2.96.

| сплит | n | Hit@1 | Hit@5 | Hit@10 | Cov@1 | Cov@5 | HN@1 | HN@5† | pairwise_clean | MRR | Token F1 | Char F1 | ROUGE-L | EM | F1\|pos | top-1 amb. | t, с |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 303 | 41.9% | 84.2% | 93.4% | 18.7% | 63.6% | 4.6% | 6.7% | 82.1% | 0.584 | 29.46 | 0.548 | 0.371 | 24.1% | 60.19 | 48.8% | 12.5 |
| calibration | 73 | 31.5% | 82.2% | 93.2% | 14.8% | 58.8% | 2.7% | 9.6% | 76.7% | 0.519 | 26.83 | 0.502 | 0.287 | 15.1% | 66.71 | 60.3% | 2.9 |
| **development** | 77 | **37.7%** | 75.3% | 93.5% | 16.6% | 49.4% | **3.9%** | 8.8% | **74.6%** | 0.550 | **25.71** | 0.499 | 0.352 | 19.5% | 59.89 | **54.5%** | 2.8 |
| seen_exploratory_dev | 105 | 25.7% | 72.4% | 92.4% | 10.3% | 50.8% | 6.7% | 8.6% | 70.0% | 0.475 | 20.89 | 0.489 | 0.266 | 16.2% | 63.05 | 60.0% | 5.5 |

† HN@5 здесь — средняя доля HN среди пяти верхних слотов, не «есть ли HN в top-5».

Train–DEV gap выбранного чекпоинта: pairwise_clean 82.1% → 74.6% (−7.5 п.п.), Token F1 29.46 → 25.71.

DEVELOPMENT по HN-корзинам:

| корзина | n | Hit@1 | HN@1 | pairwise (all) |
|---|---:|---:|---:|---:|
| 0 HN | 27 | 51.9% | 0 | — |
| 1 HN | 13 | 23.1% | 7.7% | 89.8% |
| 2–3 HN | 25 | 28.0% | 0 | 69.9% |
| 4+ HN | 12 | 41.7% | 16.7% | 72.2% |

Корзина 4+ маленькая: Hit@1 не монотонно падает. HN@1 концентрируется там, где HN много.

DEVELOPMENT по аспектам (B2 top-1):

| аспект | n | Hit@1 | HN@1 | Token F1 | EM | F1\|pos (при Hit@1) | top-1 label |
|---|---:|---:|---:|---:|---:|---:|---|
| material | 22 | 54.5% | 0 | 28.11 | 31.8% | 49.44 | pos 12, amb 9, easy 1 |
| method | 20 | 25.0% | 10.0% | 17.98 | 20.0% | 60.00 | amb 11, pos 5, easy 2, HN 2 |
| task | 19 | 42.1% | 0 | 41.16 | 21.1% | 82.52 | amb 11, pos 8 |
| metric | 16 | 25.0% | 6.2% | 13.73 | 0 | 45.83 | amb 11, pos 4, HN 1 |

Сводка top-1 на DEVELOPMENT (77): **ambiguous 42, positive 29, easy_negative 3, hard_negative 3**. HN редки; больше половины ошибок — ambiguous. Primary class выбранного чанка: robust_positive 29, gold_supporting_robust_negative 27, false_negative_candidate 9, question_sensitive 6, easy 3, HN 3.

На seen official DEV B2 слабее E0-B (Hit@1 25.7% vs 30.5%, F1 20.89 vs 22.14, HN@1 6.7% vs 4.8%, pairwise_all 69.2% vs 79.1%). B2 видел 303 answerable TRAIN-вопроса вместо полного official TRAIN E0.

---

## 8. Ambiguous audit

Два прохода: e0-ранжирование (A/B-скоры с E0-моделями) и b2-ранжирование. Корневые `summary.json` содержат только `{tag, n: 4650}` — это слабый агрегат; рабочие цифры в per-split `summary.json`.

`n_ambiguous_rows` — ambiguous-чанки **answerable**-вопросов (меньше, чем `data_summary.n_ambiguous`, куда входят и unanswerable).

### 8.1. DEVELOPMENT (651 строк, одинаковый набор у e0 и b2)

| subtype | count | e0 top-1 | b2 top-1 | e0 top-5 | b2 top-5 | mean B (e0) | mean B (b2) | mean Token F1 с чанка |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gold_supporting_robust_negative | 328 | 5.8% | **8.2%** | 32.3% | 32.3% | 0.592 | 0.431 | 3.83 |
| false_negative_candidate | 246 | 2.0% | 3.7% | 17.9% | 22.0% | 0.310 | 0.306 | 2.65 |
| question_sensitive | 77 | 9.1% | 7.8% | 44.2% | 49.4% | 0.659 | 0.474 | 7.76 |

B2 **чаще**, чем E0-B, ставит gold-supporting robust negative на top-1 (8.2% vs 5.8%). Это согласуется с 27/77 DEV-вопросов, у которых top-1 — именно этот subtype. Generic CE A на этих чанках сильно отрицательный (mean A ≈ −8.4): «не релевантно» по MS MARCO, но answerability их поднимает.

Колонка Token F1 — среднее по **всем** ambiguous-чанкам подтипа (ридер гоняется по каждому из них), а не только по тем, что попали в top-1. Поэтому 3.83 у GSRN не противоречит top-1 rate 8.2%: большинство этих чанков просто не выбираются. EM почти всегда 0, кроме question_sensitive (редко ненулевой overlap с gold-строкой).

### 8.2. Другие сплиты (B2)

| сплит | n rows | GSRN n / top-1 | FN-cand n / top-1 | q-sens n / top-1 |
|---|---:|---|---|---|
| train | 2520 | 1205 / 7.6% | 968 / 3.1% | 347 / 7.5% |
| calibration | 628 | 321 / 8.4% | 220 / 2.3% | 87 / 13.8% |
| seen_exploratory_dev | 851 | 419 / 9.1% | 322 / 3.4% | 110 / 12.7% |

Паттерн устойчив: главный distractor — **gold-supporting robust negative** (другой датасет / соседний бенчмарк в том же абзаце). Примеры из `analysis/ambiguous_audit/b2/top_errors.md` (highest answerability, labels frozen, это не переразметка): SNLI вместо целевого корпуса, SVHN/CIFAR вместо нужного, TREC-CAR vs MS MARCO, WikiText-2 vs Hutter Prize, ImageNet vs CIFAR-10.

---

## 9. R3 — reader-aware reranking

Каналы (нормализация minmax или rank, затем взвешенная сумма; zero-variance обрабатывается):

```text
score = wA * A_norm + wB * B2_norm + wQ * QA_norm
```

Фиксированные веса (не тюнились численно, только выбор архитектуры и K_pre/fusion):

| имя | (wA, wB, wQ) | смысл |
|---|---|---|
| R3_0 | (0, 1, 0) | = B2 |
| R3_1 | (0.3, 0.7, 0) | B2 + generic CE |
| R3_2 | (0, 0.6, 0.4) | B2 + QA proxy |
| R3_3 | (0.2, 0.5, 0.3) | все три |

Режим **topK**: QA-proxy только для pre-shortlist B2 размера K_pre ∈ {3,5,8}. Если wQ=0, QA-forwards на ранжировании пропускаются. **R3-full** (QA по всем чанкам) — только диагностика стоимости.

Протокол: K_pre и fusion на CALIBRATION; архитектура R3_0..R3_3 на DEVELOPMENT. Фактический ключ в коде `_selection_key`:

```text
(pairwise_clean_hn, hit@1, f1_given_selected_positive, token_f1)
```

То есть **первым** идёт pairwise, хотя §9.6 протокола перечисляет Hit@1 / F1|pos / Token F1 среди главных QA-метрик без такого приоритета. Это важно для интерпретации победителя.

### 9.1. CALIBRATION: что выбрано внутри каждой архитектуры

Для каждой архитектуры на CAL берётся max по `_selection_key` (24 точки: 4 имени × 2 fusion × 3 K). R3_0 идентичен при любом K/fusion — это B2.

Выбранные CAL-конфиги, затем оценённые на DEVELOPMENT:

| имя | выбранные K_pre / fusion | почему (CAL pairwise_clean) |
|---|---|---|
| R3_0 | 8 / minmax (всё равно) | 0.7671 |
| R3_1 | **3 / minmax** | 0.7202 — лучший pairwise у R3_1; при этом CAL Hit@1 худший в семействе (21.9%) |
| R3_2 | **5 / minmax** | 0.7548; rank K=8 имел CAL F1 33.29 vs 29.96 у выбранного, но pairwise ниже (0.739) |
| R3_3 | **5 / minmax** | 0.7151 |

CALIBRATION Token F1 (фрагмент, полный грид в `analysis/reader_aware/summary.json`):

QA-proxy на CAL **поднимает F1** и **поднимает HN@1**:

| точка | Hit@1 | HN@1 | Token F1 | pairwise_clean | F1\|pos |
|---|---:|---:|---:|---:|---:|
| R3_0 (B2) | 31.5% | 2.7% | 26.83 | 76.7% | 66.71 |
| R3_2 minmax K=5 | 31.5% | 6.8% | 29.96 | 75.5% | 70.62 |
| R3_2 rank K=8 | 37.0% | 5.5% | **33.29** | 73.9% | 73.12 |
| R3_2 minmax K=3 | 37.0% | 6.8% | 32.40 | 74.1% | 71.89 |
| R3_1 minmax K=8 | 23.3% | 4.1% | 23.41 | 71.0% | 74.34 |
| R3_3 minmax K=3 | 35.6% | 5.5% | 31.03 | 70.6% | 70.81 |

### 9.2. DEVELOPMENT: выбор архитектуры

| система | Hit@1 | Hit@5 | Cov@5 | HN@1 | pairwise_clean | MRR | Token F1 | Char F1 | ROUGE-L | EM | F1\|pos | top-1 amb. | t, с |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 / R3_0 | **37.7%** | 75.3% | 49.4% | **3.9%** | 74.6% | 0.550 | **25.71** | 0.499 | 0.352 | 19.5% | **59.89** | 54.5% | 2.8 |
| R3_1 | 31.2% | 76.6% | 49.3% | 7.8% | 67.2% | 0.512 | 22.02 | 0.455 | 0.304 | 15.6% | 58.15 | 58.4% | 3.7 |
| **R3_2 выбран** | 35.1% | 75.3% | 49.4% | 6.5% | **75.6%** | 0.531 | 23.28 | 0.476 | 0.327 | 19.5% | 54.07 | 46.8% | 17.5 |
| R3_3 не выбран | **40.3%** | 75.3% | 49.4% | 7.8% | 70.9% | 0.561 | **28.23** | 0.512 | 0.377 | **23.4%** | 60.40 | 46.8% | 2.7* |
| R3-full (диагн.) | 37.7% | 80.5% | 56.0% | 6.5% | 72.2% | 0.546 | 24.13 | 0.491 | 0.336 | 20.8% | 57.90 | 50.6% | 129.9 |

\*Время R3_3 на DEV занижено: QA-кэш уже тёплый после R3_2 (тот же K=5). Честная цена QA-shortlist — порядок **17.5 с** на 77 вопросов vs **2.8 с** у B2. R3-full — 130 с, Hit@1 как у B2, F1 24.13 всё ещё ниже B2. Полный проход QA **не** открывает большой extraction win.

Выбран R3_2, потому что pairwise_clean 0.7564 > 0.7461 у B2. По QA R3_2 **хуже** B2. R3_3 лучше B2 по Hit@1 / F1 / EM / F1|pos, хуже по HN@1 и pairwise.

Ambiguous top-1 падает с 54.5% (42/77 у B2) до 46.8% (36/77 у R3_2 и у R3_3), но это **разная** перекладка. R3_2: positive 29→27, easy 3→9, HN 3→5 — ambiguous уходит в easy/HN, positives чуть теряются. R3_3: positive 29→31, HN 3→6, easy 3→4 — часть ambiguous становится positive, часть HN.

R3_2 по аспектам vs B2 (Token F1): material 24.70 (было 28.11), method 11.59 (17.98), task 38.65 (41.16), metric 17.71 (13.73). Единственный аспект, где выбранный R3 не хуже — metric, и то слабо. У **невыбранного** R3_3 metric F1 23.01, material 31.97, task 40.53 — ближе к полезному сдвигу.

R3_2 HN-корзины DEV: 0 HN Hit@1 51.9%; 1 HN 38.5% (лучше B2 23.1%); 2–3 24.0% HN@1 8%; 4+ Hit@1 16.7% HN@1 25%. QA-proxy сильнее вредит, когда HN много.

### 9.3. seen_exploratory_dev (не для выбора)

| | Hit@1 | HN@1 | pairwise_clean | Token F1 | EM | F1\|pos | top-1 amb. |
|---|---:|---:|---:|---:|---:|---:|---:|
| B2 | 25.7% | 6.7% | 70.0% | 20.89 | 16.2% | 63.05 | 60.0% |
| R3_2 | 23.8% | 8.6% | 69.2% | 18.90 | 14.3% | 55.36 | 61.9% |

Направление то же: выбранный R3 не лучше B2.

### 9.4. Парное сравнение R3_2 − B2

Clustered bootstrap по **документу**, 5000 реплик, 1 seed, n=77 вопросов, 40 статей (`analysis/paired_comparisons.json`):

| метрика | mean Δ | 95% CI |
|---|---:|---|
| Token F1 | **−2.42** | [−12.58, +7.54] |
| Hit@1 | **−0.026** | [−0.141, +0.082] |

Оба интервала содержат 0. На этой выборке нельзя утверждать, что R3_2 хуже или лучше B2; точечная оценка против R3_2. Вывод «R3 помогает» **не поддержан**.

---

## 10. C — sweep по числу пассажей

Ранжирование фиксировано (B2 или выбранный R3). Меняется только то, сколько top-чанков склеивается в ридер. Hit@K ранжирования при этом **не меняется** (это видно: Hit@1 B2+C совпадает с B2). Меняется span. M выбирается по CALIBRATION Token F1, затем выбранный M подтверждается на DEVELOPMENT.

Это **не** Multi-passage BERT.

### 10.1. Сетка CALIBRATION

| ranking | M=1 | M=2 | M=3 | M=5 | M=8 |
|---|---:|---:|---:|---:|---:|
| B2 Token F1 | 26.83 | 29.04 | **36.12** | 31.71 | 26.88 |
| B2 EM | 15.1% | 19.2% | **24.7%** | 20.5% | 16.4% |
| B2 F1\|pos | 66.71 | 55.98 | 56.71 | 47.58 | 31.64 |
| R3_2 Token F1 | 29.96 | **33.92** | 32.40 | 31.71 | 26.88 |
| R3_2 EM | 19.2% | **23.3%** | 20.5% | 20.5% | 16.4% |
| R3_2 F1\|pos | 70.62 | 71.35 | 67.00 | 67.00 | 44.47 |

Выбрано: B2 **M\*=3**, R3_2 **M\*=2**. Первичный C в метке `c_meta` привязан к `r3_selected` → M=2. Hit@1 на всей сетке совпадает с ранкером (ранжирование не трогают).

F1|pos на CAL в целом ниже при большом M (B2: 66.71 при M=1 → 31.64 при M=8), но не строго монотонен (M=3 56.71 чуть выше M=2 55.98). Лишние пассажи дают ридеру альтернативные спаны даже когда ranking top-1 — positive.

### 10.2. DEVELOPMENT, выбранные M

| система | Token F1 | EM | F1\|pos | Char F1 | ROUGE-L | used_m | qa_forwards | reader_tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 top-1 (M=1) | **25.71** | **19.5%** | **59.89** | 0.499 | 0.352 | 1 | 1 | ~279 |
| B2 + M=3 | 20.51 | 13.0% | 30.50 | 0.436 | 0.274 | 3 | 3 | ~1124 |
| R3_2 top-1 | 23.28 | 19.5% | 54.07 | 0.476 | 0.327 | 1 | 1 | ~278 |
| R3_2 + M=2 | 21.62 | 16.9% | 48.02 | 0.465 | 0.298 | 2 | 2 | ~738 |

У B2+M=3 ranking labels те же (42 amb / 29 pos / 3 easy / 3 HN), но `qa_chunk_primary_class` другой: robust_positive **24** (не 29), HN как источник спана 7, easy 7. Ридер **забирает span не из top-1**, в том числе у вопросов, где top-1 был positive. Отсюда обвал F1|pos 59.89 → 30.50 при том же Hit@1.

CALIBRATION +9.3 F1 при M=3 **не переносится**. Это та же логика, что E0-C (там C поднял official DEV F1 22.14→25.81 и чуть опустил TRAIN 36.59→34.32): здесь held-out DEVELOPMENT показывает вред. E0-C нельзя считать подтверждённым на новой схеме сплитов.

---

## 11. Adaptive M

Политика детерминированная по B2-скорам (min-max внутри кандидатского набора): если нормализованный top-1 ≥ τ_conf **и** (top1−top2) ≥ τ_margin → M=1, иначе M_uncertain ∈ {3,5}.

Сетка CALIBRATION: τ_conf ∈ {0.55, 0.70, 0.85} × τ_margin ∈ {0.05, 0.15, 0.30} × M_uncertain ∈ {3,5} = 18 точек.

**τ_conf полностью мёртв:** 0.55 / 0.70 / 0.85 дают **идентичные** гистограммы M и F1. После min-max top-скор часто ≈ 1, порог уверенности не срабатывает. Живёт только τ_margin.

| τ_margin | M_unc | hist CAL (n=73) | Token F1 | EM |
|---|---|---|---:|---:|
| 0.05 | 3 | M1=48, M3=25 | 28.94 | 16.4% |
| 0.15 | 3 | M1=18, M3=55 | 31.30 | 19.2% |
| **0.30** | **3** | **M1=6, M3=67** | **34.18** | **23.3%** |
| 0.30 | 5 | M1=6, M5=67 | 31.14 | 20.5% |

Победитель почти равен fixed M=3 (36.12): политика в 67/73 случаях и так берёт M=3.

DEVELOPMENT выбранной политики: Token F1 **21.62**, EM **13.0%**, F1|pos **33.45**, hist **M=3 на 70 / M=1 на 7** из 77. Mean reader tokens ≈ 1058 против ≈ 1124 у always-M=3 и ≈ 279 у B2 top-1: экономия относительно fixed M=3 невелика, до top-1 B2 далеко. Adaptive **не лучше** B2 top-1 по F1.

---

## 12. D2 — абляции старой смеси (без QA-proxy)

Веса fusion a priori, не перетюнивались per variant (`weights_source: fixed_a_priori`). Архитектура по DEVELOPMENT Token F1.

| id | состав |
|---|---|
| D2a | B2 |
| D2b | B2 + RRF(dense, BM25) |
| D2c | B2 + section |
| D2d | B2 + RRF + section (ближайший к E0-D2) |
| D2e | B2 + generic CE A |
| D2f | B2 + A + section |

DEVELOPMENT:

| | Token F1 | Hit@1 | HN@1 | pairwise_clean | EM | F1\|pos | top-1 amb. |
|---|---:|---:|---:|---:|---:|---:|---:|
| **D2a = B2** | **25.71** | **37.7%** | **3.9%** | **74.6%** | **19.5%** | 59.89 | 54.5% |
| D2c +section | **25.71** | 37.7% | 3.9% | 74.6% | 19.5% | 59.89 | 54.5% |
| D2b +RRF | 24.51 | 33.8% | **9.1%** | 63.6% | 18.2% | 59.91 | 51.9% |
| D2d RRF+sec | 24.51 | 33.8% | 9.1% | 63.6% | 18.2% | 59.91 | 51.9% |
| D2e +A | 22.02 | 32.5% | 7.8% | 68.1% | 16.9% | 55.83 | 57.1% |
| D2f A+sec | 22.02 | 32.5% | 7.8% | 68.1% | 16.9% | 55.83 | 57.1% |

Section на этих весах почти не двигает **top-1** на DEVELOPMENT: D2c совпадает с D2a по Hit@1 / HN@1 / Token F1 / EM / pairwise / amb (MRR 0.549 vs 0.550). D2d совпадает с D2b по тем же колонкам таблицы. D2f совпадает с D2e по ним же, но Hit@5 уже разный (75.3% vs 76.6%). На CALIBRATION section чуть снижает pairwise (D2c 76.4% vs D2a 76.7%). Это не буквальный no-op ранжирования, но выбранный чанк на DEV у D2a/D2c один и тот же.

RRF воспроизводит патологию E0-D1: HN@1 3.9% → 9.1%, pairwise 74.6% → 63.6%. Generic CE A тоже размывает answerability.

CALIBRATION та же картина: D2a/D2c F1 26.83; D2b/D2d 20.91 Hit@1 21.9%; D2e/D2f 23.51 Hit@1 24.7%.

E0-D2 Token F1 28.89 на official DEV **не воспроизводится** как апгрейд над B2 на V2-DEVELOPMENT. Другой бэкбон (B2 vs E0-B), другой сплит, DEV больше не селекционный для этой смеси, QA-proxy в D2 сознательно не мешали (он в R3).

---

## 13. Сводка живых систем vs оракул (DEVELOPMENT, n=77)

Token F1, 0..100:

| система | статус выбора | Token F1 | Hit@1 | HN@1 |
|---|---|---:|---:|---:|
| Oracle best-positive | диагностика | 71.86 | — | — |
| R3_3 | не выбран (pairwise) | 28.23 | 40.3% | 7.8% |
| **B2 / D2a top-1** | **выбран честный кандидат** | **25.71** | **37.7%** | **3.9%** |
| R3-full | диагностика стоимости | 24.13 | 37.7% | 6.5% |
| R3_2 | выбран протоколом R3 | 23.28 | 35.1% | 6.5% |
| R3_1 | проигравший R3 | 22.02 | 31.2% | 7.8% |
| Adaptive M | выбран в C3 | 21.62 | 37.7%* | 3.9%* |
| R3_2 + M=2 | выбран в C2 для R3 | 21.62 | 35.1%* | 6.5%* |
| B2 + M=3 | выбран в C2 для B2 | 20.51 | 37.7%* | 3.9%* |

\*Hit/HN у C/adaptive — свойства ранкера, не экстракции.

---

## 14. Сравнение с E0 на seen official DEV (n=105, exploratory)

Не чистый тест: E0 смотрел этот DEV, V2 — нет (только post-hoc).

| система | Hit@1 | HN@1 | pairwise (all HN) | Token F1 | EM |
|---|---:|---:|---:|---:|---:|
| E0-A | 22.9% | 12.4% | 48.8% | 22.90 | 17.1% |
| E0-B | 30.5% | 4.8% | 79.1% | 22.14 | 15.2% |
| E0-C | 30.5% | 4.8% | 79.1% | 25.81 | 19.0% |
| E0-D1 | 19.0% | 20.0% | 41.9% | 19.83 | 15.2% |
| E0-D2 | 34.3% | 12.4% | 59.5% | **28.89** | 21.9% |
| V2-B2 | 25.7% | 6.7% | 69.2% | 20.89 | 16.2% |
| V2-R3_2 | 23.8% | 8.6% | 68.4% | 18.90 | 14.3% |

Pairwise здесь везде `pairwise_pos_gt_hn` / `pairwise_all_hn` (все исходные HN, не clean-only). V2-B2 `pairwise_clean` на этом сете — 70.0%. Источники: E0 `runs/{a,b,c,d1,d2}/42/dev/metrics.json`, V2 `runs/b2/44/seen_exploratory_dev` и `runs/reader_aware_topk/selected/seen_exploratory_dev`.

V2 не претендует на победу над E0-D2 на этом сете. Другой train-объём, другая цель (чистый INTERNAL_TEST ещё закрыт). E0-D2 к тому же имеет HN@1 12.4% — как A, не как B.

---

## 15. Ресурсы и тайминг

Общий wall-clock 2483 с, финальный QA-cache hit rate **70.7%**.

Приблизительные границы стадий по `progress.jsonl` (последняя отметка стадии):

| стадия | к моменту, с | комментарий |
|---|---:|---|
| prepare + split | 2 | |
| e0-regression | 156 | R1+R2, включая D2 replay |
| reader-oracle | 508 | 1520 reader calls |
| ambiguous-audit e0 | 1630 | самый дорогой кусок |
| b2-train (9 чекпоинтов) | 1850 | |
| b2-eval + audit b2 | 2029 | |
| reader-aware (+ R3-full) | 2303 | |
| c-sweep | 2357 | |
| c-adaptive | 2446 | |
| d2-ablate + compare + report | 2483 | |

Пик GPU ≈ 2.3–2.4 ГБ. DEV B2 2.8 с / 77 q; R3_2 topK 17.5 с; R3-full 130 с; C M=3 3.8 с wall (но 3 forwards и ~1124 token/q vs ~279).

---

## 16. Что не запускалось

- **INTERNAL_TEST** — 31 статья, 54 answerable. Замок: нужен `freeze-final` + `--allow-internal-test`.
- **D4** (`d4-train` / `d4`) sentence-level evidence — not run.
- **D5** (`d5`) answer aggregation — not run.
- D1 как отдельная default-ветка — выключен; BM25 только внутри D2b/D2d.
- Обученный Multi-passage BERT — не в скоупе.
- Reader-success meta-ranker (§9.7) — не в default.

Автоотчёт `artifacts/qa_chunk_selection_v2/reports/results_v2.md` обрезан (сырой JSON, секции D4/D5 пустые, B2/R3/C не дописаны до конца). Настоящий разбор — этот файл.

---

## 17. Во что можно верить и чего нельзя утверждать

**Поддержано этим прогоном (с оговоркой, что DEVELOPMENT — селекционный сплит):**

1. Frozen reader не главный потолок: оракул F1 71.86 vs live 25.71.
2. Answerability B2 держит низкий HN@1 (~4%) и Hit@1 37.7% на DEV; generic CE и RRF это портят.
3. Главный top-1 distractor — ambiguous, особенно gold-supporting robust negatives, не HN.
4. Сырой QA-confidence proxy на CALIBRATION поднимает F1 и HN@1; на DEVELOPMENT выбранный R3_2 не лучше B2, ДИ включает 0.
5. Ключ селекции с pairwise первым может выбрать худшую QA-систему (R3_2 вместо R3_3).
6. Склейка пассажей сырыми логитами переобучает M на CALIBRATION и вредит DEVELOPMENT (F1|pos 59.89 → 30.50).
7. Adaptive M при min-max нормализации top-скора здесь вырождается в almost-always-M=3.
8. E0 structural/lexical mix поверх B2 не апгрейд.
9. Код метрик совместим с E0 (R1 и R2 exact match).

**Нельзя утверждать:**

- что B2 — победитель на независимом тесте (INTERNAL_TEST закрыт);
- что R3_3 «на самом деле лучше» — это post-hoc по QA-метрикам на том же DEVELOPMENT, где уже выбирали архитектуру;
- что V2 улучшил E0 на official DEV;
- что можно менять production ranking.

Если когда-нибудь открывать INTERNAL_TEST, замороженный кандидат по этому прогону: **B2 b2a seed 44, ranking mode, M=1, без R3/C/D2**. R3_3 — отдельная exploratory гипотеза с **другим** ключом селекции, не текущий выбранный pipeline.

---

## 18. Указатель артефактов

| путь | содержимое |
|---|---|
| `artifacts/qa_chunk_selection_v2/config_resolved.json` | полный CLI+defaults |
| `contract.json` | инварианты (no gold, lock, reuse E0 modules) |
| `data/data_summary.json`, `splits_v2.json`, `hn_quality_flags.jsonl` | сплиты и флаги |
| `regression/e0_regression.json` | R1/R2 |
| `analysis/reader_oracle/` | оракул per split |
| `analysis/ambiguous_audit/{e0,b2}/` | subtype + `top_errors.md` |
| `models/b2/selection.json` | все seed/variant + выбранный |
| `models/b2/selected/` | frozen checkpoint b2a/44 |
| `runs/b2/44/{split}/metrics.json` | QA после B2 |
| `analysis/reader_aware/summary.json` | грид R3 |
| `runs/reader_aware_topk/selected/` | R3_2 |
| `runs/reader_aware_full/development/` | R3-full |
| `analysis/c_sweep/summary.json` | M-сетка |
| `analysis/c_adaptive/summary.json` | τ-сетка |
| `analysis/d2_ablation/summary.json` | D2a..f |
| `analysis/paired_comparisons.json` | bootstrap R3−B2 |
| `progress.jsonl` | тайминг |
| `reports/results_v2.md` | автодамп, неполный |

Протокол: [`hypothesis/qa_chunk_selection_experiments_v2.md`](qa_chunk_selection_experiments_v2.md).
