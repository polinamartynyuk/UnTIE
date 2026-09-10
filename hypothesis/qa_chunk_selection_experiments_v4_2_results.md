# Результаты эксперимента V4.2: selected-head attention поверх V6-K5

Документ фиксирует **фактические результаты** прогона `experiments/qa_chunk_selection/v4_2` по протоколу [`v4_2_selected_head_attention_protocol.md`](v4_2_selected_head_attention_protocol.md). Это не автодамп JSON (`artifacts/qa_chunk_selection_v4_2/reports/results_v4_2.md`), а полный разбор: что запускали, какие числа честные, что протокол разрешает считать подтверждённым, а что нет.

- Старт: **2026-09-09 11:51:31**, конец: **2026-09-09 12:40:44**.
- Wall-clock: **2953.0 с ≈ 49 мин 13 с**, `auto→cuda:0`, `exit=0`.
- Артефакты: `artifacts/qa_chunk_selection_v4_2/`.
- Лог: `logs/qa_chunk_selection_v4_2.log`.
- Стадии этого прогона: `STAGES=primary` = init, r0, cache, core, att-reg, att-extract, heads, att1, att2, att3, compare, freeze, report.
- ATT2 **пропущен**: ATT1 не прошёл primary gate. Exploratory `--force-att2` не включался.
- ATT3 **пропущен**: HN@1 ATT1 не превысил HN@1(V6)+ε.
- Production (`untie/`, `datasets/`, Phase 1) не менялся. Деревья E0 / V2 / V3 / V3.1 / D4–D5 / **V4** / **V4.1** не перезаписывались.
- V6 **не ретюнился**. Веса = точные per-run веса V4.1 OOF.
- Gold не использовался в runtime ranking. Gold Token F1 ≥ 80 разрешён протоколом **только на TRAIN** для head discovery (reader-easy positives). Метрики считаются после выбора чанка/ответа.

**INTERNAL_TEST не открыт.** Ниже нет ни одного QA-числа с этого сплита, кроме описи: 78 locked records. U80 и K=10+attention в этот прогон не входили.

Автодамп `analysis/selection.json` объявил архитектурой **V6-K5**, `admissible=["V6-K5"]`, `att1_gate=false`. В отличие от V4 (автодамп взял in-sample V3-A3) и V4.1 (автодамп взял V6-K10 при CI через ноль vs K=5), **здесь автоселектор совпал с freeze протокола**. Это совпадение не делает автодамп научным документом; честный разбор всё равно ниже.

---

## 0. Короткий итог

На честном 5-fold OOF (EVAL only, три seed, pairing `question_id+seed`, clustered bootstrap по `document_id`, n_boot=5000, K=5) **V6-K5 остаётся reference**. Primary гипотеза ATT1 (selected-head attention как поправка к z-нормированному V6 внутри B0 top-5) **не подтверждена**. Упрощённые CORE1/CORE2 бьют B0, но не проходят non-inferiority vs V6. CORE3 даёт более высокий Token F1 и EM, но это diagnostic-only система без B0, и vs V6 она тоже не отделена.

| Что можно утверждать | Система | n | Token F1 | ΔF1 vs B0 | 95% CI vs B0 | Hit@1 | Hit@5 | HN@1 | EM | Amb@1 | GSRN@1 |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Живой ranking-reference | **B0-live** | 1359 | **27.36** | — | — | **36.72%** | **79.03%** | **4.86%** | 20.75% | 50.85% | 33.92% |
| Замороженный V4.1 OOF, веса не ретюнились | **V6-K5** | 1359 | **34.62** | **+7.26** | **[4.38, 10.17]** | **42.90%** | 79.03% | 5.45% | 25.31% | 42.90% | 26.64% |
| B0 z + λ_m Majority | CORE1 | 1359 | 32.93 | +5.54 | [3.39, 7.69] | 40.47% | 79.03% | 5.22% | 27.08% | 46.95% | 30.54% |
| B0 z + Majority + T_i | CORE2 | 1359 | 32.47 | +5.10 | [3.12, 7.07] | 40.91% | 79.03% | 5.45% | 26.56% | 46.50% | 29.80% |
| Reader-only, без B0, **diagnostic** | CORE3 | 1359 | 35.44 | +8.03 | [4.64, 11.43] | 41.21% | 79.03% | 6.18% | **32.08%** | 40.69% | 24.87% |
| V6 z + δ A_rel selected heads | ATT1 | 1359 | 34.76 | +7.40 | [4.53, 10.44] | 42.46% | 79.03% | 6.11% | 24.87% | 41.65% | 24.21% |

**Primary: ATT1 vs V6-K5**

```text
ΔF1        = +0.14
95% CI     = [−1.24, +1.56]   включает 0
win/lose/tie = 78 / 73 / 1208
HN@1       ATT1 6.11% vs V6 5.45%   Δ = +0.66 п.п.  < ε=0.02  → hn_worse=false
Hit@1      42.46% vs 42.90%  (чуть хуже V6)
att1_gate  = False
```

Формально это **weak positive** по §52 протокола (точечный ΔF1>0, CI через 0, HN safeguard пройден), но не primary success по §51. По существу это **почти нулевой результат**: 1208 ничьих из 1359, net_positive_gain V6→ATT1 = **−6**, на seed 43 среднее ΔF1 отрицательное, на двух из пяти CAL-фолдов выбрано **δ=0** (точная копия V6).

**Главное.**

1. R0 прошёл. B0-live и V6-K5 совпали с историческими V3/V4.1 до последнего знака. Identity-пара `(qid, seed)` даёт ΔF1=0, 1359 ничьих. Isolated top-5: Hit@5/Hit@10/HN@5 у всех систем этой версии **идентичны B0** (79.03% / 92.20% / 6.90%). Attention и CORE не выходят из B0 shortlist.
2. QA cache: **2850/2850** fallback из V4.1/V4, `hit_rate=1.0`, новых QA-forward не было. Attention: **2850 live compact extracts**, 102.7 с, 0.036 с/кандидат, `eager=false`, feature version `v4.2-bi-enrich-idf-v2`. Сырые тензоры `[L,H,T,T]` не сохранялись.
3. **ATT1 не бьёт V6.** Gate требует ΔF1>0 **и** CI без нуля **и** HN safeguard. Есть только первое и третье. Hit@1 даже чуть просел. Freeze правильно записал `final_architecture=V6-K5`.
4. Head-set **не стабилен**. Mean Jaccard top-N множеств по пяти outer runs = **0.071**. Только голова `5-4` встречается в 3/5 runs; `4-5` и `11-4` — в 2/5. Inner TRAIN pairwise выбранного N часто около или **ниже** шанса 0.5 (run0: 0.30). Paired TRAIN-вопросы для discovery: 30–40 на run — на грани `min_paired_questions=30`.
5. CAL δ нестабилен: `{1.0, 0.0, 1.0, 0.0, 0.5}`. Два фолда копируют V6. Identity `δ=0` vs V6: на каждом run 100% совпадение чанка и ответа.
6. **CORE1/CORE2 не non-inferior vs V6.** Правило: нижняя граница 95% CI(CORE−V6) > −0.5 F1. CORE1: −1.72, CI [−3.99, +0.63], низ −3.99. CORE2: −2.16, CI [−4.47, +0.16], низ −4.47. Оба уверенно лучше B0, оба проще V6, оба **слабее** V6 примерно на 1.7–2.2 F1. λ по фолдам так же нестабильны, как веса V6.
7. **CORE3 нельзя объявлять победителем.** Точечно Token F1 35.44 и EM 32.08% выше V6, F1|positive 72.81 vs 65.22. Но Hit@1 ниже (41.21% vs 42.90%), HN@1 выше (6.18%), vs V6 ΔF1 +0.78 с CI [−1.76, +3.30] — и ноль, и пол −0.5 внутри интервала. Протокол: diagnostic-only, без B0, не confirmatory.
8. ATT2 не считался (иерархия: improvement только если ATT1 прошёл gate). Даже forced-exploratory ATT2 не мог бы стать confirmatory winner этой версии протокола.
9. ATT3 не считался: HN ATT1 не вышел за V6+ε, хотя точечно HN@1 вырос на 0.66 п.п.
10. INTERNAL_TEST (78) locked. B0-final не обучался. U80 и K=10+attention не смешивать с этим выводом.

OOF V4.2 — model-selection evidence на unlocked пуле, не confirmatory INTERNAL_TEST.

---

## 1. Что это за эксперимент

После V4.1 живой reference — честный OOF **V6-K5** (Token F1 34.62, ΔF1 vs B0 +7.26, CI [4.38, 10.17], Hit@1 42.90%). Knockouts V4.1 показали, что прирост почти целиком из **reader-agreement**, а не из B0-score. K=10 не отделён от K=5. C1-агрегация почти ничего не добавила.

V4.2 должен был ответить на два отдельных вопроса, не ретюня V6:

1. **Attention.** Если на TRAIN выбрать QA-головы, которые устойчиво сильнее связывают вопрос с reader-easy positive span, чем с HN span, то относительный query–span attention этих голов внутри B0 top-5 может поправить V6.
2. **V6-Core.** Можно ли почти тот же результат получить более простой формулой (B0 z + majority, опционально нормализованный Token F1).

Primary fusion:

```text
S_ATT1(c_i) = Z_q(S_V6(c_i)) + δ A_rel(c_i)
```

- `Z_q` — z-нормализация только внутри top-5 данного `(question, seed)`.
- `A_rel` — relative attention-score выбранных голов внутри того же top-5.
- Head-set и N выбираются только на TRAIN.
- `δ` выбирается только на CAL, сетка `{0.0, 0.1, 0.25, 0.5, 1.0}`.
- `δ=0` — точная копия V6.

CORE:

```text
S_CORE1 = Z_q(s_B0) + λ_m Majority
S_CORE2 = Z_q(s_B0) + λ_m Majority + λ_t T_i
S_CORE3 = Majority + T_i     (без B0; diagnostic only)
T_i     = mean_token_f1_i / 100
```

Сетки записаны в `contract.json` до EVAL: `λ_m ∈ {0.25, 0.5, 1.0, 2.0}`, `λ_t ∈ {0, 0.25, 0.5, 1.0}`. HN-фильтр на CAL: HN@1 ≤ HN@1(B0 CAL) + 0.02.

Не развивались в этом прогоне: KeyBERT/LOSCO, ownership, V7, raw QA ranking, softmax-only, QA-silver, ручной TE/TM, attention-only ranker, U80, K=10+attention, retune V6.

Primary success gate ATT1 (§51):

```text
ΔToken F1 vs V6 > 0
95% CI excludes 0
HN safeguard passed
```

Дополнительно протокол просит gain более чем на одном seed и отсутствие invariant failures.

Non-inferiority CORE vs V6: нижняя граница 95% CI(CORE−V6) > −0.5 F1, плюс HN, простота и стабильность весов.

---

## 2. Вселенная, сплиты, isolation

Те же frozen V3 CV, без новых outer folds.

| Величина | Значение | Где лежит |
|---|---|---|
| Unlocked answerable вопросы | **453** | `source_hashes.json` |
| Документы | 240 | V3 `data/v3_cv_splits.json` |
| INTERNAL_TEST records | **78, locked** | `source_hashes.json` |
| Folds | 5, document-level, seed 42 | те же, что V3 |
| `dataset` hash | `3be581a8dfb015ddbc5a6fd02d411a3ab0aa7dadc4ac06c7833618135844dc9b` | совпадает с V3 / V4 / V4.1 |
| `splits` hash | `aa34bb4f4f6c31d318b7f472b166a5e3bda1ca6dd03891d6b07efef211bc0cf7` | то же |
| B0-live | исторический V3-B0 max5 OOF, не переобучался | `reference/b0_live/` |
| V6-K5 | точная копия V4.1 OOF, веса per-run | `reference/v6_k5_v4_1/` |
| B0 seeds | 42, 43, 44 | n=1359 = 453×3 |
| Unique top-5 QA calls | **2850** | меньше, чем 5176 в V4.1, потому что V4.1 считал ещё K=10 |
| QA cache | все **fallback из V4.1/V4**, hit_rate=1.0 | `analysis/efficiency.json` |
| Attention extracts | 2850 live, cache miss_rate=1.0 (ожидаемо: новый feature contract) | `attention_cache/feature_contract.json` |
| Feature version | `v4.2-bi-enrich-idf-v2` | bidirectional Q→span enrichment + span→Q density, затем shortlist z |
| `raw_tensors_saved` | false | compact features only |
| Pairing | `question_id+seed` | как V4.1 |
| Bootstrap | mean over seeds, затем cluster `document_id`, n_boot=5000 | как V4.1 |

OOF-фолды EVAL (3 seed): run0 270, run1 273, run2 270, run3 273, run4 273. Сумма 1359, уникальных вопросов 453. Каждый вопрос ровно в одном EVAL-фолде.

Писали только в `artifacts/qa_chunk_selection_v4_2/`. CLI/launcher отказывают, если `OUT_DIR` указывает на V4.1 или более раннее дерево.

V6-веса (read-only из V4.1, **не** подбирались заново):

| Run | α | β | γ |
|---:|---:|---:|---:|
| 0 | 2.0 | 0.25 | 0.5 |
| 1 | 0.0 | 0.25 | 0.0 |
| 2 | 1.0 | 0.25 | 0.0 |
| 3 | 0.25 | 0.25 | 2.0 |
| 4 | 0.5 | 0.0 | 0.25 |

---

## 3. R0 — evaluation infrastructure

Все identity-проверки **ok=true** (`regression/r0_summary.json`).

### 3.1. B0-live

Совпадение с историческим V3-B0 по каждому ключу:

```text
Token F1     27.358413858227983
EM           0.20750551876379691
Hit@1        0.3671817512877116
Hit@5        0.7902869757174393
Hit@10       0.9220014716703459
HN@1         0.04856512141280353
Amb@1        0.5084621044885945
GSRN@1       0.33922001471670343
pairwise     0.7513879233769745
n            1359
F1 | pos     61.7022636172728
```

### 3.2. V6-K5

Совпадение с историческим V4.1 OOF:

```text
Token F1     34.624867427228835
EM           0.25312729948491536
Hit@1        0.4289919058130979
Hit@5        0.7902869757174393
HN@1         0.054451802796173655
n            1359
```

Paired identity B0 vs B0: mean_delta=0, CI [0, 0], win/lose/tie = 0/0/1359.

### 3.3. Attention / ranking regressions

| Проверка | Результат |
|---|---|
| Logits identity, `eager=false`, 4 вопроса | `start_max_abs=0`, `end_max_abs=0`, одинаковый argmax start/end **и** decoded answer text |
| Span identity | ok, n=4 |
| Top-5 identity | ok: `B0 ranked_ids are never rewritten` |
| Seed-union dedup / EVAL leakage | ok, `eval_leakage_runs=[]` |
| Span-length bias test | mass растёт со span (0.73 → 2.91); density одинакова (0.091); enrichment ≈ 1 для short и long. Feature не является «длинный span = больше attention» |

`eager=false` остался runtime-режимом: logits без eager совпали, отдельный eager-pass не нужен.

---

## 4. CORE1 / CORE2 / CORE3

Отдельный вопрос протокола: упростить reader-agreement, не доказывая attention. V6 не ретюнился; CORE — новые системы на той же QA-кэш и тех же outer folds.

### 4.1. Выбранные λ (только CAL)

| Run | CORE1 λ_m | CORE1 CAL F1 | CORE1 CAL HN@1 | CORE2 (λ_m, λ_t) | CORE2 CAL F1 | CORE2 CAL HN@1 |
|---:|---:|---:|---:|---|---:|---:|
| 0 | 2.0 | 35.73 | 6.23% | (0.5, 1.0) | 36.15 | 5.86% |
| 1 | 2.0 | 38.46 | 1.11% | (2.0, 0.0) | 38.46 | 1.11% |
| 2 | 1.0 | 26.58 | 5.13% | (1.0, 1.0) | 27.49 | 5.13% |
| 3 | 2.0 | 41.09 | 5.19% | (2.0, 0.5) | 41.46 | 5.19% |
| 4 | 0.25 | 21.69 | 5.13% | (0.5, 1.0) | 23.02 | 6.59% |

Все выбранные точки admissible по HN на своём CAL. Веса **нестабильны**, тот же паттерн, что у V6 в V4.1: разные фолды покупают разную смесь. На run1 CORE2 выбрал `λ_t=0` — формула схлопнулась в CORE1. CORE3 diagnostic переиспользовал λ CORE2 того же run, но без B0-члена.

### 4.2. OOF vs B0

Все три CORE бьют B0 по Token F1 с CI без нуля. Hit@5/10 не ломаются.

| Система | Token F1 | Δ vs B0 | 95% CI | win/lose/tie vs B0 | Hit@1 | EM | F1\|pos | HN@1 | Amb@1 | GSRN@1 | MRR |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| CORE1 | 32.93 | +5.54 | [3.39, 7.69] | 166/86/1107 | 40.47% | 27.08% | 68.91 | 5.22% | 46.95% | 30.54% | 0.566 |
| CORE2 | 32.47 | +5.10 | [3.12, 7.07] | 149/73/1137 | 40.91% | 26.56% | 67.01 | 5.45% | 46.50% | 29.80% | 0.569 |
| CORE3 | 35.44 | +8.03 | [4.64, 11.43] | 308/209/842 | 41.21% | **32.08%** | **72.81** | 6.18% | 40.69% | 24.87% | 0.566 |

CORE1/CORE2 — меньше churn vs B0, чем V6 (net_positive_gain +51 / +57 против +84 у V6). Они реже забирают top-1 у ambiguous/GSRN и поэтому слабее по Hit@1. При этом EM CORE1 (27.08%) **выше** V6 (25.31%): когда чанк выбран хорошо, majority-формула чуть лучше совпадает с точным span. Это не компенсирует проигрыш в ranking.

CORE3 без B0 сильнее «выжимает» reader-сигнал: F1|positive 72.81, EM 32.08%, Amb@1 падает до 40.69%, GSRN@1 до 24.87%. Цена — больше уходов original positive (positive→easy 36, positive→HN 23 vs 18 у V6) и HN@1 6.18%. Churn высокий: win/lose vs B0 308/209 при 842 ничьих — система часто меняет ответ, не всегда в плюс.

### 4.3. Non-inferiority vs V6

Правило: CI low (CORE − V6) > −0.5.

| | ΔF1 vs V6 | 95% CI | win/lose/tie | noninferior? |
|---|---:|---|---|---|
| CORE1 | **−1.72** | **[−3.99, +0.63]** | 132/189/1038 | **нет** (low −3.99) |
| CORE2 | **−2.16** | **[−4.47, +0.16]** | 135/190/1034 | **нет** |
| CORE3 | +0.78 | [−1.76, +3.30] | 162/192/1005 | **нет** (low −1.76); к тому же diagnostic-only |

CI CORE1/CORE2 vs V6 **включает 0**: нельзя утверждать, что они статистически хуже V6 на уровне «исключили ноль», но non-inferiority floor −0.5 **не выполнен**. Точечно они хуже. Стабильности λ нет. Reference остаётся V6-K5.

`analysis/compare_summary.json` помечает CORE1/CORE2/CORE3 как `admissible=false`, `noninferior_vs_v6=false`. HN safeguard у всех трёх `true` (не вышли за B0+ε на OOF относительно допуска протокола).

### 4.4. Переходы CORE vs B0

| Переход | V6-K5 | CORE1 | CORE2 | CORE3 |
|---|---:|---:|---:|---:|
| ambiguous → positive | 174 | 105 | 97 | 205 |
| GSRN → positive | 111 | 74 | 73 | 129 |
| easy_negative → positive | 24 | 18 | 20 | 26 |
| hard_negative → positive | 8 | 7 | 8 | 12 |
| positive → ambiguous | 84 | 57 | 45 | 123 |
| positive → HN | 18 | 13 | 13 | 23 |
| positive → easy_negative | 20 | 9 | 10 | 36 |
| **net positive gain** | **+84** | **+51** | **+57** | **+61** |

CORE1/CORE2 консервативнее V6. CORE3 агрессивнее по захвату ambiguous/GSRN, но и сильнее отдаёт уже найденные positives.

### 4.5. Аспекты CORE

n: material 378, method 345, metric 195, task 441.

| Аспект | B0 F1 | V6 F1 | CORE1 F1 | CORE2 F1 | CORE3 F1 |
|---|---:|---:|---:|---:|---:|
| material | 30.94 | 34.47 | 36.71 | 35.50 | **38.84** |
| method | 18.66 | 26.36 | 26.59 | 25.85 | **34.43** |
| metric | 17.31 | 24.24 | 22.64 | 21.95 | 27.47 |
| task | 35.54 | **45.81** | 39.20 | 39.69 | 36.85 |

Без B0 CORE3 выигрывает method/material и **проигрывает task** относительно V6 (36.85 vs 45.81; Hit@1 task 30.39% vs 41.95%, HN@1 task 9.75%). Это ещё один аргумент не ставить CORE3 в production-контур: aspect-профиль ломается там, где V6 сильнее всего по F1.

---

## 5. Attention extraction

Live compact features на тех же 2850 unique `(question, chunk)` из isolated B0 top-5.

```text
n                  2850
seconds            102.70
per_candidate_s    0.0360
eager              false
raw_tensors_saved  false
version            v4.2-bi-enrich-idf-v2
primary            Q→span enrichment + span→Q density,
                   z-scored within shortlist, then mean selected heads
att2 field         IDF-weighted Q→span token mass, then |S|/|C| enrichment;
                   span→Q unweighted
                   (поле q_to_span_token_mean; ATT2 stage не запускался)
```

Пустой / invalid span → нулевые features, не фейковый `|S|=1`. Invalid-span audit на head TRAIN: **0/0/0** invalid positive/HN/easy во всех пяти runs.

QA-кэш не пересчитывался (3.6 с на 2850 fallback hits). Основное время прогона — CORE CAL (~35 мин), не attention extract.

---

## 6. Head selection

Universe: unique TRAIN union B0 top-5 across seeds, ключ `(question_id, chunk_id)`. Positives = `primary_class == "robust_positive"` **и** gold Token F1 ≥ 80 **и** valid_span. Primary negatives = training-eligible HN с valid_span. Easy-negative fallback не понадобился ни на одном run.

### 6.1. Feasibility по runs

| Run | union q | unique pairs | paired pos+HN questions | pos chunks | HN chunks | easy | invalid p/hn/easy | fallback | unreliable |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 0 | 272 | 1731 | 37 | 219 | 126 | 242 | 0/0/0 | false | false |
| 1 | 272 | 1718 | 40 | 209 | 133 | 236 | 0/0/0 | false | false |
| 2 | 272 | 1701 | 30 | 202 | 123 | 238 | 0/0/0 | false | false |
| 3 | 272 | 1671 | 32 | 218 | 106 | 237 | 0/0/0 | false | false |
| 4 | 271 | 1729 | 38 | 238 | 109 | 265 | 0/0/0 | false | false |

`min_paired_questions=30`. Run2 сел ровно на порог (30). Это маленький paired-n для 144 голов (12 слоёв × 12 голов): ranking голов шумный.

### 6.2. Inner N и фактически замороженный set

N выбирается inner TRAIN pairwise positive>negative, сетка `{1, 2, 4, 8}`. Шанс = 0.5.

| Run | tried mean inner R (N=1,2,4,8) | selected N | inner R selected | folds selected | фактические головы | overall top R |
|---:|---|---:|---:|---|---|---|
| 0 | 0.249, 0.299, 0.283, **0.303** | 8 | **0.303** | 0.35 / 0.40 / 0.16 | **только 2:** `(4,5), (5,11)` | 0.507 / 0.507 |
| 1 | 0.514, 0.501, **0.536**, 0.505 | 4 | 0.536 | 0.53 / 0.31 / 0.77 | `(6,8), (11,0), (11,4), (10,1)` | 0.681 … 0.596 |
| 2 | 0.357, 0.334, 0.458, **0.495** | 8 | 0.495 | 0.28 / 0.61 / 0.59 | 8 голов, лидер `(5,4)` | 0.633 |
| 3 | 0.432, 0.500, 0.465, **0.520** | 8 | 0.520 | 0.55 / 0.46 / 0.56 | 8 голов, лидер `(5,4)` | 0.688 |
| 4 | 0.475, **0.543**, 0.517, 0.482 | 2 | 0.543 | 0.65 / 0.33 / 0.64 | `(5,4), (4,5)` | 0.656 / 0.604 |

Важные оговорки:

1. **Inner R часто не бьёт шанс.** Run0 выбранное N=8 имеет mean inner pairwise **0.30** — хуже подбрасывания монеты. Run2 выбранное N=8: 0.495 ≈ шанс. Только run1 и run4 чуть выше 0.5.
2. **N в артефакте ≠ число голов.** На run0 direction-stable оказались только 2 головы (`R>0.5` в ≥2 inner folds), хотя inner выбрал N=8. Поле `n=8` — запрошенный размер, список `heads` длины 2. Это честно отражает нестабильность, не баг freeze: брать top-N из пустого/короткого stable-списка.
3. `delta_forced_0=false` на всех runs: feasibility не признала discovery unreliable, хотя inner R run0 это по сути говорит.

### 6.3. Стабильность между outer runs

`analysis/head_stability.json`:

```text
mean Jaccard top-N sets across 5 runs = 0.071
recurrence:
  5-4  → 3 runs
  4-5  → 2
  11-4 → 2
  все остальные → 1
```

Почти нет пересечения. Это прямое свидетельство против «существует устойчивый task-specific head set для SciREX QA». Дальнейший ATT1 CAL/EVAL работает с **разными** головами на разных фолдах — нельзя интерпретировать OOF как тест одной замороженной архитектуры голов.

---

## 7. ATT1 — CAL, EVAL, gate

### 7.1. CAL δ

Сетка 5 точек. HN-нарушающий δ не выбирается, даже если F1 выше; если admissible пуст, минимизируют HN / меньший δ (δ=0 = V6 должен быть admissible).

| Run | выбранный δ | CAL F1 | CAL HN@1 | n admissible / 5 | что отсеялось |
|---:|---:|---:|---:|---|---|
| 0 | **1.0** | 37.65 | 5.49% | 5/5 | все ok; δ=1 чуть лучше δ=0 (37.65 vs 37.23) |
| 1 | **0.0** | 38.60 | 2.59% | 4/5 | δ=1.0 HN 5.6% inadmissible; все ненулевые δ хуже F1, чем V6 |
| 2 | **1.0** | 30.39 | 5.13% | 5/5 | δ=1 едва лучше δ=0 (30.39 vs 30.23) |
| 3 | **0.0** | 39.73 | 4.81% | 2/5 | δ≥0.25 inadmissible по HN; ненулевые δ хуже F1 |
| 4 | **0.5** | 26.72 | 6.23% | 4/5 | δ=1.0 HN 8.4% inadmissible |

δ=0 выбран в **2/5** runs — не «большинство» в смысле §53, но и не стабильный ненулевой коэффициент. Identity `ATT1(δ=0)` vs V6: на каждом run `same_chunk=n`, `same_answer=n` (270 или 273). На run1 и run3 дополнительно `delta0_vs_selected` 100% — выбранный ATT1 **есть** V6.

### 7.2. OOF EVAL ATT1

```text
n              1359
Token F1       34.7616
EM             24.87%
Char F1        0.57
ROUGE-L        0.4237
Hit@1          42.46%
Hit@5          79.03%   = B0 = V6
Hit@10         92.20%   = B0 = V6
HN@1           6.11%
HN@5           6.90%    = B0 = V6
Amb@1          41.65%
GSRN@1         24.21%
MRR            0.5706
F1 | positive  66.84    (V6: 65.22; B0: 61.70)
pairwise HN    0.4214   (шкала rerank, как V6 0.4158, не шкала B0 0.7514)
```

Vs B0: ΔF1 +7.40, CI [4.53, 10.44], win/lose/tie 310/156/893. Это почти тот же выигрыш, что у V6 (+7.26, 283/142/934). ATT1 не «новый» выигрыш против B0 — это V6 плюс шум.

### 7.3. Primary comparison ATT1 − V6

```text
n_rows=1359  n_questions=453
mean_delta = +0.142
95% CI     = [−1.243, +1.561]   включает 0
win/lose/tie = 78 / 73 / 1208
pairing    = question_id+seed
bootstrap  = mean_over_seeds_then_cluster_document_id
hn_worse   = false
att1_gate  = false
```

По seed (каждый n=453):

| Seed | mean ΔF1 ATT1−V6 | win/lose/tie |
|---:|---:|---|
| 42 | **+0.273** | 28/24/401 |
| 43 | **−0.468** | 22/24/407 |
| 44 | **+0.606** | 28/25/400 |

Gain **не** на всех трёх seed. Протокол дополнительно просил «gain более чем на одном seed»: формально 42 и 44 плюс, 43 минус. Это слабый и неоднородный сигнал, согласованный с CI через ноль.

По аспектам (ATT1 − V6, mean Token F1):

| Аспект | n | ATT1 F1 | V6 F1 | mean Δ | win/lose | ATT1 Hit@1 | V6 Hit@1 | ATT1 HN@1 | V6 HN@1 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| material | 378 | 34.55 | 34.47 | +0.08 | 14/17 | 52.65% | 54.50% | 5.03% | 3.97% |
| method | 345 | 27.26 | 26.36 | +0.90 | 16/7 | 36.23% | 34.49% | 5.80% | 4.93% |
| metric | 195 | 22.87 | 24.24 | **−1.37** | 6/10 | 34.87% | 37.44% | **8.72%** | 5.64% |
| task | 441 | 46.07 | 45.81 | +0.25 | 42/39 | 41.95% | 41.95% | 6.12% | 7.03% |

Единственный аспект с заметным плюсом — method. Metric ATT1 **хуже** V6 и по F1, и по Hit@1, и по HN. Task F1 чуть выше при том же Hit@1.

### 7.4. Классификация по протоколу

| Правило | Факт | Вердикт |
|---|---|---|
| §51 ΔF1 > 0 | +0.14 | да |
| §51 CI excludes 0 | [−1.24, 1.56] | **нет** |
| §51 HN safeguard | +0.66 п.п. < 2 п.п. | да |
| §51 gain > 1 seed | 2 из 3, один отрицательный | слабо |
| §51 invariant failures | нет | да |
| **§51 primary success** | | **нет** (`att1_gate=false`) |
| §52 weak positive | ΔF1>0, CI через 0, HN ok | формально да |
| §53 δ=0 majority | 2/5 | нет |
| §53 OOF ≈ V6 | 1208 ничьих, Δ=0.14 | **да, по сути** |
| §53 HN заметно хуже | нет | нет |

Честное чтение: **weak-positive bucket протокола, substantively near-null.** Нельзя нести ATT1 на locked test. Нельзя говорить, что attention «почти сработал». Можно говорить, что выбранный механизм не отделился от V6.

### 7.5. Переходы

**ATT1 vs B0** (из `analysis/compare.json`):

| Переход | ATT1 | V6 для сравнения |
|---|---:|---:|
| ambiguous → positive | 193 | 174 |
| GSRN → positive | 116 | 111 |
| easy_negative → positive | 27 | 24 |
| hard_negative → positive | 8 | 8 |
| positive → ambiguous | 109 | 84 |
| positive → HN | 17 | 18 |
| positive → easy_negative | 24 | 20 |
| **net positive gain** | **+78** | **+84** |

ATT1 больше крутит ambiguous (и в positive, и обратно), net vs B0 чуть хуже V6.

**ATT1 vs V6** (`att1/transitions.json`):

| Переход | Число |
|---|---:|
| positive → positive | 529 |
| ambiguous → ambiguous | 506 |
| easy_negative → easy_negative | 93 |
| hard_negative → hard_negative | 63 |
| ambiguous → positive | 43 |
| GSRN → positive | 17 |
| easy_negative → positive | 5 |
| positive → ambiguous | 39 |
| positive → easy_negative | 13 |
| positive → HN | **2** |
| **net positive gain** | **−6** |

Относительно V6 ATT1 **теряет** шесть original-positive top-1. Два ухода positive→HN. Это согласуется с Hit@1 42.46% vs 42.90% и с крошечным F1-плюсом за счёт F1|positive (66.84 vs 65.22): чуть лучше извлекает span, когда чанк уже positive, чуть хуже выбирает чанк.

---

## 8. ATT2 / ATT3 — иерархический skip

```text
[att2] ATT2 skipped: ATT1 did not pass primary gate (use --force-att2 for exploratory)
[att3] ATT3 skipped: ATT1 HN did not exceed V6+eps
```

Протокол:

- ATT2-IDF тестируется как improvement **только если** ATT1 прошёл primary gate.
- Если ATT1 не прошёл, ATT2 можно посчитать exploratory (`--force-att2`), но он **не становится confirmatory winner** в этой же версии протокола.
- ATT3 (HN-контроль) особенно нужен, если ATT1/ATT2 повышают HN@1 за допуск. Здесь HN ATT1 остался внутри ε, стадия не запускалась.

IDF-поле в compact cache уже посчитано (`q_to_span_token_mean`, version `v4.2-bi-enrich-idf-v2`), но fusion ATT2 не применялся. Нет ATT2/ATT3 метрик, которые можно было бы читать как результат этой версии.

---

## 9. Efficiency

| Статья | Число |
|---|---|
| Wall-clock всего | 2953 с ≈ 49 мин 13 с |
| QA unique top-5 | 2850, hit_rate=1.0, 0 live QA forwards |
| Attention extracts | 2850, 102.7 с, 36 мс/кандидат |
| CORE CAL | доминирующая статья (~11:51:34 → ~12:28, порядка 35 мин) |
| ATT1 CAL (5 δ × 5 runs) | ~12:28 → 12:40 |
| Peak GPU / RAM | в артефактах не записаны |
| `eager` | false |

Для следующего цикла: attention extract дешёвый относительно CORE-сетки. Дорогой шаг V4.2 — не головы, а повторный CAL ranking.

---

## 10. Freeze и что из этого следует

`final/freeze_manifest.json`:

```text
final_architecture = V6-K5
v6_retuned         = false
v6_weights         = V4.1 per-run
primary_k          = 5
internal_test      = locked
att1_gate          = false
b0_final           = NOT trained
u80                = not in V4.2
k10_attention      = not in V4.2
```

Автодамп `selected=V6-K5` на этот раз **согласован** с freeze. Это не отменяет правило: научный итог читается из gate/CI/stop-правил, не из `selection.json`.

### Что можно утверждать

1. Isolated top-5 reader-centered V6-K5 по-прежнему лучший подтверждённый rerank на unlocked OOF.
2. Selected-head attention в постановке V4.2 **не доказал** добавленную стоимость поверх V6.
3. Упрощение V6 до CORE1/CORE2 **не проходит** practical non-inferiority.
4. Head discovery на union shortlist с 30–40 paired вопросами не находит стабильного множества голов; inner pairwise часто ≤ шанса.
5. Инфраструктура (R0, cache reuse, compact attention, δ=0 identity, invalid-span=0) работает.

### Что нельзя делать дальше на основании этого прогона

- Открывать INTERNAL_TEST.
- Ставить ATT1, CORE3 или autodump-winner «вместо V6».
- Объявлять ATT2/IDF подтверждённым, не считая его, или считая exploratory в той же protocol version как winner.
- Смешивать U80 / K=10+attention в вывод V4.2.
- Ретюнить V6 «заодно» на тех же EVAL-фолдах.
- Читать Jaccard 0.07 как «головы разные, но attention в среднем помогает» — OOF как раз показывает, что не помогает статистически.

### Что логично, если будет следующий цикл

Не раздувать attention fusion, пока head discovery не перестанет быть шумом. Имеет смысл либо (a) честно закрыть attention-ветку на текущей постановке как negative/near-null, либо (b) менять **постановку discovery** (больше paired n, другой positive definition, другой feature), а не δ-сетку. CORE как engineering-упрощение тоже не готов: нужно либо принять потерю ~2 F1, либо искать другую стабилизацию agreement, не выдавая CORE3 diagnostic за систему.

K=10+attention по протоколу оформляется отдельным amendment **только если** ATT1 сначала подтвердится. Здесь не подтвердился.

---

## 11. Файлы

| Путь | Содержание |
|---|---|
| `artifacts/qa_chunk_selection_v4_2/analysis/compare.json` | полная таблица, Δ vs B0, CORE vs V6, transitions vs B0 |
| `artifacts/qa_chunk_selection_v4_2/analysis/compare_summary.json` | то же, плюс `att1_vs_v6`, `selected` |
| `artifacts/qa_chunk_selection_v4_2/analysis/selection.json` | autodump: V6-K5, gate false |
| `artifacts/qa_chunk_selection_v4_2/analysis/head_stability.json` | Jaccard / recurrence |
| `artifacts/qa_chunk_selection_v4_2/analysis/aspect_metrics.json` | F1/Hit/HN/EM по аспектам |
| `artifacts/qa_chunk_selection_v4_2/analysis/efficiency.json` | QA/attn cache |
| `artifacts/qa_chunk_selection_v4_2/att1/gate.json` | primary gate |
| `artifacts/qa_chunk_selection_v4_2/att1/transitions.json` | V6 → ATT1 |
| `artifacts/qa_chunk_selection_v4_2/core/weights_by_run.json` | CORE λ |
| `artifacts/qa_chunk_selection_v4_2/final/freeze_manifest.json` | freeze |
| `artifacts/qa_chunk_selection_v4_2/attention_cache/feature_contract.json` | extract contract |
| `artifacts/qa_chunk_selection_v4_2/regression/` | R0 / logits / span-length |
| `artifacts/qa_chunk_selection_v4_2/head_selection/run{0-4}/` | feasibility, N, heads |
| `logs/qa_chunk_selection_v4_2.log` | wall-clock |
| `artifacts/qa_chunk_selection_v4_2/reports/results_v4_2.md` | автодамп, не этот документ |
