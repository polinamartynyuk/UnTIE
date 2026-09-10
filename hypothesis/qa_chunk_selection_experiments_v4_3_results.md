# Результаты эксперимента V4.3: semantic reader agreement поверх V6-K5

Документ фиксирует **фактические результаты** прогона `experiments/qa_chunk_selection/v4_3` по протоколу [`v4_3_semantic_reader_agreement_protocol.md`](v4_3_semantic_reader_agreement_protocol.md). Это не автодамп JSON (`artifacts/qa_chunk_selection_v4_3/reports/results_v4_3.md`), а полный разбор: что запускали, какие числа честные, что протокол разрешает считать подтверждённым, а что нет.

- Старт: **2026-09-09 14:02:44**, конец: **2026-09-09 17:35:46**.
- Wall-clock: **12782.5 с ≈ 3 ч 33 мин**, `auto→cuda:0`, `exit=0`.
- Артефакты: `artifacts/qa_chunk_selection_v4_3/`.
- Лог: `logs/qa_chunk_selection_v4_3.log`.
- Стадии этого прогона: `STAGES=primary` = init, r0, cache, embed, sanity, features, sem1, lexcore, semcore, complement, adapt1, semctx1, compare, freeze, report.
- **SEMCTX1 пропущен**: SEM1 не прошёл primary gate. Exploratory `--force-semctx` не включался.
- Production (`untie/`, `datasets/`, Phase 1) не менялся. Деревья E0 / V2 / V3 / V3.1 / D4–D5 / **V4** / **V4.1** / **V4.2** не перезаписывались.
- V6 **не ретюнился**. Веса = точные per-run веса V4.1 OOF.
- Semantic encoder: frozen `sentence-transformers/all-MiniLM-L6-v2`. Без SciREX fine-tuning, без поиска encoder.
- Gold не использовался в runtime ranking. Gold / source labels разрешены протоколом **только на TRAIN** для semantic sanity и на CAL/EVAL для метрик и выбора гиперпараметров.
- Attention / K=10 / U80 / hidden-state / aspect routing в этот прогон **не входили**.

**INTERNAL_TEST не открыт.** Ниже нет ни одного QA-числа с этого сплита, кроме описи: 78 locked records.

Автодамп `analysis/selection.json` объявил архитектурой **V6-K5**, `admissible=["V6-K5"]`, `sem1_gate=false`. Как в V4.2, автоселектор совпал с freeze протокола. Это совпадение не делает автодамп научным документом; честный разбор ниже.

---

## 0. Короткий итог

На честном 5-fold OOF (EVAL only, три seed, pairing `question_id+seed`, clustered bootstrap по `document_id`, n_boot=5000, K=5) **V6-K5 остаётся reference**. Primary гипотеза H1 (SEM1 = z-нормированный V6 + δ SemRel) **не подтверждена**. Reader-only LEXCORE/SEMCORE дают более высокий point Token F1, чем V6, но это **не primary-admissible** системы протокола V4.3: они diagnostic / secondary. H2 (SEMCORE > LEXCORE) точечно положительна, но CI через ноль. H3 (ADAPT1) exploratory: лучше V6 по paired ΔF1 с CI без нуля, но не confirmatory и не заменяет freeze. H4 (SEMCTX1) не запускалась.

| Что можно утверждать | Система | n | Token F1 | ΔF1 vs B0 | 95% CI vs B0 | Hit@1 | Hit@5 | HN@1 | EM | Amb@1 | GSRN@1 |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Живой ranking-reference | **B0-live** | 1359 | **27.36** | — | — | **36.72%** | **79.03%** | **4.86%** | 20.75% | 50.85% | 33.92% |
| Замороженный V4.1 OOF, веса не ретюнились | **V6-K5** | 1359 | **34.62** | **+7.26** | **[4.38, 10.17]** | **42.90%** | 79.03% | 5.45% | 25.31% | 42.90% | 26.64% |
| V6 z + δ SemTop2 (primary H1) | SEM1 | 1359 | 34.69 | +7.32 | [4.31, 10.30] | 42.09% | 79.03% | 5.96% | 25.09% | 43.56% | 27.52% |
| Majority + lexical T (честный reader-only baseline) | LEXCORE | 1359 | 36.58 | +9.18 | [5.93, 12.48] | **43.41%** | 79.03% | 5.81% | 30.83% | 39.37% | 23.69% |
| Majority + SemRel (H2) | SEMCORE | 1359 | **37.33** | +9.94 | [6.89, 13.10] | 42.53% | 79.03% | 6.55% | 30.98% | 40.69% | 25.53% |
| Hard-route SEMCORE↔V6 (H3 exploratory) | ADAPT1 | 1359 | 36.02 | +8.64 | [5.76, 11.57] | 43.19% | 79.03% | 6.18% | 27.67% | 41.65% | 26.20% |
| Исторический V4.2 CORE3, diagnostic only | CORE3-hist | 1359 | 35.44 | +8.03 | [4.64, 11.43] | 41.21% | 79.03% | 6.18% | **32.08%** | 40.69% | 24.87% |

**Primary: SEM1 vs V6-K5**

```text
ΔF1          = +0.068
95% CI       = [−0.678, +0.815]   включает 0
win/lose/tie = 34 / 30 / 1295
HN@1         SEM1 5.96% vs V6 5.45%   Δ = +0.51 п.п.  < ε=0.02  → hn_worse=false
Hit@1        42.09% vs 42.90%  (хуже V6)
sem1_gate    = False
```

Формально это **near-null / weak positive point**: точечный ΔF1>0, HN safeguard пройден, shortlist invariants ok, gain на двух из трёх seeds — но CI через ноль, Hit@1 просел, 1295 ничьих из 1359. По §21 протокола SEM1 **не подтверждён**.

**Главное.**

1. R0 прошёл. B0-live, V6-K5 и historical CORE3 совпали с ожидаемыми identity-числами. Isolated top-5: Hit@5 / Hit@10 / HN@5 у всех систем этой версии **идентичны B0** (79.03% / 92.20% / 6.90%). Никто не выходит из B0 shortlist.
2. QA cache: **2850/2850** fallback из V4.1/V4/V4.2, `hit_rate=1.0`, новых QA-forward не было. Answer embeddings MiniLM: 2255 unique texts, ~0.56 с encode на GPU; дальше disk/memory cache hit-rate ~87% на answer store.
3. **SEM1 не бьёт V6.** Gate требует ΔF1>0 **и** CI без нуля **и** HN safeguard. Есть только первое и третье. На трёх из пяти CAL-фолдов выбрано **δ=0** (точная копия V6); на двух — δ=1.0. Identity `δ=0` vs live V6: 100% совпадение чанка и ответа на каждом run. Freeze правильно записал `final_architecture=V6-K5`.
4. TRAIN semantic sanity (диагностика, не selection): pairwise pos–pos ≫ pos–HN во всех пяти outer runs (rate ≈ 0.96–0.98). Encoder «различает» easy positives и HN на TRAIN — сигнал существует, но **не переносится в primary SEM1 gain**.
5. Lexical↔semantic pair diagnostic: из 13578 пар shortlist-ответов почти всё в `low_both` (9325) или `high_both` (1654). Пар `high_semantic_low_f1` всего **235**, `high_f1_low_semantic` — **11**. Новый тип agreement есть, но тонкий.
6. **LEXCORE** честно тюнился на той же OOF-схеме, что и SEMCORE. Token F1 36.58, Hit@1 43.41% — выше V6 точечно; vs V6 ΔF1 +1.92 с CI [−0.02, +3.89] — **нижняя граница почти на нуле**. Historical CORE3 **ниже** нового LEXCORE: ΔF1(CORE3−LEXCORE) = −1.14, CI [−2.59, +0.27]. Старый CORE3 нельзя использовать как baseline для H2.
7. **SEMCORE** даёт самый высокий Token F1 в таблице (37.33) и F1|positive 72.54 (близко к CORE3 72.81). Vs LEXCORE: ΔF1 +0.77, CI [−0.90, +2.45] — **H2 не подтверждена** (CI через 0). Vs V6: ΔF1 +2.69, CI [+0.89, +4.46] — уверенно лучше V6 по F1, но Hit@1 ниже V6 (42.53% vs 42.90%), HN@1 выше (6.55%). По протоколу SEMCORE **не primary-final** система.
8. Complementarity V6↔SEMCORE почти симметрична: 103 SEMCORE-win / 108 V6-win / 475 both-win / 673 both-lose. B0-margin diagnostic **не** показывает, что слабый B0 margin делает SEMCORE полезнее (weak SEM-win rate 6.5% vs strong 8.7%). Margin **не** добавлялся в ADAPT1.
9. **ADAPT1** (exploratory): ΔF1 vs V6 +1.38, CI [+0.50, +2.30], W/L/T 69/36/1254. Использует SEMCORE в 47.8% EVAL-строк; routing confidence available в 78.0%. CAL выбрал Q70 / ALWAYS / ALWAYS / Q70 / Q50 по runs 0–4; NEVER ни разу. Identity NEVER≡V6 и ALWAYS≡SEMCORE-when-available прошли на всех runs. Это architecture-discovery, не freeze-кандидат.
10. **SEMCTX1 не считался** (иерархия H4: только если H1 прошла). `force_semctx` не включался. Context-to-context diagnostic артефакта нет.
11. INTERNAL_TEST (78) locked. B0-final не обучался. U80 / K10 / attention не смешивать с этим выводом.

OOF V4.3 — model-selection evidence на unlocked пуле, не confirmatory INTERNAL_TEST.

---

## 1. Что это за эксперимент

После V4.2:

- ATT1 near-null vs V6 (ΔF1 +0.14, CI через 0) → attention branch закрыт как primary.
- V6-K5 остаётся подтверждённым reference (Token F1 34.62, Hit@1 42.90%).
- Diagnostic CORE3 дал высокий point F1/EM без B0, но не был fair reader-only baseline.

V4.3 должен был ответить на четыре отдельных вопроса, **не ретюня V6** и не открывая attention/K10/U80:

### H1 — primary

```text
S_SEM1(c_i) = Z_q(S_V6(c_i)) + δ_s SemRel(c_i)
```

- `Z_q` — z-нормализация только внутри B0 top-5 данного `(question, seed)`.
- `SemRel` — z-нормированный SemTop2: среднее двух наибольших cosine к **другим** valid answers shortlist (self-sim исключён).
- Encoder frozen MiniLM; `<3` valid answers → `semantic_available=false`, SemRel=0, SEM1 ≡ V6.
- `δ_s ∈ {0, 0.1, 0.25, 0.5, 1.0}` выбирается только на CAL.
- Safeguard: HN@1_SEM1 ≤ HN@1_V6_CAL + 0.02 (V6 на том же CAL, те же frozen weights).
- Tie-break: меньший δ, затем меньший HN.

Success gate (§21):

```text
ΔToken F1 vs V6 > 0
95% CI excludes 0
HN safeguard passed
```

Дополнительно (recorded, не AND в gate): gain не одним seed; shortlist invariants.

### H2 — secondary independent

Fair сравнение:

```text
S_LEXCORE = λ_m Majority + λ_t T_i
S_SEMCORE = λ_m Majority + λ_s SemRel
T_i       = mean_token_f1 / 100
```

Одинаковая OOF-схема CAL→EVAL, одинаковый B0-rank tie-break. Historical CORE3 — только diagnostic reference, не LEXCORE baseline.

Сильный успех H2: SEMCORE > LEXCORE с paired ΔF1>0 и CI без нуля.

### H3 — exploratory architecture-discovery

```text
ConsensusConf = Coh(T*) − Between(T*)
```

где `T*` — лучшая coherent triple среди valid answers (**не** SemTop2 margin). Hard-route: SEMCORE если conf ≥ τ и routing available, иначе V6. SEMCORE weights из **inner TRAIN only**; τ из CAL (Q50–Q90 + NEVER=+∞ + ALWAYS=−∞). Не confirmatory.

### H4 — conditional

SEMCTX1 = z(SEM1) + δ_c SupportRel только если H1 прошла. SupportGain = cos(QA, local evidence) − cos(Query, evidence). В этом прогоне **не запускалась**.

Не развивались: KeyBERT/LOSCO, ownership, V7, raw QA ranking, attention-only, U80, K=10, retune V6, aspect routing.

---

## 2. Вселенная, сплиты, isolation

Те же frozen V3 CV, без новых outer folds.

| Величина | Значение | Где лежит |
|---|---|---|
| Unlocked answerable вопросы | **453** | `source_hashes.json` |
| INTERNAL_TEST locked | **78** | то же |
| Документы | 240 | V3 `data/v3_cv_splits.json` |
| Outer runs | 5 | V3 splits |
| Seeds | 42, 43, 44 | config |
| OOF rows | **1359** = 453×3 | все metrics.json |
| Primary K | 5 | contract |
| Dataset hash | `3be581a8…` | `source_hashes.json` |
| Splits hash | `aa34bb4f…` | то же |

Isolation:

- Write root только `artifacts/qa_chunk_selection_v4_3/`.
- Read-only: E0, V2, V3, V3.1 prep/auto/rank, D45, V4, V4.1, V4.2.
- `regression/top5_identity.json`: ok.
- `regression/inner_train_no_eval_leakage.json`: ok, `eval_leakage_runs=[]`.
- ADAPT inner SEMCORE: `eval_leakage=[]` на всех пяти runs.
- Gold не в ranking features / not in SemRel / SupportGain.

---

## 3. R0 identity

`regression/r0_summary.json`: **ok=true**.

| Identity | Результат |
|---|---|
| B0-live vs historical V3 B0 | metric-close по Token F1 / EM / Hit / HN / Amb / GSRN |
| V6-K5 vs V4.1 OOF | то же |
| CORE3-historical vs V4.2 CORE3 OOF | то же |
| Paired identity (qid, seed) | ΔF1=0, tie=1359 |

Живые числа R0 (и весь прогон) совпадают с замороженными:

| Система | Token F1 | Hit@1 | HN@1 | EM |
|---|---:|---:|---:|---:|
| B0-live | 27.3584 | 36.72% | 4.86% | 20.75% |
| V6-K5 | 34.6249 | 42.90% | 5.45% | 25.31% |
| CORE3-historical | 35.4437 | 41.21% | 6.18% | 32.08% |

Hit@5 / Hit@10 / HN@5 = **0.790287 / 0.922001 / 0.069021** у B0, V6, CORE3, SEM1, LEXCORE, SEMCORE, ADAPT1 — shortlist isolation держится.

---

## 4. Encoder, cache, semantic availability

| Артефакт | Значение |
|---|---|
| QA needed | 2850 (453×3 seeds × top-5, с учётом union/shortlist) |
| QA fallback hits | **2850**, misses 0, hit_rate **1.0** |
| Encoder | MiniLM-L6-v2, frozen, `cuda:0` |
| Unique answer texts encoded | 2255 |
| Encode wall | 0.564 с (~0.25 мс/текст) |
| Answer store later hits | 15342 / misses 2255 → hit_rate 0.872 |
| Semantic available shortlists | **1359 / 1359 = 100%** |
| Normalization | NFC + whitespace collapse (`v4.3-nfc-ws-v1`) |

Query / local_evidence / query_answer stores в efficiency пустые: SEMCTX1 не бежал, эти embeddings не извлекались в primary path.

---

## 5. TRAIN-only semantic sanity

Диагностика «encoder вообще что-то различает на SciREX-подобных easy positives vs HN». Gold/source labels только на TRAIN. Не участвует в выборе модели.

| Outer run | n_train | n_union pairs | n_q with ≥2 easy pos | mean cos pos–pos | mean cos pos–HN | P(pospos > posHN) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 272 | 1731 | 48 | 0.943 | 0.289 | **0.976** |
| 1 | 272 | 1718 | 52 | 0.953 | 0.255 | **0.970** |
| 2 | 272 | 1701 | 56 | 0.948 | 0.271 | **0.966** |
| 3 | 272 | 1671 | 54 | 0.923 | 0.256 | **0.958** |
| 4 | 271 | 1729 | 54 | 0.950 | 0.262 | **0.982** |

`eval_leakage_runs=[]`. Вывод: frozen MiniLM на TRAIN **сепарирует** easy positives от HN. Это необходимое, но не достаточное условие для H1.

---

## 6. Lexical vs semantic pair diagnostic

`analysis/lexical_vs_semantic.json`, n_pairs=13578 (все valid answer pairs внутри shortlists × seeds):

| Bucket | n | Интерпретация |
|---|---:|---|
| low_both | 9325 | обе меры низкие — типичный shortlist шум |
| high_both | 1654 | lexical и semantic согласны |
| high_semantic_low_f1 | **235** | semantic «видит» сходство без lexical overlap |
| high_f1_low_semantic | **11** | lexical overlap без semantic cosine |

Новый сигнал есть (235 пар), но относительно редкий. Это согласуется с near-null SEM1: SemRel редко меняет argmax относительно уже сильного V6.

---

## 7. Phase A — SEM1 (H1 primary)

### 7.1. CAL выбранный δ

HN reference = **live V6** на CAL того же outer run (не EVAL-only historical OOF).

| Run | δ_s | cal Token F1 | cal HN@1 | V6 CAL HN@1 | admissible | δ=0 identity vs live V6 |
|---:|---:|---:|---:|---:|---|---|
| 0 | **0.0** | 37.23 | 0.0733 | 0.0733 | 5/5 | ok, 543/543 chunk+answer |
| 1 | **0.0** | 38.60 | 0.0259 | 0.0259 | 5/5 | ok, 543/543 |
| 2 | **0.0** | 30.23 | 0.0403 | 0.0403 | 5/5 | ok, 543/543 |
| 3 | **1.0** | 41.09 | 0.0481 | 0.0481 | 5/5 | ok, 543/543 (δ=0 ≡ V6) |
| 4 | **1.0** | 29.29 | 0.0696 | 0.0513 | 5/5 | ok, 546/546 |

Три фолда копируют V6. Два берут максимальный δ=1.0. Нестабильность δ — тот же паттерн, что у ATT1 V4.2.

На EVAL: `fallback_v6` / `delta_s=0` в **813/1359** строк; `delta_s=1.0` в **546/1359**. Больше половины OOF — буквально V6.

### 7.2. OOF SEM1

| Метрика | SEM1 | V6-K5 | Δ |
|---|---:|---:|---|
| Token F1 | 34.6923 | 34.6249 | **+0.0674** (paired mean +0.068) |
| EM | 25.09% | 25.31% | −0.22 п.п. |
| Hit@1 | 42.09% | 42.90% | **−0.81 п.п.** |
| HN@1 | 5.96% | 5.45% | +0.51 п.п. (< ε) |
| Amb@1 | 43.56% | 42.90% | +0.66 п.п. |
| GSRN@1 | 27.52% | 26.64% | +0.88 п.п. |
| F1 \| selected positive | 66.38 | 65.22 | +1.16 |

Paired bootstrap vs V6:

```text
mean_delta = +0.068
ci95       = [−0.678, +0.815]
win/lose/tie = 34 / 30 / 1295
```

Seed mean ΔF1: seed42 **+0.281**, seed43 **+0.048**, seed44 **−0.127**. Gain не одним seed (`gain_not_one_seed=true`), но один seed отрицательный. `n_bad_shortlist=0`.

### 7.3. Transitions SEM1 (vs собственный label flip внутри SEM1 rows — `sem1/transitions.json`)

Net positive gain **−11**. Крупные массы: positive→positive 551, ambiguous→ambiguous 558. Относительно V6 в `compare.transitions.SEM1` (vs B0) net_positive_gain = +73, у V6 vs B0 = +84 — SEM1 **хуже** V6 по net positive относительно B0.

### 7.4. Error buckets V6 ↔ SEM1

| Bucket | n в файле | Заметка |
|---|---:|---|
| v6_wrong → sem1_right | **21** | часто почти-синонимичные ответы (высокий token/semantic sim) |
| v6_right → sem1_wrong | **32** | больше, чем wins |

Aspect skew wins: material 11 / task 7 / method 2 / metric 1. Losses: material 16 / task 7 / metric 6 / method 3.

Примеры wins (укорочено): handwriting sequence generation → более полный character-level+handwriting; CoNLL-2005/2012 shared tasks → CoNLL 2005 and 2012 data. Losses: optical flow estimation → accurate flow estimation; CIFAR-10 and ImageNet → ImageNet.

### 7.5. Вердикт H1

**Не подтверждена.** `sem1_gate=false`. SEMCTX1 по иерархии не запускается. Freeze = V6-K5.

---

## 8. Phase B — LEXCORE (честный reader-only baseline)

### 8.1. CAL веса

Tie-break: меньший `λ_m`, затем меньший `λ_t`.

| Run | λ_m | λ_t | cal Token F1 | cal HN@1 |
|---:|---:|---:|---:|---:|
| 0 | 0.25 | **2.0** | 37.38 | 0.0696 |
| 1 | 0.25 | **2.0** | 40.21 | 0.0222 |
| 2 | 0.25 | **2.0** | 32.37 | 0.0806 |
| 3 | 0.25 | **1.0** | 42.25 | 0.0481 |
| 4 | 0.25 | **0.5** | 33.61 | 0.0769 |

`λ_m` стабилен (всегда 0.25). `λ_t` прыгает 0.5–2.0 — lexical agreement weight нестабилен по фолдам.

### 8.2. OOF LEXCORE

| Метрика | LEXCORE | V6 | CORE3-hist |
|---|---:|---:|---:|
| Token F1 | **36.58** | 34.62 | 35.44 |
| EM | 30.83% | 25.31% | **32.08%** |
| Hit@1 | **43.41%** | 42.90% | 41.21% |
| HN@1 | 5.81% | 5.45% | 6.18% |
| Amb@1 | **39.37%** | 42.90% | 40.69% |
| GSRN@1 | **23.69%** | 26.64% | 24.87% |
| F1\|pos | 70.42 | 65.22 | **72.81** |

Vs V6: ΔF1 **+1.918**, CI **[−0.015, +3.888]**, W/L/T 126/114/1119. Точечно лучше, но CI **касается нуля** снизу.

Vs historical CORE3: paired ΔF1(CORE3−LEXCORE) = **−1.141**, CI [−2.586, +0.272], W/L/T 36/80/1243. Новый fair LEXCORE **точечно сильнее** старого CORE3; CI допускает равенство. Вывод протокола подтверждён: CORE3 нельзя брать baseline для SEMCORE.

Tie-rate LEXCORE: **556/1359 = 40.9%** — часто несколько кандидатов с одинаковым majority+token score; побеждает B0 rank.

### 8.3. Aspect (Token F1)

| Aspect | V6 | LEXCORE | Δ |
|---|---:|---:|---:|
| material | 34.47 | **37.69** | +3.22 |
| method | 26.36 | **33.23** | +6.87 |
| metric | 24.24 | 24.99 | +0.75 |
| task | **45.81** | 43.39 | −2.42 |

LEXCORE повторяет паттерн CORE3: сильный method/material, **регрессия на task**. Hit@1 method 44.93% vs V6 34.49%; task Hit@1 37.41% vs V6 41.95%.

---

## 9. Phase C — SEMCORE (H2)

### 9.1. CAL веса

| Run | λ_m | λ_s | cal Token F1 | cal HN@1 |
|---:|---:|---:|---:|---:|
| 0 | 0.5 | 0.25 | 40.04 | 0.0806 |
| 1 | 0.5 | 0.25 | 38.46 | 0.0333 |
| 2 | **0.25** | 0.25 | 30.92 | 0.0916 |
| 3 | 0.5 | 0.25 | 43.38 | 0.0333 |
| 4 | 0.5 | 0.25 | 35.35 | 0.0952 |

`λ_s` стабилен (0.25). `λ_m` почти всегда 0.5 (один фолд 0.25). Стабильнее, чем LEXCORE `λ_t`.

### 9.2. OOF SEMCORE

| Метрика | SEMCORE | LEXCORE | V6 |
|---|---:|---:|---:|
| Token F1 | **37.33** | 36.58 | 34.62 |
| EM | 30.98% | 30.83% | 25.31% |
| Hit@1 | 42.53% | **43.41%** | **42.90%** |
| HN@1 | 6.55% | 5.81% | 5.45% |
| Amb@1 | 40.69% | 39.37% | 42.90% |
| GSRN@1 | 25.53% | 23.69% | 26.64% |
| F1\|pos | **72.54** | 70.42 | 65.22 |

**H2 primary comparison SEMCORE vs LEXCORE:**

```text
ΔF1        = +0.768
95% CI     = [−0.900, +2.451]   включает 0
win/lose/tie = 114 / 73 / 1172
```

Точечно SEMCORE лучше, но **не отделён** статистически. Сильный критерий успеха H2 (§28) **не выполнен**.

Vs V6: ΔF1 **+2.687**, CI **[+0.892, +4.457]**, W/L/T 165/110/1084 — F1-gain vs V6 реален. Но Hit@1 не лучше V6, HN@1 выше на ~1.0 п.п. SEMCORE **не admissible** как primary-final (`PRIMARY_FINAL_SYSTEMS = B0-live, V6-K5, SEM1`).

Tie-rate SEMCORE: **502/1359 = 36.9%** (ниже LEXCORE).

### 9.3. Aspect

| Aspect | V6 | LEXCORE | SEMCORE |
|---|---:|---:|---:|
| material F1 | 34.47 | 37.69 | **37.91** |
| method F1 | 26.36 | **33.23** | 30.91 |
| metric F1 | 24.24 | **24.99** | 23.51 |
| task F1 | 45.81 | 43.39 | **47.98** |

SEMCORE **чинит task-регрессию** LEXCORE (47.98 vs 43.39; даже выше V6 45.81), но **теряет часть method/metric** относительно LEXCORE. Hit@1 material у SEMCORE хуже (50.79% vs V6/LEX 54.50%).

### 9.4. Error buckets LEX ↔ SEM

Файловые списки capped ≈50:

| Bucket | n | Aspect skew (в файле) |
|---|---:|---|
| lex_wrong → sem_right | 50 | task 23, material 14, method 9, metric 4 |
| lex_right → sem_wrong | 50 | material 25, method 13, task 8, metric 4 |

Типичный win SEM: LEX берёт слишком общее (`limited`, длинный problem statement), SEM — конкретный dataset/short label. Типичный loss: LEX держит полный multi-dataset string, SEM сужает до одного имени (`Market-1501, CUHK03, Duke…` → `CUHK03`; `MS COCO` → `ImageNet`).

### 9.5. Вердикт H2

**Не подтверждена** как статистически отделённый semantic > lexical. Есть полезный exploratory сигнал (F1 vs V6, task repair), но не success по протоколу.

---

## 10. Complementarity V6 ↔ SEMCORE

`analysis/core_complementarity.json`, n=1359:

| Bucket | n | Доля |
|---|---:|---:|
| both win | 475 | 35.0% |
| both lose | 673 | 49.5% |
| SEMCORE win / V6 lose | **103** | 7.6% |
| V6 win / SEMCORE lose | **108** | 7.9% |

Почти симметрия complementary errors → адаптивный роутер имеет смысл **в принципе**, но margin между режимами узкий.

### B0 margin diagnostic (не в ADAPT)

| | Значение |
|---|---|
| median B0 margin | 0.0502 |
| mean margin V6-win/SEM-lose | 0.117 |
| mean margin SEM-win/V6-lose | 0.083 |
| weak (≤median) SEM-win rate | 6.47% |
| strong (>median) SEM-win rate | 8.69% |
| weak V6-win rate | 7.35% |
| strong V6-win rate | 8.54% |

Слабый B0 margin **не** ассоциирован с большей пользой SEMCORE. Гипотеза «когда B0 неуверен, бери reader consensus» **не поддержана** этим post-hoc. Правильно, что margin не вошёл в ADAPT1.

---

## 11. Phase D — ADAPT1 (H3 exploratory)

### 11.1. Inner-TRAIN SEMCORE weights

На всех runs: `λ_m=0.5`, `λ_s=0.25`, `eval_leakage=[]`, `ok=true`. Mean inner F1 ≈ 36.3–39.6. Совпадает с типичным outer SEMCORE CAL выбором.

### 11.2. CAL τ

| Run | selected τ | tau | cal F1 | cal HN@1 | V6 CAL HN | ALWAYS admissible? |
|---:|---|---:|---:|---:|---:|---|
| 0 | **Q70** | 0.4115 | 37.98 | 0.0769 | 0.0733 | (не выбран) |
| 1 | **Q70** | 0.3831 | 40.25 | 0.0296 | 0.0259 | |
| 2 | **ALWAYS** | −∞ | 32.10 | 0.0549 | 0.0403 | yes |
| 3 | **ALWAYS** | −∞ | 43.48 | 0.0370 | 0.0481 | yes |
| 4 | **Q50** | 0.3158 | 29.28 | 0.0623 | 0.0513 | ALWAYS был inadmissible (HN) |

NEVER ни разу не выбран на OOF (`n_never=0`). Identity checks:

- NEVER ≡ V6: ok на всех runs (n≈270–273).
- ALWAYS ≡ SEMCORE when routing available else V6: ok; на EVAL-run примерно 204–217 SEMCORE / 53–69 V6 fallback.

### 11.3. OOF ADAPT1

| Метрика | ADAPT1 | V6 | SEMCORE |
|---|---:|---:|---:|
| Token F1 | 36.02 | 34.62 | **37.33** |
| Hit@1 | **43.19%** | 42.90% | 42.53% |
| HN@1 | 6.18% | 5.45% | 6.55% |
| EM | 27.67% | 25.31% | 30.98% |
| F1\|pos | 67.55 | 65.22 | 72.54 |

Vs V6: ΔF1 **+1.379**, CI **[+0.496, +2.296]**, W/L/T **69/36/1254**. CI **без нуля**.

Использование на EVAL:

```text
adapt_used_semcore     = 649 / 1359 = 47.8%
routing_confidence_ok  = 1060 / 1359 = 78.0%
tau mix (rows): Q70=543, ALWAYS=543, Q50=273
```

ADAPT1 сидит **между** V6 и SEMCORE по F1: лучше V6 статистически, хуже чистого SEMCORE точечно, Hit@1 чуть выше обоих. По протоколу: **exploratory architecture-discovery, not confirmatory**, не входит в `admissible` / freeze.

Aspect ADAPT1: material 36.05, method 27.96, metric **26.14** (лучший среди новых систем), task 46.65 — компромисс без крайностей LEX/SEM.

### 11.4. Вердикт H3

Полезный exploratory сигнал (paired CI vs V6 excludes 0; NEVER/ALWAYS identities ok; no TRAIN↔EVAL leakage). **Не** основание менять freeze. Нельзя объявлять ADAPT1 новой primary architecture без отдельного confirmatory протокола и, ideally, INTERNAL_TEST.

---

## 12. Phase E — SEMCTX1 (H4)

**Не запускалась.**

Лог: `SEMCTX1 skipped: SEM1 did not pass primary gate (use --force-semctx for exploratory)`.

Каталог `semctx1/` отсутствует. `semctx1_token_f1=null`. `analysis/context_to_context.json` отсутствует (diagnostic extract жил внутри stage_semctx1).

Иерархия протокола соблюдена.

---

## 13. Сводная OOF-таблица и admissibility

Из `analysis/compare.json`:

| System | Token F1 | ΔF1 vs B0 | 95% CI vs B0 | Hit@1 | HN@1 | EM | Amb@1 | GSRN@1 | admissible |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| B0-live | 27.36 | — | — | 36.72% | 4.86% | 20.75% | 50.85% | 33.92% | true |
| **V6-K5** | **34.62** | **+7.26** | **[4.38, 10.17]** | **42.90%** | 5.45% | 25.31% | 42.90% | 26.64% | **true** |
| CORE3-hist | 35.44 | +8.03 | [4.64, 11.43] | 41.21% | 6.18% | 32.08% | 40.69% | 24.87% | false |
| SEM1 | 34.69 | +7.32 | [4.31, 10.30] | 42.09% | 5.96% | 25.09% | 43.56% | 27.52% | false |
| LEXCORE | 36.58 | +9.18 | [5.93, 12.48] | 43.41% | 5.81% | 30.83% | 39.37% | 23.69% | false |
| SEMCORE | 37.33 | +9.94 | [6.89, 13.10] | 42.53% | 6.55% | 30.98% | 40.69% | 25.53% | false |
| ADAPT1 | 36.02 | +8.64 | [5.76, 11.57] | 43.19% | 6.18% | 27.67% | 41.65% | 26.20% | false |

Ключевые парные сравнения (не vs B0):

| Сравнение | ΔF1 | 95% CI | W/L/T | Вердикт |
|---|---:|---|---|---|
| SEM1 − V6 | +0.068 | [−0.678, +0.815] | 34/30/1295 | H1 fail |
| LEXCORE − V6 | +1.918 | [−0.015, +3.888] | 126/114/1119 | borderline |
| SEMCORE − V6 | +2.687 | [+0.892, +4.457] | 165/110/1084 | F1 win, not primary |
| SEMCORE − LEXCORE | +0.768 | [−0.900, +2.451] | 114/73/1172 | H2 fail |
| CORE3 − LEXCORE | −1.141 | [−2.586, +0.272] | 36/80/1243 | CORE3 ≤ LEXCORE |
| ADAPT1 − V6 | +1.379 | [+0.496, +2.296] | 69/36/1254 | exploratory win |

`IN_SAMPLE_CONTAMINATED: false`. Selected OOF: **V6-K5**.

---

## 14. Freeze

`final/freeze_manifest.json` / `final/model_selection.json`:

```text
final_architecture = V6-K5
v6_retuned         = false
v6_weights         = V4.1 per-run
primary_k          = 5
internal_test      = locked
sem1_gate          = false
encoder            = MiniLM-L6-v2 frozen
b0_final           = NOT trained
u80 / k10 / attention = not in V4.3
adapt1             = exploratory architecture-discovery; not confirmatory
```

Что freeze **разрешает** считать подтверждённым reference после V4.3:

- V6-K5 на K=5, три seed, честный OOF;
- B0 shortlist identity;
- semantic encoder sanity на TRAIN (диагностика);
- fair LEXCORE как новый diagnostic reader-only baseline (сильнее historical CORE3 точечно).

Что freeze **запрещает** объявлять победителем:

- SEM1 / SemRel-augmented V6;
- SEMCORE / LEXCORE как primary architecture;
- ADAPT1 как confirmatory system;
- SEMCTX1 (не считался);
- любые выводы с INTERNAL_TEST;
- смешение с U80 / K10 / attention.

---

## 15. Регрессии и инварианты

| Проверка | Статус |
|---|---|
| B0 top-5 identity | ok |
| R0 B0 / V6 / CORE3 identity | ok |
| δ=0 ≡ live V6 (chunk+answer) | ok, все runs |
| SEM1 shortlist_ok | 0 bad |
| Inner TRAIN no EVAL/CAL leakage | ok |
| ADAPT NEVER ≡ V6 | ok |
| ADAPT ALWAYS ≡ SEMCORE if available else V6 | ok |
| Semantic available rate | 100% |
| Self-sim excluded / z-var→0 (unit-tested) | covered in suite |
| Prior trees not overwritten | yes |
| INTERNAL_TEST locked | yes |

HF Hub unauthenticated warning в логе — инфраструктурный, не на метрики.

---

## 16. Что это значит по существу

1. **Семантическая согласованность predicted answers** как добавка к уже сильному V6 внутри top-5 — почти нулевой эффект. V6 уже использует lexical reader agreement (exact/token/majority/support). SemTop2 редко меняет решение; когда меняет, wins ≈ losses (34 vs 30), Hit@1 даже падает.

2. **Encoder не «сломан».** TRAIN sanity 96–98% и 235 high-semantic/low-F1 пар показывают, что MiniLM несёт сигнал. Сигнал **не ортогонален** тому, что V6 уже выжал из reader, в масштабе, достаточном для primary gate.

3. **Reader-only без B0** (LEXCORE/SEMCORE) реально поднимает Token F1 и EM относительно V6, особенно method/material, ценой task (LEX) или Hit@1/HN (SEM). Это укрепляет диагноз V4.1/V4.2: большой кусок качества — из agreement answers, не из B0 score. Но без primary gate и без locked INTERNAL_TEST это **не** новая подтверждённая архитектура.

4. **Fair LEXCORE** закрывает методологическую дыру CORE3: при одинаковом тюнинге lexical reader-only уже ~36.6 F1; semantic majority+SemRel добавляет ~0.8 F1 без значимости. Historical CORE3 (35.44) не завышал и не занижал картину критично, но был **слабее** нового LEXCORE.

5. **ADAPT** показывает, что hard-routing по consensus conf может статистически улучшить V6 (+1.38 F1, CI без 0), не догоняя чистый SEMCORE. Это кандидат на отдельный confirmatory цикл, не результат V4.3 freeze.

6. **SEMCTX / local evidence** остаётся нетронутой гипотезой: H4 conditional на H1, H1 fail.

---

## 17. Честные ограничения

- Только unlocked 453; INTERNAL_TEST=78 закрыт.
- n_boot=5000 clustered по document; CI широкие на малых эффекте (SEM1, H2).
- LEXCORE/SEMCORE имеют высокий tie-rate → сильная зависимость от B0-rank tie-break.
- SEMCORE/ADAPT HN@1 выше V6; протокольный HN safeguard на CAL ≠ «HN не вырос на EVAL».
- Aspect n невелик (metric 195 rows): aspect-выводы exploratory.
- Error-bucket списки capped (~50); counts wins/losses по V6↔SEM1 полные (21/32), LEX↔SEM — усечённые примеры.
- Один frozen encoder; поиск encoder запрещён протоколом.
- SEMCTX1 не измерен даже exploratory.

---

## 18. Что дальше (не часть этого freeze)

Имеет смысл только как **отдельные** протоколы, не переписывая V4.3:

1. Не открывать SEM1/SEMCTX как primary без новой feature (не тот же SemTop2).
2. Если продолжать reader-only: confirmatory сравнение LEXCORE vs SEMCORE vs V6 с заранее зафиксированным primary endpoint и, при успехе, INTERNAL_TEST.
3. ADAPT1-подобная схема — отдельный architecture protocol (новые τ features? calibrated soft mix?), не «дотюнить V4.3».
4. Не смешивать U80 / K10 / attention в те же таблицы.
5. Не ретюнить V6 «под SemRel».

---

## 19. Файловый указатель

| Путь | Содержание |
|---|---|
| `artifacts/qa_chunk_selection_v4_3/final/freeze_manifest.json` | freeze |
| `artifacts/qa_chunk_selection_v4_3/final/model_selection.json` | selected=V6-K5 |
| `artifacts/qa_chunk_selection_v4_3/analysis/compare.json` | полная OOF таблица + deltas |
| `artifacts/qa_chunk_selection_v4_3/sem1/gate.json` | H1 gate |
| `artifacts/qa_chunk_selection_v4_3/lexcore/` / `semcore/` / `adapt1/` | OOF + CAL |
| `artifacts/qa_chunk_selection_v4_3/analysis/core_complementarity.json` | V6↔SEMCORE buckets |
| `artifacts/qa_chunk_selection_v4_3/analysis/b0_margin_diagnostic.json` | margin post-hoc |
| `artifacts/qa_chunk_selection_v4_3/analysis/error_buckets.json` | flip examples |
| `artifacts/qa_chunk_selection_v4_3/semantic_sanity/` | TRAIN encoder check |
| `artifacts/qa_chunk_selection_v4_3/regression/` | identities / leakage |
| `artifacts/qa_chunk_selection_v4_3/reports/results_v4_3.md` | автодамп, **не** этот документ |
| `logs/qa_chunk_selection_v4_3.log` | полный лог |
| `hypothesis/v4_3_semantic_reader_agreement_protocol.md` | протокол |

---

## 20. Одна фраза

**V4.3 не сдвинул confirmed reference: V6-K5 остаётся; SemRel-добавка к V6 near-null; fair reader-only и adaptive routing интересны как diagnostics/exploratory, но не проходят primary success gates протокола.**
