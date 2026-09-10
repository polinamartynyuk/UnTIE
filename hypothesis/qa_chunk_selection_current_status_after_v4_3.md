# Текущее состояние: эксперименты по отбору чанков (handoff для агента)

Документ для передачи работы **другому ИИ-агенту**. Дата фиксации: **2026-09-10**.

Это **не** протокол эксперимента и **не** научный отчёт о числах. Это операционная карта: где мы, что подтверждено, какие принципы нельзя нарушать, куда смотреть в коде и артефактах, что устарело.

Числа результатов и gates — в человеческих отчётах (см. §12). Здесь числа только там, где они нужны как **замороженный reference** или как якорь «не перепутать версии».

---

## 0. Одна строка

```text
Confirmed live reference = V6-K5 (Token F1 34.62, Hit@1 42.90%, K=5, 3 seeds).
V4.3 done: SEM1 gate failed; freeze stays V6-K5. LEXCORE/SEMCORE/ADAPT1 = diagnostic/exploratory only.
INTERNAL_TEST locked (78). Do not retune V6. Do not overwrite E0…V4.3 trees.
Do not trust hypothesis/qa_chunk_selection_current_status.md (snapshot 2026-09-04, stops at V3.1 prep).
Next work = new isolated package + new artifact root + new protocol, unless user says otherwise.
```

---

## 1. Жёсткие принципы (нарушать нельзя)

Эти правила повторяются в контрактах V4.x и протоколах. Они **важнее** желания «улучшить метрику».

1. **Production не трогать.** Не менять `untie/`, `datasets/`, Phase 1 artifact trees. Эксперименты живут только в `experiments/qa_chunk_selection/` и `artifacts/qa_chunk_selection*/`.
2. **Не перезаписывать прошлые артефакты.** Каждый major цикл пишет в **свой** root. Уже закрыты для записи:  
   `artifacts/qa_chunk_selection` (E0), `_v2`, `_v3`, `_v3_1_prep`, `_v3_1_auto`, `_v3_1_auto_v2`, `_v3_d45`, `_v4`, `_v4_1`, `_v4_2`, и после завершения прогона — **`_v4_3`** (повторный запуск в тот же OUT_DIR откажется без `ALLOW_EXISTING_OUT=1`).
3. **Gold не в runtime ranking.** Gold / `source_label` не участвуют в score features на inference. Gold разрешён для: CAL selection metrics, EVAL metrics, TRAIN-only sanity/diagnostics (явно помеченных).
4. **`source_label` не переписывать.** Overlay `target_status` (V3.1) — отдельный слой; Phase 1 labels остаются.
5. **INTERNAL_TEST закрыт** (78 records). Нет QA-чисел и нет model selection с этого сплита, пока отдельный протокол явно не откроет его. Флаг CLI `--allow-internal-test` в V4.3 **принят, но игнорируется**.
6. **V6 не ретюнить** в продолжениях после V4.1, пока новый протокол явно не скажет иное. Веса = per-run OOF из V4.1 (`artifacts/qa_chunk_selection_v4_1/r1_v6_oof/weights_by_run.json`).
7. **Shortlist isolation.** Системы V4+ переупорядочивают только B0 top-K. Hit@5/Hit@10/HN@5 должны совпадать с B0, если isolation соблюдена (в V4.3: 79.03% / 92.20% / 6.90%).
8. **EVAL не для тюнинга.** Гиперпараметры (δ, λ, τ) — только CAL (или inner-TRAIN для ADAPT weights). EVAL только OOF merge.
9. **Автодамп ≠ научный отчёт.** `artifacts/.../reports/results_v*.md` — generated. Человеческий разбор — `hypothesis/qa_chunk_selection_experiments_v*_results.md`.
10. **Не смешивать ветки в один вывод.** U80, K=10+attention, aspect routing, encoder search — если не в протоколе текущего цикла, не тащить в freeze-таблицу.
11. **Новый цикл = новый пакет + новый artifact root + новый protocol md**, а не правка старого пайплайна «на месте» с перезаписью результатов.
12. **Пользователь пишет по-русски** в чате; отчёты/handoff тоже обычно на русском. Код и идентификаторы систем — на английском (`V6-K5`, `SEM1`, …).

---

## 2. Где мы сейчас (после V4.3)

| Вопрос | Ответ |
|---|---|
| Подтверждённый live reference | **V6-K5** |
| Token F1 (OOF unlocked, 1359 = 453×3) | **34.6249** (contract `HISTORICAL_V6_K5`) |
| Hit@1 / HN@1 / Amb@1 / GSRN@1 | 42.90% / 5.45% / 42.90% / 26.64% |
| B0-live Token F1 | **27.3584** |
| Primary K | **5** |
| Seeds | 42, 43, 44 |
| V6 retuned after V4.1? | **Нет** |
| V4.2 ATT1 | gate **false** (near-null vs V6) |
| V4.3 SEM1 | gate **false** (ΔF1 +0.068, CI through 0) |
| V4.3 freeze | `final_architecture=V6-K5`, `sem1_gate=false` |
| SEMCTX1 | **не запускался** (H4 conditional on H1) |
| LEXCORE / SEMCORE / ADAPT1 | посчитаны; **не** primary-admissible; не freeze |
| INTERNAL_TEST | **locked** |
| Следующий эксперимент | **не задан** пользователем; не выдумывать «V4.4» без протокола |

Нюанс V4.1 freeze file: строка `final_architecture` там **`V6-K10`**, но `final_k=5`, а human report / последующие циклы держат **K=5** как reference (V6-K10 vs K5 CI через 0). **Актуальный confirmed reference после V4.2/V4.3 — V6-K5**, не «переходить на K10».

---

## 3. Линейка экспериментов (не путать пакеты)

| Цикл | Протокол / план | Человеческий отчёт | Код | Артефакты | Состояние |
|---|---|---|---|---|---|
| E0 | `hypothesis/qa_chunk_selection_experiments_v1.md` | числа в плане / автодамп | `experiments/qa_chunk_selection/` (корень) | `artifacts/qa_chunk_selection/` | закрыт |
| V2 | `…_v2.md` | `…_v2_results.md` | `…/v2/` | `…_v2/` | закрыт |
| V3 | `…_v3_plan_attention.md` | `…_v3_results.md` | `…/v3/` | `…_v3/` | закрыт; B0 = ranking recipe |
| V3.1 prep (manual) | `v3_1_data_preparation_protocol.md` | — | `…/v3_1_prep/` | `…_v3_1_prep/` | prep; overlay history отдельная |
| V3.1 auto silver | `v3_1_automated_data_preparation_protocol.md` + campaign | `qa_chunk_selection_v3_1_campaign_results.md` | `…/v3_1_auto_prep/` | `…_v3_1_auto/`, `…_v3_1_auto_v2/` | auto overlays; freeze_skipped в analysis |
| V3.1 ranking pkg | continuation protocol | — | `…/v3_1/` | **`artifacts/…_v3_1/` на диске нет** | код есть; ranking root не материализован как полный успешный freeze-цикл в том же смысле, что V4.x |
| V3 D4/D5 | (в V3 plan / d45 pkg) | автодамп `results_v3_d45.md` | `…/v3_d45/` | `…_v3_d45/` | audit/oracle |
| V4 | `v4_experiment_plan_7_strategies.md` | `…_v4_results.md` | `…/v4/` | `…_v4/` | закрыт; родился V6 |
| V4.1 | `v4_1_reader_centered_continuation_protocol.md` | `…_v4_1_results.md` | `…/v4_1/` | `…_v4_1/` | V6 OOF + weights; K=5 reference |
| V4.2 | `v4_2_selected_head_attention_protocol.md` | `…_v4_2_results.md` | `…/v4_2/` | `…_v4_2/` | ATT1 fail; freeze V6-K5; CORE3 diagnostic |
| V4.3 | `v4_3_semantic_reader_agreement_protocol.md` | `…_v4_3_results.md` | `…/v4_3/` | `…_v4_3/` | SEM1 fail; freeze V6-K5 |

**Устаревший handoff:** `hypothesis/qa_chunk_selection_current_status.md` (2026-09-04) останавливается на «разметить V3.1 prep». Не использовать как «где мы сейчас».

---

## 4. Вселенная данных (общая для V3+)

Заморожена в V3, дальше только **читается**.

| Величина | Значение | Путь |
|---|---|---|
| Dataset JSONL | SciREX aspect-QA instances | `artifacts/qa_chunk_selection_v3/data/dataset.jsonl` |
| Dataset SHA256 | `3be581a8…5844dc9b` | совпадает с V4.3 `source_hashes.json` |
| CV splits | 5 folds, document-level, seed 42 | `artifacts/qa_chunk_selection_v3/data/v3_cv_splits.json` |
| Splits SHA256 | `aa34bb4f…bc0cf7` | то же |
| Документы в CV | **240** | (28 unlocked docs без robust_positive — не баг) |
| Unlocked answerable | **453** | |
| INTERNAL_TEST records | **78** (locked) | |
| OOF rows при 3 seeds | **1359** | |
| Outer runs | 5 | train / calibration / eval roles per run |

Roles API: `experiments.qa_chunk_selection.v3.split.records_for_role`.

Два слоя меток (не путать):

- **`source_label`** (Phase 1): positive / hard_negative / easy / ambiguous — для метрик HN/Amb/GSRN и TRAIN sanity.
- **`target_status` overlay** (V3.1): target_evidence / target_mismatch / … — отдельный слой; в V4.x primary path ranking **не опирается** на заполненный TM overlay как на главный сигнал.

---

## 5. Эволюция reference (коротко, по смыслу)

```text
E0/V2  → bi-encoder / CE recipes
V3     → B0 (bi-encoder V2-B2a) wins OOF; Token F1 ~27.36
         attention/span fail safeguards or don’t beat B0 fairly
V4     → reader-centered strategies; V6 = B0 + answer agreement (+ support clusters)
V4.1   → V6 confirmed OOF; knockouts: gain mostly from reader agreement, not B0 score
         K=10 not separated from K=5 → keep K=5
V4.2   → selected-head attention on V6 ≈ null; CORE1/2 weaker than V6;
         CORE3 (reader-only, no B0) high point F1 but diagnostic-only
V4.3   → SemRel on V6 ≈ null (SEM1);
         fair LEXCORE > historical CORE3 pointwise;
         SEMCORE ≳ LEXCORE but CI through 0;
         ADAPT1 exploratory win vs V6, not freeze
```

Формула V6 (концептуально; реализация в `v4/v6_reader.py` + isolated apply в `v4_1/ranking.py`):

```text
S_answer  = exact_freq + mean_token_f1 + majority
S_V6      = α * b0_score + β * S_answer + γ * source_clusters
```

Default / typical weights (также fallback в contract): `α=2.0`, `β=0.25`, `γ=0.5`. Per-run OOF weights — в V4.1 `r1_v6_oof/weights_by_run.json` (для run `"0"` в проверенном файле как раз эти значения).

V4.3 systems (не все freeze-кандидаты):

```text
SEM1     = Z_q(S_V6) + δ_s SemRel          # SemRel = z(SemTop2), self excluded
LEXCORE  = λ_m Majority + λ_t (mean_token_f1/100)
SEMCORE  = λ_m Majority + λ_s SemRel
ADAPT1   = hard route SEMCORE vs V6 by ConsensusConf = Coh(T*)-Between(T*)
SEMCTX1  = Z(SEM1) + δ_c SupportRel        # только если SEM1 gate; в V4.3 skipped
```

`PRIMARY_FINAL_SYSTEMS` в V4.3 contract: `B0-live`, `V6-K5`, `SEM1` only. LEXCORE/SEMCORE/ADAPT1/CORE3 **не** auto-admissible.

---

## 6. Карта репозитория

```text
/home/aiserver/POLINA/UnTIE_project/
  untie/                          # production — READ-ONLY for these exps
  datasets/                       # frozen SciREX sources — READ-ONLY
  experiments/qa_chunk_selection/ # all experiment code
  artifacts/qa_chunk_selection*/  # all run outputs (isolated per version)
  hypothesis/                     # protocols + human reports + this handoff
  logs/qa_chunk_selection*.log    # background logs
  tests/test_qa_chunk_selection*.py
  untie_venv/                     # project venv (launcher default PYTHON)
```

### 6.1 Shared (корневой пакет `experiments/qa_chunk_selection/`)

Используется всеми версиями; не «версия эксперимента».

| Модуль | Роль |
|---|---|
| `schema.py` | `InstanceRecord`, chunks, gold_answers |
| `metrics.py` | hit@k, HN, MRR, token_f1 helpers, ranking_row, bootstrap |
| `models.py` | load frozen extractive QA / lexical smoke |
| `progress.py` | stdout + `progress.jsonl` |
| `dataset.py` | load frozen SciREX + Phase1 labels |
| `io_util.py` | PROJECT_ROOT, JSON helpers |
| `contract.py` / `config.py` / `pipeline.py` / `run.py` | **E0** orchestration |
| `ranking.py`, `evaluate.py`, `answerability.py`, … | E0 A–D |

### 6.2 Версионированные пакеты

Каждый обычно содержит: `contract.py`, `config.py`, `io_guard.py`, `pipeline.py`, `run.py`, `__main__.py`, `launch_background.sh`, плюс domain-модули.

**Актуальный код для чтения «как устроен последний цикл»:**  
`/home/aiserver/POLINA/UnTIE_project/experiments/qa_chunk_selection/v4_3/`

| Файл V4.3 | Роль |
|---|---|
| `contract.py` | grids, roots, PROTECTED_WRITE_ROOTS, historical metrics, PRIMARY_FINAL_SYSTEMS |
| `config.py` | stages aliases (`primary`, …), paths, smoke flags |
| `io_guard.py` | refuse prior trees + refuse completed V4.3 OUT_DIR |
| `run.py` / `__main__.py` | CLI → Pipeline |
| `pipeline.py` | все стадии: init→r0→cache→embed→sanity→features→sem1→lexcore→semcore→complement→adapt1→semctx1→compare→freeze→report |
| `encoder.py` | frozen MiniLM (+ HashEncoder smoke), disk embedding cache |
| `text.py` | NFC/whitespace norm, valid_answer, local_evidence_sentence |
| `features.py` | SemTop2/SemMean/SemRel, best coherent triple, SupportGain, clusters |
| `ranking.py` | SEM1/LEXCORE/SEMCORE/SEMCTX scores; B0-rank tie-break; ADAPT routing |
| `select.py` | `choose_admissible_cal` (δ before HN), `choose_max_f1`; re-exports v4_2 helpers |
| `evaluate.py` | **re-export** V4.1 seed-aware evaluate (paired_delta по qid+seed) |
| `sanity.py` | TRAIN-only pos–pos vs pos–HN cosine |
| `report.py` | autodump `reports/results_v4_3.md` |

**Что V4.3 переиспользует (не копировать слепо — импортировать):**

- `v4_1.evaluate`, `v4_1.ranking` (`apply_v6_isolated`, `shortlist_set_ok`, …)
- `v4_2.cache_util.ReadThroughCache`, `v4_2.select`, `v4_2.features.zscore_map`, `v4_2.heads.union_topk`
- `v4.v6_reader.agreement_features`, `v4.shortlist`, `v4.qa_shortlist`, `v4.answers`
- `v3.data` / `v3.split` для universe
- `v2.cache.FeatureCache`, `v2.hn_flags`

**Не редактировать** пакеты `v4_2/**`, `v4_1/**`, … «чтобы починить V4.3» — чинить в `v4_3/` или заводить `v4_4/`.

### 6.3 Ключевые артефакты V4.3

Корень: `artifacts/qa_chunk_selection_v4_3/`

| Путь | Зачем агенту |
|---|---|
| `final/freeze_manifest.json` | freeze decision |
| `final/model_selection.json` | selected=V6-K5 |
| `analysis/compare.json` | полная OOF таблица + deltas |
| `sem1/gate.json` | H1 verdict |
| `lexcore/`, `semcore/`, `adapt1/` | OOF + per-run CAL |
| `reference/{b0_live,v6_k5_v4_1,core3_v4_2_historical}/` | R0 copies |
| `regression/` | identities, leakage |
| `semantic_sanity/` | TRAIN encoder check |
| `contract.json`, `config_resolved.json` | frozen run config |
| `progress.jsonl` | timeline |
| `reports/results_v4_3.md` | autodump only |

Лог: `logs/qa_chunk_selection_v4_3.log` (прогон 2026-09-09 ~14:02→17:35, exit=0, cuda:0).

V6 weights source of truth:  
`artifacts/qa_chunk_selection_v4_1/r1_v6_oof/weights_by_run.json`  
B0 OOF rows: under V3/V4/V4.1 artifact trees (V4.3 R0 identity-сверяет).

---

## 7. Как устроен прогон (операционно)

### 7.1 Background launch (предпочтительный паттерн)

```bash
cd /home/aiserver/POLINA/UnTIE_project
STAGES=primary DEVICE=auto \
  bash experiments/qa_chunk_selection/v4_3/launch_background.sh
```

Важно: скрипты должны быть **LF**, не CRLF (`set -euo pipefail` иначе ломается с `pipefail: invalid option name`).

Env (V4.3 launcher): `STAGES`, `SMOKE`, `DEVICE`, `SEED`, `SKIP_QA`, `ALLOW_EXISTING_OUT`, `FORCE_SEMCTX`, `OUT_DIR`, `LOG`, `PYTHON` (default `untie_venv/bin/python`).

### 7.2 Stages alias `primary` (V4.3)

```text
init, r0, cache, embed, sanity, features,
sem1, lexcore, semcore, complement, adapt1, semctx1,
compare, freeze, report
```

Smoke: `SMOKE=1` → synthetic data, HashEncoder, no MiniLM download, no real QA.

### 7.3 Тесты

```bash
cd /home/aiserver/POLINA/UnTIE_project
./untie_venv/bin/python -m pytest tests/test_qa_chunk_selection_v4_3.py -q
# или: python3 -m pytest ...  (если venv недоступен)
```

Есть также: `tests/test_qa_chunk_selection_v4.py`, `_v4_1.py`, `_v4_2.py`, плюс более ранние `test_qa_chunk_selection*.py`.

Перед большим прогоном: unit+smoke на новом пакете; **не** запускать GPU primary без явной просьбы пользователя (в прошлых циклах пользователь просил «напиши код, скажи как запустить в фоне»).

### 7.4 QA cache

V4+ считает extractive QA только на B0 top-K. V4.3 при `reuse_v4_1_qa_cache=true` читает cache из V4.1/V4/V4.2 read-through; в завершённом прогоне `hit_rate=1.0`, **2850/2850** fallback hits (новых forward не было).

Inference hash: `v4-topk` (`QA_INFERENCE_HASH`).

---

## 8. Метрики и оценка (не путать)

| Правило | Деталь |
|---|---|
| Primary endpoint | Token F1 |
| Pairing | `(question_id, seed)` — **не** qid-only (это баг старых сравнений) |
| Bootstrap | mean over seeds per question, then clustered by `document_id`, n_boot=5000 |
| HN safeguard | обычно HN ≤ ref_HN + 0.02 на CAL |
| Success gate SEM1 (V4.3) | ΔF1>0 **и** CI excludes 0 **и** HN ok |
| Shortlist metrics | Hit@1/MRR могут отражать isolated ranking; Hit@5 должен = B0 |

Реализация парных дельт: `experiments/qa_chunk_selection/v4_1/evaluate.py` (`paired_delta`, `seed_design`).

---

## 9. Что V4.3 показал (только для ориентации; детали в отчёте)

Читать полностью: `hypothesis/qa_chunk_selection_experiments_v4_3_results.md`.

| Гипотеза | Результат |
|---|---|
| H1 SEM1 vs V6 | **fail** — ΔF1 +0.068, CI [−0.68, +0.82], Hit@1 хуже |
| H2 SEMCORE vs LEXCORE | **fail** сильного критерия — ΔF1 +0.77, CI through 0 |
| H3 ADAPT1 | exploratory: vs V6 ΔF1 +1.38, CI excludes 0; **не** freeze |
| H4 SEMCTX1 | skipped |

Полезные diagnostics (не freeze):

- TRAIN MiniLM sanity: pos–pos ≫ pos–HN (~0.96–0.98).
- Fair LEXCORE Token F1 36.58 > historical CORE3 35.44 (point); CORE3 не baseline для H2.
- SEMCORE max F1 37.33; чинит task vs LEXCORE, но HN/Hit tradeoffs.
- Encoder frozen MiniLM; semantic_available = 100% shortlists в том прогоне.

---

## 10. Типичные ловушки для следующего агента

1. **Перезапустить V4.3 в тот же OUT_DIR** → `io_guard` откажется (есть `sem1/metrics.json` / report / model_selection). Нужен новый OUT_DIR или явный `ALLOW_EXISTING_OUT=1` (обычно нежелательно).
2. **Взять Token F1 SEMCORE/LEXCORE и объявить победу** → протокол V4.3 не делает их primary-final; freeze = V6-K5.
3. **Сравнивать системы по qid-only** → неверно; использовать V4.1 evaluate.
4. **Положить gold в SemRel / ranking features** → запрещено.
5. **Открыть INTERNAL_TEST «для уверенности»** → запрещено без нового протокола и явного запроса.
6. **Ретюн V6 «под SemRel»** → запрещено контрактом V4.3.
7. **Править `v4_2` select tie-break** ради SEM1 → не надо; в V4.3 есть свой `choose_admissible_cal` (меньший δ **до** HN).
8. **Доверять `qa_chunk_selection_current_status.md`** → устарел (V3.1 prep era).
9. **Путать V4.1 freeze string `V6-K10` с текущим reference** → текущий confirmed = **V6-K5**.
10. **CRLF в `.sh`** → чинить line endings перед launch.
11. **Смешать U80 / attention / K10 в одну таблицу с V4.3** → нет.
12. **Считать автодамп научным отчётом** → нет; писать `hypothesis/…_results.md`.

---

## 11. Если пользователь просит «следующий эксперимент»

Минимальный чеклист (не начинать код до протокола, если задача — новый цикл):

1. Прочитать V4.3 results + protocol; зафиксировать, какая гипотеза ещё открыта.
2. Написать **новый** `hypothesis/v4_X_…_protocol.md` (или явно продолжить названный).
3. Создать **новый** пакет `experiments/qa_chunk_selection/v4_X/` с `io_guard`, запрещающим запись во все предыдущие roots **включая v4_3**.
4. Переиспользовать V6 weights / B0 / QA cache read-only.
5. Unit + smoke tests; launch script LF; **не** стартовать GPU primary без просьбы.
6. После прогона — человеческий `…_results.md`, не только autodump.
7. Обновить/добавить handoff status (этот файл или следующий `…_current_status_…`).

Кандидаты, которые **уже намечены результатами**, но **не авторизованы** как V4.4 без пользователя:

- Confirmatory reader-only (LEXCORE vs SEMCORE vs V6) с заранее зафиксированным endpoint.
- Отдельный ADAPT protocol (не «дотюнить V4.3»).
- Не возвращать SemTop2-on-V6 как primary без новой feature.
- SEMCTX только если снова появится успешный H1-подобный gate или отдельный unconditional protocol.

---

## 12. Обязательное чтение (порядок)

1. Этот файл (операционный handoff).
2. `hypothesis/qa_chunk_selection_experiments_v4_3_results.md` — что показал последний цикл.
3. `hypothesis/v4_3_semantic_reader_agreement_protocol.md` — правила H1–H4 / gates.
4. При необходимости назад: `…_v4_2_results.md`, `…_v4_1_results.md`, `…_v4_results.md`.
5. Код: `experiments/qa_chunk_selection/v4_3/pipeline.py` + `contract.py` + `ranking.py`.
6. Freeze: `artifacts/qa_chunk_selection_v4_3/final/freeze_manifest.json`.

Не читать как source of truth: `artifacts/.../reports/results_v4_3.md`, `hypothesis/qa_chunk_selection_current_status.md`.

---

## 13. Быстрые проверки достоверности окружения

```bash
# freeze
cat artifacts/qa_chunk_selection_v4_3/final/freeze_manifest.json

# V6 weights exist
test -f artifacts/qa_chunk_selection_v4_1/r1_v6_oof/weights_by_run.json && echo ok

# V3 universe
python3 -c "import json;print(json.load(open('artifacts/qa_chunk_selection_v4_3/source_hashes.json')))"

# tests
./untie_venv/bin/python -m pytest tests/test_qa_chunk_selection_v4_3.py -q --tb=line
```

Ожидаемый freeze excerpt:

```json
{
  "final_architecture": "V6-K5",
  "v6_retuned": false,
  "primary_k": 5,
  "internal_test": "locked",
  "sem1_gate": false,
  "encoder": "sentence-transformers/all-MiniLM-L6-v2",
  "encoder_frozen": true
}
```

---

## 14. Контакты с пользователем (стиль работы в этих циклах)

- Писать по-русски, коротко и по делу в чате; детали — в md.
- Не коммитить и не пушить без явной просьбы.
- Не запускать полный GPU-прогон без явной просьбы; дать команду background launch.
- Не открывать INTERNAL_TEST «заодно».
- После эксперимента: отдельный scientific results md + понятное резюме в чат.
- Production UnTIE / Phase 1 — вне scope этих чатов, если пользователь не сказал иначе.

---

*Конец handoff. Если противоречие между этим файлом и freeze/results на диске — верить артефактам и results md; затем обновить handoff.*
