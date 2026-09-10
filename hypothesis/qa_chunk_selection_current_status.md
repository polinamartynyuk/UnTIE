# Текущий статус: отбор чанков (E0 → V2 → V3 → подготовка V3.1)

Документ для передачи работы другому агенту. Дата фиксации: **2026-09-04**. Это не протокол эксперимента и не отчёт о результатах — только карта «где мы, что уже сделано, куда смотреть, что нельзя трогать».

**Где мы сейчас.** Полный прогон **V3** завершён. Честный победитель V3 — **B0** (bi-encoder, recipe V2-B2a). Overlay `target_status` в V3 **пустой** (очередь из 400 чанков, **0** заполненных меток), поэтому гипотеза H3.1 (target-mismatch) **не проверялась**. Код подготовки данных **V3.1** написан, изолирован и уже экспортировал annotator pack. Следующий человеческий шаг — **разметить очередь**. Freeze overlay и обучение по [`v3_1_experiment_continuation_protocol.md`](v3_1_experiment_continuation_protocol.md) **ещё нельзя**.

---

## 1. Одна строка для следующего агента

```text
V3 ranking done (winner = B0). Overlay labels = 0/400.
V3.1 prep code exists. Annotate to_fill.jsonl, then ingest+freeze.
Do not train V3.1. Do not open INTERNAL_TEST. Do not overwrite E0/V2/V3 artifacts.
```

---

## 2. Линейка пакетов (не путать)

| Пакет | План | Человеческий отчёт | Код | Артефакты | Состояние |
|---|---|---|---|---|---|
| E0 | [`qa_chunk_selection_experiments_v1.md`](qa_chunk_selection_experiments_v1.md) | числа живут в плане E0 / автодамп | `experiments/qa_chunk_selection/` (**без** `v2/` / `v3/`) | `artifacts/qa_chunk_selection/` | закрыт, **не перезаписывать** |
| V2 | [`qa_chunk_selection_experiments_v2.md`](qa_chunk_selection_experiments_v2.md) | [`qa_chunk_selection_experiments_v2_results.md`](qa_chunk_selection_experiments_v2_results.md) | `experiments/qa_chunk_selection/v2/` | `artifacts/qa_chunk_selection_v2/` | закрыт, **не перезаписывать** |
| V3 | [`qa_chunk_selection_experiments_v3_plan_attention.md`](qa_chunk_selection_experiments_v3_plan_attention.md) | [`qa_chunk_selection_experiments_v3_results.md`](qa_chunk_selection_experiments_v3_results.md) | `experiments/qa_chunk_selection/v3/` | `artifacts/qa_chunk_selection_v3/` | прогон завершён, **не перезаписывать** |
| V3.1 prep | [`v3_1_data_preparation_protocol.md`](v3_1_data_preparation_protocol.md) | этого отчёта ещё нет (разметки нет) | `experiments/qa_chunk_selection/v3_1_prep/` | `artifacts/qa_chunk_selection_v3_1_prep/` | код готов; **ждём разметку** |
| V3.1 ranking | [`v3_1_experiment_continuation_protocol.md`](v3_1_experiment_continuation_protocol.md) | — | кода пакета ещё нет | — | **запрещено** до freeze overlay |

Автодампы JSON→MD **не** считать человеческими отчётами:

- `artifacts/qa_chunk_selection_v2/reports/results_v2.md`
- `artifacts/qa_chunk_selection_v3/reports/results_v3.md`

Production (`untie/`, `datasets/`, Phase 1) **не менялся** и не должен меняться этими экспериментами.

---

## 3. Что уже закрыто (факты, не гипотезы)

### 3.1 V2

- Успешный полный прогон: **2026-09-02 14:05:23**, wall-clock ≈ 41 мин, `cuda:0`.
- Человеческий разбор: [`qa_chunk_selection_experiments_v2_results.md`](qa_chunk_selection_experiments_v2_results.md).
- Для продолжения достаточно: V2 дал рабочий bi-encoder recipe, который V3 взял как B0. Не перезапускать V2.

### 3.2 V3

- Успешный полный прогон: **2026-09-02 17:26:36 → 18:35:59**, wall-clock **4163.4 с ≈ 69 мин**, `cuda:0`, `smoke=False`, `skip_qa=False`, `resume=true`.
- Config hash: `950c345e0660827a`.
- Лог: `logs/qa_chunk_selection_v3.log`.
- Человеческий разбор (читать его, не автодамп): [`qa_chunk_selection_experiments_v3_results.md`](qa_chunk_selection_experiments_v3_results.md).
- Стадии, которые реально шли: `prepare, split, overlay-sample, b0–b3, structure, positive-quality, at0–at4, span-ranker, oof-compare, report`.
- Стадии, которые **не** входили в default и **не** запускались: `b4, at4-d, at5, d3, d4, d5, freeze-final, final-eval`.
- **INTERNAL_TEST не открыт.**

Первый запуск полного пайплайна упал (`AttributeError: PipelineV3 has no attribute '_device'`). После фикса продолжили с `resume=true`. CV-сплиты к тому моменту уже были заморожены и **не переписывались**.

Universe (не путать с формулировками протокола «268 документов»):

| Величина | Значение | Где лежит |
|---|---|---|
| Все записи V2 snapshot | 955 | `artifacts/qa_chunk_selection_v3/data/prepare_summary.json` |
| Unlocked answerable вопросы | **453** | то же |
| Документы в CV | **240** | `data/v3_cv_splits.json` (`n_documents`) |
| INTERNAL_TEST records (опись, без QA) | 78 | `prepare_summary.json`; сплит закрыт |
| Folds | 5, document-level, seed 42 | `v3_cv_splits.json` |
| `document_id_hash` | `21cf53200c0350ad4176b378b7c56c6eadd6f31cf479560ce1db87e12a05f90d` | `v3_cv_splits.json` |
| `split_payload_hash` | `d3647846e7be929791b359a1cb93b6722dcd0df6b8567159c9fd6d50eba2fcb8` | `v3_cv_splits.json` |

Почему 240, а не 268: 28 unlocked документов не имеют `robust_positive`. Это зафиксировано в человеческом отчёте V3, не «баг сплита».

Очередь overlay (сэмплинг сделан, меток нет):

| Величина | Значение | Где лежит |
|---|---|---|
| Очередь | 400 (200 hard + 200 representative) | `artifacts/qa_chunk_selection_v3/annotation/queue.jsonl` |
| Подтипы hard | GSRN 192, FN-cand 105, question_sensitive 103 | `annotation/sampling_summary.json` |
| Заполненные `target_status` | **0** | `annotation/target_status.jsonl` пустой |
| Taxonomy | `v3-target-status-1` | `sampling_summary.json` |

Следствие пустого overlay: `P_v3_target == P_original`; TM-метрики NaN/null; B1 ≈ B0; **H3.1 фактически не тестировалась**.

**Выбор модели V3 (OOF unlocked, не INTERNAL_TEST):**

- Selected: **b0**.
- Admissible: b0, b1, b2, span_ranker.
- Rejected: b3, structure, attention_only, text_plus_attention.
- Pareto: b0, b1.
- Честный кандидат на любой будущий freeze INTERNAL_TEST: **V3-B0, recipe b2a, M=1**. Не B1, не attention, не SpanRanker.

Ключевые числа B0 (3 seeds, n=1359 = 453×3), из человеческого отчёта:

- Token F1 **27.36**, bootstrap 95% CI **24.30–30.56**
- EM 20.75%
- Hit@1 **36.72%**, Hit@5 79.03%, Hit@10 92.20%
- HN@1 4.86%, pairwise_clean **75.14%**
- Amb@1 50.85%, GSRN@1 33.92%, F1\|pos 61.70

Сравнение с attention/span честно делать на **seed=42**, n=453: B0 Token F1 **26.46**, не на трёхсидовом 27.36. Attention/span считались только на seed 42.

Известные оговорки V3 (уже разобраны в отчёте, не «открытые баги для фикса в V3»):

- Attention AT1/AT3 считались на **полном unlocked**, не только TRAIN_FOLDS (отклонение от плана).
- Attention_only: HN@1 **18.98%**, pairwise 51.5% — падает по safeguards.
- AT1 AUROC 0.519, `has_signal=false`.
- SpanRanker: ranking ids всё ещё B0 (Hit@1 = B0-s42); K=3; Token F1 24.40; `F1|pos` 37.32 вводит в заблуждение (`selected_chunk_label` vs `qa_chunk`).
- Positive-quality: oracle best-pos F1 **73.01**; solvable F1≥50 **76.4%**. D4/D5 gates пройдены, но стадии **не запускались**.

### 3.3 Почему «у чанков уже есть метки», но overlay всё равно нужен

Phase 1 `source_label` (positive / hard-negative / easy / ambiguous) — **другой слой**, чем V3 `target_status` (`target_evidence` / `target_mismatch` / `partial_support` / `unresolved`). Ambiguous и GSRN **нельзя** автоматически превратить в TM. `source_label` **никогда не переписывается**. Overlay пишется отдельно.

---

## 4. Текущий этап: подготовка overlay V3.1

Протокол: [`v3_1_data_preparation_protocol.md`](v3_1_data_preparation_protocol.md).

Код: `experiments/qa_chunk_selection/v3_1_prep/` (изолированный пакет; пишет **только** в `artifacts/qa_chunk_selection_v3_1_prep/`).

| Файл | Роль |
|---|---|
| `contract.py` | контракт изоляции |
| `config.py` | пути, пороги, стадии |
| `io_guard.py` | запрет записи в E0 + V2 + **V3** |
| `taxonomy.py` | enum'ы target-status (переиспользует V3) |
| `context.py` | соседние предложения, контекст аннотатора |
| `annotator.py` | export / ingest |
| `qc.py` | validate, audit, coverage, CV inventory, readiness |
| `freeze.py` | freeze overlay |
| `pipeline.py` | оркестрация |
| `run.py`, `__main__.py` | CLI |
| `launch_background.sh` | фон |

Тесты: `tests/test_qa_chunk_selection_v3_1_prep.py` (проходили вместе с тестами V3 на момент написания пакета).

Переиспользует из V3, не копируя артефакты: overlay validate, роли (`v3_target_positives`, `verified_target_mismatch`), `records_from_jsonl`, `records_for_role`, `neighbor_sentences`, `ProgressLogger`, taxonomy enums.

Default stages пакета (без freeze): `init, export-annotator, ingest, validate, audit, coverage, cv-inventory, readiness`.

Пороги по умолчанию: `min_train_questions_with_TM=10`, `min_eval_questions_with_target_and_TM=10`.

### 4.1 Что уже лежит на диске (prep)

Корень: `artifacts/qa_chunk_selection_v3_1_prep/`.

Annotator pack (экспорт сделан, **n=400**, `annotator_id=annotator_1`, скоры скрыты от аннотатора):

- `annotator/to_fill.jsonl` — **сюда писать метки**
- `annotator/index.html`
- `annotator/guideline.json`
- `annotator/export_summary.json`

Hashes источников (чтение V3, не запись): `source_hashes.json`

```text
queue     b2d4341906d67a2a1183e605c9f25d866c54f914dd00b6d2be5f8a34c1fe0cf6
metadata  d8a2e7cf4bbeff5bd6fdd751ca3ff178c38f0e4b433c54c45c12242a64a928ad
splits    aa34bb4f4f6c31d318b7f472b166a5e3bda1ca6dd03891d6b07efef211bc0cf7
dataset   3be581a8dfb015ddbc5a6fd02d411a3ab0aa7dadc4ac06c7833618135844dc9b
b0_oof    cf2e979bd2f35a1246e05a3fb7f9e8a25a838bae4048dd449f55e40d2e83b02e
```

QC после попытки ingest **пустой** разметки (2026-09-04 ≈ 16:27):

- `analysis/overlay_counts.json` — все счётчики нули
- `analysis/coverage.json` — `n_eval_questions_with_target_and_TM=0`; TM@1 lower bound 0; `TargetStatusCoverage@1 ≈ 0.488` (это покрытие **известной** target-роли B0 seed=42, не verified TM)
- `analysis/cv_tm_inventory.json` — во всех 5 outer runs `n_train_questions_with_TM=0`, `n_eval_questions_with_target_and_TM=0`
- `analysis/readiness.json` — `"v3_1_training_allowed": false`, `"n_queue": 400`, `"n_filled": 0`, `"verified_target_mismatch_gt_0": false`

Лог: `logs/qa_chunk_selection_v3_1_prep.log`. Прогресс: `artifacts/qa_chunk_selection_v3_1_prep/progress.jsonl`.

Frozen overlay **ещё нет**. Ожидаемый файл после успешного freeze:

- `annotation/target_status_v3_1_frozen.jsonl`

### 4.2 Что уже пробовали и почему freeze не прошёл

Пользователь запустил freeze **до заполнения меток**. Ingest: `n_filled=0`. Freeze отказался (это правильно).

Два исправления в коде после этой попытки (уже в дереве `v3_1_prep/`):

1. Пустой `overlay_counts` больше не ставит ложный блокер `distributions_built=false`. Достаточно ключа `"n_annotated"` в payload.
2. Freeze при неготовых условиях **не падает traceback**. Пишет `analysis/freeze_skipped.json` и выходит с кодом **2**.

На диске `readiness.json` от 16:27 может ещё содержать **старое** `"distributions_built": false` — это артефакт прогона **до** фикса. Реальный блокер сейчас: **ноль заполненных меток**, не distributions. После разметки нужно заново прогнать ingest→readiness→freeze.

`--allow-partial-freeze` **не использовать**, чтобы «просто начать учить».

---

## 5. Что делать дальше (операционная последовательность)

### Сейчас — только разметка

Заполнять `artifacts/qa_chunk_selection_v3_1_prep/annotator/to_fill.jsonl` или UI `annotator/index.html`.

Обязательные поля:

- `v3_target_status`
- `v3_annotation_confidence`
- `v3_annotation_reason`

Если mismatch → `v3_mismatch_type`. Если evidence → `v3_gold_relation`. **Не** переписывать `source_label`. Не подглядывать ranking scores (они скрыты в очереди специально).

Пустой / unverified / ambiguous **не** считать TM и **не** считать «точно не TM».

### После того как метки есть

```bash
cd /home/aiserver/POLINA/UnTIE_project
STAGES=init,ingest,validate,audit,coverage,cv-inventory,readiness,freeze \
  bash experiments/qa_chunk_selection/v3_1_prep/launch_background.sh
```

Успех freeze:

- появился `artifacts/qa_chunk_selection_v3_1_prep/annotation/target_status_v3_1_frozen.jsonl`
- `analysis/readiness.json` содержит `"v3_1_training_allowed": true`
- `verified_target_mismatch_gt_0: true`
- каждый outer TRAIN проходит `min_train_questions_with_TM`
- есть достаточно `P_v3_target vs TM` по `min_eval_questions_with_target_and_TM`

Только после этого читать и реализовывать [`v3_1_experiment_continuation_protocol.md`](v3_1_experiment_continuation_protocol.md).

### Когда (ещё не сейчас) начнётся ranking V3.1

Краткий смысл continuation, чтобы не изобретать план заново:

- Те же 5 document-level folds, те же роли 3 TRAIN / 1 CAL / 1 EVAL, тот же unlocked universe, тот же reader, те же primary QA metrics. **Новые outer folds не создавать.**
- Сначала V3.1-A: audit overlay **до** обучения.
- Потом saturation/reference B0 (возможен B0-max8 как **заранее** зафиксированный reference, не post-hoc выбор по EVAL F1).
- Основной первый эксперимент — B1-TM, не «ещё один full-document ranker вместо B0».
- Attention: после отрицательного V3 — только audit/fusion **после** overlay, не как замена B0.
- Текущую ветку SpanRanker V3 **временно остановить** (раздел 25 continuation).
- INTERNAL_TEST не открывать.

Кода `experiments/qa_chunk_selection/v3_1/` (ranking) **ещё нет**. Его нельзя начинать, пока freeze+readiness не зелёные.

---

## 6. Запреты (жёсткие)

- Не открывать INTERNAL_TEST. Не считать QA с этого сплита.
- Не перезаписывать `artifacts/qa_chunk_selection/`, `artifacts/qa_chunk_selection_v2/`, `artifacts/qa_chunk_selection_v3/`.
- Не менять production: `untie/`, `datasets/`, Phase 1.
- Не использовать gold в ranking / filtering / shortlist / attention-runtime / агрегации ответа.
- Не стартовать обучение V3.1 по continuation protocol до frozen overlay и `v3_1_training_allowed: true`.
- Не мапить empty/unverified/ambiguous → TM или not-TM.
- Не переписывать `source_label`.
- Не выбирать max5 vs max8 (или любую другую гиперточку) **после** просмотра OOF EVAL Token F1.
- Не учить second-stage reranker на shortlist B0, который уже видел те же TRAIN examples (нужен cross-fitted shortlist — это в continuation, раздел 13).
- GPU/полный ranking в этом статусе **не запускать**. Prep-пакет GPU не требует.

---

## 7. Быстрые указатели по коду V3 (если нужно читать, не запускать)

Корень: `experiments/qa_chunk_selection/v3/`.

Полезные артефакты V3 (только чтение):

| Что | Путь |
|---|---|
| Frozen splits | `artifacts/qa_chunk_selection_v3/data/v3_cv_splits.json` |
| Prepare summary | `artifacts/qa_chunk_selection_v3/data/prepare_summary.json` |
| Resolved config | `artifacts/qa_chunk_selection_v3/config_resolved.json` |
| Overlay queue | `artifacts/qa_chunk_selection_v3/annotation/queue.jsonl` |
| Overlay labels (пусто) | `artifacts/qa_chunk_selection_v3/annotation/target_status.jsonl` |
| Sampling summary | `artifacts/qa_chunk_selection_v3/annotation/sampling_summary.json` |
| OOF compare / selection | `artifacts/qa_chunk_selection_v3/reports/` и JSON рядом с моделями |
| B0 OOF (seed 42), нужен coverage V3.1 | внутри `artifacts/qa_chunk_selection_v3/` (prep хэширует как `b0_oof`) |

Launcher V3 (не нужен сейчас): `experiments/qa_chunk_selection/v3/launch_background.sh` (CRLF `bash\r` уже чинили; не ломать).

---

## 8. Критерий «можно передать ranking-агенту»

Можно, когда одновременно верно:

1. В `to_fill.jsonl` (или ingest drafts) есть заполненные overlay-метки, не 0/400.
2. Freeze создал `annotation/target_status_v3_1_frozen.jsonl`.
3. `readiness.json` → `"v3_1_training_allowed": true`.
4. CV inventory показывает ненулевые TM в TRAIN и достаточный EVAL `P_v3_target vs TM`.
5. Ни один файл в деревьях E0/V2/V3 не изменился этой подготовкой.

До этого момента следующий агент должен **только** помогать с разметкой, QC ingest и freeze, либо читать отчёты V2/V3. Не писать ranking-код V3.1.
