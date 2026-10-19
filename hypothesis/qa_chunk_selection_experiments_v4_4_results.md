# Результаты V4.4: adaptive reader-consensus и development stability replay

Дата анализа: **2026-09-10**. Анализ фактически завершённого прогона по [протоколу V4.4](v4_4_adaptive_reader_consensus_protocol.md).

Это отдельный аналитический отчёт, а не копия автоматического `reports/results_v4_4.md`. Числа взяты из сохранённых артефактов; дополнительные сводки получены из тех же predictions. При подготовке отчёта не запускались обучение, QA inference, новый подбор параметров или INTERNAL_TEST. Новые CI для закрытой H2 и always-controls не рассчитывались.

## 1. Основной вывод

**Preferred development candidate остаётся V6-K5. H1 ASEM не прошёл gate из-за потери Hit@1, а не из-за нулевого прироста F1.**

ASEM повышает Token F1 с **34.6249 до 35.1112**. Сохранённый paired bootstrap даёт ΔF1 **+0.486305**, 95% **development CI [0.005367; 0.992832]**. Но Hit@1 падает с **42.8992% до 41.8690%**, то есть на **1.0302 п.п.**, при допуске только **0.5 п.п.**. HN@1 увеличивается на 0.3679 п.п., что укладывается в допуск 1 п.п.

Следствия зафиксированной иерархии:

- H1: **fail**; из четырёх условий выполнены три.
- H2 ALEX vs ASEM: **не проверялась**. ALEX рассчитан как обязательный описательный output и не может стать победителем post-hoc.
- Local support: **пропущен по H1 fail**, хотя `support=true`.
- ASOFT: не считался; `soft=false`, кроме того H1 не пройден.
- SEM-FIX и LEX-FIX имеют более высокие точечные F1, но это always-controls, не допустимые финальные кандидаты V4.4.
- INTERNAL_TEST остаётся locked; B0-final не обучен; имеется только deployment draft.

В содержательном отношении router выбирает небольшую часть полезных случаев и не демонстрирует монотонной связи «выше confidence → больше польза reader». Положительного F1-сигнала недостаточно для принятия архитектуры с наблюдаемой потерей evidence retrieval.

Источники: [final/development_selection.json](../artifacts/qa_chunk_selection_v4_4/final/development_selection.json), [asem/development_gate.json](../artifacts/qa_chunk_selection_v4_4/asem/development_gate.json), [asem/bootstrap_vs_v6.json](../artifacts/qa_chunk_selection_v4_4/asem/bootstrap_vs_v6.json).

## 2. Прогон, данные и воспроизводимость

| Параметр | Фактическое значение |
| --- | --- |
| Артефакты | artifacts/qa_chunk_selection_v4_4/ |
| Начало | 2026-09-10 13:03:04 |
| Конец | 2026-09-10 13:04:50 |
| Время по progress | 106.527 с ≈ 1 мин 46.5 с |
| Завершение | status=done; exit_code=0; smoke=false |
| Лог | logs/qa_chunk_selection_v4_4_20260910_130302_1994740.log |
| Устройство в config | auto |
| Unlocked questions | 453 |
| Документы | 240 |
| Фолды | 5; document-level; frozen split seed 42 |
| Seeds | 42, 43, 44 |
| Строки на систему | 1359 = 453 × 3 |
| Размер shortlist | exact B0 top-5 |
| INTERNAL_TEST | 78 записей по ранее зафиксированной описи; не оценивались в V4.4 |
| QA | 2850/2850 сохранённых ответов; новых forward 0 |
| Новые embeddings | 0; MiniLM не загружался |
| Semantic available | 1359/1359; valid answers: 5 в 1356 строках, 4 в 3 строках |
| Frozen encoder | sentence-transformers/all-MiniLM-L6-v2; model_hash=3b94875081c76da0 |
| Нормализация | v4.3-nfc-ws-v1 |
| V6 weights | исходные per-run веса V4.1; не ретюнились |

Время приведено в формате локального лога, без отдельного timezone-offset в самом файле. `device=cuda:0` внутри сохранённого encoder spec относится к происхождению embeddings V4.3: это не свидетельство GPU-инференса в V4.4.

Источники: [config_resolved.json](../artifacts/qa_chunk_selection_v4_4/config_resolved.json), [progress.jsonl](../artifacts/qa_chunk_selection_v4_4/progress.jsonl), [status.json](../artifacts/qa_chunk_selection_v4_4/status.json), [cache/reuse.json](../artifacts/qa_chunk_selection_v4_4/cache/reuse.json), [confidence/features.jsonl](../artifacts/qa_chunk_selection_v4_4/confidence/features.jsonl), [analysis/efficiency.json](../artifacts/qa_chunk_selection_v4_4/analysis/efficiency.json).

### 2.1. Независимая сверка сохранённых артефактов при подготовке отчёта

| Проверка | Результат |
| --- | --- |
| Исторический V4.1/V4.3 V6 identity | ok; 1359 строк |
| Пересчитанный V6 vs reference | ok; 1359 строк |
| Строки/ключи/seed всех 5 систем | 1359 уникальных (question_id, seed) на систему; одинаковое покрытие |
| Средние основных метрик | 35 независимых сверок по predictions; совпадают |
| Q70: exact source keys, availability и значение | 10 порогов; все совпадают |
| Калибровочные строки суммарно по 10 порогам | 9480; повторное использование между runs ожидаемо |
| Попадание EVAL в соответствующий threshold source | 0 |
| Routing decisions и exact component copy | 2718/2718 совпадают |
| B0 top-5 и Hit@5/Hit@10/HN@5 | 6795/6795 строк совпадают |
| SHA256 зафиксированных источников | 2468/2468 совпадают на момент анализа |

Хэши включают код и использованные cache/source файлы; эта сверка не является полным аудитом всех когда-либо созданных файлов репозитория. Исходный snapshot физически смешивает splits: код фильтрует unlocked question IDs до создания InstanceRecord. Поэтому корректно говорить об отсутствии использования locked records в ранжировании/метриках, а не об отсутствии чтения любых байтов общего файла.

Источники: [regression/historical_v6_identity.json](../artifacts/qa_chunk_selection_v4_4/regression/historical_v6_identity.json), [regression/v6_per_run_identity.json](../artifacts/qa_chunk_selection_v4_4/regression/v6_per_run_identity.json), [regression/top5_identity.json](../artifacts/qa_chunk_selection_v4_4/regression/top5_identity.json), [source_hashes.json](../artifacts/qa_chunk_selection_v4_4/source_hashes.json).

## 3. Что именно проверялось

V6 использует B0 relevance, agreement ответов reader и поддержку source clusters с историческими per-run весами V4.1. Остальные системы переупорядочивают тот же B0 top-5.

- **SEM-FIX:** `0.5 × Majority + 0.25 × SemRel`.
- **LEX-FIX:** `0.25 × Majority + 2.0 × (mean_token_f1 / 100)`.
- **SemRel:** z-нормированный SemTop2, то есть среднее двух максимальных cosine-сходств с другими valid answers; self-similarity исключена.
- **ConsensusConf:** cohesion лучшей тройки ответов минус средняя similarity этой тройки с остальными valid answers.
- **Availability:** минимум 4 valid answers и выбранный reader-кандидат принадлежит лучшей тройке.
- **ASEM/ALEX:** соответствующий fixed reader, если availability=true и confidence ≥ Q70, иначе точный V6 fallback.
- **Q70:** отдельно для каждой архитектуры и outer run, только available TRAIN+CAL строки, pooled seeds 42/43/44; минимум 100 строк; без gold, F1 и aspect labels.
- **Tie-break:** исходный B0 rank; это не добавочный B0 score в reader-only формуле.

Ничего из K10, attention, U80, hidden-state, encoder search или aspect-specific routing в этот прогон не входило.

Контракт разрешает описательный ALEX при H1 fail для выполнения требования обязательных outputs, но запрещает H2 selection. Для superiority H2 допустимый HN заранее трактуется как `HN_ALEX ≤ HN_V6 + 0.01`. В данном прогоне эта ветка не активировалась.

Источники: [contract.json](../artifacts/qa_chunk_selection_v4_4/contract.json), [fixed_readers/sem_fix.json](../artifacts/qa_chunk_selection_v4_4/fixed_readers/sem_fix.json), [fixed_readers/lex_fix.json](../artifacts/qa_chunk_selection_v4_4/fixed_readers/lex_fix.json).

## 4. Общая таблица результатов

| Система | Роль | Token F1 | ΔF1 vs V6¹ | EM | Hit@1 | HN@1 | Amb@1 | GSRN@1 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0-live | исторический baseline | 27.3584 | -7.2665 | 20.75% | 36.72% | 4.86% | 50.85% | 33.92% | 0.5418 |
| V6-K5 | reference / выбран | 34.6249 | +0.0000 | 25.31% | 42.90% | 5.45% | 42.90% | 26.64% | 0.5704 |
| SEM-FIX | always-control | 37.5108 | +2.8859 | 31.49% | 42.83% | 6.40% | 40.54% | 25.39% | 0.5801 |
| LEX-FIX | always-control | 36.9816 | +2.3568 | 30.10% | 43.49% | 5.52% | 40.40% | 24.28% | 0.5794 |
| ASEM | primary; gate fail | 35.1112 | +0.4864 | 25.97% | 41.87% | 5.81% | 43.27% | 26.64% | 0.5650 |
| ALEX | описательно; H2 skipped | 34.7743 | +0.1494 | 25.90% | 41.94% | 5.89% | 43.12% | 26.27% | 0.5652 |

¹ Простая разность средних в таблице, не новый statistical test. F1 — шкала 0–100; доли показаны в процентах. GSRN — primary class `gold_supporting_robust_negative`, частично пересекающийся с исходными labels; GSRN и Amb нельзя складывать как непересекающиеся категории. У всех систем Hit@5=79.0287%, Hit@10=92.2001%, HN@5=6.9021%: кандидаты не выходят за исходный shortlist.

Источники: [analysis/metrics.json](../artifacts/qa_chunk_selection_v4_4/analysis/metrics.json), [reference/b0_live/metrics.json](../artifacts/qa_chunk_selection_v4_4/reference/b0_live/metrics.json).

### 4.1. Что означает более высокий результат always-controls

SEM-FIX даёт максимальный F1 **37.5108**, LEX-FIX — **36.9816**. SEM-FIX выше LEX-FIX на **0.5291**, но его HN@1 хуже (6.4018% против 5.5188%), Hit@1 ниже (42.8256% против 43.4879%). Это trade-off, а не доказанное превосходство semantic scoring.

Относительно своих always-controls ASEM теряет **2.3995 F1**, ALEX — **2.2073 F1**. Router сохраняет лишь 16.85% и 6.34% соответственно от точечного прироста always-control над V6. Это описательное отношение net gains, не routing recall.

Контракт не позволяет заменить primary winner always-control после просмотра этих чисел. Для controls в артефактах V4.4 нет отдельных paired CI; в этом отчёте они не добавлялись.

### 4.2. Сопоставление с V4.3

| V4.3 reference | F1 V4.3 | V4.4 система | F1 V4.4 | Точечная разность |
| --- | --- | --- | --- | --- |
| LEXCORE V4.3 | 36.5827 | LEX-FIX | 36.9816 | +0.3989 |
| SEMCORE V4.3 | 37.3337 | SEM-FIX | 37.5108 | +0.1770 |
| ADAPT1 V4.3 | 36.0153 | ASEM | 35.1112 | -0.9040 |

Fixed readers не идентичны CAL-tuned LEXCORE/SEMCORE V4.3. ADAPT1 использовал смесь выбранных thresholds, тогда как ASEM фиксирует operating point Q70 и строит его на non-EVAL availability distribution. Более слабый ASEM по сравнению с ADAPT1 не означает нарушения reference identity: сравниваются разные routing policies на том же development-пуле. Положительный exploratory ADAPT1 не переносится автоматически на H1 V4.4.

## 5. H1: почему итог отрицательный при положительном F1

| Условие | Требование | Факт | Результат |
| --- | --- | --- | --- |
| ΔToken F1 | > 0 | 0.486305 | pass |
| 95% development CI | нижняя граница > 0 | [0.005367; 0.992832] | pass |
| HN@1 | ≤ 6.45% | 5.81% (Δ +0.3679 п.п.) | pass |
| Hit@1 | ≥ 42.40% | 41.87% (Δ -1.0302 п.п.) | FAIL |

V6 выбирает positive в **583/1359** строках, ASEM — **569/1359**: net **−14**. Для соблюдения допуска нужно минимум **577** positive-строк; ASEM не хватает **8**. Относительно непрерывной границы 42.3992% недобор составляет 0.5302 п.п. Это не погрешность округления.

ASEM выигрывает F1 в 32 строках, проигрывает в 14, имеет 1313 ничьих. 32 выигрыша относятся к 18 уникальным вопросам/документам; 14 проигрышей — к 8. Строки разных seed нельзя считать независимыми случаями.

Точная разность средних F1 равна **+0.486369979**, а `mean_delta` в bootstrap-файле — **+0.486304989**. Общая функция `clustered_bootstrap_ci` возвращает среднее bootstrap-реплик, поэтому небольшое расхождение ожидаемо. CI строится после усреднения seed-дельт внутри question, с ресэмплированием 240 document clusters (5000 реплик, seed=42).

Нижняя граница CI **+0.005367** очень близка к нулю. По фиксированному gate она положительна, но такой результат нельзя описывать как сильное независимое подтверждение: архитектура и Q70 исторически выбраны на повторно использованном пуле. Gate уже не пройден по Hit@1; перестройка CI или порога ради смены решения не проводилась.

Источники: [asem/bootstrap_vs_v6.json](../artifacts/qa_chunk_selection_v4_4/asem/bootstrap_vs_v6.json), [final/development_selection.json](../artifacts/qa_chunk_selection_v4_4/final/development_selection.json).

## 6. H2, local support и ASOFT

ALEX даёт **34.7743 F1**, то есть точечно **+0.1494** к V6 и **-0.3370** к ASEM. Hit@1 ниже V6 на 0.9566 п.п.; HN@1 выше на 0.4415 п.п. ALEX чуть лучше ASEM по Hit@1, но чуть хуже по HN@1.

**Это не проверка H2 superiority/non-inferiority.** `alex/bootstrap_vs_asem.json` содержит skipped, потому что H1 fail. Отсутствие H2 нельзя трактовать как доказанную non-inferiority или статистическое поражение ALEX.

`support_diagnostic/status.json` фиксирует `skipped=true, reason=H1 failed`. Нет результатов Q20, SupportGain/SupportRel, PREF-SUPPORT, veto-rate или good/bad-veto. Само наличие пустой директории embeddings не означает выполненный support. ASOFT не был запрошен (`soft=false`) и тоже не представлен. Выводов о качестве local evidence по этому прогону сделать нельзя.

Источники: [alex/bootstrap_vs_asem.json](../artifacts/qa_chunk_selection_v4_4/alex/bootstrap_vs_asem.json), [support_diagnostic/status.json](../artifacts/qa_chunk_selection_v4_4/support_diagnostic/status.json), [config_resolved.json](../artifacts/qa_chunk_selection_v4_4/config_resolved.json).

## 7. Q70, availability и фактические переключения

| Run | ASEM n source | ASEM Q70 | ALEX n source | ALEX Q70 | EVAL n |
| --- | --- | --- | --- | --- | --- |
| 0 | 978 | 0.387714536 | 917 | 0.407218948 | 270 |
| 1 | 977 | 0.388615248 | 913 | 0.411193966 | 273 |
| 2 | 977 | 0.387836247 | 913 | 0.411342492 | 270 |
| 3 | 985 | 0.384714123 | 920 | 0.411169211 | 273 |
| 4 | 979 | 0.384137065 | 921 | 0.397252357 | 273 |

Все 10 router-конфигураций включены: ни одна не отключилась по MIN_ROUTING_CAL_ROWS. Q70 ASEM находится в узком диапазоне 0.384137–0.388615; ALEX — 0.397252–0.411342. В этом replay численное значение квантили стабильно; это не доказывает качество самой confidence-функции. Availability различается между архитектурами, поскольку выбранные reader-кандидаты могут по-разному попадать в лучшую тройку.

| Показатель | ASEM | ALEX |
| --- | --- | --- |
| Routing available | 1224/1359 (90.07%) | 1146/1359 (84.33%) |
| Reader branch используется | 372/1359 (27.37%) | 348/1359 (25.61%) |
| Fallback | 987/1359 (72.63%) | 1011/1359 (74.39%) |
| Реально меняется top1 | 106/1359 (7.80%) | 73/1359 (5.37%) |
| Reader branch, но top1 совпал с V6 | 266/372 | 275/348 |
| Разногласия V6/reader | 579 | 463 |
| Выбрано reader среди разногласий | 18.31% | 15.77% |
| Routing precision | 30.19% | 24.66% |
| Routing recall | 19.05% | 15.38% |

`reader_only_used_rate` — частота выбора ветки, **не** частота изменения ответа/чанка. У ASEM 71.51% включений reader-ветки сохраняют тот же top1, у ALEX — 79.02%. Поэтому значения около 26–27% не означают столь же широкое воздействие router на результат.

Precision определён как good switches / switches на disagreement; F1-ties входят в знаменатель. Если условно исключить ties, доля выигрышей составляет 32/46=69.57% у ASEM и 18/31=58.06% у ALEX, но это другой показатель, не протокольный routing precision.

Источники: [confidence/run0/tau70.json](../artifacts/qa_chunk_selection_v4_4/confidence/run0/tau70.json), [confidence/run1/tau70.json](../artifacts/qa_chunk_selection_v4_4/confidence/run1/tau70.json), [confidence/run2/tau70.json](../artifacts/qa_chunk_selection_v4_4/confidence/run2/tau70.json), [confidence/run3/tau70.json](../artifacts/qa_chunk_selection_v4_4/confidence/run3/tau70.json), [confidence/run4/tau70.json](../artifacts/qa_chunk_selection_v4_4/confidence/run4/tau70.json), [analysis/routing_rate.json](../artifacts/qa_chunk_selection_v4_4/analysis/routing_rate.json), [analysis/routing_precision_recall.json](../artifacts/qa_chunk_selection_v4_4/analysis/routing_precision_recall.json).

### 7.1. Все routing-rate разрезы

#### Outer run

| Outer run | n | ASEM available | ASEM used | ASEM fallback | ALEX available | ALEX used | ALEX fallback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 270 | 91.11% | 25.56% | 74.44% | 84.81% | 26.67% | 73.33% |
| 1 | 273 | 90.48% | 24.91% | 75.09% | 85.35% | 23.44% | 76.56% |
| 2 | 270 | 91.48% | 25.56% | 74.44% | 86.30% | 23.33% | 76.67% |
| 3 | 273 | 87.55% | 29.30% | 70.70% | 82.78% | 22.71% | 77.29% |
| 4 | 273 | 89.74% | 31.50% | 68.50% | 82.42% | 31.87% | 68.13% |

#### Seed

| Seed | n | ASEM available | ASEM used | ASEM fallback | ALEX available | ALEX used | ALEX fallback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 453 | 88.30% | 27.59% | 72.41% | 83.89% | 25.83% | 74.17% |
| 43 | 453 | 90.51% | 27.59% | 72.41% | 83.66% | 25.83% | 74.17% |
| 44 | 453 | 91.39% | 26.93% | 73.07% | 85.43% | 25.17% | 74.83% |

#### Aspect

| Aspect | n | ASEM available | ASEM used | ASEM fallback | ALEX available | ALEX used | ALEX fallback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| material | 378 | 85.98% | 33.86% | 66.14% | 83.86% | 34.92% | 65.08% |
| method | 345 | 92.17% | 26.09% | 73.91% | 80.87% | 22.32% | 77.68% |
| task | 441 | 90.02% | 24.94% | 75.06% | 88.89% | 21.77% | 78.23% |
| metric | 195 | 94.36% | 22.56% | 77.44% | 81.03% | 22.05% | 77.95% |

### 7.2. Выигрыши, потери и упущенные возможности

| Система | Категория | Строк | Уникальных вопросов | Уникальных документов | Сумма ΔF1 reader−V6 |
| --- | --- | --- | --- | --- | --- |
| ASEM | good_switch | 32 | 18 | 18 | +1207.6435 |
| ASEM | bad_switch | 14 | 8 | 8 | -546.6667 |
| ASEM | unnecessary_switch | 60 | 30 | 30 | +0.0000 |
| ASEM | missed_opportunity | 136 | 85 | 77 | +7658.5220 |
| ALEX | good_switch | 18 | 12 | 12 | +756.3736 |
| ALEX | bad_switch | 13 | 7 | 7 | -553.3333 |
| ALEX | unnecessary_switch | 42 | 19 | 19 | +0.0000 |
| ALEX | missed_opportunity | 99 | 59 | 54 | +6479.8963 |

ASEM: 168 reader-better случаев среди disagreement, из них выбраны только 32; пропущены **136** (106 из-за confidence ниже Q70, 30 из-за недоступности routing). ALEX: 117 reader-better, выбраны 18; пропущены **99** (70 ниже порога, 29 unavailable).

Сумма реализованных выигрышей/потерь ASEM: +1207.6435 − 546.6667 = +660.9768 F1-пунктов по строкам; деление на 1359 даёт точный общий gain +0.486370. ALEX: +756.3736 − 553.3333 = +203.0403, или +0.149404 на строку. Большая сумма улучшений в missed opportunities — диагностический потенциал с использованием gold, **не достижимая оценка обученного router** и не основание post-hoc снижать Q70.

## 8. Confidence: механистическая проверка

Границы квинтилей: 0.168198079, 0.253968721, 0.341705558, 0.448378645. Границы общие, поскольку ConsensusConf вычисляется по одному shortlist; reader-dependent availability в распределение этих post-EVAL bins не фильтруется. Равные confidence остаются в одном bin. Вероятности и mean ΔF1 ниже относятся **только к disagreements внутри bin**, а `n` — ко всем строкам bin.

### ASEM

| Квинтиль | n | Disagreement | P(reader better) | P(V6 better) | Mean ΔF1 reader−V6 |
| --- | --- | --- | --- | --- | --- |
| 0-20% | 272 | 123 | 22.76% | 17.89% | +6.6786 |
| 20-40% | 272 | 150 | 35.33% | 10.67% | +14.1275 |
| 40-60% | 269 | 118 | 20.34% | 29.66% | -3.5476 |
| 60-80% | 274 | 100 | 37.00% | 22.00% | +9.2817 |
| 80-100% | 272 | 88 | 29.55% | 15.91% | +5.3610 |

### ALEX

| Квинтиль | n | Disagreement | P(reader better) | P(V6 better) | Mean ΔF1 reader−V6 |
| --- | --- | --- | --- | --- | --- |
| 0-20% | 272 | 107 | 22.43% | 14.02% | +12.5459 |
| 20-40% | 272 | 120 | 33.33% | 11.67% | +14.6794 |
| 40-60% | 269 | 100 | 21.00% | 36.00% | -2.7979 |
| 60-80% | 274 | 74 | 21.62% | 24.32% | +1.4082 |
| 80-100% | 272 | 62 | 25.81% | 16.13% | +4.4269 |

Обе `monotone_reader_win_rate=false`. У ASEM reader win-rate меняется 22.76% → 35.33% → 20.34% → 37.00% → 29.55%; у ALEX 22.43% → 33.33% → 21.00% → 21.62% → 25.81%. Средние gains в 20–40% confidence выше, чем в самом высоком квинтиле; 40–60% даёт отрицательный mean gain у обоих readers.

Это свидетельствует о механистической слабости данной hard-threshold эвристики на development data. Это не доказывает отсутствие любой связи или бесполезность всех consensus-признаков: монотонность здесь — описательная проверка пяти bins без отдельного теста тренда. Ни положение threshold, ни состав признаков после просмотра этих bins не менялись.

Источники: [analysis/confidence_monotonicity.json](../artifacts/qa_chunk_selection_v4_4/analysis/confidence_monotonicity.json).

## 9. Аспекты: F1 и сохранение evidence retrieval

| Aspect | n | V6 F1 | SEM-FIX F1 | LEX-FIX F1 | ASEM F1 | ΔASEM | ALEX F1 | ΔALEX |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| material | 378 | 34.4697 | 38.6367 | 37.1690 | 34.8974 | +0.4277 | 34.4300 | -0.0397 |
| method | 345 | 26.3640 | 31.6155 | 32.7647 | 26.5572 | +0.1932 | 26.2673 | -0.0966 |
| metric | 195 | 24.2406 | 24.0269 | 24.6480 | 24.5703 | +0.3297 | 24.5703 | +0.3297 |
| task | 441 | 45.8121 | 47.1199 | 45.5736 | 46.6474 | +0.8353 | 46.2364 | +0.4242 |

ASEM имеет положительный точечный F1 gain во всех четырёх аспектах, максимальный — task (+0.8353). Но Hit@1 падает на material (−1.5873 п.п.), metric (−3.0769 п.п.) и task (−0.4535 п.п.); method не меняется. В абсолютных positive-строках это −6 / 0 / −6 / −2. Рост HN главным образом связан с task (+5 строк) и metric (+2), частично компенсируется material/method (по −1).

**Ключевой вопрос H2 про method gain и task fallback не получил положительного описательного ответа.** У LEX-FIX method gain +6.4007 F1, но у ALEX method −0.0966 относительно V6: большая часть сильной стороны lexical always-control не реализована. Task у LEX-FIX −0.2385, у ALEX +0.4242 — fallback помогает этому аспекту точечно, но не сохраняет method gain. Это аспектная диагностика, а не отдельное доказательство H2.

| Aspect | Система | EM | Hit@1 | HN@1 | Amb@1 | GSRN@1 |
| --- | --- | --- | --- | --- | --- | --- |
| material | V6-K5 | 35.19% | 54.50% | 3.97% | 35.71% | 24.07% |
| material | SEM-FIX | 43.39% | 52.12% | 5.03% | 36.77% | 26.98% |
| material | LEX-FIX | 42.33% | 55.03% | 3.97% | 34.92% | 22.49% |
| material | ASEM | 35.98% | 52.91% | 3.70% | 38.36% | 26.19% |
| material | ALEX | 36.51% | 53.97% | 3.97% | 37.04% | 24.60% |
| method | V6-K5 | 23.77% | 34.49% | 4.93% | 51.01% | 33.91% |
| method | SEM-FIX | 34.49% | 42.61% | 4.64% | 41.74% | 29.28% |
| method | LEX-FIX | 32.46% | 43.77% | 4.64% | 40.58% | 28.41% |
| method | ASEM | 25.80% | 34.49% | 4.64% | 51.30% | 33.91% |
| method | ALEX | 24.93% | 34.20% | 4.93% | 51.30% | 33.91% |
| metric | V6-K5 | 14.36% | 37.44% | 5.64% | 40.00% | 20.00% |
| metric | SEM-FIX | 13.85% | 32.31% | 6.15% | 44.10% | 19.49% |
| metric | LEX-FIX | 13.85% | 31.28% | 4.62% | 42.56% | 18.97% |
| metric | ASEM | 12.82% | 34.36% | 6.67% | 41.54% | 20.00% |
| metric | ALEX | 12.82% | 34.87% | 6.67% | 41.03% | 20.00% |
| task | V6-K5 | 22.90% | 41.95% | 7.03% | 43.99% | 26.08% |
| task | SEM-FIX | 26.76% | 39.68% | 9.07% | 41.27% | 23.58% |
| task | LEX-FIX | 24.94% | 38.78% | 7.94% | 43.99% | 24.94% |
| task | ASEM | 23.36% | 41.50% | 8.16% | 41.95% | 24.26% |
| task | ALEX | 23.36% | 40.82% | 7.94% | 42.86% | 24.49% |

Источники: [analysis/aspect_metrics.json](../artifacts/qa_chunk_selection_v4_4/analysis/aspect_metrics.json).

## 10. Устойчивость по seed и outer run

| Seed | Система | n | F1 | ΔF1 vs V6 того же seed | EM | Hit@1 | HN@1 | Amb@1 | GSRN@1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | V6-K5 | 453 | 33.1222 | +0.0000 | 24.28% | 41.50% | 5.96% | 43.71% | 26.93% |
| 42 | SEM-FIX | 453 | 37.1776 | +4.0553 | 30.24% | 42.83% | 6.84% | 40.40% | 25.83% |
| 42 | LEX-FIX | 453 | 35.7237 | +2.6014 | 28.70% | 42.60% | 5.08% | 41.50% | 24.28% |
| 42 | ASEM | 453 | 33.2861 | +0.1638 | 24.28% | 40.18% | 6.18% | 44.37% | 27.37% |
| 42 | ALEX | 453 | 33.0576 | -0.0646 | 24.50% | 40.40% | 6.18% | 44.15% | 26.71% |
| 43 | V6-K5 | 453 | 34.8059 | +0.0000 | 24.94% | 43.05% | 5.96% | 41.50% | 25.83% |
| 43 | SEM-FIX | 453 | 37.6512 | +2.8453 | 31.35% | 41.72% | 6.62% | 41.06% | 24.94% |
| 43 | LEX-FIX | 453 | 37.5491 | +2.7432 | 30.02% | 43.49% | 6.40% | 38.85% | 23.40% |
| 43 | ASEM | 453 | 35.6686 | +0.8627 | 26.27% | 42.16% | 6.40% | 41.72% | 25.39% |
| 43 | ALEX | 453 | 35.0456 | +0.2397 | 25.83% | 41.72% | 6.62% | 41.94% | 25.39% |
| 44 | V6-K5 | 453 | 35.9465 | +0.0000 | 26.71% | 44.15% | 4.42% | 43.49% | 27.15% |
| 44 | SEM-FIX | 453 | 37.7035 | +1.7571 | 32.89% | 43.93% | 5.74% | 40.18% | 25.39% |
| 44 | LEX-FIX | 453 | 37.6721 | +1.7257 | 31.57% | 44.37% | 5.08% | 40.84% | 25.17% |
| 44 | ASEM | 453 | 36.3791 | +0.4326 | 27.37% | 43.27% | 4.86% | 43.71% | 27.15% |
| 44 | ALEX | 453 | 36.2197 | +0.2732 | 27.37% | 43.71% | 4.86% | 43.27% | 26.71% |

ASEM F1 gain положителен на всех трёх seed: +0.1638 / +0.8627 / +0.4326. Одновременно Hit@1 хуже на всех трёх: −1.3245 / −0.8830 / −0.8830 п.п. ALEX F1 ниже V6 на seed42 и выше на 43/44. Seed не выбирался по качеству; в draft заранее зафиксирован canonical seed42.

| Run | Система | n | F1 системы | ΔF1 vs V6 | Δ positive-строк | Δ HN-строк |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | ASEM | 270 | 38.1934 | +1.8766 | 4 | 0 |
| 0 | ALEX | 270 | 37.4197 | +1.1030 | 1 | 0 |
| 1 | ASEM | 273 | 29.8468 | -0.1282 | -4 | 0 |
| 1 | ALEX | 273 | 29.7526 | -0.2224 | -3 | 0 |
| 2 | ASEM | 270 | 39.1945 | +0.7037 | 1 | -1 |
| 2 | ALEX | 270 | 38.7871 | +0.2963 | 0 | 0 |
| 3 | ASEM | 273 | 33.8636 | +0.2930 | -5 | 1 |
| 3 | ALEX | 273 | 33.4973 | -0.0733 | -1 | 1 |
| 4 | ASEM | 273 | 34.5367 | -0.2957 | -10 | 5 |
| 4 | ALEX | 273 | 34.4878 | -0.3445 | -10 | 5 |

У ASEM положительный F1 gain на 3/5 runs, у ALEX на 2/5. Run0 даёт около **76.66% суммарного net F1 gain ASEM**, несмотря на 19.87% всех строк. Run4 особенно неблагоприятен: −10 positive-строк и +5 HN-строк у обеих архитектур. Поэтому «gain есть на трёх seed» не следует путать с равномерной устойчивостью по документным фолдам. Формальные дополнительные per-run gates не вводились.

Источники: [analysis/seed_metrics.json](../artifacts/qa_chunk_selection_v4_4/analysis/seed_metrics.json), [asem/run0/predictions.jsonl](../artifacts/qa_chunk_selection_v4_4/asem/run0/predictions.jsonl), [asem/run4/predictions.jsonl](../artifacts/qa_chunk_selection_v4_4/asem/run4/predictions.jsonl), [alex/run4/predictions.jsonl](../artifacts/qa_chunk_selection_v4_4/alex/run4/predictions.jsonl).

## 11. Переходы labels и расхождение F1 / Hit@1

| Переход | V6→ASEM | V6→ALEX |
| --- | --- | --- |
| GSRN->positive | 4 | 1 |
| ambiguous->ambiguous | 568 | 574 |
| ambiguous->easy_negative | 3 | 3 |
| ambiguous->hard_negative | 3 | 3 |
| ambiguous->positive | 9 | 3 |
| easy_negative->ambiguous | 3 | 3 |
| easy_negative->easy_negative | 116 | 116 |
| hard_negative->ambiguous | 1 | 0 |
| hard_negative->hard_negative | 72 | 74 |
| hard_negative->positive | 1 | 0 |
| net_positive_gain | -14 | -13 |
| positive->HN | 4 | 3 |
| positive->ambiguous | 16 | 9 |
| positive->easy_negative | 4 | 4 |
| positive->hard_negative | 4 | 3 |
| positive->positive | 559 | 567 |

`positive->HN` — дублирующее удобное имя для `positive->hard_negative`; его нельзя прибавлять второй раз. `GSRN->positive` — дополнительный срез, а не новая непересекающаяся категория.

ASEM приобретает 10 positive-строк (9 ambiguous→positive и 1 HN→positive), но теряет 24 (16→ambiguous, 4→HN, 4→easy_negative): net −14. Все **14** F1-проигрышей ASEM относятся к positive→non-positive. Ещё **10** таких переходов имеют одинаковый F1 у двух ответов, поэтому они ухудшают Hit@1, не учитываясь как F1-проигрыши.

ALEX приобретает 3 positive-строки и теряет 16: net −13. Все 13 F1-проигрышей — positive→non-positive; ещё 3 потери positive имеют F1-tie.

Пример скрытого для F1 ущерба: вопрос `2ebfc12285f5d426e0d0e8d2befa1af27f99a56e::metric`, seeds 42/43/44. ALEX переключается с `F - measure` из positive на `dilated skeleton ground truth` из easy_negative/hard_negative. Оба извлечённых ответа получают F1=0, но выбор evidence становится хуже. Другой пример: ASEM выбирает `BSD200` вместо `BSD200` из другого чанка — ответ совпал, label сменился positive→ambiguous.

Это не противоречие метрик. Hit@1 оценивает исходный label выбранного чанка; Token F1 оценивает извлечённую строку относительно gold. Даже positive-чанк может дать неудачный extractive span, а неверный чанк — похожую строку.

Источники: [analysis/transitions.json](../artifacts/qa_chunk_selection_v4_4/analysis/transitions.json), [analysis/error_buckets/asem.jsonl](../artifacts/qa_chunk_selection_v4_4/analysis/error_buckets/asem.jsonl), [analysis/error_buckets/alex.jsonl](../artifacts/qa_chunk_selection_v4_4/analysis/error_buckets/alex.jsonl).

## 12. Проверяемые примеры ошибок и исправлений

Ниже — до трёх разных вопросов на bucket и систему, выбранных детерминированно по убыванию абсолютной ΔF1 (при ties — порядок файла). Это иллюстрации механизма, не случайная репрезентативная выборка. Полные buckets доступны по ссылкам в конце раздела; интерпретация основана на сохранённых spans и метках, без новой ручной разметки статей.

### ASEM — good_switch

- **2d294bde112b892068636f3a48300b3c033d98da::task**, seed 43, run 2, task. V6 chunk 0 (ambiguous): «Challenging landmarks», F1=0.0000. Reader chunk 20 (positive): «face alignment», F1=100.0000. Confidence=0.399122, Q70=0.387836, available=True, reader used=True, reader∈T*=True.

- **128c727ac06fcc50f1735cb222a441eee6adcab6::task**, seed 44, run 0, task. V6 chunk 3 (ambiguous): «proposing TuckER, a new linear model for link prediction in knowledge graphs», F1=30.7692. Reader chunk 1 (positive): «link prediction», F1=100.0000. Confidence=0.569051, Q70=0.387715, available=True, reader used=True, reader∈T*=True.

- **36973330ae638571484e1f68aaf455e3e6f18ae9::task**, seed 43, run 0, task. V6 chunk 20 (ambiguous): «general object detection», F1=40.0000. Reader chunk 1 (positive): «pedestrian detection», F1=100.0000. Confidence=0.415903, Q70=0.387715, available=True, reader used=True, reader∈T*=True.

### ASEM — bad_switch

- **0b0cf7e00e7532e38238a9164f0a8db2574be2ea::method**, seed 44, run 4, method. V6 chunk 1 (positive): «the Transformer», F1=100.0000. Reader chunk 14 (ambiguous): «attention - based models», F1=0.0000. Confidence=0.565357, Q70=0.384137, available=True, reader used=True, reader∈T*=True.

- **0a49b4de21363d86599d4a058aaf4f5aed019495::task**, seed 42, run 4, task. V6 chunk 10 (positive): «machine translation», F1=100.0000. Reader chunk 11 (ambiguous): «Character - Level Translation», F1=40.0000. Confidence=0.403634, Q70=0.384137, available=True, reader used=True, reader∈T*=True.

- **3a28fe49e7a856ddd60d134696a891ed7bca5962::task**, seed 42, run 4, task. V6 chunk 0 (positive): «Pedestrian detection», F1=100.0000. Reader chunk 6 (easy_negative): «object detection», F1=50.0000. Confidence=0.494541, Q70=0.384137, available=True, reader used=True, reader∈T*=True.

### ASEM — unnecessary_switch

- **0cb8f50580cc69191144bd503e268451ce966fa6::method**, seed 44, run 0, method. V6 chunk 3 (ambiguous): «novel variations of models in the MPNN family», F1=0.0000. Reader chunk 11 (ambiguous): «MPNN», F1=0.0000. Confidence=0.477859, Q70=0.387715, available=True, reader used=True, reader∈T*=True.

- **1e7a36c4d4f96b29e3edf51b6eb61f8e16217704::task**, seed 43, run 0, task. V6 chunk 17 (positive): «word - level language modelling», F1=66.6667. Reader chunk 0 (positive): «character level language modelling», F1=66.6667. Confidence=0.493238, Q70=0.387715, available=True, reader used=True, reader∈T*=True.

- **3c1d781f2dab8da12e3cb0e4d7abfb440a340a09::task**, seed 42, run 0, task. V6 chunk 9 (ambiguous): «NLI task», F1=0.0000. Reader chunk 2 (easy_negative): «NLI», F1=0.0000. Confidence=0.480380, Q70=0.387715, available=True, reader used=True, reader∈T*=True.

### ASEM — missed_opportunity

- **07cca2bdd0dc2fee02889e17789748eba9d06ffa::material**, seed 44, run 0, material. V6 chunk 12 (ambiguous): «training dataset used for testing consists of 2 - weeks long multi - modal tracks», F1=0.0000. Reader chunk 36 (positive): «GPS», F1=100.0000. Confidence=0.049170, Q70=0.387715, available=True, reader used=False, reader∈T*=True.

- **128c727ac06fcc50f1735cb222a441eee6adcab6::method**, seed 44, run 0, method. V6 chunk 13 (ambiguous): «Bilinear models», F1=0.0000. Reader chunk 3 (positive): «TuckER», F1=100.0000. Confidence=0.343964, Q70=0.387715, available=True, reader used=False, reader∈T*=True.

- **2742a33946e20dd33140b8d6e80d5fd04fced1b2::method**, seed 44, run 1, method. V6 chunk 3 (ambiguous): «geometric descriptors», F1=0.0000. Reader chunk 19 (positive): «3DMatch», F1=100.0000. Confidence=0.205612, Q70=0.388615, available=True, reader used=False, reader∈T*=True.

### ALEX — good_switch

- **0a49b4de21363d86599d4a058aaf4f5aed019495::task**, seed 44, run 4, task. V6 chunk 9 (ambiguous): «data sparsity», F1=0.0000. Reader chunk 10 (positive): «machine translation», F1=100.0000. Confidence=0.448028, Q70=0.397252, available=True, reader used=True, reader∈T*=True.

- **128c727ac06fcc50f1735cb222a441eee6adcab6::task**, seed 44, run 0, task. V6 chunk 3 (ambiguous): «proposing TuckER, a new linear model for link prediction in knowledge graphs», F1=30.7692. Reader chunk 1 (positive): «link prediction», F1=100.0000. Confidence=0.569051, Q70=0.407219, available=True, reader used=True, reader∈T*=True.

- **3a7895b17db0cda7bbf86bcda52c46a3e03b6ded::material**, seed 43, run 2, material. V6 chunk 14 (positive): «IEMOCAP dataset. For AVEC», F1=40.0000. Reader chunk 12 (positive): «IEMOCAP», F1=100.0000. Confidence=0.470498, Q70=0.411342, available=True, reader used=True, reader∈T*=True.

### ALEX — bad_switch

- **0b0cf7e00e7532e38238a9164f0a8db2574be2ea::method**, seed 44, run 4, method. V6 chunk 1 (positive): «the Transformer», F1=100.0000. Reader chunk 14 (ambiguous): «attention - based models», F1=0.0000. Confidence=0.565357, Q70=0.397252, available=True, reader used=True, reader∈T*=True.

- **3600aac8edc5bc015e69f2ffa893c21b6d4e1057::material**, seed 43, run 4, material. V6 chunk 10 (positive): «MAFA», F1=100.0000. Reader chunk 7 (ambiguous): «WiderFace», F1=0.0000. Confidence=0.411342, Q70=0.397252, available=True, reader used=True, reader∈T*=True.

- **3a28fe49e7a856ddd60d134696a891ed7bca5962::task**, seed 42, run 4, task. V6 chunk 0 (positive): «Pedestrian detection», F1=100.0000. Reader chunk 6 (easy_negative): «object detection», F1=50.0000. Confidence=0.494541, Q70=0.397252, available=True, reader used=True, reader∈T*=True.

### ALEX — unnecessary_switch

- **0cb8f50580cc69191144bd503e268451ce966fa6::method**, seed 44, run 0, method. V6 chunk 3 (ambiguous): «novel variations of models in the MPNN family», F1=0.0000. Reader chunk 11 (ambiguous): «MPNN», F1=0.0000. Confidence=0.477859, Q70=0.407219, available=True, reader used=True, reader∈T*=True.

- **3c1d781f2dab8da12e3cb0e4d7abfb440a340a09::task**, seed 42, run 0, task. V6 chunk 9 (ambiguous): «NLI task», F1=0.0000. Reader chunk 2 (easy_negative): «NLI», F1=0.0000. Confidence=0.480380, Q70=0.407219, available=True, reader used=True, reader∈T*=True.

- **0012de6bec1f25599e4f02517637e531a71909b9::material**, seed 42, run 1, material. V6 chunk 0 (easy_negative): «MRI volumes depicting prostate», F1=0.0000. Reader chunk 8 (ambiguous): «MRI», F1=0.0000. Confidence=0.602215, Q70=0.411194, available=True, reader used=True, reader∈T*=True.

### ALEX — missed_opportunity

- **07cca2bdd0dc2fee02889e17789748eba9d06ffa::material**, seed 42, run 0, material. V6 chunk 12 (ambiguous): «training dataset used for testing consists of 2 - weeks long multi - modal tracks», F1=0.0000. Reader chunk 36 (positive): «GPS», F1=100.0000. Confidence=0.042726, Q70=0.407219, available=True, reader used=False, reader∈T*=True.

- **128c727ac06fcc50f1735cb222a441eee6adcab6::method**, seed 44, run 0, method. V6 chunk 13 (ambiguous): «Bilinear models», F1=0.0000. Reader chunk 3 (positive): «TuckER», F1=100.0000. Confidence=0.343964, Q70=0.407219, available=True, reader used=False, reader∈T*=True.

- **15ca7adccf5cd4dc309cdcaa6328f4c429ead337::material**, seed 42, run 0, material. V6 chunk 11 (ambiguous): «3D object classification», F1=0.0000. Reader chunk 10 (positive): «ModelNet40», F1=100.0000. Confidence=0.248446, Q70=0.407219, available=True, reader used=False, reader∈T*=True.

Характерные механизмы: сокращение слишком длинного ответа до точного названия может улучшать F1; переход от `the Transformer` к общему `attention - based models` ухудшает его даже при высокой confidence. Полезные названия методов (`TuckER`, `3DMatch`) могут оставаться ниже Q70. Confidence характеризует внутреннюю согласованность ответов, но не гарантирует соответствие целевому evidence.

Источники: [analysis/error_buckets/asem.jsonl](../artifacts/qa_chunk_selection_v4_4/analysis/error_buckets/asem.jsonl), [analysis/error_buckets/alex.jsonl](../artifacts/qa_chunk_selection_v4_4/analysis/error_buckets/alex.jsonl).

## 13. Дополнительные метрики и ограничения их трактовки

| Система | Attribution-aware F1 | Δ vs V6 | F1 при selected positive | Gold match из non-positive | Tie-rate reader |
| --- | --- | --- | --- | --- | --- |
| V6-K5 | 33.3097 | +0.0000 | 65.2235 | 7.28% | — |
| SEM-FIX | 35.6148 | +2.3051 | 72.5617 | 8.17% | 37.01% |
| LEX-FIX | 35.1527 | +1.8430 | 70.0383 | 8.09% | 43.19% |
| ASEM | 33.4306 | +0.1209 | 67.4530 | 7.95% | — |
| ALEX | 33.1427 | -0.1670 | 66.4984 | 7.80% | — |

Attribution-aware F1 у ASEM повышается лишь на **0.1209**, тогда как обычный F1 — на 0.4864. У ALEX attribution-aware F1 снижается на **0.1670** при небольшом положительном обычном F1. Это дополнительный аргумент против сильной интерпретации ответа-only gain, но не отдельный gate.

`attribution_aware_token_f1` наследуется из V3: F1 обнуляется для known_non_target вне positive-v3 и для target_mismatch, однако покрытие target-status неполное. Не следует трактовать эту метрику как полностью размеченную проверку атрибуции. `gold_match_from_non_original_positive` означает EM=1 **или** Token F1≥50 из non-positive чанка, а не исключительно exact match.

`pairwise_pv3_gt_tm=NaN`, `target_mismatch@1=null` — отсутствие определённой оценки при неполном overlay, не численная ошибка runtime confidence. Нулевые target-mismatch lower bounds не означают доказанное отсутствие target mismatch. Raw `NaN` в JSON не соответствует строгому JSON-стандарту; Python его читает, строгим внешним парсерам потребуется обработка. Первичные F1/Hit/HN и значения confidence конечны.

Pairwise clean-HN B0-live около 0.7514, live V6 около 0.4158: B0 несёт исторический full-score diagnostic, а V4.4 оценивает isolated ranking через унаследованные процедуры и флаги. Такой разрыв не доказывает деградацию reader и не является primary endpoint; корректнее сравнивать основные Hit/F1 и предусмотренные safeguards. Все точные вспомогательные поля сохранены в приложении A.

SEM-FIX имеет top-score tie в **503/1359 (37.01%)**, LEX-FIX — **587/1359 (43.19%)**. B0 tie-break существенен для поведения fixed readers; reader-only не означает полной независимости от порядка B0.

Источники: [analysis/metrics.json](../artifacts/qa_chunk_selection_v4_4/analysis/metrics.json), [fixed_readers/tie_rate.json](../artifacts/qa_chunk_selection_v4_4/fixed_readers/tie_rate.json).

## 14. B0-margin: только post-hoc diagnostic

Границы квинтилей B0 margin: 0.013627565, 0.035717268, 0.069161620, 0.134653175. Margin не участвовал в router.

| Система | Квинтиль margin | n | Disagreement | P(reader better) | P(V6 better) | Mean ΔF1 |
| --- | --- | --- | --- | --- | --- | --- |
| ASEM | 0-20% | 272 | 105 | 28.57% | 15.24% | +8.8542 |
| ASEM | 20-40% | 272 | 115 | 24.35% | 16.52% | +5.4933 |
| ASEM | 40-60% | 271 | 117 | 35.90% | 17.09% | +11.2259 |
| ASEM | 60-80% | 272 | 122 | 28.69% | 18.85% | +6.1931 |
| ASEM | 80-100% | 272 | 120 | 27.50% | 25.83% | +2.4294 |

| Система | Квинтиль margin | n | Disagreement | P(reader better) | P(V6 better) | Mean ΔF1 |
| --- | --- | --- | --- | --- | --- | --- |
| ALEX | 0-20% | 272 | 80 | 25.00% | 13.75% | +10.3273 |
| ALEX | 20-40% | 272 | 83 | 19.28% | 20.48% | +3.3998 |
| ALEX | 40-60% | 271 | 96 | 28.12% | 19.79% | +9.2953 |
| ALEX | 60-80% | 272 | 104 | 23.08% | 19.23% | +6.3835 |
| ALEX | 80-100% | 272 | 100 | 30.00% | 26.00% | +5.3823 |

Чёткой монотонной связи reader win-rate с margin нет. У ASEM mean gain на disagreement меньше в самом высоком margin bin, но это post-hoc описание, не основание вводить новый routing feature или перестраивать threshold в V4.4.

Источники: [analysis/b0_margin_diagnostic.json](../artifacts/qa_chunk_selection_v4_4/analysis/b0_margin_diagnostic.json).

## 15. Вычислительная стоимость

| Измерение | Значение |
| --- | --- |
| stage_init_seconds | 0.359617 |
| stage_references_seconds | 0.375964 |
| stage_cache_seconds | 0.022241 |
| lex_feature_seconds | 89.082929 |
| semantic_confidence_seconds | 0.682046 |
| stage_features_seconds | 89.902678 |
| v6_ranking_seconds | 0.263203 |
| v6_ranking_calls | 14949.000000 |
| stage_thresholds_seconds | 0.290557 |
| ASEM_routing_seconds | 0.006512 |
| ALEX_routing_seconds | 0.006296 |
| stage_replay_seconds | 14.931808 |
| stage_compare_seconds | 0.398852 |
| stage_support_seconds | 0.000081 |
| stage_diagnostics_seconds | 0.220932 |
| stage_freeze_seconds | 0.012595 |

| Показатель | Факт |
| --- | --- |
| Время до отчёта | 106.526463 с |
| Peak RAM | 1.0286 GiB; 1104408576 bytes |
| Peak GPU allocated | null: CUDA memory measurement не получено; не численное 0 |
| QA forwards | 0 |
| QA cache | 2850/2850 |
| Embedding cache hits | 6792 |
| Уникальные cached answer vectors | 2255, по hashed .npy sources |
| Новых embeddings | 0 |
| MiniLM loaded | false |
| Чтение embeddings | 0.339936 с |
| Encode seconds | 0 |

| Система | Оценка суммы cached-компонентов на question/seed |
| --- | --- |
| V6-K5 | 65.567959 мс |
| ASEM | 66.074625 мс |
| ALEX | 66.074466 мс |

Большая часть времени — около 89.9 с из 106.5 с — приходится на features. Название `lex_feature_seconds` охватывает общий `agreement_features`, включая lexical agreement, char-F1 и source-cluster поддержку: это **не изолированный microbenchmark одной формулы LEX-FIX**. Счётчик 14949 V6 ranking calls включает повторные обращения при calibration/feature construction; это не 14949 независимых QA-вызовов.

Оценки 65.568 мс V6 и 66.075 мс ASEM/ALEX суммируют компоненты shared-cache реализации. Они исключают новый QA forward и расходы full deployment; практически одинаковое время ASEM/ALEX не доказывает нулевую цену MiniLM в production. ALEX тоже нуждается в semantic confidence. Нет выполненного PREF-SUPPORT, поэтому отсутствует его измерение стоимости. Из этих данных нельзя делать вывод «полная QA-система работает за 66 мс» или объявлять ускорение относительно полного V4.3-прогона без учёта reuse.

Источники: [analysis/efficiency.json](../artifacts/qa_chunk_selection_v4_4/analysis/efficiency.json), [cache/reuse.json](../artifacts/qa_chunk_selection_v4_4/cache/reuse.json).

## 16. Финальный статус и готовность к locked test

| Поле | Значение |
| --- | --- |
| preferred_architecture | V6-K5 |
| ready_for_internal_test | False |
| b0_final_trained | False |
| internal_test | locked |
| K | 5 |
| Единые runtime V6 weights | α=0.0; β=0.25; γ=2.0 |
| Reader weights | null: adaptive architecture не выбрана |
| Routing threshold | null: V6 не нуждается в router |
| Seed policy | один canonical seed 42, заранее |
| B0 recipe | frozen V3 B0 / V2-B2a; без выбора лучшего EVAL seed |

**34.6249 F1 относится к историческому per-run crossfit V6, а не к уже обученному единому B0-final + V6_GLOBAL.** Draft нельзя называть готовой deployable моделью и нельзя переносить эту оценку на будущий runtime vector как результат его независимого теста.

В draft дополнительно сохранён идентификатор semantic encoder; для выбранного V6 отсутствие routing threshold означает, что наличие поля не включает adaptive routing. Перед отдельно согласованным locked test остаются: refit B0-final на unlocked по frozen recipe, hash полученной модели, сборка и проверка единой runtime-конфигурации. В этом прогоне эти действия не выполнялись.

Источники: [final/deployable_freeze_manifest_draft.json](../artifacts/qa_chunk_selection_v4_4/final/deployable_freeze_manifest_draft.json), [final/development_selection.json](../artifacts/qa_chunk_selection_v4_4/final/development_selection.json).

## 17. Что можно и чего нельзя заключить

Можно заключить:

1. Фиксированный Q70 adaptive SEM router даёт небольшой положительный development F1 signal, но не проходит заранее установленную защиту Hit@1.
2. Из двух целей «качество ответа» и «выбор целевого evidence» улучшение первой не компенсирует потерю второй в данном контракте.
3. Большинство выборов reader-ветки не меняет top1; recall полезных disagreement низок. Confidence не демонстрирует ожидаемой монотонности.
4. V6-K5 остаётся выбранной архитектурой V4.4. ALEX, always-controls и support не заменяют её.
5. Прогон завершился корректно; quantitative quality failure не является техническим падением исполнения.

Нельзя заключить:

- что ASEM доказанно лучше на новых данных: reused development pool и исторический выбор архитектуры оставляют selection bias;
- что SEM-FIX статистически лучше LEX-FIX или что ALEX прошёл/проиграл H2: соответствующие CI/tests здесь не выполнялись;
- что support/veto или ASOFT не работают: они не были оценены;
- что единая финальная модель уже получила F1 34.62 или что пора автоматически открывать test;
- что 1359 строк — 1359 независимых вопросов, или что seed42 выбран по результатам;
- что низкие TM lower bounds или NaN вспомогательной pairwise-метрики подтверждают отсутствие атрибуционных ошибок.

## 18. Вывод для следующего этапа

При цели завершить текущую линию model selection результат соответствует отрицательной ветке раздела 66 протокола: **зафиксировать V6 как выбранную архитектуру и переходить к подготовке единого final freeze**, а не постфактум ослаблять Hit@1 gate или менять Q70.

Если пользователь решит продолжать method development, новый reader-only/router/utility эксперимент требует отдельного заранее описанного протокола. Более высокие always-control F1 и missed opportunities полезны как направление исследования, но не отменяют решения V4.4 и не оправдывают дополнительный подбор на locked test.

В рамках подготовки этого отчёта никакие новые эксперименты не запускались и никакие thresholds, gates, результаты или исходный код не менялись.

## Приложение A. Все поля агрегированных metrics без потери точности смысла

Ниже raw scales из JSON: Token F1 / attribution-aware F1 / F1|positive — 0–100; EM/Hit/HN/Amb/GSRN/char-F1/ROUGE-L и coverage — доли; MRR и pairwise — безразмерные. `null` и `NaN` сохранены как отсутствие определённого значения, не заменены нулём. Дублирующие `*_original`/`*_v3` поля намеренно включены для прослеживаемости.

| Поле | B0-live | V6-K5 | SEM-FIX | LEX-FIX | ASEM | ALEX |
| --- | --- | --- | --- | --- | --- | --- |
| n | 1359 | 1359 | 1359 | 1359 | 1359 | 1359 |
| mrr | 0.541771127 | 0.570431907 | 0.580144931 | 0.579360040 | 0.565023518 | 0.565195213 |
| token_f1 | 27.358413858 | 34.624867427 | 37.510759002 | 36.981618251 | 35.111237406 | 34.774271616 |
| exact_match | 0.207505519 | 0.253127299 | 0.314937454 | 0.300956586 | 0.259749816 | 0.259013981 |
| char_f1 | 0.521002472 | 0.569288675 | 0.618007838 | 0.611841188 | 0.574431290 | 0.573006321 |
| rouge_l_f1 | 0.338710551 | 0.427986463 | 0.456518589 | 0.454893912 | 0.429824553 | 0.429560586 |
| attribution_aware_token_f1 | 26.802332706 | 33.309694314 | 35.614757040 | 35.152659049 | 33.430599490 | 33.142689378 |
| pairwise_pos_gt_hn | 0.749573989 | 0.729264579 | 0.735798782 | 0.736115112 | 0.726882214 | 0.727682926 |
| pairwise_clean_hn | 0.751387923 | 0.415776706 | 0.440323881 | 0.429919042 | 0.421249469 | 0.419527720 |
| pairwise_pos_gt_amb_original | 0.645075635 | 0.466676909 | 0.476363293 | 0.473119784 | 0.470897322 | 0.468227838 |
| pairwise_pos_gt_gsrn_original | 0.554654430 | 0.494557118 | 0.494275354 | 0.487929684 | 0.492347907 | 0.492541471 |
| pairwise_pv3_gt_tm | NaN | NaN | NaN | NaN | NaN | NaN |
| ambiguous_at_1_original | 0.508462104 | 0.428991906 | 0.405445180 | 0.403973510 | 0.432671082 | 0.431199411 |
| gsrn_at_1_original | 0.339220015 | 0.266372333 | 0.253863135 | 0.242825607 | 0.266372333 | 0.262693157 |
| target_status_coverage@1 | 0.491537896 | 0.571008094 | 0.594554820 | 0.596026490 | 0.567328918 | 0.568800589 |
| target_mismatch@1_lower_bound | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |
| target_mismatch_rate@1_adjudicated | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |
| wrong_target_answer_rate_lower_bound | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |
| wrong_target_coverage | 0.491537896 | 0.571008094 | 0.594554820 | 0.596026490 | 0.567328918 | 0.568800589 |
| gold_match_from_non_original_positive | 0.046357616 | 0.072847682 | 0.081677704 | 0.080941869 | 0.079470199 | 0.077998528 |
| f1_given_selected_positive | 61.702263617 | 65.223452883 | 72.561655271 | 70.038272500 | 67.452986330 | 66.498390447 |
| target_mismatch@1 | null | null | null | null | null | null |
| target_mismatch@1_note | incomplete coverage; use lower_bound and adjudicated rate | incomplete coverage; use lower_bound and adjudicated rate | incomplete coverage; use lower_bound and adjudicated rate | incomplete coverage; use lower_bound and adjudicated rate | incomplete coverage; use lower_bound and adjudicated rate | incomplete coverage; use lower_bound and adjudicated rate |
| hit@1 | 0.367181751 | 0.428991906 | 0.428256071 | 0.434878587 | 0.418690213 | 0.419426049 |
| hit@1_original | 0.367181751 | 0.428991906 | 0.428256071 | 0.434878587 | 0.418690213 | 0.419426049 |
| target_evidence_hit@1_v3 | 0.367181751 | 0.428991906 | 0.428256071 | 0.434878587 | 0.418690213 | 0.419426049 |
| hn@1 | 0.048565121 | 0.054451803 | 0.064017660 | 0.055187638 | 0.058130979 | 0.058866814 |
| hit@5 | 0.790286976 | 0.790286976 | 0.790286976 | 0.790286976 | 0.790286976 | 0.790286976 |
| hit@5_original | 0.790286976 | 0.790286976 | 0.790286976 | 0.790286976 | 0.790286976 | 0.790286976 |
| target_evidence_hit@5_v3 | 0.790286976 | 0.790286976 | 0.790286976 | 0.790286976 | 0.790286976 | 0.790286976 |
| hn@5 | 0.069021339 | 0.069021339 | 0.069021339 | 0.069021339 | 0.069021339 | 0.069021339 |
| target_status_coverage@5 | 0.478513613 | 0.478513613 | 0.478513613 | 0.478513613 | 0.478513613 | 0.478513613 |
| hit@10 | 0.922001472 | 0.922001472 | 0.922001472 | 0.922001472 | 0.922001472 | 0.922001472 |
| hit@10_original | 0.922001472 | 0.922001472 | 0.922001472 | 0.922001472 | 0.922001472 | 0.922001472 |
| target_evidence_hit@10_v3 | 0.922001472 | 0.922001472 | 0.922001472 | 0.922001472 | 0.922001472 | 0.922001472 |
| hn@10 | 0.082329444 | 0.082329444 | 0.082329444 | 0.082329444 | 0.082329444 | 0.082329444 |
| target_status_coverage@10 | 0.499048670 | 0.499048670 | 0.499048670 | 0.499048670 | 0.499048670 | 0.499048670 |

## Приложение B. Confidence bins каждого outer run

Описательные результаты без новых gates. P и mean ΔF1 — на disagreement внутри соответствующего bin. Empty bins/disagreements выводятся как n/a.

### Run 0

ASEM: edges = 0.142885134, 0.242361791, 0.341704194, 0.449991911; monotone=False.

| Bin | n | Disagreement | P(reader better) | P(V6 better) | Mean ΔF1 |
| --- | --- | --- | --- | --- | --- |
| 0-20% | 54 | 30 | 6.67% | 30.00% | -5.1285 |
| 20-40% | 54 | 26 | 19.23% | 7.69% | +9.0984 |
| 40-60% | 54 | 22 | 27.27% | 22.73% | +0.7576 |
| 60-80% | 54 | 15 | 60.00% | 33.33% | +19.3016 |
| 80-100% | 54 | 17 | 47.06% | 17.65% | +11.1686 |

ALEX: edges = 0.142885134, 0.242361791, 0.341704194, 0.449991911; monotone=False.

| Bin | n | Disagreement | P(reader better) | P(V6 better) | Mean ΔF1 |
| --- | --- | --- | --- | --- | --- |
| 0-20% | 54 | 27 | 14.81% | 22.22% | +4.9190 |
| 20-40% | 54 | 25 | 12.00% | 16.00% | +1.5290 |
| 40-60% | 54 | 15 | 46.67% | 26.67% | +24.9206 |
| 60-80% | 54 | 7 | 85.71% | 0.00% | +46.9388 |
| 80-100% | 54 | 8 | 50.00% | 0.00% | +22.3443 |

### Run 1

ASEM: edges = 0.183406863, 0.260960597, 0.336804728, 0.439603651; monotone=False.

| Bin | n | Disagreement | P(reader better) | P(V6 better) | Mean ΔF1 |
| --- | --- | --- | --- | --- | --- |
| 0-20% | 55 | 28 | 21.43% | 17.86% | +5.2721 |
| 20-40% | 54 | 31 | 22.58% | 16.13% | +5.7373 |
| 40-60% | 54 | 23 | 8.70% | 47.83% | -17.7776 |
| 60-80% | 55 | 36 | 19.44% | 5.56% | +7.2421 |
| 80-100% | 55 | 36 | 2.78% | 8.33% | -1.6865 |

ALEX: edges = 0.183406863, 0.260960597, 0.336804728, 0.439603651; monotone=False.

| Bin | n | Disagreement | P(reader better) | P(V6 better) | Mean ΔF1 |
| --- | --- | --- | --- | --- | --- |
| 0-20% | 55 | 22 | 13.64% | 4.55% | +10.7143 |
| 20-40% | 54 | 21 | 23.81% | 14.29% | +18.2313 |
| 40-60% | 54 | 19 | 5.26% | 42.11% | -6.0315 |
| 60-80% | 55 | 35 | 14.29% | 5.71% | +6.0476 |
| 80-100% | 55 | 32 | 3.12% | 9.38% | -1.8973 |

### Run 2

ASEM: edges = 0.175386191, 0.246771198, 0.332518969, 0.433808282; monotone=False.

| Bin | n | Disagreement | P(reader better) | P(V6 better) | Mean ΔF1 |
| --- | --- | --- | --- | --- | --- |
| 0-20% | 54 | 21 | 38.10% | 9.52% | +23.2885 |
| 20-40% | 53 | 29 | 37.93% | 10.34% | +19.5725 |
| 40-60% | 54 | 20 | 35.00% | 30.00% | +1.7929 |
| 60-80% | 55 | 6 | 16.67% | 50.00% | +3.1746 |
| 80-100% | 54 | 8 | 75.00% | 12.50% | +26.1310 |

ALEX: edges = 0.175386191, 0.246771198, 0.332518969, 0.433808282; monotone=False.

| Bin | n | Disagreement | P(reader better) | P(V6 better) | Mean ΔF1 |
| --- | --- | --- | --- | --- | --- |
| 0-20% | 54 | 17 | 41.18% | 11.76% | +26.2745 |
| 20-40% | 53 | 20 | 25.00% | 15.00% | +7.2372 |
| 40-60% | 54 | 19 | 26.32% | 47.37% | +0.9088 |
| 60-80% | 55 | 3 | 33.33% | 66.67% | +6.3492 |
| 80-100% | 54 | 4 | 100.00% | 0.00% | +48.5714 |

### Run 3

ASEM: edges = 0.170197497, 0.271739374, 0.342099071, 0.415463936; monotone=False.

| Bin | n | Disagreement | P(reader better) | P(V6 better) | Mean ΔF1 |
| --- | --- | --- | --- | --- | --- |
| 0-20% | 54 | 22 | 22.73% | 18.18% | +0.8938 |
| 20-40% | 55 | 18 | 22.22% | 22.22% | +1.7989 |
| 40-60% | 55 | 33 | 18.18% | 24.24% | -1.8607 |
| 60-80% | 53 | 23 | 47.83% | 21.74% | +13.5404 |
| 80-100% | 56 | 16 | 43.75% | 12.50% | +11.4286 |

ALEX: edges = 0.170197497, 0.271739374, 0.342099071, 0.415463936; monotone=False.

| Bin | n | Disagreement | P(reader better) | P(V6 better) | Mean ΔF1 |
| --- | --- | --- | --- | --- | --- |
| 0-20% | 54 | 16 | 12.50% | 12.50% | +9.2647 |
| 20-40% | 55 | 20 | 20.00% | 30.00% | +1.6190 |
| 40-60% | 55 | 17 | 17.65% | 17.65% | -1.5686 |
| 60-80% | 53 | 15 | 20.00% | 46.67% | -10.4444 |
| 80-100% | 56 | 2 | 0.00% | 50.00% | -10.0000 |

### Run 4

ASEM: edges = 0.193576621, 0.264182697, 0.355591352, 0.492606342; monotone=False.

| Bin | n | Disagreement | P(reader better) | P(V6 better) | Mean ΔF1 |
| --- | --- | --- | --- | --- | --- |
| 0-20% | 55 | 32 | 50.00% | 3.12% | +24.5975 |
| 20-40% | 54 | 30 | 40.00% | 20.00% | +9.5246 |
| 40-60% | 55 | 27 | 29.63% | 22.22% | +3.3333 |
| 60-80% | 54 | 18 | 55.56% | 11.11% | +24.6605 |
| 80-100% | 55 | 12 | 25.00% | 50.00% | -15.4762 |

ALEX: edges = 0.193576621, 0.264182697, 0.355591352, 0.492606342; monotone=False.

| Bin | n | Disagreement | P(reader better) | P(V6 better) | Mean ΔF1 |
| --- | --- | --- | --- | --- | --- |
| 0-20% | 55 | 30 | 50.00% | 3.33% | +22.5337 |
| 20-40% | 54 | 30 | 40.00% | 23.33% | +6.1912 |
| 40-60% | 55 | 31 | 29.03% | 32.26% | -3.5177 |
| 60-80% | 54 | 16 | 31.25% | 18.75% | +8.0903 |
| 80-100% | 55 | 12 | 25.00% | 50.00% | -15.4762 |

## Приложение C. Навигация по исходным артефактам

| Группа | Содержимое |
| --- | --- |
| contract.json / config_resolved.json | зафиксированные формулы, margins, seed, режимы и трактовки до replay |
| reference/ | B0, исторический V6, текущий live V6, LEXCORE/SEMCORE/ADAPT1 V4.3 |
| fixed_readers/ | веса и tie-rate |
| confidence/features.jsonl | 1359 feature packs с sim matrix, SemRel, triple, confidence, B0 margin |
| confidence/run*/tau70.json | все calibration source keys, thresholds, минимум и причины включения |
| asem/ и alex/ | predictions_scored/crossfit_predictions, per-run predictions и gate/skip files |
| sem-fix/ и lex-fix/ | предсказания и метрики always-controls |
| analysis/error_buckets/ | полные записи good/bad/unnecessary/missed buckets с answers, labels, F1 и routing fields |
| analysis/ | все агрегированные метрики, routing rates/precision/recall, confidence/margin bins, transitions, эффективность |
| support_diagnostic/status.json | условная ветка не запускалась |
| regression/ | историческая/живая V6 identity и shortlist identity |
| source_hashes.json | 2468 SHA256, сверены при подготовке отчёта |
| final/ | решение development и deployment draft |
| progress.jsonl / status.json | timeline и exit=0 |
| reports/results_v4_4.md | исходный краткий автодамп; настоящий документ содержит аналитический разбор |

Полный лог: [qa_chunk_selection_v4_4_20260910_130302_1994740.log](../logs/qa_chunk_selection_v4_4_20260910_130302_1994740.log). История: [результаты V4.3](qa_chunk_selection_experiments_v4_3_results.md). Реализация: [V4.4 README](../experiments/qa_chunk_selection/v4_4/README.md).
