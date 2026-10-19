# V4.4 — adaptive reader-consensus

Независимый пакет для [протокола V4.4](../../../hypothesis/v4_4_adaptive_reader_consensus_protocol.md).
Результат — **cross-fitted development stability replay**, не независимое подтверждение.
Пакеты предыдущих экспериментов, production и исходные данные не меняются.

## Фоновый запуск

Из корня проекта:

```bash
DEVICE=auto bash experiments/qa_chunk_selection/v4_4/launch_background.sh
```

По умолчанию результаты идут в `artifacts/qa_chunk_selection_v4_4/`.
Скрипт выводит PID, путь уникального лога в `logs/` и готовую команду `tail -f`.
PID дополнительно сохраняется рядом с логом в `.log.pid`.

Каждый запуск требует **новую, ещё не существующую** папку результатов, включая повторение
после ошибки. Перезапись и автоматическое продолжение незавершённых запусков отключены.
Для повторения укажите новый путь:

```bash
OUT_DIR="$PWD/artifacts/qa_chunk_selection_v4_4_run2" \
  DEVICE=auto bash experiments/qa_chunk_selection/v4_4/launch_background.sh
```

Переменные: `PYTHON` (по умолчанию `untie_venv/bin/python`), `OUT_DIR`, `DEVICE`,
`LOG` (только новый `.log` внутри `logs/`), `SUPPORT=0` (заранее отключить условный support),
`SOFT=1` (заранее включить ASOFT diagnostic), `SMOKE=1` (искусственные данные).
В отличие от старых launcher, `STAGES`, `SKIP_QA`, смена seed, force-gate и overwrite-флаги
не предусмотрены: обязательные проверки нельзя пропустить.

## Прогресс

- Лог: стадии, счётчики вопросов, run-specific Q70 и число строк калибровки, решение H1/H2.
- `progress.jsonl`: тот же прогресс в структурированном виде; heartbeat каждые 30 секунд.
- `status.json`: последняя завершённая стадия или ошибка; `stage=done, exit_code=0` означает завершение.
- `reports/results_v4_4.md`: автоматический отчёт.
- `final/development_selection.json`: единственный preferred candidate и решения gates.
- `final/deployable_freeze_manifest_draft.json`: черновик единой конфигурации, **не разрешение открыть test**.

## Что выполняется

1. До нового EVAL записывается неизменяемый контракт и конфигурация.
2. Проверяются frozen V3 universe, три seed и исторические V4.1/V4.3 references.
3. Читаются QA-ответы из `v4_3/cache/qa_topk.jsonl`: это экспорт кэша V4.1–V4.3.
   Новых QA-forward нет. Неполный кэш вызывает ошибку, без тихого запуска reader.
4. Answer embeddings читаются непосредственно из V4.3; нет записи в прежний кэш.
   Confidence заново вычисляется общими функциями V4.3; веса reader фиксированы.
5. Q70 считается на доступных TRAIN+CAL строках отдельно для ASEM и ALEX.
   При числе строк меньше 100 — exact V6 fallback. Gold и aspect не входят в router.
6. Выполняются development replay, строгая проверка V6 identity и top-5 isolation,
   paired bootstrap (5000; mean seeds per question, затем document clusters) и иерархия H1/H2.
7. Support Q20 и ASOFT — только при H1 pass, только diagnostics; не меняют победителя.
8. Сохраняются routing audits, confidence bins, aspect/seed metrics, transitions, ошибки,
   затраты компонентов, RAM/GPU и hashes источников/реализации.

Для основного replay не требуется загружать MiniLM: используются сохранённые embeddings.
Только conditional support загружает тот же MiniLM для новых Query/Query+Answer/LocalEvidence.
Хэш идентификатора модели и нормализация должны совпасть с V4.3, иначе запуск прекращается.
Стоимость в `analysis/efficiency.json` относится к cached replay; оценки суммы компонентов
явно отделены от latency полного production inference, где QA не закэширован.

## Зафиксированные трактовки протокола

- Раздел 24 требует ALEX среди обязательных outputs, а раздел 62 разрешает H2 только после H1.
  Поэтому ALEX всегда рассчитан описательно; **сравнение H2 и возможность замены ASEM** доступны
  только после H1 pass. При H1 fail выбор строго V6, даже если описательный ALEX выше.
- Для H2 superiority «допустимый HN» означает тот же бюджет, что H1: `HN_ALEX <= HN_V6 + 0.01`.
  Для H2 non-inferiority действуют более строгие явно заданные HN/Hit условия относительно ASEM.
- Q70 строится отдельно по availability каждого reader, pooling `(question_id, seed)` для 42/43/44;
  linear quantile. Seeds не трактуются как независимые наблюдения статистической оценки.
- Support Q20: только available TRAIN+CAL и disagreement; фильтр прохождения Q70 **не добавлен**,
  в соответствии с перечисленными условиями раздела 41. Меньше 100 — диагностический veto отключён.
- До replay выбран один canonical seed 42. Глобальный Q70 для черновика использует available
  cross-fitted строки только этого seed; при недостатке 100 строк draft остаётся заблокированным.
- V6_GLOBAL проверяется против V4.1 manifest: `(0.0, 0.25, 2.0)`. В runtime draft нет per-run weights.

## Граница этого запуска

V4.4 не обучает B0-final и не открывает INTERNAL_TEST. Из разделов 59–60 здесь реализована
подготовка draft: preferred architecture, global weights, global threshold, seed policy,
provenance hashes и список оставшихся действий. Final refit, сборка deployable pipeline и
однократный locked test — отдельный последующий этап; `ready_for_internal_test=false` всегда.

Исторический `dataset.jsonl` физически содержит разные splits. Загрузчик сканирует этот
общий файл и декодирует InstanceRecord **только** для unlocked question IDs из frozen B0 OOF.
Строки остальных splits не превращаются в записи, не поступают в QA/метрики/калибровку.
Отдельные test-файлы и test-артефакты не открываются; флага открытия test нет.

## Проверки без глобального прогона

```bash
PYTHONDONTWRITEBYTECODE=1 ./untie_venv/bin/python -B -m pytest \
  tests/test_qa_chunk_selection_v4_4.py -q -p no:cacheprovider

./untie_venv/bin/python -B -m experiments.qa_chunk_selection.v4_4 \
  --smoke --device cpu --output-dir /tmp/untie_v44_smoke_new
```

Smoke использует synthetic records, HashEncoder и bootstrap=50; это проверка исполнения,
не результат эксперимента. Тесты дополнительно проверяют enabled support, exact veto/fallback,
пороговые границы gates и запрет перезаписи.
