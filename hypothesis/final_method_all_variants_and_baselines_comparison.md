# Обобщённое сравнение вариантов UnTIE и бейзлайнов

Дата фиксации: **2026-10-08**  
Финальная архитектура: **UnTIE H3 Consistency Gate / `H3-CONSISTENCY-GATE-3`**  
Основная метрика: **answer Token F1, 0–100**

## 1. Как читать сравнение

В проекте есть два существенно разных слоя результатов.

1. **Official-split TEST:** единые подготовленные TEST-популяции, одинаковый gold-side фильтр answerable-инстансов и сопоставимая оценка. Только эти числа можно помещать в прямой итоговый рейтинг.
2. **Исторические development-эксперименты V4.4–V10:** менялись разбиения, B0, вопросы, чанкирование и правила отбора. Они объясняют развитие метода, но их абсолютные F1 нельзя напрямую ранжировать вместе с official TEST.

SciER-128 и SciER-384 построены из одних исходных документов. В `ALL_STANDARDIZED` они сначала усредняются в одно семейство SciER, после чего SciREX, SciER, RuSERRC и SciMDIX получают одинаковый вес.

TEST-статус:

- SciREX, SciER и SciMDIX не использовались в V1–V10 до official benchmark;
- RuSERRC уже использовался при разработке и является exposed;
- финальные H3-абляции выполнены после раскрытия official TEST и имеют post-hoc статус.

## 2. Реестр бейзлайнов и основных вариантов

| Код | Система | Смысл |
|---|---|---|
| B1 | QA по всем чанкам, max confidence | простой baseline без B0 |
| B2 | B0 top-1 → QA | сильный простой retrieval baseline |
| B3 | B0 top-K → QA, max confidence | простой multi-chunk baseline |
| B4a | frozen BGE top-1 → QA | внешний reranker baseline |
| B4b | frozen BGE top-K → QA | основной strong-reranker baseline |
| B5-EN | in-domain fine-tuned ModernBERT QA | сильный supervised EN baseline |
| B5S-B0K | frozen ModernBERT-SQuAD2 поверх B0 top-K | post-hoc EN baseline |
| B5S-ALL | frozen ModernBERT-SQuAD2 по всем чанкам | post-hoc EN baseline без B0 shortlist |
| G1 | frozen Qwen2.5-3B | generative baseline |
| P1 | exact V7-S2 | ранний предложенный вариант |
| P2 | H2 | safe utility/harm meta-reranker с PairLEX и FullLEX |
| P3 | H2-ACS | adaptive component selection из V9 |
| P4 | H3 fixed | H2 плюс фиксированная agreement-группа |
| FINAL | H3 Consistency Gate | shared `lambda` и exact TRAIN/refit stability rule |
| D1 | answer oracle | выбирает доступного кандидата по gold answer F1; только диагностика |
| D2 | evidence oracle | выбирает gold-positive evidence; только диагностика |

## 3. Полная official-split таблица

Все системы в этой таблице, кроме помеченных post-hoc H3-расширений в следующем разделе, были рассчитаны в едином итоговом benchmark.

| Система | SciREX | SciER-384 | SciER-128 | RuSERRC | SciMDIX | ALL_STANDARDIZED |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 14.7732 | 56.0738 | 65.4365 | 56.5473 | 33.2485 | 41.3310 |
| B2 | 24.0979 | 63.6773 | 69.3381 | 65.6714 | 52.6723 | 52.2373 |
| B3 | 19.0955 | 63.2028 | **73.1368** | 58.2139 | 38.7216 | 46.0502 |
| B4a | 17.5330 | 52.7309 | 56.0724 | 62.9798 | 42.5633 | 44.3694 |
| B4b | 18.4645 | 60.8589 | 56.1772 | 56.6114 | 36.8576 | 42.6129 |
| G1 | 17.6245 | 31.5206 | 31.5206 | 44.8174 | 27.7844 | 30.4367 |
| P1 / V7-S2 | 24.6014 | 67.2980 | 71.1316 | 65.7127 | 58.2940 | 54.4557 |
| P2 / H2 | 23.8234 | 65.8721 | 70.5551 | 65.9369 | **62.8274** | 55.2003 |
| P2 без FullLEX | 26.2151 | 65.2146 | 69.3381 | 65.7127 | 58.2940 | 54.3745 |
| P2 без PairLEX | 22.9719 | 70.6049 | 69.9904 | 66.0126 | 56.8582 | 54.0351 |
| P2 без risk | 23.8234 | 65.8721 | 69.6187 | 65.9369 | 62.8274 | 55.0833 |
| P3 / H2-ACS | 23.8234 | 65.8721 | 70.5551 | 65.9369 | **62.8274** | 55.2003 |
| P4 / H3 fixed | **26.0253** | **72.0436** | 72.7105 | **66.2114** | 59.2560 | 55.9674 |
| **FINAL / H3 Consistency Gate** | **26.0253** | **72.0436** | 72.7105 | **66.2114** | **62.8274** | **56.8603** |

Главный результат: финальный gate воспроизводит H3 на SciREX, обеих SciER-cell и RuSERRC, но использует H2 на SciMDIX. Поэтому на текущем TEST он совпадает с лучшим из H2/H3 на каждой cell.

На SciER-128 простой B3 получает `73.1368`, что немного выше финального метода `72.7105`. Это важное исключение: финальная архитектура не является лучшей реальной системой на абсолютно каждой cell.

## 4. Специализированные английские QA-бейзлайны

Эти системы доступны только для английских данных, поэтому для них нельзя вычислять общий четырёхсемейный `ALL_STANDARDIZED`.

| Система | SciREX | SciER-384 | SciER-128 | EN family-balanced |
|---|---:|---:|---:|---:|
| B5-EN, in-domain fine-tuned | **56.3056** | **100.0000** | **97.0556** | **77.4167** |
| B5S-B0K, frozen SQuAD2 | 22.9600 | 54.5416 | 62.1445 | 40.6515 |
| B5S-ALL, frozen SQuAD2 | 15.7788 | 57.3247 | 49.2373 | 34.5299 |
| P2/H2 | 23.8234 | 65.8721 | 70.5551 | 46.0185 |
| P4/H3 | 26.0253 | 72.0436 | 72.7105 | 49.2012 |
| FINAL | 26.0253 | 72.0436 | 72.7105 | 49.2012 |

Выводы:

- in-domain B5-EN значительно превосходит все несупервизированные/мета-reranking варианты на английском;
- frozen ModernBERT-SQuAD2 сам по себе слабее H2/H3;
- B5S-B0K превосходит B5S-ALL в EN-агрегате на `6.1217`, поэтому даже для готовой QA-модели shortlist B0 полезен;
- высокий результат B5-EN объясняется прежде всего in-domain дообучением, а не только архитектурой ModernBERT.

## 5. Диагностические верхние ориентиры

| Oracle | SciREX | SciER-384 | SciER-128 | RuSERRC | SciMDIX |
|---|---:|---:|---:|---:|---:|
| D1: лучший доступный answer | 47.4094 | 92.8773 | 91.1746 | 88.2977 | 88.5894 |
| D2: positive evidence | 27.6337 | 64.4797 | 76.2749 | 86.0916 | 88.3414 |

D1 показывает большой неиспользованный запас выбора ответа среди уже рассчитанных кандидатов. D2 не является верхней границей answer F1: чанк с gold-positive evidence не всегда даёт reader лучший ответ.

## 6. Все отдельные H3-refinement варианты

Эти опыты используют уже раскрытый official TEST. Они нужны для выбора финальной архитектуры, но не являются независимым подтверждением.

| Вариант | SciREX | SciER-384 | SciER-128 | RuSERRC | SciMDIX | ALL_STANDARDIZED |
|---|---:|---:|---:|---:|---:|---:|
| H3 zero-anchor | 26.0253 | 72.0436 | 72.7105 | 66.2114 | 59.2560 | 55.9674 |
| H3-BLEND, seed-specific `lambda` | 24.6360 | 69.1947 | 72.7105 | 66.1630 | 61.6370 | 55.8471 |
| H3-GROUP-RIDGE | 23.6745 | 70.4540 | 72.7105 | 66.2114 | 59.4911 | 55.2398 |
| H3-SHARED-3 | 26.0253 | 72.0436 | 72.7105 | 65.9369 | 62.8274 | 56.7917 |
| H3-SHARED-10 | **26.0506** | 69.6302 | **72.9994** | 66.0558 | 59.7790 | 55.8000 |
| H3-STABILITY-3 с CI gate | 26.0253 | 72.0436 | 72.7105 | 66.2114 | 59.2560 | 55.9674 |
| H3-STABILITY-10 с CI gate | **26.0506** | 71.3038 | **72.9994** | 66.0848 | 59.4676 | 55.9386 |
| **H3-CONSISTENCY-GATE-3** | 26.0253 | **72.0436** | 72.7105 | **66.2114** | **62.8274** | **56.8603** |

Почему выбран последний вариант:

- zero-anchor оказался точным no-op;
- отдельный `lambda` для каждого seed был нестабилен и ухудшал сильные H3-cell;
- group ridge не решил проблему SciMDIX и потерял качество на SciREX;
- shared `lambda` по трём seeds исправил SciMDIX, но ошибочно ослабил RuSERRC;
- увеличение до десяти seeds не стабилизировало перенос и ухудшило shared-3;
- CI-based stability gate был слишком строгим и отверг устойчивый выбор H2 на SciMDIX;
- exact consistency-only gate сохранил H2 на SciMDIX и вернул H3 на RuSERRC.

## 7. Статистика для финального варианта

Document-cluster bootstrap, 10 000 итераций; seeds сначала усредняются внутри вопроса, семейства имеют равный вес.

| Контраст | Delta Token F1 | 95% CI |
|---|---:|---:|
| FINAL − P2/H2 | **+1.6600** | **[+0.1396; +3.0761]** |
| FINAL − P4/H3 | +0.8929 | [0.0000; +2.9412] |
| FINAL − H3-SHARED-3 | +0.0686 | [0.0000; +0.1984] |
| P4/H3 − P2/H2 | +0.7671 | [−1.7213; +2.7234] |

Финальный метод статистически превосходит H2 в сбалансированном агрегате. Нижние границы сравнений с H3 и shared-3 равны нулю, поскольку различие локализовано соответственно в SciMDIX и RuSERRC. Универсальное превосходство над H3 пока не доказано.

## 8. Что показали абляции H2

Изменение `абляция − полный H2` в `ALL_STANDARDIZED`:

| Удалённый компонент | Delta |
|---|---:|
| FullLEX | −0.8258 |
| PairLEX | −1.1652 |
| risk | −0.1170 |

PairLEX и FullLEX полезны в общем агрегате, однако эффект сильно зависит от датасета и особенно велик на SciMDIX. Risk-компонент почти не влияет на итоговые решения: четыре из пяти cell дают те же ответы, а общий вклад равен лишь `0.1170` пункта.

## 9. История поколений V4.4–V10

Таблица ниже описывает development evidence. Абсолютные значения не следует напрямую сравнивать с official TEST.

| Версия | Основная идея | Наиболее содержательный результат | Итог |
|---|---|---|---|
| V4.4 | adaptive reader consensus, SEM/LEX/ASEM/ALEX | сильные EN эффекты; SciREX SEM-FIX `37.5108`, SciER-384 LEX `60.8250`, SciER-128 SEM `62.5857`; SciMDIX ASEM-Q5 `+2.280` к DiTy B0 | полезный reader-сигнал подтверждён, единого переносимого победителя нет |
| V4.5 | компактное русское уточнение и сильные русские ranker baselines | BGE B0: RuSERRC `69.9535`, SciMDIX `61.7486`; second stage помог SciMDIX, но не RuSERRC | качество русского B0 критично; универсальный second-stage эффект не подтверждён |
| V5 | question-conditioned residual | secondary A2: SciREX `+3.9786`; SciMDIX point estimate `+3.6378` | primary A1 не поддержан; полезные результаты остались secondary |
| V6 | универсальный UAR-G | небольшие локальные приросты, лучший eligible общий эффект недостаточен | `NO_UNIVERSAL_CANDIDATE` |
| V7 | progressive risk-aware reranking | SciMDIX LOCAL S3 `+2.4031`, CI `[+0.2039; +4.7725]` | локальный успех SciMDIX; universal candidate не поддержан |
| V8 | dual branch; HRR H0→H3 и V4.4 adaptive fusion | development H3: SciREX `+8.9159`, SciER-128 `+4.8037`; agreement вредил SciER-384 и русским cell относительно H2 | H2/H3 стали сильным ядром; `V8_UNIVERSAL_CANDIDATE_NOT_SUPPORTED` |
| V9 | H2 adaptive component selection | ACS улучшил только SciER-128 на `+2.0454`, CI `[+0.7684; +3.4460]`; остальные cell оставили H2 | `NO_UNIVERSAL_DEVELOPMENT_CANDIDATE` |
| V10 | SWITCH → RANK → RISK | SRR лучше H2 по point estimate только на SciMDIX; family-balanced `−1.058`, CI `[−2.124; −0.014]` | универсальный SRR отвергнут |
| Official benchmark | blind official split для B1–B5/G1/P1–P4 | H3/P4 лучший заранее включённый proposed-вариант: `55.9674` | H3 силён на EN и RuSERRC, но проигрывает H2 на SciMDIX |
| H3 refinements | уменьшение/отбор agreement-вклада | consistency gate: `56.8603` | зафиксирован как финальная архитектура |

## 10. Основные научные выводы всей серии

1. **B0 и supervised reader остаются сильнейшими источниками качества.** Русский BGE давал крупнейшие ранние улучшения, а in-domain ModernBERT значительно превосходит остальные английские системы.
2. **Второй компонент метода приносит самостоятельную пользу.** Это подтверждено на английских V4.4/V8 результатах и на SciMDIX, но эффект agreement неодинаков между корпусами.
3. **H2 является наиболее устойчивым ядром.** PairLEX и FullLEX в среднем полезны; сложные risk-модули и ACS не дали универсального дополнительного выигрыша.
4. **H3 особенно полезен для длинных/вариативных английских данных, но agreement может вредить тематически компактному SciMDIX.**
5. **Простое и устойчивое dataset-level решение оказалось лучше более сложных контроллеров.** Exact TRAIN/refit consistency gate добавляет один коэффициент и одно правило fallback, сохраняя концепцию H2/H3.
6. **Запас качества остаётся большим.** D1 показывает, что проблема часто находится в выборе правильного кандидата или ответа среди уже доступных top-K, а не только в retrieval recall.

## 11. Финальный вывод

Финальной архитектурой считается `H3-CONSISTENCY-GATE-3` с `ALL_STANDARDIZED = 56.8603`. Она имеет лучший наблюдаемый единый результат среди сопоставимых H2/H3-вариантов и не ухудшает ни одну official TEST cell относительно лучшего из H2 и H3.

Архитектура зафиксирована после просмотра TEST, поэтому её следует называть **финальной замороженной post-hoc архитектурой**. Следующая научно корректная проверка должна использовать новый закрытый split или новый независимый датасет без изменения правила consistency gate.

## 12. Источники

- `hypothesis/final_method_architecture.md`;
- `hypothesis/result_final_official_split_h2_family_benchmark_analysis.md`;
- `hypothesis/qa_chunk_selection_v4_4_to_v7_best_results_by_dataset.md`;
- `hypothesis/v8_dual_branch_results_analysis_2026-09-23.md`;
- `hypothesis/v9_h2_adaptive_component_selection_results_detailed.md`;
- `hypothesis/v10_srr_switch_rank_risk_results_detailed.md`;
- `hypothesis/h3_zero_anchor_agreement_ablation_results.md`;
- `hypothesis/h3_shrinkage_ablation_results_analysis.md`;
- `hypothesis/h3_shared_lambda_ablation_results_analysis.md`;
- `hypothesis/h3_lambda_robustness_results_analysis.md`;
- `hypothesis/h3_consistency_gate_results_analysis.md`.
