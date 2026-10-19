# V9 / H2-ACS: детальный анализ результатов

**Run:** `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400`  
**Статус:** `COMPLETED`  
**Время выполнения:** 2.73 часа  
**Статус оценки:** development evidence, не независимое подтверждение

## Итог эксперимента

V9 подтвердил, что TRAIN-only выбор дополнительного Agreement-компонента технически работает и может дать существенный честный outer-CV выигрыш. Однако эффект оказался локализован только в **SciER 128/24**. На этой cell S2 улучшил H2 на **+2.0454 Token F1**, 95% CI **[+0.7684; +3.4460]**, без ухудшения Hit@1/HN@1. На остальных четырёх cells selector оставил базовый H2, поэтому S2 полностью совпал с S1.

По заранее зафиксированному протоколу итог — **`NO_UNIVERSAL_DEVELOPMENT_CANDIDATE`**:

- ACS/S2 сохранил качество H2 на всех cells и дал family-balanced прирост **+0.2557**, но строго улучшил только одну cell при требовании минимум двух;
- fixed H2/S1 имеет положительный point estimate на всех пяти cells и положительный глобальный эффект, но не проходит per-cell uncertainty gate `LCB95 > -0.5` на SciER 384/50 и SciMDIX;
- диагностические fixed-конфигурации D-A/D-P/D-AP не имеют права становиться финальным кандидатом;
- повтор H2 в V9 является точным replay тех же development folds V8, а не новой независимой репликацией.

## 1. Полнота и валидность запуска

| Проверка | Результат |
|---|---|
| Полный протокол | 5 cells × 5 outer folds × 3 seeds; inner CV=4; SELECT CV=4 |
| Bootstrap | 5000 document-cluster iterations, seed 42 |
| H0/H1/H2/H3 replay | PASS; по 4818 prediction rows на систему |
| Исторические артефакты V1–V8 | Хэши до и после совпали |
| Outer-EVAL leakage в selector | Не обнаружена |
| Seed isolation | PASS |
| Общие SELECT splits для configs/seeds | PASS |
| SciER cross-cell embargo | PASS |
| Exact cross-family document overlap | 0 совпадений |
| Runtime gold leakage | Не обнаружена |
| Ошибки/traceback в логе | Не обнаружены |

Replay проверил совпадение выбранных chunk IDs, финальных ответов, switch-флагов и Token F1 для H0–H3. Это подтверждает техническую совместимость V9 с V8, но не создаёт независимых научных данных.

## 2. Системы

- **S0** — исходный B0/top-1.
- **S1** — fixed exact H2 из V8: V7 S2 + FullLEX.
- **S2** — H2-ACS: одна конфигурация выбирается на TRAIN-only SELECT_OOF отдельно для `(cell, outer_fold)` и применяется ко всем трём seeds.
- **D-A** — fixed H2 + Agreement, только диагностическая система.
- **D-P** — fixed H2 + PairLEX, diagnostic only.
- **D-AP** — fixed H2 + Agreement + PairLEX, diagnostic only.

Существенная особенность контракта: exact H2 уже содержит полный PairLEX group (`pairlex`, `pairlex_x_margin`, `qarel_x_pairlex`). Поэтому `D-P ≡ S1`, а `D-AP ≡ D-A`. V9 не создаёт новой PairLEX-абляции и не даёт новой оценки его добавочной ценности.

## 3. Основные результаты Token F1

| Cell | S0 | S1/H2 | Δ S1−S0, 95% CI | S2/ACS | Δ S2−S0, 95% CI | Δ S2−S1, 95% CI |
|---|---:|---:|---|---:|---|---|
| SciREX | 27.3584 | 33.3152 | +5.9568, [+3.1795; +8.7217] | 33.3152 | +5.9568, [+3.1795; +8.7217] | +0.0000, [+0.0000; +0.0000] |
| SciER 384/50 | 58.7635 | 58.8360 | +0.0725, [-1.1141; +1.2847] | 58.8360 | +0.0725, [-1.1141; +1.2847] | +0.0000, [+0.0000; +0.0000] |
| SciER 128/24 | 56.1123 | 58.2551 | +2.1428, [-0.1456; +4.4106] | 60.3005 | +4.1882, [+1.8538; +6.6289] | +2.0454, [+0.7684; +3.4460] |
| RuSERRC | 61.2861 | 61.7345 | +0.4484, [-0.4538; +1.4052] | 61.7345 | +0.4484, [-0.4538; +1.4052] | +0.0000, [+0.0000; +0.0000] |
| SciMDIX | 57.3307 | 60.1185 | +2.7878, [-0.7949; +6.5284] | 60.1185 | +2.7878, [-0.7949; +6.5284] | +0.0000, [+0.0000; +0.0000] |

Интерпретация по cells:

- **SciREX:** H2 даёт самый крупный базовый выигрыш, +5.9568, с полностью положительным CI. ACS не включает Agreement ни на одном outer fold, поэтому S2=S1.
- **SciER 128/24:** H2 даёт +2.1428 с CI, слегка пересекающим ноль. ACS включает Agreement в трёх outer folds и увеличивает общий выигрыш против B0 до +4.1882; прирост над H2 статистически положителен.
- **SciER 384/50:** H2 почти нейтрален, +0.0725, с широким CI. ACS везде возвращается к H2.
- **RuSERRC:** H2 даёт небольшой положительный point estimate +0.4484; LCB −0.4538 формально проходит мягкий порог −0.5, но CI включает ноль. ACS везде оставляет H2.
- **SciMDIX:** H2 даёт заметный point estimate +2.7878, но CI широк из-за малого/гетерогенного корпуса; LCB −0.7949 нарушает универсальный gate. ACS везде оставляет H2.

## 4. Глобальная family-balanced оценка

SciER 128/24 и 384/50 объединены в одну corpus family и пересэмплировались согласованно по canonical document IDs.

| Контраст | Family-balanced ΔF1 | 95% CI | SciER-family Δ |
|---|---:|---|---:|
| S1-S0 | +2.5752 | [+1.3483; +3.8021] | +1.1076 |
| S2-S0 | +2.8308 | [+1.5973; +4.0448] | +2.1303 |
| S2-S1 | +0.2557 | [+0.0967; +0.4280] | +1.0227 |

Все три глобальных CI выше нуля. Для S2−S1 aggregate gain также превышает preregistered +0.20, но весь прирост приходит из SciER-128. Поэтому aggregate criterion выполнен, а критерий «не менее двух строго улучшившихся cells» — нет.

## 5. Почему universal gate не пройден

| Условие | S1 | S2 |
|---|---|---|
| ΔF1 против S0 > 0 на всех пяти cells | PASS | PASS |
| LCB95 > −0.5 на всех пяти cells | **FAIL:** SciER-384, SciMDIX | **FAIL:** SciER-384, SciMDIX |
| Evidence safety на всех cells | PASS | PASS |
| Non-degeneracy против S0 на всех cells | PASS | PASS |
| S2 не хуже S1 ни на одной cell | — | PASS |
| S2 incremental evidence safety | — | PASS |
| Family-balanced S2−S1 ≥ +0.20 | — | PASS (+0.2557) |
| Не менее двух cells имеют S2−S1 > 0 | — | **FAIL: только SciER-128** |

Для S1 нижние границы равны −1.1141 на SciER-384 и −0.7949 на SciMDIX. Для RuSERRC LCB равен −0.4538 и формально укладывается в мягкий порог. Строгий критерий `LCB95 > 0` выполняется у S1 только на SciREX, а у S2 — на SciREX и SciER-128.

## 6. Поведение component selector

| Cell | C0/H2 | CA/H2+A | CP | CAP | Entropy, bits |
|---|---:|---:|---:|---:|---:|
| SciREX | 5 | 0 | 0 | 0 | -0.000 |
| SciER 384/50 | 5 | 0 | 0 | 0 | -0.000 |
| SciER 128/24 | 2 | 3 | 0 | 0 | 0.971 |
| RuSERRC | 5 | 0 | 0 | 0 | -0.000 |
| SciMDIX | 5 | 0 | 0 | 0 | -0.000 |

Всего: **C0 — 22/25**, **CA — 3/25**, CP/CAP — 0. На английских cells CA выбран 3 раза из 15; на русских — 0 раз из 10. Это наблюдаемый результат TRAIN-only selection, а не языковое правило.

CA выбран только для SciER-128 на folds 0, 1 и 4. TRAIN-only приросты CA над C0 на этих folds были +5.5798, +4.4091 и +3.4180; все safety, stability и activity gates прошли. Выбор устойчив к diagnostic EPS=0.0/0.1/0.2/0.3: на этих folds CA/CAP остаются допустимыми, а на остальных folds ни одна optional-конфигурация не становится допустимой.

### Полный fold-level аудит Agreement-кандидата

| Cell | Fold | Выбор | SELECT Δ(CA−C0) | 4 SELECT-fold deltas | Stable changes | Непройденные gates | Outer oracle | Regret |
|---|---:|---|---:|---|---:|---|---|---:|
| SciREX | 0 | C0 | +1.1643 | +1.879, +1.980, -0.066, +0.735 | 99/5 | incremental_hn_safety | CA | 2.9519 |
| SciREX | 1 | C0 | +3.5612 | +2.403, +3.357, +3.329, +5.050 | 89/5 | incremental_hn_safety | CA | 1.3344 |
| SciREX | 2 | C0 | +1.7577 | +0.886, -1.105, +4.283, +3.242 | 81/5 | worst_fold_guard, incremental_hit_safety | CA | 6.1811 |
| SciREX | 3 | C0 | +0.8405 | +0.134, +2.352, -0.860, +1.392 | 69/5 | worst_fold_guard, incremental_hn_safety | CA | 3.0198 |
| SciREX | 4 | C0 | -1.4474 | -1.350, +1.512, +1.468, -7.137 | 67/5 | minimum_gain, fold_sign_stability, worst_fold_guard, incremental_hit_safety, absolute_hn_safety | CA | 1.3436 |
| SciER 384/50 | 0 | C0 | -0.2942 | -2.338, -0.189, -0.701, +0.857 | 21/5 | minimum_gain, fold_sign_stability, worst_fold_guard, incremental_hit_safety, absolute_hit_safety | C0 | 0.0000 |
| SciER 384/50 | 1 | C0 | -0.0644 | -0.337, +2.549, -0.899, -1.098 | 23/5 | minimum_gain, fold_sign_stability, worst_fold_guard, incremental_hit_safety, incremental_hn_safety, absolute_hit_safety | C0 | 0.0000 |
| SciER 384/50 | 2 | C0 | -1.3176 | +0.000, -0.582, -0.060, -3.044 | 7/5 | minimum_gain, fold_sign_stability, worst_fold_guard, incremental_hit_safety, absolute_hit_safety | CA | 0.8395 |
| SciER 384/50 | 3 | C0 | -1.1823 | +0.623, -0.744, -2.161, -1.481 | 21/5 | minimum_gain, fold_sign_stability, worst_fold_guard, incremental_hit_safety, absolute_hit_safety | CA | 0.7716 |
| SciER 384/50 | 4 | C0 | +0.0562 | +2.005, -0.957, -0.113, -0.185 | 21/5 | minimum_gain, fold_sign_stability, worst_fold_guard | CA | 2.4813 |
| SciER 128/24 | 0 | CA | +5.5798 | +5.361, +5.268, +8.101, +4.208 | 81/5 | none | CA | 0.0000 |
| SciER 128/24 | 1 | CA | +4.4091 | +9.943, +3.401, +4.743, +1.756 | 53/5 | none | CA | 0.0000 |
| SciER 128/24 | 2 | C0 | +1.3393 | -3.445, +4.641, +2.855, -0.128 | 45/5 | fold_sign_stability, worst_fold_guard | CA | 0.6615 |
| SciER 128/24 | 3 | C0 | +2.5083 | -0.300, +4.937, +0.106, +3.684 | 72/5 | incremental_hit_safety | CA | 2.4160 |
| SciER 128/24 | 4 | CA | +3.4180 | +4.980, +1.440, +5.013, +2.793 | 50/5 | none | CA | 0.0000 |
| RuSERRC | 0 | C0 | -0.2934 | -0.507, -0.672, +0.090, +0.000 | 5/5 | minimum_gain, fold_sign_stability, worst_fold_guard, incremental_hit_safety, absolute_hit_safety | CA | 0.4854 |
| RuSERRC | 1 | C0 | -0.2041 | -1.142, +0.140, +0.611, -0.350 | 6/5 | minimum_gain, fold_sign_stability, worst_fold_guard | C0 | 0.0000 |
| RuSERRC | 2 | C0 | -0.0487 | +0.289, +0.198, +0.085, -0.912 | 9/5 | minimum_gain, worst_fold_guard | CA | 0.4884 |
| RuSERRC | 3 | C0 | +0.1505 | -0.822, +0.667, +0.411, +0.460 | 1/5 | minimum_gain, worst_fold_guard, activity | C0 | 0.0000 |
| RuSERRC | 4 | C0 | +0.4818 | -0.331, -0.310, +1.056, +1.614 | 1/5 | fold_sign_stability, activity | C0 | 0.0000 |
| SciMDIX | 0 | C0 | -0.3518 | -1.099, +0.060, -0.053, +0.033 | 5/5 | minimum_gain, fold_sign_stability, worst_fold_guard, incremental_hit_safety, incremental_hn_safety | C0 | 0.0000 |
| SciMDIX | 1 | C0 | -0.8213 | -0.395, +0.000, -1.358, -1.961 | 2/5 | minimum_gain, fold_sign_stability, worst_fold_guard, incremental_hit_safety, absolute_hn_safety, activity | C0 | 0.0000 |
| SciMDIX | 2 | C0 | -0.3107 | -3.553, +2.755, -0.187, +1.139 | 5/5 | minimum_gain, fold_sign_stability, worst_fold_guard | C0 | 0.0000 |
| SciMDIX | 3 | C0 | -0.3731 | -0.327, -0.120, +0.170, -1.067 | 4/5 | minimum_gain, fold_sign_stability, worst_fold_guard, incremental_hit_safety, activity | C0 | 0.0000 |
| SciMDIX | 4 | C0 | -1.4125 | +1.394, -5.394, -2.824, -0.361 | 4/5 | minimum_gain, fold_sign_stability, worst_fold_guard, incremental_hit_safety, activity | CA | 0.9543 |

Selector ведёт себя консервативно. Особенно показателен SciREX: fixed Agreement на outer EVAL полезен, но TRAIN-only CA во всех пяти folds нарушал хотя бы один preregistered guard — чаще incremental HN safety или worst-fold stability. Поэтому не включать Agreement было корректным следованием протоколу, хотя post-hoc oracle показывает большой упущенный F1.

## 7. Selector regret

| Cell | Средний regret | Максимальный regret | Folds с нулевым regret |
|---|---:|---:|---:|
| SciREX | 2.9661 | 6.1811 | 0/5 |
| SciER 384/50 | 0.8185 | 2.4813 | 2/5 |
| SciER 128/24 | 0.6155 | 2.4160 | 3/5 |
| RuSERRC | 0.1948 | 0.4884 | 3/5 |
| SciMDIX | 0.1909 | 0.9543 | 4/5 |

Outer oracle — только диагностический инструмент и не участвовал в selection. Самый большой headroom наблюдается на SciREX: средний regret 2.9661, максимум 6.1811. Это означает, что Agreement потенциально полезен, но нынешние TRAIN-only признаки устойчивости и evidence safety не умеют надёжно разрешать его включение на SciREX.

На SciER-128 selector совпал с oracle на folds 0, 1 и 4. На folds 2 и 3 он отказался от CA из-за worst-fold/sign stability и incremental Hit@1 safety соответственно; outer diagnostic впоследствии показал пользу CA, дав regret 0.6615 и 2.4160. Это цена консервативного отбора.

## 8. Fixed diagnostics: Agreement и PairLEX

| Cell | Δ D-A−S1 | 95% CI | ΔHit@1, pp | ΔHN@1, pp |
|---|---:|---|---:|---:|
| SciREX | +2.9591 | [+1.5558; +4.3717] | +2.355 | +0.662 |
| SciER 384/50 | -0.2436 | [-1.4633; +1.0433] | -1.185 | +0.000 |
| SciER 128/24 | +2.6609 | [+0.7247; +4.5940] | +2.411 | +0.000 |
| RuSERRC | -0.2207 | [-0.6738; +0.1982] | -0.131 | -0.131 |
| SciMDIX | -0.5867 | [-1.7212; +0.5217] | -0.541 | +0.180 |

Agreement уверенно полезен на SciREX и SciER-128, нейтрален/отрицателен на SciER-384 и обоих русских датасетах. Следовательно, это не просто «английский компонент»: на длинном SciER-384 он не помогает. На SciREX прирост F1 сопровождается ростом HN@1 на +0.662 pp относительно H2, что превышает incremental budget +0.50 pp и объясняет часть отказов selector.

PairLEX нельзя интерпретировать как проверенный optional-компонент V9: его incremental schema пуста, поскольку он уже встроен в H2. Все D-P результаты точно совпадают с S1, а D-AP — с D-A.

## 9. Evidence metrics

| Cell | S0 Hit@1 | S2 Hit@1 | Δ, pp | S0 HN@1 | S2 HN@1 | Δ, pp |
|---|---:|---:|---:|---:|---:|---:|
| SciREX | 36.72 | 40.18 | +3.458 | 4.86 | 5.22 | +0.368 |
| SciER 384/50 | 50.22 | 51.11 | +0.889 | 1.33 | 0.89 | -0.444 |
| SciER 128/24 | 50.64 | 56.60 | +5.957 | 0.00 | 0.00 | +0.000 |
| RuSERRC | 63.39 | 63.19 | -0.197 | 12.27 | 12.47 | +0.197 |
| SciMDIX | 48.11 | 50.81 | +2.703 | 14.05 | 12.79 | -1.261 |

S2 улучшает Hit@1 на четырёх cells и слегка снижает его на RuSERRC (−0.197 pp). HN@1 улучшается на SciER-384 и SciMDIX, остаётся нулевым на SciER-128 и слегка растёт на RuSERRC/SciREX; все изменения остаются внутри абсолютного safety budget. Относительно S1 ACS меняет evidence только на SciER-128: Hit@1 +0.709 pp, HN@1 без изменения.

## 10. Switch activity и non-degeneracy

| Cell | S1 switch rate | S1 stable switches | S2 switch rate | S2 stable switches | Stable S2≠S1 | S2−S1 non-degenerate |
|---|---:|---:|---:|---:|---:|---|
| SciREX | 58.79% | 273/5 | 58.79% | 273/5 | 0 | FAIL |
| SciER 384/50 | 22.07% | 31/3 | 22.07% | 31/3 | 0 | FAIL |
| SciER 128/24 | 56.17% | 145/3 | 61.84% | 166/3 | 40 | PASS |
| RuSERRC | 11.48% | 39/6 | 11.48% | 39/6 | 0 | FAIL |
| SciMDIX | 33.15% | 57/3 | 33.15% | 57/3 | 0 | FAIL |

S1 и S2 активны относительно B0 на всех cells. Дополнительная ACS-активность относительно H2 существует только на SciER-128: 40 устойчиво изменённых вопросов при требовании 3. На остальных cells S2=S1, поэтому отдельная component-selection activity закономерно нулевая.

Switch counts ниже считаются по prediction rows (три seeds), а не по независимым вопросам:

| Cell | System | Good | Bad | Neutral | Mean actual utility | Median | Large gain | Large harm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SciREX | S1 | 279 | 146 | 374 | 10.1318 | 0.0000 | 272 | 143 |
| SciREX | S2 | 279 | 146 | 374 | 10.1318 | 0.0000 | 272 | 143 |
| SciER 384/50 | S1 | 43 | 31 | 75 | 0.3283 | 0.0000 | 43 | 29 |
| SciER 384/50 | S2 | 43 | 31 | 75 | 0.3283 | 0.0000 | 43 | 29 |
| SciER 128/24 | S1 | 166 | 125 | 105 | 3.8148 | 0.0000 | 157 | 117 |
| SciER 128/24 | S2 | 199 | 124 | 113 | 6.7721 | 0.0000 | 188 | 114 |
| RuSERRC | S1 | 46 | 39 | 90 | 3.9049 | 0.0000 | 44 | 39 |
| RuSERRC | S2 | 46 | 39 | 90 | 3.9049 | 0.0000 | 44 | 39 |
| SciMDIX | S1 | 69 | 52 | 63 | 8.4089 | 0.0000 | 66 | 49 |
| SciMDIX | S2 | 69 | 52 | 63 | 8.4089 | 0.0000 | 66 | 49 |

## 11. Стабильность по outer folds и seeds

Формат ниже: `fold/seed: Δ(S1−S0), Δ(S2−S1)`.

| Cell | Outer folds | Seeds |
|---|---|---|
| SciREX | 0: +6.066, +0.000; 1: +4.555, +0.000; 2: +5.653, +0.000; 3: +12.121, +0.000; 4: +1.388, +0.000 | 42: +6.442, +0.000; 43: +5.816, +0.000; 44: +5.612, +0.000 |
| SciER 384/50 | 0: -3.616, +0.000; 1: +3.482, +0.000; 2: +0.473, +0.000; 3: +0.153, +0.000; 4: -0.136, +0.000 | 42: -2.300, +0.000; 43: +1.671, +0.000; 44: +0.846, +0.000 |
| SciER 128/24 | 0: +3.380, +3.807; 1: +1.564, +4.299; 2: +2.797, +0.000; 3: +5.203, +0.000; 4: -2.230, +2.121 | 42: +2.250, +3.692; 43: +4.785, +1.018; 44: -0.606, +1.427 |
| RuSERRC | 0: -0.283, +0.000; 1: -0.223, +0.000; 2: -0.190, +0.000; 3: +1.533, +0.000; 4: +1.374, +0.000 | 42: +0.221, +0.000; 43: +0.913, +0.000; 44: +0.212, +0.000 |
| SciMDIX | 0: +1.879, +0.000; 1: +1.900, +0.000; 2: +4.663, +0.000; 3: +0.041, +0.000; 4: +5.606, +0.000 | 42: +2.797, +0.000; 43: +2.665, +0.000; 44: +2.902, +0.000 |

Самые устойчивые H2-эффекты: SciREX (все folds/seeds положительны) и SciMDIX (все folds/seeds положительны, но дисперсия по документам даёт широкий bootstrap CI). На RuSERRC три folds отрицательны, на SciER-384 два folds отрицательны, а на SciER-128 seed 44 отрицателен для S1. ACS-добавка на SciER-128 положительна на всех трёх seeds и на всех трёх folds, где CA был выбран.

## 12. Post-hoc анализ по аспектам и K

### SciREX

- Аспекты: `MATERIAL` n=126: H2 +3.204, ACS +0.000; `METHOD` n=115: H2 +4.760, ACS +0.000; `METRIC` n=65: H2 +3.207, ACS +0.000; `TASK` n=147: H2 +10.469, ACS +0.000.
- K: K=4, n=1: H2 +0.000, ACS +0.000; K=5, n=452: H2 +5.970, ACS +0.000.

### SciER 384/50

- Аспекты: `MATERIAL` n=73: H2 +0.482, ACS +0.000; `METHOD` n=78: H2 -0.589, ACS +0.000; `TASK` n=74: H2 +0.365, ACS +0.000.
- K: K=4, n=6: H2 -3.086, ACS +0.000; K=5, n=219: H2 +0.159, ACS +0.000.

### SciER 128/24

- Аспекты: `MATERIAL` n=77: H2 +2.308, ACS +0.674; `METHOD` n=80: H2 +0.513, ACS +4.213; `TASK` n=78: H2 +3.651, ACS +1.177.
- K: K=5, n=235: H2 +2.143, ACS +2.045.

### RuSERRC

- Аспекты: `ADVANTAGE` n=53: H2 +0.749, ACS +0.000; `CONCLUSION` n=23: H2 +3.754, ACS +0.000; `CONTRIBUTION` n=50: H2 +3.656, ACS +0.000; `EXAMPLE` n=39: H2 +0.000, ACS +0.000; `GOAL` n=16: H2 +1.562, ACS +0.000; `METHOD` n=101: H2 -0.533, ACS +0.000; `TASK` n=104: H2 -1.006, ACS +0.000; `TOOL` n=53: H2 +2.725, ACS +0.000; `USE` n=69: H2 -1.335, ACS +0.000.
- K: K=1, n=73: H2 +0.000, ACS +0.000; K=2, n=63: H2 -0.269, ACS +0.000; K=3, n=58: H2 +2.049, ACS +0.000; K=4, n=91: H2 +2.125, ACS +0.000; K=5, n=223: H2 -0.303, ACS +0.000.

### SciMDIX

- Аспекты: `GOAL` n=27: H2 -4.222, ACS +0.000; `MATERIAL` n=10: H2 +0.000, ACS +0.000; `METHOD` n=70: H2 +7.194, ACS +0.000; `RESULT` n=19: H2 +6.946, ACS +0.000; `TASK` n=22: H2 +3.869, ACS +0.000; `TOOL` n=31: H2 -2.968, ACS +0.000; `USE` n=6: H2 +0.183, ACS +0.000.
- K: K=2, n=5: H2 +4.762, ACS +0.000; K=3, n=27: H2 +2.828, ACS +0.000; K=4, n=153: H2 +2.716, ACS +0.000.

Эти разрезы post-hoc и не использовались selector. На русских данных видна сильная аспектная неоднородность: RuSERRC теряет на METHOD/TASK/USE, SciMDIX — на GOAL/TOOL, но выигрывает на других аспектах. На SciER-128 добавка ACS особенно велика для METHOD (+4.213). Создавать aspect- или K-specific правила по этим данным нельзя без новой заранее зафиксированной версии.

## 13. Вторичные answer metrics

| Cell | System | Char F1, % | Unicode ROUGE-L, % | EM, % | Lemma F1 RU |
|---|---|---:|---:|---:|---:|
| SciREX | S0 | 52.10 | 33.87 | 15.82 | — |
| SciREX | S1 | 56.72 | 41.45 | 20.01 | — |
| SciREX | S2 | 56.72 | 41.45 | 20.01 | — |
| SciER 384/50 | S0 | 72.19 | 58.11 | 25.93 | — |
| SciER 384/50 | S1 | 72.44 | 58.16 | 26.22 | — |
| SciER 384/50 | S2 | 72.44 | 58.16 | 26.22 | — |
| SciER 128/24 | S0 | 71.01 | 55.85 | 23.97 | — |
| SciER 128/24 | S1 | 72.63 | 57.86 | 25.67 | — |
| SciER 128/24 | S2 | 74.41 | 60.04 | 30.92 | — |
| RuSERRC | S0 | 76.32 | 61.34 | 10.70 | 62.8319 |
| RuSERRC | S1 | 76.73 | 61.79 | 10.50 | 63.3497 |
| RuSERRC | S2 | 76.73 | 61.79 | 10.50 | 63.3497 |
| SciMDIX | S0 | 76.66 | 57.60 | 38.38 | 61.0072 |
| SciMDIX | S1 | 78.24 | 60.40 | 41.44 | 63.5268 |
| SciMDIX | S2 | 78.24 | 60.40 | 41.44 | 63.5268 |

Вторичные метрики согласуются с основным результатом: заметное дополнительное улучшение S2 относительно S1 возникает только на SciER-128. На RuSERRC H2 повышает lemma F1 и Unicode ROUGE-L, но EM немного снижается. На SciMDIX H2 улучшает lemma F1, Unicode ROUGE-L и EM. BERTScore отсутствует в историческом контракте и не вычислялся.

## 14. Коэффициенты

| Признак | Median standardized coefficient | IQR | Positive rate | n |
|---|---:|---|---:|---:|
| `delta_fulllex` | +5.4546 | [+3.7828; +10.1125] | 100.00% | 150 |
| `fulllex_x_margin` | +0.6145 | [-1.5498; +2.4160] | 54.67% | 150 |
| `qarel_x_fulllex` | +0.5808 | [-0.2674; +1.7942] | 66.67% | 150 |
| `pairlex` | +1.8005 | [+0.5436; +3.6410] | 88.00% | 150 |
| `pairlex_x_margin` | +0.7528 | [+0.0137; +1.4728] | 76.00% | 150 |
| `qarel_x_pairlex` | -1.8644 | [-3.3088; -1.2197] | 0.00% | 150 |
| `delta_agreement` | +4.1412 | [+3.2373; +4.7007] | 100.00% | 9 |
| `agreement_x_margin` | +2.6160 | [+1.1598; +3.0141] | 88.89% | 9 |
| `fulllex_x_agreement` | -1.1777 | [-1.4158; -0.6563] | 22.22% | 9 |

`delta_fulllex` положителен во всех 150 сохранённых S1/S2 моделях и имеет медиану +5.4546. В девяти S2-моделях с выбранным Agreement (3 folds × 3 seeds) `delta_agreement` также всегда положителен, но взаимодействие `fulllex_x_agreement` преимущественно отрицательно. PairLEX-коэффициенты относятся к уже встроенному H2 и не являются отдельным V9-эффектом. Коэффициенты описательные и не использовались для component selection; 150 моделей не являются 150 независимыми статистическими наблюдениями.

## 15. Ресурсный benchmark

| Cell | Sample | S0 median, s | S1 median, s | S2 median, s | S2/S1 median | S2/S1 p95 | VRAM ratio | Meta params S1→S2 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SciREX | 90 | 0.284 | 1.315 | 1.288 | 0.979 | 0.979 | 1.000 | 36→36 |
| SciER 384/50 | 45 | 0.268 | 1.257 | 1.265 | 1.006 | 1.010 | 1.000 | 36→36 |
| SciER 128/24 | 47 | 0.081 | 0.341 | 0.341 | 1.000 | 0.997 | 1.000 | 36→42 |
| RuSERRC | 100 | 0.072 | 0.283 | 0.280 | 0.991 | 1.003 | 1.000 | 36→36 |
| SciMDIX | 37 | 0.079 | 0.265 | 0.269 | 1.012 | 1.020 | 1.000 | 36→36 |

На каждом вопросе S1 и S2 имеют одинаковое число QA и Q-A-ranker forwards; neural parameter count и peak VRAM совпадают. Все engineering targets выполнены: median ratio 0.979–1.012, p95 ratio 0.979–1.020, VRAM ratio 1.0. На SciER-128 S2 имеет 42 meta-параметра против 36 у S1 из-за Agreement; шесть дополнительных линейных коэффициентов пренебрежимо малы относительно neural stack.

При этом H2/ACS существенно дороже B0: median latency S1/S0 составляет 4.63× на SciREX, 4.69× на SciER-384, 4.22× на SciER-128, 3.93× на RuSERRC и 3.36× на SciMDIX. Причина — QA и Q-A scoring для top-K вместо одного B0-чанка. Вывод о «нулевой стоимости» относится только к S2 относительно H2, а не ко всему методу относительно B0.

Fresh Qwen comparison имеет статус `BLOCKED_NO_EXACT_FRESH_RUNNER`; BGE оставлен только contextual reference. Поэтому прямых численных выводов о скорости относительно них делать нельзя.

## 16. Научная интерпретация

1. **H2 остаётся наиболее устойчивым общим ядром.** Он положителен по point estimate на всех cells, evidence-safe и активен, а family-balanced эффект уверенно выше нуля. Но predeclared универсальный per-cell gate он не прошёл.
2. **ACS работает, но пока локально.** На SciER-128 TRAIN-only решение переносится на outer EVAL и даёт содержательный выигрыш. На четырёх остальных cells selector безопасно возвращается к H2.
3. **Agreement зависит от типа задачи, а не только от языка.** Он полезен на SciREX и SciER-128, но не на SciER-384. Различие между двумя SciER chunking regimes особенно важно: короткие 128/24 чанки выигрывают, длинные 384/50 — нет.
4. **Selector слишком консервативен для SciREX.** Это снижает риск по HN@1, но оставляет большой диагностический headroom. Ослаблять gates после просмотра результатов нельзя; это потребовало бы новой версии и независимой проверки.
5. **Русские данные не получили добавки от Agreement.** На RuSERRC и SciMDIX CA имел отрицательный aggregate effect, поэтому возврат к C0 обоснован. При этом сам H2 даёт +0.4484 и +2.7878 соответственно, но неопределённость остаётся большой.
6. **V9 не проверил новый PairLEX-компонент.** PairLEX уже находился в exact H2, поэтому факторный план фактически выродился из четырёх уникальных конфигураций в две: H2 и H2+Agreement.

## 17. Ограничения

- Все пять corpora уже использовались при развитии V4.4–V8; результаты остаются development evidence.
- Две SciER cells имеют общую corpus lineage; глобальный bootstrap это учитывает, но их нельзя считать независимыми датасетами.
- Cell-level CI широки на SciER-384 и SciMDIX; положительный global CI не заменяет универсальный per-cell критерий.
- Outer oracle и aspect/K diagnostics post-hoc; их нельзя использовать для изменения V9 после результата.
- PairLEX не имеет независимого incremental contrast в текущем exact-H2 контракте.
- Ресурсный benchmark выполнен на одном окружении и фиксированной выборке fold0; абсолютные времена не переносятся напрямую на другое оборудование.

## 18. Итоговое решение

Протокол отработал корректно и должен интерпретироваться буквально:

> V9 не выбрал универсального development-кандидата. H2 показывает сильный глобальный и положительный во всех cells point-estimate pattern, но не проходит uncertainty gate на SciER-384 и SciMDIX. ACS надёжно улучшает H2 только на SciER-128 и потому не удовлетворяет требованию минимум двух улучшенных cells. Следующий доказательный шаг — не подстройка V9 на этих же outer folds, а freeze гипотезы и проверка на действительно untouched confirmation data либо явное ограничение области применимости метода.

## Источники внутри run

- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/analysis/primary_metrics.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/analysis/paired_bootstrap.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/analysis/global_family_bootstrap.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/analysis/development_decision.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/analysis/component_selection_analysis.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/analysis/selector_regret.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/analysis/non_degeneracy.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/analysis/switch_analysis.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/analysis/fold_stability.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/analysis/seed_stability.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/analysis/aspect_analysis.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/analysis/k_analysis.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/analysis/secondary_answer_metrics.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/analysis/coefficient_analysis.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/resource_benchmark/summary.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/audit/h0_replay.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/audit/h1_replay.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/audit/h2_replay.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/audit/h3_replay.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/audit/regression_summary.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/audit/pairlex_group.json`
- `artifacts/qa_chunk_selection_v9_h2_acs/run_20260924_134400/COMPLETED.json`
