# Материалы к статье 1: разработка UnTIE — безопасного reader-aware выбора фрагмента

Дата сборки: **2026-10-08**  
Статус: **исследовательское досье для подготовки статьи, не готовая рукопись**  
Финальная архитектура проекта: **UnTIE H3 Consistency Gate / `H3-CONSISTENCY-GATE-3`**

## 1. Назначение файла

Файл собирает максимально полный материал о разработке метода UnTIE: постановку задачи, эволюцию вариантов V1–V10, отрицательные результаты, финальную архитектуру, official-split benchmark, абляции, статистические проверки, ограничения и упомянутую в проекте литературу. Абсолютные числа из исторических development-прогонов и official TEST намеренно не смешиваются: в ходе проекта менялись данные, разбиения, B0, вопросы и чанкирование.

Для статьи полезно различать четыре уровня доказательности:

1. **Предварительно заданный official-split TEST** — главный сопоставимый benchmark H2-family.
2. **Untouched confirmation** — SciREX, SciER и SciMDIX не использовались в V1–V10 до итогового benchmark.
3. **Development evidence** — результаты, по которым строилась архитектура; они объясняют выбор компонентов, но не являются независимым подтверждением.
4. **Post-hoc evidence** — эксперименты после раскрытия official TEST, включая финальный consistency gate. Их следует описывать как выбор финальной архитектуры, требующий новой закрытой проверки.

RuSERRC в итоговом benchmark является **exposed**: он использовался при разработке. SciER-128 и SciER-384 построены из одних документов, поэтому в общей метрике сначала объединяются в одно семейство.

## 2. Возможная центральная формулировка статьи

**Рабочая идея.** Для извлечения ответа из известного длинного документа недостаточно выбрать семантически близкий чанк. Полезность кандидата определяется тем, способен ли reader получить из него правильный ответ, а риск переключения с сильного базового ранкера должен оцениваться отдельно. UnTIE решает эту задачу как безопасный meta-reranking поверх dataset/language-specific B0.

**Основной вклад, который подтверждают эксперименты.**

- единая reader-aware схема выбора evidence для английских и русских научных текстов;
- раздельное прогнозирование ожидаемой пользы переключения и риска ухудшения;
- лексические, QA и agreement-признаки, рассчитываемые на общем frozen top-K;
- fallback к B0 при отсутствии безопасной альтернативы;
- общий dataset-level consistency gate между H2 и H3 без ручных исключений по датасетам;
- подробная отрицательная доказательная база: raw attention, сложные learned routers, question-conditioned residual, universal gates и rank-risk модели не дали универсального улучшения.

## 3. Задача и обозначения

Для вопроса `q`, документа `d` и множества его чанков `C(d)` базовый ранкер B0 возвращает top-K кандидатов. Reader вычисляет для каждого кандидата ответ, confidence и производные сигналы. Система выбирает один чанк и извлечённый из него ответ.

Основная метрика — **answer Token F1, 0–100**. Дополнительно использовались:

- Exact Match;
- Char F1 и Unicode ROUGE-L;
- Lemma F1 для русских данных как дополнительная, а не основная метрика;
- Hit@1 и Hit@K по evidence;
- HN@1 — доля hard negative на первой позиции;
- pairwise accuracy и диагностические oracle-метрики.

Результаты трёх фиксированных seeds `42, 43, 44` сначала усредняются внутри вопроса. Выбор лучшего seed по TEST запрещён. Bootstrap выполняется на уровне документов; в итоговом benchmark использовано 10 000 итераций. Две SciER-cell ресэмплируются согласованно.

## 4. Базовая архитектура H2/H3

### 4.1. Общий pipeline

```text
документ
  -> детерминированные чанки
  -> dataset/language-specific B0
  -> frozen top-K
  -> один и тот же QA reader для кандидатов данной cell
  -> признаки альтернатив относительно B0 top-1
  -> utility model + harm model
  -> порог безопасного переключения
  -> новый кандидат либо fallback к B0
  -> ответ reader выбранного чанка
```

Языково- и корпусно-специфичны B0, reader, вопросы, K и чанкирование. Механика обучения meta-reranker, разделение utility/harm и правила безопасности общие.

### 4.2. H2

H2 — safe meta-reranker поверх B0. Для альтернативного кандидата относительно B0 top-1 используются:

- `delta_b0_norm` — изменение нормализованного score B0;
- `delta_qarel_raw` — разность QA relevance;
- `pairlex` и `delta_fulllex` — локальное и полное лексическое согласование;
- нормализованный ранг кандидата, эффективный K, gap top-1/top-2 и энтропия B0;
- валидность ответов кандидата и B0, признак двух пустых ответов;
- взаимодействия QA relevance, PairLEX, FullLEX и B0 margin.

Ridge оценивает ожидаемое изменение Token F1 при переключении. Logistic model оценивает вероятность вреда. Кандидат допускается только при выполнении двух условий:

```text
predicted_utility >= tau_utility
predicted_harm    <= tau_harm
```

Если допустимого кандидата нет, сохраняется B0. Зафиксированные safety-ограничения при подборе порогов: падение Hit@1 не более `0.5` процентного пункта и рост HN@1 не более `1.0` процентного пункта.

### 4.3. H3

H3 полностью сохраняет H2 и добавляет три reader-agreement признака:

- `delta_agreement`;
- `agreement_x_margin`;
- `fulllex_x_agreement`.

H3 является H2 с одной дополнительной группой признаков, а не отдельным несвязанным методом.

### 4.4. Финальное смешивание и consistency gate

Для H2 и H3 независимо обучаются utility и harm. Их прогнозы интерполируются общим для датасета коэффициентом:

```text
utility = utility_H2 + lambda * (utility_H3 - utility_H2)
harm    = harm_H2    + lambda * (harm_H3    - harm_H2)
lambda in {0.00, 0.25, 0.50, 0.75, 1.00}
```

`lambda=0` соответствует H2, `lambda=1` — исходному H3. Одно значение выбирается по среднему document-grouped OOF Token F1 трёх seeds. Tie-break: меньшая switch rate, затем большее lambda.

Финальный gate сравнивает выбор на TRAIN и после TRAIN+VAL refit:

```text
if lambda_train == lambda_refit:
    lambda_final = lambda_refit
else:
    lambda_final = 1.0
```

Gate не содержит условий по имени датасета. При нестабильности он возвращается к H3. Доверительный интервал в gate не входит.

### 4.5. Сетки обучения

- Ridge `alpha`: `{0.1, 1, 10, 100}`;
- logistic `C`: `{0.01, 0.1, 1}`;
- utility threshold: `{0, 0.25, 0.5, 1, 2}`;
- harm threshold: `{0.1, 0.2, 0.3, 0.4, 0.5}`;
- lambda: `{0, 0.25, 0.5, 0.75, 1}`.

Подбор выполняется внутри document-grouped OOF соответствующего корпуса. Параметры одного языка или датасета напрямую не переносятся на другой.

## 5. Хронология экспериментов и полученные знания

### 5.1. V1: постановка исследовательской программы

V1 зафиксировал основные направления: answerability, denoised hard negatives, global passage comparison, структурные признаки и aggregation. Это был проект эксперимента, а не итоговая оценка. Важное ограничение, сохранённое в последующих версиях: близкий по смыслу фрагмент нельзя автоматически считать отрицательным, если он может содержать альтернативное допустимое evidence.

### 5.2. V2: первый полный разбор узкого места

На DEVELOPMENT было 77 answerable-вопросов. Reader-oracle по доступным положительным чанкам получил Token F1 `71.86`; для `76.6%` вопросов существовал положительный кандидат с F1 не ниже 50. Реальный выбранный ранкер дал лишь `25.71`. Это показало, что запас находился прежде всего в выборе чанка.

R3_3 имел QA F1 `28.23`, но по заранее принятому pairwise-first критерию был выбран R3_2. Его разница с B2 составила `−2.42`, 95% CI `[−12.58; +7.54]`: доказательств улучшения не было. Конкатенация фиксированного числа чанков помогала калибровке, но адаптивное `M` на DEV дало `21.62`.

### 5.3. V3: answerability, pairwise/listwise и attention

Универсум: 453 unlocked answerable-вопроса, 240 документов, 5 document-level folds, seeds 42/43/44, 1359 seed-строк.

| Система/метрика | Значение |
|---|---:|
| B0 Token F1 | 27.36 |
| EM | 20.75 |
| Hit@1 | 36.72% |
| Hit@5 | 79.03% |
| Hit@10 | 92.20% |
| HN@1 | 4.86% |
| pairwise clean | 75.14% |
| F1 при положительном top-1 | 61.70 |
| лучший positive oracle | 73.01 |

Extended answerability дал `26.83`, pairwise — `25.86`, listwise — `25.35`; listwise не прошёл safeguard. Структурный вариант дал `26.15`, но pairwise quality упала до `68.66%`. Span-ranker дал `24.40` против B0 seed-42 `26.46`.

Attention-only оказался слабым: HN@1 `18.98%`, pairwise `51.5%`; AUROC выбранного attention-сигнала `0.519`. Важная оговорка: target-mismatch overlay фактически не содержал размеченных примеров (`0/400`), поэтому заявленная target-mismatch гипотеза не была проверена. Кампания V3.1 подготовила инфраструктуру ручной разметки, но без завершённой human annotation не считается полноценным экспериментальным результатом.

### 5.4. V4: семь направлений и появление reader-centered reranking

B0 оставался `27.36`. KeyBERT-варианты не улучшили качество; у нескольких конфигураций оптимальный вес стал нулевым, то есть они точно воспроизвели B0.

- Answer aggregation V2-A4: `30.26`, прирост `+2.90`, CI `[1.32; 4.55]`; выбранный чанк не менялся.
- Reader-centered V6: `35.04`, прирост `+7.69`, CI `[4.79; 10.59]`, Hit@1 `42.60%`, HN@1 `5.15%`.
- V3-A3 показал `38.42` в смешанной in-sample оценке, но это число признано невалидным; честный eval run-0 дал `29.90` против B0 `30.67`, разница `−0.78`, CI `[−8.24; 6.48]`.
- Ownership diagnostic AUROC `0.543`; V7 ухудшил B0 на `−5.96`.
- Small head V5 дал pool-прирост `+1.97`, но CI пересёк ноль; OOF run-0 `+0.55`.

Технические ограничения V4, которые нельзя скрывать в статье: для trainable/CAL выполнялся один outer run вместо пяти; `inner_crossfit=false`; первоначальное bootstrap-сопоставление seed было ошибочным; существовала несовместимость сериализации pairwise score; F1-порог `0.8` применялся к шкале 0–100. Эти проблемы делают V4 источником гипотез, а не финальной confirmatory оценкой.

### 5.5. V4.1: декомпозиция V6

V6-K5 дал OOF Token F1 `34.6249`, разницу с B0 `+7.26`, CI `[4.38; 10.17]`, Hit@1 `42.90%`, HN@1 `5.45%`. V6-K10 дал `37.138`, разницу `+9.75`, CI `[6.33; 13.05]`, но преимущество над K5 не было статистически подтверждено; K5 сохранили из-за заранее заданного протокола и более простой финальной фиксации.

Knockout-анализ показал, что основная польза идёт от reader agreement. Answer aggregation C1 (`34.65`) почти совпала с V6.

### 5.6. V4.2: attention heads против reader signals

Референс V6: `34.62`. CORE1 `32.93`, CORE2 `32.47`. Диагностический reader-only CORE3 `35.44`, но разница с V6 `+0.78`, CI `[−1.76; 3.30]`, HN@1 `6.18%`. ATT1 дал `34.7616`: выигрыш минимален, интервал пересёк ноль, HN@1 `6.11%`. Выбранные attention heads не дали достаточного дополнительного сигнала.

### 5.7. V4.3: semantic, lexical и adaptive reader signals

| Вариант | Token F1 | Разница с V6 | 95% CI |
|---|---:|---:|---:|
| V6 | 34.6249 | — | — |
| SEM1 | 34.6923 | +0.068 | [−0.678; 0.815] |
| LEXCORE | 36.58 | +1.918 | [−0.015; 3.888] |
| SEMCORE | 37.33 | +2.687 | [0.892; 4.457] |
| ADAPT1 | 36.02 | +1.379 | [0.496; 2.296] |

SEMCORE был лучше V6, но не был надёжно лучше LEXCORE: `+0.768`, CI `[−0.900; 2.451]`. Поэтому превосходство semantic mechanism над lexical не подтвердилось.

### 5.8. V4.4: adaptive reader consensus на нескольких корпусах

На SciREX: B0 `27.358`, V6 `34.625`, SEM-FIX `37.511`, LEX-FIX `36.982`, ASEM `35.111`, ALEX `34.774`. Адаптивный router совершил 32 полезных и 14 вредных переключений, но пропустил 136 потенциально полезных. ASEM улучшал V6 лишь на `0.486`, одновременно теряя Hit@1 на `1.03 pp`, поэтому safety gate не прошёл.

На SciER-384: B0 `51.9647`, V6 `58.7635`, разница `+6.7869`, CI `[3.3272; 10.3346]`; LEX `60.825`, прирост `+8.8494`, CI `[2.5620; 15.1781]`.

На SciER-128: B0 `52.8311`, V6 `59.8713`, разница `+7.0379`, CI `[2.2441; 11.6361]`; SEM `62.5857`, прирост `+9.7672`; LEX `62.3998`, прирост `+9.5663`.

На SciMDIX в первой конфигурации: B0 `52.5104`, V6 `52.5285`, разница `+0.0248`, CI `[−2.3884; 2.5885]`; SEM/LEX откатились к B0, ASEM/ALEX дали `51.8543`. Consensus-надстройка не подтвердилась.

Важный confound: на SciER выбор вопроса Q3 сам давал часть улучшения. Для SciER-384 V6 мог не менять ranking, но Token F1 возрастал примерно на 6.8 пункта из-за смены формулировки вопроса. Поэтому V4.4 измерял эффект полного pipeline, а не только reranking.

Прямой перенос английской конфигурации на RuSERRC дал B0 `52.768`, V6 `51.030`, SEM `47.504`, LEX `46.455`, ASEM `50.266`. Безопасная русская настройка q0/K5 дала `53.407`, прирост `+0.634`, CI пересёк ноль.

### 5.9. Русские ranker-эксперименты и V4.5

Замена ранкера оказалась сильнее consensus-компонента. Frozen BGE поднял RuSERRC B0 с `52.768` до `69.954`: `+17.167`, CI `[14.066; 20.152]`, Hit@1 `72.57%`, HN@1 `6.56%`; второй этап выбрал точный fallback.

Дополнительные ранкеры на RuSERRC:

- `ARGA100/ru-reranker-modernbert-small`: B0 `59.833`;
- `DiTy/cross-encoder-russian-msmarco`: B0 `61.286`;
- BGE оставался сильнейшим в этой серии.

На SciMDIX ModernBERT B0 дал `48.951`, DiTy — `57.331`; DiTy выше на `8.342`, CI `[2.599; 14.141]`. DiTy + K4 + ASEM-Q5 достиг `59.625`: `+2.280` над DiTy, CI `[0.770; 4.088]`, и `+7.137` над старым B0, CI `[1.171; 13.082]`. Механизм использовал source-cluster support (`gamma`), тогда как agreement-вес `beta=0`; выбранным всегда оставался Q0. Это подтверждает адаптивный контроль source-support, но не самостоятельную силу Q5-consensus.

V4.5 подтвердил, что сильный BGE B0 доминирует на RuSERRC. Gold answer oracle оставался высоким: `88.137` для RuSERRC и `91.746` для SciMDIX. При переходе DiTy→BGE на RuSERRC top-1 менялся в 579 из 1524 seed-строк (`38%`), чистый баланс положительных переходов `+140`; conditional reader F1 после попадания в положительный чанк почти не менялся (`85.78` против `86.15`). Следовательно, основное улучшение дал ranking, а не reader.

### 5.10. V5: question-conditioned residual

Primary A1 не получил универсальной поддержки. Лучшие наблюдения были вторичными:

- SciREX A2 `31.337`, `+3.9786`, CI `[1.3348; 6.6289]`;
- SciER-384 остался на B0/V6 reference `58.7635`;
- SciER-128 gated `56.7747`, `+0.6624`, CI пересёк ноль;
- RuSERRC `61.7363`, `+0.4503`, CI пересёк ноль;
- SciMDIX `60.9685`, `+3.6378`, CI `[−0.290; 7.559]`.

Вариант показал локальный потенциал, но не дал единого подтверждённого правила.

### 5.11. V6: universal UAR-G

Лучшие диагностические приросты: SciREX `+0.7014`, SciER-384 `+0.1376`, SciER-128 `+0.7362`, RuSERRC `+0.6074`, SciMDIX `+3.0732`. Ни один универсальный кандидат не прошёл общие условия. Вывод: общий gate без достаточной dataset-specific калибровки не переносится стабильно.

### 5.12. V7: progressive risk-aware policy

Primary S3 дал: SciREX `27.411`, SciER-384 `58.485`, SciER-128 `56.3982`, RuSERRC `61.6722`, SciMDIX `59.7338`. На SciMDIX разница `+2.4031`, CI `[0.2039; 4.7725]`; в среднем использовалось 1.477 QA-кандидата, максимум 3. Однако лучшие варианты различались по корпусам: local S2 на SciREX, pooled S3 на SciER-384, local S1 на SciER-128, pooled S1 на RuSERRC. Универсального победителя не было.

### 5.13. V8: формирование H2/H3

V8 дал ключевой переход к финальному семейству.

| Cell | H0 | H1 | H2 | H3 | H3−H0 |
|---|---:|---:|---:|---:|---:|
| SciREX | 27.3584 | 28.0499 | 33.3152 | 36.2743 | +8.9159 |
| SciER-128 | 56.1123 | 56.2752 | 58.2551 | 60.9160 | +4.8037 |
| SciER-384 | 58.7635 | 59.0491 | 58.8360 | 58.5924 | −0.1712 |
| RuSERRC | 61.2861 | 61.5682 | 61.7345 | 61.5138 | +0.2277 |
| SciMDIX | 57.3307 | 58.3047 | 60.1185 | 59.5318 | +2.2011 |

Agreement-вклад H3−H2: `+2.9591` SciREX, `+2.6609` SciER-128, `−0.2436` SciER-384, `−0.2207` RuSERRC, `−0.5867` SciMDIX. Family-balanced H3−H0 составил `+3.4152`, но universal gate не прошёл из-за отрицательного SciER-384, нижних границ CI и роста HN на SciREX примерно на 1.03 pp.

Альтернативная F5-core branch дала `34.799 / 60.2511 / 56.4921 / 60.9382 / 59.0459`; она заметно вредила SciREX и RuSERRC. H2/H3 признаны более содержательной линией.

### 5.14. V9: adaptive component selection

ACS оставил H2 без изменений на SciREX, SciER-384, RuSERRC и SciMDIX. На SciER-128 он поднял H2 с `58.2551` до `60.3005`, разница `+2.0454`, CI `[0.7684; 3.4460]`. Family gain был `+0.2557`; улучшилась только одна cell при требовании не менее двух. Универсальная гипотеза не прошла.

В official benchmark P3/H2-ACS воспроизвёл P2, но у RuSERRC обнаружено нарушение frozen selection: конфигурация была переопределена после refit. Поэтому P3 нельзя использовать как чистый confirmatory результат для этой cell.

### 5.15. V10: switch/rank/risk

Полная SRR-система S2 относительно H2/S1:

| Cell | H2/S1 | SRR/S2 | Разница |
|---|---:|---:|---:|
| SciREX | 33.315 | 29.112 | −4.203, CI [−6.645; −1.690] |
| SciER-384 | 58.836 | 57.892 | −0.944 |
| SciER-128 | 58.255 | 56.921 | −1.335 |
| RuSERRC | 61.734 | 61.422 | −0.313 |
| SciMDIX | 60.119 | 61.541 | +1.422, CI пересёк 0 |

Family-balanced разница `−1.058`, CI `[−2.124; −0.014]`. Evidence-risk помогал SciMDIX, но разрушал качество SciREX и SciER-384. Для SciER-128 rank-модель была почти случайной: AUC `0.543`, capture oracle alternative `26.24%`, regret `29.35%`. Редкие unsafe-классы делали обучение risk хрупким.

## 6. Итоговый official-split benchmark

### 6.1. Системы

- B1: QA по всем чанкам, выбор max confidence;
- B2: B0 top-1 → QA;
- B3: B0 top-K → QA, max confidence;
- B4a/B4b: frozen BGE top-1/top-K;
- G1: frozen Qwen2.5-3B;
- P1: V7-S2;
- P2: H2;
- P3: H2-ACS;
- P4: H3 fixed;
- FINAL: H3 Consistency Gate;
- D1/D2: answer/evidence oracle, только диагностика.

### 6.2. Главная таблица Token F1

| Система | SciREX | SciER-384 | SciER-128 | RuSERRC | SciMDIX | ALL_STANDARDIZED |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 14.7732 | 56.0738 | 65.4365 | 56.5473 | 33.2485 | 41.3310 |
| B2 | 24.0979 | 63.6773 | 69.3381 | 65.6714 | 52.6723 | 52.2373 |
| B3 | 19.0955 | 63.2028 | **73.1368** | 58.2139 | 38.7216 | 46.0502 |
| B4a | 17.5330 | 52.7309 | 56.0724 | 62.9798 | 42.5633 | 44.3694 |
| B4b | 18.4645 | 60.8589 | 56.1772 | 56.6114 | 36.8576 | 42.6129 |
| G1 | 17.6245 | 31.5206 | 31.5206 | 44.8174 | 27.7844 | 30.4367 |
| P1 | 24.6014 | 67.2980 | 71.1316 | 65.7127 | 58.2940 | 54.4557 |
| P2 / H2 | 23.8234 | 65.8721 | 70.5551 | 65.9369 | **62.8274** | 55.2003 |
| P2 без FullLEX | 26.2151 | 65.2146 | 69.3381 | 65.7127 | 58.2940 | 54.3745 |
| P2 без PairLEX | 22.9719 | 70.6049 | 69.9904 | 66.0126 | 56.8582 | 54.0351 |
| P2 без risk | 23.8234 | 65.8721 | 69.6187 | 65.9369 | 62.8274 | 55.0833 |
| P3 / H2-ACS | 23.8234 | 65.8721 | 70.5551 | 65.9369 | 62.8274 | 55.2003 |
| P4 / H3 fixed | **26.0253** | **72.0436** | 72.7105 | **66.2114** | 59.2560 | 55.9674 |
| **FINAL** | **26.0253** | **72.0436** | 72.7105 | **66.2114** | **62.8274** | **56.8603** |

FINAL воспроизводит H3 на SciREX, обеих SciER-cell и RuSERRC, а на SciMDIX выбирает H2. На SciER-128 простой B3 (`73.1368`) остаётся немного выше FINAL (`72.7105`), поэтому нельзя утверждать, что финальный метод лучший на каждой cell.

### 6.3. Gate-решения

| Датасет | lambda TRAIN | lambda TRAIN+VAL | Финальный lambda | Ветвь |
|---|---:|---:|---:|---|
| SciREX | 1.00 | 1.00 | 1.00 | H3 |
| SciER-128 | 1.00 | 1.00 | 1.00 | H3 |
| SciER-384 | 0.75 | 1.00 | 1.00 | fallback H3 |
| RuSERRC | 1.00 | 0.50 | 1.00 | fallback H3 |
| SciMDIX | 0.00 | 0.00 | 0.00 | H2 |

### 6.4. Статистические контрасты

| Контраст | Δ Token F1 | 95% CI | Интерпретация |
|---|---:|---:|---|
| P1−P2 | −0.7446 | [−3.2483; 1.2782] | различие не доказано |
| P2−B1 | +13.8693 | [6.5616; 20.2619] | надёжный выигрыш P2 |
| P2−B2 | +2.9630 | [−0.8578; 6.9570] | различие не доказано |
| P2−B3 | +9.1501 | [1.3049; 15.9812] | надёжный выигрыш в агрегате |
| P2−B4b | +12.5875 | [4.5025; 19.3205] | надёжный выигрыш |
| P2−B5-EN, EN only | −31.3982 | [−36.1871; −26.5486] | fine-tuned B5-EN сильнее |
| P3−P2 | 0.0000 | [0; 0] | итоговые ответы совпали |
| P4−P2 | +0.7671 | [−1.7213; 2.7234] | различие не доказано |

Для post-hoc FINAL:

- FINAL−H2: `+1.6600`, CI `[0.1396; 3.0761]`;
- FINAL−H3: `+0.8929`, CI `[0; 2.9412]`;
- FINAL−H3-SHARED-3: `+0.0686`, CI `[0; 0.1984]`.

Последние три числа нельзя трактовать как независимое подтверждение: gate сформулирован после просмотра official TEST.

### 6.5. Untouched confirmation

На untouched-семействах итоговый report подтверждает, что H2 надёжно превосходит B1 и B4b, но не доказывает превосходство над B2. Это более узкий и корректный claim, чем «метод превосходит все простые baseline».

### 6.6. Вторичные метрики P2/P4

| Cell | Система | Token F1 | Char F1 | ROUGE-L | EM | Lemma F1 |
|---|---|---:|---:|---:|---:|---:|
| SciREX | P2 | 23.8234 | 0.4703 | 0.2740 | 0.1429 | — |
| SciREX | P4 | 26.0253 | 0.4839 | 0.2909 | 0.1575 | — |
| SciER-384 | P2 | 65.8721 | 0.7812 | 0.6476 | 0.2889 | — |
| SciER-384 | P4 | 72.0436 | 0.8223 | 0.7106 | 0.4000 | — |
| SciER-128 | P2 | 70.5551 | 0.7788 | 0.6953 | 0.2667 | — |
| SciER-128 | P4 | 72.7105 | 0.8190 | 0.7250 | 0.4000 | — |
| RuSERRC | P2 | 65.9369 | 0.7886 | 0.6592 | 0.1633 | 66.6601 |
| RuSERRC | P4 | 66.2114 | 0.7880 | 0.6620 | 0.1600 | 66.9346 |
| SciMDIX | P2 | 62.8274 | 0.7878 | 0.6351 | 0.3929 | 67.0799 |
| SciMDIX | P4 | 59.2560 | 0.7704 | 0.5993 | 0.3571 | 63.5085 |

Лемматизация не меняет сравнительный вывод на русских данных.

### 6.7. Evidence quality

| Cell | B2 Hit@1 | P2 Hit@1 | P2 Hit@K | P2 HN@1 |
|---|---:|---:|---:|---:|
| SciREX | 0.6099 | 0.5934 | 0.9341 | нет разметки |
| SciER-384 | 0.6667 | 0.6667 | 0.9333 | 0.0000 |
| SciER-128 | 0.6000 | 0.7000 | 1.0000 | 0.0000 |
| RuSERRC | 0.7000 | 0.6900 | 1.0000 | 0.0700 |
| SciMDIX | 0.3929 | 0.5357 | 0.9286 | 0.1429 |

H2 явно улучшает evidence Hit@1 на SciMDIX и SciER-128. На RuSERRC answer F1 может улучшаться при небольшом снижении Hit@1, что показывает неполноту бинарной evidence-метрики.

## 7. Абляции и механизм

### 7.1. Вклад H2-компонентов

`Абляция − P2`; отрицательное значение означает пользу удалённого компонента.

| Абляция | SciREX | SciER family | RuSERRC | SciMDIX | ALL_STANDARDIZED |
|---|---:|---:|---:|---:|---:|
| без FullLEX | +2.3916 | −0.9372 | −0.2242 | −4.5334 | −0.8258 |
| без PairLEX | −0.8515 | +2.0840 | +0.0758 | −5.9692 | −1.1652 |
| без risk | 0.0000 | −0.4682 | 0.0000 | 0.0000 | −0.1170 |

FullLEX и PairLEX полезны в среднем прежде всего благодаря SciMDIX, но неоднородны: FullLEX мешает SciREX, PairLEX мешает SciER. Risk почти не меняет ответы — это слабое место H2.

### 7.2. H3-refinement после official TEST

| Вариант | SciREX | SciER-384 | SciER-128 | RuSERRC | SciMDIX | ALL |
|---|---:|---:|---:|---:|---:|---:|
| H3 zero-anchor / P4 | 26.0253 | 72.0436 | 72.7105 | 66.2114 | 59.2560 | 55.9674 |
| H3-BLEND | 24.6360 | 69.1947 | 72.7105 | 66.1630 | 61.6370 | 55.8471 |
| H3-GROUP-RIDGE | 23.6745 | 70.4540 | 72.7105 | 66.2114 | 59.4911 | 55.2398 |
| H3-SHARED-3 | 26.0253 | 72.0436 | 72.7105 | 65.9369 | 62.8274 | 56.7917 |
| H3-SHARED-10 | 26.0506 | 69.6302 | 72.9994 | 66.0558 | 59.7790 | 55.8000 |
| H3-STABILITY-3 с CI | 26.0253 | 72.0436 | 72.7105 | 66.2114 | 59.2560 | 55.9674 |
| H3-STABILITY-10 | 26.0506 | 71.3038 | 72.9994 | 66.0848 | 59.4676 | 55.9386 |
| **H3-CONSISTENCY-GATE-3** | **26.0253** | **72.0436** | **72.7105** | **66.2114** | **62.8274** | **56.8603** |

Seed-specific blend был нестабилен. Group ridge не решил проблему. Увеличение seeds до 10 не улучшило общий результат. Требование CI было слишком строгим и не приняло полезный SciMDIX-переход. Exact TRAIN/refit consistency оказалось наиболее компактным правилом.

### 7.3. Oracle-резерв

| Cell | P2 | Answer oracle D1 | D1−P2 | Evidence oracle D2 |
|---|---:|---:|---:|---:|
| SciREX | 23.8234 | 47.4094 | +23.5860 | 27.6337 |
| SciER-384 | 65.8721 | 92.8773 | +27.0052 | 64.4797 |
| SciER-128 | 70.5551 | 91.1746 | +20.6195 | 76.2749 |
| RuSERRC | 65.9369 | 88.2977 | +22.3608 | 86.0916 |
| SciMDIX | 62.8274 | 88.5894 | +25.7620 | 88.3414 |

Большой D1-gap на всех корпусах показывает, что хорошие ответы уже присутствуют среди кандидатов, но policy не умеет их надёжно выбирать. D2 не является верхней границей answer F1: наличие официального evidence и способность reader выдать лучший ответ — разные свойства.

## 8. Специализированные baseline и границы метода

### 8.1. English in-domain ModernBERT

| Система | SciREX | SciER-384 | SciER-128 | EN balanced |
|---|---:|---:|---:|---:|
| B5-EN, in-domain fine-tuned | 56.3056 | 100.0000 | 97.0556 | 77.4167 |
| frozen ModernBERT-SQuAD2, B0 top-K | 22.9600 | 54.5416 | 62.1445 | 40.6515 |
| frozen ModernBERT-SQuAD2, all chunks | 15.7788 | 57.3247 | 49.2373 | 34.5299 |
| P2 | 23.8234 | 65.8721 | 70.5551 | 46.0185 |
| P4/FINAL | 26.0253 | 72.0436 | 72.7105 | 49.2012 |

B5-EN получил преимущество за счёт in-domain дообучения, а не одной архитектуры ModernBERT. Frozen `smangla/ModernBERT-base-squad2`, revision `f80f767241e36220e9aa805886ab3ae035349bc3`, обученный на SQuAD 2.0 и не дообученный на SciREX/SciER, слабее H2/H3. B0 shortlist улучшил его EN-balanced результат относительно all-chunks на `+6.1217`, CI `[1.9277; 10.7132]` по направлению B0K−ALL.

### 8.2. Generative baseline

Frozen Qwen2.5-3B/G1 дал ALL `30.4367`; P2 превосходит его на всех пяти cell, family-balanced разница `+24.7636`. В текущем prompt-based режиме G1 не является конкурентоспособной заменой специализированному extractive pipeline.

## 9. Почему улучшение ограничено

1. **B0 уже силён**, особенно на RuSERRC; безопасная policy редко получает надёжный сигнал для переключения.
2. **Top-K recall высок**, но различение близких кандидатов остаётся слабым. P2 Hit@K достигает 0.93–1.00.
3. **QA confidence плохо сопоставим между чанками**; agreement может быть высоким и у одинаково неправильных ответов.
4. **Лексические признаки dataset-dependent**: помогают SciMDIX, но могут вредить SciREX/SciER.
5. **Risk-модель почти не активна** в official benchmark, а более сложная SRR risk-модель страдает от редких положительных unsafe-примеров.
6. **Малые тестовые наборы** SciER и SciMDIX дают широкие интервалы.
7. **Вопрос и ranking спутаны** в части V4.4: смена вопроса иногда улучшает reader без смены чанка.
8. **Reader ограничивает верхнюю границу**: B5-EN выше D1 на SciREX, поскольку D1 привязан к исходной reader-ветке.
9. **Universal routers переобучаются на локальную структуру корпуса** и не сохраняют знак эффекта между короткими аннотациями и длинными статьями.

## 10. Ресурсы и воспроизводимость

TRAIN+VAL B0/refit в итоговом benchmark:

| Cell | Документы | Вопросы | Эпохи B0 | Wall time, с | GPU-hours upper bound |
|---|---:|---:|---:|---:|---:|
| SciREX | 324 | 612 | 4 | 986.6 | 0.2740 |
| SciER-128 | 90 | 264 | 5 | 646.8 | 0.1797 |
| SciER-384 | 90 | 250 | 3 | 426.6 | 0.1185 |
| RuSERRC | 240 | 408 | 5 | 394.4 | 0.1095 |
| SciMDIX | 125 | 185 | 5 | 223.5 | 0.0621 |
| **Всего** | — | — | — | **2677.9** | **0.7439** |

Сохранённый inference benchmark нельзя публиковать как честное end-to-end сравнение: workers повторно использовали общий QA cache, B1 показывал неправдоподобные `0.00001–0.00010 s`, peak VRAM везде равен нулю, число загруженных параметров учитывалось непоследовательно. Для статьи нужен отдельный cold/warm benchmark без общего answer cache, с CUDA synchronization и реальным peak VRAM.

## 11. Допустимые и недопустимые claims

### Допустимо

- H2 надёжно лучше B1 и frozen BGE top-K B4b в family-balanced official benchmark.
- H2 имеет положительный point estimate против B2, но преимущество статистически не доказано.
- H3 существенно полезен на SciER-384 в сравнении с H2; agreement неоднороден по корпусам.
- Финальная архитектура — H3 Consistency Gate; на раскрытом TEST она сохраняет H3 там, где TRAIN/refit не согласованы, и выбирает H2 на SciMDIX при согласованном lambda=0.
- D1 показывает большой резерв выбора среди уже вычисленных кандидатов.
- In-domain fine-tuning reader даёт значительно больший эффект на английском, чем frozen SQuAD2 model или meta-reranking поверх старого reader.

### Нельзя заявлять без новой проверки

- что FINAL независимо подтверждён как универсально лучший метод;
- что H2/H3 превосходит все простые baseline на каждой cell;
- что agreement универсально полезен;
- что attention heads доказанно отражают причинно значимое evidence в используемом RoBERTa reader;
- что latency/VRAM текущего resource run является честным end-to-end измерением;
- что результат RuSERRC является независимым confirmation.

## 12. Что вынести в основную статью, приложение и supplementary

**Основной текст:** H2/H3, consistency gate, official main table, P2/P4/FINAL contrasts, evidence metrics, oracle gap, одна компактная хронология V3→V8.

**Приложение:** полные V4.1–V4.5, V5–V10, H3 refinement, per-cell CI, вторичные метрики, русские ranker-factorials.

**Supplementary/repository:** manifests, hashes, все OOF predictions, thresholds, fold assignments, failure reports, resource caveat, точные model revisions.

## 13. Литература, упомянутая в Markdown-файлах проекта

Ниже сохранены все содержательные упоминания, найденные в проектной документации. Перед подачей статьи библиографические поля и первичные URL следует проверить; часть локальных файлов содержит ссылки на Consensus и исторические значения числа цитирований, которые не являются стабильными библиографическими данными.

### Retrieval, hard negatives и answerability

1. Qu, Y., Ding, Y., Liu, J., et al. (2020). **RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering.** В проекте мотивирует denoised hard negatives, cross-batch negatives и очистку false negatives.
2. Wang, Z., Ng, P., Ma, X., Nallapati, R., Xiang, B. (2019). **Multi-passage BERT: A Globally Normalized BERT Model for Open-domain Question Answering.** Мотивирует сопоставимость answer scores между passages; проект отдельно предупреждает, что независимый post-hoc softmax не равен глобальной нормализации.
3. Wang, S., Yu, M., Jiang, J., et al. (2017). **Evidence Aggregation for Answer Re-Ranking in Open-Domain Question Answering.** arXiv:1711.05116. Основание для answer/evidence aggregation.
4. Hashimoto, K., Zhou, Y., Yavuz, S., Xiong, C., Yu, P. S. (2021). **Dense Hierarchical Retrieval for Open-Domain Question Answering.** Мотивирует сочетание passage-level и document/section-level semantics и in-document negatives.
5. Ni, A., Gardner, M., Dasigi, P. (2021). **Mitigating False-Negative Contexts in Multi-document Question Answering with Retrieval Marginalization.** EMNLP 2021, DOI `10.18653/v1/2021.emnlp-main.497`. Использовано для множественных допустимых evidence contexts и запрета считать альтернативный gold-support отрицательным.
6. Caciularu, A., Dagan, I., Goldberger, J., Cohan, A. (2021). **Long Context Question Answering via Supervised Contrastive Learning.** Мотивирует contrastive evidence discrimination.
7. Abdumalikov, R., Minervini, P., Kementchedjhieva, Y. (2024). **Answerability in Retrieval-Augmented Open-Domain Question Answering.** arXiv:2403.01461. Поддерживает разделение семантической близости и answerability.
8. Ferguson, J., Hajishirzi, H., Dasigi, P., Khot, T. (2022). **Retrieval Data Augmentation Informed by Downstream Question Answering Performance.** FEVER 2022, DOI `10.18653/v1/2022.fever-1.1`. Мотивирует оценивать retrieval context через downstream QA.
9. Iyer, S., Min, S., Mehdad, Y., Yih, W.-t. (2021). **RECONSIDER: Improved Re-Ranking using Span-Focused Cross-Attention for Open Domain Question Answering.** NAACL 2021, DOI `10.18653/v1/2021.naacl-main.100`. V4/V7 описаны как inspired adaptation, а не репликация.
10. Xin, C., Zhou, S., Chen, X., et al. (2025). **Aligning Retrieval with Reader Needs: Reader-Centered Passage Selection for Open-Domain Question Answering.** Мотивирует reader-derived selection signals.

### Keyword/document compatibility

11. Grootendorst, M. (2020). **KeyBERT: Minimal keyword extraction with BERT.** Zenodo, DOI `10.5281/zenodo.4461265`. В V4 использовался как lightweight document/keyphrase profile; гипотеза о document compatibility являлась собственной гипотезой проекта.

### Attention и long-context reranking

12. Li, Z., Tao, C., Feng, J., et al. (2023). **FAA: Fine-grained Attention Alignment for Cascade Document Ranking.** Использовалась как мотивация attention feedback для selector/ranker.
13. Wu, W., Wang, Y., Xiao, G., Peng, H., Fu, Y. (2024). **Retrieval Head Mechanistically Explains Long-Context Factuality.** arXiv:2404.15574. Мотивирует specialized/retrieval-head hypothesis и causal masking controls.
14. Zhang, W., Yin, F., Yen, H., Chen, D., Ye, X. (2025). **Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking.** arXiv:2506.09944. Мотивирует query-context attention и head-based scores.
15. Li, Y., Li, J., Yu, M., et al. (2026). **Query-focused and Memory-aware Reranker for Long Context Processing.** arXiv:2602.12192. Мотивирует listwise attention-score interpretation.
16. Liu, W., Wu, N., Yang, S., et al. (2025). **MuDAF: Long-Context Multi-Document Attention Focusing through Contrastive Learning on Attention Heads.** arXiv:2502.13963. Мотивирует focused-vs-distracted attention analysis.
17. Xu, P., Liang, D., Huang, Z., Xiang, B. (2021). **Attention-guided Generative Models for Extractive Question Answering.** arXiv:2110.06393. Связывает attention patterns с локализацией extractive span.
18. Neely, M., Schouten, S. F., Bleeker, M. J. R., Lucic, A. (2022). **A Song of (Dis)agreement: Evaluating the Evaluation of Explainable Artificial Intelligence in Natural Language Processing.** Использовано как ограничение: raw attention нельзя автоматически трактовать как faithful explanation.
19. Tran, L., Li, Y., Florian, R., Sun, W. (2025). **Contrastive Retrieval Heads Improve Attention-Based Re-Ranking.** Упомянута в протоколе V4.2 как ориентир для контрастного выбора attention heads.
20. Tian, Y., Mo, F., Zhang, W., Qi, Y., Nie, J.-Y. (2026). **ReAttn: Improving Attention-based Re-ranking via Attention Re-weighting.** Упомянута как связанная работа по re-weighting attention для reranking.

### False-negative filtering и кластеризация

21. Shen, T., Liu, M., Zhou, M., Xiong, D. (2022). **Recovering Gold from Black Sand: Multilingual Dense Passage Retrieval with Hard and False Negative Samples.** EMNLP 2022, [ACL Anthology](https://aclanthology.org/2022.emnlp-main.730/). Проект адаптировал идею adaptive false-negative sample filter; полный mHFN не воспроизводился.
22. Tibshirani, R., Walther, G., Hastie, T. (2001). **Estimating the Number of Clusters in a Data Set Via the Gap Statistic.** DOI [10.1111/1467-9868.00293](https://doi.org/10.1111/1467-9868.00293). В Phase 1 упомянут для выбора числа K-means clusters, а не как criterion positive/hard-negative.
23. Jang, E., Gu, S., Poole, B. (2017). **Categorical Reparameterization with Gumbel-Softmax.** [Google Research](https://research.google/pubs/categorical-reparameterization-with-gumbel-softmax/). В локальном протоколе Gumbel-Softmax разрешён только как optional ablation с фиксированной температурой; основной static pipeline использует hard assignment. Temperature decay из mHFN не переносился, поскольку encoder frozen.

### Методическая оговорка по литературе

Большинство перечисленных работ рассматривает open-domain QA, multi-document retrieval или decoder-only long-context LLM. UnTIE работает с чанками одного заранее известного документа и encoder/extractive readers. Из литературы переносились мотивация и тип сигнала, а конкретные признаки, safety gate, H2/H3 и consistency rule являются собственной экспериментальной адаптацией.

## 14. Основные внутренние источники

- `hypothesis/final_method_architecture.md`;
- `hypothesis/final_method_all_variants_and_baselines_comparison.md`;
- `hypothesis/result_final_official_split_h2_family_benchmark_analysis.md`;
- `hypothesis/h3_consistency_gate_results_analysis.md`;
- `hypothesis/qa_chunk_selection_experiments_v2_results.md` — `v4_5_*`;
- `hypothesis/v6_uar_g_universal_results_detailed.md`;
- `hypothesis/v7_rapr_progressive_risk_aware_results_detailed.md`;
- `hypothesis/v8_dual_branch_results_analysis_2026-09-23.md`;
- `hypothesis/v9_h2_adaptive_component_selection_results_detailed.md`;
- `hypothesis/v10_srr_switch_rank_risk_results_detailed.md`;
- `experiments/qa_chunk_selection/references/sources_used.md`.

