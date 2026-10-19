# Материалы к статье 2: многоязычный benchmark выбора фрагмента в научных текстах

Дата сборки: **2026-10-08**  
Статус: **исследовательское досье для подготовки статьи, не готовая рукопись**  
Охват: **SciREX, SciER-384, SciER-128, RuSERRC, SciMDIX; подготовка, контроль качества, baseline, official TEST и межъязыковой анализ**

## 1. Назначение и возможный фокус статьи

Этот файл собирает материал для отдельной статьи о данных и benchmark. Центральный предмет — не только итоговая архитектура UnTIE, а воспроизводимая постановка multilingual in-document chunk selection для научных текстов разной длины и структуры.

Корпуса принципиально неоднородны:

- **SciREX** и **SciER** — английские научные документы; SciER содержит полнотекстовые статьи и непосредственно проверяет поиск фрагмента в длинном документе;
- **RuSERRC** — русская разметка девяти риторико-содержательных аспектов;
- **SciMDIX** — русские научные аннотации с aspect, NER и relation слоями; это независимая проверка тематического извлечения, но не полный аналог длинного SciER.

Эта неоднородность является частью benchmark: метод должен переноситься между языками, длинами документов, таксономиями и плотностью candidate universe. Абсолютный F1 между корпусами не следует трактовать как прямую меру их относительной сложности; корректны сравнения систем внутри одной cell и стандартизованное семейное агрегирование.

## 2. Уровни данных и доказательности

### 2.1. Development и official benchmark

Исторические V1–V10 использовали development/OOF протоколы. Итоговый official-split benchmark заново зафиксировал TRAIN, VAL и blind TEST. Прямой итоговый рейтинг строится только на official TEST.

| Cell | Blind prediction | Answerable в оценке | Документы оценки | Статус до benchmark |
|---|---:|---:|---:|---|
| SciREX | 264 вопроса | 182 | 63 | untouched |
| SciER-384 | 30 | 30 | 10 | untouched |
| SciER-128 | 30 | 30 | 10 | untouched; те же документы, что SciER-384 |
| RuSERRC | 100 | 100 | 60 | exposed |
| SciMDIX | 121 | 28 | 17 | untouched |

Для SciREX, SciER и SciMDIX аудит не обнаружил пересечения TEST с историческими V1–V10 по ID, нормализованному тексту и DOI. Было просмотрено 9289 исторических evaluation-файлов. RuSERRC уже использовался при разработке и не является независимым confirmation.

### 2.2. Семейная агрегация

SciER-128 и SciER-384 используют одни документы. В `ALL_STANDARDIZED` они сначала усредняются в одно семейство SciER, затем SciREX, SciER, RuSERRC и SciMDIX получают равный вес. Bootstrap двух SciER-cell выполняется согласованно.

## 3. Общий конвейер подготовки

```text
официальный документ и разметка
  -> frozen source revision + SHA-256
  -> нормализация текста с проверкой offsets
  -> language-specific sentence splitting
  -> детерминированные чанки
  -> точные gold spans в координатах чанков
  -> вопросная разметка Q5
  -> frozen extractive QA по кандидатам
  -> существующий Phase 1 labeler
  -> qa_positive / hard_negative / easy_negative /
     false_negative_candidate / evidence_positive_qa_failed
  -> answerable-only TRAIN/CV
  -> все неanswerable instances остаются в audit-файлах
  -> document-grouped CV и locked TEST
```

### 3.1. Phase 1

Во всех новых корпусах использован существующий алгоритм, а не упрощённое правило по наличию иной аннотации:

1. official span задаёт gold answer/evidence;
2. frozen extractive QA проверяет, может ли вопрос получить совпадающий с gold ответ;
3. BM25 выбирает top-5 сложных отрицательных кандидатов;
4. embedding clustering отделяет вероятные false negatives;
5. чанк с аннотацией другого аспекта **не становится hard negative автоматически**;
6. gold-supporting чанк, на котором QA не воспроизвёл ответ, сохраняется как `evidence_positive_qa_failed`, а не превращается в отрицательный.

Это решение соответствует литературе о false-negative contexts и особенно важно для плотной разметки SciMDIX/SciER.

### 3.2. Answerable-only CV

В TRAIN и grouped CV входят только инстансы с хотя бы одним robust `qa_positive`. Полный набор Phase 1 сохраняется отдельно для аудита. Это делает обучение B0 совместимым с фактической extractive QA-задачей и предотвращает обучение на вопросах, для которых reader не способен получить положительный ответ при текущей формулировке.

Ограничение: answerable-фильтр измеряет совместимость разметки с конкретными frozen reader и вопросами. Он не означает, что исключённые official spans неверны.

### 3.3. Q5

Для аспекта сохраняются пять формулировок вопроса. Phase 1 исторически использовал первый primary-вопрос; поздние эксперименты могли выбирать вариант внутри TRAIN/CV. Для RuSERRC Q5 подготовлен для девяти аспектов. Calibration-документы использовались только для выбора consistency threshold из `{0.6, 0.8, 1.0}` и были исключены из CV.

Question choice является реальным компонентом pipeline. На SciER V4.4 выбор Q3 иногда улучшал итоговый ответ без изменения ranking, поэтому эффект нельзя целиком приписывать chunk reranking.

### 3.4. Hard negatives и audit classes

Пять классов имеют разные роли:

- `qa_positive`: evidence и QA-ответ согласованы;
- `evidence_positive_qa_failed`: official evidence есть, но reader не подтвердил ответ;
- `false_negative_candidate`: тематически близкий кандидат, который опасно считать отрицательным;
- `hard_negative`: близкий, но проверенно неподдерживающий кандидат;
- `easy_negative`: прочий отрицательный чанк.

HN@1 полезен только там, где hard-negative слой достаточно информативен. Нулевой HN@1 на SciER не доказывает отсутствия ошибок выбора; SciREX в official benchmark вообще не имеет сопоставимой HN-разметки.

## 4. SciREX

### 4.1. Роль в исследовании

SciREX был исходным английским корпусом разработки. Его аспекты в V2 summary: `method`, `task`, `metric`, `material`. Он использовался для формирования Phase 1, B0, attention и reader-centered гипотез.

### 4.2. Историческое разбиение V2

| Split | Документы | Вопросы | Answerable | No robust positive |
|---|---:|---:|---:|---:|
| train | 180 | 477 | 303 | 174 |
| calibration | 44 | 111 | 73 | 38 |
| development | 44 | 126 | 77 | 49 |
| internal test | 31 | 78 | 54 | 24 |
| seen exploratory dev | 65 | 163 | 105 | 58 |

Train chunk labels: 824 positive, 479 hard negative, 4731 easy negative и 3223 ambiguous. Эти исторические splits нельзя напрямую смешивать с final official split.

### 4.3. Official benchmark

Blind population — 264 вопроса/66 документов; answerable evaluation — 182 вопроса/63 документа. TRAIN+VAL refit: 324 документа, 612 вопросов, B0 обучался 4 эпохи, wall time `986.6 s`.

SciREX остаётся наиболее трудной английской cell для extractive reader. Answer oracle D1 `47.4094`, а in-domain ModernBERT B5-EN `56.3056` даже выше D1 исходной reader-ветки, что показывает одновременное ограничение selector и reader.

## 5. RuSERRC

### 5.1. Источник и парсинг

Полный локальный mirror содержит 591 документ; 587 имеют хотя бы один из выбранных аспектов. Inline markup удалялся с точным отслеживанием offsets. Source revision/hashes фиксировались; при загрузке из официального репозитория сохранялся resolved commit. Обнаружено одно parse warning, при этом итоговые sanity errors равны нулю.

Девять аспектов:

| Нормализованный аспект | Исходный tag | Число tags |
|---|---|---:|
| task | Task | 471 |
| goal | Goal | 63 |
| contribution | Contrib | 1726 |
| method | Method | 628 |
| tool | Tool | 354 |
| use | Use | 378 |
| advantage | Adv | 393 |
| example | Example | 125 |
| conclusion | Conc | 170 |

### 5.2. Подготовленная структура

- 1979 shared chunks;
- 5319 aspect instances до фильтрации;
- 1994 eligible aspect instances;
- 4458 source lexical mentions;
- 917 `qa_positive` chunks;
- 860 hard negatives;
- 2778 evidence-positive/QA-failed;
- 2474 easy negatives;
- 493 false-negative candidates.

### 5.3. Исторические splits

| Split | Документы | Вопросы | Answerable | No robust positive |
|---|---:|---:|---:|---:|
| train | 352 | 1190 | 410 | 780 |
| calibration | 88 | 304 | 102 | 202 |
| development | 88 | 302 | 118 | 184 |
| internal test | 59 | 198 | 68 | 130 |

Unlocked instances `1796`, locked `198`, grouped CV folds `5`. Поздняя Q5/B0-подготовка исключила calibration из CV и использовала только answerable из старых train+development. B0 повторял английский recipe: question-balanced BCE, sampling ratio `1:3:1`, fold calibration/early stopping, `DeepPavlov/rubert-base-cased-sentence`, seeds 42/43/44.

### 5.4. Языковой профиль

- extractive QA: локальный `rubert_ru_qa_model`;
- sentence model исходной подготовки: `DeepPavlov/rubert-base-cased-sentence`;
- поздние reranker: BGE, `ARGA100/ru-reranker-modernbert-small`, `DiTy/cross-encoder-russian-msmarco`;
- Lemma F1 использовался только как финальная дополнительная метрика.

### 5.5. Методический вывод

На RuSERRC улучшение BGE B0 оказалось намного больше, чем эффект второго компонента: старый B0 `52.768`, BGE `69.954`, `+17.167`, CI `[14.066; 20.152]`. Это показывает, что качество base ranker должно контролироваться отдельно при сравнении языков. После сильного B0 reader-consensus корректно выбирал fallback.

## 6. SciMDIX

### 6.1. Происхождение

Источник: Hugging Face `tvbat/SciMDIX`, frozen revision `69170fbdd15a4e19e247a9eb9554356d018b6f57`. Использованы русские splits и обе конфигурации `aspects` и `ner_re`. Source files и SHA-256 записаны в `artifacts/qa_chunk_selection_external_data_20260913/source_manifest.json`.

SciMDIX — bilingual RU/KZ dataset научных аннотаций по четырём доменам: IT, Linguistics, Medicine, Psychology. В этой работе использована русская часть.

### 6.2. Таксономия и mapping

| Official | Benchmark |
|---|---|
| AIM | goal |
| MATERIAL | material |
| METHOD | method |
| RESULT | result |
| TASK | task |
| TOOL | tool |
| USAGE | use |

Таксономия не объединялась. NER/RE слой сохранялся отдельно для будущего анализа.

### 6.3. Статистика подготовки 128/24

| Показатель | Значение |
|---|---:|
| Документы | 206: train 186, test 20 |
| Phase 1 instances | 1122 |
| Answerable overall | 213 |
| No-QA-positive | 909 |
| Train instances | 1001 |
| Train answerable | 185 |
| CV documents | 125 |
| Chunks | 1083 |
| Chunks/document | min 2, median 5, max 13 |
| Gold aspect mentions | 2129 |
| Native entities preserved | 6693 |
| Relations preserved | 2376 |

Phase 1 chunk classes: 287 QA positives, 1829 evidence-positive/QA-failed, 243 false-negative candidates, 393 hard negatives, 3230 easy negatives.

Answerable train по аспектам: goal `27/184`, material `10/119`, method `70/144`, result `19/181`, task `22/152`, tool `31/66`, use `6/155`. Низкая answerable rate — содержательный результат несовместимости текущего reader/вопросов с aspect spans. Offset checks прошли; gold evidence не терялось и сохранялось как ambiguous.

### 6.4. Official TEST

Blind official test содержит 121 aspect instance из 20 аннотаций; answerable evaluation — 28 вопросов из 17 документов. Малый размер объясняет широкие bootstrap-интервалы и высокую чувствительность среднего к одному вопросу. TRAIN+VAL final refit: 125 документов, 185 вопросов, 5 эпох, `223.5 s`.

### 6.5. Особый результат для метода

SciMDIX — единственная official cell, где fixed H3 хуже H2: `59.2560` против `62.8274`. Один question-level failure существенно влияет на итог. Final consistency gate выбрал lambda=0 и H2 на TRAIN и TRAIN+VAL. Это полезное наблюдение, но сам gate post-hoc и нуждается в новом blind split.

## 7. SciER

### 7.1. Происхождение

Источник: GitHub `edzq/SciER`, frozen revision `db347813de379eb200e0813bbfee350d67e7c701`. Использована PLM release, потому что она сохраняет document-level tokens, официальные sentence boundaries, exact entity spans, relations и четыре official split. В source snapshot сохранена GNU GPL v3 license.

### 7.2. Исходные данные

- 106 полнотекстовых научных статей;
- train 80, dev 10, test 10, test_ood 6;
- entity types Dataset, Method, Task;
- 24 518 entity mentions;
- 12 083 relations;
- fine-grained relation schema и отдельный OOD test.

Mapping: Dataset→material, Method→method, Task→task.

### 7.3. Два варианта чанкинга

Каноническая английская версия — 384 tokens, overlap 50, как в SciREX. Вариант 128/24 сохранён как явная ablation.

| Показатель | 128/24 | 384/50 |
|---|---:|---:|
| Документы | 106 | 106 |
| Chunks | 4806 | 1101 |
| Chunks/document, median | 42 | 10 |
| Диапазон chunks/document | 18–117 | 4–28 |
| Train Phase 1 instances | 239 | 239 |
| Train answerable | 235 | 225 |
| Answerable all splits | 310 | 295 |
| No-QA-positive | 5 | 20 |
| QA-positive chunks | 4496 | 1263 |
| Hard-negative chunks | 238 | 128 |

Exact span checks прошли в обеих версиях. Разница answerable counts является эффектом гранулярности чанков, а не потерей аннотаций.

### 7.4. Phase 1 384/50

- 315 aspect instances overall;
- train 239, dev 29, test 30, test_ood 17;
- answerable train 225, dev 25, test 30, OOD 15;
- 1263 QA positive;
- 1485 evidence-positive/QA-failed;
- 269 false-negative candidates;
- 128 hard negatives;
- 135 easy negatives.

### 7.5. Phase 1 128/24

- answerable train 235/239;
- dev 29/29;
- test 30/30;
- OOD 16/17;
- 4496 QA positive, 4951 evidence-positive/QA-failed, 967 false-negative candidates, 238 hard negatives, 3662 easy negatives.

### 7.6. Интерпретация двух cell

SciER-128 создаёт намного более крупный candidate universe и более тонкую локализацию. SciER-384 ближе к исходной английской подготовке и дешевле. В development V4.4 обе версии показали уверенный прирост reader-assisted pipeline. В official TEST B3 оказался лучшим на SciER-128 (`73.1368`), тогда как H3/FINAL — на SciER-384 (`72.0436`). Это подчёркивает, что оптимальная политика зависит от chunk granularity.

## 8. Сопоставимость языков

Историческое прямое сравнение V4.4:

| Характеристика | SciREX EN | RuSERRC RU |
|---|---:|---:|
| Вопросы | 453 | 508 |
| Документы | 240 | 300 |
| Аспекты | 4 | 9 |
| Среднее кандидатов | 19.75 | 4.05 |
| Медиана кандидатов | 19 | 4 |
| Диапазон | 4–42 | 1–9 |
| Не менее 5 кандидатов | 99.78% | 43.90% |
| Sentence model | all-MiniLM-L6-v2 | rubert-base-cased-sentence |

Русский Hit@5 нельзя напрямую сравнивать с английским: у 56.10% русских строк меньше пяти кандидатов, а Hit@10 автоматически равен 100%. В английском 95.58% строк имели не менее десяти кандидатов.

При прямом переносе V4.4:

| Система | EN Token F1 | RU Token F1 |
|---|---:|---:|
| B0 | 27.3584 | 52.7683 |
| V6-K5 | 34.6249 | 51.0300 |
| SEM-FIX | 37.5108 | 47.5042 |
| LEX-FIX | 36.9816 | 46.4551 |
| ASEM | 35.1112 | 50.2662 |

Абсолютные значения несопоставимы; направления эффекта показывают, что английский reader-consensus signal переносился плохо. Поздние русские эксперименты выяснили, что причина частично в базовом ranker, размере candidate universe, неоднородности девяти аспектов и нестабильном CAL→EVAL выборе коэффициентов.

## 9. Official benchmark: полный результат

### 9.1. Token F1

| Система | SciREX | SciER-384 | SciER-128 | RuSERRC | SciMDIX | ALL_STANDARDIZED |
|---|---:|---:|---:|---:|---:|---:|
| B1: all chunks, max QA confidence | 14.7732 | 56.0738 | 65.4365 | 56.5473 | 33.2485 | 41.3310 |
| B2: B0 top-1 | 24.0979 | 63.6773 | 69.3381 | 65.6714 | 52.6723 | 52.2373 |
| B3: B0 top-K, max QA confidence | 19.0955 | 63.2028 | **73.1368** | 58.2139 | 38.7216 | 46.0502 |
| B4a: frozen BGE top-1 | 17.5330 | 52.7309 | 56.0724 | 62.9798 | 42.5633 | 44.3694 |
| B4b: frozen BGE top-K | 18.4645 | 60.8589 | 56.1772 | 56.6114 | 36.8576 | 42.6129 |
| G1: frozen Qwen2.5-3B | 17.6245 | 31.5206 | 31.5206 | 44.8174 | 27.7844 | 30.4367 |
| P1: V7-S2 | 24.6014 | 67.2980 | 71.1316 | 65.7127 | 58.2940 | 54.4557 |
| P2: H2 | 23.8234 | 65.8721 | 70.5551 | 65.9369 | **62.8274** | 55.2003 |
| P4: H3 | **26.0253** | **72.0436** | 72.7105 | **66.2114** | 59.2560 | 55.9674 |
| FINAL: H3 consistency gate | **26.0253** | **72.0436** | 72.7105 | **66.2114** | **62.8274** | **56.8603** |

### 9.2. По отдельным корпусам

- **SciREX:** FINAL `26.0253`; выше B1, B3/B4, но различие с B2 локально не доказано. In-domain B5-EN `56.3056` показывает сильное ограничение reader.
- **SciER-384:** FINAL `72.0436`; P4−P2 `+6.1715`, CI `[1.1111; 12.0234]`, единственный надёжный локальный agreement-выигрыш.
- **SciER-128:** B3 `73.1368` немного выше FINAL `72.7105`; простой QA confidence полезен при мелких чанках.
- **RuSERRC:** FINAL `66.2114`; P4−P2 `+0.2745`, CI `[0; 0.7912]`, малый эффект; cell exposed.
- **SciMDIX:** H2 `62.8274` выше H3 `59.2560`; FINAL возвращается к H2, но решение post-hoc.

### 9.3. Специализированные английские baseline

| Система | SciREX | SciER-384 | SciER-128 | EN balanced |
|---|---:|---:|---:|---:|
| B5-EN, in-domain fine-tuned ModernBERT | 56.3056 | 100.0000 | 97.0556 | 77.4167 |
| Frozen ModernBERT-SQuAD2 + B0 top-K | 22.9600 | 54.5416 | 62.1445 | 40.6515 |
| Frozen ModernBERT-SQuAD2, all chunks | 15.7788 | 57.3247 | 49.2373 | 34.5299 |
| FINAL | 26.0253 | 72.0436 | 72.7105 | 49.2012 |

Модель frozen baseline: `smangla/ModernBERT-base-squad2`, revision `f80f767241e36220e9aa805886ab3ae035349bc3`, SQuAD 2.0. Почти идеальные B5-EN на SciER требуют осторожности из-за 10 TEST-документов, хотя leakage audit нарушений не нашёл.

### 9.4. Oracle

| Oracle | SciREX | SciER-384 | SciER-128 | RuSERRC | SciMDIX |
|---|---:|---:|---:|---:|---:|
| D1 answer oracle | 47.4094 | 92.8773 | 91.1746 | 88.2977 | 88.5894 |
| D2 evidence oracle | 27.6337 | 64.4797 | 76.2749 | 86.0916 | 88.3414 |

Разница D1−P2 составляет 20.6–27.0 F1 на всех cell. Это общий benchmark-вывод: retrieval recall уже достаточен для появления хорошего ответа, а selector/policy использует запас плохо.

## 10. Development V4.4 как внешний validation benchmark

| Язык | Датасет/протокол | B0 | Лучшая релевантная система | F1 | Δ | 95% CI |
|---|---|---:|---|---:|---:|---|
| EN | SciREX | 27.358 | V6-K5 | 34.625 | +7.267 | historical OOF |
| EN | SciER 384/50 | 51.965 | LEX | 60.825 | +8.849 | [2.562; 15.178] |
| EN | SciER 128/24 | 52.831 | SEM | 62.586 | +9.767 | [4.863; 14.773] |
| RU | RuSERRC direct transfer | 52.768 | V6-K5 | 51.030 | −1.738 | — |
| RU | RuSERRC safe tuning | 52.768 | V6 q0 K5 | 53.407 | +0.634 | [−0.617; 1.903] |
| RU | RuSERRC BGE B0 | 52.768 | BGE | 69.954 | +17.167 | [14.066; 20.152] |
| RU | SciMDIX initial | 52.510 | V6 | 52.529 | +0.025 | [−2.388; 2.589] |
| RU | SciMDIX DiTy | 57.331 | ASEM-Q5 K4 | 59.625 | +2.280 | [0.770; 4.088] |

Эта таблица важна как история external validation, но не должна объединяться арифметически с official TEST: данные и training contracts отличаются.

## 11. Контроль качества и воспроизводимость

### 11.1. Проверки подготовки

- source hashes пересчитаны;
- каждый stored gold span точно воспроизводит substring чанка;
- ни один hard negative не содержит gold support;
- document IDs непересекаются между official splits;
- CV содержит ровно answerable TRAIN documents;
- для каждого аспекта сохранено пять вопросов;
- sanity errors для SciMDIX/SciER равны нулю;
- новые и существующие тесты подготовки: 35 passed на момент отчёта.

### 11.2. Frozen provenance

SciMDIX source manifest содержит revision и SHA-256 для:

- `aspects_train_ru.parquet`;
- `aspects_test_ru.parquet`;
- `ner_re_train_ru.parquet`;
- `ner_re_test_ru.parquet`;
- source `README.md`.

SciER manifest содержит revision и SHA-256 для `train/dev/test/test_ood.jsonl`, README и LICENSE.

### 11.3. Артефакты подготовки

Для каждого корпуса сохраняются:

- `preprocessing/documents.parquet`;
- `chunks.parquet`;
- `aspect_instances.parquet`;
- `gold_mentions.parquet`;
- `source_metadata.parquet`;
- `enrichment.parquet`;
- `qa_input.parquet`;
- `data/all_phase1_instances.jsonl`;
- answerable dataset и grouped CV assignment;
- `hn_quality_flags.jsonl`;
- `questions_q5.json`;
- locked official splits;
- `source_manifest.json`, progress и completion marker.

## 12. Ограничения benchmark

1. **Corpus asymmetry.** SciMDIX — аннотации, SciER — полные статьи; они проверяют разные режимы chunk selection.
2. **Small TEST.** SciER имеет 10 in-domain TEST-документов, SciMDIX — 17 answerable-eval документов и 28 вопросов.
3. **Reader-conditioned inclusion.** Answerable-only evaluation зависит от reader и вопроса.
4. **RuSERRC exposure.** Эта cell не является независимым подтверждением.
5. **SciER coupling.** 128/384 — две обработки одних документов, не независимые наборы.
6. **Question/ranking confound.** На SciER часть gain вызвана выбором вопроса.
7. **Unequal candidate universes.** Hit@K между языками не сравним без учёта числа чанков.
8. **HN semantics differ.** Нулевой HN@1 не гарантирует правильность evidence.
9. **License audit.** SciER snapshot содержит GPLv3; для SciMDIX в сохранённой карточке явное license field не зафиксировано — перед распространением производных данных требуется отдельная проверка условий использования.
10. **Resource table invalid end-to-end.** Общий QA cache исказил latency и parameter counts; peak VRAM не измерен.
11. **Final gate post-hoc.** FINAL следует перепроверить на новом закрытом split.

## 13. Что можно заявить в статье о benchmark

### Поддерживается данными

- единая схема подготовки сохраняет official spans, неоднозначные evidence и native NER/RE;
- SciER действительно создаёт более длинный поиск: median 42 чанка при 128/24 и 10 при 384/50 против median 5 у SciMDIX;
- chunk granularity меняет answerability и относительный успех selection policy;
- strongest baseline различается по cell: B3 на SciER-128, H3 на SciER-384/SciREX/RuSERRC, H2 на SciMDIX, in-domain B5-EN на английском;
- language-specific base ranker особенно критичен для RuSERRC;
- хорошие кандидаты присутствуют, но универсальный selector остаётся главной нерешённой задачей.

### Требует осторожной формулировки

- нельзя приписывать все English V4.4 gains reranking из-за question-choice эффекта;
- нельзя считать почти 100 F1 B5-EN окончательной оценкой без более крупного внешнего test;
- нельзя представлять FINAL как независимую confirmatory победу;
- нельзя напрямую сравнивать абсолютный F1 SciMDIX и SciER как меру сложности;
- нельзя считать nonanswerable official spans ошибочными.

## 14. Рекомендуемые таблицы и рисунки статьи

1. Таблица происхождения, языка, длины, таксономии, splits, chunk size и answerable rate.
2. Диаграмма общего preprocessing pipeline и audit classes.
3. Распределение chunks/document для всех cell.
4. Диаграмма answerable rate по аспектам, особенно SciMDIX и RuSERRC.
5. Main official Token F1 table.
6. Paired deltas относительно B2/B4b с document bootstrap CI.
7. 128/24 vs 384/50 для SciER.
8. Ranker-factorial для RuSERRC/SciMDIX.
9. D1/P2 oracle-gap по cell.
10. Отдельная таблица exposure/leakage status и протокольных отклонений.

## 15. Литература и карточки датасетов, упомянутые в Markdown

### 15.1. SciMDIX

1. Batura, T., Yerimbetova, A., Mukazhanov, N., Shvarts, N., Sakenov, B., Turdalyuly, M. (2025). **Information Extraction from Multi-Domain Scientific Documents: Methods and Insights.** *Applied Sciences*, 15(16), 9086. DOI: [10.3390/app15169086](https://doi.org/10.3390/app15169086).
2. Shvarts, N., Batura, T., Mukazhanov, N., Yerimbetova, A., Turdalyuly, M., Sakenov, B. (2026). **SciMDIX: A dataset for aspect extraction from multi-domain scientific documents in Kazakh and Russian.** *Procedia Computer Science*, 275, 474–483. DOI: [10.1016/j.procs.2026.01.056](https://doi.org/10.1016/j.procs.2026.01.056).
3. Dataset repository mentioned in source card: [SciMDIX GitHub](https://github.com/tvbat/sci-text-miner-scimdix/tree/main) and Hugging Face `tvbat/SciMDIX`.

### 15.2. SciER

4. Zhang, Q., Chen, Z., Pan, H., Caragea, C., Latecki, L. J., Dragut, E. (2024). **SciER: An Entity and Relation Extraction Dataset for Datasets, Methods, and Tasks in Scientific Documents.** *Proceedings of EMNLP 2024*, 13083–13100. [ACL Anthology](https://aclanthology.org/2024.emnlp-main.726), DOI `10.18653/v1/2024.emnlp-main.726`.

Source README также упоминает supervised methods PURE, PL-Marker и HGERE как baseline исходной статьи SciER; в UnTIE benchmark они не переоценивались.

### 15.3. Литература, повлиявшая на label policy и benchmark design

5. Qu, Y., Ding, Y., Liu, J., et al. (2020). **RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering.** Denoised hard negatives.
6. Ni, A., Gardner, M., Dasigi, P. (2021). **Mitigating False-Negative Contexts in Multi-document Question Answering with Retrieval Marginalization.** DOI `10.18653/v1/2021.emnlp-main.497`. Multiple admissible evidence contexts.
7. Abdumalikov, R., Minervini, P., Kementchedjhieva, Y. (2024). **Answerability in Retrieval-Augmented Open-Domain Question Answering.** arXiv:2403.01461. Semantic relevance is not equivalent to answerability.
8. Wang, Z., Ng, P., Ma, X., Nallapati, R., Xiang, B. (2019). **Multi-passage BERT: A Globally Normalized BERT Model for Open-domain Question Answering.** Cross-passage score comparability.
9. Wang, S., Yu, M., Jiang, J., et al. (2017). **Evidence Aggregation for Answer Re-Ranking in Open-Domain Question Answering.** arXiv:1711.05116.
10. Hashimoto, K., Zhou, Y., Yavuz, S., Xiong, C., Yu, P. S. (2021). **Dense Hierarchical Retrieval for Open-Domain Question Answering.** Passage/document hierarchy.
11. Caciularu, A., Dagan, I., Goldberger, J., Cohan, A. (2021). **Long Context Question Answering via Supervised Contrastive Learning.** Evidence discrimination.
12. Ferguson, J., Hajishirzi, H., Dasigi, P., Khot, T. (2022). **Retrieval Data Augmentation Informed by Downstream Question Answering Performance.** DOI `10.18653/v1/2022.fever-1.1`.
13. Shen, T., Liu, M., Zhou, M., Xiong, D. (2022). **Recovering Gold from Black Sand: Multilingual Dense Passage Retrieval with Hard and False Negative Samples.** EMNLP 2022, [ACL Anthology](https://aclanthology.org/2022.emnlp-main.730/). Источник идеи adaptive false-negative filtering; проект не воспроизводит полный mHFN.
14. Tibshirani, R., Walther, G., Hastie, T. (2001). **Estimating the Number of Clusters in a Data Set Via the Gap Statistic.** DOI [10.1111/1467-9868.00293](https://doi.org/10.1111/1467-9868.00293). Упомянута для автоматического выбора K в K-means.
15. Jang, E., Gu, S., Poole, B. (2017). **Categorical Reparameterization with Gumbel-Softmax.** [Google Research](https://research.google/pubs/categorical-reparameterization-with-gumbel-softmax/). В проекте Gumbel-Softmax рассматривался как optional ablation; основной filter использовал deterministic hard assignment.

Эти работы используются как методическая мотивация. Benchmark является внутридокументным, поэтому не заявляется как точная репликация open-domain QA setups.

## 16. Что проверить перед подачей статьи

- получить официальный citation/license для RuSERRC и SciREX из первичных карточек;
- проверить license SciMDIX и допустимость распространения производных файлов;
- заменить secondary Consensus URLs в черновой библиографии на первичные ACL/arXiv/DOI;
- зафиксировать новую blind проверку final consistency gate;
- повторить resource benchmark без QA cache;
- приложить fold assignments, source hashes и exact model revisions;
- явно описать gold-side answerable filter до раскрытия системных predictions;
- отдельно сообщить coupled nature SciER-128/384;
- не включать OOD SciER как протестированный результат, если он не был оценён в финальном benchmark;
- отметить, что P3/RuSERRC имел freeze/refit protocol violation и не использовать его для центрального claim.

## 17. Основные внутренние источники

- `experiments/qa_chunk_selection/external_data/README.md`;
- `experiments/qa_chunk_selection/v4_4_ru_data/README.md`;
- `experiments/qa_chunk_selection/v4_4_ru_q5_b0/README.md`;
- `artifacts/qa_chunk_selection_external_data_20260913/PREPARATION_REPORT.md`;
- `artifacts/qa_chunk_selection_scier_384_50_20260913/CHUNKING_VARIANTS.md`;
- `artifacts/qa_chunk_selection_external_data_20260913/source_manifest.json`;
- `hypothesis/qa_chunk_selection_v4_4_external_datasets_results.md`;
- `hypothesis/qa_chunk_selection_v4_4_global_multilingual_results.md`;
- `hypothesis/qa_chunk_selection_experiments_v4_4_en_ru_comparison.md`;
- `hypothesis/qa_chunk_selection_v4_4_ru_ranker_factorial_results.md`;
- `hypothesis/result_final_official_split_h2_family_benchmark_analysis.md`;
- `hypothesis/final_method_all_variants_and_baselines_comparison.md`.

