# V5.0 — Multilingual Question-Conditioned Gated Residual Reranking
## План реализации многоязычного эксперимента для научно-технического extractive QA

**Статус:** development experiment / новая ветка после V4.5  
**Назначение:** техническая инструкция для ИИ-агента, который будет писать код  
**Основная задача:** проверить question-conditioned residual reranking поверх уже существующих систем без изменения и перезаписи предыдущих экспериментов.


**Ключевое ограничение:** исторические neural-модели и upstream pipeline сохраняются; новые обучаемые компоненты V5 — только лёгкие линейные meta-models над frozen/historical outputs.

**Версия документа:** reviewed/corrected. Исправлены nested-CV, SciER question-selection, label supervision, resource-budget semantics и правила междатасетной интерпретации.

---

# 0. Главная идея

V4.4/V4.5 показали, что:

1. качество первичного ранжирования фрагментов критически влияет на downstream QA;
2. правильный фрагмент часто уже находится внутри top-K;
3. простая повторяемость answer (`source-support`) и взаимная похожесть ответов (`SEM/majority`) не всегда позволяют определить лучший evidence;
4. особенно на RuSERRC тематически близкий, но аспектно неверный ответ может повторяться чаще правильного;
5. текущий multi-QA second stage дорог: он запускает QA на нескольких чанках даже тогда, когда исходный top-1 уже достаточно надёжен.

Поэтому V5 проверяет новый принцип:

> **второй этап должен оценивать не согласованность ответов между собой, а соответствие конкретного извлечённого ответа исходному аспектному вопросу и evidence-чанку, при этом дорогой top-K QA запускается только для вопросов, где первичный ранкер недостаточно уверен.**

Концептуальная схема:

```text
DOCUMENT
   ↓
historical B0 ranker
   ↓
ranking of chunks
   ↓
cheap uncertainty gate
   ├── confident
   │      ↓
   │   QA(top-1)
   │      ↓
   │   final answer
   │
   └── uncertain
          ↓
       QA(top-K)
          ↓
 question-conditioned
 residual switch scorer
          ↓
       final chunk
          ↓
       final answer
```

---

# 1. Что проверяет V5

## H1 — Question-conditioned residual signal

Даёт ли лёгкий residual scorer, использующий:

```text
question
predicted answer
candidate evidence
base-ranker score
```

дополнительное качество относительно исходного B0?

Primary comparison:

```text
V5-Q vs B0
```

## H2 — Gating / resource control

Можно ли запускать дорогой top-K QA только для части вопросов и сохранить большую часть качества V5-Q?

Primary comparison:

```text
V5-QG vs V5-Q
```

при меньшем:

```text
mean QA forwards/question
end-to-end latency
```

## H3 — Evidence safety

Не достигается ли прирост Token F1 ценой ухудшения выбора правильного supporting chunk?

Обязательно проверять:

```text
Hit@1
HN@1
positive→non-positive switches
non-positive→positive switches
```

---

# 2. Что НЕ является целью V5

V5 не должен:

- заменять QA-reader;
- fine-tune QA-reader;
- вводить новый большой Transformer;
- вводить новый cross-encoder;
- менять существующие ranker checkpoints;
- менять chunking;
- менять вопросы;
- переопределять labels;
- повторно искать лучший K;
- открывать locked test;
- перетирать V3/V4/V4.4/V4.5 artifacts;
- «спасать» V4.5 подбором новых gamma/tau;
- обучать отдельную модель на каждый aspect в primary experiment.

---

# 3. Используемые датасеты и экспериментальные ячейки

V5 должен быть реализован на **всех ранее использованных русских и английских корпусах этой линии**.

Корпуса:

```text
EN:
- SciREX
- SciER

RU:
- RuSERRC
- SciMDIX
```

Для SciER сохранить две исторические схемы чанкинга как две отдельные **protocol cells**:

```text
- SciER 384/50
- SciER 128/24
```

Важно:

> `SciER 384/50` и `SciER 128/24` — не два независимых датасета и не две независимые репликации. Это два представления одного и того же корпуса с различным candidate universe.

Поэтому:

- результаты двух SciER cells анализировать отдельно;
- не считать их независимыми наблюдениями в глобальном выводе;
- не усреднять их как два независимых корпуса;
- не использовать обе ячейки как двойное подтверждение одной гипотезы.

Если в проекте существуют дополнительные historical EN/RU cells этой же линии, агент должен:

1. найти их в global manifests;
2. перечислить;
3. **не включать автоматически**;
4. добавить только если заранее подтверждены:
   - document-grouped split;
   - воспроизводимый historical B0;
   - frozen QA reader;
   - корректный chunk-level label contract;
   - корректный candidate universe.

Добавление новой cell после просмотра V5 outer-EVAL запрещено и относится уже к V5.1.

---

# 4. Статус доказательности

Все перечисленные наборы уже участвовали в предыдущем model selection.

Следовательно:

```text
V5-A = development characterization
```

а НЕ:

```text
independent confirmation
```

Даже если outer CV выполнен без утечки.

Все CI V5-A на этих корпусах считать:

```text
development CI
```

Независимое подтверждение возможно только отдельным V5-B после final freeze на действительно untouched data.

---

# 5. Предыдущие результаты — строго read-only

Все каталоги и файлы:

```text
V3
V4
V4.1
V4.2
V4.3
V4.4
V4.5
```

считать:

```text
READ ONLY
```

Разрешено:

- читать;
- импортировать чистые функции;
- использовать frozen split definitions;
- использовать model identifiers;
- использовать cached QA predictions;
- использовать existing score maps;
- использовать existing normalization utilities;
- использовать existing metric implementations;
- использовать existing chunking;
- вычислять SHA-256;
- ссылаться на historical artifacts.

Запрещено:

- overwrite;
- patch historical JSON/JSONL;
- менять старый cache;
- менять old manifest;
- мигрировать old artifact in place;
- удалять;
- переименовывать;
- писать новые результаты внутрь old artifact directories.

Перед запуском:

```text
hash_before
```

После:

```text
hash_after
```

Хеши должны совпасть.

---

# 6. Новый каталог

Рекомендуется:

```text
experiments/qa_chunk_selection/v5_question_conditioned_residual/
```

Artifacts:

```text
artifacts/qa_chunk_selection_v5/
  run_<timestamp>/
```

Logs:

```text
logs/qa_chunk_selection_v5_<timestamp>.log
```

---

# 7. Переиспользование существующего кода

Переиспользовать готовые реализации разрешено и желательно, если это сохраняет историческую семантику эксперимента.

Разрешено переиспользовать:

```text
dataset loaders
chunk loaders
frozen split loaders
ranker inference
QA inference
semantic encoder inference
answer normalization
source-cluster calculation
metric implementations
bootstrap utilities
artifact hashing
```

Условия:

1. функция не записывает в historical artifact path;
2. её input/output contract совпадает с V5;
3. поведение подтверждается regression test;
4. нет silent fallback на другой config/model;
5. функция не использует outer-EVAL labels при построении runtime features;
6. функция не изменяет historical cache;
7. известны score direction и score semantics;
8. tokenizer/preprocessing совпадают с historical branch.

## 7.1. Не изменять historical experiment code без необходимости

Предпочтительная стратегия:

```text
import old pure helper
или
V5 wrapper/adapter around old helper
```

Не рекомендуется редактировать непосредственно модули V3/V4/V4.4/V4.5, даже если изменение кажется backward-compatible.

Если существующая функция недостаточна:

```text
создать V5-specific wrapper/copy
```

в новом namespace.

Это необходимо, чтобы старые эксперименты оставались воспроизводимыми на уровне кода, а не только данных.

## 7.2. Manifest переиспользования

Создать:

```text
reused_components.json
```

Для каждого компонента:

```text
source module
function/class
historical experiment
historical semantics
V5 wrapper path
modified = true/false
regression status
```

---

# 8. Никаких новых языковых моделей

QA-модели, энкодеры и neural rankers должны быть **теми же**, что использовались в соответствующих предыдущих экспериментах.

Запрещено:

- автоматически выбирать новую LM;
- заменять historical QA-reader;
- заменять historical semantic encoder;
- заменять historical B0 cross-encoder;
- добавлять новый Transformer/LLM как V5 scorer.

Для каждой cell восстановить exact:

```text
B0 ranker / ranking model
QA reader
semantic encoder, если он исторически использовался
tokenizers
revisions / commits
scoring formatter
```

Если exact model не удаётся восстановить:

```text
BLOCKED_MODEL_RECOVERY
```

Не угадывать checkpoint.

## 8.1. Что нового обучать разрешено

V5 вводит только **лёгкие классические meta-models поверх frozen/historical model outputs**:

```text
gate = logistic regression
residual switch = logistic regression
```

Они не заменяют historical ranker/QA/encoder и не являются новой языковой моделью.

В отчёте их называть:

```text
meta-classifier
residual switch model
```

а не новым neural reranker.

---

# 9. Model manifest per dataset

Создать:

```text
models/
  scirex_models.json
  scier_384_models.json
  scier_128_models.json
  russerrc_models.json
  scimdix_models.json
```

Для каждого сохранить:

```json
{
  "ranker": {
    "model_id": "...",
    "revision": "...",
    "tokenizer": "...",
    "role": "primary_b0"
  },
  "qa_reader": {
    "model_id": "...",
    "revision": "...",
    "tokenizer": "..."
  },
  "semantic_encoder": {
    "model_id": "...",
    "revision": "...",
    "historically_available": true
  }
}
```

---

# 10. Primary B0 system per dataset cell

V5 не должен насильно использовать один neural ranker на всех языках.

Единым является **алгоритм residual correction**, а historical B0 сохраняется для каждой cell.

## RU

Для:

```text
RuSERRC
SciMDIX
```

primary residual base:

```text
DiTy
```

точный checkpoint/revision и training recipe взять из V4.5.

BGE сохранить как:

```text
strong tested reference
```

но не использовать как primary residual base V5.

## EN

Для:

```text
SciREX
SciER 384/50
SciER 128/24
```

использовать exact historical B0 **system**, а не только название ranker checkpoint.

Под `B0 system` понимаются:

```text
ranker checkpoint
score formatter
question policy
candidate universe
chunking
K policy
score direction
fold-local training/calibration, если она была
```

Не заменять historical EN B0 на DiTy/BGE.

## 10.1. Primary meta-models не объединять между датасетами

В V5-A gate и residual switch обучаются **отдельно для каждой dataset cell**.

Не объединять в одну meta-model:

```text
SciREX + SciER
EN + RU
RuSERRC + SciMDIX
```

Причины:

- разные B0 score distributions;
- разные aspect taxonomies;
- разные candidate universes;
- разные QA readers;
- разные label contracts.

Общность метода проверяется через одинаковую архитектуру и training contract, а не через один общий fitted classifier.

---

# 11. Historical selected systems как references

В итоговой таблице V5 сохранить reference на ранее выбранные системы:

```text
SciREX historical V4.4 selected system
SciER-384 historical best/selected system
SciER-128 historical best/selected system
RuSERRC V4.5 DiTy / BGE references
SciMDIX V4.5 S3 / BGE references
```

Не переобучать их без необходимости.

Если historical predictions достаточны:

```text
read-only reference
```

---

# 12. Question policy

V5 не выполняет новый outer-level question search.

Для каждой dataset cell использовать historical question policy.

## RuSERRC

```text
Q0
```

## SciMDIX

```text
Q0
```

## SciREX

Использовать exact historical aspect question.

## SciER

Восстановить exact historical question policy V4.4.

Если question был выбран fold-locally из нескольких вариантов с использованием TRAIN/CAL labels, необходимо различать два уровня.

### Outer EVAL

Использовать historical/frozen question-selection procedure соответствующего outer fold, которая не использует outer EVAL.

### Inner META cross-fitting

Если question selection была data-dependent, её необходимо **повторить внутри каждого inner split только на inner-train/inner-cal data**.

Нельзя использовать для inner-validation вопрос, выбранный с использованием labels этой inner-validation части.

Если historical question policy невозможно воспроизвести nested-safe:

```text
BLOCKED_NESTED_QUESTION_POLICY
```

Не подменять её произвольным Q0 только ради запуска.

Если historical question был полностью фиксирован и не выбирался по данным, дополнительный inner selection не требуется.

---

# 13. K policy

V5 не ищет K заново.

Использовать exact historical K для соответствующей dataset cell.

Known expected values:

```text
RuSERRC: K=5
SciMDIX: K=4
```

Для EN:

```text
resolve from historical artifact
```

Не угадывать.

Сохранить:

```text
k_manifest.json
```

---

# 14. Candidate universe invariant

Для каждой dataset cell проверить все candidate chunk ids до top-K.

Candidate universe V5 должен совпадать с historical B0.

Сохранить:

```text
candidate_universe_hash
```

Если отличается:

```text
INVALID_COMPARISON
```

до выяснения причины.

---

# 15. Chunking invariant

Не менять:

- chunk boundaries;
- max tokens;
- overlap;
- detokenization;
- sentence segmentation;
- chunk ids;
- order;
- document text.

Для SciER две historical chunking cells сохраняются отдельно.

---

# 16. Label contract

Восстановить exact labels из historical datasets.

Не вводить новую разметку.

Минимально различать, где они существуют:

```text
positive / robust_positive
hard_negative
easy_negative
ambiguous
GSRN / special negative
```

Создать:

```text
label_contract_<dataset>.json
```

---

# 17. Что считается positive

Использовать historical definition positive.

Нельзя:

- превращать ambiguous в positive;
- считать GSRN positive;
- менять правила для удобства V5.

Если dataset имеет другую taxonomy, mapping должен быть явно описан.

---

# 18. Архитектура V5

Primary systems:

## B0

Historical primary ranker top-1 + QA(top-1).

## V5-Q

Всегда:

```text
B0 ranking
→ QA(top-K)
→ question-conditioned residual switch
```

Gate отсутствует.

Это проверяет чистую полезность нового residual signal.

## V5-QG

```text
B0 ranking
→ cheap gate
→ if confident:
      QA(top-1)
  else:
      QA(top-K)
      residual switch
```

Это resource-controlled система.

## HIST

Historical V4.4/V4.5 second-stage system.

Reference only.

## BGE

Только для тех RU cells, где BGE уже являлся historical strong reference.

---

# 19. Почему V5-Q нужен отдельно

Если сразу использовать gate, отрицательный результат невозможно разделить:

```text
новый residual signal плох
```

или:

```text
gate не пропускает нужные случаи
```

Поэтому сначала:

```text
V5-Q vs B0
```

а затем:

```text
V5-QG vs V5-Q
```

---

# 20. Cheap gate: runtime inputs

Gate работает **до top-K QA**.

Поэтому runtime gate не может использовать:

- predicted QA answer;
- QA confidence;
- source-support;
- answer-group features;
- semantic answer features;
- gold labels;
- gold answer;
- oracle utility.

Разрешены только признаки, доступные после B0 ranking и до QA.

Primary continuous features:

```text
effective_k
candidate_count
normalized_gap_12
normalized_gap_13
normalized_gap_1k
topk_score_std
topk_score_iqr
normalized_score_entropy
top1_zscore
```

Primary categorical context:

```text
aspect_id
```

`aspect_id` известен до inference, поскольку вопрос задаётся конкретным тематическим аспектом.

Использовать:

```text
OneHotEncoder(handle_unknown="ignore")
```

внутри одной pooled meta-model данного dataset.

Не обучать отдельный gate на каждый аспект.

---

# 21. Score normalization для gate

Не использовать только raw B0 probability/logit, потому что шкала различается между folds, seeds, rankers и datasets.

Для каждого вопроса получить vector:

\[
r=(r_1,\ldots,r_K)
\]

Вычислить:

\[
\mu_r,\sigma_r
\]

и:

\[
z_i = \frac{r_i-\mu_r}{\sigma_r+\epsilon}
\]

Primary:

```text
epsilon = 1e-8
```

Если:

```text
std < epsilon
```

все z-score = 0.

Normalized gaps:

\[
g_{12}=\frac{r_1-r_2}{\sigma_r+\epsilon}
\]

аналогично для `g13`, `g1k`.

---

# 22. Entropy feature

Для кросс-модельной устойчивости:

1. использовать z-normalized top-K scores;
2. применить softmax с fixed temperature 1.0;
3. считать entropy.

Не подбирать temperature.

---

# 23. Gate target

Primary target остаётся evidence-oriented.

```text
need_rerank_evidence = 1
```

если одновременно:

```text
B0 top1 не является sufficient-positive
AND
в effective top-K есть хотя бы один sufficient-positive
```

Иначе:

```text
0
```

Здесь `sufficient-positive` определяется только historical label contract конкретного датасета.

Это означает:

> в shortlist существует evidence-кандидат, который способен исправить ошибку B0.

## 23.1. Secondary oracle target

Отдельно вычислить только для диагностики:

```text
need_rerank_answer = 1
```

если внутри top-K существует кандидат, downstream QA Token F1 которого выше F1 текущего top1 не менее чем на заранее фиксированный diagnostic margin:

```text
1.0 F1 point
```

Этот target:

- не используется runtime;
- не используется для primary gate training;
- нужен для оценки того, сколько answer-quality opportunities evidence-target gate неизбежно пропускает.

## 23.2. Почему targets разделены

Primary gate оптимизирует корректность evidence.

Secondary target показывает потенциальный answer-level headroom среди уже sufficient candidates.

Не смешивать их в один post-hoc target после outer EVAL.

---

# 24. Почему gate target не равен Token-F1 oracle

Не использовать primary target:

```text
exists candidate with higher gold Token F1
```

потому что это сильнее привязывает gate к конкретной answer metric.

Но считать его как secondary diagnostic:

```text
need_rerank_f1_oracle
```

---

# 25. Gate model

Primary preprocessing/model:

```text
continuous columns:
    StandardScaler

categorical aspect_id:
    OneHotEncoder(handle_unknown="ignore")

classifier:
    LogisticRegression
```

Fixed classifier parameters:

```text
penalty = "l2"
C = 1.0
class_weight = "balanced"
solver = "liblinear"
max_iter = 2000
random_state = seed
```

Не делать hyperparameter sweep.

`class_weight="balanced"` допустим здесь, потому что gate target может быть сильно несбалансирован.

Scaler/encoder fit только на META_OOF training rows соответствующего outer fold/seed.

---

# 26. Gate output

Gate возвращает:

```text
p_need_rerank
```

Никаких aspect-specific gate models в primary.

---

# 27. Gate budget

Primary V5-QG использует заранее фиксированный **training-estimated budget target**:

```text
expected mean QA forwards/question <= 2.0
```

Этот budget выбирается только по META_OOF.

Важно:

> это не гарантирует, что фактическое среднее на outer EVAL будет строго <=2.0, потому что распределение `effective_k` и gate probabilities может измениться.

На outer EVAL запрещено менять threshold для «дотягивания» бюджета.

Нужно только измерить:

```text
actual mean QA forwards/question
actual routing rate
actual latency
```

---

# 28. Как выбрать gate threshold

На META_OOF:

1. получить `p_need_rerank`;
2. для каждого вопроса знать `effective_k`;
3. симулировать стоимость:
   - not routed → 1 QA forward;
   - routed → `effective_k` QA forwards;
4. рассмотреть уникальные probability cutoffs;
5. выбрать **наименьший threshold / максимальную routing coverage**, при которой:

```text
META_OOF expected mean QA forwards <= 2.0
```

Tie-break:

```text
выше threshold
```

то есть более консервативный routing.

Gold utility/F1 при выборе threshold не использовать.

После freeze threshold переносится на outer EVAL без изменения.

---

# 29. Secondary budget curve

Predeclared descriptive budgets на META_OOF:

```text
1.25
1.50
2.00
FULL
```

Для каждого budget зафиксировать threshold до outer EVAL.

На outer EVAL для каждого frozen threshold report:

```text
Token F1
Hit@1
HN@1
actual QA forwards/question
actual routing rate
latency
```

`FULL` = V5-Q без gate.

Primary V5-QG:

```text
META_OOF budget target = 2.0
```

Не выбирать лучший budget по outer-EVAL качеству как primary.

---

# 30. Frozen QA

QA reader не обучается.

Используется exact historical checkpoint каждого dataset.

QA output:

```text
predicted answer
```

никогда:

```text
gold answer
```

---

# 31. QA cache reuse

Переиспользовать cached answer только если совпадают:

```text
question hash
chunk hash
QA model id
QA revision
tokenizer
decoding config
answer preprocessing
```

Создать:

```text
qa_cache_audit.json
```

---

# 32. QA confidence

QA confidence полезен для residual scorer, но historical cache может не хранить logits.

Поэтому:

## Primary core V5

должен уметь работать **без QA confidence**.

## QA-confidence extension

Разрешена только если:

- exact QA checkpoint доступен;
- tokenizer exact;
- decoding exact;
- recomputed answer совпадает с historical cached answer;
- logits извлекаются без изменения decoding.

Если условия не выполняются:

```text
qa_conf_feature = unavailable
```

Не подставлять искусственный confidence.

---

# 33. QA confidence definition

QA-confidence **не входит в primary V5**.

Он допускается только как predeclared secondary extension, если historical QA implementation позволяет восстановить confidence без изменения decoded answer.

Если historical QA уже сохранял совместимый span score:

```text
reuse exact historical score
```

Если score не сохранялся, но logits можно корректно получить повторным inference:

1. использовать exact historical QA checkpoint/tokenizer;
2. использовать exact historical input construction;
3. использовать exact historical decoding;
4. сначала выполнить QA replay audit;
5. decoded span/answer должен совпадать с historical cache для всех проверяемых записей или для заранее заданной исчерпывающей выборки, если полный replay технически невозможен.

Только после успешного audit допускается один заранее фиксированный confidence:

\[
Conf_i = \log P(start_i)+\log P(end_i)
\]

или exact historical equivalent.

Нельзя выбирать между несколькими confidence definitions по outer-EVAL качеству.

Если replay не проходит:

```text
qa_conf_feature = unavailable
```

---

# 34. Question-conditioned feature: ranker Q↔A

Для каждого predicted answer:

\[
A_i = QA(Q,C_i)
\]

вычислить:

\[
QArel^{rank}_i = Ranker(Q,A_i)
\]

Критически важно использовать **тот же экземпляр/состояние upstream ranker, которое создало B0 scores для этой строки**.

То есть:

## Inner-validation при dataset-trained B0

```text
inner B0 model
→ B0 chunk scores
→ тот же inner B0 model для Q-A scores
```

## Outer EVAL

```text
outer-fold B0 model
→ B0 chunk scores
→ тот же outer-fold B0 model для Q-A scores
```

## Fully frozen B0

Использовать один и тот же frozen checkpoint.

Нельзя строить B0 score одним fold-model, а `Q-A` score другим checkpoint.

Это новый способ применения historical ranker, но не новая language model.

Использовать тот же score space и score direction, что historical ranking implementation:

```text
raw logit vs probability
```

не смешивать между B0 и Q-A feature молча.

Внутри shortlist feature затем нормализуется, поэтому абсолютная cross-model calibration не требуется.

---

# 35. Q-A ranker input

Использовать native historical pair formatter ranker-а.

Не добавлять новый prompt.

Для short answer:

```text
query = Q
document = A_i
```

или exact native equivalent.

---

# 36. Semantic encoder Q↔A

Если в соответствующей historical branch уже использовался semantic encoder, разрешается вычислить:

\[
QArel^{enc}_i = \cos(E(Q),E(A_i))
\]

Использовать exact historical encoder.

Не подменять отсутствующий encoder новым.

---

# 37. Question-answer-evidence SupportGain

`SupportGain` не входит в primary V5.

Разрешён только как secondary extension.

Предпочтительно переиспользовать exact implementation из предыдущей экспериментальной линии, если он уже существовал.

Если exact implementation восстановлен:

\[
BaseFit_i = \cos(E(Q),E(C_i))
\]

\[
AnswerFit_i = \cos(E(Q+A_i),E(C_i))
\]

\[
SupportGain_i=AnswerFit_i-BaseFit_i
\]

При этом необходимо восстановить exact:

```text
text concatenation
separator
normalization
encoder pooling
truncation
```

Если exact historical implementation не удаётся доказать:

```text
SupportGain = unavailable in V5 primary/secondary report
```

Не создавать новую вариацию SupportGain внутри текущего protocol.

---

# 38. Semantic feature availability

Semantic Q-A cosine / SupportGain разрешены только если:

- используется exact historical semantic encoder;
- известны pooling и normalization;
- preprocessing воспроизводим;
- context length не требует новой схемы обрезки;
- feature runtime-computable на новом документе.

Если любое условие нарушено:

```text
semantic extension = unavailable
```

Не заменять отсутствующий encoder новым и не менять truncation ради V5.

---

# 39. Historical source-support

Переиспользовать existing source-cluster implementation.

Для каждого candidate, если historical implementation это поддерживает:

```text
source_count
source_cluster_count
exact_answer_group_size
```

Не считать source-support главным правилом.

Теперь это только feature.

---

# 40. Primary residual candidate features

Для каждого candidate внутри routed/full top-K сформировать candidate-level frozen features.

Required:

```text
b0_score_norm
q_answer_ranker_score_norm
effective_k
candidate_count
aspect_id
```

Если historical implementation доступен:

```text
source_cluster_norm
answer_group_size_norm
```

Optional при технически корректной доступности:

```text
q_answer_encoder_cos_norm
support_gain_norm
qa_conf_norm
answer_valid
```

Дополнительно сохранить question-level B0 uncertainty context:

```text
normalized_gap_12
normalized_gap_13
normalized_score_entropy
top1_zscore
```

Эти context features одинаковы для всех candidate pairs данного вопроса, но помогают residual model учитывать, насколько исходный B0 был уверен.

---

# 41. Feature normalization

Candidate-dependent continuous features нормализовать **внутри одного question shortlist**.

Для feature `x_i`:

\[
z_i=\frac{x_i-\mu_x}{\sigma_x+\epsilon}
\]

Если feature constant:

```text
all z = 0
```

Это принципиально важно после V4.5, где absolute source bonus плохо сочетался со шкалой B0.

---

# 42. Pairwise difference и context features

Для alternative candidate `i` относительно candidate `j`:

\[
\Delta x_{ij}=x_i-x_j
\]

Для runtime основной comparison:

```text
j = B0 top1
```

Primary pairwise vector:

```text
ΔB0
ΔQ-A-ranker
Δsource, если доступно
Δanswer-group, если доступно

effective_k
candidate_count
normalized_gap_12
normalized_gap_13
normalized_score_entropy
top1_zscore
aspect_id
```

Optional:

```text
ΔQ-A-encoder
ΔSupportGain
ΔQA-confidence
answer_valid_i
answer_valid_j
```

`aspect_id` кодируется one-hot.

Question-level context **не вычитать**, поскольку он одинаков для пары и иначе исчезнет.

---

# 43. Почему difference features

Residual model должен отвечать:

> лучше ли candidate i исходного top-1?

а не:

> насколько candidate i хорош вообще?

Это сохраняет historical B0 как основной ranker.

---

# 44. Residual supervision contract

Primary pairwise supervision строится на historical source/evidence labels, но не должна слепо исключать `ambiguous`, если его семантика в конкретном датасете однозначно означает недостаточность evidence.

Для каждого dataset создать:

```text
residual_supervision_contract_<dataset>.json
```

и распределить labels по трём группам:

## A. sufficient_positive

Фрагмент считается достаточным evidence согласно historical contract.

## B. insufficient_evidence

Фрагмент явно не является достаточным evidence.

Обычно сюда относятся:

```text
hard_negative
easy_negative
```

Также могут входить:

```text
ambiguous
GSRN
```

**только если** historical annotation contract явно определяет их как insufficient/non-answerable для текущей задачи.

## C. unresolved_for_training

Labels, семантика которых не позволяет однозначно задать pairwise preference.

Их:

```text
не использовать как supervision
```

но сохранять в runtime/evaluation.

Primary training pairs:

```text
sufficient_positive > insufficient_evidence
```

внутри одного question top-K.

Это особенно важно для RuSERRC, где существенная часть полезных исправлений исторически имела тип:

```text
ambiguous → positive
```

Поэтому blanket-exclusion `ambiguous` запрещён без проверки label contract.

---

# 45. Multiple positives и same-tier pairs

Если top-K содержит несколько `sufficient_positive`:

- не создавать positive-vs-positive supervision в primary;
- не использовать gold Token F1 для определения победителя между ними;
- разрешить runtime model сравнивать их, но учитывать такие переключения отдельно.

Если top-K содержит несколько insufficient candidates:

- не создавать supervision между ними в primary.

Причина:

> primary residual model учится исправлять evidence sufficiency, а не оптимизировать gold answer metric внутри одной evidence-категории.

Отдельно report:

```text
positive→positive switches
```

и их downstream F1 effect.

---

# 46. Question-level weighting

Если вопрос создаёт `N_q` training pairs:

\[
w_{pair}=\frac{1}{N_q}
\]

Суммарный вклад каждого question должен быть сопоставим.

Не позволять документам с большим candidate count доминировать.

---

# 47. Residual switch model

Primary preprocessing/model:

```text
continuous difference/context features:
    StandardScaler

aspect_id:
    OneHotEncoder(handle_unknown="ignore")

classifier:
    LogisticRegression
```

Fixed classifier parameters:

```text
penalty = "l2"
C = 1.0
class_weight = None
solver = "liblinear"
max_iter = 2000
random_state = seed
```

Почему `class_weight=None`:

> mirrored pair construction делает бинарные классы симметричными; дополнительный automatic class balancing здесь не нужен и может нарушить заранее заданное question-level weighting.

Никакого MLP/Transformer в primary.

---

# 48. Residual target orientation

Training rows симметризовать.

Для пары `(positive, negative)`:

```text
Δfeatures = positive - negative
label = 1
```

и mirrored:

```text
Δfeatures = negative - positive
label = 0
```

Pair total weight делить на 2.

Зафиксировать:

```text
use_mirrored_pairs = true
```

---

# 49. Runtime residual decision

Исходный:

```text
C0 = B0 top1
```

Для каждого alternative:

```text
Ci, i > 0
```

вычислить:

```text
p_i = P(Ci preferred over C0)
```

Primary switch rule:

```text
best_i = argmax(p_i)

if p_best > 0.5:
    select C_best
else:
    keep C0
```

При:

```text
p_best <= 0.5
```

или numerical tie:

```text
keep B0 top1
```

Threshold `0.5` фиксирован заранее.

Не подбирать residual switch threshold по outer EVAL.

Причина: V5 должен сначала проверить полезность самого question-conditioned signal без нового threshold search.

Если позже понадобится calibrated conservative threshold, это отдельный V5.1.

---

# 50. Если несколько alternatives

Выбирать candidate с максимальным `p_i`.

Только один switch.

После switch не выполнять iterative reranking.

---

# 51. V5-Q

Для каждого question:

```text
B0 top-K
→ QA all effective top-K
→ feature extraction
→ residual switch
```

Без gate.

Это наиболее чистый тест H1.

---

# 52. V5-QG

Для каждого question:

1. B0 ranking;
2. gate features;
3. `p_need_rerank`;
4. если gate false:
   - QA B0 top1;
5. если gate true:
   - QA effective top-K;
   - residual switch;
6. final answer.

---

# 53. Важное требование к resource benchmark

Нельзя заранее вычислить QA(top-K) для всех вопросов и потом объявить gate экономным.

Для **quality evaluation** cache reuse разрешён.

Для **runtime benchmark** V5-QG должен реально выполнять только те QA forwards, которые требуются gate decision.

---

# 54. Nested CV — критическое требование

Meta-models нельзя обучать на признаках, которые получили unfair in-sample advantage от dataset-trained upstream components.

Но требование зависит от того, **как был получен historical B0**.

## Case A — B0 обучался/fine-tuned на текущем dataset

Нельзя:

```text
train B0 on documents
→ score те же documents
→ train gate/residual on these scores
```

Нужен inner document-grouped cross-fitting.

## Case B — B0 является полностью frozen/off-the-shelf

Если ranker:

- не обучался на текущем dataset;
- не калибровался на его labels;
- не выбирался fold-locally по labels в рамках текущего pipeline,

то переобучать ranker во внутренних folds не требуется.

Такой frozen ranker можно использовать для получения meta-features outer non-EVAL документов напрямую.

## Case C — mixed pipeline

Если сам ranker frozen, но data-dependent были:

```text
question selection
score calibration
reranker weights
other upstream choices
```

то именно эти data-dependent компоненты должны воспроизводиться nested-safe.

Для каждой cell создать:

```text
upstream_training_provenance.json
```

с типом:

```text
dataset_trained
fully_frozen
mixed
```

и обоснованием.

---

# 55. Outer folds

Использовать exact historical document-grouped outer folds.

Для каждой dataset cell:

```text
Outer NON-EVAL
Outer EVAL
```

Outer EVAL полностью недоступен для:

- gate training;
- residual training;
- feature selection;
- threshold;
- budget calibration;
- coefficient choice.

---

# 56. Inner meta-feature generation

Для каждой outer cell/seed действовать по `upstream_training_provenance`.

## 56.1. Если B0 dataset-trained или pipeline mixed

1. взять только outer non-EVAL documents;
2. создать deterministic document-grouped inner folds;
3. default:
   ```text
   INNER_META_FOLDS = 3
   ```
4. для каждого inner fold:
   - обучить/восстановить только те upstream components, которые historical pipeline обучал на данных;
   - выполнить data-dependent question selection/calibration только на inner-train/inner-cal;
   - score inner-validation;
   - получить historical top-K;
   - frozen QA на inner-validation top-K;
   - вычислить V5 meta-features;
5. объединить inner-validation rows в `META_OOF`.

Ни один inner-validation document не должен влиять на fitted upstream component, который создал его features.

## 56.2. Если B0 fully frozen

Inner retraining B0 не требуется.

Разрешено:

```text
frozen B0
→ score outer non-EVAL documents
→ build meta-training rows
```

поскольку B0 не обучался на этих labels/documents в рамках dataset.

Но gate/residual всё равно обучаются только на outer non-EVAL и никогда не видят outer EVAL labels.

## 56.3. Общий результат

Создать:

```text
META_TRAIN
```

с explicit provenance каждого row.

Не называть rows `META_OOF`, если upstream B0 не обучался и OOF concept к нему неприменим; в manifest использовать:

```text
meta_feature_origin = inner_oof | frozen_upstream
```

---

# 57. Малые датасеты и inner folds

Особенно проверить SciMDIX.

Если nested cross-fitting требуется, но 3-fold split нарушает minimum supervision support:

```text
INNER_META_FOLDS = 2
```

Переключение 3→2 разрешено только по pre-run feasibility.

Minimum conditions для каждого inner train:

```text
есть sufficient_positive
есть insufficient_evidence
есть оба gate target classes, если возможно
```

Если gate target имеет только один класс в отдельном inner fold:

- не менять labels;
- inner predictions всё равно сохранить;
- финальный gate для outer fold можно обучать только если объединённый META_TRAIN содержит оба класса.

Если объединённый META_TRAIN содержит один gate class:

```text
gate_status = degenerate
```

и применять deterministic safe fallback:

```text
route = false
```

для V5-QG данного outer run.

V5-Q residual evaluation при этом может продолжаться, если residual supervision валидна.

---

# 58. Не использовать historical outer OOF других folds как meta-training shortcut

Нельзя собирать meta-training для текущего outer fold из historical OOF других outer folds, если модели, создавшие эти features, могли обучаться на текущем outer EVAL.

Это потенциальная meta-level leakage.

---

# 59. Meta-model training

На `META_TRAIN` соответствующего outer fold/seed:

1. fit gate preprocessing;
2. fit gate logistic regression;
3. fit residual preprocessing;
4. fit residual switch logistic regression;
5. определить gate threshold только по resource-budget rule;
6. сохранить coefficients/scalers/encoders/threshold;
7. freeze.

После этого:

- восстановить historical outer-fold B0 system;
- score outer EVAL;
- построить runtime features;
- применить frozen gate/residual.

Outer EVAL labels не используются до финального metric computation.

Если historical outer B0 model отсутствует и должен быть обучен:

```text
train on exact allowed outer non-EVAL data
using exact historical recipe
```
до scoring EVAL.

---

# 60. Meta-model calibration

Gate probability calibration не требует отдельного Platt/isotonic в primary.

Использовать raw logistic probability.

Residual:

```text
threshold = 0.5
```

---

# 61. Gate threshold — только resource calibration

Gate threshold не оптимизируется по F1/Hit@1.

На `META_TRAIN`:

```text
choose threshold
→ target expected mean QA forwards <= 2.0
→ maximize routing coverage within this cost
```

Gold target нужен для обучения gate, но не для выбора budget threshold.

На outer EVAL:

```text
threshold frozen
```

Запрещено:

- менять threshold по фактическому EVAL budget;
- менять threshold по EVAL F1;
- выбирать другой budget после просмотра EVAL как primary.

Report actual outer-EVAL cost separately.

---

# 62. Base ranker handling внутри meta generation

Использовать `upstream_training_provenance`.

## Dataset-trained B0

Inner folds требуют отдельного B0 training/reuse с exact historical recipe.

Historical model можно переиспользовать только если:

```text
its training document set excludes the current inner-validation documents
AND
its config exactly matches required historical recipe
```

## Fully frozen B0

Не переобучать.

Использовать exact historical frozen checkpoint напрямую.

## Mixed

Переобучить/перекалибровать только data-dependent компоненты.

Не вводить новый hyperparameter search.

---

# 63. QA внутри inner cross-fit

QA reader frozen, поэтому его можно применять непосредственно к inner-validation chunks.

QA не обучается на текущем dataset в V5.

---

# 64. Semantic encoder внутри inner cross-fit

Semantic encoder frozen.

Можно batch/cache embeddings в новом V5 cache.

Не писать в historical cache.

---

# 65. New cache namespace

Создать:

```text
artifacts/.../cache/
  qa/
  ranker_qa_relevance/
  semantic/
  meta_features/
```

Cache key минимум:

```text
dataset
outer_fold
inner_fold
seed
model_revision
question_hash
chunk_hash / answer_hash
feature_version
```

---

# 66. Seeds

Использовать historical project seeds:

```text
42
43
44
```

Meta-model random_state соответствует seed.

Нельзя выбирать лучший seed.

---

# 67. Один meta-model на seed

Для каждого:

```text
dataset
outer_fold
seed
```

обучить отдельные:

```text
gate
residual switch
```

из meta-features того же seed.

---

# 68. Не смешивать seeds при training

Не объединять rows seeds 42/43/44 в один training dataset для meta-model primary.

Иначе seed replicas становятся псевдоданными.

---

# 69. Feature availability contract

Создать:

```text
feature_contract_<dataset_cell>.json
```

## Primary-required features для всех cells

```text
b0_score
q_answer_ranker
effective_k
candidate_count
aspect_id
b0_uncertainty_context
```

Эти признаки должны существовать во всех EN/RU cells.

Если `q_answer_ranker` невозможно вычислить exact historical ranker-ом:

```text
CELL_BLOCKED_PRIMARY_V5
```

## Secondary historical extensions

```text
source_support
answer_group
semantic_q_answer
support_gain
qa_conf
answer_valid
```

Их отсутствие не блокирует primary V5.

Для каждого feature сохранить:

```text
source model/function
runtime computable
requires top-K QA
historical implementation
score direction
missing policy
primary/secondary status
```

Feature с `runtime computable=false` запрещён.

---

# 70. Primary common feature set

Чтобы V5 был действительно одним методом на всех русских и английских корпусах, **primary residual feature set фиксируется одинаково для всех cells**.

## A0 — control residual model

```text
candidate:
- ΔB0

question context:
- effective_k
- candidate_count
- normalized_gap_12
- normalized_gap_13
- normalized_score_entropy
- top1_zscore
- aspect_id
```

## A1 — PRIMARY V5 question-conditioned model

```text
A0
+
- ΔQ-A-ranker
```

Именно `A1` используется для primary:

```text
V5-Q
V5-QG
```

на всех dataset cells.

Таким образом primary experiment напрямую отвечает:

> добавляет ли соответствие `Question ↔ PredictedAnswer` полезную информацию сверх B0 residual/context?

Historical source-support, semantic encoder и QA-confidence не входят в primary V5.

---

# 71. Secondary extensions

Predeclared secondary extensions:

## A2 — source extension

Если historical source/group implementation существует:

```text
A1
+
Δsource-support
Δanswer-group
```

## A3 — semantic/confidence extension

Если exact historical implementations доступны:

```text
A2
+
ΔQ-A-encoder
ΔSupportGain
ΔQA-confidence
```

Каждая extension должна быть технически объявлена доступной **до outer EVAL**.

A2/A3 нужны только для анализа:

> добавляют ли старые сигналы что-либо поверх нового question-conditioned core?

Они не могут заменить A1 как primary после просмотра результатов.

---

# 72. Primary system identity across datasets

Primary V5 для каждой cell:

```text
residual feature set = A1
gate feature set = одинаковый contract из B0 uncertainty + aspect_id
```

Это правило нельзя менять по технической доступности secondary features.

Следовательно:

```text
SciREX V5-Q
SciER V5-Q
RuSERRC V5-Q
SciMDIX V5-Q
```

проверяют одну и ту же концепцию.

Secondary A2/A3 могут иметь разную доступность и поэтому не используются для основного междатасетного вывода.

---

# 73. Никакого feature-set selection по outer EVAL

Запрещено после результатов выбирать:

```text
A1 vs A2 vs A3
```

как «лучшую V5».

Primary всегда:

```text
A1
```

Secondary tables могут показывать A2/A3, если они были predeclared и технически доступны.

Если A2/A3 превосходит A1:

> это exploratory/secondary evidence для следующей версии, но не основание переписать primary V5.

---

# 74. Predeclared feature ablations

Обязательный механизмный contrast:

```text
A1 - A0
```

Он отвечает:

> даёт ли `Question ↔ PredictedAnswer` relevance дополнительный сигнал сверх B0 uncertainty/context?

Secondary:

```text
A2 - A1
A3 - A2
```

если соответствующие extensions доступны.

Для каждой ablation использовать тот же:

```text
outer split
seed
meta-feature provenance
residual training contract
switch threshold
```

Ablations не используются для post-hoc смены primary system.

---

# 75. Historical V4.x features не добавлять автоматически

Не включать без отдельного contract:

- SEM answer-answer centrality;
- LEX score;
- Q5 score;
- V6 handcrafted total score;
- historical router confidence.

Source-support разрешён как отдельный raw feature.

---

# 76. Primary metrics

Главная:

```text
Token F1
```

Secondary:

```text
Character F1
ROUGE-L
Exact Match
BERTScore if historically defined
Lemma F1 for RU
Unicode ROUGE-L for RU
```

Использовать exact historical metric implementations.

---

# 77. Evidence metrics

Обязательно:

```text
MRR
Hit@1
Hit@K
HN@1
HN@K
F1 | selected positive
```

При наличии:

```text
Amb@1
GSRN@1
```

---

# 78. Gate metrics

Report:

```text
AUROC
PR-AUC
precision
recall
routing rate
actionable-positive prevalence
```

Но gate AUROC не является primary success metric.

---

# 79. Residual switch metrics

Report:

```text
number of changed top1
good_switch
bad_switch
neutral_switch
positive→positive
positive→non-positive
non-positive→positive
non-positive→non-positive
```

---

# 80. Good/bad definition

Primary:

```text
good_switch:
downstream Token F1 increases

bad_switch:
downstream Token F1 decreases

neutral:
F1 unchanged but chunk changed
```

Evidence transitions report independently.

---

# 81. Gate missed opportunities

Post-hoc diagnostic:

```text
gate_false
AND
there exists positive candidate in top-K
AND
B0 top1 non-positive
```

Это gold-dependent diagnostic.

Не использовать runtime.

---

# 82. Statistical unit

Seeds не считать независимыми observations.

Для paired comparison:

1. metric contribution per `(question_id, seed)`;
2. average seeds within question;
3. bootstrap document clusters.

---

# 83. Bootstrap

```text
iterations >= 5000
seed = 42
cluster = document_id
```

Report:

```text
observed_delta
CI_low
CI_high
```

---

# 84. Primary comparisons per dataset

```text
H1:
V5-Q vs B0

H2:
V5-QG vs V5-Q

H3:
V5-QG vs B0
```

Historical V4.x and BGE:

```text
secondary/reference
```

---

# 85. H1 acceptance

Development acceptance:

```text
ΔToken F1 > 0
```

желательно:

```text
development CI lower > 0
```

Safety:

```text
Hit@1 >= B0 - 0.5 pp
HN@1 <= B0 + 1.0 pp
```

Если F1 растёт, но safety нарушена:

```text
H1 unsafe
```

---

# 86. H2: оценка gate как quality–cost trade-off

Не использовать произвольную absolute non-inferiority границу как основной критерий.

V5-QG оценивается одновременно по качеству и стоимости.

H2 считается поддержанной в development-смысле, если V5-QG:

1. использует меньше фактических QA forwards/question, чем V5-Q;
2. имеет меньшую end-to-end latency, чем V5-Q;
3. не хуже B0 по Token F1;
4. проходит evidence-safety;
5. является non-dominated точкой относительно B0 и V5-Q по:

```text
Token F1 ↑
QA forwards ↓
latency ↓
```

Дополнительно report `GainRetention`:

\[
GainRetention =
\frac{F1_{QG}-F1_{B0}}
     {F1_{Q}-F1_{B0}}
\]

только если:

```text
F1(V5-Q) > F1(B0)
```

Если gain V5-Q мал, ratio интерпретировать осторожно.

Gate threshold всё равно определяется только META_TRAIN budget rule, не outer-EVAL качеством.

---

# 87. Cross-dataset interpretation

Не требовать один и тот же effect size.

Самый сильный результат:

```text
V5-Q positive on multiple EN and RU dataset cells
without material harm elsewhere
```

Если RU positive, EN null:

```text
question-conditioned residual may be most useful where B0 has aspect-confusion errors
```

Если EN positive, RU null:

```text
не заявлять language effect без дополнительного анализа
```

---

# 88. Absolute F1 между datasets не сравнивать

Сравнивать:

```text
Δ vs own B0
```

из-за различий:

- aspect taxonomy;
- candidate universe;
- chunking;
- QA reader;
- answer lengths;
- labels.

---

# 89. SciER caveat

SciER historical result смешивал:

```text
question policy
+
ranking
+
QA
```

V5 не должен заново выбирать question.

Question text должен быть frozen из historical branch.

Иначе V5 effect станет неидентифицируемым.

---

# 90. RuSERRC special diagnostic

Обязательно проверить:

```text
top-K positive recall
oracle headroom
gate coverage of known residual errors
```

потому что V4.5 показал:

```text
positive почти всегда уже в shortlist
```

---

# 91. SciMDIX special diagnostic

Проверить:

> сохраняет ли V5 положительный эффект, который historical source-support уже давал?

Сравнить:

```text
V5-Q
vs
V4.5 S3
```

secondary.

---

# 92. English special diagnostic

Для SciREX/SciER проверить:

> question-conditioned Q-A signal добавляет ли что-то поверх уже полезных reader signals?

Historical best remains reference.

---

# 93. Resource benchmark — обязательный

V5 должен исправить ошибку VRAM измерения V4.5.

Нельзя использовать:

```python
torch.cuda.max_memory_allocated()
```

без указания фактического device.

---

# 94. Correct GPU memory measurement

Для model device:

```python
device = next(model.parameters()).device
torch.cuda.reset_peak_memory_stats(device)
...
torch.cuda.synchronize(device)
peak = torch.cuda.max_memory_allocated(device)
```

Если pipeline использует несколько GPU:

- измерять каждый;
- report total and max-device separately.

---

# 95. Fresh-process benchmark

CPU RSS и GPU peak для разных systems измерять в **отдельном fresh subprocess**.

Не benchmark несколько больших systems последовательно в одном процессе для memory comparison.

Для latency/throughput report два режима:

## Single-query latency

```text
batch of one question/request
```

Primary для интерактивного deployment.

## Batched throughput

Использовать заранее фиксированную practically relevant batching policy.

Не смешивать single-query latency и batch throughput в одной колонке.

Все сравниваемые systems запускать:

```text
same machine
same software environment
same precision policy where supported
same input sample
```

---

# 96. Benchmark systems

Для каждой dataset cell:

```text
B0
V5-Q
V5-QG
historical second-stage if useful
```

RU дополнительно:

```text
BGE B0
```

---

# 97. Ranker-only benchmark

Сравнить ranker inference отдельно.

Но main resource conclusion строить по:

```text
end-to-end extraction pipeline
```

---

# 98. End-to-end benchmark

Считать реальные runtime steps.

## B0

```text
rank all candidates
QA(top1)
```

## V5-Q

```text
rank all candidates
QA(top-K)
Q-A ranker scoring
semantic features if enabled
residual scorer
```

## V5-QG

```text
rank all candidates
gate
if confident:
    QA(top1)
else:
    QA(top-K)
    Q-A ranker scoring
    semantic features
    residual scorer
```

---

# 99. QA cache и benchmark

Для scientific quality run cache разрешён.

Для runtime benchmark:

- нельзя считать cached QA retrieval как QA inference;
- выполнять actual model forwards.

---

# 100. Resource metrics

Primary:

```text
loaded parameter count
weights disk size
peak GPU VRAM
peak CPU RSS
median latency/question
p95 latency
throughput
mean QA forwards/question
mean extra ranker forwards/question
```

Secondary:

```text
training time meta-model
base-ranker retraining time
GPU-hours
```

---

# 101. Extra ranker cost

V5 использует ranker(Q,A).

Поэтому отдельно считать:

```text
mean Q-A ranker forwards/question
```

Эти inputs короткие, но стоимость не считать нулевой.

---

# 102. Semantic encoder cost

Если EXTENDED features используются:

```text
encoder forwards
encoder VRAM
latency
```

включить в end-to-end.

---

# 103. Gate overhead

Логистическая regression overhead измерить, даже если ожидается negligible.

---

# 104. Resource curves

Для frozen gate thresholds, соответствующих META_TRAIN budget targets:

```text
1.25
1.50
2.00
FULL
```

построить на outer EVAL:

```text
Token F1 vs actual QA forwards/question
Token F1 vs actual single-query latency
Token F1 vs actual end-to-end throughput
```

Не перестраивать threshold на outer EVAL для точного попадания в budget.

Это одна из центральных V5 figures.

---

# 105. Pareto analysis

Для systems compare:

```text
Token F1
peak VRAM
latency
```

Не создавать один artificial combined score.

Report Pareto frontier.

---

# 106. Не требовать V5 > BGE

Для RU BGE является strong reference.

V5 считается полезным даже если:

```text
F1(V5) < F1(BGE)
```

при условии содержательного quality-resource trade-off.

---

# 107. Improvement слабого B0 недостаточен сам по себе

В отчёте обязательно показывать BGE рядом.

Нельзя утверждать practical superiority только потому, что:

```text
V5 > DiTy
```

---

# 108. Regression tests

До full run обязательны:

1. old artifact hash before;
2. old artifact hash after;
3. exact dataset source;
4. exact split;
5. candidate universe equality;
6. chunking equality;
7. question equality;
8. ranker model identity;
9. QA model identity;
10. semantic encoder identity;
11. K identity;
12. score direction;
13. QA cache identity;
14. no gold in runtime features;
15. meta-feature OOF provenance;
16. no outer EVAL in meta train;
17. no duplicate question-seed;
18. finite scores;
19. strict serialization;
20. GPU memory correct device;
21. fresh-process memory benchmark;
22. metric regression against historical B0.

---

# 109. Gold leakage test

Runtime model inputs/features не должны содержать:

```text
gold answer
gold F1
source label
positive flag
negative type
oracle rank
```

Gold допускается только для:

```text
training targets
CAL diagnostics
evaluation
post-hoc analysis
```

---

# 110. Predicted answer may equal gold

Не проверять leakage через:

```text
gold string not contained in predicted answer
```

Predicted QA может корректно совпасть с gold.

Проверять **provenance**:

```text
answer field generated exclusively by frozen QA
```

---

# 111. Meta-feature provenance

Для каждой training row сохранить:

```text
question_id
document_id
outer_fold
inner_fold
base_model_training_docs_hash
feature_model_did_not_train_on_document = true
```

---

# 112. Smoke test

Сначала:

```text
1 dataset
1 outer fold
1 seed
small inner cross-fit
```

Проверить:

```text
base ranking
gate features
inner OOF
QA
Q-A scoring
semantic features
pair generation
gate training
residual training
runtime switch
metrics
resource instrumentation
artifact isolation
```

Smoke results не входят в scientific report.

---

# 113. Execution stages

Рекомендуемый порядок:

## Stage 0 — protocol recovery

Восстановить exact:

```text
datasets
dataset cells
splits
candidate universes
models
questions/question-selection procedures
K
metrics
label contracts
historical references
upstream training provenance
```

## Stage 1 — regression tests

Без нового outer-EVAL evaluation.

## Stage 2 — smoke

Один outer fold/seed без использования его результатов для изменения frozen scientific contract.

## Stage 3 — contract freeze

Создать и захешировать:

```text
contract_frozen.json
```

до первого полного outer-EVAL просмотра.

## Stage 4 — full V5-A execution

Запустить все заранее включённые cells по одному и тому же frozen contract:

```text
RuSERRC
SciMDIX
SciREX
SciER 384/50
SciER 128/24
```

Порядок вычисления технически может быть любым.

## Stage 5 — global analysis

Только после завершения всех cells.

Не использовать результаты более ранней cell для изменения primary protocol последующих.

---

# 114. RuSERRC как рекомендуемый технический smoke target

RuSERRC удобен для smoke/debugging, потому что:

- top-K headroom уже хорошо диагностирован;
- historical V4.5 stage2 = B0;
- strong BGE reference известен.

Но:

> полный RuSERRC outer-EVAL нельзя использовать как пилот для изменения V5 перед запуском EN/SciMDIX.

Разрешено:

```text
малый smoke subset
synthetic/debug run
один fold без просмотра scientific aggregate
```

для проверки кода.

После contract freeze полный результат любой dataset cell считается частью V5-A и не должен менять protocol других cells.

---

# 115. Contract freeze

До первого полного outer-EVAL анализа создать:

```text
contract_frozen.json
```

Содержимое минимум:

```text
included dataset cells
upstream training provenance rules
inner-fold policy
question-selection nested policy
feature definitions
feature availability rules
normalization
gate model
gate target
gate budget rule
residual supervision tiers
residual model
switch threshold = 0.5
outer folds
seeds
metrics
safety rules
resource benchmark
missing-feature handling
```

SHA-256 contract сохранить.

После hash freeze:

- никакой cell не меняет primary contract другой cell;
- любые новые идеи → V5.1.

---

# 116. Нельзя адаптировать protocol по результатам одной cell

После просмотра любого полного outer-EVAL результата запрещено:

- менять gate features;
- менять residual features;
- менять pair labels;
- менять switch threshold;
- менять budget target;
- добавлять/удалять aspect feature;
- менять question policy;
- менять K;
- менять primary feature hierarchy

для оставшихся cells.

Если обнаружен **технический bug**, который делает результат невалидным:

1. зафиксировать bug;
2. пометить affected runs invalid;
3. исправить код;
4. перезапустить **все затронутые cells**, которым изменение могло повлиять на результат;
5. сохранить audit trail.

Не исправлять методологию под уже увиденный результат.

---

# 117. Artifact structure

```text
artifacts/qa_chunk_selection_v5/
  run_<timestamp>/

    status.json
    COMPLETED.json
    contract_frozen.json
    config_resolved.json

    inputs/
      datasets.json
      split_hashes.json
      candidate_universe_hashes.json
      model_manifests/
      question_manifests/
      k_manifest.json
      label_contracts/

    regression/
      hash_before.json
      hash_after.json
      historical_metric_replay.json
      candidate_universe_check.json
      model_identity_check.json
      qa_cache_audit.json
      meta_feature_leakage_check.json
      gpu_memory_check.json

    reused_code/
      reused_components.json

    cache/
      qa/
      ranker_qa/
      semantic/
      meta_features/

    scirex/
      outer_fold_0/
        seed_42/
          inner_meta/
          gate/
          residual/
          eval/
      oof/

    scier_384/
      ...

    scier_128/
      ...

    russerrc/
      ...

    scimdix/
      ...

    resource_benchmark/
      environment.json
      scirex/
      scier_384/
      scier_128/
      russerrc/
      scimdix/

    analysis/
      primary_metrics.json
      paired_bootstrap.json
      gate_metrics.json
      switch_metrics.json
      evidence_transitions.json
      budget_curves.json
      seed_stability.json
      fold_stability.json
      aspect_metrics.json
      resource_tradeoff.json
      global_summary.json

    reports/
      results_v5.md
```

---

# 118. Что сохранять для каждого outer run

```text
dataset
outer_fold
seed

base ranker model id/revision
QA model id/revision
semantic encoder id/revision

inner split hashes
inner model hashes/configs

gate feature names
gate coefficients
gate scaler
gate threshold
predicted p_need_rerank

residual feature names
residual coefficients
residual scaler
switch probabilities

B0 ranking
top-K candidate ids
QA answers
final selected chunk
final answer

runtime counts
```

---

# 119. Interpretability

Поскольку gate и residual models линейные:

Report standardized coefficients:

```text
gate coefficients
residual coefficients
```

Но не интерпретировать magnitude как causal effect.

---

# 120. Coefficient stability

Для каждого feature report:

```text
sign across folds
median coefficient
IQR
```

Особенно:

```text
ΔQ-A-ranker
Δsource
ΔB0
```

Если sign хаотичен, отметить instability.

---

# 121. Feature importance claim

Не писать:

> feature X causes correct evidence.

Разрешено:

> feature X consistently receives positive/negative coefficient in the fitted residual model.

---

# 122. Aspect analysis

После primary results:

```text
per aspect:
N
B0 F1
V5-Q F1
V5-QG F1
Hit1
HN1
routing rate
```

Никакого aspect-specific retraining в V5.

---

# 123. Candidate-count analysis

Bucket:

```text
effective K = 1
2
3
4
5+
```

Report:

```text
gate rate
V5 gain
```

Если K=1:

```text
no reranking possible
```

---

# 124. Gate hard rule K=1

Если:

```text
effective_k <= 1
```

то:

```text
route = false
```

не спрашивать logistic model.

---

# 125. No-action shortlist

Если top-K не содержит alternative candidate после filtering:

```text
keep B0
```

---

# 126. Missing predicted answer

Если QA returns no valid answer:

использовать exact historical no-answer semantics.

Для Q-A features:

```text
valid=false
```

Не подставлять gold.

Не отправлять literal `[NO_ANSWER]` в ranker без historical precedent.

Создать:

```text
answer_valid
```

как runtime feature только если отсутствие ответа может корректно возникнуть при inference.

---

# 127. Missing feature handling

Для missing technical feature:

- impute 0 after within-question standardization;
- add missing-indicator bit только если missing может легально возникнуть runtime.

Не impute based on label.

---

# 128. Short answer truncation

Q-A ranker input должен помещаться в historical max length.

Не менять tokenizer.

Если predicted answer длиннее:

- использовать exact QA decoded answer;
- ranker tokenizer truncation следует historical pair formatter.

Сохранить truncation rate.

---

# 129. Semantic encoder truncation

Если encoder truncates:

- использовать historical encoder max length;
- report truncation rate;
- не менять chunk text.

---

# 130. Training compute plan

До full run оценить фактическое число upstream trainings.

Не использовать грубую формулу:

```text
outer × seed × inner × cells
```

без учёта provenance.

Для каждой cell записать:

```text
B0 provenance:
  fully_frozen | dataset_trained | mixed

outer model trainings
inner model trainings
question-selection/calibration reruns
QA inference count
semantic encoder inference count
meta-model fits
```

Сохранить:

```text
compute_plan.json
```

Это особенно важно, потому что fully-frozen historical B0 не требует inner ranker retraining.

---

# 131. Reuse of historical trained models

Разрешено переиспользовать existing fold model только если:

```text
training document set
exactly equals required inner/outer training set
```

и exact model artifact сохранён.

Нельзя использовать historical model, trained on superset including validation document.

---

# 132. If fold models were not saved

Retrain using exact historical recipe.

Не восстанавливать model state из predictions.

---

# 133. Training checkpoint storage

Чтобы не раздувать disk:

- сохранять final selected base models only where needed;
- inner models можно удалять **после**:
  - сохранения meta-features;
  - сохранения config/hash;
  - сохранения reproducibility manifest.

Deletion policy зафиксировать до run.

Не удалять historical files.

---

# 134. Primary global report

Для каждой cell таблица:

```text
B0
historical second-stage
V5-Q
V5-QG
strong reference if available
```

Поля:

```text
Token F1
Δ vs B0
CI
Hit@1
HN@1
MRR
QA forwards/question
latency
VRAM
```

---

# 135. Cross-language / cross-dataset summary

Итоговая таблица:

```text
Language
Corpus
Protocol cell
B0
V5-Q Δ
V5-QG Δ
routing rate
QA forwards
resource change
```

Правила:

- raw F1 между datasets не усреднять;
- SciER-384 и SciER-128 явно помечать как две preprocessing cells одного корпуса;
- не считать две SciER cells независимыми подтверждениями;
- не объединять seeds как независимые samples;
- не строить один «средний multilingual F1».

Можно descriptive report:

```text
число корпусов с положительным знаком Δ
число corpus/cells с CI выше 0
```

но без формального pooled effect, если заранее не задан корректный hierarchical analysis.

---

# 136. Success scenarios

## A. A1 improves B0 on multiple EN and RU corpora

Сильная development-поддержка общего question-conditioned residual mechanism.

SciER 384/50 и 128/24 считать одной corpus-level линией, а не двумя независимыми подтверждениями.

## B. Positive RuSERRC, neutral SciMDIX

Содержательный результат:

> V5 решает residual problem, выявленную V4.5 на RuSERRC, но не обязательно превосходит уже работающий SciMDIX source-support case.

## C. Positive SciMDIX, RuSERRC null

Question-conditioned signal не решает полностью RuSERRC aspect-confusion.

Не усложнять primary post-hoc.

## D. V5-Q works, V5-QG loses most gain

Residual mechanism полезен, но cheap gate недостаточен.

Resource-control остаётся нерешённым.

## E. V5-QG is Pareto-useful

Наиболее сильный resource-oriented outcome:

```text
V5-QG > B0 по качеству
V5-QG дешевле V5-Q
evidence safety сохранена
```

и система находится на quality–cost Pareto frontier.

Не требовать фактического outer-EVAL cost строго <=2.0: `2.0` — training-estimated budget target.

---

# 137. Failure scenarios

## V5-Q worse than B0

Зафиксировать.

Не увеличивать model complexity в том же protocol.

## Gate routes wrong cases

Отдельно анализировать gate target/feature mismatch.

## Resource advantage absent

Не утверждать resource efficiency.

---

# 138. V5-B

После V5-A, если результат перспективен:

1. выбрать **одну** final architecture;
2. заморозить feature set;
3. заморозить gate budget;
4. заморозить threshold;
5. заморозить base model per language/dataset policy;
6. найти untouched external test;
7. создать provenance audit;
8. только затем final evaluation.

---

# 139. Нельзя использовать SciREX/SciER/RuSERRC/SciMDIX как V5-B, если они уже использованы

Новый random split старых документов:

```text
не делает test независимым
```

---

# 140. Final scientific formulation при успешном V5

Потенциальная формулировка:

> Разработан механизм адаптивной остаточной коррекции ранжирования фрагментов научно-технического документа, в котором необходимость дополнительной обработки определяется по неопределённости первичного ранжирования, а решение о замене исходного фрагмента принимается на основе соответствия извлечённого замороженной QA-моделью ответа исходному аспектному вопросу и локальному evidence.

Использовать только если experiment supports it.

---

# 141. Что нельзя заявлять

Нельзя:

- «V5 подтверждён независимо» на reused datasets;
- «второй компонент универсален» при mixed results;
- «ресурсно эффективнее BGE» без end-to-end benchmark;
- «Q-A relevance причинно определяет правильность»;
- «gate знает, что ответ неправильный»;
- «меньшая модель автоматически дешевле pipeline».

---

# 142. Короткая инструкция агенту

```text
Implement V5 as a NEW isolated multilingual development experiment.

NEVER modify or overwrite V3/V4/V4.4/V4.5 artifacts or historical
experiment modules unless an explicitly approved shared utility change is
proven behavior-preserving. Prefer V5 wrappers/adapters.

Corpora:
EN:
- SciREX
- SciER

SciER protocol cells:
- 384/50
- 128/24
These are NOT independent datasets.

RU:
- RuSERRC
- SciMDIX

Recover exact historical:
- candidate universes
- document-grouped folds
- chunking
- question/question-selection policy
- K
- B0 ranker checkpoints and training provenance
- QA readers
- semantic encoders
- tokenizers/revisions
- metric implementations

Never guess missing model IDs.

Use the same historical language models.
No new Transformer/encoder/QA/neural reranker.

New trainable V5 components are only:
- logistic-regression gate
- logistic-regression residual switch

Primary systems:
B0
V5-Q  = always top-K QA + residual switch
V5-QG = cheap pre-QA gate + conditional top-K QA + residual switch

RU B0:
exact V4.5 DiTy.
BGE remains strong reference.

EN B0:
exact historical B0 SYSTEM for each cell.

Gate runtime features:
only pre-QA B0 uncertainty/candidate features + known aspect_id.
No gold and no QA-derived feature.

Gate target:
B0 top1 is not sufficient evidence AND a sufficient-positive exists in top-K.

Gate:
StandardScaler/OneHotEncoder + LogisticRegression(
    C=1.0,
    penalty="l2",
    solver="liblinear",
    class_weight="balanced",
    max_iter=2000
).

Gate threshold:
choose on META_TRAIN by resource budget only.
Primary budget target:
expected QA forwards/question <= 2.0.
Do NOT retune threshold on outer EVAL.

Primary residual feature set A1 MUST be identical across all cells:
ΔB0
ΔQ-A ranker relevance
B0 uncertainty context
effective K
candidate count
aspect_id

Historical source/group features, semantic/SupportGain and QA confidence
are SECONDARY predeclared extensions only.
They must NOT be added to primary V5-Q/V5-QG.

Residual supervision:
sufficient_positive > labels explicitly defined as insufficient evidence.
Do not blanket-drop ambiguous/GSRN; use dataset label contract.
Unknown/unresolved labels are evaluation-only.

Residual:
StandardScaler/OneHotEncoder + mirrored-pair LogisticRegression(
    C=1.0,
    penalty="l2",
    solver="liblinear",
    class_weight=None,
    max_iter=2000
).

Runtime switch threshold:
0.5 fixed.

CRITICAL META-LEAKAGE RULE:
If historical B0/upstream component was dataset-trained,
generate meta-features by inner document-grouped cross-fitting.
If B0 was fully frozen/off-the-shelf, inner ranker retraining is not needed.
If question selection/calibration was data-dependent, reproduce that
selection nested-safely inside inner folds.

Outer EVAL is evaluation only.

Seeds:
42, 43, 44.

Primary quality metric:
Token F1.

Evidence safety:
Hit@1
HN@1
evidence transitions.

Resource benchmark:
real end-to-end execution.
V5-QG must actually skip top-K QA for non-routed questions.
Do not use cache hits as inference timing.

Fix V4.5 memory measurement:
explicit actual CUDA device.
Use fresh subprocesses for memory benchmarks.

Freeze contract BEFORE first full outer-EVAL analysis.
Do not change primary protocol after seeing any full dataset result.

All previous results remain read-only.
```

---

# 143. Pre-run checklist

```text
[ ] historical artifact paths resolved
[ ] hash_before saved

[ ] exact SciREX source/folds resolved
[ ] exact SciER source/folds resolved
[ ] SciER 384/50 candidate universe resolved
[ ] SciER 128/24 candidate universe resolved
[ ] exact RuSERRC source/folds resolved
[ ] exact SciMDIX source/folds resolved

[ ] SciER two cells marked same corpus / non-independent
[ ] candidate universes verified
[ ] historical chunking verified
[ ] K verified for every cell

[ ] question policy recovered for every cell
[ ] data-dependent question-selection provenance identified
[ ] nested-safe SciER question-selection rule verified if needed

[ ] EN B0 system/checkpoints recovered
[ ] RU DiTy exact checkpoint recovered
[ ] BGE references recovered
[ ] QA readers recovered
[ ] semantic encoders recovered where historically used
[ ] revisions/tokenizers recovered

[ ] upstream_training_provenance.json created per cell
[ ] each B0 marked fully_frozen / dataset_trained / mixed

[ ] label contracts frozen
[ ] residual supervision tiers frozen
[ ] ambiguous/GSRN handling explicitly documented
[ ] metrics frozen

[ ] gate features frozen
[ ] aspect_id handling frozen
[ ] gate target frozen
[ ] gate model/solver frozen
[ ] META_TRAIN budget target=2.0 frozen

[ ] primary residual A1 frozen identically across all cells
[ ] Q-A relevance uses the same upstream ranker instance/state as B0 for every row
[ ] A2/A3 secondary feature availability frozen before EVAL
[ ] residual model/solver frozen
[ ] mirrored-pair weighting frozen
[ ] switch threshold=0.5 frozen

[ ] inner meta-fold policy frozen where required
[ ] fully-frozen B0 cells exempted from unnecessary retraining
[ ] meta provenance test passed
[ ] QA cache audit passed

[ ] resource benchmark environment frozen
[ ] actual CUDA device memory instrumentation tested
[ ] fresh-process RSS benchmark tested
[ ] single-query latency policy frozen
[ ] batch-throughput policy frozen

[ ] smoke run passed
[ ] contract_frozen.json hashed
[ ] no full outer-EVAL result viewed before freeze
```

---

# 144. Итоговая цель

V5 должен дать честный и разложимый ответ:

### 1.
Есть ли полезная информация в соответствии:

\[
Question \leftrightarrow PredictedAnswer
\]

которая позволяет лучше различать top-K evidence, чем текущие source/consensus heuristics?

### 2.
Можно ли использовать эту информацию как **residual correction**, не заменяя основной ranker?

### 3.
Можно ли запускать эту коррекцию только для неопределённых вопросов и тем самым сохранить качество при значительно меньшем числе QA-forward?

### 4.
Воспроизводится ли эта закономерность на ранее исследованных английских и русских научных корпусах при неизменных исторических моделях и preprocessing?

Любой результат — положительный, смешанный или отрицательный — должен быть сохранён без изменения protocol после просмотра outer-EVAL.
