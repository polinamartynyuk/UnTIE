# V6 / UAR-G — Universal Adaptive Residual Reranking
## Полный план реализации эксперимента для ИИ-агента

**Статус:** development protocol после V4.4 → V4.5 → V5  
**Назначение:** инструкция для ИИ-агента, который должен написать код, провести эксперимент и сформировать воспроизводимые артефакты  
**Главная цель:** проверить единый универсальный ресурсо-адаптивный метод остаточного ранжирования для EN/RU научно-технических документов, который использует уже существующие исторические модели и сигналы, но принимает решение о дополнительной обработке и переключении кандидата по прогнозу положительной downstream-полезности.


**Версия:** reviewed/corrected v2.  
**Ключевые исправления этой версии:** строгий meta-level OOF tuning; глобальный POOLED outer-fold с действительно shared weights/thresholds; question/corpus-family/lineage weighting; cross-cell embargo для SciER; scale-invariant EN/RU features; quality-aware gate selection under fixed compute budget; точный контракт S1/S2; conditional interactions для source/LEX; deterministic degenerate fallbacks; запрет смешивания seed-replicas; разделение development-selection и independent confirmation.

---

# 0. Главные обязательные ограничения

## 0.1. НЕЛЬЗЯ изменять, удалять, перезаписывать или мигрировать historical результаты

Считать **строго read-only** все артефакты, относящиеся к:

```text
V1
V2
V3
V4
V4.1
V4.2
V4.3
V4.4
V4.5
V5
```

В особенности нельзя писать в:

```text
artifacts/qa_chunk_selection_v4_4*/
artifacts/qa_chunk_selection_v4_5_ru/
artifacts/qa_chunk_selection_v5/
```

и в любые historical experiment/result folders, обнаруженные в manifests.

Также запрещено изменять:

```text
historical datasets
historical split files
historical candidate-universe files
historical QA caches
historical ranker caches
historical semantic caches
historical manifests
historical reports
historical frozen contracts
historical checkpoints
```

## 0.2. НЕЛЬЗЯ ломать код воспроизведения V4.4/V4.5/V5

Предпочтительная стратегия:

```text
READ historical implementation
        ↓
reuse pure helper / adapter
        ↓
WRITE new V6-specific wrapper/module
```

Не редактировать historical experiment modules даже если изменение кажется backward-compatible.

Если старой функции недостаточно, создать V6-specific implementation или adapter/wrapper в новом namespace.

Разрешение на изменение shared utility допускается только если одновременно:
1. доказано, что utility действительно shared;
2. historical behavior покрыт regression test;
3. изменение полностью backward-compatible;
4. до и после изменения воспроизводятся historical outputs;
5. `git diff` содержит только минимальное изменение;
6. historical artifact hashes не изменяются.

Предпочтительно всё равно избегать такого изменения.

## 0.3. Новый namespace

Создать:

```text
experiments/qa_chunk_selection_v6/
configs/qa_chunk_selection/v6_universal/
artifacts/qa_chunk_selection_v6_universal/
logs/qa_chunk_selection_v6/
```

Ни один новый файл не должен сохраняться внутри V4.x/V5 directories.

## 0.4. Git safety

До любых изменений сохранить:

```bash
git status --porcelain
git rev-parse HEAD
git diff
```

в:

```text
artifacts/qa_chunk_selection_v6_universal/<RUN>/audit/git_before/
```

Если repository уже имеет пользовательские незакоммиченные изменения:
- не отменять их;
- не делать `git reset`;
- не делать `git checkout -- .`;
- не делать `git clean`;
- не перезаписывать изменённые пользователем файлы.

После завершения сохранить аналогичный snapshot в:

```text
audit/git_after/
```

---

# 1. Научная мотивация V6

V4.4 показал, что second-stage signals способны существенно повышать качество, но их полезность неодинакова по корпусам.

V4.5 показал, что сильный ranker может почти полностью снять проблему chunk selection, а более компактный ranker оставляет headroom; при этом source-support полезен на SciMDIX, но не универсален на RuSERRC.

V5 показал, что глобальный линейный residual вида

\[
w_1\Delta B0+w_2\Delta QArel
\]

не универсален, а mirrored construction фактически занулил влияние `K`, margin, entropy и aspect на склон `ΔQ-A`.

Следовательно, V5 не проверил полноценную conditional hypothesis.

Центральная гипотеза V6:

\[
Utility(signal)
=
f(candidate\ state,K,ranker\ uncertainty,aspect,evidence\ signals)
\]

Новый метод должен оценивать вероятность того, что конкретное переключение улучшит downstream Token F1 относительно текущего B0 top-1.

---

# 2. Рабочее название

```text
UAR-G
Universal Adaptive Residual Reranking with Gating
```

Русское описание:

> универсальное адаптивное остаточное ранжирование с условным запуском дополнительной обработки.

---

# 3. Архитектура

```text
DOCUMENT
   ↓
historical compact B0 ranker
   ↓
ranked chunks
   ↓
pre-QA benefit gate
   ├── low expected benefit
   │       ↓
   │    QA(top-1)
   │       ↓
   │    final answer
   │
   └── possible benefit
           ↓
        QA(top-K)
           ↓
   candidate features:
   B0 / Q-A / source / LEX / SEM / QA
           ↓
 interaction-aware
 utility scorer
           ↓
 calibrated safe-switch
           ↓
 final evidence
           ↓
 final answer
```

---

# 4. Что означает «универсальный»

Допустимо использовать разные historical ranker checkpoints для EN и RU.

Едиными должны быть:
- feature schema;
- objective;
- interaction schema;
- gate logic;
- safe-switch logic;
- nested training procedure;
- evaluation protocol;
- decision rules.

То есть универсальность архитектурная/алгоритмическая, а не «одна и та же LM для всех языков».

---

# 5. Dataset / protocol cells

Обязательно:

```text
EN:
- SciREX
- SciER 384/50
- SciER 128/24

RU:
- RuSERRC
- SciMDIX
```

Важно: SciER 384/50 и SciER 128/24 — две protocol cells одного corpus family.

Создать:

```text
scirex      → corpus_family=scirex
scier_384   → corpus_family=scier
scier_128   → corpus_family=scier
russerrc    → corpus_family=russerrc
scimdix     → corpus_family=scimdix
```

---

# 6. Статус данных

Все пять cells уже участвовали в разработке V4.4/V4.5/V5.

Следовательно:

```text
V6-A = development characterization
```

После final freeze нужен отдельный:

```text
V6-B = confirmatory untouched test
```

Минимально желательно:
- один untouched EN corpus/test;
- один untouched RU corpus/test.

Новый random split уже просмотренных документов не является независимым test.

---

# 7. Historical models

В primary V6 не вводить новую pretrained LM.

Использовать exact historical models.

Known references:

```text
EN ranker:
cross-encoder/ms-marco-MiniLM-L6-v2
или exact historical checkpoint из manifests

RU compact ranker:
DiTy/cross-encoder-russian-msmarco

RU strong reference:
BAAI/bge-reranker-v2-m3
```

QA:
```text
exact historical QA reader per corpus
```

Semantic encoder:
```text
exact historical semantic encoder only if already used
```

Если exact checkpoint/revision восстановить нельзя:

```text
BLOCKED_MODEL_RECOVERY
```

Не угадывать.

---

# 8. Model identity contract

Для каждого model artifact сохранить:

```text
model_id
revision
checkpoint_path
checkpoint_sha256
tokenizer_id
tokenizer_revision
precision
max_length
formatter
score_direction
```

Если revision не записан, но локальный checkpoint существует:

```text
revision_source = local_historical_artifact
```

и вычислить SHA-256.

---

# 9. Canonical baseline audit — до V6 training

До любого нового обучения восстановить baseline semantics.

Особенно проверить SciER, потому что V4.4 raw B0 и V5 B0 имели разные protocol semantics.

Создать:

```text
audit/canonical_baselines.json
```

Для каждой cell:

```json
{
  "dataset_cell": "...",
  "corpus_family": "...",
  "historical_raw_b0": {},
  "v4_4_selected_system": {},
  "v4_5_selected_system": {},
  "v5_b0": {},
  "v5_primary": {},
  "question_policy": "...",
  "chunking_policy": "...",
  "candidate_universe": "...",
  "ranker": "...",
  "qa_reader": "...",
  "k": 0,
  "notes": "..."
}
```

Не переименовывать historical systems задним числом.

---

# 10. Historical replay

До V6 training воспроизвести минимум:
- historical B0 ranking;
- historical selected chunk;
- historical final answer;
- historical Token F1.

Для SciER обязательно replay fold-local question policy.

Если replay не совпадает:

```text
STOP
HISTORICAL_REPLAY_FAILED
```

---

# 11. Hash protection

Перед run вычислить hashes critical historical files:

```text
audit/historical_hashes_before.json
```

После run:

```text
audit/historical_hashes_after.json
```

Требование:

```text
before == after
```

Иначе:

```text
RUN_INVALID_HISTORY_MODIFIED
```

---

# 12. Candidate universe invariant

Не менять:
- документы;
- chunks;
- chunk ids;
- chunk text;
- chunk ordering;
- overlap;
- max length;
- detokenization;
- sentence segmentation.

Сохранить:

```text
candidate_universe_sha256
```

---

# 13. Question policy

Использовать exact historical policy.

RU:
```text
RuSERRC → Q0
SciMDIX → Q0
```

SciREX:
```text
exact historical fixed aspect question
```

SciER:
если question selection была fold-local/data-dependent, outer policy восстанавливать только из outer TRAIN/CAL; inner META generation должен повторять selection nested-safe.

---

# 14. K policy

Не искать K заново.

Known:
```text
RuSERRC = 5
SciMDIX = 4
```

EN K восстановить из manifests.

Создать:

```text
audit/k_manifest.json
```

---

# 15. Upstream provenance

Для каждой cell:

```text
fully_frozen
dataset_trained
mixed
```

Сохранить:

```text
audit/upstream_training_provenance.json
```

---

# 16. Nested anti-leakage protocol

## 16.1. LOCAL outer CV

Для каждой dataset cell сохранить historical outer protocol:

```text
5 document-grouped folds
group = canonical_document_id
seeds = 42,43,44
```

`canonical_document_id` должен быть устойчив к различным схемам чанкинга одного документа.

Для SciER обе protocol cells обязаны ссылаться на один и тот же canonical source document.

## 16.2. POOLED global outer CV — исправленная схема

POOLED должен действительно означать **одни shared meta-weights и shared thresholds внутри каждого outer fold/seed**.

Для global fold `f`:

```text
POOLED_EVAL_f = union(
    SciREX outer fold f,
    SciER-384 outer fold f,
    SciER-128 outer fold f,
    RuSERRC outer fold f,
    SciMDIX outer fold f
)
```

Затем:

```text
POOLED_TRAIN_f = все разрешённые development rows
                 минус любые canonical documents,
                 попавшие в POOLED_EVAL_f
```

Fit:

```text
ONE gate model
ONE utility model
ONE calibration policy
ONE tau_switch
ONE tau_gate
```

на каждый `(global_fold, seed)`.

Эти shared components применяются ко **всем** cells в `POOLED_EVAL_f`.

Нельзя обучать отдельный pooled model для SciREX, отдельный для RuSERRC и затем называть это shared-parameter POOLED.

## 16.3. Cross-cell embargo для SciER

Если canonical SciER document находится в EVAL хотя бы одной SciER cell:

> любые представления этого документа из обеих SciER cells исключаются из POOLED_TRAIN.

Проверять:

```text
canonical source id
normalized full-document SHA-256
```

Создать:

```text
audit/cross_cell_document_lineage.json
```

## 16.4. Cross-corpus overlap audit

До POOLED/LOCO-meta проверить exact/full-text hash overlap между всеми corpus families.

Если один исходный документ обнаружен в нескольких families:

```text
same lineage → one leakage group
```

Любое представление EVAL document исключается из training независимо от dataset label.

## 16.5. Inner upstream cross-fitting

Default:

```text
4 document-grouped folds
```

Fallback:

```text
4 → 3 → 2
```

только по pre-run feasibility.

## 16.6. Seed isolation

Seeds 42/43/44 — это repeated stochastic realizations, а не независимые training examples.

Запрещено:

```text
concatenate seed42 + seed43 + seed44 rows
→ train one meta-model
```

Для каждого seed обучается отдельный:

```text
upstream realization
meta-model
gate
calibration
threshold set
```

Seeds объединяются только на этапе статистической оценки после получения predictions.

---

# 17. Outer workflow и два уровня cross-fitting

Для каждого `(outer/global fold, seed)`:

```text
EVAL
└── sealed

ALLOWED TRAIN+CAL
│
├── upstream cross-fitting where required
│      ↓
│   honest META_FEATURE rows
│
├── meta-model cross-fitting
│      ↓
│   META_PRED_OOF
│
├── choose/tune ONLY from META_PRED_OOF:
│      C
│      calibration
│      tau_switch
│      tau_gate
│
├── refit selected meta-components on all allowed TRAIN+CAL
│
└── one pass EVAL
```

Критическое различие:

```text
META_FEATURE_OOF
```

= upstream-derived features без self-fit leakage;

```text
META_PRED_OOF
```

= predictions gate/utility model на rows, которые не участвовали в fit соответствующей meta-model.

Hyperparameters, calibration и thresholds выбираются только по `META_PRED_OOF`.

## 17.1. Meta-level OOF

Для каждого candidate hyperparameter:

1. document-grouped meta split;
2. fit preprocessing на meta-train;
3. fit meta-model на meta-train;
4. predict meta-validation;
5. объединить только validation predictions.

Нельзя:

```text
fit on all META rows
→ tune threshold on same fitted rows
```

## 17.2. POOLED meta folds

Внутри POOLED outer fold meta folds также строятся по `canonical_document_id` на объединённом TRAIN pool.

Все representations одного canonical document, включая две SciER chunking cells, должны попадать в **одну и ту же meta fold**.

## 17.3. Upstream honesty

Для dataset-trained/mixed upstream каждый row должен иметь upstream features от модели/selection procedure, которая не обучалась на этом row/document.

Сохранить:

```text
upstream_feature_origin = inner_oof | frozen_upstream
```

## 17.4. Fold-specific vs dataset-specific thresholds

В CV numerical values `tau_switch/tau_gate` могут отличаться между outer folds/seeds, потому что они fit на разных TRAIN samples.

Но в POOLED внутри одного `(global_fold, seed)` threshold **один для всех corpus cells**.

Запрещены dataset-specific thresholds.

---

# 18. Fully frozen exception

Если B0 полностью frozen/off-the-shelf и не обучался на corpus:

не переобучать его бессмысленно внутри META folds.

Можно:

```text
frozen B0
→ score outer TRAIN+CAL
→ build META rows
```

Но gate, utility scorer, calibration и thresholds всё равно обучаются только внутри outer TRAIN+CAL.

---

# 19. Dataset-trained upstream

Если ranker/question selection/calibration зависели от данных:

обязательно inner cross-fitting.

Запрещено:

```text
train upstream on doc
→ get feature on same doc
→ train meta-model
```

---

# 20. Primary residual target = downstream Token-F1 utility

Для B0 top1:

```text
C0
A0 = QA(Q,C0)
```

Для каждой **альтернативы**:

```text
Ci, i > 0
Ai = QA(Q,Ci)
```

Определить:

\[
u_i = F1(A_i,G)-F1(A_0,G)
\]

Все значения F1 здесь выражаются в исторической шкале проекта:

```text
0...100 F1 points
```

Использовать **точно ту же historical Token-F1 implementation**, что и в соответствующей cell, включая:

```text
normalization
multiple-gold aggregation
empty/no-answer handling
tokenization rules
```

Нельзя создать отдельную V6-версию Token F1 для training target.

Gold используется только для:

```text
training target
model-selection metrics inside allowed TRAIN/CAL
final evaluation
```

Gold никогда не является runtime feature.

Если historical task допускает no-answer:

```text
empty answer
```

обрабатывается exact historical evaluator; такие вопросы не исключать ad hoc.

---

# 21. Primary utility target

Primary:

```text
epsilon = 0.0 F1 point
```

\[
y_i = 1[u_i>\epsilon]
\]

То есть:

> улучшит ли переход к `Ci` downstream answer относительно B0 top1?

При:

```text
u_i = 0
```

target = 0.

Дополнительный diagnostic, не используемый для primary training:

```text
epsilon = 1.0 F1 point
```

Нельзя после EVAL заменить primary epsilon=0 на epsilon=1.

---

# 22. Question-, lineage- и corpus-family weighting

## 22.1. Question-level weighting

Один вопрос может иметь разное число alternatives.

Для q с `m_q` alternative rows:

\[
w^{question}_{q,i}=\frac{1}{m_q}
\]

Total weight каждого question = 1.

При `m_q=0` training rows нет.

## 22.2. Secondary magnitude weighting

Для secondary utility-weighted variant:

\[
r_{q,i}=clip(|u_i|,1,50)
\]

\[
w^{mag}_{q,i}
=
\frac{r_{q,i}}{\sum_j r_{q,j}}
\]

Total question weight остаётся 1.

## 22.3. SciER lineage weighting

SciER 384/50 и 128/24 могут содержать две representations одного `(canonical_document, aspect/question)`.

В POOLED нельзя давать такому semantic question двойной вес только из-за двух chunking schemes.

Создать:

```text
question_lineage_id = hash(
    corpus_family,
    canonical_document_id,
    canonical_aspect,
    canonical_question_semantics
)
```

Если один lineage представлен в обеих SciER cells:

```text
total lineage weight = 1
```

и он делится между cells/rows.

## 22.4. POOLED corpus-family weighting

После question/lineage normalization:

```text
total training weight each corpus_family = equal
```

Families:

```text
SciREX
SciER
RuSERRC
SciMDIX
```

SciER две cells вместе = одна family.

## 22.5. Seeds

Seed replicas никогда не участвуют в одном fit, поэтому им не требуется training weighting между собой.

Сохранить:

```text
analysis/training_weight_audit.json
```

Проверки:

```text
question total weights
lineage total weights
family total weights
```

---

# 23. S1 evidence-target control

S1 должен максимально чисто проверять именно structural diagnosis V5.

Поэтому S1 использует:

```text
exact V5 sufficient-positive / insufficient-evidence contract
exact V5 mirrored pair construction
exact V5 question-level weighting
exact V5 switch threshold semantics
```

Единственное содержательное изменение S1:

```text
добавлены заранее зафиксированные interaction features
```

Additive context может снова иметь близкие к нулю коэффициенты — это нормально.

Главный тест:

> становятся ли ненулевыми и полезными conditional slopes, когда context входит через interaction с candidate differences?

Не менять одновременно objective и pair construction в S1.

Если exact V5 pair contract не удаётся восстановить:

```text
S1_BLOCKED_REPLAY
```

Не заменять его «похожим» evidence target.

---

# 24. Canonical feature orientation и scale-invariant normalization

Единый контракт:

```text
higher score = better candidate
```

Если historical score имеет обратное направление — сначала orient.

## 24.1. B0

Для shortlist:

```text
b0_raw
b0_oriented
b0_z
b0_rank
```

Robust scale:

```text
scale = max(std(oriented_scores), 1e-6)
```

\[
z_i=\frac{s_i-\mu_s}{scale}
\]

Primary:

\[
\Delta B0_i=z_i-z_0
\]

## 24.2. Q-A

Аналогично:

\[
\Delta QArel_i=z(QArel_i)-z(QArel_0)
\]

## 24.3. Optional signals

LEX/SEM/QAConf/source continuous features:

```text
orient if needed
normalize within query or by effective K
compute difference to C0
```

Count features:

```text
count / effective_k
```

если это не нарушает exact historical semantics исходного signal.

## 24.4. Почему raw scores не primary

MiniLM и DiTy имеют разные score scales.

Поэтому POOLED primary model не получает raw cross-encoder logits.

Raw values сохраняются только для diagnostics.

## 24.5. Near-constant shortlist

Если `std < 1e-6`:

```text
all z = 0
normalized gaps = 0
flat_score_vector = 1
```

иначе:

```text
flat_score_vector = 0
```

Этот flag runtime-computable и допустим.

---

# 25. Q-A relevance

\[
QArel_i = Ranker(Q,A_i)
\]

Использовать тот же exact upstream ranker state, который дал B0 scores данной строки.

```text
inner-validation → same inner upstream state
outer EVAL       → same outer upstream state
fully frozen     → same frozen checkpoint
```

Нельзя:

```text
B0(C) от одной fold-model
Q-A(A) от другой fold-model
```

## Empty predicted answer

Если QA вернул пустую/invalid строку:

```text
answer_valid_i = 0
```

и применить заранее фиксированный missing policy:

```text
QArel_raw = minimum finite score within query
```

после чего выполнить обычную within-query normalization.

Дополнительно включить:

```text
answer_valid_i
answer_valid_0
```

как binary runtime features.

Не подавать empty string молча без validity flag.

---

# 26. Q-A feature contract

Сохранить:

```text
raw score
oriented score
within-query z-score
within-query rank
answer_valid
```

Primary:

```text
ΔQ-A z-score
Δanswer_valid
```

Raw/rank — diagnostics.

Не выбирать transformation по outer-EVAL качеству.

---

# 27. Historical source features

Если exact historical implementation доступна:

```text
source_cluster_count
exact_answer_group_size
source_support
```

Для candidate-level use построить exact historical signal, затем нормализовать внутри shortlist / на effective K по frozen contract.

S4 должен использовать source **не только как additive feature**, но и через заранее зафиксированные interactions, потому что V4.5 показал conditional source utility.

Required S4 source interactions:

```text
ΔSource × effective_k
ΔSource × b0_margin_12_norm
ΔSource × aspect
```

Если source implementation не существует в cell:

```text
source_available = 0
normalized source features = 0
```

Final universal candidate с source допустим только при deterministic missing-signal fallback, одинаковом для всех cells.

---

# 28. Lexical signal

Использовать exact historical LEX implementation.

Не создавать новую lexical metric.

Primary S4 lexical candidate feature:

```text
ΔLEX_norm
```

Required minimal interactions:

```text
ΔLEX × effective_k
ΔLEX × b0_margin_12_norm
```

Если LEX unavailable:

```text
lex_available = 0
ΔLEX = 0
```

Нельзя подбирать отдельную lexical formula по dataset.

---

# 29. Semantic signal

Использовать только exact historical encoder + preprocessing/pooling/truncation semantics.

Если unavailable:

```text
sem_available = 0
semantic feature = 0
```

Новый encoder запрещён.

S5 — secondary extension.

Если semantic signal входит в final candidate, missing-policy должен быть deterministic и зафиксирован до EVAL.

---

# 30. QA confidence

Разрешено только после exact QA replay audit.

Если historical decoding/logits не воспроизводятся:

```text
qa_conf_available = 0
```

Не менять QA decoder ради feature.

Если доступно:

```text
store raw confidence
orient higher=better
normalize within query
use ΔQAConf
```

QA confidence входит только в S5.

---

# 31. Availability masks и eligibility universal candidate

Optional masks:

```text
source_available
lex_available
sem_available
qa_conf_available
```

## 31.1. Dataset-proxy audit

В POOLED masks могут стать скрытым dataset identifier.

Обязательно report:

```text
availability pattern by corpus family
mutual information(mask pattern, corpus family) diagnostic
```

и выполнить:

```text
POOLED with masks
POOLED without masks
```

как diagnostic.

## 31.2. Final-candidate eligibility

Для **final universal candidate** предпочтительно, чтобы каждый используемый signal был runtime-computable под одной definition на всех five cells.

Если S4/S5 signal технически отсутствует на части cells:

```text
variant remains diagnostic
```

и не становится final universal candidate, **если только** frozen method заранее не определяет language/corpus-independent missing-value mechanism, который проходит dataset-proxy audit.

S5 с QAConf/SEM не должен автоматически становиться final winner из-за результата только на subset datasets.

---

# 32. Explicit interactions — ключевое изменение V6

## S1/S2/S3 CORE interactions

Заранее фиксировать:

```text
ΔQArel × effective_k
ΔQArel × b0_margin_12_norm
ΔB0    × b0_margin_12_norm
ΔQArel × b0_entropy_norm
ΔB0    × ΔQArel
```

Последний interaction разрешён заранее, потому что он проверяет согласие/конфликт base relevance и predicted-answer relevance.

## S4 source/LEX interactions

```text
ΔSource × effective_k
ΔSource × b0_margin_12_norm
ΔSource × aspect

ΔLEX × effective_k
ΔLEX × b0_margin_12_norm
```

## S5 optional

Не добавлять новый широкий interaction search.

Разрешить только additive:

```text
ΔSEM
ΔQAConf
```

если заранее не было иной historical hypothesis.

Не использовать automatic all-pairs expansion.

---

# 33. Aspect interactions и canonical aspect contract

Aspect известен runtime и может быть legitimate conditioning variable, но raw dataset-local label может кодировать corpus identity.

## 33.1. Canonical aspect vocabulary

До EVAL создать:

```text
audit/canonical_aspect_mapping.json
```

Правило:
- одинаковые понятия на разных языках получают один canonical id;
- разные понятия **не объединять ради увеличения N**;
- `result` и `conclusion` не считать автоматически эквивалентными;
- неизвестные/неоднозначные mappings остаются отдельными ids.

Examples:

```text
method → METHOD
task → TASK
material → MATERIAL
tool → TOOL
goal/aim → GOAL
use/usage → USE
metric → METRIC
result → RESULT
conclusion → CONCLUSION
contribution → CONTRIBUTION
advantage → ADVANTAGE
example → EXAMPLE
```

Использовать mapping только если он подтверждён historical annotation semantics.

## 33.2. LOCAL

Разрешены:

```text
ΔQArel × canonical_aspect
ΔSource × canonical_aspect
```

## 33.3. POOLED

Primary POOLED candidate использует canonical aspect interaction только если mapping audit PASS.

Обязательный universality diagnostic:

```text
POOLED with canonical aspect
POOLED without aspect
```

Нельзя использовать dataset-local aspect string напрямую.

## 33.4. Rare categories

```text
min_train_questions_per_aspect = 20
```

Rare → `OTHER` только по TRAIN scope.

OneHotEncoder:

```text
handle_unknown = ignore
```

Отдельные models/thresholds per aspect запрещены.

---

# 34. Scale-free shortlist uncertainty features

Пусть oriented B0 scores:

```text
s1 >= s2 >= ... >= sk
```

Использовать scale из раздела 24.

```text
gap12_norm = (s1-s2)/scale
gap13_norm = (s1-s3)/scale
top1_z      = (s1-mean(scores))/scale
```

Entropy:
1. z-standardize shortlist scores;
2. softmax with fixed temperature=1;
3. normalized entropy in [0,1].

Primary context:

```text
effective_k
candidate_count
gap12_norm
gap13_norm
top1_z
b0_entropy_norm
flat_score_vector
has_rank3
```

Raw top1 score не использовать в POOLED primary model.

## K=1

```text
no alternative
GateTarget=0
route=false
no utility rows
```

## K=2

```text
gap13_norm=0
has_rank3=0
```

## K>=3

```text
has_rank3=1
```

---

# 35. Запрет automatic feature search

Не использовать:

```text
PolynomialFeatures(all pairs)
automatic interaction discovery
feature importance → add next feature
```

в рамках V6.

Разрешены только interactions, перечисленные в frozen contract.

Все новые идеи после просмотра outer EVAL относятся к:

```text
V6.1
```

---

# 36. Primary utility scorer

Pipeline:

```text
continuous:
    StandardScaler

categorical:
    OneHotEncoder(handle_unknown="ignore")

classifier:
    LogisticRegression
```

Fixed:

```text
penalty = "l2"
solver = "liblinear"
max_iter = 3000
class_weight = None
random_state = seed
```

`class_weight=None` выбран намеренно:

- primary utility model должен сохранять интерпретируемый probability ordering;
- imbalance компенсируется threshold/calibration;
- question/family weights уже задаются явно.

S1 mirrored evidence control также использует:

```text
class_weight=None
```

потому что mirrored construction уже симметризует классы.

---

# 37. Regularization selection — только META_PRED_OOF

Разрешён grid:

```text
C ∈ {0.1, 1.0, 10.0}
```

Для каждого C:

1. выполнить document-grouped meta-model cross-fitting;
2. получить `META_PRED_OOF`;
3. применить candidate selection/switch simulation;
4. рассчитать Token F1 и evidence safety только на этих OOF predictions.

Выбрать C:

```text
max OOF Token F1
subject to:
ΔHit1 >= -0.5 pp
ΔHN1 <= +1.0 pp
```

Tie-break:

```text
smaller C
```

то есть более сильная regularization.

Outer EVAL не участвует.

---

# 38. Utility probability calibration и терминология

Primary S2 не оценивает математическое `E[u]` напрямую.

Он оценивает:

```text
P(u_i > 0 | runtime features)
```

Поэтому в коде и отчёте primary score называть:

```text
positive_utility_probability
```

а не `expected_utility`.

Calibration:

```text
Platt scaling only
```

Calibrator fit только на `META_PRED_OOF` raw decision scores.

После выбора C:
1. получить OOF decision scores;
2. fit Platt calibrator on OOF scores/targets;
3. choose tau_switch on calibrated OOF predictions;
4. refit utility classifier on all allowed TRAIN+CAL;
5. apply frozen calibrator to final-model decision scores.

Сохранить:

```text
base model
preprocessor
calibrator
C
tau_switch
```

Если позже понадобится прямое моделирование `E[u]`, это отдельная preregistered версия/эксперимент, а не скрытая замена S2.

---

# 39. Switch threshold

Для candidate `i>0`:

```text
p_i = calibrated P(u_i>0)
best_i = argmax p_i
```

Grid:

```text
tau_switch ∈ {
  0.50,
  0.55,
  0.60,
  0.65,
  0.70,
  0.75,
  0.80
}
```

Выбрать только по `META_PRED_OOF`:

```text
max Token F1
```

при:

```text
ΔHit1 >= -0.5 pp
ΔHN1 <= +1.0 pp
```

Tie-break:

1. выше threshold;
2. меньше switch rate.

S1 structural control не использует новый tuned threshold:

```text
tau_switch = exact V5 threshold semantics
```

чтобы S1-S0 изолировал эффект interactions.

---

# 40. Degenerate/tie safety

При:

```text
p_best == tau_switch
numerical tie
NaN/Inf
all alternatives invalid
```

оставить B0.

Если utility target внутри allowed TRAIN scope имеет только один класс:

```text
utility_model_status = degenerate
switch = false
```

Не обучать фиктивный classifier.

Если Platt calibration невозможна из-за single-class OOF target:

```text
calibration_status = unavailable
use uncalibrated logistic probability
```

но threshold всё равно выбирается только на allowed OOF predictions.

Все fallbacks записать в artifacts.

---

# 41. Benefit gate target

Gate работает на уровне вопроса.

\[
GateTarget(q)
=
1\left[
\max_{i>0} u_i > 0
\right]
\]

То есть:

> есть ли среди alternatives кандидат, способный улучшить downstream Token F1 относительно B0 top1?

Если:

```text
effective_k <= 1
```

то:

```text
GateTarget = 0
route = false
```

Gold используется только для gate training target внутри allowed TRAIN/CAL.

---

# 42. Gate runtime features

Gate работает ДО top-K QA.

Primary scale-reduced features:

```text
effective_k
candidate_count
gap12_norm
gap13_norm
has_rank3
top1_z
b0_entropy_norm
aspect
language
```

Запрещены:

```text
predicted answer
QA confidence
Q-A relevance
source/group from predicted answers
gold
utility target
```

Raw ranker score не использовать в POOLED primary gate.

---

# 43. Gate model и weighting

```text
StandardScaler
+
OneHotEncoder(handle_unknown="ignore")
+
LogisticRegression
```

Fixed:

```text
C = 1
penalty = "l2"
solver = "liblinear"
class_weight = "balanced"
max_iter = 3000
random_state = seed
```

Для POOLED/LOCO-meta дополнительно использовать equal corpus-family sample weighting.

Если gate training target single-class:

```text
gate_status = degenerate
route = false
```

Это safe fallback.

---

# 44. Gate calibration

Calibration:

```text
Platt scaling only
```

Fit на document-grouped gate `META_PRED_OOF`.

Не calibrate на predictions строк, на которых gate classifier обучался.

Если calibration невозможна из-за single-class OOF:

```text
use native gate probability
calibration_status = unavailable
```

Сохранить calibration reliability diagnostics:

```text
Brier score
ECE
reliability bins
```

только как secondary diagnostics.

---

# 45. Gate threshold: максимизация качества под фиксированным compute budget

Это критическое исправление относительно V5-style routing.

Primary constraint:

\[
E[QA\ forwards/question] \le 2.0
\]

Но среди thresholds, удовлетворяющих бюджету, нельзя автоматически выбирать максимальную routing coverage: это может пропускать много false-positive routes и снижать F1.

На gate `META_PRED_OOF` для каждого threshold:

1. выполнить **полную simulated S3 decision**:
   ```text
   gate decision
   → if routed, use OOF utility-switch prediction
   → else keep B0
   ```
2. вычислить:
   ```text
   downstream Token F1
   Hit@1
   HN@1
   mean QA forwards
   routing rate
   ```
3. оставить thresholds, где:
   ```text
   mean QA forwards <= 2.0
   ΔHit1 >= -0.5 pp
   ΔHN1 <= +1.0 pp
   ```
4. выбрать threshold с maximum OOF Token F1.

Tie-break:
1. меньше mean QA forwards;
2. выше threshold;
3. меньше routing rate.

Если ни один nontrivial threshold не проходит safety/budget:

```text
tau_gate = +inf
route=false
```

Outer EVAL:

```text
tau_gate frozen
```

Нельзя дотягивать фактический outer budget новым threshold.

Secondary preregistered operating points:

```text
budget targets = 1.25, 1.50, 2.00, FULL
```

Для 1.25/1.50/2.00 threshold также выбирается по **OOF quality under corresponding budget**, не по coverage alone.

---

# 46. Реальная cost accounting

Not routed:

```text
1 QA forward
```

Routed:

```text
effective_k QA forwards
+ effective_k Q-A ranker forwards
+ optional source/LEX/SEM/QAConf computation
+ tiny meta-model inference
```

Primary gate budget использует число QA forwards как стабильный hardware-independent proxy.

Но final resource conclusion должен учитывать:

```text
actual end-to-end latency
Q-A ranker forwards
semantic forwards
VRAM
```

Нельзя объявлять систему «дешевле» только потому, что QA-forward budget <=2, если Q-A/SEM overhead фактически уничтожает latency advantage.

---

# 47. Preregistered systems и attribution ablations

Все варианты фиксируются до первого полного outer-EVAL.

```text
S0 = canonical B0

S1 = V5-interaction control
     exact V5 evidence supervision
     exact V5 mirrored pair contract
     CORE interactions
     no gate
     exact V5 threshold semantics

S2 = utility interaction
     Ci vs B0 top1
     Token-F1 positive-utility target
     CORE interactions
     calibrated safe switch
     no gate

S3 = UAR-G CORE
     S2 + benefit gate

S4 = UAR-G multisignal
     S3 + source + LEX
     + predeclared source/LEX interactions

S5 = UAR-G full secondary
     S4 + optional historical SEM + QA confidence
```

## 47.1. Attribution-only ablations

Чтобы S4 improvement можно было интерпретировать, обязательно запустить как **diagnostic ablations, не отдельные final candidates**:

```text
S4-src = S3 + source interactions only
S4-lex = S3 + LEX interactions only
```

Для S5, если оба сигнала доступны:

```text
S5-sem    = S4 + SEM only
S5-qaconf = S4 + QAConf only
```

Они нужны для attribution и не расширяют winner search.

## 47.2. Universal eligibility

S4/S5 eligible как final universal candidate только если их signal contract выполняет требования раздела 31 на всех cells или имеет заранее валидированный universal missing mechanism.

---

# 48. Что проверяет каждый contrast

```text
S1 vs historical V5-A1:
устраняют ли explicit interactions structural failure V5
при неизменном evidence objective?

S2 vs S1:
даёт ли direct Token-F1 utility target дополнительный эффект?

S3 vs S2:
может ли benefit gate сохранить useful residual gain
при меньшем compute?

S4 vs S3:
становятся ли source/LEX переносимыми,
если их полезность задаётся условно через interactions?

S5 vs S4:
оправдывают ли SEM/QAConf дополнительную стоимость?
```

В final report обязательно иметь прямую строку:

```text
historical V5 A1
```

рядом с S1, иначе interaction hypothesis не проверена чисто.

---

# 49. Development model selection и запрет cherry-picking

Final development candidate — одна пара:

```text
(system_variant, parameterization_mode)
```

```text
system_variant ∈ {S2,S3,S4,S5}
parameterization_mode ∈ {LOCAL,POOLED}
```

S1 и attribution ablations не eligible final candidates.

LOCO-meta — transfer diagnostic.

## 49.1. Outer CV = development selection

S0-S5 outer-CV используется для development selection.

Поэтому score выбранного winner:

```text
DEVELOPMENT estimate
```

а не unbiased confirmation.

Независимое утверждение требует untouched V6-B.

## 49.2. Selection hierarchy

1. Сначала проверить eligible POOLED candidates.
2. Если хотя бы один POOLED проходит universal development criterion, выбирать среди них по section 50.
3. Если POOLED не проходит, проверить eligible LOCAL candidates.
4. LOCAL success поддерживает architectural universality, но не shared-parameter universality.

Запрещено:

```text
POOLED EN + LOCAL RU
```

как одна final architecture.

## 49.3. POOLED weights/threshold interpretation

В каждом `(global outer fold, seed)`:

```text
one pooled fitted model
one tau_switch
one tau_gate
```

для всех held-out cells.

Across folds numerical parameters naturally differ because CV retrains model.

Это не считается dataset-specific tuning.

---

# 50. Global development summary и family weighting

SciER не получает двойной вес.

\[
\Delta_{SciER}
=
\frac{
\Delta_{384}+\Delta_{128}
}{2}
\]

\[
GlobalDelta
=
mean(
\Delta_{SciREX},
\Delta_{SciER},
\Delta_{RuSERRC},
\Delta_{SciMDIX}
)
\]

Это **secondary development summary**, а не pooled statistical effect.

Если несколько eligible candidates проходят все primary conditions:

1. предпочесть POOLED mode;
2. затем higher `GlobalDelta`;
3. если различие < 0.2 F1 global — выбрать меньшую resource cost;
4. если cost также близка — выбрать более простой feature set.

Это правило фиксируется до EVAL.

---

# 51. Primary development go-to-freeze criterion

Candidate eligible для freeze как universal-development candidate, если одновременно:

```text
point ΔTokenF1 > 0
```

на всех пяти protocol cells,

и:

```text
LCB95(ΔF1) > -0.5
```

на каждой cell,

и выполняется evidence safety.

Это не доказательство superiority.

Это development gate для перехода к untouched confirmation.

---

# 52. Evidence safety

На каждой cell:

```text
ΔHit@1 >= -0.5 percentage points
ΔHN@1 <= +1.0 percentage point
```

Оценивать относительно **канонического baseline той же protocol cell**, а не относительно historical system с другой question/chunking semantics.

Если HN label отсутствует в corpus:

```text
HN@1 = not_applicable
```

и использовать доступные evidence transitions вместо искусственного HN metric.

---

# 53. Corpus dominance — diagnostic guard

Рассчитать долю каждого corpus family в суммарном positive gain.

Если один corpus family создаёт >50% суммарного gain:

```text
dominance_warning = true
```

Это **diagnostic warning**, а не автоматический scientific failure.

Причина:

- datasets имеют разный headroom;
- одинаковый effect size не обязан существовать.

Но при dominance warning нельзя описывать global gain как «равномерное улучшение».

Главное universal condition остаётся:

```text
point-positive per cell + safety
```

---

# 54. Resource criterion

Для gated final candidate primary engineering target:

```text
TRAIN/CAL-selected expected QA forwards <= 2.0
```

На outer EVAL report actual:

```text
QA forwards
Q-A forwards
latency
P95
throughput
VRAM
RSS
```

Scientific quality success и engineering target report отдельно.

Если quality-positive system не выполняет engineering target:

```text
quality_success = true
resource_success = false
```

Не скрывать такой trade-off.

---

# 55. LOCAL / POOLED / LOCO-meta

## LOCAL

Одна architecture/training contract, но meta-layer fit отдельно на TRAIN каждой cell.

Проверяет:

```text
architectural universality
```

LOCAL требует labeled local TRAIN/CAL для нового corpus.

## POOLED

Для каждого `(global fold, seed)` из раздела 16:

```text
ONE shared gate
ONE shared utility model
ONE shared calibration
ONE shared tau_switch
ONE shared tau_gate
```

fit на pooled TRAIN with family-balanced weights.

POOLED не получает:

```text
dataset_id
corpus_family_id
file path/source
historical run id
```

Разрешены только runtime-legitimate features:

```text
language
canonical aspect if audit PASS
K
normalized uncertainty
candidate signals
availability masks if proxy audit PASS
```

Это основной тест shared-parameter universality.

## LOCO-meta

Для family H:

```text
TRAIN meta-layer = all allowed rows families != H
EVAL meta-layer  = family H
```

SciER обе cells уходят вместе.

Hyperparameters/calibration/thresholds выбираются только на TRAIN families.

Held-out family labels не используются до evaluation.

Historical upstream B0/QA для held-out family могут оставаться corpus-specific.

Поэтому правильное название:

```text
LOCO-meta transfer
```

а не full zero-shot pipeline.

---

# 56. Universality interpretation

Различать:

## Architectural universality

LOCAL success:

```text
same algorithm
same features/interactions
same objectives
same fitting/threshold rules
```

но different fitted local coefficients.

## Shared-parameter universality

POOLED success:

```text
shared fitted meta-weights and thresholds
within each CV fold / final fit
```

без dataset id.

## Transfer evidence

LOCO-meta success:

> meta-layer переносится на held-out corpus family без fitting на её labels.

Но LOCO-meta остаётся development evidence, поскольку сама hypothesis/feature engineering создана после просмотра этих corpora.

## Independent universality

Только untouched V6-B позволяет делать confirmatory claim.

---

# 57. Primary model complexity

Не использовать как primary:
```text
MLP
Transformer
new cross-encoder
new LLM
```

---

# 58. Secondary nonlinear control

Разрешён **один** lightweight nonlinear diagnostic:

```text
HistGradientBoostingClassifier
```

или уже установленный в проекте:

```text
XGBoost / LightGBM
```

но только один family, выбранный до EVAL.

Ограничения:

```text
max_depth <= 3
n_estimators/max_iter <= 100
```

Hyperparameters, кроме технически обязательных, не искать широким grid.

Назначение:

> проверить, достаточно ли explicit linear interactions или decision boundary существенно нелинеен.

Критически:

```text
nonlinear control = DIAGNOSTIC ONLY
not eligible for V6 final winner
```

Если он существенно превосходит predeclared linear UAR-G:

```text
result → hypothesis for V6.1
```

Не заменять им primary model post-hoc.

---

# 59. BGE strong reference

На RU сохранить BGE как strong quality reference.

Optional mechanism control:
```text
DiTy gate → BGE on routed cases
```

Это diagnostic, не primary UAR architecture.

---

# 60. Primary metric

```text
Token F1
```

---

# 61. Secondary answer metrics

Использовать historical implementations:

```text
Character F1
Lemma F1 RU
ROUGE-L
Unicode ROUGE-L RU
Exact Match
BERTScore if historically defined
```

---

# 62. Evidence metrics

```text
Hit@1
Hit@K
HN@1
HN@K
F1 | selected positive
Amb@1 if applicable
GSRN@1 if applicable
```

---

# 63. Switch metrics

```text
good
bad
neutral

positive→positive
positive→non-positive
non-positive→positive
non-positive→non-positive
```

---

# 64. Evidence ranking metric contract

V6 primary output — выбранный final evidence, а не обязательно полный reranked list.

Поэтому стандартный `MRR` нельзя вычислять молча так, будто модель построила полный ranking.

Определить diagnostic metric:

```text
MRR_selected_first
```

Constructed ranking:
1. selected candidate → position 1;
2. remaining candidates preserve original B0 relative order.

Report label exactly as:

```text
MRR_selected_first
```

Не называть его просто `MRR` в таблицах без примечания.

Primary evidence metrics остаются:

```text
Hit@1
HN@1
transitions
```

Historical V5 MRR не использовать как V6 reranked metric.

---

# 65. Statistical unit

Seeds не считать независимыми samples.

1. per-question metric per seed;
2. average seeds внутри question;
3. document-cluster bootstrap.

---

# 66. Bootstrap и интервальная оценка

## 66.1. Per-cell paired bootstrap

Для каждой protocol cell:

```text
iterations = 5000
seed = 42
paired = true
cluster = canonical_document_id
```

До bootstrap:
1. получить metric contribution per `(question_id, seed)`;
2. усреднить seeds внутри question;
3. resample canonical documents.

Report:

```text
observed delta
95% development CI
```

## 66.2. SciER dependence

SciER 384/50 и 128/24 не считать независимыми samples.

Для любого joint SciER analysis resample один и тот же `canonical_document_id` lineage согласованно в обеих cells, если документ присутствует в обеих.

## 66.3. GlobalDelta uncertainty — secondary

Если рассчитывается CI для `GlobalDelta`, использовать hierarchical/family-aware bootstrap:

1. resample canonical documents **внутри каждой corpus family**;
2. для SciER сохранить связанность двух cells;
3. вычислить `ΔSciER = mean(Δ384,Δ128)`;
4. затем вычислить equal-family `GlobalDelta`.

Не pooling all questions across corpora, иначе крупный dataset получит скрытый больший вес.

## 66.4. Интерпретация

Все V6-A CI остаются development CI из-за research-program reuse и model selection.

Confirmatory uncertainty оценивается только на untouched V6-B.

---

# 67. Fold stability

Report:
```text
ΔF1 per outer fold
```

---

# 68. Seed stability

Report:
```text
ΔF1 per seed
```

---

# 69. Aspect analysis

Только post-hoc diagnostic.

Не менять architecture внутри V6 по aspect results.

---

# 70. Effective-K analysis

Buckets:
```text
1
2
3
4
5+
```

Report gain, routing, switch accuracy.

---

# 71. Uncertainty analysis

TRAIN-derived quartiles:
```text
B0 margin
entropy
```

Thresholds определять на TRAIN.

---

# 72. Interaction coefficient analysis

Report median/IQR/sign stability для:
```text
ΔQ-A
ΔQ-A × K
ΔQ-A × margin
ΔB0 × margin
ΔQ-A × entropy
aspect interactions
source interactions
```

Не интерпретировать коэффициенты причинно.

---

# 73. Resource benchmark — systems и sampling contract

Benchmark минимум:

```text
B0
S2 full utility
S3 gated CORE
selected final UAR candidate
BGE RU strong reference
```

Если selected candidate = S3, не дублировать его отдельной строкой.

## 73.1. Benchmark sample

Не повторять ограничение V5 с 20 вопросами как единственной resource evidence.

Preferred:

```text
all outer-EVAL questions one frozen fold/seed
```

если стоимость приемлема.

Минимум:

```text
min(100, all available questions)
```

на cell, stratified по:

```text
aspect
effective_k
route/non-route для gated system where known from frozen predictions
```

Sample IDs фиксируются **до timing** и одинаковы для всех сравниваемых systems.

Если cell <100 questions — использовать все.

## 73.2. Fold/seed

Primary technical benchmark:

```text
outer/global fold 0
seed 42
```

Дополнительно, если runtime позволяет, повторить на ещё одном seed или fold как stability check.

Не выбирать самый быстрый fold/seed post-hoc.

---

# 74. End-to-end timing protocol

B0:

```text
rank candidates
QA(top1)
```

S2/full:

```text
rank candidates
QA(top-K)
Q-A ranker
required signals
utility scorer
```

Gated:

```text
rank candidates
gate
if route=false:
    QA(top1)
else:
    QA(top-K)
    Q-A ranker
    required signals
    utility scorer
```

## 74.1. Warm-up

Перед измерением каждой системы:

```text
>=10 warm-up questions/forwards
```

Warm-up не включать в latency statistics.

## 74.2. Repeats

Каждый benchmark sample выполнить:

```text
3 timed repeats
```

при одинаковом порядке или фиксированном seeded shuffle.

Report:

```text
median across all timed requests
P95
per-repeat summary
```

## 74.3. Synchronization

Для GPU latency обязательно:

```python
torch.cuda.synchronize(device)
```

до старта и после окончания timed region.

## 74.4. Scope

Primary latency = warm-model inference.

Cold model-loading time report отдельно, если измеряется.

Не смешивать model load и request latency.

---

# 75. No hidden cache advantage

Quality experiment может использовать V6 cache.

Resource benchmark:

```text
real model inference required
```

Запрещено учитывать как inference:

```text
cached QA lookup
cached Q-A score lookup
cached semantic embedding lookup
```

Если production design действительно предполагает persistent document embeddings, можно дополнительно измерить:

```text
cached-document deployment scenario
```

но только отдельной secondary строкой.

Primary comparison должен быть одинаково честным для всех systems.

---

# 76. Resource metrics

Обязательно:

```text
mean QA forwards/question
mean Q-A ranker forwards/question
mean semantic forwards/question if used
routing rate

median single-query latency
P95 single-query latency
throughput Q/s

peak GPU VRAM
peak CPU RSS

loaded parameter count
weights size
```

По возможности:

```text
FLOPs/MACs
energy/power only if reliable instrumentation already exists
```

Также report относительные коэффициенты:

```text
latency / B0
QA forwards / B0
VRAM / B0
```

Не сводить всё в один искусственный resource score.

---

# 77. Correct VRAM

```python
device = next(model.parameters()).device
torch.cuda.reset_peak_memory_stats(device)
...
torch.cuda.synchronize(device)
peak = torch.cuda.max_memory_allocated(device)
```

---

# 78. Fresh-process benchmark

Каждую system configuration запускать в отдельном fresh subprocess для:

```text
peak CPU RSS
peak GPU VRAM
loaded model memory
```

Для latency также предпочтителен fresh process + warm-up, после чего timing выполняется без перезапуска между каждым вопросом.

Не загружать последовательно несколько больших model families в одном процессе и потом сравнивать process RSS.

---

# 79. Latency и throughput modes

Отдельно измерять:

## Single-query

```text
batch = one user question/document request
```

Это primary latency mode.

## Batched throughput

Использовать один заранее фиксированный batching contract для всех comparable systems.

Report:

```text
batch size
questions/sec
GPU utilization if available
```

Не сравнивать median batch latency одной системы с single-query latency другой.

---

# 80. Environment manifest

Сохранить:
```text
GPU
GPU UUID
driver
CUDA
PyTorch
Transformers
Python
CPU
RAM
OS
precision
batch sizes
```

---

# 81. New artifact structure

```text
artifacts/qa_chunk_selection_v6_universal/
  run_<timestamp>/

    status.json
    COMPLETED.json

    manifest.json
    frozen_contract.yaml

    audit/
      git_before/
      git_after/
      historical_hashes_before.json
      historical_hashes_after.json
      canonical_baselines.json
      candidate_universe.json
      split_hashes.json
      model_manifests/
      question_policy/
      k_manifest.json
      upstream_training_provenance.json
      historical_replay/
      regression_summary.json

    reused_code/
      reused_components.json

    cache/
      qa/
      q_a_ranker/
      lexical/
      semantic/
      source/
      meta_features/

    datasets/
      scirex/
      scier_384_50/
      scier_128_24/
      russerrc/
      scimdix/

    meta_models/
      local/
      pooled/
      loco/

    predictions/
      local/
      pooled/
      loco/

    analysis/
      primary_metrics.json
      paired_bootstrap.json
      evidence_metrics.json
      switch_analysis.json
      interaction_analysis.json
      fold_stability.json
      seed_stability.json
      aspect_analysis.json
      k_analysis.json
      uncertainty_analysis.json
      global_summary.json
      universality_summary.json
      pareto.json

    resource_benchmark/
      raw/
      summary.json
      environment.json

    reports/
      results_v6.md
      reproducibility.md
```

---

# 82. Cache isolation

Все новые caches — только V6 namespace.

Cache key минимум:
```text
dataset
outer fold
inner fold
seed
model hash
question hash
chunk hash
answer hash if applicable
feature version
```

---

# 83. Historical cache reuse

Old cache только READ.

Разрешено использовать value, если exact identity доказана.

Cache miss → писать только в V6 cache.

---

# 84. reused_components.json

Для каждого reused component:

```json
{
  "source": "...",
  "symbol": "...",
  "historical_version": "...",
  "wrapper": "...",
  "modified": false,
  "regression_test": "PASS"
}
```

---

# 85. Development execution stages

Все variants/ablations описываются и замораживаются до полного outer EVAL.

Этапы — порядок выполнения, **не** разрешение менять следующий шаг по результату предыдущего.

## A — Audit

```text
history/hash/baselines/splits/models/questions
candidate lineage
SciER embargo
```

## B — Historical replay

```text
B0
V5 A1
```

## C — S1

V5 objective + interactions.

## D — S2

Positive downstream utility probability.

## E — S3

Benefit gate with quality-under-budget threshold selection.

## F — S4/S5 + attribution diagnostics

```text
S4-src
S4-lex
S4
S5-sem
S5-qaconf
S5
```

только если signal availability позволяет.

## G — Universality

```text
LOCAL
POOLED global-fold
LOCO-meta
```

Нельзя посмотреть S1 full outer result и добавить новый interaction в S2 того же V6.

---

# 86. Predevelopment contract

До первого полного outer-EVAL создать:

```text
predevelopment_contract.yaml
```

Включить:

```text
S0-S5 exact definitions
feature normalization
interaction schema
question/family weights
meta-level OOF scheme
C grid
calibration
switch threshold grid
gate target
gate features
gate budget thresholds
LOCAL/POOLED/LOCO-meta rules
cross-cell embargo
selection hierarchy
success/safety/resource criteria
```

После hash freeze этого файла scientific logic не меняется.

Smoke/debug может исправлять только implementation bugs.

---

# 87. Frozen contract

До full outer-EVAL создать:

```text
frozen_contract.yaml
```

и SHA-256.

Минимум:

```text
datasets/cells/families
historical upstream identities
K
question policies
candidate universes
document lineage rules

feature orientation
within-query normalization
missing-value policies
answer_valid rules
aspect handling

S0-S5 definitions
interaction schema
utility epsilon
question weights
family weights
C grid
calibration family
tau_switch grid
tau_gate budget rule

meta-feature OOF rules
meta-prediction OOF rules
cross-cell embargo

metrics
safety margins
bootstrap
resource benchmark

LOCAL/POOLED/LOCO-meta protocol
development selection hierarchy
go/no-go rules
```

Outer-EVAL result не должен быть известен до hash freeze.

---

# 88. Нельзя менять после outer EVAL

Запрещено менять:
```text
features
interactions
K
loss
epsilon
weights
C grid
tau grid
gate features
gate target
budget
metrics
safety margins
```

Новая идея → V6.1.

---

# 89. Bug rule

Если найден implementation bug:
1. документировать;
2. старый run → INVALID;
3. не удалять;
4. создать новый timestamped run;
5. перезапустить все affected cells;
6. сохранить `bugfix_audit.md`.

---

# 90. Не перетирать даже ошибочный V6 run

Run directory immutable после завершения.

Никакого overwrite.

---

# 91. Status lifecycle

```text
CREATED
AUDIT_RUNNING
AUDIT_FAILED
AUDIT_PASSED
RUNNING
FAILED
COMPLETED
INVALID
SUPERSEDED
FROZEN
```

---

# 92. Atomic writes

Использовать:
```text
tmp → fsync → atomic rename
```

---

# 93. No overwrite default

```text
overwrite=False
```

Existing output path → exception.

---

# 94. Prediction storage

Для каждого `question × seed × system` сохранить:

```text
dataset_cell
corpus_family
canonical_document_id
document_sha256
question_lineage_id

question_id
canonical_aspect
language

outer/global_fold
upstream_inner_fold
meta_fold
seed

B0 top-K
B0 raw/oriented/normalized scores

uncertainty features
flat_score_vector

raw gate score
gate calibrated probability
tau_gate
gate decision

QA predicted answers
answer_valid
QA confidence if available

Q-A raw/oriented/normalized scores

candidate base features
explicit interactions
availability masks

raw utility score
positive_utility_probability
tau_switch
selected candidate

question weight
lineage weight
family weight

final answer
```

Gold-dependent fields только в:

```text
evaluation
```

namespace.

---

# 95. Runtime / evaluation namespaces

```json
{
  "runtime": {},
  "evaluation": {}
}
```

В runtime нет gold.

---

# 96. Gold leakage test

Проверить отсутствие:
```text
gold answer
gold span
gold F1
positive flag
negative label
oracle rank
utility target
```

в runtime feature graph.

---

# 97. Gate leakage test

Gate features обязаны быть pre-QA only.

---

# 98. Meta provenance

Для каждого meta-training/prediction row сохранить:

```text
document_id
canonical_document_id
document_sha256

dataset_cell
corpus_family

outer target cell
outer fold
upstream inner fold
meta-model fold

upstream model hash
upstream training document-set hash

feature producer version
target producer version
normalization version

self_excluded_from_upstream_training = true/false
self_excluded_from_meta_training = true/false

cross_cell_embargo_pass = true/false
```

Любая outer-EVAL row должна иметь:

```text
self_excluded_from_meta_training = true
```

---

# 99. Historical write guard

Реализовать forbidden prefixes.

Любая попытка write → exception.

---

# 100. Destructive filesystem calls

Запретить destructive calls на historical prefixes.

---

# 101. Smoke test

```text
1 cell
1 outer fold
1 seed
small subset
```

Только code validation.

Не использовать для scientific selection.

---

# 102. Regression tests

Минимум:

```text
test_historical_paths_read_only
test_candidate_universe_match
test_split_match
test_question_policy_replay
test_b0_replay
test_v5_a1_replay
test_qa_replay

test_q_a_same_ranker_state
test_score_direction_contract
test_within_query_normalization
test_flat_score_policy
test_k1_fast_path
test_k2_gap13_policy
test_empty_answer_policy

test_gold_not_in_runtime_features
test_gate_pre_qa_only

test_seed_isolation
test_upstream_self_exclusion
test_meta_prediction_oof
test_threshold_uses_oof_only

test_scier_family_dependency
test_scier_cross_cell_embargo
test_cross_corpus_overlap_audit

test_question_weight_sums
test_scier_lineage_weight_sums
test_pooled_family_weight_sums

test_pooled_global_fold_one_model
test_no_dataset_id_in_pooled
test_pooled_shared_threshold
test_canonical_aspect_mapping
test_aspect_proxy_ablation_configured
test_availability_proxy_audit

test_gate_threshold_quality_under_budget
test_attribution_ablations_not_eligible_winners

test_correct_cuda_device
test_resource_sample_same_across_systems
test_resource_warmup_excluded
test_no_old_artifact_changes
test_no_overwrite
```

---

# 103. CLI audit

```bash
python -m experiments.qa_chunk_selection_v6.audit   --project-root .   --read-only-history   --output artifacts/qa_chunk_selection_v6_universal/<RUN>/audit
```

---

# 104. CLI replay

```bash
python -m experiments.qa_chunk_selection_v6.replay   --config configs/qa_chunk_selection/v6_universal/main.yaml   --datasets scirex scier_384_50 scier_128_24 russerrc scimdix   --strict   --read-only-history
```

---

# 105. CLI run_dev

```bash
python -m experiments.qa_chunk_selection_v6.run_dev   --config configs/qa_chunk_selection/v6_universal/main.yaml   --datasets scirex scier_384_50 scier_128_24 russerrc scimdix   --seeds 42 43 44   --outer-folds 5   --inner-meta-folds 4   --run-dir artifacts/qa_chunk_selection_v6_universal/<RUN>
```

---

# 106. CLI analyze

```bash
python -m experiments.qa_chunk_selection_v6.analyze   --run-dir artifacts/qa_chunk_selection_v6_universal/<RUN>   --bootstrap 5000   --cluster-key document_id   --include-resources   --include-switch-analysis   --include-interactions   --include-universality
```

---

# 107. CLI freeze

```bash
python -m experiments.qa_chunk_selection_v6.freeze   --run-dir artifacts/qa_chunk_selection_v6_universal/<RUN>   --output artifacts/qa_chunk_selection_v6_universal/<RUN>/final_freeze_manifest.json
```

---

# 108. Locked test

```bash
python -m experiments.qa_chunk_selection_v6.run_locked_test   --freeze-manifest artifacts/.../final_freeze_manifest.json   --test-manifest <UNTOUCHED_TEST_MANIFEST>   --output artifacts/.../locked_test
```

Locked-test CLI не принимает tuneable arguments.

---

# 109. Confirmatory rule

После freeze метод оценивается один раз.

Никакого retune после locked test.

---

# 110. Dissertation claim при успешном подтверждении

Поскольку primary S2/S3 classifier оценивает не математическое expected utility, а вероятность положительного изменения F1, формулировка должна быть точной.

Безопасная версия:

> Разработан единый ресурсо-адаптивный метод остаточного ранжирования фрагментов научно-технического документа, оценивающий вероятность положительного изменения качества извлечения при замене решения базового ранжировщика по совокупности ранговых и вопросно-ответных признаков с учётом состояния списка кандидатов и выполняющий дополнительную обработку только при прогнозируемой целесообразности такой коррекции.

Если final S4/S5 действительно подтверждён, можно добавить:

> ...с использованием дополнительных признаков согласованности evidence.

Если подтверждён только LOCAL:

```text
claim = architectural universality
```

Если POOLED:

```text
claim += shared-parameter development evidence
```

Если LOCO-meta также положителен:

```text
claim += transfer evidence for meta-layer
```

Сильное утверждение о генерализации — только после untouched V6-B.

---

# 111. Запрещённые формулировки

Не утверждать:
```text
"первый reader-aware reranker"
"универсальность доказана" на reused development data
"всегда лучше BGE"
"каждый second-stage signal улучшает качество"
```

---

# 112. Weaker defensible claim при mixed result

> Дополнительные evidence-сигналы обладают условной полезностью; предложен и исследован ресурсо-адаптивный механизм безопасного применения residual correction, позволяющий ограничивать вычислительную стоимость и сохранять базовое решение при недостаточной уверенности в улучшении.

---

# 113. Stop rule против research-program overfitting

После заранее frozen S1-S5:

если ни один `(variant, mode)` не удовлетворяет universal development criterion:

```text
V6 universal candidate = NOT SUPPORTED
```

Не добавлять в этот же run:

```text
новые interactions
новые thresholds
новые feature transformations
новые aspect groups
новые models
```

Новая гипотеза:

```text
V6.1
```

Новый versioned protocol и новый contract hash.

Особенно запрещено продолжать feature engineering «до пяти плюсов».

---

# 114. Primary development success

```text
point ΔTokenF1 > 0 на всех five protocol cells
AND
LCB95 > -0.5 F1 на каждой
AND
evidence safety PASS
```

Для gated candidate отдельно report:

```text
resource target PASS/FAIL
```

Corpus dominance — diagnostic warning, не hard failure.

SciER cells считаются двумя protocol checks, но одной corpus family в global weighting.

---

# 115. Strict success

```text
LCB95(ΔF1) > 0
```

на всех cells.

---

# 116. Engineering targets

Predeclared targets:

```text
TRAIN/CAL-selected mean QA forwards <= 2.0
MedianLatency <= 1.25 × B0
P95Latency <= 2.5 × B0
```

VRAM target:

```text
peak VRAM ideally <= 1.10 × resident historical pipeline
```

Это engineering targets.

Они не являются статистическими законами и не должны изменяться post-hoc.

Если S4/S5 требует resident semantic encoder и VRAM >1.10×:

```text
resource target = FAIL/EXCEEDED
```

но quality result всё равно report.

---

# 117. POOLED memorization guards

POOLED запрещено получать:

```text
dataset_id
corpus_family_id
file path
source filename
historical run id
```

Обязательные audits:

```text
canonical aspect distribution by corpus
availability-mask pattern by corpus
language × aspect contingency
feature missingness by corpus
```

Особое внимание:

```text
language
canonical aspect
availability masks
```

могут вместе почти полностью идентифицировать corpus.

Report diagnostic ability to predict corpus family from these context-only features.

Если context-only classifier почти идеально восстанавливает corpus family, отметить:

```text
proxy_risk = high
```

и обязательно показать primary POOLED result без raw aspect/mask context.

Нельзя выбирать context-rich версию только потому, что она лучше на уже виденных corpora.

---

# 118. Language, aspect и deployment assumptions

`language_id` разрешён, потому что language известен runtime, а upstream rankers разные.

`aspect` разрешён как conditioning variable, потому что aspect — часть постановки задачи.

Но:

```text
no separate models
no separate thresholds
no separate feature sets
```

по аспектам.

## Deployment assumption

Если final mode = LOCAL:

новый corpus должен иметь labeled TRAIN/CAL portion для fit meta-layer.

Если final mode = POOLED:

meta-layer можно применять без local refit, если upstream feature contract совместим.

Это различие обязательно описать в dissertation.

---

# 119. Final report tables

Обязательно:

### Quality
```text
S0-S5 × all cells
```

### Universality
```text
LOCAL / POOLED / LOCO-meta
```

### Resource
```text
F1 / QA forwards / latency / P95 / VRAM
```

### Safety
```text
Hit1 / HN1 / transitions
```

---

# 120. Historical reporting cleanup

В final historical table разделять:
```text
raw B0
protocol B0
selected system
diagnostic best
```

Не смешивать.

---

# 121. SciER audit

Явно документировать отличие V4.4 и V5 B0 semantics.

---

# 122. MRR cleanup

Historical V5 MRR не использовать как reranked metric.

Для V6 вычислять только:

```text
MRR_selected_first
```

по exact semantics раздела 64.

В таблицах не подписывать его просто `MRR`, чтобы не создавать впечатление, что UAR-G строит полный learned ranking.

---

# 123. Metrics completion

Если возможно без retraining, рассчитать недостающие:
```text
Lemma F1
Unicode ROUGE-L
EM
Char F1
```

---

# 124. Error analysis

После architecture freeze:
```text
good switches
bad switches
high-confidence bad
gate misses
largest gains
largest losses
```

Не создавать новые ручные rules внутри V6.

---

# 125. Reproducibility report

Создать:
```text
reports/reproducibility.md
```

с:
```text
git commit
environment
commands
hashes
models
data
splits
seeds
all selected parameters
failed/superseded runs
```

---

# 126. Completion marker

`COMPLETED.json` создаётся только если:
```text
all cells done
all primary variants done
regression PASS
history hashes unchanged
analysis done
resources done
```

---

# 127. Failure marker

При сбое:
```text
FAILED.json
```

с traceback/config hash/stage.

---

# 128. Final historical assertion

```text
assert historical_hashes_before == historical_hashes_after
```

Иначе COMPLETED запрещён.

---

# 129. Agent implementation order

```text
1. historical write guard
2. audit/hash system
3. canonical manifest loader
4. baseline replay
5. V6 cache namespace
6. feature extraction
7. utility target builder
8. explicit interaction builder
9. utility scorer
10. benefit gate
11. nested CV orchestrator
12. S0-S5
13. metrics/bootstrap
14. LOCAL/POOLED/LOCO
15. resource benchmark
16. freeze
17. locked-test runner
```

---

# 130. Final scientific questions

V6 должен ответить:

1. Исправляют ли explicit interactions structural failure V5?
2. Лучше ли direct Token-F1 utility supervision, чем evidence-class proxy?
3. Можно ли безопасно запускать expensive second stage только когда ожидается improvement?
4. Дают ли historical V4.4/V4.5 signals дополнительную информацию внутри interaction-aware model?
5. Переносится ли единая architecture между EN/RU corpus families?
6. Есть ли quality-resource Pareto advantage относительно B0 и strong references?

---

# 131. Главное правило интерпретации

Цель:

```text
не получить пять плюсов любой ценой
```

а:

```text
заранее определить единый метод
→ честно оценить
→ сохранить результат как есть
```

Если improvement во всех cells не получится, это не повод продолжать tuning на тех же outer-EVAL.

---

# 132. Краткий итог для code-agent

```text
Implement a NEW V6/UAR-G branch.

V4.4/V4.5/V5 code, artifacts and caches are READ-ONLY.
Never overwrite them.
Hash historical inputs before/after every full run.

First:
- recover canonical baselines
- replay B0 and V5 A1
- audit SciER baseline semantics
- build canonical document lineage
- enforce cross-cell/cross-corpus embargo

Cells:
SciREX
SciER 384/50
SciER 128/24
RuSERRC
SciMDIX

SciER cells = one corpus family.

Use exact historical:
datasets
splits
chunking
questions
K
rankers
QA readers
semantic encoders

No new pretrained LM.

Primary S2 target:
u_i = TokenF1(A_i,gold) - TokenF1(A0,gold)
y_i = 1[u_i > 0]

Primary model estimates:
P(u_i > 0 | runtime features)
NOT mathematical expected utility.

Gold is target/evaluation only.
Never runtime feature.

SEED ISOLATION:
never concatenate seed replicas into one fit.

WEIGHTING:
each question total weight = 1.
Repeated SciER question lineage total weight = 1.
Each corpus family equal total weight in POOLED.

NORMALIZATION:
higher score = better.
Use within-query normalized candidate scores.
Do not mix raw MiniLM and DiTy logits in POOLED.

CORE:
ΔB0_norm
ΔQ-A_norm
effective K
normalized B0 margins
normalized entropy
answer_valid
canonical aspect if audit PASS
language

CORE interactions:
ΔQ-A × K
ΔQ-A × margin
ΔB0 × margin
ΔQ-A × entropy
ΔB0 × ΔQ-A
ΔQ-A × canonical aspect

S1:
exact V5 evidence/mirrored contract
+ interactions only.

S2:
positive Token-F1 utility classifier
+ calibrated safe switch.

S3:
S2 + pre-QA benefit gate.
Gate threshold is selected by MAX OOF F1
SUBJECT TO compute budget and evidence safety.
Do NOT maximize routing coverage alone.

S4:
S3 + source + LEX + their conditional interactions.
Run source-only and LEX-only attribution ablations.

S5:
S4 + optional SEM/QAConf.
Secondary/high-cost.

META LEAKAGE:
C/calibration/tau are selected ONLY from META_PRED_OOF.

Nested:
5 outer folds
4 inner by default
seeds 42,43,44

POOLED:
use GLOBAL fold f = union of fold f from all cells.
Train ONE pooled meta-model/threshold set per global fold and seed.
Hold out all canonical representations of every EVAL document.
This is mandatory for SciER 384 vs 128.

POOLED never receives dataset_id.
Within a pooled fold, thresholds are shared across all cells.

Run:
LOCAL
POOLED
LOCO-meta

Do not choose a different final system per dataset.

Final candidate is one:
(system variant, parameterization mode)

Prefer eligible POOLED if universal criteria pass.
Otherwise eligible LOCAL supports architectural universality only.

Development success:
positive Token-F1 point delta on all five cells
+ LCB95 > -0.5 on every cell
+ evidence safety.

Outer-CV winner estimate is DEVELOPMENT evidence.
Freeze ONE architecture before untouched EN/RU confirmation.

Never overwrite completed V6 runs.
Bugfix => new timestamped run.

A run is valid only if historical hashes are unchanged.
```

---

# 133. Pre-run checklist

```text
[ ] V6 namespace created
[ ] historical write guard implemented
[ ] git status/commit/diff saved
[ ] historical hashes BEFORE saved

[ ] V4.4/V4.5/V5 manifests resolved
[ ] V5 frozen contract checked

[ ] canonical baseline semantics resolved
[ ] SciER B0 semantics resolved
[ ] historical selected vs diagnostic labels corrected

[ ] canonical_document_id created
[ ] document hashes computed
[ ] SciER cross-cell lineage verified
[ ] cross-corpus overlap audit complete
[ ] global POOLED fold construction tested
[ ] cross-cell embargo tested

[ ] candidate universes verified
[ ] splits verified
[ ] questions verified
[ ] K verified
[ ] model hashes verified
[ ] QA replay PASS
[ ] B0 replay PASS
[ ] V5 A1 replay PASS

[ ] seed isolation frozen
[ ] score directions frozen
[ ] within-query normalization frozen
[ ] flat-score policy frozen
[ ] K=1/K=2 policies frozen
[ ] empty-answer policy frozen

[ ] canonical aspect mapping audited
[ ] aspect proxy diagnostic frozen
[ ] availability proxy audit frozen

[ ] upstream provenance classified
[ ] upstream feature OOF policy frozen
[ ] META_PRED_OOF policy frozen
[ ] threshold OOF test PASS

[ ] question weights frozen
[ ] SciER lineage weights frozen
[ ] pooled family weights frozen

[ ] utility epsilon frozen
[ ] interaction schema frozen
[ ] S1 exact V5 contract frozen
[ ] C grid frozen
[ ] calibration family frozen
[ ] tau_switch grid frozen

[ ] gate target frozen
[ ] gate feature schema frozen
[ ] gate budget frozen
[ ] gate quality-under-budget selection frozen
[ ] degenerate fallbacks frozen

[ ] S0-S5 frozen
[ ] attribution ablations frozen as NON-WINNER diagnostics
[ ] universal feature eligibility rules frozen

[ ] LOCAL/POOLED/LOCO-meta rules frozen
[ ] POOLED one-model-per-global-fold rule frozen
[ ] POOLED shared-threshold rule frozen

[ ] bootstrap policy frozen
[ ] MRR_selected_first semantics frozen
[ ] resource benchmark frozen

[ ] unit tests PASS
[ ] smoke PASS

[ ] predevelopment_contract.yaml hashed
[ ] frozen_contract.yaml hashed
[ ] no full outer-EVAL inspected before freeze
```

---

# 134. Post-run checklist

```text
[ ] all 5 protocol cells completed
[ ] all seeds completed independently
[ ] all LOCAL outer folds completed
[ ] all POOLED global folds completed

[ ] historical V5 A1 replay table saved
[ ] S0-S5 predictions saved
[ ] attribution-only predictions saved

[ ] LOCAL results saved
[ ] POOLED results saved
[ ] LOCO-meta results saved

[ ] no SciER cross-cell leakage
[ ] no cross-corpus document leakage
[ ] seed isolation audit PASS
[ ] question-weight audit PASS
[ ] SciER lineage-weight audit PASS
[ ] family-weight audit PASS
[ ] meta-prediction OOF audit PASS
[ ] pooled one-model-per-global-fold audit PASS

[ ] Token F1 computed
[ ] Hit1/HN1 computed where applicable
[ ] MRR_selected_first computed
[ ] switch transitions computed
[ ] fold stability computed
[ ] seed stability computed
[ ] aspect analysis computed
[ ] K analysis computed
[ ] uncertainty analysis computed
[ ] interaction stability computed

[ ] POOLED aspect/no-aspect diagnostic computed
[ ] availability-pattern diagnostic computed
[ ] corpus-proxy risk diagnostic computed

[ ] paired document bootstrap complete
[ ] development selection rule applied exactly once
[ ] selected CV result labelled DEVELOPMENT

[ ] resource benchmark sample >= min(100, all questions) or all-cell mode documented
[ ] identical resource sample used across systems
[ ] warm-up excluded and 3 repeats completed
[ ] resource benchmark complete
[ ] actual CUDA device confirmed
[ ] fresh-process RSS confirmed
[ ] QA and Q-A forwards counted separately

[ ] historical hashes AFTER saved
[ ] BEFORE == AFTER
[ ] git diff saved
[ ] no historical file modified

[ ] results_v6.md created
[ ] reproducibility.md created
[ ] COMPLETED.json created only if all validity checks PASS
```

---

# 135. Итоговая цель

V6 проверяет общий принцип:

\[
\boxed{\text{дополнительный сигнал полезен условно}}
\]

и в primary S2/S3 более точно:

\[
\boxed{\text{нужно оценивать вероятность положительной downstream-полезности конкретного переключения}}
\]

с учётом:

```text
B0 rank
Q-A relation
shortlist uncertainty
K
canonical aspect when legitimate
historical evidence signals
```

Дополнительная обработка выполняется только тогда, когда gate и safe-switch, настроенные исключительно на TRAIN/CAL OOF predictions, предсказывают целесообразность коррекции при допустимой вычислительной стоимости.

UAR-G рассматривается как кандидат на итоговый универсальный диссертационный метод, но статус «подтверждённого универсального метода» возможен только после freeze и независимой V6-B проверки.

---

