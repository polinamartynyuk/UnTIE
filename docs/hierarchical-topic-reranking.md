# Hierarchical topic reranking

The experimental `hierarchical-static-keywords` mode adds automatic document
topic routing before the existing extractive QA reader. It does not use a
generative model and does not require a user-provided topic label.

## Data discipline

- Build centroids, hierarchy, vocabulary and representative terms from train
  documents only.
- Use dev only with `untie.topics.tune_topic_configuration` to select routing
  and aspect/topic mixing parameters.
- Use test once for final evaluation. Never pass test documents to either
  builder or tuner.
- Gold/reference answers are not accepted by the router or inference pipeline.

`build_topic_artifact` rejects records whose `split` is not `train`;
`tune_topic_configuration` rejects records whose `split` is not `dev`.
The builder also rejects a missing `split`; `--assume-train` is an explicit
opt-in for already isolated train-only files.

## Offline build

Input is a JSON array or JSONL. Each record supports:

```json
{
  "doc_id": "paper-001",
  "split": "train",
  "title": "Optional title",
  "abstract": "Optional abstract",
  "headings": ["Methods", "Results"],
  "text": "Full document text"
}
```

Build the artifact:

```bash
python3 scripts/06_Build_topic_model.py \
  --train-dataset datasets/topic_train.jsonl \
  --output artifacts/topics/topic_model.json \
  --language en \
  --leaf-clusters 8 \
  --encoder-revision immutable-model-revision \
  --seed 42
```

The command also writes `topic_nodes.json`,
`topic_representative_terms.csv`, and `topic_build_summary.json` under
`artifacts/topics/topic_model.diagnostics/`.

The document representation is `title + abstract + headings`. If those fields
are absent, the first non-empty content paragraphs are used deterministically.
The same fallback is used during training and inference.

## Inference

Without a static aspect dictionary:

```bash
python3 -m untie.cli article.txt \
  --language en \
  --mode hierarchical-static-keywords \
  --topic-model artifacts/topics/topic_model.json \
  --question "Which task was solved?"
```

With the existing static dictionary as aspect signal and fallback:

```bash
python3 -m untie.cli article.txt \
  --language en \
  --mode hierarchical-static-keywords \
  --topic-model artifacts/topics/topic_model.json \
  --model-params model_params/scart_tuned_model.json \
  --question "Which task was solved?"
```

The artifact encoder name must match the active sentence encoder. Old
`static-keywords` JSON is loaded only by its existing loader and is never
silently interpreted as a topic artifact.
When an immutable encoder revision/fingerprint was recorded, pass
`--topic-encoder-revision` at inference to enforce it.

## Scoring and fallback

`score_difference` remains the aspect signal. Topic specificity is computed
independently as smoothed cluster-vs-rest log specificity multiplied by
document support. Chunk aspect and topic values are min-max normalized before:

```text
theta_aspect * A
+ theta_topic * T
+ theta_interaction * A * T
```

Position, frequency and uniqueness modifiers remain on the aspect component.

By default, chunks without at least one routed topic term are excluded from QA,
similar to static keyword gating but with a document-specific vocabulary taken
from the routed cluster's `representative_terms`. Disable this with
`filter_unmatched_chunks=false` in the artifact mixing config.

The safe default `max_qa_chunks=0` keeps all topic-matched chunks after
filtering. A positive limit should be selected on dev only after checking
Evidence Recall@K.

Fallback is:

```text
leaf -> parent/ancestor -> global root -> static-keywords -> baseline
```

## Cluster-specific static dictionaries

The preferred Phase 2 path stores an optional `keyword_profile` on each leaf
node. Each profile is trained with the existing static-keyword SFFS over
documents assigned to that train-derived cluster:

- the builder persists train-only `doc_id -> leaf_id` assignments so SFFS uses
  the exact KMeans membership rather than reconstructing it through routing;
- candidate evidence is collected from train documents only;
- dev references select the subset and answer strategy;
- test documents are not accepted by the profile-training API;
- clustering and centroids are not rebuilt during keyword tuning.

At inference, the document is assigned to its nearest leaf centroid. By
default its profile is passed to the existing `score_chunks` hard filter, so
only chunks matching that cluster-specific dictionary reach extractive QA.
`--soft-rerank` on `scripts/11_Evaluate_hierarchical_test.py` disables that
drop: every chunk is answered, and keywords plus title/question similarity
plus routed representative terms only rank which answer consensus keeps.
That evaluation writes to a new directory and refuses to replace
`evaluation/`, `refit/evaluation/`, `refit_no_gold_contrast/evaluation/`,
or later isolated experiment directories.

Document-level cross-encoder ranking is a separate experiment in
`scripts/12_Evaluate_cross_encoder.py`. It does not use cluster SFFS
dictionaries. Every chunk is answered; `chunk_score` is either MiniLM
cross-encoder score or a mix with the frozen global static dictionary
(`keep_unmatched=True`). Mix weights are chosen on **dev** gold char-F1
and frozen for test. The script writes
`artifacts/hierarchical_topic_optimized/en/cross_encoder/` and refuses
to replace previous evaluation directories.

A follow-up experiment in `scripts/13_Evaluate_ce_topic_shortlist.py`
keeps all-chunk QA and the baseline `AnswerAggregator`. The frozen topic
route only expands the MiniLM query with leaf `representative_terms`
(not SFFS keyword profiles). CE then shortlists top-k answers; k is
chosen on dev. Output:
`artifacts/hierarchical_topic_optimized/en/ce_topic_query/`.

If the leaf has no profile, lookup ascends toward the root. If the selected
dictionary has no chunk matches in the default hard-filter path, the pipeline
falls back to the global static dictionary and then baseline.

The old `representative_terms` remain diagnostics for interpreting clusters.
They are not used as a hard filter when an artifact contains node keyword
profiles. Artifacts without profiles retain the previous topic-term behavior.

Generate reusable evidence with the global tuner, then attach per-cluster
profiles:

```bash
python3 scripts/05_Tune_model_keywords.py \
  --language en \
  --dataset datasets/scirex_structured.json \
  --cache-dir artifacts/keyword_tuning_cache

python3 scripts/07_Tune_topic_keywords.py \
  --topic-model artifacts/topics/topic_model.json \
  --dataset datasets/scirex_structured.json \
  --evidence-dir artifacts/keyword_tuning_cache/en/field-1/evidence \
  --output artifacts/topics/topic_model.cluster_keywords.json \
  --language en
```

The second command writes a sibling `*.topic-keywords.json` summary listing
trained and skipped leaves. Leaves with too few train or dev documents are
skipped safely and use the fallback chain at inference.

Routing, fallback reason, per-chunk scores, QA chunk/token counts and ratios
are returned in `PipelineResult.metadata`. Evaluation-only helpers
`compute_retrieval_metrics` and `compute_qa_ratios` provide Evidence Recall@K,
MRR, `QAChunkRatio`, and `QATokenRatio`.


## Final train+dev keyword refit

After the keyword-trial and routing grids have already been selected on held-out
dev, refit leaf dictionaries on the union of train and dev without changing
clusters or the frozen routing gate:

```bash
python3 scripts/10_Refit_hierarchical_keywords.py \
  --dataset datasets/scirex_structured.json \
  --source-dir artifacts/hierarchical_topic_optimized/en \
  --static-model experiments/analysis_results/keyword_tuning_task/en/full_v5/scart_tuned_model.json \
  --output-dir artifacts/hierarchical_topic_optimized/en/refit_no_gold_contrast
```

The command reads `keyword_best_trial` from `manifest.json` (trial 2 in the
current SciREx run: `min_keywords=6`, `union`, `stability_threshold=0.15`) and
keeps routing from `topic_model.tuned.json` (`similarity_threshold=0.75`,
`min_margin=0.0`). It does not re-run the 4 keyword trials or the 12-gate
routing search. Candidates are collected from train and dev with attention and
IDF, but **without** aspect-vs-gold contrast (that filter stays in global
static tuning). Cluster chunk scoring defaults to `only_weight`. The selected
trial stays frozen, but the final SFFS floor is lowered to 4 by default
(`--refit-min-keywords`). Outputs go to
`refit_no_gold_contrast/` unless `--contrast-against-gold` is set, in which
case they go to `refit/`. The source tuned model is never overwritten. Use
`--force` only to replace a previous refit in that directory. Evidence from an
earlier cache in the same output directory is reused only when fingerprints
match; disabling gold contrast uses a new evidence fingerprint.

## Dev tuning

The model-agnostic tuning API evaluates candidate routing/mixing configurations
without rebuilding train-derived nodes:

```python
from untie.topics import tune_topic_configuration

result = tune_topic_configuration(
    artifact,
    dev_documents,
    candidates=[(routing_config, mixing_config)],
    evaluator=evaluate_on_dev,
)
```

The evaluator may combine existing QA quality, Evidence Recall@K, fallback
rate, harm/win rates, and compute ratios. `result.artifact` keeps the original
centroids and terms and changes only the selected routing/mixing settings.

## Remaining exclusions

Hard-negative scoring, a learned ranker, source-span consensus deduplication
and a more advanced phrase matcher remain separate experiments.
