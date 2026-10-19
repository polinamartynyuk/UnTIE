# V4.4-RU hypothesis and cross-encoder ablations

This package is an isolated Russian extension of the shared V4.4 method. It does
not import from or write to the English `v4_4` package. The completed Russian
Q5/B0 artifact is read-only and every new run owns a fresh timestamp directory.

The protocol phase checks, with outer-EVAL exclusion:

- canonical `q0` versus the best TRAIN+CAL question for each aspect;
- K=3, K=4 and K=5;
- Russian V6, SEM and LEX weights over the same score families as English V4.4;
- conservative V6 fallback to B0 when gains are not stable across source folds;
- Unicode Token F1 plus secondary lemma F1 and a separate `LEX-LEM` reader;
- the same Q70 adaptive SEM routing and an exploratory gold-free Q5-consistency
  veto calibrated at Q30 from TRAIN+CAL only. For K=3 the confidence core is the best
  coherent pair against the third answer; K=4/5 use the established triple.
- per-aspect and pooled related-aspect diagnostics without merging RuSERRC labels.

The cross-encoder phase trains the established B0 `b2a` objective for three
seeds and five grouped folds with `BAAI/bge-reranker-v2-m3`, a multilingual
query-passage reranker that is directly comparable in role to the English
MS MARCO cross-encoder. `s-nlp/ruRoberta-large-paraphrase-v1` is deliberately
excluded from this cycle because its paraphrase objective is less aligned with
answer-bearing passage relevance. Model weights are not kept
after OOF scoring, avoiding roughly tens of gigabytes of redundant fold
checkpoints. Training reports and complete OOF predictions are retained.

Validate paths without starting the experiment:

```bash
untie_venv/bin/python -m experiments.qa_chunk_selection.v4_4_ru_ablation --check-only
```

Run the complete experiment in the background:

```bash
bash experiments/qa_chunk_selection/v4_4_ru_ablation/launch_background.sh
```

The launcher prints the PID, log and output paths. The log and `progress.jsonl`
are updated by stage, model, fold, seed, epoch and batch. A failed long run can
be resumed by passing its exact output directory directly:

```bash
untie_venv/bin/python -m experiments.qa_chunk_selection.v4_4_ru_ablation \
  --phase all \
  --output-dir artifacts/qa_chunk_selection_v4_4_ru_ablation_YYYYMMDD_HHMMSS \
  --resume
```

For the inexpensive protocol ablations over the existing B0 only:

```bash
bash experiments/qa_chunk_selection/v4_4_ru_ablation/launch_background.sh --phase protocol
```

