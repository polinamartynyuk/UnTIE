# V4.4 on SciMDIX and SciER

This package runs the same adaptive reader-consensus concept on the newly prepared datasets without changing any earlier V4.4 implementation or artifact. Every run owns a new output root. Inputs are read-only and source hashes/model identities are recorded.

Default datasets are Russian SciMDIX (128/24 preparation) and English SciER (384/50 preparation). `scier_128` is available as a separate chunk-size ablation. Each dataset is processed independently from end to end:

1. five frozen questions per native aspect are evaluated with the established language-specific extractive QA model;
2. B0 is trained in five document-grouped outer folds for seeds 42, 43, and 44;
3. the best question per aspect is selected on TRAIN+CAL for each outer run;
4. V6, semantic-reader, and lexical-reader coefficients are selected separately on that dataset's TRAIN+CAL source folds;
5. each adaptive routing Q70 threshold is fitted only from TRAIN+CAL features;
6. the selected fold configuration is evaluated once on that run's EVAL documents.

No English, RuSERRC, or other-dataset learned weights or thresholds are imported. The feature definitions, grids, safety constraints, B0 training recipe, three-seed pairing, and document-grouped CV remain common.

Models match the earlier language experiments:

- English QA: local `bert_eng_qa_baseroberta_model`; English semantic encoder: local `eng_sentence_transformer_model`; English B0: `cross-encoder/ms-marco-MiniLM-L6-v2`.
- Russian QA: local `rubert_ru_qa_model`; Russian semantic encoder and B0: `DeepPavlov/rubert-base-cased-sentence`.

Run the primary pair in the background:

```bash
DEVICE=cuda:1 bash experiments/qa_chunk_selection/v4_4_external/launch_background.sh
```

Run the SciER 128/24 ablation separately:

```bash
DATASETS="scier_128" DEVICE=cuda:1 \
  bash experiments/qa_chunk_selection/v4_4_external/launch_background.sh
```

Run only one primary dataset by setting `DATASETS` to `scimdix` or `scier_384`. The launcher prints the PID, output directory, and log path. Follow it with `tail -f <LOG>`. If an interrupted run must be continued, invoke the Python module with the same output directory and `--resume`; completed dataset subdirectories are preserved.
