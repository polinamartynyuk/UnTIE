# SciMDIX and SciER preparation

This isolated pipeline prepares the Russian SciMDIX and English SciER corpora for the existing chunk-selection experiments. It never writes into prior datasets, code, or experiment artifacts; every run requires a new output directory and freezes source files with revision and SHA-256 metadata.

Chunking is language-specific: SciMDIX defaults to 128 tokens with 24-token overlap; SciER defaults to the established English SciREX settings of 384 tokens with 50-token overlap. Both values have separate CLI options, so the earlier SciER 128/24 preparation can remain as an explicit ablation.

SciMDIX uses its Russian `aspects` annotations as exact gold spans and preserves the separate NER/RE layer for later studies. SciER uses the PLM release because it retains document-level tokens, official sentence boundaries, exact entity spans, relations, and all four official splits. Its long full-text papers are chunked with their supplied sentence boundaries.

Labels are produced by the project's existing Phase 1 algorithm: extractive QA is matched to official gold answers, BM25 supplies negative candidates, and embedding clustering removes likely false negatives. An annotation of another aspect is not itself a hard-negative rule. Russian and English model profiles are selected explicitly. Q5 inventories are saved, while Phase 1 uses the first primary question exactly as the established data-preparation chain does.

The experiment-ready training file and grouped CV contain only QA-answerable instances. All Phase 1 instances, including `no_qa_positive`, remain in audit files. Official SciER dev is exploratory; official test/OOD test and SciMDIX test are kept under `locked/` and never enter CV.

Run both complete datasets in the background with downloads:

```bash
bash experiments/qa_chunk_selection/external_data/launch_background.sh
```

Use the already inspected local snapshots without network access:

```bash
SCIMDIX_SOURCE=/tmp/untie_dataset_sources/SciMDIX \
SCIER_SOURCE=/tmp/untie_dataset_sources/SciER \
DEVICE=cuda:0 \
bash experiments/qa_chunk_selection/external_data/launch_background.sh
```

The launcher prints the PID, output directory, and log file. Follow progress with:

```bash
tail -f logs/qa_chunk_selection_external_data_YYYYMMDD_HHMMSS.log
```

For a small diagnostic run, call the module directly with `--limit-docs 1`; use a new output directory each time.
