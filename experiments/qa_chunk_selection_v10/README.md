# V10 / SRR

`Switch → pairwise rank → answer risk + evidence risk` replaces only the exact
H2 decision layer. Canonical documents, questions, candidate top-K, B0, QA,
H2 runtime features, outer folds and model seeds remain frozen.

V10.0.1 records a feasibility amendment for SciER384: inner splits are document-grouped and stratified from the seed-42 evidence labels, then reused unchanged for all model seeds. Evidence-risk requires at least 5 rows and 5 questions per class in INNER_TRAIN, while the final outer refit retains the original 10-row/5-question guard. A fail-fast audit checks all requested units before training.

The package writes only under `artifacts/qa_chunk_selection_v10_srr/` and
`logs/qa_chunk_selection_v10/`. V1–V9 code and artifacts are read-only and
hashed before and after a complete run.

Full background run:

```bash
bash experiments/qa_chunk_selection_v10/launch_background.sh
```

Optional pre-run smoke (one EN and one RU cell, one fold, one seed):

```bash
./untie_venv/bin/python3 -u -m experiments.qa_chunk_selection_v10 --smoke --phase run
```
