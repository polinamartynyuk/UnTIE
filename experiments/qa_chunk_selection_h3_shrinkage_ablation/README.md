# H3 shrinkage ablation

This isolated post-hoc experiment tests two compact ways to reduce the extra
risk introduced by H3 agreement features while preserving the existing H2/H3
method family.

* `H3-BLEND` fits the existing H2 and H3 utility/harm models and selects a
  convex prediction blend from `lambda = 0, .25, .5, .75, 1`.
* `H3-GROUP-RIDGE` keeps the H3 feature schema and selects a separate L2
  multiplier for the three agreement fields from `1, 2, 4, 8, 16`.

All model hyperparameters, policy thresholds and the added shrinkage parameter
are selected from document-grouped OOF rows.  VAL is used only for development
reporting.  The final policy is refit on frozen TRAIN+VAL OOF rows and replayed
on the already stored TEST runtime features.

Because the final TEST results were already inspected before this experiment,
its TEST comparison is exploratory rather than a new blind confirmation.
The completed final benchmark is read-only and its consumed files are hashed
before and after the run.

Run both variants in the background:

```bash
bash /home/aiserver/POLINA/UnTIE_project/experiments/qa_chunk_selection_h3_shrinkage_ablation/launch_background.sh
```

Resume an interrupted run:

```bash
RUN_DIR=/absolute/path/to/existing/run bash /home/aiserver/POLINA/UnTIE_project/experiments/qa_chunk_selection_h3_shrinkage_ablation/launch_background.sh --resume
```

The launcher prints the log and artifact paths.  The run also updates
`progress.jsonl` and `status.json` and writes a heartbeat every 30 seconds.
