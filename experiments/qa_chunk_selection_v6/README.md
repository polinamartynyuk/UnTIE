# V6 / UAR-G

This package implements the frozen V6 universal adaptive residual reranking protocol. It reads the completed V5 upstream OOF features and historical datasets strictly read-only, creates a new timestamped V6 run, and fits only the V6 meta-layer.

The default development run evaluates S0, S1, S2, S3, S4-src, S4-lex and S4 in LOCAL, global-fold POOLED and LOCO-meta modes. S5 is recorded as unavailable because no exact SEM/QA-confidence contract is replayable on every protocol cell. Every seed is fitted independently.

Run in the background:

```bash
bash experiments/qa_chunk_selection_v6/launch_background.sh
```

Follow progress using the printed log path. A failed or completed run is never overwritten; resume the exact same command with `--resume` only when recovering an interrupted run.

