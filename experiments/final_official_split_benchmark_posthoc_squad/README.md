# Post-hoc frozen ModernBERT-SQuAD2 baselines

This standalone experiment adds two diagnostic English baselines without
modifying the completed final benchmark:

- `B5S-B0K`: source-run B0 top-K, frozen public SQuAD2 reader, maximum QA confidence;
- `B5S-ALL`: every eligible chunk, the same frozen reader, maximum QA confidence.

Both are explicitly post-hoc because the original TEST results were visible
before these systems were specified.

Run:

```bash
./untie_venv/bin/python -m experiments.final_official_split_benchmark_posthoc_squad --allow-downloads
```

Use `--resume --output <existing-run>` to continue an interrupted run.
