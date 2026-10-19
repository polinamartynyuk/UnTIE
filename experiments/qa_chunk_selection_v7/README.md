# V7 / RAPR

Independent implementation of the frozen Risk-Aware Progressive Residual Reranking protocol.

The experiment reads immutable V5 OOF features and frozen V6 references, writes only under
`artifacts/qa_chunk_selection_v7_rapr/`, and enforces a hard maximum of three QA calls per
question for primary `LOCAL S3`.

Run a cheap validation first:

```bash
./untie_venv/bin/python3 -m pytest -q tests/test_qa_chunk_selection_v7.py
./untie_venv/bin/python3 -m experiments.qa_chunk_selection_v7.run --smoke --phase run \
  --cells scirex --seeds 42 --outer-folds 0 --modes local \
  --run-dir artifacts/qa_chunk_selection_v7_rapr/smoke_<timestamp>
```

Start the full run in the background with `launch_background.sh`. Every run uses a new timestamp;
resume requires passing the exact existing run directory and `--resume`.
