# Full RuSERRC preparation for V4.4

This opt-in package reads the complete local 591-document mirror by default. It can also
download the three official `ruserrc_aspects` CSV parts. It records source SHA-256 hashes
and, for downloads, the resolved Git commit; then removes inline markup with exact
offset tracking, and prepares every native aspect: task, goal, contribution,
method, tool, use, advantage, example, and conclusion.

The processing mirrors the English data chain: one shared chunking per document,
source-only lexical gold mentions, frozen extractive QA, Phase 1 hard-negative
assignment, document-level train/calibration/development/internal-test roles, and
five-fold grouped CV over the unlocked roles. It reuses existing UnTIE chunking,
normalization, Phase 1 assignment, V2 splitting, V3 CV, and HN-quality code. The
Russian sentence splitter and Russian QA/embedding model profile are selected
explicitly.

Every run requires a new output directory. Raw downloads, intermediate tables,
the locked internal test, final JSONL files, progress events, source revision,
and checks all remain under that directory. Existing datasets, experiments, and
artifacts are read-only.

Run the complete preparation from the local full corpus in the background:

```bash
bash experiments/qa_chunk_selection/v4_4_ru_data/launch_background.sh
```

Select a CUDA device if needed:

```bash
DEVICE=cuda:0 bash experiments/qa_chunk_selection/v4_4_ru_data/launch_background.sh
```

For an already downloaded official checkout, pass its `ruserrc_aspects` folder:

```bash
untie_venv/bin/python -m experiments.qa_chunk_selection.v4_4_ru_data \
  --source-dir /path/to/ruserrc_aspects \
  --output-dir artifacts/qa_chunk_selection_v4_4_ru_data_local_001
```

The V4.4-compatible unlocked inputs are written to `data/dataset.jsonl`,
`data/v3_cv_splits.json`, and `data/hn_quality_flags.jsonl`. The internal test is
kept separately at `locked/internal_test.jsonl`.

Download from GitHub explicitly when network access is available:

```bash
untie_venv/bin/python -m experiments.qa_chunk_selection.v4_4_ru_data \
  --download --output-dir artifacts/qa_chunk_selection_v4_4_ru_data_download_001
```
