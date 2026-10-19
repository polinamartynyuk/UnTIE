# V9 / H2 Adaptive Component Selection

Independent implementation of `hypothesis/v9_h2_adaptive_component_selection_protocol.md`.

The package reads V1–V8 code and artifacts without modifying them and writes only to:

- `artifacts/qa_chunk_selection_v9_h2_acs/`
- `logs/qa_chunk_selection_v9/`

V9 first replays H0/H1/H2/H3 candidate IDs, final answers and switch flags. It then builds four-fold document-grouped TRAIN-only `SELECT_OOF` predictions. One component configuration is selected per `(cell, outer_fold)` from the seed-averaged predictions and shared by all three final seed fits. Outer EVAL is not visible to the selector.

## Exact PairLEX audit

Exact V8 H2 already contains the full V7 PairLEX group. Consequently, exact set-union semantics make `CP` identical to `C0` and `CAP` identical to `CA`. V9 records this fact in `audit/pairlex_group.json`, verifies the identities, and never invents a duplicate PairLEX representation. The activity and minimum-gain gates prevent those aliases from being interpreted as new PairLEX evidence.

## Commands

Static audit only:

```bash
./untie_venv/bin/python3 -m experiments.qa_chunk_selection_v9 --phase audit
```

Exact H0/H1/H2/H3 replay only:

```bash
./untie_venv/bin/python3 -m experiments.qa_chunk_selection_v9 --phase replay
```

Small structural smoke run (not scientific evidence):

```bash
./untie_venv/bin/python3 -m experiments.qa_chunk_selection_v9 --smoke --phase all --bootstrap 100
```

Full background run:

```bash
bash experiments/qa_chunk_selection_v9/launch_background.sh
```

Resume an interrupted run using its exact directory:

```bash
./untie_venv/bin/python3 -u -m experiments.qa_chunk_selection_v9 \
  --run-dir artifacts/qa_chunk_selection_v9_h2_acs/run_YYYYMMDD_HHMMSS \
  --resume --phase all
```

Progress is written both to the background log and to `<run>/progress.jsonl`; a heartbeat is emitted every 30 seconds during long stages.

