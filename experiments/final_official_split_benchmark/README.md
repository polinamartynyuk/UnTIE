# Final Official-Split Benchmark / H2 Family Confirmation

This is a separate experiment namespace. It reads V1–V10 inputs and reuses their
tested feature/policy implementations, but never writes into historical artifact
directories.

The lifecycle is deliberately one-way:

1. `audit` resolves official splits, duplicate/exposure status, model capability,
   evidence-label availability, and hashes V1–V10 readouts;
2. `develop` trains only on TRAIN and selects on VAL;
3. `freeze` locks the system/model/prompt/threshold contract;
4. `refit` retrains on TRAIN+VAL and builds TRAIN+VAL OOF meta features;
5. `predict-test` accepts only gold-free runtime records and writes immutable
   predictions for every applicable real system;
6. `evaluate` unlocks gold only after all prediction hashes are complete;
7. `resources` measures selected systems in fresh processes;
8. `report` writes the final Markdown report and verifies that V1–V10 hashes did
   not change.

The first background command intentionally stops at `READY_FOR_TEST`:

```bash
bash experiments/final_official_split_benchmark/launch_background.sh
```

The script prints the exact `RUN_DIR`, log path, and PID. After reviewing
`frozen/freeze_contract.json`, `refit/COMPLETE.json`, and
`test_exposure/exposure.json`, continue the same run:

```bash
RUN_DIR=/absolute/path/printed/by/the/first/command
nohup untie_venv/bin/python3 -m experiments.final_official_split_benchmark \
  --run-dir "$RUN_DIR" --resume --phase all --unlock-test --allow-downloads \
  >"logs/final_official_split_benchmark/$(basename "$RUN_DIR")_test.log" 2>&1 &
```

Progress is written both to the log and `<RUN_DIR>/progress.jsonl`; the current
phase is in `<RUN_DIR>/status.json`. `--allow-downloads` permits only the frozen
model IDs in the contract. If omitted, absent BGE/ModernBERT/Qwen systems are
recorded as `BLOCKED_MISSING_MODEL` rather than silently replaced.

The synthetic validation command never opens official TEST:

```bash
untie_venv/bin/python3 -m experiments.final_official_split_benchmark \
  --phase smoke --cells scirex --seeds 42 --inner-folds 3 \
  --run-dir artifacts/final_official_split_benchmark/smoke_local
```
