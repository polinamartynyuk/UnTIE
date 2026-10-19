# V8 dual-branch reader reranking

Independent implementation of `hypothesis/v8_dual_branch_hrr_and_v44_adaptive_fusion_protocol.md`.

It reads V4.4–V7 artifacts without modifying them and writes only to:

- `artifacts/qa_chunk_selection_v8_dual/`
- `logs/qa_chunk_selection_v8/`

The canonical experiment compares:

- **V8-HRR**: exact V7 S2 reference plus FullLEX and reader-agreement extensions;
- **V8-V44-AF**: fitted V44-CORE anchor plus FullLEX, exact V7 PairLEX, Q-A and harm-risk residual correction.

`SEM_COMMON=false` is frozen before evaluation because canonical V5/V7 feature rows do not contain a reproducible historical SEM feature on any cell. Layer P performs exact stored-prediction replay. Layer R is marked blocked rather than approximately reconstructed when a common historical all-candidate matrix is absent.

## Commands

Audit-only:

```bash
./untie_venv/bin/python3 -m experiments.qa_chunk_selection_v8 --phase audit
```

Small structural smoke run:

```bash
./untie_venv/bin/python3 -m experiments.qa_chunk_selection_v8 --smoke --modes local --phase all
```

Full background run:

```bash
bash experiments/qa_chunk_selection_v8/launch_background.sh
```

Resume an interrupted run with its exact directory:

```bash
./untie_venv/bin/python3 -u -m experiments.qa_chunk_selection_v8 \
  --run-dir artifacts/qa_chunk_selection_v8_dual/run_YYYYMMDD_HHMMSS \
  --resume --phase all
```
