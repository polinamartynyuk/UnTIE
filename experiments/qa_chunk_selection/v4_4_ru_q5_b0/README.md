# Russian Q5, answerable-only CV and B0 OOF

This package is an isolated continuation of the completed full-RuSERRC preparation. It reads the
unlocked `data/dataset.jsonl` and immutable preprocessing tables and writes to a new artifact
directory. It never opens `locked/internal_test.jsonl`, filters shared preprocessing tables to unlocked document IDs, and never changes earlier artifacts.

Five Russian questions are frozen for each of the nine native aspects. The old `calibration`
documents are used only to select the Q5 consistency threshold from `0.6`, `0.8` and `1.0`; they are
saved separately and excluded from CV. The five-fold CV universe contains only Q5 instances from
the old `train` and `development` parts that have at least one robust positive.

B0 follows the existing English B0 recipe (question-balanced BCE, ratio 1:3:1, fold calibration and
early stopping) with `DeepPavlov/rubert-base-cased-sentence`, for seeds 42, 43 and 44. Its OOF file contains ranker scores, ranked chunk IDs and top-1 QA metrics. The canonical reader
predictions are reused from the Q5 cache, so no extra reader forward pass is needed. Adaptive V4.4
reader logic remains outside this preparation stage.

Run in the background:

```bash
bash experiments/qa_chunk_selection/v4_4_ru_q5_b0/launch_background.sh
```

To resume an interrupted output, use the path printed by the first launch:

```bash
OUTPUT_DIR=/absolute/path/to/incomplete/artifact \
bash experiments/qa_chunk_selection/v4_4_ru_q5_b0/launch_background.sh --resume
```

Completion is indicated by `COMPLETED.json`. Existing completed output is never overwritten.

`auto` selects the CUDA device with the most free memory. A device can also be pinned explicitly:

```bash
DEVICE=cuda:1 bash experiments/qa_chunk_selection/v4_4_ru_q5_b0/launch_background.sh
```
