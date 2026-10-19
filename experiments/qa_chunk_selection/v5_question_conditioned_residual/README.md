# V5 multilingual question-conditioned residual reranking

Изолированная реализация протокола `hypothesis/v5_multilingual_question_conditioned_residual_protocol.md`.
Она читает исторические V3–V4.5 артефакты без записи в них и создаёт новый timestamped run в
`artifacts/qa_chunk_selection_v5/`.

Ячейки: SciREX, SciER 384/50, SciER 128/24, RuSERRC, SciMDIX. SciER-ячейки в отчёте отмечаются
как два зависимых представления одного корпуса.

Этапы:

1. `audit` — проверка моделей, разбиений, candidate universe, labels и заморозка контракта;
2. `run` — nested document-grouped META OOF, gate, A0/A1 residual, outer EVAL;
3. `analysis` — метрики, document-cluster bootstrap и отчёт;
4. `all` — все этапы.

Полный фоновый запуск:

```bash
bash experiments/qa_chunk_selection/v5_question_conditioned_residual/launch_background.sh
```

Продолжение конкретного run:

```bash
python3 -m experiments.qa_chunk_selection.v5_question_conditioned_residual \
  --output-dir artifacts/qa_chunk_selection_v5/run_YYYYMMDD_HHMMSS \
  --phase all --resume
```

Технический smoke (его результаты не являются научными):

```bash
bash experiments/qa_chunk_selection/v5_question_conditioned_residual/launch_background.sh \
  --cells russerrc --smoke
```

