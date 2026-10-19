"""Replay historical B0 and V5-A1 from immutable V5 features/predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.qa_chunk_selection.v3.data import records_from_jsonl
from experiments.qa_chunk_selection.v4.answers import pair_token_f1
from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS as V5_CELLS
from experiments.qa_chunk_selection.v5_question_conditioned_residual.core import runtime_choice
import joblib

from .contract import CELLS, OUTER_FOLDS, SEEDS, V5_RUN
from .io_guard import read_jsonl, write_json


def replay(v5_run: Path, cells=CELLS, *, strict=True) -> dict:
    report = {}
    for cell in cells:
        items = {str(item.question_id): item for item in records_from_jsonl(V5_CELLS[cell].records)}
        checked = 0; errors = []
        for fold in OUTER_FOLDS:
            for seed in SEEDS:
                root = v5_run / cell / f"outer_fold_{fold}" / f"seed_{seed}" / "eval"
                features = {str(r["question_id"]): [] for r in read_jsonl(root / "features.jsonl")}
                for row in read_jsonl(root / "features.jsonl"):
                    features[str(row["question_id"])].append(row)
                b0 = {str(row["question_id"]): row for row in read_jsonl(root / "B0/predictions.jsonl")}
                a1 = {str(row["question_id"]): row for row in read_jsonl(root / "V5-Q-A1/predictions.jsonl")}
                residual = joblib.load(v5_run / cell / f"outer_fold_{fold}" / f"seed_{seed}" / "residual/A1/model.joblib")
                if set(features) != set(b0) or set(features) != set(a1):
                    errors.append(f"coverage:{fold}:{seed}"); continue
                for qid, rows in features.items():
                    rows.sort(key=lambda row: int(row["b0_rank"])); item = items[qid]
                    expected = rows[0]
                    token_f1 = max((pair_token_f1(expected["qa_answer"], gold) for gold in item.gold_answers), default=0.0)
                    if int(b0[qid]["selected_chunk_id"]) != int(expected["chunk_id"]) or abs(float(b0[qid]["token_f1"]) - token_f1) > 1e-8:
                        errors.append(f"b0:{fold}:{seed}:{qid}")
                    selected = int(a1[qid]["selected_chunk_id"])
                    replayed, _ = runtime_choice(rows, residual, "A1")
                    if selected != int(replayed["chunk_id"]):
                        errors.append(f"a1_selection:{fold}:{seed}:{qid}")
                    checked += 1
        report[cell] = {"status": "PASS" if not errors else "FAIL", "questions_seed_rows": checked,
                        "errors": errors[:100], "exact_b0_answer_replay": not errors,
                        "v5_a1_contract": "stored selection belongs to exact upstream shortlist"}
        if strict and errors:
            raise RuntimeError(f"HISTORICAL_REPLAY_FAILED: {cell}: {errors[:3]}")
    return report


def parser():
    p = argparse.ArgumentParser(description="V6 historical replay")
    p.add_argument("--config", type=Path); p.add_argument("--datasets", nargs="+", default=list(CELLS))
    p.add_argument("--strict", action="store_true"); p.add_argument("--read-only-history", action="store_true")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--v5-run", type=Path, default=V5_RUN)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.read_only_history:
        raise ValueError("--read-only-history is mandatory")
    write_json(args.output / "historical_replay.json", replay(args.v5_run, tuple(args.datasets), strict=args.strict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

