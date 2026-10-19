"""V9-owned canonical feature cache using the frozen V8 feature construction."""
from __future__ import annotations

from collections import defaultdict
import copy

from experiments.qa_chunk_selection.v3.data import records_from_jsonl
from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS as V5_CELLS
from experiments.qa_chunk_selection_v6.features import enrich_question as v6_enrich
from experiments.qa_chunk_selection_v8.features import enrich_reader_features, hrr_rows
from .io_guard import read_jsonl, sha256, write_json, write_jsonl


class DataStore:
    """Reproduce V8 inputs exactly, but keep every generated cache inside V9."""

    def __init__(self, config, progress):
        self.config = config
        self.progress = progress
        self._items = {}
        self._cache = {}

    def items(self, cell):
        if cell not in self._items:
            self._items[cell] = {
                str(item.question_id): item for item in records_from_jsonl(V5_CELLS[cell].records)
            }
        return self._items[cell]

    def load(self, cell, fold, seed, role):
        key = (cell, fold, seed, role)
        if key in self._cache:
            return copy.deepcopy(self._cache[key])
        path = (
            self.config.v5_run / cell / f"outer_fold_{fold}" / f"seed_{seed}"
            / ("inner_meta/features.jsonl" if role == "meta" else "eval/features.jsonl")
        )
        raw = read_jsonl(path)
        source_groups = defaultdict(list)
        for row in raw:
            source_groups[str(row["question_id"])].append(row)
        items = self.items(cell)
        runtime, targets, training_rows = [], {}, []
        for index, (question_id, source) in enumerate(sorted(source_groups.items()), 1):
            question_runtime, target = v6_enrich(
                cell, source, items[question_id], outer_fold=fold, seed=seed,
                meta_source=role == "meta",
            )
            question_runtime = enrich_reader_features(question_runtime, items[question_id])
            target["gold_answers"] = [str(value) for value in items[question_id].gold_answers]
            target["chunk_label"] = {
                str(row["chunk_id"]): str(items[question_id].chunk_by_id(int(row["chunk_id"])).label)
                for row in question_runtime
            }
            target["primary_class"] = {
                str(row["chunk_id"]): str(items[question_id].chunk_by_id(int(row["chunk_id"])).primary_class)
                for row in question_runtime
            }
            runtime.extend(question_runtime)
            targets[(cell, question_id)] = target
            training_rows.extend(hrr_rows(question_runtime, target))
            self.progress.count(
                index, len(source_groups), item=f"{cell} {role} exact reader features",
                extra={"fold": fold, "seed": seed},
            )
        cache = self.config.run_dir / "cache/canonical_features" / cell / f"outer_fold_{fold}" / f"seed_{seed}"
        write_jsonl(cache / f"{role}_runtime.jsonl", runtime)
        write_jsonl(cache / f"{role}_hrr_targets.jsonl", training_rows)
        write_json(cache / f"{role}_cache_manifest.json", {
            "cell": cell,
            "outer_fold": fold,
            "seed": seed,
            "role": role,
            "source_features": str(path.resolve()),
            "source_features_sha256": sha256(path),
            "feature_version": "exact-v8-hrr-h2-h3",
            "agreement_version": "exact-v8-h3-minus-h2",
            "pairlex_version": "exact-v7-b0-answer-reference",
            "key_fields": [
                "dataset_cell", "canonical_document_id", "question_id", "chunk_id",
                "outer_fold", "seed",
            ],
            "config_tuning_meta_fold": "assigned during tune_policy from canonical_document_id",
        })
        result = (runtime, training_rows, targets)
        self._cache[key] = copy.deepcopy(result)
        return copy.deepcopy(result)
