"""V10-owned reproduction of the frozen V9 canonical feature cache."""
from __future__ import annotations

from collections import defaultdict
import copy

from experiments.qa_chunk_selection.v3.data import records_from_jsonl
from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS as V5_CELLS
from experiments.qa_chunk_selection_v6.features import enrich_question as v6_enrich
from experiments.qa_chunk_selection_v8.features import enrich_reader_features, hrr_rows
from experiments.qa_chunk_selection_v8.pipeline import _embargo
from .io_guard import read_jsonl, sha256, write_json, write_jsonl


class DataStore:
    def __init__(self, config, progress):
        self.config = config; self.progress = progress; self._items = {}; self._cache = {}

    def items(self, cell):
        if cell not in self._items:
            self._items[cell] = {str(x.question_id): x for x in records_from_jsonl(V5_CELLS[cell].records)}
        return self._items[cell]

    def load(self, cell, fold, seed, role):
        key = (cell, fold, seed, role)
        if key in self._cache: return copy.deepcopy(self._cache[key])
        source = self.config.v5_run / cell / f"outer_fold_{fold}" / f"seed_{seed}" / (
            "inner_meta/features.jsonl" if role == "meta" else "eval/features.jsonl"
        )
        raw = read_jsonl(source); groups = defaultdict(list)
        for row in raw: groups[str(row["question_id"])].append(row)
        items = self.items(cell); runtime = []; targets = {}; training_rows = []
        for index, (question_id, rows) in enumerate(sorted(groups.items()), 1):
            question_runtime, target = v6_enrich(
                cell, rows, items[question_id], outer_fold=fold, seed=seed, meta_source=role == "meta",
            )
            question_runtime = enrich_reader_features(question_runtime, items[question_id])
            target["gold_answers"] = [str(x) for x in items[question_id].gold_answers]
            target["chunk_label"] = {
                str(row["chunk_id"]): str(items[question_id].chunk_by_id(int(row["chunk_id"])).label)
                for row in question_runtime
            }
            target["primary_class"] = {
                str(row["chunk_id"]): str(items[question_id].chunk_by_id(int(row["chunk_id"])).primary_class)
                for row in question_runtime
            }
            runtime.extend(question_runtime); targets[(cell, question_id)] = target
            training_rows.extend(hrr_rows(question_runtime, target))
            self.progress.count(index, len(groups), item=f"{cell} {role} exact V9/H2 features", extra={"fold": fold, "seed": seed})
        historical = self.config.v9_run / "cache/canonical_features" / cell / f"outer_fold_{fold}" / f"seed_{seed}"
        historical_runtime = historical / f"{role}_runtime.jsonl"
        historical_hrr = historical / f"{role}_hrr_targets.jsonl"
        if historical_runtime.is_file() and historical_hrr.is_file():
            current_runtime = [{k: v for k, v in row.items()} for row in runtime]
            if current_runtime != read_jsonl(historical_runtime) or training_rows != read_jsonl(historical_hrr):
                raise RuntimeError(f"V9_CANONICAL_FEATURE_REPLAY_FAILED: {cell}/f{fold}/s{seed}/{role}")
        cache = self.config.run_dir / "cache/canonical_features" / cell / f"outer_fold_{fold}" / f"seed_{seed}"
        write_jsonl(cache / f"{role}_runtime.jsonl", runtime)
        write_jsonl(cache / f"{role}_hrr_targets.jsonl", training_rows)
        write_json(cache / f"{role}_manifest.json", {
            "source": str(source.resolve()), "source_sha256": sha256(source),
            "historical_v9_runtime": str(historical_runtime.resolve()),
            "historical_v9_runtime_sha256": sha256(historical_runtime) if historical_runtime.is_file() else None,
            "exact_replay": historical_runtime.is_file(), "cell": cell, "fold": fold, "seed": seed, "role": role,
        })
        result = (runtime, training_rows, targets); self._cache[key] = copy.deepcopy(result)
        return copy.deepcopy(result)

    def scope(self, cell, fold, seed):
        train = self.load(cell, fold, seed, "meta"); evaluate = self.load(cell, fold, seed, "eval")
        train_runtime, train_rows, train_targets, embargo = _embargo(
            train[0], train[1], train[2], evaluate[0],
        )
        return train_runtime, train_rows, train_targets, evaluate[0], evaluate[1], evaluate[2], embargo
