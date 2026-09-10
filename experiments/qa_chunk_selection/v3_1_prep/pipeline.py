"""V3.1 overlay preparation pipeline. Independent of V3 training/eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v3.contract import LOCKED_SPLITS, POSTHOC_SPLITS
from experiments.qa_chunk_selection.v3.data import records_from_jsonl, split_unlocked, synthetic_records
from experiments.qa_chunk_selection.v3.overlay import TAXONOMY_VERSION

from .annotator import annotator_html, coverage_completion_candidates, export_annotator_queue
from .config import PrepConfig, v3_paths
from .contract import experiment_contract_v3_1_prep
from .freeze import freeze_overlay
from .io_guard import assert_isolated_output, assert_v3_not_output, load_json, read_jsonl, write_json, write_jsonl
from .qc import (
    build_working_overlay,
    counts,
    coverage_from_b0,
    cv_inventory,
    false_negative_audit,
    file_sha256,
    ingest_annotation_files,
    locked_ids,
    overlay_for_roles,
    readiness_checklist,
    validate_overlay,
)
from .taxonomy import is_filled, taxonomy_payload


class PipelinePrep:
    def __init__(self, config: PrepConfig) -> None:
        self.config = config
        self.output_dir = assert_isolated_output(Path(config.output_dir))
        assert_v3_not_output(self.output_dir, Path(config.v3_artifact_dir))
        self.progress = ProgressLogger(self.output_dir / "progress.jsonl")
        self.v3 = v3_paths(Path(config.v3_artifact_dir))
        self.records = ()
        self.unlocked = ()
        self.splits: dict[str, Any] = {}
        self.queue: list[dict[str, Any]] = []
        self.metadata: list[dict[str, Any]] = []
        self.b0_rows: list[dict[str, Any]] = []
        self.filled: list[dict[str, Any]] = []
        self.working: dict = {}
        self.validation: dict[str, Any] = {}
        self.counts: dict[str, Any] = {}
        self.coverage: dict[str, Any] = {}
        self.inventory: dict[str, Any] = {}
        self.readiness: dict[str, Any] = {}
        self.exit_code = 0
        self._handlers: dict[str, Callable[[], None]] = {
            "init": self.stage_init,
            "export-annotator": self.stage_export,
            "expand-coverage": self.stage_expand,
            "ingest": self.stage_ingest,
            "validate": self.stage_validate,
            "audit": self.stage_audit,
            "coverage": self.stage_coverage,
            "cv-inventory": self.stage_cv_inventory,
            "readiness": self.stage_readiness,
            "freeze": self.stage_freeze,
        }

    def run(self, stages: list[str] | None = None) -> int:
        names = list(stages or self.config.stages)
        self.progress.stage("start", f"output={self.output_dir}")
        self.progress.log(
            "V3 artifacts are read-only",
            v3=str(self.v3["root"]),
            write_target=str(self.output_dir),
        )
        for name in names:
            handler = self._handlers.get(name)
            if handler is None:
                raise ValueError(f"unknown stage: {name}")
            self.progress.stage(name)
            handler()
        self.progress.stage("done", "V3.1 data-prep finished (V3 artifacts untouched)")
        return self.exit_code

    def stage_init(self) -> None:
        write_json(self.output_dir / "contract.json", experiment_contract_v3_1_prep())
        write_json(self.output_dir / "config_resolved.json", self.config.to_dict())
        write_json(self.output_dir / "annotation" / "taxonomy.json", taxonomy_payload())
        if self.config.smoke:
            self.progress.log("smoke: synthetic unlocked records")
            self.records = synthetic_records(n_docs=8)
            self.unlocked = split_unlocked(self.records)
            from experiments.qa_chunk_selection.v3.overlay import sample_annotation_queue
            from experiments.qa_chunk_selection.v3.split import build_cv_splits

            sampled = sample_annotation_queue(self.unlocked, scores=None, hard_n=6, repr_n=6, seed=self.config.seed)
            self.queue = list(sampled["queue"])
            self.metadata = list(sampled["metadata"])
            self.splits = build_cv_splits(self.unlocked, n_folds=5, seed=self.config.seed)
            self.b0_rows = []
            for item in self.unlocked:
                ids = [chunk.chunk_id for chunk in item.chunks]
                self.b0_rows.append(
                    {
                        "question_id": item.question_id,
                        "document_id": item.document_id,
                        "ranked_ids": ids,
                        "selected_chunk_id": ids[0] if ids else None,
                        "qa_chunk_id": ids[0] if ids else None,
                        "scores": {str(cid): float(len(ids) - i) for i, cid in enumerate(ids)},
                        "seed": self.config.b0_seed,
                    }
                )
        else:
            missing = [name for name, path in self.v3.items() if name != "root" and not path.exists()]
            required = ["queue", "metadata", "dataset", "splits"]
            absent = [name for name in required if not self.v3[name].exists()]
            if absent:
                raise FileNotFoundError(f"V3 artifacts missing {absent} under {self.v3['root']}")
            if missing:
                self.progress.warn(f"optional V3 files missing: {missing}")
            self.progress.log("loading V3 dataset (read-only)")
            self.records = records_from_jsonl(self.v3["dataset"])
            self.unlocked = split_unlocked(self.records)
            self.unlocked = tuple(item for item in self.unlocked if item.positives)
            self.progress.log(
                "unlocked answerable universe",
                n_questions=len(self.unlocked),
                n_docs=len({item.document_id for item in self.unlocked}),
            )
            self.queue = read_jsonl(self.v3["queue"])
            self.metadata = read_jsonl(self.v3["metadata"])
            self.splits = load_json(self.v3["splits"])
            self.b0_rows = read_jsonl(self.v3["b0_oof"]) if self.v3["b0_oof"].exists() else []
            hashes = {
                "queue": file_sha256(self.v3["queue"]),
                "metadata": file_sha256(self.v3["metadata"]),
                "splits": file_sha256(self.v3["splits"]),
                "dataset": file_sha256(self.v3["dataset"]),
            }
            if self.v3["b0_oof"].exists():
                hashes["b0_oof"] = file_sha256(self.v3["b0_oof"])
            write_json(self.output_dir / "source_hashes.json", hashes)
            if self.v3["target_status"].exists():
                existing = read_jsonl(self.v3["target_status"])
                self.progress.log(
                    "V3 target_status.jsonl is read-only and will not be overwritten",
                    n_existing=len(existing),
                )
        locked = [item for item in self.records if item.split in LOCKED_SPLITS]
        posthoc = [item for item in self.records if item.split in POSTHOC_SPLITS]
        self.progress.log(
            "split lock check",
            n_internal_test=len(locked),
            n_seen_exploratory_dev=len(posthoc),
            taxonomy=TAXONOMY_VERSION,
        )
        (self.output_dir / "drafts").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "annotator").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "working").mkdir(parents=True, exist_ok=True)

    def stage_export(self) -> None:
        if not self.unlocked:
            self.stage_init()
        payload = export_annotator_queue(
            records=self.unlocked,
            queue=self.queue,
            metadata=self.metadata,
            annotator_id=self.config.annotator_id,
            progress=self.progress,
        )
        dest = self.output_dir / "annotator"
        write_jsonl(dest / "to_fill.jsonl", payload["queue"])
        write_json(dest / "guideline.json", {"steps": payload["guideline"], "taxonomy": payload["taxonomy"]})
        (dest / "index.html").write_text(annotator_html(payload["queue"]), encoding="utf-8")
        write_json(
            dest / "export_summary.json",
            {"n": payload["n"], "annotator_id": self.config.annotator_id, "scores_hidden": True},
        )
        self.progress.log("annotator pack written", n=payload["n"], path=str(dest / "to_fill.jsonl"))

    def stage_expand(self) -> None:
        if self.config.coverage_completion_n <= 0:
            self.progress.log("coverage-completion skipped (coverage_completion_n=0)")
            return
        if not self.unlocked:
            self.stage_init()
        already = {(str(row["question_id"]), int(row["chunk_id"])) for row in self.queue}
        extra_path = self.output_dir / "annotator" / "to_fill.jsonl"
        if extra_path.exists():
            already.update((str(row["question_id"]), int(row["chunk_id"])) for row in read_jsonl(extra_path))
        extra = coverage_completion_candidates(
            records=self.unlocked,
            b0_rows=self.b0_rows,
            already=already,
            k=self.config.coverage_completion_k,
            limit=self.config.coverage_completion_n,
            annotator_id=self.config.annotator_id,
            progress=self.progress,
        )
        existing = read_jsonl(extra_path) if extra_path.exists() else list(self.queue)
        write_jsonl(extra_path, existing + extra)
        write_jsonl(self.output_dir / "annotator" / "coverage_completion.jsonl", extra)
        self.progress.log("coverage-completion appended", n=len(extra), k=self.config.coverage_completion_k)

    def stage_ingest(self) -> None:
        pattern = str(self.output_dir / self.config.annotations_glob)
        to_fill = self.output_dir / "annotator" / "to_fill.jsonl"
        paths = [Path(pattern), to_fill]
        self.filled = ingest_annotation_files(paths, progress=self.progress)
        write_jsonl(self.output_dir / "working" / "ingested.jsonl", self.filled)
        if not self.filled:
            self.progress.warn(
                "no filled overlay rows yet: annotator/to_fill.jsonl still has empty v3_target_status. "
                "Freeze/V3.1 training is not allowed until target_mismatch labels exist."
            )
        self.progress.log("ingested filled rows", n=len(self.filled))

    def stage_validate(self) -> None:
        if not self.unlocked:
            self.stage_init()
        locked = locked_ids(self.records or self.unlocked)
        self.validation = validate_overlay(
            self.filled,
            self.unlocked,
            locked=locked,
            require_filled=True,
            progress=self.progress,
        )
        write_json(self.output_dir / "working" / "validation.json", self.validation)
        self.working = build_working_overlay(self.filled, keep_low_confidence=self.config.keep_low_confidence)
        write_jsonl(self.output_dir / "working" / "target_status.jsonl", list(self.working.values()))
        self.progress.log("validation", ok=self.validation.get("ok"), n_errors=self.validation.get("n_errors"))

    def stage_audit(self) -> None:
        self.counts = counts(self.filled, keep_low_confidence=self.config.keep_low_confidence)
        audit = false_negative_audit(self.filled, keep_low_confidence=self.config.keep_low_confidence)
        write_json(self.output_dir / "analysis" / "overlay_counts.json", self.counts)
        write_json(self.output_dir / "analysis" / "false_negative_audit.json", audit)
        self.progress.log(
            "audit",
            n_tm=self.counts.get("n_target_mismatch"),
            n_te=self.counts.get("n_target_evidence"),
            n_fn_audit=audit.get("n"),
        )

    def stage_coverage(self) -> None:
        if not self.unlocked:
            self.stage_init()
        overlay = overlay_for_roles(self.working)
        self.coverage = coverage_from_b0(
            records=self.unlocked,
            b0_rows=self.b0_rows,
            overlay=overlay,
            b0_seed=self.config.b0_seed,
            progress=self.progress,
        )
        write_json(self.output_dir / "analysis" / "coverage.json", self.coverage)
        self.progress.log(
            "coverage",
            coverage_at_1=self.coverage.get("TargetStatusCoverage@1"),
            tm_lower=self.coverage.get("TargetMismatch@1_lower_bound"),
        )

    def stage_cv_inventory(self) -> None:
        if not self.splits:
            self.stage_init()
        overlay = overlay_for_roles(self.working)
        self.inventory = cv_inventory(
            records=self.unlocked,
            splits=self.splits,
            overlay=overlay,
            progress=self.progress,
        )
        write_json(self.output_dir / "analysis" / "cv_tm_inventory.json", self.inventory)
        for run in self.inventory.get("runs") or []:
            self.progress.log(
                f"run {run['run']}: train TM questions={run['n_train_questions_with_TM']} "
                f"chunks={run['n_train_TM_chunks']} eval both={run['n_eval_questions_with_target_and_TM']}"
            )

    def stage_readiness(self) -> None:
        n_queue = len(read_jsonl(self.output_dir / "annotator" / "to_fill.jsonl") or self.queue)
        n_filled = len(self.filled)
        leakage = any("leakage" in str(err.get("error")) or "locked split" in str(err.get("error")) for err in self.validation.get("errors") or [])
        rewritten = any("source_label" in str(err.get("error")) for err in self.validation.get("errors") or [])
        self.readiness = readiness_checklist(
            counts_payload=self.counts,
            validation=self.validation,
            coverage=self.coverage,
            inventory=self.inventory,
            n_queue=n_queue,
            n_filled=n_filled,
            source_label_rewritten=rewritten,
            leakage=leakage,
            min_train_questions_with_TM=self.config.min_train_questions_with_TM,
            min_eval_questions_with_target_and_TM=self.config.min_eval_questions_with_target_and_TM,
            freeze_ready=(self.output_dir / "annotation" / "target_status_v3_1_frozen.jsonl").exists(),
        )
        write_json(self.output_dir / "analysis" / "readiness.json", self.readiness)
        self.progress.log(
            "readiness",
            v3_1_training_allowed=self.readiness.get("v3_1_training_allowed"),
            blocking=self.readiness.get("blocking"),
        )

    def stage_freeze(self) -> None:
        if not self.readiness:
            self.stage_readiness()
        if not self.readiness.get("v3_1_training_allowed") and not self.config.allow_partial_freeze:
            payload = {
                "frozen": False,
                "reason": "overlay not ready",
                "blocking": list(self.readiness.get("blocking") or []),
                "n_filled": len(self.filled),
                "n_target_mismatch": (self.counts or {}).get("n_target_mismatch", 0),
                "next": (
                    "Fill artifacts/qa_chunk_selection_v3_1_prep/annotator/to_fill.jsonl "
                    "(or drafts/*.jsonl) with v3_target_status, then re-run ingest…freeze. "
                    "Do not use --allow-partial-freeze to start V3.1 training."
                ),
            }
            write_json(self.output_dir / "analysis" / "freeze_skipped.json", payload)
            self.progress.warn(
                "freeze skipped: no verified target_mismatch yet. "
                f"blocking={payload['blocking']}. Annotate the queue first."
            )
            self.exit_code = 2
            return
        rows = list(self.working.values()) if self.working else self.filled
        hashes = {}
        hash_file = self.output_dir / "source_hashes.json"
        if hash_file.exists():
            hashes = load_json(hash_file)
        paths = freeze_overlay(
            self.output_dir,
            rows=rows,
            summary=self.counts,
            manifest={
                "n_target_evidence": self.counts.get("n_target_evidence"),
                "n_target_mismatch": self.counts.get("n_target_mismatch"),
                "n_partial_support": self.counts.get("n_partial_support"),
                "n_unresolved": self.counts.get("n_unresolved"),
                "n_by_aspect": self.counts.get("target_status_x_aspect"),
                "n_by_subtype": self.counts.get("target_status_x_subtype"),
                "n_by_sampling_stratum": self.counts.get("target_status_x_stratum"),
                "n_train_questions_with_TM_by_outer_run": self.inventory.get("runs"),
                "n_eval_questions_with_target_and_TM_by_outer_run": [
                    {
                        "run": run["run"],
                        "n_eval_questions_with_target_and_TM": run["n_eval_questions_with_target_and_TM"],
                    }
                    for run in self.inventory.get("runs") or []
                ],
                "coverage": self.coverage,
                "source_hashes": hashes,
                "cv_split_hash": self.splits.get("split_payload_hash"),
                "partial": not self.readiness.get("v3_1_training_allowed"),
                "allow_overwrite_freeze": False,
            },
        )
        write_json(self.output_dir / "analysis" / "freeze_paths.json", paths)
        self.progress.log("frozen overlay written", **paths)
