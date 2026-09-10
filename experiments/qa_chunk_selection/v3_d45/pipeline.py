"""Isolated D4/D5 diagnostic pipeline. V3 artifacts are read-only."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from experiments.qa_chunk_selection.models import load_frozen_qa
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v2.cache import FeatureCache
from experiments.qa_chunk_selection.v3.contract import LOCKED_SPLITS
from experiments.qa_chunk_selection.v3.data import records_from_jsonl, split_unlocked, synthetic_records
from experiments.qa_chunk_selection.v3.split import records_for_role
from experiments.qa_chunk_selection.v3_1_auto_prep.qc import file_sha256

from .config import D45Config
from .contract import experiment_contract_v3_d45
from .d4_oracle import run_d4_oracle
from .d5_audit import run_d5_audit
from .io_guard import assert_isolated_output, load_json, write_json, write_jsonl


class PipelineD45:
    def __init__(self, config: D45Config) -> None:
        self.config = config
        self.output_dir = assert_isolated_output(Path(config.output_dir))
        self.progress = ProgressLogger(self.output_dir / "progress.jsonl")
        self.v3 = Path(config.v3_artifact_dir)
        self.unlocked = ()
        self.splits: dict[str, Any] = {}
        self.gates: dict[str, Any] = {}
        self.bundle = None
        self.cache = FeatureCache(self.output_dir / "cache")
        self.exit_code = 0
        self._handlers: dict[str, Callable[[], None]] = {
            "init": self.stage_init,
            "check-gates": self.stage_gates,
            "d4-oracle": self.stage_d4,
            "d5-audit": self.stage_d5,
            "report": self.stage_report,
        }

    def run(self, stages: list[str] | None = None) -> int:
        names = list(stages or self.config.stages)
        self.progress.stage("start", f"output={self.output_dir} smoke={self.config.smoke}")
        self.progress.log("E0/V2/V3/V3.1 are read-only", write_target=str(self.output_dir))
        for name in names:
            handler = self._handlers.get(name)
            if handler is None:
                raise ValueError(f"unknown stage: {name}")
            self.progress.stage(name)
            handler()
        self.progress.stage("done", f"cache_hit_rate={self.cache.stats().get('hit_rate')}")
        return self.exit_code

    def _answer_fn(self):
        if self.config.smoke:
            from experiments.qa_chunk_selection.models import lexical_answer

            return lexical_answer
        if self.bundle is None:
            self.progress.log("loading frozen extractive QA only (no cross-encoder)")
            self.bundle = load_frozen_qa(device=self.config.device, smoke=False)
        return self.bundle.answer

    def _model_id(self) -> str:
        if self.config.smoke or self.bundle is None:
            return "lexical-smoke" if self.config.smoke else "qa-pending"
        return str(self.bundle.qa_model_name)

    def _train_cal_records(self):
        if self.config.smoke or not self.splits.get("runs"):
            return tuple(item for item in self.unlocked if item.split in {"train", "calibration"})
        seen: dict[str, Any] = {}
        for run in self.splits.get("runs") or []:
            run_id = int(run["run"])
            for role in ("train", "calibration"):
                for item in records_for_role(self.unlocked, self.splits, run=run_id, role=role):
                    seen[item.question_id] = item
        return tuple(seen.values())

    def stage_init(self) -> None:
        write_json(self.output_dir / "contract.json", experiment_contract_v3_d45())
        write_json(self.output_dir / "config_resolved.json", self.config.to_dict())
        if self.config.smoke:
            records = synthetic_records(n_docs=8)
            self.unlocked = tuple(item for item in split_unlocked(records) if item.positives)
            self.progress.log("smoke: synthetic unlocked records", n=len(self.unlocked))
            return
        dataset = self.v3 / "data" / "dataset.jsonl"
        splits = self.v3 / "data" / "v3_cv_splits.json"
        if not dataset.exists() or not splits.exists():
            raise FileNotFoundError(f"V3 dataset/splits missing under {self.v3}")
        records = records_from_jsonl(dataset)
        self.unlocked = tuple(item for item in split_unlocked(records) if item.positives)
        self.splits = load_json(splits)
        locked = [item for item in records if item.split in LOCKED_SPLITS]
        write_json(
            self.output_dir / "source_hashes.json",
            {
                "dataset": file_sha256(dataset),
                "splits": file_sha256(splits),
                "document_id_hash": self.splits.get("document_id_hash"),
                "split_payload_hash": self.splits.get("split_payload_hash"),
            },
        )
        self.progress.log("loaded V3 universe (read-only)", n_unlocked=len(self.unlocked), n_internal_test=len(locked))

    def stage_gates(self) -> None:
        if not self.unlocked:
            self.stage_init()
        d4_path = self.v3 / "analysis" / "positive_quality" / "d4_gate.json"
        d5_path = self.v3 / "analysis" / "positive_quality" / "d5_gate.json"
        if self.config.smoke:
            self.gates = {"d4": {"pass": True, "smoke": True}, "d5": {"pass": True, "smoke": True}, "ok": True}
        else:
            d4 = load_json(d4_path) if d4_path.exists() else {"pass": False}
            d5 = load_json(d5_path) if d5_path.exists() else {"pass": False}
            self.gates = {"d4": d4, "d5": d5, "ok": bool(d4.get("pass") and d5.get("pass"))}
        write_json(self.output_dir / "analysis" / "preconditions.json", self.gates)
        if not self.gates["ok"]:
            self.progress.warn("V3 D4/D5 gates not passed; diagnostic will still run but runtime follow-up stays closed")
        else:
            self.progress.log("V3 D4/D5 gates accepted (read-only)")

    def stage_d4(self) -> None:
        if not self.unlocked:
            self.stage_init()
        records = self._train_cal_records()
        self.progress.log("D4 oracle on TRAIN/CAL positives", n_questions=len(records))
        rows, summary = run_d4_oracle(
            records,
            answer_fn=self._answer_fn(),
            cache=self.cache,
            model_id=self._model_id() if not self.config.smoke else "lexical-smoke",
            qa_answerer=None if self.config.smoke or self.bundle is None else self.bundle.qa_answerer,
            progress=self.progress,
        )
        write_jsonl(self.output_dir / "analysis" / "d4" / "oracle_rows.jsonl", rows)
        write_json(self.output_dir / "analysis" / "d4" / "oracle_summary.json", summary)
        self.progress.log(
            "D4 oracle done",
            gain=summary.get("oracle_gain_vs_full"),
            runtime_allowed=summary.get("runtime_d4_allowed"),
        )

    def stage_d5(self) -> None:
        if not self.unlocked:
            self.stage_init()
        rows, summary = run_d5_audit(self.unlocked, min_overlap=0.6)
        write_jsonl(self.output_dir / "analysis" / "d5" / "clusters.jsonl", rows)
        write_json(self.output_dir / "analysis" / "d5" / "audit.json", summary)
        self.progress.log(
            "D5 audit done",
            independent=summary.get("n_with_ge2_independent_clusters"),
            naive=summary.get("n_with_ge2_positive_chunks"),
            aggregation_allowed=summary.get("aggregation_allowed"),
        )

    def stage_report(self) -> None:
        d4 = load_json(self.output_dir / "analysis" / "d4" / "oracle_summary.json") if (self.output_dir / "analysis" / "d4" / "oracle_summary.json").exists() else {}
        d5 = load_json(self.output_dir / "analysis" / "d5" / "audit.json") if (self.output_dir / "analysis" / "d5" / "audit.json").exists() else {}
        payload = {
            "title": "V3 D4/D5 diagnostic autodump (not a human report)",
            "preconditions": self.gates,
            "d4": d4,
            "d5": d5,
            "runtime_d4_allowed": bool(d4.get("runtime_d4_allowed")),
            "d5_aggregation_allowed": bool(d5.get("aggregation_allowed")),
            "note": "Oracle windows use gold. Do not treat gains as a trained sentence selector.",
        }
        write_json(self.output_dir / "reports" / "results_v3_d45.json", payload)
        lines = [
            "# V3 D4/D5 diagnostic autodump",
            "",
            "This file is generated. It is not the human scientific report.",
            "",
            f"- D4 oracle gain vs full chunk: `{d4.get('oracle_gain_vs_full')}`",
            f"- D4 runtime selector allowed: `{d4.get('runtime_d4_allowed')}`",
            f"- D5 questions with ≥2 independent clusters: `{d5.get('n_with_ge2_independent_clusters')}` / `{d5.get('n_questions')}`",
            f"- D5 aggregation allowed: `{d5.get('aggregation_allowed')}`",
            "",
            payload["note"],
            "",
        ]
        path = self.output_dir / "reports" / "results_v3_d45.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        self.progress.log(f"wrote {path}")
