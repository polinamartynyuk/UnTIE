"""Multi-question Phase 1 extension. Does not rewrite preprocessing or gold."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ALLOWED_ASPECTS = ("method", "task", "metric", "material")
ALLOWED_STATUS = ("ready", "partially_extractable")
ENRICHMENT_MODE = "E0"
MODES = ("Q0", "Q3", "Q5")
PROTECTED_INPUT = "datasets/scirex_aspect_qa_preprocessing"
# DEV candidates: robust_positive iff QAConsistency >= threshold.
Q3_THRESHOLD_CANDIDATES = (2 / 3, 1.0)
Q5_THRESHOLD_CANDIDATES = (3 / 5, 4 / 5, 1.0)


@dataclass(frozen=True)
class MultiQuestionConfig:
    input_dir: Path
    output_dir: Path
    split: str = "train"
    mode: str = "Q3"
    n_questions: int = 3
    debug: bool = False
    debug_docs: int = 8
    seed: int = 42
    device: str = "auto"
    hard_negative_top_n: int = 5
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    gap_bootstrap: int = 10
    k_max_cap: int = 10
    min_cluster_points: int = 3
    gumbel_ablation: bool = False
    gumbel_temperature: float = 0.5
    gumbel_runs: int = 20
    enrichment_mode: str = ENRICHMENT_MODE
    positive_match_method: str = "gold_span_containment"
    containment_max_extra_tokens: int = 5
    robust_positive_threshold: float | None = None
    bm25_aggregate: str = "mean_normalized"
    canonical_predictions_path: Path | None = None
    prediction_cache_path: Path | None = None
    threshold_file: Path | None = None
    enable_score_aspect: bool = False
    document_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        expected = {"Q0": 1, "Q3": 3, "Q5": 5}[self.mode]
        if int(self.n_questions) != expected:
            raise ValueError(f"{self.mode} requires n_questions={expected}")

    def threshold_candidates(self) -> tuple[float, ...]:
        if self.mode == "Q0":
            return (1.0,)
        if self.mode == "Q3":
            return Q3_THRESHOLD_CANDIDATES
        return Q5_THRESHOLD_CANDIDATES

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "input_dir",
            "output_dir",
            "canonical_predictions_path",
            "prediction_cache_path",
            "threshold_file",
        ):
            value = payload[key]
            payload[key] = str(value) if value is not None else None
        payload["document_ids"] = list(self.document_ids)
        payload["aspects"] = list(ALLOWED_ASPECTS)
        payload["allowed_status"] = list(ALLOWED_STATUS)
        payload["fn_filter"] = "kmeans_gap_hard_assignment"
        payload["hard_negative_method"] = "bm25_multi_question_then_fn_filter"
        payload["primary_positive_rule"] = "robust_positive_threshold_on_QAConsistency"
        payload["not_primary_positive_rule"] = "any_question_correct"
        payload["temperature_decay"] = False
        payload["gumbel_role"] = "optional_ablation_no_decay"
        return payload
