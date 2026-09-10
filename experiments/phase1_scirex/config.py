"""Phase 1 configuration. Labels do not use attention, diff, or KeyBERT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ALLOWED_ASPECTS = ("method", "task", "metric", "material")
ALLOWED_STATUS = ("ready", "partially_extractable")
ENRICHMENT_MODE = "E0"
POSITIVE_MATCH_METHOD = "gold_span_containment"
CONTAINMENT_MAX_EXTRA_TOKENS = 5


@dataclass(frozen=True)
class Phase1Config:
    input_dir: Path
    output_dir: Path
    split: str = "train"
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
    positive_match_method: str = POSITIVE_MATCH_METHOD
    containment_max_extra_tokens: int = CONTAINMENT_MAX_EXTRA_TOKENS
    enable_score_aspect: bool = False
    document_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_dir"] = str(self.input_dir)
        payload["output_dir"] = str(self.output_dir)
        payload["document_ids"] = list(self.document_ids)
        payload["aspects"] = list(ALLOWED_ASPECTS)
        payload["allowed_status"] = list(ALLOWED_STATUS)
        payload["fn_filter"] = "kmeans_gap_hard_assignment"
        payload["hard_negative_method"] = "bm25_question_chunk"
        payload["gumbel_role"] = "optional_ablation_no_decay"
        payload["temperature_decay"] = False
        return payload
