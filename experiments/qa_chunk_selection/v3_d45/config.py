from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.contract import V3_ARTIFACT_ROOT

from .contract import V3_D45_ARTIFACT_ROOT

DEFAULT_STAGES = ("init", "check-gates", "d4-oracle", "d5-audit", "report")
ALL_STAGES = DEFAULT_STAGES


@dataclass(frozen=True)
class D45Config:
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V3_D45_ARTIFACT_ROOT)
    v3_artifact_dir: Path = field(default_factory=lambda: PROJECT_ROOT / V3_ARTIFACT_ROOT)
    stages: tuple[str, ...] = DEFAULT_STAGES
    device: str = "auto"
    seed: int = 42
    smoke: bool = False
    skip_qa_models: bool = False
    delta_f1_min: float = 1.0
    min_multi_cluster_rate: float = 0.20

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["v3_artifact_dir"] = str(self.v3_artifact_dir)
        payload["stages"] = list(self.stages)
        return payload


def parse_stages(raw: str) -> tuple[str, ...]:
    text = (raw or "all").strip().lower()
    if text in {"all", "default", "full"}:
        return DEFAULT_STAGES
    parts = tuple(item.strip() for item in text.split(",") if item.strip())
    unknown = [item for item in parts if item not in ALL_STAGES]
    if unknown:
        raise ValueError(f"unknown v3_d45 stages: {unknown}")
    return parts
