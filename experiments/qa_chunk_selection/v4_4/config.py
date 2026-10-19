from dataclasses import dataclass, asdict
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from .contract import ARTIFACT_ROOT


@dataclass(frozen=True)
class ExperimentConfigV44:
    output_dir: Path = PROJECT_ROOT / ARTIFACT_ROOT
    device: str = "auto"
    smoke: bool = False
    support: bool = True
    soft: bool = False

    @property
    def primary_k(self):
        return 5

    def to_dict(self):
        return {**asdict(self), "output_dir": str(self.output_dir)}
