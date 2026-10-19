from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from .contract import ARTIFACT_ROOT, DEFAULT_INPUT_ARTIFACT


@dataclass(frozen=True)
class Config:
    input_artifact: Path = DEFAULT_INPUT_ARTIFACT
    output_dir: Path = PROJECT_ROOT / ARTIFACT_ROOT
    device: str = "auto"
    support: bool = True
    soft: bool = False
    smoke: bool = False

    def to_dict(self):
        row = asdict(self)
        row["input_artifact"] = str(self.input_artifact.resolve())
        row["output_dir"] = str(self.output_dir.resolve())
        return row
