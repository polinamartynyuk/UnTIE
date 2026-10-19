from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    base_artifact: Path
    output_dir: Path
    device: str = "auto"
    seeds: tuple[int, ...] = (42, 43, 44)
    n_folds: int = 5
    split_seed: int = 42
    b0_model: str = "DeepPavlov/rubert-base-cased-sentence"
    epochs: int = 5
    patience: int = 2
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 512
    smoke: bool = False
    resume: bool = False

    def validate(self) -> None:
        if len(self.seeds) != 3 or len(set(self.seeds)) != 3:
            raise ValueError("B0 OOF requires exactly three distinct seeds")
        required = (
            self.base_artifact / "COMPLETED.json",
            self.base_artifact / "data" / "dataset.jsonl",
            self.base_artifact / "preprocessing" / "aspect_instances.parquet",
            self.base_artifact / "preprocessing" / "chunks.parquet",
            self.base_artifact / "preprocessing" / "gold_mentions.parquet",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"incomplete base artifact: {missing}")

    def to_dict(self):
        row = asdict(self)
        row["base_artifact"] = str(self.base_artifact.resolve())
        row["output_dir"] = str(self.output_dir.resolve())
        row["seeds"] = list(self.seeds)
        return row

