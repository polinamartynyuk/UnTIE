from dataclasses import asdict, dataclass
from pathlib import Path

from .contract import DEFAULT_INPUT, K_VALUES, MODEL_SPECS, SEEDS


@dataclass(frozen=True)
class Config:
    input_artifact: Path = DEFAULT_INPUT
    output_dir: Path = Path()
    phase: str = "all"
    device: str = "auto"
    models: tuple[tuple[str, str], ...] = MODEL_SPECS
    seeds: tuple[int, ...] = SEEDS
    k_values: tuple[int, ...] = K_VALUES
    epochs: int = 5
    patience: int = 2
    micro_batch_size: int = 2
    gradient_accumulation: int = 8
    learning_rate: float = 1e-5
    max_length: int = 512
    resume: bool = False
    smoke: bool = False

    def validate(self):
        if self.phase not in {"protocol", "cross-encoders", "all"}:
            raise ValueError("phase must be protocol, cross-encoders, or all")
        if not self.k_values or any(k not in {3, 4, 5} for k in self.k_values):
            raise ValueError("k-values must be a non-empty subset of 3,4,5")
        if tuple(self.seeds) != SEEDS:
            raise ValueError("this protocol requires seeds 42,43,44")
        if self.micro_batch_size < 1 or self.gradient_accumulation < 1:
            raise ValueError("batch sizes must be positive")

    def to_dict(self):
        row = asdict(self)
        row["input_artifact"] = str(self.input_artifact.resolve())
        row["output_dir"] = str(self.output_dir.resolve())
        row["models"] = [{"id": key, "name": name} for key, name in self.models]
        row["seeds"] = list(self.seeds)
        row["k_values"] = list(self.k_values)
        return row

