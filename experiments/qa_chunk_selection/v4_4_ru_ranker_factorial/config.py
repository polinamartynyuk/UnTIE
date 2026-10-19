from dataclasses import asdict, dataclass
from pathlib import Path

from .contract import DATASET_SPECS, K_VALUES, RANKER_SPECS, SEEDS


@dataclass(frozen=True)
class Config:
    output_dir: Path
    datasets: tuple[str, ...] = tuple(DATASET_SPECS)
    rankers: tuple[str, ...] = tuple(RANKER_SPECS)
    phase: str = "all"
    device: str = "auto"
    seeds: tuple[int, ...] = SEEDS
    k_values: tuple[int, ...] = K_VALUES
    epochs: int = 5
    patience: int = 2
    micro_batch_size: int = 2
    prediction_batch_size: int = 8
    gradient_accumulation: int = 8
    learning_rate: float = 1e-5
    max_length: int = 512
    n_bootstrap: int = 5000
    local_files_only: bool = False
    resume: bool = False
    smoke: bool = False

    def validate(self):
        if self.phase not in {"rankers", "protocol", "analysis", "all"}:
            raise ValueError("phase must be rankers, protocol, analysis, or all")
        if not self.datasets or set(self.datasets) - set(DATASET_SPECS):
            raise ValueError(f"unknown/empty datasets: {sorted(set(self.datasets) - set(DATASET_SPECS))}")
        if not self.rankers or set(self.rankers) - set(RANKER_SPECS):
            raise ValueError(f"unknown/empty rankers: {sorted(set(self.rankers) - set(RANKER_SPECS))}")
        if len(set(self.datasets)) != len(self.datasets) or len(set(self.rankers)) != len(self.rankers):
            raise ValueError("datasets and rankers must not contain duplicates")
        if len(set(self.k_values)) != len(self.k_values):
            raise ValueError("k-values must not contain duplicates")
        if tuple(self.seeds) != SEEDS:
            raise ValueError("this protocol requires seeds 42,43,44")
        if not self.k_values or any(k not in K_VALUES for k in self.k_values):
            raise ValueError("k-values must be a non-empty subset of 3,4,5")
        numeric = (self.epochs, self.patience, self.micro_batch_size,
                   self.prediction_batch_size, self.gradient_accumulation,
                   self.max_length, self.n_bootstrap)
        if any(int(value) < 1 for value in numeric) or self.learning_rate <= 0:
            raise ValueError("training and bootstrap parameters must be positive")

    def to_dict(self):
        row = asdict(self)
        row["output_dir"] = str(self.output_dir.resolve())
        row["datasets"] = list(self.datasets)
        row["rankers"] = list(self.rankers)
        row["seeds"] = list(self.seeds)
        row["k_values"] = list(self.k_values)
        row["dataset_specs"] = {
            key: {name: str(value.resolve()) if isinstance(value, Path) else value
                  for name, value in asdict(DATASET_SPECS[key]).items()}
            for key in self.datasets
        }
        row["ranker_specs"] = {key: asdict(RANKER_SPECS[key]) for key in self.rankers}
        return row

    def fingerprint_payload(self):
        row = self.to_dict()
        # These fields change execution mechanics, not scientific outcomes.
        for key in ("output_dir", "phase", "resume"):
            row.pop(key, None)
        return row

