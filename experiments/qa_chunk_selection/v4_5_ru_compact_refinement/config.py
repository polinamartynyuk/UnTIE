from dataclasses import asdict, dataclass
from pathlib import Path

from .contract import DATASETS, SEEDS, TRAINING_RECIPE


@dataclass(frozen=True)
class Config:
    output_dir: Path
    datasets: tuple[str, ...] = tuple(DATASETS)
    phase: str = "all"
    device: str = "auto"
    seeds: tuple[int, ...] = SEEDS
    epochs: int = int(TRAINING_RECIPE["max_epochs"])
    patience: int = int(TRAINING_RECIPE["early_stopping_patience"])
    micro_batch_size: int = int(TRAINING_RECIPE["micro_batch_size"])
    prediction_batch_size: int = 8
    gradient_accumulation: int = int(TRAINING_RECIPE["gradient_accumulation"])
    learning_rate: float = float(TRAINING_RECIPE["learning_rate"])
    max_length: int = int(TRAINING_RECIPE["max_length"])
    n_bootstrap: int = 5000
    benchmark_questions: int = 20
    benchmark_warmup: int = 10
    benchmark_repetitions: int = 10
    dity_throughput_batch: int = 16
    bge_throughput_batch: int = 8
    local_files_only: bool = False
    resume: bool = False
    smoke: bool = False

    def validate(self):
        phases = {"audit", "rankers", "stage2", "benchmark", "analysis", "all"}
        if self.phase not in phases:
            raise ValueError(f"phase must be one of {sorted(phases)}")
        if not self.datasets or set(self.datasets) - set(DATASETS):
            raise ValueError("datasets must be a non-empty subset of russerrc,scimdix")
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("datasets must not contain duplicates")
        if (not self.smoke and tuple(self.seeds) != SEEDS) or (self.smoke and tuple(self.seeds) != (42,)):
            raise ValueError("V4.5 requires seeds 42,43,44; smoke requires seed 42 only")
        positive = (self.epochs, self.patience, self.micro_batch_size,
                    self.prediction_batch_size, self.gradient_accumulation,
                    self.max_length, self.n_bootstrap, self.benchmark_questions,
                    self.benchmark_warmup, self.benchmark_repetitions,
                    self.dity_throughput_batch, self.bge_throughput_batch)
        if any(int(value) < 1 for value in positive) or self.learning_rate <= 0:
            raise ValueError("training, bootstrap and benchmark values must be positive")
        if not self.smoke:
            frozen = TRAINING_RECIPE
            actual = (self.epochs, self.patience, self.micro_batch_size,
                      self.gradient_accumulation, self.learning_rate, self.max_length)
            expected = (frozen["max_epochs"], frozen["early_stopping_patience"],
                        frozen["micro_batch_size"], frozen["gradient_accumulation"],
                        frozen["learning_rate"], frozen["max_length"])
            if actual != expected:
                raise ValueError("V4.5 scientific run requires the frozen historical training recipe")

    def to_dict(self):
        row = asdict(self)
        row["output_dir"] = str(self.output_dir.resolve())
        row["datasets"] = list(self.datasets)
        row["seeds"] = list(self.seeds)
        row["dataset_k"] = {key: DATASETS[key].k for key in self.datasets}
        return row

    def fingerprint_payload(self):
        row = self.to_dict()
        for key in ("output_dir", "phase", "resume"):
            row.pop(key, None)
        return row

