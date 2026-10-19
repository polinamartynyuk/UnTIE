from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .contract import CELLS, INNER_FOLDS, SEEDS


@dataclass(frozen=True)
class Config:
    run_dir: Path
    cells: tuple[str, ...] = CELLS
    seeds: tuple[int, ...] = SEEDS
    inner_folds: int = INNER_FOLDS
    phase: str = "all"
    device: str = "auto"
    resume: bool = False
    smoke: bool = False
    unlock_test: bool = False
    allow_downloads: bool = False
    bootstrap: int = 10_000
    benchmark_questions: int = 100

    def validate(self) -> None:
        allowed = {"audit", "develop", "freeze", "refit", "predict-test", "evaluate", "resources", "report", "all", "smoke"}
        if self.phase not in allowed:
            raise ValueError(f"unknown phase: {self.phase}")
        unknown = set(self.cells) - set(CELLS)
        if unknown:
            raise ValueError(f"unknown cells: {sorted(unknown)}")
        if self.inner_folds not in {3, 4}:
            raise ValueError("inner folds must be 3 or 4")
        if not self.seeds:
            raise ValueError("at least one seed required")
        if self.smoke and self.unlock_test:
            raise ValueError("smoke mode can never unlock TEST")

    def payload(self) -> dict:
        result = asdict(self)
        result["run_dir"] = str(self.run_dir.resolve())
        result["cells"] = list(self.cells)
        result["seeds"] = list(self.seeds)
        return result
