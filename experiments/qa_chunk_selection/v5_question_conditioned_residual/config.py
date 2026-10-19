from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .contract import CELLS, SEEDS


@dataclass(frozen=True)
class Config:
    output_dir: Path
    cells: tuple[str, ...] = tuple(CELLS)
    phase: str = "all"
    device: str = "auto"
    seeds: tuple[int, ...] = SEEDS
    n_bootstrap: int = 5000
    benchmark_questions: int = 20
    local_files_only: bool = False
    resume: bool = False
    smoke: bool = False

    def validate(self):
        if self.phase not in {"audit", "run", "benchmark", "analysis", "all"}:
            raise ValueError("phase must be audit, run, benchmark, analysis or all")
        if not self.cells or set(self.cells) - set(CELLS):
            raise ValueError(f"unknown/empty cells: {sorted(set(self.cells)-set(CELLS))}")
        expected = (42,) if self.smoke else SEEDS
        if tuple(self.seeds) != expected:
            raise ValueError(f"seeds must be {expected}")
        if self.n_bootstrap < 1 or self.benchmark_questions < 1:
            raise ValueError("bootstrap and benchmark counts must be positive")

    def to_dict(self):
        value=asdict(self)
        value["output_dir"]=str(self.output_dir.resolve())
        value["cells"]=list(self.cells); value["seeds"]=list(self.seeds)
        return value

    def fingerprint_payload(self):
        value=self.to_dict()
        for key in ("output_dir","phase","resume"):value.pop(key,None)
        return value

