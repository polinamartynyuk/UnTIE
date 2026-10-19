from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .contract import CELLS, META_FOLDS, OUTER_FOLDS, SEEDS, V5_RUN


@dataclass(frozen=True)
class Config:
    run_dir: Path
    v5_run: Path = V5_RUN
    datasets: tuple[str, ...] = CELLS
    seeds: tuple[int, ...] = SEEDS
    outer_folds: tuple[int, ...] = OUTER_FOLDS
    inner_meta_folds: int = META_FOLDS
    phase: str = "all"
    modes: tuple[str, ...] = ("local", "pooled", "loco")
    bootstrap: int = 5000
    device: str = "auto"
    benchmark_questions: int = 100
    resume: bool = False
    smoke: bool = False

    def validate(self) -> None:
        if not self.v5_run.joinpath("COMPLETED.json").is_file():
            raise FileNotFoundError(f"completed historical V5 run required: {self.v5_run}")
        if not self.datasets or set(self.datasets) - set(CELLS):
            raise ValueError(f"unknown datasets: {self.datasets}")
        if set(self.modes) - {"local", "pooled", "loco"}:
            raise ValueError(f"unknown modes: {self.modes}")
        if self.phase not in {"audit", "replay", "run", "analysis", "all"}:
            raise ValueError(f"unknown phase: {self.phase}")
        if not self.smoke and tuple(self.seeds) != SEEDS:
            raise ValueError(f"scientific run requires seed isolation for {SEEDS}")
        if self.inner_meta_folds < 2 or self.inner_meta_folds > 4:
            raise ValueError("inner meta folds must be 2..4")

    def payload(self) -> dict:
        return {"run_dir": str(self.run_dir.resolve()), "v5_run": str(self.v5_run.resolve()),
                "datasets": list(self.datasets), "seeds": list(self.seeds), "outer_folds": list(self.outer_folds),
                "inner_meta_folds": self.inner_meta_folds, "phase": self.phase, "modes": list(self.modes),
                "bootstrap": self.bootstrap, "device": self.device, "benchmark_questions": self.benchmark_questions, "resume": self.resume, "smoke": self.smoke}


def load_yaml(path: Path) -> dict:
    try:
        import yaml
        return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    except ImportError:
        return json.loads(Path(path).read_text(encoding="utf-8"))
