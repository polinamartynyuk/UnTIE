from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from .contract import CELLS, INNER_FOLDS, OUTER_FOLDS, SEEDS, V4_4_RUN, V4_4_EXTERNAL_RUN, V4_5_RUN, V5_RUN, V6_RUN, V7_RUN

@dataclass(frozen=True)
class Config:
    run_dir: Path
    v4_4_run: Path = V4_4_RUN
    v4_4_external_run: Path = V4_4_EXTERNAL_RUN
    v4_5_run: Path = V4_5_RUN
    v5_run: Path = V5_RUN
    v6_run: Path = V6_RUN
    v7_run: Path = V7_RUN
    datasets: tuple[str, ...] = CELLS
    seeds: tuple[int, ...] = SEEDS
    outer_folds: tuple[int, ...] = OUTER_FOLDS
    inner_folds: int = INNER_FOLDS
    modes: tuple[str, ...] = ("local", "pooled", "loco")
    phase: str = "all"
    bootstrap: int = 5000
    device: str = "auto"
    benchmark_questions: int = 100
    resume: bool = False
    smoke: bool = False

    def validate(self) -> None:
        required = (self.v4_4_run, self.v4_4_external_run, self.v4_5_run, self.v5_run, self.v6_run, self.v7_run)
        missing = [str(p) for p in required if not p.is_dir()]
        if missing: raise FileNotFoundError(f"historical inputs missing: {missing}")
        if not self.v5_run.joinpath("COMPLETED.json").is_file(): raise FileNotFoundError(f"completed V5 required: {self.v5_run}")
        if not self.v6_run.joinpath("COMPLETED.json").is_file(): raise FileNotFoundError(f"completed V6 required: {self.v6_run}")
        if not self.v7_run.joinpath("COMPLETED.json").is_file(): raise FileNotFoundError(f"completed V7 required: {self.v7_run}")
        if not self.datasets or set(self.datasets)-set(CELLS): raise ValueError(f"unknown cells: {self.datasets}")
        if set(self.modes)-{"local","pooled","loco"}: raise ValueError(f"unknown modes: {self.modes}")
        if self.phase not in {"audit","preservation","replay","run","analysis","benchmark","all"}: raise ValueError(self.phase)
        if not self.smoke and self.seeds != SEEDS: raise ValueError(f"scientific run requires seeds {SEEDS}")
        if not self.smoke and self.outer_folds != OUTER_FOLDS: raise ValueError("scientific run requires five outer folds")
        if not 2 <= self.inner_folds <= 4: raise ValueError("inner folds must be 2..4")

    def payload(self) -> dict:
        value = dict(self.__dict__)
        for key in ("run_dir","v4_4_run","v4_4_external_run","v4_5_run","v5_run","v6_run","v7_run"): value[key]=str(Path(value[key]).resolve())
        for key in ("datasets","seeds","outer_folds","modes"): value[key]=list(value[key])
        return value

def load_yaml(path: Path) -> dict:
    try:
        import yaml
        return dict(yaml.safe_load(Path(path).read_text()) or {})
    except ImportError:
        return json.loads(Path(path).read_text())
