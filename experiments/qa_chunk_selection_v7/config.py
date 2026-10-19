from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from .contract import CELLS, INNER_FOLDS, OUTER_FOLDS, SEEDS, V5_RUN, V6_RUN

@dataclass(frozen=True)
class Config:
    run_dir: Path
    v5_run: Path = V5_RUN
    v6_run: Path = V6_RUN
    datasets: tuple[str,...] = CELLS
    seeds: tuple[int,...] = SEEDS
    outer_folds: tuple[int,...] = OUTER_FOLDS
    inner_folds: int = INNER_FOLDS
    modes: tuple[str,...] = ("local", "pooled", "loco")
    phase: str = "all"
    bootstrap: int = 5000
    device: str = "auto"
    benchmark_questions: int = 100
    resume: bool = False
    smoke: bool = False

    def validate(self):
        if not self.v5_run.joinpath("COMPLETED.json").is_file(): raise FileNotFoundError(f"completed V5 required: {self.v5_run}")
        if not self.v6_run.joinpath("COMPLETED.json").is_file(): raise FileNotFoundError(f"completed V6 required: {self.v6_run}")
        if not self.datasets or set(self.datasets)-set(CELLS): raise ValueError(f"unknown cells: {self.datasets}")
        if set(self.modes)-{"local","pooled","loco"}: raise ValueError(f"unknown modes: {self.modes}")
        if self.phase not in {"audit","replay","run","analysis","benchmark","all"}: raise ValueError(f"unknown phase: {self.phase}")
        if not self.smoke and tuple(self.seeds)!=SEEDS: raise ValueError(f"scientific run requires seeds {SEEDS}")
        if not self.smoke and tuple(self.outer_folds)!=OUTER_FOLDS: raise ValueError("scientific run requires all five outer folds")
        if not 2 <= self.inner_folds <= 4: raise ValueError("inner folds must be 2..4")

    def payload(self):
        return {"run_dir":str(self.run_dir.resolve()),"v5_run":str(self.v5_run.resolve()),"v6_run":str(self.v6_run.resolve()),
                "datasets":list(self.datasets),"seeds":list(self.seeds),"outer_folds":list(self.outer_folds),"inner_folds":self.inner_folds,
                "modes":list(self.modes),"phase":self.phase,"bootstrap":self.bootstrap,"device":self.device,
                "benchmark_questions":self.benchmark_questions,"resume":self.resume,"smoke":self.smoke}

def load_yaml(path: Path) -> dict:
    try:
        import yaml
        return dict(yaml.safe_load(Path(path).read_text()) or {})
    except ImportError:
        return json.loads(Path(path).read_text())
