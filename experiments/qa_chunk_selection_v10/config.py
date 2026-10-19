from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .contract import CELLS, INNER_FOLDS, OUTER_FOLDS, SEEDS, V5_RUN, V8_RUN, V9_RUN


@dataclass(frozen=True)
class Config:
    run_dir: Path
    v5_run: Path = V5_RUN
    v8_run: Path = V8_RUN
    v9_run: Path = V9_RUN
    datasets: tuple[str, ...] = CELLS
    seeds: tuple[int, ...] = SEEDS
    outer_folds: tuple[int, ...] = OUTER_FOLDS
    inner_folds: int = INNER_FOLDS
    phase: str = "all"
    bootstrap: int = 5000
    device: str = "auto"
    benchmark_questions: int = 100
    resume: bool = False
    smoke: bool = False

    def validate(self):
        for root in (self.v5_run, self.v8_run, self.v9_run):
            if not Path(root).is_dir(): raise FileNotFoundError(f"historical input missing: {root}")
        for root in (self.v8_run, self.v9_run):
            if not (Path(root) / "COMPLETED.json").is_file(): raise FileNotFoundError(f"completed historical run required: {root}")
        if not self.datasets or set(self.datasets) - set(CELLS): raise ValueError(f"unknown cells: {self.datasets}")
        if self.phase not in {"audit", "replay", "run", "analysis", "benchmark", "all"}: raise ValueError(self.phase)
        if self.inner_folds != INNER_FOLDS: raise ValueError(f"V10 requires {INNER_FOLDS} inner folds")
        if not self.smoke and self.datasets != CELLS: raise ValueError(f"scientific run requires cells {CELLS}")
        if not self.smoke and self.seeds != SEEDS: raise ValueError(f"scientific run requires seeds {SEEDS}")
        if not self.smoke and self.outer_folds != OUTER_FOLDS: raise ValueError(f"scientific run requires folds {OUTER_FOLDS}")

    def payload(self):
        value = dict(self.__dict__)
        for key in ("run_dir", "v5_run", "v8_run", "v9_run"): value[key] = str(Path(value[key]).resolve())
        for key in ("datasets", "seeds", "outer_folds"): value[key] = list(value[key])
        return value


def load_yaml(path: Path) -> dict:
    try:
        import yaml
        return dict(yaml.safe_load(Path(path).read_text()) or {})
    except ImportError:
        return json.loads(Path(path).read_text())
