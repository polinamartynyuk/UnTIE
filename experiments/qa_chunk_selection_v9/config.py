from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .contract import (
    CELLS,
    INNER_FOLDS,
    OUTER_FOLDS,
    SEEDS,
    SELECT_FOLDS,
    V4_4_EXTERNAL_RUN,
    V4_4_RUN,
    V4_5_RUN,
    V5_RUN,
    V6_RUN,
    V7_RUN,
    V8_RUN,
)


@dataclass(frozen=True)
class Config:
    run_dir: Path
    v4_4_run: Path = V4_4_RUN
    v4_4_external_run: Path = V4_4_EXTERNAL_RUN
    v4_5_run: Path = V4_5_RUN
    v5_run: Path = V5_RUN
    v6_run: Path = V6_RUN
    v7_run: Path = V7_RUN
    v8_run: Path = V8_RUN
    datasets: tuple[str, ...] = CELLS
    seeds: tuple[int, ...] = SEEDS
    outer_folds: tuple[int, ...] = OUTER_FOLDS
    inner_folds: int = INNER_FOLDS
    select_folds: int = SELECT_FOLDS
    phase: str = "all"
    bootstrap: int = 5000
    device: str = "auto"
    benchmark_questions: int = 100
    resume: bool = False
    smoke: bool = False

    def validate(self) -> None:
        roots = (
            self.v4_4_run,
            self.v4_4_external_run,
            self.v4_5_run,
            self.v5_run,
            self.v6_run,
            self.v7_run,
            self.v8_run,
        )
        missing = [str(x) for x in roots if not Path(x).is_dir()]
        if missing:
            raise FileNotFoundError(f"historical inputs missing: {missing}")
        for root in (self.v5_run, self.v6_run, self.v7_run, self.v8_run):
            if not Path(root, "COMPLETED.json").is_file():
                raise FileNotFoundError(f"completed historical run required: {root}")
        if not self.datasets or set(self.datasets) - set(CELLS):
            raise ValueError(f"unknown cells: {self.datasets}")
        if self.phase not in {"audit", "replay", "run", "analysis", "benchmark", "all"}:
            raise ValueError(self.phase)
        if self.select_folds != SELECT_FOLDS:
            raise ValueError(f"V9 requires SELECT_FOLDS={SELECT_FOLDS}")
        if self.inner_folds != INNER_FOLDS:
            raise ValueError(f"V9 requires exact H2 INNER_FOLDS={INNER_FOLDS}")
        if not self.smoke and self.seeds != SEEDS:
            raise ValueError(f"scientific run requires seeds {SEEDS}")
        if not self.smoke and self.outer_folds != OUTER_FOLDS:
            raise ValueError("scientific run requires all five outer folds")
        if not self.smoke and self.datasets != CELLS:
            raise ValueError(f"scientific run requires all cells in frozen order: {CELLS}")

    def payload(self) -> dict:
        value = dict(self.__dict__)
        for key in (
            "run_dir", "v4_4_run", "v4_4_external_run", "v4_5_run",
            "v5_run", "v6_run", "v7_run", "v8_run",
        ):
            value[key] = str(Path(value[key]).resolve())
        for key in ("datasets", "seeds", "outer_folds"):
            value[key] = list(value[key])
        return value


def load_yaml(path: Path) -> dict:
    try:
        import yaml
        return dict(yaml.safe_load(Path(path).read_text()) or {})
    except ImportError:
        return json.loads(Path(path).read_text())

