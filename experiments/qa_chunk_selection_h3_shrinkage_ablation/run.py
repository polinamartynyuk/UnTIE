"""Command line entry point for the isolated H3 shrinkage ablation."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from experiments.final_official_split_benchmark.io_guard import refuse_overwrite

from .models import SYSTEM_BLEND, SYSTEM_GROUP
from .pipeline import DEFAULT_CELLS, Config, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "artifacts/final_official_split_benchmark/run_20260930_175541"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Post-hoc TRAIN-OOF H3 shrinkage ablation over frozen final benchmark features."
    )
    value.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    value.add_argument("--run-dir", type=Path)
    value.add_argument("--cells", nargs="+", choices=DEFAULT_CELLS, default=list(DEFAULT_CELLS))
    value.add_argument(
        "--systems",
        nargs="+",
        choices=(SYSTEM_BLEND, SYSTEM_GROUP),
        default=[SYSTEM_BLEND, SYSTEM_GROUP],
    )
    value.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    value.add_argument("--inner-folds", type=int, default=4)
    value.add_argument("--bootstrap", type=int, default=10_000)
    value.add_argument("--resume", action="store_true")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (
        PROJECT_ROOT / "artifacts/qa_chunk_selection_h3_shrinkage_ablation" / f"run_{stamp}"
    )
    source_run = args.source_run.resolve()
    run_dir = run_dir.resolve()
    if source_run == run_dir or source_run in run_dir.parents:
        raise ValueError("run-dir must be outside the frozen source benchmark")
    refuse_overwrite(run_dir, args.resume)
    config = Config(
        source_run=source_run,
        run_dir=run_dir,
        cells=tuple(args.cells),
        systems=tuple(args.systems),
        seeds=tuple(args.seeds),
        inner_folds=args.inner_folds,
        bootstrap=args.bootstrap,
        resume=args.resume,
    )
    return Pipeline(config).run()


if __name__ == "__main__":
    raise SystemExit(main())
