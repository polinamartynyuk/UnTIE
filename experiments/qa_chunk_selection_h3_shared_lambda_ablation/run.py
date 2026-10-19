"""CLI for the dataset-level shared-lambda ablation."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from experiments.final_official_split_benchmark.io_guard import refuse_overwrite
from experiments.qa_chunk_selection_h3_shrinkage_ablation.pipeline import DEFAULT_CELLS

from .pipeline import Config, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "artifacts/final_official_split_benchmark/run_20260930_175541"
DEFAULT_PREVIOUS = PROJECT_ROOT / "artifacts/qa_chunk_selection_h3_shrinkage_ablation/run_20261007_171427"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Choose one H2/H3 blend lambda per dataset across seeds.")
    value.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    value.add_argument("--previous-run", type=Path, default=DEFAULT_PREVIOUS)
    value.add_argument("--run-dir", type=Path)
    value.add_argument("--cells", nargs="+", choices=DEFAULT_CELLS, default=list(DEFAULT_CELLS))
    value.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    value.add_argument("--inner-folds", type=int, default=4)
    value.add_argument("--bootstrap", type=int, default=10_000)
    value.add_argument("--resume", action="store_true")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (
        PROJECT_ROOT / "artifacts/qa_chunk_selection_h3_shared_lambda_ablation" / f"run_{stamp}"
    )
    run_dir = run_dir.resolve()
    refuse_overwrite(run_dir, args.resume)
    config = Config(
        source_run=args.source_run.resolve(),
        previous_run=args.previous_run.resolve(),
        run_dir=run_dir,
        cells=tuple(args.cells),
        seeds=tuple(args.seeds),
        inner_folds=args.inner_folds,
        bootstrap=args.bootstrap,
        resume=args.resume,
    )
    return Pipeline(config).run()


if __name__ == "__main__":
    raise SystemExit(main())
