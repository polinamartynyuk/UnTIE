"""CLI for the isolated consistency-only gate."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from experiments.final_official_split_benchmark.io_guard import refuse_overwrite
from experiments.qa_chunk_selection_h3_shrinkage_ablation.pipeline import DEFAULT_CELLS

from .pipeline import Config, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "artifacts/final_official_split_benchmark/run_20260930_175541"
DEFAULT_SHARED = PROJECT_ROOT / "artifacts/qa_chunk_selection_h3_shared_lambda_ablation/run_20261007_01"


def parser():
    value = argparse.ArgumentParser(description="Replay the exact TRAIN/refit consistency-only lambda gate.")
    value.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    value.add_argument("--shared-run", type=Path, default=DEFAULT_SHARED)
    value.add_argument("--run-dir", type=Path)
    value.add_argument("--cells", nargs="+", choices=DEFAULT_CELLS, default=list(DEFAULT_CELLS))
    value.add_argument("--bootstrap", type=int, default=10_000)
    value.add_argument("--resume", action="store_true")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (
        PROJECT_ROOT / "artifacts/qa_chunk_selection_h3_consistency_gate" / f"run_{stamp}"
    )
    run_dir = run_dir.resolve()
    refuse_overwrite(run_dir, args.resume)
    return Pipeline(
        Config(
            source_run=args.source_run.resolve(),
            shared_run=args.shared_run.resolve(),
            run_dir=run_dir,
            cells=tuple(args.cells),
            bootstrap=args.bootstrap,
            resume=args.resume,
        )
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
