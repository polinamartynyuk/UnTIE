from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT

from .config import Config
from .contract import CELLS, SEEDS
from .io_guard import refuse_overwrite
from .pipeline import Pipeline


def parser():
    value = argparse.ArgumentParser(description="Final official-split H2-family benchmark")
    value.add_argument("--run-dir", type=Path)
    value.add_argument("--cells", nargs="+", default=list(CELLS))
    value.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    value.add_argument("--inner-folds", type=int, default=4)
    value.add_argument("--phase", choices=["audit", "develop", "freeze", "refit", "predict-test", "evaluate", "resources", "report", "all", "smoke"], default="all")
    value.add_argument("--device", default="auto")
    value.add_argument("--resume", action="store_true")
    value.add_argument("--unlock-test", action="store_true", help="allow one-way blind official TEST prediction after freeze/refit assertions")
    value.add_argument("--allow-downloads", action="store_true")
    value.add_argument("--bootstrap", type=int, default=10_000)
    value.add_argument("--benchmark-questions", type=int, default=100)
    return value


def main(argv=None):
    args = parser().parse_args(argv); stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or PROJECT_ROOT / "artifacts/final_official_split_benchmark" / f"run_{stamp}"
    smoke = args.phase == "smoke"
    config = Config(run_dir=run_dir, cells=tuple(args.cells), seeds=tuple(args.seeds), inner_folds=args.inner_folds,
                    phase=args.phase, device=args.device, resume=args.resume, smoke=smoke, unlock_test=args.unlock_test,
                    allow_downloads=args.allow_downloads, bootstrap=args.bootstrap, benchmark_questions=args.benchmark_questions)
    refuse_overwrite(run_dir, args.resume); return Pipeline(config).run()


if __name__ == "__main__": raise SystemExit(main())
