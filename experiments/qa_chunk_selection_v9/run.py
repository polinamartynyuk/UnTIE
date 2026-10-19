from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from .config import Config, load_yaml
from .contract import CELLS, OUTER_FOLDS, SEEDS
from .io_guard import refuse_overwrite
from .pipeline import Pipeline

ALIASES = {"scier_384_50": "scier_384", "scier_128_24": "scier_128"}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="V9 / H2 Adaptive Component Selection development experiment")
    value.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/qa_chunk_selection/v9_h2_acs/main.yaml")
    value.add_argument("--run-dir", type=Path)
    value.add_argument("--cells", "--datasets", nargs="+", default=list(CELLS))
    value.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    value.add_argument("--outer-folds", nargs="+", type=int)
    value.add_argument("--inner-folds", type=int, default=4)
    value.add_argument("--select-folds", type=int, default=4)
    value.add_argument("--phase", choices=["audit", "replay", "run", "analysis", "benchmark", "all"], default="all")
    value.add_argument("--bootstrap", type=int, default=5000)
    value.add_argument("--device", default="auto")
    value.add_argument("--benchmark-questions", type=int, default=100)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--smoke", action="store_true")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    payload = load_yaml(args.config) if args.config.is_file() else {}
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.run_dir or PROJECT_ROOT / f"artifacts/qa_chunk_selection_v9_h2_acs/run_{stamp}"
    cells = tuple(ALIASES.get(x, x) for x in args.cells)
    folds = tuple(args.outer_folds if args.outer_folds is not None else ([0] if args.smoke else OUTER_FOLDS))
    seeds = tuple(args.seeds)
    if args.smoke:
        requested = [x for x in cells if x in {"scirex", "russerrc"}]
        cells = tuple(requested[:2] or cells[:1])
        folds = folds[:1]
        seeds = seeds[:1]

    def historical(name: str, default: str) -> Path:
        return PROJECT_ROOT / payload.get(name, default)

    config = Config(
        run_dir=Path(output),
        v4_4_run=historical("historical_v4_4", "artifacts/qa_chunk_selection_v4_4"),
        v4_4_external_run=historical("historical_v4_4_external", "artifacts/qa_chunk_selection_v4_4_external_20260913_213335"),
        v4_5_run=historical("historical_v4_5", "artifacts/qa_chunk_selection_v4_5_ru/run_20260915_155234"),
        v5_run=historical("historical_v5", "artifacts/qa_chunk_selection_v5/run_20260916_135455"),
        v6_run=historical("historical_v6", "artifacts/qa_chunk_selection_v6_universal/run_20260922_132020"),
        v7_run=historical("historical_v7", "artifacts/qa_chunk_selection_v7_rapr/run_20260922_181837"),
        v8_run=historical("historical_v8", "artifacts/qa_chunk_selection_v8_dual/run_20260923_133853"),
        datasets=cells,
        seeds=seeds,
        outer_folds=folds,
        inner_folds=args.inner_folds,
        select_folds=args.select_folds,
        phase=args.phase,
        bootstrap=args.bootstrap,
        device=args.device,
        benchmark_questions=args.benchmark_questions,
        resume=args.resume,
        smoke=args.smoke,
    )
    refuse_overwrite(config.run_dir, resume=config.resume)
    return Pipeline(config).run()


if __name__ == "__main__":
    raise SystemExit(main())

