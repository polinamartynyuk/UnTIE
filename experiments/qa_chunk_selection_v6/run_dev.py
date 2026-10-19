from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT

from .config import Config, load_yaml
from .contract import CELLS, OUTER_FOLDS, SEEDS, V5_RUN
from .io_guard import canonical_hash, refuse_overwrite, write_json
from .pipeline import Pipeline


def parser():
    p = argparse.ArgumentParser(description="V6/UAR-G universal development experiment")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/qa_chunk_selection/v6_universal/main.yaml")
    p.add_argument("--datasets", nargs="+", default=list(CELLS)); p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--outer-folds", type=int, default=5); p.add_argument("--inner-meta-folds", type=int, default=4)
    p.add_argument("--run-dir", type=Path); p.add_argument("--v5-run", type=Path, default=V5_RUN)
    p.add_argument("--phase", choices=("audit", "replay", "run", "analysis", "all"), default="all")
    p.add_argument("--modes", nargs="+", default=["local", "pooled", "loco"]); p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--device", default="auto"); p.add_argument("--benchmark-questions", type=int, default=100)
    p.add_argument("--resume", action="store_true"); p.add_argument("--smoke", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv); defaults = load_yaml(args.config) if args.config.is_file() else {}
    run_dir = args.run_dir or PROJECT_ROOT / "artifacts/qa_chunk_selection_v6_universal" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = refuse_overwrite(run_dir, resume=args.resume)
    folds = (0,) if args.smoke else tuple(range(args.outer_folds)); seeds = (args.seeds[0],) if args.smoke else tuple(args.seeds)
    datasets = (args.datasets[0],) if args.smoke else tuple(args.datasets); modes = (args.modes[0],) if args.smoke else tuple(args.modes)
    config = Config(run_dir=run_dir, v5_run=args.v5_run, datasets=datasets, seeds=seeds, outer_folds=folds,
                    inner_meta_folds=args.inner_meta_folds, phase=args.phase, modes=modes,
                    bootstrap=args.bootstrap, device=args.device, benchmark_questions=args.benchmark_questions,
                    resume=args.resume, smoke=args.smoke)
    config.validate(); payload = config.payload(); payload["fingerprint"] = canonical_hash({k: v for k, v in payload.items() if k not in {"run_dir", "phase", "resume"}})
    resolved = run_dir / "config_resolved.json"
    if resolved.is_file():
        previous = json.loads(resolved.read_text())
        if previous.get("fingerprint") != payload["fingerprint"]: raise RuntimeError("resume fingerprint mismatch")
    else: write_json(resolved, payload)
    return Pipeline(config).run()


if __name__ == "__main__": raise SystemExit(main())

