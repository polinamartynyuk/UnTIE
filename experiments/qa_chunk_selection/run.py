"""CLI for isolated SciREX chunk-selection experiments A–D."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import (
    DEFAULT_CE_MODEL,
    DEFAULT_DEV_LABELS,
    DEFAULT_DEV_PREPROCESS,
    DEFAULT_OUTPUT,
    DEFAULT_TRAIN_LABELS,
    DEFAULT_TRAIN_PREPROCESS,
    ExperimentConfig,
    parse_stages,
)
from .io_util import PROJECT_ROOT
from .pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "In-document chunk selection experiments A–D on frozen SciREX Phase 1 Q5 labels. "
            "Does not write into preprocessing or Phase 1 artifact trees."
        )
    )
    parser.add_argument("--train-label-dir", type=Path, default=DEFAULT_TRAIN_LABELS)
    parser.add_argument("--dev-label-dir", type=Path, default=DEFAULT_DEV_LABELS)
    parser.add_argument("--train-preprocess-dir", type=Path, default=DEFAULT_TRAIN_PREPROCESS)
    parser.add_argument("--dev-preprocess-dir", type=Path, default=DEFAULT_DEV_PREPROCESS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-mode", choices=("all", "controlled"), default="all")
    parser.add_argument("--n-easy-negatives", type=int, default=5)
    parser.add_argument("--stages", default="all", help="comma-separated stages or 'all' / 'full'")
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CE_MODEL)
    parser.add_argument("--b-loss", choices=("binary", "pairwise"), default="binary")
    parser.add_argument("--b-epochs", type=int, default=3)
    parser.add_argument("--b-batch-size", type=int, default=16)
    parser.add_argument("--passage-m", type=int, default=5)
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true", default=True)
    parser.add_argument("--include-bertscore", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU run with lexical/TF-IDF stubs")
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--debug-docs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-unanswerable", action="store_true")
    parser.add_argument("--allow-test", action="store_true", help="refused: TEST is not labeled yet")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    skip_bertscore = not bool(args.include_bertscore)
    return ExperimentConfig(
        train_label_dir=Path(args.train_label_dir),
        dev_label_dir=Path(args.dev_label_dir),
        train_preprocess_dir=Path(args.train_preprocess_dir),
        dev_preprocess_dir=Path(args.dev_preprocess_dir),
        output_dir=Path(args.output_dir),
        seed=int(args.seed),
        device=str(args.device),
        candidate_mode=str(args.candidate_mode),
        n_easy_negatives=int(args.n_easy_negatives),
        default_passage_m=int(args.passage_m),
        b_loss=str(args.b_loss),
        b_epochs=int(args.b_epochs),
        b_batch_size=int(args.b_batch_size),
        skip_bertscore=skip_bertscore,
        skip_qa=bool(args.skip_qa),
        smoke=bool(args.smoke),
        debug_docs=int(args.debug_docs),
        limit_docs=int(args.limit_docs),
        allow_test=bool(args.allow_test),
        resume=bool(args.resume),
        include_unanswerable=bool(args.include_unanswerable),
        cross_encoder_model=str(args.cross_encoder_model),
        stages=parse_stages(args.stages),
        seeds=(int(args.seed),),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.allow_test:
        print("TEST is reserved until the feature rule is frozen. Refusing.")
        return 2
    config = config_from_args(args)
    print(f"project_root={PROJECT_ROOT}", flush=True)
    print(f"output_dir={config.output_dir}", flush=True)
    print(f"stages={','.join(config.stages)}", flush=True)
    print(f"smoke={config.smoke} device={config.device}", flush=True)
    return Pipeline(config).run()
