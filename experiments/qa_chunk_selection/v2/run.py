"""CLI for isolated SciREX chunk-selection V2. Does not write E0 artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from experiments.qa_chunk_selection.config import DEFAULT_CE_MODEL, DEFAULT_DEV_LABELS, DEFAULT_DEV_PREPROCESS, DEFAULT_TRAIN_LABELS, DEFAULT_TRAIN_PREPROCESS

from .config import ExperimentConfigV2, parse_stages
from .io_util import DEFAULT_V2_OUTPUT, E0_OUTPUT, PROJECT_ROOT
from .pipeline import PipelineV2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "V2 in-document chunk selection experiments. "
            "Writes only to artifacts/qa_chunk_selection_v2/. Never overwrites E0."
        )
    )
    parser.add_argument("--train-label-dir", type=Path, default=DEFAULT_TRAIN_LABELS)
    parser.add_argument("--dev-label-dir", type=Path, default=DEFAULT_DEV_LABELS)
    parser.add_argument("--train-preprocess-dir", type=Path, default=DEFAULT_TRAIN_PREPROCESS)
    parser.add_argument("--dev-preprocess-dir", type=Path, default=DEFAULT_DEV_PREPROCESS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_V2_OUTPUT)
    parser.add_argument("--e0-artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "qa_chunk_selection")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--stages", default="all", help="comma-separated stages, 'all' (default V2), or 'full'")
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CE_MODEL)
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--debug-docs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-internal-test", action="store_true")
    parser.add_argument("--skip-e0-replay", action="store_true")
    parser.add_argument("--include-hn-gold-warning", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfigV2:
    return ExperimentConfigV2(
        train_label_dir=Path(args.train_label_dir),
        dev_label_dir=Path(args.dev_label_dir),
        train_preprocess_dir=Path(args.train_preprocess_dir),
        dev_preprocess_dir=Path(args.dev_preprocess_dir),
        output_dir=Path(args.output_dir),
        e0_artifact_dir=Path(args.e0_artifact_dir),
        seed=int(args.seed),
        split_seed=int(args.split_seed),
        device=str(args.device),
        skip_qa=bool(args.skip_qa),
        smoke=bool(args.smoke),
        debug_docs=int(args.debug_docs),
        limit_docs=int(args.limit_docs),
        allow_internal_test=bool(args.allow_internal_test),
        resume=bool(args.resume),
        include_hn_gold_warning=bool(args.include_hn_gold_warning),
        replay_e0_inference=not bool(args.skip_e0_replay),
        cross_encoder_model=str(args.cross_encoder_model),
        stages=parse_stages(args.stages),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    if Path(config.output_dir).resolve() == E0_OUTPUT.resolve():
        print("Refusing to write V2 into E0 artifacts/qa_chunk_selection/. Use artifacts/qa_chunk_selection_v2/.")
        return 2
    print(f"project_root={PROJECT_ROOT}", flush=True)
    print(f"output_dir={config.output_dir}", flush=True)
    print(f"e0_artifacts={config.e0_artifact_dir} (read-only)", flush=True)
    print(f"stages={','.join(config.stages)}", flush=True)
    print(f"smoke={config.smoke} device={config.device}", flush=True)
    return PipelineV2(config).run()
