"""CLI for isolated V4.3. V4.2 and all earlier trees stay read-only. V6 is not retuned."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.io_util import DEFAULT_V3_OUTPUT, E0_OUTPUT, V2_OUTPUT
from experiments.qa_chunk_selection.v3_1_auto_prep.contract import V3_1_AUTO_ARTIFACT_ROOT, V3_1_RANKING_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v3_1_prep.contract import V3_1_PREP_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v3_d45.contract import V3_D45_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v4.contract import V3_1_AUTO_V2_ARTIFACT_ROOT, V4_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v4.keybert_features import resolve_encoder_device
from experiments.qa_chunk_selection.v4_1.contract import V4_1_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v4_2.contract import V4_2_ARTIFACT_ROOT

from .config import ExperimentConfigV43, parse_stages
from .contract import V4_3_ARTIFACT_ROOT
from .io_guard import assert_isolated_output, refuse_completed_output
from .pipeline import PipelineV43


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "V4.3 semantic reader agreement on frozen V6-K5. "
            "Writes only to artifacts/qa_chunk_selection_v4_3/ unless --output-dir is a new path. "
            "Never writes E0/V2/V3/V3.1/D45/V4/V4.1/V4.2 trees. Does not retune V6."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / V4_3_ARTIFACT_ROOT)
    parser.add_argument("--v3-artifact-dir", type=Path, default=DEFAULT_V3_OUTPUT)
    parser.add_argument("--v4-artifact-dir", type=Path, default=PROJECT_ROOT / V4_ARTIFACT_ROOT)
    parser.add_argument("--v4-1-artifact-dir", type=Path, default=PROJECT_ROOT / V4_1_ARTIFACT_ROOT)
    parser.add_argument("--v4-2-artifact-dir", type=Path, default=PROJECT_ROOT / V4_2_ARTIFACT_ROOT)
    parser.add_argument("--stages", default="primary", help="r0 | cache | primary | full | comma-separated stages")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--no-reuse-v4-1-qa-cache", action="store_true")
    parser.add_argument("--force-semctx", action="store_true", help="Run SEMCTX1 even if SEM1 gate failed (exploratory).")
    parser.add_argument(
        "--allow-internal-test",
        action="store_true",
        help="Accepted but ignored: INTERNAL_TEST stays locked in the default V4.3 run.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfigV43:
    config = ExperimentConfigV43(
        output_dir=Path(args.output_dir),
        v3_artifact_dir=Path(args.v3_artifact_dir),
        v4_artifact_dir=Path(args.v4_artifact_dir),
        v4_1_artifact_dir=Path(args.v4_1_artifact_dir),
        v4_2_artifact_dir=Path(args.v4_2_artifact_dir),
        stages=parse_stages(args.stages),
        device=str(args.device),
        seed=int(args.seed),
        smoke=bool(args.smoke),
        skip_qa=bool(args.skip_qa),
        allow_existing_output=bool(args.allow_existing_output),
        allow_internal_test=bool(args.allow_internal_test),
        reuse_v4_1_qa_cache=not bool(args.no_reuse_v4_1_qa_cache),
        force_semctx=bool(args.force_semctx),
    )
    if config.smoke:
        config = replace(
            config,
            n_bootstrap=50,
            delta_sem_grid=(0.0, 0.5),
            lexcore_majority_grid=(0.5, 1.0),
            lexcore_token_grid=(0.0, 0.5),
            semcore_majority_grid=(0.5, 1.0),
            semcore_sem_grid=(0.0, 0.5),
            adapt_quantiles=(0.50, 0.80),
            context_delta_grid=(0.0, 0.25),
            reuse_v4_1_qa_cache=False,
        )
    return config


def _forbidden_trees() -> dict[str, Path]:
    return {
        "E0": E0_OUTPUT.resolve(),
        "V2": V2_OUTPUT.resolve(),
        "V3": DEFAULT_V3_OUTPUT.resolve(),
        "manual V3.1 prep": (PROJECT_ROOT / V3_1_PREP_ARTIFACT_ROOT).resolve(),
        "automated silver": (PROJECT_ROOT / V3_1_AUTO_ARTIFACT_ROOT).resolve(),
        "automated silver v2": (PROJECT_ROOT / V3_1_AUTO_V2_ARTIFACT_ROOT).resolve(),
        "V3.1 ranking": (PROJECT_ROOT / V3_1_RANKING_ARTIFACT_ROOT).resolve(),
        "D4/D5": (PROJECT_ROOT / V3_D45_ARTIFACT_ROOT).resolve(),
        "V4": (PROJECT_ROOT / V4_ARTIFACT_ROOT).resolve(),
        "V4.1": (PROJECT_ROOT / V4_1_ARTIFACT_ROOT).resolve(),
        "V4.2": (PROJECT_ROOT / V4_2_ARTIFACT_ROOT).resolve(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    out = Path(config.output_dir).resolve()
    for name, path in _forbidden_trees().items():
        if out == path or path in out.parents:
            print(f"Refusing to write V4.3 into {name} artifacts: {path}", flush=True)
            return 2
    try:
        refuse_completed_output(out, allow=config.allow_existing_output)
        assert_isolated_output(out)
    except ValueError as exc:
        print(str(exc), flush=True)
        return 2
    if config.allow_internal_test:
        print("WARN --allow-internal-test is ignored; INTERNAL_TEST stays locked.", flush=True)
    print(f"project_root={PROJECT_ROOT}", flush=True)
    print(f"output_dir={config.output_dir}", flush=True)
    print(f"v3_artifacts={config.v3_artifact_dir} (read-only)", flush=True)
    print(f"v4_artifacts={config.v4_artifact_dir} (read-only)", flush=True)
    print(f"v4_1_artifacts={config.v4_1_artifact_dir} (read-only; V6-K5 weights/OOF)", flush=True)
    print(f"v4_2_artifacts={config.v4_2_artifact_dir} (read-only; historical CORE3)", flush=True)
    print(f"stages={','.join(config.stages)}", flush=True)
    resolved = "cpu" if config.smoke else resolve_encoder_device(config.device)
    print(f"smoke={config.smoke} device={config.device}->{resolved} seed={config.seed}", flush=True)
    print("V6 is not retuned. Semantic encoder is frozen. Attention/K10/U80 are not in V4.3.", flush=True)
    print("progress is written to stdout and to output_dir/progress.jsonl", flush=True)
    return PipelineV43(config).run(list(config.stages))
