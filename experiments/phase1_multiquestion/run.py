"""CLI for the multi-question SciREX Phase 1 extension."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from experiments.phase1_scirex.report import assert_isolated_output

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 1 multi-question extension: Q0 / Q3 / Q5 on preprocessed SciREX"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "scirex_aspect_qa_preprocessing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase1_multiquestion",
    )
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--mode", choices=("Q0", "Q3", "Q5"), default="Q3")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-docs", type=int, default=8)
    parser.add_argument("--select-threshold", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-file", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--gap-bootstrap", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gumbel-ablation", action="store_true")
    parser.add_argument(
        "--canonical-predictions",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase1_scirex" / "train" / "qa_chunk_dataset.parquet",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="JSONL QA cache shared across Q3/Q5 and splits. Default: output-dir/qa_prediction_cache.jsonl",
    )
    parser.add_argument("--bm25-aggregate", choices=("mean_normalized", "rrf", "mean_rank"), default="mean_normalized")
    parser.add_argument("--compare", nargs="*", default=(), help="output dirs to compare: NAME=PATH")
    parser.add_argument(
        "--document-ids-file",
        type=Path,
        default=None,
        help="Restrict to these document_id values (one per line). Used for remainder slices.",
    )
    return parser


def _n_questions(mode: str) -> int:
    return {"Q0": 1, "Q3": 3, "Q5": 5}[mode]


def _resolve_threshold(args: argparse.Namespace, mode: str) -> float | None:
    if args.debug and args.threshold is None and not args.select_threshold:
        return 1.0
    if args.threshold is not None:
        return float(args.threshold)
    if args.threshold_file is not None and Path(args.threshold_file).exists():
        payload = json.loads(Path(args.threshold_file).read_text(encoding="utf-8"))
        if mode in payload:
            return float(payload[mode]["chosen_threshold"])
        if "chosen_threshold" in payload:
            return float(payload["chosen_threshold"])
    if mode == "Q0":
        return 1.0
    if args.select_threshold:
        return None
    raise SystemExit(
        f"{mode} requires --threshold, --threshold-file from DEV, or --select-threshold on DEV"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.compare:
        from .compare import compare_runs, write_comparison

        paths = {}
        for item in args.compare:
            name, _, raw = item.partition("=")
            paths[name] = Path(raw)
        payload = compare_runs(paths)
        write_comparison(payload, Path(args.output_dir) / "q0_q3_q5_comparison.json")
        print("wrote", Path(args.output_dir) / "q0_q3_q5_comparison.json")
        return 0

    from experiments.phase1_scirex.tables import load_document_id_list

    from .checks import check_instances
    from .config import MultiQuestionConfig
    from .engine import (
        PredictionCache,
        eligible_chunks,
        load_canonical_predictions,
        load_encoder,
        load_models,
        load_preprocessing,
        preprocessing_gold_index,
        process_document,
        select_instances,
        select_threshold_on_dev,
        source_chunk_index,
    )
    from .questions import assert_canonical_baseline
    from .report import collect_statistics, write_outputs

    if args.split == "test" and not args.debug:
        print("TEST is reserved until the algorithm is frozen. Refusing a full TEST run.")
        return 2
    if args.select_threshold and args.split != "dev":
        print("Threshold must be selected on DEV, not on TRAIN/TEST.")
        return 2

    assert_canonical_baseline()
    n_questions = _n_questions(args.mode)
    output_dir = assert_isolated_output(Path(args.output_dir), PROJECT_ROOT)
    cache_path = Path(args.cache_path) if args.cache_path else output_dir / "qa_prediction_cache.jsonl"
    document_ids = load_document_id_list(args.document_ids_file) if args.document_ids_file else ()
    if args.document_ids_file is not None and not document_ids:
        print(f"document-ids-file is empty: {args.document_ids_file}")
        return 2
    config = MultiQuestionConfig(
        input_dir=Path(args.input_dir),
        output_dir=output_dir,
        split=args.split,
        mode=args.mode,
        n_questions=n_questions,
        debug=bool(args.debug),
        debug_docs=int(args.debug_docs),
        seed=int(args.seed),
        device=str(args.device),
        hard_negative_top_n=int(args.top_n),
        gap_bootstrap=int(args.gap_bootstrap),
        gumbel_ablation=bool(args.gumbel_ablation),
        bm25_aggregate=str(args.bm25_aggregate),
        canonical_predictions_path=Path(args.canonical_predictions) if args.canonical_predictions else None,
        prediction_cache_path=cache_path,
        threshold_file=Path(args.threshold_file) if args.threshold_file else None,
        robust_positive_threshold=_resolve_threshold(args, args.mode) if not args.select_threshold else None,
        document_ids=document_ids,
    )
    tables = load_preprocessing(config.input_dir)
    instances = select_instances(tables, config)
    if args.document_ids_file is not None and instances.empty:
        print("no eligible instances for the requested document ids")
        return 2
    gold_index = preprocessing_gold_index(tables["aspect_instances"])
    print(
        f"multi-q mode={config.mode} split={config.split} debug={config.debug} "
        f"instances={len(instances)} docs={instances['document_id'].nunique()} "
        f"n_questions={config.n_questions}"
    )
    cache = PredictionCache(config.prediction_cache_path)
    canonical = {}
    if config.split == "train" and config.canonical_predictions_path:
        canonical = load_canonical_predictions(config.canonical_predictions_path)
        print(f"reused canonical predictions: {len(canonical)}")
    need_qa = True
    if config.mode == "Q0" and canonical:
        need_qa = False
    answer_fn = None
    if need_qa:
        answer_fn, encode_fn, model_meta = load_models(config)  # type: ignore[arg-type]
    else:
        encode_fn, model_meta = load_encoder(config)
        model_meta["qa_reused_from_predictions"] = True
    model_meta["device"] = model_meta.get("device", config.device)
    placeholder_threshold = config.robust_positive_threshold if config.robust_positive_threshold is not None else 1.0
    results = []
    started = time.perf_counter()
    doc_ids = list(dict.fromkeys(instances["document_id"].astype(str).tolist()))
    for index, doc_id in enumerate(doc_ids, start=1):
        batch = process_document(
            document_id=doc_id,
            instance_rows=instances[instances["document_id"] == doc_id],
            chunk_rows=eligible_chunks(tables["chunks"], doc_id),
            mentions=tables["gold_mentions"],
            config=config,
            encode_fn=encode_fn,
            answer_fn=answer_fn,
            cache=cache,
            canonical=canonical,
            threshold=placeholder_threshold,
        )
        results.extend(batch)
        if index == 1 or index % 5 == 0 or index == len(doc_ids):
            print(f"  [{index}/{len(doc_ids)}] {doc_id[:12]} instances={len(batch)} elapsed={time.perf_counter()-started:.1f}s")

    threshold_selection = None
    chosen = config.robust_positive_threshold
    if args.select_threshold:
        threshold_selection = select_threshold_on_dev(
            results,
            encode_fn=encode_fn,
            config=config,
            mentions=tables["gold_mentions"],
        )
        results = threshold_selection["results"]
        chosen = float(threshold_selection["chosen_threshold"])
        print("DEV chosen_threshold", chosen, "candidates", threshold_selection["candidates"])
    extra = {
        **model_meta,
        "robust_positive_threshold": chosen,
        "n_documents_processed": len(doc_ids),
        "elapsed_sec": time.perf_counter() - started,
        "canonical_predictions_used": bool(canonical),
    }
    source = source_chunk_index(tables["chunks"])
    errors = check_instances(
        results,
        source_chunks=source,
        preprocessing_gold=gold_index,
        match_method=config.positive_match_method,
        max_extra_tokens=config.containment_max_extra_tokens,
        n_questions=config.n_questions,
    )
    stats = collect_statistics(results)
    written = write_outputs(
        output_dir,
        results=results,
        config=config,
        extra_config=extra,
        stats=stats,
        check_errors=errors,
        threshold_selection=threshold_selection,
    )
    print("sanity_errors", len(errors))
    for item in errors[:20]:
        print("  ERROR", item)
    print(
        "no_robust_positive",
        stats["no_robust_positive_instances"],
        "robust_positive",
        stats["robust_positive_chunks"],
        "sensitive",
        stats["question_sensitive_chunks"],
        "HN",
        stats["hard_negative_chunks"],
    )
    for name, path in written.items():
        print(f"  {name}: {path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
