"""CLI for SciREX Phase 1. Does not rewrite preprocessing artifacts."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 1: positive / FN / hard-negative labels on preprocessed SciREX"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "scirex_aspect_qa_preprocessing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase1_scirex",
    )
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-docs", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--gap-bootstrap", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gumbel-ablation", action="store_true")
    parser.add_argument("--gumbel-temperature", type=float, default=0.5)
    parser.add_argument(
        "--from-predictions",
        type=Path,
        default=None,
        help="Reuse stored QA spans from qa_chunk_dataset.parquet or qa_chunk_predictions.parquet; skip the QA model.",
    )
    parser.add_argument(
        "--match-method",
        choices=("gold_span_containment", "normalized_exact_match"),
        default="gold_span_containment",
    )
    parser.add_argument("--containment-max-extra-tokens", type=int, default=5)
    parser.add_argument(
        "--document-ids-file",
        type=Path,
        default=None,
        help="Restrict to these document_id values (one per line). Used for remainder slices.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .checks import check_instances
    from .config import Phase1Config
    from .engine import (
        prediction_lookup,
        load_encoder,
        load_models,
        process_document_from_predictions,
        process_document_instances,
        select_instances,
        source_chunk_index,
    )
    from .report import assert_isolated_output, collect_statistics, write_outputs
    from .tables import eligible_chunks, load_document_id_list, load_preprocessing

    if args.split == "test" and not args.debug:
        print("TEST is reserved until the algorithm is frozen. Refusing a full TEST run.")
        return 2

    output_dir = assert_isolated_output(Path(args.output_dir), PROJECT_ROOT)
    document_ids = load_document_id_list(args.document_ids_file) if args.document_ids_file else ()
    if args.document_ids_file is not None and not document_ids:
        print(f"document-ids-file is empty: {args.document_ids_file}")
        return 2
    config = Phase1Config(
        input_dir=Path(args.input_dir),
        output_dir=output_dir,
        split=args.split,
        debug=bool(args.debug),
        debug_docs=int(args.debug_docs),
        seed=int(args.seed),
        device=str(args.device),
        hard_negative_top_n=int(args.top_n),
        gap_bootstrap=int(args.gap_bootstrap),
        gumbel_ablation=bool(args.gumbel_ablation),
        gumbel_temperature=float(args.gumbel_temperature),
        positive_match_method=str(args.match_method),
        containment_max_extra_tokens=int(args.containment_max_extra_tokens),
        document_ids=document_ids,
    )
    tables = load_preprocessing(config.input_dir)
    instances = select_instances(tables, config)
    if args.document_ids_file is not None and instances.empty:
        print("no eligible instances for the requested document ids")
        return 2
    excluded_mask = (tables["aspect_instances"]["split"] == config.split) & (
        tables["aspect_instances"]["status"] == "no_extractable_gold"
    )
    if document_ids:
        excluded_mask = excluded_mask & tables["aspect_instances"]["document_id"].astype(str).isin(
            set(document_ids)
        )
    excluded = int(excluded_mask.sum())
    print(
        f"Phase 1 split={config.split} debug={config.debug} "
        f"instances={len(instances)} docs={instances['document_id'].nunique()} "
        f"top_n={config.hard_negative_top_n} match={config.positive_match_method} "
        f"max_extra={config.containment_max_extra_tokens} gumbel={config.gumbel_ablation}"
    )
    started = time.perf_counter()
    doc_ids = list(dict.fromkeys(instances["document_id"].astype(str).tolist()))
    results = []
    if args.from_predictions is not None:
        pred_path = Path(args.from_predictions)
        pred_rows = pd.read_parquet(pred_path)
        predictions = prediction_lookup(pred_rows)
        encode_fn, model_meta = load_encoder(config)
        model_meta["predictions_path"] = str(pred_path)
        for index, doc_id in enumerate(doc_ids, start=1):
            doc_instances = instances[instances["document_id"] == doc_id]
            chunk_rows = eligible_chunks(tables["chunks"], doc_id)
            batch = process_document_from_predictions(
                document_id=doc_id,
                instance_rows=doc_instances,
                chunk_rows=chunk_rows,
                mentions=tables["gold_mentions"],
                predictions=predictions,
                config=config,
                encode_fn=encode_fn,
            )
            results.extend(batch)
            if index == 1 or index % 5 == 0 or index == len(doc_ids):
                elapsed = time.perf_counter() - started
                print(f"  [{index}/{len(doc_ids)}] {doc_id[:12]} instances={len(batch)} elapsed={elapsed:.1f}s")
    else:
        answer_fn, encode_fn, model_meta = load_models(config)
        for index, doc_id in enumerate(doc_ids, start=1):
            doc_instances = instances[instances["document_id"] == doc_id]
            chunk_rows = eligible_chunks(tables["chunks"], doc_id)
            batch = process_document_instances(
                document_id=doc_id,
                instance_rows=doc_instances,
                chunk_rows=chunk_rows,
                mentions=tables["gold_mentions"],
                config=config,
                answer_fn=answer_fn,
                encode_fn=encode_fn,
            )
            results.extend(batch)
            if index == 1 or index % 5 == 0 or index == len(doc_ids):
                elapsed = time.perf_counter() - started
                print(f"  [{index}/{len(doc_ids)}] {doc_id[:12]} instances={len(batch)} elapsed={elapsed:.1f}s")

    source = source_chunk_index(tables["chunks"])
    errors = check_instances(
        results,
        source_chunks=source,
        match_method=config.positive_match_method,
        max_extra_tokens=config.containment_max_extra_tokens,
    )
    stats = collect_statistics(
        results,
        excluded_no_extractable=excluded,
        documents_in_split=int(instances["document_id"].nunique()),
    )
    extra = {
        **model_meta,
        "preprocessing_config": str(config.input_dir / "preprocess_config.json"),
        "inherited_chunk_max_tokens": 384,
        "inherited_overlap_tokens": 50,
        "check_error_count": len(errors),
        "n_documents_processed": len(doc_ids),
        "elapsed_sec": time.perf_counter() - started,
    }
    written = write_outputs(
        output_dir,
        results=results,
        config=config,
        extra_config=extra,
        stats=stats,
        check_errors=errors,
    )
    print("sanity_errors", len(errors))
    for item in errors[:20]:
        print("  ERROR", item)
    print(
        "no_qa_positive",
        stats["no_qa_positive_instances"],
        "positives",
        stats["overall"].get("positive_chunks_total"),
        "FN",
        stats["overall"].get("false_negative_candidates_total"),
        "HN",
        stats["overall"].get("hard_negatives_total"),
    )
    print("written:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
