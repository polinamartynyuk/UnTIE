"""CLI for SciREX aspect-level extractive QA preprocessing."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from ..config import PipelineConfig
from .config import AspectQAPreprocessConfig, default_output_dir
from .invariants import validate_processed_document
from .io import write_json, write_tables
from .pipeline import (
    assign_splits,
    aspect_instance_rows,
    chunk_rows,
    document_row,
    process_document,
    qa_input_rows,
    resolve_chunk_budget,
    source_metadata_row,
    stable_sample,
)
from .source import inspect_source, load_scirex_records
from .stats import collect_statistics, format_statistics, manual_review_payload


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class WhitespaceTokenizer:
    """Test/fallback tokenizer with the WordTokenizer surface."""

    def tokenize(self, text: str) -> list[str]:
        return str(text).split()

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        del kwargs
        return list(range(len(self.tokenize(text))))

    def convert_tokens_to_string(self, tokens: Sequence[str]) -> str:
        return " ".join(tokens)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare SciREX aspect-level extractive QA tables (source-only, E0)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "scirex_structured.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_dir(PROJECT_ROOT),
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N records")
    parser.add_argument(
        "--doc-ids",
        nargs="*",
        default=(),
        help="Optional explicit document ids (still never writes back to the source file)",
    )
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument(
        "--tokenizer",
        choices=("qa_model", "whitespace"),
        default="qa_model",
        help="qa_model reuses the English extractive QA tokenizer",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--manual-review", type=int, default=20)
    parser.add_argument(
        "--fixed-split",
        default=None,
        choices=("train", "dev", "test"),
        help=(
            "Assign every document this official SciREX split instead of hashing "
            "document ids. Use a separate --output directory for official dev/test "
            "so existing train tables are not overwritten."
        ),
    )
    return parser


def load_tokenizer(kind: str, *, device: str):
    if kind == "whitespace":
        return WhitespaceTokenizer()
    from ..models import ModelFactory

    config = PipelineConfig(device=device)
    return ModelFactory(config).tokenizer


def print_inspection(inspection: Any) -> None:
    print("=== source inspection ===")
    print(f"input: {inspection.path}")
    print(f"format: {inspection.format_name}")
    print(f"records: {inspection.n_records}")
    print(f"fields present: {list(inspection.fields_present)}")
    print(f"field frequency: {inspection.field_frequency}")
    print(f"first record fields: {list(inspection.first_record_fields)}")
    print("first record preview:")
    print(json.dumps(inspection.first_record_preview, ensure_ascii=False, indent=2)[:4000])
    print(f"expected schema: {list(inspection.expected_fields)}")
    print(f"missing expected: {list(inspection.missing_expected_fields)}")
    print(f"unexpected: {list(inspection.unexpected_fields)}")
    print(f"schema_matches_expected: {inspection.schema_matches_expected}")
    for note in inspection.notes:
        print(f"note: {note}")
    print("=== reused project functions ===")
    print("- untie.data.load_json_dataframe / JSONL-or-JSON loading (same strategy)")
    print("- untie.text.SentenceSplitter")
    print("- untie.chunking.ChunkBuilder")
    print("- untie.config.PipelineConfig chunk_max_tokens / overlap_tokens")
    print("- untie.models.ModelFactory.tokenizer (QA model tokenizer)")
    print("- untie.keyword_tuning.deterministic_document_split")
    print("- experiments.keybert_topic_clustering.documents heading-style title prefix")


def _select_records(
    payload: list[dict[str, Any]],
    *,
    limit: int,
    doc_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if doc_ids:
        wanted = {str(item) for item in doc_ids}
        selected = [item for item in payload if str(item.get("doc_id")) in wanted]
        missing = wanted - {str(item.get("doc_id")) for item in selected}
        if missing:
            raise ValueError(f"doc_ids not found: {sorted(missing)}")
        return selected
    if limit and limit > 0:
        return payload[:limit]
    return payload


def run(args: argparse.Namespace) -> int:
    train_output = default_output_dir(PROJECT_ROOT)
    if args.fixed_split in {"dev", "test"} and Path(args.output).resolve() == train_output.resolve():
        print(
            "Refusing to write official SciREX "
            f"{args.fixed_split} tables into {train_output}. "
            "Pass a separate --output directory."
        )
        return 2

    inspection = inspect_source(args.input)
    print_inspection(inspection)
    write_json(Path(args.output) / "source_inspection.json", asdict(inspection))
    if not inspection.schema_matches_expected:
        print("Refusing to guess missing/renamed fields. Adapt the parser from the inspection above.")
        if inspection.missing_expected_fields:
            return 2
    if args.inspect_only:
        return 0

    payload, _ = load_scirex_records(args.input)
    selected = _select_records(payload, limit=args.limit, doc_ids=args.doc_ids)
    print(f"selected documents: {len(selected)}")
    tokenizer = load_tokenizer(args.tokenizer, device=args.device)
    preprocess_config = AspectQAPreprocessConfig(fixed_split=args.fixed_split)
    pipeline_config = PipelineConfig(device=args.device)
    chunk_max = resolve_chunk_budget(tokenizer, preprocess_config, pipeline_config)
    print(
        f"chunk_max_tokens={chunk_max} overlap_tokens={pipeline_config.overlap_tokens} "
        f"INCLUDE_BIBLIOGRAPHY_IN_QA={preprocess_config.include_bibliography_in_qa} "
        f"ENABLE_SCORE_ASPECT={preprocess_config.enable_score_aspect} "
        f"enrichment_mode={preprocess_config.enrichment_mode}"
    )

    all_ids = [str(item["doc_id"]) for item in payload]
    split_of = assign_splits(all_ids, preprocess_config)
    processed = []
    for raw in selected:
        item = process_document(
            raw,
            tokenizer=tokenizer,
            config=preprocess_config,
            pipeline_config=pipeline_config,
            chunk_max_tokens=chunk_max,
            split=split_of[str(raw["doc_id"])],
        )
        validate_processed_document(item)
        processed.append(item)

    documents = [document_row(item) for item in processed]
    chunks = [row for item in processed for row in chunk_rows(item)]
    instances = [row for item in processed for row in aspect_instance_rows(item)]
    mentions = [row for item in processed for row in item.mentions]
    metadata = [source_metadata_row(item) for item in processed]
    qa_rows = [row for item in processed for row in qa_input_rows(item)]
    written = write_tables(
        Path(args.output),
        documents=documents,
        chunks=chunks,
        aspect_instances=instances,
        gold_mentions=mentions,
        source_metadata=metadata,
        enrichment=[],
        qa_input=qa_rows,
    )
    stats = collect_statistics(processed)
    write_json(Path(args.output) / "statistics.json", stats)
    write_json(
        Path(args.output) / "preprocess_config.json",
        {
            "enrichment_mode": preprocess_config.enrichment_mode,
            "enable_score_aspect": preprocess_config.enable_score_aspect,
            "include_bibliography_in_qa": preprocess_config.include_bibliography_in_qa,
            "aspect_questions": dict(preprocess_config.aspect_questions),
            "chunk_max_tokens": chunk_max,
            "overlap_tokens": pipeline_config.overlap_tokens,
            "tokenizer": args.tokenizer,
            "n_selected": len(processed),
            "source": str(args.input),
            "fixed_split": preprocess_config.fixed_split,
            "official_corpus_split": preprocess_config.fixed_split,
            "written": written,
        },
    )
    print(format_statistics(stats))
    print("written files:")
    for name, path in written.items():
        print(f"  {name}: {path}")

    sample_ids = stable_sample([item.record.doc_id for item in processed], args.manual_review)
    by_id = {item.record.doc_id: item for item in processed}
    review = [manual_review_payload(by_id[doc_id]) for doc_id in sample_ids if doc_id in by_id]
    write_json(Path(args.output) / "manual_review_sample.json", review)
    print(f"manual review sample: {len(review)} documents → {args.output}/manual_review_sample.json")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))
