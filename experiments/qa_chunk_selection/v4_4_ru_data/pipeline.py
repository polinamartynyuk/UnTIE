"""End-to-end preparation; all writes stay below one newly claimed artifact root."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from experiments.phase1_scirex.config import Phase1Config
from experiments.phase1_scirex.engine import process_document_instances
from experiments.phase1_scirex.report import chunk_dataset_rows, instance_rows
from experiments.qa_chunk_selection.io_util import write_json, write_jsonl
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.schema import ChunkRecord, GoldSpan, InstanceRecord
from experiments.qa_chunk_selection.v2.hn_flags import hn_quality_rows
from experiments.qa_chunk_selection.v2.split import apply_document_roles, split_document_ids, split_sanity
from experiments.qa_chunk_selection.v3.data import write_records
from experiments.qa_chunk_selection.v3.split import build_cv_splits
from untie.aspect_qa.chunk_index import build_indexed_chunks, resolve_context_budget
from untie.aspect_qa.io import write_tables
from untie.aspect_qa.pipeline import process_aspect
from untie.aspect_qa.source import TextSegment
from untie.config import PipelineConfig
from untie.models import ModelFactory, profile_for_language
from untie.text import RussianSentenceSplitter

from .config import ASPECT_SPECS, RuDataConfig
from .official_source import (
    SourceDocument,
    answers_by_aspect,
    download_official_parts,
    load_official_parts,
    load_structured_csv,
    sha256_path,
)


PRIMARY_TO_EXPERIMENT = {
    "qa_positive": "positive",
    "evidence_positive_qa_failed": "ambiguous",
    "confirmed_false_negative": "ambiguous",
    "false_negative_candidate": "ambiguous",
    "hard_negative": "hard_negative",
    "easy_negative": "easy_negative",
}
UNLOCKED_SPLITS = frozenset({"train", "calibration", "development"})


def claim_output(path: Path) -> Path:
    root = Path(__file__).resolve().parents[3]
    resolved = Path(path).absolute()
    if any(item.is_symlink() for item in (resolved, *resolved.parents)):
        raise ValueError("output path cannot contain symlinks")
    artifacts = root / "artifacts"
    if resolved == root or (root in resolved.parents and artifacts not in resolved.parents):
        raise ValueError("project-local output must be below artifacts/")
    protected = (
        root / "datasets",
        root / "untie",
        root / "experiments",
        root / "artifacts" / "qa_chunk_selection",
        root / "artifacts" / "qa_chunk_selection_v2",
        root / "artifacts" / "qa_chunk_selection_v3",
        root / "artifacts" / "qa_chunk_selection_v4_4",
    )
    for item in protected:
        candidate = item.resolve()
        if resolved == candidate or candidate in resolved.parents or resolved in candidate.parents:
            raise ValueError(f"refusing protected output path: {resolved}")
    if resolved.exists():
        raise FileExistsError(f"output directory must be new: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.mkdir(exist_ok=False)
    return resolved


def _local_source_manifest(source_dir: Path) -> dict[str, object]:
    files = []
    for path in sorted(source_dir.glob("aspects_data_part_*.csv")):
        files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_path(path)})
    return {"mode": "local_official_parts", "source_dir": str(source_dir.resolve()), "files": files}


def _rows_for_document(document: SourceDocument, chunks, aspects: Sequence[str]):
    segment = TextSegment(0, "", "", document.clean_text, 0, len(document.clean_text))
    by_aspect = answers_by_aspect(document)
    questions = {aspect: question for _, (aspect, question) in ASPECT_SPECS.items()}
    instance_data = []
    mentions = []
    for aspect in aspects:
        instance, mention_rows = process_aspect(
            document_id=document.document_id,
            aspect=aspect,
            question=questions[aspect],
            raw_values=by_aspect.get(aspect, ()),
            text=document.clean_text,
            chunks=chunks,
            segments=(segment,),
            include_bibliography_in_qa=True,
        )
        instance_data.append(instance)
        mentions.extend(mention_rows)
    document_row = {
        "document_id": document.document_id,
        "split": "source",
        "raw_text": document.tagged_text,
        "clean_text": document.clean_text,
        "n_chunks": len(chunks),
        "n_segments": 1,
    }
    chunk_data = [
        {
            "document_id": document.document_id,
            "chunk_id": item.chunk_id,
            "chunk_text": item.text,
            "char_start": item.char_start,
            "char_end": item.char_end,
            "token_start": item.token_start,
            "token_end": item.token_end,
            "section_type": "",
            "section_title": "",
            "eligible_for_qa": True,
            "segment_ids": [0],
            "token_count": item.token_count,
        }
        for item in chunks
    ]
    aspect_rows = [
        {
            "document_id": item.document_id,
            "split": "source",
            "aspect": item.aspect,
            "question": item.question,
            "status": item.status,
            "raw_gold_answers": list(item.raw_gold_answers),
            "canonical_gold_answers": list(item.canonical_gold_answers),
            "extractable_gold_answers": list(item.extractable_gold_answers),
            "unextractable_gold_answers": list(item.unextractable_gold_answers),
            "external_enrichment": [],
            "enrichment_candidates": [],
            "chunk_ids": list(item.chunk_ids),
            "gold_supporting_chunk_ids": list(item.gold_supporting_chunk_ids),
        }
        for item in instance_data
    ]
    qa_rows = []
    by_chunk = {item.chunk_id: item for item in chunks}
    for item in instance_data:
        if item.status == "no_extractable_gold":
            continue
        for chunk_id in item.chunk_ids:
            chunk = by_chunk[chunk_id]
            qa_rows.append(
                {
                    "document_id": item.document_id,
                    "split": "source",
                    "aspect": item.aspect,
                    "question": item.question,
                    "status": item.status,
                    "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.text,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "extractable_gold_answers": list(item.extractable_gold_answers),
                    "gold_supporting": chunk.chunk_id in set(item.gold_supporting_chunk_ids),
                    "enrichment_mode": "E0",
                }
            )
    metadata = {
        "document_id": document.document_id,
        "source_id": document.source_id,
        "source_part": document.source_part,
        "tag_counts": dict(Counter(span.tag for span in document.spans)),
        "parse_warnings": list(document.parse_warnings),
    }
    return document_row, chunk_data, aspect_rows, mentions, metadata, qa_rows


def _load_russian_models(config: RuDataConfig):
    pipeline_config = PipelineConfig(
        profile=profile_for_language("ru"),
        chunk_max_tokens=config.chunk_max_tokens,
        overlap_tokens=config.overlap_tokens,
        device=config.device,
    )
    factory = ModelFactory(pipeline_config)

    def answer(question: str, context: str) -> dict[str, Any]:
        return dict(factory.answerer(question=question, context=context))

    def encode(texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=float)
        matrix = np.asarray(factory.sentence_encoder.encode(list(texts), show_progress_bar=False), dtype=float)
        return matrix.reshape(1, -1) if matrix.ndim == 1 else matrix

    return factory, answer, encode


def _records_from_results(results, mentions: pd.DataFrame) -> tuple[InstanceRecord, ...]:
    mention_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in mentions.to_dict("records"):
        mention_index.setdefault((str(row["document_id"]), str(row["aspect"])), []).append(row)
    records = []
    for item in results:
        ordered = sorted(item.chunks, key=lambda chunk: chunk.chunk_id)
        chunk_ids = [chunk.chunk_id for chunk in ordered]
        chunks = []
        for position, chunk in enumerate(ordered):
            primary = str(chunk.primary_class)
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    label=PRIMARY_TO_EXPERIMENT[primary],
                    primary_class=primary,
                    position=position,
                    section_id=f"pos:{position}",
                    section_title="",
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    token_start=chunk.token_start,
                    token_end=chunk.token_end,
                    gold_supporting=chunk.gold_supporting,
                    previous_chunk_id=chunk_ids[position - 1] if position else None,
                    next_chunk_id=chunk_ids[position + 1] if position + 1 < len(chunk_ids) else None,
                )
            )
        spans = []
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        for mention in mention_index.get((item.document_id, item.aspect), ()):
            start, end = int(mention["char_start"]), int(mention["char_end"])
            for chunk_id in mention["chunk_ids"]:
                chunk = by_id.get(int(chunk_id))
                if chunk is None or not (chunk.char_start <= start and end <= chunk.char_end):
                    continue
                rel_start, rel_end = start - chunk.char_start, end - chunk.char_start
                spans.append(
                    GoldSpan(
                        chunk_id=chunk.chunk_id,
                        start_char=rel_start,
                        end_char=rel_end,
                        text=chunk.text[rel_start:rel_end],
                        document_start_char=start,
                        document_end_char=end,
                    )
                )
        records.append(
            InstanceRecord(
                document_id=item.document_id,
                question_id=item.qa_instance_id,
                aspect=item.aspect,
                question=item.question,
                gold_answers=tuple(item.extractable_gold_answers),
                split="source",
                chunks=tuple(chunks),
                gold_spans=tuple(spans),
                instance_status=item.instance_status,
                source_split="source",
            )
        )
    return tuple(records)


def _sanity(records: Sequence[InstanceRecord], expected_aspects: Sequence[str]) -> list[str]:
    errors = []
    qids = [item.question_id for item in records]
    if len(qids) != len(set(qids)):
        errors.append("duplicate question_id")
    allowed = set(expected_aspects)
    for item in records:
        if item.aspect not in allowed:
            errors.append(f"unexpected aspect {item.question_id}: {item.aspect}")
        if not item.gold_answers:
            errors.append(f"empty gold answers {item.question_id}")
        if not item.gold_spans:
            errors.append(f"no fully contained gold span {item.question_id}")
        for span in item.gold_spans:
            chunk = item.chunk_by_id(span.chunk_id)
            if chunk.text[span.start_char : span.end_char] != span.text:
                errors.append(f"gold span drift {item.question_id}:{span.chunk_id}")
        for chunk in item.chunks:
            if chunk.label == "hard_negative" and (chunk.gold_supporting or chunk.primary_class != "hard_negative"):
                errors.append(f"polluted hard negative {item.question_id}:{chunk.chunk_id}")
    return errors


class RuDataPipeline:
    def __init__(self, config: RuDataConfig) -> None:
        config.validate()
        self.config = config
        self.output = claim_output(config.output_dir)
        self.progress = ProgressLogger(self.output / "progress.jsonl")

    def run(self) -> dict[str, Any]:
        started = time.time()
        write_json(self.output / "config.json", self.config.to_dict())
        self.progress.stage("source", "official RuSERRC aspect corpus")
        if self.config.download:
            source_dir = self.output / "source_raw"
            source_manifest = download_official_parts(source_dir, self.progress.log)
        elif self.config.structured_source is not None:
            structured = Path(self.config.structured_source)
            source_manifest = {
                "mode": "local_complete_structured_mirror",
                "path": str(structured.resolve()),
                "bytes": structured.stat().st_size,
                "sha256": sha256_path(structured),
                "annotation_source": "inline tags in text column",
            }
            documents = list(load_structured_csv(structured))
        else:
            source_dir = Path(self.config.source_dir or "")
            source_manifest = _local_source_manifest(source_dir)
            documents = list(load_official_parts(source_dir))
        if self.config.download:
            documents = list(load_official_parts(source_dir))
        if self.config.limit_docs > 0:
            documents = documents[: self.config.limit_docs]
        write_json(self.output / "source_manifest.json", source_manifest)
        write_jsonl(
            self.output / "source_parse_warnings.jsonl",
            ({"document_id": doc.document_id, "source_id": doc.source_id, "warning": warning}
             for doc in documents for warning in doc.parse_warnings),
        )
        self.progress.log("source validated", documents=len(documents))

        self.progress.stage("models", "Russian tokenizer, extractive QA and sentence encoder")
        factory, answer_fn, encode_fn = _load_russian_models(self.config)
        tokenizer = factory.tokenizer
        questions = [question for _, (aspect, question) in ASPECT_SPECS.items() if aspect in self.config.aspects]
        chunk_budget = resolve_context_budget(
            tokenizer, questions, configured_max_tokens=self.config.chunk_max_tokens
        )
        self.progress.log(
            "Russian model profile loaded",
            qa_model=factory.config.profile.qa_model,
            sentence_model=factory.config.profile.sentence_model,
            device=factory.device,
            chunk_budget=chunk_budget,
        )

        self.progress.stage("preprocess", "offset-safe Russian chunking and aspect mentions")
        document_rows: list[dict[str, Any]] = []
        chunk_rows: list[dict[str, Any]] = []
        aspect_rows: list[dict[str, Any]] = []
        mention_rows: list[dict[str, Any]] = []
        metadata_rows: list[dict[str, Any]] = []
        qa_rows: list[dict[str, Any]] = []
        splitter = RussianSentenceSplitter()
        for index, document in enumerate(documents, start=1):
            segment = TextSegment(0, "", "", document.clean_text, 0, len(document.clean_text))
            chunks = build_indexed_chunks(
                document.clean_text,
                tokenizer,
                max_tokens=chunk_budget,
                overlap_tokens=self.config.overlap_tokens,
                segments=(segment,),
                include_bibliography_in_qa=True,
                splitter=splitter,
            )
            rows = _rows_for_document(document, chunks, self.config.aspects)
            document_rows.append(rows[0]); chunk_rows.extend(rows[1]); aspect_rows.extend(rows[2])
            mention_rows.extend(rows[3]); metadata_rows.append(rows[4]); qa_rows.extend(rows[5])
            self.progress.count(index, len(documents), item="documents preprocessed")
        prep_dir = self.output / "preprocessing"
        written_tables = write_tables(
            prep_dir,
            documents=document_rows,
            chunks=chunk_rows,
            aspect_instances=aspect_rows,
            gold_mentions=mention_rows,
            source_metadata=metadata_rows,
            qa_input=qa_rows,
        )
        self.progress.log(
            "preprocessing tables written",
            chunks=len(chunk_rows), aspects=len(aspect_rows), mentions=len(mention_rows),
        )

        self.progress.stage("phase1", "frozen Russian QA labels and hard negatives")
        tables = {
            "chunks": pd.DataFrame(chunk_rows),
            "aspect_instances": pd.DataFrame(aspect_rows),
            "gold_mentions": pd.DataFrame(mention_rows),
        }
        unextractable = tables["aspect_instances"][
            tables["aspect_instances"]["unextractable_gold_answers"].map(bool)
        ]
        if not unextractable.empty:
            sample = unextractable[["document_id", "aspect", "unextractable_gold_answers"]].head(5).to_dict("records")
            raise RuntimeError(f"source aspect text became unextractable: {sample}")
        eligible = tables["aspect_instances"][
            tables["aspect_instances"]["status"].isin(("ready", "partially_extractable"))
        ].reset_index(drop=True)
        phase_config = Phase1Config(
            input_dir=prep_dir,
            output_dir=self.output / "phase1",
            split="source",
            seed=self.config.seed,
            device=self.config.device,
            hard_negative_top_n=self.config.hard_negative_top_n,
            gap_bootstrap=self.config.gap_bootstrap,
        )
        results = []
        doc_ids = list(dict.fromkeys(eligible["document_id"].astype(str).tolist()))
        for index, document_id in enumerate(doc_ids, start=1):
            doc_instances = eligible[eligible["document_id"] == document_id]
            doc_chunks = tables["chunks"][tables["chunks"]["document_id"] == document_id].sort_values("chunk_id")
            results.extend(
                process_document_instances(
                    document_id=document_id,
                    instance_rows=doc_instances,
                    chunk_rows=doc_chunks,
                    mentions=tables["gold_mentions"],
                    config=phase_config,
                    answer_fn=answer_fn,
                    encode_fn=encode_fn,
                )
            )
            self.progress.count(index, len(doc_ids), item="documents labelled", extra={"instances": len(results)})
        phase_dir = self.output / "phase1"
        phase_dir.mkdir(parents=True, exist_ok=False)
        phase_chunks = pd.DataFrame(chunk_dataset_rows(results))
        phase_instances = pd.DataFrame(instance_rows(results))
        phase_chunks.to_parquet(phase_dir / "qa_chunk_dataset.parquet", index=False)
        phase_instances.to_parquet(phase_dir / "qa_instances.parquet", index=False)
        self.progress.log("Phase 1 tables written", instances=len(results), rows=len(phase_chunks))

        self.progress.stage("split", "document roles and frozen five-fold CV")
        records = _records_from_results(results, tables["gold_mentions"])
        roles = split_document_ids(records, seed=self.config.seed, fractions=self.config.fractions)
        assigned = apply_document_roles(records, roles, official_dev=())
        unlocked = tuple(item for item in assigned if item.split in UNLOCKED_SPLITS)
        locked = tuple(item for item in assigned if item.split == "internal_test")
        sanity_errors = _sanity(assigned, self.config.aspects)
        split_summary = split_sanity(assigned)
        if sanity_errors:
            write_json(self.output / "sanity_errors.json", sanity_errors)
            raise RuntimeError(f"sanity checks failed: {sanity_errors[:5]}")
        data_dir = self.output / "data"
        locked_dir = self.output / "locked"
        write_records(data_dir / "dataset.jsonl", unlocked)
        write_records(locked_dir / "internal_test.jsonl", locked)
        write_json(data_dir / "splits_v2.json", {**roles, "seed": self.config.seed, "frozen": True})
        cv = build_cv_splits(
            unlocked, n_folds=self.config.n_folds, seed=self.config.seed,
            output_path=data_dir / "v3_cv_splits.json",
        )
        write_jsonl(data_dir / "hn_quality_flags.jsonl", hn_quality_rows(unlocked))
        summary = {
            "source_documents": len(documents),
            "documents_with_at_least_one_selected_aspect": len(doc_ids),
            "selected_aspects": list(self.config.aspects),
            "tag_counts": dict(Counter(span.tag for doc in documents for span in doc.spans)),
            "source_parse_warning_count": sum(len(doc.parse_warnings) for doc in documents),
            "preprocessing": {
                "chunks": len(chunk_rows), "aspect_instances_all": len(aspect_rows),
                "eligible_aspect_instances": len(eligible), "lexical_mentions": len(mention_rows),
            },
            "phase1_class_counts": dict(Counter(row.primary_class for result in results for row in result.chunks)),
            "split_summary": split_summary,
            "unlocked_instances": len(unlocked),
            "locked_instances": len(locked),
            "cv_folds": cv["n_folds"],
            "sanity_errors": 0,
            "elapsed_seconds": round(time.time() - started, 3),
            "model_profile": {
                "language": "ru", "qa_model": factory.config.profile.qa_model,
                "sentence_model": factory.config.profile.sentence_model, "device": factory.device,
            },
            "written_preprocessing_tables": written_tables,
        }
        write_json(data_dir / "data_summary.json", summary)
        write_json(self.output / "sanity_errors.json", [])
        write_json(self.output / "COMPLETED.json", {"status": "complete", **summary})
        self.progress.stage("complete", f"unlocked={len(unlocked)}, locked={len(locked)}")
        return summary
