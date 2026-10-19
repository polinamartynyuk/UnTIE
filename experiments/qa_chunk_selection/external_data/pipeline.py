from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import replace
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
from experiments.qa_chunk_selection.v3.data import write_records
from experiments.qa_chunk_selection.v3.split import build_cv_splits
from untie.aspect_qa.chunk_index import build_indexed_chunks, resolve_context_budget
from untie.aspect_qa.io import write_tables
from untie.aspect_qa.source import TextSegment
from untie.config import PipelineConfig
from untie.models import ModelFactory, profile_for_language
from untie.text import RussianSentenceSplitter

from .common import (
    FixedSentenceSplitter,
    SourceMention,
    chunks_fully_containing,
    claim_output,
    detokenize_with_offsets,
    mentions_from_token_spans,
    parse_inline_annotations,
    unique_normalized,
)
from .config import (
    SCIER_NATIVE_MAP,
    SCIER_Q5,
    SCIMDIX_NATIVE_MAP,
    SCIMDIX_Q5,
    ExternalDataConfig,
)
from .sources import materialize_source, sha256_path


PRIMARY_TO_EXPERIMENT = {
    "qa_positive": "positive",
    "evidence_positive_qa_failed": "ambiguous",
    "confirmed_false_negative": "ambiguous",
    "false_negative_candidate": "ambiguous",
    "hard_negative": "hard_negative",
    "easy_negative": "easy_negative",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _normalized_text_equal(left: str, right: str) -> bool:
    return " ".join(str(left).split()) == " ".join(str(right).split())


def _base_rows(
    *, document_id: str, split: str, text: str, chunks,
    mentions: Sequence[SourceMention], questions: dict[str, tuple[str, ...]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    document = {
        "document_id": document_id, "split": split, "raw_text": text, "clean_text": text,
        "n_chunks": len(chunks), "n_segments": 1,
    }
    chunk_rows = [{
        "document_id": document_id, "split": split, "chunk_id": c.chunk_id,
        "chunk_text": c.text, "char_start": c.char_start, "char_end": c.char_end,
        "token_start": c.token_start, "token_end": c.token_end,
        "section_type": c.section_type, "section_title": c.section_title,
        "eligible_for_qa": True, "segment_ids": list(c.segment_ids), "token_count": c.token_count,
    } for c in chunks]
    mention_rows: list[dict[str, Any]] = []
    for mention in mentions:
        containing = chunks_fully_containing(mention, chunks)
        canonical = unique_normalized([mention.text])
        if not canonical:
            raise ValueError(f"empty normalized mention: {document_id}/{mention.mention_id}")
        mention_rows.append({
            "document_id": document_id, "split": split, "aspect": mention.aspect,
            "native_type": mention.native_type, "canonical_answer": canonical[0],
            "raw_answer": mention.text, "mention_id": mention.mention_id,
            "matched_text": mention.text, "char_start": mention.char_start,
            "char_end": mention.char_end, "token_start": mention.token_start,
            "token_end": mention.token_end, "chunk_ids": [c.chunk_id for c in containing],
            "in_bibliography": False,
        })
    aspect_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    for aspect, inventory in questions.items():
        targets = [m for m in mentions if m.aspect == aspect]
        supporting = sorted({c.chunk_id for m in targets for c in chunks_fully_containing(m, chunks)})
        answers = list(unique_normalized(m.text for m in targets))
        status = "ready" if answers and supporting else "no_extractable_gold"
        aspect_rows.append({
            "document_id": document_id, "split": split, "aspect": aspect,
            "question": inventory[0], "status": status,
            "raw_gold_answers": [m.text for m in targets], "canonical_gold_answers": answers,
            "extractable_gold_answers": answers if status == "ready" else [],
            "unextractable_gold_answers": answers if answers and not supporting else [],
            "external_enrichment": [], "enrichment_candidates": [],
            "chunk_ids": [c.chunk_id for c in chunks], "gold_supporting_chunk_ids": supporting,
        })
        if status == "ready":
            support_set = set(supporting)
            for chunk in chunks:
                qa_rows.append({
                    "document_id": document_id, "split": split, "aspect": aspect,
                    "question": inventory[0], "status": status, "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.text, "char_start": chunk.char_start,
                    "char_end": chunk.char_end, "extractable_gold_answers": answers,
                    "gold_supporting": chunk.chunk_id in support_set, "enrichment_mode": "E0",
                })
    return document, chunk_rows, aspect_rows, mention_rows, qa_rows


def _scimdix_documents(source: Path, limit_docs: int) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for split in ("train", "test"):
        aspects = pd.read_parquet(source / f"aspects_{split}_ru.parquet")
        ner_re = pd.read_parquet(source / f"ner_re_{split}_ru.parquet").set_index("filename")
        if len(aspects) != (186 if split == "train" else 20):
            raise ValueError(f"unexpected SciMDIX {split} size: {len(aspects)}")
        if set(aspects.filename) != set(ner_re.index):
            raise ValueError(f"SciMDIX aspects and ner_re filename mismatch in {split}")
        if limit_docs > 0:
            aspects = aspects.iloc[:limit_docs]
        for row in aspects.itertuples(index=False):
            text, mentions = parse_inline_annotations(row.aspect_annotation, SCIMDIX_NATIVE_MAP)
            auxiliary = ner_re.loc[row.filename]
            documents.append({
                "document_id": f"scimdix:{row.filename}", "source_id": str(row.filename),
                "split": split, "text": text, "mentions": mentions,
                "domain": str(row.filename).split("-", 1)[0], "source_abstract": str(row.abstract),
                "aspect_text_matches_abstract": _normalized_text_equal(text, row.abstract),
                "ner_re_text_matches_aspect": _normalized_text_equal(auxiliary.abstract, text),
                "raw_ner_entities": auxiliary.entities, "raw_ner_relations": auxiliary.relations,
            })
    return documents


def _scier_documents(source: Path, limit_docs: int) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for split, expected in (("train", 80), ("dev", 10), ("test", 10), ("test_ood", 6)):
        rows = _read_jsonl(source / f"{split}.jsonl")
        if len(rows) != expected:
            raise ValueError(f"unexpected SciER {split} size: {len(rows)}")
        if limit_docs > 0:
            rows = rows[:limit_docs]
        for row in rows:
            tokens = [str(token) for token in row["doc_tokens"]]
            flat = [str(token) for sentence in row["sentences"] for token in sentence]
            if tokens != flat:
                raise ValueError(f"SciER doc_tokens/sentences mismatch: {row['doc_id']}")
            text, offsets = detokenize_with_offsets(tokens)
            mentions = mentions_from_token_spans(row["ner"], tokens, offsets, SCIER_NATIVE_MAP)
            sentence_texts: list[str] = []
            cursor = 0
            for sentence in row["sentences"]:
                count = len(sentence)
                if count:
                    sentence_texts.append(text[offsets[cursor][0]:offsets[cursor + count - 1][1]])
                cursor += count
            if cursor != len(tokens):
                raise ValueError(f"SciER sentence boundary drift: {row['doc_id']}")
            documents.append({
                "document_id": f"scier:{split}:{row['doc_id']}", "source_id": str(row["doc_id"]),
                "split": split, "text": text, "tokens": tokens, "offsets": offsets,
                "sentences": sentence_texts, "mentions": mentions, "relations": row["relations"],
            })
    return documents


def _relation_rows_scier(document: dict[str, Any]) -> list[dict[str, Any]]:
    tokens, offsets = document["tokens"], document["offsets"]
    rows: list[dict[str, Any]] = []
    for sentence_index, relations in enumerate(document["relations"]):
        for relation_index, relation in enumerate(relations):
            s1, e1, s2, e2, relation_type = relation
            s1, e1, s2, e2 = int(s1), int(e1), int(s2), int(e2)
            if not (0 <= s1 <= e1 < len(tokens) and 0 <= s2 <= e2 < len(tokens)):
                raise ValueError(f"invalid SciER relation span: {document['document_id']}")
            left, _ = detokenize_with_offsets(tokens[s1:e1 + 1])
            right, _ = detokenize_with_offsets(tokens[s2:e2 + 1])
            rows.append({
                "document_id": document["document_id"], "split": document["split"],
                "relation_id": f"R{relation_index + 1}@S{sentence_index}",
                "relation_type": str(relation_type), "subject_text": left, "object_text": right,
                "subject_token_start": s1, "subject_token_end": e1 + 1,
                "object_token_start": s2, "object_token_end": e2 + 1,
                "subject_char_start": offsets[s1][0], "subject_char_end": offsets[e1][1],
                "object_char_start": offsets[s2][0], "object_char_end": offsets[e2][1],
            })
    return rows


def _load_models(language: str, dataset: str, config: ExternalDataConfig):
    chunk_max_tokens, overlap_tokens = config.chunking(dataset)
    pipeline_config = PipelineConfig(
        profile=profile_for_language(language), chunk_max_tokens=chunk_max_tokens,
        overlap_tokens=overlap_tokens, device=config.device,
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
    mention_index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in mentions.to_dict("records"):
        mention_index[(str(row["document_id"]), str(row["aspect"]))].append(row)
    records: list[InstanceRecord] = []
    for item in results:
        ordered = sorted(item.chunks, key=lambda chunk: chunk.chunk_id)
        ids = [chunk.chunk_id for chunk in ordered]
        chunks: list[ChunkRecord] = []
        for position, chunk in enumerate(ordered):
            primary = str(chunk.primary_class)
            if primary not in PRIMARY_TO_EXPERIMENT:
                raise ValueError(f"unknown Phase 1 class: {primary}")
            chunks.append(ChunkRecord(
                chunk_id=chunk.chunk_id, text=chunk.text,
                label=PRIMARY_TO_EXPERIMENT[primary], primary_class=primary,
                position=position, section_id=chunk.section_type or f"pos:{position}",
                section_title=chunk.section_title, char_start=chunk.char_start,
                char_end=chunk.char_end, token_start=chunk.token_start,
                token_end=chunk.token_end, gold_supporting=chunk.gold_supporting,
                previous_chunk_id=ids[position - 1] if position else None,
                next_chunk_id=ids[position + 1] if position + 1 < len(ids) else None,
                eligible_for_qa=chunk.eligible_for_qa,
            ))
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        spans: list[GoldSpan] = []
        for mention in mention_index.get((item.document_id, item.aspect), ()):
            start, end = int(mention["char_start"]), int(mention["char_end"])
            for raw_chunk_id in mention["chunk_ids"]:
                chunk = by_id.get(int(raw_chunk_id))
                if chunk is None or not (chunk.char_start <= start and end <= chunk.char_end):
                    continue
                rel_start, rel_end = start - chunk.char_start, end - chunk.char_start
                spans.append(GoldSpan(
                    chunk_id=chunk.chunk_id, start_char=rel_start, end_char=rel_end,
                    text=chunk.text[rel_start:rel_end], document_start_char=start,
                    document_end_char=end,
                ))
        records.append(InstanceRecord(
            document_id=item.document_id, question_id=item.qa_instance_id,
            aspect=item.aspect, question=item.question,
            gold_answers=tuple(item.extractable_gold_answers), split=item.split,
            chunks=tuple(chunks), gold_spans=tuple(spans),
            instance_status=item.instance_status, source_split=f"official_{item.split}",
        ))
    return tuple(records)


def _sanity(records: Sequence[InstanceRecord], expected_aspects: Sequence[str]) -> list[str]:
    errors: list[str] = []
    qids = [item.question_id for item in records]
    if len(qids) != len(set(qids)):
        errors.append("duplicate question_id")
    allowed = set(expected_aspects)
    for item in records:
        if item.aspect not in allowed:
            errors.append(f"unexpected aspect {item.question_id}: {item.aspect}")
        if not item.gold_answers or not item.gold_spans:
            errors.append(f"missing extractable gold {item.question_id}")
        for span in item.gold_spans:
            chunk = item.chunk_by_id(span.chunk_id)
            if chunk.text[span.start_char:span.end_char] != span.text:
                errors.append(f"gold span drift {item.question_id}:{span.chunk_id}")
        for chunk in item.hard_negatives:
            if chunk.gold_supporting or chunk.primary_class != "hard_negative":
                errors.append(f"polluted hard negative {item.question_id}:{chunk.chunk_id}")
    return errors


def _output_records(
    dataset_root: Path, records_by_split: dict[str, list[InstanceRecord]], seed: int, n_folds: int,
) -> dict[str, Any]:
    data_dir, heldout, locked = dataset_root / "data", dataset_root / "heldout", dataset_root / "locked"
    all_train = tuple(records_by_split.get("train", ()))
    answerable_train = tuple(item for item in all_train if item.positives)
    if not answerable_train:
        raise ValueError("no QA-answerable official training records")
    write_records(data_dir / "all_phase1_instances.jsonl", all_train)
    write_records(data_dir / "dataset.jsonl", answerable_train)
    write_records(data_dir / "answerable_dataset.jsonl", answerable_train)
    write_jsonl(data_dir / "hn_quality_flags.jsonl", hn_quality_rows(answerable_train))
    cv = build_cv_splits(
        answerable_train, n_folds=n_folds, seed=seed,
        output_path=data_dir / "v3_cv_splits.json",
    )
    write_json(data_dir / "answerable_cv_splits.json", cv)
    destinations: dict[str, str] = {
        "train_answerable": str(data_dir / "dataset.jsonl"),
        "train_all_phase1": str(data_dir / "all_phase1_instances.jsonl"),
    }
    if records_by_split.get("dev"):
        all_rows = tuple(replace(item, split="seen_exploratory_dev") for item in records_by_split["dev"])
        answerable = tuple(item for item in all_rows if item.positives)
        write_records(heldout / "official_dev_all.jsonl", all_rows)
        write_records(heldout / "official_dev.jsonl", answerable)
        destinations["dev"] = str(heldout / "official_dev.jsonl")
    for split in ("test", "test_ood"):
        if records_by_split.get(split):
            all_rows = tuple(replace(item, split=f"locked_official_{split}") for item in records_by_split[split])
            answerable = tuple(item for item in all_rows if item.positives)
            stem = "official_test_ood" if split == "test_ood" else "official_test"
            write_records(locked / f"{stem}_all.jsonl", all_rows)
            write_records(locked / f"{stem}.jsonl", answerable)
            destinations[split] = str(locked / f"{stem}.jsonl")
    return {
        "cv_folds": cv["n_folds"], "cv_documents": cv["n_documents"],
        "train_phase1_instances": len(all_train), "train_answerable_instances": len(answerable_train),
        "files": destinations,
    }


def _prepare_dataset(
    dataset: str, source: Path, output: Path, factory: ModelFactory,
    answer_fn, encode_fn, config: ExternalDataConfig, progress: ProgressLogger,
) -> dict[str, Any]:
    questions = SCIMDIX_Q5 if dataset == "scimdix" else SCIER_Q5
    chunk_max_tokens, overlap_tokens = config.chunking(dataset)
    documents = _scimdix_documents(source, config.limit_docs) if dataset == "scimdix" else _scier_documents(source, config.limit_docs)
    budget = resolve_context_budget(
        factory.tokenizer, [q for values in questions.values() for q in values],
        configured_max_tokens=chunk_max_tokens,
    )
    progress.log(f"{dataset} source validated", documents=len(documents), chunk_budget=budget)
    document_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    aspect_rows: list[dict[str, Any]] = []
    mention_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    for index, document in enumerate(documents, start=1):
        segment = TextSegment(0, "", "", document["text"], 0, len(document["text"]))
        splitter = RussianSentenceSplitter() if dataset == "scimdix" else FixedSentenceSplitter(document["sentences"])
        chunks = build_indexed_chunks(
            document["text"], factory.tokenizer, max_tokens=budget,
            overlap_tokens=overlap_tokens, segments=(segment,),
            include_bibliography_in_qa=True, splitter=splitter,
        )
        rows = _base_rows(
            document_id=document["document_id"], split=document["split"],
            text=document["text"], chunks=chunks, mentions=document["mentions"],
            questions=questions,
        )
        document_rows.append(rows[0]); chunk_rows.extend(rows[1]); aspect_rows.extend(rows[2])
        mention_rows.extend(rows[3]); qa_rows.extend(rows[4])
        metadata_rows.append({
            "document_id": document["document_id"], "source_id": document["source_id"],
            "split": document["split"], "dataset": dataset,
            "domain": document.get("domain", "computer_science"),
            "native_type_counts": dict(Counter(m.native_type for m in document["mentions"])),
            "aspect_counts": dict(Counter(m.aspect for m in document["mentions"])),
            "source_abstract": document.get("source_abstract"),
            "aspect_text_matches_abstract": document.get("aspect_text_matches_abstract"),
            "ner_re_text_matches_aspect": document.get("ner_re_text_matches_aspect"),
        })
        if dataset == "scier":
            relation_rows.extend(_relation_rows_scier(document))
        progress.count(index, len(documents), item=f"{dataset} documents chunked", extra={"chunks": len(chunk_rows)})

    prep = output / "preprocessing"
    written = write_tables(
        prep, documents=document_rows, chunks=chunk_rows, aspect_instances=aspect_rows,
        gold_mentions=mention_rows, source_metadata=metadata_rows, qa_input=qa_rows,
    )
    annotations = output / "annotations"
    _write_frame(pd.DataFrame(mention_rows), annotations / "aspect_or_entity_mentions.parquet")
    if dataset == "scier":
        _write_frame(pd.DataFrame(relation_rows), annotations / "relations.parquet")
        native_entity_count = len(mention_rows)
        native_relation_count = len(relation_rows)
    else:
        raw_rows = []
        for document in documents:
            raw_rows.append({
                "document_id": document["document_id"], "split": document["split"],
                "entities": document["raw_ner_entities"], "relations": document["raw_ner_relations"],
            })
        _write_frame(pd.DataFrame(raw_rows), annotations / "ner_re_raw.parquet")
        native_entity_count = sum(
            len([line for line in str(document["raw_ner_entities"]).splitlines() if line.strip()])
            for document in documents
        )
        native_relation_count = sum(
            len([line for line in str(document["raw_ner_relations"]).splitlines() if line.strip()])
            for document in documents
        )
    progress.log(f"{dataset} preprocessing tables written", chunks=len(chunk_rows), mentions=len(mention_rows))

    tables = {
        "chunks": pd.DataFrame(chunk_rows), "aspect_instances": pd.DataFrame(aspect_rows),
        "gold_mentions": pd.DataFrame(mention_rows),
    }
    eligible = tables["aspect_instances"][
        tables["aspect_instances"]["status"].isin(("ready", "partially_extractable"))
    ].reset_index(drop=True)
    phase_config = Phase1Config(
        input_dir=prep, output_dir=output / "phase1", split="train",
        seed=config.seed, device=config.device,
        hard_negative_top_n=config.hard_negative_top_n,
        gap_bootstrap=config.gap_bootstrap,
    )
    results = []
    doc_ids = list(dict.fromkeys(eligible["document_id"].astype(str).tolist()))
    progress.stage(f"phase1:{dataset}", "existing QA/BM25/semantic chunk labelling")
    for index, document_id in enumerate(doc_ids, start=1):
        doc_instances = eligible[eligible["document_id"] == document_id]
        doc_chunks = tables["chunks"][tables["chunks"]["document_id"] == document_id].sort_values("chunk_id")
        results.extend(process_document_instances(
            document_id=document_id, instance_rows=doc_instances, chunk_rows=doc_chunks,
            mentions=tables["gold_mentions"], config=phase_config,
            answer_fn=answer_fn, encode_fn=encode_fn,
        ))
        progress.count(index, len(doc_ids), item=f"{dataset} documents labelled", extra={"instances": len(results)})
    phase_dir = output / "phase1"
    phase_dir.mkdir(parents=True, exist_ok=False)
    phase_chunks = pd.DataFrame(chunk_dataset_rows(results))
    phase_instances = pd.DataFrame(instance_rows(results))
    _write_frame(phase_chunks, phase_dir / "qa_chunk_dataset.parquet")
    _write_frame(phase_instances, phase_dir / "qa_instances.parquet")
    phase_payload = phase_config.to_dict()
    phase_payload["upstream_default_aspects"] = phase_payload.get("aspects", [])
    phase_payload["aspects"] = list(questions)
    phase_payload["processed_splits"] = sorted(set(eligible["split"].astype(str)))
    write_json(phase_dir / "config.json", phase_payload)

    records = _records_from_results(results, tables["gold_mentions"])
    errors = _sanity(records, questions)
    if errors:
        write_json(output / "sanity_errors.json", errors)
        raise RuntimeError(f"{dataset} sanity checks failed: {errors[:5]}")
    by_split: dict[str, list[InstanceRecord]] = defaultdict(list)
    for record in records:
        by_split[record.split].append(record)
    ids_by_split = {split: {r.document_id for r in rows} for split, rows in by_split.items()}
    split_names = sorted(ids_by_split)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1:]:
            if ids_by_split[left] & ids_by_split[right]:
                raise ValueError(f"document leakage {dataset}: {left}/{right}")
    split_output = _output_records(output, by_split, config.seed, config.n_folds)
    answerable = [record for record in records if record.positives]
    no_positive = [record for record in records if not record.positives]
    summary = {
        "dataset": dataset, "source_documents": len(documents),
        "documents_by_split": dict(Counter(d["split"] for d in documents)),
        "phase1_instances": len(records), "answerable_instances": len(answerable),
        "no_qa_positive_instances": len(no_positive),
        "instances_by_split": {key: len(value) for key, value in by_split.items()},
        "answerable_by_split": dict(Counter(r.split for r in answerable)),
        "aspects": dict(Counter(r.aspect for r in records)),
        "unique_chunks": len(chunk_rows),
        "chunks_per_document": {
            "min": min((row["n_chunks"] for row in document_rows), default=0),
            "median": statistics.median((row["n_chunks"] for row in document_rows)) if document_rows else 0,
            "max": max((row["n_chunks"] for row in document_rows), default=0),
        },
        "phase1_class_counts": dict(Counter(c.primary_class for r in records for c in r.chunks)),
        "gold_mentions": len(mention_rows),
        "native_entity_mentions_preserved": native_entity_count,
        "relations_preserved": native_relation_count,
        "chunk_max_tokens": budget, "overlap_tokens": overlap_tokens,
        "labelling_method": "existing Phase 1: extractive QA gold matching, BM25 top-N, semantic FN filtering",
        "hard_negative_top_n": config.hard_negative_top_n,
        "cv_answerable_only": True, "official_test_excluded_from_cv": True,
        "written_preprocessing_tables": written, "experiment_files": split_output,
        "model_profile": {
            "language": "ru" if dataset == "scimdix" else "en",
            "qa_model": factory.config.profile.qa_model,
            "sentence_model": factory.config.profile.sentence_model,
            "device": factory.device,
        },
        "sanity_errors": 0,
    }
    write_json(output / "data_summary.json", summary)
    write_json(output / "questions_q5.json", {k: list(v) for k, v in questions.items()})
    write_json(output / "sanity_errors.json", [])
    write_json(output / "COMPLETED.json", {"status": "complete", **summary})
    return summary


class ExternalDataPipeline:
    def __init__(self, config: ExternalDataConfig) -> None:
        config.validate()
        self.config = config
        self.output = claim_output(config.output_dir)
        self.progress = ProgressLogger(self.output / "progress.jsonl")

    def run(self) -> dict[str, Any]:
        started = time.time()
        write_json(self.output / "config.json", self.config.to_dict())
        summaries: dict[str, Any] = {}
        manifests: dict[str, Any] = {}
        for dataset in self.config.datasets:
            self.progress.stage(f"source:{dataset}", "freeze official source in new artifact")
            local = self.config.scimdix_source if dataset == "scimdix" else self.config.scier_source
            if self.config.download:
                local = None
            source = self.output / "source_raw" / dataset
            manifests[dataset] = materialize_source(dataset, source, local_source=local, progress=self.progress.log)
            language = "ru" if dataset == "scimdix" else "en"
            self.progress.stage(f"models:{dataset}", f"{language} tokenizer, extractive QA and sentence encoder")
            factory, answer_fn, encode_fn = _load_models(language, dataset, self.config)
            self.progress.log(
                f"{dataset} models ready", qa_model=factory.config.profile.qa_model,
                sentence_model=factory.config.profile.sentence_model, device=factory.device,
            )
            self.progress.stage(f"preprocess:{dataset}", "offset-safe chunks and official gold spans")
            summaries[dataset] = _prepare_dataset(
                dataset, source, self.output / dataset, factory, answer_fn, encode_fn,
                self.config, self.progress,
            )
        write_json(self.output / "source_manifest.json", manifests)
        payload = {
            "status": "complete", "datasets": list(self.config.datasets),
            "elapsed_seconds": round(time.time() - started, 3), "summaries": summaries,
            "source_manifest_sha256": sha256_path(self.output / "source_manifest.json"),
        }
        write_json(self.output / "COMPLETED.json", payload)
        self.progress.stage("complete", f"datasets={','.join(self.config.datasets)}")
        return payload
