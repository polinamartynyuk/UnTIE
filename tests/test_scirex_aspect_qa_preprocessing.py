from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from untie.aspect_qa.chunk_index import build_indexed_chunks
from untie.aspect_qa.cli import PROJECT_ROOT, WhitespaceTokenizer, main
from untie.aspect_qa.config import AspectQAPreprocessConfig, default_output_dir
from untie.aspect_qa.invariants import validate_processed_document
from untie.aspect_qa.io import write_json
from untie.aspect_qa.mentions import find_lexical_mentions
from untie.aspect_qa.normalize import deduplicate_entities, identity_key, normalize_entity
from untie.aspect_qa.pipeline import (
    assign_splits,
    process_aspect,
    process_document,
    qa_input_rows,
)
from untie.aspect_qa.relabel import relabel_preprocessing_as_official_train
from untie.aspect_qa.source import inspect_source, parse_structural_segments
from untie.config import PipelineConfig
from untie.chunking import ChunkBuilder
from tests.test_core import FakeTokenizer


def _cfg(**kwargs) -> AspectQAPreprocessConfig:
    return AspectQAPreprocessConfig(overlap_tokens=2, **kwargs)


def _pipe() -> PipelineConfig:
    return PipelineConfig()


def _read_written_table(output: Path, name: str) -> pd.DataFrame:
    parquet = output / f"{name}.parquet"
    jsonl = output / f"{name}.jsonl"
    if parquet.exists():
        return pd.read_parquet(parquet)
    return pd.read_json(jsonl, orient="records", lines=True)


def test_underscores_become_single_space() -> None:
    assert normalize_entity("Text_Classification") == "Text Classification"
    assert normalize_entity("Yahoo__Answers") == "Yahoo Answers"


def test_duplicate_case_variants_share_identity_and_keep_raw() -> None:
    entities = deduplicate_entities(["fastText", "FastText"])
    assert len(entities) == 1
    assert entities[0].identity_key == identity_key("fastText") == identity_key("FastText")
    assert entities[0].raw_variants == ("fastText", "FastText")
    assert "fasttext" == entities[0].identity_key


def test_absent_entity_is_not_extractable() -> None:
    text = "The paper discusses neural networks in general."
    chunks = build_indexed_chunks(
        text,
        FakeTokenizer(),
        max_tokens=20,
        overlap_tokens=2,
        segments=parse_structural_segments(text),
        include_bibliography_in_qa=False,
    )
    instance, mentions = process_aspect(
        document_id="doc-absent",
        aspect="method",
        question="What method or model is used or proposed in the paper?",
        raw_values=["DefinitelyMissingMethodXYZ"],
        text=text,
        chunks=chunks,
        segments=parse_structural_segments(text),
        include_bibliography_in_qa=False,
    )
    assert instance.status == "no_extractable_gold"
    assert instance.extractable_gold_answers == []
    assert instance.unextractable_gold_answers == ["DefinitelyMissingMethodXYZ"]
    assert mentions == []
    assert instance.gold_supporting_chunk_ids == []


def test_partial_extractability_status() -> None:
    text = "We report Accuracy on the benchmark. No other metric is named."
    chunks = build_indexed_chunks(
        text,
        FakeTokenizer(),
        max_tokens=30,
        overlap_tokens=2,
        segments=parse_structural_segments(text),
        include_bibliography_in_qa=False,
    )
    instance, _ = process_aspect(
        document_id="doc-partial",
        aspect="metric",
        question="What evaluation metric is used in the paper?",
        raw_values=["Accuracy", "Error"],
        text=text,
        chunks=chunks,
        segments=parse_structural_segments(text),
        include_bibliography_in_qa=False,
    )
    assert instance.status == "partially_extractable"
    assert instance.extractable_gold_answers == ["Accuracy"]
    assert instance.unextractable_gold_answers == ["Error"]


def test_overlapping_chunks_share_mention_id() -> None:
    text = "alpha beta gamma delta. Accuracy is reported here. epsilon zeta eta theta."
    tokenizer = FakeTokenizer()
    chunks = build_indexed_chunks(
        text,
        tokenizer,
        max_tokens=8,
        overlap_tokens=3,
        segments=parse_structural_segments(text),
        include_bibliography_in_qa=False,
    )
    instance, mentions = process_aspect(
        document_id="doc-overlap",
        aspect="metric",
        question="What evaluation metric is used in the paper?",
        raw_values=["Accuracy"],
        text=text,
        chunks=chunks,
        segments=parse_structural_segments(text),
        include_bibliography_in_qa=False,
    )
    assert len(mentions) == 1
    mention_id = mentions[0]["mention_id"]
    covering = mentions[0]["chunk_ids"]
    assert len(covering) >= 2
    for chunk_id in covering:
        assert chunk_id in instance.gold_supporting_chunk_ids
    builder = ChunkBuilder(tokenizer, max_tokens=8, overlap_tokens=3)
    rebuilt = builder.build(builder.prepare_sentences(["alpha beta gamma delta.", "Accuracy is reported here.", "epsilon zeta eta theta."]))
    assert len(rebuilt) >= 2
    assert all(mid == mention_id for mid in [mentions[0]["mention_id"]])


def test_mention_offsets_recover_matched_text() -> None:
    text = "The model FastText is trained on Sentiment Analysis."
    mentions = find_lexical_mentions(
        text,
        canonical="FastText",
        raw_variants=("fastText", "FastText"),
        mention_prefix="doc:method:fasttext",
    )
    assert mentions
    for mention in mentions:
        assert text[mention.char_start : mention.char_end] == mention.matched_text
        assert mention.matched_text.casefold() == "fasttext"


def test_bibliography_is_kept_but_excluded_from_qa() -> None:
    text = (
        "document: A Paper section: Introduction We introduce UniqueBodyMethod here. "
        "This sentence pads the body so bibliography lands in a later chunk. "
        "Another padding sentence about unrelated background material. "
        "bibliography: References UniqueBodyMethod and BibOnlyDataset 1999. "
        "Later citations continue without the paper body. "
        "Final bibliography sentence remains inside references."
    )
    tokenizer = FakeTokenizer()
    processed = process_document(
        {
            "doc_id": "doc-bib",
            "methods": ["UniqueBodyMethod"],
            "tasks": ["Some_Task"],
            "metrics": ["Accuracy"],
            "materials": ["BibOnlyDataset"],
            "score": ["1.0"],
            "original_text": text,
        },
        tokenizer=tokenizer,
        config=_cfg(),
        pipeline_config=_pipe(),
        chunk_max_tokens=10,
        split="train",
    )
    validate_processed_document(processed)
    assert any(segment.type == "bibliography" for segment in processed.segments)
    assert any(not chunk.eligible_for_qa for chunk in processed.chunks)
    methods = next(item for item in processed.instances if item.aspect == "method")
    materials = next(item for item in processed.instances if item.aspect == "material")
    assert methods.status in {"ready", "partially_extractable"}
    assert "UniqueBodyMethod" in methods.extractable_gold_answers
    assert materials.status == "no_extractable_gold"
    assert "BibOnlyDataset" in materials.unextractable_gold_answers
    qa_rows = qa_input_rows(processed)
    assert all(row["aspect"] != "material" for row in qa_rows)
    assert all(row["chunk_text"] for row in qa_rows)
    assert not any("bibliography:" in row["chunk_text"] and row["aspect"] == "material" for row in qa_rows)


def test_no_extractable_gold_does_not_create_negative_only_qa() -> None:
    processed = process_document(
        {
            "doc_id": "doc-none",
            "methods": ["MissingMethodZZZ"],
            "tasks": ["MissingTaskZZZ"],
            "metrics": ["MissingMetricZZZ"],
            "materials": ["MissingMaterialZZZ"],
            "score": ["9.9"],
            "original_text": "section: Introduction This document never names the gold entities.",
        },
        tokenizer=FakeTokenizer(),
        config=_cfg(),
        pipeline_config=_pipe(),
        chunk_max_tokens=40,
        split="dev",
    )
    validate_processed_document(processed)
    assert all(item.status == "no_extractable_gold" for item in processed.instances)
    assert qa_input_rows(processed) == []


def test_parser_does_not_add_external_names() -> None:
    processed = process_document(
        {
            "doc_id": "doc-bert",
            "methods": ["BERT"],
            "tasks": ["Text_Classification"],
            "metrics": ["Accuracy"],
            "materials": ["Yelp"],
            "score": ["90"],
            "original_text": "section: Introduction BERT is used for Text Classification with Accuracy on Yelp.",
        },
        tokenizer=FakeTokenizer(),
        config=_cfg(),
        pipeline_config=_pipe(),
        chunk_max_tokens=40,
        split="train",
    )
    validate_processed_document(processed)
    methods = next(item for item in processed.instances if item.aspect == "method")
    assert methods.canonical_gold_answers == ["BERT"]
    assert "SciBERT" not in methods.canonical_gold_answers
    assert "Bidirectional Encoder Representations from Transformers" not in methods.raw_gold_answers
    assert methods.external_enrichment == []
    tasks = next(item for item in processed.instances if item.aspect == "task")
    assert tasks.canonical_gold_answers == ["Text Classification"]


def test_score_is_metadata_only() -> None:
    processed = process_document(
        {
            "doc_id": "doc-score",
            "methods": ["FRRN"],
            "tasks": ["Semantic_Segmentation"],
            "metrics": ["Mean_IoU"],
            "materials": ["Cityscapes"],
            "score": ["71.8%", "2.1"],
            "original_text": "section: Abstract FRRN reaches 71.8% Mean IoU on Cityscapes for Semantic Segmentation.",
        },
        tokenizer=FakeTokenizer(),
        config=_cfg(),
        pipeline_config=_pipe(),
        chunk_max_tokens=40,
        split="test",
    )
    validate_processed_document(processed)
    assert [item.aspect for item in processed.instances] == ["method", "task", "metric", "material"]
    assert all(item.aspect != "score" for item in processed.instances)
    assert list(processed.record.score) == ["71.8%", "2.1"]


def test_source_inspection_of_real_scirex() -> None:
    path = Path("datasets/scirex_structured.json")
    inspection = inspect_source(path)
    assert inspection.n_records == 306
    assert inspection.format_name == "jsonl"
    assert inspection.schema_matches_expected
    assert list(inspection.first_record_fields) == [
        "doc_id",
        "methods",
        "tasks",
        "metrics",
        "materials",
        "score",
        "original_text",
    ]


def test_whitespace_tokenizer_protocol_matches_chunk_builder() -> None:
    tokenizer = WhitespaceTokenizer()
    chunks = build_indexed_chunks(
        "Hello world. Accuracy follows.",
        tokenizer,
        max_tokens=20,
        overlap_tokens=2,
        segments=parse_structural_segments("Hello world. Accuracy follows."),
        include_bibliography_in_qa=False,
    )
    assert chunks
    assert chunks[0].text.startswith("Hello")


def test_assign_splits_fixed_split_overrides_hash() -> None:
    ids = ["doc-a", "doc-b", "doc-c"]
    hashed = assign_splits(ids, AspectQAPreprocessConfig())
    assert set(hashed.values()) <= {"train", "dev", "test"}
    assert len(set(hashed.values())) > 1
    fixed = assign_splits(ids, AspectQAPreprocessConfig(fixed_split="dev"))
    assert fixed == {item: "dev" for item in ids}


def test_cli_refuses_official_dev_into_train_output_dir() -> None:
    code = main(
        [
            "--input",
            "datasets/scirex_structured.json",
            "--output",
            str(default_output_dir(PROJECT_ROOT)),
            "--fixed-split",
            "dev",
            "--inspect-only",
        ]
    )
    assert code == 2


def test_cli_fixed_split_assigns_every_document(tmp_path: Path) -> None:
    source = tmp_path / "mini.json"
    records = [
        {
            "doc_id": "doc-a",
            "methods": ["FRRN"],
            "tasks": ["Semantic Segmentation"],
            "metrics": ["mIoU"],
            "materials": ["Cityscapes"],
            "score": ["71.8%"],
            "original_text": "section: Abstract FRRN reaches 71.8% Mean IoU on Cityscapes.",
        },
        {
            "doc_id": "doc-b",
            "methods": ["V-Net"],
            "tasks": ["Segmentation"],
            "metrics": ["Dice"],
            "materials": ["PROMISE"],
            "score": ["0.869"],
            "original_text": "section: Abstract V-Net reports Dice on PROMISE.",
        },
    ]
    source.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    output = tmp_path / "qa-out"
    code = main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--tokenizer",
            "whitespace",
            "--fixed-split",
            "test",
            "--manual-review",
            "1",
        ]
    )
    assert code == 0
    documents = _read_written_table(output, "documents")
    instances = _read_written_table(output, "aspect_instances")
    qa_input = _read_written_table(output, "qa_input")
    assert len(documents) == 2
    assert set(documents["split"].unique()) == {"test"}
    assert set(instances["split"].unique()) == {"test"}
    assert set(qa_input["split"].unique()) == {"test"}


def test_relabel_moves_all_rows_to_train_without_dropping(tmp_path: Path) -> None:
    documents = pd.DataFrame(
        [
            {"document_id": "d1", "split": "train"},
            {"document_id": "d2", "split": "dev"},
            {"document_id": "d3", "split": "test"},
        ]
    )
    instances = pd.DataFrame(
        [
            {"document_id": "d1", "aspect": "method", "split": "train"},
            {"document_id": "d2", "aspect": "task", "split": "dev"},
            {"document_id": "d3", "aspect": "metric", "split": "test"},
        ]
    )
    qa_input = pd.DataFrame(
        [
            {"document_id": "d1", "chunk_id": 0, "split": "train"},
            {"document_id": "d2", "chunk_id": 0, "split": "dev"},
        ]
    )
    metadata = pd.DataFrame(
        [
            {"document_id": "d1", "split": "train"},
            {"document_id": "d2", "split": "dev"},
            {"document_id": "d3", "split": "test"},
        ]
    )
    documents.to_json(tmp_path / "documents.jsonl", orient="records", lines=True, force_ascii=False)
    instances.to_json(tmp_path / "aspect_instances.jsonl", orient="records", lines=True, force_ascii=False)
    qa_input.to_json(tmp_path / "qa_input.jsonl", orient="records", lines=True, force_ascii=False)
    metadata.to_json(tmp_path / "source_metadata.jsonl", orient="records", lines=True, force_ascii=False)
    pd.DataFrame([{"document_id": "d1", "chunk_id": 0}]).to_json(
        tmp_path / "chunks.jsonl", orient="records", lines=True, force_ascii=False
    )
    pd.DataFrame([{"document_id": "d1", "mention_id": 0}]).to_json(
        tmp_path / "gold_mentions.jsonl", orient="records", lines=True, force_ascii=False
    )
    pd.DataFrame([{"document_id": "d1"}]).to_json(
        tmp_path / "enrichment.jsonl", orient="records", lines=True, force_ascii=False
    )

    aspect_block = {
        "method": {
            "documents": 3,
            "raw_gold_entities": 1,
            "unique_gold_entities": 1,
            "extractable_gold_entities": 1,
            "unextractable_gold_entities": 0,
            "ready_instances": 1,
            "partially_extractable_instances": 0,
            "no_extractable_gold_instances": 0,
            "gold_supporting_chunks": 1,
            "extractability_rate": 1.0,
        }
    }
    write_json(
        tmp_path / "statistics.json",
        {
            "train": {"method": {**aspect_block["method"], "documents": 1}},
            "dev": {"method": {**aspect_block["method"], "documents": 1}},
            "test": {"method": {**aspect_block["method"], "documents": 1}},
            "all": aspect_block,
            "n_documents": 3,
            "n_chunks": 1,
            "n_qa_input_rows": 2,
        },
    )
    write_json(tmp_path / "preprocess_config.json", {"n_selected": 3, "source": "x"})

    report = relabel_preprocessing_as_official_train(tmp_path)
    assert report["row_counts"]["documents"] == 3
    assert report["row_counts"]["qa_input"] == 2
    assert report["legacy_internal_split_counts"] == {"train": 1, "dev": 1, "test": 1}

    relabeled = _read_written_table(tmp_path, "documents")
    assert list(relabeled["split"]) == ["train", "train", "train"]
    assert list(relabeled["legacy_internal_split"]) == ["train", "dev", "test"]
    stats = json.loads((tmp_path / "statistics.json").read_text(encoding="utf-8"))
    assert stats["train"]["method"]["documents"] == 3
    assert stats["dev"]["method"]["documents"] == 0
    assert stats["legacy_internal_split"]["document_counts"] == {"train": 1, "dev": 1, "test": 1}
    again = relabel_preprocessing_as_official_train(tmp_path)
    assert again["already_relabeled"] is True
    assert list(_read_written_table(tmp_path, "documents")["legacy_internal_split"]) == [
        "train",
        "dev",
        "test",
    ]
