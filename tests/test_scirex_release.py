from __future__ import annotations

import json
from pathlib import Path

from untie.scirex_release import (
    detokenize_words,
    extract_clean_row,
    prepare_official_structured_splits,
    write_structured_jsonl,
)


def test_detokenize_joins_punctuation_without_spaces() -> None:
    text = detokenize_words(["Hello", ",", "world", "!", "(", "x", ")", "."])
    assert text == "Hello, world! (x)."


def test_extract_clean_row_keeps_unique_order_and_seven_fields() -> None:
    doc = {
        "doc_id": "abc",
        "words": ["section", ":", "Abstract", "FRRN", "on", "Cityscapes", "."],
        "n_ary_relations": [
            {"Method": "FRRN", "Task": "Semantic_Segmentation", "score": "71.8%"},
            {"Method": "FRRN", "Material": "Cityscapes", "Metric": "mIoU"},
            {"Method": "ResNet", "Task": "Semantic_Segmentation"},
        ],
    }
    row = extract_clean_row(doc)
    assert list(row) == [
        "doc_id",
        "methods",
        "tasks",
        "metrics",
        "materials",
        "score",
        "original_text",
    ]
    assert row["methods"] == ["FRRN", "ResNet"]
    assert row["tasks"] == ["Semantic_Segmentation"]
    assert row["metrics"] == ["mIoU"]
    assert row["materials"] == ["Cityscapes"]
    assert row["score"] == ["71.8%"]
    assert "FRRN" in row["original_text"]
    assert "section:" in row["original_text"]


def test_prepare_skips_existing_train_and_writes_dev_test(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "scirex_structured.json").write_text(
        json.dumps({"doc_id": "keep-me", "methods": [], "tasks": [], "metrics": [], "materials": [], "score": [], "original_text": "x"})
        + "\n",
        encoding="utf-8",
    )
    for split, doc_id in (("train", "train-id"), ("dev", "dev-id"), ("test", "test-id")):
        payload = {
            "doc_id": doc_id,
            "words": ["Hello", "world", "."],
            "n_ary_relations": [{"Method": "M", "Task": "T", "Metric": "Acc", "Material": "D", "score": "1"}],
        }
        (release / f"{split}.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    report = prepare_official_structured_splits(
        datasets_dir=datasets,
        release_dir=release,
        skip_existing=("train",),
        download=False,
    )
    assert report["splits"]["train"]["skipped"] is True
    train_text = (datasets / "scirex_structured.json").read_text(encoding="utf-8")
    assert "keep-me" in train_text
    assert "train-id" not in train_text
    dev = json.loads((datasets / "scirex_dev_structured.json").read_text(encoding="utf-8").splitlines()[0])
    test = json.loads((datasets / "scirex_test_structured.json").read_text(encoding="utf-8").splitlines()[0])
    assert dev["doc_id"] == "dev-id"
    assert test["doc_id"] == "test-id"
    assert dev["original_text"] == "Hello world."
    assert set(dev) == {"doc_id", "methods", "tasks", "metrics", "materials", "score", "original_text"}


def test_write_structured_jsonl_is_one_record_per_line(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    write_structured_jsonl(
        [
            {
                "doc_id": "a",
                "methods": ["M"],
                "tasks": [],
                "metrics": [],
                "materials": [],
                "score": [],
                "original_text": "Hello",
            }
        ],
        path,
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["methods"] == ["M"]
