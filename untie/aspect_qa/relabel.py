"""Relabel existing aspect-QA tables as official SciREX train without dropping rows."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ASPECT_FIELD_MAP
from .io import write_json

TABLES_WITH_SPLIT = ("documents", "aspect_instances", "qa_input", "source_metadata")
COUNTED_TABLES = (
    "documents",
    "chunks",
    "aspect_instances",
    "gold_mentions",
    "source_metadata",
    "enrichment",
    "qa_input",
)
LEGACY_SPLIT_COLUMN = "legacy_internal_split"


def _table_path(output_dir: Path, name: str) -> Path:
    parquet = output_dir / f"{name}.parquet"
    jsonl = output_dir / f"{name}.jsonl"
    if parquet.exists():
        return parquet
    if jsonl.exists():
        return jsonl
    raise FileNotFoundError(f"missing preprocessing table: {parquet}")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_json(path, orient="records", lines=True)


def _write_table(frame: pd.DataFrame, path: Path) -> Path:
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
        return path
    frame.to_json(path, orient="records", lines=True, force_ascii=False)
    return path


def _zero_aspect_block(template: dict[str, Any]) -> dict[str, Any]:
    empty: dict[str, Any] = {}
    for aspect in ASPECT_FIELD_MAP:
        src = template.get(aspect) if isinstance(template, dict) else None
        if not isinstance(src, dict):
            src = {
                "documents": 0,
                "raw_gold_entities": 0,
                "unique_gold_entities": 0,
                "extractable_gold_entities": 0,
                "unextractable_gold_entities": 0,
                "ready_instances": 0,
                "partially_extractable_instances": 0,
                "no_extractable_gold_instances": 0,
                "gold_supporting_chunks": 0,
                "extractability_rate": 0.0,
            }
        zeroed = {key: (0.0 if key == "extractability_rate" else 0) for key in src}
        empty[aspect] = zeroed
    return empty


def relabel_preprocessing_as_official_train(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    row_counts: dict[str, int] = {}
    for name in COUNTED_TABLES:
        frame = _read_table(_table_path(output_dir, name))
        row_counts[name] = int(len(frame))

    split_counts: dict[str, int] = {}
    rewritten: dict[str, str] = {}
    already = True
    for name in TABLES_WITH_SPLIT:
        path = _table_path(output_dir, name)
        frame = _read_table(path)
        if "split" not in frame.columns:
            raise ValueError(f"{path} has no split column")
        n_before = len(frame)
        if LEGACY_SPLIT_COLUMN not in frame.columns:
            frame[LEGACY_SPLIT_COLUMN] = frame["split"]
            already = False
        if name == "documents":
            split_counts = {
                str(key): int(value)
                for key, value in frame[LEGACY_SPLIT_COLUMN].value_counts().to_dict().items()
            }
        if set(frame["split"].astype(str).unique()) != {"train"}:
            already = False
        frame["split"] = "train"
        if len(frame) != n_before:
            raise RuntimeError(f"row count changed while relabeling {name}")
        rewritten[name] = str(_write_table(frame, path))
        row_counts[name] = int(len(frame))

    for name in COUNTED_TABLES:
        after = int(len(_read_table(_table_path(output_dir, name))))
        if after != row_counts[name]:
            raise RuntimeError(f"row count changed for {name}: {row_counts[name]} -> {after}")

    stats_path = output_dir / "statistics.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if "legacy_internal_split" not in stats:
        stats["legacy_internal_split"] = {
            "train": deepcopy(stats.get("train", {})),
            "dev": deepcopy(stats.get("dev", {})),
            "test": deepcopy(stats.get("test", {})),
            "document_counts": split_counts,
        }
        all_block = deepcopy(stats.get("all", {}))
        stats["train"] = deepcopy(all_block)
        stats["dev"] = _zero_aspect_block(all_block)
        stats["test"] = _zero_aspect_block(all_block)
        stats["official_corpus_split"] = "train"
        write_json(stats_path, stats)

    config_path = output_dir / "preprocess_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["official_corpus_split"] = "train"
    config["fixed_split"] = "train"
    config["legacy_internal_split_counts"] = split_counts
    config["legacy_internal_split_column"] = LEGACY_SPLIT_COLUMN
    config["split_note"] = (
        "All existing rows belong to official SciREX train. "
        "The previous hash-based 0.6/0.2/0.2 labels are in legacy_internal_split."
    )
    write_json(config_path, config)

    return {
        "output_dir": str(output_dir),
        "already_relabeled": already,
        "row_counts": row_counts,
        "legacy_internal_split_counts": split_counts,
        "rewritten": rewritten,
    }
