"""Convert official SciREX release JSONL splits into the structured JSONL used by UnTIE.

The conversion matches experiments/phase1_scirex/scirex_first_preprocessing.ipynb:
detokenize words, collect unique n-ary relation values, keep the seven-field schema.
Existing datasets/scirex_structured.json is the official SciREX train split (306 docs).
"""

from __future__ import annotations

import json
import re
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

RELEASE_URL = "https://github.com/allenai/SciREX/raw/master/scirex_dataset/release_data.tar.gz"
OFFICIAL_SPLITS = ("train", "dev", "test")
STRUCTURED_FIELDS = (
    "doc_id",
    "methods",
    "tasks",
    "metrics",
    "materials",
    "score",
    "original_text",
)
RELATION_KEY_MAP = {
    "Method": "methods",
    "Task": "tasks",
    "Metric": "metrics",
    "Material": "materials",
    "score": "score",
}
PUNCTUATION = {".", ",", ":", ";", "!", "?", "%", ")", "]", "}", "»", "..."}
OPENING_PUNCT = {"(", "[", "{", "«", "“", '"', "'"}


def detokenize_words(words: Iterable[Any]) -> str:
    text = ""
    sequence = [str(token) for token in words]
    for index, token in enumerate(sequence):
        if index == 0:
            text += token
        elif token in PUNCTUATION:
            text += token
        elif sequence[index - 1] in OPENING_PUNCT:
            text += token
        else:
            text += " " + token
    return re.sub(r"\s+([.,!?:;])", r"\1", text)


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in values:
        if item and item not in seen:
            seen[item] = None
    return list(seen)


def extract_clean_row(doc: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "doc_id": str(doc.get("doc_id", "")),
        "methods": [],
        "tasks": [],
        "metrics": [],
        "materials": [],
        "score": [],
    }
    collected: dict[str, list[str]] = {name: [] for name in ("methods", "tasks", "metrics", "materials", "score")}
    for rel in doc.get("n_ary_relations") or []:
        if not isinstance(rel, Mapping):
            continue
        for key, value in rel.items():
            target = RELATION_KEY_MAP.get(key)
            if not target or not value:
                continue
            if isinstance(value, list):
                collected[target].extend(item.strip() for item in value if isinstance(item, str) and item.strip())
            elif isinstance(value, str) and value.strip():
                collected[target].append(value.strip())
            elif isinstance(value, (int, float)):
                collected[target].append(str(value).strip())
    for name in collected:
        row[name] = _unique_preserve_order(collected[name])
    words = doc.get("words") or []
    row["original_text"] = detokenize_words(words) if words else ""
    return row


def load_release_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def structured_records(docs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [extract_clean_row(doc) for doc in docs]


def write_structured_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in records:
            payload = {field: row[field] for field in STRUCTURED_FIELDS}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return destination


def download_release_archive(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    urllib.request.urlretrieve(RELEASE_URL, dest)
    return dest


def extract_release_archive(archive: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / "train.jsonl"
    if marker.exists():
        return dest_dir
    nested = dest_dir / "release_data"
    if (nested / "train.jsonl").exists():
        return nested
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(dest_dir, filter="data")
        except TypeError:
            tar.extractall(dest_dir)
    if marker.exists():
        return dest_dir
    if (nested / "train.jsonl").exists():
        return nested
    raise FileNotFoundError(f"train.jsonl not found after extracting {archive} into {dest_dir}")


STRUCTURED_FILENAMES = {
    "train": "scirex_structured.json",
    "dev": "scirex_dev_structured.json",
    "test": "scirex_test_structured.json",
}


def prepare_official_structured_splits(
    *,
    datasets_dir: Path,
    release_dir: Path,
    skip_existing: tuple[str, ...] = ("train",),
    download: bool = True,
) -> dict[str, Any]:
    """Convert official SciREX JSONL splits. Skip existing train by default."""
    datasets_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)
    archive = release_dir / "release_data.tar.gz"
    if download:
        download_release_archive(archive)
    jsonl_dir = extract_release_archive(archive, release_dir)
    report: dict[str, Any] = {"release_dir": str(jsonl_dir), "splits": {}}
    for split in OFFICIAL_SPLITS:
        source = jsonl_dir / f"{split}.jsonl"
        output = datasets_dir / STRUCTURED_FILENAMES[split]
        if split in skip_existing and output.exists() and output.stat().st_size > 0:
            n_records = sum(1 for line in output.open(encoding="utf-8") if line.strip())
            report["splits"][split] = {
                "source": str(source),
                "output": str(output),
                "n_records": n_records,
                "skipped": True,
                "reason": "existing file left unchanged",
            }
            continue
        if not source.exists():
            raise FileNotFoundError(f"official SciREX {split} file missing: {source}")
        info = convert_official_split(source, output)
        info["skipped"] = False
        report["splits"][split] = info
    return report


def convert_official_split(jsonl_path: Path, output_path: Path) -> dict[str, Any]:
    docs = load_release_jsonl(jsonl_path)
    records = structured_records(docs)
    write_structured_jsonl(records, output_path)
    return {
        "source": str(jsonl_path),
        "output": str(output_path),
        "n_records": len(records),
    }
