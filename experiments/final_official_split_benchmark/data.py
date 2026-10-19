"""Dataset discovery and the TEST-label firewall.

The prediction side only receives :class:`RuntimeInstance`.  Conversion to an
``InstanceRecord`` (which contains gold and evidence labels) is confined to
``load_test_gold`` and is therefore unavailable before the evaluation phase.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from experiments.qa_chunk_selection.schema import ChunkRecord, GoldSpan, InstanceRecord
from experiments.qa_chunk_selection.v2.schemas import RuntimeChunk, RuntimeInstance, assert_no_gold_fields, to_runtime
from experiments.qa_chunk_selection.v3.data import instance_from_dict, records_from_jsonl, synthetic_records

from .contract import CELL_SPECS, SPLIT_NO_TEST, SPLIT_OFFICIAL_TT, SPLIT_SEED
from .io_guard import canonical_hash, read_jsonl, write_json, write_jsonl


DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)

def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def document_text(item: InstanceRecord | RuntimeInstance) -> str:
    chunks = sorted(item.chunks, key=lambda x: (x.char_start, x.chunk_id))
    if not chunks:
        return ""
    # Offset-aware reconstruction avoids counting overlap twice.
    size = max(int(x.char_end) for x in chunks)
    if 0 < size <= 10_000_000:
        chars = [""] * size
        for chunk in chunks:
            for offset, char in enumerate(chunk.text):
                pos = int(chunk.char_start) + offset
                if pos < size and not chars[pos]:
                    chars[pos] = char
        value = "".join(chars).strip()
        if value:
            return value
    return "\n".join(x.text for x in chunks)


def text_hash(item: InstanceRecord | RuntimeInstance) -> str:
    return hashlib.sha256(normalized_text(document_text(item)).encode()).hexdigest()


def _eligible(records: Iterable[InstanceRecord]) -> tuple[InstanceRecord, ...]:
    return tuple(x for x in records if x.instance_status == "ok" and x.gold_answers and x.positives)


def _ranked_docs(records: Sequence[InstanceRecord], salt: str) -> list[str]:
    docs = sorted({str(x.document_id) for x in records})
    return sorted(docs, key=lambda value: hashlib.sha256(f"{SPLIT_SEED}|{salt}|{value}".encode()).hexdigest())


def _internal_split(records: Sequence[InstanceRecord], *, test: bool) -> tuple[tuple[InstanceRecord, ...], tuple[InstanceRecord, ...], tuple[InstanceRecord, ...]]:
    """Document-grouped 64/16/20 (RuSERRC) or 80/20 (SciMDIX) split."""
    docs = _ranked_docs(records, "final-official-split")
    if test:
        n_test = max(1, round(len(docs) * .20))
        test_docs = set(docs[:n_test])
        remaining = docs[n_test:]
        n_val = max(1, round(len(remaining) * .20))
        val_docs = set(remaining[:n_val])
    else:
        test_docs = set()
        n_val = max(1, round(len(docs) * .20))
        val_docs = set(docs[:n_val])
    train = tuple(x for x in records if x.document_id not in val_docs | test_docs)
    val = tuple(x for x in records if x.document_id in val_docs)
    holdout = tuple(x for x in records if x.document_id in test_docs)
    return train, val, holdout


def development_splits(cell: str, *, smoke: bool = False) -> tuple[tuple[InstanceRecord, ...], tuple[InstanceRecord, ...]]:
    if smoke:
        rows = _eligible(synthetic_records(n_docs=12))
        train, val, _ = _internal_split(rows, test=False)
        return train, val
    spec = CELL_SPECS[cell]
    records = tuple(records_from_jsonl(spec.train))
    if cell == "scirex":
        train = _eligible(x for x in records if x.source_split == "train")
        val = _eligible(x for x in records if x.source_split in {"dev", "validation"})
    elif spec.split_mode == SPLIT_OFFICIAL_TT:
        train, val, _ = _internal_split(_eligible(records), test=False)
    elif spec.split_mode == SPLIT_NO_TEST:
        train, val, _ = _internal_split(_eligible(records), test=True)
    else:
        train = _eligible(records)
        if spec.validation is None or not spec.validation.is_file():
            raise FileNotFoundError(f"official validation missing: {spec.validation}")
        val = _eligible(records_from_jsonl(spec.validation))
    assert_disjoint(train, val, label=f"{cell} TRAIN/VAL")
    if not train or not val:
        raise RuntimeError(f"{cell}: empty canonical TRAIN or VAL population")
    return train, val


def final_training_pool(cell: str, *, smoke: bool = False) -> tuple[InstanceRecord, ...]:
    train, val = development_splits(cell, smoke=smoke)
    assert_disjoint(train, val, label=f"{cell} final pool")
    return tuple((*train, *val))


def _runtime_from_payload(payload: dict) -> RuntimeInstance:
    chunks = []
    for value in payload["chunks"]:
        chunks.append(RuntimeChunk(
            chunk_id=int(value["chunk_id"]), text=str(value["text"]), position=int(value.get("position", value["chunk_id"])),
            section_id=str(value.get("section_id", "")), section_title=str(value.get("section_title", "")),
            char_start=int(value.get("char_start", 0)), char_end=int(value.get("char_end", 0)),
            token_start=int(value.get("token_start", 0)), token_end=int(value.get("token_end", 0)),
            previous_chunk_id=value.get("previous_chunk_id"), next_chunk_id=value.get("next_chunk_id"),
            eligible_for_qa=bool(value.get("eligible_for_qa", True)),
        ))
    runtime = RuntimeInstance(
        document_id=str(payload["document_id"]), question_id=str(payload["question_id"]),
        aspect=str(payload["aspect"]), question=str(payload["question"]), split=str(payload.get("split", "test")),
        chunks=tuple(chunks),
    )
    assert_no_gold_fields(runtime.to_dict())
    return runtime


def _jsonl_runtime(path: Path) -> tuple[RuntimeInstance, ...]:
    # A field allow-list is intentional: gold_answers, spans and labels never
    # cross the returned runtime boundary.
    return tuple(_runtime_from_payload(row) for row in read_jsonl(path))


def _scirex_runtime(root: Path) -> tuple[RuntimeInstance, ...]:
    import pandas as pd
    chunks = pd.read_parquet(root / "chunks.parquet")
    aspects = pd.read_parquet(root / "aspect_instances.parquet")
    by_doc = {str(doc): frame.sort_values("chunk_id") for doc, frame in chunks.groupby("document_id")}
    result = []
    for row in aspects.to_dict("records"):
        doc = str(row["document_id"]); values = by_doc[doc]; runtime_chunks = []
        ids = values["chunk_id"].astype(int).tolist()
        for index, item in enumerate(values.to_dict("records")):
            runtime_chunks.append(RuntimeChunk(
                chunk_id=int(item["chunk_id"]), text=str(item["chunk_text"]), position=index,
                section_id=str(item.get("section_type") or ""), section_title=str(item.get("section_title") or ""),
                char_start=int(item["char_start"]), char_end=int(item["char_end"]), token_start=int(item["token_start"]), token_end=int(item["token_end"]),
                previous_chunk_id=ids[index - 1] if index else None, next_chunk_id=ids[index + 1] if index + 1 < len(ids) else None,
                eligible_for_qa=bool(item.get("eligible_for_qa", True)),
            ))
        result.append(RuntimeInstance(doc, f"{doc}::{row['aspect']}", str(row["aspect"]), str(row["question"]), "official_test", tuple(runtime_chunks)))
    for item in result:
        assert_no_gold_fields(item.to_dict())
    return tuple(result)


def load_test_runtime(cell: str, *, smoke: bool = False) -> tuple[RuntimeInstance, ...]:
    if smoke:
        raise RuntimeError("TEST_FIREWALL: smoke mode cannot read TEST")
    spec = CELL_SPECS[cell]
    if cell == "russerrc":
        records = _eligible(records_from_jsonl(spec.train))
        _, _, holdout = _internal_split(records, test=True)
        return tuple(to_runtime(x) for x in holdout)
    if cell == "scirex":
        return _scirex_runtime(Path(spec.test_runtime))
    if spec.test_runtime is None or not spec.test_runtime.is_file():
        raise FileNotFoundError(f"{cell}: official test runtime source missing: {spec.test_runtime}")
    return _jsonl_runtime(spec.test_runtime)


def _scirex_gold(root: Path) -> tuple[InstanceRecord, ...]:
    import pandas as pd
    aspects = pd.read_parquet(root / "aspect_instances.parquet")
    chunks = pd.read_parquet(root / "chunks.parquet")
    mentions = pd.read_parquet(root / "gold_mentions.parquet")
    by_doc = {str(doc): frame.sort_values("chunk_id") for doc, frame in chunks.groupby("document_id")}
    by_key = {(str(doc), str(aspect)): frame for (doc, aspect), frame in mentions.groupby(["document_id", "aspect"])}
    result = []
    for row in aspects.to_dict("records"):
        answers = tuple(str(x) for x in list(row["extractable_gold_answers"]) if str(x).strip())
        if not answers:
            continue
        doc, aspect = str(row["document_id"]), str(row["aspect"]); positives = {int(x) for x in list(row["gold_supporting_chunk_ids"])}
        raw_chunks = by_doc[doc].to_dict("records"); ids = [int(x["chunk_id"]) for x in raw_chunks]; out_chunks = []
        for index, value in enumerate(raw_chunks):
            cid = int(value["chunk_id"]); positive = cid in positives
            out_chunks.append(ChunkRecord(
                cid, str(value["chunk_text"]), "positive" if positive else "easy_negative",
                "robust_positive" if positive else "unresolved_for_training", index,
                str(value.get("section_type") or ""), str(value.get("section_title") or ""),
                int(value["char_start"]), int(value["char_end"]), int(value["token_start"]), int(value["token_end"]), positive,
                ids[index - 1] if index else None, ids[index + 1] if index + 1 < len(ids) else None, bool(value.get("eligible_for_qa", True)),
            ))
        spans = []
        for mention in by_key.get((doc, aspect), []).to_dict("records") if (doc, aspect) in by_key else []:
            for cid in list(mention["chunk_ids"]):
                chunk = next(x for x in out_chunks if x.chunk_id == int(cid))
                start = max(0, int(mention["char_start"]) - chunk.char_start)
                end = min(len(chunk.text), int(mention["char_end"]) - chunk.char_start)
                spans.append(GoldSpan(int(cid), start, end, str(mention["matched_text"]), int(mention["char_start"]), int(mention["char_end"])))
        result.append(InstanceRecord(doc, f"{doc}::{aspect}", aspect, str(row["question"]), answers, "official_test", tuple(out_chunks), tuple(spans), "ok", "official_test"))
    return tuple(result)


def load_test_gold(cell: str, *, unlocked: bool, smoke: bool = False) -> tuple[InstanceRecord, ...]:
    if not unlocked:
        raise RuntimeError("TEST_FIREWALL: gold access requires completed blind predictions and unlock assertion")
    if smoke:
        raise RuntimeError("TEST_FIREWALL: smoke mode cannot read TEST gold")
    spec = CELL_SPECS[cell]
    if cell == "russerrc":
        records = _eligible(records_from_jsonl(spec.train)); _, _, holdout = _internal_split(records, test=True); return holdout
    if cell == "scirex":
        return _scirex_gold(Path(spec.test_gold))
    if spec.test_gold is None or not spec.test_gold.is_file():
        raise FileNotFoundError(f"{cell}: official test gold missing: {spec.test_gold}")
    return _eligible(records_from_jsonl(spec.test_gold))


def assert_disjoint(left: Sequence[InstanceRecord | RuntimeInstance], right: Sequence[InstanceRecord | RuntimeInstance], *, label: str) -> None:
    left_ids = {x.document_id for x in left}; right_ids = {x.document_id for x in right}
    overlap = left_ids & right_ids
    left_hash = {text_hash(x) for x in left}; right_hash = {text_hash(x) for x in right}; duplicate = left_hash & right_hash
    if overlap or duplicate:
        raise RuntimeError(f"{label} leakage: document_ids={sorted(overlap)[:5]}, text_hashes={sorted(duplicate)[:5]}")


def split_manifest(cell: str, run_dir: Path, *, smoke: bool = False) -> dict:
    train, val = development_splits(cell, smoke=smoke)
    if smoke:
        test = ()
    else:
        test = load_test_runtime(cell)
        assert_disjoint(train, test, label=f"{cell} TRAIN/TEST")
        assert_disjoint(val, test, label=f"{cell} VAL/TEST")
    def describe(values):
        docs = sorted({x.document_id for x in values})
        return {"questions": len(values), "documents": len(docs), "document_ids": docs, "text_hashes": sorted({text_hash(x) for x in values}),
                "dois": sorted({doi.rstrip(".,;)").casefold() for x in values for doi in DOI_RE.findall(document_text(x))})}
    manifest = {"cell": cell, "mode": CELL_SPECS[cell].split_mode, "seed": SPLIT_SEED, "train": describe(train), "validation": describe(val), "test_runtime": describe(test)}
    roles = ("train", "validation", "test_runtime")
    manifest["overlap_audit"] = {}
    for i, left in enumerate(roles):
        for right in roles[i + 1:]:
            manifest["overlap_audit"][f"{left}__{right}"] = {
                "document_ids": sorted(set(manifest[left]["document_ids"]) & set(manifest[right]["document_ids"])),
                "normalized_text_hashes": sorted(set(manifest[left]["text_hashes"]) & set(manifest[right]["text_hashes"])),
                "dois": sorted(set(manifest[left]["dois"]) & set(manifest[right]["dois"])),
            }
    manifest["sha256"] = canonical_hash(manifest)
    write_json(Path(run_dir) / "split_audit" / f"{cell}.json", manifest)
    write_jsonl(Path(run_dir) / "split_manifests" / f"{cell}.jsonl", [
        {"role": role, "document_id": doc, "cell": cell}
        for role, values in (("TRAIN", train), ("VAL", val), ("TEST", test)) for doc in sorted({x.document_id for x in values})
    ])
    return manifest


def check_scier_linkage(run_dir: Path) -> dict:
    left = load_test_runtime("scier_128"); right = load_test_runtime("scier_384")
    left_docs = {x.document_id for x in left}; right_docs = {x.document_id for x in right}
    report = {"status": "PASS" if left_docs == right_docs else "FAIL", "scier_128_documents": len(left_docs), "scier_384_documents": len(right_docs), "same_source_documents": left_docs == right_docs}
    write_json(Path(run_dir) / "split_audit/scier_lineage.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("SciER 128/384 source-document linkage failed")
    return report
