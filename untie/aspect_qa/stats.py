"""Preprocessing statistics required by the aspect-QA prompt."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .config import ASPECT_FIELD_MAP
from .pipeline import ProcessedDocument


def collect_statistics(
    documents: Sequence[ProcessedDocument],
    *,
    splits: Sequence[str] = ("train", "dev", "test", "all"),
) -> dict[str, Any]:
    by_split: dict[str, list[ProcessedDocument]] = defaultdict(list)
    for item in documents:
        by_split[item.split].append(item)
        by_split["all"].append(item)

    report: dict[str, Any] = {}
    for split in splits:
        subset = by_split.get(split, [])
        aspects: dict[str, dict[str, Any]] = {}
        for aspect in ASPECT_FIELD_MAP:
            instances = [inst for doc in subset for inst in doc.instances if inst.aspect == aspect]
            raw_entities = 0
            unique_count = 0
            extractable = 0
            unextractable = 0
            supporting_chunks = 0
            ready = partially = none = 0
            for instance in instances:
                raw_entities += len(instance.raw_gold_answers)
                unique_count += len(instance.canonical_gold_answers)
                extractable += len(instance.extractable_gold_answers)
                unextractable += len(instance.unextractable_gold_answers)
                supporting_chunks += len(instance.gold_supporting_chunk_ids)
                if instance.status == "ready":
                    ready += 1
                elif instance.status == "partially_extractable":
                    partially += 1
                else:
                    none += 1
            aspects[aspect] = {
                "documents": len(subset),
                "raw_gold_entities": raw_entities,
                "unique_gold_entities": unique_count,
                "extractable_gold_entities": extractable,
                "unextractable_gold_entities": unextractable,
                "ready_instances": ready,
                "partially_extractable_instances": partially,
                "no_extractable_gold_instances": none,
                "gold_supporting_chunks": supporting_chunks,
                "extractability_rate": (extractable / unique_count) if unique_count else 0.0,
            }
        report[split] = aspects
    report["n_documents"] = len(documents)
    report["n_chunks"] = sum(len(item.chunks) for item in documents)
    report["n_qa_input_rows"] = sum(
        1
        for item in documents
        for instance in item.instances
        if instance.status != "no_extractable_gold"
        for chunk in item.chunks
        if chunk.eligible_for_qa
    )
    return report


def format_statistics(report: Mapping[str, Any]) -> str:
    lines = [
        f"documents={report.get('n_documents', 0)} chunks={report.get('n_chunks', 0)} "
        f"qa_input_rows={report.get('n_qa_input_rows', 0)}",
        "",
    ]
    for split in ("train", "dev", "test", "all"):
        aspects = report.get(split)
        if not isinstance(aspects, dict):
            continue
        lines.append(f"[{split}]")
        for aspect, stats in aspects.items():
            if not isinstance(stats, dict):
                continue
            lines.append(
                f"  {aspect}: docs={stats['documents']} raw={stats['raw_gold_entities']} "
                f"unique={stats['unique_gold_entities']} extractable={stats['extractable_gold_entities']} "
                f"unextractable={stats['unextractable_gold_entities']} ready={stats['ready_instances']} "
                f"partial={stats['partially_extractable_instances']} none={stats['no_extractable_gold_instances']} "
                f"gold_chunks={stats['gold_supporting_chunks']} "
                f"extractability={stats['extractability_rate']:.3f}"
            )
        lines.append("")
    return "\n".join(lines)


def manual_review_payload(processed: ProcessedDocument) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    mentions_by_aspect: dict[str, list[dict[str, Any]]] = {}
    for mention in processed.mentions:
        mentions_by_aspect.setdefault(str(mention["aspect"]), []).append(mention)
    for instance in processed.instances:
        source_values = {
            "method": processed.record.methods,
            "task": processed.record.tasks,
            "metric": processed.record.metrics,
            "material": processed.record.materials,
        }[instance.aspect]
        rows.append(
            {
                "DOC_ID": processed.record.doc_id,
                "ASPECT": instance.aspect,
                "QUESTION": instance.question,
                "RAW GOLD": list(instance.raw_gold_answers),
                "NORMALIZED / CANONICAL GOLD": list(instance.canonical_gold_answers),
                "EXTRACTABLE GOLD": list(instance.extractable_gold_answers),
                "UNEXTRACTABLE GOLD": list(instance.unextractable_gold_answers),
                "GOLD MENTIONS": [
                    {
                        "mention_id": item["mention_id"],
                        "matched_text": item["matched_text"],
                        "char_start": item["char_start"],
                        "char_end": item["char_end"],
                        "chunk_ids": item["chunk_ids"],
                    }
                    for item in mentions_by_aspect.get(instance.aspect, [])
                ],
                "GOLD-SUPPORTING CHUNKS": list(instance.gold_supporting_chunk_ids),
                "SOURCE CHECK": {
                    "source_field_values": list(source_values),
                    "transforms": "NFKC, _+→space, dash unification, whitespace collapse, casefold for match only",
                    "new_factual_values_added": False,
                    "external_values_used_for_gold": False,
                    "external_enrichment_count": len(instance.external_enrichment),
                },
            }
        )
    return {"document_id": processed.record.doc_id, "split": processed.split, "aspects": rows}
