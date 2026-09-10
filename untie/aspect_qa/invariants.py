"""Corpus-level invariants for source-only gold and mention offsets."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .normalize import identity_key, normalize_entity
from .pipeline import ProcessedDocument


class InvariantError(ValueError):
    pass


def assert_mention_offsets(raw_text: str, mention: Mapping[str, Any]) -> None:
    start = int(mention["char_start"])
    end = int(mention["char_end"])
    matched = str(mention["matched_text"])
    slice_text = raw_text[start:end]
    if slice_text != matched:
        raise InvariantError(
            f"mention offset mismatch id={mention.get('mention_id')}: "
            f"{slice_text!r} != {matched!r}"
        )


def assert_canonical_subset_of_raw(instance: Mapping[str, Any]) -> None:
    raw_keys = {identity_key(value) for value in instance["raw_gold_answers"]}
    for canonical in instance["canonical_gold_answers"]:
        if identity_key(canonical) not in raw_keys:
            raise InvariantError(
                f"canonical {canonical!r} is not traceable to raw_gold_answers"
            )
        if normalize_entity(canonical) != canonical and identity_key(canonical) not in raw_keys:
            raise InvariantError(f"canonical form drifted from raw gold: {canonical!r}")


def assert_extractable_has_mentions(
    instance: Mapping[str, Any],
    mentions: Sequence[Mapping[str, Any]],
) -> None:
    by_canonical: dict[str, int] = {}
    for mention in mentions:
        if mention.get("in_bibliography"):
            continue
        key = identity_key(str(mention["canonical_answer"]))
        by_canonical[key] = by_canonical.get(key, 0) + 1
    for canonical in instance["extractable_gold_answers"]:
        if by_canonical.get(identity_key(canonical), 0) < 1:
            raise InvariantError(
                f"extractable gold {canonical!r} has no confirmed non-bibliography mention"
            )
    for canonical in instance["unextractable_gold_answers"]:
        if by_canonical.get(identity_key(canonical), 0) != 0:
            raise InvariantError(
                f"unextractable gold {canonical!r} still has eligible mentions"
            )


def assert_no_external_gold(instance: Mapping[str, Any]) -> None:
    if instance.get("external_enrichment"):
        for item in instance["external_enrichment"]:
            if item.get("used_for_gold"):
                raise InvariantError("external_enrichment used_for_gold must be false in E0")
    if instance.get("enrichment_candidates"):
        for item in instance["enrichment_candidates"]:
            if item.get("used_for_gold"):
                raise InvariantError("enrichment_candidates used_for_gold must be false in E0")


def validate_processed_document(processed: ProcessedDocument) -> None:
    raw_text = processed.record.original_text
    mentions_by_aspect: dict[str, list[dict[str, Any]]] = {}
    for mention in processed.mentions:
        assert_mention_offsets(raw_text, mention)
        mentions_by_aspect.setdefault(str(mention["aspect"]), []).append(mention)
    score_aspects = [item for item in processed.instances if item.aspect == "score"]
    if score_aspects:
        raise InvariantError("score must not become an aspect instance when ENABLE_SCORE_ASPECT=False")
    for instance in processed.instances:
        payload = {
            "raw_gold_answers": instance.raw_gold_answers,
            "canonical_gold_answers": instance.canonical_gold_answers,
            "extractable_gold_answers": instance.extractable_gold_answers,
            "unextractable_gold_answers": instance.unextractable_gold_answers,
            "external_enrichment": instance.external_enrichment,
            "enrichment_candidates": instance.enrichment_candidates,
        }
        assert_canonical_subset_of_raw(payload)
        assert_extractable_has_mentions(payload, mentions_by_aspect.get(instance.aspect, []))
        assert_no_external_gold(payload)
        if instance.status == "no_extractable_gold" and instance.extractable_gold_answers:
            raise InvariantError("no_extractable_gold instance has extractable answers")
        if instance.status == "ready" and instance.unextractable_gold_answers:
            raise InvariantError("ready instance still has unextractable answers")
        if instance.status == "partially_extractable" and (
            not instance.extractable_gold_answers or not instance.unextractable_gold_answers
        ):
            raise InvariantError("partially_extractable requires both extractable and unextractable gold")
    for chunk in processed.chunks:
        if processed.record.original_text[chunk.char_start : chunk.char_end] != chunk.text:
            raise InvariantError(f"chunk {chunk.chunk_id} text does not match raw offsets")
