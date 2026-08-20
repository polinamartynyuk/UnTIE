"""Leakage-safe SFFS dictionaries for train-derived topic clusters."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from .keyword_evidence import CachedDocumentEvidence, ExtractionMetricCache
from .keyword_training import (
    MetricWeights,
    TrainingConfig,
    TuningOutcome,
    tune_global_keywords,
)
from .keyword_tuning import ObjectiveConfig
from .protocols import SentenceEncoder
from .ranking import WeightedKeyword
from .topics import (
    TopicArtifact,
    TopicDocument,
    TopicKeywordProfile,
    route_document_to_nearest_leaf,
)


@dataclass(frozen=True)
class TopicKeywordTrainingResult:
    artifact: TopicArtifact
    outcomes: Mapping[str, TuningOutcome]
    skipped_nodes: Mapping[str, str]
    assignments: Mapping[str, str]


def assign_documents_to_leaf_topics(
    documents: Sequence[TopicDocument],
    encoder: SentenceEncoder,
    artifact: TopicArtifact,
    *,
    expected_encoder_name: str | None = None,
    expected_encoder_revision: str | None = None,
) -> dict[str, str]:
    """Assign documents to nearest leaves without reading labels or references."""
    assignments: dict[str, str] = {}
    for document in documents:
        route = route_document_to_nearest_leaf(
            document,
            encoder,
            artifact,
            expected_encoder_name=expected_encoder_name,
            expected_encoder_revision=expected_encoder_revision,
        )
        if route.topics:
            assignments[document.doc_id] = route.topics[0].node_id
    return assignments


def tune_topic_keyword_profiles(
    artifact: TopicArtifact,
    train_documents: Sequence[TopicDocument],
    dev_documents: Sequence[TopicDocument],
    evidence: Sequence[CachedDocumentEvidence],
    encoder: SentenceEncoder,
    *,
    config: TrainingConfig,
    metric_cache: ExtractionMetricCache,
    objective_config: ObjectiveConfig = ObjectiveConfig(),
    metric_weights: MetricWeights = MetricWeights(),
    checkpoint_dir: str | Path | None = None,
    min_train_documents: int = 2,
    min_dev_documents: int = 1,
    expected_encoder_name: str | None = None,
    expected_encoder_revision: str | None = None,
) -> TopicKeywordTrainingResult:
    """Tune one static-style dictionary per leaf using train candidates and dev QA.

    Clusters and centroids are never rebuilt. Candidate terms come only from
    train evidence; dev references are used only by the existing SFFS objective.
    Test documents are intentionally not accepted by this API.
    """
    if min_train_documents < 1 or min_dev_documents < 1:
        raise ValueError("minimum train/dev document counts must be positive")
    _validate_documents(train_documents, "train")
    _validate_documents(dev_documents, "dev")

    all_documents = tuple(train_documents) + tuple(dev_documents)
    if len({item.doc_id for item in all_documents}) != len(all_documents):
        raise ValueError("topic keyword train/dev document ids must be disjoint")
    by_evidence = {item.doc_id: item for item in evidence}
    required_ids = {item.doc_id for item in all_documents}
    missing = sorted(required_ids.difference(by_evidence))
    if missing:
        raise ValueError(f"missing cached keyword evidence for documents: {missing[:5]}")

    stored_train_assignments = {
        item.doc_id: item.leaf_id for item in artifact.train_assignments
    }
    routed_train_documents = [
        item
        for item in train_documents
        if item.doc_id not in stored_train_assignments
    ]
    assignments = {
        item.doc_id: stored_train_assignments[item.doc_id]
        for item in train_documents
        if item.doc_id in stored_train_assignments
    }
    assignments.update(
        assign_documents_to_leaf_topics(
            routed_train_documents,
            encoder,
            artifact,
            expected_encoder_name=expected_encoder_name,
            expected_encoder_revision=expected_encoder_revision,
        )
    )
    assignments.update(
        assign_documents_to_leaf_topics(
            dev_documents,
            encoder,
            artifact,
            expected_encoder_name=expected_encoder_name,
            expected_encoder_revision=expected_encoder_revision,
        )
    )
    outcomes: dict[str, TuningOutcome] = {}
    skipped: dict[str, str] = {}
    updated_nodes = []
    for node in artifact.nodes:
        if node.children:
            updated_nodes.append(replace(node, keyword_profile=None))
            continue
        clean_node = replace(node, keyword_profile=None)
        train_ids = tuple(
            sorted(
                item.doc_id
                for item in train_documents
                if assignments.get(item.doc_id) == node.node_id
            )
        )
        dev_ids = tuple(
            sorted(
                item.doc_id
                for item in dev_documents
                if assignments.get(item.doc_id) == node.node_id
            )
        )
        if len(train_ids) < min_train_documents:
            skipped[node.node_id] = (
                f"insufficient train documents: {len(train_ids)} < {min_train_documents}"
            )
            updated_nodes.append(clean_node)
            continue
        if len(dev_ids) < min_dev_documents:
            skipped[node.node_id] = (
                f"insufficient dev documents: {len(dev_ids)} < {min_dev_documents}"
            )
            updated_nodes.append(clean_node)
            continue

        node_config = replace(
            config,
            min_document_support=min(config.min_document_support, len(train_ids)),
        )
        node_checkpoint = (
            Path(checkpoint_dir) / node.node_id
            if checkpoint_dir is not None
            else None
        )
        node_evidence = [
            by_evidence[doc_id] for doc_id in (*train_ids, *dev_ids)
        ]
        try:
            outcome = tune_global_keywords(
                node_evidence,
                config=node_config,
                metric_cache=metric_cache,
                metric_weights=metric_weights,
                objective_config=objective_config,
                split={"train": train_ids, "dev": dev_ids, "test": ()},
                checkpoint_dir=node_checkpoint,
            )
        except ValueError as error:
            if "Candidate pool is empty" not in str(error):
                raise
            skipped[node.node_id] = str(error)
            updated_nodes.append(clean_node)
            continue
        if not outcome.keywords or not math.isfinite(outcome.objective):
            skipped[node.node_id] = "SFFS produced no finite non-empty profile"
            updated_nodes.append(clean_node)
            continue

        metadata = {item.word: item for item in outcome.keyword_metadata}
        keywords = tuple(
            WeightedKeyword(
                word=word,
                lemma=metadata[word].lemma or word.casefold(),
                stem=metadata[word].stem or word.casefold(),
                attention_weight=metadata[word].attention_weight,
                score_difference=metadata[word].score_difference,
            )
            for word in outcome.keywords
        )
        profile = TopicKeywordProfile(
            keywords=keywords,
            score_chunk_strategy=outcome.strategy.score_chunk_strategy,
            choose_cluster_strategy=outcome.strategy.choose_cluster_strategy,
            choose_answer_strategy=outcome.strategy.choose_answer_strategy,
            train_document_count=len(train_ids),
            dev_document_count=len(dev_ids),
            objective=outcome.objective,
            split_hash=str(outcome.tuning_metadata()["split_hash"]),
            training_fingerprint=outcome.fingerprint,
        )
        outcomes[node.node_id] = outcome
        updated_nodes.append(replace(clean_node, keyword_profile=profile))

    return TopicKeywordTrainingResult(
        artifact=replace(artifact, nodes=tuple(updated_nodes)),
        outcomes=outcomes,
        skipped_nodes=skipped,
        assignments=assignments,
    )


def _validate_documents(
    documents: Sequence[TopicDocument],
    expected_split: str,
) -> None:
    invalid = [
        item.doc_id
        for item in documents
        if item.split.casefold() != expected_split
    ]
    if invalid:
        raise ValueError(
            f"topic keyword tuning requires {expected_split} documents only; "
            f"invalid ids: {invalid[:5]}"
        )
