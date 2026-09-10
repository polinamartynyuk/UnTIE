"""Leakage-safe SFFS dictionaries for train-derived topic clusters."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
import csv
from typing import Mapping, Sequence

from .keyword_evidence import CachedDocumentEvidence, ExtractionMetricCache
from .keyword_training import (
    MetricWeights,
    TrainingConfig,
    TuningOutcome,
    tune_global_keywords,
)
from .keyword_tuning import ObjectiveConfig, deterministic_document_split
from .protocols import SentenceEncoder
from .ranking import WeightedKeyword
from .topics import (
    TopicArtifact,
    TopicDocument,
    TopicKeywordProfile,
    route_document_to_nearest_leaf,
)


@dataclass(frozen=True)
class TopicKeywordCoverage:
    node_id: str
    train_documents: int
    dev_documents: int
    candidate_count: int
    selected_keyword_count: int
    covered_train_documents: int
    train_document_coverage: float
    mean_selected_document_support: float
    keyword_floor_reached: bool
    fit_documents: int = 0
    fit_train_documents: int = 0
    fit_dev_documents: int = 0


@dataclass(frozen=True)
class TopicKeywordTrainingResult:
    artifact: TopicArtifact
    outcomes: Mapping[str, TuningOutcome]
    skipped_nodes: Mapping[str, str]
    assignments: Mapping[str, str]
    coverage: Mapping[str, TopicKeywordCoverage] = field(default_factory=dict)


def load_selected_keyword_trial(
    path: str | Path,
    *,
    trial_index: int | None = None,
) -> dict[str, object]:
    """Load one keyword-search row without re-running the trial grid."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise ValueError(f"keyword search CSV not found: {csv_path}")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"keyword search CSV is empty: {csv_path}")

    def parsed(row: Mapping[str, str]) -> dict[str, object]:
        return {
            "trial": int(row.get("trial", 0)),
            "min_keywords": int(row["min_keywords"]),
            "max_keywords": int(row["max_keywords"]),
            "selection_policy": str(row["selection_policy"]),
            "stability_threshold": float(row["stability_threshold"]),
            "objective": float(row.get("objective", 0.0) or 0.0),
            "trained_nodes": int(row.get("trained_nodes", 0) or 0),
            "skipped_nodes": int(row.get("skipped_nodes", 0) or 0),
        }

    parsed_rows = [parsed(row) for row in rows]
    if trial_index is not None:
        matches = [row for row in parsed_rows if int(row["trial"]) == trial_index]
        if not matches:
            raise ValueError(f"keyword trial {trial_index} is not in {csv_path}")
        return matches[0]
    return min(
        parsed_rows,
        key=lambda row: (
            -float(row["objective"]),
            int(row["min_keywords"]),
            int(row["max_keywords"]),
            int(row["trial"]),
        ),
    )


DEFAULT_REFIT_MIN_KEYWORDS = 4


def resolve_refit_keyword_floor(
    trial_min_keywords: int,
    *,
    refit_min_keywords: int = DEFAULT_REFIT_MIN_KEYWORDS,
    max_keywords: int | None = None,
) -> int:
    """Lower the SFFS floor on final union-fit without raising the selected trial.

    Keyword-trial selection stays frozen. The final refit may accept fewer
    keywords so union-fit can keep stable 4-5 word dictionaries instead of
    discarding them for missing the trial floor of 6.
    """
    if trial_min_keywords < 1:
        raise ValueError("trial min_keywords must be at least 1")
    if refit_min_keywords < 1:
        raise ValueError("refit min_keywords must be at least 1")
    floor = min(int(trial_min_keywords), int(refit_min_keywords))
    if max_keywords is not None:
        if max_keywords < 1:
            raise ValueError("max_keywords must be at least 1")
        floor = min(floor, int(max_keywords))
    return floor


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


def _belongs_to_node(
    nodes: Mapping[str, object],
    leaf_id: str | None,
    node_id: str,
) -> bool:
    current = leaf_id
    while current is not None:
        if current == node_id:
            return True
        current = nodes[current].parent_id  # type: ignore[attr-defined]
    return False


def _coverage_for_keywords(
    node_id: str,
    train_ids: Sequence[str],
    dev_ids: Sequence[str],
    evidence: Mapping[str, CachedDocumentEvidence],
    keywords: Sequence[str],
    candidate_count: int,
    minimum_keywords: int,
    *,
    fit_ids: Sequence[str] = (),
    fit_train_count: int = 0,
    fit_dev_count: int = 0,
) -> TopicKeywordCoverage:
    selected = {item.casefold() for item in keywords}
    support = {
        word: {
            doc_id
            for doc_id in train_ids
            if any(
                candidate.word.casefold() == word
                for candidate in evidence[doc_id].candidates
            )
        }
        for word in selected
    }
    covered = set().union(*support.values()) if support else set()
    mean_support = (
        sum(len(doc_ids) for doc_ids in support.values()) / len(support)
        if support
        else 0.0
    )
    return TopicKeywordCoverage(
        node_id=node_id,
        train_documents=len(train_ids),
        dev_documents=len(dev_ids),
        candidate_count=candidate_count,
        selected_keyword_count=len(selected),
        covered_train_documents=len(covered),
        train_document_coverage=(len(covered) / len(train_ids) if train_ids else 0.0),
        mean_selected_document_support=mean_support,
        keyword_floor_reached=len(selected) >= minimum_keywords,
        fit_documents=len(fit_ids),
        fit_train_documents=fit_train_count,
        fit_dev_documents=fit_dev_count,
    )


def _empty_coverage(
    node_id: str,
    train_ids: Sequence[str],
    dev_ids: Sequence[str],
    candidate_count: int,
    *,
    fit_ids: Sequence[str] = (),
    fit_train_count: int = 0,
    fit_dev_count: int = 0,
) -> TopicKeywordCoverage:
    return TopicKeywordCoverage(
        node_id=node_id,
        train_documents=len(train_ids),
        dev_documents=len(dev_ids),
        candidate_count=candidate_count,
        selected_keyword_count=0,
        covered_train_documents=0,
        train_document_coverage=0.0,
        mean_selected_document_support=0.0,
        keyword_floor_reached=False,
        fit_documents=len(fit_ids),
        fit_train_documents=fit_train_count,
        fit_dev_documents=fit_dev_count,
    )


def _profile_from_outcome(
    outcome: TuningOutcome,
    *,
    train_document_count: int,
    dev_document_count: int,
) -> TopicKeywordProfile:
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
    return TopicKeywordProfile(
        keywords=keywords,
        score_chunk_strategy=outcome.strategy.score_chunk_strategy,
        choose_cluster_strategy=outcome.strategy.choose_cluster_strategy,
        choose_answer_strategy=outcome.strategy.choose_answer_strategy,
        train_document_count=max(1, train_document_count),
        dev_document_count=max(1, dev_document_count),
        objective=outcome.objective,
        split_hash=str(outcome.tuning_metadata()["split_hash"]),
        training_fingerprint=outcome.fingerprint,
    )


def _refit_internal_split(
    fit_ids: tuple[str, ...],
    *,
    node_id: str,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    node_seed = seed + int.from_bytes(
        hashlib.sha256(node_id.encode("utf-8")).digest()[:4],
        "big",
    )
    internal = deterministic_document_split(
        fit_ids,
        seed=node_seed,
        train_fraction=0.85,
        dev_fraction=0.15,
    )
    sffs_train = internal["train"]
    sffs_dev = internal["dev"]
    if not sffs_train or not sffs_dev:
        return fit_ids, fit_ids
    return sffs_train, sffs_dev


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
    """Tune static-style dictionaries for leaves and eligible ancestors.

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
    coverage: dict[str, TopicKeywordCoverage] = {}
    updated_nodes = []
    nodes = artifact.node_map
    leaf_count = sum(not node.children for node in artifact.nodes)

    for node in artifact.nodes:
        if node.node_id == artifact.root_id and leaf_count == 1:
            updated_nodes.append(replace(node, keyword_profile=None))
            continue
        clean_node = replace(node, keyword_profile=None)
        train_ids = tuple(
            sorted(
                item.doc_id
                for item in train_documents
                if _belongs_to_node(nodes, assignments.get(item.doc_id), node.node_id)
            )
        )
        dev_ids = tuple(
            sorted(
                item.doc_id
                for item in dev_documents
                if _belongs_to_node(nodes, assignments.get(item.doc_id), node.node_id)
            )
        )
        candidate_words = {
            candidate.word.casefold()
            for doc_id in train_ids
            for candidate in by_evidence[doc_id].candidates
        }
        if len(train_ids) < min_train_documents:
            skipped[node.node_id] = (
                f"insufficient train documents: {len(train_ids)} < {min_train_documents}"
            )
            coverage[node.node_id] = _empty_coverage(
                node.node_id,
                train_ids,
                dev_ids,
                len(candidate_words),
            )
            updated_nodes.append(clean_node)
            continue
        if len(dev_ids) < min_dev_documents:
            skipped[node.node_id] = (
                f"insufficient dev documents: {len(dev_ids)} < {min_dev_documents}"
            )
            coverage[node.node_id] = _empty_coverage(
                node.node_id,
                train_ids,
                dev_ids,
                len(candidate_words),
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
        node_evidence = [by_evidence[doc_id] for doc_id in (*train_ids, *dev_ids)]
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
            coverage[node.node_id] = _empty_coverage(
                node.node_id,
                train_ids,
                dev_ids,
                len(candidate_words),
            )
            updated_nodes.append(clean_node)
            continue
        if not outcome.keywords or not math.isfinite(outcome.objective):
            skipped[node.node_id] = "SFFS produced no finite non-empty profile"
            coverage[node.node_id] = _empty_coverage(
                node.node_id,
                train_ids,
                dev_ids,
                len(candidate_words),
            )
            updated_nodes.append(clean_node)
            continue
        if len(outcome.keywords) < config.min_keywords:
            skipped[node.node_id] = (
                f"keyword floor not reached: {len(outcome.keywords)} "
                f"< {config.min_keywords}"
            )
            coverage[node.node_id] = _coverage_for_keywords(
                node.node_id,
                train_ids,
                dev_ids,
                by_evidence,
                outcome.keywords,
                len(candidate_words),
                config.min_keywords,
            )
            updated_nodes.append(clean_node)
            continue

        profile = _profile_from_outcome(
            outcome,
            train_document_count=len(train_ids),
            dev_document_count=len(dev_ids),
        )
        outcomes[node.node_id] = outcome
        coverage[node.node_id] = _coverage_for_keywords(
            node.node_id,
            train_ids,
            dev_ids,
            by_evidence,
            outcome.keywords,
            len(candidate_words),
            config.min_keywords,
        )
        updated_nodes.append(replace(clean_node, keyword_profile=profile))

    return TopicKeywordTrainingResult(
        artifact=replace(artifact, nodes=tuple(updated_nodes)),
        outcomes=outcomes,
        skipped_nodes=skipped,
        assignments=assignments,
        coverage=coverage,
    )


def refit_topic_keyword_profiles(
    artifact: TopicArtifact,
    fit_documents: Sequence[TopicDocument],
    evidence: Sequence[CachedDocumentEvidence],
    encoder: SentenceEncoder,
    *,
    config: TrainingConfig,
    metric_cache: ExtractionMetricCache,
    objective_config: ObjectiveConfig = ObjectiveConfig(),
    metric_weights: MetricWeights = MetricWeights(),
    checkpoint_dir: str | Path | None = None,
    min_fit_documents: int = 4,
    expected_encoder_name: str | None = None,
    expected_encoder_revision: str | None = None,
) -> TopicKeywordTrainingResult:
    """Refit leaf and ancestor dictionaries on train+dev after hyper-parameters freeze.

    Clusters, centroids and routing stay exactly as in ``artifact``. Candidate
    terms come from all fit documents assigned to each subtree; SFFS uses an
    internal deterministic holdout of those fit documents. Test documents are
    rejected. Train KMeans membership is reused; remaining fit documents are
    routed to the nearest leaf without applying the frozen confidence gate.
    """
    if min_fit_documents < 1:
        raise ValueError("minimum fit document count must be positive")
    invalid = [
        item.doc_id
        for item in fit_documents
        if item.split.casefold() not in {"train", "dev"}
    ]
    if invalid:
        raise ValueError(
            "keyword refit accepts train/dev documents only; "
            f"invalid ids: {invalid[:5]}"
        )
    if len({item.doc_id for item in fit_documents}) != len(fit_documents):
        raise ValueError("keyword refit document ids must be unique")
    by_evidence = {item.doc_id: item for item in evidence}
    required_ids = {item.doc_id for item in fit_documents}
    missing = sorted(required_ids.difference(by_evidence))
    if missing:
        raise ValueError(f"missing cached keyword evidence for documents: {missing[:5]}")

    train_documents = [item for item in fit_documents if item.split.casefold() == "train"]
    stored_train_assignments = {
        item.doc_id: item.leaf_id for item in artifact.train_assignments
    }
    assignments = {
        item.doc_id: stored_train_assignments[item.doc_id]
        for item in train_documents
        if item.doc_id in stored_train_assignments
    }
    assignment_artifact = replace(
        artifact,
        routing=replace(
            artifact.routing,
            similarity_threshold=-1.0,
            min_margin=0.0,
        ),
    )
    assignments.update(
        assign_documents_to_leaf_topics(
            [item for item in fit_documents if item.doc_id not in assignments],
            encoder,
            assignment_artifact,
            expected_encoder_name=expected_encoder_name,
            expected_encoder_revision=expected_encoder_revision,
        )
    )

    outcomes: dict[str, TuningOutcome] = {}
    skipped: dict[str, str] = {}
    coverage: dict[str, TopicKeywordCoverage] = {}
    updated_nodes = []
    nodes = artifact.node_map
    leaf_count = sum(not node.children for node in artifact.nodes)

    for node in artifact.nodes:
        if node.node_id == artifact.root_id and leaf_count == 1:
            updated_nodes.append(replace(node, keyword_profile=None))
            continue
        clean_node = replace(node, keyword_profile=None)
        fit_ids = tuple(
            sorted(
                item.doc_id
                for item in fit_documents
                if _belongs_to_node(nodes, assignments.get(item.doc_id), node.node_id)
            )
        )
        fit_train_count = sum(
            1
            for doc_id in fit_ids
            if any(item.doc_id == doc_id and item.split.casefold() == "train" for item in fit_documents)
        )
        fit_dev_count = len(fit_ids) - fit_train_count
        train_ids = tuple(
            sorted(
                item.doc_id
                for item in train_documents
                if _belongs_to_node(nodes, assignments.get(item.doc_id), node.node_id)
            )
        )
        candidate_words = {
            candidate.word.casefold()
            for doc_id in fit_ids
            for candidate in by_evidence[doc_id].candidates
        }
        if len(fit_ids) < min_fit_documents:
            skipped[node.node_id] = (
                f"insufficient fit documents: {len(fit_ids)} < {min_fit_documents}"
            )
            coverage[node.node_id] = _empty_coverage(
                node.node_id,
                train_ids,
                (),
                len(candidate_words),
                fit_ids=fit_ids,
                fit_train_count=fit_train_count,
                fit_dev_count=fit_dev_count,
            )
            updated_nodes.append(clean_node)
            continue

        sffs_train_ids, sffs_dev_ids = _refit_internal_split(
            fit_ids,
            node_id=node.node_id,
            seed=config.seed,
        )
        node_config = replace(
            config,
            min_document_support=min(config.min_document_support, len(fit_ids)),
        )
        node_checkpoint = (
            Path(checkpoint_dir) / node.node_id
            if checkpoint_dir is not None
            else None
        )
        node_evidence = [by_evidence[doc_id] for doc_id in fit_ids]
        try:
            outcome = tune_global_keywords(
                node_evidence,
                config=node_config,
                metric_cache=metric_cache,
                metric_weights=metric_weights,
                objective_config=objective_config,
                split={
                    "train": sffs_train_ids,
                    "dev": sffs_dev_ids,
                    "test": (),
                },
                candidate_doc_ids=fit_ids,
                checkpoint_dir=node_checkpoint,
            )
        except ValueError as error:
            if "Candidate pool is empty" not in str(error):
                raise
            skipped[node.node_id] = str(error)
            coverage[node.node_id] = _empty_coverage(
                node.node_id,
                train_ids,
                sffs_dev_ids,
                len(candidate_words),
                fit_ids=fit_ids,
                fit_train_count=fit_train_count,
                fit_dev_count=fit_dev_count,
            )
            updated_nodes.append(clean_node)
            continue
        if not outcome.keywords or not math.isfinite(outcome.objective):
            skipped[node.node_id] = "SFFS produced no finite non-empty profile"
            coverage[node.node_id] = _empty_coverage(
                node.node_id,
                train_ids,
                sffs_dev_ids,
                len(candidate_words),
                fit_ids=fit_ids,
                fit_train_count=fit_train_count,
                fit_dev_count=fit_dev_count,
            )
            updated_nodes.append(clean_node)
            continue
        if len(outcome.keywords) < config.min_keywords:
            skipped[node.node_id] = (
                f"keyword floor not reached: {len(outcome.keywords)} "
                f"< {config.min_keywords}"
            )
            coverage[node.node_id] = _coverage_for_keywords(
                node.node_id,
                train_ids,
                sffs_dev_ids,
                by_evidence,
                outcome.keywords,
                len(candidate_words),
                config.min_keywords,
                fit_ids=fit_ids,
                fit_train_count=fit_train_count,
                fit_dev_count=fit_dev_count,
            )
            updated_nodes.append(clean_node)
            continue

        original_dev_count = sum(
            1
            for item in fit_documents
            if item.split.casefold() == "dev"
            and _belongs_to_node(nodes, assignments.get(item.doc_id), node.node_id)
        )
        profile = _profile_from_outcome(
            outcome,
            train_document_count=len(sffs_train_ids),
            dev_document_count=max(1, original_dev_count or len(sffs_dev_ids)),
        )
        outcomes[node.node_id] = outcome
        coverage[node.node_id] = _coverage_for_keywords(
            node.node_id,
            train_ids,
            sffs_dev_ids,
            by_evidence,
            outcome.keywords,
            len(candidate_words),
            config.min_keywords,
            fit_ids=fit_ids,
            fit_train_count=fit_train_count,
            fit_dev_count=fit_dev_count,
        )
        updated_nodes.append(replace(clean_node, keyword_profile=profile))

    return TopicKeywordTrainingResult(
        artifact=replace(artifact, nodes=tuple(updated_nodes), routing=artifact.routing),
        outcomes=outcomes,
        skipped_nodes=skipped,
        assignments=assignments,
        coverage=coverage,
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
