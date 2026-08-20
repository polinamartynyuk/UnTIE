"""Hierarchical topic adaptation for extractive-QA chunk reranking.

The module is deliberately independent from gold answers.  Topic artifacts are
built from train documents only and are consumed as a cheap inference layer.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import csv
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from .domain import ScoredChunk, TextChunk
from .protocols import SentenceEncoder
from .ranking import WeightedKeyword


TOPIC_SCHEMA_VERSION = 1
_GENERIC_TERMS = frozenset(
    {
        "analysis",
        "approach",
        "based",
        "data",
        "method",
        "methods",
        "model",
        "paper",
        "result",
        "results",
        "section",
        "study",
        "system",
        "task",
        "using",
        "work",
    }
)


class TopicArtifactError(ValueError):
    """Raised when a topic artifact is malformed or incompatible."""


class TopicFallbackReason(str, Enum):
    LOW_TOPIC_CONFIDENCE = "LOW_TOPIC_CONFIDENCE"
    NO_TOPIC_MATCHES = "NO_TOPIC_MATCHES"
    NO_TOPIC_KEYWORDS = "NO_TOPIC_KEYWORDS"
    NO_TOPIC_KEYWORD_MATCHES = "NO_TOPIC_KEYWORD_MATCHES"
    INCOMPATIBLE_TOPIC_ARTIFACT = "INCOMPATIBLE_TOPIC_ARTIFACT"
    NO_RERANKED_CHUNKS = "NO_RERANKED_CHUNKS"
    STATIC_KEYWORDS = "STATIC_KEYWORDS"
    BASELINE = "BASELINE"


@dataclass(frozen=True)
class TopicDocument:
    doc_id: str
    text: str = ""
    title: str = ""
    abstract: str = ""
    headings: tuple[str, ...] = ()
    split: str = ""


@dataclass(frozen=True)
class TopicTerm:
    term: str
    weight: float
    support: float
    child_coverage: float = 1.0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TopicTerm":
        return cls(
            term=str(value["term"]),
            weight=_finite(value["weight"], "topic term weight"),
            support=_probability(value["support"], "topic term support"),
            child_coverage=_probability(
                value.get("child_coverage", 1.0), "topic term child coverage"
            ),
        )


@dataclass(frozen=True)
class TopicKeywordProfile:
    keywords: tuple[WeightedKeyword, ...]
    score_chunk_strategy: str = "equal_weight_score_diff"
    choose_cluster_strategy: str = "weighted_score"
    choose_answer_strategy: str = "combined_score"
    train_document_count: int = 0
    dev_document_count: int = 0
    objective: float = 0.0
    split_hash: str = ""
    training_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.keywords:
            raise ValueError("topic keyword profile must contain keywords")
        words = [item.word.casefold().strip() for item in self.keywords]
        if any(not word for word in words) or len(set(words)) != len(words):
            raise ValueError("topic keyword profile words must be non-empty and unique")
        if any(
            not math.isfinite(value)
            for item in self.keywords
            for value in (item.attention_weight, item.score_difference)
        ):
            raise ValueError("topic keyword profile weights must be finite")
        if self.score_chunk_strategy not in {
            "only_score_diff",
            "only_weight",
            "equal_weight_score_diff",
        }:
            raise ValueError("invalid topic keyword score strategy")
        if self.choose_cluster_strategy not in {
            "highest_avg_score",
            "weighted_score",
            "highest_cohesion",
        }:
            raise ValueError("invalid topic keyword cluster strategy")
        if self.choose_answer_strategy not in {
            "highest_chunk_score",
            "highest_similarity",
            "combined_score",
        }:
            raise ValueError("invalid topic keyword answer strategy")
        if self.train_document_count < 1 or self.dev_document_count < 1:
            raise ValueError("topic keyword profile requires positive train/dev counts")
        if not math.isfinite(self.objective):
            raise ValueError("topic keyword profile objective must be finite")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TopicKeywordProfile":
        keywords = tuple(
            WeightedKeyword(
                word=str(item["word"]),
                lemma=str(item.get("lemma") or item["word"]).casefold(),
                stem=str(item.get("stem") or item["word"]).casefold(),
                attention_weight=_finite(
                    item.get("attention_weight", 1.0), "keyword attention_weight"
                ),
                score_difference=_finite(
                    item.get("score_difference", 1.0), "keyword score_difference"
                ),
            )
            for item in _list(value.get("keywords", []), "keyword_profile.keywords")
        )
        return cls(
            keywords=keywords,
            score_chunk_strategy=str(
                value.get("score_chunk_strategy", "equal_weight_score_diff")
            ),
            choose_cluster_strategy=str(
                value.get("choose_cluster_strategy", "weighted_score")
            ),
            choose_answer_strategy=str(
                value.get("choose_answer_strategy", "combined_score")
            ),
            train_document_count=_integer(
                value.get("train_document_count", 0),
                "keyword_profile.train_document_count",
                minimum=0,
            ),
            dev_document_count=_integer(
                value.get("dev_document_count", 0),
                "keyword_profile.dev_document_count",
                minimum=0,
            ),
            objective=_finite(value.get("objective", 0.0), "keyword_profile.objective"),
            split_hash=str(value.get("split_hash", "")),
            training_fingerprint=str(value.get("training_fingerprint", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "keywords": [asdict(item) for item in self.keywords],
            "score_chunk_strategy": self.score_chunk_strategy,
            "choose_cluster_strategy": self.choose_cluster_strategy,
            "choose_answer_strategy": self.choose_answer_strategy,
            "train_document_count": self.train_document_count,
            "dev_document_count": self.dev_document_count,
            "objective": self.objective,
            "split_hash": self.split_hash,
            "training_fingerprint": self.training_fingerprint,
        }


@dataclass(frozen=True)
class TopicNode:
    node_id: str
    parent_id: str | None
    depth: int
    document_count: int
    centroid: tuple[float, ...]
    representative_terms: tuple[TopicTerm, ...] = ()
    children: tuple[str, ...] = ()
    keyword_profile: TopicKeywordProfile | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TopicNode":
        centroid = tuple(
            _finite(item, f"centroid[{index}]")
            for index, item in enumerate(value.get("centroid", ()))
        )
        if not centroid:
            raise TopicArtifactError("topic node centroid cannot be empty")
        count = _integer(value.get("document_count"), "document_count", minimum=1)
        depth = _integer(value.get("depth"), "depth", minimum=0)
        terms = tuple(
            TopicTerm.from_dict(item)
            for item in _list(value.get("representative_terms", []), "representative_terms")
        )
        return cls(
            node_id=str(value["id"]),
            parent_id=(
                str(value["parent_id"]) if value.get("parent_id") is not None else None
            ),
            depth=depth,
            document_count=count,
            centroid=centroid,
            representative_terms=terms,
            children=tuple(str(item) for item in _list(value.get("children", []), "children")),
            keyword_profile=(
                TopicKeywordProfile.from_dict(
                    _mapping(value["keyword_profile"], "keyword_profile")
                )
                if value.get("keyword_profile") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "document_count": self.document_count,
            "centroid": list(self.centroid),
            "representative_terms": [asdict(item) for item in self.representative_terms],
            "children": list(self.children),
            "keyword_profile": (
                self.keyword_profile.to_dict() if self.keyword_profile else None
            ),
        }


@dataclass(frozen=True)
class TopicTrainAssignment:
    doc_id: str
    leaf_id: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TopicTrainAssignment":
        return cls(doc_id=str(value["doc_id"]), leaf_id=str(value["leaf_id"]))


@dataclass(frozen=True)
class TopicBuildConfig:
    leaf_clusters: int = 4
    ngram_max: int = 2
    min_topic_document_support: float = 0.2
    topic_term_top_k: int = 30
    specificity_epsilon: float = 0.5
    seed: int = 42
    stop_words: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.leaf_clusters < 1 or self.ngram_max < 1:
            raise ValueError("leaf_clusters and ngram_max must be positive")
        if not 0 <= self.min_topic_document_support <= 1:
            raise ValueError("min_topic_document_support must be between zero and one")
        if self.topic_term_top_k < 1 or self.specificity_epsilon <= 0:
            raise ValueError("topic_term_top_k and specificity_epsilon must be positive")


@dataclass(frozen=True)
class TopicRoutingConfig:
    top_k: int = 3
    beam_width: int = 2
    temperature: float = 0.1
    similarity_threshold: float = 0.0
    min_margin: float = 0.0

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (
                self.temperature,
                self.similarity_threshold,
                self.min_margin,
            )
        ):
            raise ValueError("routing values must be finite")
        if self.top_k < 1 or self.beam_width < 1 or self.temperature <= 0:
            raise ValueError("routing top_k/beam_width/temperature must be positive")
        if not -1 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between -1 and 1")
        if self.min_margin < 0:
            raise ValueError("min_margin cannot be negative")


@dataclass(frozen=True)
class TopicMixingConfig:
    aspect_threshold: float = 0.0
    theta_aspect: float = 1.0
    theta_topic: float = 0.5
    theta_interaction: float = 0.5
    weight_ratio: float = 0.5
    max_qa_chunks: int = 0
    filter_unmatched_chunks: bool = True
    minimum_topic_matches: int = 1

    def __post_init__(self) -> None:
        for name in (
            "aspect_threshold",
            "theta_aspect",
            "theta_topic",
            "theta_interaction",
            "weight_ratio",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("theta_aspect", "theta_topic", "theta_interaction"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not 0 <= self.weight_ratio <= 1:
            raise ValueError("weight_ratio must be between zero and one")
        if self.max_qa_chunks < 0:
            raise ValueError("max_qa_chunks cannot be negative")
        if isinstance(self.minimum_topic_matches, bool) or not isinstance(
            self.minimum_topic_matches, int
        ):
            raise ValueError("minimum_topic_matches must be an integer")
        if self.minimum_topic_matches < 0:
            raise ValueError("minimum_topic_matches cannot be negative")
        if self.filter_unmatched_chunks and self.minimum_topic_matches < 1:
            raise ValueError(
                "minimum_topic_matches must be at least 1 when filter_unmatched_chunks is enabled"
            )


@dataclass(frozen=True)
class TopicArtifact:
    encoder_name: str
    nodes: tuple[TopicNode, ...]
    root_id: str
    build: TopicBuildConfig
    train_assignments: tuple[TopicTrainAssignment, ...] = ()
    routing: TopicRoutingConfig = field(default_factory=TopicRoutingConfig)
    mixing: TopicMixingConfig = field(default_factory=TopicMixingConfig)
    encoder_revision: str = ""
    normalization: str = "l2"
    schema_version: int = TOPIC_SCHEMA_VERSION
    representation_fields: tuple[str, ...] = ("title", "abstract", "headings")
    fallback_policy: str = "metadata_then_first_content_paragraphs"

    def __post_init__(self) -> None:
        if self.schema_version != TOPIC_SCHEMA_VERSION:
            raise TopicArtifactError(
                f"unsupported topic schema_version: {self.schema_version}"
            )
        if self.normalization != "l2":
            raise TopicArtifactError("only l2-normalized topic artifacts are supported")
        if not self.encoder_name.strip():
            raise TopicArtifactError("topic artifact encoder name cannot be empty")
        if self.representation_fields != ("title", "abstract", "headings"):
            raise TopicArtifactError("unsupported document representation fields")
        if self.fallback_policy != "metadata_then_first_content_paragraphs":
            raise TopicArtifactError("unsupported document representation fallback policy")
        _validate_tree(self.nodes, self.root_id)
        if self.train_assignments:
            if len({item.doc_id for item in self.train_assignments}) != len(
                self.train_assignments
            ):
                raise TopicArtifactError("train assignment doc_ids must be unique")
            nodes = self.node_map
            counts: dict[str, int] = {}
            for assignment in self.train_assignments:
                node = nodes.get(assignment.leaf_id)
                if node is None or node.children:
                    raise TopicArtifactError(
                        "train assignments must reference leaf topic nodes"
                    )
                counts[assignment.leaf_id] = counts.get(assignment.leaf_id, 0) + 1
            for node in self.nodes:
                if not node.children and counts.get(node.node_id, 0) != node.document_count:
                    raise TopicArtifactError(
                        f"train assignment count mismatch for {node.node_id}"
                    )

    @property
    def node_map(self) -> dict[str, TopicNode]:
        return {node.node_id: node for node in self.nodes}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "encoder": {
                "name": self.encoder_name,
                "revision": self.encoder_revision,
                "normalization": self.normalization,
            },
            "document_representation": {
                "fields": list(self.representation_fields),
                "fallback_policy": self.fallback_policy,
            },
            "clustering": {
                "algorithm": "kmeans+deterministic-centroid-agglomeration",
                "seed": self.build.seed,
                "params": asdict(self.build),
            },
            "train_assignments": [asdict(item) for item in self.train_assignments],
            "routing": asdict(self.routing),
            "mixing": asdict(self.mixing),
            "root_id": self.root_id,
            "nodes": [node.to_dict() for node in self.nodes],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TopicArtifact":
        version = _integer(value.get("schema_version"), "schema_version", minimum=1)
        if version != TOPIC_SCHEMA_VERSION:
            raise TopicArtifactError(f"unsupported topic schema_version: {version}")
        encoder = _mapping(value.get("encoder"), "encoder")
        representation = _mapping(
            value.get("document_representation", {}), "document_representation"
        )
        clustering = _mapping(value.get("clustering"), "clustering")
        params = dict(_mapping(clustering.get("params", {}), "clustering.params"))
        params["stop_words"] = tuple(params.get("stop_words", ()))
        routing = _mapping(value.get("routing", {}), "routing")
        mixing = _mapping(value.get("mixing", {}), "mixing")
        artifact = cls(
            schema_version=version,
            encoder_name=str(encoder.get("name", "")),
            encoder_revision=str(encoder.get("revision", "")),
            normalization=str(encoder.get("normalization", "")),
            representation_fields=tuple(
                str(item)
                for item in _list(
                    representation.get("fields", ["title", "abstract", "headings"]),
                    "document_representation.fields",
                )
            ),
            fallback_policy=str(
                representation.get(
                    "fallback_policy", "metadata_then_first_content_paragraphs"
                )
            ),
            build=TopicBuildConfig(**params),
            train_assignments=tuple(
                TopicTrainAssignment.from_dict(item)
                for item in _list(
                    value.get("train_assignments", []), "train_assignments"
                )
            ),
            routing=TopicRoutingConfig(**routing),
            mixing=TopicMixingConfig(**mixing),
            root_id=str(value.get("root_id", "")),
            nodes=tuple(
                TopicNode.from_dict(item)
                for item in _list(value.get("nodes"), "nodes")
            ),
        )
        return artifact


@dataclass(frozen=True)
class RoutedTopic:
    node_id: str
    similarity: float
    weight: float
    depth: int


@dataclass(frozen=True)
class TopicRoute:
    topics: tuple[RoutedTopic, ...]
    fallback_reason: TopicFallbackReason | None = None
    embedding: tuple[float, ...] = ()


@dataclass(frozen=True)
class TopicTuningTrial:
    routing: TopicRoutingConfig
    mixing: TopicMixingConfig
    objective: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class TopicTuningResult:
    artifact: TopicArtifact
    trials: tuple[TopicTuningTrial, ...]


def document_topic_representation(
    document: TopicDocument, *, max_fallback_paragraphs: int = 3
) -> str:
    """Build the same deterministic topic representation offline and online."""
    title = _clean(document.title)
    abstract = _clean(document.abstract)
    headings = tuple(_clean(item) for item in document.headings if _clean(item))
    metadata_parts = tuple(item for item in (title, abstract, *headings) if item)
    if metadata_parts:
        return "\n".join(metadata_parts)

    paragraphs = [
        _clean(item)
        for item in re.split(r"\n\s*\n|\r\n\s*\r\n", document.text)
        if _clean(item)
    ]
    if not paragraphs:
        paragraphs = [_clean(document.text)] if _clean(document.text) else []
    return "\n".join(paragraphs[:max_fallback_paragraphs])


def infer_plain_text_metadata(text: str) -> TopicDocument:
    """Extract only cheap, deterministic metadata hints from a plain-text file."""
    lines = [_clean(line) for line in text.splitlines() if _clean(line)]
    title = lines[0] if lines and len(lines[0].split()) <= 30 else ""
    headings = tuple(
        line
        for line in lines[1:]
        if len(line.split()) <= 12
        and (line.isupper() or bool(re.match(r"^\d+(?:\.\d+)*\s+", line)))
    )
    return TopicDocument("", text=text, title=title, headings=headings[:20])


def build_topic_artifact(
    documents: Sequence[TopicDocument],
    encoder: SentenceEncoder,
    *,
    encoder_name: str,
    encoder_revision: str = "",
    config: TopicBuildConfig = TopicBuildConfig(),
    routing: TopicRoutingConfig = TopicRoutingConfig(),
    mixing: TopicMixingConfig = TopicMixingConfig(),
) -> TopicArtifact:
    """Build a deterministic topic artifact from an explicitly train-only corpus."""
    docs = tuple(sorted(documents, key=lambda item: item.doc_id))
    if not docs:
        raise ValueError("topic building requires at least one train document")
    if len({item.doc_id for item in docs}) != len(docs):
        raise ValueError("topic document ids must be unique")
    non_train = [item.doc_id for item in docs if item.split.casefold() != "train"]
    if non_train:
        raise ValueError(
            "topic artifacts may be built from train documents only; "
            f"non-train ids: {non_train[:5]}"
        )
    representations = [document_topic_representation(item) for item in docs]
    if any(not text for text in representations):
        empty = [docs[index].doc_id for index, text in enumerate(representations) if not text]
        raise ValueError(f"documents have no topic representation: {empty[:5]}")
    embeddings = _normalize_rows(np.asarray(encoder.encode(representations), dtype=float))
    if embeddings.ndim != 2 or embeddings.shape[0] != len(docs):
        raise ValueError("encoder returned an incompatible document embedding matrix")

    leaf_count = min(config.leaf_clusters, len(docs))
    if leaf_count == 1:
        labels = np.zeros(len(docs), dtype=int)
    else:
        labels = KMeans(
            n_clusters=leaf_count,
            random_state=config.seed,
            n_init=10,
        ).fit_predict(embeddings)

    members: dict[str, tuple[int, ...]] = {}
    centroids: dict[str, np.ndarray] = {}
    children: dict[str, tuple[str, ...]] = {}
    parents: dict[str, str | None] = {}
    active: list[str] = []
    for label in sorted(set(int(item) for item in labels)):
        node_id = f"topic_leaf_{label:03d}"
        indices = tuple(int(index) for index in np.flatnonzero(labels == label))
        members[node_id] = indices
        centroids[node_id] = _normalized_mean(embeddings[list(indices)])
        children[node_id] = ()
        parents[node_id] = None
        active.append(node_id)

    merge_index = 0
    while len(active) > 1:
        pairs = []
        for left_index, left in enumerate(active):
            for right in active[left_index + 1 :]:
                similarity = float(np.dot(centroids[left], centroids[right]))
                pairs.append((-similarity, left, right))
        _, left, right = min(pairs)
        parent = f"topic_parent_{merge_index:03d}"
        merge_index += 1
        parent_members = tuple(sorted((*members[left], *members[right])))
        members[parent] = parent_members
        centroids[parent] = _normalized_mean(embeddings[list(parent_members)])
        children[parent] = tuple(sorted((left, right)))
        parents[parent] = None
        parents[left] = parent
        parents[right] = parent
        active = sorted([item for item in active if item not in {left, right}] + [parent])

    if len(active) == 1 and len(members) == 1:
        leaf = active[0]
        root_id = "topic_root"
        members[root_id] = members[leaf]
        centroids[root_id] = centroids[leaf]
        children[root_id] = (leaf,)
        parents[root_id] = None
        parents[leaf] = root_id
    else:
        root_id = active[0]

    depths: dict[str, int] = {}

    def assign_depth(node_id: str, depth: int) -> None:
        depths[node_id] = depth
        for child in children[node_id]:
            assign_depth(child, depth + 1)

    assign_depth(root_id, 0)
    terms_by_node = _build_topic_terms(docs, members, children, root_id, config)
    nodes = tuple(
        TopicNode(
            node_id=node_id,
            parent_id=parents[node_id],
            depth=depths[node_id],
            document_count=len(members[node_id]),
            centroid=tuple(float(item) for item in centroids[node_id]),
            representative_terms=terms_by_node[node_id],
            children=children[node_id],
        )
        for node_id in sorted(members, key=lambda item: (depths[item], item))
    )
    train_assignments = tuple(
        TopicTrainAssignment(
            doc_id=document.doc_id,
            leaf_id=f"topic_leaf_{int(labels[index]):03d}",
        )
        for index, document in enumerate(docs)
    )
    return TopicArtifact(
        encoder_name=encoder_name,
        encoder_revision=encoder_revision,
        nodes=nodes,
        root_id=root_id,
        build=config,
        train_assignments=train_assignments,
        routing=routing,
        mixing=mixing,
    )


def route_topic_embedding(
    embedding: Sequence[float] | np.ndarray,
    artifact: TopicArtifact,
) -> TopicRoute:
    vector = _normalize_vector(np.asarray(embedding, dtype=float))
    nodes = artifact.node_map
    if vector.size != len(nodes[artifact.root_id].centroid):
        raise TopicArtifactError("document embedding dimension does not match artifact")
    beam = [artifact.root_id]
    terminal: list[tuple[str, float]] = []
    reason: TopicFallbackReason | None = None
    while beam:
        candidates: list[tuple[float, str]] = []
        next_terminal: list[tuple[str, float]] = []
        for node_id in beam:
            node = nodes[node_id]
            if not node.children:
                next_terminal.append((node_id, _cosine(vector, node.centroid)))
                continue
            scored = sorted(
                (
                    (_cosine(vector, nodes[child].centroid), child)
                    for child in node.children
                ),
                key=lambda item: (-item[0], item[1]),
            )
            if (
                artifact.routing.min_margin > 0
                and len(scored) > 1
                and scored[0][0] - scored[1][0] < artifact.routing.min_margin
            ):
                next_terminal.append((node_id, _cosine(vector, node.centroid)))
                reason = TopicFallbackReason.LOW_TOPIC_CONFIDENCE
                continue
            eligible = [
                item
                for item in scored
                if item[0] >= artifact.routing.similarity_threshold
            ]
            if not eligible:
                next_terminal.append((node_id, _cosine(vector, node.centroid)))
                reason = TopicFallbackReason.LOW_TOPIC_CONFIDENCE
                continue
            candidates.extend(eligible)
        terminal.extend(next_terminal)
        if not candidates:
            break
        candidates.sort(key=lambda item: (-item[0], item[1]))
        beam = [node_id for _, node_id in candidates[: artifact.routing.beam_width]]

    unique = {
        node_id: similarity
        for node_id, similarity in sorted(
            terminal, key=lambda item: (-item[1], item[0])
        )
    }
    selected = sorted(
        ((similarity, node_id) for node_id, similarity in unique.items()),
        key=lambda item: (-item[0], item[1]),
    )[: artifact.routing.top_k]
    if not selected:
        root = nodes[artifact.root_id]
        selected = [(_cosine(vector, root.centroid), root.node_id)]
        reason = TopicFallbackReason.LOW_TOPIC_CONFIDENCE
    weights = _softmax(
        np.asarray([similarity for similarity, _ in selected]),
        artifact.routing.temperature,
    )
    return TopicRoute(
        topics=tuple(
            RoutedTopic(
                node_id=node_id,
                similarity=float(similarity),
                weight=float(weights[index]),
                depth=nodes[node_id].depth,
            )
            for index, (similarity, node_id) in enumerate(selected)
        ),
        fallback_reason=reason,
        embedding=tuple(float(item) for item in vector),
    )


def route_document(
    document: TopicDocument,
    encoder: SentenceEncoder,
    artifact: TopicArtifact,
    *,
    expected_encoder_name: str | None = None,
    expected_encoder_revision: str | None = None,
) -> TopicRoute:
    if expected_encoder_name and expected_encoder_name != artifact.encoder_name:
        raise TopicArtifactError(
            f"topic artifact encoder {artifact.encoder_name!r} is incompatible with "
            f"{expected_encoder_name!r}"
        )
    if (
        expected_encoder_revision
        and artifact.encoder_revision != expected_encoder_revision
    ):
        raise TopicArtifactError(
            f"topic artifact encoder revision {artifact.encoder_revision!r} is "
            f"incompatible with {expected_encoder_revision!r}"
        )
    representation = document_topic_representation(document)
    if not representation:
        return TopicRoute((), TopicFallbackReason.LOW_TOPIC_CONFIDENCE)
    embedding = np.asarray(encoder.encode([representation]), dtype=float)
    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError("encoder returned an incompatible topic embedding")
    return route_topic_embedding(embedding[0], artifact)


def route_document_to_nearest_leaf(
    document: TopicDocument,
    encoder: SentenceEncoder,
    artifact: TopicArtifact,
    *,
    expected_encoder_name: str | None = None,
    expected_encoder_revision: str | None = None,
) -> TopicRoute:
    """Assign a document to the single nearest train-derived leaf centroid."""
    routed = route_document(
        document,
        encoder,
        artifact,
        expected_encoder_name=expected_encoder_name,
        expected_encoder_revision=expected_encoder_revision,
    )
    if not routed.embedding:
        return routed
    if routed.fallback_reason is TopicFallbackReason.LOW_TOPIC_CONFIDENCE:
        return TopicRoute(
            (),
            TopicFallbackReason.LOW_TOPIC_CONFIDENCE,
            routed.embedding,
        )
    vector = np.asarray(routed.embedding, dtype=float)
    leaves = [node for node in artifact.nodes if not node.children]
    if not leaves:
        return TopicRoute((), TopicFallbackReason.LOW_TOPIC_CONFIDENCE, routed.embedding)
    similarity, node = min(
        ((_cosine(vector, node.centroid), node) for node in leaves),
        key=lambda item: (-item[0], item[1].node_id),
    )
    if similarity < artifact.routing.similarity_threshold:
        return TopicRoute(
            (),
            TopicFallbackReason.LOW_TOPIC_CONFIDENCE,
            routed.embedding,
        )
    return TopicRoute(
        (
            RoutedTopic(
                node_id=node.node_id,
                similarity=float(similarity),
                weight=1.0,
                depth=node.depth,
            ),
        ),
        routed.fallback_reason,
        routed.embedding,
    )


def resolve_topic_keyword_profile(
    route: TopicRoute,
    artifact: TopicArtifact,
) -> tuple[str, TopicKeywordProfile] | None:
    """Resolve a node-specific dictionary, ascending toward the root if needed."""
    nodes = artifact.node_map
    ordered = sorted(
        route.topics,
        key=lambda item: (-item.similarity, item.node_id),
    )
    visited: set[str] = set()
    for routed in ordered:
        node_id: str | None = routed.node_id
        while node_id is not None and node_id not in visited:
            visited.add(node_id)
            node = nodes[node_id]
            if node.keyword_profile and node.keyword_profile.keywords:
                return node_id, node.keyword_profile
            node_id = node.parent_id
    return None


def resolve_topic_match_fallback(
    chunks: Sequence[TextChunk],
    route: TopicRoute,
    artifact: TopicArtifact,
) -> TopicRoute:
    """Ascend leaf routes until at least one routed topic term occurs."""
    if not route.topics:
        return route
    nodes = artifact.node_map
    current = route
    visited: set[tuple[str, ...]] = set()
    while current.topics:
        ids = tuple(sorted(item.node_id for item in current.topics))
        if ids in visited:
            break
        visited.add(ids)
        terms = {
            term.term.casefold()
            for item in current.topics
            for term in nodes[item.node_id].representative_terms
        }
        if terms and _chunks_contain_any(chunks, terms):
            return current
        parent_ids: set[str] = set()
        for item in current.topics:
            parent = nodes[item.node_id].parent_id
            if parent is None:
                continue
            parent_ids.add(parent)
        if not parent_ids:
            break
        if current.embedding:
            embedding = _normalize_vector(np.asarray(current.embedding, dtype=float))
            parent_similarities = {
                node_id: _cosine(embedding, nodes[node_id].centroid)
                for node_id in parent_ids
            }
        else:
            parent_similarities = {
                node_id: max(
                    item.similarity
                    for item in current.topics
                    if nodes[item.node_id].parent_id == node_id
                )
                for node_id in parent_ids
            }
        ordered_parents = sorted(
            parent_similarities, key=lambda node_id: (-parent_similarities[node_id], node_id)
        )
        weights = _softmax(
            np.asarray([parent_similarities[node_id] for node_id in ordered_parents]),
            artifact.routing.temperature,
        )
        current = TopicRoute(
            tuple(
                RoutedTopic(
                    node_id=node_id,
                    similarity=parent_similarities[node_id],
                    weight=float(weights[index]),
                    depth=nodes[node_id].depth,
                )
                for index, node_id in enumerate(ordered_parents)
            ),
            TopicFallbackReason.NO_TOPIC_MATCHES,
            current.embedding,
        )
    return TopicRoute(
        current.topics,
        TopicFallbackReason.NO_TOPIC_MATCHES,
        current.embedding,
    )


def score_topic_aware_chunks(
    chunks: Sequence[TextChunk],
    aspect_keywords: Sequence[WeightedKeyword],
    route: TopicRoute,
    artifact: TopicArtifact,
) -> list[ScoredChunk]:
    """Score and rerank chunks using routed topic terms and optional aspect keywords.

    When ``filter_unmatched_chunks`` is enabled, only chunks containing at least
    ``minimum_topic_matches`` representative terms from the routed topic nodes are
    kept for QA. This mirrors static keyword gating, but the active vocabulary is
    document-specific and derived from the topic route rather than a global list.
    """
    if not chunks:
        return []
    node_map = artifact.node_map
    mixing = artifact.mixing
    aspect_terms: dict[str, WeightedKeyword] = {}
    for keyword in aspect_keywords:
        for term in (keyword.word, keyword.lemma, keyword.stem):
            if _clean(term):
                aspect_terms[_clean(term).casefold()] = keyword
    present_aspect = {
        term
        for term in aspect_terms
        if any(_term_count(chunk.text, term) for chunk in chunks)
    }

    aspect_raw: list[float] = []
    topic_raw: list[float] = []
    aspect_matches: list[tuple[str, ...]] = []
    topic_matches: list[tuple[str, ...]] = []
    for chunk in chunks:
        matched_keywords: dict[str, WeightedKeyword] = {}
        positions: list[int] = []
        frequencies = 0
        for term, keyword in aspect_terms.items():
            matches = list(_term_pattern(term).finditer(chunk.text.casefold()))
            if matches:
                matched_keywords[keyword.word] = keyword
                positions.extend(match.start() for match in matches)
                frequencies += len(matches)
        aspect_base = sum(
            max(0.0, keyword.attention_weight) * mixing.weight_ratio
            + (
                keyword.score_difference
                if keyword.score_difference > mixing.aspect_threshold
                else 0.0
            )
            * (1.0 - mixing.weight_ratio)
            for keyword in matched_keywords.values()
        )
        if positions and aspect_base > 0:
            average_position = sum(
                1.0 - position / max(1, len(chunk.text)) for position in positions
            ) / len(positions)
            position_bonus = 1.0 + 0.3 * average_position
            frequency_bonus = 1.0 + 0.7 * math.log1p(frequencies)
            uniqueness_bonus = 1.0 + len(matched_keywords) / max(
                1, len(present_aspect)
            ) * 0.5
            aspect_base *= position_bonus * frequency_bonus * uniqueness_bonus
        aspect_raw.append(float(aspect_base))
        aspect_matches.append(tuple(sorted(matched_keywords)))

        topic_score = 0.0
        matched_topic_terms: set[str] = set()
        for routed in route.topics:
            node = node_map[routed.node_id]
            for term in node.representative_terms:
                frequency = _term_count(chunk.text, term.term.casefold())
                if frequency:
                    topic_score += (
                        routed.weight * term.weight * math.log1p(frequency)
                    )
                    matched_topic_terms.add(term.term)
        topic_raw.append(float(topic_score))
        topic_matches.append(tuple(sorted(matched_topic_terms)))

    if mixing.filter_unmatched_chunks:
        kept_indices = [
            index
            for index in range(len(chunks))
            if len(topic_matches[index]) >= mixing.minimum_topic_matches
        ]
    else:
        kept_indices = list(range(len(chunks)))
    if not kept_indices:
        return []

    aspect_normalized = _minmax_nonnegative([aspect_raw[index] for index in kept_indices])
    topic_normalized = _minmax_nonnegative([topic_raw[index] for index in kept_indices])
    scored = []
    for normalized_index, chunk_index in enumerate(kept_indices):
        chunk = chunks[chunk_index]
        aspect = aspect_normalized[normalized_index]
        topic = topic_normalized[normalized_index]
        score = (
            mixing.theta_aspect * aspect
            + mixing.theta_topic * topic
            + mixing.theta_interaction * aspect * topic
        )
        all_matches = tuple(
            sorted(
                set(aspect_matches[chunk_index]) | set(topic_matches[chunk_index])
            )
        )
        scored.append(
            ScoredChunk(
                chunk=chunk,
                score=float(score),
                matched_keywords=all_matches,
                keyword_scores={
                    "aspect_raw": aspect_raw[chunk_index],
                    "topic_raw": topic_raw[chunk_index],
                    "aspect_normalized": aspect,
                    "topic_normalized": topic,
                    "aspect_matches": float(len(aspect_matches[chunk_index])),
                    "topic_matches": float(len(topic_matches[chunk_index])),
                    "total_matches": float(len(all_matches)),
                },
                original_weights={
                    "signals": {
                        "theta_aspect": mixing.theta_aspect,
                        "theta_topic": mixing.theta_topic,
                        "theta_interaction": mixing.theta_interaction,
                    }
                },
            )
        )
    indexed = list(enumerate(scored))
    indexed.sort(key=lambda item: (-item[1].score, item[0]))
    ranked = [item for _, item in indexed]
    return (
        ranked[: mixing.max_qa_chunks]
        if mixing.max_qa_chunks
        else ranked
    )


def compute_retrieval_metrics(
    ranked_chunk_indices: Sequence[int],
    evidence_chunk_indices: Iterable[int],
    *,
    k: int,
) -> dict[str, float]:
    evidence = set(int(item) for item in evidence_chunk_indices)
    if k < 1:
        raise ValueError("k must be positive")
    if not evidence:
        return {"evidence_recall_at_k": 0.0, "mrr": 0.0}
    top = list(ranked_chunk_indices[:k])
    recall = len(evidence.intersection(top)) / len(evidence)
    first_rank = next(
        (index + 1 for index, chunk_index in enumerate(ranked_chunk_indices) if chunk_index in evidence),
        None,
    )
    return {
        "evidence_recall_at_k": float(recall),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }


def compute_qa_ratios(
    selected_chunks: Sequence[TextChunk],
    baseline_chunks: Sequence[TextChunk],
) -> dict[str, float | int]:
    selected_tokens = sum(item.token_count for item in selected_chunks)
    baseline_tokens = sum(item.token_count for item in baseline_chunks)
    return {
        "qa_chunks": len(selected_chunks),
        "baseline_chunks": len(baseline_chunks),
        "qa_tokens": selected_tokens,
        "baseline_tokens": baseline_tokens,
        "qa_chunk_ratio": (
            len(selected_chunks) / len(baseline_chunks) if baseline_chunks else 0.0
        ),
        "qa_token_ratio": (
            selected_tokens / baseline_tokens if baseline_tokens else 0.0
        ),
    }


def tune_topic_configuration(
    artifact: TopicArtifact,
    dev_documents: Sequence[TopicDocument],
    candidates: Sequence[tuple[TopicRoutingConfig, TopicMixingConfig]],
    evaluator: Callable[
        [TopicArtifact, Sequence[TopicDocument]], float | Mapping[str, float]
    ],
) -> TopicTuningResult:
    """Select routing/mixing on dev without changing train-built topic data."""
    documents = tuple(sorted(dev_documents, key=lambda item: item.doc_id))
    if not documents:
        raise ValueError("topic configuration tuning requires dev documents")
    invalid = [item.doc_id for item in documents if item.split.casefold() != "dev"]
    if invalid:
        raise ValueError(
            "topic configuration tuning accepts dev documents only; "
            f"invalid ids: {invalid[:5]}"
        )
    if not candidates:
        raise ValueError("topic configuration tuning requires candidates")
    trials = []
    for routing, mixing in candidates:
        candidate = replace(artifact, routing=routing, mixing=mixing)
        raw = evaluator(candidate, documents)
        if isinstance(raw, Mapping):
            metrics = {str(key): _finite(value, str(key)) for key, value in raw.items()}
            if "objective" not in metrics:
                raise ValueError("topic tuning evaluator metrics require 'objective'")
            objective = metrics["objective"]
        else:
            objective = _finite(raw, "objective")
            metrics = {"objective": objective}
        trials.append(TopicTuningTrial(routing, mixing, objective, metrics))
    ordered = tuple(
        sorted(
            trials,
            key=lambda item: (
                -item.objective,
                json.dumps(asdict(item.routing), sort_keys=True),
                json.dumps(asdict(item.mixing), sort_keys=True),
            ),
        )
    )
    best = ordered[0]
    return TopicTuningResult(
        replace(artifact, routing=best.routing, mixing=best.mixing),
        ordered,
    )


def save_topic_artifact_atomic(
    artifact: TopicArtifact, path: str | os.PathLike[str]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        artifact.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def load_topic_artifact(path: str | os.PathLike[str]) -> TopicArtifact:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TopicArtifactError(f"could not load topic artifact from {source}: {error}") from error
    if not isinstance(payload, dict):
        raise TopicArtifactError("topic artifact must be a JSON object")
    try:
        return TopicArtifact.from_dict(payload)
    except TopicArtifactError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise TopicArtifactError(f"invalid topic artifact: {error}") from error


def save_topic_diagnostics(
    artifact: TopicArtifact, directory: str | os.PathLike[str]
) -> dict[str, Path]:
    """Write compact, deterministic offline diagnostics next to an artifact."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    nodes_path = destination / "topic_nodes.json"
    terms_path = destination / "topic_representative_terms.csv"
    assignments_path = destination / "topic_train_assignments.json"
    summary_path = destination / "topic_build_summary.json"
    nodes_path.write_text(
        json.dumps(
            [node.to_dict() for node in artifact.nodes],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with terms_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "node_id",
                "depth",
                "term",
                "weight",
                "support",
                "child_coverage",
            ),
        )
        writer.writeheader()
        for node in artifact.nodes:
            for term in node.representative_terms:
                writer.writerow(
                    {
                        "node_id": node.node_id,
                        "depth": node.depth,
                        "term": term.term,
                        "weight": term.weight,
                        "support": term.support,
                        "child_coverage": term.child_coverage,
                    }
                )
    assignments_path.write_text(
        json.dumps(
            [asdict(item) for item in artifact.train_assignments],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": artifact.schema_version,
                "encoder": artifact.encoder_name,
                "seed": artifact.build.seed,
                "node_count": len(artifact.nodes),
                "leaf_count": sum(not node.children for node in artifact.nodes),
                "train_document_count": artifact.node_map[
                    artifact.root_id
                ].document_count,
                "train_assignment_count": len(artifact.train_assignments),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "nodes": nodes_path,
        "representative_terms": terms_path,
        "train_assignments": assignments_path,
        "summary": summary_path,
    }


def _build_topic_terms(
    documents: Sequence[TopicDocument],
    members: Mapping[str, tuple[int, ...]],
    children: Mapping[str, tuple[str, ...]],
    root_id: str,
    config: TopicBuildConfig,
) -> dict[str, tuple[TopicTerm, ...]]:
    texts = [
        document.text or document_topic_representation(document)
        for document in documents
    ]
    stop_words = sorted(_GENERIC_TERMS | {item.casefold() for item in config.stop_words})
    vectorizer = CountVectorizer(
        lowercase=True,
        binary=True,
        ngram_range=(1, config.ngram_max),
        stop_words=stop_words or None,
        token_pattern=r"(?u)\b[^\W\d_][^\W_]+\b",
    )
    try:
        occurrence = np.asarray(vectorizer.fit_transform(texts).toarray(), dtype=float)
    except ValueError:
        return {node_id: () for node_id in members}
    terms = vectorizer.get_feature_names_out()
    total_documents = len(documents)
    all_indices = set(range(total_documents))
    result: dict[str, tuple[TopicTerm, ...]] = {}
    for node_id, node_indices in members.items():
        inside = list(node_indices)
        outside = sorted(all_indices.difference(inside))
        inside_counts = occurrence[inside].sum(axis=0)
        support = inside_counts / len(inside)
        if outside:
            outside_counts = occurrence[outside].sum(axis=0)
            inside_probability = (
                inside_counts + config.specificity_epsilon
            ) / (len(inside) + 2 * config.specificity_epsilon)
            outside_probability = (
                outside_counts + config.specificity_epsilon
            ) / (len(outside) + 2 * config.specificity_epsilon)
            specificity = np.log(inside_probability / outside_probability)
        else:
            document_frequency = occurrence.sum(axis=0)
            specificity = np.log(
                (total_documents + config.specificity_epsilon)
                / (document_frequency + config.specificity_epsilon)
            ) + 1.0
        child_ids = children[node_id]
        coverage = np.ones(len(terms), dtype=float)
        if child_ids:
            child_presence = []
            for child_id in child_ids:
                child_indices = list(members[child_id])
                child_support = occurrence[child_indices].sum(axis=0) / len(child_indices)
                child_presence.append(
                    child_support >= config.min_topic_document_support
                )
            coverage = np.asarray(child_presence, dtype=float).mean(axis=0)
        weights = np.maximum(0.0, specificity) * support * coverage
        candidates = [
            TopicTerm(
                term=str(terms[index]),
                weight=float(weights[index]),
                support=float(support[index]),
                child_coverage=float(coverage[index]),
            )
            for index in range(len(terms))
            if support[index] >= config.min_topic_document_support
            and weights[index] > 0
        ]
        result[node_id] = tuple(
            sorted(
                candidates,
                key=lambda item: (-item.weight, -item.support, item.term),
            )[: config.topic_term_top_k]
        )
    if root_id not in result:
        result[root_id] = ()
    return result


def _validate_tree(nodes: Sequence[TopicNode], root_id: str) -> None:
    if not nodes:
        raise TopicArtifactError("topic artifact must contain nodes")
    by_id = {node.node_id: node for node in nodes}
    if len(by_id) != len(nodes):
        raise TopicArtifactError("topic node ids must be unique")
    if root_id not in by_id or by_id[root_id].parent_id is not None:
        raise TopicArtifactError("root_id must identify a root node")
    dimensions = {len(node.centroid) for node in nodes}
    if len(dimensions) != 1:
        raise TopicArtifactError("all topic centroids must have the same dimension")
    for node in nodes:
        centroid = np.asarray(node.centroid, dtype=float)
        if not np.isfinite(centroid).all() or not math.isclose(
            float(np.linalg.norm(centroid)), 1.0, rel_tol=1e-5, abs_tol=1e-5
        ):
            raise TopicArtifactError("topic centroids must be finite l2 unit vectors")
        if node.parent_id is not None and node.parent_id not in by_id:
            raise TopicArtifactError(f"unknown parent {node.parent_id!r}")
        for child in node.children:
            if child not in by_id or by_id[child].parent_id != node.node_id:
                raise TopicArtifactError(f"inconsistent child link {node.node_id!r}->{child!r}")
            if by_id[child].depth != node.depth + 1:
                raise TopicArtifactError("topic node depths are inconsistent")
    seen: set[str] = set()
    stack = [root_id]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            raise TopicArtifactError("topic hierarchy contains a cycle")
        seen.add(node_id)
        stack.extend(by_id[node_id].children)
    if seen != set(by_id):
        raise TopicArtifactError("topic hierarchy contains unreachable nodes")


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("embeddings must be a finite two-dimensional matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("topic embeddings cannot be zero vectors")
    return values / norms


def _normalize_vector(value: np.ndarray) -> np.ndarray:
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("embedding must be a finite vector")
    norm = float(np.linalg.norm(value))
    if norm == 0:
        raise ValueError("embedding cannot be a zero vector")
    return value / norm


def _normalized_mean(values: np.ndarray) -> np.ndarray:
    return _normalize_vector(np.asarray(values, dtype=float).mean(axis=0))


def _cosine(left: np.ndarray, right: Sequence[float]) -> float:
    right_array = _normalize_vector(np.asarray(right, dtype=float))
    return float(np.clip(np.dot(left, right_array), -1.0, 1.0))


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    shifted = values / temperature
    shifted -= shifted.max()
    exponential = np.exp(shifted)
    return exponential / exponential.sum()


def _minmax_nonnegative(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("chunk scores must be finite")
    if not len(array):
        return []
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum <= 0:
        return [0.0] * len(array)
    if math.isclose(maximum, minimum):
        return [1.0 if item > 0 else 0.0 for item in array]
    return [float((item - minimum) / (maximum - minimum)) for item in array]


def _term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


def _term_count(text: str, term: str) -> int:
    return len(_term_pattern(term).findall(text))


def _chunks_contain_any(chunks: Sequence[TextChunk], terms: Iterable[str]) -> bool:
    return any(_term_count(chunk.text, term) for term in terms for chunk in chunks)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _finite(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TopicArtifactError(f"{location} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise TopicArtifactError(f"{location} must be finite")
    return result


def _probability(value: Any, location: str) -> float:
    result = _finite(value, location)
    if not 0 <= result <= 1:
        raise TopicArtifactError(f"{location} must be between zero and one")
    return result


def _integer(value: Any, location: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TopicArtifactError(f"{location} must be an integer >= {minimum}")
    return value


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TopicArtifactError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise TopicArtifactError(f"{location} must be an array")
    return value
