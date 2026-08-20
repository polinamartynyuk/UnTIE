from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from untie.config import ModelProfile, PipelineConfig
from untie.domain import Sentence, TextChunk
from untie.pipelines import DocumentProcessor, HierarchicalTopicRerankingPipeline
from untie.ranking import WeightedKeyword
from untie.topics import (
    TOPIC_SCHEMA_VERSION,
    TopicArtifact,
    TopicArtifactError,
    TopicBuildConfig,
    TopicDocument,
    TopicFallbackReason,
    TopicKeywordProfile,
    TopicMixingConfig,
    TopicRoutingConfig,
    build_topic_artifact,
    compute_qa_ratios,
    compute_retrieval_metrics,
    document_topic_representation,
    load_topic_artifact,
    resolve_topic_match_fallback,
    resolve_topic_keyword_profile,
    route_document,
    route_document_to_nearest_leaf,
    route_topic_embedding,
    save_topic_artifact_atomic,
    save_topic_diagnostics,
    score_topic_aware_chunks,
    tune_topic_configuration,
)


class TopicEncoder:
    def encode(self, texts, **kwargs) -> np.ndarray:
        del kwargs
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            lowered = text.casefold()
            vector = np.asarray(
                [
                    sum(lowered.count(term) for term in ("soil", "frost", "pressure")),
                    sum(lowered.count(term) for term in ("network", "language", "token")),
                    0.1,
                ],
                dtype=float,
            )
            vectors.append(vector)
        return np.asarray(vectors)


class FakeTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.replace(".", "").split()

    def encode(self, text: str, **kwargs) -> list[int]:
        del kwargs
        return [101, *range(len(self.tokenize(text))), 102]

    def convert_tokens_to_string(self, tokens) -> str:
        return " ".join(tokens)


class FakeAnswerer:
    def __call__(self, *, question: str, context: str) -> dict:
        del question
        answer = "frost pressure" if "frost pressure" in context else context.split(".")[0]
        start = max(0, context.find(answer))
        return {
            "answer": answer,
            "score": 0.9,
            "start": start,
            "end": start + len(answer),
        }


def _documents() -> list[TopicDocument]:
    return [
        TopicDocument(
            "soil-1",
            title="Frozen soil",
            abstract="Frost pressure measurement in soil",
            text="Frost pressure measurement controls frozen soil deformation.",
            split="train",
        ),
        TopicDocument(
            "soil-2",
            title="Soil mechanics",
            abstract="Soil pressure and frost heave measurement",
            text="Soil pressure measurement predicts frost heave.",
            split="train",
        ),
        TopicDocument(
            "nlp-1",
            title="Language network",
            abstract="Token network measurement for language",
            text="Language token measurement trains a neural network.",
            split="train",
        ),
        TopicDocument(
            "nlp-2",
            title="Token models",
            abstract="Network language token measurement",
            text="A network performs language token measurement.",
            split="train",
        ),
    ]


def _artifact(**mixing_overrides) -> TopicArtifact:
    mixing = TopicMixingConfig(**mixing_overrides)
    return build_topic_artifact(
        _documents(),
        TopicEncoder(),
        encoder_name=ModelProfile.english().sentence_model,
        config=TopicBuildConfig(
            leaf_clusters=2,
            min_topic_document_support=0.5,
            topic_term_top_k=20,
            seed=7,
        ),
        routing=TopicRoutingConfig(
            top_k=2,
            beam_width=2,
            temperature=0.5,
            similarity_threshold=-1,
        ),
        mixing=mixing,
    )


def _chunk(text: str, number: int = 0) -> TextChunk:
    sentence = Sentence(text, number, tuple(text.split()))
    return TextChunk((sentence,), text, sentence.token_count)


def test_document_topic_representation_prefers_metadata_and_falls_back() -> None:
    structured = TopicDocument(
        "1",
        text="Ignored body",
        title="Title",
        abstract="Abstract",
        headings=("Methods", "Results"),
    )
    assert document_topic_representation(structured) == (
        "Title\nAbstract\nMethods\nResults"
    )
    plain = TopicDocument("2", text="First paragraph.\n\nSecond paragraph.\n\nThird.")
    assert document_topic_representation(plain) == (
        "First paragraph.\nSecond paragraph.\nThird."
    )
    assert document_topic_representation(TopicDocument("3")) == ""


def test_builder_rejects_non_train_and_prevents_vocabulary_leakage() -> None:
    with pytest.raises(ValueError, match="train documents only"):
        build_topic_artifact(
            [TopicDocument("dev", text="devleak", split="dev")],
            TopicEncoder(),
            encoder_name="fake",
        )
    artifact = _artifact()
    all_terms = {
        term.term for node in artifact.nodes for term in node.representative_terms
    }
    assert "devleak" not in all_terms


def test_builder_is_deterministic_and_builds_valid_hierarchy() -> None:
    first = _artifact().to_dict()
    second = build_topic_artifact(
        list(reversed(_documents())),
        TopicEncoder(),
        encoder_name=ModelProfile.english().sentence_model,
        config=TopicBuildConfig(
            leaf_clusters=2,
            min_topic_document_support=0.5,
            topic_term_top_k=20,
            seed=7,
        ),
        routing=TopicRoutingConfig(
            top_k=2,
            beam_width=2,
            temperature=0.5,
            similarity_threshold=-1,
        ),
    ).to_dict()
    assert first == second
    assert first["clustering"]["seed"] == 7
    nodes = {item["id"]: item for item in first["nodes"]}
    root = nodes[first["root_id"]]
    assert root["parent_id"] is None
    assert root["document_count"] == 4
    assert root["children"]
    assignments = first["train_assignments"]
    assert {item["doc_id"] for item in assignments} == {
        item.doc_id for item in _documents()
    }
    assignment_counts = {
        leaf_id: sum(item["leaf_id"] == leaf_id for item in assignments)
        for leaf_id, node in nodes.items()
        if not node["children"]
    }
    assert all(
        assignment_counts[node_id] == nodes[node_id]["document_count"]
        for node_id in assignment_counts
    )
    assert all(np.isfinite(item["centroid"]).all() for item in nodes.values())
    duplicate = replace(_documents()[1], doc_id=_documents()[0].doc_id)
    with pytest.raises(ValueError, match="unique"):
        build_topic_artifact(
            [_documents()[0], duplicate],
            TopicEncoder(),
            encoder_name="fake",
        )


def test_artifact_round_trip_and_schema_validation(tmp_path) -> None:
    path = tmp_path / "topics.json"
    artifact = _artifact()
    save_topic_artifact_atomic(artifact, path)
    loaded = load_topic_artifact(path)
    assert loaded.to_dict() == artifact.to_dict()
    legacy = artifact.to_dict()
    legacy.pop("train_assignments")
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert load_topic_artifact(path).train_assignments == ()
    save_topic_artifact_atomic(artifact, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = TOPIC_SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TopicArtifactError, match="schema_version"):
        load_topic_artifact(path)
    payload["schema_version"] = TOPIC_SCHEMA_VERSION
    payload["nodes"][0]["centroid"] = [0.0] * len(
        payload["nodes"][0]["centroid"]
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TopicArtifactError, match="unit vectors"):
        load_topic_artifact(path)
    malformed = artifact.to_dict()
    malformed["routing"]["unknown_option"] = True
    path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(TopicArtifactError, match="invalid topic artifact"):
        load_topic_artifact(path)


def test_node_keyword_profile_round_trip_and_resolution(tmp_path) -> None:
    artifact = _artifact()
    route = route_document_to_nearest_leaf(
        TopicDocument("new", title="Frozen soil pressure"),
        TopicEncoder(),
        artifact,
    )
    leaf_id = route.topics[0].node_id
    profile = TopicKeywordProfile(
        keywords=(
            WeightedKeyword("pressure", "pressure", "pressur", 0.8, 0.7),
        ),
        score_chunk_strategy="only_score_diff",
        train_document_count=2,
        dev_document_count=1,
        objective=0.25,
    )
    artifact = replace(
        artifact,
        nodes=tuple(
            replace(node, keyword_profile=profile)
            if node.node_id == leaf_id
            else node
            for node in artifact.nodes
        ),
    )
    path = tmp_path / "topic-keywords.json"
    save_topic_artifact_atomic(artifact, path)
    loaded = load_topic_artifact(path)
    resolved = resolve_topic_keyword_profile(route, loaded)
    assert resolved is not None
    assert resolved[0] == leaf_id
    assert resolved[1].keywords[0].word == "pressure"
    assert loaded.to_dict() == artifact.to_dict()


def test_soft_routing_supports_topics_mixed_documents_and_weights() -> None:
    artifact = _artifact()
    soil = route_document(
        TopicDocument("new", title="Frost soil pressure"),
        TopicEncoder(),
        artifact,
        expected_encoder_name=artifact.encoder_name,
    )
    nlp = route_document(
        TopicDocument("new", title="Language token network"),
        TopicEncoder(),
        artifact,
    )
    mixed = route_document(
        TopicDocument("new", title="Soil language network pressure"),
        TopicEncoder(),
        artifact,
    )
    assert soil.topics[0].node_id != nlp.topics[0].node_id
    assert len(mixed.topics) == 2
    assert sum(item.weight for item in mixed.topics) == pytest.approx(1.0)
    assert all(np.isfinite(item.similarity) for item in mixed.topics)
    with pytest.raises(TopicArtifactError, match="revision"):
        route_document(
            TopicDocument("new", title="Frost soil pressure"),
            TopicEncoder(),
            artifact,
            expected_encoder_revision="different-revision",
        )
    with pytest.raises(TopicArtifactError, match="dimension"):
        route_topic_embedding([1.0, 0.0], artifact)


def test_nearest_leaf_routing_respects_low_confidence_margin() -> None:
    artifact = _artifact()
    artifact = replace(
        artifact,
        routing=replace(artifact.routing, min_margin=2.0),
    )
    route = route_document_to_nearest_leaf(
        TopicDocument("ambiguous", title="soil language"),
        TopicEncoder(),
        artifact,
    )
    assert route.topics == ()
    assert route.fallback_reason is TopicFallbackReason.LOW_TOPIC_CONFIDENCE


def test_threshold_margin_and_parent_fallback_are_diagnostic() -> None:
    artifact = _artifact()
    strict = replace(
        artifact,
        routing=TopicRoutingConfig(
            top_k=2,
            beam_width=2,
            temperature=0.5,
            similarity_threshold=1.0,
        ),
    )
    route = route_document(
        TopicDocument("new", title="Frost soil"),
        TopicEncoder(),
        strict,
    )
    assert route.fallback_reason == TopicFallbackReason.LOW_TOPIC_CONFIDENCE
    assert route.topics[0].node_id == strict.root_id

    ambiguous = replace(
        artifact,
        routing=TopicRoutingConfig(
            top_k=2,
            beam_width=2,
            temperature=0.5,
            similarity_threshold=-1,
            min_margin=2.0,
        ),
    )
    margin_route = route_document(
        TopicDocument("new", title="Soil language network pressure"),
        TopicEncoder(),
        ambiguous,
    )
    assert margin_route.fallback_reason == TopicFallbackReason.LOW_TOPIC_CONFIDENCE
    assert margin_route.topics[0].node_id == ambiguous.root_id

    regular_route = route_document(
        TopicDocument("new", title="Frost soil pressure"),
        TopicEncoder(),
        artifact,
    )
    resolved = resolve_topic_match_fallback(
        [_chunk("A shared measurement is reported.")],
        regular_route,
        artifact,
    )
    assert resolved.fallback_reason == TopicFallbackReason.NO_TOPIC_MATCHES
    assert any(item.node_id == artifact.root_id for item in resolved.topics)
    root = artifact.node_map[artifact.root_id]
    expected_similarity = float(
        np.dot(
            np.asarray(regular_route.embedding),
            np.asarray(root.centroid),
        )
    )
    root_route = next(
        item for item in resolved.topics if item.node_id == artifact.root_id
    )
    assert root_route.similarity == pytest.approx(expected_similarity)


def test_aspect_topic_scoring_normalizes_and_filters_unmatched_chunks() -> None:
    artifact = _artifact(
        theta_aspect=1.0,
        theta_topic=1.0,
        theta_interaction=1.0,
    )
    route = route_document(
        TopicDocument("new", title="Frozen soil pressure"),
        TopicEncoder(),
        artifact,
    )
    chunks = [
        _chunk("Frost pressure measurement describes soil.", 0),
        _chunk("Frost background is discussed.", 1),
        _chunk("Unrelated appendix.", 2),
    ]
    keywords = [
        WeightedKeyword("pressure", "pressure", "pressure", 0.8, 0.7)
    ]
    scored = score_topic_aware_chunks(chunks, keywords, route, artifact)
    assert len(scored) == 2
    assert scored[0].chunk == chunks[0]
    assert scored[0].score > scored[-1].score
    assert all(np.isfinite(item.score) for item in scored)
    assert scored[0].keyword_scores["aspect_normalized"] == pytest.approx(1.0)
    assert scored[0].keyword_scores["topic_normalized"] == pytest.approx(1.0)
    assert scored[0].keyword_scores["aspect_matches"] == 1
    assert scored[0].keyword_scores["topic_matches"] >= 1


def test_topic_filter_can_be_disabled_to_preserve_all_chunks() -> None:
    artifact = _artifact(filter_unmatched_chunks=False)
    route = route_document(
        TopicDocument("new", title="Language network"),
        TopicEncoder(),
        artifact,
    )
    chunks = [_chunk("Completely unrelated text.", 0), _chunk("Another appendix.", 1)]
    scored = score_topic_aware_chunks(chunks, [], route, artifact)
    assert [item.score for item in scored] == [0.0, 0.0]
    assert len(scored) == 2


def test_empty_zero_scoring_returns_no_chunks_when_filtering() -> None:
    artifact = _artifact()
    route = route_document(
        TopicDocument("new", title="Language network"),
        TopicEncoder(),
        artifact,
    )
    chunks = [_chunk("Completely unrelated text.", 0), _chunk("Another appendix.", 1)]
    scored = score_topic_aware_chunks(chunks, [], route, artifact)
    assert scored == []


def test_aspect_threshold_is_a_gate_not_a_subtraction() -> None:
    base = _artifact(
        aspect_threshold=0.0,
        theta_aspect=1.0,
        theta_topic=0.0,
        theta_interaction=0.0,
        weight_ratio=0.0,
    )
    gated = replace(base, mixing=replace(base.mixing, aspect_threshold=0.5))
    route = route_document(
        TopicDocument("new", title="Frozen soil pressure"),
        TopicEncoder(),
        base,
    )
    chunks = [_chunk("Pressure appears here."), _chunk("No signal.", 1)]
    keyword = WeightedKeyword("pressure", "pressure", "pressure", 0.0, 0.7)
    base_score = score_topic_aware_chunks(chunks, [keyword], route, base)[0]
    gated_score = score_topic_aware_chunks(chunks, [keyword], route, gated)[0]
    assert gated_score.keyword_scores["aspect_raw"] == pytest.approx(
        base_score.keyword_scores["aspect_raw"]
    )


def test_pipeline_routes_without_gold_and_reports_compute() -> None:
    artifact = _artifact()
    config = PipelineConfig(
        profile=ModelProfile.english(),
        chunk_max_tokens=8,
        overlap_tokens=2,
    )
    pipeline = HierarchicalTopicRerankingPipeline(
        DocumentProcessor(FakeTokenizer(), config),
        FakeAnswerer(),
        TopicEncoder(),
        config,
        artifact,
        [WeightedKeyword("pressure", "pressure", "pressure", 0.8, 0.7)],
    )
    result = pipeline.run(
        "Frozen soil has frost pressure. An unrelated appendix follows.",
        "What is measured?",
    )
    assert result.final_answer is not None
    assert result.metadata["topic_enabled"] is True
    assert result.metadata["topic_route"]
    assert result.metadata["topic_filter_enabled"] is True
    assert result.metadata["topic_filtered_chunk_count"] >= 1
    assert result.metadata["compute"]["qa_chunk_ratio"] <= 1.0
    assert "reference" not in json.dumps(result.metadata).casefold()


def test_pipeline_uses_nearest_cluster_keyword_profile() -> None:
    artifact = _artifact()
    route = route_document_to_nearest_leaf(
        TopicDocument("new", title="Frozen soil"),
        TopicEncoder(),
        artifact,
    )
    leaf_id = route.topics[0].node_id
    profile = TopicKeywordProfile(
        keywords=(
            WeightedKeyword("pressure", "pressure", "pressure", 0.8, 0.7),
        ),
        score_chunk_strategy="only_score_diff",
        choose_cluster_strategy="weighted_score",
        choose_answer_strategy="highest_similarity",
        train_document_count=2,
        dev_document_count=1,
    )
    artifact = replace(
        artifact,
        nodes=tuple(
            replace(node, keyword_profile=profile)
            if node.node_id == leaf_id
            else node
            for node in artifact.nodes
        ),
    )
    config = PipelineConfig(
        profile=ModelProfile.english(),
        chunk_max_tokens=6,
        overlap_tokens=0,
    )
    result = HierarchicalTopicRerankingPipeline(
        DocumentProcessor(FakeTokenizer(), config),
        FakeAnswerer(),
        TopicEncoder(),
        config,
        artifact,
        [],
    ).run(
        (
            "Generic introduction contains many unrelated background words. "
            "frost pressure is measured in soil."
        ),
        "What is measured?",
        document=TopicDocument("new", title="Frozen soil"),
    )
    assert result.metadata["topic_keyword_mode"] is True
    assert result.metadata["topic_keyword_node_id"] == leaf_id
    assert result.metadata["topic_keywords"] == ["pressure"]
    assert result.metadata["compute"]["qa_chunk_ratio"] < 1.0
    assert result.final_answer is not None
    assert result.final_answer.text == "frost pressure"


def test_pipeline_ood_falls_back_to_static_then_baseline() -> None:
    artifact = _artifact()
    config = PipelineConfig(
        profile=ModelProfile.english(),
        chunk_max_tokens=12,
        overlap_tokens=2,
    )
    with_static = HierarchicalTopicRerankingPipeline(
        DocumentProcessor(FakeTokenizer(), config),
        FakeAnswerer(),
        TopicEncoder(),
        config,
        artifact,
        [WeightedKeyword("appendix", "appendix", "appendix")],
    ).run("An appendix contains unrelated material.", "What?")
    assert with_static.metadata["fallback_stage"] == TopicFallbackReason.STATIC_KEYWORDS.value

    baseline = HierarchicalTopicRerankingPipeline(
        DocumentProcessor(FakeTokenizer(), config),
        FakeAnswerer(),
        TopicEncoder(),
        config,
        artifact,
        [],
    ).run("Unrelated astronomy material.", "What?")
    assert baseline.metadata["fallback_stage"] == TopicFallbackReason.BASELINE.value


def test_static_fallback_preserves_tuned_weight_ratio() -> None:
    artifact = _artifact()
    config = PipelineConfig(
        profile=ModelProfile.english(),
        chunk_max_tokens=12,
        overlap_tokens=2,
    )
    keyword = WeightedKeyword("appendix", "appendix", "appendix", 1.0, 0.0)

    def run(weight_ratio: float):
        return HierarchicalTopicRerankingPipeline(
            DocumentProcessor(FakeTokenizer(), config),
            FakeAnswerer(),
            TopicEncoder(),
            config,
            artifact,
            [keyword],
            static_weight_ratio=weight_ratio,
        ).run("An appendix contains unrelated material.", "What?")

    only_weight = run(1.0)
    only_difference = run(0.0)
    assert only_weight.metadata["fallback_stage"] == "STATIC_KEYWORDS"
    assert only_weight.questions[0].answers[0].metadata["chunk_score"] > 0
    assert only_difference.questions[0].answers[0].metadata["chunk_score"] == 0


def test_retrieval_and_compute_metrics() -> None:
    metrics = compute_retrieval_metrics([3, 1, 2, 0], [1, 2], k=2)
    assert metrics == {"evidence_recall_at_k": 0.5, "mrr": 0.5}
    chunks = [_chunk("one two", 0), _chunk("three", 1)]
    compute = compute_qa_ratios(chunks[:1], chunks)
    assert compute["qa_chunk_ratio"] == 0.5
    assert compute["qa_token_ratio"] == pytest.approx(2 / 3)


def test_dev_tuning_changes_only_routing_and_mixing() -> None:
    artifact = _artifact()
    preferred_routing = replace(artifact.routing, top_k=1)
    preferred_mixing = replace(artifact.mixing, theta_topic=2.0)
    result = tune_topic_configuration(
        artifact,
        [TopicDocument("dev-1", title="Frost soil", split="dev")],
        [
            (artifact.routing, artifact.mixing),
            (preferred_routing, preferred_mixing),
        ],
        lambda candidate, documents: {
            "objective": candidate.mixing.theta_topic,
            "routing_coverage": float(bool(documents)),
        },
    )
    assert result.artifact.routing.top_k == 1
    assert result.artifact.mixing.theta_topic == 2.0
    assert result.artifact.nodes == artifact.nodes
    with pytest.raises(ValueError, match="dev documents only"):
        tune_topic_configuration(
            artifact,
            [TopicDocument("test-1", text="test", split="test")],
            [(artifact.routing, artifact.mixing)],
            lambda candidate, documents: 0.0,
        )


def test_topic_diagnostics_are_written(tmp_path) -> None:
    paths = save_topic_diagnostics(_artifact(), tmp_path)
    assert set(paths) == {
        "nodes",
        "representative_terms",
        "train_assignments",
        "summary",
    }
    assert all(path.is_file() for path in paths.values())
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["train_document_count"] == 4
    assert summary["train_assignment_count"] == 4
    assert summary["schema_version"] == TOPIC_SCHEMA_VERSION


def test_offline_builder_parser_and_train_loader(tmp_path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "06_Build_topic_model.py"
    spec = importlib.util.spec_from_file_location("topic_builder_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(
        ["--train-dataset", "train.json", "--output", "topics.json"]
    )
    assert args.seed == 42
    assert args.max_qa_chunks == 0
    dataset = tmp_path / "train.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "doc_id": "1",
                    "title": "Soil",
                    "abstract": "Frost pressure",
                    "headings": ["Methods"],
                    "split": "train",
                }
            ]
        ),
        encoding="utf-8",
    )
    loaded = module.load_train_documents(dataset)
    assert loaded[0].headings == ("Methods",)
    assert loaded[0].split == "train"
    missing_split = tmp_path / "missing-split.json"
    missing_split.write_text(
        json.dumps([{"doc_id": "2", "text": "Soil"}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="split is required"):
        module.load_train_documents(missing_split)
    assumed = module.load_train_documents(missing_split, assume_train=True)
    assert assumed[0].split == "train"
