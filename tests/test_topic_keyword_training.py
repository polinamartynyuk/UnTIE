from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from untie.keyword_evidence import (
    CachedChunkAnswer,
    CachedDocumentEvidence,
    CandidateObservation,
    EvidenceStore,
)
from untie.keyword_training import StrategyConfig, TrainingConfig
from untie.keyword_tuning import ObjectiveConfig
from untie.topic_keyword_training import tune_topic_keyword_profiles
from untie.topics import (
    TopicBuildConfig,
    TopicDocument,
    TopicKeywordProfile,
    build_topic_artifact,
    save_topic_artifact_atomic,
)
from untie.ranking import WeightedKeyword


class Encoder:
    def encode(self, texts, **kwargs) -> np.ndarray:
        del kwargs
        if isinstance(texts, str):
            texts = [texts]
        return np.asarray(
            [
                [text.casefold().count("soil") + 1.0, 0.1]
                for text in texts
            ],
            dtype=float,
        )


class MetricCache:
    def score(
        self,
        prediction,
        references,
        *,
        language,
        include_bertscore=False,
    ):
        del language
        exact = float(prediction in references)
        metrics = {
            "char_f1": exact,
            "token_f1": exact * 100,
            "rouge_l_f1": exact,
        }
        if include_bertscore:
            metrics["bertscore_f1"] = exact
        return metrics

    def save(self) -> None:
        pass


def _evidence(doc_id: str, *, train: bool) -> CachedDocumentEvidence:
    return CachedDocumentEvidence(
        doc_id=doc_id,
        aspect_id="task",
        question="Which task?",
        references=("soil prediction",),
        chunks=(
            CachedChunkAnswer(0, "generic introduction", 2, "wrong", 0.8),
            CachedChunkAnswer(
                1,
                "pressure identifies the soil prediction task",
                6,
                "soil prediction",
                0.9,
            ),
        ),
        answer_similarity=((1.0, 0.0), (0.0, 1.0)),
        baseline_answer="wrong",
        baseline_confidence=0.8,
        candidates=(
            (
                CandidateObservation(
                    "pressure",
                    "pressure",
                    "pressur",
                    0.8,
                    0.7,
                    (1,),
                ),
            )
            if train
            else ()
        ),
        fingerprint=f"fp-{doc_id}",
    )


def test_topic_keyword_tuning_attaches_train_only_leaf_profile(tmp_path) -> None:
    train = [
        TopicDocument(
            "train-1",
            title="Frozen soil",
            text="pressure identifies the soil prediction task",
            split="train",
        ),
        TopicDocument(
            "train-2",
            title="Soil pressure",
            text="pressure identifies the soil prediction task",
            split="train",
        ),
    ]
    dev = [
        TopicDocument(
            "dev-1",
            title="Soil evaluation",
            text="pressure identifies the soil prediction task",
            split="dev",
        )
    ]
    encoder = Encoder()
    artifact = build_topic_artifact(
        train,
        encoder,
        encoder_name="test-encoder",
        config=TopicBuildConfig(leaf_clusters=1),
    )
    result = tune_topic_keyword_profiles(
        artifact,
        train,
        dev,
        [_evidence("train-1", train=True), _evidence("train-2", train=True), _evidence("dev-1", train=False)],
        encoder,
        config=TrainingConfig(
            language="en",
            min_document_support=2,
            max_candidates=5,
            max_keywords=1,
            evaluation_budget=10,
            patience=1,
            beam_width=1,
            stability_runs=1,
            screen_before_search=False,
            strategies=(StrategyConfig(),),
        ),
        metric_cache=MetricCache(),  # type: ignore[arg-type]
        objective_config=ObjectiveConfig(
            downside_penalty=0,
            harm_penalty=0,
            fallback_penalty=0,
            size_penalty=0,
            bootstrap_samples=10,
            min_activation_rate=0,
            activation_weight=0,
            win_rate_weight=0,
            inactive_fallback_penalty=0,
        ),
        checkpoint_dir=tmp_path / "checkpoints",
    )
    profiles = [
        node.keyword_profile
        for node in result.artifact.nodes
        if node.keyword_profile is not None
    ]
    assert len(profiles) == 1
    assert [item.word for item in profiles[0].keywords] == ["pressure"]
    assert profiles[0].train_document_count == 2
    assert profiles[0].dev_document_count == 1
    assert set(result.assignments) == {"train-1", "train-2", "dev-1"}


def test_topic_keyword_tuning_rejects_test_documents() -> None:
    encoder = Encoder()
    train = [TopicDocument("train", title="soil", split="train")]
    artifact = build_topic_artifact(
        train,
        encoder,
        encoder_name="test-encoder",
        config=TopicBuildConfig(leaf_clusters=1),
    )
    with pytest.raises(ValueError, match="dev documents only"):
        tune_topic_keyword_profiles(
            artifact,
            train,
            [TopicDocument("test", title="soil", split="test")],
            [_evidence("train", train=True), _evidence("test", train=False)],
            encoder,
            config=TrainingConfig(language="en"),
            metric_cache=MetricCache(),  # type: ignore[arg-type]
        )


def test_retuning_clears_stale_profile_when_leaf_is_skipped() -> None:
    encoder = Encoder()
    train = [TopicDocument("train", title="soil", split="train")]
    dev = [TopicDocument("dev", title="soil", split="dev")]
    artifact = build_topic_artifact(
        train,
        encoder,
        encoder_name="test-encoder",
        config=TopicBuildConfig(leaf_clusters=1),
    )
    stale = TopicKeywordProfile(
        keywords=(WeightedKeyword("stale", "stale", "stale"),),
        train_document_count=2,
        dev_document_count=1,
    )
    artifact = replace(
        artifact,
        nodes=tuple(
            replace(node, keyword_profile=stale) for node in artifact.nodes
        ),
    )
    result = tune_topic_keyword_profiles(
        artifact,
        train,
        dev,
        [_evidence("train", train=True), _evidence("dev", train=False)],
        encoder,
        config=TrainingConfig(language="en"),
        metric_cache=MetricCache(),  # type: ignore[arg-type]
        min_train_documents=2,
    )
    assert all(node.keyword_profile is None for node in result.artifact.nodes)


def test_topic_keyword_cli_loads_jsonl_without_test_leakage(tmp_path) -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "07_Tune_topic_keywords.py"
    )
    spec = importlib.util.spec_from_file_location("topic_keyword_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(
                {
                    "doc_id": f"doc-{index}",
                    "original_text": f"document {index}",
                }
            )
            for index in range(10)
        ),
        encoding="utf-8",
    )
    train, dev = module.load_topic_documents(dataset, seed=42)
    assert len(train) == 6
    assert len(dev) == 2
    assert all(item.split == "train" for item in train)
    assert all(item.split == "dev" for item in dev)
    assert {item.doc_id for item in train}.isdisjoint(
        item.doc_id for item in dev
    )

    explicit = tmp_path / "explicit.jsonl"
    explicit.write_text(
        "\n".join(
            json.dumps(
                {
                    "doc_id": f"explicit-{index}",
                    "original_text": "text",
                    "split": split_name,
                }
            )
            for index, split_name in enumerate(
                ("train", "train", "dev", "test")
            )
        ),
        encoding="utf-8",
    )
    train, dev = module.load_topic_documents(explicit, seed=999)
    assert [item.doc_id for item in train] == ["explicit-0", "explicit-1"]
    assert [item.doc_id for item in dev] == ["explicit-2"]


def test_topic_keyword_cli_rejects_stale_evidence(tmp_path) -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "07_Tune_topic_keywords.py"
    )
    spec = importlib.util.spec_from_file_location("topic_keyword_cli_stale", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(
                {
                    "doc_id": f"doc-{index}",
                    "original_text": "soil pressure",
                    "tasks": ["Soil_Prediction"],
                }
            )
            for index in range(10)
        ),
        encoding="utf-8",
    )
    artifact = build_topic_artifact(
        [TopicDocument("build", title="soil", split="train")],
        Encoder(),
        encoder_name="test-encoder",
        config=TopicBuildConfig(leaf_clusters=1),
    )
    artifact_path = tmp_path / "topics.json"
    save_topic_artifact_atomic(artifact, artifact_path)
    evidence_dir = tmp_path / "evidence"
    store = EvidenceStore(evidence_dir)
    train, dev = module.load_topic_documents(dataset, seed=42)
    for document in (*train, *dev):
        store.save(_evidence(document.doc_id, train=document.split == "train"))

    args = module.build_parser().parse_args(
        [
            "--topic-model",
            str(artifact_path),
            "--dataset",
            str(dataset),
            "--evidence-dir",
            str(evidence_dir),
            "--output",
            str(tmp_path / "output.json"),
        ]
    )
    with pytest.raises(ValueError, match="missing or incompatible"):
        module.run(args)
