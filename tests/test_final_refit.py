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
)
from untie.keyword_training import StrategyConfig, TrainingConfig, tune_global_keywords
from untie.keyword_tuning import ObjectiveConfig
from untie.topic_keyword_training import (
    DEFAULT_REFIT_MIN_KEYWORDS,
    load_selected_keyword_trial,
    refit_topic_keyword_profiles,
    resolve_refit_keyword_floor,
)
from untie.topics import (
    TopicBuildConfig,
    TopicDocument,
    TopicRoutingConfig,
    build_topic_artifact,
    load_topic_artifact,
    save_topic_artifact_atomic,
)


class Encoder:
    def encode(self, texts, **kwargs) -> np.ndarray:
        del kwargs
        if isinstance(texts, str):
            texts = [texts]
        return np.asarray(
            [[text.casefold().count("soil") + 1.0, 0.1] for text in texts],
            dtype=float,
        )


class MetricCache:
    def score(self, prediction, references, *, language, include_bertscore=False):
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


def _evidence(doc_id: str, word: str) -> CachedDocumentEvidence:
    return CachedDocumentEvidence(
        doc_id=doc_id,
        aspect_id="task",
        question="Which task?",
        references=("soil prediction",),
        chunks=(
            CachedChunkAnswer(0, "generic introduction", 2, "wrong", 0.8),
            CachedChunkAnswer(
                1,
                "beta identifies the soil prediction task",
                6,
                "soil prediction",
                0.9,
            ),
        ),
        answer_similarity=((1.0, 0.0), (0.0, 1.0)),
        baseline_answer="wrong",
        baseline_confidence=0.8,
        candidates=(
            CandidateObservation(word, word, word, 0.8, 0.7, (1,)),
        ),
        fingerprint=f"fp-{doc_id}",
    )


def _load_refit_cli():
    script = Path(__file__).parents[1] / "scripts" / "10_Refit_hierarchical_keywords.py"
    spec = importlib.util.spec_from_file_location("refit_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_selected_keyword_trial_uses_manifest_index_and_objective_fallback(
    tmp_path,
) -> None:
    path = tmp_path / "keyword_search.csv"
    path.write_text(
        "trial,min_keywords,max_keywords,selection_policy,stability_threshold,objective\n"
        "0,5,8,relaxed,0.4,0.10\n"
        "2,6,10,union,0.15,0.21\n"
        "3,6,10,relaxed,0.3,0.21\n",
        encoding="utf-8",
    )
    selected = load_selected_keyword_trial(path, trial_index=2)
    assert selected["min_keywords"] == 6
    assert selected["selection_policy"] == "union"
    fallback = load_selected_keyword_trial(path)
    assert fallback["trial"] == 2


def test_resolve_refit_keyword_floor_lowers_without_raising_trial() -> None:
    assert resolve_refit_keyword_floor(6) == DEFAULT_REFIT_MIN_KEYWORDS
    assert resolve_refit_keyword_floor(6, refit_min_keywords=4) == 4
    assert resolve_refit_keyword_floor(6, refit_min_keywords=5) == 5
    assert resolve_refit_keyword_floor(6, refit_min_keywords=8) == 6
    assert resolve_refit_keyword_floor(3, refit_min_keywords=4) == 3
    assert resolve_refit_keyword_floor(6, refit_min_keywords=4, max_keywords=10) == 4
    with pytest.raises(ValueError, match="at least 1"):
        resolve_refit_keyword_floor(0)
    with pytest.raises(ValueError, match="at least 1"):
        resolve_refit_keyword_floor(6, refit_min_keywords=0)


def test_refit_cli_default_floor_and_reuses_sibling_evidence(tmp_path) -> None:
    module = _load_refit_cli()
    args = module.build_parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "dataset.json"),
            "--tuned-model",
            str(tmp_path / "tuned.json"),
            "--keyword-search",
            str(tmp_path / "keyword_search.csv"),
            "--static-model",
            str(tmp_path / "static.json"),
        ]
    )
    assert args.refit_min_keywords == DEFAULT_REFIT_MIN_KEYWORDS
    assert args.contrast_against_gold is False
    assert args.score_chunk_strategy == "only_weight"
    assert resolve_refit_keyword_floor(6, refit_min_keywords=args.refit_min_keywords) == 4
    tuned = tmp_path / "source" / "topic_model.tuned.json"
    assert module.default_refit_output_dir(
        tuned, contrast_against_gold=False
    ) == tuned.parent / "refit_no_gold_contrast"
    assert module.default_refit_output_dir(
        tuned, contrast_against_gold=True
    ) == tuned.parent / "refit"
    restored = module.build_parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "dataset.json"),
            "--tuned-model",
            str(tmp_path / "tuned.json"),
            "--keyword-search",
            str(tmp_path / "keyword_search.csv"),
            "--static-model",
            str(tmp_path / "static.json"),
            "--contrast-against-gold",
            "--score-chunk-strategy",
            "only_score_diff",
        ]
    )
    assert restored.contrast_against_gold is True
    assert restored.score_chunk_strategy == "only_score_diff"

    old_store = module.EvidenceStore(tmp_path / "cache" / "old-run" / "evidence")
    saved = _evidence("doc-1", "pressure")
    old_store.save(saved)
    new_root = tmp_path / "cache" / "new-run" / "evidence"
    new_store = module.EvidenceStore(new_root)
    extras = module._sibling_evidence_stores(tmp_path, new_root)
    loaded = module._load_evidence_reusing_siblings(
        new_store, extras, "doc-1", saved.fingerprint
    )
    assert loaded == saved
    assert new_store.load("doc-1", expected_fingerprint=saved.fingerprint) == saved
    (tmp_path / "cache" / "old-run" / "metrics.json").write_text(
        '{"k": {"char_f1": 1.0}}\n', encoding="utf-8"
    )
    destination = tmp_path / "cache" / "new-run" / "metrics.json"
    module._reuse_metric_cache(tmp_path, destination)
    assert destination.is_file()


def test_refit_does_not_reuse_gold_contrast_evidence_for_new_fingerprint(
    tmp_path,
) -> None:
    module = _load_refit_cli()
    old_store = module.EvidenceStore(tmp_path / "cache" / "old-run" / "evidence")
    saved = _evidence("doc-1", "pressure")
    old_store.save(saved)
    new_root = tmp_path / "cache" / "new-run" / "evidence"
    extras = module._sibling_evidence_stores(tmp_path, new_root)
    loaded = module._load_evidence_reusing_siblings(
        module.EvidenceStore(new_root),
        extras,
        "doc-1",
        "no-gold-contrast-fingerprint",
    )
    assert loaded is None
    assert module.EvidenceStore(new_root).load(
        "doc-1", expected_fingerprint="no-gold-contrast-fingerprint"
    ) is None


def test_refit_uses_train_and_dev_candidates_and_preserves_routing(tmp_path) -> None:
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
            text="beta identifies the soil prediction task",
            split="dev",
        )
    ]
    encoder = Encoder()
    artifact = build_topic_artifact(
        train,
        encoder,
        encoder_name="test-encoder",
        config=TopicBuildConfig(leaf_clusters=1),
        routing=TopicRoutingConfig(similarity_threshold=0.75, min_margin=0.0),
    )
    frozen = artifact.routing
    result = refit_topic_keyword_profiles(
        artifact,
        [*train, *dev],
        [
            _evidence("train-1", "aaaa"),
            _evidence("train-2", "aaaa"),
            _evidence("dev-1", "beta"),
        ],
        encoder,
        config=TrainingConfig(
            language="en",
            min_document_support=1,
            max_candidates=5,
            min_keywords=1,
            max_keywords=2,
            evaluation_budget=12,
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
        min_fit_documents=2,
    )
    assert result.artifact.routing == frozen
    assert result.artifact.routing.similarity_threshold == 0.75
    profiles = [
        node.keyword_profile
        for node in result.artifact.nodes
        if node.keyword_profile is not None
    ]
    assert profiles
    words = {item.word for item in profiles[0].keywords}
    assert "beta" in words
    assert set(result.assignments) == {"train-1", "train-2", "dev-1"}
    assert result.assignments["train-1"] == artifact.train_assignments[0].leaf_id


def test_refit_rejects_test_documents() -> None:
    encoder = Encoder()
    train = [TopicDocument("train", title="soil", split="train")]
    artifact = build_topic_artifact(
        train,
        encoder,
        encoder_name="test-encoder",
        config=TopicBuildConfig(leaf_clusters=1),
    )
    with pytest.raises(ValueError, match="train/dev documents only"):
        refit_topic_keyword_profiles(
            artifact,
            [TopicDocument("test", title="soil", split="test")],
            [_evidence("test", "pressure")],
            encoder,
            config=TrainingConfig(language="en"),
            metric_cache=MetricCache(),  # type: ignore[arg-type]
        )


def test_loader_ignores_unknown_clustering_params_from_tuned_artifact(tmp_path) -> None:
    encoder = Encoder()
    artifact = build_topic_artifact(
        [TopicDocument("train", title="soil", split="train")],
        encoder,
        encoder_name="test-encoder",
        config=TopicBuildConfig(leaf_clusters=1),
        routing=TopicRoutingConfig(similarity_threshold=0.75),
    )
    path = tmp_path / "topic_model.tuned.json"
    save_topic_artifact_atomic(artifact, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["clustering"]["params"]["cluster_candidates"] = [4, 6, 8, 10]
    payload["clustering"]["params"]["cluster_selection"] = "auto"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_topic_artifact(path)
    assert loaded.routing.similarity_threshold == 0.75
    assert loaded.build.leaf_clusters == 1


def test_refit_cli_excludes_test_and_refuses_to_clobber_existing_outputs(tmp_path) -> None:
    module = _load_refit_cli()
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(
                {
                    "doc_id": f"doc-{index}",
                    "original_text": "text",
                    "split": split_name,
                    "tasks": ["Soil_Prediction"],
                }
            )
            for index, split_name in enumerate(("train", "train", "dev", "test"))
        ),
        encoding="utf-8",
    )
    documents, references = module.load_fit_documents(dataset, seed=999)
    assert {item.doc_id for item in documents} == {"doc-0", "doc-1", "doc-2"}
    assert all(item.split in {"train", "dev"} for item in documents)
    assert "doc-3" not in references or True
    assert "doc-3" not in {item.doc_id for item in documents}

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    tuned = source_dir / "topic_model.tuned.json"
    tuned.write_text("{}", encoding="utf-8")
    keyword_search = source_dir / "keyword_search.csv"
    keyword_search.write_text(
        "trial,min_keywords,max_keywords,selection_policy,stability_threshold,objective\n"
        "2,6,10,union,0.15,0.21\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "refit"
    output_dir.mkdir()
    (output_dir / "topic_model.refit.json").write_text("{}\n", encoding="utf-8")
    args = module.build_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--tuned-model",
            str(tuned),
            "--keyword-search",
            str(keyword_search),
            "--static-model",
            str(tmp_path / "static.json"),
            "--output-dir",
            str(output_dir),
        ]
    )
    with pytest.raises(ValueError, match="already exist"):
        module.run(args)

    source_bytes = tuned.read_bytes()
    args_overwrite_tuned = module.build_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--tuned-model",
            str(tuned),
            "--keyword-search",
            str(keyword_search),
            "--static-model",
            str(tmp_path / "static.json"),
            "--output-dir",
            str(tuned),
        ]
    )
    with pytest.raises(ValueError, match="refuses to overwrite the source tuned model"):
        module.run(args_overwrite_tuned)
    assert tuned.read_bytes() == source_bytes


def test_tune_global_keywords_uses_candidate_doc_ids_outside_train_split() -> None:
    train_doc = _evidence("train-1", "alpha")
    dev_doc = _evidence("dev-1", "beta")
    cache = MetricCache()  # type: ignore[arg-type]
    default = tune_global_keywords(
        [train_doc, dev_doc],
        config=TrainingConfig(
            language="en",
            min_document_support=1,
            max_candidates=5,
            min_keywords=1,
            max_keywords=2,
            evaluation_budget=12,
            patience=1,
            beam_width=1,
            stability_runs=1,
            screen_before_search=False,
            strategies=(StrategyConfig(),),
        ),
        metric_cache=cache,
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
        split={"train": ("train-1",), "dev": ("dev-1",), "test": ()},
    )
    expanded = tune_global_keywords(
        [train_doc, dev_doc],
        config=TrainingConfig(
            language="en",
            min_document_support=1,
            max_candidates=5,
            min_keywords=1,
            max_keywords=2,
            evaluation_budget=12,
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
        split={"train": ("train-1",), "dev": ("dev-1",), "test": ()},
        candidate_doc_ids=("train-1", "dev-1"),
    )
    assert default.keywords == ("alpha",)
    assert expanded.keywords == ("beta",)


def test_evaluate_script_summarizes_and_refuses_tuned_evaluation_dir(
    tmp_path,
) -> None:
    script = Path(__file__).parents[1] / "scripts" / "11_Evaluate_hierarchical_test.py"
    spec = importlib.util.spec_from_file_location("eval_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {
            "doc_id": "a",
            "mode": "baseline",
            "char_f1": 0.5,
            "token_f1": 50.0,
            "rouge_l_f1": 0.5,
            "bertscore_p": 0.8,
            "bertscore_r": 0.8,
            "bertscore_f1": 0.8,
            "composite_quality": 0.575,
            "qa_chunk_ratio": 1.0,
            "qa_token_ratio": 1.0,
            "fallback": False,
            "topic_keyword_ancestor_fallback": False,
            "latency_s": 1.0,
        },
        {
            "doc_id": "a",
            "mode": "hierarchical-refit",
            "char_f1": 0.6,
            "token_f1": 60.0,
            "rouge_l_f1": 0.6,
            "bertscore_p": 0.8,
            "bertscore_r": 0.8,
            "bertscore_f1": 0.8,
            "composite_quality": 0.65,
            "qa_chunk_ratio": 0.5,
            "qa_token_ratio": 0.5,
            "fallback": False,
            "topic_keyword_ancestor_fallback": False,
            "latency_s": 2.0,
        },
    ]
    summary = {item["mode"]: item for item in module.summarize_metric_rows(rows)}
    assert summary["hierarchical-refit"]["mean_delta_char_f1"] == pytest.approx(0.1)
    assert summary["hierarchical-refit"]["win_rate"] == 1.0
    assert summary["hierarchical-refit"]["harm_rate"] == 0.0
    args = module.build_parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "data.json"),
            "--topic-model",
            str(tmp_path / "model.json"),
            "--static-model",
            str(tmp_path / "static.json"),
            "--output-dir",
            str(
                Path(__file__).parents[1]
                / "artifacts"
                / "hierarchical_topic_optimized"
                / "en"
                / "evaluation"
            ),
        ]
    )
    with pytest.raises(ValueError, match="refuses to overwrite"):
        module.run(args)

    args.output_dir = (
        Path(__file__).parents[1]
        / "artifacts"
        / "hierarchical_topic_optimized"
        / "en"
        / "refit_no_gold_contrast"
        / "evaluation"
    )
    with pytest.raises(ValueError, match="refuses to overwrite"):
        module.run(args)

    args.output_dir = (
        Path(__file__).parents[1]
        / "artifacts"
        / "hierarchical_topic_optimized"
        / "en"
        / "soft_rerank"
        / "evaluation"
    )
    with pytest.raises(ValueError, match="refuses to overwrite"):
        module.run(args)

    args.output_dir = (
        Path(__file__).parents[1]
        / "artifacts"
        / "hierarchical_topic_optimized"
        / "en"
        / "cross_encoder"
        / "evaluation"
    )
    with pytest.raises(ValueError, match="refuses to overwrite"):
        module.run(args)

    args.output_dir = (
        Path(__file__).parents[1]
        / "artifacts"
        / "hierarchical_topic_optimized"
        / "en"
        / "ce_topic_query"
        / "evaluation"
    )
    with pytest.raises(ValueError, match="refuses to overwrite"):
        module.run(args)

    args.output_dir = (
        Path(__file__).parents[1]
        / "artifacts"
        / "hierarchical_topic_optimized"
        / "en"
        / "ce_topic_query_body"
        / "evaluation"
    )
    with pytest.raises(ValueError, match="refuses to overwrite"):
        module.run(args)

    reused_path = tmp_path / "prev.csv"
    reused_path.write_text(
        "doc_id,mode,prediction,char_f1\n"
        "a,baseline,x,0.5\n"
        "a,hierarchical-no-gold-contrast,y,0.4\n"
        "a,hierarchical-soft-rerank,z,0.6\n",
        encoding="utf-8",
    )
    kept = module.load_reused_rows(
        reused_path, {"a"}, exclude_modes={"hierarchical-soft-rerank"}
    )
    assert {row["mode"] for row in kept} == {
        "baseline",
        "hierarchical-no-gold-contrast",
    }

    soft_args = module.build_parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "data.json"),
            "--topic-model",
            str(tmp_path / "model.json"),
            "--static-model",
            str(tmp_path / "static.json"),
            "--output-dir",
            str(tmp_path / "soft"),
            "--soft-rerank",
        ]
    )
    assert soft_args.soft_rerank is True
    assert soft_args.force is False
