from __future__ import annotations

import json

import pytest

from untie.cli import build_parser, load_static_keywords, main


def test_static_mode_parser_accepts_tuned_model() -> None:
    args = build_parser().parse_args(
        [
            "document.txt",
            "--question",
            "Which task?",
            "--mode",
            "static-keywords",
            "--model-params",
            "model.json",
        ]
    )
    assert args.mode == "static-keywords"
    assert args.model_params.name == "model.json"


def test_static_mode_requires_model_params(tmp_path) -> None:
    document = tmp_path / "doc.txt"
    document.write_text("text", encoding="utf-8")
    with pytest.raises(SystemExit, match="--model-params"):
        main(
            [
                str(document),
                "--question",
                "Which task?",
                "--mode",
                "static-keywords",
            ]
        )


def test_hierarchical_mode_requires_only_topic_model(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "document.txt",
            "--question",
            "Which task?",
            "--mode",
            "hierarchical-static-keywords",
            "--topic-model",
            "topics.json",
        ]
    )
    assert args.topic_model.name == "topics.json"
    assert args.model_params is None

    document = tmp_path / "doc.txt"
    document.write_text("text", encoding="utf-8")
    with pytest.raises(SystemExit, match="--topic-model"):
        main(
            [
                str(document),
                "--question",
                "Which task?",
                "--mode",
                "hierarchical-static-keywords",
            ]
        )


def test_hierarchical_mode_rejects_bad_artifact_before_model_loading(
    tmp_path,
) -> None:
    document = tmp_path / "doc.txt"
    document.write_text("text", encoding="utf-8")
    artifact = tmp_path / "bad-topic.json"
    artifact.write_text('{"model_name": "old static artifact"}', encoding="utf-8")
    with pytest.raises(SystemExit, match="schema_version"):
        main(
            [
                str(document),
                "--question",
                "Which task?",
                "--mode",
                "hierarchical-static-keywords",
                "--topic-model",
                str(artifact),
            ]
        )


def test_load_static_keywords_uses_metadata_and_strategy(tmp_path) -> None:
    path = tmp_path / "tuned.json"
    path.write_text(
        json.dumps(
            {
                "model_name": "Tuned",
                "description": "test",
                "fields": [
                    {
                        "field_id": 1,
                        "field_name": "task",
                        "field_type": "specific_short",
                        "description": "task",
                        "questions": ["Which task?"],
                        "keywords": ["semantic"],
                        "keyword_metadata": [
                            {
                                "word": "semantic",
                                "lemma": "semantic",
                                "stem": "semant",
                                "attention_weight": 0.8,
                                "score_difference": 0.4,
                                "document_support": 4,
                                "selection_frequency": 0.8,
                                "marginal_gain": 0.1,
                            }
                        ],
                        "tuning_metadata": {
                            "strategy": {
                                "score_chunk_strategy": "only_weight",
                                "choose_cluster_strategy": "weighted_score",
                                "choose_answer_strategy": "combined_score",
                            }
                        },
                        "text": "task",
                        "required": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    keywords, strategy = load_static_keywords(path)
    assert keywords[0].stem == "semant"
    assert keywords[0].attention_weight == 0.8
    assert strategy["score_chunk_strategy"] == "only_weight"
