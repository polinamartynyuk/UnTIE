from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from untie.model_params import (
    KeywordMetadata,
    ModelFieldParams,
    ModelParams,
    create_tuned_copy,
    load_model_params,
    save_model_params_atomic,
)


PROJECT_ROOT = Path(__file__).parents[1]


def _field(**overrides: object) -> ModelFieldParams:
    values = {
        "field_id": 1,
        "field_name": "task",
        "field_type": "specific_short",
        "description": "Main task",
        "questions": ["Which task was solved?"],
        "keywords": ["segmentation"],
        "text": "Task guidance",
        "required": True,
    }
    values.update(overrides)
    return ModelFieldParams(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("filename", ["scart_init_model.json", "ruserrc_init_model.json"])
def test_loads_current_init_schema(filename: str) -> None:
    model = load_model_params(PROJECT_ROOT / "model_params" / filename)

    assert model.model_name
    assert len(model.fields) == 1
    assert model.fields[0].questions
    assert model.fields[0].keywords == []


def test_roundtrip_normalizes_legacy_keyword_objects(tmp_path: Path) -> None:
    source = tmp_path / "legacy.json"
    source.write_text(
        json.dumps(
            {
                "model_name": "Legacy",
                "description": "Legacy keyword forms",
                "fields": [
                    {
                        "field_id": 1,
                        "field_name": "task",
                        "field_type": "specific_short",
                        "description": "Task",
                        "questions": ["Task?"],
                        "keywords": [
                            "plain",
                            {
                                "word": "semantic",
                                "lemma": "semantic",
                                "stem": "semant",
                                "weight": 0.8,
                                "score_diff": 0.4,
                                "document_support": 3,
                                "selection_frequency": 0.75,
                                "marginal_gain": 0.15,
                            },
                        ],
                        "text": "Guidance",
                        "required": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_model_params(source)
    destination = tmp_path / "saved.json"
    save_model_params_atomic(loaded, destination)
    serialized = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded.fields[0].keywords == ["plain", "semantic"]
    assert serialized["fields"][0]["keywords"] == ["plain", "semantic"]
    assert serialized["fields"][0]["keyword_metadata"][0]["attention_weight"] == 0.8
    assert serialized["fields"][0]["keyword_metadata"][0]["score_difference"] == 0.4
    assert load_model_params(destination) == loaded


def test_unknown_properties_survive_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "extended.json"
    original = {
        "model_name": "Extended",
        "description": "Preserves extensions",
        "vendor": {"revision": 2, "flags": [True, None]},
        "fields": [
            {
                "field_id": 7,
                "field_name": "method",
                "field_type": "specific_short",
                "description": "Method",
                "questions": ["Which method?"],
                "keywords": ["transformer"],
                "text": "Method guidance",
                "required": False,
                "display": {"color": "blue"},
                "keyword_metadata": [
                    {
                        "word": "transformer",
                        "lemma": "transformer",
                        "stem": "transform",
                        "attention_weight": 0.7,
                        "score_difference": 0.2,
                        "document_support": 5,
                        "selection_frequency": 0.6,
                        "marginal_gain": 0.1,
                        "source": "candidate-pool",
                    }
                ],
            }
        ],
    }
    source.write_text(json.dumps(original), encoding="utf-8")

    model = load_model_params(source)
    output = tmp_path / "roundtrip.json"
    save_model_params_atomic(model, output)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved == original


def test_supports_multiple_fields_and_tuning_metadata(tmp_path: Path) -> None:
    model = ModelParams(
        model_name="Tuned",
        description="Two fields",
        fields=[_field(), replace(_field(), field_id=2, field_name="method")],
        tuning_metadata={
            "method": "cross-validation",
            "dataset": "sample",
            "score": 0.91,
            "document_count": 12,
            "custom": {"folds": [1, 2, 3]},
        },
    )
    path = tmp_path / "nested" / "model.json"

    save_model_params_atomic(model, path)

    assert load_model_params(path) == model
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update(fields="not-a-list"),
        lambda data: data["fields"][0].update(field_id=True),
        lambda data: data["fields"][0].update(questions=["valid", 3]),
        lambda data: data["fields"][0].update(keywords=[None]),
        lambda data: data.update(tuning_metadata={"document_count": -1}),
        lambda data: data.update(tuning_metadata={"custom": object()}),
    ],
)
def test_rejects_malformed_schema(tmp_path: Path, mutation) -> None:
    data = {
        "model_name": "Model",
        "description": "Description",
        "fields": [_field().to_dict()],
    }
    mutation(data)
    path = tmp_path / "bad.json"
    if data.get("tuning_metadata", {}).get("custom") is not None:
        with pytest.raises(ValueError, match="JSON-compatible"):
            ModelParams.from_dict(data)
        return
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError):
        load_model_params(path)


def test_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="could not load model params"):
        load_model_params(path)


def test_create_tuned_copy_leaves_original_unchanged() -> None:
    original = ModelParams(
        "Original",
        "Untuned",
        [_field(extra_properties={"nested": {"value": 1}})],
        extra_properties={"owner": {"name": "team"}},
    )
    metadata = KeywordMetadata(
        word="semantic",
        lemma="semantic",
        stem="semant",
        attention_weight=0.8,
        score_difference=0.3,
        document_support=4,
        selection_frequency=0.8,
        marginal_gain=0.12,
    )
    tuned_field = replace(
        original.fields[0],
        keywords=["semantic"],
        keyword_metadata=[metadata],
    )

    tuned = create_tuned_copy(
        original,
        fields=[tuned_field],
        tuning_metadata={"method": "greedy", "custom": {"threshold": 0.5}},
    )
    tuned.extra_properties["owner"]["name"] = "other"  # type: ignore[index]
    tuned.fields[0].extra_properties["nested"]["value"] = 2  # type: ignore[index]

    assert original.fields[0].keywords == ["segmentation"]
    assert original.fields[0].keyword_metadata == []
    assert original.tuning_metadata == {}
    assert original.extra_properties == {"owner": {"name": "team"}}
    assert original.fields[0].extra_properties == {"nested": {"value": 1}}
