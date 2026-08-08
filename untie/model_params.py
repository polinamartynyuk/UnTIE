from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def _require_type(value: Any, expected: type, location: str) -> Any:
    if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
        raise ValueError(f"{location} must be {expected.__name__}")
    return value


def _require_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite")
    return result


def _validate_json(value: Any, location: str) -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must contain only finite numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} keys must be strings")
            _validate_json(item, f"{location}.{key}")
        return
    raise ValueError(f"{location} must be JSON-compatible")


def _validate_tuning_metadata(value: Any, location: str) -> dict[str, JsonValue]:
    metadata = _require_type(value, dict, location)
    _validate_json(metadata, location)
    string_keys = ("method", "dataset", "objective", "tuned_at")
    for key in string_keys:
        if key in metadata:
            _require_type(metadata[key], str, f"{location}.{key}")
    for key in ("score",):
        if key in metadata:
            _require_number(metadata[key], f"{location}.{key}")
    for key in ("document_count", "random_seed"):
        if key in metadata:
            number = _require_type(metadata[key], int, f"{location}.{key}")
            if key == "document_count" and number < 0:
                raise ValueError(f"{location}.{key} must be non-negative")
    return copy.deepcopy(metadata)


@dataclass
class KeywordMetadata:
    word: str
    lemma: str = ""
    stem: str = ""
    attention_weight: float = 0.0
    score_difference: float = 0.0
    document_support: int = 0
    selection_frequency: float = 0.0
    marginal_gain: float = 0.0
    extra_properties: dict[str, JsonValue] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], location: str = "keyword") -> "KeywordMetadata":
        if not isinstance(data, Mapping):
            raise ValueError(f"{location} must be an object")
        raw = dict(data)
        word = _require_type(raw.pop("word", None), str, f"{location}.word")
        lemma = _require_type(raw.pop("lemma", ""), str, f"{location}.lemma")
        stem = _require_type(raw.pop("stem", ""), str, f"{location}.stem")
        attention_weight = _require_number(
            raw.pop("attention_weight", raw.pop("weight", 0.0)),
            f"{location}.attention_weight",
        )
        score_difference = _require_number(
            raw.pop("score_difference", raw.pop("score_diff", 0.0)),
            f"{location}.score_difference",
        )
        document_support = _require_type(
            raw.pop("document_support", 0), int, f"{location}.document_support"
        )
        selection_frequency = _require_number(
            raw.pop("selection_frequency", 0.0), f"{location}.selection_frequency"
        )
        marginal_gain = _require_number(
            raw.pop("marginal_gain", 0.0), f"{location}.marginal_gain"
        )
        if document_support < 0 or not 0 <= selection_frequency <= 1:
            raise ValueError(
                f"{location} support must be non-negative and frequency between 0 and 1"
            )
        _validate_json(raw, f"{location} unknown properties")
        return cls(
            word=word,
            lemma=lemma,
            stem=stem,
            attention_weight=attention_weight,
            score_difference=score_difference,
            document_support=document_support,
            selection_frequency=selection_frequency,
            marginal_gain=marginal_gain,
            extra_properties=copy.deepcopy(raw),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        _validate_json(self.extra_properties, "keyword unknown properties")
        return {
            **copy.deepcopy(self.extra_properties),
            "word": self.word,
            "lemma": self.lemma,
            "stem": self.stem,
            "attention_weight": self.attention_weight,
            "score_difference": self.score_difference,
            "document_support": self.document_support,
            "selection_frequency": self.selection_frequency,
            "marginal_gain": self.marginal_gain,
        }


@dataclass
class ModelFieldParams:
    field_id: int
    field_name: str
    field_type: str
    description: str
    questions: list[str]
    keywords: list[str]
    text: str
    required: bool
    keyword_metadata: list[KeywordMetadata] = field(default_factory=list)
    tuning_metadata: dict[str, JsonValue] = field(default_factory=dict)
    extra_properties: dict[str, JsonValue] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], location: str = "field") -> "ModelFieldParams":
        if not isinstance(data, Mapping):
            raise ValueError(f"{location} must be an object")
        raw = dict(data)
        field_id = _require_type(raw.pop("field_id", None), int, f"{location}.field_id")
        field_name = _require_type(raw.pop("field_name", None), str, f"{location}.field_name")
        field_type = _require_type(raw.pop("field_type", None), str, f"{location}.field_type")
        description = _require_type(
            raw.pop("description", None), str, f"{location}.description"
        )
        questions = _string_list(raw.pop("questions", None), f"{location}.questions")
        text = _require_type(raw.pop("text", None), str, f"{location}.text")
        required = _require_type(raw.pop("required", None), bool, f"{location}.required")

        keyword_entries = _require_type(raw.pop("keywords", None), list, f"{location}.keywords")
        keywords: list[str] = []
        inline_metadata: list[KeywordMetadata] = []
        for index, entry in enumerate(keyword_entries):
            item_location = f"{location}.keywords[{index}]"
            if isinstance(entry, str):
                keywords.append(entry)
            elif isinstance(entry, dict):
                metadata = KeywordMetadata.from_dict(entry, item_location)
                keywords.append(metadata.word)
                inline_metadata.append(metadata)
            else:
                raise ValueError(f"{item_location} must be a string or object")

        metadata_entries = _require_type(
            raw.pop("keyword_metadata", []), list, f"{location}.keyword_metadata"
        )
        metadata = [
            KeywordMetadata.from_dict(item, f"{location}.keyword_metadata[{index}]")
            for index, item in enumerate(metadata_entries)
        ]
        metadata_words = {item.word for item in metadata}
        metadata.extend(item for item in inline_metadata if item.word not in metadata_words)
        tuning_metadata = _validate_tuning_metadata(
            raw.pop("tuning_metadata", {}), f"{location}.tuning_metadata"
        )
        _validate_json(raw, f"{location} unknown properties")
        return cls(
            field_id=field_id,
            field_name=field_name,
            field_type=field_type,
            description=description,
            questions=questions,
            keywords=keywords,
            text=text,
            required=required,
            keyword_metadata=metadata,
            tuning_metadata=tuning_metadata,
            extra_properties=copy.deepcopy(raw),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        _validate_json(self.extra_properties, "field unknown properties")
        result: dict[str, JsonValue] = {
            **copy.deepcopy(self.extra_properties),
            "field_id": self.field_id,
            "field_name": self.field_name,
            "field_type": self.field_type,
            "description": self.description,
            "questions": list(self.questions),
            "keywords": list(self.keywords),
            "text": self.text,
            "required": self.required,
        }
        if self.keyword_metadata:
            result["keyword_metadata"] = [item.to_dict() for item in self.keyword_metadata]
        if self.tuning_metadata:
            result["tuning_metadata"] = _validate_tuning_metadata(
                self.tuning_metadata, "field.tuning_metadata"
            )
        return result


FieldParams = ModelFieldParams


@dataclass
class ModelParams:
    model_name: str
    description: str
    fields: list[ModelFieldParams]
    tuning_metadata: dict[str, JsonValue] = field(default_factory=dict)
    extra_properties: dict[str, JsonValue] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelParams":
        if not isinstance(data, Mapping):
            raise ValueError("model params must be an object")
        raw = dict(data)
        model_name = _require_type(raw.pop("model_name", None), str, "model_name")
        description = _require_type(raw.pop("description", None), str, "description")
        field_entries = _require_type(raw.pop("fields", None), list, "fields")
        fields = [
            ModelFieldParams.from_dict(item, f"fields[{index}]")
            for index, item in enumerate(field_entries)
        ]
        tuning_metadata = _validate_tuning_metadata(
            raw.pop("tuning_metadata", {}), "tuning_metadata"
        )
        _validate_json(raw, "unknown top-level properties")
        return cls(
            model_name=model_name,
            description=description,
            fields=fields,
            tuning_metadata=tuning_metadata,
            extra_properties=copy.deepcopy(raw),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        _validate_json(self.extra_properties, "unknown top-level properties")
        result: dict[str, JsonValue] = {
            **copy.deepcopy(self.extra_properties),
            "model_name": self.model_name,
            "description": self.description,
            "fields": [item.to_dict() for item in self.fields],
        }
        if self.tuning_metadata:
            result["tuning_metadata"] = _validate_tuning_metadata(
                self.tuning_metadata, "tuning_metadata"
            )
        return result


def _string_list(value: Any, location: str) -> list[str]:
    items = _require_type(value, list, location)
    for index, item in enumerate(items):
        _require_type(item, str, f"{location}[{index}]")
    return list(items)


def load_model_params(path: str | os.PathLike[str]) -> ModelParams:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load model params from {source}: {exc}") from exc
    return ModelParams.from_dict(data)


def save_model_params_atomic(
    model: ModelParams, path: str | os.PathLike[str]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        model.to_dict(), ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def create_tuned_copy(
    model: ModelParams,
    *,
    fields: Sequence[ModelFieldParams] | None = None,
    tuning_metadata: Mapping[str, JsonValue] | None = None,
) -> ModelParams:
    """Return an independent copy with selected tuning results replaced."""
    tuned = copy.deepcopy(model)
    if fields is not None:
        tuned.fields = copy.deepcopy(list(fields))
    if tuning_metadata is not None:
        tuned.tuning_metadata = _validate_tuning_metadata(
            dict(tuning_metadata), "tuning_metadata"
        )
    return tuned
