from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Language(str, Enum):
    EN = "en"
    RU = "ru"


@dataclass(frozen=True)
class ModelProfile:
    language: Language
    qa_model: str
    sentence_model: str
    attention_supported: bool = False

    @classmethod
    def english(cls, model_root: Path | None = None) -> "ModelProfile":
        root = model_root or Path("scripts/models_processing/models")
        return cls(
            language=Language.EN,
            qa_model=str(root / "bert_eng_qa_baseroberta_model"),
            sentence_model=str(root / "eng_sentence_transformer_model"),
            attention_supported=True,
        )

    @classmethod
    def russian(cls, model_root: Path | None = None) -> "ModelProfile":
        root = model_root or Path("scripts/models_processing/models")
        return cls(
            language=Language.RU,
            qa_model=str(root / "rubert_ru_qa_model"),
            sentence_model="DeepPavlov/rubert-base-cased-sentence",
        )


@dataclass(frozen=True)
class PipelineConfig:
    profile: ModelProfile = field(default_factory=ModelProfile.english)
    chunk_max_tokens: int = 384
    overlap_tokens: int = 50
    answer_cluster_threshold: float = 0.5
    strict_answer_threshold: float = 0.9
    min_answer_threshold: float = 0.7
    keyword_idf_threshold: float = 1.5
    keyword_similarity_threshold: float = 0.5
    keyword_dissimilarity_threshold: float = 0.3
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.chunk_max_tokens < 4:
            raise ValueError("chunk_max_tokens must be at least 4")
        if not 0 <= self.overlap_tokens < self.chunk_max_tokens / 2:
            raise ValueError("overlap_tokens must be non-negative and less than half the chunk size")
        for name in (
            "answer_cluster_threshold",
            "strict_answer_threshold",
            "min_answer_threshold",
            "keyword_similarity_threshold",
            "keyword_dissimilarity_threshold",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.min_answer_threshold > self.strict_answer_threshold:
            raise ValueError("min_answer_threshold cannot exceed strict_answer_threshold")
