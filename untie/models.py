from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

from .config import ModelProfile, PipelineConfig


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass
class ExtractiveQuestionAnswerer:
    """Извлекает ответ по start/end logits независимо от версии transformers."""

    model: Any
    tokenizer: Any
    device: str
    max_length: int = 512
    max_answer_tokens: int = 64

    def __call__(self, *, question: str, context: str) -> dict[str, Any]:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Install UnTIE with the 'models' extra") from error

        encoded = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0]
        try:
            sequence_ids = encoded.sequence_ids(0)
        except (AttributeError, ValueError) as error:
            raise RuntimeError(
                "Extractive QA requires a fast tokenizer with sequence IDs"
            ) from error
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        context_indices = [
            index for index, sequence_id in enumerate(sequence_ids) if sequence_id == 1
        ]
        if not context_indices:
            return {"answer": "", "score": 0.0, "start": 0, "end": 0}

        best_start = context_indices[0]
        best_end = best_start
        best_logit_score = float("-inf")
        context_set = set(context_indices)
        for start in context_indices:
            last_end = min(start + self.max_answer_tokens, context_indices[-1] + 1)
            for end in range(start, last_end):
                if end not in context_set:
                    continue
                score = float(start_logits[start] + end_logits[end])
                if score > best_logit_score:
                    best_logit_score = score
                    best_start, best_end = start, end

        start_offset = int(offsets[best_start][0])
        end_offset = int(offsets[best_end][1])
        start_probability = torch.softmax(start_logits, dim=0)[best_start]
        end_probability = torch.softmax(end_logits, dim=0)[best_end]
        return {
            "answer": context[start_offset:end_offset],
            "score": float(start_probability * end_probability),
            "start": start_offset,
            "end": end_offset,
        }


@dataclass
class ModelFactory:
    """Лениво загружает необязательные ML-зависимости и веса моделей."""

    config: PipelineConfig

    @cached_property
    def device(self) -> str:
        return resolve_device(self.config.device)

    @cached_property
    def tokenizer(self) -> Any:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install UnTIE with the 'models' extra") from error
        return AutoTokenizer.from_pretrained(self.config.profile.qa_model)

    @cached_property
    def qa_model(self) -> Any:
        try:
            from transformers import AutoModelForQuestionAnswering
        except ImportError as error:
            raise RuntimeError("Install UnTIE with the 'models' extra") from error
        return AutoModelForQuestionAnswering.from_pretrained(
            self.config.profile.qa_model,
            output_attentions=self.config.profile.attention_supported,
        ).to(self.device)

    @cached_property
    def answerer(self) -> Any:
        return ExtractiveQuestionAnswerer(
            model=self.qa_model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    @cached_property
    def sentence_encoder(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Install UnTIE with the 'models' extra") from error
        return SentenceTransformer(self.config.profile.sentence_model, device=self.device)


def profile_for_language(language: str) -> ModelProfile:
    normalized = language.lower()
    if normalized == "en":
        return ModelProfile.english()
    if normalized == "ru":
        return ModelProfile.russian()
    raise ValueError("language must be 'en' or 'ru'")
