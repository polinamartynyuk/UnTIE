from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _token_scheme(tokenizer: Any, tokens: list[str]) -> str:
    """Определяет схему субтокенов, не меняя RoBERTa/SentencePiece-путь."""
    class_name = tokenizer.__class__.__name__.lower()
    if any(token.startswith("##") for token in tokens):
        return "wordpiece"
    if "bert" in class_name and "roberta" not in class_name:
        return "wordpiece"
    return "sentencepiece"


def reconstruct_attention_words(
    tokens: list[str],
    weights: list[float],
    *,
    scheme: str = "sentencepiece",
) -> dict[str, list[float]]:
    """Собирает слова и веса из RoBERTa/SentencePiece/WordPiece-субтокенов."""
    if len(tokens) != len(weights):
        raise ValueError("tokens and weights must have the same length")
    if scheme not in {"sentencepiece", "wordpiece"}:
        raise ValueError("scheme must be 'sentencepiece' or 'wordpiece'")

    words: dict[str, list[float]] = {}
    current = ""
    current_weights: list[float] = []

    def flush() -> None:
        nonlocal current, current_weights
        if current:
            words.setdefault(current, []).extend(current_weights)
        current = ""
        current_weights = []

    for token, weight in zip(tokens, weights):
        if scheme == "wordpiece":
            starts_word = not token.startswith("##")
        else:
            starts_word = token.startswith(("Ġ", "▁"))
        cleaned = (
            token.lstrip("Ġ▁")
            .removeprefix("##")
            .strip(".,?!:;()[]{}\"'")
            .lower()
        )
        if not cleaned:
            continue
        if starts_word:
            flush()
            current = cleaned
            current_weights = [float(weight)]
        else:
            current += cleaned
            current_weights.append(float(weight))
    flush()
    return words


@dataclass
class AttentionKeywordExtractor:
    model: Any
    tokenizer: Any
    device: str
    limit: int = 100

    def __call__(self, question: str, context: str) -> list[dict[str, float | str]]:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Install UnTIE with the 'models' extra") from error

        encoded = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=512,
        )
        try:
            sequence_ids = encoded.sequence_ids(0)
        except (AttributeError, ValueError) as error:
            raise RuntimeError(
                "Attention extraction requires a fast tokenizer with sequence IDs"
            ) from error
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        if not outputs.attentions:
            raise RuntimeError("Configured QA model does not expose attention weights")

        attention = torch.stack(outputs.attentions).mean(dim=(0, 1, 2))
        input_ids = inputs["input_ids"][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        special_ids = set(self.tokenizer.all_special_ids)
        weights = attention.mean(dim=0).detach().cpu().numpy()

        context_tokens: list[str] = []
        context_weights: list[float] = []
        for token_index, (token_id, token, weight) in enumerate(
            zip(input_ids.tolist(), tokens, weights)
        ):
            if token_id in special_ids or sequence_ids[token_index] != 1:
                continue
            context_tokens.append(token)
            context_weights.append(float(weight))
        words = reconstruct_attention_words(
            context_tokens,
            context_weights,
            scheme=_token_scheme(self.tokenizer, context_tokens),
        )

        ranked = sorted(
            (
                {"word": word, "weight": float(np.mean(word_weights))}
                for word, word_weights in words.items()
                if len(word) > 1
            ),
            key=lambda item: float(item["weight"]),
            reverse=True,
        )
        return ranked[: self.limit]
