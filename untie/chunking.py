from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .domain import Sentence, TextChunk
from .protocols import SentenceEncoder, WordTokenizer


@dataclass
class ChunkBuilder:
    tokenizer: WordTokenizer
    max_tokens: int = 384
    overlap_tokens: int = 50
    sentence_encoder: SentenceEncoder | None = None

    def __post_init__(self) -> None:
        if self.overlap_tokens >= self.max_tokens / 2:
            raise ValueError("overlap_tokens must be less than half of max_tokens")

    def prepare_sentences(self, texts: list[str]) -> list[Sentence]:
        prepared: list[Sentence] = []
        for number, text in enumerate(texts):
            tokens = tuple(self.tokenizer.tokenize(text))
            if len(tokens) <= self.max_tokens:
                prepared.append(Sentence(text, number, tokens, tuple(self.tokenizer.encode(text))))
                continue
            prepared.extend(self._split_long_sentence(text, number, tokens))
        return prepared

    def _split_long_sentence(
        self, text: str, number: int, tokens: tuple[str, ...]
    ) -> list[Sentence]:
        del text
        parts: list[Sentence] = []
        usable_size = max(1, self.max_tokens - 2)
        for start in range(0, len(tokens), usable_size):
            part_tokens = tokens[start : start + usable_size]
            part_text = self.tokenizer.convert_tokens_to_string(part_tokens)
            parts.append(
                Sentence(part_text, number, part_tokens, tuple(self.tokenizer.encode(part_text)))
            )
        return parts

    def build(self, sentences: list[Sentence]) -> list[TextChunk]:
        if not sentences:
            return []

        chunks: list[TextChunk] = []
        current: list[Sentence] = []
        current_count = 0

        for sentence in sentences:
            if current and current_count + sentence.token_count > self.max_tokens:
                chunks.append(self._to_chunk(current))
                current = self._overlap(current)
                current_count = sum(item.token_count for item in current)
            current.append(sentence)
            current_count += sentence.token_count

        if current:
            chunks.append(self._to_chunk(current))
        return chunks

    def _overlap(self, sentences: list[Sentence]) -> list[Sentence]:
        if self.overlap_tokens == 0:
            return []
        overlap: list[Sentence] = []
        count = 0
        for sentence in reversed(sentences):
            if overlap and count + sentence.token_count > self.overlap_tokens:
                break
            overlap.insert(0, sentence)
            count += sentence.token_count
        return overlap

    def _to_chunk(self, sentences: list[Sentence]) -> TextChunk:
        text = " ".join(sentence.text for sentence in sentences)
        embedding = None
        if self.sentence_encoder is not None:
            encoded = self.sentence_encoder.encode(text)
            embedding = np.asarray(encoded)
        return TextChunk(
            sentences=tuple(sentences),
            text=text,
            token_count=sum(sentence.token_count for sentence in sentences),
            embedding=embedding,
        )
