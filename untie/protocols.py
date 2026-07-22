from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray


class WordTokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...

    def encode(self, text: str, **kwargs: Any) -> Sequence[int]: ...

    def convert_tokens_to_string(self, tokens: Sequence[str]) -> str: ...


class SentenceEncoder(Protocol):
    def encode(self, texts: str | Sequence[str], **kwargs: Any) -> NDArray[np.floating[Any]]: ...


class QuestionAnswerer(Protocol):
    def __call__(self, *, question: str, context: str) -> dict[str, Any]: ...


class Lemmatizer(Protocol):
    def __call__(self, word: str) -> str: ...
