from __future__ import annotations

import re


_PROTECTED_DOT = "<prd>"
_STOP = "<stop>"
_PREFIXES = re.compile(r"\b(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)\.")
_WEBSITES = re.compile(r"\.(com|net|org|io|gov|me|edu|ru|рф|su)\b", re.IGNORECASE)
_INITIAL = re.compile(r"\b([A-Za-zА-Яа-я])\.")
_DECIMAL = re.compile(r"(\d)\.(\d)")
_RUSSIAN_PREFIXES = re.compile(
    r"\b(г-н|г-жа|проф|акад|докт|инж|ген|полк|кап|лейт|серж|мл|ст)\.",
    re.IGNORECASE,
)
_RUSSIAN_ABBREVIATIONS = re.compile(
    r"\b(?:т\.?\s?е|т\.?\s?к|и\.?\s?т\.?\s?[дп]|см|рис|табл|гл|стр|"
    r"тыс|млн|млрд|руб|коп|кв|корп)\.",
    re.IGNORECASE,
)


def _protect_dots(match: re.Match[str]) -> str:
    return match.group(0).replace(".", _PROTECTED_DOT)


class SentenceSplitter:
    """Разделяет текст на предложения по правилам исходного английского сплиттера."""

    def split(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        protected = " " + text.replace("\n", " ").strip() + " "
        protected = _PREFIXES.sub(r"\1" + _PROTECTED_DOT, protected)
        protected = _WEBSITES.sub(_PROTECTED_DOT + r"\1", protected)
        protected = _INITIAL.sub(r"\1" + _PROTECTED_DOT, protected)
        protected = _DECIMAL.sub(r"\1" + _PROTECTED_DOT + r"\2", protected)
        protected = protected.replace("Ph.D.", "Ph<prd>D<prd>")

        for punctuation in ".?!":
            protected = protected.replace(punctuation, punctuation + _STOP)

        protected = protected.replace(_PROTECTED_DOT, ".")
        return [
            sentence.strip()
            for sentence in protected.split(_STOP)
            if sentence.strip() and sentence.strip() not in {".", "?", "!"}
        ]


class RussianSentenceSplitter:
    """Разделяет русский текст, сохраняя сокращения, инициалы и числа."""

    def split(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        protected = " " + text.replace("\n", " ").strip() + " "
        protected = _RUSSIAN_PREFIXES.sub(
            lambda match: match.group(1) + _PROTECTED_DOT,
            protected,
        )
        protected = _RUSSIAN_ABBREVIATIONS.sub(_protect_dots, protected)
        protected = _WEBSITES.sub(_PROTECTED_DOT + r"\1", protected)
        protected = _INITIAL.sub(r"\1" + _PROTECTED_DOT, protected)
        protected = _DECIMAL.sub(r"\1" + _PROTECTED_DOT + r"\2", protected)

        for punctuation in ".?!…":
            protected = protected.replace(punctuation, punctuation + _STOP)

        protected = protected.replace(_PROTECTED_DOT, ".")
        return [
            sentence.strip()
            for sentence in protected.split(_STOP)
            if sentence.strip() and sentence.strip() not in {".", "?", "!", "…"}
        ]
