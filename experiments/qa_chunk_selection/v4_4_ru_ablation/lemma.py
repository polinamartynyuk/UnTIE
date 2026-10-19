"""Russian lemma overlap used as an ablation and as a secondary metric only."""

import re
from collections import Counter
from functools import lru_cache


class RussianLemmatizer:
    def __init__(self, *, smoke=False):
        self.smoke = smoke
        self._morph = None
        if not smoke:
            try:
                import pymorphy3
            except ImportError as exc:
                raise RuntimeError("pymorphy3 is required: install the project 'nlp' extra") from exc
            self._morph = pymorphy3.MorphAnalyzer()

    @lru_cache(maxsize=200000)
    def token(self, value):
        value = value.casefold()
        if self._morph is None or not re.search(r"[а-яё]", value):
            return value
        return self._morph.parse(value)[0].normal_form

    def tokens(self, text):
        return [self.token(x) for x in re.findall(r"\w+", str(text).casefold(), re.UNICODE)]

    def key(self, text):
        return " ".join(self.tokens(text))

    def f1(self, left, right):
        a, b = self.tokens(left), self.tokens(right)
        if not a or not b:
            return 100.0 if a == b else 0.0
        common = sum((Counter(a) & Counter(b)).values())
        if not common:
            return 0.0
        precision, recall = common / len(a), common / len(b)
        return 200.0 * precision * recall / (precision + recall)

    def gold_f1(self, prediction, references):
        return max((self.f1(prediction, ref) for ref in references), default=0.0)

