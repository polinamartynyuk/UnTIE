"""Small ranking heads. No frozen-QA fine-tune."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def softmax(values: Sequence[float]) -> list[float]:
    rows = np.asarray(list(values), dtype=float)
    if rows.size == 0:
        return []
    shifted = rows - np.max(rows)
    exp = np.exp(shifted)
    denom = float(np.sum(exp)) or 1.0
    return [float(item) / denom for item in exp]


def rank_by_scores(chunk_ids: Sequence[int], scores: Sequence[float]) -> list[int]:
    paired = sorted(zip([int(cid) for cid in chunk_ids], [float(s) for s in scores]), key=lambda item: (-item[1], item[0]))
    return [cid for cid, _score in paired]


class LinearHead:
    def __init__(self) -> None:
        self.coef: np.ndarray | None = None
        self.intercept = 0.0
        self.feature_names: list[str] = []

    def fit(self, matrix: Sequence[Sequence[float]], labels: Sequence[int], names: Sequence[str]) -> "LinearHead":
        self.feature_names = list(names)
        x = np.asarray(matrix, dtype=float)
        y = np.asarray(list(labels), dtype=float)
        if x.size == 0 or len(set(int(item) for item in labels)) < 2:
            self.coef = np.zeros(x.shape[1] if x.ndim == 2 and x.shape[1] else 1)
            return self
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200, solver="lbfgs")
        model.fit(x, y)
        self.coef = np.asarray(model.coef_[0], dtype=float)
        self.intercept = float(model.intercept_[0])
        return self

    def score(self, features: Sequence[float]) -> float:
        if self.coef is None:
            return 0.0
        vec = np.asarray(list(features), dtype=float)
        coef = self.coef
        if vec.size != coef.size:
            n = min(vec.size, coef.size)
            return float(self.intercept + vec[:n].dot(coef[:n]))
        return float(self.intercept + vec.dot(coef))

    def to_dict(self) -> dict[str, Any]:
        return {
            "coef": [] if self.coef is None else [float(item) for item in self.coef],
            "intercept": self.intercept,
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinearHead":
        head = cls()
        head.coef = np.asarray(payload.get("coef") or [0.0], dtype=float)
        head.intercept = float(payload.get("intercept") or 0.0)
        head.feature_names = list(payload.get("feature_names") or [])
        return head
