"""Answerability Cross-Encoder training (experiment B). Isolated from production."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .metrics import best_threshold, binary_pr_auc, binary_roc_auc, pairwise_pos_gt_hn
from .progress import ProgressLogger
from .schema import ChunkRecord, InstanceRecord


@dataclass(frozen=True)
class TrainPair:
    question: str
    text: str
    label: int
    document_id: str
    question_id: str
    chunk_id: int
    kind: str


def _repeat_to_ratio(
    positives: list[TrainPair],
    hards: list[TrainPair],
    easies: list[TrainPair],
    *,
    pos_n: int,
    hn_n: int,
    easy_n: int,
    rng: random.Random,
) -> list[TrainPair]:
    selected: list[TrainPair] = list(positives)
    if positives and hn_n > 0 and hards:
        target_hn = max(1, int(round(len(positives) * hn_n / max(pos_n, 1))))
        selected.extend(_sample_or_repeat(hards, target_hn, rng))
    if positives and easy_n > 0 and easies:
        target_easy = max(1, int(round(len(positives) * easy_n / max(pos_n, 1))))
        selected.extend(_sample_or_repeat(easies, target_easy, rng))
    rng.shuffle(selected)
    return selected


def _sample_or_repeat(items: Sequence[TrainPair], n: int, rng: random.Random) -> list[TrainPair]:
    if not items:
        return []
    if n <= len(items):
        return rng.sample(list(items), n)
    extra = [items[index % len(items)] for index in range(n - len(items))]
    return list(items) + extra


def collect_pairs(
    records: Sequence[InstanceRecord],
    *,
    ratio: tuple[int, int, int],
    seed: int,
) -> list[TrainPair]:
    rng = random.Random(int(seed))
    positives: list[TrainPair] = []
    hards: list[TrainPair] = []
    easies: list[TrainPair] = []
    for item in records:
        for chunk in item.chunks:
            pair = TrainPair(
                question=item.question,
                text=chunk.text,
                label=1 if chunk.label == "positive" else 0,
                document_id=item.document_id,
                question_id=item.question_id,
                chunk_id=chunk.chunk_id,
                kind=chunk.label,
            )
            if chunk.label == "positive":
                positives.append(pair)
            elif chunk.label == "hard_negative":
                hards.append(pair)
            elif chunk.label == "easy_negative":
                easies.append(pair)
    return _repeat_to_ratio(
        positives,
        hards,
        easies,
        pos_n=ratio[0],
        hn_n=ratio[1],
        easy_n=ratio[2],
        rng=rng,
    )


class AnswerabilityModel:
    def __init__(self, model_name: str, device: str, max_length: int = 512) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        import torch

        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.model.to(device)
        self._torch = torch

    def _batch_logits(self, questions: Sequence[str], passages: Sequence[str]) -> np.ndarray:
        encoded = self.tokenizer(
            list(questions),
            list(passages),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(**encoded).logits.view(-1)
        return logits.detach().cpu().numpy().astype(float)

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        self.model.eval()
        scores: list[float] = []
        batch_size = 16
        with self._torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                logits = self._batch_logits(
                    [item[0] for item in batch],
                    [item[1] for item in batch],
                )
                probs = 1.0 / (1.0 + np.exp(-logits))
                scores.extend(float(item) for item in probs)
        return scores

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "wrapper.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "max_length": self.max_length,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, device: str) -> "AnswerabilityModel":
        wrapper = json.loads((path / "wrapper.json").read_text(encoding="utf-8"))
        item = cls(str(path), device=device, max_length=int(wrapper.get("max_length", 512)))
        item.model_name = wrapper.get("model_name", str(path))
        return item


def train_answerability(
    train_pairs: Sequence[TrainPair],
    *,
    model_name: str,
    device: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    loss_name: str,
    margin: float,
    progress: ProgressLogger | None = None,
) -> AnswerabilityModel:
    import torch
    from torch.optim import AdamW

    model = AnswerabilityModel(model_name, device=device, max_length=max_length)
    optimizer = AdamW(model.model.parameters(), lr=float(learning_rate))
    model.model.train()
    pairs = list(train_pairs)
    if not pairs:
        raise ValueError("no answerability training pairs")
    for epoch in range(1, int(epochs) + 1):
        random.Random(epoch).shuffle(pairs)
        running = 0.0
        steps = 0
        for start in range(0, len(pairs), int(batch_size)):
            batch = pairs[start : start + int(batch_size)]
            encoded = model.tokenizer(
                [item.question for item in batch],
                [item.text for item in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model.model(**encoded).logits.view(-1)
            labels = torch.tensor([item.label for item in batch], device=device, dtype=torch.float32)
            if loss_name == "pairwise":
                pos = logits[labels > 0.5]
                neg = logits[labels < 0.5]
                if len(pos) == 0 or len(neg) == 0:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                else:
                    diffs = float(margin) - pos.unsqueeze(1) + neg.unsqueeze(0)
                    loss = torch.clamp(diffs, min=0).mean()
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1
        if progress:
            progress.log(
                f"B train epoch {epoch}/{epochs} loss={running / max(steps, 1):.4f} pairs={len(pairs)}"
            )
    model.model.eval()
    model.save(output_dir)
    return model


def evaluate_answerability_scores(
    records: Sequence[InstanceRecord],
    scorer,
) -> dict[str, Any]:
    labels: list[int] = []
    scores: list[float] = []
    pairwise_values: list[float] = []
    for item in records:
        pairs = [(item.question, chunk.text) for chunk in item.chunks]
        values = list(scorer.predict(pairs))
        by_id = {chunk.chunk_id: float(score) for chunk, score in zip(item.chunks, values)}
        for chunk, score in zip(item.chunks, values):
            if chunk.label == "ambiguous":
                continue
            labels.append(1 if chunk.label == "positive" else 0)
            scores.append(float(score))
        acc, n_pairs = pairwise_pos_gt_hn(
            by_id,
            [chunk.chunk_id for chunk in item.positives],
            [chunk.chunk_id for chunk in item.hard_negatives],
        )
        if n_pairs:
            pairwise_values.append(acc)
    return {
        "pr_auc": binary_pr_auc(labels, scores),
        "roc_auc": binary_roc_auc(labels, scores),
        "pairwise_pos_gt_hn": float(sum(pairwise_values) / len(pairwise_values))
        if pairwise_values
        else float("nan"),
        "n_scored": len(labels),
        "threshold": best_threshold(labels, scores) if labels else {"threshold": 0.5, "f1": 0.0},
    }


class FrozenAnswerabilityScorer:
    def __init__(self, model: AnswerabilityModel) -> None:
        self.model = model

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return self.model.predict(pairs)


class TfidfAnswerabilityScorer:
    """CPU fallback used by --smoke. Still trained on pos vs HN, never on gold text."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
        self.classifier = LogisticRegression(max_iter=200)
        self._fitted = False

    def fit(self, pairs: Sequence[TrainPair]) -> "TfidfAnswerabilityScorer":
        texts = [f"{item.question} [SEP] {item.text}" for item in pairs]
        labels = [item.label for item in pairs]
        if len(set(labels)) < 2:
            raise ValueError("need both positive and negative pairs to fit smoke B")
        matrix = self.vectorizer.fit_transform(texts)
        self.classifier.fit(matrix, labels)
        self._fitted = True
        return self

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not self._fitted:
            raise RuntimeError("TfidfAnswerabilityScorer is not fitted")
        texts = [f"{left} [SEP] {right}" for left, right in pairs]
        matrix = self.vectorizer.transform(texts)
        probs = self.classifier.predict_proba(matrix)
        positive_index = list(self.classifier.classes_).index(1)
        return [float(row[positive_index]) for row in probs]

    def save(self, path: Path) -> None:
        import pickle

        path.mkdir(parents=True, exist_ok=True)
        (path / "wrapper.json").write_text(
            json.dumps({"kind": "tfidf_logistic"}) + "\n", encoding="utf-8"
        )
        with (path / "model.pkl").open("wb") as handle:
            pickle.dump({"vectorizer": self.vectorizer, "classifier": self.classifier}, handle)

    @classmethod
    def load(cls, path: Path) -> "TfidfAnswerabilityScorer":
        import pickle

        item = cls()
        with (path / "model.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        item.vectorizer = payload["vectorizer"]
        item.classifier = payload["classifier"]
        item._fitted = True
        return item
