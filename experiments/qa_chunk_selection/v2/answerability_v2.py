"""B2 answerability training: question-balanced, clean HN, early stopping, 3 seeds."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.qa_chunk_selection.answerability import (
    AnswerabilityModel,
    TfidfAnswerabilityScorer,
    TrainPair,
)
from experiments.qa_chunk_selection.metrics import best_threshold, binary_pr_auc, pairwise_pos_gt_hn
from experiments.qa_chunk_selection.schema import InstanceRecord

from .hn_flags import flag_index, is_training_eligible_hn
from .io_util import write_json
from .schemas import RuntimeInstance, to_runtime


@dataclass(frozen=True)
class WeightedPair:
    pair: TrainPair
    weight: float


def _eligible_hn(item: InstanceRecord, flags) -> list:
    return [
        chunk
        for chunk in item.hard_negatives
        if is_training_eligible_hn(flags, item.question_id, chunk.chunk_id)
    ]


def unique_question_pairs(
    records: Sequence[InstanceRecord],
    flags,
    *,
    ratio: tuple[int, int, int],
) -> tuple[list[WeightedPair], dict[str, Any]]:
    pos_n, hn_n, easy_n = ratio
    total_w = float(pos_n + hn_n + easy_n) or 1.0
    w_pos, w_hn, w_easy = pos_n / total_w, hn_n / total_w, easy_n / total_w
    questions = [item for item in records if item.positives]
    n_q = max(len(questions), 1)
    selected: list[WeightedPair] = []
    for item in questions:
        pos = list(item.positives)
        hards = _eligible_hn(item, flags)
        easies = list(item.easy_negatives)
        q_weight = 1.0 / n_q
        classes = []
        if pos:
            classes.append((pos, 1, "positive", w_pos))
        if hards:
            classes.append((hards, 0, "hard_negative", w_hn))
        if easies:
            classes.append((easies, 0, "easy_negative", w_easy))
        share_sum = sum(share for *_rest, share in classes) or 1.0
        for chunks, label, kind, share in classes:
            unit = q_weight * (share / share_sum) / len(chunks)
            for chunk in chunks:
                pair = TrainPair(
                    question=item.question,
                    text=chunk.text,
                    label=label,
                    document_id=item.document_id,
                    question_id=item.question_id,
                    chunk_id=chunk.chunk_id,
                    kind=kind,
                )
                selected.append(WeightedPair(pair, unit))
    weights = defaultdict(float)
    counts = defaultdict(int)
    for row in selected:
        weights[row.pair.question_id] += row.weight
        counts[row.pair.question_id] += 1
    report = {
        "n_questions": n_q,
        "n_pairs": len(selected),
        "mean_weight": float(sum(weights.values()) / max(len(weights), 1)),
        "max_weight": float(max(weights.values()) if weights else 0.0),
        "min_weight": float(min(weights.values()) if weights else 0.0),
        "mean_samples": float(sum(counts.values()) / max(len(counts), 1)),
        "max_samples": int(max(counts.values()) if counts else 0),
        "min_samples": int(min(counts.values()) if counts else 0),
        "weight_formula": "each question_id total weight 1/n_questions; class shares from pos:hn:easy ratio; unique pairs, no HN oversampling",
    }
    if weights:
        spread = max(weights.values()) / max(min(weights.values()), 1e-12)
        report["question_weight_spread"] = float(spread)
    return selected, report


def sample_dynamic_epoch(
    records: Sequence[InstanceRecord],
    flags,
    *,
    ratio: tuple[int, int, int],
    seed: int,
    epoch: int,
) -> tuple[list[TrainPair], dict[str, int]]:
    rng = random.Random(int(seed) * 1009 + int(epoch))
    pos_n, hn_n, easy_n = ratio
    usage: dict[str, int] = defaultdict(int)
    pairs: list[TrainPair] = []
    for item in records:
        if not item.positives:
            continue
        for chunk in item.positives:
            pairs.append(
                TrainPair(item.question, chunk.text, 1, item.document_id, item.question_id, chunk.chunk_id, "positive")
            )
        hards = _eligible_hn(item, flags)
        target_hn = max(1, int(round(len(item.positives) * hn_n / max(pos_n, 1)))) if hards else 0
        if hards and target_hn:
            unique = list(hards)
            rng.shuffle(unique)
            picked = unique[: min(target_hn, len(unique))]
            while len(picked) < target_hn:
                picked.append(rng.choice(hards))
            for chunk in picked:
                usage[f"{item.question_id}:{chunk.chunk_id}"] += 1
                pairs.append(
                    TrainPair(item.question, chunk.text, 0, item.document_id, item.question_id, chunk.chunk_id, "hard_negative")
                )
        easies = list(item.easy_negatives)
        target_easy = max(1, int(round(len(item.positives) * easy_n / max(pos_n, 1)))) if easies and easy_n else 0
        if easies and target_easy:
            picked = rng.sample(easies, min(target_easy, len(easies)))
            for chunk in picked:
                pairs.append(
                    TrainPair(item.question, chunk.text, 0, item.document_id, item.question_id, chunk.chunk_id, "easy_negative")
                )
    rng.shuffle(pairs)
    return pairs, dict(usage)


def sample_pairwise_epoch(
    records: Sequence[InstanceRecord],
    flags,
    *,
    seed: int,
    epoch: int,
    max_pairs_per_question: int = 8,
) -> list[tuple[TrainPair, TrainPair]]:
    rng = random.Random(int(seed) * 9176 + int(epoch))
    triples: list[tuple[TrainPair, TrainPair]] = []
    for item in records:
        pos = list(item.positives)
        hards = _eligible_hn(item, flags)
        if not pos or not hards:
            continue
        n = min(max_pairs_per_question, max(len(pos), len(hards)))
        for _ in range(n):
            p = rng.choice(pos)
            h = rng.choice(hards)
            triples.append(
                (
                    TrainPair(item.question, p.text, 1, item.document_id, item.question_id, p.chunk_id, "positive"),
                    TrainPair(item.question, h.text, 0, item.document_id, item.question_id, h.chunk_id, "hard_negative"),
                )
            )
    rng.shuffle(triples)
    return triples


def evaluate_answerability(
    records: Sequence[InstanceRecord],
    scorer,
    flags,
) -> dict[str, Any]:
    labels: list[int] = []
    scores: list[float] = []
    pairwise_all: list[float] = []
    pairwise_clean: list[float] = []
    for item in records:
        values = list(scorer.predict([(item.question, chunk.text) for chunk in item.chunks]))
        by_id = {chunk.chunk_id: float(score) for chunk, score in zip(item.chunks, values)}
        for chunk, score in zip(item.chunks, values):
            if chunk.label == "ambiguous":
                continue
            labels.append(1 if chunk.label == "positive" else 0)
            scores.append(float(score))
        acc_all, n_all = pairwise_pos_gt_hn(
            by_id,
            [chunk.chunk_id for chunk in item.positives],
            [chunk.chunk_id for chunk in item.hard_negatives],
        )
        clean_ids = [
            chunk.chunk_id
            for chunk in item.hard_negatives
            if not flags.get((item.question_id, chunk.chunk_id), {}).get("hn_gold_string_warning")
        ]
        acc_clean, n_clean = pairwise_pos_gt_hn(
            by_id,
            [chunk.chunk_id for chunk in item.positives],
            clean_ids,
        )
        if n_all:
            pairwise_all.append(acc_all)
        if n_clean:
            pairwise_clean.append(acc_clean)
    return {
        "pr_auc": binary_pr_auc(labels, scores) if labels else float("nan"),
        "pairwise_all_hn": float(sum(pairwise_all) / len(pairwise_all)) if pairwise_all else float("nan"),
        "pairwise_clean_hn": float(sum(pairwise_clean) / len(pairwise_clean)) if pairwise_clean else float("nan"),
        "n_scored": len(labels),
        "threshold": best_threshold(labels, scores) if labels else {"threshold": 0.5, "f1": 0.0},
    }


class FrozenAnswerability:
    def __init__(self, predict_fn) -> None:
        self._predict = predict_fn

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return list(self._predict(pairs))

    def score_instance(self, instance: RuntimeInstance) -> list[float]:
        return self.predict([(instance.question, chunk.text) for chunk in instance.chunks])


def train_b2_variant(
    train_records: Sequence[InstanceRecord],
    calibration_records: Sequence[InstanceRecord],
    flags_rows: Sequence[dict[str, Any]],
    *,
    variant: str,
    seed: int,
    device: str,
    model_name: str,
    output_dir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    ratio: tuple[int, int, int],
    smoke: bool,
    progress=None,
) -> dict[str, Any]:
    flags = flag_index(flags_rows)
    rng_seed = int(seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    if smoke:
        weighted, report = unique_question_pairs(train_records, flags, ratio=ratio)
        scorer = TfidfAnswerabilityScorer().fit([row.pair for row in weighted])
        metrics = evaluate_answerability(calibration_records, scorer, flags)
        scorer.save(output_dir)
        write_json(output_dir / "training_report.json", {"variant": variant, "seed": seed, "balance": report, "calibration": metrics, "smoke": True})
        return {"scorer": scorer, "calibration": metrics, "variant": variant, "seed": seed, "balance": report}

    import torch
    from torch.optim import AdamW

    torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)
    model = AnswerabilityModel(model_name, device=device, max_length=max_length)
    optimizer = AdamW(model.model.parameters(), lr=float(learning_rate))
    best_state = None
    best_metric = float("-inf")
    best_epoch = 0
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.model.train()
        running = 0.0
        steps = 0
        if variant == "b2a":
            weighted, report = unique_question_pairs(train_records, flags, ratio=ratio)
            pairs = list(weighted)
            random.Random(rng_seed + epoch).shuffle(pairs)
            for start in range(0, len(pairs), int(batch_size)):
                batch = pairs[start : start + int(batch_size)]
                loss = _bce_weighted(model, batch, device=device, max_length=max_length)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += float(loss.item())
                steps += 1
        elif variant == "b2b":
            epoch_pairs, usage = sample_dynamic_epoch(train_records, flags, ratio=ratio, seed=seed, epoch=epoch)
            report = {"unique_hn_used": len(usage), "usage_max": max(usage.values()) if usage else 0}
            for start in range(0, len(epoch_pairs), int(batch_size)):
                batch = epoch_pairs[start : start + int(batch_size)]
                loss = _bce_pairs(model, batch, device=device, max_length=max_length)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += float(loss.item())
                steps += 1
        elif variant == "b2c":
            triples = sample_pairwise_epoch(train_records, flags, seed=seed, epoch=epoch)
            report = {"n_pairs": len(triples)}
            for start in range(0, len(triples), max(1, int(batch_size) // 2)):
                batch = triples[start : start + max(1, int(batch_size) // 2)]
                loss = _pairwise_loss(model, batch, device=device, max_length=max_length)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += float(loss.item())
                steps += 1
        else:
            raise ValueError(f"unknown B2 variant {variant}")
        metrics = evaluate_answerability(calibration_records, model, flags)
        metric = float(metrics["pairwise_clean_hn"]) if metrics["pairwise_clean_hn"] == metrics["pairwise_clean_hn"] else -1.0
        pr = float(metrics["pr_auc"]) if metrics["pr_auc"] == metrics["pr_auc"] else -1.0
        history.append({"epoch": epoch, "train_loss": running / max(steps, 1), "calibration": metrics})
        if progress:
            progress.log(
                f"B2 {variant} seed={seed} epoch {epoch}/{epochs} loss={running / max(steps, 1):.4f} "
                f"cal_pairwise_clean={metric:.4f} pr_auc={pr:.4f}"
            )
        score = (metric, pr)
        if score > (best_metric if isinstance(best_metric, tuple) else (best_metric, -1.0)):
            best_metric = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
        if epoch - best_epoch >= int(patience):
            if progress:
                progress.log(f"B2 {variant} seed={seed} early stop at epoch {epoch}, best={best_epoch}")
            break
    if best_state is not None:
        model.model.load_state_dict(best_state)
    model.save(output_dir)
    cal = evaluate_answerability(calibration_records, model, flags)
    payload = {
        "variant": variant,
        "seed": seed,
        "best_epoch": best_epoch,
        "history": history,
        "calibration": cal,
        "balance": report,
    }
    write_json(output_dir / "training_report.json", payload)
    return {"scorer": model, "calibration": cal, "variant": variant, "seed": seed, "best_epoch": best_epoch}


def _encode_batch(model: AnswerabilityModel, questions, texts, device, max_length):
    encoded = model.tokenizer(
        list(questions),
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _bce_weighted(model, batch: Sequence[WeightedPair], *, device: str, max_length: int):
    import torch

    encoded = _encode_batch(model, [row.pair.question for row in batch], [row.pair.text for row in batch], device, max_length)
    logits = model.model(**encoded).logits.view(-1)
    labels = torch.tensor([row.pair.label for row in batch], device=device, dtype=torch.float32)
    weights = torch.tensor([row.weight for row in batch], device=device, dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weights)
    return loss


def _bce_pairs(model, batch: Sequence[TrainPair], *, device: str, max_length: int):
    import torch

    encoded = _encode_batch(model, [row.question for row in batch], [row.text for row in batch], device, max_length)
    logits = model.model(**encoded).logits.view(-1)
    labels = torch.tensor([row.label for row in batch], device=device, dtype=torch.float32)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)


def _pairwise_loss(model, batch: Sequence[tuple[TrainPair, TrainPair]], *, device: str, max_length: int):
    import torch

    questions = [left.question for left, _ in batch] + [right.question for _, right in batch]
    texts = [left.text for left, _ in batch] + [right.text for _, right in batch]
    encoded = _encode_batch(model, questions, texts, device, max_length)
    logits = model.model(**encoded).logits.view(-1)
    half = len(batch)
    pos, neg = logits[:half], logits[half:]
    return -torch.nn.functional.logsigmoid(pos - neg).mean()


def load_b2(path: Path, device: str, smoke: bool = False):
    wrapper = path / "wrapper.json"
    if not wrapper.exists():
        return None
    import json

    kind = json.loads(wrapper.read_text(encoding="utf-8"))
    if kind.get("kind") == "tfidf_logistic":
        return TfidfAnswerabilityScorer.load(path)
    if smoke:
        return TfidfAnswerabilityScorer.load(path)
    return AnswerabilityModel.load(path, device=device)
