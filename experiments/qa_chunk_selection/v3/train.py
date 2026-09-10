"""V3-B0/B1/B2/B3 training. B0 is the V2-winning b2a recipe inside each CV fold."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.qa_chunk_selection.answerability import AnswerabilityModel, TfidfAnswerabilityScorer, TrainPair
from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v2.answerability_v2 import (
    FrozenAnswerability,
    WeightedPair,
    _bce_weighted,
    _encode_batch,
    _pairwise_loss,
    evaluate_answerability,
    load_b2,
    train_b2_variant,
)
from experiments.qa_chunk_selection.v2.hn_flags import flag_index
from experiments.qa_chunk_selection.v2.io_util import write_json
from experiments.qa_chunk_selection.v2.schemas import RuntimeInstance

from .roles import labeled_candidates, v3_target_positives

load_ranker = load_b2


def _ensure_both_classes(weighted: Sequence[WeightedPair]) -> list[WeightedPair]:
    rows = list(weighted)
    labels = {row.pair.label for row in rows}
    if rows and labels != {0, 1}:
        donor = rows[0]
        other_label = 0 if 1 in labels else 1
        rows.append(
            WeightedPair(
                TrainPair(
                    donor.pair.question,
                    donor.pair.text + " [NEG]" if other_label == 0 else donor.pair.text,
                    other_label,
                    donor.pair.document_id,
                    donor.pair.question_id,
                    donor.pair.chunk_id,
                    "synthetic_balance",
                ),
                donor.weight,
            )
        )
    return rows


def _ensure_pair_list(pairs: Sequence[TrainPair]) -> list[TrainPair]:
    rows = list(pairs)
    labels = {row.label for row in rows}
    if rows and labels != {0, 1}:
        donor = rows[0]
        other = 0 if 1 in labels else 1
        rows.append(
            TrainPair(
                donor.question,
                donor.text + " [NEG]" if other == 0 else donor.text,
                other,
                donor.document_id,
                donor.question_id,
                donor.chunk_id,
                "synthetic_balance",
            )
        )
    return rows


def unique_question_pairs_v3(
    records: Sequence[InstanceRecord],
    overlay,
    flags,
    *,
    ratio: tuple[int, int, int],
    easy_cap: int,
) -> tuple[list[WeightedPair], dict[str, Any]]:
    pos_n, hn_n, easy_n = ratio
    tm_n = max(hn_n, 1)
    total_w = float(pos_n + hn_n + tm_n + easy_n) or 1.0
    w_pos, w_hn, w_tm, w_easy = pos_n / total_w, hn_n / total_w, tm_n / total_w, easy_n / total_w
    questions = [item for item in records if v3_target_positives(item, overlay)]
    n_q = max(len(questions), 1)
    selected: list[WeightedPair] = []
    counts = defaultdict(int)
    for item in questions:
        by_role: dict[str, list] = defaultdict(list)
        for chunk, role in labeled_candidates(item, overlay, flags, easy_cap=easy_cap):
            by_role[role].append(chunk)
        classes = []
        if by_role["positive"]:
            classes.append((by_role["positive"], 1, "positive", w_pos))
        if by_role["hard_negative"]:
            classes.append((by_role["hard_negative"], 0, "hard_negative", w_hn))
        if by_role["target_mismatch"]:
            classes.append((by_role["target_mismatch"], 0, "target_mismatch", w_tm))
        if by_role["easy_negative"]:
            classes.append((by_role["easy_negative"], 0, "easy_negative", w_easy))
        share_sum = sum(share for *_rest, share in classes) or 1.0
        q_weight = 1.0 / n_q
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
                counts[kind] += 1
    report = {
        "n_questions": n_q,
        "n_pairs": len(selected),
        "counts": dict(counts),
        "weight_formula": "question-balanced weighted BCE; unique pairs; overlay roles; no HN oversampling",
    }
    return selected, report


def sample_pairwise_v3(
    records: Sequence[InstanceRecord],
    overlay,
    flags,
    *,
    seed: int,
    epoch: int,
    p_tm: float,
    p_hn: float,
    p_easy: float,
    max_pairs_per_question: int = 8,
    easy_cap: int = 4,
) -> list[tuple[TrainPair, TrainPair]]:
    rng = random.Random(int(seed) * 9176 + int(epoch))
    probs = np.array([p_tm, p_hn, p_easy], dtype=float)
    if probs.sum() <= 0:
        probs = np.array([1.0, 1.0, 1.0])
    probs = probs / probs.sum()
    triples: list[tuple[TrainPair, TrainPair]] = []
    for item in records:
        by_role: dict[str, list] = defaultdict(list)
        for chunk, role in labeled_candidates(item, overlay, flags, easy_cap=easy_cap):
            by_role[role].append(chunk)
        pos = by_role["positive"]
        if not pos:
            continue
        available = []
        for name, key in (("target_mismatch", "target_mismatch"), ("hard_negative", "hard_negative"), ("easy_negative", "easy_negative")):
            if by_role[key]:
                available.append(name)
        if not available:
            continue
        n = min(max_pairs_per_question, max(len(pos), 1) * 2)
        for _ in range(n):
            kind = rng.choices(("target_mismatch", "hard_negative", "easy_negative"), weights=probs, k=1)[0]
            pool = by_role[kind] or by_role["hard_negative"] or by_role["target_mismatch"] or by_role["easy_negative"]
            if not pool:
                continue
            p = rng.choice(pos)
            nchunk = rng.choice(pool)
            triples.append(
                (
                    TrainPair(item.question, p.text, 1, item.document_id, item.question_id, p.chunk_id, "positive"),
                    TrainPair(item.question, nchunk.text, 0, item.document_id, item.question_id, nchunk.chunk_id, kind),
                )
            )
    rng.shuffle(triples)
    return triples


def listwise_groups(
    records: Sequence[InstanceRecord],
    overlay,
    flags,
    *,
    easy_cap: int,
) -> list[dict[str, Any]]:
    groups = []
    for item in records:
        labeled = labeled_candidates(item, overlay, flags, easy_cap=easy_cap)
        pos = [chunk for chunk, role in labeled if role == "positive"]
        if not pos or len(labeled) < 2:
            continue
        groups.append(
            {
                "question": item.question,
                "document_id": item.document_id,
                "question_id": item.question_id,
                "texts": [chunk.text for chunk, _role in labeled],
                "chunk_ids": [chunk.chunk_id for chunk, _role in labeled],
                "is_positive": [role == "positive" for _chunk, role in labeled],
            }
        )
    return groups


def train_bce_v3(
    train_records: Sequence[InstanceRecord],
    calibration_records: Sequence[InstanceRecord],
    overlay,
    flags_rows: Sequence[dict[str, Any]],
    *,
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
    easy_cap: int,
    smoke: bool,
    progress=None,
    tag: str = "b1",
) -> dict[str, Any]:
    flags = flag_index(flags_rows)
    rng_seed = int(seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    weighted, report = unique_question_pairs_v3(train_records, overlay, flags, ratio=ratio, easy_cap=easy_cap)
    weighted = _ensure_both_classes(weighted)
    if smoke:
        scorer = TfidfAnswerabilityScorer().fit([row.pair for row in weighted])
        metrics = evaluate_answerability(calibration_records, scorer, flags)
        scorer.save(output_dir)
        write_json(output_dir / "training_report.json", {"tag": tag, "seed": seed, "balance": report, "calibration": metrics, "smoke": True})
        return {"scorer": scorer, "calibration": metrics, "seed": seed, "balance": report}

    import torch
    from torch.optim import AdamW

    torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)
    model = AnswerabilityModel(model_name, device=device, max_length=max_length)
    optimizer = AdamW(model.model.parameters(), lr=float(learning_rate))
    best_state = None
    best_metric: tuple[float, float] = (float("-inf"), float("-inf"))
    best_epoch = 0
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.model.train()
        weighted, report = unique_question_pairs_v3(train_records, overlay, flags, ratio=ratio, easy_cap=easy_cap)
        pairs = list(weighted)
        random.Random(rng_seed + epoch).shuffle(pairs)
        running = 0.0
        steps = 0
        for start in range(0, len(pairs), int(batch_size)):
            batch = pairs[start : start + int(batch_size)]
            loss = _bce_weighted(model, batch, device=device, max_length=max_length)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1
        metrics = evaluate_answerability(calibration_records, model, flags)
        metric = float(metrics["pairwise_clean_hn"]) if metrics["pairwise_clean_hn"] == metrics["pairwise_clean_hn"] else -1.0
        pr = float(metrics["pr_auc"]) if metrics["pr_auc"] == metrics["pr_auc"] else -1.0
        history.append({"epoch": epoch, "train_loss": running / max(steps, 1), "calibration": metrics})
        if progress:
            progress.log(f"{tag} seed={seed} epoch {epoch}/{epochs} loss={running / max(steps, 1):.4f} cal_pairwise_clean={metric:.4f}")
        score = (metric, pr)
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
        if epoch - best_epoch >= int(patience):
            if progress:
                progress.log(f"{tag} seed={seed} early stop at epoch {epoch}, best={best_epoch}")
            break
    if best_state is not None:
        model.model.load_state_dict(best_state)
    model.save(output_dir)
    cal = evaluate_answerability(calibration_records, model, flags)
    payload = {"tag": tag, "seed": seed, "best_epoch": best_epoch, "history": history, "calibration": cal, "balance": report}
    write_json(output_dir / "training_report.json", payload)
    return {"scorer": model, "calibration": cal, "seed": seed, "best_epoch": best_epoch}


def train_pairwise_v3(
    train_records: Sequence[InstanceRecord],
    calibration_records: Sequence[InstanceRecord],
    overlay,
    flags_rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    device: str,
    model_name: str,
    output_dir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    p_tm: float,
    p_hn: float,
    p_easy: float,
    easy_cap: int,
    smoke: bool,
    progress=None,
) -> dict[str, Any]:
    flags = flag_index(flags_rows)
    rng_seed = int(seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    if smoke:
        weighted, report = unique_question_pairs_v3(
            train_records, overlay, flags, ratio=(1, 3, 1), easy_cap=easy_cap
        )
        scorer = TfidfAnswerabilityScorer().fit([row.pair for row in _ensure_both_classes(weighted)])
        metrics = evaluate_answerability(calibration_records, scorer, flags)
        scorer.save(output_dir)
        write_json(output_dir / "training_report.json", {"tag": "b2", "seed": seed, "balance": report, "calibration": metrics, "smoke": True})
        return {"scorer": scorer, "calibration": metrics, "seed": seed}

    import torch
    from torch.optim import AdamW

    torch.manual_seed(rng_seed)
    model = AnswerabilityModel(model_name, device=device, max_length=max_length)
    optimizer = AdamW(model.model.parameters(), lr=float(learning_rate))
    best_state = None
    best_metric = float("-inf")
    best_epoch = 0
    history = []
    half = max(1, int(batch_size) // 2)
    for epoch in range(1, int(epochs) + 1):
        model.model.train()
        triples = sample_pairwise_v3(
            train_records,
            overlay,
            flags,
            seed=seed,
            epoch=epoch,
            p_tm=p_tm,
            p_hn=p_hn,
            p_easy=p_easy,
            easy_cap=easy_cap,
        )
        running = 0.0
        steps = 0
        for start in range(0, len(triples), half):
            batch = triples[start : start + half]
            if not batch:
                continue
            loss = _pairwise_loss(model, batch, device=device, max_length=max_length)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1
        metrics = evaluate_answerability(calibration_records, model, flags)
        metric = float(metrics["pairwise_clean_hn"]) if metrics["pairwise_clean_hn"] == metrics["pairwise_clean_hn"] else -1.0
        history.append({"epoch": epoch, "train_loss": running / max(steps, 1), "calibration": metrics, "n_pairs": len(triples)})
        if progress:
            progress.log(f"b2 seed={seed} epoch {epoch}/{epochs} loss={running / max(steps, 1):.4f} cal_pairwise_clean={metric:.4f}")
        if metric >= best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
        if epoch - best_epoch >= int(patience):
            break
    if best_state is not None:
        model.model.load_state_dict(best_state)
    model.save(output_dir)
    cal = evaluate_answerability(calibration_records, model, flags)
    write_json(output_dir / "training_report.json", {"tag": "b2", "seed": seed, "best_epoch": best_epoch, "history": history, "calibration": cal})
    return {"scorer": model, "calibration": cal, "seed": seed, "best_epoch": best_epoch}


def _listwise_loss(model, group: dict[str, Any], *, device: str, max_length: int, temperature: float):
    import torch

    encoded = _encode_batch(model, [group["question"]] * len(group["texts"]), group["texts"], device, max_length)
    logits = model.model(**encoded).logits.view(-1) / float(temperature)
    pos = torch.tensor(group["is_positive"], device=device, dtype=torch.bool)
    log_den = torch.logsumexp(logits, dim=0)
    log_num = torch.logsumexp(logits[pos], dim=0)
    return -(log_num - log_den)


def train_listwise_v3(
    train_records: Sequence[InstanceRecord],
    calibration_records: Sequence[InstanceRecord],
    overlay,
    flags_rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    device: str,
    model_name: str,
    output_dir: Path,
    epochs: int,
    patience: int,
    learning_rate: float,
    max_length: int,
    temperature: float,
    easy_cap: int,
    smoke: bool,
    progress=None,
) -> dict[str, Any]:
    flags = flag_index(flags_rows)
    rng_seed = int(seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    groups = listwise_groups(train_records, overlay, flags, easy_cap=easy_cap)
    if smoke:
        pairs = []
        for group in groups:
            for text, chunk_id, is_pos in zip(group["texts"], group["chunk_ids"], group["is_positive"]):
                pairs.append(
                    TrainPair(group["question"], text, int(is_pos), group["document_id"], group["question_id"], chunk_id, "positive" if is_pos else "negative")
                )
        scorer = TfidfAnswerabilityScorer().fit(_ensure_pair_list(pairs))
        metrics = evaluate_answerability(calibration_records, scorer, flags)
        scorer.save(output_dir)
        write_json(output_dir / "training_report.json", {"tag": "b3", "seed": seed, "n_groups": len(groups), "calibration": metrics, "smoke": True})
        return {"scorer": scorer, "calibration": metrics, "seed": seed}

    import torch
    from torch.optim import AdamW

    torch.manual_seed(rng_seed)
    model = AnswerabilityModel(model_name, device=device, max_length=max_length)
    optimizer = AdamW(model.model.parameters(), lr=float(learning_rate))
    best_state = None
    best_metric = float("-inf")
    best_epoch = 0
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.model.train()
        random.Random(rng_seed + epoch).shuffle(groups)
        running = 0.0
        steps = 0
        for group in groups:
            loss = _listwise_loss(model, group, device=device, max_length=max_length, temperature=temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1
        metrics = evaluate_answerability(calibration_records, model, flags)
        metric = float(metrics["pairwise_clean_hn"]) if metrics["pairwise_clean_hn"] == metrics["pairwise_clean_hn"] else -1.0
        history.append({"epoch": epoch, "train_loss": running / max(steps, 1), "calibration": metrics})
        if progress:
            progress.log(f"b3 seed={seed} epoch {epoch}/{epochs} loss={running / max(steps, 1):.4f} cal_pairwise_clean={metric:.4f}")
        if metric >= best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
        if epoch - best_epoch >= int(patience):
            break
    if best_state is not None:
        model.model.load_state_dict(best_state)
    model.save(output_dir)
    cal = evaluate_answerability(calibration_records, model, flags)
    write_json(
        output_dir / "training_report.json",
        {"tag": "b3", "seed": seed, "best_epoch": best_epoch, "history": history, "calibration": cal, "temperature": temperature, "n_groups": len(groups)},
    )
    return {"scorer": model, "calibration": cal, "seed": seed, "best_epoch": best_epoch}


def train_b0(
    train_records,
    calibration_records,
    flags_rows,
    *,
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
    return train_b2_variant(
        train_records,
        calibration_records,
        flags_rows,
        variant="b2a",
        seed=seed,
        device=device,
        model_name=model_name,
        output_dir=output_dir,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        ratio=ratio,
        smoke=smoke,
        progress=progress,
    )


def as_scorer(model) -> FrozenAnswerability:
    return FrozenAnswerability(model.predict)


def score_fn_for(model):
    frozen = as_scorer(model)

    def score_fn(runtime: RuntimeInstance) -> list[float]:
        return frozen.score_instance(runtime)

    return score_fn
