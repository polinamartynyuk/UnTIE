"""V3: TRAIN-only reader-utility positives + weighted/marginalized ranker."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments.qa_chunk_selection.metrics import token_f1
from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v2.hn_flags import is_training_eligible_hn
from experiments.qa_chunk_selection.v3.roles import labeled_candidates
from experiments.qa_chunk_selection.v3.train import (
    TrainPair,
    _ensure_pair_list,
    _listwise_loss,
    evaluate_answerability,
    train_bce_v3,
)
from experiments.qa_chunk_selection.v2.answerability_v2 import TfidfAnswerabilityScorer
from experiments.qa_chunk_selection.v2.io_util import write_json
from experiments.qa_chunk_selection.v3.roles import overlay_index

from .qa_shortlist import answer_one


def mine_utility_positives(
    train_records: Sequence[InstanceRecord],
    answer_fn: Callable[[str, str], dict[str, Any]],
    *,
    cache,
    model_id: str,
    inference_hash: str,
    f1_threshold: float,
    progress=None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = sum(1 for item in train_records for chunk in item.chunks if chunk.label == "ambiguous")
    current = 0
    for item in train_records:
        gold = list(item.gold_answers)
        pos_spans = {(chunk.char_start, chunk.char_end) for chunk in item.positives}
        for chunk in item.chunks:
            if chunk.label != "ambiguous":
                continue
            current += 1
            if progress:
                progress.count(current, max(total, 1), item="utility_qa")
            qa = answer_one(
                item,
                chunk.chunk_id,
                answer_fn,
                cache=cache,
                model_id=model_id,
                inference_hash=inference_hash,
            )
            f1 = float(token_f1(str(qa.get("qa_answer") or ""), gold))
            overlap_pos = any(
                not (chunk.char_end <= start or chunk.char_start >= end) for start, end in pos_spans
            )
            if f1 < float(f1_threshold):
                continue
            rows.append(
                {
                    "question_id": item.question_id,
                    "document_id": item.document_id,
                    "chunk_id": int(chunk.chunk_id),
                    "token_f1_vs_train_gold": f1,
                    "overlaps_original_positive": overlap_pos,
                    "role": "utility_positive_candidate",
                }
            )
    return rows


def utility_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["question_id"]), int(row["chunk_id"])): dict(row) for row in rows}


def listwise_groups_utility(
    records: Sequence[InstanceRecord],
    flags,
    utility: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    easy_cap: int,
    eta: float,
) -> list[dict[str, Any]]:
    groups = []
    for item in records:
        labeled = labeled_candidates(item, {}, flags, easy_cap=easy_cap)
        texts = []
        chunk_ids = []
        is_pos = []
        weights = []
        for chunk, role in labeled:
            texts.append(chunk.text)
            chunk_ids.append(chunk.chunk_id)
            is_pos.append(role == "positive")
            weights.append(1.0 if role == "positive" else 0.0)
        for chunk in item.chunks:
            key = (item.question_id, int(chunk.chunk_id))
            if key not in utility:
                continue
            if int(chunk.chunk_id) in chunk_ids:
                continue
            texts.append(chunk.text)
            chunk_ids.append(chunk.chunk_id)
            is_pos.append(True)
            weights.append(float(eta))
        if not any(is_pos) or len(texts) < 2:
            continue
        groups.append(
            {
                "question": item.question,
                "document_id": item.document_id,
                "question_id": item.question_id,
                "texts": texts,
                "chunk_ids": chunk_ids,
                "is_positive": is_pos,
                "weights": weights,
            }
        )
    return groups


def _weighted_listwise_loss(model, group, *, device: str, max_length: int, temperature: float):
    import torch

    encoded = __import__("experiments.qa_chunk_selection.v2.answerability_v2", fromlist=["_encode_batch"])._encode_batch(
        model, [group["question"]] * len(group["texts"]), group["texts"], device, max_length
    )
    logits = model.model(**encoded).logits.view(-1) / float(temperature)
    weights = torch.tensor(group["weights"], device=device, dtype=logits.dtype)
    strong = torch.tensor(group["is_positive"], device=device, dtype=torch.bool)
    # original positives keep weight 1; utility uses eta already stored in weights
    pos_logits = logits + torch.log(torch.clamp(weights, min=1e-6))
    mask = weights > 0
    if not bool(mask.any()):
        return _listwise_loss(model, group, device=device, max_length=max_length, temperature=temperature)
    log_num = torch.logsumexp(pos_logits[mask], dim=0)
    log_den = torch.logsumexp(logits, dim=0)
    del strong
    return -(log_num - log_den)


def train_utility_listwise(
    train_records,
    calibration_records,
    flags_rows,
    utility,
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
    eta: float,
    smoke: bool,
    progress=None,
) -> dict[str, Any]:
    from experiments.qa_chunk_selection.v2.hn_flags import flag_index
    from experiments.qa_chunk_selection.answerability import AnswerabilityModel
    import random
    import numpy as np

    from experiments.qa_chunk_selection.v4.keybert_features import resolve_encoder_device

    flags = flag_index(flags_rows)
    device = resolve_encoder_device(device)
    groups = listwise_groups_utility(train_records, flags, utility, easy_cap=easy_cap, eta=eta)
    rng_seed = int(seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        pairs = []
        for group in groups:
            for text, chunk_id, is_pos in zip(group["texts"], group["chunk_ids"], group["is_positive"]):
                pairs.append(
                    TrainPair(group["question"], text, int(is_pos), group["document_id"], group["question_id"], chunk_id, "pos" if is_pos else "neg")
                )
        scorer = TfidfAnswerabilityScorer().fit(
            _ensure_pair_list(
                pairs
                or [
                    TrainPair("q", "yes", 1, "d", "q", 0, "positive"),
                    TrainPair("q", "no", 0, "d", "q", 1, "hard_negative"),
                ]
            )
        )
        metrics = evaluate_answerability(calibration_records, scorer, flags)
        scorer.save(output_dir)
        write_json(output_dir / "training_report.json", {"tag": "v3_utility", "smoke": True, "n_groups": len(groups), "calibration": metrics})
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
            loss = _weighted_listwise_loss(model, group, device=device, max_length=max_length, temperature=temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1
        metrics = evaluate_answerability(calibration_records, model, flags)
        metric = float(metrics["pairwise_clean_hn"]) if metrics["pairwise_clean_hn"] == metrics["pairwise_clean_hn"] else -1.0
        history.append({"epoch": epoch, "train_loss": running / max(steps, 1), "calibration": metrics})
        if progress:
            progress.log(f"v3_utility seed={seed} epoch {epoch}/{epochs} loss={running / max(steps, 1):.4f} cal_pairwise={metric:.4f}")
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
        {"tag": "v3_utility", "seed": seed, "best_epoch": best_epoch, "history": history, "calibration": cal, "eta": eta, "n_groups": len(groups)},
    )
    return {"scorer": model, "calibration": cal, "seed": seed, "best_epoch": best_epoch}


def train_utility_bce(
    train_records,
    calibration_records,
    flags_rows,
    utility,
    **kwargs,
) -> dict[str, Any]:
    overlay_rows = [
        {"question_id": qid, "chunk_id": cid, "v3_target_status": "target_evidence", "v3_gold_relation": "matches_existing_gold"}
        for (qid, cid) in utility
    ]
    # Weak extra positives via overlay; they are utility_positive_candidate, not TE claims in reports.
    overlay = overlay_index(overlay_rows)
    return train_bce_v3(train_records, calibration_records, overlay, flags_rows, tag="v3_utility_bce", **kwargs)
