"""V7: RECONSIDER-inspired second-stage reranker on B0 top-K plus optional anchored synthetics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.answerability import AnswerabilityModel, TfidfAnswerabilityScorer, TrainPair
from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v2.answerability_v2 import WeightedPair, _bce_weighted, evaluate_answerability
from experiments.qa_chunk_selection.v3.train import _ensure_pair_list, score_fn_for
from experiments.qa_chunk_selection.v2.hn_flags import flag_index, is_training_eligible_hn
from experiments.qa_chunk_selection.v2.io_util import write_json
from .shortlist import top_k_ids
from .v4_ownership import anchor_prefix


def build_v7_pairs(
    records: Sequence[InstanceRecord],
    b0_by_qid: Mapping[str, Mapping[str, Any]],
    flags,
    synthetics: Sequence[Mapping[str, Any]] | None,
    *,
    k: int,
    easy_cap: int,
    use_anchor: bool,
    profiles: Mapping[str, Sequence[str]] | None = None,
) -> list[Any]:
    pairs: list[WeightedPair] = []
    for item in records:
        row = b0_by_qid.get(item.question_id)
        if row is None:
            continue
        ranked = top_k_ids(row, k)
        prefix = anchor_prefix(profiles.get(item.document_id) or []) if use_anchor and profiles else ""
        question = prefix + item.question if prefix else item.question
        pos_ids = {int(chunk.chunk_id) for chunk in item.positives}
        easy_used = 0
        for cid in ranked:
            chunk = item.chunk_by_id(int(cid))
            if int(cid) in pos_ids:
                pairs.append(WeightedPair(TrainPair(question, chunk.text, 1, item.document_id, item.question_id, cid, "positive"), 1.0))
            elif chunk.label == "hard_negative" and (not flags or is_training_eligible_hn(flags, item.question_id, int(cid))):
                pairs.append(WeightedPair(TrainPair(question, chunk.text, 0, item.document_id, item.question_id, cid, "hard_negative"), 1.0))
            elif chunk.label == "easy_negative" and easy_used < int(easy_cap):
                pairs.append(WeightedPair(TrainPair(question, chunk.text, 0, item.document_id, item.question_id, cid, "easy_negative"), 0.5))
                easy_used += 1
    for row in synthetics or []:
        if not use_anchor:
            continue
        prefix = anchor_prefix(list(row.get("anchor_phrases") or []))
        pairs.append(
            WeightedPair(
                TrainPair(
                    prefix + str(row["question"]),
                    str(row["chunk_text"]),
                    0,
                    str(row["target_document_id"]),
                    str(row["question_id"]),
                    int(row["chunk_id"]),
                    "synthetic_wrong_document",
                ),
                0.75,
            )
        )
    return pairs


def train_v7(
    train_records,
    calibration_records,
    flags_rows,
    b0_by_qid,
    synthetics,
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
    k: int,
    easy_cap: int,
    use_anchor: bool,
    profiles,
    smoke: bool,
    progress=None,
) -> dict[str, Any]:
    import random
    import numpy as np

    from experiments.qa_chunk_selection.v4.keybert_features import resolve_encoder_device

    flags = flag_index(flags_rows)
    device = resolve_encoder_device(device)
    pairs = build_v7_pairs(train_records, b0_by_qid, flags, synthetics, k=k, easy_cap=easy_cap, use_anchor=use_anchor, profiles=profiles)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng_seed = int(seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    if smoke or not pairs:
        scorer = TfidfAnswerabilityScorer().fit(_ensure_pair_list([row.pair for row in pairs] or [
            TrainPair("q", "yes", 1, "d", "q", 0, "positive"),
            TrainPair("q", "no", 0, "d", "q", 1, "hard_negative"),
        ]))
        metrics = evaluate_answerability(calibration_records, scorer, flags)
        scorer.save(output_dir)
        write_json(output_dir / "training_report.json", {"tag": "v7", "smoke": True, "n_pairs": len(pairs), "calibration": metrics})
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
        history.append({"epoch": epoch, "train_loss": running / max(steps, 1), "calibration": metrics})
        if progress:
            progress.log(f"v7 seed={seed} epoch {epoch}/{epochs} loss={running / max(steps, 1):.4f} cal_pairwise={metric:.4f}")
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
    write_json(output_dir / "training_report.json", {"tag": "v7", "seed": seed, "best_epoch": best_epoch, "history": history, "calibration": cal, "use_anchor": use_anchor})
    return {"scorer": model, "calibration": cal, "seed": seed}


def score_v7(model, item: InstanceRecord, *, use_anchor: bool, phrases: Sequence[str]):
    scorer = score_fn_for(model)
    runtime_question = (anchor_prefix(phrases) + item.question) if use_anchor else item.question
    from experiments.qa_chunk_selection.v2.schemas import to_runtime
    from dataclasses import replace

    runtime = to_runtime(item)
    runtime = replace(runtime, question=runtime_question)
    scores = scorer(runtime)
    return {int(chunk.chunk_id): float(score) for chunk, score in zip(item.chunks, scores)}
