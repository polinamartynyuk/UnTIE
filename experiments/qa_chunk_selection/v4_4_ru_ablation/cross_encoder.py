"""Cross-encoder B0 ablation with a one-logit head and bounded checkpoint storage."""

import gc
import random
from pathlib import Path

import numpy as np

from experiments.qa_chunk_selection.answerability import TfidfAnswerabilityScorer
from experiments.qa_chunk_selection.v2.answerability_v2 import (
    _bce_weighted, evaluate_answerability, unique_question_pairs,
)
from experiments.qa_chunk_selection.v2.hn_flags import flag_index
from experiments.qa_chunk_selection.io_util import write_json


class FlexibleCrossEncoder:
    def __init__(self, model_name, device, max_length):
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        config = AutoConfig.from_pretrained(model_name)
        old_labels = int(getattr(config, "num_labels", 0) or 0)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, ignore_mismatched_sizes=old_labels != 1,
        ).to(device)
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        self.model_name, self.device, self.max_length = model_name, device, int(max_length)
        self.initialization = {
            "checkpoint_num_labels": old_labels,
            "trained_num_labels": 1,
            "classification_head_reinitialized": old_labels != 1,
            "backbone_weights_preserved": True,
        }

    def _logits(self, questions, passages):
        encoded = self.tokenizer(list(questions), list(passages), padding=True,
                                 truncation=True, max_length=self.max_length,
                                 return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        return self.model(**encoded).logits.view(-1)

    def predict(self, pairs):
        if not pairs:
            return []
        import torch
        self.model.eval()
        out = []
        with torch.inference_mode():
            for start in range(0, len(pairs), 8):
                batch = pairs[start:start + 8]
                logits = self._logits([x[0] for x in batch], [x[1] for x in batch])
                out.extend(torch.sigmoid(logits).float().cpu().tolist())
        return [float(x) for x in out]


def train_cross_encoder(train_records, calibration_records, flags_rows, *, model_name,
                        seed, device, epochs, patience, micro_batch_size,
                        gradient_accumulation, learning_rate, max_length, progress,
                        report_path, smoke=False):
    flags = flag_index(flags_rows)
    weighted, balance = unique_question_pairs(train_records, flags, ratio=(1, 3, 1))
    if smoke:
        scorer = TfidfAnswerabilityScorer().fit([row.pair for row in weighted])
        report = {"smoke": True, "balance": balance,
                  "calibration": evaluate_answerability(calibration_records, scorer, flags)}
        write_json(report_path, report)
        return scorer, report

    import torch
    from torch.optim import AdamW
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    scorer = FlexibleCrossEncoder(model_name, device, max_length)
    optimizer = AdamW(scorer.model.parameters(), lr=float(learning_rate))
    use_amp = str(device).startswith("cuda")
    use_bf16 = bool(use_amp and torch.cuda.is_bf16_supported())
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and not use_bf16))
    best_state, best_epoch, best_score, history = None, 0, (-1.0, -1.0), []
    for epoch in range(1, int(epochs) + 1):
        rows = list(weighted)
        random.Random(seed + epoch).shuffle(rows)
        scorer.model.train(); optimizer.zero_grad(set_to_none=True)
        running, steps = 0.0, 0
        for start in range(0, len(rows), int(micro_batch_size)):
            batch = rows[start:start + int(micro_batch_size)]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16,
                                enabled=use_amp):
                loss = _bce_weighted(scorer, batch, device=device, max_length=max_length)
                scaled_loss = loss / int(gradient_accumulation)
            scaler.scale(scaled_loss).backward()
            boundary = ((steps + 1) % int(gradient_accumulation) == 0 or
                        start + int(micro_batch_size) >= len(rows))
            if boundary:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach().cpu()); steps += 1
            progress.count(steps, (len(rows) + micro_batch_size - 1) // micro_batch_size,
                           item=f"{model_name} seed={seed} epoch={epoch} batches")
        metrics = evaluate_answerability(calibration_records, scorer, flags)
        clean = float(metrics["pairwise_clean_hn"])
        pr_auc = float(metrics["pr_auc"])
        score = (clean if np.isfinite(clean) else -1.0, pr_auc if np.isfinite(pr_auc) else -1.0)
        history.append({"epoch": epoch, "loss": running / max(steps, 1), "calibration": metrics})
        progress.log(f"cross-encoder epoch {epoch}/{epochs}", model=model_name, seed=seed,
                     loss=running / max(steps, 1), pairwise_clean=clean, pr_auc=pr_auc)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in scorer.model.state_dict().items()}
        if epoch - best_epoch >= int(patience):
            progress.log("cross-encoder early stop", model=model_name, seed=seed, best_epoch=best_epoch)
            break
    if best_state is not None:
        scorer.model.load_state_dict(best_state)
    final = evaluate_answerability(calibration_records, scorer, flags)
    report = {"model": model_name, "seed": seed, "best_epoch": best_epoch,
              "initialization": scorer.initialization, "precision": "bf16" if use_bf16 else ("fp16" if use_amp else "fp32"),
              "effective_batch_size": int(micro_batch_size) * int(gradient_accumulation),
              "balance": balance, "history": history, "calibration": final,
              "checkpoint_retained": False,
              "note": "Weights are not retained; OOF predictions are the ablation artifact."}
    write_json(report_path, report)
    del best_state
    gc.collect()
    return scorer, report

