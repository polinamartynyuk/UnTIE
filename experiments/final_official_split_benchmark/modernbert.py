"""Small, auditable extractive-QA fine-tuning loop for the EN-only B5 system."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from experiments.qa_chunk_selection.metrics import qa_row

from .io_guard import write_json


def _examples(records, engine, scorer):
    result = []
    for item in records:
        scores = engine.scores(scorer, item)
        top = sorted(item.chunks, key=lambda x: (-scores[x.chunk_id], x.chunk_id))[: engine.spec.k]
        spans = {}
        for span in item.gold_spans:
            spans.setdefault(int(span.chunk_id), []).append(span)
        for chunk in top:
            values = spans.get(chunk.chunk_id, [])
            if values:
                span = values[0]; start, end = int(span.start_char), int(span.end_char); status = "exact_offset"
                if chunk.text[start:end] != span.text:
                    located = chunk.text.casefold().find(str(span.text).casefold())
                    if located >= 0: start, end, status = located, located + len(span.text), "realigned_exact_text"
                    else: start = end = 0; status = "unmatched_positive_span"
            else:
                start = end = 0; status = "negative_topk_chunk"
            result.append({"question": item.question, "context": chunk.text, "answer_start": start, "answer_end": end,
                           "question_id": item.question_id, "gold": list(item.gold_answers), "chunk_id": chunk.chunk_id, "alignment_status": status})
    return result


class QADataset:
    def __init__(self, examples, tokenizer, max_length=512):
        self.rows = []
        for value in examples:
            encoded = tokenizer(value["question"], value["context"], truncation="only_second", max_length=max_length,
                                padding="max_length", return_offsets_mapping=True)
            offsets = encoded.pop("offset_mapping"); sequence = encoded.sequence_ids(); cls = encoded["input_ids"].index(tokenizer.cls_token_id)
            start = end = cls
            if value["answer_end"] > value["answer_start"]:
                candidates = [i for i, sid in enumerate(sequence) if sid == 1]
                for i in candidates:
                    if offsets[i][0] <= value["answer_start"] < offsets[i][1]: start = i; break
                for i in reversed(candidates):
                    if offsets[i][0] < value["answer_end"] <= offsets[i][1]: end = i; break
            encoded["start_positions"] = start; encoded["end_positions"] = end
            self.rows.append(encoded)
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        import torch
        return {k: torch.tensor(v) for k, v in self.rows[index].items()}


class ModernBertReader:
    def __init__(self, path, device, local_files_only=True):
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_files_only)
        self.model = AutoModelForQuestionAnswering.from_pretrained(path, local_files_only=local_files_only).to(device)
        self.device = device

    def answer(self, question, context, max_length=512):
        import torch
        encoded = self.tokenizer(question, context, truncation="only_second", max_length=max_length, return_offsets_mapping=True, return_tensors="pt")
        offsets = encoded.pop("offset_mapping")[0].tolist(); sequence = encoded.sequence_ids(0)
        try:
            with torch.inference_mode():
                output = self.model(**{k: v.to(self.device) for k, v in encoded.items()})
        except Exception as exc:
            return {"answer": "", "confidence": -1e30, "start": 0, "end": 0, "error": f"{type(exc).__name__}: {exc}"}
        start_logits = output.start_logits[0].float().cpu(); end_logits = output.end_logits[0].float().cpu()
        valid = [i for i, sid in enumerate(sequence) if sid == 1]
        candidates = []
        for start in sorted(valid, key=lambda i: float(start_logits[i]), reverse=True)[:20]:
            for end in sorted([i for i in valid if start <= i <= start + 30], key=lambda i: float(end_logits[i]), reverse=True)[:20]:
                candidates.append((float(start_logits[start] + end_logits[end]), start, end))
        if not candidates: return {"answer": "", "confidence": 0.0, "start": 0, "end": 0}
        score, start, end = max(candidates); left, right = offsets[start][0], offsets[end][1]
        return {"answer": context[left:right], "confidence": score, "start": left, "end": right}


def train_model(records, engine, scorer, *, checkpoint: str, output: Path, seed: int, learning_rate: float, epochs: int,
                device: str, allow_downloads: bool, progress, revision: str | None = None) -> dict:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    kwargs = {"local_files_only": not allow_downloads, "trust_remote_code": False}
    if revision: kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **kwargs)
    model = AutoModelForQuestionAnswering.from_pretrained(checkpoint, **kwargs).to(device)
    examples = _examples(records, engine, scorer); dataset = QADataset(examples, tokenizer); loader = DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate); model.train(); history = []
    for epoch in range(1, epochs + 1):
        total = 0.0
        for step, batch in enumerate(loader, 1):
            output_value = model(**{k: v.to(device) for k, v in batch.items()}); loss = output_value.loss / 8
            loss.backward()
            if step % 8 == 0 or step == len(loader): optimizer.step(); optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach().cpu()) * 8
            progress.count(step, len(loader), item=f"ModernBERT seed={seed} epoch={epoch} batches")
        history.append({"epoch": epoch, "loss": total / max(len(loader), 1)})
    output.mkdir(parents=True, exist_ok=True); model.save_pretrained(output); tokenizer.save_pretrained(output)
    counts = {status: sum(x["alignment_status"] == status for x in examples) for status in sorted({x["alignment_status"] for x in examples})}
    report = {"checkpoint": checkpoint, "revision": revision, "seed": seed, "learning_rate": learning_rate, "epochs": epochs, "examples": len(examples),
              "supervision_alignment": counts, "supervision_sources": "current cell TRAIN only during development; TRAIN+VAL only after freeze",
              "test_gold_used": False, "history": history}
    write_json(output / "training_report.json", report); return report


def evaluate_model(records, engine, scorer, model_dir: Path) -> dict:
    reader = ModernBertReader(model_dir, engine.device); values = []
    for item in records:
        scores = engine.scores(scorer, item); chunks = sorted(item.chunks, key=lambda x: (-scores[x.chunk_id], x.chunk_id))[: engine.spec.k]
        answers = [reader.answer(item.question, chunk.text) for chunk in chunks]
        selected = max(answers, key=lambda x: x["confidence"], default={"answer": ""})
        values.append(qa_row(selected["answer"], item.gold_answers)["token_f1"])
    return {"questions": len(values), "token_f1": float(np.mean(values)) if values else 0.0}
