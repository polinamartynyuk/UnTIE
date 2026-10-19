"""Audited loader and batched decoder for a frozen ModernBERT-SQuAD2 reader.

The selected community checkpoint stores weights for a tiny custom wrapper but
its remote-code constructor is incompatible with the project's Transformers
version.  The wrapper is reproduced here and the state dict is loaded strictly.
No code from the model repository is executed.
"""
from __future__ import annotations

from pathlib import Path


class FrozenSquadReader:
    def __init__(self, snapshot: Path, device: str, *, base_model: str, max_length: int = 512,
                 batch_size: int = 8, max_answer_tokens: int = 30):
        import torch
        from safetensors.torch import load_file
        from torch import nn
        from transformers import AutoModel, AutoTokenizer

        snapshot = Path(snapshot)
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = AutoModel.from_pretrained(base_model, local_files_only=True, trust_remote_code=False)
                self.qa_outputs = nn.Linear(self.base.config.hidden_size, 2)

            def forward(self, input_ids=None, attention_mask=None):
                hidden = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                logits = self.qa_outputs(hidden)
                return logits[..., 0], logits[..., 1]

        self.model = _Model()
        incompatible = self.model.load_state_dict(load_file(str(snapshot / "model.safetensors")), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"SQuAD2 checkpoint mismatch: {incompatible}")
        self.model.to(device).eval()
        self.device = device
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.max_answer_tokens = int(max_answer_tokens)
        self.torch = torch

    def answer_chunks(self, question: str, chunks) -> dict[int, dict]:
        """Return the best extractive span and raw-logit confidence per chunk."""
        output: dict[int, dict] = {}
        values = list(chunks)
        with self.torch.inference_mode():
            for offset in range(0, len(values), self.batch_size):
                batch = values[offset: offset + self.batch_size]
                encoded = self.tokenizer(
                    [question] * len(batch), [chunk.text for chunk in batch],
                    padding=True, truncation="only_second", max_length=self.max_length,
                    return_offsets_mapping=True, return_tensors="pt",
                )
                offsets = encoded.pop("offset_mapping")
                sequence_ids = [encoded.sequence_ids(index) for index in range(len(batch))]
                start_logits, end_logits = self.model(**{key: value.to(self.device) for key, value in encoded.items()})
                start_logits = start_logits.float().cpu()
                end_logits = end_logits.float().cpu()
                for index, chunk in enumerate(batch):
                    valid = [position for position, sid in enumerate(sequence_ids[index]) if sid == 1]
                    candidates = []
                    for start in sorted(valid, key=lambda pos: float(start_logits[index, pos]), reverse=True)[:20]:
                        endings = [pos for pos in valid if start <= pos <= start + self.max_answer_tokens]
                        for end in sorted(endings, key=lambda pos: float(end_logits[index, pos]), reverse=True)[:20]:
                            candidates.append((float(start_logits[index, start] + end_logits[index, end]), start, end))
                    if not candidates:
                        output[int(chunk.chunk_id)] = {"answer": "", "confidence": -1e30, "start": 0, "end": 0}
                        continue
                    confidence, start, end = max(candidates)
                    left = int(offsets[index, start, 0]); right = int(offsets[index, end, 1])
                    output[int(chunk.chunk_id)] = {
                        "answer": chunk.text[left:right], "confidence": confidence,
                        "start": left, "end": right,
                    }
        return output

