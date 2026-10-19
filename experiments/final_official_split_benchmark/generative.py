"""Frozen deterministic Qwen baseline."""
from __future__ import annotations

from pathlib import Path

from .contract import QWEN_PROMPTS
from .data import document_text


class QwenBaseline:
    def __init__(self, model_id: str, revision: str | None, device: str, allow_downloads: bool, progress=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        source = model_id
        kwargs = {"local_files_only": not allow_downloads, "trust_remote_code": False}
        if not Path(model_id).is_dir() and revision:
            try:
                from huggingface_hub import snapshot_download
                cached = Path(snapshot_download(repo_id=model_id, revision=revision, local_files_only=True))
                required = (cached / "config.json", cached / "tokenizer.json")
                weights = (cached / "model.safetensors").is_file() or (cached / "model.safetensors.index.json").is_file()
                if all(path.is_file() for path in required) and weights:
                    source = str(cached); kwargs["local_files_only"] = True
            except Exception:
                pass
        if source == model_id and revision: kwargs["revision"] = revision
        if progress is not None:
            progress.log("loading frozen Qwen baseline", source="local_snapshot" if source != model_id or Path(model_id).is_dir() else "model_hub", revision=revision)
        self.tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        self.tokenizer.truncation_side = "right"
        dtype = torch.bfloat16 if str(device).startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype, **kwargs).to(device)
        self.device = device
        if progress is not None: progress.log("frozen Qwen baseline ready", device=str(device), source=str(source))

    def answer(self, item, language: str) -> dict:
        import torch
        prompt = QWEN_PROMPTS[language].format(question=item.question, document=document_text(item))
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=min(int(getattr(self.tokenizer, "model_max_length", 8192)), 8192))
        input_length = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self.model.generate(**{k: v.to(self.device) for k, v in encoded.items()}, do_sample=False, max_new_tokens=64, pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(generated[0, input_length:], skip_special_tokens=True).strip().splitlines()[0].strip()
        if text in {"NO_ANSWER", "НЕТ_ОТВЕТА"}: text = ""
        return {"answer": text, "prompt_tokens": int(input_length), "output_tokens": int(generated.shape[1] - input_length)}
