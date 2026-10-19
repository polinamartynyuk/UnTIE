"""Canonical B0, frozen reader, and exact V7/V8 feature materialization."""
from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.qa_chunk_selection.answerability import AnswerabilityModel, TfidfAnswerabilityScorer
from experiments.qa_chunk_selection.schema import ChunkRecord, InstanceRecord
from experiments.qa_chunk_selection.v2.schemas import RuntimeInstance
from experiments.qa_chunk_selection.v3.train import score_fn_for, train_b0 as train_legacy_b0
from experiments.qa_chunk_selection.v4_4_ru_ranker_factorial.cross_encoder import FrozenRevisionCrossEncoder
from experiments.qa_chunk_selection.v4_5_ru_compact_refinement.contract import RankerSpec
from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS as OLD_CELLS
from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import DITY_RECIPE, LEGACY_RECIPE, MINILM_SNAPSHOT
from experiments.qa_chunk_selection.v5_question_conditioned_residual.historical import HistoricalCell, tier
from experiments.qa_chunk_selection_v6.features import enrich_question
from experiments.qa_chunk_selection_v8.features import enrich_reader_features, hrr_rows

from .config import Config
from .contract import CELL_SPECS
from .io_guard import read_jsonl, write_json, write_jsonl


def runtime_as_unlabeled(item: RuntimeInstance) -> InstanceRecord:
    chunks = tuple(ChunkRecord(
        chunk_id=x.chunk_id, text=x.text, label="easy_negative", primary_class="unresolved_for_training",
        position=x.position, section_id=x.section_id, section_title=x.section_title,
        char_start=x.char_start, char_end=x.char_end, token_start=x.token_start, token_end=x.token_end,
        gold_supporting=False, previous_chunk_id=x.previous_chunk_id, next_chunk_id=x.next_chunk_id,
        eligible_for_qa=x.eligible_for_qa,
    ) for x in item.chunks)
    return InstanceRecord(item.document_id, item.question_id, item.aspect, item.question, (), item.split, chunks, (), "runtime_only", item.split)


class LoadedCrossEncoder:
    def __init__(self, path, device: str, batch_size: int = 8, max_length: int = 512, *, local_files_only: bool = True, revision: str | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        kwargs = {"local_files_only": local_files_only, "trust_remote_code": False}
        if revision: kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, **kwargs).to(device)
        self.device, self.batch_size, self.max_length = device, batch_size, max_length

    def predict(self, pairs):
        import torch
        out = []; self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start:start + self.batch_size]
                encoded = self.tokenizer([x[0] for x in batch], [x[1] for x in batch], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                logits = self.model(**{k: v.to(self.device) for k, v in encoded.items()}).logits.view(-1)
                out.extend(torch.sigmoid(logits).float().cpu().tolist())
        return [float(x) for x in out]


class Reader:
    def __init__(self, model_path: Path, device: str, smoke: bool):
        self.smoke = smoke; self.model_path = Path(model_path); self.device = device; self.answerer = None

    def _load(self):
        if self.answerer is not None or self.smoke:
            return
        # transformers 5.x removed the generic ``question-answering`` pipeline
        # registry entry. The project's canonical reader already defines direct
        # start/end-logit decoding, so use it independently of that registry.
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        from untie.models import ExtractiveQuestionAnswerer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False,
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False,
        ).to(self.device)
        self.answerer = ExtractiveQuestionAnswerer(
            model=model, tokenizer=tokenizer, device=self.device,
            max_length=512, max_answer_tokens=64,
        )

    def __call__(self, question: str, text: str) -> dict:
        if self.smoke:
            # Deterministic and extractive.  Smoke validates plumbing, never science.
            words = text.split(); answer = " ".join(words[: min(4, len(words))])
            start = text.find(answer) if answer else 0
            return {"answer": answer, "score": 1.0 / max(len(words), 1), "start": start, "end": start + len(answer)}
        self._load()
        try:
            raw = dict(self.answerer(question=question, context=text))
            return {"answer": str(raw.get("answer") or ""), "score": float(raw.get("score") or 0.0), "start": int(raw.get("start") or 0), "end": int(raw.get("end") or 0), "error": None}
        except Exception as exc:
            return {"answer": "", "score": -1e30, "start": 0, "end": 0, "error": f"{type(exc).__name__}: {exc}"}


class CellEngine:
    def __init__(self, config: Config, cell: str, progress):
        self.config, self.cell, self.progress = config, cell, progress
        self.spec = CELL_SPECS[cell]
        self.adapter = HistoricalCell(
            OLD_CELLS[cell], config.run_dir / "working", device=config.device, smoke=config.smoke,
            local_files_only=not config.allow_downloads, progress=progress,
        )
        self.device = self.adapter.device
        self.reader = Reader(OLD_CELLS[cell].qa_model, self.device, config.smoke)
        self.qa_cache_path = config.run_dir / "cache/qa" / f"{cell}.jsonl"
        self.qa_cache = {}
        for row in read_jsonl(self.qa_cache_path):
            self.qa_cache[(row["question_id"], int(row["chunk_id"]), row["question_hash"])] = row
        self.question_mapping = None

    def select_questions(self, training_records, frozen=None):
        mapping = dict(frozen) if frozen is not None else self.adapter.question_map(training_records)
        self.question_mapping = mapping
        return mapping

    def apply_questions(self, records):
        if self.question_mapping is None: raise RuntimeError("question wording must be frozen before model use")
        output = []
        for item in records:
            variant = self.question_mapping.get(item.aspect)
            question = item.question if variant is None else str(self.adapter.questions[item.aspect][int(variant)])
            output.append(replace(item, question=question))
        return tuple(output)

    @staticmethod
    def _qhash(question: str) -> str:
        return hashlib.sha256(question.encode()).hexdigest()

    def qa(self, item: InstanceRecord, chunk_id: int, question: str) -> dict:
        key = (str(item.question_id), int(chunk_id), self._qhash(question))
        if key in self.qa_cache:
            return dict(self.qa_cache[key])
        chunk = item.chunk_by_id(chunk_id); raw = self.reader(question, chunk.text)
        row = {"question_id": str(item.question_id), "chunk_id": int(chunk_id), "question_hash": key[2],
               "answer": raw["answer"], "confidence": float(raw["score"]), "start": int(raw["start"]), "end": int(raw["end"]),
               "model": str(OLD_CELLS[self.cell].qa_model.resolve()), "model_error": raw.get("error")}
        self.qa_cache[key] = row; write_jsonl(self.qa_cache_path, self.qa_cache.values())
        return row

    def train_b0(self, train: Sequence[InstanceRecord], calibration: Sequence[InstanceRecord], seed: int, work: Path):
        self.progress.stage(f"b0:{self.cell}:s{seed}", "train canonical language-specific B0 ranker")
        scorer, report = self.adapter.train_ranker(train, calibration, seed, work)
        model_dir = Path(work) / "model"
        if self.config.smoke and not model_dir.exists() and hasattr(scorer, "save"):
            scorer.save(model_dir)
        self.adapter.retain_ranker(scorer, work)
        write_json(Path(work) / "final_training_report.json", report)
        return scorer, report

    def refit_b0(self, pool: Sequence[InstanceRecord], seed: int, epochs: int, work: Path):
        """Fit on every TRAIN+VAL row for the epoch count selected in development."""
        work = Path(work); work.mkdir(parents=True, exist_ok=True)
        if OLD_CELLS[self.cell].ranker_recipe == "legacy_b2a":
            args = dict(LEGACY_RECIPE); args.update(epochs=int(epochs), patience=int(epochs) + 1)
            result = train_legacy_b0(pool, pool, self.adapter.flags, seed=seed, device=self.device,
                model_name=str(MINILM_SNAPSHOT), output_dir=work / "model", smoke=self.config.smoke,
                progress=self.progress, **args)
            scorer, report = result["scorer"], {"recipe": "legacy_b2a", "fixed_epochs": epochs, "all_train_val_in_gradient_pool": True, "calibration": result.get("calibration")}
        else:
            from experiments.qa_chunk_selection.v4_4_ru_ranker_factorial.cross_encoder import train_cross_encoder
            args = dict(DITY_RECIPE); args.update(epochs=int(epochs), patience=int(epochs) + 1)
            spec = RankerSpec("dity", OLD_CELLS[self.cell].ranker_id, OLD_CELLS[self.cell].ranker_revision, "final_refit")
            scorer, report = train_cross_encoder(pool, pool, self.adapter.flags, spec=spec, seed=seed,
                device=self.device, progress=self.progress, report_path=work / "training_report.json",
                local_files_only=not self.config.allow_downloads, smoke=self.config.smoke, **args)
        if self.config.smoke and not (work / "model").exists() and hasattr(scorer, "save"): scorer.save(work / "model")
        self.adapter.retain_ranker(scorer, work)
        write_json(work / "final_training_report.json", {**report, "fixed_epochs": int(epochs), "refit_pool_questions": len(pool), "refit_pool_documents": len({x.document_id for x in pool})})
        return scorer

    def load_b0(self, work: Path):
        path = Path(work) / "model"
        if self.config.smoke:
            return TfidfAnswerabilityScorer.load(path)
        if OLD_CELLS[self.cell].ranker_recipe == "legacy_b2a":
            return AnswerabilityModel.load(path, device=self.device)
        return LoadedCrossEncoder(path, self.device)

    @staticmethod
    def scores(scorer, item: InstanceRecord) -> dict[int, float]:
        runtime_pairs = [(item.question, x.text) for x in item.chunks]
        if hasattr(scorer, "predict"):
            values = scorer.predict(runtime_pairs)
        else:
            values = score_fn_for(scorer)(item)
        return {int(chunk.chunk_id): float(score) for chunk, score in zip(item.chunks, values)}

    @staticmethod
    def qarel(scorer, question: str, answers: dict[int, str]) -> dict[int, float]:
        if not answers:
            return {}
        values = scorer.predict([(question, answer) for answer in answers.values()])
        return {cid: float(value) for cid, value in zip(answers, values)}

    def feature_question(self, source: InstanceRecord | RuntimeInstance, scorer, seed: int, *, with_targets: bool):
        item = source if isinstance(source, InstanceRecord) else runtime_as_unlabeled(source)
        scores = self.scores(scorer, item)
        ranked = sorted(item.chunks, key=lambda x: (-scores[x.chunk_id], x.chunk_id))
        top = ranked[: min(self.spec.k, len(ranked))]
        qa = {x.chunk_id: self.qa(item, x.chunk_id, item.question) for x in top}
        answers = {cid: value["answer"] for cid, value in qa.items()}
        qarel = self.qarel(scorer, item.question, answers)
        raw = []
        for rank, chunk in enumerate(top, 1):
            raw.append({"question_id": item.question_id, "aspect_id": item.aspect, "chunk_id": chunk.chunk_id,
                        "b0_rank": rank, "b0_score_raw": scores[chunk.chunk_id], "q_answer_ranker_raw": qarel.get(chunk.chunk_id, 0.0),
                        "qa_answer": answers.get(chunk.chunk_id, ""), "qa_confidence": qa[chunk.chunk_id]["confidence"],
                        "candidate_count": len(item.chunks), "label_tier": tier(chunk) if with_targets else "unresolved_for_training"})
        runtime, target = enrich_question(self.cell, raw, item, outer_fold=0, seed=seed, meta_source=with_targets)
        runtime = enrich_reader_features(runtime, item)
        conf = {int(x["chunk_id"]): float(x["qa_confidence"]) for x in raw}
        for row in runtime:
            row["qa_confidence"] = conf[int(row["chunk_id"])]
            row["question_text"] = item.question
        if not with_targets:
            return runtime, None, []
        target["gold_answers"] = list(item.gold_answers)
        target["chunk_label"] = {str(x.chunk_id): x.label for x in top}
        target["primary_class"] = {str(x.chunk_id): x.primary_class for x in top}
        return runtime, target, hrr_rows(runtime, target)

    def materialize(self, records: Sequence[InstanceRecord | RuntimeInstance], scorer, seed: int, *, role: str, with_targets: bool):
        runtime, targets, rows = [], {}, []
        total = len(records)
        for index, item in enumerate(records, 1):
            run, target, alternatives = self.feature_question(item, scorer, seed, with_targets=with_targets)
            runtime.extend(run); rows.extend(alternatives)
            if target is not None:
                targets[(self.cell, str(item.question_id))] = target
            self.progress.count(index, total, item=f"{self.cell} {role} canonical features", extra={"seed": seed})
        return runtime, targets, rows


def release_cuda(*objects) -> None:
    for value in objects:
        del value
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError:
        pass
