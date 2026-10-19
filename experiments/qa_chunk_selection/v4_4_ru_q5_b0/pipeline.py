from __future__ import annotations

import hashlib
import json
import time
import gc
from collections import Counter
from pathlib import Path

import pandas as pd

from experiments.phase1_multiquestion.assign import relabel_with_threshold
from experiments.phase1_multiquestion.config import MultiQuestionConfig
from experiments.phase1_multiquestion.engine import PredictionCache, embeddings_from_chunks, select_threshold_on_dev
from experiments.phase1_multiquestion.report import chunk_rows, instance_rows, question_prediction_rows
from experiments.phase1_scirex.engine import mention_chunk_ids
from experiments.qa_chunk_selection.io_util import write_json, write_jsonl
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v2.evaluate_v2 import build_raw_prediction, enrich_prediction
from experiments.qa_chunk_selection.v2.hn_flags import flag_index, hn_quality_rows
from experiments.qa_chunk_selection.v2.schemas import rank_ids, to_runtime
from experiments.qa_chunk_selection.v3.data import records_from_jsonl
from experiments.qa_chunk_selection.v3.split import build_cv_splits, records_for_role
from experiments.qa_chunk_selection.v3.train import score_fn_for, train_b0
from untie.config import PipelineConfig
from untie.models import ModelFactory, profile_for_language

from .data import records_from_results, validate, validate_cv
from .q5 import process_document
from .questions import ASPECTS, QUESTIONS, validate_questions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def claim_output(config) -> Path:
    output = config.output_dir.resolve()
    project = Path(__file__).resolve().parents[3]
    artifacts = (project / "artifacts").resolve()
    base = config.base_artifact.resolve()
    if artifacts not in output.parents or output == base or base in output.parents or output in base.parents:
        raise ValueError("output must be a separate sibling directory below artifacts/")
    if any(part.is_symlink() for part in (output, *output.parents)):
        raise ValueError("output path cannot contain symlinks")
    if output.exists():
        if not config.resume or (output / "COMPLETED.json").exists():
            raise FileExistsError(f"output already exists: {output}")
        marker = output / "config.json"
        if not marker.exists():
            raise ValueError("resume requires this experiment's existing config.json")
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("experiment_id") != "v4_4_ru_q5_b0":
            raise ValueError("refusing to resume a directory owned by another experiment")
    else:
        output.mkdir(parents=True)
    return output


def load_russian_models(device: str):
    factory = ModelFactory(PipelineConfig(
        profile=profile_for_language("ru"), device=device,
        chunk_max_tokens=128, overlap_tokens=24,
    ))

    def answer(question, context):
        return dict(factory.answerer(question=question, context=context))

    def encode(texts):
        import numpy as np
        if not texts:
            return np.zeros((0, 1), dtype=float)
        values = factory.sentence_encoder.encode(list(texts), show_progress_bar=False)
        matrix = np.asarray(values, dtype=float)
        return matrix.reshape(1, -1) if matrix.ndim == 1 else matrix

    return factory, answer, encode


def choose_device(requested: str) -> tuple[str, dict]:
    """Choose the CUDA device with most free memory without changing shared model code."""
    if requested != "auto":
        return requested, {"selection": "explicit"}
    try:
        import torch
    except ImportError:
        return "cpu", {"selection": "auto_no_torch"}
    if not torch.cuda.is_available():
        return "cpu", {"selection": "auto_no_cuda"}
    memory = []
    for index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        memory.append({
            "index": index,
            "free_mb": int(free_bytes // (1024 * 1024)),
            "total_mb": int(total_bytes // (1024 * 1024)),
        })
    selected = max(memory, key=lambda row: (row["free_mb"], -row["index"]))["index"]
    return f"cuda:{selected}", {"selection": "auto_most_free", "gpus": memory}


def write_q5_tables(root: Path, results) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(question_prediction_rows(results)).to_parquet(root / "question_predictions.parquet", index=False)
    pd.DataFrame(chunk_rows(results)).to_parquet(root / "chunk_labels.parquet", index=False)
    pd.DataFrame(instance_rows(results)).to_parquet(root / "instances.parquet", index=False)


def score_eval(records, scorer, *, run: int, seed: int, flags, qa_lookup, progress=None):
    score_fn = score_fn_for(scorer)
    indexed_flags = flag_index(flags)
    rows = []
    for gold in records:
        runtime = to_runtime(gold)
        scores = score_fn(runtime)
        ranked = rank_ids(runtime.chunks, scores)
        selected = int(ranked[0]) if ranked else None
        cached = qa_lookup.get((gold.document_id, gold.aspect, selected)) if selected is not None else None
        if cached is None:
            raise RuntimeError(f"missing canonical Q5 QA cache for {gold.question_id}:{selected}")
        start_logit = cached.get("start_logit")
        end_logit = cached.get("end_logit")
        span_score = (float(start_logit) + float(end_logit)) if start_logit is not None and end_logit is not None else None
        qa = {
            "qa_answer": cached.get("qa_predicted_answer") or "",
            "qa_chunk_id": selected,
            "qa_chunk_start_char": cached.get("answer_start") or 0,
            "qa_chunk_end_char": cached.get("answer_end") or 0,
            "qa_start_logit": start_logit,
            "qa_end_logit": end_logit,
            "qa_span_score": span_score,
            "qa_confidence_proxy": span_score if span_score is not None else cached.get("qa_confidence"),
            "used_m": 1,
            "reader_tokens": len(runtime.chunk_by_id(selected).text.split()),
            "qa_forward_passes": 0,
        }
        raw = build_raw_prediction(
            runtime, ranking_method="ru_b0", scores=scores, ranked_ids=ranked, qa=qa,
            extra={"cv_run": run, "seed": seed, "qa_source": "reused_q5_canonical_cache"},
        )
        rows.append(enrich_prediction(
            raw, gold, flags=indexed_flags, skip_bertscore=True, ks=(1, 5, 10),
        ))
        if progress:
            progress.count(len(rows), len(records), item=f"B0 inference run={run} seed={seed}")
    return rows


class Pipeline:
    def __init__(self, config):
        config.validate()
        validate_questions()
        self.config = config
        self.output = claim_output(config)
        self.progress = ProgressLogger(self.output / "progress.jsonl")

    def run(self):
        started = time.time()
        config = self.config
        base = config.base_artifact.resolve()
        dataset_path = base / "data" / "dataset.jsonl"
        fingerprint = {
            "base_dataset_sha256": _sha256(dataset_path),
            "questions": QUESTIONS,
            "seeds": list(config.seeds),
            "split_seed": config.split_seed,
            "n_folds": config.n_folds,
            "b0_model": config.b0_model,
            "epochs": config.epochs,
            "patience": config.patience,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "max_length": config.max_length,
            "smoke": config.smoke,
        }
        # JSON round-trip makes tuples/lists identical between a fresh run and --resume.
        fingerprint = json.loads(json.dumps(fingerprint, ensure_ascii=False, sort_keys=True))
        config_path = self.output / "config.json"
        if config_path.exists():
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if previous.get("fingerprint") != fingerprint:
                raise RuntimeError("resume fingerprint mismatch")
        else:
            write_json(config_path, {"experiment_id": "v4_4_ru_q5_b0", **config.to_dict(), "fingerprint": fingerprint})
        (self.output / "FAILED.json").unlink(missing_ok=True)

        self.progress.stage("load", "unlocked base data")
        base_records = records_from_jsonl(dataset_path)
        unlocked_roles = {"train", "calibration", "development"}
        if any(record.split not in unlocked_roles for record in base_records):
            raise RuntimeError("base unlocked dataset contains a locked split")
        role_by_qid = {record.question_id: record.split for record in base_records}
        unlocked_documents = sorted({record.document_id for record in base_records})
        filters = [("document_id", "in", unlocked_documents)]
        instances = pd.read_parquet(base / "preprocessing" / "aspect_instances.parquet", filters=filters)
        chunks = pd.read_parquet(base / "preprocessing" / "chunks.parquet", filters=filters)
        mentions = pd.read_parquet(base / "preprocessing" / "gold_mentions.parquet", filters=filters)
        instances["qa_instance_id"] = (
            instances["document_id"].astype(str) + "::" + instances["aspect"].astype(str)
        )
        instances = instances[instances["qa_instance_id"].isin(role_by_qid)].copy()
        instances["split"] = instances["qa_instance_id"].map(role_by_qid)
        if set(instances["aspect"].astype(str)) != set(ASPECTS):
            raise RuntimeError("not all nine Russian aspects are present")
        self.progress.log("base loaded", instances=len(instances), documents=instances.document_id.nunique())

        self.progress.stage("q5", "five frozen Russian questions per aspect")
        run_device, device_report = choose_device(config.device)
        self.progress.log("compute device selected", device=run_device, **device_report)
        factory, answer_fn, encode_fn = load_russian_models(run_device)
        q5_config = MultiQuestionConfig(
            input_dir=base / "preprocessing", output_dir=self.output / "q5",
            mode="Q5", n_questions=5, seed=config.split_seed, device=config.device,
            prediction_cache_path=self.output / "q5" / "prediction_cache.jsonl",
        )
        cache = PredictionCache(q5_config.prediction_cache_path)
        results = []
        document_ids = sorted(instances.document_id.astype(str).unique())
        for index, document_id in enumerate(document_ids, 1):
            results.extend(process_document(
                document_id,
                instances[instances.document_id.astype(str) == document_id],
                chunks[chunks.document_id.astype(str) == document_id].sort_values("chunk_id"),
                mentions, q5_config, answer_fn, encode_fn, cache, threshold=0.6,
            ))
            self.progress.count(index, len(document_ids), item="documents Q5-labelled",
                                extra={"instances": len(results)})

        self.progress.stage("threshold", "base calibration only; excluded from CV")
        threshold_source = [row for row in results if row.split == "calibration"]
        if not threshold_source:
            raise RuntimeError("empty Q5 threshold-development subset")
        selection = select_threshold_on_dev(
            threshold_source, encode_fn=encode_fn, config=q5_config, mentions=mentions,
        )
        chosen = float(selection["chosen_threshold"])
        frozen_results = []
        for frozen_index, row in enumerate(results, 1):
            frozen_results.append(relabel_with_threshold(
                row, embeddings_by_id=embeddings_from_chunks(row.chunks, encode_fn),
                config=q5_config, threshold=chosen,
                gold_mention_chunk_ids=mention_chunk_ids(mentions, row.document_id, row.aspect),
            ))
            self.progress.count(frozen_index, len(results), item="instances relabelled at frozen Q5 threshold")
        q5_dir = self.output / "q5"
        write_json(q5_dir / "threshold_selection.json", {
            key: value for key, value in selection.items() if key != "results"
        })
        write_q5_tables(q5_dir, frozen_results)
        records = records_from_results(frozen_results, base_records)
        errors = validate(records, role_by_qid)
        if errors:
            raise RuntimeError(f"Q5 validation failed: {errors[:10]}")

        self.progress.stage("cv", "answerable train/development instances only")
        threshold_dev = tuple(row for row in records if row.split == "calibration")
        cv_records = tuple(
            row for row in records
            if row.split in {"train", "development"} and row.positives
        )
        data_dir = self.output / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(data_dir / "dataset_q5_all_unlocked.jsonl", (row.to_dict() for row in records))
        write_jsonl(data_dir / "q5_threshold_dev.jsonl", (row.to_dict() for row in threshold_dev))
        write_jsonl(data_dir / "answerable_dataset.jsonl", (row.to_dict() for row in cv_records))
        flags = hn_quality_rows(cv_records)
        write_jsonl(data_dir / "hn_quality_flags.jsonl", flags)
        cv = build_cv_splits(
            cv_records, n_folds=config.n_folds, seed=config.split_seed,
            output_path=data_dir / "answerable_cv_splits.json",
        )
        errors = validate_cv(cv_records, cv)
        threshold_docs = {row.document_id for row in threshold_dev}
        cv_docs = {row.document_id for row in cv_records}
        if threshold_docs & cv_docs:
            errors.append("Q5 threshold-development documents overlap CV")
        if errors:
            raise RuntimeError(f"CV validation failed: {errors}")
        validation = {
            "q5_question_count": sum(len(value) for value in QUESTIONS.values()),
            "q5_aspect_count": len(QUESTIONS),
            "q5_qid_coverage_exact": True,
            "cv_answerable_only": all(bool(row.positives) for row in cv_records),
            "cv_grouping_unit": cv["grouping_unit"],
            "threshold_dev_cv_document_overlap": 0,
            "internal_test_file_opened": False,
            "primary_classes": dict(Counter(chunk.primary_class for row in records for chunk in row.chunks)),
        }
        write_json(data_dir / "validation_report.json", validation)
        self.progress.log("answerable-only CV frozen", instances=len(cv_records),
                          documents=len(cv_docs))

        self.progress.stage("b0", "five-fold OOF for three seeds")
        training_device = factory.device
        canonical_qa = {
            (key[0], key[1], key[2]): value
            for key, value in cache.rows.items() if key[3] == 0
        }
        del answer_fn, encode_fn, cache, factory
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        oof = []
        for run in range(cv["n_folds"]):
            train = records_for_role(cv_records, cv, run=run, role="train")
            calibration = records_for_role(cv_records, cv, run=run, role="calibration")
            evaluation = records_for_role(cv_records, cv, run=run, role="eval")
            for seed in config.seeds:
                run_dir = self.output / "b0" / f"run_{run}" / f"seed_{seed}"
                predictions_path = run_dir / "predictions_scored.jsonl"
                if predictions_path.exists():
                    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                else:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    self.progress.log("B0 training", run=run, seed=seed, train=len(train),
                                      calibration=len(calibration), evaluation=len(evaluation))
                    trained = train_b0(
                        train, calibration, flags, seed=seed, device=training_device,
                        model_name=config.b0_model, output_dir=run_dir / "model",
                        epochs=config.epochs, patience=config.patience,
                        batch_size=config.batch_size, learning_rate=config.learning_rate,
                        max_length=config.max_length, ratio=(1, 3, 1),
                        smoke=config.smoke, progress=self.progress,
                    )
                    scorer = trained["scorer"]
                    rows = score_eval(evaluation, scorer, run=run, seed=seed, flags=flags,
                                      qa_lookup=canonical_qa, progress=self.progress)
                    write_jsonl(predictions_path, rows)
                    del scorer, trained
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                oof.extend(rows)
                self.progress.log("B0 fold complete", run=run, seed=seed, rows=len(rows))
        b0_dir = self.output / "b0"
        write_jsonl(b0_dir / "oof_predictions_scored.jsonl", oof)
        expected = {(row.question_id, seed) for row in cv_records for seed in config.seeds}
        actual = {(row["question_id"], int(row["seed"])) for row in oof}
        if actual != expected or len(oof) != len(expected):
            raise RuntimeError("B0 OOF coverage or uniqueness failure")

        summary = {
            "q5_threshold": chosen,
            "q5_instances": len(records),
            "threshold_dev_instances": len(threshold_dev),
            "cv_answerable_instances": len(cv_records),
            "cv_documents": len({row.document_id for row in cv_records}),
            "b0_seeds": list(config.seeds),
            "b0_oof_rows": len(oof),
            "aspects": dict(Counter(row.aspect for row in cv_records)),
            "locked_instances_processed": False,
            "elapsed_s": time.time() - started,
        }
        write_json(self.output / "summary.json", summary)
        write_json(self.output / "COMPLETED.json", {"status": "complete", **summary})
        self.progress.stage("complete", "Q5, answerable CV and Russian B0 OOF validated")
        return summary
