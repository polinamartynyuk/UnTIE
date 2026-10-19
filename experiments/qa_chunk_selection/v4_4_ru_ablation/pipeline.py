from collections import defaultdict
from itertools import product
from pathlib import Path
import gc
import json
import math
import threading
import time

import numpy as np
import pandas as pd

from experiments.qa_chunk_selection.io_util import write_json, write_jsonl
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v2.hn_flags import flag_index
from experiments.qa_chunk_selection.v3.data import records_from_jsonl
from experiments.qa_chunk_selection.v3.split import records_for_role
from experiments.qa_chunk_selection.v4.answers import normalize_predicted_answer, pair_token_f1
from experiments.qa_chunk_selection.v4.evaluate import chunk_prediction
from experiments.qa_chunk_selection.v4.shortlist import scores_from_row, top_k_ids
from experiments.qa_chunk_selection.v4.v6_reader import agreement_features
from experiments.qa_chunk_selection.v4_1.evaluate import role_of_document, row_key, summarize_rows
from experiments.qa_chunk_selection.v4_1.ranking import v6_fused_score
from experiments.qa_chunk_selection.v4_3.ranking import isolate_with_b0_tiebreak
from experiments.qa_chunk_selection.v4_3.text import is_valid_answer
from experiments.qa_chunk_selection.v4_4_ru.encoder import build_encoder
from experiments.qa_chunk_selection.v4_4_ru.evaluate import paired_delta
from experiments.qa_chunk_selection.v4_4_ru.metrics_ru import apply_unicode_rouge
from experiments.qa_chunk_selection.v4_4_ru.io_guard import read_jsonl
from experiments.qa_chunk_selection.v4_4_ru_q5_b0.pipeline import choose_device, score_eval

from .contract import (EPSILON_HN, HIT1_TOLERANCE, MIN_ROUTING_ROWS,
                       ROUTING_QUANTILE, SOURCE_OVERLAP, WEIGHT_GRID)
from .cross_encoder import train_cross_encoder
from .features import adaptive_semantic_pack, lemma_features, routing_available
from .io_guard import sha256
from .lemma import RussianLemmatizer
from .tuning import select_stable


class Pipeline:
    def __init__(self, config, output):
        self.config, self.output = config, Path(output)
        self.progress = ProgressLogger(self.output / "progress.jsonl")
        self.started = time.time()

    def run(self):
        stop = threading.Event()
        heartbeat = threading.Thread(
            target=lambda: self._heartbeat(stop), daemon=True, name="v44-ru-ablation-heartbeat")
        heartbeat.start()
        try:
            (self.output / "FAILED.json").unlink(missing_ok=True)
            self._load()
            b0_sets = {"current": self.base_b0}
            if self.config.phase in {"cross-encoders", "all"}:
                b0_sets.update(self._cross_encoders())
            if self.config.phase in {"protocol", "all"}:
                for model_id, rows in b0_sets.items():
                    done = self.output / "protocol" / model_id / "COMPLETED.json"
                    if done.exists():
                        self.progress.log("protocol variant already complete", model=model_id)
                        continue
                    self._protocol(model_id, rows)
            write_json(self.output / "COMPLETED.json", {"exit_code": 0, "experiment": "V4.4-RU ablations",
                       "elapsed_seconds": time.time() - self.started})
            (self.output / "FAILED.json").unlink(missing_ok=True)
            self.progress.stage("done", "exit=0")
            return 0
        except Exception as exc:
            write_json(self.output / "FAILED.json", {"type": type(exc).__name__, "error": str(exc)})
            self.progress.warn(f"FAILED {type(exc).__name__}: {exc}")
            raise
        finally:
            stop.set()
            heartbeat.join(timeout=1)

    def _heartbeat(self, stop):
        while not stop.wait(30):
            self.progress.log("still running")

    def _load(self):
        self.progress.stage("load", "read-only completed Russian Q5/B0 artifact")
        root = self.config.input_artifact.resolve()
        self.records = records_from_jsonl(root / "data/answerable_dataset.jsonl")
        self.items = {item.question_id: item for item in self.records}
        self.splits = json.loads((root / "data/answerable_cv_splits.json").read_text(encoding="utf-8"))
        self.flags_rows = read_jsonl(root / "data/hn_quality_flags.jsonl")
        self.flags = flag_index(self.flags_rows)
        self.base_b0 = read_jsonl(root / "b0/oof_predictions_scored.jsonl")
        self.qcache = {}
        for row in read_jsonl(root / "q5/prediction_cache.jsonl"):
            key = (str(row["document_id"]), str(row["aspect"]), int(row["chunk_id"]), int(row["question_id"]))
            if key in self.qcache:
                raise ValueError(f"duplicate Q5 cache row: {key}")
            self.qcache[key] = row
        self.qpred = pd.read_parquet(root / "q5/question_predictions.parquet")
        chunk = pd.read_parquet(root / "q5/chunk_labels.parquet")
        self.support = chunk[["qa_instance_id", "chunk_id", "gold_supporting"]]
        expected = {(qid, seed) for qid in self.items for seed in self.config.seeds}
        if {row_key(row) for row in self.base_b0} != expected:
            raise ValueError("baseline B0 coverage mismatch")
        self.progress.log("input loaded", questions=len(self.records), documents=len({x.document_id for x in self.records}),
                          q5_cache=len(self.qcache), b0_rows=len(self.base_b0))
        write_json(self.output / "input_manifest.json", {
            "input": str(root), "read_only": True,
            "sha256": {name: sha256(root / name) for name in (
                "data/answerable_dataset.jsonl", "data/answerable_cv_splits.json",
                "q5/prediction_cache.jsonl", "b0/oof_predictions_scored.jsonl")}})

    def _cross_encoders(self):
        self.progress.stage("cross-encoders", "three-seed grouped OOF B0 ablation")
        device, report = choose_device(self.config.device)
        self.progress.log("training device selected", device=device, **report)
        canonical = {(doc, aspect, cid): row for (doc, aspect, cid, qid), row in self.qcache.items() if qid == 0}
        out = {}
        for model_id, model_name in self.config.models:
            model_root = self.output / "cross_encoders" / model_id
            all_rows = []
            for run in range(5):
                train = records_for_role(self.records, self.splits, run=run, role="train")
                cal = records_for_role(self.records, self.splits, run=run, role="calibration")
                evaluation = records_for_role(self.records, self.splits, run=run, role="eval")
                for seed in self.config.seeds:
                    run_dir = model_root / f"run_{run}" / f"seed_{seed}"
                    predictions = run_dir / "predictions_scored.jsonl"
                    if predictions.exists():
                        rows = read_jsonl(predictions)
                        expected_keys = {(item.question_id, seed) for item in evaluation}
                        if {row_key(row) for row in rows} != expected_keys or len(rows) != len(expected_keys):
                            self.progress.warn("incomplete saved fold; retraining it", model=model_id, run=run, seed=seed)
                            rows = []
                        else:
                            self.progress.log("resumed cross-encoder fold", model=model_id, run=run, seed=seed, rows=len(rows))
                    else:
                        rows = []
                    if not rows:
                        run_dir.mkdir(parents=True, exist_ok=True)
                        self.progress.log("train cross-encoder", model=model_id, checkpoint=model_name,
                                          run=run, seed=seed, train=len(train), cal=len(cal), eval=len(evaluation))
                        scorer, _ = train_cross_encoder(
                            train, cal, self.flags_rows, model_name=model_name, seed=seed, device=device,
                            epochs=self.config.epochs, patience=self.config.patience,
                            micro_batch_size=self.config.micro_batch_size,
                            gradient_accumulation=self.config.gradient_accumulation,
                            learning_rate=self.config.learning_rate, max_length=self.config.max_length,
                            progress=self.progress, report_path=run_dir / "training_report.json",
                            smoke=self.config.smoke)
                        rows = score_eval(evaluation, scorer, run=run, seed=seed, flags=self.flags_rows,
                                          qa_lookup=canonical, progress=self.progress)
                        write_jsonl(predictions, rows)
                        del scorer; gc.collect()
                        try:
                            import torch
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                        except ImportError:
                            pass
                    all_rows.extend(rows)
            if {row_key(row) for row in all_rows} != {(qid, seed) for qid in self.items for seed in self.config.seeds}:
                raise ValueError(f"cross-encoder OOF coverage mismatch: {model_id}")
            write_jsonl(model_root / "oof_predictions_scored.jsonl", all_rows)
            write_json(model_root / "metrics.json", summarize_rows(all_rows))
            write_json(model_root / "bootstrap_vs_current.json",
                       paired_delta(self.base_b0, all_rows, n_boot=50 if self.config.smoke else 5000, seed=42))
            write_json(model_root / "COMPLETED.json", {"model": model_name, "rows": len(all_rows)})
            out[model_id] = all_rows
        return out

    def _question_selection(self, run):
        eval_docs = {doc for doc in self.splits["assignment"]
                     if role_of_document(doc, self.splits, run) == "eval"}
        allowed_qids = set(self.items)
        table = self.qpred[self.qpred["qa_instance_id"].isin(allowed_qids)].merge(
            self.support, on=["qa_instance_id", "chunk_id"])
        source = table[(~table.document_id.astype(str).isin(eval_docs)) & table.gold_supporting]
        scores = source.groupby(["aspect", "question_id"]).qa_correct.agg(["mean", "sum", "count"]).reset_index()
        selected = {}
        for aspect in sorted({item.aspect for item in self.records}):
            part = scores[scores.aspect == aspect].sort_values(["mean", "sum", "question_id"], ascending=[False, False, True])
            if part.empty:
                raise ValueError(f"no question-selection source for {aspect}")
            selected[aspect] = int(part.iloc[0].question_id)
        return selected, {"run": run, "selected": selected, "source_roles": ["train", "cal"],
                          "eval_documents_used": 0, "scores": scores.to_dict(orient="records")}

    def _qa(self, run, item, cid, strategy, selected):
        variant = 0 if strategy == "q0" else int(selected[item.aspect])
        source = self.qcache[(item.document_id, item.aspect, int(cid), variant)]
        answer = str(source.get("qa_predicted_answer") or "")
        return {"qa_answer": answer, "qa_chunk_id": int(cid),
                "qa_chunk_start_char": int(source.get("answer_start") or 0),
                "qa_chunk_end_char": int(source.get("answer_end") or 0),
                "qa_start_logit": source.get("start_logit"), "qa_end_logit": source.get("end_logit"),
                "qa_span_score": float(source.get("start_logit") or 0) + float(source.get("end_logit") or 0),
                "qa_question_variant": variant, "qa_forward_passes": 0}

    def _q5_consistency(self, item, cid):
        keys = [normalize_predicted_answer(self.qcache[(item.document_id, item.aspect, int(cid), qid)].get("qa_predicted_answer") or "")
                for qid in range(5)]
        counts = {key: keys.count(key) for key in keys if key}
        return float(max(counts.values()) / 5.0) if counts else 0.0

    def _protocol(self, model_id, b0_rows):
        root = self.output / "protocol" / model_id
        root.mkdir(parents=True, exist_ok=True)
        self.progress.stage("protocol", f"Russian V4.4 ablations for B0={model_id}")
        b0 = {row_key(row): row for row in b0_rows}
        selections = {}
        for run in range(5):
            selections[run], report = self._question_selection(run)
            write_json(root / f"question_selection/run_{run}.json", report)
        encoder = build_encoder(root / "semantic_cache", device=self.config.device,
                                smoke=self.config.smoke, progress=self.progress)
        lemmatizer = RussianLemmatizer(smoke=self.config.smoke)
        grid3 = [dict(zip(("alpha", "beta", "gamma"), values)) for values in product(WEIGHT_GRID, repeat=3)]
        grid2_sem = [dict(lambda_m=a, lambda_s=b) for a, b in product(WEIGHT_GRID, repeat=2)]
        grid2_lex = [dict(lambda_m=a, lambda_t=b) for a, b in product(WEIGHT_GRID, repeat=2)]
        grid2_lem = [dict(lambda_m=a, lambda_l=b) for a, b in product(WEIGHT_GRID, repeat=2)]
        all_predictions, tuning_manifest = defaultdict(list), {}
        for strategy in ("q0", "selected"):
            for k in self.config.k_values:
                cell = f"{strategy}_k{k}"
                tuning_manifest[cell] = {}
                for run in range(5):
                    selected = selections[run]
                    features, packs, lemmas, utility = {}, {}, {}, {}
                    keys = sorted(b0)
                    unique_answers = []
                    qa_maps = {}
                    for key in keys:
                        item, row = self.items[key[0]], b0[key]
                        top = top_k_ids(row, k)
                        qa = {cid: self._qa(run, item, cid, strategy, selected) for cid in top}
                        qa_maps[key] = qa
                        unique_answers.extend(x["qa_answer"] for x in qa.values() if is_valid_answer(x["qa_answer"]))
                    encoder.embed_many(unique_answers, store="answer")
                    for index, key in enumerate(keys, 1):
                        item, row, qa = self.items[key[0]], b0[key], qa_maps[key]
                        top = top_k_ids(row, k)
                        features[key] = agreement_features(item, row, qa, k=k, min_overlap=SOURCE_OVERLAP)
                        answers = {cid: qa[cid]["qa_answer"] for cid in top}
                        embeddings = {cid: encoder.embed_one(text, store="answer") for cid, text in answers.items() if is_valid_answer(text)}
                        packs[key] = adaptive_semantic_pack(answers, embeddings, top)
                        lemmas[key] = lemma_features(answers, top, lemmatizer)
                        utility[key] = {cid: {"f1": max((pair_token_f1(answers[cid], gold) for gold in item.gold_answers), default=0.0),
                                              "lemma_f1": lemmatizer.gold_f1(answers[cid], item.gold_answers),
                                              "positive": float(item.chunk_by_id(cid).label == "positive"),
                                              "hn": float(item.chunk_by_id(cid).label == "hard_negative")}
                                        for cid in top}
                        self.progress.count(index, len(keys), item=f"features {model_id}/{cell}/run{run}")

                    def ranked(key, kind, params):
                        item, row = self.items[key[0]], b0[key]
                        top = top_k_ids(row, k)
                        if kind == "v6":
                            scores = {cid: v6_fused_score(features[key][cid], **params) for cid in top}
                        elif kind == "sem":
                            scores = {cid: params["lambda_m"] * features[key][cid]["majority"] +
                                      params["lambda_s"] * packs[key]["sem_rel"][str(cid)] for cid in top}
                        elif kind == "lex":
                            scores = {cid: params["lambda_m"] * features[key][cid]["majority"] +
                                      params["lambda_t"] * features[key][cid]["mean_token_f1"] / 100 for cid in top}
                        else:
                            scores = {cid: params["lambda_m"] * lemmas[key][cid]["lemma_majority"] +
                                      params["lambda_l"] * lemmas[key][cid]["mean_lemma_f1"] / 100 for cid in top}
                        return isolate_with_b0_tiebreak(item, row, scores, k=k)

                    source = [key for key in keys if role_of_document(self.items[key[0]].document_id, self.splits, run) != "eval"]
                    fold_of = {key: int(self.splits["assignment"][self.items[key[0]].document_id]) for key in keys}
                    baseline_top = lambda key: int(b0[key]["ranked_ids"][0])
                    choices = {}
                    for kind, candidates, stable in (("v6", grid3, True), ("sem", grid2_sem, False),
                                                     ("lex", grid2_lex, False), ("lemma", grid2_lem, False)):
                        result = select_stable(source, candidates,
                            lambda key, params, name=kind: ranked(key, name, params)[0][0], utility, fold_of,
                            baseline_top=baseline_top, epsilon_hn=EPSILON_HN,
                            hit_tolerance=HIT1_TOLERANCE, require_stable=stable,
                            progress=self.progress, label=f"tune {model_id}/{cell}/run{run}/{kind}")
                        fallback = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0} if kind == "v6" else candidates[0]
                        choices[kind] = result["selected"]["params"] if result["selected"] else fallback
                        write_json(root / f"tuning/{cell}/run_{run}_{kind}.json", result)
                    tuning_manifest[cell][str(run)] = choices
                    source_conf, source_q5 = [], []
                    for key in source:
                        sem_order = ranked(key, "sem", choices["sem"])[0]
                        if routing_available(packs[key], sem_order[0]):
                            source_conf.append(float(packs[key]["consensus_conf"]))
                            source_q5.append(self._q5_consistency(self.items[key[0]], sem_order[0]))
                    tau = float(np.quantile(source_conf, ROUTING_QUANTILE)) if len(source_conf) >= MIN_ROUTING_ROWS else None
                    q5_tau = float(np.quantile(source_q5, .30)) if len(source_q5) >= MIN_ROUTING_ROWS else None
                    eval_keys = [key for key in keys if role_of_document(self.items[key[0]].document_id, self.splits, run) == "eval"]
                    for index, key in enumerate(eval_keys, 1):
                        item, row, qa = self.items[key[0]], b0[key], qa_maps[key]
                        variants = {"B0": ([int(x) for x in row["ranked_ids"]], scores_from_row(row))}
                        for kind, name in (("v6", "V6"), ("sem", "SEM"), ("lex", "LEX"), ("lemma", "LEX-LEM")):
                            variants[name] = ranked(key, kind, choices[kind])[:2]
                        sem_ranked, sem_scores = variants["SEM"]
                        use_sem = bool(tau is not None and routing_available(packs[key], sem_ranked[0]) and
                                       packs[key]["consensus_conf"] >= tau)
                        variants["ASEM"] = variants["SEM"] if use_sem else variants["V6"]
                        q5_consistency = self._q5_consistency(item, sem_ranked[0])
                        use_sem_q5 = bool(use_sem and q5_tau is not None and q5_consistency >= q5_tau)
                        variants["ASEM-Q5"] = variants["SEM"] if use_sem_q5 else variants["V6"]
                        expected_top = set(top_k_ids(row, k))
                        for name, (order, scores) in variants.items():
                            if set(int(x) for x in order[:len(expected_top)]) != expected_top:
                                raise ValueError(f"shortlist isolation failed: {model_id}/{cell}/{run}/{name}/{key}")
                            cid = int(order[0])
                            ordinal = {int(value): float(len(order) - pos) for pos, value in enumerate(order)}
                            pred = chunk_prediction(item, ranking_method=f"{name}-{cell}", scores=ordinal,
                                                    qa=qa[cid], flags=self.flags)
                            pred = apply_unicode_rouge(pred, item.gold_answers)
                            pred.update(seed=key[1], cv_run=run, aspect=item.aspect, model_id=model_id,
                                        question_strategy=strategy, k=k, selected_chunk_id=cid,
                                        ranked_ids=[int(x) for x in order], lemma_f1=utility[key][cid]["lemma_f1"],
                                        routing_tau=tau, q5_consistency_tau=q5_tau,
                                        q5_answer_consistency=q5_consistency,
                                        reader_only_used=(use_sem if name == "ASEM" else
                                                          (use_sem_q5 if name == "ASEM-Q5" else None)),
                                        consensus_conf=packs[key]["consensus_conf"])
                            all_predictions[f"{cell}/{name}"].append(pred)
                        self.progress.count(index, len(eval_keys), item=f"replay {model_id}/{cell}/run{run}")
        metrics = {}
        for name, rows in all_predictions.items():
            dest = root / "systems" / name
            write_jsonl(dest / "predictions_scored.jsonl", rows)
            summary = summarize_rows(rows)
            summary["lemma_f1"] = sum(float(x["lemma_f1"]) for x in rows) / max(len(rows), 1)
            write_json(dest / "metrics.json", summary)
            metrics[name] = summary
        comparisons = {}
        for cell in sorted({name.split('/')[0] for name in all_predictions}):
            base = all_predictions[f"{cell}/B0"]
            systems = sorted(name.split('/')[1] for name in all_predictions if name.startswith(cell + "/") and not name.endswith("/B0"))
            comparisons[cell] = {name: paired_delta(base, all_predictions[f"{cell}/{name}"],
                n_boot=50 if self.config.smoke else 5000, seed=42) for name in systems}
        write_json(root / "analysis/metrics.json", metrics)
        write_json(root / "analysis/comparisons_vs_b0.json", comparisons)
        aspects = sorted({item.aspect for item in self.records})
        write_json(root / "analysis/aspect_metrics.json", {
            name: {aspect: summarize_rows([row for row in rows if row["aspect"] == aspect])
                   for aspect in aspects} for name, rows in all_predictions.items()})
        families = {"task_goal": {"task", "goal"}, "method_tool": {"method", "tool"},
                    "task_use": {"task", "use"},
                    "contribution_conclusion": {"contribution", "conclusion"}}
        write_json(root / "analysis/aspect_family_diagnostics.json", {
            name: {family: summarize_rows([row for row in rows if row["aspect"] in members])
                   for family, members in families.items()} for name, rows in all_predictions.items()})
        write_json(root / "tuning/manifest.json", tuning_manifest)
        from .report import write_report
        write_report(root, model_id, metrics, comparisons, tuning_manifest)
        encoder.write_contract(root / "semantic_cache/encoder_spec.json")
        write_json(root / "COMPLETED.json", {"model_id": model_id, "systems": len(all_predictions),
                   "qa_forward_passes": 0, "eval_used_for_selection": False})

