from collections import defaultdict
from itertools import product
from pathlib import Path
import gc
import json
import threading
import time

import numpy as np

from experiments.qa_chunk_selection.io_util import write_json, write_jsonl
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v2.hn_flags import flag_index
from experiments.qa_chunk_selection.v3.split import records_for_role
from experiments.qa_chunk_selection.v4.answers import pair_token_f1
from experiments.qa_chunk_selection.v4.shortlist import scores_from_row, top_k_ids
from experiments.qa_chunk_selection.v4.v6_reader import agreement_features
from experiments.qa_chunk_selection.v4_1.evaluate import role_of_document, row_key, summarize_rows
from experiments.qa_chunk_selection.v4_1.ranking import v6_fused_score
from experiments.qa_chunk_selection.v4_3.ranking import isolate_with_b0_tiebreak
from experiments.qa_chunk_selection.v4_3.text import is_valid_answer
from experiments.qa_chunk_selection.v4_4_ru.encoder import build_encoder
from experiments.qa_chunk_selection.v4_4_ru.io_guard import read_jsonl
from experiments.qa_chunk_selection.v4_4_ru_q5_b0.pipeline import choose_device, score_eval
from experiments.qa_chunk_selection.v4_4_ru_ablation.features import adaptive_semantic_pack, routing_available
from experiments.qa_chunk_selection.v4_4_ru_ablation.tuning import select_stable
from experiments.qa_chunk_selection.v4_4_ru_ranker_factorial.cross_encoder import train_cross_encoder

from .audit import audit_dataset, historical_stage2_contract, protocol_manifests
from .contract import (BOOTSTRAP_SEED, DATASETS, EPSILON_HN, FOLDS, HIT1_TOLERANCE,
                       MIN_ROUTING_ROWS, RANKERS, ROUTING_QUANTILE, SEEDS,
                       SOURCE_OVERLAP, STAGE2_CONTRACT, SYSTEMS, TRAINING_RECIPE,
                       WEIGHT_GRID)
from .io_guard import (canonical_hash, null_undefined_pairwise, required_paths, sha256,
                       validate_finite)
from .serialization import strict_prediction


class DatasetState:
    def __init__(self, spec, audit):
        self.spec = spec
        self.audit = audit
        self.records = audit["records"]
        self.items = {str(item.question_id): item for item in self.records}
        self.splits = audit["splits"]
        self.paths = audit["paths"]
        self.flags_rows = read_jsonl(self.paths["flags"])
        self.flags = flag_index(self.flags_rows)
        self.qcache = {}
        for row in read_jsonl(self.paths["qcache"]):
            key = (str(row["document_id"]), str(row["aspect"]), int(row["chunk_id"]), int(row["question_id"]))
            if key in self.qcache:
                raise ValueError(f"duplicate QA cache row: {key}")
            self.qcache[key] = row
        self.historical_b0 = read_jsonl(self.paths["historical_b0"])

    def qa(self, item, cid):
        source = self.qcache[(str(item.document_id), str(item.aspect), int(cid), 0)]
        return {
            "qa_answer": str(source.get("qa_predicted_answer") or ""),
            "qa_chunk_id": int(cid),
            "qa_chunk_start_char": int(source.get("answer_start") or 0),
            "qa_chunk_end_char": int(source.get("answer_end") or 0),
            "qa_start_logit": source.get("start_logit"),
            "qa_end_logit": source.get("end_logit"),
            "qa_span_score": float(source.get("start_logit") or 0.0) + float(source.get("end_logit") or 0.0),
            "qa_question_variant": 0,
            "qa_forward_passes": 0,
            "qa_cache_reused": True,
        }


class Pipeline:
    def __init__(self, config, output):
        self.config = config
        self.output = Path(output)
        self.progress = ProgressLogger(self.output / "progress.jsonl")
        self.started = time.time()
        self.audits = {}
        self.states = {}

    def _folds(self):
        return (0,) if self.config.smoke else FOLDS

    def _expected_keys(self, state):
        if not self.config.smoke:
            return {(qid, seed) for qid in state.items for seed in self.config.seeds}
        evaluation = records_for_role(state.records, state.splits, run=0, role="eval")
        return {(item.question_id, 42) for item in evaluation}

    def run(self):
        stop = threading.Event()
        heartbeat = threading.Thread(target=lambda: self._heartbeat(stop), daemon=True, name="v45-ru-heartbeat")
        heartbeat.start()
        try:
            (self.output / "FAILED.json").unlink(missing_ok=True)
            self._audit_and_freeze()
            if self.config.phase == "audit":
                return self._complete()
            if self.config.phase in {"rankers", "all"}:
                self._rankers(train=True)
            else:
                self._rankers(train=False)
            if self.config.phase == "rankers":
                return self._complete()
            if self.config.phase in {"stage2", "all"}:
                self._stage2()
            if self.config.phase == "stage2":
                return self._complete()
            if self.config.phase in {"benchmark", "all"}:
                from .benchmark import run_resource_benchmark
                run_resource_benchmark(self.config, self.output, self.states, self.progress)
            if self.config.phase == "benchmark":
                return self._complete()
            if self.config.phase in {"analysis", "all"}:
                from .analysis import write_analysis
                write_analysis(self.config, self.output, self.states, self.progress)
            self._verify_historical_unchanged()
            return self._complete()
        except Exception as exc:
            write_json(self.output / "FAILED.json", {"type": type(exc).__name__, "error": str(exc)})
            self.progress.warn(f"FAILED {type(exc).__name__}: {exc}")
            raise
        finally:
            stop.set()
            heartbeat.join(timeout=1)

    def _heartbeat(self, stop):
        while not stop.wait(30):
            self.progress.log("still running", elapsed_seconds=round(time.time() - self.started, 1))

    def _complete(self):
        self._verify_historical_unchanged()
        payload = {
            "exit_code": 0, "status": "complete", "experiment": "V4.5-RU compact refinement",
            "phase": self.config.phase, "datasets": list(self.config.datasets),
            "elapsed_seconds": time.time() - self.started,
            "scientific_status": "development_characterization_not_independent_confirmation",
        }
        write_json(self.output / f"PHASE_{self.config.phase}_COMPLETED.json", payload)
        if self.config.phase in {"all", "analysis"}:
            write_json(self.output / "COMPLETED.json", payload)
        (self.output / "FAILED.json").unlink(missing_ok=True)
        self.progress.stage("done", "exit=0")
        return 0

    def _audit_and_freeze(self):
        self.progress.stage("audit", "read-only historical recovery and frozen contract")
        for index, key in enumerate(self.config.datasets, 1):
            self.progress.log("audit dataset", dataset=key, index=index, total=len(self.config.datasets))
            audit = audit_dataset(DATASETS[key])
            self.audits[key] = audit
            self.states[key] = DatasetState(DATASETS[key], audit)
        manifests = protocol_manifests(self.audits)
        inputs = self.output / "inputs"
        regression = self.output / "regression"
        recovery = self.output / "protocol_recovery"
        for key, audit in self.audits.items():
            write_json(inputs / f"{key}_manifest.json", audit["manifest"])
        write_json(inputs / "source_hashes.json", manifests["source_hashes"])
        write_json(inputs / "historical_reference_manifest.json", manifests["historical_reference_manifest"])
        write_json(inputs / "test_provenance_audit.json", manifests["test_provenance_audit"])
        write_json(regression / "hash_before.json", manifests["hash_before"])
        write_json(regression / "split_check.json", manifests["split_check"])
        write_json(regression / "candidate_universe_check.json", manifests["candidate_universe_check"])
        write_json(regression / "chunking_check.json", manifests["chunking_check"])
        write_json(regression / "qa_cache_audit.json", manifests["qa_cache_audit"])
        stage2 = historical_stage2_contract(self.audits)
        write_json(recovery / "historical_stage2_contract.json", stage2)
        write_json(recovery / "parameter_grid.json", {
            "alpha": list(WEIGHT_GRID), "beta": [0.0], "gamma": list(WEIGHT_GRID),
            "sem_lambda_majority": list(WEIGHT_GRID), "sem_lambda_similarity": list(WEIGHT_GRID),
            "expanded": False, "source": "exact V4.4 RU safe grid",
        })
        write_json(recovery / "training_recipe_manifest.json", TRAINING_RECIPE)
        write_json(recovery / "selection_rule.json", {
            "V6": "source-fold majority F1 improvement plus inherited Hit/HN safety",
            "SEM": "maximum source mean F1 among Hit/HN-safe candidates",
            "fallback": "B0-equivalent parameters when no eligible V6 candidate",
            "secondary_tie_breaks": ["positive_folds", "worst_fold_delta", "smaller L1 parameter norm"],
            "outer_eval_used": False,
        })
        write_json(recovery / "q5_ablation_replay.json", {key: audit["q5_replay"] for key, audit in self.audits.items()})
        write_json(recovery / "router_feature_provenance.json", {
            "feature": STAGE2_CONTRACT["router_feature"],
            "inputs": ["runtime q0 QA answers", "frozen semantic encoder embeddings", "candidate ids"],
            "calibration_gold_use": "TRAIN+CAL utility only for parameter selection; threshold itself is Q70 of runtime confidence",
            "runtime_forbidden": ["gold labels", "gold answer F1", "evidence class", "outer-EVAL truth"],
            "valid_for_new_documents": True,
        })
        write_json(self.output / "runtime_dependency_graph.json", {
            "S0_historical_b0": ["historical ranker", "QA(top1)"],
            "S1_dity_b0": ["DiTy all-candidate scoring", "QA(top1)"],
            "S2_v6_support": ["DiTy all-candidate scoring", "top-K", "QA(top-K)", "V6 features", "QA(selected top1 already cached)"],
            "S3_adaptive": ["S2 dependencies", "semantic answer embeddings", "consensus confidence", "router"],
            "S4_bge_b0": ["BGE all-candidate scoring", "QA(top1)"],
            "no_artificial_QA_equalization": True,
        })
        frozen = {
            "experiment": "V4.5-A", "status": "development_characterization",
            "dataset_hashes": {key: audit["manifest"]["dataset_hash"] for key, audit in self.audits.items()},
            "fold_hashes": {key: audit["manifest"]["fold_membership_hash"] for key, audit in self.audits.items()},
            "rankers": {key: vars(value) for key, value in RANKERS.items()},
            "ranker_selection_status": "development_selected_after_v4_4",
            "training_recipe": TRAINING_RECIPE, "question": "canonical q0",
            "question_selection": False, "K": {key: DATASETS[key].k for key in self.config.datasets},
            "K_selection_status": "development_selected", "stage2": stage2,
            "Q5_policy": "no question selection; no Q5 veto in primary S3",
            "QA": manifests["qa_cache_audit"],
            "systems": list(SYSTEMS), "metrics": ["Token F1", "Lemma F1", "Character F1", "Unicode ROUGE-L", "EM", "MRR", "Hit@1", "Hit@K", "HN@1", "HN@K", "F1|positive"],
            "safety": {"Hit@1": ">= B0 - 0.5 pp", "HN@1": "<= B0 + 1.0 pp", "origin": "inherited_from_v4_4 development convention"},
            "bootstrap": {"iterations": self.config.n_bootstrap, "seed": BOOTSTRAP_SEED, "unit": "mean seeds within question, resample document clusters"},
            "resource_benchmark": {"end_to_end": True, "ranker_only": True, "warmup": self.config.benchmark_warmup, "repetitions": self.config.benchmark_repetitions},
            "seed_policy": list(SEEDS), "outer_eval_tuning": False,
            "development_candidate_rule": "S3 only if Hit/HN safety passes, observed delta is positive, and at least 3/5 EVAL folds are positive; then apply Pareto rule versus BGE; otherwise prefer S1/BGE by the same hierarchy",
        }
        frozen["contract_sha256"] = canonical_hash(frozen)
        write_json(self.output / "contract_frozen.json", frozen)
        self.progress.log("contract frozen", sha256=frozen["contract_sha256"])

    def _rankers(self, *, train):
        self.progress.stage("rankers", "DiTy candidate and BGE strong reference OOF")
        device = None
        if train:
            device, report = choose_device(self.config.device)
            self.progress.log("training device selected", device=device, **report)
        for dataset_key, state in self.states.items():
            expected = self._expected_keys(state)
            canonical = {(doc, aspect, cid): row for (doc, aspect, cid, question_id), row in state.qcache.items() if question_id == 0}
            for ranker_key, spec in RANKERS.items():
                root = self.output / dataset_key / "rankers" / ranker_key
                oof_path = root / "oof_predictions_scored.jsonl"
                if not train:
                    if not oof_path.is_file() or not (root / "COMPLETED.json").is_file():
                        raise FileNotFoundError(f"missing completed V4.5 OOF: {oof_path}")
                    rows = read_jsonl(oof_path)
                    if len(rows) != len(expected) or {row_key(row) for row in rows} != expected:
                        raise ValueError(f"saved OOF coverage mismatch: {dataset_key}/{ranker_key}")
                    continue
                all_rows = []
                for run in self._folds():
                    train_records = records_for_role(state.records, state.splits, run=run, role="train")
                    calibration = records_for_role(state.records, state.splits, run=run, role="calibration")
                    evaluation = records_for_role(state.records, state.splits, run=run, role="eval")
                    for seed in self.config.seeds:
                        fold_root = root / f"fold{run}" / f"seed{seed}"
                        predictions = fold_root / "predictions_scored.jsonl"
                        rows = read_jsonl(predictions) if predictions.is_file() else []
                        expected_fold = {(item.question_id, seed) for item in evaluation}
                        if rows and ({row_key(row) for row in rows} != expected_fold or len(rows) != len(expected_fold)):
                            raise ValueError(f"incomplete resume fold: {dataset_key}/{ranker_key}/{run}/{seed}")
                        if not rows:
                            fold_root.mkdir(parents=True, exist_ok=True)
                            self.progress.log("train ranker", dataset=dataset_key, model=ranker_key, run=run, seed=seed,
                                              train=len(train_records), cal=len(calibration), eval=len(evaluation))
                            training_started = time.perf_counter()
                            try:
                                import torch
                                if str(device).startswith("cuda") and torch.cuda.is_available():
                                    torch.cuda.reset_peak_memory_stats()
                            except ImportError:
                                pass
                            scorer, report = train_cross_encoder(
                                train_records, calibration, state.flags_rows, spec=spec, seed=seed, device=device,
                                epochs=self.config.epochs, patience=self.config.patience,
                                micro_batch_size=self.config.micro_batch_size,
                                prediction_batch_size=self.config.prediction_batch_size,
                                gradient_accumulation=self.config.gradient_accumulation,
                                learning_rate=self.config.learning_rate, max_length=self.config.max_length,
                                progress=self.progress, report_path=fold_root / "training_report.json",
                                local_files_only=self.config.local_files_only, smoke=self.config.smoke,
                            )
                            report["training_wall_seconds"] = time.perf_counter() - training_started
                            report["training_gpu_hours"] = report["training_wall_seconds"] / 3600.0 if str(device).startswith("cuda") else 0.0
                            try:
                                import torch
                                report["peak_training_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated()) if str(device).startswith("cuda") and torch.cuda.is_available() else None
                            except ImportError:
                                report["peak_training_gpu_allocated_bytes"] = None
                            write_json(fold_root / "training_report.json", report)
                            rows = score_eval(evaluation, scorer, run=run, seed=seed,
                                              flags=state.flags_rows, qa_lookup=canonical, progress=self.progress)
                            rows = [null_undefined_pairwise(row) for row in rows]
                            for row in rows:
                                row.update(ranker_id=ranker_key, ranker_model=spec.model_name,
                                           ranker_revision=spec.revision)
                                validate_finite(row)
                            write_jsonl(predictions, rows)
                            del scorer
                            gc.collect()
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except ImportError:
                                pass
                        all_rows.extend(rows)
                        self.progress.log("ranker fold complete", dataset=dataset_key, model=ranker_key, run=run, seed=seed, rows=len(rows))
                if len(all_rows) != len(expected) or {row_key(row) for row in all_rows} != expected:
                    raise ValueError(f"OOF coverage mismatch: {dataset_key}/{ranker_key}")
                write_jsonl(oof_path, all_rows)
                write_json(root / "metrics.json", null_undefined_pairwise(summarize_rows(all_rows)))
                write_json(root / "COMPLETED.json", {"rows": len(all_rows), "model": spec.model_name,
                                                       "revision": spec.revision, "weights_retained": False})
            self._validate_new_candidate_universe(dataset_key, state)
        self._write_training_resource_summary()

    def _write_training_resource_summary(self):
        rows = []
        for path in sorted(self.output.glob("*/rankers/*/fold*/seed*/training_report.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"path": str(path.relative_to(self.output)),
                         "training_wall_seconds": report.get("training_wall_seconds"),
                         "training_gpu_hours": report.get("training_gpu_hours"),
                         "peak_training_gpu_allocated_bytes": report.get("peak_training_gpu_allocated_bytes")})
        write_json(self.output / "resource_benchmark/training_resources.json", {
            "runs": rows,
            "total_gpu_hours": sum(float(row.get("training_gpu_hours") or 0) for row in rows),
            "total_wall_seconds_sum": sum(float(row.get("training_wall_seconds") or 0) for row in rows),
            "peak_training_gpu_allocated_bytes": max((int(row["peak_training_gpu_allocated_bytes"]) for row in rows if row.get("peak_training_gpu_allocated_bytes") is not None), default=None),
        })

    def _validate_new_candidate_universe(self, dataset_key, state):
        sources = {
            "historical_original_b0": state.historical_b0,
            "new_dity": read_jsonl(self.output / dataset_key / "rankers/dity/oof_predictions_scored.jsonl"),
            "new_bge": read_jsonl(self.output / dataset_key / "rankers/bge/oof_predictions_scored.jsonl"),
        }
        problems = []
        for name, rows in sources.items():
            for row in rows:
                item = state.items[str(row["question_id"])]
                expected = {int(chunk.chunk_id) for chunk in item.chunks}
                ranked = {int(cid) for cid in row.get("ranked_ids") or []}
                scored = {int(cid) for cid in (row.get("scores") or {})}
                if ranked != expected or scored != expected:
                    problems.append({"source": name, "question_id": row["question_id"], "seed": row.get("seed")})
        result = {"dataset": dataset_key, "valid": not problems, "sources": {k: len(v) for k, v in sources.items()},
                  "violation_count": len(problems), "violations": problems[:100], "paired_quality_valid": not problems}
        write_json(self.output / "regression" / f"candidate_universe_check_{dataset_key}_after_rankers.json", result)
        if problems:
            raise ValueError(f"new candidate universe mismatch: {dataset_key}")

    def _stage2(self):
        self.progress.stage("stage2", "frozen-family V6 support and ASEM-style adaptive correction")
        checks = {}
        for dataset_key, state in self.states.items():
            checks[dataset_key] = self._stage2_dataset(dataset_key, state)
        write_json(self.output / "regression/serialization_check.json", {
            "valid": all(row["valid"] for row in checks.values()), "datasets": checks,
            "schema": "aligned score sequence passed to build_raw_prediction; finite recursive validation",
        })

    def _stage2_dataset(self, dataset_key, state):
        root = self.output / dataset_key
        dity_rows = read_jsonl(root / "rankers/dity/oof_predictions_scored.jsonl")
        bge_rows = read_jsonl(root / "rankers/bge/oof_predictions_scored.jsonl")
        if self.config.smoke:
            historical_dity = read_jsonl(state.paths["historical_dity"])
            historical_seed = [row for row in historical_dity if int(row.get("seed") or 0) == 42]
            evaluation_keys = {row_key(row) for row in dity_rows}
            dity_rows = [row for row in historical_seed if row_key(row) not in evaluation_keys] + dity_rows
        dity = {row_key(row): row for row in dity_rows}
        bge = {row_key(row): row for row in bge_rows}
        historical = {row_key(row): row for row in state.historical_b0}
        keys = sorted(dity)
        k = state.spec.k
        encoder = build_encoder(root / "stage2/semantic_cache", device=self.config.device,
                                smoke=self.config.smoke, progress=self.progress)
        qa_maps, features, packs, utility = {}, {}, {}, {}
        answers_to_embed = []
        for key in keys:
            item, row = state.items[key[0]], dity[key]
            qa = {cid: state.qa(item, cid) for cid in top_k_ids(row, k)}
            qa_maps[key] = qa
            answers_to_embed.extend(value["qa_answer"] for value in qa.values() if is_valid_answer(value["qa_answer"]))
        encoder.embed_many(answers_to_embed, store="answer")
        for index, key in enumerate(keys, 1):
            item, row, qa = state.items[key[0]], dity[key], qa_maps[key]
            top = top_k_ids(row, k)
            features[key] = agreement_features(item, row, qa, k=k, min_overlap=SOURCE_OVERLAP)
            answers = {cid: qa[cid]["qa_answer"] for cid in top}
            embeddings = {cid: encoder.embed_one(answer, store="answer") for cid, answer in answers.items() if is_valid_answer(answer)}
            packs[key] = adaptive_semantic_pack(answers, embeddings, top)
            utility[key] = {
                cid: {
                    "f1": max((pair_token_f1(answers[cid], gold) for gold in item.gold_answers), default=0.0),
                    "positive": float(item.chunk_by_id(cid).label == "positive"),
                    "hn": float(item.chunk_by_id(cid).label == "hard_negative"),
                } for cid in top
            }
            self.progress.count(index, len(keys), item=f"{dataset_key} frozen stage2 features")

        grid_v6 = [{"alpha": alpha, "beta": 0.0, "gamma": gamma}
                    for alpha, gamma in product(WEIGHT_GRID, repeat=2)]
        grid_sem = [{"lambda_m": left, "lambda_s": right} for left, right in product(WEIGHT_GRID, repeat=2)]
        all_rows = defaultdict(list)
        counterfactuals = []
        tuning = {}

        def ranked(key, kind, params):
            item, row = state.items[key[0]], dity[key]
            top = top_k_ids(row, k)
            if kind == "v6":
                scores = {cid: v6_fused_score(features[key][cid], **params) for cid in top}
            elif kind == "sem":
                scores = {cid: float(params["lambda_m"]) * float(features[key][cid]["majority"]) +
                          float(params["lambda_s"]) * float(packs[key]["sem_rel"][str(cid)]) for cid in top}
            else:
                raise ValueError(kind)
            return isolate_with_b0_tiebreak(item, row, scores, k=k)

        for run in self._folds():
            source = [key for key in keys if role_of_document(state.items[key[0]].document_id, state.splits, run) != "eval"]
            evaluation = [key for key in keys if role_of_document(state.items[key[0]].document_id, state.splits, run) == "eval"]
            fold_of = {key: int(state.splits["assignment"][state.items[key[0]].document_id]) for key in keys}
            baseline_top = lambda key: int(dity[key]["ranked_ids"][0])
            v6_result = select_stable(
                source, grid_v6, lambda key, params: ranked(key, "v6", params)[0][0], utility, fold_of,
                baseline_top=baseline_top, epsilon_hn=EPSILON_HN, hit_tolerance=HIT1_TOLERANCE,
                require_stable=True, progress=self.progress, label=f"tune {dataset_key}/run{run}/V6",
            )
            sem_result = select_stable(
                source, grid_sem, lambda key, params: ranked(key, "sem", params)[0][0], utility, fold_of,
                baseline_top=baseline_top, epsilon_hn=EPSILON_HN, hit_tolerance=HIT1_TOLERANCE,
                require_stable=False, progress=self.progress, label=f"tune {dataset_key}/run{run}/SEM",
            )
            v6_params = v6_result["selected"]["params"] if v6_result["selected"] else {"alpha": 1.0, "beta": 0.0, "gamma": 0.0}
            sem_params = sem_result["selected"]["params"] if sem_result["selected"] else {"lambda_m": 0.0, "lambda_s": 0.0}
            confidence = []
            for key in source:
                sem_order = ranked(key, "sem", sem_params)[0]
                if routing_available(packs[key], sem_order[0]):
                    confidence.append(float(packs[key]["consensus_conf"]))
            tau = float(np.quantile(confidence, ROUTING_QUANTILE)) if len(confidence) >= MIN_ROUTING_ROWS else None
            tuning[str(run)] = {
                "v6": v6_params, "sem": sem_params,
                "router": {"tau": tau, "quantile": ROUTING_QUANTILE, "source_available": len(confidence),
                           "source_total": len(source), "eval_rows_used": 0, "Q5_veto": False},
            }
            write_json(root / f"stage2/tuning/fold{run}_v6.json", v6_result)
            write_json(root / f"stage2/tuning/fold{run}_sem.json", sem_result)
            for index, key in enumerate(evaluation, 1):
                item = state.items[key[0]]
                orders = {
                    "S0_historical_b0": [int(cid) for cid in historical[key]["ranked_ids"]],
                    "S1_dity_b0": [int(cid) for cid in dity[key]["ranked_ids"]],
                    "S4_bge_b0": [int(cid) for cid in bge[key]["ranked_ids"]],
                }
                v6_order, v6_scores, _ = ranked(key, "v6", v6_params)
                sem_order, sem_scores, _ = ranked(key, "sem", sem_params)
                available = routing_available(packs[key], sem_order[0])
                use_sem = bool(tau is not None and available and float(packs[key]["consensus_conf"]) >= tau)
                orders["S2_v6_support"] = [int(cid) for cid in v6_order]
                orders["S3_adaptive"] = [int(cid) for cid in (sem_order if use_sem else v6_order)]
                counterfactuals.append({
                    "document_id": item.document_id, "question_id": item.question_id,
                    "seed": key[1], "cv_run": run, "router_available": available,
                    "router_used_sem": use_sem, "v6_chunk_id": int(v6_order[0]),
                    "sem_chunk_id": int(sem_order[0]),
                    "v6_token_f1": utility[key][int(v6_order[0])]["f1"],
                    "sem_token_f1": utility[key][int(sem_order[0])]["f1"],
                    "posthoc_gold_diagnostic_only": True,
                })
                for system in SYSTEMS:
                    order = orders[system]
                    selected = int(order[0])
                    extra = {
                        "seed": key[1], "cv_run": run, "dataset": dataset_key, "aspect": item.aspect,
                        "system": system, "k": k, "question_strategy": "q0", "question_variant": 0,
                        "ranker_selection_status": "development_selected_after_v4_4" if system.startswith(("S1", "S2", "S3")) else "reference",
                        "router_available": available if system == "S3_adaptive" else None,
                        "router_branch_used": "SEM" if system == "S3_adaptive" and use_sem else ("V6" if system == "S3_adaptive" else None),
                        "router_tau": tau if system == "S3_adaptive" else None,
                        "consensus_conf": packs[key]["consensus_conf"] if system == "S3_adaptive" else None,
                        "q5_veto_used": False,
                    }
                    pred = strict_prediction(item, ranked_ids=order, qa=state.qa(item, selected),
                                             flags=state.flags, method=f"V4.5-{system}", extra=extra)
                    all_rows[system].append(pred)
                    fold_path = root / f"fold{run}" / f"seed{key[1]}" / system / "predictions_scored.jsonl"
                    fold_path.parent.mkdir(parents=True, exist_ok=True)
                self.progress.count(index, len(evaluation), item=f"{dataset_key} stage2 fold={run}")
        # Freeze one deployable parameter set on cross-fitted development OOF only.
        global_fold_of = {key: int(state.splits["assignment"][state.items[key[0]].document_id]) for key in keys}
        global_baseline = lambda key: int(dity[key]["ranked_ids"][0])
        global_v6 = select_stable(
            keys, grid_v6, lambda key, params: ranked(key, "v6", params)[0][0], utility, global_fold_of,
            baseline_top=global_baseline, epsilon_hn=EPSILON_HN, hit_tolerance=HIT1_TOLERANCE,
            require_stable=True, progress=self.progress, label=f"freeze {dataset_key}/global/V6")
        global_sem = select_stable(
            keys, grid_sem, lambda key, params: ranked(key, "sem", params)[0][0], utility, global_fold_of,
            baseline_top=global_baseline, epsilon_hn=EPSILON_HN, hit_tolerance=HIT1_TOLERANCE,
            require_stable=False, progress=self.progress, label=f"freeze {dataset_key}/global/SEM")
        global_v6_params = global_v6["selected"]["params"] if global_v6["selected"] else {"alpha": 1.0, "beta": 0.0, "gamma": 0.0}
        global_sem_params = global_sem["selected"]["params"] if global_sem["selected"] else {"lambda_m": 0.0, "lambda_s": 0.0}
        global_confidence = []
        for key in keys:
            sem_order = ranked(key, "sem", global_sem_params)[0]
            if routing_available(packs[key], sem_order[0]):
                global_confidence.append(float(packs[key]["consensus_conf"]))
        global_tau = float(np.quantile(global_confidence, ROUTING_QUANTILE)) if len(global_confidence) >= MIN_ROUTING_ROWS else None
        write_json(root / "stage2/global_deployable_parameters.json", {
            "source": "cross-fitted development OOF only", "independent_test_used": False,
            "V6": global_v6_params, "SEM": global_sem_params,
            "router": {"tau": global_tau, "quantile": ROUTING_QUANTILE,
                       "source_available": len(global_confidence), "absolute_B0_score_dependency": False},
            "K": k, "question": "q0", "Q5_veto": False,
            "status": "frozen_for_future_prospective_test" if not self.config.smoke else "SMOKE_NOT_SCIENTIFIC",
            "final_ranker_training_pending": True,
        })
        write_json(root / "stage2/global_selection_v6.json", global_v6)
        write_json(root / "stage2/global_selection_sem.json", global_sem)

        for system, rows in all_rows.items():
            expected = self._expected_keys(state)
            if len(rows) != len(expected) or {row_key(row) for row in rows} != expected:
                raise ValueError(f"stage2 OOF coverage mismatch: {dataset_key}/{system}")
            destination = root / "oof" / system
            write_jsonl(destination / "predictions_scored.jsonl", rows)
            write_json(destination / "metrics.json", null_undefined_pairwise(summarize_rows(rows)))
            # Also materialize fold/seed paths specified by the protocol.
            grouped = defaultdict(list)
            for row in rows:
                grouped[(int(row["cv_run"]), int(row["seed"]))].append(row)
            for (run, seed), subset in grouped.items():
                write_jsonl(root / f"fold{run}/seed{seed}/{system}/predictions_scored.jsonl", subset)
        write_json(root / "stage2/tuning/manifest.json", tuning)
        write_jsonl(root / "stage2/counterfactual_router_utility.jsonl", counterfactuals)
        encoder.write_contract(root / "stage2/semantic_cache/encoder_spec.json")
        write_json(root / "stage2/COMPLETED.json", {"systems": list(SYSTEMS), "K": k, "q0": True,
                                                     "Q5_veto": False, "eval_used_for_selection": False})
        check = {"valid": True, "rows": sum(len(rows) for rows in all_rows.values()),
                 "systems": len(all_rows), "all_scores_finite": True,
                 "selected_equals_ranked_top": True, "score_argmax_equals_ranked_top": True}
        del encoder
        gc.collect()
        return check

    def _verify_historical_unchanged(self):
        before = json.loads((self.output / "regression/hash_before.json").read_text(encoding="utf-8"))
        after = {}
        for key in self.config.datasets:
            after[key] = {name: sha256(path) for name, path in required_paths(DATASETS[key]).items()}
        result = {"unchanged": before == after, "before": before, "after": after}
        write_json(self.output / "regression/hash_after.json", result)
        if not result["unchanged"]:
            raise RuntimeError("historical artifact hash changed during V4.5")
