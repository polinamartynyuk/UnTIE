"""Isolated Russian V4.4 replay over the completed answerable Q5/B0 artifact."""
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import math
import resource
import sys
import threading
import time

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v2.hn_flags import flag_index
from experiments.qa_chunk_selection.v3.data import records_from_jsonl, synthetic_records
from experiments.qa_chunk_selection.v3.split import build_cv_splits
from experiments.qa_chunk_selection.v4.evaluate import chunk_prediction
from experiments.qa_chunk_selection.v4.shortlist import scores_from_row, top_k_ids
from experiments.qa_chunk_selection.v4.v6_reader import agreement_features
from experiments.qa_chunk_selection.v4_1.evaluate import role_of_document, summarize_rows, transitions
from experiments.qa_chunk_selection.v4_1.ranking import isolate_shortlist_ranking, v6_fused_score
from experiments.qa_chunk_selection.v4_1.select import choose_v6_weights
from experiments.qa_chunk_selection.v4_2.features import zscore_map
from experiments.qa_chunk_selection.v4_3.features import b0_margin, semantic_pack, support_gain, support_rel
from experiments.qa_chunk_selection.v4_3.ranking import isolate_with_b0_tiebreak
from experiments.qa_chunk_selection.v4_3.text import cosine, is_valid_answer, qa_answer_text, query_answer_text

from .contract import (EPSILON_HN, K, LEX_FIXED, N_BOOTSTRAP, PROTOCOL, SEEDS,
                       SEM_FIXED, SOURCE_OVERLAP, STATUS, V6_GRID)
from .diagnostics import confidence_bins, error_bucket, grouped_rates, routing_audit
from .encoder import build_encoder
from .evaluate import paired_delta, row_key
from .evidence import local_evidence_sentence
from .io_guard import (claim_output, file_sha256, load_json, read_jsonl,
                       validate_input, write_json, write_jsonl)
from .metrics_ru import apply_unicode_rouge
from .routing import RouteFeature, available, calibrate, choose_prediction, support_veto
from .select import primary_gate, select_candidate


class Pipeline:
    def __init__(self, config):
        self.config = config
        self.input_dir = Path(validate_input(config.input_artifact)) if not config.smoke else Path(config.input_artifact)
        self.output_dir = claim_output(config.output_dir, config.input_artifact, smoke=config.smoke)
        self.progress = ProgressLogger(self.output_dir / "progress.jsonl")
        self.started = time.perf_counter()
        self.predictions, self.summaries = {}, {}
        self.qa_store, self.agreement, self.packs, self.fixed = {}, {}, {}, {}
        self.v6_weights, self.routes, self.thresholds = {}, {}, {}
        self.support_values, self.source_hashes = {}, {}
        self.timings = defaultdict(float)
        self.selection = None
        self.encoder = None

    def _mkdir(self, *parts):
        path = self.output_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _source(self, path):
        path = Path(path)
        self.source_hashes[str(path.resolve())] = file_sha256(path)
        return path

    def _record(self, name, rows, dest):
        rows = list(rows)
        self.predictions[name] = rows
        self.summaries[name] = summarize_rows(rows)
        write_jsonl(dest / "predictions_scored.jsonl", rows)
        write_json(dest / "metrics.json", self.summaries[name])

    def run(self):
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(30):
                self.progress.log("still running")

        monitor = threading.Thread(target=heartbeat, daemon=True)
        monitor.start()
        stages = ("init", "cache", "features", "v6_weights", "thresholds", "replay",
                  "compare", "support", "diagnostics", "freeze", "report")
        try:
            for stage in stages:
                self.progress.stage(stage)
                started = time.perf_counter()
                getattr(self, f"stage_{stage}")()
                self.timings[f"stage_{stage}_seconds"] = time.perf_counter() - started
                write_json(self.output_dir / "status.json", {"state": "running", "last_completed_stage": stage})
            write_json(self.output_dir / "COMPLETED.json", {"exit_code": 0, "experiment": "V4.4-RU"})
            write_json(self.output_dir / "status.json", {"state": "completed", "stage": "done", "exit_code": 0})
            self.progress.stage("done", "exit=0")
            return 0
        except Exception as exc:
            self.progress.warn(f"FAILED: {type(exc).__name__}: {exc}")
            write_json(self.output_dir / "status.json", {"state": "failed", "error": str(exc), "exit_code": 1})
            raise
        finally:
            stop.set()
            monitor.join(timeout=1)

    def stage_init(self):
        write_json(self.output_dir / "contract.json", {**__import__(__package__ + '.contract', fromlist=['contract']).contract(),
                   "smoke": self.config.smoke, "support_requested": self.config.support,
                   "soft_requested": self.config.soft})
        write_json(self.output_dir / "config_resolved.json", self.config.to_dict())
        self._source(PROJECT_ROOT / PROTOCOL)
        if self.config.smoke:
            self.records = synthetic_records(n_docs=15)
            self.splits = build_cv_splits(self.records, n_folds=5, seed=42)
            self.b0_rows = []
            for seed in SEEDS:
                for item in self.records:
                    ids = [c.chunk_id for c in item.chunks]
                    eval_run = next(run for run in range(5)
                                    if role_of_document(item.document_id, self.splits, run) == "eval")
                    self.b0_rows.append({"question_id": item.question_id, "document_id": item.document_id,
                        "seed": seed, "cv_run": eval_run,
                        "ranked_ids": ids, "scores": {str(cid): float(len(ids)-i) for i,cid in enumerate(ids)}})
            self.flags = {}
        else:
            data = self._source(self.input_dir / "data/answerable_dataset.jsonl")
            split_path = self._source(self.input_dir / "data/answerable_cv_splits.json")
            b0_path = self._source(self.input_dir / "b0/oof_predictions_scored.jsonl")
            self.records = records_from_jsonl(data)
            self.splits = load_json(split_path)
            self.b0_rows = read_jsonl(b0_path)
            flags_path = self.input_dir / "data/hn_quality_flags.jsonl"
            self.flags = flag_index(read_jsonl(self._source(flags_path))) if flags_path.exists() else {}
        self.items = {item.question_id: item for item in self.records}
        if len(self.items) != len(self.records) or any(not item.positives for item in self.records):
            raise ValueError("dataset must contain unique answerable instances")
        self.b0 = {row_key(row): row for row in self.b0_rows}
        expected = {(qid, seed) for qid in self.items for seed in SEEDS}
        if set(self.b0) != expected or len(self.b0_rows) != len(expected):
            raise ValueError("B0 must cover every question with exactly seeds 42, 43, 44")
        if len(self.splits.get("runs", [])) != 5:
            raise ValueError("expected five document-grouped CV runs")
        for item in self.records:
            roles = [role_of_document(item.document_id, self.splits, run) for run in range(5)]
            if roles.count("eval") != 1 or roles.count("cal") != 1 or roles.count("train") != 3:
                raise ValueError(f"invalid CV roles for {item.document_id}")
        for row in self.b0_rows:
            expected_run = next(run for run in range(5) if role_of_document(row["document_id"], self.splits, run) == "eval")
            if int(row.get("cv_run", -1)) != expected_run:
                raise ValueError("B0 cv_run does not match the answerable CV split")
        write_json(self.output_dir / "source_hashes.json", self.source_hashes)
        self.progress.log(f"loaded {len(self.records)} answerable instances / {len(set(i.document_id for i in self.records))} documents; {len(self.b0_rows)} B0 rows")

    def stage_cache(self):
        needed = {(qid, cid) for (qid, _), row in self.b0.items() for cid in top_k_ids(row, K)}
        if self.config.smoke:
            for qid, cid in needed:
                chunk = self.items[qid].chunk_by_id(cid)
                answer = self.items[qid].gold_answers[0] if cid == 0 else chunk.text.split()[0]
                start = chunk.text.find(answer)
                if start < 0:
                    answer, start = chunk.text.split()[0], 0
                self.qa_store[qid, cid] = {"question_id": qid, "chunk_id": cid,
                    "qa_answer": answer, "qa_chunk_id": cid, "qa_chunk_start_char": start,
                    "qa_chunk_end_char": start + len(answer), "qa_span_score": 1.0,
                    "qa_forward_passes": 0, "qa_source": "smoke"}
        else:
            q5_path = self._source(self.input_dir / "q5/prediction_cache.jsonl")
            canonical = {}
            for row in read_jsonl(q5_path):
                if int(row["question_id"]) != 0:
                    continue
                key = (str(row["document_id"]), str(row["aspect"]), int(row["chunk_id"]))
                if key in canonical:
                    raise ValueError("duplicate canonical Q5 cache row")
                canonical[key] = row
            for qid, cid in sorted(needed):
                item, chunk = self.items[qid], self.items[qid].chunk_by_id(cid)
                source = canonical.get((item.document_id, item.aspect, cid))
                if source is None:
                    raise ValueError(f"missing canonical Q5 cache row: {qid}:{cid}")
                answer = str(source.get("qa_predicted_answer") or "")
                start, end = int(source.get("answer_start") or 0), int(source.get("answer_end") or 0)
                if answer and chunk.text[start:end] != answer:
                    found = chunk.text.find(answer)
                    if found < 0:
                        raise ValueError(f"Q5 answer/offset mismatch: {qid}:{cid}")
                    start, end = found, found + len(answer)
                self.qa_store[qid, cid] = {"question_id": qid, "chunk_id": cid,
                    "qa_answer": answer, "qa_chunk_id": cid, "qa_chunk_start_char": start,
                    "qa_chunk_end_char": end, "qa_start_logit": source.get("start_logit"),
                    "qa_end_logit": source.get("end_logit"),
                    "qa_span_score": float(source.get("start_logit") or 0) + float(source.get("end_logit") or 0),
                    "qa_confidence_proxy": source.get("qa_confidence"), "qa_forward_passes": 0,
                    "qa_source": "reused_ru_q5_canonical_q0"}
        if set(self.qa_store) != needed:
            raise ValueError("incomplete top-K QA cache")
        write_jsonl(self._mkdir("cache") / "qa_topk.jsonl", [self.qa_store[k] for k in sorted(self.qa_store)])
        write_json(self.output_dir / "cache/reuse.json", {"needed": len(needed), "hits": len(needed),
                   "new_qa_forwards": 0, "source": "Russian Q5 canonical q0 cache"})
        self.encoder = build_encoder(self._mkdir("semantic_cache"), device=self.config.device,
                                     smoke=self.config.smoke, progress=self.progress)
        answers = [qa_answer_text(row) for row in self.qa_store.values() if is_valid_answer(qa_answer_text(row))]
        self.encoder.embed_many(answers, store="answer")
        self.encoder.write_contract(self.output_dir / "semantic_cache/encoder_spec.json")
        self.progress.log(f"QA cache ready: {len(needed)}/{len(needed)}; zero new QA forwards")

    def stage_features(self):
        for index, (key, b0) in enumerate(sorted(self.b0.items()), 1):
            qid, _ = key
            item, top = self.items[qid], top_k_ids(b0, K)
            if len(top) != min(K, len(item.chunks)) or len(set(top)) != len(top):
                raise ValueError("invalid B0 top-K")
            qa = {cid: self.qa_store[qid, cid] for cid in top}
            feats = agreement_features(item, b0, qa, k=K, min_overlap=SOURCE_OVERLAP)
            self.agreement[key] = feats
            answers = {cid: qa_answer_text(qa[cid]) for cid in top}
            embeddings = {cid: self.encoder.embed_one(text, store="answer") for cid, text in answers.items() if is_valid_answer(text)}
            pack = semantic_pack(answers, embeddings, top_ids=top)
            pack["b0_margin"] = b0_margin([scores_from_row(b0)[cid] for cid in top])
            self.packs[key] = pack
            lex = {cid: LEX_FIXED["lambda_majority"] * feats[cid]["majority"] +
                   LEX_FIXED["lambda_token_f1"] * feats[cid]["mean_token_f1"] / 100 for cid in top}
            sem = {cid: SEM_FIXED["lambda_majority"] * feats[cid]["majority"] +
                   SEM_FIXED["lambda_semantic"] * pack["sem_rel"][str(cid)] for cid in top}
            for name, scores in (("LEX-FIX", lex), ("SEM-FIX", sem)):
                self.fixed[key, name] = isolate_with_b0_tiebreak(item, b0, scores, k=K)
            self.progress.count(index, len(self.b0), item="Russian fixed-reader features")
        write_json(self._mkdir("fixed_readers") / "sem_fix.json", SEM_FIXED)
        write_json(self.output_dir / "fixed_readers/lex_fix.json", LEX_FIXED)
        write_jsonl(self._mkdir("confidence") / "features.jsonl",
                    [{"question_id": qid, "seed": seed, **pack} for (qid, seed), pack in sorted(self.packs.items())])

    def _predict(self, item, scores, qa_by_chunk, ranked_ids, name="candidate"):
        # The ordinal map preserves the already resolved B0 tie-break exactly.
        ranked = [int(cid) for cid in ranked_ids]
        cid = ranked[0]
        rank_scores = {chunk_id: float(len(ranked) - index)
                       for index, chunk_id in enumerate(ranked)}
        pred = chunk_prediction(item, ranking_method=name, scores=rank_scores,
                                qa=qa_by_chunk.get(cid), flags=self.flags)
        pred["selected_chunk_id"] = cid
        pred["ranked_ids"] = ranked
        pred["fused_scores"] = {str(int(k)): float(v) for k, v in scores.items()}
        pred["selected_chunk_primary_class"] = item.chunk_by_id(cid).primary_class
        return apply_unicode_rouge(pred, item.gold_answers)

    def stage_v6_weights(self):
        b0_by_seed = {seed: {qid: self.b0[qid, seed] for qid in self.items} for seed in SEEDS}
        for run in range(5):
            cal_items = [item for item in self.records if role_of_document(item.document_id, self.splits, run) == "cal"]
            cal_rows = []
            for source in self.b0_rows:
                if role_of_document(source["document_id"], self.splits, run) != "cal":
                    continue
                qid, seed = str(source["question_id"]), int(source["seed"])
                item = self.items[qid]
                qa = {cid: self.qa_store[qid, cid] for cid in top_k_ids(source, K)}
                scored = self._predict(item, scores_from_row(source), qa,
                                       [int(x) for x in source["ranked_ids"]], "B0-CAL")
                scored.update(seed=seed, cv_run=run, aspect=item.aspect)
                cal_rows.append(scored)
            self.progress.log(f"run {run}: selecting V6 weights on CAL only ({len(cal_items)} questions x 3 seeds)")
            choice = choose_v6_weights(cal_items, b0_by_seed, self.qa_store, cal_rows, k=K,
                min_overlap=SOURCE_OVERLAP, alpha_grid=V6_GRID, beta_grid=V6_GRID,
                gamma_grid=V6_GRID, epsilon_hn=EPSILON_HN, predict_fn=self._predict,
                seeds=SEEDS, progress=self.progress)
            self.v6_weights[run] = {key: float(choice["selected"][key]) for key in ("alpha", "beta", "gamma")}
            write_json(self._mkdir("v6_weights", f"run{run}") / "selection.json", choice)
            self.progress.log(f"run {run}: V6={self.v6_weights[run]}")
        write_json(self.output_dir / "v6_weights/weights_by_run.json", {str(k): v for k,v in self.v6_weights.items()})

    def _v6_rank(self, key, run):
        scores = {cid: v6_fused_score(feat, **self.v6_weights[run]) for cid, feat in self.agreement[key].items()}
        return isolate_shortlist_ranking(self.items[key[0]], self.b0[key], scores, k=K)

    def _feature(self, key, run, reader):
        v6_ranked, _ = self._v6_rank(key, run)
        reader_id = self.fixed[key, reader][0][0]
        item, pack = self.items[key[0]], self.packs[key]
        return RouteFeature(item.question_id, item.document_id, key[1],
            role_of_document(item.document_id, self.splits, run), pack["consensus_conf"],
            available(pack, reader_id), reader_id, v6_ranked[0])

    def stage_thresholds(self):
        for run in range(5):
            for system, reader in (("ASEM", "SEM-FIX"), ("ALEX", "LEX-FIX")):
                features = [self._feature(key, run, reader) for key in sorted(self.b0)]
                threshold = calibrate(features)
                self.thresholds[run, system] = threshold
                self.routes[run, system] = {(f.question_id, f.seed): f for f in features}
                eval_keys = {(f.question_id, f.seed) for f in features if f.role == "eval"}
                if {tuple(x) for x in threshold["source_keys"]} & eval_keys:
                    raise ValueError("EVAL leakage in routing threshold")
                self.progress.log(f"run {run} {system}: Q70={threshold['tau']}; source={threshold['n_threshold_source_rows']}; enabled={threshold['enabled']}")
            write_json(self._mkdir("confidence", f"run{run}") / "tau70.json",
                       {name: self.thresholds[run, name] for name in ("ASEM", "ALEX")})

    def _score(self, key, run, name, ranked, scores, **extra):
        item, b0 = self.items[key[0]], self.b0[key]
        qa = {cid: self.qa_store[key[0], cid] for cid in top_k_ids(b0, K)}
        pred = self._predict(item, scores, qa, ranked, name)
        pred.update(seed=key[1], cv_run=run, aspect=item.aspect, **extra)
        expected = set(top_k_ids(b0, K))
        if set(pred["ranked_ids"][:len(expected)]) != expected:
            raise ValueError("shortlist isolation violated")
        return pred

    def stage_replay(self):
        results = {name: [] for name in ("B0-live", "V6-K5", "SEM-FIX", "LEX-FIX", "ASEM", "ALEX")}
        for run in range(5):
            eval_keys = [key for key in sorted(self.b0) if self.routes[run, "ASEM"][key].role == "eval"]
            for index, key in enumerate(eval_keys, 1):
                b0_scores = scores_from_row(self.b0[key])
                b0_ranked = [int(x) for x in self.b0[key]["ranked_ids"]]
                results["B0-live"].append(self._score(key, run, "B0-live", b0_ranked, b0_scores))
                v6_ranked, v6_scores = self._v6_rank(key, run)
                v6 = self._score(key, run, "V6-K5", v6_ranked, v6_scores)
                results["V6-K5"].append(v6)
                for system, reader in (("ASEM", "SEM-FIX"), ("ALEX", "LEX-FIX")):
                    ranked, scores, tied = self.fixed[key, reader]
                    pred = self._score(key, run, reader, ranked, scores, tie=tied)
                    results[reader].append(pred)
                    feature, threshold = self.routes[run, system][key], self.thresholds[run, system]
                    routed, used = choose_prediction(v6, pred, feature, threshold)
                    routed.update(ranking_method=system, routing_available=feature.available,
                        reader_only_used=used, consensus_conf=feature.confidence, tau70=threshold["tau"],
                        n_threshold_source_rows=threshold["n_threshold_source_rows"],
                        reader_chunk_id=feature.reader_id, v6_chunk_id=feature.v6_id,
                        reader_in_best_triple=feature.reader_id in self.packs[key]["best_triple"],
                        v6_answer=v6["qa_answer"], reader_answer=pred["qa_answer"],
                        v6_f1=v6["token_f1"], reader_f1=pred["token_f1"],
                        v6_source_label=v6["selected_chunk_label"], reader_source_label=pred["selected_chunk_label"],
                        b0_margin=self.packs[key]["b0_margin"])
                    routed["error_bucket"] = error_bucket(routed)
                    results[system].append(routed)
                self.progress.count(index, len(eval_keys), item=f"run {run} Russian EVAL")
        for name, rows in results.items():
            self._record(name, rows, self._mkdir(name.lower().replace("-", "_")))
        base = {row_key(r): r for r in results["B0-live"]}
        for name, rows in results.items():
            for row in rows:
                ref = base[row_key(row)]
                if any(abs(float(row[m])-float(ref[m])) > 1e-9 for m in ("hit@5", "hit@10", "hn@5")):
                    raise ValueError(f"top-K identity failed for {name}")
        write_json(self._mkdir("regression") / "top5_identity.json", {"ok": True, "n": len(base)})

    def _delta(self, left, right):
        self.progress.log(f"paired bootstrap: {right} minus {left}")
        return paired_delta(self.predictions[left], self.predictions[right],
                            n_boot=50 if self.config.smoke else N_BOOTSTRAP, seed=42)

    def stage_compare(self):
        delta = self._delta("V6-K5", "ASEM")
        write_json(self._mkdir("asem") / "bootstrap_vs_v6.json", delta)
        h1 = primary_gate(self.summaries["V6-K5"], self.summaries["ASEM"], delta)
        write_json(self.output_dir / "asem/development_gate.json", h1)
        secondary = self._delta("ASEM", "ALEX") if h1["pass"] else None
        write_json(self._mkdir("alex") / "bootstrap_vs_asem.json", secondary or {"skipped": True, "reason": "H1 failed"})
        self.selection = select_candidate(self.summaries["V6-K5"], self.summaries["ASEM"],
                                          self.summaries["ALEX"], h1, secondary)
        self.selection.update(status=STATUS, internal_test="not available", smoke=self.config.smoke)
        write_json(self._mkdir("final") / "development_selection.json", self.selection)
        self.progress.log(f"H1={h1['pass']}; preferred={self.selection['preferred']}")

    def _support(self, key):
        if key in self.support_values:
            return self.support_values[key]
        item, top = self.items[key[0]], top_k_ids(self.b0[key], K)
        query = self.encoder.embed_one(item.question, store="query")
        base, answer = {}, {}
        for cid in top:
            qa = self.qa_store[key[0], cid]
            text = qa_answer_text(qa)
            evidence = local_evidence_sentence(item.chunk_by_id(cid).text,
                start_char=qa.get("qa_chunk_start_char", 0), end_char=qa.get("qa_chunk_end_char", 0),
                predicted_answer=text, token_window=64)
            ev = self.encoder.embed_one(evidence["text"], store="local_evidence")
            qav = self.encoder.embed_one(query_answer_text(item.question, text), store="query_answer")
            base[cid], answer[cid] = cosine(query, ev), cosine(qav, ev)
        self.support_values[key] = support_rel(support_gain(base, answer, top))
        return self.support_values[key]

    def stage_support(self):
        if not self.selection["h1"]["pass"] or not self.config.support:
            reason = "H1 failed" if not self.selection["h1"]["pass"] else "disabled by CLI"
            write_json(self._mkdir("support_diagnostic") / "status.json", {"skipped": True, "reason": reason})
            return
        pref = self.selection["preferred"]
        baseline = {row_key(r): r for r in self.predictions["V6-K5"]}
        pref_rows = {row_key(r): r for r in self.predictions[pref]}
        out, thresholds = [], {}
        for run in range(5):
            features = list(self.routes[run, pref].values())
            potential = [f for f in features if f.role in {"train", "cal"} and f.available and f.reader_id != f.v6_id]
            calibrated = []
            if len(potential) >= 100:
                for index, feature in enumerate(potential, 1):
                    calibrated.append(replace(feature, support_rel=self._support((feature.question_id, feature.seed))[feature.reader_id]))
                    self.progress.count(index, len(potential), item=f"run {run} support")
            threshold = calibrate(calibrated, support=True)
            threshold["n_potential_source_rows"] = len(potential)
            thresholds[str(run)] = threshold
            for feature in features:
                if feature.role != "eval":
                    continue
                key = (feature.question_id, feature.seed)
                row = pref_rows[key]
                switched = bool(row["reader_only_used"] and feature.reader_id != feature.v6_id)
                if threshold["enabled"] and switched:
                    feature = replace(feature, support_rel=self._support(key)[feature.reader_id])
                veto = support_veto(feature, threshold, switched=switched)
                chosen = dict(baseline[key] if veto else row)
                chosen.update(ranking_method="PREF-SUPPORT", support_veto=veto,
                              support_rel=feature.support_rel, support_threshold=threshold["tau"],
                              potential_switch=switched, pref_token_f1=row["token_f1"],
                              veto_delta_f1=baseline[key]["token_f1"]-row["token_f1"] if veto else 0.0)
                out.append(chosen)
            self.progress.log(f"run {run} support: enabled={threshold['enabled']}")
        write_json(self.output_dir / "support_diagnostic/thresholds.json", thresholds)
        self._record("PREF-SUPPORT", out, self._mkdir("support_diagnostic", "pref_support"))
        write_json(self.output_dir / "support_diagnostic/pref_support/bootstrap_vs_pref.json", self._delta(pref, "PREF-SUPPORT"))
        write_json(self.output_dir / "support_diagnostic/status.json", {"skipped": False, "diagnostic_only": True})

    def stage_diagnostics(self):
        analysis = self._mkdir("analysis")
        write_json(analysis / "routing_rate.json", {s: grouped_rates(self.predictions[s]) for s in ("ASEM", "ALEX")})
        write_json(analysis / "routing_precision_recall.json", {s: routing_audit(self.predictions[s]) for s in ("ASEM", "ALEX")})
        write_json(analysis / "confidence_monotonicity.json", {s: confidence_bins(self.predictions[s]) for s in ("ASEM", "ALEX")})
        aspects = sorted({item.aspect for item in self.records})
        names = list(self.predictions)
        write_json(analysis / "aspect_metrics.json", {name: {aspect: summarize_rows([r for r in self.predictions[name] if r["aspect"] == aspect])
                   for aspect in aspects} for name in names})
        write_json(analysis / "seed_metrics.json", {name: {str(seed): summarize_rows([r for r in self.predictions[name] if r["seed"] == seed])
                   for seed in SEEDS} for name in names})
        write_json(analysis / "transitions.json", {f"V6-K5->{name}": transitions(self.predictions["V6-K5"], self.predictions[name])
                   for name in ("ASEM", "ALEX")})
        write_json(analysis / "metrics.json", self.summaries)

    def stage_freeze(self):
        for path in (PROJECT_ROOT / "experiments/qa_chunk_selection/v4_4_ru").glob("*.py"):
            self._source(path)
        write_json(self.output_dir / "source_hashes.json", self.source_hashes)
        write_json(self.output_dir / "final/deployable_freeze_manifest_draft.json", {
            "draft": True, "language": "ru", "development_candidate": self.selection["preferred"],
            "per_run_v6_weights": self.v6_weights, "per_run_thresholds": {
                str(run): {name: self.thresholds[run, name] for name in ("ASEM", "ALEX")} for run in range(5)},
            "source_hashes": self.source_hashes, "qa_forward_passes": 0,
            "note": "Cross-fitted development configuration only; no independent test was opened."})

    def stage_report(self):
        torch = sys.modules.get("torch")
        gpu = None
        if torch is not None and torch.cuda.is_initialized():
            gpu = {str(i): int(torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count())}
        efficiency = {"total_elapsed_seconds": time.perf_counter()-self.started,
            "stage_seconds": dict(self.timings),
            "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "peak_gpu_allocated_bytes": gpu, "encoder": self.encoder.stats(), "qa_forward_passes": 0}
        write_json(self._mkdir("analysis") / "efficiency.json", efficiency)
        from .report import write_report
        write_report(self.output_dir, self.selection, self.summaries, efficiency, sorted({i.aspect for i in self.records}))

