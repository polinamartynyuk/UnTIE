"""V4.4 orchestration; no previous experiment lifecycle is invoked."""
from collections import defaultdict
from dataclasses import replace
import math
import resource
import sys
import threading
import time

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v2.hn_flags import hn_quality_rows, flag_index
from experiments.qa_chunk_selection.v3.data import synthetic_records
from experiments.qa_chunk_selection.v3.split import build_cv_splits
from experiments.qa_chunk_selection.v4.shortlist import load_b0_oof, top_k_ids, scores_from_row
from experiments.qa_chunk_selection.v4.v6_reader import agreement_features
from experiments.qa_chunk_selection.v4_1.evaluate import role_of_document
from experiments.qa_chunk_selection.v4_1.ranking import v6_fused_score, isolate_shortlist_ranking
from experiments.qa_chunk_selection.v4_2.features import zscore_map
from experiments.qa_chunk_selection.v4_3.pipeline import PipelineV43
from experiments.qa_chunk_selection.v4_3.features import semantic_pack, support_gain, support_rel, b0_margin
from experiments.qa_chunk_selection.v4_3.ranking import isolate_with_b0_tiebreak
from experiments.qa_chunk_selection.v4_3.text import (
    qa_answer_text, is_valid_answer, cosine, query_answer_text,
)
from .contract import (contract, STATUS, SEEDS, K, SEM_FIXED, LEX_FIXED, V6_GLOBAL, FINAL_SEED,
                       N_BOOTSTRAP, PROTOCOL)
from .io_guard import (claim_output, load_json, read_jsonl, write_json, write_jsonl, file_sha256,
                       load_unlocked_snapshot)
from .encoder import ReplayEncoder
from .evidence import local_evidence_sentence
from .evaluate import paired_delta, identity, row_key, summarize_rows, transitions
from .routing import RouteFeature, available, calibrate, choose_prediction, support_veto
from .select import primary_gate, select_candidate
from .diagnostics import grouped_rates, routing_audit, confidence_bins, error_bucket


class PipelineV44:
    _predict_ok = PipelineV43._predict_ok
    _record = PipelineV43._record

    def __init__(self, config):
        self.config = config
        self.output_dir = claim_output(config.output_dir)
        self.progress = ProgressLogger(self.output_dir / "progress.jsonl")
        self.v3 = PROJECT_ROOT / "artifacts/qa_chunk_selection_v3"
        self.v41 = PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_1"
        self.v43 = PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_3"
        self.predictions, self.summaries = {}, {}
        self.qa_store, self.packs, self.agreement = {}, {}, {}
        self.fixed, self.routes, self.thresholds = {}, {}, {}
        self.support_values = {}
        self.source_hashes, self.timings = {}, defaultdict(float)
        self.selection = None
        self.encoder = None
        self.started = time.perf_counter()

    def _mkdir(self, *parts):
        path = self.output_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _source(self, path):
        self.source_hashes[str(path)] = file_sha256(path)
        return path

    def run(self):
        stop = threading.Event()
        def heartbeat():
            while not stop.wait(30):
                self.progress.log("still running")
        monitor = threading.Thread(target=heartbeat, daemon=True)
        monitor.start()
        stages = ("init", "references", "cache", "features", "thresholds", "replay", "compare",
                  "support", "diagnostics", "freeze", "report")
        try:
            for stage in stages:
                self.progress.stage(stage)
                started = time.perf_counter()
                getattr(self, f"stage_{stage}")()
                self.timings[f"stage_{stage}_seconds"] = time.perf_counter() - started
                write_json(self.output_dir / "status.json", {"stage": stage, "state": "completed", "smoke": self.config.smoke})
            self.progress.stage("done", "exit=0; INTERNAL_TEST remains locked")
            write_json(self.output_dir / "status.json", {"stage": "done", "state": "completed", "exit_code": 0})
            return 0
        except Exception as exc:
            self.progress.warn(f"FAILED: {type(exc).__name__}: {exc}")
            write_json(self.output_dir / "status.json", {"state": "failed", "error": str(exc), "exit_code": 1})
            raise
        finally:
            stop.set()
            monitor.join(timeout=1)

    def stage_init(self):
        write_json(self.output_dir / "contract.json", {**contract(), "smoke": self.config.smoke,
                   "support_requested": self.config.support, "soft_requested": self.config.soft})
        write_json(self.output_dir / "config_resolved.json", self.config.to_dict())
        self._source(PROJECT_ROOT / PROTOCOL)
        if self.config.smoke:
            self.unlocked = synthetic_records(n_docs=12)
            self.splits = build_cv_splits(self.unlocked, n_folds=5, seed=42)
            self.b0_rows = []
            for seed in SEEDS:
                for item in self.unlocked:
                    ids = [c.chunk_id for c in item.chunks]
                    self.b0_rows.append({"question_id": item.question_id, "document_id": item.document_id,
                                        "seed": seed, "ranked_ids": ids,
                                        "scores": {str(cid): float(len(ids)-i) for i, cid in enumerate(ids)}})
            self.v6_weights = {run: {"alpha": 2., "beta": .25, "gamma": .5} for run in range(5)}
            self.historical_v6 = []
        else:
            self._source(self.v3 / "analysis/oof/b0/predictions_scored.jsonl")
            self.b0_rows = load_b0_oof(self.v3)
            self.splits = load_json(self._source(self.v3 / "data/v3_cv_splits.json"))
            self.unlocked = load_unlocked_snapshot(self._source(self.v3 / "data/dataset.jsonl"),
                                                   {r["question_id"] for r in self.b0_rows})
            expected = load_json(self._source(self.v43 / "source_hashes.json"))
            if (self.source_hashes[str(self.v3 / "data/dataset.jsonl")] != expected["dataset"]
                    or self.source_hashes[str(self.v3 / "data/v3_cv_splits.json")] != expected["splits"]):
                raise ValueError("Frozen V3 dataset/split hash mismatch")
            raw = load_json(self._source(self.v41 / "r1_v6_oof/weights_by_run.json"))
            self.v6_weights = {int(k): v for k, v in raw.items()}
            self.historical_v6 = read_jsonl(self._source(self.v41 / "r1_v6_oof/predictions_scored.jsonl"))
            if len(self.unlocked) != 453 or len(self.splits["runs"]) != 5 or self.splits["split_seed"] != 42:
                raise ValueError("Unexpected frozen development universe")
        self.items = {i.question_id: i for i in self.unlocked}
        self.b0 = {row_key(r): r for r in self.b0_rows}
        expected_keys = {(qid, seed) for qid in self.items for seed in SEEDS}
        if set(self.b0) != expected_keys or len(self.b0_rows) != len(expected_keys):
            raise ValueError("B0 coverage/seed identity mismatch")
        if set(self.v6_weights) != set(range(5)):
            raise ValueError("Missing per-run V6 weights; fallback weights prohibited")
        for item in self.unlocked:
            roles = [role_of_document(item.document_id, self.splits, run) for run in range(5)]
            if roles.count("eval") != 1 or roles.count("cal") != 1 or roles.count("train") != 3:
                raise ValueError("Invalid document-level role partition")
        self.flags = flag_index(hn_quality_rows(self.unlocked))
        write_json(self.output_dir / "source_hashes.json", self.source_hashes)
        self.progress.log(f"loaded {len(self.unlocked)} unlocked questions; seeds={SEEDS}; locked records excluded")

    def stage_references(self):
        if self.config.smoke:
            return
        self._record("B0-live", self.b0_rows, self._mkdir("reference", "b0_live"))
        self._record("V6-K5-historical", self.historical_v6, self._mkdir("reference", "v6_k5_oof"))
        previous = read_jsonl(self._source(self.v43 / "reference/v6_k5_v4_1/predictions_scored.jsonl"))
        check = identity(previous, self.historical_v6)
        write_json(self._mkdir("regression") / "historical_v6_identity.json", check)
        if not check["ok"]:
            raise ValueError("V4.1/V4.3 V6 identity failed")
        for source, dest in (("semcore", "semcore_v4_3"), ("lexcore", "lexcore_v4_3"),
                             ("adapt1", "adapt1_v4_3_historical")):
            rows = read_jsonl(self._source(self.v43 / source / "predictions_scored.jsonl"))
            self._record(dest, rows, self._mkdir("reference", dest))
        frozen = load_json(self._source(self.v41 / "final/freeze_manifest.json"))
        if frozen.get("final_v6_weights_global_from_runs") != V6_GLOBAL or frozen.get("final_k") != K:
            raise ValueError("V4.1 global V6 freeze differs from protocol; amendment required")

    def stage_cache(self):
        needed = {(qid, int(cid)) for (qid, seed), row in self.b0.items() for cid in top_k_ids(row, K)}
        if self.config.smoke:
            for qid, cid in needed:
                chunk = self.items[qid].chunk_by_id(cid)
                answer = chunk.text.split()[0]
                self.qa_store[qid, cid] = {"qa_answer": answer, "qa_chunk_id": cid,
                    "qa_chunk_start_char": 0, "qa_chunk_end_char": len(answer), "qa_span_score": 1.0}
        else:
            rows = read_jsonl(self._source(self.v43 / "cache/qa_topk.jsonl"))
            for row in rows:
                key = row["question_id"], int(row["chunk_id"])
                if key in needed:
                    if key in self.qa_store:
                        raise ValueError("Duplicate QA cache entry")
                    self.qa_store[key] = row
        if set(self.qa_store) != needed:
            raise ValueError("Incomplete frozen V4.3 QA cache; refusing fresh or synthetic QA in real replay")
        write_json(self._mkdir("cache") / "reuse.json", {"needed": len(needed), "hits": len(needed),
                   "new_qa_forwards": 0, "source": "V4.3 exported V4.1–V4.3 QA cache", "smoke": self.config.smoke})
        self.encoder = ReplayEncoder(self.v43 / "semantic_cache", self._mkdir("support_diagnostic", "embeddings"),
                                     smoke=self.config.smoke, device=self.config.device)
        if not self.config.smoke:
            self._source(self.v43 / "semantic_cache/encoder_spec.json")
        self.progress.log(f"QA cache ready: {len(needed)}/{len(needed)}; no new QA forwards")

    def stage_features(self):
        for index, (key, b0) in enumerate(sorted(self.b0.items()), 1):
            qid, seed = key
            item, top = self.items[qid], top_k_ids(b0, K)
            if len(top) != min(K, len(item.chunks)) or len(set(top)) != len(top):
                raise ValueError("Invalid B0 top-5")
            qa = {cid: self.qa_store[qid, cid] for cid in top}
            started = time.perf_counter()
            feats = agreement_features(item, b0, qa, k=K, min_overlap=.6)
            self.agreement[key] = feats
            lex = {cid: .25 * feats[cid]["majority"] + 2. * feats[cid]["mean_token_f1"] / 100 for cid in top}
            self.timings["lex_feature_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            answers = {cid: qa_answer_text(qa[cid]) for cid in top}
            embeddings = {cid: self.encoder.embed_one(a, store="answer") for cid, a in answers.items() if is_valid_answer(a)}
            pack = semantic_pack(answers, embeddings, top_ids=top)
            pack["b0_margin"] = b0_margin([scores_from_row(b0)[cid] for cid in top])
            self.packs[key] = pack
            sem = {cid: .5 * feats[cid]["majority"] + .25 * pack["sem_rel"][str(cid)] for cid in top}
            self.timings["semantic_confidence_seconds"] += time.perf_counter() - started
            for name, scores in (("LEX-FIX", lex), ("SEM-FIX", sem)):
                ranked, fused, tie = isolate_with_b0_tiebreak(item, b0, scores, k=K)
                self.fixed[key, name] = (ranked, fused, tie)
            self.progress.count(index, len(self.b0), item="fixed reader features")
        write_json(self._mkdir("fixed_readers") / "sem_fix.json", SEM_FIXED)
        write_json(self.output_dir / "fixed_readers/lex_fix.json", LEX_FIXED)
        write_jsonl(self._mkdir("confidence") / "features.jsonl", [
            {"question_id": qid, "seed": seed, **pack} for (qid, seed), pack in self.packs.items()])

    def _v6_rank(self, key, run):
        started = time.perf_counter()
        scores = {cid: v6_fused_score(f, **self.v6_weights[run]) for cid, f in self.agreement[key].items()}
        result = isolate_shortlist_ranking(self.items[key[0]], self.b0[key], scores, k=K)
        self.timings["v6_ranking_seconds"] += time.perf_counter() - started
        self.timings["v6_ranking_calls"] += 1
        return result

    def _feature(self, key, run, reader):
        ranked, _ = self._v6_rank(key, run)
        reader_id = self.fixed[key, reader][0][0]
        item, pack = self.items[key[0]], self.packs[key]
        return RouteFeature(item.question_id, item.document_id, key[1],
                            role_of_document(item.document_id, self.splits, run), pack["consensus_conf"],
                            available(pack, reader_id), reader_id, ranked[0])

    def stage_thresholds(self):
        for run in range(5):
            for system, reader in (("ASEM", "SEM-FIX"), ("ALEX", "LEX-FIX")):
                features = [self._feature(key, run, reader) for key in sorted(self.b0)]
                threshold = calibrate(features)
                self.thresholds[run, system] = threshold
                self.routes[run, system] = {(f.question_id, f.seed): f for f in features}
                source_keys = {tuple(k) for k in threshold["source_keys"]}
                eval_keys = {(f.question_id, f.seed) for f in features if f.role == "eval"}
                if source_keys & eval_keys:
                    raise ValueError("EVAL leakage in routing threshold")
                self.progress.log(f"run {run} {system}: Q70={threshold['tau']}; source rows={len(source_keys)}; enabled={threshold['enabled']}")
            write_json(self._mkdir("confidence", f"run{run}") / "tau70.json",
                       {s: self.thresholds[run, s] for s in ("ASEM", "ALEX")})

    def _score(self, key, run, name, ranked, scores, **extra):
        item, b0 = self.items[key[0]], self.b0[key]
        qa = {cid: self.qa_store[key[0], cid] for cid in top_k_ids(b0, K)}
        pred = self._predict_ok(item, b0, scores, qa, ranked, name, extra, preserve_scores=True)
        pred.update(seed=key[1], cv_run=run, aspect=item.aspect,
                    selected_chunk_primary_class=item.chunk_by_id(ranked[0]).primary_class)
        if not pred["shortlist_ok"]:
            raise ValueError("Shortlist isolation violated")
        return pred

    def stage_replay(self):
        results = {name: [] for name in ("V6-K5", "SEM-FIX", "LEX-FIX", "ASEM", "ALEX")}
        for run in range(5):
            eval_keys = [key for key in sorted(self.b0) if self.routes[run, "ASEM"][key].role == "eval"]
            for index, key in enumerate(eval_keys, 1):
                ranks, scores = self._v6_rank(key, run)
                v6 = self._score(key, run, "V6-K5", ranks, scores)
                results["V6-K5"].append(v6)
                for system, reader in (("ASEM", "SEM-FIX"), ("ALEX", "LEX-FIX")):
                    ranks, scores, tied = self.fixed[key, reader]
                    pred = self._score(key, run, reader, ranks, scores, tie=tied)
                    results[reader].append(pred)
                    feature, threshold = self.routes[run, system][key], self.thresholds[run, system]
                    started = time.perf_counter()
                    routed, used = choose_prediction(v6, pred, feature, threshold)
                    self.timings[f"{system}_routing_seconds"] += time.perf_counter() - started
                    routed.update(ranking_method=system, routing_available=feature.available, reader_only_used=used,
                                  consensus_conf=feature.confidence, tau70=threshold["tau"],
                                  n_threshold_source_rows=threshold["n_threshold_source_rows"],
                                  reader_chunk_id=feature.reader_id, v6_chunk_id=feature.v6_id,
                                  reader_in_best_triple=feature.reader_id in self.packs[key]["best_triple"],
                                  v6_answer=v6["qa_answer"], reader_answer=pred["qa_answer"],
                                  v6_f1=v6["token_f1"], reader_f1=pred["token_f1"],
                                  v6_source_label=v6["selected_chunk_label"], reader_source_label=pred["selected_chunk_label"],
                                  b0_margin=self.packs[key]["b0_margin"])
                    routed["error_bucket"] = error_bucket(routed)
                    results[system].append(routed)
                self.progress.count(index, len(eval_keys), item=f"run {run} development EVAL")
            for name in ("ASEM", "ALEX"):
                write_jsonl(self._mkdir(name.lower(), f"run{run}") / "predictions.jsonl",
                            [r for r in results[name] if r["cv_run"] == run])
        for name, rows in results.items():
            dest = self._mkdir("reference", "v6_live") if name == "V6-K5" else self._mkdir(name.lower())
            self._record(name, rows, dest)
            write_jsonl(dest / "crossfit_predictions.jsonl", rows)
        check = identity(self.historical_v6, results["V6-K5"]) if self.historical_v6 else {"ok": True, "smoke": True}
        write_json(self._mkdir("regression") / "v6_per_run_identity.json", check)
        if not check["ok"]:
            raise ValueError("Recomputed V6 differs from historical per-run reference")
        baseline = {row_key(r): r for r in results["V6-K5"]}
        for rows in results.values():
            for row in rows:
                ref = baseline[row_key(row)]
                if any(abs(row[k] - ref[k]) > 1e-9 for k in ("hit@5", "hit@10", "hn@5")):
                    raise ValueError("Shortlist metric identity failed")
        write_json(self.output_dir / "regression/top5_identity.json", {"ok": True, "n_rows_per_system": len(baseline)})

    def _delta(self, left, right):
        self.progress.log(f"paired development bootstrap: {right} minus {left}")
        return paired_delta(self.predictions[left], self.predictions[right],
                            n_boot=50 if self.config.smoke else N_BOOTSTRAP, seed=42)

    def stage_compare(self):
        delta = self._delta("V6-K5", "ASEM")
        write_json(self.output_dir / "asem/bootstrap_vs_v6.json", delta)
        h1 = primary_gate(self.summaries["V6-K5"], self.summaries["ASEM"], delta)
        write_json(self.output_dir / "asem/development_gate.json", h1)
        secondary = self._delta("ASEM", "ALEX") if h1["pass"] else None
        write_json(self.output_dir / "alex/bootstrap_vs_asem.json", secondary or {"skipped": True, "reason": "H1 failed"})
        self.selection = select_candidate(self.summaries["V6-K5"], self.summaries["ASEM"],
                                          self.summaries["ALEX"], h1, secondary)
        self.selection.update(status=STATUS, internal_test="locked", smoke=self.config.smoke)
        write_json(self._mkdir("final") / "development_selection.json", self.selection)
        self.progress.log(f"H1 pass={h1['pass']}; preferred development candidate={self.selection['preferred']}")

    def _support(self, key):
        if key in self.support_values:
            return self.support_values[key]
        started = time.perf_counter()
        item, top = self.items[key[0]], top_k_ids(self.b0[key], K)
        query = self.encoder.embed_one(item.question, store="query")
        base, answer = {}, {}
        evidence_rows = []
        for cid in top:
            qa = self.qa_store[key[0], cid]
            text = qa_answer_text(qa)
            evidence = local_evidence_sentence(item.chunk_by_id(cid).text,
                start_char=qa.get("qa_chunk_start_char", 0), end_char=qa.get("qa_chunk_end_char", 0),
                predicted_answer=text, token_window=64)
            ev = self.encoder.embed_one(evidence["text"], store="local_evidence")
            qa_vector = self.encoder.embed_one(query_answer_text(item.question, text), store="query_answer")
            base[cid], answer[cid] = cosine(query, ev), cosine(qa_vector, ev)
            evidence_rows.append({"chunk_id": cid, "answer": text, "evidence": evidence,
                                  "base_fit": base[cid], "answer_fit": answer[cid]})
        gain = support_gain(base, answer, top)
        rel = support_rel(gain)
        self.support_values[key] = rel
        self.timings["support_feature_seconds"] += time.perf_counter() - started
        write_json(self._mkdir("support_diagnostic", "features") / f"{len(self.support_values):06d}.json",
                   {"question_id": key[0], "seed": key[1], "rows": evidence_rows,
                    "support_gain": gain, "support_rel": rel})
        return rel

    def stage_support(self):
        if not self.selection["h1"]["pass"]:
            write_json(self._mkdir("support_diagnostic") / "status.json", {"skipped": True, "reason": "H1 failed"})
            return
        pref = self.selection["preferred"]
        baseline = {row_key(r): r for r in self.predictions["V6-K5"]}
        if self.config.soft:
            soft_rows = []
            for row in self.predictions["ASEM"]:
                key, run = row_key(row), row["cv_run"]
                threshold = self.thresholds[run, "ASEM"]
                if not row["routing_available"] or not threshold["enabled"]:
                    pred = dict(baseline[key])
                else:
                    w = 1 / (1 + math.exp(max(-700., min(700., -10 * (row["consensus_conf"] - threshold["tau"])))))
                    top = top_k_ids(self.b0[key], K)
                    _, v6_scores = self._v6_rank(key, run)
                    sem_scores = self.fixed[key, "SEM-FIX"][1]
                    v6_z = zscore_map({cid: v6_scores[cid] for cid in top})
                    sem_z = zscore_map({cid: sem_scores[cid] for cid in top})
                    scores = {cid: (1-w)*v6_z[cid]+w*sem_z[cid] for cid in top}
                    ranked, fused, _ = isolate_with_b0_tiebreak(self.items[key[0]], self.b0[key], scores, k=K)
                    pred = self._score(key, run, "ASOFT", ranked, fused, soft_weight=w)
                pred["ranking_method"] = "ASOFT"
                soft_rows.append(pred)
            self._record("ASOFT", soft_rows, self._mkdir("asoft"))
            write_json(self.output_dir / "asoft/bootstrap_vs_asem.json", self._delta("ASEM", "ASOFT"))
        if not self.config.support:
            write_json(self._mkdir("support_diagnostic") / "status.json", {"skipped": True, "reason": "disabled by CLI before replay"})
            return
        pref_rows = {row_key(r): r for r in self.predictions[pref]}
        out, thresholds = [], {}
        for run in range(5):
            features = list(self.routes[run, pref].values())
            potential = [f for f in features if f.role in {"train", "cal"} and f.available and f.reader_id != f.v6_id]
            # Do not even load MiniLM for an underpopulated support calibration set.
            if len(potential) >= 100:
                calibrated = []
                for index, f in enumerate(potential, 1):
                    rel = self._support((f.question_id, f.seed))[f.reader_id]
                    calibrated.append(replace(f, support_rel=rel))
                    self.progress.count(index, len(potential), item=f"run {run} support calibration")
                threshold = calibrate(calibrated, support=True)
            else:
                threshold = calibrate([], support=True)
                threshold["n_potential_source_rows"] = len(potential)
            thresholds[str(run)] = threshold
            for f in features:
                if f.role != "eval":
                    continue
                key = f.question_id, f.seed
                row = pref_rows[key]
                switched = row["reader_only_used"] and f.reader_id != f.v6_id
                if threshold["enabled"] and switched:
                    f = replace(f, support_rel=self._support(key)[f.reader_id])
                vetoed = support_veto(f, threshold, switched=switched)
                chosen = dict(baseline[key] if vetoed else row)
                chosen.update(ranking_method="PREF-SUPPORT", pref=pref, support_veto=vetoed,
                              support_rel=f.support_rel, support_threshold=threshold["tau"],
                              potential_switch=switched, pref_token_f1=row["token_f1"],
                              veto_delta_f1=baseline[key]["token_f1"]-row["token_f1"] if vetoed else 0.)
                out.append(chosen)
            self.progress.log(f"run {run} support diagnostic: enabled={threshold['enabled']}")
        dest = self._mkdir("support_diagnostic", "pref_support")
        write_json(self.output_dir / "support_diagnostic/thresholds.json", thresholds)
        self._record("PREF-SUPPORT", out, dest)
        write_json(dest / "bootstrap_vs_pref.json", self._delta(pref, "PREF-SUPPORT"))
        vetoes = [r for r in out if r["support_veto"]]
        write_json(dest / "veto_audit.json", {"pref": pref, "diagnostic_only": True,
            "n_veto": len(vetoes), "veto_rate": len(vetoes)/max(len(out), 1),
            "veto_rate_among_switches": len(vetoes)/max(sum(r["potential_switch"] for r in out), 1),
            "good_veto": sum(r["veto_delta_f1"] > 1e-9 for r in vetoes),
            "bad_veto": sum(r["veto_delta_f1"] < -1e-9 for r in vetoes),
            "tie_veto": sum(abs(r["veto_delta_f1"]) <= 1e-9 for r in vetoes)})
        write_json(self.output_dir / "support_diagnostic/status.json", {"skipped": False, "diagnostic_only": True})

    def stage_diagnostics(self):
        analysis = self._mkdir("analysis")
        write_json(analysis / "routing_rate.json", {s: grouped_rates(self.predictions[s]) for s in ("ASEM", "ALEX")})
        write_json(analysis / "routing_precision_recall.json", {s: routing_audit(self.predictions[s]) for s in ("ASEM", "ALEX")})
        write_json(analysis / "confidence_monotonicity.json", {s: confidence_bins(self.predictions[s]) for s in ("ASEM", "ALEX")})
        write_json(analysis / "b0_margin_diagnostic.json", {s: confidence_bins(self.predictions[s], "b0_margin") for s in ("ASEM", "ALEX")})
        for run in range(5):
            write_json(self.output_dir / f"confidence/run{run}/bins.json",
                       {s: confidence_bins([r for r in self.predictions[s] if r["cv_run"] == run]) for s in ("ASEM", "ALEX")})
        for s in ("ASEM", "ALEX"):
            write_jsonl(self._mkdir("analysis", "error_buckets") / f"{s.lower()}.jsonl",
                        [r for r in self.predictions[s] if r["error_bucket"] is not None])
        names = [s for s in ("V6-K5", "SEM-FIX", "LEX-FIX", "ASEM", "ALEX", "PREF-SUPPORT", "ASOFT") if s in self.predictions]
        write_json(analysis / "aspect_metrics.json", {s: {aspect: summarize_rows([r for r in self.predictions[s] if r["aspect"] == aspect])
                   for aspect in ("material", "method", "metric", "task")} for s in names})
        write_json(analysis / "seed_metrics.json", {s: {str(seed): summarize_rows([r for r in self.predictions[s] if r["seed"] == seed])
                   for seed in SEEDS} for s in names})
        pairs = [("V6-K5", "ASEM"), ("V6-K5", "ALEX")]
        if "PREF-SUPPORT" in self.predictions:
            pairs.append((self.selection["preferred"], "PREF-SUPPORT"))
        write_json(analysis / "transitions.json", {f"{a}->{b}": transitions(self.predictions[a], self.predictions[b]) for a,b in pairs})
        write_json(self.output_dir / "fixed_readers/tie_rate.json",
                   {s: sum(bool(r.get("tie")) for r in self.predictions[s])/len(self.predictions[s]) for s in ("SEM-FIX", "LEX-FIX")})
        write_json(analysis / "metrics.json", {s: self.summaries[s] for s in names})

    def stage_freeze(self):
        from .freeze import build_draft
        # No final fitting or locked-test inference is performed by this package.
        pref = self.selection["preferred"]
        rows = self.predictions.get(pref, [])
        threshold_rows = [RouteFeature(r["question_id"], r["document_id"], r["seed"], "cal",
                         r.get("consensus_conf"), bool(r.get("routing_available")),
                         r.get("reader_chunk_id", -1), r.get("v6_chunk_id", -1))
                         for r in rows if r["seed"] == FINAL_SEED]
        global_threshold = calibrate(threshold_rows) if pref != "V6-K5" else None
        if global_threshold is not None:
            global_threshold.update(source_roles=["cross_fitted_eval"], source_seed=FINAL_SEED,
                                    source_note="Final global threshold from unlocked cross-fitted predictions; not a per-run threshold")
        # Record hashes for implementation and every source used, including cached vectors.
        for path in (PROJECT_ROOT / "experiments/qa_chunk_selection").rglob("*.py"):
            self._source(path)
        self.source_hashes.update(self.encoder.source_hashes)
        write_json(self.output_dir / "source_hashes.json", self.source_hashes)
        draft = build_draft(pref, global_threshold, self.source_hashes, smoke=self.config.smoke)
        write_json(self.output_dir / "final/deployable_freeze_manifest_draft.json", draft)

    def stage_report(self):
        from .report import write_report
        torch = sys.modules.get("torch")
        gpu = None
        if torch is not None and torch.cuda.is_initialized():
            gpu = {str(i): int(torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count())}
        n = max(len(self.b0), 1)
        lex = self.timings["lex_feature_seconds"] / n
        semantic = self.timings["semantic_confidence_seconds"] / n
        v6 = lex + self.timings["v6_ranking_seconds"] / max(self.timings["v6_ranking_calls"], 1)
        component_estimates = {"V6-K5": v6}
        for name in ("ASEM", "ALEX"):
            component_estimates[name] = v6 + semantic + self.timings[f"{name}_routing_seconds"] / n
        if "PREF-SUPPORT" in self.predictions:
            component_estimates["PREF-SUPPORT"] = component_estimates[self.selection["preferred"]] + self.timings["support_feature_seconds"] / max(len(self.support_values), 1)
        efficiency = {"stage_and_component_seconds": dict(self.timings),
                      "component_sum_seconds_per_question_estimates": component_estimates,
                      "component_estimate_scope": "shared cached features; excludes QA forward and metric/bootstrap costs; not a standalone latency benchmark",

                      "total_elapsed_seconds": time.perf_counter()-self.started,
                      "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      "peak_gpu_allocated_bytes": gpu, "encoder": self.encoder.stats(),
                      "qa_forward_passes": 0,
                      "timing_scope": "cached development replay, not end-to-end deployment latency"}
        write_json(self.output_dir / "analysis/efficiency.json", efficiency)
        write_report(self.output_dir, self.selection, self.summaries, efficiency)
