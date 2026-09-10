"""QA-based silver overlay pipeline. Isolated; reuses frozen extractive QA."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

from experiments.qa_chunk_selection.models import load_frozen_qa
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v2.cache import FeatureCache
from experiments.qa_chunk_selection.v2.hn_flags import hn_quality_rows, is_training_eligible_hn, flag_index
from experiments.qa_chunk_selection.v3.contract import LOCKED_SPLITS, POSTHOC_SPLITS
from experiments.qa_chunk_selection.v3.data import records_from_jsonl, split_unlocked, synthetic_records
from experiments.qa_chunk_selection.v3.freeze import stable_json_hash
from experiments.qa_chunk_selection.v3.split import build_cv_splits, records_for_role
from experiments.qa_chunk_selection.v3_1_prep.context import local_context

from .calibrate import MINING_SCORE_VERSION, mining_score, select_consistency_threshold, select_ownership_delta, select_stage_a_threshold
from .cluster import assign_clusters
from .config import AutoPrepConfig, v3_paths
from .contract import (
    ANNOTATION_VERSION,
    QUESTION_SET_VERSION,
    SILVER_RULE_VERSION,
    experiment_contract_v3_1_auto,
)
from .discourse import discourse_features
from .freeze import freeze_silver
from .gold_relation import gold_relation
from .io_guard import assert_isolated_output, load_json, read_jsonl, write_json, write_jsonl
from .mapping import map_silver, paraphrase_agreement
from .qa_runner import run_qa
from .qc import control_metrics, cv_inventory, file_sha256, locked_ids, readiness_checklist, silver_counts, validate_rows
from .questions import canonical_question, ownership_for, paraphrases_for, question_payload
from .silver_roles import overlay_index


class PipelineAutoPrep:
    def __init__(self, config: AutoPrepConfig) -> None:
        self.config = config
        self.output_dir = assert_isolated_output(Path(config.output_dir))
        self.progress = ProgressLogger(self.output_dir / "progress.jsonl")
        self.v3 = v3_paths(Path(config.v3_artifact_dir))
        self.records = ()
        self.unlocked = ()
        self.splits: dict[str, Any] = {}
        self.flags = []
        self.universe: list[dict[str, Any]] = []
        self.stage_a: list[dict[str, Any]] = []
        self.fold_thresholds: dict[int, dict[str, Any]] = {}
        self.stage_b_para: dict[tuple[str, int], dict[str, Any]] = {}
        self.stage_b_own: dict[tuple[str, int], dict[str, Any]] = {}
        self.discourse: dict[tuple[str, int], dict[str, Any]] = {}
        self.overlays: dict[int, list[dict[str, Any]]] = {}
        self.gates: dict[str, Any] = {}
        self.readiness: dict[str, Any] = {}
        self.bundle = None
        self.cache = FeatureCache(self.output_dir / "cache")
        self._chunk_index: dict[tuple[str, int], Any] = {}
        self.exit_code = 0
        self._handlers: dict[str, Callable[[], None]] = {
            "init": self.stage_init,
            "freeze-qa": self.stage_freeze_qa,
            "build-universe": self.stage_build_universe,
            "stage-a": self.stage_a_run,
            "calibrate": self.stage_calibrate,
            "stage-a-gate": self.stage_a_gate,
            "stage-b": self.stage_b_run,
            "discourse": self.stage_discourse,
            "map-silver": self.stage_map,
            "cluster": self.stage_cluster,
            "qc": self.stage_qc,
            "stability": self.stage_stability,
            "readiness": self.stage_readiness,
            "freeze": self.stage_freeze,
        }

    def run(self, stages: list[str] | None = None) -> int:
        names = list(stages or self.config.stages)
        self.progress.stage("start", f"output={self.output_dir} smoke={self.config.smoke}")
        self.progress.log("E0/V2/V3/manual prep are read-only", write_target=str(self.output_dir))
        for name in names:
            handler = self._handlers.get(name)
            if handler is None:
                raise ValueError(f"unknown stage: {name}")
            self.progress.stage(name)
            handler()
        self.progress.stage("done", f"cache_hit_rate={self.cache.stats().get('hit_rate')}")
        return self.exit_code

    def _silver(self) -> Path:
        path = self.output_dir / "annotation_qa_silver"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _reuse_silver(self) -> Path | None:
        raw = self.config.reuse_stage_a_dir
        if raw is None:
            return None
        root = Path(raw)
        silver = root / "annotation_qa_silver" if (root / "annotation_qa_silver").exists() else root
        if silver.resolve() == self._silver().resolve():
            raise ValueError("reuse_stage_a_dir must be a previous run, not the current output_dir")
        return silver

    def _ensure_qa(self) -> None:
        if self.config.smoke or self.config.skip_qa_models or self.bundle is not None:
            return
        self.progress.log("loading frozen extractive QA only (no cross-encoder)")
        self.bundle = load_frozen_qa(device=self.config.device, smoke=False)

    def _answer_fn(self):
        if self.config.smoke:
            from experiments.qa_chunk_selection.models import lexical_answer

            return lexical_answer
        self._ensure_qa()
        if self.bundle is None:
            from experiments.qa_chunk_selection.models import lexical_answer

            return lexical_answer
        return self.bundle.answer

    def _model_id(self) -> str:
        if self.config.smoke or self.bundle is None:
            return "lexical-smoke"
        return str(self.bundle.qa_model_name)

    def stage_init(self) -> None:
        write_json(self.output_dir / "contract.json", experiment_contract_v3_1_auto())
        write_json(self.output_dir / "config_resolved.json", self.config.to_dict())
        if self.config.smoke:
            self.progress.log("smoke: synthetic unlocked records")
            self.records = synthetic_records(n_docs=8)
            self.unlocked = tuple(item for item in split_unlocked(self.records) if item.positives)
            self.splits = build_cv_splits(self.unlocked, n_folds=5, seed=self.config.seed)
            self.flags = hn_quality_rows(self.unlocked, include_hn_gold_warning=False)
        else:
            if not self.v3["dataset"].exists() or not self.v3["splits"].exists():
                raise FileNotFoundError(f"V3 dataset/splits missing under {self.v3['root']}")
            self.progress.log("loading V3 dataset (read-only)")
            self.records = records_from_jsonl(self.v3["dataset"])
            self.unlocked = tuple(item for item in split_unlocked(self.records) if item.positives)
            self.splits = load_json(self.v3["splits"])
            self.flags = read_jsonl(self.v3["hn_flags"]) if self.v3["hn_flags"].exists() else hn_quality_rows(self.unlocked)
            write_json(
                self.output_dir / "source_hashes.json",
                {
                    "dataset": file_sha256(self.v3["dataset"]),
                    "splits": file_sha256(self.v3["splits"]),
                    "document_id_hash": self.splits.get("document_id_hash"),
                    "split_payload_hash": self.splits.get("split_payload_hash"),
                },
            )
        locked = [item for item in self.records if item.split in LOCKED_SPLITS]
        self.progress.log(
            "universe lock",
            n_unlocked=len(self.unlocked),
            n_internal_test=len(locked),
            n_docs=len({item.document_id for item in self.unlocked}),
        )

    def stage_freeze_qa(self) -> None:
        dest = self._silver()
        write_json(dest / "aspect_questions.json", question_payload())
        write_json(dest / "attribution_questions.json", {"ownership": question_payload()["ownership"], "version": QUESTION_SET_VERSION})
        write_json(
            dest / "rules.json",
            {
                "silver_rule_version": SILVER_RULE_VERSION,
                "r_min": self.config.r_min,
                "max_easy_pass_rate": self.config.max_easy_pass_rate,
                "gold_f1_threshold": self.config.gold_f1_threshold,
                "annotation_version": ANNOTATION_VERSION,
                "mining_score_version": MINING_SCORE_VERSION,
            },
        )
        reused_manifest = None
        reuse_root = self._reuse_silver()
        if reuse_root is not None and (reuse_root / "qa_model_manifest.json").exists():
            reused_manifest = load_json(reuse_root / "qa_model_manifest.json")
        need_qa = (not self.config.smoke) and (not self.config.skip_qa_models) and (not self.config.skip_qa_until_needed or reuse_root is None)
        if need_qa:
            self._ensure_qa()
        manifest = {
            "qa_model": (reused_manifest or {}).get("qa_model") or self._model_id(),
            "smoke": self.config.smoke,
            "eval_mode": True,
            "weights_unchanged": True,
            "cross_encoder_loaded": False,
            "sentence_encoder_loaded": False,
            "question_set_version": QUESTION_SET_VERSION,
            "silver_rule_version": SILVER_RULE_VERSION,
            "question_hash": stable_json_hash(question_payload()),
            "reused_stage_a": reuse_root is not None,
            "mining_score_version": MINING_SCORE_VERSION,
        }
        existing = dest / "qa_model_manifest.json"
        if existing.exists() and self.config.resume:
            prev = load_json(existing)
            if prev.get("question_hash") != manifest["question_hash"]:
                raise PermissionError("QA question set is frozen and differs. Use a new output directory.")
            self.progress.log("reusing frozen qa_model_manifest.json")
        else:
            write_json(existing, manifest)
            self.progress.log("QA manifest frozen", model=manifest["qa_model"])

    def stage_build_universe(self) -> None:
        if not self.unlocked:
            self.stage_init()
        queue = read_jsonl(self.v3["queue"]) if (not self.config.smoke and self.v3["queue"].exists()) else []
        historical = {(str(row["question_id"]), int(row["chunk_id"])) for row in queue}
        flags = flag_index(self.flags)
        rng = random.Random(self.config.seed)
        primary = []
        controls = {"control_positive": [], "control_hn": [], "control_easy": []}
        by_q = {item.question_id: item for item in self.unlocked}
        for item in self.unlocked:
            if item.split in LOCKED_SPLITS or item.split in POSTHOC_SPLITS:
                continue
            n_chunks = len(item.chunks)
            for chunk in item.chunks:
                ctx = local_context(item, chunk)
                base = {
                    "document_id": item.document_id,
                    "question_id": item.question_id,
                    "aspect": item.aspect,
                    "question": item.question,
                    "chunk_id": chunk.chunk_id,
                    "source_label": chunk.label,
                    "source_ambiguous_subtype": chunk.primary_class if chunk.label == "ambiguous" else None,
                    "chunk_text": chunk.text,
                    "section_title": chunk.section_title,
                    "position": chunk.position,
                    "n_chunks": n_chunks,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "gold_answers": list(item.gold_answers),
                    "previous_sentence": ctx.get("previous_sentence"),
                    "next_sentence": ctx.get("next_sentence"),
                    "in_historical_queue": (item.question_id, chunk.chunk_id) in historical,
                    "split": item.split,
                }
                if chunk.label == "ambiguous":
                    row = dict(base)
                    row["universe_role"] = "primary_ambiguous"
                    primary.append(row)
                elif chunk.label == "positive":
                    controls["control_positive"].append((item, chunk, base))
                elif chunk.label == "hard_negative" and is_training_eligible_hn(flags, item.question_id, chunk.chunk_id):
                    controls["control_hn"].append((item, chunk, base))
                elif chunk.label == "easy_negative":
                    controls["control_easy"].append((item, chunk, base))
        del by_q
        cap = 4 if self.config.smoke else int(self.config.control_per_class)
        universe = list(primary)
        sample_keys = set()
        for role, pool in controls.items():
            rng.shuffle(pool)
            for index, (_item, _chunk, base) in enumerate(pool):
                row = dict(base)
                row["universe_role"] = role
                row["stage_b_control_sample"] = index < cap
                universe.append(row)
                if index < cap:
                    sample_keys.add((str(row["question_id"]), int(row["chunk_id"])))
        del sample_keys
        if self.config.limit_candidates > 0:
            universe = universe[: int(self.config.limit_candidates)]
        self.universe = universe
        write_jsonl(self._silver() / "universe.jsonl", universe)
        self.progress.log(
            "universe ready",
            n_ambiguous=sum(1 for row in universe if row["universe_role"] == "primary_ambiguous"),
            n_control=sum(1 for row in universe if str(row["universe_role"]).startswith("control_")),
        )

    def _qa(self, question: str, row: dict[str, Any], chunk) -> dict[str, Any]:
        return run_qa(
            self._answer_fn(),
            question,
            chunk,
            cache=self.cache,
            model_id=self._model_id(),
            inference_hash=SILVER_RULE_VERSION,
            qa_answerer=None if self.config.smoke or self.bundle is None else self.bundle.qa_answerer,
        )

    def _chunk(self, row: dict[str, Any]):
        if not self._chunk_index:
            self._chunk_index = {
                (item.question_id, chunk.chunk_id): chunk for item in self.unlocked for chunk in item.chunks
            }
        return self._chunk_index[(str(row["question_id"]), int(row["chunk_id"]))]

    def stage_a_run(self) -> None:
        if not self.universe:
            self.stage_build_universe()
        dest = self._silver() / "stage_a_all_ambiguous.jsonl"
        done = {}
        reuse = self._reuse_silver()
        if reuse is not None:
            src = reuse / "stage_a_all_ambiguous.jsonl"
            if not src.exists():
                raise FileNotFoundError(f"reuse Stage-A file missing: {src}")
            for row in read_jsonl(src):
                imported = dict(row)
                imported["reused_stage_a"] = True
                imported["cache_hit"] = True
                done[(str(row["question_id"]), int(row["chunk_id"]))] = imported
            write_json(
                self.output_dir / "analysis" / "reused_stage_a.json",
                {"source": str(src), "sha256": file_sha256(src), "n_imported": len(done)},
            )
            self.progress.log("imported Stage-A from previous run", n=len(done), source=str(src))
        if self.config.resume and dest.exists():
            for row in read_jsonl(dest):
                done[(str(row["question_id"]), int(row["chunk_id"]))] = row
            self.progress.log("resume stage-a", n_done=len(done))
        rows = list(done.values())
        pending = [row for row in self.universe if (str(row["question_id"]), int(row["chunk_id"])) not in done]
        total = len(self.universe)
        for index, row in enumerate(pending, start=len(done) + 1):
            question = canonical_question(row["question"], row["aspect"])
            payload = self._qa(question, row, self._chunk(row))
            gold = gold_relation(str(payload.get("qa_predicted_span") or ""), row.get("gold_answers") or [], f1_threshold=self.config.gold_f1_threshold)
            out = {
                **{key: row[key] for key in ("document_id", "question_id", "aspect", "chunk_id", "source_label", "source_ambiguous_subtype", "universe_role")},
                **payload,
                **gold,
                "canonical_question": question,
            }
            rows.append(out)
            self.progress.count(index, total, item="stage_a_qa")
        write_jsonl(dest, rows)
        self.stage_a = rows
        n_calls = sum(1 for row in rows if not row.get("cache_hit"))
        write_json(self._silver() / "stage_a_cost.json", {"n_rows": len(rows), "n_uncached_calls": n_calls})
        self.progress.log("stage A done", n=len(rows), uncached=n_calls)

    def stage_calibrate(self) -> None:
        if not self.stage_a:
            path = self._silver() / "stage_a_all_ambiguous.jsonl"
            if path.exists():
                self.stage_a = read_jsonl(path)
        if not self.unlocked:
            self.stage_init()
        by_key = {(str(row["question_id"]), int(row["chunk_id"])): row for row in self.stage_a}
        pareto_by_run = []
        for run in self.splits.get("runs") or []:
            run_id = int(run["run"])
            train = records_for_role(self.unlocked, self.splits, run=run_id, role="train")
            cal = records_for_role(self.unlocked, self.splits, run=run_id, role="calibration")
            train_raw, train_margin = _feature_pairs(train, by_key, "positive")
            if not train_raw:
                train_raw = [float(row["qa_raw_span_score"]) for row in self.stage_a if row.get("universe_role") == "control_positive"]
                train_margin = [float(row.get("qa_margin") or 0.0) for row in self.stage_a if row.get("universe_role") == "control_positive"]
            scored = []
            for row in self.stage_a:
                mined = mining_score(
                    float(row["qa_raw_span_score"]),
                    float(row.get("qa_margin") or 0.0),
                    train_raw,
                    train_margin,
                )
                scored.append({**row, **mined, "outer_run_id": run_id})
            tau = select_stage_a_threshold(
                [float(row["qa_plausibility_score"]) for row in scored if _role_of(row, self.unlocked) == "positive" and _in_records(row, cal)],
                [float(row["qa_plausibility_score"]) for row in scored if _role_of(row, self.unlocked) == "easy_negative" and _in_records(row, cal)],
                r_min=self.config.r_min if not self.config.smoke else 0.0,
                max_easy_pass=self.config.max_easy_pass_rate if not self.config.smoke else 1.0,
            )
            if self.config.smoke and not any(_in_records(row, cal) for row in scored):
                tau = {
                    "tau_qa": 0.0,
                    "cal_positive_recall": 1.0,
                    "cal_easy_pass": 1.0,
                    "ok": True,
                    "n_cal_pos": 0,
                    "reason": "smoke",
                    "selection_rule": "smoke",
                }
            for row in scored:
                row["qa_plausible"] = bool(row["qa_plausibility_score"] >= float(tau["tau_qa"]))
            run_dir = self._silver() / "oof" / f"run{run_id}"
            write_jsonl(run_dir / "stage_a_scored.jsonl", scored)
            write_json(
                run_dir / "thresholds.json",
                {**{key: value for key, value in tau.items() if key != "pareto"}, "train_n_pos": len(train_raw), "mining_score_version": MINING_SCORE_VERSION},
            )
            self.fold_thresholds[run_id] = {**tau, "scored": scored}
            pareto_by_run.append({"run": run_id, "pareto": tau.get("pareto") or [], "ok": tau.get("ok"), "reason": tau.get("reason")})
            self.progress.log(
                f"run {run_id} Stage-A threshold",
                tau=tau.get("tau_qa"),
                recall=tau.get("cal_positive_recall"),
                easy=tau.get("cal_easy_pass"),
                ok=tau.get("ok"),
                reason=tau.get("reason"),
            )
        write_json(self.output_dir / "analysis" / "stage_a_pareto.json", {"runs": pareto_by_run, "mining_score_version": MINING_SCORE_VERSION})
        candidates = []
        seen: set[tuple[str, int]] = set()
        for payload in self.fold_thresholds.values():
            for row in payload.get("scored") or []:
                key = (str(row["question_id"]), int(row["chunk_id"]))
                if not row.get("qa_plausible") or key in seen:
                    continue
                if row.get("universe_role") != "primary_ambiguous":
                    continue
                seen.add(key)
                candidates.append(row)
        write_jsonl(self._silver() / "stage_a_candidates.jsonl", candidates)
        self.progress.log("stage A candidates", n=len(candidates))

    def stage_a_gate(self) -> None:
        invalid = 0
        total = 0
        recalls = []
        easies = []
        for run_id, payload in self.fold_thresholds.items():
            recalls.append(float(payload.get("cal_positive_recall") or 0.0))
            easies.append(float(payload.get("cal_easy_pass") or 0.0))
            for row in payload.get("scored") or []:
                total += 1
                if not row.get("valid_span", True):
                    invalid += 1
        invalid_rate = (invalid / total) if total else 0.0
        items = {
            "min_positive_recall": all(value + 1e-12 >= (0.0 if self.config.smoke else self.config.r_min) for value in recalls) if recalls else self.config.smoke,
            "max_easy_pass": True if self.config.smoke else all(value <= self.config.max_easy_pass_rate for value in easies),
            "max_invalid_span": invalid_rate <= self.config.max_invalid_span_rate,
        }
        self.gates = {
            "items": items,
            "passed": all(items.values()),
            "invalid_span_rate": invalid_rate,
            "recalls": recalls,
            "easy_pass": easies,
            "mining_score_version": MINING_SCORE_VERSION,
        }
        write_json(self._silver() / "stage_a_gate.json", self.gates)
        if not self.gates["passed"]:
            self.progress.warn("Stage-A gate failed; Stage B will not run", **items)
            self.exit_code = 2
        else:
            self.progress.log("Stage-A gate passed")

    def stage_b_run(self) -> None:
        if not self.gates:
            self.stage_a_gate()
        if not self.gates.get("passed"):
            write_jsonl(self._silver() / "stage_b_paraphrases.jsonl", [])
            write_jsonl(self._silver() / "stage_b_ownership.jsonl", [])
            self.progress.warn("Stage B skipped because Stage-A gate failed")
            return
        keys = set()
        univ = {(str(row["question_id"]), int(row["chunk_id"])): row for row in self.universe}
        for payload in self.fold_thresholds.values():
            for row in payload.get("scored") or []:
                key = (str(row["question_id"]), int(row["chunk_id"]))
                meta = univ.get(key) or {}
                plausible_amb = bool(row.get("qa_plausible") and meta.get("universe_role") == "primary_ambiguous")
                sampled_control = bool(meta.get("stage_b_control_sample"))
                if plausible_amb or sampled_control:
                    keys.add(key)
        by_univ = {(str(row["question_id"]), int(row["chunk_id"])): row for row in self.universe}
        dest_p = self._silver() / "stage_b_paraphrases.jsonl"
        dest_o = self._silver() / "stage_b_ownership.jsonl"
        para_rows = []
        own_rows = []
        total = max(len(keys), 1)
        for index, key in enumerate(sorted(keys), start=1):
            row = by_univ.get(key)
            if row is None:
                continue
            chunk = self._chunk(row)
            paras = []
            for question in paraphrases_for(row["aspect"]):
                paras.append(self._qa(question, row, chunk))
            own = ownership_for(row["aspect"])
            target = self._qa(own["target"], row, chunk)
            external = self._qa(own["external"], row, chunk)
            para_rows.append({"question_id": row["question_id"], "chunk_id": row["chunk_id"], "paraphrases": paras})
            own_rows.append({"question_id": row["question_id"], "chunk_id": row["chunk_id"], "target": target, "external": external})
            self.stage_b_para[key] = {"paraphrases": paras}
            self.stage_b_own[key] = {"target": target, "external": external}
            self.progress.count(index, total, item="stage_b_qa")
        write_jsonl(dest_p, para_rows)
        write_jsonl(dest_o, own_rows)
        n_amb = sum(1 for row in self.universe if row["universe_role"] == "primary_ambiguous")
        write_json(
            self._silver() / "stage_b_cost.json",
            {
                "n_stage_b_chunks": len(keys),
                "n_ambiguous": n_amb,
                "stage_b_fraction": (len(keys) / n_amb) if n_amb else None,
            },
        )
        self.progress.log("stage B done", n=len(keys), fraction=(len(keys) / n_amb) if n_amb else None)

    def stage_discourse(self) -> None:
        if not self.universe:
            self.stage_build_universe()
        a_by = {(str(row["question_id"]), int(row["chunk_id"])): row for row in self.stage_a}
        rows = []
        for row in self.universe:
            key = (str(row["question_id"]), int(row["chunk_id"]))
            arow = a_by.get(key) or {}
            feats = discourse_features(
                row.get("chunk_text") or "",
                section_title=str(row.get("section_title") or ""),
                span_start=arow.get("qa_span_start"),
                span_end=arow.get("qa_span_end"),
                position=int(row.get("position") or 0),
                n_chunks=int(row.get("n_chunks") or 1),
            )
            payload = {"question_id": row["question_id"], "chunk_id": row["chunk_id"], **feats}
            self.discourse[key] = feats
            rows.append(payload)
        write_jsonl(self._silver() / "discourse_features.jsonl", rows)
        self.progress.log("discourse features", n=len(rows))

    def stage_map(self) -> None:
        if not self.fold_thresholds:
            self.stage_calibrate()
        if not self.discourse:
            self.stage_discourse()
        if not self.stage_b_para and (self._silver() / "stage_b_paraphrases.jsonl").exists():
            for row in read_jsonl(self._silver() / "stage_b_paraphrases.jsonl"):
                self.stage_b_para[(str(row["question_id"]), int(row["chunk_id"]))] = row
            for row in read_jsonl(self._silver() / "stage_b_ownership.jsonl"):
                self.stage_b_own[(str(row["question_id"]), int(row["chunk_id"]))] = row
        for run_id, payload in self.fold_thresholds.items():
            scored = payload.get("scored") or []
            cal = records_for_role(self.unlocked, self.splits, run=run_id, role="calibration")
            cal_ids = {item.question_id for item in cal}
            pos_agree = []
            pos_margin = []
            for row in scored:
                if row.get("source_label") != "positive" or row["question_id"] not in cal_ids:
                    continue
                key = (str(row["question_id"]), int(row["chunk_id"]))
                para = [item.get("qa_predicted_span") for item in (self.stage_b_para.get(key) or {}).get("paraphrases") or []]
                agree = paraphrase_agreement(para, str(row.get("qa_predicted_span") or ""))
                pos_agree.append(float(agree["answer_value_agreement"]))
                own = self.stage_b_own.get(key) or {}
                pos_margin.append(float((own.get("target") or {}).get("qa_raw_span_score") or 0) - float((own.get("external") or {}).get("qa_raw_span_score") or 0))
            tau_c = select_consistency_threshold(pos_agree, [])
            delta = select_ownership_delta(pos_margin)
            tau_qa = float(payload.get("tau_qa") or 0.0)
            threshold_payload = {"tau_qa": tau_qa, "tau_consistency": tau_c, "delta_ownership": delta, "run": run_id}
            threshold_hash = stable_json_hash(threshold_payload)
            qa_hash = self._model_id()
            mapped = []
            for row in scored:
                key = (str(row["question_id"]), int(row["chunk_id"]))
                para = [item.get("qa_predicted_span") for item in (self.stage_b_para.get(key) or {}).get("paraphrases") or []]
                agree = paraphrase_agreement(para, str(row.get("qa_predicted_span") or ""))
                own = self.stage_b_own.get(key) or {}
                target = own.get("target") or {}
                external = own.get("external") or {}
                disc = self.discourse.get(key) or {}
                gold = {k: row.get(k) for k in ("qa_gold_relation", "allowed_for_primary_te", "new_value_candidate")}
                decision = map_silver(
                    qa_plausible=bool(row.get("qa_plausible")),
                    consistency=float(agree["answer_value_agreement"]),
                    target_score=float(target.get("qa_raw_span_score") or 0.0),
                    external_score=float(external.get("qa_raw_span_score") or 0.0),
                    gold=gold,
                    discourse=disc,
                    paraphrase_spans=[str(item or "") for item in para],
                    canonical_span=str(row.get("qa_predicted_span") or ""),
                    target_span=str(target.get("qa_predicted_span") or ""),
                    external_span=str(external.get("qa_predicted_span") or ""),
                    tau_consistency=tau_c,
                    delta=delta,
                    far_from_threshold=float(row.get("qa_plausibility_score") or 0) - tau_qa >= 0.10,
                    near_threshold=abs(float(row.get("qa_plausibility_score") or 0) - tau_qa) < 0.05,
                )
                mapped.append(
                    {
                        **row,
                        **decision,
                        **disc,
                        "qa_paraphrase_agreement": agree["answer_value_agreement"],
                        "qa_target_probe_answer": target.get("qa_predicted_span"),
                        "qa_target_probe_score": target.get("qa_raw_span_score"),
                        "qa_external_probe_answer": external.get("qa_predicted_span"),
                        "qa_external_probe_score": external.get("qa_raw_span_score"),
                        "silver_rule_version": SILVER_RULE_VERSION,
                        "question_set_version": QUESTION_SET_VERSION,
                        "threshold_config_id": f"run{run_id}",
                        "threshold_config_hash": threshold_hash,
                        "qa_model_hash": qa_hash,
                        "outer_run_id": run_id,
                    }
                )
            self.fold_thresholds[run_id]["tau_consistency"] = tau_c
            self.fold_thresholds[run_id]["delta_ownership"] = delta
            thresholds = load_json(self._silver() / "oof" / f"run{run_id}" / "thresholds.json")
            thresholds.update({"tau_consistency": tau_c, "delta_ownership": delta})
            write_json(self._silver() / "oof" / f"run{run_id}" / "thresholds.json", thresholds)
            self.overlays[run_id] = mapped
            write_jsonl(self._silver() / "oof" / f"run{run_id}" / "silver_target_status.jsonl", mapped)
            self.progress.log(f"run {run_id} silver mapped", n=len(mapped), tau_c=tau_c, delta=delta)

    def stage_cluster(self) -> None:
        for run_id, rows in list(self.overlays.items()):
            clustered = assign_clusters(rows)
            self.overlays[run_id] = clustered
            write_jsonl(self._silver() / "oof" / f"run{run_id}" / "silver_target_status.jsonl", clustered)
            write_jsonl(
                self._silver() / "oof" / f"run{run_id}" / "silver_target_status_high.jsonl",
                [row for row in clustered if row.get("silver_confidence") == "high"],
            )
        if self.overlays:
            write_jsonl(self._silver() / "silver_target_status_all.jsonl", self.overlays[min(self.overlays)])
            write_jsonl(
                self._silver() / "silver_target_status_high.jsonl",
                [row for row in self.overlays[min(self.overlays)] if row.get("silver_confidence") == "high"],
            )
            write_jsonl(
                self._silver() / "silver_target_status_medium.jsonl",
                [row for row in self.overlays[min(self.overlays)] if row.get("silver_confidence") == "medium"],
            )
            new_vals = [
                {"question_id": row["question_id"], "chunk_id": row["chunk_id"], "value": row.get("qa_predicted_span")}
                for row in self.overlays[min(self.overlays)]
                if row.get("new_value_candidate")
            ]
            write_jsonl(self._silver() / "new_value_candidates.jsonl", new_vals)
        self.progress.log("clusters assigned", n_runs=len(self.overlays))

    def stage_qc(self) -> None:
        if not self.overlays:
            self.stage_cluster()
        locked = locked_ids(self.records or self.unlocked)
        all_rows = [row for rows in self.overlays.values() for row in rows]
        validation = validate_rows(all_rows, self.unlocked, locked)
        controls = control_metrics(self.overlays[min(self.overlays)], self.stage_a)
        counts = silver_counts(self.overlays[min(self.overlays)])
        inventory = cv_inventory(self.unlocked, self.splits, {run: overlay_index(rows) for run, rows in self.overlays.items()}, self.progress)
        write_json(self.output_dir / "analysis" / "validation.json", validation)
        write_json(self.output_dir / "analysis" / "qc_controls.json", controls)
        write_json(self._silver() / "qc_controls.jsonl", [row for row in self.overlays[min(self.overlays)] if str(row.get("universe_role")).startswith("control_")])
        write_json(self._silver() / "silver_summary.json", counts)
        write_json(self.output_dir / "analysis" / "cv_tm_inventory.json", inventory)
        self._qc_bundle = {"validation": validation, "controls": controls, "counts": counts, "inventory": inventory}
        self.progress.log("QC", tm=counts.get("n_high_silver_target_mismatch"), schema_ok=validation.get("ok"))

    def stage_stability(self) -> None:
        flips = []
        for run_id, rows in self.overlays.items():
            high = [row for row in rows if row.get("silver_confidence") == "high"]
            if not high:
                continue
            changed = 0
            for row in high:
                if abs(float(row.get("qa_ownership_margin") or 0.0)) < 0.15:
                    changed += 1
            flips.append(changed / len(high))
        payload = {"high_confidence_flip_proxy": (sum(flips) / len(flips)) if flips else None, "n_runs": len(flips)}
        write_json(self.output_dir / "analysis" / "stability.json", payload)
        self.progress.log("stability", **{k: v for k, v in payload.items() if v is not None})

    def stage_readiness(self) -> None:
        bundle = getattr(self, "_qc_bundle", {})
        n_amb = sum(1 for row in self.universe if row.get("universe_role") == "primary_ambiguous")
        self.readiness = readiness_checklist(
            validation=bundle.get("validation") or {},
            gates=self.gates,
            inventory=bundle.get("inventory") or {},
            counts=bundle.get("counts") or {},
            n_ambiguous=n_amb,
            n_stage_a=len(self.stage_a),
            freeze_ready=(self._silver() / "silver_manifest.json").exists(),
            min_train=1 if self.config.smoke else self.config.min_train_questions_with_TM_silver,
            qa_frozen=(self._silver() / "qa_model_manifest.json").exists(),
        )
        write_json(self.output_dir / "analysis" / "readiness.json", self.readiness)
        self.progress.log("readiness", allowed=self.readiness.get("v3_1_training_allowed"), blocking=self.readiness.get("blocking"))

    def stage_freeze(self) -> None:
        if not self.readiness:
            self.stage_readiness()
        if not self.readiness.get("v3_1_training_allowed") and not self.config.allow_partial_freeze:
            write_json(self.output_dir / "analysis" / "freeze_skipped.json", {"frozen": False, "blocking": self.readiness.get("blocking")})
            self.progress.warn("freeze skipped", blocking=self.readiness.get("blocking"))
            self.exit_code = 2
            return
        hashes = load_json(self.output_dir / "source_hashes.json") if (self.output_dir / "source_hashes.json").exists() else {}
        bundle = getattr(self, "_qc_bundle", {})
        qc_summary = {**self.gates, **bundle, "readiness": self.readiness, "cache": self.cache.stats()}
        write_json(self._silver() / "thresholds.json", {str(k): {kk: vv for kk, vv in v.items() if kk != "scored"} for k, v in self.fold_thresholds.items()})
        paths = freeze_silver(
            self.output_dir,
            qc_summary=qc_summary,
            manifest={
                "source_hashes": hashes,
                "cv_split_hash": self.splits.get("split_payload_hash"),
                "qa_model_manifest": load_json(self._silver() / "qa_model_manifest.json"),
                "fold_thresholds": {str(k): {kk: vv for kk, vv in v.items() if kk != "scored"} for k, v in self.fold_thresholds.items()},
                "n_runs": len(self.overlays),
                "allow_overwrite_freeze": False,
            },
        )
        write_json(self.output_dir / "analysis" / "freeze_paths.json", paths)
        self.progress.log("frozen QA-silver rules", **paths)


def _feature_pairs(records, by_key, label: str) -> tuple[list[float], list[float]]:
    raw: list[float] = []
    margin: list[float] = []
    for item in records:
        for chunk in item.chunks:
            if chunk.label != label:
                continue
            row = by_key.get((item.question_id, chunk.chunk_id))
            if row:
                raw.append(float(row["qa_raw_span_score"]))
                margin.append(float(row.get("qa_margin") or 0.0))
    return raw, margin


def _role_of(row, records) -> str | None:
    for item in records:
        if item.question_id == row["question_id"]:
            try:
                return item.chunk_by_id(int(row["chunk_id"])).label
            except KeyError:
                return None
    return None


def _in_records(row, records) -> bool:
    return any(item.question_id == row["question_id"] for item in records)
