from __future__ import annotations

import gc
import json
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v4.answers import pair_token_f1
from experiments.qa_chunk_selection.v4.v6_reader import agreement_features

from .config import Config
from .contract import (BOOTSTRAP_SEED, BUDGETS, CELLS, FEATURE_SETS, GATE_CONTINUOUS,
                       INNER_FOLDS, OUTER_FOLDS, PRIMARY_BUDGET, SEEDS)
from .core import (feature_rows, fit_gate, fit_residual, gate_row, pair_rows,
                   runtime_choice, thresholds_for_budgets)
from .historical import HistoricalCell, tier
from .io_guard import (canonical_hash, read_jsonl, source_hashes, validate_finite,
                       write_json, write_jsonl)


def grouped(rows):
    out=defaultdict(list)
    for row in rows:out[(row["question_id"],int(row["seed"]))].append(row)
    for value in out.values():value.sort(key=lambda x:int(x["b0_rank"]))
    return out


def model_manifest(model):
    if hasattr(model,"p"):
        return {"kind":"constant","positive_probability":float(model.p)}
    classifier=model.named_steps["model"]
    return {"kind":"sklearn_pipeline","classes":[int(x) for x in classifier.classes_],
            "coefficients":[float(x) for x in classifier.coef_.ravel()],
            "intercept":[float(x) for x in classifier.intercept_.ravel()]}


class DatasetRun:
    def __init__(self,parent,key):
        self.parent=parent; self.config=parent.config; self.output=parent.output
        self.progress=parent.progress; self.spec=CELLS[key]
        self.cell=HistoricalCell(self.spec,self.output,device=self.config.device,smoke=self.config.smoke,
            local_files_only=self.config.local_files_only,progress=self.progress)
        self.root=self.output/key; self.root.mkdir(parents=True,exist_ok=True)

    def folds(self):return (0,) if self.config.smoke else OUTER_FOLDS

    def _features(self,records,scorer,question_map,seed,label):
        result=[]
        for index,item in enumerate(records,1):
            scores=self.cell.score(scorer,item)
            ranked=sorted(scores,key=lambda cid:(-scores[cid],cid))
            top=ranked[:min(self.spec.k,len(ranked))]
            question=self.cell.question(item,question_map)
            answers={cid:self.cell.qa(item,cid,question) for cid in top}
            relevance=self.cell.qa_relevance(scorer,question,answers)
            tiers={cid:tier(item.chunk_by_id(cid)) for cid in top}
            qa_payload={cid:{"qa_answer":answer} for cid,answer in answers.items()}
            agreement=agreement_features(item,{"ranked_ids":ranked,"scores":{str(k):v for k,v in scores.items()}},qa_payload,k=self.spec.k,min_overlap=.6)
            secondary={"source_support":{cid:agreement[cid]["source_clusters"] for cid in top},
                       "answer_group":{cid:agreement[cid]["exact_freq"] for cid in top}}
            result.extend(feature_rows(item.question_id,item.document_id,item.aspect,seed,top,scores,
                                       answers,relevance,tiers,len(item.chunks),secondary))
            self.progress.count(index,len(records),item=label)
        return result

    def _inner_plan(self,outer_run):
        source=self.cell.meta_source(outer_run)
        fold_ids=sorted({int(self.cell.splits["assignment"][x.document_id]) for x in source})
        if len(fold_ids)!=INNER_FOLDS:raise ValueError(f"{self.spec.key}: expected four META folds, got {fold_ids}")
        plan=[]
        for val_fold in fold_ids:
            remain=[x for x in fold_ids if x!=val_fold]; cal_fold=remain[-1]
            train=[x for x in source if int(self.cell.splits["assignment"][x.document_id]) in remain[:-1]]
            cal=[x for x in source if int(self.cell.splits["assignment"][x.document_id])==cal_fold]
            val=[x for x in source if int(self.cell.splits["assignment"][x.document_id])==val_fold]
            plan.append((val_fold,train,cal,val))
        return plan

    def meta_features(self,outer_run,seed):
        root=self.root/f"outer_fold_{outer_run}/seed_{seed}/inner_meta"
        complete=root/"COMPLETED.json"; path=root/"features.jsonl"
        if complete.is_file() and path.is_file():return read_jsonl(path)
        rows=[]; provenance=[]
        for val_fold,train,cal,val in self._inner_plan(outer_run):
            work=root/f"validation_fold_{val_fold}"
            self.progress.log("inner ranker training",cell=self.spec.key,outer_fold=outer_run,
                              inner_validation_fold=val_fold,seed=seed,train=len(train),calibration=len(cal),validation=len(val))
            scorer,report=self.cell.train_ranker(train,cal,seed,work)
            qmap=self.cell.question_map([*train,*cal])
            fold_rows=self._features(val,scorer,qmap,seed,f"{self.spec.key} outer={outer_run} inner={val_fold}")
            for row in fold_rows:row["inner_validation_fold"]=val_fold
            write_jsonl(work/"features.jsonl",fold_rows); rows.extend(fold_rows)
            provenance.append({"validation_fold":val_fold,"train_documents":sorted({x.document_id for x in train}),
                "calibration_documents":sorted({x.document_id for x in cal}),"validation_documents":sorted({x.document_id for x in val}),
                "question_map":qmap,"training_report":report})
            self.cell.release(scorer,work)
        expected={(str(x.question_id),seed) for x in self.cell.meta_source(outer_run)}
        actual=set(grouped(rows))
        if actual!=expected:raise ValueError(f"{self.spec.key}: META OOF coverage mismatch")
        write_jsonl(path,rows); write_json(root/"provenance.json",provenance)
        write_json(complete,{"rows":len(rows),"questions":len(actual),"outer_eval_documents_used":0})
        return rows

    def eval_features(self,outer_run,seed):
        root=self.root/f"outer_fold_{outer_run}/seed_{seed}/eval"
        complete=root/"FEATURES_COMPLETED.json"; path=root/"features.jsonl"
        if complete.is_file() and path.is_file():return read_jsonl(path)
        train=self.cell.outer_records(outer_run,"train"); cal=self.cell.outer_records(outer_run,"calibration")
        evaluation=self.cell.outer_records(outer_run,"eval")
        self.progress.log("outer ranker training",cell=self.spec.key,outer_fold=outer_run,seed=seed,
                          train=len(train),calibration=len(cal),evaluation=len(evaluation))
        scorer,report=self.cell.train_ranker(train,cal,seed,root/"outer_model")
        qmap=self.cell.question_map([*train,*cal],outer_run=outer_run if self.spec.question_policy=="nested_q5" else None)
        rows=self._features(evaluation,scorer,qmap,seed,f"{self.spec.key} outer={outer_run} EVAL")
        write_jsonl(path,rows); write_json(root/"question_policy.json",qmap)
        self.cell.retain_ranker(scorer,root/"outer_model")
        write_json(root/"training_report.json",report); write_json(complete,{"rows":len(rows),"questions":len(grouped(rows))})
        self.cell.release(scorer,root/"outer_model",delete_model=False)
        return rows

    def _prediction(self,item,feature_group,selected,system,outer_run,seed,gate_p,route,qa_forwards):
        top=feature_group[0]; answer=str(selected["qa_answer"])
        f1=max((pair_token_f1(answer,gold) for gold in item.gold_answers),default=0.0)
        all_ranked=sorted(item.chunks,key=lambda c:(-next((r["b0_score_raw"] for r in feature_group if r["chunk_id"]==c.chunk_id),-1e99),c.chunk_id))
        positive_ranks=[i+1 for i,c in enumerate(all_ranked) if tier(c)=="sufficient_positive"]
        selected_chunk=item.chunk_by_id(int(selected["chunk_id"])); selected_tier=tier(selected_chunk)
        return {"cell":self.spec.key,"corpus":self.spec.corpus,"language":self.spec.language,
            "document_id":str(item.document_id),"question_id":str(item.question_id),"aspect":str(item.aspect),
            "outer_fold":outer_run,"seed":seed,"system":system,"selected_chunk_id":int(selected["chunk_id"]),
            "b0_chunk_id":int(top["chunk_id"]),"qa_answer":answer,"token_f1":float(f1),
            "mrr":float(1/min(positive_ranks)) if positive_ranks else 0.0,
            "hit@1":float(selected_tier=="sufficient_positive"),
            "hit@k":float(any(r["label_tier"]=="sufficient_positive" for r in feature_group)),
            "hn@1":float(str(selected_chunk.label)=="hard_negative"),
            "hn@k":float(any(str(item.chunk_by_id(r["chunk_id"]).label)=="hard_negative" for r in feature_group)),
            "amb@1":float(str(selected_chunk.label)=="ambiguous"),
            "gsrn@1":float(str(selected_chunk.primary_class)=="gold_supporting_robust_negative"),
            "f1_given_selected_positive":float(f1) if selected_tier=="sufficient_positive" else None,
            "selected_tier":selected_tier,"gate_probability":float(gate_p),"routed":bool(route),
            "changed_top1":int(selected["chunk_id"])!=int(top["chunk_id"]),
            "switch_probability":float(selected.get("switch_probability",0.0)),
            "effective_k":len(feature_group),"candidate_count":int(top["candidate_count"]),
            "qa_forwards":int(qa_forwards),"extra_ranker_forwards":int(max(qa_forwards-1,0))}

    def execute(self,outer_run,seed):
        root=self.root/f"outer_fold_{outer_run}/seed_{seed}"
        complete=root/"COMPLETED.json"
        if complete.is_file():
            self.progress.log("completed outer seed preserved",cell=self.spec.key,outer_fold=outer_run,seed=seed); return
        meta=self.meta_features(outer_run,seed); evaluation=self.eval_features(outer_run,seed)
        meta_groups=grouped(meta); eval_groups=grouped(evaluation)
        gates=[gate_row(x) for x in meta_groups.values()]
        gate=fit_gate(gates,seed)
        import pandas as pd
        gate_p=gate.predict_proba(pd.DataFrame(gates))[:,1]
        thresholds=thresholds_for_budgets(gate_p,[x["effective_k"] for x in gates])
        residuals={name:fit_residual([p for g in meta_groups.values() for p in pair_rows(g,name)],seed,name) for name in FEATURE_SETS}
        write_json(root/"gate/model.json",model_manifest(gate)|{"features":list(GATE_CONTINUOUS),"thresholds":thresholds})
        import joblib
        (root/"gate").mkdir(parents=True,exist_ok=True); joblib.dump(gate,root/"gate/model.joblib")
        for name,model in residuals.items():
            write_json(root/f"residual/{name}/model.json",model_manifest(model)|{"features":list(FEATURE_SETS[name]),"switch_threshold":0.5})
            (root/f"residual/{name}").mkdir(parents=True,exist_ok=True); joblib.dump(model,root/f"residual/{name}/model.joblib")
        outputs=defaultdict(list)
        for key,rows in sorted(eval_groups.items()):
            item=self.cell.items[key[0]]; frame=pd.DataFrame([gate_row(rows)])
            probability=float(gate.predict_proba(frame)[0,1]); top=dict(rows[0],switch_probability=0.0)
            outputs["B0"].append(self._prediction(item,rows,top,"B0",outer_run,seed,probability,False,1))
            choices={}
            for ablation,model in residuals.items():
                chosen,switch_p=runtime_choice(rows,model,ablation); chosen=dict(chosen,switch_probability=switch_p); choices[ablation]=chosen
                outputs[f"V5-Q-{ablation}"].append(self._prediction(item,rows,chosen,f"V5-Q-{ablation}",outer_run,seed,probability,True,len(rows)))
            for budget in BUDGETS:
                route=bool(len(rows)>1 and probability>=thresholds[str(budget)])
                chosen=choices["A1"] if route else top
                name=f"V5-QG-A1-B{budget:.2f}"
                outputs[name].append(self._prediction(item,rows,chosen,name,outer_run,seed,probability,route,len(rows) if route else 1))
        for name,rows in outputs.items():write_jsonl(root/f"eval/{name}/predictions.jsonl",rows)
        write_json(root/"gate/meta_predictions.json",{"rows":gates,"probabilities":[float(x) for x in gate_p],"thresholds":thresholds})
        write_json(complete,{"status":"complete","cell":self.spec.key,"outer_fold":outer_run,"seed":seed,
                            "systems":sorted(outputs),"eval_questions":len(eval_groups)})

    def run(self):
        self.progress.stage(f"{self.spec.key}:run","nested META OOF and outer EVAL")
        for run in self.folds():
            for seed in self.config.seeds:self.execute(run,seed)
        rows=defaultdict(list)
        for path in sorted(self.root.glob("outer_fold_*/seed_*/eval/*/predictions.jsonl")):
            rows[path.parent.name].extend(read_jsonl(path))
        for name,value in rows.items():write_jsonl(self.root/f"oof/{name}/predictions.jsonl",value)
        violations=[]; checked=0
        for path in self.root.glob("outer_fold_*/seed_*/inner_meta/provenance.json"):
            for fold in json.loads(path.read_text()):
                train=set(fold["train_documents"]); cal=set(fold["calibration_documents"]); val=set(fold["validation_documents"]); checked+=1
                if train&cal or train&val or cal&val:violations.append(str(path))
        write_json(self.output/f"regression/meta_feature_leakage_check_{self.spec.key}.json",{"valid":not violations,"inner_folds_checked":checked,"violations":violations})
        if violations:raise RuntimeError(f"{self.spec.key}: META provenance leakage")
        write_json(self.root/"COMPLETED.json",{"status":"complete","cell":self.spec.key,"systems":sorted(rows)})


class Pipeline:
    def __init__(self,config:Config,output:Path):
        config.validate(); self.config=config; self.output=Path(output)
        self.progress=ProgressLogger(self.output/"progress.jsonl"); self.started=time.time(); self.audits={}

    def run(self):
        stop=threading.Event(); heartbeat=threading.Thread(target=self._heartbeat,args=(stop,),daemon=True); heartbeat.start()
        try:
            self.audit_and_freeze()
            if self.config.phase=="audit":return self.complete()
            if self.config.phase in {"run","all"}:
                for key in self.config.cells:DatasetRun(self,key).run()
            if self.config.phase in {"benchmark","all"}:
                if self.config.smoke:
                    write_json(self.output/"resource_benchmark/SKIPPED.json",{"reason":"smoke ranker is TF-IDF and has no neural checkpoint; full run benchmarks actual models"})
                else:
                    from .benchmark import run_benchmarks
                    run_benchmarks(self.config,self.output,self.progress)
            if self.config.phase in {"analysis","all"}:
                from .analysis import analyze
                analyze(self.config,self.output,self.progress)
            self.verify_hashes(); return self.complete()
        except Exception as exc:
            write_json(self.output/"FAILED.json",{"type":type(exc).__name__,"error":str(exc)})
            self.progress.warn(f"FAILED {type(exc).__name__}: {exc}"); raise
        finally:stop.set(); heartbeat.join(1)

    def _heartbeat(self,stop):
        while not stop.wait(30):self.progress.log("still running",elapsed_seconds=round(time.time()-self.started,1))

    def audit_and_freeze(self):
        self.progress.stage("audit","historical recovery, invariants and contract freeze")
        hashes={}; manifests={}
        for key in self.config.cells:
            self.progress.log("audit cell",cell=key)
            cell=HistoricalCell(CELLS[key],self.output,device=self.config.device,smoke=self.config.smoke,
                                local_files_only=self.config.local_files_only,progress=self.progress)
            manifest=cell.audit(); manifests[key]=manifest
            hashes[key]={**manifest["source_hashes"],"qa_model_tree":manifest["qa_reader"]["tree"]["sha256"],
                         "ranker_source_tree":(manifest["ranker_source"]["tree"] or {}).get("sha256")}
            write_json(self.output/f"inputs/model_manifests/{key}_models.json",{"ranker":manifest["ranker"],"qa_reader":manifest["qa_reader"]})
            write_json(self.output/f"inputs/question_manifests/{key}.json",{"policy":manifest["question_policy"],"outer_eval_labels_used":False})
            write_json(self.output/f"inputs/label_contracts/{key}.json",{"sufficient_positive":["positive","robust_positive","qa_positive"],
                "insufficient_evidence":["hard_negative","easy_negative","gold_supporting_robust_negative","question_sensitive"],
                "unresolved_for_training":["ambiguous","false_negative_candidate","evidence_positive_qa_failed"],"observed":manifest["label_tiers"]})
            write_json(self.output/f"inputs/feature_contract_{key}.json",{
                "primary_required":{"A0":list(FEATURE_SETS["A0"]),"A1":list(FEATURE_SETS["A1"])},
                "secondary":{"A2":{"available":True,"source":"v4.v6_reader.agreement_features","features":["source_clusters","exact_freq"]},
                             "A3":{"available":False,"reason":"no exact common historical SupportGain/confidence replay contract across all cells"}},
                "runtime_computable":True,"gold_features":False,"score_direction":"higher is better"})
        write_json(self.output/"inputs/datasets.json",manifests)
        write_json(self.output/"inputs/k_manifest.json",{k:CELLS[k].k for k in self.config.cells})
        write_json(self.output/"inputs/split_hashes.json",{k:v["split_hash"] for k,v in manifests.items()})
        write_json(self.output/"inputs/candidate_universe_hashes.json",{k:v["candidate_universe_hash"] for k,v in manifests.items()})
        write_json(self.output/"regression/hash_before.json",hashes)
        write_json(self.output/"regression/candidate_universe_check.json",{k:{"valid":True,"hash":v["candidate_universe_hash"]} for k,v in manifests.items()})
        write_json(self.output/"regression/model_identity_check.json",{k:{"valid":True,"ranker":v["ranker"],"qa_reader":v["qa_reader"]} for k,v in manifests.items()})
        write_json(self.output/"regression/qa_cache_audit.json",{k:{"historical_cache":str(CELLS[k].qa_cache.resolve()),"read_only":True,"new_cache_namespace":f"cache/qa/{k}.jsonl"} for k in self.config.cells})
        write_json(self.output/"regression/historical_metric_replay.json",{k:{"b0_predictions":str(CELLS[k].historical_b0.resolve()),"historical_reference":str(CELLS[k].historical_reference.resolve()),"status":"read-only reference; no re-scoring before freeze"} for k in self.config.cells})
        contract={"experiment":"V5-A","status":"development_characterization","cells":list(self.config.cells),
            "scier_cells_non_independent":True,"seeds":list(self.config.seeds),"outer_folds":list(OUTER_FOLDS),
            "inner_policy":"four-way document-fold cross-fitting inside outer TRAIN+CAL",
            "upstream":"dataset-trained historical B0; inner retraining mandatory",
            "gate":{"features":list(GATE_CONTINUOUS),"model":"scaled one-hot logistic C=1 balanced liblinear","target":"top1 insufficient and topK sufficient exists","budgets":list(BUDGETS)},
            "residual":{"A0":list(FEATURE_SETS["A0"]),"A1":list(FEATURE_SETS["A1"]),"A2":list(FEATURE_SETS["A2"]),"primary":"A1","model":"mirrored pair scaled one-hot logistic C=1 liblinear","threshold":0.5},
            "question_policy":"historical; SciER repeated nested-safely","K":{k:CELLS[k].k for k in self.config.cells},
            "previous_artifacts":"read_only","outer_eval_tuning":False,"bootstrap":{"unit":"document cluster after seed mean","iterations":self.config.n_bootstrap,"seed":BOOTSTRAP_SEED}}
        contract["contract_sha256"]=canonical_hash(contract)
        existing=self.output/"contract_frozen.json"
        if existing.is_file() and json.loads(existing.read_text())!=contract:raise RuntimeError("frozen contract mismatch")
        write_json(existing,contract)
        write_json(self.output/"reused_code/reused_components.json",{
            "components":[
                {"source":"v3.train.train_b0","historical":"EN B0","modified":False},
                {"source":"v4_4_ru_ranker_factorial.cross_encoder.train_cross_encoder","historical":"V4.5 DiTy","modified":False},
                {"source":"v4.answers.pair_token_f1","historical":"V4 metrics","modified":False},
                {"source":"v3.data.records_from_jsonl","historical":"all prepared datasets","modified":False}],"regression_status":"covered by V5 tests and audit"})
        write_json(self.output/"compute_plan.json",{"per_cell":{"outer_ranker_trainings":"5 folds x 3 seeds","inner_ranker_trainings":"5 outer x 3 seeds x 4 inner"},
            "total_ranker_trainings_full":len(self.config.cells)*5*3*5,"meta_model_fits_full":len(self.config.cells)*5*3*4,
            "note":"A0+A1 residual plus one gate; historical fold weights were not retained, so exact nested retraining is required"})

    def verify_hashes(self):
        from .historical import tree_hash
        tree_hash.cache_clear()
        before=json.loads((self.output/"regression/hash_before.json").read_text()); after={}
        for key in self.config.cells:
            manifest=HistoricalCell(CELLS[key],self.output,device="cpu",smoke=self.config.smoke).audit()
            after[key]={**manifest["source_hashes"],"qa_model_tree":manifest["qa_reader"]["tree"]["sha256"],
                        "ranker_source_tree":(manifest["ranker_source"]["tree"] or {}).get("sha256")}
        write_json(self.output/"regression/hash_after.json",after)
        if before!=after:raise RuntimeError("historical input changed during V5")

    def complete(self):
        payload={"status":"complete","phase":self.config.phase,"cells":list(self.config.cells),
                 "elapsed_seconds":time.time()-self.started,"scientific_status":"development_characterization"}
        write_json(self.output/f"PHASE_{self.config.phase}_COMPLETED.json",payload)
        if self.config.phase in {"all","analysis"}:write_json(self.output/"COMPLETED.json",payload)
        (self.output/"FAILED.json").unlink(missing_ok=True); self.progress.stage("done","exit=0"); return 0

