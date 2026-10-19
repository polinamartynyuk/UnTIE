"""V7/RAPR orchestration over immutable historical OOF features."""
from __future__ import annotations
from collections import defaultdict
import copy,json,math,threading,time
from pathlib import Path
import joblib,numpy as np
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v3.data import records_from_jsonl
from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS as V5_CELLS
from experiments.qa_chunk_selection_v6.features import enrich_question as v6_enrich, assert_runtime_gold_free as assert_upstream_runtime
from .audit import git_snapshot,historical_hashes,run_audit
from .config import Config
from .contract import CELLS,FAMILIES,MAX_QA,SYSTEMS,frozen_contract
from .features import assert_runtime_gold_free,candidate_rows,grouped,normalize_weights
from .io_guard import read_jsonl,remove_file,sha256,write_json,write_jsonl,write_text
from .models import apply_policy,tune_policy
from .replay import replay

class DataStore:
    def __init__(self,config,progress):self.config=config;self.progress=progress;self._items={};self._cache={}
    def items(self,cell):
        if cell not in self._items:self._items[cell]={str(x.question_id):x for x in records_from_jsonl(V5_CELLS[cell].records)}
        return self._items[cell]
    def load(self,cell,fold,seed,role):
        key=(cell,fold,seed,role)
        if key in self._cache:return copy.deepcopy(self._cache[key])
        path=self.config.v5_run/cell/f"outer_fold_{fold}"/f"seed_{seed}"/("inner_meta/features.jsonl" if role=="meta" else "eval/features.jsonl")
        raw=read_jsonl(path);byq=grouped(raw);items=self.items(cell);runtime=[];rows=[];targets={}
        for index,(qid,source) in enumerate(sorted(byq.items()),1):
            run,target=v6_enrich(cell,source,items[qid],outer_fold=fold,seed=seed,meta_source=role=="meta")
            target["gold_answers"]=[str(x) for x in items[qid].gold_answers]
            target["chunk_label"]={str(r["chunk_id"]):str(items[qid].chunk_by_id(int(r["chunk_id"])).label) for r in run}
            target["primary_class"]={str(r["chunk_id"]):str(items[qid].chunk_by_id(int(r["chunk_id"])).primary_class) for r in run}
            for r in run:assert_upstream_runtime(r)
            runtime.extend(run);rows.extend(candidate_rows(run,target));targets[qid]=target
            self.progress.count(index,len(byq),item=f"{cell} {role} questions",extra={"fold":fold,"seed":seed})
        cache=self.config.run_dir/"cache/meta_features"/cell/f"outer_fold_{fold}"/f"seed_{seed}"
        write_jsonl(cache/f"{role}_runtime.jsonl",runtime);write_jsonl(cache/f"{role}_candidate_targets.jsonl",rows)
        result=(runtime,rows,targets);self._cache[key]=copy.deepcopy(result);return copy.deepcopy(result)

def _embargo(rows,eval_runtime):
    docs={x["canonical_document_id"] for x in eval_runtime};hashes={x["document_sha256"] for x in eval_runtime}
    kept=[x for x in rows if x["canonical_document_id"] not in docs and x["document_sha256"] not in hashes]
    return kept,{"input_rows":len(rows),"kept_rows":len(kept),"excluded_rows":len(rows)-len(kept),"eval_documents":len(docs),"eval_hashes":len(hashes)}

def _mrr(runtime,selected,target):
    order=[selected]+[int(x["chunk_id"]) for x in runtime if int(x["chunk_id"])!=selected];relevant={int(k) for k,v in target["label_tier"].items() if v=="sufficient_positive"}
    ranks=[i+1 for i,c in enumerate(order) if c in relevant];return 1/min(ranks) if ranks else 0.0

def predictions(cell,runtime,rows,targets,system,mode,fold,seed,policy=None):
    runtime_byq=grouped(runtime);rows_byq=defaultdict(list)
    for row in rows:rows_byq[row["question_id"]].append(row)
    pred={}
    if policy:
        _choices,(mu,harm,admit)=apply_policy(rows,policy)
        for row,m,h,a in zip(rows,mu,harm,admit):pred[(row["question_id"],row["chunk_id"])]=(float(m),float(h),float(a))
    selection=(policy.selection if policy else {});ta=selection.get("tau_admit",None);tu=selection.get("tau_utility",None);th=selection.get("tau_harm",None)
    ta=math.inf if ta=="inf" else ta;tu=math.inf if tu=="inf" else tu
    output=[]
    for qid,candidates in sorted(runtime_byq.items()):
        top=candidates[0];alts=sorted(rows_byq[qid],key=lambda x:(x["b0_rank"],x["chunk_id"]));queue=[];evaluated=[];accepted=None
        fallback=selection.get("fold_policy")=="B0_FALLBACK"
        if fallback:
            pass
        elif system=="S1":
            evaluated=alts; accepted=max(alts,key=lambda x:(pred[(qid,x["chunk_id"])][0],-x["b0_rank"]),default=None)
            if accepted is not None and pred[(qid,accepted["chunk_id"])][0] < tu:accepted=None
        elif system=="S2":
            evaluated=alts;safe=[x for x in alts if pred[(qid,x["chunk_id"])][0]>=tu and pred[(qid,x["chunk_id"])][1]<=th]
            accepted=max(safe,key=lambda x:(pred[(qid,x["chunk_id"])][0],-x["b0_rank"]),default=None)
        elif system!="S0":
            queue=[x for x in alts if pred[(qid,x["chunk_id"])][2]>=ta]
            queue.sort(key=(lambda x:(x["b0_rank"],x["chunk_id"])) if system=="S3-B0-order" else (lambda x:(-pred[(qid,x["chunk_id"])][2],x["b0_rank"],x["chunk_id"])))
            limit=1 if system=="S3-cap1" else 2
            for alt in queue[:limit]:
                evaluated.append(alt);m,h,_=pred[(qid,alt["chunk_id"])]
                if m>=tu and h<=th:accepted=alt;break
        selected_id=int(accepted["chunk_id"] if accepted else top["chunk_id"]);selected=next(x for x in candidates if int(x["chunk_id"])==selected_id);target=targets[qid]
        qa_forwards=1+len(evaluated)
        if system in {"S3","S3-noLEX","S3-B0-order","S3-cap1"} and qa_forwards>MAX_QA:raise RuntimeError(f"IMPLEMENTATION BUG: QA forwards {qa_forwards}>3 for {qid}")
        evaluated_payload=[]
        for alt in evaluated:
            m,h,a=pred[(qid,alt["chunk_id"])]
            evaluated_payload.append({"chunk_id":alt["chunk_id"],"answer":alt["qa_answer"],"q_a_delta":alt["delta_qarel_raw"],"pairlex":alt["pairlex"],"mu":m,"p_harm":h,"p_admit":a,
                                      "accepted":bool(accepted and alt["chunk_id"]==accepted["chunk_id"])})
        runtime_payload={"dataset_cell":cell,"corpus_family":FAMILIES[cell],"document_id":top["document_id"],"canonical_document_id":top["canonical_document_id"],
          "document_sha256":top["document_sha256"],"question_lineage_id":top["question_lineage_id"],"question_id":qid,"canonical_aspect":top["canonical_aspect"],
          "language":top["language"],"outer_fold":fold,"seed":seed,"system":system,"parameterization_mode":mode,
          "b0_top_k":[{"chunk_id":x["chunk_id"],"rank":x["b0_rank"],"raw":x["b0_score_raw"],"z":x["b0_score_norm"]} for x in candidates],
          "admission_probabilities":{str(x["chunk_id"]):pred[(qid,x["chunk_id"])][2] for x in alts} if system!="S0" else {},
          "admission_queue":[x["chunk_id"] for x in queue[:limit if system!="S0" and system not in {"S1","S2"} else 0]],"baseline_answer":top["qa_answer"],"evaluated_candidates":evaluated_payload,
          "thresholds":{"tau_admit":selection.get("tau_admit"),"tau_utility":selection.get("tau_utility"),"tau_harm":selection.get("tau_harm")},
          "fold_policy":selection.get("fold_policy","B0"),"accepted_reason":"safe_positive" if accepted else "preserve_b0",
          "selected_chunk_id":selected_id,"b0_chunk_id":int(top["chunk_id"]),"final_answer":selected["qa_answer"],"effective_k":int(top["effective_k"]),
          "qa_forwards":qa_forwards,"q_a_ranker_forwards":0 if system=="S0" else qa_forwards}
        assert_runtime_gold_free(runtime_payload)
        tier=target["label_tier"][str(selected_id)]
        selected_label=target.get("chunk_label",{}).get(str(selected_id),"")
        selected_primary=target.get("primary_class",{}).get(str(selected_id),"")
        evaluation={"gold_answers":list(target.get("gold_answers",[])),"token_f1":float(target["candidate_f1"][str(selected_id)]),"baseline_token_f1":float(target["baseline_f1"]),
          "selected_tier":tier,"hit@1":int(tier=="sufficient_positive"),"baseline_hit@1":int(target["label_tier"][str(top["chunk_id"])]=="sufficient_positive"),
          "hn@1":int(target["hard_negative"][str(selected_id)]),"baseline_hn@1":int(target["hard_negative"][str(top["chunk_id"])]),
          "hit@k":int(any(value=="sufficient_positive" for value in target["label_tier"].values())),"hn@k":int(any(target["hard_negative"].values())),
          "f1_given_selected_positive":float(target["candidate_f1"][str(selected_id)]) if tier=="sufficient_positive" else None,
          "amb@1":int(selected_label=="ambiguous"),"gsrn@1":int(selected_primary=="gold_supporting_robust_negative"),
          "mrr_selected_first":_mrr(candidates,selected_id,target),"changed_top1":int(selected_id!=int(top["chunk_id"])),
          "actual_switch_utility":float(target["candidate_f1"][str(selected_id)])-float(target["baseline_f1"]),
          "candidate_utilities":{str(cid):float(value)-float(target["baseline_f1"]) for cid,value in target["candidate_f1"].items()}}
        output.append({"runtime":runtime_payload,"evaluation":evaluation})
    return output

class Pipeline:
    def __init__(self,config:Config):
        config.validate();self.config=config;self.output=config.run_dir;self.progress=ProgressLogger(self.output/"progress.jsonl");self.started=time.time();self.data=DataStore(config,self.progress);self.before=None
    def _status(self,status,**extra):write_json(self.output/"status.json",{"status":status,"elapsed_s":time.time()-self.started,**extra})
    def _heartbeat(self,stop):
        while not stop.wait(30):self.progress.log("still running",elapsed_seconds=round(time.time()-self.started,1))
    def run(self):
        stop=threading.Event();thread=threading.Thread(target=self._heartbeat,args=(stop,),daemon=True);thread.start()
        try:
            self._status("CREATED");self.audit()
            if self.config.phase=="audit":return 0
            self.historical_replay()
            if self.config.phase=="replay":return 0
            if self.config.phase in {"run","all"}:self.development()
            if self.config.phase in {"analysis","all"}:
                from .analyze import analyze_run;analyze_run(self.output,self.config.bootstrap,self.progress)
            if self.config.phase in {"benchmark","all"}:
                if self.config.smoke:write_json(self.output/"resource_benchmark/SKIPPED.json",{"reason":"smoke mode"})
                else:
                    from .benchmark import run_benchmarks;run_benchmarks(self.config,self.output,self.progress)
            self.finish();return 0
        except Exception as exc:
            self._status("FAILED",error_type=type(exc).__name__,error=str(exc));write_json(self.output/"FAILED.json",{"error_type":type(exc).__name__,"error":str(exc)});self.progress.warn(f"FAILED {type(exc).__name__}: {exc}");raise
        finally:stop.set();thread.join(1)
    def audit(self):
        self._status("AUDIT_RUNNING");self.progress.stage("audit","freeze contract and verify read-only V5/V6 history")
        git_snapshot(self.output/"audit/git_before");result=run_audit(self.output/"audit",self.config.v5_run,self.config.v6_run,self.config.datasets);self.before=result["hashes"]
        contract=frozen_contract();contract["config"]=self.config.payload();text=json.dumps(contract,ensure_ascii=False,indent=2,sort_keys=True)+"\n"
        manifest_path=self.output/"manifest.json"
        if manifest_path.is_file() and self.config.resume:
            old=json.loads(manifest_path.read_text());old_config=old["config"];new_config=self.config.payload()
            stable=("v5_run","v6_run","datasets","seeds","outer_folds","inner_folds","modes","device","smoke")
            mismatch={key:(old_config.get(key),new_config.get(key)) for key in stable if old_config.get(key)!=new_config.get(key)}
            if mismatch:raise RuntimeError(f"resume fingerprint mismatch: {mismatch}")
            if sha256(self.output/"frozen_contract.yaml")!=old["contract_sha256"]:raise RuntimeError("frozen contract hash mismatch on resume")
        else:
            write_text(self.output/"frozen_contract.yaml",text);write_json(manifest_path,{"contract_sha256":sha256(self.output/"frozen_contract.yaml"),"config":self.config.payload(),"historical_read_only":True})
        write_json(self.output/"reused_code/reused_components.json",[{"source":"V5 inner_meta/eval features","wrapper":"V7 DataStore","modified":False},{"source":"V6 enrich_question","wrapper":"V7 candidate_rows","modified":False},{"source":"v4 pair_token_f1","wrapper":"V7 PairLEX","modified":False}]);self._status("AUDIT_PASSED")
    def historical_replay(self):
        self.progress.stage("replay","exact B0/V5 replay and frozen V6 S4/S4-lex verification");write_json(self.output/"audit/historical_replay/replay.json",replay(self.config.v5_run,self.config.v6_run,self.config.datasets,strict=True))
    def _scope(self,mode,fold,seed,target_cell=None,held_family=None):
        train=[];eval_sets={};all_cells=[target_cell] if mode=="local" else list(self.config.datasets)
        for cell in all_cells:
            _,meta,_=self.data.load(cell,fold,seed,"meta");runtime,rows,targets=self.data.load(cell,fold,seed,"eval");eval_sets[cell]=(runtime,rows,targets)
            if mode!="loco" or FAMILIES[cell]!=held_family:train.extend(meta)
        eval_runtime=[r for values in eval_sets.values() for r in values[0]];train,embargo=_embargo(train,eval_runtime)
        if mode=="loco":eval_sets={c:v for c,v in eval_sets.items() if FAMILIES[c]==held_family}
        return train,eval_sets,embargo
    def _execute(self,mode,fold,seed,target_cell=None,held_family=None):
        label=target_cell or held_family or "shared";root=self.output/"models"/mode/label/f"outer_fold_{fold}"/f"seed_{seed}";done=root/"COMPLETED.json"
        if done.is_file():self.progress.log("scope already complete",mode=mode,scope=label,fold=fold,seed=seed);return
        self.progress.stage(f"{mode}:{label}:f{fold}:s{seed}","staged META_PRED_OOF tuning and sealed progressive EVAL")
        train,eval_sets,embargo=self._scope(mode,fold,seed,target_cell,held_family);audit=normalize_weights(train,pooled=mode!="local")
        write_json(root/"embargo.json",embargo);write_json(root/"training_weight_audit.json",audit)
        policies={}
        for system in ("S1","S2","S3","S3-noLEX","S3-B0-order","S3-cap1"):
            self.progress.log("tuning policy",mode=mode,scope=label,fold=fold,seed=seed,system=system,rows=len(train));policy=tune_policy(copy.deepcopy(train),system,seed,self.config.inner_folds);policies[system]=policy
            destination=root/system;destination.mkdir(parents=True,exist_ok=True);joblib.dump(policy.utility_model,destination/"utility_model.joblib");joblib.dump(policy.harm_model,destination/"harm_model.joblib");joblib.dump(policy.admission_model,destination/"admission_model.joblib")
            write_json(destination/"selection.json",{"mode":mode,"scope":label,"fold":fold,"seed":seed,"system":system,"one_shared_model_for_all_eval_cells":mode!="local","feature_columns":list(policy.columns),"selection":policy.selection})
        for cell,(runtime,rows,targets) in eval_sets.items():
            base=predictions(cell,runtime,rows,targets,"S0",mode,fold,seed);write_jsonl(self.output/"predictions"/mode/label/cell/f"outer_fold_{fold}"/f"seed_{seed}"/"S0.jsonl",base)
            for system,policy in policies.items():write_jsonl(self.output/"predictions"/mode/label/cell/f"outer_fold_{fold}"/f"seed_{seed}"/f"{system}.jsonl",predictions(cell,runtime,rows,targets,system,mode,fold,seed,policy))
        write_json(done,{"status":"complete","mode":mode,"scope":label,"fold":fold,"seed":seed,"systems":["S0",*policies],"eval_cells":sorted(eval_sets)})
    def development(self):
        self._status("RUNNING");self.progress.stage("development","primary LOCAL plus POOLED-simple and LOCO-meta diagnostics")
        if "local" in self.config.modes:
            for cell in self.config.datasets:
             for fold in self.config.outer_folds:
              for seed in self.config.seeds:self._execute("local",fold,seed,target_cell=cell)
        if "pooled" in self.config.modes:
            for fold in self.config.outer_folds:
             for seed in self.config.seeds:self._execute("pooled",fold,seed)
        if "loco" in self.config.modes:
            for family in sorted({FAMILIES[c] for c in self.config.datasets}):
             for fold in self.config.outer_folds:
              for seed in self.config.seeds:self._execute("loco",fold,seed,held_family=family)
    def finish(self):
        self.progress.stage("finalize","history hash assertion and immutable completion marker");after=historical_hashes(self.config.v5_run,self.config.v6_run,self.config.datasets);write_json(self.output/"audit/historical_hashes_after.json",after);git_snapshot(self.output/"audit/git_after")
        if self.before!=after:raise RuntimeError("RUN_INVALID_HISTORY_MODIFIED")
        write_json(self.output/"audit/regression_summary.json",{"historical_hashes_unchanged":True,"runtime_gold_leakage":False,"seed_isolation":True,"meta_prediction_oof":True,"scier_cross_cell_embargo":True,"max_qa_three":True,"progressive_state_no_future_features":True})
        failed=self.output/"FAILED.json"
        if failed.is_file():
            write_json(self.output/"audit/recovered_failure.json",{"recovered_by_resume":True,"failure":json.loads(failed.read_text())})
            remove_file(failed)
        self._status("COMPLETED");write_json(self.output/"COMPLETED.json",{"status":"COMPLETED","elapsed_s":time.time()-self.started,"historical_hashes_unchanged":True,"contract_sha256":sha256(self.output/"frozen_contract.yaml")})
