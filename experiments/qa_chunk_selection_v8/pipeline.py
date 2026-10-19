"""V8 dual-branch orchestration over immutable V5/V7 canonical features."""
from __future__ import annotations
from collections import defaultdict
import copy,json,math,threading,time
from pathlib import Path
import joblib,numpy as np
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v3.data import records_from_jsonl
from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS as V5_CELLS
from experiments.qa_chunk_selection_v6.features import enrich_question as v6_enrich
from experiments.qa_chunk_selection_v7.replay import replay as replay_v7_upstream
from .audit import git_snapshot,historical_hashes,run_audit
from .config import Config
from .contract import (CELLS, FAMILIES, BRANCH_HRR_SYSTEMS, BRANCH_AF_SYSTEMS,
                       PRIMARY_HRR,PRIMARY_AF,SEM_COMMON,frozen_contract,hrr_columns,af_columns)
from .features import (af_rows,assert_runtime_gold_free,enrich_reader_features,fit_v44_core,grouped,hrr_rows,normalize_weights)
from .io_guard import read_jsonl,remove_file,sha256,write_json,write_jsonl,write_text
from .models import apply_policy,tune_policy
from .preservation import layer_r_availability,replay_layer_p

class DataStore:
    def __init__(self,config,progress):self.config=config;self.progress=progress;self._items={};self._cache={}
    def items(self,cell):
        if cell not in self._items:self._items[cell]={str(x.question_id):x for x in records_from_jsonl(V5_CELLS[cell].records)}
        return self._items[cell]
    def load(self,cell,fold,seed,role):
        key=(cell,fold,seed,role)
        if key in self._cache:return copy.deepcopy(self._cache[key])
        path=self.config.v5_run/cell/f"outer_fold_{fold}"/f"seed_{seed}"/("inner_meta/features.jsonl" if role=="meta" else "eval/features.jsonl")
        raw=read_jsonl(path);source_groups=defaultdict(list)
        for row in raw:source_groups[str(row["question_id"])].append(row)
        items=self.items(cell);runtime=[];targets={};hrows=[]
        for index,(qid,source) in enumerate(sorted(source_groups.items()),1):
            run,target=v6_enrich(cell,source,items[qid],outer_fold=fold,seed=seed,meta_source=role=="meta")
            run=enrich_reader_features(run,items[qid]);target["gold_answers"]=[str(x) for x in items[qid].gold_answers]
            target["chunk_label"]={str(r["chunk_id"]):str(items[qid].chunk_by_id(int(r["chunk_id"])).label) for r in run}
            target["primary_class"]={str(r["chunk_id"]):str(items[qid].chunk_by_id(int(r["chunk_id"])).primary_class) for r in run}
            runtime.extend(run);targets[(cell,qid)]=target;hrows.extend(hrr_rows(run,target))
            self.progress.count(index,len(source_groups),item=f"{cell} {role} exact reader features",extra={"fold":fold,"seed":seed})
        cache=self.config.run_dir/"cache/canonical_features"/cell/f"outer_fold_{fold}"/f"seed_{seed}"
        write_jsonl(cache/f"{role}_runtime.jsonl",runtime);write_jsonl(cache/f"{role}_hrr_targets.jsonl",hrows)
        result=(runtime,hrows,targets);self._cache[key]=copy.deepcopy(result);return copy.deepcopy(result)

def _embargo(runtime,rows,targets,eval_runtime):
    docs={x["canonical_document_id"] for x in eval_runtime};hashes={x["document_sha256"] for x in eval_runtime}
    kept_runtime=[x for x in runtime if x["canonical_document_id"] not in docs and x["document_sha256"] not in hashes]
    kept_rows=[x for x in rows if x["canonical_document_id"] not in docs and x["document_sha256"] not in hashes]
    keys={(x["dataset_cell"],x["question_id"]) for x in kept_runtime};kept_targets={k:v for k,v in targets.items() if k in keys}
    return kept_runtime,kept_rows,kept_targets,{"input_runtime_rows":len(runtime),"kept_runtime_rows":len(kept_runtime),"input_training_rows":len(rows),"kept_training_rows":len(kept_rows),"excluded_documents":len(docs),"excluded_hashes":len(hashes)}

def _mrr(values,selected,target):
    order=[selected]+[int(x["chunk_id"]) for x in values if int(x["chunk_id"])!=selected];relevant={int(k) for k,v in target["label_tier"].items() if v=="sufficient_positive"};ranks=[i+1 for i,c in enumerate(order) if c in relevant]
    return 1/min(ranks) if ranks else 0.0

def build_predictions(branch,cell,runtime,rows,targets,system,mode,fold,seed,policy=None,core=None,core_params=None):
    runtime_groups=grouped(runtime);row_groups=defaultdict(list)
    for row in rows:row_groups[(row["dataset_cell"],row["question_id"])].append(row)
    choice={};pred={};selection=policy.selection if policy else {}
    if policy:
        choices,(mu,harm)=apply_policy(rows,policy,use_harm=(branch=="hrr" or system.startswith("F5")))
        choice={(x["dataset_cell"],x["question_id"]):int(x["selected_chunk_id"]) for x in choices}
        for row,m,h in zip(rows,mu,harm):pred[(row["dataset_cell"],row["question_id"],int(row["chunk_id"]))]=(float(m),float(h))
    output=[]
    for key,values in sorted(runtime_groups.items()):
        target=targets[key];top=values[0]
        if branch=="hrr":base_id=int(top["chunk_id"]);base_answer=str(top.get("qa_answer") or "");base_kind="B0"
        else:
            anchor=core[key]["selected"];base_id=int(anchor["chunk_id"]);base_answer=str(anchor["runtime"].get("qa_answer") or "");base_kind="V44_CORE"
        selected_id=int(choice.get(key,base_id));selected=next(x for x in values if int(x["chunk_id"])==selected_id);alts=sorted(row_groups.get(key,[]),key=lambda x:(int(x["b0_rank"]),int(x["chunk_id"])))
        evaluated=[]
        for alt in alts:
            if policy:
                mu,harm=pred[(key[0],key[1],int(alt["chunk_id"]))]
                evaluated.append({"chunk_id":int(alt["chunk_id"]),"answer":str(alt.get("qa_answer") or ""),"mu":mu,"p_harm":harm,"accepted":int(alt["chunk_id"])==selected_id and selected_id!=base_id})
        qa_forwards=1 if system in {"H0"} else int(top["effective_k"])
        runtime_payload={"dataset_cell":cell,"corpus_family":FAMILIES[cell],"document_id":top["document_id"],"canonical_document_id":top["canonical_document_id"],
          "document_sha256":top["document_sha256"],"question_lineage_id":top["question_lineage_id"],"question_id":key[1],"canonical_aspect":top["canonical_aspect"],
          "language":top["language"],"outer_fold":fold,"seed":seed,"branch":branch,"system":system,"parameterization_mode":mode,
          "b0_top_k":[{"chunk_id":int(x["chunk_id"]),"rank":int(x["b0_rank"]),"raw":float(x["b0_score_raw"]),"z":float(x["b0_score_norm"]),
                       "agreement":float(x["agreement_raw"]),"fulllex":float(x["fulllex_raw"])} for x in values],
          "v44_core_parameters":core_params if branch=="v44_af" else None,"v44_anchor_chunk_id":base_id if branch=="v44_af" else None,
          "baseline_answer":base_answer,"evaluated_candidates":evaluated,"thresholds":{"tau_utility":selection.get("tau_utility"),"tau_harm":selection.get("tau_harm")},
          "fold_policy":selection.get("fold_policy",base_kind),"selected_chunk_id":selected_id,"base_chunk_id":base_id,"b0_chunk_id":int(top["chunk_id"]),
          "final_answer":str(selected.get("qa_answer") or ""),"effective_k":int(top["effective_k"]),"qa_forwards":qa_forwards,
          "q_a_ranker_forwards":int(top["effective_k"]) if policy else 0,"semantic_encoder_forwards":0}
        assert_runtime_gold_free(runtime_payload)
        tier=target["label_tier"][str(selected_id)];base_tier=target["label_tier"][str(base_id)]
        evaluation={"gold_answers":target["gold_answers"],"token_f1":float(target["candidate_f1"][str(selected_id)]),"baseline_token_f1":float(target["candidate_f1"][str(base_id)]),
          "selected_tier":tier,"hit@1":int(tier=="sufficient_positive"),"baseline_hit@1":int(base_tier=="sufficient_positive"),
          "hn@1":int(target["hard_negative"][str(selected_id)]),"baseline_hn@1":int(target["hard_negative"][str(base_id)]),
          "hit@k":int(any(v=="sufficient_positive" for v in target["label_tier"].values())),"hn@k":int(any(target["hard_negative"].values())),
          "f1_given_selected_positive":float(target["candidate_f1"][str(selected_id)]) if tier=="sufficient_positive" else None,
          "amb@1":int(target["chunk_label"].get(str(selected_id))=="ambiguous"),"gsrn@1":int(target["primary_class"].get(str(selected_id))=="gold_supporting_robust_negative"),
          "mrr_selected_first":_mrr(values,selected_id,target),"changed_top1":int(selected_id!=base_id),
          "actual_switch_utility":float(target["candidate_f1"][str(selected_id)])-float(target["candidate_f1"][str(base_id)]),
          "candidate_utilities":{str(cid):float(v)-float(target["candidate_f1"][str(base_id)]) for cid,v in target["candidate_f1"].items()}}
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
            self.preservation()
            if self.config.phase=="preservation":return 0
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
        self._status("AUDIT_RUNNING");self.progress.stage("audit","freeze both V8 contracts and hash read-only V4.4-V7 history");git_snapshot(self.output/"audit/git_before");self.before=run_audit(self.output/"audit",self.config)
        contracts={"branch_hrr_frozen.yaml":frozen_contract("hrr"),"branch_v44_af_frozen.yaml":frozen_contract("v44_af")};hashes={}
        for name,payload in contracts.items():write_text(self.output/"contracts"/name,json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n");hashes[name]=sha256(self.output/"contracts"/name)
        manifest={"config":self.config.payload(),"contract_sha256":hashes,"historical_read_only":True,"sem_common":SEM_COMMON}
        path=self.output/"manifest.json"
        if path.is_file() and self.config.resume:
            old=json.loads(path.read_text());stable=("v4_4_run","v4_4_external_run","v4_5_run","v5_run","v6_run","v7_run","datasets","seeds","outer_folds","inner_folds","modes","smoke")
            mismatch={k:(old["config"].get(k),manifest["config"].get(k)) for k in stable if old["config"].get(k)!=manifest["config"].get(k)}
            if mismatch:raise RuntimeError(f"resume fingerprint mismatch: {mismatch}")
            if old["contract_sha256"]!=hashes:raise RuntimeError("frozen V8 contract hash mismatch")
        else:write_json(path,manifest)
        self._status("AUDIT_PASSED")
    def preservation(self):
        self.progress.stage("layer_p","exact immutable V4.4 English preservation replay");replay_layer_p(self.config,self.output);layer_r_availability(self.config,self.output);self._status("PRESERVATION_PASSED")
    def historical_replay(self):
        self.progress.stage("replay","canonical B0/V6 and exact historical V7 LOCAL S2 replay");report={"upstream":replay_v7_upstream(self.config.v5_run,self.config.v6_run,self.config.datasets,strict=True),"v7_s2":{}}
        metrics=json.loads((self.config.v7_run/"analysis/primary_metrics.json").read_text())
        for cell in self.config.datasets:
            paths=sorted((self.config.v7_run/"predictions/local"/cell/cell).glob("outer_fold_*/seed_*/S2.jsonl"));rows=[r for p in paths for r in read_jsonl(p)]
            observed=float(np.mean([r["evaluation"]["token_f1"] for r in rows]));expected=float(metrics["local"][cell]["S2"]["token_f1"]);ok=bool(rows) and abs(observed-expected)<1e-10
            report["v7_s2"][cell]={"rows":len(rows),"observed":observed,"expected":expected,"status":"PASS" if ok else "FAIL"}
            if not ok:raise RuntimeError(f"V7 S2 replay failed: {cell}")
        write_json(self.output/"audit/v7_replay/replay.json",report);self._status("REPLAY_PASSED")
    def _scope(self,mode,fold,seed,target_cell=None,held_family=None):
        train_runtime=[];train_hrr=[];train_targets={};eval_sets={};cells=[target_cell] if mode=="local" else list(self.config.datasets)
        for cell in cells:
            mr,mh,mt=self.data.load(cell,fold,seed,"meta");er,eh,et=self.data.load(cell,fold,seed,"eval");eval_sets[cell]=(er,eh,et)
            if mode!="loco" or FAMILIES[cell]!=held_family:train_runtime.extend(mr);train_hrr.extend(mh);train_targets.update(mt)
        eval_runtime=[r for value in eval_sets.values() for r in value[0]];train_runtime,train_hrr,train_targets,embargo=_embargo(train_runtime,train_hrr,train_targets,eval_runtime)
        if mode=="loco":eval_sets={c:v for c,v in eval_sets.items() if FAMILIES[c]==held_family}
        return train_runtime,train_hrr,train_targets,eval_sets,embargo
    def _v7_s2_rows(self,mode,label,cell,fold,seed):
        path=self.config.v7_run/"predictions"/mode/label/cell/f"outer_fold_{fold}"/f"seed_{seed}"/"S2.jsonl"
        rows=read_jsonl(path)
        for row in rows:
            row["runtime"].update(branch="hrr",system="H1",historical_exact_replay="V7-S2")
        return rows
    def _execute(self,mode,fold,seed,target_cell=None,held_family=None):
        label=target_cell or held_family or "shared";state_root=self.output/"state"/mode/label/f"outer_fold_{fold}"/f"seed_{seed}";done=state_root/"COMPLETED.json"
        if done.is_file():self.progress.log("scope already complete",mode=mode,scope=label,fold=fold,seed=seed);return
        self.progress.stage(f"{mode}:{label}:f{fold}:s{seed}","fit frozen HRR and V44-AF ladders using META_PRED_OOF")
        tr_runtime,tr_hrr,tr_targets,eval_sets,embargo=self._scope(mode,fold,seed,target_cell,held_family);write_json(state_root/"embargo.json",embargo)
        normalize_weights(tr_hrr,pooled=mode!="local")
        hpolicies={}
        for system in BRANCH_HRR_SYSTEMS:
            if system in {"H0","H1"}:continue
            self.progress.log("tuning HRR",mode=mode,scope=label,fold=fold,seed=seed,system=system,rows=len(tr_hrr));policy=tune_policy(copy.deepcopy(tr_hrr),system,hrr_columns(system),seed,self.config.inner_folds,True);hpolicies[system]=policy;root=self.output/"branch_hrr/models"/mode/label/f"outer_fold_{fold}"/f"seed_{seed}"/system;root.mkdir(parents=True,exist_ok=True);joblib.dump(policy.utility_model,root/"utility_model.joblib");joblib.dump(policy.harm_model,root/"harm_model.joblib");write_json(root/"selection.json",{"feature_columns":list(policy.columns),"selection":policy.selection})
        core_params,core_audit=fit_v44_core(tr_runtime,tr_targets,seed);write_json(self.output/"branch_v44_af/canonical/models"/mode/label/f"outer_fold_{fold}"/f"seed_{seed}"/"v44_core_selection.json",{"params":core_params,"audit":core_audit})
        tr_af,_=af_rows(tr_runtime,tr_targets,core_params);normalize_weights(tr_af,pooled=mode!="local");fpolicies={}
        for system in BRANCH_AF_SYSTEMS:
            if system=="F0":continue
            self.progress.log("tuning V44-AF",mode=mode,scope=label,fold=fold,seed=seed,system=system,rows=len(tr_af));use_harm=system.startswith("F5");policy=tune_policy(copy.deepcopy(tr_af),system,af_columns(system),seed,self.config.inner_folds,use_harm);fpolicies[system]=policy;root=self.output/"branch_v44_af/canonical/models"/mode/label/f"outer_fold_{fold}"/f"seed_{seed}"/system;root.mkdir(parents=True,exist_ok=True);joblib.dump(policy.utility_model,root/"utility_model.joblib");joblib.dump(policy.harm_model,root/"harm_model.joblib");write_json(root/"selection.json",{"feature_columns":list(policy.columns),"selection":policy.selection})
        for cell,(runtime,hrows,targets) in eval_sets.items():
            hroot=self.output/"branch_hrr/predictions"/mode/label/cell/f"outer_fold_{fold}"/f"seed_{seed}"
            write_jsonl(hroot/"H0.jsonl",build_predictions("hrr",cell,runtime,hrows,targets,"H0",mode,fold,seed))
            write_jsonl(hroot/"H1.jsonl",self._v7_s2_rows(mode,label,cell,fold,seed))
            for system,policy in hpolicies.items():write_jsonl(hroot/f"{system}.jsonl",build_predictions("hrr",cell,runtime,hrows,targets,system,mode,fold,seed,policy))
            erows,core=af_rows(runtime,targets,core_params);froot=self.output/"branch_v44_af/canonical/predictions"/mode/label/cell/f"outer_fold_{fold}"/f"seed_{seed}"
            write_jsonl(self.output/"cache/af_features"/mode/label/cell/f"outer_fold_{fold}"/f"seed_{seed}.jsonl",erows)
            write_jsonl(froot/"F0.jsonl",build_predictions("v44_af",cell,runtime,erows,targets,"F0",mode,fold,seed,core=core,core_params=core_params))
            for system,policy in fpolicies.items():write_jsonl(froot/f"{system}.jsonl",build_predictions("v44_af",cell,runtime,erows,targets,system,mode,fold,seed,policy,core,core_params))
        write_json(done,{"status":"complete","mode":mode,"scope":label,"fold":fold,"seed":seed,"hrr_systems":list(BRANCH_HRR_SYSTEMS),"af_systems":list(BRANCH_AF_SYSTEMS),"sem_common":SEM_COMMON})
    def development(self):
        self._status("RUNNING");self.progress.stage("development","canonical Branch A/B comparison; LOCAL primary, POOLED/LOCO diagnostics")
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
        self.progress.stage("finalize","assert V4.4-V7 hashes unchanged and write completion marker");after=historical_hashes(self.config.v4_4_run,self.config.v4_4_external_run,self.config.v4_5_run,self.config.v5_run,self.config.v6_run,self.config.v7_run,self.config.datasets);write_json(self.output/"audit/hashes_after.json",after);git_snapshot(self.output/"audit/git_after")
        if self.before!=after:raise RuntimeError("RUN_INVALID_HISTORY_MODIFIED")
        write_json(self.output/"audit/regression_summary.json",{"historical_hashes_unchanged":True,"runtime_gold_leakage":False,"seed_isolation":True,"meta_prediction_oof":True,"scier_cross_cell_embargo":True,"manual_language_weights":False,"sem_common":SEM_COMMON})
        failed=self.output/"FAILED.json"
        if failed.is_file():write_json(self.output/"audit/recovered_failure.json",{"recovered_by_resume":True,"failure":json.loads(failed.read_text())});remove_file(failed)
        self._status("COMPLETED");write_json(self.output/"COMPLETED.json",{"status":"COMPLETED","elapsed_s":time.time()-self.started,"historical_hashes_unchanged":True,"primary_hrr":PRIMARY_HRR,"primary_af":PRIMARY_AF})
