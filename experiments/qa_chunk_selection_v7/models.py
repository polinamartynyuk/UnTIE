"""Strict META_PRED_OOF fitting and staged V7 policy selection."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import hashlib, math
import numpy as np
from .contract import (ADMISSION_NUMERIC,ALPHA_GRID,HARM_C_GRID,MAX_EXTRA_QA,MEAN_QA_BUDGET,
                       TAU_ADMIT_GRID,TAU_HARM_GRID,TAU_UTILITY_GRID,selector_columns)

class ConstantRegressor:
    def __init__(self,value=0):self.value=float(value)
    def predict(self,frame):return np.full(len(frame),self.value)
class ConstantClassifier:
    def __init__(self,p):self.p=float(p)
    def predict_proba(self,frame):return np.column_stack([np.full(len(frame),1-self.p),np.full(len(frame),self.p)])

def fold_for(document_id,seed,folds):return int.from_bytes(hashlib.sha256(f"{seed}|{document_id}".encode()).digest()[:8],"big")%folds
def _frame(rows):
    import pandas as pd
    return pd.DataFrame(rows)
def _preprocessor(columns):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder,StandardScaler
    return ColumnTransformer([("num",StandardScaler(),list(columns)),("language",OneHotEncoder(handle_unknown="ignore"),["language"])])
def fit_ridge(rows,columns,alpha):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    if not rows:return ConstantRegressor(0)
    y=np.asarray([r["utility"] for r in rows],float)
    if np.ptp(y)<1e-12:return ConstantRegressor(float(y.mean()))
    model=Pipeline([("pre",_preprocessor(columns)),("model",Ridge(alpha=float(alpha),fit_intercept=True))])
    model.fit(_frame(rows),y,model__sample_weight=np.asarray([r["weight"] for r in rows],float));return model
def fit_classifier(rows,columns,target,c=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    if not rows:return ConstantClassifier(0)
    y=np.asarray([r[target] for r in rows],int)
    if len(set(y.tolist()))<2:return ConstantClassifier(float(y[0]))
    model=Pipeline([("pre",_preprocessor(columns)),("model",LogisticRegression(C=float(c),solver="liblinear",class_weight=None,max_iter=3000))])
    model.fit(_frame(rows),y,model__sample_weight=np.asarray([r["weight"] for r in rows],float));return model
def predict_reg(model,rows):return np.clip(np.asarray(model.predict(_frame(rows)),float),-100,100)
def predict_prob(model,rows):return np.asarray(model.predict_proba(_frame(rows))[:,1],float)

def _oof(rows,seed,folds,fit,predict):
    assignment=[fold_for(r["canonical_document_id"],seed,folds) for r in rows];out=np.full(len(rows),np.nan)
    for f in sorted(set(assignment)):
        train=[r for r,a in zip(rows,assignment) if a!=f];idx=[i for i,a in enumerate(assignment) if a==f]
        if not train or not idx:continue
        model=fit(train);out[idx]=predict(model,[rows[i] for i in idx])
    if np.isnan(out).any():raise RuntimeError("META_PRED_OOF coverage incomplete")
    return out,assignment
def _wmean(values,weights):return float(np.average(np.asarray(values,float),weights=np.asarray(weights,float))) if len(values) else 0.0

def _simulate(rows,mu,harm,admit,system,thresholds):
    grouped=defaultdict(list)
    for i,row in enumerate(rows):grouped[(row["dataset_cell"],row["question_id"])].append((row,float(mu[i]),float(harm[i]),float(admit[i])))
    choices=[]
    for (cell,qid),candidates in grouped.items():
        candidates.sort(key=lambda x:(int(x[0]["b0_rank"]),int(x[0]["chunk_id"])))
        first=candidates[0][0];selected=None;evaluated=[]
        if system=="S1":
            evaluated=candidates;best=max(candidates,key=lambda x:(x[1],-int(x[0]["b0_rank"])))
            if best[1]>=thresholds["tau_utility"]:selected=best
        elif system=="S2":
            evaluated=candidates;safe=[x for x in candidates if x[1]>=thresholds["tau_utility"] and x[2]<=thresholds["tau_harm"]]
            if safe:selected=max(safe,key=lambda x:(x[1],-int(x[0]["b0_rank"])))
        else:
            queue=[x for x in candidates if x[3]>=thresholds["tau_admit"]]
            queue.sort(key=(lambda x:(int(x[0]["b0_rank"]),int(x[0]["chunk_id"]))) if system=="S3-B0-order" else (lambda x:(-x[3],int(x[0]["b0_rank"]),int(x[0]["chunk_id"]))))
            limit=1 if system=="S3-cap1" else MAX_EXTRA_QA
            for item in queue[:limit]:
                evaluated.append(item)
                if item[1]>=thresholds["tau_utility"] and item[2]<=thresholds["tau_harm"]:selected=item;break
        row=selected[0] if selected else first
        choices.append({"dataset_cell":cell,"question_id":qid,"f1":float(row["candidate_f1"] if selected else first["baseline_f1"]),
          "base_f1":float(first["baseline_f1"]),"hit1":int((row["candidate_tier"] if selected else first["baseline_tier"])=="sufficient_positive"),
          "base_hit1":int(first["baseline_tier"]=="sufficient_positive"),"hn1":int(row["candidate_hn"] if selected else first["baseline_hn"]),
          "base_hn1":int(first["baseline_hn"]),"switch":int(selected is not None),"qa_forwards":1+len(evaluated),
          "selected_chunk_id":int(row["chunk_id"] if selected else first["b0_chunk_id"]),"evaluated":len(evaluated)})
    return choices
def policy_metrics(choices):
    mean=lambda k:float(np.mean([x[k] for x in choices])) if choices else 0.0
    return {"token_f1":mean("f1"),"baseline_token_f1":mean("base_f1"),"delta_f1":mean("f1")-mean("base_f1"),
      "delta_hit1_pp":100*(mean("hit1")-mean("base_hit1")),"delta_hn1_pp":100*(mean("hn1")-mean("base_hn1")),
      "switch_rate":mean("switch"),"qa_forwards":mean("qa_forwards"),"max_qa_forwards":max((x["qa_forwards"] for x in choices),default=1)}
def _safe(m):return m["delta_hit1_pp"]>=-0.5-1e-12 and m["delta_hn1_pp"]<=1.0+1e-12

@dataclass
class FittedPolicy:
    system:str;utility_model:object;harm_model:object;admission_model:object;columns:tuple;selection:dict
    def predict(self,rows):
        return predict_reg(self.utility_model,rows),predict_prob(self.harm_model,rows),predict_prob(self.admission_model,rows)

def tune_policy(rows:list[dict],system:str,seed:int,folds:int)->FittedPolicy:
    if not rows:raise ValueError("no training alternatives after embargo")
    documents={r["canonical_document_id"] for r in rows}
    feasible_folds=[n for n in range(min(folds,4),1,-1) if len(documents)>=n]
    if not feasible_folds:raise ValueError("at least two training documents are required for META_PRED_OOF")
    folds=feasible_folds[0]
    columns=selector_columns(system);weights=np.asarray([r["weight"] for r in rows],float);truth=np.asarray([r["utility"] for r in rows],float)
    ridge_trials=[]
    for alpha in ALPHA_GRID:
        pred,assignment=_oof(rows,seed,folds,lambda x:fit_ridge(x,columns,alpha),predict_reg)
        ridge_trials.append((alpha,pred,_wmean(np.abs(pred-truth),weights),assignment))
    best_mae=min(x[2] for x in ridge_trials);near=[x for x in ridge_trials if x[2]<=best_mae+0.05];alpha,mu,mae,assignment=max(near,key=lambda x:x[0])
    harm_trials=[]
    from sklearn.metrics import log_loss
    for c in HARM_C_GRID:
        p,_=_oof(rows,seed,folds,lambda x:fit_classifier(x,columns,"harm_target",c),predict_prob)
        loss=float(log_loss([r["harm_target"] for r in rows],np.clip(p,1e-8,1-1e-8),sample_weight=weights,labels=[0,1]))
        harm_trials.append((c,p,loss))
    best_loss=min(x[2] for x in harm_trials);near=[x for x in harm_trials if x[2]<=best_loss+0.001];c_harm,harm,loss=min(near,key=lambda x:x[0])
    admit,_=_oof(rows,seed,folds,lambda x:fit_classifier(x,ADMISSION_NUMERIC,"admission_target",1.0),predict_prob)
    from sklearn.metrics import average_precision_score,brier_score_loss,mean_squared_error,r2_score,roc_auc_score
    from scipy.stats import spearmanr
    harm_y=np.asarray([r["harm_target"] for r in rows],int);admit_y=np.asarray([r["admission_target"] for r in rows],int)
    def auc(metric,y,p): return float(metric(y,p,sample_weight=weights)) if len(set(y.tolist()))>1 else None
    def corr(left,right,kind):
        if np.std(left)<1e-12 or np.std(right)<1e-12:return None
        return float(np.corrcoef(left,right)[0,1]) if kind=="pearson" else float(spearmanr(left,right).statistic)
    utility_diag={"mae":mae,"rmse":float(mean_squared_error(truth,mu,sample_weight=weights)**0.5),"pearson":corr(truth,mu,"pearson"),
                  "spearman":corr(truth,mu,"spearman"),"r2":float(r2_score(truth,mu,sample_weight=weights)) if np.ptp(truth)>1e-12 else None}
    harm_diag={"auroc":auc(roc_auc_score,harm_y,harm),"pr_auc":auc(average_precision_score,harm_y,harm),"log_loss":loss,
               "brier":float(brier_score_loss(harm_y,harm,sample_weight=weights))}
    admit_binary=(admit>=0.5).astype(int);positive=admit_y==1
    admission_diag={"auroc":auc(roc_auc_score,admit_y,admit),"pr_auc":auc(average_precision_score,admit_y,admit),
      "recall_at_0_5":float(((admit_binary==1)&positive).sum()/max(int(positive.sum()),1)),
      "precision_at_0_5":float(((admit_binary==1)&positive).sum()/max(int((admit_binary==1).sum()),1)),
      "positive_alternatives_missed_at_0_5":int(((admit_binary==0)&positive).sum())}
    trials=[]
    admit_grid=(None,) if system in {"S1","S2"} else TAU_ADMIT_GRID
    harm_grid=(None,) if system=="S1" else TAU_HARM_GRID
    for ta in admit_grid:
      for tu in TAU_UTILITY_GRID:
       for th in harm_grid:
        thresholds={"tau_admit":ta,"tau_utility":tu,"tau_harm":th}
        score=policy_metrics(_simulate(rows,mu,harm,admit,system,thresholds));feasible=_safe(score) and (system not in {"S3","S3-noLEX","S3-B0-order","S3-cap1"} or score["qa_forwards"]<=MEAN_QA_BUDGET+1e-12)
        trials.append({**thresholds,**score,"feasible":feasible})
    feasible=[x for x in trials if x["feasible"]]
    if feasible:
        chosen=max(feasible,key=lambda x:(x["token_f1"],-x["switch_rate"],x["tau_utility"],-(x["tau_harm"] if x["tau_harm"] is not None else 0),x["tau_admit"] or 0));fallback=False
    else:
        chosen={"tau_admit":math.inf,"tau_utility":math.inf,"tau_harm":0.0,"token_f1":policy_metrics(_simulate(rows,mu,harm,admit,"S3",{"tau_admit":math.inf,"tau_utility":math.inf,"tau_harm":0})) ["token_f1"],
                "delta_f1":0.0,"delta_hit1_pp":0.0,"delta_hn1_pp":0.0,"switch_rate":0.0,"qa_forwards":1.0,"max_qa_forwards":1,"feasible":True};fallback=True
    utility_model=fit_ridge(rows,columns,alpha);harm_model=fit_classifier(rows,columns,"harm_target",c_harm);admission_model=fit_classifier(rows,ADMISSION_NUMERIC,"admission_target",1.0)
    chosen_payload=dict(chosen)
    for key in ("tau_admit","tau_utility","tau_harm"):
        value=chosen_payload.get(key)
        if value is not None and not math.isfinite(float(value)):chosen_payload[key]="inf" if float(value)>0 else "-inf"
    selection={"alpha":alpha,"ridge_oof_mae":mae,"C_harm":c_harm,"harm_oof_log_loss":loss,
      "tau_admit":chosen_payload["tau_admit"],"tau_utility":chosen_payload["tau_utility"],"tau_harm":chosen_payload["tau_harm"],
      "fold_policy":"B0_FALLBACK" if fallback else "ACTIVE","selected_metrics":chosen_payload,
      "ridge_trials":[{"alpha":x[0],"mae":x[2]} for x in ridge_trials],"harm_trials":[{"C":x[0],"log_loss":x[2]} for x in harm_trials],
      "policy_trials":trials,"meta_fold":assignment,"inner_folds_used":folds,"selection_source":"META_PRED_OOF",
      "utility_diagnostics":utility_diag,"harm_diagnostics":harm_diag,"admission_diagnostics":admission_diag}
    return FittedPolicy(system,utility_model,harm_model,admission_model,columns,selection)

def apply_policy(rows:list[dict],policy:FittedPolicy):
    mu,harm,admit=policy.predict(rows);s=policy.selection
    def val(key,default):
        x=s.get(key,default);return math.inf if x=="inf" else x
    thresholds={"tau_admit":val("tau_admit",math.inf),"tau_utility":val("tau_utility",math.inf),"tau_harm":val("tau_harm",0.0)}
    simulation_system="B0_FALLBACK" if s.get("fold_policy")=="B0_FALLBACK" else policy.system
    return _simulate(rows,mu,harm,admit,simulation_system,thresholds),(mu,harm,admit)
