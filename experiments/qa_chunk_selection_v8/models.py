"""Strict META_PRED_OOF utility/risk fitting for both V8 branches."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import hashlib, math
import numpy as np
from .contract import ALPHA_GRID,HARM_C_GRID,TAU_HARM_GRID,TAU_UTILITY_GRID

class ConstantRegressor:
    def __init__(self,value=0):self.value=float(value)
    def predict(self,frame):return np.full(len(frame),self.value)
class ConstantClassifier:
    def __init__(self,p=0):self.p=float(p)
    def predict_proba(self,frame):return np.column_stack([np.full(len(frame),1-self.p),np.full(len(frame),self.p)])

def fold_for(document_id,seed,folds):return int.from_bytes(hashlib.sha256(f"{seed}|{document_id}".encode()).digest()[:8],"big")%folds
def _frame(rows):
    import pandas as pd
    return pd.DataFrame(rows)
def fit_ridge(rows,columns,alpha):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    if not rows:return ConstantRegressor(0)
    y=np.asarray([r["utility"] for r in rows],float)
    if np.ptp(y)<1e-12:return ConstantRegressor(float(y.mean()))
    model=Pipeline([("scale",StandardScaler()),("model",Ridge(alpha=float(alpha)))])
    model.fit(_frame(rows)[list(columns)],y,model__sample_weight=np.asarray([r["weight"] for r in rows],float));return model
def fit_harm(rows,columns,c):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    if not rows:return ConstantClassifier(0)
    y=np.asarray([r["harm_target"] for r in rows],int)
    if len(set(y.tolist()))<2:return ConstantClassifier(float(y[0]))
    model=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=float(c),solver="liblinear",class_weight=None,max_iter=3000))])
    model.fit(_frame(rows)[list(columns)],y,model__sample_weight=np.asarray([r["weight"] for r in rows],float));return model
def predict_reg(model,rows,columns):return np.clip(np.asarray(model.predict(_frame(rows)[list(columns)]),float),-100,100)
def predict_harm(model,rows,columns):return np.asarray(model.predict_proba(_frame(rows)[list(columns)])[:,1],float)

def _oof(rows,seed,folds,fit,predict):
    assignment=[fold_for(r["canonical_document_id"],seed,folds) for r in rows];out=np.full(len(rows),np.nan)
    for f in sorted(set(assignment)):
        train=[r for r,a in zip(rows,assignment) if a!=f];idx=[i for i,a in enumerate(assignment) if a==f]
        if not train or not idx:continue
        model=fit(train);out[idx]=predict(model,[rows[i] for i in idx])
    if np.isnan(out).any():raise RuntimeError("META_PRED_OOF coverage incomplete")
    return out,assignment

def simulate(rows,mu,harm,use_harm,tu,th):
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[(row["dataset_cell"],row["question_id"])].append((row,float(mu[i]),float(harm[i])))
    choices=[]
    for (cell,qid),values in groups.items():
        first=values[0][0];safe=[x for x in values if x[1]>=tu and (not use_harm or x[2]<=th)]
        selected=max(safe,key=lambda x:(x[1],-x[2],-int(x[0]["b0_rank"])),default=None);row=selected[0] if selected else first
        choices.append({"dataset_cell":cell,"question_id":qid,"selected_chunk_id":int(row["chunk_id"] if selected else first["base_chunk_id"]),
          "f1":float(row["candidate_f1"] if selected else first["base_f1"]),"base_f1":float(first["base_f1"]),
          "hit1":int((row["candidate_tier"] if selected else first["base_tier"])=="sufficient_positive"),"base_hit1":int(first["base_tier"]=="sufficient_positive"),
          "hn1":int(row["candidate_hn"] if selected else first["base_hn"]),"base_hn1":int(first["base_hn"]),"switch":int(selected is not None)})
    return choices

def metrics(choices):
    mean=lambda k:float(np.mean([x[k] for x in choices])) if choices else 0.0
    return {"token_f1":mean("f1"),"baseline_token_f1":mean("base_f1"),"delta_f1":mean("f1")-mean("base_f1"),
      "delta_hit1_pp":100*(mean("hit1")-mean("base_hit1")),"delta_hn1_pp":100*(mean("hn1")-mean("base_hn1")),"switch_rate":mean("switch")}

def _coefficients(model,columns):
    if not hasattr(model,"named_steps"):return {name:0.0 for name in columns}
    coef=np.asarray(model.named_steps["model"].coef_,float).reshape(-1)
    return {name:float(value) for name,value in zip(columns,coef)}

@dataclass
class Policy:
    system:str;columns:tuple;utility_model:object;harm_model:object;selection:dict
    def predict(self,rows):return predict_reg(self.utility_model,rows,self.columns),predict_harm(self.harm_model,rows,self.columns)

def tune_policy(rows,system,columns,seed,folds,use_harm=True):
    if not rows:raise ValueError("no training alternatives after embargo")
    docs={r["canonical_document_id"] for r in rows};folds=max(2,min(folds,4,len(docs)));weights=np.asarray([r["weight"] for r in rows],float);truth=np.asarray([r["utility"] for r in rows],float)
    ridge=[]
    for alpha in ALPHA_GRID:
        pred,assignment=_oof(rows,seed,folds,lambda x:fit_ridge(x,columns,alpha),lambda m,x:predict_reg(m,x,columns));mae=float(np.average(np.abs(pred-truth),weights=weights));ridge.append((alpha,pred,mae,assignment))
    best=min(x[2] for x in ridge);alpha,mu,mae,assignment=max([x for x in ridge if x[2]<=best+.05],key=lambda x:x[0])
    from sklearn.metrics import log_loss
    harm_trials=[]
    for c in HARM_C_GRID:
        p,_=_oof(rows,seed,folds,lambda x:fit_harm(x,columns,c),lambda m,x:predict_harm(m,x,columns));loss=float(log_loss([r["harm_target"] for r in rows],np.clip(p,1e-8,1-1e-8),sample_weight=weights,labels=[0,1]));harm_trials.append((c,p,loss))
    best=min(x[2] for x in harm_trials);c_harm,harm,loss=min([x for x in harm_trials if x[2]<=best+.001],key=lambda x:x[0])
    trials=[]
    for tu in TAU_UTILITY_GRID:
      for th in (TAU_HARM_GRID if use_harm else (1.0,)):
        score=metrics(simulate(rows,mu,harm,use_harm,tu,th));score.update(tau_utility=tu,tau_harm=th,feasible=score["delta_hit1_pp"]>=-.5-1e-12 and score["delta_hn1_pp"]<=1+1e-12);trials.append(score)
    feasible=[x for x in trials if x["feasible"]]
    if feasible:chosen=max(feasible,key=lambda x:(x["token_f1"],-x["switch_rate"],x["tau_utility"],-x["tau_harm"]));fallback=False
    else:chosen={"tau_utility":"inf","tau_harm":0.0,"token_f1":metrics(simulate(rows,mu,harm,use_harm,math.inf,0))["token_f1"],"delta_f1":0.0,"delta_hit1_pp":0.0,"delta_hn1_pp":0.0,"switch_rate":0.0,"feasible":True};fallback=True
    utility=fit_ridge(rows,columns,alpha);harm_model=fit_harm(rows,columns,c_harm)
    selection={"alpha":alpha,"ridge_oof_mae":mae,"C_harm":c_harm,"harm_oof_log_loss":loss,"tau_utility":chosen["tau_utility"],"tau_harm":chosen["tau_harm"],
      "fold_policy":"BASE_FALLBACK" if fallback else "ACTIVE","selected_metrics":chosen,"ridge_trials":[{"alpha":x[0],"mae":x[2]} for x in ridge],
      "harm_trials":[{"C":x[0],"log_loss":x[2]} for x in harm_trials],"policy_trials":trials,"meta_fold":assignment,"inner_folds_used":folds,
      "selection_source":"META_PRED_OOF","standardized_coefficients":_coefficients(utility,columns)}
    return Policy(system,tuple(columns),utility,harm_model,selection)

def apply_policy(rows,policy,use_harm=True):
    mu,harm=policy.predict(rows);s=policy.selection;tu=math.inf if s["tau_utility"]=="inf" else float(s["tau_utility"]);th=float(s["tau_harm"])
    if s["fold_policy"]=="BASE_FALLBACK":tu=math.inf
    return simulate(rows,mu,harm,use_harm,tu,th),(mu,harm)
