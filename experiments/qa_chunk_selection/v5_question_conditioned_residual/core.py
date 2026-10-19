"""Pure V5 feature, gate and mirrored residual implementations."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .contract import (BUDGETS, CONTEXT, EPSILON, FEATURE_SETS, GATE_CONTINUOUS,
                       PRIMARY_BUDGET, SWITCH_THRESHOLD)


def zscores(values):
    x = np.asarray(values, dtype=float)
    std = float(x.std())
    if std < EPSILON:
        return np.zeros_like(x)
    return (x - float(x.mean())) / (std + EPSILON)


def uncertainty(scores, candidate_count=None):
    values = np.asarray(scores, dtype=float)
    if not len(values): raise ValueError("empty shortlist")
    z = zscores(values)
    exp = np.exp(z - z.max()); p = exp / exp.sum()
    gap = lambda i: float((values[0] - values[min(i, len(values)-1)]) / (float(values.std()) + EPSILON))
    q75, q25 = np.percentile(values, [75, 25])
    return {
        "effective_k": int(len(values)), "candidate_count": int(candidate_count or len(values)),
        "normalized_gap_12": gap(1), "normalized_gap_13": gap(2),
        "normalized_gap_1k": gap(len(values)-1), "topk_score_std": float(values.std()),
        "topk_score_iqr": float(q75-q25),
        "normalized_score_entropy": float(-(p*np.log(p + EPSILON)).sum()),
        "top1_zscore": float(z[0]),
    }


def normalize_candidates(values):
    return [float(v) for v in zscores(values)]


def feature_rows(question_id, document_id, aspect, seed, rank_ids, raw_scores,
                 qa_answers, qa_relevance, label_tiers, candidate_count, secondary=None):
    ids = [int(cid) for cid in rank_ids]
    b0 = normalize_candidates([raw_scores[cid] for cid in ids])
    qar = normalize_candidates([qa_relevance[cid] for cid in ids])
    secondary = secondary or {}
    source = normalize_candidates([secondary.get("source_support",{}).get(cid,0.0) for cid in ids])
    groups = normalize_candidates([secondary.get("answer_group",{}).get(cid,0.0) for cid in ids])
    ctx = uncertainty([raw_scores[cid] for cid in ids], candidate_count)
    rows=[]
    for pos,cid in enumerate(ids):
        rows.append({
            "question_id": str(question_id), "document_id": str(document_id), "aspect_id": str(aspect),
            "seed": int(seed), "chunk_id": cid, "b0_rank": pos+1,
            "b0_score_raw": float(raw_scores[cid]), "b0_score_norm": b0[pos],
            "q_answer_ranker_raw": float(qa_relevance[cid]), "q_answer_ranker_norm": qar[pos],
            "source_support_norm": source[pos], "answer_group_norm": groups[pos],
            "qa_answer": str(qa_answers[cid]), "label_tier": str(label_tiers[cid]), **ctx,
        })
    return rows


def gate_row(rows):
    first=rows[0]
    need = first["label_tier"] != "sufficient_positive" and any(r["label_tier"]=="sufficient_positive" for r in rows)
    return {k:first[k] for k in ("question_id","document_id","aspect_id","seed",*GATE_CONTINUOUS)} | {"target":int(need)}


def pair_rows(rows, feature_set="A1"):
    positives=[r for r in rows if r["label_tier"]=="sufficient_positive"]
    negatives=[r for r in rows if r["label_tier"]=="insufficient_evidence"]
    if not positives or not negatives:return []
    pairs=[]; n=len(positives)*len(negatives); unit=1.0/n/2.0
    for pos in positives:
        for neg in negatives:
            for left,right,label in ((pos,neg,1),(neg,pos,0)):
                row={"question_id":left["question_id"],"document_id":left["document_id"],
                     "aspect_id":left["aspect_id"],"seed":left["seed"],"target":label,"weight":unit}
                row["delta_b0"]=left["b0_score_norm"]-right["b0_score_norm"]
                row["delta_q_answer_ranker"]=left["q_answer_ranker_norm"]-right["q_answer_ranker_norm"]
                row["delta_source_support"]=left["source_support_norm"]-right["source_support_norm"]
                row["delta_answer_group"]=left["answer_group_norm"]-right["answer_group_norm"]
                for name in CONTEXT:row[name]=left[name]
                pairs.append({k:v for k,v in row.items() if k in set(FEATURE_SETS[feature_set])|{"question_id","document_id","aspect_id","seed","target","weight"}})
    return pairs


def _pipeline(columns, seed, balanced):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    return Pipeline([
        ("pre",ColumnTransformer([("continuous",StandardScaler(),list(columns)),
                                   ("aspect",OneHotEncoder(handle_unknown="ignore"),["aspect_id"])])),
        ("model",LogisticRegression(C=1.0,penalty="l2",solver="liblinear",class_weight="balanced" if balanced else None,max_iter=2000,random_state=seed)),
    ])


class ConstantProbability:
    def __init__(self,p):self.p=float(p)
    def predict_proba(self,rows):
        return np.asarray([[1-self.p,self.p] for _ in range(len(rows))])


def fit_gate(rows, seed):
    import pandas as pd
    if not rows: return ConstantProbability(0.0)
    frame=pd.DataFrame(rows); y=frame.target.astype(int)
    if y.nunique()<2:return ConstantProbability(float(y.iloc[0]))
    model=_pipeline(GATE_CONTINUOUS,seed,True); model.fit(frame,y); return model


def fit_residual(rows, seed, feature_set="A1"):
    import pandas as pd
    if not rows:return ConstantProbability(0.5)
    frame=pd.DataFrame(rows); y=frame.target.astype(int)
    model=_pipeline(FEATURE_SETS[feature_set],seed,False)
    model.fit(frame,y,model__sample_weight=frame.weight.astype(float)); return model


def choose_budget_threshold(probabilities, effective_k, budget=PRIMARY_BUDGET):
    p=np.asarray(probabilities,float); k=np.asarray(effective_k,int)
    candidates=sorted(set([1.0+EPSILON,*p.tolist(),0.0]))
    feasible=[]
    for tau in candidates:
        routed=(p>=tau)&(k>1); cost=float(np.mean(np.where(routed,k,1)))
        if cost<=budget+EPSILON:feasible.append((tau,int(routed.sum()),cost))
    if not feasible:return 1.0+EPSILON
    max_routes=max(x[1] for x in feasible)
    return float(max(x[0] for x in feasible if x[1]==max_routes))


def thresholds_for_budgets(probabilities,effective_k):
    return {str(b):choose_budget_threshold(probabilities,effective_k,b) for b in BUDGETS}


def runtime_choice(rows, model, feature_set="A1"):
    import pandas as pd
    top=rows[0]; candidates=[]
    for alt in rows[1:]:
        row={"aspect_id":alt["aspect_id"],"delta_b0":alt["b0_score_norm"]-top["b0_score_norm"],
             "delta_q_answer_ranker":alt["q_answer_ranker_norm"]-top["q_answer_ranker_norm"],
             "delta_source_support":alt["source_support_norm"]-top["source_support_norm"],
             "delta_answer_group":alt["answer_group_norm"]-top["answer_group_norm"]}
        for name in CONTEXT:row[name]=alt[name]
        candidates.append((alt,row))
    if not candidates:return top,0.0
    probs=model.predict_proba(pd.DataFrame([x[1] for x in candidates]))[:,1]
    index=int(np.argmax(probs))
    return (candidates[index][0],float(probs[index])) if probs[index]>SWITCH_THRESHOLD else (top,float(probs[index]))

