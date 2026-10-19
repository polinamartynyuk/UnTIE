"""Exact recovered reader features and branch-specific training rows."""
from __future__ import annotations
from collections import defaultdict
from itertools import product
import copy, math
import numpy as np
from experiments.qa_chunk_selection.v4.v6_reader import agreement_features
from experiments.qa_chunk_selection.v4.answers import pair_token_f1
from experiments.qa_chunk_selection_v7.features import candidate_rows as v7_candidate_rows
from experiments.qa_chunk_selection_v7.models import fold_for
from .contract import V44_WEIGHT_GRID

RUNTIME_GOLD_FORBIDDEN=("gold","actual_utility","utility_target","positive_label","hn_label","oracle_rank","oracle_best")

def grouped(rows):
    out=defaultdict(list)
    for row in rows:out[(str(row["dataset_cell"]),str(row["question_id"]))].append(row)
    return {key:sorted(value,key=lambda x:(int(x["b0_rank"]),int(x["chunk_id"]))) for key,value in out.items()}

def _z(values):
    x=np.asarray(values,float);scale=float(x.std())
    return np.zeros(len(x),float) if scale<1e-8 else (x-float(x.mean()))/scale

def pairlex(answer_i,answer_0):
    a=str(answer_i or "").strip();b=str(answer_0 or "").strip()
    if not a and not b:return 1.0
    if not a or not b:return 0.0
    return float(pair_token_f1(a,b)/100.0)

def enrich_reader_features(runtime_rows,item):
    """Recover exact V4.4 agreement fields on the frozen canonical top-K."""
    values=sorted(copy.deepcopy(runtime_rows),key=lambda x:(int(x["b0_rank"]),int(x["chunk_id"])))
    b0={"ranked_ids":[int(x["chunk_id"]) for x in values],"scores":{str(x["chunk_id"]):float(x["b0_score_raw"]) for x in values}}
    qa={int(x["chunk_id"]):{"qa_answer":str(x.get("qa_answer") or ""),"qa_raw_score":0.0,"qa_margin":0.0} for x in values}
    exact=agreement_features(item,b0,qa,k=len(values),min_overlap=.6)
    full=[float(exact[int(x["chunk_id"])]["mean_token_f1"])/100.0 for x in values]
    full_z=_z(full)
    for index,row in enumerate(values):
        feat=exact[int(row["chunk_id"])]
        row.update({
            "agreement_exact_freq":float(feat["exact_freq"]),
            "agreement_majority":float(feat["majority"]),
            "agreement_raw":float(feat["exact_freq"])+float(feat["majority"]),
            "fulllex_raw":full[index],"fulllex_norm":float(full_z[index]),
            "support_raw":float(feat["source_clusters"]),
            "v44_answer_raw":float(feat["exact_freq"])+float(feat["mean_token_f1"])+float(feat["majority"]),
        })
    return values

def hrr_rows(runtime_rows,target):
    base=v7_candidate_rows(runtime_rows,target);byid={int(x["chunk_id"]):x for x in runtime_rows};top=min(runtime_rows,key=lambda x:int(x["b0_rank"]))
    out=[]
    for row in base:
        alt=byid[int(row["chunk_id"])]
        row["delta_fulllex"]=float(alt["fulllex_norm"])-float(top["fulllex_norm"])
        row["delta_agreement"]=float(alt["agreement_raw"])-float(top["agreement_raw"])
        row["delta_sem"]=0.0
        row["fulllex_x_margin"]=row["delta_fulllex"]*row["gap12_norm"]
        row["qarel_x_fulllex"]=row["delta_qarel_raw"]*row["delta_fulllex"]
        row["agreement_x_margin"]=row["delta_agreement"]*row["gap12_norm"]
        row["fulllex_x_agreement"]=row["delta_fulllex"]*row["delta_agreement"]
        row.update(base_chunk_id=int(top["chunk_id"]),base_answer=str(top.get("qa_answer") or ""),base_kind="B0",
                   base_f1=float(row["baseline_f1"]),base_tier=str(row["baseline_tier"]),base_hn=int(row["baseline_hn"]))
        out.append(row)
    return out

def _question_candidates(values,target):
    result=[]
    for row in values:
        cid=int(row["chunk_id"])
        result.append({"runtime":row,"chunk_id":cid,"f1":float(target["candidate_f1"][str(cid)]),
          "tier":str(target["label_tier"][str(cid)]),"hn":int(target["hard_negative"][str(cid)])})
    return result

def _core_score(row,params):
    return float(params["alpha"])*float(row["b0_score_raw"])+float(params["beta"])*float(row["v44_answer_raw"])+float(params["gamma"])*float(row["support_raw"])

def apply_v44_core(runtime,targets,params):
    result={}
    for key,values in grouped(runtime).items():
        candidates=_question_candidates(values,targets[key])
        ranked=sorted(candidates,key=lambda x:(-_core_score(x["runtime"],params),int(x["runtime"]["b0_rank"]),x["chunk_id"]))
        result[key]={"selected":ranked[0],"ranked":ranked,"scores":{x["chunk_id"]:_core_score(x["runtime"],params) for x in candidates}}
    return result

def _core_metrics(selection):
    rows=[x["selected"] for x in selection.values()]
    return {"token_f1":float(np.mean([x["f1"] for x in rows])) if rows else 0.0,
      "hit@1":float(np.mean([x["tier"]=="sufficient_positive" for x in rows])) if rows else 0.0,
      "hn@1":float(np.mean([x["hn"] for x in rows])) if rows else 0.0}

def fit_v44_core(runtime,targets,seed):
    """Exact V4.4 score/weight family with source-fold stability selection."""
    baseline={"alpha":1.0,"beta":0.0,"gamma":0.0};base_selection=apply_v44_core(runtime,targets,baseline);base=_core_metrics(base_selection)
    docs=sorted({x["canonical_document_id"] for x in runtime});folds=min(4,len(docs));assign={d:fold_for(d,seed,max(folds,2)) for d in docs}
    candidates=[{"alpha":a,"beta":b,"gamma":g} for a,b,g in product(V44_WEIGHT_GRID,repeat=3)];trials=[]
    for params in candidates:
        selected=apply_v44_core(runtime,targets,params);metric=_core_metrics(selected);fold_delta=[]
        for f in range(max(folds,2)):
            keys=[k for k,v in selected.items() if assign[v["selected"]["runtime"]["canonical_document_id"]]==f]
            if not keys:continue
            fold_delta.append(_core_metrics({k:selected[k] for k in keys})["token_f1"]-_core_metrics({k:base_selection[k] for k in keys})["token_f1"])
        safe=metric["hit@1"]>=base["hit@1"]-.005-1e-12 and metric["hn@1"]<=base["hn@1"]+.02+1e-12
        stable=safe and metric["token_f1"]>base["token_f1"]+1e-9 and sum(x>1e-9 for x in fold_delta)>=math.ceil(max(len(fold_delta),1)/2)
        trials.append({"params":params,"metrics":metric,"delta_f1":metric["token_f1"]-base["token_f1"],"fold_deltas":fold_delta,
                       "positive_folds":sum(x>1e-9 for x in fold_delta),"worst_fold_delta":min(fold_delta,default=0.0),"safe":safe,"stable":stable})
    eligible=[x for x in trials if x["stable"]]
    chosen=max(eligible,key=lambda x:(x["delta_f1"],x["positive_folds"],x["worst_fold_delta"],-sum(abs(float(v)) for v in x["params"].values()))) if eligible else None
    return (dict(chosen["params"]) if chosen else baseline),{"baseline":base,"selected":chosen,"fallback":chosen is None,
      "criterion":"exact V4.4 source-fold majority improvement + Hit/HN safety","n_candidates":len(trials),"tried":trials}

def af_rows(runtime,target_map,core_params):
    core=apply_v44_core(runtime,target_map,core_params);out=[]
    for key,values in grouped(runtime).items():
        state=core[key];anchor=state["selected"];anchor_row=anchor["runtime"];b0=values[0];scores=[state["scores"][int(x["chunk_id"])] for x in values];score_z=_z(scores);zmap={int(x["chunk_id"]):float(z) for x,z in zip(values,score_z)}
        ordered=sorted(scores,reverse=True);v44_margin=float(ordered[0]-ordered[1])/max(float(np.std(scores)),1e-8) if len(scores)>1 else 0.0
        pair_anchor=pairlex(anchor_row.get("qa_answer"),b0.get("qa_answer"))
        for alt in values:
            cid=int(alt["chunk_id"])
            if cid==anchor["chunk_id"]:continue
            pair=pairlex(alt.get("qa_answer"),b0.get("qa_answer"));den=max(len(values)-1,1)
            row={
              "question_id":str(alt["question_id"]),"document_id":str(alt["document_id"]),"canonical_document_id":str(alt["canonical_document_id"]),
              "document_sha256":str(alt["document_sha256"]),"question_lineage_id":str(alt["question_lineage_id"]),"dataset_cell":str(alt["dataset_cell"]),
              "corpus_family":str(alt["corpus_family"]),"language":str(alt["language"]),"canonical_aspect":str(alt["canonical_aspect"]),
              "outer_fold":int(alt["outer_fold"]),"seed":int(alt["seed"]),"chunk_id":cid,"base_chunk_id":int(anchor["chunk_id"]),
              "b0_chunk_id":int(b0["chunk_id"]),"b0_rank":int(alt["b0_rank"]),"candidate_rank_norm":float((int(alt["b0_rank"])-1)/den),
              "effective_k":int(alt["effective_k"]),"delta_s44":zmap[cid]-zmap[int(anchor["chunk_id"])],"v44_margin":v44_margin,
              "b0_margin":float(alt["gap12_norm"]),"delta_fulllex":float(alt["fulllex_norm"])-float(anchor_row["fulllex_norm"]),
              "delta_pairlex":pair-pair_anchor,"pairlex_b0_raw":pair,"delta_qarel":float(alt["qarel_norm"])-float(anchor_row["qarel_norm"]),"delta_sem":0.0,
              "answer_valid_i":int(alt["answer_valid"]),"answer_valid_0":int(anchor_row["answer_valid"]),
              "qa_answer":str(alt.get("qa_answer") or ""),"base_answer":str(anchor_row.get("qa_answer") or ""),"base_kind":"V44_CORE",
              "candidate_f1":float(target_map[key]["candidate_f1"][str(cid)]),"base_f1":float(anchor["f1"]),
              "candidate_tier":str(target_map[key]["label_tier"][str(cid)]),"base_tier":str(anchor["tier"]),
              "candidate_hn":int(target_map[key]["hard_negative"][str(cid)]),"base_hn":int(anchor["hn"]),
              "utility":float(target_map[key]["candidate_f1"][str(cid)])-float(anchor["f1"]),"harm_target":int(float(target_map[key]["candidate_f1"][str(cid)])<float(anchor["f1"])),
              "weight":1.0/max(len(values)-1,1),"s44_raw":float(state["scores"][cid]),"base_s44_raw":float(state["scores"][int(anchor["chunk_id"])]),
            }
            row.update(fulllex_x_v44_margin=row["delta_fulllex"]*v44_margin,pairlex_x_v44_margin=row["delta_pairlex"]*v44_margin,
              fulllex_x_pairlex=row["delta_fulllex"]*row["delta_pairlex"],qarel_x_v44_margin=row["delta_qarel"]*v44_margin,
              fulllex_x_qarel=row["delta_fulllex"]*row["delta_qarel"],sem_x_v44_margin=0.0,fulllex_x_sem=0.0)
            out.append(row)
    return out,core

def normalize_weights(rows,pooled):
    byq=defaultdict(list)
    for r in rows:byq[(r["dataset_cell"],r["question_id"])].append(r)
    for values in byq.values():
        total=sum(float(x.get("weight",1)) for x in values) or 1
        for x in values:x["weight"]=float(x.get("weight",1))/total
    bylineage=defaultdict(list)
    for r in rows:bylineage[r["question_lineage_id"]].append(r)
    for values in bylineage.values():
        cells={x["dataset_cell"] for x in values};div=len(cells) if any(x.startswith("scier_") for x in cells) else 1
        for x in values:x["weight"]/=div
    if pooled:
        byfamily=defaultdict(list)
        for r in rows:byfamily[r["corpus_family"]].append(r)
        for values in byfamily.values():
            total=sum(x["weight"] for x in values) or 1
            for x in values:x["weight"]/=total
    return {"questions":len(byq),"lineages":len(bylineage),"families":{f:sum(x["weight"] for x in rows if x["corpus_family"]==f) for f in sorted({x["corpus_family"] for x in rows})}}

def assert_runtime_gold_free(value,path="runtime"):
    if isinstance(value,dict):
        for key,child in value.items():
            if any(token in str(key).lower() for token in RUNTIME_GOLD_FORBIDDEN):raise AssertionError(f"gold field leaked at {path}.{key}")
            assert_runtime_gold_free(child,f"{path}.{key}")
    elif isinstance(value,list):
        for i,child in enumerate(value):assert_runtime_gold_free(child,f"{path}[{i}]")
