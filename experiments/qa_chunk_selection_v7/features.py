"""Leakage-separated V7 admission and progressive selector features."""
from __future__ import annotations
from collections import defaultdict
import math
from experiments.qa_chunk_selection.v4.answers import pair_token_f1

RUNTIME_GOLD_FORBIDDEN=("gold","actual_utility","utility_target","positive_label","hn_label","oracle_rank","oracle_best")

def grouped(rows):
    out=defaultdict(list)
    for row in rows: out[str(row["question_id"])].append(row)
    return {qid:sorted(values,key=lambda x:(int(x["b0_rank"]),int(x["chunk_id"]))) for qid,values in out.items()}

def pairlex(answer_i:str,answer_0:str)->float:
    left=str(answer_i or "").strip();right=str(answer_0 or "").strip()
    if not left and not right:return 1.0
    if not left or not right:return 0.0
    return float(pair_token_f1(left,right)/100.0)

def candidate_rows(runtime_rows:list[dict],target:dict)->list[dict]:
    """Training/evaluation rows. Gold fields stay outside runtime prediction payloads."""
    values=sorted(runtime_rows,key=lambda x:int(x["b0_rank"]));
    if len(values)<=1:return []
    top=values[0];k=len(values);out=[]
    top_answer=str(top.get("qa_answer") or "");top_valid=int(bool(top_answer.strip()))
    for alt in values[1:]:
        answer=str(alt.get("qa_answer") or "");valid=int(bool(answer.strip()))
        dq=float(alt["qarel_raw"])-float(top["qarel_raw"]) if valid and top_valid else 0.0
        rank=int(alt["b0_rank"]);den=max(k-1,1)
        row={
          "question_id":str(alt["question_id"]),"document_id":str(alt["document_id"]),
          "canonical_document_id":str(alt["canonical_document_id"]),"document_sha256":str(alt["document_sha256"]),
          "question_lineage_id":str(alt["question_lineage_id"]),"dataset_cell":str(alt["dataset_cell"]),
          "corpus_family":str(alt["corpus_family"]),"language":str(alt["language"]),
          "canonical_aspect":str(alt["canonical_aspect"]),"outer_fold":int(alt["outer_fold"]),"seed":int(alt["seed"]),
          "chunk_id":int(alt["chunk_id"]),"b0_chunk_id":int(top["chunk_id"]),"b0_rank":rank,
          "candidate_rank_norm":float((rank-1)/den),"b0_score_norm":float(alt["b0_score_norm"]),
          "delta_b0_norm":float(alt["b0_score_norm"])-float(top["b0_score_norm"]),
          "effective_k":int(alt["effective_k"]),"candidate_count":int(alt["candidate_count"]),
          "gap12_norm":float(alt["gap12_norm"]),"gap13_norm":float(alt["gap13_norm"]),
          "b0_entropy_norm":float(alt["b0_entropy_norm"]),"top1_z":float(alt["top1_z"]),
          "delta_qarel_raw":dq,"pairlex":pairlex(answer,top_answer),"historical_full_lex_norm":float(alt.get("lex_norm",0.0)),
          "answer_valid_i":valid,"answer_valid_0":top_valid,"both_empty":int(not valid and not top_valid),
          "qa_answer":answer,"baseline_answer":top_answer,
          "utility":float(target["candidate_f1"][str(alt["chunk_id"])])-float(target["baseline_f1"]),
          "candidate_f1":float(target["candidate_f1"][str(alt["chunk_id"])]),"baseline_f1":float(target["baseline_f1"]),
          "candidate_tier":str(target["label_tier"][str(alt["chunk_id"])]),"baseline_tier":str(target["label_tier"][str(top["chunk_id"])]),
          "candidate_hn":int(target["hard_negative"][str(alt["chunk_id"])]),"baseline_hn":int(target["hard_negative"][str(top["chunk_id"])]),
          "admission_target":int(float(target["candidate_f1"][str(alt["chunk_id"])])>float(target["baseline_f1"])),
          "harm_target":int(float(target["candidate_f1"][str(alt["chunk_id"])])<float(target["baseline_f1"])),
          "weight":1.0/(k-1),"upstream_feature_origin":alt.get("upstream_feature_origin"),
        }
        row.update({"qarel_x_margin":dq*row["gap12_norm"],"pairlex_x_margin":row["pairlex"]*row["gap12_norm"],
                    "qarel_x_pairlex":dq*row["pairlex"],"b0_x_margin":row["delta_b0_norm"]*row["gap12_norm"]})
        out.append(row)
    return out

def preqa_view(row):
    keys=("question_id","document_id","canonical_document_id","document_sha256","question_lineage_id","dataset_cell","corpus_family","language",
          "outer_fold","seed","chunk_id","b0_chunk_id","b0_rank","candidate_rank_norm","b0_score_norm","delta_b0_norm","effective_k","candidate_count",
          "gap12_norm","gap13_norm","b0_entropy_norm","top1_z","admission_target","weight")
    return {key:row[key] for key in keys}

def assert_runtime_gold_free(runtime):
    def visit(value,path="runtime"):
        if isinstance(value,dict):
            for key,child in value.items():
                low=str(key).lower()
                if any(token in low for token in RUNTIME_GOLD_FORBIDDEN): raise AssertionError(f"gold field leaked at {path}.{key}")
                visit(child,f"{path}.{key}")
        elif isinstance(value,list):
            for i,child in enumerate(value):visit(child,f"{path}[{i}]")
    visit(runtime)

def normalize_weights(rows:list[dict],pooled:bool)->dict:
    by_q=defaultdict(list)
    for r in rows:by_q[(r["dataset_cell"],r["question_id"])].append(r)
    for values in by_q.values():
        total=sum(float(x.get("weight",1)) for x in values) or 1
        for x in values:x["weight"]=float(x.get("weight",1))/total
    by_lineage=defaultdict(list)
    for r in rows:by_lineage[r["question_lineage_id"]].append(r)
    for values in by_lineage.values():
        cells={x["dataset_cell"] for x in values};div=len(cells) if any(x.startswith("scier_") for x in cells) else 1
        for x in values:x["weight"]/=div
    if pooled:
        by_family=defaultdict(list)
        for r in rows:by_family[r["corpus_family"]].append(r)
        for values in by_family.values():
            total=sum(x["weight"] for x in values) or 1
            for x in values:x["weight"]/=total
    return {"questions":len(by_q),"lineages":len(by_lineage),"families":{f:sum(x["weight"] for x in rows if x["corpus_family"]==f) for f in sorted({x["corpus_family"] for x in rows})}}
