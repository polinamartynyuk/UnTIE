"""Adapters around frozen historical datasets, readers and B0 training code."""
from __future__ import annotations

import gc
import hashlib
import json
import shutil
from functools import lru_cache
from collections import defaultdict
from pathlib import Path

from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v2.hn_flags import hn_quality_rows
from experiments.qa_chunk_selection.v2.schemas import to_runtime
from experiments.qa_chunk_selection.v3.data import records_from_jsonl
from experiments.qa_chunk_selection.v3.split import records_for_role
from experiments.qa_chunk_selection.v3.train import score_fn_for, train_b0
from experiments.qa_chunk_selection.v4_1.evaluate import role_of_document
from experiments.qa_chunk_selection.v4_4_ru_q5_b0.pipeline import choose_device
from experiments.qa_chunk_selection.v4_4_ru_ranker_factorial.cross_encoder import train_cross_encoder
from experiments.qa_chunk_selection.v4_5_ru_compact_refinement.contract import RankerSpec

from .contract import DITY_RECIPE, INSUFFICIENT, LEGACY_RECIPE, MINILM_SNAPSHOT, SUFFICIENT, UNRESOLVED
from .io_guard import canonical_hash, read_jsonl, sha256, write_json, write_jsonl



@lru_cache(maxsize=8)
def tree_hash(root):
    root=Path(root)
    rows=[(str(path.relative_to(root)),sha256(path)) for path in sorted(root.rglob("*")) if path.is_file()]
    return {"root":str(root.resolve()),"files":len(rows),"sha256":canonical_hash(rows)}

def _text_hash(text): return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def tier(chunk):
    label=str(getattr(chunk,"label","") or "")
    primary=str(getattr(chunk,"primary_class","") or "")
    if label in SUFFICIENT or primary in SUFFICIENT:return "sufficient_positive"
    if label in {"hard_negative","easy_negative"} or primary in INSUFFICIENT:return "insufficient_evidence"
    if label in UNRESOLVED or primary in UNRESOLVED:return "unresolved_for_training"
    return "unresolved_for_training"


class HistoricalCell:
    def __init__(self, spec, output: Path, device="auto", smoke=False, local_files_only=False,
                 progress: ProgressLogger | None=None):
        self.spec=spec; self.output=Path(output); self.smoke=smoke
        self.local_files_only=local_files_only; self.progress=progress
        self.device=choose_device(device)[0]
        all_records=records_from_jsonl(spec.records)
        historical_qids={str(row["question_id"]) for row in read_jsonl(spec.historical_b0)}
        self.records=[item for item in all_records if str(item.question_id) in historical_qids]
        if {str(item.question_id) for item in self.records} != historical_qids:
            missing=historical_qids-{str(item.question_id) for item in self.records}
            raise ValueError(f"{spec.key}: historical B0 questions missing from dataset: {sorted(missing)[:5]}")
        self.items={str(x.question_id):x for x in self.records}
        self.splits=json.loads(spec.splits.read_text(encoding="utf-8"))
        self.flags=read_jsonl(spec.flags) if spec.flags and spec.flags.is_file() else hn_quality_rows(self.records)
        self.questions=json.loads(spec.questions_q5.read_text(encoding="utf-8")) if spec.questions_q5 else {}
        self.frozen_questions=json.loads(spec.question_selection.read_text(encoding="utf-8")) if spec.question_selection else {}
        self._qa={}; self._load_qa_cache()
        self._reader=None

    def _load_qa_cache(self):
        for row in read_jsonl(self.spec.qa_cache):
            qid=str(row.get("qa_instance_id") or row.get("question_id"))
            if qid not in self.items and row.get("document_id") is not None and row.get("aspect") is not None:
                qid=f"{row['document_id']}::{row['aspect']}"
            cid=int(row.get("chunk_id",row.get("qa_chunk_id")))
            question=str(row.get("question") or (self.items[qid].question if qid in self.items else ""))
            answer=str(row.get("qa_predicted_answer") or row.get("qa_answer") or "")
            self._qa[(qid,cid,_text_hash(question))]={"answer":answer,"question":question,"source":"historical",
                "start":int(row.get("answer_start") or row.get("qa_chunk_start_char") or 0),
                "end":int(row.get("answer_end") or row.get("qa_chunk_end_char") or 0),
                "qa_correct":bool(row.get("qa_correct",False)),"gold_supporting":bool(row.get("gold_supporting",False))}
        new=self.output/"cache/qa"/f"{self.spec.key}.jsonl"
        if new.is_file():
            for row in read_jsonl(new):self._qa[(row["question_id"],int(row["chunk_id"]),row["question_hash"])]=row

    def _ensure_reader(self):
        if self._reader is not None:return
        from untie.config import PipelineConfig
        from untie.models import ModelFactory, profile_for_language
        profile=profile_for_language(self.spec.language)
        if Path(profile.qa_model).resolve()!=self.spec.qa_model.resolve():
            raise RuntimeError(f"BLOCKED_MODEL_RECOVERY: {self.spec.key} QA model differs: {profile.qa_model}")
        self._reader=ModelFactory(PipelineConfig(profile=profile,device=self.device)).answerer

    def qa(self,item,cid,question):
        key=(str(item.question_id),int(cid),_text_hash(question))
        if key in self._qa:return self._qa[key]["answer"]
        self._ensure_reader(); chunk=item.chunk_by_id(int(cid)); raw=dict(self._reader(question=question,context=chunk.text))
        row={"question_id":str(item.question_id),"chunk_id":int(cid),"question_hash":key[2],
             "chunk_hash":_text_hash(chunk.text),"qa_model":str(self.spec.qa_model.resolve()),
             "answer":str(raw.get("answer") or ""),"start":int(raw.get("start") or 0),
             "end":int(raw.get("end") or 0),"source":"v5_new_cache"}
        self._qa[key]=row
        path=self.output/"cache/qa"/f"{self.spec.key}.jsonl"
        current=read_jsonl(path) if path.is_file() else []
        write_jsonl(path,[*current,row])
        return row["answer"]

    def outer_records(self,run,role):
        return list(records_for_role(self.records,self.splits,run=run,role=role))

    def meta_source(self,run):
        return [x for x in self.records if role_of_document(x.document_id,self.splits,run)!="eval"]

    def question_map(self, records, *, outer_run=None):
        aspects=sorted({x.aspect for x in records})
        if self.spec.question_policy=="fixed_q0":return {a:None for a in aspects}
        docs={str(x.document_id) for x in records}; scores=defaultdict(lambda:[0,0])
        for (qid,_cid,_qh),row in self._qa.items():
            item=self.items.get(qid)
            if item is None or str(item.document_id) not in docs or not row.get("gold_supporting"):continue
            try:variant=self.questions[item.aspect].index(row["question"])
            except (ValueError,KeyError):continue
            scores[item.aspect,variant][0]+=int(row.get("qa_correct",False)); scores[item.aspect,variant][1]+=1
        selected={}
        for aspect in aspects:
            choices=[]
            for variant in range(len(self.questions[aspect])):
                good,total=scores[aspect,variant]; choices.append((good/max(total,1),good,-variant,variant))
            selected[aspect]=max(choices)[3]
        if outer_run is not None:
            expected={k:int(v) for k,v in self.frozen_questions[str(outer_run)]["question"].items()}
            if selected!=expected:raise RuntimeError(f"BLOCKED_NESTED_QUESTION_POLICY: historical replay mismatch {self.spec.key}/run{outer_run}: {selected} != {expected}")
        return selected

    def question(self,item,mapping):
        variant=mapping.get(item.aspect)
        return item.question if variant is None else str(self.questions[item.aspect][int(variant)])

    def train_ranker(self,train,calibration,seed,work):
        work=Path(work); work.mkdir(parents=True,exist_ok=True)
        if self.spec.ranker_recipe=="legacy_b2a":
            result=train_b0(train,calibration,self.flags,seed=seed,device=self.device,
                model_name=str(MINILM_SNAPSHOT),output_dir=work/"model",smoke=self.smoke,
                progress=self.progress,**LEGACY_RECIPE)
            return result["scorer"],{"recipe":"legacy_b2a","calibration":result.get("calibration")}
        ranker=RankerSpec("dity",self.spec.ranker_id,self.spec.ranker_revision,"historical_v45")
        scorer,report=train_cross_encoder(train,calibration,self.flags,spec=ranker,seed=seed,
            device=self.device,progress=self.progress,report_path=work/"training_report.json",
            local_files_only=self.local_files_only,smoke=self.smoke,**DITY_RECIPE)
        return scorer,report

    def score(self,scorer,item):
        fn=score_fn_for(scorer); runtime=to_runtime(item)
        values=fn(runtime)
        return {int(chunk.chunk_id):float(value) for chunk,value in zip(runtime.chunks,values)}

    def qa_relevance(self,scorer,question,answers):
        values=scorer.predict([(question,answers[cid]) for cid in answers])
        return {cid:float(value) for cid,value in zip(answers,values)}

    def retain_ranker(self,scorer,work):
        target=Path(work)/"model"
        if not target.exists() and hasattr(scorer,"model") and hasattr(scorer,"tokenizer"):
            target.mkdir(parents=True,exist_ok=True)
            scorer.model.save_pretrained(target)
            scorer.tokenizer.save_pretrained(target)
        write_json(Path(work)/"retained_model.json",{"path":str(target.resolve()),"model_id":self.spec.ranker_id,
            "revision":self.spec.ranker_revision,"purpose":"fresh-process resource benchmark"})

    def release(self,scorer,work,delete_model=True):
        del scorer; self._reader=None; gc.collect()
        if delete_model:
            shutil.rmtree(Path(work)/"model",ignore_errors=True)
        try:
            import torch
            if torch.cuda.is_available():torch.cuda.empty_cache()
        except ImportError:pass

    def audit(self):
        required={"records":self.spec.records,"splits":self.spec.splits,"qa_cache":self.spec.qa_cache,
                  "historical_b0":self.spec.historical_b0,"historical_reference":self.spec.historical_reference,
                  "qa_model_config":self.spec.qa_model/"config.json"}
        if self.spec.questions_q5:required["questions_q5"]=self.spec.questions_q5
        if self.spec.ranker_recipe=="legacy_b2a":
            required["ranker_snapshot_config"]=MINILM_SNAPSHOT/"config.json"
        missing=[str(p) for p in required.values() if not p.is_file()]
        if missing:raise FileNotFoundError(f"{self.spec.key} historical inputs missing: {missing}")
        candidates=[[str(x.question_id),[int(c.chunk_id) for c in x.chunks]] for x in self.records]
        labels=defaultdict(int)
        for item in self.records:
            for chunk in item.chunks:labels[tier(chunk)]+=1
        return {"cell":self.spec.key,"corpus":self.spec.corpus,"language":self.spec.language,
            "documents":len({x.document_id for x in self.records}),"questions":len(self.records),"k":self.spec.k,
            "ranker":{"model_id":self.spec.ranker_id,"revision":self.spec.ranker_revision,"recipe":self.spec.ranker_recipe},
            "qa_reader":{"model_id":str(self.spec.qa_model.resolve()),"tree":tree_hash(self.spec.qa_model)},
            "ranker_source":{"model_id":self.spec.ranker_id,"revision":self.spec.ranker_revision,
                             "tree":tree_hash(MINILM_SNAPSHOT) if self.spec.ranker_recipe=="legacy_b2a" else None},
            "candidate_universe_hash":canonical_hash(candidates),"split_hash":sha256(self.spec.splits),
            "source_hashes":{k:sha256(v) for k,v in required.items()},"label_tiers":dict(labels),
            "upstream_training_provenance":"dataset_trained","question_policy":self.spec.question_policy}

