from __future__ import annotations
import gc,hashlib,json,threading,time
from collections import Counter,defaultdict
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.phase1_scirex.match import classify_match
from experiments.qa_chunk_selection.io_util import append_jsonl,write_json,write_jsonl
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v2.hn_flags import flag_index,hn_quality_rows
from experiments.qa_chunk_selection.v3.data import records_from_jsonl,synthetic_records
from experiments.qa_chunk_selection.v3.split import build_cv_splits,records_for_role
from experiments.qa_chunk_selection.v3.train import score_fn_for,train_b0
from experiments.qa_chunk_selection.v2.schemas import rank_ids,to_runtime
from experiments.qa_chunk_selection.v2.evaluate_v2 import build_raw_prediction,enrich_prediction
from experiments.qa_chunk_selection.v4.answers import pair_token_f1
from experiments.qa_chunk_selection.v4.evaluate import chunk_prediction
from experiments.qa_chunk_selection.v4.shortlist import scores_from_row,top_k_ids
from experiments.qa_chunk_selection.v4.v6_reader import agreement_features
from experiments.qa_chunk_selection.v4_1.evaluate import paired_delta,role_of_document,row_key,summarize_rows,transitions
from experiments.qa_chunk_selection.v4_1.ranking import v6_fused_score
from experiments.qa_chunk_selection.v4_3.encoder import SemanticEncoder
from experiments.qa_chunk_selection.v4_3.features import semantic_pack
from experiments.qa_chunk_selection.v4_3.ranking import isolate_with_b0_tiebreak
from experiments.qa_chunk_selection.v4_3.text import is_valid_answer
from experiments.qa_chunk_selection.v4_4_ru.metrics_ru import apply_unicode_rouge
from experiments.qa_chunk_selection.v4_4_ru.routing import RouteFeature,available,calibrate,choose_prediction
from experiments.qa_chunk_selection.v4_4_ru_ablation.tuning import select_stable
from untie.config import PipelineConfig
from untie.models import ModelFactory,profile_for_language
from .config import Config,DatasetSpec,SEEDS,WEIGHT_GRID
from .io_guard import claim_output


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()

def choose_device(requested):
 if requested!='auto': return requested
 try:
  import torch
  if torch.cuda.is_available():
   rows=[(torch.cuda.mem_get_info(i)[0],i) for i in range(torch.cuda.device_count())]
   return f"cuda:{max(rows)[1]}"
 except ImportError: pass
 return 'cpu'

def model_functions(spec,device,smoke):
 if smoke:
  def answer(question,context):
   word=(context.split() or [''])[0]; return {'answer':word,'score':1.0,'start':0,'end':len(word),'start_logit':1.0,'end_logit':1.0}
  def encode(texts):
   out=[]
   for text in texts:
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(str(text).encode()).digest()[:8],'little')); out.append(rng.normal(size=16))
   return np.asarray(out,dtype=float) if out else np.zeros((0,16))
  return None,answer,encode
 factory=ModelFactory(PipelineConfig(profile=profile_for_language(spec.language),device=device))
 def answer(q,c): return dict(factory.answerer(question=q,context=c))
 def encode(texts):
  if not texts:return np.zeros((0,1))
  x=np.asarray(factory.sentence_encoder.encode(list(texts),show_progress_bar=False),dtype=float); return x.reshape(1,-1) if x.ndim==1 else x
 return factory,answer,encode

def qa_payload(row,cid):
 return {'qa_answer':str(row.get('qa_predicted_answer') or ''),'qa_chunk_id':int(cid),
  'qa_chunk_start_char':int(row.get('answer_start') or 0),'qa_chunk_end_char':int(row.get('answer_end') or 0),
  'qa_start_logit':row.get('start_logit'),'qa_end_logit':row.get('end_logit'),
  'qa_span_score':float(row.get('start_logit') or 0)+float(row.get('end_logit') or 0),'qa_forward_passes':0}

def score_b0(records,scorer,run,seed,flags,cache,progress):
 score_fn=score_fn_for(scorer); indexed=flag_index(flags); out=[]
 for i,gold in enumerate(records,1):
  runtime=to_runtime(gold); scores=score_fn(runtime); ranked=rank_ids(runtime.chunks,scores); cid=int(ranked[0])
  row=cache[(gold.document_id,gold.aspect,cid,0)]; raw=build_raw_prediction(runtime,ranking_method='b0',scores=scores,ranked_ids=ranked,qa=qa_payload(row,cid),extra={'cv_run':run,'seed':seed,'qa_source':'q5_q0'})
  out.append(enrich_prediction(raw,gold,flags=indexed,skip_bertscore=True,ks=(1,5,10))); progress.count(i,len(records),item=f'B0 eval run={run} seed={seed}')
 return out

class DatasetRun:
 def __init__(self,parent,spec):
  self.parent=parent; self.config=parent.config; self.spec=spec; self.root=parent.output/spec.key; self.root.mkdir(parents=True,exist_ok=True); self.progress=parent.progress
 def load(self):
  self.progress.stage(f'{self.spec.key}:load','read-only prepared dataset')
  if self.config.smoke:
   self.records=synthetic_records(20); self.splits=build_cv_splits(self.records,n_folds=5,seed=42); self.questions={a:[self.records[0].question+f' {i}' for i in range(5)] for a in {r.aspect for r in self.records}}
  else:
   self.records=records_from_jsonl(self.spec.base_artifact/'data/dataset.jsonl'); self.splits=json.loads((self.spec.base_artifact/'data/v3_cv_splits.json').read_text()); self.questions=json.loads((self.spec.base_artifact/'questions_q5.json').read_text())
  if any(not r.positives for r in self.records): raise ValueError(f'{self.spec.key}: non-answerable record')
  if set(self.splits['assignment'])!={r.document_id for r in self.records}: raise ValueError(f'{self.spec.key}: CV coverage mismatch')
  if any(len(self.questions.get(r.aspect,()))!=5 for r in self.records): raise ValueError(f'{self.spec.key}: Q5 coverage mismatch')
  self.items={r.question_id:r for r in self.records}; self.flags_rows=hn_quality_rows(self.records); self.flags=flag_index(self.flags_rows)
  manifest={'base':str(self.spec.base_artifact.resolve()),'read_only':True,'language':self.spec.language,'b0_model':self.spec.b0_model,'semantic_model':self.spec.semantic_model,'k':self.spec.k}
  if not self.config.smoke:
   manifest['sha256']={name:sha(self.spec.base_artifact/name) for name in ('data/dataset.jsonl','data/v3_cv_splits.json','questions_q5.json')}
  manifest_path=self.root/'input_manifest.json'
  if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest: raise RuntimeError(f'{self.spec.key}: input fingerprint mismatch')
  write_json(manifest_path,manifest)
  write_json(self.root/'contract.json',{'experiment':'V4.4 external datasets','dataset':self.spec.key,'language':self.spec.language,'seeds':list(SEEDS),'k':self.spec.k,'cv':'five document-grouped outer folds','question_selection':'TRAIN+CAL only per outer run','coefficient_selection':'V6, SEM and LEX grids on TRAIN+CAL only per outer run','routing_threshold':'Q70 from TRAIN+CAL only per outer run','eval':'used only after all fold choices are frozen','locked_official_splits_opened':False,'parameter_transfer_from_other_datasets':False})
  self.progress.log('dataset loaded',dataset=self.spec.key,instances=len(self.records),documents=len(set(r.document_id for r in self.records)))
 def q5(self):
  self.progress.stage(f'{self.spec.key}:q5','same-language extractive QA; five questions')
  path=self.root/'q5/prediction_cache.jsonl'; existing={}
  if path.exists():
   for x in path.read_text().splitlines():
    if x.strip():
     r=json.loads(x); existing[(r['document_id'],r['aspect'],int(r['chunk_id']),int(r['question_id']))]=r
  device=choose_device(self.config.device); self.factory,answer,_=model_functions(self.spec,device,self.config.smoke); rows=[]; total=sum(len(r.chunks)*5 for r in self.records); done=0
  for item in self.records:
   for chunk in item.chunks:
    for qid,q in enumerate(self.questions[item.aspect]):
     key=(item.document_id,item.aspect,chunk.chunk_id,qid); row=existing.get(key)
     if row is None:
      raw=answer(q,chunk.text); match=classify_match(str(raw.get('answer') or ''),item.gold_answers,method='gold_span_containment',max_extra_tokens=5)
      row={'document_id':item.document_id,'qa_instance_id':item.question_id,'aspect':item.aspect,'chunk_id':chunk.chunk_id,'question_id':qid,'question':q,'qa_predicted_answer':str(raw.get('answer') or ''),'qa_confidence':float(raw.get('score') or 0),'answer_start':raw.get('start'),'answer_end':raw.get('end'),'start_logit':raw.get('start_logit'),'end_logit':raw.get('end_logit'),'qa_correct':bool(match.matched),'gold_supporting':chunk.gold_supporting}
      append_jsonl(path,row)
     rows.append(row); done+=1; self.progress.count(done,total,item=f'{self.spec.key} Q5 predictions')
  # Rewrite atomically from unique complete rows; an incomplete run remains safely resumable at dataset stage granularity.
  write_jsonl(path,rows); self.qcache={(r['document_id'],r['aspect'],int(r['chunk_id']),int(r['question_id'])):r for r in rows}
  pd.DataFrame(rows).to_parquet(self.root/'q5/question_predictions.parquet',index=False)
  self.qa_model='smoke' if self.factory is None else self.factory.config.profile.qa_model
  write_json(self.root/'q5/model.json',{'language':self.spec.language,'qa_model':self.qa_model,'rows':len(rows)})
  self.factory=None; gc.collect()
  try:
   import torch
   if torch.cuda.is_available(): torch.cuda.empty_cache()
  except ImportError: pass
 def b0(self):
  self.progress.stage(f'{self.spec.key}:b0','five-fold three-seed OOF; fold-local training/calibration')
  out=[]; device=choose_device(self.config.device)
  for run in range(5):
   train=records_for_role(self.records,self.splits,run=run,role='train'); cal=records_for_role(self.records,self.splits,run=run,role='calibration'); ev=records_for_role(self.records,self.splits,run=run,role='eval')
   for seed in SEEDS:
    d=self.root/f'b0/run_{run}/seed_{seed}'; p=d/'predictions_scored.jsonl'
    rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    expected={(x.question_id,seed) for x in ev}
    if {(str(x['question_id']),int(x['seed'])) for x in rows}!=expected:
     d.mkdir(parents=True,exist_ok=True); self.progress.log('B0 training',dataset=self.spec.key,run=run,seed=seed,train=len(train),calibration=len(cal),eval=len(ev))
     trained=train_b0(train,cal,self.flags_rows,seed=seed,device=device,model_name=self.spec.b0_model,output_dir=d/'model',epochs=self.config.epochs,patience=self.config.patience,batch_size=self.config.batch_size,learning_rate=self.config.learning_rate,max_length=self.config.max_length,ratio=(1,3,1),smoke=self.config.smoke,progress=self.progress)
     rows=score_b0(ev,trained['scorer'],run,seed,self.flags_rows,self.qcache,self.progress); write_jsonl(p,rows); del trained; gc.collect()
    out.extend(rows); self.progress.log('B0 fold complete',dataset=self.spec.key,run=run,seed=seed,rows=len(rows))
  expected={(qid,s) for qid in self.items for s in SEEDS}
  if {row_key(x) for x in out}!=expected or len(out)!=len(expected): raise ValueError('B0 OOF coverage mismatch')
  self.b0_rows=out; self.b0={row_key(x):x for x in out}; write_jsonl(self.root/'b0/oof_predictions_scored.jsonl',out); write_json(self.root/'b0/metrics.json',summarize_rows(out))
 def select_questions(self,run):
  eval_docs={d for d in self.splits['assignment'] if role_of_document(d,self.splits,run)=='eval'}; table=pd.DataFrame(self.qcache.values()); source=table[(~table.document_id.isin(eval_docs)) & table.gold_supporting]
  agg=source.groupby(['aspect','question_id']).qa_correct.agg(['mean','sum','count']).reset_index(); selected={}
  for aspect in sorted({r.aspect for r in self.records}):
   part=agg[agg.aspect==aspect].sort_values(['mean','sum','question_id'],ascending=[False,False,True]); selected[aspect]=int(part.iloc[0].question_id)
  return selected,agg
 def prediction(self,item,row,ranked,scores,qa,name,run,seed):
  ordered=[int(x) for x in ranked]; ordinal={cid:float(len(ordered)-i) for i,cid in enumerate(ordered)}
  pred=chunk_prediction(item,ranking_method=name,scores=ordinal,qa=qa,flags=self.flags); pred['selected_chunk_id']=ordered[0]; pred['ranked_ids']=ordered; pred['fused_scores']={str(int(k)):float(v) for k,v in scores.items()}; pred['selected_chunk_primary_class']=item.chunk_by_id(int(ranked[0])).primary_class; pred.update(seed=seed,cv_run=run,aspect=item.aspect); return apply_unicode_rouge(pred,item.gold_answers)
 def v44(self):
  self.progress.stage(f'{self.spec.key}:v4.4','fold-local question, coefficient and routing selection')
  encoder=SemanticEncoder(cache_root=self.root/'semantic_cache',device=self.config.device,smoke=self.config.smoke,model_name=self.spec.semantic_model,normalization_version='unicode-nfc-whitespace-v1',progress=self.progress)
  systems=defaultdict(list); tuning={}; grid3=[dict(zip(('alpha','beta','gamma'),x)) for x in product(WEIGHT_GRID,repeat=3)]; grid2=[dict(lambda_m=a,lambda_x=b) for a,b in product(WEIGHT_GRID,repeat=2)]
  for run in range(5):
   selected,qreport=self.select_questions(run); write_json(self.root/f'tuning/run_{run}/question_selection.json',{'selected':selected,'eval_documents_used':0,'scores':qreport.to_dict('records')}); keys=sorted(self.b0); feats={}; packs={}; qas={}; utility={}
   texts=[]
   for key in keys:
    item,row=self.items[key[0]],self.b0[key]; qid=selected[item.aspect]; qa={cid:qa_payload(self.qcache[(item.document_id,item.aspect,cid,qid)],cid) for cid in top_k_ids(row,self.spec.k)}; qas[key]=qa; texts.extend(x['qa_answer'] for x in qa.values() if is_valid_answer(x['qa_answer']))
   encoder.embed_many(texts,store='answer')
   for i,key in enumerate(keys,1):
    item,row,qa=self.items[key[0]],self.b0[key],qas[key]; top=top_k_ids(row,self.spec.k); feats[key]=agreement_features(item,row,qa,k=self.spec.k,min_overlap=.6); answers={cid:qa[cid]['qa_answer'] for cid in top}; emb={cid:encoder.embed_one(t,store='answer') for cid,t in answers.items() if is_valid_answer(t)}; packs[key]=semantic_pack(answers,emb,top_ids=top); utility[key]={cid:{'f1':max((pair_token_f1(answers[cid],g) for g in item.gold_answers),default=0)/100,'positive':float(item.chunk_by_id(cid).label=='positive'),'hn':float(item.chunk_by_id(cid).label=='hard_negative')} for cid in top}; self.progress.count(i,len(keys),item=f'{self.spec.key} features run={run}')
   fold_of={k:int(self.splits['assignment'][self.items[k[0]].document_id]) for k in keys}; source=[k for k in keys if role_of_document(self.items[k[0]].document_id,self.splits,run)!='eval']; baseline=lambda k:int(self.b0[k]['ranked_ids'][0])
   def rank(key,kind,p):
    item,row=self.items[key[0]],self.b0[key]; top=top_k_ids(row,self.spec.k)
    if kind=='v6': scores={cid:v6_fused_score(feats[key][cid],**p) for cid in top}
    elif kind=='sem': scores={cid:p['lambda_m']*feats[key][cid]['majority']+p['lambda_x']*packs[key]['sem_rel'][str(cid)] for cid in top}
    else:scores={cid:p['lambda_m']*feats[key][cid]['majority']+p['lambda_x']*feats[key][cid]['mean_token_f1']/100 for cid in top}
    return isolate_with_b0_tiebreak(item,row,scores,k=self.spec.k)
   choices={}
   for kind,candidates in (('v6',grid3),('sem',grid2),('lex',grid2)):
    result=select_stable(source,candidates,lambda key,p,k=kind:rank(key,k,p)[0][0],utility,fold_of,baseline_top=baseline,epsilon_hn=.02,hit_tolerance=.005,require_stable=True,progress=self.progress,label=f'{self.spec.key} tune {kind} run={run}'); choices[kind]=result['selected']['params'] if result['selected'] else ({'alpha':1.,'beta':0.,'gamma':0.} if kind=='v6' else {'lambda_m':0.,'lambda_x':0.}); write_json(self.root/f'tuning/run_{run}/{kind}.json',result)
   tuning[str(run)]={'question':selected,'weights':choices}
   routes={}
   for reader in ('sem','lex'):
    route=[]
    for key in keys:
     vr=rank(key,'v6',choices['v6'])[0]; rr=rank(key,reader,choices[reader])[0]; item=self.items[key[0]]; route.append(RouteFeature(item.question_id,item.document_id,key[1],role_of_document(item.document_id,self.splits,run),packs[key].get('consensus_conf'),available(packs[key],rr[0]),rr[0],vr[0]))
    routes[reader]=( {(x.question_id,x.seed):x for x in route}, calibrate(route) ); write_json(self.root/f'tuning/run_{run}/{reader}_routing.json',routes[reader][1])
   for key in keys:
    item,row=self.items[key[0]],self.b0[key]
    if role_of_document(item.document_id,self.splits,run)!='eval':continue
    seed=key[1]; base_rank=[int(x) for x in row['ranked_ids']]; base_scores=scores_from_row(row); q0=qa_payload(self.qcache[(item.document_id,item.aspect,base_rank[0],0)],base_rank[0]); bpred=self.prediction(item,row,base_rank,base_scores,q0,'B0',run,seed); systems['B0'].append(bpred)
    vr,vs,_=rank(key,'v6',choices['v6']); vpred=self.prediction(item,row,vr,vs,qas[key][vr[0]],'V6',run,seed); systems['V6'].append(vpred)
    for reader,name in (('sem','SEM'),('lex','LEX')):
     rr,rs,_=rank(key,reader,choices[reader]); rpred=self.prediction(item,row,rr,rs,qas[key][rr[0]],name,run,seed); systems[name].append(rpred); routed,used=choose_prediction(vpred,rpred,routes[reader][0][key],routes[reader][1]); routed=dict(routed); routed.update(ranking_method='A'+name,reader_used=used); systems['A'+name].append(routed)
  write_json(self.root/'tuning/selected_by_run.json',tuning); summaries={}
  for name,rows in systems.items():
   write_jsonl(self.root/f'results/{name.lower()}/predictions_scored.jsonl',rows); summaries[name]=summarize_rows(rows); write_json(self.root/f'results/{name.lower()}/metrics.json',summaries[name])
  comparisons={name:paired_delta(systems['B0'],rows,n_boot=50 if self.config.smoke else self.config.n_bootstrap,seed=42) for name,rows in systems.items() if name!='B0'}
  write_json(self.root/'results/comparisons_vs_b0.json',comparisons); changes={name:transitions(systems['B0'],rows) for name,rows in systems.items() if name!='B0'}; write_json(self.root/'results/transitions.json',changes)
  lines=[f'# V4.4 results: {self.spec.key}','','All question choices, coefficients, and routing thresholds were selected independently on TRAIN+CAL in each outer fold; EVAL leakage: 0.','', '| System | Token F1 | Hit@1 | HN@1 | ΔF1 vs B0 |','|---|---:|---:|---:|---:|']
  for name in systems:
   metric=summaries[name]; delta=comparisons.get(name,{}).get('mean_delta',0.0); lines.append(f"| {name} | {float(metric.get('token_f1',0)):.4f} | {float(metric.get('hit@1',0)):.4f} | {float(metric.get('hn@1',0)):.4f} | {float(delta or 0):+.4f} |")
  (self.root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
  write_json(self.root/'COMPLETED.json',{'status':'complete','dataset':self.spec.key,'language':self.spec.language,'instances':len(self.records),'documents':len(set(r.document_id for r in self.records)),'models':{'qa':self.qa_model,'b0':self.spec.b0_model,'semantic':self.spec.semantic_model},'cv_leakage':0,'weights_and_thresholds':'selected separately on TRAIN+CAL in every outer fold','summaries':summaries})
 def run(self): self.load(); self.q5(); self.b0(); self.v44()

class Pipeline:
 def __init__(self,config:Config): config.validate(); self.config=config; self.output=claim_output(config.output_dir,config.resume); self.progress=ProgressLogger(self.output/'progress.jsonl')
 def run(self):
  resolved=self.config.to_dict(); config_path=self.output/'config.json'
  if config_path.exists() and json.loads(config_path.read_text())!=resolved: raise RuntimeError('resume fingerprint mismatch')
  if not config_path.exists(): write_json(config_path,resolved)
  stop=threading.Event(); thread=threading.Thread(target=lambda:[self.progress.log('still running') for _ in iter(lambda:stop.wait(30),True)],daemon=True); thread.start()
  try:
   for spec in self.config.specs():
    if (self.output/spec.key/'COMPLETED.json').exists(): self.progress.log('completed dataset preserved on resume',dataset=spec.key); continue
    DatasetRun(self,spec).run()
   write_json(self.output/'COMPLETED.json',{'status':'complete','datasets':list(self.config.datasets)}); self.progress.stage('done','all requested datasets complete'); return 0
  except Exception as e:
   write_json(self.output/'FAILED.json',{'type':type(e).__name__,'error':str(e)}); self.progress.warn(f'FAILED {type(e).__name__}: {e}'); raise
  finally: stop.set(); thread.join(1)
