from __future__ import annotations
import json, platform, subprocess, sys
from pathlib import Path
from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from .io_guard import write_json


def run_benchmarks(config,output,progress):
    progress.stage('benchmark','fresh-process end-to-end timing, warm-up excluded, three repeats')
    selected_path=output/'analysis/selected_development_candidate.json'
    selected=json.loads(selected_path.read_text()).get('selected') if selected_path.is_file() else None
    mode=(selected or {}).get('mode','pooled'); selected_variant=(selected or {}).get('variant','S4')
    if mode not in {'pooled','local'}: mode='pooled'
    systems=list(dict.fromkeys(['S0','S2','S3',selected_variant]))
    rows=[]
    for cell in config.datasets:
        for system in systems:
            progress.log('resource worker',cell=cell,system=system,mode=mode)
            command=[sys.executable,'-m','experiments.qa_chunk_selection_v6.benchmark_worker','--cell',cell,'--system',system,
                     '--mode',mode,'--run-root',str(output),'--v5-run',str(config.v5_run),'--device',config.device,
                     '--questions',str(config.benchmark_questions),'--warmup','10','--repeats','3']
            done=subprocess.run(command,cwd=PROJECT_ROOT,text=True,capture_output=True)
            if done.returncode:
                raise RuntimeError(f"resource worker failed {cell}/{system}: {(done.stderr or done.stdout)[-8000:]}")
            line=next((x for x in reversed(done.stdout.splitlines()) if x.strip().startswith('{')),None)
            if line is None: raise RuntimeError(f'no benchmark JSON for {cell}/{system}')
            row=json.loads(line); rows.append(row); write_json(output/f'resource_benchmark/raw/{cell}_{system}.json',row)
    historical_bge=PROJECT_ROOT/'artifacts/qa_chunk_selection_v4_5_ru/run_20260915_155234/resource_benchmark/summary.json'
    write_json(output/'resource_benchmark/summary.json',{'runs':rows,'sample_contract':'all fold-0 questions, capped at 100',
               'systems':systems,'selected_mode':mode,'historical_bge_reference':json.loads(historical_bge.read_text()) if historical_bge.is_file() else None,
               'bge_note':'historical reference retained because fold-trained BGE checkpoints were not retained for an exact fresh-process replay'})
    try:
        import torch
        torch_version=torch.__version__; cuda=torch.version.cuda
    except ImportError: torch_version=None; cuda=None
    write_json(output/'resource_benchmark/environment.json',{'python':sys.version,'platform':platform.platform(),
               'torch':torch_version,'cuda':cuda,'executable':sys.executable})
