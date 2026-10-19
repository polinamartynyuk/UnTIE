from __future__ import annotations

import json
import subprocess
import sys

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT

from .io_guard import write_json


def run_benchmarks(config,output,progress):
    progress.stage("benchmark","fresh-process actual end-to-end inference")
    results=[]
    for cell in config.cells:
        for system in ("B0","V5-Q","V5-QG"):
            progress.log("benchmark system",cell=cell,system=system)
            command=[sys.executable,"-m","experiments.qa_chunk_selection.v5_question_conditioned_residual.benchmark_worker",
                     "--cell",cell,"--system",system,"--run-root",str(output),"--device",config.device,
                     "--questions",str(config.benchmark_questions)]
            completed=subprocess.run(command,cwd=PROJECT_ROOT,text=True,capture_output=True)
            if completed.returncode:
                detail=(completed.stderr or completed.stdout or "no worker output")[-8000:]
                raise RuntimeError(f"benchmark worker failed for {cell}/{system} (exit={completed.returncode}):\n{detail}")
            line=next((x for x in reversed(completed.stdout.splitlines()) if x.strip().startswith("{")),None)
            if line is None:raise RuntimeError(f"benchmark produced no JSON: {completed.stdout[-1000:]}")
            row=json.loads(line); results.append(row); write_json(output/f"resource_benchmark/{cell}/{system}.json",row)
    write_json(output/"resource_benchmark/summary.json",{"runs":results,"fresh_process_per_system":True})

