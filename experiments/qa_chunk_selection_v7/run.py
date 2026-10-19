from __future__ import annotations
import argparse,datetime
from pathlib import Path
from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from .config import Config,load_yaml
from .contract import CELLS,OUTER_FOLDS,SEEDS,V5_RUN,V6_RUN
from .io_guard import refuse_overwrite
from .pipeline import Pipeline

ALIASES={"scier_384_50":"scier_384","scier_128_24":"scier_128"}
def parser():
    p=argparse.ArgumentParser(description="V7/RAPR development experiment")
    p.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs/qa_chunk_selection/v7_rapr/main.yaml")
    p.add_argument("--run-dir",type=Path);p.add_argument("--cells","--datasets",nargs="+",default=list(CELLS));p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS))
    p.add_argument("--outer-folds",nargs="+",type=int);p.add_argument("--inner-folds",type=int,default=4);p.add_argument("--modes",nargs="+",default=["local","pooled","loco"])
    p.add_argument("--phase",choices=["audit","replay","run","analysis","benchmark","all"],default="all");p.add_argument("--bootstrap",type=int,default=5000);p.add_argument("--device",default="auto")
    p.add_argument("--benchmark-questions",type=int,default=100);p.add_argument("--resume",action="store_true");p.add_argument("--smoke",action="store_true");return p
def main(argv=None):
    a=parser().parse_args(argv);payload=load_yaml(a.config) if a.config.is_file() else {};stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output=a.run_dir or PROJECT_ROOT/f"artifacts/qa_chunk_selection_v7_rapr/run_{stamp}";cells=tuple(ALIASES.get(x,x) for x in a.cells)
    folds=tuple(a.outer_folds if a.outer_folds is not None else ([0] if a.smoke else OUTER_FOLDS));seeds=tuple(a.seeds)
    if a.smoke:
        cells=cells[:1];seeds=seeds[:1];folds=folds[:1]
    config=Config(run_dir=Path(output),v5_run=PROJECT_ROOT/payload.get("historical_v5","artifacts/qa_chunk_selection_v5/run_20260916_135455"),
      v6_run=PROJECT_ROOT/payload.get("historical_v6","artifacts/qa_chunk_selection_v6_universal/run_20260922_132020"),datasets=cells,seeds=seeds,outer_folds=folds,
      inner_folds=a.inner_folds,modes=tuple(a.modes),phase=a.phase,bootstrap=a.bootstrap,device=a.device,benchmark_questions=a.benchmark_questions,resume=a.resume,smoke=a.smoke)
    refuse_overwrite(config.run_dir,resume=config.resume);return Pipeline(config).run()
if __name__=="__main__":raise SystemExit(main())
