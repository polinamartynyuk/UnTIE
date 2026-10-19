from __future__ import annotations
import argparse,traceback
from datetime import datetime
from pathlib import Path
from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from .config import Config,DEFAULT_SPECS

def parser():
 p=argparse.ArgumentParser(description="Independent cross-fitted V4.4 on SciMDIX/SciER")
 p.add_argument("--datasets",nargs="+",choices=tuple(DEFAULT_SPECS),default=("scimdix","scier_384"))
 p.add_argument("--output-dir",type=Path)
 p.add_argument("--device",default="auto"); p.add_argument("--epochs",type=int,default=5); p.add_argument("--patience",type=int,default=2)
 p.add_argument("--batch-size",type=int,default=16); p.add_argument("--learning-rate",type=float,default=2e-5); p.add_argument("--max-length",type=int,default=512)
 p.add_argument("--bootstrap",type=int,default=5000); p.add_argument("--resume",action="store_true"); p.add_argument("--smoke",action="store_true")
 return p

def main(argv=None):
 a=parser().parse_args(argv); stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
 out=a.output_dir or PROJECT_ROOT/"artifacts"/f"qa_chunk_selection_v4_4_external_{stamp}"
 cfg=Config(out,tuple(a.datasets),a.device,a.epochs,a.patience,a.batch_size,a.learning_rate,a.max_length,a.bootstrap,a.resume,a.smoke)
 try:
  from .pipeline import Pipeline
  return Pipeline(cfg).run()
 except Exception:
  traceback.print_exc(); return 1
if __name__=="__main__": raise SystemExit(main())
