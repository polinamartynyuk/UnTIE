from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT

from .config import Config
from .io_guard import canonical_hash, write_json
from .pipeline import Pipeline


def parser():
    p=argparse.ArgumentParser(description="V5 multilingual question-conditioned gated residual reranking")
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--cells",default="scirex,scier_384,scier_128,russerrc,scimdix")
    p.add_argument("--phase",choices=("audit","run","benchmark","analysis","all"),default="all")
    p.add_argument("--device",default="auto"); p.add_argument("--n-bootstrap",type=int,default=5000)
    p.add_argument("--benchmark-questions",type=int,default=20)
    p.add_argument("--local-files-only",action="store_true"); p.add_argument("--resume",action="store_true")
    p.add_argument("--smoke",action="store_true")
    return p


def prepare_output(output: Path, *, resume: bool) -> Path:
    output=Path(output)
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(f"non-empty output exists; pass --resume: {output}")
    output.mkdir(parents=True,exist_ok=True)
    return output


def main(argv=None):
    args=parser().parse_args(argv)
    output=args.output_dir or PROJECT_ROOT/"artifacts/qa_chunk_selection_v5"/f"run_{datetime.now():%Y%m%d_%H%M%S}"
    output=prepare_output(output,resume=args.resume)
    config=Config(output_dir=output,cells=tuple(x.strip() for x in args.cells.split(",") if x.strip()),phase=args.phase,
                  device=args.device,seeds=(42,) if args.smoke else (42,43,44),n_bootstrap=args.n_bootstrap,
                  benchmark_questions=args.benchmark_questions,local_files_only=args.local_files_only,resume=args.resume,smoke=args.smoke)
    config.validate(); resolved=config.to_dict(); resolved["fingerprint"]=canonical_hash(config.fingerprint_payload())
    path=output/"config_resolved.json"
    if path.is_file():
        old=json.loads(path.read_text());
        if old.get("fingerprint")!=resolved["fingerprint"]:raise RuntimeError("resume fingerprint mismatch")
    else:write_json(path,resolved)
    return Pipeline(config,output).run()


if __name__=="__main__":raise SystemExit(main())

