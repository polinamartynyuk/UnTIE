from __future__ import annotations
import argparse
import json
from pathlib import Path
from .io_guard import sha256, write_json

def main(argv=None):
    p=argparse.ArgumentParser(description="Freeze one V6 development candidate")
    p.add_argument("--run-dir",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--variant",choices=("S2","S3","S4"),required=True); p.add_argument("--mode",choices=("local","pooled"),required=True)
    args=p.parse_args(argv)
    if not (args.run_dir/"COMPLETED.json").is_file(): raise RuntimeError("only a completed valid run can be frozen")
    manifest={"status":"FROZEN","source_run":str(args.run_dir.resolve()),"variant":args.variant,"mode":args.mode,
              "contract_sha256":sha256(args.run_dir/"frozen_contract.yaml"),"retuning_allowed":False,
              "development_only":True,"locked_test_required":True}
    write_json(args.output,manifest); return 0

if __name__=="__main__": raise SystemExit(main())

