"""Locked-test entry point intentionally exposes no tuning arguments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .io_guard import refuse_overwrite, write_json

def main(argv=None):
    p=argparse.ArgumentParser(description="Apply a frozen V6 candidate once to an untouched test bundle")
    p.add_argument("--freeze-manifest",type=Path,required=True); p.add_argument("--test-manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv); freeze=json.loads(args.freeze_manifest.read_text()); test=json.loads(args.test_manifest.read_text())
    if freeze.get("status")!="FROZEN" or freeze.get("retuning_allowed") is not False: raise ValueError("invalid freeze manifest")
    if test.get("previously_inspected",True) is not False: raise ValueError("test manifest must attest previously_inspected=false")
    output=refuse_overwrite(args.output)
    write_json(output/"LOCKED_TEST_READY.json",{"freeze":freeze,"test_manifest":test,"status":"READY_FOR_REGISTERED_ADAPTER","tuneable_arguments":[]})
    return 0

if __name__=="__main__": raise SystemExit(main())

