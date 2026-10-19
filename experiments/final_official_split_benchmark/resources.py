from __future__ import annotations

import subprocess
import sys

from .io_guard import read_json, write_json


def benchmark(config, progress):
    capabilities = read_json(config.run_dir / "manifests/system_capability_matrix.json")
    focus = ("B1", "B2", "B3", "B4b", "B5-EN", "G1", "P2")
    jobs = [(cell, system) for cell in config.cells for system in focus
            if any(x["cell"] == cell and x["system"] == system and x["status"] == "RUN" for x in capabilities["rows"])]
    results = []
    for index, (cell, system) in enumerate(jobs, 1):
        progress.count(index, len(jobs), item="fresh-process resource job", extra={"cell": cell, "system": system})
        command = [sys.executable, "-m", "experiments.final_official_split_benchmark.resource_worker", "--run-dir", str(config.run_dir), "--cell", cell, "--system", system, "--questions", str(config.benchmark_questions), "--device", config.device]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            write_json(config.run_dir / "resources" / system / f"{cell}.FAILED.json", {"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
            raise RuntimeError(f"resource benchmark failed: {cell}/{system}: {completed.stderr[-1000:]}")
        results.append(read_json(config.run_dir / "resources" / system / f"{cell}.json"))
    import platform
    environment = {"python": sys.version, "platform": platform.platform(), "device_request": config.device}
    try:
        import torch
        environment.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        environment["torch"] = "unavailable"
    write_json(config.run_dir / "resources/environment.json", environment)
    training = {cell: read_json(config.run_dir / "refit" / f"{cell}_COMPLETE.json") for cell in config.cells}
    write_json(config.run_dir / "resources/summary.json", {"rows": results, "training": training, "fresh_process_per_cell_system": True, "warmup": 10, "repeats": 3})
    return results
