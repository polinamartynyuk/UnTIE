"""CUDA-aware stage timing. Peak memory is reset per measurement."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _reset_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def peak_gpu_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:
        return 0.0
    return 0.0


@contextmanager
def timed_block(*, warmup: bool = False) -> Iterator[dict[str, Any]]:
    stats: dict[str, Any] = {"elapsed_s": 0.0, "peak_gpu_mb": 0.0, "warmup": warmup}
    if warmup:
        _sync()
        yield stats
        return
    _sync()
    _reset_peak()
    started = time.perf_counter()
    try:
        yield stats
    finally:
        _sync()
        stats["elapsed_s"] = float(time.perf_counter() - started)
        stats["peak_gpu_mb"] = peak_gpu_mb()
