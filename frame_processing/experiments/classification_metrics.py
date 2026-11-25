"""Resource measurement and supervised scores for local frame classification."""

from __future__ import annotations

from collections import Counter
import ctypes
from ctypes import wintypes
from math import ceil
import os
from statistics import mean, median
import subprocess

from interface_spec.contracts import FrameType


def latency_summary(times: list[float]) -> dict:
    ordered = sorted(times)
    if not ordered:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None}
    return {"mean_ms": round(mean(ordered), 3), "median_ms": round(median(ordered), 3),
            "p95_ms": round(ordered[ceil(.95 * len(ordered)) - 1], 3)}


def classification_scores(labels: list[FrameType], predictions: list[FrameType]) -> dict:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must describe the same samples")
    pairs = Counter(zip(labels, predictions))
    matrix = {truth: {prediction: pairs[truth, prediction] for prediction in FrameType}
              for truth in FrameType}
    per_class = {}
    for category in FrameType:
        correct = pairs[category, category]
        support = sum(matrix[category].values())
        predicted = sum(row[category] for row in matrix.values())
        # Missing ground-truth classes are not included in macro F1.
        precision = None
        if predicted:
            precision = correct / predicted
        elif support:
            precision = 0.
        per_class[category] = {"support": support, "precision": precision,
                               "recall": correct / support if support else None,
                               "f1": 2 * correct / (support + predicted) if support else None}
    supported = [row["f1"] for row in per_class.values() if row["support"]]
    return {"labelled": len(labels),
            "accuracy": (sum(pairs[cat, cat] for cat in FrameType) / len(labels)) if labels else None,
            "macro_f1": mean(supported) if supported else None,
            "visual_recall": per_class[FrameType.VISUAL]["recall"],
            "classes": per_class, "confusion_matrix": matrix}


class _MemoryCounters(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "peak_working_set", "working_set", "quota_peak_paged_pool", "quota_paged_pool",
            "quota_peak_nonpaged_pool", "quota_nonpaged_pool", "pagefile_usage", "peak_pagefile_usage",
        )
    ]


def process_ram() -> dict:
    if os.name != "nt":
        return {}
    memory = _MemoryCounters()
    memory.size = ctypes.sizeof(memory)
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = wintypes.HANDLE
    measure = ctypes.windll.psapi.GetProcessMemoryInfo
    measure.argtypes = [wintypes.HANDLE, ctypes.POINTER(_MemoryCounters), wintypes.DWORD]
    measure.restype = wintypes.BOOL
    if not measure(current(), ctypes.byref(memory), memory.size):
        return {}
    return {"working_set_mib": round(memory.working_set / 1024**2, 1),
            "peak_working_set_mib": round(memory.peak_working_set / 1024**2, 1)}


def device_vram() -> dict:
    """Whole NVIDIA device 0 usage, including other processes and driver."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", "0", "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=4,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        name, used, total = (entry.strip() for entry in result.stdout.strip().rsplit(",", 2))
        return {"name": name, "used_mib": int(used), "total_mib": int(total)}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}
