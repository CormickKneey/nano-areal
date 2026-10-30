"""Memory sampler failures must remain explicit without hiding primary errors."""

import json
import sys
import threading
from types import SimpleNamespace

import probe_s4_training
import pytest
import torch
from probe_s4_training import MemoryTrace


def test_empty_memory_trace_does_not_query_mps(monkeypatch):
    def unavailable():
        raise RuntimeError("MPS unavailable")

    monkeypatch.setattr(torch.mps, "recommended_max_memory", unavailable)
    trace = MemoryTrace()
    trace.stop_event.set()
    trace.thread.start()
    report = trace.finish()
    assert report["samples"] == 0
    assert report["mps_allocated_peak_bytes"] is None
    assert report["mps_driver_peak_bytes"] is None
    assert report["mps_recommended_max_bytes"] is None
    assert report["sampling_valid"] is False
    assert report["sampler_error"] == {
        "type": "RuntimeError", "message": "MPS unavailable",
    }


def test_memory_trace_keeps_observed_peaks(monkeypatch):
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 1024)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 128)
    sampled = threading.Event()

    def driver_memory():
        sampled.set()
        return 256

    monkeypatch.setattr(torch.mps, "driver_allocated_memory", driver_memory)
    trace = MemoryTrace()
    trace.thread.start()
    assert sampled.wait(2)
    report = trace.finish()
    assert report["samples"] >= 1
    assert report["mps_allocated_peak_bytes"] == 128
    assert report["mps_driver_peak_bytes"] == 256
    assert report["mps_recommended_max_bytes"] == 1024
    assert report["sampling_valid"] is True
    assert report["sampler_error"] is None


def test_partial_memory_trace_is_not_qualified(monkeypatch):
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 1024)

    def unavailable():
        raise RuntimeError("Allocator failed")

    monkeypatch.setattr(torch.mps, "current_allocated_memory", unavailable)
    trace = MemoryTrace()
    trace.samples.append({"mps_allocated_bytes": 64, "mps_driver_bytes": 128})
    trace.thread.start()
    report = trace.finish()
    assert report["samples"] == 1
    assert report["mps_driver_peak_bytes"] == 128
    assert report["sampling_valid"] is False
    assert report["sampler_error"]["message"] == "Allocator failed"


def test_sampler_failure_preserves_primary_error_report(monkeypatch, tmp_path):
    def unavailable():
        raise RuntimeError("MPS unavailable")

    def load_failure(path):
        raise ValueError("Primary load failure")

    binding = SimpleNamespace(
        verify_files=lambda: None,
        manifest={"training_files": {}, "inference_files": {}},
    )
    monkeypatch.setattr(sys, "argv", [
        "probe_s4_training.py", "update", "--infer-model", "unused",
        "--train-model", "unused", "--output", str(tmp_path), "--group", "unused",
    ])
    monkeypatch.setattr(probe_s4_training, "require_stopped_server", lambda: None)
    monkeypatch.setattr(probe_s4_training, "memory_snapshot", lambda process: {})
    monkeypatch.setattr(probe_s4_training, "command_output", lambda command: "")
    monkeypatch.setattr(probe_s4_training, "PolicyBinding", SimpleNamespace(
        read=lambda path: binding,
    ))
    monkeypatch.setattr(probe_s4_training, "checkpoint_files", lambda path: {})
    monkeypatch.setattr(probe_s4_training, "complete_group", lambda *args: [])
    monkeypatch.setattr(probe_s4_training, "NativeQwen3VLLoRATrainer", load_failure)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", unavailable)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)
    with pytest.raises(ValueError, match="Primary load failure"):
        probe_s4_training.main()
    report = json.loads((tmp_path / "update.json").read_text())
    assert report["status"] == "error"
    assert report["error"] == "Primary load failure"
    assert report["updates"] == 0
    assert report["memory_trace"]["sampling_valid"] is False
    assert report["memory_trace"]["sampler_error"]["message"] == "MPS unavailable"
