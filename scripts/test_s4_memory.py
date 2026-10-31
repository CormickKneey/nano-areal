"""Memory sampler failures must remain explicit without hiding primary errors."""

import json
import sys
import threading
from types import SimpleNamespace

import probe_s4_training
import pytest
import torch
from probe_s4_training import MemoryTrace

from nano_areal.archon import s4_native


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


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_sampler_failure_preserves_primary_error_report(
    monkeypatch, tmp_path, cleanup_failure,
):
    def unavailable():
        raise RuntimeError("MPS unavailable")

    def load_failure(path=None):
        raise ValueError("Primary load failure")

    def fail_cleanup():
        raise RuntimeError("Cleanup failed")

    binding = SimpleNamespace(
        verify_files=lambda: None,
        manifest={"training_files": {}, "inference_files": {}},
    )
    monkeypatch.setattr(sys, "argv", [
        "probe_s4_training.py", "update", "--infer-model", "unused",
        "--train-model", "unused", "--output", str(tmp_path), "--group", "unused",
        "--aligned-arithmetic",
    ])
    monkeypatch.setattr(probe_s4_training, "require_stopped_server", lambda: None)
    monkeypatch.setattr(probe_s4_training, "memory_snapshot", lambda process: {
        "swap": "used = 0M",
    })
    monkeypatch.setattr(probe_s4_training, "command_output", lambda command: "")
    monkeypatch.setattr(probe_s4_training, "PolicyBinding", SimpleNamespace(
        read=lambda path: binding,
    ))
    monkeypatch.setattr(probe_s4_training, "checkpoint_files", lambda path: {})
    monkeypatch.setattr(probe_s4_training, "complete_group", lambda *args: [])
    monkeypatch.setattr(probe_s4_training, "NativeQwen3VLLoRATrainer", load_failure)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", unavailable)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)
    if cleanup_failure:
        trainer = SimpleNamespace(
            model=SimpleNamespace(eval=lambda: None), parameters={},
        )
        runtime = SimpleNamespace(contract=load_failure, close=fail_cleanup)
        monkeypatch.setattr(probe_s4_training, "NativeQwen3VLLoRATrainer",
                            lambda path: trainer)
        monkeypatch.setattr(probe_s4_training, "restore", lambda *args: None)
        monkeypatch.setattr(torch.mps, "synchronize", lambda: None)
        monkeypatch.setattr(s4_native, "load_affine_projections", lambda binding: {})
        monkeypatch.setattr(s4_native, "AlignedS4Runtime", lambda *args: runtime)
    with pytest.raises(ValueError, match="Primary load failure"):
        probe_s4_training.main()
    report = json.loads((tmp_path / "update.json").read_text())
    assert report["status"] == "error"
    assert report["error"] == "Primary load failure"
    assert report["updates"] == 0
    assert report["memory_trace"]["sampling_valid"] is False
    assert report["memory_trace"]["sampler_error"]["message"] == "MPS unavailable"
    assert report["memory_gate"]["passed"] is False
    if cleanup_failure:
        assert report["cleanup_error"]["message"] == "Cleanup failed"
