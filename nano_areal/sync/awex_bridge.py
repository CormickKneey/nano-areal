"""
awex weight-sync bridge.

Wraps awex WeightWriter (training side) and WeightReader (rollout side)
into a simple push/pull interface used by train.py.

awex supports:
  - NCCL P2P (GPU-to-GPU, primary mode for 2×CUDA setup)
  - Shared memory (same node, fallback)
  - Copy (in-process, for sync/macOS mode — no awex dependency)

References:
  https://github.com/inclusionAI/asystem-awex
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from nano_areal.config import WeightSyncConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend: direct copy (sync / macOS mode)
# ---------------------------------------------------------------------------

class CopyWeightSync:
    """
    Trivial weight sync for single-process sync mode.
    Training and rollout share the same Python process, so we just
    swap the vllm model weights in-place using its load_weights API.
    """

    def __init__(self):
        self._vllm_model = None     # set by attach_vllm()
        self._step_id = 0

    def attach_vllm(self, llm):
        """llm: vllm.LLM instance (used in sync mode)."""
        self._vllm_model = llm

    def push(self, state_dict: dict, step_id: int):
        """Training side: push new weights."""
        self._pending_state = state_dict
        self._step_id = step_id
        logger.debug(f"[CopySync] buffered weights at step {step_id}")

    def pull(self, step_id: int):
        """Rollout side: apply buffered weights to vllm."""
        if self._vllm_model is None or not hasattr(self, "_pending_state"):
            return
        try:
            # vllm exposes llm_engine.model_executor.driver_worker.model_runner.model
            executor = self._vllm_model.llm_engine.model_executor
            worker = executor.driver_worker
            worker.model_runner.model.load_weights(self._pending_state.items())
            logger.info(f"[CopySync] applied weights at step {step_id}")
        except Exception as e:
            logger.warning(f"[CopySync] weight update failed: {e}")


# ---------------------------------------------------------------------------
# Backend: awex NCCL (2×GPU async mode)
# ---------------------------------------------------------------------------

class AwexWeightSync:
    """
    awex-based weight sync for multi-GPU async RL.

    Writer runs in the training process; Reader runs in the rollout process.
    Both are coordinated via awex MetaServer.

    Typical usage:
        # Training process
        writer = AwexWeightSync.make_writer(trainer.model, config)
        writer.initialize()
        writer.push(step_id=42)

        # Rollout process
        reader = AwexWeightSync.make_reader(vllm_engine, config)
        reader.initialize()
        reader.pull(step_id=42)
    """

    def __init__(self, role: Literal["writer", "reader"], handler):
        self.role = role
        self._handler = handler

    @classmethod
    def make_writer(cls, model, config: "WeightSyncConfig") -> "AwexWeightSync":
        from awex import NCCLWeightsWriter
        from awex.engine.fsdp import FSDPEngine

        awex_cfg = _build_awex_config(config, role="writer")
        engine = FSDPEngine(awex_cfg, model)
        handler = NCCLWeightsWriter(engine)
        return cls(role="writer", handler=handler)

    @classmethod
    def make_reader(cls, vllm_engine, config: "WeightSyncConfig") -> "AwexWeightSync":
        from awex import WeightsReader
        from awex.engine.vllm import VLLMEngine

        awex_cfg = _build_awex_config(config, role="reader")
        engine = VLLMEngine(awex_cfg, vllm_engine)
        handler = WeightsReader(engine)
        return cls(role="reader", handler=handler)

    def initialize(self):
        self._handler.initialize()
        logger.info(f"[AwexSync/{self.role}] initialized")

    def push(self, step_id: int):
        """Training side: broadcast weights to rollout."""
        assert self.role == "writer"
        self._handler.write_weights(step_id=step_id)
        logger.info(f"[AwexSync/writer] pushed weights at step {step_id}")

    def pull(self, step_id: int):
        """Rollout side: receive new weights from trainer."""
        assert self.role == "reader"
        self._handler.update_weights(step_id=step_id)
        logger.info(f"[AwexSync/reader] pulled weights at step {step_id}")


def _build_awex_config(config: "WeightSyncConfig", role: str):
    """Build awex InferenceConfig / TrainingConfig from nano-areal config."""
    try:
        from awex import InferenceConfig, TrainingConfig
        if role == "reader":
            return InferenceConfig(meta_server_addr=config.meta_server_addr)
        return TrainingConfig(meta_server_addr=config.meta_server_addr)
    except ImportError:
        raise ImportError(
            "awex not installed. Install with: pip install awex\n"
            "Or use mode='sync' with weight_sync.backend='copy'."
        )


# ---------------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------------

def build_weight_sync(
    backend: Literal["awex", "copy"],
    role: Literal["writer", "reader"],
    model_or_engine=None,
    config: "WeightSyncConfig | None" = None,
):
    """
    Factory function — returns the appropriate sync object based on backend.

    Args:
        backend:         'awex' (2×GPU) or 'copy' (sync/macOS)
        role:            'writer' (training process) or 'reader' (rollout process)
        model_or_engine: torch model (writer) or vllm engine (reader)
        config:          WeightSyncConfig
    """
    if backend == "copy":
        sync = CopyWeightSync()
        if role == "reader" and model_or_engine is not None:
            sync.attach_vllm(model_or_engine)
        return sync

    if backend == "awex":
        if role == "writer":
            return AwexWeightSync.make_writer(model_or_engine, config)
        return AwexWeightSync.make_reader(model_or_engine, config)

    raise ValueError(f"Unknown weight sync backend: {backend!r}")
