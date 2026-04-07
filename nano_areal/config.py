from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from nano_areal.dataset import DatasetConfig


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    # Tokenizer defaults to model_name if not set
    tokenizer_name: str = ""
    max_model_len: int = 4096
    dtype: str = "bfloat16"  # bfloat16 / float16 / float32
    device: str = "cuda"     # cuda / mps / cpu

    def __post_init__(self):
        if not self.tokenizer_name:
            self.tokenizer_name = self.model_name


@dataclass
class RolloutConfig:
    # GPU assignment
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    tensor_parallel_size: int = 1

    # Sampling
    group_size: int = 8        # G: trajectories sampled per problem
    max_turns: int = 5         # BFCL multi-turn max rounds
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 1024

    # vllm server
    host: str = "127.0.0.1"
    port: int = 8000
    # OpenAI proxy server (in-process FastAPI)
    proxy_port: int = 8001


@dataclass
class TrainConfig:
    # GPU assignment
    gpu_ids: list[int] = field(default_factory=lambda: [1])

    # Optimizer
    lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # GRPO
    kl_coef: float = 0.01
    clip_eps: float = 0.2
    recompute_logprobs: bool = True   # required for off-policy data

    # Batch
    batch_size: int = 4               # problems consumed per train step
    grad_accum_steps: int = 1
    total_steps: int = 1000

    # torch.compile
    compile: bool = True
    compile_backend: str = "inductor"  # inductor / aot_eager (macOS)

    # Activation checkpointing
    gradient_checkpointing: bool = True


@dataclass
class AsyncConfig:
    # Off-policyness: max version gap between rollout and current policy.
    # Set to 0 for synchronous RL (macOS sync mode).
    max_head_offpolicyness: int = 4
    buffer_maxsize: int = 512         # max trajectories in queue


@dataclass
class WeightSyncConfig:
    backend: Literal["awex", "copy"] = "awex"
    sync_every_steps: int = 1         # sync weights every N train steps
    # awex MetaServer address (set automatically in async mode)
    meta_server_addr: str = "127.0.0.1:7777"


@dataclass
class NanoArealConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    async_rl: AsyncConfig = field(default_factory=AsyncConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Cluster mode
    mode: Literal["async", "sync"] = "async"

    # Logging
    log_dir: str = "logs"
    save_dir: str = "checkpoints"
    save_every_steps: int = 100
    eval_every_steps: int = 50

    @classmethod
    def for_macos(cls) -> "NanoArealConfig":
        """Sync mode config for single Apple Silicon machine."""
        cfg = cls()
        cfg.mode = "sync"
        cfg.model.device = "mps"
        cfg.rollout.gpu_ids = [0]
        cfg.train.gpu_ids = [0]
        cfg.train.compile = False          # torch.compile unreliable on macOS
        cfg.async_rl.max_head_offpolicyness = 0
        cfg.weight_sync.backend = "copy"
        cfg.rollout.group_size = 4
        cfg.train.batch_size = 2
        return cfg

    @classmethod
    def for_two_gpus(cls) -> "NanoArealConfig":
        """Async mode config for 2-GPU setup."""
        return cls()
