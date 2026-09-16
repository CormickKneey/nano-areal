from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from nano_areal.dataset import DatasetConfig


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen3.5-2B"
    # Tokenizer defaults to model_name if not set
    tokenizer_name: str = ""
    max_model_len: int = 4096
    dtype: str = "bfloat16"  # bfloat16 / float16 / float32
    device: str = "cuda"     # cuda / mps / cpu
    # Drop the vision tower (text-only tasks; saves ~0.4B params of memory)
    text_only: bool = False

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

    # Inference backend: "vllm" (subprocess, CUDA / vllm-metal) or "local"
    # (in-process nano-archon generator; CPU/MPS, no vllm needed)
    backend: Literal["vllm", "local"] = "vllm"

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

    # Policy-gradient objective (see nano_areal/archon/losses.py)
    loss_type: Literal["grpo", "dr_grpo", "cispo", "gspo"] = "grpo"
    adv_norm: Literal["grouped", "batch", "none"] = "grouped"
    kl_coef: float = 0.01
    kl_type: Literal["k1", "k3"] = "k1"
    clip_eps: float = 0.2
    clip_eps_high: float | None = None   # DAPO asymmetric upper clip; None = clip_eps
    recompute_logprobs: bool = True   # required for off-policy data

    # Decoupled PPO: reuse the training forward as π_prox (one mini-batch per
    # update) instead of a separate no-grad pass.
    reuse_train_logp: bool = True
    # Rejection sampling on staleness weight π_prox/π_behave. None disables.
    behav_imp_weight_cap: float | None = 5.0
    rejection_action: Literal["mask", "clamp"] = "mask"
    # Zero the advantage of responses cut off by max_tokens (stop_reason == "length")
    mask_truncated: bool = False

    # Model precision on the trainer device: bfloat16 / float16 / float32 / auto
    # (auto = bf16 on CUDA, fp32 elsewhere)
    dtype: str = "auto"

    # LoRA (rank 0 = full fine-tuning). Adapters are merged before weight sync.
    lora_rank: int = 0
    lora_alpha: float = 32.0
    lora_targets: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    freeze_vision: bool = True

    # Teacher distillation (AReaL mopd): loss = rl_coef * L_rl + distill_coef * L_distill
    rl_coef: float = 1.0
    distill_coef: float = 0.0
    # Only distill groups whose pass rate is <= this value (1.0 = every group)
    distill_trigger_max_pass_rate: float = 1.0
    # Teacher: "none" | "local:<path>" (nano-archon in-process) | "http://host:port/v1"
    teacher: str = "none"
    teacher_model_name: str = "default"

    # Process reward shaping weight (0 disables; see nano_areal/reward/prm.py)
    prm_weight: float = 0.0

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

    # Task: which agent / dataset / reward triple to run (see nano_areal/tasks.py)
    task: Literal["bfcl", "clevr", "geometry3k", "nano_swe", "chart", "dsh"] = "bfcl"
    task_data: str = ""                 # dataset path / HF id / builtin (task-specific)
    task_limit: int = 0                 # 0 = whole dataset

    # Cluster mode
    mode: Literal["async", "sync"] = "async"

    # Logging
    log_dir: str = "logs"
    save_dir: str = "checkpoints"
    save_every_steps: int = 100
    eval_every_steps: int = 50
    log_every_steps: int = 10

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
        cfg.train.lora_rank = 16
        cfg.model.max_model_len = 8192
        return cfg

    @classmethod
    def for_cpu_smoke(cls) -> "NanoArealConfig":
        """Everything in one process on CPU: local archon server + tiny model."""
        cfg = cls.for_macos()
        cfg.model.device = "cpu"
        cfg.model.dtype = "float32"
        cfg.rollout.backend = "local"
        cfg.rollout.group_size = 2
        cfg.rollout.max_new_tokens = 16
        cfg.train.batch_size = 1
        cfg.train.lora_rank = 0
        cfg.train.gradient_checkpointing = False
        cfg.train.behav_imp_weight_cap = None
        cfg.model.max_model_len = 512
        cfg.log_every_steps = 1
        return cfg

    @classmethod
    def for_two_gpus(cls) -> "NanoArealConfig":
        """Async mode config for 2-GPU setup."""
        return cls()
