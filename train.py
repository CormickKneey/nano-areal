"""
nano-areal training entry point.

Async mode (2×GPU, vLLM rollout on GPU0, trainer on GPU1):
    python train.py --task clevr --lora-rank 16

Sync mode (macOS, vllm-metal):
    python train.py --sync --task nano_swe

CPU smoke (no vllm; in-process nano-archon server, tiny random model):
    python train.py --cpu-smoke --task clevr --steps 2

Teacher distillation (Qwen3.8-27B served by vLLM elsewhere):
    python train.py --task dsh --teacher http://host:8010/v1 --distill-coef 0.5
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from pathlib import Path

import torch

from nano_areal.archon.distill import LocalArchonTeacher, RemoteTeacher, TeacherScorer
from nano_areal.archon.model import Qwen35Args, build_model
from nano_areal.archon.state_dict import load_hf_weights
from nano_areal.archon.trainer import ArchonTrainer, Trajectory
from nano_areal.config import NanoArealConfig
from nano_areal.engine.buffer import AsyncTrajectoryBuffer
from nano_areal.engine.multimodal import PromptProcessor
from nano_areal.engine.proxy import OpenAIProxyWorkflow
from nano_areal.engine.rollout import RolloutEngine
from nano_areal.reward.prm import ProcessRewardConfig, process_reward
from nano_areal.sync.awex_bridge import build_weight_sync
from nano_areal.tasks import Task, build_task
from nano_areal.viz.dashboard import HTMLDashboard
from nano_areal.viz.terminal import TrainingDashboard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("nano_areal")


# ---------------------------------------------------------------------------
# Rollout worker (producer)
# ---------------------------------------------------------------------------

async def rollout_worker(
    engine: RolloutEngine,
    workflow: OpenAIProxyWorkflow,
    dataset,
    buffer: AsyncTrajectoryBuffer,
    config: NanoArealConfig,
    stop_event: asyncio.Event,
):
    """
    Continuously sample from dataset, run agent episodes via proxy,
    and push TrajectoryTensors into the async buffer.

    Each problem gets `group_size` parallel episodes (GRPO group).
    The proxy server captures tokenization + logprobs transparently.
    """
    logger.info("Rollout worker started")
    group_size = config.rollout.group_size
    prm = ProcessRewardConfig(weight=config.train.prm_weight) if config.train.prm_weight > 0 else None

    for batch in dataset.iter_batches(batch_size=1, shuffle=True, repeat=True):
        if stop_event.is_set():
            break

        sample = batch[0]
        group_id = uuid.uuid4().hex

        # Run G episodes concurrently for the same sample (one GRPO group)
        tasks = [
            workflow.arun_episode(sample, group_id=group_id) for _ in range(group_size)
        ]
        try:
            groups: list = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            break

        for result in groups:
            if isinstance(result, Exception):
                logger.warning(f"Rollout error: {result}")
                continue

            # result is list[TrajectoryTensors] — one per LLM call in the episode
            for tt in result:
                reward = tt.reward
                if prm is not None:
                    shaping, _ = process_reward(
                        response_content=tt.response_content,
                        tool_calls=tt.tool_calls,
                        n_response_tokens=int(tt.response_mask.sum()),
                        stop_reason=tt.stop_reason,
                        reasoning_content=tt.reasoning_content,
                        cfg=prm,
                    )
                    reward += shaping
                traj = Trajectory(
                    input_ids=tt.input_ids,
                    response_mask=tt.response_mask,
                    reward=reward,
                    policy_version=engine.current_version,
                    # Multi-turn episodes yield one trajectory per LLM call;
                    # group by (prompt, turn) so turn-0 samples are compared
                    # with turn-0 samples, etc.
                    group_id=f"{group_id}:{tt.turn_id}",
                    behavior_logprobs=tt.logprobs,
                    stop_reason=tt.stop_reason,
                    pixel_values=tt.pixel_values,
                    image_grid_thw=tt.image_grid_thw,
                )
                await buffer.put(traj)

    logger.info("Rollout worker stopped")


# ---------------------------------------------------------------------------
# Training worker (consumer)
# ---------------------------------------------------------------------------

async def train_worker(
    trainer: ArchonTrainer,
    buffer: AsyncTrajectoryBuffer,
    config: NanoArealConfig,
    weight_sync_writer,
    terminal: TrainingDashboard,
    html_dash: HTMLDashboard,
    stop_event: asyncio.Event,
    engine: RolloutEngine | None = None,
):
    """
    Consume trajectories from the buffer and run GRPO training steps.
    """
    logger.info("Training worker started")
    sync_mode = config.async_rl.max_head_offpolicyness == 0

    for step in range(config.train.total_steps):
        if stop_event.is_set():
            break

        # Collect a batch of trajectories: batch_size prompts x group_size
        # episodes, so grouped advantage normalization sees whole groups.
        n = config.train.batch_size * config.rollout.group_size
        if sync_mode:
            batch = await buffer.get_batch_sync(n)
        else:
            batch = await buffer.get_batch(n=n)

        if not batch:
            await asyncio.sleep(0.1)
            continue

        # Run training step (synchronous — runs on GPU 1)
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, trainer.train_step, batch)

        buffer.update_policy_version(trainer.policy_version)

        # Weight sync: push new weights to rollout engine
        if trainer.policy_version % config.weight_sync.sync_every_steps == 0:
            if config.weight_sync.backend == "awex":
                await loop.run_in_executor(
                    None, weight_sync_writer.push, trainer.policy_version
                )
            elif config.weight_sync.backend == "copy":
                state = trainer.get_state_dict()
                if engine is not None and config.rollout.backend == "local":
                    await loop.run_in_executor(
                        None, engine.apply_weights, state, trainer.policy_version
                    )
                else:
                    weight_sync_writer.push(state, trainer.policy_version)

        # Logging & visualization
        terminal.update_stats(stats, buffer)
        html_dash.record(stats, trajectories=batch)

        if step % config.log_every_steps == 0:
            logger.info(
                f"step={step} policy_v={trainer.policy_version} "
                f"loss={stats['loss']:.4f} reward={stats['reward_mean']:.3f} "
                f"kl={stats['kl']:.4f} clip={stats['clip_frac']:.3f} "
                f"rejected={stats['rejected_frac']:.3f} "
                f"truncated={stats['truncated_frac']:.2f} "
                f"incomplete_groups={stats['n_incomplete_groups']} "
                f"distill={stats.get('distill_loss', 0.0):.4f} "
                f"images={stats.get('n_images', 0)} "
                f"buffer={buffer.qsize}"
            )

        if step % config.eval_every_steps == 0:
            html_dash.save()
            html_dash.save_trajectory_replay()

        if step % config.save_every_steps == 0 and step > 0:
            trainer.save(f"{config.save_dir}/step_{step}")
            logger.info(f"Saved checkpoint at step {step}")

    stop_event.set()
    logger.info("Training worker finished")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_teacher(
    config: NanoArealConfig, device_str: str, dtype: torch.dtype, student: torch.nn.Module
) -> TeacherScorer | None:
    spec = config.train.teacher
    if spec in ("", "none") or config.train.distill_coef <= 0:
        return None
    if spec == "self":
        import copy

        logger.info("Teacher: frozen copy of the initial policy (smoke/self-distill)")
        return LocalArchonTeacher(copy.deepcopy(student).eval())
    if spec.startswith("local:"):
        path = spec.removeprefix("local:")
        t_args = Qwen35Args.from_hf_config(path, text_only=config.model.text_only)
        t_model = build_model(t_args, torch.device(device_str), dtype)
        load_hf_weights(t_model, path, device=device_str)
        logger.info(f"Teacher (in-process): {path}")
        return LocalArchonTeacher(t_model)
    logger.info(f"Teacher (remote prompt_logprobs): {spec}")
    return RemoteTeacher(spec, model=config.train.teacher_model_name)


def _tiny_args(full: Qwen35Args) -> Qwen35Args:
    """Shrink a real config to a few-hundred-KB random model that keeps the
    real vocab / special tokens / layer-type pattern (CPU smoke tests)."""
    import dataclasses

    vision = None
    if full.vision is not None:
        vision = dataclasses.replace(
            full.vision, depth=2, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=32
        )
    return dataclasses.replace(
        full,
        dim=32,
        n_layers=4,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        intermediate_size=64,
        layer_types=[full.layer_type(i) for i in range(4)] if full.layer_types else [],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        mrope_section=[3, 3, 2] if full.mrope_section else None,
        vision=vision,
    )


async def main_async(
    config: NanoArealConfig,
    model_source: str = "modelscope",
    random_init: bool = False,
    tiny: bool = False,
):
    # ------------------------------------------------------------------
    # Build components
    # ------------------------------------------------------------------
    logger.info(f"Mode: {config.mode}  task: {config.task}  backend: {config.rollout.backend}")
    logger.info(f"Model: {config.model.model_name}")

    # Task = dataset + agent
    task: Task = build_task(config)
    dataset = task.dataset
    logger.info(f"Dataset: {len(dataset)} samples loaded ({task.name})")

    # Resolve local checkpoint path; update config so rollout engine uses local path
    model_path = _resolve_model_path(
        config.model.model_name,
        source=model_source,
        config_only=random_init,
    )
    config.model.model_name = model_path
    logger.info(f"Model path: {model_path}")

    device_str = _pick_device(config.train.gpu_ids)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(config.model.dtype, torch.float32)

    text_only = config.model.text_only and not task.multimodal
    args = Qwen35Args.from_hf_config(model_path, text_only=text_only)
    if tiny:
        args = _tiny_args(args)
    model = build_model(args, torch.device(device_str), dtype)

    # Load pretrained weights via state_dict adapter (safetensors / bin)
    if random_init:
        logger.warning("Random init requested (--random-init); weights not loaded")
    else:
        try:
            load_hf_weights(model, model_path, device=device_str)
            logger.info("Loaded pretrained weights")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights ({e}), using random init")

    # Tokenizer + image processor: reuse already-resolved model_path
    from transformers import AutoTokenizer
    tok_path = model_path  # model_path is already the resolved local path
    config.model.tokenizer_name = tok_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = (PromptProcessor.from_pretrained(tok_path, tokenizer=tokenizer)
                 if task.multimodal else PromptProcessor(tokenizer))
    if task.multimodal and not processor.supports_images:
        raise RuntimeError(f"task {task.name!r} needs an image processor; none found at {tok_path}")

    # Teacher (optional)
    teacher = _build_teacher(config, device_str, dtype, model)

    # Trainer
    trainer = ArchonTrainer(model, config.train, tokenizer, teacher=teacher)
    logger.info(f"Trainable params: {trainer.n_trainable:,} / {trainer.n_total:,} (lora_rank={config.train.lora_rank})")

    # Rollout engine (vllm | local archon server) + proxy server
    engine = RolloutEngine(
        config.model,
        config.rollout,
        tokenizer=tokenizer,
        processor=processor,
        local_model=model if config.rollout.backend == "local" else None,
    )

    # Agent: pure OpenAI SDK, no AReaL internals
    agent = task.agent

    # Async buffer
    buffer = AsyncTrajectoryBuffer(
        max_head_offpolicyness=config.async_rl.max_head_offpolicyness,
        maxsize=config.async_rl.buffer_maxsize,
    )

    # Weight sync
    weight_sync_writer = build_weight_sync(
        backend=config.weight_sync.backend,
        role="writer",
        model_or_engine=model,
        config=config.weight_sync,
    )
    weight_sync_reader = build_weight_sync(
        backend=config.weight_sync.backend,
        role="reader",
        config=config.weight_sync,
    )

    # Visualization
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    html_dash = HTMLDashboard(log_dir=config.log_dir)
    stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Start rollout server, then run workers
    # ------------------------------------------------------------------
    async with engine:
        logger.info(f"{config.rollout.backend} inference + proxy servers ready")

        # Wrap agent with proxy workflow — this is the only AReaL-aware layer
        workflow = OpenAIProxyWorkflow(
            agent=agent,
            proxy_server=engine.proxy,
            turn_discount=0.9,
        )

        if config.weight_sync.backend == "awex":
            weight_sync_writer.initialize()
            weight_sync_reader.initialize()

        with TrainingDashboard() as terminal:
            rollout_task = asyncio.create_task(
                rollout_worker(engine, workflow, dataset, buffer, config, stop_event)
            )
            train_task = asyncio.create_task(
                train_worker(
                    trainer, buffer, config, weight_sync_writer, terminal, html_dash, stop_event, engine
                )
            )

            try:
                await asyncio.gather(rollout_task, train_task)
            except (KeyboardInterrupt, asyncio.CancelledError):
                stop_event.set()
                rollout_task.cancel()
                train_task.cancel()

    # Final save
    trainer.save(f"{config.save_dir}/final")
    html_dash.save(os.path.join(config.log_dir, "dashboard_final.html"))
    html_dash.save_trajectory_replay(os.path.join(config.log_dir, "replay_final.html"))
    logger.info("Training complete.")


def _pick_device(gpu_ids: list[int]) -> str:
    """Select training device: CUDA > MPS > CPU."""
    if gpu_ids and torch.cuda.is_available():
        return f"cuda:{gpu_ids[0]}"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_CONFIG_ONLY_IGNORE = ["*.safetensors", "*.bin", "*.pt", "*.gguf", "*.msgpack", "*.h5"]


def _resolve_model_path(model_name: str, source: str = "modelscope", config_only: bool = False) -> str:
    """
    Return a local filesystem path for the model config and weights.

    Resolution order:
      1. Local directory  → return as-is
      2. modelscope       → modelscope.hub snapshot_download
      3. huggingface      → huggingface_hub snapshot_download

    `config_only` skips the weight shards (tokenizer/processor/config only).
    """
    if Path(model_name).is_dir():
        return model_name
    ignore = _CONFIG_ONLY_IGNORE if config_only else ["*.msgpack", "*.h5"]

    if source == "modelscope":
        try:
            from modelscope.hub.snapshot_download import (
                snapshot_download as ms_download,
            )
            path = ms_download(model_name, ignore_patterns=ignore)
            logger.info(f"Downloaded from ModelScope: {path}")
            return path
        except ImportError:
            logger.warning("modelscope not installed, falling back to HuggingFace")
        except Exception as e:
            logger.warning(f"ModelScope download failed ({e}), falling back to HuggingFace")

    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id=model_name, ignore_patterns=ignore)
        logger.info(f"Downloaded from HuggingFace: {path}")
        return path
    except Exception as e:
        raise RuntimeError(
            f"Cannot resolve model '{model_name}' from {source}.\n"
            f"Options:\n"
            f"  --model-path /local/path      use a local checkpoint\n"
            f"  --model-source huggingface    switch to HuggingFace\n"
            f"  pip install modelscope        install ModelScope SDK\n"
            f"Error: {e}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="nano-areal: async agentic RL for LLMs")
    parser.add_argument("--sync", action="store_true", help="Run in sync mode (macOS/single GPU)")
    parser.add_argument("--cpu-smoke", action="store_true",
                        help="Everything in-process on CPU: local archon server, sync RL, no vllm")
    parser.add_argument("--task", default="bfcl",
                        choices=["bfcl", "clevr", "geometry3k", "nano_swe", "chart", "dsh"])
    parser.add_argument("--task-data", default="",
                        help="Task dataset: HF id / path, 'synthetic' (clevr, chart), "
                             "'builtin'|'mbpp'|'humaneval' (nano_swe, dsh)")
    parser.add_argument("--task-limit", type=int, default=0, help="Use only the first N samples")
    parser.add_argument("--backend", default=None, choices=["vllm", "local"],
                        help="Inference backend (default: vllm; local = in-process nano-archon)")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B",
                        help="Model ID (ModelScope / HuggingFace) or local directory")
    parser.add_argument("--model-path", default=None,
                        help="Explicit local checkpoint directory (overrides --model)")
    parser.add_argument("--model-source", default="modelscope",
                        choices=["modelscope", "huggingface"],
                        help="Where to download the model (default: modelscope)")
    parser.add_argument("--random-init", action="store_true",
                        help="Skip weight download/load (tests, smoke runs)")
    parser.add_argument("--tiny", action="store_true",
                        help="Shrink the architecture to a tiny random model (implies --random-init)")
    parser.add_argument("--text-only", action="store_true",
                        help="Drop the vision tower (ignored for multimodal tasks)")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None, help="0 = full fine-tune")
    parser.add_argument("--teacher", default=None,
                        help="'local:<path>', 'self' (frozen initial policy), or an "
                             "OpenAI-compatible base URL that supports prompt_logprobs")
    parser.add_argument("--distill-coef", type=float, default=None)
    parser.add_argument("--distill-trigger", type=float, default=None,
                        help="Only distill groups with pass rate <= this (default 1.0)")
    parser.add_argument("--prm-weight", type=float, default=None,
                        help="Process-reward shaping weight (0 disables)")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max-offpolicy", type=int, default=None)
    parser.add_argument(
        "--loss", default="grpo", choices=["grpo", "dr_grpo", "cispo", "gspo"]
    )
    parser.add_argument(
        "--adv-norm", default="grouped", choices=["grouped", "batch", "none"]
    )
    parser.add_argument("--kl-type", default="k1", choices=["k1", "k3"])
    parser.add_argument(
        "--clip-eps-high",
        type=float,
        default=None,
        help="DAPO asymmetric upper clip (default: same as clip_eps)",
    )
    parser.add_argument(
        "--behav-imp-weight-cap",
        type=float,
        default=5.0,
        help="Rejection-sampling cap on pi_prox/pi_behave; <=0 disables",
    )
    parser.add_argument("--mask-truncated", action="store_true",
                        help="Zero advantage of responses cut off by max_tokens")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument(
        "--bfcl-data-dir", default=None,
        help="Path to downloaded BFCL-V4 data (overrides BFCL_DATA_DIR env)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.cpu_smoke:
        config = NanoArealConfig.for_cpu_smoke()
    elif args.sync:
        config = NanoArealConfig.for_macos()
    else:
        config = NanoArealConfig.for_two_gpus()

    effective_model = args.model_path or args.model
    config.model.model_name = effective_model
    config.model.tokenizer_name = effective_model
    config.model.text_only = args.text_only
    config.task = args.task
    config.task_data = args.task_data
    config.task_limit = args.task_limit
    if args.backend:
        config.rollout.backend = args.backend
    if args.max_model_len:
        config.model.max_model_len = args.max_model_len
    if args.max_new_tokens:
        config.rollout.max_new_tokens = args.max_new_tokens
    if args.lora_rank is not None:
        config.train.lora_rank = args.lora_rank
    if args.teacher:
        config.train.teacher = args.teacher
    if args.distill_coef is not None:
        config.train.distill_coef = args.distill_coef
    if args.distill_trigger is not None:
        config.train.distill_trigger_max_pass_rate = args.distill_trigger
    if args.prm_weight is not None:
        config.train.prm_weight = args.prm_weight
    if args.no_compile:
        config.train.compile = False
    config.train.total_steps = args.steps
    if args.group_size is not None:
        config.rollout.group_size = args.group_size
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    config.train.lr = args.lr
    if args.max_offpolicy is not None:
        config.async_rl.max_head_offpolicyness = args.max_offpolicy
    config.train.loss_type = args.loss
    config.train.adv_norm = args.adv_norm
    config.train.kl_type = args.kl_type
    config.train.clip_eps_high = args.clip_eps_high
    config.train.behav_imp_weight_cap = (
        args.behav_imp_weight_cap if args.behav_imp_weight_cap > 0 else None
    )
    config.train.mask_truncated = args.mask_truncated
    config.log_dir = args.log_dir
    config.save_dir = args.save_dir
    if args.bfcl_data_dir:
        config.dataset.data_dir = args.bfcl_data_dir

    asyncio.run(
        main_async(
            config,
            model_source=args.model_source,
            random_init=args.random_init or args.tiny,
            tiny=args.tiny,
        )
    )


if __name__ == "__main__":
    main()
