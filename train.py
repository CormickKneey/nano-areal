"""
nano-areal training entry point.

Async mode (2×GPU):
    python train.py

Sync mode (macOS / single GPU):
    python train.py --sync

Custom config:
    python train.py --model Qwen/Qwen3.5-0.8B-Base --steps 500
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path

import torch

from nano_areal.agent.bfcl import BFCLAgent
from nano_areal.archon.model import Qwen35Args, build_model
from nano_areal.archon.state_dict import Qwen35StateDictAdapter, load_hf_weights
from nano_areal.archon.trainer import ArchonTrainer, Trajectory
from nano_areal.config import NanoArealConfig
from nano_areal.engine.proxy import OpenAIProxyWorkflow
from nano_areal.engine.types import TrajectoryTensors
from nano_areal.dataset import BFCLDataset
from nano_areal.engine.buffer import AsyncTrajectoryBuffer
from nano_areal.engine.rollout import RolloutEngine
from nano_areal.sync.awex_bridge import build_weight_sync
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
    dataset: BFCLDataset,
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

    for batch in dataset.iter_batches(batch_size=1, shuffle=True, repeat=True):
        if stop_event.is_set():
            break

        sample = batch[0]

        # Run G episodes concurrently for the same sample
        tasks = [workflow.arun_episode(sample) for _ in range(group_size)]
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
                traj = Trajectory(
                    input_ids=tt.input_ids,
                    response_mask=tt.response_mask,
                    reward=tt.reward,
                    policy_version=engine.current_version,
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
):
    """
    Consume trajectories from the buffer and run GRPO training steps.
    """
    logger.info("Training worker started")
    sync_mode = config.async_rl.max_head_offpolicyness == 0

    for step in range(config.train.total_steps):
        if stop_event.is_set():
            break

        # Collect a batch of trajectories
        if sync_mode:
            batch = await buffer.get_batch_sync(config.train.batch_size)
        else:
            batch = await buffer.get_batch(
                n=config.train.batch_size * config.rollout.group_size
            )

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
                weight_sync_writer.push(state, trainer.policy_version)

        # Logging & visualization
        terminal.update_stats(stats, buffer)
        html_dash.record(stats, trajectories=batch)

        if step % 10 == 0:
            logger.info(
                f"step={step} policy_v={trainer.policy_version} "
                f"loss={stats['loss']:.4f} reward={stats['reward_mean']:.3f} "
                f"kl={stats['kl']:.4f} buffer={buffer.qsize}"
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

async def main_async(config: NanoArealConfig, model_source: str = "modelscope"):
    # ------------------------------------------------------------------
    # Build components
    # ------------------------------------------------------------------
    logger.info(f"Mode: {config.mode}")
    logger.info(f"Model: {config.model.model_name}")

    # Dataset
    dataset = BFCLDataset(config.dataset)
    logger.info(f"Dataset: {len(dataset)} samples loaded")

    # Resolve local checkpoint path; update config so rollout engine uses local path
    model_path = _resolve_model_path(config.model.model_name, source=model_source)
    config.model.model_name = model_path
    logger.info(f"Model path: {model_path}")

    device_str = _pick_device(config.train.gpu_ids)
    dtype = torch.bfloat16 if config.model.dtype == "bfloat16" else torch.float32

    args = Qwen35Args.from_hf_config(model_path)
    model = build_model(args, torch.device(device_str), dtype)

    # Load pretrained weights via state_dict adapter (safetensors / bin)
    try:
        load_hf_weights(model, model_path, device=device_str)
        logger.info("Loaded pretrained weights")
    except Exception as e:
        logger.warning(f"Could not load pretrained weights ({e}), using random init")

    # Tokenizer: reuse already-resolved model_path (tokenizer_name defaults to model_name)
    from transformers import AutoTokenizer
    tok_path = model_path  # model_path is already the resolved local path
    config.model.tokenizer_name = tok_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Trainer
    trainer = ArchonTrainer(model, config.train, tokenizer)

    # Rollout engine (vllm + proxy server)
    # Tokenizer is held by the proxy server, not the agent
    engine = RolloutEngine(config.model, config.rollout, tokenizer=tokenizer)

    # Agent: pure OpenAI SDK, no AReaL internals
    agent = BFCLAgent(
        max_turns=config.rollout.max_turns,
        temperature=config.rollout.temperature,
        top_p=config.rollout.top_p,
        max_tokens=config.rollout.max_new_tokens,
    )

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
        logger.info("vllm + proxy servers ready")

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
                train_worker(trainer, buffer, config, weight_sync_writer, terminal, html_dash, stop_event)
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


def _resolve_model_path(model_name: str, source: str = "modelscope") -> str:
    """
    Return a local filesystem path for the model config and weights.

    Resolution order:
      1. Local directory  → return as-is
      2. modelscope       → modelscope.hub snapshot_download
      3. huggingface      → huggingface_hub snapshot_download
    """
    if Path(model_name).is_dir():
        return model_name

    if source == "modelscope":
        try:
            from modelscope.hub.snapshot_download import snapshot_download as ms_download
            path = ms_download(model_name, ignore_patterns=["*.msgpack", "*.h5"])
            logger.info(f"Downloaded from ModelScope: {path}")
            return path
        except ImportError:
            logger.warning("modelscope not installed, falling back to HuggingFace")
        except Exception as e:
            logger.warning(f"ModelScope download failed ({e}), falling back to HuggingFace")

    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id=model_name, ignore_patterns=["*.msgpack", "*.h5"])
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
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="Model ID (ModelScope / HuggingFace) or local directory")
    parser.add_argument("--model-path", default=None,
                        help="Explicit local checkpoint directory (overrides --model)")
    parser.add_argument("--model-source", default="modelscope",
                        choices=["modelscope", "huggingface"],
                        help="Where to download the model (default: modelscope)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max-offpolicy", type=int, default=4)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument(
        "--bfcl-data-dir", default=None,
        help="Path to downloaded BFCL-V4 data (overrides BFCL_DATA_DIR env)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.sync:
        config = NanoArealConfig.for_macos()
    else:
        config = NanoArealConfig.for_two_gpus()

    effective_model = args.model_path or args.model
    config.model.model_name = effective_model
    config.model.tokenizer_name = effective_model
    config.train.total_steps = args.steps
    config.rollout.group_size = args.group_size
    config.train.batch_size = args.batch_size
    config.train.lr = args.lr
    config.async_rl.max_head_offpolicyness = args.max_offpolicy
    config.log_dir = args.log_dir
    config.save_dir = args.save_dir
    if args.bfcl_data_dir:
        config.dataset.data_dir = args.bfcl_data_dir

    asyncio.run(main_async(config, model_source=args.model_source))


if __name__ == "__main__":
    main()
