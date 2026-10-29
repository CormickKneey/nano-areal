"""Run real dsh -> Messages proxy -> MLX -> bf16 LoRA GRPO, synchronously."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import torch
from probe_mac_vlm import memory_snapshot

from nano_areal.agent.dsh import DshAgent
from nano_areal.archon.qwen3vl_lora import Qwen3VLLoRATrainer
from nano_areal.archon.trainer import Trajectory
from nano_areal.engine.buffer import AsyncTrajectoryBuffer
from nano_areal.engine.proxy import OpenAIProxyServer, OpenAIProxyWorkflow


async def run(args, report):
    from transformers import AutoTokenizer

    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    adapter = output / "adapter.safetensors"
    key = secrets.token_hex(24)
    environment = {
        **os.environ,
        "NANO_MLX_ADMIN_KEY": key,
        "HF_HOME": str(root / ".s1-cache/huggingface"),
        "HF_HUB_OFFLINE": "1",
        "TMPDIR": str(root / ".s1-cache/tmp"),
    }
    server_log = (output / "mlx-server.log").open("w")
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "nano_areal.engine.mlx_server",
            "--model-path",
            str(args.infer_model),
            "--adapter-path",
            str(adapter),
            "--port",
            str(args.infer_port),
            "--max-context",
            str(args.max_context),
        ],
        cwd=root,
        env=environment,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    proxy = None
    try:
        async with httpx.AsyncClient(timeout=600, trust_env=False) as http:
            origin = f"http://127.0.0.1:{args.infer_port}"
            for attempt in range(240):
                if server.poll() is not None:
                    raise RuntimeError("MLX server exited; inspect mlx-server.log")
                try:
                    health = await http.get(origin + "/health")
                    health.raise_for_status()
                    break
                except httpx.TransportError:
                    await asyncio.sleep(1)
            else:
                raise TimeoutError("MLX server did not become ready")
            report["inference_pid"] = server.pid
            report["memory_before_rollout"] = memory_snapshot(server.pid)
            tokenizer = AutoTokenizer.from_pretrained(
                str(args.infer_model), local_files_only=True
            )
            proxy = OpenAIProxyServer(
                origin + "/v1",
                tokenizer,
                port=args.proxy_port,
                model_name="mlx-community/Qwen3-VL-4B-Instruct-4bit",
            )
            await proxy.start()
            agent = DshAgent(
                model_name=proxy.model_name,
                max_output_tokens=args.max_tokens,
                context_window_tokens=args.max_context,
                patches=(str(root / "scripts/dsh_patch.yml"),),
            )
            workflow = OpenAIProxyWorkflow(agent, proxy)
            buffer = AsyncTrajectoryBuffer(max_head_offpolicyness=0)
            trainer = None
            for step in range(args.steps):
                proxy.expected_policy_version = step
                started = time.monotonic()
                episodes = []
                group_id = f"step-{step}"
                with tempfile.TemporaryDirectory(
                    dir=root / ".s1-cache/tmp"
                ) as directory:
                    task_root = Path(directory)
                    hidden = task_root / "hidden"
                    hidden.mkdir()
                    (hidden / "test_answer.py").write_text(
                        "import os\nfrom pathlib import Path\n"
                        'response_path = Path(os.environ["TASK_RESPONSE_PATH"])\n'
                        "answer = response_path.read_text().strip()\n"
                        'assert len(answer) == 1 and answer in "56789"\n'
                    )
                    for index in range(args.group_size):
                        workspace = task_root / f"episode-{index}"
                        workspace.mkdir()
                        task = {
                            "prompt": "Pick a random integer from 0 through 9. "
                            "Output only the digit. Do not use tools.",
                            "workspace": str(workspace),
                            "hidden_test_dir": str(hidden),
                        }
                        trajectories = await workflow.arun_episode(task)
                        if len(trajectories) != 1:
                            raise ValueError(
                                "Bandit task requires one model call per episode"
                            )
                        item = trajectories[0]
                        if item.truncated:
                            raise ValueError("Minimal episode was truncated")
                        await buffer.put(
                            Trajectory(
                                input_ids=item.input_ids,
                                response_mask=item.response_mask,
                                reward=item.reward,
                                old_log_probs=item.logprobs,
                                policy_version=step,
                                group_id=group_id,
                                truncated=item.truncated,
                            )
                        )
                        episodes.append(
                            {
                                **agent.last_result,
                                "reward": item.reward,
                                "prompt_tokens": int((1 - item.response_mask).sum()),
                                "response_tokens": int(item.response_mask.sum()),
                                "behavior_logprobs": item.logprobs[
                                    item.response_mask.bool()
                                ].tolist(),
                            }
                        )
                        print(
                            json.dumps(
                                {"step": step, "episode": index, **episodes[-1]}
                            ),
                            flush=True,
                        )
                batch = await buffer.get_batch_sync(args.group_size)
                (output / f"rollout-step-{step + 1}.json").write_text(
                    json.dumps(
                        [
                            {
                                "input_ids": item.input_ids.tolist(),
                                "response_mask": item.response_mask.tolist(),
                                "behavior_logprobs": item.old_log_probs.tolist(),
                                "reward": item.reward,
                                "policy_version": item.policy_version,
                                "group_id": item.group_id,
                            }
                            for item in batch
                        ]
                    )
                    + "\n"
                )
                if len({item.reward for item in batch}) < 2:
                    raise ValueError(
                        "No reward spread; cannot claim a nonzero GRPO update"
                    )
                if trainer is None:
                    trainer = Qwen3VLLoRATrainer(args.train_model, args.lr)
                stats = trainer.train_step(batch)
                trainer.export_mlx_adapter(adapter)
                synced = await http.post(
                    origin + "/admin/adapters",
                    headers={"x-admin-key": key},
                    json={"version": step + 1},
                )
                synced.raise_for_status()
                buffer.update_policy_version(step + 1)
                health = (await http.get(origin + "/health")).json()
                result = {
                    "step": step + 1,
                    "episodes": episodes,
                    **stats,
                    "sync": synced.json(),
                    "inference": health,
                    "memory": memory_snapshot(server.pid),
                    "elapsed_seconds": time.monotonic() - started,
                }
                report["steps"].append(result)
                (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in result.items()
                            if key != "episodes"
                        }
                    ),
                    flush=True,
                )
            report["buffer"] = vars(buffer.stats)
            report["status"] = "passed"
    finally:
        if proxy:
            await proxy.stop()
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        server_log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["metal"], default="metal")
    parser.add_argument("--sync", action="store_true", required=True)
    parser.add_argument("--agent", choices=["dsh"], default="dsh")
    parser.add_argument("--loss", choices=["grpo"], default="grpo")
    parser.add_argument(
        "--infer-model",
        type=Path,
        default=Path(".s1-cache/models/Qwen3-VL-4B-Instruct-4bit"),
    )
    parser.add_argument(
        "--train-model",
        type=Path,
        default=Path(".s1-cache/models/Qwen3-VL-4B-Instruct"),
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--infer-port", type=int, default=8010)
    parser.add_argument("--proxy-port", type=int, default=8011)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("artifacts/s1/e2e"))
    args = parser.parse_args()
    if args.steps < 1 or args.group_size < 2:
        parser.error("At least one step and two group members are required")
    root = Path(__file__).resolve().parents[1]
    (root / ".s1-cache/tmp").mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(root / ".s1-cache/tmp")
    torch.manual_seed(42)
    report = {
        "status": "blocked",
        "steps": [],
        "mode": "real_dsh_mlx_hf_lora",
        "task": "digit hidden-test bandit (5..9 pass), not nano-swe",
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "mlx",
                "mlx-vlm",
                "torch",
                "transformers",
                "peft",
                "deepseek-harness-sdk",
                "deepseek-harness-runtime-bin",
            )
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    try:
        asyncio.run(run(args, report))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        import traceback

        traceback.print_exc()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "completed_steps": len(report["steps"]),
                "error": report.get("error"),
            }
        ),
        flush=True,
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
