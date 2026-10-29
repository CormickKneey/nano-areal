"""
Rollout engine: manages the vllm inference server + OpenAI proxy server.

Architecture:
                Agent
                  │  AsyncOpenAI(base_url=proxy_url, api_key=session_key)
                  ▼
    ProxyServer  (FastAPI, port rollout_cfg.proxy_port)
                  │  tokenize + capture logprobs
                  ▼
    vllm Server  (OpenAI-compatible, port rollout_cfg.port)
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import time
from typing import TYPE_CHECKING

import httpx
from openai import AsyncOpenAI

if TYPE_CHECKING:
    from nano_areal.config import ModelConfig, RolloutConfig
    from nano_areal.engine.proxy import OpenAIProxyServer



def build_inference_command(
    *,
    backend: str,
    model: str,
    tokenizer: str,
    host: str,
    port: int,
    tensor_parallel_size: int,
    dtype: str,
    max_model_len: int,
    gpu_ids: list[int],
) -> tuple[list[str], dict[str, str]]:
    """``metal`` launches vllm-metal. ``cuda`` launches vLLM's OpenAI server."""
    if backend == "metal":
        return [
            "vllm-metal",
            "--model", model,
            "--host", host,
            "--port", str(port),
        ], {}
    if backend != "cuda":
        raise ValueError(f"Unknown inference backend {backend!r}. Expected metal or cuda.")
    return [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--tokenizer", tokenizer,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.85",
        "--served-model-name", "default",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ], {"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpu_ids)}


class RolloutEngine:
    """
    Manages the vllm subprocess and the in-process proxy server.

    Usage:
        async with RolloutEngine(model_cfg, rollout_cfg, tokenizer) as engine:
            workflow = OpenAIProxyWorkflow(agent, engine.proxy)
            ...
    """

    def __init__(
        self,
        model_config: "ModelConfig",
        rollout_config: "RolloutConfig",
        tokenizer=None,
    ):
        self.model_config = model_config
        self.rollout_config = rollout_config
        self.tokenizer = tokenizer
        self._proc: subprocess.Popen | None = None
        self._proxy: "OpenAIProxyServer | None" = None
        self.current_version: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        cfg = self.rollout_config
        mcfg = self.model_config

        infer_model = mcfg.infer_model_name or mcfg.model_name
        cmd, extra_env = build_inference_command(
            backend=cfg.inference_backend,
            model=infer_model,
            tokenizer=mcfg.tokenizer_name or mcfg.model_name,
            host=cfg.host,
            port=cfg.port,
            tensor_parallel_size=cfg.tensor_parallel_size,
            dtype=mcfg.dtype,
            max_model_len=mcfg.max_model_len,
            gpu_ids=cfg.gpu_ids,
        )
        env = {**os.environ, **extra_env}
        self._proc = subprocess.Popen(cmd, env=env)
        await self._wait_for_server()

        # Start the proxy server (in-process FastAPI)
        from nano_areal.engine.proxy import OpenAIProxyServer
        self._proxy = OpenAIProxyServer(
            vllm_base_url=f"http://{cfg.host}:{cfg.port}/v1",
            tokenizer=self.tokenizer,
            host=cfg.host,
            port=cfg.proxy_port,
            model_name="default",
        )
        await self._proxy.start()

    async def _wait_for_server(self, timeout: int = 120):
        import httpx
        url = f"http://{self.rollout_config.host}:{self.rollout_config.port}/health"
        deadline = time.time() + timeout
        async with httpx.AsyncClient() as http:
            while time.time() < deadline:
                try:
                    r = await http.get(url, timeout=2.0)
                    if r.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(2)
        raise TimeoutError(f"vllm server did not start within {timeout}s")

    async def stop(self):
        if self._proxy:
            await self._proxy.stop()
            self._proxy = None
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def proxy(self) -> "OpenAIProxyServer":
        assert self._proxy is not None, "RolloutEngine not started. Call await engine.start() first."
        return self._proxy

    # ------------------------------------------------------------------
    # Weight synchronization
    # ------------------------------------------------------------------

    async def sync_weights(self, new_version: int):
        """
        Called by the weight-sync layer after awex pushes new weights.
        Updates current_version so new trajectories are tagged correctly.
        """
        self.current_version = new_version

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()
