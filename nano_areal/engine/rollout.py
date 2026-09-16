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
import copy
import logging
import os
import subprocess
import time
from typing import TYPE_CHECKING

import torch

from nano_areal.engine.multimodal import PromptProcessor

if TYPE_CHECKING:
    from nano_areal.archon.model import Qwen35Model
    from nano_areal.config import ModelConfig, RolloutConfig
    from nano_areal.engine.local_server import LocalInferenceServer
    from nano_areal.engine.proxy import OpenAIProxyServer

logger = logging.getLogger("nano_areal.rollout")


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
        processor: PromptProcessor | None = None,
        local_model: "Qwen35Model | None" = None,
    ):
        self.model_config = model_config
        self.rollout_config = rollout_config
        self.tokenizer = tokenizer
        self.processor = processor or (PromptProcessor(tokenizer) if tokenizer is not None else None)
        # backend == "local": a *copy* of this model serves generation in-process
        self._local_model = local_model
        self._proc: subprocess.Popen | None = None
        self._proxy: "OpenAIProxyServer | None" = None
        self._local: "LocalInferenceServer | None" = None
        self.current_version: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        cfg = self.rollout_config

        if cfg.backend == "local":
            await self._start_local()
        else:
            await self._start_vllm()

        # Start the proxy server (in-process FastAPI)
        from nano_areal.engine.proxy import OpenAIProxyServer
        self._proxy = OpenAIProxyServer(
            vllm_base_url=f"http://{cfg.host}:{cfg.port}/v1",
            tokenizer=self.tokenizer,
            host=cfg.host,
            port=cfg.proxy_port,
            model_name="default",
            processor=self.processor,
        )
        await self._proxy.start()

    async def _start_local(self):
        from nano_areal.engine.local_server import (
            LocalInferenceServer,
            LocalServerConfig,
        )

        assert self._local_model is not None, "backend='local' needs local_model"
        assert self.processor is not None
        cfg = self.rollout_config
        want = self.model_config.device
        if want == "cuda" and not torch.cuda.is_available():
            want = "cpu"
        if want == "mps" and not torch.backends.mps.is_available():
            want = "cpu"
        dev = torch.device(want)
        serve_model = copy.deepcopy(self._local_model).to(dev).eval()
        serve_model.requires_grad_(False)
        self._local = LocalInferenceServer(
            serve_model,
            self.processor,
            LocalServerConfig(
                host=cfg.host,
                port=cfg.port,
                model_name="default",
                max_model_len=self.model_config.max_model_len,
                default_max_tokens=cfg.max_new_tokens,
            ),
        )
        await self._local.start()
        await self._wait_for_server(timeout=30)

    async def _start_vllm(self):
        cfg = self.rollout_config
        mcfg = self.model_config

        env = {**os.environ}
        gpu_ids_str = ",".join(str(g) for g in cfg.gpu_ids)

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", mcfg.model_name,
            "--tokenizer", mcfg.tokenizer_name,
            "--host", cfg.host,
            "--port", str(cfg.port),
            "--tensor-parallel-size", str(cfg.tensor_parallel_size),
            "--dtype", mcfg.dtype,
            "--max-model-len", str(mcfg.max_model_len),
            "--gpu-memory-utilization", "0.85",
            "--served-model-name", "default",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ]

        if mcfg.device == "mps":
            # vllm-metal on Apple Silicon — different CLI from standard vllm
            cmd = [
                "vllm-metal",
                "--model", mcfg.model_name,
                "--host", cfg.host,
                "--port", str(cfg.port),
            ]
        else:
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

        self._proc = subprocess.Popen(cmd, env=env)
        await self._wait_for_server()

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
        if self._local:
            await self._local.stop()
            self._local = None
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

    def apply_weights(self, state_dict: dict, new_version: int):
        """Copy backend: load merged trainer weights into the local server."""
        if self._local is not None:
            self._local.load_state_dict(state_dict, new_version)
        self.current_version = new_version
        logger.info(f"[local] weights updated -> v{new_version} ({len(state_dict)} tensors)")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()
