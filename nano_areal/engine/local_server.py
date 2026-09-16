"""
Local OpenAI-compatible inference server backed by nano-archon.

Drop-in replacement for the vLLM subprocess on machines without a supported
vLLM build (macOS smoke tests, CPU CI). Speed is not the point: it makes the
whole rollout -> proxy -> trainer loop runnable end-to-end with one codebase.

Endpoints
  GET  /health, /v1/models
  POST /v1/chat/completions   text + image_url messages, logprobs with
                              `return_tokens_as_token_ids`, Hermes-style
                              <tool_call> parsing, optional SSE stream
  POST /v1/completions        `prompt` as token-id list + `prompt_logprobs`
                              (the teacher-scoring primitive used by
                              nano_areal.archon.distill.RemoteTeacher)
  POST /rl/load_weights       {"path": ".../model.pt"} hot-swap weights
                              (weight sync "copy" backend)
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nano_areal.archon.generate import GenerationResult, Generator, SamplingParams
from nano_areal.archon.model import Qwen35Model
from nano_areal.engine.multimodal import PromptProcessor

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_hermes_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Split Hermes/Qwen `<tool_call>{json}</tool_call>` blocks out of `text`."""
    calls: list[dict] = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        args = obj.get("arguments", {})
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": obj.get("name", ""),
                "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False),
            },
        })
    content = _TOOL_CALL_RE.sub("", text).strip() if calls else text
    return content, calls


def split_reasoning(text: str) -> tuple[str, str]:
    """Separate a leading `<think>...</think>` block (Qwen thinking mode)."""
    m = re.match(r"\s*<think>(.*?)</think>\s*", text, re.DOTALL)
    if not m:
        return "", text
    return m.group(1).strip(), text[m.end():]


@dataclass
class LocalServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    model_name: str = "default"
    max_model_len: int = 4096
    default_max_tokens: int = 512
    stop_token_ids: tuple[int, ...] = field(default_factory=tuple)


class LocalInferenceServer:
    """Serves a single nano-archon model; requests are serialised on one worker thread."""

    def __init__(self, model: Qwen35Model, processor: PromptProcessor, cfg: LocalServerConfig):
        self.model = model.eval()
        self.processor = processor
        self.cfg = cfg
        self.generator = Generator(model)
        self.weight_version = 0
        self._lock = threading.Lock()
        self._server_task: asyncio.Task | None = None
        self.app = FastAPI(title="nano-areal local inference")
        self._setup_routes()

        tok = processor.tokenizer
        stops = set(cfg.stop_token_ids)
        for name in ("eos_token_id", "pad_token_id"):
            v = getattr(tok, name, None)
            if isinstance(v, int):
                stops.add(v)
        for special in ("<|im_end|>", "<|endoftext|>"):
            try:
                tid = tok.convert_tokens_to_ids(special)
                if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
                    stops.add(tid)
            except Exception:
                pass
        self.stop_token_ids = tuple(sorted(stops))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://{self.cfg.host}:{self.cfg.port}"

    async def start(self):
        import uvicorn
        self._server = uvicorn.Server(
            uvicorn.Config(self.app, host=self.cfg.host, port=self.cfg.port, log_level="warning")
        )
        self._server_task = asyncio.create_task(self._server.serve())
        await asyncio.sleep(0.5)

    async def stop(self):
        if self._server_task:
            self._server.should_exit = True
            try:
                await asyncio.wait_for(self._server_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._server_task.cancel()

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    def load_state_dict(self, state: dict[str, torch.Tensor], version: int):
        with self._lock:
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype
            self.model.load_state_dict(
                {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype) for k, v in state.items()},
                strict=False,
            )
            self.weight_version = version

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_sync(self, prompt_ids: list[int], sp: SamplingParams,
                       pixel_values, image_grid_thw) -> GenerationResult:
        with self._lock:
            if len(prompt_ids) + sp.max_new_tokens > self.cfg.max_model_len:
                sp.max_new_tokens = max(1, self.cfg.max_model_len - len(prompt_ids))
            return self.generator.generate(prompt_ids, sp, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    def _prompt_logprobs_sync(self, ids: list[int], pixel_values=None, image_grid_thw=None) -> list[float]:
        with self._lock:
            return self.generator.prompt_logprobs(ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    def _sampling_from_body(self, body: dict) -> SamplingParams:
        return SamplingParams(
            max_new_tokens=int(
                body.get("max_tokens") or body.get("max_completion_tokens") or self.cfg.default_max_tokens),
            temperature=float(body.get("temperature", 1.0)),
            top_p=float(body.get("top_p", 1.0)),
            top_k=int(body.get("top_k", 0) or 0),
            stop_token_ids=self.stop_token_ids,
            seed=body.get("seed"),
        )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def _setup_routes(self):
        app = self.app

        @app.get("/health")
        async def health():
            return {"status": "ok", "weight_version": self.weight_version}

        @app.get("/v1/models")
        async def models():
            return {"object": "list", "data": [{"id": self.cfg.model_name, "object": "model"}]}

        @app.post("/rl/load_weights")
        async def load_weights(request: Request):
            body = await request.json()
            state = torch.load(body["path"], map_location="cpu")
            await asyncio.get_running_loop().run_in_executor(
                None, self.load_state_dict, state, int(body.get("version", self.weight_version + 1))
            )
            return {"ok": True, "weight_version": self.weight_version}

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            body = await request.json()
            messages = body.get("messages") or []
            tools = body.get("tools") or None
            prompt = self.processor(messages, tools=tools)
            sp = self._sampling_from_body(body)
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(
                None, self._generate_sync, prompt.input_ids, sp, prompt.pixel_values, prompt.image_grid_thw
            )
            text = self.processor.decode(res.token_ids)
            reasoning, text = split_reasoning(text)
            content, tool_calls = parse_hermes_tool_calls(text)
            finish = "tool_calls" if tool_calls and res.finish_reason == "stop" else res.finish_reason

            message: dict = {"role": "assistant", "content": content}
            if reasoning:
                message["reasoning_content"] = reasoning
            if tool_calls:
                message["tool_calls"] = tool_calls
            choice: dict = {"index": 0, "message": message, "finish_reason": finish}
            if body.get("logprobs"):
                as_ids = bool(body.get("return_tokens_as_token_ids", False))
                choice["logprobs"] = {
                    "content": [
                        {
                            "token": f"token_id:{tid}" if as_ids else self.processor.tokenizer.decode([tid]),
                            "logprob": lp,
                            "bytes": None,
                            "top_logprobs": [],
                        }
                        for tid, lp in zip(res.token_ids, res.logprobs)
                    ]
                }
            payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.cfg.model_name,
                "choices": [choice],
                "usage": {
                    "prompt_tokens": res.prompt_len,
                    "completion_tokens": len(res.token_ids),
                    "total_tokens": res.prompt_len + len(res.token_ids),
                },
            }
            if body.get("stream"):
                from nano_areal.engine.proxy import _sse_from_completion
                return StreamingResponse(_sse_from_completion(payload), media_type="text/event-stream")
            return JSONResponse(payload)

        @app.post("/v1/completions")
        async def completions(request: Request):
            body = await request.json()
            prompt = body.get("prompt")
            if isinstance(prompt, str):
                ids = self.processor.encode_text(prompt)
            elif isinstance(prompt, list) and all(isinstance(x, int) for x in prompt):
                ids = list(prompt)
            else:
                raise HTTPException(400, "prompt must be a string or a list of token ids")
            loop = asyncio.get_running_loop()
            choice: dict = {"index": 0, "text": "", "finish_reason": "stop"}
            max_tokens = int(body.get("max_tokens", 0) or 0)
            if body.get("prompt_logprobs") is not None:
                lps = await loop.run_in_executor(None, self._prompt_logprobs_sync, ids)
                # vllm layout: entry i describes token i given tokens[:i]; entry 0 is None
                choice["prompt_logprobs"] = [None] + [
                    {str(t): {"logprob": lp, "rank": 1, "decoded_token": ""}} for t, lp in zip(ids[1:], lps)
                ]
            if max_tokens > 0:
                sp = self._sampling_from_body(body)
                res = await loop.run_in_executor(None, self._generate_sync, ids, sp, None, None)
                choice["text"] = self.processor.decode(res.token_ids)
                choice["finish_reason"] = res.finish_reason
                if body.get("logprobs") is not None:
                    choice["logprobs"] = {
                        "tokens": [f"token_id:{t}" for t in res.token_ids],
                        "token_logprobs": res.logprobs,
                    }
            return JSONResponse({
                "id": f"cmpl-{uuid.uuid4().hex}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.cfg.model_name,
                "choices": [choice],
            })
