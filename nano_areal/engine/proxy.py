"""
OpenAI-compatible proxy server for nano-areal.

Architecture (mirrors AReaL proxy mode):

  Agent
    │  AsyncOpenAI(base_url=proxy_url, api_key=session_key)
    ▼
  ProxyServer  (FastAPI, this file)
    │  tokenize input → capture output logprobs → cache
    ▼
  vllm  (OpenAI-compatible, GPU 0)

The agent is completely decoupled from AReaL — it only needs a base_url
and an api_key (session key). Any OpenAI-compatible agent framework works.

Session lifecycle:
  1. POST /rl/start_session          → {session_id, api_key}
  2. POST /v1/chat/completions       → ChatCompletion  (N times)
  3. POST /rl/set_reward             → {}
  4. POST /rl/end_session            → {}
  5. POST /export_trajectories       → list[TrajectoryTensors]
"""
from __future__ import annotations

import asyncio
import math
import secrets
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncOpenAI

from nano_areal.engine.messages import (
    anthropic_request_to_openai,
    openai_completion_to_messages_sse,
)
from nano_areal.engine.multimodal import has_images, prepare_single_image
from nano_areal.engine.types import InteractionRecord, TrajectoryTensors

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class ProxySession:
    session_id: str
    api_key: str                                          # unique per session
    interactions: dict[str, InteractionRecord] = field(default_factory=dict)
    _last_id: str | None = field(default=None, repr=False)
    active: bool = True

    def last_interaction(self) -> InteractionRecord | None:
        return self.interactions.get(self._last_id) if self._last_id else None


# ---------------------------------------------------------------------------
# Proxy server
# ---------------------------------------------------------------------------

ADMIN_KEY = "nano-areal-admin"   # fixed for single-node use


class OpenAIProxyServer:
    """
    FastAPI application that intercepts agent ↔ vllm traffic.

    Runs as a background asyncio task started by RolloutEngine.
    All state lives in-process — no Redis, no DB.
    """

    def __init__(
        self,
        vllm_base_url: str,
        tokenizer,
        host: str = "127.0.0.1",
        port: int = 8001,
        model_name: str = "default",
        processor=None,
    ):
        self.vllm_base_url = vllm_base_url
        self.tokenizer = tokenizer
        self.host = host
        self.port = port
        self.model_name = model_name
        self.processor = processor

        self._sessions: dict[str, ProxySession] = {}
        self._key_to_session: dict[str, str] = {}   # api_key → session_id

        self.app = FastAPI(title="nano-areal proxy")
        self._vllm: AsyncOpenAI | None = None
        self._server_task: asyncio.Task | None = None
        self._server = None
        self.expected_policy_version: int | None = None

        self._setup_routes()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        import uvicorn

        self._vllm = AsyncOpenAI(
            base_url=self.vllm_base_url,
            api_key="EMPTY",
            http_client=httpx.AsyncClient(timeout=600.0),
            max_retries=0,
        )

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        self._server = server
        self._server_task = asyncio.create_task(server.serve())
        # Give the server a moment to bind
        await asyncio.sleep(0.5)

    async def stop(self):
        if self._server_task:
            self._server.should_exit = True
            try:
                await asyncio.wait_for(self._server_task, timeout=10.0)
            except asyncio.TimeoutError:
                self._server_task.cancel()
        if self._vllm:
            await self._vllm.close()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _require_admin(self, authorization: str | None):
        key = (authorization or "").removeprefix("Bearer ").strip()
        if key != ADMIN_KEY:
            raise HTTPException(status_code=401, detail="Invalid admin key")

    def _require_session(
        self,
        authorization: str | None,
        x_api_key: str | None = None,
    ) -> ProxySession:
        # BFCL sends Authorization: Bearer. dsh sends x-api-key.
        key = (x_api_key or "").strip() or (authorization or "").removeprefix("Bearer ").strip()
        sid = self._key_to_session.get(key)
        if not sid or sid not in self._sessions:
            raise HTTPException(status_code=401, detail="Invalid session key")
        session = self._sessions[sid]
        if not session.active:
            raise HTTPException(status_code=410, detail="Session already closed")
        return session

    # ------------------------------------------------------------------
    # Token capture helpers
    # ------------------------------------------------------------------

    def _tokenize_messages(self, messages: list[dict], tools=None) -> list[int]:
        """Tokenize a message list as prompt (add_generation_prompt=True)."""
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=True,
            return_dict=False,
        )

    def _tokenize_text(self, text: str) -> list[int]:
        """Tokenize raw text without special tokens."""
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _find_parent_id(self, messages: list[dict], session: ProxySession) -> str | None:
        """
        Find the parent interaction: the cached interaction whose messages
        are a prefix of the current messages list.
        Mirrors AReaL's InteractionCache parent-child parsing.
        """
        best_parent: str | None = None
        best_len = 0
        for rid, rec in session.interactions.items():
            n = len(rec.messages)
            if n < len(messages) and messages[:n] == rec.messages and n > best_len:
                best_parent = rid
                best_len = n
        return best_parent

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def _setup_routes(self):
        app = self.app

        @app.post("/rl/start_session")
        async def start_session(authorization: str | None = Header(default=None)):
            self._require_admin(authorization)
            session_id = str(uuid.uuid4())
            api_key = secrets.token_hex(16)
            self._sessions[session_id] = ProxySession(
                session_id=session_id,
                api_key=api_key,
            )
            self._key_to_session[api_key] = session_id
            return {"session_id": session_id, "api_key": api_key}

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: str | None = Header(default=None),
            x_api_key: str | None = Header(default=None),
        ):
            session = self._require_session(authorization, x_api_key)
            body = await request.json()
            return await self._handle_chat_completion(body, session)

        @app.post("/v1/messages")
        async def messages(
            request: Request,
            authorization: str | None = Header(default=None),
            x_api_key: str | None = Header(default=None),
        ):
            # dsh sdk-minimal posts Anthropic Messages here and requires SSE.
            session = self._require_session(authorization, x_api_key)
            body = await request.json()
            try:
                openai_body = anthropic_request_to_openai(body)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            payload = await self._handle_chat_completion(openai_body, session, return_payload=True)
            sse = openai_completion_to_messages_sse(payload)
            return StreamingResponse(iter([sse]), media_type="text/event-stream")

        @app.post("/rl/set_reward")
        async def set_reward(
            request: Request,
            authorization: str | None = Header(default=None),
        ):
            session = self._require_session(authorization)
            body = await request.json()
            rid = body.get("interaction_id") or session._last_id
            reward = float(body["reward"])
            if rid and rid in session.interactions:
                session.interactions[rid].reward = reward
                return {"ok": True, "interaction_id": rid}
            raise HTTPException(status_code=404, detail=f"Interaction {rid!r} not found")

        @app.post("/rl/end_session")
        async def end_session(authorization: str | None = Header(default=None)):
            session = self._require_session(authorization)
            session.active = False
            return {"ok": True}

        @app.post("/export_trajectories")
        async def export_trajectories(
            request: Request,
            authorization: str | None = Header(default=None),
        ):
            self._require_admin(authorization)
            body = await request.json()
            session_id = body["session_id"]
            discount = float(body.get("discount", 0.9))

            session = self._sessions.get(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

            tensors = _build_trajectories(session, discount)
            # Serialize tensors to lists for JSON transport
            result = [
                {
                    "input_ids": t.input_ids.tolist(),
                    "response_mask": t.response_mask.tolist(),
                    "logprobs": t.logprobs.tolist(),
                    "reward": t.reward,
                    "truncated": t.truncated,
                    "pixel_values": (
                        t.pixel_values.tolist() if t.pixel_values is not None else None
                    ),
                    "image_grid_thw": (
                        t.image_grid_thw.tolist() if t.image_grid_thw is not None else None
                    ),
                }
                for t in tensors
            ]
            # Clean up session
            del self._sessions[session_id]
            del self._key_to_session[session.api_key]
            return result

    # ------------------------------------------------------------------
    # Core: intercept, tokenize, forward, capture
    # ------------------------------------------------------------------

    async def _handle_chat_completion(
        self,
        body: dict,
        session: ProxySession,
        return_payload: bool = False,
    ) -> JSONResponse | dict:
        messages: list[dict] = body.get("messages", [])

        # 1. Tokenize prompt
        pixel_values, image_grid_thw = None, None
        if has_images(messages):
            if self.processor is None:
                raise HTTPException(
                    422, "Image requests require an explicit local processor"
                )
            try:
                prompt_ids, pixel_values, image_grid_thw = prepare_single_image(
                    self.processor, messages, body.get("tools")
                )
            except (ValueError, OSError, KeyError) as error:
                raise HTTPException(422, str(error)) from error
        else:
            prompt_ids = self._tokenize_messages(messages, body.get("tools"))

        # 2. Find parent (for multi-turn reward discount tree)
        parent_id = self._find_parent_id(messages, session)

        # 3. Forward to vllm with logprobs enabled
        vllm_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": body.get("temperature", 1.0),
            "max_tokens": body.get("max_tokens", 1024),
            "top_p": body.get("top_p", 1.0),
            "logprobs": True,
            "top_logprobs": 1,
            "extra_body": {"return_token_ids": True, "top_k": -1},
        }
        if body.get("tools"):
            vllm_kwargs["tools"] = body["tools"]
            vllm_kwargs["tool_choice"] = body.get("tool_choice", "auto")

        response = await self._vllm.chat.completions.create(**vllm_kwargs)

        # 4. Capture output tokens and logprobs
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls_raw = choice.message.tool_calls or []

        # Output text: combine content + serialized tool call args
        output_text = content
        if tool_calls_raw:
            tc_texts = [
                f"{tc.function.name}({tc.function.arguments})"
                for tc in tool_calls_raw
            ]
            output_text = (content + " " + " ".join(tc_texts)).strip()

        output_ids = self._tokenize_text(output_text) if output_text else []

        # Extract per-token logprobs from vllm response
        token_logprobs: list[float] = []
        if choice.logprobs and choice.logprobs.content:
            token_logprobs = [lp.logprob for lp in choice.logprobs.content]
        exact = (response.model_extra or {}).get("nano_areal")
        if exact is not None:
            expected_prompt = prompt_ids
            if (self.expected_policy_version is not None
                    and exact.get("policy_version") != self.expected_policy_version):
                raise HTTPException(502, "Backend policy version differs from rollout version")
            prompt_ids = exact["prompt_token_ids"]
            output_ids = exact["completion_token_ids"]
            token_logprobs = exact["token_logprobs"]
            if prompt_ids != expected_prompt:
                raise HTTPException(502, "Backend prompt tokenization differs from proxy")
        elif getattr(response, "prompt_token_ids", None) is not None:
            backend_prompt = response.prompt_token_ids
            output_ids = getattr(choice, "token_ids", None)
            if backend_prompt != prompt_ids:
                raise HTTPException(502, "Backend prompt tokenization differs from proxy")
            if self.expected_policy_version is not None:
                raise HTTPException(502, "Backend does not report a policy version")
            prompt_ids = backend_prompt
            if (response.usage is None
                    or response.usage.prompt_tokens != len(prompt_ids)
                    or output_ids is None
                    or response.usage.completion_tokens != len(output_ids)):
                raise HTTPException(502, "Backend token IDs differ from usage")
        else:
            raise HTTPException(502, "Trajectories require exact backend token IDs")
        if not output_ids or len(token_logprobs) != len(output_ids):
            raise HTTPException(502, "Missing or misaligned behavior logprobs")
        if not all(math.isfinite(value) and value <= 0 for value in token_logprobs):
            raise HTTPException(502, "Invalid behavior logprobs")

        # 5. Build full sequence: prompt + output
        full_ids = prompt_ids + output_ids
        response_mask = [0] * len(prompt_ids) + [1] * len(output_ids)

        # 6. Parse tool calls to dicts
        tool_calls_dict = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
            for tc in tool_calls_raw
        ]

        # 7. Cache
        rid = response.id
        rec = InteractionRecord(
            response_id=rid,
            input_ids=full_ids,
            response_mask=response_mask,
            token_logprobs=token_logprobs,
            reward=None,
            messages=messages,
            response_content=content,
            tool_calls=tool_calls_dict,
            parent_id=parent_id,
            finish_reason=_finish_reason(choice.finish_reason),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        session.interactions[rid] = rec
        session._last_id = rid

        # 8. Return standard ChatCompletion JSON, or the payload for Messages SSE.
        payload = response.model_dump()
        reasoning = getattr(choice.message, "reasoning_content", None)
        if reasoning and payload.get("choices"):
            payload["choices"][0].setdefault("message", {})["reasoning_content"] = reasoning
        if return_payload:
            return payload
        return JSONResponse(content=payload)


# ---------------------------------------------------------------------------
# Reward backpropagation + trajectory assembly
# ---------------------------------------------------------------------------

def _finish_reason(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    enum_value = getattr(value, "value", None)
    return enum_value if isinstance(enum_value, str) else str(value)


def _build_trajectories(
    session: ProxySession,
    discount: float,
) -> list[TrajectoryTensors]:
    """
    Apply geometric reward discount along conversation tree,
    then convert each interaction to TrajectoryTensors.

    Discount rule (matches AReaL):
      leaf reward = R
      parent reward += R * discount
      grandparent reward += R * discount^2
      ...
    """
    interactions = session.interactions

    # Propagate rewards backward along parent chain
    for rid in reversed(list(interactions)):
        rec = interactions[rid]
        if rec.reward is None:
            rec.reward = 0.0
        pid = rec.parent_id
        if pid and pid in interactions:
            parent = interactions[pid]
            if parent.reward is None:
                parent.reward = 0.0
            parent.reward += rec.reward * discount

    return [rec.to_trajectory() for rec in interactions.values()]


# ---------------------------------------------------------------------------
# Client + Workflow (used by train.py)
# ---------------------------------------------------------------------------

class OpenAIProxyClient:
    """
    Manages one RL session with the proxy server.

    Usage:
        async with OpenAIProxyClient(proxy_url) as session:
            rewards = await my_agent.run(data,
                base_url=session.agent_base_url,
                api_key=session.api_key)
            await session.set_rewards(rewards)
        trajectories = session.trajectories
    """

    def __init__(self, proxy_base_url: str):
        self.proxy_base_url = proxy_base_url
        self.session_id: str | None = None
        self.api_key: str | None = None
        self.trajectories: list[TrajectoryTensors] = []
        self._http: httpx.AsyncClient | None = None

    @property
    def agent_base_url(self) -> str:
        return f"{self.proxy_base_url}/v1"

    async def __aenter__(self):
        self._http = httpx.AsyncClient(
            base_url=self.proxy_base_url,
            timeout=600.0,
        )
        r = await self._http.post(
            "/rl/start_session",
            headers={"Authorization": f"Bearer {ADMIN_KEY}"},
        )
        r.raise_for_status()
        data = r.json()
        self.session_id = data["session_id"]
        self.api_key = data["api_key"]
        return self

    async def __aexit__(self, *args):
        if self._http and self.session_id:
            try:
                ended = await self._http.post(
                    "/rl/end_session",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                ended.raise_for_status()
                r = await self._http.post(
                    "/export_trajectories",
                    headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                    json={"session_id": self.session_id, "discount": self._discount},
                )
                r.raise_for_status()
                import torch
                self.trajectories = [
                    TrajectoryTensors(
                        input_ids=torch.tensor(t["input_ids"], dtype=torch.long),
                        response_mask=torch.tensor(t["response_mask"], dtype=torch.float),
                        logprobs=torch.tensor(t["logprobs"], dtype=torch.float),
                        reward=t["reward"],
                        truncated=t.get("truncated", False),
                        pixel_values=(torch.tensor(t["pixel_values"], dtype=torch.float32)
                                      if t.get("pixel_values") is not None else None),
                        image_grid_thw=(torch.tensor(t["image_grid_thw"], dtype=torch.long)
                                        if t.get("image_grid_thw") is not None else None),
                    )
                    for t in r.json()
                ]
            except Exception:
                if not args or args[0] is None:
                    raise
            finally:
                await self._http.aclose()
        if self._http:
            await self._http.aclose()

    async def set_rewards(self, rewards: float | dict[str, float], discount: float = 0.9):
        """
        Assign rewards before __aexit__ triggers export.

        rewards:
          float            → apply to last interaction
          dict[id, float]  → apply per interaction
        """
        self._discount = discount
        assert self._http and self.api_key
        if isinstance(rewards, (int, float)):
            response = await self._http.post(
                "/rl/set_reward",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"reward": float(rewards)},
            )
            response.raise_for_status()
        else:
            for rid, r in rewards.items():
                response = await self._http.post(
                    "/rl/set_reward",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"interaction_id": rid, "reward": float(r)},
                )
                response.raise_for_status()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    # Set default discount so __aexit__ always has a value
    _discount: float = 0.9


class OpenAIProxyWorkflow:
    """
    Wraps any agent with `async def run(data, **extra_kwargs)` into
    a RolloutWorkflow-compatible callable.

    The agent receives:
        base_url  = proxy /v1 endpoint
        api_key   = per-session auth token
        http_client = shared httpx.AsyncClient (optional, reduces latency)

    It returns:
        float              → reward for last interaction
        dict[str, float]   → {response_id: reward} per interaction
    """

    def __init__(
        self,
        agent,
        proxy_server: OpenAIProxyServer,
        turn_discount: float = 0.9,
    ):
        self.agent = agent
        self.proxy_server = proxy_server
        self.turn_discount = turn_discount

    async def arun_episode(self, data: Any) -> list[TrajectoryTensors]:
        client = OpenAIProxyClient(self.proxy_server.base_url)
        async with client as session:
            # Shared httpx client for lower latency (avoids per-call TCP handshake)
            shared_http = httpx.AsyncClient(
                base_url=session.agent_base_url,
                timeout=60.0,
            )
            try:
                rewards = await self.agent.run(
                    data,
                    base_url=session.agent_base_url,
                    api_key=session.api_key,
                    http_client=shared_http,
                )
            finally:
                await shared_http.aclose()

            await session.set_rewards(rewards, discount=self.turn_discount)

        return client.trajectories
