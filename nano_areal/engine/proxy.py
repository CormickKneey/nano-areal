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
import json
import secrets
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

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
    ):
        self.vllm_base_url = vllm_base_url
        self.tokenizer = tokenizer
        self.host = host
        self.port = port
        self.model_name = model_name

        self._sessions: dict[str, ProxySession] = {}
        self._key_to_session: dict[str, str] = {}   # api_key → session_id

        self.app = FastAPI(title="nano-areal proxy")
        self._vllm: AsyncOpenAI | None = None
        self._server_task: asyncio.Task | None = None

        self._setup_routes()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        import uvicorn

        self._vllm = AsyncOpenAI(
            base_url=self.vllm_base_url,
            api_key="EMPTY",
            http_client=httpx.AsyncClient(timeout=120.0),
        )

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(server.serve())
        # Give the server a moment to bind
        await asyncio.sleep(0.5)

    async def stop(self):
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
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

    def _require_session(self, authorization: str | None) -> ProxySession:
        key = (authorization or "").removeprefix("Bearer ").strip()
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

    def _tokenize_messages(self, messages: list[dict]) -> list[int]:
        """Tokenize a message list as prompt (add_generation_prompt=True)."""
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        except Exception:
            # Fallback: plain concatenation
            text = " ".join(m.get("content") or "" for m in messages)
            return self.tokenizer(text, add_special_tokens=True).input_ids

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
        ):
            session = self._require_session(authorization)
            body = await request.json()
            return await self._handle_chat_completion(body, session)

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
        self, body: dict, session: ProxySession
    ) -> JSONResponse:
        messages: list[dict] = body.get("messages", [])

        # 1. Tokenize prompt
        prompt_ids = self._tokenize_messages(messages)

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
        # Pad/trim logprobs to match output_ids length
        if len(token_logprobs) < len(output_ids):
            token_logprobs += [0.0] * (len(output_ids) - len(token_logprobs))
        token_logprobs = token_logprobs[: len(output_ids)]

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
        )
        session.interactions[rid] = rec
        session._last_id = rid

        # 8. Return standard ChatCompletion JSON to agent
        return JSONResponse(content=response.model_dump())


# ---------------------------------------------------------------------------
# Reward backpropagation + trajectory assembly
# ---------------------------------------------------------------------------

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
            timeout=30.0,
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
                await self._http.post(
                    "/rl/end_session",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
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
                    )
                    for t in r.json()
                ]
            except Exception:
                pass
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
            await self._http.post(
                "/rl/set_reward",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"reward": float(rewards)},
            )
        else:
            for rid, r in rewards.items():
                await self._http.post(
                    "/rl/set_reward",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"interaction_id": rid, "reward": float(r)},
                )

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
