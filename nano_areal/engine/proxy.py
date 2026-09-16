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
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncOpenAI

from nano_areal.engine.multimodal import PromptProcessor
from nano_areal.engine.types import InteractionRecord, TrajectoryTensors

# Request fields forwarded verbatim to the backend. Anything else the agent
# sends (dsh extension fields, vendor params) is dropped rather than rejected.
_PASSTHROUGH_FIELDS = (
    "temperature", "max_tokens", "max_completion_tokens", "top_p", "stop", "seed",
    "presence_penalty", "frequency_penalty", "response_format", "tools",
    "tool_choice", "parallel_tool_calls", "reasoning_effort",
)
_EXTRA_BODY_FIELDS = ("top_k", "min_p", "repetition_penalty", "chat_template_kwargs", "thinking")


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
        processor: PromptProcessor | None = None,
    ):
        self.vllm_base_url = vllm_base_url
        self.tokenizer = tokenizer
        self.processor = processor or PromptProcessor(tokenizer)
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
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
        # Give the server a moment to bind
        await asyncio.sleep(0.5)

    async def stop(self):
        if self._server_task:
            self._server.should_exit = True
            try:
                await asyncio.wait_for(self._server_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
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

    def _tokenize_messages(self, messages: list[dict], tools: list[dict] | None = None) -> list[int]:
        """Tokenize a message list as prompt (add_generation_prompt=True)."""
        return self.processor(messages, tools=tools).input_ids

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

        # "/chat/completions" is the un-prefixed form some SDKs (dsh's
        # chat-completions protocol) build when base_url already ends in /v1.
        @app.post("/v1/chat/completions")
        @app.post("/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: str | None = Header(default=None),
        ):
            session = self._require_session(authorization)
            body = await request.json()
            return await self._handle_chat_completion(body, session)

        @app.get("/v1/models")
        @app.get("/models")
        async def models():
            return {"object": "list", "data": [{"id": self.model_name, "object": "model"}]}

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
            # Serialize tensors for JSON transport (pixel patches as base64 fp16)
            result = [t.to_json() for t in tensors]
            # Clean up session
            del self._sessions[session_id]
            del self._key_to_session[session.api_key]
            return result

    # ------------------------------------------------------------------
    # Core: intercept, tokenize, forward, capture
    # ------------------------------------------------------------------

    async def _handle_chat_completion(self, body: dict, session: ProxySession):
        messages: list[dict] = body.get("messages", [])
        tools = body.get("tools") or None
        stream = bool(body.get("stream", False))

        # 1. Tokenize prompt (text + image patches, exactly as the engine sees it)
        prompt = self.processor(messages, tools=tools)
        prompt_ids = prompt.input_ids

        # 2. Find parent (for multi-turn reward discount tree)
        parent_id = self._find_parent_id(messages, session)
        turn_id = session.interactions[parent_id].turn_id + 1 if parent_id else 0

        # 3. Forward to the backend with logprobs enabled. `return_tokens_as_token_ids`
        #    makes vllm report each logprob token as "token_id:<int>", giving
        #    exact sampled ids instead of a lossy re-tokenization of the text.
        #    Unknown agent-side fields (dsh extensions) are intentionally dropped.
        vllm_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "logprobs": True,
            "top_logprobs": 1,
            "extra_body": {"return_tokens_as_token_ids": True},
        }
        for k in _PASSTHROUGH_FIELDS:
            if k in body and body[k] is not None:
                vllm_kwargs[k] = body[k]
        vllm_kwargs.setdefault("temperature", 1.0)
        vllm_kwargs.setdefault("max_tokens", 1024)
        for k in _EXTRA_BODY_FIELDS:
            if k in body and body[k] is not None:
                vllm_kwargs["extra_body"][k] = body[k]
        if tools and "tool_choice" not in vllm_kwargs:
            vllm_kwargs["tool_choice"] = "auto"

        response = await self._vllm.chat.completions.create(**vllm_kwargs)

        # 4. Capture output tokens and logprobs
        choice = response.choices[0]
        content = choice.message.content or ""
        reasoning = _reasoning_of(choice.message)
        tool_calls_raw = choice.message.tool_calls or []

        # Output text: combine reasoning + content + serialized tool call args
        output_text = content
        if reasoning:
            output_text = f"<think>\n{reasoning}\n</think>\n{content}"
        if tool_calls_raw:
            tc_texts = [
                f"{tc.function.name}({tc.function.arguments})"
                for tc in tool_calls_raw
            ]
            output_text = (output_text + " " + " ".join(tc_texts)).strip()

        output_ids, token_logprobs = _extract_sampled_tokens(choice)
        if not output_ids:
            # Backend did not honour return_tokens_as_token_ids: re-tokenize text
            output_ids = self._tokenize_text(output_text) if output_text else []
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
            reasoning_content=reasoning,
            tool_calls=tool_calls_dict,
            parent_id=parent_id,
            stop_reason=choice.finish_reason or "stop",
            turn_id=turn_id,
            pixel_values=prompt.pixel_values,
            image_grid_thw=prompt.image_grid_thw,
            images=prompt.images,
        )
        session.interactions[rid] = rec
        session._last_id = rid

        # 8. Return standard ChatCompletion JSON (or an SSE stream) to the agent
        payload = response.model_dump()
        payload["choices"][0].pop("logprobs", None)
        if reasoning:
            payload["choices"][0]["message"]["reasoning_content"] = reasoning
        if stream:
            return StreamingResponse(
                _sse_from_completion(payload), media_type="text/event-stream"
            )
        return JSONResponse(content=payload)


def _reasoning_of(message) -> str:
    """vllm/dsh expose thinking as `reasoning_content` (or `reasoning`) on the message."""
    extra = getattr(message, "model_extra", None) or {}
    for key in ("reasoning_content", "reasoning"):
        val = extra.get(key) if key in extra else getattr(message, key, None)
        if isinstance(val, str) and val:
            return val
    return ""


def _sse_from_completion(payload: dict):
    """
    Emit a non-streaming ChatCompletion as an OpenAI-compatible SSE stream:
    role chunk → reasoning chunk → content chunk → tool_calls chunk → finish → [DONE].
    Generation itself is not incremental; this exists so streaming-only agents
    (dsh) can talk to the proxy while capture stays exact.
    """
    choice = payload["choices"][0]
    msg = choice["message"]
    base = {
        "id": payload.get("id"),
        "object": "chat.completion.chunk",
        "created": payload.get("created", int(time.time())),
        "model": payload.get("model", "default"),
    }

    def chunk(delta: dict, finish: str | None = None) -> str:
        d = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {json.dumps(d, ensure_ascii=False)}\n\n"

    async def gen():
        yield chunk({"role": "assistant", "content": ""})
        if msg.get("reasoning_content"):
            yield chunk({"reasoning_content": msg["reasoning_content"]})
        if msg.get("content"):
            yield chunk({"content": msg["content"]})
        if msg.get("tool_calls"):
            deltas = []
            for i, tc in enumerate(msg["tool_calls"]):
                deltas.append({
                    "index": i, "id": tc.get("id"), "type": "function",
                    "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
                })
            yield chunk({"tool_calls": deltas})
        yield chunk({}, finish=choice.get("finish_reason") or "stop")
        if payload.get("usage"):
            yield f"data: {json.dumps({**base, 'choices': [], 'usage': payload['usage']})}\n\n"
        yield "data: [DONE]\n\n"

    return gen()


def _extract_sampled_tokens(choice) -> tuple[list[int], list[float]]:
    """
    Read (token_ids, logprobs) from a vllm choice served with
    `return_tokens_as_token_ids`. Returns ([], logprobs) when tokens are
    plain strings so the caller can fall back to re-tokenization.
    """
    if not (choice.logprobs and choice.logprobs.content):
        return [], []
    lps: list[float] = [lp.logprob for lp in choice.logprobs.content]
    ids: list[int] = []
    for lp in choice.logprobs.content:
        tok = lp.token or ""
        if not tok.startswith("token_id:"):
            return [], lps
        try:
            ids.append(int(tok[len("token_id:"):]))
        except ValueError:
            return [], lps
    return ids, lps


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
            timeout=120.0,
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
                self.trajectories = [TrajectoryTensors.from_json(t) for t in r.json()]
            except Exception as e:  # pragma: no cover - network failure path
                import logging
                logging.getLogger(__name__).warning(f"export_trajectories failed: {e}")
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

    async def arun_episode(
        self, data: Any, group_id: str = ""
    ) -> list[TrajectoryTensors]:
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

        for t in client.trajectories:
            t.group_id = group_id
        return client.trajectories
