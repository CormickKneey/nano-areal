"""Anthropic Messages bridge for deepseek-harness (dsh).

Current ``@deepseek-ai/dsh-llm-deepseek`` rejects a configurable protocol and
POSTs ``{baseURL}/v1/messages`` (or ``{baseURL}/messages`` when the base path
already ends in ``/v1``). The request asks for ``text/event-stream``, sends
``x-api-key`` and ``anthropic-version: 2023-06-01``.

nano-areal still completes against vLLM's OpenAI endpoint, then re-encodes
one completion into the Messages SSE events dsh's translator accepts:
``message_start``, content blocks, ``message_delta`` with ``stop_reason``,
and ``message_stop``.
"""
from __future__ import annotations

import json
from typing import Any


def messages_base_url(proxy_base_url: str) -> str:
    """Workflow base (often ``http://host:port/v1``) → dsh ``DEEPSEEK_BASE_URL``.

    dsh appends ``/v1`` unless the path already ends in ``/v1``, then POSTs
    ``/messages``. Returning the origin root avoids ``/v1/v1/messages``.
    """
    url = proxy_base_url.rstrip("/")
    if url.endswith("/v1"):
        return url[: -len("/v1")]
    return url


def message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(_block_text(part))
        return " ".join(p for p in parts if p)
    return str(content)


def anthropic_request_to_openai(body: dict) -> dict:
    """Messages request → the Chat Completions body vLLM already serves.

    Unknown top-level fields (session log, plugin inventory, betas) are not
    forwarded. Inline image blocks retain their bytes for the VLM processor.
    """
    messages: list[dict] = []
    system_text = message_text(body.get("system"))
    if system_text.strip():
        messages.append({"role": "system", "content": system_text})

    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or "user"
        content = msg.get("content")
        if isinstance(content, str) or not isinstance(content, list):
            messages.append({"role": role, "content": message_text(content)})
            continue
        text_parts: list[str] = []
        content_parts: list[dict] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "tool_use":
                tool_calls.append({
                    "id": str(block.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name") or ""),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                })
            elif kind == "tool_result":
                result_content = block.get("content")
                if isinstance(result_content, list) and any(
                    isinstance(part, dict) and part.get("type") in {"image", "image_url"}
                    for part in result_content
                ):
                    raise ValueError("Images inside tool results are not supported")
                tool_results.append(block)
            elif kind == "thinking":
                continue
            elif kind == "image":
                if role != "user":
                    raise ValueError("Images are supported only in user messages")
                source = block.get("source") or {}
                if source.get("type") != "base64" or source.get("media_type") not in {
                    "image/png", "image/jpeg", "image/webp"}:
                    raise ValueError("Expected an inline PNG/JPEG/WebP image")
                content_parts.append({"type": "image_url", "image_url": {
                    "url": f"data:{source['media_type']};base64,{source.get('data', '')}"}})
            else:
                text = _block_text(block)
                if text:
                    text_parts.append(text)
                    content_parts.append({"type": "text", "text": text})
        if role == "assistant":
            item: dict = {"role": "assistant", "content": "".join(text_parts)}
            if tool_calls:
                item["tool_calls"] = tool_calls
            messages.append(item)
        else:
            if content_parts:
                has_image = any(part["type"] == "image_url" for part in content_parts)
                messages.append({"role": "user", "content": content_parts if has_image
                                 else "".join(text_parts)})
            for result in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": str(result.get("tool_use_id") or ""),
                    "content": message_text(result.get("content")),
                })

    openai: dict = {
        "model": body.get("model") or "default",
        "messages": messages,
        "max_tokens": body.get("max_tokens") or 1024,
    }
    if "temperature" in body:
        openai["temperature"] = body["temperature"]
    tools = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description") or "",
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    if tools:
        openai["tools"] = tools
        openai["tool_choice"] = "auto"
    return openai


def openai_completion_to_messages_sse(completion: dict) -> str:
    """One ChatCompletion dict → Messages SSE text."""
    choice = (completion.get("choices") or [{}])[0] or {}
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    finish = choice.get("finish_reason") or "stop"
    stop_reason = _stop_reason(finish, bool(tool_calls))
    usage = completion.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    model = completion.get("model") or "default"
    message_id = completion.get("id") or "msg_nano"

    events: list[tuple[str, dict]] = [(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        },
    )]

    index = 0
    reasoning = message.get("reasoning_content") or ""
    if isinstance(reasoning, str) and reasoning and stop_reason != "max_tokens":
        events.extend(_text_block(index, "thinking", reasoning))
        index += 1
    content = message.get("content") or ""
    if isinstance(content, str) and content:
        events.extend(_text_block(index, "text", content))
        index += 1
    if tool_calls and stop_reason != "max_tokens":
        for call in tool_calls:
            fn = call.get("function") or {}
            events.extend(_tool_block(
                index,
                str(call.get("id") or f"call_{index}"),
                str(fn.get("name") or "tool"),
                str(fn.get("arguments") or "{}"),
            ))
            index += 1
    if index == 0:
        # dsh rejects message_stop with zero blocks on an end_turn.
        events.extend(_text_block(0, "text", "" if stop_reason == "max_tokens" else "."))

    events.append((
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": {"output_tokens": output_tokens},
        },
    ))
    events.append(("message_stop", {"type": "message_stop"}))
    return "".join(
        f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        for name, payload in events
    )


def is_truncated_finish(finish_reason: str | None) -> bool:
    return finish_reason in {"length", "max_tokens"}


def _stop_reason(finish: str, has_tools: bool) -> str:
    if finish in {"length", "max_tokens"}:
        return "max_tokens"
    if has_tools or finish == "tool_calls":
        return "tool_use"
    return "end_turn"


def _text_block(index: int, kind: str, text: str) -> list[tuple[str, dict]]:
    if kind == "thinking":
        start = {"type": "thinking", "thinking": ""}
        delta = {"type": "thinking_delta", "thinking": text}
    else:
        start = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": text}
    return [
        ("content_block_start", {
            "type": "content_block_start",
            "index": index,
            "content_block": start,
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": delta,
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": index}),
    ]


def _tool_block(index: int, call_id: str, name: str, arguments: str) -> list[tuple[str, dict]]:
    return [
        ("content_block_start", {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": {},
            },
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": arguments},
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": index}),
    ]


def _block_text(block: dict) -> str:
    kind = block.get("type")
    if kind == "text":
        return str(block.get("text") or "")
    if kind in {"image", "image_url", "input_image"}:
        return "<image>"
    if kind == "tool_result":
        return message_text(block.get("content"))
    return ""
