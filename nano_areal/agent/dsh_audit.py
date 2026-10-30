"""Fail-closed reconciliation of durable dsh messages and proxy interactions."""

from __future__ import annotations

import json


def _call_signature(call: dict) -> tuple:
    function = call.get("function", call)
    arguments = function.get("arguments", "{}")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return call.get("id"), function.get("name"), arguments


def reconcile_session(events: list[dict], interactions: list[dict]) -> dict:
    messages = [
        event["data"] for event in events if event.get("type") == "assistant/message"
    ]
    errors = []
    rows = []
    if not messages or not interactions:
        errors.append("No model interactions to reconcile")
    if len(messages) != len(interactions):
        errors.append("Assistant/proxy interaction count mismatch")
    for index, (message, interaction) in enumerate(zip(messages, interactions)):
        usage = message.get("usage", {})
        counts = (usage.get("inputTokens"), usage.get("outputTokens"))
        expected = (interaction["prompt_tokens"], interaction["output_tokens"])
        if counts != expected or usage.get("totalTokens") != sum(expected):
            errors.append(f"Token count mismatch at interaction {index}")
        blocks = message.get("message", {}).get("content", [])
        calls = [block for block in blocks if block.get("type") == "tool-call"]
        if [_call_signature(call) for call in calls] != [
            _call_signature(call) for call in interaction["tool_calls"]
        ]:
            errors.append(f"Tool call mismatch at interaction {index}")
        rows.append(
            {
                "index": index,
                "dsh_tokens": counts,
                "proxy_tokens": expected,
                "tool_calls": len(calls),
            }
        )
    calls = [event["data"] for event in events if event.get("type") == "tool/call"]
    results = [
        event["data"]["message"]
        for event in events
        if event.get("type") == "tool/result"
    ]
    predicted_ids = [call["id"] for item in interactions for call in item["tool_calls"]]
    executed_ids = [call["callId"] for call in calls]
    predicted = [
        _call_signature(call) for item in interactions for call in item["tool_calls"]
    ]
    executed = [_call_signature({**call, "id": call["callId"]}) for call in calls]
    if predicted != executed:
        errors.append("Executed tool names/arguments mismatch")
    result_ids = [result["toolCallId"] for result in results]
    if (
        predicted_ids != executed_ids
        or sorted(predicted_ids) != sorted(result_ids)
        or len(set(predicted_ids)) != len(predicted_ids)
    ):
        errors.append("Predicted/executed/result tool IDs mismatch")
    for result in results:
        text = "\n".join(block.get("text", "") for block in result["content"])
        delivered = [
            message
            for item in interactions
            for message in item.get("messages", [])
            if message.get("role") == "tool"
            and message.get("tool_call_id") == result["toolCallId"]
        ]
        if not delivered or any(
            message.get("content") != text for message in delivered
        ):
            errors.append(f"Tool result not replayed exactly: {result['toolCallId']}")
    return {
        "matched": not errors,
        "errors": errors,
        "interactions": rows,
        "executed_tools": len(calls),
        "tool_results": len(results),
        "tool_errors": sum(bool(result.get("isError")) for result in results),
        "dsh_interactions": len(messages),
        "proxy_interactions": len(interactions),
    }
