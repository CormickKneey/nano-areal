"""Loopback-only, text-only MLX fallback; not a production vllm-metal server."""

from __future__ import annotations

import argparse
import math
import os
import secrets
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException


def collect_tokens(chunks) -> dict:
    text, tokens, logprobs = [], [], []
    final = None
    for chunk in chunks:
        text.append(chunk.text)
        if chunk.finish_reason is None or (
            chunk.token_ids and len(chunk.token_ids) > len(tokens)
        ):
            if chunk.token is not None:
                tokens.append(int(chunk.token))
                logprobs.append(float(chunk.logprobs.reshape(-1)[chunk.token].item()))
        if chunk.finish_reason is not None:
            final = chunk
    if final is None or tokens != final.token_ids:
        raise ValueError("MLX terminal token IDs disagree with collected tokens")
    if not tokens or len(tokens) != final.generation_tokens:
        raise ValueError("MLX token count mismatch")
    if not all(math.isfinite(value) and value <= 0 for value in logprobs):
        raise ValueError("MLX returned invalid logprobs")
    return {
        "text": "".join(text),
        "tokens": tokens,
        "logprobs": logprobs,
        "finish_reason": final.finish_reason,
        "prompt_tokens": final.prompt_tokens,
    }


def create_app(model_path: Path, adapter_path: Path, max_context: int = 2048):
    import mlx.core as mx
    from mlx_vlm import load, stream_generate
    from mlx_vlm.trainer.peft.lora_layers import LoRALinear

    model, processor = load(str(model_path.resolve()))
    mx.random.seed(42)
    tokenizer = processor.tokenizer
    for layer in model.language_model.model.layers:
        for name in ("q_proj", "v_proj"):
            attention = layer.self_attn
            setattr(
                attention,
                name,
                LoRALinear.from_base(
                    getattr(attention, name),
                    r=8,
                    scale=2.0,
                    dropout=0.0,
                ),
            )
    mx.eval(model.parameters())
    model.eval()
    app = FastAPI(title="nano-areal MLX fallback (text only)")
    version = 0
    admin_key = os.environ["NANO_MLX_ADMIN_KEY"]

    @app.get("/health")
    async def health():
        return {
            "backend": "mlx-vlm-fallback",
            "policy_version": version,
            "mlx_active_bytes": mx.get_active_memory(),
            "mlx_peak_bytes": mx.get_peak_memory(),
        }

    @app.post("/admin/adapters")
    async def adapters(body: dict, x_admin_key: str = Header(default="")):
        nonlocal version
        if not secrets.compare_digest(admin_key, x_admin_key):
            raise HTTPException(401, "Invalid admin key")
        if body.get("version") != version + 1:
            raise HTTPException(409, "Adapter version must advance by exactly one")
        weights = mx.load(str(adapter_path))
        expected = {
            f"language_model.model.layers.{index}.self_attn.{name}.{suffix}"
            for index in range(len(model.language_model.model.layers))
            for name in ("q_proj", "v_proj")
            for suffix in ("lora_a", "lora_b")
        }
        if set(weights) != expected:
            raise HTTPException(400, "Adapter keys do not match language-only LoRA")
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
        mx.clear_cache()
        version = body["version"]
        return {"policy_version": version, "adapter_tensors": len(weights)}

    @app.post("/v1/chat/completions")
    async def completion(body: dict):
        messages = body.get("messages", [])
        if any(not isinstance(item.get("content", ""), str) for item in messages):
            raise HTTPException(400, "Fallback supports text-only messages")
        if body.get("stream"):
            raise HTTPException(400, "Streaming is provided by the Messages proxy")
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tools=body.get("tools"),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        max_tokens = int(body.get("max_tokens", 32))
        if body.get("temperature", 1.0) != 1.0 or body.get("top_p", 1.0) != 1.0:
            raise HTTPException(
                400, "RL fallback requires untempered full-distribution sampling"
            )
        if max_tokens < 1 or len(prompt_ids) + max_tokens > max_context:
            raise HTTPException(400, "Request exceeds the fallback context budget")
        result = collect_tokens(
            stream_generate(
                model,
                processor,
                "",
                input_ids=mx.array([prompt_ids]),
                max_tokens=max_tokens,
                temperature=body.get("temperature", 1.0),
                top_p=1.0,
                top_k=0,
                logits_processors=[lambda tokens, logits: logits.astype(mx.float32)],
            )
        )
        if result["prompt_tokens"] != len(prompt_ids):
            raise HTTPException(500, "Prepared prompt token count mismatch")
        content = result["text"]
        if "<tool_call>" in content:
            raise HTTPException(422, "Fallback tool-call parsing is not implemented")
        return {
            "id": "mlx-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "default"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": result["finish_reason"],
                    "logprobs": {
                        "content": [
                            {
                                "token": tokenizer.decode([token]),
                                "logprob": value,
                                "bytes": None,
                                "top_logprobs": [],
                            }
                            for token, value in zip(
                                result["tokens"], result["logprobs"]
                            )
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(result["tokens"]),
                "total_tokens": len(prompt_ids) + len(result["tokens"]),
            },
            "nano_areal": {
                "prompt_token_ids": prompt_ids,
                "completion_token_ids": result["tokens"],
                "token_logprobs": result["logprobs"],
                "policy_version": version,
            },
        }

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-context", type=int, default=2048)
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.model_path, args.adapter_path, args.max_context),
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
