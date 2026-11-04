"""Run pinned dsh sdk-minimal against a local mock; optionally count backend tokens."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from probe_mac_vlm import INFER_MODEL, local_url, request_json

from nano_areal.engine.messages import (
    anthropic_request_to_openai,
    openai_completion_to_messages_sse,
)

DSH_VERSION = "0.2.1rc2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-url",
        type=local_url,
        help="Server root exposing /tokenize, e.g. http://127.0.0.1:8000",
    )
    parser.add_argument("--model", default=INFER_MODEL)
    parser.add_argument(
        "--processor-path",
        type=Path,
        help="Local tokenizer directory for an offline prompt-token count",
    )
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/s1/dsh-probe.json")
    )
    args = parser.parse_args()
    captured = []
    paths = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            paths.append(self.path)
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path not in ("/v1/messages", "/messages"):
                self.send_error(404)
                return
            captured.append(payload)
            common = {"id": "s1-mock", "created": 0, "model": args.model}
            body = openai_completion_to_messages_sse(
                {
                    **common,
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "S1_OK"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
            content_type = "text/event-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    report = {
        "status": "blocked",
        "sdk_version": DSH_VERSION,
        "model": args.model,
        "profile": "sdk-minimal",
        "protocol": "messages",
        "mode": "local_mock_not_model_inference",
        "prompt_tokens": None,
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for package in ("deepseek-harness-sdk", "deepseek-harness-runtime-bin"):
            if importlib.metadata.version(package) != DSH_VERSION:
                raise RuntimeError(
                    "Install the exact versions in scripts/requirements-s1.txt"
                )
        from deepseek_harness import DeepSeekHarness

        with tempfile.TemporaryDirectory(prefix="nano-areal-s1-") as directory:
            root = Path(directory)
            workspace = root / "task"
            workspace.mkdir()
            with DeepSeekHarness(
                profile="sdk-minimal",
                model=args.model,
                cwd=str(workspace),
                dsh_home=str(root / "dsh-home"),
                env={"HOME": str(root), "XDG_CACHE_HOME": str(root / "cache")},
                max_tokens=32,
                base_url=f"http://127.0.0.1:{server.server_port}",
                api_key="s1-local-only",
                patches=(str(Path(__file__).with_name("dsh_patch.yml").resolve()),),
                initialize_timeout_seconds=args.timeout,
                request_timeout_seconds=args.timeout,
            ) as harness:
                result = harness.run("Reply S1_OK. Do not call tools.")
                report["finish_reason"] = result.finish_reason
                report["mock_roundtrip_passed"] = (
                    result.final_response.strip() == "S1_OK"
                )
        if captured:
            payload = anthropic_request_to_openai(captured[0])
            report["request_count"] = len(captured)
            report["message_count"] = len(payload.get("messages", []))
            report["tool_count"] = len(payload.get("tools", []))
            report["prompt_json_characters"] = len(
                json.dumps(payload.get("messages", []))
            )
            if args.processor_path:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    str(args.processor_path), local_files_only=True
                )
                tokens = tokenizer.apply_chat_template(
                    payload["messages"],
                    tools=payload.get("tools"),
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
                report["prompt_tokens"] = len(tokens)
                report["token_count_source"] = (
                    "local tokenizer chat template with tools"
                )
                report["backend_template_parity_verified"] = False
            if args.tokenizer_url:
                token_payload = {
                    key: payload[key]
                    for key in ("model", "messages", "tools")
                    if key in payload
                }
                token_payload["add_generation_prompt"] = True
                tokenized = request_json(
                    args.tokenizer_url, "/tokenize", token_payload, args.timeout
                )
                count = tokenized.get("count")
                if isinstance(count, int) and count > 0:
                    report["prompt_tokens"] = count
                    report["token_count_source"] = "backend /tokenize with tools"
        if report.get("mock_roundtrip_passed") and captured:
            report["status"] = "passed"
    except Exception as error:
        report["error_type"] = type(error).__name__
    finally:
        report["request_paths"] = sorted(set(paths))
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
