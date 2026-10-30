# S3 multimodal foundation: October 30, 2026

## Status: hard-blocked, not S3 acceptance

### Follow-up: parser correction and unchanged fixture, October 30

The installed checkpoint's `chat_template.jinja` explicitly requests JSON
inside `<tool_call>` tags. `qwen3_xml` expects a different function/parameter
XML grammar. A production explicit-path diagnostic generated tool tokens but
returned an empty parsed assistant message with that parser. The proxy's
empty-message compatibility block consequently appeared as `.` in dsh.
Use **`--tool-call-parser hermes`** for this checkpoint, not `qwen3_xml`.
No package source, weights, or tool-choice policy was modified.

The same `BFCLAgent` two-turn fixture was rerun at temperature 0,
top-p 0.95, max output 128, with automatic tool choice:

| Production parser | Unchanged `List files` / `Again` | Explicit-path diagnostic |
| --- | --- | --- |
| `qwen3_xml` | 0/2 | 0/2; empty parsed responses |
| `hermes` | 0/2; asks for a path | 2/2; parsed `ls(path=".")` both turns |

The unchanged fixture omits the directory while its verifier expects `.`.
Clarification is therefore understandable, but changing its prompt would
change the comparison. **The exact acceptance comparison remains failed;
S3 is not green.** The diagnostic establishes production two-turn tool
transport, not BFCL benchmark accuracy or quality non-regression.

Reproduce against each matching production server configuration:

```bash
PYTHONPATH=. .venv/bin/python scripts/probe_bfcl_tools.py \
  --parser hermes --output artifacts/s3/bfcl-hermes-final.json
```

`--parser` labels the evidence; it does not reconfigure the running server.
Evidence: `artifacts/s3/bfcl-qwen3-xml.json`, `bfcl-hermes.json`, and
`bfcl-hermes-final.json`. Full responses, usages, tool arguments, and turn
rewards are retained. Continue only S4 foundations; do not start GRPO.

The production Messages proxy now preserves inline image bytes instead of
replacing them with `<image>`. An explicitly supplied local HF processor
expands visual prompt IDs and produces `pixel_values` / `image_grid_thw`.
Those fields survive interaction capture, JSON export, client decoding and
training-trajectory construction. Native archon recomputes interleaved
positions from the full token sequence and grid, including response tokens.
The HF bridge and legacy dense trainer reject image trajectories rather than
silently training on text alone.

The processor accepts one inline PNG/JPEG/WebP still image, at most 4 megapixels
and a 20 MB data URI. Remote/file URLs, multiple images, assistant images and
images inside tool results are rejected. No URL fetching is done by the image
preparation helper. Backend prompt IDs must exactly match processor-expanded
IDs; native output IDs and usage/logprobs must agree as in S1.

## Measured production smoke

`scripts/probe_vllm_http.py --proxy-image` passed against the official
vllm-metal server, not the fallback. A real `/v1/messages` image request returned
`Red`, 81 prompt / 2 output tokens. Captured pixels have shape `[256, 1536]`,
grid `[[1, 16, 16]]`, and logprobs
`[-0.000556314189452678, -2.3841855067985307e-7]`. The same process then completed
the real dsh no-tool replay: 559 prompt / 4 output tokens, three tool schemas.
Temporary dsh workspace paths can change prompt token counts across replays.

This second production launch used an explicit **1,342,177,280-byte KV budget**
instead of the S1 launch's fraction-derived 2.77 GiB allocation. It served the
same 8192-token maximum with one sequence. No training co-residency claim is
made for this reduced budget. Both owned inference processes were stopped
after the probe to release unified memory.

Evidence is ignored under `artifacts/s3/vllm-image-proxy.json` and
`artifacts/s3/vllm-server.log`. Offline tests cover pixel/grid JSON roundtrip,
image payload ordering, unsafe URL rejection, missing-processor rejection,
and a two-turn BFCL tool/history/reward flow. **68 regression tests pass.**
The BFCL flow test uses a fake completion provider; it is not a model-backed
BFCL benchmark or a measured BFCL accuracy non-regression claim.

## Measured CLEVR L0 probe

`nano_areal/clevr.py` reads official local train/val question files and images;
it checks required fields and rejects image paths escaping the split directory.
`scripts/train_clevr_l0.py` runs a bounded, frozen-ViT native LoRA supervised
overfit probe and requires loss on the same examples to decrease. Only answer
tokens are supervised, and the default context ceiling is 1024 tokens. This
is a loss-drop probe, not held-out accuracy or GRPO.

The plan-approved `BUAADreamer/clevr_count_70k` test parquet was downloaded to
the repository-local cache. It contains 200 real PNG/question/answer rows; the
bounded probe used one row with an embedded 275,732-byte PNG. The native MPS
LoRA run passed five steps at max context 1024:

| metric | measured value |
| --- | ---: |
| `loss_before` | `5.563095442084887e-07` |
| `loss_after` | `3.973642748178463e-08` |
| relative decrease | `92.86%` |
| samples / steps | `1 / 5` |
| peak sampled driver RSS | `1,089,257,472` bytes |
| sampled swap during probe | `3,330 MiB used` |

Evidence is in `artifacts/s3/clevr-count-loss.json` and the command log in
`artifacts/s3/clevr-count-loss.log`. The runner now accepts this local parquet
subset with `--format clevr-count-parquet`; the official CLEVR JSON/image
layout remains supported. This is a supervised same-example loss probe, not
held-out accuracy or GRPO.

After placing the official dataset under the repository-local cache:

```bash
export HF_HOME="$PWD/.s1-cache/hf" HF_HUB_OFFLINE=1
export UV_CACHE_DIR="$PWD/.s1-cache/uv" TMPDIR="$PWD/.s1-cache/tmp"
export XDG_CACHE_HOME="$PWD/.s1-cache/xdg"
PYTHONPATH=. .venv/bin/python scripts/train_clevr_l0.py \
  --data-root .s1-cache/datasets/clevr_count_70k_probe/test.parquet \
  --format clevr-count-parquet \
  --model-path .s1-cache/models/Qwen3-VL-4B-Instruct \
  --device mps --samples 1 --steps 5 --max-context 1024
```

Run this **without a resident inference server** first. The smallest real
probe command above is the one used for the measured result.

## BFCL model-backed gate

The offline two-turn `ls` fixture still passes with its fake provider, but the
same two-turn fixture was sent to the production vllm-metal completion path.
The result was **0/2 rewards**: both real model completions asked for a
directory path and emitted no tool call, while the offline baseline expects
`ls(path=".")` on both turns. Evidence is in
`artifacts/s3/bfcl-model.json` and `artifacts/s3/bfcl-model.log`.

This is a measured model-backed non-regression failure, not an acceptance
claim. The smallest reproducer is the two-turn production request in the
logged probe: vllm-metal server with `--enable-auto-tool-choice
--tool-call-parser qwen3_xml`, one `ls(path)` tool, user turns `List files`
and `Again`, `temperature=0`, and `max_tokens=128`.

S3 therefore remains hard-blocked on model-backed BFCL behavior. The CLEVR
gate is green; fake-provider BFCL tests do not replace this failure.

To reproduce production image transport, launch the S1 server with
`--kv-cache-memory-bytes 1342177280` in place of
`--gpu-memory-utilization 0.25`, then run:

```bash
PYTHONPATH=. .venv/bin/python scripts/probe_vllm_http.py --proxy-image \
  --output artifacts/s3/vllm-image-proxy.json
```

## Historical S4 no-tool smoke (superseded)

The current implementation, actual HumanEval task attempts, durable session
capture, deadline correction, and accounting failures are documented in
[S4 foundations](s4-foundations.md). Both S3 and S4 remain unaccepted.

No multi-hour S4 run was started. Because S3 is hard-blocked and documented,
the foundations smoke was run without starting RL training. A local nano-swe
task checkout, its prompt, and a trusted hidden-test directory **outside** the
checkout were provisioned. Start the production server with the S1 tool-parser
flags. The following is the concrete one-episode command that was executed:

```bash
export TASK_WORKSPACE="$PWD/.s1-cache/nano-swe/task"
export TASK_PROMPT_FILE="$PWD/.s1-cache/nano-swe/prompt.txt"
export TASK_HIDDEN_TESTS="$PWD/.s1-cache/nano-swe-hidden-tests"
PYTHONPATH=. .venv/bin/python - <<'PY'
import asyncio
import json
import os
from pathlib import Path
from transformers import AutoTokenizer
from nano_areal.agent.dsh import DshAgent
from nano_areal.engine.proxy import OpenAIProxyServer, OpenAIProxyWorkflow

async def main():
    workspace = Path(os.environ["TASK_WORKSPACE"]).resolve()
    hidden = Path(os.environ["TASK_HIDDEN_TESTS"]).resolve()
    if hidden.is_relative_to(workspace):
        raise ValueError("Hidden tests must be outside the task workspace")
    tokenizer = AutoTokenizer.from_pretrained(
        ".s1-cache/models/Qwen3-VL-4B-Instruct-4bit", local_files_only=True)
    proxy = OpenAIProxyServer("http://127.0.0.1:8000/v1", tokenizer,
        model_name="mlx-community/Qwen3-VL-4B-Instruct-4bit")
    agent = DshAgent(model_name=proxy.model_name,
        patches=(str(Path("scripts/dsh_patch.yml").resolve()),))
    await proxy.start()
    try:
        trajectories = await OpenAIProxyWorkflow(agent, proxy).arun_episode({
            "prompt": Path(os.environ["TASK_PROMPT_FILE"]).read_text(),
            "workspace": str(workspace), "hidden_test_dir": str(hidden)})
        print(json.dumps({"result": agent.last_result,
            "model_calls": len(trajectories),
            "response_tokens": [int(item.response_mask.sum()) for item in trajectories]}))
    finally:
        await proxy.stop()

asyncio.run(main())
PY
```

The S4 smoke used `.s1-cache/nano-swe/task` and the external
`.s1-cache/nano-swe-hidden-tests` directory. It completed one production dsh
model call with `577` prompt tokens and `69` response tokens, but made no tool
call and received hidden-test reward `0.0`. Evidence is in
`artifacts/s3/s4-one-episode.json` and `artifacts/s3/s4-one-episode.log`.
This is only a foundation smoke, not S4 acceptance. Before claiming S4,
persist/reconcile dsh session logs against each proxy interaction, require
real tool calls and multiple turns, classify timeouts/failures, count
task/test denominators, and run the actual nano-swe hidden suite. The current
agent deleted temporary dsh homes at that milestone; opt-in capture is now
implemented in the S4 follow-up. Withheld host-side tests are not an OS sandbox. Production
adapter/version synchronization for training also remains unverified; do not
substitute the completed digit bandit for these gates.
