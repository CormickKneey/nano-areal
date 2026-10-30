# S3 multimodal foundation: October 30, 2026

## Status: partial, not S3 acceptance

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

## CLEVR L0 entry point and remaining gate

`nano_areal/clevr.py` reads official local train/val question files and images;
it checks required fields and rejects image paths escaping the split directory.
`scripts/train_clevr_l0.py` runs a bounded, frozen-ViT native LoRA supervised
overfit probe and requires loss on the same examples to decrease. Only answer
tokens are supervised, and the default context ceiling is 1024 tokens. This
is a loss-drop probe, not held-out accuracy or GRPO.

**Not measured:** no official CLEVR dataset is present in this repository.
The attempted command below failed before model loading with
`FileNotFoundError` for
`.s1-cache/datasets/CLEVR_v1.0/questions/CLEVR_train_questions.json`.
No synthetic fixture has been presented as a CLEVR training result.

After placing the official dataset under the repository-local cache:

```bash
export HF_HOME="$PWD/.s1-cache/hf" HF_HUB_OFFLINE=1
export UV_CACHE_DIR="$PWD/.s1-cache/uv" TMPDIR="$PWD/.s1-cache/tmp"
export XDG_CACHE_HOME="$PWD/.s1-cache/xdg"
PYTHONPATH=. .venv/bin/python scripts/train_clevr_l0.py \
  --data-root .s1-cache/datasets/CLEVR_v1.0 \
  --model-path .s1-cache/models/Qwen3-VL-4B-Instruct \
  --device mps --samples 1 --steps 5 --max-context 1024
```

Run this **without a resident inference server** first. A passing real CLEVR
loss curve and a model-backed BFCL baseline comparison are still required to
close S3. The local regression fixtures do not replace either gate.

To reproduce production image transport, launch the S1 server with
`--kv-cache-memory-bytes 1342177280` in place of
`--gpu-memory-utilization 0.25`, then run:

```bash
PYTHONPATH=. .venv/bin/python scripts/probe_vllm_http.py --proxy-image \
  --output artifacts/s3/vllm-image-proxy.json
```

## S4 next command: one real tool task, not training

No multi-hour S4 run was started. Once S3 is accepted, prepare a local nano-swe
task checkout, its prompt, and a trusted hidden-test directory **outside** the
checkout. Start the production server with the S1 tool-parser flags. The
following is a concrete one-episode starting command, **not executed here**:

```bash
export TASK_WORKSPACE="$PWD/.s1-cache/nano-swe/task"
export TASK_PROMPT_FILE="$PWD/.s1-cache/nano-swe/prompt.txt"
export TASK_HIDDEN_TESTS="$PWD/.s1-cache/nano-swe/hidden-tests"
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

Before claiming S4, additionally persist/reconcile dsh session logs against
each proxy interaction, require real tool calls and multiple turns, classify
timeouts/failures, count task/test denominators, and run the actual nano-swe
hidden suite. The current agent deletes temporary dsh homes, so session-log
capture still needs implementation. Withheld host-side tests are not an OS
sandbox. Production adapter/version synchronization for training also remains
unverified; do not substitute the completed digit bandit for these gates.
