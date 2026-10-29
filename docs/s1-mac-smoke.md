# S1 Mac probe notes

## Status: partial, not accepted

Measured on **2026-10-29**, on Mac.lan: Apple Silicon M4, 32 GiB unified
memory, macOS 26.6.2. No CUDA device is required. No user workspace or cache
layout was moved, and no full model weights were downloaded.

| Check | Result |
| --- | --- |
| GRPO contracts | CPU and MPS pass: independent behavior/reference logprobs, finite singleton advantages, explicit problem groups |
| Regression suite | **35 tests pass**, including the original six smoke checks, the remote S1 suite, and new loss, wiring, buffer, and probe tests |
| Sync filtering | Zero means exactly the current policy; stale and future trajectories are rejected |
| dsh subprocess | `deepseek-harness-sdk==0.2.1rc2` and matching runtime, `sdk-minimal`, real subprocess + local mock `/v1/messages` SSE pass |
| dsh prompt | **580 tokens**, including three tool schemas, using the local Qwen tokenizer after the production Messages-to-OpenAI conversion; backend template parity is not yet verified |
| Single-image preprocessing | Synthetic 64x64 red PNG: **64 image tokens**, grid `[1, 16, 16]`, **84 total prompt tokens** |
| Real text/image completions | Blocked: no inference server; loopback requests fail with `URLError` |
| Inference memory / logprobs | Not measured; no inference model was loaded |
| bf16 + LoRA co-residency | Not measured; do not interpret processor-only measurements as training feasibility |
| L1 context | **8192 candidate only**, not validated; do not promote to 16384 yet |
| Reasoning parser | Not verified; the Instruct checkpoint does not establish a reasoning-parser name |

The local processor snapshot is `Qwen/Qwen3-VL-4B-Instruct` revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`. Processor measurements do not
establish that vllm-metal applies the same template, resizing, or token accounting.

## Environment blockers

- The pre-existing virtual environment points at a missing Python 3.13 binary.
  The working system interpreter is Python 3.14.2, outside the project's
  `>=3.12,<3.14` requirement. A repository-local diagnostic `.venv` was created
  with `uv`; this is **not** a validated project training environment.
- `uv venv --python 3.12`, with repository-local cache/install directories,
  timed out downloading the Python standalone archive from GitHub after retries.
  A direct `uv python install 3.12` retry also stalled and was stopped.
- A dry-run install of `vllm-metal` on this Python reports no matching wheel:
  the PyPI candidate is 0.1.0 with a cp312 wheel, not cp314. Do not assume the
  PyPI release contains current upstream Qwen3-VL support.
- Initial free disk space was about **5.6 GiB**; about **5 GiB** remained after
  the first diagnostic packages, falling to **4.5 GiB** after the final probes.
  Initial swap usage was about **4.2 GB**. Installing both
  inference and bf16 training checkpoints is not safe with this disk budget.
  Reclaim disk space explicitly before retrying; these probes never delete caches.
- dsh initially timed out when inheriting the normal home. An isolated child
  `HOME` and `XDG_CACHE_HOME`, in addition to `dsh_home`, resolved startup. The
  probe now always uses a disposable task directory and disposable home.

Diagnostic packages included torch 2.15.0 (MPS available), transformers 5.21.0,
torchvision 0.29.2, and pillow 13.0.0. Passing tests on this diagnostic interpreter
does not replace a rerun on supported Python 3.12.

## Reproduce the completed checks

From the repository root, with the diagnostic environment already installed:

```bash
PYTHONPATH=. .venv/bin/python -m pytest scripts -q
PYTHONPATH=. .venv/bin/python scripts/probe_grpo_parity.py --device cpu
PYTHONPATH=. .venv/bin/python scripts/probe_grpo_parity.py \
  --device mps --output artifacts/s1/grpo-parity-mps.json

PROCESSOR="$PWD/.s1-cache/hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/probe_dsh.py \
  --processor-path "$PROCESSOR"
HF_HUB_OFFLINE=1 .venv/bin/python scripts/probe_mac_vlm.py \
  --backend metal --sync --timeout 2 --processor-path "$PROCESSOR"
```

The last command currently exits **2**, with a structured blocked report.
Reports are stored under ignored `artifacts/s1/`. HTTP error bodies, credentials,
raw dsh requests, and session logs are not included in the reports. The dsh
mock returns fixed text, not real model inference. Its SSE usage counters are
zero; the reported prompt-token count comes from the local tokenizer instead.

## Resume real inference

First provide enough disk headroom for both checkpoints and package caches
(budget at least 20 GiB as a conservative installation allowance, not a measured
model requirement). Keep all new caches local to this repository:

```bash
export UV_CACHE_DIR="$PWD/.s1-cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.s1-cache/python"
export HF_HOME="$PWD/.s1-cache/hf"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
uv python install 3.12
uv venv --python 3.12 .s1-cache/venv
uv pip install --python .s1-cache/venv/bin/python \
  -r scripts/requirements-s1.txt torch numpy transformers pillow torchvision
```

Install a Qwen3-VL-capable vllm-metal build following its official installation
instructions, record the actual build/version, and check its `vllm serve --help`.
The current upstream uses `vllm serve`, not the legacy `vllm-metal --model` launch
command in the v0.1 rollout engine. The following is a **retry command, not a
verified launch on this machine**:

```bash
vllm serve mlx-community/Qwen3-VL-4B-Instruct-4bit \
  --host 127.0.0.1 --port 8000 --max-model-len 8192 --max-num-seqs 1 &
SERVER_PID=$!
.s1-cache/venv/bin/python scripts/probe_mac_vlm.py \
  --backend metal --sync --server-pid "$SERVER_PID" --processor-path "$PROCESSOR"
PYTHONPATH=. .s1-cache/venv/bin/python scripts/probe_dsh.py \
  --tokenizer-url http://127.0.0.1:8000 --processor-path "$PROCESSOR"
```

`PROCESSOR` must reference a local processor snapshot. On a clean machine,
download only processor/tokenizer/config files first; no training weights are
needed for token accounting. The Mac probe samples server RSS every 250 ms and
captures VM/swap counters. RSS is not total unified-memory usage or an exact GPU
peak. Capture idle server RSS after model load, then compare chat samples.

Acceptance still requires successful real single-image completion, finite
per-token logprobs matching completion usage, reconciliation of processor and
backend image-token counts, and observed co-residency with
`Qwen/Qwen3-VL-4B-Instruct` bf16 + language-only LoRA, frozen ViT. Run that
residency experiment only after the disk/memory blockers are removed. Do not
load a second full frozen reference model as a claimed LoRA memory optimization.

`--backend cuda` remains available on the probe as a future local-server hook;
it does not claim CUDA was tested. All default probe commands use Metal + sync.

## Algorithm alignment boundary

The S1 patch separates behavior-policy ratios from frozen-reference k3 KL,
normalizes rewards per problem group using population standard deviation, and
gives singleton/constant groups zero policy advantage. It carries captured
behavior logprobs through padding and causal shifting, rejects missing token
logprobs instead of silently filling response positions with zero, captures the
rollout version before generation, and preserves sync gap zero in CLI defaults.

This is deterministic contract coverage, **not numeric parity with a pinned
upstream AReaL checkout**, nor the S2 text/image archon parity test. The existing
token-mean reduction is retained. Buffer batching can still split groups;
episode-level/turn-level credit assignment and version-consistent asynchronous
weight updates remain S4/S5 work. The remote commit's Messages adapter, dsh agent,
truncation tracking, and loss switches are preserved, but their live-model
end-to-end validation is still pending. The training entry point is not a
functioning Qwen3-VL LoRA trainer;
do not use it as evidence that S2/S3/S4 are complete.

Official references consulted: vllm-project/vllm-metal's README and
`docs/supported_models.md`, and deepseek-ai/deepseek-harness's Python SDK and
`sdk-minimal` profile documentation. The pinned dsh patch was exercised by the
subprocess smoke, rather than assuming compatibility with newer SDK releases.
After integrating remote commit `75ca256`, the probe was upgraded from the
initial 0.1.5rc1 Chat Completions experiment to pinned 0.2.1rc2 Messages, using
the repository's production request conversion and SSE encoder. The earlier
324-token result is superseded by the 580-token measurement above.
