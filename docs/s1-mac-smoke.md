# S1 Mac probe notes

> Follow-up on 2026-10-29: `docs/e2e-agentic-rl.md` records three real dsh / MLX
> HTTP / bf16-LoRA GRPO updates. That text-only bandit uses an explicit MLX
> fallback and HF trainer bridge; it does not resolve the production
> vllm-metal blocker or complete the custom archon / coding-agent milestones.
> The historical measurements below remain the original S1 results.

## Status: partial, real MLX smoke and LoRA residency measured

Measured on **2026-10-29**, Mac mini M4, 32 GiB unified memory, macOS 26.6.2.
This run supersedes the earlier processor-only smoke on the same date. Python
**3.12.16** from `/opt/homebrew/bin/python3.12` rebuilt `.venv`; Python 3.14 is
not used. All new environments, checkpoints, temporary files, and package caches
live in this repository on `/Volumes/workspace`, not on the internal disk.

| Check | Result |
| --- | --- |
| Regression suite | **42 tests pass** on Python 3.12, including seven new MLX token-accounting tests |
| GRPO contracts | CPU and MPS pass: behavior/reference separation, finite singleton advantages, explicit problem groups; not pinned upstream numeric parity |
| Sync filtering | Current policy only at gap zero; stale and future trajectories rejected |
| Agent | **deepseek-harness**, SDK/runtime `0.2.1rc2`, `sdk-minimal`, real subprocess with local mock `/v1/messages` SSE passes; no AReaL-Harness substitution |
| dsh prompt | **563 tokens** with the inference checkpoint tokenizer, two messages and three tool schemas; mock-backed, not model-backed dsh |
| Real text completion | Direct MLX on Metal: `OK`, 13 prompt tokens, 2 generated tokens including stop |
| Real single-image completion | Direct MLX: synthetic 64x64 red PNG recognized as `Red`; **64 image tokens**, **84 prompt tokens**, 2 generated tokens including stop |
| Token logprobs | Finite and nonpositive; generated token IDs/counts match MLX terminal accounting; image values `[-0.00002288818359375, 0.0]` |
| Image-token reconciliation | MLX processor input count equals MLX generation prompt count, 84; not vllm-metal HTTP parity |
| Inference memory | MLX active after load **3,095,408,648 bytes**; single-image allocator peak **3,378,837,352 bytes** (about 3.15 GiB) |
| bf16 + LoRA co-residency | **Pass for a tiny text step**: separate MPS training process alongside active 4bit MLX inference; all parameters on MPS, frozen ViT, language-only rank-8 LoRA |
| Concurrent inference | Inference continues returning `OK`: counter **119 -> 135** during the experiment; training holds its loaded model for another 15 seconds after the step |
| Training step | 10 tokens, forward/backward/AdamW step; loss **3.5504846573**, finite adapter gradients; 2,949,120 trainable parameters in 144 tensors |
| Training memory | MPS driver snapshot **8,901,197,824 bytes** after load, **9,004,204,032 bytes** after step; **zero swap** at sampled boundaries |
| vllm-metal HTTP | **Blocked**: plugin 0.32.0 installed, matching vLLM core wheel download times out / remains incomplete; `vllm` module absent |
| L1 context | **8192 remains a candidate**, not validated by tiny requests/10-token training; do not promote to 16384 |
| Reasoning parser | Not verified; no parser name inferred from this Instruct checkpoint |

**S1 is not fully accepted.** Direct MLX proves real inference and small LoRA
feasibility, not the production vllm-metal HTTP/logprobs route, a model-backed dsh
rollout, an 8192-token training workload, or S2/S3/S4 completion. The training
entry point is still not a functioning Qwen3-VL LoRA trainer. `--backend cuda`
remains a hook on `probe_mac_vlm.py`; CUDA was not tested.

## Checkpoints and environment

The Hugging Face endpoint timed out. Both same-named checkpoints were downloaded
from ModelScope into `.s1-cache/models/`, with every file checked against the
SHA256 and size in a retained `s1-download-manifest.json`. This verifies the
mirror manifest, not independent equality to a particular HF commit.

- Inference: `mlx-community/Qwen3-VL-4B-Instruct-4bit`, weight file
  3,093,767,283 bytes; SHA256
  `90eeb02604181dbcccd0a30a1f550a4a8928ca7dcbee4aee1449239306cfdfca`.
- Training: `Qwen/Qwen3-VL-4B-Instruct`, bf16 shards 4,967,229,296 and
  3,908,490,048 bytes; SHA256 respectively
  `30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9` and
  `046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6`.
- Working direct-MLX environment: MLX 0.32.4, mlx-vlm 0.7.11, torch 2.15.0,
  transformers 5.21.1, PEFT 0.21.2, accelerate 1.15.0. Direct dependencies are
  recorded in `scripts/requirements-s1-mlx.txt`; transitive dependencies are not
  fully locked by this requirements file.
- The earlier local HF processor snapshot remains available at revision
  `ebb281ec70b05090aa6165b016eac8ec08e71b17`. Its separate dsh run counted 565
  tokens. Temporary workspace paths can affect prompt tokenization; these two
  runs are not evidence of identical templates. The earlier 580-token result is
  superseded, not a fixed token-budget constant.

## Reproduce the passing path

Run from the repository root. `uv venv --clear` replaces the existing environment.
The downloader reuses its manifest, verifies existing files, and only renames a
new `.partial` download after SHA256/size verification. An interrupted individual
file is restarted, not range-resumed.

```bash
export UV_CACHE_DIR="$PWD/.s1-cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.s1-cache/python"
export TMPDIR="$PWD/.s1-cache/tmp"
export HF_HOME="$PWD/.s1-cache/hf"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
mkdir -p "$TMPDIR" artifacts/s1
uv venv --clear --python /opt/homebrew/bin/python3.12 .venv
uv pip install --python .venv/bin/python -r scripts/requirements-s1-mlx.txt
.venv/bin/python scripts/download_s1_model.py --model infer \
  --output .s1-cache/models/Qwen3-VL-4B-Instruct-4bit
.venv/bin/python scripts/download_s1_model.py --model train \
  --output .s1-cache/models/Qwen3-VL-4B-Instruct
export HF_HUB_OFFLINE=1
PYTHONPATH=. .venv/bin/python -m pytest -q
PYTHONPATH=. .venv/bin/python scripts/probe_grpo_parity.py --device cpu \
  --output artifacts/s1/grpo-parity-cpu.json
PYTHONPATH=. .venv/bin/python scripts/probe_grpo_parity.py --device mps \
  --output artifacts/s1/grpo-parity-mps.json
PYTHONPATH=. .venv/bin/python scripts/probe_dsh.py \
  --processor-path .s1-cache/models/Qwen3-VL-4B-Instruct-4bit \
  --output artifacts/s1/dsh-probe-mlx-tokenizer.json
.venv/bin/python scripts/probe_mlx_vlm.py --backend metal --sync \
  --model-path .s1-cache/models/Qwen3-VL-4B-Instruct-4bit \
  --hold-seconds 180 > artifacts/s1/mlx-probe.stdout.json \
  2> artifacts/s1/mlx-probe.stderr.log &
INFER_PID=$!
for attempt in $(seq 1 60); do
  .venv/bin/python -c 'import json,sys; report=json.load(open("artifacts/s1/mlx-probe.json")); sys.exit(not (report["pid"] == int(sys.argv[1]) and report.get("hold", {}).get("completed_requests", 0) > 0))' \
    "$INFER_PID" 2>/dev/null && break
  kill -0 "$INFER_PID" 2>/dev/null || break
  sleep 1
done
.venv/bin/python scripts/probe_lora_residency.py \
  --model-path .s1-cache/models/Qwen3-VL-4B-Instruct \
  --inference-pid "$INFER_PID" --inference-report artifacts/s1/mlx-probe.json \
  --step --hold-seconds 15
wait "$INFER_PID"
```

`--hold-seconds` on the MLX probe keeps issuing greedy text requests once per
second, rather than merely sleeping with a potentially paged-out checkpoint.
The residency probe checks a live process and an advancing successful inference
counter. Its step targets only language attention `q_proj`/`v_proj`; it loads no
second frozen reference model and performs no weight export/sync.

MLX logprobs are normalized from float32-cast logits to avoid bf16 rounding in
log-sum-exp; confident tokens can still have a rounded logprob of zero. No zeros
are substituted for missing logprobs. The terminal MLX chunk repeats the last
token on length exhaustion but introduces the stop token on normal termination;
the collector distinguishes these cases. These are MLX generation counts, not
a claim about an OpenAI server's EOS accounting.

Raw reports/logs stay under ignored `artifacts/s1/`. No API key, raw dsh request,
or dsh session log is added to Git. The dsh mock returns fixed text and zero SSE
usage counters; its prompt count is measured separately by the tokenizer.

## Memory interpretation

MLX allocator counters, MPS driver counters, RSS, and system VM/swap snapshots
are distinct measurements. Do not add RSS and allocator numbers into a purported
unified-memory total. Mapped weights and compression can make RSS much smaller
than GPU driver allocation. MPS values are boundary snapshots, **not a measured
training peak**. The MLX image peak is an allocator peak, not whole-machine peak
usage. Zero sampled swap does not establish an 8192-token activation/KV budget.

## Remaining vllm-metal blocker and retry

An isolated Python 3.12 environment `.s1-cache/venv-metal` installed the official
`vllm-metal==0.32.0` cp312 arm64 release and its pinned MLX-LM dependency. The
existing Git proxy pointed at unavailable `127.0.0.1:7897`; only the installation
command's environment overrode it, without changing global Git configuration.
Release metadata requires matching vLLM **0.32.0** core. Direct GitHub wheel
requests repeatedly timed out; the API asset retry transferred only part of the
29,930,964-byte wheel before timing out. The partial file fails ZIP validation.
No working `vllm serve` process was claimed.

Retry with repository-local caches and the same Python:

```bash
uv venv --python /opt/homebrew/bin/python3.12 .s1-cache/venv-metal
GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=http.proxy GIT_CONFIG_VALUE_0= \
  GIT_CONFIG_KEY_1=https.proxy GIT_CONFIG_VALUE_1= \
  uv pip install --python .s1-cache/venv-metal/bin/python \
  'https://github.com/vllm-project/vllm-metal/releases/download/v0.32.0/vllm_metal-0.32.0-cp312-cp312-macosx_15_0_arm64.whl' \
  'https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0%2Bcpu-cp38-abi3-macosx_11_0_arm64.whl'
.s1-cache/venv-metal/bin/vllm serve --help
# Verify these flags against the installed help before launching:
.s1-cache/venv-metal/bin/vllm serve \
  "$PWD/.s1-cache/models/Qwen3-VL-4B-Instruct-4bit" \
  --served-model-name mlx-community/Qwen3-VL-4B-Instruct-4bit \
  --host 127.0.0.1 --port 8000 --max-model-len 8192 --max-num-seqs 1
```

This launch remains **unverified**. Once healthy, run `probe_mac_vlm.py` with
`--backend metal --sync --server-pid SERVER_PID` and the local processor path,
compare image and dsh token accounting through the backend, and rerun residency
against the server PID (omit `--inference-report` for a server without the direct
MLX hold report). Then measure an actual L1-length workload before selecting its
context budget. Deterministic GRPO contracts are not upstream numeric parity;
episode credit and asynchronous updates remain later-stage work.
