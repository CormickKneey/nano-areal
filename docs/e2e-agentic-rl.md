# Real-model dsh / Metal GRPO smoke

> October 30 follow-up: production vllm-metal HTTP is now verified and the
> custom archon passes real bf16 text/image parity. See `s1-mac-smoke.md`,
> `s2-archon.md`, and `s3-multimodal.md` for current status. The measurements
> and remaining-work list here describe the historical October 29 bandit;
> they are not S3/S4 acceptance and were not rerun for the later milestones.

## Result and acceptance boundary

Measured on **2026-10-29**, on the Mac mini M4 with 32 GiB unified memory,
macOS 26.6.2, Python 3.12.16. **Three consecutive real-model GRPO updates
completed**, with 24 real `deepseek-harness` `sdk-minimal` episodes. This is
the minimal text-only hidden-test bandit, **not nano-swe or coding-agent
benchmark completion**. No mock response or fabricated behavior logprob is
used by the runner.

The complete minimum acceptance chain is:

```text
real dsh sdk-minimal subprocess, isolated task directory
  -> /v1/messages (session-authenticated nano-areal proxy)
  -> /v1/chat/completions (loopback MLX-VLM fallback)
  -> exact prompt/generated IDs + real per-token behavior logprobs
  -> external hidden verifier -> reward -> version-zero-gap buffer
  -> bf16 Qwen3-VL + language-only LoRA, GRPO backward + AdamW
  -> atomic adapter export -> MLX adapter load -> next policy version
```

**Acceptance rule:** model-backed dsh episodes, captured token IDs/logprobs,
nonzero adapter updates, and reproducible commands are required. A mock-only
roundtrip, zero-advantage group, missing-logprob substitution, or supervised
cross-entropy step does not count. A blocked run still preserves completed
increments and reports the actual blocker; it is not promoted to full success.

| Area | State | Evidence / limit |
| --- | --- | --- |
| Real model via dsh and Messages proxy | Pass | 24 completed episodes, one LLM call each |
| Token/logprob accounting | Pass on fallback | Backend prompt IDs equal proxy template including tool schemas; generated IDs include stop; no retokenization of generated output |
| Buffer / policy version | Pass | 24 produced, 24 consumed, 0 discarded; gap 0; backend version checked on every model response |
| Real GRPO updates | Pass | Three backward/AdamW updates, finite gradients and nonzero adapter deltas |
| Weight feedback | Pass | 144 adapter tensors loaded per update, versions 1, 2, 3; subsequent episodes use the advanced version |
| Training model | Pass on HF bridge | bf16 `Qwen/Qwen3-VL-4B-Instruct`, MPS, frozen ViT, rank-8 language attention q/v LoRA |
| Custom archon S2 port | Not complete | HF-backed bridge intentionally replaces the unfinished custom architecture for this smoke; no custom M-RoPE/ViT parity claimed |
| Production vllm-metal HTTP | Still blocked | Matching vLLM wheel was incomplete in S1; this run uses MLX-VLM, not vllm-metal |
| Coding tools / nano-swe / multi-turn | Not accepted | This task explicitly requests no tools; fallback rejects generated tool calls rather than pretending to execute them |
| Images / long context | Not accepted here | Text only, approximately 574 tokens total; 8192-token training still unvalidated |
| CUDA | Not tested | Existing `--backend cuda` hook retained |
| Regression suite | Pass | 54 tests; targeted Ruff checks pass for the new bridge, server, verifier and runner |

The generic `train.py --backend metal --sync --agent dsh` path now fails early
with the equivalent runnable command below, instead of attempting the wrong
custom architecture or falling back to random weights. It is **not** claimed
to be the fully implemented production entry point.

## Reproduce

Run from `/Volumes/workspace/github/nano-areal`. Reuse the existing `.venv`
and the two verified local checkpoints described in `docs/s1-mac-smoke.md`.
No downloads are needed for the smoke.

```bash
export HF_HOME="$PWD/.s1-cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export UV_CACHE_DIR="$PWD/.s1-cache/uv"
export TMPDIR="$PWD/.s1-cache/tmp"
mkdir -p "$TMPDIR" artifacts/s1

PYTHONPATH=. .venv/bin/python scripts/run_agentic_rl.py \
  --backend metal --sync --agent dsh --loss grpo \
  --infer-model .s1-cache/models/Qwen3-VL-4B-Instruct-4bit \
  --train-model .s1-cache/models/Qwen3-VL-4B-Instruct \
  --steps 3 --group-size 8 --max-tokens 16 --max-context 2048 \
  --lr 1e-4 --output artifacts/s1/e2e-full \
  > artifacts/s1/e2e-full.log 2>&1

PYTHONPATH=. .venv/bin/python -m pytest -q
.venv/bin/python -m ruff check \
  nano_areal/archon/qwen3vl_lora.py nano_areal/engine/mlx_server.py \
  nano_areal/reward/dsh_task.py scripts/run_agentic_rl.py \
  scripts/test_agentic_rl.py
```

For one update, use `--steps 1` and a fresh output directory. The process starts
and stops its own inference server (port 8010) and proxy (8011); override
`--infer-port` / `--proxy-port` if occupied. Do not run two copies on the same
ports. Output directories are run-local, not resumable training checkpoints.

Dependencies are the existing S1 environment: MLX 0.32.4, MLX-VLM 0.7.11,
torch 2.15.0, transformers 5.21.1, PEFT 0.21.2, and matching dsh SDK/runtime
0.2.1rc2. To provision a separate environment with the same direct pins, use
`uv venv --python /opt/homebrew/bin/python3.12 .s1-cache/venv-e2e`, followed by
`uv pip install --python .s1-cache/venv-e2e/bin/python -r scripts/requirements-s1-mlx.txt -e .`.
Do not install the optional `metal` extra for this fallback: that selects the
still-blocked production vllm-metal dependency. Transitive packages are not
fully locked by this requirements file.

`mx.random.seed(42)` and `torch.manual_seed(42)` are set. Generation is still
stochastic, and temporary workspace paths can affect dsh prompts. Identical
rewards/losses across reruns are **not** promised. A constant-reward group
exits with an explicit error before claiming an update; increase
`--group-size` if necessary. Do not alter individual rewards or manufacture
advantages to obtain a passing run.

## Measurements

Raw report: `artifacts/s1/e2e-full/report.json`; server log:
`artifacts/s1/e2e-full/mlx-server.log`. These paths are ignored by Git.
The retained three-step run has group size 8, 572 prompt tokens and 2 response
tokens (digit + stop) per episode, 2,949,120 trainable parameters, and one
microbatch per trajectory. Prompt lengths are measurements, not constants.

| Step | Rewards | GRPO loss | Grad norm before clipping | Adapter delta L1 | Seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 0,0,1,0,0,0,0,0 | 0.11461282 | 0.88511628 | 147.33540 | 72.73 |
| 2 | 0,0,0,1,0,0,0,0 | -0.06532994 | 0.07764415 | 195.47167 | 69.86 |
| 3 | 0,1,0,0,0,0,0,1 | 0.06960384 | 2.49024224 | 192.30356 | 68.84 |

The sum of these step times is approximately 211.43 seconds, including
rollouts and adapter synchronization and, for step 1, lazy trainer loading;
it excludes initial inference-server startup. Gradients are clipped at norm 1.
The reward means (0.125, 0.125, 0.25) are not evidence of statistically
significant learning on such a small, artificial task.

| Memory counter | Recorded value |
| --- | ---: |
| MLX active after adapter synchronization | 3,102,486,576 bytes |
| MLX allocator peak | 3,886,318,180 bytes, about 3.62 GiB |
| MPS driver after step 1 | 10,104,160,256 bytes, about 9.41 GiB |
| MPS driver after steps 2/3 | 10,112,548,864 bytes, about 9.42 GiB |
| System swap at recorded boundaries | 0 bytes |

These are **different allocators and boundary counters**, not a measured
whole-machine training peak. Do not sum driver, RSS, and allocator counters.
The system also uses compression. Zero sampled swap does not establish a
long-sequence activation budget or absence of transient memory pressure.

## Loss, reference and quantization caveats

The behavior logprobs come from float32-normalized MLX logits, never from
re-encoding the returned text. The terminal collector distinguishes the
repeated final chunk on length exhaustion from a newly emitted stop token.
A rounded zero logprob on a highly confident stop is a measured float32
result, not a substitute for an absent value. Empty, misaligned, nonfinite,
or positive behavior logprobs are rejected.

Sampling uses **temperature 1, top-p 1, top-k 0**, overriding checkpoint top-k
defaults so that the reported untempered logprobs describe the actual sampling
distribution. Other temperature/top-p settings are rejected by this fallback.

The GRPO ratio is `exp(current_bf16_logprob - captured_behavior_logprob)`,
clipped at 0.2. The reference is the original frozen bf16 base, evaluated
with the same model's adapters disabled, without a second resident copy;
the k3 KL coefficient is 0.01. Complete-group advantages are computed before
microbatching and the loss is weighted by response-token count. A regression
test compares microbatch gradients with the full-group gradient.

**4bit inference is not numerically the same policy as bf16 training.**
The measured mean absolute behavior/current logprob differences across all
response tokens were **1.45561, 1.19470, 0.83775 nats**, and mean ratios were
**0.52842, 1.30565, 0.61341**. Stop tokens with near-zero error are included
in these means. The drift is substantial and is reported, not silently
replaced by bf16 recomputation. Quantization, backend kernels and checkpoint
provenance need further parity investigation; the S1 ModelScope manifests
do not establish independent HF revision equality. Version-zero-gap
synchronization prevents stale adapters, but does **not** make this an exact
on-policy or cross-framework numeric-parity demonstration.

PEFT A/B matrices are transposed into MLX `lora_a`/`lora_b`, using scale
`alpha/r = 16/8 = 2`. Adapter names are restricted to language attention q/v
projections. A small real MLX regression test validates the transpose/scale;
it is not a full Qwen3-VL archon text/image parity test.

## Task and security boundary

The prompt asks for a random digit from 0 through 9, without tools. Trusted
external tests award 1 for exactly one digit in 5 through 9, otherwise 0.
The target is absent from the prompt. The verifier reads the final dsh
response **after** the subprocess exits; it cannot rewrite the model output.
Only `completed` dsh episodes are eligible; truncation is rejected by this
minimal runner. There are no preset rewards in this path.

Each episode has a fresh workspace and dsh home under repository-local
`.s1-cache/tmp/`; hidden tests are outside that workspace. Tests are withheld
from normal workspace contents, **not protected by an OS sandbox**. A
malicious tool-using process could inspect sibling paths, so this is not
adequate isolation for adversarial coding evaluation. Trusted verifier
scripts execute on the host and must not be untrusted task submissions.

Both servers bind loopback only. The runner generates an ephemeral adapter
admin key passed through the child environment, not argv or reports; no
external model API key is needed. The unchanged dsh patch only directs its
provider's base URL to the local proxy. It does not replace or edit the
`sdk-minimal` agent. Raw prompts/session logs and weight files are not
committed. The latest runner also saves ignored token-level
`rollout-step-N.json` replay artifacts, without session credentials.

## Remaining work

- Finish the custom archon Qwen3-VL state mapping, interleaved M-RoPE,
  ViT/DeepStack and LoRA port; add the planned real text + image
  `test_archon_parity.py`. The current HF bridge does not finish S2.
- Restore production vllm-metal HTTP once the matching core wheel is usable;
  verify exact backend token IDs, tool schemas, and behavior logprobs there.
- Implement faithful tool-call parsing and multi-turn token accounting in
  the MLX fallback or production backend, then run an actual coding task
  with isolated hidden tests / nano-swe. Do not present this bandit as S4
  coding-agent acceptance.
- Investigate the measured quantization/backend/checkpoint policy drift
  before interpreting loss curves or choosing a longer training run.
- Measure longer sequences, image-bearing trajectories, and whole-machine
  memory pressure before accepting an 8192-token budget. CUDA remains optional.

Initial debugging failures are preserved locally: the first attempt exposed
transformers returning `BatchEncoding` unless `return_dict=False`; the
second used a binary-choice prompt with no reward spread and correctly
claimed zero updates. A subsequent one-step run passed, followed by the
three-step run reported here after explicitly disabling top-k truncation.

A separate final-code replay (`artifacts/s1/e2e-replay/report.json`) also
completed one real update: loss 0.10898121, gradient norm 1.11451137,
72.12 seconds, bf16 base and frozen ViT verified. Its ignored
`rollout-step-1.json` contains the captured token-level replay data.
