# S2 native Qwen3-VL: October 30, 2026

## Acceptance: passed for text and one still image

The native implementation in `nano_areal/archon/qwen3vl.py` runs without calling
Transformers model forward methods. It reuses archon's dense attention/MLP
blocks, with float32 RMS normalization, and implements interleaved T/H/W
M-RoPE, the Conv3d patch embed, learned bilinear vision positions, axial vision
RoPE, full noncausal ViT attention, spatial merging and DeepStack injection.
The three 4B DeepStack taps (5/11/17) are injected after language layers 0/1/2.
The tied output projection reuses the language embedding weight.

The strict safetensors loader instantiates on meta, assigns one shard at a
time, checks every key and shape, and rejects missing/unknown/duplicate keys.
It does not load a second resident reference model for training. Bidirectional
state-key mapping and a small checkpoint roundtrip are regression-tested.

LoRA freezes the entire base, including ViT, and installs float32 rank-8,
alpha-16 q/v adapters. Adapter disabling is nest-safe for reference-policy
evaluation. Export transposes A/B to the existing MLX bridge's format and
requires its scale of 2. `NativeQwen3VLLoRATrainer` reuses the established GRPO
group/microbatch loss, but evaluates native archon logits. The existing bandit
runner exposes `--trainer-implementation archon`; the historical HF bridge is
still its default, and the already-accepted bandit was not rerun or re-scored.

## Real Mac measurements

Mac mini M4 / 32 GiB, Homebrew Python 3.12.16, torch 2.15.0,
transformers 5.21.1. Local bf16 `Qwen3-VL-4B-Instruct` checkpoint from S1.
HF and native models are loaded **sequentially**, never simultaneously. The
production inference server is stopped during this probe.

| Check | Result |
| --- | --- |
| Full text logits | 15 prompt tokens; **max/mean absolute error 0 / 0** |
| Full single-image logits | 64x64 red image, 81 prompt tokens; **max/mean absolute error 0 / 0** |
| Tolerance | bf16 outputs compared in float32, **atol 0.02, rtol 0**; no relaxed tolerance or selected-vocabulary comparison |
| M-RoPE positions | Native text/image positions equal independently calculated HF positions exactly |
| Tiny randomized contract | CPU fp32 text and rectangular 4x6 patch-grid image match HF; checkpoint and adapter contracts pass |
| Short MPS step 1 | Text, loss 20.81250381, gradient norm 58.11257172 |
| Short MPS step 2 | Image, loss 17.31250381, gradient norm 74.78968048 |
| Short MPS step 3 | Text, loss 17.06252289, gradient norm 65.53650665 |
| Native load driver snapshot | 9,868,886,016 bytes |
| Largest step-boundary driver snapshot | 9,944,383,488 bytes (about 9.26 GiB), not a sampled whole-machine peak |
| Swap | Existing S1 swap stayed 832.81 MiB at all sampled S2 boundaries; no increase |
| Runtime | 8.35 seconds for sequential reference/native parity and 3 steps |
| Regression | 62 tests pass after this milestone |

The step target is a short teacher-forced next token, alternating text/image,
not CLEVR evaluation, BFCL scoring, or an RL quality claim. These measurements
do not qualify 8192-token training or concurrent production inference.

### Numeric debugging resolved, not hidden

An initial image run failed (max error 18.8984375) while text was already
exact. Layer probes isolated two bf16-sensitive differences: reordered
bilinear interpolation arithmetic and constructing vision inverse frequencies
on MPS instead of float32 CPU before transfer. Matching the checkpoint's
reference arithmetic eliminated all measured vision and final-logit error.
The tolerance was not raised. Ignored intermediate logs preserve that failure.

## Reproduce

```bash
export HF_HOME="$PWD/.s1-cache/hf" HF_HUB_OFFLINE=1
export UV_CACHE_DIR="$PWD/.s1-cache/uv" TMPDIR="$PWD/.s1-cache/tmp"
export XDG_CACHE_HOME="$PWD/.s1-cache/xdg"
mkdir -p "$TMPDIR" artifacts/s2
PYTHONPATH=. .venv/bin/python -m pytest -q
PYTHONPATH=. .venv/bin/python scripts/test_archon_parity.py \
  --model-path .s1-cache/models/Qwen3-VL-4B-Instruct \
  --device mps --atol 0.02 --steps 3
```

Evidence stays local in `artifacts/s2/archon-parity.json` and
`artifacts/s2/archon-parity.log`. No weights or raw sessions are committed.

## Remaining boundaries

- Exactly one still image in one unpadded sample; video, packed/padded image
  batches and incremental decoding caches are deliberately rejected or absent.
- S3 must carry pixels/grid metadata through the proxy and exported trajectories;
  text-only fallback requests do not establish multimodal RL acceptance.
- A real local CLEVR L0 loss curve and unchanged BFCL behavior remain S3 gates.
- S4 still needs actual tools, nano-swe hidden tests, session/proxy accounting
  and a countable Mac sync pass rate. No multi-hour S4 training was started.
