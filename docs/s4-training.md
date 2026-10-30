# S4 production policy preflight: October 30, 2026

## Continue5: isolated decode rounding, parity still blocked

The first remaining gate was investigated without starting production, adding
rollout starts, changing any policy manifest, or updating an adapter. Original
S1/S2/S3 and bounded S4 coding acceptance are unchanged. **Full S4 training is
not accepted; updates remain 0.** The bound stays **MAE <= 0.01**.

An isolated actual-checkpoint experiment identifies an input-dependent
rounding difference, not just a different dequantized weight. In the installed
MLX Metal `quantized.h`, the 4-bit `load_vector` bias sum adds groups of four
bf16 inputs before promotion to float32. The measured single-row quantized
path therefore differs from multiplying either bf16 or float32 dequantized
weights. A reference adds the bias-weighted difference between rounded
four-value sums and float32 sums to the float32 affine product.

`artifacts/s4/policy-aligned-audit/affine-isolation.json` uses seed 42,
bf16 standard-normal inputs, and the actual layer-0 tensors:

| Projection / input rows | bf16 dense MAE | float32 affine MAE | corrected MAE / max |
| --- | ---: | ---: | ---: |
| q_proj, 2560 -> 4096 / 1 | 0.00600267 | 0.00577891 | 0 / 0 |
| down_proj, 9728 -> 2560 / 1 | 0.01183670 | 0.01161732 | 0 / 0 |
| q_proj / 128 | 0 | 0.00126842 | 0.00647333 / 0.078125 |
| down_proj / 128 | 0 | 0.00253158 | 0.01303589 / 0.25 |

The correction is specific to the measured decode path: applying it to the
128-row prefill path **makes results worse**. It is not a universal replacement
for linear layers. `nano_areal/archon/s4_operators.py` is a diagnostic reference,
not installed by the trainer. A separate synthetic MLX comparison independently
checks the PyTorch formula and its finite/nonzero input gradients.

Whole-group counterfactuals still fail. All use the same untouched complete
HumanEval/0 group, **9 turns / 636 response tokens**, policy
`s4-policy-v0-d43b14bef7d62f10`, rewards [1,1], advantages [0,0]. Files below
are under `artifacts/s4/policy-aligned-audit/`; these are diagnostics, not newly
bound rollout preflights:

| Diagnostic file | MAE | Seconds |
| --- | ---: | ---: |
| decode-operator-diagnostic.json | 0.01241930111 | 66.46 |
| decode-rms-diagnostic.json | 0.01419322152 | 68.62 |
| decode-rms-rope-diagnostic.json | 0.01281768727 | 70.18 |
| decode-fused-all-diagnostic.json | 0.01309490551 | 72.06 |
| dense-fused-all-diagnostic.json | 0.01323832762 | 47.15 |
| decode-prefill-head-diagnostic.json | 0.01241974193 | 65.46 |
| head-bias-rounding-diagnostic.json | 0.01090651186 | 40.68 |
| head-fp32-attention-diagnostic.json | 0.01197469242 | 41.93 |

The hook-based diagnostic separates generated rows from prompt rows, optionally
checks single-rounding RMSNorm, float32 RoPE, fused SwiGLU, and a dense first
prefill head. The final variant also isolates float32 SDPA accumulation;
it does not close parity. These are isolated hypotheses, **not proof of the production
prefill/decode/cache execution schedule**. The head-only correction still has
maximum token error **0.49179697**. None closes the group gate, and none is
enabled in existing production or training paths. The original bound
preflights **0.00486984 /53** and **0.01112227 sampled /0**, float32-head-only
**0.01057059**, and unaligned **0.10771345 / 0.10248081** remain unchanged.

Reproduce the final head-isolation counterfactual with a new output filename:

```bash
export HF_HOME="$PWD/.s1-cache/hf" HF_HUB_OFFLINE=1
export UV_CACHE_DIR="$PWD/.s1-cache/uv" TMPDIR="$PWD/.s1-cache/tmp"
export XDG_CACHE_HOME="$PWD/.s1-cache/xdg" PYTHONPATH=.
# Keep the owned production API AND engine stopped.
.venv/bin/python scripts/diagnose_s4_operators.py \
  --policy artifacts/s4/policy-aligned/policy-v0.json \
  --group artifacts/s4/policy-sampled-0 \
  --dense-projections --prefill-head \
  --output artifacts/s4/policy-aligned-audit/head-rounding-reproduce.json
.venv/bin/python -m pytest -q scripts/test_s4_operators.py \
  --basetemp=.s1-cache/tmp/pytest-operator-reproduce
```

The diagnostic refuses existing output files and live production listeners,
checks policy files and complete-group provenance, and records source/evidence
hashes, completion/error status, and per-turn errors. Diagnostic exit success
means the experiment completed, **not** that policy parity passed.

No additional sampling is unlocked: cumulative declared sampling stays
**24/49**, with **25 failures retained**. No informative complete group,
changed adapter, v1 feedback, or backward/optimizer/export memory result is
claimed. The next numerical investigation must account for actual production
prefill/decode segmentation, cache reuse, and attention arithmetic before
binding and measuring a new operator policy. Validation: **108 regression
tests pass**, targeted Ruff and `git diff --check` pass. No production server
was started during this continuation.

## Aligned-base continuation

`scripts/align_s4_base.py` exports the actual served affine 4-bit tensors using
MLX dequantization, restores HF key layout and Conv3D axes, and records source
and output hashes. It preserves all 713 tensors, dequantizing 253 language
modules including the tied embedding. The original checkpoints and S1/S2
loaders are unchanged. Conversion alone does not assert numerical parity.

The old captured group diagnostic has MAE **0.00516071** after alignment,
versus the retained **0.10248081** original-base result. This diagnostic is
not a bound-policy rollout. A fresh immutable v0 and fresh production
HumanEval/53 group (`artifacts/s4/policy-aligned-53`) pass **2/2** official
external pytest files with exact reconciliation. With serving stopped,
`policy-aligned/preflight.json` measures MAE **0.00486984**, below the unchanged
**0.01** bound, in **46.26 seconds**. Rewards **[1,1]** still give advantages
**[0,0]**; no update is performed. These two starts increase the retained
engineering denominator from **17/31** to **19/33**, not benchmark accuracy.

This passes deterministic /53, **not universal policy parity**. A second
sequential preflight over the temperature-0.8 /0 complete group
(`policy-aligned-audit/preflight.json`) measures **0.01112227 MAE** over
**9 turns / 636 response tokens**, maximum error **0.49179697**, in **52.84
seconds**. It **fails** the unchanged **0.01** bound and exits **2**; rewards
remain [1,1], advantages [0,0]. Sequence lengths are 1177–1898. Existing system
swap used is **3670.88 -> 3662.88 MiB**, free-memory percentages **92% -> 93%**.
These remain forward-only measurements, not backward peaks.

An unbound diagnostic uses float32 affine arithmetic only for the output head
against untouched captured probabilities. MAE improves only to **0.01057059**,
still failing. No manifest or production rollout is rewritten. Dequantized
bf16 dense operations and MLX fused quantized kernels have different precision
paths; full effective-operator alignment remains to investigate. No threshold
is loosened or successful group substituted for this failure.

Subsequent sampling was declared before each group, retaining the same prompt,
512-token turn cap, 120-second activity timeout, and unchanged external rewards:

| Evidence under artifacts/s4 | Task | Temperature | Passed/started | Training eligible | Blocking outcome |
| --- | --- | ---: | ---: | ---: | --- |
| policy-aligned-53 | HumanEval/53 | 0 | 2/2 | 2/2 | Rewards [1,1], zero advantages |
| policy-sampled-0 | HumanEval/0 | 0.8 | 2/2 | 2/2 | Rewards [1,1], zero advantages |
| policy-sampled-32 | HumanEval/32 | 0.8 | 0/2 | 1/2 | One verified failure, one max-tokens |
| policy-sampled-38 | HumanEval/38 | 1.0 | 0/4 | 3/4 | Three verified failures, one timeout |
| policy-sampled-91 | HumanEval/91 | 0.8 | 0/4 | 3/4 | Three verified failures, one timeout |
| policy-sampled-109 | HumanEval/109 | 0.8 | 3/4 | 3/4 | Three successes, one max-tokens |

This continuation adds **7/18**, for cumulative **24/49** engineering starts;
all **25 failures** remain. These selected diagnostic tasks are not HumanEval
accuracy or pass@k. The /109 group has apparent rewards [1,1,0,1], but its
max-tokens episode is incomplete: it cannot supply a valid negative advantage.
The complete successful groups still have zero advantage. Failed checks are
not relabeled, discounted turn rewards are not substituted, and episodes from
different groups/tasks are not pooled or cherry-picked. Every group has zero
backend-only completions; the two timeouts fail strict reconciliation.

Complete, reconciled external-test failures now export their original reward;
timeouts, truncations and accounting failures stay in the denominator but are
ineligible for updates. The coding report still requires success; a separate
training-eligibility field never converts a failed check into a coding pass.
No informative complete group was obtained. **Updates remain 0; no v1 adapter,
production feedback transition, or backward memory qualification is claimed.**

Reproduce conversion with the repository-local Metal Python environment:

```bash
PYTHONPATH=. .s1-cache/venv-metal/bin/python scripts/align_s4_base.py \
  --source .s1-cache/models/Qwen3-VL-4B-Instruct-4bit \
  --output .s1-cache/models/Qwen3-VL-4B-Instruct-aligned
```

Use the existing prepare/serve/probe/stop/preflight sequence below with the
aligned training path and fresh evidence directories. Never overwrite a
converted snapshot, edit an old manifest, or reuse captured probabilities as
though they came from another version. The preflight now checks requested
checkpoint paths/hashes against the binding and rejects an MAE bound above
0.01. Full training, nonzero export, feedback and backward memory remain
unaccepted pending subsequent measurements.

### Bounded update contract

The opt-in `update` phase rechecks the complete group and the unchanged 0.01
behavior bound in the same native process before backward. It performs exactly
one AdamW step, using one advantage per externally verified episode, repeated
across its turns. Loss is averaged over response tokens within each episode,
then equally over episodes. Turn discounts never create episode advantages.
Homogeneous groups exit without backward or an optimizer step. The reference
is the frozen aligned base with adapters disabled, not another resident model.

The update has a 600-second between-turn deadline, finite-gradient checks,
gradient clipping, and nonzero gradient/delta checks. It exports immutable
`adapter-v1`, checks an exact PEFT roundtrip and changed tensor-file SHA-256,
and binds a new `policy-v1.json` to its parent and complete-group file hashes.
Existing update evidence or v1 output is never overwritten. A live local
production listener blocks native loading. Production must subsequently restart
on v1; an exported adapter alone is not feedback acceptance.

Memory qualification requires sampled MPS driver usage within PyTorch's
recommended device budget and no increase in system swap used. Existing swap
is reported separately. A 100ms sampler records MPS allocated/driver peaks;
process RSS high-water and synchronized forward/backward/optimizer/export
snapshots are also retained. These are sampled, not allocator-exact peaks,
and only qualify the measured bounded sequences, never 8192-token training or
co-residency. Failing the memory gate remains a blocked result even after an
optimizer/export succeeds.

```bash
# Stop the owned API/engine first. Run only with an informative COMPLETE group.
PYTHONPATH=. .venv/bin/python scripts/probe_s4_training.py update \
  --infer-model .s1-cache/models/Qwen3-VL-4B-Instruct-4bit \
  --train-model .s1-cache/models/Qwen3-VL-4B-Instruct-aligned \
  --output artifacts/s4/policy-aligned --group "$COMPLETE_GROUP"
# Restart production using policy-v1.json / adapter-v1, collect fresh episodes,
# stop it again, then preflight the fresh group with --policy-version 1.
```

## Original preflight before alignment

S1/S2 and the measured S3 gates remain unchanged. The coding gates remain
accepted. This continuation adds a native PEFT export, immutable local policy
binding, complete coding-group exports and a real native policy preflight.
**No optimizer step, updated adapter, or trained-policy feedback is claimed.**

Implementation milestone: `468cdef`.

## Identity and export contract

- The native archon rank-8 q/v LoRA exports untransposed PEFT A/B tensors,
  preserving float32 adapter parameters and alpha/rank scaling. The existing
  transposed custom MLX bridge remains unchanged. The native-to-PEFT-to-native
  roundtrip is exact; there are 2,949,120 trainable parameters.
- SHA-256 covers the actual training and inference checkpoint files separately,
  their loader/tokenizer metadata, and both PEFT files. A content-derived model
  name embeds version 0. The two different bases are never asserted equivalent.
- The owned production server starts with static `--lora-modules`; production
  logs confirm activation of 72 modules. `/v1/models` must match adapter root,
  base root and parent. Every completion must return the pinned model identity.
  Sessions reject changing or removing a binding between recorded turns.
- The proxy retains exact backend IDs/logprobs and strict usage checks. Policy
  model/version survive interaction, JSON export and client decoding. The probe
  independently reconciles dsh logs, all backend attempts and exported tokens;
  only passing episodes receive disk training exports. An incomplete or failed
  two-episode group cannot enter the native preflight.
- This is an **owned, sequential, static-snapshot** protocol, not backend-signed
  attestation or support for concurrent in-place adapter mutation. Use a fresh
  manifest/name and restart the owned server for a different policy. Runtime
  adapter updating is not enabled. Initial B tensors are zero: successful v0
  loading does not prove a nonzero update or a v0-to-v1 transition.

## Coding evidence and retained denominator

All new episodes use the same generic instruction, temperature 0, 512 output
tokens, 120-second activity deadline, fresh workspaces and official external
pytest checks. No hidden test or canonical solution enters model messages.

| Evidence under artifacts/s4 | Task | Passed | Model turns | Tool turns |
| --- | --- | ---: | --- | --- |
| policy-sync-53 | HumanEval/53 | 2/2 | 4, 4 | 3, 3 |
| policy-sync-0 | HumanEval/0 | 2/2 | 4, 4 | 3, 3 |
| policy-final-53 | HumanEval/53 | 2/2 | 4, 4 | 3, 3 |

All six episodes strictly reconcile, pass 1/1 official external test file each,
and have no failed requests or backend-only completions. Earlier evidence is
retained. Cumulative official-task engineering exploration is **17/31**, with
the previous **11/25** and its **14 failures** intact. Repeated selected tasks
and diagnostic configurations are not HumanEval accuracy or pass@k.

## Measured training blockers

The first full HumanEval/53 preflight (`policy-sync/preflight.json`) evaluated
all eight turns and 390 response tokens, without overwriting production
behavior logprobs. It took 42.01 seconds. Token-weighted bf16/4-bit behavior
logprob MAE is **0.10771345**, above the predeclared **0.01** preflight bound;
maximum per-token error is **4.96346903**. Matching adapter files/version names
does not make the quantized serving base equal to the bf16 training base.
This bound is a new coding-policy preflight criterion, not a revision of S2's
HF/native full-logit tolerance.

The final-code replay (`policy-final/preflight.json`) independently confirms
the failure in **41.85 seconds** over **8 turns / 396 response tokens**:
token-weighted MAE **0.10248081**, maximum token error **5.21190739**, threshold
unchanged at **0.01**, exit **2**. Its manifest model is
`s4-policy-v0-3eb0f9fb840aa4bf`; the initial replay used
`s4-policy-v0-6f0c78fbb794650a`. Different manifest paths and expanded metadata
coverage change the identity; these are independent v0 snapshots, not a
v0-to-v1 update. Both export the same seeded initial PEFT tensor file hash.

The complete group's actual episode rewards are **[1, 1]**. Group-normalized
advantages are **[0, 0]**. Turn-discounted rewards must not be treated as
independent episode rewards to manufacture a spread. No rewards were changed,
no passing episode was relabeled, and behavior logprobs were not replaced by
bf16 recomputation. A useful nonzero GRPO update is therefore blocked before
backward/AdamW. The preflight exits **2**, retains its report, and exports no
trained adapter. The script is intentionally a preflight, not a training loop.

## Memory scope

Training and serving are loaded **sequentially**, never together. The first
native full-group preflight sampled 8,954,668,288 MPS allocated bytes and
9,884,614,656 driver bytes after forward. These are snapshots, not activation
peaks or backward qualification; sequence lengths were 1179–1642, not 8192.
System swap was already 3814.88 MiB before preflight and 3806.88 MiB afterward.
The final preflight has the same after-forward MPS snapshots; system swap is
3790.88 MiB before and 3782.88 MiB after. Existing swap is not cleared or
misreported as newly allocated training memory.

The first server's installed Metal paged-cache planner allocated **18.91 GiB**
despite `--kv-cache-memory-bytes 1342177280`; do not interpret that CLI value as
a measured 1.25 GiB limit. The final replay adds
`--gpu-memory-utilization 0.35`, and the actual layout is **4.69 GiB** (2133
blocks). Server logs and system snapshots are retained under `policy-final`.
This does not alter S1 defaults or establish pressure-free training residency.

## Reproduce the original unaligned blocked gate

Run from the repository root; choose fresh evidence directories for new starts.

```bash
export HF_HOME="$PWD/.s1-cache/hf" HF_HUB_OFFLINE=1
export UV_CACHE_DIR="$PWD/.s1-cache/uv" TMPDIR="$PWD/.s1-cache/tmp"
export XDG_CACHE_HOME="$PWD/.s1-cache/xdg"
mkdir -p "$TMPDIR"
export POLICY_OUT=artifacts/s4/policy-reproduce
export GROUP_OUT=artifacts/s4/policy-reproduce-53

PYTHONPATH=. .venv/bin/python scripts/probe_s4_training.py prepare \
  --infer-model .s1-cache/models/Qwen3-VL-4B-Instruct-4bit \
  --train-model .s1-cache/models/Qwen3-VL-4B-Instruct --output "$POLICY_OUT"
MODEL=$(.venv/bin/python -c \
  'import json,os; print(json.load(open(os.environ["POLICY_OUT"]+"/policy-v0.json"))["model"])')

# Foreground in a separate terminal with the same variables; stop after rollout.
.s1-cache/venv-metal/bin/vllm serve \
  "$PWD/.s1-cache/models/Qwen3-VL-4B-Instruct-4bit" \
  --served-model-name mlx-community/Qwen3-VL-4B-Instruct-4bit \
  --host 127.0.0.1 --port 8000 --max-model-len 8192 --max-num-seqs 1 \
  --gpu-memory-utilization 0.35 --kv-cache-memory-bytes 1342177280 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --enable-lora --max-lora-rank 8 \
  --lora-modules "{\"name\":\"$MODEL\",\"path\":\"$PWD/$POLICY_OUT/adapter-v0\",\"base_model_name\":\"mlx-community/Qwen3-VL-4B-Instruct-4bit\"}"

PYTHONPATH=. .venv/bin/python scripts/probe_s4.py \
  --task-manifest .s1-cache/nano-swe/human-eval-53/manifest.json \
  --model "$MODEL" --policy-manifest "$POLICY_OUT/policy-v0.json" \
  --episodes 2 --timeout 120 --temperature 0 --output "$GROUP_OUT"

# Stop the owned production server before loading the trainer.
PYTHONPATH=. .venv/bin/python scripts/probe_s4_training.py preflight \
  --infer-model .s1-cache/models/Qwen3-VL-4B-Instruct-4bit \
  --train-model .s1-cache/models/Qwen3-VL-4B-Instruct \
  --output "$POLICY_OUT" --group "$GROUP_OUT"
# Expected exit 2: real policy drift and homogeneous rewards, not success.
```

## Remaining acceptance

1. Alignment passes deterministic /53 at **0.00486984**, but sampled /0 fails
   at **0.01112227** against **0.01**. Resolve residual operator precision.
   Preserve original 0.10771345 / 0.10248081 evidence and captured probabilities.
2. Obtain a genuinely informative complete group under declared sampling and
   unchanged external rewards. Keep successful groups as zero-advantage groups.
3. Only then run 1–3 bounded optimizer steps, export a changed adapter, restart
   production on its immutable version and reconcile fresh coding episodes.
4. Measure backward/optimizer/export and serving memory, including swap and
   activation peaks. No co-resident or long-context qualification is implied.

Validation: **106 regression tests pass**, targeted Ruff and `git diff --check`
pass. Owned production servers are stopped; caches, weights and raw evidence
remain repository-local and ignored. No multi-hour GRPO was started.
