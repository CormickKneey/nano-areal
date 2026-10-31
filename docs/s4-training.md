# S4 production policy preflight: October 31, 2026

## Continue12: whole-group shared autograd scheduling

The opt-in native runtime now evaluates a complete declared group with shared
prefix/KV autograd and one episode-balanced backward, instead of reconstructing
turn-zero history for every independent backward. Non-reentrant checkpoints
also cover each token's final normalization, affine head and log-softmax, so
the whole group does not retain vocabulary-sized score intermediates. Frozen
reference and differentiable policy graphs remain separate. No historical KV
is detached; projection/attention arithmetic, official rewards, allocation
limits, the **0.01** MAE threshold and independent **600-second** phase budgets
are unchanged. The operator revision **native-s4-affine-cache-r5** is not
adapter version v1.

Regression tests compare shared versus independent complete-history values
and gradients in fp32, check native bf16 gradient norm error within **1/128**,
reject retained vocabulary-sized head intermediates, verify identical
episode-balanced optimizer results, and reject an update after its backward
deadline. Existing causal-gradient and retained-KV tests are unchanged.

The source-bound original sampled /0 preflight reproduces exactly **MAE
0.009696634512675821 <= 0.01**, all **9 turns / 636 response tokens**, in
**187.73480583401397 seconds** (`policy-continue12-group-0/preflight.json`).
Unchanged rewards **[1,1]** / advantages **[0,0]** still block that group's
update.

### r5 measured boundary

`policy-continue12-group-update-10/update.json` independently reproduces
whole /10 **MAE 0.007422680729897122**, all **9 turns / 1338 tokens**, with
unchanged official rewards **[1,0]** / advantages **[1,-1]**. Its separate
frozen-reference pass plus differentiable group forward exhaust the unchanged
**600-second** update budget before a backward. Final status **error**,
**Updates = 0**, elapsed **981.019436417002 seconds**. A cleanup allocation
also fails; the primary timeout and secondary OOM are both retained.

The **8837** valid 100ms samples record allocated peak **19,398,518,528** and
driver peak **26,794,999,808 <= 26,800,603,136 bytes**. Swap increases
**4242.62 -> 4314.44 MiB**, **+71.82 MiB**; the partial memory gate **fails**.
No complete backward, optimizer, export, v1 or feedback is qualified.

### r6 selective checkpoint and exact-zero reference identity

The next operator revision, **native-s4-affine-cache-r6**, retains only
single-token MLP matrix-product outputs during layer checkpointing. It
recomputes the other operations, preserving their original arithmetic and all
historical KV gradients. Both MPS `aten.linear` and CPU `aten.mm` dispatches
are covered. A native two-layer/two-turn storage test counts exactly **24**
retained matrix products / **20,480 bytes**, excluding prefills, attention and
head outputs. Full-model retention still needs an actual measured budget.

For a verified exactly zero LoRA-B adapter with finite LoRA-A and all adapters
enabled, native policy values equal the frozen native base exactly. The
training path uses those native values as detached **reference scores only**,
avoiding another identical full-group decode. It never detaches current-policy
graphs or any historical KV. A changed/nonzero adapter falls back to actual
adapter-disabled native evaluation. Native regression tests compare both
reference branches with explicit frozen-base evaluation at exact equality.
No captured production score is used as a differentiable reference.

Per-turn update snapshots now include elapsed update-phase time. **169 tests
pass in 9.48 seconds**; targeted Ruff and diff checks pass. A selective
checkpoint/saved-tensor-hooks compatibility warning is confined to the
storage-inspection test.

The r6 original /0 confirmation again reproduces exactly **MAE
0.009696634512675821**, **187.14005770799122 seconds**
(`policy-continue12-selective-0/preflight.json`). Its bounded whole /10 run
independently reproduces exactly **0.007422680729897122**, then completes all
**nine differentiable forwards** in **481.54610141698504 seconds** of its
update phase. It times out during the single group backward at the unchanged
**600-second** limit. Final status **error**, **Updates = 0**, total elapsed
**980.7099036249856 seconds**; cleanup also OOMs, with both errors retained.

The **8708** valid samples record allocated peak **25,265,392,896** and driver
peak **26,788,708,352 <= 26,800,603,136 bytes**. Swap grows **4712.50 ->
9215.19 MiB**, **+4502.69 MiB**: the partial memory gate **fails**. Retaining
MLP products increases pressure and does not qualify complete backward,
optimizer or export. No immutable v1 or feedback exists from r5/r6.

### r7 bounded row batching, without retained MLP products

The next native schedule batches up to **16 causal decode rows** through
each layer, retaining the affine bias-rounding formula, Metal RMS, rounded
SwiGLU and bf16 Split-KV partition arithmetic. Its query offset is the full
KV length minus the query-row count; future response tokens remain causally
masked. The head still scores one token at a time. This changes projection
GEMM shapes, so previous numerical acceptance is not inherited: original
/0 and whole /10 must be remeasured under the same **0.01** bound.

MLP selective retention is **disabled** in this native schedule; its diagnostic
implementation and storage tests remain available. Complete prefix/KV
autograd, whole-group episode weighting, one backward, exact-zero reference
identity and **600-second** phase limits remain unchanged. The operator
revision **native-s4-affine-cache-r7** is not adapter v1. **171 tests pass in
9.55 seconds**, with no warnings; targeted Ruff and diff checks pass.
FP32 shared/batched versus independent sequential complete-history gradients
use the original **rtol 1e-4 / atol 1e-5**. The native checkpoint control uses
the same batched schedule without checkpointing, preserving its original
**rtol 1/128 / atol 1e-5**.

The r7 original /0 numerical probe **fails**, **MAE
0.010030500258761312 > 0.01**, **56.922539166000206 seconds**
(`policy-continue12-batched-0`). Tokenwise causal attention with batched
projections, r8, also **fails**, **0.011196943518213669**, **84.74214708298678
seconds** (`policy-continue12-row-attention-0`). Four-row full batching, r9,
**fails**, **0.0113531636021469**, **85.14666100000613 seconds**
(`policy-continue12-batched4-0`). No update is attempted on these operators;
all nine turns / 636 tokens and original rewards remain intact. Source patches
against `7685e5d` are retained as `continue12-r7/r8/r9-source.patch` under
`artifacts/s4/`; no failed measurement is relabeled as accepted.

### r10 tokenwise forward, batched frozen-linear input gradients

The next schedule preserves each frozen decode projection's original
**one-token forward**, including its affine correction, and each token's
causal Split-KV attention and head. Only the frozen linear input-gradient
matrix multiplication is batched over the checkpoint's up-to-16 rows.
This is the ordinary linear derivative; weights remain frozen. A first-order
custom autograd function saves the weight, not historical KV, and passes input
gradients to the complete shared history. Elementwise normalization, RoPE and
SwiGLU operate on the row group; layer products are recomputed, not retained.

CPU float64 tests check exact tokenwise forward equality and derivatives at
**rtol/atol 1e-12**; MPS float32 uses **rtol 1e-4 / atol 1e-5**. Complete-group
causal-gradient tests and native checkpoint controls remain. **177 tests pass
in 10.53 seconds**, targeted Ruff and diff checks pass. The operator revision
**native-s4-affine-cache-r10** is not adapter v1. Its source-bound original
/0 preflight reproduces exactly **MAE 0.009696634512675821 <= 0.01**, all
**9 turns / 636 tokens**, in **170.7715755419922 seconds**
(`policy-continue12-tokenwise-0/preflight.json`). Whole /10 qualification and
a measured complete update are pending.

## Continue11: persistent differentiable KV blocks

`cd7472b` replaces retained full-cache checkpoint inputs with immutable linked
KV blocks. Each layer returns only its new key/value block; complete history
is concatenated **inside** its non-reentrant checkpoint and reconstructed
during backward. Prefix reuse slices the original blocks without detaching
them. This bounds retained KV tensor storage to the generated history rather
than retaining a full concatenation for every decoded token. It does not
change attention, projections, rounding, rewards, or allocation limits.

The operator revision is **native-s4-affine-cache-r2**, not adapter version v1.
CPU regression coverage compares multi-turn checkpointed values and all
adapter gradients with a full causal forward, verifies nonzero gradients on
the reused historical value block, checks linear block storage, and exercises
more than 1000 linked blocks without recursive traversal. Existing native MPS
checkpoint-gradient tests still pass. **158 tests pass in 13.21 seconds**;
targeted Ruff and `git diff --check` pass.

The source-bound original sampled /0 confirmation reproduces exactly **MAE
0.009696634512675821 <= 0.01**, all **9 turns / 636 response tokens**, in
**207.98193591699237 seconds**
(`policy-continue11-chunks-0/preflight.json`). Its unchanged rewards **[1,1]**
and advantages **[0,0]** still block updating that group. This is a numerical
confirmation, not a no-swap memory qualification: system swap grows during
that preflight.

The first r2 update probe (`policy-continue11-chunks-update-10/update.json`)
independently reproduces the exact whole /10 MAE **0.007422680729897122**.
Its first forward drops to **18,932,501,504 allocated / 21,375,959,040 driver
bytes**, versus continue10's **22,302,444,544 / 23,590,551,552**. First backward
is **17,992,641,024 / 21,378,056,192 bytes**. However, the second differentiable
turn still OOMs, now at the output-head `logits.float()` allocation. The final
report records **status=error, updates=0**, **662.6399107500038 seconds**
including the full preflight, sampled driver peak **26,803,388,416** versus
budget **26,800,603,136 bytes**, and swap growth **999.37 MiB**. Its sampler
retains **5982** valid samples; both the primary OOM and cleanup OOM survive.
This is a failed memory gate, not an optimizer or export qualification.

Inspection identifies a separate retained allocation: the head's frozen bf16
affine biases are converted to fp32 separately for every scored token, and
linear backward retains each conversion. The follow-up **r3** runtime stores
one shared fp32 head-bias tensor. It is exactly the same value previously
produced by `.float()`; neither forward arithmetic nor gradients are replaced.
A saved-tensor regression checks shared storage across differentiable tokens
and exact forward/gradient equality with the original conversion path.
`437d236` records this follow-up; **159 tests pass in 11.18 seconds**, with
targeted Ruff and `git diff --check` passing. The fresh r3 original /0
confirmation (`policy-continue11-shared-head-0/preflight.json`) again yields
exactly **0.009696634512675821 MAE**, all **9 turns / 636 response tokens**,
in **204.15084787501837 seconds**. Its homogeneous rewards remain unchanged.
The r3 whole /10 bounded probe again reproduces exactly
**0.007422680729897122 MAE**. It completes the first **two** backwards without
OOM, then hits the unchanged **600-second update-phase deadline** during the
third differentiable turn. Its final report is **status=error, updates=0**,
**1048.0352956249844 seconds** including preflight. The sampled allocated /
driver peaks are **19,398,518,528 / 26,782,416,896 bytes**, below the unchanged
**26,800,603,136-byte** driver budget, with swap delta **-1534.88 MiB** and
**9524** valid samples. Its partial-run `memory_gate.passed=true` is **not**
complete backward/optimizer/export memory qualification: no optimizer or
export is reached. Evidence: `policy-continue11-shared-head-update-10`.

The next **r4** runtime groups up to **16** decode tokens inside each layer
checkpoint. Arithmetic remains sequential one-token projection, attention,
MLP and head evaluation, not batched GEMM or full-causal substitution. Each
checkpoint concatenates earlier blocks once, extends its local full cache
sequentially, and returns only the new block. This reduces the backward
fan-out from per-token concatenations into all prior individual token blocks.
Frozen adapter-disabled reference caches also survive differentiable current
evaluations; current-policy graphs are still reconstructed in full, and
`start_update()` clears all preflight caches. Tests compare chunk sizes 1, 4
and 16 with full causal values and adapter gradients, retain explicit nonzero
historical gradients, and check frozen-reference cache preservation.
**161 tests pass in 9.62 seconds**; targeted Ruff and `git diff --check` pass.
`5eeb63e` records r4. Its source-bound original /0 confirmation
(`policy-continue11-grouped-0/preflight.json`) reproduces exactly
**0.009696634512675821 MAE**, all **9 turns / 636 tokens**, in
**188.55584724998334 seconds**. The independent whole /10 update probe remains
the gate for backward, optimizer/export, memory and eventual production-v1
feedback; no deadline or allocation bound is changed.

### Final r4 bounded result: update-phase runtime blocks acceptance

`policy-continue11-grouped-update-10/update.json` independently reproduces
the exact whole /10 **MAE 0.007422680729897122**, unchanged rewards **[1,0]**
and advantages **[1,-1]**, all **9 turns / 1338 response tokens**. It completes
two backwards and the third forward, then times out during checkpoint
recomputation in the third backward at the unchanged **600-second** phase
limit. Final status is **error**, **Updates = 0**, total elapsed
**975.4926384160062 seconds** including its numerical preflight and cleanup.
Cleanup succeeds; the timeout is retained rather than a stale success status.

| Stage | MPS allocated bytes | MPS driver bytes |
|---|---:|---:|
| First forward | 18,608,835,584 | 21,384,347,648 |
| First backward | 18,190,821,888 | 22,460,186,624 |
| Second forward | 19,094,518,272 | 22,726,524,928 |
| Second backward | 18,282,984,960 | 22,846,062,592 |
| Third forward | 19,029,800,704 | 22,843,965,440 |

The **8803** valid 100ms samples record allocated peak **20,515,623,424** and
driver peak **26,780,319,744 bytes**, below the unchanged recommended budget
**26,800,603,136 bytes**. Swap decreases **4798.62 -> 4486.62 MiB**
(**-312.0 MiB**). Consequently the report's **partial-run**
`memory_gate.passed=true`, but this does **not** qualify complete backward,
optimizer or export memory: those phases have not all completed. First/second
backwards and the subsequent forward now survive the old second-turn OOM;
no full S4 memory acceptance follows from that improvement.

There is **no changed adapter, immutable v1, optimizer step or production-v1
feedback**. No new rollout starts occur: engineering accounting remains
**36/81 successes, all 45 failures retained**; /20 and /54 remain unstarted.
S1, S2 and accepted S3/bounded coding gates are unchanged. Owned serving and
native processes are stopped; port 8000 is not listening.

Final validation: **161 tests pass in 10.57 seconds**, targeted Ruff and
`git diff --check` pass. Both final r4 reports' native/probe/policy/group
source and evidence hashes verify against current files; original immutable
v0 checkpoint/adapter/policy hashes verify. No v1 files exist in either output.

Next, reduce complete-history replay/backward scheduling cost without
detaching historical KV, weakening source binding or extending either phase
budget. In particular, the current differentiable response call rebuilds
history from turn zero for each separately backpropagated turn. A complete
group-level autograd schedule needs its own gradient-equivalence and memory
tests before replacing that path. Requalify original /0 and whole /10 after
any further native/probe/update changes, then complete exactly one bounded
step, exact changed immutable export and fresh production/native-v1 feedback.
**Full S4 training remains NOT accepted.**

### Reproduce the r4 boundary

Run only with the owned API and engine stopped. Use a fresh output directory;
do not overwrite existing reports, adapters or policy manifests.

```bash
export HF_HOME="$PWD/.s1-cache/hf" HF_HUB_OFFLINE=1
export UV_CACHE_DIR="$PWD/.s1-cache/uv" TMPDIR="$PWD/.s1-cache/tmp"
export XDG_CACHE_HOME="$PWD/.s1-cache/xdg" PYTHONPATH=.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9
OUT=artifacts/s4/policy-continue11-reproduce
mkdir "$OUT"
cp artifacts/s4/policy-aligned/policy-v0.json "$OUT/policy-v0.json"
ln -s "$PWD/artifacts/s4/policy-aligned/adapter-v0" "$OUT/adapter-v0"
uv run --no-project --python .venv/bin/python python \
  scripts/probe_s4_training.py update \
  --infer-model .s1-cache/models/Qwen3-VL-4B-Instruct-4bit \
  --train-model .s1-cache/models/Qwen3-VL-4B-Instruct-aligned \
  --output "$OUT" --group artifacts/s4/policy-continue10-10 \
  --aligned-arithmetic
```

For original /0 source requalification, select `preflight` and group
`artifacts/s4/policy-sampled-0` in another fresh output directory. Exit 2 is
expected for its homogeneous rewards even when the numerical gate passes.

All continue10 failures and historical numerical measurements below remain
unchanged. Full S4 training is not accepted without the measured remaining
update, export, feedback and memory gates.

## Continue10: first informative complete whole group

The predeclared `continue10-sampling-plan.json` selects independent pairs for
HumanEval/10, /20 and /54 at temperature **1.0**, no seed, **2048** output
tokens and a **300-second** activity deadline, stopping further starts at the
first informative complete group. Only the /10 pair starts. Both episodes
complete with strict reconciliation and unchanged official external pytest
rewards **[1,0]**, giving episode advantages **[1,-1]**. The failed candidate
contains a Python syntax error that causes external pytest collection to
fail; this is a candidate failure, not the broken /38 official helper. No
test, prompt, reward, episode subset or turn discount is changed.

`policy-continue10-10` retains all **2/2** starts, with **9 turns / 1338 response
tokens** (611 successful-episode tokens and 727 failed-episode tokens).
Cumulative engineering sampling becomes **36/81 successes, 45 failures
retained**; this is not HumanEval accuracy or pass@k. The /20 and /54 pairs
remain unstarted, not denominator entries. The owned production API and engine
stop before the new whole-group native preflight. Existing /0 and /53
numerical evidence and the **0.01** threshold remain unchanged. An informative
group alone does not qualify an optimizer step, changed adapter or full S4.

### Native qualification and bounded update preparation

The exact whole /10 group passes native preflight at **MAE
0.007422680729897122 <= 0.01**, all **9 turns / 1338 tokens**, in
**382.967845 seconds** (`policy-continue10-native-10/preflight.json`). It has
nonzero episode advantages; no turn discount or proxy reward creates spread.

An initial update invocation is interrupted during its preflight, before any
backward, to configure a conservative MPS allocation limit. A second invocation
fails allocator initialization: high watermark **1.0** conflicts with the
default low watermark **1.4**. Its stderr and initial `started` report remain
untouched; the old memory sampler raises on an empty trace and masks the
primary failure, so that stale report is not a running or successful update.
Both attempts have **0 optimizer steps**. The declared retry explicitly sets
high **1.0**, low **0.9**, preserving the existing device-budget/no-swap gate.

`305a1e4` fixes only probe failure reporting: empty/failed memory samples cannot
hide the primary exception or qualify memory acceptance, observed partial
peaks remain visible, and allocator environment settings are recorded. Four
new regressions cover empty, valid, partially failed and primary-error report
paths. **155 tests pass**, targeted Ruff and `git diff --check` pass. Native
arithmetic, policy, checkpoints and rollout exports are unchanged.

Because the probe source changes, a fresh source-bound original sampled /0
confirmation runs before retrying /10. It reproduces exactly **MAE
0.009696634512675821**, all **9 turns / 636 tokens**, in **186.278161 seconds**
(`policy-continue10-confirmed-0/preflight.json`). Its numerical gate passes;
the original homogeneous **[1,1] / [0,0]** rewards/advantages still block any
update on that group. Historical reports are not overwritten.

### Complete reward spread exposes a native memory boundary

The first correctly configured bounded attempt rechecks all /10 turns at
exactly **0.007422680729897122 MAE**, then completes its first forward and
backward. The forward snapshot is **22,302,444,544 allocated /
23,590,551,552 driver bytes**; after backward it is **17,992,641,024 allocated /
23,609,425,920 driver bytes**. During the second differentiable turn, cached
attention's `key.repeat_interleave` fails with MPS out-of-memory under the
recommended-working-set allocation limit. There is **no optimizer step or
export**. A subsequent weight-conversion OOM in `AlignedS4Runtime.close()`
prevents the old finalizer from persisting the error and sampler peaks.
Therefore `policy-continue10-update-bounded-10/update.json` has a **stale**
`preflight-passed-no-update` status; its stderr
`continue10-update-final-10.log` records both errors. It is not a successful
update or a complete memory qualification.

`eb0f7c9` guards cleanup failures, preserves the primary exception and records
failure-time memory without allowing failed sampling to pass the memory gate.
Five memory-reporting cases now pass; the full suite is **156 passing tests**
in **11.89 seconds**, with targeted Ruff and `git diff --check` passing. This
does not change native arithmetic, relax allocation/parity/reward thresholds,
or fix the gradient-history memory demand itself. The retained-attempt ledger
is `continue10-retained-update-failures.json`; original logs/reports stay
untouched.

The final-source /0 confirmation again reproduces exactly
**0.009696634512675821**, all **9 turns / 636 tokens**, in **184.876783
seconds** (`policy-continue10-final-0/preflight.json`). The fresh reporting-fixed
attempt (`policy-continue10-update-final-10/update.json`) independently passes
the whole /10 group at **0.007422680729897122**, then records **status=error,
updates=0** after **504.870530 seconds** total. One forward/backward completes;
the next differentiable forward fails allocating a cached-value concatenation.
The original OOM and cleanup OOM are now both persisted, rather than leaving
a stale success-like preflight status. No allocation limit is disabled.

| Final bounded-attempt measurement | Bytes / result |
| --- | ---: |
| First gradient forward allocated / driver | 22,299,757,568 / 23,598,940,160 |
| First completed backward allocated / driver | 17,992,641,024 / 23,609,425,920 |
| 100ms sampled allocated peak | 26,227,601,408 |
| 100ms sampled driver peak | 26,803,388,416 |
| Recommended driver budget | 26,800,603,136 |
| Process RSS high-water | 1,045,921,792 |
| Valid memory samples | 4558 |
| System swap used before / after | 4836.38 / 3969.81 MiB |
| Memory gate | **failed** |
| Optimizer / export memory | **not reached** |

Swap decreases by **866.57 MiB**, but the driver peak exceeds the unchanged
recommended budget and the gradient forward aborts. This is a measured failed
bounded attempt, not full-checkpoint backward/optimizer/export qualification.
There is no `adapter-v1` or `policy-v1.json`, no production-v1 rollout, and
**full S4 training remains NOT accepted**. All owned serving/native processes
are stopped; full model loads remain sequential.

The next gate is memory-efficient complete-history differentiation, including
retained per-token KV tensors and repeated history replay. Do not detach old
KV gradients, disable memory limits, discard the failed episode or loosen
**MAE <= 0.01**. After arithmetic/probe changes, requalify the original /0 and
the exact informative /10 whole group before another bounded update, then
require changed immutable v1, fresh production feedback and all memory gates.

Reproduce with stopped production API/engine and a fresh output directory;
never overwrite an existing update report or v1 adapter:

```bash
export HF_HOME="$PWD/.s1-cache/hf" HF_HUB_OFFLINE=1
export UV_CACHE_DIR="$PWD/.s1-cache/uv" TMPDIR="$PWD/.s1-cache/tmp"
export XDG_CACHE_HOME="$PWD/.s1-cache/xdg" PYTHONPATH=.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9
OUT=artifacts/s4/policy-continue10-reproduce
mkdir "$OUT"
cp artifacts/s4/policy-aligned/policy-v0.json "$OUT/policy-v0.json"
ln -s "$PWD/artifacts/s4/policy-aligned/adapter-v0" "$OUT/adapter-v0"
.venv/bin/python scripts/probe_s4_training.py update \
  --infer-model .s1-cache/models/Qwen3-VL-4B-Instruct-4bit \
  --train-model .s1-cache/models/Qwen3-VL-4B-Instruct-aligned \
  --output "$OUT" --group artifacts/s4/policy-continue10-10 \
  --aligned-arithmetic
# The update rechecks this whole group's MAE before any backward/optimizer.
# An update attempt or changed export alone does not accept full S4.
```

## Continue9: declared sampling still has no informative complete group

Continue9 declares and completes three additional two-episode groups with the
unchanged generic tool instruction and official external pytest rewards. The
new groups add **6/6** engineering starts, taking cumulative declared sampling
to **35/79 successes and 44 failures retained**. This is engineering sampling,
not HumanEval accuracy or pass@k.

| Output | Task | Rewards | Eligibility | Outcome |
| --- | --- | --- | ---: | --- |
| `policy-continue9-38` | HumanEval/38 | `[0,0]` | 2/2 | Complete homogeneous failures |
| `policy-continue9-109` | HumanEval/109 | `[0,1]` | 1/2 | One activity-timeout failure, one complete success |
| `policy-continue9-0` | HumanEval/0 | `[1,1]` | 2/2 | Complete homogeneous successes |

The /38 failures are retained as external pytest results; its official test
file raises `NameError` for the provided `encode_cyclic` helper while checking
`decode_cyclic`, so the group cannot be used as an informative reward group.
The /109 timeout is ineligible, and the /0 pair has zero advantages. No
episode subset, pooled task, proxy reward, or turn discount is used. Therefore
no newly sampled group qualifies the native Gate-1 replay, **updates remain 0**,
and no optimizer, adapter export, v1 feedback, or backward/optimizer/export
memory measurement is claimed. The production server was stopped before any
native consideration; no serving/native co-residency occurred.

## Continue8: bound native parity passes; complete reward spread still blocked

The explicit native arithmetic runtime passes the original COMPLETE sampled
HumanEval/0 group at **MAE 0.00969663451268 <= 0.01**, over all **9 turns /
636 response tokens**, in **188.33 seconds** including initialization. This
is `artifacts/s4/policy-aligned-native/preflight.json`, not an MLX replay score.
Its immutable policy/adapter and original group evidence remain unchanged;
the report binds their hashes to the native arithmetic source hashes.
`policy_parity_passed=true`, but overall status is still **blocked** because
external rewards **[1,1]** yield advantages **[0,0]**. **Updates = 0**.

A separate source-bound confirmation reproduces the same MAE in **191.33
seconds**. A third final-source confirmation after the phase-budget guard,
`policy-aligned-native-final/preflight.json`, again measures exactly
**0.009696634512675821**, all nine turns, in **187.30 seconds**. The sole
blocker remains homogeneous rewards; no optimizer phase runs. The opt-in
runtime also passes the original deterministic /53
group at **0.00665944127949**, all **8 turns / 396 tokens**, in **123.59
seconds** (`policy-aligned-native-53/preflight.json`). Its rewards remain
homogeneous. The older default-native /53 result is not overwritten.

This qualifies only the measured group and runtime. It does not qualify a
new sampled group, backward/optimizer/export memory, a changed adapter, or
v1 production feedback. **Full S4 training is NOT accepted.** The historical
default-native /0 **0.01112226647**, deterministic /53 **0.00486984055**, and
unaligned **0.10771345 / 0.10248081** measurements remain intact. S1/S2/S3 and
bounded coding acceptance are unchanged; no existing default is replaced.

### Remaining arithmetic distinguished

- **Initialize rotary frequencies on the executing device.** CPU-created
  FP32 inverse frequencies differ from MLX Metal by up to 5.96046448e-8.
  Computing the same power expression on MPS matches the independently
  generated MLX frequencies and bf16 trig tables for positions 0–2047 exactly.
  On the original production trace, native device-initialized RoPE matches
  **648/648 Q and 648/648 K rows**, versus the previous 397/648 and 425/648.
  Products remain bf16; no all-FP32 RoPE replacement is used.
- **Decode attention rounds partition outputs before merging.** The installed
  production fallback dispatch splits this single-sequence decode into
  512-token partitions even below 4096 context tokens, according to its
  occupancy gate. Partition softmax/accumulation is FP32, normalized partition
  outputs are stored in bf16, and the reducer merges those rounded outputs
  with FP32 exp-sum weights. One unsplit FP32 SDPA call is not equivalent.
  Independent actual-kernel fixtures cover CPU/MPS references, masked future
  partitions, vectorized causal rows and finite nonzero input gradients.
- **Preserve two-stage RMS rounding, but match its reduction.** The new native
  Metal diagnostic uses four-value per-thread FP32 accumulation, SIMD/group
  reductions and precise reciprocal square root, then casts normalized values
  to bf16 before multiplying bf16 norm weights. This is not the rejected
  single-rounding RMS replacement. On shared production inputs all four norm
  types, including Q, now match **648/648 rows exactly**. An opt-in first-order
  autograd implementation also passes small independent forward/gradient
  tests; this is not a measured full-checkpoint backward result.
- **Keep the verified prior arithmetic.** Short-prefill Split-K, bf16 sigmoid
  intermediates, and affine decode four-value bias sums remain necessary.
  Shared-input first-layer projections compare 63 prefill and 63 decode rows:
  corrected split reduction and bias rounding have mean activation MAEs
  **8.60698513e-9 / 1.30970544e-8**, respectively. These activation metrics are
  not token-weighted policy gate scores.

The complete native cached diagnostic with all corrections measures
**0.00969663444614** in **169.96 seconds**. The subsequent actual preflight
above uses `AlignedS4Runtime`, not a diagnostic hook installed as a default.
It keeps frozen affine weights resident in FP32, bf16 prefill/activations,
and native FP32 LoRA parameters. Gradient evaluation recomputes the complete
replay history with per-layer non-reentrant checkpoints; prefix KV is not
detached or imported from the production trace. A segment-level **600-second**
runtime budget bounds each phase. After a complete successful preflight,
the update receives its own 600-second budget; a partial preflight cannot
renew the budget. Operator revision `r1` is not adapter v1.

The first bound preflight's final forward snapshot is **18,290,435,584 MPS
allocated / 21,208,186,880 driver bytes**. Swap is **5184.12 -> 4524.69 MiB**.
These are forward snapshots, **not backward peaks or a memory acceptance**.

### Retained counterfactuals

All complete rows use the entire unchanged /0 group; no episode or token
subset is accepted. Files are ignored under `artifacts/s4/policy-aligned-audit/`.

| File prefix (`.json`) | MAE | Seconds |
| --- | ---: | ---: |
| continue8-full-affine-rounded | 0.01082937519 | 66.70 |
| continue8-full-affine-device-rope | 0.01158532291 | 65.89 |
| continue8-dense-device-rope | 0.01098795013 | 43.92 |
| continue8-full-splitkv-device-rope | 0.01134980014 | 70.32 |
| continue8-cached-splitkv | 0.01327270881 | 170.18 |
| continue8-cached-metal-rms | 0.00969663445 | 169.96 |
| continue8-dense-metal-rms | 0.01012256273 | 42.83 |
| continue8-dense-metal-splitkv | 0.01190115122 | 45.98 |
| continue8-dense-metal-affine-head | 0.01096810926 | 43.57 |

The uncached/full-forward near miss **0.01012256273** still fails **0.01**.
The early uncached-weight cached experiment was stopped with owned SIGINT;
`continue8-cached-affine-rounded.json` retains `KeyboardInterrupt`, **301.04
seconds**, **1/9 turns / 21/636 tokens**. Its partial 0.01660425 is not a group
score. Production attention observation independently preserves the reference
**0.00000123694609**, max **0.00016190298**; that reference is not native
acceptance. Observational capture and native model loads remain sequential.

The corrected independent attention capture
`continue8-shared-attention-final.json` observes only the first prefill and
first decode at layers 0/17/35, not every token. Prefill FP32 SDPA activation
MAEs are **0 / 3.07336450e-8 / 0**. Decode partition-rounding activation MAEs
are **2.91038305e-11 / 1.78813934e-7 / 1.90734863e-6**, versus unsplit FP32
**2.24474570e-5 / 0.00018571666 / 0.00085682585**. These observations explain
the operator choice but do not replace a source-bound whole-group gate.

### Sampling declaration and next gate

Before fresh starts, `artifacts/s4/continue8-sampling-plan.json` declares
HumanEval/32 then /109, four episodes each, temperature 0.8, no seed, fresh
workspaces, **1024 output tokens / 180-second activity deadline**, unchanged
generic read/write/bash instruction and official external pytest rewards.
The larger completion budget is explicit, not a reward or MAE threshold
change; the probe default remains 512 tokens. Every started failure stays in
the denominator. Require each entire original four-episode group to be
COMPLETE and reconciled, with genuine external reward spread. Never subset
episodes, pool tasks, or use truncations/proxy rewards to manufacture advantage.
At declaration time cumulative sampling remains **24/49, 25 failures retained**;
planned starts are not completed evidence.

The first declared eight starts completed: /32 **0/4**, eligibility
**[true,true,true,false]**, with three complete external-test failures and
one max-tokens failure; /109 **2/4**, eligibility **[false,true,true,true]**,
with one max-tokens failure, two successes and one complete external-test
failure. Both entire four-episode groups are ineligible. The complete /109
failure is real but cannot be extracted from its incomplete original group.
Cumulative accounting is **26/57 starts, 31 failures retained**.

Only after those results, `continue8-sampling-plan-second.json` declares four
new independent two-episode /109 groups, in order `pair-a` through `pair-d`,
with the same temperature 0.8, 1024-token cap, 180-second deadline, instruction
and external reward. Complete all eight declared starts. Select the first
informative eligible whole pair in declared order for its own native gate;
never split the preceding groups, subset a pair or pool rewards across groups.

All eight second-batch starts completed. The ordered pairs have rewards
**[1,1] / [1,0] / [1,0] / [0,0]**, with eligibility
**[true,true] / [true,false] / [true,false] / [false,true]**. The first pair
has zero advantages, and every later pair includes a max-tokens episode.
This adds **4/8**, taking cumulative accounting to **30/65, 35 failures**.

After that batch, `continue8-sampling-plan-third.json` declares four new
independent /109 pairs, `long-a` through `long-d`, before any starts, with
temperature 0.8, **2048 output tokens / 240-second activity deadline**.
This is another explicit bounded completion-budget increase, not a relaxed
reward, eligibility or numerical gate; the CLI default stays 512. Complete
all eight starts and use the same first-eligible-whole-group selection rule.

The third batch also completed all eight starts. Its ordered reward pairs
are **[0,0] / [0,0] / [1,0] / [0,1]**, with eligibility
**[false,true] / [false,false] / [true,false] / [false,true]**. Every pair
contains a max-tokens failure; none can supply a valid negative advantage.
The extra completion budget did not establish completion, so no larger
unbounded sampling or optimizer run is inferred from it.

### Final sampling ledger and remaining gate

All outputs below are ignored under `artifacts/s4/`; no solutions, hidden
tests, weights or raw sessions are committed. Each row is its entire
predeclared group, not a selected episode subset.

| Output suffix | Episodes | Passed | Eligible | Outcome |
| --- | ---: | ---: | ---: | --- |
| policy-continue8-32 | 4 | 0 | 3 | Three complete failures, one truncation |
| policy-continue8-109 | 4 | 2 | 3 | Mixed rewards, one truncation |
| policy-continue8-109-pair-a | 2 | 2 | 2 | Complete, rewards [1,1], zero advantage |
| policy-continue8-109-pair-b | 2 | 1 | 1 | One success, one truncation |
| policy-continue8-109-pair-c | 2 | 1 | 1 | One success, one truncation |
| policy-continue8-109-pair-d | 2 | 0 | 1 | One truncation, one complete failure |
| policy-continue8-109-long-a | 2 | 0 | 1 | One truncation, one complete failure |
| policy-continue8-109-long-b | 2 | 0 | 0 | Two truncations |
| policy-continue8-109-long-c | 2 | 1 | 1 | One success, one truncation |
| policy-continue8-109-long-d | 2 | 1 | 1 | One truncation, one success |

This continuation adds **8/24** engineering starts: **14** complete eligible
episodes (eight successes and six genuine external-test failures), plus
**10** ineligible max-tokens failures. Cumulative declared sampling is now
**32/73 starts, all 41 failures retained**, not HumanEval accuracy. No new
informative COMPLETE group exists. Do not combine the six valid failures
with successes from another original group or treat the ten truncations as
negative-advantage episodes. The numerical gate is qualified only for its
measured original /0 and /53 groups, not hypothetical future groups.

**Updates remain 0.** The owned production API and engine are stopped before
native validation; no serving/native co-residency is introduced. No changed
adapter, immutable v1, v1 feedback or full-checkpoint backward/optimizer/export
memory gate exists. Full S4 training remains unaccepted. Next work starts
with a newly declared genuinely informative COMPLETE group, then that whole
group's native MAE <= 0.01 preflight, before one bounded update. Regression
validation passes **151 tests in 9.71 seconds**, targeted Ruff and
`git diff --check`; tests do not substitute for training measurements.

For native reproduction, use a fresh output directory containing a byte-copy
of the original `policy-v0.json` and an `adapter-v0` symlink to the original
immutable adapter. Keep production stopped, then run:

```bash
export HF_HOME="$PWD/.s1-cache/hf" HF_HUB_OFFLINE=1
export UV_CACHE_DIR="$PWD/.s1-cache/uv" TMPDIR="$PWD/.s1-cache/tmp"
export XDG_CACHE_HOME="$PWD/.s1-cache/xdg" PYTHONPATH=.
.venv/bin/python scripts/probe_s4_training.py preflight \
  --infer-model .s1-cache/models/Qwen3-VL-4B-Instruct-4bit \
  --train-model .s1-cache/models/Qwen3-VL-4B-Instruct-aligned \
  --output artifacts/s4/policy-aligned-native-reproduce \
  --group artifacts/s4/policy-sampled-0 --aligned-arithmetic
# Expected overall exit 2 for this zero-advantage group; numerical gate passes.
```

No update/export/feedback acceptance follows from this numerical pass.

## Continue7: layerwise arithmetic isolated; gate 1 remains blocked

No production/native default, training runtime, policy manifest, original
probability or reward changes in this continuation. S1/S2/S3 and bounded S4
coding acceptance remain intact. **Full S4 training is NOT accepted.** The
bound native /53 **0.00486984** and sampled /0 **0.01112227** results, unchanged
**MAE <= 0.01** threshold, and historical unaligned **0.10771345 / 0.10248081**
failures remain authoritative. All new results below are **unbound diagnostics**.

The new observational traces retain only the last activation row of each
prefill and first decode segment, at every layer, in original group order.
They cover **18 segments / 648 layer-position pairs** in the same untouched
**9-turn / 636-response-token** HumanEval/0 group. Exact input evidence hashes,
segment offsets, full stage shapes and trace hashes must match before a
comparison. Final normalization permits full-row versus last-row evaluation
because normalization is rowwise. Trace capture is capped at 32 segments;
both replay programs reject forward deadlines outside (0, 600] seconds.
Hook removal and source hashes are recorded; traces do not replace probabilities.

The observed prefill suffix lengths are **[1157,274,70,57,20,272,83,33,30]**;
reused tokens remain **[0,1168,1536,1616,1136,1168,1520,1648,1744]**. Tracing
preserves the production replay's **0.00000123694609 MAE**, maximum
**0.00016190298**, and the original dense cached native **0.01101615323**.
Production replay is still a reference, not native acceptance.

### Isolated findings

- **Short-prefill Split-K is a real missing operator distinction.** MLX
  v0.32.2's transposed quantized matrix dispatch chooses partitions based on
  actual row/output tiles. Each partial product is stored in bf16; the small
  column reduction also rounds in bf16, with up to eight lanes followed by
  ordered lane addition. Promoting that reduction to FP32 is not equivalent.
  For 2560 input features, a 70-row prefill uses one partition for 4096-output
  Q but five for 1024-output K/V; a 57-row Q projection uses two. The diagnostic
  reference matches an independent actual MLX synthetic projection exactly
  on CPU and MPS, with finite/nonzero CPU input gradients. On the actual group,
  it removes the first-layer Q/K/V last-row differences in **all nine prefills**.
  That is not a claim about all rows/layers or whole-group numerical acceptance.
- **Do not replace native RMSNorm with single-rounding RMSNorm.** On shared
  production inputs, native input/post-attention/K normalization matches
  **648/648** last rows exactly; Q normalization matches **636/648**, mean
  activation MAE **0.00000003349312**, maximum **0.03125**. Single-rounding
  replacements match zero complete rows for all four norm types. Their mean
  activation errors are much larger; they are not improvements in precision
  equivalence merely because they use more FP32 arithmetic.
- **SwiGLU needs the production sigmoid's intermediate rounding.** Its bf16
  expression evaluates `tail = 1 / (1 + exp(abs(gate)))`, chooses `tail` or
  `1 - tail`, and rounds both multiplications. The diagnostic `rounded_swiglu`
  matches **551/648** shared-input layer-position rows exactly, mean activation
  MAE **0.00000002097436**, maximum **0.0009765625**. Native SiLU and the previous
  all-FP32 single-rounding hypothesis have mean activation MAEs
  **0.00023012847 / 0.00028516994**. An independent integer-gate MLX fixture is
  exact on CPU/MPS and has finite/nonzero CPU gate/up gradients. This does not
  qualify arbitrary activation ranges, full-model backward, or an optimizer.
- **Preserve bf16 RoPE products while investigating trig differences.** With
  shared production Q/K normalization outputs, native RoPE's mean row MAEs
  are **0.00000651924 / 0.00000814590**. FP32 products with bf16 trig increase
  them to **0.00082417156 / 0.00094739381**; FP32 products and trig increase
  them further to **0.00109733434 / 0.00128359145**. These are unweighted means
  over 648 activation rows per Q/K side, not token-weighted logprob gate scores.
  Small remaining trig discrepancies and production attention arithmetic
  are not closed by this finding.

### Whole-group results and retained failure

All files below are ignored under `artifacts/s4/policy-aligned-audit/`.
Every complete row evaluates the full unchanged group; no subset is accepted.

| File | Execution | MAE | Seconds |
| --- | --- | ---: | ---: |
| paged-layer-trace.json | Production paged/prefix, observational layer trace | 0.00000123694609 | 28.54 |
| paged-rope-trace.json | Same, with post-RoPE Q/K trace | 0.00000123694609 | 29.60 |
| paged-rope-confirmation.json | Final-source repeat; byte-identical trace, kernel headers hashed | 0.00000123694609 | 29.04 |
| native-layer-trace.json | Dense native cached/prefix, observational trace | 0.01101615323 | 79.83 |
| native-splitk-trace.json | Dense cache + corrected prefill Split-K | 0.01116234690 | 84.54 |
| native-splitk-rope-trace.json | Independent Split-K repeat with post-RoPE trace | 0.01116234690 | 85.40 |
| native-splitk-rounded-swiglu.json | Dense cache + Split-K + rounded sigmoid | 0.01063418281 | 83.28 |
| native-full-rounded-swiglu.json | Full native forward, rounded sigmoid only | 0.01061233982 | 45.01 |
| native-full-rounded-rms.json | Same plus single-rounding RMSNorm | 0.01327380076 | 47.44 |
| native-full-rounded-head.json | Rounded sigmoid + affine decode head, dense first head | 0.01119566381 | 42.91 |

The rejected initial `native-splitk-fused.json` attempt used an incorrect
FP32 Split-K reduction plus the earlier fused RMS/SwiGLU hypotheses. It was
stopped with owned SIGINT and retained as `status=error`,
`error_type=KeyboardInterrupt`, **365.40 seconds**, **2/9 turns, 116/636 tokens**.
Its partial MAE **0.00372007981** is **not a complete-group score or acceptance**.
The corrected BF16 reduction was independently checked before the complete
Split-K results above. This interruption adds no rollout start.

### Reproduce

Use new output filenames; never overwrite original evidence. Keep production
API/engine stopped. MLX and full native model loads remain sequential.

```bash
export HF_HOME="$PWD/.s1-cache/hf" HF_HUB_OFFLINE=1
export UV_CACHE_DIR="$PWD/.s1-cache/uv" TMPDIR="$PWD/.s1-cache/tmp"
export XDG_CACHE_HOME="$PWD/.s1-cache/xdg" PYTHONPATH=.
.s1-cache/venv-metal/bin/python scripts/diagnose_s4_replay.py \
  --policy artifacts/s4/policy-aligned/policy-v0.json \
  --group artifacts/s4/policy-sampled-0 --execution paged --prefix-reuse \
  --trace-output artifacts/s4/policy-aligned-audit/new-paged-trace.safetensors \
  --output artifacts/s4/policy-aligned-audit/new-paged-trace.json
.venv/bin/python scripts/diagnose_s4_operators.py \
  --policy artifacts/s4/policy-aligned/policy-v0.json \
  --group artifacts/s4/policy-sampled-0 --cached-execution --prefix-reuse \
  --dense-projections --dense-head --splitk-prefill \
  --trace-output artifacts/s4/policy-aligned-audit/new-native-trace.safetensors \
  --trace-reference artifacts/s4/policy-aligned-audit/new-paged-trace.safetensors \
  --output artifacts/s4/policy-aligned-audit/new-native-trace.json
.venv/bin/python scripts/diagnose_s4_trace.py \
  --policy artifacts/s4/policy-aligned/policy-v0.json \
  --trace artifacts/s4/policy-aligned-audit/new-paged-trace.safetensors \
  --output artifacts/s4/policy-aligned-audit/new-shared-inputs.json
# Separate counterfactual: add --rounded-swiglu to the native command,
# with new output/trace filenames. Do not combine it with --fused-swiglu.
PYTHONPATH=. .venv/bin/python -m pytest -q \
  --basetemp=.s1-cache/tmp/pytest-continue7-reproduce
```

Next: combine verified prefill dispatch and activation references with the
still-mismatching quantized decode/head paths, using shared-input traces to
separate local errors from accumulated hidden/cache differences. Preserve
the native RMSNorm and bf16 RoPE rounding until evidence supports changes.
Any proposed native runtime must pass unchanged bound whole-group preflight
and gradient/provenance checks before further sampling or optimizer execution.

No new rollout starts: cumulative declared sampling remains **24/49**, with
**25 failures retained**. No informative COMPLETE group; /109's truncated
failure is still ineligible. **Updates = 0**; no changed immutable v1 adapter,
fresh v1 production feedback, or backward/optimizer/export memory measurement.
No production server was started in this continuation.

Validation: **136 regression tests pass**, including observational cached
scores, post-RoPE capture, provenance/schedule/shape guards, nonfinite and
bounded-capture rejection, both deadline validators, exact independent MLX
Split-K/SwiGLU checks on CPU/MPS, and finite/nonzero CPU operator gradients.
Targeted Ruff and `git diff --check` pass. This is not full-model MPS backward
or optimizer/export memory evidence.

## Continue6: production cache schedule reproduced

The same untouched sampled HumanEval/0 group (**9 turns / 636 response tokens**)
now has an independent teacher-forced reference using the installed production
`vllm-metal` paged attention wrappers and kernels. This does not start an HTTP
server, sample new episodes, change a manifest, or replace the native trainer.
The probe checks the original immutable binding and requires an exactly zero-B,
finite adapter before replaying the base alone. Changed adapters are rejected,
not silently ignored. All results remain **unbound diagnostics**.

| Evidence under artifacts/s4/policy-aligned-audit | Execution | MAE | Seconds |
| --- | --- | ---: | ---: |
| paged-replay.json | Paged prefill + one-token decode, fresh cache per turn | 0.00995035268 | 55.43 |
| paged-prefix-replay.json | Same kernels, 16-token block-aligned prefix reuse | 0.00000123695 | 26.13 |
| paged-prefix-confirmation.json | Independent repeat with binary/kernel source hashes | 0.00000123695 | 27.12 |
| mlx-cached-replay.json | Standard MLX KVCache, fresh cache per turn | 0.01347209809 | 53.73 |
| native-cached-dense.json | Native dense prefill/decode + prefix reuse | 0.01101615323 | 77.09 |
| native-cached-head.json | Native cache + prefix reuse, affine single-row head | 0.01087339527 | 221.82 |
| native-cached-arithmetic.json | Native dense cache + prefix, fused RMS/SwiGLU, FP32 SDPA | 0.01277691887 | 105.58 |

The replay uses the original tokens and probabilities, bf16 paged KV storage,
36 wrapped attention layers, absolute text positions, and a single-row output
head for both the final prefill token and each decode token. Prefix reuse is
bounded by the longest matching **already processed** prefix, rounded down to
16-token blocks, leaving at least one prompt token to process. It carries the
cache across turns and episode boundaries in original group order. This
reconstructs the sampled behavior probabilities to about 1.24e-6 aggregate MAE;
it is stronger evidence for the production cache schedule than row-splitting
hooks or a standard MLX model forward. It does not prove an exact scheduler
trace or qualify any other group, adapter, context length, or gradient path.
The independent repeat gives the identical aggregate MAE and maximum token
error **0.00016190298**. Reused-token counts in group order are
**[0,1168,1536,1616,1136,1168,1520,1648,1744]**. The confirmation records
MLX **0.32.2**, MLX-VLM **0.7.6**, vllm-metal **0.32.0**, and hashes of the
installed Metal extension, metallibs, and kernel sources as well as wrappers.

The native cached reference is separate from `response_logprobs`: production,
native architecture defaults, accepted S1/S2/S3, and the bound policy preflight
are untouched. CPU tests compare cached and full causal scores, exercise
block-aligned prefix reuse with a nonzero offset, and check finite/nonzero
adapter gradients. Passing these reference contracts is not a measured MPS
backward or a policy runtime transition. The unchanged bound native preflight
is still **0.01112227**, above **0.01**.

Neither the corrected cached output head nor the cached arithmetic combination
closes native parity. A separate all-corrected cached attempt with FP32 RoPE
(`native-cached-fused.json`) was interrupted and retained with `status=error`,
`error_type=KeyboardInterrupt`: only **1/9 turns, 21/636 tokens** completed.
Its partial MAE **0.01597123034** is not a whole-group result; its stored elapsed
time refers to the last completed turn, not total wall time to interruption.
There is no selection of its partial group as evidence for acceptance. Native
diagnostics now enforce a forward deadline of at most 600 seconds, including
between cached segments, and reuse uploaded quantization tensors rather than
copying the same tensors from CPU for every generated token.

```bash
export HF_HOME="$PWD/.s1-cache/hf" HF_HUB_OFFLINE=1
export UV_CACHE_DIR="$PWD/.s1-cache/uv" TMPDIR="$PWD/.s1-cache/tmp"
export XDG_CACHE_HOME="$PWD/.s1-cache/xdg" PYTHONPATH=.
# Stop the owned production API AND engine before any replay/native load.
.s1-cache/venv-metal/bin/python scripts/diagnose_s4_replay.py \
  --policy artifacts/s4/policy-aligned/policy-v0.json \
  --group artifacts/s4/policy-sampled-0 --execution paged --prefix-reuse \
  --output artifacts/s4/policy-aligned-audit/paged-prefix-reproduce.json
# Omit --prefix-reuse for fresh-cache paged execution; use --execution cached
# for the standard MLX cache comparison. Always choose a new output filename.
.venv/bin/python scripts/diagnose_s4_operators.py \
  --policy artifacts/s4/policy-aligned/policy-v0.json \
  --group artifacts/s4/policy-sampled-0 --cached-execution --prefix-reuse \
  --dense-projections --dense-head \
  --output artifacts/s4/policy-aligned-audit/native-cached-dense-reproduce.json
```

Do not confuse the near-exact production replay with native numerical gate
acceptance. No new declared sampling is unlocked: **24/49 starts**, **25 failures
retained**, no informative COMPLETE group. **Updates = 0**; no changed v1 adapter,
fresh v1 production feedback, or backward/optimizer/export memory result.
Full S4 training remains unaccepted.

Validation: **120 tests pass**, including cached/full CPU score and adapter
gradient parity, changed-prefix invalidation, zero-B-only replay guards,
contiguous-response checks, and rejection of unbounded diagnostic deadlines.
Targeted Ruff and `git diff --check` pass. Reproduce the suite with
`PYTHONPATH=. .venv/bin/python -m pytest -q --basetemp=.s1-cache/tmp/pytest-replay`.

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
