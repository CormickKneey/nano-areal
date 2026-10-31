# S4 dsh / nano-swe foundations: October 31, 2026

## Status: bounded coding gate passes; training remains unaccepted

Continue12 replaces repeated per-turn backwards with complete-group shared
autograd, token-head checkpoints, batched frozen-linear/full-KV attention
gradients and selective frozen MLP product retention. No historical KV is
detached. Committed r13 (`2c8e77f`) requalifies original /0 at
**0.009696634512675821** and exact whole /10 at **0.007422680729897122**,
both below unchanged **0.01**. Nine forwards finish in **358.38379254200845s**,
but the single backward still reaches **600s** before an optimizer step.
Driver peak **26,786,611,200 <= 26,800,603,136 bytes**; swap **+123.44 MiB**
fails the no-growth gate. Faster r14/r15 Metal attention fails whole-policy
parity and is archived, not promoted. **Updates = 0; no v1 or feedback**.
Sampling stays **36/81 successes, 45 failures retained**. See
[continue12 measurements and historical failures](s4-training.md).

Continue11 bounds retained KV storage with persistent differentiable blocks,
shares the frozen fp32 head biases, and groups sequential decode arithmetic
inside 16-token layer checkpoints. No historical KV is detached. Final r4
source-bound original /0 and exact whole /10 MAEs reproduce exactly
**0.009696634512675821** and **0.007422680729897122**, respectively, against
the unchanged **0.01** bound. **161 tests pass**. The final bounded update
completes two backwards and a third forward without OOM, then hits the
unchanged **600-second update-phase deadline** in the third backward.
**Updates remain 0**; no v1 adapter or production-v1 feedback exists.

Its partial-run sampled driver peak is **26,780,319,744 bytes**, below the
unchanged **26,800,603,136-byte** budget, with swap delta **-312.0 MiB**.
The report's `memory_gate.passed=true` covers only the interrupted trajectory,
**not** complete backward/optimizer/export qualification. Complete-history
replay scheduling/runtime is now the immediate blocker, not permission to
extend budgets or detach KV. Sampling remains **36/81 successes, 45 failures
retained**, with no new starts. See [continue11 evidence](s4-training.md).
All owned serving/native processes are stopped. **Full S4 is NOT accepted.**

Continue10 obtains the first informative COMPLETE whole group: HumanEval/10
has unchanged external rewards **[1,0]**, advantages **[1,-1]**, strict
reconciliation and no truncation/timeout. The zero reward is a candidate
syntax-error failure, not a broken official test helper. Both starts remain
in the ledger: **36/81 successes, 45 failures retained**, including continue9's
six starts. The declared /20 and /54 pairs do not start because the plan stops
at the first informative group. The exact whole group's native preflight
passes the unchanged **MAE <= 0.01** at **0.007422680729897122**, all
**9 turns / 1338 tokens**. A bounded
attempt completes its first backward but exceeds the recommended MPS memory
allocation limit in its second differentiable turn. **Updates remain 0**;
there is no changed adapter, v1 feedback or complete optimizer/export memory
qualification. Reporting fixes retain sampler/cleanup failures instead of
mistaking stale statuses for successful updates. **156 tests pass.** See
[training evidence](s4-training.md); the blocker moves from complete reward
spread to gradient-history memory, not full S4 acceptance.

The reporting-fixed final attempt records **status=error, updates=0** after
**504.87 seconds**, including its own full numerical preflight. Sampled driver
peak **26,803,388,416 bytes** exceeds the unchanged recommended budget
**26,800,603,136 bytes**; the second gradient forward OOMs despite system swap
decreasing **4836.38 -> 3969.81 MiB**. No optimizer/export memory or v1 feedback
is measured. The final-probe original /0 confirmation still reproduces exactly
**0.009696634512675821**. Owned production/native processes are stopped.

Continue8 closes the original sampled /0 numerical gate in an explicit,
source-bound native runtime: **0.00969663451268 MAE <= 0.01**, all **9 turns /
636 response tokens**, repeated with the same result. The original default
native **0.01112227** failure remains historical evidence. Device-initialized
RoPE, native Metal RMS reduction, and bf16 Split-KV partition-output rounding
complete the prior affine/Split-K/SwiGLU arithmetic. This is not promotion of
the unbound production reference. See [native training evidence](s4-training.md).
The original group still has rewards **[1,1]**, advantages **[0,0]** and
**updates = 0**. Three predeclared batches add **8/24** engineering starts,
taking cumulative accounting to **32/73, all 41 failures retained**. Fourteen
new episodes are complete, but the only complete whole group has rewards
[1,1]; every mixed original group contains a truncation. No subsets or pooled
rewards qualify. No changed adapter, v1 feedback or full-checkpoint backward
memory is accepted. **151 regression tests pass. Full S4 training remains
unaccepted.** The opt-in original /53 preflight also passes at **0.00665944**;
the historical default-native **0.00486984** remains intact.

Continue7 adds observational paged/prefix layer traces and shared-input
operator isolation. Short-prefill Split-K and bf16 sigmoid rounding are now
independently reproduced; native RMSNorm should not be replaced with a
single-rounding variant. Neither correction closes the unchanged **0.01**
whole-group logprob gate: cached Split-K + rounded sigmoid is **0.01063418**,
full-forward rounded sigmoid **0.01061234**. These are unbound diagnostics,
not native policy acceptance. The bound sampled /0 remains **0.01112227**.
See [continue7 evidence](s4-training.md). No new sampling, updates or backward
memory qualification; declared **24/49**, **25 failures retained**, updates **0**.
Current regression coverage is **136 passing tests**; no training gate is inferred.

Continue5 isolates MLX single-token affine bias-sum rounding: a diagnostic
reference exactly matches two isolated actual-checkpoint projections, but
whole sampled /0 still fails (head-isolation MAE **0.01090651**, bound **0.01**).
No production/trainer operators are changed and no new rollout starts occur.
Declared sampling remains **24/49**, **25 failures retained**; updates remain
**0**. See [operator investigation](s4-training.md). Regression coverage is now
**108 tests**; this does not qualify backward memory or full S4 training.

S3's unchanged BFCL fixture now passes 2/2 under production hermes; see
`docs/s3-multimodal.md`. On the same production backend, the final fixed S4
configuration solves **HumanEval/53 2/2 and HumanEval/0 2/2** with official
external checks, real multi-turn tools, durable logs and exact accounting.
No GRPO, adapter update, or training run was started. This accepts the bounded
coding/session/hidden-check gate, not full S4 training or benchmark accuracy.

The subsequent [production training preflight](s4-training.md) pins native PEFT
exports to served model identities and passes six more coding episodes.
The original bf16/4-bit drift is retained as failed evidence. The aligned-base
continuation passes deterministic /53 (MAE **0.00486984** versus **0.01**),
but sampled /0 still fails at **0.01112227**. Complete successful groups have
zero advantages, and other
sampled groups contain timeouts/truncations. No informative complete group,
optimizer update, changed export or backward memory gate is accepted.

### File tools and cancellation correction

The prior patch used `select`, which this Loader silently skipped. Patches
must target row `id`, and `config` replaces rather than deep-merges the row.
`scripts/dsh_patch.yml` now restates the adapter settings and enables the
installed `tool-fs` row's `read`, `write`, and `edit` tools. The generic task
instruction reads README, writes solution.py with structured tools, and runs
a local check using bash. Neither solutions nor official hidden checks are
provided in that instruction. Successful sessions actually execute
`read -> write -> bash -> final`, sometimes with an additional local check.

Each proxy session now owns its pending requests and a request ledger.
Ending a session marks it inactive, cancels and joins only its requests,
and prevents export until it is ended and drained. Failed requests/cancelled
episodes discard training export, not backend evidence. Complete reconciled
external-check failures now export unchanged rewards for training, never a
coding pass. Backend responses, usages,
proxy records and durable dsh logs remain independently retained. A
`response-ready` ledger entry does **not** assert delivery to dsh. Completed
backend-only responses are still rejected by strict delivered-message
reconciliation; no response list is trimmed to manufacture agreement.

Python task cancellation also closes and joins the owned SDK runtime before
returning and persists a cancelled result. The embedded Uvicorn server no
longer captures process signals: its previous SIGINT handler let the model
continue and later interrupted shutdown instead of cancelling the episode.
The corrected real Ctrl-C probe exits 130 with one started failed episode,
one cancelled backend request, zero backend-only completions, and a complete
report. The 2-second deadline probe similarly records a timeout plus one
cancelled request. Both remain failed episodes: an unreplayed final tool
result is rejected and no successful training export is claimed. Null usage
on a cancelled backend request means unknown, never zero computed tokens.

### Final fixed groups and full exploration denominator

All rows use fresh workspaces. Final groups use temperature 0, 512 output
tokens, a 120-second activity deadline, and the same generic task instruction.
Each successful episode passes 1/1 external pytest test file containing the
official `check(candidate)`, not merely local examples.

| Final evidence under artifacts/s4 | Episodes passed | Model turns | Tool turns | Hidden test files passed |
| --- | ---: | --- | --- | ---: |
| accepted-53/report.json | 2/2 | 4, 4 | 3, 3 | 2/2 |
| accepted-0/report.json | 2/2 | 5, 4 | 4, 3 | 2/2 |

Every final episode exactly matches dsh input/output/total usage, tool calls,
executions/results, result-text replay, backend attempts and exported token
counts. No final episode has a failed request or backend-only response.

Failures are not removed from the overall engineering denominator:

| Evidence directory under artifacts/s4 | Started | Passed | Outcome |
| --- | ---: | ---: | --- |
| humaneval-smoke | 2 | 0 | Earlier accounting/verification failures |
| humaneval-deterministic | 2 | 0 | Max-tokens and interrupted orphan session |
| humaneval-bounded | 2 | 0 | Earlier timeouts |
| humaneval-53 | 2 | 0 | Earlier timeouts/accounting failures |
| file-tools-53 | 2 | 0 | Ignored select patch; max-tokens |
| file-tools-fixed-53 | 2 | 2 | Correct id patch; successful tools/checks |
| cancel-owned-53 | 1 | 0 | Deliberate deadline; initial cancellation audit |
| owned-final-53 | 2 | 2 | Owned-request implementation replay |
| owned-final-0 | 2 | 2 | Official nontrivial function task replay |
| interrupted-owned-0 | 2 | 1 | First passed; SIGINT exposed Uvicorn ownership bug |
| interrupted-fixed-0 | 1 | 0 | Deliberate Ctrl-C; classified, reaped and retained |
| cancel-final-53 | 1 | 0 | Deliberate deadline; classified, reaped and retained |
| accepted-53 | 2 | 2 | Final fixed group |
| accepted-0 | 2 | 2 | Final fixed group |
| policy-sync-53 | 2 | 2 | Versioned PEFT production replay |
| policy-sync-0 | 2 | 2 | Versioned PEFT production replay |
| policy-final-53 | 2 | 2 | Final provenance and reduced-cache replay |
| policy-aligned-53 | 2 | 2 | Aligned-base production parity group |
| policy-sampled-0 | 2 | 2 | Temperature 0.8; complete, zero advantages |
| policy-sampled-32 | 2 | 0 | Temperature 0.8; verified failure and max-tokens |
| policy-sampled-38 | 4 | 0 | Temperature 1.0; three verified failures and timeout |
| policy-sampled-91 | 4 | 0 | Temperature 0.8; three verified failures and timeout |
| policy-sampled-109 | 4 | 3 | Temperature 0.8; three passes and max-tokens |
| **All started official-task episodes** | **49** | **24** | **25 failures retained** |

The new run contributes 11/17; the previous handoff contributes 0/8. The
earlier deterministic report has only one finished row, but two retained
dsh homes prove two starts: the interrupted attempt is still counted.
The first SIGINT diagnostic has two recorded rows but no final summary;
its second row still counts as a failure. The corrected SIGINT run planned
two episodes but started only one; its unstarted second episode is not a
failure or a pass. Mixed prompts/configurations, repeated tasks and deliberate
cancellation tests make the earlier 11/25 an exploration ledger, not HumanEval
accuracy. The training-preflight continuation adds 6/6 for 17/31. The aligned
continuation adds 7/18 for **24/49**; all 25 failures remain. The /109 group
cannot train on its apparent [1,1,0,1] rewards: its zero-reward episode is
truncated, so the group is incomplete. No failed episode is dropped and no
cross-task or cross-group selection is used to manufacture advantages.

Validation at the coding milestone: **79 regression tests pass** (the later
policy-preflight milestone has **86**, aligned continuation **106**), including asynchronous SDK
cancellation, per-session request isolation, active-session export rejection,
failed-episode discard without evidence deletion, embedded signal ownership,
and existing S1/S2/S3 contracts. The targeted Ruff command below passes;
unrelated pre-existing style violations in proxy.py are not swept into this
change. Owned production inference is stopped after these measurements.

## Implemented contracts

- `DshAgent(session_log_dir=...)` allocates a unique retained dsh home per run.
  The original `session.v4.jsonl`, SDK root events, and `result.json` survive
  success, error, and timeout. Without this option the previous temporary-home
  behavior remains. Raw homes contain package caches and sensitive sessions;
  all evidence is ignored under repository-local `artifacts/s4/`.
- The installed SDK's `request_timeout_seconds` bounds RPC requests, **not**
  `Session.run()`'s subsequent notification/idle wait. A joined watchdog now
  closes the owned runtime at the activity deadline and raises `TimeoutError`.
  Initialization has its separate SDK bound; bounded runtime shutdown follows
  the activity deadline. No detached inference or training worker is launched.
- `reconcile_session` compares every delivered assistant's input/output/total
  usage with proxy token masks, ordered tool IDs/names/arguments, actual
  `tool/call` and `tool/result` records, and exact replay of tool-result text.
  Missing/extra interactions, empty sessions, unexecuted calls and unreplayed
  results fail closed. The smoke additionally checks exported trajectory
  counts/token counts, all backend attempts/usages, and dsh failed attempts.
- `probe_s4.py` retains backend responses (including native IDs/logprobs),
  proxy records, failed requests and original dsh logs. In-flight backend
  requests are cancelled and joined by their session before export; the smoke
  also has a defensive 30-second drain before the next episode,
  so a previous episode cannot append into the next episode's accounting.
  A backend response completed after dsh has stopped is still recorded; it
  cannot be claimed as a matched dsh interaction.
- External verification has explicit `script` and `pytest` runners. The
  compatibility wrapper keeps `script`; nano-swe uses `pytest` so functions
  named `test_*` actually execute. The previous script-only invocation of the
  cached test file would only import its test definitions. Reports contain
  **test-file** denominators, return codes, output, collection errors and
  timeouts. A test file is not an individual assertion or a benchmark task.
- `build_nano_swe.py` materializes an actual local HumanEval row as a README
  task plus an external trusted test file. Only the problem prompt is exposed;
  neither hidden checks nor `canonical_solution` enter the task workspace or
  model messages. The manifest records task ID, entry point, source SHA-256,
  and static assertion sites. It is host-side withholding, **not an OS sandbox**.
- Each smoke attempt gets a fresh workspace under `.s1-cache/nano-swe/`.
  A pass requires completed dsh, independent hidden-test score 1, at least two
  tool-producing model turns, exactly one durable session, and full accounting.
  Timeouts, backend errors, max-tokens, insufficient tools, accounting mismatch
  and verification failure cannot earn a smoke pass.

## Production multi-turn transport fix

The checkpoint uses JSON tool calls; production must use `hermes`, not
`qwen3_xml` (see the S3 follow-up). The first real tool-result replay then
exposed another issue: the proxy emitted an assistant tool-only history item
with `content=""`. vLLM's multimodal message normalization treats that as a
nonempty text-block list, so the Qwen template inserts an extra newline before
the first tool call. The local tokenizer did not, and the strict prompt-ID
check correctly rejected the next request with 502.

Messages conversion now represents tool-only assistant content as `[]`.
Production `/tokenize` and local template IDs matched exactly for that shape;
the same comparison with `""` did not. Token-ID validation was **not relaxed**.

## Historical outcomes before the file-tool correction

Backend: official production vllm-metal HTTP, local 4-bit Qwen3-VL-4B,
8192-token context, one sequence, 1,342,177,280-byte KV budget, `hermes`.
No inference/training co-residency is claimed. All owned servers were stopped.

### Handwritten plumbing control, not benchmark performance

`artifacts/s4/reconciled/report.json`: the pre-existing handwritten `add(a,b)`
group passed **2/2 episodes**, with 5/4 model interactions and 4/3
tool-producing turns respectively. Every delivered dsh/proxy input/output
count and tool history matched; external pytest verification passed 1/1 test
file in each episode. This control is deliberately separate from actual
HumanEval quality; its prompt and sampling differ from the final runner.

### Actual HumanEval-derived nano-swe attempts

Source: `openai/human-eval`, downloaded into
`.s1-cache/datasets/human-eval/HumanEval.jsonl.gz`.
Compressed source SHA-256:
`b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`.
The builder used HumanEval/0 (`has_close_elements`) and HumanEval/53 (`add`),
with official `check(candidate)` functions wrapped in external pytest tests.

| Evidence directory under `artifacts/s4/` | Task | Attempts | Result |
| --- | --- | ---: | --- |
| `humaneval-smoke` | HumanEval/0 | 2 | 0 passes; shell/syntax repair loops, context overflow/backend retries, then hidden-test failure |
| `humaneval-deterministic` | HumanEval/0 | 2 started | 0 passes; first max-tokens, second externally interrupted when the SDK deadline gap was discovered; only the first has a finalized report row |
| `humaneval-bounded` | HumanEval/0 | 2 | 0 passes; both watchdog timeouts; also exposed pre-drain cross-episode accounting contamination |
| `humaneval-53` | HumanEval/53 | 2 | 0 passes; both 35-second activity timeouts with the final drain/export-accounting code |

Across these exploratory official-task runs: **0/8 started episodes pass**,
including the interrupted attempt as unsuccessful, across **0/2 distinct
task groups solved**. These selected tasks and changing diagnostic conditions
are not a representative HumanEval score or a pass@k estimate. The interrupted
attempt has raw logs but incomplete proxy accounting; it is not silently
dropped from the denominator.

In the final HumanEval/53 run, durable logs contain 16/7 assistant messages
and 18/14 actual tool executions/results. Proxy capture contains 17/8
completed model interactions: the extra response in each episode completed
after the dsh deadline and has no delivered dsh counterpart. The runner rejects
both; the discrepancy is visible rather than repaired by trimming counts.
The test-file budget is 2, passed 0; verification was not reached after the
timeouts. Do not interpret this as two completed hidden-test executions.

The model repeatedly produces invalid shell quoting / Python, sometimes
ignoring the explicit heredoc/printf guidance. A shell tool's successful
invocation is not evidence that the generated program passes. Both actual
task quality and cancellation-aware accounting remain blockers.

## Reproduce

```bash
export HF_HOME="$PWD/.s1-cache/hf" HF_HUB_OFFLINE=1
export UV_CACHE_DIR="$PWD/.s1-cache/uv" TMPDIR="$PWD/.s1-cache/tmp"
export XDG_CACHE_HOME="$PWD/.s1-cache/xdg"
mkdir -p "$TMPDIR" artifacts/s4 .s1-cache/datasets/human-eval

# Only needed if the trusted source is not already cached:
curl --fail --location \
  https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz \
  -o .s1-cache/datasets/human-eval/HumanEval.jsonl.gz

.venv/bin/python scripts/build_nano_swe.py \
  --data-path .s1-cache/datasets/human-eval/HumanEval.jsonl.gz \
  --task-id HumanEval/53

.s1-cache/venv-metal/bin/vllm serve \
  "$PWD/.s1-cache/models/Qwen3-VL-4B-Instruct-4bit" \
  --served-model-name mlx-community/Qwen3-VL-4B-Instruct-4bit \
  --host 127.0.0.1 --port 8000 --max-model-len 8192 --max-num-seqs 1 \
  --kv-cache-memory-bytes 1342177280 --enable-auto-tool-choice \
  --tool-call-parser hermes

# In another terminal with the same environment:
PYTHONPATH=. .venv/bin/python scripts/probe_s4.py \
  --task-manifest .s1-cache/nano-swe/human-eval-53/manifest.json \
  --episodes 2 --timeout 120 --temperature 0 \
  --output artifacts/s4/reproduce-53

.venv/bin/python scripts/build_nano_swe.py \
  --data-path .s1-cache/datasets/human-eval/HumanEval.jsonl.gz \
  --task-id HumanEval/0
PYTHONPATH=. .venv/bin/python scripts/probe_s4.py \
  --task-manifest .s1-cache/nano-swe/human-eval-0/manifest.json \
  --episodes 2 --timeout 120 --temperature 0 \
  --output artifacts/s4/reproduce-0
# Nonzero exit is expected unless every episode passes all gates.
# Stop the owned production server after the probe (Ctrl-C in its terminal).

PYTHONPATH=. .venv/bin/python -m pytest -q
.venv/bin/ruff check nano_areal/agent/dsh.py nano_areal/agent/dsh_audit.py \
  nano_areal/reward/dsh_task.py scripts/probe_s4.py scripts/test_s4.py \
  scripts/build_nano_swe.py scripts/probe_bfcl_tools.py
```

Previous validation: **75 regression tests passed**, including watchdog termination,
success/timeout persistence, argument/replay/count mismatch rejection,
external pytest execution, verifier timeouts and source/test separation.
The listed targeted Ruff command passes. Earlier failed logs remain local.

## Remaining training gates

The unchanged S3 gate, structured coding tools, fixed task-group checks and
explicit cancellation ownership requested in this continuation are measured.
No multi-hour GRPO is authorized or started by this result. Before claiming
full S4 training acceptance, validate production adapter/version synchronization,
bounded complete-group rollout/update/export behavior, memory budgets and
policy provenance. The earlier quantized/bf16 logprob drift and unqualified
8192-token co-residency limits remain; S1/S2 acceptance is not broadened here.
Keep completed backend-only responses as explicit accounting failures, and
include every future failed/interrupted start in the exploration denominator.
