# S4 dsh / nano-swe foundations: October 30, 2026

## Status: implemented foundations, not accepted

S3 remains blocked by the unchanged model-backed BFCL fixture. No GRPO,
adapter update, or training run was started. This milestone implements and
exercises durable capture, strict accounting, real tools, trusted verification,
episode deadlines, and explicit denominators. Actual HumanEval-derived tasks
still fail, and cancelled requests do not fully reconcile. S4 is **not green**.

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
  requests are drained before the next episode, or cancelled after 30 seconds,
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

## Measured outcomes, including failures

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
  --episodes 2 --timeout 35 --temperature 0 \
  --output artifacts/s4/humaneval-53
# Nonzero exit is expected unless every episode passes all gates.
# Stop the owned production server after the probe (Ctrl-C in its terminal).

PYTHONPATH=. .venv/bin/python -m pytest -q
.venv/bin/ruff check nano_areal/agent/dsh.py nano_areal/agent/dsh_audit.py \
  nano_areal/reward/dsh_task.py scripts/probe_s4.py scripts/test_s4.py \
  scripts/build_nano_swe.py scripts/probe_bfcl_tools.py
```

Validation: **75 regression tests pass**, including watchdog termination,
success/timeout persistence, argument/replay/count mismatch rejection,
external pytest execution, verifier timeouts and source/test separation.
The listed targeted Ruff command passes. Earlier failed logs remain local.

## Next gates, in order

1. S3: resolve or formally revise the ambiguous unchanged BFCL gate; the
   explicit-path diagnostic cannot substitute for it.
2. S4: make actual coding tasks finish reliably with real tools; consider
   configured file-write/edit tools instead of fragile shell quoting, without
   providing canonical solutions or exposing hidden tests.
3. Make cancellation/session ownership explicit through proxy capture and
   export. A completed backend-only response must be classified and retained,
   not confused with a delivered dsh turn or a training trajectory.
4. Repeat a fixed task group with unchanged prompts/budgets and all attempts
   counted. Only reconsider multi-step GRPO after the above gates are real.
