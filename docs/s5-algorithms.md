# S5 algorithm-switch unit acceptance

Accepted on **2026-11-04**, continue17. This is offline unit-contract acceptance,
not a new training measurement or an alternative-loss Metal training claim.
The continue16 S4 measurement stays closed at source tip `1764619`, qualified
chain Updates=20 and immutable policy `s4-policy-v20-8b386344d079b75f`.

## Shared trajectory and dispatcher path

`scripts/test_s5.py` adds **86 pytest cases**. All four loss names use the same
`Trajectory` -> `ArchonTrainer.train_step` -> `compute_policy_loss` path.
The trainer tests inject a tiny CPU policy and real AdamW without invoking
device selection, checkpoint loading, compilation, serving or a native trainer.
They verify padding, response-mask shifting, captured behavior-logprob alignment,
interleaved task group IDs, zeroed truncated rewards, finite/nonzero backward,
an actual parameter update and a frozen reference. The normal constructor is
covered by the existing training-contract tests.

The tests additionally lock the repository's current surrogate behavior:

- GRPO: default wrapper equals explicit dispatcher selection, including implicit
  detached behavior logprobs; token-level clipped ratios have exact gradients.
- CISPO: clipped detached importance weights multiply current logprobs. Positive
  clipped samples retain their policy gradient, unlike clipped GRPO samples.
- GSPO: response-length-normalized geometric sequence ratios, broadcast across
  response tokens, with exact losses/gradients and padding excluded.
- Dr.GRPO: unscaled centered rewards and the existing sequence-mean then
  batch-mean reduction; unequal response lengths have explicit gradient checks.
- Every variant: masked tokens have zero gradients, reference/behavior/reward
  and precomputed-advantage tensors do not receive gradients, zero advantages
  retain KL regularization, and entirely masked batches give zero loss/gradient.

These are contracts for the implemented surrogates, not a claim of complete
paper parity, distributed execution, alternative-loss native Metal qualification
or benchmark improvement. In particular, the current Dr.GRPO reduction is
sequence-mean, not a newly introduced fixed maximum-length denominator.

## Incomplete groups and truncation

The contiguous-group matrix covers batch sizes 0 through 5 at group size 3,
both scaled/unscaled advantages and both `on_incomplete` policies:

- `batch`: an incomplete batch falls back to whole-batch normalization.
- `zeros`: complete groups normalize independently and the short tail is zero,
  including a batch with no complete group. The latter previously normalized
  despite requesting zeros; continue17 fixes it without changing defaults.
- Invalid `on_incomplete` values fail explicitly; normalization detaches rewards
  before computing statistics, including empty input.
- Explicit `group_ids` normalize each observed task independently. A singleton
  task gets zero advantage; `group_size` and the contiguous-tail policy do not
  reconstruct missing task members. This remains an observed-group contract,
  not a completeness guarantee for real asynchronous rollout groups.
- Captured `finish_reason="length"` sets truncation and zero reward; `stop`
  preserves reward. The trainer also independently zeroes truncated rewards.

## Buffer and Metal defaults

`AsyncTrajectoryBuffer.get_batch` now calls the shared `is_offpolicy` predicate;
`get_batch_sync` continues to delegate to the same consumer. Tests cover negative,
exact, boundary and excessive gaps, policy updates after enqueue, stale/future
discarding, object identity, timeout/partial batches and accounting in both APIs.
Gap 0 accepts only the current policy; positive gaps use the same inclusive rule.

`TrainConfig`, `NanoArealConfig.for_macos`, `train.py --sync` and
`train.py --backend metal` still default to GRPO. Explicit CLI choices reach the
generic trainer configuration without starting it. A source-level guard checks
that `scripts/run_agentic_rl.py` still permits **only GRPO**, without importing
MLX. No alternate native Metal loss is enabled by this acceptance.

## Validation and retained evidence

From the repository root, using only repository-local caches:

```bash
export UV_CACHE_DIR="$PWD/.s1-cache/uv"
export TMPDIR="$PWD/.s1-cache/tmp"
export PYTHONPATH=.
uv run --no-sync python -m pytest -q scripts/test_s5.py \
  --basetemp=.s1-cache/tmp/pytest-s5-reproduce
uv run --no-sync python -m pytest -q \
  --basetemp=.s1-cache/tmp/pytest-s5-full-reproduce
uv run --no-sync python scripts/test_s1.py
.venv/bin/ruff check scripts/test_s5.py nano_areal/archon/losses.py \
  nano_areal/engine/buffer.py
```

Measured results: **86 passed in 1.27s** for S5; **329 passed, 1 warning in
18.95s** for the full suite; **7 S1 smoke probes passed**; targeted Ruff passed.
The warning is the existing selective-checkpoint saved-tensor-hook FutureWarning.
These are test counts, not task pass rates or optimizer-work measurements.

Ignored local evidence stays under `.s1-cache/continue17-*`: initial regression
run **15 failed / 69 passed** (including the initial CPU-fixture mock failure),
intermediate **84 passed**, final focused/full/smoke and lint logs. No failed log
was overwritten. The S4 audit bindings still hash to
`b712a88d4569279731f3c9024ec6c3dfb0a3603cc776ff8e9c48868da70724fd`.
No S4 evidence artifacts, acceptance thresholds or trained policies changed.
No server or long training job was started; port 8000 had no listener at closeout.
