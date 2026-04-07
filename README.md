<div align="center">

# nano-areal

<p>
  <a href="https://github.com/inclusionAI/AReaL"><img src="https://img.shields.io/badge/based%20on-AReaL-blue" alt="AReaL"></a>
  <img src="https://img.shields.io/badge/python-3.12-3776AB" alt="Python 3.12">
  <img src="https://img.shields.io/badge/model-Qwen3--0.6B-orange" alt="Qwen3-0.6B">
  <img src="https://img.shields.io/badge/dataset-BFCL--V4-green" alt="BFCL-V4">
  <img src="https://img.shields.io/badge/lines-~1500-lightgrey" alt="~1500 lines">
</p>

**A minimal, readable implementation of [AReaL](https://github.com/inclusionAI/AReaL) — async agentic RL for LLMs.**

*~1500 lines. No framework magic. Every concept is in plain sight.*

</div>

---

## What is this?

AReaL is an async RL system from Ant Group which is production-grade and powerful, but reading it is hard for beginners (super high-level abstraction 😅).

**nano-areal** strips it to the core ideas:

<table>
<tr><th>AReaL</th><th>nano-areal</th></tr>
<tr><td>Async RL with disaggregated GPUs</td><td>✅ <code>asyncio</code> producer–consumer, <code>max_head_offpolicyness</code></td></tr>
<tr><td>Agentic RL via OpenAI proxy</td><td>✅ FastAPI proxy hijacks traffic; agent uses plain <code>openai</code> SDK</td></tr>
<tr><td>Archon PyTorch-native training</td><td>✅ Qwen3 from scratch — RoPE, GQA, QK-norm, SwiGLU, FSDP2</td></tr>
<tr><td>awex weight sync</td><td>✅ NCCL P2P push; <code>copy</code> fallback for macOS sync mode</td></tr>
<tr><td>Multi-turn function calling</td><td>✅ BFCL-V4 dataset, AST reward verifier, turn-discount backprop</td></tr>
</table>

---

## Core Idea: The Proxy

The hardest part of agentic RL is capturing token-level data (IDs, log-probs) **without touching the agent code**.

AReaL solves this with an **OpenAI-compatible proxy server**. nano-areal does the same in ~200 lines:

```
 BFCLAgent                              vllm / vllm-metal
    │                                          ▲
    │  AsyncOpenAI(base_url=proxy_url)         │  forward + logprobs=True
    ▼                                          │
 ProxyServer (FastAPI :8001) ─────────────────┘
    │  tokenize input → capture output tokens → cache per session
    │  POST /rl/set_reward   ←  agent scores each turn
    │  POST /export_trajectories  →  reward backprop along turn tree
    ▼
 AsyncTrajectoryBuffer  →  GRPO trainer (GPU 1 / MPS)
```

**The agent never imports AReaL.** Any OpenAI-compatible agent framework works.

---

## Async RL

Rollout and training run concurrently on separate GPUs:

```python
await asyncio.gather(
    rollout_worker(engine, workflow, dataset, buffer),  # GPU 0: generate
    train_worker(trainer, buffer),                      # GPU 1: learn
)
```

Stale trajectories are automatically discarded:

```python
# trainer is at version 42; trajectory was generated at version 37
# gap = 5 > max_head_offpolicyness = 4  →  discarded, not fed to loss
```

---

## Quick Start

> **Prerequisites** — download BFCL-V4 data first:
> ```bash
> UV_PROJECT_ENVIRONMENT=/path/to/venv uv run python scripts/download_bfcl.py
> ```

<table>
<tr><th>Mode</th><th>Command</th></tr>
<tr>
<td>🖥️ 2×GPU async (recommended)</td>
<td>

```bash
uv run python train.py --model Qwen/Qwen3-0.6B
```

</td>
</tr>
<tr>
<td>🍎 macOS sync (debug)</td>
<td>

```bash
UV_PROJECT_ENVIRONMENT=~/venvs/nano-areal \
  uv run python train.py --sync
```

</td>
</tr>
<tr>
<td>📁 Local checkpoint</td>
<td>

```bash
uv run python train.py --model-path /path/to/checkpoint
```

</td>
</tr>
</table>

Logs land in `logs/`. Open `logs/dashboard_final.html` for training curves and trajectory replay.

### macOS Setup Note

vllm-metal requires **Python 3.12** and the venv should live on an APFS volume (not exFAT):

```bash
uv python install 3.12
uv python pin 3.12
uv venv ~/venvs/nano-areal --python 3.12
UV_PROJECT_ENVIRONMENT=~/venvs/nano-areal uv sync --extra metal --no-dev
```

---

## Project Layout

```
nano_areal/
├── config.py            all config as plain dataclasses
├── dataset.py           BFCL-V4 loader (3 files merged by index)
├── reward/bfcl.py       AST function-call verifier + turn discount
│
├── engine/
│   ├── proxy.py         OpenAI proxy server  ← the interesting bit
│   ├── buffer.py        async trajectory buffer + off-policyness filter
│   ├── rollout.py       vllm / vllm-metal process manager
│   └── types.py         InteractionRecord, TrajectoryTensors
│
├── agent/
│   ├── base.py          AgentBase: async def run(data, **extra_kwargs)
│   └── bfcl.py          multi-turn BFCL agent (plain openai SDK)
│
├── archon/
│   ├── model.py         Qwen3 in pure PyTorch (RoPE, GQA, QK-norm, SwiGLU)
│   ├── state_dict.py    HuggingFace ↔ archon weight key translation
│   └── trainer.py       GRPO loss + FSDP2 + torch.compile
│
├── sync/awex_bridge.py  awex NCCL weight sync (copy fallback for macOS)
└── viz/
    ├── terminal.py      Rich live dashboard
    └── dashboard.py     Plotly HTML training curves + trajectory replay

train.py                 entry point
scripts/
├── download_bfcl.py     fetch BFCL-V4 from gorilla GitHub
└── test_sync.py         unit tests (GRPO, buffer, reward, state_dict)
```

---

## GRPO in 10 Lines

```python
# group-normalize rewards → advantages
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

# PPO-clip objective
ratio = (log_probs - ref_log_probs).exp()
pg_loss = -torch.min(ratio * adv, ratio.clamp(1-ε, 1+ε) * adv)

# KL penalty keeps policy close to reference
kl = (log_probs - ref_log_probs) * mask

loss = ((pg_loss + β * kl) * mask).sum() / mask.sum()
```

Log-probs are **recomputed** at train time — required because rollout data is off-policy.

---

## nano-archon: Qwen3 from Scratch

The training model is a pure PyTorch re-implementation of Qwen3 — no HuggingFace forward pass:

```
embed_tokens
└── layers × 28
    ├── input_layernorm     (RMSNorm)
    ├── self_attn
    │   ├── q/k/v/o_proj    (GQA: 16 heads, 8 KV heads, head_dim=128)
    │   └── q_norm / k_norm (per-head RMSNorm — Qwen3 specific)
    ├── post_attention_layernorm
    └── mlp                 (SwiGLU: gate × up → down)
norm  (RMSNorm)
lm_head  (weight-tied with embed_tokens)
```

Weights load directly from HuggingFace / ModelScope safetensors via a key translation table.

---

## Write Your Own Agent

Any async function returning `{response_id: reward}` works:

```python
class MyAgent(AgentBase):
    async def run(self, data, **extra_kwargs):
        client = AsyncOpenAI(
            base_url=extra_kwargs["base_url"],
            api_key=extra_kwargs["api_key"],
        )
        response = await client.chat.completions.create(
            model="default", messages=data["messages"]
        )
        reward = score(response.choices[0].message.content, data["answer"])
        return {response.id: reward}

workflow = OpenAIProxyWorkflow(agent=MyAgent(), proxy_server=engine.proxy)
```

The proxy handles tokenization, log-prob capture, and reward backpropagation — the agent is framework-agnostic.

---

## Hardware

<table>
<tr><th>Setup</th><th>Mode</th><th>Notes</th></tr>
<tr><td>2× RTX 4090 / A100</td><td><code>async</code> (default)</td><td>Full async RL, awex NCCL weight sync</td></tr>
<tr><td>1× GPU (CUDA)</td><td><code>--sync</code></td><td><code>max_head_offpolicyness=0</code>, copy weight sync</td></tr>
<tr><td>Apple Silicon (M-series)</td><td><code>--sync</code></td><td>vllm-metal (MLX) + MPS training, Python 3.12</td></tr>
</table>

---

## References

<table>
<tr>
  <td><a href="https://github.com/inclusionAI/AReaL"><b>AReaL</b></a></td>
  <td>The production system this is based on — async agentic RL at scale</td>
</tr>
<tr>
  <td><a href="https://github.com/GeeeekExplorer/nano-vllm"><b>nano-vllm</b></a></td>
  <td>Inspiration for the nano-* readable-implementation style</td>
</tr>
<tr>
  <td><a href="https://gorilla.cs.berkeley.edu/"><b>BFCL-V4</b></a></td>
  <td>Berkeley Function Calling Leaderboard — multi-turn tool-use benchmark</td>
</tr>
<tr>
  <td><a href="https://arxiv.org/abs/2402.03300"><b>GRPO</b></a></td>
  <td>Group Relative Policy Optimization (DeepSeek-Math)</td>
</tr>
<tr>
  <td><a href="https://github.com/inclusionAI/asystem-awex"><b>awex</b></a></td>
  <td>NCCL-based async weight exchange between training and rollout GPUs</td>
</tr>
</table>
