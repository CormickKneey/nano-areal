

<div align="center">

**[English](README_EN.md)** | 中文

# nano-areal

<p>
  <a href="https://github.com/inclusionAI/AReaL"><img src="https://img.shields.io/badge/based%20on-AReaL-blue" alt="AReaL"></a>
  <img src="https://img.shields.io/badge/python-3.12-3776AB" alt="Python 3.12">
  <img src="https://img.shields.io/badge/model-Qwen3--0.6B-orange" alt="Qwen3-0.6B">
  <img src="https://img.shields.io/badge/dataset-BFCL--V4-green" alt="BFCL-V4">
  <img src="https://img.shields.io/badge/lines-~1500-lightgrey" alt="~1500 lines">
</p>

**一个简洁、易读的 [AReaL](https://github.com/inclusionAI/AReaL) 实现 —— 面向 LLM 的异步智能体强化学习。**

*约 1500 行代码。无框架魔法。每个概念都清晰可见。*

</div>

---

## 这是什么？

AReaL 是蚂蚁集团开发的异步强化学习系统，生产级且功能强大，但对初学者来说阅读门槛较高（高度抽象 😅）。

**nano-areal** 将其精简为核心理念：

<table>
<tr><th>AReaL</th><th>nano-areal</th></tr>
<tr><td>生产级大规模部署</td><td>✅ 轻装上阵，本地 Mac 可运行（~1500 行，vllm-metal + MPS）</td></tr>
<tr><td>支持 GPU 分离的异步 RL</td><td>✅ <code>asyncio</code> 生产者-消费者模型，<code>max_head_offpolicyness</code></td></tr>
<tr><td>通过 OpenAI 代理实现智能体 RL</td><td>✅ FastAPI 代理劫持流量；智能体使用原生 <code>openai</code> SDK</td></tr>
<tr><td>Archon PyTorch 原生训练</td><td>✅ 从零实现的 Qwen3 —— RoPE、GQA、QK-norm、SwiGLU、FSDP2</td></tr>
<tr><td>awex 权重同步</td><td>✅ NCCL P2P 推送；macOS 同步模式使用 <code>copy</code> 回退</td></tr>
<tr><td>多轮函数调用</td><td>✅ BFCL-V4 数据集，AST 奖励验证器，轮次折扣反向传播</td></tr>
</table>

---

## 核心理念：代理服务器

智能体强化学习最难的部分是在**不修改智能体代码**的前提下捕获 token 级别的数据（ID、log-probs）。

AReaL 通过 **OpenAI 兼容的代理服务器**解决此问题。nano-areal 用约 200 行代码实现相同功能：

```
 BFCLAgent                              vllm / vllm-metal
    │                                          ▲
    │  AsyncOpenAI(base_url=proxy_url)         │  forward + logprobs=True
    ▼                                          │
 ProxyServer (FastAPI :8001) ─────────────────┘
    │  tokenize 输入 → 捕获输出 token → 按会话缓存
    │  POST /rl/set_reward   ←  智能体为每轮评分
    │  POST /export_trajectories  →  沿轮次树反向传播奖励
    ▼
 AsyncTrajectoryBuffer  →  GRPO 训练器 (GPU 1 / MPS)
```

**智能体无需导入 AReaL。** 任何 OpenAI 兼容的智能体框架均可使用。

---

## 异步 RL

采样和训练在独立 GPU 上并发运行：

```python
await asyncio.gather(
    rollout_worker(engine, workflow, dataset, buffer),  # GPU 0: 生成
    train_worker(trainer, buffer),                      # GPU 1: 学习
)
```

陈旧的轨迹会自动丢弃：

```python
# 训练器版本为 42；轨迹生成于版本 37
# 差距 = 5 > max_head_offpolicyness = 4  →  丢弃，不用于损失计算
```

---

## 快速开始

> **前置条件** —— 首先下载 BFCL-V4 数据：
> ```bash
> uv run python scripts/download_bfcl.py
> ```

<table>
<tr><th>模式</th><th>命令</th></tr>
<tr>
<td>🖥️ 双 GPU 异步（推荐）</td>
<td>

```bash
uv run python train.py --model Qwen/Qwen3-0.6B
```

</td>
</tr>
<tr>
<td>🍎 macOS 同步（调试）</td>
<td>

```bash
uv run python train.py --sync
```

</td>
</tr>
<tr>
<td>📁 本地检查点</td>
<td>

```bash
uv run python train.py --model-path /path/to/checkpoint
```

</td>
</tr>
</table>

日志保存在 `logs/`。打开 `logs/dashboard_final.html` 查看训练曲线和轨迹回放。

### macOS 设置说明

vllm-metal 需要 **Python 3.12**，且虚拟环境应位于 APFS 卷（而非 exFAT）：

```bash
uv python install 3.12
uv python pin 3.12
uv venv ~/venvs/nano-areal --python 3.12
uv sync --extra metal --no-dev
```

---

## 项目结构

```
nano_areal/
├── config.py            所有配置均为普通数据类
├── dataset.py           BFCL-V4 加载器（3 个文件按索引合并）
├── reward/bfcl.py       AST 函数调用验证器 + 轮次折扣
│
├── engine/
│   ├── proxy.py         OpenAI 代理服务器  ← 核心部分
│   ├── buffer.py        异步轨迹缓冲区 + off-policyness 过滤
│   ├── rollout.py       vllm / vllm-metal 进程管理器
│   └── types.py         InteractionRecord, TrajectoryTensors
│
├── agent/
│   ├── base.py          AgentBase: async def run(data, **extra_kwargs)
│   └── bfcl.py          多轮 BFCL 智能体（原生 openai SDK）
│
├── archon/
│   ├── model.py         纯 PyTorch 实现的 Qwen3（RoPE、GQA、QK-norm、SwiGLU）
│   ├── state_dict.py    HuggingFace ↔ archon 权重键映射
│   └── trainer.py       GRPO 损失 + FSDP2 + torch.compile
│
├── sync/awex_bridge.py  awex NCCL 权重同步（macOS 使用 copy 回退）
└── viz/
    ├── terminal.py      Rich 实时仪表板
    └── dashboard.py     Plotly HTML 训练曲线 + 轨迹回放

train.py                 入口点
scripts/
├── download_bfcl.py     从 gorilla GitHub 获取 BFCL-V4
└── test_sync.py         单元测试（GRPO、缓冲区、奖励、state_dict）
```

---

## 10 行代码理解 GRPO

```python
# 组归一化奖励 → 优势
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

# PPO-clip 目标
ratio = (log_probs - ref_log_probs).exp()
pg_loss = -torch.min(ratio * adv, ratio.clamp(1-ε, 1+ε) * adv)

# KL 惩罚使策略接近参考模型
kl = (log_probs - ref_log_probs) * mask

loss = ((pg_loss + β * kl) * mask).sum() / mask.sum()
```

Log-probs 在训练时**重新计算** —— 这是必需的，因为采样数据是 off-policy 的。

---

## nano-archon：从零实现的 Qwen3

训练模型是 Qwen3 的纯 PyTorch 重新实现 —— 不使用 HuggingFace 的前向传播：

```
embed_tokens
└── layers × 28
    ├── input_layernorm     (RMSNorm)
    ├── self_attn
    │   ├── q/k/v/o_proj    (GQA: 16 头，8 KV 头，head_dim=128)
    │   └── q_norm / k_norm (每头 RMSNorm —— Qwen3 特有)
    ├── post_attention_layernorm
    └── mlp                 (SwiGLU: gate × up → down)
norm  (RMSNorm)
lm_head  (与 embed_tokens 权重共享)
```

权重通过键映射表直接从 HuggingFace / ModelScope 的 safetensors 加载。

---

## 编写自己的智能体

任何返回 `float`（最后一轮奖励）或 `{response_id: reward}` 的异步函数都可用：

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

代理处理 tokenization、log-prob 捕获和奖励反向传播 —— 智能体与框架无关。

---

## 硬件要求

<table>
<tr><th>配置</th><th>模式</th><th>说明</th></tr>
<tr><td>2× RTX 4090 / A100</td><td><code>async</code>（默认）</td><td>完整异步 RL，awex NCCL 权重同步</td></tr>
<tr><td>1× GPU (CUDA)</td><td><code>--sync</code></td><td><code>max_head_offpolicyness=0</code>，copy 权重同步</td></tr>
<tr><td>Apple Silicon (M 系列)</td><td><code>--sync</code></td><td>vllm-metal (MLX) + MPS 训练，Python 3.12</td></tr>
</table>

---

## 参考资料

<table>
<tr>
  <td><a href="https://github.com/inclusionAI/AReaL"><b>AReaL</b></a></td>
  <td>本项目基于的生产级系统 —— 大规模异步智能体强化学习</td>
</tr>
<tr>
  <td><a href="https://github.com/GeeeekExplorer/nano-vllm"><b>nano-vllm</b></a></td>
  <td>nano-* 可读实现风格的灵感来源</td>
</tr>
<tr>
  <td><a href="https://gorilla.cs.berkeley.edu/"><b>BFCL-V4</b></a></td>
  <td>Berkeley Function Calling Leaderboard —— 多轮工具使用基准</td>
</tr>
<tr>
  <td><a href="https://arxiv.org/abs/2402.03300"><b>GRPO</b></a></td>
  <td>Group Relative Policy Optimization（DeepSeek-Math）</td>
</tr>
<tr>
  <td><a href="https://github.com/inclusionAI/asystem-awex"><b>awex</b></a></td>
  <td>基于 NCCL 的训练和采样 GPU 间异步权重交换</td>
</tr>
</table>
