"""
Autoregressive sampling for nano-archon (batch size 1, KV / GDN state caches).

Used by the local OpenAI-compatible server (`nano_areal.engine.local_server`)
so the whole rollout→train loop runs on CPU/MPS without vLLM. Also provides
`prompt_logprobs`, the primitive a teacher model needs for distillation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from nano_areal.archon.gdn import causal_conv1d, chunk_gated_delta_rule
from nano_areal.archon.model import Qwen35Model
from nano_areal.archon.mrope import apply_rope


@dataclass
class SamplingParams:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stop_token_ids: tuple[int, ...] = ()
    seed: int | None = None


@dataclass
class GenerationResult:
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str          # "stop" | "length"
    prompt_len: int


class _LayerState:
    __slots__ = ("kind", "k", "v", "conv", "rec")

    def __init__(self, kind: str):
        self.kind = kind
        self.k: Tensor | None = None
        self.v: Tensor | None = None
        self.conv: Tensor | None = None
        self.rec: Tensor | None = None


def _sample(logits: Tensor, sp: SamplingParams, gen: torch.Generator | None) -> tuple[int, float]:
    logits = logits.float()
    logp_full = F.log_softmax(logits, dim=-1)
    if sp.temperature <= 0:
        tok = int(logits.argmax().item())
        return tok, float(logp_full[tok].item())
    scaled = logits / sp.temperature
    if sp.top_k > 0:
        kth = torch.topk(scaled, sp.top_k).values[-1]
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    if sp.top_p < 1.0:
        srt, idx = torch.sort(scaled, descending=True)
        cum = F.softmax(srt, -1).cumsum(-1)
        remove = cum - F.softmax(srt, -1) > sp.top_p
        srt = srt.masked_fill(remove, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf")).scatter(0, idx, srt)
    probs = F.softmax(scaled, -1)
    tok = int(torch.multinomial(probs, 1, generator=gen).item())
    # report the *behaviour* logprob (temperature-scaled distribution actually sampled from)
    return tok, float(torch.log(probs[tok]).item())


class Generator:
    def __init__(self, model: Qwen35Model):
        self.model = model
        self.args = model.args
        self.tm = model.language_model

    # -- prefill/decode primitives ------------------------------------------

    @torch.no_grad()
    def _prefill(
        self, input_ids: Tensor, pixel_values: Tensor | None, image_grid_thw: Tensor | None
    ) -> tuple[Tensor, list[_LayerState], int]:
        """Full forward over the prompt, capturing per-layer caches. Returns last-token hidden."""
        positions = self.model.positions_for(input_ids, image_grid_thw)
        x = self.model.embed(input_ids, pixel_values, image_grid_thw)
        cos, sin = self.tm.rotary_emb(positions, dtype=x.dtype)
        states: list[_LayerState] = []
        for layer in self.tm.layers:
            st = _LayerState(layer.layer_type)
            h = layer.input_layernorm(x)
            if st.kind == "linear_attention":
                gdn = layer.linear_attn
                mixed, z, beta, g, B, T = gdn._project(h)
                K = gdn.conv1d.kernel_size[0]
                st.conv = F.pad(mixed, (K - 1, 0))[:, :, -(K - 1):]
                conv = causal_conv1d(mixed, gdn.conv1d.weight.squeeze(1), gdn.conv1d.bias)
                q, k, v = gdn._split_heads(conv, B, T)
                out, st.rec = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, output_final_state=True)
                out = gdn.norm(out.reshape(-1, gdn.head_v_dim), z.reshape(-1, gdn.head_v_dim))
                h = gdn.out_proj(out.reshape(B, T, -1))
            else:
                attn = layer.self_attn
                q, k, v, gate = attn._qkv(h)
                q, k = apply_rope(q, k, cos, sin)
                st.k, st.v = k, v
                o = attn._attend(q, k, v, None, causal=True).reshape(h.shape[0], h.shape[1], -1)
                if gate is not None:
                    o = o * torch.sigmoid(gate)
                h = attn.o_proj(o)
            x = x + h
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            states.append(st)
        x = self.tm.norm(x)
        next_pos = int(positions.max().item()) + 1
        return x, states, next_pos

    @torch.no_grad()
    def _decode_step(self, token: Tensor, pos: int, states: list[_LayerState]) -> Tensor:
        x = self.tm.embed_tokens(token.view(1, 1))
        positions = torch.tensor([[pos]], device=x.device)
        if self.args.mrope_section is not None:
            positions = positions.unsqueeze(0).expand(3, -1, -1)
        cos, sin = self.tm.rotary_emb(positions, dtype=x.dtype)
        for layer, st in zip(self.tm.layers, states):
            h = layer.input_layernorm(x)
            if st.kind == "linear_attention":
                h, st.conv, st.rec = layer.linear_attn.forward_step(h, st.conv, st.rec)
            else:
                h, st.k, st.v = layer.self_attn.forward_step(h, cos, sin, st.k, st.v)
            x = x + h
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.tm.norm(x)

    # -- public API ----------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: list[int],
        sp: SamplingParams,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        on_token=None,
    ) -> GenerationResult:
        dev = next(self.model.parameters()).device
        ids = torch.tensor([input_ids], device=dev)
        gen = None
        if sp.seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(sp.seed)
        h, states, pos = self._prefill(ids, pixel_values, image_grid_thw)
        logits = self.model.logits(h[:, -1])[0]
        out_ids: list[int] = []
        out_lp: list[float] = []
        finish = "length"
        for _ in range(sp.max_new_tokens):
            tok, lp = _sample(logits.cpu(), sp, gen)
            out_ids.append(tok)
            out_lp.append(lp)
            if on_token is not None:
                on_token(tok)
            if tok in sp.stop_token_ids:
                finish = "stop"
                break
            h = self._decode_step(torch.tensor(tok, device=dev), pos, states)
            pos += 1
            logits = self.model.logits(h[:, -1])[0]
        return GenerationResult(out_ids, out_lp, finish, len(input_ids))

    @torch.no_grad()
    def prompt_logprobs(
        self,
        input_ids: list[int],
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
    ) -> list[float]:
        """log P(token_t | <t) for t >= 1 (length len(input_ids) - 1)."""
        dev = next(self.model.parameters()).device
        ids = torch.tensor([input_ids], device=dev)
        logits = self.model(ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        logp = F.log_softmax(logits[0, :-1].float(), -1)
        return logp.gather(1, ids[0, 1:].unsqueeze(-1)).squeeze(-1).tolist()
