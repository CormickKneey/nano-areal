"""
S1 probe: peak training memory of the nano-archon model vs. sequence length.

Builds the model from a HF config.json (random init, no weights needed),
runs forward + backward on one sequence per length and reports the peak
allocated memory. Used to decide max_model_len for dsh trajectories
(docs/v0.2-plan.md §7) before real training.

    uv run python scripts/probe_train_memory.py --model Qwen/Qwen3-0.6B \
        --seq-lens 4096 8192 16384 32768 [--no-checkpointing] [--optimizer]

Notes:
  * Measures the archon implementation that exists today (dense Qwen3);
    the Qwen3.5 hybrid block lands in S2 and should be re-probed.
  * `--optimizer` also allocates AdamW state (fp32 m/v) to approximate the
    full training footprint; omit it to measure activations only.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nano_areal.archon.model import Qwen35Args, build_model


def _resolve(model: str) -> str:
    if Path(model).is_dir():
        return model
    try:
        from modelscope.hub.snapshot_download import snapshot_download

        return snapshot_download(model, allow_patterns=["config.json", "*.json"])
    except Exception:
        from huggingface_hub import snapshot_download

        return snapshot_download(model, allow_patterns=["config.json"])


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _peak_reset(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _peak_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated()
    if device.type == "mps":
        return torch.mps.driver_allocated_memory()
    return 0


def run_one(
    model, seq_len: int, device: torch.device, use_ckpt: bool, optimizer
) -> dict:
    ids = torch.randint(0, model.args.vocab_size, (1, seq_len), device=device)
    mask = torch.ones_like(ids, dtype=torch.float)
    mask[:, : seq_len // 2] = 0  # half prompt, half response

    _peak_reset(device)
    t0 = time.perf_counter()

    if use_ckpt:
        # Per-layer activation checkpointing (mirrors gradient_checkpointing=True)
        B, T = ids.shape
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = model.embed_tokens(ids)
        cos, sin = model.rope(positions, seq_len=T, dtype=x.dtype)
        for layer in model.layers:
            x = checkpoint(layer, x, cos, sin, None, use_reentrant=False)
        x = model.norm(x)
        logits = (
            model.lm_head(x)
            if model.lm_head is not None
            else F.linear(x, model.embed_tokens.weight)
        )
        logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        tok = logp.gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        loss = -(tok * mask[:, 1:]).sum() / mask[:, 1:].sum()
    else:
        loss = -model.compute_log_probs(ids, mask).sum() / mask[:, 1:].sum()

    loss.backward()
    if optimizer is not None:
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = _peak_bytes(device)
    model.zero_grad(set_to_none=True)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    return {"seq_len": seq_len, "peak_gb": peak / 2**30, "fwd_bwd_s": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--seq-lens", type=int, nargs="+", default=[4096, 8192, 16384, 32768]
    )
    ap.add_argument("--no-checkpointing", action="store_true")
    ap.add_argument("--optimizer", action="store_true", help="include AdamW state")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args()

    device = _device()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    path = _resolve(args.model)
    margs = Qwen35Args.from_hf_config(path)
    model = build_model(margs, device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"model={args.model} params={n_params / 1e9:.2f}B "
        f"device={device} dtype={args.dtype} "
        f"checkpointing={not args.no_checkpointing} optimizer={args.optimizer}"
    )

    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=1e-6) if args.optimizer else None
    )

    results = []
    for L in args.seq_lens:
        try:
            r = run_one(model, L, device, not args.no_checkpointing, optimizer)
            print(
                f"  seq_len={L:6d}  peak={r['peak_gb']:6.2f} GB  "
                f"fwd+bwd={r['fwd_bwd_s']:.2f}s"
            )
        except (torch.OutOfMemoryError, RuntimeError) as e:  # MPS raises RuntimeError
            if "out of memory" not in str(e).lower() and not isinstance(
                e, torch.OutOfMemoryError
            ):
                raise
            r = {"seq_len": L, "peak_gb": None, "error": "OOM"}
            print(f"  seq_len={L:6d}  OOM")
            _peak_reset(device)
        results.append(r)

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "model": args.model,
                    "params": n_params,
                    "device": str(device),
                    "checkpointing": not args.no_checkpointing,
                    "optimizer": args.optimizer,
                    "results": results,
                },
                indent=2,
            )
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
