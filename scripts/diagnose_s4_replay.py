"""Replay untouched S4 text groups with installed MLX execution paths."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from probe_s4_training import complete_group, require_stopped_server
from safetensors.torch import load_file

from nano_areal.archon.s4_update import trajectory_from_turn
from nano_areal.sync.policy import PolicyBinding, file_digest


def response_span(item):
    trajectory, old = trajectory_from_turn(item)
    positions = torch.nonzero(trajectory.response_mask, as_tuple=True)[0].tolist()
    if positions != list(range(positions[0], len(item["input_ids"]))):
        raise ValueError("Replay requires a contiguous response suffix")
    return positions[0], old[trajectory.response_mask].tolist()


def require_zero_adapter(adapter):
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    sides = {side: [value for name, value in weights.items()
                    if name.endswith(f".lora_{side}.weight")] for side in ("A", "B")}
    if (not sides["A"] or len(sides["A"]) != len(sides["B"])
            or len(weights) != len(sides["A"]) * 2
            or not all(bool(torch.isfinite(value).all()) for value in weights.values())
            or any(bool(value.any()) for value in sides["B"])):
        raise ValueError("Base-only replay requires a finite, exactly zero-B adapter")


def common_prefix(previous, current, limit, block_size):
    matched = 0
    for left, right in zip(previous[:limit], current[:limit]):
        if left != right:
            break
        matched += 1
    return matched // block_size * block_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--group", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execution", choices=("full", "cached", "paged"), required=True,
    )
    parser.add_argument("--prefix-reuse", action="store_true")
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Diagnostic evidence must not be overwritten")
    if args.block_size < 16 or args.block_size % 16:
        parser.error("Paged block size must be a positive multiple of 16")
    if args.prefix_reuse and args.execution != "paged":
        parser.error("Prefix reuse is supported only by the paged replay")
    require_stopped_server()
    binding = PolicyBinding.read(args.policy)
    binding.verify_files()
    episodes = complete_group(args.group, binding)
    for episode in episodes:
        for item in episode["turns"]:
            response_span(item)
    require_zero_adapter(Path(binding.manifest["adapter"]))
    evidence = [args.policy, args.group / "report.json", *[
        args.group / f"episode-{index}.json" for index in range(len(episodes))
    ]]
    hashes = {str(path): file_digest(path) for path in evidence}
    report = {
        "scope": "Unbound teacher-forced diagnostic; not policy acceptance",
        "status": "running", "execution": args.execution,
        "prefix_reuse": args.prefix_reuse, "block_size": args.block_size,
        "policy": binding.model, "updates": 0, "evidence_sha256": hashes,
        "source_sha256": {str(Path(__file__)): file_digest(Path(__file__))},
        "planned_turns": sum(len(episode["turns"]) for episode in episodes),
        "turns": [], "max_logprob_mae": 0.01,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    started = time.monotonic()
    try:
        import mlx.core as mx
        import mlx_vlm.models.qwen3_vl.language as language_source
        from mlx_vlm.models.cache import KVCache
        from mlx_vlm.utils import load_model

        model_roots = {Path(name).parent for name in binding.manifest[
            "inference_files"] if Path(name).name == "config.json"}
        if len(model_roots) != 1:
            raise ValueError("Replay requires one bound inference checkpoint")
        model_path = model_roots.pop()
        model = load_model(model_path).language_model
        if not model.args.tie_word_embeddings:
            raise ValueError("Replay requires the bound tied output head")
        report["packages"] = {name: importlib.metadata.version(name)
                              for name in ("mlx", "mlx-vlm", "vllm-metal")}
        report["source_sha256"][language_source.__file__] = file_digest(
            Path(language_source.__file__))
        paged = None
        if args.execution == "paged":
            from vllm_metal.attention import context
            from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
            from vllm_metal.attention.impls import sdpa, sdpa_wrapper

            config = model.args
            max_length = max(len(item["input_ids"]) for episode in episodes
                             for item in episode["turns"])
            blocks = (max_length + args.block_size - 1) // args.block_size
            paged = MetalPagedKVCache(
                num_layers=len(model.layers), num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim, num_blocks=blocks, block_size=args.block_size,
                dtype=mx.bfloat16,
            )
            patched = sdpa_wrapper.patch_sdpa_attention(model, paged, args.block_size)
            if patched != len(model.layers):
                raise ValueError("Paged replay did not wrap every attention layer")
            report["patched_layers"] = patched
            for module in (context, sdpa, sdpa_wrapper):
                report["source_sha256"][module.__file__] = file_digest(
                    Path(module.__file__))
            metal_root = Path(sdpa.__file__).parents[2] / "metal"
            for pattern in ("*.so", "*.metallib", "kernels_v2/*.metal"):
                for source in sorted(metal_root.glob(pattern)):
                    report["source_sha256"][str(source)] = file_digest(source)

        previous = []
        for episode_index, episode in enumerate(episodes):
            for turn_index, item in enumerate(episode["turns"]):
                ids = item["input_ids"]
                prompt, captured = response_span(item)
                reused = common_prefix(previous, ids, prompt - 1, args.block_size
                                       ) if args.prefix_reuse else 0
                cache = ([KVCache() for _ in model.layers]
                         if args.execution == "cached" else None)
                current = []
                segments = [(0, len(ids) - 1)] if args.execution == "full" else [
                    (reused, prompt), *[(index, index + 1)
                                       for index in range(prompt, len(ids) - 1)]
                ]
                for start, end in segments:
                    inputs = mx.array([ids[start:end]], dtype=mx.int32)
                    positions = mx.broadcast_to(mx.arange(start, end)[None, None, :],
                                                (3, 1, end - start))
                    if paged is not None:
                        block_ids = list(range(blocks))
                        if end == prompt:
                            context.prepare_unified(
                                [], [(block_ids, end - start, start)], args.block_size)
                        else:
                            context.prepare_unified(
                                [(block_ids, start)], [], args.block_size)
                    try:
                        hidden = model.model(
                            inputs, cache=cache, position_ids=positions)
                        if args.execution != "full":
                            hidden = hidden[:, -1:]
                        logits = model.model.embed_tokens.as_linear(hidden).astype(
                            mx.float32)
                        labels = (ids[prompt:] if args.execution == "full"
                                  else [ids[end]])
                        if args.execution == "full":
                            logits = logits[:, prompt - 1:]
                        scores = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                        selected = mx.take_along_axis(
                            scores, mx.array([labels])[..., None], -1)
                        current.extend(selected.reshape(-1).tolist())
                        if paged is not None:
                            mx.eval(paged.key_caches, paged.value_caches)
                    finally:
                        if paged is not None:
                            context.clear_context()
                if len(current) != len(captured):
                    raise ValueError("Replay response token count mismatch")
                error = (torch.tensor(current) - torch.tensor(captured)).abs()
                if not bool(torch.isfinite(error).all()):
                    raise ValueError("Nonfinite replay logprobs")
                row = {"episode": episode_index, "turn": turn_index,
                       "tokens": len(current), "prompt_tokens": prompt,
                       "reused_tokens": reused, "mae": float(error.mean()),
                       "max": float(error.max()), "first_error": float(error[0]),
                       "absolute_error_sum": float(error.sum())}
                report["turns"].append(row)
                report["response_tokens"] = sum(
                    row["tokens"] for row in report["turns"])
                report["mae"] = sum(row["absolute_error_sum"] for row in report["turns"]
                                    ) / report["response_tokens"]
                report["elapsed_seconds"] = time.monotonic() - started
                save()
                print(json.dumps(row), flush=True)
                previous = ids[:-1]
                mx.clear_cache()
        binding.verify_files()
        if hashes != {str(path): file_digest(path) for path in evidence}:
            raise ValueError("Captured evidence changed during replay")
        report.update(status="diagnostic-complete", within_bound=report["mae"] <= 0.01)
        save()
    except BaseException as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        save()
        raise
    print(json.dumps({key: value for key, value in report.items() if key != "turns"}))


if __name__ == "__main__":
    main()
