"""Export the served MLX affine base as native HF-layout floating weights."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from nano_areal.sync.policy import checkpoint_files


def hf_key(key):
    if key.startswith("language_model.model."):
        return "model.language_model." + key.removeprefix("language_model.model.")
    if key.startswith("vision_tower."):
        return "model.visual." + key.removeprefix("vision_tower.")
    raise ValueError(f"Unsupported MLX checkpoint key: {key}")


def convert(source, output):
    import mlx.core as mx

    config = json.loads((source / "config.json").read_text())
    quantization = config.pop("quantization")
    config.pop("quantization_config", None)
    if quantization != {"group_size": 64, "bits": 4, "mode": "affine"}:
        raise ValueError("Only the measured 4-bit affine base is supported")
    source_files = checkpoint_files(source)
    output.mkdir(parents=True, exist_ok=False)
    arrays = {}
    for shard in sorted(source.glob("*.safetensors")):
        arrays.update(mx.load(str(shard)))
    consumed = set()
    weight_map = {}
    batch = {}
    batch_bytes = 0
    shard_index = 0
    quantized = 0

    def flush():
        nonlocal batch, batch_bytes, shard_index
        if not batch:
            return
        name = f"aligned-{shard_index:04d}.safetensors"
        mx.save_safetensors(str(output / name), batch)
        weight_map.update({key: name for key in batch})
        shard_index += 1
        batch, batch_bytes = {}, 0
        mx.clear_cache()

    for key in sorted(arrays):
        if key in consumed or key.endswith((".scales", ".biases")):
            continue
        tensor = arrays[key]
        prefix = key.removesuffix(".weight")
        if key.endswith(".weight") and prefix + ".scales" in arrays:
            tensor = mx.dequantize(
                tensor, arrays[prefix + ".scales"], arrays[prefix + ".biases"],
                **quantization,
            )
            consumed.update({prefix + ".scales", prefix + ".biases"})
            quantized += 1
        if key == "vision_tower.patch_embed.proj.weight":
            tensor = tensor.transpose(0, 4, 1, 2, 3)
        mapped = hf_key(key)
        if mapped in weight_map or mapped in batch:
            raise ValueError(f"Duplicate converted key: {mapped}")
        mx.eval(tensor)
        batch[mapped] = tensor
        batch_bytes += tensor.nbytes
        consumed.add(key)
        if batch_bytes >= 512 * 1024 * 1024:
            flush()
    flush()
    if consumed != set(arrays):
        raise ValueError(f"Unconsumed quantization tensors: {set(arrays) - consumed}")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}, indent=2) + "\n"
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "preprocessor_config.json", "generation_config.json"):
        if (source / name).exists():
            shutil.copyfile(source / name, output / name)
    if checkpoint_files(source) != source_files:
        raise ValueError("Source checkpoint changed during conversion")
    report = {
        "source_files": source_files, "output_files": checkpoint_files(output),
        "quantization": quantization, "dequantized_modules": quantized,
        "tensor_count": len(weight_map),
        "scope": "Weight alignment only; numerical behavior parity unverified",
    }
    (output / "alignment.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ("quantization", "dequantized_modules", "tensor_count", "scope")
    print(json.dumps({key: report[key] for key in fields}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
