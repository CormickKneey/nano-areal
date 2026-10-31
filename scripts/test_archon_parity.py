"""Native Qwen3-VL contracts and sequential real-checkpoint bf16 parity probe."""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import torch
from PIL import Image
from probe_mac_vlm import memory_snapshot
from torch.nn import functional as F
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from nano_areal.archon.qwen3vl import Qwen3VLModel, add_lora, export_mlx_lora
from nano_areal.archon.state_dict import qwen3vl_archon_key, qwen3vl_hf_key


def tiny_config():
    return Qwen3VLConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 128,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "depth": 3,
            "num_heads": 4,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0, 1],
        },
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
        tie_word_embeddings=True,
    )


class ArchonParityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        config = tiny_config()
        config._attn_implementation = "sdpa"
        self.reference = Qwen3VLForConditionalGeneration(config).eval()
        self.model = Qwen3VLModel(config.to_dict()).eval()
        weights = {
            qwen3vl_hf_key(key): value
            for key, value in self.reference.state_dict().items()
            if key != "lm_head.weight"
        }
        self.model.load_state_dict(weights, strict=True)

    def test_text_and_rectangular_image_match_hf(self):
        for with_image in (False, True):
            ids = torch.tensor(
                [[3, 122] + ([120] * 6 if with_image else [4]) + [123, 5]]
            )
            inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            if with_image:
                inputs.update(
                    pixel_values=torch.randn(24, 24),
                    image_grid_thw=torch.tensor([[1, 4, 6]]),
                )
            types = (ids == 120).long()
            with torch.no_grad():
                expected = self.reference(
                    **inputs, mm_token_type_ids=types, use_cache=False
                ).logits
                actual = self.model(**inputs)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)

    def test_image_contracts_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "require pixels"):
            self.model(torch.tensor([[120]]))
        with self.assertRaisesRegex(ValueError, "Video"):
            self.model(torch.tensor([[121]]))
        with self.assertRaisesRegex(ValueError, "feature count"):
            self.model(
                torch.tensor([[3]]),
                pixel_values=torch.randn(16, 24),
                image_grid_thw=torch.tensor([[1, 4, 4]]),
            )

    def test_lora_freezes_base_and_updates(self):
        ids = torch.tensor([[3, 4, 5, 6]])
        with torch.no_grad():
            before = self.model(ids)
        parameters = add_lora(self.model, rank=2, alpha=4)
        torch.testing.assert_close(self.model(ids), before, atol=0, rtol=0)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in self.model.visual.parameters()
            )
        )
        self.assertEqual(len(parameters), 12)
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        loss = self.model(ids).float().square().mean()
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in parameters
            )
        )
        optimizer.step()
        self.assertGreater(float((self.model(ids) - before).abs().sum().detach()), 0)

    def test_unknown_state_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unmapped"):
            qwen3vl_hf_key("model.unexpected.weight")

    def test_strict_checkpoint_roundtrip(self):
        from safetensors.torch import save_file

        state = {
            qwen3vl_archon_key(key): value
            for key, value in self.model.state_dict().items()
        }
        self.assertEqual(
            {qwen3vl_hf_key(key): value for key, value in state.items()}.keys(),
            self.model.state_dict().keys(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(self.model.config))
            save_file(state, str(root / "model.safetensors"))
            restored = Qwen3VLModel.from_checkpoint(root, dtype=torch.float32)
            for key, tensor in self.model.state_dict().items():
                torch.testing.assert_close(
                    restored.state_dict()[key], tensor, atol=0, rtol=0
                )
            state.pop(next(iter(state)))
            save_file(state, str(root / "model.safetensors"))
            with self.assertRaisesRegex(ValueError, "Missing"):
                Qwen3VLModel.from_checkpoint(root)

    def test_adapter_context_and_mlx_export(self):
        from safetensors.torch import load_file

        ids = torch.tensor([[3, 4, 5]])
        baseline = self.model(ids).detach()
        add_lora(self.model, rank=2, alpha=4)
        with torch.no_grad():
            self.model.language_model.layers[0].attn.q_proj.lora_b.fill_(0.5)
        with self.model.disable_adapter():
            torch.testing.assert_close(self.model(ids), baseline, atol=0, rtol=0)
            with self.model.disable_adapter():
                self.assertFalse(
                    self.model.language_model.layers[0].attn.q_proj.enabled
                )
            self.assertFalse(self.model.language_model.layers[0].attn.q_proj.enabled)
        self.assertTrue(self.model.language_model.layers[0].attn.q_proj.enabled)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            export_mlx_lora(self.model, path)
            weights = load_file(str(path))
            key = "language_model.model.layers.0.self_attn.q_proj.lora_b"
            expected = (
                self.model.language_model.layers[0]
                .attn.q_proj.lora_b.detach()
                .bfloat16()
                .T
            )
            torch.testing.assert_close(weights[key], expected, atol=0, rtol=0)


def clear_memory(device):
    gc.collect()
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()


def snapshot(device):
    result = memory_snapshot(os.getpid())
    if device == "mps":
        torch.mps.synchronize()
        result.update(
            mps_driver_bytes=torch.mps.driver_allocated_memory(),
            mps_allocated_bytes=torch.mps.current_allocated_memory(),
        )
    return result


def probe(args):
    started = time.monotonic()
    report = {
        "status": "failed",
        "dtype": "bfloat16",
        "atol": args.atol,
        "rtol": 0,
        "device": args.device,
        "memory_before": snapshot(args.device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        processor = AutoProcessor.from_pretrained(
            args.model_path, local_files_only=True
        )
        batches = {}
        for name in ("text", "image"):
            content = [{"type": "text", "text": "Name the color in one word."}]
            if name == "image":
                content.insert(
                    0, {"type": "image", "image": Image.new("RGB", (64, 64), "red")}
                )
            batch = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            batches[name] = {
                key: value.to(args.device)
                for key, value in batch.items()
                if key
                in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
            }
        reference = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map=args.device,
            attn_implementation="sdpa",
        ).eval()
        expected = {}
        expected_positions = {}
        with torch.no_grad():
            for name, batch in batches.items():
                types = (batch["input_ids"] == reference.config.image_token_id).long()
                expected[name] = (
                    reference(**batch, mm_token_type_ids=types, use_cache=False)
                    .logits.float()
                    .cpu()
                )
                expected_positions[name] = reference.model.get_rope_index(
                    input_ids=batch["input_ids"],
                    mm_token_type_ids=types,
                    image_grid_thw=batch.get("image_grid_thw"),
                    attention_mask=batch["attention_mask"],
                )[0].cpu()
        report["reference_memory"] = snapshot(args.device)
        del reference
        clear_memory(args.device)
        model = Qwen3VLModel.from_checkpoint(args.model_path, args.device).eval()
        report["native_load_memory"] = snapshot(args.device)
        report["parity"] = {}
        with torch.no_grad():
            for name, batch in batches.items():
                positions = model.position_ids(
                    batch["input_ids"], batch.get("image_grid_thw")
                )
                torch.testing.assert_close(
                    positions.cpu(), expected_positions[name], atol=0, rtol=0
                )
                actual = model(**batch).float().cpu()
                difference = (actual - expected[name]).abs()
                report["parity"][name] = {
                    "max_abs_error": float(difference.max()),
                    "mean_abs_error": float(difference.mean()),
                    "prompt_tokens": batch["input_ids"].numel(),
                    "passed": bool((difference <= args.atol).all()),
                }
        if not all(result["passed"] for result in report["parity"].values()):
            raise AssertionError("Real checkpoint bf16 parity exceeds tolerance")
        parameters = add_lora(model)
        model.train()
        model.gradient_checkpointing = True
        optimizer = torch.optim.AdamW(parameters, lr=1e-4)
        report["training"] = []
        for step in range(args.steps):
            batch = batches["image" if step % 2 else "text"]
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch, logits_to_keep=2).float()
            label = batch["input_ids"][:, -1:]
            loss = F.cross_entropy(
                logits[:, :1].reshape(-1, logits.shape[-1]), label.flatten()
            )
            loss.backward()
            if not all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in parameters
            ):
                raise ValueError("Missing or nonfinite native LoRA gradients")
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            report["training"].append(
                {
                    "step": step + 1,
                    "input": "image" if step % 2 else "text",
                    "loss": float(loss.detach()),
                    "gradient_norm": float(norm),
                    "memory": snapshot(args.device),
                }
            )
        report["trainable_parameters"] = sum(
            parameter.numel() for parameter in parameters
        )
        report["frozen_vit"] = all(
            not parameter.requires_grad for parameter in model.visual.parameters()
        )
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=".s1-cache/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/s2/archon-parity.json")
    )
    probe(parser.parse_args())
