"""Qwen3-VL LoRA GRPO trainers with explicit HF and native archon backends."""

from __future__ import annotations

import re
from pathlib import Path

import torch

from nano_areal.archon.losses import compute_policy_loss, group_advantages


class Qwen3VLLoRATrainer:
    def __init__(self, model_path: Path, lr: float = 1e-4):
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3VLForConditionalGeneration

        self.model = get_peft_model(
            Qwen3VLForConditionalGeneration.from_pretrained(
                str(model_path.resolve()),
                local_files_only=True,
                dtype=torch.bfloat16,
                device_map="mps",
            ),
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)",
            ),
        )
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        self.model.config.use_cache = False
        self.parameters = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        if not self.parameters or not all(
            ".language_model." in name and ".lora_" in name for name in self.parameters
        ):
            raise ValueError("Only language-model LoRA may be trainable")
        if any(
            parameter.requires_grad
            for name, parameter in self.model.named_parameters()
            if ".visual." in name
        ):
            raise ValueError("ViT must remain frozen")
        if any(parameter.device.type != "mps" for parameter in self.model.parameters()):
            raise ValueError("All parameters must reside on MPS")
        if self.model.get_input_embeddings().weight.dtype != torch.bfloat16:
            raise ValueError("Training base must be bf16")
        self.optimizer = torch.optim.AdamW(list(self.parameters.values()), lr=lr)

    def response_logprobs(self, trajectory):
        ids = trajectory.input_ids.to("mps").unsqueeze(0)
        positions = torch.nonzero(trajectory.response_mask[1:], as_tuple=True)[0].to(
            "mps"
        )
        logits = self.model(
            input_ids=ids,
            use_cache=False,
            logits_to_keep=positions,
        ).logits.float()
        labels = ids[:, 1:][:, positions]
        return logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    def train_step(self, trajectories) -> dict:
        rewards = torch.tensor([item.reward for item in trajectories], device="mps")
        group_ids = [item.group_id for item in trajectories]
        advantages, _ = group_advantages(rewards, None, group_ids=group_ids)
        total_tokens = sum(int(item.response_mask.sum()) for item in trajectories)
        if not total_tokens:
            raise ValueError("No response tokens in training batch")
        self.optimizer.zero_grad(set_to_none=True)
        previous = {
            name: value.detach().cpu().float().clone()
            for name, value in self.parameters.items()
        }
        total_loss, ratios, drift, policy_logprobs = 0.0, [], [], []
        for index, item in enumerate(trajectories):
            self.model.eval()
            with torch.no_grad(), self.model.disable_adapter():
                reference = self.response_logprobs(item)
            self.model.train()
            current = self.response_logprobs(item)
            mask = item.response_mask.bool()
            old = item.old_log_probs[mask].to("mps").unsqueeze(0)
            loss, stats = compute_policy_loss(
                "grpo",
                current,
                reference,
                rewards[index : index + 1],
                torch.ones_like(current),
                old_log_probs=old,
                kl_coef=0.01,
                advantages=advantages[index : index + 1],
            )
            weight = current.numel() / total_tokens
            (loss * weight).backward()
            total_loss += float(loss.detach()) * weight
            ratios.append(stats["ratio_mean"])
            drift.append(float((current.detach() - old).abs().mean()))
            policy_logprobs.append(current.detach().cpu().tolist()[0])
        if not all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in self.parameters.values()
        ):
            raise ValueError("Missing or nonfinite LoRA gradients")
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(list(self.parameters.values()), 1.0)
        )
        self.optimizer.step()
        torch.mps.synchronize()
        delta = sum(
            float((value.detach().cpu().float() - previous[name]).abs().sum())
            for name, value in self.parameters.items()
        )
        if not delta or not grad_norm:
            raise ValueError("GRPO update has zero gradient or zero adapter delta")
        return {
            "loss": total_loss,
            "rewards": rewards.cpu().tolist(),
            "advantages": advantages.cpu().tolist(),
            "grad_norm": grad_norm,
            "adapter_delta_l1": delta,
            "response_tokens": total_tokens,
            "behavior_logprob_mae": sum(drift) / len(drift),
            "ratio_mean": sum(ratios) / len(ratios),
            "policy_logprobs_before_update": policy_logprobs,
            "trainable_parameters": sum(
                value.numel() for value in self.parameters.values()
            ),
            "base_dtype": str(self.model.get_input_embeddings().weight.dtype),
            "freeze_vit": True,
            "reference": "frozen bf16 base, adapters disabled, no second model",
            "mps_allocated_bytes": torch.mps.current_allocated_memory(),
            "mps_driver_bytes": torch.mps.driver_allocated_memory(),
        }

    def export_mlx_adapter(self, path: Path):
        from safetensors.torch import save_file

        weights = {}
        for name, parameter in self.parameters.items():
            match = re.search(
                r"language_model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)"
                r"\.lora_([AB])\.default\.weight$",
                name,
            )
            if match is None:
                raise ValueError(f"Unsupported adapter parameter: {name}")
            index, projection, side = match.groups()
            key = (
                f"language_model.model.layers.{index}.self_attn.{projection}"
                f".lora_{side.lower()}"
            )
            weights[key] = parameter.detach().cpu().to(torch.bfloat16).T.contiguous()
        temporary = path.with_suffix(".tmp.safetensors")
        save_file(weights, str(temporary))
        temporary.replace(path)


class NativeQwen3VLLoRATrainer(Qwen3VLLoRATrainer):
    def __init__(self, model_path: Path, lr: float = 1e-4):
        from nano_areal.archon.qwen3vl import Qwen3VLModel, add_lora

        self.model = Qwen3VLModel.from_checkpoint(model_path, "mps")
        add_lora(self.model)
        self.model.gradient_checkpointing = True
        self.parameters = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.optimizer = torch.optim.AdamW(list(self.parameters.values()), lr=lr)

    def response_logprobs(self, trajectory):
        ids = trajectory.input_ids.to("mps").unsqueeze(0)
        positions = torch.nonzero(trajectory.response_mask[1:], as_tuple=True)[0].to(
            "mps"
        )
        logits = self.model(input_ids=ids, logits_to_keep=positions).float()
        labels = ids[:, 1:][:, positions]
        return logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    def export_mlx_adapter(self, path: Path):
        from nano_areal.archon.qwen3vl import export_mlx_lora

        export_mlx_lora(self.model, path)
