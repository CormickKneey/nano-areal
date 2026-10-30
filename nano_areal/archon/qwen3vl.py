"""Native single-image Qwen3-VL, independent of Transformers forward methods."""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nano_areal.archon.model import Qwen35Args, RMSNorm, TransformerBlock, _rotate_half


class InterleavedMRoPE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        head_dim = config["head_dim"]
        rope = config.get("rope_parameters", config.get("rope_scaling", {}))
        if rope.get("rope_type", "default") != "default":
            raise ValueError("Only default Qwen3-VL interleaved M-RoPE is supported")
        self.sections = rope.get("mrope_section", [24, 20, 20])
        theta = rope.get("rope_theta", config.get("rope_theta", 5000000))
        self.register_buffer(
            "inv_freq",
            1 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim),
            persistent=False,
        )

    def forward(self, positions: Tensor, dtype: torch.dtype):
        frequencies = positions[..., None].float() * self.inv_freq.float()
        interleaved = frequencies[0].clone()
        for axis in (1, 2):
            selected = slice(axis, self.sections[axis] * 3, 3)
            interleaved[..., selected] = frequencies[axis, ..., selected]
        angles = torch.cat((interleaved, interleaved), dim=-1)
        return angles.cos().to(dtype).unsqueeze(2), angles.sin().to(dtype).unsqueeze(2)


class VisionPatchEmbed(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.shape = (
            config["in_channels"],
            config["temporal_patch_size"],
            config["patch_size"],
            config["patch_size"],
        )
        self.proj = nn.Conv3d(
            self.shape[0],
            config["hidden_size"],
            kernel_size=self.shape[1:],
            stride=self.shape[1:],
        )

    def forward(self, pixels: Tensor):
        return self.proj(
            pixels.reshape(-1, *self.shape).to(self.proj.weight.dtype)
        ).flatten(1)


class VisionAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden = config["hidden_size"]
        self.heads = config["num_heads"]
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, hidden: Tensor, cos: Tensor, sin: Tensor):
        length = hidden.shape[0]
        query, key, value = (
            self.qkv(hidden)
            .reshape(length, 3, self.heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query = (
            query.float() * cos[:, None] + _rotate_half(query.float()) * sin[:, None]
        ).to(hidden.dtype)
        key = (
            key.float() * cos[:, None] + _rotate_half(key.float()) * sin[:, None]
        ).to(hidden.dtype)
        attended = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            is_causal=False,
        )
        return self.proj(attended.squeeze(0).transpose(0, 1).reshape(length, -1))


class VisionMLP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.linear_fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.linear_fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.approximate = (
            "tanh" if config["hidden_act"] == "gelu_pytorch_tanh" else "none"
        )

    def forward(self, hidden):
        return self.linear_fc2(
            F.gelu(self.linear_fc1(hidden), approximate=self.approximate)
        )


class VisionBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.norm1 = nn.LayerNorm(config["hidden_size"], eps=1e-6)
        self.norm2 = nn.LayerNorm(config["hidden_size"], eps=1e-6)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(self, hidden, cos, sin):
        hidden = hidden + self.attn(self.norm1(hidden), cos, sin)
        return hidden + self.mlp(self.norm2(hidden))


class VisionMerger(nn.Module):
    def __init__(self, config: dict, postshuffle: bool):
        super().__init__()
        self.width = config["hidden_size"] * config["spatial_merge_size"] ** 2
        self.postshuffle = postshuffle
        self.norm = nn.LayerNorm(
            self.width if postshuffle else config["hidden_size"], eps=1e-6
        )
        self.linear_fc1 = nn.Linear(self.width, self.width)
        self.linear_fc2 = nn.Linear(self.width, config["out_hidden_size"])

    def forward(self, hidden):
        if self.postshuffle:
            hidden = hidden.reshape(-1, self.width)
        hidden = self.norm(hidden).reshape(-1, self.width)
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden)))


class VisionTransformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.patch_embed = VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(
            config["num_position_embeddings"], config["hidden_size"]
        )
        self.blocks = nn.ModuleList(
            [VisionBlock(config) for _ in range(config["depth"])]
        )
        self.merger = VisionMerger(config, False)
        self.deepstack_merger_list = nn.ModuleList(
            [VisionMerger(config, True) for _ in config["deepstack_visual_indexes"]]
        )

    def forward(self, pixels: Tensor, grid_thw: Tensor):
        if grid_thw.shape != (1, 3) or int(grid_thw[0, 0]) != 1:
            raise ValueError("Native archon currently supports exactly one still image")
        height, width = (int(size) for size in grid_thw[0, 1:])
        merge = self.config["spatial_merge_size"]
        if height % merge or width % merge or pixels.shape[0] != height * width:
            raise ValueError("Invalid image grid or patch count")
        device = pixels.device
        rows, columns = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )

        def block_order(values):
            return (
                values.reshape(height // merge, merge, width // merge, merge)
                .permute(0, 2, 1, 3)
                .reshape(-1)
            )

        rows, columns = block_order(rows), block_order(columns)
        side = math.isqrt(self.config["num_position_embeddings"])

        def interpolation_axis(coordinates, size):
            source = coordinates.float() * (side - 1) / max(size - 1, 1)
            lower = source.floor()
            offsets = torch.arange(2, device=device)
            taps = (lower.long()[:, None] + offsets).clamp(0, side - 1)
            distance = (source[:, None] - lower[:, None] - offsets).abs()
            return taps, (1 - distance).clamp(min=0)

        row_taps, row_weights = interpolation_axis(rows, height)
        column_taps, column_weights = interpolation_axis(columns, width)
        indices = (row_taps[:, :, None] * side + column_taps[:, None, :]).reshape(-1, 4)
        weights = (row_weights[:, :, None] * column_weights[:, None, :]).reshape(-1, 4)
        hidden = self.patch_embed(pixels)
        hidden = hidden + (self.pos_embed(indices) * weights[..., None]).sum(1).to(
            hidden.dtype
        )
        head_dim = self.config["hidden_size"] // self.config["num_heads"]
        inverse = 1 / 10000 ** (
            torch.arange(0, head_dim // 2, 2, device="cpu").float() / (head_dim // 2)
        )
        inverse = inverse.to(device)
        angles = torch.cat(
            (rows[:, None] * inverse, columns[:, None] * inverse), dim=-1
        )
        angles = torch.cat((angles, angles), dim=-1)
        cos, sin = angles.cos(), angles.sin()
        deepstack = []
        indexes = self.config["deepstack_visual_indexes"]
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, cos, sin)
            if index in indexes:
                deepstack.append(
                    self.deepstack_merger_list[indexes.index(index)](hidden)
                )
        return self.merger(hidden), deepstack


class NativeLanguageModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        args = Qwen35Args(
            dim=config["hidden_size"],
            n_layers=config["num_hidden_layers"],
            n_heads=config["num_attention_heads"],
            n_kv_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            intermediate_size=config["intermediate_size"],
            vocab_size=config["vocab_size"],
            rms_norm_eps=config["rms_norm_eps"],
            attention_bias=config.get("attention_bias", False),
        )
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(args) for _ in range(args.n_layers)]
        )
        self.norm = RMSNorm(args.dim, args.rms_norm_eps)
        self.rope = InterleavedMRoPE(config)


class Qwen3VLModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        if config.get("model_type") != "qwen3_vl":
            raise ValueError("Expected a Qwen3-VL dense checkpoint")
        self.config = config
        self.language_model = NativeLanguageModel(config["text_config"])
        self.visual = VisionTransformer(config["vision_config"])
        self.tied = config.get("tie_word_embeddings", False)
        self.lm_head = (
            None
            if self.tied
            else nn.Linear(
                config["text_config"]["hidden_size"],
                config["text_config"]["vocab_size"],
                bias=False,
            )
        )
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    @contextmanager
    def disable_adapter(self):
        adapters = [
            module for module in self.modules() if isinstance(module, LoRALinear)
        ]
        previous = [module.enabled for module in adapters]
        try:
            for module in adapters:
                module.enabled = False
            yield
        finally:
            for module, enabled in zip(adapters, previous, strict=True):
                module.enabled = enabled

    def position_ids(self, input_ids: Tensor, grid: Tensor | None = None):
        batch, length = input_ids.shape
        positions = (
            torch.arange(length, device=input_ids.device)
            .expand(3, batch, length)
            .clone()
        )
        if grid is None:
            return positions
        if batch != 1 or grid.shape != (1, 3) or int(grid[0, 0]) != 1:
            raise ValueError("Image batches and video are not supported")
        mask = input_ids[0] == self.config["image_token_id"]
        selected = mask.nonzero().flatten()
        merge = self.config["vision_config"]["spatial_merge_size"]
        height, width = (int(size) // merge for size in grid[0, 1:])
        if selected.numel() != height * width:
            raise ValueError("Image placeholder count differs from image grid")
        start = int(selected[0])
        if not torch.equal(
            selected,
            torch.arange(start, start + height * width, device=input_ids.device),
        ):
            raise ValueError("Image placeholders must be contiguous")
        positions[0, 0, selected] = start
        positions[1, 0, selected] = (
            torch.arange(height, device=input_ids.device).repeat_interleave(width)
            + start
        )
        positions[2, 0, selected] = (
            torch.arange(width, device=input_ids.device).repeat(height) + start
        )
        end = start + height * width
        positions[:, :, end:] -= height * width - max(height, width)
        return positions

    def forward(
        self,
        input_ids: Tensor,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        logits_to_keep=0,
    ):
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("Pixels and image grid must be supplied together")
        if (
            pixel_values is not None
            and attention_mask is not None
            and not bool(attention_mask.bool().all())
        ):
            raise ValueError("Padded image batches are not supported")
        if bool((input_ids == self.config["video_token_id"]).any()):
            raise ValueError("Video input is not supported")
        if pixel_values is None and bool(
            (input_ids == self.config["image_token_id"]).any()
        ):
            raise ValueError("Image placeholders require pixels")
        hidden = self.language_model.embed_tokens(input_ids)
        visual_mask = input_ids == self.config["image_token_id"]
        deepstack = []
        if pixel_values is not None:
            features, deepstack = self.visual(pixel_values, image_grid_thw)
            if int(visual_mask.sum()) != features.shape[0]:
                raise ValueError("Image feature count differs from placeholders")
            hidden = hidden.masked_scatter(
                visual_mask[..., None], features.to(hidden.dtype)
            )
        if position_ids is None:
            position_ids = self.position_ids(input_ids, image_grid_thw)
        cos, sin = self.language_model.rope(position_ids, hidden.dtype)
        mask = None
        if attention_mask is not None:
            length = input_ids.shape[1]
            mask = torch.ones(
                length, length, device=input_ids.device, dtype=torch.bool
            ).tril()
            mask = mask[None, None] & attention_mask[:, None, None, :].bool()
        for index, layer in enumerate(self.language_model.layers):
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(layer, hidden, cos, sin, mask, use_reentrant=False)
            else:
                hidden = layer(hidden, cos, sin, mask)
            if index < len(deepstack):
                hidden = hidden.clone()
                hidden[visual_mask] = hidden[visual_mask] + deepstack[index].to(
                    hidden.dtype
                )
        hidden = self.language_model.norm(hidden)
        if isinstance(logits_to_keep, Tensor):
            hidden = hidden[:, logits_to_keep]
        elif logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        weight = (
            self.language_model.embed_tokens.weight
            if self.tied
            else self.lm_head.weight
        )
        return F.linear(hidden, weight)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device="cpu", dtype=torch.bfloat16):
        from nano_areal.archon.state_dict import load_qwen3vl_weights

        path = Path(path)
        with torch.device("meta"):
            model = cls(json.loads((path / "config.json").read_text()))
        load_qwen3vl_weights(model, path, device, dtype)
        model.language_model.rope = InterleavedMRoPE(model.config["text_config"]).to(
            device
        )
        return model


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.enabled = True

    def forward(self, hidden):
        result = self.base(hidden)
        if self.enabled:
            update = F.linear(F.linear(hidden.float(), self.lora_a), self.lora_b)
            result = result + (update * self.scale).to(result.dtype)
        return result


def add_lora(model: Qwen3VLModel, rank=8, alpha=16):
    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    model.requires_grad_(False)
    for layer in model.language_model.layers:
        for name in ("q_proj", "v_proj"):
            base = getattr(layer.attn, name)
            if isinstance(base, LoRALinear):
                raise ValueError("LoRA is already installed")
            setattr(layer.attn, name, LoRALinear(base, rank, alpha))
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def export_mlx_lora(model: Qwen3VLModel, path: Path):
    from safetensors.torch import save_file

    weights = {}
    for index, layer in enumerate(model.language_model.layers):
        for projection in ("q_proj", "v_proj"):
            module = getattr(layer.attn, projection)
            if not isinstance(module, LoRALinear):
                raise ValueError("Expected language attention LoRA adapters")
            if module.scale != 2:
                raise ValueError("MLX bridge currently requires LoRA alpha/rank = 2")
            for side in ("a", "b"):
                name = (
                    f"language_model.model.layers.{index}.self_attn."
                    f"{projection}.lora_{side}"
                )
                tensor = getattr(module, f"lora_{side}")
                weights[name] = tensor.detach().cpu().to(torch.bfloat16).T.contiguous()
    temporary = path.with_suffix(".tmp.safetensors")
    save_file(weights, str(temporary))
    temporary.replace(path)
