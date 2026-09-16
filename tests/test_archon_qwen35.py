"""Numerical parity of nano-archon Qwen3.5 against the HF reference (tiny random weights, CPU)."""
import pytest
import torch

from nano_areal.archon.model import Qwen35Args, Qwen35Model
from nano_areal.archon.state_dict import (
    Qwen35StateDictAdapter,
    archon_key_to_hf,
    hf_key_to_archon,
)

transformers = pytest.importorskip("transformers")
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration  # noqa: E402

TINY_TEXT = dict(
    vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
    linear_value_head_dim=16, linear_conv_kernel_dim=4,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25,
                         mrope_section=[3, 3, 2], mrope_interleaved=True),
    max_position_embeddings=1024, tie_word_embeddings=True, attn_output_gate=True,
)
TINY_VISION = dict(
    depth=2, hidden_size=32, intermediate_size=64, num_heads=4, patch_size=4,
    spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
    num_position_embeddings=16, in_channels=3, hidden_act="gelu_pytorch_tanh",
    deepstack_visual_indexes=[],
)
IMAGE_TOKEN, VSTART, VEND = 500, 501, 502


def _make_pair(seed: int = 0):
    torch.manual_seed(seed)
    cfg = Qwen3_5Config(
        text_config=dict(TINY_TEXT), vision_config=dict(TINY_VISION),
        image_token_id=IMAGE_TOKEN, vision_start_token_id=VSTART,
        vision_end_token_id=VEND, video_token_id=503, tie_word_embeddings=True,
    )
    hf = Qwen3_5ForConditionalGeneration(cfg).eval()
    # randomise the params HF leaves at constant init so the test is meaningful
    with torch.no_grad():
        for n, p in hf.named_parameters():
            if p.dim() == 1 and ("norm" in n or "A_log" in n or "dt_bias" in n):
                p.add_(torch.randn_like(p) * 0.1)
    args = Qwen35Args.from_hf_dict(hf.config.to_dict())
    archon = Qwen35Model(args).eval()
    sd = Qwen35StateDictAdapter().hf_to_archon(hf.state_dict())
    sd.pop("lm_head.weight")  # tied
    missing, unexpected = archon.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert not missing, missing
    return hf, archon, args


def test_args_from_hf_dict_parses_hybrid_layout():
    _, _, args = _make_pair()
    assert args.is_hybrid
    assert args.layer_types == TINY_TEXT["layer_types"]
    assert args.attn_output_gate and args.zero_centered_norm
    assert args.partial_rotary_factor == 0.25 and args.mrope_section == [3, 3, 2]
    assert args.vision is not None and args.vision.out_hidden_size == 64
    assert args.image_token_id == IMAGE_TOKEN


def test_text_forward_matches_hf():
    hf, archon, _ = _make_pair()
    ids = torch.randint(0, 500, (2, 37))
    with torch.no_grad():
        ref = hf(input_ids=ids).logits
        out = archon(ids)
    torch.testing.assert_close(out, ref, atol=2e-4, rtol=1e-4)


def test_multimodal_forward_matches_hf():
    hf, archon, args = _make_pair(1)
    # one image: grid (t=1, h=4, w=4) patches -> 4 merged tokens
    grid = torch.tensor([[1, 4, 4]])
    pixel = torch.randn(16, 3 * 2 * 4 * 4)
    n_img_tok = 4
    prefix = torch.randint(0, 500, (1, 5))
    suffix = torch.randint(0, 500, (1, 7))
    ids = torch.cat([
        prefix, torch.tensor([[VSTART]]), torch.full((1, n_img_tok), IMAGE_TOKEN),
        torch.tensor([[VEND]]), suffix,
    ], dim=1)
    mm_types = (ids == IMAGE_TOKEN).long()
    with torch.no_grad():
        ref = hf(input_ids=ids, pixel_values=pixel, image_grid_thw=grid, mm_token_type_ids=mm_types).logits
        out = archon(ids, pixel_values=pixel, image_grid_thw=grid)
    torch.testing.assert_close(out, ref, atol=3e-4, rtol=1e-4)


def test_gdn_chunk_matches_recurrent_step():
    """Chunked prefill vs. token-by-token recurrence must agree."""
    from nano_areal.archon.gdn import GatedDeltaNet
    torch.manual_seed(0)
    gdn = GatedDeltaNet(dim=32, num_k_heads=2, num_v_heads=4, head_k_dim=8, head_v_dim=8).eval()
    x = torch.randn(2, 70, 32)
    with torch.no_grad():
        full = gdn(x)
        conv_s, rec_s = gdn.init_state(2, x.device, x.dtype)
        steps = []
        for t in range(x.shape[1]):
            o, conv_s, rec_s = gdn.forward_step(x[:, t:t + 1], conv_s, rec_s)
            steps.append(o)
    torch.testing.assert_close(torch.cat(steps, 1), full, atol=1e-4, rtol=1e-4)


def test_state_dict_key_roundtrip():
    keys = [
        "model.language_model.layers.3.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.visual.blocks.1.attn.qkv.weight",
        "lm_head.weight",
    ]
    for k in keys:
        assert archon_key_to_hf(hf_key_to_archon(k)) == k
    assert hf_key_to_archon("mtp.layers.0.mlp.gate_proj.weight") is None
    # dense Qwen3 layout
    got = hf_key_to_archon("model.layers.0.self_attn.q_proj.weight")
    assert got == "language_model.layers.0.self_attn.q_proj.weight"
    assert archon_key_to_hf("language_model.norm.weight", dense_layout=True) == "model.norm.weight"


def test_dense_qwen3_args_defaults():
    cfg = dict(
        model_type="qwen3", hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        num_hidden_layers=2, vocab_size=100, intermediate_size=128, rope_theta=1e6,
        tie_word_embeddings=True, head_dim=16,
    )
    args = Qwen35Args.from_hf_dict(cfg)
    assert not args.is_hybrid and not args.attn_output_gate and not args.zero_centered_norm
    assert args.mrope_section is None and args.partial_rotary_factor == 1.0
    m = Qwen35Model(args)
    out = m(torch.randint(0, 100, (1, 9)))
    assert out.shape == (1, 9, 100)
