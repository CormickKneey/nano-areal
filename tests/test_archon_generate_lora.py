import torch

from nano_areal.archon.generate import Generator, SamplingParams
from nano_areal.archon.lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    count_trainable,
    merge_lora_state_dict,
)
from nano_areal.archon.model import Qwen35Args, Qwen35Model


def tiny_args(**kw) -> Qwen35Args:
    base = dict(
        dim=32, n_layers=4, n_heads=2, n_kv_heads=1, head_dim=16, vocab_size=64,
        intermediate_size=64, rope_theta=1e4, layer_types=["linear_attention"] * 3 + ["full_attention"],
        attn_output_gate=True, partial_rotary_factor=0.25, mrope_section=[1, 1, 0],
        zero_centered_norm=True, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
    )
    base.update(kw)
    return Qwen35Args(**base)


def test_greedy_generate_matches_full_forward():
    torch.manual_seed(0)
    model = Qwen35Model(tiny_args()).eval()
    gen = Generator(model)
    prompt = [3, 5, 7, 11, 13]
    res = gen.generate(prompt, SamplingParams(max_new_tokens=6, temperature=0.0))
    assert len(res.token_ids) == 6 and res.finish_reason == "length"
    # Re-derive greedily with full (non-cached) forwards
    ids = list(prompt)
    for expected in res.token_ids:
        with torch.no_grad():
            logits = model(torch.tensor([ids]))[0, -1]
        assert int(logits.argmax()) == expected
        ids.append(expected)
    # logprobs from cache == full-forward prompt_logprobs of the completed sequence
    plp = gen.prompt_logprobs(ids)[len(prompt) - 1:]
    torch.testing.assert_close(torch.tensor(plp), torch.tensor(res.logprobs), atol=1e-4, rtol=1e-4)


def test_stop_token_ends_generation():
    torch.manual_seed(0)
    model = Qwen35Model(tiny_args()).eval()
    gen = Generator(model)
    first = gen.generate([1, 2, 3], SamplingParams(max_new_tokens=4, temperature=0.0)).token_ids[0]
    res = gen.generate([1, 2, 3], SamplingParams(max_new_tokens=4, temperature=0.0, stop_token_ids=(first,)))
    assert res.token_ids == [first] and res.finish_reason == "stop"


def test_lora_apply_and_merge():
    torch.manual_seed(0)
    model = Qwen35Model(tiny_args())
    ref_sd = {k: v.clone() for k, v in model.state_dict().items()}
    apply_lora(model, LoRAConfig(rank=4, alpha=8))
    trainable, total = count_trainable(model)
    assert 0 < trainable < total
    assert isinstance(model.language_model.layers[0].linear_attn.in_proj_qkv, LoRALinear)
    assert isinstance(model.language_model.layers[3].self_attn.q_proj, LoRALinear)
    # B is zero-init → merged == original
    merged = merge_lora_state_dict(model)
    assert set(merged) == set(ref_sd)
    for k in ref_sd:
        torch.testing.assert_close(merged[k], ref_sd[k])
    # after a change to B the merged weight moves and a fresh model can load it
    with torch.no_grad():
        model.language_model.layers[3].self_attn.q_proj.lora_B.fill_(0.1)
    merged = merge_lora_state_dict(model)
    fresh = Qwen35Model(tiny_args())
    fresh.load_state_dict(merged, strict=True)
    ids = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        torch.testing.assert_close(fresh(ids), model(ids), atol=1e-5, rtol=1e-5)
