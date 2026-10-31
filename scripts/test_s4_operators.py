"""Decode rounding and gradient contracts, independently checked against MLX."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from diagnose_s4_projections import projection_variants

from nano_areal.archon.s4_operators import (
    affine_decode,
    affine_decode_weights,
    affine_weights,
    device_rope_frequencies,
    prefill_split_count,
    rounded_swiglu,
    splitk_prefill,
    splitkv_attention,
    tokenwise_frozen_linear,
    tokenwise_splitkv_attention,
)


def test_shared_projection_uses_original_prefill_shape():
    packed = torch.full((256, 128), 0x76543210, dtype=torch.int64)
    scales = torch.full((256, 16), 0.015625, dtype=torch.bfloat16)
    biases = torch.full_like(scales, -0.05)
    weights = affine_weights(packed, scales, biases).bfloat16()
    hidden = torch.randn(1, 1024, dtype=torch.bfloat16)
    variants = projection_variants(hidden, weights, packed, scales, biases, 128)
    expected = splitk_prefill(hidden[:, None].expand(1, 128, 1024), weights)[:, -1]
    torch.testing.assert_close(variants["splitk_bf16"], expected, rtol=0, atol=0)
    decode = projection_variants(hidden, weights, packed, scales, biases, 1)
    assert "splitk_bf16" not in decode
    torch.testing.assert_close(decode["affine_bias_rounding"],
                               affine_decode(hidden, packed, scales, biases))


def test_device_rope_config_contract():
    config = {"head_dim": 128, "rope_theta": 5000000}
    actual = device_rope_frequencies(config, "cpu")
    torch.testing.assert_close(actual, 1 / 5000000 ** (torch.arange(64).float() / 64))
    nested = {"head_dim": 128, "rope_parameters": {"rope_theta": 10000}}
    assert not torch.equal(actual, device_rope_frequencies(nested, "cpu"))
    with pytest.raises(ValueError, match="default rotary"):
        device_rope_frequencies(
            {**config, "rope_scaling": {"rope_type": "yarn"}}, "cpu")


def test_resident_affine_preserves_values_and_input_gradients():
    packed = torch.full((8, 8), 0x76543210, dtype=torch.int64)
    scales = torch.full((8, 1), 0.015625, dtype=torch.bfloat16)
    biases = torch.full_like(scales, -0.05)
    hidden = torch.randn(2, 64, dtype=torch.bfloat16, requires_grad=True)
    actual = affine_decode_weights(
        hidden, affine_weights(packed, scales, biases), biases)
    expected = affine_decode(hidden, packed, scales, biases)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().mean().backward()
    assert bool(torch.isfinite(hidden.grad).all()) and bool(hidden.grad.abs().any())
    with pytest.raises(ValueError, match="matching fp32"):
        affine_decode_weights(
            hidden, affine_weights(packed, scales, biases).half(), biases)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_tokenwise_frozen_forward_and_batched_input_gradients(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is required")
    torch.manual_seed(83)
    dtype = torch.float64 if device == "cpu" else torch.float32
    hidden = torch.randn(1, 16, 128, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(256, 128, device=device, dtype=dtype)
    gradient = torch.randn(1, 16, 256, device=device, dtype=dtype)
    actual = tokenwise_frozen_linear(hidden, weight)
    expected = torch.cat([torch.nn.functional.linear(
        hidden[:, index:index + 1], weight) for index in range(16)], dim=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward(gradient)
    observed = hidden.grad.clone()
    hidden.grad = None
    expected.backward(gradient)
    torch.testing.assert_close(observed, hidden.grad,
                               rtol=1e-12 if device == "cpu" else 1e-4,
                               atol=1e-12 if device == "cpu" else 1e-5)
    assert not weight.requires_grad and bool(torch.isfinite(observed).all())
    with pytest.raises(ValueError, match="frozen weights"):
        tokenwise_frozen_linear(hidden, weight.requires_grad_())


def test_splitkv_masks_future_partitions_and_has_finite_gradients():
    query = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(1, 2, 1025, 8, dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    actual = splitkv_attention(query, key, value, 510)
    rows = torch.cat([splitkv_attention(query[..., index:index + 1, :],
                                      key[..., :511 + index, :],
                                      value[..., :511 + index, :], 510 + index)
                      for index in range(3)], dim=-2)
    torch.testing.assert_close(actual, rows, rtol=0, atol=0)
    actual.float().square().mean().backward()
    for gradient in (query.grad, key.grad, value.grad):
        assert bool(torch.isfinite(gradient).all()) and bool(gradient.abs().any())
    assert not bool(key.grad[..., 513:, :].any())
    with pytest.raises(ValueError, match="compatible bf16"):
        splitkv_attention(query.float(), key, value, 510)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("offset", [510, 1024])
@pytest.mark.parametrize("frozen_key", [False, True])
def test_tokenwise_attention_preserves_values_and_complete_kv_gradients(
    device, offset, frozen_key,
):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is required")
    torch.manual_seed(44)
    query = torch.randn(1, 2, 16, 128, device=device, dtype=torch.bfloat16,
                        requires_grad=True)
    key = torch.randn(1, 2, offset + 23, 128, device=device, dtype=torch.bfloat16,
                      requires_grad=not frozen_key)
    value = torch.randn_like(key, requires_grad=True)
    gradient = torch.randn_like(query)
    targets = [tensor for tensor in (query, key, value) if tensor.requires_grad]
    expected = torch.cat([
        splitkv_attention(query[..., index:index + 1, :],
                          key[..., :offset + index + 1, :],
                          value[..., :offset + index + 1, :], offset + index)
        for index in range(16)], dim=-2)
    expected_gradients = torch.autograd.grad(expected, targets, gradient)
    actual = tokenwise_splitkv_attention(query, key, value, offset)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradients = torch.autograd.grad(actual, targets, gradient)
    for observed, reference in zip(gradients, expected_gradients, strict=True):
        assert bool(torch.isfinite(observed).all())
        difference = (observed.float() - reference.float()).norm()
        assert float(difference / reference.float().norm()) <= 1 / 128
    assert bool(gradients[-1][..., :offset, :].abs().any())
    assert not bool(gradients[-1][..., offset + 16:, :].any())


def test_mlx_paged_splitkv_rounding(tmp_path):
    executable = Path(".s1-cache/venv-metal/bin/python")
    if not executable.exists() or not torch.backends.mps.is_available():
        pytest.skip("Repository-local MLX and MPS are required")
    program = '''
import mlx.core as mx
import numpy as np
from pathlib import Path
from vllm_metal.metal import get_ops
directory = Path(DIRECTORY)
ops = get_ops()
mx.random.seed(44)
query = mx.random.normal((4, 32, 128)).astype(mx.bfloat16)
key = mx.random.normal((73, 16, 8, 128)).astype(mx.bfloat16)
value = mx.random.normal(key.shape).astype(mx.bfloat16)
output = []
for index in range(4):
    end = 1155 + index
    output.append(ops.paged_attention_primitive(
        query[index:index + 1], key, value, 8, 128 ** -0.5, 0.,
        mx.arange(73, dtype=mx.uint32)[None], mx.array([end], dtype=mx.uint32),
        mx.array([0, 1], dtype=mx.uint32), 16, end, -1))
arrays = {"query": query, "key": key.reshape(-1, 8, 128)[:1158],
          "value": value.reshape(-1, 8, 128)[:1158],
          "output": mx.concatenate(output)}
for name, tensor in arrays.items():
    np.save(directory / (name + ".npy"), np.asarray(tensor.astype(mx.float32)))
'''
    subprocess.run(
        [str(executable), "-c", program.replace("DIRECTORY", repr(str(tmp_path)))],
        env={**os.environ}, check=True, capture_output=True, text=True, timeout=60,
    )
    arrays = {name: torch.from_numpy(np.load(tmp_path / f"{name}.npy"))
              for name in ("query", "key", "value", "output")}
    expected = arrays["output"].transpose(0, 1)[None]
    for device in ("cpu", "mps"):
        query = arrays["query"].transpose(0, 1)[None].to(device).bfloat16()
        key, value = [arrays[name].transpose(0, 1)[None].repeat_interleave(
            4, dim=1).to(device).bfloat16() for name in ("key", "value")]
        actual = splitkv_attention(query, key, value, 1154).float().cpu()
        error = (actual - expected).abs()
        assert float(error.mean()) < 2e-6
        assert float(error.max()) < 0.002


def test_mlx_mps_initialized_rope_constants(tmp_path):
    executable = Path(".s1-cache/venv-metal/bin/python")
    if not executable.exists() or not torch.backends.mps.is_available():
        pytest.skip("Repository-local MLX and MPS are required")
    program = '''
import mlx.core as mx
import numpy as np
from pathlib import Path
directory = Path(DIRECTORY)
freq = 1 / (5000000 ** (mx.arange(0, 128, 2).astype(mx.float32) / 128))
angle = mx.arange(2048).astype(mx.float32)[:, None] * freq
for name, value in {"freq": freq, "cos": mx.cos(angle).astype(mx.bfloat16),
                    "sin": mx.sin(angle).astype(mx.bfloat16)}.items():
    np.save(directory / (name + ".npy"), np.asarray(value.astype(mx.float32)))
'''
    subprocess.run(
        [str(executable), "-c", program.replace("DIRECTORY", repr(str(tmp_path)))],
        env={**os.environ}, check=True, capture_output=True, text=True, timeout=60,
    )
    arrays = {name: torch.from_numpy(np.load(tmp_path / f"{name}.npy"))
              for name in ("freq", "cos", "sin")}
    config = {"head_dim": 128, "rope_theta": 5000000}
    frequency = device_rope_frequencies(config, "mps")
    torch.testing.assert_close(frequency.cpu(), arrays["freq"], rtol=0, atol=0)
    assert not torch.equal(device_rope_frequencies(config, "cpu"), arrays["freq"])
    angles = torch.arange(2048, device="mps").float()[:, None] * frequency
    for name, value in (("cos", angles.cos()), ("sin", angles.sin())):
        torch.testing.assert_close(value.bfloat16().float().cpu(), arrays[name],
                                   rtol=0, atol=0)


def test_affine_unpack_and_decode_gradients():
    packed = torch.full((8, 8), 0x76543210, dtype=torch.int64)
    scales = torch.ones(8, 1, dtype=torch.bfloat16)
    biases = torch.full_like(scales, -3)
    expected = torch.arange(8).repeat(8).float() - 3
    assert torch.equal(affine_weights(packed, scales, biases)[0], expected)
    hidden = torch.randn(2, 64, dtype=torch.bfloat16, requires_grad=True)
    output = affine_decode(hidden, packed, scales, biases)
    assert output.shape == (2, 8) and output.dtype == hidden.dtype
    output.float().square().mean().backward()
    assert bool(torch.isfinite(hidden.grad).all()) and bool(hidden.grad.abs().any())
    with pytest.raises(ValueError, match="bf16"):
        affine_decode(hidden.float(), packed, scales, biases)
    with pytest.raises(ValueError, match="groups of 64"):
        affine_weights(packed, scales[:, :0], biases[:, :0])


def test_prefill_split_shapes_and_gradients():
    assert prefill_split_count(1157, 2560, 4096) == 1
    assert prefill_split_count(70, 2560, 4096) == 1
    assert prefill_split_count(70, 2560, 1024) == 5
    assert prefill_split_count(57, 2560, 4096) == 2
    assert prefill_split_count(20, 2560, 1024) == 10
    with pytest.raises(ValueError, match="64-aligned"):
        prefill_split_count(20, 65, 32)
    hidden = torch.randn(1, 128, 1024, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 1024, dtype=torch.bfloat16)
    output = splitk_prefill(hidden, weight)
    output.float().square().mean().backward()
    assert bool(torch.isfinite(hidden.grad).all()) and bool(hidden.grad.abs().any())
    with pytest.raises(ValueError, match="prefill >=16"):
        splitk_prefill(hidden[:, :1], weight)


def test_mlx_bf16_splitk_reference(tmp_path):
    executable = Path(".s1-cache/venv-metal/bin/python")
    if not executable.exists():
        pytest.skip("Repository-local MLX environment is unavailable")
    program = '''
import mlx.core as mx
import numpy as np
from pathlib import Path
directory = Path(DIRECTORY)
mx.random.seed(43)
weight = mx.random.normal((256, 1024)).astype(mx.bfloat16)
packed, scales, biases = mx.quantize(weight, group_size=64, bits=4)
hidden = mx.random.normal((1, 128, 1024)).astype(mx.bfloat16)
dequantized = mx.dequantize(packed, scales, biases, group_size=64, bits=4)
fused = mx.quantized_matmul(hidden, packed, scales, biases, group_size=64, bits=4)
for name, value in {"weight": dequantized, "hidden": hidden, "fused": fused}.items():
    np.save(directory / (name + ".npy"), np.asarray(value.astype(mx.float32)))
'''
    subprocess.run(
        [str(executable), "-c", program.replace("DIRECTORY", repr(str(tmp_path)))],
        env={**os.environ}, check=True, capture_output=True, text=True, timeout=60,
    )
    arrays = {name: torch.from_numpy(np.load(tmp_path / f"{name}.npy"))
              for name in ("weight", "hidden", "fused")}
    hidden, weight = arrays["hidden"].bfloat16(), arrays["weight"].bfloat16()
    dense = torch.nn.functional.linear(hidden, weight).float()
    corrected = splitk_prefill(hidden, weight).float()
    assert float((dense - arrays["fused"]).abs().mean()) > 0
    torch.testing.assert_close(corrected, arrays["fused"], rtol=0, atol=0)
    if torch.backends.mps.is_available():
        candidate = splitk_prefill(hidden.to("mps"), weight.to("mps")).float().cpu()
        torch.testing.assert_close(candidate, arrays["fused"], rtol=0, atol=0)


def test_mlx_bf16_decode_reference(tmp_path):
    executable = Path(".s1-cache/venv-metal/bin/python")
    if not executable.exists():
        pytest.skip("Repository-local MLX environment is unavailable")
    program = '''
import mlx.core as mx
import numpy as np
from pathlib import Path
directory = Path(DIRECTORY)
mx.random.seed(42)
weight = mx.random.normal((64, 256)).astype(mx.bfloat16)
packed, scales, biases = mx.quantize(weight, group_size=64, bits=4)
hidden = mx.random.normal((1, 256)).astype(mx.bfloat16)
fused = mx.quantized_matmul(hidden, packed, scales, biases, group_size=64, bits=4)
for name, value in {"packed": packed, "scales": scales, "biases": biases,
                    "hidden": hidden, "fused": fused}.items():
    if name != "packed":
        value = value.astype(mx.float32)
    np.save(directory / (name + ".npy"), np.asarray(value))
'''
    subprocess.run(
        [str(executable), "-c", program.replace("DIRECTORY", repr(str(tmp_path)))],
        env={**os.environ}, check=True, capture_output=True, text=True, timeout=60,
    )
    arrays = {name: torch.from_numpy(np.load(tmp_path / f"{name}.npy"))
              for name in ("packed", "scales", "biases", "hidden", "fused")}
    hidden = arrays["hidden"].bfloat16()
    packed = arrays["packed"]
    scales, biases = arrays["scales"].bfloat16(), arrays["biases"].bfloat16()
    precise = torch.nn.functional.linear(
        hidden.float(), affine_weights(packed, scales, biases)
    )
    corrected = affine_decode(hidden, packed, scales, biases).float()
    assert float((precise.bfloat16().float() - arrays["fused"]).abs().mean()) > 0
    torch.testing.assert_close(corrected, arrays["fused"], rtol=0, atol=0)


def test_mlx_rounded_swiglu_and_gradients(tmp_path):
    executable = Path(".s1-cache/venv-metal/bin/python")
    if not executable.exists():
        pytest.skip("Repository-local MLX environment is unavailable")
    program = '''
import mlx.core as mx
import numpy as np
from mlx_vlm.models.activations import swiglu
from pathlib import Path
directory = Path(DIRECTORY)
gate = mx.arange(-8, 9).astype(mx.bfloat16)[None]
up = mx.arange(1, 18).astype(mx.bfloat16)[None] / 16
output = swiglu(gate, up)
for name, value in {"gate": gate, "up": up, "output": output}.items():
    np.save(directory / (name + ".npy"), np.asarray(value.astype(mx.float32)))
'''
    subprocess.run(
        [str(executable), "-c", program.replace("DIRECTORY", repr(str(tmp_path)))],
        env={**os.environ}, check=True, capture_output=True, text=True, timeout=60,
    )
    arrays = {name: torch.from_numpy(np.load(tmp_path / f"{name}.npy"))
              for name in ("gate", "up", "output")}
    gate = arrays["gate"].bfloat16().requires_grad_()
    up = arrays["up"].bfloat16().requires_grad_()
    actual = rounded_swiglu(gate, up)
    torch.testing.assert_close(actual.float(), arrays["output"], rtol=0, atol=0)
    actual.float().square().mean().backward()
    for gradient in (gate.grad, up.grad):
        assert bool(torch.isfinite(gradient).all()) and bool(gradient.abs().any())
    with pytest.raises(ValueError, match="matching bf16"):
        rounded_swiglu(gate.float(), up)
    if torch.backends.mps.is_available():
        candidate = rounded_swiglu(gate.to("mps"), up.to("mps")).float().cpu()
        torch.testing.assert_close(candidate, arrays["output"], rtol=0, atol=0)
