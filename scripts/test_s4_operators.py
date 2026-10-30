"""Decode rounding and gradient contracts, independently checked against MLX."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from nano_areal.archon.s4_operators import (
    affine_decode,
    affine_weights,
    prefill_split_count,
    rounded_swiglu,
    splitk_prefill,
)


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
