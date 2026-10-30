"""Decode rounding and gradient contracts, independently checked against MLX."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from nano_areal.archon.s4_operators import affine_decode, affine_weights


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
