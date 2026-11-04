"""Forward-only native Metal reduction checked independently against MLX."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from nano_areal.archon.model import RMSNorm
from nano_areal.archon.s4_metal import differentiable_metal_rms, metal_rms


def test_metal_rms_rejects_unmeasured_devices():
    with pytest.raises(ValueError, match="measured bf16 MPS"):
        metal_rms(torch.ones(1, 128).bfloat16(), torch.ones(128).bfloat16(), 1e-6)


def test_mlx_metal_rms_forward_and_gradient_rejection(tmp_path):
    executable = Path(".s1-cache/venv-metal/bin/python")
    if not executable.exists() or not torch.backends.mps.is_available():
        pytest.skip("Repository-local MLX and MPS are required")
    program = '''
import mlx.core as mx
import numpy as np
from pathlib import Path
directory = Path(DIRECTORY)
mx.random.seed(45)
for width in (128, 2560):
    hidden = mx.random.normal((5, width)).astype(mx.bfloat16)
    weight = mx.random.normal((width,)).astype(mx.bfloat16)
    output = mx.fast.rms_norm(hidden, weight, 1e-6)
    for name, value in {"hidden": hidden, "weight": weight, "output": output}.items():
        np.save(directory / f"{width}-{name}.npy", np.asarray(value.astype(mx.float32)))
'''
    subprocess.run(
        [str(executable), "-c", program.replace("DIRECTORY", repr(str(tmp_path)))],
        env={**os.environ}, check=True, capture_output=True, text=True, timeout=60,
    )
    for width in (128, 2560):
        arrays = {name: torch.from_numpy(np.load(tmp_path / f"{width}-{name}.npy"))
                  for name in ("hidden", "weight", "output")}
        hidden = arrays["hidden"].to("mps").bfloat16()
        weight = arrays["weight"].to("mps").bfloat16()
        actual = metal_rms(hidden, weight, 1e-6)
        torch.testing.assert_close(actual.float().cpu(), arrays["output"],
                                   rtol=0, atol=0)
        hidden.requires_grad_()
        with pytest.raises(ValueError, match="forward-only"):
            metal_rms(hidden, weight, 1e-6)


@pytest.mark.parametrize("width", [128, 2560])
def test_differentiable_metal_rms_first_order_matches_native(width):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required")
    torch.manual_seed(46)
    hidden = torch.randn(1, 5, width, device="mps", dtype=torch.bfloat16,
                         requires_grad=True)
    reference = hidden.detach().clone().requires_grad_()
    module = RMSNorm(width).to(device="mps", dtype=torch.bfloat16)
    with torch.no_grad():
        module.weight.normal_()
    weight = module.weight.detach().clone().requires_grad_()
    gradient = torch.randn_like(hidden)
    actual = differentiable_metal_rms(hidden, weight, module.eps)
    actual.backward(gradient)
    expected = module(reference)
    expected.backward(gradient)
    torch.testing.assert_close(actual.float(), expected.float(),
                               rtol=1 / 128, atol=1e-5)
    for current, original in ((hidden.grad, reference.grad),
                              (weight.grad, module.weight.grad)):
        assert bool(torch.isfinite(current).all()) and bool(current.abs().any())
        torch.testing.assert_close(current.float(), original.float(),
                                   rtol=1 / 128, atol=1e-4)
