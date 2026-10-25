"""Eager fast dispatch must preserve the registered op under dispatch modes."""

import pytest
import torch
import torch_npu  # noqa: F401
from torch.utils._python_dispatch import TorchDispatchMode

from flash_attn_npu import flash_attn_varlen_func
from scripts.benchmark_varlen_vs_cann import compare_dual, reference_chunk


class ObserveAttention(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.seen = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen.add(func._schema.name)
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_eager_and_dispatch_mode_gradients(dtype):
    torch.manual_seed(2026)
    cpu = [torch.randn(129, h, 128).to(dtype) for h in (4, 1, 1)]
    dout = torch.randn_like(cpu[0])
    q, k, v = [x.npu().requires_grad_() for x in cpu]
    device_dout = dout.npu()
    cu = torch.tensor([0, 129], dtype=torch.int32).npu()
    scale = 128**-0.5
    refs = [
        reference_chunk(*(x[None] for x in (*cpu, dout)), scale, True, upcast=upcast)
        for upcast in (True, False)
    ]

    def training():
        out = flash_attn_varlen_func(q, k, v, cu, cu, 129, 129, causal=True, softmax_scale=scale)
        return out, *torch.autograd.grad(out, (q, k, v), device_dout)

    eager = training()
    mode = ObserveAttention()
    with mode:
        dispatched = training()
    assert "flash_attn_npu::_flash_attn_varlen_forward" in mode.seen
    assert "flash_attn_npu::_flash_attn_varlen_backward" in mode.seen
    for outputs in (eager, dispatched):
        for index, output in enumerate(outputs):
            check = compare_dual(output.cpu(), refs[0][index][0], refs[1][index][0])
            assert check["passed"], (index, check)
