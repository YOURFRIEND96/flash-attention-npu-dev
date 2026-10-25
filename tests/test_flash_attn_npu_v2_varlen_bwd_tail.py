"""Regression coverage for tail GEMMs and full-attention workspace packing."""

import pytest
import torch
import torch_npu

if "Ascend950" in (torch_npu.npu.get_device_name() if torch_npu.npu.device_count() else ""):
    pytest.skip("flash_attn_npu v2 requires Ascend 910", allow_module_level=True)

from flash_attn_npu import flash_attn_varlen_func
from tests.common.attention_ref import ref_flash_attention_pair
from tests.common.compare import assert_fa_close
from tests.common.test_utils import make_cu_seqlens, make_padded_varlen_mask, pad_packed_tensor


def _check_backward(lengths, dtype, causal, heads, kv_heads, softcap, normal_inputs=False, dim=128):
    total, maximum = sum(lengths), max(lengths)
    generator = torch.Generator().manual_seed(2026)

    def random_tensor(h):
        values = (
            torch.randn(total, h, dim, generator=generator)
            if normal_inputs
            else torch.rand(total, h, dim, generator=generator) - 0.5
        )
        return values.to(dtype)

    refs = [random_tensor(h).requires_grad_(True) for h in (heads, kv_heads, kv_heads)]
    inputs = [x.detach().npu().requires_grad_(True) for x in refs]
    dout = random_tensor(heads)
    cu = make_cu_seqlens(lengths).npu()
    # T5's scale=1 exposes the bug more strongly than 1/sqrt(D).
    out = flash_attn_varlen_func(
        *inputs,
        cu,
        cu,
        maximum,
        maximum,
        softmax_scale=1.0,
        causal=causal,
        softcap=softcap,
    )
    grads = torch.autograd.grad(out, inputs, dout.npu())

    valid, _, mask = make_padded_varlen_mask(lengths, lengths, maximum, maximum, causal, -1, -1)
    padded = [pad_packed_tensor(x, lengths, maximum) for x in refs]
    high, _, low, _ = ref_flash_attention_pair(
        *padded, 1.0, mask, dtype, softcap, differentiable=True
    )
    high, low = high[valid], low[valid]
    high_grads = torch.autograd.grad(high, refs, dout, retain_graph=True)
    low_grads = torch.autograd.grad(low, refs, dout)
    for name, actual, ref, pt in zip(
        ("out", "dq", "dk", "dv"), (out, *grads), (high, *high_grads), (low, *low_grads)
    ):
        assert_fa_close(actual, ref, pt, softcap=softcap, name=name)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seqlen", [1, 129, 257, 385, 513])
@pytest.mark.parametrize(
    "causal,heads,kv_heads,softcap",
    [(False, 4, 4, 0.0), (True, 4, 4, 0.0), (False, 4, 1, 0.0), (True, 4, 2, 2.0)],
    ids=["full-mha", "causal-mha", "full-mqa", "causal-gqa-softcap"],
)
def test_varlen_backward_single_kv_row(dtype, seqlen, causal, heads, kv_heads, softcap):
    # Exact 128*n+1 lengths are essential: near-max random lengths can miss
    # the one-row GEMM tail. Equal Q/K lengths and D=128 select FAGVarlenOpt.
    _check_backward([seqlen, seqlen + 1], dtype, causal, heads, kv_heads, softcap)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [8, 2])
@pytest.mark.parametrize(
    "lengths",
    [[257, 256, 257, 257, 257, 257, 257, 256], [257, 256] * 16],
    ids=["mixed-8", "mixed-32"],
)
def test_varlen_backward_full_workspace_boundary(dtype, kv_heads, lengths):
    # 3x3 and 2x2 tile grids leave odd free space in a 16-tile segment.
    # Full attention must floor the number of two-row groups that fit;
    # causal's ceil rule would emit 17 tiles and corrupt adjacent workspace.
    _check_backward(lengths, dtype, False, 8, kv_heads, 0.0, normal_inputs=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [8, 2])
@pytest.mark.parametrize("seqlen", [257, 279, 384])
@pytest.mark.parametrize("batch", [3, 17])
def test_causal_segment_mapping(dtype, kv_heads, seqlen, batch):
    # Cover incomplete last segments, crossing batch/head boundaries, and
    # multiple scheduling rounds in the causal address mapping.
    _check_backward([seqlen] * batch, dtype, True, 8, kv_heads, 0.0, normal_inputs=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [8, 2])
def test_uniform_and_ragged_tiling_cache(dtype, kv_heads):
    # Same shape and total, different valid prefix contents: cached tiling
    # must not infer or cache actual lengths from tensor shape/identity.
    for lengths in ([279] * 3, [278, 279, 280], [279] * 3):
        _check_backward(lengths, dtype, True, 8, kv_heads, 0.0, normal_inputs=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [64, 192, 256])
@pytest.mark.parametrize("causal", [False, True])
def test_varlen_other_head_dimensions(dtype, dim, causal):
    # Check the general-backward dispatch at other head dimensions as well
    # as the D=128 optimized path above, with unequal Q/KV gradient sizes.
    _check_backward(
        [129, 257] * 8 + [1], dtype, causal, 8, 2, 0.0, normal_inputs=True, dim=dim
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_large_batch_ragged_full_gqa(dtype):
    # The large-batch full-MHA route must not divert grouped-head workloads.
    _check_backward([5, 9] * 64 + [1], dtype, False, 8, 2, 0.0, normal_inputs=True)
