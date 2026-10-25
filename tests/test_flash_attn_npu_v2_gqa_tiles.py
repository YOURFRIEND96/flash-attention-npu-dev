"""Grouped Q heads must preserve token-tile LSE extents, including tails."""

from itertools import accumulate

import pytest
import torch
import torch_npu  # noqa: F401

from flash_attn_npu import flash_attn_varlen_func
from scripts.benchmark_varlen_vs_cann import compare_dual, reference_chunk


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "qlens,klens",
    [
        ([128, 129], [128, 129]),
        ([257, 279], [257, 279]),
        ([256, 320], [256, 320]),
        ([279, 129], [257, 97]),
        ([129, 257], [257, 279]),
    ],
)
def test_gqa_token_tiles(dtype, causal, qlens, klens):
    torch.manual_seed(2026)
    cpu = [
        torch.randn(n, h, 128).to(dtype)
        for n, h in [(sum(qlens), 8), (sum(klens), 2), (sum(klens), 2)]
    ]
    dout = torch.randn(cpu[0].shape).to(dtype)
    inputs = [x.npu().requires_grad_() for x in cpu]
    cuq, cuk = [
        torch.tensor([0, *accumulate(lens)], dtype=torch.int32).npu() for lens in (qlens, klens)
    ]
    scale = 128**-0.5
    out, lse, _ = flash_attn_varlen_func(
        *inputs,
        cuq,
        cuk,
        max(qlens),
        max(klens),
        causal=causal,
        softmax_scale=scale,
        return_attn_probs=True,
    )
    grads = torch.autograd.grad(out, inputs, dout.npu())
    expected = [[], []]
    expected_lse = []
    qi = ki = 0
    for nq, nk in zip(qlens, klens):
        args = [
            cpu[0][qi : qi + nq][None],
            cpu[1][ki : ki + nk][None],
            cpu[2][ki : ki + nk][None],
            dout[qi : qi + nq][None],
        ]
        for index, upcast in enumerate((True, False)):
            expected[index].append(reference_chunk(*args, scale, causal, upcast=upcast))
        scores = (
            torch.einsum(
                "qhd,khd->hqk", args[0][0].float(), args[1][0].float().repeat_interleave(4, dim=1)
            )
            * scale
        )
        if causal:
            keep = torch.arange(nk)[None, :] <= torch.arange(nq)[:, None] + nk - nq
            scores.masked_fill_(~keep[None], float("-inf"))
        expected_lse.append(scores.logsumexp(-1).nan_to_num(neginf=float("inf")))
        qi += nq
        ki += nk
    for i, actual in enumerate((out, *grads)):
        high, low = [torch.cat([batch[i][0] for batch in ref]) for ref in expected]
        check = compare_dual(actual.cpu(), high, low)
        assert check["passed"], (i, check)
    torch.testing.assert_close(lse.cpu(), torch.cat(expected_lse, dim=1), atol=3e-5, rtol=3e-5)
