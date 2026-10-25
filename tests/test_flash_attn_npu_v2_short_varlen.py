"""Short Vector path: head grouping, ragged/right-aligned masks and NT LSE."""

from itertools import accumulate

import pytest
import torch
import torch_npu  # noqa: F401

from flash_attn_npu import flash_attn_varlen_func
from scripts.benchmark_varlen_vs_cann import compare_dual, reference_chunk


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "qlens,klens",
    [
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        ([4, 1, 3, 2], [1, 4, 2, 3]),
        ([1, 2, 3, 4], [31, 32, 33, 64]),
        ([4, 3, 2, 1], [128, 129, 257, 512]),
        ([0, 1, 4, 3, 2, 4], [33, 0, 1, 32, 257, 512]),
    ],
)
def test_short_varlen_forward_backward(dtype, heads, causal, qlens, klens):
    torch.manual_seed(17)
    cpu = [
        (torch.rand(n, heads, 128) - 0.5).to(dtype) for n in (sum(qlens), sum(klens), sum(klens))
    ]
    dout = (torch.rand_like(cpu[0].float()) - 0.5).to(dtype)
    inputs = [x.npu().requires_grad_() for x in cpu]
    cuq, cuk = [
        torch.tensor([0, *accumulate(lens)], dtype=torch.int32).npu() for lens in (qlens, klens)
    ]
    out, lse, _ = flash_attn_varlen_func(
        *inputs,
        cuq,
        cuk,
        max(qlens),
        max(klens),
        causal=causal,
        softmax_scale=1.0,
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
            expected[index].append(reference_chunk(*args, 1.0, causal, upcast=upcast))
        scores = torch.einsum("qhd,khd->hqk", args[0][0].float(), args[1][0].float())
        if causal:
            keep = torch.arange(nk)[None, :] <= torch.arange(nq)[:, None] + nk - nq
            scores.masked_fill_(~keep[None], float("-inf"))
        expected_lse.append(
            torch.logsumexp(scores, -1).masked_fill(
                torch.isneginf(torch.logsumexp(scores, -1)), float("inf")
            )
        )
        qi += nq
        ki += nk
    for i, actual in enumerate((out, *grads)):
        high, low = [torch.cat([batch[i][0] for batch in ref]) for ref in expected]
        check = compare_dual(actual.cpu(), high, low)
        assert check["passed"], (i, check)
    torch.testing.assert_close(lse.cpu(), torch.cat(expected_lse, dim=1), atol=3e-5, rtol=3e-5)
