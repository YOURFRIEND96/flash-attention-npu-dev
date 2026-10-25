"""Private scalar tiling must read changing ragged prefixes on every call."""

from itertools import accumulate

import pytest
import torch
import torch_npu  # noqa: F401

from flash_attn_npu import flash_attn_varlen_func
from scripts.benchmark_varlen_vs_cann import compare_dual, reference_chunk


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_ragged_prefix_mutation_across_streams(dtype, causal, monkeypatch):
    from flash_attn_npu import flash_attn_npu_interface as interface

    def unexpected_metadata(*args, **kwargs):
        raise AssertionError("plain eager TND must use actual-prefix task counting")

    monkeypatch.setattr(interface, "get_scheduler_metadata", unexpected_metadata)
    # Every configuration has B=5, totalQ=770 and totalK=804, with identical
    # declared maxima. Actual tile/head grouping changes at 128/129 and512/513.
    # Empty first/middle/last Q batches and nonempty Q with empty K are intentional.
    layouts = [
        ([0, 128, 129, 513, 0], [33, 0, 257, 511, 3]),
        ([129, 0, 512, 0, 129], [0, 129, 511, 164, 0]),
        ([128, 129, 0, 0, 513], [131, 0, 33, 0, 640]),
    ]
    layouts.append(layouts[0])
    generator = torch.Generator().manual_seed(2026)
    cpu = [
        torch.randn(n, heads, 128, generator=generator).to(dtype)
        for n, heads in ((770, 8), (804, 2), (804, 2))
    ]
    dout = torch.randn(cpu[0].shape, generator=generator).to(dtype)
    inputs = [x.npu().requires_grad_() for x in cpu]
    dout_npu = dout.npu()
    cuq = torch.empty(6, dtype=torch.int32, device=inputs[0].device)
    cuk = torch.empty_like(cuq)
    prefix_addresses = (cuq.data_ptr(), cuk.data_ptr())
    original_stream = torch.npu.current_stream()
    alternate_stream = torch.npu.Stream(device=inputs[0].device)
    alternate_stream.wait_stream(original_stream)
    scale = 128**-0.5

    for index, (qlens, klens) in enumerate(layouts):
        assert sum(qlens) == 770 and sum(klens) == 804
        assert max(qlens) <= 640 and max(klens) <= 640
        stream = original_stream if index % 2 == 0 else alternate_stream
        with torch.npu.stream(stream):
            # Reuse the same Tensor objects/storage: neither identity nor shape
            # may stand in for inspecting their updated contents.
            cuq.copy_(torch.tensor([0, *accumulate(qlens)], dtype=torch.int32))
            cuk.copy_(torch.tensor([0, *accumulate(klens)], dtype=torch.int32))
            out, lse, _ = flash_attn_varlen_func(
                *inputs,
                cuq,
                cuk,
                640,
                640,
                softmax_scale=scale,
                causal=causal,
                return_attn_probs=True,
            )
            grads = torch.autograd.grad(out, inputs, dout_npu)
        # Complete all uses before mutating shared prefixes on another stream.
        stream.synchronize()
        assert (cuq.data_ptr(), cuk.data_ptr()) == prefix_addresses

        references = [[], []]
        expected_lse = []
        qi = ki = 0
        for nq, nk in zip(qlens, klens):
            args = [
                cpu[0][qi : qi + nq][None],
                cpu[1][ki : ki + nk][None],
                cpu[2][ki : ki + nk][None],
                dout[qi : qi + nq][None],
            ]
            for target, upcast in zip(references, (True, False)):
                target.append(reference_chunk(*args, scale, causal, upcast=upcast))
            scores = torch.einsum(
                "qhd,khd->hqk", args[0][0].float(), args[1][0].float().repeat_interleave(4, dim=1)
            ) * scale
            if causal:
                keep = torch.arange(nk)[None, :] <= torch.arange(nq)[:, None] + nk - nq
                scores.masked_fill_(~keep[None], float("-inf"))
            expected_lse.append(scores.logsumexp(-1).nan_to_num(neginf=float("inf")))
            qi += nq
            ki += nk
        for component, actual in enumerate((out, *grads)):
            high, low = [torch.cat([chunk[component][0] for chunk in ref]) for ref in references]
            check = compare_dual(actual.cpu(), high, low)
            assert check["passed"], (index, component, check)
        torch.testing.assert_close(lse.cpu(), torch.cat(expected_lse, dim=1), atol=3e-5, rtol=3e-5)
