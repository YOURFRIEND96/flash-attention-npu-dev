"""Backward dispatch must inspect current prefixes without a host equality read."""

from itertools import accumulate

import pytest
import torch
import torch_npu  # noqa: F401

from flash_attn_npu import flash_attn_varlen_func
from scripts.benchmark_varlen_vs_cann import compare_dual, reference_chunk


def _check_packed_reference(cpu, dout, qlens, klens, actual, scale, causal, layout):
    """Use the same independent dual references as dynamic-metadata coverage."""
    references = [[], []]
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
        qi += nq
        ki += nk
    for component, (name, value) in enumerate(zip(("out", "dq", "dk", "dv"), actual)):
        high, low = [torch.cat([chunk[component][0] for chunk in ref]) for ref in references]
        check = compare_dual(value.cpu(), high, low)
        assert check["passed"], (layout, name, check)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("kv_heads", [8, 2], ids=["mha", "gqa4"])
def test_device_backward_dispatch_mutating_prefixes(dtype, causal, kv_heads):
    # Keep B, totalQ, totalK and both declared/actual maxima fixed. Equality
    # changes while every scalar tiling/cache key remains unchanged. The
    # unequal state includes Q>0/K=0, Q=0/K>0 and right-aligned invalid rows.
    equal_a = [0, 128, 129, 513, 1, 0]
    unequal_k = [1, 513, 0, 129, 0, 128]
    equal_b = [513, 0, 1, 129, 128, 0]
    layouts = [
        (equal_a, equal_a),
        (equal_a, unequal_k),
        (equal_b, equal_b),
        (equal_a, equal_a),
    ]
    total, maximum = 771, 513
    generator = torch.Generator().manual_seed(2026)
    cpu = [
        torch.randn(total, heads, 128, generator=generator).to(dtype)
        for heads in (8, kv_heads, kv_heads)
    ]
    dout = torch.randn(cpu[0].shape, generator=generator).to(dtype)
    inputs = [x.npu().requires_grad_() for x in cpu]
    dout_npu = dout.npu()
    # Contiguous views with distinct nonzero offsets ensure both native paths
    # read the logical tensor start, rather than the backing allocation start.
    cuq = torch.empty(len(equal_a) + 4, dtype=torch.int32, device=inputs[0].device)[3:]
    cuk = torch.empty(len(equal_a) + 3, dtype=torch.int32, device=inputs[0].device)[2:]
    assert cuq.data_ptr() != cuk.data_ptr()
    prefix_addresses = (cuq.data_ptr(), cuk.data_ptr())
    prefix_versions = (cuq._version, cuk._version)
    original_stream = torch.npu.current_stream()
    alternate_stream = torch.npu.Stream(device=inputs[0].device)
    alternate_stream.wait_stream(original_stream)
    scale = 128**-0.5

    for index, (qlens, klens) in enumerate(layouts):
        assert sum(qlens) == sum(klens) == total
        assert max(qlens) == max(klens) == maximum
        stream = original_stream if index % 2 == 0 else alternate_stream
        with torch.npu.stream(stream):
            # Deliberately bypass the original tensors' version counters, as
            # an external device producer can. A pointer/version cache must
            # not treat these mutations as unchanged prefix contents.
            cuq.data.copy_(torch.tensor([0, *accumulate(qlens)], dtype=torch.int32))
            cuk.data.copy_(torch.tensor([0, *accumulate(klens)], dtype=torch.int32))
            assert (cuq.data_ptr(), cuk.data_ptr()) == prefix_addresses
            assert (cuq._version, cuk._version) == prefix_versions
            out = flash_attn_varlen_func(
                *inputs,
                cuq,
                cuk,
                maximum,
                maximum,
                softmax_scale=scale,
                causal=causal,
            )
            # Initialize tiling and allocation resources before inspecting a
            # second backward. Reuse the graph, but inspect both route states.
            warm_grads = torch.autograd.grad(out, inputs, dout_npu, retain_graph=True)
            stream.synchronize()
            del warm_grads
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                grads = torch.autograd.grad(out, inputs, dout_npu)
                stream.synchronize()

        # The preceding synchronization also completes every use before the
        # next layout overwrites shared prefixes from the other stream.
        events = {event.key for event in prof.key_averages()}
        assert any("FlashAttnVarlenFuncBackward" in name for name in events), events
        forbidden = events & {"aten::equal", "aten::_local_scalar_dense"}
        assert not forbidden, (index, forbidden)
        # Do not ban all aten::_to_copy events: an allowed device dtype copy
        # is not evidence of a device-to-host transfer.
        _check_packed_reference(
            cpu, dout, qlens, klens, (out, *grads), scale, causal, index
        )
