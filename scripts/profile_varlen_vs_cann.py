#!/usr/bin/env python3
"""Diagnostic timing/profiling with the exact benchmark inputs; no accuracy claims."""

import argparse
import json
from pathlib import Path
import sys

from benchmark_varlen_vs_cann import REPO, case_lengths, measure, parse_cases, prepare_case


def measure_phase(case, dtype, name, forward, inputs, dout, phase, args, torch, torch_npu):
    # Match run_case's tensor lifetimes: forward has no retained graph;
    # backward owns one graph, released before forward_backward starts.
    if phase == "backward":
        retained = forward()

        def fn():
            return torch.autograd.grad(retained, inputs, dout, retain_graph=True)

    elif phase == "forward_backward":

        def fn():
            return torch.autograd.grad(forward(), inputs, dout)

    else:
        fn = forward
    row = dict(case=case.name, dtype=dtype, operator=name, phase=phase, accuracy="not_checked")
    row.update(measure(fn, args.warmup, args.iterations, args.repeats))
    print(json.dumps(row), flush=True)
    if args.trace and phase != "forward_backward":
        tag = f"{case.name.replace(' ', '_')}_{dtype}_{name}_{phase}"
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            record_shapes=True,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=getattr(torch_npu.profiler.AiCMetrics, args.metrics),
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output_dir / tag)),
        ):
            for _ in range(3):
                with torch.autograd.profiler.record_function(tag):
                    fn()
                torch.npu.synchronize()
    return row


def measure_case(case, dtype, args, flash, torch, torch_npu):
    # A function scope releases all closures/tensors before the next case's
    # input allocation, as the full accuracy benchmark does in run_case.
    lengths = case_lengths(case, {}, args.seed)
    inputs, dout, fa, cann = prepare_case(case, *lengths, dtype, args, flash, torch_npu)
    rows = []
    for name, forward in (("flash", fa), ("cann", cann)):
        # Reproduce the benchmark's accuracy-run allocations without claiming
        # an accuracy check or paying for CPU references in this diagnostic.
        out = forward()
        grads = torch.autograd.grad(out, inputs, dout)
        torch.npu.synchronize()
        del out, grads
        for phase in ("forward", "backward", "forward_backward"):
            if args.phase and phase not in args.phase:
                continue
            rows.append(
                measure_phase(
                    case, dtype, name, forward, inputs, dout, phase, args, torch, torch_npu
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--case", action="append")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="fp16")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--metrics",
        choices=("PipeUtilization", "L2Cache", "Memory", "MemoryUB", "ResourceConflictRatio"),
        default="PipeUtilization",
        help="Hardware counters to collect with --trace; timing remains diagnostic",
    )
    parser.add_argument(
        "--phase", action="append", choices=("forward", "backward", "forward_backward")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu

    torch.set_num_threads(8)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    sys.path.insert(0, str(REPO))
    from flash_attn_npu import flash_attn_varlen_func

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in parse_cases(REPO.parent / "testcase.md"):
        if args.case and case.name not in args.case:
            continue
        for dtype in ("fp16", "bf16") if args.dtype == "both" else (args.dtype,):
            rows.extend(measure_case(case, dtype, args, flash_attn_varlen_func, torch, torch_npu))
            (args.output_dir / "timings.json").write_text(json.dumps(rows, indent=2) + "\n")
            torch.npu.empty_cache()


if __name__ == "__main__":
    main()
