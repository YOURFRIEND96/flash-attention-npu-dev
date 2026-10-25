#!/usr/bin/env python3
"""Compare FlashAttention v2 and CANN training attention on testcase.md.

Example (from the repository root, after building the v2 extension):
  python scripts/benchmark_varlen_vs_cann.py --device 1 --dtype both
  python scripts/benchmark_varlen_vs_cann.py --dry-run

The default CANN API is torch_npu.npu_fusion_attention_v3, whose sequence-length
arguments are CPU Tensors copied from the same NPU prefixes supplied to Flash.
The explicit device-to-host mode adapts these prefixes to the older int[] API
inside every forward; host-precomputed preserves the old diagnostic convention.
Both backward paths use torch.autograd.grad.
All output and Q/K/V gradient elements are checked against chunked CPU FP32
and low-precision references using the repository's dual-reference error rule.
Correctness, initial input transfers, input construction and masks are NOT timed.
Forward timing has autograd enabled; backward reuses a retained forward graph;
forward+backward creates a fresh graph each iteration. NPU events measure the
eager stream interval, including submission gaps, NOT isolated kernel time.
Prefix adaptation required by the selected CANN mode is included in F/F+B.

Missing per-sequence lengths are synthesized: reserve one maximum-length
sequence, balance the remaining tokens, then shuffle with a fixed seed. Supply
--lengths-json to override: {"T5 Encoder": {"q": [...], "kv": [...]}, ...}.
Reports include the full lengths, configuration, checks and timing samples.
"""

import argparse
import ast
from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from itertools import accumulate
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback


REPO = Path(__file__).resolve().parents[1]
API_DOC = (
    "https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/"
    "docs/en/custom_APIs/torch_npu/torch_npu-npu_fusion_attention.md"
)
CANN_LENGTHS_MODES = ("device-tensor", "device-to-host", "host-precomputed")


def add_cann_lengths_arguments(parser):
    parser.add_argument(
        "--cann-lengths-mode", choices=CANN_LENGTHS_MODES, default="device-tensor",
        help="device-tensor: timed bulk D2H to V3 CPU Tensor metadata; device-to-host: D2H to the "
        "legacy API; host-precomputed: legacy diagnostic without prefix-transfer cost",
    )


def cann_api_name(mode):
    if mode not in CANN_LENGTHS_MODES:
        raise ValueError(f"Unknown CANN lengths mode: {mode}")
    return "npu_fusion_attention_v3" if mode == "device-tensor" else "npu_fusion_attention"


def cann_lengths_description(mode):
    if mode == "device-tensor":
        return (
            "CANN lengths mode=device-tensor: Flash and CANN share the same NPU int32 "
            "cumulative-length sources. On this installation npu_fusion_attention_v3 "
            "requires CPU Tensor metadata: every forward performs .cpu()[1:] on each "
            "prefix, preserving int32 and dropping the initial zero. D2H is included in "
            "forward/forward_backward. CPU int32 was validated on this installation; "
            "the installed documentation lists int64 only. Retained backward uses native "
            "V3 autograd's saved metadata; no extra D2H is injected."
        )
    if mode == "device-to-host":
        return (
            "CANN lengths mode=device-to-host: each legacy npu_fusion_attention forward "
            "reads the same NPU int32 prefixes as Flash using one bulk .cpu().tolist()[1:] "
            "per prefix. D2H and Python-list conversion are included in forward/forward_backward. "
            "Retained backward reuses CANN autograd's saved metadata; no extra D2H is injected."
        )
    if mode == "host-precomputed":
        return (
            "CANN lengths mode=host-precomputed: legacy diagnostic using CPU cumulative "
            "length tuples prepared outside timing. This excludes device-prefix transfer "
            "cost and is not the shared-device-prefix comparison."
        )
    raise ValueError(f"Unknown CANN lengths mode: {mode}")


def make_cann_forward(q, k, v, cuq, cuk, torch_npu, kwargs, *,
                      lengths_mode="device-tensor", host_lengths=None):
    """Keep all requested prefix adaptation inside the measured forward call."""
    name = cann_api_name(lengths_mode)
    operator = getattr(torch_npu, name, None)
    if operator is None:
        raise RuntimeError(f"Selected CANN lengths mode {lengths_mode!r} requires torch_npu.{name}; "
                           "select another mode explicitly if this API is unavailable")
    if lengths_mode == "host-precomputed" and host_lengths is None:
        raise ValueError("host-precomputed requires explicit CPU cumulative lengths")

    def forward():
        if lengths_mode == "device-tensor":
            qends, kends = cuq.cpu()[1:], cuk.cpu()[1:]
        elif lengths_mode == "device-to-host":
            qends, kends = cuq.cpu().tolist()[1:], cuk.cpu().tolist()[1:]
        else:
            qends, kends = host_lengths
        return operator(q, k, v, actual_seq_qlen=qends, actual_seq_kvlen=kends, **kwargs)[0]

    return forward


@dataclass
class Case:
    name: str
    batch: int
    q_shape: tuple
    kv_shape: tuple
    out_shape: tuple
    max_q: int
    max_kv: int
    causal: bool
    scale: float


def parse_cases(path):
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [s.strip() for s in line.strip().strip("|").split("|")]
        if len(cells) != 8 or not cells[1].isdigit():
            continue
        name, batch, qs, ks, os, maximum, mask, scale = cells
        shapes = [tuple(ast.literal_eval(s)) for s in (qs, ks, os)]
        if any(len(s) != 3 or any(type(x) is not int or x <= 0 for x in s) for s in shapes):
            raise ValueError(f"{name}: expected positive TND shapes")
        q, kv, out = shapes
        if q != out or q[2] != kv[2] or q[1] % kv[1]:
            raise ValueError(f"{name}: incompatible Q/K/V/O shapes")
        mq, mk = map(int, maximum.split("/"))
        if mask not in ("causal", "full"):
            raise ValueError(f"{name}: unsupported mask {mask!r}")
        scale = scale.replace(" ", "")
        value = 1 / math.sqrt(float(scale[3:])) if scale.startswith("1/√") else float(scale)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name}: scale must be positive and finite")
        cases.append(Case(name, int(batch), q, kv, out, mq, mk, mask == "causal", value))
    if not cases or len({c.name for c in cases}) != len(cases):
        raise ValueError("No cases found, or duplicate case names")
    return cases


def validate_lengths(lengths, batch, total, maximum):
    if (
        len(lengths) != batch
        or any(type(n) is not int or n <= 0 for n in lengths)
        or sum(lengths) != total
        or max(lengths) != maximum
    ):
        raise ValueError(f"Lengths must have count={batch}, sum={total}, max={maximum}")
    return list(lengths)


def make_lengths(batch, total, maximum, seed):
    if batch < 1 or maximum < 1 or not maximum + batch - 1 <= total <= batch * maximum:
        raise ValueError(f"Impossible lengths: batch={batch}, total={total}, max={maximum}")
    if batch == 1:
        return [total]
    base, remainder = divmod(total - maximum, batch - 1)
    lengths = [maximum] + [base + (i < remainder) for i in range(batch - 1)]
    random.Random(seed).shuffle(lengths)
    return validate_lengths(lengths, batch, total, maximum)


def case_lengths(case, overrides, seed):
    override = overrides.get(case.name)
    if override is not None:
        q = validate_lengths(override["q"], case.batch, case.q_shape[0], case.max_q)
        k = validate_lengths(override["kv"], case.batch, case.kv_shape[0], case.max_kv)
        return q, k
    return (
        make_lengths(case.batch, case.q_shape[0], case.max_q, seed),
        make_lengths(case.batch, case.kv_shape[0], case.max_kv, seed),
    )


def reference_chunk(q, k, v, dout, scale, causal, upcast=True):
    """Independent batched math; tensors are [B,S,H,D], including GQA."""
    import torch

    leaves = [
        (x.detach().float() if upcast else x.detach()).requires_grad_(True) for x in (q, k, v)
    ]
    qr, kr, vr = leaves
    repeats = qr.shape[2] // kr.shape[2]
    kr = kr.repeat_interleave(repeats, dim=2)
    vr = vr.repeat_interleave(repeats, dim=2)
    # Match the two-reference convention used in tests/common/attention_ref.py.
    qr, kr = qr.transpose(1, 2), kr.permute(0, 2, 3, 1)
    logits = (qr * scale) @ kr if upcast else qr @ (kr * scale)
    if causal:
        sq, sk = q.shape[1], k.shape[1]
        row = torch.arange(sq, device=q.device)[:, None]
        col = torch.arange(sk, device=q.device)[None, :]
        allowed = col <= row + sk - sq
        logits = logits.masked_fill(~allowed, float("-inf"))
        # Fully masked rows must yield zero outputs and gradients.
        valid = allowed.any(dim=-1)
        logits = torch.where(valid[None, None, :, None], logits, 0.0)
        probs = logits.softmax(dim=-1) * valid[None, None, :, None]
    else:
        probs = logits.softmax(dim=-1)
    out = (probs @ vr.transpose(1, 2)).transpose(1, 2)
    grads = torch.autograd.grad(out, leaves, dout.to(out.dtype))
    return (out.detach(), *(g.detach() for g in grads))


def reference_all(q, k, v, dout, qlens, klens, case, chunk_size, upcast=True):
    """Group equal lengths and bound attention-matrix memory with small chunks."""
    import torch

    q, k, v, dout = [x.detach().cpu() for x in (q, k, v, dout)]
    results = [
        torch.empty(x.shape, dtype=torch.float32 if upcast else q.dtype) for x in (q, q, k, v)
    ]
    cq, ck = [0, *accumulate(qlens)], [0, *accumulate(klens)]
    groups = defaultdict(list)
    for b, lengths in enumerate(zip(qlens, klens)):
        groups[lengths].append(b)
    for indices in groups.values():
        for start in range(0, len(indices), chunk_size):
            batch = indices[start : start + chunk_size]
            args = [
                torch.stack([x[offsets[b] : offsets[b + 1]] for b in batch])
                for x, offsets in ((q, cq), (k, ck), (v, ck), (dout, cq))
            ]
            values = reference_chunk(*args, case.scale, case.causal, upcast=upcast)
            for dest, value, offsets in zip(results, values, (cq, cq, ck, ck)):
                for i, b in enumerate(batch):
                    dest[offsets[b] : offsets[b + 1]] = value[i]
    return results


def compare(actual, expected, atol, rtol):
    import torch

    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    if actual.shape != expected.shape:
        return {"passed": False, "error": "shape mismatch"}
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        return {
            "passed": False,
            "error": "nonfinite tensor",
            "actual_nan": int(actual.isnan().sum()),
            "actual_inf": int(actual.isinf().sum()),
            "reference_nan": int(expected.isnan().sum()),
            "reference_inf": int(expected.isinf().sum()),
        }
    diff = (actual - expected).abs()
    bad = diff > atol + rtol * expected.abs()
    return {
        "passed": not bool(bad.any()),
        "max_abs": diff.max().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "relative_l2": (diff.norm() / expected.norm().clamp_min(1e-12)).item(),
        "bad_elements": int(bad.sum()),
        "elements": actual.numel(),
        "atol": atol,
        "rtol": rtol,
    }


def compare_dual(actual, high, low):
    """Repository rule: max error <= max(2*low_precision_error + 2*ULP, 1e-5)."""
    import torch

    actual = actual.detach().cpu()
    # The repository reference returns outputs/gradients in the input dtype.
    ref, low = high.to(actual.dtype), low.to(actual.dtype)
    if not bool(torch.isfinite(ref).all() and torch.isfinite(low).all()):
        return {"passed": False, "error": "nonfinite reference"}
    baseline_error = (low.float() - ref.float()).abs().max().item()
    ulp = (ref + 0.3 - 0.3 - ref).abs().max().item()
    tolerance = max(2.0 * baseline_error + 2.0 * ulp, 1e-5)
    metrics = compare(actual, ref, tolerance, 0.0)
    metrics.update(rule="dual_reference", low_precision_max_abs=baseline_error, ulp=ulp)
    return metrics


def measure(fn, warmup, iterations, repeats):
    import torch

    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    event_samples, wall_samples = [], []
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    # Initialize event resources outside the measured repetitions.
    start.record()
    end.record()
    torch.npu.synchronize()
    for _ in range(repeats):
        torch.npu.synchronize()
        wall_start = time.perf_counter()
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        wall_samples.append((time.perf_counter() - wall_start) * 1e6 / iterations)
        event_samples.append(start.elapsed_time(end) * 1000 / iterations)
    return {
        "event_us": statistics.median(event_samples),
        "wall_us": statistics.median(wall_samples),
        "event_samples_us": event_samples,
        "wall_samples_us": wall_samples,
    }


def prepare_case(case, qlens, klens, dtype_name, args, flash, torch_npu):
    import torch

    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    # CPU-generated inputs are reproducible and identical for both operators.
    q, k, v = [
        torch.randn(shape, dtype=torch.float32).to(dtype).to("npu").requires_grad_(True)
        for shape in (case.q_shape, case.kv_shape, case.kv_shape)
    ]
    dout = torch.randn(case.out_shape, dtype=torch.float32).to(dtype).to("npu")
    cq, ck = [0, *accumulate(qlens)], [0, *accumulate(klens)]
    cuq = torch.tensor(cq, dtype=torch.int32, device="npu")
    cuk = torch.tensor(ck, dtype=torch.int32, device="npu")
    # CANN sparse_mode=3 requires the compressed 2048x2048 causal template.
    mask = torch.ones((2048, 2048), dtype=torch.bool).triu(1).to("npu") if case.causal else None
    cann_kwargs = dict(
        head_num=case.q_shape[1],
        input_layout="TND",
        atten_mask=mask,
        scale=case.scale,
        keep_prob=1.0,
        pre_tockens=2147483647,
        next_tockens=0 if case.causal else 2147483647,
        sparse_mode=3 if case.causal else 0,
    )

    def fa_forward():
        return flash(
            q,
            k,
            v,
            cuq,
            cuk,
            case.max_q,
            case.max_kv,
            dropout_p=0.0,
            softmax_scale=case.scale,
            causal=case.causal,
        )

    cann_mode = getattr(args, "cann_lengths_mode", "device-tensor")
    cann_forward = make_cann_forward(
        q, k, v, cuq, cuk, torch_npu, cann_kwargs,
        lengths_mode=cann_mode,
        host_lengths=(tuple(cq[1:]), tuple(ck[1:])) if cann_mode == "host-precomputed" else None,
    )

    return (q, k, v), dout, fa_forward, cann_forward


def run_case(case, qlens, klens, dtype_name, args, flash, torch_npu):
    import torch

    (q, k, v), dout, fa_forward, cann_forward = prepare_case(
        case, qlens, klens, dtype_name, args, flash, torch_npu
    )
    result = {"case": case.name, "dtype": dtype_name, "status": "running", "operators": {}}
    atol = args.atol if args.atol is not None else (0.005 if dtype_name == "fp16" else 0.05)
    rtol = args.rtol if args.rtol is not None else (0.005 if dtype_name == "fp16" else 0.05)
    print(f"  Computing full CPU dual references: {case.name} / {dtype_name}", flush=True)
    golden = reference_all(q, k, v, dout, qlens, klens, case, args.reference_batch)
    low = reference_all(q, k, v, dout, qlens, klens, case, args.reference_batch, upcast=False)
    outputs = {}
    for name, forward in (("flash", fa_forward), ("cann", cann_forward)):
        print(f"  Checking and timing {name}", flush=True)
        out = forward()
        grads = torch.autograd.grad(out, (q, k, v), dout)
        values = [x.detach().cpu() for x in (out, *grads)]
        outputs[name] = values
        checks = {}
        for label, actual, ref, pt in zip(("out", "dq", "dk", "dv"), values, golden, low):
            if args.atol is not None or args.rtol is not None:
                checks[label] = compare(actual, ref, atol, rtol)
                checks[label]["rule"] = "pointwise_override"
            else:
                checks[label] = compare_dual(actual, ref, pt)
        print(f"    accuracy: { {k: v['passed'] for k, v in checks.items()} }", flush=True)
        del out, grads
        timing = {"forward": measure(forward, args.warmup, args.iterations, args.repeats)}
        retained_out = forward()

        def backward(out=retained_out):
            return torch.autograd.grad(out, (q, k, v), dout, retain_graph=True)

        timing["backward"] = measure(backward, args.warmup, args.iterations, args.repeats)
        del backward, retained_out

        def training_step():
            return torch.autograd.grad(forward(), (q, k, v), dout)

        timing["forward_backward"] = measure(
            training_step, args.warmup, args.iterations, args.repeats
        )
        result["operators"][name] = {"checks": checks, "timing": timing}
    result["cross_comparison"] = {
        label: compare(fa, cann, atol, rtol)
        for label, fa, cann in zip(("out", "dq", "dk", "dv"), outputs["flash"], outputs["cann"])
    }
    result["status"] = (
        "passed"
        if all(
            check["passed"]
            for op in result["operators"].values()
            for check in op["checks"].values()
        )
        else "accuracy_failed"
    )
    result["speedup"] = {
        phase: result["operators"]["cann"]["timing"][phase]["event_us"]
        / result["operators"]["flash"]["timing"][phase]["event_us"]
        for phase in ("forward", "backward", "forward_backward")
    }
    return result


def report_table(report):
    lines = [
        "| Case | dtype | phase | Flash (us) | CANN (us) | CANN / Flash | accuracy |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in report["results"]:
        for phase in ("forward", "backward", "forward_backward"):
            if "speedup" not in row:
                lines.append(
                    f"| {row['case']} | {row['dtype']} | {phase} | N/A | N/A | N/A | {row['status']} |"
                )
                continue
            fa = row["operators"]["flash"]["timing"][phase]["event_us"]
            cann = row["operators"]["cann"]["timing"][phase]["event_us"]
            ratio = row["speedup"][phase]
            lines.append(
                f"| {row['case']} | {row['dtype']} | {phase} | {fa:.3f} | {cann:.3f} | {ratio:.3f}x | {row['status']} |"
            )
    return "\n".join(lines)


def performance_averages(report):
    """Equal-case means and total-time ratio; never hide missing/failed cases."""
    expected = len(report["cases"]) * (2 if report["config"]["dtype"] == "both" else 1)
    summary = {}
    for phase in ("forward", "backward", "forward_backward"):
        pairs = []
        for row in report["results"]:
            if row["status"] != "passed":
                continue
            timing = row.get("operators", {})
            try:
                fa, cann = [timing[name]["timing"][phase]["event_us"] for name in ("flash", "cann")]
            except KeyError:
                continue
            if all(math.isfinite(value) and value > 0 for value in (fa, cann)):
                pairs.append((fa, cann))
        ratios = [cann / fa for fa, cann in pairs]
        count = len(pairs)
        geometric = math.exp(sum(map(math.log, ratios)) / count) if count else None
        summary[phase] = {
            "included_cases": count,
            "expected_cases": expected,
            "complete": count == expected and expected > 0,
            "geometric_speedup": geometric,
            "arithmetic_speedup": sum(ratios) / count if count else None,
            "total_time_speedup": sum(c for _, c in pairs) / sum(f for f, _ in pairs)
            if count
            else None,
            "individual_cases_within_5pct": sum(r >= 1 / 1.05 for r in ratios),
            "mean_within_5pct": geometric >= 1 / 1.05 if count == expected and count else None,
        }
    return summary


def write_report(report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    report["averages"] = performance_averages(report)
    (directory / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    notes = (
        "NPU Event eager stream latency; includes submission gaps, not isolated kernel time. "
        "Median of repeated loop averages. CANN / Flash > 1 means Flash is faster.\n\n"
        "Forward records autograd; backward reuses a retained graph; forward_backward "
        "builds a fresh graph per iteration. Input/mask construction and reference checks are excluded.\n\n"
        "All out/dQ/dK/dV elements are checked against CPU FP32 and low-precision attention "
        "using the repository dual-reference rule (unless --atol/--rtol override it). "
        "Performance from accuracy_failed rows is diagnostic only.\n\n"
        "cross_comparison in JSON is an additional pointwise diagnostic between the two "
        "operators; the overall status comes from their independent reference checks.\n\n"
        "Sequence lengths are synthetic unless supplied with --lengths-json; "
        "see results.json for full lengths, versions, tolerances and samples.\n\n"
    )
    # Historical reports without this option used precomputed Host tuples.
    notes += cann_lengths_description(
        report["config"].get("cann_lengths_mode", "host-precomputed"),
    ) + "\n\n"
    environment = report.get("environment", {})
    if environment:
        notes += (
            "Environment: "
            + ", ".join(
                f"{key}={environment[key]}"
                for key in ("device", "device_index", "torch", "torch_npu", "cann")
            )
            + "\n\n"
        )
    errors = "\n".join(
        f"- {row['case']} / {row['dtype']}: {row['error']}"
        for row in report["results"]
        if "error" in row
    )
    accuracy = [
        "| Case | dtype | operator | out | dQ | dK | dV |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["results"]:
        for name, op in row.get("operators", {}).items():
            cells = []
            for label in ("out", "dq", "dk", "dv"):
                check = op["checks"][label]
                status = "PASS" if check["passed"] else "FAIL"
                cells.append(
                    f"{status} ({check['max_abs']:.4g})"
                    if "max_abs" in check
                    else f"{status} ({check.get('error', 'unknown error')})"
                )
            accuracy.append(
                f"| {row['case']} | {row['dtype']} | {name} | " + " | ".join(cells) + " |"
            )
    averages = [
        "Average speedups (CANN / Flash). Only accuracy-passed rows are included; incomplete coverage is not an acceptance result.",
        "",
        "| Phase | Checked / expected | Geometric mean | Arithmetic mean | Total-time ratio | Complete |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for phase, values in report["averages"].items():
        means = [
            f"{values[key]:.3f}x" if values[key] is not None else "N/A"
            for key in ("geometric_speedup", "arithmetic_speedup", "total_time_speedup")
        ]
        averages.append(
            f"| {phase} | {values['included_cases']} / {values['expected_cases']} | "
            + " | ".join(means)
            + f" | {values['complete']} |"
        )
    (directory / "results.md").write_text(
        notes
        + "\n".join(averages)
        + "\n\n"
        + report_table(report)
        + "\n\nAccuracy (maximum absolute error in parentheses):\n\n"
        + "\n".join(accuracy)
        + "\n\n"
        + errors
        + "\n",
        encoding="utf-8",
    )
    with (directory / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case",
                "dtype",
                "status",
                "phase",
                "flash_event_us",
                "cann_event_us",
                "cann_over_flash",
                "flash_wall_us",
                "cann_wall_us",
                "cann_api",
                "cann_lengths_mode",
            ]
        )
        for row in report["results"]:
            for phase in ("forward", "backward", "forward_backward"):
                if "speedup" not in row:
                    writer.writerow(
                        [row["case"], row["dtype"], row["status"], phase, "", "", "", "", "",
                         report.get("cann_api", "torch_npu.npu_fusion_attention"),
                         report["config"].get("cann_lengths_mode", "host-precomputed")]
                    )
                    continue
                fa, cann = [row["operators"][op]["timing"][phase] for op in ("flash", "cann")]
                writer.writerow(
                    [
                        row["case"],
                        row["dtype"],
                        row["status"],
                        phase,
                        fa["event_us"],
                        cann["event_us"],
                        row["speedup"][phase],
                        fa["wall_us"],
                        cann["wall_us"],
                        report.get("cann_api", "torch_npu.npu_fusion_attention"),
                        report["config"].get("cann_lengths_mode", "host-precomputed"),
                    ]
                )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cases", type=Path, default=REPO.parent / "testcase.md")
    parser.add_argument("--case", action="append", help="Exact case name; repeat to select several")
    parser.add_argument("--lengths-json", type=Path)
    add_cann_lengths_arguments(parser)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="both")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reference-batch", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument(
        "--output-dir", type=Path, default=REPO.parent / "benchmark_results" / "varlen_vs_cann"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate cases and lengths without importing torch"
    )
    args = parser.parse_args(argv)
    if (
        min(args.iterations, args.repeats, args.reference_batch, args.cpu_threads) <= 0
        or args.warmup < 0
    ):
        parser.error(
            "Iteration/repeat/batch/thread counts must be positive; warmup must be nonnegative"
        )
    for value in (args.atol, args.rtol):
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error("Tolerances must be finite and nonnegative")
    cases = parse_cases(args.cases)
    if args.case:
        unknown = set(args.case) - {c.name for c in cases}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        cases = [c for c in cases if c.name in args.case]
    overrides = json.loads(args.lengths_json.read_text()) if args.lengths_json else {}
    lengths = {c.name: case_lengths(c, overrides, args.seed) for c in cases}
    dtypes = ("fp16", "bf16") if args.dtype == "both" else (args.dtype,)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cann_api": "torch_npu." + cann_api_name(args.cann_lengths_mode),
        "api_documentation": (
            "torch_npu/_op_plugin_docs.py:npu_fusion_attention_v3 (installed documentation)"
            if args.cann_lengths_mode == "device-tensor" else API_DOC
        ),
        "cann_lengths_description": cann_lengths_description(args.cann_lengths_mode),
        "cann_prefix_source_dtype": "int32",
        "cases": [
            {
                **asdict(c),
                "q_lengths": lengths[c.name][0],
                "kv_lengths": lengths[c.name][1],
                "length_source": "provided" if c.name in overrides else "synthetic_balanced",
            }
            for c in cases
        ],
        "results": [],
    }
    for c in cases:
        ql, kl = lengths[c.name]
        print(
            f"{c.name}: Q={c.q_shape}, KV={c.kv_shape}, B={c.batch}, "
            f"lengths Q={min(ql)}..{max(ql)}, KV={min(kl)}..{max(kl)}, scale={c.scale:.8g}",
            flush=True,
        )
    if args.dry_run:
        report["results"] = [
            {"case": c.name, "dtype": dt, "status": "dry_run"} for c in cases for dt in dtypes
        ]
        write_report(report, args.output_dir)
        return 0
    try:
        import torch
        import torch_npu

        torch.set_num_threads(args.cpu_threads)
        if not torch.npu.is_available():
            raise RuntimeError("No accessible NPU. Run on an Ascend 910 host with device access.")
        torch.npu.set_device(args.device)
        name = torch.npu.get_device_name(args.device)
        if "Ascend910" not in name:
            raise RuntimeError(f"FlashAttention v2 requires Ascend 910; got {name}")
        torch.npu.set_compile_mode(jit_compile=False)
        sys.path.insert(0, str(REPO))
        from flash_attn_npu import flash_attn_varlen_func
        import flash_attn_npu

        report["environment"] = {
            "device": name,
            "device_index": args.device,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "cann": torch_npu.utils.get_cann_version(),
            "flash_version": flash_attn_npu.__version__,
            "flash_module": flash_attn_npu.__file__,
            "jit_compile": False,
            "python": sys.version,
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "launch_environment": {
                key: os.environ.get(key)
                for key in (
                    "ASCEND_LAUNCH_BLOCKING",
                    "TASK_QUEUE_ENABLE",
                    "ASCEND_RT_VISIBLE_DEVICES",
                )
            },
        }
        # Identify the installed binary even when the working tree is dirty
        # or a subsequent experiment rebuilds the same extension path.
        binaries = {}
        for path in Path(flash_attn_npu.__file__).parent.glob("*.so"):
            with path.open("rb") as binary:
                binaries[str(path)] = hashlib.file_digest(binary, "sha256").hexdigest()
        report["environment"]["native_binary_sha256"] = binaries
        interface_path = REPO / "flash_attn_npu" / "flash_attn_npu_interface.py"
        with interface_path.open("rb") as interface_file:
            report["environment"]["interface_sha256"] = hashlib.file_digest(
                interface_file, "sha256"
            ).hexdigest()
    except Exception as exc:
        report["results"] = [
            {"case": c.name, "dtype": dt, "status": "environment_error", "error": str(exc)}
            for c in cases
            for dt in dtypes
        ]
        write_report(report, args.output_dir)
        print(f"Environment error: {exc}", file=sys.stderr)
        return 2
    for c in cases:
        for dt in dtypes:
            try:
                result = run_case(c, *lengths[c.name], dt, args, flash_attn_varlen_func, torch_npu)
            except Exception as exc:
                traceback.print_exc()
                result = {"case": c.name, "dtype": dt, "status": "runtime_error", "error": str(exc)}
            report["results"].append(result)
            write_report(report, args.output_dir)
            print(f"  {c.name} / {dt}: {result['status']}", flush=True)
            torch.npu.empty_cache()
    print("\n" + report_table(report))
    print(f"\nReports: {args.output_dir.resolve()}")
    return 0 if all(r["status"] == "passed" for r in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
