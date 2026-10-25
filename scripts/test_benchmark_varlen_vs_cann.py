"""CPU-only checks: python -m unittest discover -s scripts -p 'test_benchmark*.py'."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

import benchmark_varlen_vs_cann as bench


class BenchmarkTests(unittest.TestCase):
    def test_device_tensor_prefixes_are_copied_on_every_forward(self):
        class Prefix:
            def __init__(self, values):
                self.tensor = torch.tensor(values, dtype=torch.int32)
                self.copies = 0

            def cpu(self):
                self.copies += 1
                return self.tensor

        cuq, cuk = Prefix([0, 2, 5]), Prefix([0, 4, 7])
        q, k, v = (torch.randn(5, 2).requires_grad_() for _ in range(3))
        calls = []

        def v3(qq, kk, vv, **kwargs):
            self.assertIs(qq, q)
            self.assertIs(kk, k)
            self.assertIs(vv, v)
            qends, kends = kwargs["actual_seq_qlen"], kwargs["actual_seq_kvlen"]
            self.assertEqual(qends.dtype, torch.int32)
            self.assertEqual(kends.dtype, torch.int32)
            self.assertEqual(qends.device.type, "cpu")
            self.assertEqual(kends.device.type, "cpu")
            self.assertEqual(qends.data_ptr(), cuq.tensor[1:].data_ptr())
            self.assertEqual(kends.data_ptr(), cuk.tensor[1:].data_ptr())
            calls.append((qends.clone(), kends.clone()))
            return ((qq + kk + vv) * qends.sum(),)

        forward = bench.make_cann_forward(
            q, k, v, cuq, cuk,
            SimpleNamespace(npu_fusion_attention_v3=v3), {"input_layout": "TND"},
        )
        self.assertEqual(calls, [])
        self.assertEqual((cuq.copies, cuk.copies), (0, 0))
        out = forward()
        self.assertEqual((cuq.copies, cuk.copies), (1, 1))
        for _ in range(2):
            grads = torch.autograd.grad(out, (q, k, v), torch.ones_like(out), retain_graph=True)
            for grad in grads:
                torch.testing.assert_close(grad, torch.full_like(grad, 7))
        self.assertEqual((cuq.copies, cuk.copies), (1, 1))
        cuq.tensor.copy_(torch.tensor([0, 3, 5], dtype=torch.int32))
        cuk.tensor.copy_(torch.tensor([0, 1, 7], dtype=torch.int32))
        forward()
        self.assertEqual((cuq.copies, cuk.copies), (2, 2))
        self.assertEqual([tuple(x.tolist() for x in call) for call in calls],
                         [([2, 5], [4, 7]), ([3, 5], [1, 7])])

    def test_device_to_host_copies_once_per_prefix_and_not_in_retained_backward(self):
        class Prefix:
            def __init__(self, values):
                self.tensor = torch.tensor(values, dtype=torch.int32)
                self.copies = 0

            def cpu(self):
                self.copies += 1
                return self.tensor

        cuq, cuk = Prefix([0, 2, 5]), Prefix([0, 3, 5])
        inputs = tuple(torch.randn(5, 2).requires_grad_() for _ in range(3))
        calls = []

        def legacy(q, k, v, **kwargs):
            qends, kends = kwargs["actual_seq_qlen"], kwargs["actual_seq_kvlen"]
            self.assertIsInstance(qends, list)
            self.assertIsInstance(kends, list)
            self.assertTrue(all(type(value) is int for value in (*qends, *kends)))
            calls.append((qends, kends))
            return ((q + k + v) * sum(qends),)

        forward = bench.make_cann_forward(
            *inputs, cuq, cuk, SimpleNamespace(npu_fusion_attention=legacy), {},
            lengths_mode="device-to-host",
        )
        self.assertEqual((cuq.copies, cuk.copies), (0, 0))
        out = forward()
        self.assertEqual((cuq.copies, cuk.copies), (1, 1))
        for _ in range(2):
            grads = torch.autograd.grad(out, inputs, torch.ones_like(out), retain_graph=True)
            for grad in grads:
                torch.testing.assert_close(grad, torch.full_like(grad, 7))
        self.assertEqual((cuq.copies, cuk.copies), (1, 1))
        cuq.tensor.copy_(torch.tensor([0, 4, 5], dtype=torch.int32))
        cuk.tensor.copy_(torch.tensor([0, 1, 5], dtype=torch.int32))
        forward()
        self.assertEqual((cuq.copies, cuk.copies), (2, 2))
        self.assertEqual(calls, [([2, 5], [3, 5]), ([4, 5], [1, 5])])

    def test_host_precomputed_is_explicit_and_device_tensor_never_silently_falls_back(self):
        calls = []

        def legacy(q, k, v, **kwargs):
            calls.append((kwargs["actual_seq_qlen"], kwargs["actual_seq_kvlen"]))
            return (q,)

        module = SimpleNamespace(npu_fusion_attention=legacy)
        # Plain objects have no CPU/cast methods: this explicit legacy mode
        # must use the supplied Host metadata without inspecting the prefixes.
        q, k, v, cuq, cuk = (object() for _ in range(5))
        forward = bench.make_cann_forward(
            q, k, v, cuq, cuk, module, {}, lengths_mode="host-precomputed",
            host_lengths=((2, 5), (4, 7)),
        )
        forward()
        self.assertEqual(calls, [((2, 5), (4, 7))])
        with self.assertRaisesRegex(RuntimeError, "requires torch_npu.npu_fusion_attention_v3"):
            bench.make_cann_forward(q, k, v, cuq, cuk, module, {})
        self.assertEqual(len(calls), 1)

    def test_average_ratios_and_failed_case_coverage(self):
        def row(flash, cann, status="passed"):
            return {
                "status": status,
                "operators": {
                    name: {"timing": {"forward": {"event_us": value}}}
                    for name, value in (("flash", flash), ("cann", cann))
                },
            }

        report = {"cases": [{}, {}], "config": {"dtype": "fp16"}, "results": [row(1, 2), row(4, 2)]}
        result = bench.performance_averages(report)["forward"]
        self.assertAlmostEqual(result["geometric_speedup"], 1.0)
        self.assertAlmostEqual(result["arithmetic_speedup"], 1.25)
        self.assertAlmostEqual(result["total_time_speedup"], 0.8)
        self.assertTrue(result["complete"])
        self.assertTrue(result["mean_within_5pct"])
        report["cases"].append({})
        report["results"].append(row(0.01, 1000, "accuracy_failed"))
        result = bench.performance_averages(report)["forward"]
        self.assertAlmostEqual(result["geometric_speedup"], 1.0)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["mean_within_5pct"])

    def test_markdown_cases_and_exact_lengths(self):
        cases = bench.parse_cases(bench.REPO.parent / "testcase.md")
        self.assertEqual(len(cases), 5)
        for case in cases:
            q, k = bench.case_lengths(case, {}, 2026)
            for lengths, total, maximum in (
                (q, case.q_shape[0], case.max_q),
                (k, case.kv_shape[0], case.max_kv),
            ):
                self.assertEqual(len(lengths), case.batch)
                self.assertEqual(sum(lengths), total)
                self.assertEqual(max(lengths), maximum)
                self.assertGreater(min(lengths), 0)
            self.assertEqual((q, k), bench.case_lengths(case, {}, 2026))
            if case.name.startswith("T5"):
                self.assertEqual(case.scale, 1.0)

    def test_invalid_lengths_and_override(self):
        with self.assertRaises(ValueError):
            bench.make_lengths(4, 10, 10, 0)
        case = bench.Case("custom", 2, (5, 4, 8), (7, 2, 8), (5, 4, 8), 3, 4, True, 1)
        self.assertEqual(
            bench.case_lengths(case, {"custom": {"q": [2, 3], "kv": [4, 3]}}, 0),
            ([2, 3], [4, 3]),
        )
        with self.assertRaises(ValueError):
            bench.case_lengths(case, {"custom": {"q": [2, 2], "kv": [4, 3]}}, 0)

    def test_reference_against_sdpa_with_gqa_and_unequal_lengths(self):
        torch.manual_seed(42)
        # Includes noncausal cross attention, bottom-right causal alignment,
        # fully masked rows, different packing offsets and repeated length groups.
        for causal in (False, True):
            qlens, klens = [2, 5, 2], [4, 3, 4]
            case = bench.Case("tiny", 3, (9, 4, 8), (11, 2, 8), (9, 4, 8), 5, 4, causal, 0.4)
            q, k, v = [torch.randn(shape) for shape in (case.q_shape, case.kv_shape, case.kv_shape)]
            dout = torch.randn_like(q)
            actual = bench.reference_all(q, k, v, dout, qlens, klens, case, 1)
            leaves = [x.detach().requires_grad_(True) for x in (q, k, v)]
            chunks = []
            qi, ki = 0, 0
            for sq, sk in zip(qlens, klens):
                qq = leaves[0][qi : qi + sq].transpose(0, 1).unsqueeze(0)
                kk, vv = [
                    x[ki : ki + sk].repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0)
                    for x in leaves[1:]
                ]
                mask = torch.arange(sk)[None, :] <= torch.arange(sq)[:, None] + sk - sq
                out = F.scaled_dot_product_attention(
                    qq, kk, vv, attn_mask=mask if causal else None, scale=case.scale
                )
                chunks.append(out.squeeze(0).transpose(0, 1))
                qi, ki = qi + sq, ki + sk
            expected_out = torch.cat(chunks)
            expected = (expected_out, *torch.autograd.grad(expected_out, leaves, dout))
            for got, ref in zip(actual, expected):
                torch.testing.assert_close(got, ref, atol=2e-6, rtol=2e-5)

    def test_comparator_rejects_errors_and_nonfinite(self):
        ref = torch.ones(10)
        self.assertTrue(bench.compare(ref, ref, 1e-3, 1e-3)["passed"])
        bad = ref.clone()
        bad[0] = 2
        self.assertFalse(bench.compare(bad, ref, 1e-3, 1e-3)["passed"])
        bad[0] = float("nan")
        self.assertFalse(bench.compare(bad, ref, 1e-3, 1e-3)["passed"])

    def test_dual_reference_accepts_rounding_but_rejects_large_error(self):
        high = torch.tensor([1.0, 0.0])
        low = torch.tensor([1.01, 0.001], dtype=torch.float16)
        rounded = torch.tensor([1.015, 0.005], dtype=torch.float16)
        self.assertTrue(bench.compare_dual(rounded, high, low)["passed"])
        corrupted = torch.tensor([1.5, 0.005], dtype=torch.float16)
        self.assertFalse(bench.compare_dual(corrupted, high, low)["passed"])
        self.assertFalse(bench.compare_dual(rounded, high, low * float("nan"))["passed"])

    def test_dry_run_reports_are_explicitly_unmeasured(self):
        with tempfile.TemporaryDirectory() as directory:
            code = bench.main(["--dry-run", "--case", "T5 Cross", "--output-dir", directory])
            self.assertEqual(code, 0)
            report = json.loads((Path(directory) / "results.json").read_text())
            self.assertEqual(report["results"][0]["status"], "dry_run")
            self.assertIn("N/A", (Path(directory) / "results.md").read_text())
            self.assertNotIn("speedup", report["results"][0])
            self.assertEqual(report["config"]["cann_lengths_mode"], "device-tensor")
            self.assertEqual(report["cann_prefix_source_dtype"], "int32")
            self.assertEqual(report["cann_api"], "torch_npu.npu_fusion_attention_v3")
            self.assertIn("mode=device-tensor", (Path(directory) / "results.md").read_text())

    def test_explicit_legacy_modes_are_recorded_in_reports(self):
        for mode in ("device-to-host", "host-precomputed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                code = bench.main(["--dry-run", "--case", "T5 Decoder", "--output-dir", directory,
                                   "--cann-lengths-mode", mode])
                self.assertEqual(code, 0)
                report = json.loads((Path(directory) / "results.json").read_text())
                self.assertEqual(report["config"]["cann_lengths_mode"], mode)
                self.assertEqual(report["cann_api"], "torch_npu.npu_fusion_attention")
                self.assertIn("mode=" + mode, (Path(directory) / "results.md").read_text())


if __name__ == "__main__":
    unittest.main()
