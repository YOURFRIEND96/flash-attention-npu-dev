"""CPU-only checks: python -m unittest discover -s scripts -p 'test_benchmark*.py'."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F

import benchmark_varlen_vs_cann as bench


class BenchmarkTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
