# Copyright 2026 INT21 AI
# SPDX-License-Identifier: MIT

import argparse
import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import torch

from rmsnorm.benchmarks import benchmark_rmsnorm
from rmsnorm.benchmarks.benchmark_rmsnorm import (
    build_parser,
    comparable_results,
    effective_input_pool_size,
    format_mn_pair,
    format_preset_listing,
    format_preset_rows,
    format_speedup,
    format_summary,
    gate_preset_names,
    input_pool_entry_bytes,
    parse_pair,
    preset_mn_pairs,
    preset_gate_config,
    ratio_failures,
    resolve_mn_pairs,
    run_gate,
    selected_mn_pairs,
    speedup_failures,
    speedup_from_ratio,
    speedup_summary,
    supported_mn_pairs,
    threshold_failures,
    validate_args,
)


class BenchmarkHelperTest(unittest.TestCase):
    def parse_args(self, *args):
        return build_parser().parse_args(list(args))

    def test_import_does_not_load_benchmark_ops(self):
        self.assertIsNone(benchmark_rmsnorm._BENCHMARK_OPS)

    def test_parse_pair(self):
        self.assertEqual(parse_pair("8192,262144"), (8192, 262144))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_pair("8192")

    def test_selected_mn_pairs_filters_default_rows(self):
        self.assertEqual(selected_mn_pairs(n_values=[262144]), [(8192, 262144)])
        self.assertEqual(selected_mn_pairs(m_values=[32768], n_values=[4096]), [(32768, 4096)])

    def test_selected_mn_pairs_filters_explicit_pairs(self):
        pairs = [(1, 2), (3, 4)]
        self.assertEqual(selected_mn_pairs(n_values=[4], pair_values=pairs), [(3, 4)])
        with self.assertRaisesRegex(ValueError, "no benchmark rows matched"):
            selected_mn_pairs(n_values=[8], pair_values=pairs)

    def test_selected_mn_pairs_filters_preset_rows(self):
        self.assertEqual(
            preset_gate_config("large-backward"),
            {
                "preset": "large-backward",
                "mode": "backward",
                "backward": True,
                "residual": False,
                "residual_out_dtype": "same",
                "dtype_names": ("bfloat16", "float16"),
                "mn_pairs": [
                    (32768, 65536),
                    (16384, 131072),
                    (8192, 262144),
                ],
                "min_speedup": 0.08,
            },
        )
        self.assertEqual(
            preset_mn_pairs("large-backward"),
            [(32768, 65536), (16384, 131072), (8192, 262144)],
        )
        self.assertEqual(
            preset_gate_config("mid-backward"),
            {
                "preset": "mid-backward",
                "mode": "backward",
                "backward": True,
                "residual": False,
                "residual_out_dtype": "same",
                "dtype_names": ("bfloat16", "float16"),
                "mn_pairs": [(32768, 16384)],
                "min_speedup": 0.10,
            },
        )
        self.assertEqual(format_mn_pair(32768, 32768), "32768x32768")
        self.assertEqual(
            format_preset_rows(preset_mn_pairs("large-backward")),
            "32768x65536|16384x131072|8192x262144",
        )
        self.assertEqual(
            format_preset_listing(),
            (
                "preset,mode,gate_dtypes,min_speedup,rows,description\n"
                "large-backward,backward,"
                "bfloat16|float16,0.08,"
                "32768x65536|16384x131072|8192x262144,"
                "large backward rows with maintained fp16 and bf16 speedup gates\n"
                "mid-backward,backward,"
                "bfloat16|float16,0.10,"
                "32768x16384,"
                "mid-sized backward row with maintained fp16 and bf16 speedup gates"
            ),
        )
        self.assertEqual(
            selected_mn_pairs(preset="large-backward", n_values=[131072, 262144]),
            [(16384, 131072), (8192, 262144)],
        )
        with self.assertRaisesRegex(ValueError, "--pair cannot be combined"):
            selected_mn_pairs(pair_values=[(1, 2)], preset="large-backward")
        with self.assertRaisesRegex(ValueError, "unknown benchmark preset"):
            selected_mn_pairs(preset="not-a-preset")
        with self.assertRaisesRegex(ValueError, "unknown benchmark preset"):
            preset_gate_config("not-a-preset")
        self.assertEqual(gate_preset_names("large-backward"), ("large-backward",))
        self.assertEqual(gate_preset_names("mid-backward"), ("mid-backward",))
        self.assertEqual(gate_preset_names("all"), ("mid-backward", "large-backward"))
        with self.assertRaisesRegex(ValueError, "unknown benchmark gate"):
            gate_preset_names("not-a-gate")

    def test_supported_mn_pairs_filters_float32_backward_large_rows(self):
        pairs = [(32768, 65536), (8192, 262144)]
        self.assertEqual(
            supported_mn_pairs(pairs, backward=True, dtype=torch.float32),
            [(32768, 65536)],
        )
        self.assertEqual(
            supported_mn_pairs(pairs, backward=True, dtype=torch.float16),
            pairs,
        )
        with self.assertRaisesRegex(ValueError, "no benchmark rows remain"):
            supported_mn_pairs([(8192, 262144)], backward=True, dtype=torch.float32)

    def test_validate_args_rejects_invalid_thresholds_and_modes(self):
        invalid_cases = [
            (("--backward", "--residual"), "--residual is only supported"),
            (("--residual-out-dtype", "float32"), "--residual-out-dtype requires"),
            (("--fail-ratio", "0"), "--fail-ratio must be positive"),
            (("--min-speedup", "-0.1"), "--min-speedup must be nonnegative"),
            (("--input-pool-size", "0"), "--input-pool-size must be positive"),
            (("--pool-memory-fraction", "0"), "--pool-memory-fraction must be"),
            (("--pool-memory-fraction", "1.1"), "--pool-memory-fraction must be"),
            (("--preset", "large-backward"), "--preset large-backward requires"),
            (("--preset", "mid-backward"), "--preset mid-backward requires"),
            (("--gate", "large-backward", "--dtype", "bfloat16"), "--gate cannot be combined"),
            (("--gate", "large-backward", "--preset", "large-backward"), "--gate cannot be combined"),
            (("--gate", "large-backward", "--backward"), "--gate cannot be combined"),
            (("--gate", "large-backward", "--min-speedup", "0.1"), "--gate cannot be combined"),
        ]
        for argv, message in invalid_cases:
            with self.subTest(argv=argv):
                with self.assertRaisesRegex(ValueError, message):
                    validate_args(self.parse_args(*argv))
        validate_args(self.parse_args("--list-presets", "--preset", "large-backward"))
        validate_args(self.parse_args("--gate", "large-backward"))
        validate_args(self.parse_args("--gate", "all"))

    def test_input_pool_size_is_memory_capped(self):
        self.assertEqual(
            input_pool_entry_bytes(2, 16, torch.float16, backward=True, residual=False),
            2 * 16 * 2 + 16 * 4 + 2 * 16 * 2 + 2 * 4,
        )
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            with mock.patch.object(torch.cuda, "mem_get_info", return_value=(1024, 2048)):
                self.assertEqual(
                    effective_input_pool_size(
                        2,
                        16,
                        torch.float16,
                        backward=True,
                        residual=False,
                        residual_out_dtype="same",
                        requested=8,
                        memory_fraction=0.8,
                    ),
                    3,
                )
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(
                effective_input_pool_size(
                    2,
                    16,
                    torch.float16,
                    backward=False,
                    residual=False,
                    residual_out_dtype="same",
                    requested=8,
                    memory_fraction=0.8,
                ),
                8,
            )

    def test_resolve_mn_pairs_applies_mode_support_filters(self):
        args = self.parse_args("--dtype", "float32", "--backward", "--preset", "large-backward")
        validate_args(args)
        self.assertEqual(
            resolve_mn_pairs(args, torch.float32),
            [(32768, 65536), (16384, 131072)],
        )

    def test_ratio_failures_ignores_unsupported_rows(self):
        results = [(32768, 256, 0.95), (8192, 262144, 1.03), (32768, 32768, None)]
        self.assertEqual(
            comparable_results(results),
            [(32768, 256, 0.95), (8192, 262144, 1.03)],
        )
        self.assertEqual(ratio_failures(results, None), [])
        self.assertEqual(ratio_failures(results, 1.0), [(8192, 262144, 1.03)])
        self.assertEqual(ratio_failures(results, 1.05), [])
        self.assertEqual(threshold_failures(results, fail_ratio=1.05), [])
        with self.assertRaisesRegex(ValueError, "ratio threshold exceeded"):
            threshold_failures(results, fail_ratio=1.0)
        with self.assertRaisesRegex(ValueError, "no comparable QuACK rows"):
            threshold_failures([(1, 2, None)], fail_ratio=1.0)

    def test_speedup_failures_ignores_unsupported_rows(self):
        results = [(32768, 512, 0.89), (8192, 262144, 0.92), (32768, 32768, None)]
        self.assertEqual(speedup_failures(results, None), [])
        self.assertEqual(speedup_failures(results, 0.10), [(8192, 262144, 0.92)])
        self.assertEqual(speedup_failures(results, 0.08), [])
        self.assertEqual(threshold_failures(results, min_speedup=0.08), [])
        with self.assertRaisesRegex(ValueError, "speedup threshold not met"):
            threshold_failures(results, min_speedup=0.10)
        self.assertAlmostEqual(speedup_from_ratio(0.8), 0.25)
        self.assertEqual(format_speedup(0.8), "0.2500")
        self.assertEqual(format_speedup(None), "nan")

    def test_speedup_summary_uses_comparable_rows(self):
        results = [(1, 2, 0.8), (3, 4, 0.9), (5, 6, None)]
        summary = speedup_summary(results)
        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["min_speedup"], 1.0 / 0.9 - 1.0)
        self.assertAlmostEqual(summary["max_speedup"], 0.25)
        self.assertAlmostEqual(
            summary["geomean_speedup"],
            1.0 / ((0.8 * 0.9) ** 0.5) - 1.0,
        )
        self.assertEqual(format_summary(summary), "summary,2,0.1111,0.1785,0.2500")
        self.assertIsNone(speedup_summary([(1, 2, None)]))
        self.assertEqual(format_summary(None), "summary,0,nan,nan,nan")

    def test_run_gate_uses_preset_contract(self):
        args = self.parse_args("--gate", "all", "--summary")

        def fake_bench_pair(
            M,
            N,
            dtype,
            backward,
            residual,
            residual_out_dtype,
            warmup,
            rep,
            input_pool_size,
            pool_memory_fraction,
            provider_order,
        ):
            self.assertTrue(backward)
            self.assertFalse(residual)
            self.assertEqual(residual_out_dtype, "same")
            self.assertEqual(input_pool_size, 64)
            self.assertEqual(pool_memory_fraction, 0.8)
            self.assertEqual(provider_order, "balanced")
            return 1.0, 1.25, None, input_pool_size

        output = io.StringIO()
        with mock.patch.object(benchmark_rmsnorm, "bench_pair", side_effect=fake_bench_pair):
            with mock.patch.object(torch.cuda, "empty_cache"):
                with redirect_stdout(output):
                    run_gate(args)

        text = output.getvalue()
        self.assertIn("preset,dtype,M,N,pool_size,rmsnorm_ms,quack_ms,ratio,speedup", text)
        self.assertIn("large-backward,bfloat16,32768,65536,64,1.000000,1.250000,0.8000,0.2500", text)
        self.assertIn("large-backward,float16,8192,262144,64,1.000000,1.250000,0.8000,0.2500", text)
        self.assertIn("mid-backward,bfloat16,32768,16384,64,1.000000,1.250000,0.8000,0.2500", text)
        self.assertIn("mid-backward,float16,32768,16384,64,1.000000,1.250000,0.8000,0.2500", text)
        self.assertIn("summary,large-backward,bfloat16,3,0.2500,0.2500,0.2500", text)
        self.assertIn("summary,large-backward,float16,3,0.2500,0.2500,0.2500", text)
        self.assertIn("summary,mid-backward,bfloat16,1,0.2500,0.2500,0.2500", text)
        self.assertIn("summary,mid-backward,float16,1,0.2500,0.2500,0.2500", text)

    def test_forward_bench_pair_warms_both_providers_before_timing(self):
        events = []
        pool = [{"x": "x0", "w": "w0"}, {"x": "x1", "w": "w1"}]

        def ours_fwd(*args, **kwargs):
            events.append("ours_fwd")

        def quack_fwd(*args, **kwargs):
            events.append("quack_fwd")

        def fake_do_bench(fn, warmup, rep):
            events.append("bench")
            fn()
            return 1.0 if events.count("bench") == 1 else 2.0

        with mock.patch.object(benchmark_rmsnorm, "make_input_pool", return_value=pool):
            with mock.patch.object(
                benchmark_rmsnorm,
                "benchmark_ops",
                return_value=(ours_fwd, object(), quack_fwd, object(), fake_do_bench),
            ):
                with mock.patch.object(torch, "manual_seed"):
                    with mock.patch.object(torch.cuda, "synchronize", side_effect=lambda: events.append("sync")):
                        ours, quack, quack_error, pool_size = benchmark_rmsnorm.bench_pair(
                            2,
                            16,
                            torch.float16,
                            backward=False,
                            residual=False,
                            residual_out_dtype="same",
                            warmup=10,
                            rep=100,
                            provider_order="rmsnorm-first",
                        )

        self.assertEqual((ours, quack, quack_error, pool_size), (1.0, 2.0, None, 2))
        self.assertEqual(events[:3], ["ours_fwd", "quack_fwd", "sync"])
        self.assertEqual(events[3], "bench")

    def test_balanced_provider_order_averages_both_orders(self):
        events = []
        pool = [{"x": "x0", "w": "w0"}]
        bench_times = iter([1.0, 3.0, 5.0, 7.0])

        def ours_fwd(*args, **kwargs):
            events.append("ours_fwd")

        def quack_fwd(*args, **kwargs):
            events.append("quack_fwd")

        def fake_do_bench(fn, warmup, rep):
            events.append("bench")
            fn()
            return next(bench_times)

        with mock.patch.object(benchmark_rmsnorm, "make_input_pool", return_value=pool):
            with mock.patch.object(
                benchmark_rmsnorm,
                "benchmark_ops",
                return_value=(ours_fwd, object(), quack_fwd, object(), fake_do_bench),
            ):
                with mock.patch.object(torch, "manual_seed"):
                    with mock.patch.object(torch.cuda, "is_available", return_value=False):
                        with mock.patch.object(torch.cuda, "synchronize"):
                            ours, quack, quack_error, pool_size = benchmark_rmsnorm.bench_pair(
                                2,
                                16,
                                torch.float16,
                                backward=False,
                                residual=False,
                                residual_out_dtype="same",
                                warmup=10,
                                rep=100,
                                provider_order="balanced",
                            )

        self.assertEqual((ours, quack, quack_error, pool_size), (4.0, 4.0, None, 1))
        bench_events = [event for event in events if event == "bench"]
        self.assertEqual(len(bench_events), 4)


if __name__ == "__main__":
    unittest.main()
