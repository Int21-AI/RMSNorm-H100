# Copyright 2026 INT21 AI
# SPDX-License-Identifier: MIT

import argparse
import gc
import math
import os
import sys

import torch

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.dirname(PACKAGE_ROOT)
QUACK_ROOT = os.environ.get("QUACK_ROOT", os.path.join(WORKSPACE_ROOT, "quack"))
if os.path.isdir(QUACK_ROOT) and QUACK_ROOT not in sys.path:
    sys.path.append(QUACK_ROOT)


MN_PAIRS = [
    (32768, 256),
    (32768, 512),
    (32768, 1024),
    (32768, 2048),
    (32768, 4096),
    (32768, 8192),
    (32768, 16384),
    (32768, 32768),
    (32768, 65536),
    (16384, 131072),
    (8192, 262144),
]

MN_PRESETS = {
    "mid-backward": [
        (32768, 16384),
    ],
    "large-backward": [
        (32768, 65536),
        (16384, 131072),
        (8192, 262144),
    ],
}
MN_PRESET_MODES = {
    "large-backward": "backward",
    "mid-backward": "backward",
}
MN_PRESET_DESCRIPTIONS = {
    "large-backward": "large backward rows with maintained fp16 and bf16 speedup gates",
    "mid-backward": "mid-sized backward row with maintained fp16 and bf16 speedup gates",
}
MN_PRESET_GATE_DTYPES = {
    "large-backward": ("bfloat16", "float16"),
    "mid-backward": ("bfloat16", "float16"),
}
MN_PRESET_MIN_SPEEDUPS = {
    "large-backward": 0.08,
    "mid-backward": 0.10,
}
GATE_ALL = "all"
GATE_CHOICES = tuple(sorted(MN_PRESETS)) + (GATE_ALL,)
DEFAULT_INPUT_POOL_SIZE = 64
DEFAULT_POOL_MEMORY_FRACTION = 0.80
DEFAULT_PROVIDER_ORDER = "balanced"
PROVIDER_ORDERS = ("rmsnorm-first", "quack-first", "balanced")

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


_BENCHMARK_OPS = None


def benchmark_ops():
    global _BENCHMARK_OPS
    if _BENCHMARK_OPS is None:
        from quack.rmsnorm import rmsnorm_bwd as quack_bwd
        from quack.rmsnorm import rmsnorm_fwd as quack_fwd
        from rmsnorm import rmsnorm_bwd, rmsnorm_fwd
        from triton.testing import do_bench

        _BENCHMARK_OPS = rmsnorm_fwd, rmsnorm_bwd, quack_fwd, quack_bwd, do_bench
    return _BENCHMARK_OPS


def parse_pair(value):
    try:
        m_text, n_text = value.split(",", 1)
        return int(m_text), int(n_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--pair must be formatted as M,N") from exc


def preset_mn_pairs(preset):
    if preset is None:
        return MN_PAIRS
    try:
        return MN_PRESETS[preset]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark preset: {preset}") from exc


def format_mn_pair(M, N):
    return f"{M}x{N}"


def format_preset_rows(pairs):
    return "|".join(format_mn_pair(M, N) for M, N in pairs)


def preset_listing_rows():
    return [
        (
            name,
            MN_PRESET_MODES[name],
            "|".join(MN_PRESET_GATE_DTYPES[name]),
            f"{MN_PRESET_MIN_SPEEDUPS[name]:.2f}",
            format_preset_rows(MN_PRESETS[name]),
            MN_PRESET_DESCRIPTIONS[name],
        )
        for name in sorted(MN_PRESETS)
    ]


def format_preset_listing():
    lines = ["preset,mode,gate_dtypes,min_speedup,rows,description"]
    lines.extend(",".join(row) for row in preset_listing_rows())
    return "\n".join(lines)


def preset_gate_config(preset):
    if preset not in MN_PRESETS:
        raise ValueError(f"unknown benchmark preset: {preset}")
    mode = MN_PRESET_MODES[preset]
    return {
        "preset": preset,
        "mode": mode,
        "backward": mode == "backward",
        "residual": False,
        "residual_out_dtype": "same",
        "dtype_names": MN_PRESET_GATE_DTYPES[preset],
        "mn_pairs": MN_PRESETS[preset],
        "min_speedup": MN_PRESET_MIN_SPEEDUPS[preset],
    }


def gate_preset_names(gate):
    if gate == GATE_ALL:
        return tuple(MN_PRESETS)
    if gate not in MN_PRESETS:
        raise ValueError(f"unknown benchmark gate: {gate}")
    return (gate,)


def selected_mn_pairs(m_values=None, n_values=None, pair_values=None, preset=None):
    if pair_values is not None and preset is not None:
        raise ValueError("--pair cannot be combined with --preset")
    pairs = pair_values or preset_mn_pairs(preset)
    m_filter = set(m_values or [])
    n_filter = set(n_values or [])
    selected = [
        (M, N)
        for M, N in pairs
        if (not m_filter or M in m_filter) and (not n_filter or N in n_filter)
    ]
    if not selected:
        raise ValueError("no benchmark rows matched the requested filters")
    return selected


def supported_mn_pairs(mn_pairs, backward, dtype):
    selected = [
        (M, N)
        for M, N in mn_pairs
        if not (backward and dtype == torch.float32 and N > 128 * 1024)
    ]
    if not selected:
        raise ValueError("no benchmark rows remain after applying supported-mode filters")
    return selected


def ratio_failures(results, threshold):
    if threshold is None:
        return []
    return [
        (M, N, ratio)
        for M, N, ratio in results
        if ratio is not None and ratio > threshold
    ]


def comparable_results(results):
    return [(M, N, ratio) for M, N, ratio in results if ratio is not None]


def speedup_failures(results, min_speedup):
    if min_speedup is None:
        return []
    max_ratio = 1.0 / (1.0 + min_speedup)
    return [
        (M, N, ratio)
        for M, N, ratio in results
        if ratio is not None and ratio > max_ratio
    ]


def speedup_from_ratio(ratio):
    return (1.0 / ratio) - 1.0


def format_speedup(ratio):
    return "nan" if ratio is None else f"{speedup_from_ratio(ratio):.4f}"


def speedup_summary(results):
    comparable = comparable_results(results)
    if not comparable:
        return None
    ratios = [ratio for _, _, ratio in comparable]
    speedups = [speedup_from_ratio(ratio) for ratio in ratios]
    geomean_ratio = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
    return {
        "count": len(ratios),
        "min_speedup": min(speedups),
        "geomean_speedup": speedup_from_ratio(geomean_ratio),
        "max_speedup": max(speedups),
    }


def format_summary(summary):
    if summary is None:
        return "summary,0,nan,nan,nan"
    return (
        f"summary,{summary['count']},"
        f"{summary['min_speedup']:.4f},"
        f"{summary['geomean_speedup']:.4f},"
        f"{summary['max_speedup']:.4f}"
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=DTYPES, default=None)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--residual-out-dtype", choices=("same", "float32"), default="same")
    parser.add_argument("--m", type=int, action="append", help="Only run rows with this M")
    parser.add_argument("--n", type=int, action="append", help="Only run rows with this N")
    parser.add_argument(
        "--pair",
        type=parse_pair,
        action="append",
        help="Run an explicit M,N row; may be repeated and can be combined with --m/--n",
    )
    parser.add_argument(
        "--preset",
        choices=MN_PRESETS,
        help="Use a maintained row preset; can be combined with --m/--n filters",
    )
    parser.add_argument(
        "--gate",
        choices=GATE_CHOICES,
        help=(
            "Run a maintained benchmark gate across its listed gate_dtypes and "
            "min_speedup target; use 'all' to run every maintained gate"
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "--input-pool-size",
        type=int,
        default=DEFAULT_INPUT_POOL_SIZE,
        help=(
            "Target number of random input sets to rotate through during timing; "
            "large rows may use fewer based on --pool-memory-fraction"
        ),
    )
    parser.add_argument(
        "--pool-memory-fraction",
        type=float,
        default=DEFAULT_POOL_MEMORY_FRACTION,
        help="Maximum fraction of currently free CUDA memory to use for pooled inputs",
    )
    parser.add_argument(
        "--provider-order",
        choices=PROVIDER_ORDERS,
        default=DEFAULT_PROVIDER_ORDER,
        help=(
            "Provider timing order. 'balanced' runs fresh pooled timings in "
            "both orders and averages each provider's time."
        ),
    )
    parser.add_argument(
        "--fail-ratio",
        type=float,
        default=None,
        help="Exit nonzero if any supported row has rmsnorm_ms / quack_ms above this value",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=None,
        help=(
            "Exit nonzero if any supported row has less than this fractional "
            "speedup over QuACK, measured as quack_ms / rmsnorm_ms - 1"
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print aggregate speedup summary rows after the per-row benchmark CSV",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List maintained benchmark presets and exit without running benchmarks",
    )
    return parser


def validate_args(args):
    if args.list_presets:
        return
    if args.input_pool_size < 1:
        raise ValueError("--input-pool-size must be positive")
    if args.pool_memory_fraction <= 0 or args.pool_memory_fraction > 1:
        raise ValueError("--pool-memory-fraction must be in (0, 1]")
    if args.gate:
        forbidden = []
        if args.preset is not None:
            forbidden.append("--preset")
        if args.pair is not None:
            forbidden.append("--pair")
        if args.m is not None:
            forbidden.append("--m")
        if args.n is not None:
            forbidden.append("--n")
        if args.backward:
            forbidden.append("--backward")
        if args.residual:
            forbidden.append("--residual")
        if args.residual_out_dtype != "same":
            forbidden.append("--residual-out-dtype")
        if args.fail_ratio is not None:
            forbidden.append("--fail-ratio")
        if args.min_speedup is not None:
            forbidden.append("--min-speedup")
        if args.dtype is not None:
            forbidden.append("--dtype")
        if forbidden:
            raise ValueError(f"--gate cannot be combined with {', '.join(forbidden)}")
        return
    if args.backward and args.residual:
        raise ValueError("--residual is only supported for forward benchmarks")
    if not args.residual and args.residual_out_dtype != "same":
        raise ValueError("--residual-out-dtype requires --residual")
    if args.fail_ratio is not None and args.fail_ratio <= 0:
        raise ValueError("--fail-ratio must be positive")
    if args.min_speedup is not None and args.min_speedup < 0:
        raise ValueError("--min-speedup must be nonnegative")
    if (
        args.preset is not None
        and MN_PRESET_MODES[args.preset] == "backward"
        and not args.backward
    ):
        raise ValueError(f"--preset {args.preset} requires --backward")


def resolve_mn_pairs(args, dtype):
    mn_pairs = selected_mn_pairs(args.m, args.n, args.pair, args.preset)
    return supported_mn_pairs(mn_pairs, args.backward, dtype)


def dtype_nbytes(dtype):
    return torch.empty((), dtype=dtype).element_size()


def input_pool_entry_bytes(M, N, dtype, backward, residual):
    elements = M * N
    nbytes = dtype_nbytes(dtype)
    total = elements * nbytes
    total += N * torch.empty((), dtype=torch.float32).element_size()
    if backward:
        total += elements * nbytes
        total += M * torch.empty((), dtype=torch.float32).element_size()
    elif residual:
        total += elements * nbytes
    return total


def transient_output_bytes(M, N, dtype, backward, residual, residual_out_dtype):
    elements = M * N
    nbytes = dtype_nbytes(dtype)
    total = elements * nbytes
    if backward:
        total += N * torch.empty((), dtype=torch.float32).element_size()
    elif residual:
        residual_nbytes = (
            torch.empty((), dtype=torch.float32).element_size()
            if residual_out_dtype == "float32"
            else nbytes
        )
        total += elements * residual_nbytes
    return total


def effective_input_pool_size(
    M,
    N,
    dtype,
    backward,
    residual,
    residual_out_dtype,
    requested,
    memory_fraction,
):
    if requested < 1:
        raise ValueError("--input-pool-size must be positive")
    if memory_fraction <= 0 or memory_fraction > 1:
        raise ValueError("--pool-memory-fraction must be in (0, 1]")
    if not torch.cuda.is_available():
        return requested
    free_bytes, _ = torch.cuda.mem_get_info()
    entry_bytes = input_pool_entry_bytes(M, N, dtype, backward, residual)
    output_bytes = transient_output_bytes(M, N, dtype, backward, residual, residual_out_dtype)
    budget = int(free_bytes * memory_fraction) - output_bytes
    if entry_bytes <= 0 or budget <= 0:
        return 1
    return max(1, min(requested, budget // entry_bytes))


def make_input_pool(
    M,
    N,
    dtype,
    backward,
    residual,
    residual_out_dtype,
    requested_pool_size,
    pool_memory_fraction,
):
    pool_size = effective_input_pool_size(
        M,
        N,
        dtype,
        backward,
        residual,
        residual_out_dtype,
        requested_pool_size,
        pool_memory_fraction,
    )
    pool = []
    for _ in range(pool_size):
        x = torch.randn(M, N, device="cuda", dtype=dtype)
        entry = {
            "x": x,
            "w": torch.randn(N, device="cuda", dtype=torch.float32),
        }
        if backward:
            entry["dout"] = torch.randn_like(x)
            entry["rstd"] = torch.rsqrt(x.float().square().mean(dim=-1) + 1e-6)
        elif residual:
            entry["residual"] = torch.randn_like(x)
        pool.append(entry)
    return pool


def rotating_pool_call(pool, fn):
    index = 0

    def call():
        nonlocal index
        entry = pool[index]
        index = (index + 1) % len(pool)
        return fn(entry)

    return call


def mean(values):
    return sum(values) / len(values)


def threshold_failures(results, fail_ratio=None, min_speedup=None):
    if (fail_ratio is not None or min_speedup is not None) and not comparable_results(results):
        raise ValueError("benchmark threshold requested but no comparable QuACK rows ran")
    failures = ratio_failures(results, fail_ratio)
    if failures:
        details = ", ".join(f"{M}x{N}={ratio:.4f}" for M, N, ratio in failures)
        raise ValueError(f"benchmark ratio threshold exceeded: {details}")
    failures = speedup_failures(results, min_speedup)
    if failures:
        details = ", ".join(
            f"{M}x{N}={speedup_from_ratio(ratio):.2%}" for M, N, ratio in failures
        )
        raise ValueError(f"benchmark speedup threshold not met: {details}")
    return []


def bench_pair(
    M,
    N,
    dtype,
    backward,
    residual,
    residual_out_dtype,
    warmup,
    rep,
    input_pool_size=DEFAULT_INPUT_POOL_SIZE,
    pool_memory_fraction=DEFAULT_POOL_MEMORY_FRACTION,
    provider_order=DEFAULT_PROVIDER_ORDER,
):
    if provider_order == "balanced":
        timings = []
        for order in ("rmsnorm-first", "quack-first"):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            timings.append(
                bench_pair(
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
                    order,
                )
            )
        quack_errors = [row[2] for row in timings]
        if any(error is not None for error in quack_errors):
            return timings[0]
        return (
            mean([row[0] for row in timings]),
            mean([row[1] for row in timings]),
            None,
            min(row[3] for row in timings),
        )

    rmsnorm_fwd, rmsnorm_bwd, quack_fwd, quack_bwd, do_bench = benchmark_ops()
    torch.manual_seed(0)
    pool = make_input_pool(
        M,
        N,
        dtype,
        backward,
        residual,
        residual_out_dtype,
        input_pool_size,
        pool_memory_fraction,
    )
    pool_size = len(pool)
    if backward:
        if residual:
            raise ValueError("--residual is only supported for forward benchmarks")
        first = pool[0]
        rmsnorm_bwd(first["x"], first["w"], first["dout"], first["rstd"])
        quack_bwd(first["x"], first["w"], first["dout"], first["rstd"])
        torch.cuda.synchronize()
        ours_call = rotating_pool_call(
            pool,
            lambda entry: rmsnorm_bwd(
                entry["x"], entry["w"], entry["dout"], entry["rstd"]
            ),
        )
        quack_call = rotating_pool_call(
            pool,
            lambda entry: quack_bwd(
                entry["x"], entry["w"], entry["dout"], entry["rstd"]
            ),
        )
        if provider_order == "quack-first":
            quack = do_bench(quack_call, warmup=warmup, rep=rep)
            ours = do_bench(ours_call, warmup=warmup, rep=rep)
        else:
            ours = do_bench(ours_call, warmup=warmup, rep=rep)
            quack = do_bench(quack_call, warmup=warmup, rep=rep)
        quack_error = None
    else:
        residual_dtype = torch.float32 if residual_out_dtype == "float32" else None
        first = pool[0]
        quack_unsupported = residual and dtype == torch.float32 and N >= 32768
        quack_error = None
        rmsnorm_fwd(
            first["x"],
            first["w"],
            residual=first.get("residual"),
            residual_dtype=residual_dtype,
        )
        if not quack_unsupported:
            try:
                quack_fwd(
                    first["x"],
                    first["w"],
                    residual=first.get("residual"),
                    residual_dtype=residual_dtype,
                )
            except RuntimeError as exc:
                message = str(exc).splitlines()[0]
                quack_error = f"{exc.__class__.__name__}: {message}"
        torch.cuda.synchronize()
        ours_call = rotating_pool_call(
            pool,
            lambda entry: rmsnorm_fwd(
                entry["x"],
                entry["w"],
                residual=entry.get("residual"),
                residual_dtype=residual_dtype,
            ),
        )
        if quack_unsupported:
            ours = do_bench(ours_call, warmup=warmup, rep=rep)
            quack = None
            quack_error = "quack_unsupported"
        elif quack_error is not None:
            ours = do_bench(ours_call, warmup=warmup, rep=rep)
            quack = None
        else:
            quack_call = rotating_pool_call(
                pool,
                lambda entry: quack_fwd(
                    entry["x"],
                    entry["w"],
                    residual=entry.get("residual"),
                    residual_dtype=residual_dtype,
                ),
            )
            if provider_order == "quack-first":
                quack = do_bench(quack_call, warmup=warmup, rep=rep)
                ours = do_bench(ours_call, warmup=warmup, rep=rep)
            else:
                ours = do_bench(ours_call, warmup=warmup, rep=rep)
                quack = do_bench(quack_call, warmup=warmup, rep=rep)
    return ours, quack, quack_error, pool_size


def print_benchmark_row(
    M, N, pool_size, ours, quack, quack_error, results, dtype_name=None, preset_name=None
):
    prefix_parts = []
    if preset_name is not None:
        prefix_parts.append(preset_name)
    if dtype_name is not None:
        prefix_parts.append(dtype_name)
    prefix = ",".join(prefix_parts)
    if prefix:
        prefix = f"{prefix},"
    if quack_error is not None:
        results.append((M, N, None))
        print(f"{prefix}{M},{N},{pool_size},{ours:.6f},{quack_error},nan,nan", flush=True)
        return
    ratio = ours / quack
    results.append((M, N, ratio))
    print(
        f"{prefix}{M},{N},{pool_size},{ours:.6f},{quack:.6f},{ratio:.4f},{format_speedup(ratio)}",
        flush=True,
    )


def run_rows(mn_pairs, dtype, args, dtype_name=None, preset_name=None):
    results = []
    for M, N in mn_pairs:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ours, quack, quack_error, pool_size = bench_pair(
            M,
            N,
            dtype,
            args.backward,
            args.residual,
            args.residual_out_dtype,
            args.warmup,
            args.rep,
            args.input_pool_size,
            args.pool_memory_fraction,
            args.provider_order,
        )
        print_benchmark_row(
            M,
            N,
            pool_size,
            ours,
            quack,
            quack_error,
            results,
            dtype_name=dtype_name,
            preset_name=preset_name,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def run_gate(args):
    print("preset,dtype,M,N,pool_size,rmsnorm_ms,quack_ms,ratio,speedup")
    summaries = []
    threshold_errors = []
    for preset_name in gate_preset_names(args.gate):
        config = preset_gate_config(preset_name)
        for dtype_name in config["dtype_names"]:
            dtype = DTYPES[dtype_name]
            gate_args = argparse.Namespace(
                backward=config["backward"],
                residual=config["residual"],
                residual_out_dtype=config["residual_out_dtype"],
                warmup=args.warmup,
                rep=args.rep,
                input_pool_size=args.input_pool_size,
                pool_memory_fraction=args.pool_memory_fraction,
                provider_order=args.provider_order,
            )
            results = run_rows(
                config["mn_pairs"],
                dtype,
                gate_args,
                dtype_name=dtype_name,
                preset_name=preset_name,
            )
            summaries.append((preset_name, dtype_name, speedup_summary(results)))
            try:
                threshold_failures(results, min_speedup=config["min_speedup"])
            except ValueError as exc:
                threshold_errors.append(f"{preset_name}/{dtype_name}: {exc}")
    if args.summary:
        print("summary,preset,dtype,count,min_speedup,geomean_speedup,max_speedup")
        for preset_name, dtype_name, summary in summaries:
            if summary is None:
                print(f"summary,{preset_name},{dtype_name},0,nan,nan,nan")
            else:
                print(
                    f"summary,{preset_name},{dtype_name},{summary['count']},"
                    f"{summary['min_speedup']:.4f},"
                    f"{summary['geomean_speedup']:.4f},"
                    f"{summary['max_speedup']:.4f}"
                )
    if threshold_errors:
        raise ValueError("; ".join(threshold_errors))


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.list_presets:
        print(format_preset_listing())
        return
    try:
        validate_args(args)
        if args.gate:
            run_gate(args)
            return
        dtype = DTYPES[args.dtype or "bfloat16"]
        mn_pairs = resolve_mn_pairs(args, dtype)
    except ValueError as exc:
        parser.error(str(exc))

    print("M,N,pool_size,rmsnorm_ms,quack_ms,ratio,speedup")
    results = run_rows(mn_pairs, dtype, args)
    if args.summary:
        print("summary,count,min_speedup,geomean_speedup,max_speedup")
        print(format_summary(speedup_summary(results)))
    try:
        threshold_failures(results, args.fail_ratio, args.min_speedup)
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
