# Copyright 2026 INT21 AI
# SPDX-License-Identifier: MIT

import argparse
from contextlib import redirect_stdout
import gc
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.dirname(PACKAGE_ROOT)
QUACK_ROOT = os.environ.get("QUACK_ROOT", os.path.join(WORKSPACE_ROOT, "quack"))
if os.path.isdir(QUACK_ROOT) and QUACK_ROOT not in sys.path:
    sys.path.insert(0, QUACK_ROOT)


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
    "large-backward": "large backward rows with maintained fp16 and bf16 1% ratio gates",
    "mid-backward": "mid-sized backward row with maintained fp16 and bf16 1% ratio gates",
}
MN_PRESET_GATE_DTYPES = {
    "large-backward": ("bfloat16", "float16"),
    "mid-backward": ("bfloat16", "float16"),
}
MN_PRESET_FAIL_RATIOS = {
    "large-backward": 1.01,
    "mid-backward": 1.01,
}
GATE_ALL = "all"
GATE_CHOICES = tuple(sorted(MN_PRESET_GATE_DTYPES)) + (GATE_ALL,)
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
_QUACK_MODULE = None
_QUACK_FWD_TUNER = None
_QUACK_BWD_TUNER = None


def configure_quack_root(root):
    global QUACK_ROOT
    if root is None:
        return
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise ValueError(f"QuACK root does not exist: {root}")
    imported = sys.modules.get("quack")
    if imported is not None:
        imported_file = getattr(imported, "__file__", None)
        if imported_file is not None and not os.path.abspath(imported_file).startswith(root + os.sep):
            raise RuntimeError(
                f"QuACK was already imported from {imported_file}; cannot switch to {root}"
            )
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    QUACK_ROOT = root


def configure_quack_autotune(mode):
    if mode == "fresh":
        os.environ["QUACK_FORCE_CACHE_UPDATE"] = "1"
    elif mode == "cached":
        os.environ.pop("QUACK_FORCE_CACHE_UPDATE", None)
    else:
        raise ValueError(f"unknown QuACK autotune mode: {mode}")


def _require_quack_autotuner(name, tuner):
    configs = getattr(tuner, "configs", None)
    cache = getattr(tuner, "cache", None)
    if configs is None or cache is None or len(configs) <= 1:
        raise ImportError(
            f"QuACK {name} is not an active multi-config autotuner. "
            "Use a QuACK checkout that exports the exhaustive RMSNorm tuned APIs."
        )


def selected_quack_config(backward):
    tuner = _QUACK_BWD_TUNER if backward else _QUACK_FWD_TUNER
    if tuner is None:
        raise RuntimeError("QuACK benchmark operations have not been initialized")
    config = getattr(tuner, "best_config", None)
    if config is None:
        raise RuntimeError("QuACK autotuner did not expose a selected best_config")
    return " ".join(str(config).replace(",", ";").split())


def _git_state(path):
    try:
        revision = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(path), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def quack_provenance(autotune_mode):
    benchmark_ops()
    module_file = Path(_QUACK_MODULE.__file__).resolve()
    checkout_root = module_file.parents[1]
    revision, dirty = _git_state(checkout_root)
    package = importlib.import_module("quack")
    return {
        "autotune_mode": autotune_mode,
        "quack_version": getattr(package, "__version__", None),
        "quack_module": str(module_file),
        "quack_checkout": str(checkout_root),
        "quack_revision": revision,
        "quack_dirty": dirty,
        "quack_fwd_config_count": len(_QUACK_FWD_TUNER.configs),
        "quack_bwd_config_count": len(_QUACK_BWD_TUNER.configs),
    }


def emit_quack_provenance(args):
    provenance = quack_provenance(args.quack_autotune)
    print(
        "quack_provenance=" + json.dumps(provenance, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )
    if args.quack_provenance_json is not None:
        path = Path(args.quack_provenance_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


def benchmark_ops():
    global _BENCHMARK_OPS, _QUACK_MODULE, _QUACK_FWD_TUNER, _QUACK_BWD_TUNER
    if _BENCHMARK_OPS is None:
        quack_rmsnorm = importlib.import_module("quack.rmsnorm")
        quack_get_sm_count = getattr(
            quack_rmsnorm, "get_sm_count", getattr(quack_rmsnorm, "_get_sm_count", None)
        )
        quack_fwd_tuned = getattr(quack_rmsnorm, "rmsnorm_fwd_tuned", None)
        quack_bwd_tuned = getattr(quack_rmsnorm, "rmsnorm_bwd_tuned", None)
        missing = [
            name
            for name, value in (
                ("get_sm_count/_get_sm_count", quack_get_sm_count),
                ("rmsnorm_fwd_tuned", quack_fwd_tuned),
                ("rmsnorm_bwd_tuned", quack_bwd_tuned),
            )
            if value is None
        ]
        if missing:
            raise ImportError(
                "QuACK RMSNorm autotuned benchmark APIs are missing: "
                f"{', '.join(missing)}. Set QUACK_ROOT to a QuACK checkout "
                "that exports autotuned RMSNorm entry points."
            )
        _require_quack_autotuner("rmsnorm_fwd_tuned", quack_fwd_tuned)
        _require_quack_autotuner("rmsnorm_bwd_tuned", quack_bwd_tuned)
        from rmsnorm import rmsnorm_bwd, rmsnorm_fwd
        from triton.testing import do_bench

        def quack_fwd(
            x,
            weight=None,
            bias=None,
            residual=None,
            out_dtype=None,
            residual_dtype=None,
            eps=1e-6,
            store_rstd=False,
        ):
            out_dtype = x.dtype if out_dtype is None else out_dtype
            out = torch.empty_like(x, dtype=out_dtype)
            rstd = (
                torch.empty(*x.shape[:-1], device=x.device, dtype=torch.float32)
                if store_rstd
                else None
            )
            if residual is not None and residual_dtype is None:
                residual_dtype = residual.dtype
            if residual is not None or (
                residual_dtype is not None and residual_dtype != x.dtype
            ):
                residual_out = torch.empty_like(
                    x,
                    dtype=residual_dtype if residual_dtype is not None else x.dtype,
                )
            else:
                residual_out = None
            per_head = (
                (weight is not None and weight.dim() == 2)
                or (bias is not None and bias.dim() == 2)
            )
            quack_fwd_tuned(
                x,
                weight,
                out,
                bias,
                rstd,
                None,
                residual,
                residual_out,
                eps,
                False,
                per_head,
            )
            if residual_out is None:
                residual_out = x
            return out, residual_out, rstd

        def quack_bwd(
            x,
            weight,
            dout,
            rstd,
            dresidual_out=None,
            has_bias=False,
            has_residual=False,
        ):
            device = x.device
            N = x.size(-1)
            per_head = x.dim() == 3
            dx = torch.empty_like(x)
            if dresidual_out is not None and dresidual_out.dtype != dx.dtype:
                dresidual = torch.empty_like(x, dtype=dresidual_out.dtype)
            else:
                dresidual = None
            sm_count = quack_get_sm_count(N, device)
            if per_head:
                H = x.size(1)
                sm_count = max(round(sm_count / H), 1)
            else:
                H = None
            if weight is not None:
                dw_shape = (sm_count, H, N) if per_head else (sm_count, N)
                dw_partial = torch.empty(dw_shape, device=device, dtype=torch.float32)
            else:
                dw_partial = None
            db_shape = (sm_count, H, N) if per_head else (sm_count, N)
            db_partial = (
                torch.empty(db_shape, device=device, dtype=torch.float32)
                if has_bias
                else None
            )

            if x.numel() > 0:
                quack_bwd_tuned(
                    x,
                    weight,
                    dout,
                    rstd,
                    dx,
                    dw_partial,
                    db_partial,
                    dresidual_out,
                    dresidual,
                    sm_count,
                    per_head,
                    dw_partial is not None,
                    db_partial is not None,
                )
                dw = dw_partial.sum(dim=0).to(weight.dtype) if weight is not None else None
                db = db_partial.sum(dim=0).to(weight.dtype) if has_bias else None
            else:
                dw = torch.zeros_like(weight) if weight is not None else None
                db = torch.zeros_like(weight) if has_bias else None
            if has_residual and dresidual is None:
                dresidual = dx
            return dx, dw, db, dresidual

        _QUACK_MODULE = quack_rmsnorm
        _QUACK_FWD_TUNER = quack_fwd_tuned
        _QUACK_BWD_TUNER = quack_bwd_tuned
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
            "|".join(MN_PRESET_GATE_DTYPES.get(name, ())) or "none",
            f"{MN_PRESET_FAIL_RATIOS[name]:.2f}"
            if name in MN_PRESET_FAIL_RATIOS
            else "none",
            format_preset_rows(MN_PRESETS[name]),
            MN_PRESET_DESCRIPTIONS[name],
        )
        for name in sorted(MN_PRESETS)
    ]


def format_preset_listing():
    lines = ["preset,mode,gate_dtypes,fail_ratio,rows,description"]
    lines.extend(",".join(row) for row in preset_listing_rows())
    return "\n".join(lines)


def preset_gate_config(preset):
    if preset not in MN_PRESETS:
        raise ValueError(f"unknown benchmark preset: {preset}")
    if preset not in MN_PRESET_GATE_DTYPES:
        raise ValueError(f"preset is not a benchmark gate: {preset}")
    mode = MN_PRESET_MODES[preset]
    return {
        "preset": preset,
        "mode": mode,
        "backward": mode == "backward",
        "residual": False,
        "residual_out_dtype": "same",
        "dtype_names": MN_PRESET_GATE_DTYPES[preset],
        "mn_pairs": MN_PRESETS[preset],
        "fail_ratio": MN_PRESET_FAIL_RATIOS[preset],
    }


def gate_preset_names(gate):
    if gate == GATE_ALL:
        return tuple(sorted(MN_PRESET_GATE_DTYPES))
    if gate not in MN_PRESET_GATE_DTYPES:
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
            "fail_ratio target; use 'all' to run every maintained gate"
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
        "--quack-root",
        default=None,
        help="QuACK checkout containing the exhaustive RMSNorm tuned APIs",
    )
    parser.add_argument(
        "--quack-autotune",
        choices=("fresh", "cached"),
        default="fresh",
        help=(
            "Use 'fresh' to force QuACK to benchmark every valid config for each "
            "compared item, or 'cached' to reuse its per-item autotune cache"
        ),
    )
    parser.add_argument(
        "--quack-provenance-json",
        default=None,
        help="Optional path for QuACK source and autotune provenance metadata",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Write benchmark CSV rows directly to this path. This keeps native "
            "compiler diagnostics on the process stdout instead of corrupting CSV output."
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
        quack_configs = {row[4] for row in timings}
        if len(quack_configs) != 1:
            raise RuntimeError(
                "QuACK selected different autotune configs across provider orders: "
                f"{sorted(quack_configs)}"
            )
        return (
            mean([row[0] for row in timings]),
            mean([row[1] for row in timings]),
            None,
            min(row[3] for row in timings),
            quack_configs.pop(),
        )

    rmsnorm_fwd, rmsnorm_bwd, quack_fwd, quack_bwd, do_bench = benchmark_ops()
    if backward and residual:
        raise ValueError("--residual is only supported for forward benchmarks")
    residual_dtype = (
        torch.float32
        if not backward and residual_out_dtype == "float32"
        else None
    )
    quack_unsupported = (
        not backward and residual and dtype == torch.float32 and N >= 32768
    )
    quack_error = None
    quack_config = None

    # Autotune QuACK before allocating the large timing pool. Otherwise the
    # pool can consume most VRAM and make valid QuACK configs fail with OOM.
    torch.manual_seed(0)
    tuning_pool = make_input_pool(
        M,
        N,
        dtype,
        backward,
        residual,
        residual_out_dtype,
        1,
        pool_memory_fraction,
    )
    tuning_entry = tuning_pool[0]
    if backward:
        quack_bwd(
            tuning_entry["x"],
            tuning_entry["w"],
            tuning_entry["dout"],
            tuning_entry["rstd"],
        )
        quack_config = selected_quack_config(backward=True)
    elif not quack_unsupported:
        try:
            quack_fwd(
                tuning_entry["x"],
                tuning_entry["w"],
                residual=tuning_entry.get("residual"),
                residual_dtype=residual_dtype,
            )
            quack_config = selected_quack_config(backward=False)
        except RuntimeError as exc:
            message = str(exc).splitlines()[0]
            quack_error = f"{exc.__class__.__name__}: {message}"
    torch.cuda.synchronize()
    del tuning_pool, tuning_entry
    gc.collect()
    torch.cuda.empty_cache()

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
    else:
        first = pool[0]
        rmsnorm_fwd(
            first["x"],
            first["w"],
            residual=first.get("residual"),
            residual_dtype=residual_dtype,
        )
        if not quack_unsupported and quack_error is None:
            quack_fwd(
                first["x"],
                first["w"],
                residual=first.get("residual"),
                residual_dtype=residual_dtype,
            )
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
    return ours, quack, quack_error, pool_size, quack_config


def print_benchmark_row(
    M,
    N,
    pool_size,
    ours,
    quack,
    quack_error,
    quack_config,
    results,
    dtype_name=None,
    preset_name=None,
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
        print(
            f"{prefix}{M},{N},{pool_size},{ours:.6f},{quack_error},nan,nan,",
            flush=True,
        )
        return
    ratio = ours / quack
    results.append((M, N, ratio))
    print(
        f"{prefix}{M},{N},{pool_size},{ours:.6f},{quack:.6f},{ratio:.4f},"
        f"{format_speedup(ratio)},{quack_config}",
        flush=True,
    )


def run_rows(mn_pairs, dtype, args, dtype_name=None, preset_name=None):
    results = []
    for M, N in mn_pairs:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ours, quack, quack_error, pool_size, quack_config = bench_pair(
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
            quack_config,
            results,
            dtype_name=dtype_name,
            preset_name=preset_name,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def run_gate(args):
    print(
        "preset,dtype,M,N,pool_size,rmsnorm_ms,quack_ms,ratio,speedup,quack_config"
    )
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
                threshold_failures(results, fail_ratio=config["fail_ratio"])
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


def run_benchmark(args, parser):
    try:
        validate_args(args)
        configure_quack_root(args.quack_root)
        configure_quack_autotune(args.quack_autotune)
        emit_quack_provenance(args)
        if args.gate:
            run_gate(args)
            return
        dtype = DTYPES[args.dtype or "bfloat16"]
        mn_pairs = resolve_mn_pairs(args, dtype)
    except ValueError as exc:
        parser.error(str(exc))

    print("M,N,pool_size,rmsnorm_ms,quack_ms,ratio,speedup,quack_config")
    results = run_rows(mn_pairs, dtype, args)
    if args.summary:
        print("summary,count,min_speedup,geomean_speedup,max_speedup")
        print(format_summary(speedup_summary(results)))
    try:
        threshold_failures(results, args.fail_ratio, args.min_speedup)
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.list_presets:
        print(format_preset_listing())
        return
    if args.output is None:
        run_benchmark(args, parser)
        return
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        with redirect_stdout(output):
            run_benchmark(args, parser)


if __name__ == "__main__":
    main()
