# RMSNorm for Hopper

> One of the first four open-source GPU implementations produced by
> [INT21's PTX Kernel Factory](https://int21.ai).

`rmsnorm-h100` provides production-oriented RMSNorm kernels for NVIDIA Hopper
(`sm_90`), validated on an NVIDIA GH200 on July 8, 2026. The implementation
supports forward and backward execution and is written in CUDA C++ and inline
PTX.

## Highlights

- **1.45% BF16 and 1.56% FP16 forward geomean speedups** across the complete
  11-row benchmark against freshly autotuned QuACK.
- Corrected FP32 forward geomean: **0.03%**. A prior 16.19% claim came from two
  invalid partial-row launches and has been removed.
- Backward gate rows are maintained against autotuned QuACK with a 1% ratio
  tolerance for BF16 and FP16.
- Passed the complete validated package suite: **52 tests plus 72 subtests**.
- Supports FP16, BF16, and FP32; residual fusion; autograd; affine weight and
  bias; per-head parameters; and very large hidden dimensions.
- Uses no CUTLASS, CuTe, Triton, or QuACK implementation dependency. QuACK is
  used only as an external correctness and performance comparison target.

The benchmark report includes the boundaries of the result: some forward and
backward rows remain behind autotuned QuACK after enabling QuACK autotune per
shape. These are operator-level measurements, not claims about full-model
speedups.
Application-level gains depend on the model, workload, shapes, and surrounding
software stack. See [GH200 Results](#gh200-results) for the measured cases and
the paths to the raw CSV output.

## About PTX Kernel Factory

[PTX Kernel Factory](https://int21.ai) is INT21's system for generating and
improving low-level NVIDIA GPU software. Human engineers define the operation,
correctness requirements, target hardware, and success metric. The factory
coordinates multiple AI agents to generate candidates, compile them, reject
incorrect results, benchmark valid implementations, and carry useful evidence
into the next round.

This repository is part of the first public release:

| Workload | NVIDIA Hopper | NVIDIA Blackwell |
| --- | --- | --- |
| RMSNorm | **This repository** | [RMSNorm-B200](https://github.com/Int21-AI/RMSNorm-B200) |
| Kimi Delta Attention | [KDA-H100](https://github.com/Int21-AI/KDA-H100) | [KDA-B200](https://github.com/Int21-AI/KDA-B200) |

PTX Kernel Factory is in beta. Learn more and request early access at
[int21.ai](https://int21.ai).

## Requirements

- Linux
- Python 3.10+
- PyTorch with CUDA support
- CUDA toolkit with `nvcc`
- NVIDIA Hopper GPU (`sm_90`; H100 or GH200)

The extension is JIT-compiled on first CUDA use through
`torch.utils.cpp_extension`. On `sm_90`, the loader defaults
`TORCH_CUDA_ARCH_LIST` to `9.0a` so Hopper cluster PTX instructions are enabled.
Set `RMSNORM_VERBOSE_BUILD=1` to show the build commands.

## Install

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e . --no-build-isolation
python -m pytest -q
```

The distribution is named `rmsnorm-h100`; the import package is `rmsnorm`.

## API

```python
import torch
from rmsnorm import rmsnorm

x = torch.randn(32768, 16384, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(16384, device="cuda", dtype=torch.float32)
y = rmsnorm(x, weight, eps=1e-6)
```

The package exports QuACK-compatible entry points:

- `rmsnorm`, `rmsnorm_fwd`, `rmsnorm_bwd`
- `rmsnorm_fwd_tuned`, `rmsnorm_bwd_tuned`
- `rmsnorm_ref`, `rmsnorm_bwd_ref`
- `QuackRMSNorm`
- `layernorm_fwd` and LayerNorm reference helpers

CUDA fast paths support `float16`, `bfloat16`, and `float32`, optional residual
fusion, FP32 residual output, affine weight/bias, autograd, per-head affine
parameters, and very large hidden dimensions. Inputs that do not satisfy a
vectorized CUDA specialization (for example CPU tensors, non-vector-aligned
hidden sizes, mixed affine layouts, or compilation tracing) use the explicit
PyTorch reference-compatible fallback instead of silently producing an invalid
launch.

## Implementation

The package sources live under `rmsnorm/csrc/`:

- `ptx_rmsnorm_kernels.cu`: CUDA kernels, inline PTX vector loads/stores,
  `cp.async`, Hopper cluster barriers, distributed shared-memory paths, and
  forward/backward launch heuristics.
- `torch_bindings.cpp`: input validation, current-stream launches, output and
  workspace allocation, and PyTorch bindings.

A package test scans these sources and fails if CUTLASS or CuTe headers are
introduced. Neither library is a build or runtime dependency.

## Correctness

Run all tests from the repository root:

```bash
python -m pytest -q
```

The suite checks CPU/reference fallbacks and CUDA forward/backward results
against FP32 PyTorch references across dtypes, residual modes, output dtypes,
affine layouts, autograd, and large-row specialized kernels. CUDA tests require
a GPU and are skipped when CUDA is unavailable.

The verified GH200 run on July 8, 2026 passed:

```text
52 passed, 72 subtests passed
```

## Fair QuACK Benchmark

Install QuACK in a sibling checkout or set `QUACK_ROOT` to its repository:

```text
workspace/
├── quack/
└── rmsnorm-h100/
```

List maintained benchmark presets or run the gateable preset set:

```bash
rmsnorm-benchmark --list-presets
rmsnorm-benchmark --gate all --summary
```

Rows listed with `gate_dtypes=none`, if any, are runnable focused presets but
are not included in `--gate all`.

Run focused comparisons:

```bash
rmsnorm-benchmark --dtype bfloat16 --summary
rmsnorm-benchmark --dtype bfloat16 --backward --pair 32768,65536
rmsnorm-benchmark --dtype bfloat16 --residual --summary
rmsnorm-benchmark --dtype bfloat16 --residual \
  --residual-out-dtype float32 --summary
```

`--quack-autotune fresh` is the default. Use `--quack-root` to select the exact
checkout and `--quack-provenance-json` to record its revision, dirty state,
version, and candidate counts.

The benchmark is intentionally structured to avoid favorable special-casing:

- QuACK is measured through its autotuned RMSNorm entry points for each
  compared shape, dtype, and mode;
- fresh mode bypasses QuACK's disk cache and benchmarks every launch-valid
  candidate for each comparison key;
- QuACK autotuning happens before the large timing pool is allocated, so valid
  candidates are not rejected because the benchmark consumed device memory;
- each CSV row records the selected QuACK configuration and provenance can be
  emitted as JSON;
- both providers receive the same tensor objects and valid `rstd` values;
- random input pools rotate tensors during timing instead of reusing one input;
- both providers are warmed before measurement;
- the default `--provider-order balanced` measures fresh pools in both provider
  orders and averages each provider's timings;
- the same `triton.testing.do_bench` timer, warmup, repetition count, CUDA
  stream, dtype, shape, and operation semantics are used for both providers;
- unsupported QuACK rows are reported as unsupported and excluded from ratios,
  never counted as wins;
- threshold gates fail the process if any comparable maintained row exceeds its
  declared `fail_ratio` tolerance; current maintained gates use `1.01`.

`ratio` is `rmsnorm_ms / quack_ms`; `speedup` is
`quack_ms / rmsnorm_ms - 1`. Lower ratio and higher speedup are better.

### GH200 Results

Measured July 8, 2026 on an NVIDIA GH200 480GB (`sm_90`), CUDA 13.2,
PyTorch 2.12.0+cu132, with 10 warmup, 100 repetitions, a 64-entry target input
pool subject to the memory cap, balanced provider order, and fresh per-key
QuACK autotuning.

| Section | Rows | Min speedup | Geomean speedup | Max speedup |
|---|---:|---:|---:|---:|
| Forward BF16 | 11 | -1.23% | 1.45% | 15.10% |
| Forward FP16 | 11 | -1.16% | 1.56% | 13.28% |
| Forward FP32 | 11 | -1.21% | 0.03% | 2.71% |
| Residual BF16 | 11 | -0.92% | 1.19% | 8.74% |
| Residual FP16 | 11 | -0.99% | 1.52% | 8.39% |
| FP32 residual output, BF16 input | 11 | -0.78% | 0.18% | 2.55% |
| FP32 residual output, FP16 input | 11 | -0.80% | 0.34% | 4.38% |

The corrected audit found that FP32 `N=8192` and `N=32768` previously launched
fewer threads than the kernel's row tile required. Those overrides were removed
and the launcher now rejects `threads < threads_per_row`.

Post-baseline tuning removed the stale FP16 `N=65536` four-way split override.
The existing eight-way kernel improved PTX latency from 2.398744 ms to
2.380786 ms (0.75%), reducing the fresh QuACK deficit from 1.11% to 0.37%.

Nsight Compute found the remaining close losses to be primarily memory-bound.
At FP16 `N=8192`, PTX reached 3.55 TB/s and 88.20% DRAM throughput; QuACK
reached 3.58 TB/s and 89.00%. Lower-register, alternate-cluster, cache-policy,
and shared-reload candidates were rejected when they failed the balanced
rotating-pool benchmark.

The full methodology, backward table, correctness audit, tuning delta, rejected
experiments, and artifact map are in
[`benchmarks/results/2026-07-08-report.md`](benchmarks/results/2026-07-08-report.md).
Raw CSV and provenance files are under:

- `benchmarks/results/baseline-2026-07-08/`
- `benchmarks/results/tuned-2026-07-08/`
- `benchmarks/results/profile-2026-07-08/`

Historical pre-audit CSVs remain under `benchmarks/results/current/` for
traceability. They are superseded by the July 8 audit:

- `current/autotuned-quack-backward-bfloat16-large.csv`
- `current/autotuned-quack-backward-bfloat16-mid.csv`
- `current/autotuned-quack-backward-float16-large.csv`
- `current/autotuned-quack-backward-float16-mid.csv`
- `current/autotuned-quack-forward-bfloat16-over-65536-partial.csv`
- `current/autotuned-quack-forward-float32-65536-row.csv`
- `current/autotuned-quack-forward-{bfloat16,float16,float32}-through-65536.csv`
- `current/autotuned-quack-residual-{bfloat16,float16}-16384-row.csv`
- `current/autotuned-quack-residual-{bfloat16,float16}-through-65536.csv`
- `current/autotuned-quack-residual-fp32-{bfloat16,float16}-16384-row.csv`
- `current/autotuned-quack-residual-fp32-{bfloat16,float16}-through-65536.csv`

Benchmark numbers are environment-specific and should be regenerated on the
production driver, CUDA, PyTorch, and QuACK revisions used for deployment.

## License

MIT. See [`LICENSE`](LICENSE).
