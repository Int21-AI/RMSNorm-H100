# RMSNorm for Hopper

> One of the first four open-source GPU implementations produced by
> [INT21's PTX Kernel Factory](https://int21.ai).

`rmsnorm-h100` provides production-oriented RMSNorm kernels for NVIDIA Hopper
(`sm_90`), validated on an NVIDIA GH200 on June 8, 2026. The implementation
supports forward and backward execution and is written in CUDA C++ and inline
PTX.

## Highlights

- **8.17% faster than QuACK** on geometric mean across the 11 maintained BF16
  forward shapes.
- **15.03% to 34.09% faster** across the selected FP16 and BF16 backward
  comparisons.
- Passed the complete validated package suite: **48 tests plus 65 subtests**.
- Supports FP16, BF16, and FP32; residual fusion; autograd; affine weight and
  bias; per-head parameters; and very large hidden dimensions.
- Uses no CUTLASS, CuTe, Triton, or QuACK implementation dependency. QuACK is
  used only as an external correctness and performance comparison target.

The benchmark report includes the boundaries of the result: the slowest FP16
and FP32 forward rows were 0.42% and 0.84% behind QuACK. These are
operator-level measurements, not claims about full-model speedups.
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
pytest -q
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
pytest -q
```

The suite checks CPU/reference fallbacks and CUDA forward/backward results
against FP32 PyTorch references across dtypes, residual modes, output dtypes,
affine layouts, autograd, and large-row specialized kernels. CUDA tests require
a GPU and are skipped when CUDA is unavailable.

The verified GH200 run on June 8, 2026 passed:

```text
48 passed, 65 subtests passed
```

## Fair QuACK Benchmark

Install QuACK in a sibling checkout or set `QUACK_ROOT` to its repository:

```text
workspace/
├── quack/
└── rmsnorm-h100/
```

List or run the maintained gates:

```bash
rmsnorm-benchmark --list-presets
rmsnorm-benchmark --gate all --summary
```

Run focused comparisons:

```bash
rmsnorm-benchmark --dtype bfloat16 --summary
rmsnorm-benchmark --dtype bfloat16 --backward --pair 32768,65536
rmsnorm-benchmark --dtype bfloat16 --residual --summary
rmsnorm-benchmark --dtype bfloat16 --residual \
  --residual-out-dtype float32 --summary
```

The benchmark is intentionally structured to avoid favorable special-casing:

- both providers receive the same tensor objects and valid `rstd` values;
- random input pools rotate tensors during timing instead of reusing one input;
- both providers are warmed before measurement;
- the default `--provider-order balanced` measures fresh pools in both provider
  orders and averages each provider's timings;
- the same `triton.testing.do_bench` timer, warmup, repetition count, CUDA
  stream, dtype, shape, and operation semantics are used for both providers;
- unsupported QuACK rows are reported as unsupported and excluded from ratios,
  never counted as wins;
- threshold gates fail the process if any comparable maintained row misses its
  declared minimum speedup.

`ratio` is `rmsnorm_ms / quack_ms`; `speedup` is
`quack_ms / rmsnorm_ms - 1`. Lower ratio and higher speedup are better.

### GH200 Results

Measured June 8, 2026 on an NVIDIA GH200 480GB (`sm_90`), CUDA 13.2,
PyTorch 2.12.0+cu132, with the default 10 warmup iterations, 100 repetitions,
64-entry target input pool, memory cap, and balanced provider order.

| Preset | Dtype | Shape | rmsnorm ms | QuACK ms | Speedup |
|---|---:|---:|---:|---:|---:|
| mid-backward | bfloat16 | 32768x16384 | 0.934867 | 1.147325 | 22.73% |
| mid-backward | float16 | 32768x16384 | 0.933622 | 1.114430 | 19.37% |
| large-backward | bfloat16 | 32768x65536 | 4.315051 | 5.086070 | 17.87% |
| large-backward | bfloat16 | 16384x131072 | 4.539361 | 6.008948 | 32.37% |
| large-backward | bfloat16 | 8192x262144 | 3.859709 | 5.175591 | 34.09% |
| large-backward | float16 | 32768x65536 | 4.342459 | 4.995056 | 15.03% |
| large-backward | float16 | 16384x131072 | 4.507835 | 5.997536 | 33.05% |
| large-backward | float16 | 8192x262144 | 3.849178 | 5.026853 | 30.60% |

Balanced full-forward geometric-mean speedups across the 11 maintained shapes
were 8.17% for BF16, 2.95% for FP16, and 1.16% for FP32. Every BF16 forward row
was at least 0.35% faster than QuACK. The slowest FP16 and FP32 rows were 0.42%
and 0.84% behind QuACK respectively; these are reported rather than hidden.
BF16 residual-forward geometric-mean speedup was 5.76% for same-dtype residual
output and 2.06% for FP32 residual output.

Version 0.1.1 narrows the no-rstd FP16/BF16 CTA from 512 to 128 threads at
`N=8192` and from 1024 to 512 threads at `N=32768`. Against the immediately
preceding balanced baseline, package latency improved 1.83%/1.83% at `N=8192`
and 1.91%/1.79% at `N=32768` for BF16/FP16 respectively. FP32, residual, and
rstd-producing launches retain their previous dispatch.

Version 0.1.2 uses four cluster CTAs instead of eight for the FP16,
weight-only, no-rstd `N=65536` path. Latency improved from 2.396176 ms to
2.380595 ms (0.65%) against the immediately preceding balanced baseline, moving
the row from parity to 0.65% faster than QuACK. The override is intentionally
scoped away from residual fusion: same-dtype and FP32 residual-output checks
remain 8.57% and 1.80% faster than QuACK, respectively.

Nsight Compute measured the FP16 `N=32768` kernel at 89.84% of available DRAM
throughput and only 20.30% compute throughput. Thread-count, cache-policy,
weight-preload, persistent shared-weight, and alternate cluster experiments
were rejected when they failed balanced benchmarks. This indicates the
remaining large-row forward gap is primarily memory-bandwidth headroom rather
than missing arithmetic optimization.

Raw CSV output is committed under `benchmarks/results/`:

- `gh200-2026-06-08-backward-gates.csv`
- `gh200-2026-06-08-forward-{bfloat16,float16,float32}.csv`
- `gh200-2026-06-08-residual-bfloat16.csv`
- `gh200-2026-06-08-residual-fp32-bfloat16.csv`
- `gh200-2026-06-08-residual-float16-n65536.csv`
- `gh200-2026-06-08-residual-fp32-float16-n65536.csv`

Benchmark numbers are environment-specific and should be regenerated on the
production driver, CUDA, PyTorch, and QuACK revisions used for deployment.

## License

MIT. See [`LICENSE`](LICENSE).
