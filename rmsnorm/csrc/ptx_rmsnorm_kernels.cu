// Copyright 2026 INT21 AI
// SPDX-License-Identifier: MIT

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
namespace cg = cooperative_groups;

// fp32 register-cluster kernels keep row tiles in registers; reserving dynamic
// shared memory as a carveout was neutral or slower on the large rows.
constexpr int kRegClusterCarveoutBytes = 0;
constexpr int kHalfRegClusterCarveoutBytes = 36 * 1024;
// Deeper weight prefetching helps the fp32 kVecs>=16 register-cluster paths,
// but it increases register pressure enough to hurt the half/bf16 variants.
constexpr int kRegClusterWeightPrefetchVecs = 12;
// Residual large-row half/bf16 paths have enough memory work to benefit from
// weight prefetching, but fp16 and bf16 hit the register-pressure cliff at
// different depths.
constexpr int kResidualClusterFp16WeightPrefetchVecs = 5;
constexpr int kResidualClusterBf16WeightPrefetchVecs = 6;

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t status__ = (expr);                                              \
    if (status__ != cudaSuccess) {                                              \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                               std::to_string(__LINE__) + ": " +               \
                               cudaGetErrorString(status__));                   \
    }                                                                           \
  } while (0)

enum class DType { kF16, kBF16, kF32 };
enum class Mode { kForward, kBackward, kBoth };
enum class ResidualOutDType { kSame, kF32 };

struct Options {
  Mode mode = Mode::kForward;
  int M = 4096;
  int N = 4096;
  int H = 1;
  int threads = 0;
  int partial_blocks = 0;
  int reduce_chunks = 0;
  int warmup_iterations = 10;
  int iterations = 100;
  float eps = 1.0e-6f;
  DType dtype = DType::kBF16;
  bool has_weight = true;
  bool has_bias = false;
  bool has_residual = false;
  bool store_rstd = false;
  bool has_dresidual_out = false;
  ResidualOutDType residual_out_dtype = ResidualOutDType::kSame;
  bool verify = true;
  bool benchmark = true;
  bool verbose = false;
  unsigned seed = 20260514u;
};

bool parse_bool(char const* value) {
  if (value == nullptr) {
    return true;
  }
  std::string text(value);
  return text == "1" || text == "true" || text == "True" || text == "yes" ||
         text == "on";
}

DType parse_dtype(std::string const& text) {
  if (text == "fp16" || text == "float16" || text == "half") {
    return DType::kF16;
  }
  if (text == "bf16" || text == "bfloat16") {
    return DType::kBF16;
  }
  if (text == "fp32" || text == "float32" || text == "float") {
    return DType::kF32;
  }
  std::cerr << "Invalid dtype: " << text << "\n";
  std::exit(EXIT_FAILURE);
}

char const* dtype_name(DType dtype) {
  switch (dtype) {
    case DType::kF16:
      return "fp16";
    case DType::kBF16:
      return "bf16";
    case DType::kF32:
      return "fp32";
  }
  return "unknown";
}

void print_usage(char const* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --mode=<fwd|bwd|both>       Which RMSNorm path to run\n"
      << "  --M=<int>                   Number of normalization rows\n"
      << "  --N=<int>                   Hidden dimension\n"
      << "  --H=<int>                   Heads for per-head weight indexing\n"
      << "  --threads=<int>             Threads per CTA, multiple of 32\n"
      << "  --partial_blocks=<int>      Override backward persistent CTA count\n"
      << "  --reduce_chunks=<int>       Chunk final backward dw reduction\n"
      << "  --dtype=<fp16|bf16|fp32>    Input/output dtype\n"
      << "  --weight=<bool>             Use fp32 weight\n"
      << "  --bias=<bool>               Use fp32 bias / compute db in bwd\n"
      << "  --residual=<bool>           Add same-dtype residual in fwd\n"
      << "  --residual_out_dtype=<same|fp32>\n"
      << "  --store_rstd=<bool>         Store fwd rstd\n"
      << "  --dresidual_out=<bool>      Add grad from residual_out in bwd\n"
      << "  --eps=<float>               Epsilon\n"
      << "  --iterations=<int>          Timed iterations\n"
      << "  --warmup_iterations=<int>   Warmup iterations\n"
      << "  --verify=<bool>             Run CPU numerical check\n"
      << "  --benchmark=<bool>          Run timed benchmark\n"
      << "  --seed=<int>                RNG seed\n"
      << "  --verbose=<bool>            Print extra diagnostics\n";
}

void parse_options(int argc, char const** argv, Options& options) {
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }

    std::string key = arg;
    char const* value = nullptr;
    auto eq = arg.find('=');
    if (eq != std::string::npos) {
      key = arg.substr(0, eq);
      value = argv[i] + eq + 1;
    } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
      value = argv[++i];
    }

    if (key == "--mode" && value) {
      std::string mode(value);
      if (mode == "fwd" || mode == "forward") {
        options.mode = Mode::kForward;
      } else if (mode == "bwd" || mode == "backward") {
        options.mode = Mode::kBackward;
      } else if (mode == "both") {
        options.mode = Mode::kBoth;
      } else {
        std::cerr << "Invalid mode: " << mode << "\n";
        std::exit(EXIT_FAILURE);
      }
    } else if (key == "--M" && value) {
      options.M = std::atoi(value);
    } else if (key == "--N" && value) {
      options.N = std::atoi(value);
    } else if (key == "--H" && value) {
      options.H = std::atoi(value);
    } else if (key == "--threads" && value) {
      options.threads = std::atoi(value);
    } else if (key == "--partial_blocks" && value) {
      options.partial_blocks = std::atoi(value);
    } else if (key == "--reduce_chunks" && value) {
      options.reduce_chunks = std::atoi(value);
    } else if (key == "--dtype" && value) {
      options.dtype = parse_dtype(value);
    } else if (key == "--weight") {
      options.has_weight = parse_bool(value);
    } else if (key == "--bias") {
      options.has_bias = parse_bool(value);
    } else if (key == "--residual") {
      options.has_residual = parse_bool(value);
    } else if (key == "--residual_out_dtype" && value) {
      std::string dtype(value);
      if (dtype == "same") {
        options.residual_out_dtype = ResidualOutDType::kSame;
      } else if (dtype == "fp32" || dtype == "float32") {
        options.residual_out_dtype = ResidualOutDType::kF32;
      } else {
        std::cerr << "Invalid residual_out_dtype: " << dtype << "\n";
        std::exit(EXIT_FAILURE);
      }
    } else if (key == "--store_rstd") {
      options.store_rstd = parse_bool(value);
    } else if (key == "--dresidual_out") {
      options.has_dresidual_out = parse_bool(value);
    } else if (key == "--eps" && value) {
      options.eps = std::strtof(value, nullptr);
    } else if (key == "--iterations" && value) {
      options.iterations = std::atoi(value);
    } else if (key == "--warmup_iterations" && value) {
      options.warmup_iterations = std::atoi(value);
    } else if (key == "--verify") {
      options.verify = parse_bool(value);
    } else if (key == "--benchmark") {
      options.benchmark = parse_bool(value);
    } else if (key == "--seed" && value) {
      options.seed = static_cast<unsigned>(std::strtoul(value, nullptr, 10));
    } else if (key == "--verbose") {
      options.verbose = parse_bool(value);
    } else {
      std::cerr << "Unknown or malformed option: " << arg << "\n";
      print_usage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }
}

int heuristic_threads(int N) {
  if (N == 4096) {
    return 256;
  }
  if (N >= 8192 && N <= 16384) {
    return 512;
  }
  if (N == 32768) {
    return 1024;
  }
  return N <= 16384 ? 128 : 256;
}

int bwd_threads(int N) {
  if (N <= 256) {
    return 32;
  }
  if (N <= 1024) {
    return 64;
  }
  if (N <= 2048) {
    return 256;
  }
  if (N <= 4096) {
    return 128;
  }
  // Half/bf16 backward rows at 8192/16384 are register-pressure sensitive on
  // H100; these narrower CTAs beat the older 512-thread shape without changing
  // the partial-block schedule.
  if (N <= 8192) {
    return 256;
  }
  if (N <= 16384) {
    return 128;
  }
  if (N <= 32768) {
    return 128;
  }
  if (N <= 65536) {
    return 256;
  }
  if (N <= 131072) {
    return 512;
  }
  return 256;
}

int bwd_partial_blocks(int N, int M, int requested = 0) {
  if (requested > 0) {
    return std::max(1, std::min(M, requested));
  }
  int device = 0;
  int sm_count = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  int multiple =
      N <= 256 ? 8
               : (N <= 1024 ? 8 : (N <= 2048 ? 4 : (N <= 4096 ? 2 : 1)));
  int blocks = N <= 8192 ? sm_count * multiple
                         : (N <= 16384 ? sm_count : sm_count * 2);
  return std::max(1, std::min(M, blocks));
}

int bwd_reduce_chunks(int N, int partial_blocks, int requested = 0) {
  if (requested > 0) {
    return std::max(1, std::min(partial_blocks, requested));
  }
  if (partial_blocks < 128) {
    return 1;
  }
  if (N <= 256) {
    return std::min(partial_blocks, 16);
  }
  if (N <= 512) {
    return std::min(partial_blocks, 32);
  }
  if (N <= 4096) {
    return std::min(partial_blocks, 16);
  }
  if (N <= 8192) {
    return std::min(partial_blocks, 16);
  }
  return 1;
}

int fwd_threads_per_row(int N) {
  if (N <= 2048) {
    return 32;
  }
  if (N == 4096) {
    return 256;
  }
  if (N >= 8192 && N <= 16384) {
    return 512;
  }
  if (N == 32768) {
    return 1024;
  }
  return N <= 16384 ? 128 : 256;
}

template <typename X>
int fwd_split_parts(int N) {
  if constexpr (sizeof(X) >= 4) {
    if (N <= 32768) {
      return 1;
    }
    if (N <= 65536) {
      return 16;
    }
    if (N <= 131072) {
      return 8;
    }
    if (N <= 262144) {
      return 16;
    }
    return 16;
  } else {
    if (N <= 32768) {
      return 1;
    }
    if (N <= 65536) {
      return 8;
    }
    if (N <= 131072) {
      return 16;
    }
    // For half/bf16 large rows, the 16-way split keeps the register-cluster
    // kernel faster than the 8-way and lower-thread-count variants.
    return 16;
  }
}

template <typename X>
int bwd_cluster_parts(int N) {
  if (N <= 4096) {
    return 1;
  }
  if constexpr (sizeof(X) >= 4) {
    if (N <= 8192) {
      // H100 fp32 backward at 8192 is faster as a single CTA-row tile; the
      // two-part cluster pays more launch/reduction overhead than it saves.
      return 1;
    }
    if (N <= 16384) {
      return 2;
    }
    if (N <= 32768) {
      // More, narrower fp32 row parts reduce the per-cluster tile at N=32768
      // enough to beat the older 4-way split without changing larger rows.
      return 16;
    }
    if (N <= 65536) {
      return 16;
    }
    return 16;
  } else {
    if (N <= 8192) {
      return 1;
    }
    if (N <= 16384) {
      return 2;
    }
    return 16;
  }
}

template <typename T>
struct Vec128 {
  static constexpr int kLanes = 16 / sizeof(T);
  uint4 bits;

  __device__ __forceinline__ void load(T const* ptr) {
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
                 : "l"(ptr));
  }

  __device__ __forceinline__ void load_cg(T const* ptr) {
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
                 : "l"(ptr));
  }

  __device__ __forceinline__ void load_shared(T const* ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
                 : "r"(smem_addr));
  }

  __device__ __forceinline__ void store(T* ptr) const {
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(ptr),
                 "r"(bits.x), "r"(bits.y), "r"(bits.z), "r"(bits.w));
  }

  __device__ __forceinline__ void store_shared(T* ptr) const {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(smem_addr),
                 "r"(bits.x), "r"(bits.y), "r"(bits.z), "r"(bits.w));
  }

  __device__ __forceinline__ float get(int lane) const {
    if constexpr (std::is_same_v<T, float>) {
      uint32_t word =
          lane == 0 ? bits.x : (lane == 1 ? bits.y : (lane == 2 ? bits.z : bits.w));
      return __uint_as_float(word);
    } else if constexpr (std::is_same_v<T, __half>) {
      uint32_t word =
          lane < 2 ? bits.x : (lane < 4 ? bits.y : (lane < 6 ? bits.z : bits.w));
      uint16_t hbits = static_cast<uint16_t>((word >> ((lane & 1) * 16)) & 0xffffu);
      union {
        uint16_t u;
        __half h;
      } cvt;
      cvt.u = hbits;
      return __half2float(cvt.h);
    } else {
      uint32_t word =
          lane < 2 ? bits.x : (lane < 4 ? bits.y : (lane < 6 ? bits.z : bits.w));
      uint16_t bbits = static_cast<uint16_t>((word >> ((lane & 1) * 16)) & 0xffffu);
      __nv_bfloat16_raw raw;
      raw.x = bbits;
      return __bfloat162float(__nv_bfloat16(raw));
    }
  }

  __device__ __forceinline__ void set(int lane, float value) {
    if constexpr (std::is_same_v<T, float>) {
      uint32_t word = __float_as_uint(value);
      if (lane == 0) {
        bits.x = word;
      } else if (lane == 1) {
        bits.y = word;
      } else if (lane == 2) {
        bits.z = word;
      } else {
        bits.w = word;
      }
    } else if constexpr (std::is_same_v<T, __half>) {
      union {
        uint16_t u;
        __half h;
      } cvt;
      cvt.h = __float2half_rn(value);
      uint32_t mask = 0xffffu << ((lane & 1) * 16);
      uint32_t packed = static_cast<uint32_t>(cvt.u) << ((lane & 1) * 16);
      uint32_t* word =
          lane < 2 ? &bits.x : (lane < 4 ? &bits.y : (lane < 6 ? &bits.z : &bits.w));
      *word = (*word & ~mask) | packed;
    } else {
      __nv_bfloat16_raw raw = static_cast<__nv_bfloat16_raw>(__float2bfloat16(value));
      uint32_t mask = 0xffffu << ((lane & 1) * 16);
      uint32_t packed = static_cast<uint32_t>(raw.x) << ((lane & 1) * 16);
      uint32_t* word =
          lane < 2 ? &bits.x : (lane < 4 ? &bits.y : (lane < 6 ? &bits.z : &bits.w));
      *word = (*word & ~mask) | packed;
    }
  }
};

template <typename T>
struct Vec64 {
  static constexpr int kLanes = 8 / sizeof(T);
  static_assert(kLanes == 4, "Vec64 is only used for half/bf16 mixed rows");
  uint2 bits;

  __device__ __forceinline__ void load(T const* ptr) {
    asm volatile("ld.global.v2.u32 {%0, %1}, [%2];"
                 : "=r"(bits.x), "=r"(bits.y)
                 : "l"(ptr));
  }

  __device__ __forceinline__ void store(T* ptr) const {
    asm volatile("st.global.v2.u32 [%0], {%1, %2};" ::"l"(ptr),
                 "r"(bits.x), "r"(bits.y));
  }

  __device__ __forceinline__ float get(int lane) const {
    uint32_t word = lane < 2 ? bits.x : bits.y;
    uint16_t raw = static_cast<uint16_t>((word >> ((lane & 1) * 16)) & 0xffffu);
    if constexpr (std::is_same_v<T, __half>) {
      union {
        uint16_t u;
        __half h;
      } cvt;
      cvt.u = raw;
      return __half2float(cvt.h);
    } else {
      __nv_bfloat16_raw bf_raw;
      bf_raw.x = raw;
      return __bfloat162float(__nv_bfloat16(bf_raw));
    }
  }

  __device__ __forceinline__ void set(int lane, float value) {
    uint32_t packed;
    if constexpr (std::is_same_v<T, __half>) {
      union {
        uint16_t u;
        __half h;
      } cvt;
      cvt.h = __float2half_rn(value);
      packed = static_cast<uint32_t>(cvt.u) << ((lane & 1) * 16);
    } else {
      __nv_bfloat16_raw raw = static_cast<__nv_bfloat16_raw>(__float2bfloat16(value));
      packed = static_cast<uint32_t>(raw.x) << ((lane & 1) * 16);
    }
    uint32_t mask = 0xffffu << ((lane & 1) * 16);
    uint32_t* word = lane < 2 ? &bits.x : &bits.y;
    *word = (*word & ~mask) | packed;
  }
};

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  union {
    __nv_bfloat162 v;
    uint32_t u;
  } cvt;
  cvt.v = __floats2bfloat162_rn(lo, hi);
  return cvt.u;
}

__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  union {
    __half2 v;
    uint32_t u;
  } cvt;
  cvt.v = __floats2half2_rn(lo, hi);
  return cvt.u;
}

template <typename T, int kLanes>
__device__ __forceinline__ void set_vec_from_float_lanes(
    Vec128<T>& vec, float const (&values)[kLanes]) {
  static_assert(kLanes == 4 || kLanes == 8);
  if constexpr (std::is_same_v<T, float>) {
    vec.bits.x = __float_as_uint(values[0]);
    vec.bits.y = __float_as_uint(values[1]);
    vec.bits.z = __float_as_uint(values[2]);
    vec.bits.w = __float_as_uint(values[3]);
  } else if constexpr (std::is_same_v<T, __half>) {
    vec.bits.x = pack_f16x2(values[0], values[1]);
    vec.bits.y = pack_f16x2(values[2], values[3]);
    if constexpr (kLanes > 4) {
      vec.bits.z = pack_f16x2(values[4], values[5]);
      vec.bits.w = pack_f16x2(values[6], values[7]);
    }
  } else {
    vec.bits.x = pack_bf16x2(values[0], values[1]);
    vec.bits.y = pack_bf16x2(values[2], values[3]);
    if constexpr (kLanes > 4) {
      vec.bits.z = pack_bf16x2(values[4], values[5]);
      vec.bits.w = pack_bf16x2(values[6], values[7]);
    }
  }
}

template <int kLanes>
__device__ __forceinline__ void store_float_lanes(float* ptr,
                                                  float const (&values)[kLanes]) {
  static_assert(kLanes == 4 || kLanes == 8);
  Vec128<float> vec0;
  vec0.bits.x = __float_as_uint(values[0]);
  vec0.bits.y = __float_as_uint(values[1]);
  vec0.bits.z = __float_as_uint(values[2]);
  vec0.bits.w = __float_as_uint(values[3]);
  vec0.store(ptr);
  if constexpr (kLanes > 4) {
    Vec128<float> vec1;
    vec1.bits.x = __float_as_uint(values[4]);
    vec1.bits.y = __float_as_uint(values[5]);
    vec1.bits.z = __float_as_uint(values[6]);
    vec1.bits.w = __float_as_uint(values[7]);
    vec1.store(ptr + 4);
  }
}

template <typename RO, int kLanes>
__device__ __forceinline__ void store_residual_lanes(
    RO* ptr, float const (&values)[kLanes]) {
  if constexpr (std::is_same_v<RO, float>) {
    store_float_lanes<kLanes>(ptr, values);
  } else {
    Vec128<RO> vec;
    set_vec_from_float_lanes(vec, values);
    vec.store(ptr);
  }
}

template <typename T>
__device__ __forceinline__ float accumulate_sq_vec(Vec128<T> const& vec,
                                                   float sq) {
  constexpr int kLanes = Vec128<T>::kLanes;
  if constexpr (std::is_same_v<T, float>) {
    float x0 = __uint_as_float(vec.bits.x);
    float x1 = __uint_as_float(vec.bits.y);
    float x2 = __uint_as_float(vec.bits.z);
    float x3 = __uint_as_float(vec.bits.w);
    sq = fmaf(x0, x0, sq);
    sq = fmaf(x1, x1, sq);
    sq = fmaf(x2, x2, sq);
    sq = fmaf(x3, x3, sq);
    return sq;
  } else {
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = vec.get(lane);
      sq = fmaf(xv, xv, sq);
    }
    return sq;
  }
}

template <typename X, int kLanes>
__device__ __forceinline__ void add_vec_lanes(Vec128<X> const& lhs,
                                              Vec128<X> const& rhs,
                                              float (&values)[kLanes]) {
  static_assert(kLanes == 4 || kLanes == 8);
  if constexpr (std::is_same_v<X, __half>) {
    static_assert(kLanes == 8);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      uint32_t lhs_bits =
          pair == 0 ? lhs.bits.x
                    : (pair == 1 ? lhs.bits.y
                                 : (pair == 2 ? lhs.bits.z : lhs.bits.w));
      uint32_t rhs_bits =
          pair == 0 ? rhs.bits.x
                    : (pair == 1 ? rhs.bits.y
                                 : (pair == 2 ? rhs.bits.z : rhs.bits.w));
      union {
        uint32_t u;
        __half2 v;
      } lhs_pair, rhs_pair;
      lhs_pair.u = lhs_bits;
      rhs_pair.u = rhs_bits;
      float2 lhs_f = __half22float2(lhs_pair.v);
      float2 rhs_f = __half22float2(rhs_pair.v);
      values[pair * 2] = lhs_f.x + rhs_f.x;
      values[pair * 2 + 1] = lhs_f.y + rhs_f.y;
    }
  } else if constexpr (std::is_same_v<X, __nv_bfloat16>) {
    static_assert(kLanes == 8);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      uint32_t lhs_bits =
          pair == 0 ? lhs.bits.x
                    : (pair == 1 ? lhs.bits.y
                                 : (pair == 2 ? lhs.bits.z : lhs.bits.w));
      uint32_t rhs_bits =
          pair == 0 ? rhs.bits.x
                    : (pair == 1 ? rhs.bits.y
                                 : (pair == 2 ? rhs.bits.z : rhs.bits.w));
      union {
        uint32_t u;
        __nv_bfloat162 v;
      } lhs_pair, rhs_pair;
      lhs_pair.u = lhs_bits;
      rhs_pair.u = rhs_bits;
      float2 lhs_f = __bfloat1622float2(lhs_pair.v);
      float2 rhs_f = __bfloat1622float2(rhs_pair.v);
      values[pair * 2] = lhs_f.x + rhs_f.x;
      values[pair * 2 + 1] = lhs_f.y + rhs_f.y;
    }
  } else {
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      values[lane] = lhs.get(lane) + rhs.get(lane);
    }
  }
}

template <typename T>
__host__ __device__ __forceinline__ float scalar_to_float(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, __half>) {
    return __half2float(value);
  } else {
    return __bfloat162float(value);
  }
}

template <typename T>
__host__ __device__ __forceinline__ T scalar_from_float(float value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, __half>) {
    return __float2half_rn(value);
  } else {
    return __float2bfloat16(value);
  }
}

__device__ __forceinline__ float ptx_shfl_bfly(float value, int offset) {
  float out;
  asm volatile(
      "{ .reg .b32 b; mov.b32 b, %1; "
      "shfl.sync.bfly.b32 b, b, %2, 0x1f, 0xffffffff; mov.b32 %0, b; }"
      : "=f"(out)
      : "f"(value), "r"(offset));
  return out;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    value += ptx_shfl_bfly(value, offset);
  }
  return value;
}

__device__ float cta_reduce_sum(float value, float* smem_acc) {
  if (blockDim.x <= 32) {
    return warp_reduce_sum(value);
  }

  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane_id = tid & 31;
  int warps = (blockDim.x + 31) >> 5;

  value = warp_reduce_sum(value);
  if (lane_id == 0) {
    smem_acc[warp_id] = value;
  }
  __syncthreads();

  value = lane_id < warps ? smem_acc[lane_id] : 0.0f;
  return warp_reduce_sum(value);
}

__device__ float cta_reduce_sum_leader(float value, float* smem_acc) {
  if (blockDim.x <= 32) {
    return warp_reduce_sum(value);
  }

  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane_id = tid & 31;
  int warps = (blockDim.x + 31) >> 5;

  value = warp_reduce_sum(value);
  if (lane_id == 0) {
    smem_acc[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0) {
    value = lane_id < warps ? smem_acc[lane_id] : 0.0f;
    return warp_reduce_sum(value);
  }
  return 0.0f;
}

__device__ float row_group_reduce_sum(float value, float* smem_acc,
                                      int threads_per_row) {
  if (threads_per_row <= 32) {
    return warp_reduce_sum(value);
  }

  int tid = threadIdx.x;
  int row_group = tid / threads_per_row;
  int row_tid = tid - row_group * threads_per_row;
  int warp_in_row = row_tid >> 5;
  int lane_id = row_tid & 31;
  int warps_per_row = threads_per_row >> 5;
  int smem_base = row_group * warps_per_row;

  value = warp_reduce_sum(value);
  if (lane_id == 0) {
    smem_acc[smem_base + warp_in_row] = value;
  }
  __syncthreads();

  if (warp_in_row == 0) {
    value = lane_id < warps_per_row ? smem_acc[smem_base + lane_id] : 0.0f;
    value = warp_reduce_sum(value);
    if (lane_id == 0) {
      smem_acc[smem_base] = value;
    }
  }
  __syncthreads();
  return smem_acc[smem_base];
}

__device__ __forceinline__ float ptx_rsqrt(float value) {
  float out;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(value));
  return out;
}

__device__ __forceinline__ void cp_async_128(void* smem_ptr,
                                             void const* gmem_ptr) {
  uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
               :
               : "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group_0() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group_1() {
  asm volatile("cp.async.wait_group 1;" ::: "memory");
}

__device__ __forceinline__ uint32_t smem_ptr_to_uint(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t rank = 0;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  return rank;
}

__device__ __forceinline__ void cluster_arrive_relaxed_barrier() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;");
}

__device__ __forceinline__ void cluster_wait_barrier() {
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

__device__ __forceinline__ void mbarrier_init(uint64_t* barrier, int count) {
  uint32_t addr = smem_ptr_to_uint(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr),
               "r"(count)
               : "memory");
}

__device__ __forceinline__ void mbarrier_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* barrier,
                                                           uint32_t bytes) {
  uint32_t addr = smem_ptr_to_uint(barrier);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :
               : "r"(addr), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_remote(uint64_t* barrier,
                                                        uint32_t cta_rank) {
  uint32_t addr = smem_ptr_to_uint(barrier);
  asm volatile(
      "{ .reg .b32 remote_addr; "
      "mapa.shared::cluster.u32 remote_addr, %0, %1; "
      "mbarrier.arrive.shared::cluster.b64 _, [remote_addr]; }"
      :
      : "r"(addr), "r"(cta_rank)
      : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* barrier, int phase) {
  uint32_t addr = smem_ptr_to_uint(barrier);
  uint32_t ticks = 0x989680;
  asm volatile(
      "{ .reg .pred p; "
      "wait_loop%=: "
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2; "
      "@p bra wait_done%=; "
      "bra wait_loop%=; "
      "wait_done%=: }"
      :
      : "r"(addr), "r"(phase), "r"(ticks)
      : "memory");
}

__device__ __forceinline__ void fence_view_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void store_shared_remote_float(
    float value, float* smem_ptr, uint64_t* mbarrier_ptr, uint32_t cta_rank) {
  uint32_t smem_addr = smem_ptr_to_uint(smem_ptr);
  uint32_t mbar_addr = smem_ptr_to_uint(mbarrier_ptr);
  asm volatile(
      "{ .reg .b32 remote_smem; .reg .b32 remote_mbar; "
      "mapa.shared::cluster.u32 remote_smem, %0, %3; "
      "mapa.shared::cluster.u32 remote_mbar, %1, %3; "
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 "
      "[remote_smem], %2, [remote_mbar]; }"
      :
      : "r"(smem_addr), "r"(mbar_addr), "f"(value), "r"(cta_rank)
      : "memory");
}

__device__ float cluster_reduce_sum_mbar(float value, float* reduce_buf,
                                         uint64_t* full_barrier, int stage,
                                         int phase, int parts) {
  int tid = threadIdx.x;
  int lane_id = tid & 31;
  int warp_id = tid >> 5;
  int warps = (blockDim.x + 31) >> 5;
  int rank = static_cast<int>(cluster_rank());

  value = warp_reduce_sum(value);
  float* stage_buf = reduce_buf + stage * 16 * 16;
  if (warp_id == 0 && lane_id == 0) {
    mbarrier_arrive_expect_tx(full_barrier + stage,
                              static_cast<uint32_t>(warps * parts *
                                                    sizeof(float)));
  }
  if (lane_id < parts) {
    store_shared_remote_float(value, stage_buf + warp_id * 16 + rank,
                              full_barrier + stage,
                              static_cast<uint32_t>(lane_id));
  }
  mbarrier_wait(full_barrier + stage, phase);

  float sum = 0.0f;
  int values = warps * parts;
  if (parts == 16) {
    for (int idx = lane_id; idx < values; idx += 32) {
      sum += stage_buf[idx];
    }
  } else if (parts == 8) {
    for (int idx = lane_id; idx < values; idx += 32) {
      int reduce_warp = idx >> 3;
      int reduce_rank = idx & 7;
      sum += stage_buf[reduce_warp * 16 + reduce_rank];
    }
  } else if (parts == 4) {
    for (int idx = lane_id; idx < values; idx += 32) {
      int reduce_warp = idx >> 2;
      int reduce_rank = idx & 3;
      sum += stage_buf[reduce_warp * 16 + reduce_rank];
    }
  } else {
    for (int idx = lane_id; idx < values; idx += 32) {
      int reduce_warp = idx >> 1;
      int reduce_rank = idx & 1;
      sum += stage_buf[reduce_warp * 16 + reduce_rank];
    }
  }
  return warp_reduce_sum(sum);
}

template <typename X, typename RO, int kMaxVecs>
__global__ void rmsnorm_fwd_cached_kernel(
    X* __restrict__ out, RO* __restrict__ residual_out,
    float* __restrict__ rstd_out, X const* __restrict__ x,
    float const* __restrict__ weight, float const* __restrict__ bias,
    X const* __restrict__ residual, int M, int N, int H, int threads_per_row,
    float eps) {
  __shared__ float smem_acc[64];
  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  int tid = threadIdx.x;
  int row_group = tid / threads_per_row;
  int row_tid = tid - row_group * threads_per_row;
  int rows_per_cta = blockDim.x / threads_per_row;
  int row = blockIdx.x * rows_per_cta + row_group;
  int head = H > 1 ? row % H : 0;
  int vecs = N / kLanes;
  int row_base = row * N;
  int affine_base = head * N;

  XVec x_vec[kMaxVecs];
  XVec res_vec[kMaxVecs];
  bool pred[kMaxVecs];
  float sq = 0.0f;

#pragma unroll
  for (int iter = 0; iter < kMaxVecs; ++iter) {
    int vec_idx = iter * threads_per_row + row_tid;
    bool valid = row < M && vec_idx < vecs;
    pred[iter] = valid;
    if (valid) {
      int elem = vec_idx * kLanes;
      x_vec[iter].load(x + row_base + elem);
      if (residual != nullptr) {
        res_vec[iter].load(residual + row_base + elem);
      }
#pragma unroll
      for (int lane = 0; lane < kLanes; ++lane) {
        float xv = x_vec[iter].get(lane);
        if (residual != nullptr) {
          xv += res_vec[iter].get(lane);
        }
        sq = fmaf(xv, xv, sq);
      }
    } else {
      x_vec[iter].bits = make_uint4(0, 0, 0, 0);
      res_vec[iter].bits = make_uint4(0, 0, 0, 0);
    }
  }

  float sq_sum = row_group_reduce_sum(sq, smem_acc, threads_per_row);
  float rstd = ptx_rsqrt(sq_sum / static_cast<float>(N) + eps);
  if (rstd_out != nullptr && row < M && row_tid == 0) {
    rstd_out[row] = rstd;
  }

#pragma unroll
  for (int iter = 0; iter < kMaxVecs; ++iter) {
    if (pred[iter]) {
      int elem = (iter * threads_per_row + row_tid) * kLanes;
      Vec128<float> weight_vec0;
      Vec128<float> weight_vec1;
      Vec128<float> bias_vec0;
      Vec128<float> bias_vec1;
      if (weight != nullptr) {
        weight_vec0.load(weight + affine_base + elem);
        if constexpr (kLanes > 4) {
          weight_vec1.load(weight + affine_base + elem + 4);
        }
      }
      if (bias != nullptr) {
        bias_vec0.load(bias + affine_base + elem);
        if constexpr (kLanes > 4) {
          bias_vec1.load(bias + affine_base + elem + 4);
        }
      }
      XVec out_vec;
      float out_values[kLanes];
      float residual_values[kLanes];
#pragma unroll
      for (int lane = 0; lane < kLanes; ++lane) {
        float xv = x_vec[iter].get(lane);
        if (residual != nullptr) {
          xv += res_vec[iter].get(lane);
        }
        if (residual_out != nullptr) {
          residual_values[lane] = xv;
        }
        float y = xv * rstd;
        if (weight != nullptr) {
          float wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
          y *= wv;
        }
        if (bias != nullptr) {
          float bv = lane < 4 ? bias_vec0.get(lane) : bias_vec1.get(lane - 4);
          y += bv;
        }
        out_values[lane] = y;
      }
      if (residual_out != nullptr) {
        if constexpr (std::is_same_v<X, RO>) {
          XVec residual_vec;
          set_vec_from_float_lanes(residual_vec, residual_values);
          residual_vec.store(residual_out + row_base + elem);
        } else {
          store_residual_lanes<RO, kLanes>(residual_out + row_base + elem,
                                           residual_values);
        }
      }
      set_vec_from_float_lanes(out_vec, out_values);
      out_vec.store(out + row_base + elem);
    }
  }
}

template <typename X, int kMaxVecs>
__global__ void rmsnorm_fwd_weight_cached_kernel(
    X* __restrict__ out, X const* __restrict__ x,
    float const* __restrict__ weight, int M, int N, int H, float eps) {
  __shared__ float smem_acc[33];
  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  int tid = threadIdx.x;
  int row = blockIdx.x;
  int head = H > 1 ? row % H : 0;
  int vecs = N / kLanes;
  int row_base = row * N;
  int affine_base = head * N;

  XVec x_vec[kMaxVecs];
  bool pred[kMaxVecs];
  float sq = 0.0f;

#pragma unroll
  for (int iter = 0; iter < kMaxVecs; ++iter) {
    int vec_idx = iter * blockDim.x + tid;
    bool valid = vec_idx < vecs;
    pred[iter] = valid;
    if (valid) {
      int elem = vec_idx * kLanes;
      x_vec[iter].load(x + row_base + elem);
#pragma unroll
      for (int lane = 0; lane < kLanes; ++lane) {
        float xv = x_vec[iter].get(lane);
        sq = fmaf(xv, xv, sq);
      }
    } else {
      x_vec[iter].bits = make_uint4(0, 0, 0, 0);
    }
  }

  float sq_sum = cta_reduce_sum(sq, smem_acc);
  float rstd = ptx_rsqrt(sq_sum / static_cast<float>(N) + eps);

#pragma unroll
  for (int iter = 0; iter < kMaxVecs; ++iter) {
    if (pred[iter]) {
      int elem = (iter * blockDim.x + tid) * kLanes;
      Vec128<float> weight_vec0;
      Vec128<float> weight_vec1;
      weight_vec0.load(weight + affine_base + elem);
      if constexpr (kLanes > 4) {
        weight_vec1.load(weight + affine_base + elem + 4);
      }
      XVec out_vec;
      float out_values[kLanes];
#pragma unroll
      for (int lane = 0; lane < kLanes; ++lane) {
        float xv = x_vec[iter].get(lane);
        float wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
        out_values[lane] = xv * rstd * wv;
      }
      set_vec_from_float_lanes(out_vec, out_values);
      out_vec.store(out + row_base + elem);
    }
  }
}

template <typename X, int kVecs, int kThreads>
__global__ __launch_bounds__(kThreads, 2) void rmsnorm_fwd_weight_cpasync_full_kernel(
    X* __restrict__ out, X const* __restrict__ x,
    float const* __restrict__ weight, int M, int N, int H, float eps) {
  extern __shared__ __align__(16) uint4 smem_vec[];
  X* smem_x = reinterpret_cast<X*>(smem_vec);
  __shared__ float smem_acc[33];
  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  int tid = threadIdx.x;
  int row = blockIdx.x;
  int head = H > 1 ? row % H : 0;
  int row_base = row * N;
  int affine_base = head * N;

  XVec x_vec[kVecs];
  float sq = 0.0f;

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int elem = (iter * kThreads + tid) * kLanes;
    cp_async_128(smem_x + elem, x + row_base + elem);
  }
  cp_async_commit_group();
  cp_async_wait_group_0();

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int elem = (iter * kThreads + tid) * kLanes;
    x_vec[iter].load_shared(smem_x + elem);
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec[iter].get(lane);
      sq = fmaf(xv, xv, sq);
    }
  }

  float sq_sum = cta_reduce_sum(sq, smem_acc);
  float rstd = ptx_rsqrt(sq_sum / static_cast<float>(N) + eps);

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int elem = (iter * kThreads + tid) * kLanes;
    Vec128<float> weight_vec0;
    Vec128<float> weight_vec1;
    weight_vec0.load(weight + affine_base + elem);
    if constexpr (kLanes > 4) {
      weight_vec1.load(weight + affine_base + elem + 4);
    }
    XVec out_vec;
    float out_values[kLanes];
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec[iter].get(lane);
      float wv;
      if constexpr (kLanes > 4) {
        wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
      } else {
        wv = weight_vec0.get(lane);
      }
      out_values[lane] = xv * rstd * wv;
    }
    set_vec_from_float_lanes(out_vec, out_values);
    out_vec.store(out + row_base + elem);
  }
}

template <typename X, int kVecs, bool kPreloadWeight = false,
          bool kAssumeH1 = false, int kLaunchThreads = 1024,
          int kMinBlocks = 1>
__global__ __launch_bounds__(kLaunchThreads, kMinBlocks) void rmsnorm_fwd_weight_full_kernel(
    X* __restrict__ out, X const* __restrict__ x,
    float const* __restrict__ weight, int M, int N, int H, float eps) {
  __shared__ float smem_acc[33];
  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  int tid = threadIdx.x;
  int row = blockIdx.x;
  int row_base = row * N;
  int affine_base = 0;
  if constexpr (!kAssumeH1) {
    int head = H > 1 ? row % H : 0;
    affine_base = head * N;
  }

  XVec x_vec[kVecs];
  Vec128<float> pre_weight_vec0[kPreloadWeight ? kVecs : 1];
  Vec128<float> pre_weight_vec1[kPreloadWeight ? kVecs : 1];
  float sq = 0.0f;

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int elem = (iter * blockDim.x + tid) * kLanes;
    x_vec[iter].load(x + row_base + elem);
    if constexpr (kPreloadWeight) {
      pre_weight_vec0[iter].load(weight + affine_base + elem);
      if constexpr (kLanes > 4) {
        pre_weight_vec1[iter].load(weight + affine_base + elem + 4);
      }
    }
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec[iter].get(lane);
      sq = fmaf(xv, xv, sq);
    }
  }

  float sq_sum = cta_reduce_sum(sq, smem_acc);
  float rstd = ptx_rsqrt(sq_sum / static_cast<float>(N) + eps);

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int elem = (iter * blockDim.x + tid) * kLanes;
    Vec128<float> weight_vec0;
    Vec128<float> weight_vec1;
    if constexpr (!kPreloadWeight) {
      weight_vec0.load(weight + affine_base + elem);
      if constexpr (kLanes > 4) {
        weight_vec1.load(weight + affine_base + elem + 4);
      }
    }
    XVec out_vec;
    float out_values[kLanes];
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec[iter].get(lane);
      float wv;
      if constexpr (kLanes > 4) {
        if constexpr (kPreloadWeight) {
          wv = lane < 4 ? pre_weight_vec0[iter].get(lane)
                        : pre_weight_vec1[iter].get(lane - 4);
        } else {
          wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
        }
      } else {
        if constexpr (kPreloadWeight) {
          wv = pre_weight_vec0[iter].get(lane);
        } else {
          wv = weight_vec0.get(lane);
        }
      }
      out_values[lane] = xv * rstd * wv;
    }
    set_vec_from_float_lanes(out_vec, out_values);
    out_vec.store(out + row_base + elem);
  }
}

template <typename X, int kVecs, bool kAssumeH1 = false>
__global__ void rmsnorm_fwd_weight_multirow_full_kernel(
    X* __restrict__ out, X const* __restrict__ x,
    float const* __restrict__ weight, int M, int N, int H, float eps) {
  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;
  constexpr int kThreadsPerRow = 32;

  int tid = threadIdx.x;
  int row_group = tid / kThreadsPerRow;
  int row_tid = tid - row_group * kThreadsPerRow;
  int rows_per_cta = blockDim.x / kThreadsPerRow;
  int row = blockIdx.x * rows_per_cta + row_group;
  if (row >= M) {
    return;
  }
  int row_base = row * N;
  int affine_base = 0;
  if constexpr (!kAssumeH1) {
    int head = H > 1 ? row % H : 0;
    affine_base = head * N;
  }

  XVec x_vec[kVecs];
  float sq = 0.0f;

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int elem = (iter * kThreadsPerRow + row_tid) * kLanes;
    x_vec[iter].load(x + row_base + elem);
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec[iter].get(lane);
      sq = fmaf(xv, xv, sq);
    }
  }

  float sq_sum = warp_reduce_sum(sq);
  float rstd = ptx_rsqrt(sq_sum / static_cast<float>(N) + eps);

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int elem = (iter * kThreadsPerRow + row_tid) * kLanes;
    Vec128<float> weight_vec0;
    Vec128<float> weight_vec1;
    XVec out_vec;
    weight_vec0.load(weight + affine_base + elem);
    if constexpr (kLanes > 4) {
      weight_vec1.load(weight + affine_base + elem + 4);
    }
    float out_values[kLanes];
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec[iter].get(lane);
      float wv;
      if constexpr (kLanes > 4) {
        wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
      } else {
        wv = weight_vec0.get(lane);
      }
      out_values[lane] = xv * rstd * wv;
    }
    set_vec_from_float_lanes(out_vec, out_values);
    out_vec.store(out + row_base + elem);
  }
}

template <typename X, typename RO, int kVecs, int kThreadsPerRow,
          bool kEarlyMixedResidualStore = false, bool kAssumeH1 = false,
          bool kPreloadWeight = false>
__global__ void rmsnorm_fwd_residual_multirow_full_kernel(
    X* __restrict__ out, RO* __restrict__ residual_out,
    X const* __restrict__ x, X const* __restrict__ residual,
    float const* __restrict__ weight, int M, int N, int H, float eps) {
  __shared__ float smem_acc[64];
  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  int tid = threadIdx.x;
  int row_group = tid / kThreadsPerRow;
  int row_tid = tid - row_group * kThreadsPerRow;
  int rows_per_cta = blockDim.x / kThreadsPerRow;
  int row = blockIdx.x * rows_per_cta + row_group;
  if (row >= M) {
    return;
  }
  int row_base = row * N;
  int affine_base = 0;
  if constexpr (!kAssumeH1) {
    int head = H > 1 ? row % H : 0;
    affine_base = head * N;
  }

  XVec x_vec[kVecs];
  XVec residual_vec[kVecs];
  Vec128<float> pre_weight_vec0[kPreloadWeight ? kVecs : 1];
  Vec128<float> pre_weight_vec1[kPreloadWeight ? kVecs : 1];
  float sq = 0.0f;

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int elem = (iter * kThreadsPerRow + row_tid) * kLanes;
    x_vec[iter].load(x + row_base + elem);
    residual_vec[iter].load(residual + row_base + elem);
    if constexpr (kPreloadWeight) {
      pre_weight_vec0[iter].load(weight + affine_base + elem);
      if constexpr (kLanes > 4) {
        pre_weight_vec1[iter].load(weight + affine_base + elem + 4);
      }
    }
    float residual_values[kLanes];
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec[iter].get(lane) + residual_vec[iter].get(lane);
      if constexpr (kEarlyMixedResidualStore && !std::is_same_v<X, RO>) {
        residual_values[lane] = xv;
      }
      sq = fmaf(xv, xv, sq);
    }
    if constexpr (kEarlyMixedResidualStore && !std::is_same_v<X, RO>) {
      store_residual_lanes<RO, kLanes>(residual_out + row_base + elem,
                                       residual_values);
    }
  }

  float sq_sum = row_group_reduce_sum(sq, smem_acc, kThreadsPerRow);
  float rstd = ptx_rsqrt(sq_sum / static_cast<float>(N) + eps);

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int elem = (iter * kThreadsPerRow + row_tid) * kLanes;
    Vec128<float> weight_vec0;
    Vec128<float> weight_vec1;
    XVec out_vec;
    float residual_values[kLanes];
    float out_values[kLanes];
    if constexpr (kPreloadWeight) {
      weight_vec0 = pre_weight_vec0[iter];
      if constexpr (kLanes > 4) {
        weight_vec1 = pre_weight_vec1[iter];
      }
    } else {
      weight_vec0.load(weight + affine_base + elem);
      if constexpr (kLanes > 4) {
        weight_vec1.load(weight + affine_base + elem + 4);
      }
    }
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec[iter].get(lane) + residual_vec[iter].get(lane);
      float wv;
      if constexpr (kLanes > 4) {
        wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
      } else {
        wv = weight_vec0.get(lane);
      }
      if constexpr (std::is_same_v<X, RO> || !kEarlyMixedResidualStore) {
        residual_values[lane] = xv;
      }
      out_values[lane] = xv * rstd * wv;
    }
    set_vec_from_float_lanes(out_vec, out_values);
    if constexpr (std::is_same_v<X, RO>) {
      XVec residual_out_vec;
      set_vec_from_float_lanes(residual_out_vec, residual_values);
      residual_out_vec.store(residual_out + row_base + elem);
      out_vec.store(out + row_base + elem);
    } else if constexpr (!kEarlyMixedResidualStore) {
      out_vec.store(out + row_base + elem);
      store_residual_lanes<RO, kLanes>(residual_out + row_base + elem,
                                       residual_values);
    } else {
      out_vec.store(out + row_base + elem);
    }
  }
}

template <typename X, typename RO, int kVecs, int kThreadsPerRow,
          bool kEarlyMixedResidualStore = false, bool kPreloadWeight = false>
void launch_fwd_residual_multirow_full(
    bool assume_h1, dim3 grid, dim3 block, cudaStream_t stream, X* out,
    RO* residual_out, X const* x, X const* residual, float const* weight,
    int M, int N, int H, float eps) {
  if (assume_h1) {
    rmsnorm_fwd_residual_multirow_full_kernel<
        X, RO, kVecs, kThreadsPerRow, kEarlyMixedResidualStore, true,
        kPreloadWeight>
        <<<grid, block, 0, stream>>>(out, residual_out, x, residual, weight,
                                     M, N, H, eps);
  } else {
    rmsnorm_fwd_residual_multirow_full_kernel<
        X, RO, kVecs, kThreadsPerRow, kEarlyMixedResidualStore, false,
        kPreloadWeight>
        <<<grid, block, 0, stream>>>(out, residual_out, x, residual, weight,
                                     M, N, H, eps);
  }
}

template <typename X, int kVecs, int kThreadsPerRow, bool kAssumeH1 = false>
__global__ void rmsnorm_fwd_residual_mixed_vec4_full_kernel(
    X* __restrict__ out, float* __restrict__ residual_out,
    X const* __restrict__ x, X const* __restrict__ residual,
    float const* __restrict__ weight, int M, int N, int H, float eps) {
  __shared__ float smem_acc[64];
  using XVec = Vec64<X>;
  constexpr int kLanes = XVec::kLanes;

  int tid = threadIdx.x;
  int row_group = tid / kThreadsPerRow;
  int row_tid = tid - row_group * kThreadsPerRow;
  int rows_per_cta = blockDim.x / kThreadsPerRow;
  int row = blockIdx.x * rows_per_cta + row_group;
  if (row >= M) {
    return;
  }
  int row_base = row * N;
  int affine_base = 0;
  if constexpr (!kAssumeH1) {
    int head = H > 1 ? row % H : 0;
    affine_base = head * N;
  }

  XVec x_vec[kVecs];
  XVec residual_vec[kVecs];
  float sq = 0.0f;

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int elem = (iter * kThreadsPerRow + row_tid) * kLanes;
    x_vec[iter].load(x + row_base + elem);
    residual_vec[iter].load(residual + row_base + elem);
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec[iter].get(lane) + residual_vec[iter].get(lane);
      sq = fmaf(xv, xv, sq);
    }
  }

  float sq_sum = row_group_reduce_sum(sq, smem_acc, kThreadsPerRow);
  float rstd = ptx_rsqrt(sq_sum / static_cast<float>(N) + eps);

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int elem = (iter * kThreadsPerRow + row_tid) * kLanes;
    Vec128<float> weight_vec;
    XVec out_vec;
    out_vec.bits = make_uint2(0, 0);
    float residual_values[kLanes];
    weight_vec.load(weight + affine_base + elem);
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec[iter].get(lane) + residual_vec[iter].get(lane);
      residual_values[lane] = xv;
      out_vec.set(lane, xv * rstd * weight_vec.get(lane));
    }
    out_vec.store(out + row_base + elem);
    store_float_lanes<kLanes>(residual_out + row_base + elem, residual_values);
  }
}

template <typename X, int kVecs, int kThreadsPerRow>
void launch_fwd_residual_mixed_vec4(
    bool assume_h1, dim3 grid, dim3 block, cudaStream_t stream, X* out,
    float* residual_out, X const* x, X const* residual, float const* weight,
    int M, int N, int H, float eps) {
  if (assume_h1) {
    rmsnorm_fwd_residual_mixed_vec4_full_kernel<X, kVecs, kThreadsPerRow, true>
        <<<grid, block, 0, stream>>>(out, residual_out, x, residual, weight,
                                     M, N, H, eps);
  } else {
    rmsnorm_fwd_residual_mixed_vec4_full_kernel<X, kVecs, kThreadsPerRow>
        <<<grid, block, 0, stream>>>(out, residual_out, x, residual, weight,
                                     M, N, H, eps);
  }
}

template <typename X, typename RO>
__global__ void rmsnorm_fwd_streaming_kernel(
    X* __restrict__ out, RO* __restrict__ residual_out,
    float* __restrict__ rstd_out, X const* __restrict__ x,
    float const* __restrict__ weight, float const* __restrict__ bias,
    X const* __restrict__ residual, int M, int N, int H, int threads_per_row,
    float eps) {
  __shared__ float smem_acc[64];
  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  int tid = threadIdx.x;
  int row_group = tid / threads_per_row;
  int row_tid = tid - row_group * threads_per_row;
  int rows_per_cta = blockDim.x / threads_per_row;
  int row = blockIdx.x * rows_per_cta + row_group;
  int head = H > 1 ? row % H : 0;
  int vecs = N / kLanes;
  int row_base = row * N;
  int affine_base = head * N;

  float sq = 0.0f;
  for (int vec_idx = row_tid; row < M && vec_idx < vecs; vec_idx += threads_per_row) {
    int elem = vec_idx * kLanes;
    XVec x_vec;
    XVec res_vec;
    x_vec.load_cg(x + row_base + elem);
    if (residual != nullptr) {
      res_vec.load(residual + row_base + elem);
    }
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec.get(lane);
      if (residual != nullptr) {
        xv += res_vec.get(lane);
      }
      sq = fmaf(xv, xv, sq);
    }
  }

  float sq_sum = row_group_reduce_sum(sq, smem_acc, threads_per_row);
  float rstd = ptx_rsqrt(sq_sum / static_cast<float>(N) + eps);
  if (rstd_out != nullptr && row < M && row_tid == 0) {
    rstd_out[row] = rstd;
  }

  for (int vec_idx = row_tid; row < M && vec_idx < vecs; vec_idx += threads_per_row) {
    int elem = vec_idx * kLanes;
    XVec x_vec;
    XVec res_vec;
    XVec out_vec;
    Vec128<float> weight_vec0;
    Vec128<float> weight_vec1;
    Vec128<float> bias_vec0;
    Vec128<float> bias_vec1;
    x_vec.load_cg(x + row_base + elem);
    if (residual != nullptr) {
      res_vec.load(residual + row_base + elem);
    }
    if (weight != nullptr) {
      weight_vec0.load(weight + affine_base + elem);
      if constexpr (kLanes > 4) {
        weight_vec1.load(weight + affine_base + elem + 4);
      }
    }
    if (bias != nullptr) {
      bias_vec0.load(bias + affine_base + elem);
      if constexpr (kLanes > 4) {
        bias_vec1.load(bias + affine_base + elem + 4);
      }
    }
    float out_values[kLanes];
    float residual_values[kLanes];
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec.get(lane);
      if (residual != nullptr) {
        xv += res_vec.get(lane);
      }
      if (residual_out != nullptr) {
        residual_values[lane] = xv;
      }
      float y = xv * rstd;
      if (weight != nullptr) {
        float wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
        y *= wv;
      }
      if (bias != nullptr) {
        float bv = lane < 4 ? bias_vec0.get(lane) : bias_vec1.get(lane - 4);
        y += bv;
      }
      out_values[lane] = y;
    }
    if (residual_out != nullptr) {
      if constexpr (std::is_same_v<X, RO>) {
        XVec residual_vec;
        set_vec_from_float_lanes(residual_vec, residual_values);
        residual_vec.store(residual_out + row_base + elem);
      } else {
        store_residual_lanes<RO, kLanes>(residual_out + row_base + elem,
                                         residual_values);
      }
    }
    set_vec_from_float_lanes(out_vec, out_values);
    out_vec.store(out + row_base + elem);
  }
}

template <typename X>
__global__ void rmsnorm_fwd_split_sums_kernel(
    float* __restrict__ partial_sums, X const* __restrict__ x, int M, int N,
    int parts) {
  __shared__ float smem_acc[33];
  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  int tid = threadIdx.x;
  int row = blockIdx.x;
  int part = blockIdx.y;
  int vecs = N / kLanes;
  int vecs_per_part = (vecs + parts - 1) / parts;
  int begin = part * vecs_per_part;
  int end = min(vecs, begin + vecs_per_part);
  int row_base = row * N;

  float sq = 0.0f;
  for (int vec_idx = begin + tid; vec_idx < end; vec_idx += blockDim.x) {
    int elem = vec_idx * kLanes;
    XVec x_vec;
    x_vec.load(x + row_base + elem);
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec.get(lane);
      sq = fmaf(xv, xv, sq);
    }
  }
  float sq_sum = cta_reduce_sum(sq, smem_acc);
  if (tid == 0) {
    partial_sums[row * parts + part] = sq_sum;
  }
}

template <typename X>
__global__ void rmsnorm_fwd_split_output_kernel(
    X* __restrict__ out, X const* __restrict__ x,
    float const* __restrict__ weight, float const* __restrict__ partial_sums,
    int M, int N, int H, int parts, float eps) {
  __shared__ float smem_rstd;
  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  int tid = threadIdx.x;
  int row = blockIdx.x;
  int part = blockIdx.y;
  int head = H > 1 ? row % H : 0;
  int vecs = N / kLanes;
  int vecs_per_part = (vecs + parts - 1) / parts;
  int begin = part * vecs_per_part;
  int end = min(vecs, begin + vecs_per_part);
  int row_base = row * N;
  int affine_base = head * N;

  if (tid == 0) {
    float sq_sum = 0.0f;
#pragma unroll
    for (int p = 0; p < 16; ++p) {
      if (p < parts) {
        sq_sum += partial_sums[row * parts + p];
      }
    }
    smem_rstd = ptx_rsqrt(sq_sum / static_cast<float>(N) + eps);
  }
  __syncthreads();
  float rstd = smem_rstd;

  for (int vec_idx = begin + tid; vec_idx < end; vec_idx += blockDim.x) {
    int elem = vec_idx * kLanes;
    XVec x_vec;
    XVec out_vec;
    Vec128<float> weight_vec0;
    Vec128<float> weight_vec1;
    x_vec.load(x + row_base + elem);
    weight_vec0.load(weight + affine_base + elem);
    if constexpr (kLanes > 4) {
      weight_vec1.load(weight + affine_base + elem + 4);
    }
    float out_values[kLanes];
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec.get(lane);
      float wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
      out_values[lane] = xv * rstd * wv;
    }
    set_vec_from_float_lanes(out_vec, out_values);
    out_vec.store(out + row_base + elem);
  }
}

template <typename X>
__global__ void rmsnorm_fwd_cluster_kernel(
    X* __restrict__ out, X const* __restrict__ x,
    float const* __restrict__ weight, int M, int N, int H, int parts,
    float eps) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  X* smem_x = reinterpret_cast<X*>(smem_raw);
  __shared__ float smem_acc[33];
  __shared__ float smem_sum;
  __shared__ float smem_rstd;

  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  cg::cluster_group cluster = cg::this_cluster();
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int part = blockIdx.y % parts;
  int head = H > 1 ? row % H : 0;
  int vecs = N / kLanes;
  int vecs_per_part = (vecs + parts - 1) / parts;
  int begin_vec = part * vecs_per_part;
  int end_vec = min(vecs, begin_vec + vecs_per_part);
  int row_base = row * N;
  int affine_base = head * N;

  float sq = 0.0f;
  for (int vec_idx = begin_vec + tid; vec_idx < end_vec; vec_idx += blockDim.x) {
    int elem = vec_idx * kLanes;
    XVec x_vec;
    x_vec.load(x + row_base + elem);
    x_vec.store_shared(smem_x + elem - begin_vec * kLanes);
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec.get(lane);
      sq = fmaf(xv, xv, sq);
    }
  }
  float local_sum = cta_reduce_sum_leader(sq, smem_acc);
  if (tid == 0) {
    smem_sum = local_sum;
  }
  cluster.sync();

  float total_sum = 0.0f;
  if (tid == 0) {
#pragma unroll
    for (int p = 0; p < 16; ++p) {
      if (p < parts) {
        float* remote_sum = cluster.map_shared_rank(&smem_sum, p);
        total_sum += *remote_sum;
      }
    }
    smem_rstd = ptx_rsqrt(total_sum / static_cast<float>(N) + eps);
  }
  cluster.sync();
  float rstd = smem_rstd;

  for (int vec_idx = begin_vec + tid; vec_idx < end_vec; vec_idx += blockDim.x) {
    int elem = vec_idx * kLanes;
    XVec x_vec;
    XVec out_vec;
    Vec128<float> weight_vec0;
    Vec128<float> weight_vec1;
    x_vec.load_shared(smem_x + elem - begin_vec * kLanes);
    weight_vec0.load(weight + affine_base + elem);
    if constexpr (kLanes > 4) {
      weight_vec1.load(weight + affine_base + elem + 4);
    }
    float out_values[kLanes];
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec.get(lane);
      float wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
      out_values[lane] = xv * rstd * wv;
    }
    set_vec_from_float_lanes(out_vec, out_values);
    out_vec.store(out + row_base + elem);
  }
  cluster.sync();
}

template <typename X, int kParts, int kVecs>
__global__ void rmsnorm_fwd_cluster_full_kernel(
    X* __restrict__ out, X const* __restrict__ x,
    float const* __restrict__ weight, int M, int N, int H, float eps) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  X* smem_x = reinterpret_cast<X*>(smem_raw);
  __shared__ float smem_acc[33];
  __shared__ float smem_sum;
  __shared__ float smem_rstd;

  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  cg::cluster_group cluster = cg::this_cluster();
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int part = blockIdx.y % kParts;
  int head = H > 1 ? row % H : 0;
  int begin_vec = part * kVecs * blockDim.x;
  int row_base = row * N;
  int affine_base = head * N;

  float sq = 0.0f;
#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    int local_elem = local_vec * kLanes;
    XVec x_vec;
    x_vec.load(x + row_base + elem);
    x_vec.store_shared(smem_x + local_elem);
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec.get(lane);
      sq = fmaf(xv, xv, sq);
    }
  }

  float local_sum = cta_reduce_sum_leader(sq, smem_acc);
  if (tid == 0) {
    smem_sum = local_sum;
  }
  cluster.sync();

  if (tid == 0) {
    float total_sum = 0.0f;
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      float* remote_sum = cluster.map_shared_rank(&smem_sum, p);
      total_sum += *remote_sum;
    }
    smem_rstd = ptx_rsqrt(total_sum / static_cast<float>(N) + eps);
  }
  __syncthreads();
  float rstd = smem_rstd;

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    int local_elem = local_vec * kLanes;
    XVec x_vec;
    Vec128<float> weight_vec0;
    Vec128<float> weight_vec1;
    x_vec.load_shared(smem_x + local_elem);
    weight_vec0.load(weight + affine_base + elem);
    if constexpr (kLanes > 4) {
      weight_vec1.load(weight + affine_base + elem + 4);
    }
    XVec out_vec;
    float out_values[kLanes];
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec.get(lane);
      float wv;
      if constexpr (kLanes > 4) {
        wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
      } else {
        wv = weight_vec0.get(lane);
      }
      out_values[lane] = xv * rstd * wv;
    }
    set_vec_from_float_lanes(out_vec, out_values);
    out_vec.store(out + row_base + elem);
  }
  if constexpr (kParts > 8) {
    cluster.sync();
  }
}

template <typename X, int kParts, int kVecs, bool kAssumeH1 = false,
          bool kPrefetchTid0 = false>
__global__ void rmsnorm_fwd_cluster_reg_full_kernel(
    X* __restrict__ out, X const* __restrict__ x,
    float const* __restrict__ weight, int M, int N, int H, float eps) {
  __shared__ float smem_acc[33];
  __shared__ float smem_reduce[16];
  __shared__ float smem_rstd;
  __shared__ uint64_t smem_sum_mbar;

  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  cg::cluster_group cluster = cg::this_cluster();
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int part = blockIdx.y % kParts;
  int begin_vec = part * kVecs * blockDim.x;
  int row_base = row * N;
  int affine_base = 0;
  if constexpr (!kAssumeH1) {
    int head = H > 1 ? row % H : 0;
    affine_base = head * N;
  }

  if (tid == 0) {
    mbarrier_init(&smem_sum_mbar, 1);
  }
  mbarrier_init_fence();
  cluster_arrive_relaxed_barrier();

  XVec x_vec[kVecs];
  float sq = 0.0f;
#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    x_vec[iter].load(x + row_base + elem);
    sq = accumulate_sq_vec(x_vec[iter], sq);
  }

  float local_sum = cta_reduce_sum_leader(sq, smem_acc);
  cluster_wait_barrier();

  constexpr int kWeightPrefetchVecs =
      (sizeof(X) >= 4 && kVecs >= kRegClusterWeightPrefetchVecs)
          ? kRegClusterWeightPrefetchVecs
          : 1;
  Vec128<float> weight_pref0[kWeightPrefetchVecs];
  Vec128<float> weight_pref1[kWeightPrefetchVecs];
  if constexpr (kPrefetchTid0) {
#pragma unroll
    for (int pref = 0; pref < kWeightPrefetchVecs; ++pref) {
      int elem = (begin_vec + pref * blockDim.x + tid) * kLanes;
      weight_pref0[pref].load(weight + affine_base + elem);
      if constexpr (kLanes > 4) {
        weight_pref1[pref].load(weight + affine_base + elem + 4);
      }
    }
  } else if (tid != 0) {
#pragma unroll
    for (int pref = 0; pref < kWeightPrefetchVecs; ++pref) {
      int elem = (begin_vec + pref * blockDim.x + tid) * kLanes;
      weight_pref0[pref].load(weight + affine_base + elem);
      if constexpr (kLanes > 4) {
        weight_pref1[pref].load(weight + affine_base + elem + 4);
      }
    }
  }

  if (tid == 0) {
    int rank = static_cast<int>(cluster_rank());
    mbarrier_arrive_expect_tx(&smem_sum_mbar,
                              static_cast<uint32_t>(kParts * sizeof(float)));
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      store_shared_remote_float(local_sum, smem_reduce + rank, &smem_sum_mbar,
                                static_cast<uint32_t>(p));
    }
    mbarrier_wait(&smem_sum_mbar, 0);
    float total_sum = 0.0f;
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      total_sum += smem_reduce[p];
    }
    smem_rstd = ptx_rsqrt(total_sum / static_cast<float>(N) + eps);
  }
  __syncthreads();
  float rstd = smem_rstd;

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    Vec128<float> weight_vec0;
    Vec128<float> weight_vec1;
    if (iter < kWeightPrefetchVecs && (kPrefetchTid0 || tid != 0)) {
      weight_vec0 = weight_pref0[iter];
      if constexpr (kLanes > 4) {
        weight_vec1 = weight_pref1[iter];
      }
    } else {
      weight_vec0.load(weight + affine_base + elem);
      if constexpr (kLanes > 4) {
        weight_vec1.load(weight + affine_base + elem + 4);
      }
    }
    XVec out_vec;
    float out_values[kLanes];
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec[iter].get(lane);
      float wv;
      if constexpr (kLanes > 4) {
        wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
      } else {
        wv = weight_vec0.get(lane);
      }
      out_values[lane] = xv * rstd * wv;
    }
    set_vec_from_float_lanes(out_vec, out_values);
    out_vec.store(out + row_base + elem);
  }
  if constexpr (kParts > 16) {
    cluster.sync();
  }
}

template <typename X, typename RO, int kParts, int kVecs,
          bool kAssumeH1 = false, bool kUsePackedMixedAdd = false>
__global__ void rmsnorm_fwd_cluster_residual_reg_full_kernel(
    X* __restrict__ out, RO* __restrict__ residual_out,
    X const* __restrict__ x, X const* __restrict__ residual,
    float const* __restrict__ weight, int M, int N, int H, float eps) {
  __shared__ float smem_acc[33];
  __shared__ float smem_reduce[16];
  __shared__ float smem_rstd;
  __shared__ uint64_t smem_sum_mbar;

  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  cg::cluster_group cluster = cg::this_cluster();
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int part = blockIdx.y % kParts;
  int begin_vec = part * kVecs * blockDim.x;
  int row_base = row * N;
  int affine_base = 0;
  if constexpr (!kAssumeH1) {
    int head = H > 1 ? row % H : 0;
    affine_base = head * N;
  }

  if (tid == 0) {
    mbarrier_init(&smem_sum_mbar, 1);
  }
  mbarrier_init_fence();
  cluster_arrive_relaxed_barrier();

  XVec x_vec[kVecs];
  XVec residual_vec[kVecs];
  float sq = 0.0f;
#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    x_vec[iter].load(x + row_base + elem);
    residual_vec[iter].load(residual + row_base + elem);
    float residual_values[kLanes];
    if constexpr (kUsePackedMixedAdd) {
      add_vec_lanes<X, kLanes>(x_vec[iter], residual_vec[iter],
                               residual_values);
    }
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv;
      if constexpr (kUsePackedMixedAdd) {
        xv = residual_values[lane];
      } else {
        xv = x_vec[iter].get(lane) + residual_vec[iter].get(lane);
      }
      sq = fmaf(xv, xv, sq);
    }
  }

  float local_sum = cta_reduce_sum_leader(sq, smem_acc);
  cluster_wait_barrier();

  constexpr int kResidualHalfWeightPrefetchVecs =
      std::is_same_v<X, __half> ? kResidualClusterFp16WeightPrefetchVecs
                                : kResidualClusterBf16WeightPrefetchVecs;
  constexpr bool kPrefetchHalfWeights =
      sizeof(X) < 4 && std::is_same_v<X, RO> &&
      kVecs >= kResidualHalfWeightPrefetchVecs;
  constexpr bool kPrefetchFp32Weights =
      sizeof(X) >= 4 && kVecs >= kRegClusterWeightPrefetchVecs;
  constexpr int kWeightPrefetchVecs =
      kPrefetchFp32Weights
          ? kRegClusterWeightPrefetchVecs
          : (kPrefetchHalfWeights ? kResidualHalfWeightPrefetchVecs : 1);
  Vec128<float> weight_pref0[kWeightPrefetchVecs];
  Vec128<float> weight_pref1[kWeightPrefetchVecs];
  if constexpr (kPrefetchHalfWeights || kPrefetchFp32Weights) {
    if (tid != 0) {
#pragma unroll
      for (int pref = 0; pref < kWeightPrefetchVecs; ++pref) {
        int elem = (begin_vec + pref * blockDim.x + tid) * kLanes;
        weight_pref0[pref].load(weight + affine_base + elem);
        if constexpr (kLanes > 4) {
          weight_pref1[pref].load(weight + affine_base + elem + 4);
        }
      }
    }
  }

  if (tid == 0) {
    int rank = static_cast<int>(cluster_rank());
    mbarrier_arrive_expect_tx(&smem_sum_mbar,
                              static_cast<uint32_t>(kParts * sizeof(float)));
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      store_shared_remote_float(local_sum, smem_reduce + rank, &smem_sum_mbar,
                                static_cast<uint32_t>(p));
    }
    mbarrier_wait(&smem_sum_mbar, 0);
    float total_sum = 0.0f;
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      total_sum += smem_reduce[p];
    }
    smem_rstd = ptx_rsqrt(total_sum / static_cast<float>(N) + eps);
  }
  __syncthreads();
  float rstd = smem_rstd;

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    Vec128<float> weight_vec0;
    Vec128<float> weight_vec1;
    if constexpr (kPrefetchHalfWeights || kPrefetchFp32Weights) {
      if (iter < kWeightPrefetchVecs && tid != 0) {
        weight_vec0 = weight_pref0[iter];
        if constexpr (kLanes > 4) {
          weight_vec1 = weight_pref1[iter];
        }
      } else {
        weight_vec0.load(weight + affine_base + elem);
        if constexpr (kLanes > 4) {
          weight_vec1.load(weight + affine_base + elem + 4);
        }
      }
    } else {
      weight_vec0.load(weight + affine_base + elem);
      if constexpr (kLanes > 4) {
        weight_vec1.load(weight + affine_base + elem + 4);
      }
    }
    XVec out_vec;
    float residual_values[kLanes];
    float out_values[kLanes];
    if constexpr (kUsePackedMixedAdd) {
      add_vec_lanes<X, kLanes>(x_vec[iter], residual_vec[iter],
                               residual_values);
    }
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv;
      if constexpr (kUsePackedMixedAdd) {
        xv = residual_values[lane];
      } else {
        xv = x_vec[iter].get(lane) + residual_vec[iter].get(lane);
        residual_values[lane] = xv;
      }
      float wv;
      if constexpr (kLanes > 4) {
        wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
      } else {
        wv = weight_vec0.get(lane);
      }
      out_values[lane] = xv * rstd * wv;
    }
    set_vec_from_float_lanes(out_vec, out_values);
    if constexpr (std::is_same_v<X, RO>) {
      XVec residual_out_vec;
      set_vec_from_float_lanes(residual_out_vec, residual_values);
      residual_out_vec.store(residual_out + row_base + elem);
      out_vec.store(out + row_base + elem);
    } else {
      // For mixed fp32 residual_out, issuing the narrower normalized output
      // store first keeps the large-row clustered path much closer to QuACK.
      out_vec.store(out + row_base + elem);
      store_residual_lanes<RO, kLanes>(residual_out + row_base + elem,
                                       residual_values);
    }
  }
  if constexpr (kParts > 16) {
    cluster.sync();
  }
}

template <typename X, typename RO, int kParts, int kVecs,
          bool kAssumeH1 = false, bool kUsePackedMixedAdd = false,
          bool kUseCgStagingLoad = false>
__global__ void rmsnorm_fwd_cluster_residual_smem_full_kernel(
    X* __restrict__ out, RO* __restrict__ residual_out,
    X const* __restrict__ x, X const* __restrict__ residual,
    float const* __restrict__ weight, int M, int N, int H, float eps) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  X* smem_x = reinterpret_cast<X*>(smem_raw);
  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;
  X* smem_residual = smem_x + kVecs * blockDim.x * kLanes;
  __shared__ float smem_acc[33];
  __shared__ float smem_reduce[16];
  __shared__ float smem_rstd;
  __shared__ uint64_t smem_sum_mbar;

  cg::cluster_group cluster = cg::this_cluster();
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int part = blockIdx.y % kParts;
  int begin_vec = part * kVecs * blockDim.x;
  int row_base = row * N;
  int affine_base = 0;
  if constexpr (!kAssumeH1) {
    int head = H > 1 ? row % H : 0;
    affine_base = head * N;
  }

  if (tid == 0) {
    mbarrier_init(&smem_sum_mbar, 1);
  }
  mbarrier_init_fence();
  cluster_arrive_relaxed_barrier();

  float sq = 0.0f;
#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    int local_elem = local_vec * kLanes;
    XVec x_vec;
    XVec residual_vec;
    if constexpr (kUseCgStagingLoad) {
      x_vec.load_cg(x + row_base + elem);
      residual_vec.load_cg(residual + row_base + elem);
    } else {
      x_vec.load(x + row_base + elem);
      residual_vec.load(residual + row_base + elem);
    }
    x_vec.store_shared(smem_x + local_elem);
    residual_vec.store_shared(smem_residual + local_elem);
    float residual_values[kLanes];
    if constexpr (kUsePackedMixedAdd) {
      add_vec_lanes<X, kLanes>(x_vec, residual_vec, residual_values);
    }
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv;
      if constexpr (kUsePackedMixedAdd) {
        xv = residual_values[lane];
      } else {
        xv = x_vec.get(lane) + residual_vec.get(lane);
        residual_values[lane] = xv;
      }
      sq = fmaf(xv, xv, sq);
    }
  }

  float local_sum = cta_reduce_sum_leader(sq, smem_acc);
  cluster_wait_barrier();

  if (tid == 0) {
    int rank = static_cast<int>(cluster_rank());
    mbarrier_arrive_expect_tx(&smem_sum_mbar,
                              static_cast<uint32_t>(kParts * sizeof(float)));
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      store_shared_remote_float(local_sum, smem_reduce + rank, &smem_sum_mbar,
                                static_cast<uint32_t>(p));
    }
    mbarrier_wait(&smem_sum_mbar, 0);
    float total_sum = 0.0f;
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      total_sum += smem_reduce[p];
    }
    smem_rstd = ptx_rsqrt(total_sum / static_cast<float>(N) + eps);
  }
  __syncthreads();
  float rstd = smem_rstd;

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    int local_elem = local_vec * kLanes;
    XVec x_vec;
    XVec residual_vec;
    Vec128<float> weight_vec0;
    Vec128<float> weight_vec1;
    x_vec.load_shared(smem_x + local_elem);
    residual_vec.load_shared(smem_residual + local_elem);
    weight_vec0.load(weight + affine_base + elem);
    if constexpr (kLanes > 4) {
      weight_vec1.load(weight + affine_base + elem + 4);
    }
    XVec out_vec;
    float residual_values[kLanes];
    float out_values[kLanes];
    if constexpr (kUsePackedMixedAdd) {
      add_vec_lanes<X, kLanes>(x_vec, residual_vec, residual_values);
    }
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv;
      if constexpr (kUsePackedMixedAdd) {
        xv = residual_values[lane];
      } else {
        xv = x_vec.get(lane) + residual_vec.get(lane);
        residual_values[lane] = xv;
      }
      float wv;
      if constexpr (kLanes > 4) {
        wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
      } else {
        wv = weight_vec0.get(lane);
      }
      out_values[lane] = xv * rstd * wv;
    }
    set_vec_from_float_lanes(out_vec, out_values);
    out_vec.store(out + row_base + elem);
    store_residual_lanes<RO, kLanes>(residual_out + row_base + elem,
                                     residual_values);
  }
  if constexpr (kParts > 16) {
    cluster.sync();
  }
}

template <typename X, int kParts, int kVecs, bool kAssumeH1 = false>
__global__ void rmsnorm_fwd_cluster_mbar_full_kernel(
    X* __restrict__ out, X const* __restrict__ x,
    float const* __restrict__ weight, int M, int N, int H, float eps) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  X* smem_x = reinterpret_cast<X*>(smem_raw);
  __shared__ float smem_acc[33];
  __shared__ float smem_reduce[16];
  __shared__ float smem_rstd;
  __shared__ uint64_t smem_sum_mbar;

  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  cg::cluster_group cluster = cg::this_cluster();
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int part = blockIdx.y % kParts;
  int begin_vec = part * kVecs * blockDim.x;
  int row_base = row * N;
  int affine_base = 0;
  if constexpr (!kAssumeH1) {
    int head = H > 1 ? row % H : 0;
    affine_base = head * N;
  }

  if (tid == 0) {
    mbarrier_init(&smem_sum_mbar, 1);
  }
  mbarrier_init_fence();
  cluster_arrive_relaxed_barrier();

  float sq = 0.0f;
#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    int local_elem = local_vec * kLanes;
    XVec x_vec;
    x_vec.load(x + row_base + elem);
    x_vec.store_shared(smem_x + local_elem);
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec.get(lane);
      sq = fmaf(xv, xv, sq);
    }
  }

  float local_sum = cta_reduce_sum_leader(sq, smem_acc);
  cluster_wait_barrier();

  if (tid == 0) {
    int rank = static_cast<int>(cluster_rank());
    mbarrier_arrive_expect_tx(&smem_sum_mbar,
                              static_cast<uint32_t>(kParts * sizeof(float)));
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      store_shared_remote_float(local_sum, smem_reduce + rank, &smem_sum_mbar,
                                static_cast<uint32_t>(p));
    }
    mbarrier_wait(&smem_sum_mbar, 0);
    float total_sum = 0.0f;
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      total_sum += smem_reduce[p];
    }
    smem_rstd = ptx_rsqrt(total_sum / static_cast<float>(N) + eps);
  }
  __syncthreads();
  float rstd = smem_rstd;

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    int local_elem = local_vec * kLanes;
    XVec x_vec;
    Vec128<float> weight_vec0;
    Vec128<float> weight_vec1;
    x_vec.load_shared(smem_x + local_elem);
    weight_vec0.load(weight + affine_base + elem);
    if constexpr (kLanes > 4) {
      weight_vec1.load(weight + affine_base + elem + 4);
    }
    XVec out_vec;
    float out_values[kLanes];
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec.get(lane);
      float wv;
      if constexpr (kLanes > 4) {
        wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
      } else {
        wv = weight_vec0.get(lane);
      }
      out_values[lane] = xv * rstd * wv;
    }
    set_vec_from_float_lanes(out_vec, out_values);
    out_vec.store(out + row_base + elem);
  }
  if constexpr (kParts > 16) {
    cluster.sync();
  }
}

template <typename X, int kParts, int kVecs, bool kAssumeH1 = false>
__global__ void rmsnorm_fwd_cluster_mbar_cpasync_full_kernel(
    X* __restrict__ out, X const* __restrict__ x,
    float const* __restrict__ weight, int M, int N, int H, float eps) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  X* smem_x = reinterpret_cast<X*>(smem_raw);
  __shared__ float smem_acc[33];
  __shared__ float smem_reduce[16];
  __shared__ float smem_rstd;
  __shared__ uint64_t smem_sum_mbar;

  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  int tid = threadIdx.x;
  int row = blockIdx.x;
  int part = blockIdx.y % kParts;
  int begin_vec = part * kVecs * blockDim.x;
  int row_base = row * N;
  int affine_base = 0;
  if constexpr (!kAssumeH1) {
    int head = H > 1 ? row % H : 0;
    affine_base = head * N;
  }

  if (tid == 0) {
    mbarrier_init(&smem_sum_mbar, 1);
  }
  mbarrier_init_fence();
  cluster_arrive_relaxed_barrier();

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    int local_elem = local_vec * kLanes;
    cp_async_128(smem_x + local_elem, x + row_base + elem);
  }
  cp_async_commit_group();
  cp_async_wait_group_0();

  float sq = 0.0f;
#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int local_elem = local_vec * kLanes;
    XVec x_vec;
    x_vec.load_shared(smem_x + local_elem);
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec.get(lane);
      sq = fmaf(xv, xv, sq);
    }
  }

  float local_sum = cta_reduce_sum_leader(sq, smem_acc);
  cluster_wait_barrier();

  if (tid == 0) {
    int rank = static_cast<int>(cluster_rank());
    mbarrier_arrive_expect_tx(&smem_sum_mbar,
                              static_cast<uint32_t>(kParts * sizeof(float)));
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      store_shared_remote_float(local_sum, smem_reduce + rank, &smem_sum_mbar,
                                static_cast<uint32_t>(p));
    }
    mbarrier_wait(&smem_sum_mbar, 0);
    float total_sum = 0.0f;
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      total_sum += smem_reduce[p];
    }
    smem_rstd = ptx_rsqrt(total_sum / static_cast<float>(N) + eps);
  }
  __syncthreads();
  float rstd = smem_rstd;

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    int local_elem = local_vec * kLanes;
    XVec x_vec;
    Vec128<float> weight_vec0;
    Vec128<float> weight_vec1;
    x_vec.load_shared(smem_x + local_elem);
    weight_vec0.load(weight + affine_base + elem);
    if constexpr (kLanes > 4) {
      weight_vec1.load(weight + affine_base + elem + 4);
    }
    XVec out_vec;
    float out_values[kLanes];
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec.get(lane);
      float wv;
      if constexpr (kLanes > 4) {
        wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
      } else {
        wv = weight_vec0.get(lane);
      }
      out_values[lane] = xv * rstd * wv;
    }
    set_vec_from_float_lanes(out_vec, out_values);
    out_vec.store(out + row_base + elem);
  }
}

template <typename X, int kParts, int kVecs, bool kAssumeH1 = false>
__device__ __forceinline__ void rmsnorm_fwd_cluster_mbar_cpasync_weight_impl(
    X* __restrict__ out, X const* __restrict__ x,
    float const* __restrict__ weight, int M, int N, int H, float eps,
    unsigned char* smem_raw, float* smem_acc, float* smem_reduce,
    float* smem_rstd, uint64_t* smem_sum_mbar) {
  X* smem_x = reinterpret_cast<X*>(smem_raw);

  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  int tid = threadIdx.x;
  int row = blockIdx.x;
  int part = blockIdx.y % kParts;
  int begin_vec = part * kVecs * blockDim.x;
  int row_base = row * N;
  int affine_base = 0;
  if constexpr (!kAssumeH1) {
    int head = H > 1 ? row % H : 0;
    affine_base = head * N;
  }

  if (tid == 0) {
    mbarrier_init(smem_sum_mbar, 1);
  }
  mbarrier_init_fence();
  cluster_arrive_relaxed_barrier();

  Vec128<float> weight_pref0[kVecs];
  Vec128<float> weight_pref1[kLanes > 4 ? kVecs : 1];
#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    int local_elem = local_vec * kLanes;
    cp_async_128(smem_x + local_elem, x + row_base + elem);
  }
  cp_async_commit_group();

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    weight_pref0[iter].load(weight + affine_base + elem);
    if constexpr (kLanes > 4) {
      weight_pref1[iter].load(weight + affine_base + elem + 4);
    }
  }

  cp_async_wait_group_0();

  float sq = 0.0f;
#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int local_elem = local_vec * kLanes;
    XVec x_vec;
    x_vec.load_shared(smem_x + local_elem);
    sq = accumulate_sq_vec(x_vec, sq);
  }

  float local_sum = cta_reduce_sum_leader(sq, smem_acc);
  cluster_wait_barrier();

  if (tid == 0) {
    int rank = static_cast<int>(cluster_rank());
    mbarrier_arrive_expect_tx(smem_sum_mbar,
                              static_cast<uint32_t>(kParts * sizeof(float)));
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      store_shared_remote_float(local_sum, smem_reduce + rank, smem_sum_mbar,
                                static_cast<uint32_t>(p));
    }
    mbarrier_wait(smem_sum_mbar, 0);
    float total_sum = 0.0f;
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      total_sum += smem_reduce[p];
    }
    *smem_rstd = ptx_rsqrt(total_sum / static_cast<float>(N) + eps);
  }
  __syncthreads();
  float rstd = *smem_rstd;

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    int local_elem = local_vec * kLanes;
    XVec x_vec;
    XVec out_vec;
    Vec128<float> weight_vec0;
    Vec128<float> weight_vec1;
    float out_values[kLanes];
    x_vec.load_shared(smem_x + local_elem);
    weight_vec0 = weight_pref0[iter];
    if constexpr (kLanes > 4) {
      weight_vec1 = weight_pref1[iter];
    }
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float wv;
      if constexpr (kLanes > 4) {
        wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
      } else {
        wv = weight_vec0.get(lane);
      }
      out_values[lane] = x_vec.get(lane) * rstd * wv;
    }
    set_vec_from_float_lanes(out_vec, out_values);
    out_vec.store(out + row_base + elem);
  }
}

template <typename X, int kParts, int kVecs, bool kAssumeH1 = false>
__global__ void rmsnorm_fwd_cluster_mbar_cpasync_weight_full_kernel(
    X* __restrict__ out, X const* __restrict__ x,
    float const* __restrict__ weight, int M, int N, int H, float eps) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __shared__ float smem_acc[33];
  __shared__ float smem_reduce[16];
  __shared__ float smem_rstd;
  __shared__ uint64_t smem_sum_mbar;
  rmsnorm_fwd_cluster_mbar_cpasync_weight_impl<X, kParts, kVecs, kAssumeH1>(
      out, x, weight, M, N, H, eps, smem_raw, smem_acc, smem_reduce,
      &smem_rstd, &smem_sum_mbar);
}

template <typename X, int kParts, int kVecs, bool kAssumeH1 = false>
__launch_bounds__(256, 1) __global__ void
rmsnorm_fwd_cluster_mbar_cpasync_weight_256_full_kernel(
    X* __restrict__ out, X const* __restrict__ x,
    float const* __restrict__ weight, int M, int N, int H, float eps) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __shared__ float smem_acc[33];
  __shared__ float smem_reduce[16];
  __shared__ float smem_rstd;
  __shared__ uint64_t smem_sum_mbar;
  rmsnorm_fwd_cluster_mbar_cpasync_weight_impl<X, kParts, kVecs, kAssumeH1>(
      out, x, weight, M, N, H, eps, smem_raw, smem_acc, smem_reduce,
      &smem_rstd, &smem_sum_mbar);
}

template <typename X, int kParts, int kVecs>
__global__ void rmsnorm_fwd_cluster_bcast_full_kernel(
    X* __restrict__ out, X const* __restrict__ x,
    float const* __restrict__ weight, int M, int N, int H, float eps) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  X* smem_x = reinterpret_cast<X*>(smem_raw);
  __shared__ float smem_acc[33];
  __shared__ float smem_sum;
  __shared__ float smem_rstd;

  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  cg::cluster_group cluster = cg::this_cluster();
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int part = blockIdx.y % kParts;
  int head = H > 1 ? row % H : 0;
  int begin_vec = part * kVecs * blockDim.x;
  int row_base = row * N;
  int affine_base = head * N;

  float sq = 0.0f;
#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    int local_elem = local_vec * kLanes;
    XVec x_vec;
    x_vec.load_cg(x + row_base + elem);
    x_vec.store_shared(smem_x + local_elem);
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec.get(lane);
      sq = fmaf(xv, xv, sq);
    }
  }

  float local_sum = cta_reduce_sum_leader(sq, smem_acc);
  if (tid == 0) {
    smem_sum = local_sum;
  }
  cluster.sync();

  if (static_cast<int>(cluster_rank()) == 0 && tid == 0) {
    float total_sum = 0.0f;
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      float* remote_sum = cluster.map_shared_rank(&smem_sum, p);
      total_sum += *remote_sum;
    }
    float rstd = ptx_rsqrt(total_sum / static_cast<float>(N) + eps);
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      float* remote_rstd = cluster.map_shared_rank(&smem_rstd, p);
      *remote_rstd = rstd;
    }
  }
  cluster.sync();
  float rstd = smem_rstd;

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    int local_elem = local_vec * kLanes;
    XVec x_vec;
    Vec128<float> weight_vec0;
    Vec128<float> weight_vec1;
    x_vec.load_shared(smem_x + local_elem);
    weight_vec0.load(weight + affine_base + elem);
    if constexpr (kLanes > 4) {
      weight_vec1.load(weight + affine_base + elem + 4);
    }
    XVec out_vec;
    out_vec.bits = make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec.get(lane);
      float wv;
      if constexpr (kLanes > 4) {
        wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
      } else {
        wv = weight_vec0.get(lane);
      }
      out_vec.set(lane, xv * rstd * wv);
    }
    out_vec.store(out + row_base + elem);
  }
}

template <typename X, int kParts, int kVecs>
__global__ void rmsnorm_fwd_cluster_twopass_full_kernel(
    X* __restrict__ out, X const* __restrict__ x,
    float const* __restrict__ weight, int M, int N, int H, float eps) {
  __shared__ float smem_acc[33];
  __shared__ float smem_reduce[16];
  __shared__ float smem_rstd;
  __shared__ uint64_t smem_sum_mbar;

  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  cg::cluster_group cluster = cg::this_cluster();
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int part = blockIdx.y % kParts;
  int head = H > 1 ? row % H : 0;
  int begin_vec = part * kVecs * blockDim.x;
  int row_base = row * N;
  int affine_base = head * N;

  if (tid == 0) {
    mbarrier_init(&smem_sum_mbar, 1);
  }
  mbarrier_init_fence();
  cluster_arrive_relaxed_barrier();

  float sq = 0.0f;
#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    XVec x_vec;
    x_vec.load_cg(x + row_base + elem);
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec.get(lane);
      sq = fmaf(xv, xv, sq);
    }
  }

  float local_sum = cta_reduce_sum_leader(sq, smem_acc);
  cluster_wait_barrier();

  if (tid == 0) {
    int rank = static_cast<int>(cluster_rank());
    mbarrier_arrive_expect_tx(&smem_sum_mbar,
                              static_cast<uint32_t>(kParts * sizeof(float)));
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      store_shared_remote_float(local_sum, smem_reduce + rank, &smem_sum_mbar,
                                static_cast<uint32_t>(p));
    }
    mbarrier_wait(&smem_sum_mbar, 0);
    float total_sum = 0.0f;
#pragma unroll
    for (int p = 0; p < kParts; ++p) {
      total_sum += smem_reduce[p];
    }
    smem_rstd = ptx_rsqrt(total_sum / static_cast<float>(N) + eps);
  }
  __syncthreads();
  float rstd = smem_rstd;

#pragma unroll
  for (int iter = 0; iter < kVecs; ++iter) {
    int local_vec = iter * blockDim.x + tid;
    int elem = (begin_vec + local_vec) * kLanes;
    XVec x_vec;
    Vec128<float> weight_vec0;
    Vec128<float> weight_vec1;
    x_vec.load_cg(x + row_base + elem);
    weight_vec0.load(weight + affine_base + elem);
    if constexpr (kLanes > 4) {
      weight_vec1.load(weight + affine_base + elem + 4);
    }
    XVec out_vec;
    float out_values[kLanes];
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float xv = x_vec.get(lane);
      float wv;
      if constexpr (kLanes > 4) {
        wv = lane < 4 ? weight_vec0.get(lane) : weight_vec1.get(lane - 4);
      } else {
        wv = weight_vec0.get(lane);
      }
      out_values[lane] = xv * rstd * wv;
    }
    set_vec_from_float_lanes(out_vec, out_values);
    out_vec.store(out + row_base + elem);
  }
  if constexpr (kParts > 16) {
    cluster.sync();
  }
}

template <typename X>
__global__ void rmsnorm_bwd_streaming_kernel(
    X* __restrict__ dx, X* __restrict__ dresidual,
    float* __restrict__ dw, float* __restrict__ db, X const* __restrict__ x,
    float const* __restrict__ weight, X const* __restrict__ dout,
    X const* __restrict__ dresidual_out, float const* __restrict__ rstd_in,
    int M, int N, int H) {
  __shared__ float smem_acc[33];
  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  int tid = threadIdx.x;
  int row = blockIdx.x;
  int head = H > 1 ? row % H : 0;
  int vecs = N / kLanes;
  int row_base = row * N;
  int affine_base = head * N;
  float rstd = rstd_in[row];

  float dot = 0.0f;
  for (int vec_idx = tid; vec_idx < vecs; vec_idx += blockDim.x) {
    int elem = vec_idx * kLanes;
    XVec x_vec;
    XVec dout_vec;
    x_vec.load(x + row_base + elem);
    dout_vec.load(dout + row_base + elem);
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      int col = elem + lane;
      float x_hat = x_vec.get(lane) * rstd;
      float wdy = dout_vec.get(lane);
      if (weight != nullptr) {
        wdy *= weight[affine_base + col];
      }
      dot = fmaf(x_hat, wdy, dot);
    }
  }

  float mean_xhat_wdy = cta_reduce_sum(dot, smem_acc) / static_cast<float>(N);

  for (int vec_idx = tid; vec_idx < vecs; vec_idx += blockDim.x) {
    int elem = vec_idx * kLanes;
    XVec x_vec;
    XVec dout_vec;
    XVec dresout_vec;
    XVec dx_vec;
    dx_vec.bits = make_uint4(0, 0, 0, 0);
    x_vec.load(x + row_base + elem);
    dout_vec.load(dout + row_base + elem);
    if (dresidual_out != nullptr) {
      dresout_vec.load(dresidual_out + row_base + elem);
    }
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      int col = elem + lane;
      float xv = x_vec.get(lane);
      float dy = dout_vec.get(lane);
      float x_hat = xv * rstd;
      float wdy = dy;
      if (weight != nullptr) {
        wdy *= weight[affine_base + col];
      }
      float grad = (wdy - x_hat * mean_xhat_wdy) * rstd;
      if (dresidual_out != nullptr) {
        grad += dresout_vec.get(lane);
      }
      dx_vec.set(lane, grad);
      if (dw != nullptr) {
        atomicAdd(dw + affine_base + col, dy * x_hat);
      }
      if (db != nullptr) {
        atomicAdd(db + affine_base + col, dy);
      }
    }
    dx_vec.store(dx + row_base + elem);
    if (dresidual != nullptr) {
      dx_vec.store(dresidual + row_base + elem);
    }
  }
}

template <typename X, int kMaxVecs>
__global__ void rmsnorm_bwd_weight_partial_kernel(
    X* __restrict__ dx, float* __restrict__ dw_partial,
    X const* __restrict__ x, float const* __restrict__ weight,
    X const* __restrict__ dout, float const* __restrict__ rstd_in, int M,
    int N) {
  __shared__ float smem_acc[33];
  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  int tid = threadIdx.x;
  int vecs = N / kLanes;

  float dw_acc[kMaxVecs][kLanes];
#pragma unroll
  for (int iter = 0; iter < kMaxVecs; ++iter) {
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      dw_acc[iter][lane] = 0.0f;
    }
  }

  for (int row = blockIdx.x; row < M; row += gridDim.x) {
    int row_base = row * N;
    float rstd = rstd_in[row];
    XVec x_vec[kMaxVecs];
    XVec dout_vec[kMaxVecs];
    Vec128<float> weight_vec0[kMaxVecs];
    Vec128<float> weight_vec1[kMaxVecs];
    bool pred[kMaxVecs];
    float dot = 0.0f;

#pragma unroll
    for (int iter = 0; iter < kMaxVecs; ++iter) {
      int vec_idx = iter * blockDim.x + tid;
      bool valid = vec_idx < vecs;
      pred[iter] = valid;
      if (valid) {
        int elem = vec_idx * kLanes;
        x_vec[iter].load(x + row_base + elem);
        dout_vec[iter].load(dout + row_base + elem);
        weight_vec0[iter].load(weight + elem);
        if constexpr (kLanes > 4) {
          weight_vec1[iter].load(weight + elem + 4);
        }
#pragma unroll
        for (int lane = 0; lane < kLanes; ++lane) {
          float xv = x_vec[iter].get(lane);
          float dy = dout_vec[iter].get(lane);
          float wv = lane < 4 ? weight_vec0[iter].get(lane)
                              : weight_vec1[iter].get(lane - 4);
          float x_hat = xv * rstd;
          dot = fmaf(x_hat, dy * wv, dot);
        }
      }
    }

    float mean_xhat_wdy = cta_reduce_sum(dot, smem_acc) / static_cast<float>(N);

#pragma unroll
    for (int iter = 0; iter < kMaxVecs; ++iter) {
      if (pred[iter]) {
        int elem = (iter * blockDim.x + tid) * kLanes;
        XVec dx_vec;
        dx_vec.bits = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int lane = 0; lane < kLanes; ++lane) {
          float xv = x_vec[iter].get(lane);
          float dy = dout_vec[iter].get(lane);
          float wv = lane < 4 ? weight_vec0[iter].get(lane)
                              : weight_vec1[iter].get(lane - 4);
          float x_hat = xv * rstd;
          float grad = (dy * wv - x_hat * mean_xhat_wdy) * rstd;
          dx_vec.set(lane, grad);
          dw_acc[iter][lane] += dy * x_hat;
        }
        dx_vec.store(dx + row_base + elem);
      }
    }
  }

#pragma unroll
  for (int iter = 0; iter < kMaxVecs; ++iter) {
    int vec_idx = iter * blockDim.x + tid;
    if (vec_idx < vecs) {
      int elem = vec_idx * kLanes;
      store_float_lanes<kLanes>(
          dw_partial + static_cast<size_t>(blockIdx.x) * N + elem,
          dw_acc[iter]);
    }
  }
}

template <typename X, int kThreadsPerRow, int kRowsPerBlock>
__global__ void rmsnorm_bwd_weight_small_rows_kernel(
    X* __restrict__ dx, float* __restrict__ dw_partial,
    X const* __restrict__ x, float const* __restrict__ weight,
    X const* __restrict__ dout, float const* __restrict__ rstd_in, int M,
    int N) {
  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;
  static_assert(kThreadsPerRow * kRowsPerBlock == 128);
  static_assert(kThreadsPerRow % 32 == 0);

  __shared__ float smem_acc[kRowsPerBlock * (kThreadsPerRow / 32)];
  __shared__ float smem_dw[kRowsPerBlock * kThreadsPerRow * kLanes];

  int tid = threadIdx.x;
  int row_group = tid / kThreadsPerRow;
  int row_tid = tid - row_group * kThreadsPerRow;
  int elem = row_tid * kLanes;

  Vec128<float> w0;
  Vec128<float> w1;
  w0.load(weight + elem);
  if constexpr (kLanes > 4) {
    w1.load(weight + elem + 4);
  }
  float weight_reg[kLanes];
#pragma unroll
  for (int lane = 0; lane < kLanes; ++lane) {
    if constexpr (kLanes > 4) {
      weight_reg[lane] = lane < 4 ? w0.get(lane) : w1.get(lane - 4);
    } else {
      weight_reg[lane] = w0.get(lane);
    }
  }

  float dw_acc[kLanes];
#pragma unroll
  for (int lane = 0; lane < kLanes; ++lane) {
    dw_acc[lane] = 0.0f;
  }

  for (int row_block = blockIdx.x * kRowsPerBlock; row_block < M;
       row_block += gridDim.x * kRowsPerBlock) {
    int row = row_block + row_group;
    bool valid_row = row < M;
    int row_base = row * N;
    float rstd = valid_row ? rstd_in[row] : 0.0f;
    XVec x_vec;
    XVec dout_vec;
    if (valid_row) {
      x_vec.load(x + row_base + elem);
      dout_vec.load(dout + row_base + elem);
    } else {
      x_vec.bits = make_uint4(0, 0, 0, 0);
      dout_vec.bits = make_uint4(0, 0, 0, 0);
    }

    float dot = 0.0f;
    float xhat_reg[kLanes];
    float dout_reg[kLanes];
    float wdy_reg[kLanes];
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float x_hat = x_vec.get(lane) * rstd;
      float dy_f = dout_vec.get(lane);
      float wdy = dy_f * weight_reg[lane];
      xhat_reg[lane] = x_hat;
      dout_reg[lane] = dy_f;
      wdy_reg[lane] = wdy;
      dot = fmaf(x_hat, wdy, dot);
    }

    float mean_xhat_wdy =
        row_group_reduce_sum(dot, smem_acc, kThreadsPerRow) /
        static_cast<float>(N);

    XVec dx_vec;
    dx_vec.bits = make_uint4(0, 0, 0, 0);
    if constexpr (std::is_same_v<X, __nv_bfloat16>) {
#pragma unroll
      for (int lane = 0; lane < kLanes; lane += 2) {
        float x_hat0 = xhat_reg[lane];
        float dy_f0 = dout_reg[lane];
        float wdy0 = wdy_reg[lane];
        float grad0 = (wdy0 - x_hat0 * mean_xhat_wdy) * rstd;
        float x_hat1 = xhat_reg[lane + 1];
        float dy_f1 = dout_reg[lane + 1];
        float wdy1 = wdy_reg[lane + 1];
        float grad1 = (wdy1 - x_hat1 * mean_xhat_wdy) * rstd;
        uint32_t packed = pack_bf16x2(grad0, grad1);
        if (lane == 0) {
          dx_vec.bits.x = packed;
        } else if (lane == 2) {
          dx_vec.bits.y = packed;
        } else if (lane == 4) {
          dx_vec.bits.z = packed;
        } else {
          dx_vec.bits.w = packed;
        }
        dw_acc[lane] += dy_f0 * x_hat0;
        dw_acc[lane + 1] += dy_f1 * x_hat1;
      }
    } else {
#pragma unroll
      for (int lane = 0; lane < kLanes; ++lane) {
        float x_hat = xhat_reg[lane];
        float dy_f = dout_reg[lane];
        float grad = (wdy_reg[lane] - x_hat * mean_xhat_wdy) * rstd;
        dx_vec.set(lane, grad);
        dw_acc[lane] += dy_f * x_hat;
      }
    }
    if (valid_row) {
      dx_vec.store(dx + row_base + elem);
    }
  }

#pragma unroll
  for (int lane = 0; lane < kLanes; ++lane) {
    smem_dw[(row_group * kThreadsPerRow + row_tid) * kLanes + lane] =
        dw_acc[lane];
  }
  __syncthreads();

  if (row_group == 0) {
    float sum_lanes[kLanes];
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      float sum = 0.0f;
#pragma unroll
      for (int group = 0; group < kRowsPerBlock; ++group) {
        sum += smem_dw[(group * kThreadsPerRow + row_tid) * kLanes + lane];
      }
      sum_lanes[lane] = sum;
    }
    store_float_lanes<kLanes>(
        dw_partial + static_cast<size_t>(blockIdx.x) * N + elem, sum_lanes);
  }
}

template <typename X, int kMaxVecs, bool kExact = false>
__global__ void rmsnorm_bwd_weight_smem_persistent_kernel(
    X* __restrict__ dx, float* __restrict__ dw_partial,
    X const* __restrict__ x, float const* __restrict__ weight,
    X const* __restrict__ dout, float const* __restrict__ rstd_in, int M,
    int N) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  X* smem_x = reinterpret_cast<X*>(smem_raw);
  X* smem_dout = smem_x + 2 * N;
  __shared__ float smem_acc[33];

  int tid = threadIdx.x;
  int vecs = N / kLanes;

  float weight_reg[kMaxVecs][kLanes];
  float dw_acc[kMaxVecs][kLanes];
#pragma unroll
  for (int iter = 0; iter < kMaxVecs; ++iter) {
    int vec_idx = iter * blockDim.x + tid;
    if constexpr (kExact) {
      int elem = vec_idx * kLanes;
      Vec128<float> w0;
      Vec128<float> w1;
      w0.load(weight + elem);
      if constexpr (kLanes > 4) {
        w1.load(weight + elem + 4);
      }
#pragma unroll
      for (int lane = 0; lane < kLanes; ++lane) {
        if constexpr (kLanes > 4) {
          weight_reg[iter][lane] =
              lane < 4 ? w0.get(lane) : w1.get(lane - 4);
        } else {
          weight_reg[iter][lane] = w0.get(lane);
        }
      }
    } else if (vec_idx < vecs) {
      int elem = vec_idx * kLanes;
      Vec128<float> w0;
      Vec128<float> w1;
      w0.load(weight + elem);
      if constexpr (kLanes > 4) {
        w1.load(weight + elem + 4);
      }
#pragma unroll
      for (int lane = 0; lane < kLanes; ++lane) {
        if constexpr (kLanes > 4) {
          weight_reg[iter][lane] =
              lane < 4 ? w0.get(lane) : w1.get(lane - 4);
        } else {
          weight_reg[iter][lane] = w0.get(lane);
        }
      }
    } else {
#pragma unroll
      for (int lane = 0; lane < kLanes; ++lane) {
        weight_reg[iter][lane] = 0.0f;
      }
    }
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      dw_acc[iter][lane] = 0.0f;
    }
  }

  int row0 = blockIdx.x;
#pragma unroll
  for (int iter = 0; iter < kMaxVecs; ++iter) {
    int vec_idx = iter * blockDim.x + tid;
    if (row0 < M && (kExact || vec_idx < vecs)) {
      int elem = vec_idx * kLanes;
      cp_async_128(smem_x + elem, x + row0 * N + elem);
      cp_async_128(smem_dout + elem, dout + row0 * N + elem);
    }
  }
  cp_async_commit_group();

  int stage = 0;
  for (int row = row0; row < M; row += gridDim.x) {
    int next_row = row + gridDim.x;
    float rstd = rstd_in[row];
    if (next_row < M) {
      X* next_x = smem_x + (stage ^ 1) * N;
      X* next_dout = smem_dout + (stage ^ 1) * N;
#pragma unroll
      for (int iter = 0; iter < kMaxVecs; ++iter) {
        int vec_idx = iter * blockDim.x + tid;
        if (kExact || vec_idx < vecs) {
          int elem = vec_idx * kLanes;
          cp_async_128(next_x + elem, x + next_row * N + elem);
          cp_async_128(next_dout + elem, dout + next_row * N + elem);
        }
      }
      cp_async_commit_group();
      cp_async_wait_group_1();
    } else {
      cp_async_wait_group_0();
    }

    X* cur_x = smem_x + stage * N;
    X* cur_dout = smem_dout + stage * N;
    float dot = 0.0f;
    float xhat_reg[kMaxVecs][kLanes];
    float dout_reg[kMaxVecs][kLanes];
    float wdy_reg[kMaxVecs][kLanes];
    bool pred[kMaxVecs];

#pragma unroll
    for (int iter = 0; iter < kMaxVecs; ++iter) {
      int vec_idx = iter * blockDim.x + tid;
      bool valid = kExact || vec_idx < vecs;
      pred[iter] = valid;
      if (valid) {
        int elem = vec_idx * kLanes;
        XVec x_vec;
        XVec dout_vec;
        x_vec.load_shared(cur_x + elem);
        dout_vec.load_shared(cur_dout + elem);
#pragma unroll
        for (int lane = 0; lane < kLanes; ++lane) {
          float x_f = x_vec.get(lane);
          float dy_f = dout_vec.get(lane);
          float x_hat = x_f * rstd;
          float wdy = dy_f * weight_reg[iter][lane];
          xhat_reg[iter][lane] = x_hat;
          dout_reg[iter][lane] = dy_f;
          wdy_reg[iter][lane] = wdy;
          dot = fmaf(x_hat, wdy, dot);
        }
      } else {
#pragma unroll
        for (int lane = 0; lane < kLanes; ++lane) {
          xhat_reg[iter][lane] = 0.0f;
          dout_reg[iter][lane] = 0.0f;
          wdy_reg[iter][lane] = 0.0f;
        }
      }
    }

    float mean_xhat_wdy =
        cta_reduce_sum(dot, smem_acc) / static_cast<float>(N);

#pragma unroll
    for (int iter = 0; iter < kMaxVecs; ++iter) {
      if (kExact || pred[iter]) {
        int elem = (iter * blockDim.x + tid) * kLanes;
        XVec dx_vec;
        dx_vec.bits = make_uint4(0, 0, 0, 0);
        if constexpr (std::is_same_v<X, __nv_bfloat16>) {
#pragma unroll
          for (int lane = 0; lane < kLanes; lane += 2) {
            float x_hat0 = xhat_reg[iter][lane];
            float dy_f0 = dout_reg[iter][lane];
            float wdy0 = wdy_reg[iter][lane];
            float grad0 = (wdy0 - x_hat0 * mean_xhat_wdy) * rstd;
            float x_hat1 = xhat_reg[iter][lane + 1];
            float dy_f1 = dout_reg[iter][lane + 1];
            float wdy1 = wdy_reg[iter][lane + 1];
            float grad1 = (wdy1 - x_hat1 * mean_xhat_wdy) * rstd;
            uint32_t packed = pack_bf16x2(grad0, grad1);
            if (lane == 0) {
              dx_vec.bits.x = packed;
            } else if (lane == 2) {
              dx_vec.bits.y = packed;
            } else if (lane == 4) {
              dx_vec.bits.z = packed;
            } else {
              dx_vec.bits.w = packed;
            }
            dw_acc[iter][lane] += dy_f0 * x_hat0;
            dw_acc[iter][lane + 1] += dy_f1 * x_hat1;
          }
        } else {
#pragma unroll
          for (int lane = 0; lane < kLanes; ++lane) {
            float x_hat = xhat_reg[iter][lane];
            float dy_f = dout_reg[iter][lane];
            float wdy = wdy_reg[iter][lane];
            float grad = (wdy - x_hat * mean_xhat_wdy) * rstd;
            dx_vec.set(lane, grad);
            dw_acc[iter][lane] += dy_f * x_hat;
          }
        }
        dx_vec.store(dx + row * N + elem);
      }
    }
    stage ^= 1;
  }

#pragma unroll
  for (int iter = 0; iter < kMaxVecs; ++iter) {
    int vec_idx = iter * blockDim.x + tid;
    if (kExact || vec_idx < vecs) {
      int elem = vec_idx * kLanes;
      store_float_lanes<kLanes>(
          dw_partial + static_cast<size_t>(blockIdx.x) * N + elem,
          dw_acc[iter]);
    }
  }
}

template <typename X, int kMaxVecs, bool kReloadDout, bool kExact = false>
__global__ void rmsnorm_bwd_weight_cluster_smem_persistent_kernel(
    X* __restrict__ dx, float* __restrict__ dw_partial,
    X const* __restrict__ x, float const* __restrict__ weight,
    X const* __restrict__ dout, float const* __restrict__ rstd_in, int M,
    int N, int parts) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  cg::cluster_group cluster = cg::this_cluster();
  X* smem_x = reinterpret_cast<X*>(smem_raw);
  X* smem_dout = smem_x + 2 * ((N / kLanes + parts - 1) / parts) * kLanes;
  __shared__ float smem_reduce[2 * 16 * 16];
  __shared__ uint64_t smem_mbar[4];

  int tid = threadIdx.x;
  int part = blockIdx.y % parts;
  int vecs = N / kLanes;
  int vecs_per_part = (vecs + parts - 1) / parts;
  int begin_vec = part * vecs_per_part;
  int end_vec = min(vecs, begin_vec + vecs_per_part);
  int part_vecs = end_vec - begin_vec;
  int begin_elem = begin_vec * kLanes;
  int elems_per_part = vecs_per_part * kLanes;
  int num_warps = (blockDim.x + 31) >> 5;

  if (tid < 2) {
    mbarrier_init(smem_mbar + tid, 1);
    mbarrier_init(smem_mbar + 2 + tid, num_warps * parts);
  }
  mbarrier_init_fence();
  cluster.sync();

  float weight_reg[kMaxVecs][kLanes];
  float dw_acc[kMaxVecs][kLanes];
#pragma unroll
  for (int iter = 0; iter < kMaxVecs; ++iter) {
    int vec_local = iter * blockDim.x + tid;
    if (kExact || vec_local < part_vecs) {
      int elem = begin_elem + vec_local * kLanes;
      Vec128<float> w0;
      Vec128<float> w1;
      w0.load(weight + elem);
      if constexpr (kLanes > 4) {
        w1.load(weight + elem + 4);
      }
#pragma unroll
      for (int lane = 0; lane < kLanes; ++lane) {
        if constexpr (kLanes > 4) {
          weight_reg[iter][lane] =
              lane < 4 ? w0.get(lane) : w1.get(lane - 4);
        } else {
          weight_reg[iter][lane] = w0.get(lane);
        }
      }
    } else {
#pragma unroll
      for (int lane = 0; lane < kLanes; ++lane) {
        weight_reg[iter][lane] = 0.0f;
      }
    }
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      dw_acc[iter][lane] = 0.0f;
    }
  }

  int row0 = blockIdx.x;
#pragma unroll
  for (int iter = 0; iter < kMaxVecs; ++iter) {
    int vec_local = iter * blockDim.x + tid;
    if (row0 < M && (kExact || vec_local < part_vecs)) {
      int elem_local = vec_local * kLanes;
      int elem = begin_elem + elem_local;
      cp_async_128(smem_x + elem_local, x + row0 * N + elem);
      cp_async_128(smem_dout + elem_local, dout + row0 * N + elem);
    }
  }
  cp_async_commit_group();

  int stage = 0;
  int consumer_phase = 0;
  int producer_phase = 1;
  for (int row = row0; row < M; row += gridDim.x) {
    int next_row = row + gridDim.x;
    float rstd = rstd_in[row];
    if (next_row < M) {
      X* next_x = smem_x + (stage ^ 1) * elems_per_part;
      X* next_dout = smem_dout + (stage ^ 1) * elems_per_part;
#pragma unroll
      for (int iter = 0; iter < kMaxVecs; ++iter) {
        int vec_local = iter * blockDim.x + tid;
        if (kExact || vec_local < part_vecs) {
          int elem_local = vec_local * kLanes;
          int elem = begin_elem + elem_local;
          cp_async_128(next_x + elem_local, x + next_row * N + elem);
          cp_async_128(next_dout + elem_local, dout + next_row * N + elem);
        }
      }
      cp_async_commit_group();
      cp_async_wait_group_1();
    } else {
      cp_async_wait_group_0();
    }
    __syncthreads();

    X* cur_x = smem_x + stage * elems_per_part;
    X* cur_dout = smem_dout + stage * elems_per_part;
    float dot = 0.0f;
    float xhat_reg[kMaxVecs][kLanes];

    if constexpr (kReloadDout) {
#pragma unroll
      for (int iter = 0; iter < kMaxVecs; ++iter) {
        int vec_local = iter * blockDim.x + tid;
        if (kExact || vec_local < part_vecs) {
          int elem_local = vec_local * kLanes;
          XVec x_vec;
          XVec dout_vec;
          x_vec.load_shared(cur_x + elem_local);
          dout_vec.load_shared(cur_dout + elem_local);
#pragma unroll
          for (int lane = 0; lane < kLanes; ++lane) {
            float x_hat = x_vec.get(lane) * rstd;
            float wdy = dout_vec.get(lane) * weight_reg[iter][lane];
            xhat_reg[iter][lane] = x_hat;
            dot = fmaf(x_hat, wdy, dot);
          }
        } else {
#pragma unroll
          for (int lane = 0; lane < kLanes; ++lane) {
            xhat_reg[iter][lane] = 0.0f;
          }
        }
      }
    } else {
      float dout_reg[kMaxVecs][kLanes];
      float wdy_reg[kMaxVecs][kLanes];
#pragma unroll
      for (int iter = 0; iter < kMaxVecs; ++iter) {
        int vec_local = iter * blockDim.x + tid;
        if (kExact || vec_local < part_vecs) {
          int elem_local = vec_local * kLanes;
          XVec x_vec;
          XVec dout_vec;
          x_vec.load_shared(cur_x + elem_local);
          dout_vec.load_shared(cur_dout + elem_local);
#pragma unroll
          for (int lane = 0; lane < kLanes; ++lane) {
            float dy_f = dout_vec.get(lane);
            float x_hat = x_vec.get(lane) * rstd;
            float wdy = dy_f * weight_reg[iter][lane];
            xhat_reg[iter][lane] = x_hat;
            dout_reg[iter][lane] = dy_f;
            wdy_reg[iter][lane] = wdy;
            dot = fmaf(x_hat, wdy, dot);
          }
        } else {
#pragma unroll
          for (int lane = 0; lane < kLanes; ++lane) {
            xhat_reg[iter][lane] = 0.0f;
            dout_reg[iter][lane] = 0.0f;
            wdy_reg[iter][lane] = 0.0f;
          }
        }
      }

      mbarrier_wait(smem_mbar + 2 + stage, producer_phase);
      float mean_xhat_wdy =
          cluster_reduce_sum_mbar(dot, smem_reduce, smem_mbar, stage,
                                  consumer_phase, parts) /
          static_cast<float>(N);
      fence_view_async_shared();
      __syncwarp();
      if ((tid & 31) < parts) {
        mbarrier_arrive_remote(smem_mbar + 2 + stage,
                               static_cast<uint32_t>(tid & 31));
      }

#pragma unroll
      for (int iter = 0; iter < kMaxVecs; ++iter) {
        int vec_local = iter * blockDim.x + tid;
        if (kExact || vec_local < part_vecs) {
          int elem_local = vec_local * kLanes;
          int elem = begin_elem + elem_local;
          XVec dx_vec;
          dx_vec.bits = make_uint4(0, 0, 0, 0);
          if constexpr (std::is_same_v<X, __nv_bfloat16>) {
#pragma unroll
            for (int lane = 0; lane < kLanes; lane += 2) {
              float x_hat0 = xhat_reg[iter][lane];
              float dy_f0 = dout_reg[iter][lane];
              float wdy0 = wdy_reg[iter][lane];
              float grad0 = (wdy0 - x_hat0 * mean_xhat_wdy) * rstd;
              float x_hat1 = xhat_reg[iter][lane + 1];
              float dy_f1 = dout_reg[iter][lane + 1];
              float wdy1 = wdy_reg[iter][lane + 1];
              float grad1 = (wdy1 - x_hat1 * mean_xhat_wdy) * rstd;
              uint32_t packed = pack_bf16x2(grad0, grad1);
              if (lane == 0) {
                dx_vec.bits.x = packed;
              } else if (lane == 2) {
                dx_vec.bits.y = packed;
              } else if (lane == 4) {
                dx_vec.bits.z = packed;
              } else {
                dx_vec.bits.w = packed;
              }
              dw_acc[iter][lane] += dy_f0 * x_hat0;
              dw_acc[iter][lane + 1] += dy_f1 * x_hat1;
            }
          } else {
#pragma unroll
            for (int lane = 0; lane < kLanes; ++lane) {
              float x_hat = xhat_reg[iter][lane];
              float dy_f = dout_reg[iter][lane];
              float wdy = wdy_reg[iter][lane];
              float grad = (wdy - x_hat * mean_xhat_wdy) * rstd;
              dx_vec.set(lane, grad);
              dw_acc[iter][lane] += dy_f * x_hat;
            }
          }
          dx_vec.store(dx + row * N + elem);
        }
      }
      stage ^= 1;
      if (stage == 0) {
        consumer_phase ^= 1;
        producer_phase ^= 1;
      }
      continue;
    }

    mbarrier_wait(smem_mbar + 2 + stage, producer_phase);
    float mean_xhat_wdy =
        cluster_reduce_sum_mbar(dot, smem_reduce, smem_mbar, stage,
                                consumer_phase, parts) /
        static_cast<float>(N);
    fence_view_async_shared();
    __syncwarp();
    if ((tid & 31) < parts) {
      mbarrier_arrive_remote(smem_mbar + 2 + stage,
                             static_cast<uint32_t>(tid & 31));
    }

#pragma unroll
    for (int iter = 0; iter < kMaxVecs; ++iter) {
      int vec_local = iter * blockDim.x + tid;
      if (kExact || vec_local < part_vecs) {
        int elem_local = vec_local * kLanes;
        int elem = begin_elem + elem_local;
        XVec dout_vec;
        XVec dx_vec;
        dout_vec.load_shared(cur_dout + elem_local);
        dx_vec.bits = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int lane = 0; lane < kLanes; ++lane) {
          float x_hat = xhat_reg[iter][lane];
          float dy_f = dout_vec.get(lane);
          float wdy = dy_f * weight_reg[iter][lane];
          float grad = (wdy - x_hat * mean_xhat_wdy) * rstd;
          dx_vec.set(lane, grad);
          dw_acc[iter][lane] += dy_f * x_hat;
        }
        dx_vec.store(dx + row * N + elem);
      }
    }
    stage ^= 1;
    if (stage == 0) {
      consumer_phase ^= 1;
      producer_phase ^= 1;
    }
  }

  stage ^= 1;
  if (stage == 0) {
    producer_phase ^= 1;
  }
  mbarrier_wait(smem_mbar + 2 + stage, producer_phase);

#pragma unroll
  for (int iter = 0; iter < kMaxVecs; ++iter) {
    int vec_local = iter * blockDim.x + tid;
    if (kExact || vec_local < part_vecs) {
      int elem_local = vec_local * kLanes;
      int elem = begin_elem + elem_local;
      store_float_lanes<kLanes>(
          dw_partial + static_cast<size_t>(blockIdx.x) * N + elem,
          dw_acc[iter]);
    }
  }
}

template <typename X, int kMaxVecs, bool kExact = false>
__global__ void rmsnorm_bwd_weight_cluster_partial_kernel(
    X* __restrict__ dx, float* __restrict__ dw_partial,
    X const* __restrict__ x, float const* __restrict__ weight,
    X const* __restrict__ dout, float const* __restrict__ rstd_in, int M,
    int N, int parts) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  using XVec = Vec128<X>;
  constexpr int kLanes = XVec::kLanes;

  int vecs = N / kLanes;
  int vecs_per_part = (vecs + parts - 1) / parts;
  int elems_per_part = vecs_per_part * kLanes;
  X* smem_x = reinterpret_cast<X*>(smem_raw);
  X* smem_dout = smem_x + elems_per_part;
  float* smem_w = reinterpret_cast<float*>(smem_dout + elems_per_part);
  __shared__ float smem_reduce[16 * 16];
  __shared__ uint64_t smem_mbar[2];

  cg::cluster_group cluster = cg::this_cluster();
  int tid = threadIdx.x;
  int part = blockIdx.y % parts;
  int begin_vec = part * vecs_per_part;
  int end_vec = min(vecs, begin_vec + vecs_per_part);
  int part_vecs = end_vec - begin_vec;
  int begin_elem = begin_vec * kLanes;
  int num_warps = (blockDim.x + 31) >> 5;

  if (tid == 0) {
    mbarrier_init(smem_mbar, 1);
    mbarrier_init(smem_mbar + 1, num_warps * parts);
  }
  mbarrier_init_fence();
  cluster.sync();

  float dw_acc[kMaxVecs][kLanes];
#pragma unroll
  for (int iter = 0; iter < kMaxVecs; ++iter) {
#pragma unroll
    for (int lane = 0; lane < kLanes; ++lane) {
      dw_acc[iter][lane] = 0.0f;
    }
  }

  for (int vec_local = tid; vec_local < part_vecs; vec_local += blockDim.x) {
    int elem_local = vec_local * kLanes;
    int elem = begin_elem + elem_local;
    Vec128<float> w0;
    Vec128<float> w1;
    w0.load(weight + elem);
    w0.store_shared(smem_w + elem_local);
    if constexpr (kLanes > 4) {
      w1.load(weight + elem + 4);
      w1.store_shared(smem_w + elem_local + 4);
    }
  }
  __syncthreads();

  int consumer_phase = 0;
  int producer_phase = 1;
  for (int row = blockIdx.x; row < M; row += gridDim.x) {
    int row_base = row * N;
    float rstd = rstd_in[row];
    float dot = 0.0f;

#pragma unroll
    for (int iter = 0; iter < kMaxVecs; ++iter) {
      int vec_local = iter * blockDim.x + tid;
      bool valid = kExact || vec_local < part_vecs;
      if (valid) {
        int elem_local = vec_local * kLanes;
        int elem = begin_elem + elem_local;
        XVec xv;
        XVec dyv;
        Vec128<float> w0;
        Vec128<float> w1;
        xv.load(x + row_base + elem);
        dyv.load(dout + row_base + elem);
        xv.store_shared(smem_x + elem_local);
        dyv.store_shared(smem_dout + elem_local);
        w0.load_shared(smem_w + elem_local);
        if constexpr (kLanes > 4) {
          w1.load_shared(smem_w + elem_local + 4);
        }
#pragma unroll
        for (int lane = 0; lane < kLanes; ++lane) {
          float x_f = xv.get(lane);
          float dy_f = dyv.get(lane);
          float w_f = lane < 4 ? w0.get(lane) : w1.get(lane - 4);
          dot = fmaf(x_f * rstd, dy_f * w_f, dot);
        }
      }
    }

    mbarrier_wait(smem_mbar + 1, producer_phase);
    float mean_xhat_wdy =
        cluster_reduce_sum_mbar(dot, smem_reduce, smem_mbar, 0,
                                consumer_phase, parts) /
        static_cast<float>(N);
    fence_view_async_shared();
    __syncwarp();
    if ((tid & 31) < parts) {
      mbarrier_arrive_remote(smem_mbar + 1,
                             static_cast<uint32_t>(tid & 31));
    }

#pragma unroll
    for (int iter = 0; iter < kMaxVecs; ++iter) {
      int vec_local = iter * blockDim.x + tid;
      bool valid = kExact || vec_local < part_vecs;
      if (valid) {
        int elem_local = vec_local * kLanes;
        int elem = begin_elem + elem_local;
        XVec xv;
        XVec dyv;
        XVec dxv;
        Vec128<float> w0;
        Vec128<float> w1;
        xv.load_shared(smem_x + elem_local);
        dyv.load_shared(smem_dout + elem_local);
        w0.load_shared(smem_w + elem_local);
        if constexpr (kLanes > 4) {
          w1.load_shared(smem_w + elem_local + 4);
        }
        dxv.bits = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int lane = 0; lane < kLanes; ++lane) {
          float x_f = xv.get(lane);
          float dy_f = dyv.get(lane);
          float w_f = lane < 4 ? w0.get(lane) : w1.get(lane - 4);
          float x_hat = x_f * rstd;
          float grad = (dy_f * w_f - x_hat * mean_xhat_wdy) * rstd;
          dxv.set(lane, grad);
          dw_acc[iter][lane] += dy_f * x_hat;
        }
        dxv.store(dx + row_base + elem);
      }
    }
    consumer_phase ^= 1;
    producer_phase ^= 1;
  }

  mbarrier_wait(smem_mbar + 1, producer_phase);

#pragma unroll
  for (int iter = 0; iter < kMaxVecs; ++iter) {
    int vec_local = iter * blockDim.x + tid;
    if (kExact || vec_local < part_vecs) {
      int elem_local = vec_local * kLanes;
      int elem = begin_elem + elem_local;
      store_float_lanes<kLanes>(
          dw_partial + static_cast<size_t>(blockIdx.x) * N + elem,
          dw_acc[iter]);
    }
  }
}

__global__ void reduce_dw_partial_kernel(float* __restrict__ dw,
                                         float const* __restrict__ dw_partial,
                                         int partial_blocks, int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= N) {
    return;
  }
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  int block = 0;
  for (; block + 3 < partial_blocks; block += 4) {
    size_t base = static_cast<size_t>(block) * N + col;
    sum0 += dw_partial[base];
    sum1 += dw_partial[base + N];
    sum2 += dw_partial[base + 2 * static_cast<size_t>(N)];
    sum3 += dw_partial[base + 3 * static_cast<size_t>(N)];
  }
  float sum = (sum0 + sum1) + (sum2 + sum3);
  for (; block < partial_blocks; ++block) {
    sum += dw_partial[static_cast<size_t>(block) * N + col];
  }
  dw[col] = sum;
}

__global__ void reduce_dw_partial_chunks_kernel(
    float* __restrict__ scratch, float const* __restrict__ dw_partial,
    int partial_blocks, int N, int chunk_size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int chunk = blockIdx.y;
  if (col >= N) {
    return;
  }
  int begin = chunk * chunk_size;
  int end = min(partial_blocks, begin + chunk_size);
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  int block = begin;
  for (; block + 3 < end; block += 4) {
    size_t base = static_cast<size_t>(block) * N + col;
    sum0 += dw_partial[base];
    sum1 += dw_partial[base + N];
    sum2 += dw_partial[base + 2 * static_cast<size_t>(N)];
    sum3 += dw_partial[base + 3 * static_cast<size_t>(N)];
  }
  float sum = (sum0 + sum1) + (sum2 + sum3);
  for (; block < end; ++block) {
    sum += dw_partial[static_cast<size_t>(block) * N + col];
  }
  scratch[static_cast<size_t>(chunk) * N + col] = sum;
}

__global__ void reduce_dw_partial_chunks_atomic_kernel(
    float* __restrict__ dw, float const* __restrict__ dw_partial,
    int partial_blocks, int N, int chunk_size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int chunk = blockIdx.y;
  if (col >= N) {
    return;
  }
  int begin = chunk * chunk_size;
  int end = min(partial_blocks, begin + chunk_size);
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  int block = begin;
  for (; block + 3 < end; block += 4) {
    size_t base = static_cast<size_t>(block) * N + col;
    sum0 += dw_partial[base];
    sum1 += dw_partial[base + N];
    sum2 += dw_partial[base + 2 * static_cast<size_t>(N)];
    sum3 += dw_partial[base + 3 * static_cast<size_t>(N)];
  }
  float sum = (sum0 + sum1) + (sum2 + sum3);
  for (; block < end; ++block) {
    sum += dw_partial[static_cast<size_t>(block) * N + col];
  }
  atomicAdd(dw + col, sum);
}

void launch_reduce_dw(float* dw, float const* dw_partial, float* scratch,
                      int partial_blocks, int N, int reduce_chunks,
                      cudaStream_t stream) {
  int reduce_threads = N <= 8192 ? 64 : (N <= 16384 ? 128 : 256);
  dim3 block(reduce_threads);
  dim3 grid((N + reduce_threads - 1) / reduce_threads);
  if (reduce_chunks > 1 && scratch != nullptr) {
    int chunks = std::min(partial_blocks, reduce_chunks);
    int chunk_size = (partial_blocks + chunks - 1) / chunks;
    dim3 chunk_grid(grid.x, chunks);
    if (N == 8192) {
      CUDA_CHECK(cudaMemsetAsync(dw, 0, static_cast<size_t>(N) * sizeof(float),
                                 stream));
      reduce_dw_partial_chunks_atomic_kernel<<<chunk_grid, block, 0, stream>>>(
          dw, dw_partial, partial_blocks, N, chunk_size);
      return;
    }
    reduce_dw_partial_chunks_kernel<<<chunk_grid, block, 0, stream>>>(
        scratch, dw_partial, partial_blocks, N, chunk_size);
    reduce_dw_partial_kernel<<<grid, block, 0, stream>>>(dw, scratch, chunks, N);
    return;
  }
  reduce_dw_partial_kernel<<<grid, block, 0, stream>>>(dw, dw_partial,
                                                       partial_blocks, N);
}

template <typename X, typename RO>
void launch_fwd(Options const& options, X* out, RO* residual_out,
                float* rstd, X const* x, float const* weight,
                float const* bias, X const* residual, float* partial_sums,
                cudaStream_t stream) {
  int threads = options.threads;
  int threads_per_row = fwd_threads_per_row(options.N);
  if constexpr (sizeof(X) < 4) {
    // Residual fwd benefits from fewer threads per row than the no-residual
    // path here; broader 128-thread use hits cached-kernel resource limits.
    if (residual == nullptr && residual_out == nullptr && rstd == nullptr &&
        weight != nullptr && bias == nullptr &&
        (options.N == 8192 || options.N == 32768)) {
      threads_per_row = threads;
    } else if (residual == nullptr && residual_out == nullptr && rstd == nullptr &&
        weight != nullptr && bias == nullptr &&
        std::is_same_v<X, __half> && options.N == 256) {
      // The smallest fp16 no-residual row is launch-overhead sensitive under
      // pooled inputs; two rows per CTA improved the full forward section and
      // balanced provider-order check over the previous wider grouping.
      threads = 64;
      threads_per_row = 32;
    } else if (residual != nullptr && residual_out != nullptr &&
        std::is_same_v<X, RO> && options.N == 2048) {
      threads_per_row = 64;
    } else if (residual != nullptr && residual_out != nullptr &&
               std::is_same_v<X, RO> && options.N == 8192) {
      threads = 1024;
      threads_per_row = 512;
    } else if (residual != nullptr && residual_out != nullptr &&
               options.N == 8192) {
      threads_per_row = 128;
    } else if (residual != nullptr && residual_out != nullptr &&
               options.N == 16384) {
      threads_per_row = 256;
    } else if constexpr (!std::is_same_v<X, RO>) {
      if (residual != nullptr && residual_out != nullptr &&
          options.N == 512) {
        // bf16 mixed residual_out at N=512 is launch-overhead sensitive; two
        // rows per CTA improves the pooled row while fp16 prefers one row.
        threads = std::is_same_v<X, __nv_bfloat16> ? 128 : 64;
        threads_per_row = 64;
      } else if (residual != nullptr && residual_out != nullptr &&
          options.N == 1024) {
        threads_per_row = 128;
      } else if (residual != nullptr && residual_out != nullptr &&
                 options.N == 2048) {
        // Mixed fp32 residual_out at 2048 is register-pressure sensitive; the
        // 256-thread one-row vec4 path improves the pooled benchmark slightly
        // for both fp16 and bf16.
        threads = 512;
        threads_per_row = 256;
      } else if (residual != nullptr && residual_out != nullptr &&
                 options.N == 4096) {
        threads = 1024;
        threads_per_row = 512;
      } else if (residual != nullptr && residual_out != nullptr &&
                 options.N == 131072) {
        threads = 256;
        threads_per_row = 256;
      } else if (residual != nullptr && residual_out != nullptr &&
                 std::is_same_v<X, __half> && options.N == 262144) {
        // Mixed fp32 residual_out has a much heavier store footprint; using
        // 512 threads selects the kVecs=4 clustered shape and relieves register
        // pressure in the large-row tail.
        threads = 512;
        threads_per_row = 512;
      } else if (residual != nullptr && residual_out != nullptr &&
                 std::is_same_v<X, __nv_bfloat16> &&
                 options.N == 262144) {
        threads = 512;
        threads_per_row = 512;
      }
    } else if constexpr (std::is_same_v<X, RO> &&
                         std::is_same_v<X, __half>) {
      if (residual == nullptr && residual_out == nullptr && rstd == nullptr &&
          weight != nullptr && bias == nullptr && options.N == 262144) {
        // The large fp16 no-residual row is register-pressure sensitive; the
        // 512-thread clustered shape cuts per-thread work from kVecs=8 to 4.
        threads = 512;
        threads_per_row = 512;
      } else if (residual != nullptr && residual_out != nullptr &&
                 rstd == nullptr && weight != nullptr && bias == nullptr &&
                 options.N == 262144) {
        // Same-dtype residual has the same large-row register pressure as the
        // no-residual fp16 path, plus an extra residual_out store.
        threads = 512;
        threads_per_row = 512;
      }
    } else if constexpr (std::is_same_v<X, RO> &&
                         std::is_same_v<X, __nv_bfloat16>) {
      if (residual != nullptr && residual_out != nullptr && rstd == nullptr &&
          weight != nullptr && bias == nullptr && options.N == 262144) {
        // Match the fp16 same-dtype residual shape for the bf16 large tail.
        threads = 512;
        threads_per_row = 512;
      }
    }
  } else if constexpr (std::is_same_v<X, RO>) {
    // fp32 residual exact rows prefer shape-specific thread counts under the
    // pooled benchmark harness: small rows need fewer CTA rows, while 4096 and
    // 16384 benefit from wider one-row CTAs.
    if (residual != nullptr && residual_out != nullptr && rstd == nullptr &&
        weight != nullptr && bias == nullptr &&
        (options.N == 256 || options.N == 512)) {
      threads = 64;
      threads_per_row = 64;
    } else if (residual != nullptr && residual_out != nullptr && rstd == nullptr &&
        weight != nullptr && bias == nullptr && options.N == 2048) {
      // fp32 residual rows at 2048 are faster with lower per-thread work than
      // the default 32-thread-per-row exact shape.
      threads = 256;
      threads_per_row = 256;
    } else if (residual != nullptr && residual_out != nullptr &&
               rstd == nullptr && weight != nullptr && bias == nullptr &&
               options.N == 4096) {
      threads = 512;
      threads_per_row = 512;
    } else if (residual != nullptr && residual_out != nullptr &&
               rstd == nullptr && weight != nullptr && bias == nullptr &&
               options.N == 16384) {
      threads = 1024;
      threads_per_row = 1024;
    }
    if (residual == nullptr && residual_out == nullptr && rstd == nullptr &&
        weight != nullptr && bias == nullptr) {
      if (options.N == 256) {
        // Three rows per CTA was the best measured fp32 small-row compromise:
        // raw-order win with a smaller balanced-order gap than the wider groups.
        threads = 96;
        threads_per_row = 32;
      } else if (options.N == 1024) {
        threads = 32;
        threads_per_row = 32;
      } else if (options.N == 16384) {
        threads = 1024;
        threads_per_row = 1024;
      } else if (options.N == 262144) {
        threads = 512;
        threads_per_row = 512;
      }
    }
  }
  using XVec = Vec128<X>;
  int vecs = options.N / XVec::kLanes;
  int vecs_per_thread = (vecs + threads_per_row - 1) / threads_per_row;
  int rows_per_cta = std::max(1, threads / threads_per_row);
  dim3 grid((options.M + rows_per_cta - 1) / rows_per_cta);
  dim3 block(threads);
  if constexpr (std::is_same_v<X, RO>) {
    if (weight != nullptr && bias == nullptr && residual == nullptr &&
        residual_out == nullptr && rstd == nullptr && rows_per_cta > 1 &&
        threads_per_row == 32 &&
        vecs == vecs_per_thread * threads_per_row) {
      if (vecs_per_thread == 1) {
        if (options.H == 1) {
          rmsnorm_fwd_weight_multirow_full_kernel<X, 1, true>
              <<<grid, block, 0, stream>>>(out, x, weight, options.M,
                                           options.N, options.H, options.eps);
        } else {
          rmsnorm_fwd_weight_multirow_full_kernel<X, 1>
              <<<grid, block, 0, stream>>>(out, x, weight, options.M,
                                           options.N, options.H, options.eps);
        }
        return;
      }
      if (vecs_per_thread == 2) {
        if (options.H == 1) {
          rmsnorm_fwd_weight_multirow_full_kernel<X, 2, true>
              <<<grid, block, 0, stream>>>(out, x, weight, options.M,
                                           options.N, options.H, options.eps);
        } else {
          rmsnorm_fwd_weight_multirow_full_kernel<X, 2>
              <<<grid, block, 0, stream>>>(out, x, weight, options.M,
                                           options.N, options.H, options.eps);
        }
        return;
      }
      if (vecs_per_thread == 4) {
        if (options.H == 1) {
          rmsnorm_fwd_weight_multirow_full_kernel<X, 4, true>
              <<<grid, block, 0, stream>>>(out, x, weight, options.M,
                                           options.N, options.H, options.eps);
        } else {
          rmsnorm_fwd_weight_multirow_full_kernel<X, 4>
              <<<grid, block, 0, stream>>>(out, x, weight, options.M,
                                           options.N, options.H, options.eps);
        }
        return;
      }
      if (vecs_per_thread == 8) {
        if (options.H == 1) {
          rmsnorm_fwd_weight_multirow_full_kernel<X, 8, true>
              <<<grid, block, 0, stream>>>(out, x, weight, options.M,
                                           options.N, options.H, options.eps);
        } else {
          rmsnorm_fwd_weight_multirow_full_kernel<X, 8>
              <<<grid, block, 0, stream>>>(out, x, weight, options.M,
                                           options.N, options.H, options.eps);
        }
        return;
      }
      if (vecs_per_thread == 16) {
        if (options.H == 1) {
          rmsnorm_fwd_weight_multirow_full_kernel<X, 16, true>
              <<<grid, block, 0, stream>>>(out, x, weight, options.M,
                                           options.N, options.H, options.eps);
        } else {
          rmsnorm_fwd_weight_multirow_full_kernel<X, 16>
              <<<grid, block, 0, stream>>>(out, x, weight, options.M,
                                           options.N, options.H, options.eps);
        }
        return;
      }
    }
  }
  if (weight != nullptr && bias == nullptr && residual != nullptr &&
      residual_out != nullptr && rstd == nullptr && options.N <= 32768 &&
      // The fp32 exact residual tile loses to the generic cached path at
      // this pooled mid-row shape; keep the exact path for the other rows
      // where it still wins.
      !(std::is_same_v<X, float> && options.N == 8192) &&
      vecs == vecs_per_thread * threads_per_row) {
    bool exact_residual_assume_h1 =
        options.H == 1 &&
        (std::is_same_v<X, RO> ||
         (!std::is_same_v<X, RO> &&
          (options.N == 512 || options.N == 1024 || options.N == 2048 ||
           options.N == 4096 || options.N == 8192)));
    // Exact residual rows avoid the generic nullable/predicated cached kernel
    // and vectorize both residual_out and out stores.
    if (threads_per_row == 32 && vecs_per_thread == 1) {
      launch_fwd_residual_multirow_full<X, RO, 1, 32>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 32 && vecs_per_thread == 2) {
      launch_fwd_residual_multirow_full<X, RO, 2, 32>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 32 && vecs_per_thread == 4) {
      launch_fwd_residual_multirow_full<X, RO, 4, 32>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 32 && vecs_per_thread == 8) {
      launch_fwd_residual_multirow_full<X, RO, 8, 32>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 32 && vecs_per_thread == 16) {
      launch_fwd_residual_multirow_full<X, RO, 16, 32>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 64 && vecs_per_thread == 2) {
      launch_fwd_residual_multirow_full<X, RO, 2, 64>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if constexpr (!std::is_same_v<X, RO> && std::is_same_v<RO, float> &&
                  sizeof(X) < 4) {
      // Keep the smallest mixed residual_out row on the same four-lane
      // vectorization used by the wider mixed rows.
      if (options.N == 512) {
        launch_fwd_residual_mixed_vec4<X, 2, 64>(
            options.H == 1, grid, block, stream, out, residual_out, x,
            residual, weight, options.M, options.N, options.H, options.eps);
        return;
      }
    }
    if (threads_per_row == 64 && vecs_per_thread == 1) {
      launch_fwd_residual_multirow_full<X, RO, 1, 64,
                                        !std::is_same_v<X, RO> &&
                                            std::is_same_v<X, __nv_bfloat16>,
                                        !std::is_same_v<X, RO> &&
                                            std::is_same_v<X, __half>>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 64 && vecs_per_thread == 4) {
      launch_fwd_residual_multirow_full<X, RO, 4, 64>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if constexpr (!std::is_same_v<X, RO> && std::is_same_v<RO, float> &&
                  sizeof(X) < 4) {
      // Mixed fp32 residual_out rows are limited by the wider residual store.
      // Use four input lanes per thread, matching the fp32 store width, instead
      // of the normal eight-lane half/bf16 vectorization.
      if (options.N == 1024) {
        launch_fwd_residual_mixed_vec4<X, 2, 128>(
            options.H == 1, grid, block, stream, out, residual_out, x,
            residual, weight, options.M, options.N, options.H, options.eps);
        return;
      }
      if (options.N == 2048) {
        dim3 vec4_grid(options.M);
        dim3 vec4_block(256);
        launch_fwd_residual_mixed_vec4<X, 2, 256>(
            options.H == 1, vec4_grid, vec4_block, stream, out, residual_out,
            x, residual, weight, options.M, options.N, options.H, options.eps);
        return;
      }
      if (options.N == 4096) {
        dim3 vec4_grid(options.M);
        dim3 vec4_block(512);
        launch_fwd_residual_mixed_vec4<X, 2, 512>(
            options.H == 1, vec4_grid, vec4_block, stream, out, residual_out,
            x, residual, weight, options.M, options.N, options.H, options.eps);
        return;
      }
      if (options.N == 8192) {
        dim3 vec4_grid(options.M);
        dim3 vec4_block(1024);
        launch_fwd_residual_mixed_vec4<X, 2, 1024>(
            options.H == 1, vec4_grid, vec4_block, stream, out, residual_out,
            x, residual, weight, options.M, options.N, options.H, options.eps);
        return;
      }
    }
    if (threads_per_row == 128 && vecs_per_thread == 8) {
      launch_fwd_residual_multirow_full<X, RO, 8, 128>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 128 && vecs_per_thread == 1) {
      launch_fwd_residual_multirow_full<X, RO, 1, 128,
                                        !std::is_same_v<X, RO>,
                                        !std::is_same_v<X, RO>>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 128 && vecs_per_thread == 4) {
      launch_fwd_residual_multirow_full<X, RO, 4, 128>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 128 && vecs_per_thread == 2) {
      launch_fwd_residual_multirow_full<X, RO, 2, 128>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 256 && vecs_per_thread == 2) {
      launch_fwd_residual_multirow_full<X, RO, 2, 256>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 256 && vecs_per_thread == 1) {
      launch_fwd_residual_multirow_full<X, RO, 1, 256,
                                        !std::is_same_v<X, RO>>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 256 && vecs_per_thread == 4) {
      launch_fwd_residual_multirow_full<X, RO, 4, 256>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 256 && vecs_per_thread == 8) {
      launch_fwd_residual_multirow_full<X, RO, 8, 256>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 512 && vecs_per_thread == 4) {
      launch_fwd_residual_multirow_full<X, RO, 4, 512>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 512 && vecs_per_thread == 2) {
      launch_fwd_residual_multirow_full<X, RO, 2, 512>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 512 && vecs_per_thread == 1) {
      launch_fwd_residual_multirow_full<X, RO, 1, 512,
                                        !std::is_same_v<X, RO>,
                                        !std::is_same_v<X, RO>>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 512 && vecs_per_thread == 8) {
      launch_fwd_residual_multirow_full<X, RO, 8, 512>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 1024 && vecs_per_thread == 4) {
      // Mixed fp32 residual_out at N=32768 is faster when the residual store
      // happens during the sum pass; other exact shapes regressed with it.
      launch_fwd_residual_multirow_full<X, RO, 4, 1024,
                                        !std::is_same_v<X, RO>>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
    if (threads_per_row == 1024 && vecs_per_thread == 8) {
      launch_fwd_residual_multirow_full<X, RO, 8, 1024>(
          exact_residual_assume_h1, grid, block, stream, out, residual_out, x,
          residual, weight, options.M, options.N, options.H, options.eps);
      return;
    }
  }
  if (weight != nullptr && bias == nullptr && residual != nullptr &&
      residual_out != nullptr && rstd == nullptr && rows_per_cta == 1) {
    int parts = fwd_split_parts<X>(options.N);
    if constexpr (!std::is_same_v<X, RO> && sizeof(X) < 4) {
      if (options.N == 65536) {
        parts = 4;
      } else if (options.N == 131072) {
        parts = 8;
      }
    }
    if (parts > 1 && partial_sums != nullptr) {
      using ClusterResidualKernel = void (*)(X*, RO*, X const*, X const*,
                                             float const*, int, int, int,
                                             float);
      int vecs_per_part = (vecs + parts - 1) / parts;
      cudaLaunchAttribute attrs[1]{};
      attrs[0].id = cudaLaunchAttributeClusterDimension;
      attrs[0].val.clusterDim.x = 1;
      attrs[0].val.clusterDim.y = parts;
      attrs[0].val.clusterDim.z = 1;
      cudaLaunchConfig_t config{};
      config.gridDim = dim3(options.M, parts, 1);
      config.blockDim = block;
      config.dynamicSmemBytes = 0;
      config.stream = stream;
      config.attrs = attrs;
      config.numAttrs = 1;
      if (vecs_per_part == 4 * threads && parts == 8) {
        ClusterResidualKernel kernel = nullptr;
        if constexpr (!std::is_same_v<X, RO> && sizeof(X) < 4) {
          if (options.H == 1 && std::is_same_v<X, __nv_bfloat16>) {
            kernel = rmsnorm_fwd_cluster_residual_reg_full_kernel<
                X, RO, 8, 4, true>;
            static std::once_flag attr_once_h1;
            std::call_once(attr_once_h1, [&]() {
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            });
          } else {
            kernel = rmsnorm_fwd_cluster_residual_reg_full_kernel<X, RO, 8, 4>;
            static std::once_flag attr_once;
            std::call_once(attr_once, [&]() {
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            });
          }
        } else {
          kernel = rmsnorm_fwd_cluster_residual_reg_full_kernel<X, RO, 8, 4>;
          static std::once_flag attr_once;
          std::call_once(attr_once, [&]() {
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
          });
        }
        if constexpr (sizeof(X) < 4) {
          config.dynamicSmemBytes = kHalfRegClusterCarveoutBytes;
        }
        CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, residual_out, x,
                                      residual, weight, options.M, options.N,
                                      options.H, options.eps));
        return;
      }
      if (vecs_per_part == 4 * threads && parts == 16) {
        if constexpr (!std::is_same_v<X, RO> &&
                      (std::is_same_v<X, __half> ||
                       std::is_same_v<X, __nv_bfloat16>)) {
          if (options.N == 262144) {
            // Mixed fp32 residual_out at the largest row is register-pressure
            // bound. Keep x/residual in shared memory across the cluster
            // reduction to reduce live vector state during the barrier.
            ClusterResidualKernel kernel =
                rmsnorm_fwd_cluster_residual_smem_full_kernel<
                    X, RO, 16, 4, false, true>;
            int residual_smem_bytes =
                2 * vecs_per_part * XVec::kLanes * static_cast<int>(sizeof(X));
            static std::once_flag attr_once_smem_mixed;
            std::call_once(attr_once_smem_mixed, [&]() {
              if (residual_smem_bytes >= 48 * 1024) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    residual_smem_bytes));
              }
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            });
            config.dynamicSmemBytes = residual_smem_bytes;
            CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, residual_out,
                                          x, residual, weight, options.M,
                                          options.N, options.H, options.eps));
            return;
          }
        }
        ClusterResidualKernel kernel =
            rmsnorm_fwd_cluster_residual_reg_full_kernel<X, RO, 16, 4>;
        static std::once_flag attr_once;
        std::call_once(attr_once, [&]() {
          CUDA_CHECK(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        });
        if constexpr (sizeof(X) < 4) {
          config.dynamicSmemBytes = kHalfRegClusterCarveoutBytes;
        }
        CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, residual_out, x,
                                      residual, weight, options.M, options.N,
                                      options.H, options.eps));
        return;
      }
      if (vecs_per_part == 8 * threads && parts == 16) {
        ClusterResidualKernel kernel =
            rmsnorm_fwd_cluster_residual_reg_full_kernel<X, RO, 16, 8>;
        static std::once_flag attr_once;
        std::call_once(attr_once, [&]() {
          CUDA_CHECK(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        });
        if constexpr (sizeof(X) < 4) {
          config.dynamicSmemBytes = kHalfRegClusterCarveoutBytes;
        }
        CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, residual_out, x,
                                      residual, weight, options.M, options.N,
                                      options.H, options.eps));
        return;
      }
      if (vecs_per_part == 8 * threads && parts == 4) {
        ClusterResidualKernel kernel = nullptr;
        if constexpr (!std::is_same_v<X, RO> && sizeof(X) < 4) {
          if (options.N == 65536) {
            // Mixed fp32 residual_out at 64K is register-pressure bound. A
            // four-way split with x/residual reloaded from shared memory trims
            // the pooled gap to QuACK without disturbing same-dtype residual.
            kernel = options.H == 1
                         ? rmsnorm_fwd_cluster_residual_smem_full_kernel<
                               X, RO, 4, 8, true, true, true>
                         : rmsnorm_fwd_cluster_residual_smem_full_kernel<
                               X, RO, 4, 8, false, true, true>;
            int residual_smem_bytes =
                2 * vecs_per_part * XVec::kLanes * static_cast<int>(sizeof(X));
            static std::once_flag attr_once_smem_mixed;
            std::call_once(attr_once_smem_mixed, [&]() {
              if (residual_smem_bytes >= 48 * 1024) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    residual_smem_bytes));
              }
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            });
            config.dynamicSmemBytes = residual_smem_bytes;
            CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, residual_out,
                                          x, residual, weight, options.M,
                                          options.N, options.H, options.eps));
            return;
          }
          if (options.H == 1) {
            kernel =
                rmsnorm_fwd_cluster_residual_reg_full_kernel<X, RO, 4, 8, true>;
            static std::once_flag attr_once_h1;
            std::call_once(attr_once_h1, [&]() {
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            });
          } else {
            kernel = rmsnorm_fwd_cluster_residual_reg_full_kernel<X, RO, 4, 8>;
            static std::once_flag attr_once;
            std::call_once(attr_once, [&]() {
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            });
          }
        } else {
          kernel = rmsnorm_fwd_cluster_residual_reg_full_kernel<X, RO, 4, 8>;
          static std::once_flag attr_once;
          std::call_once(attr_once, [&]() {
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
          });
        }
        if constexpr (sizeof(X) < 4) {
          config.dynamicSmemBytes = kHalfRegClusterCarveoutBytes;
        }
        CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, residual_out, x,
                                      residual, weight, options.M, options.N,
                                      options.H, options.eps));
        return;
      }
      if (vecs_per_part == 8 * threads && parts == 8) {
        if constexpr (!std::is_same_v<X, RO> &&
                      (std::is_same_v<X, __half> ||
                       std::is_same_v<X, __nv_bfloat16>)) {
          if (options.N == 131072) {
            ClusterResidualKernel kernel =
                options.H == 1
                    ? rmsnorm_fwd_cluster_residual_smem_full_kernel<
                          X, RO, 8, 8, true, true>
                    : rmsnorm_fwd_cluster_residual_smem_full_kernel<
                          X, RO, 8, 8, false, true>;
            int residual_smem_bytes =
                2 * vecs_per_part * XVec::kLanes * static_cast<int>(sizeof(X));
            static std::once_flag attr_once_smem_mixed;
            std::call_once(attr_once_smem_mixed, [&]() {
              if (residual_smem_bytes >= 48 * 1024) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    residual_smem_bytes));
              }
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            });
            config.dynamicSmemBytes = residual_smem_bytes;
            CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, residual_out,
                                          x, residual, weight, options.M,
                                          options.N, options.H, options.eps));
            return;
          }
        }
        ClusterResidualKernel kernel =
            rmsnorm_fwd_cluster_residual_reg_full_kernel<X, RO, 8, 8>;
        static std::once_flag attr_once;
        std::call_once(attr_once, [&]() {
          CUDA_CHECK(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        });
        if constexpr (sizeof(X) < 4) {
          config.dynamicSmemBytes = kHalfRegClusterCarveoutBytes;
        }
        CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, residual_out, x,
                                      residual, weight, options.M, options.N,
                                      options.H, options.eps));
        return;
      }
      if (vecs_per_part == 16 * threads && parts == 8) {
        ClusterResidualKernel kernel =
            rmsnorm_fwd_cluster_residual_reg_full_kernel<X, RO, 8, 16>;
        static std::once_flag attr_once;
        std::call_once(attr_once, [&]() {
          CUDA_CHECK(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        });
        if constexpr (sizeof(X) >= 4) {
          config.dynamicSmemBytes = kRegClusterCarveoutBytes;
        }
        CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, residual_out, x,
                                      residual, weight, options.M, options.N,
                                      options.H, options.eps));
        return;
      }
      if (vecs_per_part == 16 * threads && parts == 16) {
        ClusterResidualKernel kernel =
            rmsnorm_fwd_cluster_residual_reg_full_kernel<X, RO, 16, 16>;
        static std::once_flag attr_once;
        std::call_once(attr_once, [&]() {
          CUDA_CHECK(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        });
        if constexpr (sizeof(X) >= 4) {
          config.dynamicSmemBytes = kRegClusterCarveoutBytes;
        }
        CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, residual_out, x,
                                      residual, weight, options.M, options.N,
                                      options.H, options.eps));
        return;
      }
    }
  }
  if constexpr (std::is_same_v<X, RO>) {
    if (weight != nullptr && bias == nullptr && residual == nullptr &&
        residual_out == nullptr && rstd == nullptr && rows_per_cta == 1) {
      int parts = fwd_split_parts<X>(options.N);
      if constexpr (std::is_same_v<X, __half>) {
        if (options.N == 65536) {
          parts = 4;
        }
      }
      if constexpr (sizeof(X) >= 4) {
        if (options.N == 262144) {
          // The largest fp32 no-residual row uses a shared-memory cp.async
          // cluster specialization below; a 16-way split keeps each staged
          // tile small enough to avoid the register-cluster pressure.
          parts = 16;
        }
      }
      if (vecs == 16 * threads && vecs_per_thread == 16 &&
          options.N <= 32768) {
        rmsnorm_fwd_weight_full_kernel<X, 16>
            <<<options.M, block, 0, stream>>>(out, x, weight, options.M,
                                              options.N, options.H,
                                              options.eps);
        return;
      }
      if (parts > 1 && partial_sums != nullptr) {
        using ClusterKernel = void (*)(X*, X const*, float const*, int, int,
                                       int, int, float);
        using ClusterFullKernel = void (*)(X*, X const*, float const*, int,
                                           int, int, float);
        int vecs_per_part = (vecs + parts - 1) / parts;
        int smem_bytes = vecs_per_part * XVec::kLanes * sizeof(X);
        cudaLaunchAttribute attrs[1]{};
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = 1;
        attrs[0].val.clusterDim.y = parts;
        attrs[0].val.clusterDim.z = 1;
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(options.M, parts, 1);
        config.blockDim = block;
        config.dynamicSmemBytes = smem_bytes;
        config.stream = stream;
        config.attrs = attrs;
        config.numAttrs = 1;
        if (vecs_per_part == 8 * threads && parts == 2) {
          ClusterFullKernel kernel = rmsnorm_fwd_cluster_mbar_full_kernel<X, 2, 8>;
          static std::once_flag attr_once;
          std::call_once(attr_once, [&]() {
            if (smem_bytes >= 48 * 1024) {
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                  smem_bytes));
            }
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
          });
          CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, x, weight,
                                        options.M, options.N, options.H,
                                        options.eps));
        } else if (vecs_per_part == 4 * threads && parts == 2) {
          if (options.H == 1) {
            ClusterFullKernel kernel =
                rmsnorm_fwd_cluster_mbar_full_kernel<X, 2, 4, true>;
            static std::once_flag attr_once_h1;
            std::call_once(attr_once_h1, [&]() {
              if (smem_bytes >= 48 * 1024) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem_bytes));
              }
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            });
            CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, x, weight,
                                          options.M, options.N, options.H,
                                          options.eps));
          } else {
            ClusterFullKernel kernel =
                rmsnorm_fwd_cluster_mbar_full_kernel<X, 2, 4>;
            static std::once_flag attr_once;
            std::call_once(attr_once, [&]() {
              if (smem_bytes >= 48 * 1024) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem_bytes));
              }
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            });
            CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, x, weight,
                                          options.M, options.N, options.H,
                                          options.eps));
          }
	        } else if (vecs_per_part == 8 * threads && parts == 4) {
          if constexpr (sizeof(X) < 4) {
            ClusterFullKernel kernel =
                rmsnorm_fwd_cluster_reg_full_kernel<X, 4, 8>;
            static std::once_flag attr_once;
            std::call_once(attr_once, [&]() {
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            });
            config.dynamicSmemBytes = 0;
            CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, x, weight,
                                          options.M, options.N, options.H,
                                          options.eps));
          } else {
            ClusterFullKernel kernel = rmsnorm_fwd_cluster_full_kernel<X, 4, 8>;
            static std::once_flag attr_once;
            std::call_once(attr_once, [&]() {
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
              if (smem_bytes >= 48 * 1024) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem_bytes));
              }
            });
	            CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, x, weight,
	                                          options.M, options.N, options.H,
	                                          options.eps));
	          }
	        } else if (vecs_per_part == 8 * threads && parts == 8) {
          ClusterFullKernel kernel = rmsnorm_fwd_cluster_mbar_full_kernel<X, 8, 8>;
          static std::once_flag attr_once;
          std::call_once(attr_once, [&]() {
            if (smem_bytes >= 48 * 1024) {
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                  smem_bytes));
            }
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
          });
          CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, x, weight,
                                        options.M, options.N, options.H,
                                        options.eps));
        } else if (vecs_per_part == 4 * threads && parts == 8) {
          if (options.H == 1) {
            ClusterFullKernel kernel =
                rmsnorm_fwd_cluster_mbar_full_kernel<X, 8, 4, true>;
            static std::once_flag attr_once_h1;
            std::call_once(attr_once_h1, [&]() {
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            });
            CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, x, weight,
                                          options.M, options.N, options.H,
                                          options.eps));
          } else {
            ClusterFullKernel kernel = rmsnorm_fwd_cluster_mbar_full_kernel<X, 8, 4>;
            static std::once_flag attr_once;
            std::call_once(attr_once, [&]() {
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            });
            CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, x, weight,
                                          options.M, options.N, options.H,
                                          options.eps));
          }
        } else if (vecs_per_part == 16 * threads && parts == 8) {
          ClusterFullKernel kernel = nullptr;
          bool use_cpasync_weight = false;
          if (options.H == 1) {
            if constexpr (std::is_same_v<X, float>) {
              if (options.N == 131072) {
                kernel =
                    rmsnorm_fwd_cluster_mbar_cpasync_weight_256_full_kernel<
                        X, 8, 16, true>;
                static std::once_flag attr_once_h1_cpasync_weight_128k;
                std::call_once(attr_once_h1_cpasync_weight_128k, [&]() {
                  CUDA_CHECK(cudaFuncSetAttribute(
                      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                      smem_bytes));
                  CUDA_CHECK(cudaFuncSetAttribute(
                      kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
                });
                config.dynamicSmemBytes = smem_bytes;
                use_cpasync_weight = true;
              } else {
                kernel = rmsnorm_fwd_cluster_reg_full_kernel<X, 8, 16, true>;
                static std::once_flag attr_once_h1;
                std::call_once(attr_once_h1, [&]() {
                  CUDA_CHECK(cudaFuncSetAttribute(
                      kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
                });
              }
            } else {
              kernel = rmsnorm_fwd_cluster_reg_full_kernel<X, 8, 16, true>;
              static std::once_flag attr_once_h1;
              std::call_once(attr_once_h1, [&]() {
                CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
              });
            }
          } else {
            kernel = rmsnorm_fwd_cluster_reg_full_kernel<X, 8, 16>;
            static std::once_flag attr_once;
            std::call_once(attr_once, [&]() {
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            });
          }
          if constexpr (sizeof(X) >= 4) {
            if (!use_cpasync_weight) {
              config.dynamicSmemBytes = kRegClusterCarveoutBytes;
            }
          }
          CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, x, weight,
                                        options.M, options.N, options.H,
                                        options.eps));
        } else if (vecs_per_part == 8 * threads && parts == 16) {
          ClusterFullKernel kernel = nullptr;
          bool use_cpasync_weight = false;
          if constexpr (std::is_same_v<X, float>) {
            if (options.H == 1 && options.N == 262144) {
              kernel = rmsnorm_fwd_cluster_mbar_cpasync_weight_full_kernel<
                  X, 16, 8, true>;
              static std::once_flag attr_once_h1_cpasync_weight_16way;
              std::call_once(attr_once_h1_cpasync_weight_16way, [&]() {
                CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    64 * 1024));
                CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
              });
              config.dynamicSmemBytes = smem_bytes;
              use_cpasync_weight = true;
            } else {
              kernel = rmsnorm_fwd_cluster_reg_full_kernel<X, 16, 8>;
            }
          } else {
            kernel = rmsnorm_fwd_cluster_reg_full_kernel<X, 16, 8>;
          }
          if (!use_cpasync_weight) {
            static std::once_flag attr_once;
            std::call_once(attr_once, [&]() {
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            });
            if constexpr (sizeof(X) < 4) {
              config.dynamicSmemBytes = kHalfRegClusterCarveoutBytes;
            }
          }
          CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, x, weight,
                                        options.M, options.N, options.H,
                                        options.eps));
        } else if (vecs_per_part == 16 * threads && parts == 16) {
          ClusterFullKernel kernel = rmsnorm_fwd_cluster_reg_full_kernel<X, 16, 16>;
          static std::once_flag attr_once;
          std::call_once(attr_once, [&]() {
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
          });
          if constexpr (sizeof(X) >= 4) {
            config.dynamicSmemBytes = kRegClusterCarveoutBytes;
          }
          CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, x, weight,
                                        options.M, options.N, options.H,
                                        options.eps));
        } else if (vecs_per_part == 4 * threads && parts == 16) {
          ClusterFullKernel kernel = nullptr;
          if constexpr (sizeof(X) < 4) {
            if (options.H == 1) {
              kernel =
                  rmsnorm_fwd_cluster_mbar_cpasync_full_kernel<X, 16, 4, true>;
              static std::once_flag attr_once_h1;
              std::call_once(attr_once_h1, [&]() {
                CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeNonPortableClusterSizeAllowed,
                    1));
              });
            } else {
              kernel = rmsnorm_fwd_cluster_mbar_cpasync_full_kernel<X, 16, 4>;
              static std::once_flag attr_once;
              std::call_once(attr_once, [&]() {
                CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeNonPortableClusterSizeAllowed,
                    1));
              });
            }
          } else {
            if (options.H == 1) {
              kernel = rmsnorm_fwd_cluster_reg_full_kernel<X, 16, 4, true>;
              static std::once_flag attr_once_h1;
              std::call_once(attr_once_h1, [&]() {
                CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
              });
            } else {
              kernel = rmsnorm_fwd_cluster_reg_full_kernel<X, 16, 4>;
              static std::once_flag attr_once;
              std::call_once(attr_once, [&]() {
                CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
              });
            }
            config.dynamicSmemBytes = 0;
          }
          CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, x, weight,
                                        options.M, options.N, options.H,
                                        options.eps));
        } else {
          ClusterKernel kernel = rmsnorm_fwd_cluster_kernel<X>;
          static std::once_flag attr_once;
          std::call_once(attr_once, [&]() {
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            if (smem_bytes >= 48 * 1024) {
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                  smem_bytes));
            }
          });
          CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, out, x, weight,
                                        options.M, options.N, options.H, parts,
                                        options.eps));
        }
        return;
      }
      if (vecs == vecs_per_thread * threads && vecs_per_thread == 1) {
        rmsnorm_fwd_weight_full_kernel<X, 1>
            <<<options.M, block, 0, stream>>>(out, x, weight, options.M,
                                              options.N, options.H,
                                              options.eps);
        return;
      }
      if (vecs == vecs_per_thread * threads && vecs_per_thread == 2) {
        if (options.H == 1 && options.N == 4096) {
          rmsnorm_fwd_weight_full_kernel<X, 2, true, true>
              <<<options.M, block, 0, stream>>>(out, x, weight, options.M,
                                                options.N, options.H,
                                                options.eps);
        } else if (options.N == 4096) {
          rmsnorm_fwd_weight_full_kernel<X, 2, true, false>
              <<<options.M, block, 0, stream>>>(out, x, weight, options.M,
                                                options.N, options.H,
                                                options.eps);
        } else if (options.N == 8192) {
          rmsnorm_fwd_weight_cpasync_full_kernel<X, 2, 512>
              <<<options.M, block, options.N * sizeof(X), stream>>>(
                  out, x, weight, options.M, options.N, options.H, options.eps);
        } else {
          rmsnorm_fwd_weight_full_kernel<X, 2>
              <<<options.M, block, 0, stream>>>(out, x, weight, options.M,
                                                options.N, options.H,
                                                options.eps);
        }
        return;
      }
      if (vecs == vecs_per_thread * threads && vecs_per_thread == 4) {
        if constexpr (std::is_same_v<X, float>) {
          if (options.H == 1 && options.N == 8192) {
            rmsnorm_fwd_weight_full_kernel<X, 4, true, true>
                <<<options.M, block, 0, stream>>>(out, x, weight, options.M,
                                                  options.N, options.H,
                                                  options.eps);
          } else {
            rmsnorm_fwd_weight_full_kernel<X, 4>
                <<<options.M, block, 0, stream>>>(out, x, weight, options.M,
                                                  options.N, options.H,
                                                  options.eps);
          }
        } else {
          rmsnorm_fwd_weight_full_kernel<X, 4>
              <<<options.M, block, 0, stream>>>(out, x, weight, options.M,
                                                options.N, options.H,
                                                options.eps);
        }
        return;
      }
      if (vecs == vecs_per_thread * threads && vecs_per_thread == 8) {
        if (options.H == 1) {
          rmsnorm_fwd_weight_full_kernel<X, 8, false, true>
              <<<options.M, block, 0, stream>>>(out, x, weight, options.M,
                                                options.N, options.H,
                                                options.eps);
        } else {
          rmsnorm_fwd_weight_full_kernel<X, 8>
              <<<options.M, block, 0, stream>>>(out, x, weight, options.M,
                                                options.N, options.H,
                                                options.eps);
        }
        return;
      }
      if (vecs == vecs_per_thread * threads && vecs_per_thread == 16 &&
          options.N <= 16384) {
        rmsnorm_fwd_weight_full_kernel<X, 16>
            <<<options.M, block, 0, stream>>>(out, x, weight, options.M,
                                              options.N, options.H,
                                              options.eps);
        return;
      }
      if (vecs_per_thread <= 4) {
        rmsnorm_fwd_weight_cached_kernel<X, 4>
            <<<options.M, block, 0, stream>>>(out, x, weight, options.M,
                                              options.N, options.H,
                                              options.eps);
        return;
      }
      if (vecs_per_thread <= 8) {
        rmsnorm_fwd_weight_cached_kernel<X, 8>
            <<<options.M, block, 0, stream>>>(out, x, weight, options.M,
                                              options.N, options.H,
                                              options.eps);
        return;
      }
      if (vecs_per_thread <= 16 && options.N <= 16384) {
        rmsnorm_fwd_weight_cached_kernel<X, 16>
            <<<options.M, block, 0, stream>>>(out, x, weight, options.M,
                                              options.N, options.H,
                                              options.eps);
        return;
      }
      rmsnorm_fwd_streaming_kernel<X, RO>
          <<<options.M, block, 0, stream>>>(out, nullptr, nullptr, x, weight,
                                            nullptr, nullptr, options.M,
                                            options.N, options.H,
                                            threads_per_row, options.eps);
      return;
    }
  }
  if ((residual_out != nullptr || rstd != nullptr) && options.N == 32768) {
    // The 1024-thread cached shape is register-heavy at this row width. The
    // streaming path is used for residual_out and rstd-producing launches,
    // while the no-rstd benchmark path stays on the specialized kernels above.
    rmsnorm_fwd_streaming_kernel<X, RO>
        <<<grid, block, 0, stream>>>(out, residual_out, rstd, x, weight, bias,
                                     residual, options.M, options.N, options.H,
                                     threads_per_row, options.eps);
    return;
  }
  if (vecs_per_thread <= 4) {
    rmsnorm_fwd_cached_kernel<X, RO, 4>
        <<<grid, block, 0, stream>>>(out, residual_out, rstd, x, weight, bias,
                                     residual, options.M, options.N, options.H,
                                     threads_per_row, options.eps);
  } else if (vecs_per_thread <= 8) {
    rmsnorm_fwd_cached_kernel<X, RO, 8>
        <<<grid, block, 0, stream>>>(out, residual_out, rstd, x, weight, bias,
                                     residual, options.M, options.N, options.H,
                                     threads_per_row, options.eps);
  } else if (vecs_per_thread <= 16) {
    rmsnorm_fwd_cached_kernel<X, RO, 16>
        <<<grid, block, 0, stream>>>(out, residual_out, rstd, x, weight, bias,
                                     residual, options.M, options.N, options.H,
                                     threads_per_row, options.eps);
  } else if (vecs_per_thread <= 32) {
    rmsnorm_fwd_cached_kernel<X, RO, 32>
        <<<grid, block, 0, stream>>>(out, residual_out, rstd, x, weight, bias,
                                     residual, options.M, options.N, options.H,
                                     threads_per_row, options.eps);
  } else {
    rmsnorm_fwd_streaming_kernel<X, RO>
        <<<grid, block, 0, stream>>>(out, residual_out, rstd, x, weight, bias,
                                     residual, options.M, options.N, options.H,
                                     threads_per_row, options.eps);
  }
}

template <typename X>
void launch_bwd(Options const& options, X* dx, X* dresidual, float* dw,
                float* db, X const* x, float const* weight, X const* dout,
                X const* dresidual_out, float const* rstd,
                float* dw_partial, float* dw_partial_scratch,
                cudaStream_t stream) {
  if (dw != nullptr && db == nullptr && dresidual == nullptr &&
      dresidual_out == nullptr && weight != nullptr && options.H == 1 &&
      dw_partial != nullptr) {
    int partial_blocks =
        bwd_partial_blocks(options.N, options.M, options.partial_blocks);
    using XVec = Vec128<X>;
    int vecs = options.N / XVec::kLanes;
    int parts = bwd_cluster_parts<X>(options.N);
    int vecs_per_thread = (vecs + options.threads - 1) / options.threads;
    if constexpr (sizeof(X) >= 4) {
      if (options.N == 256) {
        // fp32 Vec128 covers four elements, so use 64 threads per row to
        // cover the full 256-wide row in the small-row persistent path.
        rmsnorm_bwd_weight_small_rows_kernel<X, 64, 2>
            <<<partial_blocks, 128, 0, stream>>>(dx, dw_partial, x, weight,
                                                 dout, rstd, options.M,
                                                 options.N);
        int reduce_chunks =
            bwd_reduce_chunks(options.N, partial_blocks, options.reduce_chunks);
        launch_reduce_dw(dw, dw_partial, dw_partial_scratch, partial_blocks,
                         options.N, reduce_chunks, stream);
        return;
      }
    }
    if constexpr (sizeof(X) < 4) {
      if (options.N == 256) {
        rmsnorm_bwd_weight_small_rows_kernel<X, 32, 4>
            <<<partial_blocks, 128, 0, stream>>>(dx, dw_partial, x, weight,
                                                 dout, rstd, options.M,
                                                 options.N);
        int reduce_chunks =
            bwd_reduce_chunks(options.N, partial_blocks, options.reduce_chunks);
        launch_reduce_dw(dw, dw_partial, dw_partial_scratch, partial_blocks,
                         options.N, reduce_chunks, stream);
        return;
      }
      if (options.N == 512) {
        rmsnorm_bwd_weight_small_rows_kernel<X, 64, 2>
            <<<partial_blocks, 128, 0, stream>>>(dx, dw_partial, x, weight,
                                                 dout, rstd, options.M,
                                                 options.N);
        int reduce_chunks =
            bwd_reduce_chunks(options.N, partial_blocks, options.reduce_chunks);
        launch_reduce_dw(dw, dw_partial, dw_partial_scratch, partial_blocks,
                         options.N, reduce_chunks, stream);
        return;
      }
    }
    dim3 block(options.threads);
    if (parts == 1) {
      int smem_bytes =
          4 * options.N * static_cast<int>(sizeof(X));
      using SmemKernel = void (*)(X*, float*, X const*, float const*, X const*,
                                  float const*, int, int);
      bool exact_tile = vecs == vecs_per_thread * options.threads;
      if (vecs_per_thread <= 1) {
        SmemKernel kernel =
            exact_tile ? rmsnorm_bwd_weight_smem_persistent_kernel<X, 1, true>
                       : rmsnorm_bwd_weight_smem_persistent_kernel<X, 1, false>;
        if (smem_bytes >= 48 * 1024) {
          CUDA_CHECK(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        }
        kernel<<<partial_blocks, block, smem_bytes, stream>>>(
            dx, dw_partial, x, weight, dout, rstd, options.M, options.N);
      } else if (vecs_per_thread <= 2) {
        SmemKernel kernel =
            exact_tile ? rmsnorm_bwd_weight_smem_persistent_kernel<X, 2, true>
                       : rmsnorm_bwd_weight_smem_persistent_kernel<X, 2, false>;
        if (smem_bytes >= 48 * 1024) {
          CUDA_CHECK(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        }
        kernel<<<partial_blocks, block, smem_bytes, stream>>>(
            dx, dw_partial, x, weight, dout, rstd, options.M, options.N);
      } else if (vecs_per_thread <= 4) {
        SmemKernel kernel =
            exact_tile ? rmsnorm_bwd_weight_smem_persistent_kernel<X, 4, true>
                       : rmsnorm_bwd_weight_smem_persistent_kernel<X, 4, false>;
        if (smem_bytes >= 48 * 1024) {
          CUDA_CHECK(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        }
        kernel<<<partial_blocks, block, smem_bytes, stream>>>(
            dx, dw_partial, x, weight, dout, rstd, options.M, options.N);
      } else if (vecs_per_thread <= 8) {
        SmemKernel kernel =
            exact_tile ? rmsnorm_bwd_weight_smem_persistent_kernel<X, 8, true>
                       : rmsnorm_bwd_weight_smem_persistent_kernel<X, 8, false>;
        if (smem_bytes >= 48 * 1024) {
          CUDA_CHECK(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        }
        kernel<<<partial_blocks, block, smem_bytes, stream>>>(
            dx, dw_partial, x, weight, dout, rstd, options.M, options.N);
      } else {
        rmsnorm_bwd_streaming_kernel<X><<<options.M, block, 0, stream>>>(
            dx, dresidual, dw, db, x, weight, dout, dresidual_out, rstd, options.M,
            options.N, options.H);
        return;
      }
      int reduce_chunks =
          bwd_reduce_chunks(options.N, partial_blocks, options.reduce_chunks);
      launch_reduce_dw(dw, dw_partial, dw_partial_scratch, partial_blocks,
                       options.N, reduce_chunks, stream);
      return;
    }
    if (parts > 1) {
      int vecs_per_part = (vecs + parts - 1) / parts;
      vecs_per_thread = (vecs_per_part + options.threads - 1) / options.threads;
      int elems_per_part = vecs_per_part * XVec::kLanes;
      bool use_smem_cluster =
          (sizeof(X) < 4 && (options.N <= 16384 || options.N >= 262144)) ||
          (sizeof(X) >= 4 && options.N >= 16384);
      int smem_bytes =
          use_smem_cluster
              ? 4 * elems_per_part * static_cast<int>(sizeof(X))
              : elems_per_part *
                    (2 * static_cast<int>(sizeof(X)) + static_cast<int>(sizeof(float)));
      using ClusterKernel = void (*)(X*, float*, X const*, float const*, X const*,
                                     float const*, int, int, int);
      cudaLaunchAttribute attrs[1]{};
      attrs[0].id = cudaLaunchAttributeClusterDimension;
      attrs[0].val.clusterDim.x = 1;
      attrs[0].val.clusterDim.y = parts;
      attrs[0].val.clusterDim.z = 1;
      cudaLaunchConfig_t config{};
      config.gridDim = dim3(partial_blocks, parts, 1);
      config.blockDim = block;
      config.dynamicSmemBytes = smem_bytes;
      config.stream = stream;
      config.attrs = attrs;
      config.numAttrs = 1;
      bool reload_dout = vecs_per_thread > 4;
      bool exact_tile = vecs_per_part == vecs_per_thread * options.threads;
      if (vecs_per_thread <= 1) {
        ClusterKernel kernel = nullptr;
        if (use_smem_cluster) {
          if (reload_dout) {
            kernel =
                exact_tile
                    ? rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 1,
                                                                        true, true>
                    : rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 1,
                                                                        true, false>;
          } else {
            kernel =
                exact_tile
                    ? rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 1,
                                                                        false, true>
                    : rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 1,
                                                                        false, false>;
          }
        } else {
          kernel = exact_tile
                       ? rmsnorm_bwd_weight_cluster_partial_kernel<X, 1, true>
                       : rmsnorm_bwd_weight_cluster_partial_kernel<X, 1, false>;
        }
        CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        if (smem_bytes >= 48 * 1024) {
          CUDA_CHECK(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        }
        CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, dx, dw_partial, x, weight,
                                      dout, rstd, options.M, options.N, parts));
      } else if (vecs_per_thread <= 2) {
        ClusterKernel kernel = nullptr;
        if (use_smem_cluster) {
          if (reload_dout) {
            kernel =
                exact_tile
                    ? rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 2,
                                                                        true, true>
                    : rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 2,
                                                                        true, false>;
          } else {
            kernel =
                exact_tile
                    ? rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 2,
                                                                        false, true>
                    : rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 2,
                                                                        false, false>;
          }
        } else {
          kernel = exact_tile
                       ? rmsnorm_bwd_weight_cluster_partial_kernel<X, 2, true>
                       : rmsnorm_bwd_weight_cluster_partial_kernel<X, 2, false>;
        }
        CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        if (smem_bytes >= 48 * 1024) {
          CUDA_CHECK(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        }
        CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, dx, dw_partial, x, weight,
                                      dout, rstd, options.M, options.N, parts));
      } else if (vecs_per_thread <= 4) {
        ClusterKernel kernel = nullptr;
        if (use_smem_cluster) {
          if (reload_dout) {
            kernel =
                exact_tile
                    ? rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 4,
                                                                        true, true>
                    : rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 4,
                                                                        true, false>;
          } else {
            kernel =
                exact_tile
                    ? rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 4,
                                                                        false, true>
                    : rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 4,
                                                                        false, false>;
          }
        } else {
          kernel = exact_tile
                       ? rmsnorm_bwd_weight_cluster_partial_kernel<X, 4, true>
                       : rmsnorm_bwd_weight_cluster_partial_kernel<X, 4, false>;
        }
        CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        if (smem_bytes >= 48 * 1024) {
          CUDA_CHECK(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        }
        CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, dx, dw_partial, x, weight,
                                      dout, rstd, options.M, options.N, parts));
      } else if (vecs_per_thread <= 8) {
        ClusterKernel kernel = nullptr;
        if (use_smem_cluster) {
          if (reload_dout) {
            kernel =
                exact_tile
                    ? rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 8,
                                                                        true, true>
                    : rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 8,
                                                                        true, false>;
          } else {
            kernel =
                exact_tile
                    ? rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 8,
                                                                        false, true>
                    : rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 8,
                                                                        false, false>;
          }
        } else {
          kernel = exact_tile
                       ? rmsnorm_bwd_weight_cluster_partial_kernel<X, 8, true>
                       : rmsnorm_bwd_weight_cluster_partial_kernel<X, 8, false>;
        }
        CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        if (smem_bytes >= 48 * 1024) {
          CUDA_CHECK(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        }
        CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, dx, dw_partial, x, weight,
                                      dout, rstd, options.M, options.N, parts));
      } else if (vecs_per_thread <= 16) {
        ClusterKernel kernel = nullptr;
        if (use_smem_cluster) {
          if (reload_dout) {
            kernel =
                exact_tile
                    ? rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 16,
                                                                        true, true>
                    : rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 16,
                                                                        true, false>;
          } else {
            kernel =
                exact_tile
                    ? rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 16,
                                                                        false, true>
                    : rmsnorm_bwd_weight_cluster_smem_persistent_kernel<X, 16,
                                                                        false, false>;
          }
        } else {
          kernel = exact_tile
                       ? rmsnorm_bwd_weight_cluster_partial_kernel<X, 16, true>
                       : rmsnorm_bwd_weight_cluster_partial_kernel<X, 16, false>;
        }
        CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        if (smem_bytes >= 48 * 1024) {
          CUDA_CHECK(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        }
        CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, dx, dw_partial, x, weight,
                                      dout, rstd, options.M, options.N, parts));
      } else {
        rmsnorm_bwd_streaming_kernel<X><<<options.M, block, 0, stream>>>(
            dx, dresidual, dw, db, x, weight, dout, dresidual_out, rstd, options.M,
            options.N, options.H);
        return;
      }
      int reduce_chunks =
          bwd_reduce_chunks(options.N, partial_blocks, options.reduce_chunks);
      launch_reduce_dw(dw, dw_partial, dw_partial_scratch, partial_blocks,
                       options.N, reduce_chunks, stream);
      return;
    }
    if (vecs_per_thread <= 1) {
      rmsnorm_bwd_weight_partial_kernel<X, 1>
          <<<partial_blocks, block, 0, stream>>>(dx, dw_partial, x, weight, dout,
                                                 rstd, options.M, options.N);
    } else if (vecs_per_thread <= 4) {
      rmsnorm_bwd_weight_partial_kernel<X, 4>
          <<<partial_blocks, block, 0, stream>>>(dx, dw_partial, x, weight, dout,
                                                 rstd, options.M, options.N);
    } else if (vecs_per_thread <= 8) {
      rmsnorm_bwd_weight_partial_kernel<X, 8>
          <<<partial_blocks, block, 0, stream>>>(dx, dw_partial, x, weight, dout,
                                                 rstd, options.M, options.N);
    } else {
      rmsnorm_bwd_streaming_kernel<X><<<options.M, block, 0, stream>>>(
          dx, dresidual, dw, db, x, weight, dout, dresidual_out, rstd, options.M,
          options.N, options.H);
      return;
    }
    int reduce_chunks =
        bwd_reduce_chunks(options.N, partial_blocks, options.reduce_chunks);
    launch_reduce_dw(dw, dw_partial, dw_partial_scratch, partial_blocks,
                     options.N, reduce_chunks, stream);
    return;
  }
  if (dw != nullptr) {
    CUDA_CHECK(cudaMemsetAsync(dw, 0, static_cast<size_t>(options.H) * options.N * sizeof(float),
                               stream));
  }
  if (db != nullptr) {
    CUDA_CHECK(cudaMemsetAsync(db, 0, static_cast<size_t>(options.H) * options.N * sizeof(float),
                               stream));
  }
  dim3 grid(options.M);
  dim3 block(options.threads);
  rmsnorm_bwd_streaming_kernel<X><<<grid, block, 0, stream>>>(
      dx, dresidual, dw, db, x, weight, dout, dresidual_out, rstd, options.M,
      options.N, options.H);
}

template <typename T>
void fill_random(std::vector<T>& values, std::mt19937& rng, float scale = 1.0f) {
  std::normal_distribution<float> normal(0.0f, scale);
  for (auto& v : values) {
    float x = std::max(-4.0f, std::min(4.0f, normal(rng)));
    v = scalar_from_float<T>(x);
  }
}

void fill_random_float(std::vector<float>& values, std::mt19937& rng,
                       float scale = 1.0f) {
  std::normal_distribution<float> normal(0.0f, scale);
  for (auto& v : values) {
    v = std::max(-4.0f, std::min(4.0f, normal(rng)));
  }
}

template <typename X, typename RO>
void reference_fwd(std::vector<X>& ref_out, std::vector<RO>& ref_residual_out,
                   std::vector<float>& ref_rstd, std::vector<X> const& x,
                   std::vector<float> const& weight,
                   std::vector<float> const& bias,
                   std::vector<X> const& residual, Options const& options) {
  ref_out.resize(static_cast<size_t>(options.M) * options.N);
  if (options.has_residual) {
    ref_residual_out.resize(static_cast<size_t>(options.M) * options.N);
  }
  if (options.store_rstd) {
    ref_rstd.resize(options.M);
  }
  for (int row = 0; row < options.M; ++row) {
    int head = options.H > 1 ? row % options.H : 0;
    int row_base = row * options.N;
    int affine_base = head * options.N;
    float sq = 0.0f;
    for (int col = 0; col < options.N; ++col) {
      float xv = scalar_to_float(x[row_base + col]);
      if (options.has_residual) {
        xv += scalar_to_float(residual[row_base + col]);
      }
      sq = std::fma(xv, xv, sq);
    }
    float rstd = 1.0f / std::sqrt(sq / static_cast<float>(options.N) + options.eps);
    if (options.store_rstd) {
      ref_rstd[row] = rstd;
    }
    for (int col = 0; col < options.N; ++col) {
      float xv = scalar_to_float(x[row_base + col]);
      if (options.has_residual) {
        xv += scalar_to_float(residual[row_base + col]);
        ref_residual_out[row_base + col] = scalar_from_float<RO>(xv);
      }
      float y = xv * rstd;
      if (options.has_weight) {
        y *= weight[affine_base + col];
      }
      if (options.has_bias) {
        y += bias[affine_base + col];
      }
      ref_out[row_base + col] = scalar_from_float<X>(y);
    }
  }
}

template <typename X>
void reference_bwd(std::vector<X>& ref_dx, std::vector<X>& ref_dresidual,
                   std::vector<float>& ref_dw, std::vector<float>& ref_db,
                   std::vector<X> const& x, std::vector<float> const& weight,
                   std::vector<X> const& dout,
                   std::vector<X> const& dresidual_out,
                   std::vector<float> const& rstd, Options const& options) {
  ref_dx.resize(static_cast<size_t>(options.M) * options.N);
  if (options.has_dresidual_out) {
    ref_dresidual.resize(static_cast<size_t>(options.M) * options.N);
  }
  if (options.has_weight) {
    ref_dw.assign(static_cast<size_t>(options.H) * options.N, 0.0f);
  }
  if (options.has_bias) {
    ref_db.assign(static_cast<size_t>(options.H) * options.N, 0.0f);
  }
  for (int row = 0; row < options.M; ++row) {
    int head = options.H > 1 ? row % options.H : 0;
    int row_base = row * options.N;
    int affine_base = head * options.N;
    float dot = 0.0f;
    for (int col = 0; col < options.N; ++col) {
      float x_hat = scalar_to_float(x[row_base + col]) * rstd[row];
      float wdy = scalar_to_float(dout[row_base + col]);
      if (options.has_weight) {
        wdy *= weight[affine_base + col];
      }
      dot = std::fma(x_hat, wdy, dot);
    }
    float mean = dot / static_cast<float>(options.N);
    for (int col = 0; col < options.N; ++col) {
      float xv = scalar_to_float(x[row_base + col]);
      float dy = scalar_to_float(dout[row_base + col]);
      float x_hat = xv * rstd[row];
      float wdy = dy;
      if (options.has_weight) {
        wdy *= weight[affine_base + col];
      }
      float grad = (wdy - x_hat * mean) * rstd[row];
      if (options.has_dresidual_out) {
        grad += scalar_to_float(dresidual_out[row_base + col]);
      }
      ref_dx[row_base + col] = scalar_from_float<X>(grad);
      if (options.has_dresidual_out) {
        ref_dresidual[row_base + col] = scalar_from_float<X>(grad);
      }
      if (options.has_weight) {
        ref_dw[affine_base + col] += dy * x_hat;
      }
      if (options.has_bias) {
        ref_db[affine_base + col] += dy;
      }
    }
  }
}

template <typename T>
bool compare_vector(std::vector<T> const& got, std::vector<T> const& ref,
                    char const* name, float atol, float rtol) {
  if (got.size() != ref.size()) {
    std::cerr << name << " size mismatch: got " << got.size() << " ref "
              << ref.size() << "\n";
    return false;
  }
  double sum_abs = 0.0;
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  size_t bad = 0;
  size_t bad_idx = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    float g = scalar_to_float(got[i]);
    float r = scalar_to_float(ref[i]);
    float abs_diff = std::abs(g - r);
    float rel_diff = abs_diff / std::max(std::abs(r), 1.0e-6f);
    sum_abs += abs_diff;
    if (abs_diff > max_abs) {
      max_abs = abs_diff;
      max_rel = rel_diff;
      bad_idx = i;
    }
    if (abs_diff > atol + rtol * std::abs(r)) {
      ++bad;
    }
  }
  std::cout << name << ": max_abs=" << max_abs << " max_rel=" << max_rel
            << " mean_abs=" << (sum_abs / std::max<size_t>(got.size(), 1))
            << " bad=" << bad << " at index " << bad_idx << "\n";
  return bad == 0;
}

bool compare_float_vector(std::vector<float> const& got,
                          std::vector<float> const& ref, char const* name,
                          float atol, float rtol) {
  if (got.size() != ref.size()) {
    std::cerr << name << " size mismatch: got " << got.size() << " ref "
              << ref.size() << "\n";
    return false;
  }
  double sum_abs = 0.0;
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  size_t bad = 0;
  size_t bad_idx = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    float abs_diff = std::abs(got[i] - ref[i]);
    float rel_diff = abs_diff / std::max(std::abs(ref[i]), 1.0e-6f);
    sum_abs += abs_diff;
    if (abs_diff > max_abs) {
      max_abs = abs_diff;
      max_rel = rel_diff;
      bad_idx = i;
    }
    if (abs_diff > atol + rtol * std::abs(ref[i])) {
      ++bad;
    }
  }
  std::cout << name << ": max_abs=" << max_abs << " max_rel=" << max_rel
            << " mean_abs=" << (sum_abs / std::max<size_t>(got.size(), 1))
            << " bad=" << bad << " at index " << bad_idx << "\n";
  return bad == 0;
}

template <typename X>
float dtype_atol() {
  if constexpr (std::is_same_v<X, float>) {
    return 2.0e-3f;
  } else if constexpr (std::is_same_v<X, __half>) {
    return 1.0e-2f;
  } else {
    return 1.0e-1f;
  }
}

template <typename X, typename RO>
struct DeviceBuffers {
  X* x = nullptr;
  X* out = nullptr;
  X* residual = nullptr;
  RO* residual_out = nullptr;
  float* rstd = nullptr;
  X* dout = nullptr;
  X* dx = nullptr;
  X* dresidual_out = nullptr;
  X* dresidual = nullptr;
  float* weight = nullptr;
  float* bias = nullptr;
  float* dw = nullptr;
  float* db = nullptr;
  float* partial_sums = nullptr;
  float* dw_partial = nullptr;
  float* dw_partial_scratch = nullptr;

  void allocate(Options const& options) {
    size_t elements = static_cast<size_t>(options.M) * options.N;
    size_t affine = static_cast<size_t>(options.H) * options.N;
    CUDA_CHECK(cudaMalloc(&x, elements * sizeof(X)));
    CUDA_CHECK(cudaMalloc(&out, elements * sizeof(X)));
    if (options.has_residual) {
      CUDA_CHECK(cudaMalloc(&residual, elements * sizeof(X)));
      CUDA_CHECK(cudaMalloc(&residual_out, elements * sizeof(RO)));
    }
    if (options.store_rstd || options.mode != Mode::kForward) {
      CUDA_CHECK(cudaMalloc(&rstd, options.M * sizeof(float)));
    }
    if (options.has_weight) {
      CUDA_CHECK(cudaMalloc(&weight, affine * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&dw, affine * sizeof(float)));
    }
    if (options.has_bias) {
      CUDA_CHECK(cudaMalloc(&bias, affine * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&db, affine * sizeof(float)));
    }
    if (options.mode != Mode::kForward && options.has_weight &&
        !options.has_bias && !options.has_dresidual_out && options.H == 1) {
      int partial_blocks =
          bwd_partial_blocks(options.N, options.M, options.partial_blocks);
      CUDA_CHECK(cudaMalloc(&dw_partial,
                            static_cast<size_t>(partial_blocks) * options.N *
                                sizeof(float)));
      int reduce_chunks =
          bwd_reduce_chunks(options.N, partial_blocks, options.reduce_chunks);
      if (reduce_chunks > 1) {
        CUDA_CHECK(cudaMalloc(&dw_partial_scratch,
                              static_cast<size_t>(reduce_chunks) * options.N *
                                  sizeof(float)));
      }
    }
    int parts = fwd_split_parts<X>(options.N);
    if (options.mode != Mode::kBackward && options.has_weight &&
        !options.has_bias && !options.has_residual && !options.store_rstd &&
        parts > 1) {
      CUDA_CHECK(cudaMalloc(&partial_sums,
                            static_cast<size_t>(options.M) * parts * sizeof(float)));
    }
    if (options.mode != Mode::kForward) {
      CUDA_CHECK(cudaMalloc(&dout, elements * sizeof(X)));
      CUDA_CHECK(cudaMalloc(&dx, elements * sizeof(X)));
      if (options.has_dresidual_out) {
        CUDA_CHECK(cudaMalloc(&dresidual_out, elements * sizeof(X)));
        CUDA_CHECK(cudaMalloc(&dresidual, elements * sizeof(X)));
      }
    }
  }

  void release() {
    cudaFree(x);
    cudaFree(out);
    cudaFree(residual);
    cudaFree(residual_out);
    cudaFree(rstd);
    cudaFree(dout);
    cudaFree(dx);
    cudaFree(dresidual_out);
    cudaFree(dresidual);
    cudaFree(weight);
    cudaFree(bias);
    cudaFree(dw);
    cudaFree(db);
    cudaFree(partial_sums);
    cudaFree(dw_partial);
    cudaFree(dw_partial_scratch);
  }
};

template <typename X, typename RO>
double benchmark(Options const& options, DeviceBuffers<X, RO>& d) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < options.iterations; ++i) {
    if (options.mode == Mode::kForward) {
      launch_fwd(options, d.out, d.residual_out, options.store_rstd ? d.rstd : nullptr,
                 d.x, options.has_weight ? d.weight : nullptr,
                 options.has_bias ? d.bias : nullptr,
                 options.has_residual ? d.residual : nullptr, d.partial_sums,
                 stream);
    } else {
      launch_bwd(options, d.dx, options.has_dresidual_out ? d.dresidual : nullptr,
                 options.has_weight ? d.dw : nullptr, options.has_bias ? d.db : nullptr,
                 d.x, options.has_weight ? d.weight : nullptr, d.dout,
                 options.has_dresidual_out ? d.dresidual_out : nullptr, d.rstd,
                 d.dw_partial, d.dw_partial_scratch, stream);
    }
  }
  cudaGraph_t graph;
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t graph_exec;
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  for (int i = 0; i < options.warmup_iterations; ++i) {
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return static_cast<double>(elapsed_ms) / options.iterations;
}

template <typename X, typename RO>
bool run_typed(Options const& options) {
  size_t elements = static_cast<size_t>(options.M) * options.N;
  size_t affine = static_cast<size_t>(options.H) * options.N;
  std::mt19937 rng(options.seed);

  bool need_fwd_ref = options.verify && options.mode != Mode::kBackward;
  bool need_bwd_ref = options.verify && options.mode != Mode::kForward;
  bool needs_bwd_data = options.mode != Mode::kForward;
  std::vector<X> h_x(elements);
  std::vector<X> h_out(need_fwd_ref ? elements : 0);
  std::vector<X> h_out_ref;
  std::vector<X> h_residual(options.has_residual ? elements : 0);
  std::vector<RO> h_residual_out(need_fwd_ref && options.has_residual ? elements : 0);
  std::vector<RO> h_residual_out_ref;
  std::vector<float> h_rstd(need_fwd_ref && options.store_rstd ? options.M : 0);
  std::vector<float> h_rstd_ref;
  std::vector<float> h_weight(affine);
  std::vector<float> h_bias(options.has_bias ? affine : 0);
  std::vector<X> h_dout(needs_bwd_data ? elements : 0);
  std::vector<X> h_dx(need_bwd_ref ? elements : 0);
  std::vector<X> h_dx_ref;
  std::vector<X> h_dresidual_out(needs_bwd_data && options.has_dresidual_out ? elements : 0);
  std::vector<X> h_dresidual(need_bwd_ref && options.has_dresidual_out ? elements : 0);
  std::vector<X> h_dresidual_ref;
  std::vector<float> h_dw(need_bwd_ref && options.has_weight ? affine : 0);
  std::vector<float> h_dw_ref;
  std::vector<float> h_db(need_bwd_ref && options.has_bias ? affine : 0);
  std::vector<float> h_db_ref;

  fill_random(h_x, rng);
  if (options.has_residual) {
    fill_random(h_residual, rng);
  }
  fill_random_float(h_weight, rng);
  if (options.has_bias) {
    fill_random_float(h_bias, rng);
  }
  if (needs_bwd_data) {
    fill_random(h_dout, rng);
    if (options.has_dresidual_out) {
      fill_random(h_dresidual_out, rng);
    }
  }

  if (need_fwd_ref) {
    reference_fwd(h_out_ref, h_residual_out_ref, h_rstd_ref, h_x, h_weight, h_bias,
                  h_residual, options);
  }

  if (options.mode != Mode::kForward) {
    if (h_rstd_ref.empty()) {
      if (options.verify) {
        std::vector<X> tmp_out;
        std::vector<RO> tmp_resout;
        Options rstd_options = options;
        rstd_options.store_rstd = true;
        reference_fwd(tmp_out, tmp_resout, h_rstd_ref, h_x, h_weight, h_bias,
                      h_residual, rstd_options);
      } else {
        h_rstd_ref.resize(options.M);
        std::uniform_real_distribution<float> dist(0.25f, 1.25f);
        for (float& v : h_rstd_ref) {
          v = dist(rng);
        }
      }
    }
    if (options.verify) {
      reference_bwd(h_dx_ref, h_dresidual_ref, h_dw_ref, h_db_ref, h_x, h_weight,
                    h_dout, h_dresidual_out, h_rstd_ref, options);
    }
  }

  DeviceBuffers<X, RO> d;
  d.allocate(options);
  CUDA_CHECK(cudaMemcpy(d.x, h_x.data(), elements * sizeof(X), cudaMemcpyHostToDevice));
  if (options.has_residual) {
    CUDA_CHECK(cudaMemcpy(d.residual, h_residual.data(), elements * sizeof(X),
                          cudaMemcpyHostToDevice));
  }
  if (options.has_weight) {
    CUDA_CHECK(cudaMemcpy(d.weight, h_weight.data(), affine * sizeof(float),
                          cudaMemcpyHostToDevice));
  }
  if (options.has_bias) {
    CUDA_CHECK(cudaMemcpy(d.bias, h_bias.data(), affine * sizeof(float),
                          cudaMemcpyHostToDevice));
  }
  if (options.mode != Mode::kForward) {
    CUDA_CHECK(cudaMemcpy(d.dout, h_dout.data(), elements * sizeof(X),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.rstd, h_rstd_ref.data(), options.M * sizeof(float),
                          cudaMemcpyHostToDevice));
    if (options.has_dresidual_out) {
      CUDA_CHECK(cudaMemcpy(d.dresidual_out, h_dresidual_out.data(),
                            elements * sizeof(X), cudaMemcpyHostToDevice));
    }
  }

  bool ok = true;
  if (options.verify && options.mode != Mode::kBackward) {
    launch_fwd(options, d.out, options.has_residual ? d.residual_out : nullptr,
               options.store_rstd ? d.rstd : nullptr, d.x,
               options.has_weight ? d.weight : nullptr,
               options.has_bias ? d.bias : nullptr,
               options.has_residual ? d.residual : nullptr, d.partial_sums,
               nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d.out, elements * sizeof(X),
                          cudaMemcpyDeviceToHost));
    ok &= compare_vector(h_out, h_out_ref, "fwd.out", dtype_atol<X>(), 1.0e-3f);
    if (options.has_residual) {
      CUDA_CHECK(cudaMemcpy(h_residual_out.data(), d.residual_out,
                            elements * sizeof(RO), cudaMemcpyDeviceToHost));
      ok &= compare_vector(h_residual_out, h_residual_out_ref, "fwd.residual_out",
                           dtype_atol<RO>(), 1.0e-3f);
    }
    if (options.store_rstd) {
      CUDA_CHECK(cudaMemcpy(h_rstd.data(), d.rstd, options.M * sizeof(float),
                            cudaMemcpyDeviceToHost));
      ok &= compare_float_vector(h_rstd, h_rstd_ref, "fwd.rstd", 2.0e-3f, 2.0e-3f);
    }
  }

  if (options.verify && options.mode != Mode::kForward) {
    launch_bwd(options, d.dx, options.has_dresidual_out ? d.dresidual : nullptr,
               options.has_weight ? d.dw : nullptr, options.has_bias ? d.db : nullptr,
               d.x, options.has_weight ? d.weight : nullptr, d.dout,
               options.has_dresidual_out ? d.dresidual_out : nullptr, d.rstd,
               d.dw_partial, d.dw_partial_scratch, nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_dx.data(), d.dx, elements * sizeof(X), cudaMemcpyDeviceToHost));
    ok &= compare_vector(h_dx, h_dx_ref, "bwd.dx", dtype_atol<X>(), 1.0e-3f);
    if (options.has_dresidual_out) {
      CUDA_CHECK(cudaMemcpy(h_dresidual.data(), d.dresidual, elements * sizeof(X),
                            cudaMemcpyDeviceToHost));
      ok &= compare_vector(h_dresidual, h_dresidual_ref, "bwd.dresidual",
                           dtype_atol<X>(), 1.0e-3f);
    }
    if (options.has_weight) {
      CUDA_CHECK(cudaMemcpy(h_dw.data(), d.dw, affine * sizeof(float),
                            cudaMemcpyDeviceToHost));
      ok &= compare_float_vector(h_dw, h_dw_ref, "bwd.dw", 2.0e-2f, 2.0e-3f);
    }
    if (options.has_bias) {
      CUDA_CHECK(cudaMemcpy(h_db.data(), d.db, affine * sizeof(float),
                            cudaMemcpyDeviceToHost));
      ok &= compare_float_vector(h_db, h_db_ref, "bwd.db", 2.0e-2f, 2.0e-3f);
    }
  }

  if (ok && options.benchmark && options.iterations > 0) {
    double avg_ms = benchmark(options, d);
    double bytes = 0.0;
    if (options.mode == Mode::kForward) {
      bytes = 2.0 * static_cast<double>(elements) * sizeof(X);
      if (options.has_weight) {
        bytes += static_cast<double>(affine) * sizeof(float);
      }
      if (options.has_bias) {
        bytes += static_cast<double>(affine) * sizeof(float);
      }
      if (options.has_residual) {
        bytes += static_cast<double>(elements) * (sizeof(X) + sizeof(RO));
      }
      if (options.store_rstd) {
        bytes += static_cast<double>(options.M) * sizeof(float);
      }
    } else {
      bytes = 3.0 * static_cast<double>(elements) * sizeof(X);
      if (options.has_weight) {
        bytes += 2.0 * static_cast<double>(affine) * sizeof(float);
      }
      if (options.has_bias) {
        bytes += 2.0 * static_cast<double>(affine) * sizeof(float);
      }
      if (options.has_dresidual_out) {
        bytes += 2.0 * static_cast<double>(elements) * sizeof(X);
      }
      bytes += static_cast<double>(options.M) * sizeof(float);
    }
    std::cout << std::fixed << std::setprecision(4)
              << "Kernel execution time: " << avg_ms << " ms\n"
              << "Source-style throughput: "
              << bytes / (avg_ms * 1.0e-3) / 1.0e9 << " GB/s\n";
  }

  d.release();
  return ok;
}

bool validate_options(Options& options) {
  if (options.M <= 0 || options.N <= 0 || options.H <= 0 ||
      options.iterations < 0 || options.warmup_iterations < 0) {
    std::cerr << "M, N, H, warmup_iterations, and iterations must be positive.\n";
    return false;
  }
  if (options.partial_blocks < 0) {
    std::cerr << "partial_blocks must be non-negative.\n";
    return false;
  }
  if (options.reduce_chunks < 0) {
    std::cerr << "reduce_chunks must be non-negative.\n";
    return false;
  }
  if (options.M % options.H != 0) {
    std::cerr << "M must be divisible by H for per-head weight indexing.\n";
    return false;
  }
  if (options.threads == 0) {
    options.threads =
        options.mode == Mode::kBackward ? bwd_threads(options.N)
                                        : heuristic_threads(options.N);
  }
  if (options.threads <= 0 || options.threads > 1024 ||
      (options.threads % 32) != 0) {
    std::cerr << "threads must be a positive multiple of 32 no larger than 1024.\n";
    return false;
  }
  int lanes = options.dtype == DType::kF32 ? 4 : 8;
  if (options.N % lanes != 0) {
    std::cerr << "N must be divisible by " << lanes
              << " for the current vectorized dtype path.\n";
    return false;
  }
  if (!options.has_residual &&
      options.residual_out_dtype == ResidualOutDType::kF32) {
    std::cerr << "residual_out_dtype=fp32 requires --residual=true.\n";
    return false;
  }
  if (options.mode == Mode::kBoth && options.benchmark) {
    std::cerr << "--mode=both is verification-only in this standalone harness; "
                 "use --benchmark=false or choose --mode=fwd/--mode=bwd.\n";
    return false;
  }
  return true;
}

template <typename X>
bool dispatch_residual_out(Options const& options) {
  if (options.residual_out_dtype == ResidualOutDType::kF32) {
    return run_typed<X, float>(options);
  }
  return run_typed<X, X>(options);
}

bool dispatch_dtype(Options const& options) {
  switch (options.dtype) {
    case DType::kF16:
      return dispatch_residual_out<__half>(options);
    case DType::kBF16:
      return dispatch_residual_out<__nv_bfloat16>(options);
    case DType::kF32:
      return dispatch_residual_out<float>(options);
  }
  return false;
}

void configure_options(Options& options, Mode mode, int dtype_code, int M, int N, int H,
                       int threads, float eps, bool has_weight, bool has_bias,
                       bool has_residual, bool store_rstd, bool has_dresidual_out,
                       bool residual_out_fp32) {
  options.mode = mode;
  options.M = M;
  options.N = N;
  options.H = H;
  options.threads = threads;
  options.eps = eps;
  options.dtype = static_cast<DType>(dtype_code);
  options.has_weight = has_weight;
  options.has_bias = has_bias;
  options.has_residual = has_residual;
  options.store_rstd = store_rstd;
  options.has_dresidual_out = has_dresidual_out;
  options.residual_out_dtype =
      residual_out_fp32 ? ResidualOutDType::kF32 : ResidualOutDType::kSame;
  options.verify = false;
  options.benchmark = false;
  if (!validate_options(options)) {
    throw std::invalid_argument("invalid RMSNorm launch options");
  }
}

template <typename X>
void launch_fwd_dtype(Options const& options, bool residual_out_fp32, void* out,
                      void* residual_out, float* rstd, void const* x,
                      float const* weight, float const* bias, void const* residual,
                      float* partial_sums, cudaStream_t stream) {
  if (residual_out_fp32) {
    launch_fwd<X, float>(
        options, static_cast<X*>(out), static_cast<float*>(residual_out), rstd,
        static_cast<X const*>(x), weight, bias, static_cast<X const*>(residual),
        partial_sums, stream);
  } else {
    launch_fwd<X, X>(
        options, static_cast<X*>(out), static_cast<X*>(residual_out), rstd,
        static_cast<X const*>(x), weight, bias, static_cast<X const*>(residual),
        partial_sums, stream);
  }
}

template <typename X>
void launch_bwd_dtype(Options const& options, void* dx, void* dresidual, float* dw,
                      float* db, void const* x, float const* weight, void const* dout,
                      void const* dresidual_out, float const* rstd, float* dw_partial,
                      float* dw_partial_scratch, cudaStream_t stream) {
  launch_bwd<X>(
      options, static_cast<X*>(dx), static_cast<X*>(dresidual), dw, db,
      static_cast<X const*>(x), weight, static_cast<X const*>(dout),
      static_cast<X const*>(dresidual_out), rstd, dw_partial, dw_partial_scratch,
      stream);
}

extern "C" int rmsnorm_cuda_heuristic_threads(int N) {
  return heuristic_threads(N);
}

extern "C" int rmsnorm_cuda_bwd_threads(int N) {
  return bwd_threads(N);
}

extern "C" int rmsnorm_cuda_bwd_partial_blocks(int N, int M, int requested) {
  return bwd_partial_blocks(N, M, requested);
}

extern "C" int rmsnorm_cuda_bwd_reduce_chunks(int N, int partial_blocks, int requested) {
  return bwd_reduce_chunks(N, partial_blocks, requested);
}

extern "C" int rmsnorm_cuda_fwd_split_parts(int dtype_code, int N) {
  DType dtype = static_cast<DType>(dtype_code);
  switch (dtype) {
    case DType::kF16:
      return fwd_split_parts<__half>(N);
    case DType::kBF16:
      return fwd_split_parts<__nv_bfloat16>(N);
    case DType::kF32:
      return fwd_split_parts<float>(N);
  }
  return 1;
}

extern "C" void rmsnorm_cuda_fwd(int dtype_code, bool residual_out_fp32, void* out,
                                 void* residual_out, float* rstd, void const* x,
                                 float const* weight, float const* bias,
                                 void const* residual, float* partial_sums, int M, int N,
                                 int H, int threads, float eps, cudaStream_t stream) {
  Options options;
  configure_options(options, Mode::kForward, dtype_code, M, N, H, threads, eps,
                    weight != nullptr, bias != nullptr, residual != nullptr,
                    rstd != nullptr, false, residual_out_fp32);
  switch (options.dtype) {
    case DType::kF16:
      launch_fwd_dtype<__half>(options, residual_out_fp32, out, residual_out, rstd, x,
                               weight, bias, residual, partial_sums, stream);
      return;
    case DType::kBF16:
      launch_fwd_dtype<__nv_bfloat16>(options, residual_out_fp32, out, residual_out,
                                      rstd, x, weight, bias, residual, partial_sums,
                                      stream);
      return;
    case DType::kF32:
      launch_fwd_dtype<float>(options, residual_out_fp32, out, residual_out, rstd, x,
                              weight, bias, residual, partial_sums, stream);
      return;
  }
  throw std::invalid_argument("unsupported RMSNorm dtype code");
}

extern "C" void rmsnorm_cuda_bwd(int dtype_code, void* dx, void* dresidual, float* dw,
                                 float* db, void const* x, float const* weight,
                                 void const* dout, void const* dresidual_out,
                                 float const* rstd, float* dw_partial,
                                 float* dw_partial_scratch, int M, int N, int H,
                                 int threads, float eps, cudaStream_t stream) {
  Options options;
  configure_options(options, Mode::kBackward, dtype_code, M, N, H, threads, eps,
                    weight != nullptr, db != nullptr, false, false,
                    dresidual_out != nullptr, false);
  switch (options.dtype) {
    case DType::kF16:
      launch_bwd_dtype<__half>(options, dx, dresidual, dw, db, x, weight, dout,
                               dresidual_out, rstd, dw_partial, dw_partial_scratch,
                               stream);
      return;
    case DType::kBF16:
      launch_bwd_dtype<__nv_bfloat16>(options, dx, dresidual, dw, db, x, weight, dout,
                                      dresidual_out, rstd, dw_partial,
                                      dw_partial_scratch, stream);
      return;
    case DType::kF32:
      launch_bwd_dtype<float>(options, dx, dresidual, dw, db, x, weight, dout,
                              dresidual_out, rstd, dw_partial, dw_partial_scratch,
                              stream);
      return;
  }
  throw std::invalid_argument("unsupported RMSNorm dtype code");
}

}  // namespace

#ifdef RMSNORM_STANDALONE
int main(int argc, char const** argv) {
  Options options;
  parse_options(argc, argv, options);
  if (!validate_options(options)) {
    return EXIT_FAILURE;
  }

  std::cout << "Running PTX RMSNorm "
            << (options.mode == Mode::kForward
                    ? "forward"
                    : (options.mode == Mode::kBackward ? "backward" : "both"))
            << " M=" << options.M << " N=" << options.N << " H=" << options.H
            << " threads=" << options.threads << " dtype="
            << dtype_name(options.dtype)
            << " weight=" << (options.has_weight ? "true" : "false")
            << " bias=" << (options.has_bias ? "true" : "false")
            << " residual=" << (options.has_residual ? "true" : "false")
            << "\n";

  if (options.verbose) {
    int lanes = options.dtype == DType::kF32 ? 4 : 8;
    int vecs = options.N / lanes;
    int vecs_per_thread = (vecs + options.threads - 1) / options.threads;
    std::cout << "Vector lanes=" << lanes
              << " vecs_per_thread=" << vecs_per_thread << "\n";
  }

  bool ok = dispatch_dtype(options);
  std::cout << (ok ? "PASS" : "FAIL") << "\n";
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
