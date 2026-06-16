// Copyright 2026 INT21 AI
// SPDX-License-Identifier: MIT

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <tuple>

extern "C" int rmsnorm_cuda_heuristic_threads(int N);
extern "C" int rmsnorm_cuda_bwd_threads(int N);
extern "C" int rmsnorm_cuda_bwd_partial_blocks(int N, int M, int requested);
extern "C" int rmsnorm_cuda_bwd_reduce_chunks(int N, int partial_blocks, int requested);
extern "C" int rmsnorm_cuda_fwd_split_parts(int dtype_code, int N);

extern "C" void rmsnorm_cuda_fwd(int dtype_code, bool residual_out_fp32, void* out,
                                 void* residual_out, float* rstd, void const* x,
                                 float const* weight, float const* bias,
                                 void const* residual, float* partial_sums, int M, int N,
                                 int H, int threads, float eps, cudaStream_t stream);

extern "C" void rmsnorm_cuda_bwd(int dtype_code, void* dx, void* dresidual, float* dw,
                                 float* db, void const* x, float const* weight,
                                 void const* dout, void const* dresidual_out,
                                 float const* rstd, float* dw_partial,
                                 float* dw_partial_scratch, int M, int N, int H,
                                 int threads, float eps, cudaStream_t stream);

namespace {

using OptionalTensor = c10::optional<at::Tensor>;

int dtype_code(at::ScalarType dtype) {
  switch (dtype) {
    case at::kHalf:
      return 0;
    case at::kBFloat16:
      return 1;
    case at::kFloat:
      return 2;
    default:
      TORCH_CHECK(false, "unsupported RMSNorm dtype: ", dtype);
  }
}

void check_cuda_contiguous(at::Tensor const& tensor, char const* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void* optional_data(OptionalTensor const& tensor) {
  return tensor.has_value() && tensor->defined() ? tensor->data_ptr() : nullptr;
}

float* optional_float_data(OptionalTensor const& tensor, char const* name) {
  if (!tensor.has_value() || !tensor->defined()) {
    return nullptr;
  }
  check_cuda_contiguous(*tensor, name);
  TORCH_CHECK(tensor->scalar_type() == at::kFloat, name, " must be float32");
  return tensor->data_ptr<float>();
}

void check_same_shape(at::Tensor const& tensor, at::Tensor const& ref, char const* name) {
  TORCH_CHECK(tensor.sizes() == ref.sizes(), name, " shape must match x");
}

int fwd_threads_for_dtype(int dtype_code, int64_t N) {
  if (dtype_code != 2 && N == 8192) {
    return 128;
  }
  if (dtype_code != 2 && N == 32768) {
    return 512;
  }
  return rmsnorm_cuda_heuristic_threads(static_cast<int>(N));
}

int bwd_threads_for_dtype(int dtype_code, int64_t N) {
  // fp32 N=32768 benefits from the 256-thread cluster-bwd shape; half/bf16
  // regress there, so keep the override out of the shared CUDA heuristic.
  if (dtype_code == 2 && N == 32768) {
    return 256;
  }
  return rmsnorm_cuda_bwd_threads(static_cast<int>(N));
}

}  // namespace

void fwd_(at::Tensor x, OptionalTensor weight, OptionalTensor bias, OptionalTensor residual,
          at::Tensor out, OptionalTensor residual_out, OptionalTensor rstd, int64_t H,
          double eps) {
  check_cuda_contiguous(x, "x");
  check_cuda_contiguous(out, "out");
  TORCH_CHECK(x.dim() == 2, "x must be flattened to 2D");
  TORCH_CHECK(out.scalar_type() == x.scalar_type(), "out dtype must match x dtype");
  check_same_shape(out, x, "out");
  TORCH_CHECK(H > 0, "H must be positive");
  TORCH_CHECK(x.size(0) % H == 0, "M must be divisible by H");
  dtype_code(x.scalar_type());

  if (x.numel() == 0) {
    return;
  }

  if (residual.has_value() && residual->defined()) {
    check_cuda_contiguous(*residual, "residual");
    TORCH_CHECK(residual->scalar_type() == x.scalar_type(), "residual dtype must match x dtype");
    check_same_shape(*residual, x, "residual");
  }
  if (residual_out.has_value() && residual_out->defined()) {
    check_cuda_contiguous(*residual_out, "residual_out");
    TORCH_CHECK(residual_out->scalar_type() == x.scalar_type() ||
                    residual_out->scalar_type() == at::kFloat,
                "residual_out must have x dtype or float32 dtype");
    check_same_shape(*residual_out, x, "residual_out");
  }
  if (rstd.has_value() && rstd->defined()) {
    check_cuda_contiguous(*rstd, "rstd");
    TORCH_CHECK(rstd->scalar_type() == at::kFloat, "rstd must be float32");
    TORCH_CHECK(rstd->numel() == x.size(0), "rstd must have one element per row");
  }

  float* weight_ptr = optional_float_data(weight, "weight");
  float* bias_ptr = optional_float_data(bias, "bias");
  if (weight.has_value() && weight->defined()) {
    TORCH_CHECK(weight->numel() == H * x.size(1), "weight shape must be (N,) or (H, N)");
  }
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->numel() == H * x.size(1), "bias shape must be (N,) or (H, N)");
  }

  float* cluster_path_flag = nullptr;
  int code = dtype_code(x.scalar_type());
  int split_parts = rmsnorm_cuda_fwd_split_parts(code, static_cast<int>(x.size(1)));
  // The CUDA launcher uses partial_sums as a non-null gate for clustered
  // large-row kernels; residual fwd has clustered paths for same-dtype and
  // fp32 residual_out.
  bool residual_cluster =
      optional_data(residual) != nullptr && residual_out.has_value() &&
      residual_out->defined() &&
      (residual_out->scalar_type() == x.scalar_type() ||
       residual_out->scalar_type() == at::kFloat);
  if (weight_ptr != nullptr && bias_ptr == nullptr && optional_data(rstd) == nullptr &&
      out.scalar_type() == x.scalar_type() && split_parts > 1 &&
      (optional_data(residual) == nullptr || residual_cluster)) {
    // The CUDA launcher uses this argument only as a non-null gate for the
    // clustered large-row path. Reuse an existing pointer to avoid a hot-path
    // allocator call on every benchmark iteration.
    cluster_path_flag = reinterpret_cast<float*>(out.data_ptr());
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  rmsnorm_cuda_fwd(code,
                   residual_out.has_value() && residual_out->defined() &&
                       residual_out->scalar_type() == at::kFloat,
                   out.data_ptr(), optional_data(residual_out),
                   rstd.has_value() && rstd->defined() ? rstd->data_ptr<float>() : nullptr,
                   x.data_ptr(), weight_ptr, bias_ptr, optional_data(residual),
                   cluster_path_flag,
                   static_cast<int>(x.size(0)), static_cast<int>(x.size(1)),
                   static_cast<int>(H), rmsnorm_cuda_heuristic_threads(x.size(1)),
                   static_cast<float>(eps), stream);
}

void bwd_(at::Tensor x, OptionalTensor weight, at::Tensor dout, at::Tensor rstd, at::Tensor dx,
          OptionalTensor dw, OptionalTensor db, OptionalTensor dresidual_out,
          OptionalTensor dresidual, int64_t H, double eps) {
  check_cuda_contiguous(x, "x");
  check_cuda_contiguous(dout, "dout");
  check_cuda_contiguous(rstd, "rstd");
  check_cuda_contiguous(dx, "dx");
  TORCH_CHECK(x.dim() == 2, "x must be flattened to 2D");
  TORCH_CHECK(dout.scalar_type() == x.scalar_type(), "dout dtype must match x dtype");
  TORCH_CHECK(dx.scalar_type() == x.scalar_type(), "dx dtype must match x dtype");
  check_same_shape(dout, x, "dout");
  check_same_shape(dx, x, "dx");
  TORCH_CHECK(rstd.scalar_type() == at::kFloat, "rstd must be float32");
  TORCH_CHECK(rstd.numel() == x.size(0), "rstd must have one element per row");
  TORCH_CHECK(H > 0, "H must be positive");
  TORCH_CHECK(x.size(0) % H == 0, "M must be divisible by H");

  if (x.numel() == 0) {
    return;
  }

  float* weight_ptr = optional_float_data(weight, "weight");
  float* dw_ptr = optional_float_data(dw, "dw");
  float* db_ptr = optional_float_data(db, "db");
  if (weight.has_value() && weight->defined()) {
    TORCH_CHECK(weight->numel() == H * x.size(1), "weight shape must be (N,) or (H, N)");
  }
  if (dw.has_value() && dw->defined()) {
    TORCH_CHECK(dw->numel() == H * x.size(1), "dw shape must match weight shape");
  }
  if (db.has_value() && db->defined()) {
    TORCH_CHECK(db->numel() == H * x.size(1), "db shape must match weight shape");
  }
  if (dresidual_out.has_value() && dresidual_out->defined()) {
    check_cuda_contiguous(*dresidual_out, "dresidual_out");
    TORCH_CHECK(dresidual_out->scalar_type() == x.scalar_type(),
                "dresidual_out dtype must match x dtype");
    check_same_shape(*dresidual_out, x, "dresidual_out");
  }
  if (dresidual.has_value() && dresidual->defined()) {
    check_cuda_contiguous(*dresidual, "dresidual");
    TORCH_CHECK(dresidual->scalar_type() == x.scalar_type(), "dresidual dtype must match x dtype");
    check_same_shape(*dresidual, x, "dresidual");
  }

  at::Tensor dw_partial;
  at::Tensor dw_partial_scratch;
  bool fast_weight_only = dw_ptr != nullptr && db_ptr == nullptr && optional_data(dresidual) == nullptr &&
                          optional_data(dresidual_out) == nullptr && weight_ptr != nullptr && H == 1;
  if (fast_weight_only) {
    int partial_blocks = rmsnorm_cuda_bwd_partial_blocks(
        static_cast<int>(x.size(1)), static_cast<int>(x.size(0)), 0);
    int reduce_chunks =
        rmsnorm_cuda_bwd_reduce_chunks(static_cast<int>(x.size(1)), partial_blocks, 0);
    dw_partial = at::empty({partial_blocks, x.size(1)}, x.options().dtype(at::kFloat));
    if (reduce_chunks > 1) {
      dw_partial_scratch = at::empty({reduce_chunks, x.size(1)}, x.options().dtype(at::kFloat));
    }
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int code = dtype_code(x.scalar_type());
  rmsnorm_cuda_bwd(code, dx.data_ptr(), optional_data(dresidual), dw_ptr,
                   db_ptr, x.data_ptr(), weight_ptr, dout.data_ptr(), optional_data(dresidual_out),
                   rstd.data_ptr<float>(),
                   dw_partial.defined() ? dw_partial.data_ptr<float>() : nullptr,
                   dw_partial_scratch.defined() ? dw_partial_scratch.data_ptr<float>() : nullptr,
                   static_cast<int>(x.size(0)), static_cast<int>(x.size(1)),
                   static_cast<int>(H), bwd_threads_for_dtype(code, x.size(1)),
                   static_cast<float>(eps), stream);
}

at::Tensor fwd_fast(at::Tensor x, at::Tensor weight, double eps) {
  int code = dtype_code(x.scalar_type());
  int64_t N = x.size(1);

  at::Tensor out = at::empty_strided(x.sizes(), x.strides(), x.options());
  float* cluster_path_flag = nullptr;
  if (N > 16384) {
    int split_parts = rmsnorm_cuda_fwd_split_parts(code, static_cast<int>(N));
    cluster_path_flag = split_parts > 1 ? reinterpret_cast<float*>(out.data_ptr()) : nullptr;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  rmsnorm_cuda_fwd(code, false, out.data_ptr(), nullptr, nullptr, x.data_ptr(),
                   weight.data_ptr<float>(), nullptr, nullptr, cluster_path_flag,
                   static_cast<int>(x.size(0)), static_cast<int>(x.size(1)), 1,
                   fwd_threads_for_dtype(code, N), static_cast<float>(eps), stream);
  return out;
}

std::tuple<at::Tensor, at::Tensor> fwd_residual_fast(at::Tensor x, at::Tensor weight,
                                                     at::Tensor residual,
                                                     bool residual_out_fp32, double eps) {
  int code = dtype_code(x.scalar_type());
  int64_t N = x.size(1);

  at::Tensor out = at::empty(x.sizes(), x.options());
  at::Tensor residual_out = at::empty(
      x.sizes(), residual_out_fp32 ? x.options().dtype(at::kFloat) : x.options());
  float* cluster_path_flag = nullptr;
  if (N > 16384) {
    int split_parts = rmsnorm_cuda_fwd_split_parts(code, static_cast<int>(N));
    cluster_path_flag =
        split_parts > 1 ? reinterpret_cast<float*>(out.data_ptr()) : nullptr;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  rmsnorm_cuda_fwd(code, residual_out_fp32, out.data_ptr(), residual_out.data_ptr(), nullptr,
                   x.data_ptr(), weight.data_ptr<float>(), nullptr, residual.data_ptr(),
                   cluster_path_flag, static_cast<int>(x.size(0)), static_cast<int>(x.size(1)),
                   1, rmsnorm_cuda_heuristic_threads(static_cast<int>(N)),
                   static_cast<float>(eps), stream);
  return {out, residual_out};
}

std::tuple<at::Tensor, at::Tensor> bwd_fast(at::Tensor x, at::Tensor weight, at::Tensor dout,
                                            at::Tensor rstd) {
  at::Tensor dx = at::empty_strided(x.sizes(), x.strides(), x.options());
  at::Tensor dw = at::empty_strided(weight.sizes(), weight.strides(), weight.options());
  int code = dtype_code(x.scalar_type());
  int partial_blocks = rmsnorm_cuda_bwd_partial_blocks(
      static_cast<int>(x.size(1)), static_cast<int>(x.size(0)), 0);
  int reduce_chunks =
      rmsnorm_cuda_bwd_reduce_chunks(static_cast<int>(x.size(1)), partial_blocks, 0);
  thread_local at::Tensor dw_partial_cache;
  thread_local at::Tensor dw_partial_scratch_cache;
  std::array<int64_t, 2> partial_shape_arr{partial_blocks, x.size(1)};
  at::IntArrayRef partial_shape(partial_shape_arr);
  if (!dw_partial_cache.defined() || dw_partial_cache.sizes() != partial_shape ||
      dw_partial_cache.device() != x.device()) {
    dw_partial_cache = at::empty(partial_shape, x.options().dtype(at::kFloat));
  }
  if (reduce_chunks > 1) {
    std::array<int64_t, 2> scratch_shape_arr{reduce_chunks, x.size(1)};
    at::IntArrayRef scratch_shape(scratch_shape_arr);
    if (!dw_partial_scratch_cache.defined() ||
        dw_partial_scratch_cache.sizes() != scratch_shape ||
        dw_partial_scratch_cache.device() != x.device()) {
      dw_partial_scratch_cache =
          at::empty(scratch_shape, x.options().dtype(at::kFloat));
    }
  } else {
    dw_partial_scratch_cache = at::Tensor();
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  rmsnorm_cuda_bwd(code, dx.data_ptr(), nullptr, dw.data_ptr<float>(),
                   nullptr, x.data_ptr(), weight.data_ptr<float>(), dout.data_ptr(), nullptr,
                   rstd.data_ptr<float>(), dw_partial_cache.data_ptr<float>(),
                   dw_partial_scratch_cache.defined()
                       ? dw_partial_scratch_cache.data_ptr<float>()
                       : nullptr,
                   static_cast<int>(x.size(0)), static_cast<int>(x.size(1)), 1,
                   bwd_threads_for_dtype(code, x.size(1)), static_cast<float>(1e-6), stream);
  return {dx, dw};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd_", &fwd_, "RMSNorm forward CUDA/PTX launcher");
  m.def("bwd_", &bwd_, "RMSNorm backward CUDA/PTX launcher");
  m.def("fwd_fast", &fwd_fast, "RMSNorm common forward fast path");
  m.def("fwd_residual_fast", &fwd_residual_fast, "RMSNorm common residual forward fast path");
  m.def("bwd_fast", &bwd_fast, "RMSNorm common backward fast path");
}
