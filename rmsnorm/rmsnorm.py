# Copyright 2026 INT21 AI
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from ._extension import get_extension


_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_FWD_FAST = None
_FWD_RESIDUAL_FAST = None
_BWD_FAST = None
_SUPPORTED_DTYPE_NAMES = "float32, float16 or bfloat16"


def _fwd_fast_op():
    global _FWD_FAST
    if _FWD_FAST is None:
        _FWD_FAST = get_extension().fwd_fast
    return _FWD_FAST


def _fwd_residual_fast_op():
    global _FWD_RESIDUAL_FAST
    if _FWD_RESIDUAL_FAST is None:
        _FWD_RESIDUAL_FAST = get_extension().fwd_residual_fast
    return _FWD_RESIDUAL_FAST


def _bwd_fast_op():
    global _BWD_FAST
    if _BWD_FAST is None:
        _BWD_FAST = get_extension().bwd_fast
    return _BWD_FAST


def _ensure_contiguous(t: Tensor) -> Tensor:
    return t if t.stride(-1) == 1 and t.is_contiguous() else t.contiguous()


def _vectorized_n(dtype: torch.dtype, n: int) -> bool:
    lanes = 4 if dtype == torch.float32 else 8
    return n % lanes == 0


def _check_x_shape(x_shape: torch.Size) -> None:
    if len(x_shape) == 0:
        raise ValueError("RMSNorm input must have at least one dimension")
    if x_shape[-1] == 0:
        raise ValueError("RMSNorm hidden dimension must be nonzero")


def _affine_head_count(name: str, tensor: Optional[Tensor], x_shape: torch.Size) -> int:
    if tensor is None:
        return 1
    if tensor.dim() == 1:
        if tensor.shape[0] != x_shape[-1]:
            raise ValueError(
                f"{name} shape must be (N,) or (H, N) with N={x_shape[-1]}, "
                f"got {tuple(tensor.shape)}"
            )
        return 1
    if tensor.dim() == 2:
        if len(x_shape) < 2:
            raise ValueError(f"per-head {name} requires x to have at least two dimensions")
        if tensor.shape[1] != x_shape[-1]:
            raise ValueError(
                f"{name} shape must be (N,) or (H, N) with N={x_shape[-1]}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.shape[0] != x_shape[-2]:
            raise ValueError(
                f"per-head {name} expected x.shape[-2] == {tensor.shape[0]}, got {x_shape[-2]}"
            )
        return int(tensor.shape[0])
    raise ValueError(f"{name} shape must be (N,) or (H, N), got {tuple(tensor.shape)}")


def _check_same_device(name: str, tensor: Optional[Tensor], device: torch.device) -> None:
    if tensor is not None and tensor.device != device:
        raise ValueError(f"{name} must be on the same device as x")


def _check_tensor_dtype(name: str, tensor: Optional[Tensor]) -> None:
    if tensor is not None and tensor.dtype not in _SUPPORTED_DTYPES:
        raise AssertionError(f"{name} must be {_SUPPORTED_DTYPE_NAMES}")


def _check_optional_dtype(name: str, dtype: Optional[torch.dtype]) -> None:
    if dtype is not None and dtype not in _SUPPORTED_DTYPES:
        raise AssertionError(f"{name} must be {_SUPPORTED_DTYPE_NAMES}")


def _validate_fwd_dtypes(
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    residual: Optional[Tensor],
    out_dtype: torch.dtype,
    residual_dtype: Optional[torch.dtype],
) -> None:
    if x.dtype not in _SUPPORTED_DTYPES:
        raise AssertionError("Unsupported dtype")
    _check_tensor_dtype("Weight", weight)
    _check_tensor_dtype("Bias", bias)
    _check_tensor_dtype("Residual", residual)
    _check_optional_dtype("out_dtype", out_dtype)
    _check_optional_dtype("residual_dtype", residual_dtype)


def _head_count(weight: Optional[Tensor], bias: Optional[Tensor], x_shape: torch.Size) -> int:
    _check_x_shape(x_shape)
    weight_h = _affine_head_count("weight", weight, x_shape)
    bias_h = _affine_head_count("bias", bias, x_shape)
    if weight_h > 1 and bias_h > 1 and weight_h != bias_h:
        raise ValueError(f"weight and bias head counts must match, got {weight_h} and {bias_h}")
    return max(weight_h, bias_h)


def _validate_fwd_shapes(
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    residual: Optional[Tensor],
) -> int:
    _check_same_device("weight", weight, x.device)
    _check_same_device("bias", bias, x.device)
    _check_same_device("residual", residual, x.device)
    h = _head_count(weight, bias, x.shape)
    if residual is not None and residual.shape != x.shape:
        raise ValueError(
            f"residual shape must match x shape {tuple(x.shape)}, got {tuple(residual.shape)}"
        )
    return h


def _extension_supports_affine_shapes(
    weight: Optional[Tensor], bias: Optional[Tensor], h: int
) -> bool:
    if h <= 1:
        return True
    return not (
        (weight is not None and weight.dim() == 1)
        or (bias is not None and bias.dim() == 1)
    )


def _validate_bwd_shapes(
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    dout: Tensor,
    rstd: Tensor,
    dresidual_out: Optional[Tensor],
) -> int:
    _check_same_device("weight", weight, x.device)
    _check_same_device("bias", bias, x.device)
    _check_same_device("dout", dout, x.device)
    _check_same_device("rstd", rstd, x.device)
    _check_same_device("dresidual_out", dresidual_out, x.device)
    h = _head_count(weight, bias, x.shape)
    if dout.shape != x.shape:
        raise ValueError(f"dout shape must match x shape {tuple(x.shape)}, got {tuple(dout.shape)}")
    if rstd.dtype != torch.float32:
        raise ValueError("rstd must be float32")
    rows = x.numel() // x.shape[-1]
    if rstd.numel() != rows:
        raise ValueError(f"rstd must have one element per row, got {rstd.numel()} for {rows} rows")
    if dresidual_out is not None and dresidual_out.shape != x.shape:
        raise ValueError(
            f"dresidual_out shape must match x shape {tuple(x.shape)}, "
            f"got {tuple(dresidual_out.shape)}"
        )
    return h


def _flatten_rows(x: Tensor, h: int) -> Tensor:
    if h == 1:
        return x.reshape(-1, x.shape[-1])
    if x.shape[-2] != h:
        raise ValueError(f"per-head RMSNorm expected x.shape[-2] == {h}, got {x.shape[-2]}")
    return x.reshape(-1, x.shape[-1])


def _affine_f32(t: Optional[Tensor]) -> Optional[Tensor]:
    if t is None:
        return None
    return t if t.dtype == torch.float32 and t.is_contiguous() else t.float().contiguous()


def _sum_to_affine_shape(values: Tensor, affine_shape: torch.Size) -> Tensor:
    reduce_ndim = values.dim() - len(affine_shape)
    if reduce_ndim < 0:
        raise ValueError(
            f"affine shape {tuple(affine_shape)} cannot be larger than input shape {tuple(values.shape)}"
        )
    if reduce_ndim == 0:
        return values
    return values.sum(dim=tuple(range(reduce_ndim)))


def _can_use_extension(
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    residual: Optional[Tensor],
    out_dtype: Optional[torch.dtype],
    residual_dtype: Optional[torch.dtype],
    h: Optional[int] = None,
) -> bool:
    h = _validate_fwd_shapes(x, weight, bias, residual) if h is None else h
    if not x.is_cuda or x.dtype not in _SUPPORTED_DTYPES or not _vectorized_n(x.dtype, x.shape[-1]):
        return False
    if out_dtype is not None and out_dtype != x.dtype:
        return False
    if residual is not None and residual.dtype != x.dtype:
        return False
    if residual is None and residual_dtype not in (None, x.dtype):
        return False
    if residual is not None and residual_dtype not in (None, residual.dtype, torch.float32):
        return False
    return _extension_supports_affine_shapes(weight, bias, h)


def rmsnorm_ref(x: Tensor, w: Optional[Tensor] = None, bias=None, residual=None, eps=1e-6):
    x_f32 = x.float()
    if residual is not None:
        x_f32 = x_f32 + residual.float()
    rstd = torch.rsqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + eps)
    out = x_f32 * rstd
    if w is not None:
        out = out * w.float()
    if bias is not None:
        out = out + bias.float()
    out = out.to(x.dtype)
    if residual is None:
        return out
    return out, x_f32.to(residual.dtype)


def _fallback_fwd(
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    residual: Optional[Tensor],
    out_dtype: torch.dtype,
    residual_dtype: Optional[torch.dtype],
    eps: float,
    store_rstd: bool,
):
    x_f32 = x.float()
    if residual is not None:
        x_f32 = x_f32 + residual.float()
    rstd_full = torch.rsqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + eps)
    y = x_f32 * rstd_full
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()
    out = y.to(out_dtype)
    rstd = rstd_full.squeeze(-1).contiguous() if store_rstd else None
    if residual is None:
        residual_out = x if residual_dtype in (None, x.dtype) else x.to(residual_dtype)
    else:
        residual_out = x_f32.to(residual_dtype or residual.dtype)
    return out, residual_out, rstd


def rmsnorm_fwd(
    x: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    store_rstd: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x_dim = x.dim()
    n = x.shape[-1] if x_dim > 0 else 0
    x_dtype = x.dtype
    x_device = x.device

    if (
        residual is not None
        and bias is None
        and not store_rstd
        and out_dtype == x_dtype
        and residual_dtype in (None, x_dtype, torch.float32)
        and x.is_cuda
        and x_dim == 2
        and n != 0
        and x_dtype in _SUPPORTED_DTYPES
        and x.is_contiguous()
        and _vectorized_n(x_dtype, n)
        and weight is not None
        and weight.device == x_device
        and weight.dtype == torch.float32
        and weight.dim() == 1
        and weight.is_contiguous()
        and weight.numel() == n
        and residual.device == x_device
        and residual.dtype == x_dtype
        and residual.shape == x.shape
        and residual.is_contiguous()
    ):
        op = (
            _FWD_RESIDUAL_FAST
            if _FWD_RESIDUAL_FAST is not None
            else _fwd_residual_fast_op()
        )
        out, residual_out = op(
            x, weight, residual, residual_dtype == torch.float32, float(eps)
        )
        return out, residual_out, None

    common_fast = (
        x.is_cuda
        and x_dim == 2
        and n != 0
        and x_dtype in _SUPPORTED_DTYPES
        and x.is_contiguous()
        and _vectorized_n(x_dtype, n)
        and out_dtype == x_dtype
        and weight is not None
        and weight.device == x_device
        and weight.dtype == torch.float32
        and weight.dim() == 1
        and weight.is_contiguous()
        and weight.numel() == n
        and bias is None
        and not store_rstd
    )

    if (
        common_fast
        and residual_dtype in (None, x_dtype)
        and residual is None
    ):
        op = _FWD_FAST if _FWD_FAST is not None else _fwd_fast_op()
        out = op(x, weight, float(eps))
        return out, x, None

    _validate_fwd_dtypes(x, weight, bias, residual, out_dtype, residual_dtype)
    h = _validate_fwd_shapes(x, weight, bias, residual)

    if not _can_use_extension(x, weight, bias, residual, out_dtype, residual_dtype, h=h):
        return _fallback_fwd(x, weight, bias, residual, out_dtype, residual_dtype, eps, store_rstd)

    x_contig = _ensure_contiguous(x)
    residual_contig = _ensure_contiguous(residual) if residual is not None else None
    x_flat = _flatten_rows(x_contig, h)
    residual_flat = _flatten_rows(residual_contig, h) if residual_contig is not None else None

    out = torch.empty_like(x_contig, dtype=out_dtype)
    out_flat = _flatten_rows(out, h)
    rstd = torch.empty(x_flat.shape[0], device=x.device, dtype=torch.float32) if store_rstd else None

    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        ro_dtype = residual_dtype if residual_dtype is not None else x.dtype
        residual_out = torch.empty_like(x_contig, dtype=ro_dtype)
        residual_out_flat = _flatten_rows(residual_out, h)
    else:
        residual_out = x_contig
        residual_out_flat = None

    ext = get_extension()
    ext.fwd_(
        x_flat,
        _affine_f32(weight),
        _affine_f32(bias),
        residual_flat,
        out_flat,
        residual_out_flat,
        rstd,
        h,
        float(eps),
    )
    return out.reshape_as(x_contig), residual_out.reshape_as(x_contig), (
        rstd.reshape(x_contig.shape[:-1]) if rstd is not None else None
    )


def rmsnorm_bwd_ref(x, w, dout, rstd, eps=1e-6):
    x_f32 = x.float()
    x_hat = x_f32 * rstd.unsqueeze(-1)
    wdy = dout.float() * w.float() if w is not None else dout.float()
    c1 = (x_hat * wdy).mean(dim=-1, keepdim=True)
    dx = (wdy - x_hat * c1) * rstd.unsqueeze(-1)
    if w is None:
        return dx.to(x.dtype), None
    dw = _sum_to_affine_shape(dout.float() * x_hat, w.shape)
    return dx.to(x.dtype), dw.to(w.dtype)


def rmsnorm_bwd(
    x: Tensor,
    weight: Optional[Tensor],
    dout: Tensor,
    rstd: Tensor,
    dresidual_out: Optional[Tensor] = None,
    has_bias: bool = False,
    has_residual: bool = False,
    bias: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    if x.dtype not in _SUPPORTED_DTYPES:
        raise AssertionError("Unsupported dtype")
    _check_tensor_dtype("Weight", weight)
    _check_tensor_dtype("Bias", bias)
    _check_tensor_dtype("dout", dout)
    _check_tensor_dtype("dresidual_out", dresidual_out)
    h = _validate_bwd_shapes(x, weight, bias, dout, rstd, dresidual_out)

    if (
        x.is_cuda
        and x.dim() == 2
        and x.is_contiguous()
        and dout.is_contiguous()
        and rstd.is_contiguous()
        and dout.shape == x.shape
        and rstd.dtype == torch.float32
        and rstd.numel() == x.shape[0]
        and x.dtype == dout.dtype
        and _vectorized_n(x.dtype, x.shape[-1])
        and weight is not None
        and weight.dtype == torch.float32
        and weight.dim() == 1
        and weight.is_contiguous()
        and weight.numel() == x.shape[-1]
        and dresidual_out is None
        and not has_bias
        and not has_residual
    ):
        dx, dw = _bwd_fast_op()(x, weight, dout, rstd)
        return dx, dw, None, None

    fallback = (
        not x.is_cuda
        or x.dtype != dout.dtype
        or not _vectorized_n(x.dtype, x.shape[-1])
        or (dresidual_out is not None and dresidual_out.dtype != x.dtype)
        or not _extension_supports_affine_shapes(weight, bias, h)
    )
    if fallback:
        dx, dw = rmsnorm_bwd_ref(x, weight, dout, rstd)
        if dresidual_out is not None:
            dx = (dx.float() + dresidual_out.float()).to(x.dtype)
        if has_bias:
            db_dtype = (
                bias.dtype if bias is not None else (weight.dtype if weight is not None else x.dtype)
            )
            if bias is not None:
                db_shape = bias.shape
            elif weight is not None:
                db_shape = weight.shape
            else:
                db_shape = torch.Size((x.shape[-1],))
            db = _sum_to_affine_shape(dout.float(), db_shape).to(db_dtype)
        else:
            db = None
        dresidual = dx if has_residual else None
        return dx, dw, db, dresidual

    x_contig = _ensure_contiguous(x)
    dout_contig = _ensure_contiguous(dout)
    dresidual_out_contig = _ensure_contiguous(dresidual_out) if dresidual_out is not None else None
    x_flat = _flatten_rows(x_contig, h)
    dout_flat = _flatten_rows(dout_contig, h)
    dresidual_out_flat = (
        _flatten_rows(dresidual_out_contig, h) if dresidual_out_contig is not None else None
    )
    rstd_flat = rstd.contiguous().reshape(-1)

    dx = torch.empty_like(x_contig)
    dx_flat = _flatten_rows(dx, h)
    weight_f32 = _affine_f32(weight)
    bias_f32 = _affine_f32(bias)
    dw_f32 = (
        torch.empty(weight_f32.shape, device=x.device, dtype=torch.float32)
        if weight_f32 is not None
        else None
    )
    if has_bias:
        if bias_f32 is not None:
            db_shape = bias_f32.shape
        elif weight_f32 is not None:
            db_shape = weight_f32.shape
        else:
            db_shape = (h, x.shape[-1]) if h > 1 else (x.shape[-1],)
        db_f32 = torch.empty(db_shape, device=x.device, dtype=torch.float32)
    else:
        db_f32 = None
    dresidual = torch.empty_like(x_contig) if dresidual_out_flat is not None else None
    dresidual_flat = _flatten_rows(dresidual, h) if dresidual is not None else None

    ext = get_extension()
    ext.bwd_(
        x_flat,
        weight_f32,
        dout_flat,
        rstd_flat,
        dx_flat,
        dw_f32,
        db_f32,
        dresidual_out_flat,
        dresidual_flat,
        h,
        float(1e-6),
    )

    dw = dw_f32.reshape_as(weight).to(weight.dtype) if weight is not None else None
    if db_f32 is None:
        db = None
    elif bias is not None:
        db = db_f32.reshape_as(bias).to(bias.dtype)
    elif weight is not None:
        db = db_f32.reshape_as(weight).to(weight.dtype)
    else:
        db = db_f32.reshape((h, x.shape[-1]) if h > 1 else (x.shape[-1],)).to(x.dtype)
    if has_residual and dresidual is None:
        dresidual = dx
    return dx.reshape_as(x_contig), dw, db, dresidual.reshape_as(x_contig) if dresidual is not None else None


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        residual=None,
        out_dtype=None,
        residual_dtype=None,
        eps=1e-6,
        prenorm=False,
    ):
        need_grad = any(ctx.needs_input_grad[:4])
        out, residual_out, rstd = rmsnorm_fwd(
            x,
            weight,
            bias=bias,
            residual=residual,
            out_dtype=out_dtype,
            residual_dtype=residual_dtype,
            eps=eps,
            store_rstd=need_grad,
        )
        ctx.x_dtype = x.dtype
        ctx.residual_dtype = residual.dtype if residual is not None else None
        ctx.save_for_backward(x if residual is None else residual_out, weight, bias, rstd)
        ctx.has_bias = bias is not None
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        if residual is not None and prenorm:
            return out, residual_out
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        x, weight, bias, rstd = ctx.saved_tensors
        dresidual_out = args[0] if ctx.prenorm and ctx.has_residual else None
        if dout is None:
            dx = dresidual_out.to(ctx.x_dtype) if dresidual_out is not None else None
            dresidual = (
                dresidual_out.to(ctx.residual_dtype)
                if dresidual_out is not None and ctx.residual_dtype is not None
                else None
            )
            return dx, None, None, dresidual, *([None] * 4)
        dx, dw, db, dresidual = rmsnorm_bwd(
            x,
            weight,
            dout.contiguous(),
            rstd,
            dresidual_out.contiguous() if dresidual_out is not None else None,
            ctx.has_bias,
            has_residual=ctx.has_residual,
            bias=bias,
        )
        if dx is not None and dx.dtype != ctx.x_dtype:
            dx = dx.to(ctx.x_dtype)
        if (
            dresidual is not None
            and ctx.residual_dtype is not None
            and dresidual.dtype != ctx.residual_dtype
        ):
            dresidual = dresidual.to(ctx.residual_dtype)
        return dx, dw, db, dresidual, *([None] * 4)


def rmsnorm(
    x: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    prenorm: bool = False,
) -> Tensor:
    out_dtype = out_dtype or x.dtype
    _validate_fwd_dtypes(x, weight, bias, residual, out_dtype, residual_dtype)
    h = _validate_fwd_shapes(x, weight, bias, residual)
    if torch.compiler.is_compiling() or not _can_use_extension(
        x, weight, bias, residual, out_dtype, residual_dtype, h=h
    ):
        out, residual_out, _ = _fallback_fwd(
            x, weight, bias, residual, out_dtype, residual_dtype, eps, False
        )
        if residual is None or not prenorm:
            return out
        return out, residual_out

    x_shape = x.shape
    x_contig = _ensure_contiguous(x)
    residual_contig = _ensure_contiguous(residual) if residual is not None else None
    flatten_for_fast_path = h == 1
    x_arg = _flatten_rows(x_contig, h) if flatten_for_fast_path else x_contig
    residual_arg = (
        _flatten_rows(residual_contig, h)
        if flatten_for_fast_path and residual_contig is not None
        else residual_contig
    )
    result = RMSNormFunction.apply(
        x_arg, weight, bias, residual_arg, out_dtype, residual_dtype, eps, prenorm
    )
    if not flatten_for_fast_path:
        return result
    if isinstance(result, tuple):
        return tuple(t.reshape(x_shape) for t in result)
    return result.reshape(x_shape)


class QuackRMSNorm(torch.nn.RMSNorm):
    def __init__(
        self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True, device=None, dtype=None
    ):
        super().__init__(dim, eps, elementwise_affine, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return rmsnorm(x, self.weight, eps=self.eps)


def layernorm_ref(x: Tensor, w: Tensor, eps: float = 1e-6) -> Tensor:
    return torch.nn.functional.layer_norm(x.float(), w.shape, w.float(), None, eps).to(x.dtype)


def layernorm_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    eps: float = 1e-6,
    return_rstd: bool = False,
    return_mean: bool = False,
):
    out = torch.nn.functional.layer_norm(
        x.float(), weight.shape, weight.float(), bias.float() if bias is not None else None, eps
    ).to(x.dtype)
    x_f32 = x.float()
    mean_for_var = x_f32.mean(dim=-1) if return_rstd or return_mean else None
    var = ((x_f32 - mean_for_var.unsqueeze(-1)) ** 2).mean(dim=-1) if return_rstd else None
    rstd = torch.rsqrt(var + eps) if return_rstd else None
    if return_rstd and return_mean:
        return out, rstd, mean_for_var
    if return_rstd:
        return out, rstd
    if return_mean:
        return out, mean_for_var
    return out


def layernorm_rstd_ref(x: Tensor, eps: float = 1e-6):
    mean = x.float().mean(dim=-1, keepdim=True)
    var = ((x.float() - mean) ** 2).mean(dim=-1)
    return torch.rsqrt(var + eps)


def layernorm_mean_ref(x: Tensor) -> Tensor:
    return x.float().mean(dim=-1)


rmsnorm_fwd_tuned = rmsnorm_fwd
rmsnorm_bwd_tuned = rmsnorm_bwd
