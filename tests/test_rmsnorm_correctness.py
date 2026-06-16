# Copyright 2026 INT21 AI
# SPDX-License-Identifier: MIT

import unittest

import torch

from rmsnorm import layernorm_fwd, rmsnorm, rmsnorm_bwd, rmsnorm_fwd


def _rmsnorm_ref(x, weight=None, residual=None, bias=None, eps=1e-6):
    x_f = x.float()
    if residual is not None:
        x_f = x_f + residual.float()
    rstd = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)
    out = x_f * rstd
    if weight is not None:
        out = out * weight.float()
    if bias is not None:
        out = out + bias.float()
    return out, x_f, rstd.squeeze(-1)


def _assert_close(test, actual, expected, *, atol, rtol, label):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        max_diff = (actual.float() - expected.float()).abs().max().item()
        test.fail(f"{label} mismatch: max_diff={max_diff}")


def _sum_to_shape(values, shape):
    reduce_ndim = values.dim() - len(shape)
    if reduce_ndim == 0:
        return values
    return values.sum(dim=tuple(range(reduce_ndim)))


def _rmsnorm_bwd_expected(x, weight, dout, rstd, dresidual_out=None):
    xhat = x.float() * rstd.unsqueeze(-1)
    wdy = dout.float()
    if weight is not None:
        wdy = wdy * weight.float()
    dx = (wdy - xhat * (xhat * wdy).mean(dim=-1, keepdim=True)) * rstd.unsqueeze(-1)
    if dresidual_out is not None:
        dx = dx + dresidual_out.float()
    dw = _sum_to_shape(dout.float() * xhat, weight.shape) if weight is not None else None
    return dx, dw


class RmsNormFallbackTest(unittest.TestCase):
    def test_public_fallback_honors_output_and_residual_dtype(self):
        torch.manual_seed(0)
        x = torch.randn(3, 16, dtype=torch.float16)
        residual = torch.randn_like(x)
        weight = torch.randn(16, dtype=torch.float32)

        out, residual_out = rmsnorm(
            x,
            weight,
            residual=residual,
            residual_dtype=torch.float32,
            prenorm=True,
        )

        expected, expected_residual, _ = _rmsnorm_ref(x, weight, residual)
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(residual_out.dtype, torch.float32)
        _assert_close(
            self,
            out,
            expected.to(torch.float16),
            atol=3e-2,
            rtol=3e-2,
            label="fallback out",
        )
        _assert_close(
            self,
            residual_out,
            expected_residual,
            atol=1e-5,
            rtol=1e-5,
            label="fallback residual_out",
        )

    def test_public_fallback_honors_out_dtype(self):
        torch.manual_seed(1)
        x = torch.randn(2, 16, dtype=torch.float32)
        weight = torch.randn(16, dtype=torch.float32)

        out = rmsnorm(x, weight, out_dtype=torch.float16)

        expected, _, _ = _rmsnorm_ref(x, weight)
        self.assertEqual(out.dtype, torch.float16)
        _assert_close(
            self,
            out,
            expected.to(torch.float16),
            atol=3e-2,
            rtol=3e-2,
            label="fallback out_dtype",
        )

    def test_public_validates_supported_dtypes(self):
        x = torch.randn(2, 16, dtype=torch.float32)

        with self.assertRaisesRegex(AssertionError, "Unsupported dtype"):
            rmsnorm(torch.ones(2, 16, dtype=torch.int32))
        with self.assertRaisesRegex(AssertionError, "Weight must"):
            rmsnorm(x, torch.ones(16, dtype=torch.int32))
        with self.assertRaisesRegex(AssertionError, "Bias must"):
            rmsnorm(x, None, bias=torch.ones(16, dtype=torch.int32))
        with self.assertRaisesRegex(AssertionError, "Residual must"):
            rmsnorm(x, None, residual=torch.ones_like(x, dtype=torch.int32))
        with self.assertRaisesRegex(AssertionError, "out_dtype must"):
            rmsnorm(x, None, out_dtype=torch.int32)
        with self.assertRaisesRegex(AssertionError, "residual_dtype must"):
            rmsnorm(x, None, residual_dtype=torch.int32)

    def test_fwd_validates_supported_dtypes_before_fallback(self):
        x = torch.randn(2, 16, dtype=torch.float32)

        with self.assertRaisesRegex(AssertionError, "Weight must"):
            rmsnorm_fwd(x, torch.ones(16, dtype=torch.int32))
        with self.assertRaisesRegex(AssertionError, "Residual must"):
            rmsnorm_fwd(x, None, residual=torch.ones_like(x, dtype=torch.int32))

    def test_public_validates_residual_shape(self):
        x = torch.randn(2, 16, dtype=torch.float32)
        residual = torch.randn(1, 16, dtype=torch.float32)
        weight = torch.randn(16, dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "residual shape must match x shape"):
            rmsnorm(x, weight, residual=residual)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for device validation")
    def test_public_validates_tensor_devices(self):
        x = torch.randn(2, 16, device="cuda", dtype=torch.float32)
        weight = torch.randn(16, dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "weight must be on the same device as x"):
            rmsnorm(x, weight)

    def test_public_validates_affine_shape(self):
        x = torch.randn(2, 3, 16, dtype=torch.float32)
        weight = torch.randn(2, 16, dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "per-head weight expected"):
            rmsnorm(x, weight)

    def test_public_validates_input_rank_and_hidden_size(self):
        with self.assertRaisesRegex(ValueError, "at least one dimension"):
            rmsnorm(torch.tensor(1.0))

        with self.assertRaisesRegex(ValueError, "hidden dimension must be nonzero"):
            rmsnorm(torch.empty(2, 0, dtype=torch.float32))

    def test_backward_validates_input_rank_and_hidden_size(self):
        with self.assertRaisesRegex(ValueError, "at least one dimension"):
            rmsnorm_bwd(
                torch.tensor(1.0),
                None,
                torch.tensor(1.0),
                torch.tensor(1.0),
            )

        with self.assertRaisesRegex(ValueError, "hidden dimension must be nonzero"):
            rmsnorm_bwd(
                torch.empty(2, 0, dtype=torch.float32),
                None,
                torch.empty(2, 0, dtype=torch.float32),
                torch.empty(2, dtype=torch.float32),
            )

    def test_backward_validates_rstd_dtype(self):
        x = torch.randn(2, 16, dtype=torch.float32)
        dout = torch.randn_like(x)
        rstd = torch.rsqrt(x.square().mean(dim=-1) + 1e-6).half()

        with self.assertRaisesRegex(ValueError, "rstd must be float32"):
            rmsnorm_bwd(x, None, dout, rstd)

    def test_backward_validates_supported_gradient_dtypes(self):
        x = torch.randn(2, 16, dtype=torch.float32)
        rstd = torch.rsqrt(x.square().mean(dim=-1) + 1e-6)

        with self.assertRaisesRegex(AssertionError, "dout must"):
            rmsnorm_bwd(x, None, torch.ones_like(x, dtype=torch.int32), rstd)
        with self.assertRaisesRegex(AssertionError, "dresidual_out must"):
            rmsnorm_bwd(
                x,
                None,
                torch.randn_like(x),
                rstd,
                dresidual_out=torch.ones_like(x, dtype=torch.int32),
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for device validation")
    def test_backward_validates_tensor_devices(self):
        x = torch.randn(2, 16, device="cuda", dtype=torch.float32)
        dout = torch.randn_like(x)
        rstd = torch.rsqrt(x.cpu().square().mean(dim=-1) + 1e-6)

        with self.assertRaisesRegex(ValueError, "rstd must be on the same device as x"):
            rmsnorm_bwd(x, None, dout, rstd)

    def test_public_fallback_bias_backward_without_weight(self):
        torch.manual_seed(8)
        x = torch.randn(2, 16, dtype=torch.float32, requires_grad=True)
        bias = torch.randn(16, dtype=torch.float32, requires_grad=True)
        upstream = torch.randn_like(x)

        out = rmsnorm(x, None, bias=bias)
        (out * upstream).sum().backward()

        x_f = x.detach().float()
        rstd = torch.rsqrt(x_f.square().mean(dim=-1) + 1e-6)
        xhat = x_f * rstd[:, None]
        expected_dx = (
            upstream - xhat * (xhat * upstream).mean(dim=-1, keepdim=True)
        ) * rstd[:, None]
        expected_db = upstream.sum(dim=0)

        _assert_close(
            self,
            x.grad,
            expected_dx,
            atol=1e-5,
            rtol=1e-5,
            label="fallback bias dx",
        )
        _assert_close(
            self,
            bias.grad,
            expected_db,
            atol=1e-5,
            rtol=1e-5,
            label="fallback bias db",
        )

    def test_backward_fallback_adds_dresidual_out(self):
        torch.manual_seed(10)
        x = torch.randn(2, 16, dtype=torch.float32)
        dout = torch.randn_like(x)
        dresidual_out = torch.randn_like(x)
        rstd = torch.rsqrt(x.float().square().mean(dim=-1) + 1e-6)

        dx, dw, db, dresidual = rmsnorm_bwd(
            x,
            None,
            dout,
            rstd,
            dresidual_out=dresidual_out,
            has_residual=True,
        )

        expected_dx, _ = _rmsnorm_bwd_expected(x, None, dout, rstd, dresidual_out)
        _assert_close(
            self,
            dx,
            expected_dx,
            atol=1e-5,
            rtol=1e-5,
            label="fallback direct residual dx",
        )
        _assert_close(
            self,
            dresidual,
            expected_dx,
            atol=1e-5,
            rtol=1e-5,
            label="fallback direct residual dresidual",
        )
        self.assertIsNone(dw)
        self.assertIsNone(db)

    def test_backward_fallback_reduces_1d_affine_across_all_rows(self):
        torch.manual_seed(15)
        x = torch.randn(2, 3, 16, dtype=torch.float32)
        weight = torch.randn(16, dtype=torch.float32)
        dout = torch.randn_like(x)
        rstd = torch.rsqrt(x.float().square().mean(dim=-1) + 1e-6)

        dx, dw, db, dresidual = rmsnorm_bwd(
            x,
            weight,
            dout,
            rstd,
            has_bias=True,
        )

        expected_dx, expected_dw = _rmsnorm_bwd_expected(x, weight, dout, rstd)
        expected_db = _sum_to_shape(dout.float(), weight.shape)
        self.assertEqual(dw.shape, weight.shape)
        self.assertEqual(db.shape, weight.shape)
        _assert_close(
            self,
            dx,
            expected_dx,
            atol=1e-5,
            rtol=1e-5,
            label="fallback rank3 dx",
        )
        _assert_close(
            self,
            dw,
            expected_dw,
            atol=1e-5,
            rtol=1e-5,
            label="fallback rank3 dw",
        )
        _assert_close(
            self,
            db,
            expected_db,
            atol=1e-5,
            rtol=1e-5,
            label="fallback rank3 db",
        )
        self.assertIsNone(dresidual)

    def test_backward_fallback_reduces_per_head_affine(self):
        torch.manual_seed(16)
        x = torch.randn(2, 3, 16, dtype=torch.float32)
        weight = torch.randn(3, 16, dtype=torch.float32)
        bias = torch.randn(3, 16, dtype=torch.float16)
        dout = torch.randn_like(x)
        rstd = torch.rsqrt(x.float().square().mean(dim=-1) + 1e-6)

        dx, dw, db, dresidual = rmsnorm_bwd(
            x,
            weight,
            dout,
            rstd,
            has_bias=True,
            bias=bias,
        )

        expected_dx, expected_dw = _rmsnorm_bwd_expected(x, weight, dout, rstd)
        expected_db = _sum_to_shape(dout.float(), bias.shape)
        self.assertEqual(dw.shape, weight.shape)
        self.assertEqual(db.shape, bias.shape)
        self.assertEqual(db.dtype, bias.dtype)
        _assert_close(
            self,
            dx,
            expected_dx,
            atol=1e-5,
            rtol=1e-5,
            label="fallback per-head dx",
        )
        _assert_close(
            self,
            dw,
            expected_dw,
            atol=1e-5,
            rtol=1e-5,
            label="fallback per-head dw",
        )
        _assert_close(
            self,
            db,
            expected_db.to(bias.dtype),
            atol=3e-2,
            rtol=3e-2,
            label="fallback per-head db",
        )
        self.assertIsNone(dresidual)


class LayerNormWrapperTest(unittest.TestCase):
    def test_layernorm_fwd_can_return_rstd_without_mean(self):
        torch.manual_seed(5)
        x = torch.randn(3, 16, dtype=torch.float16)
        weight = torch.randn(16, dtype=torch.float32)
        bias = torch.randn(16, dtype=torch.float32)

        out, rstd = layernorm_fwd(x, weight, bias=bias, return_rstd=True)

        x_f = x.float()
        mean = x_f.mean(dim=-1)
        var = ((x_f - mean[:, None]) ** 2).mean(dim=-1)
        expected_rstd = torch.rsqrt(var + 1e-6)
        expected_out = torch.nn.functional.layer_norm(
            x_f, weight.shape, weight.float(), bias.float(), 1e-6
        ).to(x.dtype)
        self.assertEqual(out.dtype, x.dtype)
        _assert_close(
            self,
            out,
            expected_out,
            atol=3e-2,
            rtol=3e-2,
            label="layernorm out",
        )
        _assert_close(
            self,
            rstd,
            expected_rstd,
            atol=1e-5,
            rtol=1e-5,
            label="layernorm rstd",
        )

    def test_layernorm_fwd_can_return_rstd_and_mean(self):
        torch.manual_seed(6)
        x = torch.randn(3, 16, dtype=torch.float32)
        weight = torch.randn(16, dtype=torch.float32)

        _, rstd, mean = layernorm_fwd(
            x, weight, return_rstd=True, return_mean=True
        )

        expected_mean = x.float().mean(dim=-1)
        expected_var = ((x.float() - expected_mean[:, None]) ** 2).mean(dim=-1)
        expected_rstd = torch.rsqrt(expected_var + 1e-6)
        _assert_close(
            self,
            mean,
            expected_mean,
            atol=1e-6,
            rtol=1e-6,
            label="layernorm mean",
        )
        _assert_close(
            self,
            rstd,
            expected_rstd,
            atol=1e-6,
            rtol=1e-6,
            label="layernorm rstd mean",
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for extension tests")
class RmsNormCudaTest(unittest.TestCase):
    def test_forward_fast_paths(self):
        torch.manual_seed(2)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for n in (256, 32768):
                with self.subTest(dtype=dtype, n=n):
                    x = torch.randn(4, n, device="cuda", dtype=dtype)
                    weight = torch.randn(n, device="cuda", dtype=torch.float32)
                    out, residual_out, rstd = rmsnorm_fwd(x, weight, store_rstd=True)
                    torch.cuda.synchronize()

                    expected, _, expected_rstd = _rmsnorm_ref(x, weight)
                    atol = 1e-4 if dtype == torch.float32 else 3e-2
                    rtol = 1e-4 if dtype == torch.float32 else 3e-2
                    _assert_close(
                        self,
                        out,
                        expected.to(dtype),
                        atol=atol,
                        rtol=rtol,
                        label="fwd out",
                    )
                    self.assertEqual(residual_out.data_ptr(), x.data_ptr())
                    _assert_close(
                        self,
                        rstd,
                        expected_rstd,
                        atol=1e-5,
                        rtol=1e-5,
                        label="fwd rstd",
                    )

    def test_forward_no_rstd_exact_path(self):
        torch.manual_seed(23)
        cases = (
            (torch.float16, 8192),
            (torch.bfloat16, 8192),
            (torch.float16, 32768),
            (torch.bfloat16, 32768),
            (torch.float16, 65536),
            (torch.float32, 1024),
            (torch.float32, 16384),
        )
        for dtype, n in cases:
            with self.subTest(dtype=dtype, n=n):
                x = torch.randn(4, n, device="cuda", dtype=dtype)
                weight = torch.randn(n, device="cuda", dtype=torch.float32)
                out, residual_out, rstd = rmsnorm_fwd(x, weight)
                torch.cuda.synchronize()

                expected, _, _ = _rmsnorm_ref(x, weight)
                self.assertIsNone(rstd)
                self.assertEqual(residual_out.data_ptr(), x.data_ptr())
                _assert_close(
                    self,
                    out,
                    expected.to(dtype),
                    atol=1e-4 if dtype == torch.float32 else 3e-2,
                    rtol=1e-4 if dtype == torch.float32 else 3e-2,
                    label="no-rstd fwd out",
                )

    def test_mixed_residual_forward_fast_paths(self):
        torch.manual_seed(3)
        for dtype in (torch.float16, torch.bfloat16):
            for n in (512, 1024, 2048, 4096, 8192, 32768, 65536, 131072):
                with self.subTest(dtype=dtype, n=n):
                    x = torch.randn(4 if n <= 32768 else 2, n, device="cuda", dtype=dtype)
                    residual = torch.randn_like(x)
                    weight = torch.randn(n, device="cuda", dtype=torch.float32)
                    out, residual_out, _ = rmsnorm_fwd(
                        x,
                        weight,
                        residual=residual,
                        residual_dtype=torch.float32,
                    )
                    torch.cuda.synchronize()

                    expected, expected_residual, _ = _rmsnorm_ref(x, weight, residual)
                    _assert_close(
                        self,
                        out,
                        expected.to(dtype),
                        atol=3e-2,
                        rtol=3e-2,
                        label="mixed fwd out",
                    )
                    self.assertEqual(residual_out.dtype, torch.float32)
                    _assert_close(
                        self,
                        residual_out,
                        expected_residual,
                        atol=1e-5,
                        rtol=1e-5,
                        label="mixed residual_out",
                    )

    def test_residual_forward_fast_path_output_dtypes(self):
        torch.manual_seed(33)
        cases = (
            (torch.float16, 256),
            (torch.bfloat16, 256),
            (torch.float32, 256),
            (torch.float32, 512),
            (torch.float32, 2048),
            (torch.float32, 4096),
            (torch.float32, 8192),
            (torch.float32, 16384),
        )
        for dtype, n in cases:
            for residual_dtype in (None, torch.float32):
                with self.subTest(dtype=dtype, n=n, residual_dtype=residual_dtype):
                    x = torch.randn(4, n, device="cuda", dtype=dtype)
                    residual = torch.randn_like(x)
                    weight = torch.randn(n, device="cuda", dtype=torch.float32)
                    out, residual_out, rstd = rmsnorm_fwd(
                        x,
                        weight,
                        residual=residual,
                        residual_dtype=residual_dtype,
                    )
                    torch.cuda.synchronize()

                    expected, expected_residual, _ = _rmsnorm_ref(x, weight, residual)
                    out_atol = 1e-4 if dtype == torch.float32 else 3e-2
                    out_rtol = 1e-4 if dtype == torch.float32 else 3e-2
                    residual_out_dtype = residual_dtype or dtype
                    self.assertIsNone(rstd)
                    self.assertEqual(residual_out.dtype, residual_out_dtype)
                    _assert_close(
                        self,
                        out,
                        expected.to(dtype),
                        atol=out_atol,
                        rtol=out_rtol,
                        label="residual fwd out",
                    )
                    _assert_close(
                        self,
                        residual_out,
                        expected_residual.to(residual_out_dtype),
                        atol=1e-4 if residual_out_dtype == torch.float32 else 3e-2,
                        rtol=1e-4 if residual_out_dtype == torch.float32 else 3e-2,
                        label="residual fwd residual_out",
                    )

    def test_large_row_specialized_forward_paths(self):
        torch.manual_seed(7)
        cases = (
            (torch.float16, None),
            (torch.float32, None),
            (torch.float16, "same"),
            (torch.bfloat16, "same"),
            (torch.float16, "float32"),
            (torch.bfloat16, "float32"),
        )
        n = 262144
        for dtype, residual_mode in cases:
            with self.subTest(dtype=dtype, residual_mode=residual_mode):
                x = torch.randn(2, n, device="cuda", dtype=dtype)
                weight = torch.randn(n, device="cuda", dtype=torch.float32)
                residual = torch.randn_like(x) if residual_mode is not None else None
                residual_dtype = torch.float32 if residual_mode == "float32" else None
                out, residual_out, _ = rmsnorm_fwd(
                    x,
                    weight,
                    residual=residual,
                    residual_dtype=residual_dtype,
                )
                torch.cuda.synchronize()

                expected, expected_residual, _ = _rmsnorm_ref(
                    x, weight, residual
                )
                atol = 1e-4 if dtype == torch.float32 else 3e-2
                rtol = 1e-4 if dtype == torch.float32 else 3e-2
                _assert_close(
                    self,
                    out,
                    expected.to(dtype),
                    atol=atol,
                        rtol=rtol,
                        label="large fwd out",
                    )
                if residual_mode is None:
                    self.assertEqual(residual_out.data_ptr(), x.data_ptr())
                else:
                    expected_dtype = torch.float32 if residual_mode == "float32" else dtype
                    self.assertEqual(residual_out.dtype, expected_dtype)
                    _assert_close(
                        self,
                        residual_out,
                        expected_residual.to(expected_dtype),
                        atol=1e-4 if expected_dtype == torch.float32 else 3e-2,
                        rtol=1e-4 if expected_dtype == torch.float32 else 3e-2,
                        label="large residual_out",
                    )

    def test_backward_fast_paths(self):
        torch.manual_seed(4)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for n in (256, 2048):
                with self.subTest(dtype=dtype, n=n):
                    x = torch.randn(4, n, device="cuda", dtype=dtype)
                    weight = torch.randn(n, device="cuda", dtype=torch.float32)
                    dout = torch.randn_like(x)
                    rstd = torch.rsqrt(x.float().square().mean(dim=-1) + 1e-6)

                    dx, dw, db, dresidual = rmsnorm_bwd(x, weight, dout, rstd)
                    torch.cuda.synchronize()

                    xhat = x.float() * rstd[:, None]
                    wdy = dout.float() * weight
                    expected_dx = (
                        (wdy - xhat * (xhat * wdy).mean(dim=-1, keepdim=True))
                        * rstd[:, None]
                    ).to(dtype)
                    expected_dw = (dout.float() * xhat).sum(dim=0)

                    atol = 1e-4 if dtype == torch.float32 else 3e-2
                    rtol = 1e-4 if dtype == torch.float32 else 3e-2
                    _assert_close(
                        self,
                        dx,
                        expected_dx,
                        atol=atol,
                        rtol=rtol,
                        label="bwd dx",
                    )
                    _assert_close(
                        self,
                        dw,
                        expected_dw,
                        atol=3e-1,
                        rtol=3e-2,
                        label="bwd dw",
                    )
                    self.assertIsNone(db)
                    self.assertIsNone(dresidual)

    def test_public_per_head_weight_backward(self):
        torch.manual_seed(11)
        x = torch.randn(2, 3, 256, device="cuda", dtype=torch.float16, requires_grad=True)
        weight = torch.randn(3, 256, device="cuda", dtype=torch.float32, requires_grad=True)
        upstream = torch.randn_like(x)

        out = rmsnorm(x, weight)
        (out * upstream).sum().backward()
        torch.cuda.synchronize()

        expected, _, rstd = _rmsnorm_ref(x.detach(), weight.detach())
        expected_dx, expected_dw = _rmsnorm_bwd_expected(
            x.detach(), weight.detach(), upstream, rstd
        )
        _assert_close(
            self,
            out,
            expected.to(out.dtype),
            atol=3e-2,
            rtol=3e-2,
            label="per-head public out",
        )
        _assert_close(
            self,
            x.grad,
            expected_dx.to(x.dtype),
            atol=3e-2,
            rtol=3e-2,
            label="per-head public dx",
        )
        _assert_close(
            self,
            weight.grad,
            expected_dw,
            atol=3e-1,
            rtol=3e-2,
            label="per-head public dw",
        )

    def test_public_per_head_mixed_bias_backward(self):
        torch.manual_seed(14)
        x = torch.randn(2, 3, 256, device="cuda", dtype=torch.float16, requires_grad=True)
        weight = torch.randn(3, 256, device="cuda", dtype=torch.float32, requires_grad=True)
        bias = torch.randn(3, 256, device="cuda", dtype=torch.float16, requires_grad=True)
        upstream = torch.randn_like(x)

        out = rmsnorm(x, weight, bias=bias)
        (out * upstream).sum().backward()
        torch.cuda.synchronize()

        expected, _, rstd = _rmsnorm_ref(x.detach(), weight.detach(), bias=bias.detach())
        expected_dx, expected_dw = _rmsnorm_bwd_expected(
            x.detach(), weight.detach(), upstream, rstd
        )
        expected_db = upstream.float().sum(dim=0)
        _assert_close(
            self,
            out,
            expected.to(out.dtype),
            atol=3e-2,
            rtol=3e-2,
            label="per-head bias out",
        )
        _assert_close(
            self,
            x.grad,
            expected_dx.to(x.dtype),
            atol=3e-2,
            rtol=3e-2,
            label="per-head bias dx",
        )
        _assert_close(
            self,
            weight.grad,
            expected_dw,
            atol=3e-1,
            rtol=3e-2,
            label="per-head bias dw",
        )
        self.assertEqual(bias.grad.dtype, bias.dtype)
        _assert_close(
            self,
            bias.grad,
            expected_db.to(bias.dtype),
            atol=3e-2,
            rtol=3e-2,
            label="per-head bias db",
        )

    def test_public_mixed_shared_per_head_affine_falls_back(self):
        torch.manual_seed(17)
        x = torch.randn(2, 3, 256, device="cuda", dtype=torch.float16, requires_grad=True)
        weight = torch.randn(256, device="cuda", dtype=torch.float32, requires_grad=True)
        bias = torch.randn(3, 256, device="cuda", dtype=torch.float32, requires_grad=True)
        upstream = torch.randn_like(x)

        out = rmsnorm(x, weight, bias=bias)
        (out * upstream).sum().backward()
        torch.cuda.synchronize()

        expected, _, rstd = _rmsnorm_ref(x.detach(), weight.detach(), bias=bias.detach())
        expected_dx, expected_dw = _rmsnorm_bwd_expected(
            x.detach(), weight.detach(), upstream, rstd
        )
        expected_db = upstream.float().sum(dim=0)
        _assert_close(
            self,
            out,
            expected.to(out.dtype),
            atol=3e-2,
            rtol=3e-2,
            label="mixed affine fallback out",
        )
        _assert_close(
            self,
            x.grad,
            expected_dx.to(x.dtype),
            atol=3e-2,
            rtol=3e-2,
            label="mixed affine fallback dx",
        )
        _assert_close(
            self,
            weight.grad,
            expected_dw,
            atol=3e-1,
            rtol=3e-2,
            label="mixed affine fallback dw",
        )
        _assert_close(
            self,
            bias.grad,
            expected_db,
            atol=3e-1,
            rtol=3e-2,
            label="mixed affine fallback db",
        )

    def test_backward_mixed_shared_per_head_affine_falls_back(self):
        torch.manual_seed(18)
        x = torch.randn(2, 3, 256, device="cuda", dtype=torch.float16)
        weight = torch.randn(256, device="cuda", dtype=torch.float32)
        bias = torch.randn(3, 256, device="cuda", dtype=torch.float16)
        dout = torch.randn_like(x)
        rstd = torch.rsqrt(x.float().square().mean(dim=-1) + 1e-6)

        dx, dw, db, dresidual = rmsnorm_bwd(
            x,
            weight,
            dout,
            rstd,
            has_bias=True,
            bias=bias,
        )
        torch.cuda.synchronize()

        expected_dx, expected_dw = _rmsnorm_bwd_expected(x, weight, dout, rstd)
        expected_db = dout.float().sum(dim=0)
        _assert_close(
            self,
            dx,
            expected_dx.to(x.dtype),
            atol=3e-2,
            rtol=3e-2,
            label="mixed affine bwd dx",
        )
        _assert_close(
            self,
            dw,
            expected_dw,
            atol=3e-1,
            rtol=3e-2,
            label="mixed affine bwd dw",
        )
        self.assertEqual(db.shape, bias.shape)
        self.assertEqual(db.dtype, bias.dtype)
        _assert_close(
            self,
            db,
            expected_db.to(bias.dtype),
            atol=3e-2,
            rtol=3e-2,
            label="mixed affine bwd db",
        )
        self.assertIsNone(dresidual)

    def test_autograd_residual_only_requires_rstd(self):
        torch.manual_seed(12)
        x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
        residual = torch.randn_like(x, requires_grad=True)
        weight = torch.randn(256, device="cuda", dtype=torch.float32)
        upstream = torch.randn_like(x)

        out = rmsnorm(x, weight, residual=residual)
        (out * upstream).sum().backward()
        torch.cuda.synchronize()

        residual_sum = x.detach().float() + residual.detach().float()
        rstd = torch.rsqrt(residual_sum.square().mean(dim=-1) + 1e-6)
        expected_dx, _ = _rmsnorm_bwd_expected(residual_sum, weight, upstream, rstd)
        _assert_close(
            self,
            residual.grad,
            expected_dx.to(residual.dtype),
            atol=3e-2,
            rtol=3e-2,
            label="residual-only grad",
        )

    def test_mixed_prenorm_residual_direct_grad(self):
        torch.manual_seed(13)
        x = torch.randn(3, 512, device="cuda", dtype=torch.float16, requires_grad=True)
        residual = torch.randn_like(x, requires_grad=True)
        weight = torch.randn(512, device="cuda", dtype=torch.float32)
        upstream = torch.randn_like(x)
        dresidual_out = torch.randn(x.shape, device="cuda", dtype=torch.float32)

        out, residual_out = rmsnorm(
            x,
            weight,
            residual=residual,
            residual_dtype=torch.float32,
            prenorm=True,
        )
        (out.float() * upstream.float()).sum().backward(retain_graph=True)
        norm_x_grad = x.grad.detach().clone()
        norm_residual_grad = residual.grad.detach().clone()
        x.grad = None
        residual.grad = None
        (out.float() * upstream.float()).sum().backward(retain_graph=True)
        (residual_out * dresidual_out).sum().backward()
        torch.cuda.synchronize()

        self.assertEqual(residual_out.dtype, torch.float32)
        residual_sum = x.detach().float() + residual.detach().float()
        rstd = torch.rsqrt(residual_sum.square().mean(dim=-1) + 1e-6)
        expected_dx, _ = _rmsnorm_bwd_expected(
            residual_sum, weight, upstream, rstd, dresidual_out
        )
        _assert_close(
            self,
            norm_x_grad,
            (expected_dx - dresidual_out).to(x.dtype),
            atol=3e-2,
            rtol=3e-2,
            label="mixed prenorm norm-only dx",
        )
        _assert_close(
            self,
            norm_residual_grad,
            (expected_dx - dresidual_out).to(residual.dtype),
            atol=3e-2,
            rtol=3e-2,
            label="mixed prenorm norm-only dresidual",
        )
        _assert_close(
            self,
            x.grad,
            expected_dx.to(x.dtype),
            atol=3e-2,
            rtol=3e-2,
            label="mixed prenorm dx",
        )
        _assert_close(
            self,
            residual.grad,
            expected_dx.to(residual.dtype),
            atol=3e-2,
            rtol=3e-2,
            label="mixed prenorm dresidual",
        )

    def test_autograd_bias_backward_without_weight(self):
        torch.manual_seed(9)
        x = torch.randn(4, 256, device="cuda", dtype=torch.float32, requires_grad=True)
        bias = torch.randn(256, device="cuda", dtype=torch.float32, requires_grad=True)
        upstream = torch.randn_like(x)

        out = rmsnorm(x, None, bias=bias)
        (out * upstream).sum().backward()
        torch.cuda.synchronize()

        x_f = x.detach().float()
        rstd = torch.rsqrt(x_f.square().mean(dim=-1) + 1e-6)
        xhat = x_f * rstd[:, None]
        expected_dx = (
            upstream - xhat * (xhat * upstream).mean(dim=-1, keepdim=True)
        ) * rstd[:, None]
        expected_db = upstream.sum(dim=0)

        _assert_close(
            self,
            x.grad,
            expected_dx,
            atol=1e-4,
            rtol=1e-4,
            label="cuda bias dx",
        )
        _assert_close(
            self,
            bias.grad,
            expected_db,
            atol=1e-4,
            rtol=1e-4,
            label="cuda bias db",
        )


if __name__ == "__main__":
    unittest.main()
