# Copyright 2026 INT21 AI
# SPDX-License-Identifier: MIT

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def _default_arch_list() -> str:
    if not torch.cuda.is_available():
        return ""
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) == (9, 0):
        return "9.0a"
    if (major, minor) == (10, 0):
        return "10.0a"
    return f"{major}.{minor}"


@lru_cache(maxsize=1)
def get_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("rmsnorm CUDA extension requires a CUDA device")

    arch_list = _default_arch_list()
    if arch_list:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list)

    root = Path(__file__).resolve().parent
    sources = [
        root / "csrc" / "torch_bindings.cpp",
        root / "csrc" / "ptx_rmsnorm_kernels.cu",
    ]
    return load(
        name="rmsnorm_ptx_ext",
        sources=[str(p) for p in sources],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=os.environ.get("RMSNORM_VERBOSE_BUILD", "0") == "1",
        with_cuda=True,
    )
