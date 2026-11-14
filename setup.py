"""把 C++ 几何内核构建成可安装的 pybind11 扩展。"""
from __future__ import annotations

import platform
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def _extra_compile_args() -> list[str]:
    """按平台选择的激进优化参数。

    这些参数与既有的本地构建脚本（`cpp/build_nano.sh`）保持一致，但在非
    Apple Silicon 主机上会优雅降级。
    """
    args = ["-O3", "-std=c++17", "-Wall", "-pthread"]
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine == "arm64":
        # Apple Silicon —— 针对本机 M 系列芯片调优。
        args.extend(
            [
                "-mcpu=apple-m4",
                "-ffast-math",
                "-funroll-loops",
                "-fvectorize",
                "-fslp-vectorize",
            ]
        )
    elif system == "Linux":
        args.extend(["-march=native", "-ffast-math", "-funroll-loops"])
    return args


ext_modules = [
    Pybind11Extension(
        "kakeya_ai._tube_kernel",
        sources=["cpp/kernel/tube_kernel.cpp"],
        extra_compile_args=_extra_compile_args(),
        cxx_std=17,
    ),
    Pybind11Extension(
        "kakeya_ai._nano_kernel",
        sources=["cpp/kernel/nano_kernel.cpp"],
        extra_compile_args=_extra_compile_args(),
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
