"""C++ 扩展加载器：**全库唯一的加载真源**（2026-10-07）。

本项目有三处需要加载可选编译扩展的地方——``fast_kernel`` / ``nano_kernel``
（体积与包含测试的热点）与 ``grains_cpp``（分格热点）。此前这三处**各自抄了
一份加载逻辑**，于是同一件事有三份实现。这正是局限 #22 记下的问题：

    > `accel.py` / `fast_kernel` / `nano_kernel` 各自的加载逻辑仍是三份独立
    > 实现，没有单一真源 ⇒ 同类不一致可能仍潜藏于其中。

实测已经证明"各写各的"会出事：``grains_cpp`` 的那一份**只试了"已安装到包内"
一条路径**，失败即 ``raise ImportError``，把"可选加速缺失"升级成"功能不可用"
（#22 正文）。把加载逻辑收敛到本模块后，这类不一致**不可能再出现**——
只有一个地方可以出错，且它被测。

约定（与内部设计文档 §ABI/编译风险 一致）
------------------------------------------------

**任何 C++ 加载失败都静默返回 ``None``**，由调用方决定回退（本项目一律回退
纯 Python）。本模块**从不抛异常**——这是那条约定的落点。

三级回退（顺序即优先级）
------------------------

1. **包内**：``from kakeya_ai import <modname>``（``pip install -e .`` 的产物）；
2. **legacy ``cpp/`` 目录**：手工 ``bash cpp/build_nano.sh`` 的产物，
   先试 ``__file__`` 旁的 ``cpp/``、再试 CWD 下的 ``cpp/``；
3. **顶层导入**：用户自行放进 ``PYTHONPATH`` 的模块。
"""

from __future__ import annotations

import importlib.util
import os
import sysconfig
from types import ModuleType
from typing import Optional

#: 已加载模块的缓存：``modname -> ModuleType``（加载失败则不落键）。
#:
#: **为什么缓存值得**：加载一次要动 ``importlib`` 与 dlopen，而
#: ``is_cpp_available()`` 会被热点路径反复调用（``fast_kernel`` 每个kernel
#: 函数入口都查一次）。缓存让重复查询变成一次 dict 命中。
_LOADED: dict[str, ModuleType] = {}


def load_cpp_extension(modname: str) -> Optional[ModuleType]:
    """按三级回退加载名为 ``modname`` 的 C++ 扩展；全失败返回 ``None``。

    参数
    ----
    modname:
        扩展的模块名（如 ``"_tube_kernel"`` / ``"_nano_kernel"``）。**不带包前缀**——
        包内那级会自己拼 ``kakeya_ai.``。

    返回
    ----
    加载到的模块对象；三级都失败时返回 ``None``。**不抛异常**（见模块 docstring）。

    缓存
    ----
    成功的加载结果按 ``modname`` 缓存；随后同名调用直接返回缓存。**失败的
    结果不缓存**——否则一次瞬时失败（如并发 dlopen、临时缺文件）会让后续
    调用永久看不到扩展。
    """
    cached = _LOADED.get(modname)
    if cached is not None:
        return cached

    module = _try_package(modname)
    if module is None:
        module = _try_legacy_dir(modname)
    if module is None:
        module = _try_top_level(modname)

    if module is not None:
        _LOADED[modname] = module
    return module


def _try_package(modname: str) -> Optional[ModuleType]:
    """    第 1 级：包内自带的编译产物（由 ``pip install`` / ``setup.py`` 构建）。"""
    try:
        # 编译扩展只有 ``.pyi`` 存根、没有 ``.py`` 源码，故 pyright 报
        # ``reportMissingModuleSource`` —— 这是"可选编译产物"的固有现象。
        pkg = importlib.import_module("kakeya_ai")
        return getattr(pkg, modname)
    except Exception:  # 刻意兜底：任何失败都继续降级（BLE001 未启用，不加 noqa）
        return None


def _try_legacy_dir(modname: str) -> Optional[ModuleType]:
    """第 2 级：源码树旁的 legacy ``cpp/`` 目录（手工构建产物）。"""
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    candidates = (
        os.path.join(os.path.dirname(__file__), "..", "..", "cpp"),
        os.path.join(os.getcwd(), "cpp"),
    )
    for candidate in candidates:
        so_path = os.path.abspath(os.path.join(candidate, f"{modname}{ext_suffix}"))
        if not os.path.exists(so_path):
            continue
        module = _load_from_path(modname, so_path)
        if module is not None:
            return module
    return None


def _load_from_path(modname: str, so_path: str) -> Optional[ModuleType]:
    """从一个具体 ``.so`` 路径加载扩展；失败返回 ``None``。"""
    try:
        spec = importlib.util.spec_from_file_location(modname, so_path)
        # spec_from_file_location 的返回类型是 Optional：路径存在但扩展名
        # 不被识别时会给 None。不检查就 module_from_spec(None) 会抛 TypeError。
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:  # 刻意兜底：换下一个候选/下一级（BLE001 未启用，不加 noqa）
        return None


def _try_top_level(modname: str) -> Optional[ModuleType]:
    """第 3 级：顶层导入（用户自置于 ``PYTHONPATH``）。"""
    try:
        return importlib.import_module(modname)
    except ImportError:
        return None


def reset_cache() -> None:
    """清空加载缓存（**测试用**：让每条用例从干净状态出发）。"""
    _LOADED.clear()
