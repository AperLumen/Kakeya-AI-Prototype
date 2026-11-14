"""``grains_cpp`` 的**首次**测试：C++ 扩展缺失时必须回退，而非崩溃。

**为什么现在才写**：该模块自落地起**无任何测试、无任何调用方**（覆盖率 0%），
于是下面这个缺陷从"注释承诺"到"实现"的落差**长期无人发现**：

* 文件开头注释写着 "fall back to the legacy ``cpp/`` directory layout if the
  extension has not been rebuilt yet"，
* 但代码**只试了"已安装到包内"那一条路径**，失败即 ``_tk = None``，
  然后 ``_assign_tubes_to_grid_cells`` **直接 ``raise ImportError``**
  ⇒ **注释与实现不符**，且违反本项目"任何 C++ 加载失败都静默切回纯 Python"
  的约定（内部设计文档 §ABI/编译风险）。

**实测证据（2026-10-03）**：本机 ``cpp/_tube_kernel.cpython-312-darwin.so``
**存在且可加载**（含 ``assign_tubes_to_grid_cells``），但 ``grains_cpp`` 仍报
``ImportError: C++ tube_kernel extension is not available`` —— 纯粹因为**没试
legacy 路径**。

**本文件守两件事**（都不依赖机器上是否真的编译了扩展）：

1. **回退语义**：扩展不可用时**不得抛异常**，应返回 ``None`` 并由调用方
   走纯 Python；且纯 Python 回退与 C++ 版**判据一致**。
2. **三级回退的完整性**：包内 → legacy ``cpp/`` → 顶层导入（与 ``fast_kernel``
   同一套语义），任一级都不应让"可选加速缺失"升级成"功能不可用"。
"""

from __future__ import annotations

import numpy as np
import pytest

from kakeya_ai import grains_cpp, grains_fast
from kakeya_ai.primitives import Direction, Tube


def _fan(n: int = 6) -> list[Tube]:
    """规则扇形构型（与 ``test_grains_fast.py`` 同一造法）。"""
    return [
        Tube(
            center=np.zeros(3),
            direction=Direction.from_spherical(theta, phi),
            length=1.0,
            delta=0.05,
        )
        for theta in np.linspace(0.1, np.pi - 0.1, n)
        for phi in np.linspace(0, 2 * np.pi, n, endpoint=False)
    ]


class TestFallbackSemantics:
    """扩展缺失时必须是"降级"，不是"不可用"。"""

    def test_assign_returns_none_when_extension_missing(self, monkeypatch):
        """``_tk is None`` 时返回 ``None``（**不抛异常**）。

        这是本文件的核心断言——修复前这里是 ``raise ImportError``，
        与文件头注释承诺的 fallback 相反。
        """
        monkeypatch.setattr(grains_cpp, "_tk", None)
        bboxes = np.zeros((2, 6), dtype=float)
        result = grains_cpp._assign_tubes_to_grid_cells(
            bboxes, np.zeros(3), np.ones(3), 2, 2, 2
        )
        assert result is None, (
            "扩展缺失时应返回 None 交给调用方回退，而不是抛异常 —— "
            "本项目约定'任何 C++ 加载失败都静默切回纯 Python'"
        )

    def test_pure_python_fallback_matches_cpp_on_same_bboxes(self):
        """纯 Python 回退与 **C++ 版**在同一 bbox 集合上给出**逐位相同**的结果。

        **为什么不用 ``grains_fast`` 对拍**（原设计，已改）：它的分格热点是
        **内联在 ``guth_grains_skeleton`` 里**的，没有可独立调用的函数
        （实测确认）——硬要比就得端到端造管集，而那会把"分格判据是否一致"与
        "上层几何是否一致"两件事混在一起。**直接与 C++ 对拍更聚焦**。

        本机未构建扩展时自动 skip（不把"没编译"误判成"代码错"）。
        """
        if grains_cpp._tk is None:
            pytest.skip("本机未加载到 C++ 扩展，无法对拍")

        # 造几类包围盒：规整、小格、越界、完全在网格外
        bboxes = np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [0.2, 0.3, 0.2, 0.3, 0.2, 0.3],
                [0.9, 1.1, 0.4, 0.6, 0.1, 0.2],   # hi 越过网格上界
                [5.0, 6.0, 5.0, 6.0, 5.0, 6.0],   # 完全在网格外
            ],
            dtype=float,
        )
        lo, hi = np.zeros(3), np.ones(3)
        nx = ny = nz = 4

        cpp = grains_cpp._assign_tubes_to_grid_cells(bboxes, lo, hi, nx, ny, nz)
        assert cpp is not None
        py_ids, py_loc = grains_cpp._assign_tubes_to_grid_cells_py(
            bboxes, lo, hi, nx, ny, nz
        )
        cpp_ids, cpp_loc = cpp

        # 顺序可能不同（C++ 与 Python 的遍历序不保证一致），故按 (cell, tube) 集合比
        cpp_pairs = sorted(zip(cpp_ids.tolist(), cpp_loc.tolist(), strict=True))
        py_pairs = sorted(zip(py_ids.tolist(), py_loc.tolist(), strict=True))
        assert py_pairs == cpp_pairs, (
            "纯 Python 回退与 C++ 版的分格结果不一致 —— 两者应当是同一判据\n"
            f"  python {len(py_pairs)} 对, cpp {len(cpp_pairs)} 对\n"
            f"  仅 python 有: {sorted(set(py_pairs) - set(cpp_pairs))[:5]}\n"
            f"  仅 cpp 有:    {sorted(set(cpp_pairs) - set(py_pairs))[:5]}"
        )

    def test_fallback_produces_same_prism_count_as_grains_fast(self):
        """端到端：强制走回退路径时，``grains_cpp`` 与 ``grains_fast`` 结果一致。

        这条覆盖"调用方接住了 ``None``"这一环——只测 ``_assign_*`` 返回 ``None``
        是不够的：**若调用方没处理 ``None``，仍会在解包时崩溃**。
        """
        tubes = _fan()
        expected = len(grains_fast.guth_grains_skeleton(tubes, None).prisms)

        # 强制回退（不改生产代码，只在该测试内）
        import kakeya_ai.grains_cpp as gc

        original = gc._assign_tubes_to_grid_cells
        try:
            gc._assign_tubes_to_grid_cells = lambda *a, **kw: None  # type: ignore[assignment]
            got = len(gc.guth_grains_skeleton(tubes, None).prisms)
        finally:
            gc._assign_tubes_to_grid_cells = original  # type: ignore[assignment]

        assert got == expected, (
            f"回退路径产出 {got} 个 prism，grains_fast 是 {expected} —— "
            "两者判据应当一致（同一个 bbox-overlap 判据）"
        )


class TestLoaderTriesLegacyPath:
    """加载器必须真的尝试 legacy ``cpp/`` 路径（注释早就承诺了）。"""

    def test_loader_is_a_function_with_fallback_chain(self):
        """本模块应有一个**独立可测**的加载函数（而非 import 期的裸 try）。

        修复前是模块级 ``try: from kakeya_ai import _tube_kernel`` —— 只有一级，
        且无法单测"legacy 路径"这一级。
        """
        assert hasattr(grains_cpp, "_load_tube_kernel"), (
            "缺少 _load_tube_kernel()——回退链应当是独立可测的函数，"
            "与 fast_kernel._load_cpp_module 同一形态"
        )
        assert callable(grains_cpp._load_tube_kernel)

    def test_loader_finds_legacy_cpp_directory_if_built(self):
        """若源码树旁有 legacy 构建产物，加载器**必须**能加载它。

        本机未构建时自动 skip（不把"没编译"误判成"代码错"）。
        """
        import sysconfig
        from pathlib import Path

        ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        repo_cpp = Path(grains_cpp.__file__).resolve().parents[2] / "cpp"
        so = repo_cpp / f"_tube_kernel{ext}"
        if not so.exists():
            pytest.skip(f"本机未构建 legacy 扩展（{so.name} 不存在）")

        module = grains_cpp._load_tube_kernel()
        assert module is not None, (
            f"legacy 扩展存在于 {so} 但加载器没找到它 —— "
            "这正是 2026-10-03 修掉的那个缺陷"
        )
        assert hasattr(module, "assign_tubes_to_grid_cells"), (
            "加载到的模块缺少 assign_tubes_to_grid_cells —— 可能加载错了东西"
        )
