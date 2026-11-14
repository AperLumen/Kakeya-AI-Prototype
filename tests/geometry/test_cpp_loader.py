"""守门：C++ 扩展加载是**全库唯一的真源**（局限 #22 的收敛）。

**为什么单独守这件事**：此前 ``fast_kernel`` / ``nano_kernel`` / ``grains_cpp``
**各抄了一份三级回退**。重复本身不是问题，"三份实现会各自漂移"才是——实测就
漂过：``grains_cpp`` 那一份只试了"包内"一级，把"可选加速缺失"升级成了"功能
不可用"（#22 正文）。2026-10-07 把三级回退收敛进 :mod:``kakeya_ai.cpp_loader``
后，这类漂移**在结构上不可能**——只剩一个地方能出错。

**本文件守三件事**：

1. **语法级**：三个加载点都不再自带 ``spec_from_file_location`` / ``sysconfig``
   这类回退细节，只调用 ``cpp_loader``；后端模块里不再出现 ``importlib.util``。
   这是"单一真源"的**可机器判定**判据（不靠人读注释）。
2. **语义级**：``cpp_loader`` 三级有序回退、失败返回 ``None`` 且**绝不抛异常**，
   包内命中时**不再**降级，CWD 旁的 legacy 目录也能命中。
3. **一致性**：三个加载点对同一扩展给出**同一个模块对象**（共享缓存/同一 ``.so``）。

这些用例都**不依赖本机是否真的编译了扩展**——用 ``monkeypatch`` 注入式验证，
所以 CI（.so 已构建）与本机（可能没构建）跑出的结论一致。
"""

from __future__ import annotations

import importlib
import sysconfig
import textwrap
from pathlib import Path

import pytest

from kakeya_ai import cpp_loader, fast_kernel, grains_cpp, nano_kernel

REPO = Path(__file__).resolve().parents[2]


def _source(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


class TestSingleSourceOfTruth:
    """三个加载点只调用 ``cpp_loader``，不再各写一份回退细节。"""

    @pytest.mark.parametrize(
        "module", [fast_kernel, nano_kernel, grains_cpp], ids=lambda m: m.__name__
    )
    def test_backend_does_not_reimplement_fallback(self, module):
        """后端模块**不得**再出现回退链的实现细节。

        判据是**语法级**的（扫源码），因此与机器上是否编译了扩展无关：
        只要还有一处 ``spec_from_file_location`` / ``sysconfig.get_config_var``，
        就说明回退逻辑又在某处被复制了一份——正是 #22 的形状。
        """
        src = _source(module)
        forbidden = [
            "spec_from_file_location",  # legacy 目录加载的实现细节
            "get_config_var",           # EXT_SUFFIX 推导
            "module_from_spec",
        ]
        leaked = [name for name in forbidden if name in src]
        assert not leaked, (
            f"{module.__name__} 仍自带 C++ 回退实现细节 {leaked} —— "
            "三级回退应只在 kakeya_ai.cpp_loader 里实现一次（局限 #22）"
        )

    @pytest.mark.parametrize(
        "module", [fast_kernel, nano_kernel, grains_cpp], ids=lambda m: m.__name__
    )
    def test_backend_delegates_to_cpp_loader(self, module):
        """三个加载点都必须**真的**引用 ``cpp_loader``（而不是只删了旧代码）。"""
        src = _source(module)
        assert "cpp_loader" in src, (
            f"{module.__name__} 没有引用 cpp_loader —— "
            "它应当把加载委托给全库唯一真源"
        )
        assert "load_cpp_extension" in src, (
            f"{module.__name__} 引用了 cpp_loader，但没调用 load_cpp_extension()"
        )

    def test_cpp_loader_is_the_only_importlib_user(self):
        """只有 ``cpp_loader`` 允许 import ``importlib.util``。

        ``fast_kernel`` / ``nano_kernel`` / ``grains_cpp`` 此前都直接 import 它。
        现在它们应改用 ``cpp_loader``，故不该再出现这个 import。
        """
        for module in (fast_kernel, nano_kernel, grains_cpp):
            src = _source(module)
            assert "importlib.util" not in src, (
                f"{module.__name__} 仍直接 import importlib.util —— "
                "路径加载应只在 cpp_loader 内发生"
            )


class TestFallbackSemantics:
    """``cpp_loader`` 的三级回退语义（注入式，与是否编译无关）。"""

    def test_returns_none_when_all_levels_fail(self, monkeypatch):
        """三级全失败时返回 ``None``，**不抛异常**（本项目约定）。"""
        monkeypatch.setattr(cpp_loader, "_LOADED", {})
        monkeypatch.setattr(cpp_loader, "_try_package", lambda modname: None)
        monkeypatch.setattr(cpp_loader, "_try_legacy_dir", lambda modname: None)
        monkeypatch.setattr(cpp_loader, "_try_top_level", lambda modname: None)

        assert cpp_loader.load_cpp_extension("_definitely_absent") is None

    def test_package_level_wins_and_stops_descent(self, monkeypatch):
        """包内命中即返回，**不再**降级（三级是"或"，不是"并"）。"""
        sentinel = object()
        monkeypatch.setattr(cpp_loader, "_LOADED", {})
        monkeypatch.setattr(cpp_loader, "_try_package", lambda modname: sentinel)

        def _boom(modname):  # 若被调用即说明没在包内那级停住
            raise AssertionError("包内命中后不应再尝试 legacy 级")

        monkeypatch.setattr(cpp_loader, "_try_legacy_dir", _boom)
        assert cpp_loader.load_cpp_extension("_x") is sentinel

    def test_legacy_dir_used_when_package_missing(self, monkeypatch):
        """包内缺失时降级到 legacy ``cpp/`` 目录。"""
        sentinel = object()
        monkeypatch.setattr(cpp_loader, "_LOADED", {})
        monkeypatch.setattr(cpp_loader, "_try_package", lambda modname: None)
        monkeypatch.setattr(cpp_loader, "_try_legacy_dir", lambda modname: sentinel)
        assert cpp_loader.load_cpp_extension("_x") is sentinel

    def test_top_level_used_as_last_resort(self, monkeypatch):
        """包内与 legacy 都缺失时，才轮到顶层导入。"""
        sentinel = object()
        monkeypatch.setattr(cpp_loader, "_LOADED", {})
        monkeypatch.setattr(cpp_loader, "_try_package", lambda modname: None)
        monkeypatch.setattr(cpp_loader, "_try_legacy_dir", lambda modname: None)
        monkeypatch.setattr(cpp_loader, "_try_top_level", lambda modname: sentinel)
        assert cpp_loader.load_cpp_extension("_x") is sentinel

    def test_failure_is_not_cached(self, monkeypatch):
        """失败**不缓存**：一次瞬时失败不该让后续调用永久失明。"""
        monkeypatch.setattr(cpp_loader, "_LOADED", {})
        monkeypatch.setattr(cpp_loader, "_try_legacy_dir", lambda modname: None)
        monkeypatch.setattr(cpp_loader, "_try_top_level", lambda modname: None)

        monkeypatch.setattr(cpp_loader, "_try_package", lambda modname: None)
        assert cpp_loader.load_cpp_extension("_x") is None
        assert "_x" not in cpp_loader._LOADED

        sentinel = object()
        monkeypatch.setattr(cpp_loader, "_try_package", lambda modname: sentinel)
        assert cpp_loader.load_cpp_extension("_x") is sentinel

    def test_success_is_cached(self, monkeypatch):
        """成功要缓存：第二次调用不再走回退链（热点路径反复查可用性）。"""
        sentinel = object()
        calls = {"n": 0}
        monkeypatch.setattr(cpp_loader, "_LOADED", {})

        def _pkg(modname):
            calls["n"] += 1
            return sentinel

        monkeypatch.setattr(cpp_loader, "_try_package", _pkg)
        assert cpp_loader.load_cpp_extension("_x") is sentinel
        assert cpp_loader.load_cpp_extension("_x") is sentinel
        assert calls["n"] == 1, "第二次调用应命中缓存，不再重新加载"


class TestLegacyDirResolution:
    """legacy ``cpp/`` 目录的查找逻辑（真文件系统，不 skip）。"""

    def test_loads_from_cwd_relative_cpp_dir(self, monkeypatch, tmp_path):
        """CWD 下的 ``cpp/`` 也是候选之一（降级链的一条）。

        造一个**假的** ``cpp/<modname><EXT_SUFFIX>`` 文件（内容不是真扩展，
        dlopen 会失败）——但对拍的是"**路径被找到并尝试**"这一事实，用
        ``_load_from_path`` 打桩即可观测到它被带着正确路径调用。
        """
        ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        fake_cpp = tmp_path / "cpp"
        fake_cpp.mkdir()
        fake_so = fake_cpp / f"_fakemod{ext}"
        fake_so.write_bytes(b"not a real extension")

        seen: list[str] = []
        monkeypatch.setattr(
            cpp_loader, "_load_from_path",
            lambda modname, so_path: seen.append(so_path) or "loaded",
        )
        monkeypatch.setattr(cpp_loader, "_LOADED", {})
        monkeypatch.chdir(tmp_path)

        result = cpp_loader._try_legacy_dir("_fakemod")
        assert result == "loaded"
        assert str(fake_so) in seen, (
            f"CWD 下的 {fake_so} 未被候选路径命中 —— legacy 查找应含 CWD/cpp"
        )


class TestBackendsAgreeOnSameExtension:
    """三个加载点对同一扩展应给出**同一个模块对象**（共享同一 ``.so``）。"""

    def test_fast_and_grains_share_tube_kernel(self):
        """``fast_kernel`` 与 ``grains_cpp`` 都加载 ``_tube_kernel`` ⇒ 同一对象。

        扩展未构建时自动 skip（不把"没编译"误判成"代码错"）。
        """
        if not fast_kernel.is_cpp_available() or grains_cpp._tk is None:
            pytest.skip("本机未构建 C++ 扩展")

        assert fast_kernel._CPP_MODULE is grains_cpp._tk, (
            "两个加载点给出了不同的 _tube_kernel 对象 —— "
            "它们应当通过 cpp_loader 共享同一次加载"
        )

    def test_nano_is_a_distinct_extension(self):
        """``nano_kernel`` 是**另一个**扩展（``_nano_kernel``），不应与 tube 混同。"""
        if not nano_kernel.is_cpp_available() or not fast_kernel.is_cpp_available():
            pytest.skip("本机未构建 C++ 扩展")

        assert nano_kernel._CPP_MODULE is not fast_kernel._CPP_MODULE, (
            "_nano_kernel 与 _tube_kernel 是不同扩展，不应是同一对象"
        )
        assert nano_kernel._CPP_MODULE.__file__ != fast_kernel._CPP_MODULE.__file__
