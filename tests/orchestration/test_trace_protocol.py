"""``.kpt`` **协议版本与演进规则**的测试。

这套校验的意义是"宁可拒绝，不要误读"：对一份定义了哈希链的文件，猜着读的
后果是静默产出错误的复核结论。所以这里钉住的不是"能否解析"，而是
**遇到不认识的版本必须拒绝，且拒绝信息要能让人判断是谁太旧**。

同时钉住**向后兼容**：协议早期（正式加校验之前）写出的文件，只要版本号在
受支持集合内，就必须继续可读可复核——`docs/traces/` 里入库的历史产物不能
因为加了校验而失效。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kakeya_ai.primitives import Direction, Tube
from kakeya_ai.proof_orchestrator import ProofOrchestrator
from kakeya_ai.trace_format import (
    MAGIC,
    PROTOCOL,
    PROTOCOL_VERSION,
    SEP,
    SUPPORTED_PROTOCOLS,
    iter_steps,
    peek_trace,
    read_trace,
    verify_trace,
    write_trace,
)


def _make_trace(tmp_path: Path) -> Path:
    tubes = [
        Tube(center=np.array([0.0, 0.0, 0.0]),
             direction=Direction.from_spherical(1.0, 0.0),
             length=1.0, delta=0.1),
        Tube(center=np.array([0.1, 0.1, 0.1]),
             direction=Direction.from_spherical(1.2, 0.5),
             length=1.0, delta=0.1),
    ]
    result = ProofOrchestrator(tubes, max_iterations=2, grid_n=8).run()
    return write_trace(result, tmp_path / "base.kpt", params={"grid_n": 8})


def _with_protocol(path: Path, tmp_path: Path, value: object, name: str) -> Path:
    """把 header 的 protocol 字段替换为 ``value``（可为非字符串）；value 为
    ``_REMOVE`` 时删除该字段。"""
    lines = path.read_text(encoding="utf-8").split("\n")
    header = json.loads(lines[1])
    if value is _REMOVE:
        header.pop("protocol", None)
    else:
        header["protocol"] = value
    lines[1] = json.dumps(header, ensure_ascii=False)
    out = tmp_path / f"{name}.kpt"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


_REMOVE = object()


ALL_ENTRY_POINTS = ("peek", "read", "iter", "verify")


def _call_entry(name: str, path: Path):
    if name == "peek":
        return peek_trace(path)
    if name == "read":
        return read_trace(path)
    if name == "iter":
        return list(iter_steps(path))
    return verify_trace(path, recompute=False)


class TestProtocolConstants:
    def test_current_protocol_is_supported(self):
        """当前写出的协议版本必须在受支持集合里，否则新文件自己都读不了。"""
        assert PROTOCOL in SUPPORTED_PROTOCOLS

    def test_protocol_version_agrees_with_protocol_string(self):
        """``kpt/1`` 与 ``PROTOCOL_VERSION == 1`` 必须一致。

        两个真源一旦不同步，就会出现"改了一个忘了另一个"的静默错误。
        """
        assert PROTOCOL == f"kpt/{PROTOCOL_VERSION}"

    def test_supported_protocols_is_a_frozenset(self):
        assert isinstance(SUPPORTED_PROTOCOLS, frozenset)


class TestWriteDeclaresProtocol:
    def test_written_file_declares_current_protocol(self, tmp_path):
        path = _make_trace(tmp_path)
        header = json.loads(path.read_text(encoding="utf-8").split("\n")[1])
        assert header["protocol"] == PROTOCOL

    def test_written_file_magic_unchanged(self, tmp_path):
        """魔数仍是 KPT1——协议校验是**追加**的，不改变文件首行。"""
        path = _make_trace(tmp_path)
        assert path.read_text(encoding="utf-8").split("\n")[0] == MAGIC


class TestUnsupportedProtocolIsRejected:
    """不认识的版本必须被**所有**入口一致拒绝。"""

    @pytest.mark.parametrize("bad", ["kpt/2", "kpt/99", "kpt/0", "KPT1", ""])
    def test_unknown_version_rejected_everywhere(self, tmp_path, bad):
        base = _make_trace(tmp_path)
        path = _with_protocol(base, tmp_path, bad, f"proto_{bad.replace('/', '_')}")
        for entry in ALL_ENTRY_POINTS:
            with pytest.raises(ValueError):
                _call_entry(entry, path)

    def test_missing_protocol_field_rejected(self, tmp_path):
        base = _make_trace(tmp_path)
        path = _with_protocol(base, tmp_path, _REMOVE, "no_proto")
        for entry in ALL_ENTRY_POINTS:
            with pytest.raises(ValueError):
                _call_entry(entry, path)

    @pytest.mark.parametrize("bad", [1, 1.0, None, True, ["kpt/1"], {"v": 1}])
    def test_non_string_protocol_rejected(self, tmp_path, bad):
        base = _make_trace(tmp_path)
        path = _with_protocol(base, tmp_path, bad, f"type_{type(bad).__name__}")
        for entry in ALL_ENTRY_POINTS:
            with pytest.raises(ValueError):
                _call_entry(entry, path)


class TestErrorMessageIsActionable:
    """拒绝信息必须让人能判断"是文件太旧还是程序太旧"。"""

    def test_error_names_both_file_and_supported_versions(self, tmp_path):
        base = _make_trace(tmp_path)
        path = _with_protocol(base, tmp_path, "kpt/7", "future")
        with pytest.raises(ValueError) as excinfo:
            read_trace(path)
        message = str(excinfo.value)
        assert "kpt/7" in message, "必须报出文件声明的版本"
        assert PROTOCOL in message, "必须报出本程序支持的版本"

    def test_missing_protocol_error_explains_expectation(self, tmp_path):
        base = _make_trace(tmp_path)
        path = _with_protocol(base, tmp_path, _REMOVE, "no_proto_msg")
        with pytest.raises(ValueError) as excinfo:
            read_trace(path)
        assert "protocol" in str(excinfo.value)


class TestBackwardCompatibility:
    """加校验不能弄坏既有文件——这是本次改动的硬约束。"""

    def test_current_protocol_still_round_trips(self, tmp_path):
        path = _make_trace(tmp_path)
        tf = read_trace(path)
        assert tf.header["protocol"] == PROTOCOL
        assert tf.tubes, "几何应被正常还原"

    def test_trace_without_domain_radius_still_verifies(self, tmp_path):
        """协议早期的文件没有 ``domain_radius``，必须仍按历史基准域复核。

        这正是"向后兼容的改动不动版本号"这条规则要保护的场景。
        """
        path = _make_trace(tmp_path)
        header = json.loads(path.read_text(encoding="utf-8").split("\n")[1])
        header["params"].pop("domain_radius", None)
        lines = path.read_text(encoding="utf-8").split("\n")
        lines[1] = json.dumps(header, ensure_ascii=False)
        legacy = tmp_path / "legacy.kpt"
        legacy.write_text("\n".join(lines), encoding="utf-8")

        # 改了 header 的字节，整文件哈希当然不再匹配——那是**预期**的，
        # 恰恰说明字节级哈希在干活。
        # 本用例要验的是别的事：协议校验**不拦**它，且缺 domain_radius
        # 不会让复核过程报错（按历史基准域 1.0 处理）。
        report = verify_trace(legacy, recompute=False)
        assert not report.file_hash_ok, "改了字节，整文件哈希该失效"
        assert report.metadata_ok, "缺 domain_radius 不是元数据错误"
        assert report.geometry_hash_ok, "几何未动，几何哈希该仍然匹配"

        # 且**不会有**任何关于 domain_radius 的抱怨。
        assert not any("domain_radius" in n for n in report.notes)

    def test_legacy_trace_reads_without_error(self, tmp_path):
        """缺 domain_radius 的历史文件必须能被正常读出结构。"""
        path = _make_trace(tmp_path)
        header = json.loads(path.read_text(encoding="utf-8").split("\n")[1])
        header["params"].pop("domain_radius", None)
        lines = path.read_text(encoding="utf-8").split("\n")
        lines[1] = json.dumps(header, ensure_ascii=False)
        legacy = tmp_path / "legacy_read.kpt"
        legacy.write_text("\n".join(lines), encoding="utf-8")

        tf = read_trace(legacy)
        assert "domain_radius" not in tf.params()
        assert tf.header["protocol"] == PROTOCOL

    def test_peek_still_reports_protocol(self, tmp_path):
        path = _make_trace(tmp_path)
        assert peek_trace(path).protocol == PROTOCOL


class TestStructureUnchanged:
    """追加校验不该改变文件结构约定。"""

    def test_still_two_separators(self, tmp_path):
        path = _make_trace(tmp_path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert sum(1 for ln in lines if ln.strip() == SEP) == 2
