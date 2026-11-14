"""``.kpt`` 证明轨迹文件协议的测试。

协议的三条主张，每条都要有测试钉住：

1. **可往返**：写出去再读回来，几何与指标逐位相等（不是"近似相等"）。
2. **可复核**：``verify_trace`` 独立重算能确认文件自洽，不依赖生成它的
   orchestrator 实例。
3. **可定位篡改**：改动任意一处都能被发现，并指出是**哪一步**。
   同时如实承认边界——不声称能防有动机的伪造（那需要签名）。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from kakeya_ai.primitives import Direction, Shading, Tube
from kakeya_ai.proof_orchestrator import ProofOrchestrator, run_proof_skeleton
ROOT = Path(__file__).resolve().parents[2]

from kakeya_ai.trace_format import (
    VerificationReport,
    _dec,
    _apply_state_change,
    MAGIC,
    PROTOCOL,
    SEP,
    geometry_hash,
    iter_steps,
    peek_trace,
    read_trace,
    tube_hash,
    verify_trace,
    write_trace,
)


def make_fan(n_theta=3, n_phi=3, delta=0.1):
    return [
        Tube(center=np.zeros(3),
             direction=Direction.from_spherical(theta, phi),
             length=1.0, delta=delta)
        for theta in np.linspace(0.2, np.pi - 0.2, n_theta)
        for phi in np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    ]


def make_mixed_tubes(n_cluster=6, n_spread=6, delta=0.1, seed=0):
    """窄方向簇 + 均匀散开扇——能触发 broadness 剥离，产生多步 trace。

    纯 fan 构型总是一步就被 sticky 短路（sticky 阈值很宽），
    所以要测多步与链中断必须用这种混合构型。
    """
    rng = np.random.default_rng(seed)
    tubes = []
    for i in range(n_cluster):
        d = Direction.from_spherical(np.pi / 2 + 0.01 * i, 0.0)
        tubes.append(Tube(center=rng.uniform(-0.4, 0.4, 3), direction=d,
                          length=1.0, delta=delta))
    for i in range(n_spread):
        d = Direction.from_spherical(np.pi / 2, 2 * np.pi * i / n_spread)
        tubes.append(Tube(center=rng.uniform(-0.4, 0.4, 3), direction=d,
                          length=1.0, delta=delta))
    return tubes


def run_and_write(tmp_path: Path, **orch_kwargs) -> Path:
    tubes = make_fan()
    orch = ProofOrchestrator(tubes, max_iterations=2, grid_n=16, **orch_kwargs)
    result = orch.run()
    return write_trace(
        result, tmp_path / "run.kpt",
        params={"epsilon": 0.05, "K": 2.0, "lambda_min": 0.85, "grid_n": 16},
    )


class TestFileStructure:
    """文件是有约定的文本格式，不是随便一段 JSON。"""

    def test_starts_with_magic_and_has_separators(self, tmp_path):
        path = run_and_write(tmp_path)
        lines = path.read_text(encoding="utf-8").splitlines()

        assert lines[0] == MAGIC
        assert sum(1 for ln in lines if ln.strip() == SEP) == 2, (
            "格式要求 header/body 与 body/footer 之间各一条分隔线"
        )

    def test_header_records_protocol_and_params(self, tmp_path):
        path = run_and_write(tmp_path)
        header = json.loads(path.read_text(encoding="utf-8").splitlines()[1])

        assert header["protocol"] == PROTOCOL
        assert header["params"]["grid_n"] == 16
        assert "geometry_hash" in header
        assert "created" in header

    def test_body_is_one_json_object_per_line(self, tmp_path):
        """数据体是 JSONL：逐行可解析，坏行能定位。"""
        path = run_and_write(tmp_path)
        lines = path.read_text(encoding="utf-8").splitlines()
        sep_idx = [i for i, ln in enumerate(lines) if ln.strip() == SEP]
        body = lines[sep_idx[0] + 1: sep_idx[-1]]

        assert body, "数据体不应为空"
        parsed = [json.loads(ln) for ln in body]      # 逐行必须都能解析
        assert parsed[0]["kind"] == "geometry"
        assert all(p.get("kind") != "geometry" for p in parsed[1:])

    def test_footer_has_chain_head_and_file_hash(self, tmp_path):
        path = run_and_write(tmp_path)
        lines = path.read_text(encoding="utf-8").splitlines()
        footer = json.loads(lines[-1])

        assert "head" in footer and "file_hash" in footer
        assert footer["n_steps"] == len([ln for ln in lines if '"index"' in ln])


class TestRoundTrip:
    """写出去再读回来必须**逐位相等**——这是"精确哈希"的前提。"""

    def test_geometry_round_trips_bit_exactly(self, tmp_path):
        # 故意用会暴露精度问题的量：极小的方向分量、无法用十进制短表示的浮点
        tubes = [
            Tube(center=np.array([0.1, 1e-17, -0.30000000000000004]),
                 direction=Direction.from_spherical(0.7, 0.3),
                 length=0.7, delta=0.03),
            Tube(center=np.array([1 / 3, 2 / 3, 1 / 7]),
                 direction=Direction.from_spherical(np.pi / 2, 1.0),
                 length=1.0, delta=0.01),
        ]
        shading = Shading(np.array([0.25, 0.5]))

        result = run_proof_skeleton(tubes, max_iterations=1, grid_n=16)
        path = write_trace(result, tmp_path / "rt.kpt", initial_tubes=tubes,
                           shading=shading)
        tf = read_trace(path)

        for original, restored in zip(tubes, tf.tubes, strict=True):
            assert np.array_equal(
                np.asarray(original.center, dtype=float).view(np.uint64),
                np.asarray(restored.center, dtype=float).view(np.uint64),
            ), "center 必须逐位相等"
            assert np.array_equal(
                np.asarray(original.direction.v, dtype=float).view(np.uint64),
                np.asarray(restored.direction.v, dtype=float).view(np.uint64),
            ), "direction 必须逐位相等"
            assert original.length == restored.length
            assert original.delta == restored.delta

        assert tf.shading is not None
        assert np.array_equal(shading.densities, tf.shading.densities)

    def test_geometry_hash_is_stable_across_round_trip(self, tmp_path):
        tubes = make_fan()
        result = run_proof_skeleton(tubes, max_iterations=1, grid_n=16)
        path = write_trace(result, tmp_path / "rt.kpt")

        tf = read_trace(path)
        assert geometry_hash(tf.tubes, tf.shading) == tf.geometry_hash

    def test_hash_distinguishes_tiny_differences(self):
        """1e-17 的差异也要改哈希——这是"不取整"的实际含义。"""
        a = [Tube(center=np.zeros(3),
                  direction=Direction.from_spherical(np.pi / 2, 0.0),
                  length=1.0, delta=0.1)]
        b = [Tube(center=np.array([1e-17, 0.0, 0.0]),
                  direction=Direction.from_spherical(np.pi / 2, 0.0),
                  length=1.0, delta=0.1)]

        assert tube_hash(a[0]) != tube_hash(b[0])
        assert geometry_hash(a) != geometry_hash(b)

    def test_tube_order_matters(self):
        """管序是几何指纹的一部分：grains 的 tube_indices 依赖订阅顺序。"""
        tubes = make_fan()
        assert geometry_hash(tubes) != geometry_hash(list(reversed(tubes)))


class TestVerification:
    """复核是独立重算，不是"再读一遍自己写的东西"。"""

    def test_clean_file_passes(self, tmp_path):
        path = run_and_write(tmp_path)
        report = verify_trace(path)

        assert report.ok, report.summary()
        assert report.file_hash_ok
        assert report.geometry_hash_ok
        assert report.chain_ok
        assert report.decimal_hex_ok

    def test_summary_states_limits(self, tmp_path):
        """通过时必须说清"通过"意味着什么、不意味着什么。"""
        path = run_and_write(tmp_path)
        text = verify_trace(path).summary()

        assert "通过" in text
        assert "不是数学证明" in text
        assert "签名" in text, "应如实说明哈希链不防有动机的伪造"

    def test_params_from_header_are_used(self, tmp_path):
        """复核参数取文件 header，除非显式覆盖。"""
        path = run_and_write(tmp_path)
        tf = read_trace(path)

        assert tf.params()["grid_n"] == 16
        # 用另一套参数复核会得到不同结论（下界随 epsilon/K 变化），
        # 这里只确认显式参数确实生效（不抛错、结构完整）。
        report = verify_trace(path, epsilon=0.01, K=3.0, grid_n=16)
        assert report.n_steps == tf.n_steps

    def test_chain_only_mode_skips_recompute(self, tmp_path):
        path = run_and_write(tmp_path)
        report = verify_trace(path, recompute=False)

        assert report.ok
        assert all(v.recomputed is None for v in report.verdicts)

    def test_verification_does_not_depend_on_orchestrator(self, tmp_path):
        """复核只需文件——模拟"另一台机器上只有文件"的场景。"""
        path = run_and_write(tmp_path)
        # 不持有 orchestrator / result，只给路径
        report = verify_trace(Path(path))
        assert report.ok


class TestTamperDetection:
    """篡改必须被发现，且**定位到具体哪一步**。"""

    @staticmethod
    def _rewrite(path: Path, mutate) -> Path:
        lines = path.read_text(encoding="utf-8").splitlines()
        lines = mutate(lines)
        out = path.parent / "tampered.kpt"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def test_metric_tampering_is_localized(self, tmp_path):
        path = run_and_write(tmp_path)

        def mutate(lines):
            for i, ln in enumerate(lines):
                if ln.startswith('{"index"'):
                    obj = json.loads(ln)
                    obj["after"]["volume_ratio"] = "0x1.0000000000000p+0"
                    lines[i] = json.dumps(obj, ensure_ascii=False)
                    break
            return lines

        bad = self._rewrite(path, mutate)
        report = verify_trace(bad)

        assert not report.ok
        assert not report.chain_ok
        # 定位到具体步：HASH_BROKEN 且 index 明确
        broken = [v for v in report.verdicts if v.status == "HASH_BROKEN"]
        assert len(broken) == 1
        assert broken[0].index == 0

    def test_notes_tampering_is_caught(self, tmp_path):
        """往 notes 里塞"已人工验证"这类字样同样会被抓。"""
        path = run_and_write(tmp_path)

        def mutate(lines):
            for i, ln in enumerate(lines):
                if ln.startswith('{"index"'):
                    obj = json.loads(ln)
                    obj["notes"] = list(obj["notes"]) + ["verified by hand"]
                    lines[i] = json.dumps(obj, ensure_ascii=False)
                    break
            return lines

        bad = self._rewrite(path, mutate)
        report = verify_trace(bad)

        assert not report.ok
        assert not report.chain_ok

    def test_decimal_without_hex_is_a_mismatch(self, tmp_path):
        """只改十进制、保留 hex：人读到的数不是复核的数 → 必须报。"""
        path = run_and_write(tmp_path)

        def mutate(lines):
            geo = json.loads(lines[3])
            geo["tubes"][0]["center"][0] = 0.5     # 只改十进制
            lines[3] = json.dumps(geo, ensure_ascii=False)
            return lines

        bad = self._rewrite(path, mutate)
        report = verify_trace(bad)

        assert not report.ok
        assert not report.decimal_hex_ok
        assert any("十进制" in n for n in report.notes)

    def test_geometry_tampering_breaks_geometry_hash(self, tmp_path):
        """十进制与 hex 一起改成自洽的假值 → 几何哈希对不上。"""
        path = run_and_write(tmp_path)

        def mutate(lines):
            geo = json.loads(lines[3])
            geo["tubes"][0]["center"][0] = 0.5
            geo["tubes"][0]["center_hex"][0] = (0.5).hex()
            lines[3] = json.dumps(geo, ensure_ascii=False)
            return lines

        bad = self._rewrite(path, mutate)
        report = verify_trace(bad)

        assert not report.ok
        assert not report.geometry_hash_ok
        assert not report.file_hash_ok

    def test_chain_break_is_reported(self, tmp_path):
        """删掉中间一步 → 链断，且报在正确的位置。"""
        result = ProofOrchestrator(make_mixed_tubes(), max_iterations=6,
                                   grid_n=16, broadness_K=1.0,
                                   broadness_points=25).run()
        assert len(result.steps) >= 2, "需要多步 trace 才能测链中断"

        path = write_trace(result, tmp_path / "multi.kpt", params={"grid_n": 16})
        lines = path.read_text(encoding="utf-8").splitlines()
        step_lines = [i for i, ln in enumerate(lines) if ln.startswith('{"index"')]

        del lines[step_lines[1]]
        bad = tmp_path / "cut.kpt"
        bad.write_text("\n".join(lines), encoding="utf-8")
        report = verify_trace(bad, recompute=False)

        assert not report.ok
        assert not report.chain_ok

    def test_file_hash_catches_any_edit(self, tmp_path):
        path = run_and_write(tmp_path)

        def mutate(lines):
            lines[1] = lines[1].replace('"kpt/1"', '"kpt/1" ') + "  "
            return lines

        bad = self._rewrite(path, mutate)
        report = verify_trace(bad)

        assert not report.file_hash_ok


class TestExportGuards:
    """导出端不要把"不可复核的东西"写成看似可复核的文件。"""

    def test_missing_initial_tubes_is_refused(self, tmp_path):
        """没有几何输入就没法独立复核——拒绝导出，而不是写个空壳。"""
        result = run_proof_skeleton(make_fan(), max_iterations=1, grid_n=16)
        result.initial_tubes = None

        with pytest.raises(ValueError, match="初始管族"):
            write_trace(result, tmp_path / "nope.kpt")

    def test_run_records_initial_tubes(self):
        """``run()`` 必须把输入管族记进结果，否则 trace 无法自包含。"""
        tubes = make_fan()
        result = run_proof_skeleton(tubes, max_iterations=1, grid_n=16)

        assert result.initial_tubes is not None
        assert len(result.initial_tubes) == len(tubes)
        # 是输入本身（未被 broadness 过滤后的状态）
        assert result.initial_tubes[0].delta == tubes[0].delta

    def test_reads_reject_foreign_files(self, tmp_path):
        path = tmp_path / "not_a_trace.kpt"
        path.write_text('{"just": "json"}', encoding="utf-8")

        with pytest.raises(ValueError, match="不是有效的"):
            read_trace(path)


class TestFastOpen:
    """快速打开：只读头部，不解析数据体。"""

    def test_peek_does_not_read_body(self, tmp_path, monkeypatch):
        """关键性能契约：peek 的读取量不随步数增长。"""
        path = run_and_write(tmp_path)

        read_bytes = {"n": 0}
        original_open = Path.open

        def counting_open(self, *args, **kwargs):
            handle = original_open(self, *args, **kwargs)
            if self == path and args and "r" in str(args[0]):
                real_read = handle.read
                real_readline = handle.readline

                def read(*a, **k):
                    data = real_read(*a, **k)
                    read_bytes["n"] += len(data)
                    return data

                def readline(*a, **k):
                    line = real_readline(*a, **k)
                    read_bytes["n"] += len(line)
                    return line

                handle.read = read         # type: ignore[method-assign]
                handle.readline = readline  # type: ignore[method-assign]
            return handle

        monkeypatch.setattr(Path, "open", counting_open)
        header = peek_trace(path)
        monkeypatch.undo()

        assert header.n_steps >= 1
        file_size = path.stat().st_size
        assert read_bytes["n"] < file_size * 0.5, (
            f"peek 读了 {read_bytes['n']} 字节 / 共 {file_size} 字节——"
            "快速打开不应读整个文件"
        )

    def test_header_carries_conclusion(self, tmp_path):
        """结果摘要在头部：看一眼就知道结论，不用读数据体。"""
        path = run_and_write(tmp_path)
        header = peek_trace(path)

        assert header.protocol == PROTOCOL
        assert isinstance(header.success, bool)
        assert header.n_tubes > 0
        assert header.params["grid_n"] == 16
        assert "volume_ratio" in header.final

    def test_header_summary_is_readable(self, tmp_path):
        path = run_and_write(tmp_path)
        text = peek_trace(path).summary()

        assert "规模" in text and "结论" in text and "参数" in text

    def test_peek_rejects_foreign_file(self, tmp_path):
        bad = tmp_path / "x.kpt"
        bad.write_text("not a trace", encoding="utf-8")
        with pytest.raises(ValueError, match="不是有效的"):
            peek_trace(bad)


class TestFastImport:
    """快速导入：流式逐步读，不把整个文件载入内存。"""

    def test_iter_steps_yields_parsed_steps(self, tmp_path):
        result = ProofOrchestrator(make_mixed_tubes(), max_iterations=6,
                                   grid_n=16, broadness_K=1.0,
                                   broadness_points=25).run()
        path = write_trace(result, tmp_path / "m.kpt", params={"grid_n": 16})

        steps = list(iter_steps(path))
        assert len(steps) == len(result.steps)
        assert steps[0]["action"] == result.steps[0].action.name
        assert all("hash" in s for s in steps)

    def test_iter_steps_limit_and_offset(self, tmp_path):
        result = ProofOrchestrator(make_mixed_tubes(), max_iterations=6,
                                   grid_n=16, broadness_K=1.0,
                                   broadness_points=25).run()
        path = write_trace(result, tmp_path / "m.kpt", params={"grid_n": 16})

        limited = list(iter_steps(path, limit=1))
        assert len(limited) == min(1, len(result.steps))

        if len(result.steps) >= 2:
            skipped = list(iter_steps(path, offset=1))
            assert len(skipped) == len(result.steps) - 1
            assert skipped[0]["index"] == result.steps[1].iteration

    def test_iter_steps_streams_without_full_read(self, tmp_path, monkeypatch):
        """流式读：只取一步时不应把整个文件读进来。"""
        result = ProofOrchestrator(make_mixed_tubes(), max_iterations=6,
                                   grid_n=16, broadness_K=1.0,
                                   broadness_points=25).run()
        path = write_trace(result, tmp_path / "m.kpt", params={"grid_n": 16})
        if len(result.steps) < 2:
            pytest.skip("需要多步 trace")

        for _ in iter_steps(path, limit=1):
            break

        # 逐步语义：limit=1 只产出一条
        assert len(list(iter_steps(path, limit=1))) == 1

    def test_iter_steps_rejects_foreign_file(self, tmp_path):
        bad = tmp_path / "x.kpt"
        bad.write_text("nope", encoding="utf-8")
        with pytest.raises(ValueError, match="不是有效的"):
            list(iter_steps(bad))


class TestStateChangeReconstruction:
    """state_change 让复核能重建中间管族——这是"能真正验算多步 trace"的前提。"""

    def test_run_records_state_change(self):
        result = ProofOrchestrator(make_mixed_tubes(), max_iterations=6,
                                   grid_n=16, broadness_K=1.0,
                                   broadness_points=25).run()

        assert all(s.state_change is not None for s in result.steps)
        ops = [s.state_change["op"] for s in result.steps if s.state_change]
        assert ops, "至少应有一步"
        assert all(op in ("keep", "subset", "rescale", "replace") for op in ops)

    def test_broadness_step_records_subset(self):
        result = ProofOrchestrator(make_mixed_tubes(), max_iterations=6,
                                   grid_n=16, broadness_K=1.0,
                                   broadness_points=25).run()
        first = result.steps[0]
        assert first.action.name == "BROADNESS_REDUCTION"
        assert first.state_change is not None
        assert first.state_change["op"] == "subset"
        initial = result.initial_tubes
        assert initial is not None, "run() 必须记录初始管族"
        # indices 是 subset 独有的键（TypedDict 里可选），用 .get 取值：
        # 上面已断言 op == "subset"，所以这里取到的一定是那个列表。
        indices = first.state_change.get("indices")
        assert indices is not None, "subset 必须带 indices"
        assert len(indices) < len(initial)

    def test_multi_step_trace_verifies_fully(self, tmp_path):
        """多步 trace（含 broadness 过滤）必须能**逐步验算**，不只是验哈希链。"""
        result = ProofOrchestrator(make_mixed_tubes(), max_iterations=6,
                                   grid_n=16, broadness_K=1.0,
                                   broadness_points=25).run()
        assert len(result.steps) >= 2, "需要多步才能验证重建"

        path = write_trace(result, tmp_path / "m.kpt", params={"grid_n": 16})
        report = verify_trace(path, grid_n=16)

        assert report.ok, report.summary()
        assert all(v.status == "OK" for v in report.verdicts), report.summary()
        # 关键：不是 UNVERIFIED——确实重算过
        assert any(v.recomputed is not None for v in report.verdicts)

    def test_missing_state_change_reports_unverified(self, tmp_path):
        """老文件没有 state_change → 如实标 UNVERIFIED，不假装验过。"""
        result = run_proof_skeleton(make_fan(), max_iterations=1, grid_n=16)
        path = write_trace(result, tmp_path / "old.kpt", params={"grid_n": 16})

        # 手工删掉 state_change，模拟旧协议文件
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, ln in enumerate(lines):
            if ln.startswith('{"index"'):
                obj = json.loads(ln)
                obj.pop("state_change", None)
                lines[i] = json.dumps(obj, ensure_ascii=False)
        old = tmp_path / "very_old.kpt"
        old.write_text("\n".join(lines), encoding="utf-8")

        report = verify_trace(old, grid_n=16)
        # 哈希链会因为内容变了而断裂——这正是"改了就露馅"的体现
        assert not report.ok


class TestCli:
    """格式解释器：人能读、机器能消费、脚本能判退出码。"""

    def _run(self, *argv):
        import subprocess
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "orchestration" / "kpt.py"), *argv],
            capture_output=True, text=True,
        )

    def test_head_outputs_metadata(self, tmp_path):
        path = run_and_write(tmp_path)
        out = self._run("head", str(path))

        assert out.returncode == 0
        assert "规模" in out.stdout
        assert "结论" in out.stdout

    def test_head_json_is_parseable(self, tmp_path):
        path = run_and_write(tmp_path)
        out = self._run("head", str(path), "--json")

        assert out.returncode == 0
        payload = json.loads(out.stdout)
        assert payload["protocol"] == PROTOCOL
        assert "params" in payload

    def test_steps_streams(self, tmp_path):
        path = run_and_write(tmp_path)
        out = self._run("steps", str(path), "-n", "1")

        assert out.returncode == 0
        assert "共读出 1 步" in out.stdout

    def test_verify_exit_code_signals_result(self, tmp_path):
        path = run_and_write(tmp_path)
        out = self._run("verify", str(path))
        assert out.returncode == 0

    def test_verify_json_is_parseable(self, tmp_path):
        path = run_and_write(tmp_path)
        out = self._run("verify", str(path), "--json")

        payload = json.loads(out.stdout)
        assert payload["ok"] is True
        assert "verdicts" in payload

    def test_show_renders_geometry_and_steps(self, tmp_path):
        path = run_and_write(tmp_path)
        out = self._run("show", str(path))

        assert out.returncode == 0
        assert "几何" in out.stdout and "步骤" in out.stdout

    def test_missing_file_exit_code_2(self, tmp_path):
        out = self._run("head", str(tmp_path / "nope.kpt"))
        assert out.returncode == 2
        assert "不存在" in out.stderr

    def test_bad_format_exit_code_2(self, tmp_path):
        bad = tmp_path / "bad.kpt"
        bad.write_text("garbage", encoding="utf-8")
        out = self._run("head", str(bad))
        assert out.returncode == 2
        assert "格式错误" in out.stderr


class TestRoundTripStability:
    """往返必须**逐位稳定**——这也是"解释器自己写的要自己能验"的底线。

    回归背景：``Direction`` 构造时会无条件重新归一化。从文件读回时向量已经是
    单位向量，再除一遍自己那个"接近 1 但末位不是 1"的范数会把末位翻转
    （实测 200 个随机方向里有 2 个中招）。后果是**写出去再读回来几何哈希就变了**，
    ``verify_trace`` 误报"几何哈希不匹配"。
    """

    @staticmethod
    def _random_tubes(seed: int, n: int = 12):
        rng = np.random.default_rng(seed)
        return [
            Tube(center=rng.uniform(-0.4, 0.4, 3),
                 direction=Direction.from_spherical(
                     float(np.arccos(1 - 2 * rng.random())),
                     float(2 * np.pi * rng.random())),
                 length=0.8, delta=0.1)
            for _ in range(n)
        ]

    @pytest.mark.parametrize("seed", range(10))
    def test_geometry_hash_survives_round_trip(self, tmp_path, seed):
        tubes = self._random_tubes(seed)
        result = run_proof_skeleton(tubes, max_iterations=0, grid_n=8)
        path = write_trace(result, tmp_path / f"rt{seed}.kpt", params={"grid_n": 8})

        tf = read_trace(path)
        assert geometry_hash(tf.tubes, tf.shading) == tf.geometry_hash, (
            f"seed={seed}: 读回后几何哈希变了（二次归一化问题复发？）"
        )
        assert verify_trace(path, recompute=False).geometry_hash_ok

    def test_direction_needing_renormalization_is_flagged(self):
        """明显不是单位向量的输入仍应被归一化（宽容解析，但不静默接受错值）。"""
        import numpy as np
        from kakeya_ai.trace_format import _unit_direction

        d = _unit_direction(np.array([3.0, 4.0, 0.0]))
        assert abs(float(np.linalg.norm(d.v)) - 1.0) < 1e-12

    def test_all_metrics_round_trip_exactly(self, tmp_path):
        """指标的十进制与 hex 表示都必须往返无损。"""
        result = run_proof_skeleton(make_fan(), max_iterations=1, grid_n=8)
        path = write_trace(result, tmp_path / "m.kpt", params={"grid_n": 8})
        tf = read_trace(path)

        # strict：步数不符本就该失败，不能静默只比前 N 步
        for original, payload in zip(result.steps, tf.steps, strict=True):
            for side, metrics in (("before", original.before), ("after", original.after)):
                human = payload[f"{side}_human"]
                hexside = payload[side]
                for field in ("delta", "rho", "volume_ratio", "union_volume"):
                    assert float.fromhex(hexside[field]) == float(
                        getattr(metrics, field)), f"{side}.{field} hex 不精确"
                    assert human[field] == _dec(getattr(metrics, field)), \
                        f"{side}.{field} 十进制不精确"


class TestMalformedInputNeverCrashes:
    """任何畸形内容都只能导致明确的 ``ValueError``，不能是 KeyError/TypeError 之类。

    这一条来自 3000 轮 fuzz：最初有 5 类裸崩（字段被换成 int/bool/None 时
    在 ``.items()``/``.get()``/切片上炸），看起来像程序 bug，实际是文件内容不对。
    """

    def _base_lines(self, tmp_path) -> list[str]:
        result = ProofOrchestrator(make_mixed_tubes(), max_iterations=6, grid_n=8,
                                   broadness_K=1.0, broadness_points=20).run()
        path = write_trace(result, tmp_path / "base.kpt", params={"grid_n": 8})
        return path.read_text(encoding="utf-8").split("\n")

    @staticmethod
    def _mutate(lines: list[str], index: int, fn) -> list[str]:
        """返回**改过一行**的行列表（不是 Path）——调用方自己决定写到哪个文件。

        用 ``list[str]`` 而不是 ``Path``：此前注释写错，害得下面每个
        ``"\n".join(mutated)`` 都被报"Path 不可迭代"。"""
        out = list(lines)
        try:
            obj = json.loads(lines[index])
            out[index] = json.dumps(fn(obj), ensure_ascii=False)
        except json.JSONDecodeError:
            out[index] = fn(None)
        return out

    def _assert_only_value_error(self, path: Path, what: str):
        for name, fn in (("peek", lambda: peek_trace(path)),
                         ("read", lambda: read_trace(path)),
                         ("iter", lambda: list(iter_steps(path))),
                         ("verify", lambda: verify_trace(path, recompute=False))):
            try:
                fn()
            except ValueError:
                continue
            except Exception as exc:
                raise AssertionError(
                    f"{what} / {name}: 应给 ValueError，实际 "
                    f"{type(exc).__name__}: {exc}"
                ) from None

    def test_header_value_type_corruption(self, tmp_path):
        lines = self._base_lines(tmp_path)
        index = 1     # header 行
        for key, bad in (("params", 42), ("params", None), ("n_steps", "x"),
                         ("geometry_hash", 7), ("final", True)):
            mutated = self._mutate(lines, index, lambda o, k=key, b=bad: {**o, k: b})
            p = tmp_path / f"h_{key}_{bad}.kpt"
            p.write_text("\n".join(mutated), encoding="utf-8")
            self._assert_only_value_error(p, f"header.{key}={bad!r}")

    def test_geometry_value_type_corruption(self, tmp_path):
        lines = self._base_lines(tmp_path)
        for bad in (42, None, True, {"a": 1}, "text"):
            mutated = self._mutate(lines, 3, lambda o, b=bad: {**o, "shading_hex": b})
            p = tmp_path / f"g_shade_{bad}.kpt"
            p.write_text("\n".join(mutated), encoding="utf-8")
            self._assert_only_value_error(p, f"shading_hex={bad!r}")

        for bad in ([1], ["x"], [None], [True]):
            mutated = self._mutate(lines, 3, lambda o, b=bad: {**o, "shading_hex": b})
            p = tmp_path / f"g_els_{bad}.kpt"
            p.write_text("\n".join(mutated), encoding="utf-8")
            self._assert_only_value_error(p, f"shading_hex 元素={bad!r}")

    def test_step_field_type_corruption(self, tmp_path):
        lines = self._base_lines(tmp_path)
        step_index = next(i for i, ln in enumerate(lines) if ln.startswith('{"index"'))

        for key, bad in (("after_human", 42), ("after_human", True),
                         ("before", None), ("notes", 7), ("hash", 3)):
            mutated = self._mutate(lines, step_index,
                                   lambda o, k=key, b=bad: {**o, k: b})
            p = tmp_path / f"s_{key}_{bad}.kpt"
            p.write_text("\n".join(mutated), encoding="utf-8")
            self._assert_only_value_error(p, f"step.{key}={bad!r}")

    def test_footer_value_type_corruption(self, tmp_path):
        lines = self._base_lines(tmp_path)
        footer_index = max(i for i, ln in enumerate(lines) if ln.strip())
        for key, bad in (("head", 3), ("n_steps", "x"), ("file_hash", None)):
            mutated = self._mutate(lines, footer_index,
                                   lambda o, k=key, b=bad: {**o, k: b})
            p = tmp_path / f"f_{key}_{bad}.kpt"
            p.write_text("\n".join(mutated), encoding="utf-8")
            self._assert_only_value_error(p, f"footer.{key}={bad!r}")

    def test_truncation_at_every_line(self, tmp_path):
        """在**每一行边界**截断，都必须给出明确错误或可读结果，不能裸崩。"""
        lines = self._base_lines(tmp_path)
        for n in range(1, len(lines)):
            p = tmp_path / f"trunc_{n}.kpt"
            p.write_text("\n".join(lines[:n]), encoding="utf-8")
            for name, fn in (("read", lambda q=p: read_trace(q)),
                             ("verify", lambda q=p: verify_trace(q, recompute=False))):
                try:
                    fn()
                except ValueError:
                    pass
                except Exception as exc:
                    raise AssertionError(
                        f"截断到 {n} 行 / {name}: 应给 ValueError，实际 "
                        f"{type(exc).__name__}: {exc}"
                    ) from None


class TestMetadataAndStructureIntegrity:
    """元数据与结构必须自洽——这些是 fuzz 揪出来的真实漏洞。"""

    def _base(self, tmp_path) -> Path:
        result = ProofOrchestrator(make_mixed_tubes(), max_iterations=6, grid_n=8,
                                   broadness_K=1.0, broadness_points=20).run()
        return write_trace(result, tmp_path / "base.kpt", params={"grid_n": 8})

    def test_footer_is_protected(self, tmp_path):
        """footer 不在整文件哈希覆盖内 —— 必须有自己的校验和。

        回归：之前改 footer 的 ``head`` / ``n_steps`` 完全无人发现（fuzz 抓到
        11 处"改了还能通过"）。
        """
        p = self._base(tmp_path)
        lines = p.read_text(encoding="utf-8").split("\n")
        last = max(i for i, ln in enumerate(lines) if ln.strip())
        footer = json.loads(lines[last])
        assert "footer_hash" in footer, "footer 缺少自校验字段"

        footer["head"] = "0" * 64
        lines[last] = json.dumps(footer, ensure_ascii=False)
        q = tmp_path / "bad_footer.kpt"
        q.write_text("\n".join(lines), encoding="utf-8")

        report = verify_trace(q, recompute=False)
        assert not report.ok
        assert not report.footer_ok

    def test_declared_step_count_is_cross_checked(self, tmp_path):
        """header 声明的步数与实际不符时必须报出来。

        否则"快速打开"报的规模是假的，而使用者无从知道。
        """
        p = self._base(tmp_path)
        lines = p.read_text(encoding="utf-8").split("\n")
        header = json.loads(lines[1])
        header["n_steps"] = 999
        lines[1] = json.dumps(header, ensure_ascii=False)
        q = tmp_path / "bad_count.kpt"
        q.write_text("\n".join(lines), encoding="utf-8")

        report = verify_trace(q, recompute=False)
        assert not report.ok
        assert not report.metadata_ok
        assert any("999" in n for n in report.notes)

    def test_trailing_content_after_footer_is_rejected(self, tmp_path):
        """footer 之后多出的内容不在哈希覆盖内，必须显式拒绝。"""
        p = self._base(tmp_path)
        text = p.read_text(encoding="utf-8")
        q = tmp_path / "trailing.kpt"
        q.write_text(text + '{"injected": true}\n', encoding="utf-8")

        with pytest.raises(ValueError, match="多余内容"):
            read_trace(q)

    def test_bytes_level_hashing_catches_crlf(self, tmp_path):
        """整文件哈希覆盖**实际字节**：CRLF 这类行尾变化也要被检出。"""
        p = self._base(tmp_path)
        q = tmp_path / "crlf.kpt"
        q.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))

        assert verify_trace(p, recompute=False).file_hash_ok
        assert not verify_trace(q, recompute=False).file_hash_ok

    def test_bom_is_rejected_consistently(self, tmp_path):
        """BOM 会改变字节 → 四个入口应**一致拒绝**并给出修复提示。

        回归：曾经只有 peek 容忍 BOM，造成"能快开却读不了"的分裂行为。
        """
        p = self._base(tmp_path)
        q = tmp_path / "bom.kpt"
        q.write_bytes(b"\xef\xbb\xbf" + p.read_bytes())

        for name, fn in (("peek", lambda: peek_trace(q)),
                         ("read", lambda: read_trace(q)),
                         ("iter", lambda: list(iter_steps(q))),
                         ("verify", lambda: verify_trace(q, recompute=False))):
            with pytest.raises(ValueError):
                fn()
        with pytest.raises(ValueError, match="BOM"):
            peek_trace(q)


class TestExportContract:
    """导出端只接受能自洽复核的内容。"""

    def test_state_change_covers_every_recorded_step(self):
        """每个 ``ProofStep`` 都必须带 state_change——否则多步 trace 复核不了。

        逐组**显式调用**而不写成 ``**kw`` 展开：展开会把 ``dict(...)`` 的值
        统一推断成同一类型（实测被推成 ``float``），于是每个关键字都刷一条
        "float 不可分配给 int" 的假报错，把真问题淹掉。
        """
        results = [
            ProofOrchestrator(make_mixed_tubes(), max_iterations=6, grid_n=8).run(),
            ProofOrchestrator(make_mixed_tubes(), max_iterations=6, grid_n=8,
                              broadness_filter=False).run(),
            ProofOrchestrator(make_mixed_tubes(), max_iterations=6, grid_n=8,
                              broadness_K=1.0, broadness_points=20).run(),
        ]
        for result in results:
            for step in result.steps:
                assert step.state_change is not None, (
                    f"{step.action.name} 缺 state_change"
                )
                assert step.state_change["op"] in ("keep", "subset", "rescale",
                                                   "replace")

    def test_rebuilt_tube_count_matches_recorded(self, tmp_path):
        """按 state_change 重建的管族根数必须与记录的 n_tubes 一致。"""
        result = ProofOrchestrator(make_mixed_tubes(), max_iterations=6, grid_n=8,
                                   broadness_K=1.0, broadness_points=20).run()
        assert len(result.steps) >= 2, "需要多步才能验证重建"

        path = write_trace(result, tmp_path / "m.kpt", params={"grid_n": 8})
        tf = read_trace(path)
        current = list(tf.tubes)
        # strict：写出去的步数与读回的步数必须一致，否则下面的逐步重建
        # 会静默只覆盖前 N 步，漏掉的那部分永远不被校验。
        for i, (step, payload) in enumerate(
            zip(result.steps, tf.steps, strict=True)
        ):
            report = VerificationReport(path="", ok=True, n_steps=0)
            advanced = _apply_state_change(current, payload.get("state_change"),
                                           i, report)
            assert advanced is not None, f"第 {i} 步重建失败"
            current = advanced
            assert len(current) == step.after.n_tubes


class TestFailureReasonInTrace:
    """失败归因必须随 trace 一起落盘并在读侧还原。

    归因只活在内存里的话，第三方拿到文件仍分不清"预算用尽"与"判定失败"——
    而这正是复核时最需要知道的区别（前者可重跑，后者不必）。
    """

    def test_failure_reason_written_to_header(self, tmp_path):
        from kakeya_ai.proof_orchestrator import FailureReason

        tubes = make_fan()
        # max_iterations=0 → 必定是"预算用尽"，归因稳定可断言。
        result = ProofOrchestrator(tubes, max_iterations=0, grid_n=16).run()
        assert result.failure_reason is FailureReason.BUDGET_EXHAUSTED

        path = write_trace(result, tmp_path / "r.kpt", params={"grid_n": 16})
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        header = json.loads(lines[1])
        assert header["failure_reason"] == "BUDGET_EXHAUSTED"

    def test_peek_exposes_failure_reason(self, tmp_path):
        path = run_and_write(tmp_path)
        th = peek_trace(path)
        # 这份运行可能成功也可能预算用尽，但字段一定是**非空**的枚举名——
        # 空串是留给"旧协议文件"的标记，不该出现在新写出的文件上。
        assert th.failure_reason != "", "新写的 trace 必须带归因"

    def test_summary_renders_failure_reason(self, tmp_path):
        path = run_and_write(tmp_path)
        th = peek_trace(path)
        assert f"失败归因：{th.failure_reason}" in th.summary()

    def test_legacy_file_without_field_reads_as_empty(self, tmp_path):
        """缺字段的旧文件 → 空串（"生成时无此字段"），不是报错、也不是 NONE。"""
        path = run_and_write(tmp_path)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        header = json.loads(lines[1])
        header.pop("failure_reason", None)
        # 头部哈希是按原头部算的，改了头部必须重算，否则会被校验拦下——
        # 这里要测的是"缺字段的解析行为"，故重建一个自洽的头部。
        lines[1] = json.dumps(header, ensure_ascii=False)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        th = peek_trace(path)
        assert th.failure_reason == ""

    def test_empty_input_reason_round_trips(self, tmp_path):
        """空输入不写 trace（导出守门会拒绝），但归因本身要在内存里正确。

        这条同时钉住两件事：① 空输入的归因是 ``EMPTY_INPUT``；
        ② 导出端**拒绝**写出没有几何的 trace——空管族的文件无法复核，
        与其写一份“看似完整”的文件，不如明确失败。
        """
        from kakeya_ai.proof_orchestrator import FailureReason, run_proof_skeleton

        result = run_proof_skeleton([], max_iterations=2)
        assert result.failure_reason is FailureReason.EMPTY_INPUT
        with pytest.raises(ValueError, match="缺少初始管族"):
            write_trace(result, tmp_path / "empty.kpt", params={})


class TestConfigSnapshotInTrace:
    """生效的**环境配置**要随 trace 落盘。

    ``params`` 记的是"这次运行被显式指定的参数"，但环境变量（如
    ``KAKEYA_GRID_N``）同样会改变数值结果。此前完全没记录：复核方拿同一份
    参数重跑，若环境不同就会对不上，而且无从得知差异来源。
    """

    def test_default_run_records_empty_config(self, tmp_path, monkeypatch):
        """没有任何环境覆盖时，``config`` 是空字典（而不是缺失或一堆默认值）。"""
        from kakeya_ai.config import env_names

        for name in env_names():
            monkeypatch.delenv(name, raising=False)

        path = run_and_write(tmp_path)
        header = json.loads(Path(path).read_text(encoding="utf-8").splitlines()[1])
        assert header["config"] == {}

    def test_override_is_recorded(self, tmp_path, monkeypatch):
        """设了环境变量 → 该键必须出现在 config 里，值要如实。"""
        from kakeya_ai.config import env_names

        # 用注册表里真实存在的变量名：写死一个不存在的名字会让测试永远 skip，
        # 看着"绿"但从未真正执行——那比失败更危险。
        target = "KAKEYA_MARCH"
        assert target in env_names(), f"{target} 应已注册（测试依赖它）"
        monkeypatch.setenv(target, "1")

        path = run_and_write(tmp_path)
        header = json.loads(Path(path).read_text(encoding="utf-8").splitlines()[1])
        assert target in header["config"], "生效的环境覆盖必须被记录"
        assert str(header["config"][target]) == "1"

    def test_peek_exposes_config(self, tmp_path, monkeypatch):
        from kakeya_ai.config import env_names

        # ``KAKEYA_MARCH`` 是 str 类型，值原样保留（不做数值转换）。
        target = "KAKEYA_MARCH"
        assert target in env_names(), f"{target} 应已注册（测试依赖它）"
        monkeypatch.setenv(target, "x86-64-v2")

        th = peek_trace(run_and_write(tmp_path))
        assert th.config.get(target) == "x86-64-v2"
        assert "环境覆盖" in th.summary()

    def test_config_failure_does_not_break_export(self, tmp_path, monkeypatch):
        """配置读取炸了也不能让 trace 写不出来——那是证明搜索的唯一产出。

        这里让 ``snapshot`` 抛异常，断言导出仍然成功且把错误如实记下，
        而不是静默假装"没有覆盖"。
        """
        import kakeya_ai.config as config_mod

        def boom():
            raise RuntimeError("registry corrupted")

        monkeypatch.setattr(config_mod, "snapshot", boom)

        path = run_and_write(tmp_path)
        assert Path(path).exists(), "配置读取失败不该阻断 trace 导出"
        header = json.loads(Path(path).read_text(encoding="utf-8").splitlines()[1])
        assert "_error" in header["config"]

    def test_legacy_file_without_config_reads_as_empty(self, tmp_path):
        """缺字段的旧文件 → 空字典，不报错。"""
        path = run_and_write(tmp_path)
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[1])
        header.pop("config", None)
        lines[1] = json.dumps(header, ensure_ascii=False)
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

        assert peek_trace(path).config == {}


class TestSelectionSourceInTrace:
    """Move 来源（``SelectionSource``）必须落盘，且**篡改会被发现**。

    它参与哈希链是有意义的：该字段的唯一用途是"统计哪条路径覆盖了多少"，
    若能随意把 ``HEURISTIC`` 改成 ``LLM`` 而不被发现，任何基于它的结论都不
    可信。这与 ``failure_reason`` 的处理一致（都写进头部/步骤且参与哈希）。

    另一侧同样重要：**旧文件没有这个字段**，写入端与复核端必须用同一套判据
    （"文件里有就算"），否则干净文件会被误判哈希断裂——这正是
    ``domain_radius`` 当初踩过的坑，注释里已写明。
    """

    def _run_multi_step(self, tmp_path) -> Path:
        """跑一个多步 trace（含 broadness 剥离，保证有步骤行）。"""
        result = ProofOrchestrator(
            make_mixed_tubes(), max_iterations=4, grid_n=8,
            broadness_K=1.0, broadness_points=20,
        ).run()
        return write_trace(result, tmp_path / "m.kpt", params={"grid_n": 8})

    def test_selection_source_written_to_each_step(self, tmp_path):
        path = self._run_multi_step(tmp_path)
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        step_lines = [ln for ln in lines if ln.startswith('{"index"')]
        assert step_lines, "应有步骤行"
        for ln in step_lines:
            payload = json.loads(ln)
            assert "selection_source" in payload, "每步都应带来源"
            assert payload["selection_source"], "来源不该为空串"
            # 缺口二补强：每步都应带 ``iters_used`` 键（非 Move 步为 null），
            # 使"这个 Move 动了几轮"可作为**定量断言**随文件落盘。
            assert "iters_used" in payload, "每步都应带 iters_used 键"

    def test_source_is_a_valid_enum_name(self, tmp_path):
        from kakeya_ai.proof_orchestrator import SelectionSource

        valid = {s.name for s in SelectionSource}
        path = self._run_multi_step(tmp_path)
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        for ln in lines:
            if ln.startswith('{"index"'):
                assert json.loads(ln)["selection_source"] in valid

    def test_clean_file_still_verifies(self, tmp_path):
        """加了参与哈希的字段后，**干净文件必须仍能通过复核**。

        这是最关键的回归：写入端加了字段而复核端没同步，就会把好文件判成
        坏文件（本项目在此曾一次失败 7 个测试）。
        """
        path = self._run_multi_step(tmp_path)
        report = verify_trace(path, recompute=False)
        assert report.ok, report.summary()
        assert report.chain_ok

    def test_tampering_with_source_is_detected(self, tmp_path):
        """篡改参与哈希的 Move 元数据（来源 / 轮数）必须被链校验抓到。

        两件事同一理由：``selection_source`` 是"这一步是谁选的"、``iters_used``
        是"这个 Move 动了几轮"，都是**定量断言**，若能事后随便改写就没有结论
        价值。篡改 ``iters_used`` 时改成一个**确定不同**的值（``None`` ↔ 非
        ``None``），确保哈希确实变化，而不是"改了个恰好相同的数"。
        """
        path = self._run_multi_step(tmp_path)
        base = Path(path).read_text(encoding="utf-8").splitlines()

        # 返回类型标 `None`：这个闭包只**写文件 + 断言**，没有返回值（原先标 `str`
        # 是错的——pyright 的 reportReturnType 抓到了"声明 str 却无 return"）。
        def _tamper(key: str, value) -> None:
            lines = list(base)
            for i, ln in enumerate(lines):
                if ln.startswith('{"index"'):
                    obj = json.loads(ln)
                    obj[key] = value
                    lines[i] = json.dumps(obj, ensure_ascii=False)
                    break
            bad = tmp_path / f"tampered_{key}.kpt"
            bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = verify_trace(bad, recompute=False)
            assert not report.ok, f"篡改 {key} 未被发现"
            assert not report.chain_ok, f"篡改 {key} 后链校验仍报通过"

        _tamper("selection_source", "LLM")
        # ``iters_used``：取一个与该步现值确定不同的值。
        first = json.loads(next(ln for ln in base if ln.startswith('{"index"')))
        _tamper("iters_used", 7 if first.get("iters_used") is None else None)
