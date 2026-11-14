"""编排器**可复现性**与 ``.kpt`` 种子传递的测试。

背景（2026-09-17）：编排器调用 ``nsp_factorization`` 时**没有传递种子**，
而该函数在 ``seed=None`` 时用 ``np.random.default_rng(None)`` 取新熵。后果不是
"测试偶尔失败"这么轻：项目以"同一输入 → 可独立复核的 trace"为卖点，而主循环
唯一的随机来源未固定，意味着**复核方拿到完全相同的参数也复现不出同一份 trace**。

这些用例钉住四件事：

1. 同一 seed 两次运行逐位一致（可复现）；
2. ``seed=None`` 是**显式**的探索模式，且如实反映在结果与 trace 里；
3. 种子真的被传进了 NSP 因子化（不是"记了却没用"）；
4. 同一运行内的多次 NSP 调用使用**不同**的种子（否则重复因子化没有意义）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kakeya_ai.primitives import Direction, Tube
from kakeya_ai.proof_orchestrator import ProofOrchestrator, run_proof_skeleton
from kakeya_ai.trace_format import peek_trace, read_trace, verify_trace, write_trace


def make_tubes(delta: float = 0.1, n_theta: int = 4, n_phi: int = 4):
    """确定性的 Kakeya fan（不含随机数，保证用例可复现）。"""
    return [
        Tube(center=np.zeros(3),
             direction=Direction.from_spherical(theta, phi),
             length=1.0, delta=delta)
        for theta in np.linspace(0.15, np.pi - 0.15, n_theta)
        for phi in np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    ]


def _fingerprint(result):
    """动作序列 + 每步体积比，用于判定"两次运行是否同一路径"。"""
    return (
        tuple(s.action.name for s in result.steps),
        tuple(s.before.volume_ratio for s in result.steps),
        tuple(s.after.volume_ratio for s in result.steps),
    )


class TestReproducibility:
    def test_same_seed_gives_identical_path(self):
        for seed in (0, 1, 7, 17):
            a = ProofOrchestrator(make_tubes(), max_iterations=4, grid_n=16,
                                  seed=seed).run()
            b = ProofOrchestrator(make_tubes(), max_iterations=4, grid_n=16,
                                  seed=seed).run()
            assert _fingerprint(a) == _fingerprint(b), (
                f"seed={seed} 两次运行结果不同"
            )

    def test_default_seed_is_deterministic(self):
        """**不传 seed 也必须确定**——默认值取 0 而非 None。"""
        a = ProofOrchestrator(make_tubes(), max_iterations=4, grid_n=16).run()
        b = ProofOrchestrator(make_tubes(), max_iterations=4, grid_n=16).run()
        assert _fingerprint(a) == _fingerprint(b)
        assert a.seed == 0

    def test_different_seeds_may_differ(self):
        """种子是**参数**：不同值可以给出不同路径（不是 bug）。"""
        results = {}
        for seed in range(8):
            r = ProofOrchestrator(make_tubes(), max_iterations=4, grid_n=16,
                                  seed=seed).run()
            results[seed] = _fingerprint(r)
        assert len(set(results.values())) > 1, (
            "8 个不同 seed 给出完全相同的结果，种子可能没传到 NSP"
        )

    def test_run_proof_skeleton_forwards_seed(self):
        a = run_proof_skeleton(make_tubes(), max_iterations=3, grid_n=16, seed=5)
        b = run_proof_skeleton(make_tubes(), max_iterations=3, grid_n=16, seed=5)
        assert a.seed == 5
        assert _fingerprint(a) == _fingerprint(b)

    def test_empty_tubes_records_seed(self):
        result = ProofOrchestrator([], seed=11).run()
        assert result.seed == 11
        assert result.steps == []


class TestSeedActuallyReachesNSP:
    """种子必须真的传进 NSP——"记了却没用"与"没记"一样糟。"""

    def test_nsp_receives_derived_seed(self, monkeypatch):
        """拦下 ``nsp_factorization`` 的调用参数，确认 seed 非 None。"""
        from kakeya_ai import proof_orchestrator as po

        captured = []
        original = po.nsp_factorization

        def spy(tubes, delta=None, **kwargs):
            captured.append(kwargs.get("seed"))
            return original(tubes, delta=delta, **kwargs)

        monkeypatch.setattr(po, "nsp_factorization", spy)
        ProofOrchestrator(make_tubes(), max_iterations=3, grid_n=16, seed=42).run()

        assert captured, "本用例的前提是构型会触发 NSP（否则没测到）"
        assert all(s is not None for s in captured), (
            f"有 NSP 调用没收到种子：{captured}"
        )

    def test_nsp_successive_calls_use_distinct_seeds(self, monkeypatch):
        """同一次运行内多次 NSP 调用应使用不同种子。

        否则"再次因子化"只是把同一批随机副本重撒一遍，策略上毫无变化。
        """
        from kakeya_ai import proof_orchestrator as po

        captured = []
        original = po.nsp_factorization

        def spy(tubes, delta=None, **kwargs):
            captured.append(kwargs.get("seed"))
            return original(tubes, delta=delta, **kwargs)

        monkeypatch.setattr(po, "nsp_factorization", spy)
        # 大 max_iterations 提高触发多次 NSP 的机会。
        ProofOrchestrator(make_tubes(), max_iterations=6, grid_n=16, seed=0).run()

        if len(captured) > 1:
            assert len(set(captured)) == len(captured), (
                f"同一次运行内 NSP 复用了种子：{captured}"
            )

    def test_none_seed_passes_none_to_nsp(self, monkeypatch):
        """``seed=None``（探索模式）必须把 None 传下去，而不是悄悄补一个数。"""
        from kakeya_ai import proof_orchestrator as po

        captured = []
        original = po.nsp_factorization

        def spy(tubes, delta=None, **kwargs):
            captured.append(kwargs.get("seed"))
            return original(tubes, delta=delta, **kwargs)

        monkeypatch.setattr(po, "nsp_factorization", spy)
        ProofOrchestrator(make_tubes(), max_iterations=3, grid_n=16, seed=None).run()

        assert captured, "本用例的前提是构型会触发 NSP"
        assert all(s is None for s in captured)


class TestSeedInTraceParams:
    """种子必须写进 ``.kpt`` 的 ``params``，否则复核方无从重放。"""

    def test_run_seed_written_as_integer(self, tmp_path):
        result = run_proof_skeleton(make_tubes(), max_iterations=3, grid_n=16, seed=9)
        path = write_trace(result, tmp_path / "seeded.kpt")
        header = json.loads(path.read_text(encoding="utf-8").split("\n")[1])
        assert header["params"]["run_seed"] == 9

    def test_non_deterministic_run_written_as_null(self, tmp_path):
        """探索模式写 ``null``——如实声明"不可重放"，不写一个假的种子。"""
        result = run_proof_skeleton(make_tubes(), max_iterations=3, grid_n=16,
                                    seed=None)
        path = write_trace(result, tmp_path / "explore.kpt")
        header = json.loads(path.read_text(encoding="utf-8").split("\n")[1])
        assert "run_seed" in header["params"]
        assert header["params"]["run_seed"] is None

    def test_key_name_avoids_clash_with_input_seed(self, tmp_path):
        """键名必须是 ``run_seed``，不能占用 ``seed``。

        既有 trace 里的 ``seed`` 指**输入管族**的构造种子（对编排器的 NSP
        副本毫无作用）。混用会让复核方误用种子。
        """
        result = run_proof_skeleton(make_tubes(), max_iterations=3, grid_n=16, seed=1)
        # 调用方按旧习惯传了一个 ——不该被 run_seed 覆盖或混淆。
        path = write_trace(result, tmp_path / "both.kpt", params={"seed": 7})
        header = json.loads(path.read_text(encoding="utf-8").split("\n")[1])
        assert header["params"]["seed"] == 7, "调用方给的输入种子应保留"
        assert header["params"]["run_seed"] == 1, "编排器种子另用一个键"

    def test_explicit_params_seed_not_overwritten(self, tmp_path):
        """调用方显式给 ``run_seed`` 时不被覆盖（setdefault 语义）。"""
        result = run_proof_skeleton(make_tubes(), max_iterations=3, grid_n=16, seed=1)
        path = write_trace(result, tmp_path / "explicit.kpt", params={"run_seed": 99})
        header = json.loads(path.read_text(encoding="utf-8").split("\n")[1])
        assert header["params"]["run_seed"] == 99


class TestReproducibilityReported:
    """复核报告必须让调用方看到"能否重放"，而不是让他以为一定能。"""

    def test_seeded_trace_reports_seeded(self, tmp_path):
        result = run_proof_skeleton(make_tubes(), max_iterations=3, grid_n=16, seed=0)
        path = write_trace(result, tmp_path / "s.kpt")
        report = verify_trace(path, recompute=False)
        assert report.reproducibility == "seeded"

    def test_non_deterministic_trace_reports_it(self, tmp_path):
        result = run_proof_skeleton(make_tubes(), max_iterations=3, grid_n=16,
                                    seed=None)
        path = write_trace(result, tmp_path / "n.kpt")
        report = verify_trace(path, recompute=False)
        assert report.reproducibility == "non-deterministic"
        assert any("不可重放" in n for n in report.notes)

    def test_legacy_trace_reports_unknown(self, tmp_path):
        """缺 ``run_seed`` 的旧文件：报 unknown，不假装知道。"""
        result = run_proof_skeleton(make_tubes(), max_iterations=3, grid_n=16, seed=0)
        path = write_trace(result, tmp_path / "l.kpt")
        lines = path.read_text(encoding="utf-8").split("\n")
        header = json.loads(lines[1])
        header["params"].pop("run_seed", None)
        lines[1] = json.dumps(header, ensure_ascii=False)
        legacy = tmp_path / "legacy.kpt"
        legacy.write_text("\n".join(lines), encoding="utf-8")

        report = verify_trace(legacy, recompute=False)
        assert report.reproducibility == "unknown"
        assert any("run_seed" in n for n in report.notes)

    def test_reproducibility_does_not_affect_ok(self, tmp_path):
        """非确定性是**能力限制**，不是文件错误：不该让 ok 变 False。"""
        result = run_proof_skeleton(make_tubes(), max_iterations=3, grid_n=16,
                                    seed=None)
        path = write_trace(result, tmp_path / "nd.kpt")
        report = verify_trace(path, recompute=False)
        assert report.ok, "非确定性运行的文件本身仍应自洽"
        assert report.reproducibility == "non-deterministic"

    def test_peek_reports_run_seed(self, tmp_path):
        result = run_proof_skeleton(make_tubes(), max_iterations=3, grid_n=16, seed=4)
        path = write_trace(result, tmp_path / "p.kpt")
        header = peek_trace(path)
        assert header.params.get("run_seed") == 4


class TestLegacyTraceStillVerifies:
    """加种子不得弄坏已入库的历史 trace。"""

    def test_repo_trace_still_passes(self):
        traces = sorted(Path("docs/traces").glob("*.kpt"))
        if not traces:
            pytest.skip("docs/traces/ 下没有入库 trace")
        for path in traces:
            report = verify_trace(path)
            assert report.ok, f"{path.name} 复核失败：\n{report.summary()}"


# ---------------------------------------------------------------------------
# 入库样本必须能被**当前代码重放**（不只是自洽）
# ---------------------------------------------------------------------------

#: 入库样本里**由当前代码生成**的那一份（2026-09-18 起）。
#:
#: 为什么要有这个区分：`docs/traces/` 下另有一份 2026-09-15 的样本，它记录的是
#: `69452dd` 修复前的行为（当时 `max_wolff_error` 默认 10.0，15 组构型里 14 组
#: 在第一步被误判 sticky 而短路）。那份**只保证自洽**，不保证可重放——它是历史
#: 证据，留着是为了证明"旧协议文件仍能复核通过"。若拿它做重放断言，测试会
#: 长期报红，而那不是缺陷（详见 `docs/traces/README.md`）。
CURRENT_TRACE = Path("docs/traces/20260918_105624_demo.kpt")


def _require_trace() -> None:
    """入库 trace 不随公开仓库分发；缺失时相关用例跳过而非报红。"""
    if not CURRENT_TRACE.exists():
        pytest.skip("入库 trace 不随公开仓库分发；本地生成样本后相关用例自动恢复")


def _load_demo_module():
    """加载 ``scripts/orchestration/demo_proof_orchestrator.py``（不在包里，只能按路径载）。"""
    import importlib.util

    path = (Path(__file__).resolve().parents[2]
            / "scripts" / "orchestration" / "demo_proof_orchestrator.py")
    spec = importlib.util.spec_from_file_location("demo_proof_orchestrator", path)
    # 同上：``spec`` 与 ``spec.loader`` 都是 ``Optional``，显式断言避免它们
    # 为 None 时在下一行抛难懂的 TypeError。
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRepoTraceIsReplayable:
    """入库样本可被当前代码**按参数重放**出同一份轨迹。

    这一条补的是此前的盲区：`verify` 只证明"文件自洽、重算一致"，**不证明**
    "别人拿同样的参数能跑出同一个结果"。2026-09-18 发现入库轨迹与当前代码行为
    不符（sticky 判定修复后结论从"1 步 sticky"变成"3 步耗尽"），而当时的测试
    只断言了前者，落差无人拦截。

    开销：重放约 33 秒、峰值内存约 43 MB（实测）。按仓库资源约定标 ``slow``
    ——CI 上跑全量时包含它，本机日常回归默认跳过。
    """

    def test_trace_exists(self):
        if not CURRENT_TRACE.exists():
            pytest.skip("入库 trace 不随公开仓库分发；本地生成样本后本条自动恢复")

    @pytest.mark.slow
    def test_replay_reproduces_actions_and_metrics(self):
        """按轨迹里记录的参数重跑，动作序列与关键指标必须一致。"""
        _require_trace()
        header = peek_trace(CURRENT_TRACE)
        params = header.params

        demo = _load_demo_module()
        tubes = demo.make_challenging_tubes(n=params["n"], seed=params["seed"])
        result = run_proof_skeleton(
            tubes,
            epsilon=params["epsilon"],
            K=params["K"],
            lambda_min=params["lambda_min"],
            max_iterations=params["iters"],
            grid_n=params["grid_n"],
            use_nano=params["use_nano"],
            seed=params["run_seed"],
        )

        recorded = read_trace(CURRENT_TRACE)
        recorded_actions = [s["action"] for s in recorded.steps]
        replay_actions = [s.action.name for s in result.steps]

        assert replay_actions == recorded_actions, (
            "重放的动作序列与入库样本不同——入库样本已不能代表当前代码的行为。\n"
            f"  入库：{recorded_actions}\n"
            f"  重放：{replay_actions}\n"
            "要么修代码，要么重新生成样本（并说明为何行为变了）。"
        )

        # 体积比是 Theorem 1.2 的判据，必须落到同一个值（trace 存的是
        # float.hex，逐位可比；这里放宽到相对误差 1e-9 以兼容后端差异）。
        recorded_ratio = recorded.steps[-1]["after_human"]["volume_ratio"]
        replay_ratio = result.final_metrics.volume_ratio
        assert replay_ratio == pytest.approx(recorded_ratio, rel=1e-9), (
            f"最终 volume_ratio 不同：入库 {recorded_ratio} vs 重放 {replay_ratio}"
        )

    def test_declared_params_are_complete(self):
        """入库样本必须自带重放所需的全部参数（缺一项就无从重放）。"""
        _require_trace()
        header = peek_trace(CURRENT_TRACE)
        for key in ("epsilon", "K", "lambda_min", "grid_n", "use_nano",
                    "n", "iters", "seed", "run_seed"):
            assert key in header.params, f"入库样本的 params 缺 {key!r}"
        # run_seed 是编排器的种子（可复现的前提）；seed 是输入管族的构造种子。
        # 两者都必须在，缺任何一个都重放不出来。
        assert header.params["run_seed"] is not None, (
            "入库样本的 run_seed 为 null，说明它是探索模式产物，无法重放"
        )
