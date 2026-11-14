"""守门：`ProofOrchestrator.run()` 的**幂等性**——同一实例多次运行 == 每次新建实例。

**背景**（2026-09-30 实测发现）：`run()` 此前会把"按次运行"的状态留到下一次，
其中 ``_nsp_calls``（NSP 种子派生计数器，``seed + k``）影响最大：

```
同一实例连跑 3 次:  4 步 / 4 步 / 2 步     ← 每次不同
三个独立实例:        4 步 / 4 步 / 4 步     ← 一致
```

原因是第二/三次 ``run()`` 的第一处 NSP 用了 ``seed + 1`` / ``seed + 2``，
**生成的是另一组管族**。后果：任何"重复跑同一构型做消融/对比"的实验都会
**静默失真**——而且看起来一切正常（都是 success=True）。

同一类状态还有 ``_move_attempts``（Move 失效记忆）与 ``_scorer_gate_*``
（打分器门控统计）。本文件把"`run()` 是新的一次运行"这条约定钉住。

**为什么单测这条**：它不属于任何单个功能，而是**调用约定**——没有被谁
"专门负责"，于是最容易在后续改动里悄悄退化。
"""

from __future__ import annotations

import numpy as np
import pytest

from kakeya_ai.primitives import Direction, Tube
from kakeya_ai.proof_orchestrator import ProofOrchestrator


def _random_tubes(seed: int, n: int, delta: float = 0.1) -> list[Tube]:
    """随机构型（采样顺序见 ``test_scorer_move_guard._random_tubes``）。"""
    rng = np.random.default_rng(seed)
    tubes = []
    for _ in range(n):
        center = rng.uniform(-0.3, 0.3, 3)
        d = rng.normal(size=3)
        d = d / np.linalg.norm(d)
        tubes.append(
            Tube(
                center=center,
                direction=Direction.from_array(d.tolist()),
                length=1.0,
                delta=delta,
            )
        )
    return tubes


def _orch(tubes: list[Tube], **kw) -> ProofOrchestrator:
    return ProofOrchestrator(
        tubes,
        epsilon=0.05,
        K=2.0,
        max_iterations=kw.pop("max_iterations", 12),
        grid_n=16,
        **kw,
    )


def _signature(result) -> tuple:
    """一次运行的**可观测签名**：动作序列 + 每步前后指标 + 成败。"""
    return (
        result.success,
        tuple(s.action.name for s in result.steps),
        tuple(
            (round(s.before.rho, 12), round(s.after.rho, 12),
             round(s.before.volume_ratio, 12), round(s.after.volume_ratio, 12))
            for s in result.steps
        ),
    )


# ---------------------------------------------------------------------------
# 1. 核心：同实例多次 run == 新实例
# ---------------------------------------------------------------------------


def test_repeated_run_matches_fresh_instance():
    """**核心判据**：同一实例连跑 4 次，每次都必须等于"新建实例跑一次"。

    这条直接覆盖 ``_nsp_calls`` 累积那个缺陷（修复前第 2 次即分叉）。
    """
    tubes = _random_tubes(0, 20)
    fresh = _signature(_orch(tubes).run())

    reused = _orch(tubes)
    for i in range(4):
        sig = _signature(reused.run())
        assert sig == fresh, (
            f"同一实例第 {i + 1} 次 run() 的结果与新实例不同——"
            "run() 不是幂等的（有按次运行的状态没归零）"
        )


def test_repeated_run_across_several_configurations():
    """多个构型都幂等（单点可能是特例）。"""
    for seed in range(4):
        for n in (12, 20, 32):
            tubes = _random_tubes(seed, n)
            fresh = _signature(_orch(tubes).run())
            reused = _orch(tubes)
            reused.run()
            assert _signature(reused.run()) == fresh, (
                f"seed={seed} n={n}: 第二次 run() 与新实例不同"
            )


def test_nsp_calls_counter_resets_each_run():
    """``_nsp_calls`` 每次 run 后必须是"本次运行内的调用次数"，不是累计值。

    直接盯那个变量——上一轮就是它导致分叉（``seed + k`` 里的 k 跨运行累加）。
    """
    tubes = _random_tubes(0, 20)
    orch = _orch(tubes)
    first_counts: list[int] = []
    for _ in range(3):
        orch.run()
        first_counts.append(orch._nsp_calls)
    assert len(set(first_counts)) == 1, (
        f"`_nsp_calls` 三次 run 后分别是 {first_counts}——它在跨运行累积，"
        "会让 NSP 种子变成 seed+k（k 含上次运行的调用数）"
    )


def test_move_attempts_do_not_leak_into_next_run():
    """``_move_attempts``（Move 失效记忆）是**本次运行内**的记忆。

    ⚠️ 不能断言"run() 结束后为空"——它在运行**过程中**累积（记录本次试过
    哪些 Move），结束后非空是**正常**的。要测的是它**不会漏进下一次**：
    做法是跑一次拿到记忆，再跑一次，看第二次的签名是否仍等于新实例。
    """
    tubes = _random_tubes(0, 20)
    fresh = _signature(_orch(tubes).run())

    reused = _orch(tubes)
    reused.run()
    # 上一次的记忆此刻必然非空（运行中记下的），这是正常的。
    assert reused._move_attempts, (
        "跑完一次后没有任何 Move 记忆——前提不对（这批构型应做过 Move），"
        "下面的「不泄漏」断言就失去意义"
    )
    leaked = dict(reused._move_attempts)

    assert _signature(reused.run()) == fresh, (
        f"第二次 run() 与新实例不同（上次留下的记忆 {list(leaked)} 泄漏了）"
    )


def test_scorer_gate_stats_reset_each_run():
    """打分器门控统计按次归零（与 ``test_scorer_gate_stats`` 同源，此处再钉）。"""
    tubes = _random_tubes(0, 20)
    orch = _orch(tubes)
    orch.run()
    orch._scorer_gate_accepts["SURROGATE"] = 999
    orch.run()
    assert orch.scorer_gate_stats()["SURROGATE"]["accepts"] < 999, (
        "门控统计没在 run() 开头归零——两次运行的数据混在一起了"
    )


# ---------------------------------------------------------------------------
# 2. 前提守卫：这些测试必须真的跑到了"会分叉"的路径
# ---------------------------------------------------------------------------


def test_configuration_actually_triggers_nsp():
    """**前提守卫**：上面那些测试用的构型必须**真的触发过 NSP**。

    若某次重构让这批构型不再走 NSP（如阈值调整），``_nsp_calls`` 恒为 0，
    于是"种子累积"那个缺陷测不出来、测试退化成假通过。
    """
    tubes = _random_tubes(0, 20)
    orch = _orch(tubes)
    result = orch.run()
    actions = [s.action.name for s in result.steps]
    assert "NSP_FACTORIZATION" in actions, (
        f"这批构型没走 NSP（动作序列 {actions}）——幂等性测试的前提没了，"
        "需换构型或确认 NSP 的触发条件"
    )
    assert orch._nsp_calls >= 1, "走了 NSP 但计数器为 0——计数没接上"


def test_runs_are_nontrivial():
    """**前提守卫**：结果不能是"一步就终止"，否则幂等性无从检验。

    若构型全是 sticky（1 步终止），上面的测试等于在比较两个空序列。
    """
    tubes = _random_tubes(0, 20)
    result = _orch(tubes).run()
    assert len(result.steps) >= 3, (
        f"这批构型只有 {len(result.steps)} 步就终止——太短，"
        "检验不出多次运行之间的差异"
    )


def test_non_default_seed_is_also_idempotent():
    """显式 seed 时也要幂等（``_next_nsp_seed`` 依赖 ``self.seed``）。"""
    tubes = _random_tubes(1, 20)
    fresh = _signature(_orch(tubes, seed=7).run())
    reused = _orch(tubes, seed=7)
    reused.run()
    assert _signature(reused.run()) == fresh, "显式 seed=7 时第二次 run() 分叉"


def test_seed_none_is_also_idempotent():
    """``seed=None``（不固定随机）时，两次运行**可以不同**——这不是缺陷。

    但**结构**必须一致（step 数可能因随机而变），所以这里只断言不抛异常、
    且 NSP 计数器仍归零（归零与是否随机无关）。
    """
    tubes = _random_tubes(2, 20)
    orch = _orch(tubes, seed=None)
    orch.run()
    first = orch._nsp_calls
    orch.run()
    assert orch._nsp_calls == first, (
        "seed=None 时 NSP 计数器也应每次归零——归零与随机性无关"
    )


def test_empty_tubes_still_resets():
    """空输入提前返回**之前**也必须先归零——否则脏状态留在实例上。

    ``run()`` 在 `if not self.tubes` 处提前 return；归零若写在它之后，
    空输入这条路径就不会归零。
    """
    orch = ProofOrchestrator([], max_iterations=3, grid_n=16)
    orch._nsp_calls = 42
    orch._move_attempts = {"MOVE1_LONGER_GRAINS": ((), True)}
    orch.run()
    assert orch._nsp_calls == 0, (
        "空输入提前返回时没有归零——归零语句必须在 `if not self.tubes` 之前"
    )
    assert orch._move_attempts == {}, (
        "空输入时 `_move_attempts` 没被归零（它本该先清后判）"
    )


@pytest.mark.parametrize("n", [8, 64])
def test_idempotent_at_other_sizes(n: int):
    """不同管数下也幂等（参数化，覆盖边界）。"""
    tubes = _random_tubes(3, n)
    fresh = _signature(_orch(tubes).run())
    reused = _orch(tubes)
    reused.run()
    assert _signature(reused.run()) == fresh, f"n={n} 时第二次 run() 分叉"
