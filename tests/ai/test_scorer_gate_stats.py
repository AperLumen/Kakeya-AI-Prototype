"""守门：``scorer_gate_stats()`` 必须把"给了答案但被拦下"独立计出来。

**背景**：``SelectionSource`` 只记"最终谁做的决定"，于是 surrogate 给了 10 次
答案、9 次被守卫拦下时，占比统计显示它只贡献了 1 次——**看不出它试过 10 次**。
而这个"自信但没用"的比例正是训练最该盯的量（修复前实测：surrogate 的 Move
有效率仅 **12%**）。

文本形式的 ``_selection_notes`` 里虽有记录，但文本不适合作聚合。本文件守
``scorer_gate_stats()`` 这个结构化入口。

三条：
1. 被拦下会真的计数（构造一个必被拦下的场景）；
2. ``scorer_gate_stats`` 的口径与 ``SelectionSource`` 占比**不同**——
   被拦下的那些在占比里算 HEURISTIC，这里必须另算（否则这条接口没意义）；
3. 分母为 0 时返回 ``0.0`` 而不是 ``nan``（nan 会静默污染下游累加）。
"""

from __future__ import annotations

import numpy as np
import pytest

from kakeya_ai.ai.surrogate_scorer import SurrogateScorer
from kakeya_ai.ai.training_data import FEATURE_NAMES
from kakeya_ai.primitives import Direction, Tube
from kakeya_ai.proof_orchestrator import (
    ProofOrchestrator,
    SelectionSource,
)


def _random_tubes(seed: int, n: int, delta: float = 0.1) -> list[Tube]:
    """随机构型（采样顺序见 ``test_scorer_move_guard._random_tubes``）。"""
    rng = np.random.default_rng(seed)
    tubes = []
    for _ in range(n):
        center = rng.uniform(-0.3, 0.3, 3)
        d = rng.normal(size=3)
        d = d / np.linalg.norm(d)
        # ``from_array`` 声明吃 ``Sequence[float]``，ndarray 不在其中，
        # ``.tolist()`` 把它翻译成类型检查器看得懂的类型（值不变）。
        tubes.append(
            Tube(
                center=center,
                direction=Direction.from_array(d.tolist()),
                length=1.0,
                delta=delta,
            )
        )
    return tubes


def _sweep(**kw):
    """扫多个构型，返回 (accepts, blocks, 见过的 selection_source 集合)。"""
    accepts = blocks = 0
    sources: set = set()
    for seed in range(6):
        for n in (12, 20):
            orch = ProofOrchestrator(
                _random_tubes(seed, n),
                epsilon=0.05,
                K=2.0,
                max_iterations=12,
                grid_n=16,
                **kw,
            )
            result = orch.run()
            st = orch.scorer_gate_stats()["SURROGATE"]
            accepts += int(st["accepts"])
            blocks += int(st["blocks"])
            for s in result.steps:
                sources.add(s.selection_source)
    return accepts, blocks, sources


# ---------------------------------------------------------------------------
# 1. 被拦下真的会计数
# ---------------------------------------------------------------------------


def _surrogate_available() -> bool:
    """surrogate 权重是否随仓库分发。

    **没有权重时本文件的前提不成立**：编排器会静默退回启发式（见
    ``ProofOrchestrator._load_surrogate``），于是 ``SURROGATE`` 的 accepts/blocks
    永远为 0。把那当成"守卫失效"会误导——故相关用例 skip 并说明原因，
    有权限拿到权重的环境（内部训练产物）仍会完整执行。
    """
    return any(p.is_file() for p in ProofOrchestrator._default_weights_candidates())


def test_blocks_are_counted():
    """**核心**：扫一批构型，必须观察到 blocks > 0。

    实测（2026-09-30）在这批构型上 surrogate 给出 4 个候选、其中一半被拦下。
    若这条挂了：要么守卫不再拦任何东西（行为回归），要么计数没接上
    （接口是装饰品）。两种都该被拦住。
    """
    if not _surrogate_available():
        pytest.skip("surrogate 权重未随仓库分发，SURROGATE 不参与选择，计数必为 0")

    accepts, blocks, _ = _sweep()
    assert accepts + blocks > 0, (
        "整批构型里 surrogate 一次候选都没给出——测不出计数，"
        "换个构型或检查 surrogate 是否被自动加载"
    )
    assert blocks > 0, (
        f"预期观察到被守卫拦下的候选，实测 0 次（accepts={accepts}）——"
        "either 守卫失效，either 计数没接上"
    )


def test_effectiveness_is_consistent_with_counts():
    """``effectiveness`` 必须等于 ``accepts / offered``，且 ``offered`` 是两者之和。"""
    for seed in range(4):
        orch = ProofOrchestrator(
            _random_tubes(seed, 20),
            epsilon=0.05,
            K=2.0,
            max_iterations=12,
            grid_n=16,
        )
        orch.run()
        for src, st in orch.scorer_gate_stats().items():
            assert st["offered"] == st["accepts"] + st["blocks"], (
                f"{src}: offered != accepts + blocks"
            )
            if st["offered"] > 0:
                assert st["effectiveness"] == pytest.approx(
                    st["accepts"] / st["offered"]
                )
            else:
                # 关键：分母为 0 时必须是 0.0，不是 nan。nan 会静默污染
                # 下游的累加与比较（本项目已在别处踩过"统计量变 nan"）。
                assert st["effectiveness"] == 0.0
                assert st["effectiveness"] == st["effectiveness"], (
                    "effectiveness 是 nan——分母为 0 时应返回 0.0"
                )


def test_stats_differ_from_selection_source_shares():
    """**这条是接口存在的理由**：两组数必须能不同。

    被拦下的次数**不进** ``SelectionSource``（那里会算成 HEURISTIC），所以
    ``offered`` 应可能大于 ``SelectionSource==SURROGATE`` 的步数。若两者
    **恒等**，说明本接口只是在重复已有信息、没有增加维度。
    """
    accepts, blocks, sources = _sweep()
    if blocks == 0:
        pytest.skip("这批构型没出现拦下，无法比较（test_blocks_are_counted 会抓）")

    offered = accepts + blocks
    assert offered > accepts, (
        "offered 未超过 accepts——说明被拦下的那些没被单独计数"
    )
    # 且被拦下时最终来源不该是 SURROGATE（它退回了启发式）。
    assert SelectionSource.HEURISTIC in sources, (
        "被拦下后应退回启发式，但整批里没见到 HEURISTIC 来源"
    )


def test_stats_reset_between_runs():
    """**同一个 orchestrator 连跑两次，计数不累积**。

    ⚠️ 不能靠"两次 offered 相等"来判——本项目有**多个跨 run 保留的状态**：
    ``_move_attempts``（Move 失效记忆）、以及 NSP 因子化的随机种子（每次
    运行递增，于是生成的 grains 都不同：实测 166 → 208 → 早停）。清空前者
    也换不来相同的初始条件。

    因此做法是**绕过"行为是否相同"这个不可控前提**，直接测"归零"这件事：
    在第二次 run 之前人为把计数**设成大数**，跑完后若计数仍是大数（或比
    单次运行大得多），说明 run() 开头没归零。

    这样测的是**归零动作本身**，不依赖两次运行的轨迹是否一致。
    """
    orch = ProofOrchestrator(
        _random_tubes(0, 20), epsilon=0.05, K=2.0, max_iterations=12, grid_n=16
    )
    orch.run()

    # 人为污染：若 run() 不归零，这些脏数据会留在结果里。
    sentinel = 1000
    orch._scorer_gate_accepts["SURROGATE"] = sentinel
    orch._scorer_gate_blocks["SURROGATE"] = sentinel
    orch.run()
    after = orch.scorer_gate_stats()["SURROGATE"]

    assert after["accepts"] < sentinel and after["blocks"] < sentinel, (
        f"第二次 run() 后计数仍 ≥ {sentinel} ({after})——"
        "run() 开头没有把打分器门控计数归零"
    )
    # 且必须落在"单次运行"的合理量级内（不可能超过迭代数）。
    assert after["offered"] <= orch.max_iterations, (
        f"第二次 offered={after['offered']} 超过 max_iterations="
        f"{orch.max_iterations}——量级像累积了"
    )


def test_stats_covers_both_scorer_sources():
    """两个来源都要有条目（即使某次为 0），调用方不必判键存在。"""
    orch = ProofOrchestrator(
        _random_tubes(0, 12), epsilon=0.05, K=2.0, max_iterations=3, grid_n=16
    )
    orch.run()
    stats = orch.scorer_gate_stats()
    for name in ("LLM", "SURROGATE"):
        assert name in stats, f"缺少 {name} 条目"
        for key in ("accepts", "blocks", "offered", "effectiveness"):
            assert key in stats[name], f"{name} 缺少 {key}"


def test_llm_gate_is_counted_too():
    """LLM 路径也要计数——两者口径必须对称，否则只有一半被测到。

    注入一个"总是提 MOVE2"的假 LLM scorer（签名须与 ``_try_llm_scorer`` 一致：
    ``f(prompt, state=...) -> (scores_dict, source)``）。MOVE2 的条件在多数
    状态下已满足，因此应观察到 LLM 的 blocks > 0。

    ⚠️ 初版这个测试用 ``except Exception: continue`` + ``pytest.skip``，于是
    **签名写错导致从未真正执行**、变异"LLM accepts 不计数"被静默放行。
    现在改为：假 scorer 的签名已被验证可用，**不允许 skip**——若某构型失败，
    换构型；一个都没成功就直接 fail（漏测比错测更危险）。
    """

    class _AlwaysMove2:
        """总是提 MOVE2 的假 LLM 打分器。"""

        def __call__(self, prompt, *, state=None, **kwargs):
            return {2: 10.0, 1: 1.0, 3: 1.0}, "fake"

    llm_blocks = 0
    llm_offered = 0
    successes = 0
    for seed in range(8):
        orch = ProofOrchestrator(
            _random_tubes(seed, 20),
            epsilon=0.05,
            K=2.0,
            max_iterations=8,
            grid_n=16,
            llm_scorer=_AlwaysMove2(),
            surrogate=None,
        )
        orch.run()
        st = orch.scorer_gate_stats()["LLM"]
        llm_offered += int(st["offered"])
        llm_blocks += int(st["blocks"])
        successes += 1

    assert successes == 8, "有构型未能运行——不应发生（假 scorer 签名已匹配）"
    assert llm_offered > 0, (
        "整批构型里 LLM 一次候选都没给出——假 scorer 可能没被调用"
        "（检查签名是否为 f(prompt, state=...) -> (scores, source)）"
    )
    assert llm_blocks > 0, (
        f"预期观察到 LLM 被守卫拦下的候选，实测 0 次（offered={llm_offered}）"
        "——LLM 计数可能没接上"
    )


def test_llm_accepts_are_counted():
    """**LLM 的 accepts 也要计数**——不能只测 blocks。

    ⚠️ 初版只测了 ``blocks > 0``，于是"``accepts`` 不计数"的变异**被放行**：
    那批构型里 LLM 提的 MOVE2 全部被拦下，accepts 恒为 0，少加一次看不出来。

    提 **MOVE3**：实测（见下）在 12~100 根管的初始 grains 上
    ``check_move3_condition`` **恒为 False**（m1_ok/m2_ok 则恒为 True），
    因此 MOVE3 是这批构型上唯一能真正通过守卫的 Move。
    """

    class _AlwaysMove3:
        """总是提 MOVE3 的假 LLM 打分器。

        另一条路是提 MOVE1/MOVE2——实测它们的条件在初始 grains 上**已满足**，
        必被守卫拦下（只测得到 blocks、测不到 accepts）。
        """

        def __call__(self, prompt, *, state=None, **kwargs):
            return {3: 10.0, 1: 1.0, 2: 1.0}, "fake"

    llm_accepts = 0
    for seed in range(6):
        for n in (12, 20, 32):
            orch = ProofOrchestrator(
                _random_tubes(seed, n),
                epsilon=0.05,
                K=2.0,
                max_iterations=8,
                grid_n=16,
                llm_scorer=_AlwaysMove3(),
                surrogate=None,
            )
            orch.run()
            llm_accepts += int(orch.scorer_gate_stats()["LLM"]["accepts"])

    assert llm_accepts > 0, (
        "扫了 18 个构型，LLM 提的 MOVE3 一次都没被采纳——"
        "accepts 计数没接上，或 MOVE3 条件的前提变了（见 docstring）"
    )


def test_move3_condition_is_initially_unsatisfied():
    """**前提守卫**：上面那条测试依赖"初始 grains 上 MOVE3 条件为 False"。

    条件函数或 grains 构造一旦变化，上面那条测试会变成"测不到 accepts"的
    假通过。这里把前提本身钉住：前提没了就红，提示去选别的 Move。
    """
    from kakeya_ai.moves import check_move3_condition

    checked = 0
    for seed in range(3):
        for n in (12, 32):
            orch = ProofOrchestrator(
                _random_tubes(seed, n),
                epsilon=0.05,
                K=2.0,
                max_iterations=3,
                grid_n=16,
            )
            state = orch.initialize()
            grains = orch._build_grains(state)
            metrics = orch._compute_metrics(state)
            ok = check_move3_condition(
                grains, 1.0 / (metrics.delta ** orch.zeta)
            )
            assert ok is False, (
                f"n={n} seed={seed}: MOVE3 条件变成了 True——"
                "test_llm_accepts_are_counted 需改用别的 Move"
            )
            checked += 1
    assert checked == 6


def test_feature_names_unchanged():
    """**回归保护**：本改动不得动 ``FEATURE_NAMES``。

    可用性**不是**特征——它是评估维度。把它加进特征会改变 ``input_dim``、
    让所有旧权重失效（``FEATURE_NAMES`` 是模型输入维数的唯一真源）。
    """
    assert len(FEATURE_NAMES) == 8, (
        f"FEATURE_NAMES 维数变了（{len(FEATURE_NAMES)}）——若确实要扩维，"
        "需同步重训所有权重并更新 feature_names 元数据，不是本轮的事"
    )


def test_surrogate_scorer_import_is_available():
    """``SurrogateScorer`` 可从该路径导入（本文件与 orchestrator 共用它）。"""
    assert SurrogateScorer is not None
