"""守门：**动态调度器**——按提问拆子问题、逐个子问题决定调度哪些模块（2026-10-07）。

**用户在问什么**（2026-10-07 定调）："这是一个**动态**的调度——要判断每一个
提问、或者说每一个提问里面的**子问题**，它需要**调度哪一些模块**。"

**本文件守三件事**：

1. **分解正确**：提问里的关键词 → 正确的**模块集合**（问"体积+包含"得到两个
   子问题；问"证明"得到 proof；无关提问得到空；**紧邻否定**排除被否定的模块）。
2. **模块选择正确**：不同模块得到**该得的后端**（纯计算的 volume/contains →
   几何内核；proof/grains/moves → host，不套内核后端）。
3. **资源分配接了预测模型**：给了单核墙钟时，**纯计算**模块的线程数必须来自
   `predict_schedule`（与直接调用一致）；非并行模块**不设线程数**（避免"看着
   像并行其实没并"的误导）。留核（`reserve_cores`）、每模块成本、E 核调度效率
   参数化均需可验证。

**"动态"的判据**：不同提问产出**不同**计划——这正是"判断每个提问需要哪些模块"。
若调度是静态的（总返回同一组），下面"不同提问给出不同模块集"的用例会红。
"""

from __future__ import annotations

import pytest

from nano.model.chip_model import (
    M4_REFERENCE,
    estimate_single_core_ms,
    predict_schedule,
)
from nano.runtime.resource_arbiter import (
    POLICY_FIFO,
    PRIORITY_BATCH,
    PRIORITY_INTERACTIVE,
    NodeLedger,
    ResourceGrant,
)
from nano.scheduling.scheduler import (
    DEFAULT_E_CORE_EFFICIENCY,
    MODULE_DEPENDENCIES,
    MODULE_REGISTRY,
    ModuleKind,
    ProblemSize,
    compute_execution_order,
    decompose_query,
    plan_query,
    resolve_module,
    verify_registry,
)


class TestDecomposition:
    """提问 → 子问题（模块集合）必须按内容动态变化。"""

    def test_volume_and_contains_split_into_two(self):
        """同时问"体积"与"是否在集合内" ⇒ 拆出 volume + contains 两个子问题。"""
        subs = decompose_query("算一下并集体积，并检查这些点是否在集合内")
        mods = {s.module for s in subs}
        assert ModuleKind.VOLUME_VOXEL in mods, f"未拆出体积：{subs}"
        assert ModuleKind.CONTAINS in mods, f"未拆出包含：{subs}"

    def test_proof_query_maps_to_proof(self):
        """问"跑一遍证明编排" ⇒ 只得到 proof（不该误加几何模块）。"""
        subs = decompose_query("跑一遍证明编排，看下界是否成立")
        assert [s.module for s in subs] == [ModuleKind.PROOF], subs

    def test_unrelated_query_yields_empty(self):
        """无关提问 ⇒ **空**（不硬塞默认模块，调用方据此知道"无可调度计算"）。"""
        assert decompose_query("你好，今天天气怎么样") == []

    def test_empty_query_yields_empty(self):
        assert decompose_query("") == []
        assert decompose_query("   ") == []

    def test_different_queries_give_different_modules(self):
        """**动态性判据**：不同提问 ⇒ 不同模块集合。

        若调度是静态的（不管问什么都返回同一组），这条会红——它把"动态"这
        个词变成可判定的性质。
        """
        a = {s.module for s in decompose_query("算体积")}
        b = {s.module for s in decompose_query("跑证明")}
        assert a != b, f"两种提问给出相同模块集 {a} —— 调度不是动态的"

    def test_subquestions_ordered_by_query_position(self):
        """子问题按触发词在提问里的**位置**排，不是关键词表顺序。

        "先检查包含、再算体积" ⇒ contains 在前、volume 在后——保留用户表述的
        先后（调用方据此决定先做哪个）。
        """
        subs = decompose_query("先检查这些点是否在集合内，再算并集体积")
        assert [s.module for s in subs] == [
            ModuleKind.CONTAINS, ModuleKind.VOLUME_VOXEL,
        ], f"未按提问先后排序：{[s.module.value for s in subs]}"

    def test_bare_mc_keyword_matches(self):
        """裸 "mc"（无空格、非"蒙特卡洛"）也必须命中 VOLUME_MC。"""
        subs = decompose_query("用mc算体积")
        assert any(s.module == ModuleKind.VOLUME_MC for s in subs), (
            f"裸 mc 未命中蒙特卡洛模块：{subs}"
        )

    def test_mc_and_voxel_overlap_kept_and_ordered(self):
        """"蒙特卡洛并集体积"同时命中 MC 与 VOXEL——**两者都保留**（确定口径 +
        统计口径双轨），且按位置排（mc 在前）。

        这是刻意的取舍：体素口径可逐位核对、MC 口径带噪可给 CI；是否两个都算
        由调用方决定，分解器只如实报告"两个口径都相关"。
        """
        subs = decompose_query("蒙特卡洛并集体积")
        assert [s.module for s in subs] == [
            ModuleKind.VOLUME_MC, ModuleKind.VOLUME_VOXEL,
        ], [s.module.value for s in subs]

    def test_trigger_position_recorded(self):
        """触发词的位置必须记入 ``pos``（审计"它在原文哪一处"）。

        触发词取该模块所有关键词在提问里的**最早出现**——"算一下并集体积"里
        "并集"（@3）早于"体积"（@5），故触发词是"并集"。
        """
        subs = decompose_query("算一下并集体积")
        v = next(s for s in subs if s.module == ModuleKind.VOLUME_VOXEL)
        assert (v.trigger, v.pos) == ("并集", 3), (v.trigger, v.pos)

    def test_negated_module_is_excluded(self):
        """紧邻否定标记（"不要/别用/排除/除了"）⇒ 被修饰的模块**不**调度。"""
        assert decompose_query("不要蒙特卡洛") == []
        assert decompose_query("别用mc") == []
        assert decompose_query("排除采样") == []

    def test_negation_keeps_other_modules(self):
        """"不要蒙特卡洛，用体素算体积" ⇒ 只保留 VOXEL，MC 被否定排除。"""
        subs = decompose_query("不要蒙特卡洛，用体素算体积")
        assert [s.module for s in subs] == [ModuleKind.VOLUME_VOXEL], subs


class TestModuleSelection:
    """模块 → 后端选择必须符合"纯计算走内核、编排走 host"。"""

    def test_compute_modules_use_kernel_backend(self):
        """volume / contains（纯计数计算）⇒ 几何内核后端（nano/fast），不是 host。"""
        plan = plan_query("算体积并检查这些点是否在集合内", nano_available=True)
        assert len(plan.tasks) >= 2, "应拆出 volume + contains 两个子问题"
        for t in plan.tasks:
            assert t.backend in ("nano", "fast"), (
                f"纯计算模块 {t.sub.module.value} 选了非内核后端 {t.backend}"
            )

    def test_orchestration_module_uses_host(self):
        """proof（端到端编排）⇒ host 后端（不是几何内核）。"""
        plan = plan_query("跑证明编排", nano_available=True)
        assert plan.tasks[0].backend == "host", plan.tasks[0]

    def test_no_nano_falls_back(self):
        """f32 内核不可用时，纯计算模块退回非 nano 后端（不崩、不误选 nano）。"""
        plan = plan_query("算体积", nano_available=False)
        assert plan.tasks[0].backend in ("fast", "numpy"), plan.tasks[0]
        assert plan.tasks[0].backend != "nano"


class TestResourceAssignment:
    """资源分配必须**接预测模型**：线程数来自 `predict_schedule`。"""

    def test_threads_match_predict_schedule(self):
        """给了单核墙钟 ⇒ 纯计算模块的线程数必须与 `predict_schedule` 一致
        （用调度器**实际使用的默认参数**：E 核调度效率 + P 核降频比）。"""
        t1 = 51.37
        plan = plan_query("算体积", single_core_ms=t1, nano_available=True)
        expected = predict_schedule(
            t1, p_cores=M4_REFERENCE.p_cores, e_cores=M4_REFERENCE.e_cores,
            e_core_efficiency=DEFAULT_E_CORE_EFFICIENCY,
            p_core_all_ratio=M4_REFERENCE.p_core_all_ratio,
        ).best_n
        assert plan.tasks[0].threads == expected, (
            f"线程数 {plan.tasks[0].threads} ≠ 预测最优 {expected} —— "
            "调度没接预测模型（或默认参数漂移）"
        )

    def test_non_parallel_module_gets_no_threads(self):
        """非并行模块（proof/grains/moves）**不设线程数**（0），不误导。"""
        for q in ("跑证明编排", "做 grains 分解", "执行 move 压缩"):
            plan = plan_query(q, single_core_ms=51.4)
            for t in plan.tasks:
                if not t.sub.module.value in ("volume_voxel", "volume_mc", "contains"):
                    assert t.threads == 0, (
                        f"{t.sub.module.value} 非并行却给了线程数 {t.threads}"
                    )

    def test_no_single_core_means_default_threads(self):
        """没给单核墙钟 ⇒ 线程=芯片可调度核数（总核数，不留核时=全核），
        而非 0——"用芯片默认全核"应落成**具体数字**，否则计划读不出实际开几核。"""
        plan = plan_query("算体积")
        assert plan.tasks[0].threads == 10, plan.tasks[0]


class TestReserveCoresAndParams:
    """留核、每模块成本、E 核调度效率参数化必须真实生效。"""

    def test_reserve_cores_reduces_default_threads(self):
        """不留核时默认全核 10；留 2 核 ⇒ 可调度 8。"""
        plan = plan_query("算体积", reserve_cores=2)
        assert plan.tasks[0].threads == 8, plan.tasks[0]

    def test_reserve_cores_caps_predicted_threads(self):
        """8 P 核芯片单核墙钟→预测最优 8，但留 2 核 ⇒ 上限 6。"""
        from dataclasses import replace

        chip8 = replace(M4_REFERENCE, p_cores=8, e_cores=0)
        plan = plan_query("算体积", chip=chip8, single_core_ms=51.4, reserve_cores=2)
        assert plan.tasks[0].threads == 6, plan.tasks[0]

    def test_reserve_cores_validation(self):
        """留核数必须合法：负值或 ≥ 总核数都拒绝。"""
        with pytest.raises(ValueError):
            plan_query("算体积", reserve_cores=-1)
        with pytest.raises(ValueError):
            plan_query("算体积", reserve_cores=10)  # 把全核都留掉

    def test_per_module_cost_overrides_scalar(self):
        """每模块单核墙钟（``single_core_ms_by_module``）优先于全局标量。"""
        t1_scalar = 51.4
        t1_volume = 200.0
        plan = plan_query(
            "算体积并检查这些点是否在集合内",
            single_core_ms=t1_scalar,
            single_core_ms_by_module={ModuleKind.VOLUME_VOXEL: t1_volume},
        )
        vol = next(t for t in plan.tasks if t.sub.module == ModuleKind.VOLUME_VOXEL)
        con = next(t for t in plan.tasks if t.sub.module == ModuleKind.CONTAINS)
        exp_vol = predict_schedule(
            t1_volume, p_cores=M4_REFERENCE.p_cores, e_cores=M4_REFERENCE.e_cores,
            e_core_efficiency=DEFAULT_E_CORE_EFFICIENCY,
            p_core_all_ratio=M4_REFERENCE.p_core_all_ratio,
        ).best_n
        exp_con = predict_schedule(
            t1_scalar, p_cores=M4_REFERENCE.p_cores, e_cores=M4_REFERENCE.e_cores,
            e_core_efficiency=DEFAULT_E_CORE_EFFICIENCY,
            p_core_all_ratio=M4_REFERENCE.p_core_all_ratio,
        ).best_n
        assert vol.threads == exp_vol
        assert con.threads == exp_con

    def test_e_core_efficiency_param_flows_through(self):
        """E 核调度效率参数化：线程建议与 ``predict_schedule(r, d)`` 一致，且计划
        里的假设如实反映传入的 r（不是硬编码默认值）。"""
        r = 0.5
        t1 = 51.4
        plan = plan_query("算体积", single_core_ms=t1, e_core_efficiency=r)
        expected = predict_schedule(
            t1, p_cores=M4_REFERENCE.p_cores, e_cores=M4_REFERENCE.e_cores,
            e_core_efficiency=r,
            p_core_all_ratio=M4_REFERENCE.p_core_all_ratio,
        ).best_n
        assert plan.tasks[0].threads == expected
        assert any(f"{r:.2f}" in a for a in plan.assumptions), plan.assumptions


class TestPlanShape:
    """计划对象本身的性质。"""

    def test_explain_is_nonempty_string(self):
        plan = plan_query("算体积并检查这些点是否在集合内", single_core_ms=51.4)
        s = plan.explain()
        assert isinstance(s, str) and len(s) > 20
        assert "volume_voxel" in s and "contains" in s

    def test_empty_query_plan_explains_emptiness(self):
        plan = plan_query("你好")
        assert plan.tasks == []
        assert "无可调度" in plan.explain() or "没有匹配" in plan.explain()

    def test_custom_chip_respected(self):
        """换一块芯片（不同 P/E）⇒ 线程建议随之变化（用 custom chip 验证）。"""
        from dataclasses import replace

        # 8 个 P 核、无 E 核：预测最优应是 8
        chip8 = replace(M4_REFERENCE, p_cores=8, e_cores=0)
        plan = plan_query("算体积", chip=chip8, single_core_ms=51.4)
        assert plan.tasks[0].threads == 8, plan.tasks[0]


class TestExecutionLink:
    """执行衔接：计划里的每个 ModuleKind 都对应一个**真实模块与入口**。"""

    def test_every_module_kind_has_a_target(self):
        """每个 ModuleKind 都必须登记真实模块——漏一个会让计划"无执行落点"。"""
        for kind in ModuleKind:
            assert kind in MODULE_REGISTRY, f"{kind} 未登记真实模块"

    def test_targets_are_concrete_and_known(self):
        """映射必须是具体的 ``模块.入口``（非空、非占位），且是**本机核对过的
        真实入口**（2026-10-07 核对主包 src/kakeya_ai）。"""
        expected = {
            ModuleKind.VOLUME_VOXEL: ("kakeya_ai.volume", "estimate_union_volume_voxel"),
            ModuleKind.VOLUME_MC: ("kakeya_ai.volume", "estimate_union_volume_monte_carlo_with_ci"),
            ModuleKind.CONTAINS: ("kakeya_ai.accel", "contains_points"),
            ModuleKind.GRAINS: ("kakeya_ai.grains", "partition_into_rho_tubes"),
            ModuleKind.MOVES: ("kakeya_ai.moves", "apply_moves"),
            ModuleKind.PROOF: ("kakeya_ai.proof_orchestrator", "run_proof_skeleton"),
        }
        for kind, (mod, entry) in expected.items():
            t = resolve_module(kind)
            assert (t.module, t.entry) == (mod, entry), f"{kind} 映射漂移：{t}"

    def test_plan_carries_target(self):
        """计划里每个 TaskPlan 都带 target（真实模块+入口），explain() 可见。"""
        plan = plan_query("算体积并检查这些点是否在集合内")
        for t in plan.tasks:
            assert t.target is not None, f"{t.sub.module} 无 target"
            assert t.target.module.startswith("kakeya_ai."), t.target
        assert "kakeya_ai." in plan.explain()

    def test_registry_verifies_resolvable(self):
        """find_spec 核对：在 pytest 的 sys.path（pythonpath=src）下，所有登记
        模块都应**可解析**（不 import 运行时代码，只查 spec 是否存在）。"""
        status = verify_registry()
        assert all(status.values()), (
            f"有模块在 sys.path 上解析不到：{ {k.value: v for k, v in status.items()} }"
        )


class TestExecutionOrder:
    """**跨子问题执行顺序**：拓扑分层 + 核预算裁定（并行/串行）。

    这是"调度系统"区别于"平铺任务清单"的关键：两个纯计算子问题各要满核时，
    计划必须诚实说"串行跑"，而不是各自给满核（那会在 10 核上开出 2×9=18 线程
    的超订计划）。此前只按提问先后排平序、不判并发（见 PROJECT.md §8 边界）。
    """

    def test_single_task_single_wave(self):
        """单个子问题 ⇒ 一层、单任务（无并发对象）。"""
        plan = plan_query("算体积")
        order = plan.execution_order
        assert len(order.waves) == 1
        w = order.waves[0]
        assert w.parallel is True
        assert w.task_indices == (0,)
        assert order.has_parallelism is False

    def test_two_compute_tasks_oversubscribe_serialize(self):
        """两个纯计算子问题各要 8 线程、可调度仅 8 核 ⇒ 判**串行**（不超订）。

        这暴露并修复旧计划的误导：旧 `tasks` 会给两个子问题各 8 线程（合计 16），
        但执行顺序层如实说"串行、各用满线程"。
        """
        plan = plan_query("算体积并检查这些点是否在集合内", reserve_cores=2)
        assert len(plan.tasks) == 2
        assert plan.available_cores == 8
        order = plan.execution_order
        assert len(order.waves) == 1, "无依赖 ⇒ 同一拓扑层"
        w = order.waves[0]
        assert w.parallel is False, "并发线程需求 16 > 8 ⇒ 应串行"
        assert w.task_indices == (0, 1)
        assert w.threads_per_task == (8, 8)
        assert order.has_parallelism is False
        assert "串行" in w.rationale

    def test_host_tasks_overlap_in_parallel(self):
        """线程数 0 的 host 任务（grains/proof）不占内核预算 ⇒ 可并行。"""
        plan = plan_query("做 grains 分解并跑证明")
        assert [t.sub.module for t in plan.tasks] == [
            ModuleKind.GRAINS, ModuleKind.PROOF,
        ], [t.sub.module.value for t in plan.tasks]
        order = plan.execution_order
        assert len(order.waves) == 1
        w = order.waves[0]
        assert w.parallel is True
        assert w.threads_per_task == (0, 0)
        assert order.has_parallelism is True

    def test_injected_dependency_layers_waves(self):
        """声明式依赖 ⇒ 拓扑分层：contains 依赖 volume ⇒ 先 volume、后 contains。"""
        plan = plan_query("算体积并检查这些点是否在集合内")
        order = compute_execution_order(
            plan,
            dependencies={ModuleKind.CONTAINS: {ModuleKind.VOLUME_VOXEL}},
        )
        assert len(order.waves) == 2, [w.task_indices for w in order.waves]
        assert order.waves[0].task_indices == (0,), "volume 应先执行"
        assert order.waves[1].task_indices == (1,), "contains 应后执行"

    def test_default_dependencies_table_is_empty(self):
        """默认依赖表为空（每个子问题都是对已就绪管族的原子操作），但机制已备。"""
        assert MODULE_DEPENDENCIES == {}

    def test_empty_plan_has_no_waves(self):
        """无关提问 ⇒ 空计划 ⇒ 无可执行步骤。"""
        plan = plan_query("你好")
        order = plan.execution_order
        assert order.waves == ()
        assert order.has_parallelism is False

    def test_explain_includes_execution_order(self):
        """explain() 应把执行顺序（并行/串行）展示出来，读者可据此刻断。"""
        plan = plan_query("算体积并检查这些点是否在集合内", reserve_cores=2)
        assert "执行顺序" in plan.explain()
        assert "串行" in plan.explain()

    def test_available_cores_override(self):
        """核预算裁定可用显式 ``available_cores`` 覆盖计划里的值。"""
        plan = plan_query("算体积并检查这些点是否在集合内")
        # 计划默认可调度 10 核（不留核），但显式给 20 核 ⇒ 两个 9 线程任务可并行。
        order = compute_execution_order(plan, available_cores=20)
        assert order.waves[0].parallel is True
        assert order.has_parallelism is True


class TestMultiTenantArbiter:
    """**多租户资源仲裁**：把"芯片有多少核"换成"这个租户被授予多少核"。

    这是把单租户独占调度器**插进云/多租户**的最小一步（2026-10-08）：节点级
    账本（`NodeLedger`）发放 `ResourceGrant`，`plan_query(grant=...)` 消费它，
    于是两个租户各自的计划**不会**都宣布"我要满核"、在 10 核上开出 2×8 线程——
    这正是 `compute_execution_order` 已修复的"计划内超订"的跨租户放大版。
    """

    def test_grant_is_validated(self):
        """授权参数非法必须当场拒绝（核数 <1、空租户名、负内存）。"""
        import pytest

        with pytest.raises(ValueError):
            ResourceGrant("A", cores=0)
        with pytest.raises(ValueError):
            ResourceGrant("", cores=4)
        with pytest.raises(ValueError):
            ResourceGrant("A", cores=4, memory_bytes=-1)

    def test_ledger_capacity_math(self):
        """账本容量：可分配 = 总量 - 保留；空闲 = 可分配 - 已发放。"""
        led = NodeLedger(total_cores=10, reserved_cores=2)
        assert led.allocatable_cores == 8
        assert led.free_cores == 8
        d = led.admit("A", cores=8)
        assert d.granted
        assert d.grant is not None          # granted ⇒ 必有 grant（类型上仍可空）
        assert d.grant.cores == 8
        assert led.allocated_cores == 8 and led.free_cores == 0

    def test_admit_is_idempotent(self):
        """同一租户重复 admit ⇒ 返回既有授权、**不重复记账**（防账面翻倍）。"""
        led = NodeLedger(total_cores=10, reserved_cores=2)
        first = led.admit("A", cores=8).grant
        again = led.admit("A", cores=4)
        assert again.grant is first, "重复 admit 应返回同一授权"
        assert led.allocated_cores == 8, "重复 admit 不该叠加"

    def test_equal_share_caps_single_tenant(self):
        """等分策略：单租户上限 = 可分配核数（此例 8），请求 9 也只给 8。"""
        led = NodeLedger(total_cores=10, reserved_cores=2)
        d = led.admit("A", cores=9)
        assert d.grant is not None          # granted ⇒ 必有 grant（类型上仍可空）
        assert d.grant.cores == 8, d.grant

    def test_reject_when_full(self):
        """空闲耗尽 ⇒ admit 返回 ``granted=False``（**返回而非抛异常**）。"""
        led = NodeLedger(total_cores=4, reserved_cores=0)
        assert led.admit("A", cores=4).granted
        d = led.admit("B", cores=4)
        assert d.granted is False and d.grant is None
        assert "拒绝" in d.reason or "不足" in d.reason

    def test_release_returns_cores(self):
        """释放后核数回到空闲池（"释放给我看"的审计）。"""
        led = NodeLedger(total_cores=10, reserved_cores=2)
        led.admit("A", cores=8)
        freed = led.release("A")
        assert freed is not None and freed.cores == 8
        assert led.free_cores == 8 and led.allocated_cores == 0
        assert led.release("A") is None, "重复释放返回 None"

    def test_fifo_policy_greedy(self):
        """``fifo`` 策略：能吃多少给多少（不像等分那样按活跃数摊）。"""
        led = NodeLedger(total_cores=10, reserved_cores=2, policy=POLICY_FIFO)
        first = led.admit("A", cores=8)
        assert first.grant is not None
        assert first.grant.cores == 8
        assert led.admit("B", cores=8).granted is False

    def test_grant_injection_drives_plan_cores(self):
        """**核心接线**：`plan_query(grant=...)` 的可调度核数必须来自授权，
        而不是芯片拓扑（10 核）——这是"单租户规划器"变"租户内规划器"的判据。"""
        led = NodeLedger(total_cores=10, reserved_cores=2)
        grant = led.admit("A", cores=3).grant
        plan = plan_query("算体积", single_core_ms=51.4, grant=grant)
        assert plan.available_cores == 3, "核预算应来自授权（3），而非芯片（10）"
        assert plan.grant is grant
        # 线程建议不超过授权核数。
        assert plan.tasks[0].threads <= 3, plan.tasks[0]
        assert any("多租户" in a for a in plan.assumptions), plan.assumptions

    def test_two_tenants_do_not_oversubscribe_node(self):
        """**反超订判据**：两个租户各拿一半核，各自的计划**不越权**——
        合并线程需求不超过节点可分配核数（8），而不像单租户那样各自要满核。"""
        led = NodeLedger(total_cores=10, reserved_cores=2)
        gA = led.admit("A", cores=4).grant
        gB = led.admit("B", cores=4).grant
        pA = plan_query("算体积", single_core_ms=51.4, grant=gA)
        pB = plan_query("算体积", single_core_ms=51.4, grant=gB)
        assert pA.available_cores + pB.available_cores <= led.allocatable_cores
        assert pA.tasks[0].threads == 4 and pB.tasks[0].threads == 4

    def test_single_tenant_default_unchanged(self):
        """**行为不变**：不传 grant 时，可调度核数仍由芯片拓扑 - 留核算出
        （`grant is None`、`available_cores == 8`）——多租户是**可选**接入。"""
        plan = plan_query("算体积", reserve_cores=2)
        assert plan.grant is None
        assert plan.available_cores == 8

    def test_ledger_validates_capacity(self):
        """账本自身参数校验：总核 <1、保留 ≥ 总量、未知策略都拒绝。"""
        import pytest

        with pytest.raises(ValueError):
            NodeLedger(total_cores=0)
        with pytest.raises(ValueError):
            NodeLedger(total_cores=4, reserved_cores=4)
        with pytest.raises(ValueError):
            NodeLedger(total_cores=4, policy="bogus")

    def test_ledger_explain_is_auditable(self):
        """``explain()`` 报出容量/保留/已发放/各租户——可审计。"""
        led = NodeLedger(total_cores=10, reserved_cores=2)
        led.admit("A", cores=4)
        s = led.explain()
        assert "总核 10" in s and "保留 2" in s
        assert "租户 'A'" in s


class TestMultiTenantQueue:
    """**准入队列 + 优先级 + 释放后重平衡**（P1，2026-10-08）。

    补上 P0 如实登记的缺口：`admit` 空闲为 0 时**硬拒**后到者（A 先独占后 B
    被拒）。`submit` 改为**入队**（返回票），`release` + `drain` 按
    ``(priority, seq)`` 兑现——**不抢占**，但避免后到者被永久饿死。
    """

    def test_submit_grants_when_capacity_free(self):
        """有容量 ⇒ `submit` 直接发授权（与 admit 同）。"""
        led = NodeLedger(total_cores=10, reserved_cores=2)
        d = led.submit("A", cores=4)
        assert d.grant is not None          # granted ⇒ 必有 grant（类型上仍可空）
        assert d.status == "granted" and d.grant.cores == 4
        assert d.ticket is None

    def test_submit_queues_when_full(self):
        """空闲耗尽 ⇒ `submit` **入队**（不是硬拒），返回票。"""
        led = NodeLedger(total_cores=4, reserved_cores=0)
        led.submit("A", cores=4)
        d = led.submit("B", cores=4)
        assert d.status == "queued" and d.queued is True
        assert d.ticket is not None and d.ticket.tenant == "B"
        assert led.queued_count == 1

    def test_submit_is_idempotent_for_queued_tenant(self):
        """同租户重复 `submit`（仍在队列）⇒ 返回同一张票、不重复入队。"""
        led = NodeLedger(total_cores=4, reserved_cores=0)
        led.submit("A", cores=4)
        t1 = led.submit("B", cores=2).ticket
        t2 = led.submit("B", cores=2).ticket
        assert t1 is t2 and led.queued_count == 1

    def test_release_then_drain_admits_waiter(self):
        """**核心修复**：A 释放后 `drain` 把等待的 B 准入——不再永久被拒。"""
        led = NodeLedger(total_cores=4, reserved_cores=0)
        led.submit("A", cores=4)
        led.submit("B", cores=4)
        assert led.queued_count == 1
        led.release("A")
        admitted = led.drain()
        assert all(d.grant is not None for d in admitted)
        assert [d.grant.tenant for d in admitted if d.grant] == ["B"]
        assert led.grant_for("B") is not None
        assert led.queued_count == 0

    def test_priority_interactive_before_batch(self):
        """优先级：交互式（0）先于批处理（1）被服务，即使它**后到**。"""
        led = NodeLedger(total_cores=2, reserved_cores=0)
        led.submit("A", cores=2)
        led.submit("B", cores=2, priority=PRIORITY_BATCH)       # 先到、低优先级
        led.submit("C", cores=2, priority=PRIORITY_INTERACTIVE)  # 后到、高优先级
        assert [t.tenant for t in led.pending()] == ["C", "B"], led.pending()
        led.release("A")
        admitted = led.drain()
        assert all(d.grant is not None for d in admitted)
        assert [d.grant.tenant for d in admitted if d.grant] == ["C"], \
            "交互式应插队先服务"
        assert led.grant_for("B") is None and led.queued_count == 1

    def test_fifo_tiebreak_within_same_priority(self):
        """同优先级 ⇒ 按到达顺序（seq）服务。"""
        led = NodeLedger(total_cores=2, reserved_cores=0)
        led.submit("A", cores=2)
        led.submit("first", cores=2, priority=PRIORITY_BATCH)
        led.submit("second", cores=2, priority=PRIORITY_BATCH)
        assert [t.tenant for t in led.pending()] == ["first", "second"]
        led.release("A")
        led.drain()
        assert led.grant_for("first") is not None
        assert led.grant_for("second") is None

    def test_drain_reduces_grant_to_fair_cap(self):
        """装不下整份需求时，按**公平上限**截断发放（部分满足），不卡整份。

        构造：4 核节点，A+C 各持 2 核占满；B 要 4 而排队。A 释放 2 核后
        active 只剩 C（1 个），fair cap = 4 // 2 = 2 ⇒ B 被截断到 2（而非 4）。
        """
        led = NodeLedger(total_cores=4, reserved_cores=0)
        led.submit("A", cores=2)
        led.submit("C", cores=2)
        db = led.submit("B", cores=4)
        assert db.status == "queued", db
        led.release("A")
        admitted = led.drain()
        assert admitted and admitted[0].grant is not None
        assert admitted[0].grant.tenant == "B"
        assert admitted[0].grant.cores == 2, "应截断到公平上限（4//2），而非 4"

    def test_drain_empty_queue_returns_nothing(self):
        """队列为空 ⇒ `drain` 返回空列表（幂等、无副作用）。"""
        led = NodeLedger(total_cores=4, reserved_cores=0)
        assert led.drain() == []

    def test_queue_independent_of_scheduler(self):
        """队列是**仲裁层**的事——入队后 `plan_query` 仍能拿**已授予**的授权跑，
        未授予的租户根本拿不到 grant（不会越权）。"""
        led = NodeLedger(total_cores=10, reserved_cores=2)
        gA = led.submit("A", cores=8).grant
        dB = led.submit("B", cores=8)
        assert dB.status == "queued" and dB.grant is None
        planA = plan_query("算体积", single_core_ms=51.4, grant=gA)
        assert planA.available_cores == 8


class TestScaleAwarePlanning:
    """规模感知（2026-10-30）：给了 ``sizes``，单核墙钟由 chip_model 从规模推算。

    守门要点：线程数必须与"用模型推出的 t1 调 ``predict_schedule``"一致；
    显式 ms（每模块 > 全局）优先于规模推算；非 pair 级模块（proof 等）给了
    sizes 也**忽略**；非法键/值与非法校准参数当场报错。
    """

    def _expected_threads(self, t1: float) -> int:
        """与调度器同参数调 predict_schedule 的期望最优线程数。"""
        return predict_schedule(
            t1, p_cores=M4_REFERENCE.p_cores, e_cores=M4_REFERENCE.e_cores,
            e_core_efficiency=DEFAULT_E_CORE_EFFICIENCY,
            p_core_all_ratio=M4_REFERENCE.p_core_all_ratio,
        ).best_n

    def test_sizes_drive_threads_via_model(self):
        """sizes + 实测 ns/pair ⇒ t1 = 12.8ms ⇒ 线程数 = predict_schedule(t1)。"""
        plan = plan_query(
            "算体积",
            sizes={ModuleKind.VOLUME_VOXEL: ProblemSize(64, 200_000)},
            measured_ns_per_pair=1.0,
        )
        t1 = estimate_single_core_ms(64, 200_000, measured_ns_per_pair=1.0)
        assert plan.tasks[0].threads == self._expected_threads(t1), (
            f"线程数 {plan.tasks[0].threads} ≠ 规模推算后预测最优 "
            f"{self._expected_threads(t1)} —— sizes 没接进调度"
        )

    def test_rationale_names_scale_source(self):
        """理由必须写明"由规模 … 推算"，可审计墙钟从哪来。"""
        plan = plan_query(
            "算体积",
            sizes={ModuleKind.VOLUME_VOXEL: ProblemSize(64, 200_000)},
        )
        assert "规模 64 管×200000 点推算" in plan.tasks[0].rationale

    def test_size_key_accepts_value_string(self):
        """键也可是 ``ModuleKind.value`` 字符串（与 single_core_ms_by_module 同口径）。"""
        plan = plan_query(
            "算体积",
            sizes={"volume_voxel": ProblemSize(64, 200_000)},
            measured_ns_per_pair=1.0,
        )
        assert plan.tasks[0].threads > 0
        assert "推算" in plan.tasks[0].rationale

    def test_explicit_scalar_overrides_sizes(self):
        """全局标量 ms 优先于规模推算（显式输入 > 模型推算）。"""
        plan = plan_query(
            "算体积",
            single_core_ms=51.37,
            sizes={ModuleKind.VOLUME_VOXEL: ProblemSize(64, 200_000)},
            measured_ns_per_pair=1.0,
        )
        assert plan.tasks[0].threads == self._expected_threads(51.37)
        assert "全局显式" in plan.tasks[0].rationale
        assert "推算" not in plan.tasks[0].rationale

    def test_per_module_overrides_sizes(self):
        """每模块 ms 优先于规模推算（优先级最高）。"""
        plan = plan_query(
            "算体积",
            single_core_ms_by_module={ModuleKind.VOLUME_VOXEL: 200.0},
            sizes={ModuleKind.VOLUME_VOXEL: ProblemSize(64, 200_000)},
            measured_ns_per_pair=1.0,
        )
        assert plan.tasks[0].threads == self._expected_threads(200.0)
        assert "每模块显式" in plan.tasks[0].rationale

    def test_non_pair_module_ignores_sizes(self):
        """proof 无 pair 计数模型 ⇒ 给了 sizes 也忽略（不设线程数，如实登记）。"""
        plan = plan_query(
            "跑证明编排",
            sizes={ModuleKind.PROOF: ProblemSize(64, 200_000)},
            measured_ns_per_pair=1.0,
        )
        assert plan.tasks[0].threads == 0

    def test_problem_size_validation(self):
        """ProblemSize 的管数/点数必须为正。"""
        with pytest.raises(ValueError):
            ProblemSize(0, 100)
        with pytest.raises(ValueError):
            ProblemSize(64, 0)

    def test_invalid_size_key_rejected(self):
        with pytest.raises(ValueError):
            plan_query("算体积", sizes={"nope": ProblemSize(64, 1)})

    def test_invalid_size_value_rejected(self):
        """值必须是 ProblemSize，不是随手一个 tuple。"""
        with pytest.raises(ValueError):
            plan_query(
                "算体积",
                sizes={ModuleKind.VOLUME_VOXEL: (64, 200_000)},
            )

    def test_bad_calibration_rejected(self):
        """实测 ns/pair 必须为正；model_efficiency 须在 (0,1]。"""
        sizes = {ModuleKind.VOLUME_VOXEL: ProblemSize(64, 200_000)}
        with pytest.raises(ValueError):
            plan_query("算体积", sizes=sizes, measured_ns_per_pair=-1.0)
        with pytest.raises(ValueError):
            plan_query("算体积", sizes=sizes, model_efficiency=1.5)

    def test_explain_mentions_scale_awareness(self):
        plan = plan_query(
            "算体积", sizes={ModuleKind.VOLUME_VOXEL: ProblemSize(64, 200_000)}
        )
        assert "规模感知" in plan.explain()



