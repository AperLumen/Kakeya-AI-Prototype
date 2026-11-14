"""守门：**架构清单**（nano/manifest/architecture_manifest.py）——单一真源不漂移（2026-10-30）。

**用户在问什么**（2026-10-30 定调）：\"整个优化基于模型——架构做成插件形式
（SKILL.md），让模型实现平替架构的更迭。\" 本文件把这句话落成可判定的守门：

1. **清单 ↔ C++ ↔ Python 三方同源**：SHARED_QUANTITIES 登记的值必须等于
   各层源码里的实测值（解析源码比对，不靠运行）——本项目反复踩过的
   \"同一物理量两处各记一个数\"的系统性解法。
2. **DAG 生成物一致性**：docs/diagrams/scheduler_flow.mmd 是清单的渲染
   产物——手改 mmd 而不改清单即红（改架构必须走清单）。
3. **DAG 结构语义**：chip_model/predict_schedule 是"被消费的模型"、
   expert_router 是分工边界——布局语义由数据钉住，防"改图时把模型
   输入画进主链"这类漂移。

**未知问题（如实登记）**：SKILL 注入后模型的决策是否真系统性偏向清单——
本文件守的是"清单不撒谎"，不声称回答了那个经验问题（测量协议见
PROJECT.md 第 III 篇 §6）。
"""

from __future__ import annotations

import re
from pathlib import Path

from nano.manifest.architecture_manifest import (
    EXPERIMENT_PROTOCOL,
    EXPERT_POOL,
    PRINCIPLES,
    SCHEDULER_FLOW_CLASSDEFS,
    SCHEDULER_FLOW_EDGES,
    SCHEDULER_FLOW_MMD,
    SCHEDULER_FLOW_NODES,
    SCHEDULER_FLOW_STATEMENTS,
    SHARED_QUANTITIES,
    SKILL_INFLUENCE_EXPERIMENT,
    merge_experts,
    render_scheduler_flow_mmd,
    split_expert,
)

REPO = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _extract_constexpr(text: str, name: str) -> float:
    """从 C++ 源码抽 ``NAME = <number>;`` 的数值（constexpr 风格）。"""
    m = re.search(rf"{name}\s*=\s*([0-9.]+)", text)
    assert m is not None, f"在源码里找不到常量 {name} —— 名字变了就改本测试"
    return float(m.group(1))


def _extract_fn_return(text: str, name: str) -> float:
    """从 C++ 源码抽 ``name() { ... return <number>; }`` 的返回值。"""
    m = re.search(rf"{name}\(\)[^{{]*{{[^}}]*?return\s+([0-9.]+)", text, re.S)
    assert m is not None, f"在源码里找不到函数 {name} —— 名字变了就改本测试"
    return float(m.group(1))


class TestSharedQuantitiesTripleSource:
    """清单登记值 == C++ 实测值（三方同源的清单一侧）。"""

    def test_partition_weight_manifest_cpp(self):
        """E 核分派权重：清单 == parallel.hpp == bench_micro.cpp（0.60）。

        两个 C++ 锚点都要查——漏一个就会重演"两处各记一个数"。
        """
        qty = next(q for q in SHARED_QUANTITIES if q.name == "e_core_partition_weight")
        hpp_val = _extract_fn_return(_read("cpp/kernel/parallel.hpp"),
                                     "e_core_partition_weight")
        micro_val = _extract_constexpr(_read("nano/isolated/bench_micro.cpp"),
                                       "E_CORE_WEIGHT")
        assert qty.value == hpp_val == micro_val, (
            f"分派权重三方漂移：清单 {qty.value} / parallel.hpp {hpp_val} / "
            f"bench_micro {micro_val} —— 改值必须按 KAKEYA_BENCH_PARTITION=1 "
            "重跑标定后三处同改"
        )

    def test_partition_weight_is_not_scheduler_efficiency(self):
        """分派权重 ≠ predict_schedule 的 E 核调度效率（0.44）——两个量。

        出现同值即是漂移信号：要么有人把两个量合并了（错），要么改了
        其一没改另一个的对照（本测试不假设哪个对，只要求"保持不同"）。
        """
        from nano.scheduling.scheduler import DEFAULT_E_CORE_EFFICIENCY

        qty = next(q for q in SHARED_QUANTITIES if q.name == "e_core_partition_weight")
        assert DEFAULT_E_CORE_EFFICIENCY != qty.value, (
            "分派权重与 E 核调度效率变成同一个数了——两者口径不同"
            "（见 parallel.hpp / architecture_manifest.py 注释），请核实"
        )

    def test_sampling_overhead_manifest_cpp_python(self):
        """融合采样开销：清单 == bench_micro.cpp == chip_model.py。"""
        from nano.model.chip_model import FUSED_MC_SAMPLING_OVERHEAD_NS

        qty = next(q for q in SHARED_QUANTITIES
                   if q.name == "fused_mc_sampling_overhead_ns")
        micro_val = _extract_constexpr(_read("nano/isolated/bench_micro.cpp"),
                                       "FUSED_MC_SAMPLING_OVERHEAD_NS")
        assert qty.value == micro_val == FUSED_MC_SAMPLING_OVERHEAD_NS, (
            f"采样开销三方漂移：清单 {qty.value} / bench_micro {micro_val} / "
            f"chip_model {FUSED_MC_SAMPLING_OVERHEAD_NS}"
        )

    def test_every_quantity_anchor_exists(self):
        """每条共享量的 C++ 锚点文件与符号必须真实存在（防登记空头支票）。"""
        for q in SHARED_QUANTITIES:
            for rel, symbol in q.layer_cpp:
                path = REPO / rel
                assert path.exists(), f"锚点文件不存在：{rel}"
                assert symbol in path.read_text(encoding="utf-8"), (
                    f"{rel} 里找不到符号 {symbol} —— 改名必须同步清单"
                )


class TestSchedulerFlowGenerated:
    """scheduler_flow.mmd 是清单的渲染产物：一致性 + 结构语义。"""

    def test_render_matches_disk(self):
        """手改 mmd 而不改清单即红——改架构必须走清单。"""
        assert render_scheduler_flow_mmd() == SCHEDULER_FLOW_MMD.read_text(
            encoding="utf-8"
        ), (
            f"{SCHEDULER_FLOW_MMD} 与清单渲染不一致——请改 "
            "nano/manifest/architecture_manifest.py 后重跑生成器（不要手改 mmd）"
        )

    def test_main_chain_is_pure_core_then_gate(self):
        """调度主链（q → … → out）必须全由 core 节点 + 判定 gate 组成。

        chip_model / predict_schedule / expert_router 是 model 类——若哪天
        谁把模型输入画进主链（或反过来），本用例立刻暴露。
        """
        edges = {(e.src, e.dst) for e in SCHEDULER_FLOW_EDGES}
        # 主链：从 q 出发、不经过 model 节点的可达路径。
        model_nodes = {n.id for n in SCHEDULER_FLOW_NODES.values()
                       if n.klass == "model"}
        adj: dict[str, list[str]] = {}
        for e in SCHEDULER_FLOW_EDGES:
            adj.setdefault(e.src, []).append(e.dst)
        reachable: set[str] = set()
        stack = ["q"]
        while stack:
            cur = stack.pop()
            if cur in reachable or cur in model_nodes:
                continue
            reachable.add(cur)
            stack.extend(adj.get(cur, []))
        assert reachable.isdisjoint(model_nodes), (
            f"主链混入了模型输入节点：{reachable & model_nodes}"
        )
        assert reachable == {n.id for n in SCHEDULER_FLOW_NODES.values()
                             if n.klass in ("core", "gate", "out")}, (
            "主链节点集合与预期不符（core/gate/out 应全部在主链上）"
        )
        assert ("serial", "out") in edges and ("par", "out") in edges

    def test_expert_router_is_boundary(self):
        """expert_router 只连向 q（分工边界），不进入计划链。"""
        expert_edges = [e for e in SCHEDULER_FLOW_EDGES if "expert" in (e.src, e.dst)]
        assert len(expert_edges) == 1
        e = expert_edges[0]
        assert e.src == "expert" and e.dst == "q", (
            f"expert_router 的边应是 分工边界 → 提问，得到 {e.src} → {e.dst}"
        )

    def test_classdefs_cover_all_nodes(self):
        """每个节点的样式类必须有对应 classDef（防渲染成无样式裸节点）。"""
        defined = {k for k, _ in SCHEDULER_FLOW_CLASSDEFS}
        used = {n.klass for n in SCHEDULER_FLOW_NODES.values()}
        assert used <= defined, f"缺少 classDef：{used - defined}"

    def test_statements_declare_every_node_exactly_once(self):
        """语句表里每个节点恰好声明一次（独立或内联），边引用已声明节点。"""
        declared: list[str] = []
        for stmt in SCHEDULER_FLOW_STATEMENTS:
            if stmt[0] == "node":
                declared.append(stmt[1])
            elif stmt[0] == "edge" and stmt[4]:
                declared.append(stmt[4])
        assert sorted(declared) == sorted(SCHEDULER_FLOW_NODES), (
            "声明集合与节点登记表不符（漏声明/重复声明/登记表多余）"
        )
        assert len(declared) == len(set(declared)), "节点被重复声明"


class TestPrinciplesWellFormed:
    """原则表自身质量：id 唯一、每条带落实机制（enforcement 不得为空）。"""

    def test_unique_ids_and_nonempty_enforcement(self):
        ids = [p.id for p in PRINCIPLES]
        assert len(ids) == len(set(ids)), f"原则 id 重复：{ids}"
        for p in PRINCIPLES:
            assert p.enforcement.strip(), f"{p.id} 缺 enforcement（原则无人守）"
            assert p.statement.strip() and p.title.strip()


class TestExpertPool:
    """专家池单一真源（V3 阶段 3）：清单 EXPERT_POOL ↔ default_experts()。

    守门要点：池组成**只准**长在清单里——default_experts() 是生成物。
    任何一边改了另一边没跟，本类立刻红。
    """

    def test_pool_matches_default_experts(self):
        """名字/关键词/模型档/priority/temperature/effort 网格逐位一致。"""
        from kakeya_ai.ai.expert_router import default_experts

        # 注入哨兵模型名：model_role → 默认模型的映射必须精确可判。
        experts = {e.name: e for e in default_experts(fast_model="FAST", deep_model="DEEP")}
        pool = {s.name: s for s in EXPERT_POOL}
        assert set(pool) == set(experts), (
            f"池成员漂移：清单 {sorted(pool)} vs 代码 {sorted(experts)}"
        )
        deep_grid = {"low": {"num_predict": 512}, "high": {"num_predict": 1024},
                     "max": {"num_predict": 2048}}
        fast_grid = {"low": {"num_predict": 256}, "high": {"num_predict": 512},
                     "max": {"num_predict": 1024}}
        for name, spec in pool.items():
            exp = experts[name]
            assert exp.keywords == list(spec.keywords), (
                f"{name} 关键词漂移——改池组成只准改清单"
            )
            assert exp.priority == spec.priority, f"{name} priority 漂移"
            assert exp.options["temperature"] == spec.temperature
            want_model = "DEEP" if spec.model_role == "deep" else "FAST"
            assert exp.model == want_model, (
                f"{name} model_role={spec.model_role} 但 model={exp.model}"
            )
            want_grid = deep_grid if spec.model_role == "deep" else fast_grid
            assert exp.effort_options == want_grid, f"{name} effort 网格漂移"

    def test_general_is_protected_fallback(self):
        """general 必须在池中、无关键词——兜底语义不可被更迭操作破坏。"""
        general = next(s for s in EXPERT_POOL if s.name == "general")
        assert general.keywords == ()
        assert any(s.name == "general" for s in EXPERT_POOL)

    def test_merge_experts_invariants(self):
        """MERGE 操作：general 不可合并、重名拒绝、角色取强、原池不变。"""
        import pytest

        merged = merge_experts(EXPERT_POOL, "writing", "code", "tech_write")
        names = [s.name for s in merged]
        assert "writing" not in names and "code" not in names
        assert "tech_write" in names and "general" in names and "math" in names
        tw = next(s for s in merged if s.name == "tech_write")
        assert tw.model_role == "fast"  # fast + fast = fast
        assert set(tw.keywords) == {
            k for s in EXPERT_POOL for k in s.keywords
            if s.name in ("writing", "code")
        }
        assert len(EXPERT_POOL) == 5, "原池必须不可变（返回新池）"
        with pytest.raises(ValueError):
            merge_experts(EXPERT_POOL, "general", "math", "gm")
        with pytest.raises(ValueError):
            merge_experts(EXPERT_POOL, "math", "math", "mm")
        with pytest.raises(ValueError):
            merge_experts(EXPERT_POOL, "math", "code", "math")  # 重名

    def test_split_expert_partition_invariant(self):
        """SPLIT 操作：keywords 必须是原集划分；general 不可拆分。"""
        import pytest

        code = next(s for s in EXPERT_POOL if s.name == "code")
        kws = sorted(code.keywords)
        half = len(kws) // 2
        pool2 = split_expert(
            EXPERT_POOL, "code", ("code_sys", "code_algo"),
            keywords=(tuple(kws[:half]), tuple(kws[half:])),
        )
        names = [s.name for s in pool2]
        assert "code" not in names and "code_sys" in names and "code_algo" in names
        reunited = sorted(
            k for s in pool2 if s.name in ("code_sys", "code_algo") for k in s.keywords
        )
        assert reunited == kws, "拆分不得丢词或造词"
        with pytest.raises(ValueError):
            split_expert(  # 非划分：丢了最后一个词
                EXPERT_POOL, "code", ("a", "b"),
                keywords=(tuple(kws[:half]), tuple(kws[half:-1])),
            )
        with pytest.raises(ValueError):
            split_expert(EXPERT_POOL, "general", ("g1", "g2"),
                         keywords=((), ()))


class TestSkillInfluenceExperiment:
    """未知问题的测量实验登记质量（如实登记：待执行，无结论）。"""

    def test_experiment_wellformed(self):
        ids = [t.id for t in SKILL_INFLUENCE_EXPERIMENT]
        assert len(ids) == len(set(ids)), f"任务 id 重复：{ids}"
        assert len(SKILL_INFLUENCE_EXPERIMENT) >= 3, "任务集至少 3 个跨层任务"
        for t in SKILL_INFLUENCE_EXPERIMENT:
            assert t.prompt.strip() and t.layers and t.rubric
        assert EXPERIMENT_PROTOCOL.strip()
