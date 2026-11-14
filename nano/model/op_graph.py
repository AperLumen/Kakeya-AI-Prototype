"""算子图抽象第一步（Q4 #15，2026-11-13）。

把 kernel 从"单个 KernelSpec"提升为**算子图**：节点是带算力/带宽标注的
算子（沿用 ``KernelSpec`` 计数口径），边是数据流（张量字节数）。v1 的求值
是**逐节点 roofline + 串行求和**（单引擎保守口径；PE 阵列映射的并行
调度是后续）。

两条真实路径入图：
- ``voxel_volume_graph``：体素网格法（包含测试主 kernel + 归约）；
- ``mc_volume_graph``：Monte Carlo 采样法（采样 kernel + 计数归约）。

**计数口径纪律**（沿用 chip_model 传统）：实测逐行计数的标
``source="measured: ..."``；按算法结构估算的标 ``source="estimated: ..."``
——两者绝不混写。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .chip_model import ChipSpec, KernelSpec, TUBE_GEOMETRY_KERNEL

#: MC 采样 kernel：每个采样点一次 box 包含测试的近似计数（结构估算，
#: 未逐行实测——标 estimated，将来实测后替换）。
MC_SAMPLE_KERNEL = KernelSpec(
    name="mc_sample_point_membership",
    flops_per_pair=18,          # 坐标变换 + 每管距离判定（结构估算）
    bytes_per_point=12,         # 采样点坐标（float32×3）
    bytes_per_tube_struct=TUBE_GEOMETRY_KERNEL.bytes_per_tube_struct,
    source="estimated: 结构推算（未逐行实测；voxel 口径见 TUBE_GEOMETRY_KERNEL）",
)

#: 归约 kernel：字节级求和，算力可忽略、带宽主导。
REDUCTION_KERNEL = KernelSpec(
    name="count_reduction",
    flops_per_pair=1,
    bytes_per_point=1,          # 每点一个 0/1
    bytes_per_tube_struct=0,
    source="estimated: 归约是带宽主导（FLOP 可忽略）",
)


@dataclass(frozen=True)
class OpNode:
    """算子图节点：一个算子的一次调用。"""

    name: str
    kernel: KernelSpec
    n_pairs: int                # 本次调用的 (元素) 对数
    n_points: int               # 数据流口径的点数（带宽摊销用）
    n_tubes: int


@dataclass(frozen=True)
class OpEdge:
    """数据流边：上游节点产出 ``bytes`` 字节被下游消费。"""

    src: str
    dst: str
    bytes: int


@dataclass
class OpGraph:
    """算子图（v1：拓扑序串行求值）。"""

    name: str
    nodes: List[OpNode] = field(default_factory=list)
    edges: List[OpEdge] = field(default_factory=list)

    def add(self, node: OpNode) -> None:
        if any(n.name == node.name for n in self.nodes):
            raise ValueError(f"节点重名：{node.name}")
        self.nodes.append(node)

    def connect(self, src: str, dst: str, bytes_: int) -> None:
        names = {n.name for n in self.nodes}
        if src not in names or dst not in names:
            raise ValueError(f"边 {src}->{dst} 引用未定义节点")
        self.edges.append(OpEdge(src, dst, int(bytes_)))


def _node_time_ns(node: OpNode, chip: ChipSpec) -> Tuple[float, float, float]:
    """单节点 roofline：返回 (compute_ns, memory_ns, 取大者)。"""
    flops_per_s = chip.peak_flops_gflops * 1e9
    compute_ns = node.kernel.flops_per_pair * node.n_pairs / flops_per_s * 1e9
    bytes_per_s = chip.peak_bandwidth_gbytes * 1e9
    bytes_total = node.kernel.bytes_per_point * node.n_points
    memory_ns = bytes_total / bytes_per_s * 1e9
    return compute_ns, memory_ns, max(compute_ns, memory_ns)


def evaluate_graph(graph: OpGraph, chip: ChipSpec) -> Dict:
    """求值算子图：逐节点 roofline + 串行总和 + 瓶颈节点。"""
    per_node = []
    total = 0.0
    for node in graph.nodes:
        c, m, t = _node_time_ns(node, chip)
        per_node.append({
            "name": node.name,
            "compute_ns": c,
            "memory_ns": m,
            "time_ns": t,
            "bound": "compute" if c >= m else "memory",
        })
        total += t
    bottleneck = max(per_node, key=lambda r: r["time_ns"]) if per_node else None
    return {
        "graph": graph.name,
        "chip": chip.name,
        "per_node": per_node,
        "total_ns": total,
        "total_ms": total / 1e6,
        "bottleneck": bottleneck["name"] if bottleneck else None,
        "edge_bytes": sum(e.bytes for e in graph.edges),
    }


def voxel_volume_graph(n_tubes: int, grid_n: int) -> OpGraph:
    """体素网格法路径：包含测试主 kernel → 计数归约。"""
    n_points = grid_n ** 3
    g = OpGraph(name=f"voxel_volume(n_tubes={n_tubes}, grid={grid_n})")
    g.add(OpNode("contains", TUBE_GEOMETRY_KERNEL,
                 n_pairs=n_tubes * n_points, n_points=n_points,
                 n_tubes=n_tubes))
    g.add(OpNode("reduce", REDUCTION_KERNEL,
                 n_pairs=n_points, n_points=n_points, n_tubes=n_tubes))
    g.connect("contains", "reduce", n_points)  # 每点 0/1 掩码
    return g


def mc_volume_graph(n_tubes: int, n_samples: int) -> OpGraph:
    """Monte Carlo 路径：采样点成员判定 → 计数归约。"""
    g = OpGraph(name=f"mc_volume(n_tubes={n_tubes}, samples={n_samples})")
    g.add(OpNode("sample", MC_SAMPLE_KERNEL,
                 n_pairs=n_tubes * n_samples, n_points=n_samples,
                 n_tubes=n_tubes))
    g.add(OpNode("reduce", REDUCTION_KERNEL,
                 n_pairs=n_samples, n_points=n_samples, n_tubes=n_tubes))
    g.connect("sample", "reduce", n_samples)
    return g
