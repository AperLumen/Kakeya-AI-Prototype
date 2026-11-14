"""算子图抽象（nano/model/op_graph.py，Q4 #15）的守门。

钉住的行为：

- MC 与 voxel 两条真实路径入图（节点/边结构正确）；
- 求值确定性且屋顶线口径正确（瓶颈 = max(compute, memory) 的节点）；
- voxel 主节点是算力受限（与 chip_model 单 kernel 分析一致），归约是
  带宽受限——两条结论反了就是接线错了；
- 图结构校验（重名节点、悬空边）显式报错；
- 总计 = 逐节点之和（串行口径不许悄悄换）。
"""

import pytest

from nano.model.chip_model import ChipSpec
from nano.model.op_graph import (
    OpGraph,
    OpNode,
    TUBE_GEOMETRY_KERNEL,
    evaluate_graph,
    mc_volume_graph,
    voxel_volume_graph,
)

M4 = ChipSpec(
    name="M4", peak_flops_gflops=3584.0, peak_bandwidth_gbytes=120.0,
    n_processing_elements=10,
)


def test_two_paths_build():
    vg = voxel_volume_graph(n_tubes=16, grid_n=16)
    mg = mc_volume_graph(n_tubes=16, n_samples=4096)
    assert [n.name for n in vg.nodes] == ["contains", "reduce"]
    assert [n.name for n in mg.nodes] == ["sample", "reduce"]
    assert len(vg.edges) == len(mg.edges) == 1


def test_voxel_main_node_is_compute_bound():
    r = evaluate_graph(voxel_volume_graph(n_tubes=16, grid_n=16), M4)
    contains = next(n for n in r["per_node"] if n["name"] == "contains")
    assert contains["bound"] == "compute"
    reduce_ = next(n for n in r["per_node"] if n["name"] == "reduce")
    assert reduce_["bound"] == "memory"
    assert r["bottleneck"] == "contains"


def test_total_is_serial_sum():
    r = evaluate_graph(voxel_volume_graph(n_tubes=8, grid_n=8), M4)
    assert r["total_ns"] == pytest.approx(
        sum(n["time_ns"] for n in r["per_node"]), rel=1e-12)


def test_deterministic():
    g = mc_volume_graph(n_tubes=16, n_samples=1024)
    assert evaluate_graph(g, M4) == evaluate_graph(g, M4)


def test_graph_validation():
    g = OpGraph("t")
    g.add(OpNode("a", TUBE_GEOMETRY_KERNEL, 10, 10, 2))
    with pytest.raises(ValueError, match="重名"):
        g.add(OpNode("a", TUBE_GEOMETRY_KERNEL, 10, 10, 2))
    with pytest.raises(ValueError, match="未定义节点"):
        g.connect("a", "ghost", 100)
