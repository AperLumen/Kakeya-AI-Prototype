"""拆核并发规划（nano/scheduling/core_split.py，Q4 #14）的守门。

钉住的行为：

- 相同双任务在异构拓扑上，拆核并发优于各自满核串行（E 核低效时尤其）；
- 分配合法性：每任务 ≥1 核、总和 ≤ 总核数；
- 重任务该拿更多核（轻重配对时分配不对称）；
- 单任务退化为"全核独占"且不声称提速；
- 非法输入显式报错（空任务、非正时长、任务数超核数）；
- 串行基线 = 各任务满核预测之和（不许被改口径）。
"""

import pytest

from nano.scheduling.core_split import plan_core_split

# M4 拓扑（本机实测口径）：4P + 6E，E 核效率 ~0.4。
M4 = {"p_cores": 4, "e_cores": 6, "e_core_efficiency": 0.4, "p_core_all_ratio": 0.81}


def test_two_equal_tasks_split_beats_serial():
    plan = plan_core_split([1000.0, 1000.0], **M4)
    assert plan.worth_it
    assert sum(plan.allocation) <= 10
    assert all(n >= 1 for n in plan.allocation)
    assert plan.speedup > 1.1  # 明显收益，不是噪声级别


def test_equal_tasks_get_balanced_split():
    plan = plan_core_split([800.0, 800.0], **M4)
    a, b = plan.allocation
    assert abs(a - b) <= 1  # 相同任务不该分得很偏


def test_heavy_task_gets_more_cores():
    plan = plan_core_split([4000.0, 500.0], **M4)
    assert plan.allocation[0] > plan.allocation[1]


def test_single_task_takes_all_cores_no_fake_speedup():
    plan = plan_core_split([1000.0], **M4)
    assert plan.allocation == (10,)
    assert plan.speedup == pytest.approx(1.0)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="不能为空"):
        plan_core_split([], **M4)
    with pytest.raises(ValueError, match="必须为正"):
        plan_core_split([100.0, -1.0], **M4)
    with pytest.raises(ValueError, match="超过总核数"):
        plan_core_split([1.0] * 12, **M4)  # 12 任务 > 10 核


def test_serial_baseline_is_full_core_sum():
    tasks = [1000.0, 1000.0]
    plan = plan_core_split(tasks, **M4)
    # 串行基线必须等于 2 × 满核单任务预测（口径守门：不许悄悄换定义）。
    from nano.model.chip_model import predict_schedule
    full = predict_schedule(1000.0, **M4).predicted_ms[10]
    assert plan.serial_baseline_ms == pytest.approx(2 * full, rel=1e-6)
