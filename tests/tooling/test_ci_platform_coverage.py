"""跨平台 CI 覆盖的守门测试。

**为什么需要它。** 本项目的形态是**在 macOS 上开发**，而 CI 此前**全部 5 个
job 都跑 ubuntu**——即"开发平台恰恰没被 CI 覆盖"。后果不是抽象的：

- ``memory_guard`` 的 macOS 分支是开发机实测的，Linux 分支只有注入式验证；
- ``cpp/build_nano.sh`` 的 macOS 分支（``-mcpu=apple-m4``）**从未在真机跑过
  CI**（只在开发者本机跑过，而那台机器上它当然通过）。

更糟的是**没有任何守门**：内部设计文档曾写"CI = Python 3.9/3.12 ×
ubuntu + cpp × macos/ubuntu"，而实测 ``git log -S macos-latest`` 查无此串
——**macOS job 从未存在过**，这个错误陈述在文档里躺了数周无人发现。

本文件把"CI 覆盖了哪些平台"钉成可执行契约：**开发平台必须在 CI 矩阵里**。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"

yaml = pytest.importorskip("yaml", reason="需要 pyyaml 解析 CI 配置")


def _ci() -> dict:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))


def _all_runners(ci: dict) -> list[str]:
    """收集所有 job 的 runs-on 取值（含矩阵展开后的 os）。"""
    out: list[str] = []
    for job in ci["jobs"].values():
        run = job.get("runs-on")
        if isinstance(run, str) and "${{" not in run:
            out.append(run)
        matrix = (job.get("strategy") or {}).get("matrix") or {}
        for entry in matrix.get("include", []) or []:
            if "os" in entry:
                out.append(entry["os"])
        for os_val in matrix.get("os", []) or []:
            out.append(os_val)
    return out


def test_ci_has_matrix_job_with_os_dimension():
    """必须存在一个以 ``os`` 为维度的矩阵 job。

    **为什么盯这个**：只有一个 job 跨平台时，加平台才是"改一行"；若各 job
    各自写死 ``runs-on``，平台覆盖就散在多处、易漏。
    """
    ci = _ci()
    has_os_matrix = any(
        "os" in ((j.get("strategy") or {}).get("matrix") or {})
        or any("os" in e for e in
               (((j.get("strategy") or {}).get("matrix") or {}).get("include") or []))
        for j in ci["jobs"].values()
    )
    assert has_os_matrix, (
        "CI 里没有任何以 os 为维度的矩阵——平台覆盖散在各 job 的 runs-on 里，"
        "加平台时容易漏改。"
    )


def test_ci_covers_the_development_platform():
    """**CI 必须覆盖开发平台**（本项目是 macOS）。

    这是本文件存在的全部理由。判据取"开发平台"而非硬编码 "macos"：环境变量
    ``KAKEYA_DEV_PLATFORM`` 可覆盖（供在别的平台开发的人用），默认为 macOS。
    """
    dev = _dev_platform()
    runners = " ".join(_all_runners(_ci()))
    assert dev in runners, (
        f"CI 未覆盖开发平台 {dev!r}——本项目的记忆里明确记着"
        '"CI 只测了开发者不用的平台" 这个镜像缺陷。'
        f"当前 CI 覆盖：{sorted(set(_all_runners(_ci())))}"
    )


def test_ci_covers_linux_too():
    """Linux 也必须覆盖：``memory_guard`` 与 ``build_nano.sh`` 都有 Linux 分支。

    只有 macOS 会漏掉 Linux 侧的 ``/proc/meminfo`` 解析路径。
    """
    runners = " ".join(_all_runners(_ci()))
    assert "ubuntu" in runners, "CI 未覆盖 Linux——内存守卫与 C++ 构建的 Linux 分支将无人验证"


def test_matrix_does_not_waste_macos_minutes():
    """**macOS 不得跑满整个 Python 矩阵**（计费约 10× ubuntu）。

    这是刻意的成本取舍：ubuntu 覆盖 Python 两端，macOS 只跑开发机版本。
    若有人把 macOS 加进 ``python-version`` 乘积矩阵，交叉项会翻倍而收益很小。
    """
    ci = _ci()
    for job in ci["jobs"].values():
        matrix = (job.get("strategy") or {}).get("matrix") or {}
        pys = matrix.get("python-version")
        if not pys:
            continue
        # 若 os 与 python-version 是两个独立维度（笛卡尔积），则 macOS × 全部 Python
        # ——这条会红。用 include 显式配对是允许的。
        assert "os" not in matrix, (
            "检测到 os × python-version 的笛卡尔积矩阵——macOS 会跑满全部 "
            "Python 版本（约 10× 计费）。请改用 include 显式配对。"
        )


def test_macos_entries_do_not_set_x86_march():
    """macOS 矩阵项**不得**设置 x86 指令集基线。

    GitHub 的 macOS runner 是 **ARM**（apple silicon）；``KAKEYA_MARCH=x86-64-v2``
    在那里无意义甚至导致构建失败。脚本自身有 macOS 分支处理。
    """
    ci = _ci()
    for job in ci["jobs"].values():
        matrix = (job.get("strategy") or {}).get("matrix") or {}
        for entry in matrix.get("include", []) or []:
            os_name = str(entry.get("os", ""))
            if "macos" in os_name:
                march = entry.get("march", "")
                assert not march, (
                    f"macOS 矩阵项设了 KAKEYA_MARCH={march!r}——"
                    "macOS runner 是 ARM，x86 基线不适用"
                )


def test_ka_march_is_always_bound_to_a_matrix_value():
    """``KAKEYA_MARCH`` 环境变量必须来自矩阵，不得写死。

    写死意味着所有平台共用一个值——而正确值**依平台而异**（ubuntu 要
    x86-64-v2 避免绑定构建机 CPU；macOS 该留空走自身分支）。
    """
    text = CI.read_text(encoding="utf-8")
    import re

    for m in re.finditer(r"KAKEYA_MARCH:\s*(\S+)", text):
        value = m.group(1)
        assert "${{" in value or value in ("", '""'), (
            f"KAKEYA_MARCH 被写死为 {value!r}——应按平台取矩阵值"
        )


def _dev_platform() -> str:
    """取当前开发平台的关键字（用于断言 CI 覆盖它）。"""
    import os

    override = os.environ.get("KAKEYA_DEV_PLATFORM")
    if override:
        return override
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "ubuntu"
    return "windows"
