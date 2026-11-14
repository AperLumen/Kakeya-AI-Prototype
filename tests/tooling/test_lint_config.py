"""静态检查配置的守门测试。

**为什么需要。** ``# noqa`` 指令只有在其对应规则**已启用**时才有意义。
若规则不在 ``select`` 里，``noqa`` 就是空转的——它不报错、不生效，却让
读者以为"这里已被静态检查豁免"。

这不是假想的风险，而是实际发生过：为解释"为何刻意保留某写法"而加上的
``# noqa: PLW2901``、``# noqa: PLR0124``、``# noqa: BLE001`` 全部无效，
因为本项目 ``select`` 只开 ``F`` 与 ``E9``。RUF100 一次性报出 40 处，
其中核心包 5 处。

**修法不是简单地删 noqa**：那会丢掉"这里刻意如此"的信息。正确做法是保留
解释性注释、去掉无效指令（见本次改动）。本测试则防止它再次发生。

**不是 ``slow``**（2026-09-30 按实测修正）。此处曾标 ``slow``，理由是
"调用 ruff 需数秒"——但 ruff 是**已编译的 Rust 二进制**，实测本文件 3 条
用例共 **0.07 秒**，那个假设从未被验证。

去掉标记的实质理由不只是省时间：**本文件正是“改名漏改”的探针**。
2026-09-30 全量测试暴露的 6 处 ``F821 Undefined name``（``ollama_client`` /
``rag`` 改名后残留的旧引用）就是靠 ``ruff check .`` 抓到的——若默认跳过，
那批缺陷会继续潜伏。守门型用例的价值就在"每次回归都跑"。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ruff_json(*args: str) -> list[dict]:
    """调用 ruff 并返回 JSON 诊断列表。

    ruff 是 dev 依赖（pyproject 的 dev extra）；未安装时跳过而不是失败——
    本测试守的是"静态检查配置"，缺了检查器本身没有可守的对象。
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--output-format", "json", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("ruff 不可用")
    # ruff 有诊断时退出码非 0，那是正常路径，故不看 returncode。
    if not proc.stdout.strip():
        pytest.skip(f"ruff 无输出（stderr: {proc.stderr[:200]}）")
    return json.loads(proc.stdout)


def test_no_unused_noqa_in_repo():
    """源码树不得有**无效的** ``# noqa`` 指令（2026-10-09 扩范围 + 修正判据）。

    无效指令会误导读者以为"这里已豁免静态检查"，而实际上检查从没管过它。
    此前只约束 ``src/``；现覆盖全部 Python 源码树（含 ``scripts/``、``tests/`` 与
    ``nano/`` 这批独立子系统）。

    **判据用 ``--extend-select RUF100``（关键）**：``--select RUF100`` 会**覆盖**
    项目 select，于是把**启用规则**（如 ``C4``）的 noqa 也误判为"未使用"并删除——
    实测这么干会删掉**有效**的 ``# noqa: C408``，使 ``ruff check .`` 立刻报 C408。
    ``--extend-select`` 是在项目规则之上**追加** RUF100，只有真正空转的指令才被
    报出。这是本守门的核心，别改回 ``--select``。
    """
    findings: list = []
    for tree in ("src", "scripts", "tests", "nano"):
        findings.extend(_ruff_json("--extend-select", "RUF100", f"{tree}/"))
    if not findings:
        return
    lines = [
        f"  {f['filename'].replace(str(REPO_ROOT) + '/', '')}:"
        f"{f['location']['row']} — {f['message']}"
        for f in findings
    ]
    pytest.fail(
        "存在无效的 # noqa 指令（对应规则未启用）：\n"
        + "\n".join(lines)
        + "\n修法：删掉 noqa，保留解释性注释说明'为何刻意如此'。"
    )


#: **刻意保留的"未使用"导入**：可用性探测。
#:
#: 形如 ``try: import numba; return "numba_cpu"`` 的写法里，名字**故意不引用**——
#: 能 import 就说明装了。这类 F401 不能删（删了就毁掉探测），也不能加 抑制指令
#: （F401 在项目 ignore 列表里，noqa 会变成空转指令而被 RUF100 抓住）。
#: 键是 ``(相对路径, 符号名)``；**不要在别处顺手加**——每一项都得说清"为什么刻意"。
DEAD_IMPORT_ALLOWLIST: dict[tuple[str, str], str] = {}


def test_no_dead_imports_outside_allowlist():
    """全库不得有**死导入**（F401），除非在 ``DEAD_IMPORT_ALLOWLIST`` 里说清理由。

    **为什么这条值得存在**：``pyproject.toml`` 的 ruff ``ignore`` 列表里含 ``F401``
    （为了让上面那种探测写法不被报错），副作用是**死导入全库永久静默**——实测
    2026-10-11 一次扫描发现 **9 处**，其中 `scripts/geometry/stress_test.py` 的
    ``import nano_kernel as nk`` 还是**必然崩溃**的（cpp/ 里的产物名是
    ``_nano_kernel...so``，见 ``cpp_loader.py`` 的命名约定），把"跑压测"变成
    ``ModuleNotFoundError``。没有守门时，这类问题只会继续积累，且只在真正跑那个
    脚本时才暴露。

    **判据用 ``--select F401``**：它**覆盖**项目 select，故能把被 ignore 的 F401
    报出来（``--extend-select`` 做不到——ignore 优先级更高）。与
    ``test_no_unused_noqa_in_repo`` 相反，那里必须用 ``--extend-select``，原因见其
    docstring；两条守门的取舍方向不同，别互相"统一"。
    """
    findings = _ruff_json(
        "--select", "F401",
        "src/", "scripts/", "tests/", "nano/",
    )
    offenders = []
    for f in findings:
        rel = f["filename"].replace(str(REPO_ROOT) + "/", "")
        # 符号名从 message 的第一个反引号对里取（ruff 的固定格式：`` `x` imported but unused``）
        msg = f["message"]
        symbol = msg.split("`")[1] if "`" in msg else "?"
        if (rel, symbol) in DEAD_IMPORT_ALLOWLIST:
            continue
        offenders.append(f"  {rel}:{f['location']['row']} — 未使用的导入 `{symbol}`")
    if offenders:
        pytest.fail(
            "发现死导入（F401）：\n" + "\n".join(sorted(set(offenders)))
            + "\n修法：删掉它。若确属**可用性探测**（import 只为判断能不能 import），"
            "请把 (文件, 符号) 加进 DEAD_IMPORT_ALLOWLIST 并写明理由——"
            "**不要**加抑制指令（F401 已在 ignore 列表，那样会变成空转声明）。"
        )


def test_ruff_config_is_loadable():
    """ruff 能读到配置且默认检查通过——防止配置写坏导致检查静默失效。"""
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"`ruff check .` 未通过：\n{proc.stdout}\n{proc.stderr}"
    )
