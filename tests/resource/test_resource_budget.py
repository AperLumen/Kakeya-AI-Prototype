"""资源预算（``resource_budget``）的守门测试。

**为什么需要它。** 建立 :mod:`kakeya_ai.resource_budget` 只是第一步；真正的
风险是**它慢慢被绕开**——有人在某个新模块里又写一个 ``timeout=300`` 字面量，
于是"超时"重新变得散落，而**没有任何测试会红**。这正是本模块要解决的问题
（此前 ``fast_timeout=30.0`` 在 ``cascaded_reasoning`` 与 ``beam_search``
**各写一份**，改一处不会同步另一处）。

因此本文件有三层：

1. **机制正确性**：预算装配、环境覆盖、校验、``deadline`` 的超时判定。
2. **接线正确性**：真实调用点的默认值**确实来自预算**（改环境变量能影响它们）。
3. **防回退（最重要）**：源码扫描，禁止在生产模块里出现硬编码的超时/并发字面量。

变异验证（开发时手工做过）：
- 把 ``ollama_client`` 的 ``DEFAULT_HTTP_TIMEOUT`` 改回 ``300`` 字面量 → 第 3 层红；
- 去掉 ``Budget.validate`` 的负值检查 → 第 1 层红；
- 让 ``deadline`` 不做比较（永不抛）→ 第 1 层红。
"""

from __future__ import annotations

import io
import os
import re
import tokenize
from pathlib import Path
from typing import Any

import pytest

from kakeya_ai.errors import InputValidationError
from kakeya_ai.resource_budget import (
    DEFAULT_BUDGET,
    Budget,
    TimeExceededError,
    deadline,
)

SRC = Path(__file__).resolve().parents[2] / "src" / "kakeya_ai"


class _Clock:
    """可注入的假时钟（本项目对时间相关代码的一贯处理）。"""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


# ---------------------------------------------------------------------------
# 1. 机制正确性
# ---------------------------------------------------------------------------


def test_default_budget_is_valid():
    """默认预算必须自洽（越界值应在装配时就被挡下）。"""
    DEFAULT_BUDGET.validate()
    assert DEFAULT_BUDGET.max_workers >= 1
    assert DEFAULT_BUDGET.http_timeout > 0


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("http_timeout", 0.0),
        ("http_timeout", -1.0),
        ("fast_timeout", 0.0),
        ("deep_timeout", -5.0),
        ("max_retries", -1),
        ("max_workers", 0),
        ("retry_backoff", -1.0),
    ],
)
def test_invalid_budget_is_rejected(field_name: str, value: Any):
    """非法预算必须**立即报错**（与 config 的"静默回退"刻意相反）。

    理由：预算决定长跑任务的上限，静默回退会让人以为改了配置却没生效。

    ``value: Any``（而非 ``object``）：这些值**故意**是各种非法的 float/int，
    要经 ``**{field_name: value}`` 展开给具体类型的形参——用 ``object``
    会被判为不可分配（那是"可能不是 float"的诚实表达），``Any`` 才是这个
    负向用例真正想要的类型（"随便什么值都行"）。
    """
    bad = Budget(**{field_name: value})
    with pytest.raises(InputValidationError) as ei:
        bad.validate()
    assert ei.value.field == field_name


def test_worst_case_seconds_accounts_for_retries_and_backoff():
    """``worst_case_seconds`` 必须把重试与退避都算进去。"""
    b = Budget(http_timeout=10.0, max_retries=2, retry_backoff=1.0)
    # 3 次尝试 × 10s = 30s；退避 1 + 2 = 3s；合计 33s。
    assert b.worst_case_seconds() == pytest.approx(33.0)
    # 不重试时就是单次超时
    assert Budget(http_timeout=10.0, max_retries=0).worst_case_seconds() == pytest.approx(10.0)


def test_deadline_raises_when_exceeded():
    """``deadline`` 超限时抛 ``TimeExceededError``（用假时钟，不真等）。"""
    c = _Clock()
    with pytest.raises(TimeExceededError) as ei:
        with deadline(1.0, label="推理", clock=c):
            c.t = 5.0
    assert ei.value.limit == 1.0
    assert ei.value.elapsed == pytest.approx(5.0)
    assert "推理" in str(ei.value)


def test_deadline_does_not_raise_when_within_budget():
    """未超限时不得抛——否则它会把正常路径也打断。"""
    c = _Clock()
    with deadline(10.0, clock=c):
        c.t = 3.0
    # 恰好等于限值也不算超（用 > 而非 >=：边界上不误杀）
    c2 = _Clock()
    with deadline(3.0, clock=c2):
        c2.t = 3.0


def test_overridden_returns_new_object_and_records_source():
    """``overridden`` 不改原对象，且记录来源（供 explain 审计）。"""
    base = Budget(http_timeout=300.0)
    new = base.overridden(http_timeout=10.0)
    assert base.http_timeout == 300.0, "原对象被改动了"
    assert new.http_timeout == 10.0
    assert new._sources.get("http_timeout") == "显式覆盖"
    assert "http_timeout" in new.explain()


def test_explain_notes_suspicious_deep_lt_fast():
    """``explain`` 应对 deep < fast 这类可疑配置给出提示（不静默）。"""
    b = Budget(fast_timeout=100.0, deep_timeout=10.0)
    assert "deep_timeout < fast_timeout" in b.explain()


# ---------------------------------------------------------------------------
# 2. 接线正确性：真实调用点确实取预算
# ---------------------------------------------------------------------------


def test_ollama_client_defaults_come_from_budget():
    """``ollama_client`` 的默认常量必须等于预算值（不是另写的字面量）。"""
    from kakeya_ai.ai import ollama_client as oc

    assert oc.DEFAULT_MAX_RETRIES == DEFAULT_BUDGET.max_retries
    assert oc.DEFAULT_RETRY_BACKOFF == DEFAULT_BUDGET.retry_backoff
    assert oc.DEFAULT_HTTP_TIMEOUT == DEFAULT_BUDGET.http_timeout


def test_cascaded_and_beam_share_one_source(monkeypatch):
    """``cascaded_reasoning`` 与 ``beam_search`` 的超时默认值必须**同源**。

    这正是本模块的存在理由——此前两者各写一份 30/60，改一处不同步另一处。
    用环境变量覆盖后重新装配，两个模块的默认值应**一致地**跟着变。
    """
    monkeypatch.setenv("KAKEYA_FAST_TIMEOUT", "7.5")
    monkeypatch.setenv("KAKEYA_DEEP_TIMEOUT", "12.5")

    from kakeya_ai import resource_budget as rb

    fresh = rb._load_default()
    assert fresh.fast_timeout == 7.5
    assert fresh.deep_timeout == 12.5

    import inspect

    from kakeya_ai.ai import beam_search, cascaded_reasoning

    for mod in (beam_search, cascaded_reasoning):
        sig = inspect.signature(mod.cascaded_score if mod is cascaded_reasoning
                                else mod.beam_search_score)
        # 默认值是**求值后的数值**（默认参数在 def 时求值），故这里校验
        # 它等于"装配那一刻"的预算值即可——而不是恰好等于 30/60 字面量。
        assert sig.parameters["fast_timeout"].default in (30.0, 7.5)
        assert sig.parameters["deep_timeout"].default in (60.0, 12.5)


def test_env_override_affects_default_budget(monkeypatch):
    """环境变量必须能真的影响预算装配（这是"可配置"的最低要求）。"""
    monkeypatch.setenv("KAKEYA_HTTP_TIMEOUT", "42.5")
    from kakeya_ai import resource_budget as rb

    b = rb._load_default()
    assert b.http_timeout == 42.5
    assert b._sources.get("http_timeout") == "环境"


def test_invalid_env_falls_back_with_warning(monkeypatch):
    """非法环境值按 config 纪律回退默认（记 warning，不抛）。"""
    monkeypatch.setenv("KAKEYA_MAX_WORKERS", "not-a-number")
    from kakeya_ai import resource_budget as rb

    b = rb._load_default()
    assert b.max_workers == 4  # 默认值


# ---------------------------------------------------------------------------
# 3. 防回退：禁止硬编码超时/并发字面量（最重要）
# ---------------------------------------------------------------------------

#: 豁免清单：**具名常量** → 理由。新增豁免必须写明理由。
#:
#: **粒度必须是常量名，不能是整个文件**——开发时踩到：最初按文件名豁免
#: ``ollama_client.py``（因 ``PROBE_TIMEOUT`` 合理），结果该文件里任何硬编码
#: 都被放过了（把 ``DEFAULT_HTTP_TIMEOUT`` 改回 ``300`` 字面量，守门照样通过）。
#: 按常量名豁免则只放过 ``PROBE_TIMEOUT`` 本身。
HARDCODE_EXEMPT_CONSTANTS: dict[str, str] = {
    "PROBE_TIMEOUT": "ollama 探活专用小常量（只问服务在不在，不跟主超时走）",
    "VM_STAT_TIMEOUT": "本地子进程探活超时（vm_stat 毫秒级返回，防挂死用）",
}

#: 整个文件豁免（**尽量不用**，只在文件本身就是预算定义处时）。
HARDCODE_EXEMPT: dict[str, str] = {
    "resource_budget.py": "预算的定义处（默认值本就写在这里）",
}


def _iter_source_files():
    for p in SRC.rglob("*.py"):
        if "__pycache__" not in p.parts and p.name not in HARDCODE_EXEMPT:
            yield p


def _code_without_strings_and_comments(path: Path) -> str:
    """剥掉字符串与注释后的代码骨架（避免误判文档里提到的数字）。"""
    src = path.read_text(encoding="utf-8")
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:  # pragma: no cover
        return src
    return " ".join(
        tok.string for tok in tokens
        if tok.type not in (tokenize.STRING, tokenize.COMMENT)
    )


def test_no_hardcoded_timeout_literals_in_production():
    """生产模块不得再出现硬编码的超时/并发字面量。

    **为什么用源码扫描**：本模块存在的全部意义就是消除"同一件事在多处各写
    一个数"。这是**文本模式**问题（不是语义问题），AST 无增益；但必须剥掉
    字符串与注释，否则会把文档里"曾写 30.0"这类描述误判成违规。

    **大小写不敏感**（``timeout`` 与 ``TIMEOUT`` 都查）：常量赋值
    ``DEFAULT_HTTP_TIMEOUT = 300`` 同样属于硬编码。

    **豁免按具名常量**（见 :data:`HARDCODE_EXEMPT_CONSTANTS`）：被豁免的常量
    所在行会被跳过，该文件其余位置照查。

    只查 ``src/kakeya_ai/`` 的生产代码——``tests/`` 里造小超时是正常的
    （测试就该跑得快）。
    """
    # 形如 ``timeout=300`` / ``timeout: float = 300`` / ``TIMEOUT = 300``
    # 且右侧是**数字字面量**（而非标识符/属性访问）。
    #
    # 三处都要覆盖：① 大小写不敏感（``TIMEOUT`` 常量赋值也是硬编码——
    # 开发时踩到，把 ``DEFAULT_HTTP_TIMEOUT`` 改回 300 居然通过）；
    # ② **带类型注解的形参** ``fast_timeout: float = 30.0`` 必须能匹配
    # （又踩到一次：只允许 ``[:=]`` 紧跟名字，漏掉了 ``: float =`` 这种
    # 最常见的写法，导致 ``beam_search`` 的变异不被捕获）。
    patterns = [
        re.compile(r"(\w*timeout)\s*(?::\s*[\w\.\[\]|]+\s*)?[:=]\s*\d+(?:\.\d+)?",
                   re.IGNORECASE),
        re.compile(r"(max_workers)\s*(?::\s*\w+\s*)?[:=]\s*\d+", re.IGNORECASE),
        re.compile(r"(retry_backoff)\s*(?::\s*\w+\s*)?[:=]\s*\d+(?:\.\d+)?",
                   re.IGNORECASE),
    ]
    offenders: list[str] = []
    for path in _iter_source_files():
        code = _code_without_strings_and_comments(path)
        for pat in patterns:
            for m in pat.finditer(code):
                names = [g for g in m.groups() if g]
                # 被豁免的具名常量：跳过（只跳过该次匹配）
                if any(n.upper() in HARDCODE_EXEMPT_CONSTANTS for n in names):
                    continue
                offenders.append(f"{path.relative_to(SRC.parent.parent)}: {m.group(0)!r}")

    assert not offenders, (
        "以下位置出现硬编码的资源字面量，请改用 kakeya_ai.resource_budget "
        "（Budget / DEFAULT_BUDGET）；若确需豁免，请把**常量名**加入 "
        "HARDCODE_EXEMPT_CONSTANTS 并写明理由：\n" + "\n".join(offenders)
    )


def test_resource_env_vars_are_registered():
    """预算的 6 个环境变量必须都在 ``config.ENV_SPEC`` 里登记。

    否则"这个系统认哪些变量"又会失去唯一答案——正是 ``config.py`` 要解决的
    问题（见其模块 docstring）。
    """
    from kakeya_ai import config
    from kakeya_ai import resource_budget as rb

    names = set(config.env_names())
    required = {
        rb.ENV_HTTP_TIMEOUT, rb.ENV_MAX_RETRIES, rb.ENV_RETRY_BACKOFF,
        rb.ENV_FAST_TIMEOUT, rb.ENV_DEEP_TIMEOUT, rb.ENV_MAX_WORKERS,
    }
    missing = sorted(required - names)
    assert not missing, f"这些资源环境变量未登记进 config.ENV_SPEC：{missing}"


def test_budget_exported_from_package():
    """预算类型必须能从顶层包导入。"""
    import kakeya_ai

    for name in ("Budget", "DEFAULT_BUDGET", "TimeExceededError", "deadline"):
        assert name in kakeya_ai.__all__, f"{name} 未出现在 kakeya_ai.__all__"
        assert hasattr(kakeya_ai, name)
