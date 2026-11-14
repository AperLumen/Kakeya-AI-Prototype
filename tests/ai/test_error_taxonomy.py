"""错误分类体系的守门测试。

**为什么需要它。** 建 ``errors.py`` 这类"分层"很容易退化成装饰品：类定义了，
但没人用；或者用了一阵又被改回裸 ``ValueError``——而**没有任何测试会红**。
本文件把三件事钉成可执行契约：

1. **结构**：继承关系与实际异常树一致（``except KakeyaError`` 必须能全覆盖）。
2. **语义分派**：关键路径真的抛**正确的类型**（用真实调用触发，不是构造假对象）。
   这是最有判别力的一条——它同时验证了"类型定义对"与"抛出点迁移了"。
3. **防回退**：关键模块里不再出现裸 ``ValueError``（见最后一条，带豁免清单）。

**为什么第 2 条要用真实调用而不是直接 ``raise``**：直接 ``raise
InputValidationError(...)`` 只能证明"类能被实例化"，证明不了"迁移做完了"。
必须让被测函数**自己**抛出来，才能同时钉住"抛出点"。

变异验证（开发时手工做过）：把 ``moves.py`` 的 ``ContractViolationError`` 改回
``ValueError`` → 第 2 条红；把 ``InputValidationError`` 的基类从 ``ValueError``
拿掉 → 第 1 条红。
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

import pytest

import kakeya_ai
from kakeya_ai import errors as E
from kakeya_ai.memory_guard import InsufficientMemoryError

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "kakeya_ai"

#: 全部自定义异常类（新增时应同步这里——这是"清单即契约"）。
ALL_ERRORS = [
    E.KakeyaError,
    E.InputValidationError,
    E.TraceFormatError,
    E.ContractViolationError,
    E.BackendUnavailableError,
    E.NumericalLimitError,
    E.NotImplementedYetError,
    InsufficientMemoryError,
]


# ---------------------------------------------------------------------------
# 1. 结构契约
# ---------------------------------------------------------------------------


def test_all_errors_inherit_from_kakeya_error():
    """每个自定义异常都必须继承 ``KakeyaError``。

    否则 ``except KakeyaError`` 会漏掉它——而调用方的惯用写法正是
    "一个 except 兜住本包的全部错误"。
    """
    for cls in ALL_ERRORS:
        assert issubclass(cls, E.KakeyaError), (
            f"{cls.__name__} 未继承 KakeyaError，except KakeyaError 会漏掉它"
        )


@pytest.mark.parametrize(
    ("cls", "builtin"),
    [
        # 向后兼容契约：每个子类都必须同时继承一个内置异常，
        # 使既有 `except ValueError` / `except RuntimeError` 不失效。
        (E.InputValidationError, ValueError),
        (E.TraceFormatError, ValueError),
        (E.ContractViolationError, ValueError),
        (E.BackendUnavailableError, RuntimeError),
        (E.NumericalLimitError, RuntimeError),
        (E.NotImplementedYetError, NotImplementedError),
        (InsufficientMemoryError, RuntimeError),
    ],
)
def test_backward_compat_with_builtin(cls, builtin):
    """每个子类都仍是一个内置异常——这是"迁移可渐进"的前提。

    若哪天真要断开与内置类型的继承，**必须**同时确认全仓没有既存的
    ``except <builtin>`` 依赖它（本项目多处如此，故当前刻意保留）。
    """
    assert issubclass(cls, builtin), (
        f"{cls.__name__} 不再是 {builtin.__name__} 子类，既有的 "
        f"`except {builtin.__name__}` 会漏掉它"
    )


def test_catch_all_works():
    """``except KakeyaError`` 必须能捕获每一个自定义异常。"""
    for cls in ALL_ERRORS:
        with pytest.raises(E.KakeyaError):
            if cls is InsufficientMemoryError:
                raise cls(1, 2, 3)  # 该异常要求三个位置参数
            raise cls("测试消息")


def test_not_implemented_is_not_plain_notimplementederror():
    """``NotImplementedYetError`` 可与内置 ``NotImplementedError`` 区分。

    内置的那个也用于"抽象方法未覆写"，语义不同（那是子类忘实现，本类是
    "本版本就没有"）。两者若无法区分，捕获时就分不清该改代码还是查文档。
    """
    assert issubclass(E.NotImplementedYetError, NotImplementedError)
    # 反向：一个裸 NotImplementedError 不应被当成我们的类型
    plain = NotImplementedError("x")
    assert not isinstance(plain, E.NotImplementedYetError)


# ---------------------------------------------------------------------------
# 2. 语义分派：真实调用触发正确类型
# ---------------------------------------------------------------------------


def test_trace_format_error_on_bad_file(tmp_path):
    """坏 ``.kpt`` 文件必须抛 ``TraceFormatError``（而非裸 ValueError）。

    同时验证它仍是 ``ValueError``——``kpt_cli`` 靠 ``except ValueError``
    打印"格式错误"，这条兼容性不能丢。
    """
    from kakeya_ai.trace_format import TraceFormatError, read_trace

    bad = tmp_path / "bad.kpt"
    bad.write_text("NOT-A-KPT-HEADER\n", encoding="utf-8")
    with pytest.raises(TraceFormatError) as ei:
        read_trace(bad)
    assert isinstance(ei.value, ValueError), "TraceFormatError 丢了 ValueError 兼容性"


def test_contract_violation_when_tubes_missing():
    """``error_mode='actual'`` 缺 ``tubes`` 必须抛 ``ContractViolationError``。

    这是本项目反复强调的一条：**"声称用某模式"必须连带要求其前提参数**，
    否则会静默退回 proxy、让 ``condition_kind`` 说谎。异常类型要能体现
    "是漏传了必需依赖"，而不是"参数值非法"。
    """
    from kakeya_ai.errors import ContractViolationError
    from kakeya_ai.grains import GrainsDecomposition
    from kakeya_ai.moves import move3_wider_grains

    empty = GrainsDecomposition([])
    with pytest.raises(ContractViolationError) as ei:
        move3_wider_grains(empty, 0.01, error_mode="actual", tubes=None, delta=0.1)
    # 结构化字段应指出缺的是谁
    assert "tubes" in ei.value.missing


def test_input_validation_on_bad_error_mode():
    """非法的 ``error_mode`` 取值抛 ``InputValidationError``（与"漏传"区分）。"""
    from kakeya_ai.errors import InputValidationError
    from kakeya_ai.grains import GrainsDecomposition
    from kakeya_ai.moves import move3_wider_grains

    empty = GrainsDecomposition([])
    with pytest.raises(InputValidationError):
        move3_wider_grains(empty, 0.01, error_mode="bogus", delta=0.1)


def test_nonzero_direction_is_input_validation():
    """零方向向量是**输入非法**（不是格式问题），应为 ``InputValidationError``。"""
    import numpy as np

    from kakeya_ai.errors import InputValidationError
    from kakeya_ai.primitives import Direction

    with pytest.raises(InputValidationError):
        Direction(np.array([0.0, 0.0, 0.0]))


def test_missing_bin_metadata_is_contract_violation():
    """prism 缺 ``bin_tube_count`` 元数据 → ``ContractViolationError``。

    **这是一条修正过的分类**：开发时最初把它归成 ``NumericalLimitError``
    （"算得出但达不到"），跑回归才发现分类错了——元数据缺失是**前提不满足**
    （数据没带必需字段），不是"算过但不够"。两者处置不同：前者要补数据/补
    fallback，后者要放宽目标或提高精度。这条测试把正确分类钉住。

    同时它仍是 ``ValueError``（历史兼容，既有测试靠这个断言）。
    """
    from kakeya_ai.errors import ContractViolationError
    from kakeya_ai.grains import GrainsDecomposition, Prism, guth_grains_skeleton
    from kakeya_ai.moves import move1_longer_grains

    # ``test_moves`` 自 2026-11-06 起在 ``tests/orchestration/`` 子目录下，
    # 故导入路径带上分组名（``tests/`` 各子目录都有 ``__init__.py``，是正式包）。
    from tests.orchestration.test_moves import make_fan_tubes

    tubes = make_fan_tubes(8)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    # 抹掉 bin 元数据，模拟"合并后丢失"。
    stripped = GrainsDecomposition(
        prisms=[
            Prism(bounds=p.bounds, tube_indices=p.tube_indices, density=p.density,
                  local_kt_error=p.local_kt_error)
            for p in grains.prisms
        ]
    )
    with pytest.raises(ContractViolationError) as ei:
        move1_longer_grains(stripped, rho=0.25, delta=0.05, zeta=0.01,
                            tube_ratio=None, max_iterations=2)
    assert isinstance(ei.value, ValueError), "ContractViolationError 丢了 ValueError 兼容性"


def test_insufficient_memory_is_backend_unavailable():
    """内存不足属"环境/资源不可用"——可降级/重试，不是改代码。"""
    from kakeya_ai.errors import BackendUnavailableError

    assert issubclass(InsufficientMemoryError, BackendUnavailableError)
    exc = InsufficientMemoryError(1, 2, 3)
    assert exc.backend == "system-memory"


def test_numerical_limit_carries_shortfall():
    """``NumericalLimitError`` 的运行时可算出 ``shortfall``（结构化字段可用）。

    这条钉住"结构化字段优于一句消息"的设计：调用方能拿到数值继续处置，
    不必解析消息文本。
    """
    exc = E.NumericalLimitError("达不到", target=1.0, actual=1.116)
    assert exc.shortfall == pytest.approx(0.116)
    assert E.NumericalLimitError("缺值").shortfall is None


# ---------------------------------------------------------------------------
# 3. 防回退：关键模块不再出现裸 ValueError
# ---------------------------------------------------------------------------

#: 豁免清单：这些模块里**允许**出现 ``raise ValueError``，各有明确原因。
#: 新增豁免必须在注释里写明理由（否则就是让守门变松）。
VALUE_ERROR_EXEMPT: dict[str, str] = {
    # errors.py 是定义地——子类继承 ValueError，源码里必然出现该词。
    "errors.py": "异常体系的定义文件本身",
}


def _iter_source_files():
    for p in SRC.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        yield p


def test_no_bare_valueerror_in_core_modules():
    """核心计算模块不应再出现 ``raise ValueError``（应抛语义类型）。

    **为什么用源码扫描而不用 AST**：这里要防的是"有人新增一处裸 ValueError"，
    是个**文本模式**问题；AST 判据对本规则没有增益（``raise`` 语句的形状本就
    简单）。但为免把**注释/文档字符串**里提到的 "ValueError" 误判，先剥离字符串
    与注释再匹配——本项目在"文本 vs 代码"上栽过跟头（见项目记忆：
    文本匹配分不清"代码"与"描述代码的字符串"）。

    豁免见 :data:`VALUE_ERROR_EXEMPT`。
    """
    offenders: list[str] = []
    for path in _iter_source_files():
        if path.name in VALUE_ERROR_EXEMPT:
            continue
        src = path.read_text(encoding="utf-8")
        # 去掉字符串字面量与注释，只留代码骨架，避免误判文档里提到的词。
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
        except tokenize.TokenError:  # pragma: no cover - 语法错误另有测试
            continue
        code = " ".join(
            tok.string
            for tok in tokens
            if tok.type not in (tokenize.STRING, tokenize.COMMENT)
        )
        for m in re.finditer(r"raise\s+ValueError\b", code):
            line_no = code[: m.start()].count(" ") + 1  # 相对位置，仅作提示
            offenders.append(f"{path.relative_to(REPO_ROOT)}（令牌序 ~{line_no}）")

    assert not offenders, (
        "以下位置仍在抛裸 ValueError，请改用 kakeya_ai.errors 里的语义类型"
        "（InputValidationError / TraceFormatError / ContractViolationError / …）；"
        f"若确需豁免，请加入 VALUE_ERROR_EXEMPT 并写明理由：{offenders}"
    )


def test_errors_module_is_exported_from_package():
    """异常类型必须能从顶层包导入——否则调用方无法按类型分派。

    **必须查包级 ``kakeya_ai.__all__``，而不是 ``errors.__all__``**：
    前者才是"能否 ``from kakeya_ai import X``"的真源。开发时踩到——
    只查 ``errors.__all__`` 时，把条目从**包级** ``__all__`` 删掉，
    守门照样通过（变异未捕获，实为装饰品）。
    """
    for name in E.__all__:
        assert name in kakeya_ai.__all__, (
            f"{name} 未出现在 kakeya_ai.__all__ 里——"
            f"调用方无法 `from kakeya_ai import {name}`"
        )
        assert hasattr(kakeya_ai, name), f"kakeya_ai 缺属性 {name}"
    # InsufficientMemoryError 也属这棵树，一并要求可导入。
    assert "InsufficientMemoryError" in kakeya_ai.__all__
    assert hasattr(kakeya_ai, "InsufficientMemoryError")
