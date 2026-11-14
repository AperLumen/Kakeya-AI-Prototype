"""导出契约测试：``__all__`` 声明的名字必须真的存在。

**为什么需要这组测试。** ``__all__`` 是包对外的公开承诺，但 Python 不会
校验它：写了不存在的名字，平时一切正常，直到有人执行
``from kakeya_ai.ai import *`` 才抛 ``AttributeError`` 崩溃。

这正是实际发生过的事：``format_branch_scoring_prefix`` 与
``format_branch_scoring_dynamic`` 在 ``kakeya_ai.ai.__all__`` 里躺了很久，
但 ``from .prompts import (...)`` 从未导入它们——任何用星号导入的代码
（包括某些交互式探索、notebook、第三方集成）一跑就崩。

这类缺陷的共性是**静态检查与常规测试都看不见**：不 import * 就不会暴露。
所以用一组显式的契约测试钉住，而不是依赖偶然发现。
"""

import pytest

import kakeya_ai
import kakeya_ai.ai


#: 需要检查导出契约的包（顶层 + 主要子包）。
PACKAGES = [
    pytest.param(kakeya_ai, id="kakeya_ai"),
    pytest.param(kakeya_ai.ai, id="kakeya_ai.ai"),
]


@pytest.mark.parametrize("package", PACKAGES)
def test_all_names_exist(package):
    """``__all__`` 里每个名字都必须是模块的真实属性。"""
    dangling = [n for n in package.__all__ if not hasattr(package, n)]
    assert not dangling, (
        f"{package.__name__}.__all__ 声明了不存在的名字：{dangling}。"
        "这会让 `from ... import *` 抛 AttributeError。"
    )


@pytest.mark.parametrize("package", PACKAGES)
def test_all_has_no_duplicates(package):
    """``__all__`` 不应有重复项（重复通常意味着合并时出错）。"""
    seen = set()
    dupes = [n for n in package.__all__ if n in seen or seen.add(n)]
    assert not dupes, f"{package.__name__}.__all__ 有重复项：{dupes}"


@pytest.mark.parametrize("package", PACKAGES)
def test_star_import_works(package):
    """``from <pkg> import *`` 必须能执行——这是导出契约的最终检验。

    用 exec 在受限命名空间里执行，避免污染本模块。
    """
    namespace: dict = {}
    exec(f"from {package.__name__} import *", namespace)
    # 每个 __all__ 名字都应落进命名空间
    missing = [n for n in package.__all__ if n not in namespace]
    assert not missing, (
        f"`from {package.__name__} import *` 未带入：{missing}"
    )


def test_public_api_is_not_empty():
    """防止 __all__ 被误清空（清空会让上面的检查全部空转通过）。"""
    assert len(kakeya_ai.__all__) > 50
    assert len(kakeya_ai.ai.__all__) > 20


def test_proof_failure_reason_exported():
    """本项新增的失败归因是公共 API，应能从包顶层直接取到。"""
    from kakeya_ai import FAILURE_REASON_TEXT, FailureReason

    assert FailureReason.UNKNOWN is not FailureReason.BUDGET_EXHAUSTED
    assert set(FAILURE_REASON_TEXT) == set(FailureReason)
