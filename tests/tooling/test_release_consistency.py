"""发布一致性守门：版本号四处真源必须互相同步。

**为什么需要它。** 本项目的包版本有**四个**落点，此前**没有任何机制保证它们一致**：

=======  ==========================================  ==========
文件      内容                                         现状
=======  ==========================================  ==========
``version.py``      ``__version__ = "0.1.0"``              唯一真源（代码可 import）
``pyproject.toml``  ``dynamic = ["version"]``             正确引用真源，不写死
``CITATION.cff``    ``version: 0.1.0``                      **手工同步**
``CHANGELOG.md``    ``## [未发布]`` / ``## [0.1.0]``      **手工同步**
=======  ==========================================  ==========

``CITATION.cff`` 自己的注释就写着"版本号与 ``version.py`` 保持一致；发布时同步
更新"——**靠人记得**。而本项目已有两类实证：派生数字（survey 的统计）与"诚实
标注"都因为"靠人记得"而漂移过，每次都是补守门才止住。版本号是同类问题里**后果
最重**的一个：引用元数据里的版本错了，别人按它引用的就是错的。

因此本文件把"四处一致"钉成可执行契约，并顺带守两条 CHANGELOG 结构规范
（同版本同类别只应一个标题——此前 ``[未发布]`` 段里 ``### 修复`` 重复了 6 次）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

CHANGELOG = REPO_ROOT / "CHANGELOG.md"
CITATION = REPO_ROOT / "CITATION.cff"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _version_py() -> str:
    """从 ``version.py`` 取版本（导入而非正则——它是唯一真源）。"""
    from kakeya_ai.version import __version__

    return __version__


def _citation_version() -> str:
    text = CITATION.read_text(encoding="utf-8")
    m = re.search(r"^version:\s*(\S+)\s*$", text, re.MULTILINE)
    assert m is not None, "CITATION.cff 里找不到顶层 version 字段"
    return m.group(1).strip().strip('"\'')


def _changelog_versions() -> list[str]:
    """CHANGELOG 里所有版本节标题（按出现顺序）。``[未发布]`` 排除。"""
    text = CHANGELOG.read_text(encoding="utf-8")
    return re.findall(r"^## \[([^\]]+)\]", text, re.MULTILINE)


# ---------------------------------------------------------------------------
# 1. 版本号四处一致
# ---------------------------------------------------------------------------


def test_citation_version_matches_package_version():
    """``CITATION.cff`` 的版本必须等于 ``version.py``。

    **为什么这条最有价值**：引用元数据是给**外部**用的。别人按 ``CITATION.cff``
    引用本文档，若版本号与包实际版本不符，引用就是错的——而这件事**不会以任何
    方式报错**。
    """
    pkg = _version_py()
    cited = _citation_version()
    assert cited == pkg, (
        f"CITATION.cff 写 version: {cited}，但包版本是 {pkg}。"
        "发布时请同步；两者必须一致（CITATION.cff 的注释已声明该要求，"
        "但此前无守门，只能靠人记得）。"
    )


def test_changelog_has_current_version_section():
    """``CHANGELOG.md`` 必须有当前包版本的节标题。

    否则"这个包里到底有什么变更"在变更记录里查不到——对一个要对外交付的
    系统，这是可追溯性的底线。
    """
    pkg = _version_py()
    versions = _changelog_versions()
    assert pkg in versions, (
        f"CHANGELOG.md 里没有 [{pkg}] 节；现有版本节：{versions}。"
        "发版时应在顶部为该版本开一节。"
    )


def test_pyproject_reads_version_dynamically():
    """``pyproject.toml`` 必须**动态**读取版本，不得写死。

    写死会形成第二个真源——两处不同步时"哪个对"就没有答案（本项目
    ``trace_format`` 已确立"不设重复真源"的原则）。
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in text or "dynamic = ['version']" in text, (
        "pyproject.toml 未声明 dynamic version——版本号可能出现第二个真源"
    )
    # 且不得在 [project] 下有硬编码 version
    project_block = text.split("[project]", 1)[1].split("\n[", 1)[0]
    assert not re.search(r"^version\s*=", project_block, re.MULTILINE), (
        "pyproject.toml 的 [project] 段写了硬编码 version，与 version.py 形成两个真源"
    )


def test_version_is_valid_semver():
    """版本号必须形如 ``X.Y.Z``（本项目声明遵循 SemVer）。"""
    pkg = _version_py()
    assert re.fullmatch(r"\d+\.\d+\.\d+", pkg), (
        f"版本号 {pkg!r} 不是 X.Y.Z 形式；见 version.py 的 SemVer 说明"
    )


# ---------------------------------------------------------------------------
# 2. CHANGELOG 结构规范（Keep a Changelog）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("section", ["未发布"])
def test_changelog_has_no_duplicate_category_headers(section: str):
    """同一版本节里，每个类别（``### 新增`` 等）只能出现**一次**。

    **实证**：``[未发布]`` 段此前累积了 13 个类别标题，其中 ``### 修复`` 重复
    **6 次**、``### 新增`` 重复 4 次——因为每次加变更就追加一个新标题，从不
    合并。这不影响内容，但让"这一版改了哪些类"无法一眼看出，也违反 Keep a
    Changelog 的格式约定（该文件开头声明遵循它）。
    """
    text = CHANGELOG.read_text(encoding="utf-8")
    m = re.search(rf"^## \[{re.escape(section)}\](.*?)(?=^## \[|\Z)",
                  text, re.MULTILINE | re.DOTALL)
    assert m is not None, f"CHANGELOG.md 里找不到 ## [{section}] 节"
    body = m.group(1)
    headers = re.findall(r"^### (.+?)\s*$", body, re.MULTILINE)
    seen: dict[str, int] = {}
    for h in headers:
        seen[h] = seen.get(h, 0) + 1
    dupes = {k: v for k, v in seen.items() if v > 1}
    assert not dupes, (
        f"CHANGELOG 的 [{section}] 节里有重复的类别标题：{dupes}。"
        "同一版本同一类别只应有一个 ### 标题——新增条目应并入既有标题下，"
        "而不是再开一个。"
    )


def test_changelog_categories_are_known():
    """类别标题必须在项目声明的集合内（``新增/变更/修复/移除/安全``）。"""
    allowed = {"新增", "变更", "修复", "移除", "安全"}
    text = CHANGELOG.read_text(encoding="utf-8")
    unknown = sorted(set(re.findall(r"^### (.+?)\s*$", text, re.MULTILINE)) - allowed)
    assert not unknown, (
        f"CHANGELOG 出现未声明的类别：{unknown}。"
        f"允许的类别见文件开头的「变更类型」：{sorted(allowed)}"
    )


def test_unreleased_section_exists():
    """必须保留 ``[未发布]`` 节——它是"下次发版要收进去的东西"的落点。

    没有它，新变更只能塞进某个已发布版本（造成"发布后再改"的历史失真）。
    """
    assert "未发布" in _changelog_versions(), (
        "CHANGELOG.md 缺少 [未发布] 节"
    )
