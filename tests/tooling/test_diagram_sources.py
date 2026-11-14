"""守门：架构图图源（``docs/diagrams/*.mmd``）的完整性。

**为什么只守图源、不守 PNG 产物**

本仓库的架构图以 Mermaid 源（``.mmd``）为**唯一真源**：GitHub 与编辑器都能直接
渲染 ```mermaid 代码块，``docs/diagrams/`` 里的六个文件就是"架构"的交付物本身。
``scripts/docs_build/render_diagrams.sh`` 仍保留，但它只是**可选**的导出工具
（需要 Node + mermaid-cli + Chromium），不再有仓内消费者——原先消费 PNG 的
LaTeX 流水线不在本仓库范围内。

因此这里不检查 PNG 是否存在/是否比源新：

* **无消费者**：没有文件引用 ``docs/figures/*.png``，产物不存在也不影响任何人；
* **mtime 判据在全新克隆上不可靠**：git 检出会把所有文件的 mtime 设成同一时刻，
  "PNG 比 .mmd 新"取决于检出顺序，会让守门随机变红——这类"看天吃饭"的断言
  比没有断言更糟。

**保留的两条**（都是图源自身的性质，与渲染无关）：

1. 图源目录存在且非空（防止目录被误删后守门静默空转）；
2. 双语成对：每个 ``xxx.mmd`` 必须有 ``xxx_en.mmd``，反之亦然。
   约定是"靠目录/后缀区分语言、文件名主干相同"——缺一侧就意味着某语言的
   读者会看到失真的架构（本项目在别处已多次记录这类静默漂移）。

``scheduler_flow.mmd`` 与架构清单的同源性另有更严格的守门
（``test_architecture_manifest.py``：清单是生成器，手改 mmd 即红），此处不重复。
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "docs" / "diagrams"

#: 英文版后缀。``xxx.mmd`` 是中文（默认），``xxx_en.mmd`` 是英文。
EN_SUFFIX = "_en"


def _mmd_files() -> list[Path]:
    return sorted(SRC_DIR.glob("*.mmd"))


def test_mmd_source_dir_exists() -> None:
    """图源目录必须存在且非空——否则下面两条会在空集合上"全部通过"。"""
    assert SRC_DIR.is_dir(), f"架构图源目录缺失：{SRC_DIR}"
    assert _mmd_files(), f"{SRC_DIR} 下没有任何 .mmd 文件"


def test_every_mmd_has_bilingual_counterpart() -> None:
    """每个图源都要有中英两版：``xxx.mmd`` ↔ ``xxx_en.mmd``。"""
    stems = {p.stem for p in _mmd_files()}
    missing_en = sorted(s for s in stems if not s.endswith(EN_SUFFIX)
                        and f"{s}{EN_SUFFIX}" not in stems)
    orphan_en = sorted(s for s in stems if s.endswith(EN_SUFFIX)
                       and s[: -len(EN_SUFFIX)] not in stems)

    problems = []
    if missing_en:
        problems.append(
            "缺英文版（应补 "
            + "、".join(f"{s}{EN_SUFFIX}.mmd" for s in missing_en)
            + "）"
        )
    if orphan_en:
        problems.append(
            "英文版没有对应中文源（应补 "
            + "、".join(f"{s[: -len(EN_SUFFIX)]}.mmd" for s in orphan_en)
            + "，或删掉这些孤立英文版）"
        )
    assert not problems, "架构图双语不成对：\n  " + "\n  ".join(problems)


# ---------------------------------------------------------------------------
# README 内嵌图与图源的一致性
# ---------------------------------------------------------------------------

#: 两版首页各内嵌一张调度主链图：中文版对中文图源，英文版对英文图源。
#:
#: **为什么英文版也要守**：漂移的机制与中文版一模一样——改清单、重新生成图源、
#: 只同步了其中一版 README，另一版就继续展示**旧架构**且不报错。只守中文版等于
#: 默认"英文读者看到过时架构没关系"，这与本仓库的双语约定相悖。
README_PAIRS = [
    pytest.param(REPO_ROOT / "README.md", SRC_DIR / "scheduler_flow.mmd", id="zh"),
    pytest.param(REPO_ROOT / "README.en.md", SRC_DIR / "scheduler_flow_en.mmd", id="en"),
]


def _normalized(lines: list[str]) -> str:
    """去掉 ``%%`` 注释行后的图体，供逐字比对（空行压缩、行尾空格清理）。"""
    import re

    body = "\n".join(l.rstrip() for l in lines if not l.startswith("%%"))
    return re.sub(r"\n{3,}", "\n\n", body).strip()


@pytest.mark.parametrize(("readme", "diagram"), README_PAIRS)
def test_readme_mermaid_block_matches_source(readme: Path, diagram: Path) -> None:
    """README 内嵌的 mermaid 必须与对应语言的图源同源。

    **为什么需要**：README 是大多数人对架构的第一印象，内嵌图是"抄"自图源的。
    抄本必然漂移——改了清单、忘了改 README，读者看到的就是**旧架构**，而且
    没有任何报错。这里把两者钉死：README 的 ```mermaid 块必须等于图源的图体
    （``%%`` 注释行不参与比对，README 里不需要这些内部说明）。

    改架构的正确姿势：改清单 → 重新生成 ``scheduler_flow*.mmd`` → 把图体同步到
    两版 README（或直接跑本测试看差异）。
    """
    assert readme.is_file(), f"README 缺失：{readme}"
    text = readme.read_text(encoding="utf-8")

    start = text.find("```mermaid")
    assert start >= 0, f"{readme.name} 里没有 ```mermaid 代码块（架构图被删掉了？）"
    start = text.index("\n", start) + 1
    end = text.index("```", start)
    embedded = text[start:end]

    source = _normalized(diagram.read_text(encoding="utf-8").splitlines())
    assert _normalized(embedded.splitlines()) == source, (
        f"{readme.name} 内嵌的调度主链图与 {diagram.name} 不一致。\n"
        "修法：改架构走清单 → 重新生成 mmd → 把图体同步进 README 的 ```mermaid 块。"
    )


@pytest.mark.parametrize(("readme", "other"), [
    pytest.param(REPO_ROOT / "README.md", "README.en.md", id="zh"),
    pytest.param(REPO_ROOT / "README.en.md", "README.md", id="en"),
])
def test_readme_links_the_other_language(readme: Path, other: str) -> None:
    """两版首页必须互相链接。

    **为什么需要**：语言切换链接是**唯一**的入口，断了不会有任何报错——英文读者
    落在中文首页、中文读者不知道有英文版，都属于"静默失去一半读者"。改名或移动
    文件时这条会立刻红。
    """
    assert readme.is_file(), f"README 缺失：{readme}"
    text = readme.read_text(encoding="utf-8")
    assert f"({other})" in text, (
        f"{readme.name} 里没有指向 {other} 的语言切换链接"
        "（形如 `[English](README.en.md)` / `[简体中文](README.md)`）。"
    )
