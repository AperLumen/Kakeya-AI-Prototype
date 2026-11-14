"""守门：已诚实标注的"启发式/代理量"不得被悄悄改成"论文公式"。

**背景。** 本项目对"哪些实现有论文依据、哪些是自创代理"标注得相当诚实
（"computational proxy"、"approximate"、"plausible skeleton" 等随处可见）。
但这套标注有一个结构性弱点：**它是纯文本的，删掉不会让任何测试变红**。
一旦有人（或某次重构）把"这是代理量"的段落删掉，代码就会重新**看起来**像
忠实实现——而 ``induction_step_g`` 已经真实发生过这一幕（内部设计文档
写着是近似，代码 docstring 却写着"计算 Proposition 1.7 中的 ω 压缩量"）。

**这组测试把标注钉成可执行契约**：断言这些函数的 docstring 里**仍然**含有
免责表述。它保护的不是数值行为，而是**可读结论的诚实性**。

**为什么用 ``inspect.getdoc`` 而不是读源码做子串匹配。**
子串匹配分不清"代码"与"描述代码的字符串/注释"——本项目在这上面踩过坑
（用 ``"find_pdfs" in src`` 判是否调用了某 API，结果文件注释里正好有这个
词，检查静默放行）。``inspect.getdoc`` 取的是**运行时 docstring 对象**，
不涉及源码文本解析，因此不存在这类误判。
"""

from __future__ import annotations

import inspect
from typing import Callable, List, Tuple

import pytest

from kakeya_ai import grains, moves

#: (函数, 必须出现的关键词之一, 缺口说明)
#:
#: 关键词用"变体组"而非单一短语：同一条约束在不同函数里措辞不同
#: （proxy / approximate / plausible），只要组内命中一个即算标注仍在。
#: 组**之间**不重叠，避免把"修好了 A"误判成"B 也标了"。
_DISCLAIMERS: List[Tuple[Callable, Tuple[str, ...], str]] = [
    (
        moves.regularize_shading,
        ("proxy", "approximate", "not the paper", "代理"),
        "Move #3 后条件的前提；删掉标注会让'密度下限'看起来像论文的 "
        "regular shading 构造",
    ),
    (
        grains.broadness_at_points,
        ("proxy", "simplified", "sampled", "approximate", "代理", "简化"),
        "论文的 broadness 定义是'任意帽'；实现只在采样点/有限半径上检查",
    ),
    (
        grains.broadness_reduction,
        ("proxy", "paper's construction", "代理"),
        "剥离窄簇后**没有**论文里'小帽上的归纳'承接——不标注会被当成完整实现",
    ),
    (
        grains.adaptive_partition_into_rho_tubes,
        ("adapt", "growing", "until", "自适应", "不断扩大", "直到"),
        "论文的 ρ-tube 覆盖有比例条件；此处是自创的'扩到每 bin 够数'规则",
    ),
    (
        grains.two_scale_grains_skeleton,
        ("skeleton", "骨架"),
        "名字像论文的统一双尺度构造，实为 ρ-bin × Guth 骨架的拼接",
    ),
    (
        grains._build_grains_for_tube_subset,
        ("approximation", "proxy", "approximate", "近似"),
        "其 local_kt_error 是 n|T|/|G| proxy，非论文的 C_KT-CW^loc(G)",
    ),
]


@pytest.mark.parametrize(
    "fn,keywords,gap",
    _DISCLAIMERS,
    ids=[f.__name__ for f, _kw, _g in _DISCLAIMERS],
)
def test_disclaimer_still_present(fn, keywords, gap):
    """docstring 必须仍含免责表述——删掉即红。"""
    doc = inspect.getdoc(fn) or ""
    low = doc.lower()
    assert any(kw.lower() in low for kw in keywords), (
        f"{fn.__module__}.{fn.__qualname__} 的 docstring 里找不到任何免责表述 "
        f"{keywords}。\n"
        f"该实现的缺口是：{gap}\n"
        "若确实已完成论文级实现，请同时更新本测试与 CHANGELOG；"
        "否则请恢复标注——'看起来像论文公式'比'承认是代理量'危险得多。"
    )


def test_moves_module_header_keeps_skeleton_wording():
    """``moves.py`` 的模块头必须仍声明这是"结构骨架"而非完整算法。

    它声明了整份 ``moves.py`` 的定位（"plausible geometric transformations
    ... so that the iterative loop can be wired up"）。这个定位一旦从模块头
    消失，读者会以为三个 Move 是论文算法的忠实实现。
    """
    doc = inspect.getdoc(moves) or ""
    low = doc.lower()
    assert "plausible" in low or "skeleton" in low or "骨架" in low, (
        "moves.py 模块头不再声明'这是骨架/plausible 变换'"
    )


def test_slab_volume_documents_exactness_and_history():
    """``Slab.volume`` 必须声明它现在是**精确**闭式，并保留历史缺陷记录。

    该属性曾取"主导轴盒面面积 × 厚度"（忽略法向、系统性偏小，真值差约 23%），
    导致 ``frostman_slab_error`` 偏大、``is_sticky`` 偏 non-sticky。2026-10-09
    改为板 ∩ 盒的精确交集体积（广义 Irwin–Hall 闭式）。本测试把**结论**钉住：
    docstring 必须①声明精确、②给出方法、③保留"曾偏小"的历史（否则后来者会
    以为它一直是精确的，也会在无记录的情况下把它改回近似式）。
    """
    from kakeya_ai.wolff_axioms import Slab

    doc = inspect.getdoc(Slab.volume) or ""
    assert "精确" in doc, "Slab.volume 的 docstring 未声明它是精确体积"
    assert "Irwin" in doc or "闭式" in doc, (
        "Slab.volume 的 docstring 未说明计算方法（Irwin–Hall 闭式）"
    )
    assert "偏小" in doc, (
        "Slab.volume 的 docstring 未保留历史缺陷方向（曾系统性偏小）"
    )
