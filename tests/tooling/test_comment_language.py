"""解释性内容的语言：**必须全中文**（硬守门）。

## 约定

源码里的**解释性内容**（docstring 与注释）**一律中文**，
英文只留给标识符、文件路径、shell 命令、环境变量名、模型名、格式/协议名、数学符号。
`# type: ignore` / `# pragma: no cover` 这类**工具指令**原样保留（指令本体不译，
其后的说明文字用中文）。

## 为什么需要守门

本约定此前**只覆盖 `docs/` 与根 `README*`**，源码没有任何约定，于是每份 `.py` 沿用了
作者当时的语言。2026-10-10 一次全库迁移前的实测（247 个含解释性散文的 `.py`）：

| 类别 | 文件数 |
| --- | --- |
| 只有**英文**散文 | 41 |
| 只有**中文**散文 | 127 |
| **同一文件内中英并存** | 72 |

非中文块合计 **942 个纯英文 + 27 个中英混用**。最伤协作的是**同一文件内中英并存**——
读到一半换语言，且"同一个概念在两处各写一遍、语言还不同"（如 `src/kakeya_ai/ai/` 的
旧模块全英文，而新增的 `model_router.py` 全中文）。

迁移已完成，**全库 0 英文 / 0 混用**，故 `BASELINE` 全为 `(0, 0)`——本守门现在是
**硬门**：新增任何英文散文都会失败。

## 守住什么

* 每个顶层目录的英文块数与混用块数**不得超过** `BASELINE`（现全为 0）。
* 基线里没登记的目录不允许出现英文块（挡住"新开一个目录全是英文"）。
* 分类器自身的单测（短标签、中文夹术语、纯英文、真混用各判对）。

**局限（如实写清）**：它按**目录**汇总，故"一个文件加英文、另一个删英文"能互相抵消；
也不检查未被 `scan()` 覆盖的扩展名（目前只扫 `.py`）。
"""
from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "quality"))

# 必须先 sys.path.insert 才能导入 scripts/ 下的模块（pytest 的 pythonpath 已含
# scripts，但直接跑本文件时不一定）——故这一行刻意不在文件顶部。
import audit_comment_language as acl

#: 棘轮基线：`{顶层目录: (英文块数, 混用块数)}`。
#: **由 `python scripts/quality/audit_comment_language.py --baseline` 生成**——刻意不手抄，
#: 手抄的数字必然过期（本项目已多次踩）。
#:
#: 迁移时**可以且应该**把它往下调（改完注释后跑那条命令，把新数字贴回来）。
#: 往上调只在一处情形下正确：把一段英文注释**翻译**成中文但顺带新增了中文内容，
#: 导致……不会发生。故：**任何上调都应当被质疑**。
BASELINE = {
    '.': (0, 0),
    'docs': (0, 0),
    'nano': (0, 0),
    'scripts': (0, 0),
    'src': (0, 0),
    'tests': (0, 0),
}


@pytest.fixture(scope="module")
def scanned():
    return acl.scan(REPO_ROOT)


def test_scan_is_not_vacuous(scanned):
    """**先挡"扫到空"导致的假通过**。

    这条不是形式主义：如果 `scan()` 因某个正则/路径 bug 返回空字典，下面所有棘轮
    断言都会**通过**（0 ≤ 基线），守门静默失效。故先把"扫到了合理的量"钉住。
    """
    assert len(scanned) > 100, f"只扫到 {len(scanned)} 个文件，疑似 scan 失效"
    total = sum(g["n"] for g in scanned.values())
    assert total > 1000, f"只扫到 {total} 个解释性块，疑似抽取失效"
    # 必须覆盖到三个主要代码目录，否则"按目录汇总"没意义。
    # 用 `aggregate` 的口径取顶层目录——它把根目录下的散文件归到 `"."`，
    # 直接 `p.split("/")[0]` 会把 `setup.py` 当成一个"目录名"（踩过）。
    tops = set(acl.aggregate(scanned))
    assert {"src", "tests", "scripts"} <= tops


def test_english_blocks_do_not_grow(scanned):
    """每个目录的**英文散文块数**不得超过基线（棘轮）。"""
    agg = acl.aggregate(scanned)
    worse = []
    for top, (base_en, _) in BASELINE.items():
        now = agg.get(top, (0, 0))[0]
        if now > base_en:
            worse.append(f"{top}: {base_en} → {now}（+{now - base_en}）")
    assert not worse, (
        "解释性内容的英文块数增加了——新增/修改的 docstring 与注释应当用中文"
        "（约定：源码解释性内容一律中文，见本文件 docstring）。若确有意为之，请重新生成基线并说明理由：\n  "
        + "\n  ".join(worse)
        + "\n取数：python scripts/quality/audit_comment_language.py --baseline"
    )


def test_mixed_blocks_do_not_grow(scanned):
    """每个目录的**混用块数**不得超过基线（棘轮）。"""
    agg = acl.aggregate(scanned)
    worse = []
    for top, (_, base_mix) in BASELINE.items():
        now = agg.get(top, (0, 0))[1]
        if now > base_mix:
            worse.append(f"{top}: {base_mix} → {now}（+{now - base_mix}）")
    assert not worse, (
        "解释性内容的**中英混用**块数增加了——同一段解释里别中英文混着写"
        "（约定：源码解释性内容一律中文，见本文件 docstring）：\n  " + "\n  ".join(worse)
        + "\n取数：python scripts/quality/audit_comment_language.py --baseline"
    )


def test_no_unlisted_directory_has_english_blocks(scanned):
    """基线里**没登记**的目录不允许出现英文块——挡住"新开一个目录全是英文"。"""
    agg = acl.aggregate(scanned)
    unlisted = {t: v for t, v in agg.items() if t not in BASELINE and v[0] > 0}
    assert not unlisted, (
        f"这些目录不在语言基线里却含英文散文：{unlisted}。"
        "新目录请用中文写解释性内容；确需登记就更新 BASELINE 并说明理由。"
    )


class TestClassifier:
    """分类器本身要准——否则棘轮数字是错的（这一类 bug 最隐蔽）。"""

    def test_short_tags_are_not_prose(self):
        """`# pragma: no cover` 之类短标签**不算**英文散文。

        回归：初版没这个阈值，把 `# pragma: no cover` 记成"一个英文块"，于是中文
        文件被误判成"中英混用"（`proof_orchestrator.py` 曾被虚增 14 个英文块）。
        """
        assert acl.classify("# pragma: no cover") is None
        assert acl.classify("# keep / subset / rescale") is None
        assert acl.classify("#") is None
        assert acl.classify("") is None

    def test_chinese_prose_with_terms_is_zh(self):
        """中文散文里夹英文标识符/路径 —— 仍判中文（这正是本项目的写法）。"""
        assert acl.classify(
            "把签名库编译成约束解码用的 JSON Schema，交给 Ollama 的 format 参数。"
        ) == "ZH"

    def test_pure_english_prose_is_en(self):
        assert acl.classify(
            "Record of a single Move application; carries the input and output counts."
        ) == "EN"

    def test_balanced_mix_is_mix(self):
        text = "This helper 负责把 grains 合并 along the long axis, keeping bins separate."
        assert acl.classify(text) == "MIX"


class TestExtraction:
    def test_extracts_module_class_and_function_docstrings_plus_comments(self):
        src = (
            '"""模块 docstring，够长且是中文，用于通过阈值判定。"""\n'
            "# 一条注释，也够长，同样是中文，用于判定\n"
            "class C:\n"
            '    """类的 docstring，中文，长度足够用于判定其语言。"""\n'
            "    def f(self):\n"
            '        """函数的 docstring，中文，长度足够用于判定语言。"""\n'
        )
        blocks = acl.explanatory_blocks(src)
        assert len(blocks) == 4           # 模块 + 类 + 函数 + 注释

    def test_unparsable_source_is_empty_not_crash(self):
        assert acl.measure("def broken(:\n")["n"] == 0
