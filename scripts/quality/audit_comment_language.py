#!/usr/bin/env python3
"""审计源码里**解释性内容**（docstring + 注释）的语言：中文 / 英文 / 混用。

**为什么需要它**（2026-10-10）：内部设计文档的《文档写作标准》此前只规定
`docs/` 与根 `README*` 用中文，**源码没有任何约定**——于是每份文件沿用了作者当时
的语言，同一概念在两处各写一遍、语言还不同（例如 `src/kakeya_ai/ai/` 的旧模块全
英文，而后来新增的 `model_router.py` 全中文）。协作时读到一半换语言很费神。

本脚本给出**可复现的数字**（不改任何文件），并 `--baseline` 输出棘轮基线供
`tests/test_comment_language.py` 冻结当前状态。

判定口径（三个阈值都可调，但**改阈值必须同时改基线**）：

* 只统计**解释性散文**：长度 ≥ `--min-chars`（默认 25）且含 ≥ `--min-words`
  （默认 4）个英文词。故 ``# pragma: no cover`` 这类短标签**不计**——它们不是
  "用英文解释了什么"，把它们算进去只会虚增分母。
* 逐块算 ``中文字数 / (中文字数 + 英文词数)``：
  ≥0.75 判 **ZH**、≤0.25 判 **EN**、中间判 **MIX**。
  （用"字数 vs 词数"而非"字数 vs 字数"，是因为中文按字、英文按词计更接近"信息量"；
  一个英文词约等于一个汉字的信息量。）

用法::

    python scripts/quality/audit_comment_language.py            # 全库报告
    python scripts/quality/audit_comment_language.py --baseline # 输出棘轮基线表
    python scripts/quality/audit_comment_language.py --list-en  # 列出"只有英文散文"的文件
"""
from __future__ import annotations

import argparse
import ast
import collections
import io
import json
import pathlib
import re
import sys
import textwrap
import tokenize

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

#: 切分"短语"的边界 = CJK 汉字 **+ CJK/全角标点**。
#:
#: 为什么把标点也算进来：``需要 id、title、model、messages`` 里的字段名被顿号隔开，
#: 若只按汉字切分，它会连成一个 4 词的"英文短语"而误判成"混用"——但那明显是
#: **中文句子 + 几个英文字段名**（实测踩过，旧子系统的 docstring）。
#: 顿号/逗号/括号/分号本来就是**短语边界**，跨过它们的词不该算"一串英文"。
PHRASE_SEP = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\u3000-\u303f\uff00-\uffef\u2018\u2019\u201c\u201d\u2014\u2026]"
)
#: 英文"词"的边界。**`_` 与 `.` 计入词内**：否则 ``p_core_all_ratio`` 会被拆成
#: 4 个词（p/core/all/ratio）、``nano.model.chip_model`` 拆成 3 个，于是一条纯中文的
#: 字段说明会被凑够 ≥4 词而误判成"混用"（实测踩过）。标识符按**一个词**算。
WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.]*")

#: 判定阈值（与基线绑定，改动需同时重算基线）。
MIN_CHARS = 12
MIN_WORDS = 4
ZH_CUT = 0.75
EN_CUT = 0.25
#: "英文散文片段"的最小连续词数。**这条是度量准不准的关键**，见 :func:`english_prose_words`。
PROSE_RUN = 4

#: 判定"这一行是**代码/公式/ASCII 图**而非散文"。命中即**不参与语言测量**。
#:
#: 为什么需要：中文注释里天然夹着公式（``speedup = n * (1 - ...)``）、调用示例
#: （``python -c "from x import y"``）、算子表（``3 MUL + 2 ADD``）。它们的"英文词"
#: 很多，但**不是用英文解释了什么**——算进去会把纯中文文件误判成英文/混用
#: （实测：`nano/model/chip_model.py` 的公式与算子表就自造了 3 个假阳性）。
_CODEISH = re.compile(r"[=⇒→←]|^\s{4,}\S|::|────")

#: 行内标记（Sphinx **cross-reference role** 与 markdown 行内代码）。测词数前**先剥掉**。
#:
#: 为什么：``:data:`POLICY_FIFO``` 里的标识符是**代码引用**、不是散文用词。不剥掉的话，
#: 一行如 ``: 默认准入策略（见 :data:`A` / :data:`B`）。`` 会被凑出 ≥4 个"英文词"而
#: 误判成"中英混用"——**实测踩过**（`nano/runtime/resource_arbiter.py` 就靠这个自造了一个假阳性）。
_ROLE = re.compile(r":\w+:`[^`]*`|``[^`]*``|`[^`]*`")

#: **纯工具指令**（没有散文可译）。整块只剩这些 ⇒ 不计入语言统计。
#:
#: 为什么必须排除：`# type: ignore`、`# pragma: no cover`、以及 ruff 的抑制指令，都是**给工具读的
#: 指令**，不是"用英文解释了什么"。它们**没有可译的文字**——永远判成 EN/MIX，于是
#: 无论怎么迁移，这几个文件都会一直"非零"。那是**度量口径错**，不是漏译。
#: 实测：`expert_router.py` 的 4 行 `# type: ignore` 就自造了 2 个假阳性。
#:
#: 判据：**行首**若是指令关键字，就把**指令本体**剥掉，只测其后的说明文字。
#:
#: 为什么是"剥掉前缀"而不是"整行必须是纯指令"：本项目惯例是"指令 + 中文说明"
#: （如 `# pragma: no cover - 仅调试用`）。若要求整行纯指令才算数，那条中文说明就
#: 永远测不到；若整行都算，`pragma`/`no`/`cover` 三个词又会把中文说明压成 MIX。
#: 剥掉前缀后两边都对：指令不计入英文，说明文字照常测。
_NOQA_KEY = "noqa"            # 拆开写，免得 ruff 把本文件的注释当成真指令
_DIRECTIVE_PREFIX = re.compile(
    r"^(?:#\s*)?(?:type:\s*ignore\s*(?:\[[^\]]*\])?"
    r"|pragma:\s*no\s*cover"
    r"|" + _NOQA_KEY + r"(?::[\s\w,]*)?"
    r"|doctest:[\s\w+\-]*"
    r"|pyright:[\s\w\-]*"
    r"|mypy:[\s\w\-]*"
    r"|ruff:[\s\w\-]*"
    r"|fmt:[\s\w]*"
    r"|isort:[\s\w]*)"
    r"[\s\-:.,]*",
    re.IGNORECASE,
)

#: 代码形态的 token（含 `/` 的路径、带扩展名的文件名、`a.b.c` 点式名）。测词数前**剥掉**。
#:
#: 为什么：`把 a.b.c 转成 a/b/c.py` 里 `a`/`b`/`c`/`py`/`init` 会被当成 5 个"英文词"，
#: 把一条纯中文注释压成"混用"（实测踩过，旧子系统）。路径是**代码**，
#: 不是"用英文解释了什么"。
_PATHISH = re.compile(r"\b[\w.\-]*\.[A-Za-z]{1,4}\b|\b\w+(?:/\w+)+\b")

#: 命令行成分（长/短选项、纯数字、RST 下划线行）。测词数前剥掉。
#:
#: 为什么：脚本 docstring 里常带 `Examples` 用例，如
#: ``--mode tubes --n 16 --show``。这些"词"（mode/tubes/n/show）不是**散文**——
#: 实测把一条纯中文说明压成"混用"（`scripts/visualization/visualize_kakeya.py` 的最后一块）。
_CLI = re.compile(r"(?<!\w)--?\w[\w\-]*|(?<!\w)\d+(?!\w)|^\s*-{3,}\s*$", re.M)

#: 跳过的目录（构建产物/依赖，不是本仓库源码）。
SKIP_PARTS = frozenset({"__pycache__", ".venv", "node_modules", ".git", "build", "dist"})


def _prose_only(text: str) -> str:
    """丢掉代码/公式/ASCII 图的行，只留"可能是散文"的行（见 :data:`_CODEISH`）。

    **先 ``dedent`` 再判缩进**：docstring 的正文本身就有缩进（函数内嵌的 docstring
    每行 4 空格起），若直接按"缩进 4+ 即代码"过滤，会把**正文全丢掉**——实测踩过，
    那会让英文块数被系统性低估。去公共前导空白后，只有**再往内缩**的才是代码块。
    """
    dedented = textwrap.dedent(text)
    lines = []
    for line in dedented.splitlines():
        if _CODEISH.search(line):
            continue                      # 代码/公式/ASCII 图：不是散文
        lines.append(_DIRECTIVE_PREFIX.sub("", line.strip()))   # 剥掉指令本体
    prose = _ROLE.sub(" ", "\n".join(lines))                   # 剥掉 RST 角色/行内代码
    prose = _PATHISH.sub(" ", prose)                           # 剥掉路径/文件名
    return _CLI.sub(" ", prose)                                # 剥掉 CLI 选项/数字/下划线


def english_prose_words(text: str) -> int:
    """只数**成串**的英文词（连续 ≥ ``PROSE_RUN`` 个词算一个散文片段）。

    **为什么不直接数所有英文词**（这条是踩出来的）：中文技术注释天然夹着大量英文
    **术语与标识符**——``SUB``/``MUL``/``ADD`` 这种算子名、``cpp/kernel/nano_kernel.cpp``
    这种路径、``Estimate.assumptions`` 这种字段。若把它们都算进"英文量"，一段纯粹的
    中文注释会被误判成"中英混用"。实测：`nano/model/chip_model.py` 的算子计数表（一整张
    中文说明 + 少量算子缩写）被判成 6 个 MIX 块——**全是假阳性**。

    判据改为"**连续成串**才算散文"：把文本按 CJK 切开，每一段里若英文词 ≥
    ``PROSE_RUN`` 个，才把该段的词计入。孤立的 `SUB` / 路径 / 字段名**不计**。
    这样"中文正文 + 英文术语"（本项目的目标写法）判为中文，而"整句英文"仍被判为英文。
    """
    total = 0
    for segment in PHRASE_SEP.split(_prose_only(text)):
        words = WORD.findall(segment)
        if len(words) >= PROSE_RUN:
            total += len(words)
    return total


def classify(text: str) -> str | None:
    """把一段文本判成 ``"ZH"`` / ``"EN"`` / ``"MIX"``；不是解释性散文则 ``None``。"""
    s = text.strip()
    if len(s) < MIN_CHARS:
        return None
    zh = len(CJK.findall(s))
    en = english_prose_words(s)
    if zh == 0 and en == 0:
        return None          # 全是路径/标识符，没有成句的解释
    if zh == 0:
        return "EN" if en >= MIN_WORDS else None
    if en == 0:
        return "ZH"
    ratio = zh / (zh + en)
    if ratio >= ZH_CUT:
        return "ZH"
    if ratio <= EN_CUT:
        return "EN"
    return "MIX"


def explanatory_blocks(source: str) -> list[str]:
    """抽取一个源文件里的全部解释性文本（docstring 与注释）。解析失败则返回 ``[]``。"""
    out: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                out.append(doc)
    # **连续的注释行要合成一块**：`tokenize` 每个注释行给一个 COMMENT token，
    # 若逐行分类，一条多行中文注释里"恰好英文词多"的那一行会被单独判成 MIX —— 实测
    # 过，那会把 MIX 数虚增（我新写的测试文件里一行 `#:` 续行就自造了一个 MIX 块）。
    # 语义上"一个注释块 = 一处解释"，故按行号是否相邻来合并。
    # 逐行取注释，并记录它是不是**独立成行**的注释（行首到 `#` 之间只有空白）。
    src_lines = source.splitlines()
    comments: list[tuple[int, str, bool]] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type != tokenize.COMMENT:
                continue
            before = src_lines[tok.start[0] - 1][: tok.start[1]]
            comments.append(
                (tok.start[0], tok.string.lstrip("#").strip(), not before.strip())
            )
    except (tokenize.TokenError, IndentationError):
        pass

    # **只把相邻的"独立成行"注释合并成一块**。
    #
    # 为什么不无条件合并相邻行：`# center x` / `# length` / `# delta` 是**三行不同代码**的
    # 行尾注释（各自说明一个数组下标），合并后会变成一个 3 词的"英文短语"而被判成英文——
    # 实测踩过（`tests/test_fast_kernel.py`），是**假阳性**。行尾注释各自独立，本就不属
    # 于同一段解释。
    run: list[str] = []
    prev_line: int | None = None
    prev_standalone = False
    for line, text, standalone in comments:
        merges = (prev_line is not None and line == prev_line + 1
                  and standalone and prev_standalone)
        if not merges and run:
            out.append("\n".join(run))
            run = []
        run.append(text)
        prev_line, prev_standalone = line, standalone
    if run:
        out.append("\n".join(run))
    return out


def measure(source: str) -> dict[str, int]:
    """返回 ``{"ZH": n, "EN": n, "MIX": n, "n": n}``（``n`` 是解释性散文块总数）。"""
    labels = [x for b in explanatory_blocks(source) if (x := classify(b))]
    return {
        "ZH": labels.count("ZH"),
        "EN": labels.count("EN"),
        "MIX": labels.count("MIX"),
        "n": len(labels),
    }


def scan(root: pathlib.Path, exts: tuple[str, ...] = (".py",)) -> dict[str, dict[str, int]]:
    """扫描 ``root`` 下的源码，返回 ``{相对路径: measure(...)}``（只含**有**散文的文件）。"""
    res: dict[str, dict[str, int]] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix not in exts or not path.is_file():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        got = measure(source)
        if got["n"]:
            res[str(path.relative_to(root))] = got
    return res


def aggregate(res: dict[str, dict[str, int]]) -> dict[str, tuple[int, int]]:
    """按**顶层目录**汇总 ``(EN 块数, MIX 块数)``。"""
    agg: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for path, got in res.items():
        top = path.split("/")[0] if "/" in path else "."
        agg[top][0] += got["EN"]
        agg[top][1] += got["MIX"]
    return {k: (v[0], v[1]) for k, v in agg.items()}


def _report(res: dict[str, dict[str, int]]) -> None:
    pure_en = sorted(p for p, g in res.items() if not g["ZH"] and not g["MIX"])
    pure_zh = sorted(p for p, g in res.items() if not g["EN"] and not g["MIX"])
    both = sorted(p for p, g in res.items() if g["ZH"] and g["EN"])
    print(f"含解释性散文的文件: {len(res)}")
    print(f"  只有中文散文                : {len(pure_zh)}")
    print(f"  只有英文散文                : {len(pure_en)}")
    print(f"  **同一文件内中英并存**      : {len(both)}")
    print("\n=== 按目录（EN 块 / MIX 块 / 文件数）===")
    agg = aggregate(res)
    for top in sorted(agg):
        n = sum(1 for p in res if (p.split("/")[0] if "/" in p else ".") == top)
        print(f"  {top:16} EN={agg[top][0]:4}  MIX={agg[top][1]:4}  文件={n:3}")
    tot = (sum(v[0] for v in agg.values()), sum(v[1] for v in agg.values()))
    print(f"  {'合计':16} EN={tot[0]:4}  MIX={tot[1]:4}  文件={len(res):3}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="审计源码解释性内容的语言（只读）")
    ap.add_argument("--root", default=".", help="仓库根（默认当前目录）")
    ap.add_argument("--baseline", action="store_true",
                    help="输出棘轮基线表（Python 字面量，供测试冻结当前状态）")
    ap.add_argument("--list-en", action="store_true", help="列出只有英文散文的文件")
    ap.add_argument("--json", action="store_true", help="输出逐文件 JSON")
    args = ap.parse_args(argv)

    root = pathlib.Path(args.root)
    res = scan(root)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.list_en:
        for p in sorted(p for p, g in res.items() if not g["ZH"] and not g["MIX"]):
            print(p)
        return 0
    if args.baseline:
        print("BASELINE = {")
        for top, (en, mix) in sorted(aggregate(res).items()):
            print(f'    {top!r}: ({en}, {mix}),')
        print("}")
        return 0
    _report(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
