#!/usr/bin/env bash
# 把 docs/diagrams/*.mmd 导出成 PNG（**可选工具**，产物不入库）。
#
# ## 什么时候用
#
# 架构图的唯一真源是 `.mmd`——GitHub / VS Code 直接渲染 ```mermaid 代码块，
# 本仓库没有任何文件引用 PNG。只有**需要位图**的场合才跑本脚本：
# 贴进幻灯片、发给不装编辑器的人、或截图存档。
#
# ## 前置依赖
#
#   node / npm                —— 任意 18+ 版本
#   @mermaid-js/mermaid-cli   —— npm install -g @mermaid-js/mermaid-cli
#
# 首次运行 Puppeteer 会下载 Chromium（约 300 MB，缓存在 ~/.cache/puppeteer/）。
#
# ## 用法
#
#   bash scripts/docs_build/render_diagrams.sh            # 导出全部
#   bash scripts/docs_build/render_diagrams.sh move_*     # 只导出匹配的（通配）
#
# ## 输出位置
#
#   docs/figures/diagram_<名>.png        中文版
#   docs/figures/en/diagram_<名>.png     英文版（靠目录区分语言，文件名相同）
#
# `docs/figures/` 已被 .gitignore 忽略：PNG 是产物，不入库。
#
# ## 中文字体
#
# Mermaid/Puppeteer 用的是系统 Chromium，**中文字体走系统**（macOS 自带
# PingFang SC）。缺字形会静默画成豆腐块，导出后请**自己看一眼**——这条没有
# 守门（PNG 是可选产物，守门只盯图源）。
set -euo pipefail

# 仓库根：脚本在 ``scripts/docs_build/`` 下，故上溯**两级**
# （``docs_build/`` → ``scripts/`` → 仓库根）。2026-11-06 拆子目录前是上溯一级。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="$ROOT/docs/diagrams"
OUT_DIR="$ROOT/docs/figures"

# 优先用 PATH 里的 mmdc；也允许用环境变量 MMDC 显式指定（非登录 shell 常丢 PATH）。
if [ -n "${MMDC:-}" ] && [ -x "${MMDC}" ]; then
    :
elif command -v mmdc >/dev/null 2>&1; then
    MMDC="$(command -v mmdc)"
else
    echo "错误：找不到 mmdc。装法：" >&2
    echo "  npm install -g @mermaid-js/mermaid-cli" >&2
    echo "（已有安装可用 MMDC=/path/to/mmdc 指定）" >&2
    exit 1
fi

pattern="${1:-*}"
shopt -s nullglob
files=("$SRC_DIR"/${pattern}.mmd)
shopt -u nullglob

if [ ${#files[@]} -eq 0 ]; then
    echo "没有匹配 $pattern 的 .mmd 文件（在 ${SRC_DIR}）" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "mmdc: $MMDC"
echo "渲染 ${#files[@]} 个图 → $OUT_DIR"
echo

fail=0
for f in "${files[@]}"; do
    name="$(basename "$f" .mmd)"
    # 双语约定：中文放 figures/，英文放 figures/en/，**文件名相同**。
    # 命名规则是
    #   xxx.mmd     -> figures/diagram_xxx.png       （中文，默认）
    #   xxx_en.mmd  -> figures/en/diagram_xxx.png    （英文）
    # 与测试里的 bilingual 守门同一口径：靠**目录**区分语言。
    case "$name" in
        *_en)
            stem="${name%_en}"
            out="$OUT_DIR/en/diagram_${stem}.png"
            mkdir -p "$OUT_DIR/en"
            ;;
        *)
            out="$OUT_DIR/diagram_${name}.png"
            ;;
    esac
    # -b white：透明背景贴进浅色/深色底稿都会出问题，显式给白底。
    # -s 3：3 倍缩放，缩到文档页宽后仍清晰（默认 1 太糊）。
    if "$MMDC" -i "$f" -o "$out" -b white -s 3 >/dev/null 2>&1; then
        printf "  OK   %-28s → %s\n" "$name" "${out#$OUT_DIR/}"
    else
        printf "  FAIL %-28s\n" "$name"
        # 重跑一次不吞输出，便于定位语法错误。
        "$MMDC" -i "$f" -o "$out" -b white -s 3 2>&1 | tail -20
        fail=1
    fi
done

echo
if [ $fail -eq 0 ]; then
    echo "全部渲染成功。"
    echo "提示：改动 .mmd 后**必须重跑本脚本**——PNG 是产物，不会自动更新。"
else
    echo "有图渲染失败，见上方输出。" >&2
fi
exit $fail
