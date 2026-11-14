"""静态检查守门的契约测试：**类型检查、lint、Lean 真编译必须在 CI 里**。

**为什么需要它。** 2026-10-02 把全库 pyright 从
**181 errors / 3 warnings 清到 0 / 0**（含 `tests/` 与 `scripts/`）。但当时
CI 的 `lint` job **只跑 ruff**，而 `ruff` **不做类型检查**——于是这个成果
**零守门**：下次任何改动都可以把报错加回去，而没有任何东西会红。

更具体的代价：那轮 pyright 顺带挖出了 **6 个真 bug**，其中两个是"必然崩"：
`model_router.route_answer` 的 `dict.update(str)`（抛 ValueError）、
`visualize_kakeya.py --mode moves` 的漏传必填参数（抛 TypeError）。
**没有守门时，这类问题只会重新积累、且要再靠"有人想起来跑一次"才发现。**

这与 ``test_ci_platform_coverage.py``（把"CI 覆盖哪些平台"钉成契约）是同一
思路：**配置类的事实必须可验证**，否则它静默腐化（那次的教训是文档写"CI 有
macOS job"而该 job 从未存在过）。

**本文件只验证"配置里有这一步"，不重跑 pyright**：本机有"一次最多两个测试
文件"的资源约束（见 ``CONTRIBUTING.md``），在测试里跑全库类型检查不现实。
真正的执行在 CI（那里没有该约束）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT = REPO_ROOT / "pyproject.toml"
PYRIGHT_CONFIG = REPO_ROOT / "pyrightconfig.json"

yaml = pytest.importorskip("yaml", reason="需要 pyyaml 解析 CI 配置")


def _ci() -> dict:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))


def test_ci_runs_pyright():
    """CI 必须有一个跑 pyright 的步骤。

    判据用 "某步骤的 run 里出现 pyright"，而不是匹配步骤名——名字可以随便
    改，命令才是实质。
    """
    ci = _ci()
    runs = [
        step.get("run", "")
        for job in ci["jobs"].values()
        for step in job.get("steps", [])
    ]
    matching = [r for r in runs if "pyright" in r]
    assert matching, (
        "CI 没有任何步骤跑 pyright——类型检查零守门，报错会重新积累。"
        "（2026-10-02 已把全库清到 0/0，需要守住。）"
    )


def test_pyright_step_is_not_allowed_to_fail():
    """类型检查步骤**不得** ``continue-on-error``。

    ``Ruff format check`` 刻意是软性的（存量未格式化，强制会淹没真问题），
    但类型检查不同：目标就是**保持 0**，允许失败等于没守。
    """
    ci = _ci()
    for job in ci["jobs"].values():
        for step in job.get("steps", []):
            if "pyright" in step.get("run", ""):
                assert not step.get("continue-on-error"), (
                    f"pyright 步骤 '{step.get('name')}' 设了 continue-on-error，"
                    "等于没有守门"
                )


def test_pyright_is_declared_as_dev_dependency():
    """``pyright`` 必须在 ``[project.optional-dependencies].dev`` 里。

    否则 CI 得靠 ``pip install pyright`` 这种"绕过依赖声明"的写法，
    本地开发也无法一次装齐（``pip install -e .[dev]``）。
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10：stdlib 的 tomllib 是 3.11 才有
        import tomli as tomllib

    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dev = data["project"]["optional-dependencies"]["dev"]
    assert any(dep.split(">")[0].split("=")[0].strip() == "pyright" for dep in dev), (
        f"pyright 未声明在 dev 依赖里：{dev}"
    )


def _strip_jsonc_comments(text: str) -> str:
    """去掉 JSONC 的 ``//`` 行注释，使标准 ``json`` 模块能解析。

    ``pyrightconfig.json`` 刻意用了 JSONC（pyright 支持 ``//`` 注释，那些注释
    解释了每个配置项的**为什么**）。但标准 ``json`` 模块不认注释——直接读会
    抛 ``JSONDecodeError``。这里只剥注释、不做别的语法放宽。

    **只处理行注释**：本文件没用块注释与尾逗号；若将来用了，本函数需同步扩展
    （否则又会在这里报错——那正是这条守门存在的意义）。
    """
    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        out.append(line)
    return "\n".join(out)


def test_pyrightconfig_exists_and_covers_src_and_scripts():
    """仓库根必须有 ``pyrightconfig.json``，且 ``extraPaths`` 含 src 与 scripts。

    **为什么盯这个**：没有它时命令行 pyright 会报 4 处幽灵错误
    （tests import scripts 下的模块，而 ``.vscode/settings.json`` 的
    ``extraPaths`` **只有 IDE 读**）。CI 用的是命令行，所以配置文件是必需的。
    """
    import json

    assert PYRIGHT_CONFIG.is_file(), "缺少 pyrightconfig.json（CI 的类型检查依赖它）"
    raw = PYRIGHT_CONFIG.read_text(encoding="utf-8")
    cfg = json.loads(_strip_jsonc_comments(raw))
    paths = cfg.get("extraPaths", [])
    for expected in ("src", "scripts"):
        assert expected in paths, (
            f"pyrightconfig.json 的 extraPaths 缺 {expected!r}：{paths}"
        )

