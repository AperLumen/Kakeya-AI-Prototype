"""版本号的单一真源。

**为什么单独一个模块**：版本号此前只活在 ``pyproject.toml`` 里，运行时的包
``kakeya_ai`` 查不到自己的版本——服务启动时无法在日志/``/api/health`` 里报告
"我跑的是哪个版本"，产物（``.kpt`` trace、实验报告 JSON）也无法记录生成它的
软件版本。工业化交付要求"任何一份产物都能追到产生它的代码版本"，所以版本号
必须有一个**可被代码 import** 的真源。

``pyproject.toml`` 通过 ``[tool.setuptools.dynamic]`` 读这里，因此
**唯一改动点就是这个文件**；``CHANGELOG.md`` 与 git tag 以它为准。

版本号含义（SemVer）：

- **major**：不兼容的接口/格式变更（含 ``.kpt`` 协议版本抬升）。
- **minor**：向后兼容的功能新增（新增可选字段、新增模块、新增后端）。
- **patch**：向后兼容的缺陷修复（数值不变、格式不变）。

当前 ``0.x``：**尚未承诺接口稳定性**——公共 API 仍可能在不发 major 的情况下
调整。对外提供的稳定性承诺是 ``.kpt`` 文件协议（见 ``trace_format.py``），
它的版本由 ``protocol`` 字段独立管理，与此处的包版本解耦。
"""

from __future__ import annotations

#: 包版本号（唯一真源；由 ``pyproject.toml`` 的 ``[tool.setuptools.dynamic]`` 段落指向此处读取）。
__version__ = "0.3.0"

__all__ = ["__version__"]
