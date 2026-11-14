#!/usr/bin/env python
r"""``.kpt`` 格式解释器的命令行入口（薄壳）。

实现与用法见 ``src/kakeya_ai/kpt_cli.py``；这里只负责把仓库的 ``src/`` 加进
导入路径并转发，方便直接 ``.venv/bin/python scripts/orchestration/kpt.py ...`` 调用。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kakeya_ai.kpt_cli import main

if __name__ == "__main__":
    sys.exit(main())
