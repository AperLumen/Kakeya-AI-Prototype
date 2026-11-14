#!/bin/bash
# 构建 C++ 加速扩展（_nano_kernel 与 _tube_kernel）：按平台自动选编译参数。
set -euo pipefail
cd "$(dirname "$0")"

# 选择 Python：优先仓库内虚拟环境，否则用 PATH 中的 python3（CI 环境）。
if [ -x "../.venv/bin/python" ]; then
    PYTHON="../.venv/bin/python"
else
    PYTHON="$(command -v python3)"
fi

# Build via Python to handle spaces in paths robustly.
"$PYTHON" - <<'PY'
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile

import pybind11

py_includes = [
    subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"], text=True).strip().split()[0][2:],
    pybind11.get_include(),
]

compiler = "clang++" if shutil.which("clang++") else "g++"

base = [compiler, "-O3", "-std=c++17", "-fPIC", "-shared", "-pthread"]
if sys.platform == "darwin":
    # macOS 上扩展模块允许未定义符号（运行时由解释器解析）。
    base += ["-undefined", "dynamic_lookup"]

    # --- SDK 兼容兜底（2026-09-27）---
    # 症状：CommandLineTools 的**默认** SDK 可能比本机 ld 新。实测默认
    # MacOSX27.0.sdk 的 .tbd 声明了 `arm64e.x1-*`，而本机 ld 不认识，于是
    # **任何** C++ 链接都失败（连 int main(){} 也不成）。
    #
    # 旧实现在此情形下靠"逐级回落 flags"碰巧成功（回落到无 -mcpu 那档时
    # 恰好绕开），但代价是：每次构建都往 stderr 刷几十行 ld error，让人误以为
    # 构建坏了；而且一旦将来取消回落就会整体失败。
    #
    # 这里显式探测：默认 SDK 链不通就换成能链通的最旧-可用 SDK。
    # 纯环境问题，不是代码要求；Linux/正常 macOS 上探测一次即通过。
    def _sdk_ok(folder=None):
        """默认 SDK（或指定 SDK）能否链接一个最小 C++ 程序。"""
        probe = os.path.join(tempfile.mkdtemp(), "probe.cpp")
        with open(probe, "w") as fh:
            fh.write("int main(){return 0;}\n")
        out = probe + ".bin"
        cmd = [compiler]
        if folder:
            # 注意必须空格分隔：clang 不认 "-isysroot=<path>"（会把 '=' 当路径一部分）
            cmd += ["-isysroot", folder]
        cmd += [probe, "-o", out]
        try:
            return subprocess.run(cmd, capture_output=True).returncode == 0
        except OSError:
            return False

    if not _sdk_ok():
        _sdk_root = "/Library/Developer/CommandLineTools/SDKs"
        if os.path.isdir(_sdk_root):
            def _ver(name):
                nums = []
                for part in name.replace("MacOSX", "").replace(".sdk", "").split("."):
                    nums.append(int(part) if part.isdigit() else 0)
                return tuple(nums)
            for _name in sorted(os.listdir(_sdk_root), key=_ver, reverse=True):
                if not _name.endswith(".sdk"):
                    continue
                _path = os.path.join(_sdk_root, _name)
                if _sdk_ok(_path):
                    base += ["-isysroot", _path]
                    print(f"[build_nano] 默认 SDK 链接失败，改用 {_name}",
                          file=sys.stderr)
                    break

# ---------------------------------------------------------------------------
# 指令集基线（可移植性 vs 峰值性能的取舍）
#
# 默认行为按场景区分，由环境变量 KAKEYA_MARCH 控制：
#
#   未设置          → Linux/其他平台用 "-march=native"
#                     （本机开发最快；但产物**只能在同款 CPU 上跑**）
#   KAKEYA_MARCH=x86-64-v2  → 用指定的基线（CI/出包时用）
#   KAKEYA_MARCH=none       → 不加任何 -march（最保守，任何 x86-64 可跑）
#
# 为什么需要它：`-march=native` 会让产物绑定构建机的指令集。若在 CI 上
# 构建 wheel 再分发给指令集较旧的用户，"Illegal instruction" 只在用户
# 机器上才炸——构建机上一路绿灯。出包场景必须显式指定基线。
# ---------------------------------------------------------------------------
_forced = os.environ.get("KAKEYA_MARCH")

# 架构相关优化标志，按优先级排列；不支持时逐级回落。
if sys.platform == "darwin" and platform.machine() == "arm64":
    # Apple Silicon (M4)：aggressive float32 auto-vectorization
    if _forced and _forced != "native":
        # 强制基线：macOS arm64 上对应 -mcpu=<目标>
        _mcpu = [] if _forced == "none" else [f"-mcpu={_forced}"]
        arch_variants = [
            [*_mcpu, "-ffast-math", "-funroll-loops"],
            [],
        ]
    else:
        arch_variants = [
            ["-mcpu=apple-m4", "-ffast-math", "-funroll-loops", "-fvectorize", "-fslp-vectorize"],
            ["-ffast-math", "-funroll-loops"],
            [],
        ]
else:
    if _forced:
        _march = [] if _forced == "none" else [f"-march={_forced}"]
    else:
        _march = ["-march=native"]
    arch_variants = [
        [*_march, "-ffast-math", "-funroll-loops"],
        ["-ffast-math", "-funroll-loops"],
        [],
    ]
    if _forced:
        print(f"[build_nano] 指令集基线：{_forced or 'none'}（KAKEYA_MARCH 已设置）",
              file=sys.stderr)

out_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
# (源文件, 模块名) 对；tube_kernel 需要同样按当前 Python 版本重编。
kernels = [("kernel/nano_kernel.cpp", "_nano_kernel"), ("kernel/tube_kernel.cpp", "_tube_kernel")]

failed = []
for src, name in kernels:
    out = name + out_suffix
    built = False
    for flags in arch_variants:
        cmd = base + flags + ["-Wall"] + ["-I" + i for i in py_includes] + [
            src,
            "-o", out,
        ]
        print(" ".join(cmd))
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            built = True
            break
        print(f"compile failed (rc={rc}), trying fallback flags...", file=sys.stderr)
    if not built:
        failed.append(src)

if failed:
    print("failed to build: " + ", ".join(failed), file=sys.stderr)
    sys.exit(1)
PY

EXT_SUFFIX=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
echo "Built: _nano_kernel${EXT_SUFFIX} and _tube_kernel${EXT_SUFFIX}"
