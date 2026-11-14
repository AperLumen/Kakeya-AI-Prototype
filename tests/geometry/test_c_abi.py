"""C ABI（cpp/ka_abi.*）的测试：调用正确性 + 与 Python 路径的一致性。

为什么需要这组测试
------------------
C ABI 的几何算法是 **在 cpp/abi/ka_abi.cpp 里重述了一遍**（不是复用
cpp/kernel/tube_kernel.cpp），因为后者在编译期依赖 pybind11/Python 头文件，
而 C ABI 的目标恰恰是"在没有 Python 的机器上也能编译"。

代价是**同一份逻辑存在两份实现**，就有走偏的风险。本文件钉住两条：

1. **交叉一致性**：同一输入下，C ABI 与 Python 的 pybind11 路径必须给出
   **逐位相同**的结果。这是本项目"跨后端逐位同"不变量在 C ABI 上的延伸。
   任何一侧改动导致不一致，这里会失败。
2. **纯 C 可调用性**：cpp/abi/test_abi.c 用 C 编译器（不是 C++）编译并运行，
   证明头文件真的对 C 合法（漏 extern "C"、用了 C++ 关键字等都会在这里暴露）。

本机（macOS）可直接构建；无编译器时整组 skip，不让缺失工具链变成假失败。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CPP_DIR = ROOT / "cpp"


def _sdk_flags() -> list[str]:
    """macOS 上的 -isysroot 修正。

    背景（2026-09-27 实测）：本机 CommandLineTools 带的默认 SDK 是
    MacOSX27.0.sdk，其 .tbd 声明了 `arm64e.x1-*` 架构，而本机 ld 不认识，
    导致**任何** C++ 链接都失败（连 `int main(){}` 也不成）。较旧的
    MacOSX26.5.sdk 可以正常链接。

    本函数只在默认 SDK 不可用时才生效，并挑选能链接的最新的旧 SDK。
    这是**开发机的环境修正**，不是代码要求 —— Linux/CI 上返回空列表。
    """
    if sys.platform != "darwin":
        return []
    sdk_root = Path("/Library/Developer/CommandLineTools/SDKs")
    if not sdk_root.is_dir():
        return []

    probe = Path("/tmp/_ka_sdk_probe.cpp")
    probe.write_text("int main(){return 0;}\n", encoding="utf-8")

    def links(folder: Path) -> bool:
        try:
            # 必须用空格分隔（"-isysroot", path），**不能**写 "-isysroot=<path>"：
            # clang 会把 '=' 当成路径的一部分，报
            # `no such sysroot directory: '=/Library/…'`，然后回落到默认 SDK。
            r = subprocess.run(
                ["c++", "-isysroot", str(folder), str(probe),
                 "-o", "/tmp/_ka_sdk_probe"],
                capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return False
        return r.returncode == 0

    # 默认 SDK 能用就不折腾。
    try:
        r = subprocess.run(["c++", str(probe), "-o", "/tmp/_ka_sdk_probe"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            return []
    except (OSError, subprocess.SubprocessError):
        pass

    # 挑版本号最大的可用 SDK（MacOSX26.5 > MacOSX15.4）。
    def version_key(p: Path) -> tuple:
        nums = []
        for part in p.name.replace("MacOSX", "").replace(".sdk", "").split("."):
            nums.append(int(part) if part.isdigit() else 0)
        return tuple(nums)

    candidates = sorted(
        (p for p in sdk_root.glob("MacOSX*.sdk")),
        key=version_key, reverse=True)
    for c in candidates:
        if links(c):
            return ["-isysroot", str(c)]
    return []


def _build_env() -> dict | None:
    """返回可用的构建参数；不可构建时返回 None（测试整组 skip）。"""
    if shutil.which("c++") is None or shutil.which("cc") is None:
        return None
    flags = _sdk_flags()
    return {
        "CXXFLAGS": " ".join(["-O3", "-std=c++17", "-Wall", "-Wextra",
                              "-fPIC", "-fvisibility=hidden", *flags]),
        "CFLAGS": " ".join(["-O2", "-std=c11", "-Wall", "-Wextra", *flags]),
    }


@pytest.fixture(scope="module")
def built() -> Path:
    """构建 C ABI 库、CLI 与 C 测试程序。返回 CLI 路径。

    注意这个 fixture 的**取舍**（2026-09-27 变异测试暴露后调整）：

    - 工具链 / 构建系统**缺失**（没有 cc、没有 make）→ ``skip``。
      这是环境不具备，不是代码错。
    - 构建**失败**（有工具链但编译不过）→ **``fail``，不 skip**。
      早期版本把构建失败也 skip 了；结果注入一个语法/类型缺陷时，
      整组测试静默跳过（"4 skipped"），看起来"没事"——这正是第 IV 篇
      第 7 条说的"该报的没报"。有工具链却编不过就是真问题。
    - ``make check``（运行 cpp/abi/test_abi.c）失败 → ``fail``。
    """
    if shutil.which("make") is None:
        pytest.skip("本机没有 make")

    env_args = _build_env()
    if env_args is None:
        pytest.skip("本机没有可用的 C/C++ 工具链")

    def run_make(target: str):
        cmd = ["make", "-f", "Makefile", target, *[
            f"{k}={v}" for k, v in env_args.items()]]
        return subprocess.run(cmd, cwd=CPP_DIR, capture_output=True, text=True,
                              timeout=300)

    r = run_make("all")
    assert r.returncode == 0, (
        f"C ABI 构建失败（有工具链却编不过 = 真问题）：\n"
        f"stdout={r.stdout[-1200:]}\nstderr={r.stderr[-1200:]}")

    r = run_make("check")
    assert r.returncode == 0, (
        f"纯 C 调用测试失败：\nstdout={r.stdout[-1500:]}\nstderr={r.stderr[-800:]}")

    cli = CPP_DIR / "build" / "ka"
    assert cli.exists(), "CLI 未产出"
    return cli


def _write_tubes_csv(path: Path, tubes) -> None:
    """把 Tube 列表写成 C ABI 的 CSV（8 列，无表头）。"""
    lines = []
    for t in tubes:
        lines.append(",".join(repr(float(x)) for x in [
            t.center[0], t.center[1], t.center[2],
            t.direction.v[0], t.direction.v[1], t.direction.v[2],
            t.length, t.delta]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_tubes():
    from kakeya_ai.primitives import Direction, Tube
    dirs = ([0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1])
    out = []
    for d in dirs:
        v = np.array(d, dtype=np.float64)
        v /= np.linalg.norm(v)
        out.append(Tube(center=np.zeros(3), direction=Direction(v),
                        length=2.0, delta=0.1))
    return out


class TestPureCFallback:
    """cpp/abi/test_abi.c 已由 fixture 编译并运行（纯 C，不是 C++）。"""

    def test_c_compiler_accepts_header(self, built):
        """能走到这里就说明 C 编译器接受了头文件且全部断言通过。"""
        assert built.exists()


class TestCrossConsistencyWithPython:
    """C ABI 与 Python/pybind11 路径必须逐位一致。"""

    def test_voxel_bit_identical(self, built, tmp_path):
        """多个 grid_n 下体积估计必须逐位相同。

        特意选非 2 的幂（24/97）——若哪一侧偷偷换了累加顺序或精度，
        二进制结果会立刻分叉。
        """
        from kakeya_ai import fast_kernel

        tubes = _make_tubes()
        csv = tmp_path / "tubes.csv"
        _write_tubes_csv(csv, tubes)

        for grid_n in (16, 24, 48, 97):
            r = subprocess.run(
                [str(built), "volume", "--tubes", str(csv), "--grid-n", str(grid_n)],
                capture_output=True, text=True, timeout=300)
            assert r.returncode == 0, f"CLI 失败：{r.stderr}"
            c_vol = float(r.stdout.strip().splitlines()[-1])
            py_vol = fast_kernel.fast_voxel_volume(tubes, grid_n=grid_n)
            assert c_vol == py_vol, (
                f"grid_n={grid_n} 不一致：C={c_vol!r} Python={py_vol!r}")

    def test_contains_matches(self, built, tmp_path):
        """管-点包含判定必须逐点相同。"""
        from kakeya_ai import fast_kernel

        tubes = _make_tubes()
        csv = tmp_path / "tubes.csv"
        _write_tubes_csv(csv, tubes)

        pts = np.array([
            [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.05, 0.0, 0.0],
            [0.9, 0.9, 0.9], [0.0, 0.09, 0.0], [1.1, 0.0, 0.0],
        ], dtype=np.float64)
        pcsv = tmp_path / "points.csv"
        pcsv.write_text("\n".join(",".join(repr(float(x)) for x in p) for p in pts)
                        + "\n", encoding="utf-8")

        r = subprocess.run(
            [str(built), "contains", "--tubes", str(csv), "--points", str(pcsv)],
            capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, f"CLI 失败：{r.stderr}"
        rows = [ln for ln in r.stdout.strip().splitlines()[1:] if ln.strip()]
        c_mask = [int(ln.split(",")[1]) for ln in rows]

        py_mask = fast_kernel.fast_contains_points(tubes, pts).astype(int).tolist()
        assert c_mask == py_mask, f"包含判定不一致：C={c_mask} Python={py_mask}"

    def test_abi_version_reported(self, built):
        """CLI 应报出与头文件一致的 ABI 版本号。"""
        r = subprocess.run([str(built), "abi-version"],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0
        assert r.stdout.strip() == "1"


class TestMCFusedCLI:
    """`ka mc-fused`（融合版 MC，2026-10-30）：确定性 + 统计一致 + 错误路径。

    融合版在片内 splitmix64 采样（点数组不进内存），是 C ABI 与 f32 pybind
    层共享同一融合语义的落点（实现各层自包含，见 geo_core.hpp 注释）。
    """

    def test_same_seed_bit_identical(self, built, tmp_path):
        """逐样本 RNG 只依赖 (seed,j) ⇒ 同 seed 两次运行必须逐位相同。"""
        tubes = _make_tubes()
        csv = tmp_path / "tubes.csv"
        _write_tubes_csv(csv, tubes)

        def run() -> str:
            r = subprocess.run(
                [str(built), "mc-fused", "--tubes", str(csv),
                 "--samples", "200000", "--seed", "42"],
                capture_output=True, text=True, timeout=300)
            assert r.returncode == 0, f"CLI 失败：{r.stderr}"
            return r.stdout.strip().splitlines()[-1]

        assert run() == run(), "同 seed 两次运行必须逐位相同"

    def test_matches_voxel_reference(self, built, tmp_path):
        """融合 MC 与确定性体素参考（grid=97）在统计容差内一致。

        500k 样本、命中率 ~2.5% 时双采样 σ≈0.002，容差 0.02 留一个量级。
        """
        from kakeya_ai import fast_kernel

        tubes = _make_tubes()
        csv = tmp_path / "tubes.csv"
        _write_tubes_csv(csv, tubes)

        r = subprocess.run(
            [str(built), "mc-fused", "--tubes", str(csv),
             "--samples", "500000", "--seed", "7"],
            capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, f"CLI 失败：{r.stderr}"
        c_vol = float(r.stdout.strip().splitlines()[-1])

        ref = fast_kernel.fast_voxel_volume(tubes, grid_n=97)
        assert abs(c_vol - ref) < 0.02, (
            f"融合 MC {c_vol} 与体素参考 {ref} 偏差超统计容差")

    def test_bad_bounds_rejected(self, built, tmp_path):
        """hi<lo 的 bounds 必须非零退出（库返回 KA_ERR_BOUNDS）。"""
        tubes = _make_tubes()
        csv = tmp_path / "tubes.csv"
        _write_tubes_csv(csv, tubes)

        r = subprocess.run(
            [str(built), "mc-fused", "--tubes", str(csv),
             "--samples", "1000", "--bounds", "1,-1,-1,1,-1,1"],
            capture_output=True, text=True, timeout=300)
        assert r.returncode != 0, "hi<lo 的 bounds 必须报错"
