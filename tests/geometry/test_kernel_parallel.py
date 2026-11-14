"""守门：C++ 内核的**多核并行**路径必须与串行**逐位一致**（2026-10-07）。

**为什么单独守这件事**：2026-10-07 给 ``voxel_union_volume`` / ``mc_union_volume``
的热点循环加了 ``std::thread`` 并行（按 iz 层 / 点区间切分，见
``cpp/kernel/parallel.hpp``）。并行化**唯一允许的副作用是变快**——结果必须与串行
**逐位相同**，否则会破坏本项目的核心不变量："换个后端/换次运行也算得一样"
（``.kpt`` 复核依赖它）。

本设计只并行"**整型计数**归约"（每个 worker 局部计数，最后按确定顺序求和）：
整数加法满足结合律，故并行与串行逐位相同。**浮点累加不走这条路**——浮点加法
不满足结合律，换个求和顺序结果就会漂。这是本文件要钉住的性质。

**怎么测**：线程数经环境变量 ``KAKEYA_NUM_THREADS`` **每进程读一次**（结果被
static 缓存），所以同一进程里改不了——必须在**子进程**里以不同线程数各跑一遍，
再逐位比对。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]

#: 在子进程里跑的一段小脚本：造固定构型，打印 voxel 与 mc（固定 seed）的结果。
#: 用 ``repr`` 输出浮点以保留**完整二进制精度**（``str`` 会丢末位）。
_CHILD = r"""
import json, sys
import numpy as np
from kakeya_ai.fast_kernel import fast_voxel_volume, fast_mc_volume, is_cpp_available
from kakeya_ai.primitives import Direction, Tube

def make_tubes(n, seed=0):
    rng = np.random.default_rng(seed)
    return [Tube(center=rng.uniform(-0.5, 0.5, 3),
                 direction=Direction.from_spherical(rng.uniform(0.1, np.pi - 0.1),
                                                    rng.uniform(0, 2 * np.pi)),
                 length=1.0, delta=0.05) for _ in range(n)]

out = {"cpp": is_cpp_available(), "results": {}}
for n in (16, 64):
    tubes = make_tubes(n)
    for g in (32, 64, 128):
        v = fast_voxel_volume(tubes, grid_n=g)
        out["results"][f"voxel_n{n}_g{g}"] = repr(float(v))
tubes = make_tubes(64)
# MC 用**固定 seed**，否则每进程抽的随机点不同，比的是"采样差异"而非"并行差异"。
out["results"]["mc_n64_seed7"] = repr(float(fast_mc_volume(tubes, n_samples=300_000, seed=7)))
print(json.dumps(out))
"""


def _run_child(threads: int) -> dict:
    """在子进程里以指定线程数跑内核，返回结果字典。"""
    env = dict(os.environ)
    env["KAKEYA_NUM_THREADS"] = str(threads)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"子进程（threads={threads}）失败：\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="module")
def thread_results() -> dict[int, dict]:
    """跑 1 / 2 / 4 / 8 线程各一遍（子进程隔离）。扩展未构建则 skip。"""
    probe = _run_child(1)
    if not probe["cpp"]:
        pytest.skip("本机未构建 C++ 扩展，无法验证并行路径")
    return {t: _run_child(t) for t in (1, 2, 4, 8)}


class TestParallelIsBitIdentical:
    """并行结果必须与串行（1 线程）**逐位相同**。"""

    def test_voxel_and_mc_match_serial_exactly(self, thread_results):
        """2/4/8 线程的每个结果都必须等于 1 线程的结果（逐位）。

        这条是本次改动**唯一允许的副作用是变快**的落点。断言用 ``repr``
        字符串比较——浮点的 ``==`` 对 ``nan`` 会静默为假，而字符串比较
        对 nan 也是"同串才等"，更严格也更直白。
        """
        serial = thread_results[1]["results"]
        for threads in (2, 4, 8):
            got = thread_results[threads]["results"]
            for key, expected in serial.items():
                assert got[key] == expected, (
                    f"{threads} 线程下 {key} 与串行不一致："
                    f"串行={expected}，并行={got[key]} —— "
                    "并行只允许变快，不允许改变结果"
                )

    def test_thread_count_actually_differs(self, thread_results):
        """确认并行**真的开了多线程**（而非全部静默走串行）。

        若某次改动让并行路径悄悄退回串行，上面的"逐位一致"会**恒真**而失去
        意义。这里用时间做**粗**判据：8 线程应明显快于 1 线程（取总量级，
        不卡精确倍数，避免机器负载噪声把测试打 flaky）。
        """
        # 该判据靠 wall-clock，可能受机器负载影响；仅在**明显**没提速时才失败。
        import time

        def _timed(threads_env: int) -> float:
            env = dict(os.environ)
            env["KAKEYA_NUM_THREADS"] = str(threads_env)
            env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
            script = (
                "import numpy as np\n"
                "from kakeya_ai.fast_kernel import fast_voxel_volume\n"
                "from kakeya_ai.primitives import Direction, Tube\n"
                "rng=np.random.default_rng(0)\n"
                "tubes=[Tube(center=rng.uniform(-0.5,0.5,3),"
                "direction=Direction.from_spherical(rng.uniform(0.1,np.pi-0.1),"
                "rng.uniform(0,2*np.pi)),length=1.0,delta=0.05) for _ in range(64)]\n"
                "import time;t0=time.perf_counter()\n"
                "for _ in range(5): fast_voxel_volume(tubes, grid_n=128)\n"
                "print((time.perf_counter()-t0)/5)\n"
            )
            proc = subprocess.run(
                [sys.executable, "-c", script], cwd=REPO, capture_output=True,
                text=True, env=env, timeout=180,
            )
            assert proc.returncode == 0, proc.stderr
            return float(proc.stdout.strip().splitlines()[-1])

        t_serial = _timed(1)
        t_parallel = _timed(8)
        assert t_parallel < t_serial * 0.5, (
            f"8 线程 {t_parallel*1000:.1f} ms 未明显快于串行 {t_serial*1000:.1f} ms —— "
            "并行路径可能已静默退回串行（那会让上面的逐位一致断言变得无意义）"
        )


class TestTopologyAwareScheduling:
    """芯片拓扑探测必须正确，且默认线程数不再"拍脑袋"。"""

    def _topology(self) -> dict:
        """通过子进程取拓扑（避免本进程的 static 缓存与其它用例互相影响）。"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
        script = (
            "from kakeya_ai.nano_kernel import _load_cpp_module\n"
            "import json\n"
            "m = _load_cpp_module()\n"
            "print(json.dumps(m.topology()))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script], cwd=REPO, capture_output=True,
            text=True, env=env, timeout=120,
        )
        if proc.returncode != 0:
            pytest.skip(f"C++ 扩展不可用：{proc.stderr[-200:]}")
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_topology_is_self_consistent(self):
        """拓扑字段必须自洽：total = p + e（异构时），且都为正。"""
        topo = self._topology()
        assert topo["p_cores"] > 0, f"性能核数应为正：{topo}"
        assert topo["total_cores"] > 0, f"总核数应为正：{topo}"
        if topo["heterogeneous"]:
            assert topo["e_cores"] > 0, f"异构芯片应探到能效核：{topo}"
            assert topo["p_cores"] + topo["e_cores"] == topo["total_cores"], (
                f"异构芯片上 P+E 应等于总数：{topo}"
            )

    def test_default_uses_all_cores_not_a_guessed_cap(self):
        """默认线程数应为**全部核心**，而非此前拍脑袋的 `min(hc, 8)`。

        实测（M4，见 `cpp/kernel/parallel.hpp` 注释的加速曲线）：P+E 一起用最快，
        E 核仍贡献 ~1.4×。故默认**不该**把上限卡在 8。这条在**本机满核**
        时才有判别力（容器/小机器上 total_cores 本就可能 ≤8）。
        """
        topo = self._topology()
        assert topo["num_threads"] == topo["total_cores"], (
            f"默认线程数 {topo['num_threads']} 应等于总核数 {topo['total_cores']} —— "
            "不该再有写死的上限（除非显式设 KAKEYA_NUM_THREADS）"
        )

    def test_env_override_wins_over_topology(self):
        """`KAKEYA_NUM_THREADS` 必须**压过**拓扑默认（给 Ollama 留核的入口）。"""
        env = dict(os.environ)
        env["KAKEYA_NUM_THREADS"] = "3"
        env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
        script = (
            "from kakeya_ai.nano_kernel import _load_cpp_module\n"
            "import json\n"
            "print(json.dumps(_load_cpp_module().topology()))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script], cwd=REPO, capture_output=True,
            text=True, env=env, timeout=120,
        )
        if proc.returncode != 0:
            pytest.skip(f"C++ 扩展不可用：{proc.stderr[-200:]}")
        topo = json.loads(proc.stdout.strip().splitlines()[-1])
        assert topo["num_threads"] == 3, (
            f"设了 KAKEYA_NUM_THREADS=3 却用了 {topo['num_threads']} 线程 —— "
            "环境变量必须压过拓扑默认"
        )
        # 但拓扑本身仍是"这台机器的物理事实"，不该被环境变量改写。
        assert topo["total_cores"] >= 3

