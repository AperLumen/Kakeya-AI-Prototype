#!/usr/bin/env python3
"""主 proof orchestrator 骨架的演示。

Run:
    .venv/bin/python scripts/orchestration/demo_proof_orchestrator.py

这里展示 orchestrator 如何把以下环节串联起来：
  - sticky check
  - grains decomposition
  - Move #1/#2/#3 启发式选择
  - volume lower bound 验证

**输出即 ``.kpt``**：默认把这次运行写成 ``docs/traces/<时间戳>.kpt``
（``--no-trace`` 关闭，``--trace PATH`` 指定路径）。格式与解释器见
``src/kakeya_ai/trace_format.py`` 与 ``scripts/orchestration/kpt.py``——

    .venv/bin/python scripts/orchestration/kpt.py head <file>.kpt     # 快速打开
    .venv/bin/python scripts/orchestration/kpt.py verify <file>.kpt   # 独立复核
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from kakeya_ai import Direction, Tube, run_proof_skeleton, write_trace
from kakeya_ai.primitives import TubeCollection


def make_challenging_tubes(n: int = 8, seed: int = 7) -> list[Tube]:
    """构造一个既非完全 sticky 也非平凡的 tube 排布。"""
    tubes = []
    rng = np.random.default_rng(seed)
    for theta in np.linspace(0.2, np.pi - 0.2, n):
        for phi in np.linspace(0, 2 * np.pi, n, endpoint=False):
            if rng.random() < 0.25:
                d = Direction.from_spherical(theta, phi)
                offset = rng.normal(scale=0.05, size=3)
                tubes.append(
                    Tube(center=offset, direction=d, length=0.8, delta=0.04)
                )
    return tubes


def make_move_reaching_tubes(n: int = 16, delta: float = 0.05,
                            spread: float = 0.05) -> list[Tube]:
    """构造一个**能走到 Move 分支**的管族（2026-09-30 新增）。

    **为什么需要第二种构型**：``make_challenging_tubes`` 产出的管族实测判为
    **sticky**，而 ``ProofOrchestrator.step()`` 里 sticky 检查**直接终局**
    （数学上正确：sticky 情形验下界即可）——于是那种 trace 里
    **一个 Move 步都没有**，`proof_steps_to_samples()` 转换后得到 **0 个样本**。

    实测入库的 3 个 ``.kpt`` 全是这种情况（共 7 步、0 个 Move 样本），
    而 surrogate 需要的是"**真的发生了 Move 选择**"的轨迹。

    本构型（方向在极角上轻微展开 + 均匀铺方位角）实测：
      - 判为**非** sticky；
      - 会走到 ``GRAINS_DECOMPOSITION`` 与 Move 分支；
      - 产出 **2 个真实 Move 样本**（``MOVE3`` + ``MOVE1``）。

    参数取值是实测出来的（``n=16, delta=0.05, spread=0.05``），不是猜的。
    """
    return [
        Tube(
            center=np.zeros(3),
            direction=Direction.from_spherical(np.pi / 2 + spread * i, 2 * np.pi * i / n),
            length=1.0,
            delta=delta,
        )
        for i in range(n)
    ]


#: 可选的构型名 → 构造函数。加新构型时**必须同时**说明它测什么。
_ARRANGEMENTS = {
    "challenging": make_challenging_tubes,
    "move-reaching": make_move_reaching_tubes,
}


def main():
    parser = argparse.ArgumentParser(description="Demo proof orchestrator skeleton")
    parser.add_argument("--n", type=int, default=8, help="Grid size for tube generation")
    # 默认值与 docs/traces/README.md 记载的入库样本一致（`--iters 3 --grid-n 24`）。
    # 此前是 5/48，于是"照着文档跑"与"直接跑脚本"得到两份不同的样本，
    # 而入库样本只对应前者——默认值对齐后就不会再踩这个坑。
    parser.add_argument("--iters", type=int, default=3, help="Max iterations")
    parser.add_argument("--grid-n", type=int, default=24, help="Voxel grid size")
    parser.add_argument("--use-nano", action="store_true", help="Use nano_kernel for volume")
    parser.add_argument("--trace", type=str, default=None,
                        help="输出的 .kpt 路径（默认 docs/traces/<时间戳>.kpt）")
    parser.add_argument("--no-trace", action="store_true",
                        help="不输出 .kpt 文件")
    parser.add_argument(
        "--arrangement", choices=sorted(_ARRANGEMENTS), default="challenging",
        help=(
            "管族构型。'challenging'（默认）实测判为 **sticky**，trace 里"
            "**没有 Move 步**（故产不出 surrogate 训练样本）；"
            "'move-reaching' 实测会走到 Move 分支，产出 2 个真实 Move 样本——"
            "要攒 surrogate 训练数据时用它。"
        ),
    )
    args = parser.parse_args()

    tubes = _ARRANGEMENTS[args.arrangement](n=args.n)
    print(f"Generated {len(tubes)} tubes")

    t0 = time.perf_counter()
    result = run_proof_skeleton(
        tubes,
        epsilon=0.05,
        K=2.0,
        lambda_min=0.85,
        max_iterations=args.iters,
        grid_n=args.grid_n,
        use_nano=args.use_nano,
    )
    elapsed = time.perf_counter() - t0

    print(result.trace_summary())
    print(f"\nElapsed: {elapsed:.3f}s")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")

    if not args.no_trace:
        if args.trace:
            out = Path(args.trace)
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = Path("docs") / "traces" / f"{stamp}_demo.kpt"
        # 参数一起落盘：复核方要复现同样的体积估计，必须知道这些值。
        write_trace(
            result, out,
            params={
                "epsilon": 0.05,
                "K": 2.0,
                "lambda_min": 0.85,
                "grid_n": args.grid_n,
                "use_nano": bool(args.use_nano),
                "n": args.n,
                "iters": args.iters,
                "seed": 7,
            },
            extra={"producer": "scripts/orchestration/demo_proof_orchestrator.py"},
        )
        print(f"\nTrace 已写出：{out}")
        print(f"  快速打开：.venv/bin/python scripts/orchestration/kpt.py head {out}")
        print(f"  独立复核：.venv/bin/python scripts/orchestration/kpt.py verify {out}")


if __name__ == "__main__":
    main()
