# 纳秒级 float32 几何内核优化（隔离微基准）

> 本文件在 `nano/isolated/` 下（2026-10-07 从 `isolated_nano/` 迁入）。`nano/`
> 是**芯片/加速器研发资产**的归属目录，见上层 [../README.md](../README.md)。

在已有 `cpp/kernel/tube_kernel.cpp`（float64）的基础上，新增 `cpp/kernel/nano_kernel.cpp` 与 Python 包装层 `src/kakeya_ai/nano_kernel.py`，把 tube 几何核心计算压到 **~1 ns / tube-point pair**。

## 关键优化

1. **float32 数据**：内存带宽减半，NEON/SVE2 SIMD 宽度翻倍。
2. **预计算 `1 / |AB|^2`**：内层循环的除法改为乘法。
3. **无分支 clamp**：用 `fmin/fmax` 替代分段 `t` 投影。
4. **编译参数**：`-O3 -mcpu=apple-m4 -ffast-math -funroll-loops -fvectorize -fslp-vectorize`。
5. **多核并行**（`cpp/kernel/parallel.hpp`）：`voxel` 按 iz 层、`mc` 按点区间切分到
   `std::thread`；只并行**整型计数**归约，故与串行**逐位一致**。默认用全核
   （拓扑感知），`KAKEYA_NUM_THREADS` 可覆盖。
6. **AABB 预筛**（2026-10-07，**本轮收益最大**）：每个体素/点先判"是否在管的
   轴对齐包围盒外"，盒外跳过昂贵的距离计算。本机实测 f64 voxel（grid=128，
   64 管，10 线程）**46.5 → 19.6 ms（2.37×）**，且**逐位一致**（盒保守外扩）。
   见架构清单里 nano 内核的取舍登记。

## 文件

- `cpp/kernel/nano_kernel.cpp`：C++ 扩展实现
- `cpp/build_nano.sh`：构建脚本（自动处理路径中的空格）
- `src/kakeya_ai/nano_kernel.py`：Python 包装层，API 与 `fast_kernel` 对齐
- `tests/test_nano_kernel.py`：正确性与回归测试
- `nano/isolated/bench_micro.cpp`：纯 C++ 微基准，排除 Python 开销
- `nano/isolated/benchmark_nano.py`：nano 内核自测
- `nano/isolated/benchmark_compare.py`：与 `fast_kernel` 横向对比

## 构建

```bash
cd cpp
./build_nano.sh
```

## 微基准结果（Apple M4）

### 纯 C++ 微基准（**含并行扫描 + 与芯片模型对齐**，2026-10-07 扩展）

```bash
cd nano/isolated
clang++ -O3 -mcpu=apple-m4 -ffast-math -std=c++17 -pthread bench_micro.cpp -o bench_micro
./bench_micro                 # 默认 64 管 / 100000 点 / 100 轮
./bench_micro 64 100000 200   # 自定义 n_tubes n_points n_iter
KAKEYA_NUM_THREADS=1 ./bench_micro   # 只测串行
KAKEYA_BENCH_PARTITION=1 ./bench_micro  # 追加 even vs P/E 加权切分的 A/B（异构分派判据）
KAKEYA_BENCH_FUSED_MC=1 ./bench_micro    # 追加融合 MC 片内采样开销标定（配对交替协议）
```

示例输出：

```
Pure C++ micro-benchmark (no Python overhead)
  Tubes: 64, Points: 100000, Iterations: 100
  Topology: 4P + 6E = 10 cores (heterogeneous)
  Threads in use: 10 (all cores)
  Serial count: 55316 (prevents DCE)
  Serial:  6.461401 ms/call  |  1.0096 ns/pair

  Parallel sweep (bit-identical count verified):
    threads       ms/call      ns/pair  speedup count_match
    2            3.272609       0.5113    1.97x yes
    4            1.791874       0.2800    3.61x yes
    10           1.359100       0.2124    4.75x yes

  Roofline (single-core, matches chip_model.M4_REFERENCE):
    P-core peak:          35.2 GFLOP/s (float32, 4.4 GHz x 4 x 2 FMA)
    single-core ceiling:  0.7102 ns/pair
    measured/ceiling:     1.42x
    --> 单核已用 P 核峰值算力的 70%
```

**这个微基准现在做三件事**（即"纳秒内核 × 芯片调度/模型"的合流）：

1. **单核延迟**：串行 ns/pair（本机约 1.0），地面真值、无 Python 开销。
2. **多核扫描**：按线程数扫描并**逐位核对计数与串行一致**（`count_match=yes`）。
   本机 10 线程约 **4.75×**。
3. **与 `nano.model.chip_model` 对齐的 roofline**：报告单核上限与"已用峰值百分比"。
   **两处共享的常量**（`FLOPS_PER_PAIR=25`、`p_core_gflops=35.2`）由
   `tests/test_microbench_integration.py` 强制一致——漂移即红。

**物理合理性门控**：若 `measured/ceiling < 1.0`（实测**快于**理论上限，物理上
不可能），本基准**运行检查函数**——独立复算芯片峰值（`clock × lanes × 2 × (P+E×ratio)`）
与反推需要的核数——**按计算结果**打印判定并**非零退出（2）**。与 Python 侧
`chip_model.diagnose_superpeak` 同一设计（门控**依赖函数**、不是套预写文字，见
芯片模型篇 §5）。这是把"179% 峰值"那类错误**自动拦下**。

### Python 包装层（含 pybind11/Python 开销）

```bash
.venv/bin/python nano/isolated/benchmark_compare.py
```

示例输出（64 tubes）：

```
contains_points (64 tubes x 10000 points)
  f64 fast_kernel:   903.6 us/call,  1.41 ns/pair
  f32 nano_kernel:   922.0 us/call,  1.44 ns/pair

mc_volume (64 tubes x 200000 samples)
  f64 fast_kernel:    28.1 ms/call,  2.19 ns/pair
  f32 nano_kernel:    14.3 ms/call,  1.11 ns/pair  (1.97x)

voxel_volume (64 tubes x 48^3 voxels)
  f64 fast_kernel:    13.9 ms/call,  1.96 ns/pair
  f32 nano_kernel:     7.8 ms/call,  1.10 ns/pair  (1.78x)
```

## 使用

```python
from kakeya_ai.nano_kernel import nano_mc_volume, nano_voxel_volume, nano_contains_points

vol = nano_mc_volume(tubes, n_samples=200_000, seed=42)
mask = nano_contains_points(tubes, points.astype(np.float32))
```

注意：nano 内核是低精度/高吞吐版本，结果与 `fast_kernel` 在 float32 误差范围内一致，不建议用于需要严格 double 精度的场景。
