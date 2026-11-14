/*
 * geo_core.hpp — 几何内核的**单一实现**（header-only，无 Python 依赖）
 *
 * 为什么需要它
 * ------------
 * 这段几何逻辑此前在 **两处各写了一遍**：
 *
 *   cpp/kernel/tube_kernel.cpp   — 供 pybind11 用（编译期依赖 Python 头文件）
 *   cpp/abi/ka_abi.cpp        — C ABI 用（目标：能在没有 Python 的机器上编译）
 *
 * 两处实现即便"意图相同"，也会漂移。实测（2026-09-27）：重构前两者的
 * point_segment_dist2 已经在 `const` 限定符上不一致 —— 语义等价，但这正
 * 说明**靠人工同步不可靠**。本文件把几何核心抽成唯一实现，两边共同包含。
 *
 * 约束（决定了它只能是 header-only）
 * ----------------------------------
 * 1. **不依赖 pybind11 / Python 头文件** —— 否则 ka_abi.cpp 就无法在无
 *    Python 环境编译，那正是 C ABI 存在的理由。
 * 2. **不依赖第三方库** —— 只用 `<cmath>` / `<vector>`。
 * 3. `inline` / header-only，不引入新的编译单元与链接依赖。
 *
 * 数值不变量（改动时必须保持）
 * ----------------------------
 * - 所有浮点运算的顺序**不得改变**：本项目要求"跨后端逐位同"，
 *   调整累加顺序或精度会直接破坏它（见 CONTRIBUTING.md 三条不变量）。
 * - 退化情形 `ab_len2 < 1e-24` 的处理与 `t` 的截断必须保持原样。
 */

#ifndef KA_GEO_CORE_HPP
#define KA_GEO_CORE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ka_geo {

/* ---------------------------------------------------------------------------
 * 几何原语
 * ------------------------------------------------------------------------- */

/**
 * 点 p 到线段 [a, b] 的最短距离的平方。
 *
 * 同时通过 ``t`` 输出投影参数（已截断到 [0, 1]）。
 *
 * **数值契约**：本函数的运算顺序与分支结构是本项目"跨后端逐位同"的组成部分，
 * 重构时不要"顺手整理"——表达式重排会改变浮点结果。
 */
inline double point_segment_dist2(
    double px, double py, double pz,
    double ax, double ay, double az,
    double bx, double by, double bz,
    double &t) {
    double abx = bx - ax, aby = by - ay, abz = bz - az;
    double apx = px - ax, apy = py - ay, apz = pz - az;
    double ab_len2 = abx * abx + aby * aby + abz * abz;

    if (ab_len2 < 1e-24) {
        t = 0.0;
        return apx * apx + apy * apy + apz * apz;
    }

    t = (apx * abx + apy * aby + apz * abz) / ab_len2;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;

    double cx = ax + t * abx;
    double cy = ay + t * aby;
    double cz = az + t * abz;
    double dx = px - cx, dy = py - cy, dz = pz - cz;
    return dx * dx + dy * dy + dz * dz;
}

/* ---------------------------------------------------------------------------
 * 管的预计算表示
 * ------------------------------------------------------------------------- */

/**
 * 一根 δ-管的轴端点 + 半径平方。
 *
 * 预计算一次、在热点循环里复用：把 (center, direction, length, delta)
 * 转成端点可省掉每点重复的乘法。
 */
struct TubeData {
    double ax, ay, az;
    double bx, by, bz;
    double delta_sq;
};

/** 每根管在紧凑 double 数组里的步长（与 Python 侧 tubes_to_array 一致）。 */
constexpr int TUBE_STRIDE = 8;

/**
 * 从 8 个 double 的一行构造 TubeData。
 *
 * 行布局：cx,cy,cz, dx,dy,dz, length, delta —— 与 Python 侧逐字段对应。
 */
inline TubeData make_tube_data(const double *row) {
    double cx = row[0], cy = row[1], cz = row[2];
    double dx = row[3], dy = row[4], dz = row[5];
    double length = row[6];
    double delta = row[7];
    double half = 0.5 * length;
    TubeData td;
    td.ax = cx - half * dx;
    td.ay = cy - half * dy;
    td.az = cz - half * dz;
    td.bx = cx + half * dx;
    td.by = cy + half * dy;
    td.bz = cz + half * dz;
    td.delta_sq = delta * delta;
    return td;
}

/** 批量构造：n 根管 → TubeData 向量。 */
inline std::vector<TubeData> build_tube_data(const double *tubes, int32_t n) {
    std::vector<TubeData> tds;
    tds.reserve(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        tds.push_back(make_tube_data(tubes + static_cast<size_t>(i) * TUBE_STRIDE));
    }
    return tds;
}

/** 点是否落在**任意一根**管内。 */
inline bool point_in_any(const std::vector<TubeData> &tds,
                         double px, double py, double pz) {
    double t = 0.0;
    for (const auto &td : tds) {
        double dist2 = point_segment_dist2(
            px, py, pz, td.ax, td.ay, td.az, td.bx, td.by, td.bz, t);
        if (dist2 <= td.delta_sq) return true;
    }
    return false;
}

/* ---------------------------------------------------------------------------
 * 体素网格并集体积
 * ------------------------------------------------------------------------- */

/**
 * 规则网格估计 |∪T|：体素中心落在任一管内即计为"内"。
 *
 * bounds 布局：[x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]（与 Python 侧一致）。
 *
 * 注意复杂度是 O(grid_n³ × n_tubes)，调用方负责选规模。
 */
inline double voxel_union_volume(const std::vector<TubeData> &tds,
                                 const double *bounds, int32_t grid_n) {
    double lo[3] = {bounds[0], bounds[2], bounds[4]};
    double hi[3] = {bounds[1], bounds[3], bounds[5]};
    double voxel_size[3] = {
        (hi[0] - lo[0]) / grid_n,
        (hi[1] - lo[1]) / grid_n,
        (hi[2] - lo[2]) / grid_n
    };
    double voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2];

    int64_t count_inside = 0;
    for (int32_t iz = 0; iz < grid_n; ++iz) {
        double z = lo[2] + voxel_size[2] * (iz + 0.5);
        for (int32_t iy = 0; iy < grid_n; ++iy) {
            double y = lo[1] + voxel_size[1] * (iy + 0.5);
            for (int32_t ix = 0; ix < grid_n; ++ix) {
                double x = lo[0] + voxel_size[0] * (ix + 0.5);
                if (point_in_any(tds, x, y, z)) ++count_inside;
            }
        }
    }
    return static_cast<double>(count_inside) * voxel_volume;
}

/* ---------------------------------------------------------------------------
 * 蒙特卡洛"采样⊕判定"融合（f64 口径，2026-10-30）
 * ------------------------------------------------------------------------- */

/*
 * splitmix64 — 与 nano_kernel.cpp 的 splitmix64_next **同一算法**（f32 层
 * 的片内采样用它），此处按本层惯例刻意复制一份（保持 header-only、层间
 * 自包含；f32/f64 两层只共享算法不共享代码，改算法时必须两处同改——
 * 由 tests/test_c_abi.py 的统计一致性用例与 test_nano_kernel.py 的
 * 确定性用例共同兜底）。
 */
inline uint64_t splitmix64_next(uint64_t &state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* 均匀量 ∈ [0,1)，53 位精度（与 double 的有效位匹配）。 */
inline double unit_uniform01(uint64_t &state) {
    return (splitmix64_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

/**
 * 融合版蒙特卡洛并集体积：随机点在**寄存器里现算**（不进内存——点数组
 * 的生成/转换/传递往返全部消除，能耗模型量化见 nano/chip_model.py 的
 * estimate_energy_pj 与 PROJECT.md 第 III 篇 §6）。
 *
 * **逐样本独立 RNG**：第 j 个样本的随机流只依赖 (seed, j)——
 * ``state_0 = seed + (j+1)·GOLDEN``，三次 splitmix64 轮出 u1,u2,u3。
 * 无顺序依赖 ⇒ 同 seed 结果逐位确定（本层串行；f32 层并行下亦确定）。
 *
 * bounds 布局与 voxel_union_volume 相同：[x_lo,x_hi,y_lo,y_hi,z_lo,z_hi]。
 * box_volume 由 (hi-lo) 连乘内部得出。复杂度 O(n_samples × n_tubes)。
 */
inline double mc_union_volume_fused(const std::vector<TubeData> &tds,
                                    const double *bounds,
                                    int64_t n_samples,
                                    uint64_t seed) {
    const double lx = bounds[0], hx = bounds[1];
    const double ly = bounds[2], hy = bounds[3];
    const double lz = bounds[4], hz = bounds[5];
    const double box_volume = (hx - lx) * (hy - ly) * (hz - lz);

    int64_t inside = 0;
    for (int64_t j = 0; j < n_samples; ++j) {
        uint64_t state =
            seed + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(j + 1);
        const double px = lx + (hx - lx) * unit_uniform01(state);
        const double py = ly + (hy - ly) * unit_uniform01(state);
        const double pz = lz + (hz - lz) * unit_uniform01(state);
        if (point_in_any(tds, px, py, pz)) ++inside;
    }
    return static_cast<double>(inside) / static_cast<double>(n_samples) * box_volume;
}

} /* namespace ka_geo */

#endif /* KA_GEO_CORE_HPP */
