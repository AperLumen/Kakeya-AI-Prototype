/*
 * nano_kernel.cpp — 纳秒级 float32 几何内核
 *
 * 目标：在 tube 几何计算上把单点/小批量延迟压到纳秒级。
 * 策略：
 *   1. float32 数据：内存带宽减半，SIMD 宽度翻倍。
 *   2. 预计算每根管的 inv_ab_len2，把内层除法换成乘法。
 *   3. fmin/fmax 无分支 clamp，减少预测失败。
 *   4. 内层循环使用 clang loop vectorize 提示，依赖 M4 NEON/SVE2 自动向量化。
 *   5. 编译：-O3 -mcpu=apple-m4 -ffast-math -funroll-loops
 *
 * 用 pybind11 绑定，API 与 tube_kernel 保持一致但只接受 float32 arrays。
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

#include "parallel.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// float32 核心几何：点到线段距离平方
// ---------------------------------------------------------------------------

inline float point_segment_dist2_f32(
    float px, float py, float pz,
    float ax, float ay, float az,
    float abx, float aby, float abz,
    float inv_ab_len2
) {
    float apx = px - ax;
    float apy = py - ay;
    float apz = pz - az;

    float t = (apx * abx + apy * aby + apz * abz) * inv_ab_len2;
    t = std::fmin(std::fmax(t, 0.0f), 1.0f);

    float cx = ax + t * abx;
    float cy = ay + t * aby;
    float cz = az + t * abz;
    float dx = px - cx;
    float dy = py - cy;
    float dz = pz - cz;
    return dx * dx + dy * dy + dz * dz;
}

// ---------------------------------------------------------------------------
// 1. 批量管-点包含测试 (float32)
// ---------------------------------------------------------------------------

py::array_t<bool> tube_contains_points_batch_f32(
    py::array_t<float> tubes_arr,
    py::array_t<float> points_arr
) {
    auto tubes = tubes_arr.unchecked<2>();
    auto points = points_arr.unchecked<2>();

    int n_tubes = static_cast<int>(tubes.shape(0));
    int n_points = static_cast<int>(points.shape(0));

    auto result = py::array_t<bool>(n_points);
    auto result_ptr = result.mutable_unchecked<1>();

    for (int j = 0; j < n_points; ++j)
        result_ptr(j) = false;

    struct TubeData {
        float ax, ay, az;
        float abx, aby, abz;
        float inv_ab_len2;
        float delta_sq;
    };
    std::vector<TubeData> tube_data(n_tubes);

    for (int i = 0; i < n_tubes; ++i) {
        float cx = tubes(i, 0), cy = tubes(i, 1), cz = tubes(i, 2);
        float dx = tubes(i, 3), dy = tubes(i, 4), dz = tubes(i, 5);
        float length = tubes(i, 6);
        float delta = tubes(i, 7);
        float half = 0.5f * length;
        float ax = cx - half * dx, ay = cy - half * dy, az = cz - half * dz;
        float bx = cx + half * dx, by = cy + half * dy, bz = cz + half * dz;
        float abx = bx - ax, aby = by - ay, abz = bz - az;
        float ab_len2 = abx * abx + aby * aby + abz * abz;
        tube_data[i] = {ax, ay, az, abx, aby, abz, 1.0f / ab_len2, delta * delta};
    }

    for (int i = 0; i < n_tubes; ++i) {
        const auto &td = tube_data[i];
        #pragma clang loop vectorize(enable) interleave(enable)
        for (int j = 0; j < n_points; ++j) {
            if (result_ptr(j)) continue;
            float dist2 = point_segment_dist2_f32(
                points(j, 0), points(j, 1), points(j, 2),
                td.ax, td.ay, td.az,
                td.abx, td.aby, td.abz,
                td.inv_ab_len2
            );
            if (dist2 <= td.delta_sq)
                result_ptr(j) = true;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// 2. 体素网格并集体积估计 (float32)
// ---------------------------------------------------------------------------

double voxel_union_volume_f32(
    py::array_t<float> tubes_arr,
    py::array_t<float> bounds_arr,
    int grid_n
) {
    auto tubes = tubes_arr.unchecked<2>();
    auto bounds = bounds_arr.unchecked<2>();

    int n_tubes = static_cast<int>(tubes.shape(0));

    float lo[3] = {bounds(0, 0), bounds(1, 0), bounds(2, 0)};
    float hi[3] = {bounds(0, 1), bounds(1, 1), bounds(2, 1)};
    float dims[3] = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
    float voxel_size[3] = {dims[0] / grid_n, dims[1] / grid_n, dims[2] / grid_n};
    double voxel_volume = static_cast<double>(voxel_size[0]) * voxel_size[1] * voxel_size[2];

    struct TubeData {
        float ax, ay, az;
        float abx, aby, abz;
        float inv_ab_len2;
        float delta_sq;
        // AABB 预筛（2026-10-07）：管线段端点的轴对齐包围盒，**每轴各外扩 delta
        // 再放宽一点点**，保证"距离 ≤ delta 的点"一定落在盒内 ⇒ 可用于**保守**
        // 跳过（盒外 ⇒ 必然 dist2 > delta_sq ⇒ 该管对本点必不命中）。
        float bx_lo, by_lo, bz_lo;
        float bx_hi, by_hi, bz_hi;
    };
    std::vector<TubeData> tube_data(n_tubes);

    for (int i = 0; i < n_tubes; ++i) {
        float cx = tubes(i, 0), cy = tubes(i, 1), cz = tubes(i, 2);
        float dx = tubes(i, 3), dy = tubes(i, 4), dz = tubes(i, 5);
        float length = tubes(i, 6);
        float delta = tubes(i, 7);
        float half = 0.5f * length;
        float ax = cx - half * dx, ay = cy - half * dy, az = cz - half * dz;
        float bx = cx + half * dx, by = cy + half * dy, bz = cz + half * dz;
        float abx = bx - ax, aby = by - ay, abz = bz - az;
        float ab_len2 = abx * abx + aby * aby + abz * abz;

        // 保守外扩：delta 再乘 (1+2^-19) 覆盖 float 舍入（只放宽、绝不收紧——
        // 收紧会误拒真命中点，改变结果；放宽只是少省一点、不影响等价性）。
        float m = delta * 1.000002f;
        float lox = std::fmin(ax, bx) - m, hix = std::fmax(ax, bx) + m;
        float loy = std::fmin(ay, by) - m, hiy = std::fmax(ay, by) + m;
        float loz = std::fmin(az, bz) - m, hiz = std::fmax(az, bz) + m;
        tube_data[i] = {ax, ay, az, abx, aby, abz, 1.0f / ab_len2, delta * delta,
                        lox, loy, loz, hix, hiy, hiz};
    }

    int64_t count_inside = 0;

    // 并行：按 iz 层切分（每个 worker 扫自己那几层，局部计数）。
    // 逐位一致：计数是 int64 加法，满足结合律；每个 worker 只读 tube_data。
    count_inside = ka::parallel_sum_units(
        grid_n,
        static_cast<int64_t>(grid_n) * grid_n * (n_tubes + 1),
        [&](int64_t iz_begin, int64_t iz_end) -> int64_t {
            int64_t local = 0;
            for (int64_t iz = iz_begin; iz < iz_end; ++iz) {
                float z = lo[2] + voxel_size[2] * (static_cast<float>(iz) + 0.5f);
                for (int iy = 0; iy < grid_n; ++iy) {
                    float y = lo[1] + voxel_size[1] * (iy + 0.5f);
                    #pragma clang loop vectorize(enable) interleave(enable)
                    for (int ix = 0; ix < grid_n; ++ix) {
                        float x = lo[0] + voxel_size[0] * (ix + 0.5f);
                        bool inside = false;
                        for (int i = 0; i < n_tubes && !inside; ++i) {
                            const auto &td = tube_data[i];
                            // AABB 预筛：盒外即必然不命中，跳过昂贵的距离计算。
                            // 判据是**保守**的（盒含所有"距离≤delta"的点），
                            // 故跳过的点要么本就不命中、要么在盒内照常判 ⇒ 结果
                            // 与不筛**逐位相同**。
                            if (x < td.bx_lo || x > td.bx_hi ||
                                y < td.by_lo || y > td.by_hi ||
                                z < td.bz_lo || z > td.bz_hi) {
                                continue;
                            }
                            float dist2 = point_segment_dist2_f32(
                                x, y, z,
                                td.ax, td.ay, td.az,
                                td.abx, td.aby, td.abz,
                                td.inv_ab_len2
                            );
                            if (dist2 <= td.delta_sq)
                                inside = true;
                        }
                        if (inside) ++local;
                    }
                }
            }
            return local;
        });

    return static_cast<double>(count_inside) * voxel_volume;
}

// ---------------------------------------------------------------------------
// 3. 蒙特卡洛并集体积估计 (float32)
// ---------------------------------------------------------------------------

double mc_union_volume_f32(
    py::array_t<float> tubes_arr,
    py::array_t<float> points_arr,
    double box_volume
) {
    auto tubes = tubes_arr.unchecked<2>();
    auto points = points_arr.unchecked<2>();

    int n_tubes = static_cast<int>(tubes.shape(0));
    int n_points = static_cast<int>(points.shape(0));

    struct TubeData {
        float ax, ay, az;
        float abx, aby, abz;
        float inv_ab_len2;
        float delta_sq;
        // AABB 预筛（与 voxel 版同一构造，见那里的说明）。
        float bx_lo, by_lo, bz_lo;
        float bx_hi, by_hi, bz_hi;
    };
    std::vector<TubeData> tube_data(n_tubes);

    for (int i = 0; i < n_tubes; ++i) {
        float cx = tubes(i, 0), cy = tubes(i, 1), cz = tubes(i, 2);
        float dx = tubes(i, 3), dy = tubes(i, 4), dz = tubes(i, 5);
        float length = tubes(i, 6);
        float delta = tubes(i, 7);
        float half = 0.5f * length;
        float ax = cx - half * dx, ay = cy - half * dy, az = cz - half * dz;
        float bx = cx + half * dx, by = cy + half * dy, bz = cz + half * dz;
        float abx = bx - ax, aby = by - ay, abz = bz - az;
        float ab_len2 = abx * abx + aby * aby + abz * abz;
        float m = delta * 1.000002f;
        float lox = std::fmin(ax, bx) - m, hix = std::fmax(ax, bx) + m;
        float loy = std::fmin(ay, by) - m, hiy = std::fmax(ay, by) + m;
        float loz = std::fmin(az, bz) - m, hiz = std::fmax(az, bz) + m;
        tube_data[i] = {ax, ay, az, abx, aby, abz, 1.0f / ab_len2, delta * delta,
                        lox, loy, loz, hix, hiy, hiz};
    }

    int64_t count_inside = 0;

    // 并行：按点区间切分。每个 worker 只读 tube_data 与 points，局部计数。
    count_inside = ka::parallel_sum_units(
        n_points,
        static_cast<int64_t>(n_tubes) + 1,
        [&](int64_t j_begin, int64_t j_end) -> int64_t {
            int64_t local = 0;
            for (int64_t j = j_begin; j < j_end; ++j) {
                float px = points(j, 0), py = points(j, 1), pz = points(j, 2);
                bool inside = false;
                for (int i = 0; i < n_tubes && !inside; ++i) {
                    const auto &td = tube_data[i];
                    if (px < td.bx_lo || px > td.bx_hi ||
                        py < td.by_lo || py > td.by_hi ||
                        pz < td.bz_lo || pz > td.bz_hi) {
                        continue;
                    }
                    float dist2 = point_segment_dist2_f32(
                        px, py, pz,
                        td.ax, td.ay, td.az,
                        td.abx, td.aby, td.abz,
                        td.inv_ab_len2
                    );
                    if (dist2 <= td.delta_sq)
                        inside = true;
                }
                if (inside) ++local;
            }
            return local;
        });

    return static_cast<double>(count_inside) / n_points * box_volume;
}

// ---------------------------------------------------------------------------
// 蒙特卡洛"采样⊕判定"融合（2026-10-30）——算子融合的第一个落点
// ---------------------------------------------------------------------------
//
// 数据流动机（PROJECT.md 第 III 篇 §6"算子融合"方向）：旧路径
// `mc_union_volume_f32(tubes, points, box_volume)` 的点数组由 numpy 生成后
// 跨 pybind 传入——20 万样本 ≈ 每样本 72 B 的内存往返（f64 生成写 24 +
// 转 f32 读 24/写 12 + pybind 读 12），且每个点**恰好被消费一次**
// （信息零复用）。融合版把随机采样搬进内核：只传 (bounds, n_samples,
// seed)，点坐标由 splitmix64 在寄存器里现算（与体素内核"坐标由索引
// 现算、不进内存"同一原则），点数组的内存往返归零。
//
// **逐样本独立 RNG**：第 j 个样本的随机流只依赖 (seed, j)——
// ``state_0 = seed + (j+1)·GOLDEN``，随后三次 splitmix64 轮出
// u1,u2,u3。无顺序依赖 ⇒ 结果与并行切分方式、线程调度无关：同 seed
// 同计数（动态调度下也确定）。均匀量取 53 位精度 ∈ [0,1)。
namespace {

inline uint64_t splitmix64_next(uint64_t &state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline float unit_uniform(uint64_t &state) {
    return static_cast<float>(
        (splitmix64_next(state) >> 11) * (1.0 / 9007199254740992.0));
}

}  // namespace

int64_t mc_union_volume_fused_f32(
    py::array_t<float> tubes_arr,
    py::array_t<float> bounds_arr,
    int64_t n_samples,
    uint64_t seed
) {
    if (n_samples <= 0) return 0;

    auto tubes = tubes_arr.unchecked<2>();
    auto bounds = bounds_arr.unchecked<2>();
    const int n_tubes = static_cast<int>(tubes.shape(0));

    const float lx = bounds(0, 0), hx = bounds(0, 1);
    const float ly = bounds(1, 0), hy = bounds(1, 1);
    const float lz = bounds(2, 0), hz = bounds(2, 1);
    const float sx = hx - lx, sy = hy - ly, sz = hz - lz;

    struct TubeData {
        float ax, ay, az;
        float abx, aby, abz;
        float inv_ab_len2;
        float delta_sq;
        // AABB 预筛（与 mc_union_volume_f32 同一构造）。
        float bx_lo, by_lo, bz_lo;
        float bx_hi, by_hi, bz_hi;
    };
    std::vector<TubeData> tube_data(n_tubes);

    for (int i = 0; i < n_tubes; ++i) {
        float cx = tubes(i, 0), cy = tubes(i, 1), cz = tubes(i, 2);
        float dx = tubes(i, 3), dy = tubes(i, 4), dz = tubes(i, 5);
        float length = tubes(i, 6);
        float delta = tubes(i, 7);
        float half = 0.5f * length;
        float ax = cx - half * dx, ay = cy - half * dy, az = cz - half * dz;
        float bx = cx + half * dx, by = cy + half * dy, bz = cz + half * dz;
        float abx = bx - ax, aby = by - ay, abz = bz - az;
        float ab_len2 = abx * abx + aby * aby + abz * abz;
        float m = delta * 1.000002f;
        float lox = std::fmin(ax, bx) - m, hix = std::fmax(ax, bx) + m;
        float loy = std::fmin(ay, by) - m, hiy = std::fmax(ay, by) + m;
        float loz = std::fmin(az, bz) - m, hiz = std::fmax(az, bz) + m;
        tube_data[i] = {ax, ay, az, abx, aby, abz, 1.0f / ab_len2, delta * delta,
                        lox, loy, loz, hix, hiy, hiz};
    }

    const int64_t inside_count = ka::parallel_sum_units(
        n_samples, static_cast<int64_t>(n_tubes) + 1,
        [&](int64_t j_begin, int64_t j_end) -> int64_t {
            int64_t local = 0;
            for (int64_t j = j_begin; j < j_end; ++j) {
                uint64_t state =
                    seed + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(j + 1);
                const float px = lx + sx * unit_uniform(state);
                const float py = ly + sy * unit_uniform(state);
                const float pz = lz + sz * unit_uniform(state);
                bool inside = false;
                for (int i = 0; i < n_tubes && !inside; ++i) {
                    const auto &td = tube_data[i];
                    if (px < td.bx_lo || px > td.bx_hi ||
                        py < td.by_lo || py > td.by_hi ||
                        pz < td.bz_lo || pz > td.bz_hi) {
                        continue;
                    }
                    float dist2 = point_segment_dist2_f32(
                        px, py, pz,
                        td.ax, td.ay, td.az,
                        td.abx, td.aby, td.abz,
                        td.inv_ab_len2
                    );
                    if (dist2 <= td.delta_sq)
                        inside = true;
                }
                if (inside) ++local;
            }
            return local;
        });

    return inside_count;
}

// ---------------------------------------------------------------------------
// 4. 最小延迟：单个 tube-point 距离（返回 float32）
// ---------------------------------------------------------------------------

float single_tube_point_dist2_f32(
    float px, float py, float pz,
    float cx, float cy, float cz,
    float dx, float dy, float dz,
    float length
) {
    float half = 0.5f * length;
    float ax = cx - half * dx, ay = cy - half * dy, az = cz - half * dz;
    float bx = cx + half * dx, by = cy + half * dy, bz = cz + half * dz;
    float abx = bx - ax, aby = by - ay, abz = bz - az;
    float ab_len2 = abx * abx + aby * aby + abz * abz;
    return point_segment_dist2_f32(px, py, pz, ax, ay, az, abx, aby, abz, 1.0f / ab_len2);
}

// ---------------------------------------------------------------------------
// Python 绑定
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_nano_kernel, m) {
    m.doc() = "C++ float32 nanosecond-level geometry kernel for delta-tubes";

    m.def("tube_contains_points_batch_f32", &tube_contains_points_batch_f32,
          "Batch test: which points are inside at least one tube (float32).\n\n"
          "tubes: (n_tubes, 8) float32 array\n"
          "points: (n_points, 3) float32 array\n"
          "Returns: (n_points,) bool array");

    m.def("voxel_union_volume_f32", &voxel_union_volume_f32,
          "Voxel-based estimate of union volume (float32).\n\n"
          "tubes: (n_tubes, 8) float32 array\n"
          "bounds: (3, 2) float32 array\n"
          "grid_n: int\n"
          "Returns: double (volume estimate)");

    m.def("mc_union_volume_f32", &mc_union_volume_f32,
          "Monte Carlo estimate of union volume (float32).\n\n"
          "tubes: (n_tubes, 8) float32 array\n"
          "points: (n_samples, 3) float32 array\n"
          "box_volume: double\n"
          "Returns: double (volume estimate)");

    m.def("mc_union_volume_fused_f32", &mc_union_volume_fused_f32,
          "融合版 MC：片内 splitmix64 采样 + 判定，点数组不进内存"
          "（2026-10-30，算子融合）。\n\n"
          "tubes: (n_tubes, 8) float32 array\n"
          "bounds: (3, 2) float32 array\n"
          "n_samples: int64, seed: uint64\n"
          "Returns: int64 (n_inside，体积 = n_inside/n_samples * box_volume)");

    m.def("single_tube_point_dist2_f32", &single_tube_point_dist2_f32,
          "Distance squared from one point to one tube's central segment (float32).");

    // 芯片拓扑 + 实际并行线程数（2026-10-07）。供 Python 侧查询与测试：
    // 让"这台机器有几个 P/E 核、本进程实际开几个线程"成为**可观测**的事实，
    // 而不是藏在 C++ static 里。
    m.def("topology", []() {
        const ka::Topology &t = ka::topology();
        py::dict d;
        d["p_cores"] = t.p_cores;
        d["e_cores"] = t.e_cores;
        d["total_cores"] = t.total_cores;
        d["heterogeneous"] = t.heterogeneous;
        d["num_threads"] = ka::max_threads();
        return d;
    }, "CPU topology and effective parallel thread count.");
}
