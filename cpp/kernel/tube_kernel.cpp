/*
 * tube_kernel.cpp — C++ 高性能几何内核
 *
 * 热点路径：
 *   1. 批量管-点包含测试 (tube_contains_points_batch)
 *   2. 体素网格并集体积估计 (voxel_union_volume)
 *   3. 蒙特卡洛并集体积估计 (mc_union_volume)
 *
 * 用 pybind11 绑定到 Python，替代 primitives.py 中的纯 Python 循环。
 *
 * 编译（作为 kakeya_ai 包的一部分）：
 *   pip install -e .
 *
 * 手动调试编译：
 *   c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes)
 *       tube_kernel.cpp -o _tube_kernel$(python3-config --extension-suffix)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

#include "geo_core.hpp"
#include "parallel.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// 核心几何：点-线段距离
//
// **不再在本文件里定义** —— 实现见 cpp/kernel/geo_core.hpp，与 C ABI
// （cpp/abi/ka_abi.cpp）共用同一份。原因：此前两处各写一遍，实测已出现细节漂移
// （2026-09-27 比对发现 `const` 限定符不一致）。共用后不可能再漂。
//
// 数值契约：该函数的运算顺序是"跨后端逐位同"不变量的一部分，
// 因此保留一个 using 别名（而非内联展开）以确保调用点写法不变。
// ---------------------------------------------------------------------------

using ka_geo::point_segment_dist2;

// ---------------------------------------------------------------------------
// 1. 批量管-点包含测试
// ---------------------------------------------------------------------------

/**
 * 对一组管和一组点，返回 (n_points,) bool 数组，
 * 表示每个点是否被至少一根管包含。
 *
 * tubes: (n_tubes, 8) — [center_x, center_y, center_z, dir_x, dir_y, dir_z, length, delta]
 * points: (n_points, 3)
 */
py::array_t<bool> tube_contains_points_batch(
    py::array_t<double> tubes_arr,
    py::array_t<double> points_arr
) {
    auto tubes = tubes_arr.unchecked<2>();   // (n_tubes, 8)
    auto points = points_arr.unchecked<2>(); // (n_points, 3)

    int n_tubes = static_cast<int>(tubes.shape(0));
    int n_points = static_cast<int>(points.shape(0));

    auto result = py::array_t<bool>(n_points);
    auto result_ptr = result.mutable_unchecked<1>();

    // 初始化为 false
    for (int j = 0; j < n_points; ++j)
        result_ptr(j) = false;

    double delta_sq;
    double t;

    for (int i = 0; i < n_tubes; ++i) {
        double cx = tubes(i, 0), cy = tubes(i, 1), cz = tubes(i, 2);
        double dx = tubes(i, 3), dy = tubes(i, 4), dz = tubes(i, 5);
        double length = tubes(i, 6);
        double delta = tubes(i, 7);
        delta_sq = delta * delta;

        // 线段端点
        double half = 0.5 * length;
        double ax = cx - half * dx, ay = cy - half * dy, az = cz - half * dz;
        double bx = cx + half * dx, by = cy + half * dy, bz = cz + half * dz;

        for (int j = 0; j < n_points; ++j) {
            if (result_ptr(j)) continue;  // 已被包含

            double dist2 = point_segment_dist2(
                points(j, 0), points(j, 1), points(j, 2),
                ax, ay, az, bx, by, bz, t
            );
            if (dist2 <= delta_sq)
                result_ptr(j) = true;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// 2. 体素网格并集体积估计
// ---------------------------------------------------------------------------

/**
 * 体素网格法估计 |∪T|。
 *
 * tubes: (n_tubes, 8)
 * bounds: (3, 2) — [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
 * grid_n: 每轴体素数
 *
 * 返回体积估计。
 */
double voxel_union_volume(
    py::array_t<double> tubes_arr,
    py::array_t<double> bounds_arr,
    int grid_n
) {
    auto tubes = tubes_arr.unchecked<2>();
    auto bounds = bounds_arr.unchecked<2>();

    int n_tubes = static_cast<int>(tubes.shape(0));

    double lo[3] = {bounds(0, 0), bounds(1, 0), bounds(2, 0)};
    double hi[3] = {bounds(0, 1), bounds(1, 1), bounds(2, 1)};
    double dims[3] = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
    double voxel_size[3] = {dims[0] / grid_n, dims[1] / grid_n, dims[2] / grid_n};
    double voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2];

    // 逐体素检查：是否被任何管包含
    // 体素中心坐标
    int64_t count_inside = 0;

    // 预计算每根管的端点和 delta^2
    struct TubeData {
        double ax, ay, az, bx, by, bz, delta_sq;
        // AABB 预筛（2026-10-07，与 nano_kernel 同构造）：盒外即必不命中，
        // 可跳过昂贵的 point_segment_dist2。保守外扩，只放宽不收紧。
        double bx_lo, by_lo, bz_lo;
        double bx_hi, by_hi, bz_hi;
    };
    std::vector<TubeData> tube_data(n_tubes);

    for (int i = 0; i < n_tubes; ++i) {
        double cx = tubes(i, 0), cy = tubes(i, 1), cz = tubes(i, 2);
        double dx = tubes(i, 3), dy = tubes(i, 4), dz = tubes(i, 5);
        double length = tubes(i, 6);
        double delta = tubes(i, 7);
        double half = 0.5 * length;
        double ax = cx - half * dx, ay = cy - half * dy, az = cz - half * dz;
        double bx = cx + half * dx, by = cy + half * dy, bz = cz + half * dz;
        double m = delta * 1.000000000000001;  // ≈ delta*(1+2^-52)
        tube_data[i] = {
            ax, ay, az, bx, by, bz, delta * delta,
            std::fmin(ax, bx) - m, std::fmin(ay, by) - m, std::fmin(az, bz) - m,
            std::fmax(ax, bx) + m, std::fmax(ay, by) + m, std::fmax(az, bz) + m,
        };
    }

    // 并行：按 iz 层切分；每个 worker 的 t 是**局部变量**（不可共享）。
    // 逐位一致：计数是 int64 加法，满足结合律。
    count_inside = ka::parallel_sum_units(
        grid_n,
        static_cast<int64_t>(grid_n) * grid_n * (n_tubes + 1),
        [&](int64_t iz_begin, int64_t iz_end) -> int64_t {
            int64_t local = 0;
            for (int64_t iz = iz_begin; iz < iz_end; ++iz) {
                double z = lo[2] + voxel_size[2] * (static_cast<double>(iz) + 0.5);
                for (int iy = 0; iy < grid_n; ++iy) {
                    double y = lo[1] + voxel_size[1] * (iy + 0.5);
                    for (int ix = 0; ix < grid_n; ++ix) {
                        double x = lo[0] + voxel_size[0] * (ix + 0.5);

                        bool inside = false;
                        double t = 0.0;
                        for (int i = 0; i < n_tubes && !inside; ++i) {
                            const auto &td = tube_data[i];
                            // AABB 预筛：盒外必不命中（保守盒含所有命中点）。
                            if (x < td.bx_lo || x > td.bx_hi ||
                                y < td.by_lo || y > td.by_hi ||
                                z < td.bz_lo || z > td.bz_hi) {
                                continue;
                            }
                            double dist2 = point_segment_dist2(
                                x, y, z,
                                td.ax, td.ay, td.az,
                                td.bx, td.by, td.bz,
                                t
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
// 3. 蒙特卡洛并集体积估计
// ---------------------------------------------------------------------------

/**
 * 蒙特卡洛法估计 |∪T|。
 *
 * tubes: (n_tubes, 8)
 * points: (n_samples, 3) — 预生成的随机采样点
 *
 * 返回体积估计。
 */
double mc_union_volume(
    py::array_t<double> tubes_arr,
    py::array_t<double> points_arr,
    double box_volume
) {
    auto tubes = tubes_arr.unchecked<2>();
    auto points = points_arr.unchecked<2>();

    int n_tubes = static_cast<int>(tubes.shape(0));
    int n_points = static_cast<int>(points.shape(0));

    // 预计算管数据
    struct TubeData {
        double ax, ay, az, bx, by, bz, delta_sq;
        // AABB 预筛（与 voxel 版同构造）。
        double bx_lo, by_lo, bz_lo;
        double bx_hi, by_hi, bz_hi;
    };
    std::vector<TubeData> tube_data(n_tubes);

    for (int i = 0; i < n_tubes; ++i) {
        double cx = tubes(i, 0), cy = tubes(i, 1), cz = tubes(i, 2);
        double dx = tubes(i, 3), dy = tubes(i, 4), dz = tubes(i, 5);
        double length = tubes(i, 6);
        double delta = tubes(i, 7);
        double half = 0.5 * length;
        double ax = cx - half * dx, ay = cy - half * dy, az = cz - half * dz;
        double bx = cx + half * dx, by = cy + half * dy, bz = cz + half * dz;
        double m = delta * 1.000000000000001;
        tube_data[i] = {
            ax, ay, az, bx, by, bz, delta * delta,
            std::fmin(ax, bx) - m, std::fmin(ay, by) - m, std::fmin(az, bz) - m,
            std::fmax(ax, bx) + m, std::fmax(ay, by) + m, std::fmax(az, bz) + m,
        };
    }

    int64_t count_inside = 0;

    // 并行：按点区间切分；t 在 worker 内局部声明，避免跨线程共享。
    count_inside = ka::parallel_sum_units(
        n_points,
        static_cast<int64_t>(n_tubes) + 1,
        [&](int64_t j_begin, int64_t j_end) -> int64_t {
            int64_t local = 0;
            double t = 0.0;
            for (int64_t j = j_begin; j < j_end; ++j) {
                double px = points(j, 0), py = points(j, 1), pz = points(j, 2);
                bool inside = false;
                for (int i = 0; i < n_tubes && !inside; ++i) {
                    const auto &td = tube_data[i];
                    if (px < td.bx_lo || px > td.bx_hi ||
                        py < td.by_lo || py > td.by_hi ||
                        pz < td.bz_lo || pz > td.bz_hi) {
                        continue;
                    }
                    double dist2 = point_segment_dist2(
                        px, py, pz,
                        td.ax, td.ay, td.az,
                        td.bx, td.by, td.bz,
                        t
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
// 4. 网格单元 tube 分配 (grains 构造热点)
// ---------------------------------------------------------------------------

/**
 * 将每根 δ-管的轴对齐包围盒分配到可能重叠的网格单元中。
 *
 * tube_bboxes: (n_tubes, 6) — [xlo, xhi, ylo, yhi, zlo, zhi]
 * lo, hi: (3,) — 网格包围盒
 * nx, ny, nz: 各轴网格数
 *
 * 返回 (cell_ids, tube_indices) 两个等长 int 数组。cell_id 编码为
 *   ix * ny * nz + iy * nz + iz
 * 在 Python 端可以据此 groupby。
 */
std::pair<py::array_t<int>, py::array_t<int>> assign_tubes_to_grid_cells(
    py::array_t<double> tube_bboxes_arr,
    py::array_t<double> lo_arr,
    py::array_t<double> hi_arr,
    int nx, int ny, int nz
) {
    auto bboxes = tube_bboxes_arr.unchecked<2>();
    auto lo = lo_arr.unchecked<1>();
    auto hi = hi_arr.unchecked<1>();

    int n_tubes = static_cast<int>(bboxes.shape(0));

    double lox = lo(0), loy = lo(1), loz = lo(2);
    double stepx = (hi(0) - lox) / std::max(nx, 1);
    double stepy = (hi(1) - loy) / std::max(ny, 1);
    double stepz = (hi(2) - loz) / std::max(nz, 1);
    double inv_stepx = 1.0 / stepx;
    double inv_stepy = 1.0 / stepy;
    double inv_stepz = 1.0 / stepz;

    std::vector<int> cell_ids;
    std::vector<int> tube_indices;
    cell_ids.reserve(static_cast<size_t>(n_tubes * 4));
    tube_indices.reserve(static_cast<size_t>(n_tubes * 4));

    for (int t = 0; t < n_tubes; ++t) {
        double xlo = bboxes(t, 0), xhi = bboxes(t, 1);
        double ylo = bboxes(t, 2), yhi = bboxes(t, 3);
        double zlo = bboxes(t, 4), zhi = bboxes(t, 5);

        int ix_min = std::max(0, static_cast<int>(std::floor((xlo - lox) * inv_stepx)));
        int ix_max = std::min(nx - 1, static_cast<int>(std::ceil((xhi - lox) * inv_stepx)));
        int iy_min = std::max(0, static_cast<int>(std::floor((ylo - loy) * inv_stepy)));
        int iy_max = std::min(ny - 1, static_cast<int>(std::ceil((yhi - loy) * inv_stepy)));
        int iz_min = std::max(0, static_cast<int>(std::floor((zlo - loz) * inv_stepz)));
        int iz_max = std::min(nz - 1, static_cast<int>(std::ceil((zhi - loz) * inv_stepz)));

        if (ix_min > ix_max || iy_min > iy_max || iz_min > iz_max)
            continue;

        for (int ix = ix_min; ix <= ix_max; ++ix) {
            double cxlo = lox + stepx * ix;
            double cxhi = cxlo + stepx;
            if (xhi < cxlo || xlo > cxhi)
                continue;
            int base_x = ix * ny * nz;
            for (int iy = iy_min; iy <= iy_max; ++iy) {
                double cylo = loy + stepy * iy;
                double cyhi = cylo + stepy;
                if (yhi < cylo || ylo > cyhi)
                    continue;
                int base_xy = base_x + iy * nz;
                for (int iz = iz_min; iz <= iz_max; ++iz) {
                    double czlo = loz + stepz * iz;
                    double czhi = czlo + stepz;
                    // bbox overlap test
                    if (zhi >= czlo && zlo <= czhi) {
                        cell_ids.push_back(base_xy + iz);
                        tube_indices.push_back(t);
                    }
                }
            }
        }
    }

    py::array_t<int> cell_ids_arr(static_cast<py::ssize_t>(cell_ids.size()));
    py::array_t<int> tube_indices_arr(static_cast<py::ssize_t>(tube_indices.size()));

    auto cids = cell_ids_arr.mutable_unchecked<1>();
    auto tids = tube_indices_arr.mutable_unchecked<1>();

    for (size_t i = 0; i < cell_ids.size(); ++i) {
        cids(static_cast<py::ssize_t>(i)) = cell_ids[i];
        tids(static_cast<py::ssize_t>(i)) = tube_indices[i];
    }

    return std::make_pair(cell_ids_arr, tube_indices_arr);
}

// ---------------------------------------------------------------------------
// 5. 管对相交测试
// ---------------------------------------------------------------------------

/**
 * 判断两根 δ-管是否相交。
 * 简化判定：两线段最短距离 ≤ δ₁ + δ₂。
 *
 * tubes_a: (n_a, 8), tubes_b: (n_b, 8)
 * 返回 (n_a, n_b) bool 矩阵。
 */
py::array_t<bool> tube_pair_intersect(
    py::array_t<double> tubes_a_arr,
    py::array_t<double> tubes_b_arr
) {
    auto tubes_a = tubes_a_arr.unchecked<2>();
    auto tubes_b = tubes_b_arr.unchecked<2>();

    int n_a = static_cast<int>(tubes_a.shape(0));
    int n_b = static_cast<int>(tubes_b.shape(0));

    auto result = py::array_t<bool>({n_a, n_b});
    auto r = result.mutable_unchecked<2>();

    for (int i = 0; i < n_a; ++i) {
        double cax = tubes_a(i, 0), cay = tubes_a(i, 1), caz = tubes_a(i, 2);
        double dax = tubes_a(i, 3), day = tubes_a(i, 4), daz = tubes_a(i, 5);
        double la = tubes_a(i, 6), delta_a = tubes_a(i, 7);
        double half_a = 0.5 * la;
        double a1x = cax - half_a * dax, a1y = cay - half_a * day, a1z = caz - half_a * daz;
        double a2x = cax + half_a * dax, a2y = cay + half_a * day, a2z = caz + half_a * daz;
        double r_sum = delta_a;  // 加上 b 的 delta

        for (int j = 0; j < n_b; ++j) {
            double cbx = tubes_b(j, 0), cby = tubes_b(j, 1), cbz = tubes_b(j, 2);
            double dbx = tubes_b(j, 3), dby = tubes_b(j, 4), dbz = tubes_b(j, 5);
            double lb = tubes_b(j, 6), delta_b = tubes_b(j, 7);
            double half_b = 0.5 * lb;
            double b1x = cbx - half_b * dbx, b1y = cby - half_b * dby, b1z = cbz - half_b * dbz;
            double b2x = cbx + half_b * dbx, b2y = cby + half_b * dby, b2z = cbz + half_b * dbz;

            double threshold = r_sum + delta_b;
            double threshold_sq = threshold * threshold;

            // 简化：用点到线段距离做近似（对 δ-管足够）
            // 检查 b 的端点到 a 线段的距离
            double t_a, t_b;
            double d1 = point_segment_dist2(b1x, b1y, b1z, a1x, a1y, a1z, a2x, a2y, a2z, t_a);
            double d2 = point_segment_dist2(b2x, b2y, b2z, a1x, a1y, a1z, a2x, a2y, a2z, t_b);
            // 也检查 a 的端点到 b 线段
            double d3 = point_segment_dist2(a1x, a1y, a1z, b1x, b1y, b1z, b2x, b2y, b2z, t_a);
            double d4 = point_segment_dist2(a2x, a2y, a2z, b1x, b1y, b1z, b2x, b2y, b2z, t_b);

            double min_dist_sq = std::min({d1, d2, d3, d4});
            r(i, j) = (min_dist_sq <= threshold_sq);
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Python 绑定
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_tube_kernel, m) {
    m.doc() = "C++ high-performance geometry kernel for delta-tube operations";

    m.def("tube_contains_points_batch", &tube_contains_points_batch,
          "Batch test: which points are inside at least one tube.\n\n"
          "tubes: (n_tubes, 8) array [cx,cy,cz, dx,dy,dz, length, delta]\n"
          "points: (n_points, 3) array\n"
          "Returns: (n_points,) bool array");

    m.def("voxel_union_volume", &voxel_union_volume,
          "Voxel-based estimate of union volume.\n\n"
          "tubes: (n_tubes, 8) array\n"
          "bounds: (3, 2) array\n"
          "grid_n: int\n"
          "Returns: double (volume estimate)");

    m.def("mc_union_volume", &mc_union_volume,
          "Monte Carlo estimate of union volume.\n\n"
          "tubes: (n_tubes, 8) array\n"
          "points: (n_samples, 3) array\n"
          "box_volume: double\n"
          "Returns: double (volume estimate)");

    m.def("tube_pair_intersect", &tube_pair_intersect,
          "Pairwise tube intersection test.\n\n"
          "tubes_a: (n_a, 8), tubes_b: (n_b, 8)\n"
          "Returns: (n_a, n_b) bool array");

    m.def("assign_tubes_to_grid_cells", &assign_tubes_to_grid_cells,
          "Assign each tube bbox to all grid cells it may overlap.\n\n"
          "tube_bboxes: (n_tubes, 6) [xlo,xhi,ylo,yhi,zlo,zhi]\n"
          "lo, hi: (3,) grid bounds\n"
          "nx, ny, nz: int grid counts\n"
          "Returns: (cell_ids, tube_indices) int arrays of equal length");
}
